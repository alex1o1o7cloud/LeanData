import Mathlib

namespace road_trip_ratio_l1849_184904

/-- Road trip distance calculation -/
theorem road_trip_ratio : 
  ∀ (D R : ℝ),
  D > 0 →
  R > 0 →
  D / 2 = 40 →
  2 * (D + R * D + 40) = 560 - (D + R * D + 40) →
  R = 5 / 6 := by
sorry

end road_trip_ratio_l1849_184904


namespace equation_solution_l1849_184912

theorem equation_solution : ∃ x : ℝ, (3*x + 4*x = 600 - (2*x + 6*x + x)) ∧ x = 37.5 := by
  sorry

end equation_solution_l1849_184912


namespace cubic_yards_to_cubic_feet_l1849_184992

/-- Given that 1 yard equals 3 feet, prove that 5 cubic yards is equal to 135 cubic feet. -/
theorem cubic_yards_to_cubic_feet :
  (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 27 * (1 / 3 : ℝ) * (1 / 3 : ℝ) * (1 / 3 : ℝ) →
  5 * (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 135 * (1 / 3 : ℝ) * (1 / 3 : ℝ) * (1 / 3 : ℝ) :=
by sorry

end cubic_yards_to_cubic_feet_l1849_184992


namespace egg_tray_problem_l1849_184983

theorem egg_tray_problem (eggs_per_tray : ℕ) (total_eggs : ℕ) : 
  eggs_per_tray = 10 → total_eggs = 70 → total_eggs / eggs_per_tray = 7 := by
  sorry

end egg_tray_problem_l1849_184983


namespace right_triangular_prism_relation_l1849_184901

/-- 
Given a right triangular prism with mutually perpendicular lateral edges of lengths a, b, and c,
and base height h, prove that 1/h^2 = 1/a^2 + 1/b^2 + 1/c^2.
-/
theorem right_triangular_prism_relation (a b c h : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hh : h > 0) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 := by
  sorry

end right_triangular_prism_relation_l1849_184901


namespace triangle_areas_l1849_184963

theorem triangle_areas (BD DC : ℝ) (area_ABD : ℝ) :
  BD / DC = 2 / 5 →
  area_ABD = 28 →
  ∃ (area_ADC area_ABC : ℝ),
    area_ADC = 70 ∧
    area_ABC = 98 :=
by sorry

end triangle_areas_l1849_184963


namespace system_unique_solution_l1849_184950

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  4 * Real.sqrt y = x - a ∧ y^2 - x^2 + 2*y - 4*x - 3 = 0

-- Define the set of a values for which the system has a unique solution
def unique_solution_set : Set ℝ := {a | a < -5 ∨ a > -1}

-- Theorem statement
theorem system_unique_solution :
  ∀ a : ℝ, (∃! (x y : ℝ), system x y a) ↔ a ∈ unique_solution_set :=
sorry

end system_unique_solution_l1849_184950


namespace pictures_deleted_vacation_pictures_deleted_l1849_184967

theorem pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) :
  zoo_pics + museum_pics - remaining_pics =
  (zoo_pics + museum_pics) - remaining_pics :=
by sorry

theorem vacation_pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) :
  zoo_pics = 15 →
  museum_pics = 18 →
  remaining_pics = 2 →
  zoo_pics + museum_pics - remaining_pics = 31 :=
by sorry

end pictures_deleted_vacation_pictures_deleted_l1849_184967


namespace stratified_sample_theorem_l1849_184931

/-- Calculates the number of female students in a stratified sample -/
def stratified_sample_females (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℕ :=
  (female_students * sample_size) / total_students

/-- Theorem: In a class of 54 students with 18 females, a stratified sample of 9 students contains 3 females -/
theorem stratified_sample_theorem :
  stratified_sample_females 54 18 9 = 3 := by
  sorry

end stratified_sample_theorem_l1849_184931


namespace difference_zero_for_sqrt_three_l1849_184937

-- Define the custom operation
def custom_op (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem difference_zero_for_sqrt_three :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := Real.sqrt 3
  x - y = 0 :=
by
  sorry

end difference_zero_for_sqrt_three_l1849_184937


namespace complex_equation_result_l1849_184949

theorem complex_equation_result (x y : ℝ) (h : Complex.I * Real.exp (-1) + 2 = y + x * Complex.I) : x^3 + y = 1 := by
  sorry

end complex_equation_result_l1849_184949


namespace town_population_l1849_184958

/-- 
Given a town with initial population P:
- 100 new people move in
- 400 of the original population move out
- The population is halved every year for 4 years
- After 4 years, the population is 60 people

Prove that the initial population P was 1260 people.
-/
theorem town_population (P : ℕ) : (P + 100 - 400) / (2^4 : ℕ) = 60 → P = 1260 := by
  sorry

end town_population_l1849_184958


namespace boys_in_jakes_class_l1849_184919

/-- Calculates the number of boys in a class given the ratio of girls to boys and the total number of students -/
def number_of_boys (girls_ratio : ℕ) (boys_ratio : ℕ) (total_students : ℕ) : ℕ :=
  (boys_ratio * total_students) / (girls_ratio + boys_ratio)

/-- Proves that in a class with a 3:4 ratio of girls to boys and 35 total students, there are 20 boys -/
theorem boys_in_jakes_class :
  number_of_boys 3 4 35 = 20 := by
  sorry

end boys_in_jakes_class_l1849_184919


namespace chess_matches_l1849_184915

theorem chess_matches (n : ℕ) (m : ℕ) (h1 : n = 5) (h2 : m = 3) :
  (n * (n - 1) * m) / 2 = 30 := by
  sorry

end chess_matches_l1849_184915


namespace complement_of_N_in_M_l1849_184900

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem complement_of_N_in_M :
  (M \ N) = {-1, 0, 3} := by sorry

end complement_of_N_in_M_l1849_184900


namespace laundry_loads_count_l1849_184999

theorem laundry_loads_count :
  let wash_time : ℚ := 45 / 60  -- wash time in hours
  let dry_time : ℚ := 1  -- dry time in hours
  let total_time : ℚ := 14  -- total time in hours
  let load_time : ℚ := wash_time + dry_time  -- time per load in hours
  ∃ (loads : ℕ), (loads : ℚ) * load_time = total_time ∧ loads = 8
  := by sorry

end laundry_loads_count_l1849_184999


namespace solve_scarf_knitting_problem_l1849_184965

/-- Represents the time (in hours) to knit various items --/
structure KnittingTime where
  hat : ℝ
  mitten : ℝ
  sock : ℝ
  sweater : ℝ

/-- The problem of finding the time to knit a scarf --/
def scarf_knitting_problem (kt : KnittingTime) (num_children : ℕ) (total_time : ℝ) : Prop :=
  let scarf_time := (total_time - num_children * (kt.hat + 2 * kt.mitten + 2 * kt.sock + kt.sweater)) / num_children
  scarf_time = 3

/-- The theorem stating the solution to the scarf knitting problem --/
theorem solve_scarf_knitting_problem :
  ∀ (kt : KnittingTime) (num_children : ℕ),
  kt.hat = 2 ∧ kt.mitten = 1 ∧ kt.sock = 1.5 ∧ kt.sweater = 6 ∧ num_children = 3 →
  scarf_knitting_problem kt num_children 48 :=
by
  sorry


end solve_scarf_knitting_problem_l1849_184965


namespace water_addition_proof_l1849_184932

/-- Proves that adding 23 litres of water to a 45-litre mixture with initial milk to water ratio of 4:1 results in a new mixture with milk to water ratio of 1.125 -/
theorem water_addition_proof (initial_volume : ℝ) (initial_ratio : ℚ) (water_added : ℝ) (final_ratio : ℚ) : 
  initial_volume = 45 ∧ 
  initial_ratio = 4/1 ∧ 
  water_added = 23 ∧ 
  final_ratio = 1125/1000 →
  let initial_milk := (initial_ratio / (initial_ratio + 1)) * initial_volume
  let initial_water := (1 / (initial_ratio + 1)) * initial_volume
  let final_water := initial_water + water_added
  initial_milk / final_water = final_ratio :=
by sorry

end water_addition_proof_l1849_184932


namespace linear_function_composition_l1849_184970

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 1) →
  (∀ x, f x = 2 * x + 1/3) ∨ (∀ x, f x = -2 * x - 1) :=
by sorry

end linear_function_composition_l1849_184970


namespace train_length_l1849_184947

/-- Given a train traveling at 270 kmph and crossing a pole in 5 seconds, its length is 375 meters. -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) (h1 : speed_kmph = 270) (h2 : crossing_time = 5) :
  let speed_ms := speed_kmph * 1000 / 3600
  speed_ms * crossing_time = 375 := by sorry

end train_length_l1849_184947


namespace arithmetic_progression_sum_l1849_184962

theorem arithmetic_progression_sum (n : ℕ) : 
  (n ≥ 3 ∧ n ≤ 14) ↔ 
  (n : ℝ) / 2 * (2 * 25 + (n - 1) * (-3)) ≥ 66 := by
  sorry

end arithmetic_progression_sum_l1849_184962


namespace train_travel_time_l1849_184987

/-- Proves that a train traveling at 150 km/h for 1200 km takes 8 hours -/
theorem train_travel_time :
  ∀ (speed distance time : ℝ),
    speed = 150 ∧ distance = 1200 ∧ time = distance / speed →
    time = 8 := by
  sorry

end train_travel_time_l1849_184987


namespace consecutive_non_divisors_l1849_184925

theorem consecutive_non_divisors (n : ℕ) (k : ℕ) : 
  (∀ i ∈ Finset.range 250, i ≠ k ∧ i ≠ k + 1 → n % i = 0) →
  (n % k ≠ 0 ∧ n % (k + 1) ≠ 0) →
  1 ≤ k →
  k ≤ 249 →
  k = 127 := by
sorry

end consecutive_non_divisors_l1849_184925


namespace linear_equation_solution_l1849_184924

theorem linear_equation_solution (m : ℝ) : (2 * m + 2 = 0) → m = -1 := by
  sorry

end linear_equation_solution_l1849_184924


namespace divisibility_of_sum_of_powers_l1849_184959

theorem divisibility_of_sum_of_powers (k : ℕ) : 
  Odd k → (∃ (n : ℕ), n = 9 * 7 * 4 * k ∧ 2018 ∣ (1 + 2^n + 3^n + 4^n)) :=
sorry

end divisibility_of_sum_of_powers_l1849_184959


namespace matrix_A_nonsingular_l1849_184913

/-- Prove that the matrix A defined by the given conditions is nonsingular -/
theorem matrix_A_nonsingular 
  (k : ℕ) 
  (i j : Fin k → ℕ)
  (h_i : ∀ m n, m < n → i m < i n)
  (h_j : ∀ m n, m < n → j m < j n)
  (A : Matrix (Fin k) (Fin k) ℚ)
  (h_A : ∀ r s, A r s = (Nat.choose (i r + j s) (i r) : ℚ)) :
  Matrix.det A ≠ 0 := by
  sorry

end matrix_A_nonsingular_l1849_184913


namespace second_number_is_six_l1849_184905

theorem second_number_is_six (x y : ℝ) (h : 3 * y - x = 2 * y + 6) : y = 6 := by
  sorry

end second_number_is_six_l1849_184905


namespace plane_equation_proof_l1849_184921

/-- A plane in 3D space represented by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if two planes are parallel -/
def Plane.isParallelTo (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = -2 ∧ given_plane.B = 1 ∧ given_plane.C = -3 ∧ given_plane.D = 7 →
  point.x = 1 ∧ point.y = 4 ∧ point.z = -2 →
  ∃ (result_plane : Plane),
    result_plane.A = 2 ∧ 
    result_plane.B = -1 ∧ 
    result_plane.C = 3 ∧ 
    result_plane.D = 8 ∧
    point.liesOn result_plane ∧
    result_plane.isParallelTo given_plane :=
by sorry

end plane_equation_proof_l1849_184921


namespace subway_ways_l1849_184902

theorem subway_ways (total : ℕ) (bus : ℕ) (subway : ℕ) : 
  total = 7 → bus = 4 → total = bus + subway → subway = 3 := by
  sorry

end subway_ways_l1849_184902


namespace remainder_of_product_divided_by_11_l1849_184976

theorem remainder_of_product_divided_by_11 : (108 * 110) % 11 = 0 := by
  sorry

end remainder_of_product_divided_by_11_l1849_184976


namespace brooks_theorem_l1849_184953

/-- A graph represented by its vertex set and an adjacency relation -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The maximum degree of a graph -/
def maxDegree {V : Type*} (G : Graph V) : ℕ :=
  sorry

/-- The chromatic number of a graph -/
def chromaticNumber {V : Type*} (G : Graph V) : ℕ :=
  sorry

/-- Brooks' theorem: The chromatic number of a graph is at most one more than its maximum degree -/
theorem brooks_theorem {V : Type*} (G : Graph V) :
  chromaticNumber G ≤ maxDegree G + 1 :=
sorry

end brooks_theorem_l1849_184953


namespace tape_length_problem_l1849_184961

theorem tape_length_problem (original_length : ℝ) : 
  (original_length > 0) →
  (original_length * (1 - 1/5) * (1 - 3/4) = 1.5) →
  (original_length = 7.5) := by
sorry

end tape_length_problem_l1849_184961


namespace debt_payment_average_l1849_184923

theorem debt_payment_average (total_payments : ℕ) (first_payment_amount : ℕ) 
  (first_payment_count : ℕ) (payment_increase : ℕ) :
  total_payments = 52 →
  first_payment_count = 12 →
  first_payment_amount = 410 →
  payment_increase = 65 →
  (first_payment_count * first_payment_amount + 
   (total_payments - first_payment_count) * (first_payment_amount + payment_increase)) / 
   total_payments = 460 :=
by sorry

end debt_payment_average_l1849_184923


namespace y_value_theorem_l1849_184943

theorem y_value_theorem (y : ℝ) :
  (y / 5) / 3 = 5 / (y / 3) → y = 15 ∨ y = -15 := by
  sorry

end y_value_theorem_l1849_184943


namespace root_difference_square_range_l1849_184928

/-- Given a quadratic equation with two distinct real roots, 
    prove that the square of the difference of the roots has a specific range -/
theorem root_difference_square_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - 2*a*x₁ + 2*a^2 - 3*a + 2 = 0 ∧
   x₂^2 - 2*a*x₂ + 2*a^2 - 3*a + 2 = 0) →
  ∃ y : ℝ, 0 < y ∧ y ≤ 1 ∧
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ →
    x₁^2 - 2*a*x₁ + 2*a^2 - 3*a + 2 = 0 →
    x₂^2 - 2*a*x₂ + 2*a^2 - 3*a + 2 = 0 →
    (x₁ - x₂)^2 = y :=
by sorry

end root_difference_square_range_l1849_184928


namespace complementary_angles_adjustment_l1849_184942

theorem complementary_angles_adjustment (x y : ℝ) (h1 : x + y = 90) (h2 : x / y = 3 / 7) :
  let new_x := x * 1.2
  let new_y := 90 - new_x
  (y - new_y) / y * 100 = 8.57143 := by sorry

end complementary_angles_adjustment_l1849_184942


namespace sum_reciprocals_geq_nine_l1849_184954

theorem sum_reciprocals_geq_nine (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  1/x + 1/y + 1/z ≥ 9 := by
  sorry

end sum_reciprocals_geq_nine_l1849_184954


namespace hyperbola_parabola_focus_coincidence_l1849_184917

/-- The value of p for which the right focus of the hyperbola x^2 - y^2/3 = 1 
    coincides with the focus of the parabola y^2 = 2px -/
theorem hyperbola_parabola_focus_coincidence (p : ℝ) : 
  (∃ (x y : ℝ), x^2 - y^2/3 = 1 ∧ y^2 = 2*p*x ∧ 
   (x, y) = (2, 0) ∧ (x, y) = (p/2, 0)) → 
  p = 4 := by
sorry

end hyperbola_parabola_focus_coincidence_l1849_184917


namespace school_population_l1849_184944

theorem school_population (b g t : ℕ) : 
  b = 6 * g → g = 5 * t → b + g + t = 36 * t :=
by sorry

end school_population_l1849_184944


namespace complex_number_properties_l1849_184938

theorem complex_number_properties (i : ℂ) (h : i^2 = -1) :
  let z₁ : ℂ := 2 / (-1 + i)
  z₁^4 = -4 ∧ Complex.abs z₁ = Real.sqrt 2 := by
sorry

end complex_number_properties_l1849_184938


namespace correct_equation_transformation_l1849_184982

theorem correct_equation_transformation (x : ℝ) : x - 1 = 4 → x = 5 := by
  sorry

end correct_equation_transformation_l1849_184982


namespace arithmetic_sequence_sum_l1849_184997

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

/-- The sum of the first 3k terms of an arithmetic sequence with first term k^2 + k and common difference 1 -/
theorem arithmetic_sequence_sum (k : ℕ) :
  arithmetic_sum (k^2 + k) 1 (3 * k) = 3 * k^3 + (15 / 2) * k^2 - (3 / 2) * k :=
by sorry

end arithmetic_sequence_sum_l1849_184997


namespace second_person_work_days_l1849_184951

/-- Represents the number of days two people take to complete a task together -/
def two_people_time : ℝ := 10

/-- Represents the number of days one person takes to complete the task alone -/
def one_person_time : ℝ := 70

/-- Represents the number of days the first person took to complete the remaining work after the second person left -/
def remaining_work_time : ℝ := 42

/-- Represents the number of days the second person worked before leaving -/
def second_person_work_time : ℝ := 4

/-- Theorem stating that given the conditions, the second person worked for 4 days before leaving -/
theorem second_person_work_days :
  two_people_time = 10 ∧
  one_person_time = 70 ∧
  remaining_work_time = 42 →
  second_person_work_time = 4 :=
by sorry

end second_person_work_days_l1849_184951


namespace y_intercept_of_line_l1849_184981

/-- Given a line with equation 4x + 6y - 2z = 24 and z = 3, prove that the y-intercept is (0, 5) -/
theorem y_intercept_of_line (x y z : ℝ) :
  4 * x + 6 * y - 2 * z = 24 →
  z = 3 →
  x = 0 →
  y = 5 := by
sorry

end y_intercept_of_line_l1849_184981


namespace max_d_value_l1849_184914

def a (n : ℕ) : ℕ := 150 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (k : ℕ), d k = 601 ∧ ∀ (n : ℕ), d n ≤ 601 :=
sorry

end max_d_value_l1849_184914


namespace repeating_decimal_eq_fraction_l1849_184986

/-- The repeating decimal 0.4̅5̅6̅ as a rational number -/
def repeating_decimal : ℚ := 0.4 + (56 : ℚ) / 990

/-- The fraction 226/495 -/
def fraction : ℚ := 226 / 495

/-- Theorem stating that the repeating decimal 0.4̅5̅6̅ is equal to the fraction 226/495 -/
theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l1849_184986


namespace parabola_vertex_sum_max_l1849_184989

theorem parabola_vertex_sum_max (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let M := let x_v := 3 * T / 2
            let y_v := -3 * a * T^2 / 4
            x_v + y_v
  (passes_through 0 0) ∧ 
  (passes_through (3 * T) 0) ∧
  (passes_through (3 * T + 1) 35) →
  ∀ m : ℝ, M ≤ m → m ≤ 3 := by sorry

end parabola_vertex_sum_max_l1849_184989


namespace largest_solution_reciprocal_sixth_power_l1849_184978

/-- Given that x is the largest solution to the equation log_{2x^3} 2 + log_{4x^4} 2 = -1,
    prove that 1/x^6 = 4 -/
theorem largest_solution_reciprocal_sixth_power (x : ℝ) 
  (h : x > 0)
  (eq : Real.log 2 / Real.log (2 * x^3) + Real.log 2 / Real.log (4 * x^4) = -1)
  (largest : ∀ y > 0, Real.log 2 / Real.log (2 * y^3) + Real.log 2 / Real.log (4 * y^4) = -1 → y ≤ x) :
  1 / x^6 = 4 := by
sorry

end largest_solution_reciprocal_sixth_power_l1849_184978


namespace inequality_proof_l1849_184996

theorem inequality_proof (x y z : ℝ) (h : x + y + z = 1) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 3) ≤ 3 * Real.sqrt 3 := by
  sorry

end inequality_proof_l1849_184996


namespace brian_running_time_l1849_184908

theorem brian_running_time (todd_time brian_time : ℕ) : 
  todd_time = 88 → 
  brian_time = todd_time + 8 → 
  brian_time = 96 := by
sorry

end brian_running_time_l1849_184908


namespace total_items_sold_l1849_184972

/-- The total revenue from all items sold -/
def total_revenue : ℝ := 2550

/-- The average price of a pair of ping pong rackets -/
def ping_pong_price : ℝ := 9.8

/-- The average price of a tennis racquet -/
def tennis_price : ℝ := 35

/-- The average price of a badminton racket -/
def badminton_price : ℝ := 15

/-- The number of each type of equipment sold -/
def items_per_type : ℕ := 42

theorem total_items_sold :
  3 * items_per_type = 126 ∧
  (ping_pong_price + tennis_price + badminton_price) * items_per_type = total_revenue :=
by sorry

end total_items_sold_l1849_184972


namespace fraction_sum_difference_l1849_184909

theorem fraction_sum_difference : (7 : ℚ) / 12 + 8 / 15 - 2 / 5 = 43 / 60 := by
  sorry

end fraction_sum_difference_l1849_184909


namespace emily_number_is_3000_l1849_184907

def is_valid_number (n : ℕ) : Prop :=
  n % 250 = 0 ∧ n % 60 = 0 ∧ 1000 < n ∧ n < 4000

theorem emily_number_is_3000 : ∃! n : ℕ, is_valid_number n :=
  sorry

end emily_number_is_3000_l1849_184907


namespace urn_problem_l1849_184985

theorem urn_problem (w : ℕ) : 
  (10 : ℝ) / (10 + w) * 9 / (9 + w) = 0.4285714285714286 → w = 5 := by
  sorry

end urn_problem_l1849_184985


namespace mask_production_in_july_l1849_184977

def initial_production : ℕ := 3000
def months_passed : ℕ := 4

theorem mask_production_in_july :
  initial_production * (2 ^ months_passed) = 48000 :=
by
  sorry

end mask_production_in_july_l1849_184977


namespace remainder_444_power_444_mod_13_l1849_184929

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l1849_184929


namespace triangle_number_puzzle_l1849_184975

theorem triangle_number_puzzle :
  ∀ (A B C D E F : ℕ),
    A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F →
    D + E + B = 14 →
    A + C + F = 6 →
    A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 := by
  sorry

end triangle_number_puzzle_l1849_184975


namespace binary_to_decimal_1100101_l1849_184971

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1100101₂ -/
def binary_number : List Bool := [true, false, true, false, false, true, true]

/-- Theorem: The decimal equivalent of 1100101₂ is 101 -/
theorem binary_to_decimal_1100101 :
  binary_to_decimal binary_number = 101 := by
  sorry

end binary_to_decimal_1100101_l1849_184971


namespace repeating_decimal_seven_three_five_equals_fraction_l1849_184952

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  sorry

theorem repeating_decimal_seven_three_five_equals_fraction : 
  repeatingDecimalToRational ⟨7, 35⟩ = 728 / 99 := by
  sorry

end repeating_decimal_seven_three_five_equals_fraction_l1849_184952


namespace circle_intersection_perpendicular_l1849_184973

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the intersect relation between circles
variable (intersect : Circle → Circle → Prop)

-- Define the on_circle relation between points and circles
variable (on_circle : Point → Circle → Prop)

-- Define the distance function between points
variable (dist : Point → Point → ℝ)

-- Define the intersect_line_circle relation
variable (intersect_line_circle : Point → Point → Circle → Point → Prop)

-- Define the center_of_arc relation
variable (center_of_arc : Point → Point → Circle → Point → Prop)

-- Define the intersection_of_lines relation
variable (intersection_of_lines : Point → Point → Point → Point → Point → Prop)

-- Define the perpendicular relation
variable (perpendicular : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_perpendicular 
  (C₁ C₂ : Circle) 
  (A B P Q M N C D E : Point) :
  intersect C₁ C₂ →
  on_circle P C₁ →
  on_circle Q C₂ →
  dist A P = dist A Q →
  intersect_line_circle P Q C₁ M →
  intersect_line_circle P Q C₂ N →
  center_of_arc B P C₁ C →
  center_of_arc B Q C₂ D →
  intersection_of_lines C M D N E →
  perpendicular A E C D :=
by sorry

end circle_intersection_perpendicular_l1849_184973


namespace stone_placement_possible_l1849_184998

/-- Represents the state of the stone placement game -/
structure GameState where
  cellStones : Nat → Bool
  bagStones : Nat

/-- Defines the allowed moves in the game -/
inductive Move
  | PlaceInFirst : Move
  | RemoveFromFirst : Move
  | PlaceInNext : Nat → Move
  | RemoveFromNext : Nat → Move

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.PlaceInFirst => sorry
  | Move.RemoveFromFirst => sorry
  | Move.PlaceInNext n => sorry
  | Move.RemoveFromNext n => sorry

/-- Checks if a cell contains a stone -/
def hasStone (state : GameState) (cell : Nat) : Bool :=
  state.cellStones cell

/-- The main theorem stating that with 10 stones, 
    we can place a stone in any cell from 1 to 1023 -/
theorem stone_placement_possible :
  ∀ n : Nat, n ≤ 1023 → 
  ∃ (moves : List Move), 
    let finalState := (moves.foldl applyMove 
      { cellStones := fun _ => false, bagStones := 10 })
    hasStone finalState n := by sorry

end stone_placement_possible_l1849_184998


namespace tons_to_pounds_l1849_184922

-- Define the basic units
def ounces_per_pound : ℕ := 16

-- Define the packet weight in ounces
def packet_weight_ounces : ℕ := 16 * ounces_per_pound + 4

-- Define the number of packets
def num_packets : ℕ := 1840

-- Define the capacity of the gunny bag in tons
def bag_capacity_tons : ℕ := 13

-- Define the weight of all packets in ounces
def total_weight_ounces : ℕ := num_packets * packet_weight_ounces

-- Define the relation between tons and pounds
def pounds_per_ton : ℕ := 2000

-- Theorem statement
theorem tons_to_pounds : 
  total_weight_ounces = bag_capacity_tons * pounds_per_ton * ounces_per_pound :=
sorry

end tons_to_pounds_l1849_184922


namespace greatest_divisor_of_p_plus_one_l1849_184980

theorem greatest_divisor_of_p_plus_one (n : ℕ+) : 
  ∃ (d : ℕ), d = 6 ∧ 
  (∀ (p : ℕ), Prime p → p % 3 = 2 → ¬(p ∣ n) → d ∣ (p + 1)) ∧
  (∀ (k : ℕ), k > d → ∃ (p : ℕ), Prime p ∧ p % 3 = 2 ∧ ¬(p ∣ n) ∧ ¬(k ∣ (p + 1))) :=
by sorry

end greatest_divisor_of_p_plus_one_l1849_184980


namespace triangle_exists_l1849_184940

/-- Theorem: A triangle exists given an angle, sum of two sides, and a median -/
theorem triangle_exists (α : Real) (sum_sides : Real) (median : Real) :
  ∃ (a b c : Real),
    0 < a ∧ 0 < b ∧ 0 < c ∧
    0 < α ∧ α < π ∧
    a + b = sum_sides ∧
    ((a + b) / 2)^2 + (c / 2)^2 = median^2 + ((a - b) / 2)^2 ∧
    c^2 = a^2 + b^2 - 2 * a * b * Real.cos α :=
by sorry


end triangle_exists_l1849_184940


namespace optimal_price_maximizes_profit_max_profit_at_optimal_price_l1849_184910

/-- Profit function given price x -/
def profit (x : ℝ) : ℝ :=
  let P := -750 * x + 15000
  x * P - 4 * P - 7000

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 12

/-- The maximum profit achieved at the optimal price -/
def max_profit : ℝ := 41000

/-- Theorem stating that the optimal price maximizes profit -/
theorem optimal_price_maximizes_profit :
  profit optimal_price = max_profit ∧
  ∀ x : ℝ, profit x ≤ max_profit :=
sorry

/-- Theorem stating that the maximum profit is achieved at the optimal price -/
theorem max_profit_at_optimal_price :
  ∀ x : ℝ, x ≠ optimal_price → profit x < max_profit :=
sorry

end optimal_price_maximizes_profit_max_profit_at_optimal_price_l1849_184910


namespace olympic_volunteer_allocation_l1849_184948

/-- The number of ways to allocate n distinct objects into k distinct groups,
    where each group must contain at least one object. -/
def allocationSchemes (n k : ℕ) : ℕ :=
  if n < k then 0
  else (n - 1).choose (k - 1) * k.factorial

/-- The number of allocation schemes for 5 volunteers to 4 projects -/
theorem olympic_volunteer_allocation :
  allocationSchemes 5 4 = 240 :=
by sorry

end olympic_volunteer_allocation_l1849_184948


namespace point_on_line_l1849_184974

/-- Given a line equation and two points on the line, prove the value of some_value -/
theorem point_on_line (m n some_value : ℝ) : 
  (m = n / 6 - 2 / 5) →  -- First point (m, n) satisfies the line equation
  (m + 3 = (n + some_value) / 6 - 2 / 5) →  -- Second point (m + 3, n + some_value) satisfies the line equation
  some_value = -12 / 5 := by
sorry

end point_on_line_l1849_184974


namespace time_to_install_one_window_l1849_184935

/-- Proves that the time to install one window is 5 hours -/
theorem time_to_install_one_window
  (total_windows : ℕ)
  (installed_windows : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_windows = 10)
  (h2 : installed_windows = 6)
  (h3 : time_for_remaining = 20)
  : (time_for_remaining : ℚ) / (total_windows - installed_windows : ℚ) = 5 := by
  sorry


end time_to_install_one_window_l1849_184935


namespace quadratic_equations_solutions_l1849_184918

theorem quadratic_equations_solutions :
  (∀ x, 4 * x^2 = 12 * x ↔ x = 0 ∨ x = 3) ∧
  (∀ x, 3/4 * x^2 - 2*x - 1/2 = 0 ↔ x = (4 + Real.sqrt 22) / 3 ∨ x = (4 - Real.sqrt 22) / 3) :=
by sorry

end quadratic_equations_solutions_l1849_184918


namespace triangle_problem_l1849_184960

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- Define the problem statement
theorem triangle_problem (t : Triangle) 
  (h1 : Real.tan t.A = Real.sin t.B)  -- tan A = sin B
  (h2 : ∃ (D : ℝ), 2 * D = t.a ∧ t.b = t.c)  -- BD = DC (implying 2D = a and b = c)
  (h3 : t.c = t.b) :  -- AD = AB (implying c = b)
  (2 * t.a * t.c = t.b^2 + t.c^2 - t.a^2) ∧ 
  (Real.sin t.A / Real.sin t.C = 2 * Real.sqrt 2 - 2) :=
by sorry

end triangle_problem_l1849_184960


namespace max_dimes_grace_l1849_184990

/-- The value of a dime in cents -/
def dime_value : ℚ := 10

/-- The value of a penny in cents -/
def penny_value : ℚ := 1

/-- The total amount Grace has in cents -/
def total_amount : ℚ := 480

theorem max_dimes_grace : 
  ∀ d : ℕ, d * (dime_value + penny_value) ≤ total_amount → d ≤ 43 :=
by sorry

end max_dimes_grace_l1849_184990


namespace train_bridge_crossing_time_l1849_184955

/-- Time for a train to cross a bridge with another train coming from the opposite direction -/
theorem train_bridge_crossing_time
  (train1_length : ℝ)
  (train1_speed : ℝ)
  (bridge_length : ℝ)
  (train2_length : ℝ)
  (train2_speed : ℝ)
  (h1 : train1_length = 110)
  (h2 : train1_speed = 60)
  (h3 : bridge_length = 170)
  (h4 : train2_length = 90)
  (h5 : train2_speed = 45)
  : ∃ (time : ℝ), abs (time - 280 / (60 * 1000 / 3600 + 45 * 1000 / 3600)) < 0.1 :=
by
  sorry


end train_bridge_crossing_time_l1849_184955


namespace sum_of_three_numbers_l1849_184927

theorem sum_of_three_numbers : 4.75 + 0.303 + 0.432 = 5.485 := by
  sorry

end sum_of_three_numbers_l1849_184927


namespace first_discount_percentage_l1849_184991

theorem first_discount_percentage (initial_price final_price : ℝ) 
  (second_discount : ℝ) (h1 : initial_price = 560) 
  (h2 : final_price = 313.6) (h3 : second_discount = 0.3) : 
  ∃ (first_discount : ℝ), 
    first_discount = 0.2 ∧ 
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end first_discount_percentage_l1849_184991


namespace amy_flash_drive_files_l1849_184906

theorem amy_flash_drive_files (initial_music : ℕ) (initial_video : ℕ) (deleted : ℕ) (downloaded : ℕ)
  (h1 : initial_music = 26)
  (h2 : initial_video = 36)
  (h3 : deleted = 48)
  (h4 : downloaded = 15) :
  initial_music + initial_video - deleted + downloaded = 29 := by
  sorry

end amy_flash_drive_files_l1849_184906


namespace max_disjoint_paths_iff_equal_outgoing_roads_l1849_184966

/-- Represents a city in the network -/
structure City where
  id : Nat

/-- Represents the road network -/
structure RoadNetwork where
  n : Nat
  cities : Finset City
  roads : City → City → Prop

/-- The maximum number of disjoint paths between two cities -/
def maxDisjointPaths (net : RoadNetwork) (start finish : City) : Nat :=
  sorry

/-- The number of outgoing roads from a city -/
def outgoingRoads (net : RoadNetwork) (city : City) : Nat :=
  sorry

theorem max_disjoint_paths_iff_equal_outgoing_roads
  (net : RoadNetwork) (A V : City) :
  maxDisjointPaths net A V = maxDisjointPaths net V A ↔
  outgoingRoads net A = outgoingRoads net V :=
by sorry

end max_disjoint_paths_iff_equal_outgoing_roads_l1849_184966


namespace age_difference_l1849_184964

/-- Given the ages of Frank, Ty, Carla, and Karen, prove that Ty's current age is 4 years more than twice Carla's age. -/
theorem age_difference (frank_future ty_now carla_now karen_now : ℕ) : 
  karen_now = 2 →
  carla_now = karen_now + 2 →
  frank_future = 36 →
  frank_future = ty_now * 3 + 5 →
  ty_now > 2 * carla_now →
  ty_now - 2 * carla_now = 4 := by
  sorry

end age_difference_l1849_184964


namespace multiples_of_nine_count_l1849_184911

theorem multiples_of_nine_count (N : ℕ) : 
  (∃ (count : ℕ), count = (Nat.div N 9 - Nat.div 10 9 + 1) ∧ count = 1110) → N = 9989 := by
  sorry

end multiples_of_nine_count_l1849_184911


namespace roots_greater_than_two_range_l1849_184969

theorem roots_greater_than_two_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m-4)*x + (6-m) = 0 → x > 2) →
  -2 < m ∧ m ≤ 2 - 2*Real.sqrt 3 :=
by sorry

end roots_greater_than_two_range_l1849_184969


namespace line_equation_from_circle_and_symmetry_l1849_184933

/-- The equation of a line given a circle and a point of symmetry -/
theorem line_equation_from_circle_and_symmetry (x y : ℝ) :
  let circle := {(x, y) | x^2 + (y - 4)^2 = 4}
  let center := (0, 4)
  let P := (2, 0)
  ∃ l : Set (ℝ × ℝ), 
    (∀ (p : ℝ × ℝ), p ∈ l ↔ x - 2*y + 3 = 0) ∧
    (∀ (q : ℝ × ℝ), q ∈ circle → ∃ (r : ℝ × ℝ), r ∈ l ∧ 
      center.1 + r.1 = q.1 + P.1 ∧ 
      center.2 + r.2 = q.2 + P.2) := by
  sorry

end line_equation_from_circle_and_symmetry_l1849_184933


namespace bicycle_rental_theorem_l1849_184916

/-- Represents the rental time for a bicycle. -/
inductive RentalTime
  | LessThanTwo
  | TwoToThree
  | ThreeToFour

/-- Calculates the rental fee based on the rental time. -/
def rentalFee (time : RentalTime) : ℕ :=
  match time with
  | RentalTime.LessThanTwo => 0
  | RentalTime.TwoToThree => 2
  | RentalTime.ThreeToFour => 4

/-- Represents the probabilities for each rental time for a person. -/
structure RentalProbabilities where
  lessThanTwo : ℚ
  twoToThree : ℚ
  threeToFour : ℚ

/-- The rental probabilities for person A. -/
def probA : RentalProbabilities :=
  { lessThanTwo := 1/4, twoToThree := 1/2, threeToFour := 1/4 }

/-- The rental probabilities for person B. -/
def probB : RentalProbabilities :=
  { lessThanTwo := 1/2, twoToThree := 1/4, threeToFour := 1/4 }

/-- Calculates the probability that two people pay the same fee. -/
def probSameFee (pA pB : RentalProbabilities) : ℚ :=
  pA.lessThanTwo * pB.lessThanTwo +
  pA.twoToThree * pB.twoToThree +
  pA.threeToFour * pB.threeToFour

/-- Calculates the expected value of the sum of fees for two people. -/
def expectedSumFees (pA pB : RentalProbabilities) : ℚ :=
  0 * (pA.lessThanTwo * pB.lessThanTwo) +
  2 * (pA.lessThanTwo * pB.twoToThree + pA.twoToThree * pB.lessThanTwo) +
  4 * (pA.lessThanTwo * pB.threeToFour + pA.twoToThree * pB.twoToThree + pA.threeToFour * pB.lessThanTwo) +
  6 * (pA.twoToThree * pB.threeToFour + pA.threeToFour * pB.twoToThree) +
  8 * (pA.threeToFour * pB.threeToFour)

theorem bicycle_rental_theorem :
  probSameFee probA probB = 5/16 ∧
  expectedSumFees probA probB = 7/2 := by
  sorry

end bicycle_rental_theorem_l1849_184916


namespace smallest_visible_sum_l1849_184993

/-- Represents a die in the cube -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents the 4x4x4 cube made of dice -/
def Cube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube -/
def visible_sum (c : Cube) : ℕ :=
  sorry

theorem smallest_visible_sum (c : Cube) : 
  visible_sum c ≥ 144 :=
sorry

end smallest_visible_sum_l1849_184993


namespace common_tangent_implies_a_equals_one_l1849_184988

/-- Given two curves y = (1/2e)x^2 and y = a ln x with a common tangent at their common point P(s, t), prove that a = 1 -/
theorem common_tangent_implies_a_equals_one (s t a : ℝ) : 
  t = (1 / (2 * Real.exp 1)) * s^2 →  -- Point P(s, t) lies on the first curve
  t = a * Real.log s →                -- Point P(s, t) lies on the second curve
  (s / Real.exp 1 = a / s) →          -- Common tangent condition
  a = 1 := by
sorry


end common_tangent_implies_a_equals_one_l1849_184988


namespace small_jars_count_l1849_184984

/-- Proves that the number of small jars is 62 given the conditions of the problem -/
theorem small_jars_count :
  ∀ (small_jars large_jars : ℕ),
    small_jars + large_jars = 100 →
    3 * small_jars + 5 * large_jars = 376 →
    small_jars = 62 := by
  sorry

end small_jars_count_l1849_184984


namespace initial_blue_marbles_l1849_184945

theorem initial_blue_marbles (blue red : ℕ) : 
  (blue : ℚ) / red = 5 / 3 →
  ((blue - 10 : ℚ) / (red + 25) = 1 / 4) →
  blue = 19 := by
sorry

end initial_blue_marbles_l1849_184945


namespace quadratic_common_roots_l1849_184957

theorem quadratic_common_roots : 
  ∀ (p : ℚ) (x : ℚ),
  (9 * x^2 - 3 * (p + 6) * x + 6 * p + 5 = 0 ∧
   6 * x^2 - 3 * (p + 4) * x + 6 * p + 14 = 0) ↔
  ((p = -32/9 ∧ x = -1) ∨ (p = 32/3 ∧ x = 3)) := by sorry

end quadratic_common_roots_l1849_184957


namespace distance_between_foci_l1849_184903

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) + Real.sqrt ((x + 6)^2 + (y - 7)^2) = 24

/-- The first focus of the ellipse -/
def F₁ : ℝ × ℝ := (2, -3)

/-- The second focus of the ellipse -/
def F₂ : ℝ × ℝ := (-6, 7)

/-- The theorem stating the distance between the foci -/
theorem distance_between_foci :
  Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) = 2 * Real.sqrt 41 := by
  sorry

end distance_between_foci_l1849_184903


namespace football_team_throwers_l1849_184930

/-- Represents the number of throwers on a football team -/
def num_throwers (total_players right_handed_players : ℕ) : ℕ :=
  total_players - (3 * right_handed_players - 2 * total_players)

theorem football_team_throwers :
  let total_players : ℕ := 70
  let right_handed_players : ℕ := 57
  num_throwers total_players right_handed_players = 31 :=
by sorry

end football_team_throwers_l1849_184930


namespace floor_ceil_inequality_l1849_184979

theorem floor_ceil_inequality (a b c : ℝ) 
  (h : ⌈a⌉ + ⌈b⌉ + ⌈c⌉ + ⌊a + b⌋ + ⌊b + c⌋ + ⌊c + a⌋ = 2020) :
  ⌊a⌋ + ⌊b⌋ + ⌊c⌋ + ⌈a + b + c⌉ ≥ 1346 := by
  sorry

end floor_ceil_inequality_l1849_184979


namespace k_value_at_4_l1849_184968

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - x + 1

-- Define k as a function of h's roots
def k (α β γ : ℝ) (x : ℝ) : ℝ := -(x - α^3) * (x - β^3) * (x - γ^3)

theorem k_value_at_4 (α β γ : ℝ) :
  h α = 0 → h β = 0 → h γ = 0 →  -- α, β, γ are roots of h
  k α β γ 0 = 1 →                -- k(0) = 1
  k α β γ 4 = -61 :=             -- k(4) = -61
by sorry

end k_value_at_4_l1849_184968


namespace sqrt_three_squared_l1849_184939

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end sqrt_three_squared_l1849_184939


namespace gcd_values_count_l1849_184995

theorem gcd_values_count (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (s : Finset ℕ), s.card = 6 ∧ ∀ (x : ℕ), x ∈ s ↔ ∃ (c d : ℕ+), Nat.gcd c d = x ∧ Nat.gcd c d * Nat.lcm c d = 360) :=
by sorry

end gcd_values_count_l1849_184995


namespace factorization_equality_l1849_184926

theorem factorization_equality (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) := by
  sorry

end factorization_equality_l1849_184926


namespace cat_puppy_weight_difference_l1849_184934

theorem cat_puppy_weight_difference : 
  let num_puppies : ℕ := 4
  let puppy_weight : ℝ := 7.5
  let num_cats : ℕ := 14
  let cat_weight : ℝ := 2.5
  let total_puppy_weight := (num_puppies : ℝ) * puppy_weight
  let total_cat_weight := (num_cats : ℝ) * cat_weight
  total_cat_weight - total_puppy_weight = 5
  := by sorry

end cat_puppy_weight_difference_l1849_184934


namespace simplify_expression_l1849_184946

theorem simplify_expression (w : ℝ) : w + 2 - 3*w - 4 + 5*w + 6 - 7*w - 8 = -4*w - 4 := by
  sorry

end simplify_expression_l1849_184946


namespace intersection_M_N_l1849_184920

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x : ℕ | x > 0 ∧ 4 % x = 0}

theorem intersection_M_N : M ∩ N = {1, 2, 4} := by sorry

end intersection_M_N_l1849_184920


namespace number_of_workers_l1849_184994

/-- Given the wages for two groups of workers, prove the number of workers in the first group -/
theorem number_of_workers (W : ℕ) : 
  (6 * W * (9975 / (5 * 19)) = 9450) →
  (W = 15) := by
sorry

end number_of_workers_l1849_184994


namespace correct_2star_reviews_l1849_184956

/-- The number of 2-star reviews for Indigo Restaurant --/
def num_2star_reviews : ℕ := 
  let total_reviews : ℕ := 18
  let num_5star : ℕ := 6
  let num_4star : ℕ := 7
  let num_3star : ℕ := 4
  let avg_rating : ℚ := 4
  1

/-- Theorem stating that the number of 2-star reviews is correct --/
theorem correct_2star_reviews : 
  let total_reviews : ℕ := 18
  let num_5star : ℕ := 6
  let num_4star : ℕ := 7
  let num_3star : ℕ := 4
  let avg_rating : ℚ := 4
  num_2star_reviews = 1 ∧ 
  (5 * num_5star + 4 * num_4star + 3 * num_3star + 2 * num_2star_reviews : ℚ) / total_reviews = avg_rating :=
by sorry

end correct_2star_reviews_l1849_184956


namespace volume_of_one_gram_l1849_184941

/-- Given a substance with a density of 200 kg per cubic meter, 
    the volume of 1 gram of this substance is 5 cubic centimeters. -/
theorem volume_of_one_gram (density : ℝ) (h : density = 200) : 
  (1 / density) * (100 ^ 3) = 5 :=
sorry

end volume_of_one_gram_l1849_184941


namespace coefficient_x5y3_in_binomial_expansion_l1849_184936

theorem coefficient_x5y3_in_binomial_expansion :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^(8 - k) * (1 : ℕ)^k) = 256 ∧
  Nat.choose 8 3 = 56 :=
by sorry

end coefficient_x5y3_in_binomial_expansion_l1849_184936
