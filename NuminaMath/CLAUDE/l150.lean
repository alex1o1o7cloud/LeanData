import Mathlib

namespace shark_sightings_problem_l150_15032

/-- Shark sightings problem -/
theorem shark_sightings_problem 
  (daytona : ℕ) 
  (cape_may long_beach santa_cruz : ℕ) :
  daytona = 26 ∧
  daytona = 3 * cape_may + 5 ∧
  long_beach = 2 * cape_may ∧
  long_beach = daytona - 4 ∧
  santa_cruz = cape_may + long_beach + 3 ∧
  santa_cruz = daytona - 9 →
  cape_may = 7 ∧ long_beach = 22 ∧ santa_cruz = 32 := by
sorry

end shark_sightings_problem_l150_15032


namespace even_function_property_l150_15021

theorem even_function_property (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x > 0, f x = 10^x) →  -- f(x) = 10^x for x > 0
  (∀ x < 0, f x = 10^(-x)) := by  -- f(x) = 10^(-x) for x < 0
sorry

end even_function_property_l150_15021


namespace cyclists_speed_l150_15038

/-- Proves that the cyclist's speed is 11 miles per hour given the problem conditions --/
theorem cyclists_speed (hiker_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) :
  hiker_speed = 4 →
  cyclist_travel_time = 5 / 60 →
  hiker_catch_up_time = 13.75 / 60 →
  ∃ (cyclist_speed : ℝ), cyclist_speed = 11 :=
by
  sorry

#check cyclists_speed

end cyclists_speed_l150_15038


namespace monotonic_increasing_iff_b_range_l150_15035

/-- The function y = (1/3)x³ + bx² + (b+2)x + 3 is monotonically increasing on ℝ 
    if and only if b < -1 or b > 2 -/
theorem monotonic_increasing_iff_b_range (b : ℝ) : 
  (∀ x : ℝ, StrictMono (fun x => (1/3) * x^3 + b * x^2 + (b + 2) * x + 3)) ↔ 
  (b < -1 ∨ b > 2) := by
sorry

end monotonic_increasing_iff_b_range_l150_15035


namespace number_of_routes_l150_15055

def grid_size : ℕ := 3

def total_moves : ℕ := 2 * grid_size

def right_moves : ℕ := grid_size

def down_moves : ℕ := grid_size

theorem number_of_routes : Nat.choose total_moves right_moves = 20 := by
  sorry

end number_of_routes_l150_15055


namespace tangent_line_count_possibilities_l150_15007

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ
  radius_pos : radius > 0

/-- Counts the number of distinct values in a list of natural numbers -/
def countDistinctValues (list : List ℕ) : ℕ :=
  (list.toFinset).card

/-- The possible numbers of tangent lines for two non-overlapping circles -/
def possibleTangentLineCounts : List ℕ := [0, 3, 4]

/-- Theorem stating that for two non-overlapping circles with radii 5 and 8,
    the number of possible distinct values for the count of tangent lines is 3 -/
theorem tangent_line_count_possibilities (circle1 circle2 : Circle)
    (h1 : circle1.radius = 5)
    (h2 : circle2.radius = 8)
    (h_non_overlap : circle1 ≠ circle2) :
    countDistinctValues possibleTangentLineCounts = 3 := by
  sorry

end tangent_line_count_possibilities_l150_15007


namespace casket_inscription_proof_l150_15017

/-- Represents a craftsman who can make caskets -/
inductive Craftsman
| Bellini
| Cellini
| CelliniSon

/-- Represents a casket with an inscription -/
structure Casket where
  maker : Craftsman
  inscription : String

/-- Determines if an inscription is true for a pair of caskets -/
def isInscriptionTrue (c1 c2 : Casket) (inscription : String) : Prop :=
  match inscription with
  | "At least one of these boxes was made by Cellini's son" =>
    c1.maker = Craftsman.CelliniSon ∨ c2.maker = Craftsman.CelliniSon
  | _ => False

/-- Cellini's son never engraves true statements -/
axiom celliniSonFalsity (c : Casket) :
  c.maker = Craftsman.CelliniSon → ¬(isInscriptionTrue c c c.inscription)

/-- The inscription that solves the problem -/
def problemInscription : String :=
  "At least one of these boxes was made by Cellini's son"

theorem casket_inscription_proof :
  ∃ (c1 c2 : Casket),
    (c1.inscription = problemInscription) ∧
    (c2.inscription = problemInscription) ∧
    (c1.maker = c2.maker) ∧
    (c1.maker = Craftsman.Bellini ∨ c1.maker = Craftsman.Cellini) ∧
    (¬∃ (c : Casket), c.inscription = problemInscription →
      c.maker = Craftsman.Bellini ∨ c.maker = Craftsman.Cellini) ∧
    (∀ (c : Casket), c.inscription = problemInscription →
      ¬(c.maker = Craftsman.Bellini ∨ c.maker = Craftsman.Cellini)) :=
by
  sorry


end casket_inscription_proof_l150_15017


namespace min_a_for_monotonic_odd_function_l150_15048

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x > 0 then Real.exp x + a
  else if x < 0 then -(Real.exp (-x) + a)
  else 0

-- State the theorem
theorem min_a_for_monotonic_odd_function :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = -f a (-x)) → -- f is odd
  (∀ x y : ℝ, x < y → f a x ≤ f a y) → -- f is monotonic
  a ≥ -1 ∧ 
  ∀ b : ℝ, (∀ x : ℝ, f b x = -f b (-x)) → 
            (∀ x y : ℝ, x < y → f b x ≤ f b y) → 
            b ≥ a :=
by sorry

end min_a_for_monotonic_odd_function_l150_15048


namespace stating_at_least_two_different_selections_l150_15069

/-- The number of available courses -/
def num_courses : ℕ := 6

/-- The number of courses each student must choose -/
def courses_per_student : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- 
Theorem stating that the number of ways in which at least two out of three students 
can select different combinations of 2 courses from a set of 6 courses is equal to 2520.
-/
theorem at_least_two_different_selections : 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose num_courses courses_per_student) * num_students - 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student) * num_students + 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose (num_courses - 2 * courses_per_student) courses_per_student) = 2520 :=
by sorry

end stating_at_least_two_different_selections_l150_15069


namespace greatest_integer_with_gcd_six_l150_15053

theorem greatest_integer_with_gcd_six (n : ℕ) : 
  (n < 50 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 50 ∧ Nat.gcd m 18 = 6 → m ≤ n) → n = 42 := by
  sorry

end greatest_integer_with_gcd_six_l150_15053


namespace min_value_of_sum_of_squares_l150_15009

theorem min_value_of_sum_of_squares (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 → (a + 2)^2 + (b + 2)^2 ≤ (x + 2)^2 + (y + 2)^2) ∧
  (a + 2)^2 + (b + 2)^2 = 25/2 :=
sorry

end min_value_of_sum_of_squares_l150_15009


namespace interior_angle_sum_regular_polygon_l150_15073

theorem interior_angle_sum_regular_polygon (n : ℕ) (h : n > 2) :
  let exterior_angle : ℝ := 20
  let interior_angle_sum : ℝ := (n - 2) * 180
  (360 / exterior_angle = n) →
  interior_angle_sum = 2880 :=
by sorry

end interior_angle_sum_regular_polygon_l150_15073


namespace area_of_triangle_l150_15063

noncomputable def m (x y : ℝ) : ℝ × ℝ := (2 * Real.cos x, y - 2 * Real.sqrt 3 * Real.sin x * Real.cos x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (1, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + 1

theorem area_of_triangle (x y a b c : ℝ) : 
  (∃ k : ℝ, m x y = k • n x) → 
  f (c / 2) = 3 → 
  c = 2 * Real.sqrt 6 → 
  a + b = 6 → 
  (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = Real.sqrt 3 := by
sorry

end area_of_triangle_l150_15063


namespace trapezoid_determines_plane_l150_15071

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A trapezoid in 3D space --/
structure Trapezoid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  parallel_sides : (A.x - B.x) * (C.y - D.y) = (A.y - B.y) * (C.x - D.x) ∧
                   (A.x - D.x) * (B.y - C.y) = (A.y - D.y) * (B.x - C.x)

/-- A plane in 3D space --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to determine if a point lies on a plane --/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Theorem: A trapezoid uniquely determines a plane --/
theorem trapezoid_determines_plane (t : Trapezoid) :
  ∃! plane : Plane, point_on_plane t.A plane ∧
                    point_on_plane t.B plane ∧
                    point_on_plane t.C plane ∧
                    point_on_plane t.D plane :=
sorry

end trapezoid_determines_plane_l150_15071


namespace four_digit_perfect_squares_l150_15096

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

theorem four_digit_perfect_squares :
  (∀ n : ℕ, is_four_digit n ∧ all_even_digits n ∧ ∃ k, n = k^2 ↔ 
    n = 4624 ∨ n = 6084 ∨ n = 6400 ∨ n = 8464) ∧
  (¬ ∃ n : ℕ, is_four_digit n ∧ all_odd_digits n ∧ ∃ k, n = k^2) :=
sorry

end four_digit_perfect_squares_l150_15096


namespace complement_intersection_equals_set_l150_15095

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem complement_intersection_equals_set : 
  (A ∪ B)ᶜ ∩ (A ∩ B)ᶜ = {-1, 1} := by sorry

end complement_intersection_equals_set_l150_15095


namespace sequence_pattern_l150_15058

def sequence_sum (a b : ℕ) : ℕ := a + b - 1

theorem sequence_pattern : 
  (sequence_sum 6 7 = 12) ∧
  (sequence_sum 8 9 = 16) ∧
  (sequence_sum 5 6 = 10) ∧
  (sequence_sum 7 8 = 14) →
  sequence_sum 3 3 = 5 := by
sorry

end sequence_pattern_l150_15058


namespace min_wins_to_advance_exactly_ten_wins_l150_15091

def football_advancement (total_matches win_matches loss_matches : ℕ) : Prop :=
  let draw_matches := total_matches - win_matches - loss_matches
  3 * win_matches + draw_matches ≥ 33

theorem min_wins_to_advance :
  ∀ win_matches : ℕ,
    football_advancement 15 win_matches 2 →
    win_matches ≥ 10 :=
by
  sorry

theorem exactly_ten_wins :
  football_advancement 15 10 2 ∧
  ∀ win_matches : ℕ, win_matches < 10 → ¬(football_advancement 15 win_matches 2) :=
by
  sorry

end min_wins_to_advance_exactly_ten_wins_l150_15091


namespace car_trip_average_speed_l150_15040

/-- Calculates the average speed of a car trip given the following conditions:
  * The total trip duration is 6 hours
  * The car travels at an average speed of 75 mph for the first 4 hours
  * The car travels at an average speed of 60 mph for the remaining hours
-/
theorem car_trip_average_speed : 
  let total_time : ℝ := 6
  let first_part_time : ℝ := 4
  let second_part_time : ℝ := total_time - first_part_time
  let first_part_speed : ℝ := 75
  let second_part_speed : ℝ := 60
  let total_distance : ℝ := first_part_speed * first_part_time + second_part_speed * second_part_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 70 := by sorry

end car_trip_average_speed_l150_15040


namespace overlap_difference_l150_15000

def total_students : ℕ := 232
def geometry_students : ℕ := 144
def biology_students : ℕ := 119

theorem overlap_difference :
  (min geometry_students biology_students) - 
  (geometry_students + biology_students - total_students) = 88 :=
by sorry

end overlap_difference_l150_15000


namespace b_alone_time_l150_15097

/-- The time it takes for A and B together to complete the task -/
def time_AB : ℝ := 3

/-- The time it takes for B and C together to complete the task -/
def time_BC : ℝ := 6

/-- The time it takes for A and C together to complete the task -/
def time_AC : ℝ := 4.5

/-- The rate at which A completes the task -/
def rate_A : ℝ := sorry

/-- The rate at which B completes the task -/
def rate_B : ℝ := sorry

/-- The rate at which C completes the task -/
def rate_C : ℝ := sorry

theorem b_alone_time (h1 : rate_A + rate_B = 1 / time_AB)
                     (h2 : rate_B + rate_C = 1 / time_BC)
                     (h3 : rate_A + rate_C = 1 / time_AC) :
  1 / rate_B = 7.2 := by
  sorry

end b_alone_time_l150_15097


namespace horner_method_value_l150_15020

def horner_polynomial (x : ℝ) : ℝ := (((-6 * x + 5) * x + 0) * x + 2) * x + 6

theorem horner_method_value :
  horner_polynomial 3 = -115 := by
  sorry

end horner_method_value_l150_15020


namespace power_division_equality_l150_15086

theorem power_division_equality (a b : ℝ) : (a^2 * b)^3 / ((-a * b)^2) = a^4 * b := by sorry

end power_division_equality_l150_15086


namespace slope_equals_one_implies_m_equals_one_l150_15079

/-- Given two points M(-2, m) and N(m, 4), if the slope of the line passing through M and N
    is equal to 1, then m = 1. -/
theorem slope_equals_one_implies_m_equals_one (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end slope_equals_one_implies_m_equals_one_l150_15079


namespace parallelogram_area_24_16_l150_15067

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters -/
theorem parallelogram_area_24_16 : parallelogram_area 24 16 = 384 := by
  sorry

end parallelogram_area_24_16_l150_15067


namespace trigonometric_inequality_l150_15003

theorem trigonometric_inequality : 
  let a := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let b := (2 * Real.tan (76 * π / 180)) / (1 + Real.tan (76 * π / 180) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  c < a ∧ a < b := by sorry

end trigonometric_inequality_l150_15003


namespace sequence_general_term_l150_15042

theorem sequence_general_term (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ+, S n = (n + 2 : ℚ) / 3 * a n) →
  ∀ n : ℕ+, a n = (n * (n + 1) : ℚ) / 2 := by
  sorry

end sequence_general_term_l150_15042


namespace travel_statements_correct_l150_15099

/-- Represents a traveler (cyclist or motorcyclist) --/
structure Traveler where
  startTime : ℝ
  arrivalTime : ℝ
  distanceTraveled : ℝ → ℝ
  speed : ℝ → ℝ

/-- The travel scenario between two towns --/
structure TravelScenario where
  cyclist : Traveler
  motorcyclist : Traveler
  totalDistance : ℝ

/-- Properties of the travel scenario --/
def TravelScenario.properties (scenario : TravelScenario) : Prop :=
  -- The total distance is 80km
  scenario.totalDistance = 80 ∧
  -- The cyclist starts 3 hours before the motorcyclist
  scenario.cyclist.startTime + 3 = scenario.motorcyclist.startTime ∧
  -- The cyclist arrives 1 hour before the motorcyclist
  scenario.cyclist.arrivalTime + 1 = scenario.motorcyclist.arrivalTime ∧
  -- The cyclist's speed pattern (acceleration then constant)
  (∃ t₀ : ℝ, ∀ t, t ≥ scenario.cyclist.startTime → 
    (t ≤ t₀ → scenario.cyclist.speed t < scenario.cyclist.speed (t + 1)) ∧
    (t > t₀ → scenario.cyclist.speed t = scenario.cyclist.speed t₀)) ∧
  -- The motorcyclist's constant speed
  (∀ t₁ t₂, scenario.motorcyclist.speed t₁ = scenario.motorcyclist.speed t₂) ∧
  -- The catch-up time
  (∃ t : ℝ, t = scenario.motorcyclist.startTime + 1.5 ∧
    scenario.cyclist.distanceTraveled t = scenario.motorcyclist.distanceTraveled t)

/-- The main theorem stating the correctness of all statements --/
theorem travel_statements_correct (scenario : TravelScenario) 
  (h : scenario.properties) : 
  -- Statement 1: Timing difference
  (scenario.cyclist.startTime + 3 = scenario.motorcyclist.startTime ∧
   scenario.cyclist.arrivalTime + 1 = scenario.motorcyclist.arrivalTime) ∧
  -- Statement 2: Speed patterns
  (∃ t₀ : ℝ, ∀ t, t ≥ scenario.cyclist.startTime → 
    (t ≤ t₀ → scenario.cyclist.speed t < scenario.cyclist.speed (t + 1)) ∧
    (t > t₀ → scenario.cyclist.speed t = scenario.cyclist.speed t₀)) ∧
  (∀ t₁ t₂, scenario.motorcyclist.speed t₁ = scenario.motorcyclist.speed t₂) ∧
  -- Statement 3: Catch-up time
  (∃ t : ℝ, t = scenario.motorcyclist.startTime + 1.5 ∧
    scenario.cyclist.distanceTraveled t = scenario.motorcyclist.distanceTraveled t) :=
by sorry

end travel_statements_correct_l150_15099


namespace max_triangle_area_in_ellipse_l150_15057

/-- The maximum area of a triangle inscribed in an ellipse with semi-axes a and b -/
theorem max_triangle_area_in_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (A : ℝ), A = a * b * (3 * Real.sqrt 3) / 4 ∧
  ∀ (triangle_area : ℝ), triangle_area ≤ A :=
sorry

end max_triangle_area_in_ellipse_l150_15057


namespace max_apartments_in_complex_l150_15093

/-- Represents an apartment complex -/
structure ApartmentComplex where
  num_buildings : ℕ
  num_floors : ℕ
  apartments_per_floor : ℕ

/-- The maximum number of apartments in the complex -/
def max_apartments (complex : ApartmentComplex) : ℕ :=
  complex.num_buildings * complex.num_floors * complex.apartments_per_floor

/-- Theorem stating the maximum number of apartments in the given complex -/
theorem max_apartments_in_complex :
  ∃ (complex : ApartmentComplex),
    complex.num_buildings ≤ 22 ∧
    complex.num_buildings > 0 ∧
    complex.num_floors ≤ 6 ∧
    complex.apartments_per_floor = 5 ∧
    max_apartments complex = 660 := by
  sorry

end max_apartments_in_complex_l150_15093


namespace min_cuts_for_eleven_days_max_rings_for_n_cuts_l150_15045

/-- Represents a chain of rings -/
structure Chain where
  rings : ℕ

/-- Represents a stay at the inn -/
structure Stay where
  days : ℕ

/-- Calculates the minimum number of cuts required for a given chain and stay -/
def minCuts (chain : Chain) (stay : Stay) : ℕ :=
  sorry

/-- Calculates the maximum number of rings in a chain for a given number of cuts -/
def maxRings (cuts : ℕ) : ℕ :=
  sorry

theorem min_cuts_for_eleven_days (chain : Chain) (stay : Stay) :
  chain.rings = 11 → stay.days = 11 → minCuts chain stay = 2 :=
  sorry

theorem max_rings_for_n_cuts (n : ℕ) :
  maxRings n = (n + 1) * 2^n - 1 :=
  sorry

end min_cuts_for_eleven_days_max_rings_for_n_cuts_l150_15045


namespace compound_composition_l150_15027

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Carbon atoms in the compound -/
def num_C : ℕ := 6

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 7

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 192

/-- Calculates the number of Hydrogen atoms in the compound -/
def num_H : ℕ := 8

theorem compound_composition :
  (num_C : ℝ) * atomic_weight_C + (num_O : ℝ) * atomic_weight_O + (num_H : ℝ) * atomic_weight_H = molecular_weight := by
  sorry

end compound_composition_l150_15027


namespace inequality_equivalence_l150_15081

theorem inequality_equivalence (x : ℝ) : 
  (2 < (x - 1)⁻¹ ∧ (x - 1)⁻¹ < 3 ∧ x ≠ 1) ↔ (4/3 < x ∧ x < 3/2) :=
sorry

end inequality_equivalence_l150_15081


namespace function_properties_l150_15028

open Real

theorem function_properties (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
    (h_diff : DifferentiableOn ℝ f (Set.Icc a b)) (h_a_lt_b : a < b)
    (h_f'_a : deriv f a > 0) (h_f'_b : deriv f b < 0) :
  (∃ x₀ ∈ Set.Icc a b, f x₀ > f b) ∧
  (∃ x₀ ∈ Set.Icc a b, f a - f b = (deriv (deriv f)) x₀ * (a - b)) :=
by sorry

end function_properties_l150_15028


namespace range_of_m_for_real_roots_l150_15029

theorem range_of_m_for_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m = 0 ∨ (m-1)*x^2 + 2*x + 1 = 0 ∨ (m-2)*x^2 + 2*x - 1 = 0) →
  (∃ y : ℝ, y^2 - y + m = 0 ∨ (m-1)*y^2 + 2*y + 1 = 0 ∨ (m-2)*y^2 + 2*y - 1 = 0) →
  (x ≠ y) →
  (m ≤ 1/4 ∨ (1 ≤ m ∧ m ≤ 2)) := by
sorry

end range_of_m_for_real_roots_l150_15029


namespace canoe_kayak_difference_l150_15036

/-- Represents the daily rental cost of a canoe in dollars -/
def canoe_cost : ℚ := 11

/-- Represents the daily rental cost of a kayak in dollars -/
def kayak_cost : ℚ := 16

/-- Represents the ratio of canoes to kayaks rented -/
def rental_ratio : ℚ := 4 / 3

/-- Represents the total revenue in dollars -/
def total_revenue : ℚ := 460

/-- Represents the number of kayaks rented -/
def kayaks : ℕ := 15

/-- Represents the number of canoes rented -/
def canoes : ℕ := 20

theorem canoe_kayak_difference :
  canoes - kayaks = 5 ∧
  canoe_cost * canoes + kayak_cost * kayaks = total_revenue ∧
  (canoes : ℚ) / kayaks = rental_ratio := by
  sorry

end canoe_kayak_difference_l150_15036


namespace inequality_proof_l150_15039

theorem inequality_proof (a r : ℝ) (n : ℕ) 
  (ha : a ≥ -2) (hr : r ≥ 0) (hn : n ≥ 1) :
  r^(2*n) + a*r^n + 1 ≥ (1 - r)^(2*n) := by
  sorry

end inequality_proof_l150_15039


namespace total_workers_is_18_l150_15059

/-- Represents the total number of workers in a workshop -/
def total_workers : ℕ := sorry

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 6

/-- Represents the average salary of all workers in the workshop -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians in the workshop -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technicians in the workshop -/
def avg_salary_non_technicians : ℕ := 6000

/-- Theorem stating that given the conditions, the total number of workers is 18 -/
theorem total_workers_is_18 :
  (total_workers * avg_salary_all = 
    num_technicians * avg_salary_technicians + 
    (total_workers - num_technicians) * avg_salary_non_technicians) →
  total_workers = 18 := by
  sorry

end total_workers_is_18_l150_15059


namespace function_range_properties_l150_15049

open Set

/-- Given a function f with maximum M and minimum m on [a, b], prove the following statements -/
theorem function_range_properties
  (f : ℝ → ℝ) (a b M m : ℝ) (h_max : ∀ x ∈ Icc a b, f x ≤ M) (h_min : ∀ x ∈ Icc a b, m ≤ f x) :
  (∀ p, (∀ x ∈ Icc a b, p ≤ f x) → p ∈ Iic m) ∧
  (∀ p, (∃ x ∈ Icc a b, p = f x) → p ∈ Icc m M) ∧
  (∀ p, (∃ x ∈ Icc a b, p ≤ f x) → p ∈ Iic M) :=
by sorry


end function_range_properties_l150_15049


namespace complexity_not_greater_for_power_of_two_exists_number_with_greater_or_equal_complexity_l150_15056

/-- The complexity of an integer is the number of factors in its prime factorization -/
def complexity (n : ℕ) : ℕ := sorry

/-- For n = 2^k, all numbers between n and 2n have complexity not greater than that of n -/
theorem complexity_not_greater_for_power_of_two (k : ℕ) :
  ∀ m : ℕ, 2^k ≤ m → m ≤ 2^(k+1) → complexity m ≤ complexity (2^k) := by sorry

/-- For any n > 1, there exists at least one number between n and 2n with complexity greater than or equal to that of n -/
theorem exists_number_with_greater_or_equal_complexity (n : ℕ) (h : n > 1) :
  ∃ m : ℕ, n < m ∧ m < 2*n ∧ complexity m ≥ complexity n := by sorry

end complexity_not_greater_for_power_of_two_exists_number_with_greater_or_equal_complexity_l150_15056


namespace cyclic_quadrilateral_angle_l150_15025

/-- Represents a cyclic quadrilateral ABCD with angles α, β, γ, and ω -/
structure CyclicQuadrilateral where
  α : ℝ
  β : ℝ
  γ : ℝ
  ω : ℝ
  sum_180 : α + β + γ + ω = 180

/-- Theorem: In a cyclic quadrilateral ABCD, if α = c, β = 43°, γ = 59°, and ω = d, then d = 42° -/
theorem cyclic_quadrilateral_angle (q : CyclicQuadrilateral) (h1 : q.α = 36) (h2 : q.β = 43) (h3 : q.γ = 59) : q.ω = 42 := by
  sorry

#check cyclic_quadrilateral_angle

end cyclic_quadrilateral_angle_l150_15025


namespace circle_intersection_radius_range_l150_15004

/-- The range of r values for which there are two points P satisfying the given conditions -/
theorem circle_intersection_radius_range :
  ∀ (r : ℝ),
  (∃ (P₁ P₂ : ℝ × ℝ),
    P₁ ≠ P₂ ∧
    ((P₁.1 - 3)^2 + (P₁.2 - 4)^2 = r^2) ∧
    ((P₂.1 - 3)^2 + (P₂.2 - 4)^2 = r^2) ∧
    ((P₁.1 + 2)^2 + P₁.2^2 + (P₁.1 - 2)^2 + P₁.2^2 = 40) ∧
    ((P₂.1 + 2)^2 + P₂.2^2 + (P₂.1 - 2)^2 + P₂.2^2 = 40)) ↔
  (1 < r ∧ r < 9) :=
by sorry

end circle_intersection_radius_range_l150_15004


namespace inequality_implication_l150_15085

theorem inequality_implication (a b c : ℝ) (h : a < b) : -a * c^2 ≥ -b * c^2 := by
  sorry

end inequality_implication_l150_15085


namespace problem_statement_l150_15074

theorem problem_statement (a b c d x : ℤ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : x = -1)  -- x is the largest negative integer
  : x^2 - (a + b - c * d)^2012 + (-c * d)^2011 = -1 := by
  sorry

end problem_statement_l150_15074


namespace line_translation_distance_l150_15016

/-- Two lines in a 2D Cartesian coordinate system -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- The vertical distance between two parallel lines -/
def vertical_distance (l1 l2 : Line2D) : ℝ :=
  l2.intercept - l1.intercept

/-- Theorem: The vertical distance between l1 and l2 is 6 units -/
theorem line_translation_distance :
  let l1 : Line2D := { slope := -2, intercept := -2 }
  let l2 : Line2D := { slope := -2, intercept := 4 }
  vertical_distance l1 l2 = 6 := by
  sorry


end line_translation_distance_l150_15016


namespace martha_blocks_l150_15011

/-- The number of blocks Martha ends with is equal to her initial blocks plus the blocks she finds -/
theorem martha_blocks (initial_blocks found_blocks : ℕ) :
  initial_blocks + found_blocks = initial_blocks + found_blocks :=
by sorry

#check martha_blocks 4 80

end martha_blocks_l150_15011


namespace binomial_variance_three_fourths_l150_15030

def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_three_fourths (p : ℝ) 
  (h1 : 0 ≤ p) (h2 : p ≤ 1) 
  (h3 : binomial_variance 3 p = 3/4) : p = 1/2 := by
  sorry

end binomial_variance_three_fourths_l150_15030


namespace smallest_numbers_with_special_property_l150_15033

theorem smallest_numbers_with_special_property :
  ∃ (a b : ℕ), a > b ∧ 
    (∃ (k : ℕ), a^2 - b^2 = k^3) ∧
    (∃ (m : ℕ), a^3 - b^3 = m^2) ∧
    (∀ (x y : ℕ), x > y → 
      (∃ (k : ℕ), x^2 - y^2 = k^3) →
      (∃ (m : ℕ), x^3 - y^3 = m^2) →
      (x > a ∨ (x = a ∧ y ≥ b))) ∧
    a = 10 ∧ b = 6 :=
by sorry

end smallest_numbers_with_special_property_l150_15033


namespace mrs_hilt_total_distance_l150_15031

/-- Calculate the total distance walked by Mrs. Hilt -/
def total_distance (
  water_fountain_dist : ℝ)
  (main_office_dist : ℝ)
  (teachers_lounge_dist : ℝ)
  (water_fountain_increase : ℝ)
  (main_office_increase : ℝ)
  (teachers_lounge_increase : ℝ)
  (water_fountain_visits : ℕ)
  (main_office_visits : ℕ)
  (teachers_lounge_visits : ℕ) : ℝ :=
  let water_fountain_return := water_fountain_dist * (1 + water_fountain_increase)
  let main_office_return := main_office_dist * (1 + main_office_increase)
  let teachers_lounge_return := teachers_lounge_dist * (1 + teachers_lounge_increase)
  (water_fountain_dist + water_fountain_return) * water_fountain_visits +
  (main_office_dist + main_office_return) * main_office_visits +
  (teachers_lounge_dist + teachers_lounge_return) * teachers_lounge_visits

/-- Theorem stating that Mrs. Hilt's total walking distance is 699 feet -/
theorem mrs_hilt_total_distance :
  total_distance 30 50 35 0.15 0.10 0.20 4 2 3 = 699 := by
  sorry

end mrs_hilt_total_distance_l150_15031


namespace log_product_less_than_one_l150_15013

theorem log_product_less_than_one : Real.log 9 * Real.log 11 < 1 := by
  sorry

end log_product_less_than_one_l150_15013


namespace cos_theta_value_l150_15088

theorem cos_theta_value (θ : Real) 
  (h : (1 + Real.sin θ + Real.cos θ) / (1 + Real.sin θ - Real.cos θ) = 1/2) : 
  Real.cos θ = -3/5 := by
sorry

end cos_theta_value_l150_15088


namespace trigonometric_equation_solution_l150_15061

theorem trigonometric_equation_solution (x : ℝ) : 
  3 - 7 * (Real.cos x)^2 * Real.sin x - 3 * (Real.sin x)^3 = 0 ↔ 
  (∃ k : ℤ, x = π / 2 + 2 * k * π) ∨ 
  (∃ k : ℤ, x = (-1)^k * π / 6 + k * π) :=
by sorry

end trigonometric_equation_solution_l150_15061


namespace sin_cube_identity_l150_15094

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = (-1/4 : ℝ) * Real.sin (3 * θ) + (3/4 : ℝ) * Real.sin θ := by
  sorry

end sin_cube_identity_l150_15094


namespace zoo_animals_legs_count_l150_15046

theorem zoo_animals_legs_count : 
  ∀ (total_heads : ℕ) (rabbit_count : ℕ) (peacock_count : ℕ),
    total_heads = 60 →
    rabbit_count = 36 →
    peacock_count = total_heads - rabbit_count →
    4 * rabbit_count + 2 * peacock_count = 192 :=
by
  sorry

end zoo_animals_legs_count_l150_15046


namespace arthurs_walk_distance_l150_15051

/-- Represents Arthur's walk in blocks -/
structure ArthursWalk where
  east : ℕ
  north : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the total distance of Arthur's walk in miles -/
def total_distance (walk : ArthursWalk) : ℚ :=
  (walk.east + walk.north + walk.south + walk.west) * (1 / 4)

/-- Theorem: Arthur's specific walk equals 6.5 miles -/
theorem arthurs_walk_distance :
  let walk : ArthursWalk := { east := 8, north := 10, south := 3, west := 5 }
  total_distance walk = 13 / 2 := by
  sorry

end arthurs_walk_distance_l150_15051


namespace simplify_trig_expression_l150_15041

theorem simplify_trig_expression :
  let sin30 : ℝ := 1 / 2
  let cos30 : ℝ := Real.sqrt 3 / 2
  ∀ (sin10 sin20 cos10 : ℝ),
    (sin10 + sin20 * cos30) / (cos10 - sin20 * sin30) = Real.sqrt 3 / 3 :=
by sorry

end simplify_trig_expression_l150_15041


namespace final_bill_calculation_l150_15075

def original_bill : ℝ := 400
def late_charge_rate : ℝ := 0.02

def final_amount : ℝ := original_bill * (1 + late_charge_rate)^3

theorem final_bill_calculation : 
  ∃ (ε : ℝ), abs (final_amount - 424.48) < ε ∧ ε > 0 :=
by sorry

end final_bill_calculation_l150_15075


namespace sock_ratio_l150_15010

/-- The ratio of black socks to blue socks in Mr. Lin's original order --/
theorem sock_ratio : 
  ∀ (x y : ℕ), 
  (x = 4) → -- Mr. Lin ordered 4 pairs of black socks
  ((2 * x + y) * 3 = (2 * y + x) * 2) → -- 50% increase when swapped
  (y = 4 * x) -- Ratio of black to blue is 1:4
  := by sorry

end sock_ratio_l150_15010


namespace committee_selection_l150_15008

theorem committee_selection (n : ℕ) (h : Nat.choose n 2 = 15) : Nat.choose n 4 = 15 := by
  sorry

end committee_selection_l150_15008


namespace probability_from_odds_probability_3_5_odds_l150_15052

/-- Given odds of a:b in favor of an event, the probability of the event occurring is a/(a+b) -/
theorem probability_from_odds (a b : ℕ) (h : a > 0 ∧ b > 0) :
  let odds := a / b
  let probability := a / (a + b)
  probability = odds / (1 + odds) :=
by sorry

/-- The probability of an event with odds 3:5 in its favor is 3/8 -/
theorem probability_3_5_odds :
  let a := 3
  let b := 5
  let probability := a / (a + b)
  probability = 3 / 8 :=
by sorry

end probability_from_odds_probability_3_5_odds_l150_15052


namespace correct_operation_l150_15037

theorem correct_operation : ∀ x : ℝ, 
  (∃ y : ℝ, y ^ 2 = 4 ∧ y > 0) ∧ 
  (3 * x^3 + 2 * x^3 ≠ 5 * x^6) ∧ 
  ((x + 1)^2 ≠ x^2 + 1) ∧ 
  (x^8 / x^4 ≠ x^2) :=
by sorry

end correct_operation_l150_15037


namespace polly_breakfast_time_l150_15083

/-- The number of minutes Polly spends cooking breakfast every day -/
def breakfast_time : ℕ := sorry

/-- The number of minutes Polly spends cooking lunch every day -/
def lunch_time : ℕ := 5

/-- The number of days in a week Polly spends 10 minutes cooking dinner -/
def short_dinner_days : ℕ := 4

/-- The number of minutes Polly spends cooking dinner on short dinner days -/
def short_dinner_time : ℕ := 10

/-- The number of minutes Polly spends cooking dinner on long dinner days -/
def long_dinner_time : ℕ := 30

/-- The total number of minutes Polly spends cooking in a week -/
def total_cooking_time : ℕ := 305

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem polly_breakfast_time :
  breakfast_time * days_in_week +
  lunch_time * days_in_week +
  short_dinner_time * short_dinner_days +
  long_dinner_time * (days_in_week - short_dinner_days) =
  total_cooking_time ∧
  breakfast_time = 20 := by sorry

end polly_breakfast_time_l150_15083


namespace choose_product_equals_8400_l150_15077

theorem choose_product_equals_8400 : Nat.choose 10 3 * Nat.choose 8 4 = 8400 := by
  sorry

end choose_product_equals_8400_l150_15077


namespace ensemble_size_l150_15054

/-- Represents the "Sunshine" ensemble --/
structure Ensemble where
  violin_players : ℕ
  bass_players : ℕ
  violin_avg_age : ℝ
  bass_avg_age : ℝ

/-- Represents the ensemble after Igor's switch --/
structure EnsembleAfterSwitch where
  violin_players : ℕ
  bass_players : ℕ
  violin_avg_age : ℝ
  bass_avg_age : ℝ

/-- Theorem stating the size of the ensemble --/
theorem ensemble_size (e : Ensemble) (e_after : EnsembleAfterSwitch) : 
  e.violin_players + e.bass_players = 23 :=
by
  have h1 : e.violin_avg_age = 22 := by sorry
  have h2 : e.bass_avg_age = 45 := by sorry
  have h3 : e_after.violin_players = e.violin_players + 1 := by sorry
  have h4 : e_after.bass_players = e.bass_players - 1 := by sorry
  have h5 : e_after.violin_avg_age = e.violin_avg_age + 1 := by sorry
  have h6 : e_after.bass_avg_age = e.bass_avg_age + 1 := by sorry
  sorry

#check ensemble_size

end ensemble_size_l150_15054


namespace sufficient_not_necessary_l150_15018

theorem sufficient_not_necessary (x : ℝ) : 
  (1 / x > 2 → x < 1 / 2) ∧ ¬(x < 1 / 2 → 1 / x > 2) := by
  sorry

end sufficient_not_necessary_l150_15018


namespace parallel_vectors_x_value_l150_15062

def vector_a : ℝ × ℝ := (3, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (vector_a.1 * (vector_b x).2 = vector_a.2 * (vector_b x).1) → x = -3 := by
  sorry

end parallel_vectors_x_value_l150_15062


namespace square_difference_identity_l150_15005

theorem square_difference_identity (x : ℝ) : (x + 1)^2 - x^2 = 2*x + 1 := by
  sorry

end square_difference_identity_l150_15005


namespace expansion_equality_l150_15002

theorem expansion_equality (m : ℝ) : (m + 2) * (m - 3) = m^2 - m - 6 := by
  sorry

end expansion_equality_l150_15002


namespace oak_grove_books_after_donations_l150_15024

/-- Represents the number of books in Oak Grove libraries -/
structure OakGroveLibraries where
  public_library : ℕ
  school_libraries : ℕ
  community_center : ℕ

/-- Calculates the total number of books after donations -/
def total_books_after_donations (libs : OakGroveLibraries) (public_donation : ℕ) (community_donation : ℕ) : ℕ :=
  libs.public_library + libs.school_libraries + libs.community_center - public_donation - community_donation

/-- Theorem stating the total number of books after donations -/
theorem oak_grove_books_after_donations :
  let initial_libraries : OakGroveLibraries := {
    public_library := 1986,
    school_libraries := 5106,
    community_center := 3462
  }
  let public_donation : ℕ := 235
  let community_donation : ℕ := 328
  total_books_after_donations initial_libraries public_donation community_donation = 9991 := by
  sorry


end oak_grove_books_after_donations_l150_15024


namespace infinitely_many_representable_terms_l150_15076

-- Define the sequence type
def PositiveIntegerSequence := ℕ → ℕ+

-- Define the property that the sequence is strictly increasing
def StrictlyIncreasing (a : PositiveIntegerSequence) : Prop :=
  ∀ k, a k < a (k + 1)

-- State the theorem
theorem infinitely_many_representable_terms 
  (a : PositiveIntegerSequence) 
  (h : StrictlyIncreasing a) : 
  ∃ S : Set ℕ, (Set.Infinite S) ∧ 
    (∀ m ∈ S, ∃ (p q x y : ℕ), 
      p ≠ q ∧ 
      x > 0 ∧ 
      y > 0 ∧ 
      (a m : ℕ) = x * (a p : ℕ) + y * (a q : ℕ)) :=
by
  sorry

end infinitely_many_representable_terms_l150_15076


namespace odd_periodic_function_sum_l150_15064

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_period : HasPeriod f 4) :
  f 2005 + f 2006 + f 2007 = f 2 := by
  sorry

end odd_periodic_function_sum_l150_15064


namespace nina_running_distance_l150_15022

theorem nina_running_distance (total : ℝ) (first : ℝ) (second_known : ℝ) 
  (h1 : total = 0.83)
  (h2 : first = 0.08)
  (h3 : second_known = 0.08) :
  total - (first + second_known) = 0.67 := by
  sorry

end nina_running_distance_l150_15022


namespace root_of_polynomial_l150_15072

theorem root_of_polynomial (a₁ a₂ a₃ a₄ a₅ b : ℤ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ 
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ 
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ 
  a₄ ≠ a₅ →
  a₁ + a₂ + a₃ + a₄ + a₅ = 9 →
  (b - a₁) * (b - a₂) * (b - a₃) * (b - a₄) * (b - a₅) = 2009 →
  b = 10 := by
sorry

end root_of_polynomial_l150_15072


namespace set_membership_implies_x_values_l150_15082

theorem set_membership_implies_x_values (A : Set ℝ) (x : ℝ) :
  A = {2, 4, x^2 - x} → 6 ∈ A → x = 3 ∨ x = -2 := by
  sorry

end set_membership_implies_x_values_l150_15082


namespace cube_pyramid_equal_volume_l150_15023

/-- Given a cube with edge length 6 and a square-based pyramid with base edge length 10,
    if their volumes are equal, then the height of the pyramid is 162/25. -/
theorem cube_pyramid_equal_volume (h : ℚ) : 
  (6 : ℚ)^3 = (1/3 : ℚ) * 10^2 * h → h = 162/25 := by
  sorry

end cube_pyramid_equal_volume_l150_15023


namespace bob_probability_after_three_turns_l150_15001

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The game state after a certain number of turns -/
structure GameState :=
  (current_player : Player)
  (turn : ℕ)

/-- The probability of a player having the ball after a certain number of turns -/
def probability_has_ball (player : Player) (turns : ℕ) : ℚ :=
  sorry

theorem bob_probability_after_three_turns :
  probability_has_ball Player.Bob 3 = 11/16 := by
  sorry

end bob_probability_after_three_turns_l150_15001


namespace cars_without_features_l150_15090

theorem cars_without_features (total : ℕ) (air_bag : ℕ) (power_windows : ℕ) (both : ℕ)
  (h1 : total = 65)
  (h2 : air_bag = 45)
  (h3 : power_windows = 30)
  (h4 : both = 12) :
  total - (air_bag + power_windows - both) = 2 := by
  sorry

end cars_without_features_l150_15090


namespace fibonacci_inequality_l150_15065

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (fibonacci n / fibonacci (n - 1) : ℚ) < (a / b : ℚ) →
  (a / b : ℚ) < (fibonacci (n + 1) / fibonacci n : ℚ) →
  b ≥ fibonacci (n + 1) :=
by
  sorry

end fibonacci_inequality_l150_15065


namespace project_completion_time_l150_15050

/-- Proves that the total time to complete a project is 15 days given the specified conditions. -/
theorem project_completion_time 
  (a_rate : ℝ) 
  (b_rate : ℝ) 
  (a_quit_before : ℝ) 
  (h1 : a_rate = 1 / 20) 
  (h2 : b_rate = 1 / 30) 
  (h3 : a_quit_before = 5) : 
  ∃ (total_time : ℝ), total_time = 15 ∧ 
    (total_time - a_quit_before) * (a_rate + b_rate) + 
    a_quit_before * b_rate = 1 :=
sorry

end project_completion_time_l150_15050


namespace total_turnips_proof_l150_15098

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := sally_turnips + mary_turnips

theorem total_turnips_proof : total_turnips = 242 := by
  sorry

end total_turnips_proof_l150_15098


namespace chad_bbq_ice_cost_l150_15066

/-- The cost of ice for Chad's BBQ --/
def ice_cost (people : ℕ) (ice_per_person : ℕ) (pack_size : ℕ) (cost_per_pack : ℚ) : ℚ :=
  let total_ice := people * ice_per_person
  let packs_needed := (total_ice + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * cost_per_pack

/-- Theorem stating the cost of ice for Chad's BBQ --/
theorem chad_bbq_ice_cost :
  ice_cost 15 2 10 3 = 9 := by
  sorry

end chad_bbq_ice_cost_l150_15066


namespace inequality_solution_l150_15092

theorem inequality_solution :
  {x : ℝ | |(6 - x) / 4| < 3 ∧ x ≥ 2} = Set.Ici 2 ∩ Set.Iio 18 := by sorry

end inequality_solution_l150_15092


namespace function_inequality_l150_15060

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_critical : deriv f 1 = 0) 
  (h_condition : ∀ x : ℝ, (x - 1) * (deriv f x) > 0) : 
  f 0 + f 2 > 2 * f 1 := by
  sorry

end function_inequality_l150_15060


namespace first_book_has_200_words_l150_15015

/-- The number of words in Jenny's first book --/
def first_book_words : ℕ := sorry

/-- The number of words Jenny can read per hour --/
def reading_speed : ℕ := 100

/-- The number of words in the second book --/
def second_book_words : ℕ := 400

/-- The number of words in the third book --/
def third_book_words : ℕ := 300

/-- The number of days Jenny plans to read --/
def reading_days : ℕ := 10

/-- The average number of minutes Jenny spends reading per day --/
def daily_reading_minutes : ℕ := 54

/-- Theorem stating that the first book has 200 words --/
theorem first_book_has_200_words :
  first_book_words = 200 := by sorry

end first_book_has_200_words_l150_15015


namespace hex_9A3_to_base_4_l150_15047

/-- Converts a single hexadecimal digit to its decimal representation -/
def hex_to_dec (h : Char) : ℕ :=
  match h with
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- Default case, though it should never be reached for valid hex digits

/-- Converts a hexadecimal number (as a string) to its decimal representation -/
def hex_to_dec_num (s : String) : ℕ :=
  s.foldl (fun acc d => 16 * acc + hex_to_dec d) 0

/-- Converts a natural number to its base 4 representation (as a list of digits) -/
def to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The main theorem: 9A3₁₆ is equal to 212203₄ -/
theorem hex_9A3_to_base_4 :
  to_base_4 (hex_to_dec_num "9A3") = [2, 1, 2, 2, 0, 3] := by
  sorry

end hex_9A3_to_base_4_l150_15047


namespace max_value_at_negative_one_l150_15006

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem max_value_at_negative_one (a b : ℝ) :
  (∀ x, f a b x ≤ 0) ∧
  (f a b (-1) = 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-1)| < δ → f a b x < f a b (-1) + ε) →
  a + b = 11 :=
by sorry

end max_value_at_negative_one_l150_15006


namespace pirate_coins_l150_15087

/-- Represents the number of pirates --/
def num_pirates : ℕ := 15

/-- Calculates the number of coins remaining after the k-th pirate takes their share --/
def coins_after (k : ℕ) (initial_coins : ℕ) : ℚ :=
  (num_pirates - k : ℚ) / num_pirates * initial_coins

/-- Checks if a given number of initial coins results in each pirate receiving a whole number of coins --/
def valid_distribution (initial_coins : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ num_pirates → (coins_after k initial_coins - coins_after (k+1) initial_coins).isInt

/-- The statement to be proved --/
theorem pirate_coins :
  ∃ initial_coins : ℕ,
    valid_distribution initial_coins ∧
    (∀ n : ℕ, n < initial_coins → ¬valid_distribution n) ∧
    coins_after (num_pirates - 1) initial_coins = 1001 := by
  sorry


end pirate_coins_l150_15087


namespace equation_simplification_l150_15012

theorem equation_simplification :
  (Real.sqrt ((7 : ℝ)^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end equation_simplification_l150_15012


namespace soccer_ball_weight_l150_15078

theorem soccer_ball_weight (soccer_ball_weight bicycle_weight : ℝ) : 
  5 * soccer_ball_weight = 3 * bicycle_weight →
  2 * bicycle_weight = 60 →
  soccer_ball_weight = 18 := by
  sorry

end soccer_ball_weight_l150_15078


namespace pentagon_rectangle_ratio_l150_15089

theorem pentagon_rectangle_ratio : 
  let pentagon_perimeter : ℝ := 100
  let rectangle_perimeter : ℝ := 100
  let pentagon_side := pentagon_perimeter / 5
  let rectangle_width := rectangle_perimeter / 6
  pentagon_side / rectangle_width = 6 / 5 := by sorry

end pentagon_rectangle_ratio_l150_15089


namespace number_from_percentage_l150_15044

theorem number_from_percentage (x : ℝ) : 0.15 * 0.30 * 0.50 * x = 117 → x = 5200 := by
  sorry

end number_from_percentage_l150_15044


namespace rebecca_work_hours_l150_15068

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Rebecca worked 56 hours. -/
theorem rebecca_work_hours :
  ∀ x : ℕ,
  let thomas_hours := x
  let toby_hours := 2 * x - 10
  let rebecca_hours := toby_hours - 8
  (thomas_hours + toby_hours + rebecca_hours = 157) →
  rebecca_hours = 56 :=
by
  sorry

end rebecca_work_hours_l150_15068


namespace difference_of_squares_l150_15014

theorem difference_of_squares (x : ℝ) : 1 - x^2 = (1 - x) * (1 + x) := by
  sorry

end difference_of_squares_l150_15014


namespace equation_solution_l150_15043

theorem equation_solution (x y : ℚ) 
  (eq1 : 4 * x + y = 20) 
  (eq2 : x + 2 * y = 17) : 
  5 * x^2 + 18 * x * y + 5 * y^2 = 696 + 5/7 := by
  sorry

end equation_solution_l150_15043


namespace encoded_CDE_value_l150_15019

/-- Represents the digits in the base 7 encoding system -/
inductive Digit
  | A | B | C | D | E | F | G

/-- Represents a number in the base 7 encoding system -/
def EncodedNumber := List Digit

/-- Converts an EncodedNumber to its base 10 representation -/
def to_base_10 : EncodedNumber → ℕ := sorry

/-- Checks if two EncodedNumbers are consecutive -/
def are_consecutive (a b : EncodedNumber) : Prop := sorry

/-- The main theorem -/
theorem encoded_CDE_value :
  ∃ (bcg bcf bad : EncodedNumber),
    (are_consecutive bcg bcf) ∧
    (are_consecutive bcf bad) ∧
    bcg = [Digit.B, Digit.C, Digit.G] ∧
    bcf = [Digit.B, Digit.C, Digit.F] ∧
    bad = [Digit.B, Digit.A, Digit.D] →
    to_base_10 [Digit.C, Digit.D, Digit.E] = 329 := by
  sorry

end encoded_CDE_value_l150_15019


namespace leading_coefficient_of_f_l150_15080

/-- Given a polynomial f satisfying f(x + 1) - f(x) = 6x + 4 for all x,
    prove that the leading coefficient of f is 3. -/
theorem leading_coefficient_of_f (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 6 * x + 4) →
  ∃ c, ∀ x, f x = 3 * x^2 + x + c :=
sorry

end leading_coefficient_of_f_l150_15080


namespace coefficient_x_squared_in_expansion_l150_15084

theorem coefficient_x_squared_in_expansion (x : ℝ) :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k : ℝ) * x^k * (1:ℝ)^(5-k)) =
  10 * x^2 + (Finset.range 6).sum (fun k => if k ≠ 2 then (Nat.choose 5 k : ℝ) * x^k * (1:ℝ)^(5-k) else 0) :=
by sorry


end coefficient_x_squared_in_expansion_l150_15084


namespace inscribed_hexagon_area_l150_15026

/-- A regular hexagon inscribed in a square with specific properties -/
structure InscribedHexagon where
  square_perimeter : ℝ
  square_side_length : ℝ
  hexagon_side_length : ℝ
  hexagon_area : ℝ
  perimeter_constraint : square_perimeter = 160
  side_length_relation : square_side_length = square_perimeter / 4
  hexagon_side_relation : hexagon_side_length = square_side_length / 2
  area_formula : hexagon_area = 3 * Real.sqrt 3 / 2 * hexagon_side_length ^ 2

/-- The theorem stating the area of the inscribed hexagon -/
theorem inscribed_hexagon_area (h : InscribedHexagon) : h.hexagon_area = 600 * Real.sqrt 3 := by
  sorry

end inscribed_hexagon_area_l150_15026


namespace fish_ratio_l150_15034

theorem fish_ratio (jerk_fish : ℕ) (total_fish : ℕ) : 
  jerk_fish = 144 → total_fish = 432 → 
  (total_fish - jerk_fish) / jerk_fish = 2 := by
sorry

end fish_ratio_l150_15034


namespace gcd_630_945_l150_15070

theorem gcd_630_945 : Nat.gcd 630 945 = 315 := by
  sorry

end gcd_630_945_l150_15070
