import Mathlib

namespace NUMINAMATH_CALUDE_gas_cost_calculation_l69_6919

theorem gas_cost_calculation (total_cost : ℚ) : 
  (total_cost / 5 - 15 = total_cost / 7) → 
  total_cost = 262.5 := by
sorry

end NUMINAMATH_CALUDE_gas_cost_calculation_l69_6919


namespace NUMINAMATH_CALUDE_painting_price_decrease_l69_6951

theorem painting_price_decrease (original_price : ℝ) (h_positive : original_price > 0) :
  let first_year_price := original_price * 1.25
  let final_price := original_price * 1.0625
  let second_year_decrease := (first_year_price - final_price) / first_year_price
  second_year_decrease = 0.15 := by sorry

end NUMINAMATH_CALUDE_painting_price_decrease_l69_6951


namespace NUMINAMATH_CALUDE_cow_field_difference_l69_6929

theorem cow_field_difference (total : ℕ) (males : ℕ) (females : ℕ) : 
  total = 300 →
  females = 2 * males →
  total = males + females →
  (females / 2 : ℕ) - (males / 2 : ℕ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_cow_field_difference_l69_6929


namespace NUMINAMATH_CALUDE_goldies_hourly_rate_l69_6968

/-- Goldie's pet-sitting earnings problem -/
theorem goldies_hourly_rate (hours_last_week hours_this_week total_earnings : ℚ) 
  (h1 : hours_last_week = 20)
  (h2 : hours_this_week = 30)
  (h3 : total_earnings = 250) :
  total_earnings / (hours_last_week + hours_this_week) = 5 := by
  sorry

end NUMINAMATH_CALUDE_goldies_hourly_rate_l69_6968


namespace NUMINAMATH_CALUDE_circle_in_rectangle_ratio_l69_6950

theorem circle_in_rectangle_ratio (r s : ℝ) (h1 : r > 0) (h2 : s > 0) : 
  (π * r^2 = 2 * r * s - π * r^2) → (s / (2 * r) = π / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_in_rectangle_ratio_l69_6950


namespace NUMINAMATH_CALUDE_fraction_identity_l69_6915

theorem fraction_identity (m : ℕ) (hm : m > 0) :
  (1 : ℚ) / (m * (m + 1)) = 1 / m - 1 / (m + 1) ∧
  (1 : ℚ) / (6 * 7) = 1 / 6 - 1 / 7 ∧
  ∃ (x : ℚ), x = 4 ∧ 1 / ((x - 1) * (x - 2)) + 1 / (x * (x - 1)) = 1 / x :=
by sorry

end NUMINAMATH_CALUDE_fraction_identity_l69_6915


namespace NUMINAMATH_CALUDE_total_onions_grown_l69_6908

theorem total_onions_grown (sara_onions sally_onions fred_onions : ℕ)
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : fred_onions = 9) :
  sara_onions + sally_onions + fred_onions = 18 := by
sorry

end NUMINAMATH_CALUDE_total_onions_grown_l69_6908


namespace NUMINAMATH_CALUDE_oblique_square_area_theorem_main_theorem_l69_6916

/-- Represents a square in an oblique projection --/
structure ObliqueSquare where
  side : ℝ
  projectedSide : ℝ

/-- The area of the original square given its oblique projection --/
def originalArea (s : ObliqueSquare) : ℝ := s.side * s.side

/-- Theorem stating that for a square with a projected side of 4 units,
    the area of the original square can be either 16 or 64 --/
theorem oblique_square_area_theorem (s : ObliqueSquare) 
  (h : s.projectedSide = 4) :
  originalArea s = 16 ∨ originalArea s = 64 := by
  sorry

/-- Main theorem combining the above results --/
theorem main_theorem : 
  ∃ (s1 s2 : ObliqueSquare), 
    s1.projectedSide = 4 ∧ 
    s2.projectedSide = 4 ∧ 
    originalArea s1 = 16 ∧ 
    originalArea s2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_oblique_square_area_theorem_main_theorem_l69_6916


namespace NUMINAMATH_CALUDE_weed_ratio_l69_6925

/-- Represents the number of weeds pulled on each day --/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Defines the conditions of the weed-pulling problem --/
def weed_problem (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.thursday = w.wednesday / 5 ∧
  w.friday = w.thursday - 10 ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- The theorem to be proved --/
theorem weed_ratio (w : WeedCount) (h : weed_problem w) : 
  w.wednesday = 3 * w.tuesday :=
sorry

end NUMINAMATH_CALUDE_weed_ratio_l69_6925


namespace NUMINAMATH_CALUDE_peter_marbles_l69_6932

/-- The number of marbles Peter lost -/
def lost_marbles : ℕ := 15

/-- The number of marbles Peter currently has -/
def current_marbles : ℕ := 18

/-- The initial number of marbles Peter had -/
def initial_marbles : ℕ := lost_marbles + current_marbles

theorem peter_marbles : initial_marbles = 33 := by
  sorry

end NUMINAMATH_CALUDE_peter_marbles_l69_6932


namespace NUMINAMATH_CALUDE_lily_pad_coverage_l69_6993

/-- Represents the size of the lily pad patch as a fraction of the lake -/
def LilyPadSize := ℚ

/-- The number of days it takes for the patch to cover the entire lake -/
def TotalDays : ℕ := 37

/-- The fraction of the lake that is covered after a given number of days -/
def coverage (days : ℕ) : LilyPadSize :=
  (1 : ℚ) / (2 ^ (TotalDays - days))

/-- Theorem stating that it takes 36 days to cover three-fourths of the lake -/
theorem lily_pad_coverage :
  coverage 36 = (3 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_lily_pad_coverage_l69_6993


namespace NUMINAMATH_CALUDE_arm_wrestling_streaks_l69_6907

/-- Represents the outcome of a single round of arm wrestling -/
inductive Winner : Type
| Richard : Winner
| Shreyas : Winner

/-- Counts the number of streaks in a list of outcomes -/
def count_streaks (outcomes : List Winner) : Nat :=
  sorry

/-- Generates all possible outcomes for n rounds of arm wrestling -/
def generate_outcomes (n : Nat) : List (List Winner) :=
  sorry

/-- Counts the number of outcomes with more than k streaks in n rounds -/
def count_outcomes_with_more_than_k_streaks (n k : Nat) : Nat :=
  sorry

theorem arm_wrestling_streaks :
  count_outcomes_with_more_than_k_streaks 10 3 = 932 :=
sorry

end NUMINAMATH_CALUDE_arm_wrestling_streaks_l69_6907


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l69_6918

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def lies_on (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_lines_sum (a b c : ℝ) :
  perpendicular (-a/4) (2/5) →
  lies_on 1 c a 4 (-2) →
  lies_on 1 c 2 (-5) b →
  a + b + c = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l69_6918


namespace NUMINAMATH_CALUDE_cylinder_height_l69_6956

/-- The height of a right cylinder with radius 2 feet and surface area 12π square feet is 1 foot. -/
theorem cylinder_height (π : ℝ) (h : ℝ) : 
  (2 * π * 2^2 + 2 * π * 2 * h = 12 * π) → h = 1 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_l69_6956


namespace NUMINAMATH_CALUDE_saras_high_school_basketball_games_l69_6962

theorem saras_high_school_basketball_games 
  (defeated_games won_games total_games : ℕ) : 
  defeated_games = 4 → 
  won_games = 8 → 
  total_games = defeated_games + won_games → 
  total_games = 12 :=
by sorry

end NUMINAMATH_CALUDE_saras_high_school_basketball_games_l69_6962


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l69_6910

theorem triangle_perimeter_bound :
  ∀ (a b c : ℝ),
  a = 7 →
  b ≥ 14 →
  a + b > c →
  a + c > b →
  b + c > a →
  a + b + c < 42 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l69_6910


namespace NUMINAMATH_CALUDE_factor_w4_minus_81_l69_6963

theorem factor_w4_minus_81 (w : ℝ) : w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_w4_minus_81_l69_6963


namespace NUMINAMATH_CALUDE_area_FYG_is_86_4_l69_6948

/-- A trapezoid with the given properties -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  area : ℝ

/-- The area of triangle FYG in the given trapezoid -/
def area_FYG (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the area of triangle FYG is 86.4 square units -/
theorem area_FYG_is_86_4 (t : Trapezoid) 
  (h1 : t.EF = 24)
  (h2 : t.GH = 36)
  (h3 : t.area = 360) :
  area_FYG t = 86.4 := by sorry

end NUMINAMATH_CALUDE_area_FYG_is_86_4_l69_6948


namespace NUMINAMATH_CALUDE_martin_trip_distance_l69_6969

/-- Calculates the total distance traveled during a two-part journey -/
def total_distance (total_time hours_per_half : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  speed1 * hours_per_half + speed2 * hours_per_half

/-- Proves that the total distance traveled in the given conditions is 620 km -/
theorem martin_trip_distance :
  let total_time : ℝ := 8
  let speed1 : ℝ := 70
  let speed2 : ℝ := 85
  let hours_per_half : ℝ := total_time / 2
  total_distance total_time hours_per_half speed1 speed2 = 620 := by
  sorry

#eval total_distance 8 4 70 85

end NUMINAMATH_CALUDE_martin_trip_distance_l69_6969


namespace NUMINAMATH_CALUDE_prob_same_roll_6_7_l69_6985

/-- The probability of rolling a specific number on a fair die with n sides -/
def prob_roll (n : ℕ) : ℚ := 1 / n

/-- The probability of rolling the same number on two dice with sides n and m -/
def prob_same_roll (n m : ℕ) : ℚ := (prob_roll n) * (prob_roll m)

theorem prob_same_roll_6_7 :
  prob_same_roll 6 7 = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_roll_6_7_l69_6985


namespace NUMINAMATH_CALUDE_range_of_2a_plus_c_l69_6983

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

-- State the theorem
theorem range_of_2a_plus_c (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c)
  (h2 : t.b = Real.sqrt 3) :
  Real.sqrt 3 < 2 * t.a + t.c ∧ 2 * t.a + t.c ≤ 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_c_l69_6983


namespace NUMINAMATH_CALUDE_speed_increase_time_reduction_l69_6942

/-- Represents Vanya's speed to school -/
def speed : ℝ := by sorry

/-- Theorem stating the relationship between speed increase and time reduction -/
theorem speed_increase_time_reduction :
  (speed + 2) / speed = 2.5 →
  (speed + 4) / speed = 4 := by sorry

end NUMINAMATH_CALUDE_speed_increase_time_reduction_l69_6942


namespace NUMINAMATH_CALUDE_complement_of_M_l69_6978

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 3, 5}

theorem complement_of_M :
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l69_6978


namespace NUMINAMATH_CALUDE_juice_price_proof_l69_6957

def total_paid : ℚ := 370 / 100
def muffin_price : ℚ := 75 / 100
def muffin_count : ℕ := 3

theorem juice_price_proof :
  total_paid - (muffin_price * muffin_count) = 145 / 100 := by
  sorry

end NUMINAMATH_CALUDE_juice_price_proof_l69_6957


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l69_6952

def vector1 : ℝ × ℝ × ℝ := (3, 4, 5)
def vector2 (m : ℝ) : ℝ × ℝ × ℝ := (2, m, 3)
def vector3 (m : ℝ) : ℝ × ℝ × ℝ := (2, 3, m)

def volume (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  abs (a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1))

theorem parallelepiped_volume (m : ℝ) :
  m > 0 →
  volume vector1 (vector2 m) (vector3 m) = 20 →
  m = (9 + Real.sqrt 249) / 6 :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l69_6952


namespace NUMINAMATH_CALUDE_total_students_is_240_l69_6967

/-- The number of students from Know It All High School -/
def know_it_all_students : ℕ := 50

/-- The number of students from Karen High School -/
def karen_high_students : ℕ := (3 * know_it_all_students) / 5

/-- The combined number of students from Know It All High School and Karen High School -/
def combined_students : ℕ := know_it_all_students + karen_high_students

/-- The number of students from Novel Corona High School -/
def novel_corona_students : ℕ := 2 * combined_students

/-- The total number of students at the competition -/
def total_students : ℕ := know_it_all_students + karen_high_students + novel_corona_students

/-- Theorem stating that the total number of students at the competition is 240 -/
theorem total_students_is_240 : total_students = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_240_l69_6967


namespace NUMINAMATH_CALUDE_tangent_line_property_l69_6902

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^(3/2) - Real.log x - 2/3

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := (1/x) * (x^(3/2) - 1)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := |f x + f' x|

-- State the theorem
theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) (h₄ : g x₁ = g x₂) : x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_property_l69_6902


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_special_property_l69_6996

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  n % 7 = 1 ∧
  (10 * (n % 10) + n / 10) % 7 = 1

theorem two_digit_numbers_with_special_property :
  {n : ℕ | is_valid_number n} = {22, 29, 92, 99} :=
by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_special_property_l69_6996


namespace NUMINAMATH_CALUDE_locker_count_proof_l69_6998

/-- The cost of each digit in cents -/
def digit_cost : ℚ := 3

/-- The total cost of labeling all lockers in dollars -/
def total_cost : ℚ := 771.90

/-- The number of lockers -/
def num_lockers : ℕ := 6369

/-- The cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let one_digit := (min n 9 : ℚ) * digit_cost / 100
  let two_digit := (min n 99 - min n 9 : ℚ) * 2 * digit_cost / 100
  let three_digit := (min n 999 - min n 99 : ℚ) * 3 * digit_cost / 100
  let four_digit := (min n 9999 - min n 999 : ℚ) * 4 * digit_cost / 100
  let five_digit := (n - min n 9999 : ℚ) * 5 * digit_cost / 100
  one_digit + two_digit + three_digit + four_digit + five_digit

theorem locker_count_proof :
  labeling_cost num_lockers = total_cost := by
  sorry

end NUMINAMATH_CALUDE_locker_count_proof_l69_6998


namespace NUMINAMATH_CALUDE_parallelepiped_sphere_properties_l69_6971

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Check if a sphere touches an edge of the parallelepiped -/
def touchesEdge (s : Sphere) (p1 p2 : Point3D) : Prop := sorry

/-- Check if a point is on an edge of the parallelepiped -/
def onEdge (p : Point3D) (p1 p2 : Point3D) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculate the volume of a parallelepiped -/
def volume (p : Parallelepiped) : ℝ := sorry

theorem parallelepiped_sphere_properties 
  (p : Parallelepiped) (s : Sphere) (K : Point3D) :
  (distance p.A p.A1 = distance p.B p.C) →  -- A₁A perpendicular to ABCD face
  (touchesEdge s p.B p.B1) →
  (touchesEdge s p.B1 p.C1) →
  (touchesEdge s p.C1 p.C) →
  (touchesEdge s p.C p.B) →
  (touchesEdge s p.C p.D) →
  (touchesEdge s p.A1 p.D1) →
  (onEdge K p.C p.D) →
  (distance p.C K = 4) →
  (distance K p.D = 1) →
  (distance p.A p.A1 = 8) ∧
  (volume p = 256) ∧
  (s.radius = 2 * Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_parallelepiped_sphere_properties_l69_6971


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l69_6934

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2 * y = 6) :
  ∃ (min : ℝ), min = 16 ∧ ∀ (a b : ℝ), a + 2 * b = 6 → 2^a + 4^b ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l69_6934


namespace NUMINAMATH_CALUDE_triangle_side_range_l69_6966

theorem triangle_side_range (a : ℝ) : 
  (∃ (x y z : ℝ), x = 3 ∧ y = 2*a - 1 ∧ z = 4 ∧ 
    x + y > z ∧ x + z > y ∧ y + z > x) ↔ 
  (1 < a ∧ a < 4) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l69_6966


namespace NUMINAMATH_CALUDE_average_of_geometric_sequence_l69_6933

/-- The average of the numbers 5y, 10y, 20y, 40y, and 80y is equal to 31y -/
theorem average_of_geometric_sequence (y : ℝ) : 
  (5*y + 10*y + 20*y + 40*y + 80*y) / 5 = 31*y := by
  sorry

end NUMINAMATH_CALUDE_average_of_geometric_sequence_l69_6933


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l69_6939

/-- Represents a digit in base 9 --/
def Digit := Fin 9

/-- Represents the cryptarithm LAKE + KALE + LEAK = KLAE in base 9 --/
def Cryptarithm (L A K E : Digit) : Prop :=
  (L.val + K.val + L.val) % 9 = K.val ∧
  (A.val + A.val + E.val) % 9 = L.val ∧
  (K.val + L.val + A.val) % 9 = A.val ∧
  (E.val + E.val + K.val) % 9 = E.val

/-- All digits are distinct --/
def DistinctDigits (L A K E : Digit) : Prop :=
  L ≠ A ∧ L ≠ K ∧ L ≠ E ∧ A ≠ K ∧ A ≠ E ∧ K ≠ E

theorem cryptarithm_solution :
  ∃ (L A K E : Digit),
    Cryptarithm L A K E ∧
    DistinctDigits L A K E ∧
    L.val = 0 ∧ E.val = 8 ∧ K.val = 4 ∧ (A.val = 1 ∨ A.val = 2 ∨ A.val = 3 ∨ A.val = 5 ∨ A.val = 6 ∨ A.val = 7) :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l69_6939


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l69_6922

def A : Set Int := {-1, 0, 2}
def B : Set Int := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l69_6922


namespace NUMINAMATH_CALUDE_base_two_representation_of_125_l69_6982

/-- Represents a natural number in base 2 as a list of bits (least significant bit first) -/
def BaseTwoRepresentation := List Bool

/-- Converts a natural number to its base 2 representation -/
def toBaseTwoRepresentation (n : ℕ) : BaseTwoRepresentation :=
  sorry

/-- Converts a base 2 representation to its decimal (base 10) value -/
def fromBaseTwoRepresentation (bits : BaseTwoRepresentation) : ℕ :=
  sorry

theorem base_two_representation_of_125 :
  toBaseTwoRepresentation 125 = [true, false, true, true, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_base_two_representation_of_125_l69_6982


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l69_6958

/-- Given a line equation x/a + y/b = 1 where a > 0 and b > 0, 
    and the line passes through the point (2, 3),
    prove that the minimum value of 2a + b is 7 + 4√3 -/
theorem min_value_of_2a_plus_b (a b : ℝ) : 
  a > 0 → b > 0 → 2 / a + 3 / b = 1 → 
  ∀ x y, x / a + y / b = 1 → 
  (2 * a + b) ≥ 7 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l69_6958


namespace NUMINAMATH_CALUDE_car_distance_18_hours_l69_6906

/-- Calculates the total distance traveled by a car with increasing speed -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  let finalSpeed := initialSpeed + speedIncrease * (hours - 1)
  hours * (initialSpeed + finalSpeed) / 2

/-- Theorem stating the total distance traveled by the car in 18 hours -/
theorem car_distance_18_hours :
  totalDistance 30 5 18 = 1305 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_18_hours_l69_6906


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_three_half_l69_6937

theorem sin_cos_sum_equals_sqrt_three_half : 
  Real.sin (10 * Real.pi / 180) * Real.cos (50 * Real.pi / 180) + 
  Real.cos (10 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_three_half_l69_6937


namespace NUMINAMATH_CALUDE_certain_number_proof_l69_6972

theorem certain_number_proof : ∃ n : ℝ, n = 36 ∧ n + 3 * 4.0 = 48 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l69_6972


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l69_6927

theorem cube_root_equation_solution :
  ∃! x : ℝ, (10 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l69_6927


namespace NUMINAMATH_CALUDE_area_equality_in_divided_triangle_l69_6965

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Represents a triangle with its three vertices -/
structure Triangle :=
  (A B C : Point)

/-- Given a triangle and a ratio, returns a point on one of its sides -/
def pointOnSide (T : Triangle) (ratio : ℝ) (side : Fin 3) : Point := sorry

theorem area_equality_in_divided_triangle (ABC : Triangle) :
  let D := pointOnSide ABC (1/3) 0
  let E := pointOnSide ABC (1/3) 1
  let F := pointOnSide ABC (1/3) 2
  let G := pointOnSide (Triangle.mk D E F) (1/2) 0
  let H := pointOnSide (Triangle.mk D E F) (1/2) 1
  let I := pointOnSide (Triangle.mk D E F) (1/2) 2
  triangleArea D A G + triangleArea E B H + triangleArea F C I = triangleArea G H I :=
by sorry

end NUMINAMATH_CALUDE_area_equality_in_divided_triangle_l69_6965


namespace NUMINAMATH_CALUDE_limit_expression_equals_six_l69_6946

theorem limit_expression_equals_six :
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| ∧ |h| < δ →
    |((3 + h)^2 - 3^2) / h - 6| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_expression_equals_six_l69_6946


namespace NUMINAMATH_CALUDE_max_value_of_f_l69_6980

-- Define the function f(x) = √(x-3) + √(6-x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 3) + Real.sqrt (6 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 6 ∧
  f x = Real.sqrt 6 ∧
  ∀ (y : ℝ), 3 ≤ y ∧ y ≤ 6 → f y ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l69_6980


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l69_6945

/-- Two planes are mutually perpendicular -/
def mutually_perpendicular (α β : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (n : Line) (β : Plane) : Prop := sorry

/-- Two planes intersect at a line -/
def planes_intersect_at_line (α β : Plane) (l : Line) : Prop := sorry

/-- A line is perpendicular to another line -/
def line_perpendicular_to_line (n l : Line) : Prop := sorry

theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (l m n : Line)
  (h1 : mutually_perpendicular α β)
  (h2 : planes_intersect_at_line α β l)
  (h3 : line_parallel_to_plane m α)
  (h4 : line_perpendicular_to_plane n β) :
  line_perpendicular_to_line n l :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l69_6945


namespace NUMINAMATH_CALUDE_floor_of_e_l69_6940

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_l69_6940


namespace NUMINAMATH_CALUDE_apple_sorting_probability_l69_6955

def ratio_large_to_small : ℚ := 9 / 1
def prob_large_to_small : ℚ := 5 / 100
def prob_small_to_large : ℚ := 2 / 100

theorem apple_sorting_probability : 
  let total_apples := ratio_large_to_small + 1
  let prob_large := ratio_large_to_small / total_apples
  let prob_small := 1 / total_apples
  let prob_large_sorted_large := 1 - prob_large_to_small
  let prob_small_sorted_large := prob_small_to_large
  let prob_sorted_large := prob_large * prob_large_sorted_large + prob_small * prob_small_sorted_large
  let prob_large_and_sorted_large := prob_large * prob_large_sorted_large
  (prob_large_and_sorted_large / prob_sorted_large) = 855 / 857 :=
by sorry

end NUMINAMATH_CALUDE_apple_sorting_probability_l69_6955


namespace NUMINAMATH_CALUDE_susy_initial_followers_l69_6986

/-- Represents the number of followers gained by a student over three weeks -/
structure FollowerGain where
  week1 : ℕ
  week2 : ℕ
  week3 : ℕ

/-- Represents a student with their school size and follower information -/
structure Student where
  schoolSize : ℕ
  initialFollowers : ℕ
  followerGain : FollowerGain

def totalFollowersAfterThreeWeeks (student : Student) : ℕ :=
  student.initialFollowers + student.followerGain.week1 + student.followerGain.week2 + student.followerGain.week3

theorem susy_initial_followers
  (susy : Student)
  (sarah : Student)
  (h1 : susy.schoolSize = 800)
  (h2 : sarah.schoolSize = 300)
  (h3 : susy.followerGain.week1 = 40)
  (h4 : susy.followerGain.week2 = susy.followerGain.week1 / 2)
  (h5 : susy.followerGain.week3 = susy.followerGain.week2 / 2)
  (h6 : sarah.initialFollowers = 50)
  (h7 : max (totalFollowersAfterThreeWeeks susy) (totalFollowersAfterThreeWeeks sarah) = 180) :
  susy.initialFollowers = 110 := by
  sorry

end NUMINAMATH_CALUDE_susy_initial_followers_l69_6986


namespace NUMINAMATH_CALUDE_gcd_m5_plus_125_m_plus_3_l69_6904

theorem gcd_m5_plus_125_m_plus_3 (m : ℕ) (h : m > 16) :
  Nat.gcd (m^5 + 5^3) (m + 3) = if (m + 3) % 27 ≠ 0 then 1 else Nat.gcd 27 (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_gcd_m5_plus_125_m_plus_3_l69_6904


namespace NUMINAMATH_CALUDE_illuminated_part_depends_on_position_l69_6928

/-- Represents a right circular cone on a plane -/
structure Cone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone

/-- Represents the position of a light source -/
structure LightSource where
  H : ℝ  -- distance from the plane
  l : ℝ  -- distance from the height of the cone

/-- Represents the illuminated part of a circle -/
structure IlluminatedPart where
  angle : ℝ  -- angle of the illuminated arc

/-- Calculates the illuminated part of a circle with radius R on the plane -/
noncomputable def calculateIlluminatedPart (cone : Cone) (light : LightSource) (R : ℝ) : IlluminatedPart :=
  sorry

/-- Theorem stating that the illuminated part can be determined by the relative position of the light source -/
theorem illuminated_part_depends_on_position (cone : Cone) (light : LightSource) (R : ℝ) :
  ∃ (ip : IlluminatedPart), ip = calculateIlluminatedPart cone light R ∧
  (light.H > cone.h ∨ light.H = cone.h ∨ light.H < cone.h) :=
  sorry

end NUMINAMATH_CALUDE_illuminated_part_depends_on_position_l69_6928


namespace NUMINAMATH_CALUDE_num_divisors_360_eq_24_l69_6994

/-- The number of positive divisors of 360 -/
def num_divisors_360 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 360 is 24 -/
theorem num_divisors_360_eq_24 : num_divisors_360 = 24 := by sorry

end NUMINAMATH_CALUDE_num_divisors_360_eq_24_l69_6994


namespace NUMINAMATH_CALUDE_highway_extension_remaining_miles_l69_6984

/-- Proves that given the highway extension conditions, 250 miles still need to be added -/
theorem highway_extension_remaining_miles 
  (current_length : ℝ) 
  (final_length : ℝ) 
  (first_day_miles : ℝ) 
  (second_day_multiplier : ℝ) :
  current_length = 200 →
  final_length = 650 →
  first_day_miles = 50 →
  second_day_multiplier = 3 →
  final_length - current_length - first_day_miles - (second_day_multiplier * first_day_miles) = 250 := by
  sorry

#check highway_extension_remaining_miles

end NUMINAMATH_CALUDE_highway_extension_remaining_miles_l69_6984


namespace NUMINAMATH_CALUDE_single_circle_percentage_l69_6913

/-- The number of children participating in the game -/
def n : ℕ := 10

/-- Calculates the double factorial of a natural number -/
def double_factorial (k : ℕ) : ℕ :=
  if k ≤ 1 then 1 else k * double_factorial (k - 2)

/-- Calculates the number of configurations where n children form a single circle -/
def single_circle_configs (n : ℕ) : ℕ := double_factorial (2 * n - 2)

/-- Calculates the total number of possible configurations for n children -/
def total_configs (n : ℕ) : ℕ := 387099936  -- This is the precomputed value for n = 10

/-- The main theorem to be proved -/
theorem single_circle_percentage :
  (single_circle_configs n : ℚ) / (total_configs n) = 12 / 25 := by
  sorry

#eval (single_circle_configs n : ℚ) / (total_configs n)

end NUMINAMATH_CALUDE_single_circle_percentage_l69_6913


namespace NUMINAMATH_CALUDE_percentage_error_division_vs_multiplication_l69_6924

theorem percentage_error_division_vs_multiplication (x : ℝ) (h : x > 0) : 
  (|4 * x - x / 4| / (4 * x)) * 100 = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_error_division_vs_multiplication_l69_6924


namespace NUMINAMATH_CALUDE_fish_added_l69_6959

theorem fish_added (initial_fish final_fish : ℕ) (h1 : initial_fish = 10) (h2 : final_fish = 13) :
  final_fish - initial_fish = 3 := by
  sorry

end NUMINAMATH_CALUDE_fish_added_l69_6959


namespace NUMINAMATH_CALUDE_john_slurpees_l69_6921

def slurpee_problem (money_given : ℕ) (slurpee_cost : ℕ) (change : ℕ) : ℕ :=
  (money_given - change) / slurpee_cost

theorem john_slurpees :
  slurpee_problem 20 2 8 = 6 :=
by sorry

end NUMINAMATH_CALUDE_john_slurpees_l69_6921


namespace NUMINAMATH_CALUDE_area_of_large_rectangle_l69_6903

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: Area of large rectangle formed by three identical smaller rectangles -/
theorem area_of_large_rectangle (small_rect : Rectangle) 
    (h1 : small_rect.width = 7)
    (h2 : small_rect.height ≥ small_rect.width) : 
  (Rectangle.area { width := 3 * small_rect.height, height := small_rect.width }) = 294 := by
  sorry

#check area_of_large_rectangle

end NUMINAMATH_CALUDE_area_of_large_rectangle_l69_6903


namespace NUMINAMATH_CALUDE_smiths_age_problem_l69_6975

/-- Represents a 4-digit number in the form abba -/
def mirroredNumber (a b : Nat) : Nat :=
  1000 * a + 100 * b + 10 * b + a

theorem smiths_age_problem :
  ∃! n : Nat,
    59 < n ∧ n < 100 ∧
    (∃ b : Nat, b < 10 ∧ (mirroredNumber (n / 10) b) % 7 = 0) ∧
    n = 67 := by
  sorry

end NUMINAMATH_CALUDE_smiths_age_problem_l69_6975


namespace NUMINAMATH_CALUDE_smallest_d_value_l69_6944

def σ (v : Fin 4 → ℕ) : Finset (Fin 4 → ℕ) := sorry

theorem smallest_d_value (a b c d : ℕ) :
  0 < a → a < b → b < c → c < d →
  (∃ (s : ℕ), ∃ (v₁ v₂ v₃ : Fin 4 → ℕ),
    v₁ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₂ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₃ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧
    (∀ i : Fin 4, v₁ i + v₂ i + v₃ i = s)) →
  d ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_value_l69_6944


namespace NUMINAMATH_CALUDE_third_term_is_16_l69_6977

/-- Geometric sequence with common ratio 2 and sum of first 4 terms equal to 60 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 + a 4 = 60)

/-- The third term of the geometric sequence is 16 -/
theorem third_term_is_16 (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_16_l69_6977


namespace NUMINAMATH_CALUDE_f_properties_l69_6999

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties :
  -1 < a → a < 0 → x > 0 →
  (∃ (max_val min_val : ℝ),
    (a = -1/2 → 
      (∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y ≤ max_val) ∧
      (∃ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y = max_val) ∧
      max_val = 1/2 + (Real.exp 1)^2/4) ∧
    (a = -1/2 → 
      (∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y ≥ min_val) ∧
      (∃ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y = min_val) ∧
      min_val = 5/4)) ∧
  (∀ y z, 0 < y → y < Real.sqrt (-a/(a+1)) → z ≥ Real.sqrt (-a/(a+1)) → 
    f a y ≥ f a (Real.sqrt (-a/(a+1))) ∧ f a z ≥ f a (Real.sqrt (-a/(a+1)))) ∧
  (∀ y, y > 0 → f a y > 1 + a/2 * Real.log (-a) ↔ 1/Real.exp 1 - 1 < a) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l69_6999


namespace NUMINAMATH_CALUDE_great_fourteen_soccer_league_games_l69_6900

theorem great_fourteen_soccer_league_games (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : 
  teams_per_division = 7 →
  intra_division_games = 3 →
  inter_division_games = 1 →
  (teams_per_division * (
    (teams_per_division - 1) * intra_division_games + 
    teams_per_division * inter_division_games
  )) / 2 = 175 := by
  sorry

end NUMINAMATH_CALUDE_great_fourteen_soccer_league_games_l69_6900


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l69_6914

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l69_6914


namespace NUMINAMATH_CALUDE_delta_y_over_delta_x_l69_6926

/-- Given a function f(x) = 2x² + 1, prove that Δy/Δx = 4 + 2Δx for points P(1, 3) and Q(1 + Δx, 3 + Δy) -/
theorem delta_y_over_delta_x (Δx : ℝ) (Δy : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 1
  Δy = f (1 + Δx) - f 1 →
  Δx ≠ 0 →
  Δy / Δx = 4 + 2 * Δx := by
sorry

end NUMINAMATH_CALUDE_delta_y_over_delta_x_l69_6926


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l69_6989

/-- Given a triangle DEF where the measure of angle D is three times the measure of angle F,
    and angle F measures 18°, prove that the measure of angle E is 108°. -/
theorem angle_measure_in_triangle (D E F : ℝ) (h1 : D = 3 * F) (h2 : F = 18) :
  E = 108 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l69_6989


namespace NUMINAMATH_CALUDE_order_of_a_b_c_l69_6954

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_a_b_c : a > b ∧ a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_a_b_c_l69_6954


namespace NUMINAMATH_CALUDE_rearrangement_methods_l69_6936

theorem rearrangement_methods (n m k : ℕ) (hn : n = 8) (hm : m = 4) (hk : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_methods_l69_6936


namespace NUMINAMATH_CALUDE_point_above_line_l69_6991

/-- A point (x, y) is above a line ax + by + c = 0 if ax + by + c < 0 -/
def IsAboveLine (x y a b c : ℝ) : Prop := a * x + b * y + c < 0

theorem point_above_line (t : ℝ) :
  IsAboveLine (-2) t 1 (-2) 4 → t > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_above_line_l69_6991


namespace NUMINAMATH_CALUDE_parallelogram_construction_l69_6949

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the predicates
variable (lies_on : Point → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (is_center : Point → Point → Point → Point → Point → Prop)

-- State the theorem
theorem parallelogram_construction
  (l₁ l₂ l₃ l₄ : Line)
  (O : Point)
  (not_parallel : ¬ parallel l₁ l₂ ∧ ¬ parallel l₁ l₃ ∧ ¬ parallel l₁ l₄ ∧
                  ¬ parallel l₂ l₃ ∧ ¬ parallel l₂ l₄ ∧ ¬ parallel l₃ l₄)
  (O_not_on_lines : ¬ lies_on O l₁ ∧ ¬ lies_on O l₂ ∧ ¬ lies_on O l₃ ∧ ¬ lies_on O l₄) :
  ∃ (A B C D : Point),
    lies_on A l₁ ∧ lies_on B l₂ ∧ lies_on C l₃ ∧ lies_on D l₄ ∧
    is_center O A B C D :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_construction_l69_6949


namespace NUMINAMATH_CALUDE_negation_P_necessary_not_sufficient_for_negation_Q_l69_6930

def P (x : ℝ) : Prop := |x - 2| ≥ 1

def Q (x : ℝ) : Prop := x^2 - 3*x + 2 ≥ 0

theorem negation_P_necessary_not_sufficient_for_negation_Q :
  (∀ x, ¬(Q x) → ¬(P x)) ∧ 
  (∃ x, ¬(P x) ∧ Q x) :=
by sorry

end NUMINAMATH_CALUDE_negation_P_necessary_not_sufficient_for_negation_Q_l69_6930


namespace NUMINAMATH_CALUDE_prime_factorization_of_large_number_l69_6992

theorem prime_factorization_of_large_number :
  1007021035035021007001 = 7^7 * 11^7 * 13^7 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_of_large_number_l69_6992


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l69_6961

/-- Proves that a car's initial fuel efficiency is 24 miles per gallon given specific conditions -/
theorem car_fuel_efficiency 
  (initial_efficiency : ℝ) 
  (improvement_factor : ℝ) 
  (tank_capacity : ℝ) 
  (additional_miles : ℝ) 
  (h1 : improvement_factor = 4/3) 
  (h2 : tank_capacity = 12) 
  (h3 : additional_miles = 96) 
  (h4 : tank_capacity * initial_efficiency * improvement_factor - 
        tank_capacity * initial_efficiency = additional_miles) : 
  initial_efficiency = 24 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l69_6961


namespace NUMINAMATH_CALUDE_hope_project_protractors_l69_6976

theorem hope_project_protractors :
  ∀ (x y z : ℕ),
  x > 31 ∧ z > 33 ∧
  10 * x + 15 * y + 20 * z = 1710 ∧
  8 * x + 2 * y + 8 * z = 664 →
  6 * x + 7 * y + 5 * z = 680 :=
by sorry

end NUMINAMATH_CALUDE_hope_project_protractors_l69_6976


namespace NUMINAMATH_CALUDE_eight_digit_divisibility_l69_6912

theorem eight_digit_divisibility (a b : Nat) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0) : 
  ∃ k : Nat, (a * 10 + b) * 1010101 = 101 * k := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_divisibility_l69_6912


namespace NUMINAMATH_CALUDE_tetrahedron_count_is_twelve_l69_6909

/-- A regular triangular prism -/
structure RegularTriangularPrism where
  vertices : Finset (Fin 6)
  vertex_count : vertices.card = 6

/-- The number of ways to choose 4 vertices from 6 -/
def choose_four (prism : RegularTriangularPrism) : Nat :=
  Nat.choose 6 4

/-- The number of cases where 4 chosen points are coplanar -/
def coplanar_cases : Nat := 3

/-- The number of tetrahedrons that can be formed -/
def tetrahedron_count (prism : RegularTriangularPrism) : Nat :=
  choose_four prism - coplanar_cases

/-- Theorem: The number of tetrahedrons is 12 -/
theorem tetrahedron_count_is_twelve (prism : RegularTriangularPrism) :
  tetrahedron_count prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_count_is_twelve_l69_6909


namespace NUMINAMATH_CALUDE_museum_visit_arrangements_l69_6990

theorem museum_visit_arrangements (n m : ℕ) (hn : n = 6) (hm : m = 6) : 
  (Nat.choose n 2) * (m - 1) ^ (n - 2) = 15 * 625 := by
  sorry

end NUMINAMATH_CALUDE_museum_visit_arrangements_l69_6990


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l69_6935

theorem algebraic_expression_equality (x : ℝ) (h : x^2 + 3*x - 5 = 2) : 
  2*x^2 + 6*x - 3 = 11 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l69_6935


namespace NUMINAMATH_CALUDE_binomial_ratio_equals_one_l69_6917

-- Define the binomial coefficient for real numbers
noncomputable def binomial (r : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else (r * binomial (r - 1) (k - 1)) / k

-- State the theorem
theorem binomial_ratio_equals_one :
  (binomial (1/2 : ℝ) 1000 * 4^1000) / binomial 2000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_ratio_equals_one_l69_6917


namespace NUMINAMATH_CALUDE_parallel_planes_from_skew_lines_l69_6981

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_from_skew_lines 
  (α β : Plane) (l m : Line) : 
  α ≠ β →
  skew l m →
  parallel_line_plane l α →
  parallel_line_plane m α →
  parallel_line_plane l β →
  parallel_line_plane m β →
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_skew_lines_l69_6981


namespace NUMINAMATH_CALUDE_square_remainder_mod_nine_l69_6979

theorem square_remainder_mod_nine (N : ℤ) : 
  (N % 9 = 2 ∨ N % 9 = 7) → (N^2 % 9 = 4) := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_mod_nine_l69_6979


namespace NUMINAMATH_CALUDE_expression_evaluation_l69_6973

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -2) :
  5 * a * (b - 2) + 2 * a * (2 - b) = -24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l69_6973


namespace NUMINAMATH_CALUDE_triangle_proof_l69_6941

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.BC = 7 ∧ t.AB = 3 ∧ (Real.sin t.C) / (Real.sin t.B) = 3/5

-- Theorem statement
theorem triangle_proof (t : Triangle) (h : triangle_properties t) :
  t.AC = 5 ∧ t.A = Real.pi * 2/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l69_6941


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l69_6938

/-- A linear function with slope m and y-intercept b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determine if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I => ∃ x y, x > 0 ∧ y > 0 ∧ y = f.m * x + f.b
  | Quadrant.II => ∃ x y, x < 0 ∧ y > 0 ∧ y = f.m * x + f.b
  | Quadrant.III => ∃ x y, x < 0 ∧ y < 0 ∧ y = f.m * x + f.b
  | Quadrant.IV => ∃ x y, x > 0 ∧ y < 0 ∧ y = f.m * x + f.b

theorem linear_function_not_in_quadrant_III (f : LinearFunction)
  (h1 : f.m < 0)
  (h2 : f.b > 0) :
  ¬(passesThrough f Quadrant.III) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l69_6938


namespace NUMINAMATH_CALUDE_product_inequality_l69_6911

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l69_6911


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l69_6920

theorem sufficient_not_necessary_condition (x : ℝ) (h : x > 0) :
  (x + 1 / x ≥ 2) ∧ (∃ a : ℝ, a > 1 ∧ ∀ y : ℝ, y > 0 → y + a / y ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l69_6920


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l69_6923

theorem polynomial_division_remainder (x : ℂ) : 
  (x^75 + x^60 + x^45 + x^30 + x^15 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l69_6923


namespace NUMINAMATH_CALUDE_sams_age_l69_6901

theorem sams_age (sam drew : ℕ) 
  (h1 : sam + drew = 54)
  (h2 : sam = drew / 2) :
  sam = 18 := by
sorry

end NUMINAMATH_CALUDE_sams_age_l69_6901


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l69_6905

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (fun acc d => acc * 10 + d) 0

theorem unique_five_digit_number : ∃! n : ℕ, 
  is_five_digit n ∧ 
  (∃ pos : Fin 5, n + remove_digit n pos = 54321) :=
by
  use 49383
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l69_6905


namespace NUMINAMATH_CALUDE_bus_speed_calculation_l69_6964

/-- Proves that a bus stopping for 12 minutes per hour with an average speed of 40 km/hr including stoppages has an average speed of 50 km/hr excluding stoppages. -/
theorem bus_speed_calculation (stop_time : ℝ) (avg_speed_with_stops : ℝ) :
  stop_time = 12 →
  avg_speed_with_stops = 40 →
  let moving_time : ℝ := 60 - stop_time
  let speed_ratio : ℝ := moving_time / 60
  (speed_ratio * (60 / moving_time) * avg_speed_with_stops) = 50 := by
  sorry

#check bus_speed_calculation

end NUMINAMATH_CALUDE_bus_speed_calculation_l69_6964


namespace NUMINAMATH_CALUDE_team_winning_percentage_l69_6943

theorem team_winning_percentage 
  (first_games : ℕ) 
  (total_games : ℕ) 
  (first_win_rate : ℚ) 
  (remaining_win_rate : ℚ) 
  (h1 : first_games = 30)
  (h2 : total_games = 60)
  (h3 : first_win_rate = 2/5)
  (h4 : remaining_win_rate = 4/5) : 
  (first_win_rate * first_games + remaining_win_rate * (total_games - first_games)) / total_games = 3/5 := by
sorry

end NUMINAMATH_CALUDE_team_winning_percentage_l69_6943


namespace NUMINAMATH_CALUDE_counterfeit_coin_identifiable_l69_6988

/-- Represents the type of coin -/
inductive CoinType
| Gold
| Silver

/-- Represents a coin with its type and whether it's counterfeit -/
structure Coin where
  type : CoinType
  isCounterfeit : Bool

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal
| LeftHeavier
| RightHeavier

/-- Represents a group of coins -/
def CoinGroup := List Coin

/-- Represents a weighing action -/
def Weighing := CoinGroup → CoinGroup → WeighingResult

/-- The total number of coins -/
def totalCoins : Nat := 27

/-- The number of gold coins -/
def goldCoins : Nat := 13

/-- The number of silver coins -/
def silverCoins : Nat := 14

/-- The maximum number of weighings allowed -/
def maxWeighings : Nat := 3

/-- Axiom: There is exactly one counterfeit coin -/
axiom one_counterfeit (coins : List Coin) : 
  coins.length = totalCoins → ∃! c, c ∈ coins ∧ c.isCounterfeit

/-- Axiom: Counterfeit gold coin is lighter than real gold coins -/
axiom counterfeit_gold_lighter (w : Weighing) (c1 c2 : Coin) :
  c1.type = CoinType.Gold ∧ c2.type = CoinType.Gold ∧ c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.RightHeavier

/-- Axiom: Counterfeit silver coin is heavier than real silver coins -/
axiom counterfeit_silver_heavier (w : Weighing) (c1 c2 : Coin) :
  c1.type = CoinType.Silver ∧ c2.type = CoinType.Silver ∧ c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.LeftHeavier

/-- Axiom: Real coins of the same type have equal weight -/
axiom real_coins_equal_weight (w : Weighing) (c1 c2 : Coin) :
  c1.type = c2.type ∧ ¬c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.Equal

/-- The main theorem: It's possible to identify the counterfeit coin in at most three weighings -/
theorem counterfeit_coin_identifiable (coins : List Coin) (w : Weighing) :
  coins.length = totalCoins →
  ∃ (strategy : List (CoinGroup × CoinGroup)), 
    strategy.length ≤ maxWeighings ∧
    ∃ (counterfeit : Coin), counterfeit ∈ coins ∧ counterfeit.isCounterfeit ∧
    ∀ (c : Coin), c ∈ coins ∧ c.isCounterfeit → c = counterfeit :=
  sorry


end NUMINAMATH_CALUDE_counterfeit_coin_identifiable_l69_6988


namespace NUMINAMATH_CALUDE_half_liar_day_determination_l69_6931

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the half-liar's statement type
structure Statement where
  yesterday : Day
  tomorrow : Day

-- Define the function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Define the function to get the previous day
def prevDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Define the theorem
theorem half_liar_day_determination
  (statement_week_ago : Statement)
  (statement_today : Statement)
  (h1 : statement_week_ago.yesterday = Day.Wednesday ∧ statement_week_ago.tomorrow = Day.Thursday)
  (h2 : statement_today.yesterday = Day.Friday ∧ statement_today.tomorrow = Day.Sunday)
  (h3 : ∀ (d : Day), nextDay (nextDay (nextDay (nextDay (nextDay (nextDay (nextDay d)))))) = d)
  : ∃ (today : Day), today = Day.Saturday :=
by
  sorry


end NUMINAMATH_CALUDE_half_liar_day_determination_l69_6931


namespace NUMINAMATH_CALUDE_jackies_activities_exceed_day_l69_6970

/-- Represents the duration of Jackie's daily activities in hours -/
structure DailyActivities where
  working : ℝ
  exercising : ℝ
  sleeping : ℝ
  commuting : ℝ
  meals : ℝ
  language_classes : ℝ
  phone_calls : ℝ
  reading : ℝ

/-- Theorem stating that Jackie's daily activities exceed 24 hours -/
theorem jackies_activities_exceed_day (activities : DailyActivities) 
  (h1 : activities.working = 8)
  (h2 : activities.exercising = 3)
  (h3 : activities.sleeping = 8)
  (h4 : activities.commuting = 1)
  (h5 : activities.meals = 2)
  (h6 : activities.language_classes = 1.5)
  (h7 : activities.phone_calls = 0.5)
  (h8 : activities.reading = 40 / 60) :
  activities.working + activities.exercising + activities.sleeping + 
  activities.commuting + activities.meals + activities.language_classes + 
  activities.phone_calls + activities.reading > 24 := by
  sorry

#check jackies_activities_exceed_day

end NUMINAMATH_CALUDE_jackies_activities_exceed_day_l69_6970


namespace NUMINAMATH_CALUDE_sum_of_squares_l69_6947

theorem sum_of_squares (x y z : ℝ) 
  (h1 : x^2 - 6*y = 10)
  (h2 : y^2 - 8*z = -18)
  (h3 : z^2 - 10*x = -40) :
  x^2 + y^2 + z^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l69_6947


namespace NUMINAMATH_CALUDE_jean_burglary_charges_l69_6995

/-- Represents the charges and sentences for Jean's case -/
structure CriminalCase where
  arson_counts : ℕ
  burglary_charges : ℕ
  petty_larceny_charges : ℕ
  arson_sentence : ℕ
  burglary_sentence : ℕ
  petty_larceny_sentence : ℕ
  total_sentence : ℕ

/-- Calculates the total sentence for a given criminal case -/
def total_sentence (case : CriminalCase) : ℕ :=
  case.arson_counts * case.arson_sentence +
  case.burglary_charges * case.burglary_sentence +
  case.petty_larceny_charges * case.petty_larceny_sentence

/-- Theorem stating that Jean's case has 2 burglary charges -/
theorem jean_burglary_charges :
  ∃ (case : CriminalCase),
    case.arson_counts = 3 ∧
    case.petty_larceny_charges = 6 * case.burglary_charges ∧
    case.arson_sentence = 36 ∧
    case.burglary_sentence = 18 ∧
    case.petty_larceny_sentence = case.burglary_sentence / 3 ∧
    total_sentence case = 216 ∧
    case.burglary_charges = 2 :=
sorry

end NUMINAMATH_CALUDE_jean_burglary_charges_l69_6995


namespace NUMINAMATH_CALUDE_value_added_to_number_l69_6953

theorem value_added_to_number (n v : ℤ) : n = 9 → 3 * (n + 2) = v + n → v = 24 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_number_l69_6953


namespace NUMINAMATH_CALUDE_males_band_not_orchestra_zero_l69_6974

/-- Represents the membership of students in band and orchestra --/
structure MusicGroups where
  total : ℕ
  females_band : ℕ
  males_band : ℕ
  females_orchestra : ℕ
  males_orchestra : ℕ
  females_both : ℕ

/-- The number of males in the band who are not in the orchestra is 0 --/
theorem males_band_not_orchestra_zero (g : MusicGroups)
  (h1 : g.total = 250)
  (h2 : g.females_band = 120)
  (h3 : g.males_band = 90)
  (h4 : g.females_orchestra = 90)
  (h5 : g.males_orchestra = 120)
  (h6 : g.females_both = 70) :
  g.males_band - (g.males_band + g.males_orchestra - (g.total - (g.females_band + g.females_orchestra - g.females_both))) = 0 := by
  sorry

#check males_band_not_orchestra_zero

end NUMINAMATH_CALUDE_males_band_not_orchestra_zero_l69_6974


namespace NUMINAMATH_CALUDE_forbidden_city_area_scientific_notation_l69_6960

/-- The area of the Forbidden City in square meters -/
def forbidden_city_area : ℝ := 720000

/-- Scientific notation representation of the Forbidden City's area -/
def scientific_notation : ℝ := 7.2 * (10 ^ 5)

theorem forbidden_city_area_scientific_notation :
  forbidden_city_area = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_forbidden_city_area_scientific_notation_l69_6960


namespace NUMINAMATH_CALUDE_binomial_coefficient_1000_1000_l69_6997

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1000_1000_l69_6997


namespace NUMINAMATH_CALUDE_two_by_two_paper_covers_nine_vertices_l69_6987

/-- Represents a square paper on a grid -/
structure SquarePaper where
  side_length : ℕ
  min_vertices_covered : ℕ

/-- Counts the number of vertices covered by a square paper on a grid -/
def count_vertices_covered (paper : SquarePaper) : ℕ :=
  (paper.side_length + 1) ^ 2

/-- Theorem: A 2x2 square paper covering at least 7 vertices covers exactly 9 vertices -/
theorem two_by_two_paper_covers_nine_vertices (paper : SquarePaper)
  (h1 : paper.side_length = 2)
  (h2 : paper.min_vertices_covered ≥ 7) :
  count_vertices_covered paper = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_by_two_paper_covers_nine_vertices_l69_6987
