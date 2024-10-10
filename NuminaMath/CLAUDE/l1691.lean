import Mathlib

namespace subset_property_j_bound_l1691_169103

variable (m n : ℕ+)
variable (A : Finset ℕ)
variable (B : Finset ℕ)
variable (S : Finset (ℕ × ℕ))

def setA : Finset ℕ := Finset.range n
def setB : Finset ℕ := Finset.range m

def property_j (S : Finset (ℕ × ℕ)) : Prop :=
  ∀ (a b x y : ℕ), (a, b) ∈ S → (x, y) ∈ S → (a - x) * (b - y) ≤ 0

theorem subset_property_j_bound :
  A = setA m → B = setB n → S ⊆ A ×ˢ B → property_j S → S.card ≤ m + n - 1 := by
  sorry

end subset_property_j_bound_l1691_169103


namespace exists_motion_with_one_stationary_point_l1691_169160

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A rigid body in 3D space -/
structure RigidBody where
  points : Set Point3D

/-- A motion of a rigid body -/
def Motion := RigidBody → ℝ → RigidBody

/-- A point is stationary under a motion if its position doesn't change over time -/
def IsStationary (p : Point3D) (m : Motion) (b : RigidBody) : Prop :=
  ∀ t : ℝ, p ∈ (m b t).points → p ∈ b.points

/-- A motion has exactly one stationary point -/
def HasExactlyOneStationaryPoint (m : Motion) (b : RigidBody) : Prop :=
  ∃! p : Point3D, IsStationary p m b ∧ p ∈ b.points

/-- Theorem: There exists a motion for a rigid body where exactly one point remains stationary -/
theorem exists_motion_with_one_stationary_point :
  ∃ (b : RigidBody) (m : Motion), HasExactlyOneStationaryPoint m b :=
sorry

end exists_motion_with_one_stationary_point_l1691_169160


namespace sixth_episode_length_is_115_l1691_169189

/-- The length of the sixth episode in a series of six episodes -/
def sixth_episode_length (ep1 ep2 ep3 ep4 ep5 total : ℕ) : ℕ :=
  total - (ep1 + ep2 + ep3 + ep4 + ep5)

/-- Theorem stating the length of the sixth episode -/
theorem sixth_episode_length_is_115 :
  sixth_episode_length 58 62 65 71 79 450 = 115 := by
  sorry

end sixth_episode_length_is_115_l1691_169189


namespace tangent_and_cosine_relations_l1691_169191

theorem tangent_and_cosine_relations (θ : Real) (h : Real.tan θ = 2) :
  (Real.tan (π / 4 - θ) = -1 / 3) ∧ (Real.cos (2 * θ) = -3 / 5) := by
  sorry

end tangent_and_cosine_relations_l1691_169191


namespace ball_bird_intersection_time_l1691_169149

/-- The time at which a ball thrown off a cliff and a bird flying upwards from the base of the cliff are at the same height -/
theorem ball_bird_intersection_time : 
  ∃ t : ℝ, t > 0 ∧ (60 - 9*t - 8*t^2 = 3*t^2 + 4*t) ∧ t = 20/11 := by
  sorry

#check ball_bird_intersection_time

end ball_bird_intersection_time_l1691_169149


namespace percentage_difference_l1691_169117

theorem percentage_difference (x y z : ℝ) : 
  x = 1.25 * y →
  x + y + z = 1110 →
  z = 300 →
  (y - z) / z = 0.2 := by
  sorry

end percentage_difference_l1691_169117


namespace value_congr_digitSum_mod_nine_divisible_by_nine_iff_digitSum_divisible_by_nine_l1691_169186

/-- Represents a non-negative integer as a list of its digits in reverse order -/
def Digits := List Nat

/-- Computes the value of a number from its digits -/
def value (d : Digits) : Nat :=
  d.enum.foldl (fun acc (i, digit) => acc + digit * 10^i) 0

/-- Computes the sum of digits -/
def digitSum (d : Digits) : Nat :=
  d.sum

/-- States that for any number, its value is congruent to its digit sum modulo 9 -/
theorem value_congr_digitSum_mod_nine (d : Digits) :
  value d ≡ digitSum d [MOD 9] := by
  sorry

/-- The main theorem: a number is divisible by 9 iff its digit sum is divisible by 9 -/
theorem divisible_by_nine_iff_digitSum_divisible_by_nine (d : Digits) :
  9 ∣ value d ↔ 9 ∣ digitSum d := by
  sorry

end value_congr_digitSum_mod_nine_divisible_by_nine_iff_digitSum_divisible_by_nine_l1691_169186


namespace three_digit_sum_l1691_169174

theorem three_digit_sum (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  a < 10 ∧ b < 10 ∧ c < 10 →
  21 * (a + b + c) = 231 →
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 6) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) ∨ 
  (a = 6 ∧ b = 2 ∧ c = 3) ∨ 
  (a = 6 ∧ b = 3 ∧ c = 2) :=
sorry

end three_digit_sum_l1691_169174


namespace comparison_problems_l1691_169169

theorem comparison_problems :
  (-0.1 < -0.01) ∧
  (-(-1) = abs (-1)) ∧
  (-abs (-7/8) < -(5/6)) := by sorry

end comparison_problems_l1691_169169


namespace solution_when_a_is_3_root_of_multiplicity_l1691_169172

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (a * x + 1) / (x - 1) - 2 / (1 - x) = 1

-- Part 1: Prove that when a = 3, the solution is x = -2
theorem solution_when_a_is_3 :
  ∃ x : ℝ, x ≠ 1 ∧ equation 3 x ∧ x = -2 :=
sorry

-- Part 2: Prove that the equation has a root of multiplicity when a = -3
theorem root_of_multiplicity :
  ∃ x : ℝ, x = 1 ∧ equation (-3) x :=
sorry

end solution_when_a_is_3_root_of_multiplicity_l1691_169172


namespace distance_circle_to_line_l1691_169165

/-- The distance from the center of the circle ρ=2cos θ to the line 2ρsin(θ + π/3)=1 is (√3 - 1) / 2 -/
theorem distance_circle_to_line :
  let circle : ℝ → ℝ → Prop := λ ρ θ ↦ ρ = 2 * Real.cos θ
  let line : ℝ → ℝ → Prop := λ ρ θ ↦ 2 * ρ * Real.sin (θ + π/3) = 1
  let circle_center : ℝ × ℝ := (1, 0)
  let distance := (Real.sqrt 3 - 1) / 2
  ∃ (d : ℝ), d = distance ∧ 
    d = (|Real.sqrt 3 * circle_center.1 + circle_center.2 - 1|) / Real.sqrt (3 + 1) :=
by sorry

end distance_circle_to_line_l1691_169165


namespace factoring_expression_l1691_169125

theorem factoring_expression (x : ℝ) : 5*x*(x-2) + 9*(x-2) = (x-2)*(5*x+9) := by
  sorry

end factoring_expression_l1691_169125


namespace tea_cheese_ratio_l1691_169104

/-- Represents the prices of items in Ursula's purchase -/
structure PurchasePrices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ

/-- The conditions of Ursula's purchase -/
def purchase_conditions (p : PurchasePrices) : Prop :=
  p.butter + p.bread + p.cheese + p.tea = 21 ∧
  p.bread = p.butter / 2 ∧
  p.butter = 0.8 * p.cheese ∧
  p.tea = 10

/-- The theorem stating the ratio of tea price to cheese price -/
theorem tea_cheese_ratio (p : PurchasePrices) :
  purchase_conditions p → p.tea / p.cheese = 2 := by
  sorry

end tea_cheese_ratio_l1691_169104


namespace fraction_sum_theorem_l1691_169155

theorem fraction_sum_theorem (a b c : ℝ) 
  (h : a / (35 - a) + b / (55 - b) + c / (70 - c) = 8) :
  7 / (35 - a) + 11 / (55 - b) + 14 / (70 - c) = 2.2 := by
  sorry

end fraction_sum_theorem_l1691_169155


namespace hairdresser_initial_amount_l1691_169159

def hairdresser_savings (initial_amount : ℕ) : Prop :=
  let first_year_spent := initial_amount / 2
  let second_year_spent := initial_amount / 3
  let third_year_spent := 200
  let remaining := initial_amount - first_year_spent - second_year_spent - third_year_spent
  (remaining = 50) ∧ 
  (first_year_spent = initial_amount / 2) ∧
  (second_year_spent = initial_amount / 3) ∧
  (third_year_spent = 200)

theorem hairdresser_initial_amount : 
  ∃ (initial_amount : ℕ), hairdresser_savings initial_amount ∧ initial_amount = 1500 :=
by
  sorry

end hairdresser_initial_amount_l1691_169159


namespace two_circles_k_value_l1691_169195

/-- Two circles centered at the origin with given properties --/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ × ℝ
  -- Distance QR
  QR : ℝ
  -- Conditions
  center_origin : True
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : S.1^2 + S.2^2 = r^2
  S_on_y_axis : S.1 = 0
  radius_difference : R - r = QR

/-- Theorem stating the value of k for the given two circles --/
theorem two_circles_k_value (c : TwoCircles) (h1 : c.P = (10, 2)) (h2 : c.QR = 5) :
  ∃ k : ℝ, c.S = (0, k) ∧ (k = Real.sqrt 104 - 5 ∨ k = -(Real.sqrt 104 - 5)) := by
  sorry

end two_circles_k_value_l1691_169195


namespace parents_without_jobs_l1691_169140

/-- The percentage of parents without full-time jobs -/
def percentage_without_jobs (mother_job_rate : ℝ) (father_job_rate : ℝ) (mother_percentage : ℝ) : ℝ :=
  100 - (mother_job_rate * mother_percentage + father_job_rate * (100 - mother_percentage))

theorem parents_without_jobs :
  percentage_without_jobs 90 75 40 = 19 := by
  sorry

end parents_without_jobs_l1691_169140


namespace right_triangle_median_on_hypotenuse_l1691_169107

/-- Given a right triangle with legs of lengths 6 and 8, 
    the length of the median on the hypotenuse is 5. -/
theorem right_triangle_median_on_hypotenuse : 
  ∀ (a b c m : ℝ), 
    a = 6 → 
    b = 8 → 
    c^2 = a^2 + b^2 → 
    m = c / 2 → 
    m = 5 := by
  sorry

end right_triangle_median_on_hypotenuse_l1691_169107


namespace cubic_factorization_l1691_169183

theorem cubic_factorization (y : ℝ) : y^3 - 16*y = y*(y+4)*(y-4) := by
  sorry

end cubic_factorization_l1691_169183


namespace average_pages_is_23_l1691_169163

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 161

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Calculates the average number of pages read per day -/
def average_pages_per_day : ℚ := total_pages / days_in_week

/-- Theorem stating that the average number of pages read per day is 23 -/
theorem average_pages_is_23 : average_pages_per_day = 23 := by
  sorry

end average_pages_is_23_l1691_169163


namespace unique_score_determination_l1691_169119

/-- Scoring system function -/
def score (c w : ℕ) : ℕ := 30 + 4 * c - w

/-- Proposition: There exists a unique combination of c and w such that the score is 92,
    and this is the only score above 90 that allows for a unique determination of c and w -/
theorem unique_score_determination :
  ∃! (c w : ℕ), score c w = 92 ∧
  (∀ (c' w' : ℕ), score c' w' > 90 ∧ score c' w' ≠ 92 → 
    ∃ (c'' w'' : ℕ), c'' ≠ c' ∧ w'' ≠ w' ∧ score c'' w'' = score c' w') :=
sorry

end unique_score_determination_l1691_169119


namespace acute_triangle_side_range_l1691_169170

-- Define an acute triangle with sides 3, 4, and a
def is_acute_triangle (a : ℝ) : Prop :=
  a > 0 ∧ 3 > 0 ∧ 4 > 0 ∧
  a + 3 > 4 ∧ a + 4 > 3 ∧ 3 + 4 > a ∧
  a^2 < 3^2 + 4^2 ∧ 3^2 < a^2 + 4^2 ∧ 4^2 < a^2 + 3^2

-- Theorem statement
theorem acute_triangle_side_range :
  ∀ a : ℝ, is_acute_triangle a → Real.sqrt 7 < a ∧ a < 5 :=
by sorry

end acute_triangle_side_range_l1691_169170


namespace point_B_coordinates_l1691_169138

/-- Given two points A and B in a 2D plane, this theorem proves that
    if the vector from A to B is (3, 4) and A has coordinates (-2, -1),
    then B has coordinates (1, 3). -/
theorem point_B_coordinates
  (A B : ℝ × ℝ)
  (h1 : A = (-2, -1))
  (h2 : B.1 - A.1 = 3 ∧ B.2 - A.2 = 4) :
  B = (1, 3) := by
  sorry

end point_B_coordinates_l1691_169138


namespace total_red_balloons_l1691_169147

/-- The total number of red balloons Fred, Sam, and Dan have is 72. -/
theorem total_red_balloons : 
  let fred_balloons : ℕ := 10
  let sam_balloons : ℕ := 46
  let dan_balloons : ℕ := 16
  fred_balloons + sam_balloons + dan_balloons = 72 := by
  sorry

end total_red_balloons_l1691_169147


namespace find_number_l1691_169154

theorem find_number (x : ℝ) : x^2 * 15^2 / 356 = 51.193820224719104 → x = 9 ∨ x = -9 := by
  sorry

end find_number_l1691_169154


namespace cos_decreasing_interval_l1691_169128

theorem cos_decreasing_interval (k : ℤ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 3)
  let a := k * π + π / 6
  let b := k * π + 2 * π / 3
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x :=
by sorry

end cos_decreasing_interval_l1691_169128


namespace science_club_enrollment_l1691_169179

theorem science_club_enrollment (total : ℕ) (math physics chem : ℕ) 
  (math_physics math_chem physics_chem : ℕ) (all_three : ℕ) 
  (h_total : total = 150)
  (h_math : math = 90)
  (h_physics : physics = 70)
  (h_chem : chem = 40)
  (h_math_physics : math_physics = 20)
  (h_math_chem : math_chem = 15)
  (h_physics_chem : physics_chem = 10)
  (h_all_three : all_three = 5) :
  total - (math + physics + chem - math_physics - math_chem - physics_chem + all_three) = 5 := by
  sorry

end science_club_enrollment_l1691_169179


namespace unique_solution_l1691_169151

theorem unique_solution : ∃! x : ℝ, (x = (1/x)*(-x) - 5) ∧ (x^2 - 3*x + 2 ≥ 0) := by
  sorry

end unique_solution_l1691_169151


namespace farm_problem_solution_l1691_169126

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_daily_rate : ℕ  -- Planned hectares per day
  actual_daily_rate : ℕ   -- Actual hectares per day
  extra_days : ℕ          -- Additional days worked
  remaining_area : ℕ      -- Hectares left to plough

/-- Calculates the total area and initially planned days for a given farm field problem -/
def solve_farm_problem (field : FarmField) : ℕ × ℕ :=
  sorry

/-- Theorem stating the solution to the specific farm field problem -/
theorem farm_problem_solution :
  let field := FarmField.mk 90 85 2 40
  solve_farm_problem field = (3780, 42) :=
sorry

end farm_problem_solution_l1691_169126


namespace parabola_line_intersection_right_angle_l1691_169124

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in 2D space of the form y^2 = kx -/
structure Parabola where
  k : ℝ

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def on_parabola (p : Point) (para : Parabola) : Prop :=
  p.y^2 = para.k * p.x

def right_angle (a b c : Point) : Prop :=
  (b.x - a.x) * (c.x - a.x) + (b.y - a.y) * (c.y - a.y) = 0

theorem parabola_line_intersection_right_angle :
  ∀ (l : Line) (para : Parabola) (a b c : Point),
    l.a = 1 ∧ l.b = -2 ∧ l.c = -1 →
    para.k = 4 →
    on_line a l ∧ on_parabola a para →
    on_line b l ∧ on_parabola b para →
    on_parabola c para →
    right_angle a c b →
    (c.x = 1 ∧ c.y = -2) ∨ (c.x = 9 ∧ c.y = -6) :=
by sorry

end parabola_line_intersection_right_angle_l1691_169124


namespace triangle_not_right_angle_l1691_169157

theorem triangle_not_right_angle (a b c : ℝ) (h_sum : a + b + c = 180) (h_ratio : ∃ k : ℝ, a = 3*k ∧ b = 4*k ∧ c = 5*k) : ¬(a = 90 ∨ b = 90 ∨ c = 90) := by
  sorry

end triangle_not_right_angle_l1691_169157


namespace part1_part2_l1691_169150

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part1 (t : Triangle) (h : (t.a^2 + t.c^2 - t.b^2) / 4 = 1/2 * t.a * t.c * Real.sin t.B) :
  t.B = π/4 := by
  sorry

-- Part 2
theorem part2 (t : Triangle) 
  (h1 : t.a * t.c = Real.sqrt 3)
  (h2 : Real.sin t.A = Real.sqrt 3 * Real.sin t.B)
  (h3 : t.C = π/6) :
  t.c = 1 := by
  sorry

end part1_part2_l1691_169150


namespace other_denomination_is_70_l1691_169196

/-- Proves that the other denomination of travelers checks is $70 --/
theorem other_denomination_is_70 
  (total_checks : ℕ)
  (total_worth : ℕ)
  (known_denomination : ℕ)
  (known_count : ℕ)
  (remaining_average : ℕ)
  (h1 : total_checks = 30)
  (h2 : total_worth = 1800)
  (h3 : known_denomination = 50)
  (h4 : known_count = 15)
  (h5 : remaining_average = 70)
  (h6 : known_count * known_denomination + (total_checks - known_count) * remaining_average = total_worth) :
  ∃ (other_denomination : ℕ), other_denomination = 70 ∧ 
    known_count * known_denomination + (total_checks - known_count) * other_denomination = total_worth :=
by sorry

end other_denomination_is_70_l1691_169196


namespace trombone_players_l1691_169161

/-- Represents the number of players for each instrument in an orchestra -/
structure Orchestra where
  total : Nat
  drummer : Nat
  trumpet : Nat
  frenchHorn : Nat
  violin : Nat
  cello : Nat
  contrabass : Nat
  clarinet : Nat
  flute : Nat
  maestro : Nat

/-- Theorem stating the number of trombone players in the orchestra -/
theorem trombone_players (o : Orchestra)
  (h1 : o.total = 21)
  (h2 : o.drummer = 1)
  (h3 : o.trumpet = 2)
  (h4 : o.frenchHorn = 1)
  (h5 : o.violin = 3)
  (h6 : o.cello = 1)
  (h7 : o.contrabass = 1)
  (h8 : o.clarinet = 3)
  (h9 : o.flute = 4)
  (h10 : o.maestro = 1) :
  o.total - (o.drummer + o.trumpet + o.frenchHorn + o.violin + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro) = 4 := by
  sorry


end trombone_players_l1691_169161


namespace f_properties_and_value_l1691_169143

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem stating the properties of f and its value at -1 -/
theorem f_properties_and_value :
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧ 
  (∀ x, f x = 3 * f⁻¹ x + 5) ∧
  (f 0 = 3) →
  f (-1) = 2 * Real.sqrt 3 / 3 := by sorry

end f_properties_and_value_l1691_169143


namespace lcm_18_24_l1691_169168

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l1691_169168


namespace circular_road_circumference_sum_l1691_169181

theorem circular_road_circumference_sum (R : ℝ) (h1 : R > 0) : 
  let r := R / 3
  let road_width := R - r
  road_width = 7 →
  2 * Real.pi * R + 2 * Real.pi * r = 28 * Real.pi :=
by
  sorry

end circular_road_circumference_sum_l1691_169181


namespace power_relation_l1691_169111

theorem power_relation (a : ℝ) (m n : ℤ) (hm : a ^ m = 4) (hn : a ^ n = 2) :
  a ^ (m - 2 * n) = 1 := by
  sorry

end power_relation_l1691_169111


namespace unique_solution_abs_equation_l1691_169171

theorem unique_solution_abs_equation :
  ∃! y : ℝ, y * |y| = -3 * y + 5 :=
by
  -- The unique solution is (-3 + √29) / 2
  use (-3 + Real.sqrt 29) / 2
  sorry

end unique_solution_abs_equation_l1691_169171


namespace pages_ratio_day2_to_day1_l1691_169199

/-- Represents the number of pages read on each day --/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Theorem stating the ratio of pages read on day 2 to day 1 --/
theorem pages_ratio_day2_to_day1 (pages : DailyPages) : 
  pages.day1 = 63 →
  pages.day3 = pages.day2 + 10 →
  pages.day4 = 29 →
  pages.day1 + pages.day2 + pages.day3 + pages.day4 = 354 →
  pages.day2 / pages.day1 = 2 := by
  sorry

#check pages_ratio_day2_to_day1

end pages_ratio_day2_to_day1_l1691_169199


namespace max_additional_bricks_l1691_169194

/-- Represents the weight capacity of a truck in terms of bags of sand -/
def sand_capacity : ℕ := 50

/-- Represents the weight capacity of a truck in terms of bricks -/
def brick_capacity : ℕ := 400

/-- Represents the number of bags of sand already in the truck -/
def sand_load : ℕ := 32

/-- Calculates the equivalent number of bricks for a given number of sand bags -/
def sand_to_brick_equiv (sand : ℕ) : ℕ :=
  (brick_capacity * sand) / sand_capacity

theorem max_additional_bricks : 
  sand_to_brick_equiv (sand_capacity - sand_load) = 144 := by
  sorry

end max_additional_bricks_l1691_169194


namespace initial_speed_proof_l1691_169144

/-- Proves that given the conditions of the journey, the initial speed must be 60 mph -/
theorem initial_speed_proof (v : ℝ) : 
  (v * 3 + 85 * 2) / 5 = 70 → v = 60 := by
  sorry

end initial_speed_proof_l1691_169144


namespace inequality_proof_l1691_169112

theorem inequality_proof (a b : ℝ) (h : a * b ≥ 0) :
  a^4 + 2*a^3*b + 2*a*b^3 + b^4 ≥ 6*a^2*b^2 := by
  sorry

end inequality_proof_l1691_169112


namespace largest_odd_digit_multiple_of_11_l1691_169102

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem largest_odd_digit_multiple_of_11 :
  ∀ n : ℕ,
    n < 10000 →
    has_only_odd_digits n →
    is_divisible_by_11 n →
    n ≤ 9559 :=
sorry

end largest_odd_digit_multiple_of_11_l1691_169102


namespace sum_of_two_angles_in_plane_l1691_169180

/-- 
Given three angles meeting at a point in a plane, where one angle is 130°, 
prove that the sum of the other two angles is 230°.
-/
theorem sum_of_two_angles_in_plane (x y : ℝ) : 
  x + y + 130 = 360 → x + y = 230 := by sorry

end sum_of_two_angles_in_plane_l1691_169180


namespace polynomial_multiplication_l1691_169193

theorem polynomial_multiplication (x : ℝ) : (x + 1) * (x^2 - x + 1) = x^3 + 1 := by
  sorry

end polynomial_multiplication_l1691_169193


namespace abc_inequality_l1691_169187

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (hsum : a^2 + b^2 + c^2 = 9) : a * b * c + 1 > 3 * a := by
  sorry

end abc_inequality_l1691_169187


namespace root_in_interval_l1691_169133

def f (x : ℝ) := x^2 - 1

theorem root_in_interval : ∃ x : ℝ, -2 < x ∧ x < 0 ∧ f x = 0 := by
  sorry

end root_in_interval_l1691_169133


namespace pure_imaginary_product_l1691_169166

theorem pure_imaginary_product (b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (y : ℝ), (1 + b * Complex.I) * (2 + Complex.I) = y * Complex.I) →
  b = 2 := by sorry

end pure_imaginary_product_l1691_169166


namespace no_real_solution_for_log_equation_l1691_169106

theorem no_real_solution_for_log_equation :
  ¬ ∃ (x : ℝ), (Real.log (x + 4) + Real.log (x - 2) = Real.log (x^2 - 6*x - 5)) ∧
               (x + 4 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 6*x - 5 > 0) := by
  sorry

end no_real_solution_for_log_equation_l1691_169106


namespace transformation_confluence_l1691_169173

/-- Represents a word in the alphabet {a, b} --/
inductive Word
| empty : Word
| cons_a : ℕ → Word → Word
| cons_b : ℕ → Word → Word

/-- Represents a transformation rule --/
structure TransformRule where
  k : ℕ
  l : ℕ
  k' : ℕ
  l' : ℕ
  h_k : k ≥ 1
  h_l : l ≥ 1
  h_k' : k' ≥ 1
  h_l' : l' ≥ 1

/-- Applies a transformation rule to a word --/
def applyRule (rule : TransformRule) (w : Word) : Option Word :=
  sorry

/-- Checks if a word is terminal with respect to a rule --/
def isTerminal (rule : TransformRule) (w : Word) : Prop :=
  applyRule rule w = none

/-- Represents a sequence of transformations --/
def TransformSequence := List (TransformRule × Word)

/-- Applies a sequence of transformations to a word --/
def applySequence (seq : TransformSequence) (w : Word) : Word :=
  sorry

theorem transformation_confluence (rule : TransformRule) (w : Word) :
  ∀ (seq1 seq2 : TransformSequence),
    isTerminal rule (applySequence seq1 w) →
    isTerminal rule (applySequence seq2 w) →
    applySequence seq1 w = applySequence seq2 w :=
  sorry

end transformation_confluence_l1691_169173


namespace interest_calculation_l1691_169108

/-- Problem Statement: A sum is divided into two parts with specific interest conditions. -/
theorem interest_calculation (total_sum second_sum : ℕ) 
  (first_rate second_rate : ℚ) (first_years : ℕ) : 
  total_sum = 2795 →
  second_sum = 1720 →
  first_rate = 3/100 →
  second_rate = 5/100 →
  first_years = 8 →
  ∃ (second_years : ℕ),
    (total_sum - second_sum) * first_rate * first_years = 
    second_sum * second_rate * second_years ∧
    second_years = 3 := by
  sorry


end interest_calculation_l1691_169108


namespace total_swimming_time_l1691_169118

/-- Represents the swimming times for various events -/
structure SwimmingTimes where
  freestyle : ℕ
  backstroke : ℕ
  butterfly : ℕ
  breaststroke : ℕ
  sidestroke : ℕ
  individual_medley : ℕ

/-- Calculates the total time for all events -/
def total_time (times : SwimmingTimes) : ℕ :=
  times.freestyle + times.backstroke + times.butterfly + 
  times.breaststroke + times.sidestroke + times.individual_medley

/-- Theorem stating the total time for all events -/
theorem total_swimming_time :
  ∀ (times : SwimmingTimes),
    times.freestyle = 48 →
    times.backstroke = times.freestyle + 4 + 2 →
    times.butterfly = times.backstroke + 3 + 3 →
    times.breaststroke = times.butterfly + 2 - 1 →
    times.sidestroke = times.butterfly + 5 + 4 →
    times.individual_medley = times.breaststroke + 6 + 3 →
    total_time times = 362 := by
  sorry

#eval total_time { freestyle := 48, backstroke := 54, butterfly := 60, 
                   breaststroke := 61, sidestroke := 69, individual_medley := 70 }

end total_swimming_time_l1691_169118


namespace equation_solution_l1691_169146

theorem equation_solution (x : ℝ) (h : x ≠ 3) :
  (x + 6) / (x - 3) = 5 / 2 ↔ x = 9 := by
sorry

end equation_solution_l1691_169146


namespace twelfth_day_is_monday_l1691_169139

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat

/-- The theorem to be proved -/
theorem twelfth_day_is_monday (m : Month) : 
  m.fridayCount = 5 ∧ 
  m.firstDay ≠ DayOfWeek.Friday ∧ 
  m.lastDay ≠ DayOfWeek.Friday ∧
  m.dayCount ≥ 28 ∧ m.dayCount ≤ 31 →
  (DayOfWeek.Monday : DayOfWeek) = 
    match (m.firstDay, 11) with
    | (DayOfWeek.Sunday, n) => DayOfWeek.Wednesday
    | (DayOfWeek.Monday, n) => DayOfWeek.Thursday
    | (DayOfWeek.Tuesday, n) => DayOfWeek.Friday
    | (DayOfWeek.Wednesday, n) => DayOfWeek.Saturday
    | (DayOfWeek.Thursday, n) => DayOfWeek.Sunday
    | (DayOfWeek.Friday, n) => DayOfWeek.Monday
    | (DayOfWeek.Saturday, n) => DayOfWeek.Tuesday
  := by sorry

end twelfth_day_is_monday_l1691_169139


namespace trains_combined_length_l1691_169109

/-- The combined length of two trains crossing a platform -/
theorem trains_combined_length (speed_A speed_B : ℝ) (platform_length time : ℝ) : 
  speed_A = 72 * (5/18) → 
  speed_B = 54 * (5/18) → 
  platform_length = 210 → 
  time = 26 → 
  (speed_A + speed_B) * time - platform_length = 700 := by
  sorry


end trains_combined_length_l1691_169109


namespace train_length_l1691_169188

/-- Calculates the length of a train given the time it takes to cross a bridge and pass a lamp post. -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ) :
  bridge_length = 150 →
  bridge_time = 7.5 →
  lamp_time = 2.5 →
  ∃ (train_length : ℝ), train_length = 75 ∧ 
    (train_length / lamp_time = (train_length + bridge_length) / bridge_time) :=
by sorry


end train_length_l1691_169188


namespace total_painting_cost_l1691_169120

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def count_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

def house_cost (address : ℕ) : ℚ :=
  (1.5 : ℚ) * (count_digits address)

def side_cost (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℚ :=
  List.sum (List.map house_cost (List.map (arithmetic_sequence a₁ d) (List.range n)))

theorem total_painting_cost :
  side_cost 5 6 25 + side_cost 2 6 25 = 171 := by
  sorry

end total_painting_cost_l1691_169120


namespace valid_three_digit_numbers_l1691_169152

/-- The count of three-digit numbers. -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical non-adjacent digits. -/
def excluded_numbers : ℕ := 81

/-- The count of valid three-digit numbers after exclusion. -/
def valid_numbers : ℕ := total_three_digit_numbers - excluded_numbers

theorem valid_three_digit_numbers :
  valid_numbers = 819 :=
by sorry

end valid_three_digit_numbers_l1691_169152


namespace equation_solutions_l1691_169167

theorem equation_solutions :
  (∃ x : ℚ, (1/2) * x - 3 = 2 * x + 1/2 ∧ x = -7/3) ∧
  (∃ x : ℚ, (x-3)/2 - (2*x+1)/3 = 1 ∧ x = -17) := by
  sorry

end equation_solutions_l1691_169167


namespace orthogonal_projection_area_range_l1691_169176

/-- Regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  base_side : ℝ
  lateral_edge : ℝ

/-- Orthogonal projection area of a regular quadrangular pyramid -/
def orthogonal_projection_area (p : RegularQuadrangularPyramid) (angle : ℝ) : ℝ :=
  sorry

/-- Theorem: Range of orthogonal projection area -/
theorem orthogonal_projection_area_range 
  (p : RegularQuadrangularPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_edge = Real.sqrt 6) : 
  ∀ angle, 2 ≤ orthogonal_projection_area p angle ∧ 
           orthogonal_projection_area p angle ≤ 4 :=
sorry

end orthogonal_projection_area_range_l1691_169176


namespace vectors_form_basis_l1691_169162

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

def is_basis (v w : V) : Prop :=
  LinearIndependent ℝ ![v, w] ∧ Submodule.span ℝ {v, w} = ⊤

theorem vectors_form_basis (ha : a ≠ 0) (hb : b ≠ 0) (hnc : ¬ ∃ (k : ℝ), a = k • b) :
  is_basis V (a + b) (a - b) := by
  sorry

end vectors_form_basis_l1691_169162


namespace final_price_calculation_l1691_169116

/-- The markup percentage applied to the cost price -/
def markup : ℝ := 0.15

/-- The cost price of the computer table -/
def costPrice : ℝ := 5565.217391304348

/-- The final price paid by the customer -/
def finalPrice : ℝ := 6400

/-- Theorem stating that the final price is equal to the cost price plus the markup -/
theorem final_price_calculation :
  finalPrice = costPrice * (1 + markup) := by sorry

end final_price_calculation_l1691_169116


namespace product_has_34_digits_l1691_169100

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The product of two large numbers -/
def n : ℕ := 3659893456789325678 * 342973489379256

/-- Theorem stating that the product has 34 digits -/
theorem product_has_34_digits : num_digits n = 34 := by sorry

end product_has_34_digits_l1691_169100


namespace daisy_percentage_is_62_l1691_169158

/-- Represents the composition of flowers in a garden -/
structure Garden where
  total : ℝ
  yellow_ratio : ℝ
  yellow_tulip_ratio : ℝ
  red_daisy_ratio : ℝ

/-- The percentage of daisies in the garden -/
def daisy_percentage (g : Garden) : ℝ :=
  ((g.yellow_ratio - g.yellow_ratio * g.yellow_tulip_ratio) + 
   ((1 - g.yellow_ratio) * g.red_daisy_ratio)) * 100

/-- Theorem stating that the percentage of daisies in the garden is 62% -/
theorem daisy_percentage_is_62 (g : Garden) 
  (h1 : g.yellow_tulip_ratio = 1/5)
  (h2 : g.red_daisy_ratio = 1/2)
  (h3 : g.yellow_ratio = 4/10) :
  daisy_percentage g = 62 := by
  sorry

#eval daisy_percentage { total := 100, yellow_ratio := 0.4, yellow_tulip_ratio := 0.2, red_daisy_ratio := 0.5 }

end daisy_percentage_is_62_l1691_169158


namespace initial_money_calculation_l1691_169105

/-- Proves that the initial amount of money is $160 given the conditions of the problem -/
theorem initial_money_calculation (your_weekly_savings : ℕ) (friend_initial_money : ℕ) 
  (friend_weekly_savings : ℕ) (weeks : ℕ) (h1 : your_weekly_savings = 7) 
  (h2 : friend_initial_money = 210) (h3 : friend_weekly_savings = 5) (h4 : weeks = 25) :
  ∃ (your_initial_money : ℕ), 
    your_initial_money + your_weekly_savings * weeks = 
    friend_initial_money + friend_weekly_savings * weeks ∧ 
    your_initial_money = 160 := by
  sorry

end initial_money_calculation_l1691_169105


namespace complex_number_properties_l1691_169130

theorem complex_number_properties (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 1 + 2*I) (hz₂ : z₂ = 3 - 4*I) : 
  (Complex.im (z₁ * z₂) = 2) ∧ 
  (Complex.re (z₁ * z₂) > 0 ∧ Complex.im (z₁ * z₂) > 0) ∧
  (Complex.re z₁ > 0 ∧ Complex.im z₁ > 0) := by
  sorry


end complex_number_properties_l1691_169130


namespace original_number_l1691_169192

theorem original_number (x : ℚ) : (3 * (x + 3) - 4) / 3 = 10 → x = 25 / 3 := by
  sorry

end original_number_l1691_169192


namespace cube_volume_from_surface_area_l1691_169142

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 150 → s^3 = 125 := by
  sorry

end cube_volume_from_surface_area_l1691_169142


namespace second_derivative_at_pi_over_six_l1691_169136

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem second_derivative_at_pi_over_six :
  (deriv^[2] f) (π / 6) = -(1 - Real.sqrt 3) / 2 := by sorry

end second_derivative_at_pi_over_six_l1691_169136


namespace circle_center_trajectory_equation_l1691_169153

/-- The trajectory of the center of a circle passing through (4,0) and 
    intersecting the y-axis with a chord of length 8 -/
def circle_center_trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1 - 16}

/-- The fixed point through which the circle passes -/
def fixed_point : ℝ × ℝ := (4, 0)

/-- The length of the chord cut by the circle on the y-axis -/
def chord_length : ℝ := 8

/-- Theorem stating that the trajectory of the circle's center satisfies the given equation -/
theorem circle_center_trajectory_equation :
  ∀ (x y : ℝ), (x, y) ∈ circle_center_trajectory ↔ y^2 = 8*x - 16 :=
sorry

end circle_center_trajectory_equation_l1691_169153


namespace customers_without_tip_l1691_169164

theorem customers_without_tip (total_customers : ℕ) (total_tips : ℕ) (tip_per_customer : ℕ) :
  total_customers = 7 →
  total_tips = 6 →
  tip_per_customer = 3 →
  total_customers - (total_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_without_tip_l1691_169164


namespace modular_inverse_27_mod_28_l1691_169127

theorem modular_inverse_27_mod_28 : ∃ a : ℕ, 0 ≤ a ∧ a ≤ 27 ∧ (27 * a) % 28 = 1 :=
by
  use 27
  sorry

end modular_inverse_27_mod_28_l1691_169127


namespace unique_positive_solution_l1691_169131

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 2023*x^5 - 2021*x^4

-- Theorem statement
theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end unique_positive_solution_l1691_169131


namespace min_value_theorem_l1691_169121

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 36/x^2 ≥ 12 * (4^(1/4)) ∧
  (x^2 + 6*x + 36/x^2 = 12 * (4^(1/4)) ↔ x = 36^(1/3)) := by
  sorry

end min_value_theorem_l1691_169121


namespace complement_of_union_is_empty_l1691_169101

def U : Finset Nat := {1, 2, 3, 4}
def A : Finset Nat := {1, 3, 4}
def B : Finset Nat := {2, 3, 4}

theorem complement_of_union_is_empty :
  (U \ (A ∪ B) : Finset Nat) = ∅ := by sorry

end complement_of_union_is_empty_l1691_169101


namespace parallel_to_y_axis_l1691_169110

/-- Given two points P and Q in a Cartesian coordinate system,
    where P has coordinates (m, 3) and Q has coordinates (2-2m, m-3),
    and PQ is parallel to the y-axis, prove that m = 2/3. -/
theorem parallel_to_y_axis (m : ℚ) : 
  let P : ℚ × ℚ := (m, 3)
  let Q : ℚ × ℚ := (2 - 2*m, m - 3)
  (P.1 = Q.1) → m = 2/3 := by
  sorry

end parallel_to_y_axis_l1691_169110


namespace acute_angle_range_l1691_169190

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

def acute_angle_obtainable (x : ℝ) : Prop :=
  let dot_product := (a x).1 * (b x).1 + (a x).2 * (b x).2
  dot_product > 0 ∧ dot_product < (Real.sqrt ((a x).1^2 + (a x).2^2) * Real.sqrt ((b x).1^2 + (b x).2^2))

theorem acute_angle_range :
  ∀ x : ℝ, acute_angle_obtainable x ↔ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 3) :=
by sorry

end acute_angle_range_l1691_169190


namespace grandfather_wins_l1691_169123

/-- The number of games played -/
def total_games : ℕ := 12

/-- Points scored by grandfather for each win -/
def grandfather_points : ℕ := 1

/-- Points scored by grandson for each win -/
def grandson_points : ℕ := 3

/-- Theorem stating the number of games won by the grandfather -/
theorem grandfather_wins (x : ℕ) 
  (h1 : x ≤ total_games)
  (h2 : x * grandfather_points = (total_games - x) * grandson_points) : 
  x = 9 := by
  sorry

end grandfather_wins_l1691_169123


namespace special_table_sum_l1691_169113

/-- Represents a 2 × 7 table where each column after the first is the sum and difference of the previous column --/
def SpecialTable := Fin 7 → Fin 2 → ℤ

/-- The rule for generating subsequent columns --/
def nextColumn (col : Fin 2 → ℤ) : Fin 2 → ℤ :=
  fun i => if i = 0 then col 0 + col 1 else col 0 - col 1

/-- Checks if the table follows the special rule --/
def isValidTable (t : SpecialTable) : Prop :=
  ∀ j : Fin 6, t (j.succ) = nextColumn (t j)

/-- The theorem to be proved --/
theorem special_table_sum (t : SpecialTable) : 
  isValidTable t → t 6 0 = 96 → t 6 1 = 64 → t 0 0 + t 0 1 = 20 := by
  sorry

#check special_table_sum

end special_table_sum_l1691_169113


namespace fourth_power_sqrt_equals_256_l1691_169145

theorem fourth_power_sqrt_equals_256 (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end fourth_power_sqrt_equals_256_l1691_169145


namespace inequality_multiplication_l1691_169122

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end inequality_multiplication_l1691_169122


namespace room_length_calculation_l1691_169198

/-- Given a room with width 2.75 m and a floor paving cost of 600 per sq. metre
    resulting in a total cost of 10725, the length of the room is 6.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  width = 2.75 ∧ cost_per_sqm = 600 ∧ total_cost = 10725 →
  total_cost = (6.5 * width * cost_per_sqm) :=
by sorry

end room_length_calculation_l1691_169198


namespace f_has_minimum_value_neg_twelve_l1691_169137

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 9

theorem f_has_minimum_value_neg_twelve :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≥ f x₀ ∧ f x₀ = -12 := by
  sorry

end f_has_minimum_value_neg_twelve_l1691_169137


namespace smallest_integer_l1691_169132

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 50) :
  ∀ c : ℕ, c > 0 ∧ Nat.lcm a c / Nat.gcd a c = 50 → b ≤ c := by
  sorry

#check smallest_integer

end smallest_integer_l1691_169132


namespace division_and_power_equality_l1691_169129

theorem division_and_power_equality : ((-125) / (-25)) ^ 3 = 125 := by
  sorry

end division_and_power_equality_l1691_169129


namespace stratified_sampling_primary_schools_l1691_169175

theorem stratified_sampling_primary_schools 
  (total_schools : ℕ) 
  (primary_schools : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_schools = 250) 
  (h2 : primary_schools = 150) 
  (h3 : sample_size = 30) :
  (primary_schools : ℚ) / total_schools * sample_size = 18 := by
sorry

end stratified_sampling_primary_schools_l1691_169175


namespace sum_of_tens_digits_l1691_169197

/-- Given single-digit numbers A, B, C, D such that A + B + C + D = 22,
    the sum of the tens digits of (A + B) and (C + D) equals 4. -/
theorem sum_of_tens_digits (A B C D : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10)
    (h5 : A + B + C + D = 22) : (A + B) / 10 + (C + D) / 10 = 4 := by
  sorry

end sum_of_tens_digits_l1691_169197


namespace midnight_temperature_l1691_169148

/-- 
Given an initial temperature, a temperature rise, and a temperature drop,
calculate the final temperature.
-/
def final_temperature (initial : Int) (rise : Int) (drop : Int) : Int :=
  initial + rise - drop

/--
Theorem: Given the specific temperature changes in the problem,
the final temperature is -4°C.
-/
theorem midnight_temperature : final_temperature (-3) 6 7 = -4 := by
  sorry

end midnight_temperature_l1691_169148


namespace b_share_is_360_l1691_169135

/-- Represents the rental information for a person --/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total horse-months for a rental --/
def horsemonths (r : RentalInfo) : ℕ := r.horses * r.months

theorem b_share_is_360 (total_rent : ℕ) (a b c : RentalInfo) 
  (h1 : total_rent = 870)
  (h2 : a = ⟨12, 8⟩)
  (h3 : b = ⟨16, 9⟩)
  (h4 : c = ⟨18, 6⟩) :
  (horsemonths b * total_rent) / (horsemonths a + horsemonths b + horsemonths c) = 360 := by
  sorry

#eval (16 * 9 * 870) / (12 * 8 + 16 * 9 + 18 * 6)

end b_share_is_360_l1691_169135


namespace decimal_representation_digits_l1691_169141

theorem decimal_representation_digits (n : ℕ) (d : ℕ) (h : n / d = 7^3 / (14^2 * 125)) : 
  (∃ k : ℕ, n / d = k / 1000 ∧ k < 1000 ∧ k ≥ 100) := by
  sorry

end decimal_representation_digits_l1691_169141


namespace partition_iff_even_l1691_169182

def is_valid_partition (n : ℕ) (partition : List (List ℕ)) : Prop :=
  partition.length = n ∧
  partition.all (λ l => l.length = 4) ∧
  (partition.join.toFinset : Finset ℕ) = Finset.range (4 * n + 1) \ {0} ∧
  ∀ l ∈ partition, ∃ x ∈ l, 3 * x = (l.sum - x)

theorem partition_iff_even (n : ℕ) :
  (∃ partition : List (List ℕ), is_valid_partition n partition) ↔ Even n :=
sorry

end partition_iff_even_l1691_169182


namespace first_group_size_first_group_size_proof_l1691_169156

/-- Given two groups of workers building walls, this theorem proves that the number of workers in the first group is 20, based on the given conditions. -/
theorem first_group_size : ℕ :=
  let wall_length_1 : ℝ := 66
  let days_1 : ℕ := 4
  let wall_length_2 : ℝ := 567.6
  let days_2 : ℕ := 8
  let workers_2 : ℕ := 86
  let workers_1 := (wall_length_1 * days_2 * workers_2) / (wall_length_2 * days_1)
  20

/-- Proof of the theorem -/
theorem first_group_size_proof : first_group_size = 20 := by
  sorry

end first_group_size_first_group_size_proof_l1691_169156


namespace largest_prime_divisor_13_plus_14_factorial_l1691_169134

theorem largest_prime_divisor_13_plus_14_factorial (p : ℕ) :
  (p.Prime ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14)) →
  p ≤ 13 :=
by
  sorry

end largest_prime_divisor_13_plus_14_factorial_l1691_169134


namespace angle_SPQ_is_20_degrees_l1691_169178

/-- A geometric configuration with specific angle measures -/
structure GeometricConfiguration where
  -- Point Q lies on PR and Point S lies on QT (implied by the existence of these angles)
  angle_QST : ℝ  -- Measure of angle QST
  angle_TSP : ℝ  -- Measure of angle TSP
  angle_RQS : ℝ  -- Measure of angle RQS

/-- Theorem stating that under the given conditions, angle SPQ measures 20 degrees -/
theorem angle_SPQ_is_20_degrees (config : GeometricConfiguration)
  (h1 : config.angle_QST = 180)
  (h2 : config.angle_TSP = 50)
  (h3 : config.angle_RQS = 150) :
  config.angle_TSP + 20 = config.angle_RQS :=
sorry

end angle_SPQ_is_20_degrees_l1691_169178


namespace company_choices_eq_24_l1691_169114

/-- The number of ways two students can choose companies with exactly one overlap -/
def company_choices : ℕ :=
  -- Number of ways to choose the shared company
  4 *
  -- Ways for student A to choose the second company
  3 *
  -- Ways for student B to choose the second company
  2

/-- Theorem stating that the number of ways to choose companies with one overlap is 24 -/
theorem company_choices_eq_24 : company_choices = 24 := by
  sorry

end company_choices_eq_24_l1691_169114


namespace smallest_abs_value_not_one_l1691_169177

theorem smallest_abs_value_not_one : ¬(∀ q : ℚ, q ≠ 0 → |q| ≥ 1) := by
  sorry

end smallest_abs_value_not_one_l1691_169177


namespace total_commute_time_is_19_point_1_l1691_169185

/-- Represents the commute schedule for a week --/
structure CommuteSchedule where
  normalWalkTime : ℝ
  normalBikeTime : ℝ
  wednesdayExtraTime : ℝ
  fridayExtraTime : ℝ
  rainIncreaseFactor : ℝ
  mondayIsWalking : Bool
  tuesdayIsBiking : Bool
  wednesdayIsWalking : Bool
  thursdayIsWalking : Bool
  fridayIsBiking : Bool
  mondayIsRainy : Bool
  thursdayIsRainy : Bool

/-- Calculates the total commute time for a week given a schedule --/
def totalCommuteTime (schedule : CommuteSchedule) : ℝ :=
  let mondayTime := if schedule.mondayIsWalking then
    (if schedule.mondayIsRainy then schedule.normalWalkTime * (1 + schedule.rainIncreaseFactor) else schedule.normalWalkTime) * 2
  else schedule.normalBikeTime * 2

  let tuesdayTime := if schedule.tuesdayIsBiking then schedule.normalBikeTime * 2
  else schedule.normalWalkTime * 2

  let wednesdayTime := if schedule.wednesdayIsWalking then (schedule.normalWalkTime + schedule.wednesdayExtraTime) * 2
  else schedule.normalBikeTime * 2

  let thursdayTime := if schedule.thursdayIsWalking then
    (if schedule.thursdayIsRainy then schedule.normalWalkTime * (1 + schedule.rainIncreaseFactor) else schedule.normalWalkTime) * 2
  else schedule.normalBikeTime * 2

  let fridayTime := if schedule.fridayIsBiking then (schedule.normalBikeTime + schedule.fridayExtraTime) * 2
  else schedule.normalWalkTime * 2

  mondayTime + tuesdayTime + wednesdayTime + thursdayTime + fridayTime

/-- The main theorem stating that given the specific schedule, the total commute time is 19.1 hours --/
theorem total_commute_time_is_19_point_1 :
  let schedule : CommuteSchedule := {
    normalWalkTime := 2
    normalBikeTime := 1
    wednesdayExtraTime := 0.5
    fridayExtraTime := 0.25
    rainIncreaseFactor := 0.2
    mondayIsWalking := true
    tuesdayIsBiking := true
    wednesdayIsWalking := true
    thursdayIsWalking := true
    fridayIsBiking := true
    mondayIsRainy := true
    thursdayIsRainy := true
  }
  totalCommuteTime schedule = 19.1 := by sorry

end total_commute_time_is_19_point_1_l1691_169185


namespace museum_revenue_l1691_169115

def minutes_between (start_hour start_min end_hour end_min : ℕ) : ℕ :=
  (end_hour - start_hour) * 60 + end_min - start_min

def total_intervals (interval_length : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / interval_length

def total_people (people_per_interval intervals : ℕ) : ℕ :=
  people_per_interval * intervals

def student_tickets (total_people : ℕ) : ℕ :=
  total_people / 4

def regular_tickets (student_tickets : ℕ) : ℕ :=
  3 * student_tickets

def total_revenue (student_tickets regular_tickets : ℕ) (student_price regular_price : ℕ) : ℕ :=
  student_tickets * student_price + regular_tickets * regular_price

theorem museum_revenue : 
  let total_mins := minutes_between 9 0 17 55
  let intervals := total_intervals 5 total_mins
  let total_ppl := total_people 30 intervals
  let students := student_tickets total_ppl
  let regulars := regular_tickets students
  total_revenue students regulars 4 8 = 22456 := by
  sorry

end museum_revenue_l1691_169115


namespace water_dispenser_capacity_l1691_169184

/-- A cylindrical water dispenser with capacity x liters -/
structure WaterDispenser where
  capacity : ℝ
  cylindrical : Bool

/-- The water dispenser contains 60 liters when it is 25% full -/
def quarter_full (d : WaterDispenser) : Prop :=
  0.25 * d.capacity = 60

/-- Theorem: A cylindrical water dispenser that contains 60 liters when 25% full has a total capacity of 240 liters -/
theorem water_dispenser_capacity (d : WaterDispenser) 
  (h1 : d.cylindrical = true) 
  (h2 : quarter_full d) : 
  d.capacity = 240 := by
  sorry

end water_dispenser_capacity_l1691_169184
