import Mathlib

namespace sufficient_not_necessary_condition_l3883_388326

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Add necessary conditions for a valid triangle
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  -- Add cosine law
  cos_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := t.b = t.c ∨ t.a = t.c ∨ t.a = t.b

-- State the theorem
theorem sufficient_not_necessary_condition (t : Triangle) :
  (t.a = 2 * t.b * Real.cos t.C → isIsosceles t) ∧
  ∃ t' : Triangle, isIsosceles t' ∧ t'.a ≠ 2 * t'.b * Real.cos t'.C :=
by sorry

end sufficient_not_necessary_condition_l3883_388326


namespace divisibility_theorem_l3883_388392

theorem divisibility_theorem (a b c : ℕ+) 
  (h1 : a ∣ b^5)
  (h2 : b ∣ c^5)
  (h3 : c ∣ a^5) :
  (a * b * c) ∣ (a + b + c)^31 := by
  sorry

end divisibility_theorem_l3883_388392


namespace slope_range_ordinate_range_l3883_388336

-- Define the point A
def A : ℝ × ℝ := (0, 3)

-- Define the line l
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def conditions (C : Circle) (k : ℝ) : Prop :=
  C.radius = 1 ∧
  C.center.2 = line_l C.center.1 ∧
  C.center.2 = C.center.1 - 1 ∧
  ∃ (x y : ℝ), (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ∧ y = k * x + 3

-- Define the theorems to be proved
theorem slope_range (C : Circle) :
  (∃ k, conditions C k) → ∃ k, -3/4 ≤ k ∧ k ≤ 0 :=
sorry

theorem ordinate_range (C : Circle) :
  (∃ M : ℝ × ℝ, (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = 1 ∧
   (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - 0)^2 + (M.2 - 0)^2)) →
  -4 ≤ C.center.2 ∧ C.center.2 ≤ 4/5 :=
sorry

end slope_range_ordinate_range_l3883_388336


namespace percentage_relation_l3883_388324

/-- Given that j is 25% less than p, j is 20% less than t, and t is q% less than p, prove that q = 6.25% -/
theorem percentage_relation (p t j : ℝ) (q : ℝ) 
  (h1 : j = p * (1 - 0.25))
  (h2 : j = t * (1 - 0.20))
  (h3 : t = p * (1 - q / 100)) :
  q = 6.25 := by
sorry

end percentage_relation_l3883_388324


namespace inequality_range_l3883_388372

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x > 2*a*x + a) ↔ -4 < a ∧ a < -1 := by
  sorry

end inequality_range_l3883_388372


namespace line_parabola_intersection_l3883_388395

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line equation y = k(x-2) + 1 passing through P(2,1) -/
def line (k x y : ℝ) : Prop := y = k*(x-2) + 1

/-- The number of common points between the line and the parabola -/
inductive CommonPoints
  | one
  | two
  | none

/-- Theorem stating the conditions for the number of common points -/
theorem line_parabola_intersection (k : ℝ) :
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2) ↔ k = 0 ∧
  ¬(∃ p q : ℝ × ℝ, p ≠ q ∧ parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ line k p.1 p.2 ∧ line k q.1 q.2) ∧
  ¬(∀ p : ℝ × ℝ, parabola p.1 p.2 → ¬line k p.1 p.2) :=
sorry

end line_parabola_intersection_l3883_388395


namespace sqrt_8_to_6th_power_l3883_388304

theorem sqrt_8_to_6th_power : (Real.sqrt 8) ^ 6 = 512 := by
  sorry

end sqrt_8_to_6th_power_l3883_388304


namespace cone_generatrix_length_l3883_388359

/-- 
Given a cone with base radius √2 whose lateral surface can be unfolded into a semicircle,
prove that the length of its generatrix is 2√2.
-/
theorem cone_generatrix_length 
  (base_radius : ℝ) 
  (h_base_radius : base_radius = Real.sqrt 2) 
  (lateral_surface_is_semicircle : Bool) 
  (h_lateral_surface : lateral_surface_is_semicircle = true) : 
  ∃ (generatrix_length : ℝ), generatrix_length = 2 * Real.sqrt 2 := by
  sorry

end cone_generatrix_length_l3883_388359


namespace calculation_proof_l3883_388311

theorem calculation_proof : 
  ((0.8 + (1 / 5)) * 24 + 6.6) / (9 / 14) - 7.6 = 40 := by
  sorry

end calculation_proof_l3883_388311


namespace greatest_integer_sqrt_l3883_388338

theorem greatest_integer_sqrt (N : ℤ) : 
  (∀ m : ℤ, m ≤ Real.sqrt (2007^2 - 20070 + 31) → m ≤ N) ∧ 
  N ≤ Real.sqrt (2007^2 - 20070 + 31) → 
  N = 2002 := by
  sorry

end greatest_integer_sqrt_l3883_388338


namespace second_largest_is_five_l3883_388329

def number_set : Finset ℕ := {5, 8, 4, 3, 2}

theorem second_largest_is_five :
  ∃ (x : ℕ), x ∈ number_set ∧ 
  (∀ y ∈ number_set, y ≠ x → y ≤ x) ∧
  (∃ z ∈ number_set, z > x) ∧
  x = 5 := by
  sorry

end second_largest_is_five_l3883_388329


namespace degree_to_radian_conversion_l3883_388380

theorem degree_to_radian_conversion (π : ℝ) (h : π > 0) :
  let degree_to_radian (d : ℝ) := d * (π / 180)
  degree_to_radian 15 = π / 12 := by
sorry

end degree_to_radian_conversion_l3883_388380


namespace vector_MN_l3883_388337

def M : ℝ × ℝ := (-3, 3)
def N : ℝ × ℝ := (-5, -1)

theorem vector_MN : N.1 - M.1 = -2 ∧ N.2 - M.2 = -4 := by sorry

end vector_MN_l3883_388337


namespace g_negative_101_l3883_388307

/-- A function g satisfying the given functional equation -/
def g_function (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x = x * g y + g x

theorem g_negative_101 (g : ℝ → ℝ) (h1 : g_function g) (h2 : g 1 = 7) : 
  g (-101) = -95 :=
sorry

end g_negative_101_l3883_388307


namespace johns_journey_speed_l3883_388328

theorem johns_journey_speed (total_distance : ℝ) (first_duration : ℝ) (second_duration : ℝ) (second_speed : ℝ) (S : ℝ) :
  total_distance = 240 →
  first_duration = 2 →
  second_duration = 3 →
  second_speed = 50 →
  total_distance = first_duration * S + second_duration * second_speed →
  S = 45 := by
  sorry

end johns_journey_speed_l3883_388328


namespace total_sweaters_61_l3883_388371

def sweaters_fortnight (day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 : ℕ) : Prop :=
  day1 = 8 ∧
  day2 = day1 + 2 ∧
  day3_4 = day2 - 4 ∧
  day5 = day3_4 ∧
  day6 = day1 / 2 ∧
  day7 = 0 ∧
  day8_10 = (day1 + day2 + day3_4 * 2 + day5 + day6) * 3 * 3 / (4 * 6) ∧
  day11 = day8_10 / 3 / 3 ∧
  day12_13 = day8_10 / 2 / 3 ∧
  day14 = 1

theorem total_sweaters_61 :
  ∀ day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 : ℕ,
  sweaters_fortnight day1 day2 day3_4 day5 day6 day7 day8_10 day11 day12_13 day14 →
  day1 + day2 + day3_4 * 2 + day5 + day6 + day7 + day8_10 + day11 + day12_13 * 2 + day14 = 61 :=
by
  sorry

end total_sweaters_61_l3883_388371


namespace experiment_arrangements_l3883_388334

/-- Represents the number of procedures in the experiment -/
def num_procedures : ℕ := 6

/-- Represents whether procedure A is at the beginning or end -/
inductive A_position
| beginning
| end

/-- Calculates the number of arrangements for a given A position -/
def arrangements_for_A_position (pos : A_position) : ℕ := 
  (Nat.factorial (num_procedures - 3)) * 2

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ :=
  arrangements_for_A_position A_position.beginning + 
  arrangements_for_A_position A_position.end

/-- Theorem stating that the total number of arrangements is 96 -/
theorem experiment_arrangements :
  total_arrangements = 96 := by sorry

end experiment_arrangements_l3883_388334


namespace probability_theorem_l3883_388398

/-- Represents a club with members -/
structure Club where
  total : ℕ
  girls : ℕ
  boys : ℕ
  girls_under_18 : ℕ

/-- Calculates the probability of choosing two girls with at least one under 18 -/
def probability_two_girls_one_under_18 (club : Club) : ℚ :=
  let total_combinations := club.total.choose 2
  let girls_combinations := club.girls.choose 2
  let underaged_combinations := 
    club.girls_under_18 * (club.girls - club.girls_under_18) + club.girls_under_18.choose 2
  (underaged_combinations : ℚ) / total_combinations

/-- The main theorem to prove -/
theorem probability_theorem (club : Club) 
    (h1 : club.total = 15)
    (h2 : club.girls = 8)
    (h3 : club.boys = 7)
    (h4 : club.girls_under_18 = 3)
    (h5 : club.total = club.girls + club.boys) :
  probability_two_girls_one_under_18 club = 6/35 := by
  sorry

#eval probability_two_girls_one_under_18 ⟨15, 8, 7, 3⟩

end probability_theorem_l3883_388398


namespace sum_of_multiples_l3883_388306

theorem sum_of_multiples (p q : ℤ) : 
  (∃ m : ℤ, p = 5 * m) → (∃ n : ℤ, q = 10 * n) → (∃ k : ℤ, p + q = 5 * k) := by
  sorry

end sum_of_multiples_l3883_388306


namespace cost_difference_l3883_388390

def dan_money : ℕ := 5
def chocolate_cost : ℕ := 3
def candy_bar_cost : ℕ := 7

theorem cost_difference : candy_bar_cost - chocolate_cost = 4 := by
  sorry

end cost_difference_l3883_388390


namespace area_of_semicircle_with_inscribed_rectangle_l3883_388339

/-- A semicircle with an inscribed 1 × 3 rectangle -/
structure InscribedRectangleSemicircle where
  /-- The radius of the semicircle -/
  radius : ℝ
  /-- The width of the inscribed rectangle -/
  rect_width : ℝ
  /-- The length of the inscribed rectangle -/
  rect_length : ℝ
  /-- The width of the rectangle is 1 -/
  width_is_one : rect_width = 1
  /-- The length of the rectangle is 3 -/
  length_is_three : rect_length = 3
  /-- The rectangle is inscribed in the semicircle -/
  inscribed : radius^2 = (rect_width / 2)^2 + (rect_length / 2)^2

/-- The area of the semicircle with an inscribed 1 × 3 rectangle is 13π/8 -/
theorem area_of_semicircle_with_inscribed_rectangle 
  (s : InscribedRectangleSemicircle) : 
  π * s.radius^2 / 2 = 13 * π / 8 := by
  sorry

#check area_of_semicircle_with_inscribed_rectangle

end area_of_semicircle_with_inscribed_rectangle_l3883_388339


namespace skee_ball_tickets_value_l3883_388351

/-- The number of tickets Kaleb won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 8

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 5

/-- The number of candies Kaleb can buy -/
def candies_bought : ℕ := 3

/-- The number of tickets Kaleb won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candy_cost * candies_bought - whack_a_mole_tickets

theorem skee_ball_tickets_value : skee_ball_tickets = 7 := by
  sorry

end skee_ball_tickets_value_l3883_388351


namespace equation_solution_l3883_388344

theorem equation_solution : ∃ x : ℝ, 4*x + 4 - x - 2*x + 2 - 2 - x + 2 + 6 = 0 ∧ x = 0 := by
  sorry

end equation_solution_l3883_388344


namespace min_tiles_to_cover_l3883_388399

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Theorem stating the minimum number of tiles needed to cover the specified area -/
theorem min_tiles_to_cover (tileSize : Dimensions) (regionSize : Dimensions) (coveredSize : Dimensions) : 
  tileSize.length = 3 →
  tileSize.width = 4 →
  regionSize.length = feetToInches 3 →
  regionSize.width = feetToInches 6 →
  coveredSize.length = feetToInches 1 →
  coveredSize.width = feetToInches 1 →
  (area regionSize - area coveredSize) / area tileSize = 204 := by
  sorry

end min_tiles_to_cover_l3883_388399


namespace tribe_organization_ways_l3883_388363

/-- The number of members in the tribe -/
def tribeSize : ℕ := 13

/-- The number of supporting chiefs -/
def numSupportingChiefs : ℕ := 3

/-- The number of inferiors for each supporting chief -/
def numInferiors : ℕ := 2

/-- Calculate the number of ways to organize the tribe's leadership -/
def organizationWays : ℕ := 
  tribeSize * (tribeSize - 1) * (tribeSize - 2) * (tribeSize - 3) * 
  Nat.choose (tribeSize - 4) 2 * 
  Nat.choose (tribeSize - 6) 2 * 
  Nat.choose (tribeSize - 8) 2

/-- Theorem stating that the number of ways to organize the leadership is 12355200 -/
theorem tribe_organization_ways : organizationWays = 12355200 := by
  sorry

end tribe_organization_ways_l3883_388363


namespace value_of_b_l3883_388383

theorem value_of_b (a b t : ℝ) 
  (eq1 : a - t / 6 * b = 20)
  (eq2 : a - t / 5 * b = -10)
  (t_val : t = 60) : b = 15 := by
  sorry

end value_of_b_l3883_388383


namespace circle1_correct_circle2_correct_l3883_388305

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 8)^2 + (y + 3)^2 = 25
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the points
def point_A1 : ℝ × ℝ := (5, 1)
def point_A2 : ℝ × ℝ := (-1, 5)
def point_B2 : ℝ × ℝ := (5, 5)
def point_C2 : ℝ × ℝ := (6, -2)

-- Theorem for circle 1
theorem circle1_correct :
  circle1 (point_A1.1) (point_A1.2) ∧
  ∀ (x y : ℝ), circle1 x y → (x - 8)^2 + (y + 3)^2 = 25 := by sorry

-- Theorem for circle 2
theorem circle2_correct :
  circle2 (point_A2.1) (point_A2.2) ∧
  circle2 (point_B2.1) (point_B2.2) ∧
  circle2 (point_C2.1) (point_C2.2) ∧
  ∀ (x y : ℝ), circle2 x y → x^2 + y^2 - 4*x - 2*y - 20 = 0 := by sorry

end circle1_correct_circle2_correct_l3883_388305


namespace uber_lyft_cost_difference_uber_lyft_cost_difference_proof_l3883_388396

/-- The cost difference between Uber and Lyft rides --/
theorem uber_lyft_cost_difference : ℝ :=
  let taxi_cost : ℝ := 15  -- Derived from the 20% tip condition
  let lyft_cost : ℝ := taxi_cost + 4
  let uber_cost : ℝ := 22
  uber_cost - lyft_cost

/-- Proof of the cost difference between Uber and Lyft rides --/
theorem uber_lyft_cost_difference_proof :
  uber_lyft_cost_difference = 3 := by
  sorry

end uber_lyft_cost_difference_uber_lyft_cost_difference_proof_l3883_388396


namespace log_inequality_may_not_hold_l3883_388373

theorem log_inequality_may_not_hold (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  ¬ (∀ a b : ℝ, 1/a < 1/b ∧ 1/b < 0 → Real.log (-a) / Real.log (-b) ≥ 0) :=
by sorry

end log_inequality_may_not_hold_l3883_388373


namespace largest_constant_inequality_l3883_388368

theorem largest_constant_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3)) ∧
  ∀ k > 2, ∃ a' b' c' d' : ℝ, 
    (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ (0 ≤ c' ∧ c' ≤ 1) ∧ (0 ≤ d' ∧ d' ≤ 1) ∧
    (a'^2 * b' + b'^2 * c' + c'^2 * d' + d'^2 * a' + 4 < k * (a'^3 + b'^3 + c'^3 + d'^3)) :=
by sorry

end largest_constant_inequality_l3883_388368


namespace base_prime_repr_360_l3883_388354

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The number 360 -/
def n : ℕ := 360

theorem base_prime_repr_360 :
  base_prime_repr n = [3, 2, 1] :=
by
  sorry

end base_prime_repr_360_l3883_388354


namespace central_cell_value_l3883_388322

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the 3x3 table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

/-- The theorem stating that the central cell value is 0.00081 -/
theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end central_cell_value_l3883_388322


namespace first_term_of_constant_ratio_l3883_388366

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℕ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → arithmetic_sum a d (4 * n) / arithmetic_sum a d n = c) →
  a = 5 / 2 :=
by sorry

end first_term_of_constant_ratio_l3883_388366


namespace no_real_roots_l3883_388365

theorem no_real_roots : ¬ ∃ (x : ℝ), x^2 + 3*x + 3 = 0 := by
  sorry

end no_real_roots_l3883_388365


namespace opposite_sides_condition_l3883_388386

/-- 
Given a real number m, if the points (1, 2) and (1, 1) are on opposite sides of the line y - 3x - m = 0, 
then -2 < m < -1.
-/
theorem opposite_sides_condition (m : ℝ) : 
  (2 - 3 * 1 - m) * (1 - 3 * 1 - m) < 0 → -2 < m ∧ m < -1 := by
  sorry

end opposite_sides_condition_l3883_388386


namespace train_length_train_length_problem_l3883_388360

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + person_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  relative_speed_mps * passing_time

/-- Proof that a train with speed 56 km/hr passing a man running at 6 km/hr in the opposite direction in 6.386585847325762 seconds has a length of approximately 110 meters. -/
theorem train_length_problem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_length 56 6 6.386585847325762 - 110| < ε :=
sorry

end train_length_train_length_problem_l3883_388360


namespace regular_polygon_with_18_degree_exterior_angle_has_20_sides_l3883_388378

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  exterior_angle : ℝ
  regular : exterior_angle * (sides : ℝ) = 360

-- Theorem statement
theorem regular_polygon_with_18_degree_exterior_angle_has_20_sides :
  ∀ p : RegularPolygon, p.exterior_angle = 18 → p.sides = 20 := by
  sorry

end regular_polygon_with_18_degree_exterior_angle_has_20_sides_l3883_388378


namespace football_campers_count_l3883_388321

theorem football_campers_count (total : ℕ) (basketball : ℕ) (soccer : ℕ) 
  (h1 : total = 88) 
  (h2 : basketball = 24) 
  (h3 : soccer = 32) : 
  total - soccer - basketball = 32 := by
sorry

end football_campers_count_l3883_388321


namespace hiking_duration_is_six_hours_l3883_388345

/-- Represents the hiking scenario with given initial weights and consumption rates. --/
structure HikingScenario where
  initialWater : ℝ
  initialFood : ℝ
  initialGear : ℝ
  waterConsumptionRate : ℝ
  foodConsumptionRate : ℝ

/-- Calculates the remaining weight after a given number of hours. --/
def remainingWeight (scenario : HikingScenario) (hours : ℝ) : ℝ :=
  scenario.initialWater + scenario.initialFood + scenario.initialGear -
  scenario.waterConsumptionRate * hours -
  scenario.foodConsumptionRate * hours

/-- Theorem stating that under the given conditions, the hiking duration is 6 hours. --/
theorem hiking_duration_is_six_hours (scenario : HikingScenario)
  (h1 : scenario.initialWater = 20)
  (h2 : scenario.initialFood = 10)
  (h3 : scenario.initialGear = 20)
  (h4 : scenario.waterConsumptionRate = 2)
  (h5 : scenario.foodConsumptionRate = 2/3)
  (h6 : remainingWeight scenario 6 = 34) :
  ∃ (h : ℝ), h = 6 ∧ remainingWeight scenario h = 34 := by
  sorry


end hiking_duration_is_six_hours_l3883_388345


namespace exactly_one_true_l3883_388331

-- Define the polynomials
def A (x : ℝ) : ℝ := 2 * x^2
def B (x : ℝ) : ℝ := x + 1
def C (x : ℝ) : ℝ := -2 * x
def D (y : ℝ) : ℝ := y^2
def E (x y : ℝ) : ℝ := 2 * x - y

-- Define the three statements
def statement1 : Prop :=
  ∀ y : ℕ+, ∀ x : ℝ, B x * C x + A x + D y + E x y > 0

def statement2 : Prop :=
  ∃ x y : ℝ, A x + D y + 2 * E x y = -2

def statement3 : Prop :=
  ∀ x : ℝ, ∀ m : ℝ,
    (∃ k : ℝ, 3 * (A x - B x) + m * B x * C x = k * x^2 + (3 : ℝ)) →
    3 * (A x - B x) + m * B x * C x > -3

theorem exactly_one_true : (statement1 ∧ ¬statement2 ∧ ¬statement3) ∨
                           (¬statement1 ∧ statement2 ∧ ¬statement3) ∨
                           (¬statement1 ∧ ¬statement2 ∧ statement3) :=
  sorry

end exactly_one_true_l3883_388331


namespace equal_slopes_imply_equal_angles_l3883_388374

/-- Theorem: For two lines with inclination angles in [0, π) and equal slopes, their inclination angles are equal. -/
theorem equal_slopes_imply_equal_angles (α₁ α₂ : Real) (k₁ k₂ : Real) :
  0 ≤ α₁ ∧ α₁ < π →
  0 ≤ α₂ ∧ α₂ < π →
  k₁ = Real.tan α₁ →
  k₂ = Real.tan α₂ →
  k₁ = k₂ →
  α₁ = α₂ := by
  sorry

end equal_slopes_imply_equal_angles_l3883_388374


namespace find_y_l3883_388391

theorem find_y (x : ℝ) (y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 24) : y = 96 := by
  sorry

end find_y_l3883_388391


namespace tyler_meal_combinations_l3883_388387

/-- The number of meat options available -/
def num_meats : ℕ := 4

/-- The number of vegetable options available -/
def num_vegetables : ℕ := 4

/-- The number of dessert options available -/
def num_desserts : ℕ := 5

/-- The number of bread options available -/
def num_breads : ℕ := 3

/-- The number of vegetables Tyler must choose -/
def vegetables_to_choose : ℕ := 2

/-- The number of breads Tyler must choose -/
def breads_to_choose : ℕ := 2

/-- Calculates the number of ways to choose k items from n items without replacement and without order -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of ways to choose k items from n items without replacement but with order -/
def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The total number of meal combinations Tyler can choose -/
def total_combinations : ℕ := 
  num_meats * choose num_vegetables vegetables_to_choose * num_desserts * permute num_breads breads_to_choose

theorem tyler_meal_combinations : total_combinations = 720 := by
  sorry

end tyler_meal_combinations_l3883_388387


namespace points_to_win_match_jeff_tennis_points_l3883_388309

/-- Calculates the number of points needed to win a tennis match given the total playing time,
    point scoring interval, and number of games won. -/
theorem points_to_win_match 
  (total_time : ℕ) 
  (point_interval : ℕ) 
  (games_won : ℕ) : ℕ :=
  let total_minutes := total_time * 60
  let total_points := total_minutes / point_interval
  total_points / games_won

/-- Proves that 8 points are needed to win a match given the specific conditions. -/
theorem jeff_tennis_points : points_to_win_match 2 5 3 = 8 := by
  sorry

end points_to_win_match_jeff_tennis_points_l3883_388309


namespace total_money_for_76_members_l3883_388375

/-- Calculates the total money collected in rupees given the number of members in a group -/
def totalMoneyCollected (members : ℕ) : ℚ :=
  (members * members : ℕ) / 100

/-- Proves that for a group of 76 members, the total money collected is ₹57.76 -/
theorem total_money_for_76_members :
  totalMoneyCollected 76 = 57.76 := by
  sorry

end total_money_for_76_members_l3883_388375


namespace sin_alpha_for_point_neg_one_three_l3883_388370

/-- Given an angle α whose terminal side passes through the point (-1, 3),
    prove that sin α = (3 * √10) / 10 -/
theorem sin_alpha_for_point_neg_one_three (α : Real) :
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 3) →
  Real.sin α = (3 * Real.sqrt 10) / 10 := by
  sorry

end sin_alpha_for_point_neg_one_three_l3883_388370


namespace circle_T_six_three_l3883_388367

-- Define the operation ⊤
def circle_T (a b : ℤ) : ℤ := 4 * a - 7 * b

-- Theorem statement
theorem circle_T_six_three : circle_T 6 3 = 3 := by
  sorry

end circle_T_six_three_l3883_388367


namespace jack_coffee_batch_size_l3883_388397

/-- Proves that Jack makes 1.5 gallons of cold brew coffee in each batch given the conditions --/
theorem jack_coffee_batch_size :
  let coffee_per_2days : ℝ := 96  -- ounces
  let days : ℝ := 24
  let hours_per_batch : ℝ := 20
  let total_hours : ℝ := 120
  let ounces_per_gallon : ℝ := 128
  
  let total_coffee := (days / 2) * coffee_per_2days
  let total_gallons := total_coffee / ounces_per_gallon
  let num_batches := total_hours / hours_per_batch
  let gallons_per_batch := total_gallons / num_batches
  
  gallons_per_batch = 1.5 := by sorry

end jack_coffee_batch_size_l3883_388397


namespace division_result_l3883_388346

theorem division_result : (0.05 : ℚ) / (0.002 : ℚ) = 25 := by
  sorry

end division_result_l3883_388346


namespace smallest_sum_with_gcd_conditions_l3883_388355

theorem smallest_sum_with_gcd_conditions (a b c : ℕ+) : 
  (Nat.gcd a.val (Nat.gcd b.val c.val) = 1) →
  (Nat.gcd a.val (b.val + c.val) > 1) →
  (Nat.gcd b.val (c.val + a.val) > 1) →
  (Nat.gcd c.val (a.val + b.val) > 1) →
  (∃ (x y z : ℕ+), x.val + y.val + z.val < a.val + b.val + c.val) →
  a.val + b.val + c.val ≥ 30 :=
by sorry

end smallest_sum_with_gcd_conditions_l3883_388355


namespace inequality_proof_l3883_388317

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l3883_388317


namespace base_height_calculation_l3883_388347

/-- Given a sculpture height and total height, calculate the base height -/
theorem base_height_calculation (sculpture_height_feet : ℚ) (sculpture_height_inches : ℚ) (total_height : ℚ) : 
  sculpture_height_feet = 2 ∧ 
  sculpture_height_inches = 10 ∧ 
  total_height = 3.6666666666666665 →
  total_height - (sculpture_height_feet + sculpture_height_inches / 12) = 5 / 6 := by
  sorry

end base_height_calculation_l3883_388347


namespace rug_area_theorem_l3883_388312

theorem rug_area_theorem (total_floor_area : ℝ) (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_floor_area = 140)
  (h2 : two_layer_area = 22)
  (h3 : three_layer_area = 19) :
  total_floor_area + two_layer_area + 2 * three_layer_area = 200 :=
by sorry

end rug_area_theorem_l3883_388312


namespace max_z_value_l3883_388385

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 0) (h2 : x*y + y*z + z*x = -3) :
  ∃ (max_z : ℝ), z ≤ max_z ∧ max_z = 2 := by
sorry

end max_z_value_l3883_388385


namespace conic_section_properties_l3883_388320

/-- A conic section defined by the equation x^2 + x + 2y - 2 = 0 -/
def conic_section (x y : ℝ) : Prop := x^2 + x + 2*y - 2 = 0

/-- The first line: x - 2y + 3 = 0 -/
def line1 (x y : ℝ) : Prop := x - 2*y + 3 = 0

/-- The second line: 5x + 2y - 6 = 0 -/
def line2 (x y : ℝ) : Prop := 5*x + 2*y - 6 = 0

/-- Point P -/
def P : ℝ × ℝ := (-1, 1)

/-- Point Q -/
def Q : ℝ × ℝ := (2, -2)

/-- Point R -/
def R : ℝ × ℝ := (1, 0)

/-- The conic section is tangent to line1 at point P, tangent to line2 at point Q, and passes through point R -/
theorem conic_section_properties :
  (conic_section P.1 P.2 ∧ line1 P.1 P.2) ∧
  (conic_section Q.1 Q.2 ∧ line2 Q.1 Q.2) ∧
  conic_section R.1 R.2 :=
sorry

end conic_section_properties_l3883_388320


namespace polygon_diagonals_l3883_388362

/-- A polygon with interior angle sum of 1800 degrees has 9 diagonals from one vertex -/
theorem polygon_diagonals (n : ℕ) : 
  (n - 2) * 180 = 1800 → n - 3 = 9 := by
  sorry

end polygon_diagonals_l3883_388362


namespace triangle_BC_length_l3883_388335

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.A.1 = 1 ∧ t.A.2 = 1 ∧
  t.B.2 = parabola t.B.1 ∧
  t.C.2 = parabola t.C.1 ∧
  t.B.2 = t.C.2 ∧
  (1/2 * (t.C.1 - t.B.1) * (t.B.2 - t.A.2) = 32)

-- Theorem statement
theorem triangle_BC_length (t : Triangle) :
  triangle_conditions t → (t.C.1 - t.B.1 = 8) :=
by sorry

end triangle_BC_length_l3883_388335


namespace largest_negative_integer_l3883_388341

theorem largest_negative_integer :
  ∃! n : ℤ, n < 0 ∧ ∀ m : ℤ, m < 0 → m ≤ n :=
by
  sorry

end largest_negative_integer_l3883_388341


namespace constant_value_l3883_388314

def f (x : ℝ) : ℝ := 3 * x - 5

theorem constant_value : ∃ c : ℝ, 2 * f 3 - c = f (3 - 2) ∧ c = 10 := by
  sorry

end constant_value_l3883_388314


namespace complex_number_magnitude_squared_l3883_388343

theorem complex_number_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 - 7*I) :
  Complex.abs z^2 = 65/8 := by
  sorry

end complex_number_magnitude_squared_l3883_388343


namespace sum_of_digits_greatest_prime_divisor_16385_l3883_388369

def greatest_prime_divisor (n : Nat) : Nat :=
  sorry

def sum_of_digits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor 16385) = 19 := by
  sorry

end sum_of_digits_greatest_prime_divisor_16385_l3883_388369


namespace bacon_only_count_l3883_388308

theorem bacon_only_count (total_bacon : ℕ) (both : ℕ) (h1 : total_bacon = 569) (h2 : both = 218) :
  total_bacon - both = 351 := by
  sorry

end bacon_only_count_l3883_388308


namespace five_digit_multiplication_reversal_l3883_388364

theorem five_digit_multiplication_reversal :
  ∃! (a b c d e : ℕ),
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    0 ≤ e ∧ e ≤ 9 ∧
    a ≠ 0 ∧
    (10000 * a + 1000 * b + 100 * c + 10 * d + e) * 9 =
    10000 * e + 1000 * d + 100 * c + 10 * b + a ∧
    a = 1 ∧ b = 0 ∧ c = 9 ∧ d = 8 ∧ e = 9 :=
by sorry

end five_digit_multiplication_reversal_l3883_388364


namespace remaining_water_l3883_388313

-- Define the initial amount of water
def initial_water : ℚ := 3

-- Define the first usage
def first_usage : ℚ := 5/4

-- Define the second usage
def second_usage : ℚ := 1/3

-- Theorem to prove
theorem remaining_water :
  initial_water - first_usage - second_usage = 17/12 := by
  sorry

end remaining_water_l3883_388313


namespace last_three_digits_of_2_power_10000_l3883_388318

theorem last_three_digits_of_2_power_10000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^10000 ≡ 1 [ZMOD 1000] := by
sorry

end last_three_digits_of_2_power_10000_l3883_388318


namespace total_balloons_count_l3883_388356

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem total_balloons_count : total_balloons = 81 := by
  sorry

end total_balloons_count_l3883_388356


namespace unique_trig_value_l3883_388350

open Real

theorem unique_trig_value (x : ℝ) (h1 : 0 < x) (h2 : x < π / 3) 
  (h3 : cos x = tan x) : sin x = (-1 + Real.sqrt 5) / 2 := by
  sorry

end unique_trig_value_l3883_388350


namespace mrs_thompson_chicken_cost_l3883_388353

/-- Given the total cost, number of chickens, and cost of potatoes, 
    calculate the cost of each chicken. -/
def chicken_cost (total : ℚ) (num_chickens : ℕ) (potato_cost : ℚ) : ℚ :=
  (total - potato_cost) / num_chickens

/-- Prove that each chicken costs $3 given the problem conditions -/
theorem mrs_thompson_chicken_cost :
  chicken_cost 15 3 6 = 3 := by
  sorry

end mrs_thompson_chicken_cost_l3883_388353


namespace lcm_gcd_product_30_75_l3883_388348

theorem lcm_gcd_product_30_75 : Nat.lcm 30 75 * Nat.gcd 30 75 = 2250 := by
  sorry

end lcm_gcd_product_30_75_l3883_388348


namespace no_root_greater_than_three_l3883_388301

theorem no_root_greater_than_three : 
  ¬∃ x : ℝ, (x > 3 ∧ 
    ((3 * x^2 - 2 = 25) ∨ 
     ((2*x-1)^2 = (x-1)^2) ∨ 
     (x^2 - 7 = x - 1 ∧ x ≥ 1))) := by
  sorry

end no_root_greater_than_three_l3883_388301


namespace symmetric_log_value_of_a_l3883_388316

/-- Given a function f and a real number a, we say f is symmetric to log₂(x+a) with respect to y = x -/
def symmetric_to_log (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = 2^x - a

theorem symmetric_log_value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_sym : symmetric_to_log f a) (h_sum : f 2 + f 4 = 6) : a = 7 := by
  sorry

end symmetric_log_value_of_a_l3883_388316


namespace trig_identity_l3883_388319

theorem trig_identity (α : Real) 
  (h : Real.sin α + Real.cos α = 1/5) : 
  ((Real.sin α - Real.cos α)^2 = 49/25) ∧ 
  (Real.sin α^3 + Real.cos α^3 = 37/125) := by
  sorry

end trig_identity_l3883_388319


namespace maria_name_rearrangement_time_l3883_388379

/-- The time in hours to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (repeated_letters : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  let total_rearrangements := (Nat.factorial name_length) / (Nat.factorial repeated_letters)
  let minutes_needed := total_rearrangements / rearrangements_per_minute
  (minutes_needed : ℚ) / 60

/-- Theorem stating that the time to write all rearrangements of Maria's name is 0.125 hours -/
theorem maria_name_rearrangement_time :
  time_to_write_rearrangements 5 1 8 = 1/8 := by
  sorry

end maria_name_rearrangement_time_l3883_388379


namespace x_axis_intersection_correct_y_coord_correct_l3883_388394

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 5 * y - 2 * x = 10

/-- The point where the line intersects the x-axis -/
def x_axis_intersection : ℝ × ℝ := (-5, 0)

/-- The y-coordinate when x = -5 -/
def y_coord_at_neg_five : ℝ := 0

/-- Theorem stating that x_axis_intersection is on the line and has y-coordinate 0 -/
theorem x_axis_intersection_correct :
  line_equation x_axis_intersection.1 x_axis_intersection.2 ∧ x_axis_intersection.2 = 0 := by sorry

/-- Theorem stating that when x = -5, the y-coordinate is y_coord_at_neg_five -/
theorem y_coord_correct : line_equation (-5) y_coord_at_neg_five := by sorry

end x_axis_intersection_correct_y_coord_correct_l3883_388394


namespace unique_root_quadratic_l3883_388349

/-- A function f(x) = ax^2 - x - 1 has exactly one root if and only if a = 0 or a = -1/4 -/
theorem unique_root_quadratic (a : ℝ) : 
  (∃! x, a * x^2 - x - 1 = 0) ↔ (a = 0 ∨ a = -1/4) := by
sorry

end unique_root_quadratic_l3883_388349


namespace remainder_problem_l3883_388300

theorem remainder_problem (N : ℕ) (D : ℕ) : 
  (N % 158 = 50) → (N % D = 13) → (D > 13) → (D < 158) → D = 37 := by
  sorry

end remainder_problem_l3883_388300


namespace first_division_meiosis_characteristics_l3883_388358

/-- Represents the behavior of chromosomes during cell division -/
inductive ChromosomeBehavior
  | separate
  | notSeparate

/-- Represents the behavior of centromeres during cell division -/
inductive CentromereBehavior
  | split
  | notSplit

/-- Represents the characteristics of a cell division -/
structure CellDivisionCharacteristics where
  chromosomeBehavior : ChromosomeBehavior
  centromereBehavior : CentromereBehavior

/-- Represents the first division of meiosis -/
def firstDivisionMeiosis : CellDivisionCharacteristics := sorry

/-- Theorem stating the characteristics of the first division of meiosis -/
theorem first_division_meiosis_characteristics :
  firstDivisionMeiosis.chromosomeBehavior = ChromosomeBehavior.separate ∧
  firstDivisionMeiosis.centromereBehavior = CentromereBehavior.notSplit :=
sorry

end first_division_meiosis_characteristics_l3883_388358


namespace price_decrease_percentage_l3883_388393

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 775)
  (h2 : new_price = 620) :
  (original_price - new_price) / original_price * 100 = 20 := by
  sorry

end price_decrease_percentage_l3883_388393


namespace arrangements_with_pair_eq_10080_l3883_388389

/-- The number of ways to arrange n people in a line. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 8 people in a line where two specific individuals must always stand next to each other. -/
def arrangements_with_pair : ℕ :=
  factorial 7 * factorial 2

theorem arrangements_with_pair_eq_10080 :
  arrangements_with_pair = 10080 := by sorry

end arrangements_with_pair_eq_10080_l3883_388389


namespace remaining_squares_l3883_388325

/-- A chocolate bar with rectangular shape -/
structure ChocolateBar where
  length : ℕ
  width : ℕ
  total_squares : ℕ
  h_width : width = 6
  h_length : length ≥ 9
  h_total : total_squares = length * width

/-- The number of squares removed by Irena and Jack -/
def squares_removed : ℕ := 12 + 9

/-- The theorem stating the number of remaining squares -/
theorem remaining_squares (bar : ChocolateBar) : 
  bar.total_squares - squares_removed = 45 := by
  sorry

#check remaining_squares

end remaining_squares_l3883_388325


namespace number_of_observations_proof_l3883_388330

theorem number_of_observations_proof (initial_mean : ℝ) (incorrect_obs : ℝ) (correct_obs : ℝ) (new_mean : ℝ) :
  initial_mean = 32 →
  incorrect_obs = 23 →
  correct_obs = 48 →
  new_mean = 32.5 →
  ∃ n : ℕ, n > 0 ∧ n * initial_mean + (correct_obs - incorrect_obs) = n * new_mean ∧ n = 50 :=
by
  sorry

#check number_of_observations_proof

end number_of_observations_proof_l3883_388330


namespace rearrangement_maintains_ratio_l3883_388333

/-- Represents a figure made of sticks -/
structure StickFigure where
  num_sticks : ℕ
  area : ℝ

/-- The total number of sticks -/
def total_sticks : ℕ := 20

/-- The number of fixed sticks -/
def fixed_sticks : ℕ := 12

/-- The initial figure with 14 sticks -/
def initial_figure_14 : StickFigure := ⟨14, 3⟩

/-- The initial figure with 6 sticks -/
def initial_figure_6 : StickFigure := ⟨6, 1⟩

/-- The rearranged figure with 7 sticks -/
def rearranged_figure_7 : StickFigure := ⟨7, 1⟩

/-- The rearranged figure with 13 sticks -/
def rearranged_figure_13 : StickFigure := ⟨13, 3⟩

/-- Theorem stating that the rearrangement maintains the area ratio -/
theorem rearrangement_maintains_ratio :
  (initial_figure_14.area / initial_figure_6.area = 
   rearranged_figure_13.area / rearranged_figure_7.area) ∧
  (total_sticks = initial_figure_14.num_sticks + initial_figure_6.num_sticks) ∧
  (total_sticks = rearranged_figure_13.num_sticks + rearranged_figure_7.num_sticks) ∧
  (fixed_sticks + rearranged_figure_13.num_sticks - rearranged_figure_7.num_sticks = initial_figure_14.num_sticks) :=
by sorry

end rearrangement_maintains_ratio_l3883_388333


namespace rooster_ratio_l3883_388332

theorem rooster_ratio (total : ℕ) (roosters : ℕ) (hens : ℕ) :
  total = 80 →
  total = roosters + hens →
  roosters + (1/4 : ℚ) * hens = 35 →
  (roosters : ℚ) / total = 1/4 := by
  sorry

end rooster_ratio_l3883_388332


namespace phone_call_cost_per_minute_l3883_388376

/-- Proves that the cost per minute of each phone call is $0.05 given the specified conditions --/
theorem phone_call_cost_per_minute 
  (call_duration : ℝ) 
  (customers_per_week : ℕ) 
  (monthly_bill : ℝ) 
  (weeks_per_month : ℕ) : 
  call_duration = 1 →
  customers_per_week = 50 →
  monthly_bill = 600 →
  weeks_per_month = 4 →
  (monthly_bill / (customers_per_week * weeks_per_month * call_duration * 60)) = 0.05 := by
  sorry

end phone_call_cost_per_minute_l3883_388376


namespace contrapositive_square_inequality_l3883_388303

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x > y) → ¬(x^2 > y^2)) ↔ (x ≤ y → x^2 ≤ y^2) :=
by sorry

end contrapositive_square_inequality_l3883_388303


namespace shortest_tree_height_l3883_388357

/-- Proves that the height of the shortest tree is 50 feet given the conditions of the problem. -/
theorem shortest_tree_height (tallest middle shortest : ℝ) : 
  tallest = 150 ∧ 
  middle = 2/3 * tallest ∧ 
  shortest = 1/2 * middle →
  shortest = 50 := by
  sorry

end shortest_tree_height_l3883_388357


namespace least_possible_z_l3883_388340

theorem least_possible_z (x y z : ℤ) : 
  Even x → Odd y → Odd z → y - x > 5 → (∀ w, Odd w → w - x ≥ 9 → z ≤ w) → z = 11 :=
by sorry

end least_possible_z_l3883_388340


namespace solve_bucket_problem_l3883_388361

def bucket_problem (b1 b2 b3 b4 b5 : ℕ) : Prop :=
  b1 = 11 ∧ b2 = 13 ∧ b3 = 12 ∧ b4 = 16 ∧ b5 = 10 →
  (b5 + b2 = 23) →
  (b1 + b3 + b4 = 39)

theorem solve_bucket_problem :
  ∀ b1 b2 b3 b4 b5 : ℕ, bucket_problem b1 b2 b3 b4 b5 :=
by
  sorry

end solve_bucket_problem_l3883_388361


namespace inequality_equivalence_l3883_388388

theorem inequality_equivalence (x : ℝ) :
  (3 * x - 5 ≥ 9 - 2 * x) ↔ (x ≥ 14 / 5) := by
  sorry

end inequality_equivalence_l3883_388388


namespace gold_distribution_l3883_388381

theorem gold_distribution (n : ℕ) (a₁ : ℚ) (d : ℚ) : 
  n = 10 → 
  (4 * a₁ + 6 * d = 3) → 
  (3 * a₁ + 24 * d = 4) → 
  d = 7/78 :=
by sorry

end gold_distribution_l3883_388381


namespace percent_relation_l3883_388310

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : z = 1.2 * x) : 
  y = 0.75 * x := by
sorry

end percent_relation_l3883_388310


namespace rolling_cone_surface_area_l3883_388377

/-- The surface area described by the height of a rolling cone -/
theorem rolling_cone_surface_area (h l : ℝ) (h_pos : 0 < h) (l_pos : 0 < l) :
  let surface_area := π * h^3 / l
  surface_area = π * h^3 / l :=
by sorry

end rolling_cone_surface_area_l3883_388377


namespace train_bridge_crossing_time_l3883_388302

/-- Proves that a train of given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 255) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l3883_388302


namespace volleyball_points_product_l3883_388352

def first_10_games : List ℕ := [5, 6, 4, 7, 5, 6, 2, 3, 4, 9]

def total_first_10 : ℕ := first_10_games.sum

theorem volleyball_points_product :
  ∀ (points_11 points_12 : ℕ),
    points_11 < 15 →
    points_12 < 15 →
    (total_first_10 + points_11) % 11 = 0 →
    (total_first_10 + points_11 + points_12) % 12 = 0 →
    points_11 * points_12 = 20 := by
sorry

end volleyball_points_product_l3883_388352


namespace largest_digit_sum_is_8_l3883_388382

/-- Represents a three-digit decimal as a fraction 1/y where y is an integer between 1 and 16 -/
def IsValidFraction (a b c : ℕ) (y : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  1 < y ∧ y ≤ 16 ∧
  (100 * a + 10 * b + c : ℚ) / 1000 = 1 / y

/-- The sum of digits a, b, and c is at most 8 given the conditions -/
theorem largest_digit_sum_is_8 :
  ∀ a b c y : ℕ, IsValidFraction a b c y → a + b + c ≤ 8 :=
by sorry

end largest_digit_sum_is_8_l3883_388382


namespace iris_rose_ratio_l3883_388342

/-- Proves that given an initial ratio of irises to roses of 2:5, 
    with 25 roses initially and 20 roses added, 
    maintaining the same ratio results in a total of 18 irises. -/
theorem iris_rose_ratio (initial_roses : ℕ) (added_roses : ℕ) 
  (iris_ratio : ℕ) (rose_ratio : ℕ) : 
  initial_roses = 25 →
  added_roses = 20 →
  iris_ratio = 2 →
  rose_ratio = 5 →
  (iris_ratio : ℚ) / rose_ratio * (initial_roses + added_roses) = 18 := by
  sorry

#check iris_rose_ratio

end iris_rose_ratio_l3883_388342


namespace lecture_orderings_l3883_388384

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of valid orderings for n lecturers with k lecturers having specific ordering constraints --/
def validOrderings (n k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1)

/-- Theorem stating that the number of valid orderings for 7 lecturers with 3 having specific constraints is 120 --/
theorem lecture_orderings : validOrderings n k = 120 := by
  sorry

end lecture_orderings_l3883_388384


namespace point_on_circle_l3883_388315

/-- Given a circle C with maximum radius 2 containing points (2,y) and (-2,0),
    prove that the y-coordinate of (2,y) is 0 -/
theorem point_on_circle (y : ℝ) : 
  (∃ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    radius ≤ 2 ∧
    (2, y) ∈ C ∧
    (-2, 0) ∈ C ∧
    C = {p : ℝ × ℝ | dist p center = radius}) →
  y = 0 := by
  sorry

end point_on_circle_l3883_388315


namespace expression_values_l3883_388323

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = -5 ∨ expr = -1 ∨ expr = 1 ∨ expr = 5 := by
  sorry

end expression_values_l3883_388323


namespace arithmetic_sequence_n_is_27_l3883_388327

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first : a 1 = 20
  last : ∃ n : ℕ, a n = 54
  sum : ∃ n : ℕ, (n : ℝ) / 2 * (a 1 + a n) = 999

/-- The number of terms in the arithmetic sequence is 27 -/
theorem arithmetic_sequence_n_is_27 (seq : ArithmeticSequence) : 
  ∃ n : ℕ, n = 27 ∧ seq.a n = 54 ∧ (n : ℝ) / 2 * (seq.a 1 + seq.a n) = 999 := by
  sorry

end arithmetic_sequence_n_is_27_l3883_388327
