import Mathlib

namespace solution_set_and_inequality_l1319_131983

def f (x : ℝ) := -x + |2*x + 1|

def M : Set ℝ := {x | f x < 2}

theorem solution_set_and_inequality :
  (M = {x : ℝ | -1 < x ∧ x < 1}) ∧
  (∀ a b : ℝ, a ∈ M → b ∈ M → 2 * |a * b| + 1 > |a| + |b|) := by sorry

end solution_set_and_inequality_l1319_131983


namespace darnel_running_results_l1319_131993

/-- Represents Darnel's running activities --/
structure RunningActivity where
  sprint1 : Real
  sprint2 : Real
  jog1 : Real
  jog2 : Real
  walk : Real

/-- Calculates the total distance covered in all activities --/
def totalDistance (activity : RunningActivity) : Real :=
  activity.sprint1 + activity.sprint2 + activity.jog1 + activity.jog2 + activity.walk

/-- Calculates the additional distance sprinted compared to jogging and walking --/
def additionalSprint (activity : RunningActivity) : Real :=
  (activity.sprint1 + activity.sprint2) - (activity.jog1 + activity.jog2 + activity.walk)

/-- Theorem stating the total distance and additional sprint for Darnel's activities --/
theorem darnel_running_results (activity : RunningActivity)
  (h1 : activity.sprint1 = 0.88)
  (h2 : activity.sprint2 = 1.12)
  (h3 : activity.jog1 = 0.75)
  (h4 : activity.jog2 = 0.45)
  (h5 : activity.walk = 0.32) :
  totalDistance activity = 3.52 ∧ additionalSprint activity = 0.48 := by
  sorry

end darnel_running_results_l1319_131993


namespace min_value_reciprocal_sum_l1319_131981

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 9 / y) ≥ 16 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 9 / y = 16 :=
sorry

end min_value_reciprocal_sum_l1319_131981


namespace lines_parallel_iff_a_eq_one_l1319_131947

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_slopes_equal {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The lines ax + 2y - 2 = 0 and x + (a+1)y + 1 = 0 are parallel if and only if a = 1 -/
theorem lines_parallel_iff_a_eq_one (a : ℝ) :
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, 
    (a * x + 2 * y - 2 = 0 ↔ y = (-a/2) * x + b1) ∧ 
    (x + (a+1) * y + 1 = 0 ↔ y = (-1/(a+1)) * x + b2)) ↔ 
  a = 1 :=
sorry

end lines_parallel_iff_a_eq_one_l1319_131947


namespace multiplicative_inverse_203_mod_397_l1319_131974

theorem multiplicative_inverse_203_mod_397 : ∃ x : ℤ, 0 ≤ x ∧ x < 397 ∧ (203 * x) % 397 = 1 :=
by
  use 309
  sorry

end multiplicative_inverse_203_mod_397_l1319_131974


namespace parallelogram_diagonal_triangle_area_l1319_131984

/-- Given a parallelogram with area 128 square meters, the area of a triangle formed by its diagonal is 64 square meters. -/
theorem parallelogram_diagonal_triangle_area (P : Real) (h : P = 128) : P / 2 = 64 := by
  sorry

end parallelogram_diagonal_triangle_area_l1319_131984


namespace factorization_equality_l1319_131945

theorem factorization_equality (a b : ℝ) : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) := by
  sorry

end factorization_equality_l1319_131945


namespace min_side_length_is_optimal_l1319_131972

/-- The minimum side length of a square satisfying the given conditions -/
def min_side_length : ℝ := 25

/-- The condition that the area of the square is at least 625 square feet -/
def area_condition (s : ℝ) : Prop := s^2 ≥ 625

/-- The condition that there exists a smaller square inside with side length equal to half the side length of the larger square -/
def inner_square_condition (s : ℝ) : Prop := ∃ (inner_s : ℝ), inner_s = s / 2

/-- Theorem stating that the minimum side length satisfies both conditions and is minimal -/
theorem min_side_length_is_optimal :
  area_condition min_side_length ∧
  inner_square_condition min_side_length ∧
  ∀ s, s < min_side_length → ¬(area_condition s ∧ inner_square_condition s) :=
by sorry

end min_side_length_is_optimal_l1319_131972


namespace intersection_implies_x_value_l1319_131944

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, 2*x, x^2}

theorem intersection_implies_x_value :
  ∀ x : ℝ, A x ∩ B x = {1, 4} → x = -2 := by sorry

end intersection_implies_x_value_l1319_131944


namespace fundraiser_group_composition_l1319_131997

theorem fundraiser_group_composition (p : ℕ) : 
  (∃ (initial_girls : ℕ),
    -- Initial condition: 30% of the group are girls
    initial_girls = (3 * p) / 10 ∧
    -- After changes: 25% of the group are girls
    (initial_girls - 3 : ℚ) / (p + 2) = 1 / 4 →
    -- Prove that the initial number of girls was 21
    initial_girls = 21) :=
by sorry

end fundraiser_group_composition_l1319_131997


namespace circle_radius_is_sqrt_two_l1319_131949

/-- A circle inside a right angle with specific properties -/
structure CircleInRightAngle where
  /-- The radius of the circle -/
  R : ℝ
  /-- The length of chord AB -/
  AB : ℝ
  /-- The length of chord CD -/
  CD : ℝ
  /-- The circle is inside a right angle -/
  inside_right_angle : True
  /-- The circle is tangent to one side of the angle -/
  tangent_to_side : True
  /-- The circle intersects the other side at points A and B -/
  intersects_side : True
  /-- The circle intersects the angle bisector at points C and D -/
  intersects_bisector : True
  /-- AB = √6 -/
  h_AB : AB = Real.sqrt 6
  /-- CD = √7 -/
  h_CD : CD = Real.sqrt 7

/-- The theorem stating that the radius of the circle is √2 -/
theorem circle_radius_is_sqrt_two (c : CircleInRightAngle) : c.R = Real.sqrt 2 := by
  sorry

end circle_radius_is_sqrt_two_l1319_131949


namespace perpendicular_lines_m_values_l1319_131931

/-- Given two lines l₁ and l₂ in the form ax + by + c = 0, 
    this function returns true if they are perpendicular. -/
def are_perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

/-- Theorem stating that for lines l₁: (m+2)x-(m-2)y+2=0 and l₂: 3x+my-1=0,
    if they are perpendicular, then m = 6 or m = -1 -/
theorem perpendicular_lines_m_values :
  ∀ m : ℝ, are_perpendicular (m + 2) (-(m - 2)) 3 m → m = 6 ∨ m = -1 := by
  sorry

#check perpendicular_lines_m_values

end perpendicular_lines_m_values_l1319_131931


namespace tic_tac_toe_strategy_l1319_131904

/-- Represents a 10x10 tic-tac-toe board -/
def Board := Fin 10 → Fin 10 → Bool

/-- Counts the number of sets of five consecutive marks for a player -/
def count_sets (b : Board) (player : Bool) : ℕ := sorry

/-- Calculates the score for the first player (X) -/
def score (b : Board) : ℤ :=
  (count_sets b true : ℤ) - (count_sets b false : ℤ)

/-- A strategy for a player -/
def Strategy := Board → Fin 10 × Fin 10

/-- Applies a strategy to a board, returning the updated board -/
def apply_strategy (b : Board) (s : Strategy) (player : Bool) : Board := sorry

/-- Represents a full game play -/
def play_game (s1 s2 : Strategy) : Board := sorry

theorem tic_tac_toe_strategy :
  (∃ (s : Strategy), ∀ (s2 : Strategy), score (play_game s s2) ≥ 0) ∧
  (¬ ∃ (s : Strategy), ∀ (s2 : Strategy), score (play_game s s2) > 0) :=
sorry

end tic_tac_toe_strategy_l1319_131904


namespace ellipse_focal_property_l1319_131933

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The focus of the ellipse -/
def F : ℝ × ℝ := (2, 0)

/-- The point P -/
def P (p : ℝ) : ℝ × ℝ := (p, 0)

/-- A chord AB passing through F -/
def chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ∃ (t : ℝ), (1 - t) • A + t • B = F

/-- The angle equality condition -/
def angle_equality (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  let P := (p, 0)
  let AP := (A.1 - p, A.2)
  let BP := (B.1 - p, B.2)
  let PF := (2 - p, 0)
  AP.1 * PF.1 + AP.2 * PF.2 = BP.1 * PF.1 + BP.2 * PF.2

/-- The main theorem -/
theorem ellipse_focal_property :
  ∃! (p : ℝ), p > 0 ∧
    (∀ A B : ℝ × ℝ, chord A B → angle_equality p A B) ∧
    p = 3 := by sorry

end ellipse_focal_property_l1319_131933


namespace arithmetic_sequence_property_l1319_131978

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 100) :
  2 * a 9 - a 10 = 20 :=
sorry

end arithmetic_sequence_property_l1319_131978


namespace principal_amount_is_16065_l1319_131930

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem: Given the specified conditions, the principal amount is 16065 -/
theorem principal_amount_is_16065 :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 5
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 16065 := by
  sorry

end principal_amount_is_16065_l1319_131930


namespace mod_equivalence_l1319_131957

theorem mod_equivalence (n : ℕ) : 
  (179 * 933 / 7) % 50 = n ∧ 0 ≤ n ∧ n < 50 → n = 1 := by sorry

end mod_equivalence_l1319_131957


namespace fran_speed_l1319_131996

/-- Given Joann's bike ride parameters and Fran's time, calculate Fran's required speed --/
theorem fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) :
  (joann_speed * joann_time) / fran_time = 60 / 3.5 := by
  sorry

end fran_speed_l1319_131996


namespace area_of_triangle_BXC_l1319_131937

/-- Represents a trapezoid with bases and area -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Represents the triangle formed by the intersection of diagonals -/
structure DiagonalTriangle where
  area : ℝ

/-- Theorem stating the area of triangle BXC in the given trapezoid -/
theorem area_of_triangle_BXC (trapezoid : Trapezoid) (triangle : DiagonalTriangle) :
  trapezoid.base1 = 15 ∧ 
  trapezoid.base2 = 35 ∧ 
  trapezoid.area = 375 →
  triangle.area = 78.75 := by sorry

end area_of_triangle_BXC_l1319_131937


namespace seven_lines_regions_l1319_131991

/-- The number of regions created by n lines in a plane, where no two lines are parallel and no three lines meet at a single point -/
def num_regions (n : ℕ) : ℕ := 1 + n + n * (n - 1) / 2

/-- Seven lines in a plane with the given conditions divide the plane into 29 regions -/
theorem seven_lines_regions : num_regions 7 = 29 := by
  sorry

end seven_lines_regions_l1319_131991


namespace intersection_at_midpoint_l1319_131970

/-- A line with equation x - y = c intersects the line segment from (1, 4) to (3, 8) at its midpoint -/
theorem intersection_at_midpoint (c : ℝ) : 
  (∃ (x y : ℝ), x - y = c ∧ 
    x = (1 + 3) / 2 ∧ 
    y = (4 + 8) / 2 ∧ 
    (x, y) = ((1 + 3) / 2, (4 + 8) / 2)) → 
  c = -4 := by
sorry

end intersection_at_midpoint_l1319_131970


namespace part1_part2_l1319_131907

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |2*x + a|

-- Part 1: Prove that for a=1, f(x) + |x-1| ≥ 3 for all x
theorem part1 : ∀ x : ℝ, f 1 x + |x - 1| ≥ 3 := by sorry

-- Part 2: Prove that the minimum value of f(x) is 2 if and only if a = 2 or a = -6
theorem part2 : (∃ x : ℝ, f a x = 2) ∧ (∀ y : ℝ, f a y ≥ 2) ↔ a = 2 ∨ a = -6 := by sorry

end part1_part2_l1319_131907


namespace smallest_integer_side_of_triangle_l1319_131901

theorem smallest_integer_side_of_triangle (s : ℕ) : 
  (4 : ℝ) ≤ s ∧ 
  (7.8 : ℝ) + s > 11 ∧ 
  (7.8 : ℝ) + 11 > s ∧ 
  11 + s > (7.8 : ℝ) ∧
  ∀ (t : ℕ), t < s → 
    ((7.8 : ℝ) + (t : ℝ) ≤ 11 ∨ 
     (7.8 : ℝ) + 11 ≤ (t : ℝ) ∨ 
     11 + (t : ℝ) ≤ (7.8 : ℝ)) :=
by sorry

end smallest_integer_side_of_triangle_l1319_131901


namespace point_C_coordinates_l1319_131992

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle bisector line
def AngleBisector : ℝ → ℝ := fun x ↦ 2 * x

-- Define the condition that y=2x is the angle bisector of ∠C
def IsAngleBisector (t : Triangle) : Prop :=
  AngleBisector (t.C.1) = t.C.2

theorem point_C_coordinates (t : Triangle) :
  t.A = (-4, 2) →
  t.B = (3, 1) →
  IsAngleBisector t →
  t.C = (2, 4) := by
  sorry


end point_C_coordinates_l1319_131992


namespace cubic_real_root_l1319_131912

/-- Given a cubic polynomial ax^3 + 3x^2 + bx - 65 = 0 where a and b are real numbers,
    and -2 - 3i is one of its roots, the real root of this polynomial is 5/2. -/
theorem cubic_real_root (a b : ℝ) :
  (∃ (z : ℂ), z = -2 - 3*I ∧ a * z^3 + 3 * z^2 + b * z - 65 = 0) →
  (∃ (x : ℝ), a * x^3 + 3 * x^2 + b * x - 65 = 0 ∧ x = 5/2) :=
by sorry

end cubic_real_root_l1319_131912


namespace candy_eating_problem_l1319_131940

/-- Represents the number of candies eaten by each person -/
structure CandyEaten where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the rates at which each person eats candies -/
structure EatingRates where
  andrey : ℚ
  boris : ℚ
  denis : ℚ

/-- The theorem statement based on the given problem -/
theorem candy_eating_problem (rates : EatingRates) 
  (h1 : rates.andrey * 4 = rates.boris * 3)
  (h2 : rates.denis * 6 = rates.andrey * 7)
  (h3 : rates.andrey + rates.boris + rates.denis = 70) :
  ∃ (eaten : CandyEaten), 
    eaten.andrey = 24 ∧ 
    eaten.boris = 18 ∧ 
    eaten.denis = 28 ∧
    eaten.andrey + eaten.boris + eaten.denis = 70 := by
  sorry

end candy_eating_problem_l1319_131940


namespace exists_student_with_sqrt_k_classes_l1319_131938

/-- Represents a school with students and classes -/
structure School where
  n : ℕ  -- number of classes
  k : ℕ  -- number of students
  shared_class : Fin k → Fin k → Fin n
  class_size : Fin n → ℕ
  h1 : ∀ i j, i ≠ j → shared_class i j = shared_class j i
  h2 : ∀ i, class_size i < k
  h3 : ¬ ∃ m, k - 1 = m * m

/-- The number of classes a student attends -/
def num_classes_attended (s : School) (student : Fin s.k) : ℕ :=
  (Finset.univ.filter (λ c : Fin s.n => ∃ other, s.shared_class student other = c)).card

/-- Main theorem: There exists a student who has attended at least √k classes -/
theorem exists_student_with_sqrt_k_classes (s : School) :
  ∃ student : Fin s.k, s.k.sqrt ≤ num_classes_attended s student := by
  sorry

end exists_student_with_sqrt_k_classes_l1319_131938


namespace purely_imaginary_complex_number_l1319_131953

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - m) : ℂ) + m * I).re = 0 → m = 1 :=
by sorry

end purely_imaginary_complex_number_l1319_131953


namespace zach_stadium_goal_years_l1319_131948

/-- The number of years required to save enough money to visit all major league baseball stadiums. -/
def years_to_visit_stadiums (num_stadiums : ℕ) (cost_per_stadium : ℕ) (annual_savings : ℕ) : ℕ :=
  (num_stadiums * cost_per_stadium) / annual_savings

/-- Theorem stating that it takes 18 years to save enough money to visit all 30 major league baseball stadiums
    given an average cost of $900 per stadium and annual savings of $1,500. -/
theorem zach_stadium_goal_years :
  years_to_visit_stadiums 30 900 1500 = 18 := by
  sorry

end zach_stadium_goal_years_l1319_131948


namespace elsas_marbles_l1319_131962

theorem elsas_marbles (initial : ℕ) (lost_breakfast : ℕ) (given_lunch : ℕ) (received_mom : ℕ) : 
  initial = 40 →
  lost_breakfast = 3 →
  given_lunch = 5 →
  received_mom = 12 →
  initial - lost_breakfast - given_lunch + received_mom + 2 * given_lunch = 54 := by
  sorry

end elsas_marbles_l1319_131962


namespace leahs_garden_darker_tiles_l1319_131986

/-- Represents a square garden with a symmetrical tile pattern -/
structure SymmetricalGarden where
  -- The size of the repeating block
  block_size : ℕ
  -- The size of the center square in each block
  center_size : ℕ
  -- The number of darker tiles in the center square
  darker_tiles_in_center : ℕ

/-- The fraction of darker tiles in the garden -/
def fraction_of_darker_tiles (g : SymmetricalGarden) : ℚ :=
  (g.darker_tiles_in_center * (g.block_size / g.center_size)^2 : ℚ) / g.block_size^2

/-- Theorem stating the fraction of darker tiles in Leah's garden -/
theorem leahs_garden_darker_tiles :
  ∃ (g : SymmetricalGarden), 
    g.block_size = 4 ∧ 
    g.center_size = 2 ∧ 
    g.darker_tiles_in_center = 3 ∧ 
    fraction_of_darker_tiles g = 3/4 := by
  sorry

end leahs_garden_darker_tiles_l1319_131986


namespace quadratic_sum_l1319_131914

/-- Given a quadratic expression 4x^2 - 8x + 5, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals 6 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (4 * x^2 - 8 * x + 5 = a * (x - h)^2 + k) ∧ (a + h + k = 6) := by
sorry

end quadratic_sum_l1319_131914


namespace quadratic_root_ratio_l1319_131939

theorem quadratic_root_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end quadratic_root_ratio_l1319_131939


namespace quadratic_equation_solution_l1319_131999

theorem quadratic_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 5 * x^2 + 8 * x - 24 = 0 ∧ x = 6/5 := by
  sorry

end quadratic_equation_solution_l1319_131999


namespace division_problem_l1319_131929

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 52 → 
  divisor = 3 → 
  remainder = 4 → 
  dividend = divisor * quotient + remainder →
  quotient = 16 := by
sorry

end division_problem_l1319_131929


namespace smallest_greater_discount_l1319_131958

def discount_sequence_1 (x : ℝ) : ℝ := (1 - 0.2) * (1 - 0.1) * x
def discount_sequence_2 (x : ℝ) : ℝ := (1 - 0.08)^3 * x
def discount_sequence_3 (x : ℝ) : ℝ := (1 - 0.15) * (1 - 0.12) * x

def effective_discount_1 (x : ℝ) : ℝ := x - discount_sequence_1 x
def effective_discount_2 (x : ℝ) : ℝ := x - discount_sequence_2 x
def effective_discount_3 (x : ℝ) : ℝ := x - discount_sequence_3 x

theorem smallest_greater_discount : 
  ∀ x > 0, 
    effective_discount_1 x / x < 0.29 ∧ 
    effective_discount_2 x / x < 0.29 ∧ 
    effective_discount_3 x / x < 0.29 ∧
    ∀ n : ℕ, n < 29 → 
      (effective_discount_1 x / x > n / 100 ∨ 
       effective_discount_2 x / x > n / 100 ∨ 
       effective_discount_3 x / x > n / 100) :=
by sorry

end smallest_greater_discount_l1319_131958


namespace division_problem_l1319_131935

theorem division_problem (n x : ℝ) (h1 : n = 4.5) (h2 : (n / x) * 12 = 9) : x = 6 := by
  sorry

end division_problem_l1319_131935


namespace quadratic_ratio_l1319_131946

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 800*x + 2400

-- Define the completed square form
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

-- Theorem statement
theorem quadratic_ratio :
  ∃ (b c : ℝ), (∀ x, f x = g x b c) ∧ (c / b = -394) := by
  sorry

end quadratic_ratio_l1319_131946


namespace product_and_reciprocal_relation_l1319_131911

theorem product_and_reciprocal_relation (x y : ℝ) :
  x > 0 ∧ y > 0 ∧ x * y = 12 ∧ 1 / x = 3 * (1 / y) → x + y = 8 := by
  sorry

end product_and_reciprocal_relation_l1319_131911


namespace matildas_father_chocolates_l1319_131977

/-- Calculates the number of chocolate bars Matilda's father had left -/
def fathersRemainingChocolates (initialBars : ℕ) (people : ℕ) (givenToMother : ℕ) (eaten : ℕ) : ℕ :=
  let barsPerPerson := initialBars / people
  let givenToFather := people * (barsPerPerson / 2)
  givenToFather - givenToMother - eaten

/-- Proves that Matilda's father had 5 chocolate bars left -/
theorem matildas_father_chocolates :
  fathersRemainingChocolates 20 5 3 2 = 5 := by
  sorry

#eval fathersRemainingChocolates 20 5 3 2

end matildas_father_chocolates_l1319_131977


namespace guitar_ratio_l1319_131919

/-- The ratio of Davey's guitars to Barbeck's guitars is 1:1 -/
theorem guitar_ratio (davey barbeck : ℕ) : 
  davey = 18 → davey = barbeck → davey / barbeck = 1 := by
  sorry

end guitar_ratio_l1319_131919


namespace total_milk_poured_l1319_131975

/-- Represents a bottle with a certain capacity -/
structure Bottle where
  capacity : ℝ

/-- Represents the amount of milk poured into a bottle -/
def pour (b : Bottle) (fraction : ℝ) : ℝ := b.capacity * fraction

theorem total_milk_poured (bottle1 bottle2 : Bottle) 
  (h1 : bottle1.capacity = 4)
  (h2 : bottle2.capacity = 8)
  (h3 : pour bottle2 (5.333333333333333 / bottle2.capacity) = 5.333333333333333) :
  pour bottle1 (5.333333333333333 / bottle2.capacity) + 
  pour bottle2 (5.333333333333333 / bottle2.capacity) = 8 := by
sorry

end total_milk_poured_l1319_131975


namespace product_sum_relation_l1319_131928

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 1 → b = 7 → b - a = 4 := by
  sorry

end product_sum_relation_l1319_131928


namespace katie_sugar_calculation_l1319_131980

/-- Given a recipe that requires a total amount of sugar and an amount already added,
    calculate the remaining amount needed. -/
def remaining_sugar (total : ℝ) (added : ℝ) : ℝ :=
  total - added

theorem katie_sugar_calculation :
  let total_required : ℝ := 3
  let already_added : ℝ := 0.5
  remaining_sugar total_required already_added = 2.5 := by
sorry

end katie_sugar_calculation_l1319_131980


namespace alyssas_attended_games_l1319_131906

theorem alyssas_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 31) 
  (h2 : missed_games = 18) : 
  total_games - missed_games = 13 := by
  sorry

end alyssas_attended_games_l1319_131906


namespace no_tie_in_total_hr_l1319_131950

/-- Represents the months of the baseball season -/
inductive Month
| Mar
| Apr
| May
| Jun
| Jul
| Aug
| Sep

/-- Returns the number of home runs hit by Johnson in a given month -/
def johnson_hr (m : Month) : ℕ :=
  match m with
  | Month.Mar => 2
  | Month.Apr => 11
  | Month.May => 15
  | Month.Jun => 9
  | Month.Jul => 7
  | Month.Aug => 12
  | Month.Sep => 14

/-- Returns the number of home runs hit by Carter in a given month -/
def carter_hr (m : Month) : ℕ :=
  match m with
  | Month.Mar => 0
  | Month.Apr => 5
  | Month.May => 8
  | Month.Jun => 18
  | Month.Jul => 6
  | Month.Aug => 15
  | Month.Sep => 10

/-- Calculates the cumulative home runs for a player up to and including a given month -/
def cumulative_hr (hr_func : Month → ℕ) (m : Month) : ℕ :=
  match m with
  | Month.Mar => hr_func Month.Mar
  | Month.Apr => hr_func Month.Mar + hr_func Month.Apr
  | Month.May => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May
  | Month.Jun => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun
  | Month.Jul => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul
  | Month.Aug => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul + hr_func Month.Aug
  | Month.Sep => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul + hr_func Month.Aug + hr_func Month.Sep

theorem no_tie_in_total_hr : ∀ m : Month, cumulative_hr johnson_hr m ≠ cumulative_hr carter_hr m := by
  sorry

end no_tie_in_total_hr_l1319_131950


namespace taxi_overtakes_bus_l1319_131924

/-- 
Given a taxi and a bus with the following conditions:
- The taxi travels at 45 mph
- The bus travels 30 mph slower than the taxi
- The taxi starts 4 hours after the bus
This theorem proves that the taxi will overtake the bus in 2 hours.
-/
theorem taxi_overtakes_bus (taxi_speed : ℝ) (bus_speed : ℝ) (head_start : ℝ) 
  (overtake_time : ℝ) :
  taxi_speed = 45 →
  bus_speed = taxi_speed - 30 →
  head_start = 4 →
  overtake_time = 2 →
  taxi_speed * overtake_time = bus_speed * (overtake_time + head_start) :=
by
  sorry

#check taxi_overtakes_bus

end taxi_overtakes_bus_l1319_131924


namespace division_remainder_problem_l1319_131916

theorem division_remainder_problem (D : ℕ) : 
  (D / 12 = 70) → (D / 21 = 40) → (D % 21 = 0) :=
by
  sorry

end division_remainder_problem_l1319_131916


namespace spring_properties_l1319_131961

-- Define the spring's properties
def initial_length : ℝ := 18
def extension_rate : ℝ := 2

-- Define the relationship between mass and length
def spring_length (mass : ℝ) : ℝ := initial_length + extension_rate * mass

theorem spring_properties :
  (spring_length 4 = 26) ∧
  (∀ x y, x < y → spring_length x < spring_length y) ∧
  (∀ x, spring_length x = 2 * x + 18) ∧
  (spring_length 12 = 42) := by
  sorry

end spring_properties_l1319_131961


namespace diagonals_15_gon_l1319_131905

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex 15-gon is 90 -/
theorem diagonals_15_gon : num_diagonals 15 = 90 := by
  sorry

end diagonals_15_gon_l1319_131905


namespace constant_remainder_condition_l1319_131900

-- Define the polynomials
def dividend (a : ℝ) (x : ℝ) : ℝ := 12 * x^3 - 9 * x^2 + a * x + 8
def divisor (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the theorem
theorem constant_remainder_condition (a : ℝ) :
  (∃ (r : ℝ), ∀ (x : ℝ), ∃ (q : ℝ), dividend a x = divisor x * q + r) ↔ a = -7 := by
  sorry

end constant_remainder_condition_l1319_131900


namespace brick_length_is_25_cm_l1319_131917

/-- Proves that the length of each brick is 25 cm, given the wall dimensions,
    brick dimensions (except length), and the number of bricks needed. -/
theorem brick_length_is_25_cm
  (wall_length : ℝ)
  (wall_height : ℝ)
  (wall_thickness : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (num_bricks : ℝ)
  (h_wall_length : wall_length = 2)
  (h_wall_height : wall_height = 3)
  (h_wall_thickness : wall_thickness = 0.02)
  (h_brick_width : brick_width = 0.11)
  (h_brick_height : brick_height = 0.06)
  (h_num_bricks : num_bricks = 72.72727272727273)
  : ∃ (brick_length : ℝ), brick_length = 0.25 :=
by sorry

end brick_length_is_25_cm_l1319_131917


namespace survey_sample_size_l1319_131910

/-- Represents a survey conducted on students -/
structure Survey where
  numSelected : ℕ

/-- Definition of sample size for a survey -/
def sampleSize (s : Survey) : ℕ := s.numSelected

/-- Theorem stating that the sample size of the survey is 200 -/
theorem survey_sample_size :
  ∃ (s : Survey), s.numSelected = 200 ∧ sampleSize s = 200 := by
  sorry

end survey_sample_size_l1319_131910


namespace black_balls_count_l1319_131956

theorem black_balls_count (red_balls : ℕ) (prob_red : ℚ) (black_balls : ℕ) : 
  red_balls = 3 → prob_red = 1/4 → black_balls = 9 → 
  (red_balls : ℚ) / (red_balls + black_balls : ℚ) = prob_red :=
by sorry

end black_balls_count_l1319_131956


namespace solve_equation_l1319_131988

theorem solve_equation : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ∧ x = 11 := by
  sorry

end solve_equation_l1319_131988


namespace nuts_per_student_l1319_131926

theorem nuts_per_student (bags : ℕ) (students : ℕ) (nuts_per_bag : ℕ) 
  (h1 : bags = 65) 
  (h2 : students = 13) 
  (h3 : nuts_per_bag = 15) : 
  (bags * nuts_per_bag) / students = 75 := by
sorry

end nuts_per_student_l1319_131926


namespace set_intersection_equality_l1319_131941

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- State the theorem
theorem set_intersection_equality : M ∩ N = {y | y ≥ 1} := by sorry

end set_intersection_equality_l1319_131941


namespace first_half_rate_is_5_4_l1319_131954

/-- Represents a cricket game with two halves --/
structure CricketGame where
  total_overs : ℕ
  target_runs : ℕ
  second_half_rate : ℚ

/-- Calculates the run rate for the first half of the game --/
def first_half_run_rate (game : CricketGame) : ℚ :=
  let first_half_overs : ℚ := game.total_overs / 2
  let second_half_runs : ℚ := game.second_half_rate * first_half_overs
  let first_half_runs : ℚ := game.target_runs - second_half_runs
  first_half_runs / first_half_overs

/-- Theorem stating the first half run rate for the given game conditions --/
theorem first_half_rate_is_5_4 (game : CricketGame) 
    (h1 : game.total_overs = 50)
    (h2 : game.target_runs = 400)
    (h3 : game.second_half_rate = 53 / 5) : 
  first_half_run_rate game = 27 / 5 := by
  sorry

#eval (53 : ℚ) / 5  -- Outputs 10.6
#eval (27 : ℚ) / 5  -- Outputs 5.4

end first_half_rate_is_5_4_l1319_131954


namespace mn_positive_necessary_not_sufficient_l1319_131955

/-- A curve represented by the equation mx^2 + ny^2 = 1 is an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem mn_positive_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → m * n > 0) ∧
  (∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n) :=
sorry

end mn_positive_necessary_not_sufficient_l1319_131955


namespace car_catch_up_time_l1319_131925

/-- The time it takes for a car to catch up to a truck on a highway -/
theorem car_catch_up_time (truck_speed car_speed : ℝ) (head_start : ℝ) : 
  truck_speed = 45 →
  car_speed = 60 →
  head_start = 1 →
  ∃ t : ℝ, t = 6 ∧ car_speed * t = truck_speed * (t + head_start) + truck_speed * head_start :=
by
  sorry


end car_catch_up_time_l1319_131925


namespace conference_center_distance_l1319_131973

theorem conference_center_distance
  (initial_speed : ℝ)
  (initial_distance : ℝ)
  (late_time : ℝ)
  (speed_increase : ℝ)
  (early_time : ℝ)
  (h1 : initial_speed = 40)
  (h2 : initial_distance = 40)
  (h3 : late_time = 1.5)
  (h4 : speed_increase = 20)
  (h5 : early_time = 0.25)
  : ∃ (total_distance : ℝ), total_distance = 310 :=
by
  sorry

end conference_center_distance_l1319_131973


namespace rationalize_and_minimize_sum_l1319_131927

theorem rationalize_and_minimize_sum : ∃ (A B C D : ℕ),
  (D > 0) ∧
  (∀ (p : ℕ), Prime p → ¬(p^2 ∣ B)) ∧
  ((A : ℝ) * Real.sqrt B + C) / D = (Real.sqrt 32) / (Real.sqrt 16 - Real.sqrt 2) ∧
  (∀ (A' B' C' D' : ℕ),
    (D' > 0) →
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ B')) →
    ((A' : ℝ) * Real.sqrt B' + C') / D' = (Real.sqrt 32) / (Real.sqrt 16 - Real.sqrt 2) →
    A + B + C + D ≤ A' + B' + C' + D') ∧
  A + B + C + D = 21 :=
by sorry

end rationalize_and_minimize_sum_l1319_131927


namespace tan_sum_product_equality_l1319_131951

theorem tan_sum_product_equality (α β γ : ℝ) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_sum : α + β + γ = π / 2) : 
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end tan_sum_product_equality_l1319_131951


namespace largest_angle_in_triangle_l1319_131903

theorem largest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a + b = 105 →      -- Sum of two angles is 7/6 of a right angle (90° * 7/6 = 105°)
  b = a + 40 →       -- One angle is 40° larger than the other
  max a (max b c) = 75 := by
  sorry

end largest_angle_in_triangle_l1319_131903


namespace amy_blue_balloons_l1319_131979

theorem amy_blue_balloons :
  let total_balloons : ℕ := 67
  let red_balloons : ℕ := 29
  let green_balloons : ℕ := 17
  let blue_balloons : ℕ := total_balloons - red_balloons - green_balloons
  blue_balloons = 21 := by
  sorry

end amy_blue_balloons_l1319_131979


namespace statement_c_not_always_true_l1319_131976

theorem statement_c_not_always_true :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by sorry

end statement_c_not_always_true_l1319_131976


namespace photo_lineup_arrangements_l1319_131932

/-- The number of ways to arrange 4 boys and 3 girls in a line such that no two boys are adjacent -/
def boys_not_adjacent : ℕ := 144

/-- The number of ways to arrange 4 boys and 3 girls in a line such that no two girls are adjacent -/
def girls_not_adjacent : ℕ := 1440

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

theorem photo_lineup_arrangements :
  (boys_not_adjacent = 144) ∧ (girls_not_adjacent = 1440) := by
  sorry

end photo_lineup_arrangements_l1319_131932


namespace complex_equation_solution_l1319_131987

theorem complex_equation_solution (z : ℂ) 
  (h : 20 * Complex.abs z ^ 2 = 3 * Complex.abs (z + 3) ^ 2 + Complex.abs (z ^ 2 + 2) ^ 2 + 37) :
  z + 9 / z = -3 :=
by sorry

end complex_equation_solution_l1319_131987


namespace min_value_x_plus_2y_l1319_131902

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 3) : 
  ∀ z, x + 2*y ≥ z → z ≥ 2 :=
sorry

end min_value_x_plus_2y_l1319_131902


namespace find_t_l1319_131908

-- Define a decreasing function f on ℝ
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define the property that f passes through (0, 5) and (3, -1)
def passes_through_points (f : ℝ → ℝ) : Prop :=
  f 0 = 5 ∧ f 3 = -1

-- Define the solution set of |f(x+t)-2|<3
def solution_set (f : ℝ → ℝ) (t : ℝ) : Set ℝ :=
  {x : ℝ | |f (x + t) - 2| < 3}

-- State the theorem
theorem find_t (f : ℝ → ℝ) (t : ℝ) :
  is_decreasing f →
  passes_through_points f →
  solution_set f t = Set.Ioo (-1) 2 →
  t = 1 := by sorry

end find_t_l1319_131908


namespace positive_poly_nonneg_ratio_l1319_131995

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- A polynomial with nonnegative real coefficients -/
def NonnegPolynomial := {p : RealPolynomial // ∀ i, 0 ≤ p.coeff i}

/-- The theorem statement -/
theorem positive_poly_nonneg_ratio
  (P : RealPolynomial)
  (h : ∀ x : ℝ, 0 < x → 0 < P.eval x) :
  ∃ (Q R : NonnegPolynomial), ∀ x : ℝ, 0 < x →
    P.eval x = (Q.val.eval x) / (R.val.eval x) :=
sorry

end positive_poly_nonneg_ratio_l1319_131995


namespace sin_graph_transformation_l1319_131998

/-- 
Given two trigonometric functions f(x) = 3sin(2x - π/6) and g(x) = 3sin(x + π/2),
prove that the graph of g(x) can be obtained from the graph of f(x) by 
extending the x-coordinates to twice their original values and 
then shifting the resulting graph to the left by 2π/3 units.
-/
theorem sin_graph_transformation (x : ℝ) : 
  3 * Real.sin (x + π/2) = 3 * Real.sin ((2*x - π/6) / 2 + 2*π/3) := by
sorry

end sin_graph_transformation_l1319_131998


namespace part_one_part_two_l1319_131960

/-- Condition p: (x - a)(x - 3a) < 0 -/
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

/-- Condition q: (x - 3) / (x - 2) ≤ 0 -/
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

/-- Part 1: When a = 1 and p ∧ q is true, then 2 < x < 3 -/
theorem part_one (x : ℝ) (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

/-- Part 2: When p is necessary but not sufficient for q, and a > 0, then 1 ≤ a ≤ 2 -/
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, q x → p x a) 
  (h3 : ∃ x, p x a ∧ ¬q x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end part_one_part_two_l1319_131960


namespace berkeley_class_as_l1319_131968

theorem berkeley_class_as (abraham_total : ℕ) (abraham_as : ℕ) (berkeley_total : ℕ) :
  abraham_total = 20 →
  abraham_as = 12 →
  berkeley_total = 30 →
  (berkeley_total : ℚ) * (abraham_as : ℚ) / (abraham_total : ℚ) = 18 :=
by sorry

end berkeley_class_as_l1319_131968


namespace total_gold_value_l1319_131994

/-- Calculates the total value of gold for Legacy, Aleena, and Briana -/
theorem total_gold_value (legacy_bars : ℕ) (aleena_bars_diff : ℕ) (briana_bars : ℕ)
  (legacy_aleena_value : ℕ) (briana_value : ℕ) :
  legacy_bars = 12 →
  aleena_bars_diff = 4 →
  briana_bars = 8 →
  legacy_aleena_value = 3500 →
  briana_value = 4000 →
  (legacy_bars * legacy_aleena_value) +
  ((legacy_bars - aleena_bars_diff) * legacy_aleena_value) +
  (briana_bars * briana_value) = 102000 :=
by sorry

end total_gold_value_l1319_131994


namespace distribute_6_4_l1319_131942

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 4 indistinguishable boxes is 262 -/
theorem distribute_6_4 : distribute 6 4 = 262 := by sorry

end distribute_6_4_l1319_131942


namespace lanas_muffin_goal_l1319_131989

/-- Lana's muffin sale problem -/
theorem lanas_muffin_goal (morning_sales afternoon_sales more_needed : ℕ) 
  (h1 : morning_sales = 12)
  (h2 : afternoon_sales = 4)
  (h3 : more_needed = 4) :
  morning_sales + afternoon_sales + more_needed = 20 := by
  sorry

end lanas_muffin_goal_l1319_131989


namespace complex_equation_sum_l1319_131952

theorem complex_equation_sum (a b : ℝ) :
  (Complex.I : ℂ)⁻¹ * (a + Complex.I) = 1 + b * Complex.I → a + b = 0 := by
  sorry

end complex_equation_sum_l1319_131952


namespace number_thought_of_l1319_131964

theorem number_thought_of (x : ℝ) : (x / 5 + 8 = 61) → x = 265 := by
  sorry

end number_thought_of_l1319_131964


namespace terms_before_negative17_l1319_131965

/-- An arithmetic sequence with first term 103 and common difference -7 -/
def arithmeticSequence (n : ℕ) : ℤ := 103 - 7 * (n - 1)

/-- The position of -17 in the sequence -/
def positionOfNegative17 : ℕ := 18

theorem terms_before_negative17 :
  (∀ k < positionOfNegative17 - 1, arithmeticSequence k > -17) ∧
  arithmeticSequence positionOfNegative17 = -17 :=
sorry

end terms_before_negative17_l1319_131965


namespace team_selection_count_l1319_131922

def people : Finset Char := {'a', 'b', 'c', 'd', 'e'}

theorem team_selection_count :
  let all_selections := (people.powerset.filter (fun s => s.card = 2)).card
  let invalid_selections := (people.erase 'a').card
  all_selections - invalid_selections = 16 := by
  sorry

end team_selection_count_l1319_131922


namespace sphere_volume_from_surface_area_l1319_131936

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 324 * π →
    (4 / 3) * π * r^3 = 972 * π :=
by
  sorry

end sphere_volume_from_surface_area_l1319_131936


namespace midpoint_coordinates_l1319_131943

/-- Given two points M and N in a plane, and P as the midpoint of MN, 
    prove that P has the specified coordinates. -/
theorem midpoint_coordinates (M N P : ℝ × ℝ) : 
  M = (3, -2) → N = (-5, -1) → P = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) → 
  P = (-1, -3/2) := by
  sorry

end midpoint_coordinates_l1319_131943


namespace trapezium_height_l1319_131969

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 18) (harea : area = 475) :
  (2 * area) / (a + b) = 25 := by
  sorry

end trapezium_height_l1319_131969


namespace compound_interest_principal_is_5000_l1319_131985

-- Define the simple interest rate
def simple_interest_rate : ℝ := 0.10

-- Define the compound interest rate
def compound_interest_rate : ℝ := 0.12

-- Define the simple interest time period
def simple_interest_time : ℕ := 5

-- Define the compound interest time period
def compound_interest_time : ℕ := 2

-- Define the simple interest principal
def simple_interest_principal : ℝ := 1272

-- Define the function to calculate simple interest
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * (time : ℝ)

-- Define the function to calculate compound interest
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

-- Theorem to prove
theorem compound_interest_principal_is_5000 :
  ∃ (compound_principal : ℝ),
    simple_interest simple_interest_principal simple_interest_rate simple_interest_time =
    (1/2) * compound_interest compound_principal compound_interest_rate compound_interest_time ∧
    compound_principal = 5000 := by
  sorry

end compound_interest_principal_is_5000_l1319_131985


namespace base_four_of_156_l1319_131982

def base_four_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_four_of_156 :
  base_four_representation 156 = [2, 1, 3, 0] := by sorry

end base_four_of_156_l1319_131982


namespace quiz_competition_l1319_131915

theorem quiz_competition (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : correct_score = 10)
  (h3 : incorrect_score = -5)
  (h4 : total_score = 850) :
  ∃ (incorrect : ℕ), 
    incorrect = 10 ∧ 
    (total_questions - incorrect : ℤ) * correct_score + incorrect * incorrect_score = total_score :=
by sorry

end quiz_competition_l1319_131915


namespace quotient_reciprocal_sum_l1319_131966

theorem quotient_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 45) (hprod : x * y = 500) : 
  (x / y) + (y / x) = 41 / 20 := by
  sorry

end quotient_reciprocal_sum_l1319_131966


namespace system_solution_l1319_131934

theorem system_solution (a b c d e f : ℝ) 
  (eq1 : 4 * a = (b + c + d + e)^4)
  (eq2 : 4 * b = (c + d + e + f)^4)
  (eq3 : 4 * c = (d + e + f + a)^4)
  (eq4 : 4 * d = (e + f + a + b)^4)
  (eq5 : 4 * e = (f + a + b + c)^4)
  (eq6 : 4 * f = (a + b + c + d)^4) :
  a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4 ∧ e = 1/4 ∧ f = 1/4 := by
sorry

end system_solution_l1319_131934


namespace periodic_function_decomposition_l1319_131923

-- Define the type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define the property of being 2π-periodic
def isPeriodic2Pi (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x

-- Define the property of being an even function
def isEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the property of being π-periodic
def isPeriodicPi (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x + Real.pi) = f x

theorem periodic_function_decomposition (f : RealFunction) (h : isPeriodic2Pi f) :
  ∃ (f₁ f₂ f₃ f₄ : RealFunction),
    (∀ i ∈ [f₁, f₂, f₃, f₄], isEven i ∧ isPeriodicPi i) ∧
    (∀ x : ℝ, f x = f₁ x + f₂ x * Real.cos x + f₃ x * Real.sin x + f₄ x * Real.sin (2 * x)) :=
sorry

end periodic_function_decomposition_l1319_131923


namespace no_real_a_for_single_solution_l1319_131990

theorem no_real_a_for_single_solution :
  ¬ ∃ (a : ℝ), ∃! (x : ℝ), |x^2 + 4*a*x + 5*a| ≤ 3 := by
  sorry

end no_real_a_for_single_solution_l1319_131990


namespace inequality_solution_set_l1319_131913

-- Define the inequality
def inequality (x : ℝ) : Prop := (2 - x) / (x + 4) > 1

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < -1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end inequality_solution_set_l1319_131913


namespace product_of_roots_cubic_equation_l1319_131909

theorem product_of_roots_cubic_equation : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 4 * x^2 + x - 5
  ∃ a b c : ℝ, (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ a * b * c = 5/3 := by
  sorry

end product_of_roots_cubic_equation_l1319_131909


namespace additional_grazing_area_l1319_131918

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 18^2 - π * 12^2 = 180 * π := by
  sorry

end additional_grazing_area_l1319_131918


namespace dragon_legs_correct_l1319_131971

/-- Represents the number of legs of a three-headed dragon -/
def dragon_legs : ℕ := 14

/-- Represents the number of centipedes -/
def num_centipedes : ℕ := 5

/-- Represents the number of three-headed dragons -/
def num_dragons : ℕ := 7

/-- The total number of heads in the herd -/
def total_heads : ℕ := 26

/-- The total number of legs in the herd -/
def total_legs : ℕ := 298

/-- Each centipede has one head -/
def centipede_heads : ℕ := 1

/-- Each centipede has 40 legs -/
def centipede_legs : ℕ := 40

/-- Each dragon has three heads -/
def dragon_heads : ℕ := 3

theorem dragon_legs_correct :
  (num_centipedes * centipede_heads + num_dragons * dragon_heads = total_heads) ∧
  (num_centipedes * centipede_legs + num_dragons * dragon_legs = total_legs) :=
by sorry

end dragon_legs_correct_l1319_131971


namespace sphere_surface_area_l1319_131967

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * (Real.rpow 2 (1/3))^2 := by
  sorry

end sphere_surface_area_l1319_131967


namespace shopping_cost_calculation_l1319_131959

-- Define the prices and quantities
def carrot_price : ℚ := 2
def carrot_quantity : ℕ := 7
def milk_price : ℚ := 3
def milk_quantity : ℕ := 4
def pineapple_price : ℚ := 5
def pineapple_quantity : ℕ := 3
def pineapple_discount : ℚ := 0.5
def flour_price : ℚ := 8
def flour_quantity : ℕ := 1
def cookie_price : ℚ := 10
def cookie_quantity : ℕ := 1

-- Define the store's discount conditions
def store_discount_threshold : ℚ := 40
def store_discount_rate : ℚ := 0.1

-- Define the coupon conditions
def coupon_value : ℚ := 5
def coupon_threshold : ℚ := 25

-- Calculate the total cost before discounts
def total_before_discounts : ℚ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  pineapple_price * pineapple_quantity * (1 - pineapple_discount) +
  flour_price * flour_quantity +
  cookie_price * cookie_quantity

-- Apply store discount if applicable
def after_store_discount : ℚ :=
  if total_before_discounts > store_discount_threshold then
    total_before_discounts * (1 - store_discount_rate)
  else
    total_before_discounts

-- Apply coupon if applicable
def final_cost : ℚ :=
  if after_store_discount > coupon_threshold then
    after_store_discount - coupon_value
  else
    after_store_discount

-- Theorem to prove
theorem shopping_cost_calculation :
  final_cost = 41.35 := by sorry

end shopping_cost_calculation_l1319_131959


namespace range_of_x_range_of_a_l1319_131921

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 3*a*x + 2*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x (x : ℝ) (h : p x 2 ∨ q x) : 2 < x ∧ x < 4 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h' : ∃ x, ¬(p x a) ∧ q x) : 3/2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_x_range_of_a_l1319_131921


namespace isabela_spent_2800_l1319_131963

/-- The total amount Isabela spent on cucumbers and pencils -/
def total_spent (cucumber_price : ℝ) (pencil_price : ℝ) (cucumber_count : ℕ) 
  (pencil_discount : ℝ) : ℝ :=
  let pencil_count := cucumber_count / 2
  let pencil_cost := pencil_count * pencil_price * (1 - pencil_discount)
  let cucumber_cost := cucumber_count * cucumber_price
  pencil_cost + cucumber_cost

/-- Theorem stating that Isabela spent $2800 on cucumbers and pencils -/
theorem isabela_spent_2800 : 
  total_spent 20 20 100 0.2 = 2800 := by
  sorry

end isabela_spent_2800_l1319_131963


namespace book_spending_is_correct_l1319_131920

def allowance : ℚ := 50

def game_fraction : ℚ := 1/4
def snack_fraction : ℚ := 1/5
def toy_fraction : ℚ := 2/5

def book_spending : ℚ := allowance - (allowance * game_fraction + allowance * snack_fraction + allowance * toy_fraction)

theorem book_spending_is_correct : book_spending = 7.5 := by sorry

end book_spending_is_correct_l1319_131920
