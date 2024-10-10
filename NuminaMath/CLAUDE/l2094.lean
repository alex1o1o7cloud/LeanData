import Mathlib

namespace polynomial_simplification_l2094_209450

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 6 * x - 8) + (-5 * x^4 - 2 * x^3 + 4 * x^2 - 6 * x + 7) = 
  -3 * x^4 + x^3 - x^2 - 1 := by
  sorry

end polynomial_simplification_l2094_209450


namespace angle_measure_proof_l2094_209452

theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles are supplementary
  C = 7 * D →    -- C is 7 times D
  C = 157.5 :=   -- Measure of angle C
by sorry

end angle_measure_proof_l2094_209452


namespace cost_doubling_cost_percentage_increase_l2094_209484

theorem cost_doubling (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) : 
  t * (2 * b)^4 = 16 * (t * b^4) := by
  sorry

theorem cost_percentage_increase (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  (t * (2 * b)^4) / (t * b^4) * 100 = 1600 := by
  sorry

end cost_doubling_cost_percentage_increase_l2094_209484


namespace part_one_part_two_l2094_209477

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem part_one :
  ∀ x : ℝ, (1 < x ∧ x < 3) ∧ (2 < x ∧ x ≤ 3) ↔ (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, a > 0 →
  ((∀ x : ℝ, 2 < x ∧ x ≤ 3 → a < x ∧ x < 3*a) ∧
   (∃ x : ℝ, a < x ∧ x < 3*a ∧ ¬(2 < x ∧ x ≤ 3))) →
  (1 < a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l2094_209477


namespace sector_area_l2094_209481

/-- The area of a circular sector given its arc length and radius -/
theorem sector_area (l r : ℝ) (hl : l > 0) (hr : r > 0) : 
  (l * r) / 2 = (l * r) / 2 := by
  sorry

end sector_area_l2094_209481


namespace find_m_l2094_209440

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

-- Define the complement of A in U
def complement_A (m : ℕ) : Set ℕ := U \ A m

-- Theorem statement
theorem find_m : ∃ m : ℕ, complement_A m = {2, 3} ∧ m = 4 := by
  sorry

end find_m_l2094_209440


namespace definite_integral_exp_minus_2x_l2094_209463

theorem definite_integral_exp_minus_2x : 
  ∫ x in (0: ℝ)..1, (Real.exp x - 2 * x) = Real.exp 1 - 2 := by sorry

end definite_integral_exp_minus_2x_l2094_209463


namespace integers_between_sqrt3_and_sqrt13_two_and_three_between_sqrt3_and_sqrt13_only_two_and_three_between_sqrt3_and_sqrt13_l2094_209453

theorem integers_between_sqrt3_and_sqrt13 :
  ∃ (n : ℤ), (↑n : ℝ) > Real.sqrt 3 ∧ (↑n : ℝ) < Real.sqrt 13 :=
by
  sorry

theorem two_and_three_between_sqrt3_and_sqrt13 :
  (2 : ℝ) > Real.sqrt 3 ∧ (2 : ℝ) < Real.sqrt 13 ∧
  (3 : ℝ) > Real.sqrt 3 ∧ (3 : ℝ) < Real.sqrt 13 :=
by
  sorry

theorem only_two_and_three_between_sqrt3_and_sqrt13 :
  ∀ (n : ℤ), (↑n : ℝ) > Real.sqrt 3 ∧ (↑n : ℝ) < Real.sqrt 13 → n = 2 ∨ n = 3 :=
by
  sorry

end integers_between_sqrt3_and_sqrt13_two_and_three_between_sqrt3_and_sqrt13_only_two_and_three_between_sqrt3_and_sqrt13_l2094_209453


namespace ellipse_major_axis_length_l2094_209471

/-- The length of the major axis of the ellipse x^2 + 4y^2 = 100 is 20 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + 4*y^2 = 100}
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ ellipse ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    2*a = 20 :=
by sorry

end ellipse_major_axis_length_l2094_209471


namespace percent_relationship_l2094_209499

theorem percent_relationship (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) :
  y / x = 0.25 := by
sorry

end percent_relationship_l2094_209499


namespace expression_value_l2094_209430

theorem expression_value : ∃ x : ℕ, (8000 * 6000 : ℕ) = 480 * x ∧ x = 100000 := by
  sorry

end expression_value_l2094_209430


namespace monday_sunday_speed_ratio_l2094_209454

/-- Proves that the ratio of speeds on Monday (first 32 miles) to Sunday is 2:1 -/
theorem monday_sunday_speed_ratio 
  (total_distance : ℝ) 
  (sunday_speed : ℝ) 
  (monday_first_distance : ℝ) 
  (monday_first_speed : ℝ) :
  total_distance = 120 →
  monday_first_distance = 32 →
  (total_distance / sunday_speed) * 1.6 = 
    (monday_first_distance / monday_first_speed) + 
    ((total_distance - monday_first_distance) / (sunday_speed / 2)) →
  monday_first_speed / sunday_speed = 2 := by
  sorry

end monday_sunday_speed_ratio_l2094_209454


namespace largest_angle_right_in_special_triangle_l2094_209457

/-- A triangle with sides a, b, c and semiperimeter s -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  s : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  semiperimeter_def : s = (a + b + c) / 2

/-- The radii of the circles tangent to the sides of the triangle -/
structure TriangleRadii (t : Triangle) where
  r : ℝ  -- inradius
  ra : ℝ  -- exradius opposite to side a
  rb : ℝ  -- exradius opposite to side b
  rc : ℝ  -- exradius opposite to side c
  radii_relations : 
    t.s * r = (t.s - t.a) * ra ∧
    t.s * r = (t.s - t.b) * rb ∧
    t.s * r = (t.s - t.c) * rc

/-- The radii form a geometric progression -/
def radii_in_geometric_progression (t : Triangle) (tr : TriangleRadii t) : Prop :=
  ∃ q : ℝ, q > 1 ∧ tr.ra = q * tr.r ∧ tr.rb = q^2 * tr.r ∧ tr.rc = q^3 * tr.r

/-- The largest angle in a triangle is 90 degrees -/
def largest_angle_is_right (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

theorem largest_angle_right_in_special_triangle (t : Triangle) (tr : TriangleRadii t) 
  (h : radii_in_geometric_progression t tr) : 
  largest_angle_is_right t :=
sorry

end largest_angle_right_in_special_triangle_l2094_209457


namespace howlers_lineup_count_l2094_209437

def total_players : Nat := 15
def lineup_size : Nat := 6
def excluded_players : Nat := 3

theorem howlers_lineup_count :
  (Nat.choose (total_players - excluded_players) lineup_size) +
  (excluded_players * Nat.choose (total_players - excluded_players) (lineup_size - 1)) = 3300 :=
by sorry

end howlers_lineup_count_l2094_209437


namespace distance_after_two_hours_l2094_209436

/-- Anna's jogging speed in miles per minute -/
def anna_speed : ℚ := 1 / 20

/-- Mark's running speed in miles per minute -/
def mark_speed : ℚ := 3 / 40

/-- The time period in minutes -/
def time_period : ℚ := 2 * 60

/-- The theorem stating the distance between Anna and Mark after 2 hours -/
theorem distance_after_two_hours :
  anna_speed * time_period + mark_speed * time_period = 9 := by sorry

end distance_after_two_hours_l2094_209436


namespace not_passed_implies_not_all_correct_l2094_209433

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (passed : Student → Prop)
variable (answered_all_correctly : Student → Prop)

-- State the given condition
variable (h : ∀ s : Student, answered_all_correctly s → passed s)

-- Theorem to prove
theorem not_passed_implies_not_all_correct (s : Student) :
  ¬(passed s) → ¬(answered_all_correctly s) :=
by
  sorry


end not_passed_implies_not_all_correct_l2094_209433


namespace book_pages_l2094_209421

/-- The number of pages Ceasar has already read -/
def pages_read : ℕ := 147

/-- The number of pages Ceasar has left to read -/
def pages_left : ℕ := 416

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_read + pages_left

/-- Theorem stating that the total number of pages in the book is 563 -/
theorem book_pages : total_pages = 563 := by
  sorry

end book_pages_l2094_209421


namespace tangent_triangle_area_l2094_209417

-- Define the curve
def curve (x y : ℝ) : Prop := x * y - x + 2 * y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (1, 2)

-- Define the tangent line at point A
def tangent_line (x y : ℝ) : Prop := x + 3 * y - 7 = 0

-- Theorem statement
theorem tangent_triangle_area : 
  curve point_A.1 point_A.2 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    tangent_line x₁ y₁ ∧ 
    tangent_line x₂ y₂ ∧ 
    x₁ = 0 ∧ 
    y₂ = 0 ∧ 
    (1/2 * x₂ * y₁ = 49/6)) := by
  sorry

end tangent_triangle_area_l2094_209417


namespace ellipse_line_intersection_slope_product_l2094_209468

/-- An ellipse passing through (2,0) with eccentricity √3/2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eq : a^2 = 4
  h_ecc : (a^2 - b^2) / a^2 = 3/4

/-- A line passing through (1,0) with non-zero slope -/
structure Line where
  k : ℝ
  h_k_nonzero : k ≠ 0

/-- The theorem statement -/
theorem ellipse_line_intersection_slope_product (C : Ellipse) (l : Line) :
  ∃ k' : ℝ, l.k * k' = -1/4 := by sorry

end ellipse_line_intersection_slope_product_l2094_209468


namespace water_flow_solution_l2094_209432

/-- Represents the water flow problem --/
def water_flow_problem (t : ℝ) : Prop :=
  let initial_rate : ℝ := 2 / 10  -- 2 cups per 10 minutes
  let final_rate : ℝ := 4 / 10    -- 4 cups per 10 minutes
  let initial_duration : ℝ := 2 * t  -- flows for t minutes twice
  let final_duration : ℝ := 60    -- flows for 60 minutes at final rate
  let total_water : ℝ := initial_rate * initial_duration + final_rate * final_duration
  let remaining_water : ℝ := total_water / 2
  remaining_water = 18 ∧ t = 30

/-- Theorem stating the solution to the water flow problem --/
theorem water_flow_solution :
  ∃ t : ℝ, water_flow_problem t :=
sorry

end water_flow_solution_l2094_209432


namespace go_pieces_probability_l2094_209449

theorem go_pieces_probability (p_black p_white : ℝ) 
  (h_black : p_black = 1/7)
  (h_white : p_white = 12/35) :
  p_black + p_white = 17/35 := by
  sorry

end go_pieces_probability_l2094_209449


namespace inequality_solution_l2094_209438

theorem inequality_solution (x : ℝ) : x^2 - 3*x - 10 < 0 ∧ x > 1 → 1 < x ∧ x < 5 := by
  sorry

end inequality_solution_l2094_209438


namespace cos_48_degrees_l2094_209426

theorem cos_48_degrees :
  ∃ x : ℝ, 4 * x^3 - 3 * x - (1 + Real.sqrt 5) / 4 = 0 ∧
  Real.cos (48 * π / 180) = (1 / 2) * x + (Real.sqrt 3 / 2) * Real.sqrt (1 - x^2) := by
  sorry

end cos_48_degrees_l2094_209426


namespace unique_quadratic_solution_l2094_209476

theorem unique_quadratic_solution :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, q * x^2 - 8 * x + 2 = 0) :=
by sorry

end unique_quadratic_solution_l2094_209476


namespace manager_salary_is_4200_l2094_209428

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager's salary is added. -/
def managerSalary (numEmployees : ℕ) (avgSalary : ℚ) (avgIncrease : ℚ) : ℚ :=
  (avgSalary + avgIncrease) * (numEmployees + 1) - avgSalary * numEmployees

/-- Proves that the manager's salary is 4200 given the problem conditions. -/
theorem manager_salary_is_4200 :
  managerSalary 15 1800 150 = 4200 := by
  sorry

#eval managerSalary 15 1800 150

end manager_salary_is_4200_l2094_209428


namespace Tricia_age_is_5_l2094_209403

-- Define the ages as natural numbers
def Vincent_age : ℕ := 22
def Rupert_age : ℕ := Vincent_age - 2
def Khloe_age : ℕ := Rupert_age - 10
def Eugene_age : ℕ := Khloe_age * 3
def Yorick_age : ℕ := Eugene_age * 2
def Amilia_age : ℕ := Yorick_age / 4
def Tricia_age : ℕ := Amilia_age / 3

-- Theorem statement
theorem Tricia_age_is_5 : Tricia_age = 5 := by
  sorry

end Tricia_age_is_5_l2094_209403


namespace xyz_sum_root_l2094_209473

theorem xyz_sum_root (x y z : ℝ) 
  (eq1 : y + z = 24)
  (eq2 : z + x = 26)
  (eq3 : x + y = 28) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 83655 := by
  sorry

end xyz_sum_root_l2094_209473


namespace greatest_x_value_l2094_209405

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 210000) :
  x ≤ 4 ∧ ∃ y : ℤ, y > 4 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 210000 :=
by sorry

end greatest_x_value_l2094_209405


namespace smallest_seating_arrangement_l2094_209427

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if a seating arrangement satisfies the condition that any new person must sit next to someone -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ seating.total_chairs → 
    ∃ i j, i ≠ j ∧ 
           (i % seating.total_chairs + 1 = k ∨ (i + 1) % seating.total_chairs + 1 = k) ∧
           (j % seating.total_chairs + 1 = k ∨ (j + 1) % seating.total_chairs + 1 = k)

/-- The main theorem to prove -/
theorem smallest_seating_arrangement :
  ∀ n < 25, ¬(satisfies_condition ⟨100, n⟩) ∧ 
  satisfies_condition ⟨100, 25⟩ := by
  sorry

#check smallest_seating_arrangement

end smallest_seating_arrangement_l2094_209427


namespace average_donation_l2094_209495

def donations : List ℝ := [10, 12, 13.5, 40.8, 19.3, 20.8, 25, 16, 30, 30]

theorem average_donation : (donations.sum / donations.length) = 21.74 := by
  sorry

end average_donation_l2094_209495


namespace endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l2094_209413

/-- Given a line segment with one endpoint (6,4) and midpoint (3,10),
    the sum of the coordinates of the other endpoint is 16. -/
theorem endpoint_coordinate_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 4) ∧
    midpoint = (3, 10) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 16

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : ∃ (endpoint2 : ℝ × ℝ),
  endpoint_coordinate_sum (6, 4) (3, 10) endpoint2 := by
  sorry

end endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l2094_209413


namespace placement_theorem_l2094_209404

def number_of_placements (n : ℕ) : ℕ := 
  Nat.choose 4 2 * (n * (n - 1))

theorem placement_theorem : number_of_placements 4 = 72 := by
  sorry

end placement_theorem_l2094_209404


namespace triangle_properties_l2094_209475

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.b - t.c) / t.a = Real.cos t.C / Real.cos t.A) 
  (h2 : t.a = Real.sqrt 5) 
  (h3 : (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2) :
  t.A = π / 3 ∧ t.a + t.b + t.c = Real.sqrt 5 + Real.sqrt 11 := by
  sorry

end triangle_properties_l2094_209475


namespace existence_of_invariant_sequences_l2094_209408

/-- A binary sequence is a function from ℕ to {0, 1} -/
def BinarySeq := ℕ → Fin 2

/-- Remove odd-indexed elements from a sequence -/
def removeOdd (s : BinarySeq) : BinarySeq :=
  fun n => s (2 * n + 1)

/-- Remove even-indexed elements from a sequence -/
def removeEven (s : BinarySeq) : BinarySeq :=
  fun n => s (2 * n)

/-- A sequence is invariant under odd removal if removing odd-indexed elements results in the same sequence -/
def invariantUnderOddRemoval (s : BinarySeq) : Prop :=
  ∀ n, s n = removeOdd s n

/-- A sequence is invariant under even removal if removing even-indexed elements results in the same sequence -/
def invariantUnderEvenRemoval (s : BinarySeq) : Prop :=
  ∀ n, s n = removeEven s n

theorem existence_of_invariant_sequences :
  (∃ s : BinarySeq, invariantUnderOddRemoval s) ∧
  (∃ s : BinarySeq, invariantUnderEvenRemoval s) :=
sorry

end existence_of_invariant_sequences_l2094_209408


namespace curve_condition_iff_l2094_209441

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A curve defined by a function f(x, y) = 0 -/
structure Curve where
  f : ℝ → ℝ → ℝ

/-- Predicate for a point being on a curve -/
def IsOnCurve (p : Point) (c : Curve) : Prop :=
  c.f p.x p.y = 0

/-- Theorem stating that f(x, y) = 0 is a necessary and sufficient condition
    for a point P(x, y) to be on the curve f(x, y) = 0 -/
theorem curve_condition_iff (c : Curve) (p : Point) :
  IsOnCurve p c ↔ c.f p.x p.y = 0 := by sorry

end curve_condition_iff_l2094_209441


namespace textbook_transfer_l2094_209496

theorem textbook_transfer (initial_a initial_b transfer : ℕ) 
  (h1 : initial_a = 200)
  (h2 : initial_b = 200)
  (h3 : transfer = 40) :
  (initial_b + transfer) = (initial_a - transfer) * 3 / 2 := by
  sorry

end textbook_transfer_l2094_209496


namespace power_three_mod_five_l2094_209485

theorem power_three_mod_five : 3^244 % 5 = 1 := by
  sorry

end power_three_mod_five_l2094_209485


namespace boa_constrictor_length_alberts_boa_length_l2094_209409

/-- The length of Albert's boa constrictor given the length of his garden snake and their relative sizes. -/
theorem boa_constrictor_length (garden_snake_length : ℕ) (relative_size : ℕ) : ℕ :=
  garden_snake_length * relative_size

/-- Proof that Albert's boa constrictor is 70 inches long. -/
theorem alberts_boa_length : boa_constrictor_length 10 7 = 70 := by
  sorry

end boa_constrictor_length_alberts_boa_length_l2094_209409


namespace common_difference_is_two_l2094_209467

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0
  h_sum : a 1 + a 2 + a 3 = 9
  h_geometric : ∃ r : ℝ, r ≠ 0 ∧ a 2 = r * a 1 ∧ a 5 = r * a 2

/-- The common difference of the arithmetic sequence is 2 -/
theorem common_difference_is_two (seq : ArithmeticSequence) : seq.d = 2 := by
  sorry

end common_difference_is_two_l2094_209467


namespace tricia_age_l2094_209483

theorem tricia_age (vincent_age : ℕ)
  (h1 : vincent_age = 22)
  (rupert_age : ℕ)
  (h2 : rupert_age = vincent_age - 2)
  (khloe_age : ℕ)
  (h3 : rupert_age = khloe_age + 10)
  (eugene_age : ℕ)
  (h4 : khloe_age = eugene_age / 3)
  (yorick_age : ℕ)
  (h5 : yorick_age = 2 * eugene_age)
  (amilia_age : ℕ)
  (h6 : amilia_age = yorick_age / 4)
  (tricia_age : ℕ)
  (h7 : tricia_age = amilia_age / 3) :
  tricia_age = 5 := by
sorry

end tricia_age_l2094_209483


namespace epidemic_duration_l2094_209459

structure Dwarf :=
  (id : ℕ)
  (status : ℕ → Nat)  -- 0: healthy, 1: sick, 2: immune

def Epidemic (population : List Dwarf) (day : ℕ) : Prop :=
  ∃ (d : Dwarf), d ∈ population ∧ d.status day = 1

def VisitsSickFriends (d : Dwarf) (population : List Dwarf) (day : ℕ) : Prop :=
  d.status day = 0 → ∃ (sick : Dwarf), sick ∈ population ∧ sick.status day = 1

def BecomeSick (d : Dwarf) (population : List Dwarf) (day : ℕ) : Prop :=
  VisitsSickFriends d population day → d.status (day + 1) = 1

def ImmunityPeriod (d : Dwarf) (day : ℕ) : Prop :=
  d.status day = 1 → d.status (day + 1) = 2

def CannotInfectImmune (d : Dwarf) (day : ℕ) : Prop :=
  d.status day = 2 → d.status (day + 1) ≠ 1

theorem epidemic_duration (population : List Dwarf) :
  (∃ (d : Dwarf), d ∈ population ∧ d.status 0 = 2) →
    ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ Epidemic population m
  ∧
  (∀ (d : Dwarf), d ∈ population → d.status 0 ≠ 2) →
    ∃ (n : ℕ), ∀ (m : ℕ), m ≥ n → ¬(Epidemic population m) :=
by sorry

end epidemic_duration_l2094_209459


namespace late_fee_is_150_l2094_209445

/-- Calculates the late fee for electricity payment -/
def calculate_late_fee (cost_per_watt : ℝ) (watts_used : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid - cost_per_watt * watts_used

/-- Proves that the late fee is $150 given the problem conditions -/
theorem late_fee_is_150 :
  let cost_per_watt : ℝ := 4
  let watts_used : ℝ := 300
  let total_paid : ℝ := 1350
  calculate_late_fee cost_per_watt watts_used total_paid = 150 := by
  sorry

end late_fee_is_150_l2094_209445


namespace arithmetic_sequence_problem_l2094_209424

theorem arithmetic_sequence_problem :
  ∀ (a d : ℝ),
  (a - d) + a + (a + d) = 6 ∧
  (a - d) * a * (a + d) = -10 →
  ((a - d = 5 ∧ a = 2 ∧ a + d = -1) ∨
   (a - d = -1 ∧ a = 2 ∧ a + d = 5)) :=
by sorry

end arithmetic_sequence_problem_l2094_209424


namespace quadratic_distinct_roots_condition_l2094_209498

/-- For a quadratic equation (k+2)x^2 + 4x + 1 = 0 to have two distinct real roots, 
    k must satisfy: k < 2 and k ≠ -2 -/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k + 2) * x^2 + 4 * x + 1 = 0 ∧ 
   (k + 2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
by sorry

end quadratic_distinct_roots_condition_l2094_209498


namespace complex_number_quadrant_l2094_209458

theorem complex_number_quadrant (z : ℂ) (h : (3 + 4*I)*z = 25) : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end complex_number_quadrant_l2094_209458


namespace S_and_S_l2094_209478

-- Define the systems S and S'
def S (x y : ℝ) : Prop :=
  y * (x^4 - y^2 + x^2) = x ∧ x * (x^4 - y^2 + x^2) = 1

def S' (x y : ℝ) : Prop :=
  y * (x^4 - y^2 + x^2) = x ∧ y = x^2

-- Theorem stating that S and S' do not have the same set of solutions
theorem S_and_S'_different_solutions :
  ¬(∀ x y : ℝ, S x y ↔ S' x y) :=
sorry

end S_and_S_l2094_209478


namespace no_common_real_solution_l2094_209491

theorem no_common_real_solution :
  ¬ ∃ (x y : ℝ), (x^2 + y^2 + 16 = 0) ∧ (x^2 - 3*y + 12 = 0) := by
  sorry

end no_common_real_solution_l2094_209491


namespace arithmetic_sequence_b_formula_l2094_209414

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a (3^n)

theorem arithmetic_sequence_b_formula (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 8 →
  a 8 = 26 →
  ∀ n : ℕ, b a n = 3^(n+1) + 2 :=
by sorry

end arithmetic_sequence_b_formula_l2094_209414


namespace parabola_vertex_l2094_209466

/-- The vertex of the parabola y = x^2 - 6x + 1 has coordinates (3, -8) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 6*x + 1 → ∃ (h k : ℝ), h = 3 ∧ k = -8 ∧ ∀ x, y = (x - h)^2 + k := by
  sorry

end parabola_vertex_l2094_209466


namespace geometric_sequence_seventh_term_l2094_209456

/-- Given a geometric sequence {a_n} with a₁ = 1 and a₄ = 8, prove that a₇ = 64 -/
theorem geometric_sequence_seventh_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1 →                                -- First term condition
  a 4 = 8 →                                -- Fourth term condition
  a 7 = 64 :=                              -- Conclusion to prove
by sorry

end geometric_sequence_seventh_term_l2094_209456


namespace angle_D_measure_l2094_209472

/-- Given a geometric figure with angles A, B, C, and D, prove that when 
    m∠A = 50°, m∠B = 35°, and m∠C = 35°, then m∠D = 120°. -/
theorem angle_D_measure (A B C D : Real) 
    (hA : A = 50) 
    (hB : B = 35)
    (hC : C = 35) : 
  D = 120 := by
  sorry

end angle_D_measure_l2094_209472


namespace one_black_two_white_reachable_l2094_209400

-- Define the urn state as a pair of natural numbers (white, black)
def UrnState := ℕ × ℕ

-- Define the initial state
def initial_state : UrnState := (50, 150)

-- Define the four operations
def op1 (s : UrnState) : UrnState := (s.1, s.2 - 2)
def op2 (s : UrnState) : UrnState := (s.1, s.2 - 1)
def op3 (s : UrnState) : UrnState := (s.1, s.2)
def op4 (s : UrnState) : UrnState := (s.1 - 3, s.2 + 2)

-- Define a predicate for valid states (non-negative marbles)
def valid_state (s : UrnState) : Prop := s.1 ≥ 0 ∧ s.2 ≥ 0

-- Define the reachability relation
inductive reachable : UrnState → Prop where
  | initial : reachable initial_state
  | op1 : ∀ s, reachable s → valid_state (op1 s) → reachable (op1 s)
  | op2 : ∀ s, reachable s → valid_state (op2 s) → reachable (op2 s)
  | op3 : ∀ s, reachable s → valid_state (op3 s) → reachable (op3 s)
  | op4 : ∀ s, reachable s → valid_state (op4 s) → reachable (op4 s)

-- Theorem stating that the configuration (2, 1) is reachable
theorem one_black_two_white_reachable : reachable (2, 1) := by sorry

end one_black_two_white_reachable_l2094_209400


namespace equation_solutions_l2094_209401

theorem equation_solutions : 
  let f (x : ℝ) := (18*x - x^2) / (x + 2) * (x + (18 - x) / (x + 2))
  ∀ x : ℝ, f x = 56 ↔ x = 4 ∨ x = -14/17 := by
  sorry

end equation_solutions_l2094_209401


namespace tank_length_proof_l2094_209465

/-- Proves that the length of a rectangular tank is 3 feet given specific conditions -/
theorem tank_length_proof (l : ℝ) : 
  let w : ℝ := 6
  let h : ℝ := 2
  let cost_per_sqft : ℝ := 20
  let total_cost : ℝ := 1440
  let surface_area : ℝ := 2 * l * w + 2 * l * h + 2 * w * h
  total_cost = cost_per_sqft * surface_area → l = 3 := by
  sorry

end tank_length_proof_l2094_209465


namespace geometric_sequence_sum_l2094_209469

theorem geometric_sequence_sum (a₁ a₂ a₃ a₄ a₅ : ℕ) (q : ℚ) :
  (a₁ > 0) →
  (a₂ > a₁) → (a₃ > a₂) → (a₄ > a₃) → (a₅ > a₄) →
  (a₂ = a₁ * q) → (a₃ = a₂ * q) → (a₄ = a₃ * q) → (a₅ = a₄ * q) →
  (a₁ + a₂ + a₃ + a₄ + a₅ = 211) →
  (a₁ = 16 ∧ q = 3/2) :=
by sorry

end geometric_sequence_sum_l2094_209469


namespace counterfeit_banknote_theorem_l2094_209420

/-- Represents a banknote with a natural number denomination -/
structure Banknote where
  denomination : ℕ

/-- Represents a collection of banknotes -/
def BanknoteCollection := List Banknote

/-- The detector's reading of the total sum -/
def detectorSum (collection : BanknoteCollection) : ℕ := sorry

/-- The actual sum of genuine banknotes -/
def genuineSum (collection : BanknoteCollection) : ℕ := sorry

/-- Predicate to check if a collection has pairwise different denominations -/
def hasPairwiseDifferentDenominations (collection : BanknoteCollection) : Prop := sorry

/-- Predicate to check if a collection has exactly one counterfeit banknote -/
def hasExactlyOneCounterfeit (collection : BanknoteCollection) : Prop := sorry

/-- The denomination of the counterfeit banknote -/
def counterfeitDenomination (collection : BanknoteCollection) : ℕ := sorry

theorem counterfeit_banknote_theorem (collection : BanknoteCollection) 
  (h1 : hasPairwiseDifferentDenominations collection)
  (h2 : hasExactlyOneCounterfeit collection) :
  detectorSum collection - genuineSum collection = counterfeitDenomination collection := by
  sorry

end counterfeit_banknote_theorem_l2094_209420


namespace sum_of_reciprocals_l2094_209460

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) :
  1 / x + 1 / y = 2 := by
  sorry

end sum_of_reciprocals_l2094_209460


namespace fraction_addition_l2094_209434

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 9 + 3 = (33 + 5 * d) / 9 := by
  sorry

end fraction_addition_l2094_209434


namespace complex_expression_equality_l2094_209406

theorem complex_expression_equality : 
  let x := (11 + 6 * Real.sqrt 2) * Real.sqrt (11 - 6 * Real.sqrt 2) - 
           (11 - 6 * Real.sqrt 2) * Real.sqrt (11 + 6 * Real.sqrt 2)
  let y := Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2) - 
           Real.sqrt (Real.sqrt 5 + 1)
  x / y = 28 + 14 * Real.sqrt 2 := by sorry

end complex_expression_equality_l2094_209406


namespace eight_pow_zero_minus_log_100_l2094_209422

theorem eight_pow_zero_minus_log_100 : 8^0 - Real.log 100 / Real.log 10 = -1 := by
  sorry

end eight_pow_zero_minus_log_100_l2094_209422


namespace students_taking_algebra_or_drafting_but_not_both_l2094_209490

-- Define the sets of students
def algebra : Finset ℕ := sorry
def drafting : Finset ℕ := sorry
def geometry : Finset ℕ := sorry

-- State the theorem
theorem students_taking_algebra_or_drafting_but_not_both : 
  (algebra.card + drafting.card - (algebra ∩ drafting).card) - ((geometry ∩ drafting).card - (algebra ∩ geometry ∩ drafting).card) = 42 :=
by
  -- Given conditions
  have h1 : (algebra ∩ drafting).card = 15 := sorry
  have h2 : algebra.card = 30 := sorry
  have h3 : (drafting \ algebra).card = 14 := sorry
  have h4 : (geometry \ (algebra ∪ drafting)).card = 8 := sorry
  have h5 : ((geometry ∩ drafting) \ algebra).card = 5 := sorry
  
  sorry -- Proof goes here

end students_taking_algebra_or_drafting_but_not_both_l2094_209490


namespace binomial_20_4_l2094_209423

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by sorry

end binomial_20_4_l2094_209423


namespace fixed_point_of_exponential_function_l2094_209442

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2*x - 1)
  f (1/2) = 1 := by sorry

end fixed_point_of_exponential_function_l2094_209442


namespace furniture_pricing_l2094_209402

/-- The cost price of furniture before markup and discount -/
def cost_price : ℝ := 7777.78

/-- The markup percentage applied to the cost price -/
def markup_percentage : ℝ := 0.20

/-- The discount percentage applied to the total price -/
def discount_percentage : ℝ := 0.10

/-- The final price paid by the customer after markup and discount -/
def final_price : ℝ := 8400

theorem furniture_pricing :
  final_price = (1 - discount_percentage) * (1 + markup_percentage) * cost_price := by
  sorry

end furniture_pricing_l2094_209402


namespace geometric_series_r_value_l2094_209416

theorem geometric_series_r_value (b r : ℝ) (h1 : r ≠ 1) (h2 : r ≠ -1) : 
  (b / (1 - r) = 18) → 
  (b * r^2 / (1 - r^2) = 6) → 
  r = 1/2 := by
sorry

end geometric_series_r_value_l2094_209416


namespace triangle_area_problem_l2094_209461

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * x * 3*x = 72) : x = 4 * Real.sqrt 3 := by
  sorry

end triangle_area_problem_l2094_209461


namespace division_simplification_l2094_209439

theorem division_simplification (a : ℝ) (h : a ≠ 0) : 6 * a / (2 * a) = 3 := by
  sorry

end division_simplification_l2094_209439


namespace complex_sum_example_l2094_209482

theorem complex_sum_example : 
  let z₁ : ℂ := 1 + 7*I
  let z₂ : ℂ := -2 - 4*I
  z₁ + z₂ = -1 + 3*I :=
by sorry

end complex_sum_example_l2094_209482


namespace gcd_lcm_sum_for_special_case_l2094_209492

theorem gcd_lcm_sum_for_special_case (a b : ℕ) (h : a = 1999 * b) :
  Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end gcd_lcm_sum_for_special_case_l2094_209492


namespace select_team_count_l2094_209444

/-- The number of ways to select a team of 8 members (4 boys and 4 girls) from a group of 10 boys and 12 girls -/
def selectTeam (totalBoys : ℕ) (totalGirls : ℕ) (teamSize : ℕ) (boysInTeam : ℕ) (girlsInTeam : ℕ) : ℕ :=
  Nat.choose totalBoys boysInTeam * Nat.choose totalGirls girlsInTeam

/-- Theorem stating that the number of ways to select the team is 103950 -/
theorem select_team_count :
  selectTeam 10 12 8 4 4 = 103950 := by
  sorry

end select_team_count_l2094_209444


namespace square_area_from_diagonal_l2094_209488

theorem square_area_from_diagonal (d : ℝ) (h : d = 40) :
  let s := d / Real.sqrt 2
  s * s = 800 := by sorry

end square_area_from_diagonal_l2094_209488


namespace point_not_on_transformed_plane_l2094_209474

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let originalPlane : Plane := { a := 1, b := -1, c := -1, d := -1 }
  let k : ℝ := 4
  let transformedPlane := transformPlane originalPlane k
  let point : Point3D := { x := 7, y := 0, z := -1 }
  ¬(pointOnPlane point transformedPlane) := by
  sorry


end point_not_on_transformed_plane_l2094_209474


namespace flu_outbreak_l2094_209455

theorem flu_outbreak (initial_infected : ℕ) (infected_after_two_rounds : ℕ) :
  initial_infected = 1 →
  infected_after_two_rounds = 81 →
  ∃ (avg_infected_per_round : ℕ),
    avg_infected_per_round = 8 ∧
    initial_infected + avg_infected_per_round + avg_infected_per_round * (avg_infected_per_round + 1) = infected_after_two_rounds ∧
    infected_after_two_rounds * avg_infected_per_round + infected_after_two_rounds = 729 :=
by sorry

end flu_outbreak_l2094_209455


namespace solve_for_y_l2094_209412

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 2 = y + 6) (h2 : x = -4) : y = 24 := by
  sorry

end solve_for_y_l2094_209412


namespace calculation_result_l2094_209486

theorem calculation_result : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end calculation_result_l2094_209486


namespace final_distance_to_catch_up_l2094_209419

/-- Represents the state of the race at any given point --/
structure RaceState where
  alex_lead : Int
  distance_covered : Nat

/-- Calculates the new race state after a terrain change --/
def update_race_state (current_state : RaceState) (alex_gain : Int) : RaceState :=
  { alex_lead := current_state.alex_lead + alex_gain,
    distance_covered := current_state.distance_covered }

def race_length : Nat := 5000

theorem final_distance_to_catch_up :
  let initial_state : RaceState := { alex_lead := 0, distance_covered := 200 }
  let after_uphill := update_race_state initial_state 300
  let after_downhill := update_race_state after_uphill (-170)
  let final_state := update_race_state after_downhill 440
  final_state.alex_lead = 570 := by sorry

end final_distance_to_catch_up_l2094_209419


namespace probability_one_genuine_one_defective_l2094_209448

/-- The probability of selecting exactly one genuine product and one defective product
    when randomly selecting two products from a set of 5 genuine products and 1 defective product. -/
theorem probability_one_genuine_one_defective :
  let total_products : ℕ := 5 + 1
  let genuine_products : ℕ := 5
  let defective_products : ℕ := 1
  let total_selections : ℕ := Nat.choose total_products 2
  let favorable_selections : ℕ := genuine_products * defective_products
  (favorable_selections : ℚ) / total_selections = 1 / 3 :=
by sorry

end probability_one_genuine_one_defective_l2094_209448


namespace complement_of_A_l2094_209487

def A : Set ℝ := {x : ℝ | (x - 2) / (x - 1) ≥ 0}

theorem complement_of_A : 
  (Set.univ \ A : Set ℝ) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by sorry

end complement_of_A_l2094_209487


namespace athlete_track_arrangements_l2094_209443

/-- The number of ways to arrange 5 athletes on 5 tracks with exactly two in their numbered tracks -/
def athleteArrangements : ℕ := 20

/-- The number of ways to choose 2 items from a set of 5 -/
def choose5_2 : ℕ := 10

/-- The number of derangements of 3 objects -/
def derangement3 : ℕ := 2

theorem athlete_track_arrangements :
  athleteArrangements = choose5_2 * derangement3 :=
sorry

end athlete_track_arrangements_l2094_209443


namespace smallest_special_number_l2094_209429

def is_special (n : ℕ) : Prop :=
  (n > 3429) ∧ (∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number :
  ∀ m : ℕ, is_special m → m ≥ 3450 :=
by sorry

end smallest_special_number_l2094_209429


namespace system_solution_ratio_l2094_209451

theorem system_solution_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c)
    (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : c / d = -1 / 3 := by
  sorry

end system_solution_ratio_l2094_209451


namespace sqrt_equation_solutions_l2094_209411

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt x = 18 / (11 - Real.sqrt x)) ↔ (x = 81 ∨ x = 4) :=
by sorry

end sqrt_equation_solutions_l2094_209411


namespace gcf_lcm_sum_l2094_209462

def numbers : List Nat := [18, 24, 36]

theorem gcf_lcm_sum (C D : Nat) (hC : C = Nat.gcd 18 (Nat.gcd 24 36)) 
  (hD : D = Nat.lcm 18 (Nat.lcm 24 36)) : C + D = 78 := by
  sorry

end gcf_lcm_sum_l2094_209462


namespace one_correct_statement_l2094_209418

theorem one_correct_statement :
  (∃! n : Nat, n = 1 ∧
    (∀ a b : ℝ, a + b = 0 → a = -b) ∧
    (3^2 = 6) ∧
    (∀ a : ℚ, a > -a) ∧
    (∀ a b : ℝ, |a| = |b| → a = b)) :=
sorry

end one_correct_statement_l2094_209418


namespace f_divisibility_by_3_smallest_n_for_2017_l2094_209493

def f : ℕ → ℤ
  | 0 => 0  -- base case
  | n + 1 => if n.succ % 2 = 0 then -f (n.succ / 2) else f n + 1

theorem f_divisibility_by_3 (n : ℕ) : 3 ∣ f n ↔ 3 ∣ n := by sorry

def geometric_sum (n : ℕ) : ℕ := (4^(n+1) - 1) / 3

theorem smallest_n_for_2017 : 
  f (geometric_sum 1008) = 2017 ∧ 
  ∀ m : ℕ, m < geometric_sum 1008 → f m ≠ 2017 := by sorry

end f_divisibility_by_3_smallest_n_for_2017_l2094_209493


namespace unique_prime_pair_sum_53_l2094_209410

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating that there is exactly one pair of primes summing to 53 -/
theorem unique_prime_pair_sum_53 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
sorry

end unique_prime_pair_sum_53_l2094_209410


namespace minimum_packs_for_90_cans_l2094_209494

/-- Represents the available pack sizes for soda cans -/
def PackSizes : List Nat := [6, 12, 24]

/-- The total number of cans we need to buy -/
def TotalCans : Nat := 90

/-- A function that calculates the minimum number of packs needed -/
def MinimumPacks (packSizes : List Nat) (totalCans : Nat) : Nat :=
  sorry -- Proof implementation goes here

/-- Theorem stating that the minimum number of packs needed is 5 -/
theorem minimum_packs_for_90_cans : 
  MinimumPacks PackSizes TotalCans = 5 := by
  sorry -- Proof goes here

end minimum_packs_for_90_cans_l2094_209494


namespace subset_implies_a_nonpositive_l2094_209431

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

-- Theorem statement
theorem subset_implies_a_nonpositive (a : ℝ) (h : B ⊆ A a) : a ≤ 0 := by
  sorry

end subset_implies_a_nonpositive_l2094_209431


namespace derivative_of_two_sin_x_l2094_209447

theorem derivative_of_two_sin_x (x : ℝ) :
  deriv (λ x => 2 * Real.sin x) x = 2 * Real.cos x := by
  sorry

end derivative_of_two_sin_x_l2094_209447


namespace factorization_problems_l2094_209435

theorem factorization_problems (x y : ℝ) : 
  (x^3 - 6*x^2 + 9*x = x*(x-3)^2) ∧ 
  ((x-2)^2 - x + 2 = (x-2)*(x-3)) ∧ 
  ((x^2 + y^2)^2 - 4*x^2*y^2 = (x+y)^2*(x-y)^2) := by
  sorry

end factorization_problems_l2094_209435


namespace seven_fourth_mod_hundred_l2094_209480

theorem seven_fourth_mod_hundred : 7^4 % 100 = 1 := by
  sorry

end seven_fourth_mod_hundred_l2094_209480


namespace vector_relation_in_triangle_l2094_209415

/-- Given a triangle ABC and a point D, if AB = 4DB, then CD = (1/4)CA + (3/4)CB -/
theorem vector_relation_in_triangle (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (B - A) = 4 • (B - D) →
  (D - C) = (1/4) • (A - C) + (3/4) • (B - C) := by
  sorry

end vector_relation_in_triangle_l2094_209415


namespace correct_divisor_l2094_209497

theorem correct_divisor (incorrect_result : ℝ) (dividend : ℝ) (h1 : incorrect_result = 204) (h2 : dividend = 30.6) :
  ∃ (correct_divisor : ℝ), 
    dividend / (correct_divisor * 10) = incorrect_result ∧
    correct_divisor = (dividend / incorrect_result) / 10 := by
  sorry

end correct_divisor_l2094_209497


namespace smallest_b_for_nonprime_cubic_l2094_209407

theorem smallest_b_for_nonprime_cubic (x : ℤ) : ∃ (b : ℕ+), ∀ (x : ℤ), ¬ Prime (x^3 + b^2) ∧ ∀ (k : ℕ+), k < b → ∃ (y : ℤ), Prime (y^3 + k^2) :=
sorry

end smallest_b_for_nonprime_cubic_l2094_209407


namespace equation_solutions_l2094_209464

theorem equation_solutions :
  (∀ x : ℝ, 2 * x^2 - 1 = 49 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, (x + 3)^3 = 64 ↔ x = 1) := by
  sorry

end equation_solutions_l2094_209464


namespace interest_rate_is_four_percent_l2094_209446

/-- Given a principal sum and an interest rate, if the simple interest
    for 5 years is one-fifth of the principal, then the interest rate is 4% -/
theorem interest_rate_is_four_percent 
  (P : ℝ) -- Principal sum
  (R : ℝ) -- Interest rate as a percentage
  (h : P > 0) -- Assumption that principal is positive
  (h_interest : P / 5 = (P * R * 5) / 100) -- Condition that interest is one-fifth of principal
  : R = 4 := by
sorry

end interest_rate_is_four_percent_l2094_209446


namespace inscribed_box_sphere_radius_l2094_209470

theorem inscribed_box_sphere_radius (a b c s : ℝ) : 
  a > 0 → b > 0 → c > 0 → s > 0 →
  (a + b + c = 18) →
  (2 * a * b + 2 * b * c + 2 * a * c = 216) →
  (4 * s^2 = a^2 + b^2 + c^2) →
  s = Real.sqrt 27 := by
sorry

end inscribed_box_sphere_radius_l2094_209470


namespace hawk_percentage_is_65_percent_l2094_209489

/-- Represents the percentage of birds that are hawks in the nature reserve -/
def hawk_percentage : ℝ := sorry

/-- Represents the percentage of non-hawks that are paddyfield-warblers -/
def paddyfield_warbler_ratio : ℝ := 0.4

/-- Represents the ratio of kingfishers to paddyfield-warblers -/
def kingfisher_to_warbler_ratio : ℝ := 0.25

/-- Represents the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
def other_birds_percentage : ℝ := 0.35

theorem hawk_percentage_is_65_percent :
  hawk_percentage = 0.65 ∧
  paddyfield_warbler_ratio * (1 - hawk_percentage) +
  kingfisher_to_warbler_ratio * paddyfield_warbler_ratio * (1 - hawk_percentage) +
  hawk_percentage +
  other_birds_percentage = 1 :=
sorry

end hawk_percentage_is_65_percent_l2094_209489


namespace circle_equation_l2094_209425

/-- Given a circle with center (-1, 2) passing through the point (2, -2),
    its standard equation is (x+1)^2 + (y-2)^2 = 25 -/
theorem circle_equation (x y : ℝ) : 
  let center := (-1, 2)
  let point_on_circle := (2, -2)
  (x + 1)^2 + (y - 2)^2 = 25 ↔ 
    (∃ (r : ℝ), r > 0 ∧
      (x - center.1)^2 + (y - center.2)^2 = r^2 ∧
      (point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2 = r^2) :=
by sorry

end circle_equation_l2094_209425


namespace father_son_age_sum_father_son_age_sum_proof_l2094_209479

/-- Given that:
  1) Eighteen years ago, the father was 3 times as old as his son.
  2) Now, the father is twice as old as his son.
  Prove that the sum of their current ages is 108 years. -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun (son_age father_age : ℕ) =>
    (father_age - 18 = 3 * (son_age - 18)) →
    (father_age = 2 * son_age) →
    (son_age + father_age = 108)

/-- Proof of the theorem -/
theorem father_son_age_sum_proof : ∃ (son_age father_age : ℕ),
  father_son_age_sum son_age father_age :=
by
  sorry

end father_son_age_sum_father_son_age_sum_proof_l2094_209479
