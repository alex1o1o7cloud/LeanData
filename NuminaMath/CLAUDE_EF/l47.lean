import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_closed_chain_condition_l47_4725

/-- 
Represents the possibility of forming a single closed chain 
in a rectangular thread grid with dimensions k and n.
-/
def can_form_closed_chain (k n : ℕ) : Prop :=
  Even (k * n) ∧ k ≥ 2 ∧ n ≥ 2

/-- 
Theorem stating the necessary and sufficient conditions for 
forming a single closed chain in a rectangular thread grid.
-/
theorem closed_chain_condition (k n : ℕ) : 
  can_form_closed_chain k n ↔ 
    Even (k * n) ∧ k ≥ 2 ∧ n ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_closed_chain_condition_l47_4725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_li_qiang_spent_seven_yuan_l47_4710

/-- Represents the possible combinations of New Year's card quantities -/
structure CardCombination where
  x : ℕ  -- quantity of 0.1 yuan cards
  y : ℕ  -- quantity of 0.15 yuan cards
  z : ℕ  -- quantity of 0.25 yuan cards
  w : ℕ  -- quantity of 0.4 yuan cards

/-- Theorem stating that Li Qiang spent exactly 7 yuan on New Year's cards -/
theorem li_qiang_spent_seven_yuan :
  ∀ (c : CardCombination),
    c.x + c.y + c.z + c.w = 30 →  -- total of 30 cards
    (c.x = 5 ∧ c.y = 5 ∧ c.z = 10 ∧ c.w = 10) ∨
    (c.x = 5 ∧ c.y = 10 ∧ c.z = 5 ∧ c.w = 10) ∨
    (c.x = 10 ∧ c.y = 5 ∧ c.z = 5 ∧ c.w = 10) ∨
    (c.x = 10 ∧ c.y = 5 ∧ c.z = 10 ∧ c.w = 5) ∨
    (c.x = 5 ∧ c.y = 10 ∧ c.z = 10 ∧ c.w = 5) ∨
    (c.x = 10 ∧ c.y = 10 ∧ c.z = 5 ∧ c.w = 5) →  -- 5 of two types and 10 of the other two types
    (0.1 : ℚ) * c.x + (0.15 : ℚ) * c.y + (0.25 : ℚ) * c.z + (0.4 : ℚ) * c.w = 7 := by
  sorry

#check li_qiang_spent_seven_yuan

end NUMINAMATH_CALUDE_ERRORFEEDBACK_li_qiang_spent_seven_yuan_l47_4710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_5_sqrt5_div_2_l47_4758

/-- Parabola defined by x²=4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point H with coordinates (1, -1) -/
def H : ℝ × ℝ := (1, -1)

/-- A point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Tangent line to the parabola at a given point -/
def tangent_line (p : PointOnParabola) (x y : ℝ) : Prop :=
  y - p.y = (1/2 * p.x) * (x - p.x)

/-- Two points A and B on the parabola -/
axiom A : PointOnParabola
axiom B : PointOnParabola

/-- Lines HA and HB are tangent to the parabola at A and B respectively -/
axiom HA_tangent : tangent_line A H.1 H.2
axiom HB_tangent : tangent_line B H.1 H.2

/-- The area of triangle ABH -/
noncomputable def area_triangle_ABH : ℝ :=
  5 * Real.sqrt 5 / 2

/-- Main theorem: The area of triangle ABH is 5√5/2 -/
theorem area_is_5_sqrt5_div_2 : area_triangle_ABH = 5 * Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_5_sqrt5_div_2_l47_4758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_centered_at_origin_l47_4791

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![5, 0; 0, 5]

theorem dilation_centered_at_origin (v : Fin 2 → ℝ) :
  dilation_matrix.mulVec v = (5 : ℝ) • v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_centered_at_origin_l47_4791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_l47_4717

/-- The locus of intersection points of two lines with given conditions -/
theorem intersection_locus (e f : Set (ℝ × ℝ)) (x y : ℝ) :
  (∃ (m₁ : ℝ), e = {(x, y) | y - 1 = m₁ * (x - 1)}) →
  (∃ (m₂ : ℝ), f = {(x, y) | y - 1 = m₂ * (x + 1)}) →
  (∃ (m₁ m₂ : ℝ), m₁ - m₂ = 2 ∨ m₂ - m₁ = 2) →
  (x, y) ∈ e ∩ f →
  (x, y) ≠ (1, 1) →
  (x, y) ≠ (-1, 1) →
  (y = x^2 ∨ y = 2 - x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_l47_4717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_snacks_at_least_35_percent_l47_4759

/-- The percentage of major airline companies that equip their planes with wireless internet access. -/
def wireless_internet_percentage : ℝ := 35

/-- The percentage of major airline companies that offer passengers free on-board snacks. -/
def free_snacks_percentage : ℝ := 35

/-- The greatest possible percentage of major airline companies that offer both wireless internet and free on-board snacks. -/
def both_services_percentage : ℝ := 35

/-- Theorem stating that the percentage of major airlines offering free on-board snacks is at least 35%. -/
theorem free_snacks_at_least_35_percent :
  free_snacks_percentage ≥ 35 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_snacks_at_least_35_percent_l47_4759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l47_4797

-- Define the functions f and g
noncomputable def f (x : ℝ) := 3 - 2 * Real.log x / Real.log 2
noncomputable def g (x : ℝ) := Real.log x / Real.log 2

-- Define the function h
noncomputable def h (x : ℝ) := (f x + 2 * g x) ^ (f x)

-- State the theorem
theorem problem_solution :
  (∃ (x : ℝ), x ∈ Set.Icc 1 4 ∧ h x = (1 : ℝ) / 3) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 1 4 ∧ h x = 27) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 1 4 → h x ≤ 27 ∧ h x ≥ (1 : ℝ) / 3) ∧
  (∀ (k : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 1 4 → f (x^2) * f (Real.sqrt x) > k * g x) ↔ k < -3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l47_4797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_15th_and_2014th_l47_4766

def customSequence : ℕ → ℕ
| 0 => 1
| n + 1 => sorry  -- Define the sequence based on the pattern

theorem sequence_15th_and_2014th :
  customSequence 14 = 25 ∧ customSequence 2013 = 3965 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_15th_and_2014th_l47_4766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l47_4726

open Real

-- Define the curves and line
noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (cos θ, 1 + sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 4 * sin (θ + π/3)

noncomputable def line_l : ℝ := π/6

-- Define the intersection points
noncomputable def point_A : ℝ × ℝ := curve_C1 line_l

noncomputable def point_B : ℝ × ℝ := 
  (curve_C2 line_l * cos line_l, curve_C2 line_l * sin line_l)

-- State the theorem
theorem intersection_distance : 
  let dist := sqrt ((point_B.1 - point_A.1)^2 + (point_B.2 - point_A.2)^2)
  dist = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l47_4726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l47_4748

-- Define the nabla operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_calculation :
  ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 →
  nabla (nabla a b) (nabla c d) = 360 / 623 := by
  intros a b c d ha hb hc hd
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l47_4748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_walk_leveling_cost_l47_4778

-- Define constants
def inner_radius : ℝ := 16
def walk_width : ℝ := 3
def cost_per_sqm : ℝ := 2

-- Define the theorem
theorem circular_walk_leveling_cost :
  let outer_radius := inner_radius + walk_width
  let walk_area := π * (outer_radius^2 - inner_radius^2)
  let total_cost := walk_area * cost_per_sqm
  ⌊total_cost⌋ = 659 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_walk_leveling_cost_l47_4778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_first_quadrant_third_quadrant_sum_of_distances_l47_4783

-- Define the point P
def P (x : ℝ) : ℝ × ℝ := (2*x, 3*x - 1)

-- Part 1
theorem angle_bisector_first_quadrant (x : ℝ) :
  (P x).1 > 0 ∧ (P x).2 > 0 ∧ (P x).1 = (P x).2 → x = 1 := by
  sorry

-- Part 2
theorem third_quadrant_sum_of_distances (x : ℝ) :
  (P x).1 < 0 ∧ (P x).2 < 0 ∧ abs (P x).1 + abs (P x).2 = 16 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_first_quadrant_third_quadrant_sum_of_distances_l47_4783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisor_l47_4735

theorem existence_of_divisor (a c : ℕ) (b : ℤ) (ha : a > 1) (hb : b ≠ 0) :
  ∃ (n : ℕ), ∃ (x : ℕ), (c * x + 1) ∣ (Int.natAbs (a^n + b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisor_l47_4735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_propositions_l47_4706

theorem quadratic_equation_propositions (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  let roots := {x : ℝ | f x = 0}
  ¬ (f 1 = 0) ∧
  (f 3 = 0) ∧
  (∃ x y, x ∈ roots ∧ y ∈ roots ∧ x + y = 2) ∧
  (∃ x y, x ∈ roots ∧ y ∈ roots ∧ x * y < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_propositions_l47_4706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arrangement_probability_l47_4701

def total_lamps : ℕ := 7
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 3
def lamps_on : ℕ := 4

def probability_specific_arrangement : ℚ := 4 / 49

theorem specific_arrangement_probability :
  (Nat.choose (total_lamps - 2) (red_lamps - 2) * Nat.choose (total_lamps - 2) (lamps_on - 2) : ℚ) /
  (Nat.choose total_lamps lamps_on * Nat.choose total_lamps red_lamps) = probability_specific_arrangement := by
  sorry

#eval probability_specific_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arrangement_probability_l47_4701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l47_4737

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, |x + 1| - |x - 2| > m) → m < -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l47_4737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l47_4786

def S (n : ℕ) : ℤ := n ^ 2 - 10 * n

def a (n : ℕ) : ℤ := S (n + 1) - S n

theorem sequence_properties :
  (∀ n : ℕ, a n = 2 * (n + 1) - 11) ∧
  (∀ n : ℕ, a 0 ≤ a n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l47_4786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_tape_theorem_l47_4784

/-- Calculates the remaining length of tape in centimeters -/
noncomputable def remaining_tape_length (initial_length : ℝ) (given_away : ℝ) (used_in_class : ℝ) : ℝ :=
  (initial_length * 10 - given_away - used_in_class) / 10

theorem remaining_tape_theorem (initial_length given_away used_in_class : ℝ) :
  initial_length = 65 →
  given_away = 125 →
  used_in_class = 153 →
  remaining_tape_length initial_length given_away used_in_class = 37.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_tape_theorem_l47_4784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2013_degrees_l47_4709

theorem sin_2013_degrees : Real.sin (2013 * π / 180) = -Real.sin (33 * π / 180) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2013_degrees_l47_4709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l47_4727

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b : Fin 2 → ℝ := ![1, -2]

-- Define perpendicularity of vectors
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

-- Define the magnitude of a vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

-- The main theorem
theorem vector_magnitude (x : ℝ) :
  perpendicular (a x) b →
  magnitude (a x + 2 • b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l47_4727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_properties_l47_4764

-- Define the ellipse (C)
def ellipseC (x y : ℝ) : Prop := x^2 / 12 + y^2 / 6 = 1

-- Define the circle (R)
def circleR (x y x₀ y₀ : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = 4

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the slope equation
def slope_equation (k₁ k₂ x₀ y₀ : ℝ) : Prop :=
  k₁ * k₂ - (k₁ + k₂) / (x₀ * y₀) + 1 = 0

theorem ellipse_tangent_properties :
  (∃ x₀ y₀ : ℝ, 
    ellipseC x₀ y₀ ∧ 
    first_quadrant x₀ y₀ ∧
    (∀ x y : ℝ, circleR x y x₀ y₀ ↔ (x - 2)^2 + (y - 2)^2 = 8)) ∧
  (¬ ∃ x₀ y₀ k₁ k₂ : ℝ,
    ellipseC x₀ y₀ ∧
    first_quadrant x₀ y₀ ∧
    slope_equation k₁ k₂ x₀ y₀) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_properties_l47_4764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_square_decomposition_for_incommensurable_rectangle_l47_4793

/-- Two real numbers are incommensurable if their ratio is irrational -/
def Incommensurable (a b : ℝ) : Prop := Irrational (a / b)

/-- A list of real numbers is pairwise distinct if all elements are different from each other -/
def PairwiseDistinct (l : List ℝ) : Prop :=
  ∀ i j, i ≠ j → l.get? i ≠ l.get? j

theorem no_square_decomposition_for_incommensurable_rectangle (a b : ℝ) 
  (h_positive : a > 0 ∧ b > 0) (h_incomm : Incommensurable a b) :
  ¬∃ (l : List ℝ), (PairwiseDistinct l) ∧ 
    (∀ x ∈ l, x > 0) ∧ 
    (l.foldl (· + ·) 0 = a * b) := by
  sorry

#check no_square_decomposition_for_incommensurable_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_square_decomposition_for_incommensurable_rectangle_l47_4793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l47_4753

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 and 18, and a distance of 5 between them, is 95. -/
theorem trapezium_area_example : trapezium_area 20 18 5 = 95 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic
  simp [mul_add, add_mul]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l47_4753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_red_two_blue_probability_l47_4728

/-- The probability of selecting two red and two blue marbles from a bag -/
theorem two_red_two_blue_probability (red : ℕ) (blue : ℕ) : 
  red = 12 → blue = 8 → (Nat.choose (red + blue) 4 : ℚ) * (1848 / 4845) = 
  (Nat.choose red 2 : ℚ) * (Nat.choose blue 2 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_red_two_blue_probability_l47_4728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_lily_arrangement_l47_4724

theorem rose_lily_arrangement (n m : ℕ) (h1 : n = 5) (h2 : m = 3) :
  (Nat.factorial (n + 1)) * (Nat.factorial m) = 4320 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_lily_arrangement_l47_4724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_point_l47_4707

/-- Given an angle α whose terminal side passes through the point (3, 4), prove that tan α = 4/3 -/
theorem tan_alpha_point (α : ℝ) (h : ∃ (t : ℝ), t > 0 ∧ t * 3 = Real.cos α ∧ t * 4 = Real.sin α) : 
  Real.tan α = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_point_l47_4707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_max_weight_l47_4754

/-- The maximum weight a truck can carry -/
def max_truck_weight (boxes box_weight crates crate_weight sacks sack_weight bags bag_weight : ℕ) : ℕ :=
  boxes * box_weight + crates * crate_weight + sacks * sack_weight + bags * bag_weight

/-- Theorem stating the maximum weight the truck can carry -/
theorem truck_max_weight :
  max_truck_weight 100 100 10 60 50 50 10 40 = 13500 := by
  -- Unfold the definition of max_truck_weight
  unfold max_truck_weight
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_max_weight_l47_4754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_m_l47_4782

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2

-- State the theorem
theorem solve_for_m :
  (∀ x : ℝ, x ≠ 0 → f (x + 1/x) = x^2 + 1/x^2) →
  ∀ m : ℝ, f m = 7 →
  m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_m_l47_4782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l47_4771

/-- The area of the figure bounded by x = 2 cos t, y = 6 sin t, and y = 3 (where y ≥ 3) -/
noncomputable def boundedArea : ℝ := 4 * Real.pi - 3 * Real.sqrt 3

/-- The parametric equations of the ellipse -/
noncomputable def ellipseParametric (t : ℝ) : ℝ × ℝ := (2 * Real.cos t, 6 * Real.sin t)

/-- The line y = 3 -/
def horizontalLine : ℝ → ℝ := fun _ ↦ 3

theorem area_of_bounded_figure :
  ∃ (t₁ t₂ : ℝ), t₁ < t₂ ∧ t₁ ∈ Set.Icc 0 (2 * Real.pi) ∧ t₂ ∈ Set.Icc 0 (2 * Real.pi) ∧
  (∀ t ∈ Set.Icc t₁ t₂, (ellipseParametric t).2 ≥ 3) ∧
  (∫ (t : ℝ) in t₁..t₂, (ellipseParametric t).1 * (6 * Real.cos t)) = boundedArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l47_4771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_divisor_bound_l47_4757

theorem composite_divisor_bound (k p n : ℕ) : 
  k > 1 → 
  Nat.Prime p → 
  n = k * p + 1 → 
  ¬Nat.Prime n → 
  (2^(n-1) - 1) % n = 0 → 
  n < 2^k := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_divisor_bound_l47_4757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_test_score_l47_4741

/-- Given two test scores, calculates their average -/
def average (score1 score2 : ℕ) : ℚ :=
  (score1 + score2) / 2

theorem second_test_score 
  (first_score : ℕ) 
  (new_average : ℚ) 
  (h1 : first_score = 78) 
  (h2 : new_average = 81) : 
  ∃ (second_score : ℕ), average first_score second_score = new_average ∧ second_score = 84 := by
  -- Proof goes here
  sorry

#check second_test_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_test_score_l47_4741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_theorem_l47_4750

noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (n : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * time)

noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem loan_difference_theorem (initialLoan : ℝ) : 
  initialLoan = 10000 →
  let compoundRate : ℝ := 0.1
  let simpleRate : ℝ := 0.12
  let compoundingPeriods : ℝ := 4
  let totalTime : ℝ := 10
  let halfTime : ℝ := 5
  let compoundAmount := compoundInterest initialLoan compoundRate compoundingPeriods halfTime
  let halfPayment := compoundAmount / 2
  let remainingCompound := compoundInterest halfPayment compoundRate compoundingPeriods halfTime
  let totalCompound := halfPayment + remainingCompound
  let totalSimple := simpleInterest initialLoan simpleRate totalTime
  roundToNearest (totalSimple - totalCompound) = 382 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_theorem_l47_4750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_z_z_in_second_quadrant_l47_4732

-- Define the complex number z as a function of m
noncomputable def z (m : ℝ) : ℂ := (m * (m - 2)) / (m - 1) + (m^2 + 2*m - 3) * Complex.I

-- Theorem for purely imaginary z
theorem purely_imaginary_z (m : ℝ) :
  z m = Complex.I * (z m).im ↔ m = 0 ∨ m = 2 := by sorry

-- Theorem for z in the second quadrant
theorem z_in_second_quadrant (m : ℝ) :
  (z m).re < 0 ∧ (z m).im > 0 ↔ m < -3 ∨ (1 < m ∧ m < 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_z_z_in_second_quadrant_l47_4732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_intersection_A_complement_B_domain_f_l47_4719

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -8 < x ∧ x < -2}
def B : Set ℝ := {x : ℝ | x < -3}

-- Define function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - 2^x) + Real.log (x + 1)

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x : ℝ | x < -2} := by sorry

-- Theorem for A ∩ (ℝ \ B)
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -3 ≤ x ∧ x < -2} := by sorry

-- Theorem for the domain of f
theorem domain_f : {x : ℝ | ∃ y, f x = y} = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_intersection_A_complement_B_domain_f_l47_4719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l47_4752

theorem max_subset_size (n : ℕ) :
  let M := Finset.range (2 * n + 2) \ {0}
  (∃ (A : Finset ℕ), A ⊆ M ∧ (∀ a b : ℕ, a ∈ A → b ∈ A → a + b ≠ 2 * n + 2) ∧ A.card = n + 1) ∧
  (∀ (B : Finset ℕ), B ⊆ M → (∀ a b : ℕ, a ∈ B → b ∈ B → a + b ≠ 2 * n + 2) → B.card ≤ n + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l47_4752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l47_4749

/-- The eccentricity of a hyperbola with specific intersection properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (F₁ F₂ P Q : ℝ × ℝ) :
  let hyperbola := λ (p : ℝ × ℝ) ↦ p.1^2 / a^2 - p.2^2 / b^2 = 1
  let right_branch := λ (p : ℝ × ℝ) ↦ p.1 > 0 ∧ hyperbola p
  let c := Real.sqrt (a^2 + b^2)
  F₁ = (-c, 0) →
  F₂ = (c, 0) →
  (∃ (t : ℝ), P = (t * F₂.1, t * F₂.2) ∧ right_branch P) →
  (∃ (s : ℝ), Q = (s * F₂.1, s * F₂.2) ∧ right_branch Q) →
  ((P.1 - F₁.1) * (Q.1 - P.1) + (P.2 - F₁.2) * (Q.2 - P.2) = 0) →
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 5/12 * Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) →
  c / a = Real.sqrt 37 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l47_4749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_angles_l47_4704

theorem cos_difference_angles (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1/2) 
  (h2 : Real.sin α + Real.sin β = 1/3) : 
  Real.cos (α - β) = -59/72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_angles_l47_4704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_l47_4774

theorem min_xy_value (x y : ℝ) 
  (h : 1 + (Real.cos (x + y - 1))^2 = (x^2 + y^2 + 2*(x + 1)*(1 - y)) / (x - y + 1)) : 
  (∀ z : ℝ, x * y ≥ (1/4 : ℝ)) ∧ ∃ x₀ y₀ : ℝ, x₀ * y₀ = (1/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_l47_4774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l47_4713

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal length 2c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < b ∧ b < a
  h_eq : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The sum of distances from any point on the ellipse to its foci -/
noncomputable def sum_distances (e : Ellipse) : ℝ := 2 * e.a

theorem ellipse_eccentricity_half (e : Ellipse) 
  (h_arithmetic : ∃ (d₁ d₂ : ℝ), d₁ + d₂ = sum_distances e ∧ d₂ - 2*e.c = 2*e.c - d₁) :
  eccentricity e = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l47_4713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l47_4723

theorem problem_statement (a b : ℝ) 
  (h : ({1, a + b, a} : Set ℝ) = {0, b / a, b}) : 
  b^2010 - a^2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l47_4723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_28_l47_4779

-- Define the vertices of the triangle
def A : ℚ × ℚ := (5, -2)
def B : ℚ × ℚ := (12, 6)
def C : ℚ × ℚ := (5, 6)

-- Define the area calculation function
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Theorem statement
theorem triangle_area_is_28 : 
  triangleArea A B C = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_28_l47_4779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_periodic_decimal_l47_4746

theorem mixed_periodic_decimal (n : ℕ+) :
  ∃ (a b : ℕ) (k : ℕ+), 
    (3 * n^2 + 6 * n + 2 : ℚ) / (n * (n + 1) * (n + 2)) = a + b / (10^(k : ℕ) - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_periodic_decimal_l47_4746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l47_4796

theorem partial_fraction_decomposition_product :
  ∃ A B C : ℝ, 
    (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 3 → 
      (x^2 - 2*x - 15) / (x^3 - 3*x^2 - x + 3) = 
      A/(x-1) + B/(x+1) + C/(x-3)) → 
    A * B * C = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l47_4796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_two_pi_thirds_side_values_l47_4721

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the condition ab = c² - a² - b²
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * t.b = t.c^2 - t.a^2 - t.b^2

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ :=
  Real.sqrt 3 * (t.a * t.b / 4)

-- Theorem 1
theorem angle_C_is_two_pi_thirds (t : Triangle) (h : satisfiesCondition t) :
  ∃ (C : ℝ), C = 2 * Real.pi / 3 ∧ C = Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)) := by
  sorry

-- Theorem 2
theorem side_values (t : Triangle) (h1 : satisfiesCondition t) (h2 : area t = 2 * Real.sqrt 3) (h3 : t.c = 2 * Real.sqrt 7) :
  (t.a = 2 ∧ t.b = 4) ∨ (t.a = 4 ∧ t.b = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_two_pi_thirds_side_values_l47_4721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_theorem_l47_4787

def is_valid_polynomial (P : ℂ → ℂ) : Prop :=
  ∃ (a b c d : ℤ), ∀ z, P z = z^5 + a*z^4 + b*z^3 + c*z^2 + d*z + (P 0)

def has_three_integer_roots (P : ℂ → ℂ) : Prop :=
  ∃ (p q r : ℤ), P p = 0 ∧ P q = 0 ∧ P r = 0

theorem polynomial_root_theorem (P : ℂ → ℂ) :
  is_valid_polynomial P →
  has_three_integer_roots P →
  P (Complex.mk (3/2) (Real.sqrt 15/2)) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_theorem_l47_4787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l47_4705

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

-- Define the point of interest
def point : ℝ × ℝ := (1, -5/3)

-- Theorem statement
theorem tangent_slope_angle :
  let df := deriv f
  let slope := df point.fst
  let angle := Real.arctan slope
  angle = π/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l47_4705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l47_4716

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem necessary_not_sufficient
  (a b : Line) (α β : Plane)
  (h1 : parallel a b)
  (h2 : perpendicular_line_plane b α) :
  (∀ (a : Line) (β : Plane),
    parallel_line_plane a β →
    perpendicular_plane_plane α β) ∧
  (∃ (a : Line) (β : Plane),
    perpendicular_plane_plane α β ∧
    ¬ parallel_line_plane a β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l47_4716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_3_equals_11_l47_4760

-- Define the functions t and s
noncomputable def t (x : ℝ) : ℝ := 5*x - 7
noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 7) / 5  -- Inverse of t(x)
  x^2 + 4*x - 1

-- Theorem statement
theorem s_of_3_equals_11 : s 3 = 11 := by
  -- Unfold the definition of s
  unfold s
  -- Simplify the expression
  simp
  -- The rest of the proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_3_equals_11_l47_4760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l47_4747

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The addition problem structure -/
structure AdditionProblem where
  F : Digit
  I : Digit
  V : Digit
  T : Digit
  E : Digit
  N : Digit

/-- Checks if all digits in the problem are different -/
def allDifferent (p : AdditionProblem) : Prop :=
  p.F ≠ p.I ∧ p.F ≠ p.V ∧ p.F ≠ p.T ∧ p.F ≠ p.E ∧ p.F ≠ p.N ∧
  p.I ≠ p.V ∧ p.I ≠ p.T ∧ p.I ≠ p.E ∧ p.I ≠ p.N ∧
  p.V ≠ p.T ∧ p.V ≠ p.E ∧ p.V ≠ p.N ∧
  p.T ≠ p.E ∧ p.T ≠ p.N ∧
  p.E ≠ p.N

/-- Checks if the addition is valid -/
def isValidAddition (p : AdditionProblem) : Prop :=
  100 * p.F.val + 10 * p.I.val + p.V.val +
  100 * p.F.val + 10 * p.I.val + p.V.val =
  100 * p.T.val + 10 * p.E.val + p.N.val

/-- Main theorem -/
theorem unique_solution :
  ∀ (p : AdditionProblem),
    allDifferent p →
    isValidAddition p →
    p.F.val = 8 →
    p.V.val % 2 = 0 →
    p.I.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l47_4747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l47_4743

/-- A parabola with vertex (-1, -4) passing through (0, -3) -/
structure Parabola where
  equation : ℝ → ℝ
  vertex : equation (-1) = -4
  passes_through : equation 0 = -3

/-- Theorem stating the properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (∀ x, p.equation x = (x + 1)^2 - 4) ∧
  p.equation 1 = 0 ∧ p.equation (-3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l47_4743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_description_is_correct_l47_4744

/-- Represents the ratio of current egg consumption to past egg consumption -/
def consumption_ratio : ℝ := 2.1  -- Example value, more than 2

/-- Assumption that the current consumption is more than double the past consumption -/
axiom more_than_double : consumption_ratio > 2

/-- The phrase used to describe the increase in consumption -/
def description : String := "more than twice as many"

/-- Theorem stating that the description is correct given the consumption ratio -/
theorem description_is_correct : description = "more than twice as many" := by
  rfl  -- Reflexivity proves the equality

#check description_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_description_is_correct_l47_4744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_conditions_imply_sum_l47_4715

-- Define the equations as noncomputable
noncomputable def equation1 (p q x : ℝ) : ℝ := (x + p) * (x + q) * (x - 7) / ((x - 2) ^ 2)
noncomputable def equation2 (p q x : ℝ) : ℝ := (x + 2 * p) * (x - 2) * (x - 10) / ((x + q) * (x - 7))

-- State the theorem
theorem root_conditions_imply_sum (p q : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    equation1 p q x₁ = 0 ∧ equation1 p q x₂ = 0 ∧ equation1 p q x₃ = 0 ∧
    (∀ x : ℝ, equation1 p q x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
  (∃! x : ℝ, equation2 p q x = 0) →
  50 * p + 3 * q = -80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_conditions_imply_sum_l47_4715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_in_second_quadrant_l47_4740

theorem angle_sum_in_second_quadrant (θ : Real) :
  θ ∈ Set.Icc (π/2) π →
  Real.tan (θ + π/4) = 1/2 →
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_in_second_quadrant_l47_4740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_lines_intersect_l47_4714

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a line that intersects the square
structure IntersectingLine where
  start : ℝ × ℝ
  ending : ℝ × ℝ

-- Define the property of dividing the square into quadrilaterals with area ratio 2:3
def dividesIntoQuadrilateralsWithRatio (s : Square) (l : IntersectingLine) : Prop :=
  ∃ (a₁ a₂ : ℝ), a₁ > 0 ∧ a₂ > 0 ∧ a₁ / a₂ = 2 / 3 ∧
  a₁ + a₂ = s.side * s.side

-- Define the theorem
theorem at_least_three_lines_intersect (s : Square) 
  (lines : Finset IntersectingLine) 
  (h_count : lines.card = 9)
  (h_divide : ∀ l ∈ lines, dividesIntoQuadrilateralsWithRatio s l) :
  ∃ (p : ℝ × ℝ), ∃ (l₁ l₂ l₃ : IntersectingLine),
    l₁ ∈ lines ∧ l₂ ∈ lines ∧ l₃ ∈ lines ∧
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ ∧
    (p ∈ Set.Icc l₁.start l₁.ending) ∧
    (p ∈ Set.Icc l₂.start l₂.ending) ∧
    (p ∈ Set.Icc l₃.start l₃.ending) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_lines_intersect_l47_4714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adults_who_ate_correct_l47_4755

/-- Represents the meal capacity in terms of adults and children -/
structure MealCapacity where
  adults : Nat
  children : Nat

/-- Represents the group of people and meal situation -/
structure TrekkingGroup where
  totalMealCapacity : MealCapacity
  remainingChildrenCapacity : Nat

/-- Calculates the number of adults who had their meal -/
def adultsWhoAte (group : TrekkingGroup) : Nat :=
  (group.totalMealCapacity.children - group.remainingChildrenCapacity) * 
  group.totalMealCapacity.adults / group.totalMealCapacity.children

theorem adults_who_ate_correct (group : TrekkingGroup) : 
  group.totalMealCapacity.adults = 70 ∧ 
  group.totalMealCapacity.children = 90 ∧ 
  group.remainingChildrenCapacity = 45 → 
  adultsWhoAte group = 35 := by
  sorry

#eval adultsWhoAte { totalMealCapacity := { adults := 70, children := 90 }, remainingChildrenCapacity := 45 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adults_who_ate_correct_l47_4755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_condition_l47_4736

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x - b * Real.sin x + 3

theorem inverse_condition (a b : ℝ) :
  Function.Injective (f a b) ↔ (b = 0 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_condition_l47_4736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_product_inequality_l47_4708

/-- The least common multiple of a list of positive natural numbers. -/
def lcm_list (l : List ℕ+) : ℕ+ :=
  l.foldl (fun a b => (a.lcm b : ℕ+)) 1

/-- Theorem: For positive integers k, m, and n, the product of pairwise LCMs is greater than or equal to the square of the LCM of all three. -/
theorem lcm_product_inequality (k m n : ℕ+) :
  (lcm_list [k, m]) * (lcm_list [m, n]) * (lcm_list [n, k]) ≥ (lcm_list [k, m, n])^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_product_inequality_l47_4708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_fraction_integer_l47_4729

theorem binomial_fraction_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  (((3 * n - 4 * k + 2) / (k + 2)) * (Nat.choose n k : ℚ)).isInt ↔
  (k + 2) ∣ n := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_fraction_integer_l47_4729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l47_4792

noncomputable def f (a b x : ℝ) : ℝ := (b - 2^x) / (2^(x+1) + a)

theorem odd_function_properties (a b : ℝ) 
  (h_odd : ∀ x, f a b x = -f a b (-x)) :
  (a = 2 ∧ b = 1) ∧
  (∀ x y, x < y → f a b x > f a b y) ∧
  (∀ k, (∀ x : ℝ, x ≥ 1 → f a b (k * 3^x) + f a b (3^x - 9^x + 2) > 0) → k < 4/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l47_4792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_lambda_l47_4775

-- Define the hyperbola and its properties
def Hyperbola (lambda : ℝ) := {P : ℝ × ℝ | P.1^2 - P.2^2 = lambda}

def LeftBranch (lambda : ℝ) := {P : ℝ × ℝ | P ∈ Hyperbola lambda ∧ P.1 < 0}

noncomputable def RightFocus (lambda : ℝ) : ℝ × ℝ := (Real.sqrt (2 * lambda), 0)

noncomputable def LeftFocus (lambda : ℝ) : ℝ × ℝ := (-Real.sqrt (2 * lambda), 0)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem hyperbola_lambda (lambda : ℝ) (P : ℝ × ℝ) (h1 : lambda > 0) (h2 : P ∈ LeftBranch lambda)
  (h3 : distance P (RightFocus lambda) = 6)
  (h4 : P.2 = 0)  -- This represents PF1 being perpendicular to the real axis
  : lambda = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_lambda_l47_4775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l47_4770

/-- A triangle type -/
structure Triangle where
  -- Define the necessary fields for a triangle

/-- A square type -/
structure Square where
  side : ℝ

/-- Predicate for an isosceles right triangle -/
def IsoscelesRightTriangle (t : Triangle) : Prop := sorry

/-- Predicate for a square inscribed in a triangle -/
def InscribedSquare (s : Square) (t : Triangle) : Prop := sorry

/-- Predicate for a square with one side along the hypotenuse of a triangle -/
def sideAlongHypotenuse (s : Square) (t : Triangle) : Prop := sorry

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side ^ 2

/-- Predicate for a square touching one leg of a triangle -/
def touchesOneLeg (s : Square) (t : Triangle) : Prop := sorry

/-- Predicate for a square touching two points on the hypotenuse of a triangle -/
def touchesTwoPointsOnHypotenuse (s : Square) (t : Triangle) : Prop := sorry

/-- Given an isosceles right triangle ABC with a square of area 400 cm² inscribed along its hypotenuse,
    the area of a square inscribed in the same triangle touching one leg and two points on the hypotenuse
    is 1600/9 cm². -/
theorem inscribed_square_area (ABC : Triangle) (S₁ S₂ : Square) :
  IsoscelesRightTriangle ABC →
  InscribedSquare S₁ ABC →
  sideAlongHypotenuse S₁ ABC →
  S₁.area = 400 →
  InscribedSquare S₂ ABC →
  touchesOneLeg S₂ ABC →
  touchesTwoPointsOnHypotenuse S₂ ABC →
  S₂.area = 1600 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l47_4770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_l47_4742

noncomputable def parametric_x (α : ℝ) : ℝ := 3 * Real.cos α + 1
noncomputable def parametric_y (α : ℝ) : ℝ := -Real.cos α

theorem parametric_to_ordinary :
  ∀ x y : ℝ,
  (∃ α : ℝ, x = parametric_x α ∧ y = parametric_y α) ↔
  (x + 3 * y - 1 = 0 ∧ -2 ≤ x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_l47_4742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l47_4730

/-- Calculates the length of a train given the speeds of two trains, the time they take to clear each other, and the length of the other train. -/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (clearing_time : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * clearing_time
  total_distance - other_train_length

/-- The theorem stating the length of the first train given the problem conditions. -/
theorem first_train_length :
  let speed1 : ℝ := 80  -- km/h
  let speed2 : ℝ := 65  -- km/h
  let clearing_time : ℝ := 7.596633648618456  -- seconds
  let second_train_length : ℝ := 165  -- meters
  let first_train_length := calculate_train_length speed1 speed2 clearing_time second_train_length
  ∃ ε > 0, abs (first_train_length - 141.019) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l47_4730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_le_half_parallelogram_area_l47_4722

/-- A parallelogram in a 2D plane -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : Sorry

/-- A triangle with vertices on the sides of a parallelogram -/
structure TriangleOnParallelogram (p : Parallelogram) where
  vertices : Fin 3 → ℝ × ℝ
  on_sides : ∀ i, ∃ j k, (vertices i) ∈ Set.Icc (p.vertices j) (p.vertices k)

/-- The area of a geometric shape -/
noncomputable def area (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of a triangle on a parallelogram is at most half the area of the parallelogram -/
theorem triangle_area_le_half_parallelogram_area (p : Parallelogram) (t : TriangleOnParallelogram p) :
  area t.vertices ≤ (1/2) * area p.vertices := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_le_half_parallelogram_area_l47_4722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_cost_l47_4761

/-- Represents the cost and rate of a water source -/
structure WaterSource where
  rate : ℝ  -- gallons per hour
  cost : ℝ  -- cents per gallon

/-- Calculates the total cost to fill a pool -/
noncomputable def total_cost (pool_volume : ℝ) (hose : WaterSource) (pump : WaterSource) : ℝ :=
  let combined_rate := hose.rate + pump.rate
  let time_to_fill := pool_volume / combined_rate
  (hose.rate * time_to_fill * hose.cost + pump.rate * time_to_fill * pump.cost) / 100

/-- Theorem stating the total cost to fill the pool -/
theorem pool_fill_cost :
  let pool_volume : ℝ := 5000
  let hose : WaterSource := { rate := 100, cost := 0.1 }
  let pump : WaterSource := { rate := 150, cost := 0.125 }
  total_cost pool_volume hose pump = 5.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_cost_l47_4761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l47_4788

-- Define a hyperbola
structure Hyperbola where
  equation : ℝ → ℝ → ℝ → Prop
  asymptotes : ℝ → ℝ → Prop

-- Define proposition α
def α (h : Hyperbola) : Prop :=
  ∃ a : ℝ, a > 0 ∧ h.equation = fun x y _ => x^2 - y^2 = a^2

-- Define proposition β
def β (h : Hyperbola) : Prop :=
  ∃ θ : ℝ, θ = Real.pi/2 ∧ h.asymptotes = fun x y => y = x ∨ y = -x

-- Theorem statement
theorem hyperbola_asymptotes (h : Hyperbola) :
  (α h → β h) ∧ ∃ h', β h' ∧ ¬(α h') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l47_4788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_watch_display_sum_l47_4795

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

def watch_display_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

theorem max_watch_display_sum :
  ∃ (h m : ℕ), is_valid_hour h ∧ is_valid_minute m ∧
  ∀ (h' m' : ℕ), is_valid_hour h' → is_valid_minute m' →
  watch_display_sum h' m' ≤ watch_display_sum h m ∧
  watch_display_sum h m = 23 :=
by
  sorry

#eval watch_display_sum 23 59

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_watch_display_sum_l47_4795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_origin_passing_tangents_l47_4763

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2

-- State the theorem
theorem tangent_line_and_origin_passing_tangents 
  (a : ℝ) 
  (h1 : f a 1 = 5) : 
  -- Part 1: Equation of tangent line at (1, 5)
  (∃ m b : ℝ, m = 11 ∧ b = -6 ∧ 
    ∀ x : ℝ, (deriv (f a)) 1 * (x - 1) + f a 1 = m * x + b) ∧
  -- Part 2: Tangent lines passing through origin
  (∃ x1 x2 : ℝ, x1 = 0 ∧ x2 = -2 ∧
    (deriv (f a)) x1 * x1 = f a x1 ∧
    (deriv (f a)) x2 * x2 = f a x2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_origin_passing_tangents_l47_4763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stamps_purchased_l47_4794

theorem max_stamps_purchased (stamp_price : ℕ) (total_amount : ℕ) : 
  stamp_price = 33 → total_amount = 3200 → 
  (∃ (n : ℕ), n * stamp_price ≤ total_amount ∧ 
    ∀ (m : ℕ), m * stamp_price ≤ total_amount → m ≤ n) → 
  ∃ (n : ℕ), n = 96 ∧ n * stamp_price ≤ total_amount ∧ 
    ∀ (m : ℕ), m * stamp_price ≤ total_amount → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stamps_purchased_l47_4794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_equation_solution_l47_4739

theorem sin_3x_equation_solution (n : ℝ) :
  (∃ x : ℝ, Real.sin (3 * x) = (n^2 - 5*n + 3) * Real.sin x ∧ Real.sin x ≠ 0) ↔
  (1 ≤ n ∧ n ≤ 4) ∨ n = (5 + Real.sqrt 17) / 2 ∨ n = (5 - Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_equation_solution_l47_4739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l47_4718

theorem remainder_problem (y : ℕ) (h : (7 * y) % 29 = 1) : (15 + y) % 29 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l47_4718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_point_always_on_graph_l47_4765

-- Define the function f(x) = a^(x-2) + 2
noncomputable def f (a x : ℝ) : ℝ := a^(x-2) + 2

-- State the theorem
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 3 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the exponent
  simp [pow_zero]
  -- Evaluate the expression
  norm_num

-- Prove that (2, 3) is always on the graph for any valid a
theorem point_always_on_graph (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ f a x = y := by
  -- Use the point (2, 3)
  use 2, 3
  -- Split the goal into three parts
  constructor
  · rfl  -- x = 2
  constructor
  · rfl  -- y = 3
  · -- Show f a 2 = 3
    exact fixed_point_of_exponential_function a h1 h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_point_always_on_graph_l47_4765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_w_is_10_l47_4700

/-- Given proportionality relationships between x, y, z, and w, prove that x = 1/8 when w = 10 -/
theorem x_value_when_w_is_10 
  (x y z : ℝ → ℝ) (m n k : ℝ) 
  (h1 : ∀ w, x w = m * (y w)^3)
  (h2 : ∀ w, y w = n / (z w)^(1/2))
  (h3 : ∀ w, z w = k * w^2)
  (h4 : x 5 = 8) :
  x 10 = 1/8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_w_is_10_l47_4700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l47_4772

open Real Set

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x - π / 6)

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ∈ Icc 0 (π / 2), f x = y} = Icc (-3/2) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l47_4772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_percentage_l47_4734

/-- Represents the cost, markup, and discount for an item -/
structure Item where
  cost : ℝ
  markup : ℝ
  discount : ℝ

/-- Calculates the selling price of an item -/
def sellingPrice (item : Item) : ℝ :=
  item.cost + item.markup - item.discount

/-- Calculates the profit for an item -/
def profit (item : Item) : ℝ :=
  sellingPrice item - item.cost

/-- Calculates the total cost of a list of items -/
def totalCost (items : List Item) : ℝ :=
  items.foldl (fun acc item => acc + item.cost) 0

/-- Calculates the total profit of a list of items -/
def totalProfit (items : List Item) : ℝ :=
  items.foldl (fun acc item => acc + profit item) 0

/-- Calculates the profit percentage -/
noncomputable def profitPercentage (items : List Item) : ℝ :=
  (totalProfit items / totalCost items) * 100

theorem merchant_profit_percentage : 
  let items : List Item := [
    { cost := 50, markup := 15, discount := 10 },
    { cost := 75, markup := 20, discount := 5 },
    { cost := 100, markup := 40, discount := 15 },
    { cost := 150, markup := 60, discount := 20 },
    { cost := 200, markup := 80, discount := 30 }
  ]
  abs (profitPercentage items - 23.48) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_percentage_l47_4734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_valid_cube_configurations_l47_4776

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
| Center
| End

/-- Represents a configuration of squares -/
structure SquareConfiguration where
  baseSquares : Fin 6 → Unit
  additionalSquare : AttachmentPosition

/-- Predicate to determine if a configuration can be folded into a cube with one face missing -/
def canFoldToCube (config : SquareConfiguration) : Bool :=
  match config.additionalSquare with
  | AttachmentPosition.Center => true
  | AttachmentPosition.End => false

/-- The set of all possible configurations -/
def allConfigurations : Finset SquareConfiguration := sorry

theorem three_valid_cube_configurations :
  (allConfigurations.filter (fun c => canFoldToCube c)).card = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_valid_cube_configurations_l47_4776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_with_sectors_l47_4769

/-- The area of the shaded region in a regular hexagon with circular sectors --/
theorem shaded_area_hexagon_with_sectors (s r : ℝ) (h1 : s = 8) (h2 : r = 4) :
  let hexagon_area := 6 * (Real.sqrt 3 / 4 * s^2)
  let sector_area := 6 * (Real.pi * r^2 / 6)
  hexagon_area - sector_area = 96 * Real.sqrt 3 - 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_with_sectors_l47_4769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l47_4711

-- Define the coordinates of point A
def A : ℝ × ℝ := (0, 1)

-- Define the curve that point B moves along
noncomputable def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define a point B on the curve
noncomputable def B (x₀ : ℝ) : ℝ × ℝ := (x₀, curve x₀)

-- Define the midpoint M of line segment AB
noncomputable def M (x₀ : ℝ) : ℝ × ℝ := ((x₀ + A.1) / 2, (curve x₀ + A.2) / 2)

-- Theorem statement
theorem midpoint_trajectory :
  ∀ x : ℝ, (M x).2 = 4 * (M x).1^2 + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l47_4711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_13_l47_4781

-- Define the arithmetic square root
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Theorem statement
theorem arithmetic_sqrt_13 : arithmetic_sqrt 13 = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_13_l47_4781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l47_4762

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / x - Real.log ((1 + a * x) / (1 - x)) / Real.log 2

theorem odd_function_implies_a_equals_one :
  (∀ x : ℝ, f a (-x) = -(f a x)) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l47_4762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_l47_4789

/-- Calculates the gain percent given the cost price and selling price -/
noncomputable def gain_percent (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The gain percent is 60% when a cycle is bought for Rs. 900 and sold for Rs. 1440 -/
theorem cycle_gain_percent :
  let cost_price : ℝ := 900
  let selling_price : ℝ := 1440
  gain_percent cost_price selling_price = 60 := by
  -- Unfold the definition of gain_percent
  unfold gain_percent
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_l47_4789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_square_in_sequence_l47_4731

/-- Defines a sequence by concatenating "2014" repeatedly -/
def our_sequence (n : ℕ) : ℕ :=
  let base := 2014
  Nat.ofDigits 10 (List.replicate (n + 1) base)

/-- Theorem: No number in the sequence is a perfect square -/
theorem no_perfect_square_in_sequence :
  ∀ n : ℕ, ¬∃ m : ℕ, our_sequence n = m^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_square_in_sequence_l47_4731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l47_4799

-- Define the hyperbola E
def E (x y : ℝ) : Prop := y^2 / 9 - x^2 / 4 = 1

-- Define the shared asymptotes hyperbola
def shared_asymptotes (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1

-- Define a point on the hyperbola E
def point_on_E : Prop := E (-2) (3 * Real.sqrt 2)

-- Define a line intersecting E at two points
def intersecting_line (A B : ℝ × ℝ) : Prop := 
  E A.1 A.2 ∧ E B.1 B.2

-- Define the midpoint of a line segment
def midpoint_of (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define a tangent line to E at a point
def tangent_line (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  E P.1 P.2 → 9 * P.1 * x - 4 * P.2 * y + 36 = 0

-- Main theorem
theorem hyperbola_properties :
  point_on_E →
  (∀ x y, E x y ↔ y^2 / 9 - x^2 / 4 = 1) ∧
  (∀ A B, intersecting_line A B → midpoint_of A B (1, 4) → 
    ∀ x y, 9*x - 16*y + 55 = 0 ↔ (y - 4 = (9/16) * (x - 1))) ∧
  (∀ A, E A.1 A.2 → ∀ x y, tangent_line A x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l47_4799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l47_4712

-- Define the curve (marked as noncomputable due to dependency on Real.exp)
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the point of tangency
def P : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem tangent_line_at_P :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ x - y + 1 = 0) ∧ 
    (f P.1 = P.2) ∧
    (Set.EqOn (fun x => m * x + b) f {P.1}) ∧
    (HasDerivAt f (Real.exp P.1) P.1) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l47_4712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blake_room_count_l47_4756

/-- The number of rooms Blake can prime and paint. -/
def number_of_rooms : ℕ :=
  let primer_original_price : ℚ := 30
  let primer_discount : ℚ := 20 / 100
  let paint_price : ℚ := 25
  let total_spent : ℚ := 245
  let primer_discounted_price : ℚ := primer_original_price * (1 - primer_discount)
  let cost_per_room : ℚ := primer_discounted_price + paint_price
  (total_spent / cost_per_room).floor.toNat

theorem blake_room_count :
  number_of_rooms = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blake_room_count_l47_4756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_set_primes_l47_4790

def An (n : ℕ) : Set ℕ :=
  {k : ℕ | 1 < k ∧ k < n ∧ Nat.Coprime k n}

def all_elements_prime (s : Set ℕ) : Prop :=
  ∀ x ∈ s, Nat.Prime x

theorem coprime_set_primes :
  {n : ℕ | n > 2 ∧ all_elements_prime (An n)} =
  {3, 4, 6, 8, 12, 18, 24, 30} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_set_primes_l47_4790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l47_4738

theorem problem_solution (X : ℝ) : 
  (213 * 16 = 3408) → 
  ((213 * 16) + (1.6 * 2.13) = X) → 
  (X - ((5/2) * 1.25) + 3 * Real.log 8 / Real.log 2 + Real.sin (π/2) = 3418.283) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l47_4738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l47_4773

/-- Represents the tank and its properties -/
structure Tank where
  capacity : ℚ
  initial_level : ℚ
  inflow_rate : ℚ
  outflow_rate1 : ℚ
  outflow_rate2 : ℚ

/-- Calculates the time required to fill the tank completely -/
def time_to_fill (tank : Tank) : ℚ :=
  let net_flow_rate := tank.inflow_rate - (tank.outflow_rate1 + tank.outflow_rate2)
  let remaining_volume := tank.capacity - tank.initial_level
  remaining_volume / net_flow_rate

/-- Theorem stating that the time to fill the tank is 60 minutes -/
theorem tank_fill_time :
  let tank := Tank.mk 10 5 (1/2) (1/4) (1/6)
  time_to_fill tank = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l47_4773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_identity_l47_4780

theorem trig_sum_identity (A B C : Real) 
  (h1 : Real.sin A + Real.sin B + Real.sin C = 0)
  (h2 : Real.cos A + Real.cos B + Real.cos C = 0) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_identity_l47_4780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l47_4702

/-- Helper function to calculate the area of a quadrilateral -/
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  sorry -- Actual implementation would go here

/-- The area of a quadrilateral formed by the intersection of four lines -/
theorem quadrilateral_area (l₁ l₂ l₃ l₄ : ℝ × ℝ → Prop) : 
  (∀ x y, l₁ (x, y) ↔ x - 4*y + 5 = 0) →
  (∀ x y, l₂ (x, y) ↔ 2*x + y - 8 = 0) →
  (∀ x y, l₃ (x, y) ↔ x - 4*y + 14 = 0) →
  (∀ x y, l₄ (x, y) ↔ 2*x + y + 1 = 0) →
  ∃ A B C D : ℝ × ℝ, 
    l₁ A ∧ l₂ A ∧
    l₂ B ∧ l₃ B ∧
    l₃ C ∧ l₄ C ∧
    l₄ D ∧ l₁ D ∧
    area_quadrilateral A B C D = 27 * Real.sqrt 34 / 17 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l47_4702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_self_transforming_periodic_sequence_l47_4703

-- Define a type for the letters in the sequence
inductive Letter : Type
| A : Letter
| B : Letter

-- Define an instance of Inhabited for Letter
instance : Inhabited Letter := ⟨Letter.A⟩

-- Define a function to represent the transformation rule
def transform (l : Letter) : List Letter :=
  match l with
  | Letter.A => [Letter.A, Letter.B, Letter.A]
  | Letter.B => [Letter.B, Letter.B, Letter.A]

-- Define an infinite sequence type
def InfiniteSeq := ℕ → Letter

-- Define a property for periodic sequences
def isPeriodic (s : InfiniteSeq) (n : ℕ) : Prop :=
  ∀ i : ℕ, s i = s (i + n)

-- Define a property for self-transforming sequences
def isSelfTransforming (s : InfiniteSeq) : Prop :=
  ∃ k : ℕ, ∀ i : ℕ,
    (transform (s i)).get! 0 = s (3*i + k) ∧
    (transform (s i)).get! 1 = s (3*i + k + 1) ∧
    (transform (s i)).get! 2 = s (3*i + k + 2)

-- The main theorem
theorem no_self_transforming_periodic_sequence :
  ¬∃ (s : InfiniteSeq) (n : ℕ), n > 0 ∧ isPeriodic s n ∧ isSelfTransforming s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_self_transforming_periodic_sequence_l47_4703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defeat_dragon_l47_4745

/-- Represents the state of the Dragon's heads after each swing -/
structure DragonState :=
  (heads : ℕ)

/-- Calculates the number of heads after one swing -/
def swing (state : DragonState) : DragonState :=
  let remainingHeads := state.heads - min state.heads 5
  let newHeads := remainingHeads % 9
  { heads := remainingHeads + newHeads }

/-- Determines if the Dragon is defeated (5 or fewer heads) -/
def isDefeated (state : DragonState) : Prop :=
  state.heads ≤ 5

/-- Theorem: It takes exactly 40 swings to defeat the Dragon -/
theorem defeat_dragon (initialHeads : ℕ) (h : initialHeads = 198) :
  ∃ n : ℕ, n = 40 ∧ 
  isDefeated ((Nat.iterate swing n) { heads := initialHeads }) ∧
  ∀ m : ℕ, m < 40 → ¬isDefeated ((Nat.iterate swing m) { heads := initialHeads }) := by
  sorry

#check defeat_dragon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defeat_dragon_l47_4745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_a_l47_4767

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then Real.sqrt (x + 1)
  else if x ≥ 0 then 2 * x
  else 0  -- This case is added to make the function total

-- State the theorem
theorem f_inverse_a (a : ℝ) (h1 : f a = f (a - 1)) : f (1 / a) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_a_l47_4767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_green_points_l47_4733

theorem min_green_points (total_points : ℕ) (distance : ℝ) 
  (h_total : total_points = 2020)
  (h_distance : distance = 2020) :
  ∃ (green_points : ℕ),
    green_points ≥ 46 ∧
    ∀ (black_points : ℕ),
      black_points + green_points = total_points →
      (∀ (b : ℕ), b < black_points → 
        ∃! (g1 g2 : ℕ), g1 < green_points ∧ g2 < green_points ∧ g1 ≠ g2 ∧
          (∃ (p_b p_g1 p_g2 : ℝ × ℝ),
            ‖p_g1 - p_b‖ = distance ∧ ‖p_g2 - p_b‖ = distance)) →
      green_points = 46 := by
  sorry

#check min_green_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_green_points_l47_4733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_discount_problem_l47_4785

/-- Calculates the final price of a saree after two sequential discounts -/
noncomputable def final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100)

/-- Theorem stating that for a saree initially priced at 400, 
    if two discounts are applied sequentially, with the second discount being 5%, 
    and the final price is 304, then the first discount must be 20% -/
theorem saree_discount_problem (discount1 : ℝ) :
  final_price 400 discount1 5 = 304 → discount1 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_discount_problem_l47_4785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l47_4798

/-- Given a hyperbola C with equation x²-(y²/b²)=1, prove that under certain conditions, 
    its asymptotes have the equation x±√3y=0 -/
theorem hyperbola_asymptotes 
  (b : ℝ) 
  (h1 : b > 0) 
  (C : Set (ℝ × ℝ)) 
  (h2 : C = {(x, y) | x^2 - y^2 / b^2 = 1}) 
  (F : ℝ × ℝ) 
  (h3 : F.1 > 0 ∧ F.2 = 0) -- F is on positive x-axis
  (h4 : ∀ (x y : ℝ), (x, y) ∈ C → (x - F.1)^2 + y^2 = (x + F.1)^2 + y^2) -- F is right focus
  (A B : ℝ × ℝ)
  (h5 : A ∈ C ∧ A.1 > 0 ∧ A.2 > 0) -- A is in first quadrant on hyperbola
  (h6 : B.1 > 0 ∧ B.2 > 0) -- B is in first quadrant
  (h7 : A.1 = B.1 ∧ A.1 = F.1) -- A, B, F are on same vertical line
  (h8 : (B.1, B.2) ∈ {(x, y) | y = b * x ∨ y = -b * x}) -- B is on asymptote
  (h9 : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (F.1 - A.1)^2 + (F.2 - A.2)^2) -- |AB| = |AF|
  : {(x, y) | x + Real.sqrt 3 * y = 0 ∨ x - Real.sqrt 3 * y = 0} = 
    {(x, y) | y = b * x ∨ y = -b * x} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l47_4798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_buses_stop_12_minutes_l47_4777

/-- Represents a bus with its speeds excluding and including stoppages -/
structure Bus where
  speed_excluding : ℚ
  speed_including : ℚ

/-- Calculates the stopped time per hour for a given bus in minutes -/
noncomputable def stopped_time (b : Bus) : ℚ :=
  60 * (b.speed_excluding - b.speed_including) / b.speed_excluding

/-- The three buses from the problem -/
def bus_A : Bus := { speed_excluding := 60, speed_including := 48 }
def bus_B : Bus := { speed_excluding := 75, speed_including := 60 }
def bus_C : Bus := { speed_excluding := 90, speed_including := 72 }

theorem all_buses_stop_12_minutes :
  stopped_time bus_A = 12 ∧ stopped_time bus_B = 12 ∧ stopped_time bus_C = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_buses_stop_12_minutes_l47_4777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_of_sums_l47_4768

/-- For an arithmetic sequence with first term a and common difference d,
    Sₙ represents the sum of the first n terms, and
    Tₙ represents the sum of the first n values of Sₖ. -/
noncomputable def T (a d m : ℝ) : ℝ := m * (m + 1) * (3 * a + (m - 1) * d) / 6

/-- The theorem states that for an arithmetic sequence with first term a and common difference d,
    T_m = m(m+1)(3a + (m-1)d) / 6, where m = a + 99d -/
theorem arithmetic_sequence_sum_of_sums (a d : ℝ) :
  T a d (a + 99 * d) = (a + 99 * d) * (a + 99 * d + 1) * (3 * a + (a + 99 * d - 1) * d) / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_of_sums_l47_4768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_correct_l47_4751

/-- Calculates the final price of a dress for a staff member after discounts and tax -/
noncomputable def final_price (d P Q R x : ℝ) : ℝ :=
  0.85 * d * (1 - P / 100) * (1 - Q / 100) * (1 - R / 100) * (1 + x / 100)

/-- Theorem stating that the calculated final price is correct -/
theorem final_price_correct (d P Q R x : ℝ) :
  let initial_discount := d * 0.85
  let staff_discount1 := initial_discount * (1 - P / 100)
  let staff_discount2 := staff_discount1 * (1 - Q / 100)
  let staff_discount3 := staff_discount2 * (1 - R / 100)
  let with_tax := staff_discount3 * (1 + x / 100)
  with_tax = final_price d P Q R x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_correct_l47_4751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l47_4720

theorem order_of_values : 
  let a : ℚ := -4
  let b : ℚ := 1/4
  let c : ℚ := 4
  let d : ℚ := 1
  a < b ∧ b < d ∧ d < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l47_4720
