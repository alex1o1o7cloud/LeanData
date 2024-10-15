import Mathlib

namespace NUMINAMATH_CALUDE_point_in_region_l1427_142714

def is_in_region (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 25 ∧ 
  (x + y ≠ 0 → -1 ≤ x / (x + y) ∧ x / (x + y) ≤ 1) ∧
  (x ≠ 0 → 0 ≤ x / y ∧ x / y ≤ 1)

theorem point_in_region (x y : ℝ) :
  is_in_region x y ↔ 
    (x^2 + y^2 ≤ 25 ∧ 
     ((x ≠ 0 ∧ y ≠ 0 ∧ 0 ≤ x / y ∧ x / y ≤ 1) ∨ 
      (x = 0 ∧ -5 ≤ y ∧ y ≤ 5) ∨ 
      (y = 0 ∧ 0 ≤ x ∧ x ≤ 5))) :=
  sorry

end NUMINAMATH_CALUDE_point_in_region_l1427_142714


namespace NUMINAMATH_CALUDE_value_of_y_l1427_142770

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l1427_142770


namespace NUMINAMATH_CALUDE_max_xy_value_l1427_142772

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1427_142772


namespace NUMINAMATH_CALUDE_correct_num_episodes_l1427_142706

/-- The number of episodes in a TV mini series -/
def num_episodes : ℕ := 6

/-- The length of each episode in minutes -/
def episode_length : ℕ := 50

/-- The total watching time in hours -/
def total_watching_time : ℕ := 5

/-- Theorem stating that the number of episodes is correct -/
theorem correct_num_episodes :
  num_episodes * episode_length = total_watching_time * 60 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_episodes_l1427_142706


namespace NUMINAMATH_CALUDE_length_breadth_difference_l1427_142731

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  perimeter : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ

/-- Theorem stating the difference between length and breadth of the plot -/
theorem length_breadth_difference (plot : RectangularPlot)
  (h1 : plot.length = 61)
  (h2 : plot.perimeter * plot.fencing_rate = plot.fencing_cost)
  (h3 : plot.fencing_rate = 26.50)
  (h4 : plot.fencing_cost = 5300)
  (h5 : plot.perimeter = 2 * (plot.length + plot.breadth))
  (h6 : plot.length > plot.breadth) :
  plot.length - plot.breadth = 22 := by
  sorry

end NUMINAMATH_CALUDE_length_breadth_difference_l1427_142731


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1427_142799

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1427_142799


namespace NUMINAMATH_CALUDE_circular_well_diameter_l1427_142703

/-- Proves that a circular well with given depth and volume has a specific diameter -/
theorem circular_well_diameter 
  (depth : ℝ) 
  (volume : ℝ) 
  (h_depth : depth = 8) 
  (h_volume : volume = 25.132741228718345) : 
  2 * (volume / (Real.pi * depth))^(1/2 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_circular_well_diameter_l1427_142703


namespace NUMINAMATH_CALUDE_f_composition_eq_inverse_e_l1427_142717

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_eq_inverse_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_eq_inverse_e_l1427_142717


namespace NUMINAMATH_CALUDE_not_all_distinct_l1427_142751

/-- A sequence of non-negative rational numbers satisfying a(m) + a(n) = a(mn) -/
def NonNegativeSequence (a : ℕ → ℚ) : Prop :=
  (∀ n, a n ≥ 0) ∧ (∀ m n, a m + a n = a (m * n))

/-- The theorem stating that not all elements of the sequence can be distinct -/
theorem not_all_distinct (a : ℕ → ℚ) (h : NonNegativeSequence a) :
  ∃ i j, i ≠ j ∧ a i = a j :=
sorry

end NUMINAMATH_CALUDE_not_all_distinct_l1427_142751


namespace NUMINAMATH_CALUDE_S_remainder_mod_1000_l1427_142792

/-- The sum of all three-digit positive integers from 500 to 999 with all digits distinct -/
def S : ℕ := sorry

/-- Theorem stating that the remainder of S divided by 1000 is 720 -/
theorem S_remainder_mod_1000 : S % 1000 = 720 := by sorry

end NUMINAMATH_CALUDE_S_remainder_mod_1000_l1427_142792


namespace NUMINAMATH_CALUDE_number_of_arrangements_l1427_142779

/-- The number of volunteers --/
def n : ℕ := 6

/-- The number of exhibition areas --/
def m : ℕ := 4

/-- The number of exhibition areas that should have one person --/
def k : ℕ := 2

/-- The number of exhibition areas that should have two people --/
def l : ℕ := 2

/-- The constraint that two specific volunteers cannot be in the same group --/
def constraint : Prop := True

/-- The function that calculates the number of arrangement plans --/
def arrangement_plans (n m k l : ℕ) (constraint : Prop) : ℕ := sorry

/-- Theorem stating that the number of arrangement plans is 156 --/
theorem number_of_arrangements :
  arrangement_plans n m k l constraint = 156 := by sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l1427_142779


namespace NUMINAMATH_CALUDE_fraction_value_l1427_142709

theorem fraction_value (m n : ℝ) (h : |m - 1/4| + (n + 3)^2 = 0) : n / m = -12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1427_142709


namespace NUMINAMATH_CALUDE_complement_union_A_B_l1427_142724

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l1427_142724


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l1427_142758

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x + 1)) ↔ x ≠ -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l1427_142758


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1427_142762

theorem roots_of_quadratic_equation (α β : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = α ∨ x = β) →
  7 * α^3 + 10 * β^4 = 697 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1427_142762


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1427_142795

def p (x₁ x₂ : ℝ) : Prop := x₁^2 + 5*x₁ - 6 = 0 ∧ x₂^2 + 5*x₂ - 6 = 0

def q (x₁ x₂ : ℝ) : Prop := x₁ + x₂ = -5

theorem p_sufficient_not_necessary :
  (∀ x₁ x₂, p x₁ x₂ → q x₁ x₂) ∧ (∃ y₁ y₂, q y₁ y₂ ∧ ¬p y₁ y₂) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1427_142795


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l1427_142757

theorem quadratic_root_in_unit_interval 
  (a b c m : ℝ) 
  (ha : a > 0) 
  (hm : m > 0) 
  (h_sum : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l1427_142757


namespace NUMINAMATH_CALUDE_equation_solution_l1427_142715

theorem equation_solution : ∃ x : ℝ, (x / 2 - 1 = 3) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1427_142715


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l1427_142746

-- Define the conditions
def p (x : ℝ) := x^2 - 1 > 0
def q (x : ℝ) := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l1427_142746


namespace NUMINAMATH_CALUDE_triangle_inequality_inside_l1427_142773

/-- A point is inside a triangle if it's in the interior of the triangle --/
def PointInsideTriangle (A B C M : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
  M = α • A + β • B + γ • C

/-- The theorem statement --/
theorem triangle_inequality_inside (A B C M : EuclideanSpace ℝ (Fin 2)) 
  (h : PointInsideTriangle A B C M) : 
  dist M B + dist M C < dist A B + dist A C := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_inside_l1427_142773


namespace NUMINAMATH_CALUDE_jones_earnings_proof_l1427_142794

/-- Dr. Jones' monthly earnings in dollars -/
def monthly_earnings : ℝ := 6000

/-- Dr. Jones' monthly expenses and savings -/
theorem jones_earnings_proof :
  monthly_earnings - (
    640 +  -- House rental
    380 +  -- Food expense
    (monthly_earnings / 4) +  -- Electric and water bill
    (monthly_earnings / 5)  -- Insurances
  ) = 2280  -- Remaining money after expenses
  := by sorry

end NUMINAMATH_CALUDE_jones_earnings_proof_l1427_142794


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1427_142719

/-- The volume of a sphere inscribed in a cube with edge length 10 inches -/
theorem inscribed_sphere_volume :
  let edge_length : ℝ := 10
  let radius : ℝ := edge_length / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  sphere_volume = (500 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1427_142719


namespace NUMINAMATH_CALUDE_cube_surface_area_l1427_142759

def edge_length : ℝ := 7

def surface_area_of_cube (edge : ℝ) : ℝ := 6 * edge^2

theorem cube_surface_area : 
  surface_area_of_cube edge_length = 294 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1427_142759


namespace NUMINAMATH_CALUDE_inequality_proof_l1427_142749

theorem inequality_proof (x y z u : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hz1 : z ≠ 1) (hu1 : u ≠ 1)
  (h_all_gt : (x > 1 ∧ y > 1 ∧ z > 1 ∧ u > 1) ∨ (x < 1 ∧ y < 1 ∧ z < 1 ∧ u < 1)) :
  (Real.log x ^ 3 / Real.log y) / (x + y + z) +
  (Real.log y ^ 3 / Real.log z) / (y + z + u) +
  (Real.log z ^ 3 / Real.log u) / (z + u + x) +
  (Real.log u ^ 3 / Real.log x) / (u + x + y) ≥
  16 / (x + y + z + u) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1427_142749


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_l1427_142730

theorem fraction_sum_equals_two (a b : ℝ) (ha : a ≠ 0) : 
  (2*b + a) / a + (a - 2*b) / a = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_l1427_142730


namespace NUMINAMATH_CALUDE_stating_transport_equation_transport_scenario_proof_l1427_142710

/-- Represents the scenario of two transports moving towards each other. -/
structure TransportScenario where
  x : ℝ  -- Speed of transport A in mph
  T : ℝ  -- Time in hours after transport A's departure when they are 348 miles apart

/-- 
  Theorem stating the relationship between the speeds and time
  for the given transport scenario.
-/
theorem transport_equation (scenario : TransportScenario) :
  let x := scenario.x
  let T := scenario.T
  2 * x * T + 18 * T - x - 18 = 258 := by
  sorry

/-- 
  Proves that the equation holds for the given transport scenario
  where two transports start 90 miles apart, with one traveling at speed x mph
  and the other at (x + 18) mph, starting 1 hour later, and end up 348 miles apart.
-/
theorem transport_scenario_proof (x : ℝ) (T : ℝ) :
  let scenario : TransportScenario := { x := x, T := T }
  (2 * x * T + 18 * T - x - 18 = 258) ↔
  (x * T + (x + 18) * (T - 1) = 348 - 90) := by
  sorry

end NUMINAMATH_CALUDE_stating_transport_equation_transport_scenario_proof_l1427_142710


namespace NUMINAMATH_CALUDE_point_equidistant_from_axes_l1427_142745

theorem point_equidistant_from_axes (a : ℝ) : 
  (∀ (x y : ℝ), x = a - 2 ∧ y = 6 - 2*a → |x| = |y|) → 
  (a = 8/3 ∨ a = 4) :=
sorry

end NUMINAMATH_CALUDE_point_equidistant_from_axes_l1427_142745


namespace NUMINAMATH_CALUDE_max_intersections_10_points_l1427_142707

/-- The maximum number of intersection points of perpendicular bisectors for n points -/
def max_intersections (n : ℕ) : ℕ :=
  Nat.choose n 3 + 3 * Nat.choose n 4

/-- Theorem stating the maximum number of intersection points for 10 points -/
theorem max_intersections_10_points :
  max_intersections 10 = 750 := by sorry

end NUMINAMATH_CALUDE_max_intersections_10_points_l1427_142707


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1427_142793

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 0 ∧ 
    log_base 4 (x - 1) + log_base (Real.sqrt 4) (x^2 - 1) + log_base (1/4) (x - 1) = 2 ∧
    x = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1427_142793


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1427_142785

/-- Jessie's weight loss journey -/
theorem jessie_weight_loss (initial_weight lost_weight : ℕ) 
  (h1 : initial_weight = 192)
  (h2 : lost_weight = 126) :
  initial_weight - lost_weight = 66 :=
by sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l1427_142785


namespace NUMINAMATH_CALUDE_savings_goal_theorem_l1427_142737

/-- Calculates the amount to save per paycheck given the total savings goal,
    number of months, and number of paychecks per month. -/
def amount_per_paycheck (total_savings : ℚ) (num_months : ℕ) (paychecks_per_month : ℕ) : ℚ :=
  total_savings / (num_months * paychecks_per_month)

/-- Proves that saving $100 per paycheck for 15 months with 2 paychecks per month
    results in a total savings of $3000. -/
theorem savings_goal_theorem :
  amount_per_paycheck 3000 15 2 = 100 := by
  sorry

#eval amount_per_paycheck 3000 15 2

end NUMINAMATH_CALUDE_savings_goal_theorem_l1427_142737


namespace NUMINAMATH_CALUDE_lucy_fraction_of_edna_l1427_142732

-- Define the field length
def field_length : ℚ := 24

-- Define Mary's distance as a fraction of the field length
def mary_distance : ℚ := 3/8 * field_length

-- Define Edna's distance as a fraction of Mary's distance
def edna_distance : ℚ := 2/3 * mary_distance

-- Define Lucy's distance as mary_distance - 4
def lucy_distance : ℚ := mary_distance - 4

-- Theorem to prove
theorem lucy_fraction_of_edna : lucy_distance / edna_distance = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fraction_of_edna_l1427_142732


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1427_142729

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > b + 1 → a > b) ∧ ¬(a > b → a > b + 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1427_142729


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1427_142796

/-- Definition of the diamond operation -/
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - a

/-- Theorem stating that if 4 ◇ y = 44, then y = 48/7 -/
theorem diamond_equation_solution :
  diamond 4 y = 44 → y = 48 / 7 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1427_142796


namespace NUMINAMATH_CALUDE_B_subset_A_l1427_142742

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 2 ∧ p.1 + 2 * p.2 = 6}

-- Theorem statement
theorem B_subset_A : B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_B_subset_A_l1427_142742


namespace NUMINAMATH_CALUDE_sequence_inequality_l1427_142760

theorem sequence_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let z := Complex.mk a b
  let seq := fun (n : ℕ+) => z ^ n.val
  let a_n := fun (n : ℕ+) => (seq n).re
  let b_n := fun (n : ℕ+) => (seq n).im
  ∀ n : ℕ+, (Complex.abs (a_n (n + 1)) + Complex.abs (b_n (n + 1))) / (Complex.abs (a_n n) + Complex.abs (b_n n)) ≥ (a^2 + b^2) / (a + b) :=
by sorry


end NUMINAMATH_CALUDE_sequence_inequality_l1427_142760


namespace NUMINAMATH_CALUDE_side_face_area_l1427_142797

/-- A rectangular box with specific proportions and volume -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * width = 0.5 * (length * height)
  top_1_5_side : length * height = 1.5 * (width * height)
  volume : length * width * height = 3000

/-- The area of the side face of the box is 200 -/
theorem side_face_area (b : Box) : b.width * b.height = 200 := by
  sorry

end NUMINAMATH_CALUDE_side_face_area_l1427_142797


namespace NUMINAMATH_CALUDE_altered_difference_larger_l1427_142769

theorem altered_difference_larger (a b : ℤ) (h1 : a > b) (h2 : b > 0) :
  (1.03 : ℝ) * (a : ℝ) - 0.98 * (b : ℝ) > (a : ℝ) - (b : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_altered_difference_larger_l1427_142769


namespace NUMINAMATH_CALUDE_q_min_at_two_l1427_142711

/-- The function q in terms of x -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 18

/-- The theorem stating that q is minimized to 0 when x = 2 -/
theorem q_min_at_two : 
  (∀ x : ℝ, q x ≥ q 2) ∧ q 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_q_min_at_two_l1427_142711


namespace NUMINAMATH_CALUDE_max_airlines_with_both_amenities_is_zero_l1427_142787

/-- Represents a type of plane -/
inductive PlaneType
| A
| B

/-- Represents whether a plane has both amenities -/
def has_both_amenities : PlaneType → Bool
| PlaneType.A => true
| PlaneType.B => false

/-- Represents a fleet composition -/
structure FleetComposition :=
  (type_a_percent : ℚ)
  (type_b_percent : ℚ)
  (sum_to_one : type_a_percent + type_b_percent = 1)
  (valid_range : 0.1 ≤ type_a_percent ∧ type_a_percent ≤ 0.9)

/-- Minimum number of planes in a fleet -/
def min_fleet_size : ℕ := 5

/-- Theorem: The maximum percentage of airlines offering both amenities on all planes is 0% -/
theorem max_airlines_with_both_amenities_is_zero :
  ∀ (fc : FleetComposition),
    ¬(∀ (plane : PlaneType), has_both_amenities plane = true) :=
by sorry

end NUMINAMATH_CALUDE_max_airlines_with_both_amenities_is_zero_l1427_142787


namespace NUMINAMATH_CALUDE_initial_geese_count_l1427_142701

theorem initial_geese_count (G : ℕ) : 
  (G / 2 + 4 = 12) → G = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_geese_count_l1427_142701


namespace NUMINAMATH_CALUDE_investment_result_l1427_142716

/-- Calculates the final amount after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the given investment scenario results in approximately $3045.28 --/
theorem investment_result :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℕ := 21
  let result := compound_interest principal rate time
  ∃ ε > 0, |result - 3045.28| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_result_l1427_142716


namespace NUMINAMATH_CALUDE_nine_distinct_values_of_z_l1427_142743

/-- Given two integers x and y between 100 and 999 inclusive, where y is formed by swapping
    the hundreds and tens digits of x (units digit remains the same), prove that the absolute
    difference z = |x - y| can have exactly 9 distinct values. -/
theorem nine_distinct_values_of_z (x y : ℤ) (z : ℕ) :
  (100 ≤ x ∧ x ≤ 999) →
  (100 ≤ y ∧ y ≤ 999) →
  (∃ a b c : ℕ, x = 100 * a + 10 * b + c ∧ y = 10 * a + 100 * b + c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) →
  z = |x - y| →
  ∃ (S : Finset ℕ), S.card = 9 ∧ z ∈ S ∧ ∀ w ∈ S, ∃ k : ℕ, w = 90 * k ∧ k ≤ 8 :=
sorry

#check nine_distinct_values_of_z

end NUMINAMATH_CALUDE_nine_distinct_values_of_z_l1427_142743


namespace NUMINAMATH_CALUDE_fraction_meaningful_range_l1427_142784

theorem fraction_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (x - 2)) ↔ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_range_l1427_142784


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1427_142761

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ

/-- Calculate the average runs per inning -/
def average (b : Batsman) : ℚ :=
  b.totalRuns / b.innings

/-- Calculate the increase in average -/
def averageIncrease (before after : ℚ) : ℚ :=
  after - before

theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 11) 
  (h2 : b.lastInningRuns = 90) 
  (h3 : average b = 40) :
  averageIncrease 
    (average { innings := b.innings - 1, totalRuns := b.totalRuns - b.lastInningRuns, lastInningRuns := 0 }) 
    (average b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1427_142761


namespace NUMINAMATH_CALUDE_triangle_circumradius_l1427_142713

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l1427_142713


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l1427_142791

theorem walnut_trees_planted (current : ℕ) (final : ℕ) (planted : ℕ) : 
  current = 4 → final = 10 → planted = final - current → planted = 6 := by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l1427_142791


namespace NUMINAMATH_CALUDE_max_value_of_f_l1427_142739

-- Define the function to be maximized
def f (a b c : ℝ) : ℝ := a * b + b * c + 2 * a * c

-- State the theorem
theorem max_value_of_f :
  ∀ a b c : ℝ,
  a ≥ 0 → b ≥ 0 → c ≥ 0 →
  a + b + c = 1 →
  f a b c ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1427_142739


namespace NUMINAMATH_CALUDE_water_remaining_l1427_142798

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → 
  used = 11/8 → 
  remaining = initial - used → 
  remaining = 13/8 :=
by
  sorry

#eval (13/8 : ℚ) -- To show that 13/8 is equivalent to 1 5/8

end NUMINAMATH_CALUDE_water_remaining_l1427_142798


namespace NUMINAMATH_CALUDE_b_upper_bound_l1427_142734

theorem b_upper_bound (b : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2), Real.sqrt (1 - x^2) > x + b) → 
  b < 0 := by
sorry

end NUMINAMATH_CALUDE_b_upper_bound_l1427_142734


namespace NUMINAMATH_CALUDE_cubic_factorization_l1427_142702

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1427_142702


namespace NUMINAMATH_CALUDE_product_evaluation_l1427_142728

theorem product_evaluation (a : ℤ) (h : a = 1) : (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1427_142728


namespace NUMINAMATH_CALUDE_point_quadrant_l1427_142789

/-- Given that point A(a, -b) is in the first quadrant, prove that point B(a, b) is in the fourth quadrant -/
theorem point_quadrant (a b : ℝ) : 
  (a > 0 ∧ -b > 0) → (a > 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_quadrant_l1427_142789


namespace NUMINAMATH_CALUDE_sum_of_roots_l1427_142781

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x = 1) 
  (hy : y^3 - 3*y^2 + 5*y = 5) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1427_142781


namespace NUMINAMATH_CALUDE_tan_is_odd_l1427_142726

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define the property of being an odd function
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem tan_is_odd : IsOdd Real.tan := by
  sorry

end NUMINAMATH_CALUDE_tan_is_odd_l1427_142726


namespace NUMINAMATH_CALUDE_perfect_33rd_power_l1427_142753

theorem perfect_33rd_power (x y : ℕ+) (h : ∃ k : ℕ+, (x * y^10 : ℕ) = k^33) :
  ∃ m : ℕ+, (x^10 * y : ℕ) = m^33 := by
  sorry

end NUMINAMATH_CALUDE_perfect_33rd_power_l1427_142753


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l1427_142748

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ (5 * x - 4 : ℝ) < (3 - 2 * x : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l1427_142748


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1427_142704

theorem solution_set_inequality (x : ℝ) : 
  Set.Icc 1 2 = {x | (x - 1) * (x - 2) ≤ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1427_142704


namespace NUMINAMATH_CALUDE_face_value_from_discounts_l1427_142741

/-- Face value calculation given banker's discount and true discount -/
theorem face_value_from_discounts
  (BD : ℚ) -- Banker's discount
  (TD : ℚ) -- True discount
  (h1 : BD = 42)
  (h2 : TD = 36)
  : BD = TD + (BD - TD) :=
by
  sorry

#check face_value_from_discounts

end NUMINAMATH_CALUDE_face_value_from_discounts_l1427_142741


namespace NUMINAMATH_CALUDE_airplane_seats_theorem_l1427_142708

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 300

/-- Represents the number of First Class seats -/
def first_class_seats : ℕ := 30

/-- Represents the percentage of Business Class seats -/
def business_class_percentage : ℚ := 20 / 100

/-- Represents the percentage of Economy Class seats -/
def economy_class_percentage : ℚ := 70 / 100

/-- Theorem stating that the total number of seats is 300 -/
theorem airplane_seats_theorem :
  (first_class_seats : ℚ) +
  (business_class_percentage * total_seats) +
  (economy_class_percentage * total_seats) =
  total_seats :=
sorry

end NUMINAMATH_CALUDE_airplane_seats_theorem_l1427_142708


namespace NUMINAMATH_CALUDE_set_union_problem_l1427_142733

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l1427_142733


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1427_142778

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with 60° inclination passing through the focus
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define a point in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the intersection point A
def point_A (x y : ℝ) : Prop :=
  parabola x y ∧ line x y ∧ first_quadrant x y

-- The main theorem
theorem parabola_line_intersection :
  ∀ x y : ℝ, point_A x y → |((x, y) : ℝ × ℝ) - focus| = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1427_142778


namespace NUMINAMATH_CALUDE_airway_graph_diameter_at_most_two_l1427_142776

/-- A simple graph with 20 vertices and 172 edges -/
structure AirwayGraph where
  V : Finset (Fin 20)
  E : Finset (Fin 20 × Fin 20)
  edge_count : E.card = 172
  simple : ∀ (e : Fin 20 × Fin 20), e ∈ E → e.1 ≠ e.2
  undirected : ∀ (e : Fin 20 × Fin 20), e ∈ E → (e.2, e.1) ∈ E
  at_most_one : ∀ (u v : Fin 20), u ≠ v → ({(u, v), (v, u)} ∩ E).card ≤ 1

/-- The diameter of an AirwayGraph is at most 2 -/
theorem airway_graph_diameter_at_most_two (G : AirwayGraph) :
  ∀ (u v : Fin 20), u ≠ v → ∃ (w : Fin 20), (u = w ∨ (u, w) ∈ G.E) ∧ (w = v ∨ (w, v) ∈ G.E) :=
sorry

end NUMINAMATH_CALUDE_airway_graph_diameter_at_most_two_l1427_142776


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l1427_142765

theorem floor_expression_equals_eight :
  ⌊(2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l1427_142765


namespace NUMINAMATH_CALUDE_flamingo_percentage_among_non_parrots_l1427_142774

/-- Given the distribution of birds in a wildlife reserve, this theorem proves
    that flamingos constitute 50% of the non-parrot birds. -/
theorem flamingo_percentage_among_non_parrots :
  let total_percentage : ℝ := 100
  let flamingo_percentage : ℝ := 40
  let parrot_percentage : ℝ := 20
  let eagle_percentage : ℝ := 15
  let owl_percentage : ℝ := total_percentage - flamingo_percentage - parrot_percentage - eagle_percentage
  let non_parrot_percentage : ℝ := total_percentage - parrot_percentage
  (flamingo_percentage / non_parrot_percentage) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_flamingo_percentage_among_non_parrots_l1427_142774


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l1427_142744

theorem largest_power_of_two_dividing_difference : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (17^4 - 13^4) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (17^4 - 13^4) → m ≤ k :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l1427_142744


namespace NUMINAMATH_CALUDE_nancy_coffee_expenditure_l1427_142768

/-- Represents Nancy's coffee consumption and expenditure over a period of time. -/
structure CoffeeConsumption where
  morning_price : ℝ
  afternoon_price : ℝ
  days : ℕ

/-- Calculates the total expenditure on coffee given Nancy's consumption pattern. -/
def total_expenditure (c : CoffeeConsumption) : ℝ :=
  c.days * (c.morning_price + c.afternoon_price)

/-- Theorem stating that Nancy's total expenditure on coffee over 20 days is $110.00. -/
theorem nancy_coffee_expenditure :
  let c : CoffeeConsumption := {
    morning_price := 3.00,
    afternoon_price := 2.50,
    days := 20
  }
  total_expenditure c = 110.00 := by
  sorry

end NUMINAMATH_CALUDE_nancy_coffee_expenditure_l1427_142768


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1427_142712

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  (-1 < k ∧ k < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1427_142712


namespace NUMINAMATH_CALUDE_water_depth_relationship_l1427_142718

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : Real
  baseDiameter : Real
  horizontalWaterDepth : Real

/-- Calculates the water depth when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) : Real :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_relationship (tank : WaterTank) 
  (h : tank.height = 20) 
  (d : tank.baseDiameter = 5) 
  (w : tank.horizontalWaterDepth = 2) : 
  ∃ ε > 0, abs (verticalWaterDepth tank - 0.9) < ε :=
sorry

end NUMINAMATH_CALUDE_water_depth_relationship_l1427_142718


namespace NUMINAMATH_CALUDE_cartesian_to_polar_equivalence_curve_transformation_l1427_142766

-- Part I
theorem cartesian_to_polar_equivalence :
  let x : ℝ := -Real.sqrt 2
  let y : ℝ := Real.sqrt 2
  let r : ℝ := 2
  let θ : ℝ := 3 * Real.pi / 4
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by sorry

-- Part II
theorem curve_transformation (x y x' y' : ℝ) :
  x' = 5 * x →
  y' = 3 * y →
  (2 * x' ^ 2 + 8 * y' ^ 2 = 1) →
  (25 * x ^ 2 + 36 * y ^ 2 = 1) := by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_equivalence_curve_transformation_l1427_142766


namespace NUMINAMATH_CALUDE_parabola_theorem_l1427_142720

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define line l passing through K
def l (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the condition for A and B being on C and l
def intersectionPoints (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ l m A.1 A.2 ∧ l m B.1 B.2

-- Define the symmetry condition for A and D
def symmetricPoints (A D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = -D.2

-- Define the dot product condition
def dotProductCondition (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 8/9

-- State the theorem
theorem parabola_theorem (m : ℝ) (A B D : ℝ × ℝ) :
  intersectionPoints m A B →
  symmetricPoints A D →
  dotProductCondition A B →
  (∃ (t : ℝ), F.1 = D.1 + t * (B.1 - D.1) ∧ F.2 = D.2 + t * (B.2 - D.2)) ∧
  (∃ (M : ℝ × ℝ), M.1 = 1/9 ∧ M.2 = 0 ∧
    ∀ (x y : ℝ), (x - M.1)^2 + (y - M.2)^2 = 4/9 →
      (x - K.1)^2 + (y - K.2)^2 ≥ 4/9 ∧
      (x - B.1)^2 + (y - B.2)^2 ≥ 4/9 ∧
      (x - D.1)^2 + (y - D.2)^2 ≥ 4/9) :=
by sorry

end NUMINAMATH_CALUDE_parabola_theorem_l1427_142720


namespace NUMINAMATH_CALUDE_show_attendance_ratio_l1427_142782

/-- The ratio of attendees at the second showing to the debut show is 4 -/
theorem show_attendance_ratio : 
  ∀ (debut_attendance second_attendance ticket_price total_revenue : ℕ),
    debut_attendance = 200 →
    ticket_price = 25 →
    total_revenue = 20000 →
    second_attendance = total_revenue / ticket_price →
    second_attendance / debut_attendance = 4 := by
  sorry

end NUMINAMATH_CALUDE_show_attendance_ratio_l1427_142782


namespace NUMINAMATH_CALUDE_reasoning_forms_mapping_l1427_142790

/-- Represents the different forms of reasoning -/
inductive ReasoningForm
  | Inductive
  | Deductive
  | Analogical

/-- Represents the different reasoning descriptions -/
inductive ReasoningDescription
  | SpecificToSpecific
  | PartToWholeOrIndividualToGeneral
  | GeneralToSpecific

/-- Maps a reasoning description to its corresponding reasoning form -/
def descriptionToForm (d : ReasoningDescription) : ReasoningForm :=
  match d with
  | ReasoningDescription.SpecificToSpecific => ReasoningForm.Analogical
  | ReasoningDescription.PartToWholeOrIndividualToGeneral => ReasoningForm.Inductive
  | ReasoningDescription.GeneralToSpecific => ReasoningForm.Deductive

theorem reasoning_forms_mapping :
  (descriptionToForm ReasoningDescription.SpecificToSpecific = ReasoningForm.Analogical) ∧
  (descriptionToForm ReasoningDescription.PartToWholeOrIndividualToGeneral = ReasoningForm.Inductive) ∧
  (descriptionToForm ReasoningDescription.GeneralToSpecific = ReasoningForm.Deductive) :=
sorry

end NUMINAMATH_CALUDE_reasoning_forms_mapping_l1427_142790


namespace NUMINAMATH_CALUDE_susans_remaining_money_is_830_02_l1427_142740

/-- Calculates Susan's remaining money after expenses --/
def susans_remaining_money (swimming_earnings babysitting_earnings online_earnings_euro : ℚ)
  (exchange_rate tax_rate clothes_percent books_percent gifts_percent : ℚ) : ℚ :=
  let online_earnings_dollar := online_earnings_euro * exchange_rate
  let total_earnings := swimming_earnings + babysitting_earnings + online_earnings_dollar
  let tax_amount := online_earnings_dollar * tax_rate
  let after_tax := online_earnings_dollar - tax_amount
  let clothes_spend := total_earnings * clothes_percent
  let after_clothes := total_earnings - clothes_spend
  let books_spend := after_clothes * books_percent
  let after_books := after_clothes - books_spend
  let gifts_spend := after_books * gifts_percent
  after_books - gifts_spend

/-- Theorem stating that Susan's remaining money is $830.02 --/
theorem susans_remaining_money_is_830_02 :
  susans_remaining_money 1000 500 300 1.20 0.02 0.30 0.25 0.15 = 830.02 := by
  sorry

end NUMINAMATH_CALUDE_susans_remaining_money_is_830_02_l1427_142740


namespace NUMINAMATH_CALUDE_triangle_count_l1427_142767

theorem triangle_count : ∃! (n : ℕ), n = 59 ∧
  (∀ (a b c : ℕ), (a < b ∧ b < c) →
    (b = 60) →
    (c - b = b - a) →
    (a + b + c = 180) →
    (0 < a ∧ a < b ∧ b < c) →
    (∃ (d : ℕ), a = 60 - d ∧ c = 60 + d ∧ 0 < d ∧ d < 60)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_count_l1427_142767


namespace NUMINAMATH_CALUDE_xiaoming_scoring_frequency_l1427_142725

/-- The frequency of scoring given total shots and goals -/
def scoring_frequency (total_shots : ℕ) (goals : ℕ) : ℚ :=
  (goals : ℚ) / (total_shots : ℚ)

/-- Theorem stating that given 80 total shots and 50 goals, the frequency of scoring is 0.625 -/
theorem xiaoming_scoring_frequency :
  scoring_frequency 80 50 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_scoring_frequency_l1427_142725


namespace NUMINAMATH_CALUDE_percentage_calculation_l1427_142747

theorem percentage_calculation (x : ℝ) (h : 70 = 0.56 * x) : 1.25 * x = 156.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1427_142747


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l1427_142763

theorem integer_pair_divisibility (m n : ℕ+) :
  (∃ k : ℤ, (m : ℤ) + n^2 = k * ((m : ℤ)^2 - n)) ∧
  (∃ l : ℤ, (m : ℤ)^2 + n = l * (n^2 - m)) →
  ((m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨
   (m = 2 ∧ n = 1) ∨ (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l1427_142763


namespace NUMINAMATH_CALUDE_nabla_computation_l1427_142735

def nabla (a b : ℕ) : ℕ := 3 + a^b

theorem nabla_computation : nabla (nabla 3 2) 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_nabla_computation_l1427_142735


namespace NUMINAMATH_CALUDE_cos_negative_thirteen_pi_fourths_l1427_142780

theorem cos_negative_thirteen_pi_fourths : 
  Real.cos (-13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_thirteen_pi_fourths_l1427_142780


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_solution_set_is_axes_l1427_142771

theorem equation_represents_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x * y = 0 :=
by sorry

-- The following definitions are to establish the connection
-- between the algebraic equation and its geometric interpretation

def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

theorem solution_set_is_axes :
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2} = x_axis ∪ y_axis :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_solution_set_is_axes_l1427_142771


namespace NUMINAMATH_CALUDE_min_floor_sum_l1427_142783

theorem min_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_floor_sum_l1427_142783


namespace NUMINAMATH_CALUDE_min_plates_for_five_colors_l1427_142727

/-- The minimum number of plates to pull out to guarantee a matching pair -/
def min_plates_for_match (num_colors : ℕ) : ℕ :=
  num_colors + 1

/-- Theorem stating that for 5 colors, the minimum number of plates to pull out for a match is 6 -/
theorem min_plates_for_five_colors :
  min_plates_for_match 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_plates_for_five_colors_l1427_142727


namespace NUMINAMATH_CALUDE_star_distance_l1427_142756

/-- The distance between a star and Earth given the speed of light and time taken for light to reach Earth -/
theorem star_distance (c : ℝ) (t : ℝ) (y : ℝ) (h1 : c = 3 * 10^5) (h2 : t = 10) (h3 : y = 3.1 * 10^7) :
  c * (t * y) = 9.3 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_star_distance_l1427_142756


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1427_142777

theorem linear_equation_solution :
  ∃ x : ℚ, 3 * x + 5 = 500 - (4 * x + 6 * x) ∧ x = 495 / 13 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1427_142777


namespace NUMINAMATH_CALUDE_star_composition_l1427_142705

-- Define the star operations
def star_right (y : ℝ) : ℝ := 10 - y
def star_left (y : ℝ) : ℝ := y - 10

-- State the theorem
theorem star_composition : star_left (star_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l1427_142705


namespace NUMINAMATH_CALUDE_max_books_robert_can_buy_l1427_142755

theorem max_books_robert_can_buy (book_cost : ℚ) (available_money : ℚ) : 
  book_cost = 875/100 → available_money = 250 → 
  (∃ n : ℕ, n * book_cost ≤ available_money ∧ 
    ∀ m : ℕ, m * book_cost ≤ available_money → m ≤ n) → 
  (∃ n : ℕ, n * book_cost ≤ available_money ∧ 
    ∀ m : ℕ, m * book_cost ≤ available_money → m ≤ n) ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_max_books_robert_can_buy_l1427_142755


namespace NUMINAMATH_CALUDE_fraction_change_l1427_142752

theorem fraction_change (x : ℚ) : 
  (1.2 * (5 : ℚ)) / (7 * (1 - x / 100)) = 20 / 21 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_change_l1427_142752


namespace NUMINAMATH_CALUDE_game_probability_limit_l1427_142775

/-- Represents the state of money distribution among players -/
inductive GameState
  | AllOne
  | TwoOneZero

/-- Transition probability matrix for the game -/
def transitionMatrix : Matrix GameState GameState ℝ := sorry

/-- The probability of all players having $1 after n bell rings -/
def prob_all_one (n : ℕ) : ℝ := sorry

/-- The limit of prob_all_one as n approaches infinity -/
def limit_prob_all_one : ℝ := sorry

theorem game_probability_limit :
  limit_prob_all_one = 1/4 := by sorry

end NUMINAMATH_CALUDE_game_probability_limit_l1427_142775


namespace NUMINAMATH_CALUDE_exponential_inverse_sum_l1427_142786

-- Define the exponential function f
def f (x : ℝ) : ℝ := sorry

-- Define the inverse function g
def g (x : ℝ) : ℝ := sorry

-- Theorem statement
theorem exponential_inverse_sum :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = a^x) →  -- f is an exponential function
  (f (1 + Real.sqrt 3) * f (1 - Real.sqrt 3) = 9) →  -- Given condition
  (∀ (x : ℝ), g (f x) = x ∧ f (g x) = x) →  -- g is the inverse of f
  (g (Real.sqrt 10 + 1) + g (Real.sqrt 10 - 1) = 2) :=
by sorry

end NUMINAMATH_CALUDE_exponential_inverse_sum_l1427_142786


namespace NUMINAMATH_CALUDE_jimin_yuna_problem_l1427_142721

/-- Given a line of students ordered by height, calculate the number of students between two specific positions. -/
def students_between (total : ℕ) (pos1 : ℕ) (pos2 : ℕ) : ℕ :=
  if pos1 > pos2 then pos1 - pos2 - 1 else pos2 - pos1 - 1

theorem jimin_yuna_problem :
  let total_students : ℕ := 32
  let jimin_position : ℕ := 27
  let yuna_position : ℕ := 11
  students_between total_students jimin_position yuna_position = 15 := by
  sorry

end NUMINAMATH_CALUDE_jimin_yuna_problem_l1427_142721


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1427_142738

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x, f x) ↔ ∀ x, ¬ f x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1427_142738


namespace NUMINAMATH_CALUDE_hotel_room_cost_l1427_142736

theorem hotel_room_cost (original_friends : ℕ) (additional_friends : ℕ) (cost_decrease : ℚ) :
  original_friends = 5 →
  additional_friends = 2 →
  cost_decrease = 15 →
  ∃ total_cost : ℚ,
    total_cost / original_friends - total_cost / (original_friends + additional_friends) = cost_decrease ∧
    total_cost = 262.5 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_cost_l1427_142736


namespace NUMINAMATH_CALUDE_linear_function_iteration_l1427_142722

/-- Given a linear function f and its iterations, prove that ab = 6 -/
theorem linear_function_iteration (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x + b
  let f₁ : ℝ → ℝ := f
  let f₂ : ℝ → ℝ := λ x ↦ f (f₁ x)
  let f₃ : ℝ → ℝ := λ x ↦ f (f₂ x)
  let f₄ : ℝ → ℝ := λ x ↦ f (f₃ x)
  let f₅ : ℝ → ℝ := λ x ↦ f (f₄ x)
  (∀ x, f₅ x = 32 * x + 93) → a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_linear_function_iteration_l1427_142722


namespace NUMINAMATH_CALUDE_barn_paint_area_l1427_142754

/-- Represents the dimensions of a rectangular prism -/
structure BarnDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in the barn -/
def total_paint_area (dim : BarnDimensions) : ℝ :=
  let end_wall_area := 2 * dim.width * dim.height
  let side_wall_area := 2 * dim.length * dim.height
  let ceiling_area := dim.length * dim.width
  let partition_area := 2 * dim.length * dim.height
  2 * (end_wall_area + side_wall_area) + ceiling_area + partition_area

/-- The barn dimensions -/
def barn : BarnDimensions :=
  { length := 15
  , width := 12
  , height := 6 }

theorem barn_paint_area :
  total_paint_area barn = 1008 := by
  sorry

end NUMINAMATH_CALUDE_barn_paint_area_l1427_142754


namespace NUMINAMATH_CALUDE_pipeline_construction_equation_l1427_142764

theorem pipeline_construction_equation 
  (total_length : ℝ) 
  (efficiency_increase : ℝ) 
  (days_ahead : ℝ) 
  (x : ℝ) 
  (h1 : total_length = 3000)
  (h2 : efficiency_increase = 0.2)
  (h3 : days_ahead = 10)
  (h4 : x > 0) :
  total_length / x - total_length / ((1 + efficiency_increase) * x) = days_ahead :=
sorry

end NUMINAMATH_CALUDE_pipeline_construction_equation_l1427_142764


namespace NUMINAMATH_CALUDE_calculation_proof_l1427_142788

theorem calculation_proof :
  ((-11 : ℤ) + 8 + (-4) = -7) ∧
  (-1^2023 - |1 - (1/3 : ℚ)| * (-3/2)^2 = -5/2) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1427_142788


namespace NUMINAMATH_CALUDE_unique_magnitude_complex_roots_l1427_142700

theorem unique_magnitude_complex_roots (z : ℂ) :
  (3 * z^2 - 18 * z + 55 = 0) →
  ∃! m : ℝ, ∃ z₁ z₂ : ℂ, (3 * z₁^2 - 18 * z₁ + 55 = 0) ∧
                         (3 * z₂^2 - 18 * z₂ + 55 = 0) ∧
                         (Complex.abs z₁ = m) ∧
                         (Complex.abs z₂ = m) :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_complex_roots_l1427_142700


namespace NUMINAMATH_CALUDE_point_outside_circle_l1427_142723

/-- Given a line ax + by = 1 and a circle x^2 + y^2 = 1 that intersect at two distinct points,
    prove that the point (a, b) lies outside the circle. -/
theorem point_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  (∃ x' y' : ℝ, x' ≠ y' ∧ a * x' + b * y' = 1 ∧ x'^2 + y'^2 = 1) →
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1427_142723


namespace NUMINAMATH_CALUDE_hollow_block_3_9_5_cubes_l1427_142750

/-- Calculates the number of unit cubes needed to construct the outer shell of a hollow rectangular block. -/
def hollow_block_cubes (length width depth : ℕ) : ℕ :=
  2 * (length * width) +  -- top and bottom
  2 * ((width * depth) - (width * 2)) +  -- longer sides
  2 * ((length * depth) - (length * 2) - 2)  -- shorter sides

/-- Theorem stating that a hollow rectangular block with dimensions 3 x 9 x 5 requires 122 unit cubes. -/
theorem hollow_block_3_9_5_cubes : 
  hollow_block_cubes 3 9 5 = 122 := by
  sorry

#eval hollow_block_cubes 3 9 5  -- Should output 122

end NUMINAMATH_CALUDE_hollow_block_3_9_5_cubes_l1427_142750
