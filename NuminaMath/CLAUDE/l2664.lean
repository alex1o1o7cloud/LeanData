import Mathlib

namespace NUMINAMATH_CALUDE_bad_carrots_count_l2664_266443

theorem bad_carrots_count (faye_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) :
  faye_carrots = 23 →
  mom_carrots = 5 →
  good_carrots = 12 →
  faye_carrots + mom_carrots - good_carrots = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l2664_266443


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2664_266401

/-- An ellipse with equation x² + y²/m = 1, foci on x-axis, and major axis twice the minor axis -/
structure Ellipse (m : ℝ) :=
  (equation : ∀ (x y : ℝ), x^2 + y^2/m = 1)
  (foci_on_x_axis : True)  -- This is a placeholder, as we can't directly represent this geometrically
  (major_twice_minor : True)  -- This is a placeholder for the condition

/-- The value of m for the given ellipse properties is 1/4 -/
theorem ellipse_m_value :
  ∀ (m : ℝ), Ellipse m → m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2664_266401


namespace NUMINAMATH_CALUDE_min_value_ratio_l2664_266441

theorem min_value_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ d₁ : ℝ, ∀ n, a (n + 1) = a n + d₁) →
  (∃ d₂ : ℝ, ∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d₂) →
  (∀ n, S n = (n * (a 1 + a n)) / 2) →
  (∀ n, (S (n + 10)) / (a n) ≥ 21) ∧
  (∃ n, (S (n + 10)) / (a n) = 21) :=
sorry

end NUMINAMATH_CALUDE_min_value_ratio_l2664_266441


namespace NUMINAMATH_CALUDE_peaches_picked_up_correct_l2664_266425

/-- Represents the fruit stand inventory --/
structure FruitStand where
  initialPeaches : ℕ
  initialOranges : ℕ
  peachesSold : ℕ
  orangesAdded : ℕ
  finalPeaches : ℕ
  finalOranges : ℕ

/-- Calculates the number of peaches picked up from the orchard --/
def peachesPickedUp (stand : FruitStand) : ℕ :=
  stand.finalPeaches - (stand.initialPeaches - stand.peachesSold)

/-- Theorem stating that the number of peaches picked up is correct --/
theorem peaches_picked_up_correct (stand : FruitStand) :
  peachesPickedUp stand = stand.finalPeaches - (stand.initialPeaches - stand.peachesSold) :=
by
  sorry

/-- Sally's fruit stand inventory --/
def sallysStand : FruitStand := {
  initialPeaches := 13
  initialOranges := 5
  peachesSold := 7
  orangesAdded := 22
  finalPeaches := 55
  finalOranges := 27
}

#eval peachesPickedUp sallysStand

end NUMINAMATH_CALUDE_peaches_picked_up_correct_l2664_266425


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_81_l2664_266414

theorem arithmetic_square_root_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_81_l2664_266414


namespace NUMINAMATH_CALUDE_apartment_rent_calculation_l2664_266426

/-- Proves that the rent for a shared apartment is $1100 given specific conditions -/
theorem apartment_rent_calculation (utilities groceries roommate_payment : ℕ) 
  (h1 : utilities = 114)
  (h2 : groceries = 300)
  (h3 : roommate_payment = 757) :
  ∃ (rent : ℕ), rent = 1100 ∧ (rent + utilities + groceries) / 2 = roommate_payment :=
by sorry

end NUMINAMATH_CALUDE_apartment_rent_calculation_l2664_266426


namespace NUMINAMATH_CALUDE_inequality_proof_l2664_266487

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2664_266487


namespace NUMINAMATH_CALUDE_xy_z_squared_plus_one_representation_l2664_266445

theorem xy_z_squared_plus_one_representation (x y z : ℕ+) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end NUMINAMATH_CALUDE_xy_z_squared_plus_one_representation_l2664_266445


namespace NUMINAMATH_CALUDE_continued_proportionate_reduction_eq_euclidean_gcd_l2664_266472

/-- The Method of Continued Proportionate Reduction as used in ancient Chinese mathematics -/
def continued_proportionate_reduction (a b : ℕ) : ℕ :=
  sorry

/-- The Euclidean algorithm for finding the greatest common divisor -/
def euclidean_gcd (a b : ℕ) : ℕ :=
  sorry

/-- Theorem stating the equivalence of the two methods -/
theorem continued_proportionate_reduction_eq_euclidean_gcd :
  ∀ a b : ℕ, continued_proportionate_reduction a b = euclidean_gcd a b :=
sorry

end NUMINAMATH_CALUDE_continued_proportionate_reduction_eq_euclidean_gcd_l2664_266472


namespace NUMINAMATH_CALUDE_equation_solution_l2664_266499

theorem equation_solution : 
  ∃! x : ℚ, (x - 15) / 3 = (3 * x + 11) / 8 :=
by
  use -153
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2664_266499


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2664_266477

/-- Theorem: Volume of a tetrahedron
  Given a tetrahedron with:
  - a, b: lengths of opposite edges
  - α: angle between these edges
  - c: distance between the lines containing these edges
  The volume V of the tetrahedron is given by V = (1/6) * a * b * c * sin(α)
-/
theorem tetrahedron_volume 
  (a b c : ℝ) 
  (α : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hα : 0 < α ∧ α < π) :
  ∃ V : ℝ, V = (1/6) * a * b * c * Real.sin α := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_volume_l2664_266477


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2664_266475

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2664_266475


namespace NUMINAMATH_CALUDE_quadratic_point_relation_l2664_266462

/-- The quadratic function f(x) = (x - 1)^2 -/
def f (x : ℝ) : ℝ := (x - 1)^2

theorem quadratic_point_relation (m : ℝ) :
  f m < f (m + 1) → m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_relation_l2664_266462


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_count_l2664_266492

/-- The number of values of a for which the line y = 2x + a passes through
    the vertex of the parabola y = x^2 + 2a^2 -/
theorem line_through_parabola_vertex_count : 
  ∃! (s : Finset ℝ), 
    (∀ a ∈ s, ∃ x y : ℝ, 
      y = 2 * x + a ∧ 
      y = x^2 + 2 * a^2 ∧ 
      ∀ x' : ℝ, x'^2 + 2 * a^2 ≤ x^2 + 2 * a^2) ∧ 
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_count_l2664_266492


namespace NUMINAMATH_CALUDE_point_on_line_trig_identity_l2664_266439

theorem point_on_line_trig_identity (θ : Real) :
  2 * Real.cos θ + Real.sin θ = 0 →
  Real.cos (2 * θ) + (1/2) * Real.sin (2 * θ) = -1 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_trig_identity_l2664_266439


namespace NUMINAMATH_CALUDE_distance_AD_between_41_and_42_l2664_266412

-- Define points A, B, C, and D in a 2D plane
variable (A B C D : ℝ × ℝ)

-- Define the conditions
variable (h1 : B.1 > A.1 ∧ B.2 = A.2) -- B is due east of A
variable (h2 : C.1 = B.1 ∧ C.2 > B.2) -- C is due north of B
variable (h3 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = 300) -- AC = 10√3
variable (h4 : Real.cos (Real.arctan ((C.2 - A.2) / (C.1 - A.1))) = 1/2) -- Angle BAC = 60°
variable (h5 : D.1 = C.1 ∧ D.2 = C.2 + 30) -- D is 30 meters due north of C

-- Theorem statement
theorem distance_AD_between_41_and_42 :
  41 < Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) ∧
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) < 42 :=
sorry

end NUMINAMATH_CALUDE_distance_AD_between_41_and_42_l2664_266412


namespace NUMINAMATH_CALUDE_bob_distance_when_meeting_l2664_266418

/-- Prove that Bob walked 8 miles when he met Yolanda given the following conditions:
  - The total distance between X and Y is 17 miles
  - Yolanda starts walking from X to Y
  - Bob starts walking from Y to X one hour after Yolanda
  - Yolanda's walking rate is 3 miles per hour
  - Bob's walking rate is 4 miles per hour
-/
theorem bob_distance_when_meeting (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) 
  (h1 : total_distance = 17)
  (h2 : yolanda_rate = 3)
  (h3 : bob_rate = 4) :
  ∃ t : ℝ, t > 0 ∧ yolanda_rate * (t + 1) + bob_rate * t = total_distance ∧ bob_rate * t = 8 := by
  sorry


end NUMINAMATH_CALUDE_bob_distance_when_meeting_l2664_266418


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_four_l2664_266489

def A : Set ℝ := {x | x ≥ 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x ≤ 0}

theorem intersection_implies_m_equals_four (m : ℝ) : A ∩ B m = {x | 1 ≤ x ∧ x ≤ 4} → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_four_l2664_266489


namespace NUMINAMATH_CALUDE_ac_over_bd_equals_15_l2664_266459

theorem ac_over_bd_equals_15 
  (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 5 * d) 
  (h4 : d ≠ 0) : 
  (a * c) / (b * d) = 15 := by
sorry

end NUMINAMATH_CALUDE_ac_over_bd_equals_15_l2664_266459


namespace NUMINAMATH_CALUDE_power_product_equality_l2664_266415

theorem power_product_equality (a b : ℝ) : a^3 * b^3 = (a*b)^3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2664_266415


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2664_266440

/-- An isosceles triangle with sides of 3cm and 7cm has a perimeter of 13cm. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 3 ∧ b = 7 ∧ c = 3 →  -- Two sides are 3cm, one side is 7cm
  (a = b ∨ b = c ∨ a = c) →  -- The triangle is isosceles
  a + b + c = 13 :=  -- The perimeter is 13cm
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2664_266440


namespace NUMINAMATH_CALUDE_frog_expected_returns_l2664_266484

/-- Represents the probability of moving in a certain direction or getting eaten -/
def move_probability : ℚ := 1 / 3

/-- Represents the frog's position on the number line -/
def Position : Type := ℤ

/-- Calculates the probability of returning to the starting position from a given position -/
noncomputable def prob_return_to_start (pos : Position) : ℝ :=
  sorry

/-- Calculates the expected number of returns to the starting position before getting eaten -/
noncomputable def expected_returns : ℝ :=
  sorry

/-- The main theorem stating the expected number of returns -/
theorem frog_expected_returns :
  expected_returns = (3 * Real.sqrt 5 - 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_frog_expected_returns_l2664_266484


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2664_266488

/-- The equation of a hyperbola with specific properties -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a : ℝ), x^2/a^2 + y^2/4 = 1) →  -- Ellipse equation
  (∀ (t : ℝ), y = 2*x ∨ y = -2*x) →  -- Asymptotes
  (x^2 - y^2/4 = 1) →                -- Proposed hyperbola equation
  (x = 1 ∧ y = 0) →                  -- Right vertex of the ellipse
  True                               -- The equation represents the correct hyperbola
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2664_266488


namespace NUMINAMATH_CALUDE_h_shape_perimeter_is_44_l2664_266434

/-- The perimeter of a rectangle with length l and width w -/
def rectanglePerimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

/-- The perimeter of an H shape formed by two vertical rectangles and one horizontal rectangle -/
def hShapePerimeter (v_length v_width h_length h_width : ℝ) : ℝ :=
  2 * (rectanglePerimeter v_length v_width) + 
  (rectanglePerimeter h_length h_width) - 
  2 * (2 * h_width)

theorem h_shape_perimeter_is_44 : 
  hShapePerimeter 6 3 6 2 = 44 := by sorry

end NUMINAMATH_CALUDE_h_shape_perimeter_is_44_l2664_266434


namespace NUMINAMATH_CALUDE_S_max_at_14_l2664_266431

/-- The sequence term for index n -/
def a (n : ℕ) : ℤ := 43 - 3 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := n * (40 + 43 - 3 * n) / 2

/-- The theorem stating that S reaches its maximum when n = 14 -/
theorem S_max_at_14 : ∀ k : ℕ, k > 0 → S 14 ≥ S k := by sorry

end NUMINAMATH_CALUDE_S_max_at_14_l2664_266431


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l2664_266417

theorem triangle_inequality_squared (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a < b + c ∧ b < a + c ∧ c < a + b) : 
  a^2 < a*b + a*c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_squared_l2664_266417


namespace NUMINAMATH_CALUDE_first_group_size_is_eight_l2664_266474

/-- The number of men in the first group that can complete a work in 18 days, working 8 hours a day -/
def first_group_size : ℕ := sorry

/-- The number of hours worked per day by both groups -/
def hours_per_day : ℕ := 8

/-- The number of days the first group takes to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def second_group_size : ℕ := 12

/-- The number of days the second group takes to complete the work -/
def days_second_group : ℕ := 12

/-- The total amount of work done is constant and equal for both groups -/
axiom work_done_equal : first_group_size * hours_per_day * days_first_group = second_group_size * hours_per_day * days_second_group

theorem first_group_size_is_eight : first_group_size = 8 := by sorry

end NUMINAMATH_CALUDE_first_group_size_is_eight_l2664_266474


namespace NUMINAMATH_CALUDE_no_rectangle_satisfies_conditions_l2664_266406

theorem no_rectangle_satisfies_conditions (p q : ℝ) (hp : p > q) (hq : q > 0) :
  ¬∃ x y : ℝ, x < p ∧ y < q ∧ x + y = (p + q) / 2 ∧ x * y = p * q / 4 := by
  sorry

end NUMINAMATH_CALUDE_no_rectangle_satisfies_conditions_l2664_266406


namespace NUMINAMATH_CALUDE_triangle_side_length_l2664_266435

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a + b + c = 20 →
  (1/2) * b * c * Real.sin A = 10 * Real.sqrt 3 →
  A = π / 3 →
  a = 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2664_266435


namespace NUMINAMATH_CALUDE_sara_green_marbles_l2664_266433

def total_green_marbles : ℕ := 7
def tom_green_marbles : ℕ := 4
def sara_red_marbles : ℕ := 5

theorem sara_green_marbles :
  ∃ (x : ℕ), x = total_green_marbles - tom_green_marbles :=
sorry

end NUMINAMATH_CALUDE_sara_green_marbles_l2664_266433


namespace NUMINAMATH_CALUDE_team_a_win_probability_l2664_266481

/-- The probability of Team A winning a single game -/
def p_win : ℚ := 3/5

/-- The probability of Team A losing a single game -/
def p_lose : ℚ := 2/5

/-- The number of ways to choose 2 wins out of 3 games -/
def combinations : ℕ := 3

theorem team_a_win_probability :
  combinations * p_win^3 * p_lose = 162/625 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l2664_266481


namespace NUMINAMATH_CALUDE_total_savings_theorem_l2664_266427

/-- The amount of money saved per month in dollars -/
def monthly_savings : ℕ := 4000

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: If Abigail saves $4,000 every month for an entire year, 
    the total amount saved will be $48,000 -/
theorem total_savings_theorem : 
  monthly_savings * months_in_year = 48000 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_theorem_l2664_266427


namespace NUMINAMATH_CALUDE_c_2017_value_l2664_266423

/-- Sequence a_n -/
def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => a n + 3

/-- Sequence b_n -/
def b : ℕ → ℕ
  | 0 => 3
  | n + 1 => 3 * b n

/-- Sequence c_n -/
def c (n : ℕ) : ℕ := b (a n - 1)

theorem c_2017_value : c 2016 = 27^2017 := by sorry

end NUMINAMATH_CALUDE_c_2017_value_l2664_266423


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_neg_six_fifths_l2664_266419

def h (x : ℝ) : ℝ := 5 * x + 6

theorem h_zero_iff_b_eq_neg_six_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = -6/5 := by sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_neg_six_fifths_l2664_266419


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l2664_266424

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l2664_266424


namespace NUMINAMATH_CALUDE_factorization_equality_l2664_266482

theorem factorization_equality (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2664_266482


namespace NUMINAMATH_CALUDE_distribute_and_simplify_l2664_266493

theorem distribute_and_simplify (a : ℝ) : -3 * a^2 * (4*a - 3) = -12*a^3 + 9*a^2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_simplify_l2664_266493


namespace NUMINAMATH_CALUDE_negative_two_minus_six_l2664_266444

theorem negative_two_minus_six : -2 - 6 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_minus_six_l2664_266444


namespace NUMINAMATH_CALUDE_bus_speed_and_interval_l2664_266468

/-- The speed of buses and interval between departures in a traffic scenario --/
theorem bus_speed_and_interval (a b c : ℝ) (hc : c > b) (hb : b > 0) (ha : a > 0) :
  ∃ (x t : ℝ),
    (a + x) * b = t * x ∧
    (x - a) * c = t * x ∧
    x = a * (c + b) / (c - b) ∧
    t = 2 * b * c / (b + c) := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_and_interval_l2664_266468


namespace NUMINAMATH_CALUDE_unique_solution_tan_equation_l2664_266470

theorem unique_solution_tan_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan (150 * π / 180 - x * π / 180) =
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_tan_equation_l2664_266470


namespace NUMINAMATH_CALUDE_cube_square_fraction_inequality_l2664_266416

theorem cube_square_fraction_inequality (s r : ℝ) (hs : s > 0) (hr : r > 0) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_square_fraction_inequality_l2664_266416


namespace NUMINAMATH_CALUDE_problem_statement_l2664_266442

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) :
  (x - 3)^2 + 36/(x - 3)^2 = 12.375 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2664_266442


namespace NUMINAMATH_CALUDE_marsupial_protein_consumption_l2664_266464

theorem marsupial_protein_consumption (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_consumed : ℝ) : 
  absorption_rate = 0.40 →
  absorbed_amount = 16 →
  absorbed_amount = absorption_rate * total_consumed →
  total_consumed = 40 := by
  sorry

end NUMINAMATH_CALUDE_marsupial_protein_consumption_l2664_266464


namespace NUMINAMATH_CALUDE_counterexample_exists_l2664_266449

theorem counterexample_exists : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2664_266449


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l2664_266454

theorem power_mod_thirteen : (6 ^ 1234 : ℕ) % 13 = 10 := by sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l2664_266454


namespace NUMINAMATH_CALUDE_sin_870_degrees_l2664_266447

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_870_degrees_l2664_266447


namespace NUMINAMATH_CALUDE_profit_equation_l2664_266428

/-- The profit function for a commodity -/
def profit (x : ℝ) : ℝ :=
  let cost_price : ℝ := 30
  let quantity_sold : ℝ := 200 - x
  (x - cost_price) * quantity_sold

theorem profit_equation (x : ℝ) : profit x = -x^2 + 230*x - 6000 := by
  sorry

end NUMINAMATH_CALUDE_profit_equation_l2664_266428


namespace NUMINAMATH_CALUDE_total_handshakes_l2664_266429

def number_of_couples : ℕ := 15

-- Define the number of handshakes between men
def handshakes_between_men (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define the number of handshakes between women
def handshakes_between_women (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define the number of handshakes between men and women (excluding spouses)
def handshakes_men_women (n : ℕ) : ℕ := n * (n - 1)

theorem total_handshakes :
  handshakes_between_men number_of_couples +
  handshakes_between_women number_of_couples +
  handshakes_men_women number_of_couples = 420 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_l2664_266429


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2664_266463

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2664_266463


namespace NUMINAMATH_CALUDE_book_pages_count_l2664_266438

/-- Represents the number of pages Bill reads on a given day -/
def pagesReadOnDay (day : ℕ) : ℕ := 10 + 2 * (day - 1)

/-- Represents the total number of pages Bill has read up to a given day -/
def totalPagesRead (days : ℕ) : ℕ := (days * (pagesReadOnDay 1 + pagesReadOnDay days)) / 2

theorem book_pages_count :
  ∀ (total_days : ℕ) (reading_days : ℕ),
  total_days = 14 →
  reading_days = total_days - 2 →
  (totalPagesRead reading_days : ℚ) = (3/4) * (336 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_book_pages_count_l2664_266438


namespace NUMINAMATH_CALUDE_move_right_four_units_l2664_266446

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally in a Cartesian coordinate system -/
def moveRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem move_right_four_units :
  let p := Point.mk (-2) 3
  moveRight p 4 = Point.mk 2 3 := by
  sorry

end NUMINAMATH_CALUDE_move_right_four_units_l2664_266446


namespace NUMINAMATH_CALUDE_largest_zero_correct_l2664_266478

/-- Sequence S defined recursively -/
def S : ℕ → ℤ
  | 0 => 0
  | (n + 1) => S n + (n + 1) * (if S n < n + 1 then 1 else -1)

/-- Predicate for S[k] = 0 -/
def is_zero (k : ℕ) : Prop := S k = 0

/-- The largest k ≤ 2010 such that S[k] = 0 -/
def largest_zero : ℕ := 1092

theorem largest_zero_correct :
  is_zero largest_zero ∧
  ∀ k, k ≤ 2010 → is_zero k → k ≤ largest_zero :=
by sorry

end NUMINAMATH_CALUDE_largest_zero_correct_l2664_266478


namespace NUMINAMATH_CALUDE_solution_difference_l2664_266483

theorem solution_difference (r s : ℝ) : 
  ((r - 4) * (r + 4) = 24 * r - 96) →
  ((s - 4) * (s + 4) = 24 * s - 96) →
  r ≠ s →
  r > s →
  r - s = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2664_266483


namespace NUMINAMATH_CALUDE_exponent_division_l2664_266495

theorem exponent_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2664_266495


namespace NUMINAMATH_CALUDE_product_divisible_by_12_l2664_266479

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability that a single roll is not divisible by 3 -/
def prob_not_div_3 : ℚ := 5 / 8

/-- The probability that a single roll is divisible by 4 -/
def prob_div_4 : ℚ := 1 / 4

/-- The probability that the product of the rolls is divisible by 12 -/
def prob_div_12 : ℚ := 149 / 256

theorem product_divisible_by_12 :
  (1 - prob_not_div_3 ^ num_dice) *
  (1 - (1 - prob_div_4) ^ num_dice - num_dice * prob_div_4 * (1 - prob_div_4) ^ (num_dice - 1)) =
  prob_div_12 := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_12_l2664_266479


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l2664_266453

/-- Given two cylinders A and B, where the radius of A equals the height of B,
    and the height of A equals the radius of B, if the volume of A is three times
    the volume of B, then the volume of A can be expressed as 9πh^3,
    where h is the height of A. -/
theorem cylinder_volume_relation (h r : ℝ) : 
  h > 0 → r > 0 → 
  (π * r^2 * h) = 3 * (π * h^2 * r) → 
  ∃ (N : ℝ), π * r^2 * h = N * π * h^3 ∧ N = 9 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l2664_266453


namespace NUMINAMATH_CALUDE_recurring_decimal_equiv_recurring_decimal_lowest_terms_l2664_266408

def recurring_decimal : ℚ := 433 / 990

theorem recurring_decimal_equiv : recurring_decimal = 0.4375375375375375375375375375375 := by sorry

theorem recurring_decimal_lowest_terms : ∀ a b : ℤ, (a : ℚ) / b = recurring_decimal → Nat.gcd a.natAbs b.natAbs = 1 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_equiv_recurring_decimal_lowest_terms_l2664_266408


namespace NUMINAMATH_CALUDE_time_to_see_all_animals_after_import_l2664_266448

/-- Calculates the time required to see all animal types after importing new species -/
def time_to_see_all_animals (initial_types : ℕ) (time_per_type : ℕ) (new_species : ℕ) : ℕ :=
  (initial_types + new_species) * time_per_type

/-- Proves that the time required to see all animal types after importing new species is 54 minutes -/
theorem time_to_see_all_animals_after_import :
  time_to_see_all_animals 5 6 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_time_to_see_all_animals_after_import_l2664_266448


namespace NUMINAMATH_CALUDE_min_dot_product_l2664_266422

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define a point on the hyperbola in the first quadrant
def point_on_hyperbola (M : ℝ × ℝ) : Prop :=
  hyperbola M.1 M.2 ∧ M.1 > 0 ∧ M.2 > 0

-- Define the tangent line at point M
def tangent_line (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  point_on_hyperbola M ∧ 
  ∃ (t : ℝ), P = (M.1 + t, M.2 + t) ∧ Q = (M.1 - t, M.2 - t)

-- Define P in the first quadrant
def P_in_first_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 > 0

-- Define R on the same asymptote as Q
def R_on_asymptote (Q R : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), R = (k * Q.1, k * Q.2)

-- Theorem statement
theorem min_dot_product 
  (M P Q R : ℝ × ℝ) 
  (h1 : tangent_line M P Q)
  (h2 : P_in_first_quadrant P)
  (h3 : R_on_asymptote Q R) :
  ∃ (min_value : ℝ), 
    (∀ (R' : ℝ × ℝ), R_on_asymptote Q R' → 
      (R'.1 - P.1) * (R'.1 - Q.1) + (R'.2 - P.2) * (R'.2 - Q.2) ≥ min_value) ∧
    min_value = -1/2 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l2664_266422


namespace NUMINAMATH_CALUDE_first_player_wins_l2664_266437

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game --/
inductive Player | First | Second

/-- Represents a move in the game --/
structure Move :=
  (top_left : ℕ × ℕ)
  (size : ℕ)

/-- The game state --/
structure GameState :=
  (grid : Grid)
  (current_player : Player)
  (moves : List Move)

/-- Checks if a move is valid --/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Applies a move to the game state --/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over --/
def is_game_over (state : GameState) : Prop :=
  sorry

/-- Determines the winner of the game --/
def winner (state : GameState) : Option Player :=
  sorry

/-- Represents a strategy for playing the game --/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player --/
def is_winning_strategy (strategy : Strategy) (player : Player) : Prop :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player --/
theorem first_player_wins :
  ∃ (strategy : Strategy), is_winning_strategy strategy Player.First :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2664_266437


namespace NUMINAMATH_CALUDE_triangle_longest_side_range_l2664_266405

/-- Given a triangle with perimeter 12 and b as the longest side, prove the range of b -/
theorem triangle_longest_side_range (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive side lengths
  a + b + c = 12 →         -- perimeter is 12
  b ≥ a ∧ b ≥ c →          -- b is the longest side
  4 < b ∧ b < 6 :=         -- range of b
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_range_l2664_266405


namespace NUMINAMATH_CALUDE_not_p_or_not_q_is_true_l2664_266455

theorem not_p_or_not_q_is_true :
  ∀ (a b c : ℝ),
  let p := ∀ (a b c : ℝ), a > b → a + c > b + c
  let q := ∀ (a b c : ℝ), a > b ∧ b > 0 → a * c > b * c
  ¬p ∨ ¬q := by sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_is_true_l2664_266455


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2664_266409

noncomputable def x₁ (t C₁ C₂ : ℝ) : ℝ :=
  (C₁ + C₂ - 2 * t^2) / (2 * (C₁ - t^2) * (C₂ - t^2))

noncomputable def x₂ (t C₁ C₂ : ℝ) : ℝ :=
  (C₂ - C₁) / (2 * (C₁ - t^2) * (C₂ - t^2))

theorem solution_satisfies_system (t C₁ C₂ : ℝ) :
  deriv (fun t => x₁ t C₁ C₂) t = 2 * (x₁ t C₁ C₂)^2 * t + 2 * (x₂ t C₁ C₂)^2 * t ∧
  deriv (fun t => x₂ t C₁ C₂) t = 4 * (x₁ t C₁ C₂) * (x₂ t C₁ C₂) * t :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2664_266409


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l2664_266494

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (2, 3)

/-- First line equation -/
def line1 (x y : ℝ) : Prop := 10 * x - 5 * y = 5

/-- Second line equation -/
def line2 (x y : ℝ) : Prop := 8 * x + 2 * y = 22

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l2664_266494


namespace NUMINAMATH_CALUDE_exists_universal_shape_l2664_266490

/-- Represents a tetrimino --/
structure Tetrimino where
  cells : Finset (ℤ × ℤ)
  cell_count : cells.card = 4

/-- Represents the five types of tetriminoes --/
inductive TetriminoType
  | O
  | I
  | L
  | T
  | Z

/-- A shape is a set of cells in the plane --/
def Shape := Finset (ℤ × ℤ)

/-- Rotation of a tetrimino --/
def rotate (t : Tetrimino) : Tetrimino := sorry

/-- Check if a shape can be composed using only one type of tetrimino --/
def canComposeWithType (s : Shape) (type : TetriminoType) : Prop := sorry

/-- The main theorem --/
theorem exists_universal_shape :
  ∃ (s : Shape), ∀ (type : TetriminoType), canComposeWithType s type := by sorry

end NUMINAMATH_CALUDE_exists_universal_shape_l2664_266490


namespace NUMINAMATH_CALUDE_dormitory_problem_l2664_266420

theorem dormitory_problem (x : ℕ) 
  (h1 : x > 0) 
  (h2 : 4 * x + 18 < 6 * x) 
  (h3 : 4 * x + 18 > 6 * (x - 1)) : 
  x = 10 ∨ x = 11 := by
sorry

end NUMINAMATH_CALUDE_dormitory_problem_l2664_266420


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2664_266404

-- Define the polynomial f
def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

-- Theorem stating that f satisfies all conditions
theorem f_satisfies_conditions :
  -- f is a polynomial in x, y, z (implied by its definition)
  -- f is of degree 4 in x (implied by its definition)
  -- First condition
  (∀ x y z : ℝ, f x (z^2) y + f x (y^2) z = 0) ∧
  -- Second condition
  (∀ x y z : ℝ, f (z^3) y x + f (x^3) y z = 0) :=
by sorry


end NUMINAMATH_CALUDE_f_satisfies_conditions_l2664_266404


namespace NUMINAMATH_CALUDE_max_prime_difference_l2664_266476

theorem max_prime_difference (a b c d : ℕ) : 
  a.Prime ∧ b.Prime ∧ c.Prime ∧ d.Prime ∧
  (a + b + c + 18 + d).Prime ∧ (a + b + c + 18 - d).Prime ∧
  (b + c).Prime ∧ (c + d).Prime ∧
  (a + b + c = 2010) ∧
  (a ≠ 3 ∧ b ≠ 3 ∧ c ≠ 3 ∧ d ≠ 3) ∧
  (d ≤ 50) →
  (∃ (p q : ℕ), (p.Prime ∧ q.Prime ∧ 
    (p = a ∨ p = b ∨ p = c ∨ p = d ∨ 
     p = a + b + c + 18 + d ∨ p = a + b + c + 18 - d ∨
     p = b + c ∨ p = c + d) ∧
    (q = a ∨ q = b ∨ q = c ∨ q = d ∨ 
     q = a + b + c + 18 + d ∨ q = a + b + c + 18 - d ∨
     q = b + c ∨ q = c + d) ∧
    p - q ≤ 2067) ∧
   ∀ (r s : ℕ), (r.Prime ∧ s.Prime ∧ 
    (r = a ∨ r = b ∨ r = c ∨ r = d ∨ 
     r = a + b + c + 18 + d ∨ r = a + b + c + 18 - d ∨
     r = b + c ∨ r = c + d) ∧
    (s = a ∨ s = b ∨ s = c ∨ s = d ∨ 
     s = a + b + c + 18 + d ∨ s = a + b + c + 18 - d ∨
     s = b + c ∨ s = c + d) →
    r - s ≤ 2067)) :=
by sorry

end NUMINAMATH_CALUDE_max_prime_difference_l2664_266476


namespace NUMINAMATH_CALUDE_mr_blue_flower_bed_yield_l2664_266497

/-- Represents the dimensions and yield of a flower bed -/
structure FlowerBed where
  length_paces : ℕ
  width_paces : ℕ
  pace_length : ℝ
  yield_per_sqft : ℝ

/-- Calculates the expected rose petal yield from a flower bed -/
def expected_yield (fb : FlowerBed) : ℝ :=
  (fb.length_paces : ℝ) * fb.pace_length *
  (fb.width_paces : ℝ) * fb.pace_length *
  fb.yield_per_sqft

/-- Theorem stating the expected yield for Mr. Blue's flower bed -/
theorem mr_blue_flower_bed_yield :
  let fb : FlowerBed := {
    length_paces := 18,
    width_paces := 24,
    pace_length := 1.5,
    yield_per_sqft := 0.4
  }
  expected_yield fb = 388.8 := by sorry

end NUMINAMATH_CALUDE_mr_blue_flower_bed_yield_l2664_266497


namespace NUMINAMATH_CALUDE_weakly_decreasing_exp_weakly_decreasing_ln_condition_weakly_decreasing_cos_condition_l2664_266496

-- Definition of weakly decreasing function
def WeaklyDecreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x ≥ f y) ∧
  (∀ x ∈ I, ∀ y ∈ I, x < y → x * f x ≤ y * f y)

-- Theorem 1
theorem weakly_decreasing_exp (x : ℝ) :
  WeaklyDecreasing (fun x => x / Real.exp x) (Set.Ioo 1 2) :=
sorry

-- Theorem 2
theorem weakly_decreasing_ln_condition (m : ℝ) :
  WeaklyDecreasing (fun x => Real.log x / x) (Set.Ioi m) → m ≥ Real.exp 1 :=
sorry

-- Theorem 3
theorem weakly_decreasing_cos_condition (k : ℝ) :
  WeaklyDecreasing (fun x => Real.cos x + k * x^2) (Set.Ioo 0 (Real.pi / 2)) →
  2 / (3 * Real.pi) ≤ k ∧ k ≤ 1 / Real.pi :=
sorry

end NUMINAMATH_CALUDE_weakly_decreasing_exp_weakly_decreasing_ln_condition_weakly_decreasing_cos_condition_l2664_266496


namespace NUMINAMATH_CALUDE_total_days_2000_to_2003_l2664_266400

def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 ≠ 0 || year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDaysInRange (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |>.sum

theorem total_days_2000_to_2003 :
  totalDaysInRange 2000 2003 = 1461 :=
by
  sorry

end NUMINAMATH_CALUDE_total_days_2000_to_2003_l2664_266400


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l2664_266466

theorem unique_solution_floor_equation :
  ∃! x : ℝ, x + ⌊x⌋ = 20.2 ∧ x = 10.2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l2664_266466


namespace NUMINAMATH_CALUDE_principal_interest_difference_l2664_266461

/-- Calculate the difference between principal and interest for a simple interest loan. -/
theorem principal_interest_difference
  (principal : ℕ)
  (rate : ℕ)
  (time : ℕ)
  (h1 : principal = 6200)
  (h2 : rate = 5)
  (h3 : time = 10) :
  principal - (principal * rate * time) / 100 = 3100 :=
by
  sorry

end NUMINAMATH_CALUDE_principal_interest_difference_l2664_266461


namespace NUMINAMATH_CALUDE_simple_random_sampling_is_most_appropriate_l2664_266458

/-- Represents a box containing units of a product -/
structure Box where
  name : String
  units : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Cluster

/-- Determines if a sampling method is appropriate for the given boxes and sample size -/
def is_appropriate_sampling_method (boxes : List Box) (sample_size : ℕ) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.SimpleRandom => true
  | _ => false

theorem simple_random_sampling_is_most_appropriate :
  let boxes : List Box := [
    { name := "large", units := 120 },
    { name := "medium", units := 60 },
    { name := "small", units := 20 }
  ]
  let sample_size : ℕ := 25
  ∀ method : SamplingMethod,
    is_appropriate_sampling_method boxes sample_size method →
    method = SamplingMethod.SimpleRandom :=
by
  sorry


end NUMINAMATH_CALUDE_simple_random_sampling_is_most_appropriate_l2664_266458


namespace NUMINAMATH_CALUDE_point_not_in_A_when_a_negative_l2664_266485

-- Define the set A
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 ≥ 0 ∧ a * p.1 + p.2 ≥ 2 ∧ p.1 - a * p.2 ≤ 2}

-- Theorem statement
theorem point_not_in_A_when_a_negative :
  ∀ a : ℝ, a < 0 → (1, 1) ∉ A a :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_A_when_a_negative_l2664_266485


namespace NUMINAMATH_CALUDE_sin_eighth_integral_l2664_266407

theorem sin_eighth_integral : ∫ x in (0)..(2*Real.pi), (Real.sin x)^8 = (35 * Real.pi) / 64 := by sorry

end NUMINAMATH_CALUDE_sin_eighth_integral_l2664_266407


namespace NUMINAMATH_CALUDE_negative_integers_satisfying_condition_l2664_266403

def satisfies_condition (a : Int) : Prop :=
  (4 * a + 1 : ℚ) / 6 > -2

theorem negative_integers_satisfying_condition :
  {a : Int | a < 0 ∧ satisfies_condition a} = {-1, -2, -3} := by
  sorry

end NUMINAMATH_CALUDE_negative_integers_satisfying_condition_l2664_266403


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2664_266491

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {y | ∃ x, y = 2^x - 1}

theorem intersection_of_A_and_B : A ∩ B = {m | -1 < m ∧ m < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2664_266491


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l2664_266498

/-- The cost price of a bicycle given a series of sales with specified profit margins -/
theorem bicycle_cost_price
  (profit_A_to_B : Real)
  (profit_B_to_C : Real)
  (profit_C_to_D : Real)
  (final_price : Real)
  (h1 : profit_A_to_B = 0.50)
  (h2 : profit_B_to_C = 0.25)
  (h3 : profit_C_to_D = 0.15)
  (h4 : final_price = 320.75) :
  ∃ (cost_price : Real),
    cost_price = final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D)) :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l2664_266498


namespace NUMINAMATH_CALUDE_price_difference_l2664_266413

theorem price_difference (P : ℝ) (P_positive : P > 0) : 
  let new_price := P * 1.2
  let discounted_price := new_price * 0.8
  new_price - discounted_price = P * 0.24 :=
by sorry

end NUMINAMATH_CALUDE_price_difference_l2664_266413


namespace NUMINAMATH_CALUDE_fourth_grade_classrooms_difference_l2664_266452

theorem fourth_grade_classrooms_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 20 →
  guinea_pigs_per_class = 3 →
  num_classes = 5 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_classrooms_difference_l2664_266452


namespace NUMINAMATH_CALUDE_students_not_visiting_any_exhibit_l2664_266457

def total_students : ℕ := 52
def botanical_visitors : ℕ := 12
def animal_visitors : ℕ := 26
def technology_visitors : ℕ := 23
def botanical_and_animal : ℕ := 5
def botanical_and_technology : ℕ := 2
def animal_and_technology : ℕ := 4
def all_three : ℕ := 1

theorem students_not_visiting_any_exhibit : 
  total_students - (botanical_visitors + animal_visitors + technology_visitors
                    - botanical_and_animal - botanical_and_technology - animal_and_technology
                    + all_three) = 1 := by sorry

end NUMINAMATH_CALUDE_students_not_visiting_any_exhibit_l2664_266457


namespace NUMINAMATH_CALUDE_min_sum_last_three_digits_equal_l2664_266410

/-- 
Given two positive integers m and n, where n > m ≥ 1, 
and the last three digits of 1978^n and 1978^m are equal,
prove that the minimum value of m + n is 106.
-/
theorem min_sum_last_three_digits_equal (m n : ℕ) : 
  m ≥ 1 → n > m → 
  (1978^n) % 1000 = (1978^m) % 1000 → 
  ∃ (m₀ n₀ : ℕ), m₀ ≥ 1 ∧ n₀ > m₀ ∧ 
    (1978^n₀) % 1000 = (1978^m₀) % 1000 ∧
    m₀ + n₀ = 106 ∧ 
    ∀ (m' n' : ℕ), m' ≥ 1 → n' > m' → 
      (1978^n') % 1000 = (1978^m') % 1000 → 
      m' + n' ≥ 106 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_last_three_digits_equal_l2664_266410


namespace NUMINAMATH_CALUDE_product_of_odd_negative_integers_l2664_266436

def odd_negative_integers : List ℤ := sorry

theorem product_of_odd_negative_integers :
  let product := (List.prod odd_negative_integers)
  (product < 0) ∧ (product % 10 = -5) := by
  sorry

end NUMINAMATH_CALUDE_product_of_odd_negative_integers_l2664_266436


namespace NUMINAMATH_CALUDE_min_value_a2_b2_l2664_266467

/-- Given that (ax^2 + b/x)^6 has a coefficient of 20 for x^3, 
    the minimum value of a^2 + b^2 is 2 -/
theorem min_value_a2_b2 (a b : ℝ) : 
  (∃ c : ℝ, c = 20 ∧ 
   c = (Nat.choose 6 3 : ℝ) * a^3 * b^3) → 
  ∀ x y : ℝ, x^2 + y^2 ≥ 2 ∧ (x^2 + y^2 = 2 → x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a2_b2_l2664_266467


namespace NUMINAMATH_CALUDE_lottery_probabilities_l2664_266469

/-- Represents the total number of lottery tickets -/
def total_tickets : ℕ := 12

/-- Represents the number of winning tickets -/
def winning_tickets : ℕ := 2

/-- Represents the number of people -/
def num_people : ℕ := 4

/-- Represents the probability of giving 2 winning tickets to different people -/
def prob_different_people : ℚ := 9/11

/-- Represents the probability of giving 1 winning ticket to A and 1 to B -/
def prob_A_and_B : ℚ := 3/22

/-- Theorem stating the probabilities for the lottery ticket distribution -/
theorem lottery_probabilities :
  (prob_different_people = 9/11) ∧ (prob_A_and_B = 3/22) :=
sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l2664_266469


namespace NUMINAMATH_CALUDE_sum_of_digits_theorem_l2664_266471

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: Given S(n) = 1365, S(n+1) = 1360 -/
theorem sum_of_digits_theorem (n : ℕ) (h : S n = 1365) : S (n + 1) = 1360 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_theorem_l2664_266471


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2664_266480

/-- Given that the terminal side of angle α intersects the unit circle 
    at point P(-4/5, 3/5), prove that cos 2α = 7/25 -/
theorem cos_double_angle_special_case (α : Real) 
  (h : ∃ (P : Real × Real), P.1 = -4/5 ∧ P.2 = 3/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
       P.1 = Real.cos α ∧ P.2 = Real.sin α) : 
  Real.cos (2 * α) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2664_266480


namespace NUMINAMATH_CALUDE_tshirt_packages_l2664_266460

theorem tshirt_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 56)
  (h2 : tshirts_per_package = 2) :
  total_tshirts / tshirts_per_package = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_tshirt_packages_l2664_266460


namespace NUMINAMATH_CALUDE_group_size_calculation_l2664_266430

theorem group_size_calculation (n : ℕ) : 
  (n : ℝ) * 14 + 34 = ((n : ℝ) + 1) * 16 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2664_266430


namespace NUMINAMATH_CALUDE_same_solution_implies_m_value_l2664_266456

theorem same_solution_implies_m_value :
  ∀ (m : ℝ) (x : ℝ),
    (-5 * x - 6 = 3 * x + 10) ∧
    (-2 * m - 3 * x = 10) →
    m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_value_l2664_266456


namespace NUMINAMATH_CALUDE_bike_owners_without_scooters_l2664_266486

theorem bike_owners_without_scooters (total_population : ℕ) 
  (bike_owners : ℕ) (scooter_owners : ℕ) 
  (h1 : total_population = 420)
  (h2 : bike_owners = 380)
  (h3 : scooter_owners = 82)
  (h4 : ∀ p, p ∈ Set.range (Fin.val : Fin total_population → ℕ) → 
    (p ∈ Set.range (Fin.val : Fin bike_owners → ℕ) ∨ 
     p ∈ Set.range (Fin.val : Fin scooter_owners → ℕ))) :
  bike_owners - (bike_owners + scooter_owners - total_population) = 338 :=
sorry

end NUMINAMATH_CALUDE_bike_owners_without_scooters_l2664_266486


namespace NUMINAMATH_CALUDE_total_teaching_time_l2664_266402

/-- Represents a teacher's class schedule -/
structure Schedule where
  math_classes : ℕ
  science_classes : ℕ
  history_classes : ℕ
  math_duration : ℝ
  science_duration : ℝ
  history_duration : ℝ

/-- Calculates the total teaching time for a given schedule -/
def total_time (s : Schedule) : ℝ :=
  s.math_classes * s.math_duration +
  s.science_classes * s.science_duration +
  s.history_classes * s.history_duration

/-- Eduardo's teaching schedule -/
def eduardo : Schedule :=
  { math_classes := 3
    science_classes := 4
    history_classes := 2
    math_duration := 1
    science_duration := 1.5
    history_duration := 2 }

/-- Frankie's teaching schedule (double of Eduardo's) -/
def frankie : Schedule :=
  { math_classes := 2 * eduardo.math_classes
    science_classes := 2 * eduardo.science_classes
    history_classes := 2 * eduardo.history_classes
    math_duration := eduardo.math_duration
    science_duration := eduardo.science_duration
    history_duration := eduardo.history_duration }

/-- Theorem: The total teaching time for Eduardo and Frankie is 39 hours -/
theorem total_teaching_time : total_time eduardo + total_time frankie = 39 := by
  sorry


end NUMINAMATH_CALUDE_total_teaching_time_l2664_266402


namespace NUMINAMATH_CALUDE_triangle_area_angle_relation_l2664_266421

theorem triangle_area_angle_relation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let A := Real.sqrt 3 / 12 * (a^2 + c^2 - b^2)
  (∃ (B : ℝ), 0 < B ∧ B < π ∧ A = 1/2 * a * c * Real.sin B) → 
  (∃ (B : ℝ), 0 < B ∧ B < π ∧ B = π/6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_angle_relation_l2664_266421


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2664_266473

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (is_arithmetic_sequence a → a 1 + a 3 = 2 * a 2) ∧
  (∃ a : ℕ → ℝ, a 1 + a 3 = 2 * a 2 ∧ ¬is_arithmetic_sequence a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2664_266473


namespace NUMINAMATH_CALUDE_max_students_on_playground_l2664_266465

def total_pencils : ℕ := 170
def total_notebooks : ℕ := 268
def total_erasers : ℕ := 120
def leftover_pencils : ℕ := 8
def shortage_notebooks : ℕ := 2
def leftover_erasers : ℕ := 12

theorem max_students_on_playground :
  let distributed_pencils := total_pencils - leftover_pencils
  let distributed_notebooks := total_notebooks + shortage_notebooks
  let distributed_erasers := total_erasers - leftover_erasers
  let max_students := Nat.gcd distributed_pencils (Nat.gcd distributed_notebooks distributed_erasers)
  max_students = 54 ∧
  (∃ (p n e : ℕ), 
    distributed_pencils = max_students * p ∧
    distributed_notebooks = max_students * n ∧
    distributed_erasers = max_students * e) ∧
  (∀ s : ℕ, s > max_students →
    ¬(∃ (p n e : ℕ),
      distributed_pencils = s * p ∧
      distributed_notebooks = s * n ∧
      distributed_erasers = s * e)) :=
by sorry

end NUMINAMATH_CALUDE_max_students_on_playground_l2664_266465


namespace NUMINAMATH_CALUDE_probability_two_red_one_blue_is_11_70_l2664_266451

def total_marbles : ℕ := 16
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 4

def probability_two_red_one_blue : ℚ :=
  (red_marbles * (red_marbles - 1) * blue_marbles) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem probability_two_red_one_blue_is_11_70 :
  probability_two_red_one_blue = 11 / 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_one_blue_is_11_70_l2664_266451


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_roots_l2664_266432

theorem perpendicular_lines_from_quadratic_roots : 
  ∀ (k₁ k₂ : ℝ), 
    k₁^2 - 3*k₁ - 1 = 0 → 
    k₂^2 - 3*k₂ - 1 = 0 → 
    k₁ ≠ k₂ →
    k₁ * k₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_roots_l2664_266432


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l2664_266411

theorem complementary_angles_ratio (a b : ℝ) : 
  a > 0 → b > 0 → -- angles are positive
  a + b = 90 → -- angles are complementary
  a = 4 * b → -- ratio of angles is 4:1
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l2664_266411


namespace NUMINAMATH_CALUDE_right_trapezoid_perimeter_l2664_266450

/-- A right trapezoid with upper base a, lower base b, height h, and leg l. -/
structure RightTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  l : ℝ

/-- The perimeter of a right trapezoid. -/
def perimeter (t : RightTrapezoid) : ℝ := t.a + t.b + t.h + t.l

/-- The theorem stating the conditions and the result for the right trapezoid problem. -/
theorem right_trapezoid_perimeter (t : RightTrapezoid) :
  t.a < t.b →
  π * t.h^2 * t.a + (1/3) * π * t.h^2 * (t.b - t.a) = 80 * π →
  π * t.h^2 * t.b + (1/3) * π * t.h^2 * (t.b - t.a) = 112 * π →
  (1/3) * π * (t.a^2 + t.a * t.b + t.b^2) * t.h = 156 * π →
  perimeter t = 20 + 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_perimeter_l2664_266450
