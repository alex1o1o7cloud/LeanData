import Mathlib

namespace NUMINAMATH_CALUDE_final_share_is_132_75_l2751_275149

/-- Calculates the final amount each person receives after combining and sharing their updated amounts equally. -/
def final_share (emani_initial : ℚ) (howard_difference : ℚ) (jamal_initial : ℚ) 
                (emani_increase : ℚ) (howard_increase : ℚ) (jamal_increase : ℚ) : ℚ :=
  let howard_initial := emani_initial - howard_difference
  let emani_updated := emani_initial * (1 + emani_increase)
  let howard_updated := howard_initial * (1 + howard_increase)
  let jamal_updated := jamal_initial * (1 + jamal_increase)
  let total_updated := emani_updated + howard_updated + jamal_updated
  total_updated / 3

/-- Theorem stating that each person receives $132.75 after combining and sharing their updated amounts equally. -/
theorem final_share_is_132_75 :
  final_share 150 30 75 (20/100) (10/100) (15/100) = 132.75 := by
  sorry

end NUMINAMATH_CALUDE_final_share_is_132_75_l2751_275149


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2751_275172

def original_proposition (x : ℝ) : Prop := x = 1 → x^2 - 3*x + 2 = 0

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) ↔ 
  (∀ x : ℝ, original_proposition x) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2751_275172


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l2751_275120

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) (num_progressives : ℚ) :
  num_provinces = 5 →
  num_traditionalists_per_province = num_progressives / 15 →
  (num_provinces * num_traditionalists_per_province) / (num_progressives + num_provinces * num_traditionalists_per_province) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l2751_275120


namespace NUMINAMATH_CALUDE_albany_syracuse_distance_l2751_275123

/-- The distance between Albany and Syracuse satisfies the equation relating to travel times. -/
theorem albany_syracuse_distance (D : ℝ) : D > 0 → D / 50 + D / 38.71 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_albany_syracuse_distance_l2751_275123


namespace NUMINAMATH_CALUDE_percent_of_percent_l2751_275108

theorem percent_of_percent (x : ℝ) : (0.3 * (0.6 * x)) = (0.18 * x) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l2751_275108


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2751_275128

open Set

-- Define the universal set
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2751_275128


namespace NUMINAMATH_CALUDE_ramesh_discount_percentage_l2751_275117

/-- The discount percentage Ramesh received on the refrigerator --/
def discount_percentage (purchase_price transport_cost installation_cost no_discount_sale_price : ℚ) : ℚ :=
  let labelled_price := no_discount_sale_price / 1.1
  let discount := labelled_price - purchase_price
  (discount / labelled_price) * 100

/-- Theorem stating the discount percentage Ramesh received --/
theorem ramesh_discount_percentage :
  let purchase_price : ℚ := 14500
  let transport_cost : ℚ := 125
  let installation_cost : ℚ := 250
  let no_discount_sale_price : ℚ := 20350
  abs (discount_percentage purchase_price transport_cost installation_cost no_discount_sale_price - 21.62) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ramesh_discount_percentage_l2751_275117


namespace NUMINAMATH_CALUDE_specific_right_triangle_perimeter_l2751_275165

/-- A right triangle with integer side lengths, one of which is 11. -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  is_right : a^2 + b^2 = c^2
  has_eleven : a = 11 ∨ b = 11

/-- The perimeter of a right triangle. -/
def perimeter (t : RightTriangle) : ℕ := t.a + t.b + t.c

/-- Theorem stating that the perimeter of the specific right triangle is 132. -/
theorem specific_right_triangle_perimeter :
  ∃ t : RightTriangle, perimeter t = 132 :=
sorry

end NUMINAMATH_CALUDE_specific_right_triangle_perimeter_l2751_275165


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l2751_275182

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 751 * 503 - 1) ∧ 
  ((956734 + x) % (751 * 503) = 0) ∧
  ∀ y : ℕ, y < x → ((956734 + y) % (751 * 503) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l2751_275182


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2751_275180

/-- Given a system of linear equations with a parameter m, prove that when the system has a nontrivial solution, the ratio of xz/y^2 is 20. -/
theorem system_solution_ratio (m : ℚ) (x y z : ℚ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + m*y + 4*z = 0 →
  4*x + m*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  (∃ m, x + m*y + 4*z = 0 ∧ 4*x + m*y - 3*z = 0 ∧ 3*x + 5*y - 4*z = 0) →
  x*z / (y^2) = 20 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2751_275180


namespace NUMINAMATH_CALUDE_ratio_problem_l2751_275124

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) : 
  second_part = 10 →
  percent = 20 →
  first_part / second_part = percent / 100 →
  first_part = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2751_275124


namespace NUMINAMATH_CALUDE_fraction_equality_l2751_275104

theorem fraction_equality (b : ℕ+) : 
  (b : ℚ) / (b + 15 : ℚ) = 3/4 → b = 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2751_275104


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2751_275132

theorem complex_magnitude_problem (Z : ℂ) (h : (2 + Complex.I) * Z = 3 - Complex.I) :
  Complex.abs Z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2751_275132


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l2751_275181

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 18 * y = 2 ∧ 4 * y - x = 6 ∧ x = -58/3 ∧ y = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l2751_275181


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_fraction_l2751_275177

theorem ceiling_negative_sqrt_fraction : ⌈-Real.sqrt (81 / 9)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_fraction_l2751_275177


namespace NUMINAMATH_CALUDE_sqrt_five_squared_times_seven_fourth_l2751_275107

theorem sqrt_five_squared_times_seven_fourth (x : ℝ) : 
  x = Real.sqrt (5^2 * 7^4) → x = 245 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_times_seven_fourth_l2751_275107


namespace NUMINAMATH_CALUDE_task_assignment_count_l2751_275183

def number_of_ways (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem task_assignment_count : 
  (number_of_ways 10 4) * (number_of_ways 4 2) * (number_of_ways 2 1) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_task_assignment_count_l2751_275183


namespace NUMINAMATH_CALUDE_washer_dryer_total_cost_l2751_275157

/-- The cost of a washer-dryer combination -/
def washer_dryer_cost (dryer_cost washer_cost_difference : ℕ) : ℕ :=
  dryer_cost + (dryer_cost + washer_cost_difference)

/-- Theorem: The washer-dryer combination costs $1200 -/
theorem washer_dryer_total_cost : 
  washer_dryer_cost 490 220 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_total_cost_l2751_275157


namespace NUMINAMATH_CALUDE_gain_percent_for_equal_cost_and_selling_l2751_275148

/-- Given that the cost price of 50 articles equals the selling price of 30 articles,
    prove that the gain percent is 200/3. -/
theorem gain_percent_for_equal_cost_and_selling (C S : ℝ) 
  (h : 50 * C = 30 * S) : 
  (S - C) / C * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_for_equal_cost_and_selling_l2751_275148


namespace NUMINAMATH_CALUDE_sports_club_members_l2751_275195

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  badminton : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the total number of members in the sports club -/
def total_members (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub),
    club.badminton = 17 ∧
    club.tennis = 19 ∧
    club.both = 9 ∧
    club.neither = 3 ∧
    total_members club = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l2751_275195


namespace NUMINAMATH_CALUDE_sum_of_u_and_v_l2751_275136

theorem sum_of_u_and_v (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = 17) 
  (eq2 : 5 * u - 2 * v = 1) : 
  u + v = -8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_u_and_v_l2751_275136


namespace NUMINAMATH_CALUDE_paul_work_days_l2751_275115

/-- The number of days it takes Rose to complete the work -/
def rose_days : ℝ := 120

/-- The number of days it takes Paul and Rose together to complete the work -/
def combined_days : ℝ := 48

/-- The number of days it takes Paul to complete the work alone -/
def paul_days : ℝ := 80

/-- Theorem stating that given Rose's and combined work rates, Paul's individual work rate can be determined -/
theorem paul_work_days (rose : ℝ) (combined : ℝ) (paul : ℝ) 
  (h_rose : rose = rose_days) 
  (h_combined : combined = combined_days) :
  paul = paul_days :=
by sorry

end NUMINAMATH_CALUDE_paul_work_days_l2751_275115


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2751_275170

/-- An ellipse with foci on the y-axis is represented by the equation (x^2)/(15-k) + (y^2)/(k-9) = 1,
    where k is a real number. This theorem states that the range of k is (12, 15). -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  k ∈ Set.Ioo 12 15 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2751_275170


namespace NUMINAMATH_CALUDE_power_of_power_l2751_275184

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2751_275184


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2751_275110

/-- Two fixed circles in a 2D plane -/
structure FixedCircles where
  C₁ : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x + 4)^2 + y^2 = 2
  C₂ : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - 4)^2 + y^2 = 2

/-- A moving circle tangent to both fixed circles -/
structure MovingCircle (fc : FixedCircles) where
  center : ℝ × ℝ
  isTangentToC₁ : Prop
  isTangentToC₂ : Prop

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 14 = 1 ∨ x = 0

/-- Theorem stating that the trajectory of the moving circle's center
    is described by the given equation -/
theorem moving_circle_trajectory (fc : FixedCircles) :
  ∀ (mc : MovingCircle fc), trajectory mc.center.1 mc.center.2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2751_275110


namespace NUMINAMATH_CALUDE_four_solutions_three_solutions_l2751_275164

/-- The equation x^2 - 4|x| + k = 0 with integer k and x -/
def equation (k : ℤ) (x : ℤ) : Prop := x^2 - 4 * x.natAbs + k = 0

/-- The set of integer solutions to the equation -/
def solution_set (k : ℤ) : Set ℤ := {x : ℤ | equation k x}

theorem four_solutions :
  solution_set 3 = {1, -1, 3, -3} :=
sorry

theorem three_solutions :
  solution_set 0 = {0, 4, -4} :=
sorry

end NUMINAMATH_CALUDE_four_solutions_three_solutions_l2751_275164


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l2751_275163

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l2751_275163


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2751_275127

/-- A regular polygon with side length 5 units and exterior angle 120 degrees has a perimeter of 15 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n ≥ 3 →
  side_length = 5 →
  exterior_angle = 120 →
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2751_275127


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2751_275113

def is_necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem quadratic_roots_condition :
  ∀ m n : ℝ,
  let roots := {x : ℝ | x^2 - m*x + n = 0}
  is_necessary_not_sufficient
    (m > 2 ∧ n > 1)
    (∀ x ∈ roots, x > 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2751_275113


namespace NUMINAMATH_CALUDE_road_trip_duration_l2751_275155

theorem road_trip_duration 
  (initial_duration : ℕ) 
  (stretch_interval : ℕ) 
  (food_stops : ℕ) 
  (gas_stops : ℕ) 
  (stop_duration : ℕ) 
  (h1 : initial_duration = 14)
  (h2 : stretch_interval = 2)
  (h3 : food_stops = 2)
  (h4 : gas_stops = 3)
  (h5 : stop_duration = 20) :
  initial_duration + 
  (initial_duration / stretch_interval + food_stops + gas_stops) * stop_duration / 60 = 18 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_duration_l2751_275155


namespace NUMINAMATH_CALUDE_rational_function_simplification_l2751_275112

theorem rational_function_simplification (x : ℝ) (h : x ≠ -1) :
  (x^3 + 4*x^2 + 5*x + 2) / (x + 1) = x^2 + 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_simplification_l2751_275112


namespace NUMINAMATH_CALUDE_simplify_expression_l2751_275178

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4) = |x - 2| + |x + 2| := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2751_275178


namespace NUMINAMATH_CALUDE_sequence_third_term_l2751_275102

theorem sequence_third_term (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n : ℕ, S n = 2 - 2^(n+1)) → 
  a 3 = -8 := by
sorry

end NUMINAMATH_CALUDE_sequence_third_term_l2751_275102


namespace NUMINAMATH_CALUDE_perpendicular_tangents_exist_and_unique_l2751_275188

/-- The line on which we search for the point. -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 2

/-- The parabola to which we find tangents. -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- A point is on a tangent line to the parabola. -/
def is_on_tangent (x y x₀ y₀ : ℝ) : Prop :=
  y = y₀ + 2 * x₀ * (x - x₀)

/-- Two lines are perpendicular. -/
def are_perpendicular (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1

/-- The theorem stating the existence and uniqueness of the point and its tangents. -/
theorem perpendicular_tangents_exist_and_unique :
  ∃! x₀ y₀ k₁ k₂,
    line x₀ y₀ ∧
    parabola x₀ y₀ ∧
    are_perpendicular (2 * x₀) (2 * x₀) ∧
    (∀ x y, is_on_tangent x y x₀ y₀ → (y = -1/4 + k₁ * (x - 1) ∨ y = -1/4 + k₂ * (x - 1))) ∧
    k₁ = 2 + Real.sqrt 5 ∧
    k₂ = 2 - Real.sqrt 5 :=
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_exist_and_unique_l2751_275188


namespace NUMINAMATH_CALUDE_sequence_general_term_l2751_275147

theorem sequence_general_term (a : ℕ+ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ+, a (n + 1) = (2 * a n) / (2 + a n)) →
  ∀ n : ℕ+, a n = 2 / (n + 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2751_275147


namespace NUMINAMATH_CALUDE_fourth_competitor_jump_distance_l2751_275179

theorem fourth_competitor_jump_distance (first_jump second_jump third_jump fourth_jump : ℕ) :
  first_jump = 22 ∧
  second_jump = first_jump + 1 ∧
  third_jump = second_jump - 2 ∧
  fourth_jump = third_jump + 3 →
  fourth_jump = 24 := by
  sorry

end NUMINAMATH_CALUDE_fourth_competitor_jump_distance_l2751_275179


namespace NUMINAMATH_CALUDE_bert_kangaroo_count_l2751_275167

/-- The number of kangaroos Kameron has -/
def kameron_kangaroos : ℕ := 100

/-- The number of kangaroos Bert buys per day -/
def bert_daily_increase : ℕ := 2

/-- The number of days until Bert has the same number of kangaroos as Kameron -/
def days_until_equal : ℕ := 40

/-- The number of kangaroos Bert currently has -/
def bert_current_kangaroos : ℕ := 20

theorem bert_kangaroo_count :
  bert_current_kangaroos + bert_daily_increase * days_until_equal = kameron_kangaroos :=
sorry

end NUMINAMATH_CALUDE_bert_kangaroo_count_l2751_275167


namespace NUMINAMATH_CALUDE_f_max_value_when_a_eq_one_unique_root_f_eq_g_l2751_275166

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x
def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (2*a + 1) * x + (a + 1) * Real.log x

-- Theorem for the maximum value of f when a = 1
theorem f_max_value_when_a_eq_one :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 y ≤ f 1 x ∧ f 1 x = -1 := by sorry

-- Theorem for the unique root of f(x) = g(x) when a ≥ 1
theorem unique_root_f_eq_g (a : ℝ) (h : a ≥ 1) :
  ∃! (x : ℝ), x > 0 ∧ f a x = g a x := by sorry

end

end NUMINAMATH_CALUDE_f_max_value_when_a_eq_one_unique_root_f_eq_g_l2751_275166


namespace NUMINAMATH_CALUDE_utensil_pack_composition_l2751_275153

/-- Represents a pack of utensils -/
structure UtensilPack where
  knives : ℕ
  forks : ℕ
  spoons : ℕ
  total : knives + forks + spoons = 30

/-- Theorem about the composition of utensil packs -/
theorem utensil_pack_composition 
  (pack : UtensilPack) 
  (h : 5 * pack.spoons = 50) : 
  pack.spoons = 10 ∧ pack.knives + pack.forks = 20 := by
  sorry


end NUMINAMATH_CALUDE_utensil_pack_composition_l2751_275153


namespace NUMINAMATH_CALUDE_triangle_properties_l2751_275151

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 2 * t.b) 
  (h2 : 2 * Real.sin t.A = 3 * Real.sin (2 * t.C)) 
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 7) / 2) : 
  t.a = (3 * Real.sqrt 2 / 2) * t.b ∧ 
  (t.c * ((3 * Real.sqrt 7) / 4)) / (2 * ((3 * Real.sqrt 7) / 2)) = (3 * Real.sqrt 7) / 4 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2751_275151


namespace NUMINAMATH_CALUDE_square_area_is_9_l2751_275133

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The square ABCD -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The square is entirely below the x-axis -/
def below_x_axis (s : Square) : Prop :=
  s.A.2 ≤ 0 ∧ s.B.2 ≤ 0 ∧ s.C.2 ≤ 0 ∧ s.D.2 ≤ 0

/-- The square is inscribed within the region bounded by the parabola and the x-axis -/
def inscribed_in_parabola (s : Square) : Prop :=
  s.A.2 = 0 ∧ s.B.2 = 0 ∧ s.C.2 = f s.C.1 ∧ s.D.2 = f s.D.1

/-- The top vertex A lies at (2, 0) -/
def top_vertex_at_2_0 (s : Square) : Prop :=
  s.A = (2, 0)

/-- The theorem stating that the area of the square is 9 -/
theorem square_area_is_9 (s : Square) 
    (h1 : below_x_axis s)
    (h2 : inscribed_in_parabola s)
    (h3 : top_vertex_at_2_0 s) : 
  (s.B.1 - s.A.1)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_9_l2751_275133


namespace NUMINAMATH_CALUDE_x_minus_y_equals_twenty_l2751_275171

theorem x_minus_y_equals_twenty (x y : ℝ) 
  (h1 : x * (y + 2) = 100) 
  (h2 : y * (x + 2) = 60) : 
  x - y = 20 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_twenty_l2751_275171


namespace NUMINAMATH_CALUDE_min_box_value_l2751_275140

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a*x + b)*(b*x + a) = 36*x^2 + Box*x + 36) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  Box = a^2 + b^2 →
  ∃ (min_Box : ℤ), (∀ Box', (∃ a' b' : ℤ, 
    (∀ x, (a'*x + b')*(b'*x + a') = 36*x^2 + Box'*x + 36) ∧
    a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
    Box' = a'^2 + b'^2) → 
    min_Box ≤ Box') ∧
  min_Box = 72 :=
sorry

end NUMINAMATH_CALUDE_min_box_value_l2751_275140


namespace NUMINAMATH_CALUDE_olivia_calculation_l2751_275152

def round_to_nearest_ten (n : Int) : Int :=
  10 * ((n + 5) / 10)

theorem olivia_calculation : round_to_nearest_ten ((57 + 68) - 15) = 110 := by
  sorry

end NUMINAMATH_CALUDE_olivia_calculation_l2751_275152


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2751_275175

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x | x - 1 ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2751_275175


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l2751_275198

/-- Proves that 20% of motorists who exceed the speed limit do not receive tickets -/
theorem speeding_ticket_percentage (M : ℝ) (h1 : M > 0) : 
  let exceed_limit := 0.125 * M
  let receive_ticket := 0.1 * M
  (exceed_limit - receive_ticket) / exceed_limit = 0.2 := by
sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l2751_275198


namespace NUMINAMATH_CALUDE_tractors_moved_l2751_275144

/-- Represents the farming field scenario -/
structure FarmingField where
  initialTractors : ℕ
  initialDays : ℕ
  initialHectaresPerDay : ℕ
  remainingTractors : ℕ
  remainingDays : ℕ
  remainingHectaresPerDay : ℕ

/-- The theorem stating the number of tractors moved -/
theorem tractors_moved (field : FarmingField)
  (h1 : field.initialTractors = 6)
  (h2 : field.initialDays = 4)
  (h3 : field.initialHectaresPerDay = 120)
  (h4 : field.remainingTractors = 4)
  (h5 : field.remainingDays = 5)
  (h6 : field.remainingHectaresPerDay = 144)
  (h7 : field.initialTractors * field.initialDays * field.initialHectaresPerDay =
        field.remainingTractors * field.remainingDays * field.remainingHectaresPerDay) :
  field.initialTractors - field.remainingTractors = 2 := by
  sorry

#check tractors_moved

end NUMINAMATH_CALUDE_tractors_moved_l2751_275144


namespace NUMINAMATH_CALUDE_domain_of_f_l2751_275173

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 3)) / (x^2 + 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 < x ∧ x ≠ -1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2751_275173


namespace NUMINAMATH_CALUDE_volume_of_five_adjacent_cubes_l2751_275106

/-- The volume of a solid formed by placing n equal cubes with side length s adjacent to each other -/
def volume_of_adjacent_cubes (n : ℕ) (s : ℝ) : ℝ := n * s^3

/-- Theorem: The volume of a solid formed by placing five equal cubes with side length 5 cm adjacent to each other is 625 cm³ -/
theorem volume_of_five_adjacent_cubes :
  volume_of_adjacent_cubes 5 5 = 625 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_five_adjacent_cubes_l2751_275106


namespace NUMINAMATH_CALUDE_helmet_sales_theorem_l2751_275105

/-- Represents the monthly sales data and pricing information for helmets --/
structure HelmetSalesData where
  april_sales : ℕ
  june_sales : ℕ
  cost_price : ℕ
  reference_price : ℕ
  reference_volume : ℕ
  volume_change_rate : ℕ
  target_profit : ℕ

/-- Calculates the monthly growth rate given the sales data --/
def calculate_growth_rate (data : HelmetSalesData) : ℚ :=
  sorry

/-- Calculates the optimal selling price given the sales data --/
def calculate_optimal_price (data : HelmetSalesData) : ℕ :=
  sorry

/-- Theorem stating the correct growth rate and optimal price --/
theorem helmet_sales_theorem (data : HelmetSalesData) 
  (h1 : data.april_sales = 150)
  (h2 : data.june_sales = 216)
  (h3 : data.cost_price = 30)
  (h4 : data.reference_price = 40)
  (h5 : data.reference_volume = 600)
  (h6 : data.volume_change_rate = 10)
  (h7 : data.target_profit = 10000) :
  calculate_growth_rate data = 1/5 ∧ calculate_optimal_price data = 50 := by
  sorry

end NUMINAMATH_CALUDE_helmet_sales_theorem_l2751_275105


namespace NUMINAMATH_CALUDE_complex_first_quadrant_a_range_l2751_275186

theorem complex_first_quadrant_a_range (a : ℝ) :
  let z : ℂ := Complex.mk a (1 - a)
  (0 < z.re ∧ 0 < z.im) → (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_first_quadrant_a_range_l2751_275186


namespace NUMINAMATH_CALUDE_inequality_proof_l2751_275139

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b^2 + b / c^2 + c / a^2 ≥ 1 / a + 1 / b + 1 / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2751_275139


namespace NUMINAMATH_CALUDE_tower_levels_l2751_275162

theorem tower_levels (steps_per_level : ℕ) (blocks_per_step : ℕ) (total_blocks : ℕ) :
  steps_per_level = 8 →
  blocks_per_step = 3 →
  total_blocks = 96 →
  total_blocks / (steps_per_level * blocks_per_step) = 4 :=
by sorry

end NUMINAMATH_CALUDE_tower_levels_l2751_275162


namespace NUMINAMATH_CALUDE_four_boxes_volume_l2751_275130

/-- The volume of a cube with edge length s -/
def cube_volume (s : ℝ) : ℝ := s ^ 3

/-- The total volume of n identical cubes with edge length s -/
def total_volume (n : ℕ) (s : ℝ) : ℝ := n * cube_volume s

/-- Theorem: The total volume of four cubic boxes, each with an edge length of 5 meters, is 500 cubic meters -/
theorem four_boxes_volume : total_volume 4 5 = 500 := by
  sorry

end NUMINAMATH_CALUDE_four_boxes_volume_l2751_275130


namespace NUMINAMATH_CALUDE_constant_r_is_circle_l2751_275145

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- A circle centered at the origin -/
def Circle (radius : ℝ) := {p : PolarPoint | p.r = radius}

/-- The set of points satisfying r = 5 in polar coordinates -/
def ConstantR : Set PolarPoint := {p : PolarPoint | p.r = 5}

/-- Theorem stating that ConstantR is a circle with radius 5 -/
theorem constant_r_is_circle : ConstantR = Circle 5 := by sorry

end NUMINAMATH_CALUDE_constant_r_is_circle_l2751_275145


namespace NUMINAMATH_CALUDE_microphotonics_budget_percentage_l2751_275185

theorem microphotonics_budget_percentage 
  (total_degrees : ℝ)
  (home_electronics : ℝ)
  (food_additives : ℝ)
  (genetically_modified_microorganisms : ℝ)
  (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : home_electronics = 24)
  (h3 : food_additives = 15)
  (h4 : genetically_modified_microorganisms = 29)
  (h5 : industrial_lubricants = 8)
  (h6 : basic_astrophysics_degrees = 43.2) : 
  (100 - (home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + (basic_astrophysics_degrees / total_degrees * 100))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_microphotonics_budget_percentage_l2751_275185


namespace NUMINAMATH_CALUDE_sum_a_b_equals_five_l2751_275142

theorem sum_a_b_equals_five (a b : ℝ) (h1 : a + 2*b = 8) (h2 : 3*a + 4*b = 18) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_five_l2751_275142


namespace NUMINAMATH_CALUDE_solution_implies_a_equals_one_l2751_275193

def f (x a : ℝ) : ℝ := |x - a| - 2

theorem solution_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, |f x a| < 1 ↔ (x ∈ Set.Ioo (-2) 0 ∨ x ∈ Set.Ioo 2 4)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_a_equals_one_l2751_275193


namespace NUMINAMATH_CALUDE_fifth_group_students_l2751_275160

theorem fifth_group_students (total : Nat) (group1 group2 group3 group4 : Nat)
  (h1 : total = 40)
  (h2 : group1 = 6)
  (h3 : group2 = 9)
  (h4 : group3 = 8)
  (h5 : group4 = 7) :
  total - (group1 + group2 + group3 + group4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fifth_group_students_l2751_275160


namespace NUMINAMATH_CALUDE_gear_rotation_problem_l2751_275174

/-- The number of revolutions per minute of gear q -/
def q_rpm : ℝ := 40

/-- The time elapsed in minutes -/
def time : ℝ := 1.5

/-- The difference in revolutions between gears q and p after 90 seconds -/
def rev_diff : ℝ := 45

/-- The number of revolutions per minute of gear p -/
def p_rpm : ℝ := 10

theorem gear_rotation_problem :
  p_rpm * time + rev_diff = q_rpm * time := by sorry

end NUMINAMATH_CALUDE_gear_rotation_problem_l2751_275174


namespace NUMINAMATH_CALUDE_sin_cos_relation_l2751_275146

theorem sin_cos_relation (α : Real) (h : Real.sin (π / 3 + α) = 1 / 3) :
  Real.cos (α - 7 * π / 6) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l2751_275146


namespace NUMINAMATH_CALUDE_common_solution_for_all_a_l2751_275135

/-- The linear equation (a-3)x + (2a-5)y + 6-a = 0 has a common solution (7, -3) for all values of a. -/
theorem common_solution_for_all_a :
  ∀ (a : ℝ), (a - 3) * 7 + (2 * a - 5) * (-3) + 6 - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_common_solution_for_all_a_l2751_275135


namespace NUMINAMATH_CALUDE_tangent_line_minimum_sum_l2751_275103

/-- Given a circle and a line that are tangent, prove the minimum value of a + b -/
theorem tangent_line_minimum_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (a - 1) * x + (b - 1) * y + a + b = 0) →
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (a - 1) * x + (b - 1) * y + a + b = 0) →
  a + b ≥ 2 * Real.sqrt 2 - 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_sum_l2751_275103


namespace NUMINAMATH_CALUDE_workshop_average_salary_l2751_275156

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (technician_salary : ℕ) 
  (other_salary : ℕ) 
  (h1 : total_workers = 18) 
  (h2 : num_technicians = 6) 
  (h3 : technician_salary = 12000) 
  (h4 : other_salary = 6000) :
  (num_technicians * technician_salary + (total_workers - num_technicians) * other_salary) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l2751_275156


namespace NUMINAMATH_CALUDE_luck_represents_6789_l2751_275100

/-- Represents a mapping from characters to digits -/
def DigitMapping := Char → Nat

/-- The 12-letter code -/
def code : String := "AMAZING LUCK"

/-- The condition that the code represents digits 0-9 and repeats for two more digits -/
def valid_mapping (m : DigitMapping) : Prop :=
  ∀ i : Fin 12, 
    m (code.get ⟨i⟩) = if i < 10 then i else i - 10

/-- The substring we're interested in -/
def substring : String := "LUCK"

/-- The theorem to prove -/
theorem luck_represents_6789 (m : DigitMapping) (h : valid_mapping m) : 
  (m 'L', m 'U', m 'C', m 'K') = (6, 7, 8, 9) := by
  sorry

end NUMINAMATH_CALUDE_luck_represents_6789_l2751_275100


namespace NUMINAMATH_CALUDE_evaluate_expression_l2751_275137

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 0) : z * (2 * z - 5 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2751_275137


namespace NUMINAMATH_CALUDE_garrison_provisions_l2751_275190

theorem garrison_provisions (initial_men : ℕ) (initial_days : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) : 
  initial_men = 2000 →
  initial_days = 54 →
  reinforcement = 1600 →
  days_before_reinforcement = 18 →
  let total_provisions := initial_men * initial_days
  let used_provisions := initial_men * days_before_reinforcement
  let remaining_provisions := total_provisions - used_provisions
  let total_men_after_reinforcement := initial_men + reinforcement
  (remaining_provisions / total_men_after_reinforcement : ℚ) = 20 := by
sorry

end NUMINAMATH_CALUDE_garrison_provisions_l2751_275190


namespace NUMINAMATH_CALUDE_tenth_power_sum_of_roots_l2751_275143

theorem tenth_power_sum_of_roots (r s : ℂ) : 
  (r^2 - 2*r + 4 = 0) → (s^2 - 2*s + 4 = 0) → r^10 + s^10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_tenth_power_sum_of_roots_l2751_275143


namespace NUMINAMATH_CALUDE_total_chickens_on_farm_l2751_275159

/-- Proves that the total number of chickens on a farm is 120, given the number of hens and their relation to roosters. -/
theorem total_chickens_on_farm (num_hens : ℕ) (num_roosters : ℕ) : 
  num_hens = 52 → 
  num_hens + 16 = num_roosters → 
  num_hens + num_roosters = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_chickens_on_farm_l2751_275159


namespace NUMINAMATH_CALUDE_percentage_less_relation_l2751_275111

/-- Given three real numbers A, B, and C, where A is 35% less than C,
    and B is 10.76923076923077% less than A, prove that B is
    approximately 42% less than C. -/
theorem percentage_less_relation (A B C : ℝ) 
  (h1 : A = 0.65 * C)  -- A is 35% less than C
  (h2 : B = 0.8923076923076923 * A)  -- B is 10.76923076923077% less than A
  : ∃ (ε : ℝ), abs (B - 0.58 * C) < ε ∧ ε < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_relation_l2751_275111


namespace NUMINAMATH_CALUDE_class_artworks_count_l2751_275138

/-- Represents the number of artworks created by a group of students -/
structure Artworks :=
  (paintings : ℕ)
  (drawings : ℕ)
  (sculptures : ℕ)

/-- Calculates the total number of artworks -/
def total_artworks (a : Artworks) : ℕ :=
  a.paintings + a.drawings + a.sculptures

theorem class_artworks_count :
  let total_students : ℕ := 36
  let group1_students : ℕ := 24
  let group2_students : ℕ := 12
  let total_kits : ℕ := 48
  let group1_sharing_ratio : ℕ := 3  -- 1 kit per 3 students
  let group2_sharing_ratio : ℕ := 2  -- 1 kit per 2 students
  
  let group1_first_half : Artworks := ⟨2, 4, 1⟩
  let group1_second_half : Artworks := ⟨1, 5, 3⟩
  let group2_first_third : Artworks := ⟨3, 6, 3⟩
  let group2_second_third : Artworks := ⟨4, 7, 1⟩
  
  let group1_artworks : Artworks := ⟨
    12 * group1_first_half.paintings + 12 * group1_second_half.paintings,
    12 * group1_first_half.drawings + 12 * group1_second_half.drawings,
    12 * group1_first_half.sculptures + 12 * group1_second_half.sculptures
  ⟩
  
  let group2_artworks : Artworks := ⟨
    4 * group2_first_third.paintings + 8 * group2_second_third.paintings,
    4 * group2_first_third.drawings + 8 * group2_second_third.drawings,
    4 * group2_first_third.sculptures + 8 * group2_second_third.sculptures
  ⟩
  
  let total_class_artworks : Artworks := ⟨
    group1_artworks.paintings + group2_artworks.paintings,
    group1_artworks.drawings + group2_artworks.drawings,
    group1_artworks.sculptures + group2_artworks.sculptures
  ⟩
  
  total_artworks total_class_artworks = 336 := by sorry

end NUMINAMATH_CALUDE_class_artworks_count_l2751_275138


namespace NUMINAMATH_CALUDE_range_of_x_minus_2y_range_of_2a_plus_3b_l2751_275199

-- Problem 1
theorem range_of_x_minus_2y (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) :
  ∃ (z : ℝ), -3 ≤ z ∧ z ≤ 2 ∧ ∃ (x' y' : ℝ), 
    (-1 ≤ x' ∧ x' ≤ 2) ∧ (0 ≤ y' ∧ y' ≤ 1) ∧ z = x' - 2 * y' :=
sorry

-- Problem 2
theorem range_of_2a_plus_3b (a b : ℝ) (hab1 : -1 < a + b ∧ a + b < 3) (hab2 : 2 < a - b ∧ a - b < 4) :
  ∃ (z : ℝ), -9/2 < z ∧ z < 13/2 ∧ ∃ (a' b' : ℝ), 
    (-1 < a' + b' ∧ a' + b' < 3) ∧ (2 < a' - b' ∧ a' - b' < 4) ∧ z = 2 * a' + 3 * b' :=
sorry

end NUMINAMATH_CALUDE_range_of_x_minus_2y_range_of_2a_plus_3b_l2751_275199


namespace NUMINAMATH_CALUDE_odd_swaps_change_perm_l2751_275176

/-- Represents a permutation of three elements -/
inductive Perm3
  | abc
  | acb
  | bac
  | bca
  | cab
  | cba

/-- Represents whether a permutation is "correct" or "incorrect" -/
def isCorrect (p : Perm3) : Bool :=
  match p with
  | Perm3.abc => true
  | Perm3.bca => true
  | Perm3.cab => true
  | _ => false

/-- Represents a single adjacent swap -/
def swap (p : Perm3) : Perm3 :=
  match p with
  | Perm3.abc => Perm3.acb
  | Perm3.acb => Perm3.abc
  | Perm3.bac => Perm3.bca
  | Perm3.bca => Perm3.bac
  | Perm3.cab => Perm3.cba
  | Perm3.cba => Perm3.cab

/-- Theorem: After an odd number of swaps, the permutation cannot be the same as the initial one -/
theorem odd_swaps_change_perm (n : Nat) (h : Odd n) (p : Perm3) :
  (n.iterate swap p) ≠ p :=
  sorry

#check odd_swaps_change_perm

end NUMINAMATH_CALUDE_odd_swaps_change_perm_l2751_275176


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2751_275187

theorem perfect_square_condition (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  let n : Nat := ((p - 1) / 2) ^ 2
  ∃ (k : Nat), n * p + n^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2751_275187


namespace NUMINAMATH_CALUDE_trig_inequality_l2751_275121

theorem trig_inequality : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l2751_275121


namespace NUMINAMATH_CALUDE_work_completion_time_l2751_275122

/-- Represents the work rate of a group of workers -/
def WorkRate (num_workers : ℕ) (days : ℕ) : ℚ :=
  1 / (num_workers * days)

/-- The theorem statement -/
theorem work_completion_time 
  (men_rate : WorkRate 8 20 = WorkRate 12 20) 
  (total_work : ℚ := 1) :
  (6 : ℚ) * WorkRate 8 20 + (11 : ℚ) * WorkRate 12 20 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2751_275122


namespace NUMINAMATH_CALUDE_postage_fee_420g_l2751_275126

/-- Calculates the postage fee for a given weight in grams -/
def postage_fee (weight : ℕ) : ℚ :=
  0.7 + 0.4 * ((weight - 1) / 100 : ℕ)

/-- The postage fee for a 420g book is 2.3 yuan -/
theorem postage_fee_420g : postage_fee 420 = 2.3 := by sorry

end NUMINAMATH_CALUDE_postage_fee_420g_l2751_275126


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2751_275194

-- Define the function g(x)
noncomputable def g : ℝ → ℤ
| x => if x > -1 then Int.ceil (1 / (x + 1))
       else if x < -1 then Int.floor (1 / (x + 1))
       else 0  -- arbitrary value for x = -1, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2751_275194


namespace NUMINAMATH_CALUDE_smallest_prime_longest_sequence_l2751_275125

def A₁₁ : ℕ := 30

def is_prime_sequence (p : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → Nat.Prime (p + k * A₁₁)

theorem smallest_prime_longest_sequence :
  ∃ n : ℕ, 
    Nat.Prime 7 ∧ 
    is_prime_sequence 7 n ∧
    ∀ p < 7, Nat.Prime p → ∀ m : ℕ, is_prime_sequence p m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_longest_sequence_l2751_275125


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2751_275161

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2751_275161


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2751_275134

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  1/a + 2/b + 9/c + 8/d + 18/e + 32/f ≥ 24 ∧ 
  ∃ (a' b' c' d' e' f' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 9 ∧
    1/a' + 2/b' + 9/c' + 8/d' + 18/e' + 32/f' = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2751_275134


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2751_275191

theorem geometric_sequence_property (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_product : a 1 * a 7 = 36) : 
  a 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2751_275191


namespace NUMINAMATH_CALUDE_sample_size_determination_l2751_275131

theorem sample_size_determination (total_population : Nat) (n : Nat) : 
  total_population = 36 →
  n > 0 →
  total_population % n = 0 →
  (total_population / n) % 6 = 0 →
  35 % (n + 1) = 0 →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_determination_l2751_275131


namespace NUMINAMATH_CALUDE_journey_equations_l2751_275119

theorem journey_equations (total_time bike_speed walk_speed total_distance : ℝ)
  (h_total_time : total_time = 20)
  (h_bike_speed : bike_speed = 200)
  (h_walk_speed : walk_speed = 70)
  (h_total_distance : total_distance = 3350) :
  ∃ x y : ℝ,
    x + y = total_time ∧
    bike_speed * x + walk_speed * y = total_distance :=
by sorry

end NUMINAMATH_CALUDE_journey_equations_l2751_275119


namespace NUMINAMATH_CALUDE_train_speeds_equal_l2751_275168

-- Define the speeds and times
def speed_A : ℝ := 110
def time_A_after_meeting : ℝ := 9
def time_B_after_meeting : ℝ := 4

-- Define the theorem
theorem train_speeds_equal :
  ∀ (speed_B : ℝ) (time_before_meeting : ℝ),
    speed_B > 0 →
    time_before_meeting > 0 →
    speed_A * time_before_meeting + speed_A * time_A_after_meeting =
    speed_B * time_before_meeting + speed_B * time_B_after_meeting →
    speed_A * time_before_meeting = speed_B * time_before_meeting →
    speed_B = speed_A :=
by
  sorry

#check train_speeds_equal

end NUMINAMATH_CALUDE_train_speeds_equal_l2751_275168


namespace NUMINAMATH_CALUDE_function_identity_l2751_275141

def is_strictly_increasing (f : ℕ+ → ℤ) : Prop :=
  ∀ m n : ℕ+, m > n → f m > f n

theorem function_identity (f : ℕ+ → ℤ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ+, f (m * n) = f m * f n)
  (h3 : is_strictly_increasing f) :
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2751_275141


namespace NUMINAMATH_CALUDE_doughnuts_distribution_l2751_275196

theorem doughnuts_distribution (total_doughnuts : ℕ) (total_boxes : ℕ) (first_two_boxes : ℕ) (doughnuts_per_first_two : ℕ) :
  total_doughnuts = 72 →
  total_boxes = 6 →
  first_two_boxes = 2 →
  doughnuts_per_first_two = 12 →
  (total_doughnuts - first_two_boxes * doughnuts_per_first_two) % (total_boxes - first_two_boxes) = 0 →
  (total_doughnuts - first_two_boxes * doughnuts_per_first_two) / (total_boxes - first_two_boxes) = 12 :=
by sorry

end NUMINAMATH_CALUDE_doughnuts_distribution_l2751_275196


namespace NUMINAMATH_CALUDE_vector_operation_l2751_275129

/-- Given two plane vectors a and b, prove that -2a - b equals the specified result. -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (1, -1)) :
  (-2 : ℝ) • a - b = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l2751_275129


namespace NUMINAMATH_CALUDE_fraction_division_and_addition_l2751_275169

theorem fraction_division_and_addition : 
  (5 / 6 : ℚ) / (9 / 10 : ℚ) + 1 / 15 = 402 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_and_addition_l2751_275169


namespace NUMINAMATH_CALUDE_max_coincident_area_folded_triangle_l2751_275101

theorem max_coincident_area_folded_triangle :
  let a := 3 / 2
  let b := Real.sqrt 5 / 2
  let c := Real.sqrt 2
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let height := 2 * area / a
  let max_coincident_area := area + (1 / (2 * height)) - (1 / (4 * height^2)) - (3 / (4 * height^2))
  max_coincident_area = 9 / 28 := by sorry

end NUMINAMATH_CALUDE_max_coincident_area_folded_triangle_l2751_275101


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2751_275197

theorem divisibility_equivalence (m n : ℕ+) : 
  (19 ∣ (11 * m.val + 2 * n.val)) ↔ (19 ∣ (18 * m.val + 5 * n.val)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2751_275197


namespace NUMINAMATH_CALUDE_box_surface_area_l2751_275192

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle --/
def rectangleArea (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the cardboard sheet with cut corners --/
structure CutCardboard where
  sheet : Rectangle
  smallCutSize : ℕ
  largeCutSize : ℕ

/-- Calculates the surface area of the interior of the box formed from the cut cardboard --/
def interiorSurfaceArea (c : CutCardboard) : ℕ :=
  rectangleArea c.sheet -
  (2 * rectangleArea ⟨c.smallCutSize, c.smallCutSize⟩) -
  (2 * rectangleArea ⟨c.largeCutSize, c.largeCutSize⟩)

theorem box_surface_area :
  let cardboard : CutCardboard := {
    sheet := { length := 35, width := 25 },
    smallCutSize := 3,
    largeCutSize := 4
  }
  interiorSurfaceArea cardboard = 825 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_l2751_275192


namespace NUMINAMATH_CALUDE_complex_fraction_magnitude_l2751_275154

/-- Given that i is the imaginary unit, prove that |((5+3i)/(4-i))| = √2 -/
theorem complex_fraction_magnitude : 
  Complex.abs ((5 + 3 * Complex.I) / (4 - Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_magnitude_l2751_275154


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2751_275150

theorem rectangular_plot_length_difference (b x : ℝ) : 
  b + x = 64 →                         -- length is 64 meters
  26.5 * (2 * (b + x) + 2 * b) = 5300 →  -- cost of fencing
  x = 28 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2751_275150


namespace NUMINAMATH_CALUDE_negation_equivalence_l2751_275118

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (planet : U → Prop)
variable (orbits_sun : U → Prop)

-- Define the original statement
def every_planet_orbits_sun : Prop := ∀ x, planet x → orbits_sun x

-- Define the negation we want to prove
def some_planets_dont_orbit_sun : Prop := ∃ x, planet x ∧ ¬(orbits_sun x)

-- Theorem statement
theorem negation_equivalence : 
  ¬(every_planet_orbits_sun U planet orbits_sun) ↔ some_planets_dont_orbit_sun U planet orbits_sun :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2751_275118


namespace NUMINAMATH_CALUDE_roots_sum_properties_l2751_275158

theorem roots_sum_properties (a : ℤ) (x₁ x₂ : ℝ) (h_odd : Odd a) (h_roots : x₁^2 + a*x₁ - 1 = 0 ∧ x₂^2 + a*x₂ - 1 = 0) :
  ∀ n : ℕ, 
    (∃ k : ℤ, x₁^n + x₂^n = k) ∧ 
    (∃ m : ℤ, x₁^(n+1) + x₂^(n+1) = m) ∧ 
    (Int.gcd (↑⌊x₁^n + x₂^n⌋) (↑⌊x₁^(n+1) + x₂^(n+1)⌋) = 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_properties_l2751_275158


namespace NUMINAMATH_CALUDE_at_least_one_prime_between_nfact_minus_n_and_nfact_l2751_275189

theorem at_least_one_prime_between_nfact_minus_n_and_nfact (n : ℕ) (h : n > 2) :
  ∃ p : ℕ, Prime p ∧ n! - n < p ∧ p < n! :=
sorry

end NUMINAMATH_CALUDE_at_least_one_prime_between_nfact_minus_n_and_nfact_l2751_275189


namespace NUMINAMATH_CALUDE_no_solution_cosine_sine_equation_l2751_275116

theorem no_solution_cosine_sine_equation :
  ∀ x : ℝ, Real.cos (Real.cos (Real.cos (Real.cos x))) > Real.sin (Real.sin (Real.sin (Real.sin x))) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_cosine_sine_equation_l2751_275116


namespace NUMINAMATH_CALUDE_min_value_of_f_in_interval_l2751_275114

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the interval [-4, 4]
def interval : Set ℝ := Set.Icc (-4) 4

-- Theorem statement
theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -16 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_in_interval_l2751_275114


namespace NUMINAMATH_CALUDE_equation_solution_range_l2751_275109

theorem equation_solution_range : 
  {k : ℝ | ∃ x : ℝ, 2*k*(Real.sin x) = 1 + k^2} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2751_275109
