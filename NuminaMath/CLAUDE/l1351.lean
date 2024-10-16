import Mathlib

namespace NUMINAMATH_CALUDE_shifted_sine_value_l1351_135195

theorem shifted_sine_value (g f : ℝ → ℝ) :
  (∀ x, g x = Real.sin (x - π/6)) →
  (∀ x, f x = g (x - π/6)) →
  f (π/6) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_shifted_sine_value_l1351_135195


namespace NUMINAMATH_CALUDE_sum_of_z_values_l1351_135110

theorem sum_of_z_values (f : ℝ → ℝ) (h : ∀ x, f (x / 3) = x^2 + x + 1) :
  let z₁ := (2 : ℝ) / 9
  let z₂ := -(1 : ℝ) / 3
  (f (3 * z₁) = 7 ∧ f (3 * z₂) = 7) ∧ z₁ + z₂ = -(1 : ℝ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_z_values_l1351_135110


namespace NUMINAMATH_CALUDE_unique_parallel_line_in_plane_l1351_135101

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  parallel_line_plane : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop
  parallel_lines : Line → Line → Prop

/-- The theorem statement -/
theorem unique_parallel_line_in_plane 
  (S : Space3D) (l : S.Line) (α : S.Plane) : 
  (¬ S.parallel_line_plane l α) → 
  (¬ S.line_in_plane l α) → 
  ∃! m : S.Line, S.line_in_plane m α ∧ S.parallel_lines m l :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_in_plane_l1351_135101


namespace NUMINAMATH_CALUDE_square_side_length_l1351_135147

-- Define the circumference of the largest inscribed circle
def circle_circumference : ℝ := 37.69911184307752

-- Define π as a constant (approximation)
def π : ℝ := 3.141592653589793

-- Theorem statement
theorem square_side_length (circle_circumference : ℝ) (π : ℝ) :
  let radius := circle_circumference / (2 * π)
  let diameter := 2 * radius
  diameter = 12 := by sorry

end NUMINAMATH_CALUDE_square_side_length_l1351_135147


namespace NUMINAMATH_CALUDE_range_of_x_minus_cosy_l1351_135184

theorem range_of_x_minus_cosy (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) :
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_minus_cosy_l1351_135184


namespace NUMINAMATH_CALUDE_marks_used_days_ratio_l1351_135107

def total_allotted_days : ℕ := 20
def hours_per_day : ℕ := 8
def unused_hours : ℕ := 80

theorem marks_used_days_ratio :
  let unused_days : ℕ := unused_hours / hours_per_day
  let used_days : ℕ := total_allotted_days - unused_days
  (used_days : ℚ) / total_allotted_days = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_marks_used_days_ratio_l1351_135107


namespace NUMINAMATH_CALUDE_part_one_part_two_l1351_135183

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Define the complement of B in ℝ
def CompB (a : ℝ) : Set ℝ := {x | x ≤ a - 1 ∨ x ≥ a + 1}

-- Part I
theorem part_one : A ∩ (CompB 2) = {x | x ≥ 3} := by sorry

-- Part II
theorem part_two : ∀ a : ℝ, B a ⊆ A → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1351_135183


namespace NUMINAMATH_CALUDE_smartphone_savings_theorem_l1351_135151

/-- The amount saved per month in yuan -/
def monthly_savings : ℕ := 530

/-- The cost of the smartphone in yuan -/
def smartphone_cost : ℕ := 2000

/-- The number of months required to save for the smartphone -/
def months_required : ℕ := 4

theorem smartphone_savings_theorem : 
  monthly_savings * months_required ≥ smartphone_cost :=
sorry

end NUMINAMATH_CALUDE_smartphone_savings_theorem_l1351_135151


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1351_135134

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x - y - a + 3 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 4 = 0

-- Theorem stating that the line intersects the circle for any real a
theorem line_intersects_circle (a : ℝ) : 
  ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1351_135134


namespace NUMINAMATH_CALUDE_anna_initial_ham_slices_l1351_135159

/-- The number of slices of ham Anna puts in each sandwich. -/
def slices_per_sandwich : ℕ := 3

/-- The number of sandwiches Anna wants to make. -/
def total_sandwiches : ℕ := 50

/-- The additional number of ham slices Anna needs. -/
def additional_slices : ℕ := 119

/-- The initial number of ham slices Anna has. -/
def initial_slices : ℕ := total_sandwiches * slices_per_sandwich - additional_slices

theorem anna_initial_ham_slices :
  initial_slices = 31 := by sorry

end NUMINAMATH_CALUDE_anna_initial_ham_slices_l1351_135159


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l1351_135103

def watch_problem (selling_price_loss : ℝ) (selling_price_profit : ℝ) (profit_percentage : ℝ) : Prop :=
  let cost_price := selling_price_profit / (1 + profit_percentage / 100)
  let loss := cost_price - selling_price_loss
  let loss_percentage := (loss / cost_price) * 100
  selling_price_loss < cost_price ∧ 
  selling_price_profit > cost_price ∧
  loss_percentage = 5

theorem watch_loss_percentage : 
  watch_problem 1140 1260 5 := by sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_l1351_135103


namespace NUMINAMATH_CALUDE_number_problem_l1351_135174

theorem number_problem (x : ℝ) : 50 + 5 * 12 / (x / 3) = 51 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1351_135174


namespace NUMINAMATH_CALUDE_airplane_seats_l1351_135193

theorem airplane_seats : ∃ (s : ℝ), 
  (30 : ℝ) + 0.2 * s + 0.75 * s = s ∧ s = 600 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l1351_135193


namespace NUMINAMATH_CALUDE_total_students_correct_l1351_135144

/-- The total number of students who appeared for the examination -/
def total_students : ℕ := 840

/-- The percentage of students who passed the examination -/
def pass_percentage : ℚ := 35 / 100

/-- The number of students who failed the examination -/
def failed_students : ℕ := 546

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct : 
  (1 - pass_percentage) * total_students = failed_students := by sorry

end NUMINAMATH_CALUDE_total_students_correct_l1351_135144


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1351_135194

theorem max_product_under_constraint (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (constraint1 : x * y * z + w = (x + w) * (y + w) * (z + w))
  (constraint2 : x + y + z + w = 1) : 
  x * y * z * w ≤ 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1351_135194


namespace NUMINAMATH_CALUDE_track_length_proof_l1351_135133

/-- The length of the circular track -/
def track_length : ℝ := 330

/-- The distance Pamela runs before the first meeting -/
def pamela_first_meeting : ℝ := 120

/-- The additional distance Jane runs between the first and second meeting -/
def jane_additional : ℝ := 210

/-- Proves that the track length is correct given the meeting conditions -/
theorem track_length_proof :
  ∃ (pamela_speed jane_speed : ℝ),
    pamela_speed > 0 ∧ jane_speed > 0 ∧
    pamela_first_meeting / (track_length - pamela_first_meeting) = pamela_speed / jane_speed ∧
    (track_length - pamela_first_meeting + jane_additional) / (pamela_first_meeting + track_length - jane_additional) = jane_speed / pamela_speed :=
by sorry


end NUMINAMATH_CALUDE_track_length_proof_l1351_135133


namespace NUMINAMATH_CALUDE_convex_set_enclosure_l1351_135119

-- Define a convex set in 2D space
variable (Φ : Set (ℝ × ℝ))

-- Define the property of being convex
def IsConvex (S : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of being centrally symmetric
def IsCentrallySymmetric (S : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of one set enclosing another
def Encloses (S T : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a set
noncomputable def Area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a triangle
def IsTriangle (S : Set (ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem convex_set_enclosure (h : IsConvex Φ) : 
  ∃ S : Set (ℝ × ℝ), 
    IsConvex S ∧ 
    IsCentrallySymmetric S ∧ 
    Encloses S Φ ∧ 
    Area S ≤ 2 * Area Φ ∧
    (IsTriangle Φ → Area S ≥ 2 * Area Φ) := by
  sorry

end NUMINAMATH_CALUDE_convex_set_enclosure_l1351_135119


namespace NUMINAMATH_CALUDE_equation_solution_l1351_135155

theorem equation_solution : 
  ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1351_135155


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1351_135122

theorem arithmetic_mean_problem (x : ℝ) :
  (x + 3*x + 1000 + 3000) / 4 = 2018 ↔ x = 1018 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1351_135122


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1351_135181

theorem trigonometric_expression_equality : 
  (Real.sin (330 * π / 180) * Real.tan (-13 * π / 3)) / 
  (Real.cos (-19 * π / 6) * Real.cos (690 * π / 180)) = 
  -2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1351_135181


namespace NUMINAMATH_CALUDE_simplify_power_expression_l1351_135162

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l1351_135162


namespace NUMINAMATH_CALUDE_reciprocal_and_opposite_sum_l1351_135114

theorem reciprocal_and_opposite_sum (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposite numbers
  : 3 * a * b + 2 * c + 2 * d = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_opposite_sum_l1351_135114


namespace NUMINAMATH_CALUDE_parabola_focus_and_tangent_point_l1351_135199

noncomputable section

/-- Parabola C with parameter p -/
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

/-- Line passing through the focus -/
def focus_line (x y : ℝ) : Prop := x + 2*y - 2 = 0

/-- Directrix of the parabola -/
def directrix (p : ℝ) (y : ℝ) : Prop := y = -p/2

/-- Tangent line to the parabola from point (m, -p/2) -/
def tangent_line (p m k x y : ℝ) : Prop := y = -p/2 + k*(x - m)

/-- Area of triangle AMN -/
def triangle_area (m : ℝ) : ℝ := (1/2) * Real.sqrt (m^2 + 4)

theorem parabola_focus_and_tangent_point (p : ℝ) :
  p > 0 →
  (∃ x y : ℝ, parabola p x y ∧ focus_line x y) →
  (∃ m : ℝ, directrix p (-p/2) ∧ triangle_area m = Real.sqrt 5 / 2) →
  (∃ m : ℝ, m = 1 ∨ m = -1) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_and_tangent_point_l1351_135199


namespace NUMINAMATH_CALUDE_propositions_are_true_l1351_135129

-- Define the propositions
def similar_triangles_equal_perimeters : Prop := sorry
def similar_triangles_equal_angles : Prop := sorry
def sqrt_9_not_negative_3 : Prop := sorry
def diameter_bisects_chord : Prop := sorry
def diameter_bisects_arcs : Prop := sorry

-- Theorem to prove
theorem propositions_are_true :
  (similar_triangles_equal_perimeters ∨ similar_triangles_equal_angles) ∧
  sqrt_9_not_negative_3 ∧
  (diameter_bisects_chord ∧ diameter_bisects_arcs) :=
by
  sorry

end NUMINAMATH_CALUDE_propositions_are_true_l1351_135129


namespace NUMINAMATH_CALUDE_semicircle_chord_length_l1351_135187

theorem semicircle_chord_length (R a b : ℝ) (h1 : R > 0) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b = R) (h5 : (π/2) * (R^2 - a^2 - b^2) = 10*π) : 
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_chord_length_l1351_135187


namespace NUMINAMATH_CALUDE_f_ratio_calc_l1351_135186

axiom f : ℝ → ℝ

axiom f_property : ∀ (a b : ℝ), b^2 * f a = a^2 * f b

axiom f2_nonzero : f 2 ≠ 0

theorem f_ratio_calc : (f 6 - f 3) / f 2 = 27 / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_ratio_calc_l1351_135186


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l1351_135177

/-- A trapezoid with angles in arithmetic progression -/
structure ArithmeticTrapezoid where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference
  angle_sum : a + (a + d) + (a + 2*d) + (a + 3*d) = 360
  largest_angle : a + 3*d = 150

theorem smallest_angle_measure (t : ArithmeticTrapezoid) : t.a = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l1351_135177


namespace NUMINAMATH_CALUDE_line_slope_angle_l1351_135189

/-- The slope angle of a line given by parametric equations -/
def slope_angle (x y : ℝ → ℝ) : ℝ := sorry

theorem line_slope_angle :
  let x : ℝ → ℝ := λ t => Real.sin θ + t * Real.sin (15 * π / 180)
  let y : ℝ → ℝ := λ t => Real.cos θ - t * Real.sin (75 * π / 180)
  slope_angle x y = 105 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_line_slope_angle_l1351_135189


namespace NUMINAMATH_CALUDE_bookstore_travel_options_l1351_135104

theorem bookstore_travel_options (bus_ways subway_ways : ℕ) 
  (h1 : bus_ways = 3) 
  (h2 : subway_ways = 4) : 
  bus_ways + subway_ways = 7 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_travel_options_l1351_135104


namespace NUMINAMATH_CALUDE_correct_grade12_sample_l1351_135153

/-- Calculates the number of students to be drawn from grade 12 in a stratified sample -/
def students_from_grade12 (total_students : ℕ) (grade10_students : ℕ) (grade11_students : ℕ) (sample_size : ℕ) : ℕ :=
  let grade12_students := total_students - (grade10_students + grade11_students)
  (grade12_students * sample_size) / total_students

/-- Theorem stating the correct number of students to be drawn from grade 12 -/
theorem correct_grade12_sample : 
  students_from_grade12 2400 820 780 120 = 40 := by
  sorry

end NUMINAMATH_CALUDE_correct_grade12_sample_l1351_135153


namespace NUMINAMATH_CALUDE_industrial_lubricants_allocation_l1351_135148

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  basic_astrophysics_degrees : ℝ
  total_degrees : ℝ

/-- Theorem stating that the industrial lubricants allocation is 8% --/
theorem industrial_lubricants_allocation
  (budget : BudgetAllocation)
  (h1 : budget.microphotonics = 12)
  (h2 : budget.home_electronics = 24)
  (h3 : budget.food_additives = 15)
  (h4 : budget.genetically_modified_microorganisms = 29)
  (h5 : budget.basic_astrophysics_degrees = 43.2)
  (h6 : budget.total_degrees = 360) :
  100 - (budget.microphotonics + budget.home_electronics + budget.food_additives +
    budget.genetically_modified_microorganisms + budget.basic_astrophysics_degrees *
    100 / budget.total_degrees) = 8 := by
  sorry


end NUMINAMATH_CALUDE_industrial_lubricants_allocation_l1351_135148


namespace NUMINAMATH_CALUDE_soccer_attendance_difference_l1351_135138

theorem soccer_attendance_difference (seattle_estimate chicago_estimate : ℕ) 
  (seattle_actual chicago_actual : ℝ) : 
  seattle_estimate = 40000 →
  chicago_estimate = 50000 →
  seattle_actual ≥ 0.85 * seattle_estimate ∧ seattle_actual ≤ 1.15 * seattle_estimate →
  chicago_actual ≥ chicago_estimate / 1.15 ∧ chicago_actual ≤ chicago_estimate / 0.85 →
  ∃ (max_diff : ℕ), max_diff = 25000 ∧ 
    ∀ (diff : ℝ), diff = chicago_actual - seattle_actual → 
      diff ≤ max_diff ∧ 
      (max_diff - 500 < diff ∨ diff < max_diff + 500) :=
by sorry

end NUMINAMATH_CALUDE_soccer_attendance_difference_l1351_135138


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l1351_135115

theorem product_of_roots_plus_one (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (1 + p) * (1 + q) * (1 + r) = 51 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l1351_135115


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l1351_135128

theorem maintenance_check_increase (original_days new_days : ℝ) 
  (h1 : original_days = 20)
  (h2 : new_days = 25) :
  ((new_days - original_days) / original_days) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l1351_135128


namespace NUMINAMATH_CALUDE_dispatch_plans_count_l1351_135173

theorem dispatch_plans_count : ∀ (n m k : ℕ),
  n = 6 → m = 4 → k = 2 →
  (Nat.choose n k) * (n - k) * (n - k - 1) = 180 :=
by sorry

end NUMINAMATH_CALUDE_dispatch_plans_count_l1351_135173


namespace NUMINAMATH_CALUDE_p_h_neg_three_equals_eight_l1351_135146

-- Define the function h
def h (x : ℝ) : ℝ := 2 * x^2 - 10

-- Define the theorem
theorem p_h_neg_three_equals_eight 
  (p : ℝ → ℝ) -- p is a function from reals to reals
  (h_def : ∀ x, h x = 2 * x^2 - 10) -- definition of h
  (p_h_three : p (h 3) = 8) -- given condition
  : p (h (-3)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_p_h_neg_three_equals_eight_l1351_135146


namespace NUMINAMATH_CALUDE_proposition_is_false_l1351_135158

theorem proposition_is_false : ∃ (angle1 angle2 : ℝ),
  angle1 + angle2 = 90 ∧ angle1 = angle2 :=
by sorry

end NUMINAMATH_CALUDE_proposition_is_false_l1351_135158


namespace NUMINAMATH_CALUDE_five_divides_x_l1351_135141

theorem five_divides_x (x y : ℕ) (hx : x > 1) (heq : 2 * x^2 - 1 = y^15) : 5 ∣ x := by
  sorry

end NUMINAMATH_CALUDE_five_divides_x_l1351_135141


namespace NUMINAMATH_CALUDE_middle_to_tallest_tree_ratio_l1351_135131

/-- Given three trees in a town square, prove the ratio of the middle height tree to the tallest tree -/
theorem middle_to_tallest_tree_ratio 
  (tallest_height : ℝ) 
  (shortest_height : ℝ) 
  (h_tallest : tallest_height = 150) 
  (h_shortest : shortest_height = 50) 
  (h_middle_relation : ∃ middle_height : ℝ, middle_height = 2 * shortest_height) :
  ∃ (middle_height : ℝ), middle_height / tallest_height = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_middle_to_tallest_tree_ratio_l1351_135131


namespace NUMINAMATH_CALUDE_B_power_106_l1351_135178

def B : Matrix (Fin 3) (Fin 3) ℤ := ![![0, 1, 0], ![0, 0, -1], ![0, 1, 0]]

theorem B_power_106 : B^106 = ![![0, 0, -1], ![0, -1, 0], ![0, 0, -1]] := by sorry

end NUMINAMATH_CALUDE_B_power_106_l1351_135178


namespace NUMINAMATH_CALUDE_square_plus_abs_eq_zero_l1351_135197

theorem square_plus_abs_eq_zero (x y : ℝ) :
  x^2 + |y + 8| = 0 → x = 0 ∧ y = -8 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_abs_eq_zero_l1351_135197


namespace NUMINAMATH_CALUDE_instrument_probability_l1351_135135

theorem instrument_probability (total : ℕ) (at_least_one : ℕ) (two_or_more : ℕ) :
  total = 800 →
  at_least_one = total / 5 →
  two_or_more = 32 →
  (at_least_one - two_or_more : ℚ) / total = 1 / 6.25 := by
  sorry

end NUMINAMATH_CALUDE_instrument_probability_l1351_135135


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1351_135182

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1351_135182


namespace NUMINAMATH_CALUDE_expression_equals_one_half_l1351_135113

theorem expression_equals_one_half :
  (4 * 6) / (12 * 8) * (5 * 12 * 8) / (4 * 5 * 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_half_l1351_135113


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1351_135180

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1351_135180


namespace NUMINAMATH_CALUDE_tetrahedron_division_ratio_l1351_135145

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Calculates the volume of a part of a tetrahedron cut by a plane -/
def partialTetrahedronVolume (t : Tetrahedron) (p : Plane) : ℝ := sorry

/-- Checks if a point lies on a line segment between two other points -/
def isOnLineSegment (p : Point3D) (a : Point3D) (b : Point3D) : Prop := sorry

/-- Checks if a point lies on the extension of a line segment beyond a point -/
def isOnLineExtension (p : Point3D) (a : Point3D) (b : Point3D) : Prop := sorry

/-- Theorem: The plane divides the tetrahedron in the ratio 2:33 -/
theorem tetrahedron_division_ratio (ABCD : Tetrahedron) (K M N : Point3D) (p : Plane) : 
  isOnLineSegment K ABCD.A ABCD.D ∧ 
  isOnLineExtension N ABCD.A ABCD.B ∧ 
  isOnLineExtension M ABCD.A ABCD.C ∧ 
  (ABCD.A.x - K.x) / (K.x - ABCD.D.x) = 3 ∧
  (N.x - ABCD.B.x) = (ABCD.B.x - ABCD.A.x) ∧
  (M.x - ABCD.C.x) / (ABCD.C.x - ABCD.A.x) = 1/3 ∧
  (p.a * K.x + p.b * K.y + p.c * K.z + p.d = 0) ∧
  (p.a * M.x + p.b * M.y + p.c * M.z + p.d = 0) ∧
  (p.a * N.x + p.b * N.y + p.c * N.z + p.d = 0) →
  (partialTetrahedronVolume ABCD p) / (tetrahedronVolume ABCD) = 2/35 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_division_ratio_l1351_135145


namespace NUMINAMATH_CALUDE_correct_allocation_schemes_l1351_135157

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 6

/-- Represents the number of venues -/
def num_venues : ℕ := 3

/-- Represents the number of volunteers per group -/
def group_size : ℕ := 2

/-- Represents that volunteers A and B must be in the same group -/
def fixed_pair : ℕ := 1

/-- The number of ways to allocate volunteers to venues -/
def allocation_schemes : ℕ := 18

/-- Theorem stating that the number of allocation schemes is correct -/
theorem correct_allocation_schemes :
  (num_volunteers.choose group_size * (num_volunteers - group_size).choose group_size / 2) *
  num_venues.factorial = allocation_schemes := by
  sorry

end NUMINAMATH_CALUDE_correct_allocation_schemes_l1351_135157


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l1351_135137

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 55)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : total_players - (total_players - throwers) / 3 = 49 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l1351_135137


namespace NUMINAMATH_CALUDE_students_walking_home_l1351_135130

theorem students_walking_home (car_pickup : ℚ) (bus_ride : ℚ) (cycle_home : ℚ) 
  (h1 : car_pickup = 1/3)
  (h2 : bus_ride = 1/5)
  (h3 : cycle_home = 1/8)
  (h4 : car_pickup + bus_ride + cycle_home + (walk_home : ℚ) = 1) :
  walk_home = 41/120 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l1351_135130


namespace NUMINAMATH_CALUDE_cross_section_distance_from_apex_l1351_135121

-- Define the structure of a right pentagonal pyramid
structure RightPentagonalPyramid where
  -- Add any necessary fields

-- Define a cross section of the pyramid
structure CrossSection where
  area : ℝ
  distanceFromApex : ℝ

-- Define the theorem
theorem cross_section_distance_from_apex 
  (pyramid : RightPentagonalPyramid)
  (section1 section2 : CrossSection)
  (h1 : section1.area = 125 * Real.sqrt 3)
  (h2 : section2.area = 500 * Real.sqrt 3)
  (h3 : section2.distanceFromApex - section1.distanceFromApex = 12)
  (h4 : section2.area > section1.area) :
  section2.distanceFromApex = 24 := by
sorry

end NUMINAMATH_CALUDE_cross_section_distance_from_apex_l1351_135121


namespace NUMINAMATH_CALUDE_cube_of_sqrt_three_l1351_135171

theorem cube_of_sqrt_three (x : ℝ) (h : Real.sqrt (x - 3) = 3) : (x - 3)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_sqrt_three_l1351_135171


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1351_135149

def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 2016) + y^2 / (k - 2018) = 1

def asymptote_equation (x y : ℝ) : Prop :=
  x + y = 0 ∨ x - y = 0

theorem hyperbola_asymptotes (k : ℤ) :
  (∃ x y : ℝ, hyperbola_equation x y (k : ℝ)) →
  (∀ x y : ℝ, hyperbola_equation x y (k : ℝ) → asymptote_equation x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1351_135149


namespace NUMINAMATH_CALUDE_fifteenth_prime_l1351_135102

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime : nthPrime 15 = 47 := by sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l1351_135102


namespace NUMINAMATH_CALUDE_complex_magnitude_three_fourths_plus_three_i_l1351_135168

theorem complex_magnitude_three_fourths_plus_three_i :
  Complex.abs (3 / 4 + 3 * Complex.I) = Real.sqrt 153 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_three_fourths_plus_three_i_l1351_135168


namespace NUMINAMATH_CALUDE_building_height_from_shadows_l1351_135109

/-- Given a flagstaff and a building casting shadows under similar conditions,
    calculate the height of the building using the concept of similar triangles. -/
theorem building_height_from_shadows
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagstaff_height : flagstaff_height = 17.5)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  ∃ (building_height : ℝ),
    (building_height / building_shadow = flagstaff_height / flagstaff_shadow) ∧
    (abs (building_height - 12.44) < 0.01) :=
sorry

end NUMINAMATH_CALUDE_building_height_from_shadows_l1351_135109


namespace NUMINAMATH_CALUDE_mudits_age_l1351_135188

theorem mudits_age : ∃ (x : ℕ), x + 16 = 3 * (x - 4) ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_mudits_age_l1351_135188


namespace NUMINAMATH_CALUDE_orange_price_calculation_l1351_135125

-- Define the price function for oranges
def orange_price (mass : ℝ) : ℝ := sorry

-- State the theorem
theorem orange_price_calculation 
  (proportional : ∀ m₁ m₂ : ℝ, orange_price m₁ / m₁ = orange_price m₂ / m₂)
  (given_price : orange_price 12 = 36) :
  orange_price 2 = 6 := by sorry

end NUMINAMATH_CALUDE_orange_price_calculation_l1351_135125


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1351_135112

theorem decimal_to_fraction : 
  (3.126 : ℚ) = 1563 / 500 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1351_135112


namespace NUMINAMATH_CALUDE_new_average_weight_l1351_135198

/-- Given a bowling team with the following properties:
  * The original team has 7 players
  * The original average weight is 103 kg
  * Two new players join the team
  * One new player weighs 110 kg
  * The other new player weighs 60 kg
  
  Prove that the new average weight of the team is 99 kg -/
theorem new_average_weight 
  (original_players : Nat) 
  (original_avg_weight : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : original_players = 7)
  (h2 : original_avg_weight = 103)
  (h3 : new_player1_weight = 110)
  (h4 : new_player2_weight = 60) :
  let total_weight := original_players * original_avg_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  total_weight / new_total_players = 99 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l1351_135198


namespace NUMINAMATH_CALUDE_unpacked_boxes_correct_l1351_135142

-- Define the cookie types
inductive CookieType
  | LemonChaletCremes
  | ThinMints
  | Samoas
  | Trefoils

-- Define the function for boxes per case
def boxesPerCase (c : CookieType) : ℕ :=
  match c with
  | CookieType.LemonChaletCremes => 12
  | CookieType.ThinMints => 15
  | CookieType.Samoas => 10
  | CookieType.Trefoils => 18

-- Define the function for boxes sold
def boxesSold (c : CookieType) : ℕ :=
  match c with
  | CookieType.LemonChaletCremes => 31
  | CookieType.ThinMints => 26
  | CookieType.Samoas => 17
  | CookieType.Trefoils => 44

-- Define the function for unpacked boxes
def unpackedBoxes (c : CookieType) : ℕ :=
  boxesSold c % boxesPerCase c

-- Theorem statement
theorem unpacked_boxes_correct (c : CookieType) :
  unpackedBoxes c =
    match c with
    | CookieType.LemonChaletCremes => 7
    | CookieType.ThinMints => 11
    | CookieType.Samoas => 7
    | CookieType.Trefoils => 8
  := by sorry

end NUMINAMATH_CALUDE_unpacked_boxes_correct_l1351_135142


namespace NUMINAMATH_CALUDE_ellipse_theorem_parabola_theorem_l1351_135160

-- Define the ellipses
def ellipse1 (x y : ℝ) := x^2/9 + y^2/4 = 1
def ellipse2 (x y : ℝ) := x^2/12 + y^2/7 = 1

-- Define the parabolas
def parabola1 (x y : ℝ) := x^2 = -2 * Real.sqrt 2 * y
def parabola2 (x y : ℝ) := y^2 = -8 * x

-- Theorem for the ellipse
theorem ellipse_theorem :
  (ellipse2 (-3) 2) ∧
  (∀ (x y : ℝ), ellipse1 x y ↔ ellipse2 x y) := by sorry

-- Theorem for the parabolas
theorem parabola_theorem :
  (parabola1 (-4) (-4 * Real.sqrt 2)) ∧
  (parabola2 (-4) (-4 * Real.sqrt 2)) ∧
  (∀ (x y : ℝ), parabola1 x y → x = 0 ∨ y = 0) ∧
  (∀ (x y : ℝ), parabola2 x y → x = 0 ∨ y = 0) := by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_parabola_theorem_l1351_135160


namespace NUMINAMATH_CALUDE_cubic_roots_cube_l1351_135166

theorem cubic_roots_cube (a b c : ℝ) (α β γ : ℂ) :
  (∀ x : ℂ, x^3 + a*x^2 + b*x + c = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∀ x : ℂ, x^3 + (-a^3 + 3*a*b - 3*c)*x^2 + (-b^3 + 3*a*b*c)*x + c^3 = 0 ↔ 
    x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_cube_l1351_135166


namespace NUMINAMATH_CALUDE_sum_greater_than_one_l1351_135190

theorem sum_greater_than_one : 
  (let a := [1/4, 2/8, 3/4]
   let b := [3, -1.5, -0.5]
   let c := [0.25, 0.75, 0.05]
   let d := [3/2, -3/4, 1/4]
   let e := [1.5, 1.5, -2]
   (a.sum > 1 ∧ c.sum > 1) ∧
   (b.sum ≤ 1 ∧ d.sum ≤ 1 ∧ e.sum ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_one_l1351_135190


namespace NUMINAMATH_CALUDE_root_product_theorem_l1351_135111

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 - x₁^2 + 5 = 0) → 
  (x₂^5 - x₂^2 + 5 = 0) → 
  (x₃^5 - x₃^2 + 5 = 0) → 
  (x₄^5 - x₄^2 + 5 = 0) → 
  (x₅^5 - x₅^2 + 5 = 0) → 
  let f := fun x : ℂ ↦ x^2 + 1
  (f x₁) * (f x₂) * (f x₃) * (f x₄) * (f x₅) = 37 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1351_135111


namespace NUMINAMATH_CALUDE_evaluate_expression_l1351_135150

theorem evaluate_expression : 8^6 * 27^6 * 8^15 * 27^15 = 216^21 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1351_135150


namespace NUMINAMATH_CALUDE_addition_preserves_inequality_l1351_135123

theorem addition_preserves_inequality (a b c d : ℝ) :
  a > b → c > d → a + c > b + d := by sorry

end NUMINAMATH_CALUDE_addition_preserves_inequality_l1351_135123


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1351_135179

/-- Given two circles in the xy-plane, prove that their common chord has a specific equation. -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l1351_135179


namespace NUMINAMATH_CALUDE_initial_milk_collected_l1351_135152

/-- Proves that the initial amount of milk collected equals 30,000 gallons -/
theorem initial_milk_collected (
  pumping_hours : ℕ)
  (pumping_rate : ℕ)
  (adding_hours : ℕ)
  (adding_rate : ℕ)
  (milk_left : ℕ)
  (h1 : pumping_hours = 4)
  (h2 : pumping_rate = 2880)
  (h3 : adding_hours = 7)
  (h4 : adding_rate = 1500)
  (h5 : milk_left = 28980)
  (h6 : ∃ initial_milk : ℕ, 
    initial_milk + adding_hours * adding_rate - pumping_hours * pumping_rate = milk_left) :
  ∃ initial_milk : ℕ, initial_milk = 30000 := by
sorry


end NUMINAMATH_CALUDE_initial_milk_collected_l1351_135152


namespace NUMINAMATH_CALUDE_initial_to_doubled_ratio_l1351_135167

theorem initial_to_doubled_ratio (x : ℚ) (h : 3 * (2 * x + 5) = 111) : 
  x / (2 * x) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_initial_to_doubled_ratio_l1351_135167


namespace NUMINAMATH_CALUDE_function_composition_l1351_135172

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * x + 7

-- State the theorem
theorem function_composition (x : ℝ) : 
  (fun x => f (x - 1)) = (fun x => 2 * x + 5) → f (x^2) = 2 * x^2 + 7 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l1351_135172


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1351_135169

/-- The ratio of volumes of two cubes, one with sides of 2 meters and another with sides of 100 centimeters. -/
theorem cube_volume_ratio : 
  let cube1_side : ℝ := 2  -- Side length of Cube 1 in meters
  let cube2_side : ℝ := 100 / 100  -- Side length of Cube 2 in meters (100 cm converted to m)
  let cube1_volume := cube1_side ^ 3
  let cube2_volume := cube2_side ^ 3
  cube1_volume / cube2_volume = 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1351_135169


namespace NUMINAMATH_CALUDE_alpha_beta_equivalence_l1351_135120

theorem alpha_beta_equivalence (α β : ℝ) :
  (α + β > 0) ↔ (α + β > Real.cos α - Real.cos β) := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_equivalence_l1351_135120


namespace NUMINAMATH_CALUDE_not_term_of_sequence_l1351_135170

theorem not_term_of_sequence (n : ℕ+) : 25 - 2 * (n : ℤ) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_term_of_sequence_l1351_135170


namespace NUMINAMATH_CALUDE_det_specific_matrix_l1351_135105

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 2; 0, 6, -1; 5, -3, 1]
  Matrix.det A = -34 := by
    sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l1351_135105


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l1351_135139

/-- Proves that for a hyperbola x²/a² - y² = 1 with a > 0, 
    if one of its asymptotes is y + 2x = 0, then a = 2 -/
theorem hyperbola_asymptote_a_value (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 = 1 ∧ y + 2*x = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l1351_135139


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1351_135136

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ := sorry

/-- The distance from a focus to an asymptote of a hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := sorry

/-- Theorem: If the distance from a focus to an asymptote is 1/4 of the focal distance,
    then the eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_dist : focus_to_asymptote_distance h = (1/4) * focal_distance h) : 
    eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1351_135136


namespace NUMINAMATH_CALUDE_trajectory_is_ray_l1351_135175

/-- The set of complex numbers z satisfying |z+1| - |z-1| = 2 -/
def S : Set ℂ :=
  {z : ℂ | Complex.abs (z + 1) - Complex.abs (z - 1) = 2}

/-- A ray starting from (1, 0) and extending to the right -/
def R : Set ℂ :=
  {z : ℂ | ∃ (t : ℝ), t ≥ 0 ∧ z = 1 + t}

/-- Theorem stating that S equals R -/
theorem trajectory_is_ray : S = R := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_ray_l1351_135175


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1351_135191

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : a 1 > 0) :
  (is_increasing_sequence a → a 1 ^ 2 < a 2 ^ 2) ∧
  ¬(a 1 ^ 2 < a 2 ^ 2 → is_increasing_sequence a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1351_135191


namespace NUMINAMATH_CALUDE_lowest_possible_score_l1351_135161

def test_count : Nat := 6
def max_score : Nat := 100
def target_average : Nat := 85
def min_score : Nat := 75

def first_four_scores : List Nat := [79, 88, 94, 91]

theorem lowest_possible_score :
  ∀ (score1 score2 : Nat),
  (score1 ≥ min_score) →
  (score2 ≥ min_score) →
  (List.sum first_four_scores + score1 + score2) / test_count = target_average →
  (∀ (s : Nat), s ≥ min_score ∧ s < score1 →
    (List.sum first_four_scores + s + score2) / test_count < target_average) →
  score1 = min_score ∨ score2 = min_score :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l1351_135161


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1351_135196

theorem rectangle_area (perimeter : ℝ) (h1 : perimeter = 56) : ℝ :=
  let side_length := perimeter / 8
  let square_area := side_length ^ 2
  3 * square_area

theorem rectangle_area_proof :
  rectangle_area 56 rfl = 147 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1351_135196


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1351_135163

theorem sin_alpha_value (α : Real) : 
  (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) ∈ {(x, y) | ∃ r > 0, x = r * Real.cos α ∧ y = r * Real.sin α} → 
  Real.sin α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1351_135163


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l1351_135192

theorem multiplication_addition_equality : 3.5 * 0.3 + 1.2 * 0.4 = 1.53 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l1351_135192


namespace NUMINAMATH_CALUDE_sequence_existence_l1351_135116

theorem sequence_existence (a b : ℕ) (h1 : b > a) (h2 : a > 1) (h3 : ¬(a ∣ b))
  (b_seq : ℕ → ℕ) (h4 : ∀ n, b_seq (n + 1) ≥ 2 * b_seq n) :
  ∃ a_seq : ℕ → ℕ,
    (∀ n, a_seq (n + 1) - a_seq n = a ∨ a_seq (n + 1) - a_seq n = b) ∧
    (∀ m l, a_seq m + a_seq l ∉ Set.range b_seq) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_l1351_135116


namespace NUMINAMATH_CALUDE_man_gained_three_toys_cost_l1351_135143

/-- The number of toys whose cost price the man gained -/
def toys_gained (num_sold : ℕ) (selling_price : ℕ) (cost_price : ℕ) : ℕ :=
  (selling_price - num_sold * cost_price) / cost_price

theorem man_gained_three_toys_cost :
  toys_gained 18 27300 1300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_man_gained_three_toys_cost_l1351_135143


namespace NUMINAMATH_CALUDE_remaining_staff_count_l1351_135126

/-- Calculates the remaining staff in a cafe after some leave --/
theorem remaining_staff_count 
  (initial_chefs initial_waiters initial_busboys initial_hostesses : ℕ)
  (leaving_chefs leaving_waiters leaving_busboys leaving_hostesses : ℕ)
  (h1 : initial_chefs = 16)
  (h2 : initial_waiters = 16)
  (h3 : initial_busboys = 10)
  (h4 : initial_hostesses = 5)
  (h5 : leaving_chefs = 6)
  (h6 : leaving_waiters = 3)
  (h7 : leaving_busboys = 4)
  (h8 : leaving_hostesses = 2) :
  (initial_chefs - leaving_chefs) + (initial_waiters - leaving_waiters) + 
  (initial_busboys - leaving_busboys) + (initial_hostesses - leaving_hostesses) = 32 := by
  sorry

end NUMINAMATH_CALUDE_remaining_staff_count_l1351_135126


namespace NUMINAMATH_CALUDE_system_solution_l1351_135176

theorem system_solution :
  ∃! (x y : ℤ), 16*x + 24*y = 32 ∧ 24*x + 16*y = 48 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l1351_135176


namespace NUMINAMATH_CALUDE_consecutive_integers_product_not_square_l1351_135132

theorem consecutive_integers_product_not_square (a : ℕ) : 
  let A := Finset.range 20
  let sum := A.sum (λ i => a + i)
  let prod := A.prod (λ i => a + i)
  (sum % 23 ≠ 0) → (prod % 23 ≠ 0) → ¬ ∃ (n : ℕ), prod = n^2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_not_square_l1351_135132


namespace NUMINAMATH_CALUDE_sector_area_of_circle_l1351_135108

/-- Given a circle with circumference 16π, prove that the area of a sector
    subtending a central angle of 90° is 16π. -/
theorem sector_area_of_circle (C : ℝ) (θ : ℝ) (h1 : C = 16 * Real.pi) (h2 : θ = 90) :
  let r := C / (2 * Real.pi)
  let A := Real.pi * r^2
  let sector_area := (θ / 360) * A
  sector_area = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sector_area_of_circle_l1351_135108


namespace NUMINAMATH_CALUDE_not_all_mages_are_wizards_l1351_135100

-- Define the universe of discourse
variable {U : Type}

-- Define predicates for being a mage, sorcerer, and wizard
variable (Mage Sorcerer Wizard : U → Prop)

-- State the theorem
theorem not_all_mages_are_wizards :
  (∃ x, Mage x ∧ ¬Sorcerer x) →
  (∀ x, Mage x ∧ Wizard x → Sorcerer x) →
  ∃ x, Mage x ∧ ¬Wizard x :=
by sorry

end NUMINAMATH_CALUDE_not_all_mages_are_wizards_l1351_135100


namespace NUMINAMATH_CALUDE_coin_overlap_area_l1351_135154

theorem coin_overlap_area (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) :
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  ∃ (overlap_area : ℝ),
    overlap_area = 52 ∧
    overlap_area = (36 + 16 * Real.sqrt 2 + 2 * Real.pi) / 
      ((square_side - coin_diameter) * (square_side - coin_diameter)) :=
by sorry

end NUMINAMATH_CALUDE_coin_overlap_area_l1351_135154


namespace NUMINAMATH_CALUDE_star_five_three_l1351_135165

-- Define the ★ operation
def star (a b : ℚ) : ℚ := a^2 + 2*a/b

-- State the theorem
theorem star_five_three : star 5 3 = 85/3 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l1351_135165


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l1351_135127

theorem scientific_notation_proof : 
  (192000000 : ℝ) = 1.92 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l1351_135127


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1351_135156

theorem smallest_five_digit_divisible_by_first_five_primes : 
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    2 ∣ n ∧ 
    3 ∣ n ∧ 
    5 ∣ n ∧ 
    7 ∣ n ∧ 
    11 ∣ n ∧ 
    ∀ m : ℕ, 
      m ≥ 10000 ∧ 
      m < 100000 ∧ 
      2 ∣ m ∧ 
      3 ∣ m ∧ 
      5 ∣ m ∧ 
      7 ∣ m ∧ 
      11 ∣ m → 
      n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1351_135156


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l1351_135185

/-- Proves that the new supervisor's salary is $960 given the conditions of the problem -/
theorem new_supervisor_salary
  (num_workers : ℕ)
  (num_total : ℕ)
  (initial_avg_salary : ℚ)
  (old_supervisor_salary : ℚ)
  (new_avg_salary : ℚ)
  (h_num_workers : num_workers = 8)
  (h_num_total : num_total = num_workers + 1)
  (h_initial_avg : initial_avg_salary = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_avg : new_avg_salary = 440)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 960 ∧
    (num_workers : ℚ) * initial_avg_salary + old_supervisor_salary = (num_total : ℚ) * initial_avg_salary ∧
    (num_workers : ℚ) * initial_avg_salary + new_supervisor_salary = (num_total : ℚ) * new_avg_salary :=
by sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l1351_135185


namespace NUMINAMATH_CALUDE_prob_score_over_14_is_0_3_expected_value_is_13_6_l1351_135140

-- Define the success rates and point values
def three_week_success_rate : ℝ := 0.7
def four_week_success_rate : ℝ := 0.3
def three_week_success_points : ℝ := 8
def three_week_failure_points : ℝ := 4
def four_week_success_points : ℝ := 15
def four_week_failure_points : ℝ := 6

-- Define the probability of scoring more than 14 points
-- in a sequence of a three-week jump followed by a four-week jump
def prob_score_over_14 : ℝ :=
  three_week_success_rate * four_week_success_rate +
  (1 - three_week_success_rate) * four_week_success_rate

-- Define the expected value of the total score for two consecutive three-week jumps
def expected_value_two_three_week_jumps : ℝ :=
  (1 - three_week_success_rate)^2 * (2 * three_week_failure_points) +
  2 * three_week_success_rate * (1 - three_week_success_rate) * (three_week_success_points + three_week_failure_points) +
  three_week_success_rate^2 * (2 * three_week_success_points)

-- Theorem statements
theorem prob_score_over_14_is_0_3 : prob_score_over_14 = 0.3 := by sorry

theorem expected_value_is_13_6 : expected_value_two_three_week_jumps = 13.6 := by sorry

end NUMINAMATH_CALUDE_prob_score_over_14_is_0_3_expected_value_is_13_6_l1351_135140


namespace NUMINAMATH_CALUDE_sum_100th_group_value_l1351_135106

/-- The sum of the three numbers in the 100th group of the sequence (n, n^2, n^3) -/
def sum_100th_group : ℕ := 100 + 100^2 + 100^3

/-- Theorem stating that the sum of the 100th group is 1010100 -/
theorem sum_100th_group_value : sum_100th_group = 1010100 := by
  sorry

end NUMINAMATH_CALUDE_sum_100th_group_value_l1351_135106


namespace NUMINAMATH_CALUDE_min_points_in_segment_seven_is_minimum_l1351_135124

-- Define the type for points on the number line
def Point := ℝ

-- Define the segments
def Segment := Set Point

-- Define the three segments
def leftSegment : Segment := {x : ℝ | x < -2}
def middleSegment : Segment := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def rightSegment : Segment := {x : ℝ | x > 2}

-- Define a property for a set of points
def hasThreePointsInOneSegment (points : Set Point) : Prop :=
  (points ∩ leftSegment).ncard ≥ 3 ∨
  (points ∩ middleSegment).ncard ≥ 3 ∨
  (points ∩ rightSegment).ncard ≥ 3

-- The main theorem
theorem min_points_in_segment :
  ∀ n : ℕ, n ≥ 7 →
    ∀ points : Set Point, points.ncard = n →
      hasThreePointsInOneSegment points :=
sorry

theorem seven_is_minimum :
  ∃ points : Set Point, points.ncard = 6 ∧
    ¬hasThreePointsInOneSegment points :=
sorry

end NUMINAMATH_CALUDE_min_points_in_segment_seven_is_minimum_l1351_135124


namespace NUMINAMATH_CALUDE_system_solution_l1351_135118

theorem system_solution :
  ∃ (x y : ℝ), (1/2 * x - 3/2 * y = -1) ∧ (2 * x + y = 3) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1351_135118


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1351_135164

theorem power_tower_mod_500 : 7^(7^(7^7)) % 500 = 343 := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1351_135164


namespace NUMINAMATH_CALUDE_absolute_value_sum_lower_bound_l1351_135117

theorem absolute_value_sum_lower_bound :
  (∀ x : ℝ, |x + 2| + |x - 1| ≥ 3) ∧
  (∀ ε > 0, ∃ x : ℝ, |x + 2| + |x - 1| < 3 + ε) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_lower_bound_l1351_135117
