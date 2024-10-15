import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l446_44696

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)      -- e is negative
  (h4 : |e| = 1)    -- absolute value of e is 1
  : (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l446_44696


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l446_44656

theorem quadratic_form_h_value : ∃ (a k : ℝ), ∀ x : ℝ, 
  3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l446_44656


namespace NUMINAMATH_CALUDE_largest_circle_at_a_l446_44684

/-- A pentagon with circles centered at each vertex -/
structure PentagonWithCircles where
  -- Lengths of the pentagon sides
  ab : ℝ
  bc : ℝ
  cd : ℝ
  de : ℝ
  ea : ℝ
  -- Radii of the circles
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  r_d : ℝ
  r_e : ℝ
  -- Conditions for circles touching on sides
  h_ab : r_a + r_b = ab
  h_bc : r_b + r_c = bc
  h_cd : r_c + r_d = cd
  h_de : r_d + r_e = de
  h_ea : r_e + r_a = ea

/-- The circle centered at A has the largest radius -/
theorem largest_circle_at_a (p : PentagonWithCircles)
  (h_ab : p.ab = 16)
  (h_bc : p.bc = 14)
  (h_cd : p.cd = 17)
  (h_de : p.de = 13)
  (h_ea : p.ea = 14) :
  p.r_a = max p.r_a (max p.r_b (max p.r_c (max p.r_d p.r_e))) :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_at_a_l446_44684


namespace NUMINAMATH_CALUDE_right_triangle_circle_intersection_l446_44616

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point D
def D : ℝ × ℝ := sorry

-- Define the properties of the triangle and circle
def is_right_triangle (t : Triangle) : Prop :=
  sorry

def circle_intersects_BC (t : Triangle) (c : Circle) : Prop :=
  sorry

def AC_is_diameter (t : Triangle) (c : Circle) : Prop :=
  sorry

-- Theorem statement
theorem right_triangle_circle_intersection 
  (t : Triangle) (c : Circle) :
  is_right_triangle t →
  circle_intersects_BC t c →
  AC_is_diameter t c →
  t.A.1 - t.B.1 = 18 →
  t.A.1 - t.C.1 = 30 →
  D.1 - t.B.1 = 14.4 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_circle_intersection_l446_44616


namespace NUMINAMATH_CALUDE_arc_length_problem_l446_44658

theorem arc_length_problem (r : ℝ) (θ : ℝ) (a : ℝ) :
  r = 18 →
  θ = π / 3 →
  r * θ = a * π →
  a = 6 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_problem_l446_44658


namespace NUMINAMATH_CALUDE_students_in_all_classes_l446_44623

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  code_only : ℕ
  chess_only : ℕ
  photo_only : ℕ
  code_chess : ℕ
  code_photo : ℕ
  chess_photo : ℕ
  all_three : ℕ

/-- The problem statement and conditions --/
theorem students_in_all_classes 
  (total_students : ℕ)
  (code_students : ℕ)
  (chess_students : ℕ)
  (photo_students : ℕ)
  (multi_class_students : ℕ)
  (h1 : total_students = 25)
  (h2 : code_students = 12)
  (h3 : chess_students = 15)
  (h4 : photo_students = 10)
  (h5 : multi_class_students = 10)
  (combinations : ClassCombinations)
  (h6 : total_students = 
    combinations.code_only + combinations.chess_only + combinations.photo_only + 
    combinations.code_chess + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three)
  (h7 : code_students = 
    combinations.code_only + combinations.code_chess + combinations.code_photo + 
    combinations.all_three)
  (h8 : chess_students = 
    combinations.chess_only + combinations.code_chess + combinations.chess_photo + 
    combinations.all_three)
  (h9 : photo_students = 
    combinations.photo_only + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three)
  (h10 : multi_class_students = 
    combinations.code_chess + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three) :
  combinations.all_three = 4 := by
  sorry


end NUMINAMATH_CALUDE_students_in_all_classes_l446_44623


namespace NUMINAMATH_CALUDE_additional_boys_on_slide_l446_44682

theorem additional_boys_on_slide (initial_boys total_boys : ℕ) 
  (h1 : initial_boys = 22)
  (h2 : total_boys = 35) :
  total_boys - initial_boys = 13 := by
  sorry

end NUMINAMATH_CALUDE_additional_boys_on_slide_l446_44682


namespace NUMINAMATH_CALUDE_value_range_sqrt_16_minus_4_pow_x_l446_44642

theorem value_range_sqrt_16_minus_4_pow_x :
  ∀ x : ℝ, 0 ≤ Real.sqrt (16 - 4^x) ∧ Real.sqrt (16 - 4^x) < 4 := by
  sorry

end NUMINAMATH_CALUDE_value_range_sqrt_16_minus_4_pow_x_l446_44642


namespace NUMINAMATH_CALUDE_parallel_vectors_l446_44627

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
axiom parallel_iff_cross_zero {u v : ℝ × ℝ} : 
  (∃ (k : ℝ), u = k • v ∨ v = k • u) ↔ u.1 * v.2 - u.2 * v.1 = 0

/-- Given vectors a and b, prove that a is parallel to b if and only if y = -6 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (-1, 3)) (h2 : b = (2, y)) :
  (∃ (k : ℝ), a = k • b ∨ b = k • a) ↔ y = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l446_44627


namespace NUMINAMATH_CALUDE_parabola_point_range_l446_44604

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the theorem
theorem parabola_point_range :
  ∃ (m : ℝ), 
    (m > 0) ∧
    (∀ (x₁ x₂ : ℝ),
      (-1 < x₁ ∧ x₁ < 1) →
      (m - 1 < x₂ ∧ x₂ < m) →
      (f x₁ ≠ f x₂)) ∧
    ((2 ≤ m ∧ m ≤ 3) ∨ m ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_parabola_point_range_l446_44604


namespace NUMINAMATH_CALUDE_water_added_to_tank_l446_44693

/-- The amount of water added to a tank -/
def water_added (capacity : ℚ) (initial_fraction : ℚ) (final_fraction : ℚ) : ℚ :=
  capacity * (final_fraction - initial_fraction)

/-- Theorem: The amount of water added to a 40-gallon tank, 
    initially 3/4 full and ending up 7/8 full, is 5 gallons -/
theorem water_added_to_tank : 
  water_added 40 (3/4) (7/8) = 5 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l446_44693


namespace NUMINAMATH_CALUDE_white_daisies_count_l446_44654

theorem white_daisies_count :
  ∀ (white pink red : ℕ),
    pink = 9 * white →
    red = 4 * pink - 3 →
    white + pink + red = 273 →
    white = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_white_daisies_count_l446_44654


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_zero_l446_44674

def M : Set ℝ := {x : ℝ | |x| ≤ 2}
def N : Set ℝ := {x : ℝ | x^2 - 3*x = 0}

theorem M_intersect_N_equals_zero : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_zero_l446_44674


namespace NUMINAMATH_CALUDE_circle_M_equation_l446_44667

/-- Circle M passing through two points with center on a line -/
def circle_M (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), 
    (a - b - 4 = 0) ∧ 
    (((-1) - a)^2 + ((-4) - b)^2 = (x - a)^2 + (y - b)^2) ∧
    ((6 - a)^2 + (3 - b)^2 = (x - a)^2 + (y - b)^2)

/-- Theorem: The equation of circle M is (x-3)^2 + (y+1)^2 = 25 -/
theorem circle_M_equation : 
  ∀ x y : ℝ, circle_M x y ↔ (x - 3)^2 + (y + 1)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_circle_M_equation_l446_44667


namespace NUMINAMATH_CALUDE_art_gallery_display_ratio_l446_44630

/-- Theorem about the ratio of displayed art pieces to total pieces in a gallery -/
theorem art_gallery_display_ratio 
  (total_pieces : ℕ)
  (sculptures_not_displayed : ℕ)
  (h_total : total_pieces = 3150)
  (h_sculptures_not_displayed : sculptures_not_displayed = 1400)
  (h_display_ratio : ∀ d : ℕ, d > 0 → (d : ℚ) / 6 = (sculptures_not_displayed : ℚ))
  (h_not_display_ratio : ∀ n : ℕ, n > 0 → (n : ℚ) / 3 = ((total_pieces - sculptures_not_displayed) : ℚ)) :
  (total_pieces : ℚ) / 3 = (total_pieces - sculptures_not_displayed * 3 / 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_art_gallery_display_ratio_l446_44630


namespace NUMINAMATH_CALUDE_f_is_even_l446_44663

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_is_even (g : ℝ → ℝ) (h : isOdd g) :
  isEven (fun x ↦ |g (x^4)|) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l446_44663


namespace NUMINAMATH_CALUDE_female_employees_count_l446_44611

/-- The total number of female employees in a company, given specific conditions. -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) (female_managers : ℕ) :
  female_managers = 280 →
  (2 : ℚ) / 5 * total_employees = female_managers + (2 : ℚ) / 5 * male_employees →
  total_employees = male_employees + 700 →
  700 = total_employees - male_employees :=
by sorry

end NUMINAMATH_CALUDE_female_employees_count_l446_44611


namespace NUMINAMATH_CALUDE_glasses_per_pitcher_l446_44689

theorem glasses_per_pitcher (total_glasses : ℕ) (num_pitchers : ℕ) 
  (h1 : total_glasses = 54) 
  (h2 : num_pitchers = 9) : 
  total_glasses / num_pitchers = 6 := by
sorry

end NUMINAMATH_CALUDE_glasses_per_pitcher_l446_44689


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l446_44643

-- Define the number of herbs and gems
def num_herbs : ℕ := 4
def num_gems : ℕ := 6

-- Define the number of incompatible combinations for one gem
def incompatible_combinations : ℕ := 3

-- Define the number of herbs that can be used with the specific gem
def specific_gem_combinations : ℕ := 1

-- Theorem statement
theorem wizard_elixir_combinations :
  let total_combinations := num_herbs * num_gems
  let remaining_after_incompatible := total_combinations - incompatible_combinations
  let valid_combinations := remaining_after_incompatible - (num_herbs - specific_gem_combinations)
  valid_combinations = 18 := by
  sorry


end NUMINAMATH_CALUDE_wizard_elixir_combinations_l446_44643


namespace NUMINAMATH_CALUDE_derivative_properties_neg_l446_44687

open Real

-- Define the properties of functions f and g
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

def positive_derivative_pos (f : ℝ → ℝ) : Prop := ∀ x > 0, deriv f x > 0

def negative_derivative_pos (g : ℝ → ℝ) : Prop := ∀ x > 0, deriv g x < 0

-- State the theorem
theorem derivative_properties_neg
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (hf_even : even_function f)
  (hg_odd : odd_function g)
  (hf_pos : positive_derivative_pos f)
  (hg_pos : negative_derivative_pos g) :
  ∀ x < 0, deriv f x < 0 ∧ deriv g x < 0 :=
sorry

end NUMINAMATH_CALUDE_derivative_properties_neg_l446_44687


namespace NUMINAMATH_CALUDE_integer_operation_problem_l446_44626

theorem integer_operation_problem : ∃! x : ℤ, 
  ∃ r : ℤ, 0 ≤ r ∧ r < 7 ∧ ((x - 77) * 8 = 37 * 7 + r) ∧ x = 110 := by
  sorry

end NUMINAMATH_CALUDE_integer_operation_problem_l446_44626


namespace NUMINAMATH_CALUDE_abs_sum_minus_product_equals_two_l446_44651

theorem abs_sum_minus_product_equals_two
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  |a| / a + |b| / b + |c| / c - (a * b * c) / |a * b * c| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minus_product_equals_two_l446_44651


namespace NUMINAMATH_CALUDE_length_of_AE_l446_44660

/-- The length of segment AE in a 7x5 grid where AB meets CD at E -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 4) →
  B = (6, 0) →
  C = (6, 4) →
  D = (2, 0) →
  E = (4, 2) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • B) →
  (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (1 - s) • C + s • D) →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 10 * Real.sqrt 13 / 9 := by
  sorry


end NUMINAMATH_CALUDE_length_of_AE_l446_44660


namespace NUMINAMATH_CALUDE_quartic_roots_sum_product_l446_44662

theorem quartic_roots_sum_product (a b : ℝ) : 
  a^4 - 6*a - 2 = 0 → b^4 - 6*b - 2 = 0 → a * b + a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quartic_roots_sum_product_l446_44662


namespace NUMINAMATH_CALUDE_max_cards_purchasable_l446_44680

def initial_money : ℚ := 965 / 100
def earned_money : ℚ := 535 / 100
def card_cost : ℚ := 95 / 100

theorem max_cards_purchasable : 
  ⌊(initial_money + earned_money) / card_cost⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_max_cards_purchasable_l446_44680


namespace NUMINAMATH_CALUDE_problem_statement_l446_44659

-- Define the function f
def f (a b c x : ℝ) : ℝ := a*x + b*x - c*x

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define an obtuse triangle
def is_obtuse (a b c : ℝ) : Prop :=
  a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2

theorem problem_statement 
  (a b c : ℝ) 
  (h1 : c > a ∧ a > 0) 
  (h2 : c > b ∧ b > 0) 
  (h3 : triangle_inequality a b c) :
  (∃ x : ℝ, ¬ triangle_inequality (a*x) (b*x) (c*x)) ∧ 
  (is_obtuse a b c → ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f a b c x = 0) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l446_44659


namespace NUMINAMATH_CALUDE_boys_percentage_in_specific_classroom_l446_44621

/-- Represents the composition of a classroom -/
structure Classroom where
  total_people : ℕ
  boy_girl_ratio : ℚ
  student_teacher_ratio : ℕ

/-- Calculates the percentage of boys in the classroom -/
def boys_percentage (c : Classroom) : ℚ :=
  sorry

/-- Theorem stating the percentage of boys in the specific classroom scenario -/
theorem boys_percentage_in_specific_classroom :
  let c : Classroom := {
    total_people := 36,
    boy_girl_ratio := 2 / 3,
    student_teacher_ratio := 6
  }
  boys_percentage c = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_percentage_in_specific_classroom_l446_44621


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l446_44652

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3), 
    prove that a - b = 3 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a - b = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l446_44652


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l446_44622

/-- Represents the number of households in each category -/
structure HouseholdCounts where
  farmers : ℕ
  workers : ℕ
  intellectuals : ℕ

/-- Represents the sample sizes -/
structure SampleSizes where
  farmers : ℕ
  total : ℕ

/-- Theorem stating the relationship between the household counts, 
    sample sizes, and the expected total sample size -/
theorem stratified_sampling_theorem 
  (counts : HouseholdCounts) 
  (sample : SampleSizes) : 
  counts.farmers = 1500 →
  counts.workers = 401 →
  counts.intellectuals = 99 →
  sample.farmers = 75 →
  sample.total = 100 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l446_44622


namespace NUMINAMATH_CALUDE_max_value_2x_3y_l446_44601

theorem max_value_2x_3y (x y : ℝ) (h : x^2 + y^2 = 16*x + 8*y + 20) :
  ∃ (M : ℝ), M = 33 ∧ 2*x + 3*y ≤ M ∧ ∃ (x₀ y₀ : ℝ), 2*x₀ + 3*y₀ = M ∧ x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 20 :=
sorry

end NUMINAMATH_CALUDE_max_value_2x_3y_l446_44601


namespace NUMINAMATH_CALUDE_line_through_midpoint_l446_44620

-- Define the points
def P : ℝ × ℝ := (1, 3)
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 6)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + y - 6 = 0

-- Theorem statement
theorem line_through_midpoint :
  (P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →  -- P is midpoint of AB
  (A.2 = 0) →  -- A is on x-axis
  (B.1 = 0) →  -- B is on y-axis
  (line_equation P.1 P.2) →  -- Line passes through P
  (∀ x y, line_equation x y ↔ 3 * x + y - 6 = 0) :=  -- Prove the line equation
by sorry

end NUMINAMATH_CALUDE_line_through_midpoint_l446_44620


namespace NUMINAMATH_CALUDE_even_number_of_solutions_l446_44683

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  (y^2 + 6) * (x - 1) = y * (x^2 + 1) ∧
  (x^2 + 6) * (y - 1) = x * (y^2 + 1)

/-- The set of solutions to the system -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | system p.1 p.2}

/-- The number of solutions is finite -/
axiom finite_solutions : Set.Finite solution_set

/-- Theorem: The system has an even number of real solutions -/
theorem even_number_of_solutions : ∃ n : ℕ, n % 2 = 0 ∧ Set.ncard solution_set = n := by
  sorry

end NUMINAMATH_CALUDE_even_number_of_solutions_l446_44683


namespace NUMINAMATH_CALUDE_inequality_solution_l446_44650

-- Define the parameter a
variable (a : ℝ)

-- Define the condition a < -1
variable (h : a < -1)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iio (-1) ∪ Set.Ioi (1/a))

-- State the theorem
theorem inequality_solution :
  ∀ x, (a * x - 1) / (x + 1) < 0 ↔ solution_set a x :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l446_44650


namespace NUMINAMATH_CALUDE_patio_layout_l446_44635

theorem patio_layout (r c : ℕ) : 
  r * c = 160 ∧ 
  (r + 4) * (c - 2) = 160 ∧ 
  r % 5 = 0 ∧ 
  c % 5 = 0 → 
  r = 16 := by sorry

end NUMINAMATH_CALUDE_patio_layout_l446_44635


namespace NUMINAMATH_CALUDE_total_seashells_is_fifty_l446_44668

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := 13

/-- The total number of seashells found by Tim and Sally -/
def total_seashells : ℕ := tim_seashells + sally_seashells

/-- Theorem: The total number of seashells found by Tim and Sally is 50 -/
theorem total_seashells_is_fifty : total_seashells = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_is_fifty_l446_44668


namespace NUMINAMATH_CALUDE_parallel_line_plane_not_all_parallel_l446_44603

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Define the specific objects
variable (a b : Line) (α : Plane)

-- State the theorem
theorem parallel_line_plane_not_all_parallel 
  (h1 : ¬ contains α b)
  (h2 : contains α a)
  (h3 : parallel b α) :
  ¬ (∀ (l : Line), contains α l → parallel_lines b l) := by
  sorry


end NUMINAMATH_CALUDE_parallel_line_plane_not_all_parallel_l446_44603


namespace NUMINAMATH_CALUDE_final_running_distance_l446_44679

/-- Calculates the final daily running distance after a 5-week program -/
theorem final_running_distance
  (initial_distance : ℕ)  -- Initial daily running distance in miles
  (increase_rate : ℕ)     -- Weekly increase in miles
  (increase_weeks : ℕ)    -- Number of weeks with distance increase
  (h1 : initial_distance = 3)
  (h2 : increase_rate = 1)
  (h3 : increase_weeks = 4)
  : initial_distance + increase_rate * increase_weeks = 7 :=
by sorry

end NUMINAMATH_CALUDE_final_running_distance_l446_44679


namespace NUMINAMATH_CALUDE_intersection_point_sum_l446_44646

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem intersection_point_sum (h1 : f (-3) = 3) (h2 : f 1 = 3)
  (h3 : ∃! p : ℝ × ℝ, f p.1 = f (p.1 - 4) ∧ f p.1 = p.2) :
  ∃ p : ℝ × ℝ, f p.1 = f (p.1 - 4) ∧ f p.1 = p.2 ∧ p.1 + p.2 = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l446_44646


namespace NUMINAMATH_CALUDE_remainder_of_N_mod_45_l446_44638

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

def N : ℕ := concatenate_integers 44

theorem remainder_of_N_mod_45 : N % 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_N_mod_45_l446_44638


namespace NUMINAMATH_CALUDE_at_least_one_is_one_l446_44691

theorem at_least_one_is_one (x y z : ℝ) 
  (h1 : (1 / x) + (1 / y) + (1 / z) = 1) 
  (h2 : 1 / (x + y + z) = 1) : 
  x = 1 ∨ y = 1 ∨ z = 1 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_is_one_l446_44691


namespace NUMINAMATH_CALUDE_train_length_l446_44653

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 56 → time_s = 9 → ∃ (length_m : ℝ), abs (length_m - 140.04) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l446_44653


namespace NUMINAMATH_CALUDE_quadratic_radical_equality_l446_44661

theorem quadratic_radical_equality (x y : ℚ) : 
  (x - y*x + y - 1 = 2 ∧ x + y - 1 = 3*x + 2*y - 4) → x*y = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equality_l446_44661


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l446_44686

theorem sqrt_equation_solution (x : ℝ) : 
  x > 2 → (Real.sqrt (8 * x) / Real.sqrt (4 * (x - 2)) = 5 / 2) → x = 50 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l446_44686


namespace NUMINAMATH_CALUDE_reach_power_of_three_l446_44606

/-- Represents the possible operations on the blackboard -/
inductive Operation
  | triple_minus_one : Operation  -- 3k - 1
  | double_plus_one : Operation   -- 2k + 1
  | half : Operation              -- k / 2

/-- Applies an operation to a number if the result is an integer -/
def apply_operation (k : ℤ) (op : Operation) : Option ℤ :=
  match op with
  | Operation.triple_minus_one => some (3 * k - 1)
  | Operation.double_plus_one => some (2 * k + 1)
  | Operation.half => if k % 2 = 0 then some (k / 2) else none

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def apply_sequence (n : ℤ) (seq : OperationSequence) : Option ℤ :=
  seq.foldl (fun acc op => acc.bind (fun k => apply_operation k op)) (some n)

/-- The main theorem -/
theorem reach_power_of_three (n : ℤ) (h : n ≥ 1) :
  ∃ (seq : OperationSequence), apply_sequence n seq = some (3^2023) :=
sorry

end NUMINAMATH_CALUDE_reach_power_of_three_l446_44606


namespace NUMINAMATH_CALUDE_group_meal_cost_l446_44639

/-- Calculates the total cost for a group to eat at a restaurant --/
def total_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Theorem: The total cost for the given group is $15 --/
theorem group_meal_cost : total_cost 12 7 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_group_meal_cost_l446_44639


namespace NUMINAMATH_CALUDE_number_of_math_classes_school_play_volunteers_l446_44698

/-- Given information about volunteers for a school Christmas play, prove the number of participating math classes. -/
theorem number_of_math_classes (total_needed : ℕ) (students_per_class : ℕ) (teachers : ℕ) (more_needed : ℕ) : ℕ :=
  let current_volunteers := total_needed - more_needed
  let x := (current_volunteers - teachers) / students_per_class
  x

/-- Prove that the number of math classes participating is 6. -/
theorem school_play_volunteers : number_of_math_classes 50 5 13 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_math_classes_school_play_volunteers_l446_44698


namespace NUMINAMATH_CALUDE_parallel_lines_unique_m_l446_44628

/-- Given two lines l₁ and l₂, prove that m = -4 is the only value that makes them parallel -/
theorem parallel_lines_unique_m : ∃! m : ℝ, 
  (∀ x y : ℝ, (m - 2) * x - 3 * y - 1 = 0 ↔ ((m - 2) / 3) * x - 1 / 3 = y) ∧ 
  (∀ x y : ℝ, m * x + (m + 2) * y + 1 = 0 ↔ (-m / (m + 2)) * x - 1 / (m + 2) = y) ∧
  ((m - 2) / 3 = -m / (m + 2)) ∧
  (m - 2 ≠ -m) ∧
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_unique_m_l446_44628


namespace NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l446_44610

theorem quadratic_polynomial_conditions (p : ℝ → ℝ) :
  (∀ x, p x = 1.8 * x^2 - 5.4 * x - 32.4) →
  p (-3) = 0 ∧ p 6 = 0 ∧ p 7 = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l446_44610


namespace NUMINAMATH_CALUDE_smallest_integer_sum_product_squares_l446_44631

theorem smallest_integer_sum_product_squares :
  ∃ (a : ℕ), a > 0 ∧ 
  (∃ (b : ℕ), 10 + a = b^2) ∧ 
  (∃ (c : ℕ), 10 * a = c^2) ∧
  (∀ (x : ℕ), x > 0 ∧ x < a → 
    (¬∃ (y : ℕ), 10 + x = y^2) ∨ 
    (¬∃ (z : ℕ), 10 * x = z^2)) ∧
  a = 90 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_sum_product_squares_l446_44631


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l446_44695

theorem arithmetic_calculations :
  ((-7) * (-5) - 90 / (-15) = 41) ∧
  ((-1)^10 * 2 - (-2)^3 / 4 = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l446_44695


namespace NUMINAMATH_CALUDE_problem_statement_l446_44681

theorem problem_statement (a : ℝ) (h : a^2 + a = 0) : 4*a^2 + 4*a + 2011 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l446_44681


namespace NUMINAMATH_CALUDE_length_AM_l446_44665

/-- Square ABCD with side length 9 -/
structure Square (A B C D : ℝ × ℝ) :=
  (side_length : ℝ)
  (is_square : side_length = 9)

/-- Point P on AB such that AP:PB = 7:2 -/
def P (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Quarter circle with center C and radius CB -/
def QuarterCircle (C B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Point E where tangent from P meets the quarter circle -/
def E (P : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- Point Q where tangent from P meets AD -/
def Q (P : ℝ × ℝ) (A D : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Point K where CE and DB meet -/
def K (C E D B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Point M where AK and PQ meet -/
def M (A K P Q : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem length_AM (A B C D : ℝ × ℝ) (square : Square A B C D) :
  let P := P A B
  let circle := QuarterCircle C B
  let E := E P circle
  let Q := Q P A D
  let K := K C E D B
  let M := M A K P Q
  distance A M = 85 / 22 := by
  sorry

end NUMINAMATH_CALUDE_length_AM_l446_44665


namespace NUMINAMATH_CALUDE_percentage_first_division_l446_44645

theorem percentage_first_division (total_students : ℕ) 
  (second_division_percent : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percent = 54 / 100 →
  just_passed = 48 →
  ∃ (first_division_percent : ℚ),
    first_division_percent + second_division_percent + (just_passed : ℚ) / total_students = 1 ∧
    first_division_percent = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_first_division_l446_44645


namespace NUMINAMATH_CALUDE_circle_properties_l446_44609

theorem circle_properties (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  (∃ k : ℝ, k = Real.sqrt 3 ∧ 
    (∀ t : ℝ, x ≠ 0 → y / x ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ / x₀ = k)) ∧
  (∃ k : ℝ, k = -Real.sqrt 3 ∧ 
    (∀ t : ℝ, x ≠ 0 → k ≤ y / x) ∧
    (∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ / x₀ = k)) ∧
  (∃ k : ℝ, k = -2 + Real.sqrt 6 ∧ 
    (∀ t : ℝ, y - x ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = k)) ∧
  (∃ k : ℝ, k = -2 - Real.sqrt 6 ∧ 
    (∀ t : ℝ, k ≤ y - x) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = k)) ∧
  (∃ k : ℝ, k = 7 + 4 * Real.sqrt 3 ∧ 
    (∀ t : ℝ, x^2 + y^2 ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = k)) ∧
  (∃ k : ℝ, k = 7 - 4 * Real.sqrt 3 ∧ 
    (∀ t : ℝ, k ≤ x^2 + y^2) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = k)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l446_44609


namespace NUMINAMATH_CALUDE_range_of_fraction_l446_44677

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  (1/6 : ℝ) ≤ x/y ∧ x/y ≤ (4/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l446_44677


namespace NUMINAMATH_CALUDE_stating_count_line_segments_correct_l446_44670

/-- Represents a regular n-sided convex polygon with n exterior points. -/
structure PolygonWithExteriorPoints (n : ℕ) where
  -- n ≥ 3 to ensure it's a valid polygon
  valid : n ≥ 3

/-- 
Calculates the number of line segments that can be drawn between all pairs 
of interior and exterior points of a regular n-sided convex polygon, 
excluding those connecting adjacent vertices.
-/
def countLineSegments (p : PolygonWithExteriorPoints n) : ℕ :=
  (n * (n - 3)) / 2 + n + n * (n - 3)

/-- 
Theorem stating that the number of line segments is correctly calculated 
by the formula (n(n-3)/2) + n + n(n-3).
-/
theorem count_line_segments_correct (p : PolygonWithExteriorPoints n) :
  countLineSegments p = (n * (n - 3)) / 2 + n + n * (n - 3) := by
  sorry

end NUMINAMATH_CALUDE_stating_count_line_segments_correct_l446_44670


namespace NUMINAMATH_CALUDE_part_one_part_two_l446_44608

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |2*x - 2*b| + 3

-- Part I
theorem part_one :
  let a : ℝ := 1
  let b : ℝ := 1
  {x : ℝ | f x a b > 8} = {x : ℝ | x < -1 ∨ x > 1.5} := by sorry

-- Part II
theorem part_two :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ x : ℝ, f x a b ≥ 5) →
  (∃ x : ℝ, f x a b = 5) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 →
    (∀ x : ℝ, f x a' b' ≥ 5) →
    (∃ x : ℝ, f x a' b' = 5) →
    1/a + 1/b ≤ 1/a' + 1/b') →
  1/a + 1/b = (3 + 2 * Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l446_44608


namespace NUMINAMATH_CALUDE_four_digit_solution_l446_44694

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def digit_value (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10^place)) % 10

def number_from_digits (a b c d : ℕ) : ℕ :=
  1000 * a + 100 * b + 10 * c + d

theorem four_digit_solution :
  let abcd := 2996
  let dcba := number_from_digits (digit_value abcd 0) (digit_value abcd 1) (digit_value abcd 2) (digit_value abcd 3)
  is_four_digit abcd ∧ is_four_digit dcba ∧ 2 * abcd + 1000 = dcba := by
  sorry

end NUMINAMATH_CALUDE_four_digit_solution_l446_44694


namespace NUMINAMATH_CALUDE_magnitude_of_one_minus_i_to_eighth_l446_44600

theorem magnitude_of_one_minus_i_to_eighth : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_one_minus_i_to_eighth_l446_44600


namespace NUMINAMATH_CALUDE_sample_standard_deviation_l446_44602

/-- Given a sample of 5 individuals with values a, 0, 1, 2, 3, where the average is 1,
    prove that the standard deviation of the sample is √2. -/
theorem sample_standard_deviation (a : ℝ) : 
  (a + 0 + 1 + 2 + 3) / 5 = 1 →
  Real.sqrt (((a - 1)^2 + (-1)^2 + 0^2 + 1^2 + 2^2) / 5) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sample_standard_deviation_l446_44602


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l446_44644

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p ^ k * (1 - p) ^ (n - k)

/-- The probability of exactly 5 successes in 7 independent trials with 3/4 probability of success in each trial is 5103/16384. -/
theorem chocolate_milk_probability :
  binomial_probability 7 5 (3/4) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l446_44644


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l446_44640

/-- Given a square with perimeter 48 meters, its area is 144 square meters. -/
theorem square_area_from_perimeter :
  ∀ s : ℝ,
  s > 0 →
  4 * s = 48 →
  s * s = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l446_44640


namespace NUMINAMATH_CALUDE_greatest_common_multiple_15_10_under_150_l446_44612

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_15_10_under_150 :
  ∃ (k : ℕ), k < 150 ∧ 
             is_common_multiple 15 10 k ∧ 
             ∀ (j : ℕ), j < 150 → is_common_multiple 15 10 j → j ≤ k :=
by
  use 120
  sorry

#eval 120  -- Expected output: 120

end NUMINAMATH_CALUDE_greatest_common_multiple_15_10_under_150_l446_44612


namespace NUMINAMATH_CALUDE_sin_theta_value_l446_44632

theorem sin_theta_value (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.sin (θ - π / 3) = 5 / 13) :
  Real.sin θ = (5 + 12 * Real.sqrt 3) / 26 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l446_44632


namespace NUMINAMATH_CALUDE_parabola_intercept_sum_l446_44619

/-- Parabola equation -/
def parabola (y : ℝ) : ℝ := y^2 - 4*y + 4

/-- X-intercept of the parabola -/
def a : ℝ := parabola 0

/-- Y-intercepts of the parabola -/
def b_and_c : Set ℝ := {y | parabola y = 0}

theorem parabola_intercept_sum :
  ∃ (b c : ℝ), b ∈ b_and_c ∧ c ∈ b_and_c ∧ a + b + c = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_intercept_sum_l446_44619


namespace NUMINAMATH_CALUDE_expression_evaluation_l446_44678

theorem expression_evaluation :
  let a : ℤ := -1
  let b : ℤ := 2
  3 * (a^2 * b + a * b^2) - 2 * (a^2 * b - 1) - 2 * a * b^2 - 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l446_44678


namespace NUMINAMATH_CALUDE_a_minus_c_values_l446_44634

theorem a_minus_c_values (a b c : ℕ) 
  (h1 : a > b) 
  (h2 : a^2 - a*b - a*c + b*c = 7) : 
  a - c = 1 ∨ a - c = 7 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_values_l446_44634


namespace NUMINAMATH_CALUDE_james_socks_count_l446_44614

/-- The number of pairs of red socks James has -/
def red_pairs : ℕ := 20

/-- The number of red socks James has -/
def red_socks : ℕ := red_pairs * 2

/-- The number of black socks James has -/
def black_socks : ℕ := red_socks / 2

/-- The number of red and black socks combined -/
def red_black_socks : ℕ := red_socks + black_socks

/-- The number of white socks James has -/
def white_socks : ℕ := red_black_socks * 2

/-- The total number of socks James has -/
def total_socks : ℕ := red_socks + black_socks + white_socks

theorem james_socks_count : total_socks = 180 := by
  sorry

end NUMINAMATH_CALUDE_james_socks_count_l446_44614


namespace NUMINAMATH_CALUDE_polygon_sides_l446_44647

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l446_44647


namespace NUMINAMATH_CALUDE_parabola_theorem_l446_44618

/-- A parabola passing through specific points with given conditions -/
def Parabola (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 1

theorem parabola_theorem (a b : ℝ) (m : ℝ) :
  Parabola a b 1 = -2 ∧ 
  Parabola a b (-2) = 13 ∧ 
  ∃ y₁ y₂ : ℝ, Parabola a b 5 = y₁ ∧ 
             Parabola a b m = y₂ ∧ 
             y₂ = 12 - y₁ ∧ 
             y₁ ≠ y₂ 
  → a = 1 ∧ b = -4 ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_theorem_l446_44618


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_tangent_sum_l446_44657

/-- If the sum of tangents of angle differences in a triangle is zero, then the triangle is isosceles. -/
theorem isosceles_triangle_from_tangent_sum (A B C : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_tangent_sum : Real.tan (A - B) + Real.tan (B - C) + Real.tan (C - A) = 0) : 
  A = B ∨ B = C ∨ C = A :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_tangent_sum_l446_44657


namespace NUMINAMATH_CALUDE_sum_of_triangles_34_l446_44673

/-- The triangle operation defined as a * b - c -/
def triangle_op (a b c : ℕ) : ℕ := a * b - c

/-- Theorem stating that the sum of two specific triangle operations equals 34 -/
theorem sum_of_triangles_34 : triangle_op 3 5 2 + triangle_op 4 6 3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangles_34_l446_44673


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l446_44617

theorem smallest_angle_solution : 
  ∃ x : ℝ, x > 0 ∧ 
    6 * Real.sin x * (Real.cos x)^3 - 6 * (Real.sin x)^3 * Real.cos x = 3 * Real.sqrt 3 / 2 ∧
    ∀ y : ℝ, y > 0 → 
      6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3 * Real.sqrt 3 / 2 → 
      x ≤ y ∧
    x = π / 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l446_44617


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_monotone_increasing_sufficiency_l446_44671

/-- A function f is monotonically increasing on an interval (a, +∞) if for any x₁, x₂ in the interval
    where x₁ < x₂, we have f(x₁) < f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The function f(x) = x^2 + mx - 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2

/-- Theorem: If f(x) = x^2 + mx - 2 is monotonically increasing on (2, +∞), then m ≥ -4 -/
theorem monotone_increasing_condition (m : ℝ) :
  MonotonicallyIncreasing (f m) 2 → m ≥ -4 := by
  sorry

/-- Theorem: If m ≥ -4, then f(x) = x^2 + mx - 2 is monotonically increasing on (2, +∞) -/
theorem monotone_increasing_sufficiency (m : ℝ) :
  m ≥ -4 → MonotonicallyIncreasing (f m) 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_monotone_increasing_sufficiency_l446_44671


namespace NUMINAMATH_CALUDE_k_of_five_eq_eight_point_five_l446_44629

noncomputable def h (x : ℝ) : ℝ := 5 / (3 - x)

noncomputable def h_inverse (x : ℝ) : ℝ := 3 - 5 / x

noncomputable def k (x : ℝ) : ℝ := 1 / (h_inverse x) + 8

theorem k_of_five_eq_eight_point_five : k 5 = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_k_of_five_eq_eight_point_five_l446_44629


namespace NUMINAMATH_CALUDE_product_ab_l446_44649

theorem product_ab (a b : ℚ) (h1 : 2 * a + 5 * b = 40) (h2 : 4 * a + 3 * b = 41) :
  a * b = 3315 / 98 := by
sorry

end NUMINAMATH_CALUDE_product_ab_l446_44649


namespace NUMINAMATH_CALUDE_john_uber_profit_l446_44697

/-- Calculates the net profit of an Uber driver given their income and expenses --/
def uberDriverNetProfit (grossIncome : ℕ) (carPurchasePrice : ℕ) (monthlyMaintenance : ℕ) 
  (maintenancePeriod : ℕ) (annualInsurance : ℕ) (tireReplacement : ℕ) (tradeInValue : ℕ) 
  (taxRate : ℚ) : ℤ :=
  let totalMaintenance := monthlyMaintenance * maintenancePeriod
  let taxAmount := (grossIncome : ℚ) * taxRate
  let totalExpenses := carPurchasePrice + totalMaintenance + annualInsurance + tireReplacement + taxAmount.ceil
  (grossIncome : ℤ) - (totalExpenses : ℤ) + (tradeInValue : ℤ)

/-- Theorem stating that John's net profit as an Uber driver is $6,300 --/
theorem john_uber_profit : 
  uberDriverNetProfit 30000 20000 300 12 1200 400 6000 (15/100) = 6300 := by
  sorry

end NUMINAMATH_CALUDE_john_uber_profit_l446_44697


namespace NUMINAMATH_CALUDE_exam_students_count_l446_44648

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
  (T = 80 * N) →
  ((T - 350) / (N - 5) = 90) →
  N = 10 := by
sorry

end NUMINAMATH_CALUDE_exam_students_count_l446_44648


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l446_44672

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 → c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l446_44672


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l446_44613

/-- Atomic weight of Copper in g/mol -/
def Cu_weight : ℝ := 63.546

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Number of Copper atoms in the compound -/
def Cu_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def C_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := Cu_count * Cu_weight + C_count * C_weight + O_count * O_weight

theorem compound_molecular_weight : molecular_weight = 123.554 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l446_44613


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l446_44664

def C : Set Nat := {67, 71, 72, 73, 79}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧
  (∀ (m : Nat), m ∈ C → ∀ (p : Nat), Nat.Prime p → p ∣ m →
    ∃ (q : Nat), q ∣ n ∧ Nat.Prime q ∧ q ≤ p) ∧
  n = 72 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l446_44664


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l446_44666

def y : ℕ := 2^3 * 3^3 * 4^4 * 5^5 * 6^6 * 7^7 * 8^8 * 11^3

theorem smallest_multiplier_for_perfect_square :
  ∃! k : ℕ, k > 0 ∧ 
  (∃ m : ℕ, k * y = m^2) ∧
  (∀ j : ℕ, j > 0 → j < k → ¬∃ n : ℕ, j * y = n^2) ∧
  k = 2310 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l446_44666


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l446_44615

theorem baker_cakes_sold (initial_cakes : ℕ) (bought_cakes : ℕ) (remaining_cakes : ℕ) :
  initial_cakes = 121 →
  bought_cakes = 170 →
  remaining_cakes = 186 →
  ∃ (sold_cakes : ℕ), sold_cakes = 105 ∧ initial_cakes - sold_cakes + bought_cakes = remaining_cakes :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l446_44615


namespace NUMINAMATH_CALUDE_fishing_earnings_l446_44624

/-- Calculates the total earnings from fishing over a period including a specific day --/
theorem fishing_earnings (rate : ℝ) (past_catch : ℝ) (today_multiplier : ℝ) :
  let past_earnings := rate * past_catch
  let today_catch := past_catch * today_multiplier
  let today_earnings := rate * today_catch
  let total_earnings := past_earnings + today_earnings
  (rate = 20 ∧ past_catch = 80 ∧ today_multiplier = 2) →
  total_earnings = 4800 :=
by
  sorry

end NUMINAMATH_CALUDE_fishing_earnings_l446_44624


namespace NUMINAMATH_CALUDE_carrot_weight_calculation_l446_44675

/-- The weight of carrots installed by the merchant -/
def carrot_weight : ℝ := sorry

/-- The total weight of all vegetables installed -/
def total_weight : ℝ := 36

/-- The weight of zucchini installed -/
def zucchini_weight : ℝ := 13

/-- The weight of broccoli installed -/
def broccoli_weight : ℝ := 8

/-- The weight of vegetables sold -/
def sold_weight : ℝ := 18

theorem carrot_weight_calculation :
  (carrot_weight + zucchini_weight + broccoli_weight = total_weight) ∧
  (total_weight = 2 * sold_weight) →
  carrot_weight = 15 := by sorry

end NUMINAMATH_CALUDE_carrot_weight_calculation_l446_44675


namespace NUMINAMATH_CALUDE_ahmed_has_thirteen_goats_l446_44699

/-- The number of goats Adam has -/
def adam_goats : ℕ := 7

/-- The number of goats Andrew has -/
def andrew_goats : ℕ := 5 + 2 * adam_goats

/-- The number of goats Ahmed has -/
def ahmed_goats : ℕ := andrew_goats - 6

/-- Theorem stating that Ahmed has 13 goats -/
theorem ahmed_has_thirteen_goats : ahmed_goats = 13 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_has_thirteen_goats_l446_44699


namespace NUMINAMATH_CALUDE_chord_inequality_l446_44605

/-- Given a semicircle with unit radius and four consecutive chords with lengths a, b, c, d,
    prove that a^2 + b^2 + c^2 + d^2 + abc + bcd < 4 -/
theorem chord_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hchords : ∃ (A B C D E : ℝ × ℝ), 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = c^2 ∧
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = d^2 ∧
    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧ 
    D.1^2 + D.2^2 = 1 ∧ E.1^2 + E.2^2 = 1 ∧
    (A.2 ≥ 0 ∧ B.2 ≥ 0 ∧ C.2 ≥ 0 ∧ D.2 ≥ 0 ∧ E.2 ≥ 0)) :
  a^2 + b^2 + c^2 + d^2 + a*b*c + b*c*d < 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_inequality_l446_44605


namespace NUMINAMATH_CALUDE_triangle_third_side_l446_44669

theorem triangle_third_side (a b : ℝ) (n : ℕ) : 
  a = 3.14 → b = 0.67 → 
  (n : ℝ) + b > a ∧ (n : ℝ) + a > b ∧ a + b > (n : ℝ) →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l446_44669


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l446_44633

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 2^(1/4))
    (h₂ : a₂ = 2^(1/6)) (h₃ : a₃ = 2^(1/12)) :
  let r := a₂ / a₁
  a₃ * r = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l446_44633


namespace NUMINAMATH_CALUDE_counterexample_exists_l446_44655

theorem counterexample_exists : ∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ ¬(Nat.Prime (p^2 - 2) ∧ Odd (p^2 - 2)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l446_44655


namespace NUMINAMATH_CALUDE_average_milk_production_per_cow_l446_44636

theorem average_milk_production_per_cow (num_cows : ℕ) (total_milk : ℕ) (num_days : ℕ) 
  (h_cows : num_cows = 40)
  (h_milk : total_milk = 12000)
  (h_days : num_days = 30) :
  total_milk / num_cows / num_days = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_milk_production_per_cow_l446_44636


namespace NUMINAMATH_CALUDE_correct_production_l446_44641

/-- Represents the production of each shift in a car manufacturing plant. -/
structure ShiftProduction where
  day : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given shift production satisfies the problem conditions. -/
def satisfiesConditions (p : ShiftProduction) : Prop :=
  p.day = 4 * p.second ∧
  p.third = (3 * p.second) / 2 ∧
  p.day + p.second + p.third = 8000

/-- Theorem stating that the given production numbers satisfy the problem conditions. -/
theorem correct_production : satisfiesConditions ⟨4923, 1231, 1846⟩ := by
  sorry

#check correct_production

end NUMINAMATH_CALUDE_correct_production_l446_44641


namespace NUMINAMATH_CALUDE_no_prime_solution_l446_44607

theorem no_prime_solution :
  ∀ p : ℕ, Prime p → 2 * p^3 - 5 * p + 14 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l446_44607


namespace NUMINAMATH_CALUDE_adult_tickets_correct_l446_44676

/-- The number of adult tickets sold at the Rotary Club's Omelet Breakfast --/
def adult_tickets : ℕ :=
  let small_children_tickets : ℕ := 53
  let older_children_tickets : ℕ := 35
  let senior_tickets : ℕ := 37
  let small_children_omelet : ℚ := 1/2
  let older_children_omelet : ℕ := 1
  let adult_omelet : ℕ := 2
  let senior_omelet : ℚ := 3/2
  let extra_omelets : ℕ := 25
  let total_eggs : ℕ := 584
  let eggs_per_omelet : ℕ := 3
  26

theorem adult_tickets_correct : adult_tickets = 26 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_correct_l446_44676


namespace NUMINAMATH_CALUDE_cookie_difference_l446_44625

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_choc : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def morning_raisin : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def morning_choc : ℕ := 237

/-- The total number of chocolate chip cookies Helen baked -/
def total_choc : ℕ := yesterday_choc + morning_choc

/-- The difference between chocolate chip cookies and raisin cookies -/
theorem cookie_difference : total_choc - morning_raisin = 25 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l446_44625


namespace NUMINAMATH_CALUDE_min_value_expression_l446_44690

theorem min_value_expression (x y : ℝ) : x^2 + y^2 - 6*x + 4*y + 18 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l446_44690


namespace NUMINAMATH_CALUDE_sum_product_equal_470_l446_44692

theorem sum_product_equal_470 : 
  (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equal_470_l446_44692


namespace NUMINAMATH_CALUDE_probability_odd_divisor_25_factorial_l446_44685

theorem probability_odd_divisor_25_factorial (n : ℕ) (h : n = 25) :
  let factorial := n.factorial
  let total_divisors := (factorial.divisors.filter (· > 0)).card
  let odd_divisors := (factorial.divisors.filter (λ d => d > 0 ∧ d % 2 = 1)).card
  (odd_divisors : ℚ) / total_divisors = 1 / 23 := by
  sorry

end NUMINAMATH_CALUDE_probability_odd_divisor_25_factorial_l446_44685


namespace NUMINAMATH_CALUDE_event_children_count_l446_44688

/-- Calculates the number of children at an event after adding more children --/
theorem event_children_count (total_guests men_count added_children : ℕ) : 
  total_guests = 80 →
  men_count = 40 →
  added_children = 10 →
  let women_count := men_count / 2
  let initial_children := total_guests - (men_count + women_count)
  initial_children + added_children = 30 := by
  sorry

#check event_children_count

end NUMINAMATH_CALUDE_event_children_count_l446_44688


namespace NUMINAMATH_CALUDE_sally_out_of_pocket_l446_44637

theorem sally_out_of_pocket (provided : ℕ) (book_cost : ℕ) (students : ℕ) :
  provided = 320 →
  book_cost = 12 →
  students = 30 →
  (students * book_cost - provided : ℤ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sally_out_of_pocket_l446_44637
