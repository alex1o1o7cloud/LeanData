import Mathlib

namespace sandy_age_l2267_226756

/-- Given two people, Sandy and Molly, where Sandy is 18 years younger than Molly
    and the ratio of their ages is 7:9, prove that Sandy is 63 years old. -/
theorem sandy_age (sandy_age molly_age : ℕ) 
    (h1 : molly_age = sandy_age + 18)
    (h2 : sandy_age * 9 = molly_age * 7) : 
  sandy_age = 63 := by
sorry

end sandy_age_l2267_226756


namespace sqrt_sum_equals_seven_l2267_226770

theorem sqrt_sum_equals_seven (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end sqrt_sum_equals_seven_l2267_226770


namespace reciprocal_of_2023_l2267_226782

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_2023 : reciprocal 2023 = 1 / 2023 := by
  sorry

end reciprocal_of_2023_l2267_226782


namespace cricket_average_score_l2267_226729

/-- Given the average score for 10 matches and the average score for the first 6 matches,
    calculate the average score for the last 4 matches. -/
theorem cricket_average_score (total_matches : ℕ) (first_matches : ℕ) 
    (total_average : ℚ) (first_average : ℚ) :
  total_matches = 10 →
  first_matches = 6 →
  total_average = 389/10 →
  first_average = 41 →
  (total_average * total_matches - first_average * first_matches) / (total_matches - first_matches) = 143/4 :=
by sorry

end cricket_average_score_l2267_226729


namespace trace_equality_for_cubed_matrices_l2267_226751

open Matrix

theorem trace_equality_for_cubed_matrices
  (A B : Matrix (Fin 2) (Fin 2) ℝ)
  (h_not_commute : A * B ≠ B * A)
  (h_cubed_equal : A^3 = B^3) :
  ∀ n : ℕ, Matrix.trace (A^n) = Matrix.trace (B^n) :=
by sorry

end trace_equality_for_cubed_matrices_l2267_226751


namespace least_integer_absolute_value_l2267_226709

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3*y - 4| ≤ 25 → y ≥ -7) ∧ |3*(-7) - 4| ≤ 25 := by
  sorry

end least_integer_absolute_value_l2267_226709


namespace quadratic_function_unique_l2267_226737

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 2 * x^2 - 10 * x

-- State the theorem
theorem quadratic_function_unique :
  (∀ x, f x < 0 ↔ 0 < x ∧ x < 5) →
  (∀ x ∈ Set.Icc (-1) 4, f x ≤ 12) →
  (∃ x ∈ Set.Icc (-1) 4, f x = 12) →
  (∀ x, f x = 2 * x^2 - 10 * x) :=
by sorry

-- Note: The condition a < 0 is not used in this theorem as it's only relevant for part II of the original problem

end quadratic_function_unique_l2267_226737


namespace f_sum_eq_two_l2267_226723

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_eq_two :
  let f' := deriv f
  f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end f_sum_eq_two_l2267_226723


namespace percent_problem_l2267_226746

theorem percent_problem (x : ℝ) : 0.01 = (10 / 100) * x → x = 0.1 := by sorry

end percent_problem_l2267_226746


namespace helicopter_performance_l2267_226786

/-- Heights of helicopter A's performances in km -/
def heights_A : List ℝ := [3.6, -2.4, 2.8, -1.5, 0.9]

/-- Heights of helicopter B's performances in km -/
def heights_B : List ℝ := [3.8, -2, 4.1, -2.3]

/-- The highest altitude reached by helicopter A -/
def highest_altitude_A : ℝ := 3.6

/-- The final altitude of helicopter A after 5 performances -/
def final_altitude_A : ℝ := 3.4

/-- The required height change for helicopter B's 5th performance -/
def height_change_B : ℝ := -0.2

theorem helicopter_performance :
  (heights_A.maximum? = some highest_altitude_A) ∧
  (heights_A.sum = final_altitude_A) ∧
  (heights_B.sum + height_change_B = final_altitude_A) := by
  sorry

end helicopter_performance_l2267_226786


namespace total_students_agreed_l2267_226731

def third_grade_students : ℕ := 256
def fourth_grade_students : ℕ := 525
def third_grade_agreement_rate : ℚ := 60 / 100
def fourth_grade_agreement_rate : ℚ := 45 / 100

theorem total_students_agreed :
  ⌊third_grade_agreement_rate * third_grade_students⌋ +
  ⌊fourth_grade_agreement_rate * fourth_grade_students⌋ = 390 := by
  sorry

end total_students_agreed_l2267_226731


namespace triangle_angle_proof_l2267_226735

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m = (√3, -1) and n = (cos A, sin A), prove that if m ⊥ n and
    a * cos B + b * cos A = c * sin C, then B = π/6. -/
theorem triangle_angle_proof (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  m = (Real.sqrt 3, -1) →
  n = (Real.cos A, Real.sin A) →
  m.1 * n.1 + m.2 * n.2 = 0 →
  a * Real.cos B + b * Real.cos A = c * Real.sin C →
  B = π / 6 := by
sorry

end triangle_angle_proof_l2267_226735


namespace cos_pi_half_minus_two_alpha_l2267_226779

theorem cos_pi_half_minus_two_alpha (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (π / 2 - 2 * α) = 4 / 5 := by
  sorry

end cos_pi_half_minus_two_alpha_l2267_226779


namespace quadratic_root_triple_relation_l2267_226708

theorem quadratic_root_triple_relation (a b c : ℝ) (α β : ℝ) : 
  a ≠ 0 →
  a * α^2 + b * α + c = 0 →
  a * β^2 + b * β + c = 0 →
  β = 3 * α →
  3 * b^2 = 16 * a * c := by
  sorry

end quadratic_root_triple_relation_l2267_226708


namespace balloon_cost_difference_l2267_226792

/-- The cost of a helium balloon in dollars -/
def helium_cost : ℚ := 1.50

/-- The cost of a foil balloon in dollars -/
def foil_cost : ℚ := 2.50

/-- The number of helium balloons Allan bought -/
def allan_helium : ℕ := 2

/-- The number of foil balloons Allan bought -/
def allan_foil : ℕ := 3

/-- The number of helium balloons Jake bought -/
def jake_helium : ℕ := 4

/-- The number of foil balloons Jake bought -/
def jake_foil : ℕ := 2

/-- The total cost of Allan's balloons -/
def allan_total : ℚ := allan_helium * helium_cost + allan_foil * foil_cost

/-- The total cost of Jake's balloons -/
def jake_total : ℚ := jake_helium * helium_cost + jake_foil * foil_cost

/-- Theorem stating the difference in cost between Jake's and Allan's balloons -/
theorem balloon_cost_difference : jake_total - allan_total = 0.50 := by
  sorry

end balloon_cost_difference_l2267_226792


namespace min_b_value_l2267_226773

noncomputable section

variables (a b : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1

def g (x : ℝ) : ℝ := x^2 - 2 * b * x + 4 / 3

theorem min_b_value (h1 : a = 1/3) 
  (h2 : ∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 3, f x₁ ≥ g x₂) :
  b ≥ Real.sqrt 2 := by sorry

end min_b_value_l2267_226773


namespace problem_solution_l2267_226747

def f (x : ℝ) := |x + 1| + |x - 3|

theorem problem_solution :
  (∀ x : ℝ, f x < 6 ↔ -2 < x ∧ x < 4) ∧
  (∀ a : ℝ, (∃ x : ℝ, f x = |a - 2|) → (a ≥ 6 ∨ a ≤ -2)) := by
  sorry

end problem_solution_l2267_226747


namespace arithmetic_sequence_terms_l2267_226787

/-- An arithmetic sequence with first term 11, last term 101, and common difference 5 has 19 terms. -/
theorem arithmetic_sequence_terms : ∀ (a : ℕ → ℕ),
  (a 0 = 11) →  -- First term is 11
  (∀ n, a (n + 1) - a n = 5) →  -- Common difference is 5
  (∃ k, a k = 101) →  -- Last term is 101
  (∃ k, k = 19 ∧ a (k - 1) = 101) :=
by sorry

end arithmetic_sequence_terms_l2267_226787


namespace total_boys_across_grades_l2267_226774

/-- Represents the number of students in each grade level -/
structure GradeLevel where
  girls : ℕ
  boys : ℕ

/-- Calculates the total number of boys across all grade levels -/
def totalBoys (gradeA gradeB gradeC : GradeLevel) : ℕ :=
  gradeA.boys + gradeB.boys + gradeC.boys

/-- Theorem stating the total number of boys across three grade levels -/
theorem total_boys_across_grades (gradeA gradeB gradeC : GradeLevel) 
  (hA : gradeA.girls = 256 ∧ gradeA.girls = gradeA.boys + 52)
  (hB : gradeB.girls = 360 ∧ gradeB.boys = gradeB.girls - 40)
  (hC : gradeC.girls = 168 ∧ gradeC.boys = gradeC.girls) : 
  totalBoys gradeA gradeB gradeC = 692 := by
  sorry


end total_boys_across_grades_l2267_226774


namespace lovely_class_size_l2267_226707

/-- Proves that the number of students in Mrs. Lovely's class is 29 given the jelly bean distribution conditions. -/
theorem lovely_class_size :
  ∀ (g : ℕ),
  let b := g + 3
  let total_jelly_beans := 420
  let remaining_jelly_beans := 18
  let distributed_jelly_beans := total_jelly_beans - remaining_jelly_beans
  g * g + b * b = distributed_jelly_beans →
  g + b = 29 := by
  sorry

end lovely_class_size_l2267_226707


namespace line_intersects_x_axis_l2267_226757

theorem line_intersects_x_axis :
  ∃ (x : ℝ), 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by sorry

end line_intersects_x_axis_l2267_226757


namespace incorrect_expression_l2267_226732

theorem incorrect_expression (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬(b / a > 1) := by
  sorry

end incorrect_expression_l2267_226732


namespace length_AE_is_seven_l2267_226783

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (13, 14, 15)

-- Define the altitude from A
def altitude_A (t : Triangle) : ℝ × ℝ := sorry

-- Define point D where altitude intersects BC
def point_D (t : Triangle) : ℝ × ℝ := sorry

-- Define incircles of ABD and ACD
def incircle_ABD (t : Triangle) : Circle := sorry
def incircle_ACD (t : Triangle) : Circle := sorry

-- Define the common external tangent
def common_external_tangent (c1 c2 : Circle) : Line := sorry

-- Define point E where the common external tangent intersects AD
def point_E (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of AE
def length_AE (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem length_AE_is_seven (t : Triangle) :
  side_lengths t = (13, 14, 15) →
  length_AE t = 7 := by sorry

end length_AE_is_seven_l2267_226783


namespace closest_to_fraction_l2267_226788

def fraction : ℚ := 501 / (1 / 4)

def options : List ℤ := [1800, 1900, 2000, 2100, 2200]

theorem closest_to_fraction :
  (2000 : ℤ) = (options.argmin (λ x => |↑x - fraction|)).get
    (by sorry) := by sorry

end closest_to_fraction_l2267_226788


namespace curve_C_symmetric_about_y_axis_l2267_226794

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = Real.sqrt (p.1^2 + (p.2 - 4)^2)}

-- Define symmetry about y-axis
def symmetric_about_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem curve_C_symmetric_about_y_axis : symmetric_about_y_axis C := by
  sorry

end curve_C_symmetric_about_y_axis_l2267_226794


namespace min_days_to_plant_trees_l2267_226742

def trees_planted (n : ℕ) : ℕ := 2 * (2^n - 1)

theorem min_days_to_plant_trees : 
  ∀ n : ℕ, n > 0 → (trees_planted n ≥ 100 → n ≥ 6) ∧ (trees_planted 6 ≥ 100) :=
by sorry

end min_days_to_plant_trees_l2267_226742


namespace cow_count_l2267_226727

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- The total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- Theorem: If the total number of legs is 30 more than twice the number of heads,
    then there are 15 cows in the group -/
theorem cow_count (g : AnimalGroup) :
  totalLegs g = 2 * totalHeads g + 30 → g.cows = 15 := by
  sorry


end cow_count_l2267_226727


namespace part_one_part_two_l2267_226791

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part 1
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, (2 < x ∧ x < 3) → ∃ a : ℝ, a > 0 ∧ a < x ∧ x < 3*a) →
  ∃ a : ℝ, 1 ≤ a ∧ a ≤ 2 :=
sorry

end part_one_part_two_l2267_226791


namespace homework_difference_l2267_226784

def math_homework_pages : ℕ := 3
def reading_homework_pages : ℕ := 4

theorem homework_difference : reading_homework_pages - math_homework_pages = 1 := by
  sorry

end homework_difference_l2267_226784


namespace license_plate_ratio_l2267_226736

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate. -/
def old_letters : ℕ := 2

/-- The number of digits in an old license plate. -/
def old_digits : ℕ := 5

/-- The number of letters in a new license plate. -/
def new_letters : ℕ := 4

/-- The number of digits in a new license plate. -/
def new_digits : ℕ := 2

/-- The ratio of new possible license plates to old possible license plates. -/
theorem license_plate_ratio :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) =
  (num_letters ^ 2 : ℚ) / (num_digits ^ 3 : ℚ) := by
  sorry

end license_plate_ratio_l2267_226736


namespace students_without_A_count_l2267_226714

/-- Represents the number of students who received an A in a specific combination of subjects -/
structure GradeDistribution where
  total : Nat
  history : Nat
  math : Nat
  computing : Nat
  historyAndMath : Nat
  historyAndComputing : Nat
  mathAndComputing : Nat
  allThree : Nat

/-- Calculates the number of students who didn't receive an A in any subject -/
def studentsWithoutA (g : GradeDistribution) : Nat :=
  g.total - (g.history + g.math + g.computing - g.historyAndMath - g.historyAndComputing - g.mathAndComputing + g.allThree)

theorem students_without_A_count (g : GradeDistribution) 
  (h_total : g.total = 40)
  (h_history : g.history = 10)
  (h_math : g.math = 18)
  (h_computing : g.computing = 9)
  (h_historyAndMath : g.historyAndMath = 5)
  (h_historyAndComputing : g.historyAndComputing = 3)
  (h_mathAndComputing : g.mathAndComputing = 4)
  (h_allThree : g.allThree = 2) :
  studentsWithoutA g = 13 := by
  sorry

end students_without_A_count_l2267_226714


namespace expression_equality_l2267_226704

theorem expression_equality : (-3)^2 - Real.sqrt 4 + (1/2)⁻¹ = 9 := by
  sorry

end expression_equality_l2267_226704


namespace friday_ice_cream_amount_l2267_226712

/-- The amount of ice cream eaten on Friday night -/
def friday_ice_cream : ℝ := 3.5 - 0.25

/-- The total amount of ice cream eaten over two nights -/
def total_ice_cream : ℝ := 3.5

/-- The amount of ice cream eaten on Saturday night -/
def saturday_ice_cream : ℝ := 0.25

/-- Proof that the amount of ice cream eaten on Friday night is 3.25 pints -/
theorem friday_ice_cream_amount : friday_ice_cream = 3.25 := by
  sorry

end friday_ice_cream_amount_l2267_226712


namespace triangle_theorem_l2267_226744

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * cos (2 * t.C) + 2 * t.c * cos t.A * cos t.C + t.a + t.b = 0)
  (h2 : t.b = 4 * sin t.B) : 
  t.C = 2 * π / 3 ∧ 
  (∀ S : ℝ, S = 1/2 * t.a * t.b * sin t.C → S ≤ Real.sqrt 3) := by
  sorry

end triangle_theorem_l2267_226744


namespace wednesday_temperature_l2267_226750

/-- The temperature on Wednesday given the temperatures for the other days of the week and the average temperature --/
theorem wednesday_temperature
  (sunday : ℝ) (monday : ℝ) (tuesday : ℝ) (thursday : ℝ) (friday : ℝ) (saturday : ℝ) 
  (average : ℝ)
  (h_sunday : sunday = 40)
  (h_monday : monday = 50)
  (h_tuesday : tuesday = 65)
  (h_thursday : thursday = 82)
  (h_friday : friday = 72)
  (h_saturday : saturday = 26)
  (h_average : average = 53)
  : ∃ (wednesday : ℝ), 
    (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = average ∧ 
    wednesday = 36 := by
  sorry

end wednesday_temperature_l2267_226750


namespace abc_inequality_l2267_226767

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3 / (a * b * c) := by
  sorry

end abc_inequality_l2267_226767


namespace positive_real_solution_l2267_226706

theorem positive_real_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) :=
by sorry

end positive_real_solution_l2267_226706


namespace no_810_triple_l2267_226762

/-- Converts a list of digits in base 8 to a natural number -/
def fromBase8 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a list of digits in base 10 to a natural number -/
def fromBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 10 * acc + d) 0

/-- Checks if a number is an 8-10 triple -/
def is810Triple (n : Nat) : Prop :=
  n > 0 ∧ ∃ digits : List Nat, 
    (∀ d ∈ digits, d < 8) ∧
    fromBase8 digits = n ∧
    fromBase10 digits = 3 * n

theorem no_810_triple : ¬∃ n : Nat, is810Triple n := by
  sorry

end no_810_triple_l2267_226762


namespace arithmetic_calculation_l2267_226759

theorem arithmetic_calculation : (120 / 6 * 2 / 3 : ℚ) = 40 / 3 := by
  sorry

end arithmetic_calculation_l2267_226759


namespace f_increasing_iff_a_range_l2267_226772

/-- The function f(x) defined in terms of parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a / (a^2 - 2)) * (a^x - a^(-x))

/-- Theorem stating the conditions for f to be an increasing function -/
theorem f_increasing_iff_a_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (a > Real.sqrt 2 ∨ 0 < a ∧ a < 1) :=
sorry

end f_increasing_iff_a_range_l2267_226772


namespace pentagon_reconstruction_l2267_226768

-- Define the pentagon and extended points
variable (A B C D E A' A'' B' C' D' E' : ℝ × ℝ)

-- Define the conditions of the construction
axiom midpoint_AB' : A' = 2 * B - A
axiom midpoint_A'A'' : A'' = 2 * B' - A'
axiom midpoint_BC' : C' = 2 * C - B
axiom midpoint_CD' : D' = 2 * D - C
axiom midpoint_DE' : E' = 2 * E - D
axiom midpoint_EA' : A' = 2 * A - E

-- State the theorem
theorem pentagon_reconstruction :
  A = (1/31 : ℝ) • A' + (2/31 : ℝ) • A'' + (4/31 : ℝ) • B' + 
      (8/31 : ℝ) • C' + (16/31 : ℝ) • D' + (0 : ℝ) • E' :=
sorry

end pentagon_reconstruction_l2267_226768


namespace right_triangle_acute_angle_ratio_l2267_226745

theorem right_triangle_acute_angle_ratio (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Angles are positive
  α + β = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  β = 5 * α →      -- One angle is 5 times the other
  β = 75 := by
sorry

end right_triangle_acute_angle_ratio_l2267_226745


namespace trapezium_area_l2267_226793

theorem trapezium_area (a b h θ : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) (hθ : θ = 30 * π / 180) :
  (a + b) / 2 * (h * Real.sin θ) = 123.5 :=
sorry

end trapezium_area_l2267_226793


namespace garden_tree_distance_l2267_226766

/-- Calculates the distance between consecutive trees in a garden. -/
def distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : ℚ :=
  if num_trees > 1 then
    (yard_length : ℚ) / ((num_trees - 1) : ℚ)
  else
    0

/-- Proves that the distance between consecutive trees is 28 meters. -/
theorem garden_tree_distance :
  distance_between_trees 700 26 = 28 := by
  sorry

end garden_tree_distance_l2267_226766


namespace sine_is_odd_and_has_zero_point_l2267_226758

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to have a zero point
def has_zero_point (f : ℝ → ℝ) : Prop := ∃ x, f x = 0

theorem sine_is_odd_and_has_zero_point :
  is_odd Real.sin ∧ has_zero_point Real.sin :=
sorry

end sine_is_odd_and_has_zero_point_l2267_226758


namespace quadratic_equation_solution_l2267_226776

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -5
  let c : ℝ := 3
  let x₁ : ℝ := 3/2
  let x₂ : ℝ := 1
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by
  sorry


end quadratic_equation_solution_l2267_226776


namespace nonzero_terms_count_l2267_226760

/-- The number of nonzero terms in the expansion of (2x+3)(x^2 + 2x + 4) - 2(x^3 + x^2 - 3x + 1) + (x-2)(x+5) is 2 -/
theorem nonzero_terms_count (x : ℝ) : 
  let expansion := (2*x+3)*(x^2 + 2*x + 4) - 2*(x^3 + x^2 - 3*x + 1) + (x-2)*(x+5)
  ∃ (a b : ℝ), expansion = a*x^2 + b*x ∧ a ≠ 0 ∧ b ≠ 0 :=
by sorry

end nonzero_terms_count_l2267_226760


namespace fraction_meaningful_l2267_226715

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / ((x - 1) * (x + 2))) ↔ (x ≠ 1 ∧ x ≠ -2) := by sorry

end fraction_meaningful_l2267_226715


namespace pure_imaginary_product_l2267_226711

theorem pure_imaginary_product (m : ℝ) : 
  (Complex.I : ℂ).im * ((1 + m * Complex.I) * (1 - Complex.I)).re = 0 → m = -1 := by
  sorry

end pure_imaginary_product_l2267_226711


namespace tylers_age_l2267_226720

theorem tylers_age (tyler clay : ℕ) 
  (h1 : tyler = 3 * clay + 1) 
  (h2 : tyler + clay = 21) : 
  tyler = 16 := by
sorry

end tylers_age_l2267_226720


namespace fifth_term_value_l2267_226724

def a (n : ℕ+) : ℚ := n / (n^2 + 25)

theorem fifth_term_value : a 5 = 1 / 10 := by
  sorry

end fifth_term_value_l2267_226724


namespace meaningful_fraction_range_l2267_226748

theorem meaningful_fraction_range :
  ∀ x : ℝ, (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_fraction_range_l2267_226748


namespace smallest_integer_in_A_l2267_226739

def A : Set ℝ := {x | |x - 2| ≤ 5}

theorem smallest_integer_in_A : 
  ∃ (n : ℤ), (n : ℝ) ∈ A ∧ ∀ (m : ℤ), (m : ℝ) ∈ A → n ≤ m :=
by sorry

end smallest_integer_in_A_l2267_226739


namespace hyperbola_parabola_focus_l2267_226701

theorem hyperbola_parabola_focus (a : ℝ) : 
  a > 0 → 
  (∃ (x y : ℝ), x^2 - y^2 = a^2) → 
  (∃ (x y : ℝ), y^2 = 4*x) → 
  (∃ (c : ℝ), c > 0 ∧ c^2 - a^2 = a^2) →
  (∃ (f : ℝ × ℝ), f = (1, 0) ∧ f.1 = c) →
  a = Real.sqrt 2 / 2 := by
sorry

end hyperbola_parabola_focus_l2267_226701


namespace inscribed_squares_ratio_l2267_226778

/-- Given a circle with two inscribed squares:
    - The first square is inscribed in the circle
    - The second square is inscribed in the segment of the circle cut off by one side of the first square
    This theorem states that the ratio of the side lengths of these squares is 5:1 -/
theorem inscribed_squares_ratio (r : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (2 * a) ^ 2 + (2 * a) ^ 2 = (2 * r) ^ 2 →  -- First square inscribed in circle
  (a + 2 * b) ^ 2 + b ^ 2 = r ^ 2 →          -- Second square inscribed in segment
  a / b = 5 := by
  sorry

end inscribed_squares_ratio_l2267_226778


namespace pure_imaginary_complex_number_l2267_226740

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I = (0 : ℂ) + y*I ∧ y ≠ 0) → x = 2 :=
by
  sorry

end pure_imaginary_complex_number_l2267_226740


namespace max_pies_without_ingredients_l2267_226713

/-- Given a set of pies with specific ingredient distributions, 
    calculate the maximum number of pies without any of the specified ingredients -/
theorem max_pies_without_ingredients 
  (total_pies : ℕ) 
  (chocolate_pies : ℕ) 
  (blueberry_pies : ℕ) 
  (vanilla_pies : ℕ) 
  (almond_pies : ℕ) 
  (h_total : total_pies = 60)
  (h_chocolate : chocolate_pies ≥ 20)
  (h_blueberry : blueberry_pies = 45)
  (h_vanilla : vanilla_pies ≥ 24)
  (h_almond : almond_pies ≥ 6) :
  total_pies - blueberry_pies ≤ 15 :=
sorry

end max_pies_without_ingredients_l2267_226713


namespace extreme_value_and_monotonicity_l2267_226722

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem extreme_value_and_monotonicity :
  (f 1 = -1 ∧ f' 1 = 0) ∧
  (∀ x, x < -1 → f' x > 0) ∧
  (∀ x, x > 1 → f' x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
sorry

end extreme_value_and_monotonicity_l2267_226722


namespace jim_gas_spending_l2267_226743

/-- The amount of gas Jim bought in each state, in gallons -/
def gas_amount : ℝ := 10

/-- The price of gas per gallon in North Carolina, in dollars -/
def nc_price : ℝ := 2

/-- The additional price per gallon in Virginia compared to North Carolina, in dollars -/
def price_difference : ℝ := 1

/-- The total amount Jim spent on gas in both states -/
def total_spent : ℝ := gas_amount * nc_price + gas_amount * (nc_price + price_difference)

theorem jim_gas_spending :
  total_spent = 50 := by sorry

end jim_gas_spending_l2267_226743


namespace min_value_theorem_l2267_226755

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  x + 2 / (x - 2) ≥ 2 + 2 * Real.sqrt 2 ∧
  (x + 2 / (x - 2) = 2 + 2 * Real.sqrt 2 ↔ x = 2 + Real.sqrt 2) :=
by sorry

end min_value_theorem_l2267_226755


namespace custom_op_result_l2267_226771

def custom_op (a b : ℤ) : ℤ := b^2 - a*b

theorem custom_op_result : custom_op (custom_op (-1) 2) 3 = -9 := by
  sorry

end custom_op_result_l2267_226771


namespace product_of_difference_and_sum_of_squares_l2267_226721

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 10) 
  (h2 : a^2 + b^2 = 210) : 
  a * b = 55 := by
sorry

end product_of_difference_and_sum_of_squares_l2267_226721


namespace statement_implies_innocence_statement_proves_innocence_l2267_226710

-- Define the possible roles of a person
inductive Role
  | Knight
  | Liar
  | Normal

-- Define the statement made by the defendant
def statement (role : Role) (guilty : Bool) : Prop :=
  (role = Role.Knight ∧ ¬guilty) ∨ (role = Role.Liar ∧ guilty)

-- Define what it means to be a criminal
def isCriminal (role : Role) : Prop :=
  role = Role.Knight ∨ role = Role.Liar

-- Theorem: The statement implies innocence for all possible roles
theorem statement_implies_innocence (role : Role) :
  (∀ r, isCriminal r → (statement r true ↔ ¬statement r false)) →
  statement role false →
  ¬isCriminal role ∨ ¬statement role true :=
by sorry

-- The main theorem: The statement proves innocence
theorem statement_proves_innocence :
  ∀ role, ¬isCriminal role ∨ ¬statement role true :=
by sorry

end statement_implies_innocence_statement_proves_innocence_l2267_226710


namespace dot_product_calculation_l2267_226781

def vector_a : ℝ × ℝ := (-2, -6)

theorem dot_product_calculation (b : ℝ × ℝ) 
  (angle_condition : Real.cos (120 * π / 180) = -1/2)
  (magnitude_b : Real.sqrt ((b.1)^2 + (b.2)^2) = Real.sqrt 10) :
  (vector_a.1 * b.1 + vector_a.2 * b.2) = -10 := by
  sorry

end dot_product_calculation_l2267_226781


namespace inequality_always_true_implies_a_less_than_seven_l2267_226726

theorem inequality_always_true_implies_a_less_than_seven (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 4| > a) → a < 7 := by
  sorry

end inequality_always_true_implies_a_less_than_seven_l2267_226726


namespace sin_cos_sum_20_10_l2267_226799

theorem sin_cos_sum_20_10 : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_20_10_l2267_226799


namespace johns_umbrella_cost_l2267_226738

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas unit_cost : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * unit_cost

/-- Proof that John's total umbrella cost is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end johns_umbrella_cost_l2267_226738


namespace average_distance_is_17_l2267_226780

-- Define the distances traveled on each day
def monday_distance : ℝ := 12
def tuesday_distance : ℝ := 18
def wednesday_distance : ℝ := 21

-- Define the number of days
def num_days : ℝ := 3

-- Define the total distance
def total_distance : ℝ := monday_distance + tuesday_distance + wednesday_distance

-- Theorem: The average distance traveled per day is 17 miles
theorem average_distance_is_17 : total_distance / num_days = 17 := by
  sorry

end average_distance_is_17_l2267_226780


namespace two_distinct_roots_root_one_case_l2267_226775

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (m - 3) * x - m

-- Theorem stating that the equation has two distinct real roots for all m
theorem two_distinct_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0 :=
sorry

-- Theorem for the case when one root is 1
theorem root_one_case :
  ∃ m : ℝ, quadratic_equation m 1 = 0 ∧ 
  (∃ x : ℝ, x ≠ 1 ∧ quadratic_equation m x = 0) ∧
  m = 2 ∧
  (∃ x : ℝ, x = -2 ∧ quadratic_equation m x = 0) :=
sorry

end two_distinct_roots_root_one_case_l2267_226775


namespace correct_article_usage_l2267_226797

/-- Represents the possible article choices --/
inductive Article
  | A
  | The
  | None

/-- Represents a sentence with two article slots --/
structure Sentence where
  firstArticle : Article
  secondArticle : Article

/-- Checks if the article usage is correct for the given sentence --/
def isCorrectArticleUsage (s : Sentence) : Prop :=
  s.firstArticle = Article.A ∧ s.secondArticle = Article.None

/-- Theorem stating that the correct article usage is "a" for the first blank and no article for the second --/
theorem correct_article_usage :
  ∃ (s : Sentence), isCorrectArticleUsage s :=
sorry


end correct_article_usage_l2267_226797


namespace monotonic_decreasing_odd_function_property_l2267_226749

-- Define a monotonically decreasing function on ℝ
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Define an odd function on ℝ
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem monotonic_decreasing_odd_function_property
  (f : ℝ → ℝ) (h1 : MonoDecreasing f) (h2 : OddFunction f) :
  -f (-3) < f (-4) := by
  sorry

end monotonic_decreasing_odd_function_property_l2267_226749


namespace new_people_in_country_l2267_226733

theorem new_people_in_country (born : ℕ) (immigrated : ℕ) : 
  born = 90171 → immigrated = 16320 → born + immigrated = 106491 :=
by
  sorry

end new_people_in_country_l2267_226733


namespace polynomial_identity_l2267_226716

-- Define g as a polynomial function
variable (g : ℝ → ℝ)

-- State the theorem
theorem polynomial_identity 
  (h : ∀ x, g (x^2 + 2) = x^4 + 6*x^2 + 8) :
  ∀ x, g (x^2 - 1) = x^4 - 1 := by
  sorry

end polynomial_identity_l2267_226716


namespace cube_root_simplification_l2267_226728

theorem cube_root_simplification :
  (20^3 + 30^3 + 60^3 : ℝ)^(1/3) = 10 * 251^(1/3) := by
  sorry

end cube_root_simplification_l2267_226728


namespace river_current_speed_l2267_226719

/-- Represents the speed of a motorboat in various conditions -/
structure MotorboatSpeed where
  still : ℝ  -- Speed in still water
  current : ℝ  -- River current speed
  wind : ℝ  -- Wind speed (positive for tailwind, negative for headwind)

/-- Calculates the effective speed of the motorboat -/
def effectiveSpeed (s : MotorboatSpeed) : ℝ := s.still + s.current + s.wind

/-- Theorem: River current speed is 1 mile per hour -/
theorem river_current_speed 
  (distance : ℝ) 
  (downstream_time upstream_time : ℝ) 
  (h : distance = 24 ∧ downstream_time = 4 ∧ upstream_time = 6) 
  (s : MotorboatSpeed) 
  (h_downstream : effectiveSpeed { still := s.still, current := s.current, wind := -s.wind } * downstream_time = distance) 
  (h_upstream : effectiveSpeed { still := s.still, current := -s.current, wind := s.wind } * upstream_time = distance) :
  s.current = 1 := by
  sorry

end river_current_speed_l2267_226719


namespace john_mean_score_l2267_226789

def john_scores : List ℝ := [95, 88, 90, 92, 94, 89]

theorem john_mean_score : 
  (john_scores.sum / john_scores.length : ℝ) = 91.3333 := by
  sorry

end john_mean_score_l2267_226789


namespace chameleon_distance_l2267_226718

/-- A chameleon is a sequence of letters a, b, and c. -/
structure Chameleon (n : ℕ) where
  sequence : List Char
  length_eq : sequence.length = 3 * n
  count_a : sequence.count 'a' = n
  count_b : sequence.count 'b' = n
  count_c : sequence.count 'c' = n

/-- A swap is a transposition of two adjacent letters in a chameleon. -/
def swap (c : Chameleon n) (i : ℕ) : Chameleon n :=
  sorry

/-- The minimum number of swaps required to transform one chameleon into another. -/
def min_swaps (x y : Chameleon n) : ℕ :=
  sorry

/-- For any chameleon, there exists another chameleon that requires at least 3n²/2 swaps to reach. -/
theorem chameleon_distance (n : ℕ) (hn : 0 < n) (x : Chameleon n) :
  ∃ y : Chameleon n, 3 * n^2 / 2 ≤ min_swaps x y :=
  sorry

end chameleon_distance_l2267_226718


namespace range_of_f_l2267_226795

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = Set.Ici 3 :=
sorry

end range_of_f_l2267_226795


namespace odd_function_graph_point_l2267_226790

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_graph_point (f : ℝ → ℝ) (a : ℝ) :
  is_odd_function f → f (-a) = -f a :=
by
  sorry

end odd_function_graph_point_l2267_226790


namespace line_segment_param_sum_of_squares_l2267_226785

/-- Given a line segment connecting (-3,9) and (2,12), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (-3,9), prove that a^2 + b^2 + c^2 + d^2 = 124 -/
theorem line_segment_param_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = -3 ∧ d = 9) →
  (a + b = 2 ∧ c + d = 12) →
  a^2 + b^2 + c^2 + d^2 = 124 :=
by sorry


end line_segment_param_sum_of_squares_l2267_226785


namespace sector_to_cone_base_radius_l2267_226761

/-- Given a sector with central angle 120° and radius 9 cm, when formed into a cone,
    the radius of the base circle is 3 cm. -/
theorem sector_to_cone_base_radius (θ : ℝ) (R : ℝ) (r : ℝ) : 
  θ = 120 → R = 9 → r = (θ / 360) * R → r = 3 :=
by sorry

end sector_to_cone_base_radius_l2267_226761


namespace optimal_container_l2267_226777

-- Define the container parameters
def volume : ℝ := 8
def length : ℝ := 2
def min_height : ℝ := 3
def bottom_cost : ℝ := 40
def lateral_cost : ℝ := 20

-- Define the cost function
def cost (width height : ℝ) : ℝ :=
  bottom_cost * length * width + lateral_cost * (2 * (length + width) * height)

-- State the theorem
theorem optimal_container :
  ∃ (width height : ℝ),
    width > 0 ∧
    height ≥ min_height ∧
    length * width * height = volume ∧
    cost width height = 1520 / 3 ∧
    width = 4 / 3 ∧
    ∀ (w h : ℝ), w > 0 → h ≥ min_height → length * w * h = volume → cost w h ≥ 1520 / 3 := by
  sorry

end optimal_container_l2267_226777


namespace prob_no_adjacent_birch_value_l2267_226796

/-- The number of pine trees -/
def num_pine : ℕ := 6

/-- The number of cedar trees -/
def num_cedar : ℕ := 5

/-- The number of birch trees -/
def num_birch : ℕ := 7

/-- The total number of trees -/
def total_trees : ℕ := num_pine + num_cedar + num_birch

/-- The number of slots for birch trees -/
def num_slots : ℕ := num_pine + num_cedar + 1

/-- The probability of no two birch trees being adjacent when arranged randomly -/
def prob_no_adjacent_birch : ℚ := (num_slots.choose num_birch : ℚ) / (total_trees.choose num_birch)

theorem prob_no_adjacent_birch_value : prob_no_adjacent_birch = 1 / 40 := by
  sorry

end prob_no_adjacent_birch_value_l2267_226796


namespace new_cards_count_l2267_226705

def cards_per_page : ℕ := 3
def old_cards : ℕ := 10
def pages_used : ℕ := 6

theorem new_cards_count : 
  pages_used * cards_per_page - old_cards = 8 := by sorry

end new_cards_count_l2267_226705


namespace complex_modulus_inequality_l2267_226754

theorem complex_modulus_inequality (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  ‖z‖ ≤ |x| + |y| := by sorry

end complex_modulus_inequality_l2267_226754


namespace lady_bird_biscuits_l2267_226764

/-- The number of biscuits Lady Bird can make with a given amount of flour -/
def biscuits_from_flour (flour : ℚ) : ℚ :=
  (flour * 9) / (5/4)

/-- The number of biscuits per guest Lady Bird can allow -/
def biscuits_per_guest (total_biscuits : ℚ) (guests : ℕ) : ℚ :=
  total_biscuits / guests

theorem lady_bird_biscuits :
  let flour_used : ℚ := 5
  let num_guests : ℕ := 18
  let total_biscuits := biscuits_from_flour flour_used
  biscuits_per_guest total_biscuits num_guests = 2 := by
sorry

end lady_bird_biscuits_l2267_226764


namespace league_members_count_l2267_226717

/-- Represents the cost of items and total expenditure in the Rockham Soccer League --/
structure LeagueCosts where
  sock_cost : ℕ
  tshirt_cost_difference : ℕ
  set_discount : ℕ
  total_expenditure : ℕ

/-- Calculates the number of members in the Rockham Soccer League --/
def calculate_members (costs : LeagueCosts) : ℕ :=
  sorry

/-- Theorem stating that the number of members in the league is 150 --/
theorem league_members_count (costs : LeagueCosts)
  (h1 : costs.sock_cost = 5)
  (h2 : costs.tshirt_cost_difference = 6)
  (h3 : costs.set_discount = 3)
  (h4 : costs.total_expenditure = 3100) :
  calculate_members costs = 150 :=
sorry

end league_members_count_l2267_226717


namespace smallest_square_cover_l2267_226734

/-- The smallest square that can be covered by 3x4 rectangles -/
def smallest_square_side : ℕ := 12

/-- The area of a 3x4 rectangle -/
def rectangle_area : ℕ := 3 * 4

/-- The number of 3x4 rectangles needed to cover the smallest square -/
def num_rectangles : ℕ := smallest_square_side^2 / rectangle_area

theorem smallest_square_cover :
  ∀ (side : ℕ), 
  side % smallest_square_side = 0 →
  side^2 % rectangle_area = 0 →
  (side^2 / rectangle_area ≥ num_rectangles) ∧
  (num_rectangles * rectangle_area = smallest_square_side^2) :=
sorry

end smallest_square_cover_l2267_226734


namespace geometric_probability_models_l2267_226700

-- Define the characteristics of a geometric probability model
structure GeometricProbabilityModel where
  infiniteOutcomes : Bool
  equallyLikely : Bool

-- Define the four probability models
def model1 : GeometricProbabilityModel :=
  { infiniteOutcomes := true,
    equallyLikely := true }

def model2 : GeometricProbabilityModel :=
  { infiniteOutcomes := true,
    equallyLikely := true }

def model3 : GeometricProbabilityModel :=
  { infiniteOutcomes := false,
    equallyLikely := true }

def model4 : GeometricProbabilityModel :=
  { infiniteOutcomes := true,
    equallyLikely := true }

-- Function to check if a model is a geometric probability model
def isGeometricProbabilityModel (model : GeometricProbabilityModel) : Bool :=
  model.infiniteOutcomes ∧ model.equallyLikely

-- Theorem stating which models are geometric probability models
theorem geometric_probability_models :
  isGeometricProbabilityModel model1 ∧
  isGeometricProbabilityModel model2 ∧
  ¬isGeometricProbabilityModel model3 ∧
  isGeometricProbabilityModel model4 :=
sorry

end geometric_probability_models_l2267_226700


namespace sad_children_count_l2267_226763

theorem sad_children_count (total_children : ℕ) (happy_children : ℕ) (neither_happy_nor_sad : ℕ)
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neither_happy_nor_sad_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : neither_happy_nor_sad = 20)
  (h4 : boys = 17)
  (h5 : girls = 43)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_happy_nor_sad_boys = 5)
  (h9 : total_children = happy_children + neither_happy_nor_sad + (total_children - happy_children - neither_happy_nor_sad))
  (h10 : boys + girls = total_children) :
  total_children - happy_children - neither_happy_nor_sad = 10 := by
  sorry

end sad_children_count_l2267_226763


namespace isosceles_trapezoid_area_is_two_l2267_226741

/-- Represents an isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The number of small triangles along each base -/
  num_triangles : ℕ
  /-- The area of each small triangle -/
  small_triangle_area : ℝ
  /-- Assumption that each small triangle has an area of 1 -/
  h_area_is_one : small_triangle_area = 1

/-- Calculates the area of the isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  2 * t.num_triangles * t.small_triangle_area

/-- Theorem stating that the area of the isosceles trapezoid is 2 -/
theorem isosceles_trapezoid_area_is_two (t : IsoscelesTrapezoid) :
  trapezoid_area t = 2 := by
  sorry

#check isosceles_trapezoid_area_is_two

end isosceles_trapezoid_area_is_two_l2267_226741


namespace subtract_negative_l2267_226798

theorem subtract_negative : 2 - (-3) = 5 := by
  sorry

end subtract_negative_l2267_226798


namespace proposition_1_proposition_4_l2267_226702

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Axioms for the relations
axiom perpendicular_def (l : Line) (p : Plane) :
  perpendicular l p → ∀ (l' : Line), parallel_line_plane l' p → perpendicular_lines l l'

axiom parallel_plane_trans (p1 p2 p3 : Plane) :
  parallel_plane p1 p2 → parallel_plane p2 p3 → parallel_plane p1 p3

axiom perpendicular_parallel (l : Line) (p1 p2 : Plane) :
  perpendicular l p1 → parallel_plane p1 p2 → perpendicular l p2

-- Theorem 1
theorem proposition_1 (m n : Line) (α : Plane) :
  perpendicular m α → parallel_line_plane n α → perpendicular_lines m n := by sorry

-- Theorem 2
theorem proposition_4 (m : Line) (α β γ : Plane) :
  parallel_plane α β → parallel_plane β γ → perpendicular m α → perpendicular m γ := by sorry

end proposition_1_proposition_4_l2267_226702


namespace perpendicular_iff_m_eq_two_l2267_226730

/-- Two vectors in R² -/
def Vector2 := ℝ × ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Definition of vector AB -/
def AB (m : ℝ) : Vector2 := (m + 3, 2 * m + 1)

/-- Definition of vector CD -/
def CD (m : ℝ) : Vector2 := (m + 3, -5)

/-- Theorem stating that AB and CD are perpendicular if and only if m = 2 -/
theorem perpendicular_iff_m_eq_two :
  ∀ m : ℝ, dot_product (AB m) (CD m) = 0 ↔ m = 2 :=
sorry

end perpendicular_iff_m_eq_two_l2267_226730


namespace range_of_p_l2267_226752

noncomputable def p (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

theorem range_of_p :
  Set.range (fun (x : ℝ) ↦ p x) = Set.Ici 9 :=
by
  sorry

end range_of_p_l2267_226752


namespace marbles_redistribution_l2267_226725

/-- The number of marbles Tyrone gives to Eric -/
def marblesGiven : ℕ := 19

/-- The initial number of marbles Tyrone had -/
def tyronesInitial : ℕ := 120

/-- The initial number of marbles Eric had -/
def ericsInitial : ℕ := 15

/-- Proposition: The number of marbles Tyrone gave to Eric satisfies the conditions -/
theorem marbles_redistribution :
  (tyronesInitial - marblesGiven) = 3 * (ericsInitial + marblesGiven) ∧
  marblesGiven > 0 ∧
  marblesGiven < tyronesInitial :=
by
  sorry

#check marbles_redistribution

end marbles_redistribution_l2267_226725


namespace rotation_transform_triangles_l2267_226765

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Clockwise rotation around a point -/
def rotateClockwise (center : Point) (angle : ℝ) (p : Point) : Point :=
  sorry

/-- Check if two triangles are congruent -/
def areCongruent (t1 t2 : Triangle) : Prop :=
  sorry

theorem rotation_transform_triangles (m x y : ℝ) : 
  let ABC := Triangle.mk (Point.mk 0 0) (Point.mk 0 12) (Point.mk 16 0)
  let A'B'C' := Triangle.mk (Point.mk 24 18) (Point.mk 36 18) (Point.mk 24 2)
  let center := Point.mk x y
  0 < m → m < 180 →
  (areCongruent (Triangle.mk 
    (rotateClockwise center m ABC.A)
    (rotateClockwise center m ABC.B)
    (rotateClockwise center m ABC.C)) A'B'C') →
  m + x + y = 108 :=
sorry

end rotation_transform_triangles_l2267_226765


namespace trigonometric_sum_product_form_l2267_226769

open Real

theorem trigonometric_sum_product_form :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, cos (2 * x) + cos (6 * x) + cos (10 * x) + cos (14 * x) = 
      (a : ℝ) * cos (b * x) * cos (c * x) * cos (d * x)) ∧
    (a : ℕ) + b + c + d = 18 :=
by sorry

end trigonometric_sum_product_form_l2267_226769


namespace trajectory_of_P_l2267_226753

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The line equation -/
def line (k m x y : ℝ) : Prop := y = k*x + m

/-- Condition that k is not equal to ±2 -/
def k_condition (k : ℝ) : Prop := k ≠ 2 ∧ k ≠ -2

/-- The trajectory equation -/
def trajectory (x y : ℝ) : Prop := x^2/25 - 4*y^2/25 = 1

/-- Main theorem: The trajectory of point P -/
theorem trajectory_of_P (k m x y : ℝ) :
  k_condition k →
  (∃ (x₀ y₀ : ℝ), hyperbola x₀ y₀ ∧ line k m x₀ y₀) →
  y ≠ 0 →
  trajectory x y :=
sorry

end trajectory_of_P_l2267_226753


namespace list_price_problem_l2267_226703

theorem list_price_problem (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.30 * (list_price - 25)) → list_price = 35 := by
  sorry

end list_price_problem_l2267_226703
