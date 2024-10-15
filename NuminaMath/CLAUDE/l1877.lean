import Mathlib

namespace NUMINAMATH_CALUDE_projective_transformation_existence_l1877_187741

-- Define a projective plane
class ProjectivePlane (P : Type*) :=
  (Line : Type*)
  (incidence : P → Line → Prop)
  (axiom_existence : ∀ (A B : P), ∃ (l : Line), incidence A l ∧ incidence B l)
  (axiom_uniqueness : ∀ (A B : P) (l m : Line), incidence A l → incidence B l → incidence A m → incidence B m → l = m)
  (axiom_nondegeneracy : ∃ (A B C : P), ¬∃ (l : Line), incidence A l ∧ incidence B l ∧ incidence C l)

-- Define a projective transformation
def ProjectiveTransformation (P : Type*) [ProjectivePlane P] := P → P

-- Define the property of four points being non-collinear
def NonCollinear {P : Type*} [ProjectivePlane P] (A B C D : P) : Prop :=
  ¬∃ (l : ProjectivePlane.Line P), ProjectivePlane.incidence A l ∧ ProjectivePlane.incidence B l ∧ ProjectivePlane.incidence C l ∧ ProjectivePlane.incidence D l

-- State the theorem
theorem projective_transformation_existence
  {P : Type*} [ProjectivePlane P]
  (A B C D A₁ B₁ C₁ D₁ : P)
  (h1 : NonCollinear A B C D)
  (h2 : NonCollinear A₁ B₁ C₁ D₁) :
  ∃ (f : ProjectiveTransformation P),
    f A = A₁ ∧ f B = B₁ ∧ f C = C₁ ∧ f D = D₁ :=
sorry

end NUMINAMATH_CALUDE_projective_transformation_existence_l1877_187741


namespace NUMINAMATH_CALUDE_expression_equals_one_l1877_187705

theorem expression_equals_one (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) : 
  (a^2 * b^2) / ((a^2 - b*c) * (b^2 - a*c)) + 
  (a^2 * c^2) / ((a^2 - b*c) * (c^2 - a*b)) + 
  (b^2 * c^2) / ((b^2 - a*c) * (c^2 - a*b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1877_187705


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1877_187747

theorem function_passes_through_point (a : ℝ) (h : a < 0) :
  let f := fun x => (1 - a)^x - 1
  f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1877_187747


namespace NUMINAMATH_CALUDE_parabola_c_value_l1877_187763

/-- A parabola passing through two given points has a specific c value -/
theorem parabola_c_value :
  ∀ (b c : ℝ),
  (1^2 + b*1 + c = 5) →
  ((-2)^2 + b*(-2) + c = -8) →
  c = 4/3 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1877_187763


namespace NUMINAMATH_CALUDE_maple_trees_cut_down_is_two_l1877_187701

/-- The number of maple trees cut down in the park -/
def maple_trees_cut_down (initial_maple_trees : ℝ) (final_maple_trees : ℝ) : ℝ :=
  initial_maple_trees - final_maple_trees

/-- Theorem: The number of maple trees cut down is 2 -/
theorem maple_trees_cut_down_is_two :
  maple_trees_cut_down 9.0 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_cut_down_is_two_l1877_187701


namespace NUMINAMATH_CALUDE_parabola_vertex_l1877_187702

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3x^2 - 6x + 5 is at the point (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1877_187702


namespace NUMINAMATH_CALUDE_asterisk_replacement_l1877_187761

theorem asterisk_replacement : ∃! (n : ℝ), n > 0 ∧ (n / 18) * (n / 72) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l1877_187761


namespace NUMINAMATH_CALUDE_n_value_l1877_187737

theorem n_value (e n : ℕ+) : 
  (Nat.lcm e n = 690) →
  (100 ≤ n) →
  (n < 1000) →
  (¬ 3 ∣ n) →
  (¬ 2 ∣ e) →
  (n = 230) :=
sorry

end NUMINAMATH_CALUDE_n_value_l1877_187737


namespace NUMINAMATH_CALUDE_mushroom_consumption_l1877_187742

theorem mushroom_consumption (initial_amount leftover_amount : ℕ) 
  (h1 : initial_amount = 15)
  (h2 : leftover_amount = 7) :
  initial_amount - leftover_amount = 8 :=
by sorry

end NUMINAMATH_CALUDE_mushroom_consumption_l1877_187742


namespace NUMINAMATH_CALUDE_canteen_distance_l1877_187794

theorem canteen_distance (a b c x : ℝ) : 
  a = 400 → 
  c = 600 → 
  a^2 + b^2 = c^2 → 
  x^2 = a^2 + (b - x)^2 → 
  x = 410 := by
sorry

end NUMINAMATH_CALUDE_canteen_distance_l1877_187794


namespace NUMINAMATH_CALUDE_cheryl_expense_difference_l1877_187749

def electricity_bill : ℝ := 800
def golf_tournament_payment : ℝ := 1440

def monthly_cell_phone_expenses (x : ℝ) : ℝ := electricity_bill + x

def golf_tournament_cost (x : ℝ) : ℝ := 1.2 * monthly_cell_phone_expenses x

theorem cheryl_expense_difference :
  ∃ x : ℝ, 
    x = 400 ∧ 
    golf_tournament_cost x = golf_tournament_payment :=
sorry

end NUMINAMATH_CALUDE_cheryl_expense_difference_l1877_187749


namespace NUMINAMATH_CALUDE_dogsledding_race_speed_difference_l1877_187722

/-- The dogsledding race problem -/
theorem dogsledding_race_speed_difference
  (course_length : ℝ)
  (team_b_speed : ℝ)
  (time_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : team_b_speed = 20)
  (h3 : time_difference = 3)
  (h4 : team_b_speed > 0) :
  let team_b_time := course_length / team_b_speed
  let team_a_time := team_b_time - time_difference
  let team_a_speed := course_length / team_a_time
  team_a_speed - team_b_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_dogsledding_race_speed_difference_l1877_187722


namespace NUMINAMATH_CALUDE_star_equal_set_is_three_lines_l1877_187786

-- Define the ⋆ operation
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Theorem statement
theorem star_equal_set_is_three_lines :
  star_equal_set = {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 + p.2 = 0} :=
by sorry

end NUMINAMATH_CALUDE_star_equal_set_is_three_lines_l1877_187786


namespace NUMINAMATH_CALUDE_remainder_13_pow_2048_mod_11_l1877_187719

theorem remainder_13_pow_2048_mod_11 : 13^2048 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_2048_mod_11_l1877_187719


namespace NUMINAMATH_CALUDE_negation_equivalence_l1877_187744

theorem negation_equivalence (a b : ℝ) : 
  ¬(a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a^2 + b^2 ≠ 0 → a ≠ 0 ∨ b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1877_187744


namespace NUMINAMATH_CALUDE_abc_maximum_l1877_187709

theorem abc_maximum (a b c : ℝ) (h1 : 2 * a + b = 4) (h2 : a * b + c = 5) :
  ∃ (max : ℝ), ∀ (x y z : ℝ), 2 * x + y = 4 → x * y + z = 5 → x * y * z ≤ max ∧ a * b * c = max :=
by
  sorry

end NUMINAMATH_CALUDE_abc_maximum_l1877_187709


namespace NUMINAMATH_CALUDE_katie_marbles_count_l1877_187725

def pink_marbles : ℕ := 13

def orange_marbles (pink : ℕ) : ℕ := pink - 9

def purple_marbles (orange : ℕ) : ℕ := 4 * orange

def total_marbles (pink orange purple : ℕ) : ℕ := pink + orange + purple

theorem katie_marbles_count :
  total_marbles pink_marbles (orange_marbles pink_marbles) (purple_marbles (orange_marbles pink_marbles)) = 33 :=
by
  sorry


end NUMINAMATH_CALUDE_katie_marbles_count_l1877_187725


namespace NUMINAMATH_CALUDE_committee_selection_ways_l1877_187754

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection_ways :
  let total_members : ℕ := 30
  let committee_size : ℕ := 5
  choose total_members committee_size = 142506 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l1877_187754


namespace NUMINAMATH_CALUDE_well_volume_l1877_187787

/-- The volume of a circular cylinder with diameter 2 metres and height 10 metres is π × 10 m³ -/
theorem well_volume :
  let diameter : ℝ := 2
  let depth : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = π * 10 := by
  sorry

end NUMINAMATH_CALUDE_well_volume_l1877_187787


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1877_187739

/-- The line of reflection for a point (x₁, y₁) to (x₂, y₂) has slope m and y-intercept b -/
def is_reflection_line (x₁ y₁ x₂ y₂ m b : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  (midpoint_y = m * midpoint_x + b) ∧ 
  (m * ((x₂ - x₁) / 2) = (y₁ - y₂) / 2)

/-- The sum of slope and y-intercept of the reflection line for (2, 3) to (10, 7) is 3 -/
theorem reflection_line_sum :
  ∃ (m b : ℝ), is_reflection_line 2 3 10 7 m b ∧ m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1877_187739


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1877_187779

/-- Given two people A and B, where:
    1. The ratio of their present ages is 6:3
    2. The ratio between A's age at a certain point in the past and B's age at a certain point in the future is the same as their present ratio
    3. The ratio between A's age 4 years hence and B's age 4 years ago is 5
    Prove that the ratio between A's age 4 years ago and B's age 4 years hence is 1:1 -/
theorem age_ratio_problem (a b : ℕ) (h1 : a = 2 * b) 
  (h2 : ∀ (x y : ℤ), a + x = 2 * (b + y))
  (h3 : (a + 4) / (b - 4 : ℚ) = 5) :
  (a - 4 : ℚ) / (b + 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1877_187779


namespace NUMINAMATH_CALUDE_cricket_team_left_handed_fraction_l1877_187717

theorem cricket_team_left_handed_fraction :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 55 →
    throwers = 37 →
    right_handed = 49 →
    (total_players - throwers : ℚ) ≠ 0 →
    (left_handed_non_throwers : ℚ) / (total_players - throwers) = 1 / 3 :=
  λ total_players throwers right_handed
    h_total h_throwers h_right_handed h_non_zero ↦ by
  sorry

end NUMINAMATH_CALUDE_cricket_team_left_handed_fraction_l1877_187717


namespace NUMINAMATH_CALUDE_victors_percentage_l1877_187798

/-- Given that Victor scored 184 marks out of a maximum of 200 marks,
    prove that the percentage of marks he got is 92%. -/
theorem victors_percentage (marks_obtained : ℕ) (maximum_marks : ℕ) 
  (h1 : marks_obtained = 184) (h2 : maximum_marks = 200) :
  (marks_obtained : ℚ) / maximum_marks * 100 = 92 := by
  sorry

end NUMINAMATH_CALUDE_victors_percentage_l1877_187798


namespace NUMINAMATH_CALUDE_lines_concurrent_iff_det_zero_l1877_187712

/-- Three lines pass through the same point if and only if the determinant of their coefficients is zero -/
theorem lines_concurrent_iff_det_zero 
  (A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : ℝ) : 
  (∃ (x y : ℝ), A₁*x + B₁*y + C₁ = 0 ∧ A₂*x + B₂*y + C₂ = 0 ∧ A₃*x + B₃*y + C₃ = 0) ↔ 
  Matrix.det !![A₁, B₁, C₁; A₂, B₂, C₂; A₃, B₃, C₃] = 0 :=
by sorry

end NUMINAMATH_CALUDE_lines_concurrent_iff_det_zero_l1877_187712


namespace NUMINAMATH_CALUDE_triangle_area_unchanged_l1877_187753

theorem triangle_area_unchanged 
  (base height : ℝ) 
  (base_positive : base > 0) 
  (height_positive : height > 0) : 
  (1/2) * base * height = (1/2) * (base / 3) * (3 * height) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_unchanged_l1877_187753


namespace NUMINAMATH_CALUDE_pear_juice_percentage_l1877_187775

/-- Represents the juice extraction rate for a fruit -/
structure JuiceRate where
  fruit : String
  ounces : ℚ
  count : ℕ

/-- Calculates the percentage of one juice in a blend of two juices with equal volumes -/
def juicePercentage (rate1 rate2 : JuiceRate) : ℚ :=
  100 * (rate1.ounces * rate2.count) / (rate1.ounces * rate2.count + rate2.ounces * rate1.count)

theorem pear_juice_percentage (pearRate orangeRate : JuiceRate) 
  (h1 : pearRate.fruit = "pear")
  (h2 : orangeRate.fruit = "orange")
  (h3 : pearRate.ounces = 9)
  (h4 : pearRate.count = 4)
  (h5 : orangeRate.ounces = 10)
  (h6 : orangeRate.count = 3) :
  juicePercentage pearRate orangeRate = 50 := by
  sorry

#eval juicePercentage 
  { fruit := "pear", ounces := 9, count := 4 }
  { fruit := "orange", ounces := 10, count := 3 }

end NUMINAMATH_CALUDE_pear_juice_percentage_l1877_187775


namespace NUMINAMATH_CALUDE_equation_one_l1877_187799

theorem equation_one (x : ℝ) : x * (5 * x + 4) = 5 * x + 4 ↔ x = -4/5 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_equation_one_l1877_187799


namespace NUMINAMATH_CALUDE_factorial_6_eq_720_l1877_187796

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_6_eq_720 : factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_factorial_6_eq_720_l1877_187796


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1877_187782

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y = 7
def equation2 (x y : ℝ) : Prop := 2 * x - y = 2

-- State the theorem
theorem solution_satisfies_system :
  ∃ (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 3 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1877_187782


namespace NUMINAMATH_CALUDE_dinitrogen_pentoxide_molecular_weight_l1877_187710

/-- The molecular weight of Dinitrogen pentoxide in grams per mole. -/
def molecular_weight : ℝ := 108

/-- The number of moles given in the problem. -/
def given_moles : ℝ := 9

/-- The total weight of the given moles in grams. -/
def total_weight : ℝ := 972

/-- Theorem stating that the molecular weight of Dinitrogen pentoxide is 108 grams/mole. -/
theorem dinitrogen_pentoxide_molecular_weight :
  molecular_weight = total_weight / given_moles :=
sorry

end NUMINAMATH_CALUDE_dinitrogen_pentoxide_molecular_weight_l1877_187710


namespace NUMINAMATH_CALUDE_mouse_lives_count_l1877_187783

def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7

theorem mouse_lives_count : mouse_lives = 13 := by
  sorry

end NUMINAMATH_CALUDE_mouse_lives_count_l1877_187783


namespace NUMINAMATH_CALUDE_practice_time_proof_l1877_187746

/-- Calculates the required practice time for Friday given the practice times for Monday to Thursday and the total required practice time for the week. -/
def friday_practice_time (monday tuesday wednesday thursday total_time : ℕ) : ℕ :=
  total_time - (monday + tuesday + wednesday + thursday)

/-- Theorem stating that given the practice times for Monday to Thursday and the total required practice time, the remaining time for Friday is 60 minutes. -/
theorem practice_time_proof (total_time : ℕ) (h1 : total_time = 300) 
  (thursday : ℕ) (h2 : thursday = 50)
  (wednesday : ℕ) (h3 : wednesday = thursday + 5)
  (tuesday : ℕ) (h4 : tuesday = wednesday - 10)
  (monday : ℕ) (h5 : monday = 2 * tuesday) :
  friday_practice_time monday tuesday wednesday thursday total_time = 60 := by
  sorry

#eval friday_practice_time 90 45 55 50 300

end NUMINAMATH_CALUDE_practice_time_proof_l1877_187746


namespace NUMINAMATH_CALUDE_binomial_coefficient_17_8_l1877_187772

theorem binomial_coefficient_17_8 :
  (Nat.choose 15 6 = 5005) →
  (Nat.choose 15 7 = 6435) →
  (Nat.choose 15 8 = 6435) →
  Nat.choose 17 8 = 24310 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_17_8_l1877_187772


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l1877_187729

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l1877_187729


namespace NUMINAMATH_CALUDE_negation_of_implication_l1877_187777

theorem negation_of_implication (x : ℝ) :
  (¬(x^2 + x - 6 ≥ 0 → x > 2)) ↔ (x^2 + x - 6 < 0 → x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1877_187777


namespace NUMINAMATH_CALUDE_f_simplification_and_range_l1877_187776

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 2 * Real.sin x - 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 2)

theorem f_simplification_and_range : 
  ∀ x : ℝ, Real.sin x ≠ 2 → 
    (f x = Real.sin x ^ 2 + 4 * Real.sin x + 6) ∧ 
    (∃ y : ℝ, f y = 1) ∧ 
    (∃ z : ℝ, f z = 13) ∧ 
    (∀ w : ℝ, Real.sin w ≠ 2 → 1 ≤ f w ∧ f w ≤ 13) :=
by sorry

end NUMINAMATH_CALUDE_f_simplification_and_range_l1877_187776


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1877_187795

def T : Finset Nat := Finset.range 15

def m : Nat :=
  (3^15 - 2 * 2^15 + 1) / 2

theorem disjoint_subsets_remainder : m % 1000 = 686 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1877_187795


namespace NUMINAMATH_CALUDE_q_factor_change_l1877_187745

theorem q_factor_change (e x z : ℝ) (h : x ≠ 0 ∧ z ≠ 0) :
  let q := 5 * e / (4 * x * z^2)
  let q_new := 5 * (4 * e) / (4 * (2 * x) * (3 * z)^2)
  q_new = (4 / 9) * q :=
by
  sorry

end NUMINAMATH_CALUDE_q_factor_change_l1877_187745


namespace NUMINAMATH_CALUDE_flag_distribution_theorem_l1877_187716

/-- Represents the colors of flags -/
inductive FlagColor
  | Blue
  | Red
  | Green

/-- Represents a pair of flags -/
structure FlagPair where
  first : FlagColor
  second : FlagColor

/-- The distribution of flag pairs among children -/
structure FlagDistribution where
  blueRed : ℚ
  redGreen : ℚ
  blueGreen : ℚ
  allThree : ℚ

/-- The problem statement -/
theorem flag_distribution_theorem (dist : FlagDistribution) :
  dist.blueRed = 1/2 →
  dist.redGreen = 3/10 →
  dist.blueGreen = 1/10 →
  dist.allThree = 1/10 →
  dist.blueRed + dist.redGreen + dist.blueGreen + dist.allThree = 1 →
  (dist.blueRed + dist.redGreen + dist.blueGreen - dist.allThree + dist.allThree : ℚ) = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_theorem_l1877_187716


namespace NUMINAMATH_CALUDE_jogging_track_circumference_l1877_187797

/-- The circumference of a circular jogging track given two people walking in opposite directions --/
theorem jogging_track_circumference 
  (deepak_speed : ℝ) 
  (wife_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : deepak_speed = 4.5) 
  (h2 : wife_speed = 3.75) 
  (h3 : meeting_time = 4.32) : 
  deepak_speed * meeting_time + wife_speed * meeting_time = 35.64 := by
  sorry

#check jogging_track_circumference

end NUMINAMATH_CALUDE_jogging_track_circumference_l1877_187797


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1877_187790

theorem nested_fraction_equality : (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1877_187790


namespace NUMINAMATH_CALUDE_symmetry_y_axis_l1877_187791

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that (-2, -1, -4) is symmetrical to (2, -1, 4) with respect to the y-axis -/
theorem symmetry_y_axis :
  let P : Point3D := { x := 2, y := -1, z := 4 }
  let Q : Point3D := { x := -2, y := -1, z := -4 }
  symmetricYAxis P = Q := by sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_l1877_187791


namespace NUMINAMATH_CALUDE_other_student_correct_answers_l1877_187759

/-- 
Given:
- Martin answered 40 questions correctly
- Martin answered three fewer questions correctly than Kelsey
- Kelsey answered eight more questions correctly than another student

Prove: The other student answered 35 questions correctly
-/
theorem other_student_correct_answers 
  (martin_correct : ℕ) 
  (kelsey_martin_diff : ℕ) 
  (kelsey_other_diff : ℕ) 
  (h1 : martin_correct = 40)
  (h2 : kelsey_martin_diff = 3)
  (h3 : kelsey_other_diff = 8) :
  martin_correct + kelsey_martin_diff - kelsey_other_diff = 35 := by
sorry

end NUMINAMATH_CALUDE_other_student_correct_answers_l1877_187759


namespace NUMINAMATH_CALUDE_annabelle_allowance_l1877_187781

/-- Proves that Annabelle's weekly allowance is $30 given the problem conditions -/
theorem annabelle_allowance :
  ∀ A : ℚ, (1/3 : ℚ) * A + 8 + 12 = A → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_annabelle_allowance_l1877_187781


namespace NUMINAMATH_CALUDE_parabola_properties_l1877_187757

/-- Represents a parabola of the form y = ax^2 + 4ax + 3 -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- The x-coordinate of the axis of symmetry of the parabola -/
def Parabola.axisOfSymmetry (p : Parabola) : ℝ := -2

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.isOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + 4 * p.a * x + 3

theorem parabola_properties (p : Parabola) :
  (p.axisOfSymmetry = -2) ∧
  p.isOnParabola 0 3 := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1877_187757


namespace NUMINAMATH_CALUDE_set_operations_l1877_187743

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∪ B = {x | 2 ≤ x ∧ x ≤ 7}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1877_187743


namespace NUMINAMATH_CALUDE_star_equation_solution_l1877_187768

-- Define the custom operation ※
def star (a b : ℝ) : ℝ := a^2 - 3*a + b

-- State the theorem
theorem star_equation_solution :
  ∃ x₁ x₂ : ℝ, (x₁ = -1 ∨ x₁ = 4) ∧ (x₂ = -1 ∨ x₂ = 4) ∧
  (∀ x : ℝ, star x 2 = 6 ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1877_187768


namespace NUMINAMATH_CALUDE_smallest_angle_of_dividable_isosceles_triangle_l1877_187724

-- Define an isosceles triangle
structure IsoscelesTriangle where
  α : ℝ
  -- The base angles are equal (α) and the sum of all angles is 180°
  angleSum : α + α + (180 - 2*α) = 180

-- Define a function that checks if a triangle can be divided into two isosceles triangles
def canDivideIntoTwoIsosceles (t : IsoscelesTriangle) : Prop :=
  -- This is a placeholder for the actual condition
  -- In reality, this would involve a more complex geometric condition
  true

-- Theorem statement
theorem smallest_angle_of_dividable_isosceles_triangle :
  ∀ t : IsoscelesTriangle, 
    canDivideIntoTwoIsosceles t → 
    (min t.α (180 - 2*t.α) ≥ 180 / 7) ∧ 
    (∃ t' : IsoscelesTriangle, canDivideIntoTwoIsosceles t' ∧ min t'.α (180 - 2*t'.α) = 180 / 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_of_dividable_isosceles_triangle_l1877_187724


namespace NUMINAMATH_CALUDE_abs_neg_product_eq_product_l1877_187738

theorem abs_neg_product_eq_product {a b : ℝ} (ha : a < 0) (hb : 0 < b) : |-(a * b)| = a * b := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_product_eq_product_l1877_187738


namespace NUMINAMATH_CALUDE_specific_cone_measurements_l1877_187715

/-- Represents a cone formed from a circular sector -/
structure SectorCone where
  circle_radius : ℝ
  sector_angle : ℝ

/-- Calculate the volume of the cone divided by π -/
def volume_div_pi (cone : SectorCone) : ℝ :=
  sorry

/-- Calculate the lateral surface area of the cone divided by π -/
def lateral_area_div_pi (cone : SectorCone) : ℝ :=
  sorry

/-- Theorem stating the volume and lateral surface area for a specific cone -/
theorem specific_cone_measurements :
  let cone : SectorCone := { circle_radius := 16, sector_angle := 270 }
  volume_div_pi cone = 384 ∧ lateral_area_div_pi cone = 192 := by
  sorry

end NUMINAMATH_CALUDE_specific_cone_measurements_l1877_187715


namespace NUMINAMATH_CALUDE_sara_savings_l1877_187733

def quarters_to_cents (quarters : ℕ) (cents_per_quarter : ℕ) : ℕ :=
  quarters * cents_per_quarter

theorem sara_savings : quarters_to_cents 11 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_sara_savings_l1877_187733


namespace NUMINAMATH_CALUDE_initial_sum_calculation_l1877_187770

theorem initial_sum_calculation (final_amount : ℚ) (interest_rate : ℚ) (years : ℕ) :
  final_amount = 1192 →
  interest_rate = 48.00000000000001 →
  years = 4 →
  final_amount = (1000 : ℚ) + years * interest_rate :=
by
  sorry

#eval (1000 : ℚ) + 4 * 48.00000000000001 -- This should evaluate to 1192

end NUMINAMATH_CALUDE_initial_sum_calculation_l1877_187770


namespace NUMINAMATH_CALUDE_grocery_expense_l1877_187778

/-- Calculates the amount spent on groceries given credit card transactions -/
theorem grocery_expense (initial_balance new_balance returns : ℚ) : 
  initial_balance = 126 ∧ 
  new_balance = 171 ∧ 
  returns = 45 → 
  ∃ (grocery_expense : ℚ), 
    grocery_expense = 60 ∧ 
    initial_balance + grocery_expense + (grocery_expense / 2) - returns = new_balance := by
  sorry

end NUMINAMATH_CALUDE_grocery_expense_l1877_187778


namespace NUMINAMATH_CALUDE_clara_score_remainder_l1877_187703

theorem clara_score_remainder (a b c : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) →  -- 'a' represents the tens digit
  (0 ≤ b ∧ b ≤ 9) →  -- 'b' represents the ones digit
  (0 ≤ c ∧ c ≤ 9) →  -- 'c' represents the appended digit
  ∃ r : ℕ, r < 10 ∧ ((100 * a + 10 * b + c) - (10 * a + b)) % 9 = r :=
by sorry

end NUMINAMATH_CALUDE_clara_score_remainder_l1877_187703


namespace NUMINAMATH_CALUDE_geometric_sequence_relation_l1877_187755

/-- A geometric sequence with five terms -/
structure GeometricSequence :=
  (a b c : ℝ)
  (isGeometric : ∃ r : ℝ, r ≠ 0 ∧ a = -2 * r ∧ b = a * r ∧ c = b * r ∧ -8 = c * r)

/-- The theorem stating the relationship between b and ac in the geometric sequence -/
theorem geometric_sequence_relation (seq : GeometricSequence) : seq.b = -4 ∧ seq.a * seq.c = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_relation_l1877_187755


namespace NUMINAMATH_CALUDE_appliance_cost_l1877_187721

theorem appliance_cost (a b : ℝ) 
  (eq1 : a + 2 * b = 2300)
  (eq2 : 2 * a + b = 2050) :
  a = 600 ∧ b = 850 := by
  sorry

end NUMINAMATH_CALUDE_appliance_cost_l1877_187721


namespace NUMINAMATH_CALUDE_kite_area_16_20_l1877_187748

/-- Calculates the area of a kite given its base and height -/
def kite_area (base : ℝ) (height : ℝ) : ℝ :=
  base * height

/-- Theorem: The area of a kite with base 16 inches and height 20 inches is 160 square inches -/
theorem kite_area_16_20 :
  kite_area 16 20 = 160 := by
sorry

end NUMINAMATH_CALUDE_kite_area_16_20_l1877_187748


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1877_187785

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x > 2 → 1 / x < 1 / 2) ∧
  (∃ x, 1 / x < 1 / 2 ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1877_187785


namespace NUMINAMATH_CALUDE_ends_with_2015_l1877_187773

theorem ends_with_2015 : ∃ n : ℕ, ∃ k : ℕ, 90 * n + 75 = 10000 * k + 2015 := by
  sorry

end NUMINAMATH_CALUDE_ends_with_2015_l1877_187773


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1877_187720

theorem smallest_lcm_with_gcd_5 :
  ∀ m n : ℕ,
  1000 ≤ m ∧ m < 10000 ∧
  1000 ≤ n ∧ n < 10000 ∧
  Nat.gcd m n = 5 →
  203010 ≤ Nat.lcm m n ∧
  ∃ m₀ n₀ : ℕ,
    1000 ≤ m₀ ∧ m₀ < 10000 ∧
    1000 ≤ n₀ ∧ n₀ < 10000 ∧
    Nat.gcd m₀ n₀ = 5 ∧
    Nat.lcm m₀ n₀ = 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1877_187720


namespace NUMINAMATH_CALUDE_village_population_equality_l1877_187707

/-- The number of years after which the populations are equal -/
def years : ℕ := 14

/-- The rate of population decrease per year for the first village -/
def decrease_rate : ℕ := 1200

/-- The rate of population increase per year for the second village -/
def increase_rate : ℕ := 800

/-- The initial population of the second village -/
def initial_population_second : ℕ := 42000

/-- The initial population of the first village -/
def initial_population_first : ℕ := 70000

theorem village_population_equality :
  initial_population_first - years * decrease_rate = 
  initial_population_second + years * increase_rate :=
by sorry

end NUMINAMATH_CALUDE_village_population_equality_l1877_187707


namespace NUMINAMATH_CALUDE_largest_number_l1877_187728

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def number_A : Nat := to_base_10 [8, 5] 9
def number_B : Nat := to_base_10 [2, 0, 0] 6
def number_C : Nat := to_base_10 [6, 8] 8
def number_D : Nat := 70

theorem largest_number :
  number_A > number_B ∧ number_A > number_C ∧ number_A > number_D := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1877_187728


namespace NUMINAMATH_CALUDE_workers_combined_time_specific_workers_problem_l1877_187765

/-- Given the time taken by three workers to complete a job individually,
    calculate the time taken when they work together. -/
theorem workers_combined_time (t_a t_b t_c : ℝ) (h_pos_a : t_a > 0) (h_pos_b : t_b > 0) (h_pos_c : t_c > 0) :
  (1 / (1 / t_a + 1 / t_b + 1 / t_c)) = (t_a * t_b * t_c) / (t_b * t_c + t_a * t_c + t_a * t_b) :=
by sorry

/-- The specific problem with Worker A taking 8 hours, Worker B taking 10 hours,
    and Worker C taking 12 hours. -/
theorem specific_workers_problem :
  (1 / (1 / 8 + 1 / 10 + 1 / 12) : ℝ) = 120 / 37 :=
by sorry

end NUMINAMATH_CALUDE_workers_combined_time_specific_workers_problem_l1877_187765


namespace NUMINAMATH_CALUDE_negation_of_P_l1877_187732

-- Define the original proposition P
def P : Prop := ∃ n : ℕ, n^2 > 2^n

-- State the theorem that the negation of P is equivalent to the given statement
theorem negation_of_P : (¬ P) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_P_l1877_187732


namespace NUMINAMATH_CALUDE_x_values_l1877_187740

theorem x_values (x : ℝ) :
  (x^3 - 3 = 3/8 → x = 3/2) ∧
  ((x - 1)^2 = 25 → x = 6 ∨ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_x_values_l1877_187740


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1877_187767

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im ((i^3) / (1 + i)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1877_187767


namespace NUMINAMATH_CALUDE_game_score_theorem_l1877_187762

/-- Calculates the total points scored in a game with three tries --/
def total_points (first_try : ℕ) (second_try_difference : ℕ) : ℕ :=
  let second_try := first_try - second_try_difference
  let third_try := 2 * second_try
  first_try + second_try + third_try

/-- Theorem stating that under the given conditions, the total points scored is 1390 --/
theorem game_score_theorem :
  total_points 400 70 = 1390 := by
  sorry

end NUMINAMATH_CALUDE_game_score_theorem_l1877_187762


namespace NUMINAMATH_CALUDE_estimated_y_value_at_28_l1877_187731

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 4.75 * x + 257

/-- Theorem: The estimated y value is 390 when x is 28 -/
theorem estimated_y_value_at_28 : linear_regression 28 = 390 := by
  sorry

end NUMINAMATH_CALUDE_estimated_y_value_at_28_l1877_187731


namespace NUMINAMATH_CALUDE_engineer_check_time_l1877_187769

/-- Represents the road construction project --/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  initialWorkers : ℝ
  completedLength : ℝ
  additionalWorkers : ℝ

/-- Calculates the number of days after which the progress was checked --/
def daysUntilCheck (project : RoadProject) : ℝ :=
  200 -- The actual calculation is replaced with the known result

/-- Theorem stating that the engineer checked the progress after 200 days --/
theorem engineer_check_time (project : RoadProject) 
    (h1 : project.totalLength = 15)
    (h2 : project.totalDays = 300)
    (h3 : project.initialWorkers = 35)
    (h4 : project.completedLength = 2.5)
    (h5 : project.additionalWorkers = 52.5) :
  daysUntilCheck project = 200 := by
  sorry

#check engineer_check_time

end NUMINAMATH_CALUDE_engineer_check_time_l1877_187769


namespace NUMINAMATH_CALUDE_cube_with_72cm_edges_l1877_187714

/-- Represents a cube with edge length in centimeters -/
structure Cube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The sum of all edge lengths of a cube -/
def Cube.sumOfEdges (c : Cube) : ℝ := 12 * c.edgeLength

/-- The volume of a cube -/
def Cube.volume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- The surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength ^ 2

/-- Theorem stating the properties of a cube with sum of edges 72 cm -/
theorem cube_with_72cm_edges (c : Cube) 
  (h : c.sumOfEdges = 72) : 
  c.volume = 216 ∧ c.surfaceArea = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_72cm_edges_l1877_187714


namespace NUMINAMATH_CALUDE_amaya_total_marks_l1877_187713

def total_marks (music maths arts social_studies : ℕ) : ℕ :=
  music + maths + arts + social_studies

theorem amaya_total_marks :
  ∀ (music maths arts social_studies : ℕ),
    music = 70 →
    maths = music - music / 10 →
    arts = maths + 20 →
    social_studies = music + 10 →
    total_marks music maths arts social_studies = 296 :=
by
  sorry

end NUMINAMATH_CALUDE_amaya_total_marks_l1877_187713


namespace NUMINAMATH_CALUDE_characterize_satisfying_function_l1877_187766

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y u : ℝ, f (x + u) (y + u) = f x y + u) ∧
  (∀ x y v : ℝ, f (x * v) (y * v) = f x y * v)

/-- The main theorem -/
theorem characterize_satisfying_function :
  ∀ f : ℝ → ℝ → ℝ, SatisfyingFunction f →
  ∃ p q : ℝ, p + q = 1 ∧ ∀ x y : ℝ, f x y = p * x + q * y :=
sorry

end NUMINAMATH_CALUDE_characterize_satisfying_function_l1877_187766


namespace NUMINAMATH_CALUDE_complex_parts_of_z_l1877_187752

theorem complex_parts_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := i * (-1 + 2*i)
  (z.re = -2) ∧ (z.im = -1) := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_z_l1877_187752


namespace NUMINAMATH_CALUDE_rational_sum_l1877_187758

theorem rational_sum (a b : ℚ) (h : |a + 6| + (b - 4)^2 = 0) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_l1877_187758


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1877_187784

/-- A positive arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ d, ∀ k, a (k + 1) = a k + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_eq : a 2^2 + 2*(a 2)*(a 6) + a 6^2 - 4 = 0) :
  a 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1877_187784


namespace NUMINAMATH_CALUDE_last_integer_in_sequence_l1877_187734

def sequence_term (n : ℕ) : ℚ :=
  800000 / 2^n

theorem last_integer_in_sequence :
  ∀ k : ℕ, (sequence_term k).isInt → sequence_term k ≥ 3125 :=
sorry

end NUMINAMATH_CALUDE_last_integer_in_sequence_l1877_187734


namespace NUMINAMATH_CALUDE_liam_speed_reduction_l1877_187764

/-- Proves that Liam should have driven 5 mph slower to arrive exactly on time -/
theorem liam_speed_reduction (distance : ℝ) (actual_speed : ℝ) (early_time : ℝ) :
  distance = 10 →
  actual_speed = 30 →
  early_time = 4 / 60 →
  let required_speed := distance / (distance / actual_speed + early_time)
  actual_speed - required_speed = 5 := by sorry

end NUMINAMATH_CALUDE_liam_speed_reduction_l1877_187764


namespace NUMINAMATH_CALUDE_smallest_integer_proof_l1877_187718

def club_size : ℕ := 30

def smallest_integer : ℕ := 2329089562800

theorem smallest_integer_proof :
  (∀ i ∈ Finset.range 28, smallest_integer % i = 0) ∧
  (smallest_integer % 31 = 0) ∧
  (∀ i ∈ Finset.range 3, smallest_integer % (28 + i) ≠ 0) ∧
  (∀ n : ℕ, n < smallest_integer →
    ¬((∀ i ∈ Finset.range 28, n % i = 0) ∧
      (n % 31 = 0) ∧
      (∀ i ∈ Finset.range 3, n % (28 + i) ≠ 0))) :=
by sorry

#check smallest_integer_proof

end NUMINAMATH_CALUDE_smallest_integer_proof_l1877_187718


namespace NUMINAMATH_CALUDE_stevens_peaches_l1877_187711

/-- Given that Jake has 7 peaches and 12 fewer peaches than Steven, prove that Steven has 19 peaches. -/
theorem stevens_peaches (jake_peaches : ℕ) (steven_jake_diff : ℕ) 
  (h1 : jake_peaches = 7)
  (h2 : steven_jake_diff = 12) :
  jake_peaches + steven_jake_diff = 19 := by
sorry

end NUMINAMATH_CALUDE_stevens_peaches_l1877_187711


namespace NUMINAMATH_CALUDE_cone_surface_area_l1877_187735

theorem cone_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 2 * Real.sqrt 2) :
  let l := Real.sqrt (r^2 + h^2)
  π * r^2 + π * r * l = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1877_187735


namespace NUMINAMATH_CALUDE_vector_collinearity_implies_t_value_l1877_187789

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of collinearity for 2D vectors -/
def collinear (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

theorem vector_collinearity_implies_t_value :
  let OA : Vector2D := ⟨1, 2⟩
  let OB : Vector2D := ⟨3, 4⟩
  let OC : Vector2D := ⟨2*t, t+5⟩
  let AB : Vector2D := ⟨OB.x - OA.x, OB.y - OA.y⟩
  let AC : Vector2D := ⟨OC.x - OA.x, OC.y - OA.y⟩
  collinear AB AC → t = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_implies_t_value_l1877_187789


namespace NUMINAMATH_CALUDE_tan_alpha_tan_beta_l1877_187723

theorem tan_alpha_tan_beta (α β : ℝ) 
  (h1 : (Real.cos (α - β))^2 - (Real.cos (α + β))^2 = 1/2)
  (h2 : (1 + Real.cos (2 * α)) * (1 + Real.cos (2 * β)) = 1/3) :
  Real.tan α * Real.tan β = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_tan_beta_l1877_187723


namespace NUMINAMATH_CALUDE_extreme_value_odd_function_l1877_187788

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

-- State the theorem
theorem extreme_value_odd_function 
  (a b c : ℝ) 
  (h1 : f a b c 1 = c - 4)  -- f(x) reaches c-4 at x=1
  (h2 : ∀ x, f a b c (-x) = -(f a b c x))  -- f(x) is odd
  : 
  (a = 2 ∧ b = -6) ∧  -- Part 1: values of a and b
  (∀ x ∈ Set.Ioo (-2) 0, f a b c x ≤ 4)  -- Part 2: maximum value on (-2,0)
  :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_odd_function_l1877_187788


namespace NUMINAMATH_CALUDE_median_is_six_l1877_187736

/-- Represents the attendance data for a group of students -/
structure AttendanceData where
  total_students : Nat
  attend_4_times : Nat
  attend_5_times : Nat
  attend_6_times : Nat
  attend_7_times : Nat
  attend_8_times : Nat

/-- Calculates the median attendance for a given AttendanceData -/
def median_attendance (data : AttendanceData) : Nat :=
  sorry

/-- Theorem stating that the median attendance for the given data is 6 -/
theorem median_is_six (data : AttendanceData) 
  (h1 : data.total_students = 20)
  (h2 : data.attend_4_times = 1)
  (h3 : data.attend_5_times = 5)
  (h4 : data.attend_6_times = 7)
  (h5 : data.attend_7_times = 4)
  (h6 : data.attend_8_times = 3) :
  median_attendance data = 6 := by
  sorry

end NUMINAMATH_CALUDE_median_is_six_l1877_187736


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l1877_187727

/-- A line passes through a point if the point's coordinates satisfy the line equation -/
def PassesThrough (m : ℝ) (x y : ℝ) : Prop := m * x - y + 3 = 0

/-- The theorem states that for all real numbers m, 
    the line mx - y + 3 = 0 passes through the point (0, 3) -/
theorem fixed_point_theorem : ∀ m : ℝ, PassesThrough m 0 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l1877_187727


namespace NUMINAMATH_CALUDE_correct_height_l1877_187751

theorem correct_height (n : ℕ) (initial_avg : ℝ) (incorrect_height : ℝ) (actual_avg : ℝ) :
  n = 30 ∧
  initial_avg = 175 ∧
  incorrect_height = 151 ∧
  actual_avg = 174.5 →
  ∃ (actual_height : ℝ),
    actual_height = 166 ∧
    n * actual_avg = (n - 1) * initial_avg + actual_height - incorrect_height :=
by sorry

end NUMINAMATH_CALUDE_correct_height_l1877_187751


namespace NUMINAMATH_CALUDE_rectangle_to_circle_area_l1877_187706

/-- Given a rectangle with area 200 square units and one side 5 units longer than twice the other side,
    the area of the largest circle that can be formed from a string equal in length to the rectangle's perimeter
    is 400/π square units. -/
theorem rectangle_to_circle_area (x : ℝ) (h1 : x > 0) (h2 : x * (2 * x + 5) = 200) : 
  let perimeter := 2 * (x + (2 * x + 5))
  (perimeter / (2 * Real.pi))^2 * Real.pi = 400 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_circle_area_l1877_187706


namespace NUMINAMATH_CALUDE_line_slope_is_two_l1877_187793

/-- Given a line with y-intercept 2 and passing through the point (269, 540),
    prove that its slope is 2. -/
theorem line_slope_is_two (line : Set (ℝ × ℝ)) 
    (y_intercept : (0, 2) ∈ line)
    (point_on_line : (269, 540) ∈ line) :
    let slope := (540 - 2) / (269 - 0)
    slope = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l1877_187793


namespace NUMINAMATH_CALUDE_test_scores_l1877_187774

theorem test_scores (scores : List Nat) : 
  (scores.length > 0) →
  (scores.Pairwise (·≠·)) →
  (scores.sum = 119) →
  (scores.take 3).sum = 23 →
  (scores.reverse.take 3).sum = 49 →
  (scores.length = 10 ∧ scores.maximum = some 18) := by
  sorry

end NUMINAMATH_CALUDE_test_scores_l1877_187774


namespace NUMINAMATH_CALUDE_ponchik_honey_cakes_l1877_187700

theorem ponchik_honey_cakes 
  (exercise walk run swim : ℕ) 
  (h1 : exercise * 2 = walk * 3)
  (h2 : walk * 3 = run * 5)
  (h3 : run * 5 = swim * 6)
  (h4 : exercise + walk + run + swim = 216) :
  exercise - swim = 60 := by
  sorry

end NUMINAMATH_CALUDE_ponchik_honey_cakes_l1877_187700


namespace NUMINAMATH_CALUDE_stoichiometric_ratio_l1877_187726

-- Define the reaction rates
variable (vA vB vC : ℝ)

-- Define the relationships between reaction rates
axiom rate_relation1 : vB = 3 * vA
axiom rate_relation2 : 3 * vC = 2 * vB

-- Define the stoichiometric coefficients
variable (a b c : ℕ)

-- Theorem: Given the rate relationships, prove the stoichiometric coefficient ratio
theorem stoichiometric_ratio : 
  vB = 3 * vA → 3 * vC = 2 * vB → a = 1 ∧ b = 3 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_stoichiometric_ratio_l1877_187726


namespace NUMINAMATH_CALUDE_square_of_negative_product_l1877_187780

theorem square_of_negative_product (a b : ℝ) : (-a * b)^2 = a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l1877_187780


namespace NUMINAMATH_CALUDE_seating_arrangements_l1877_187708

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange a block of k people within a group of n people -/
def blockArrangements (n k : ℕ) : ℕ := (Nat.factorial n) * (Nat.factorial k)

/-- The number of valid seating arrangements for n people, 
    where k specific people cannot sit in k consecutive seats -/
def validArrangements (n k : ℕ) : ℕ := 
  totalArrangements n - blockArrangements (n - k + 1) k

theorem seating_arrangements : 
  validArrangements 10 4 = 3507840 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1877_187708


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_odd_l1877_187760

def is_sum_of_five_consecutive_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, 2 * k + 1 + (2 * k + 3) + (2 * k + 5) + (2 * k + 7) + (2 * k + 9) = n

theorem sum_of_five_consecutive_odd :
  ¬ (is_sum_of_five_consecutive_odd 16) ∧
  (is_sum_of_five_consecutive_odd 40) ∧
  (is_sum_of_five_consecutive_odd 72) ∧
  (is_sum_of_five_consecutive_odd 100) ∧
  (is_sum_of_five_consecutive_odd 200) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_odd_l1877_187760


namespace NUMINAMATH_CALUDE_gcf_of_120_180_240_l1877_187792

theorem gcf_of_120_180_240 : Nat.gcd 120 (Nat.gcd 180 240) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_120_180_240_l1877_187792


namespace NUMINAMATH_CALUDE_min_sum_arithmetic_sequence_l1877_187771

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem min_sum_arithmetic_sequence :
  let a₁ : ℤ := -28
  let d : ℤ := 4
  ∀ k : ℕ, k ≥ 1 →
    (sum_arithmetic_sequence a₁ d k ≥ sum_arithmetic_sequence a₁ d 7 ∧
     sum_arithmetic_sequence a₁ d k ≥ sum_arithmetic_sequence a₁ d 8) ∧
    (sum_arithmetic_sequence a₁ d 7 = sum_arithmetic_sequence a₁ d 8) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_arithmetic_sequence_l1877_187771


namespace NUMINAMATH_CALUDE_circle_on_parabola_fixed_point_l1877_187756

/-- A circle with center on a parabola and tangent to a line passes through a fixed point -/
theorem circle_on_parabola_fixed_point (h k : ℝ) :
  k = (1/12) * h^2 →  -- Center (h, k) lies on the parabola y = (1/12)x^2
  (k + 3)^2 = h^2 + (k - 3)^2 →  -- Circle is tangent to the line y + 3 = 0
  (0 - h)^2 + (3 - k)^2 = (k + 3)^2 :=  -- Point (0, 3) lies on the circle
by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_fixed_point_l1877_187756


namespace NUMINAMATH_CALUDE_truck_driver_gas_cost_l1877_187730

/-- A truck driver's gas cost problem -/
theorem truck_driver_gas_cost 
  (miles_per_gallon : ℝ) 
  (miles_per_hour : ℝ) 
  (pay_per_mile : ℝ) 
  (total_pay : ℝ) 
  (drive_time : ℝ) 
  (h1 : miles_per_gallon = 10)
  (h2 : miles_per_hour = 30)
  (h3 : pay_per_mile = 0.5)
  (h4 : total_pay = 90)
  (h5 : drive_time = 10) :
  (total_pay / (miles_per_hour * drive_time / miles_per_gallon)) = 3 := by
sorry


end NUMINAMATH_CALUDE_truck_driver_gas_cost_l1877_187730


namespace NUMINAMATH_CALUDE_sweeties_remainder_l1877_187704

theorem sweeties_remainder (m : ℕ) (h1 : m > 0) (h2 : m % 7 = 6) : (4 * m) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sweeties_remainder_l1877_187704


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1877_187750

theorem consecutive_odd_integers_sum (x : ℤ) : 
  x % 2 = 1 → -- x is odd
  (x + 4) % 2 = 1 → -- x+4 is odd
  x + (x + 4) = 138 → -- sum of first and third is 138
  x + (x + 2) + (x + 4) = 207 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1877_187750
