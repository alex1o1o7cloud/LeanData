import Mathlib

namespace selected_number_in_fourth_group_l3758_375819

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Calculates the selected number for a given group in the systematic sampling -/
def selectedNumber (sampling : SystematicSampling) (groupIndex : Nat) : Nat :=
  sampling.startingNumber + (groupIndex - 1) * (sampling.totalStudents / sampling.sampleSize)

theorem selected_number_in_fourth_group (sampling : SystematicSampling) 
  (h1 : sampling.totalStudents = 1200)
  (h2 : sampling.sampleSize = 80)
  (h3 : sampling.startingNumber = 6) :
  selectedNumber sampling 4 = 51 := by
  sorry

end selected_number_in_fourth_group_l3758_375819


namespace fraction_subtraction_l3758_375809

theorem fraction_subtraction : (3 + 5 + 7) / (2 + 4 + 6) - (2 - 4 + 6) / (3 - 5 + 7) = 9 / 20 := by
  sorry

end fraction_subtraction_l3758_375809


namespace no_valid_n_for_ap_l3758_375842

theorem no_valid_n_for_ap : ¬∃ (n : ℕ), n > 1 ∧ 
  ∃ (a : ℤ), 136 = (n : ℤ) * (2 * a + (n - 1) * 3) / 2 := by
  sorry

end no_valid_n_for_ap_l3758_375842


namespace percent_of_y_l3758_375880

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 5 + (3 * y) / 10) / y = 0.7 := by
  sorry

end percent_of_y_l3758_375880


namespace modulus_of_z_l3758_375854

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (1 + Complex.I)^2 + 1

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 13 := by
  sorry

end modulus_of_z_l3758_375854


namespace ryan_chinese_hours_l3758_375864

/-- Represents the daily learning schedule for Ryan -/
structure LearningSchedule where
  english_hours : ℕ
  chinese_hours : ℕ
  total_days : ℕ
  total_hours : ℕ

/-- Theorem: Ryan spends 7 hours each day on learning Chinese -/
theorem ryan_chinese_hours (schedule : LearningSchedule) 
  (h1 : schedule.english_hours = 6)
  (h2 : schedule.total_days = 5)
  (h3 : schedule.total_hours = 65) :
  schedule.chinese_hours = 7 := by
sorry

end ryan_chinese_hours_l3758_375864


namespace volume_of_inscribed_cube_l3758_375831

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem volume_of_inscribed_cube (outer_cube_edge : ℝ) (h : outer_cube_edge = 16) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_edge : ℝ := sphere_diameter / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 12288 * Real.sqrt 3 / 27 := by
  sorry

end volume_of_inscribed_cube_l3758_375831


namespace mark_additional_spending_l3758_375850

-- Define Mark's initial amount
def initial_amount : ℚ := 180

-- Define the amount spent in the first store
def first_store_spent (initial : ℚ) : ℚ := (1/2 * initial) + 14

-- Define the amount spent in the second store before the additional spending
def second_store_initial_spent (initial : ℚ) : ℚ := 1/3 * initial

-- Theorem to prove
theorem mark_additional_spending :
  initial_amount - first_store_spent initial_amount - second_store_initial_spent initial_amount = 16 := by
  sorry

end mark_additional_spending_l3758_375850


namespace min_modulus_m_for_real_root_l3758_375805

theorem min_modulus_m_for_real_root (m : ℂ) : 
  (∃ x : ℝ, (1 + 2*I)*x^2 + m*x + (1 - 2*I) = 0) → 
  ∀ m' : ℂ, (∃ x : ℝ, (1 + 2*I)*x^2 + m'*x + (1 - 2*I) = 0) → 
  Complex.abs m ≥ 2 ∧ (Complex.abs m = 2 → Complex.abs m' ≥ Complex.abs m) :=
by sorry

end min_modulus_m_for_real_root_l3758_375805


namespace range_of_a_l3758_375856

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ({1, 2} : Set ℝ) → 3 * x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (proposition_p a ∧ proposition_q a) → (a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3)) :=
by sorry

end range_of_a_l3758_375856


namespace cylinder_surface_area_l3758_375843

/-- The total surface area of a cylinder with diameter 9 and height 15 is 175.5π -/
theorem cylinder_surface_area :
  let d : ℝ := 9  -- diameter
  let h : ℝ := 15 -- height
  let r : ℝ := d / 2 -- radius
  let base_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  let total_area : ℝ := 2 * base_area + lateral_area
  total_area = 175.5 * π := by
  sorry

end cylinder_surface_area_l3758_375843


namespace sequence_fourth_term_l3758_375816

theorem sequence_fourth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = 2 * a n - 2) : a 4 = 16 := by
  sorry

end sequence_fourth_term_l3758_375816


namespace sum_of_digits_of_second_smallest_divisible_by_all_less_than_8_l3758_375878

def is_divisible_by_all_less_than_8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → n % k = 0

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_second_smallest_divisible_by_all_less_than_8 :
  ∃ N : ℕ, second_smallest is_divisible_by_all_less_than_8 N ∧ sum_of_digits N = 12 :=
sorry

end sum_of_digits_of_second_smallest_divisible_by_all_less_than_8_l3758_375878


namespace positive_x_y_l3758_375858

theorem positive_x_y (x y : ℝ) (h1 : x - y < x) (h2 : x + y > y) : x > 0 ∧ y > 0 := by
  sorry

end positive_x_y_l3758_375858


namespace equation_roots_l3758_375861

-- Define the equation
def equation (x : ℝ) : Prop :=
  (21 / (x^2 - 9) - 3 / (x - 3) = 1)

-- Define the roots
def roots : Set ℝ := {-3, 7}

-- Theorem statement
theorem equation_roots :
  ∀ x : ℝ, x ∈ roots ↔ equation x ∧ x ≠ 3 ∧ x ≠ -3 :=
sorry

end equation_roots_l3758_375861


namespace fabric_cutting_l3758_375863

theorem fabric_cutting (initial_length : ℚ) (desired_length : ℚ) :
  initial_length = 2/3 →
  desired_length = 1/2 →
  initial_length - (initial_length / 4) = desired_length :=
by
  sorry

end fabric_cutting_l3758_375863


namespace coplanar_condition_l3758_375889

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the scalar m
variable (m : ℝ)

-- Define the condition for coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (A - D) = a • (B - D) + b • (C - D) + c • (0 : V)

-- State the theorem
theorem coplanar_condition (h : ∀ A B C D : V, 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + m • (D - O) = 0 → coplanar A B C D) : 
  m = -7 := by sorry

end coplanar_condition_l3758_375889


namespace sqrt_transformation_l3758_375823

theorem sqrt_transformation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end sqrt_transformation_l3758_375823


namespace AD_length_l3758_375891

-- Define the points A, B, C, D, and M
variable (A B C D M : Point)

-- Define the length function
variable (length : Point → Point → ℝ)

-- State the conditions
axiom equal_segments : length A B = length B C ∧ length B C = length C D ∧ length C D = length A D / 4
axiom M_midpoint : length A M = length M D
axiom MC_length : length M C = 7

-- State the theorem
theorem AD_length : length A D = 56 / 3 := by sorry

end AD_length_l3758_375891


namespace parabola_and_line_equations_l3758_375867

-- Define the parabola E
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the focus of the parabola
def focus : Point := ⟨1, 0⟩

-- Define the midpoint M
def M : Point := ⟨2, 1⟩

-- Define the property of A and B being on the parabola E
def on_parabola (E : Parabola) (p : Point) : Prop :=
  E.equation p.x p.y

-- Define the property of M being the midpoint of AB
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Theorem statement
theorem parabola_and_line_equations 
  (E : Parabola) (A B : Point) 
  (h1 : on_parabola E A) 
  (h2 : on_parabola E B) 
  (h3 : A ≠ B) 
  (h4 : is_midpoint M A B) :
  (∀ (x y : ℝ), E.equation x y ↔ y^2 = 4*x) ∧ 
  (∀ (x y : ℝ), (y - M.y = 2*(x - M.x)) ↔ (2*x - y - 3 = 0)) :=
sorry

end parabola_and_line_equations_l3758_375867


namespace organization_growth_l3758_375887

def population_growth (initial : ℕ) (years : ℕ) : ℕ :=
  match years with
  | 0 => initial
  | n + 1 => 3 * (population_growth initial n - 5) + 5

theorem organization_growth :
  population_growth 20 6 = 10895 := by
  sorry

end organization_growth_l3758_375887


namespace polar_to_cartesian_and_intersections_l3758_375899

/-- A circle in polar form -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- A line in polar form -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Given a circle C₁ and a line C₂ in polar form, 
    prove their Cartesian equations and intersection points -/
theorem polar_to_cartesian_and_intersections 
  (C₁ : PolarCircle) 
  (C₂ : PolarLine) 
  (h₁ : C₁.equation = fun ρ θ ↦ ρ = 4 * Real.sin θ)
  (h₂ : C₂.equation = fun ρ θ ↦ ρ * Real.cos (θ - π/4) = 2 * Real.sqrt 2) :
  ∃ (f₁ f₂ : ℝ → ℝ → Prop) (p₁ p₂ : PolarPoint),
    (∀ x y, f₁ x y ↔ x^2 + (y-2)^2 = 4) ∧
    (∀ x y, f₂ x y ↔ x + y = 4) ∧
    p₁ = ⟨4, π/2⟩ ∧
    p₂ = ⟨2 * Real.sqrt 2, π/4⟩ ∧
    C₁.equation p₁.ρ p₁.θ ∧
    C₂.equation p₁.ρ p₁.θ ∧
    C₁.equation p₂.ρ p₂.θ ∧
    C₂.equation p₂.ρ p₂.θ := by
  sorry

end polar_to_cartesian_and_intersections_l3758_375899


namespace continuous_function_zero_on_interval_l3758_375820

theorem continuous_function_zero_on_interval 
  (f : ℝ → ℝ) 
  (hf_cont : Continuous f) 
  (hf_eq : ∀ x, f (2 * x^2 - 1) = 2 * x * f x) : 
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x = 0 := by sorry

end continuous_function_zero_on_interval_l3758_375820


namespace inequality_with_cosine_condition_l3758_375868

theorem inequality_with_cosine_condition (α β : ℝ) 
  (h : Real.cos α * Real.cos β > 0) : 
  -(Real.tan (α/2))^2 ≤ (Real.tan ((β-α)/2)) / (Real.tan ((β+α)/2)) ∧
  (Real.tan ((β-α)/2)) / (Real.tan ((β+α)/2)) ≤ (Real.tan (β/2))^2 :=
by sorry

end inequality_with_cosine_condition_l3758_375868


namespace scissors_count_l3758_375836

/-- The number of scissors initially in the drawer -/
def initial_scissors : ℕ := 39

/-- The number of scissors Dan added to the drawer -/
def added_scissors : ℕ := 13

/-- The total number of scissors after Dan's addition -/
def total_scissors : ℕ := initial_scissors + added_scissors

/-- Theorem stating that the total number of scissors is 52 -/
theorem scissors_count : total_scissors = 52 := by
  sorry

end scissors_count_l3758_375836


namespace correct_equation_transformation_l3758_375846

theorem correct_equation_transformation :
  ∀ x : ℚ, 3 * x = -7 ↔ x = -7/3 := by
  sorry

end correct_equation_transformation_l3758_375846


namespace average_speed_ratio_l3758_375866

/-- Given that Eddy travels 480 km in 3 hours and Freddy travels 300 km in 4 hours,
    prove that the ratio of their average speeds is 32:15. -/
theorem average_speed_ratio (eddy_distance : ℝ) (eddy_time : ℝ) (freddy_distance : ℝ) (freddy_time : ℝ)
    (h1 : eddy_distance = 480)
    (h2 : eddy_time = 3)
    (h3 : freddy_distance = 300)
    (h4 : freddy_time = 4) :
    (eddy_distance / eddy_time) / (freddy_distance / freddy_time) = 32 / 15 := by
  sorry

#check average_speed_ratio

end average_speed_ratio_l3758_375866


namespace map_scale_conversion_l3758_375811

/-- Given a scale where 1 inch represents 500 feet, and a path measuring 6.5 inches on a map,
    the actual length of the path in feet is 3250. -/
theorem map_scale_conversion (scale : ℝ) (map_length : ℝ) (actual_length : ℝ) : 
  scale = 500 → map_length = 6.5 → actual_length = scale * map_length → actual_length = 3250 :=
by sorry

end map_scale_conversion_l3758_375811


namespace stating_max_principals_in_period_l3758_375808

/-- Represents the duration of the entire period in years -/
def total_period : ℕ := 10

/-- Represents the duration of each principal's term in years -/
def term_length : ℕ := 4

/-- Represents the maximum number of principals that can serve during the total period -/
def max_principals : ℕ := 3

/-- 
Theorem stating that given a total period of 10 years and principals serving 
exactly one 4-year term each, the maximum number of principals that can serve 
during this period is 3.
-/
theorem max_principals_in_period : 
  ∀ (num_principals : ℕ), 
  (num_principals * term_length ≥ total_period) → 
  (num_principals ≤ max_principals) :=
by sorry

end stating_max_principals_in_period_l3758_375808


namespace equation_solutions_l3758_375855

theorem equation_solutions :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end equation_solutions_l3758_375855


namespace quadratic_inequality_always_true_l3758_375847

theorem quadratic_inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + (a + 1) * x + 1 ≥ 0) → a = 1 := by
  sorry

end quadratic_inequality_always_true_l3758_375847


namespace fraction_problem_l3758_375824

theorem fraction_problem (x : ℚ) (h1 : x * 180 = 36) (h2 : x < 0.3) : x = 1/5 := by
  sorry

end fraction_problem_l3758_375824


namespace derek_walking_time_l3758_375832

/-- The time it takes Derek to walk a mile with his brother -/
def time_with_brother : ℝ := 12

/-- The time it takes Derek to walk a mile without his brother -/
def time_without_brother : ℝ := 9

/-- The additional time it takes to walk 20 miles with his brother -/
def additional_time : ℝ := 60

theorem derek_walking_time :
  time_with_brother = 12 ∧
  time_without_brother * 20 + additional_time = time_with_brother * 20 :=
sorry

end derek_walking_time_l3758_375832


namespace soccer_tournament_points_l3758_375859

theorem soccer_tournament_points (n k : ℕ) (h1 : n ≥ 3) (h2 : 2 ≤ k) (h3 : k ≤ n - 1) :
  let min_points := 3 * n - (3 * k + 1) / 2 - 2
  ∀ (team_points : ℕ → ℕ),
    (∀ i, i < n → team_points i ≤ 3 * (n - 1)) →
    (∀ i j, i < n → j < n → i ≠ j → 
      team_points i + team_points j ≥ 1 ∧ team_points i + team_points j ≤ 4) →
    (∀ i, i < n → team_points i ≥ min_points) →
    ∃ (top_teams : Finset ℕ),
      top_teams.card ≤ k ∧
      ∀ j, j < n → j ∉ top_teams → team_points j < min_points :=
by sorry


end soccer_tournament_points_l3758_375859


namespace range_of_m_l3758_375874

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≤ 4 := by
  sorry

end range_of_m_l3758_375874


namespace same_grade_percentage_l3758_375830

/-- Represents the grade distribution for two assignments --/
structure GradeDistribution :=
  (aa ab ac ad : ℕ)
  (ba bb bc bd : ℕ)
  (ca cb cc cd : ℕ)
  (da db dc dd : ℕ)

/-- The total number of students --/
def totalStudents : ℕ := 40

/-- The grade distribution for the English class --/
def englishClassDistribution : GradeDistribution :=
  { aa := 3, ab := 2, ac := 1, ad := 0,
    ba := 1, bb := 6, bc := 3, bd := 1,
    ca := 0, cb := 2, cc := 7, cd := 2,
    da := 0, db := 1, dc := 2, dd := 2 }

/-- Calculates the number of students who received the same grade on both assignments --/
def sameGradeCount (dist : GradeDistribution) : ℕ :=
  dist.aa + dist.bb + dist.cc + dist.dd

/-- Theorem: The percentage of students who received the same grade on both assignments is 45% --/
theorem same_grade_percentage :
  (sameGradeCount englishClassDistribution : ℚ) / totalStudents * 100 = 45 := by
  sorry

end same_grade_percentage_l3758_375830


namespace total_puppies_l3758_375862

/-- Given an initial number of puppies and a number of additional puppies,
    prove that the total number of puppies is equal to the sum of the initial number
    and the additional number. -/
theorem total_puppies (initial_puppies additional_puppies : ℝ) :
  initial_puppies + additional_puppies = initial_puppies + additional_puppies :=
by sorry

end total_puppies_l3758_375862


namespace max_notebooks_is_11_l3758_375841

def single_notebook_cost : ℕ := 2
def pack_4_cost : ℕ := 6
def pack_7_cost : ℕ := 9
def total_money : ℕ := 15
def max_pack_7 : ℕ := 1

def notebooks_count (singles pack_4 pack_7 : ℕ) : ℕ :=
  singles + 4 * pack_4 + 7 * pack_7

def total_cost (singles pack_4 pack_7 : ℕ) : ℕ :=
  single_notebook_cost * singles + pack_4_cost * pack_4 + pack_7_cost * pack_7

theorem max_notebooks_is_11 :
  ∃ (singles pack_4 pack_7 : ℕ),
    notebooks_count singles pack_4 pack_7 = 11 ∧
    total_cost singles pack_4 pack_7 ≤ total_money ∧
    pack_7 ≤ max_pack_7 ∧
    ∀ (s p4 p7 : ℕ),
      total_cost s p4 p7 ≤ total_money →
      p7 ≤ max_pack_7 →
      notebooks_count s p4 p7 ≤ 11 :=
by sorry

end max_notebooks_is_11_l3758_375841


namespace mr_a_speed_l3758_375876

/-- Proves that Mr. A's speed is 30 kmph given the problem conditions --/
theorem mr_a_speed (initial_distance : ℝ) (mrs_a_speed : ℝ) (bee_speed : ℝ) (bee_distance : ℝ)
  (h1 : initial_distance = 120)
  (h2 : mrs_a_speed = 10)
  (h3 : bee_speed = 60)
  (h4 : bee_distance = 180) :
  ∃ (mr_a_speed : ℝ), mr_a_speed = 30 ∧ 
    (bee_distance / bee_speed) * (mr_a_speed + mrs_a_speed) = initial_distance :=
by sorry

end mr_a_speed_l3758_375876


namespace simplify_and_evaluate_evaluate_at_zero_zero_in_range_l3758_375857

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 2) :
  (1 / (1 - x) + 1) / ((x^2 - 4*x + 4) / (x^2 - 1)) = (x + 1) / (x - 2) :=
by sorry

-- Evaluation at x = 0
theorem evaluate_at_zero :
  (1 / (1 - 0) + 1) / ((0^2 - 4*0 + 4) / (0^2 - 1)) = -1/2 :=
by sorry

-- Range constraint
def in_range (x : ℝ) : Prop := -2 < x ∧ x < 3

-- Proof that 0 is in the range
theorem zero_in_range : in_range 0 :=
by sorry

end simplify_and_evaluate_evaluate_at_zero_zero_in_range_l3758_375857


namespace hyperbola_condition_ellipse_y_focus_condition_l3758_375834

-- Define the curve C
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (5 - t) + y^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) := (5 - t) * (t - 1) < 0

-- Define what it means for C to be an ellipse with focus on y-axis
def is_ellipse_y_focus (t : ℝ) := t - 1 > 5 - t ∧ t - 1 > 0 ∧ 5 - t > 0

-- Statement 1
theorem hyperbola_condition (t : ℝ) : 
  t < 1 → is_hyperbola t :=
sorry

-- Statement 2
theorem ellipse_y_focus_condition (t : ℝ) :
  is_ellipse_y_focus t → 3 < t ∧ t < 5 :=
sorry

end hyperbola_condition_ellipse_y_focus_condition_l3758_375834


namespace equation_c_is_quadratic_l3758_375812

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (4x-3)(3x+1)=0 -/
def f (x : ℝ) : ℝ := (4*x - 3) * (3*x + 1)

/-- Theorem: The equation (4x-3)(3x+1)=0 is a quadratic equation -/
theorem equation_c_is_quadratic : is_quadratic_equation f := by
  sorry

end equation_c_is_quadratic_l3758_375812


namespace intersection_M_N_l3758_375853

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {-1, 1, 3} := by sorry

end intersection_M_N_l3758_375853


namespace f_750_value_l3758_375883

/-- A function satisfying f(xy) = f(x)/y for positive reals -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 ∧ y > 0 → f (x * y) = f x / y

theorem f_750_value (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 1000 = 4) :
  f 750 = 16/3 := by
  sorry

end f_750_value_l3758_375883


namespace partnership_theorem_l3758_375852

def partnership (total_capital : ℝ) (total_profit : ℝ) (a_profit : ℝ) : Prop :=
  let b_share : ℝ := (1 / 4) * total_capital
  let c_share : ℝ := (1 / 5) * total_capital
  let d_share : ℝ := total_capital - (b_share + c_share + (83 / 249) * total_capital)
  let a_share : ℝ := (83 / 249) * total_capital
  (a_profit / total_profit = 83 / 249) ∧
  (b_share + c_share + d_share + a_share = total_capital) ∧
  (d_share ≥ 0)

theorem partnership_theorem (total_capital : ℝ) (total_profit : ℝ) (a_profit : ℝ) 
  (h1 : total_capital > 0)
  (h2 : total_profit = 2490)
  (h3 : a_profit = 830) :
  partnership total_capital total_profit a_profit :=
by
  sorry

end partnership_theorem_l3758_375852


namespace product_difference_difference_of_products_l3758_375817

theorem product_difference (a b c d : ℝ) (h : a * b = c) : 
  a * d - a * b = a * (d - b) :=
by sorry

theorem difference_of_products : 
  (16.47 * 34) - (16.47 * 24) = 164.7 :=
by sorry

end product_difference_difference_of_products_l3758_375817


namespace sufficient_not_necessary_l3758_375839

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a - b < a^2 - b^2) ∧
  (∃ a b : ℝ, a - b < a^2 - b^2 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end sufficient_not_necessary_l3758_375839


namespace tom_needs_163_blue_tickets_l3758_375803

/-- Represents the number of tickets Tom has -/
structure TomTickets where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Conversion rates between ticket types -/
def yellowToRed : ℕ := 10
def redToBlue : ℕ := 10

/-- Number of yellow tickets needed to win a Bible -/
def yellowToWin : ℕ := 10

/-- Tom's current tickets -/
def tomCurrentTickets : TomTickets :=
  { yellow := 8, red := 3, blue := 7 }

/-- Calculate the total number of blue tickets Tom has -/
def totalBlueTickets (t : TomTickets) : ℕ :=
  t.yellow * yellowToRed * redToBlue + t.red * redToBlue + t.blue

/-- Calculate the number of blue tickets needed to win -/
def blueTicketsToWin : ℕ := yellowToWin * yellowToRed * redToBlue

/-- Theorem: Tom needs 163 more blue tickets to win a Bible -/
theorem tom_needs_163_blue_tickets :
  blueTicketsToWin - totalBlueTickets tomCurrentTickets = 163 := by
  sorry


end tom_needs_163_blue_tickets_l3758_375803


namespace sqrt_x_plus_3_meaningful_l3758_375828

theorem sqrt_x_plus_3_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 := by
  sorry

end sqrt_x_plus_3_meaningful_l3758_375828


namespace sum_even_integers_200_to_400_l3758_375896

def even_integers_between (a b : ℕ) : List ℕ :=
  (List.range (b - a + 1)).map (fun i => a + 2 * i)

theorem sum_even_integers_200_to_400 :
  (even_integers_between 200 400).sum = 30100 := by
  sorry

end sum_even_integers_200_to_400_l3758_375896


namespace proportion_result_l3758_375804

/-- Given a proportion x : 474 :: 537 : 26, prove that x rounded to the nearest integer is 9795 -/
theorem proportion_result : 
  ∃ x : ℝ, (x / 474 = 537 / 26) ∧ (round x = 9795) :=
by sorry

end proportion_result_l3758_375804


namespace distance_to_work_is_27_l3758_375845

/-- The distance to work in miles -/
def distance_to_work : ℝ := 27

/-- The total commute time in hours -/
def total_commute_time : ℝ := 1.5

/-- The average speed to work in mph -/
def speed_to_work : ℝ := 45

/-- The average speed from work in mph -/
def speed_from_work : ℝ := 30

/-- Theorem stating that the distance to work is 27 miles -/
theorem distance_to_work_is_27 :
  distance_to_work = 27 ∧
  total_commute_time = distance_to_work / speed_to_work + distance_to_work / speed_from_work :=
by sorry

end distance_to_work_is_27_l3758_375845


namespace probability_third_defective_correct_probability_correct_under_conditions_l3758_375829

/-- Represents the probability of drawing a defective item as the third draw
    given the conditions of the problem. -/
def probability_third_defective (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ) : ℚ :=
  7 / 36

/-- Theorem stating the probability of drawing a defective item as the third draw
    under the given conditions. -/
theorem probability_third_defective_correct :
  probability_third_defective 10 3 3 = 7 / 36 := by
  sorry

/-- Checks if the conditions of the problem are met. -/
def valid_conditions (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ) : Prop :=
  total_items = 10 ∧ defective_items = 3 ∧ items_drawn = 3

/-- Theorem stating that the probability is correct under the given conditions. -/
theorem probability_correct_under_conditions
  (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ)
  (h : valid_conditions total_items defective_items items_drawn) :
  probability_third_defective total_items defective_items items_drawn = 7 / 36 := by
  sorry

end probability_third_defective_correct_probability_correct_under_conditions_l3758_375829


namespace meetings_percentage_of_workday_l3758_375815

def workday_hours : ℕ := 10
def first_meeting_minutes : ℕ := 40

def second_meeting_minutes : ℕ := 2 * first_meeting_minutes
def third_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes
def workday_minutes : ℕ := workday_hours * 60

theorem meetings_percentage_of_workday :
  (total_meeting_minutes : ℚ) / (workday_minutes : ℚ) = 2 / 5 :=
sorry

end meetings_percentage_of_workday_l3758_375815


namespace largest_class_size_l3758_375802

/-- Represents the number of students in each class of a school --/
structure School :=
  (largest_class : ℕ)

/-- Calculates the total number of students in the school --/
def total_students (s : School) : ℕ :=
  s.largest_class + (s.largest_class - 2) + (s.largest_class - 4) + (s.largest_class - 6) + (s.largest_class - 8)

/-- Theorem stating that a school with 5 classes, where each class has 2 students less than the previous class, 
    and a total of 105 students, has 25 students in the largest class --/
theorem largest_class_size :
  ∃ (s : School), total_students s = 105 ∧ s.largest_class = 25 :=
by
  sorry

end largest_class_size_l3758_375802


namespace pipe_B_rate_correct_l3758_375838

/-- The rate at which pipe B fills the tank -/
def pipe_B_rate : ℝ := 30

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 750

/-- The rate at which pipe A fills the tank in liters per minute -/
def pipe_A_rate : ℝ := 40

/-- The rate at which pipe C drains the tank in liters per minute -/
def pipe_C_rate : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 45

/-- The duration of each pipe's operation in a cycle in minutes -/
def cycle_duration : ℝ := 3

/-- Theorem stating that the calculated rate of pipe B is correct -/
theorem pipe_B_rate_correct : 
  pipe_B_rate = (tank_capacity - fill_time / cycle_duration * (pipe_A_rate - pipe_C_rate)) / 
                (fill_time / cycle_duration) :=
by sorry

end pipe_B_rate_correct_l3758_375838


namespace min_different_numbers_l3758_375818

theorem min_different_numbers (total : ℕ) (max_freq : ℕ) (min_diff : ℕ) : 
  total = 2019 →
  max_freq = 10 →
  min_diff = 225 →
  (∀ k : ℕ, k < min_diff → k * (max_freq - 1) + max_freq < total) ∧
  (min_diff * (max_freq - 1) + max_freq ≥ total) := by
  sorry

end min_different_numbers_l3758_375818


namespace gcd_2183_1947_l3758_375888

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := by
  sorry

end gcd_2183_1947_l3758_375888


namespace original_number_is_192_l3758_375827

theorem original_number_is_192 (N : ℚ) : 
  (((N / 8 + 8) - 30) * 6) = 12 → N = 192 := by
sorry

end original_number_is_192_l3758_375827


namespace simplify_fraction_l3758_375872

theorem simplify_fraction (a : ℝ) (ha : a ≠ 0) :
  (a - 1) / a / (a - 1 / a) = 1 / (a + 1) := by
  sorry

end simplify_fraction_l3758_375872


namespace fraction_simplification_l3758_375877

theorem fraction_simplification :
  (1 / 3 + 1 / 4) / (2 / 5 - 1 / 6) = 5 / 2 := by sorry

end fraction_simplification_l3758_375877


namespace superhero_movie_count_l3758_375825

/-- The number of movies watched by Dalton -/
def dalton_movies : ℕ := 7

/-- The number of movies watched by Hunter -/
def hunter_movies : ℕ := 12

/-- The number of movies watched by Alex -/
def alex_movies : ℕ := 15

/-- The number of movies watched together by all three -/
def movies_watched_together : ℕ := 2

/-- The total number of different movies watched -/
def total_different_movies : ℕ := dalton_movies + hunter_movies + alex_movies - 2 * movies_watched_together

theorem superhero_movie_count : total_different_movies = 32 := by
  sorry

end superhero_movie_count_l3758_375825


namespace range_of_a_l3758_375821

-- Define the conditions p and q as functions
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x))  -- ¬p is necessary for ¬q
  (h3 : ∃ x, ¬(p a x) ∧ q x)     -- ¬p is not sufficient for ¬q
  : a ≤ -4 := by
  sorry

end range_of_a_l3758_375821


namespace some_mythical_creatures_are_magical_beings_l3758_375895

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Dragon : U → Prop)
variable (MythicalCreature : U → Prop)
variable (MagicalBeing : U → Prop)

-- State the theorem
theorem some_mythical_creatures_are_magical_beings
  (h1 : ∀ x, Dragon x → MythicalCreature x)
  (h2 : ∃ x, MagicalBeing x ∧ Dragon x) :
  ∃ x, MythicalCreature x ∧ MagicalBeing x :=
by
  sorry


end some_mythical_creatures_are_magical_beings_l3758_375895


namespace geometric_sequence_fourth_term_range_l3758_375882

theorem geometric_sequence_fourth_term_range 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : 0 < a 1 ∧ a 1 < 1)
  (h_a2 : 1 < a 2 ∧ a 2 < 2)
  (h_a3 : 2 < a 3 ∧ a 3 < 4) :
  2 * Real.sqrt 2 < a 4 ∧ a 4 < 16 := by
  sorry

end geometric_sequence_fourth_term_range_l3758_375882


namespace cos_45_degrees_l3758_375892

theorem cos_45_degrees : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

end cos_45_degrees_l3758_375892


namespace blood_type_distribution_l3758_375851

theorem blood_type_distribution (total : ℕ) (type_a : ℕ) (type_b : ℕ) : 
  (2 : ℚ) / 9 * total = type_a →
  (2 : ℚ) / 5 * total = type_b →
  type_a = 10 →
  type_b = 18 := by
sorry

end blood_type_distribution_l3758_375851


namespace m_three_sufficient_not_necessary_l3758_375860

def a (m : ℝ) : ℝ × ℝ := (-9, m^2)
def b : ℝ × ℝ := (1, -1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem m_three_sufficient_not_necessary :
  (∃ (m : ℝ), m ≠ 3 ∧ parallel (a m) b) ∧
  (∀ (m : ℝ), m = 3 → parallel (a m) b) :=
sorry

end m_three_sufficient_not_necessary_l3758_375860


namespace total_spent_on_toys_l3758_375890

def other_toys_cost : ℕ := 1000
def lightsaber_cost : ℕ := 2 * other_toys_cost

theorem total_spent_on_toys : other_toys_cost + lightsaber_cost = 3000 := by
  sorry

end total_spent_on_toys_l3758_375890


namespace books_remaining_l3758_375840

theorem books_remaining (initial_books : ℕ) (given_away : ℕ) (sold : ℕ) : 
  initial_books = 108 → given_away = 35 → sold = 11 → 
  initial_books - given_away - sold = 62 := by
sorry

end books_remaining_l3758_375840


namespace hair_cut_total_l3758_375893

theorem hair_cut_total (day1 : Float) (day2 : Float) (h1 : day1 = 0.38) (h2 : day2 = 0.5) :
  day1 + day2 = 0.88 := by
  sorry

end hair_cut_total_l3758_375893


namespace unique_n_satisfying_conditions_l3758_375849

def P (n : ℕ) : ℕ := sorry

theorem unique_n_satisfying_conditions : 
  ∃! n : ℕ, n > 1 ∧ 
    P n = n - 8 ∧ 
    P (n + 60) = n + 52 :=
sorry

end unique_n_satisfying_conditions_l3758_375849


namespace problem_solution_l3758_375865

theorem problem_solution (a b c : ℤ) : 
  a < b → b < c → 
  (a + b + c) / 3 = 4 * b → 
  c / b = 11 → 
  a = 0 := by
sorry

end problem_solution_l3758_375865


namespace students_looking_up_fraction_l3758_375814

theorem students_looking_up_fraction : 
  ∀ (total_students : ℕ) (eyes_saw_plane : ℕ) (eyes_per_student : ℕ),
    total_students = 200 →
    eyes_saw_plane = 300 →
    eyes_per_student = 2 →
    (eyes_saw_plane / eyes_per_student : ℚ) / total_students = 3/4 := by
  sorry

end students_looking_up_fraction_l3758_375814


namespace disjoint_subsets_remainder_l3758_375869

def T : Finset ℕ := Finset.range 12

def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

theorem disjoint_subsets_remainder (T : Finset ℕ) (m : ℕ) :
  T = Finset.range 12 →
  m = (3^12 - 2 * 2^12 + 1) / 2 →
  m % 1000 = 625 := by
  sorry

end disjoint_subsets_remainder_l3758_375869


namespace area_ratio_PQRV_ABCD_l3758_375881

-- Define the squares and points
variable (A B C D P Q R V : ℝ × ℝ)

-- Define the properties of the squares
def is_square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    ‖B - A‖ = s ∧ ‖C - B‖ = s ∧ ‖D - C‖ = s ∧ ‖A - D‖ = s ∧
    (B - A) • (C - B) = 0

-- Define that P is on side AB
def P_on_AB (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = A + t • (B - A)

-- Define the condition AP = 3 * PB
def AP_eq_3PB (A B P : ℝ × ℝ) : Prop :=
  ‖P - A‖ = 3 * ‖B - P‖

-- Define the area of a square
def area (A B C D : ℝ × ℝ) : ℝ :=
  ‖B - A‖^2

-- Theorem statement
theorem area_ratio_PQRV_ABCD 
  (h1 : is_square A B C D)
  (h2 : is_square P Q R V)
  (h3 : P_on_AB A B P)
  (h4 : AP_eq_3PB A B P) :
  area P Q R V / area A B C D = 1/16 := by
  sorry

end area_ratio_PQRV_ABCD_l3758_375881


namespace mary_eggs_problem_l3758_375835

theorem mary_eggs_problem (initial_eggs found_eggs final_eggs : ℕ) 
  (h1 : found_eggs = 4)
  (h2 : final_eggs = 31)
  (h3 : final_eggs = initial_eggs + found_eggs) :
  initial_eggs = 27 := by
  sorry

end mary_eggs_problem_l3758_375835


namespace min_value_quadratic_expression_l3758_375875

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4 ≥ -1 ∧
  ∃ x y : ℝ, 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4 = -1 := by
  sorry

end min_value_quadratic_expression_l3758_375875


namespace concert_cost_for_two_l3758_375894

def concert_cost (ticket_price : ℝ) (processing_fee_rate : ℝ) (parking_fee : ℝ) (entrance_fee : ℝ) (num_people : ℕ) : ℝ :=
  let total_ticket_price := ticket_price * num_people
  let processing_fee := total_ticket_price * processing_fee_rate
  let total_entrance_fee := entrance_fee * num_people
  total_ticket_price + processing_fee + parking_fee + total_entrance_fee

theorem concert_cost_for_two :
  concert_cost 50 0.15 10 5 2 = 135 := by
  sorry

end concert_cost_for_two_l3758_375894


namespace infinite_52_divisible_cells_l3758_375800

/-- Represents a position in the grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- The value at a node given its position in the spiral -/
def spiral_value (p : Position) : ℕ := sorry

/-- The sum of values at the four corners of a cell -/
def cell_sum (p : Position) : ℕ :=
  spiral_value p + spiral_value ⟨p.x + 1, p.y⟩ + 
  spiral_value ⟨p.x + 1, p.y + 1⟩ + spiral_value ⟨p.x, p.y + 1⟩

/-- Predicate for whether a number is divisible by 52 -/
def divisible_by_52 (n : ℕ) : Prop := n % 52 = 0

/-- The main theorem to be proved -/
theorem infinite_52_divisible_cells :
  ∀ n : ℕ, ∃ p : Position, p.x ≥ n ∧ p.y ≥ n ∧ divisible_by_52 (cell_sum p) :=
sorry

end infinite_52_divisible_cells_l3758_375800


namespace floor_length_proof_l3758_375806

/-- Represents a rectangular tile with length and width -/
structure Tile where
  length : ℕ
  width : ℕ

/-- Represents a rectangular floor with length and width -/
structure Floor where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on a floor -/
def maxTiles (t : Tile) (f : Floor) : ℕ :=
  let tilesAcross := f.width / t.width
  let tilesDown := f.length / t.length
  tilesAcross * tilesDown

theorem floor_length_proof (t : Tile) (f : Floor) (h1 : t.length = 25) (h2 : t.width = 16) 
    (h3 : f.width = 120) (h4 : maxTiles t f = 54) : f.length = 175 := by
  sorry

end floor_length_proof_l3758_375806


namespace circle_center_locus_l3758_375870

/-- Given a circle passing through A(0,a) with a chord of length 2a on the x-axis,
    prove that the locus of its center C(x,y) satisfies x^2 = 2ay -/
theorem circle_center_locus (a : ℝ) (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (x^2 + (y - a)^2 = r^2) ∧  -- Circle passes through A(0,a)
    (y^2 + a^2 = r^2))         -- Chord on x-axis has length 2a
  → x^2 = 2*a*y := by sorry

end circle_center_locus_l3758_375870


namespace sqrt_twelve_less_than_four_l3758_375879

theorem sqrt_twelve_less_than_four : Real.sqrt 12 < 4 := by
  sorry

end sqrt_twelve_less_than_four_l3758_375879


namespace cycle_sale_gain_percent_l3758_375885

/-- Calculates the gain percent for a cycle sale given the original price, discount percentage, refurbishing cost, and selling price. -/
def cycleGainPercent (originalPrice discountPercent refurbishCost sellingPrice : ℚ) : ℚ :=
  let discountAmount := (discountPercent / 100) * originalPrice
  let purchasePrice := originalPrice - discountAmount
  let totalCostPrice := purchasePrice + refurbishCost
  let gain := sellingPrice - totalCostPrice
  (gain / totalCostPrice) * 100

/-- Theorem stating that the gain percent for the given cycle sale scenario is 62.5% -/
theorem cycle_sale_gain_percent :
  cycleGainPercent 1200 25 300 1950 = 62.5 := by
  sorry

#eval cycleGainPercent 1200 25 300 1950

end cycle_sale_gain_percent_l3758_375885


namespace necessary_but_not_sufficient_condition_l3758_375822

open Real

-- Define the proposition
def P (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 1 2, x^2 - a > 0

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  (∃ a : ℝ, a < 2 ∧ ¬(P a)) ∧
  (∀ a : ℝ, P a → a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l3758_375822


namespace sum_of_powers_l3758_375844

theorem sum_of_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 5)
  (h4 : a^4 + b^4 = 7) :
  a^10 + b^10 = 19 := by
sorry

end sum_of_powers_l3758_375844


namespace contrapositive_equivalence_l3758_375873

-- Define the quadratic equation
def has_real_roots (m : ℕ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

-- Define the original proposition
def original_prop (m : ℕ) : Prop := m > 0 → has_real_roots m

-- Define the contrapositive
def contrapositive (m : ℕ) : Prop := ¬(has_real_roots m) → m ≤ 0

-- Theorem statement
theorem contrapositive_equivalence :
  ∀ m : ℕ, m > 0 → (original_prop m ↔ contrapositive m) :=
by sorry

end contrapositive_equivalence_l3758_375873


namespace min_additional_games_proof_l3758_375898

/-- The minimum number of additional games the Sharks need to win -/
def min_additional_games : ℕ := 145

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The initial number of games won by the Sharks -/
def initial_sharks_wins : ℕ := 2

/-- Predicate to check if a given number of additional games satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (initial_sharks_wins + n : ℚ) / (initial_games + n) ≥ 98 / 100

theorem min_additional_games_proof :
  satisfies_condition min_additional_games ∧
  ∀ m : ℕ, m < min_additional_games → ¬ satisfies_condition m :=
by sorry

end min_additional_games_proof_l3758_375898


namespace arithmetic_mean_of_fractions_l3758_375813

theorem arithmetic_mean_of_fractions : 
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = (67 / 144 : ℚ) := by
  sorry

end arithmetic_mean_of_fractions_l3758_375813


namespace cash_percentage_is_ten_percent_l3758_375807

def total_amount : ℝ := 1000
def raw_materials_cost : ℝ := 500
def machinery_cost : ℝ := 400

theorem cash_percentage_is_ten_percent :
  (total_amount - (raw_materials_cost + machinery_cost)) / total_amount * 100 = 10 := by
  sorry

end cash_percentage_is_ten_percent_l3758_375807


namespace linear_functions_coefficient_difference_l3758_375801

/-- Given linear functions f and g, and their composition h with a known inverse,
    prove that the difference of coefficients of f is 5. -/
theorem linear_functions_coefficient_difference (a b : ℝ) : 
  (∃ (f g h : ℝ → ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (∀ x, g x = -2 * x + 7) ∧ 
    (∀ x, h x = f (g x)) ∧ 
    (∀ x, Function.invFun h x = x + 9)) → 
  a - b = 5 := by
sorry

end linear_functions_coefficient_difference_l3758_375801


namespace quadratic_root_product_l3758_375848

theorem quadratic_root_product (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 : ℂ) + Complex.I ∈ {z : ℂ | z ^ 2 + p * z + q = 0} →
  p * q = -20 := by
sorry

end quadratic_root_product_l3758_375848


namespace company_fund_calculation_l3758_375837

theorem company_fund_calculation (n : ℕ) 
  (h1 : 80 * n - 15 = 70 * n + 155) : 
  80 * n - 15 = 1345 := by
  sorry

end company_fund_calculation_l3758_375837


namespace two_vertical_asymptotes_l3758_375826

-- Define the numerator and denominator of the rational function
def numerator (x : ℝ) : ℝ := x + 2
def denominator (x : ℝ) : ℝ := x^2 + 8*x + 15

-- Define a function to check if a given x-value is a vertical asymptote
def is_vertical_asymptote (x : ℝ) : Prop :=
  denominator x = 0 ∧ numerator x ≠ 0

-- Theorem stating that there are exactly 2 vertical asymptotes
theorem two_vertical_asymptotes :
  ∃ (a b : ℝ), a ≠ b ∧
    is_vertical_asymptote a ∧
    is_vertical_asymptote b ∧
    ∀ (x : ℝ), is_vertical_asymptote x → (x = a ∨ x = b) :=
by sorry

end two_vertical_asymptotes_l3758_375826


namespace solution_set_part1_range_of_a_part2_l3758_375833

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > -a) ↔ a > -3/2 :=
sorry

end solution_set_part1_range_of_a_part2_l3758_375833


namespace stones_combine_l3758_375884

/-- Definition of similar sizes -/
def similar (a b : ℕ) : Prop := a ≤ b ∧ b ≤ 2 * a

/-- A step in the combining process -/
inductive CombineStep (n : ℕ)
  | combine (i j : Fin n) (h : i.val < j.val) (hsimilar : similar (Fin.val i) (Fin.val j)) : CombineStep n

/-- A sequence of combining steps -/
def CombineSequence (n : ℕ) := List (CombineStep n)

/-- The final state after combining -/
def FinalState (n : ℕ) : Prop := ∃ (seq : CombineSequence n), 
  (∀ i : Fin n, i.val = 1) → (∃ j : Fin n, j.val = n ∧ ∀ k : Fin n, k ≠ j → k.val = 0)

/-- The main theorem -/
theorem stones_combine (n : ℕ) : FinalState n := by sorry

end stones_combine_l3758_375884


namespace triangle_gp_common_ratio_bounds_l3758_375810

/-- The common ratio of a geometric progression forming the sides of a triangle -/
def common_ratio_triangle_gp : Set ℝ :=
  {q : ℝ | (Real.sqrt 5 - 1) / 2 ≤ q ∧ q ≤ (Real.sqrt 5 + 1) / 2}

/-- Theorem: The common ratio of a geometric progression forming the sides of a triangle
    is bounded by (√5 - 1)/2 and (√5 + 1)/2 -/
theorem triangle_gp_common_ratio_bounds (a : ℝ) (q : ℝ) 
    (h_a : a > 0) (h_q : q ≥ 1) 
    (h_triangle : a + a*q > a*q^2 ∧ a + a*q^2 > a*q ∧ a*q + a*q^2 > a) :
  q ∈ common_ratio_triangle_gp := by
  sorry

end triangle_gp_common_ratio_bounds_l3758_375810


namespace third_day_temperature_l3758_375886

/-- Given three temperatures with a known average and two known values, 
    calculate the third temperature. -/
theorem third_day_temperature 
  (avg : ℚ) 
  (temp1 temp2 : ℚ) 
  (h_avg : avg = -7)
  (h_temp1 : temp1 = -14)
  (h_temp2 : temp2 = -8)
  (h_sum : 3 * avg = temp1 + temp2 + temp3) :
  temp3 = 1 := by
  sorry

#check third_day_temperature

end third_day_temperature_l3758_375886


namespace total_adjusted_income_equals_1219_72_l3758_375871

def initial_investment : ℝ := 6800
def stock_allocation : ℝ := 0.6
def bond_allocation : ℝ := 0.3
def cash_allocation : ℝ := 0.1
def inflation_rate : ℝ := 0.02

def cash_interest_rates : Fin 3 → ℝ
| 0 => 0.01
| 1 => 0.02
| 2 => 0.03

def stock_gains : Fin 3 → ℝ
| 0 => 0.08
| 1 => 0.04
| 2 => 0.10

def bond_returns : Fin 3 → ℝ
| 0 => 0.05
| 1 => 0.06
| 2 => 0.04

def adjusted_annual_income (i : Fin 3) : ℝ :=
  let stock_income := initial_investment * stock_allocation * stock_gains i
  let bond_income := initial_investment * bond_allocation * bond_returns i
  let cash_income := initial_investment * cash_allocation * cash_interest_rates i
  let total_income := stock_income + bond_income + cash_income
  total_income * (1 - inflation_rate)

theorem total_adjusted_income_equals_1219_72 :
  (adjusted_annual_income 0) + (adjusted_annual_income 1) + (adjusted_annual_income 2) = 1219.72 :=
sorry

end total_adjusted_income_equals_1219_72_l3758_375871


namespace sufficient_but_not_necessary_l3758_375897

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem sufficient_but_not_necessary : 
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end sufficient_but_not_necessary_l3758_375897
