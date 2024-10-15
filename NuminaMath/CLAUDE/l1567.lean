import Mathlib

namespace NUMINAMATH_CALUDE_intersection_distance_l1567_156730

-- Define the curves and ray
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y θ : ℝ) : Prop := x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sin θ
def ray (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∧ x ≥ 0

-- Define the intersection points
def point_A (x y : ℝ) : Prop := C₁ x y ∧ ray x y
def point_B (x y θ : ℝ) : Prop := C₂ x y θ ∧ ray x y

-- Theorem statement
theorem intersection_distance :
  ∀ (x₁ y₁ x₂ y₂ θ : ℝ),
  point_A x₁ y₁ → point_B x₂ y₂ θ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l1567_156730


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1567_156706

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1567_156706


namespace NUMINAMATH_CALUDE_omitted_number_proof_l1567_156792

/-- Sequence of even numbers starting from 2 -/
def evenSeq (n : ℕ) : ℕ := 2 * n

/-- Sum of even numbers from 2 to 2n -/
def evenSum (n : ℕ) : ℕ := n * (n + 1)

/-- The incorrect sum obtained -/
def incorrectSum : ℕ := 2014

/-- The omitted number -/
def omittedNumber : ℕ := 56

theorem omitted_number_proof :
  ∃ n : ℕ, evenSum n - incorrectSum = omittedNumber ∧
  evenSeq (n + 1) = omittedNumber :=
sorry

end NUMINAMATH_CALUDE_omitted_number_proof_l1567_156792


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1567_156772

/-- Represents the number of employees in each title category -/
structure TitleCount where
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Represents the result of a stratified sampling -/
structure SampleResult where
  senior : Nat
  intermediate : Nat
  junior : Nat

def total_employees : Nat := 150
def sample_size : Nat := 30

def population : TitleCount := {
  senior := 45,
  intermediate := 90,
  junior := 15
}

def stratified_sample (pop : TitleCount) (total : Nat) (sample : Nat) : SampleResult :=
  { senior := sample * pop.senior / total,
    intermediate := sample * pop.intermediate / total,
    junior := sample * pop.junior / total }

theorem stratified_sampling_theorem :
  stratified_sample population total_employees sample_size =
  { senior := 9, intermediate := 18, junior := 3 } := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1567_156772


namespace NUMINAMATH_CALUDE_spinning_tops_theorem_l1567_156793

/-- The number of spinning tops obtained from gift boxes --/
def spinning_tops_count (red_price yellow_price : ℕ) (red_tops yellow_tops : ℕ) 
  (total_spent total_boxes : ℕ) : ℕ :=
  let red_boxes := (total_spent - yellow_price * total_boxes) / (red_price - yellow_price)
  let yellow_boxes := total_boxes - red_boxes
  red_boxes * red_tops + yellow_boxes * yellow_tops

/-- Theorem stating the number of spinning tops obtained --/
theorem spinning_tops_theorem : 
  spinning_tops_count 5 9 3 5 600 72 = 336 := by
  sorry

#eval spinning_tops_count 5 9 3 5 600 72

end NUMINAMATH_CALUDE_spinning_tops_theorem_l1567_156793


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_original_l1567_156758

-- Define a line in 2D space
structure Line2D where
  f : ℝ → ℝ → ℝ
  is_line : ∀ x y, f x y = 0 ↔ (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a ≠ 0 ∨ b ≠ 0))

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on a line
def on_line (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y = 0

-- Define what it means for a point to not be on a line
def not_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y ≠ 0

-- Define parallel lines
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1.f x y = k * l2.f x y

-- Theorem statement
theorem line_through_point_parallel_to_original 
  (l : Line2D) (p1 p2 : Point2D) 
  (h1 : on_line p1 l) 
  (h2 : not_on_line p2 l) : 
  ∃ l2 : Line2D, 
    (∀ x y, l2.f x y = l.f x y - l.f p1.x p1.y - l.f p2.x p2.y) ∧ 
    on_line p2 l2 ∧ 
    parallel l l2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_original_l1567_156758


namespace NUMINAMATH_CALUDE_disjunction_false_l1567_156764

theorem disjunction_false :
  ¬(
    (∃ x : ℝ, x^2 + 1 < 2*x) ∨
    (∀ m : ℝ, (∀ x : ℝ, m*x^2 - m*x + 1 > 0) → (0 < m ∧ m < 4))
  ) := by sorry

end NUMINAMATH_CALUDE_disjunction_false_l1567_156764


namespace NUMINAMATH_CALUDE_parabola_h_values_l1567_156749

/-- Represents a parabola of the form y = -(x - h)² -/
def Parabola (h : ℝ) : ℝ → ℝ := fun x ↦ -((x - h)^2)

/-- The domain of x values -/
def Domain : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem parabola_h_values (h : ℝ) :
  (∀ x ∈ Domain, Parabola h x ≤ -1) ∧
  (∃ x ∈ Domain, Parabola h x = -1) →
  h = 2 ∨ h = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_h_values_l1567_156749


namespace NUMINAMATH_CALUDE_elevator_time_l1567_156777

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 9

/-- Represents the number of steps per floor -/
def steps_per_floor : ℕ := 30

/-- Represents the number of steps Jake descends per second -/
def jake_steps_per_second : ℕ := 3

/-- Represents the time difference (in seconds) between Jake and the elevator reaching the ground floor -/
def time_difference : ℕ := 30

/-- Calculates the total number of steps Jake needs to descend -/
def total_steps : ℕ := (num_floors - 1) * steps_per_floor

/-- Calculates the time (in seconds) it takes Jake to reach the ground floor -/
def jake_time : ℕ := total_steps / jake_steps_per_second

/-- Theorem stating that the elevator takes 50 seconds to reach the ground level -/
theorem elevator_time : jake_time - time_difference = 50 := by sorry

end NUMINAMATH_CALUDE_elevator_time_l1567_156777


namespace NUMINAMATH_CALUDE_batch_size_calculation_l1567_156796

theorem batch_size_calculation (N : ℕ) (sample_size : ℕ) (prob : ℚ) 
  (h1 : sample_size = 30)
  (h2 : prob = 1/4)
  (h3 : (sample_size : ℚ) / N = prob) : 
  N = 120 := by
  sorry

end NUMINAMATH_CALUDE_batch_size_calculation_l1567_156796


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_fraction_l1567_156703

theorem simplify_and_rationalize_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18 + Real.sqrt 32) = (5 * Real.sqrt 2) / 36 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_fraction_l1567_156703


namespace NUMINAMATH_CALUDE_correct_propositions_l1567_156736

-- Define the propositions
def proposition1 : Prop := sorry
def proposition2 : Prop := sorry
def proposition3 : Prop := sorry
def proposition4 : Prop := sorry

-- Define a function to check if a proposition is correct
def is_correct (p : Prop) : Prop := sorry

-- Theorem statement
theorem correct_propositions :
  is_correct proposition2 ∧ 
  is_correct proposition3 ∧ 
  ¬is_correct proposition1 ∧ 
  ¬is_correct proposition4 :=
sorry

end NUMINAMATH_CALUDE_correct_propositions_l1567_156736


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1567_156708

theorem quadratic_inequality_solution (x : ℝ) :
  (2 * x^2 - 5 * x + 2 > 0) ↔ (x < (1 : ℝ) / 2 ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1567_156708


namespace NUMINAMATH_CALUDE_triangle_side_length_l1567_156707

/-- Given a triangle ABC with side lengths and altitude, prove BC = 4 -/
theorem triangle_side_length (A B C : ℝ × ℝ) (h : ℝ) : 
  let d := (fun (P Q : ℝ × ℝ) => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  (d A B = 2 * Real.sqrt 3) →
  (d A C = 2) →
  (h = Real.sqrt 3) →
  (h * d B C = 2 * Real.sqrt 3 * 2) →
  d B C = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1567_156707


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1567_156728

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 290)
  (h3 : crossing_time = 23.998080153587715) :
  (((train_length + bridge_length) / crossing_time) * 3.6) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1567_156728


namespace NUMINAMATH_CALUDE_angle_A_is_pi_third_max_perimeter_is_3_sqrt_3_l1567_156719

/-- Triangle ABC with angles A, B, C and opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vectors m and n are orthogonal -/
def vectors_orthogonal (t : Triangle) : Prop :=
  t.a * (Real.cos t.C + Real.sqrt 3 * Real.sin t.C) + (t.b + t.c) * (-1) = 0

/-- Theorem: If vectors are orthogonal, then angle A is π/3 -/
theorem angle_A_is_pi_third (t : Triangle) (h : vectors_orthogonal t) : t.A = π / 3 := by
  sorry

/-- Maximum perimeter when a = √3 -/
def max_perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: When a = √3, the maximum perimeter is 3√3 -/
theorem max_perimeter_is_3_sqrt_3 (t : Triangle) (h : t.a = Real.sqrt 3) :
  max_perimeter t ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_third_max_perimeter_is_3_sqrt_3_l1567_156719


namespace NUMINAMATH_CALUDE_rectangle_y_value_l1567_156732

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices := [(1, y), (-5, y), (1, -2), (-5, -2)]
  let length := 1 - (-5)
  let height := y - (-2)
  let area := length * height
  area = 56 → y = 22/3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l1567_156732


namespace NUMINAMATH_CALUDE_student_team_signup_l1567_156744

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of sports teams --/
def num_teams : ℕ := 3

/-- The function that calculates the number of ways students can sign up for teams --/
def ways_to_sign_up (students : ℕ) (teams : ℕ) : ℕ := teams ^ students

/-- Theorem stating that there are 81 ways for 4 students to sign up for 3 teams --/
theorem student_team_signup :
  ways_to_sign_up num_students num_teams = 81 := by
  sorry

end NUMINAMATH_CALUDE_student_team_signup_l1567_156744


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l1567_156731

theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x + 6 = a*(x - h)^2 + k ∧ k = -25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l1567_156731


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l1567_156746

theorem corrected_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 48 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  corrected_sum / n = 36.5 := by
sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l1567_156746


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1567_156750

/-- The volume of a cube given its space diagonal -/
theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  (d / Real.sqrt 3) ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1567_156750


namespace NUMINAMATH_CALUDE_holistic_substitution_l1567_156762

theorem holistic_substitution (a : ℝ) (x : ℝ) :
  (a^2 + 3*a - 2 = 0) →
  (5*a^3 + 15*a^2 - 10*a + 2020 = 2020) ∧
  ((x^2 + 2*x - 3 = 0) → 
   (x = 1 ∨ x = -3) →
   ((2*x + 3)^2 + 2*(2*x + 3) - 3 = 0) →
   (x = -1 ∨ x = -3)) :=
by sorry

end NUMINAMATH_CALUDE_holistic_substitution_l1567_156762


namespace NUMINAMATH_CALUDE_quadratic_interval_l1567_156757

theorem quadratic_interval (x : ℝ) : 
  (6 ≤ x^2 + 5*x + 6 ∧ x^2 + 5*x + 6 ≤ 12) ↔ ((-6 ≤ x ∧ x ≤ -5) ∨ (0 ≤ x ∧ x ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_interval_l1567_156757


namespace NUMINAMATH_CALUDE_green_chips_count_l1567_156742

theorem green_chips_count (total : ℕ) (blue : ℕ) (white : ℕ) (green : ℕ) : 
  blue = 3 →
  blue = (10 * total) / 100 →
  white = (50 * total) / 100 →
  green = total - blue - white →
  green = 12 := by
sorry

end NUMINAMATH_CALUDE_green_chips_count_l1567_156742


namespace NUMINAMATH_CALUDE_kelly_cheese_packages_l1567_156711

-- Define the problem parameters
def days_per_week : ℕ := 5
def oldest_child_cheese_per_day : ℕ := 2
def youngest_child_cheese_per_day : ℕ := 1
def cheese_per_package : ℕ := 30
def weeks : ℕ := 4

-- Define the theorem
theorem kelly_cheese_packages :
  (days_per_week * (oldest_child_cheese_per_day + youngest_child_cheese_per_day) * weeks + cheese_per_package - 1) / cheese_per_package = 2 :=
by sorry

end NUMINAMATH_CALUDE_kelly_cheese_packages_l1567_156711


namespace NUMINAMATH_CALUDE_function_equality_l1567_156752

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then -x^2 + 1 else x - 1

theorem function_equality (a : ℝ) : f (a + 1) = f a ↔ a = -1/2 ∨ a = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l1567_156752


namespace NUMINAMATH_CALUDE_max_A_value_l1567_156767

/-- Represents the board configuration after chip removal operations -/
structure BoardConfig where
  white_columns : Nat
  white_rows : Nat
  black_columns : Nat
  black_rows : Nat

/-- Calculates the number of remaining chips for a given color -/
def remaining_chips (config : BoardConfig) (color : Bool) : Nat :=
  if color then config.white_columns * config.white_rows
  else config.black_columns * config.black_rows

/-- The size of the board -/
def board_size : Nat := 2018

/-- Theorem stating the maximum value of A -/
theorem max_A_value :
  ∃ (config : BoardConfig),
    config.white_columns + config.black_columns = board_size ∧
    config.white_rows + config.black_rows = board_size ∧
    ∀ (other_config : BoardConfig),
      other_config.white_columns + other_config.black_columns = board_size →
      other_config.white_rows + other_config.black_rows = board_size →
      min (remaining_chips config true) (remaining_chips config false) ≥
      min (remaining_chips other_config true) (remaining_chips other_config false) ∧
    min (remaining_chips config true) (remaining_chips config false) = 1018081 :=
sorry

end NUMINAMATH_CALUDE_max_A_value_l1567_156767


namespace NUMINAMATH_CALUDE_smallest_number_of_cars_l1567_156718

theorem smallest_number_of_cars (N : ℕ) : 
  N > 2 ∧ 
  N % 5 = 2 ∧ 
  N % 6 = 2 ∧ 
  N % 7 = 2 → 
  (∀ m : ℕ, m > 2 ∧ m % 5 = 2 ∧ m % 6 = 2 ∧ m % 7 = 2 → m ≥ N) →
  N = 212 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_cars_l1567_156718


namespace NUMINAMATH_CALUDE_birthday_problem_l1567_156788

/-- The number of months in the fantasy world -/
def num_months : ℕ := 10

/-- The number of people in the room -/
def num_people : ℕ := 60

/-- The largest number n such that at least n people are guaranteed to have birthdays in the same month -/
def largest_guaranteed_group : ℕ := 6

theorem birthday_problem :
  ∀ (birthday_distribution : Fin num_people → Fin num_months),
  ∃ (month : Fin num_months),
  (Finset.filter (λ person => birthday_distribution person = month) Finset.univ).card ≥ largest_guaranteed_group ∧
  ∀ n > largest_guaranteed_group,
  ∃ (bad_distribution : Fin num_people → Fin num_months),
  ∀ (month : Fin num_months),
  (Finset.filter (λ person => bad_distribution person = month) Finset.univ).card < n :=
sorry

end NUMINAMATH_CALUDE_birthday_problem_l1567_156788


namespace NUMINAMATH_CALUDE_biology_marks_l1567_156766

def marks_english : ℕ := 86
def marks_mathematics : ℕ := 85
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 87
def average_marks : ℕ := 85
def total_subjects : ℕ := 5

theorem biology_marks :
  ∃ (marks_biology : ℕ),
    marks_biology = average_marks * total_subjects - (marks_english + marks_mathematics + marks_physics + marks_chemistry) ∧
    marks_biology = 85 := by
  sorry

end NUMINAMATH_CALUDE_biology_marks_l1567_156766


namespace NUMINAMATH_CALUDE_black_area_after_three_changes_l1567_156753

/-- Represents the fraction of black area remaining after a single change -/
def blackAreaAfterOneChange (initialBlackArea : ℚ) : ℚ :=
  initialBlackArea * (5/6) * (9/10)

/-- Calculates the fraction of black area remaining after n changes -/
def blackAreaAfterNChanges (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => blackAreaAfterOneChange (blackAreaAfterNChanges n)

/-- The main theorem stating that after 3 changes, 27/64 of the original area remains black -/
theorem black_area_after_three_changes :
  blackAreaAfterNChanges 3 = 27/64 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_three_changes_l1567_156753


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l1567_156776

theorem exactly_one_greater_than_one 
  (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (prod_eq_one : a * b * c = 1)
  (sum_gt_recip_sum : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l1567_156776


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1567_156771

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - a ≥ 0 ∧ 2*x < 4))) → 
  (-2 < a ∧ a ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1567_156771


namespace NUMINAMATH_CALUDE_electric_bicycle_sales_l1567_156722

theorem electric_bicycle_sales (model_a_first_quarter : Real) 
  (model_bc_first_quarter : Real) (a : Real) :
  model_a_first_quarter = 0.56 ∧ 
  model_bc_first_quarter = 1 - model_a_first_quarter ∧
  0.56 * (1 + 0.23) + (1 - 0.56) * (1 - a / 100) = 1 + 0.12 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_electric_bicycle_sales_l1567_156722


namespace NUMINAMATH_CALUDE_factorization_72_P_72_l1567_156726

/-- P(n) represents the number of ways to write a positive integer n 
    as a product of integers greater than 1, where order matters. -/
def P (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 72 is 2^3 * 3^2 -/
theorem factorization_72 : 72 = 2^3 * 3^2 := sorry

/-- The main theorem: P(72) = 17 -/
theorem P_72 : P 72 = 17 := sorry

end NUMINAMATH_CALUDE_factorization_72_P_72_l1567_156726


namespace NUMINAMATH_CALUDE_vector_perpendicular_problem_l1567_156721

theorem vector_perpendicular_problem (a b c : ℝ × ℝ) (k : ℝ) :
  a = (1, 2) →
  b = (1, 1) →
  c = (a.1 + k * b.1, a.2 + k * b.2) →
  b.1 * c.1 + b.2 * c.2 = 0 →
  k = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_problem_l1567_156721


namespace NUMINAMATH_CALUDE_burger_cost_proof_l1567_156723

/-- The cost of Uri's purchase in cents -/
def uri_cost : ℕ := 450

/-- The cost of Gen's purchase in cents -/
def gen_cost : ℕ := 480

/-- The number of burgers Uri bought -/
def uri_burgers : ℕ := 3

/-- The number of sodas Uri bought -/
def uri_sodas : ℕ := 2

/-- The number of burgers Gen bought -/
def gen_burgers : ℕ := 2

/-- The number of sodas Gen bought -/
def gen_sodas : ℕ := 3

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 78

theorem burger_cost_proof :
  ∃ (soda_cost : ℕ),
    uri_burgers * burger_cost + uri_sodas * soda_cost = uri_cost ∧
    gen_burgers * burger_cost + gen_sodas * soda_cost = gen_cost :=
by sorry

end NUMINAMATH_CALUDE_burger_cost_proof_l1567_156723


namespace NUMINAMATH_CALUDE_total_slices_needed_l1567_156782

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of bread slices needed for each sandwich -/
def slices_per_sandwich : ℕ := 3

/-- Theorem stating the total number of bread slices needed -/
theorem total_slices_needed : num_sandwiches * slices_per_sandwich = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_needed_l1567_156782


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1567_156705

theorem consecutive_integers_product_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1567_156705


namespace NUMINAMATH_CALUDE_school_attendance_l1567_156779

/-- The number of students who came to school given the number of female students,
    the difference between female and male students, and the number of absent students. -/
def students_who_came_to_school (female_students : ℕ) (female_male_difference : ℕ) (absent_students : ℕ) : ℕ :=
  female_students + (female_students - female_male_difference) - absent_students

/-- Theorem stating that given the specific conditions, 1261 students came to school. -/
theorem school_attendance : students_who_came_to_school 658 38 17 = 1261 := by
  sorry

end NUMINAMATH_CALUDE_school_attendance_l1567_156779


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l1567_156704

/-- Proves that the new supervisor's salary must be $870 to maintain the same average salary --/
theorem new_supervisor_salary
  (num_workers : ℕ)
  (num_total : ℕ)
  (initial_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_average : ℚ)
  (h_num_workers : num_workers = 8)
  (h_num_total : num_total = num_workers + 1)
  (h_initial_average : initial_average = 430)
  (h_old_supervisor_salary : old_supervisor_salary = 870)
  (h_new_average : new_average = initial_average)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 870 ∧
    (num_workers : ℚ) * initial_average + old_supervisor_salary = num_total * initial_average ∧
    (num_workers : ℚ) * new_average + new_supervisor_salary = num_total * new_average :=
by
  sorry


end NUMINAMATH_CALUDE_new_supervisor_salary_l1567_156704


namespace NUMINAMATH_CALUDE_wedding_guests_l1567_156780

/-- The number of guests at Jenny's wedding --/
def total_guests : ℕ := 80

/-- The number of guests who want chicken --/
def chicken_guests : ℕ := 20

/-- The number of guests who want steak --/
def steak_guests : ℕ := 60

/-- The cost of a chicken entree in dollars --/
def chicken_cost : ℕ := 18

/-- The cost of a steak entree in dollars --/
def steak_cost : ℕ := 25

/-- The total catering budget in dollars --/
def total_budget : ℕ := 1860

theorem wedding_guests :
  (chicken_guests + steak_guests = total_guests) ∧
  (steak_guests = 3 * chicken_guests) ∧
  (chicken_cost * chicken_guests + steak_cost * steak_guests = total_budget) := by
  sorry

end NUMINAMATH_CALUDE_wedding_guests_l1567_156780


namespace NUMINAMATH_CALUDE_spaceship_journey_l1567_156768

def total_distance : Real := 0.7
def earth_to_x : Real := 0.5
def y_to_earth : Real := 0.1

theorem spaceship_journey : 
  total_distance - earth_to_x - y_to_earth = 0.1 := by sorry

end NUMINAMATH_CALUDE_spaceship_journey_l1567_156768


namespace NUMINAMATH_CALUDE_relative_error_comparison_l1567_156727

theorem relative_error_comparison :
  let line1_length : ℝ := 25
  let line1_error : ℝ := 0.05
  let line2_length : ℝ := 125
  let line2_error : ℝ := 0.25
  let relative_error1 : ℝ := line1_error / line1_length
  let relative_error2 : ℝ := line2_error / line2_length
  relative_error1 = relative_error2 :=
by sorry

end NUMINAMATH_CALUDE_relative_error_comparison_l1567_156727


namespace NUMINAMATH_CALUDE_range_of_a_l1567_156717

theorem range_of_a (a : ℝ) : 
  (¬∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1567_156717


namespace NUMINAMATH_CALUDE_sign_white_area_l1567_156710

/-- Represents the dimensions and areas of the letters in the sign --/
structure LetterAreas where
  m_area : ℝ
  a_area : ℝ
  t_area : ℝ
  h_area : ℝ

/-- Calculates the white area of the sign after drawing the letters "MATH" --/
def white_area (sign_width sign_height : ℝ) (letters : LetterAreas) : ℝ :=
  sign_width * sign_height - (letters.m_area + letters.a_area + letters.t_area + letters.h_area)

/-- Theorem stating that the white area of the sign is 42.5 square units --/
theorem sign_white_area :
  let sign_width := 20
  let sign_height := 4
  let letters := LetterAreas.mk 12 7.5 7 11
  white_area sign_width sign_height letters = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_sign_white_area_l1567_156710


namespace NUMINAMATH_CALUDE_remainder_of_geometric_sum_l1567_156700

def geometric_sum (r : ℕ) (n : ℕ) : ℕ :=
  (r^(n + 1) - 1) / (r - 1)

theorem remainder_of_geometric_sum :
  (geometric_sum 7 2004) % 1000 = 801 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_geometric_sum_l1567_156700


namespace NUMINAMATH_CALUDE_percentage_difference_l1567_156741

theorem percentage_difference (C : ℝ) (A B : ℝ) 
  (hA : A = 0.7 * C) 
  (hB : B = 0.63 * C) : 
  (A - B) / A = 0.1 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1567_156741


namespace NUMINAMATH_CALUDE_city_population_ratio_l1567_156769

/-- Given the population relationships between cities X, Y, and Z, 
    prove that the ratio of City X's population to City Z's population is 6:1 -/
theorem city_population_ratio 
  (Z : ℕ) -- Population of City Z
  (Y : ℕ) -- Population of City Y
  (X : ℕ) -- Population of City X
  (h1 : Y = 2 * Z) -- City Y's population is twice City Z's
  (h2 : ∃ k : ℕ, X = k * Y) -- City X's population is some multiple of City Y's
  (h3 : X = 6 * Z) -- The ratio of City X's to City Z's population is 6
  : X / Z = 6 := by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l1567_156769


namespace NUMINAMATH_CALUDE_fourth_guard_distance_l1567_156787

/-- Represents a rectangular facility with guards -/
structure Facility :=
  (length : ℝ)
  (width : ℝ)
  (perimeter : ℝ)
  (three_guards_distance : ℝ)

/-- The theorem to prove -/
theorem fourth_guard_distance (f : Facility) 
  (h1 : f.length = 200)
  (h2 : f.width = 300)
  (h3 : f.perimeter = 2 * (f.length + f.width))
  (h4 : f.three_guards_distance = 850) :
  f.perimeter - f.three_guards_distance = 150 := by
  sorry

#check fourth_guard_distance

end NUMINAMATH_CALUDE_fourth_guard_distance_l1567_156787


namespace NUMINAMATH_CALUDE_real_numbers_closed_closed_set_contains_zero_l1567_156791

-- Definition of a closed set
def is_closed_set (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S ∧ (x * y) ∈ S

-- Theorem 1: The set of real numbers is a closed set
theorem real_numbers_closed : is_closed_set Set.univ := by sorry

-- Theorem 2: If S is a closed set, then 0 is an element of S
theorem closed_set_contains_zero (S : Set ℝ) (h : is_closed_set S) (h_nonempty : S.Nonempty) : 
  (0 : ℝ) ∈ S := by sorry

end NUMINAMATH_CALUDE_real_numbers_closed_closed_set_contains_zero_l1567_156791


namespace NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l1567_156790

/-- Given a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 and eccentricity 2,
    if the product of the distances from a point on the hyperbola to its two asymptotes is 3/4,
    then the length of the conjugate axis is 2√3. -/
theorem hyperbola_conjugate_axis_length 
  (a b : ℝ) 
  (h1 : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → 
    (|b*x - a*y| * |b*x + a*y|) / (a^2 + b^2) = 3/4)
  (h2 : a^2 + b^2 = 5*a^2) :
  2*b = 2*Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l1567_156790


namespace NUMINAMATH_CALUDE_right_triangle_angle_identity_l1567_156794

theorem right_triangle_angle_identity (α β γ : Real) 
  (h_right_triangle : α + β + γ = π)
  (h_right_angle : α = π/2 ∨ β = π/2 ∨ γ = π/2) : 
  Real.sin α * Real.sin β * Real.sin (α - β) + 
  Real.sin β * Real.sin γ * Real.sin (β - γ) + 
  Real.sin γ * Real.sin α * Real.sin (γ - α) + 
  Real.sin (α - β) * Real.sin (β - γ) * Real.sin (γ - α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_identity_l1567_156794


namespace NUMINAMATH_CALUDE_football_field_length_prove_football_field_length_l1567_156781

theorem football_field_length : ℝ → Prop :=
  fun length =>
    (4 * length + 500 = 1172) →
    length = 168

-- The proof is omitted
theorem prove_football_field_length : football_field_length 168 := by
  sorry

end NUMINAMATH_CALUDE_football_field_length_prove_football_field_length_l1567_156781


namespace NUMINAMATH_CALUDE_largest_number_l1567_156756

theorem largest_number (a b c d : ℝ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = Real.sqrt 3) :
  d = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1567_156756


namespace NUMINAMATH_CALUDE_equation_solutions_l1567_156761

theorem equation_solutions : 
  (∃ x : ℝ, x^2 - 2*x - 7 = 0 ↔ x = 1 + 2*Real.sqrt 2 ∨ x = 1 - 2*Real.sqrt 2) ∧
  (∃ x : ℝ, 3*(x-2)^2 = x*(x-2) ↔ x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1567_156761


namespace NUMINAMATH_CALUDE_total_players_in_ground_l1567_156745

theorem total_players_in_ground (cricket_players hockey_players football_players softball_players : ℕ) 
  (h1 : cricket_players = 22)
  (h2 : hockey_players = 15)
  (h3 : football_players = 21)
  (h4 : softball_players = 19) :
  cricket_players + hockey_players + football_players + softball_players = 77 := by
  sorry

end NUMINAMATH_CALUDE_total_players_in_ground_l1567_156745


namespace NUMINAMATH_CALUDE_cookie_sales_proof_l1567_156720

theorem cookie_sales_proof (total_value : ℝ) (choc_price plain_price : ℝ) (plain_boxes : ℝ) :
  total_value = 1586.25 →
  choc_price = 1.25 →
  plain_price = 0.75 →
  plain_boxes = 793.125 →
  ∃ (choc_boxes : ℝ), 
    choc_price * choc_boxes + plain_price * plain_boxes = total_value ∧
    choc_boxes + plain_boxes = 1586.25 :=
by sorry

end NUMINAMATH_CALUDE_cookie_sales_proof_l1567_156720


namespace NUMINAMATH_CALUDE_unique_function_identity_l1567_156715

theorem unique_function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x ≠ 0, f x = x^2 * f (1/x))
  (h2 : ∀ x y, f (x + y) = f x + f y)
  (h3 : f 1 = 1) :
  ∀ x, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_function_identity_l1567_156715


namespace NUMINAMATH_CALUDE_kannon_oranges_last_night_l1567_156786

/-- Represents the number of fruits Kannon ate --/
structure FruitCount where
  apples : ℕ
  bananas : ℕ
  oranges : ℕ

/-- The total number of fruits eaten over two meals --/
def totalFruits : ℕ := 39

/-- Kannon's fruit consumption for last night --/
def lastNight : FruitCount where
  apples := 3
  bananas := 1
  oranges := 4  -- This is what we want to prove

/-- Kannon's fruit consumption for today --/
def today : FruitCount where
  apples := lastNight.apples + 4
  bananas := 10 * lastNight.bananas
  oranges := 2 * (lastNight.apples + 4)

/-- The theorem to prove --/
theorem kannon_oranges_last_night :
  lastNight.oranges = 4 ∧
  lastNight.apples + lastNight.bananas + lastNight.oranges +
  today.apples + today.bananas + today.oranges = totalFruits := by
  sorry


end NUMINAMATH_CALUDE_kannon_oranges_last_night_l1567_156786


namespace NUMINAMATH_CALUDE_button_numbers_l1567_156737

theorem button_numbers (x y : ℕ) 
  (h1 : y - x = 480) 
  (h2 : y = 4 * x + 30) : 
  y = 630 := by
  sorry

end NUMINAMATH_CALUDE_button_numbers_l1567_156737


namespace NUMINAMATH_CALUDE_quadratic_satisfies_conditions_l1567_156763

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

-- State the theorem
theorem quadratic_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_satisfies_conditions_l1567_156763


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l1567_156799

theorem max_value_expression (x : ℝ) :
  x^6 / (x^12 + 3*x^8 - 6*x^6 + 12*x^4 + 36) ≤ 1/18 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^8 - 6*x^6 + 12*x^4 + 36) = 1/18 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l1567_156799


namespace NUMINAMATH_CALUDE_g_composed_four_times_is_even_l1567_156729

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

theorem g_composed_four_times_is_even 
  (g : ℝ → ℝ) 
  (h : is_even_function g) : 
  is_even_function (fun x ↦ g (g (g (g x)))) :=
by
  sorry

end NUMINAMATH_CALUDE_g_composed_four_times_is_even_l1567_156729


namespace NUMINAMATH_CALUDE_savings_ratio_l1567_156702

/-- Proves that the ratio of Nora's savings to Lulu's savings is 5:1 given the problem conditions -/
theorem savings_ratio (debt : ℝ) (lulu_savings : ℝ) (remaining_per_person : ℝ)
  (h1 : debt = 40)
  (h2 : lulu_savings = 6)
  (h3 : remaining_per_person = 2)
  (h4 : ∃ (x : ℝ), x > 0 ∧ ∃ (tamara_savings : ℝ),
    x * lulu_savings = 3 * tamara_savings ∧
    debt + 3 * remaining_per_person = lulu_savings + x * lulu_savings + tamara_savings) :
  ∃ (nora_savings : ℝ), nora_savings / lulu_savings = 5 := by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l1567_156702


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1567_156735

theorem sqrt_sum_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt a + Real.sqrt (a + 5) < Real.sqrt (a + 2) + Real.sqrt (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1567_156735


namespace NUMINAMATH_CALUDE_price_reduction_5_price_reduction_20_no_2200_profit_l1567_156740

/-- Represents the supermarket's sales model -/
structure SupermarketSales where
  initial_profit : ℕ
  initial_sales : ℕ
  price_reduction : ℕ
  sales_increase_rate : ℕ

/-- Calculates the new sales volume after a price reduction -/
def new_sales_volume (s : SupermarketSales) : ℕ :=
  s.initial_sales + s.price_reduction * s.sales_increase_rate

/-- Calculates the new profit per item after a price reduction -/
def new_profit_per_item (s : SupermarketSales) : ℤ :=
  s.initial_profit - s.price_reduction

/-- Calculates the total daily profit after a price reduction -/
def total_daily_profit (s : SupermarketSales) : ℤ :=
  (new_profit_per_item s) * (new_sales_volume s)

/-- Theorem: A price reduction of $5 results in 40 items sold and $1800 daily profit -/
theorem price_reduction_5 (s : SupermarketSales) 
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2)
  (h4 : s.price_reduction = 5) :
  new_sales_volume s = 40 ∧ total_daily_profit s = 1800 := by sorry

/-- Theorem: A price reduction of $20 results in $2100 daily profit -/
theorem price_reduction_20 (s : SupermarketSales)
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2)
  (h4 : s.price_reduction = 20) :
  total_daily_profit s = 2100 := by sorry

/-- Theorem: There is no price reduction that results in $2200 daily profit -/
theorem no_2200_profit (s : SupermarketSales)
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2) :
  ∀ (x : ℕ), total_daily_profit { s with price_reduction := x } ≠ 2200 := by sorry

end NUMINAMATH_CALUDE_price_reduction_5_price_reduction_20_no_2200_profit_l1567_156740


namespace NUMINAMATH_CALUDE_round_trip_time_l1567_156739

/-- Calculates the time for a round trip given boat speed, current speed, and distance -/
theorem round_trip_time (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) :
  boat_speed = 18 ∧ current_speed = 4 ∧ distance = 85.56 →
  (distance / (boat_speed - current_speed) + distance / (boat_speed + current_speed)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_time_l1567_156739


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l1567_156765

theorem quadratic_root_ratio (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x = 4*y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) →
  16 * b^2 / (a * c) = 100 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l1567_156765


namespace NUMINAMATH_CALUDE_expression_bounds_l1567_156775

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
    Real.sqrt (e^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
  Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
  Real.sqrt (e^2 + (1-a)^2) ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l1567_156775


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1567_156760

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + 3a_8 + a_15 = 60,
    prove that 2a_9 - a_10 = 12 -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1567_156760


namespace NUMINAMATH_CALUDE_plane_equation_correct_l1567_156712

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The foot of the perpendicular from the origin to the plane -/
def footPoint : Point3D := ⟨10, -5, 2⟩

/-- The coefficients of the plane equation -/
def planeCoeffs : PlaneCoefficients := ⟨10, -5, 2, -129⟩

/-- Checks if a point satisfies the plane equation -/
def satisfiesPlaneEquation (p : Point3D) (c : PlaneCoefficients) : Prop :=
  c.A * p.x + c.B * p.y + c.C * p.z + c.D = 0

/-- Checks if a vector is perpendicular to another vector -/
def isPerpendicular (v1 v2 : Point3D) : Prop :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z = 0

/-- Theorem stating that the given plane equation is correct -/
theorem plane_equation_correct :
  satisfiesPlaneEquation footPoint planeCoeffs ∧
  isPerpendicular footPoint ⟨planeCoeffs.A, planeCoeffs.B, planeCoeffs.C⟩ ∧
  planeCoeffs.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs planeCoeffs.A) (Int.natAbs planeCoeffs.B))
          (Nat.gcd (Int.natAbs planeCoeffs.C) (Int.natAbs planeCoeffs.D)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l1567_156712


namespace NUMINAMATH_CALUDE_area_between_line_and_curve_l1567_156773

theorem area_between_line_and_curve : 
  let f (x : ℝ) := 3 * x
  let g (x : ℝ) := x^2
  let lower_bound := (0 : ℝ)
  let upper_bound := (3 : ℝ)
  let area := ∫ x in lower_bound..upper_bound, (f x - g x)
  area = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_between_line_and_curve_l1567_156773


namespace NUMINAMATH_CALUDE_problem_statement_l1567_156709

theorem problem_statement (r p q : ℝ) 
  (hr : r > 0) 
  (hpq : p * q ≠ 0) 
  (hineq : p^2 * r > q^2 * r) : 
  p^2 > q^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1567_156709


namespace NUMINAMATH_CALUDE_mathematicians_set_l1567_156754

-- Define the type for famous figures
inductive FamousFigure
| BillGates
| Gauss
| LiuXiang
| Nobel
| ChenJingrun
| ChenXingshen
| Gorky
| Einstein

-- Define the set of all famous figures
def allFigures : Set FamousFigure :=
  {FamousFigure.BillGates, FamousFigure.Gauss, FamousFigure.LiuXiang, 
   FamousFigure.Nobel, FamousFigure.ChenJingrun, FamousFigure.ChenXingshen, 
   FamousFigure.Gorky, FamousFigure.Einstein}

-- Define the property of being a mathematician
def isMathematician : FamousFigure → Prop :=
  fun figure => match figure with
  | FamousFigure.Gauss => True
  | FamousFigure.ChenJingrun => True
  | FamousFigure.ChenXingshen => True
  | _ => False

-- Theorem: The set of mathematicians is equal to {Gauss, Chen Jingrun, Chen Xingshen}
theorem mathematicians_set :
  {figure ∈ allFigures | isMathematician figure} =
  {FamousFigure.Gauss, FamousFigure.ChenJingrun, FamousFigure.ChenXingshen} :=
by sorry

end NUMINAMATH_CALUDE_mathematicians_set_l1567_156754


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1567_156755

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X : Polynomial ℝ)^5 + 1 = (X^2 - 4*X + 5) * q + 76 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1567_156755


namespace NUMINAMATH_CALUDE_original_triangle_area_l1567_156713

-- Define the properties of the triangle
def is_oblique_projection (original : Real → Real → Real → Real) 
  (projected : Real → Real → Real → Real) : Prop := sorry

def is_equilateral (triangle : Real → Real → Real → Real) : Prop := sorry

def side_length (triangle : Real → Real → Real → Real) : Real := sorry

def area (triangle : Real → Real → Real → Real) : Real := sorry

-- Theorem statement
theorem original_triangle_area 
  (original projected : Real → Real → Real → Real) :
  is_oblique_projection original projected →
  is_equilateral projected →
  side_length projected = 1 →
  (area projected) / (area original) = Real.sqrt 2 / 4 →
  area original = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l1567_156713


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1567_156725

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1567_156725


namespace NUMINAMATH_CALUDE_sum_y_z_is_twice_x_l1567_156701

theorem sum_y_z_is_twice_x (x y z : ℝ) 
  (h1 : 0.6 * (x - y) = 0.3 * (x + y)) 
  (h2 : 0.4 * (x + z) = 0.2 * (y + z)) : 
  (y + z) / x = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_y_z_is_twice_x_l1567_156701


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1567_156738

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The probability of a four-digit palindrome -/
def prob_digit_palindrome : ℚ := 1 / 100

/-- The probability of a four-letter palindrome with at least one 'X' -/
def prob_letter_palindrome : ℚ := 1 / 8784

/-- The probability of both four-digit and four-letter palindromes occurring -/
def prob_both_palindromes : ℚ := prob_digit_palindrome * prob_letter_palindrome

/-- The probability of at least one palindrome in a license plate -/
def prob_at_least_one_palindrome : ℚ := 
  prob_digit_palindrome + prob_letter_palindrome - prob_both_palindromes

theorem license_plate_palindrome_probability :
  prob_at_least_one_palindrome = 8883 / 878400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1567_156738


namespace NUMINAMATH_CALUDE_parallel_vectors_k_l1567_156734

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k (k : ℝ) :
  let a : ℝ × ℝ := (2*k + 2, 4)
  let b : ℝ × ℝ := (k + 1, 8)
  parallel a b → k = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_l1567_156734


namespace NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_D_l1567_156785

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B ↔ C))
variable (h3 : (D → C) ∧ ¬(C → D))

-- Theorem to prove
theorem A_sufficient_not_necessary_for_D : 
  (A → D) ∧ ¬(D → A) :=
sorry

end NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_D_l1567_156785


namespace NUMINAMATH_CALUDE_stream_speed_ratio_l1567_156724

/-- Given a boat and a stream where:
  1. It takes twice as long to row against the stream as with it for the same distance.
  2. The speed of the boat in still water is three times the speed of the stream.
  This theorem proves that the speed of the stream is one-third of the speed of the boat in still water. -/
theorem stream_speed_ratio (B S : ℝ) (h1 : B = 3 * S) 
  (h2 : (1 : ℝ) / (B - S) = 2 * (1 / (B + S))) : S / B = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_ratio_l1567_156724


namespace NUMINAMATH_CALUDE_product_of_squares_l1567_156789

theorem product_of_squares (N : ℕ+) 
  (h : ∃! (a₁ b₁ a₂ b₂ a₃ b₃ : ℕ+), 
    a₁^2 * b₁^2 = N ∧ 
    a₂^2 * b₂^2 = N ∧ 
    a₃^2 * b₃^2 = N ∧
    (a₁, b₁) ≠ (a₂, b₂) ∧ 
    (a₁, b₁) ≠ (a₃, b₃) ∧ 
    (a₂, b₂) ≠ (a₃, b₃)) :
  ∃ (a₁ b₁ a₂ b₂ a₃ b₃ : ℕ+), 
    a₁^2 * b₁^2 * a₂^2 * b₂^2 * a₃^2 * b₃^2 = N^3 :=
sorry

end NUMINAMATH_CALUDE_product_of_squares_l1567_156789


namespace NUMINAMATH_CALUDE_gauss_family_mean_age_l1567_156783

/-- The ages of the Gauss family children -/
def gauss_ages : List ℕ := [7, 7, 7, 14, 15]

/-- The number of children in the Gauss family -/
def num_children : ℕ := gauss_ages.length

/-- The mean age of the Gauss family children -/
def mean_age : ℚ := (gauss_ages.sum : ℚ) / num_children

theorem gauss_family_mean_age : mean_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_gauss_family_mean_age_l1567_156783


namespace NUMINAMATH_CALUDE_differential_pricing_profitability_l1567_156751

theorem differential_pricing_profitability 
  (n : ℝ) (t : ℝ) (h_n_pos : n > 0) (h_t_pos : t > 0) : 
  let shorts_ratio : ℝ := 0.75
  let suits_ratio : ℝ := 0.25
  let businessmen_ratio : ℝ := 0.8
  let tourists_ratio : ℝ := 0.2
  let uniform_revenue := n * t
  let differential_revenue (X : ℝ) := 
    (shorts_ratio * n * t) + 
    (suits_ratio * businessmen_ratio * n * (t + X))
  ∃ X : ℝ, X ≥ 0 ∧ 
    ∀ Y : ℝ, Y ≥ 0 → differential_revenue Y ≥ uniform_revenue → Y ≥ X ∧
    differential_revenue X = uniform_revenue ∧
    X = t / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_differential_pricing_profitability_l1567_156751


namespace NUMINAMATH_CALUDE_integer_product_condition_l1567_156747

theorem integer_product_condition (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 3) := by
sorry

end NUMINAMATH_CALUDE_integer_product_condition_l1567_156747


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1567_156743

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l
def line_l (x y : ℝ) (m : ℝ) : Prop := x = m * y + 1

-- Define perpendicularity of vectors
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem ellipse_and_line_theorem :
  -- Conditions
  (ellipse_C (-2) 0) ∧
  (ellipse_C (Real.sqrt 2) ((Real.sqrt 2) / 2)) ∧
  -- Existence of intersection points M and N
  ∃ (x1 y1 x2 y2 : ℝ),
    (ellipse_C x1 y1) ∧
    (ellipse_C x2 y2) ∧
    (∃ (m : ℝ), line_l x1 y1 m ∧ line_l x2 y2 m) ∧
    -- M and N are distinct
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    -- OM ⊥ ON
    (perpendicular x1 y1 x2 y2) →
  -- Conclusion
  ∃ (m : ℝ), (m = 1/2 ∨ m = -1/2) ∧ line_l 1 0 m := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1567_156743


namespace NUMINAMATH_CALUDE_xy_greater_than_xz_l1567_156759

theorem xy_greater_than_xz (x y z : ℝ) 
  (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 1) : x * y > x * z := by
  sorry

end NUMINAMATH_CALUDE_xy_greater_than_xz_l1567_156759


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l1567_156714

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The triangle PQR with given coordinates -/
def trianglePQR : Triangle :=
  { P := (0, 10)
    Q := (3, 0)
    R := (9, 0) }

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line that bisects the area of triangle PQR and passes through Q -/
def bisectingLine (t : Triangle) : Line :=
  sorry

/-- Theorem: The sum of the slope and y-intercept of the bisecting line is -20/3 -/
theorem bisecting_line_sum (t : Triangle) (h : t = trianglePQR) :
    (bisectingLine t).slope + (bisectingLine t).yIntercept = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l1567_156714


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1567_156770

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Define what it means for a color to appear on infinitely many lines
def AppearsOnInfinitelyManyLines (f : ColoringFunction) (c : Color) : Prop :=
  ∀ n : ℕ, ∃ y : ℤ, y > n ∧ (∀ m : ℕ, ∃ x : ℤ, x > m ∧ f (x, y) = c)

-- Define what it means to be a parallelogram
def IsParallelogram (A B C D : ℤ × ℤ) : Prop :=
  B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2

-- Main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (AppearsOnInfinitelyManyLines f Color.White) ∧
  (AppearsOnInfinitelyManyLines f Color.Red) ∧
  (AppearsOnInfinitelyManyLines f Color.Black) ∧
  (∀ A B C : ℤ × ℤ, f A = Color.White → f B = Color.Red → f C = Color.Black →
    ∃ D : ℤ × ℤ, f D = Color.Red ∧ IsParallelogram A B C D) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1567_156770


namespace NUMINAMATH_CALUDE_roots_difference_implies_k_value_l1567_156733

theorem roots_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, 
    (r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0) ∧ 
    ((r+4)^2 - k*(r+4) + 10 = 0 ∧ (s+4)^2 - k*(s+4) + 10 = 0)) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_roots_difference_implies_k_value_l1567_156733


namespace NUMINAMATH_CALUDE_base7_addition_subtraction_l1567_156798

-- Define a function to convert base 7 to decimal
def base7ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Define a function to convert decimal to base 7
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

-- Define the numbers in base 7
def n1 : List Nat := [0, 0, 0, 1]  -- 1000₇
def n2 : List Nat := [6, 6, 6]     -- 666₇
def n3 : List Nat := [4, 3, 2, 1]  -- 1234₇

-- State the theorem
theorem base7_addition_subtraction :
  decimalToBase7 (base7ToDecimal n1 + base7ToDecimal n2 - base7ToDecimal n3) = [4, 5, 2] := by
  sorry

end NUMINAMATH_CALUDE_base7_addition_subtraction_l1567_156798


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1567_156784

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ a, (0 < a ∧ a < 1) → (a + 1) * (a - 2) < 0) ∧
  (∃ a, (a + 1) * (a - 2) < 0 ∧ (a ≤ 0 ∨ a ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1567_156784


namespace NUMINAMATH_CALUDE_ideal_function_fixed_point_l1567_156774

/-- An ideal function is a function f: [0,1] → ℝ satisfying:
    1) ∀ x ∈ [0,1], f(x) ≥ 0
    2) f(1) = 1
    3) ∀ x₁ x₂ ≥ 0 with x₁ + x₂ ≤ 1, f(x₁ + x₂) ≥ f(x₁) + f(x₂) -/
def IdealFunction (f : Real → Real) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧ 
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

theorem ideal_function_fixed_point 
  (f : Real → Real) (h : IdealFunction f) 
  (x₀ : Real) (hx₀ : x₀ ∈ Set.Icc 0 1) 
  (hfx₀ : f x₀ ∈ Set.Icc 0 1) (hffx₀ : f (f x₀) = x₀) : 
  f x₀ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_ideal_function_fixed_point_l1567_156774


namespace NUMINAMATH_CALUDE_n_is_composite_l1567_156748

/-- The number of zeros in the given number -/
def num_zeros : ℕ := 2^1974 + 2^1000 - 1

/-- The number to be proven composite -/
def n : ℕ := 10^(num_zeros + 1) + 1

/-- Theorem stating that n is composite -/
theorem n_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b :=
sorry

end NUMINAMATH_CALUDE_n_is_composite_l1567_156748


namespace NUMINAMATH_CALUDE_transformed_circle_center_l1567_156716

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

def circle_center : ℝ × ℝ := (4, -3)

theorem transformed_circle_center :
  let reflected := reflect_x circle_center
  let translated_right := translate reflected 5 0
  let final_position := translate translated_right 0 3
  final_position = (9, 6) := by sorry

end NUMINAMATH_CALUDE_transformed_circle_center_l1567_156716


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1567_156795

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the condition given in the problem
def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = (16 : ℝ) ^ n

-- Theorem statement
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : satisfies_condition a) : 
  ∃ r : ℝ, (∀ n : ℕ, a (n + 1) = r * a n) ∧ r = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1567_156795


namespace NUMINAMATH_CALUDE_twenty_four_is_seventy_five_percent_of_thirty_two_l1567_156797

theorem twenty_four_is_seventy_five_percent_of_thirty_two (x : ℝ) :
  24 / x = 75 / 100 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_is_seventy_five_percent_of_thirty_two_l1567_156797


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l1567_156778

theorem simplify_nested_roots : 
  (65536 : ℝ) = 2^16 →
  (((1 / 65536)^(1/2))^(1/3))^(1/4) = 1 / (4^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l1567_156778
