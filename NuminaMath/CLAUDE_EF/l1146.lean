import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_three_people_6x5_l1146_114658

/-- The number of ways to select 3 people from a rows x columns grid, 
    such that no two selected people are in the same row or column -/
def select_three_people (rows : Nat) (columns : Nat) : Nat :=
  Nat.choose rows 3 * Nat.choose columns 3 * Nat.factorial 3

/-- Theorem stating that the number of ways to select 3 people 
    from a 6x5 grid, with the given constraints, is 1200 -/
theorem select_three_people_6x5 : 
  select_three_people 6 5 = 1200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_three_people_6x5_l1146_114658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1146_114675

-- Define the train length in meters
noncomputable def train_length : ℝ := 200

-- Define the train speed in kilometers per hour
noncomputable def train_speed_kmph : ℝ := 80

-- Define the platform length in meters
noncomputable def platform_length : ℝ := 288.928

-- Define the function to convert speed from kmph to m/s
noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Define the total distance the train needs to travel
noncomputable def total_distance : ℝ :=
  train_length + platform_length

-- Define the time taken for the train to cross the platform
noncomputable def crossing_time : ℝ :=
  total_distance / (kmph_to_mps train_speed_kmph)

-- Theorem stating that the crossing time is approximately 22 seconds
theorem train_crossing_time_approx :
  ∃ ε > 0, |crossing_time - 22| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l1146_114675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_sampling_percentage_l1146_114691

theorem candy_sampling_percentage (total_sample_percent : ℝ) 
  (not_caught_percent : ℝ) (caught_percent : ℝ) : Prop :=
  total_sample_percent = 27.5 ∧
  not_caught_percent = 20 ∧
  caught_percent = 22 ∧
  caught_percent = total_sample_percent * (100 - not_caught_percent) / 100

example : ∃ (total_sample_percent not_caught_percent caught_percent : ℝ),
  candy_sampling_percentage total_sample_percent not_caught_percent caught_percent := by
  use 27.5, 20, 22
  simp [candy_sampling_percentage]
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_sampling_percentage_l1146_114691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1146_114615

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 + 32 * y = 144

/-- The distance between the foci of the hyperbola -/
noncomputable def foci_distance : ℝ := 38 * Real.sqrt 7 / 72

/-- Theorem stating that the distance between the foci of the hyperbola
    defined by the given equation is equal to (38√7)/72 -/
theorem hyperbola_foci_distance :
  ∀ x y : ℝ, hyperbola_equation x y →
  (∃ f₁ f₂ : ℝ × ℝ, (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = foci_distance^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1146_114615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_100_divisors_l1146_114695

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ :=
  (Finset.filter (· ∣ n.val) (Finset.range n.val)).card + 1

/-- A positive integer n has exactly 100 positive divisors -/
def has_100_divisors (n : ℕ+) : Prop :=
  num_divisors n = 100

theorem smallest_with_100_divisors :
  ∃ (n : ℕ+), has_100_divisors n ∧ ∀ (m : ℕ+), has_100_divisors m → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_100_divisors_l1146_114695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_for_increasing_f_l1146_114642

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x) / Real.log a

-- State the theorem
theorem exists_a_for_increasing_f :
  ∃ a : ℝ, a > 1 ∧ 
    ∀ x y : ℝ, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f a x < f a y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_for_increasing_f_l1146_114642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_alpha_l1146_114609

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 4) + 1
noncomputable def g (α : ℝ) (x : ℝ) : ℝ := x^α

-- State the theorem
theorem fixed_point_alpha (a : ℝ) (α : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∃ x y : ℝ, f a x = y ∧ g α x = y) → α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_alpha_l1146_114609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1146_114645

/-- The area of the region enclosed by the given equation. -/
noncomputable def area_of_region_enclosed_by (equation : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem stating that the area of the region enclosed by the given equation is 265π/4. -/
theorem area_of_region : 
  let equation := fun x y : ℝ => x^2 + y^2 + 10*x + 2 = 5*y - 4*x + 13
  area_of_region_enclosed_by equation = 265 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1146_114645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1146_114663

/-- Given a geometric sequence {a_n} with a₁ = 2/3 and a₄ = ∫₁⁴(1+2x)dx, the common ratio is 3. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- condition for geometric sequence
  a 1 = 2/3 →
  a 4 = ∫ x in (1 : ℝ)..4, (1 + 2*x) →
  a 2 / a 1 = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1146_114663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_neither_sufficient_nor_necessary_l1146_114640

-- Define a triangle
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides
  (angle_sum : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define the condition acos(A) = bcos(B)
def Condition (t : Triangle) : Prop :=
  t.a * Real.cos t.A = t.b * Real.cos t.B

-- Theorem statement
theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ t : Triangle, Condition t → IsIsosceles t) ∧
  ¬(∀ t : Triangle, IsIsosceles t → Condition t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_neither_sufficient_nor_necessary_l1146_114640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platforms_l1146_114618

/-- Calculates the time for a train to cross a platform given the train's speed -/
noncomputable def time_to_cross (train_length : ℝ) (platform_length : ℝ) (train_speed : ℝ) : ℝ :=
  (train_length + platform_length) / train_speed

/-- Proves that a train crossing two platforms of different lengths takes the calculated time -/
theorem train_crossing_platforms 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (h1 : train_length = 190) 
  (h2 : platform1_length = 140) 
  (h3 : platform2_length = 250) 
  (h4 : time1 = 15) :
  time_to_cross train_length platform2_length 
    ((train_length + platform1_length) / time1) = 20 := by
  sorry

#check train_crossing_platforms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platforms_l1146_114618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_l1146_114673

theorem intersection_size (U A B : Finset ℕ) : 
  (U.card = 193) →
  (B.card = 49) →
  ((U \ (A ∪ B)).card = 59) →
  (A.card = 110) →
  ((A ∩ B).card = 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_l1146_114673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1146_114628

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (f x + f y) = f (f x) + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1146_114628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_sqrt3_l1146_114643

open Real

-- Define the expressions
noncomputable def expressionA : ℝ := sqrt 2 * sin (15 * π / 180) + sqrt 2 * cos (15 * π / 180)
noncomputable def expressionD : ℝ := (1 + tan (15 * π / 180)) / (1 - tan (15 * π / 180))

-- State the theorem
theorem expressions_equal_sqrt3 : 
  expressionA = sqrt 3 ∧ expressionD = sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_sqrt3_l1146_114643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l1146_114694

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - m * x - 1

theorem f_inequality_solution (m : ℝ) :
  (m = -3 → {x : ℝ | f m x ≤ 0} = {x : ℝ | x ≥ 1 ∨ x ≤ 1/2}) ∧
  ({x : ℝ | f m x + m > 0} = ∅ ↔ m ∈ Set.Iic (-2 * Real.sqrt 3 / 3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l1146_114694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_incorrect_answers_to_pass_l1146_114683

/-- Represents an exam with the given conditions -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ
  unanswered_score : ℤ
  passing_score : ℤ
  min_correct : ℕ

/-- Represents a student's exam performance -/
structure ExamPerformance where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ

/-- Calculates the total score for a given exam performance -/
def calculate_score (exam : Exam) (performance : ExamPerformance) : ℤ :=
  exam.correct_score * (performance.correct : ℤ) +
  exam.incorrect_score * (performance.incorrect : ℤ) +
  exam.unanswered_score * (performance.unanswered : ℤ)

/-- Checks if a performance passes the exam -/
def passes_exam (exam : Exam) (performance : ExamPerformance) : Prop :=
  calculate_score exam performance ≥ exam.passing_score ∧
  performance.correct ≥ exam.min_correct ∧
  performance.correct + performance.incorrect + performance.unanswered = exam.total_questions

/-- The main theorem stating the maximum number of incorrect answers while still passing -/
theorem max_incorrect_answers_to_pass (exam : Exam) 
  (h_exam : exam.total_questions = 30 ∧ 
            exam.correct_score = 4 ∧ 
            exam.incorrect_score = -1 ∧ 
            exam.unanswered_score = 0 ∧ 
            exam.passing_score = 85 ∧ 
            exam.min_correct = 22) : 
  ∃ (performance : ExamPerformance), 
    passes_exam exam performance ∧ 
    performance.incorrect = 3 ∧ 
    ∀ (other_performance : ExamPerformance), 
      passes_exam exam other_performance → 
      other_performance.incorrect ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_incorrect_answers_to_pass_l1146_114683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_theta_l1146_114624

theorem cos_pi_third_plus_theta (θ : ℝ) 
  (h : Real.cos (π/6 - θ) = 2 * Real.sqrt 2 / 3) : 
  Real.cos (π/3 + θ) = 1/3 ∨ Real.cos (π/3 + θ) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_theta_l1146_114624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_rectangle_area_l1146_114607

/-- Given a triangle with base b and altitude h, and two rectangles inscribed in it,
    where the first rectangle has height x and the second rectangle has height 2x,
    prove that the area of the second rectangle is (2bx(h-3x))/h. -/
theorem second_rectangle_area (b h x : ℝ) (hb : b > 0) (hh : h > 0) (hx : x > 0) (hx_lt_h : 3*x < h) :
  2 * x * (b * (h - 3*x) / h) = (2 * b * x * (h - 3*x)) / h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_rectangle_area_l1146_114607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_perpendicular_to_polar_axis_l1146_114674

-- Define the circle in polar coordinates
def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define a line perpendicular to the polar axis
def perpendicular_to_polar_axis (ρ θ : ℝ) : Prop := ∃ (k : ℝ), ρ * Real.cos θ = k

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 0)

-- State the theorem
theorem line_through_circle_center_perpendicular_to_polar_axis 
  (ρ θ : ℝ) : 
  (perpendicular_to_polar_axis ρ θ ∧ 
   (ρ * Real.cos θ, ρ * Real.sin θ) = circle_center) → 
  ρ * Real.cos θ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_perpendicular_to_polar_axis_l1146_114674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_480km_l1146_114650

/-- Represents a train with a departure time and speed -/
structure Train where
  departureTime : ℚ
  speed : ℚ

/-- Calculates the meeting point of two trains -/
noncomputable def meetingPoint (trainA trainB : Train) : ℚ :=
  let timeDiff := trainB.departureTime - trainA.departureTime
  let distanceAhead := trainA.speed * timeDiff
  let relativeSpeed := trainB.speed - trainA.speed
  let catchUpTime := distanceAhead / relativeSpeed
  trainB.speed * catchUpTime

/-- Theorem stating that the trains meet 480 km from Delhi -/
theorem trains_meet_at_480km (trainA trainB : Train) 
  (h1 : trainA.departureTime = 0)
  (h2 : trainB.departureTime = 2)
  (h3 : trainA.speed = 60)
  (h4 : trainB.speed = 80) : 
  meetingPoint trainA trainB = 480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_480km_l1146_114650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_theorem_l1146_114623

-- Define the given conditions
noncomputable def marguerite_distance : ℝ := 150
noncomputable def marguerite_time : ℝ := 3
noncomputable def sam_total_time : ℝ := 4
noncomputable def sam_stop_time : ℝ := 0.5

-- Define the average speed
noncomputable def average_speed : ℝ := marguerite_distance / marguerite_time

-- Define Sam's actual driving time
noncomputable def sam_driving_time : ℝ := sam_total_time - sam_stop_time

-- Theorem to prove
theorem sam_distance_theorem : 
  average_speed * sam_driving_time = 175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_theorem_l1146_114623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1146_114611

theorem equation_solutions : 
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, x ≠ 2 ∧ x ≠ 5 ∧ (3*x^2 - 15*x) / (x^2 - 7*x + 10) = x - 4) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1146_114611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_root_l1146_114678

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Define a real constant k
variable (k : ℝ)

-- State the theorem
theorem inverse_function_root 
  (h1 : f 5 + 5 = k)  -- 5 is a root of f(x) + x = k
  (h2 : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f)  -- f_inv is the inverse function of f
  : f_inv (k - 5) + (k - 5) = k :=  -- k - 5 is a root of f^(-1)(x) + x = k
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_root_l1146_114678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_count_l1146_114657

/-- Represents the number of chocolates in a box -/
def Chocolates : Type := ℕ

/-- Represents the original number of chocolates in the box -/
def original_count (c : Chocolates) : ℕ := c

/-- Represents the number of chocolates with nuts -/
def with_nuts (c : Chocolates) : ℕ := original_count c / 2

/-- Represents the number of chocolates without nuts -/
def without_nuts (c : Chocolates) : ℕ := original_count c / 2

/-- Represents the number of chocolates with nuts that were eaten -/
def eaten_with_nuts (c : Chocolates) : ℕ := (with_nuts c * 4) / 5

/-- Represents the number of chocolates without nuts that were eaten -/
def eaten_without_nuts (c : Chocolates) : ℕ := without_nuts c / 2

/-- Represents the total number of chocolates eaten -/
def total_eaten (c : Chocolates) : ℕ := eaten_with_nuts c + eaten_without_nuts c

/-- Represents the number of chocolates left -/
def left (c : Chocolates) : ℕ := original_count c - total_eaten c

/-- Theorem stating that given the conditions, the original count of chocolates is 80 -/
theorem chocolate_count (c : Chocolates) : left c = 28 → original_count c = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_count_l1146_114657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l1146_114679

theorem percentage_difference (x y z n : ℝ) : 
  x = 8 * y →
  y = 2 * abs (z - n) →
  z = 1.1 * n →
  let y_decreased := 0.75 * y
  (x - y_decreased) / x * 100 = 90.625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l1146_114679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_period_days_l1146_114697

/-- The number of days required for a given percentage of water to evaporate -/
noncomputable def evaporation_days (initial_amount : ℝ) (daily_rate : ℝ) (evaporation_percentage : ℝ) : ℝ :=
  (evaporation_percentage / 100) * initial_amount / daily_rate

/-- Theorem: The number of days for 2% of 10 ounces to evaporate at 0.004 ounce per day is 50 -/
theorem evaporation_period_days : 
  evaporation_days 10 0.004 2 = 50 := by
  -- Unfold the definition of evaporation_days
  unfold evaporation_days
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_period_days_l1146_114697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1146_114680

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - (floor x : ℝ)

-- Theorem statement
theorem f_properties :
  (f (-0.5) = 0.5) ∧
  (Set.range f = Set.Ioc 0 1) ∧
  (StrictMonoOn f (Set.Ioo (-2) (-1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1146_114680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_field_area_is_1440_l1146_114664

def farm_field_area (planned_rate : ℕ) (actual_rate : ℕ) (extra_days : ℕ) (remaining : ℕ) : ℕ :=
  let total_area := planned_rate * ((actual_rate * (extra_days + planned_rate / actual_rate) + remaining - planned_rate) / (planned_rate - actual_rate))
  total_area + remaining

theorem farm_field_area_is_1440 :
  farm_field_area 100 85 2 40 = 1440 := by
  sorry

#eval farm_field_area 100 85 2 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_field_area_is_1440_l1146_114664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_equivalence_range_a_B_subset_A_range_a_A_union_B_eq_R_range_a_A_inter_B_nonempty_l1146_114662

-- Define sets A and B
def A : Set ℝ := {x | -x^2 + 2*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | a - 2 ≤ x ∧ x ≤ 2*a + 3}

-- Theorem 1: Set A equivalence
theorem set_A_equivalence : A = {x | x < -1 ∨ x > 3} := by sorry

-- Theorem 2: Range of a for B ⊆ A
theorem range_a_B_subset_A : ∀ a : ℝ, (B a ⊆ A) ↔ (a < -2 ∨ a > 5) := by sorry

-- Theorem 3: Range of a for A ∪ B = ℝ
theorem range_a_A_union_B_eq_R : ∀ a : ℝ, (A ∪ B a = Set.univ) ↔ (0 ≤ a ∧ a ≤ 1) := by sorry

-- Theorem 4: Range of a for A ∩ B ≠ ∅
theorem range_a_A_inter_B_nonempty : ∀ a : ℝ, (A ∩ B a).Nonempty ↔ (a ≥ -5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_equivalence_range_a_B_subset_A_range_a_A_union_B_eq_R_range_a_A_inter_B_nonempty_l1146_114662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_consumption_wednesday_l1146_114670

/-- Represents the relationship between data analysis hours and tea consumption. -/
structure DataAnalysisTeaRelation where
  k : ℝ
  h_pos : k > 0

/-- The tea consumption for a given number of data analysis hours. -/
noncomputable def tea_consumption (r : DataAnalysisTeaRelation) (hours : ℝ) : ℝ :=
  r.k / hours

theorem tea_consumption_wednesday 
  (r : DataAnalysisTeaRelation)
  (h_sunday : tea_consumption r 12 = 1.5)
  (h_total_hours : ℝ)
  (h_total_positive : h_total_hours > 0)
  : tea_consumption r 8 = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_consumption_wednesday_l1146_114670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_boy_unskilled_negates_all_children_skilled_l1146_114630

-- Define the universe of discourse
variable {U : Type}

-- Define predicates
variable (Boy : U → Prop)
variable (SkilledPainter : U → Prop)

-- Define the statements
def AllChildrenSkilled (Boy SkilledPainter : U → Prop) : Prop := 
  ∀ x : U, (Boy x ∨ ¬Boy x) → SkilledPainter x

def AtLeastOneBoyUnskilled (Boy SkilledPainter : U → Prop) : Prop := 
  ∃ x : U, Boy x ∧ ¬SkilledPainter x

-- Theorem statement
theorem at_least_one_boy_unskilled_negates_all_children_skilled 
  (Boy SkilledPainter : U → Prop) :
  AtLeastOneBoyUnskilled Boy SkilledPainter ↔ ¬AllChildrenSkilled Boy SkilledPainter :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_boy_unskilled_negates_all_children_skilled_l1146_114630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1146_114653

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y A B C : ℝ) : ℝ :=
  |A * x + B * y + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from point P(1, -4) to the line 4x + 3y - 2 = 0 is 2 -/
theorem distance_point_to_line_example : distance_point_to_line 1 (-4) 4 3 (-2) = 2 := by
  -- Unfold the definition of distance_point_to_line
  unfold distance_point_to_line
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1146_114653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l1146_114685

open Matrix

theorem matrix_transformation (a b c d : ℝ) : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 0, 1]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  N * M = !![a + 3*b, b; c + 3*d, d] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l1146_114685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombusAreaApprox_l1146_114612

/-- The area of a rhombus with side length 8 cm and an acute angle of 55 degrees -/
noncomputable def rhombusArea : ℝ := 
  let sideLength : ℝ := 8
  let acuteAngle : ℝ := 55 * Real.pi / 180  -- Convert degrees to radians
  sideLength^2 * Real.sin acuteAngle

/-- Theorem stating that the area of the rhombus is approximately 52.4288 square centimeters -/
theorem rhombusAreaApprox : 
  ‖rhombusArea - 52.4288‖ < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombusAreaApprox_l1146_114612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_zero_has_three_roots_l1146_114646

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else Real.sqrt x - 1

-- Theorem statement
theorem f_f_eq_zero_has_three_roots :
  ∃ (a b c : ℝ), a < b ∧ b < c ∧
  (∀ x : ℝ, f (f x) = 0 ↔ x = a ∨ x = b ∨ x = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_zero_has_three_roots_l1146_114646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_equation_C_equation_line_l_length_chord_EF_l1146_114627

-- Define the conic curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- Define the fixed point A
noncomputable def A : ℝ × ℝ := (0, -Real.sqrt 3)

-- Define the left focus F₁
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)

-- Define the line l passing through A and F₁
def l (x y : ℝ) : Prop := x + y + Real.sqrt 3 = 0

-- Theorem for the general equation of C
theorem general_equation_C (x y : ℝ) : 
  (∃ θ : ℝ, C θ = (x, y)) ↔ x^2/4 + y^2 = 1 := by sorry

-- Theorem for the equation of line l
theorem equation_line_l (x y : ℝ) : 
  l x y ↔ x + y + Real.sqrt 3 = 0 := by sorry

-- Theorem for the length of chord EF
theorem length_chord_EF : 
  let intersections := {(x, y) | l x y ∧ (∃ θ : ℝ, C θ = (x, y))}
  ∃ E F : ℝ × ℝ, E ∈ intersections ∧ F ∈ intersections ∧ 
    Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = 8/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_equation_C_equation_line_l_length_chord_EF_l1146_114627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_l1146_114648

/-- A circle in the xy-plane with diameter endpoints (-8, 0) and (32, 0) -/
noncomputable def circle_diameter : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p = ((-8 : ℝ) * (1 - t) + 32 * t, 0)}

/-- The center of the circle -/
noncomputable def circle_center : ℝ × ℝ := ((32 - (-8)) / 2, 0)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := (32 - (-8)) / 2

/-- A point lies on the circle if its distance from the center equals the radius -/
def on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

theorem point_on_circle (x : ℝ) :
  on_circle (x, 20) → x = 12 := by
  sorry

#check point_on_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_l1146_114648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l1146_114631

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_equality (y : ℝ) (h : y = 8.4) :
  (floor 6.5 : ℝ) * (floor (2 / 3) : ℝ) + (floor 2 : ℝ) * 7.2 + (floor y : ℝ) - 6.0 = 16.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l1146_114631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1146_114672

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2*x + 5

-- State the theorem
theorem f_properties :
  (∀ x y, x < y ∧ x < -2/3 ∧ y < -2/3 → f x < f y) ∧
  (∀ x y, x < y ∧ x > 1 ∧ y > 1 → f x < f y) ∧
  (∀ x y, x < y ∧ x > -2/3 ∧ y < 1 → f x > f y) ∧
  (∀ x, x ∈ Set.Icc (-1) 2 → f x ≤ 7) ∧
  (f 2 = 7) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1146_114672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1146_114647

/-- The area of the triangle formed by the x-intercept, y-intercept, and origin of the linear function y = kx + k + 1 -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  1/2 * (2 + k + 1/k)

/-- Theorem stating that the minimum area of the triangle is 2 when k > 0 -/
theorem min_triangle_area :
  ∀ k : ℝ, k > 0 → triangle_area k ≥ 2 ∧ ∃ k₀ : ℝ, k₀ > 0 ∧ triangle_area k₀ = 2 := by
  sorry

#check min_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1146_114647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l1146_114661

/-- Three points in 3D space are collinear if the vectors between them are proportional -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ 
    (p2.fst - p1.fst, p2.snd - p1.snd, p2.2.2 - p1.2.2) = 
    k • (p3.fst - p1.fst, p3.snd - p1.snd, p3.2.2 - p1.2.2)

/-- If the points (1, a, b), (a, b, 3), and (b, 3, a) are collinear, then b + a = 3 -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (1, a, b) (a, b, 3) (b, 3, a) → b + a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l1146_114661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1146_114638

noncomputable def f (x : ℝ) : ℝ := (2 * Real.sin x * Real.cos x ^ 2) / (1 + Real.sin x)

theorem f_range : 
  ∀ y : ℝ, (∃ x : ℝ, f x = y) → -4 < y ∧ y ≤ 1/2 ∧
  (∀ ε > 0, ∃ x₁ x₂ : ℝ, f x₁ > -4 - ε ∧ f x₂ < 1/2 + ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1146_114638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_students_l1146_114665

theorem smallest_possible_students (n : ℕ) : n ≥ 200 → (
  let total_attended : ℕ := n / 4
  let both_competitions : ℕ := n / 40
  let hinting_competition : ℕ := (33 * n) / 200
  let cheating_competition : ℕ := (11 * n) / 100
  (3 * n) / 4 = n - total_attended ∧
  both_competitions = (total_attended * 10) / 100 ∧
  hinting_competition = (3 * cheating_competition) / 2 ∧
  hinting_competition + cheating_competition - both_competitions = total_attended ∧
  hinting_competition % 1 = 0 ∧
  cheating_competition % 1 = 0
) → n ≥ 200 := by
  sorry

#check smallest_possible_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_students_l1146_114665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_odd_polygon_l1146_114603

/-- The probability of two randomly chosen diagonals intersecting in a convex polygon with 2n + 1 vertices -/
theorem intersection_probability_odd_polygon (n : ℕ) : 
  let vertices := 2 * n + 1
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let intersecting_diagonals := Nat.choose vertices 4
  let Prob : ℚ := (n * (2 * n - 1)) / (3 * (2 * n^2 - n - 2))
  (↑intersecting_diagonals : ℚ) / (↑(Nat.choose total_diagonals 2) : ℚ) = Prob :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_odd_polygon_l1146_114603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_price_l1146_114652

theorem discount_price (a : ℝ) (h : a > 0) : 
  ∃ original_price : ℝ, original_price * (1 - 0.2) = a ∧ original_price = (5/4) * a := by
  use (5/4) * a
  constructor
  · -- Prove that (5/4) * a * (1 - 0.2) = a
    sorry
  · -- Prove that (5/4) * a = (5/4) * a
    rfl

#check discount_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_price_l1146_114652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_function_l1146_114622

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * (x + φ) - Real.pi / 3)

theorem min_phi_for_odd_function :
  ∀ φ : ℝ, φ > 0 →
  (∀ x : ℝ, g φ (-x) = -(g φ x)) →
  φ ≥ Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_function_l1146_114622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l1146_114632

/-- Calculates the annual interest rate given the principal, final amount, time, and compounding frequency. -/
noncomputable def calculate_interest_rate (principal : ℝ) (final_amount : ℝ) (time : ℝ) (compounding_frequency : ℝ) : ℝ :=
  ((final_amount / principal) ^ (1 / (compounding_frequency * time)) - 1) * compounding_frequency

theorem interest_rate_is_five_percent (principal : ℝ) (final_amount : ℝ) (time : ℝ) (compounding_frequency : ℝ) :
  principal = 2800 →
  final_amount = 3087 →
  time = 2 →
  compounding_frequency = 1 →
  calculate_interest_rate principal final_amount time compounding_frequency = 0.05 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 2800 3087 2 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l1146_114632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_bounded_region_l1146_114682

/-- The region in three-dimensional space bounded by the given inequalities -/
def BoundedRegion : Set (Fin 3 → ℝ) :=
  {p | (|p 0| + |p 1| + |p 2| ≤ 2) ∧ (|p 0| + |p 1| + |p 2 - 2| ≤ 2)}

/-- The volume of the region bounded by the given inequalities -/
theorem volume_of_bounded_region :
  MeasureTheory.volume BoundedRegion = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_bounded_region_l1146_114682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_l1146_114654

noncomputable def series_sum (n : ℕ) : ℚ :=
  if n % 2 = 0
  then (n + 1) / (2 ^ (n + 2))
  else (n + 1) / (3 ^ n)

noncomputable def infinite_series_sum : ℚ := ∑' n, series_sum n

theorem sum_of_fractions (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : Nat.Coprime a b) 
  (h4 : (a : ℚ) / (b : ℚ) = infinite_series_sum) : 
  a + b = 241 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_l1146_114654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_numbers_product_theorem_l1146_114692

theorem ten_numbers_product_theorem (S : Finset ℝ) :
  (S.card = 10) →
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → x > 0 ∧ y > 0) →
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y) →
  (∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    ∀ d e, d ∈ S ∧ e ∈ S ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧ e ≠ a ∧ e ≠ b ∧ e ≠ c → a * b * c > d * e) ∨
  (∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    ∀ d e f g, d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧
      d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
      e ≠ a ∧ e ≠ b ∧ e ≠ c ∧
      f ≠ a ∧ f ≠ b ∧ f ≠ c ∧
      g ≠ a ∧ g ≠ b ∧ g ≠ c →
      a * b * c > d * e * f * g) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_numbers_product_theorem_l1146_114692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_11_value_l1146_114605

def is_arithmetic_sequence (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) - s n = d

theorem a_11_value (a : ℕ → ℚ) :
  is_arithmetic_sequence (λ n ↦ 1 / (a n + 1)) →
  a 3 = 2 →
  a 7 = 1 →
  a 11 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_11_value_l1146_114605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_special_binomial_expansion_l1146_114688

/-- 
Given a positive integer n where in the binomial expansion of (x-1)^n, 
only the 5th binomial coefficient is the largest, 
the constant term in the binomial expansion of (2√x - 1/√x)^n is 1120.
-/
theorem constant_term_special_binomial_expansion (n : ℕ+) 
  (h : ∀ k : ℕ, k ≠ 4 → Nat.choose n k ≤ Nat.choose n 4) : 
  (Finset.range (n + 1)).sum (λ k ↦ 
    (-1)^k * Nat.choose n k * 2^(n - k) * (1 : ℚ)) = 1120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_special_binomial_expansion_l1146_114688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_inequality_l1146_114633

noncomputable section

-- Define a structure for a triangle
structure Triangle :=
  (A B C : ℝ)
  (R S a b c : ℝ)
  (h_R : R > 0)
  (h_S : S > 0)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_c : c > 0)

theorem triangle_tangent_sum_inequality (t : Triangle) :
  Real.tan (t.A / 2) + Real.tan (t.B / 2) + Real.tan (t.C / 2) ≤ 9 * t.R^2 / (4 * t.S) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_inequality_l1146_114633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1146_114621

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 10 = 0

noncomputable def line_eq (x y b : ℝ) : Prop := y = x + b

noncomputable def dist_point_line (x y b : ℝ) : ℝ := 
  |x - y + b| / Real.sqrt 2

noncomputable def three_points_condition (b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ circle_eq x₃ y₃ ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃) ∧
    dist_point_line x₁ y₁ b = 2 * Real.sqrt 2 ∧
    dist_point_line x₂ y₂ b = 2 * Real.sqrt 2 ∧
    dist_point_line x₃ y₃ b = 2 * Real.sqrt 2

theorem circle_line_distance (b : ℝ) :
  three_points_condition b → -2 ≤ b ∧ b ≤ 2 :=
by sorry

#check circle_line_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1146_114621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1146_114620

/-- A parabola with focus F and parameter p > 0 -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_p_pos : p > 0

/-- A point on a parabola -/
structure PointOnParabola (c : Parabola) where
  point : ℝ × ℝ
  h_on_parabola : (point.2)^2 = 2 * c.p * point.1

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_equation (c : Parabola) 
    (m : PointOnParabola c) 
    (h_m_x : m.point.1 = 2) 
    (h_distance : distance m.point c.focus = 6) : 
  c.p = 8 ∧ (fun y => y^2 = 16 * m.point.1) = (fun y => y^2 = 2 * c.p * m.point.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1146_114620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_assessment_theorem_l1146_114681

/-- Represents a basketball player's shooting performance --/
structure Player where
  shot_probability : ℝ

/-- Represents the rules of the basketball assessment --/
structure AssessmentRules where
  total_rounds : ℕ
  shots_per_round : ℕ

/-- Calculates the probability of passing a single round --/
def pass_round_probability (p : Player) (rules : AssessmentRules) : ℝ :=
  1 - (1 - p.shot_probability) ^ rules.shots_per_round

/-- Calculates the expected number of passed rounds --/
def expected_passed_rounds (p : Player) (rules : AssessmentRules) : ℝ :=
  rules.total_rounds * pass_round_probability p rules

/-- The main theorem to prove --/
theorem basketball_assessment_theorem (p : Player) (rules : AssessmentRules) 
    (h1 : p.shot_probability = 0.6)
    (h2 : rules.total_rounds = 5)
    (h3 : rules.shots_per_round = 2) :
    pass_round_probability p rules = 0.84 ∧ 
    expected_passed_rounds p rules = 4.2 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_assessment_theorem_l1146_114681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_difference_l1146_114677

/-- The custom operation ⊗ defined for nonzero real numbers -/
noncomputable def otimes (a b : ℝ) : ℝ := a^2 / b

/-- Theorem stating that [(1⊗2)⊗3] - [1⊗(2⊗3)] = -2/3 -/
theorem otimes_difference :
  (otimes (otimes 1 2) 3) - (otimes 1 (otimes 2 3)) = -2/3 :=
by
  -- Expand the definition of otimes
  unfold otimes
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_difference_l1146_114677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_coin_prob_l1146_114600

/-- The probability of getting heads for an unfair coin -/
noncomputable def p_heads : ℚ := 3/4

/-- The number of coin tosses -/
def n_tosses : ℕ := 60

/-- The probability of getting an even number of heads after n tosses -/
noncomputable def P (n : ℕ) : ℚ := 1/2 * (1 + (1/2)^n)

/-- The probability of getting an odd number of tails after n tosses -/
noncomputable def Q (n : ℕ) : ℚ := 1/2 * (1 - (1/2)^n)

/-- The main theorem: probability of even heads and odd tails after 60 tosses -/
theorem unfair_coin_prob : P n_tosses * Q n_tosses = 1/4 * (1 - 1/4^60) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_coin_prob_l1146_114600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_hyperbola_l1146_114639

/-- The curve defined by the polar equation r = 3 cot θ csc θ is a hyperbola -/
theorem polar_equation_is_hyperbola :
  ∀ (r θ : ℝ), r = 3 * (Real.cos θ / Real.sin θ) * (1 / Real.sin θ) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_hyperbola_l1146_114639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l1146_114619

theorem imaginary_part_of_complex_fraction (i : ℂ) : i * i = -1 → Complex.im (2 * i / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l1146_114619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1146_114671

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  h1 : d < 0 -- Common difference is negative
  h2 : a 2 + a 6 = 10 -- Sum condition
  h3 : a 2 * a 6 = 21 -- Product condition

/-- The sequence b_n defined as 2^(a_n) -/
def b (seq : ArithmeticSequence) (n : ℕ) : ℕ := 2^(Int.toNat (seq.a n))

/-- The product of the first n terms of sequence b_n -/
def T (seq : ArithmeticSequence) (n : ℕ) : ℕ := 
  Finset.prod (Finset.range n) (fun i => b seq (i + 1))

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 9 - n) ∧ 
  (∃ m, ∀ n, T seq n ≤ 2^36 ∧ T seq m = 2^36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1146_114671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l1146_114602

/-- The arc length of a sector with given radius and central angle -/
noncomputable def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * 2 * Real.pi * radius

theorem sector_arc_length :
  let radius : ℝ := 3
  let centralAngle : ℝ := 120
  arcLength radius centralAngle = 2 * Real.pi := by
  -- Unfold the definition of arcLength
  unfold arcLength
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l1146_114602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_when_a_is_negative_four_range_of_a_for_minimum_constraint_l1146_114601

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + 2 * x

-- Theorem for part 1
theorem extremum_when_a_is_negative_four :
  ∃ (x : ℝ), x > 0 ∧ IsLocalMin (f (-4)) x ∧ f (-4) x = 4 - 4 * log 2 := by sorry

-- Theorem for part 2
theorem range_of_a_for_minimum_constraint :
  ∀ (a : ℝ), (∀ (x : ℝ), x > 0 → f a x ≥ -a) ↔ -2 ≤ a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_when_a_is_negative_four_range_of_a_for_minimum_constraint_l1146_114601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_sides_product_squared_l1146_114668

/-- Right triangle with given area and side lengths -/
structure RightTriangle where
  area : ℝ
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ

/-- Function to calculate the third side of a right triangle -/
noncomputable def thirdSide (t : RightTriangle) : ℝ :=
  if t.leg1 = t.hypotenuse then t.leg2
  else if t.leg2 = t.hypotenuse then t.leg1
  else t.hypotenuse

/-- Theorem statement -/
theorem third_sides_product_squared
  (t1 t2 : RightTriangle)
  (h1 : t1.area = 8)
  (h2 : t2.area = 18)
  (h3 : t1.leg1 = t2.hypotenuse ∨ t1.leg2 = t2.hypotenuse)
  (h4 : t1.hypotenuse = t2.leg1 ∨ t1.hypotenuse = t2.leg2)
  (h5 : Real.cos (Real.pi / 3) * t2.hypotenuse = t2.leg1 ∨ Real.cos (Real.pi / 3) * t2.hypotenuse = t2.leg2) :
  (thirdSide t1 * thirdSide t2)^2 = 576 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_sides_product_squared_l1146_114668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_and_point_check_l1146_114656

noncomputable section

def line1_start : ℝ × ℝ := (2, 1)
def line1_dir : ℝ × ℝ := (4, -1)
def line2_start : ℝ × ℝ := (3, 7)
def line2_dir : ℝ × ℝ := (-2, 5)
def point_to_check : ℝ × ℝ := (5, 0)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  (dot_product v1 v2) / (vector_magnitude v1 * vector_magnitude v2)

def point_on_line (p start dir : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p = (start.1 + t * dir.1, start.2 + t * dir.2)

theorem line_angle_and_point_check :
  cos_angle line1_dir line2_dir = -13 / Real.sqrt 493 ∧
  ¬ point_on_line point_to_check line1_start line1_dir := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_and_point_check_l1146_114656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1146_114660

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 10, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 10, 0)

def is_on_hyperbola (M : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ |M.1 - F₁.1| - |M.1 - F₂.1| = k

def vector_product (A B M : ℝ × ℝ) : ℝ :=
  (M.1 - A.1) * (M.1 - B.1) + (M.2 - A.2) * (M.2 - B.2)

noncomputable def vector_length (A M : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)

theorem hyperbola_equation (M : ℝ × ℝ) 
  (h1 : is_on_hyperbola M)
  (h2 : vector_product F₁ F₂ M = 0)
  (h3 : vector_length F₁ M * vector_length F₂ M = 2) :
  ∃ (x y : ℝ), M = (x, y) ∧ x^2/9 - y^2 = 1 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1146_114660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1146_114651

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x = 1 then a else (1/2)^(abs (x - 1)) + 1

theorem range_of_a (a : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
                           x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
                           x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
                           x₄ ≠ x₅ ∧
                           (∀ (x : ℝ), 2*(f a x)^2 - (2*a + 3)*(f a x) + 3*a = 0 ↔
                                       x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ ∨ x = x₅)) →
  (a > 1 ∧ a < 3/2) ∨ (a > 3/2 ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1146_114651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1146_114644

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) / Real.log a - Real.log (1 - x) / Real.log a

-- Theorem stating the properties of the function f
theorem f_properties (a : ℝ) (h_a_pos : a > 0) (h_a_neq_1 : a ≠ 1) :
  -- 1. The domain of f is (-1, 1)
  (∀ x, f a x ≠ 0 → -1 < x ∧ x < 1) ∧
  -- 2. f is an odd function
  (∀ x, f a (-x) = -f a x) ∧
  -- 3. When 0 < a < 1, the solution to f(x) > 0 is (-1, 0)
  (0 < a ∧ a < 1 → ∀ x, f a x > 0 ↔ -1 < x ∧ x < 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1146_114644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_roots_l1146_114655

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - x - 2

-- Theorem stating that f(x) has exactly two distinct real roots
theorem f_has_two_roots : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_roots_l1146_114655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_diagonals_of_234_parallelepiped_l1146_114689

/-- The angle between diagonals of a rectangular parallelepiped with edges 2, 3, and 4 -/
noncomputable def angle_between_diagonals (a b c : ℝ) : ℝ :=
  Real.arctan ((3 * Real.sqrt 5) / 5)

/-- Theorem: The angle between diagonals of a rectangular parallelepiped with edges 2, 3, and 4 is arctan(3√5/5) -/
theorem angle_between_diagonals_of_234_parallelepiped :
  angle_between_diagonals 2 3 4 = Real.arctan ((3 * Real.sqrt 5) / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_diagonals_of_234_parallelepiped_l1146_114689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_sum_l1146_114629

theorem percentage_sum : (30 * 0.1 + 50 * 0.15 : ℝ) = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_sum_l1146_114629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_A_under_f_l1146_114610

def A : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1}

noncomputable def f : ℝ × ℝ → ℝ × ℝ := fun p => (Real.exp (Real.log 3 * p.1), Real.exp (Real.log 3 * p.2))

theorem image_of_A_under_f :
  f '' A = {p : ℝ × ℝ | p.1 * p.2 = 3 ∧ p.1 > 0 ∧ p.2 > 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_A_under_f_l1146_114610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1146_114669

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := lg ((2 * x) / (x + 1))

-- State the theorem
theorem function_properties :
  -- Condition: f(1) = 0
  f 1 = 0 ∧
  -- Condition: For x > 0, f(x) - f(1/x) = lg x
  (∀ x : ℝ, x > 0 → f x - f (1/x) = lg x) ∧
  -- Domain of f
  (∀ x : ℝ, (x < -1 ∨ x > 0) ↔ ((2 * x) / (x + 1) > 0)) ∧
  -- Range of t
  (∀ t : ℝ, (∃ x : ℝ, f x = lg t) ↔ (0 < t ∧ t ≠ 2)) ∧
  -- Range of m
  (∀ m : ℝ, (∀ x : ℝ, f x ≠ lg (8 * x + m)) ↔ (0 ≤ m ∧ m < 18)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1146_114669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_third_derivative_at_one_g_derivative_l1146_114641

-- Part I
def f (x : ℝ) : ℝ := (2 * x^2 + 3) * (3 * x - 2)

theorem f_third_derivative_at_one : 
  (deriv^[3] f) 1 = 19 := by sorry

-- Part II
noncomputable def g (x : ℝ) : ℝ := Real.exp x / x + x * Real.sin x

theorem g_derivative (x : ℝ) (h : x ≠ 0) : 
  deriv g x = (Real.exp x * (x - 1)) / x^2 + Real.sin x + x * Real.cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_third_derivative_at_one_g_derivative_l1146_114641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1146_114637

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2*x + 2)

-- State the theorem about the range of g
theorem range_of_g :
  Set.range g = {y : ℝ | -1/2 ≤ y ∧ y ≤ 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1146_114637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_trigonometric_equation_l1146_114626

open Real Set

/-- The number of zeros of a function in [0, 2π) -/
noncomputable def numberOfZeros (f : ℝ → ℝ) : ℕ :=
  Nat.card {x ∈ Icc 0 (2 * π) | f x = 0}

theorem min_roots_trigonometric_equation
  (k₀ k₁ k₂ : ℕ) (hk₀₁ : k₀ < k₁) (hk₁₂ : k₁ < k₂) (A₁ A₂ : ℝ) :
  2 * k₀ ≤ numberOfZeros (fun x ↦ sin (k₀ * x) + A₁ * sin (k₁ * x) + A₂ * sin (k₂ * x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_trigonometric_equation_l1146_114626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_equals_one_l1146_114625

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + a)

theorem extreme_point_implies_a_equals_one (a : ℝ) :
  (∃ (f' : ℝ → ℝ), HasDerivAt f' (f a (-1)) (-1) ∧ f' (-1) = 0) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_equals_one_l1146_114625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_circles_tangents_l1146_114649

/-- Two circles with different radii, centers on x-axis, sharing one point -/
structure TwoCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius1 : ℝ
  radius2 : ℝ
  h1 : center1.2 = 0
  h2 : center2.2 = 0
  h3 : radius1 ≠ radius2
  h4 : radius1 > 0
  h5 : radius2 > 0
  h6 : ((-3 : ℝ) - center1.1) ^ 2 = radius1 ^ 2
  h7 : ((-3 : ℝ) - center2.1) ^ 2 = radius2 ^ 2

/-- The number of common tangents for two circles -/
def numCommonTangents (c : TwoCircles) : Fin 4 :=
  sorry

/-- The main theorem -/
theorem two_circles_tangents (c : TwoCircles) : 
  numCommonTangents c = 1 ∨ numCommonTangents c = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_circles_tangents_l1146_114649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_colored_segments_l1146_114616

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a segment between two adjacent points -/
structure Segment where
  start : GridPoint
  finish : GridPoint

/-- Represents the color of a point or segment -/
inductive Color
  | Red
  | Blue

/-- Represents the grid with its coloring -/
structure ColoredGrid where
  k : ℕ
  n : ℕ
  pointColor : GridPoint → Color
  segmentColor : Segment → Option Color

/-- Count the number of points of a given color in a row or column -/
def countColor (grid : ColoredGrid) (c : Color) (i : ℕ) (isRow : Bool) : ℕ :=
  sorry

/-- Conditions for a valid coloring of the grid -/
def isValidColoring (grid : ColoredGrid) : Prop :=
  (∀ row : ℕ, row < 2 * grid.n →
    (countColor grid Color.Red row true = grid.k) ∧
    (countColor grid Color.Blue row true = grid.k)) ∧
  (∀ col : ℕ, col < 2 * grid.k →
    (countColor grid Color.Red col false = grid.n) ∧
    (countColor grid Color.Blue col false = grid.n)) ∧
  (∀ seg : Segment, 
    grid.segmentColor seg = some Color.Red ↔ 
      grid.pointColor seg.start = Color.Red ∧ grid.pointColor seg.finish = Color.Red) ∧
  (∀ seg : Segment, 
    grid.segmentColor seg = some Color.Blue ↔ 
      grid.pointColor seg.start = Color.Blue ∧ grid.pointColor seg.finish = Color.Blue)

/-- Count the number of segments of a given color -/
def countColoredSegments (grid : ColoredGrid) (c : Color) : ℕ :=
  sorry

/-- The main theorem: number of red segments equals number of blue segments -/
theorem equal_colored_segments (grid : ColoredGrid) 
  (h : isValidColoring grid) : 
  countColoredSegments grid Color.Red = countColoredSegments grid Color.Blue :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_colored_segments_l1146_114616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_is_34_l1146_114696

/-- A parallelogram ABCD with given side lengths -/
structure Parallelogram where
  AB : ℝ
  BC : ℝ → ℝ
  CD : ℝ → ℝ
  AD : ℝ
  is_parallelogram : (∀ x, AB = CD x) ∧ (∀ y, BC y = AD)

/-- The product of x and y in the given parallelogram is 34 -/
theorem product_xy_is_34 (ABCD : Parallelogram)
  (h1 : ABCD.AB = 38)
  (h2 : ABCD.BC = fun y => 3 * y^3)
  (h3 : ABCD.CD = fun x => 2 * x + 4)
  (h4 : ABCD.AD = 24) :
  ∃ x y : ℝ, x * y = 34 ∧ ABCD.BC y = ABCD.AD ∧ ABCD.CD x = ABCD.AB := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_is_34_l1146_114696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1146_114608

theorem power_equality (a b : ℝ) (h1 : (60 : ℝ)^a = 3) (h2 : (60 : ℝ)^b = 5) :
  (12 : ℝ)^((1 - a - b) / (2 * (1 - b))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1146_114608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conversion_780_deg_conversion_neg_1560_deg_conversion_67_5_deg_conversion_neg_10_3_pi_conversion_pi_12_conversion_7_4_pi_l1146_114635

-- Define the conversion factor
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- Define the conversions
noncomputable def convert_780_deg_to_rad : ℝ := 780 * deg_to_rad
noncomputable def convert_neg_1560_deg_to_rad : ℝ := -1560 * deg_to_rad
noncomputable def convert_67_5_deg_to_rad : ℝ := 67.5 * deg_to_rad
noncomputable def convert_neg_10_3_pi_to_deg : ℝ := -10 / 3 * Real.pi / deg_to_rad
noncomputable def convert_pi_12_to_deg : ℝ := Real.pi / 12 / deg_to_rad
noncomputable def convert_7_4_pi_to_deg : ℝ := 7 / 4 * Real.pi / deg_to_rad

-- State the theorems
theorem conversion_780_deg : convert_780_deg_to_rad = 13 * Real.pi / 3 := by sorry

theorem conversion_neg_1560_deg : convert_neg_1560_deg_to_rad = -26 * Real.pi / 3 := by sorry

theorem conversion_67_5_deg : convert_67_5_deg_to_rad = 3 * Real.pi / 8 := by sorry

theorem conversion_neg_10_3_pi : convert_neg_10_3_pi_to_deg = -600 := by sorry

theorem conversion_pi_12 : convert_pi_12_to_deg = 15 := by sorry

theorem conversion_7_4_pi : convert_7_4_pi_to_deg = 315 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conversion_780_deg_conversion_neg_1560_deg_conversion_67_5_deg_conversion_neg_10_3_pi_conversion_pi_12_conversion_7_4_pi_l1146_114635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l1146_114606

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 4)^2 + (y - 2)^2 = 5

-- Define the distance function from a point to the origin
noncomputable def distToOrigin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem max_distance_to_origin :
  ∃ (max_dist : ℝ), 
    (∀ (x y : ℝ), circleC x y → distToOrigin x y ≤ max_dist) ∧
    (∃ (x y : ℝ), circleC x y ∧ distToOrigin x y = max_dist) ∧
    max_dist = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l1146_114606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l1146_114698

/-- An ellipse with semi-major axis 4 and semi-minor axis 2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 16) + (p.2^2 / 4) = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) :=
  {f | f ∈ Ellipse ∧ ∃ (p : ℝ × ℝ), p ∈ Ellipse → dist f p + dist ((-f.1, -f.2)) p = 8}

/-- Two points on the ellipse symmetric with respect to the origin -/
def SymmetricPoints (P Q : ℝ × ℝ) : Prop :=
  P ∈ Ellipse ∧ Q ∈ Ellipse ∧ Q = (-P.1, -P.2)

/-- The area of a quadrilateral given its four vertices -/
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  abs ((A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) -
       (B.1 * A.2 + C.1 * B.2 + D.1 * C.2 + A.1 * D.2)) / 2

/-- The theorem statement -/
theorem ellipse_quadrilateral_area
  (F₁ F₂ P Q : ℝ × ℝ)
  (h_foci : F₁ ∈ Foci ∧ F₂ ∈ Foci)
  (h_symmetric : SymmetricPoints P Q)
  (h_equal_dist : dist P Q = dist F₁ F₂) :
  area_quadrilateral P F₁ Q F₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l1146_114698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l1146_114667

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)
def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem tangent_and_inequality (k : ℝ) :
  (∃ t : ℝ, (deriv f t) * (x - t) + f t = g k x) ↔ k = 2 ∧
  (k > 0 → (∃ m : ℝ, m > 0 ∧ ∀ x ∈ Set.Ioo 0 m, |f x - g k x| > 2 * x) ↔ k > 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l1146_114667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_in_cube_l1146_114684

/-- The length of the line segment between two points in 3D space -/
noncomputable def distance3D (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- The theorem stating that the length of the line segment between (0,0,5) and (4,4,9) is 4√3 -/
theorem segment_length_in_cube : distance3D 0 0 5 4 4 9 = 4 * Real.sqrt 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_in_cube_l1146_114684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sum_range_l1146_114687

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope 1 -/
structure Line where
  b : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Theorem stating the range of K1 + K2 -/
theorem slope_sum_range 
  (l : Line) 
  (par : Parabola) 
  (A B : Point)
  (h_distinct : A ≠ B)
  (h_above_x : A.y > 0 ∧ B.y > 0)
  (h_on_line : A.y = A.x + l.b ∧ B.y = B.x + l.b)
  (h_on_parabola : A.y^2 = 2 * par.p * A.x ∧ B.y^2 = 2 * par.p * B.x)
  (K1 : ℝ) (K2 : ℝ)
  (h_K1 : K1 = A.y / A.x)
  (h_K2 : K2 = B.y / B.x) :
  ∀ k : ℝ, k > 4 → ∃ (l' : Line) (par' : Parabola) (A' B' : Point) (K1' K2' : ℝ), K1' + K2' = k :=
by
  intro k h_k_gt_4
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sum_range_l1146_114687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l1146_114659

/-- If the terminal side of angle α passes through point P(a, 2a) where a < 0, then cos α = -√5/5 -/
theorem cosine_of_angle_through_point (α : ℝ) (a : ℝ) :
  a < 0 →
  ∃ (P : ℝ × ℝ), P = (a, 2*a) ∧ (P.1 = a * Real.cos α ∧ P.2 = a * Real.sin α) →
  Real.cos α = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l1146_114659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1146_114690

open Real

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the vectors
noncomputable def e₁ (t : Triangle) : ℝ × ℝ := (2 * cos t.C, t.c / 2 - t.b)
noncomputable def e₂ (t : Triangle) : ℝ × ℝ := (t.a / 2, 1)

-- Define perpendicularity of vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h_perp : perpendicular (e₁ t) (e₂ t)) :
  (cos (2 * t.A) = -1/2) ∧ 
  (t.a = 2 → 4 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1146_114690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_l1146_114676

theorem min_value_trig_function :
  ∀ x : ℝ, Real.cos x ≠ 0 → Real.sin x ≠ 0 →
    4 / (Real.cos x)^2 + 9 / (Real.sin x)^2 ≥ 25 ∧
    ∃ y : ℝ, Real.cos y ≠ 0 ∧ Real.sin y ≠ 0 ∧ 4 / (Real.cos y)^2 + 9 / (Real.sin y)^2 = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_l1146_114676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l1146_114617

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Calculates the volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseEdge^2 * p.altitude

/-- Theorem: Volume of frustum after cutting smaller pyramid -/
theorem frustum_volume_ratio (p : SquarePyramid) 
  (h1 : p.baseEdge = 40)
  (h2 : p.altitude = 20) :
  let smallerPyramid : SquarePyramid := {
    baseEdge := p.baseEdge / 3,
    altitude := p.altitude / 3
  }
  let frustumVolume := pyramidVolume p - pyramidVolume smallerPyramid
  frustumVolume / pyramidVolume p = 87 / 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l1146_114617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_green_probability_is_four_ninths_l1146_114636

/-- Represents the number of jellybeans of each color -/
structure JellyBeanCounts where
  red : Nat
  blue : Nat
  green : Nat

/-- Calculates the probability of picking exactly two green jellybeans -/
def probability_two_green (counts : JellyBeanCounts) (pick : Nat) : ℚ :=
  let total := counts.red + counts.blue + counts.green
  let green_choices := Nat.choose counts.green 2
  let other_choices := Nat.choose (total - counts.green) (pick - 2)
  let favorable_outcomes := green_choices * other_choices
  let total_outcomes := Nat.choose total pick
  ↑favorable_outcomes / ↑total_outcomes

/-- The main theorem to be proved -/
theorem two_green_probability_is_four_ninths :
  let counts : JellyBeanCounts := { red := 5, blue := 3, green := 7 }
  let pick := 4
  probability_two_green counts pick = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_green_probability_is_four_ninths_l1146_114636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l1146_114634

noncomputable def ellipse_foci : ℝ × ℝ × ℝ × ℝ := (1, 0, -1, 0)
noncomputable def point_on_ellipse : ℝ × ℝ := (3, 4)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def ellipse_center (f1x f1y f2x f2y : ℝ) : ℝ × ℝ :=
  ((f1x + f2x) / 2, (f1y + f2y) / 2)

theorem ellipse_property (a b h k : ℝ) (ha : a > 0) (hb : b > 0) :
  let (f1x, f1y, f2x, f2y) := ellipse_foci
  let (px, py) := point_on_ellipse
  let (cx, cy) := ellipse_center f1x f1y f2x f2y
  (px - cx)^2 / a^2 + (py - cy)^2 / b^2 = 1 →
  distance f1x f1y px py + distance f2x f2y px py = 2 * a →
  a + |k| = (Real.sqrt 20 + Real.sqrt 32) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l1146_114634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1146_114666

open Set
open Real

/-- The function f(x) = 1 / (x^2 + 1) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

/-- The range of f is (0, 1] -/
theorem range_of_f : range f = Ioo 0 1 ∪ {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1146_114666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_75_degrees_is_2_30_pm_l1146_114686

/-- Represents the time on a 12-hour analog clock -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  is_pm : Bool

/-- Calculates the angle swept by the hour hand from 12 o'clock -/
noncomputable def hour_hand_angle (t : ClockTime) : ℝ :=
  30 * (t.hours % 12 : ℝ) + 0.5 * (t.minutes : ℝ)

/-- Theorem: If the hour hand moves through 75 degrees from noon, the ending time is 2:30 PM -/
theorem clock_75_degrees_is_2_30_pm :
  ∀ t : ClockTime, hour_hand_angle t = 75 → t = ⟨2, 30, true⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_75_degrees_is_2_30_pm_l1146_114686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_velocity_l1146_114613

/-- Represents the motion of a ball dropped from a height with a perfectly elastic collision. -/
structure BouncingBall where
  h : ℝ  -- Initial height
  g : ℝ  -- Acceleration due to gravity
  v₀ : ℝ := 0  -- Initial velocity (set to 0)

/-- The time when the ball first hits the ground -/
noncomputable def BouncingBall.impact_time (ball : BouncingBall) : ℝ :=
  Real.sqrt (2 * ball.h / ball.g)

/-- The velocity of the ball at the moment of impact -/
noncomputable def BouncingBall.impact_velocity (ball : BouncingBall) : ℝ :=
  ball.g * ball.impact_time

/-- The position of the ball at time t -/
noncomputable def BouncingBall.position (ball : BouncingBall) (t : ℝ) : ℝ :=
  if t ≤ ball.impact_time then
    ball.h - 0.5 * ball.g * t^2
  else
    let t' := t - ball.impact_time
    ball.impact_velocity * t' - 0.5 * ball.g * t'^2

/-- The instantaneous velocity of the ball at time t -/
noncomputable def BouncingBall.velocity (ball : BouncingBall) (t : ℝ) : ℝ :=
  if t ≤ ball.impact_time then
    -ball.g * t
  else
    ball.impact_velocity - ball.g * (t - ball.impact_time)

/-- The average velocity of the ball from initial release to time t -/
noncomputable def BouncingBall.avg_velocity (ball : BouncingBall) (t : ℝ) : ℝ :=
  (ball.h - ball.position t) / t

/-- Theorem stating that there exists a time when average velocity equals instantaneous velocity -/
theorem exists_equal_velocity (ball : BouncingBall) :
  ∃ t > 0, ball.avg_velocity t = ball.velocity t :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_velocity_l1146_114613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_2023_x_coordinate_l1146_114699

-- Define the sequence of points
def A (n : ℕ) : ℝ × ℝ := sorry
def B (n : ℕ) : ℝ × ℝ := sorry
def C (n : ℕ) : ℝ × ℝ := sorry

-- Conditions
axiom A_on_line : ∀ n : ℕ, A n = (A n).1 • (1, 1) + (0, 1)
axiom C_on_x_axis : ∀ n : ℕ, (C n).2 = 0

-- Square property
axiom square_property : ∀ n : ℕ, 
  (B (n+1)).1 - (B n).1 = 2 * ((B n).1 - (B (n-1)).1)

-- Initial conditions
axiom B_1 : B 1 = (1, 1)

-- Theorem to prove
theorem B_2023_x_coordinate : (B 2023).1 = 2^2023 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_2023_x_coordinate_l1146_114699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_m_range_l1146_114693

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2 + x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - 2*x + m

-- Statement for the tangent line equation
theorem tangent_line_equation :
  ∃ (y : ℝ → ℝ), (∀ x, y x = 3*x - 1) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |f (1 + h) - (y (1 + h))| < ε * |h|) :=
by
  sorry

-- Statement for the range of m
theorem m_range (m : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f x ≥ g m x) → m ≤ -5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_m_range_l1146_114693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_third_quadrant_l1146_114604

theorem cos_B_third_quadrant (B : ℝ) (h1 : B ∈ Set.Ioo π (3*π/2)) 
  (h2 : Real.sin B = -5/13) : Real.cos B = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_third_quadrant_l1146_114604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_proj_intersect_N_empty_M_and_N_disjoint_l1146_114614

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1}
def N : Set ℝ := {x | ∃ y, y = Real.log x}

-- Define the projection of M onto the x-axis
def M_proj : Set ℝ := {x | ∃ y, (x, y) ∈ M}

-- State the theorem
theorem M_proj_intersect_N_empty : M_proj ∩ N = ∅ := by
  sorry

-- Additional theorem to show that M and N are disjoint in the sense of the original problem
theorem M_and_N_disjoint : ∀ x y, (x, y) ∈ M → x ∉ N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_proj_intersect_N_empty_M_and_N_disjoint_l1146_114614
