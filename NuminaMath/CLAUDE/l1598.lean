import Mathlib

namespace vector_opposite_directions_x_value_l1598_159834

/-- Two vectors are in opposite directions if one is a negative scalar multiple of the other -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = (-k • b)

theorem vector_opposite_directions_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, -x)
  let b : ℝ × ℝ := (x, -6)
  opposite_directions a b → x = -Real.sqrt 6 :=
by
  sorry

end vector_opposite_directions_x_value_l1598_159834


namespace oven_capacity_correct_l1598_159880

/-- The number of pies Marcus can fit in his oven at once. -/
def oven_capacity : ℕ := 5

/-- The number of batches Marcus bakes. -/
def batches : ℕ := 7

/-- The number of pies Marcus drops. -/
def dropped_pies : ℕ := 8

/-- The number of pies left after dropping. -/
def remaining_pies : ℕ := 27

/-- Theorem stating that the oven capacity is correct given the conditions. -/
theorem oven_capacity_correct : 
  batches * oven_capacity - dropped_pies = remaining_pies :=
by sorry

end oven_capacity_correct_l1598_159880


namespace elective_subjects_theorem_l1598_159894

def subjects := 6
def chosen := 3

def choose (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

theorem elective_subjects_theorem :
  -- Statement A
  (choose 5 3 = choose 5 2) ∧
  -- Statement C
  (choose subjects chosen - choose 4 1 = choose subjects chosen - choose (subjects - 2) (chosen - 2)) :=
sorry

end elective_subjects_theorem_l1598_159894


namespace smallest_integer_in_special_average_l1598_159815

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem smallest_integer_in_special_average (m n : ℕ) 
  (h1 : is_two_digit m) 
  (h2 : is_three_digit n) 
  (h3 : (m + n) / 2 = m + n / 1000) : 
  min m n = 1 :=
sorry

end smallest_integer_in_special_average_l1598_159815


namespace square_difference_l1598_159824

theorem square_difference (a b : ℝ) 
  (h1 : 3 * a + 3 * b = 18) 
  (h2 : a - b = 4) : 
  a^2 - b^2 = 24 := by
sorry

end square_difference_l1598_159824


namespace toby_camera_roll_photos_l1598_159899

/-- The number of photos on Toby's camera roll initially -/
def initial_photos : ℕ := 79

/-- The number of photos Toby deleted initially -/
def deleted_initially : ℕ := 7

/-- The number of photos Toby added of his cat -/
def added_photos : ℕ := 15

/-- The number of photos Toby deleted after editing -/
def deleted_after_editing : ℕ := 3

/-- The final number of photos on Toby's camera roll -/
def final_photos : ℕ := 84

theorem toby_camera_roll_photos :
  initial_photos - deleted_initially + added_photos - deleted_after_editing = final_photos :=
by sorry

end toby_camera_roll_photos_l1598_159899


namespace max_value_abcd_l1598_159822

def S : Finset ℕ := {1, 3, 5, 7}

theorem max_value_abcd (a b c d : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) 
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  (∀ (w x y z : ℕ), w ∈ S → x ∈ S → y ∈ S → z ∈ S → 
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → 
    w * x + x * y + y * z + z * w ≤ a * b + b * c + c * d + d * a) →
  a * b + b * c + c * d + d * a = 64 :=
sorry

end max_value_abcd_l1598_159822


namespace queen_heart_jack_probability_l1598_159887

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of queens in a standard deck -/
def QueenCount : ℕ := 4

/-- Number of hearts in a standard deck -/
def HeartCount : ℕ := 13

/-- Number of jacks in a standard deck -/
def JackCount : ℕ := 4

/-- Probability of drawing a queen as the first card, a heart as the second card, 
    and a jack as the third card from a standard 52-card deck -/
def probabilityQueenHeartJack : ℚ := 1 / 663

theorem queen_heart_jack_probability :
  probabilityQueenHeartJack = 
    (QueenCount / StandardDeck) * 
    (HeartCount / (StandardDeck - 1)) * 
    (JackCount / (StandardDeck - 2)) := by
  sorry

end queen_heart_jack_probability_l1598_159887


namespace trig_identity_l1598_159857

theorem trig_identity (α : ℝ) :
  (Real.sin (6 * α) + Real.sin (7 * α) + Real.sin (8 * α) + Real.sin (9 * α)) /
  (Real.cos (6 * α) + Real.cos (7 * α) + Real.cos (8 * α) + Real.cos (9 * α)) =
  Real.tan (15 * α / 2) := by
sorry

end trig_identity_l1598_159857


namespace prob_A_not_in_A_is_two_thirds_l1598_159870

-- Define the number of volunteers and communities
def num_volunteers : ℕ := 4
def num_communities : ℕ := 3

-- Define a type for volunteers and communities
inductive Volunteer : Type
| A | B | C | D

inductive Community : Type
| A | B | C

-- Define an assignment as a function from Volunteer to Community
def Assignment := Volunteer → Community

-- Define a valid assignment
def valid_assignment (a : Assignment) : Prop :=
  ∀ c : Community, ∃ v : Volunteer, a v = c

-- Define the probability that volunteer A is not in community A
def prob_A_not_in_A (total_assignments : ℕ) (valid_assignments : ℕ) : ℚ :=
  (valid_assignments - (total_assignments / num_communities)) / valid_assignments

-- State the theorem
theorem prob_A_not_in_A_is_two_thirds :
  ∃ (total_assignments valid_assignments : ℕ),
    total_assignments > 0 ∧
    valid_assignments > 0 ∧
    valid_assignments ≤ total_assignments ∧
    prob_A_not_in_A total_assignments valid_assignments = 2/3 :=
sorry

end prob_A_not_in_A_is_two_thirds_l1598_159870


namespace consecutive_naturals_integer_quotient_l1598_159829

theorem consecutive_naturals_integer_quotient :
  ∃! (n : ℕ), (n + 1 : ℚ) / n = ⌊(n + 1 : ℚ) / n⌋ ∧ n = 1 := by
  sorry

end consecutive_naturals_integer_quotient_l1598_159829


namespace motion_rate_of_change_l1598_159823

-- Define the law of motion
def s (t : ℝ) : ℝ := 2 * t^2 + 1

-- Define the rate of change function
def rate_of_change (d : ℝ) : ℝ := 4 + 2 * d

-- Theorem statement
theorem motion_rate_of_change (d : ℝ) :
  let t₁ := 1
  let t₂ := 1 + d
  (s t₂ - s t₁) / (t₂ - t₁) = rate_of_change d :=
by sorry

end motion_rate_of_change_l1598_159823


namespace primitive_root_modulo_power_of_prime_l1598_159897

theorem primitive_root_modulo_power_of_prime
  (p : Nat) (x α : Nat)
  (h_prime : Nat.Prime p)
  (h_alpha : α ≥ 2)
  (h_primitive_root : IsPrimitiveRoot x p)
  (h_not_congruent : ¬ (x^(p^(α-2)*(p-1)) ≡ 1 [MOD p^α])) :
  IsPrimitiveRoot x (p^α) :=
sorry

end primitive_root_modulo_power_of_prime_l1598_159897


namespace solution_set_when_a_is_one_range_of_a_for_all_real_solution_l1598_159835

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 1| + |a*x - 3*a|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | |x - 1| + |x - 3| ≥ 5} = {x : ℝ | x ≥ 9/2 ∨ x ≤ -1/2} := by sorry

-- Part 2
theorem range_of_a_for_all_real_solution :
  {a : ℝ | a > 0 ∧ ∀ x, f a x ≥ 5} = {a : ℝ | a ≥ 2} := by sorry

end solution_set_when_a_is_one_range_of_a_for_all_real_solution_l1598_159835


namespace marble_probability_difference_l1598_159853

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1500

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1500

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- Theorem stating the absolute difference between P_s and P_d -/
theorem marble_probability_difference : |P_s - P_d| = 15 / 44985 := by sorry

end marble_probability_difference_l1598_159853


namespace distance_A_to_C_l1598_159825

/-- Given four collinear points A, B, C, and D in that order, with specific distance relationships,
    prove that the distance from A to C is 15. -/
theorem distance_A_to_C (A B C D : ℝ) : 
  A < B ∧ B < C ∧ C < D →  -- Points are on a line in order
  D - A = 24 →             -- Distance from A to D is 24
  D - B = 3 * (B - A) →    -- Distance from B to D is 3 times the distance from A to B
  C - B = (D - B) / 2 →    -- C is halfway between B and D
  C - A = 15 := by         -- Distance from A to C is 15
sorry

end distance_A_to_C_l1598_159825


namespace symmetric_second_quadrant_condition_l1598_159828

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry about the origin -/
def symmetricAboutOrigin (p : Point2D) : Prop :=
  ∃ q : Point2D, q.x = -p.x ∧ q.y = -p.y

/-- Definition of a point being in the second quadrant -/
def inSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating the condition for m -/
theorem symmetric_second_quadrant_condition (m : ℝ) :
  symmetricAboutOrigin ⟨-m, m-3⟩ ∧ inSecondQuadrant ⟨m, 3-m⟩ → m < 0 :=
sorry

end symmetric_second_quadrant_condition_l1598_159828


namespace solution_eq_200_div_253_l1598_159809

/-- A binary operation on nonzero real numbers satisfying certain properties -/
def diamond (a b : ℝ) : ℝ := sorry

/-- The binary operation satisfies a ◇ (b ◇ c) = (a ◇ b) · c -/
axiom diamond_assoc (a b c : ℝ) : a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = (diamond a b) * c

/-- The binary operation satisfies a ◇ a = 1 -/
axiom diamond_self (a : ℝ) : a ≠ 0 → diamond a a = 1

/-- The solution to the equation 2024 ◇ (8 ◇ x) = 200 is 200/253 -/
theorem solution_eq_200_div_253 : ∃ (x : ℝ), x ≠ 0 ∧ diamond 2024 (diamond 8 x) = 200 ∧ x = 200/253 := by sorry

end solution_eq_200_div_253_l1598_159809


namespace exam_result_proof_l1598_159893

/-- Represents the result of an examination --/
structure ExamResult where
  total_questions : ℕ
  correct_score : ℤ
  wrong_score : ℤ
  unanswered_score : ℤ
  total_score : ℤ
  correct_answers : ℕ
  wrong_answers : ℕ
  unanswered : ℕ

/-- Theorem stating the correct number of answers for the given exam conditions --/
theorem exam_result_proof (exam : ExamResult) : 
  exam.total_questions = 75 ∧ 
  exam.correct_score = 5 ∧ 
  exam.wrong_score = -2 ∧ 
  exam.unanswered_score = -1 ∧ 
  exam.total_score = 215 ∧
  exam.correct_answers + exam.wrong_answers + exam.unanswered = exam.total_questions ∧
  exam.correct_score * exam.correct_answers + exam.wrong_score * exam.wrong_answers + exam.unanswered_score * exam.unanswered = exam.total_score →
  exam.correct_answers = 52 ∧ exam.wrong_answers = 23 ∧ exam.unanswered = 0 := by
  sorry

end exam_result_proof_l1598_159893


namespace shaded_area_of_divided_triangle_l1598_159860

theorem shaded_area_of_divided_triangle (leg_length : ℝ) (total_divisions : ℕ) (shaded_divisions : ℕ) : 
  leg_length = 10 → 
  total_divisions = 20 → 
  shaded_divisions = 12 → 
  (1/2 * leg_length * leg_length * (shaded_divisions / total_divisions : ℝ)) = 30 := by
  sorry

end shaded_area_of_divided_triangle_l1598_159860


namespace plant_branches_problem_l1598_159866

theorem plant_branches_problem :
  ∃ (x : ℕ),
    (1 : ℕ) + x + x * x = 91 ∧
    (∀ y : ℕ, (1 : ℕ) + y + y * y = 91 → y ≤ x) ∧
    x = 9 :=
by sorry

end plant_branches_problem_l1598_159866


namespace machine_purchase_price_l1598_159862

/-- Proves that given the specified conditions, the original purchase price of the machine was Rs 9000 -/
theorem machine_purchase_price 
  (repair_cost : ℕ) 
  (transport_cost : ℕ) 
  (profit_percentage : ℚ) 
  (selling_price : ℕ) 
  (h1 : repair_cost = 5000)
  (h2 : transport_cost = 1000)
  (h3 : profit_percentage = 50 / 100)
  (h4 : selling_price = 22500) :
  ∃ (purchase_price : ℕ), 
    selling_price = (1 + profit_percentage) * (purchase_price + repair_cost + transport_cost) ∧
    purchase_price = 9000 := by
  sorry


end machine_purchase_price_l1598_159862


namespace day_of_week_in_consecutive_years_l1598_159811

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℕ
  is_leap : Bool

/-- Returns the day of the week for a given day number in a year -/
def day_of_week (y : Year) (day_number : ℕ) : DayOfWeek :=
  sorry

/-- Returns the next year -/
def next_year (y : Year) : Year :=
  sorry

/-- Returns the previous year -/
def prev_year (y : Year) : Year :=
  sorry

theorem day_of_week_in_consecutive_years 
  (y : Year)
  (h1 : day_of_week y 250 = DayOfWeek.Friday)
  (h2 : day_of_week (next_year y) 150 = DayOfWeek.Friday) :
  day_of_week (prev_year y) 50 = DayOfWeek.Thursday :=
sorry

end day_of_week_in_consecutive_years_l1598_159811


namespace inequality_proof_l1598_159800

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end inequality_proof_l1598_159800


namespace quadratic_coefficient_l1598_159874

theorem quadratic_coefficient (b : ℝ) (m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/4 = (x + m)^2 + 1/8) → 
  b = -Real.sqrt 2 / 2 := by
sorry

end quadratic_coefficient_l1598_159874


namespace absolute_value_of_negative_not_negative_l1598_159817

theorem absolute_value_of_negative_not_negative (x : ℝ) (h : x < 0) : |x| ≠ x := by
  sorry

end absolute_value_of_negative_not_negative_l1598_159817


namespace cos_theta_for_point_l1598_159878

/-- If the terminal side of angle θ passes through point P(-12, 5), then cos θ = -12/13 -/
theorem cos_theta_for_point (θ : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = -12 ∧ r * Real.sin θ = 5) → 
  Real.cos θ = -12/13 := by
sorry

end cos_theta_for_point_l1598_159878


namespace solve_system_l1598_159850

theorem solve_system (x y : ℚ) (h1 : 3 * x - y = 9) (h2 : x + 4 * y = 11) : x = 47 / 13 := by
  sorry

end solve_system_l1598_159850


namespace equation_solution_l1598_159807

theorem equation_solution : 
  ∀ x : ℝ, (x + 1) * (x + 3) = x + 1 ↔ x = -1 ∨ x = -2 := by
sorry

end equation_solution_l1598_159807


namespace jason_career_percentage_increase_l1598_159803

/-- Represents the career progression of a military person --/
structure MilitaryCareer where
  join_age : ℕ
  years_to_chief : ℕ
  retirement_age : ℕ
  years_after_master_chief : ℕ

/-- Calculates the percentage increase in time from chief to master chief
    compared to the time to become a chief --/
def percentage_increase (career : MilitaryCareer) : ℚ :=
  let total_years := career.retirement_age - career.join_age
  let years_chief_to_retirement := total_years - career.years_to_chief
  let years_to_master_chief := years_chief_to_retirement - career.years_after_master_chief
  (years_to_master_chief - career.years_to_chief) / career.years_to_chief * 100

/-- Theorem stating that for Jason's career, the percentage increase is 25% --/
theorem jason_career_percentage_increase :
  let jason_career := MilitaryCareer.mk 18 8 46 10
  percentage_increase jason_career = 25 := by
  sorry

end jason_career_percentage_increase_l1598_159803


namespace parabola_focus_to_line_distance_l1598_159845

/-- The distance from the focus of the parabola y² = 2x to the line x - √3y = 0 is 1/4 -/
theorem parabola_focus_to_line_distance : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*x}
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 3 * y = 0}
  let focus : ℝ × ℝ := (1/2, 0)
  ∃ d : ℝ, d = 1/4 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2) ≥ d :=
by sorry

end parabola_focus_to_line_distance_l1598_159845


namespace smallest_solution_floor_equation_l1598_159892

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 194 ∧
    (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 25) ∧
    ⌊x^2⌋ - ⌊x⌋^2 = 25 :=
by sorry

end smallest_solution_floor_equation_l1598_159892


namespace boxes_with_neither_markers_nor_crayons_l1598_159861

/-- The number of boxes containing neither markers nor crayons -/
def empty_boxes (total boxes_with_markers boxes_with_crayons boxes_with_both : ℕ) : ℕ :=
  total - (boxes_with_markers + boxes_with_crayons - boxes_with_both)

/-- Theorem: Given the conditions of the problem, there are 5 boxes with neither markers nor crayons -/
theorem boxes_with_neither_markers_nor_crayons :
  empty_boxes 15 9 5 4 = 5 := by
  sorry

end boxes_with_neither_markers_nor_crayons_l1598_159861


namespace inverse_variation_problem_l1598_159802

-- Define the constant k
def k : ℝ := 4^2 * (256 ^ (1/4))

-- State the theorem
theorem inverse_variation_problem (x y : ℝ) 
  (h1 : x^2 * y^(1/4) = k)  -- x² and ⁴√y are inversely proportional
  (h2 : x * y = 128)        -- xy = 128
  : y = 8 := by
  sorry

-- Note: The condition x = 4 when y = 256 is implicitly used in the definition of k

end inverse_variation_problem_l1598_159802


namespace frisbee_price_problem_l1598_159890

theorem frisbee_price_problem (total_frisbees : ℕ) (total_receipts : ℕ) (price_a : ℕ) (min_frisbees_b : ℕ) :
  total_frisbees = 60 →
  price_a = 3 →
  total_receipts = 204 →
  min_frisbees_b = 24 →
  ∃ (frisbees_a frisbees_b price_b : ℕ),
    frisbees_a + frisbees_b = total_frisbees ∧
    frisbees_b ≥ min_frisbees_b ∧
    price_a * frisbees_a + price_b * frisbees_b = total_receipts ∧
    price_b = 4 := by
  sorry

#check frisbee_price_problem

end frisbee_price_problem_l1598_159890


namespace simplify_and_evaluate_l1598_159858

theorem simplify_and_evaluate (m : ℝ) (h : m = -2) :
  m / (m^2 - 9) / (1 + 3 / (m - 3)) = 1 := by
  sorry

end simplify_and_evaluate_l1598_159858


namespace inequalities_with_distinct_positive_reals_l1598_159831

theorem inequalities_with_distinct_positive_reals 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a^4 + b^4 > a^3*b + a*b^3) ∧ (a^5 + b^5 > a^3*b^2 + a^2*b^3) := by
  sorry

end inequalities_with_distinct_positive_reals_l1598_159831


namespace complementary_angles_difference_l1598_159821

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- angles are complementary
  a = 5 * b → -- ratio of angles is 5:1
  |a - b| = 60 := by sorry

end complementary_angles_difference_l1598_159821


namespace distance_difference_after_six_hours_l1598_159810

/-- Represents a cyclist with a given travel rate in miles per hour. -/
structure Cyclist where
  name : String
  rate : ℝ

/-- Calculates the distance traveled by a cyclist in a given time. -/
def distance_traveled (c : Cyclist) (time : ℝ) : ℝ :=
  c.rate * time

/-- The time period in hours for which we calculate the travel distance. -/
def travel_time : ℝ := 6

/-- Carmen, a cyclist with a travel rate of 15 miles per hour. -/
def carmen : Cyclist :=
  { name := "Carmen", rate := 15 }

/-- Daniel, a cyclist with a travel rate of 12.5 miles per hour. -/
def daniel : Cyclist :=
  { name := "Daniel", rate := 12.5 }

/-- Theorem stating the difference in distance traveled between Carmen and Daniel after 6 hours. -/
theorem distance_difference_after_six_hours :
    distance_traveled carmen travel_time - distance_traveled daniel travel_time = 15 := by
  sorry

end distance_difference_after_six_hours_l1598_159810


namespace polygon_sides_count_l1598_159832

-- Define a convex polygon with n sides
def ConvexPolygon (n : ℕ) := n ≥ 3

-- Define the sum of interior angles of a polygon
def SumOfInteriorAngles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the theorem
theorem polygon_sides_count 
  (n : ℕ) 
  (h_convex : ConvexPolygon n) 
  (h_sum : SumOfInteriorAngles n - (2 * (SumOfInteriorAngles n / (n - 1)) - 20) = 2790) :
  n = 18 := by sorry

end polygon_sides_count_l1598_159832


namespace forty_third_digit_of_one_thirteenth_l1598_159846

/-- The decimal representation of 1/13 as a sequence of digits after the decimal point -/
def decimalRep : ℕ → Fin 10
  | n => sorry

/-- The length of the repeating sequence in the decimal representation of 1/13 -/
def repeatLength : ℕ := 6

/-- The 43rd digit after the decimal point in the decimal representation of 1/13 is 0 -/
theorem forty_third_digit_of_one_thirteenth : decimalRep 42 = 0 := by sorry

end forty_third_digit_of_one_thirteenth_l1598_159846


namespace problem_solution_l1598_159826

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 3)
  (h3 : z^2 / x = 4) :
  x = 576^(1/7) := by
sorry

end problem_solution_l1598_159826


namespace fractional_equation_solution_l1598_159827

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 3 → ((x - 2) / (x - 3) = 2 / (x - 3) ↔ x = 4) :=
by
  sorry

end fractional_equation_solution_l1598_159827


namespace perfect_square_trinomial_l1598_159852

theorem perfect_square_trinomial (k : ℝ) : 
  (∀ x, ∃ a b : ℝ, x^2 - (k-1)*x + 25 = (a*x + b)^2) ↔ (k = 11 ∨ k = -9) :=
sorry

end perfect_square_trinomial_l1598_159852


namespace negation_of_positive_product_l1598_159818

theorem negation_of_positive_product (x y : ℝ) :
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ (x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0) :=
by sorry

end negation_of_positive_product_l1598_159818


namespace toy_blocks_difference_l1598_159881

theorem toy_blocks_difference (red_blocks yellow_blocks blue_blocks : ℕ) : 
  red_blocks = 18 →
  yellow_blocks = red_blocks + 7 →
  red_blocks + yellow_blocks + blue_blocks = 75 →
  blue_blocks - red_blocks = 14 :=
by
  sorry

end toy_blocks_difference_l1598_159881


namespace sqrt_sum_problem_l1598_159859

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) * Real.sqrt (36 - y^2) = 12) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7.8 := by
  sorry

end sqrt_sum_problem_l1598_159859


namespace farmer_sold_two_ducks_l1598_159888

/-- Represents the farmer's market scenario -/
structure FarmerMarket where
  duck_price : ℕ
  chicken_price : ℕ
  chickens_sold : ℕ
  wheelbarrow_profit : ℕ

/-- Calculates the number of ducks sold given the market conditions -/
def ducks_sold (market : FarmerMarket) : ℕ :=
  let total_earnings := 2 * market.wheelbarrow_profit
  let chicken_earnings := market.chicken_price * market.chickens_sold
  (total_earnings - chicken_earnings) / market.duck_price

/-- Theorem stating that the number of ducks sold is 2 -/
theorem farmer_sold_two_ducks : 
  ∀ (market : FarmerMarket), 
  market.duck_price = 10 ∧ 
  market.chicken_price = 8 ∧ 
  market.chickens_sold = 5 ∧ 
  market.wheelbarrow_profit = 60 →
  ducks_sold market = 2 := by
  sorry


end farmer_sold_two_ducks_l1598_159888


namespace largest_prime_divisor_l1598_159854

/-- Converts a base-5 number (represented as a list of digits) to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- The base-5 representation of the number in question --/
def base5Number : List Nat := [1, 2, 0, 1, 0, 2, 0, 1]

/-- The decimal representation of the number --/
def decimalNumber : Nat := base5ToDecimal base5Number

/-- Proposition: The largest prime divisor of the given number is 139 --/
theorem largest_prime_divisor :
  ∃ (d : Nat), d.Prime ∧ d ∣ decimalNumber ∧ d = 139 ∧ ∀ (p : Nat), p.Prime → p ∣ decimalNumber → p ≤ d :=
sorry

end largest_prime_divisor_l1598_159854


namespace sum_of_integers_between_2_and_15_l1598_159814

theorem sum_of_integers_between_2_and_15 : 
  (Finset.range 12).sum (fun i => i + 3) = 102 := by
  sorry

end sum_of_integers_between_2_and_15_l1598_159814


namespace x_over_y_value_l1598_159843

theorem x_over_y_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 5 * y) / (y + 5 * x) = 2) : 
  x / y = 0.6 := by
sorry

end x_over_y_value_l1598_159843


namespace jake_and_kendra_weight_l1598_159830

/-- Jake's current weight in pounds -/
def jake_weight : ℕ := 198

/-- The weight Jake would lose in pounds -/
def weight_loss : ℕ := 8

/-- Kendra's weight in pounds -/
def kendra_weight : ℕ := (jake_weight - weight_loss) / 2

/-- The combined weight of Jake and Kendra in pounds -/
def combined_weight : ℕ := jake_weight + kendra_weight

/-- Theorem stating the combined weight of Jake and Kendra -/
theorem jake_and_kendra_weight : combined_weight = 293 := by
  sorry

end jake_and_kendra_weight_l1598_159830


namespace wire_cutting_l1598_159844

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (shorter_part : ℝ) : 
  total_length = 180 ∧ difference = 32 → 
  shorter_part + (shorter_part + difference) = total_length →
  shorter_part = 74 := by sorry

end wire_cutting_l1598_159844


namespace cos_pi_eighth_times_cos_five_pi_eighth_l1598_159867

theorem cos_pi_eighth_times_cos_five_pi_eighth :
  Real.cos (π / 8) * Real.cos (5 * π / 8) = -Real.sqrt 2 / 4 := by
  sorry

end cos_pi_eighth_times_cos_five_pi_eighth_l1598_159867


namespace derivative_at_one_l1598_159871

-- Define the function
def f (x : ℝ) : ℝ := (2*x + 1)^2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 12 := by sorry

end derivative_at_one_l1598_159871


namespace mariam_neighborhood_houses_l1598_159812

/-- The number of houses on one side of the main road -/
def houses_on_first_side : ℕ := 40

/-- The function representing the number of houses on the other side of the road -/
def f (x : ℕ) : ℕ := x^2 + 3*x

/-- The total number of houses in Mariam's neighborhood -/
def total_houses : ℕ := houses_on_first_side + f houses_on_first_side

theorem mariam_neighborhood_houses :
  total_houses = 1760 := by sorry

end mariam_neighborhood_houses_l1598_159812


namespace hallies_reading_l1598_159895

/-- Proves that given the conditions of Hallie's reading pattern, she read 63 pages on the first day -/
theorem hallies_reading (total_pages : ℕ) (day1 : ℕ) : 
  total_pages = 354 → 
  day1 + 2 * day1 + (2 * day1 + 10) + 29 = total_pages → 
  day1 = 63 := by
  sorry

#check hallies_reading

end hallies_reading_l1598_159895


namespace hotel_fee_proof_l1598_159886

/-- The flat fee for the first night in a hotel -/
def flat_fee : ℝ := 87.5

/-- The nightly fee for each subsequent night -/
def nightly_fee : ℝ := 52.5

/-- Alice's total cost for a 4-night stay -/
def alice_cost : ℝ := 245

/-- Bob's total cost for a 6-night stay -/
def bob_cost : ℝ := 350

/-- The number of nights in Alice's stay -/
def alice_nights : ℕ := 4

/-- The number of nights in Bob's stay -/
def bob_nights : ℕ := 6

theorem hotel_fee_proof :
  (flat_fee + (alice_nights - 1 : ℝ) * nightly_fee = alice_cost) ∧
  (flat_fee + (bob_nights - 1 : ℝ) * nightly_fee = bob_cost) :=
by sorry

#check hotel_fee_proof

end hotel_fee_proof_l1598_159886


namespace point_on_angle_terminal_side_l1598_159848

theorem point_on_angle_terminal_side (y : ℝ) :
  let P : ℝ × ℝ := (-1, y)
  let θ : ℝ := 2 * Real.pi / 3
  (P.1 = -1) →   -- x-coordinate is -1
  (Real.tan θ = y / P.1) →  -- point is on terminal side of angle θ
  y = Real.sqrt 3 := by
sorry

end point_on_angle_terminal_side_l1598_159848


namespace language_interview_probability_l1598_159875

theorem language_interview_probability 
  (total_students : ℕ) 
  (french_students : ℕ) 
  (spanish_students : ℕ) 
  (both_languages : ℕ) 
  (h1 : total_students = 28)
  (h2 : french_students = 20)
  (h3 : spanish_students = 23)
  (h4 : both_languages = 17)
  (h5 : both_languages ≤ french_students)
  (h6 : both_languages ≤ spanish_students)
  (h7 : french_students ≤ total_students)
  (h8 : spanish_students ≤ total_students) :
  (1 : ℚ) - (Nat.choose (french_students - both_languages + (spanish_students - both_languages)) 2 : ℚ) / (Nat.choose total_students 2) = 20 / 21 :=
sorry

end language_interview_probability_l1598_159875


namespace unique_number_count_l1598_159855

/-- The number of unique 5-digit numbers that can be formed by rearranging
    the digits 3, 7, 3, 2, 2, 0, where the number doesn't start with 0. -/
def unique_numbers : ℕ := 24

/-- The set of digits available for forming the numbers. -/
def digits : Finset ℕ := {3, 7, 2, 0}

/-- The total number of digits to be used. -/
def total_digits : ℕ := 5

/-- The number of positions where 0 can be placed (not in the first position). -/
def zero_positions : ℕ := 4

/-- The number of times 3 appears in the original number. -/
def count_three : ℕ := 2

/-- The number of times 2 appears in the original number. -/
def count_two : ℕ := 2

theorem unique_number_count :
  unique_numbers = (zero_positions * Nat.factorial (total_digits - 1)) /
                   (Nat.factorial count_three * Nat.factorial count_two) :=
sorry

end unique_number_count_l1598_159855


namespace probability_not_greater_than_four_l1598_159856

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem probability_not_greater_than_four 
  (a₁ : ℝ) 
  (d : ℝ) 
  (n : ℕ) 
  (h₁ : a₁ = 12) 
  (h₂ : d = -2) 
  (h₃ : n = 16) : 
  (Finset.filter (fun i => arithmetic_sequence a₁ d i ≤ 4) (Finset.range n)).card / n = 3/4 := by
sorry

end probability_not_greater_than_four_l1598_159856


namespace income_scientific_notation_l1598_159849

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (figures : ℕ) : ScientificNotation :=
  sorry

theorem income_scientific_notation :
  let income : ℝ := 31.534 * 1000000000
  let scientific_form := toScientificNotation income
  let rounded_form := roundToSignificantFigures scientific_form 2
  rounded_form = ScientificNotation.mk 3.2 10 sorry :=
sorry

end income_scientific_notation_l1598_159849


namespace students_in_both_clubs_l1598_159833

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (in_either_club : ℕ)
  (h1 : total_students = 300)
  (h2 : drama_club = 100)
  (h3 : science_club = 140)
  (h4 : in_either_club = 220) :
  drama_club + science_club - in_either_club = 20 :=
by sorry

end students_in_both_clubs_l1598_159833


namespace union_of_A_and_B_l1598_159896

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -2} := by sorry

end union_of_A_and_B_l1598_159896


namespace emily_quiz_score_l1598_159837

def emily_scores : List ℝ := [96, 88, 90, 85, 94]

theorem emily_quiz_score (target_mean : ℝ) (sixth_score : ℝ) :
  target_mean = 92 ∧ sixth_score = 99 →
  (emily_scores.sum + sixth_score) / 6 = target_mean := by
  sorry

end emily_quiz_score_l1598_159837


namespace jenny_basket_eggs_l1598_159872

def is_valid_basket_size (n : ℕ) : Prop :=
  n ≥ 5 ∧ 30 % n = 0 ∧ 42 % n = 0

theorem jenny_basket_eggs : ∃! n : ℕ, is_valid_basket_size n ∧ ∀ m : ℕ, is_valid_basket_size m → m ≤ n :=
by sorry

end jenny_basket_eggs_l1598_159872


namespace student_walking_distance_l1598_159840

theorem student_walking_distance 
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (bus_speed_with_students : ℝ)
  (empty_bus_speed : ℝ)
  (h1 : total_distance = 1)
  (h2 : walking_speed = 4)
  (h3 : bus_speed_with_students = 40)
  (h4 : empty_bus_speed = 60)
  (h5 : ∀ x : ℝ, 0 < x ∧ x < 1 → 
    x / walking_speed = 
    (1 - x) / bus_speed_with_students + 
    (1 - 2*x) / empty_bus_speed) :
  ∃ x : ℝ, x = 5 / 37 ∧ 
    x / walking_speed = 
    (1 - x) / bus_speed_with_students + 
    (1 - 2*x) / empty_bus_speed :=
sorry

end student_walking_distance_l1598_159840


namespace second_diff_is_arithmetic_sequence_l1598_159876

-- Define the cube function
def cube (n : ℕ) : ℕ := n^3

-- Define the first difference of cubes
def first_diff (n : ℕ) : ℕ := cube (n + 1) - cube n

-- Define the second difference of cubes
def second_diff (n : ℕ) : ℕ := first_diff (n + 1) - first_diff n

-- Theorem stating that the second difference is 6n + 6
theorem second_diff_is_arithmetic_sequence (n : ℕ) : second_diff n = 6 * n + 6 := by
  sorry

end second_diff_is_arithmetic_sequence_l1598_159876


namespace squirrel_journey_time_l1598_159877

theorem squirrel_journey_time : 
  let first_leg_distance : ℝ := 2
  let first_leg_speed : ℝ := 5
  let second_leg_distance : ℝ := 3
  let second_leg_speed : ℝ := 3
  let first_leg_time : ℝ := first_leg_distance / first_leg_speed
  let second_leg_time : ℝ := second_leg_distance / second_leg_speed
  let total_time_hours : ℝ := first_leg_time + second_leg_time
  let total_time_minutes : ℝ := total_time_hours * 60
  total_time_minutes = 84 := by
sorry


end squirrel_journey_time_l1598_159877


namespace quadratic_inequality_condition_l1598_159879

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 4*x ≥ 0 ↔ x ≥ 4 ∨ x ≤ 0 := by
  sorry

end quadratic_inequality_condition_l1598_159879


namespace archimedes_schools_l1598_159868

/-- The number of students in Euclid's contest -/
def euclid_participants : ℕ := 69

/-- The number of students per school team -/
def team_size : ℕ := 4

/-- The total number of participants in Archimedes' contest -/
def total_participants : ℕ := euclid_participants + 100

/-- Beth's rank in the contest -/
def beth_rank : ℕ := 45

/-- Carla's rank in the contest -/
def carla_rank : ℕ := 80

/-- Andrea's teammates with lower scores -/
def andreas_lower_teammates : ℕ := 2

theorem archimedes_schools :
  ∃ (num_schools : ℕ), 
    num_schools * team_size = total_participants ∧
    num_schools = 43 :=
sorry

end archimedes_schools_l1598_159868


namespace cube_minus_cylinder_volume_l1598_159884

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h1 : cube_side = 6)
  (h2 : cylinder_radius = 3)
  (h3 : cylinder_height = cube_side) :
  cube_side^3 - π * cylinder_radius^2 * cylinder_height = 216 - 54*π :=
by sorry

end cube_minus_cylinder_volume_l1598_159884


namespace max_det_bound_l1598_159883

theorem max_det_bound (M : Matrix (Fin 17) (Fin 17) ℤ) 
  (h : ∀ i j, M i j = 1 ∨ M i j = -1) :
  |M.det| ≤ 327680 * 2^16 := by
  sorry

end max_det_bound_l1598_159883


namespace kevins_cards_l1598_159841

theorem kevins_cards (x y : ℕ) : x + y = 8 * x → y = 7 * x := by
  sorry

end kevins_cards_l1598_159841


namespace division_remainder_proof_l1598_159865

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 140 → divisor = 15 → quotient = 9 → 
  dividend = divisor * quotient + remainder → remainder = 5 := by
sorry

end division_remainder_proof_l1598_159865


namespace tank_capacity_l1598_159891

theorem tank_capacity (x : ℝ) 
  (h1 : x / 3 + 180 = 2 * x / 3) : x = 540 := by
  sorry

#check tank_capacity

end tank_capacity_l1598_159891


namespace half_squared_equals_quarter_l1598_159808

theorem half_squared_equals_quarter : (1 / 2 : ℝ) ^ 2 = 0.25 := by
  sorry

end half_squared_equals_quarter_l1598_159808


namespace regression_line_intercept_l1598_159801

/-- Prove that a regression line with slope 1.23 passing through (4, 5) has y-intercept 0.08 -/
theorem regression_line_intercept (slope : ℝ) (center_x center_y : ℝ) (y_intercept : ℝ) : 
  slope = 1.23 → center_x = 4 → center_y = 5 → 
  center_y = slope * center_x + y_intercept →
  y_intercept = 0.08 := by
sorry

end regression_line_intercept_l1598_159801


namespace max_pages_copied_l1598_159889

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The available amount in dollars -/
def available_dollars : ℕ := 25

/-- Convert dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Calculate the number of full pages that can be copied -/
def pages_copied (cents : ℕ) (cost : ℕ) : ℕ := cents / cost

theorem max_pages_copied : 
  pages_copied (dollars_to_cents available_dollars) cost_per_page = 833 := by
  sorry

end max_pages_copied_l1598_159889


namespace modified_lucas_units_digit_l1598_159816

/-- Modified Lucas sequence -/
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | n + 2 => M (n + 1) + M n + 2

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem modified_lucas_units_digit :
  unitsDigit (M (M 6)) = unitsDigit (M 11) :=
sorry

end modified_lucas_units_digit_l1598_159816


namespace unique_solution_log_equation_l1598_159847

theorem unique_solution_log_equation :
  ∃! x : ℝ, (Real.log (2 * x + 1) = Real.log (x^2 - 2)) ∧ (2 * x + 1 > 0) ∧ (x^2 - 2 > 0) ∧ x = 3 :=
by sorry

end unique_solution_log_equation_l1598_159847


namespace exists_different_degree_same_characteristic_l1598_159819

/-- A characteristic of a polynomial -/
def characteristic (P : Polynomial ℝ) : ℝ := sorry

/-- Theorem: There exist two polynomials with different degrees but the same characteristic -/
theorem exists_different_degree_same_characteristic :
  ∃ (P1 P2 : Polynomial ℝ), 
    (Polynomial.degree P1 ≠ Polynomial.degree P2) ∧ 
    (characteristic P1 = characteristic P2) := by
  sorry

end exists_different_degree_same_characteristic_l1598_159819


namespace coefficient_of_x2y_div_3_l1598_159898

/-- Definition of a coefficient in a monomial -/
def coefficient (term : ℚ × (ℕ → ℕ)) : ℚ := term.1

/-- The monomial x^2 * y / 3 -/
def monomial : ℚ × (ℕ → ℕ) := (1/3, fun n => if n = 1 then 2 else if n = 2 then 1 else 0)

/-- Theorem: The coefficient of x^2 * y / 3 is 1/3 -/
theorem coefficient_of_x2y_div_3 : coefficient monomial = 1/3 := by sorry

end coefficient_of_x2y_div_3_l1598_159898


namespace repeating_decimal_to_fraction_l1598_159885

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), x * 10^(2*n + 2) - (x * 10^(2*n + 2)).floor = 0.36) ∧ x = 4/11 := by
  sorry

end repeating_decimal_to_fraction_l1598_159885


namespace smallest_divisor_of_Q_l1598_159838

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def Q (visible : Finset ℕ) : ℕ := 
  visible.prod id

theorem smallest_divisor_of_Q : 
  ∀ visible : Finset ℕ, visible ⊆ die_numbers → visible.card = 7 → 
    (∃ k : ℕ, Q visible = 192 * k) ∧ 
    ∀ m : ℕ, m < 192 → (∃ v : Finset ℕ, v ⊆ die_numbers ∧ v.card = 7 ∧ ¬(∃ k : ℕ, Q v = m * k)) :=
by sorry

end smallest_divisor_of_Q_l1598_159838


namespace wire_length_ratio_l1598_159851

/-- The length of each wire piece used by Bonnie, in inches -/
def bonnie_wire_length : ℕ := 8

/-- The number of wire pieces used by Bonnie -/
def bonnie_wire_count : ℕ := 12

/-- The length of each wire piece used by Clyde, in inches -/
def clyde_wire_length : ℕ := 2

/-- The side length of Clyde's unit cubes, in inches -/
def clyde_cube_side : ℕ := 1

/-- The number of wire pieces needed for one cube frame -/
def wire_pieces_per_cube : ℕ := 12

theorem wire_length_ratio :
  (bonnie_wire_length * bonnie_wire_count : ℚ) / 
  (clyde_wire_length * wire_pieces_per_cube * bonnie_wire_length ^ 3) = 1 / 128 := by
  sorry

end wire_length_ratio_l1598_159851


namespace student_count_problem_l1598_159842

theorem student_count_problem : 
  ∃ n : ℕ, n > 1 ∧ 
  (n - 1) % 2 = 1 ∧ 
  (n - 1) % 7 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m < n → (m - 1) % 2 ≠ 1 ∨ (m - 1) % 7 ≠ 1) ∧
  n = 44 := by
sorry

end student_count_problem_l1598_159842


namespace min_value_theorem_range_theorem_l1598_159820

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the left focus F₁
def F₁ : ℝ × ℝ := (-2, 0)

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Define a point M on the ellipse
def M : ℝ × ℝ := sorry

-- Distance between two points
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

-- Statement for the minimum value
theorem min_value_theorem :
  ∀ M, is_on_ellipse M.1 M.2 →
  ∃ m : ℝ, m = distance M P + (3/2) * distance M F₁ ∧
  m ≥ 11/2 ∧
  ∃ M₀, is_on_ellipse M₀.1 M₀.2 ∧ distance M₀ P + (3/2) * distance M₀ F₁ = 11/2 :=
sorry

-- Statement for the range of values
theorem range_theorem :
  ∀ M, is_on_ellipse M.1 M.2 →
  ∃ r : ℝ, r = distance M P + distance M F₁ ∧
  6 - Real.sqrt 2 < r ∧ r < 6 + Real.sqrt 2 :=
sorry

end min_value_theorem_range_theorem_l1598_159820


namespace min_value_expression_l1598_159836

theorem min_value_expression (a b c : ℝ) (h1 : b > a) (h2 : a > c) (h3 : b ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c + a)^2) / b^2 ≥ 5.5 ∧
  ∃ (a' b' c' : ℝ), b' > a' ∧ a' > c' ∧ b' ≠ 0 ∧
    ((a' + b')^2 + (b' + c')^2 + (c' + a')^2) / b'^2 = 5.5 :=
by sorry

end min_value_expression_l1598_159836


namespace parallel_vectors_x_value_l1598_159805

/-- Given two parallel vectors a = (-3, 2) and b = (1, x), prove that x = -2/3 -/
theorem parallel_vectors_x_value (x : ℚ) : 
  let a : ℚ × ℚ := (-3, 2)
  let b : ℚ × ℚ := (1, x)
  (∃ (k : ℚ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) → x = -2/3 :=
by sorry

end parallel_vectors_x_value_l1598_159805


namespace absolute_value_sum_inequality_l1598_159813

theorem absolute_value_sum_inequality (x : ℝ) :
  |x - 1| + |x - 2| > 5 ↔ x < -1 ∨ x > 4 :=
by sorry

end absolute_value_sum_inequality_l1598_159813


namespace total_candy_count_l1598_159869

theorem total_candy_count (brother_candy : ℕ) (wendy_boxes : ℕ) (pieces_per_box : ℕ) : 
  brother_candy = 6 → wendy_boxes = 2 → pieces_per_box = 3 →
  brother_candy + wendy_boxes * pieces_per_box = 12 :=
by sorry

end total_candy_count_l1598_159869


namespace set_intersection_equality_l1598_159804

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | |x - 1| > 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define the open interval (2, 3]
def open_interval : Set ℝ := Set.Ioc 2 3

-- Theorem statement
theorem set_intersection_equality :
  (Set.compl A ∩ B) = open_interval :=
sorry

end set_intersection_equality_l1598_159804


namespace simplify_expression_l1598_159806

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end simplify_expression_l1598_159806


namespace kyles_rent_calculation_l1598_159839

def monthly_income : ℕ := 3200

def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries_eating_out : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350
def gas_maintenance : ℕ := 350

def total_expenses : ℕ :=
  utilities + retirement_savings + groceries_eating_out + insurance +
  miscellaneous + car_payment + gas_maintenance

def rent : ℕ := monthly_income - total_expenses

theorem kyles_rent_calculation :
  rent = 1250 :=
sorry

end kyles_rent_calculation_l1598_159839


namespace arthurs_fitness_routine_l1598_159863

/-- The expected number of chocolate balls eaten during Arthur's fitness routine -/
def expected_chocolate_balls (n : ℕ) : ℝ :=
  if n < 2 then 0 else 1

/-- Arthur's fitness routine on Édes Street -/
theorem arthurs_fitness_routine (n : ℕ) (h : n ≥ 2) :
  expected_chocolate_balls n = 1 := by
  sorry

#check arthurs_fitness_routine

end arthurs_fitness_routine_l1598_159863


namespace intersection_M_N_l1598_159882

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end intersection_M_N_l1598_159882


namespace princess_puff_whisker_count_l1598_159873

/-- The number of whiskers Princess Puff has -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers : ℕ := 22

theorem princess_puff_whisker_count :
  princess_puff_whiskers = 14 ∧
  catman_do_whiskers = 22 ∧
  catman_do_whiskers = 2 * princess_puff_whiskers - 6 :=
by sorry

end princess_puff_whisker_count_l1598_159873


namespace two_valid_antonyms_exist_l1598_159864

/-- A word is represented as a string of characters. -/
def Word := String

/-- The maximum allowed length for an antonym. -/
def MaxLength : Nat := 10

/-- Predicate to check if a word is an antonym of "seldom". -/
def IsAntonymOfSeldom (w : Word) : Prop := sorry

/-- Predicate to check if two words have distinct meanings. -/
def HasDistinctMeaning (w1 w2 : Word) : Prop := sorry

/-- Theorem stating the existence of two valid antonyms for "seldom". -/
theorem two_valid_antonyms_exist : 
  ∃ (w1 w2 : Word), 
    IsAntonymOfSeldom w1 ∧ 
    IsAntonymOfSeldom w2 ∧ 
    w1.length ≤ MaxLength ∧ 
    w2.length ≤ MaxLength ∧ 
    w1.front = 'o' ∧ 
    w2.front = 'u' ∧ 
    HasDistinctMeaning w1 w2 :=
  sorry

end two_valid_antonyms_exist_l1598_159864
