import Mathlib

namespace NUMINAMATH_CALUDE_not_decreasing_everywhere_l2892_289291

theorem not_decreasing_everywhere (f : ℝ → ℝ) (h : f 1 < f 2) :
  ¬(∀ x y : ℝ, x < y → f x ≥ f y) :=
sorry

end NUMINAMATH_CALUDE_not_decreasing_everywhere_l2892_289291


namespace NUMINAMATH_CALUDE_remainder_theorem_l2892_289281

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 60 * k + 1) :
  (n^2 + 2*n + 3) % 60 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2892_289281


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l2892_289293

theorem magnitude_of_complex_power : 
  Complex.abs ((2 : ℂ) + 2 * Complex.I * Real.sqrt 2) ^ 6 = 1728 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l2892_289293


namespace NUMINAMATH_CALUDE_bicycle_selection_l2892_289250

theorem bicycle_selection (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  (n * (n - 1) * (n - 2)) = 2730 :=
sorry

end NUMINAMATH_CALUDE_bicycle_selection_l2892_289250


namespace NUMINAMATH_CALUDE_multiplication_increase_l2892_289239

theorem multiplication_increase (n : ℕ) (x : ℚ) (h : n = 25) :
  n * x = n + 375 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_increase_l2892_289239


namespace NUMINAMATH_CALUDE_a_seven_value_l2892_289201

/-- An arithmetic sequence where the reciprocals of terms form an arithmetic sequence -/
def reciprocal_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, 1 / a (n + 1) - 1 / a n = d

theorem a_seven_value (a : ℕ → ℝ) 
  (h_seq : reciprocal_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 4) :
  a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_a_seven_value_l2892_289201


namespace NUMINAMATH_CALUDE_twelve_team_tournament_matches_l2892_289208

/-- Calculates the total number of matches in a round-robin tournament. -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 12 teams, the total number of matches is 66. -/
theorem twelve_team_tournament_matches :
  total_matches 12 = 66 := by
  sorry

#eval total_matches 12  -- This will evaluate to 66

end NUMINAMATH_CALUDE_twelve_team_tournament_matches_l2892_289208


namespace NUMINAMATH_CALUDE_red_straws_per_mat_l2892_289215

theorem red_straws_per_mat (orange_per_mat green_per_mat total_straws mats : ℕ)
  (h1 : orange_per_mat = 30)
  (h2 : green_per_mat = orange_per_mat / 2)
  (h3 : total_straws = 650)
  (h4 : mats = 10) :
  (total_straws - (orange_per_mat + green_per_mat) * mats) / mats = 20 :=
by sorry

end NUMINAMATH_CALUDE_red_straws_per_mat_l2892_289215


namespace NUMINAMATH_CALUDE_system_solutions_l2892_289213

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := y + Real.sqrt (y - 3 * x) + 3 * x = 12
def equation2 (x y : ℝ) : Prop := y^2 + y - 3 * x - 9 * x^2 = 144

-- State the theorem
theorem system_solutions :
  (∀ x y : ℝ, equation1 x y ∧ equation2 x y ↔ (x = -24 ∧ y = 72) ∨ (x = -4/3 ∧ y = 12)) :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l2892_289213


namespace NUMINAMATH_CALUDE_henrys_age_l2892_289255

/-- Given that the sum of Henry and Jill's present ages is 41, and 7 years ago Henry was twice the age of Jill, prove that Henry's present age is 25. -/
theorem henrys_age (h_age j_age : ℕ) 
  (sum_condition : h_age + j_age = 41)
  (past_condition : h_age - 7 = 2 * (j_age - 7)) :
  h_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_henrys_age_l2892_289255


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l2892_289246

/-- Proves that the ratio of Rahul's present age to Deepak's present age is 4:3 -/
theorem rahul_deepak_age_ratio :
  let rahul_future_age : ℕ := 26
  let years_to_future : ℕ := 6
  let deepak_present_age : ℕ := 15
  let rahul_present_age : ℕ := rahul_future_age - years_to_future
  (rahul_present_age : ℚ) / deepak_present_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l2892_289246


namespace NUMINAMATH_CALUDE_sum_of_terms_l2892_289261

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  roots_property : a 2 + a 16 = 6 ∧ a 2 * a 16 = -1

/-- The sum of specific terms in the arithmetic sequence equals 15 -/
theorem sum_of_terms (seq : ArithmeticSequence) : 
  seq.a 5 + seq.a 6 + seq.a 9 + seq.a 12 + seq.a 13 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_terms_l2892_289261


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_l2892_289282

theorem consecutive_integers_sqrt (x y : ℤ) : 
  (y = x + 1) →  -- x and y are consecutive integers
  (x < Real.sqrt 30) →  -- x < √30
  (Real.sqrt 30 < y) →  -- √30 < y
  Real.sqrt (2 * x + y) = 4 ∨ Real.sqrt (2 * x + y) = -4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_l2892_289282


namespace NUMINAMATH_CALUDE_alvin_age_l2892_289218

/-- Alvin's age -/
def A : ℕ := 30

/-- Simon's age -/
def S : ℕ := 10

/-- Theorem stating that Alvin's age is 30, given the conditions -/
theorem alvin_age : 
  (S + 5 = A / 2) → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_alvin_age_l2892_289218


namespace NUMINAMATH_CALUDE_square_of_negative_two_a_squared_l2892_289266

theorem square_of_negative_two_a_squared (a : ℝ) : (-2 * a^2)^2 = 4 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_a_squared_l2892_289266


namespace NUMINAMATH_CALUDE_cube_side_ratio_l2892_289286

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 16 → a / b = 4 := by
sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l2892_289286


namespace NUMINAMATH_CALUDE_chess_club_officers_l2892_289271

def choose_officers (n : ℕ) (k : ℕ) (special_pair : ℕ) : ℕ :=
  (n - special_pair).choose k * (n - special_pair - k + 1).choose (k - 1) +
  special_pair * (special_pair - 1) * (n - special_pair)

theorem chess_club_officers :
  choose_officers 24 3 2 = 9372 :=
sorry

end NUMINAMATH_CALUDE_chess_club_officers_l2892_289271


namespace NUMINAMATH_CALUDE_nested_sum_value_l2892_289248

def nested_sum (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (n + 1000 : ℚ) + (1/3) * nested_sum (n-1)

theorem nested_sum_value :
  nested_sum 999 = 999.5 + 1498.5 * 3^997 :=
sorry

end NUMINAMATH_CALUDE_nested_sum_value_l2892_289248


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l2892_289230

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then x = -2 -/
theorem parallel_vectors_imply_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l2892_289230


namespace NUMINAMATH_CALUDE_rectangle_area_l2892_289207

/-- Given a rectangle with perimeter 100 meters and length three times its width, 
    its area is 468.75 square meters. -/
theorem rectangle_area (w : ℝ) (l : ℝ) : 
  (2 * l + 2 * w = 100) → 
  (l = 3 * w) → 
  (l * w = 468.75) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2892_289207


namespace NUMINAMATH_CALUDE_combined_eighth_grade_percentage_l2892_289292

/-- Represents the percentage of 8th grade students in a school -/
structure School :=
  (total_students : ℕ)
  (eighth_grade_percentage : ℚ)

/-- Calculates the total number of 8th grade students in both schools -/
def total_eighth_graders (oakwood pinecrest : School) : ℚ :=
  (oakwood.total_students : ℚ) * oakwood.eighth_grade_percentage / 100 +
  (pinecrest.total_students : ℚ) * pinecrest.eighth_grade_percentage / 100

/-- Calculates the total number of students in both schools -/
def total_students (oakwood pinecrest : School) : ℕ :=
  oakwood.total_students + pinecrest.total_students

/-- Theorem stating that the percentage of 8th graders in both schools combined is 57% -/
theorem combined_eighth_grade_percentage 
  (oakwood : School) 
  (pinecrest : School)
  (h1 : oakwood.total_students = 150)
  (h2 : pinecrest.total_students = 250)
  (h3 : oakwood.eighth_grade_percentage = 60)
  (h4 : pinecrest.eighth_grade_percentage = 55) :
  (total_eighth_graders oakwood pinecrest) / (total_students oakwood pinecrest : ℚ) * 100 = 57 :=
sorry

end NUMINAMATH_CALUDE_combined_eighth_grade_percentage_l2892_289292


namespace NUMINAMATH_CALUDE_oxford_high_school_teachers_l2892_289209

theorem oxford_high_school_teachers (num_classes : ℕ) (students_per_class : ℕ) (total_people : ℕ) :
  num_classes = 15 →
  students_per_class = 20 →
  total_people = 349 →
  total_people = num_classes * students_per_class + 1 + 48 :=
by
  sorry

end NUMINAMATH_CALUDE_oxford_high_school_teachers_l2892_289209


namespace NUMINAMATH_CALUDE_range_of_a_for_three_roots_l2892_289214

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x + a

-- State the theorem
theorem range_of_a_for_three_roots (a : ℝ) :
  (∃ m n p : ℝ, m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ 
    f a m = 2024 ∧ f a n = 2024 ∧ f a p = 2024) →
  2022 < a ∧ a < 2026 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_three_roots_l2892_289214


namespace NUMINAMATH_CALUDE_tea_cost_price_l2892_289265

/-- Represents the cost price per kg of the 80 kg tea -/
def x : ℝ := 15

/-- The total amount of tea in kg -/
def total_tea : ℝ := 100

/-- The amount of tea with known cost price in kg -/
def known_tea : ℝ := 20

/-- The amount of tea with unknown cost price in kg -/
def unknown_tea : ℝ := 80

/-- The cost price per kg of the known tea -/
def known_tea_price : ℝ := 20

/-- The sale price per kg of the mixed tea -/
def sale_price : ℝ := 21.6

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.35

theorem tea_cost_price : 
  (unknown_tea * x + known_tea * known_tea_price) * (1 + profit_percentage) = 
  total_tea * sale_price := by sorry

end NUMINAMATH_CALUDE_tea_cost_price_l2892_289265


namespace NUMINAMATH_CALUDE_a_investment_value_l2892_289228

/-- Represents the investment and profit distribution in a partnership business -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that given the conditions of the problem, A's investment is 8000 -/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 4000)
  (hc : p.c_investment = 2000)
  (hprofit : p.total_profit = 252000)
  (hshare : p.c_profit_share = 36000)
  : p.a_investment = 8000 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_value_l2892_289228


namespace NUMINAMATH_CALUDE_tony_age_is_twelve_l2892_289232

/-- Represents Tony's work and payment information -/
structure TonyWork where
  hoursPerDay : ℕ
  payPerHourPerYear : ℚ
  workDays : ℕ
  totalEarnings : ℚ

/-- Calculates Tony's age based on his work information -/
def calculateAge (work : TonyWork) : ℕ :=
  sorry

/-- Theorem stating that Tony's age at the end of the five-month period was 12 -/
theorem tony_age_is_twelve (work : TonyWork) 
  (h1 : work.hoursPerDay = 2)
  (h2 : work.payPerHourPerYear = 1)
  (h3 : work.workDays = 60)
  (h4 : work.totalEarnings = 1140) :
  calculateAge work = 12 :=
sorry

end NUMINAMATH_CALUDE_tony_age_is_twelve_l2892_289232


namespace NUMINAMATH_CALUDE_theresa_final_week_hours_l2892_289275

def hours_worked : List Nat := [9, 12, 6, 13, 11]
def total_weeks : Nat := 6
def required_average : Nat := 9

theorem theresa_final_week_hours :
  ∃ x : Nat, 
    (hours_worked.sum + x) / total_weeks = required_average ∧ 
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_theresa_final_week_hours_l2892_289275


namespace NUMINAMATH_CALUDE_octahedron_edge_length_is_four_l2892_289205

/-- A regular octahedron circumscribed around four identical balls -/
structure OctahedronWithBalls where
  /-- The radius of each ball -/
  ball_radius : ℝ
  /-- The edge length of the octahedron -/
  edge_length : ℝ
  /-- The condition that three balls are touching each other on the floor -/
  balls_touching : ball_radius = 2
  /-- The condition that the fourth ball rests on top of the other three -/
  fourth_ball_on_top : True

/-- The theorem stating that the edge length of the octahedron is 4 units -/
theorem octahedron_edge_length_is_four (o : OctahedronWithBalls) : o.edge_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_edge_length_is_four_l2892_289205


namespace NUMINAMATH_CALUDE_sum_after_changes_l2892_289279

theorem sum_after_changes (A B : ℤ) (h : A + B = 100) : 
  (A - 35) + (B + 15) = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_changes_l2892_289279


namespace NUMINAMATH_CALUDE_square_root_sum_l2892_289296

theorem square_root_sum (y : ℝ) :
  Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4 →
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l2892_289296


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2892_289277

/-- Given that 50 cows eat 50 bags of husk in 50 days, 
    prove that one cow will eat one bag of husk in the same number of days. -/
theorem cow_husk_consumption (days : ℕ) 
  (h : 50 * 50 = 50 * days) : 
  1 * 1 = 1 * days :=
by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2892_289277


namespace NUMINAMATH_CALUDE_square_of_complex_fraction_l2892_289236

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem square_of_complex_fraction : (2 * i / (1 - i)) ^ 2 = -2 * i := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_fraction_l2892_289236


namespace NUMINAMATH_CALUDE_siblings_total_weight_siblings_total_weight_is_88_l2892_289221

/-- The total weight of two siblings, where one weighs 50 kg and the other weighs 12 kg less. -/
theorem siblings_total_weight : ℝ :=
  let antonio_weight : ℝ := 50
  let sister_weight : ℝ := antonio_weight - 12
  antonio_weight + sister_weight
  
/-- Prove that the total weight of the siblings is 88 kg. -/
theorem siblings_total_weight_is_88 : siblings_total_weight = 88 := by
  sorry

end NUMINAMATH_CALUDE_siblings_total_weight_siblings_total_weight_is_88_l2892_289221


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l2892_289278

/-- Given a point M with coordinates (2,3), prove that its symmetric point N 
    with respect to the x-axis has coordinates (2, -3) -/
theorem symmetric_point_x_axis : 
  let M : ℝ × ℝ := (2, 3)
  let N : ℝ × ℝ := (M.1, -M.2)
  N = (2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l2892_289278


namespace NUMINAMATH_CALUDE_selectThreePeopleIs600_l2892_289290

/-- The number of ways to select 3 people from a 5×5 matrix,
    such that no two selected people are in the same row or column. -/
def selectThreePeople : ℕ :=
  let numColumns : ℕ := 5
  let numRows : ℕ := 5
  let numPeopleToSelect : ℕ := 3
  let waysToChooseColumns : ℕ := Nat.choose numColumns numPeopleToSelect
  let waysToChooseFirstPerson : ℕ := numRows
  let waysToChooseSecondPerson : ℕ := numRows - 1
  let waysToChooseThirdPerson : ℕ := numRows - 2
  waysToChooseColumns * waysToChooseFirstPerson * waysToChooseSecondPerson * waysToChooseThirdPerson

/-- Theorem stating that the number of ways to select 3 people
    from a 5×5 matrix, such that no two selected people are in
    the same row or column, is equal to 600. -/
theorem selectThreePeopleIs600 : selectThreePeople = 600 := by
  sorry

end NUMINAMATH_CALUDE_selectThreePeopleIs600_l2892_289290


namespace NUMINAMATH_CALUDE_marks_total_votes_l2892_289200

/-- Calculates the total votes Mark received in an election given specific conditions --/
theorem marks_total_votes (first_area_voters : ℕ) 
  (first_area_undecided_percent : ℚ)
  (first_area_mark_percent : ℚ)
  (remaining_area_mark_multiplier : ℕ)
  (remaining_area_undecided_percent : ℚ)
  (remaining_area_population_increase : ℚ) :
  first_area_voters = 100000 →
  first_area_undecided_percent = 5 / 100 →
  first_area_mark_percent = 70 / 100 →
  remaining_area_mark_multiplier = 2 →
  remaining_area_undecided_percent = 7 / 100 →
  remaining_area_population_increase = 20 / 100 →
  ∃ (total_votes : ℕ), total_votes = 199500 ∧
    total_votes = 
      (first_area_voters * (1 - first_area_undecided_percent) * first_area_mark_percent).floor +
      (remaining_area_mark_multiplier * 
        (first_area_voters * (1 - first_area_undecided_percent) * first_area_mark_percent).floor) :=
by
  sorry


end NUMINAMATH_CALUDE_marks_total_votes_l2892_289200


namespace NUMINAMATH_CALUDE_initial_seashells_count_l2892_289274

/-- The number of seashells Tim found initially -/
def initial_seashells : ℕ := sorry

/-- The number of starfish Tim found -/
def starfish : ℕ := 110

/-- The number of seashells Tim gave to Sara -/
def seashells_given : ℕ := 172

/-- The number of seashells Tim has now -/
def current_seashells : ℕ := 507

/-- Theorem stating that the initial number of seashells is equal to 
    the current number of seashells plus the number of seashells given away -/
theorem initial_seashells_count : 
  initial_seashells = current_seashells + seashells_given := by sorry

end NUMINAMATH_CALUDE_initial_seashells_count_l2892_289274


namespace NUMINAMATH_CALUDE_f_zero_gt_f_one_l2892_289288

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def isEvenOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, -a ≤ x ∧ x ≤ a → f x = f (-x)

def isMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → (f x ≤ f y ∨ f y ≤ f x)

-- State the theorem
theorem f_zero_gt_f_one
  (h_even : isEvenOn f 5)
  (h_mono : isMonotonicOn f 0 5)
  (h_ineq : f (-3) < f (-1)) :
  f 0 > f 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_gt_f_one_l2892_289288


namespace NUMINAMATH_CALUDE_percentage_of_filled_holes_l2892_289264

theorem percentage_of_filled_holes (total_holes : ℕ) (unfilled_holes : ℕ) 
  (h1 : total_holes = 8) 
  (h2 : unfilled_holes = 2) 
  (h3 : unfilled_holes < total_holes) : 
  (((total_holes - unfilled_holes) : ℚ) / total_holes) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_filled_holes_l2892_289264


namespace NUMINAMATH_CALUDE_set_equality_l2892_289203

def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

theorem set_equality : (A ∩ B) ∪ C = (A ∪ C) ∩ (B ∪ C) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2892_289203


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l2892_289253

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 7 = 0
def l (x : ℝ) : Prop := x = 1

-- Define the intersection points
def intersection_points (x y : ℝ) : Prop := C₁ x y ∧ C₂ x y

-- Define the line y = x
def y_eq_x (x y : ℝ) : Prop := y = x

-- Define circle C₃
def C₃ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

theorem circle_intersection_theorem :
  (∀ x y, intersection_points x y → l x) ∧
  (∃ x₀ y₀, C₃ x₀ y₀ ∧ y_eq_x x₀ y₀ ∧ (∀ x y, intersection_points x y → C₃ x y)) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l2892_289253


namespace NUMINAMATH_CALUDE_skew_symmetric_times_symmetric_is_zero_l2892_289223

/-- Given a skew-symmetric matrix A and a symmetric matrix B, prove that their product is the zero matrix -/
theorem skew_symmetric_times_symmetric_is_zero (a b c : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, c, -b; -c, 0, a; b, -a, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![a^2, a*b, a*c; a*b, b^2, b*c; a*c, b*c, c^2]
  A * B = 0 := by sorry

end NUMINAMATH_CALUDE_skew_symmetric_times_symmetric_is_zero_l2892_289223


namespace NUMINAMATH_CALUDE_larger_circle_radius_l2892_289219

/-- The radius of a circle that is internally tangent to four externally tangent circles of radius 2 -/
theorem larger_circle_radius : ℝ := by
  -- Define the radius of the smaller circles
  let small_radius : ℝ := 2

  -- Define the number of smaller circles
  let num_small_circles : ℕ := 4

  -- Define the angle between the centers of adjacent smaller circles
  let angle_between_centers : ℝ := 360 / num_small_circles

  -- Define the radius of the larger circle
  let large_radius : ℝ := small_radius * (1 + Real.sqrt 2)

  -- Prove that the radius of the larger circle is 2(1 + √2)
  sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l2892_289219


namespace NUMINAMATH_CALUDE_total_apples_packed_l2892_289237

/-- Calculates the total number of apples packed in two weeks under specific conditions -/
theorem total_apples_packed (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_per_week : ℕ) (reduced_apples : ℕ) : 
  apples_per_box = 40 →
  boxes_per_day = 50 →
  days_per_week = 7 →
  reduced_apples = 500 →
  (apples_per_box * boxes_per_day * days_per_week) + 
  ((apples_per_box * boxes_per_day - reduced_apples) * days_per_week) = 24500 := by
sorry

end NUMINAMATH_CALUDE_total_apples_packed_l2892_289237


namespace NUMINAMATH_CALUDE_Q_when_b_is_one_Q_subset_P_iff_b_in_range_l2892_289206

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}
def Q (b : ℝ) : Set ℝ := {x | x^2 - (b+2)*x + 2*b ≤ 0}

-- Theorem 1: When b = 1, Q = {x | 1 ≤ x ≤ 2}
theorem Q_when_b_is_one : Q 1 = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: Q ⊆ P if and only if b ∈ [1, 4]
theorem Q_subset_P_iff_b_in_range : ∀ b : ℝ, Q b ⊆ P ↔ 1 ≤ b ∧ b ≤ 4 := by sorry

end NUMINAMATH_CALUDE_Q_when_b_is_one_Q_subset_P_iff_b_in_range_l2892_289206


namespace NUMINAMATH_CALUDE_no_linear_term_iff_m_eq_two_l2892_289298

/-- The expression (2x-m)(x+1) does not contain a linear term of x if and only if m = 2 -/
theorem no_linear_term_iff_m_eq_two (x m : ℝ) : 
  (2 * x - m) * (x + 1) = 2 * x^2 - m ↔ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_iff_m_eq_two_l2892_289298


namespace NUMINAMATH_CALUDE_football_original_price_l2892_289269

theorem football_original_price : 
  ∀ (original_price : ℝ), 
  (original_price * 0.8 + 25 = original_price) → 
  original_price = 125 := by
sorry

end NUMINAMATH_CALUDE_football_original_price_l2892_289269


namespace NUMINAMATH_CALUDE_orange_pill_cost_l2892_289287

/-- The cost of an orange pill given the conditions of Bob's treatment --/
theorem orange_pill_cost : 
  ∀ (duration : ℕ) (total_cost : ℚ) (blue_pill_cost : ℚ),
  duration = 21 →
  total_cost = 735 →
  blue_pill_cost + 3 + blue_pill_cost = total_cost / duration →
  blue_pill_cost + 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_orange_pill_cost_l2892_289287


namespace NUMINAMATH_CALUDE_cos_minus_sin_2pi_non_decreasing_l2892_289225

def T_non_decreasing (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) ≥ f x

theorem cos_minus_sin_2pi_non_decreasing :
  T_non_decreasing (fun x => Real.cos x - Real.sin x) (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_2pi_non_decreasing_l2892_289225


namespace NUMINAMATH_CALUDE_S_is_infinite_l2892_289210

/-- The largest prime divisor of a positive integer -/
def largest_prime_divisor (n : ℕ+) : ℕ+ :=
  sorry

/-- The set of positive integers n where the largest prime divisor of n^4 + n^2 + 1
    equals the largest prime divisor of (n+1)^4 + (n+1)^2 + 1 -/
def S : Set ℕ+ :=
  {n | largest_prime_divisor (n^4 + n^2 + 1) = largest_prime_divisor ((n+1)^4 + (n+1)^2 + 1)}

/-- The main theorem: S is an infinite set -/
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_is_infinite_l2892_289210


namespace NUMINAMATH_CALUDE_correct_expression_l2892_289252

theorem correct_expression (x y : ℚ) (h : x / y = 5 / 6) : 
  (x + 3 * y) / x = 23 / 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_expression_l2892_289252


namespace NUMINAMATH_CALUDE_three_piece_suit_cost_l2892_289229

/-- The cost of a jacket in pounds -/
def jacket_cost : ℝ := sorry

/-- The cost of a pair of trousers in pounds -/
def trousers_cost : ℝ := sorry

/-- The cost of a waistcoat in pounds -/
def waistcoat_cost : ℝ := sorry

/-- Two jackets and three pairs of trousers cost £380 -/
axiom two_jackets_three_trousers : 2 * jacket_cost + 3 * trousers_cost = 380

/-- A pair of trousers costs the same as two waistcoats -/
axiom trousers_equals_two_waistcoats : trousers_cost = 2 * waistcoat_cost

/-- The cost of a three-piece suit is £190 -/
theorem three_piece_suit_cost : jacket_cost + trousers_cost + waistcoat_cost = 190 := by
  sorry

end NUMINAMATH_CALUDE_three_piece_suit_cost_l2892_289229


namespace NUMINAMATH_CALUDE_garden_width_is_correct_garden_area_is_correct_l2892_289273

/-- Represents a rectangular flower garden -/
structure FlowerGarden where
  length : ℝ
  width : ℝ
  area : ℝ

/-- The flower garden has the given dimensions -/
def garden : FlowerGarden where
  length := 4
  width := 35.8
  area := 143.2

/-- Theorem: The width of the flower garden is 35.8 meters -/
theorem garden_width_is_correct : garden.width = 35.8 := by
  sorry

/-- Theorem: The area of the garden is equal to length times width -/
theorem garden_area_is_correct : garden.area = garden.length * garden.width := by
  sorry

end NUMINAMATH_CALUDE_garden_width_is_correct_garden_area_is_correct_l2892_289273


namespace NUMINAMATH_CALUDE_tan_alpha_for_point_l2892_289233

/-- If the terminal side of angle α passes through the point (-4, -3), then tan α = 3/4 -/
theorem tan_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -4 ∧ t * Real.sin α = -3) → 
  Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_for_point_l2892_289233


namespace NUMINAMATH_CALUDE_bed_price_ratio_bed_to_frame_ratio_l2892_289259

/-- Given a bed frame price, a bed price multiple, a discount rate, and a final price,
    calculate the ratio of the bed's price to the bed frame's price. -/
theorem bed_price_ratio
  (bed_frame_price : ℝ)
  (bed_price_multiple : ℝ)
  (discount_rate : ℝ)
  (final_price : ℝ)
  (h1 : bed_frame_price = 75)
  (h2 : discount_rate = 0.2)
  (h3 : final_price = 660)
  (h4 : (1 - discount_rate) * (bed_frame_price + bed_frame_price * bed_price_multiple) = final_price) :
  bed_price_multiple = 10 := by
sorry

/-- The ratio of the bed's price to the bed frame's price is 10:1. -/
theorem bed_to_frame_ratio (bed_price_multiple : ℝ) 
  (h : bed_price_multiple = 10) : 
  bed_price_multiple / 1 = 10 / 1 := by
sorry

end NUMINAMATH_CALUDE_bed_price_ratio_bed_to_frame_ratio_l2892_289259


namespace NUMINAMATH_CALUDE_biquadratic_equation_with_given_root_l2892_289227

theorem biquadratic_equation_with_given_root (x : ℝ) :
  (2 + Real.sqrt 3 : ℝ) ^ 4 - 14 * (2 + Real.sqrt 3 : ℝ) ^ 2 + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_biquadratic_equation_with_given_root_l2892_289227


namespace NUMINAMATH_CALUDE_f_is_square_iff_n_eq_one_l2892_289224

/-- The number of non-empty subsets of {1, ..., n} with gcd 1 -/
def f (n : ℕ+) : ℕ := sorry

/-- f(n) is a perfect square iff n = 1 -/
theorem f_is_square_iff_n_eq_one (n : ℕ+) : 
  ∃ m : ℕ, f n = m ^ 2 ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_f_is_square_iff_n_eq_one_l2892_289224


namespace NUMINAMATH_CALUDE_solve_for_A_l2892_289276

theorem solve_for_A (A : ℤ) (h : A - 10 = 15) : A = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l2892_289276


namespace NUMINAMATH_CALUDE_largest_power_dividing_powProduct_l2892_289297

/-- pow(n) is the largest power of the largest prime that divides n -/
def pow (n : ℕ) : ℕ :=
  sorry

/-- The product of pow(n) from 2 to 2023 -/
def powProduct : ℕ :=
  sorry

theorem largest_power_dividing_powProduct : 
  (∀ m : ℕ, 462^m ∣ powProduct → m ≤ 202) ∧ 462^202 ∣ powProduct :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_powProduct_l2892_289297


namespace NUMINAMATH_CALUDE_inequality_proof_l2892_289270

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2892_289270


namespace NUMINAMATH_CALUDE_total_wax_required_l2892_289283

/-- Given the amount of wax already available and the additional amount needed,
    calculate the total wax required for the feathers. -/
theorem total_wax_required 
  (wax_available : ℕ) 
  (wax_needed : ℕ) 
  (h1 : wax_available = 331) 
  (h2 : wax_needed = 22) : 
  wax_available + wax_needed = 353 := by
  sorry

end NUMINAMATH_CALUDE_total_wax_required_l2892_289283


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2892_289217

theorem expression_simplification_and_evaluation (x : ℝ) 
  (hx_neq_neg1 : x ≠ -1) (hx_neq_0 : x ≠ 0) (hx_neq_1 : x ≠ 1) :
  (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) = 1 / (x + 1) ∧
  (1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2892_289217


namespace NUMINAMATH_CALUDE_cube_sum_equals_thirteen_l2892_289202

theorem cube_sum_equals_thirteen (a b : ℝ) 
  (h1 : a^3 - 3*a*b^2 = 39)
  (h2 : b^3 - 3*a^2*b = 26) :
  a^2 + b^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_thirteen_l2892_289202


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l2892_289216

def initial_amount : ℚ := 204
def spend_half (x : ℚ) : ℚ := x / 2

theorem isabel_piggy_bank :
  spend_half (spend_half initial_amount) = 51 := by
  sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l2892_289216


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2892_289284

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y t : ℝ, f (x + t + f y) = f (f x) + f t + y) →
  (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2892_289284


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2892_289249

/-- Given a quadratic equation of the form (kx^2 + 5kx + k) = 0 with equal roots when k = 0.64,
    the coefficient of x^2 is 0.64 -/
theorem quadratic_coefficient (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 5 * k * x + k = 0) ∧ 
  (∀ x y : ℝ, k * x^2 + 5 * k * x + k = 0 ∧ k * y^2 + 5 * k * y + k = 0 → x = y) ∧
  k = 0.64 → 
  k = 0.64 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2892_289249


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l2892_289220

/-- Given a cloth sale with a loss, calculate the cost price per metre. -/
theorem cost_price_per_metre
  (total_metres : ℕ)
  (total_selling_price : ℕ)
  (loss_per_metre : ℕ)
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_metre = 5) :
  (total_selling_price + total_metres * loss_per_metre) / total_metres = 35 := by
  sorry

#check cost_price_per_metre

end NUMINAMATH_CALUDE_cost_price_per_metre_l2892_289220


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l2892_289267

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (n : ℕ) (green red : ℕ) (blue yellow : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 7 2 3 1 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l2892_289267


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2892_289258

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (2 + t * Complex.I) = 4 * Real.sqrt 10 → t = 2 * Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2892_289258


namespace NUMINAMATH_CALUDE_integer_set_condition_l2892_289263

theorem integer_set_condition (a : ℕ+) : 
  (∃ X : Finset ℤ, X.card = 6 ∧ 
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ 36 → 
      ∃ x y : ℤ, x ∈ X ∧ y ∈ X ∧ 37 ∣ (a.val * x + y - k))
  ↔ (a.val % 37 = 6 ∨ a.val % 37 = 31) :=
by sorry

end NUMINAMATH_CALUDE_integer_set_condition_l2892_289263


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2892_289285

theorem isosceles_triangle_base_length 
  (congruent_side : ℝ) 
  (perimeter : ℝ) 
  (h1 : congruent_side = 6) 
  (h2 : perimeter = 20) :
  perimeter - 2 * congruent_side = 8 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2892_289285


namespace NUMINAMATH_CALUDE_amy_started_with_101_seeds_l2892_289260

/-- The number of seeds Amy planted in her garden -/
def amy_garden_problem (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + small_gardens * seeds_per_small_garden

/-- Theorem stating that Amy started with 101 seeds -/
theorem amy_started_with_101_seeds :
  amy_garden_problem 47 9 6 = 101 := by
  sorry

end NUMINAMATH_CALUDE_amy_started_with_101_seeds_l2892_289260


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l2892_289294

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 → h1 > 0 → r2 > 0 → h2 > 0 →
  r2 = 1.1 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.21 * h2 := by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l2892_289294


namespace NUMINAMATH_CALUDE_simplify_expression_l2892_289242

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 4*(2^(n+1))) / (4*(2^(n+4))) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2892_289242


namespace NUMINAMATH_CALUDE_triangle_line_equations_l2892_289243

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Given triangle -/
def givenTriangle : Triangle :=
  { A := (-5, 0)
    B := (3, -3)
    C := (0, 2) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of line AC -/
def lineAC : LineEquation :=
  { a := 2
    b := -5
    c := 10 }

/-- The equation of the median to side BC -/
def medianBC : LineEquation :=
  { a := 1
    b := 13
    c := 5 }

theorem triangle_line_equations (t : Triangle) :
  t = givenTriangle →
  (lineAC.a * t.A.1 + lineAC.b * t.A.2 + lineAC.c = 0 ∧
   lineAC.a * t.C.1 + lineAC.b * t.C.2 + lineAC.c = 0) ∧
  (medianBC.a * t.A.1 + medianBC.b * t.A.2 + medianBC.c = 0 ∧
   medianBC.a * ((t.B.1 + t.C.1) / 2) + medianBC.b * ((t.B.2 + t.C.2) / 2) + medianBC.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l2892_289243


namespace NUMINAMATH_CALUDE_max_strong_boys_is_ten_l2892_289238

/-- A type representing a boy with height and weight -/
structure Boy where
  height : ℕ
  weight : ℕ

/-- A group of 10 boys -/
def Boys := Fin 10 → Boy

/-- Predicate to check if one boy is not inferior to another -/
def not_inferior (a b : Boy) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Predicate to check if a boy is strong (not inferior to any other boy) -/
def is_strong (boys : Boys) (i : Fin 10) : Prop :=
  ∀ j : Fin 10, j ≠ i → not_inferior (boys i) (boys j)

/-- Theorem stating that it's possible to have 10 strong boys -/
theorem max_strong_boys_is_ten :
  ∃ (boys : Boys), (∀ i j : Fin 10, i ≠ j → boys i ≠ boys j) ∧
                   (∀ i : Fin 10, is_strong boys i) := by
  sorry

end NUMINAMATH_CALUDE_max_strong_boys_is_ten_l2892_289238


namespace NUMINAMATH_CALUDE_equal_distances_l2892_289280

def circular_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (b - a + n) % n

theorem equal_distances : ∃ n : ℕ, 
  n > 0 ∧ 
  circular_distance n 31 7 = circular_distance n 31 14 ∧ 
  n = 41 := by
  sorry

end NUMINAMATH_CALUDE_equal_distances_l2892_289280


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_l2892_289222

theorem probability_at_least_one_girl (total_students : Nat) (boys : Nat) (girls : Nat) 
  (selected : Nat) (h1 : total_students = boys + girls) (h2 : total_students = 5) 
  (h3 : boys = 3) (h4 : girls = 2) (h5 : selected = 3) : 
  (Nat.choose total_students selected - Nat.choose boys selected) / 
  Nat.choose total_students selected = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_l2892_289222


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l2892_289241

def is_solution (x y z : ℕ) : Prop :=
  (x + y) / z = (Nat.factorial x + Nat.factorial y) / Nat.factorial z

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, is_solution x y z →
    ((x = 1 ∧ y = 1 ∧ z = 2) ∨
     (x = 2 ∧ y = 2 ∧ z = 1) ∨
     (x = y ∧ y = z ∧ z ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l2892_289241


namespace NUMINAMATH_CALUDE_halfway_fraction_l2892_289257

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 6) :
  (a : ℚ) / b + ((c : ℚ) / d - (a : ℚ) / b) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l2892_289257


namespace NUMINAMATH_CALUDE_candies_left_after_event_l2892_289247

/-- Calculates the number of candies left after a carousel event --/
theorem candies_left_after_event (
  num_clowns : ℕ
  ) (num_children : ℕ
  ) (initial_supply : ℕ
  ) (candies_per_clown : ℕ
  ) (candies_per_child : ℕ
  ) (candies_as_prizes : ℕ
  ) (h1 : num_clowns = 4
  ) (h2 : num_children = 30
  ) (h3 : initial_supply = 1200
  ) (h4 : candies_per_clown = 10
  ) (h5 : candies_per_child = 15
  ) (h6 : candies_as_prizes = 100
  ) : initial_supply - (num_clowns * candies_per_clown + num_children * candies_per_child + candies_as_prizes) = 610 := by
  sorry

end NUMINAMATH_CALUDE_candies_left_after_event_l2892_289247


namespace NUMINAMATH_CALUDE_dans_age_proof_l2892_289262

/-- Dan's present age in years -/
def dans_present_age : ℕ := 16

/-- Theorem stating that Dan's present age satisfies the given condition -/
theorem dans_age_proof :
  dans_present_age + 16 = 4 * (dans_present_age - 8) :=
by sorry

end NUMINAMATH_CALUDE_dans_age_proof_l2892_289262


namespace NUMINAMATH_CALUDE_second_year_interest_rate_l2892_289226

/-- Proves that given an initial investment of $15,000 with a 10% simple annual interest rate
    for the first year, and a final amount of $17,325 after two years, the interest rate of
    the second year's investment is 5%. -/
theorem second_year_interest_rate
  (initial_investment : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_year_rate = 0.1)
  (h3 : final_amount = 17325)
  : ∃ (second_year_rate : ℝ),
    final_amount = initial_investment * (1 + first_year_rate) * (1 + second_year_rate) ∧
    second_year_rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_second_year_interest_rate_l2892_289226


namespace NUMINAMATH_CALUDE_omega_range_for_four_zeros_l2892_289254

/-- Given a function f(x) = cos(ωx) - 1 with ω > 0, if f has exactly 4 zeros 
    in the interval [0, 2π], then 3 ≤ ω < 4. -/
theorem omega_range_for_four_zeros (ω : ℝ) (h_pos : ω > 0) : 
  (∃! (s : Finset ℝ), s.card = 4 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.cos (ω * x) = 1) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.cos (ω * x) = 1 → x ∈ s)) →
  3 ≤ ω ∧ ω < 4 := by
sorry

end NUMINAMATH_CALUDE_omega_range_for_four_zeros_l2892_289254


namespace NUMINAMATH_CALUDE_opposite_sides_iff_a_range_l2892_289212

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in the 2D plane of the form ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Check if a point is on one side of a line -/
def onOneSide (p : Point) (l : Line) : ℝ :=
  l.a * p.x + l.b * p.y + l.c

/-- The main theorem -/
theorem opposite_sides_iff_a_range (a : ℝ) :
  let P : Point := ⟨1, a⟩
  let Q : Point := ⟨a, -2⟩
  let l : Line := ⟨1, -2, 1⟩
  (onOneSide P l) * (onOneSide Q l) < 0 ↔ a < -5 ∨ a > 1 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_iff_a_range_l2892_289212


namespace NUMINAMATH_CALUDE_function_properties_l2892_289235

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem function_properties :
  (∀ x y, x < y → x < -1 → y < -1 → f x < f y) ∧
  (∀ x y, x < y → x > 3 → y > 3 → f x < f y) ∧
  (∀ x ∈ interval, f x ≤ 5) ∧
  (∃ x ∈ interval, f x = 5) ∧
  (∀ x ∈ interval, f x ≥ -22) ∧
  (∃ x ∈ interval, f x = -22) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2892_289235


namespace NUMINAMATH_CALUDE_missing_number_is_36_l2892_289299

def known_numbers : List ℕ := [1, 22, 23, 24, 25, 27, 2]

theorem missing_number_is_36 (mean : ℚ) (total_count : ℕ) (h_mean : mean = 20) (h_count : total_count = 8) :
  ∃ x : ℕ, (x :: known_numbers).sum / total_count = mean :=
sorry

end NUMINAMATH_CALUDE_missing_number_is_36_l2892_289299


namespace NUMINAMATH_CALUDE_nested_radical_value_l2892_289268

/-- The value of the infinite nested radical sqrt(16 + sqrt(16 + sqrt(16 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt 16)))))

/-- Theorem stating that the nested radical equals (1 + sqrt(65)) / 2 -/
theorem nested_radical_value : nestedRadical = (1 + Real.sqrt 65) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l2892_289268


namespace NUMINAMATH_CALUDE_distance_from_origin_l2892_289251

theorem distance_from_origin (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l2892_289251


namespace NUMINAMATH_CALUDE_remainder_times_seven_l2892_289240

theorem remainder_times_seven (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 972345 →
  divisor = 145 →
  remainder < divisor →
  remainder * 7 = 840 := by
sorry

end NUMINAMATH_CALUDE_remainder_times_seven_l2892_289240


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2892_289272

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2008)
  (h2 : x + 2008 * Real.cos y = 2007)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2892_289272


namespace NUMINAMATH_CALUDE_z_minus_two_purely_imaginary_l2892_289231

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem z_minus_two_purely_imaginary (z : ℂ) (h : z = 2 - I) : 
  is_purely_imaginary (z - 2) := by
  sorry

end NUMINAMATH_CALUDE_z_minus_two_purely_imaginary_l2892_289231


namespace NUMINAMATH_CALUDE_birth_death_rate_interval_birth_death_rate_problem_l2892_289245

theorem birth_death_rate_interval (birth_rate : ℕ) (death_rate : ℕ) (daily_increase : ℕ) : ℕ :=
  let net_rate := birth_rate - death_rate
  let intervals_per_day := daily_increase / net_rate
  let minutes_per_day := 24 * 60
  minutes_per_day / intervals_per_day

theorem birth_death_rate_problem :
  birth_death_rate_interval 10 2 345600 = 48 := by
  sorry

end NUMINAMATH_CALUDE_birth_death_rate_interval_birth_death_rate_problem_l2892_289245


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2892_289244

theorem sqrt_equation_solution :
  ∀ x : ℝ, (Real.sqrt x + Real.sqrt (x + 3) = 12) → x = 19881 / 576 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2892_289244


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l2892_289234

/-- The number of ways to partition n indistinguishable objects into at most k parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to partition 7 indistinguishable objects into at most 4 parts -/
theorem seven_balls_four_boxes : partition_count 7 4 = 11 := by sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l2892_289234


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l2892_289289

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_one_plus_i : (1 + i)^2 = 2*i :=
sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l2892_289289


namespace NUMINAMATH_CALUDE_physics_class_grade_distribution_l2892_289211

theorem physics_class_grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℚ) : 
  total_students = 40 →
  prob_A = (1/2) * prob_B →
  prob_C = 2 * prob_B →
  prob_D = (3/10) * prob_B →
  prob_A + prob_B + prob_C + prob_D = 1 →
  (prob_B * total_students : ℚ) = 200/19 :=
by sorry

end NUMINAMATH_CALUDE_physics_class_grade_distribution_l2892_289211


namespace NUMINAMATH_CALUDE_expression_evaluation_l2892_289295

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 4
  5 * x^y + 2 * y^x = 533 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2892_289295


namespace NUMINAMATH_CALUDE_max_length_special_arithmetic_progression_l2892_289256

/-- An arithmetic progression of natural numbers with common difference 2 -/
def ArithmeticProgression (a₁ : ℕ) (n : ℕ) : Fin n → ℕ :=
  λ i => a₁ + 2 * i.val

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The maximum length of the special arithmetic progression -/
def MaxLength : ℕ := 3

theorem max_length_special_arithmetic_progression :
  ∀ a₁ n : ℕ,
    (∀ k : Fin n, IsPrime ((ArithmeticProgression a₁ n k)^2 + 1)) →
    n ≤ MaxLength :=
by sorry

end NUMINAMATH_CALUDE_max_length_special_arithmetic_progression_l2892_289256


namespace NUMINAMATH_CALUDE_sales_volume_function_correct_profit_at_95_yuan_max_profit_at_110_yuan_max_profit_value_l2892_289204

/-- Represents the weekly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1500

/-- Represents the weekly profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 80) * (sales_volume x)

/-- The cost price of each shirt -/
def cost_price : ℝ := 80

/-- The minimum allowed selling price -/
def min_price : ℝ := 90

/-- The maximum allowed selling price -/
def max_price : ℝ := 110

theorem sales_volume_function_correct :
  ∀ x, sales_volume x = -10 * x + 1500 := by sorry

theorem profit_at_95_yuan :
  profit 95 = 8250 := by sorry

theorem max_profit_at_110_yuan :
  ∀ x, min_price ≤ x ∧ x ≤ max_price → profit x ≤ profit 110 := by sorry

theorem max_profit_value :
  profit 110 = 12000 := by sorry

end NUMINAMATH_CALUDE_sales_volume_function_correct_profit_at_95_yuan_max_profit_at_110_yuan_max_profit_value_l2892_289204
