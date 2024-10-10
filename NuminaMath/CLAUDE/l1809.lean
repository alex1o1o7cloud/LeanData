import Mathlib

namespace sqrt_equation_solution_l1809_180972

theorem sqrt_equation_solution : ∃ (x : ℝ), x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end sqrt_equation_solution_l1809_180972


namespace triangle_problem_l1809_180986

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) :
  (2 * t.a + t.c) * Real.cos t.B + t.b * Real.cos t.C = 0 →
  (1 / 2) * t.a * t.c * Real.sin t.B = 15 * Real.sqrt 3 →
  t.a + t.b + t.c = 30 →
  Real.sin (2 * t.B) / (Real.sin t.A + Real.sin t.C) = -7 / 8 := by
  sorry

end triangle_problem_l1809_180986


namespace f_properties_l1809_180992

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x - 3)

theorem f_properties :
  (∀ y ∈ Set.range f, 0 < y ∧ y ≤ 81) ∧
  (∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ < x₂ → f x₂ < f x₁) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ < f x₂) := by
  sorry

end f_properties_l1809_180992


namespace parabola_y_axis_intersection_l1809_180968

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- Theorem: The intersection point of the parabola y = x^2 - 3x + 2 with the y-axis is (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃ (y : ℝ), f 0 = y ∧ y = 2 :=
sorry

end parabola_y_axis_intersection_l1809_180968


namespace lines_relationship_l1809_180957

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate for non-coplanar lines -/
def nonCoplanar (l₁ l₂ : Line3D) : Prop := sorry

/-- Predicate for intersecting lines -/
def intersects (l₁ l₂ : Line3D) : Prop := sorry

/-- Theorem: Given non-coplanar lines l₁ and l₂, and lines m₁ and m₂ that both intersect with l₁ and l₂,
    the positional relationship between m₁ and m₂ is either intersecting or non-coplanar -/
theorem lines_relationship (l₁ l₂ m₁ m₂ : Line3D)
  (h₁ : nonCoplanar l₁ l₂)
  (h₂ : intersects m₁ l₁)
  (h₃ : intersects m₁ l₂)
  (h₄ : intersects m₂ l₁)
  (h₅ : intersects m₂ l₂) :
  intersects m₁ m₂ ∨ nonCoplanar m₁ m₂ := by
  sorry

end lines_relationship_l1809_180957


namespace probability_two_red_balls_l1809_180931

/-- The probability of picking 2 red balls from a bag with 3 red, 2 blue, and 3 green balls -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 3 →
  blue_balls = 2 →
  green_balls = 3 →
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 3 / 28 :=
by sorry

end probability_two_red_balls_l1809_180931


namespace club_size_after_four_years_l1809_180971

def club_growth (b : ℕ → ℕ) : Prop :=
  b 0 = 20 ∧ ∀ k, b (k + 1) = 4 * b k - 12

theorem club_size_after_four_years (b : ℕ → ℕ) (h : club_growth b) : b 4 = 4100 := by
  sorry

end club_size_after_four_years_l1809_180971


namespace equilateral_hyperbola_through_point_l1809_180920

/-- An equilateral hyperbola is a hyperbola with perpendicular asymptotes -/
def is_equilateral_hyperbola (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x y => a * (x^2 - y^2) + b * x + c * y + 1

/-- A point (x, y) lies on a curve defined by function f if f x y = 0 -/
def point_on_curve (f : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  f x y = 0

/-- A curve is symmetric about the x-axis if for every point (x, y) on the curve,
    the point (x, -y) is also on the curve -/
def symmetric_about_x_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x y = 0 → f x (-y) = 0

/-- A curve is symmetric about the y-axis if for every point (x, y) on the curve,
    the point (-x, y) is also on the curve -/
def symmetric_about_y_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x y = 0 → f (-x) y = 0

theorem equilateral_hyperbola_through_point :
  ∃ f : ℝ → ℝ → ℝ,
    is_equilateral_hyperbola f ∧
    point_on_curve f 3 (-1) ∧
    symmetric_about_x_axis f ∧
    symmetric_about_y_axis f ∧
    f = λ x y => x^2 - y^2 - 8 := by sorry

end equilateral_hyperbola_through_point_l1809_180920


namespace sufficient_not_necessary_l1809_180954

-- Define the condition for a line to be tangent to the circle
def is_tangent (k : ℝ) : Prop :=
  1 + k^2 = 4

-- Define the main theorem
theorem sufficient_not_necessary :
  (∀ k, k = Real.sqrt 3 → is_tangent k) ∧
  (∃ k, k ≠ Real.sqrt 3 ∧ is_tangent k) :=
by sorry

end sufficient_not_necessary_l1809_180954


namespace clock_angle_at_3_37_clock_angle_proof_l1809_180937

/-- The acute angle between clock hands at 3:37 -/
theorem clock_angle_at_3_37 : ℝ :=
  let hours : ℕ := 3
  let minutes : ℕ := 37
  let total_hours : ℕ := 12
  let degrees_per_hour : ℝ := 30

  let minute_angle : ℝ := (minutes : ℝ) / 60 * 360
  let hour_angle : ℝ := (hours : ℝ) * degrees_per_hour + (minutes : ℝ) / 60 * degrees_per_hour

  let angle_diff : ℝ := |minute_angle - hour_angle|
  let acute_angle : ℝ := min angle_diff (360 - angle_diff)

  113.5

/-- Proof of the clock angle theorem -/
theorem clock_angle_proof : clock_angle_at_3_37 = 113.5 := by
  sorry

end clock_angle_at_3_37_clock_angle_proof_l1809_180937


namespace stating_trail_mix_theorem_l1809_180985

/-- Represents the number of bags of nuts -/
def nuts : ℕ := 16

/-- Represents the number of portions that can be made -/
def portions : ℕ := 2

/-- Represents the number of bags of dried fruit -/
def dried_fruit : ℕ := 2

/-- 
Theorem stating that given 16 bags of nuts and the constraint that the maximum 
number of equal portions is 2 with no bags left over, the number of bags of 
dried fruit must be 2.
-/
theorem trail_mix_theorem : 
  (nuts + dried_fruit) % portions = 0 ∧ 
  ∀ n : ℕ, n > dried_fruit → (nuts + n) % portions ≠ 0 :=
sorry

end stating_trail_mix_theorem_l1809_180985


namespace davids_biology_marks_l1809_180984

/-- Given David's marks in four subjects and his average marks, calculate his marks in Biology. -/
theorem davids_biology_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 98)
  (h3 : physics_marks = 99)
  (h4 : chemistry_marks = 100)
  (h5 : average_marks = 98.2)
  (h6 : average_marks = (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / 5) :
  biology_marks = 98 := by
  sorry

#check davids_biology_marks

end davids_biology_marks_l1809_180984


namespace basketball_score_proof_l1809_180939

theorem basketball_score_proof :
  ∀ (S : ℕ) (x : ℕ),
    S > 0 →
    S % 4 = 0 →
    S % 7 = 0 →
    S / 4 + 2 * S / 7 + 15 + x = S →
    x ≤ 14 →
    x = 11 :=
by
  sorry

end basketball_score_proof_l1809_180939


namespace melissa_points_per_game_l1809_180902

theorem melissa_points_per_game 
  (total_points : ℕ) 
  (num_games : ℕ) 
  (points_per_game : ℕ) 
  (h1 : total_points = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = num_games * points_per_game) : 
  points_per_game = 7 := by
  sorry

end melissa_points_per_game_l1809_180902


namespace triangle_perimeter_bound_l1809_180953

theorem triangle_perimeter_bound : 
  ∀ s : ℝ, 
  s > 0 → 
  s + 6 > 21 → 
  s + 21 > 6 → 
  6 + 21 > s → 
  54 > 6 + 21 + s ∧ 
  ∀ n : ℕ, n < 54 → ∃ t : ℝ, t > 0 ∧ t + 6 > 21 ∧ t + 21 > 6 ∧ 6 + 21 > t ∧ n ≤ 6 + 21 + t :=
by sorry

#check triangle_perimeter_bound

end triangle_perimeter_bound_l1809_180953


namespace magazine_cost_l1809_180981

theorem magazine_cost (m : ℝ) 
  (h1 : 8 * m < 12) 
  (h2 : 11 * m > 16.5) : 
  m = 1.5 := by
sorry

end magazine_cost_l1809_180981


namespace partial_fraction_decomposition_l1809_180906

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ), A = -1/2 ∧ B = 5/2 ∧ C = -5 ∧
  ∀ (x : ℚ), x ≠ 0 → x^2 ≠ 2 →
  (2*x^2 - 5*x + 1) / (x^3 - 2*x) = A / x + (B*x + C) / (x^2 - 2) := by
  sorry

end partial_fraction_decomposition_l1809_180906


namespace time_walking_away_l1809_180949

-- Define the walking speed in miles per hour
def walking_speed : ℝ := 2

-- Define the total distance walked in miles
def total_distance : ℝ := 12

-- Define the theorem
theorem time_walking_away : 
  (total_distance / 2) / walking_speed = 3 := by
  sorry

end time_walking_away_l1809_180949


namespace largest_n_satisfying_inequality_l1809_180987

theorem largest_n_satisfying_inequality : 
  (∀ n : ℕ, n ≤ 7 → (1 : ℚ) / 4 + n / 6 < 3 / 2) ∧ 
  (∀ n : ℕ, n > 7 → (1 : ℚ) / 4 + n / 6 ≥ 3 / 2) := by
  sorry

end largest_n_satisfying_inequality_l1809_180987


namespace time_capsule_depth_relation_l1809_180945

/-- Represents the relationship between the depths of time capsules buried by Southton and Northton -/
theorem time_capsule_depth_relation (x y z : ℝ) : 
  (y = 4 * x + z) ↔ (y - 4 * x = z) :=
by sorry

end time_capsule_depth_relation_l1809_180945


namespace range_of_f_l1809_180907

def f (x : ℤ) : ℤ := x + 1

def domain : Set ℤ := {-1, 1, 2}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {0, 2, 3} := by sorry

end range_of_f_l1809_180907


namespace triangle_ratio_l1809_180933

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  A = π / 3 ∧
  a = Real.sqrt 3 →
  (a + b) / (Real.sin A + Real.sin B) = 2 :=
by sorry

end triangle_ratio_l1809_180933


namespace inequality_pattern_l1809_180928

theorem inequality_pattern (x : ℝ) (n : ℕ) (h : x > 0) : 
  x + (n^n : ℝ) / x^n ≥ (n : ℝ) + 1 := by
  sorry

end inequality_pattern_l1809_180928


namespace multiply_inverse_square_equals_cube_l1809_180923

theorem multiply_inverse_square_equals_cube (x : ℝ) : x * (1/7)^2 = 7^3 ↔ x = 16807 := by
  sorry

end multiply_inverse_square_equals_cube_l1809_180923


namespace inequality_solution_l1809_180926

theorem inequality_solution (x : ℝ) :
  (3 - x) / (5 + 2*x) ≤ 0 ↔ x < -5/2 ∨ x ≥ 3 :=
sorry

end inequality_solution_l1809_180926


namespace quadratic_expression_equality_l1809_180929

theorem quadratic_expression_equality (a b : ℝ) : 
  ((-11 * -8)^(3/2) + 5 * Real.sqrt 16) * ((a - 2) + (b + 3)) = 
  ((176 * Real.sqrt 22) + 20) * (a + b + 1) := by
  sorry

end quadratic_expression_equality_l1809_180929


namespace inequality_solution_set_l1809_180979

theorem inequality_solution_set (x : ℝ) : 
  (x - 1) * (x - 2) * (x - 3)^2 > 0 ↔ x < 1 ∨ 1 < x ∧ x < 2 ∨ 2 < x ∧ x < 3 ∨ x > 3 := by
  sorry

end inequality_solution_set_l1809_180979


namespace count_good_numbers_formula_l1809_180989

/-- A number is considered "good" if it contains an even number (including zero) of the digit 8 -/
def is_good (x : ℕ) : Prop := sorry

/-- The count of "good numbers" with length not exceeding n -/
def count_good_numbers (n : ℕ) : ℕ := sorry

/-- The main theorem: The count of "good numbers" with length not exceeding n 
    is equal to (8^n + 10^n) / 2 - 1 -/
theorem count_good_numbers_formula (n : ℕ) (h : n > 0) : 
  count_good_numbers n = (8^n + 10^n) / 2 - 1 := by sorry

end count_good_numbers_formula_l1809_180989


namespace log_sum_greater_than_exp_l1809_180918

theorem log_sum_greater_than_exp (x : ℝ) (h : x < 0) :
  Real.log 2 + Real.log 5 > Real.exp x := by sorry

end log_sum_greater_than_exp_l1809_180918


namespace abc_remainder_mod_seven_l1809_180974

theorem abc_remainder_mod_seven (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 →
  (a + 2*b + 3*c) % 7 = 1 →
  (2*a + 3*b + c) % 7 = 2 →
  (3*a + b + 2*c) % 7 = 1 →
  (a*b*c) % 7 = 0 := by
  sorry

end abc_remainder_mod_seven_l1809_180974


namespace spelling_bee_points_l1809_180942

theorem spelling_bee_points (max_points : ℕ) : max_points = 5 := by
  -- Define Dulce's points
  let dulce_points : ℕ := 3

  -- Define Val's points in terms of Max and Dulce's points
  let val_points : ℕ := 2 * (max_points + dulce_points)

  -- Define the total points of Max's team
  let team_points : ℕ := max_points + dulce_points + val_points

  -- Define the opponents' team points
  let opponents_points : ℕ := 40

  -- Express that Max's team is behind by 16 points
  have team_difference : team_points = opponents_points - 16 := by sorry

  -- Prove that max_points = 5
  sorry

end spelling_bee_points_l1809_180942


namespace complement_of_A_in_U_l1809_180927

-- Define the universal set U as the real numbers
def U := ℝ

-- Define set A as the non-negative real numbers
def A := {x : ℝ | x ≥ 0}

-- State the theorem
theorem complement_of_A_in_U : Set.compl A = {x : ℝ | x < 0} := by
  sorry

end complement_of_A_in_U_l1809_180927


namespace jerry_payment_l1809_180909

/-- Calculates the total payment for Jerry's work --/
theorem jerry_payment (painting_time counter_time_multiplier lawn_mowing_time hourly_rate : ℕ) 
  (h1 : counter_time_multiplier = 3)
  (h2 : painting_time = 8)
  (h3 : lawn_mowing_time = 6)
  (h4 : hourly_rate = 15) :
  (painting_time + counter_time_multiplier * painting_time + lawn_mowing_time) * hourly_rate = 570 :=
by sorry

end jerry_payment_l1809_180909


namespace overall_score_calculation_l1809_180965

/-- Calculate the overall score for a job applicant given their test scores and weights -/
theorem overall_score_calculation
  (written_score : ℝ)
  (interview_score : ℝ)
  (written_weight : ℝ)
  (interview_weight : ℝ)
  (h1 : written_score = 80)
  (h2 : interview_score = 60)
  (h3 : written_weight = 0.6)
  (h4 : interview_weight = 0.4)
  (h5 : written_weight + interview_weight = 1) :
  written_score * written_weight + interview_score * interview_weight = 72 :=
by sorry

end overall_score_calculation_l1809_180965


namespace community_size_after_five_years_l1809_180967

def community_growth (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | m + 1 => 4 * community_growth m - 15

theorem community_size_after_five_years :
  community_growth 5 = 15365 := by
  sorry

end community_size_after_five_years_l1809_180967


namespace absolute_value_trig_expression_l1809_180999

theorem absolute_value_trig_expression : 
  |(-3 : ℝ)| + Real.sqrt 3 * Real.sin (60 * π / 180) - (1 / 2) = 4 := by
sorry

end absolute_value_trig_expression_l1809_180999


namespace rationalize_denominator_sqrt_5_12_l1809_180973

theorem rationalize_denominator_sqrt_5_12 : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end rationalize_denominator_sqrt_5_12_l1809_180973


namespace exists_ten_points_five_kites_l1809_180960

/-- A point on a 4x4 grid --/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A kite formed by four points on the grid --/
structure Kite where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- Check if four points form a valid kite --/
def is_valid_kite (k : Kite) : Prop :=
  -- Two pairs of adjacent sides have equal length
  -- Diagonals intersect at a right angle
  -- One diagonal bisects the other
  sorry

/-- Count the number of kites formed by a set of points --/
def count_kites (points : Finset GridPoint) : Nat :=
  sorry

/-- Theorem stating that there exists an arrangement of 10 points forming exactly 5 kites --/
theorem exists_ten_points_five_kites :
  ∃ (points : Finset GridPoint),
    points.card = 10 ∧ count_kites points = 5 :=
  sorry

end exists_ten_points_five_kites_l1809_180960


namespace min_at_five_l1809_180903

/-- The function to be minimized -/
def f (c : ℝ) : ℝ := (c - 3)^2 + (c - 4)^2 + (c - 8)^2

/-- The theorem stating that 5 minimizes the function f -/
theorem min_at_five : 
  ∀ x : ℝ, f 5 ≤ f x :=
sorry

end min_at_five_l1809_180903


namespace line_slope_l1809_180915

theorem line_slope (A B : ℝ × ℝ) : 
  A.1 = 2 * Real.sqrt 3 ∧ A.2 = -1 ∧ B.1 = Real.sqrt 3 ∧ B.2 = 2 →
  (B.2 - A.2) / (B.1 - A.1) = -Real.sqrt 3 := by
sorry

end line_slope_l1809_180915


namespace m_range_l1809_180914

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, 2 * x - x^2 < m

def q (m : ℝ) : Prop := |m - 1| ≥ 2

-- State the theorem
theorem m_range :
  (∀ m : ℝ, ¬(¬(p m))) ∧ (∀ m : ℝ, ¬(p m ∧ q m)) →
  ∀ m : ℝ, (m ∈ Set.Ioo 1 3) ↔ (p m ∧ ¬(q m)) :=
sorry

end m_range_l1809_180914


namespace coffee_maker_discount_l1809_180958

/-- Calculates the discount amount given the original price and discounted price. -/
def discount_amount (original_price discounted_price : ℝ) : ℝ :=
  original_price - discounted_price

/-- Proves that the discount amount is 20 dollars for a coffee maker with an original price
    of 90 dollars and a discounted price of 70 dollars. -/
theorem coffee_maker_discount : discount_amount 90 70 = 20 := by
  sorry

end coffee_maker_discount_l1809_180958


namespace angela_jacob_insect_ratio_l1809_180901

/-- Proves that the ratio of Angela's insects to Jacob's insects is 1:2 -/
theorem angela_jacob_insect_ratio :
  let dean_insects : ℕ := 30
  let jacob_insects : ℕ := 5 * dean_insects
  let angela_insects : ℕ := 75
  (angela_insects : ℚ) / jacob_insects = 1 / 2 := by
  sorry

end angela_jacob_insect_ratio_l1809_180901


namespace sqrt_three_minus_two_times_sqrt_three_plus_two_l1809_180916

theorem sqrt_three_minus_two_times_sqrt_three_plus_two : (Real.sqrt 3 - 2) * (Real.sqrt 3 + 2) = -1 := by
  sorry

end sqrt_three_minus_two_times_sqrt_three_plus_two_l1809_180916


namespace bethany_current_age_l1809_180943

/-- Bethany's current age -/
def bethany_age : ℕ := sorry

/-- Bethany's sister's current age -/
def sister_age : ℕ := sorry

/-- Bethany's brother's current age -/
def brother_age : ℕ := sorry

/-- Theorem stating Bethany's current age given the conditions -/
theorem bethany_current_age :
  (bethany_age - 3 = 2 * (sister_age - 3)) ∧
  (bethany_age - 3 = brother_age - 3 + 4) ∧
  (sister_age + 5 = 16) ∧
  (brother_age + 5 = 21) →
  bethany_age = 19 := by sorry

end bethany_current_age_l1809_180943


namespace simple_interest_principal_l1809_180983

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  rate = 4.5 →
  time = 4 →
  interest = 144 →
  principal * rate * time / 100 = interest →
  principal = 800 :=
by
  sorry

end simple_interest_principal_l1809_180983


namespace palm_meadows_rooms_l1809_180955

theorem palm_meadows_rooms (two_bed_rooms three_bed_rooms : ℕ) : 
  two_bed_rooms = 8 →
  two_bed_rooms * 2 + three_bed_rooms * 3 = 31 →
  two_bed_rooms + three_bed_rooms = 13 := by
sorry

end palm_meadows_rooms_l1809_180955


namespace emir_savings_correct_l1809_180966

/-- The amount Emir has saved from his allowance -/
def emirSavings (dictionaryCost cookbookCost dinosaurBookCost additionalNeeded : ℕ) : ℕ :=
  dictionaryCost + cookbookCost + dinosaurBookCost - additionalNeeded

theorem emir_savings_correct (dictionaryCost cookbookCost dinosaurBookCost additionalNeeded : ℕ) :
  emirSavings dictionaryCost cookbookCost dinosaurBookCost additionalNeeded =
  dictionaryCost + cookbookCost + dinosaurBookCost - additionalNeeded :=
by sorry

end emir_savings_correct_l1809_180966


namespace simplify_expression_l1809_180919

theorem simplify_expression : (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 + Real.sqrt 3)) = 1 - Real.sqrt 3 / 2 := by
  sorry

end simplify_expression_l1809_180919


namespace initial_books_eq_sold_plus_left_l1809_180917

/-- The number of books Paul had initially -/
def initial_books : ℕ := 136

/-- The number of books Paul sold -/
def books_sold : ℕ := 109

/-- The number of books Paul was left with after the sale -/
def books_left : ℕ := 27

/-- Theorem stating that the initial number of books is equal to the sum of books sold and books left -/
theorem initial_books_eq_sold_plus_left : initial_books = books_sold + books_left := by
  sorry

end initial_books_eq_sold_plus_left_l1809_180917


namespace train_speed_l1809_180988

theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 500) (h2 : crossing_time = 50) :
  train_length / crossing_time = 10 := by
  sorry

end train_speed_l1809_180988


namespace officer_jawan_groups_count_l1809_180995

/-- The number of combinations of n items taken k at a time -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 officers from 7 and 4 jawans from 12 -/
def officer_jawan_groups : ℕ :=
  binomial 7 2 * binomial 12 4

theorem officer_jawan_groups_count :
  officer_jawan_groups = 20790 := by sorry

end officer_jawan_groups_count_l1809_180995


namespace absolute_difference_l1809_180961

theorem absolute_difference (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : 
  |x + 1| - |x - 3| = -4 := by
  sorry

end absolute_difference_l1809_180961


namespace rectangle_sides_and_solvability_l1809_180904

/-- Given a rectangle with perimeter k and area t, this theorem proves the lengths of its sides
    and the condition for solvability. -/
theorem rectangle_sides_and_solvability (k t : ℝ) (k_pos : k > 0) (t_pos : t > 0) :
  let a := (k + Real.sqrt (k^2 - 16*t)) / 4
  let b := (k - Real.sqrt (k^2 - 16*t)) / 4
  (k^2 ≥ 16*t) →
  (a + b = k/2 ∧ a * b = t ∧ a > 0 ∧ b > 0) :=
by sorry

end rectangle_sides_and_solvability_l1809_180904


namespace parabola_intersection_through_focus_l1809_180921

/-- The parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  m : ℝ
  b : ℝ

/-- Theorem statement -/
theorem parabola_intersection_through_focus 
  (para : Parabola) 
  (l : Line)
  (A B : Point)
  (N : ℝ) -- x-coordinate of N
  (h_not_perpendicular : l.m ≠ 0)
  (h_intersect : A.y^2 = 2*para.p*A.x ∧ B.y^2 = 2*para.p*B.x)
  (h_on_line : A.y = l.m * A.x + l.b ∧ B.y = l.m * B.x + l.b)
  (h_different_quadrants : A.y * B.y < 0)
  (h_bisect : abs ((A.y / (A.x - N)) + (B.y / (B.x - N))) = abs (A.y / (A.x - N) - B.y / (B.x - N))) :
  ∃ (t : ℝ), l.m * (para.p / 2) + l.b = 0 :=
sorry

end parabola_intersection_through_focus_l1809_180921


namespace incorrect_representation_l1809_180978

/-- Represents a repeating decimal -/
structure RepeatingDecimal where
  nonRepeating : ℕ → ℕ  -- P: mapping from position to digit
  repeating : ℕ → ℕ     -- Q: mapping from position to digit
  r : ℕ                 -- length of non-repeating part
  s : ℕ                 -- length of repeating part

/-- The decimal representation of a RepeatingDecimal -/
def decimalRepresentation (d : RepeatingDecimal) : ℚ :=
  sorry

/-- Theorem stating that the given representation is incorrect -/
theorem incorrect_representation (d : RepeatingDecimal) :
  ∃ (P Q : ℕ), 
    (10^d.r * 10^(2*d.s) * decimalRepresentation d ≠ 
     (P * 100 + Q * 10 + Q : ℚ) + decimalRepresentation d) :=
  sorry

end incorrect_representation_l1809_180978


namespace daily_food_cost_l1809_180913

theorem daily_food_cost (purchase_price : ℕ) (vaccination_cost : ℕ) (selling_price : ℕ) (num_days : ℕ) (profit : ℕ) :
  purchase_price = 600 →
  vaccination_cost = 500 →
  selling_price = 2500 →
  num_days = 40 →
  profit = 600 →
  (selling_price - (purchase_price + vaccination_cost) - profit) / num_days = 20 := by
  sorry

end daily_food_cost_l1809_180913


namespace inequality_proof_l1809_180930

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  1/(1+a*b) + 1/(1+b*c) + 1/(1+c*a) ≥ 3/2 := by
sorry

end inequality_proof_l1809_180930


namespace c_range_l1809_180952

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → x + 1/x > 1/c

def range_c (c : ℝ) : Prop := (0 < c ∧ c ≤ 1/2) ∨ c ≥ 1

theorem c_range (c : ℝ) (h1 : c > 0) (h2 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) : range_c c := by
  sorry

end c_range_l1809_180952


namespace square_with_tens_digit_seven_l1809_180922

/-- Given a number A with more than one digit, if the tens digit of A^2 is 7, 
    then the units digit of A^2 is 6. -/
theorem square_with_tens_digit_seven (A : ℕ) : 
  A > 9 → 
  (A^2 / 10) % 10 = 7 → 
  A^2 % 10 = 6 :=
by sorry

end square_with_tens_digit_seven_l1809_180922


namespace power_of_power_three_l1809_180994

theorem power_of_power_three : (3^4)^2 = 6561 := by
  sorry

end power_of_power_three_l1809_180994


namespace cube_vector_sum_divisible_by_11_l1809_180964

/-- The size of the cube. -/
def cubeSize : ℕ := 1000

/-- The sum of squares of integers from 0 to n. -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The sum of squares of lengths of vectors from origin to all integer points in the cube. -/
def sumOfVectorLengthSquares : ℕ :=
  3 * (cubeSize + 1)^2 * sumOfSquares cubeSize

theorem cube_vector_sum_divisible_by_11 :
  sumOfVectorLengthSquares % 11 = 0 := by
  sorry

end cube_vector_sum_divisible_by_11_l1809_180964


namespace factor_polynomial_l1809_180951

theorem factor_polynomial (x : ℝ) : 90 * x^3 - 135 * x^9 = 45 * x^3 * (2 - 3 * x^6) := by
  sorry

end factor_polynomial_l1809_180951


namespace sum_of_odd_powers_zero_l1809_180982

theorem sum_of_odd_powers_zero (a b : ℝ) (n : ℕ) (h1 : a + b = 0) (h2 : a ≠ 0) :
  a^(2*n+1) + b^(2*n+1) = 0 := by
sorry

end sum_of_odd_powers_zero_l1809_180982


namespace autumn_pencil_count_l1809_180976

/-- Calculates the final number of pencils Autumn has -/
def final_pencil_count (initial : ℕ) (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - (misplaced + broken) + (found + bought)

/-- Theorem stating that Autumn's final pencil count is correct -/
theorem autumn_pencil_count :
  final_pencil_count 20 7 3 4 2 = 16 := by
  sorry

end autumn_pencil_count_l1809_180976


namespace function_properties_l1809_180935

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem function_properties (a : ℝ) (h : a > 1) :
  (∀ x, x ∈ Set.Icc 1 a → f a x ∈ Set.Icc 1 a) ∧
  (∀ x, x ∈ Set.Icc 1 a → f a x = x) →
  a = 2 ∧
  (∀ x ≤ 2, ∀ y ≤ x, f a x ≤ f a y) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 (a+1) → x₂ ∈ Set.Icc 1 (a+1) → |f a x₁ - f a x₂| ≤ 4) →
  2 ≤ a ∧ a ≤ 3 ∧
  (∃ x ∈ Set.Icc 1 3, f a x = 0) →
  Real.sqrt 5 ≤ a ∧ a ≤ 3 :=
by sorry

end function_properties_l1809_180935


namespace inequality_holds_for_all_x_l1809_180975

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, x^2 - x - m + 1 > 0) ↔ m < 3/4 := by sorry

end inequality_holds_for_all_x_l1809_180975


namespace sum_digits_of_numeric_hex_count_l1809_180962

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Represents a hexadecimal number --/
def Hexadecimal := List HexDigit

/-- Converts a natural number to hexadecimal --/
def toHex (n : ℕ) : Hexadecimal :=
  sorry

/-- Checks if a hexadecimal number uses only numeric digits --/
def usesOnlyNumericDigits (h : Hexadecimal) : Bool :=
  sorry

/-- Counts numbers representable in hexadecimal using only numeric digits --/
def countNumericHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem sum_digits_of_numeric_hex_count :
  sumDigits (countNumericHex 2000) = 25 := by
  sorry

end sum_digits_of_numeric_hex_count_l1809_180962


namespace inequality_holds_for_nonzero_reals_l1809_180932

theorem inequality_holds_for_nonzero_reals (x : ℝ) (h : x ≠ 0) :
  (x^3 - 2*x^5 + x^6) / (x - 2*x^2 + x^4) ≥ -1 := by
  sorry

end inequality_holds_for_nonzero_reals_l1809_180932


namespace pencil_distribution_l1809_180950

theorem pencil_distribution (total_pens : Nat) (total_pencils : Nat) (max_students : Nat) :
  total_pens = 1001 →
  total_pencils = 910 →
  max_students = 91 →
  (∃ (students : Nat), students ≤ max_students ∧ 
    total_pens % students = 0 ∧ 
    total_pencils % students = 0) →
  total_pencils / max_students = 10 := by
  sorry

end pencil_distribution_l1809_180950


namespace power_of_product_rule_l1809_180977

theorem power_of_product_rule (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 := by
  sorry

end power_of_product_rule_l1809_180977


namespace circle_tangent_properties_l1809_180940

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 3 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent lines PA and PB
def tangent_PA (xa ya xp yp : ℝ) : Prop := 
  circle_M xa ya ∧ point_P xp yp ∧ (xa - xp) * (xa + 1) + (ya - yp) * ya = 0

def tangent_PB (xb yb xp yp : ℝ) : Prop := 
  circle_M xb yb ∧ point_P xp yp ∧ (xb - xp) * (xb + 1) + (yb - yp) * yb = 0

-- Theorem statement
theorem circle_tangent_properties :
  ∃ (min_area : ℝ) (chord_length : ℝ) (fixed_point : ℝ × ℝ),
    (min_area = 2 * Real.sqrt 3) ∧
    (chord_length = Real.sqrt 6) ∧
    (fixed_point = (-1/2, -1/2)) ∧
    (∀ xa ya xb yb xp yp : ℝ,
      tangent_PA xa ya xp yp →
      tangent_PB xb yb xp yp →
      -- 1. Minimum area of quadrilateral PAMB
      (xa - xp)^2 + (ya - yp)^2 + (xb - xp)^2 + (yb - yp)^2 ≥ min_area^2 ∧
      -- 2. Length of chord AB when |PA| is shortest
      ((xa - xp)^2 + (ya - yp)^2 = (xb - xp)^2 + (yb - yp)^2 →
        (xa - xb)^2 + (ya - yb)^2 = chord_length^2) ∧
      -- 3. Line AB passes through the fixed point
      (ya - yb) * (fixed_point.1 - xa) = (xa - xb) * (fixed_point.2 - ya)) :=
sorry

end circle_tangent_properties_l1809_180940


namespace matching_pair_probability_l1809_180996

def black_socks : ℕ := 12
def blue_socks : ℕ := 10

def total_socks : ℕ := black_socks + blue_socks

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

def total_ways : ℕ := choose total_socks 2
def black_matching_ways : ℕ := choose black_socks 2
def blue_matching_ways : ℕ := choose blue_socks 2
def matching_ways : ℕ := black_matching_ways + blue_matching_ways

theorem matching_pair_probability :
  (matching_ways : ℚ) / total_ways = 111 / 231 :=
sorry

end matching_pair_probability_l1809_180996


namespace extraneous_root_condition_l1809_180948

/-- The equation has an extraneous root when m = -4 -/
theorem extraneous_root_condition (m : ℝ) : 
  (m = -4) → 
  (∃ (x : ℝ), x ≠ 2 ∧ 
    (m / (x - 2) - (2 * x) / (2 - x) = 1) ∧
    (m / (2 - 2) - (2 * 2) / (2 - 2) ≠ 1)) :=
by sorry


end extraneous_root_condition_l1809_180948


namespace mitch_family_milk_consumption_l1809_180934

/-- The total milk consumption in cartons for Mitch's family in one week -/
def total_milk_consumption (regular_milk soy_milk : ℝ) : ℝ :=
  regular_milk + soy_milk

/-- Proof that Mitch's family's total milk consumption is 0.6 cartons in one week -/
theorem mitch_family_milk_consumption :
  let regular_milk : ℝ := 0.5
  let soy_milk : ℝ := 0.1
  total_milk_consumption regular_milk soy_milk = 0.6 := by
  sorry

end mitch_family_milk_consumption_l1809_180934


namespace edward_lives_left_l1809_180998

theorem edward_lives_left (initial_lives : ℕ) (lives_lost : ℕ) : 
  initial_lives = 15 → lives_lost = 8 → initial_lives - lives_lost = 7 := by
  sorry

end edward_lives_left_l1809_180998


namespace car_speed_problem_l1809_180905

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 270 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := new_time_factor * original_time
  let new_speed := distance / new_time
  new_speed = 30 := by
sorry

end car_speed_problem_l1809_180905


namespace square_perimeter_sum_l1809_180912

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 45) :
  4 * (x + y) = 4 * (Real.sqrt 65 + 2 * Real.sqrt 5) := by
  sorry

end square_perimeter_sum_l1809_180912


namespace sum_of_fourth_powers_l1809_180959

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 4) : 
  x^4 + y^4 = 8432 := by sorry

end sum_of_fourth_powers_l1809_180959


namespace marias_test_score_l1809_180969

theorem marias_test_score (scores : Fin 4 → ℕ) : 
  scores 0 = 80 →
  scores 2 = 90 →
  scores 3 = 100 →
  (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 85 →
  scores 1 = 70 := by
sorry

end marias_test_score_l1809_180969


namespace quadratic_equation_b_range_l1809_180924

theorem quadratic_equation_b_range :
  ∀ (b c : ℝ),
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 + b*x + c = 0) →
  (0 ≤ 3*b + c) →
  (3*b + c ≤ 3) →
  b ∈ Set.Icc 0 2 :=
by sorry

end quadratic_equation_b_range_l1809_180924


namespace worker_travel_time_l1809_180941

theorem worker_travel_time (normal_speed : ℝ) (slower_speed : ℝ) (usual_time : ℝ) (delay : ℝ) :
  slower_speed = (5 / 6) * normal_speed →
  delay = 12 →
  slower_speed * (usual_time + delay) = normal_speed * usual_time →
  usual_time = 60 := by
sorry

end worker_travel_time_l1809_180941


namespace hole_large_enough_for_person_l1809_180900

/-- Represents a two-dimensional shape --/
structure Shape :=
  (perimeter : ℝ)

/-- Represents a hole cut in a shape --/
structure Hole :=
  (opening_size : ℝ)

/-- Represents a person --/
structure Person :=
  (size : ℝ)

/-- Function to create a hole in a shape --/
def cut_hole (s : Shape) : Hole :=
  sorry

/-- Theorem stating that it's possible to cut a hole in a shape that a person can fit through --/
theorem hole_large_enough_for_person (s : Shape) (p : Person) :
  ∃ (h : Hole), h = cut_hole s ∧ h.opening_size > p.size :=
sorry

end hole_large_enough_for_person_l1809_180900


namespace quadratic_equations_solutions_l1809_180910

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, (3 * x₁^2 = 4 - 2 * x₁ ∧ 3 * x₂^2 = 4 - 2 * x₂) ∧ 
    x₁ = (-1 + Real.sqrt 13) / 3 ∧ x₂ = (-1 - Real.sqrt 13) / 3) ∧
  (∃ y₁ y₂ : ℝ, (y₁ * (y₁ - 7) = 8 * (7 - y₁) ∧ y₂ * (y₂ - 7) = 8 * (7 - y₂)) ∧
    y₁ = 7 ∧ y₂ = -8) :=
by sorry


end quadratic_equations_solutions_l1809_180910


namespace product_equality_l1809_180980

theorem product_equality (a b : ℤ) : 
  (∃ C : ℤ, a * (a - 5) = C ∧ b * (b - 8) = C) → 
  (a * (a - 5) = 0 ∨ a * (a - 5) = 84) :=
by sorry

end product_equality_l1809_180980


namespace monotonic_decreasing_implies_a_leq_neg_seven_l1809_180947

/-- A quadratic function f(x) = x^2 + (a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x + 2

/-- The property of f being monotonically decreasing on (-∞, 4] -/
def monotonically_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → x ≤ 4 → f a x ≥ f a y

/-- Theorem: If f is monotonically decreasing on (-∞, 4], then a ≤ -7 -/
theorem monotonic_decreasing_implies_a_leq_neg_seven (a : ℝ) :
  monotonically_decreasing a → a ≤ -7 := by sorry

end monotonic_decreasing_implies_a_leq_neg_seven_l1809_180947


namespace negation_of_universal_proposition_l1809_180911

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + Real.sin x + 1 < 0) ↔ (∃ x : ℝ, x^2 + Real.sin x + 1 ≥ 0) :=
by sorry

end negation_of_universal_proposition_l1809_180911


namespace geometric_sequence_a5_l1809_180956

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = 6 →
  a 3 + a 5 + a 7 = 78 →
  a 5 = 18 := by
sorry

end geometric_sequence_a5_l1809_180956


namespace trigonometric_identity_l1809_180970

theorem trigonometric_identity (A B C : Real) (h : A + B + C = π) :
  Real.sin A * Real.cos B * Real.cos C + 
  Real.cos A * Real.sin B * Real.cos C + 
  Real.cos A * Real.cos B * Real.sin C = 
  Real.sin A * Real.sin B * Real.sin C := by
  sorry

end trigonometric_identity_l1809_180970


namespace find_d_l1809_180944

theorem find_d : ∃ d : ℝ, 
  (∃ x : ℤ, x = ⌊d⌋ ∧ 3 * x^2 + 19 * x - 84 = 0) ∧ 
  (∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ y = d - ⌊d⌋ ∧ 5 * y^2 - 28 * y + 12 = 0) ∧
  d = 3.2 := by
sorry

end find_d_l1809_180944


namespace quadratic_equation_roots_l1809_180993

theorem quadratic_equation_roots (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + k = 0 ∧ x₂^2 - x₂ + k = 0 :=
by
  sorry

end quadratic_equation_roots_l1809_180993


namespace parallel_iff_m_eq_neg_three_l1809_180908

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Vector a as defined in the problem -/
def a : ℝ × ℝ := (1, -2)

/-- Vector b as defined in the problem -/
def b (m : ℝ) : ℝ × ℝ := (1 + m, 1 - m)

/-- The main theorem: vectors a and b are parallel if and only if m = -3 -/
theorem parallel_iff_m_eq_neg_three :
  ∀ m : ℝ, are_parallel a (b m) ↔ m = -3 := by sorry

end parallel_iff_m_eq_neg_three_l1809_180908


namespace derivative_extrema_l1809_180990

-- Define the function
def f (x : ℝ) := x^4 - 6*x^2 + 1

-- Define the derivative of the function
def f' (x : ℝ) := 4*x^3 - 12*x

-- Define the interval
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem derivative_extrema :
  (∃ x ∈ interval, ∀ y ∈ interval, f' y ≤ f' x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f' y ≥ f' x) ∧
  (∃ x ∈ interval, f' x = 72) ∧
  (∃ x ∈ interval, f' x = -8) :=
sorry

end derivative_extrema_l1809_180990


namespace roses_distribution_l1809_180925

theorem roses_distribution (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) 
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  : (initial_roses - stolen_roses) / people = 4 := by
  sorry

end roses_distribution_l1809_180925


namespace complex_expression_simplification_l1809_180991

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  3 * (4 - 2*i) - 2*i*(3 - 2*i) + (1 + i)*(2 + i) = 9 - 9*i :=
by sorry

end complex_expression_simplification_l1809_180991


namespace pilot_fish_speed_is_30_l1809_180946

/-- Calculates the speed of a pilot fish given initial conditions -/
def pilotFishSpeed (keanuSpeed : ℝ) (sharkSpeedMultiplier : ℝ) (pilotFishIncreaseFactor : ℝ) : ℝ :=
  let sharkSpeedIncrease := keanuSpeed * (sharkSpeedMultiplier - 1)
  keanuSpeed + pilotFishIncreaseFactor * sharkSpeedIncrease

/-- Theorem stating that under given conditions, the pilot fish's speed is 30 mph -/
theorem pilot_fish_speed_is_30 :
  pilotFishSpeed 20 2 (1/2) = 30 := by
  sorry

end pilot_fish_speed_is_30_l1809_180946


namespace remainder_theorem_l1809_180997

theorem remainder_theorem : ∃ q : ℤ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 159 := by
  sorry

end remainder_theorem_l1809_180997


namespace fourth_team_odd_l1809_180938

/-- Calculates the odd for the fourth team in a soccer bet -/
theorem fourth_team_odd (odd1 odd2 odd3 : ℝ) (bet_amount expected_winnings : ℝ) :
  odd1 = 1.28 →
  odd2 = 5.23 →
  odd3 = 3.25 →
  bet_amount = 5.00 →
  expected_winnings = 223.0072 →
  ∃ (odd4 : ℝ), abs (odd4 - 2.061) < 0.001 ∧ 
    odd1 * odd2 * odd3 * odd4 = expected_winnings / bet_amount :=
by
  sorry

#check fourth_team_odd

end fourth_team_odd_l1809_180938


namespace tiling_condition_l1809_180936

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a square -/
inductive Color
  | Black
  | White

/-- Determines the color of a square based on its position -/
def squareColor (s : Square) : Color :=
  if (s.row.val + s.col.val) % 2 = 0 then Color.Black else Color.White

/-- Represents a chessboard with two squares removed -/
structure ChessboardWithRemovedSquares where
  removed1 : Square
  removed2 : Square
  different : removed1 ≠ removed2

/-- Represents the possibility of tiling the chessboard with dominoes -/
def canTile (board : ChessboardWithRemovedSquares) : Prop :=
  squareColor board.removed1 ≠ squareColor board.removed2

/-- Theorem stating the condition for possible tiling -/
theorem tiling_condition (board : ChessboardWithRemovedSquares) :
  canTile board ↔ squareColor board.removed1 ≠ squareColor board.removed2 := by sorry

end tiling_condition_l1809_180936


namespace pyramid_volume_transformation_l1809_180963

theorem pyramid_volume_transformation (s h : ℝ) : 
  (1/3 : ℝ) * s^2 * h = 72 → 
  (1/3 : ℝ) * (3*s)^2 * (2*h) = 1296 := by
sorry

end pyramid_volume_transformation_l1809_180963
