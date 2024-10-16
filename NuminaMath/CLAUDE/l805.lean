import Mathlib

namespace NUMINAMATH_CALUDE_summer_reading_goal_l805_80599

/-- The number of books Carlos read in June -/
def june_books : ℕ := 42

/-- The number of books Carlos read in July -/
def july_books : ℕ := 28

/-- The number of books Carlos read in August -/
def august_books : ℕ := 30

/-- Carlos' goal for the number of books to read during summer vacation -/
def summer_goal : ℕ := june_books + july_books + august_books

theorem summer_reading_goal : summer_goal = 100 := by
  sorry

end NUMINAMATH_CALUDE_summer_reading_goal_l805_80599


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l805_80579

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem parallel_vectors_x_value :
  let p : ℝ × ℝ := (2, -3)
  let q : ℝ × ℝ := (x, 6)
  are_parallel p q → x = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l805_80579


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l805_80567

theorem absolute_value_simplification : |(-6 - 4)| = 6 + 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l805_80567


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l805_80501

theorem technician_round_trip_completion (distance : ℝ) : 
  distance > 0 → 
  (distance + 0.4 * distance) / (2 * distance) = 0.7 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l805_80501


namespace NUMINAMATH_CALUDE_min_value_of_function_l805_80551

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (4 * x^2 + 8 * x + 13) / (6 * (1 + x)) ≥ 2 ∧
  ∃ y > 0, (4 * y^2 + 8 * y + 13) / (6 * (1 + y)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l805_80551


namespace NUMINAMATH_CALUDE_average_math_chem_is_25_l805_80546

/-- Given a student's scores in mathematics, physics, and chemistry,
    prove that the average of mathematics and chemistry scores is 25 -/
theorem average_math_chem_is_25 
  (M P C : ℕ) -- Marks in Mathematics, Physics, and Chemistry
  (h1 : M + P = 30) -- Total marks in mathematics and physics is 30
  (h2 : C = P + 20) -- Chemistry score is 20 more than physics score
  : (M + C) / 2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_average_math_chem_is_25_l805_80546


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l805_80517

/-- Given two vectors a and b in ℝ², where a = (x - 1, 2) and b = (2, 1),
    if a is perpendicular to b, then x = 0. -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l805_80517


namespace NUMINAMATH_CALUDE_window_area_ratio_l805_80552

/-- Proves that for a rectangle with semicircles at either end, where the ratio of the length (AD)
    to the width (AB) is 3:2 and the width is 30 inches, the ratio of the area of the rectangle
    to the combined area of the semicircles is 6:π. -/
theorem window_area_ratio :
  let AB : ℝ := 30
  let AD : ℝ := (3/2) * AB
  let rectangle_area : ℝ := AD * AB
  let semicircle_radius : ℝ := AB / 2
  let semicircles_area : ℝ := π * semicircle_radius^2
  rectangle_area / semicircles_area = 6 / π :=
by sorry

end NUMINAMATH_CALUDE_window_area_ratio_l805_80552


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_3_range_of_a_for_inequality_l805_80506

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Statement for part 1
theorem solution_set_f_greater_than_3 :
  {x : ℝ | f x > 3} = {x : ℝ | x > 0} := by sorry

-- Statement for part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∃ x, f x + 1 ≤ 4^a - 5 * 2^a} = 
  {a : ℝ | a ≤ 0 ∨ a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_3_range_of_a_for_inequality_l805_80506


namespace NUMINAMATH_CALUDE_phoenix_hike_length_l805_80550

/-- Represents the length of Phoenix's hike on the Rocky Path Trail -/
theorem phoenix_hike_length 
  (day1 day2 day3 day4 : ℝ) 
  (first_two_days : day1 + day2 = 22)
  (second_third_avg : (day2 + day3) / 2 = 13)
  (last_two_days : day3 + day4 = 30)
  (first_third_days : day1 + day3 = 26) :
  day1 + day2 + day3 + day4 = 52 :=
by
  sorry


end NUMINAMATH_CALUDE_phoenix_hike_length_l805_80550


namespace NUMINAMATH_CALUDE_no_solution_l805_80557

/-- ab is a two-digit number -/
def ab : ℕ := sorry

/-- ba is a two-digit number, which is the reverse of ab -/
def ba : ℕ := sorry

/-- ab and ba are distinct -/
axiom ab_ne_ba : ab ≠ ba

/-- There is no real number x that satisfies the equation (ab)^x - 2 = (ba)^x - 7 -/
theorem no_solution : ¬∃ x : ℝ, (ab : ℝ) ^ x - 2 = (ba : ℝ) ^ x - 7 := by sorry

end NUMINAMATH_CALUDE_no_solution_l805_80557


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l805_80538

/-- The probability of selecting 3 plates of the same color from a set of 6 red plates and 5 blue plates is 2/11. -/
theorem same_color_plate_probability :
  let total_plates : ℕ := 11
  let red_plates : ℕ := 6
  let blue_plates : ℕ := 5
  let selected_plates : ℕ := 3
  let total_combinations := Nat.choose total_plates selected_plates
  let red_combinations := Nat.choose red_plates selected_plates
  let blue_combinations := Nat.choose blue_plates selected_plates
  let same_color_combinations := red_combinations + blue_combinations
  (same_color_combinations : ℚ) / total_combinations = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l805_80538


namespace NUMINAMATH_CALUDE_circle_distance_theorem_l805_80570

theorem circle_distance_theorem (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → ∃ x1 y1 x2 y2 : ℝ, 
    x1^2 + y1^2 = 1 ∧ 
    x2^2 + y2^2 = 1 ∧ 
    (x1 - a)^2 + (y1 - 1)^2 = 4 ∧ 
    (x2 - a)^2 + (y2 - 1)^2 = 4 ∧ 
    (x1, y1) ≠ (x2, y2)) → 
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_distance_theorem_l805_80570


namespace NUMINAMATH_CALUDE_total_seedlings_sold_l805_80524

/-- Represents the number of seedlings sold for each type -/
structure Seedlings where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Theorem stating the total number of seedlings sold given the conditions -/
theorem total_seedlings_sold (s : Seedlings) : 
  (s.A : ℚ) / s.B = 1 / 2 →
  (s.B : ℚ) / s.C = 3 / 4 →
  3 * s.A + 2 * s.B + s.C = 29000 →
  s.A + s.B + s.C = 17000 := by
  sorry

#check total_seedlings_sold

end NUMINAMATH_CALUDE_total_seedlings_sold_l805_80524


namespace NUMINAMATH_CALUDE_meeting_point_distance_from_top_l805_80508

-- Define the race parameters
def race_length : ℝ := 12
def uphill_length : ℝ := 6
def downhill_length : ℝ := 6

-- Define Jack's parameters
def jack_start_time : ℝ := 0
def jack_uphill_speed : ℝ := 12
def jack_downhill_speed : ℝ := 18

-- Define Jill's parameters
def jill_start_time : ℝ := 0.25  -- 15 minutes = 0.25 hours
def jill_uphill_speed : ℝ := 14
def jill_downhill_speed : ℝ := 19

-- Define the theorem
theorem meeting_point_distance_from_top : 
  ∃ (meeting_time : ℝ) (meeting_distance : ℝ),
    meeting_time > jack_start_time + (uphill_length / jack_uphill_speed) ∧
    meeting_time > jill_start_time ∧
    meeting_time < jill_start_time + (uphill_length / jill_uphill_speed) ∧
    meeting_distance = uphill_length - (meeting_time - jill_start_time) * jill_uphill_speed ∧
    meeting_distance = downhill_length - (meeting_time - (jack_start_time + uphill_length / jack_uphill_speed)) * jack_downhill_speed ∧
    meeting_distance = 699 / 64 := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_distance_from_top_l805_80508


namespace NUMINAMATH_CALUDE_bike_shop_wheels_l805_80553

/-- The number of wheels on all vehicles in a bike shop -/
def total_wheels (num_bicycles num_tricycles : ℕ) : ℕ :=
  2 * num_bicycles + 3 * num_tricycles

/-- Theorem stating that the total number of wheels from 50 bicycles and 20 tricycles is 160 -/
theorem bike_shop_wheels :
  total_wheels 50 20 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bike_shop_wheels_l805_80553


namespace NUMINAMATH_CALUDE_article_sale_loss_percent_l805_80515

/-- Theorem: Given an article with a 35% gain at its original selling price,
    when sold at 2/3 of the original price, the loss percent is 10%. -/
theorem article_sale_loss_percent (cost_price : ℝ) (original_price : ℝ) :
  original_price = cost_price * (1 + 35 / 100) →
  let new_price := (2 / 3) * original_price
  let loss := cost_price - new_price
  let loss_percent := (loss / cost_price) * 100
  loss_percent = 10 := by
sorry

end NUMINAMATH_CALUDE_article_sale_loss_percent_l805_80515


namespace NUMINAMATH_CALUDE_no_intersection_l805_80559

-- Define the functions
def f (x : ℝ) : ℝ := |3 * x + 4|
def g (x : ℝ) : ℝ := -|4 * x - 1|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_l805_80559


namespace NUMINAMATH_CALUDE_prob_xi_equals_three_l805_80539

/-- A random variable following a binomial distribution B(6, 1/2) -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function for ξ -/
def P (k : ℕ) : ℝ := sorry

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Theorem: The probability that ξ equals 3 is 5/16 -/
theorem prob_xi_equals_three : P 3 = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_xi_equals_three_l805_80539


namespace NUMINAMATH_CALUDE_trigonometric_product_upper_bound_l805_80535

theorem trigonometric_product_upper_bound :
  ∀ x y z : ℝ,
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 4.5 ∧
  ∃ x y z : ℝ,
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_product_upper_bound_l805_80535


namespace NUMINAMATH_CALUDE_cat_kittens_count_l805_80531

def animal_shelter_problem (initial_cats : ℕ) (new_cats : ℕ) (adopted_cats : ℕ) (final_cats : ℕ) : ℕ :=
  let total_before_events := initial_cats + new_cats
  let after_adoption := total_before_events - adopted_cats
  final_cats - after_adoption + 1

theorem cat_kittens_count : animal_shelter_problem 6 12 3 19 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cat_kittens_count_l805_80531


namespace NUMINAMATH_CALUDE_pentagonal_gcd_one_l805_80549

theorem pentagonal_gcd_one (n : ℕ+) : 
  let P : ℕ+ → ℕ := fun m => (m * (3 * m - 1)) / 2
  Nat.gcd (5 * P n) (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_gcd_one_l805_80549


namespace NUMINAMATH_CALUDE_airplane_seats_l805_80558

theorem airplane_seats :
  ∀ (total_seats : ℝ),
  (30 : ℝ) + 0.2 * total_seats + 0.75 * total_seats = total_seats →
  total_seats = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l805_80558


namespace NUMINAMATH_CALUDE_greatest_five_digit_sum_l805_80583

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem greatest_five_digit_sum (M : ℕ) :
  is_five_digit M ∧ 
  digit_product M = 180 ∧ 
  (∀ n : ℕ, is_five_digit n ∧ digit_product n = 180 → n ≤ M) →
  digit_sum M = 20 := by
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_sum_l805_80583


namespace NUMINAMATH_CALUDE_solution_set_equality_l805_80545

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := {x : ℝ | x ≠ 0 ∧ 1 / x < 1 / 2}

-- State the theorem
theorem solution_set_equality : solution_set = Set.Ioi 2 ∪ Set.Iio 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l805_80545


namespace NUMINAMATH_CALUDE_open_box_volume_is_5760_l805_80588

/-- Calculate the volume of an open box formed by cutting squares from a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem: The volume of the open box is 5760 m³ -/
theorem open_box_volume_is_5760 :
  openBoxVolume 52 36 8 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_is_5760_l805_80588


namespace NUMINAMATH_CALUDE_hyperbola_equation_l805_80566

/-- The hyperbola equation -/
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x^2 = 4 * Real.sqrt 6 * y

/-- The point A lies on the hyperbola -/
def A_on_hyperbola (a b c m n : ℝ) : Prop := hyperbola a b m n

/-- The point B is on the imaginary axis of the hyperbola -/
def B_on_imaginary_axis (b : ℝ) : Prop := b = Real.sqrt 6

/-- The vector relation between BA and AF -/
def vector_relation (c m n : ℝ) : Prop :=
  m - 0 = 2 * (c - m) ∧ n - Real.sqrt 6 = 2 * (0 - n)

/-- The main theorem -/
theorem hyperbola_equation (a b c m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : hyperbola a b m n)
  (h2 : parabola m n)
  (h3 : A_on_hyperbola a b c m n)
  (h4 : B_on_imaginary_axis b)
  (h5 : vector_relation c m n) :
  a = 2 ∧ b = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l805_80566


namespace NUMINAMATH_CALUDE_computer_multiplications_l805_80541

theorem computer_multiplications (multiplications_per_second : ℕ) (hours : ℕ) : 
  multiplications_per_second = 15000 → 
  hours = 3 → 
  multiplications_per_second * (hours * 3600) = 162000000 := by
  sorry

end NUMINAMATH_CALUDE_computer_multiplications_l805_80541


namespace NUMINAMATH_CALUDE_product_of_sums_l805_80574

/-- The sum of numbers of the form 2k+1 where k ranges from 0 to n -/
def odd_sum (n : ℕ) : ℕ := (n + 1)^2

/-- The sum of the first n even numbers -/
def even_sum (n : ℕ) : ℕ := n * (n + 1)

/-- The product of odd_sum and even_sum is equal to (n+1)^3 * n -/
theorem product_of_sums (n : ℕ) : odd_sum n * even_sum n = (n + 1)^3 * n := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l805_80574


namespace NUMINAMATH_CALUDE_unique_solution_l805_80571

def base_6_value (s h e : ℕ) : ℕ := s * 36 + h * 6 + e

theorem unique_solution :
  ∀ (s h e : ℕ),
    s ≠ 0 ∧ h ≠ 0 ∧ e ≠ 0 →
    s < 6 ∧ h < 6 ∧ e < 6 →
    s ≠ h ∧ s ≠ e ∧ h ≠ e →
    base_6_value s h e + base_6_value 0 h e = base_6_value h e s →
    s = 4 ∧ h = 2 ∧ e = 5 ∧ (s + h + e) % 6 = 5 ∧ ((s + h + e) / 6) % 6 = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l805_80571


namespace NUMINAMATH_CALUDE_cylinder_volume_l805_80591

/-- The volume of a cylinder with specific geometric conditions -/
theorem cylinder_volume (l α β : ℝ) (hl : l > 0) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  ∃ V : ℝ, V = (π * l^3 * Real.sin (2*α) * Real.cos α^3) / (8 * Real.cos (α + β) * Real.cos (α - β)) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l805_80591


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l805_80530

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 - I) / I
  (z.im : ℝ) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l805_80530


namespace NUMINAMATH_CALUDE_student_team_signup_l805_80536

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

end NUMINAMATH_CALUDE_student_team_signup_l805_80536


namespace NUMINAMATH_CALUDE_odd_function_value_l805_80527

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : IsOdd f)
  (h_neg : ∀ x < 0, f x = x^2 + a*x)
  (h_f2 : f 2 = 6) :
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l805_80527


namespace NUMINAMATH_CALUDE_greatest_ABDBA_div_by_11_l805_80500

/-- Represents a five-digit number in the form AB,DBA -/
structure ABDBA where
  a : Nat
  b : Nat
  d : Nat
  h1 : a < 10
  h2 : b < 10
  h3 : d < 10
  h4 : a ≠ b
  h5 : a ≠ d
  h6 : b ≠ d

/-- Converts ABDBA to its numerical value -/
def ABDBA.toNat (n : ABDBA) : Nat :=
  n.a * 10000 + n.b * 1000 + n.d * 100 + n.b * 10 + n.a

/-- Theorem stating the greatest ABDBA number divisible by 11 -/
theorem greatest_ABDBA_div_by_11 :
  ∀ n : ABDBA, n.toNat ≤ 96569 ∧ n.toNat % 11 = 0 →
  ∃ m : ABDBA, m.toNat = 96569 ∧ m.toNat % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_ABDBA_div_by_11_l805_80500


namespace NUMINAMATH_CALUDE_correct_scores_theorem_l805_80578

/-- Represents a class with exam scores -/
structure ExamClass where
  studentCount : Nat
  initialAverage : ℝ
  initialVariance : ℝ
  studentAInitialScore : ℝ
  studentAActualScore : ℝ
  studentBInitialScore : ℝ
  studentBActualScore : ℝ

/-- Calculates the new average and variance after correcting two scores -/
def correctScores (c : ExamClass) : ℝ × ℝ :=
  let newAverage := c.initialAverage
  let newVariance := c.initialVariance - 25
  (newAverage, newVariance)

theorem correct_scores_theorem (c : ExamClass) 
  (h1 : c.studentCount = 48)
  (h2 : c.initialAverage = 70)
  (h3 : c.initialVariance = 75)
  (h4 : c.studentAInitialScore = 50)
  (h5 : c.studentAActualScore = 80)
  (h6 : c.studentBInitialScore = 100)
  (h7 : c.studentBActualScore = 70) :
  correctScores c = (70, 50) := by
  sorry

end NUMINAMATH_CALUDE_correct_scores_theorem_l805_80578


namespace NUMINAMATH_CALUDE_perpendicular_vectors_sum_l805_80511

/-- Given two perpendicular vectors a and b in ℝ², prove their sum is (3, -1) -/
theorem perpendicular_vectors_sum (a b : ℝ × ℝ) :
  a.1 = x ∧ a.2 = 1 ∧ b = (1, -2) ∧ a.1 * b.1 + a.2 * b.2 = 0 →
  a + b = (3, -1) := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_sum_l805_80511


namespace NUMINAMATH_CALUDE_circle_equation_l805_80537

/-- Given a circle with center (2, 1) and a line containing its common chord
    with the circle x^2 + y^2 - 3x = 0 passing through (5, -2),
    prove that the equation of the circle is (x-2)^2 + (y-1)^2 = 4 -/
theorem circle_equation (x y : ℝ) :
  let center := (2, 1)
  let known_circle := fun (x y : ℝ) => x^2 + y^2 - 3*x = 0
  let common_chord_point := (5, -2)
  let circle_eq := fun (x y : ℝ) => (x - 2)^2 + (y - 1)^2 = 4
  (∃ (line : ℝ → ℝ → Prop),
    (∀ x y, line x y ↔ known_circle x y ∨ circle_eq x y) ∧
    line common_chord_point.1 common_chord_point.2) →
  circle_eq x y :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l805_80537


namespace NUMINAMATH_CALUDE_f_of_two_equals_eleven_l805_80521

/-- A function f satisfying the given conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b * x + 3

/-- The theorem stating that f(2) = 11 under the given conditions -/
theorem f_of_two_equals_eleven (a b : ℝ) 
  (h1 : f a b 1 = 7) 
  (h3 : f a b 3 = 15) : 
  f a b 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_eleven_l805_80521


namespace NUMINAMATH_CALUDE_latest_departure_time_l805_80585

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Subtracts minutes from a given time -/
def subtractMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes - m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem latest_departure_time 
  (performance_time : Time) 
  (travel_time : Nat) 
  (h1 : performance_time = { hours := 8, minutes := 30 })
  (h2 : travel_time = 20) :
  subtractMinutes performance_time travel_time = { hours := 8, minutes := 10 } := by
  sorry

end NUMINAMATH_CALUDE_latest_departure_time_l805_80585


namespace NUMINAMATH_CALUDE_solution_comparison_l805_80534

theorem solution_comparison (a a' b b' : ℝ) 
  (ha : a > 0) (ha' : a' > 0) 
  (heq1 : ∃ x, 2 * a * x + b = 0) 
  (heq2 : ∃ x', 2 * a' * x' + b' = 0) 
  (hineq : (- b / (2 * a))^2 > (- b' / (2 * a'))^2) : 
  b^2 / a^2 > b'^2 / a'^2 := by
  sorry

end NUMINAMATH_CALUDE_solution_comparison_l805_80534


namespace NUMINAMATH_CALUDE_petya_max_win_margin_l805_80563

theorem petya_max_win_margin :
  ∀ (p1 p2 v1 v2 : ℕ),
    p1 + p2 + v1 + v2 = 27 →
    p1 = v1 + 9 →
    v2 = p2 + 9 →
    p1 + p2 > v1 + v2 →
    p1 + p2 - (v1 + v2) ≤ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_petya_max_win_margin_l805_80563


namespace NUMINAMATH_CALUDE_root_sum_product_l805_80598

theorem root_sum_product (p q r : ℂ) : 
  (5 * p ^ 3 - 10 * p ^ 2 + 17 * p - 7 = 0) →
  (5 * q ^ 3 - 10 * q ^ 2 + 17 * q - 7 = 0) →
  (5 * r ^ 3 - 10 * r ^ 2 + 17 * r - 7 = 0) →
  p * q + p * r + q * r = 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l805_80598


namespace NUMINAMATH_CALUDE_even_quadratic_implies_m_eq_two_l805_80543

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + (m-2)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-2)*x + 1

theorem even_quadratic_implies_m_eq_two (m : ℝ) (h : IsEven (f m)) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_m_eq_two_l805_80543


namespace NUMINAMATH_CALUDE_probability_relates_to_uncertain_events_l805_80556

-- Define the basic types of events
inductive Event
  | Certain
  | Impossible
  | Random

-- Define a probability function
def probability (e : Event) : Real :=
  match e with
  | Event.Certain => 1
  | Event.Impossible => 0
  | Event.Random => sorry -- Assumes a value between 0 and 1

-- Define what it means for an event to be uncertain
def is_uncertain (e : Event) : Prop :=
  e = Event.Random

-- State the theorem
theorem probability_relates_to_uncertain_events :
  ∃ (e : Event), is_uncertain e ∧ 0 < probability e ∧ probability e < 1 :=
sorry

end NUMINAMATH_CALUDE_probability_relates_to_uncertain_events_l805_80556


namespace NUMINAMATH_CALUDE_flood_probability_l805_80514

theorem flood_probability (p_30 p_40 : ℝ) 
  (h1 : p_30 = 0.8) 
  (h2 : p_40 = 0.85) : 
  (p_40 - p_30) / (1 - p_30) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_flood_probability_l805_80514


namespace NUMINAMATH_CALUDE_sum_of_digits_of_special_palindrome_l805_80589

/-- A function that checks if a natural number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ (n / 10 % 10 = n / 10 % 10)

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + (n / 10 % 10) + (n % 10)

/-- Theorem stating that if x is a three-digit palindrome and x + 50 is also a three-digit palindrome,
    then the sum of digits of x is 19 -/
theorem sum_of_digits_of_special_palindrome (x : ℕ) :
  isThreeDigitPalindrome x ∧ isThreeDigitPalindrome (x + 50) → sumOfDigits x = 19 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_special_palindrome_l805_80589


namespace NUMINAMATH_CALUDE_assignment_methods_count_l805_80525

/-- The number of ways to select and assign representatives -/
def assign_representatives (num_boys num_girls num_reps min_boys min_girls : ℕ) : ℕ :=
  -- Number of ways to select 2 boys and 2 girls
  (Nat.choose num_boys 2 * Nat.choose num_girls 2 * Nat.factorial num_reps) +
  -- Number of ways to select 3 boys and 1 girl
  (Nat.choose num_boys 3 * Nat.choose num_girls 1 * Nat.factorial num_reps)

/-- Theorem stating the number of assignment methods -/
theorem assignment_methods_count :
  assign_representatives 5 4 4 2 1 = 2400 :=
by sorry

end NUMINAMATH_CALUDE_assignment_methods_count_l805_80525


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_truncated_pyramid_l805_80569

/-- A regular truncated quadrilateral pyramid with an inscribed sphere -/
structure TruncatedPyramid where
  a : ℝ  -- height of the lateral face
  inscribed_sphere : Prop  -- property that a sphere can be inscribed

/-- The lateral surface area of a truncated quadrilateral pyramid -/
def lateral_surface_area (tp : TruncatedPyramid) : ℝ :=
  4 * tp.a^2

/-- Theorem: The lateral surface area of a regular truncated quadrilateral pyramid
    with an inscribed sphere is 4a^2, where a is the height of the lateral face -/
theorem lateral_surface_area_of_truncated_pyramid (tp : TruncatedPyramid) :
  tp.inscribed_sphere → lateral_surface_area tp = 4 * tp.a^2 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_truncated_pyramid_l805_80569


namespace NUMINAMATH_CALUDE_ellipse_tangent_line_l805_80587

/-- The equation of the tangent line to an ellipse -/
theorem ellipse_tangent_line 
  (a b x₀ y₀ : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_on_ellipse : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y : ℝ, (x₀ * x / a^2 + y₀ * y / b^2 = 1) ↔ 
    (∃ t : ℝ, x = x₀ + t * (-b^2 * x₀) ∧ y = y₀ + t * (a^2 * y₀) ∧ 
    ∀ u : ℝ, (x₀ + u * (-b^2 * x₀))^2 / a^2 + (y₀ + u * (a^2 * y₀))^2 / b^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_line_l805_80587


namespace NUMINAMATH_CALUDE_expression_evaluation_l805_80504

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/2
  2 * x^2 + (-x^2 - 2*x*y + 2*y^2) - 3*(x^2 - x*y + 2*y^2) = -10 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l805_80504


namespace NUMINAMATH_CALUDE_sum_integers_minus20_to_10_l805_80593

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_minus20_to_10 :
  sum_integers (-20) 10 = -155 := by sorry

end NUMINAMATH_CALUDE_sum_integers_minus20_to_10_l805_80593


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l805_80568

/-- Given a hyperbola with one asymptote y = -2x + 5 and foci with x-coordinate 2,
    prove that the equation of the other asymptote is y = 2x - 3 -/
theorem hyperbola_other_asymptote (x y : ℝ) :
  let asymptote1 : ℝ → ℝ := λ x => -2 * x + 5
  let foci_x : ℝ := 2
  let center_x : ℝ := foci_x
  let center_y : ℝ := asymptote1 center_x
  let asymptote2 : ℝ → ℝ := λ x => 2 * x - 3
  (∀ x, y = asymptote1 x) → (y = asymptote2 x) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l805_80568


namespace NUMINAMATH_CALUDE_license_plate_count_l805_80594

def license_plate_combinations : ℕ :=
  (Nat.choose 26 2) * (Nat.choose 5 2) * (Nat.choose 3 2) * 24 * 10 * 9 * 8

theorem license_plate_count : license_plate_combinations = 56016000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l805_80594


namespace NUMINAMATH_CALUDE_celestia_badges_l805_80520

def total_badges : ℕ := 83
def hermione_badges : ℕ := 14
def luna_badges : ℕ := 17

theorem celestia_badges : 
  total_badges - hermione_badges - luna_badges = 52 := by
  sorry

end NUMINAMATH_CALUDE_celestia_badges_l805_80520


namespace NUMINAMATH_CALUDE_car_original_price_l805_80542

/-- 
Given a car sale scenario where:
1. A car is sold at a 10% loss to a friend
2. The friend sells it for Rs. 54000 with a 20% gain

This theorem proves that the original cost price of the car was Rs. 50000.
-/
theorem car_original_price : ℝ → Prop :=
  fun original_price =>
    let friend_buying_price := 0.9 * original_price
    let friend_selling_price := 54000
    (1.2 * friend_buying_price = friend_selling_price) →
    (original_price = 50000)

-- The proof is omitted
example : car_original_price 50000 := by sorry

end NUMINAMATH_CALUDE_car_original_price_l805_80542


namespace NUMINAMATH_CALUDE_no_equal_result_from_19_and_98_l805_80519

/-- Represents the two possible operations: squaring or adding one -/
inductive Operation
  | square
  | addOne

/-- Applies the given operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.square => n * n
  | Operation.addOne => n + 1

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem stating that it's impossible to obtain the same number from 19 and 98
    using the same number of operations -/
theorem no_equal_result_from_19_and_98 :
  ¬ ∃ (ops1 ops2 : List Operation) (result : ℕ),
    ops1.length = ops2.length ∧
    applyOperations 19 ops1 = result ∧
    applyOperations 98 ops2 = result :=
  sorry


end NUMINAMATH_CALUDE_no_equal_result_from_19_and_98_l805_80519


namespace NUMINAMATH_CALUDE_x_over_y_equals_two_l805_80528

theorem x_over_y_equals_two (x y : ℝ) 
  (h1 : 3 < (x^2 - y^2) / (x^2 + y^2))
  (h2 : (x^2 - y^2) / (x^2 + y^2) < 4)
  (h3 : ∃ (n : ℤ), x / y = n) :
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_equals_two_l805_80528


namespace NUMINAMATH_CALUDE_jose_fowls_count_l805_80575

/-- The number of fowls Jose has is the sum of his chickens and ducks -/
theorem jose_fowls_count :
  let chickens : ℕ := 28
  let ducks : ℕ := 18
  let fowls : ℕ := chickens + ducks
  fowls = 46 := by sorry

end NUMINAMATH_CALUDE_jose_fowls_count_l805_80575


namespace NUMINAMATH_CALUDE_car_a_distance_at_2016th_meeting_l805_80510

/-- Represents a car with its current position and speed -/
structure Car where
  position : ℝ
  speed : ℝ

/-- The problem setup -/
def problem_setup : Prop :=
  ∃ (car_a car_b : Car) (distance : ℝ),
    distance = 900 ∧
    car_a.position = 0 ∧
    car_b.position = distance ∧
    ((car_a.position < car_b.position ∧ car_a.speed = 40) ∨
     (car_a.position > car_b.position ∧ car_a.speed = 50)) ∧
    ((car_b.position > car_a.position ∧ car_b.speed = 50) ∨
     (car_b.position < car_a.position ∧ car_b.speed = 40))

/-- The theorem to be proved -/
theorem car_a_distance_at_2016th_meeting :
  problem_setup →
  ∃ (total_distance : ℝ),
    total_distance = 1813900 ∧
    (∀ (t : ℝ), t ≥ 0 →
      ∃ (car_a car_b : Car),
        (car_a.position = total_distance ∨ car_b.position = total_distance) ∧
        (∀ (prev_meetings : ℕ), prev_meetings < 2016 →
          ∃ (t' : ℝ), t' < t ∧ car_a.position = car_b.position)) :=
by sorry

end NUMINAMATH_CALUDE_car_a_distance_at_2016th_meeting_l805_80510


namespace NUMINAMATH_CALUDE_quadratic_completion_l805_80592

theorem quadratic_completion (y : ℝ) : ∃ (k : ℤ) (a : ℝ), y^2 + 12*y + 40 = (y + a)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l805_80592


namespace NUMINAMATH_CALUDE_geometry_theorem_l805_80581

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Axioms
axiom different_lines {l1 l2 : Line} : l1 ≠ l2
axiom different_planes {p1 p2 : Plane} : p1 ≠ p2

-- Theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) :
  (perpendicular m n ∧ perpendicularLP m α ∧ ¬subset n α → parallel n α) ∧
  (perpendicular m n ∧ perpendicularLP m α ∧ perpendicularLP n β → perpendicularPP α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l805_80581


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l805_80595

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 64)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : (total_players - throwers) * 2 / 3 + throwers = 55 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l805_80595


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_nonpositive_max_value_when_a_is_3_min_value_when_a_is_3_l805_80547

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Part I: Increasing function implies a ≤ 0
theorem increasing_function_implies_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, ∀ y : ℝ, x < y → f a x < f a y) →
  a ≤ 0 :=
sorry

-- Part II: Maximum and minimum values when a = 3
theorem max_value_when_a_is_3 :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f 3 x ≤ 1) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f 3 x = 1) :=
sorry

theorem min_value_when_a_is_3 :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → -3 ≤ f 3 x) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f 3 x = -3) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_nonpositive_max_value_when_a_is_3_min_value_when_a_is_3_l805_80547


namespace NUMINAMATH_CALUDE_equation_solution_l805_80532

theorem equation_solution :
  ∃ x : ℝ, 4 * (x - 2) * (x + 5) = (2 * x - 3) * (2 * x + 11) + 11 ∧ x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l805_80532


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l805_80582

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the simplification of 4(2-i) + 2i(3-2i) is 12 + 2i -/
theorem simplify_complex_expression : 4 * (2 - i) + 2 * i * (3 - 2 * i) = 12 + 2 * i :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l805_80582


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l805_80512

/-- A quadratic function with axis of symmetry at x = 9.5 -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_symmetry (d e f : ℝ) :
  (∀ x, p d e f (9.5 + x) = p d e f (9.5 - x)) →  -- axis of symmetry at x = 9.5
  p d e f (-1) = 1 →  -- p(-1) = 1
  ∃ n : ℤ, p d e f 20 = n →  -- p(20) is an integer
  p d e f 20 = 1 := by  -- prove p(20) = 1
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l805_80512


namespace NUMINAMATH_CALUDE_min_games_to_satisfy_condition_l805_80590

/-- The number of teams in the tournament -/
def num_teams : ℕ := 20

/-- The total number of possible games between all teams -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The maximum number of unplayed games while satisfying the condition -/
def max_unplayed_games : ℕ := (num_teams / 2) ^ 2

/-- A function that checks if the number of played games satisfies the condition -/
def satisfies_condition (played_games : ℕ) : Prop :=
  played_games ≥ total_games - max_unplayed_games

/-- The theorem stating the minimum number of games that must be played -/
theorem min_games_to_satisfy_condition :
  ∃ (min_games : ℕ), satisfies_condition min_games ∧
  ∀ (n : ℕ), n < min_games → ¬satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_min_games_to_satisfy_condition_l805_80590


namespace NUMINAMATH_CALUDE_zoo_field_trip_vans_l805_80518

/-- The number of vans needed for a field trip --/
def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

/-- Theorem: The number of vans needed for the zoo field trip is 6 --/
theorem zoo_field_trip_vans : vans_needed 5 25 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_zoo_field_trip_vans_l805_80518


namespace NUMINAMATH_CALUDE_frame_interior_edges_sum_l805_80572

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_length : ℝ
  outer_width : ℝ
  frame_width : ℝ

/-- Calculates the area of the frame -/
def frame_area (f : Frame) : ℝ :=
  f.outer_length * f.outer_width - (f.outer_length - 2 * f.frame_width) * (f.outer_width - 2 * f.frame_width)

/-- Calculates the sum of the lengths of the four interior edges of the frame -/
def interior_edges_sum (f : Frame) : ℝ :=
  2 * ((f.outer_length - 2 * f.frame_width) + (f.outer_width - 2 * f.frame_width))

/-- Theorem stating that for a frame with given conditions, the sum of interior edges is 8 inches -/
theorem frame_interior_edges_sum :
  ∀ (f : Frame),
    f.frame_width = 2 →
    f.outer_length = 8 →
    frame_area f = 32 →
    interior_edges_sum f = 8 := by
  sorry

end NUMINAMATH_CALUDE_frame_interior_edges_sum_l805_80572


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l805_80516

theorem absolute_value_nonnegative (a : ℝ) : ¬(|a| < 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l805_80516


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l805_80577

theorem square_difference_of_integers (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 20) :
  a^2 - b^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l805_80577


namespace NUMINAMATH_CALUDE_triangle_area_l805_80554

theorem triangle_area (base height : ℝ) (h1 : base = 3) (h2 : height = 4) :
  (base * height) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l805_80554


namespace NUMINAMATH_CALUDE_polynomial_composition_l805_80507

theorem polynomial_composition (g : ℝ → ℝ) :
  (∀ x, g x ^ 2 = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_composition_l805_80507


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l805_80540

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) -- Points in 2D plane
  (is_right_triangle : (Q.1 - R.1) * (P.1 - R.1) + (Q.2 - R.2) * (P.2 - R.2) = 0) -- Right angle condition
  (cos_R : ((Q.1 - R.1) * (P.1 - R.1) + (Q.2 - R.2) * (P.2 - R.2)) / 
           (Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) * Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)) = 3/5) -- cos R = 3/5
  (RP_length : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 10) -- RP = 10
  : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 8 := by -- PQ = 8
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l805_80540


namespace NUMINAMATH_CALUDE_vacant_seats_l805_80523

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 45/100) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l805_80523


namespace NUMINAMATH_CALUDE_max_value_of_f_l805_80513

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x - Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l805_80513


namespace NUMINAMATH_CALUDE_sum_of_powers_l805_80576

theorem sum_of_powers (a b : ℝ) : 
  (a + b = 1) → 
  (a^2 + b^2 = 3) → 
  (a^3 + b^3 = 4) → 
  (a^4 + b^4 = 7) → 
  (a^5 + b^5 = 11) → 
  (∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) →
  a^6 + b^6 = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_l805_80576


namespace NUMINAMATH_CALUDE_all_students_same_room_probability_l805_80505

/-- The number of rooms available for assignment. -/
def num_rooms : ℕ := 4

/-- The number of students being assigned to rooms. -/
def num_students : ℕ := 3

/-- The probability of a student being assigned to any specific room. -/
def prob_per_room : ℚ := 1 / num_rooms

/-- The total number of possible assignment outcomes. -/
def total_outcomes : ℕ := num_rooms ^ num_students

/-- The number of favorable outcomes (all students in the same room). -/
def favorable_outcomes : ℕ := num_rooms

/-- The probability that all students are assigned to the same room. -/
theorem all_students_same_room_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_all_students_same_room_probability_l805_80505


namespace NUMINAMATH_CALUDE_quadratic_factorization_l805_80561

theorem quadratic_factorization (y a b : ℤ) : 
  (3 * y^2 - 7 * y - 6 = (3 * y + a) * (y + b)) → (a - b = 5) := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l805_80561


namespace NUMINAMATH_CALUDE_equation_roots_imply_m_range_l805_80533

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*m*x + 4*m^2 + 2*m + 3 = 0 ∨ x^2 + (2*m + 1)*x + m^2 = 0) → 
  (m ≤ -3/2 ∨ m ≥ -1/4) := by
sorry

end NUMINAMATH_CALUDE_equation_roots_imply_m_range_l805_80533


namespace NUMINAMATH_CALUDE_sum_specific_repeating_decimals_l805_80502

/-- Represents a repeating decimal with a whole number part and a repeating part -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ
  base : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (d.base - 1 : ℚ)

/-- The sum of specific repeating decimals equals 3948/9999 -/
theorem sum_specific_repeating_decimals :
  let d1 := RepeatingDecimal.toRational ⟨0, 3, 10⟩
  let d2 := RepeatingDecimal.toRational ⟨0, 6, 100⟩
  let d3 := RepeatingDecimal.toRational ⟨0, 9, 10000⟩
  d1 + d2 + d3 = 3948 / 9999 := by
  sorry

#eval (3948 : ℚ) / 9999

end NUMINAMATH_CALUDE_sum_specific_repeating_decimals_l805_80502


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l805_80529

/-- A point on an ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / 16 + y^2 / 4 = 1

/-- The distance from a point to a focus -/
def distance_to_focus (P : PointOnEllipse) (F : ℝ × ℝ) : ℝ := sorry

/-- The foci of the ellipse -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) := sorry

theorem ellipse_focal_property (P : PointOnEllipse) :
  distance_to_focus P (foci.1) = 3 →
  distance_to_focus P (foci.2) = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l805_80529


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l805_80503

/-- A polynomial that is always a perfect square for integer inputs can be expressed as (dx + e)^2 -/
theorem perfect_square_polynomial
  (a b c : ℤ)
  (h : ∀ (x : ℤ), ∃ (y : ℤ), a * x^2 + b * x + c = y^2) :
  ∃ (d e : ℤ), ∀ (x : ℤ), a * x^2 + b * x + c = (d * x + e)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l805_80503


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l805_80596

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 4 + 2 * Complex.I) :
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l805_80596


namespace NUMINAMATH_CALUDE_condition_a_necessary_not_sufficient_l805_80565

-- Define Condition A
def condition_a (x y : ℝ) : Prop :=
  2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3

-- Define Condition B
def condition_b (x y : ℝ) : Prop :=
  0 < x ∧ x < 1 ∧ 2 < y ∧ y < 3

-- Theorem stating that Condition A is necessary but not sufficient for Condition B
theorem condition_a_necessary_not_sufficient :
  (∀ x y : ℝ, condition_b x y → condition_a x y) ∧
  (∃ x y : ℝ, condition_a x y ∧ ¬condition_b x y) := by
  sorry

end NUMINAMATH_CALUDE_condition_a_necessary_not_sufficient_l805_80565


namespace NUMINAMATH_CALUDE_pats_calculation_l805_80584

theorem pats_calculation (x : ℝ) : 
  (x / 8) - 20 = 12 → 
  1800 < (x * 8) + 20 ∧ (x * 8) + 20 < 2200 :=
by
  sorry

end NUMINAMATH_CALUDE_pats_calculation_l805_80584


namespace NUMINAMATH_CALUDE_consecutive_non_primes_l805_80580

theorem consecutive_non_primes (k : ℕ+) : ∃ n : ℕ, ∀ i : ℕ, i < k → ¬ Nat.Prime (n + i) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_l805_80580


namespace NUMINAMATH_CALUDE_fair_ride_cost_l805_80544

theorem fair_ride_cost (total_tickets : ℕ) (booth_tickets : ℕ) (num_rides : ℕ) 
  (h1 : total_tickets = 79) 
  (h2 : booth_tickets = 23) 
  (h3 : num_rides = 8) : 
  (total_tickets - booth_tickets) / num_rides = 7 := by
  sorry

end NUMINAMATH_CALUDE_fair_ride_cost_l805_80544


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l805_80509

/-- Given a principal amount and an interest rate, if the simple interest
    for 2 years is $400 and the compound interest for 2 years is $440,
    then the interest rate is 20%. -/
theorem interest_rate_calculation (P r : ℝ) :
  P * r * 2 = 400 →
  P * ((1 + r)^2 - 1) = 440 →
  r = 0.20 := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l805_80509


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l805_80526

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculates the hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Theorem stating Sheila's hourly wage --/
theorem sheila_hourly_wage (sheila : WorkSchedule)
  (h1 : sheila.hours_mon_wed_fri = 8)
  (h2 : sheila.hours_tue_thu = 6)
  (h3 : sheila.weekly_earnings = 468) :
  hourly_wage sheila = 13 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l805_80526


namespace NUMINAMATH_CALUDE_dice_roll_probability_l805_80522

/-- The probability of rolling a number other than 1 on a standard die -/
def prob_not_one : ℚ := 5 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability of rolling four dice and getting no 1s -/
def prob_no_ones : ℚ := prob_not_one ^ num_dice

theorem dice_roll_probability :
  prob_no_ones = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l805_80522


namespace NUMINAMATH_CALUDE_total_splash_width_is_seven_l805_80548

/-- The width of a splash made by a pebble in meters -/
def pebble_splash_width : ℚ := 1/4

/-- The width of a splash made by a rock in meters -/
def rock_splash_width : ℚ := 1/2

/-- The width of a splash made by a boulder in meters -/
def boulder_splash_width : ℚ := 2

/-- The number of pebbles thrown -/
def num_pebbles : ℕ := 6

/-- The number of rocks thrown -/
def num_rocks : ℕ := 3

/-- The number of boulders thrown -/
def num_boulders : ℕ := 2

/-- The total width of splashes made by throwing pebbles, rocks, and boulders -/
def total_splash_width : ℚ :=
  num_pebbles * pebble_splash_width +
  num_rocks * rock_splash_width +
  num_boulders * boulder_splash_width

theorem total_splash_width_is_seven :
  total_splash_width = 7 := by sorry

end NUMINAMATH_CALUDE_total_splash_width_is_seven_l805_80548


namespace NUMINAMATH_CALUDE_seven_x_plus_four_is_odd_l805_80555

theorem seven_x_plus_four_is_odd (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_seven_x_plus_four_is_odd_l805_80555


namespace NUMINAMATH_CALUDE_spinner_divisible_by_three_probability_l805_80573

/-- Represents the possible outcomes of the spinner -/
inductive SpinnerOutcome
  | One
  | Two
  | Four

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinnerOutcome
  tens : SpinnerOutcome
  units : SpinnerOutcome

/-- Converts a SpinnerOutcome to its numerical value -/
def spinnerValue (outcome : SpinnerOutcome) : Nat :=
  match outcome with
  | SpinnerOutcome.One => 1
  | SpinnerOutcome.Two => 2
  | SpinnerOutcome.Four => 4

/-- Checks if a ThreeDigitNumber is divisible by 3 -/
def isDivisibleByThree (n : ThreeDigitNumber) : Bool :=
  (spinnerValue n.hundreds + spinnerValue n.tens + spinnerValue n.units) % 3 = 0

/-- Calculates the probability of getting a number divisible by 3 -/
def probabilityDivisibleByThree : ℚ :=
  let totalOutcomes := 27  -- 3^3
  let favorableOutcomes := 6  -- Counted from the problem
  favorableOutcomes / totalOutcomes

/-- Main theorem: The probability of getting a number divisible by 3 is 2/9 -/
theorem spinner_divisible_by_three_probability :
  probabilityDivisibleByThree = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_spinner_divisible_by_three_probability_l805_80573


namespace NUMINAMATH_CALUDE_polyhedron_formula_l805_80586

/-- Represents a convex polyhedron with specific face configuration -/
structure Polyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  hexagons : ℕ
  T : ℕ
  P : ℕ
  H : ℕ
  faces_sum : faces = triangles + pentagons + hexagons
  faces_types : faces = 32 ∧ triangles = 10 ∧ pentagons = 8 ∧ hexagons = 14

/-- Calculates the number of edges in the polyhedron -/
def edges (poly : Polyhedron) : ℕ :=
  (3 * poly.triangles + 5 * poly.pentagons + 6 * poly.hexagons) / 2

/-- Calculates the number of vertices in the polyhedron using Euler's formula -/
def vertices (poly : Polyhedron) : ℕ :=
  edges poly - poly.faces + 2

/-- Theorem stating that for the given polyhedron, 100P + 10T + V = 249 -/
theorem polyhedron_formula (poly : Polyhedron) : 100 * poly.P + 10 * poly.T + vertices poly = 249 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_formula_l805_80586


namespace NUMINAMATH_CALUDE_quadratic_lower_bound_l805_80560

theorem quadratic_lower_bound 
  (f : ℝ → ℝ) 
  (a b : ℤ) 
  (h1 : ∀ x, f x = x^2 + a*x + b) 
  (h2 : ∀ x, f x ≥ -9/10) : 
  ∀ x, f x ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_lower_bound_l805_80560


namespace NUMINAMATH_CALUDE_equation_solution_l805_80597

theorem equation_solution : ∃ Z : ℤ, 80 - (5 - (Z + 2 * (7 - 8 - 5))) = 89 ∧ Z = 26 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l805_80597


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l805_80564

theorem complex_on_imaginary_axis (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 1) → z.re = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l805_80564


namespace NUMINAMATH_CALUDE_grade_assignments_l805_80562

/-- The number of possible grades to assign -/
def num_grades : ℕ := 3

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignments :
  num_grades ^ num_students = 531441 := by
sorry

end NUMINAMATH_CALUDE_grade_assignments_l805_80562
