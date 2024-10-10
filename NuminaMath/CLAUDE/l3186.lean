import Mathlib

namespace smallest_n_for_perfect_square_sum_l3186_318622

theorem smallest_n_for_perfect_square_sum (n : ℕ) : n = 7 ↔ 
  (∀ k ≥ n, ∀ x ∈ Finset.range k, ∃ y ∈ Finset.range k, y ≠ x ∧ ∃ m : ℕ, x + y = m^2) ∧
  (∀ n' < n, ∃ k ≥ n', ∃ x ∈ Finset.range k, ∀ y ∈ Finset.range k, y = x ∨ ∀ m : ℕ, x + y ≠ m^2) :=
sorry

end smallest_n_for_perfect_square_sum_l3186_318622


namespace probability_gpa_at_least_3_5_l3186_318696

/-- Represents the possible grades a student can receive --/
inductive Grade
| A
| B
| C
| D

/-- Converts a grade to its point value --/
def gradeToPoints : Grade → ℕ
| Grade.A => 4
| Grade.B => 3
| Grade.C => 2
| Grade.D => 1

/-- Calculates the GPA given a list of grades --/
def calculateGPA (grades : List Grade) : ℚ :=
  (grades.map gradeToPoints).sum / 4

/-- Represents the probability of getting each grade in a subject --/
structure GradeProbability where
  probA : ℚ
  probB : ℚ
  probC : ℚ
  probD : ℚ

/-- The probability distribution for English grades --/
def englishProb : GradeProbability :=
  { probA := 1/6
  , probB := 1/4
  , probC := 7/12
  , probD := 0 }

/-- The probability distribution for History grades --/
def historyProb : GradeProbability :=
  { probA := 1/4
  , probB := 1/3
  , probC := 5/12
  , probD := 0 }

/-- Theorem stating the probability of getting a GPA of at least 3.5 --/
theorem probability_gpa_at_least_3_5 :
  let mathGrade := Grade.A
  let scienceGrade := Grade.A
  let probAtLeast3_5 := (englishProb.probA * historyProb.probA) +
                        (englishProb.probA * historyProb.probB) +
                        (englishProb.probB * historyProb.probA) +
                        (englishProb.probA * historyProb.probC) +
                        (englishProb.probC * historyProb.probA) +
                        (englishProb.probB * historyProb.probB)
  probAtLeast3_5 = 11/24 := by
  sorry

end probability_gpa_at_least_3_5_l3186_318696


namespace number_equation_l3186_318607

theorem number_equation (x : ℤ) : 45 - (x - (37 - (15 - 20))) = 59 ↔ x = 28 := by sorry

end number_equation_l3186_318607


namespace sum_of_squares_verify_sum_of_squares_l3186_318616

theorem sum_of_squares : ℕ → Prop
  | 1009 => 1009 = 15^2 + 28^2
  | 2018 => 2018 = 43^2 + 13^2
  | _ => True

theorem verify_sum_of_squares :
  sum_of_squares 1009 ∧ sum_of_squares 2018 := by
  sorry

end sum_of_squares_verify_sum_of_squares_l3186_318616


namespace complex_equation_implies_sum_l3186_318644

def complex (a b : ℝ) : ℂ := Complex.mk a b

theorem complex_equation_implies_sum (a b : ℝ) :
  complex 9 3 * complex a b = complex 10 4 →
  a + b = 6/5 := by
  sorry

end complex_equation_implies_sum_l3186_318644


namespace cubic_monotonicity_implies_one_intersection_one_intersection_not_implies_monotonicity_l3186_318626

-- Define the cubic function
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define strict monotonicity
def strictly_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ (∀ x y, x < y → f x > f y)

-- Define the property of intersecting x-axis exactly once
def intersects_x_axis_once (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Main theorem
theorem cubic_monotonicity_implies_one_intersection
  (a b c d : ℝ) (h : a ≠ 0) :
  strictly_monotonic (f a b c d) →
  intersects_x_axis_once (f a b c d) :=
sorry

-- Counterexample theorem
theorem one_intersection_not_implies_monotonicity :
  ∃ a b c d : ℝ,
    intersects_x_axis_once (f a b c d) ∧
    ¬strictly_monotonic (f a b c d) :=
sorry

end cubic_monotonicity_implies_one_intersection_one_intersection_not_implies_monotonicity_l3186_318626


namespace tower_height_differences_l3186_318648

/-- Heights of towers in meters -/
def CN_Tower_height : ℝ := 553
def CN_Tower_Space_Needle_diff : ℝ := 369
def Eiffel_Tower_height : ℝ := 330
def Jeddah_Tower_predicted_height : ℝ := 1000

/-- Calculate the Space Needle height -/
def Space_Needle_height : ℝ := CN_Tower_height - CN_Tower_Space_Needle_diff

/-- Theorem stating the height differences -/
theorem tower_height_differences :
  (Eiffel_Tower_height - Space_Needle_height = 146) ∧
  (Jeddah_Tower_predicted_height - Eiffel_Tower_height = 670) :=
by sorry

end tower_height_differences_l3186_318648


namespace fraction_sum_equation_l3186_318614

theorem fraction_sum_equation (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = 2 / 7 → 
  ((x = 4 ∧ y = 28) ∨ (x = 28 ∧ y = 4)) :=
by sorry

end fraction_sum_equation_l3186_318614


namespace function_minimum_at_three_l3186_318672

/-- The function f(x) = x(x - c)^2 has a minimum value at x = 3 if and only if c = 3 -/
theorem function_minimum_at_three (c : ℝ) : 
  (∀ x, x * (x - c)^2 ≥ 3 * (3 - c)^2) ↔ c = 3 := by sorry

end function_minimum_at_three_l3186_318672


namespace factorial_multiple_l3186_318695

theorem factorial_multiple (m n : ℕ) : 
  ∃ k : ℕ, (2 * m).factorial * (2 * n).factorial = k * m.factorial * n.factorial * (m + n).factorial := by
sorry

end factorial_multiple_l3186_318695


namespace pastor_prayer_difference_l3186_318610

/-- Pastor Paul's daily prayer count on weekdays -/
def paul_weekday : ℕ := 20

/-- Pastor Paul's Sunday prayer count -/
def paul_sunday : ℕ := 2 * paul_weekday

/-- Pastor Bruce's weekday prayer count -/
def bruce_weekday : ℕ := paul_weekday / 2

/-- Pastor Bruce's Sunday prayer count -/
def bruce_sunday : ℕ := 2 * paul_sunday

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Number of weekdays in a week -/
def weekdays : ℕ := 6

theorem pastor_prayer_difference :
  paul_weekday * weekdays + paul_sunday - (bruce_weekday * weekdays + bruce_sunday) = 20 := by
  sorry

end pastor_prayer_difference_l3186_318610


namespace subtraction_proof_l3186_318628

theorem subtraction_proof :
  900000009000 - 123456789123 = 776543220777 := by
  sorry

end subtraction_proof_l3186_318628


namespace power_calculation_l3186_318654

theorem power_calculation : 16^16 * 2^10 / 4^22 = 2^30 := by
  sorry

end power_calculation_l3186_318654


namespace sqrt_one_minus_two_sin_two_cos_two_l3186_318671

theorem sqrt_one_minus_two_sin_two_cos_two (h : π / 2 < 2 ∧ 2 < π) :
  Real.sqrt (1 - 2 * Real.sin 2 * Real.cos 2) = Real.sin 2 - Real.cos 2 := by
  sorry

end sqrt_one_minus_two_sin_two_cos_two_l3186_318671


namespace solution_set_inequality_l3186_318639

/-- The solution set of the inequality x + 2/(x+1) > 2 -/
theorem solution_set_inequality (x : ℝ) : x + 2 / (x + 1) > 2 ↔ x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi 1 := by
  sorry

end solution_set_inequality_l3186_318639


namespace hyperbola_asymptote_equation_l3186_318620

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the distance from the right focus to the left vertex is equal to twice
    the distance from it to the asymptote, then the equation of its asymptote
    is 4x ± 3y = 0. -/
theorem hyperbola_asymptote_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_distance : ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 →
    (∃ (c : ℝ), a + c = 2 * (b * c / Real.sqrt (a^2 + b^2)))) :
  ∃ (k : ℝ), k > 0 ∧ (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (4*x = 3*y ∨ 4*x = -3*y)) := by
  sorry

end hyperbola_asymptote_equation_l3186_318620


namespace min_m_and_range_l3186_318658

noncomputable section

def f (x : ℝ) := (1 + x)^2 - 2 * Real.log (1 + x)

theorem min_m_and_range (x₀ : ℝ) (h : x₀ ∈ Set.Icc 0 1) :
  (∀ x ∈ Set.Icc 0 1, f x - (4 - 2 * Real.log 2) ≤ 0) ∧
  (f x₀ - 1 ≤ 0 → ∀ m : ℝ, f x₀ - m ≤ 0 → m ≥ 1) :=
by sorry

end

end min_m_and_range_l3186_318658


namespace largest_consecutive_sum_45_l3186_318673

/-- The sum of n consecutive positive integers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

/-- The largest number of positive consecutive integers that sum to 45 -/
theorem largest_consecutive_sum_45 :
  (∃ (k : ℕ), k > 0 ∧ consecutiveSum 9 k = 45) ∧
  (∀ (n k : ℕ), n > 9 → k > 0 → consecutiveSum n k ≠ 45) :=
sorry

end largest_consecutive_sum_45_l3186_318673


namespace savings_calculation_l3186_318699

/-- Calculates the amount left in savings after distributing funds to family members --/
def savings_amount (initial : ℚ) (wife_fraction : ℚ) (son1_fraction : ℚ) (son2_fraction : ℚ) : ℚ :=
  let wife_share := wife_fraction * initial
  let after_wife := initial - wife_share
  let son1_share := son1_fraction * after_wife
  let after_son1 := after_wife - son1_share
  let son2_share := son2_fraction * after_son1
  after_son1 - son2_share

/-- Theorem stating the amount left in savings after distribution --/
theorem savings_calculation :
  savings_amount 2000 (2/5) (2/5) (40/100) = 432 := by
  sorry

end savings_calculation_l3186_318699


namespace power_difference_lower_bound_l3186_318617

theorem power_difference_lower_bound 
  (m n : ℕ) 
  (h1 : m > 1) 
  (h2 : 2^(2*m + 1) - n^2 ≥ 0) : 
  2^(2*m + 1) - n^2 ≥ 7 := by
sorry

end power_difference_lower_bound_l3186_318617


namespace rectangle_length_equals_nine_l3186_318659

-- Define the side length of the square
def square_side : ℝ := 6

-- Define the width of the rectangle
def rectangle_width : ℝ := 4

-- Define the area of the square
def square_area : ℝ := square_side * square_side

-- Define the area of the rectangle
def rectangle_area (length : ℝ) : ℝ := length * rectangle_width

-- Theorem statement
theorem rectangle_length_equals_nine :
  ∃ (length : ℝ), rectangle_area length = square_area ∧ length = 9 := by
  sorry

end rectangle_length_equals_nine_l3186_318659


namespace largest_n_divisible_by_seven_l3186_318621

def expression (n : ℕ) : ℤ :=
  10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36

theorem largest_n_divisible_by_seven :
  ∃ (n : ℕ), n < 50000 ∧
    7 ∣ expression n ∧
    ∀ (m : ℕ), m < 50000 → 7 ∣ expression m → m ≤ n :=
by
  use 49999
  sorry

end largest_n_divisible_by_seven_l3186_318621


namespace translation_theorem_l3186_318631

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D space -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_theorem :
  let A : Point := { x := -3, y := 2 }
  let A' : Point := translate (translate A 4 0) 0 (-3)
  A'.x = 1 ∧ A'.y = -1 := by sorry

end translation_theorem_l3186_318631


namespace special_right_triangle_angles_l3186_318690

/-- A right triangle with the property that when rotated four times,
    each time aligning the shorter leg with the hypotenuse and
    matching the vertex of the acute angle with the vertex of the right angle,
    results in an isosceles fifth triangle. -/
structure SpecialRightTriangle where
  /-- The measure of one of the acute angles in the triangle -/
  α : Real
  /-- The triangle is a right triangle -/
  is_right_triangle : α + (90 - α) + 90 = 180
  /-- The fifth triangle is isosceles -/
  fifth_triangle_isosceles : 4 * α = 180 - 4 * (90 + α)

/-- Theorem stating that the acute angles in the special right triangle are both 90°/11 -/
theorem special_right_triangle_angles (t : SpecialRightTriangle) : t.α = 90 / 11 := by
  sorry

end special_right_triangle_angles_l3186_318690


namespace solution_system1_solution_system2_l3186_318674

-- System 1
def system1 (x y : ℝ) : Prop :=
  3 * (x - 1) = y + 5 ∧ 5 * (y - 1) = 3 * (x + 5)

-- System 2
def system2 (x y a : ℝ) : Prop :=
  2 * x + 4 * y = a ∧ 7 * x - 2 * y = 3 * a

theorem solution_system1 :
  ∃ x y : ℝ, system1 x y ∧ x = 5 ∧ y = 7 := by sorry

theorem solution_system2 :
  ∀ a : ℝ, ∃ x y : ℝ, system2 x y a ∧ x = 7 / 16 * a ∧ y = 1 / 32 * a := by sorry

end solution_system1_solution_system2_l3186_318674


namespace symmetry_implies_x_equals_one_l3186_318685

/-- A function f: ℝ → ℝ has symmetric graphs for y = f(x-1) and y = f(1-x) with respect to x = 1 -/
def has_symmetric_graphs (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 1) = f (1 - x)

/-- If a function has symmetric graphs for y = f(x-1) and y = f(1-x), 
    then they are symmetric with respect to x = 1 -/
theorem symmetry_implies_x_equals_one (f : ℝ → ℝ) 
    (h : has_symmetric_graphs f) : 
    ∃ a : ℝ, a = 1 ∧ ∀ x : ℝ, f (a + (x - a)) = f (a - (x - a)) :=
  sorry

end symmetry_implies_x_equals_one_l3186_318685


namespace equation_solution_l3186_318667

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (x^3 - 3*x^2) / (x^2 - 4) + 2*x = -16 ↔ x = (23 + Real.sqrt 145) / 6 ∨ x = (23 - Real.sqrt 145) / 6 :=
by sorry

end equation_solution_l3186_318667


namespace martha_has_115_cards_l3186_318624

/-- The number of cards Martha has at the end of the transactions -/
def martha_final_cards : ℕ :=
  let initial_cards : ℕ := 3
  let cards_from_emily : ℕ := 25
  let cards_from_alex : ℕ := 43
  let cards_from_jenny : ℕ := 58
  let cards_given_to_sam : ℕ := 14
  initial_cards + cards_from_emily + cards_from_alex + cards_from_jenny - cards_given_to_sam

/-- Theorem stating that Martha ends up with 115 cards -/
theorem martha_has_115_cards : martha_final_cards = 115 := by
  sorry

end martha_has_115_cards_l3186_318624


namespace ryan_lost_leaves_l3186_318637

theorem ryan_lost_leaves (initial_leaves : ℕ) (broken_leaves : ℕ) (remaining_leaves : ℕ) : 
  initial_leaves = 89 → broken_leaves = 43 → remaining_leaves = 22 → 
  initial_leaves - (initial_leaves - remaining_leaves - broken_leaves) - broken_leaves = remaining_leaves :=
by
  sorry

end ryan_lost_leaves_l3186_318637


namespace field_trip_students_l3186_318679

/-- The number of people a van can hold -/
def van_capacity : ℕ := 5

/-- The number of adults going on the trip -/
def num_adults : ℕ := 3

/-- The number of vans needed for the trip -/
def num_vans : ℕ := 3

/-- The number of students going on the field trip -/
def num_students : ℕ := van_capacity * num_vans - num_adults

theorem field_trip_students : num_students = 12 := by
  sorry

end field_trip_students_l3186_318679


namespace triangle_formation_l3186_318645

theorem triangle_formation (a : ℝ) : 
  (0 < a ∧ a + 3 > 5 ∧ a + 5 > 3 ∧ 3 + 5 > a) ↔ a = 4 :=
by sorry

end triangle_formation_l3186_318645


namespace point_M_coordinates_l3186_318605

-- Define point M
def M (a : ℝ) : ℝ × ℝ := (a + 3, a + 1)

-- Define the condition for a point to be on the x-axis
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Theorem statement
theorem point_M_coordinates :
  ∀ a : ℝ, on_x_axis (M a) → M a = (2, 0) :=
by sorry

end point_M_coordinates_l3186_318605


namespace division_result_l3186_318634

theorem division_result (x : ℚ) : x / 5000 = 0.0114 → x = 57 := by
  sorry

end division_result_l3186_318634


namespace cheerleader_count_l3186_318656

theorem cheerleader_count (size2 : ℕ) (size6 : ℕ) : 
  size2 = 4 → size6 = 10 → size2 + size6 + (size6 / 2) = 19 := by
  sorry

end cheerleader_count_l3186_318656


namespace equation_represents_pair_of_lines_l3186_318651

theorem equation_represents_pair_of_lines : 
  ∃ (m₁ m₂ : ℝ), ∀ (x y : ℝ), 4 * x^2 - 9 * y^2 = 0 ↔ (y = m₁ * x ∨ y = m₂ * x) :=
sorry

end equation_represents_pair_of_lines_l3186_318651


namespace polynomial_irreducibility_l3186_318693

theorem polynomial_irreducibility (n : ℕ) (h : n > 1) :
  Irreducible (Polynomial.X ^ n + 5 * Polynomial.X ^ (n - 1) + 3 : Polynomial ℤ) := by
  sorry

end polynomial_irreducibility_l3186_318693


namespace survey_properties_l3186_318603

/-- Represents a student in the survey -/
structure Student where
  physicalCondition : String

/-- Represents the survey conducted by the school -/
structure Survey where
  students : List Student
  classes : Nat

/-- Defines the sample of the survey -/
def sample (s : Survey) : String :=
  s.students.map (λ student => student.physicalCondition) |> toString

/-- Defines the sample size of the survey -/
def sampleSize (s : Survey) : Nat :=
  s.students.length

/-- Theorem stating the properties of the survey -/
theorem survey_properties (s : Survey) 
  (h1 : s.students.length = 190)
  (h2 : s.classes = 19) :
  sample s = "physical condition of 190 students" ∧ 
  sampleSize s = 190 := by
  sorry

#check survey_properties

end survey_properties_l3186_318603


namespace geometric_sequence_sum_l3186_318618

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 1 = 1 → q = -2 →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
  sorry

end geometric_sequence_sum_l3186_318618


namespace age_ratio_problem_l3186_318647

/-- Represents the ratio of two numbers as a pair of integers -/
structure Ratio where
  num : Int
  den : Int
  pos : 0 < den

def Ratio.of (a b : Int) (h : 0 < b) : Ratio :=
  ⟨a, b, h⟩

theorem age_ratio_problem (rahul_future_age deepak_current_age : ℕ) 
  (h1 : rahul_future_age = 50)
  (h2 : deepak_current_age = 33) :
  ∃ (r : Ratio), r = Ratio.of 4 3 (by norm_num) ∧ 
    (rahul_future_age - 6 : ℚ) / deepak_current_age = r.num / r.den := by
  sorry

end age_ratio_problem_l3186_318647


namespace correct_arrangement_count_l3186_318641

/-- The number of ways to arrange 3 boys and 2 girls in a line, with the girls being adjacent -/
def arrangement_count : ℕ := 48

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 2

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_arrangement_count : 
  arrangement_count = 
    (Nat.factorial num_boys) * 
    (Nat.choose (num_boys + 1) 1) * 
    (Nat.factorial num_girls) :=
by sorry

end correct_arrangement_count_l3186_318641


namespace a_gt_b_gt_c_l3186_318627

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi
noncomputable def c : ℝ := Real.log 0.9 / Real.log 2

theorem a_gt_b_gt_c : a > b ∧ b > c := by sorry

end a_gt_b_gt_c_l3186_318627


namespace systematic_sampling_interval_for_given_problem_l3186_318638

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

theorem systematic_sampling_interval_for_given_problem :
  let populationSize : ℕ := 1000
  let sampleSize : ℕ := 40
  systematicSamplingInterval populationSize sampleSize = 25 := by
  sorry

#eval systematicSamplingInterval 1000 40

end systematic_sampling_interval_for_given_problem_l3186_318638


namespace sum_of_y_coordinates_l3186_318604

theorem sum_of_y_coordinates : ∀ y₁ y₂ : ℝ,
  (4 - (-1))^2 + (y₁ - 3)^2 = 8^2 →
  (4 - (-1))^2 + (y₂ - 3)^2 = 8^2 →
  y₁ + y₂ = 6 := by sorry

end sum_of_y_coordinates_l3186_318604


namespace problem_solution_l3186_318663

theorem problem_solution (x y : ℝ) 
  (h1 : x^2 + y^2 - x*y = 2) 
  (h2 : x^4 + y^4 + x^2*y^2 = 8) : 
  x^8 + y^8 + x^2014*y^2014 = 48 := by
  sorry

end problem_solution_l3186_318663


namespace simple_interest_theorem_l3186_318666

/-- Proves that when the simple interest on a sum of money is 2/5 of the principal amount,
    and the rate is 4% per annum, the time period is 10 years. -/
theorem simple_interest_theorem (P : ℝ) (R : ℝ) (T : ℝ) :
  R = 4 →
  (2 / 5) * P = (P * R * T) / 100 →
  T = 10 := by
sorry


end simple_interest_theorem_l3186_318666


namespace twice_difference_l3186_318689

/-- Given two real numbers m and n, prove that 2(m-n) is equivalent to twice the difference between m and n -/
theorem twice_difference (m n : ℝ) : 2 * (m - n) = 2 * m - 2 * n := by
  sorry

end twice_difference_l3186_318689


namespace fraction_sum_lower_bound_l3186_318652

theorem fraction_sum_lower_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2 := by
  sorry

end fraction_sum_lower_bound_l3186_318652


namespace at_least_one_not_less_than_one_l3186_318687

theorem at_least_one_not_less_than_one (a b c : ℝ) (h : a + b + c = 3) :
  ¬(a < 1 ∧ b < 1 ∧ c < 1) := by
sorry

end at_least_one_not_less_than_one_l3186_318687


namespace min_sum_abc_l3186_318611

def is_min_sum (a b c : ℕ) : Prop :=
  ∀ x y z : ℕ, 
    (Nat.lcm (Nat.lcm x y) z = 48) → 
    (Nat.gcd x y = 4) → 
    (Nat.gcd y z = 3) → 
    a + b + c ≤ x + y + z

theorem min_sum_abc : 
  ∃ a b c : ℕ,
    (Nat.lcm (Nat.lcm a b) c = 48) ∧ 
    (Nat.gcd a b = 4) ∧ 
    (Nat.gcd b c = 3) ∧ 
    (is_min_sum a b c) ∧ 
    (a + b + c = 31) :=
sorry

end min_sum_abc_l3186_318611


namespace max_value_theorem_l3186_318601

theorem max_value_theorem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 2) :
  (x^2 * y) / (x + y) + (y^2 * z) / (y + z) + (z^2 * x) / (z + x) ≤ 1 ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧
    (x^2 * y) / (x + y) + (y^2 * z) / (y + z) + (z^2 * x) / (z + x) = 1 :=
by sorry

end max_value_theorem_l3186_318601


namespace simplest_form_sum_l3186_318646

theorem simplest_form_sum (a b : ℕ) (h : a = 63 ∧ b = 117) :
  let (n, d) := (a / gcd a b, b / gcd a b)
  n + d = 20 := by
sorry

end simplest_form_sum_l3186_318646


namespace average_score_is_106_l3186_318669

/-- The average bowling score of three people -/
def average_bowling_score (score1 score2 score3 : ℕ) : ℚ :=
  (score1 + score2 + score3 : ℚ) / 3

/-- Theorem: The average bowling score of Gretchen, Mitzi, and Beth is 106 -/
theorem average_score_is_106 :
  average_bowling_score 120 113 85 = 106 := by
  sorry

end average_score_is_106_l3186_318669


namespace martins_bells_l3186_318625

theorem martins_bells (S B : ℤ) : 
  S = B / 3 + B^2 / 4 →
  S + B = 52 →
  B > 0 →
  B = 12 := by
sorry

end martins_bells_l3186_318625


namespace min_nuts_is_480_l3186_318688

/-- Represents the nut-gathering process of three squirrels -/
structure NutGathering where
  n1 : ℕ  -- nuts picked by first squirrel
  n2 : ℕ  -- nuts picked by second squirrel
  n3 : ℕ  -- nuts picked by third squirrel

/-- Checks if the nut distribution satisfies the given conditions -/
def is_valid_distribution (ng : NutGathering) : Prop :=
  let total := ng.n1 + ng.n2 + ng.n3
  let s1_final := (5 * ng.n1) / 6 + ng.n2 / 12 + (3 * ng.n3) / 16
  let s2_final := ng.n1 / 12 + (3 * ng.n2) / 4 + (3 * ng.n3) / 16
  let s3_final := ng.n1 / 12 + ng.n2 / 4 + (5 * ng.n3) / 8
  (s1_final : ℚ) / 5 = (s2_final : ℚ) / 3 ∧
  (s2_final : ℚ) / 3 = (s3_final : ℚ) / 2 ∧
  s1_final * 3 = s2_final * 5 ∧
  s2_final * 2 = s3_final * 3 ∧
  (5 * ng.n1) % 6 = 0 ∧
  ng.n2 % 12 = 0 ∧
  (3 * ng.n3) % 16 = 0 ∧
  ng.n1 % 12 = 0 ∧
  (3 * ng.n2) % 4 = 0 ∧
  ng.n2 % 4 = 0 ∧
  (5 * ng.n3) % 8 = 0

/-- The least possible total number of nuts -/
def min_total_nuts : ℕ := 480

/-- Theorem stating that the minimum total number of nuts is 480 -/
theorem min_nuts_is_480 :
  ∀ ng : NutGathering, is_valid_distribution ng →
    ng.n1 + ng.n2 + ng.n3 ≥ min_total_nuts :=
by sorry

end min_nuts_is_480_l3186_318688


namespace max_ab_value_l3186_318635

/-- Two circles C₁ and C₂ -/
structure Circles where
  a : ℝ
  b : ℝ

/-- C₁: (x-a)² + (y+2)² = 4 -/
def C₁ (c : Circles) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y + 2)^2 = 4

/-- C₂: (x+b)² + (y+2)² = 1 -/
def C₂ (c : Circles) (x y : ℝ) : Prop :=
  (x + c.b)^2 + (y + 2)^2 = 1

/-- The circles are externally tangent -/
def externally_tangent (c : Circles) : Prop :=
  c.a + c.b = 3

/-- The maximum value of ab is 9/4 -/
theorem max_ab_value (c : Circles) (h : externally_tangent c) :
  c.a * c.b ≤ 9/4 ∧ ∃ (c' : Circles), externally_tangent c' ∧ c'.a * c'.b = 9/4 :=
sorry

end max_ab_value_l3186_318635


namespace no_linear_term_implies_a_value_l3186_318670

theorem no_linear_term_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ∃ b c d : ℝ, (x^2 + a*x - 2) * (x - 1) = x^3 + b*x^2 + d) → a = -2 := by
  sorry

end no_linear_term_implies_a_value_l3186_318670


namespace tens_digit_of_6_pow_18_l3186_318680

/-- The tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- The tens digit of 6^18 is 1 -/
theorem tens_digit_of_6_pow_18 : tens_digit (6^18) = 1 := by
  sorry

end tens_digit_of_6_pow_18_l3186_318680


namespace odd_function_root_property_l3186_318600

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being a root
def IsRoot (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

-- State the theorem
theorem odd_function_root_property
  (f : ℝ → ℝ) (x₀ : ℝ)
  (h_odd : OddFunction f)
  (h_root : IsRoot (fun x => f x - Real.exp x) x₀) :
  IsRoot (fun x => f x * Real.exp x + 1) (-x₀) := by
sorry

end odd_function_root_property_l3186_318600


namespace plant_beds_calculation_l3186_318649

/-- Calculate the number of plant beds required for given vegetable plantings -/
theorem plant_beds_calculation (bean_seedlings pumpkin_seeds radishes : ℕ)
  (bean_per_row pumpkin_per_row radish_per_row : ℕ)
  (rows_per_bed : ℕ)
  (h1 : bean_seedlings = 64)
  (h2 : pumpkin_seeds = 84)
  (h3 : radishes = 48)
  (h4 : bean_per_row = 8)
  (h5 : pumpkin_per_row = 7)
  (h6 : radish_per_row = 6)
  (h7 : rows_per_bed = 2) :
  (bean_seedlings / bean_per_row + pumpkin_seeds / pumpkin_per_row + radishes / radish_per_row) / rows_per_bed = 14 := by
  sorry

end plant_beds_calculation_l3186_318649


namespace power_of_three_product_fourth_root_l3186_318606

theorem power_of_three_product_fourth_root (x : ℝ) : 
  (3^12 * 3^8)^(1/4) = 81 := by
  sorry

end power_of_three_product_fourth_root_l3186_318606


namespace right_triangle_area_l3186_318665

/-- The area of a right triangle with hypotenuse 14 inches and one 45-degree angle is 49 square inches. -/
theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 14 →
  angle = 45 * (π / 180) →
  let leg := hypotenuse / Real.sqrt 2
  let area := (1 / 2) * leg * leg
  area = 49 := by
sorry

end right_triangle_area_l3186_318665


namespace job_size_ratio_l3186_318615

/-- Given two jobs with different numbers of workers and days, 
    prove that the ratio of work done in the new job to the original job is 3. -/
theorem job_size_ratio (original_workers original_days new_workers new_days : ℕ) 
    (h1 : original_workers = 250)
    (h2 : original_days = 16)
    (h3 : new_workers = 600)
    (h4 : new_days = 20) : 
    (new_workers * new_days) / (original_workers * original_days) = 3 := by
  sorry


end job_size_ratio_l3186_318615


namespace polynomial_evaluation_l3186_318630

theorem polynomial_evaluation :
  let y : ℤ := -2
  (y^3 - y^2 + 2*y + 2 : ℤ) = -14 := by sorry

end polynomial_evaluation_l3186_318630


namespace remainder_theorem_l3186_318619

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 7 * k - 1) :
  (n^2 + 3*n + 4) % 7 = 2 := by sorry

end remainder_theorem_l3186_318619


namespace spinner_probability_l3186_318660

-- Define the spinner sections
def spinner_sections : List ℕ := [3, 6, 1, 4, 8, 10, 2, 7]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := sorry

-- Define a function to count elements satisfying a condition
def count_if (l : List ℕ) (f : ℕ → Prop) : ℕ := sorry

-- Theorem statement
theorem spinner_probability :
  let favorable_outcomes := count_if spinner_sections (λ n => is_prime n ∨ is_odd n)
  let total_outcomes := spinner_sections.length
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end spinner_probability_l3186_318660


namespace share_of_B_l3186_318678

theorem share_of_B (total : ℚ) (a b c : ℚ) : 
  total = 595 → 
  a = (2/3) * b → 
  b = (1/4) * c → 
  a + b + c = total → 
  b = 105 := by
sorry

end share_of_B_l3186_318678


namespace cube_root_of_21952_l3186_318642

theorem cube_root_of_21952 : ∃ n : ℕ, n^3 = 21952 ∧ n = 28 := by
  sorry

end cube_root_of_21952_l3186_318642


namespace zero_is_monomial_l3186_318684

/-- Definition of a monomial as an algebraic expression with only one term -/
def is_monomial (expr : ℚ) : Prop := true

/-- Theorem: 0 is a monomial -/
theorem zero_is_monomial : is_monomial 0 := by
  sorry

end zero_is_monomial_l3186_318684


namespace perfect_squares_existence_l3186_318686

theorem perfect_squares_existence (a : ℕ) (h1 : Odd a) (h2 : a > 17) 
  (h3 : ∃ x : ℕ, 3 * a - 2 = x^2) : 
  ∃ b c : ℕ, b ≠ c ∧ b > 0 ∧ c > 0 ∧ 
    (∃ w x y z : ℕ, a + b = w^2 ∧ a + c = x^2 ∧ b + c = y^2 ∧ a + b + c = z^2) :=
by sorry

end perfect_squares_existence_l3186_318686


namespace max_students_distribution_l3186_318676

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 4860) 
  (h_pencils : pencils = 3645) : 
  (Nat.gcd (2 * pens) (3 * pencils)) / 6 = 202 := by
  sorry

end max_students_distribution_l3186_318676


namespace bug_position_after_2010_jumps_l3186_318655

/-- Represents the points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is odd-numbered -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Calculates the next point based on the current point -/
def next_point (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.two
  | Point.five => Point.two

/-- Calculates the point after n jumps -/
def point_after_jumps (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | Nat.succ m => next_point (point_after_jumps start m)

theorem bug_position_after_2010_jumps :
  point_after_jumps Point.five 2010 = Point.two :=
sorry

end bug_position_after_2010_jumps_l3186_318655


namespace ramanujan_identities_l3186_318653

/-- The function f₂ₙ as defined in Ramanujan's identities -/
def f (n : ℕ) (a b c d : ℝ) : ℝ :=
  (b + c + d)^(2*n) + (a + b + c)^(2*n) + (a - d)^(2*n) - 
  (a + c + d)^(2*n) - (a + b + d)^(2*n) - (b - c)^(2*n)

/-- Ramanujan's identities -/
theorem ramanujan_identities (a b c d : ℝ) (h : a * d = b * c) : 
  f 1 a b c d = 0 ∧ f 2 a b c d = 0 ∧ 64 * (f 3 a b c d) * (f 5 a b c d) = 45 * (f 4 a b c d)^2 := by
  sorry

end ramanujan_identities_l3186_318653


namespace senior_score_is_140_8_l3186_318697

/-- Represents the AHSME exam results at Century High School -/
structure AHSMEResults where
  total_students : ℕ
  average_score : ℝ
  senior_ratio : ℝ
  senior_score_ratio : ℝ

/-- Calculates the mean score of seniors given AHSME results -/
def senior_mean_score (results : AHSMEResults) : ℝ :=
  sorry

/-- Theorem stating that the mean score of seniors is 140.8 -/
theorem senior_score_is_140_8 (results : AHSMEResults)
  (h1 : results.total_students = 120)
  (h2 : results.average_score = 110)
  (h3 : results.senior_ratio = 1 / 1.4)
  (h4 : results.senior_score_ratio = 1.6) :
  senior_mean_score results = 140.8 := by
  sorry

end senior_score_is_140_8_l3186_318697


namespace floor_abs_negative_56_3_l3186_318677

theorem floor_abs_negative_56_3 : ⌊|(-56.3 : ℝ)|⌋ = 56 := by sorry

end floor_abs_negative_56_3_l3186_318677


namespace similar_triangles_hypotenuse_l3186_318657

-- Define the properties of the triangles
def smallTriangleArea : ℝ := 8
def largeTriangleArea : ℝ := 200
def smallTriangleHypotenuse : ℝ := 10

-- Define the theorem
theorem similar_triangles_hypotenuse :
  ∃ (smallLeg1 smallLeg2 largeLeg1 largeLeg2 largeHypotenuse : ℝ),
    -- Conditions for the smaller triangle
    smallLeg1 > 0 ∧ smallLeg2 > 0 ∧
    smallLeg1 * smallLeg2 / 2 = smallTriangleArea ∧
    smallLeg1^2 + smallLeg2^2 = smallTriangleHypotenuse^2 ∧
    -- Conditions for the larger triangle
    largeLeg1 > 0 ∧ largeLeg2 > 0 ∧
    largeLeg1 * largeLeg2 / 2 = largeTriangleArea ∧
    largeLeg1^2 + largeLeg2^2 = largeHypotenuse^2 ∧
    -- Similarity condition
    largeLeg1 / smallLeg1 = largeLeg2 / smallLeg2 ∧
    -- Conclusion
    largeHypotenuse = 50 := by
  sorry

end similar_triangles_hypotenuse_l3186_318657


namespace average_of_first_four_l3186_318683

theorem average_of_first_four (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (h2 : (numbers 3 + numbers 4 + numbers 5) / 3 = 35)
  (h3 : numbers 3 = 25) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 18.75 := by
sorry

end average_of_first_four_l3186_318683


namespace fish_tank_water_l3186_318681

theorem fish_tank_water (current : ℝ) : 
  (current + 7 = 14.75) → current = 7.75 := by
  sorry

end fish_tank_water_l3186_318681


namespace sum_of_coefficients_l3186_318632

def polynomial (x : ℝ) : ℝ :=
  3 * (2 * x^6 - x^5 + 4 * x^3 - 7) - 5 * (x^4 - 2 * x^3 + 3 * x^2 + 1) + 6 * (x^7 - 5)

theorem sum_of_coefficients :
  polynomial 1 = 5 := by sorry

end sum_of_coefficients_l3186_318632


namespace cubic_polynomial_real_root_l3186_318675

/-- Given a cubic polynomial ax³ + 3x² + bx - 125 = 0 where a and b are real numbers,
    and -2 - 3i is a root of this polynomial, the real root of the polynomial is 5/2. -/
theorem cubic_polynomial_real_root (a b : ℝ) : 
  (∃ (z : ℂ), z = -2 - 3*I ∧ a * z^3 + 3 * z^2 + b * z - 125 = 0) →
  (∃ (x : ℝ), a * x^3 + 3 * x^2 + b * x - 125 = 0 ∧ x = 5/2) :=
by sorry

end cubic_polynomial_real_root_l3186_318675


namespace triangle_angle_determinant_l3186_318602

theorem triangle_angle_determinant (θ φ ψ : Real) 
  (h : θ + φ + ψ = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.cos θ, Real.sin θ, 1],
    ![Real.cos φ, Real.sin φ, 1],
    ![Real.cos ψ, Real.sin ψ, 1]
  ]
  Matrix.det M = 0 := by sorry

end triangle_angle_determinant_l3186_318602


namespace smallest_a_value_l3186_318633

/-- Represents a parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating the smallest possible value of a for the given parabola -/
theorem smallest_a_value (p : Parabola) 
  (vertex_x : p.a * (1/3)^2 + p.b * (1/3) + p.c = -5/9) 
  (a_positive : p.a > 0)
  (sum_integer : ∃ n : ℤ, p.a + p.b + p.c = n) :
  p.a ≥ 5/4 ∧ ∃ (q : Parabola), q.a = 5/4 ∧ 
    q.a * (1/3)^2 + q.b * (1/3) + q.c = -5/9 ∧ 
    q.a > 0 ∧ 
    (∃ n : ℤ, q.a + q.b + q.c = n) :=
by sorry

end smallest_a_value_l3186_318633


namespace area_of_five_presentable_set_l3186_318694

/-- A complex number is five-presentable if it can be expressed as w - 1/w for some w with |w| = 5 -/
def FivePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = w - 1 / w

/-- The set of all five-presentable complex numbers -/
def T : Set ℂ :=
  {z : ℂ | FivePresentable z}

/-- The area of the set T -/
noncomputable def area_T : ℝ := sorry

theorem area_of_five_presentable_set :
  area_T = 624 * Real.pi / 25 := by sorry

end area_of_five_presentable_set_l3186_318694


namespace tan_alpha_value_l3186_318662

theorem tan_alpha_value (α : Real) (h : Real.tan α = 3) :
  (1 + Real.cos α ^ 2) / (Real.sin α * Real.cos α + Real.sin α ^ 2) = 11/12 := by
  sorry

end tan_alpha_value_l3186_318662


namespace correct_stratified_sampling_l3186_318668

/-- Represents a stratum in the population -/
structure Stratum where
  size : ℕ
  sample_size : ℕ

/-- Represents the population and sample -/
structure Population where
  total_size : ℕ
  sample_size : ℕ
  strata : List Stratum

/-- Checks if the sampling is stratified -/
def is_stratified_sampling (pop : Population) : Prop :=
  pop.strata.all (fun stratum => 
    stratum.sample_size * pop.total_size = pop.sample_size * stratum.size)

/-- The given population data -/
def school_population : Population :=
  { total_size := 1000
  , sample_size := 40
  , strata := 
    [ { size := 400, sample_size := 16 }  -- Blood type O
    , { size := 250, sample_size := 10 }  -- Blood type A
    , { size := 250, sample_size := 10 }  -- Blood type B
    , { size := 100, sample_size := 4 }   -- Blood type AB
    ]
  }

/-- The main theorem to prove -/
theorem correct_stratified_sampling :
  is_stratified_sampling school_population ∧
  school_population.strata.map (fun s => s.sample_size) = [16, 10, 10, 4] :=
sorry

end correct_stratified_sampling_l3186_318668


namespace cos_difference_value_l3186_318682

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end cos_difference_value_l3186_318682


namespace cyclic_sum_inequality_l3186_318643

theorem cyclic_sum_inequality (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end cyclic_sum_inequality_l3186_318643


namespace table_tennis_players_l3186_318664

theorem table_tennis_players (singles_tables doubles_tables : ℕ) : 
  singles_tables + doubles_tables = 13 → 
  4 * doubles_tables = 2 * singles_tables + 4 → 
  4 * doubles_tables = 20 := by
  sorry

end table_tennis_players_l3186_318664


namespace sum_of_digits_n_n_is_greatest_divisor_l3186_318691

/-- The greatest number that divides 1305, 4665, and 6905 leaving the same remainder -/
def n : ℕ := 1120

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m < 10 then m else (m % 10) + sum_of_digits (m / 10)

/-- Theorem stating that the sum of digits of n is 4 -/
theorem sum_of_digits_n : sum_of_digits n = 4 := by
  sorry

/-- Theorem stating that n is the greatest number that divides 1305, 4665, and 6905 
    leaving the same remainder -/
theorem n_is_greatest_divisor : 
  ∀ m : ℕ, m > n → ¬(∃ r : ℕ, 1305 % m = r ∧ 4665 % m = r ∧ 6905 % m = r) := by
  sorry

end sum_of_digits_n_n_is_greatest_divisor_l3186_318691


namespace intersection_of_A_and_B_l3186_318650

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | -Real.sqrt 3 ≤ x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l3186_318650


namespace smallest_c_value_l3186_318629

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : c ≥ 0) (h_nonneg_d : d ≥ 0)
  (h_cos_eq : ∀ x : ℤ, Real.cos (c * x + d) = Real.cos (17 * x)) :
  c ≥ 17 ∧ ∃ (c' : ℝ), c' ≥ 0 ∧ c' < 17 → ¬(∀ x : ℤ, Real.cos (c' * x + d) = Real.cos (17 * x)) :=
by sorry

end smallest_c_value_l3186_318629


namespace student_arrangements_l3186_318613

/-- The number of students in the row -/
def n : ℕ := 7

/-- Calculate the number of arrangements where two students are adjacent -/
def arrangements_two_adjacent (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where three students are adjacent -/
def arrangements_three_adjacent (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where two students are adjacent and one student is not at either end -/
def arrangements_two_adjacent_one_not_end (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where three students are together and the other four are together -/
def arrangements_two_groups (n : ℕ) : ℕ := sorry

theorem student_arrangements :
  arrangements_two_adjacent n = 1440 ∧
  arrangements_three_adjacent n = 720 ∧
  arrangements_two_adjacent_one_not_end n = 960 ∧
  arrangements_two_groups n = 288 := by sorry

end student_arrangements_l3186_318613


namespace real_part_of_i_times_3_minus_i_l3186_318692

theorem real_part_of_i_times_3_minus_i : ∃ (z : ℂ), z = Complex.I * (3 - Complex.I) ∧ z.re = 1 := by
  sorry

end real_part_of_i_times_3_minus_i_l3186_318692


namespace absolute_value_inequality_l3186_318698

theorem absolute_value_inequality (x : ℝ) : |2*x + 1| - 2*|x - 1| > 0 ↔ x > (1/4 : ℝ) := by
  sorry

end absolute_value_inequality_l3186_318698


namespace card_73_is_8_l3186_318608

def card_sequence : List String := [
  "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
  "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"
]

def cycle_length : Nat := card_sequence.length

theorem card_73_is_8 : 
  card_sequence[(73 - 1) % cycle_length] = "8" := by
  sorry

end card_73_is_8_l3186_318608


namespace equation_solution_l3186_318612

theorem equation_solution : ∃ x : ℝ, 30 - (5 * 2) = 3 + x ∧ x = 17 := by
  sorry

end equation_solution_l3186_318612


namespace a_in_M_sufficient_not_necessary_for_a_in_N_l3186_318640

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {-1, 0, 1, 2}

theorem a_in_M_sufficient_not_necessary_for_a_in_N :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end a_in_M_sufficient_not_necessary_for_a_in_N_l3186_318640


namespace sum_m_twice_n_l3186_318636

/-- The sum of m and twice n is equal to m + 2n -/
theorem sum_m_twice_n (m n : ℤ) : m + 2*n = m + 2*n := by sorry

end sum_m_twice_n_l3186_318636


namespace calculation_proof_l3186_318623

theorem calculation_proof :
  ((-1 : ℝ) ^ 2021 + |-(Real.sqrt 3)| + (8 : ℝ) ^ (1/3) - Real.sqrt 16 = -3 + Real.sqrt 3) ∧
  (-(1 : ℝ) ^ 2 - (27 : ℝ) ^ (1/3) + |1 - Real.sqrt 2| = -5 + Real.sqrt 2) := by
  sorry

end calculation_proof_l3186_318623


namespace complex_number_quadrant_l3186_318609

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 : ℂ) / (2 + Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end complex_number_quadrant_l3186_318609


namespace complex_square_equality_l3186_318661

theorem complex_square_equality (a b : ℝ) (i : ℂ) 
  (h1 : i^2 = -1) 
  (h2 : a + i = 2 - b*i) : 
  (a + b*i)^2 = 3 - 4*i := by
sorry

end complex_square_equality_l3186_318661
