import Mathlib

namespace books_returned_percentage_l208_20873

-- Define the initial number of books
def initial_books : ℕ := 75

-- Define the final number of books
def final_books : ℕ := 63

-- Define the number of books loaned out (rounded to 40)
def loaned_books : ℕ := 40

-- Define the percentage of books returned
def percentage_returned : ℚ := 70

-- Theorem statement
theorem books_returned_percentage :
  (((initial_books - final_books : ℚ) / loaned_books) * 100 = percentage_returned) :=
sorry

end books_returned_percentage_l208_20873


namespace unique_three_digit_number_l208_20824

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Returns the two-digit number formed by removing the first digit -/
def remove_first_digit (n : ThreeDigitNumber) : ℕ :=
  n.value % 100

/-- Checks if a three-digit number satisfies the division condition -/
def satisfies_division_condition (n : ThreeDigitNumber) : Prop :=
  let two_digit := remove_first_digit n
  n.value / two_digit = 8 ∧ n.value % two_digit = 6

theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber, satisfies_division_condition n ∧ n.value = 342 :=
sorry

end unique_three_digit_number_l208_20824


namespace ratio_change_l208_20859

theorem ratio_change (x y : ℚ) : 
  x / y = 3 / 4 → 
  y = 40 → 
  (x + 10) / (y + 10) = 4 / 5 := by
sorry

end ratio_change_l208_20859


namespace sphere_surface_area_circumscribing_unit_cube_l208_20805

/-- The surface area of a sphere that circumscribes a cube with edge length 1 is 3π. -/
theorem sphere_surface_area_circumscribing_unit_cube : 
  ∃ (S : ℝ), S = 3 * Real.pi ∧ 
  (∃ (r : ℝ), r > 0 ∧ 
    -- The radius is half the length of the cube's space diagonal
    r = (Real.sqrt 3) / 2 ∧ 
    -- The surface area formula
    S = 4 * Real.pi * r^2) := by
  sorry

end sphere_surface_area_circumscribing_unit_cube_l208_20805


namespace degree_of_polynomial_l208_20821

-- Define the polynomial
def p (a b : ℝ) : ℝ := 3 * a^2 - a * b^2 + 2 * a^2 - 3^4

-- Theorem statement
theorem degree_of_polynomial :
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), (∃ (a b : ℝ), p a b ≠ 0 ∧ 
    (∀ (c d : ℝ), a^m * b^(n-m) = c^m * d^(n-m) → p a b = p c d)) →
  (∀ (k : ℕ), k > n → 
    (∀ (a b : ℝ), ∃ (c d : ℝ), a^k * b^(n-k) = c^k * d^(n-k) ∧ p a b = p c d))) :=
sorry

end degree_of_polynomial_l208_20821


namespace donation_problem_l208_20862

theorem donation_problem (total_donation_A total_donation_B : ℝ)
  (percent_more : ℝ) (diff_avg_donation : ℝ)
  (h1 : total_donation_A = 1200)
  (h2 : total_donation_B = 1200)
  (h3 : percent_more = 0.2)
  (h4 : diff_avg_donation = 5) :
  ∃ (students_A students_B : ℕ),
    students_A = 48 ∧ 
    students_B = 40 ∧
    students_A = (1 + percent_more) * students_B ∧
    (total_donation_B / students_B) - (total_donation_A / students_A) = diff_avg_donation :=
by
  sorry


end donation_problem_l208_20862


namespace third_month_sale_l208_20883

/-- Proves that the sale in the third month is 6855 given the conditions of the problem -/
theorem third_month_sale (sales : Fin 6 → ℕ) : 
  (sales 0 = 6335) → 
  (sales 1 = 6927) → 
  (sales 3 = 7230) → 
  (sales 4 = 6562) → 
  (sales 5 = 5091) → 
  ((sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 6500) → 
  sales 2 = 6855 := by
sorry

end third_month_sale_l208_20883


namespace a_sixth_bounds_l208_20865

-- Define the condition
def condition (a : ℝ) : Prop := a^5 - a^3 + a = 2

-- State the theorem
theorem a_sixth_bounds {a : ℝ} (h : condition a) : 3 < a^6 ∧ a^6 < 4 := by
  sorry

end a_sixth_bounds_l208_20865


namespace baseball_game_attendance_difference_proof_baseball_game_attendance_difference_l208_20844

theorem baseball_game_attendance_difference : ℕ → Prop :=
  fun difference =>
    ∀ (second_game_attendance : ℕ)
      (first_game_attendance : ℕ)
      (third_game_attendance : ℕ)
      (last_week_total : ℕ),
    second_game_attendance = 80 →
    first_game_attendance = second_game_attendance - 20 →
    third_game_attendance = second_game_attendance + 15 →
    last_week_total = 200 →
    difference = (first_game_attendance + second_game_attendance + third_game_attendance) - last_week_total →
    difference = 35

-- The proof of the theorem
theorem proof_baseball_game_attendance_difference : 
  baseball_game_attendance_difference 35 := by
  sorry

end baseball_game_attendance_difference_proof_baseball_game_attendance_difference_l208_20844


namespace suresh_work_hours_l208_20815

/-- Proves that Suresh worked for 9 hours given the conditions of the problem -/
theorem suresh_work_hours 
  (suresh_rate : ℚ) 
  (ashutosh_rate : ℚ) 
  (ashutosh_remaining_hours : ℚ) 
  (h1 : suresh_rate = 1 / 15)
  (h2 : ashutosh_rate = 1 / 20)
  (h3 : ashutosh_remaining_hours = 8)
  : ∃ x : ℚ, x * suresh_rate + ashutosh_remaining_hours * ashutosh_rate = 1 ∧ x = 9 := by
  sorry

end suresh_work_hours_l208_20815


namespace sixth_term_of_special_sequence_l208_20807

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sixth_term_of_special_sequence :
  ∀ (a : ℕ → ℝ),
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 = 2 →
  a 3 = 2 →
  a 4 = 2 →
  a 5 = 2 →
  a 6 = 2 :=
by sorry

end sixth_term_of_special_sequence_l208_20807


namespace equation_solution_l208_20802

theorem equation_solution : 
  ∃ y : ℚ, (40 / 70)^2 = Real.sqrt (y / 70) → y = 17920 / 2401 := by
  sorry

end equation_solution_l208_20802


namespace symmetric_sine_graph_l208_20806

theorem symmetric_sine_graph (φ : Real) : 
  (-Real.pi / 2 < φ ∧ φ < Real.pi / 2) →
  (∀ x, Real.sin (2 * x + φ) = Real.sin (2 * (2 * Real.pi / 3 - x) + φ)) →
  φ = -Real.pi / 6 := by
sorry

end symmetric_sine_graph_l208_20806


namespace profit_percent_calculation_l208_20892

theorem profit_percent_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.9 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = (1 / 9) * 100 := by
sorry

end profit_percent_calculation_l208_20892


namespace sphere_volume_from_surface_area_l208_20891

/-- Given a sphere with surface area 12π, its volume is 4√3π -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 12 * π → (4 / 3) * π * r^3 = 4 * Real.sqrt 3 * π := by
  sorry

end sphere_volume_from_surface_area_l208_20891


namespace greatest_common_divisor_of_120_and_m_l208_20876

theorem greatest_common_divisor_of_120_and_m (m : ℕ) : 
  (∃ d₁ d₂ d₃ d₄ : ℕ, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
    {d : ℕ | d ∣ 120 ∧ d ∣ m} = {d₁, d₂, d₃, d₄}) →
  Nat.gcd 120 m = 8 :=
by sorry

end greatest_common_divisor_of_120_and_m_l208_20876


namespace birds_in_tree_l208_20841

/-- Given a tree with an initial number of birds and additional birds that fly up to it,
    prove that the total number of birds is the sum of the initial and additional birds. -/
theorem birds_in_tree (initial_birds additional_birds : ℕ) 
  (h1 : initial_birds = 179)
  (h2 : additional_birds = 38) :
  initial_birds + additional_birds = 217 := by
  sorry

end birds_in_tree_l208_20841


namespace min_value_sum_reciprocals_l208_20890

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  (1 : ℝ) / (2*a + 3*b) + 1 / (2*b + 3*c) + 1 / (2*c + 3*a) ≥ 9/5 := by
  sorry

end min_value_sum_reciprocals_l208_20890


namespace total_driving_hours_l208_20849

/-- Carl's driving schedule --/
structure DrivingSchedule :=
  (mon : ℕ) (tue : ℕ) (wed : ℕ) (thu : ℕ) (fri : ℕ)

/-- Calculate total hours for a week --/
def weeklyHours (s : DrivingSchedule) : ℕ :=
  s.mon + s.tue + s.wed + s.thu + s.fri

/-- Carl's normal schedule --/
def normalSchedule : DrivingSchedule :=
  ⟨2, 3, 4, 2, 5⟩

/-- Carl's schedule after promotion --/
def promotedSchedule : DrivingSchedule :=
  ⟨3, 5, 7, 6, 5⟩

/-- Carl's schedule for the second week with two days off --/
def secondWeekSchedule : DrivingSchedule :=
  ⟨3, 5, 0, 0, 5⟩

theorem total_driving_hours :
  weeklyHours promotedSchedule + weeklyHours secondWeekSchedule = 39 := by
  sorry

#eval weeklyHours promotedSchedule + weeklyHours secondWeekSchedule

end total_driving_hours_l208_20849


namespace sequence_term_l208_20893

def S (n : ℕ) := 2 * n^2 + n

theorem sequence_term (n : ℕ) (h : n > 0) : 
  (∀ k, k > 0 → S k - S (k-1) = 4*k - 1) → 
  S n - S (n-1) = 4*n - 1 :=
sorry

end sequence_term_l208_20893


namespace gift_splitting_l208_20816

theorem gift_splitting (initial_cost : ℝ) (dropout_count : ℕ) (extra_cost : ℝ) : 
  initial_cost = 120 ∧ 
  dropout_count = 4 ∧ 
  extra_cost = 8 →
  ∃ (n : ℕ), 
    n > dropout_count ∧
    initial_cost / (n - dropout_count : ℝ) = initial_cost / n + extra_cost ∧
    n = 10 := by
  sorry

end gift_splitting_l208_20816


namespace modulo_thirteen_equivalence_l208_20846

theorem modulo_thirteen_equivalence : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 13 ∧ 52801 ≡ n [ZMOD 13] ∧ n = 8 := by
  sorry

end modulo_thirteen_equivalence_l208_20846


namespace infinite_fraction_reciprocal_l208_20851

theorem infinite_fraction_reciprocal (y : ℝ) : 
  y = 1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + y)) → 
  1 / ((y + 1) * (y - 2)) = -(Real.sqrt 3) - 2 :=
by sorry

end infinite_fraction_reciprocal_l208_20851


namespace smallest_integer_satisfying_inequality_l208_20823

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 4 / 7 ↔ y ≥ 1 := by
  sorry

end smallest_integer_satisfying_inequality_l208_20823


namespace sixteen_right_triangles_l208_20818

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a right-angled triangle
structure RightTriangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ

-- Function to check if two circles do not intersect
def nonIntersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 > (c1.radius + c2.radius)^2

-- Function to check if a line is tangent to a circle
def isTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (circle : Circle) : Prop :=
  ∃ p : ℝ × ℝ, line p p ∧ 
    let (x, y) := p
    let (cx, cy) := circle.center
    (x - cx)^2 + (y - cy)^2 = circle.radius^2

-- Function to check if a line is a common external tangent
def isCommonExternalTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (c1 c2 : Circle) : Prop :=
  isTangent line c1 ∧ isTangent line c2

-- Function to check if a line is a common internal tangent
def isCommonInternalTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (c1 c2 : Circle) : Prop :=
  isTangent line c1 ∧ isTangent line c2

-- Main theorem
theorem sixteen_right_triangles (c1 c2 : Circle) :
  nonIntersecting c1 c2 →
  ∃! (triangles : Finset RightTriangle),
    triangles.card = 16 ∧
    ∀ t ∈ triangles,
      ∃ (hypotenuse leg1 leg2 internalTangent : ℝ × ℝ → ℝ × ℝ → Prop),
        isCommonExternalTangent hypotenuse c1 c2 ∧
        isTangent leg1 c1 ∧
        isTangent leg2 c2 ∧
        isCommonInternalTangent internalTangent c1 c2 ∧
        (∃ p : ℝ × ℝ, internalTangent p p ∧ leg1 p p ∧ leg2 p p) :=
by
  sorry

end sixteen_right_triangles_l208_20818


namespace gcd_factorial_problem_l208_20809

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 12) / (Nat.factorial 5)) = 2520 := by
  sorry

end gcd_factorial_problem_l208_20809


namespace probability_of_one_hit_l208_20852

/-- Represents a single shot result -/
inductive Shot
| Hit
| Miss

/-- Represents the result of three shots -/
structure ThreeShots :=
  (first second third : Shot)

/-- Counts the number of hits in a ThreeShots -/
def count_hits (shots : ThreeShots) : Nat :=
  match shots with
  | ⟨Shot.Hit, Shot.Hit, Shot.Hit⟩ => 3
  | ⟨Shot.Hit, Shot.Hit, Shot.Miss⟩ => 2
  | ⟨Shot.Hit, Shot.Miss, Shot.Hit⟩ => 2
  | ⟨Shot.Miss, Shot.Hit, Shot.Hit⟩ => 2
  | ⟨Shot.Hit, Shot.Miss, Shot.Miss⟩ => 1
  | ⟨Shot.Miss, Shot.Hit, Shot.Miss⟩ => 1
  | ⟨Shot.Miss, Shot.Miss, Shot.Hit⟩ => 1
  | ⟨Shot.Miss, Shot.Miss, Shot.Miss⟩ => 0

/-- Converts a digit to a Shot -/
def digit_to_shot (d : Nat) : Shot :=
  if d ∈ [1, 2, 3, 4] then Shot.Hit else Shot.Miss

/-- Converts a three-digit number to ThreeShots -/
def number_to_three_shots (n : Nat) : ThreeShots :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ⟨digit_to_shot d1, digit_to_shot d2, digit_to_shot d3⟩

theorem probability_of_one_hit (data : List Nat) : 
  data.length = 20 →
  (data.filter (fun n => count_hits (number_to_three_shots n) = 1)).length = 9 →
  (data.filter (fun n => count_hits (number_to_three_shots n) = 1)).length / data.length = 9 / 20 :=
sorry

end probability_of_one_hit_l208_20852


namespace laura_shirt_count_l208_20863

def pants_count : ℕ := 2
def pants_price : ℕ := 54
def shirt_price : ℕ := 33
def money_given : ℕ := 250
def change_received : ℕ := 10

theorem laura_shirt_count :
  (money_given - change_received - pants_count * pants_price) / shirt_price = 4 := by
  sorry

end laura_shirt_count_l208_20863


namespace problem_solution_l208_20847

theorem problem_solution (x y : ℝ) : 
  x = 0.7 * y →
  x = 210 →
  y = 300 ∧ ¬(∃ k : ℤ, y = 7 * k) := by
  sorry

end problem_solution_l208_20847


namespace quadratic_inequality_solution_range_l208_20812

/-- The range of m for which the quadratic inequality mx^2 - mx + 1 < 0 has a non-empty solution set -/
theorem quadratic_inequality_solution_range :
  {m : ℝ | ∃ x, m * x^2 - m * x + 1 < 0} = {m | m < 0 ∨ m > 4} := by sorry

end quadratic_inequality_solution_range_l208_20812


namespace circle_area_difference_l208_20896

/-- The difference in area between a circle with radius 30 inches and a circle with circumference 60π inches is 0 square inches. -/
theorem circle_area_difference : 
  let r1 : ℝ := 30
  let c2 : ℝ := 60 * Real.pi
  let r2 : ℝ := c2 / (2 * Real.pi)
  let area1 : ℝ := Real.pi * r1^2
  let area2 : ℝ := Real.pi * r2^2
  area1 - area2 = 0 := by sorry

end circle_area_difference_l208_20896


namespace sum_of_cubes_zero_l208_20800

theorem sum_of_cubes_zero (x y : ℝ) (h1 : x + y = 0) (h2 : x * y = -1) : x^3 + y^3 = 0 := by
  sorry

end sum_of_cubes_zero_l208_20800


namespace tom_free_lessons_l208_20898

/-- Calculates the number of free dance lessons given the total number of lessons,
    cost per lesson, and total amount paid. -/
def free_lessons (total_lessons : ℕ) (cost_per_lesson : ℕ) (total_paid : ℕ) : ℕ :=
  total_lessons - (total_paid / cost_per_lesson)

/-- Proves that Tom received 2 free dance lessons given the problem conditions. -/
theorem tom_free_lessons :
  let total_lessons : ℕ := 10
  let cost_per_lesson : ℕ := 10
  let total_paid : ℕ := 80
  free_lessons total_lessons cost_per_lesson total_paid = 2 := by
  sorry


end tom_free_lessons_l208_20898


namespace triangle_perimeter_range_l208_20826

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_perimeter_range (t : Triangle) 
  (h1 : Real.sin (3 * t.B / 2 + π / 4) = Real.sqrt 2 / 2)
  (h2 : t.a + t.c = 2) :
  3 ≤ t.a + t.b + t.c ∧ t.a + t.b + t.c < 4 := by
  sorry

end triangle_perimeter_range_l208_20826


namespace consecutive_interior_equal_parallel_false_l208_20811

-- Define the concept of lines
variable (Line : Type)

-- Define the concept of angles
variable (Angle : Type)

-- Define what it means for lines to be parallel
variable (parallel : Line → Line → Prop)

-- Define what it means for angles to be consecutive interior angles
variable (consecutive_interior : Angle → Angle → Line → Line → Prop)

-- Define what it means for angles to be equal
variable (angle_equal : Angle → Angle → Prop)

-- Statement to be proven false
theorem consecutive_interior_equal_parallel_false :
  ¬(∀ (l1 l2 : Line) (a1 a2 : Angle), 
    consecutive_interior a1 a2 l1 l2 → angle_equal a1 a2 → parallel l1 l2) :=
sorry

end consecutive_interior_equal_parallel_false_l208_20811


namespace actual_distance_calculation_l208_20831

/-- Calculates the actual distance between two towns given map distance, scale, and conversion factor. -/
theorem actual_distance_calculation (map_distance : ℝ) (scale : ℝ) (mile_to_km : ℝ) : 
  map_distance = 20 →
  scale = 5 →
  mile_to_km = 1.60934 →
  map_distance * scale * mile_to_km = 160.934 := by
sorry

end actual_distance_calculation_l208_20831


namespace birthday_crayons_proof_l208_20808

/-- The number of crayons Paul got for his birthday. -/
def birthday_crayons : ℕ := 253

/-- The number of crayons Paul lost or gave away. -/
def lost_crayons : ℕ := 70

/-- The number of crayons Paul had left by the end of the school year. -/
def remaining_crayons : ℕ := 183

/-- Theorem stating that the number of crayons Paul got for his birthday
    is equal to the sum of lost crayons and remaining crayons. -/
theorem birthday_crayons_proof :
  birthday_crayons = lost_crayons + remaining_crayons :=
by sorry

end birthday_crayons_proof_l208_20808


namespace quadratic_function_properties_l208_20828

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : ∀ x, quadratic_function a b c (-x + 1) = quadratic_function a b c (x + 1))
  (h2 : quadratic_function a b c 2 = 0)
  (h3 : ∃! x, quadratic_function a b c x = x) :
  a = -1/2 ∧ b = 1 ∧ c = 0 ∧
  ∃ m n : ℝ, m = -4 ∧ n = 0 ∧
    (∀ x, m ≤ x ∧ x ≤ n → 3*m ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 3*n) :=
sorry

end quadratic_function_properties_l208_20828


namespace baseball_team_wins_l208_20885

theorem baseball_team_wins (total_games wins : ℕ) (h1 : total_games = 130) (h2 : wins = 101) :
  let losses := total_games - wins
  wins - 3 * losses = 14 := by
  sorry

end baseball_team_wins_l208_20885


namespace min_value_xy_plus_x_squared_l208_20825

theorem min_value_xy_plus_x_squared (x y : ℝ) (h1 : x * y > 0) (h2 : x^2 * y = 2) :
  x * y + x^2 ≥ 4 ∧ (x * y + x^2 = 4 ↔ y = 1 ∧ x = Real.sqrt 2) :=
by sorry

end min_value_xy_plus_x_squared_l208_20825


namespace casey_nail_coats_l208_20879

/-- The time it takes to apply and dry one coat of nail polish -/
def coat_time : ℕ := 20 + 20

/-- The total time spent on decorating nails -/
def total_time : ℕ := 120

/-- The number of coats applied to each nail -/
def num_coats : ℕ := total_time / coat_time

theorem casey_nail_coats : num_coats = 3 := by
  sorry

end casey_nail_coats_l208_20879


namespace impossible_partition_l208_20860

theorem impossible_partition : ¬ ∃ (A B C : Finset ℕ),
  (A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (Finset.card A = 3) ∧ (Finset.card B = 3) ∧ (Finset.card C = 3) ∧
  (∃ (a₁ a₂ a₃ : ℕ), A = {a₁, a₂, a₃} ∧ max a₁ (max a₂ a₃) = a₁ + a₂ + a₃ - max a₁ (max a₂ a₃)) ∧
  (∃ (b₁ b₂ b₃ : ℕ), B = {b₁, b₂, b₃} ∧ max b₁ (max b₂ b₃) = b₁ + b₂ + b₃ - max b₁ (max b₂ b₃)) ∧
  (∃ (c₁ c₂ c₃ : ℕ), C = {c₁, c₂, c₃} ∧ max c₁ (max c₂ c₃) = c₁ + c₂ + c₃ - max c₁ (max c₂ c₃)) :=
by
  sorry


end impossible_partition_l208_20860


namespace error_clock_correct_time_fraction_l208_20869

/-- Represents a 24-hour digital clock with a minute display error -/
structure ErrorClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ
  /-- The number of minutes in an hour -/
  minutes_per_hour : ℕ
  /-- The number of minutes with display error per hour -/
  error_minutes_per_hour : ℕ

/-- The fraction of the day the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  (clock.hours_per_day * (clock.minutes_per_hour - clock.error_minutes_per_hour)) /
  (clock.hours_per_day * clock.minutes_per_hour)

/-- Theorem stating the correct time fraction for the given clock -/
theorem error_clock_correct_time_fraction :
  let clock : ErrorClock := {
    hours_per_day := 24,
    minutes_per_hour := 60,
    error_minutes_per_hour := 1
  }
  correct_time_fraction clock = 59 / 60 := by
  sorry

end error_clock_correct_time_fraction_l208_20869


namespace regular_polygon_with_36_degree_central_angle_l208_20884

theorem regular_polygon_with_36_degree_central_angle (n : ℕ) 
  (h : n > 0) 
  (central_angle : ℝ) 
  (h_central_angle : central_angle = 36) : 
  (360 : ℝ) / central_angle = 10 → n = 10 := by
  sorry

end regular_polygon_with_36_degree_central_angle_l208_20884


namespace chord_length_implies_a_value_l208_20836

theorem chord_length_implies_a_value (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*x + a = 0 ∧ a*x + y + 1 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*a*x₁ + a = 0 ∧ 
    x₂^2 + y₂^2 - 2*a*x₂ + a = 0 ∧
    a*x₁ + y₁ + 1 = 0 ∧ 
    a*x₂ + y₂ + 1 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  a = -2 := by
sorry

end chord_length_implies_a_value_l208_20836


namespace adam_marbles_l208_20801

theorem adam_marbles (greg_marbles : ℕ) (greg_more_than_adam : ℕ) 
  (h1 : greg_marbles = 43)
  (h2 : greg_more_than_adam = 14) :
  greg_marbles - greg_more_than_adam = 29 := by
  sorry

end adam_marbles_l208_20801


namespace eggs_per_unit_is_twelve_l208_20899

/-- Represents the number of eggs in one unit -/
def eggs_per_unit : ℕ := 12

/-- Represents the number of units supplied to the first store daily -/
def units_to_first_store : ℕ := 5

/-- Represents the number of eggs supplied to the second store daily -/
def eggs_to_second_store : ℕ := 30

/-- Represents the total number of eggs supplied to both stores in a week -/
def total_eggs_per_week : ℕ := 630

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem stating that the number of eggs in one unit is 12 -/
theorem eggs_per_unit_is_twelve :
  eggs_per_unit * units_to_first_store * days_in_week +
  eggs_to_second_store * days_in_week = total_eggs_per_week :=
by sorry

end eggs_per_unit_is_twelve_l208_20899


namespace exists_m_intersecting_line_and_circle_l208_20887

/-- A line intersects a circle if and only if the distance from the center of the circle to the line is less than the radius of the circle. -/
axiom line_intersects_circle_iff_distance_lt_radius {a b c x₀ y₀ r : ℝ} :
  (∃ x y, a * x + b * y + c = 0 ∧ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2) < r

/-- The theorem stating that there exists an integer m between 2 and 7 (exclusive) such that the line 4x + 3y + 2m = 0 intersects with the circle (x + 3)² + (y - 1)² = 1. -/
theorem exists_m_intersecting_line_and_circle :
  ∃ m : ℤ, 2 < m ∧ m < 7 ∧
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * (m : ℝ) = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) := by
  sorry

end exists_m_intersecting_line_and_circle_l208_20887


namespace white_balls_count_l208_20833

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 10 →
  red = 7 →
  purple = 3 →
  prob_not_red_purple = 9/10 →
  total - (green + yellow + red + purple) = 50 := by
  sorry

#check white_balls_count

end white_balls_count_l208_20833


namespace cube_root_of_negative_two_sqrt_two_l208_20832

theorem cube_root_of_negative_two_sqrt_two (x : ℝ) :
  x = ((-2 : ℝ) ^ (1/2 : ℝ)) → x = ((-2 * (2 ^ (1/2 : ℝ))) ^ (1/3 : ℝ)) :=
by sorry

end cube_root_of_negative_two_sqrt_two_l208_20832


namespace maddie_makeup_palettes_l208_20864

/-- The number of makeup palettes Maddie bought -/
def num_palettes : ℕ := 3

/-- The cost of each makeup palette in dollars -/
def palette_cost : ℚ := 15

/-- The total cost of lipsticks in dollars -/
def lipstick_cost : ℚ := 10

/-- The total cost of hair color boxes in dollars -/
def hair_color_cost : ℚ := 12

/-- The total amount Maddie paid in dollars -/
def total_paid : ℚ := 67

/-- Theorem stating that the number of makeup palettes Maddie bought is correct -/
theorem maddie_makeup_palettes : 
  (num_palettes : ℚ) * palette_cost + lipstick_cost + hair_color_cost = total_paid := by
  sorry

#check maddie_makeup_palettes

end maddie_makeup_palettes_l208_20864


namespace f_not_monotonic_iff_l208_20894

noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 1

def is_not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem f_not_monotonic_iff (k : ℝ) :
  is_not_monotonic f (k - 1) (k + 1) ↔ 1 ≤ k ∧ k < 3/2 := by sorry

end f_not_monotonic_iff_l208_20894


namespace complex_equation_solution_l208_20835

theorem complex_equation_solution : ∃ (a : ℝ) (b c : ℂ),
  a + b + c = 5 ∧
  a * b + b * c + c * a = 7 ∧
  a * b * c = 3 ∧
  (a = 1 ∨ a = 3) :=
by sorry

end complex_equation_solution_l208_20835


namespace skittles_division_l208_20897

theorem skittles_division (total_skittles : ℕ) (num_students : ℕ) (skittles_per_student : ℕ) :
  total_skittles = 27 →
  num_students = 9 →
  total_skittles = num_students * skittles_per_student →
  skittles_per_student = 3 := by
  sorry

end skittles_division_l208_20897


namespace town_population_proof_l208_20803

theorem town_population_proof (new_people : ℕ) (moved_out : ℕ) (years : ℕ) (final_population : ℕ) :
  new_people = 100 →
  moved_out = 400 →
  years = 4 →
  final_population = 60 →
  (∃ original_population : ℕ,
    original_population = 1260 ∧
    final_population = ((original_population + new_people - moved_out) / 2^years)) :=
by sorry

end town_population_proof_l208_20803


namespace pencil_distribution_l208_20820

theorem pencil_distribution (total_pencils : ℕ) (num_students : ℕ) 
  (h1 : total_pencils = 125) (h2 : num_students = 25) :
  total_pencils / num_students = 5 := by
  sorry

end pencil_distribution_l208_20820


namespace cubic_polynomial_root_l208_20817

theorem cubic_polynomial_root (a b : ℝ) : 
  (∃ (x : ℂ), x^3 + a*x^2 - x + b = 0 ∧ x = 2 - 3*I) → 
  (a = 7.5 ∧ b = -45.5) := by
sorry

end cubic_polynomial_root_l208_20817


namespace arithmetic_error_correction_l208_20853

theorem arithmetic_error_correction : ∃! x : ℝ, 3 * x - 4 = x / 3 + 4 := by
  sorry

end arithmetic_error_correction_l208_20853


namespace train_crossing_time_l208_20822

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time : 
  ∀ (train_length : ℝ) (train_speed : ℝ),
  train_length = 120 →
  train_speed = 27 →
  (2 * train_length) / (2 * train_speed * (1000 / 3600)) = 16 := by
  sorry

end train_crossing_time_l208_20822


namespace range_of_R_l208_20889

/-- The polar equation of curve C1 is ρ = R (R > 0) -/
def C1 (R : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = R^2 ∧ R > 0}

/-- The parametric equation of curve C2 is x = 2 + sin²α, y = sin²α -/
def C2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ α : ℝ, p.1 = 2 + Real.sin α ^ 2 ∧ p.2 = Real.sin α ^ 2}

/-- C1 and C2 have common points -/
def have_common_points (R : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ C1 R ∧ p ∈ C2

theorem range_of_R :
  ∀ R : ℝ, have_common_points R ↔ 2 ≤ R ∧ R ≤ Real.sqrt 10 :=
sorry

end range_of_R_l208_20889


namespace unique_magnitude_of_complex_roots_l208_20813

theorem unique_magnitude_of_complex_roots : 
  ∃! r : ℝ, ∃ z : ℂ, z^2 - 4*z + 29 = 0 ∧ Complex.abs z = r :=
sorry

end unique_magnitude_of_complex_roots_l208_20813


namespace solution_set_implies_sum_l208_20845

/-- If the solution set of x² - mx - 6n < 0 is {x | -3 < x < 6}, then m + n = 6 -/
theorem solution_set_implies_sum (m n : ℝ) : 
  (∀ x, x^2 - m*x - 6*n < 0 ↔ -3 < x ∧ x < 6) → 
  m + n = 6 := by
sorry

end solution_set_implies_sum_l208_20845


namespace right_triangle_sine_cosine_inequality_l208_20870

theorem right_triangle_sine_cosine_inequality 
  (A B C : Real) 
  (h_right_angle : C = Real.pi / 2) 
  (h_acute_A : 0 < A ∧ A < Real.pi / 4) :
  Real.sin B > Real.cos B := by
  sorry

end right_triangle_sine_cosine_inequality_l208_20870


namespace hyperbola_k_range_l208_20895

-- Define the curve
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k + 2) - y^2 / (6 - 2*k) = 1

-- Define the range of k
def k_range (k : ℝ) : Prop := -2 < k ∧ k < 3

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k_range k := by sorry

end hyperbola_k_range_l208_20895


namespace optimal_viewing_distance_l208_20819

/-- The optimal distance from which to view a painting -/
theorem optimal_viewing_distance (a b : ℝ) (ha : a > 0) (hb : b > a) :
  ∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → 
    (b - a) / (y + a * b / y) ≤ (b - a) / (x + a * b / x) :=
by
  -- The proof goes here
  sorry

end optimal_viewing_distance_l208_20819


namespace absolute_value_plus_power_l208_20848

theorem absolute_value_plus_power : |-5| + 2^0 = 6 := by
  sorry

end absolute_value_plus_power_l208_20848


namespace remainder_difference_l208_20858

theorem remainder_difference (m n : ℕ) (hm : m % 6 = 2) (hn : n % 6 = 3) (h_gt : m > n) :
  (m - n) % 6 = 5 := by
  sorry

end remainder_difference_l208_20858


namespace greatest_integer_difference_l208_20861

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) :
  ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 8 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end greatest_integer_difference_l208_20861


namespace meter_to_step_conversion_l208_20874

-- Define our units of measurement
variable (hops skips jumps steps meters : ℚ)

-- Define the relationships between units
variable (hop_skip_relation : 2 * hops = 3 * skips)
variable (jump_hop_relation : 4 * jumps = 6 * hops)
variable (jump_meter_relation : 5 * jumps = 20 * meters)
variable (skip_step_relation : 15 * skips = 10 * steps)

-- State the theorem
theorem meter_to_step_conversion :
  1 * meters = 3/8 * steps :=
sorry

end meter_to_step_conversion_l208_20874


namespace solution_set_of_inequalities_l208_20857

theorem solution_set_of_inequalities :
  let S := {x : ℝ | x - 1 < 0 ∧ |x| < 2}
  S = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end solution_set_of_inequalities_l208_20857


namespace power_of_power_three_l208_20830

theorem power_of_power_three : (3^3)^(3^3) = 27^27 := by
  sorry

end power_of_power_three_l208_20830


namespace line_passes_through_fixed_point_l208_20827

/-- A line in the form kx - y + 1 = 3k passes through the point (3, 1) for all values of k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), k * 3 - 1 + 1 = 3 * k := by sorry

end line_passes_through_fixed_point_l208_20827


namespace one_cubic_yard_equals_27_cubic_feet_l208_20871

-- Define the conversion rate between yards and feet
def yard_to_feet : ℝ := 3

-- Define a cubic yard in terms of cubic feet
def cubic_yard_to_cubic_feet : ℝ := yard_to_feet ^ 3

-- Theorem statement
theorem one_cubic_yard_equals_27_cubic_feet :
  cubic_yard_to_cubic_feet = 27 := by
  sorry

end one_cubic_yard_equals_27_cubic_feet_l208_20871


namespace impossibility_theorem_l208_20881

/-- Represents the number of boxes -/
def n : ℕ := 100

/-- Represents the initial number of stones in each box -/
def initial_stones (i : ℕ) : ℕ := i

/-- Represents the condition for moving stones between boxes -/
def can_move (a b : ℕ) : Prop := a + b = 101

/-- Represents the desired final configuration -/
def desired_config (stones : ℕ → ℕ) : Prop :=
  stones 70 = 69 ∧ stones 50 = 51 ∧ 
  ∀ i, i ≠ 70 ∧ i ≠ 50 → stones i = initial_stones i

/-- Main theorem: It's impossible to achieve the desired configuration -/
theorem impossibility_theorem :
  ¬ ∃ (stones : ℕ → ℕ), 
    (∀ i j, i ≠ j → can_move (stones i) (stones j) → 
      ∃ k l, k ≠ l ∧ stones k + stones l = 101) ∧
    desired_config stones :=
sorry

end impossibility_theorem_l208_20881


namespace circles_properties_l208_20878

-- Define the circles O and M
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_M A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_M B.1 B.2 ∧
  A ≠ B

-- Define the theorem
theorem circles_properties 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) :
  (∃ (T1 T2 : ℝ × ℝ), T1 ≠ T2 ∧ 
    (∀ x y, circle_O x y → (x - T1.1) * T1.1 + (y - T1.2) * T1.2 = 0) ∧
    (∀ x y, circle_M x y → (x - T1.1) * T1.1 + (y - T1.2) * T1.2 = 0) ∧
    (∀ x y, circle_O x y → (x - T2.1) * T2.1 + (y - T2.2) * T2.2 = 0) ∧
    (∀ x y, circle_M x y → (x - T2.1) * T2.1 + (y - T2.2) * T2.2 = 0)) ∧
  (∀ x y, circle_O x y ↔ circle_M (2*A.1 - x) (2*A.2 - y)) ∧
  (∃ E F : ℝ × ℝ, circle_O E.1 E.2 ∧ circle_M F.1 F.2 ∧
    ∀ E' F' : ℝ × ℝ, circle_O E'.1 E'.2 → circle_M F'.1 F'.2 →
      (E.1 - F.1)^2 + (E.2 - F.2)^2 ≥ (E'.1 - F'.1)^2 + (E'.2 - F'.2)^2) ∧
  (∃ E F : ℝ × ℝ, circle_O E.1 E.2 ∧ circle_M F.1 F.2 ∧
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = (4 + Real.sqrt 5)^2) :=
sorry

end circles_properties_l208_20878


namespace rectangle_dimensions_l208_20842

theorem rectangle_dimensions (vertical_side : ℝ) (square_side : ℝ) (horizontal_side : ℝ) : 
  vertical_side = 28 →
  square_side = 10 →
  (vertical_side - square_side) ^ 2 + (horizontal_side - square_side) ^ 2 = vertical_side ^ 2 →
  horizontal_side = 45 :=
by sorry

end rectangle_dimensions_l208_20842


namespace domain_of_f_l208_20810

-- Define the function f
def f (x : ℝ) : ℝ := (x - 5) ^ (1/4) + (x - 6) ^ (1/5) + (x - 7) ^ (1/2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 7} :=
by
  sorry

-- Note: The proof is omitted as per instructions

end domain_of_f_l208_20810


namespace cosine_graph_shift_l208_20880

theorem cosine_graph_shift (x : ℝ) :
  4 * Real.cos (2 * (x - π/8) + π/4) = 4 * Real.cos (2 * x) := by
  sorry

end cosine_graph_shift_l208_20880


namespace imaginary_power_minus_fraction_l208_20872

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_minus_fraction : i^7 - 2 / i = i := by sorry

end imaginary_power_minus_fraction_l208_20872


namespace gcd_108_45_is_9_l208_20843

theorem gcd_108_45_is_9 : Nat.gcd 108 45 = 9 := by
  -- Euclidean algorithm
  have h1 : 108 = 2 * 45 + 18 := by sorry
  have h2 : 45 = 2 * 18 + 9 := by sorry
  have h3 : 18 = 2 * 9 := by sorry

  -- Method of successive subtraction
  have s1 : 108 - 45 = 63 := by sorry
  have s2 : 63 - 45 = 18 := by sorry
  have s3 : 45 - 18 = 27 := by sorry
  have s4 : 27 - 18 = 9 := by sorry
  have s5 : 18 - 9 = 9 := by sorry

  sorry -- Proof to be completed

end gcd_108_45_is_9_l208_20843


namespace absolute_value_difference_l208_20882

theorem absolute_value_difference : |-3 * (7 - 15)| - |(5 - 7)^2 + (-4)^2| = 4 := by sorry

end absolute_value_difference_l208_20882


namespace rain_ratio_proof_l208_20868

/-- Proves that the ratio of rain time on the third day to the second day is 2:1 -/
theorem rain_ratio_proof (first_day : ℕ) (second_day : ℕ) (total_time : ℕ) :
  first_day = 10 →
  second_day = first_day + 2 →
  total_time = 46 →
  ∃ (third_day : ℕ), 
    first_day + second_day + third_day = total_time ∧
    third_day = 2 * second_day :=
by
  sorry

end rain_ratio_proof_l208_20868


namespace geometric_arithmetic_sequence_problem_l208_20839

theorem geometric_arithmetic_sequence_problem 
  (a b : ℕ → ℝ)
  (h_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n)
  (h_arithmetic : ∀ n : ℕ, b (n + 2) - b (n + 1) = b (n + 1) - b n)
  (h_a_product : a 1 * a 5 * a 9 = -8)
  (h_b_sum : b 2 + b 5 + b 8 = 6 * Real.pi)
  : Real.cos ((b 4 + b 6) / (1 - a 3 * a 7)) = -1/2 := by
  sorry

end geometric_arithmetic_sequence_problem_l208_20839


namespace equal_numbers_product_l208_20866

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 25 →
  c = 18 →
  d = e →
  d * e = 506.25 := by
  sorry

end equal_numbers_product_l208_20866


namespace smallest_prime_digit_sum_23_l208_20838

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem stating that 499 is the smallest prime whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 : 
  (is_prime 499 ∧ digit_sum 499 = 23) ∧ 
  ∀ n : ℕ, n < 499 → ¬(is_prime n ∧ digit_sum n = 23) := by sorry

end smallest_prime_digit_sum_23_l208_20838


namespace combined_population_l208_20804

/-- The combined population of Port Perry and Lazy Harbor given the specified conditions -/
theorem combined_population (wellington_pop : ℕ) (port_perry_pop : ℕ) (lazy_harbor_pop : ℕ) 
  (h1 : port_perry_pop = 7 * wellington_pop)
  (h2 : port_perry_pop = lazy_harbor_pop + 800)
  (h3 : wellington_pop = 900) : 
  port_perry_pop + lazy_harbor_pop = 11800 := by
  sorry

end combined_population_l208_20804


namespace driver_net_rate_of_pay_l208_20886

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gas_price = 2.50)
  : (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 := by
  sorry

end driver_net_rate_of_pay_l208_20886


namespace complex_magnitude_l208_20850

theorem complex_magnitude (z : ℂ) (h : z + Complex.I = z * Complex.I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_l208_20850


namespace circle_center_l208_20829

/-- The center of the circle given by the equation x^2 + 10x + y^2 - 14y + 25 = 0 is (-5, 7) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 10*x + y^2 - 14*y + 25 = 0) → 
  (∃ r : ℝ, (x + 5)^2 + (y - 7)^2 = r^2) :=
by sorry

end circle_center_l208_20829


namespace ceiling_floor_product_l208_20875

theorem ceiling_floor_product (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → y ∈ Set.Ioo (-9 : ℝ) (-8 : ℝ) := by
  sorry

end ceiling_floor_product_l208_20875


namespace loan_interest_period_l208_20837

/-- The problem of determining the number of years for B's gain --/
theorem loan_interest_period (principal : ℝ) (rate_A rate_C : ℝ) (gain : ℝ) : 
  principal = 1500 →
  rate_A = 0.10 →
  rate_C = 0.115 →
  gain = 67.5 →
  (rate_C - rate_A) * principal * 3 = gain :=
by sorry

end loan_interest_period_l208_20837


namespace nonagon_coloring_theorem_l208_20854

/-- A type representing the colors used to color the nonagon vertices -/
inductive Color
| A
| B
| C

/-- A type representing the vertices of a regular nonagon -/
inductive Vertex
| One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- A function type representing a coloring of the nonagon -/
def Coloring := Vertex → Color

/-- Predicate to check if two vertices are adjacent in a regular nonagon -/
def adjacent (v1 v2 : Vertex) : Prop := sorry

/-- Predicate to check if three vertices form an equilateral triangle in a regular nonagon -/
def equilateralTriangle (v1 v2 v3 : Vertex) : Prop := sorry

/-- Predicate to check if a coloring is valid according to the given conditions -/
def validColoring (c : Coloring) : Prop :=
  (∀ v1 v2, adjacent v1 v2 → c v1 ≠ c v2) ∧
  (∀ v1 v2 v3, equilateralTriangle v1 v2 v3 → c v1 ≠ c v2 ∧ c v1 ≠ c v3 ∧ c v2 ≠ c v3)

/-- The minimum number of colors needed for a valid coloring -/
def m : Nat := 3

/-- The total number of valid colorings using m colors -/
def n : Nat := 18

/-- The main theorem stating that the product of m and n is 54 -/
theorem nonagon_coloring_theorem : m * n = 54 := by sorry

end nonagon_coloring_theorem_l208_20854


namespace concyclic_roots_l208_20856

theorem concyclic_roots (m : ℝ) : 
  (∀ x : ℂ, (x^2 - 2*x + 2 = 0 ∨ x^2 + 2*m*x + 1 = 0) → 
    (∃ (a b r : ℝ), ∀ y : ℂ, (y^2 - 2*y + 2 = 0 ∨ y^2 + 2*m*y + 1 = 0) → 
      (y.re - a)^2 + (y.im - b)^2 = r^2)) ↔ 
  (-1 < m ∧ m < 1) ∨ m = -3/2 := by
sorry

end concyclic_roots_l208_20856


namespace max_integer_k_l208_20877

theorem max_integer_k (x y k : ℝ) : 
  x - 4*y = k - 1 →
  2*x + y = k →
  x - y ≤ 0 →
  ∀ m : ℤ, m ≤ k → m ≤ 0 :=
by sorry

end max_integer_k_l208_20877


namespace stickers_used_for_decoration_l208_20867

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 26
def birthday_stickers : ℕ := 20
def given_away_stickers : ℕ := 6
def left_stickers : ℕ := 2

theorem stickers_used_for_decoration :
  initial_stickers + bought_stickers + birthday_stickers - given_away_stickers - left_stickers = 58 :=
by sorry

end stickers_used_for_decoration_l208_20867


namespace product_adjacent_faces_is_144_l208_20840

/-- Represents a face of the cube --/
structure Face :=
  (number : Nat)

/-- Represents the cube formed from the numbered net --/
structure Cube :=
  (faces : List Face)
  (adjacent_to_one : List Face)
  (h_adjacent : adjacent_to_one.length = 4)

/-- The product of the numbers on the faces adjacent to face 1 --/
def product_adjacent_faces (c : Cube) : Nat :=
  c.adjacent_to_one.map Face.number |>.foldl (· * ·) 1

/-- Theorem stating that the product of numbers on faces adjacent to face 1 is 144 --/
theorem product_adjacent_faces_is_144 (c : Cube) 
  (h_adjacent_numbers : c.adjacent_to_one.map Face.number = [2, 3, 4, 6]) :
  product_adjacent_faces c = 144 := by
  sorry

end product_adjacent_faces_is_144_l208_20840


namespace difference_of_reciprocals_l208_20855

theorem difference_of_reciprocals (p q : ℚ) : 
  3 / p = 6 → 3 / q = 18 → p - q = 1 / 3 := by
  sorry

end difference_of_reciprocals_l208_20855


namespace dividend_calculation_l208_20888

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 8)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 141 := by
  sorry

end dividend_calculation_l208_20888


namespace max_non_managers_l208_20834

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 5 / 24 →
  non_managers ≤ 38 :=
by sorry

end max_non_managers_l208_20834


namespace problem_statement_l208_20814

theorem problem_statement (f : ℝ → ℝ) : 
  (∀ x, f x = (x^4 + 2*x^3 + 4*x - 5)^2004 + 2004) →
  f (Real.sqrt 3 - 1) = 2005 := by
sorry

end problem_statement_l208_20814
