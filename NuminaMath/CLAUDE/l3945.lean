import Mathlib

namespace NUMINAMATH_CALUDE_credibility_is_97_5_percent_l3945_394526

/-- Critical values table -/
def critical_values : List (Float × Float) := [
  (0.15, 2.072),
  (0.10, 2.706),
  (0.05, 3.841),
  (0.025, 5.024),
  (0.010, 6.635),
  (0.001, 10.828)
]

/-- The calculated K^2 value -/
def K_squared : Float := 6.109

/-- Function to determine credibility based on K^2 value and critical values table -/
def determine_credibility (K_sq : Float) (crit_vals : List (Float × Float)) : Float :=
  let lower_bound := crit_vals.find? (fun (p, k) => K_sq > k)
  let upper_bound := crit_vals.find? (fun (p, k) => K_sq ≤ k)
  match lower_bound, upper_bound with
  | some (p_lower, _), some (p_upper, _) => 100 * (1 - p_lower)
  | _, _ => 0  -- Default case if bounds are not found

/-- Theorem stating the credibility of the relationship -/
theorem credibility_is_97_5_percent :
  determine_credibility K_squared critical_values = 97.5 :=
sorry

end NUMINAMATH_CALUDE_credibility_is_97_5_percent_l3945_394526


namespace NUMINAMATH_CALUDE_owen_final_turtles_l3945_394583

/-- Represents the number of turtles each person has at different times --/
structure TurtleCount where
  owen_initial : ℕ
  johanna_initial : ℕ
  owen_after_month : ℕ
  johanna_after_month : ℕ
  owen_final : ℕ

/-- Calculates the final number of turtles Owen has --/
def calculate_final_turtles (t : TurtleCount) : Prop :=
  t.owen_initial = 21 ∧
  t.johanna_initial = t.owen_initial - 5 ∧
  t.owen_after_month = 2 * t.owen_initial ∧
  t.johanna_after_month = t.johanna_initial / 2 ∧
  t.owen_final = t.owen_after_month + t.johanna_after_month ∧
  t.owen_final = 50

theorem owen_final_turtles :
  ∃ t : TurtleCount, calculate_final_turtles t :=
sorry

end NUMINAMATH_CALUDE_owen_final_turtles_l3945_394583


namespace NUMINAMATH_CALUDE_sock_profit_percentage_l3945_394531

/-- Calculates the percentage profit on 4 pairs of socks given the following conditions:
  * 9 pairs of socks were bought
  * Each pair costs $2
  * $0.2 profit is made on 5 pairs
  * Total profit is $3
-/
theorem sock_profit_percentage 
  (total_pairs : Nat) 
  (cost_per_pair : ℚ) 
  (profit_on_five : ℚ) 
  (total_profit : ℚ) 
  (h1 : total_pairs = 9)
  (h2 : cost_per_pair = 2)
  (h3 : profit_on_five = 5 * (1 / 5))
  (h4 : total_profit = 3) :
  let remaining_pairs := total_pairs - 5
  let remaining_profit := total_profit - profit_on_five
  let remaining_cost := remaining_pairs * cost_per_pair
  (remaining_profit / remaining_cost) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_sock_profit_percentage_l3945_394531


namespace NUMINAMATH_CALUDE_unique_solution_trig_equation_l3945_394502

theorem unique_solution_trig_equation :
  ∃! (n : ℕ+), Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (2 * n.val) / 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trig_equation_l3945_394502


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l3945_394565

/-- The probability of getting exactly 3 successes in 7 independent trials,
    where each trial has a success probability of 3/8. -/
theorem magic_8_ball_probability : 
  (Nat.choose 7 3 : ℚ) * (3/8)^3 * (5/8)^4 = 590625/2097152 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l3945_394565


namespace NUMINAMATH_CALUDE_employed_males_percentage_l3945_394555

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 60)
  (h2 : employed_females_percentage = 75)
  : (employed_percentage / 100 * (1 - employed_females_percentage / 100) * 100 = 15) := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l3945_394555


namespace NUMINAMATH_CALUDE_hoseok_has_least_paper_l3945_394506

def jungkook_paper : ℕ := 10
def hoseok_paper : ℕ := 7
def seokjin_paper : ℕ := jungkook_paper - 2

theorem hoseok_has_least_paper : 
  hoseok_paper < jungkook_paper ∧ hoseok_paper < seokjin_paper := by
sorry

end NUMINAMATH_CALUDE_hoseok_has_least_paper_l3945_394506


namespace NUMINAMATH_CALUDE_ramsey_33_l3945_394516

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a color (Red or Blue) -/
inductive Color
  | Red
  | Blue

/-- A function type representing a coloring of line segments -/
def Coloring := Fin 9 → Fin 9 → Color

/-- Predicate to check if four points are coplanar -/
def are_coplanar (p₁ p₂ p₃ p₄ : Point3D) : Prop := sorry

/-- Predicate to check if a set of points forms a monochromatic triangle under a given coloring -/
def has_monochromatic_triangle (points : Fin 9 → Point3D) (coloring : Coloring) : Prop := sorry

theorem ramsey_33 (points : Fin 9 → Point3D) 
  (h_not_coplanar : ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l → 
    ¬are_coplanar (points i) (points j) (points k) (points l)) :
  ∀ coloring : Coloring, has_monochromatic_triangle points coloring := by
  sorry

end NUMINAMATH_CALUDE_ramsey_33_l3945_394516


namespace NUMINAMATH_CALUDE_circle_equations_l3945_394564

-- Define the parallel lines
def line1 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + y + Real.sqrt 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 2 * Real.sqrt 2 * a = 0

-- Define the circle N
def circleN (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define point B
def pointB : ℝ × ℝ := (3, -2)

-- Define the line of symmetry
def lineSymmetry (x : ℝ) : Prop := x = -1

-- Define point C
def pointC : ℝ × ℝ := (-5, -2)

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 5)^2 + (y + 2)^2 = 49

-- Theorem statement
theorem circle_equations :
  ∃ (a : ℝ),
    (∀ x y, line1 a x y ↔ line2 a x y) →
    (∀ x y, circleN x y) →
    (pointC.1 = -pointB.1 - 2 ∧ pointC.2 = pointB.2) →
    (∀ x y, circleC x y) ∧
    (∃ x y, circleN x y ∧ circleC x y ∧
      (x - 3)^2 + (y - 4)^2 + ((x + 5)^2 + (y + 2)^2).sqrt = 10) :=
by sorry

end NUMINAMATH_CALUDE_circle_equations_l3945_394564


namespace NUMINAMATH_CALUDE_school2_selection_l3945_394578

/-- Represents the number of students selected from a school in a system sampling. -/
def studentsSelected (schoolSize totalStudents selectedStudents : ℕ) : ℚ :=
  (schoolSize : ℚ) * (selectedStudents : ℚ) / (totalStudents : ℚ)

/-- The main theorem about the number of students selected from School 2. -/
theorem school2_selection :
  let totalStudents : ℕ := 360
  let school1Size : ℕ := 123
  let school2Size : ℕ := 123
  let school3Size : ℕ := 114
  let totalSelected : ℕ := 60
  let remainingSelected : ℕ := totalSelected - 1
  let remainingStudents : ℕ := totalStudents - 1
  Int.ceil (studentsSelected school2Size remainingStudents remainingSelected) = 20 := by
  sorry

#check school2_selection

end NUMINAMATH_CALUDE_school2_selection_l3945_394578


namespace NUMINAMATH_CALUDE_percentage_green_shirts_l3945_394574

/-- The percentage of students wearing green shirts in a school, given the following conditions:
  * The total number of students is 700
  * 45% of students wear blue shirts
  * 23% of students wear red shirts
  * 119 students wear colors other than blue, red, or green
-/
theorem percentage_green_shirts (total : ℕ) (blue_percent red_percent : ℚ) (other : ℕ) :
  total = 700 →
  blue_percent = 45 / 100 →
  red_percent = 23 / 100 →
  other = 119 →
  (((total : ℚ) - (blue_percent * total + red_percent * total + other)) / total) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_green_shirts_l3945_394574


namespace NUMINAMATH_CALUDE_opposite_abs_neg_five_l3945_394569

theorem opposite_abs_neg_five : -(abs (-5)) = -5 := by sorry

end NUMINAMATH_CALUDE_opposite_abs_neg_five_l3945_394569


namespace NUMINAMATH_CALUDE_downstream_distance_l3945_394525

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (travel_time : ℝ)
  (h1 : boat_speed = 22)
  (h2 : stream_speed = 5)
  (h3 : travel_time = 8) :
  boat_speed + stream_speed * travel_time = 216 :=
by
  sorry

#check downstream_distance

end NUMINAMATH_CALUDE_downstream_distance_l3945_394525


namespace NUMINAMATH_CALUDE_prime_product_divisible_by_four_l3945_394548

theorem prime_product_divisible_by_four (p q : ℕ) : 
  Prime p → Prime q → Prime (p * q + 1) → 
  4 ∣ ((2 * p + q) * (p + 2 * q)) := by
sorry

end NUMINAMATH_CALUDE_prime_product_divisible_by_four_l3945_394548


namespace NUMINAMATH_CALUDE_student_council_choices_l3945_394503

/-- Represents the composition of the student council -/
structure StudentCouncil where
  freshmen : Nat
  sophomores : Nat
  juniors : Nat

/-- The given student council composition -/
def council : StudentCouncil := ⟨6, 5, 4⟩

/-- Number of ways to choose one person as president -/
def choosePresident (sc : StudentCouncil) : Nat :=
  sc.freshmen + sc.sophomores + sc.juniors

/-- Number of ways to choose one person from each grade -/
def chooseOneFromEach (sc : StudentCouncil) : Nat :=
  sc.freshmen * sc.sophomores * sc.juniors

/-- Number of ways to choose two people from different grades -/
def chooseTwoFromDifferent (sc : StudentCouncil) : Nat :=
  sc.freshmen * sc.sophomores +
  sc.freshmen * sc.juniors +
  sc.sophomores * sc.juniors

theorem student_council_choices :
  choosePresident council = 15 ∧
  chooseOneFromEach council = 120 ∧
  chooseTwoFromDifferent council = 74 := by
  sorry

end NUMINAMATH_CALUDE_student_council_choices_l3945_394503


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3945_394550

/-- Proves that a boat's speed in still water is 51 kmph given the conditions -/
theorem boat_speed_in_still_water 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (stream_speed : ℝ) 
  (h1 : upstream_time = 2 * downstream_time)
  (h2 : stream_speed = 17) : 
  ∃ (boat_speed : ℝ), boat_speed = 51 ∧ 
    (boat_speed + stream_speed) * downstream_time = 
    (boat_speed - stream_speed) * upstream_time := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3945_394550


namespace NUMINAMATH_CALUDE_circle_radii_relation_l3945_394589

/-- Given three circles with centers A, B, C, touching each other and a line l,
    with radii a, b, and c respectively, prove that 1/√c = 1/√a + 1/√b. -/
theorem circle_radii_relation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / Real.sqrt c = 1 / Real.sqrt a + 1 / Real.sqrt b := by
sorry

end NUMINAMATH_CALUDE_circle_radii_relation_l3945_394589


namespace NUMINAMATH_CALUDE_find_x_l3945_394510

theorem find_x (x : ℕ+) 
  (n : ℤ) (h_n : n = x.val^2 + 2*x.val + 17)
  (d : ℤ) (h_d : d = 2*x.val + 5)
  (h_div : n = d * x.val + 7) : 
  x.val = 2 := by
sorry

end NUMINAMATH_CALUDE_find_x_l3945_394510


namespace NUMINAMATH_CALUDE_peach_difference_l3945_394551

theorem peach_difference (audrey_peaches paul_peaches : ℕ) 
  (h1 : audrey_peaches = 26) 
  (h2 : paul_peaches = 48) : 
  paul_peaches - audrey_peaches = 22 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3945_394551


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3945_394573

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3945_394573


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3945_394594

theorem pie_eating_contest (a b c : ℚ) 
  (ha : a = 4/5) (hb : b = 5/6) (hc : c = 3/4) : 
  (max a (max b c) - min a (min b c) : ℚ) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3945_394594


namespace NUMINAMATH_CALUDE_yard_length_with_11_trees_l3945_394571

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1) * distanceBetweenTrees

/-- Theorem: The length of a yard with 11 equally spaced trees, 
    with 15 meters between consecutive trees, is 150 meters -/
theorem yard_length_with_11_trees : 
  yardLength 11 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_with_11_trees_l3945_394571


namespace NUMINAMATH_CALUDE_expression_simplification_l3945_394563

theorem expression_simplification (x y : ℚ) (hx : x = 4) (hy : y = -1/4) :
  ((x + y) * (3 * x - y) + y^2) / (-x) = -23/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3945_394563


namespace NUMINAMATH_CALUDE_days_without_calls_l3945_394567

/-- The number of days in the year -/
def total_days : ℕ := 365

/-- The calling frequencies of the three grandchildren -/
def call_frequencies : List ℕ := [4, 6, 8]

/-- Calculate the number of days with at least one call -/
def days_with_calls (frequencies : List ℕ) (total : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem days_without_calls (frequencies : List ℕ) (total : ℕ) :
  frequencies = call_frequencies → total = total_days →
  total - days_with_calls frequencies total = 244 :=
by sorry

end NUMINAMATH_CALUDE_days_without_calls_l3945_394567


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3945_394556

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) :
  k ≠ 0 →
  p ≠ 1 →
  r ≠ 1 →
  p ≠ r →
  a₂ = k * p →
  a₃ = k * p^2 →
  b₂ = k * r →
  b₃ = k * r^2 →
  a₃ - b₃ = 5 * (a₂ - b₂) →
  p + r = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3945_394556


namespace NUMINAMATH_CALUDE_right_angled_triangle_check_l3945_394582

theorem right_angled_triangle_check : 
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt 5
  (a * a + b * b = c * c) ∧ 
  ¬(1 * 1 + 1 * 1 = Real.sqrt 3 * Real.sqrt 3) ∧
  ¬(0.2 * 0.2 + 0.3 * 0.3 = 0.5 * 0.5) ∧
  ¬((1/3) * (1/3) + (1/4) * (1/4) = (1/5) * (1/5)) := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_check_l3945_394582


namespace NUMINAMATH_CALUDE_number_of_bs_l3945_394501

/-- Represents the number of students who earn each grade in a biology class. -/
structure GradeDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The conditions of the biology class grade distribution. -/
def validGradeDistribution (g : GradeDistribution) : Prop :=
  g.a + g.b + g.c + g.d = 40 ∧
  g.a = 12 * g.b / 10 ∧
  g.c = g.b ∧
  g.d = g.b / 2

/-- The theorem stating that the number of B's in the class is 11. -/
theorem number_of_bs (g : GradeDistribution) 
  (h : validGradeDistribution g) : g.b = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bs_l3945_394501


namespace NUMINAMATH_CALUDE_number_problem_l3945_394572

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3945_394572


namespace NUMINAMATH_CALUDE_audio_cassette_count_audio_cassette_count_proof_l3945_394593

/-- Proves that the number of audio cassettes in the first set is 7 given the problem conditions --/
theorem audio_cassette_count : ℕ :=
  let video_cost : ℕ := 300
  let some_audio_and_3_video_cost : ℕ := 1110
  let five_audio_and_4_video_cost : ℕ := 1350
  7

theorem audio_cassette_count_proof :
  let video_cost : ℕ := 300
  let some_audio_and_3_video_cost : ℕ := 1110
  let five_audio_and_4_video_cost : ℕ := 1350
  ∃ (audio_cost : ℕ) (first_set_count : ℕ),
    first_set_count * audio_cost + 3 * video_cost = some_audio_and_3_video_cost ∧
    5 * audio_cost + 4 * video_cost = five_audio_and_4_video_cost ∧
    first_set_count = audio_cassette_count :=
by
  sorry

end NUMINAMATH_CALUDE_audio_cassette_count_audio_cassette_count_proof_l3945_394593


namespace NUMINAMATH_CALUDE_triangle_theorem_l3945_394595

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c) 
  (h2 : t.c = 3 * t.a) : 
  t.B = π/3 ∧ Real.sin t.A = Real.sqrt 21 / 14 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3945_394595


namespace NUMINAMATH_CALUDE_fireflies_problem_l3945_394549

theorem fireflies_problem (initial : ℕ) : 
  (initial + 8 - 2 = 9) → initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_fireflies_problem_l3945_394549


namespace NUMINAMATH_CALUDE_cookies_left_after_sales_l3945_394560

/-- Calculates the number of cookies left after sales throughout the day -/
theorem cookies_left_after_sales (initial : ℕ) (morning_dozens : ℕ) (lunch : ℕ) (afternoon : ℕ) :
  initial = 120 →
  morning_dozens = 3 →
  lunch = 57 →
  afternoon = 16 →
  initial - (morning_dozens * 12 + lunch + afternoon) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_left_after_sales_l3945_394560


namespace NUMINAMATH_CALUDE_factorization_problem_triangle_shape_l3945_394532

-- Problem 1
theorem factorization_problem (a b : ℝ) :
  a^2 - 6*a*b + 9*b^2 - 36 = (a - 3*b - 6) * (a - 3*b + 6) := by sorry

-- Problem 2
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^2 + c^2 + 2*b^2 - 2*a*b - 2*b*c = 0) :
  a = b ∧ b = c := by sorry

end NUMINAMATH_CALUDE_factorization_problem_triangle_shape_l3945_394532


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3945_394570

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_minimum (p q : ℝ) :
  (∀ x, f p q x ≥ f p q q) ∧ 
  (f p q q = (p + q)^2) →
  ((p = 0 ∧ q = 0) ∨ (p = -1 ∧ q = 1/2)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3945_394570


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3945_394557

theorem complex_number_in_fourth_quadrant : ∃ (z : ℂ), z = Complex.mk (Real.sin 3) (Real.cos 3) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3945_394557


namespace NUMINAMATH_CALUDE_sine_inequality_range_l3945_394537

theorem sine_inequality_range (a : ℝ) : 
  (∃ x : ℝ, Real.sin x < a) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_range_l3945_394537


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l3945_394514

theorem prime_square_mod_180 (p : Nat) (h_prime : Prime p) (h_gt_5 : p > 5) :
  p ^ 2 % 180 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l3945_394514


namespace NUMINAMATH_CALUDE_elections_with_past_officers_count_l3945_394546

def total_candidates : ℕ := 16
def past_officers : ℕ := 7
def positions : ℕ := 5

def elections_with_past_officers : ℕ := Nat.choose total_candidates positions - Nat.choose (total_candidates - past_officers) positions

theorem elections_with_past_officers_count : elections_with_past_officers = 4242 := by
  sorry

end NUMINAMATH_CALUDE_elections_with_past_officers_count_l3945_394546


namespace NUMINAMATH_CALUDE_log_sum_two_five_equals_one_l3945_394552

theorem log_sum_two_five_equals_one : Real.log 2 + Real.log 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_two_five_equals_one_l3945_394552


namespace NUMINAMATH_CALUDE_hyperbola_y_axis_condition_l3945_394521

/-- Represents a conic section of the form mx^2 + ny^2 = 1 -/
structure ConicSection (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate for a hyperbola with foci on the y-axis -/
def IsHyperbolaOnYAxis (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0

theorem hyperbola_y_axis_condition (m n : ℝ) :
  (IsHyperbolaOnYAxis m n → m * n < 0) ∧
  ¬(m * n < 0 → IsHyperbolaOnYAxis m n) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_y_axis_condition_l3945_394521


namespace NUMINAMATH_CALUDE_right_triangle_identification_l3945_394509

theorem right_triangle_identification (a b c : ℝ) : 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 9) →
  (a^2 + b^2 = c^2 ↔ a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l3945_394509


namespace NUMINAMATH_CALUDE_ab_equals_e_cubed_l3945_394580

theorem ab_equals_e_cubed (a b : ℝ) (h1 : Real.exp (2 - a) = a) (h2 : b * (Real.log b - 1) = Real.exp 3) : a * b = Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_e_cubed_l3945_394580


namespace NUMINAMATH_CALUDE_rhombus_area_l3945_394597

/-- A rhombus with diagonals of lengths 10 and 30 has an area of 150 -/
theorem rhombus_area (d₁ d₂ area : ℝ) (h₁ : d₁ = 10) (h₂ : d₂ = 30) 
    (h₃ : area = (d₁ * d₂) / 2) : area = 150 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3945_394597


namespace NUMINAMATH_CALUDE_eight_sided_die_probability_l3945_394544

/-- Represents the number of sides on the die -/
def sides : ℕ := 8

/-- Represents the event where the first roll is greater than or equal to the second roll -/
def favorable_outcomes (s : ℕ) : ℕ := (s * (s + 1)) / 2

/-- The probability of the first roll being greater than or equal to the second roll -/
def probability (s : ℕ) : ℚ := (favorable_outcomes s) / (s^2 : ℚ)

/-- Theorem stating that for an 8-sided die, the probability of the first roll being 
    greater than or equal to the second roll is 9/16 -/
theorem eight_sided_die_probability : probability sides = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_eight_sided_die_probability_l3945_394544


namespace NUMINAMATH_CALUDE_rational_a_condition_l3945_394512

theorem rational_a_condition (m n : ℤ) : 
  ∃ (a : ℚ), a = (m^4 + n^4 + m^2*n^2) / (4*m^2*n^2) :=
by sorry

end NUMINAMATH_CALUDE_rational_a_condition_l3945_394512


namespace NUMINAMATH_CALUDE_binary_10110100_is_180_l3945_394522

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10110100_is_180 :
  binary_to_decimal [false, false, true, false, true, true, false, true] = 180 := by
  sorry

end NUMINAMATH_CALUDE_binary_10110100_is_180_l3945_394522


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l3945_394517

/-- The number of y-intercepts for the parabola x = 3y^2 - 6y + 3 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 6 * y + 3
  ∃! y : ℝ, f y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l3945_394517


namespace NUMINAMATH_CALUDE_birds_berries_consumption_l3945_394588

theorem birds_berries_consumption (num_birds : ℕ) (total_berries : ℕ) (num_days : ℕ) 
  (h1 : num_birds = 5)
  (h2 : total_berries = 140)
  (h3 : num_days = 4) :
  total_berries / num_days / num_birds = 7 := by
  sorry

end NUMINAMATH_CALUDE_birds_berries_consumption_l3945_394588


namespace NUMINAMATH_CALUDE_cannot_determine_heavier_l3945_394579

variable (M P O : ℝ)

def mandarin_lighter_than_pear := M < P
def orange_heavier_than_mandarin := O > M

theorem cannot_determine_heavier (h1 : mandarin_lighter_than_pear M P) 
  (h2 : orange_heavier_than_mandarin O M) : 
  ¬(∀ x y : ℝ, (x < y) ∨ (y < x)) :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_heavier_l3945_394579


namespace NUMINAMATH_CALUDE_complement_M_in_U_l3945_394504

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x > 0}

-- Define the set M
def M : Set ℝ := {x : ℝ | x > 1}

-- Define the complement of M in U
def complementMU : Set ℝ := {x : ℝ | x ∈ U ∧ x ∉ M}

-- Theorem statement
theorem complement_M_in_U :
  complementMU = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l3945_394504


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3945_394524

/-- The sum of a geometric sequence with 6 terms, initial term 10, and common ratio 3 is 3640 -/
theorem geometric_sequence_sum : 
  let a : ℕ := 10  -- initial term
  let r : ℕ := 3   -- common ratio
  let n : ℕ := 6   -- number of terms
  a * (r^n - 1) / (r - 1) = 3640 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3945_394524


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_4_solution_existence_condition_l3945_394559

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_equals_4 :
  {x : ℝ | |2*x - 4| < 8 - |x - 1|} = Set.Ioo (-1) (13/3) := by sorry

-- Theorem for the second part of the problem
theorem solution_existence_condition (a : ℝ) :
  (∃ x, f a x > 8 + |2*x - 1|) ↔ (a > 9 ∨ a < -7) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_4_solution_existence_condition_l3945_394559


namespace NUMINAMATH_CALUDE_max_distance_on_curve_l3945_394562

/-- The maximum distance between a point on the curve y² = 4 - 2x² and the point (0, -√2) -/
theorem max_distance_on_curve : ∃ (max_dist : ℝ),
  max_dist = 2 + Real.sqrt 2 ∧
  ∀ (x y : ℝ),
    y^2 = 4 - 2*x^2 →
    Real.sqrt ((x - 0)^2 + (y - (-Real.sqrt 2))^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_on_curve_l3945_394562


namespace NUMINAMATH_CALUDE_speed_of_sound_calculation_l3945_394530

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between hearing the first and second blast in seconds -/
def time_between_blasts : ℝ := 30 * 60 + 24

/-- The time between the occurrence of the first and second blast in seconds -/
def time_between_blast_occurrences : ℝ := 30 * 60

/-- The distance from the blast site when hearing the second blast in meters -/
def distance_at_second_blast : ℝ := 7920

/-- Theorem stating that the speed of sound is 330 m/s given the problem conditions -/
theorem speed_of_sound_calculation :
  speed_of_sound = distance_at_second_blast / (time_between_blasts - time_between_blast_occurrences) :=
by sorry

end NUMINAMATH_CALUDE_speed_of_sound_calculation_l3945_394530


namespace NUMINAMATH_CALUDE_greatest_consecutive_even_sum_180_l3945_394568

/-- The sum of n consecutive even integers starting from 2a is n(2a + n - 1) -/
def sumConsecutiveEvenIntegers (n : ℕ) (a : ℤ) : ℤ := n * (2 * a + n - 1)

/-- 45 is the greatest number of consecutive even integers whose sum is 180 -/
theorem greatest_consecutive_even_sum_180 :
  ∀ n : ℕ, n > 45 → ¬∃ a : ℤ, sumConsecutiveEvenIntegers n a = 180 ∧
  ∃ a : ℤ, sumConsecutiveEvenIntegers 45 a = 180 :=
by sorry

#check greatest_consecutive_even_sum_180

end NUMINAMATH_CALUDE_greatest_consecutive_even_sum_180_l3945_394568


namespace NUMINAMATH_CALUDE_expression_evaluation_l3945_394539

theorem expression_evaluation (x : ℝ) (h : x = -3) :
  (5 + 2*x*(x+2) - 4^2) / (x - 4 + x^2) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3945_394539


namespace NUMINAMATH_CALUDE_lamp_height_difference_l3945_394585

theorem lamp_height_difference (old_height new_height : Real) 
  (h1 : old_height = 1)
  (h2 : new_height = 2.33) :
  new_height - old_height = 1.33 := by
sorry

end NUMINAMATH_CALUDE_lamp_height_difference_l3945_394585


namespace NUMINAMATH_CALUDE_first_part_distance_is_18_l3945_394547

/-- Represents a cyclist's trip with given parameters -/
structure CyclistTrip where
  totalTime : ℝ
  speed1 : ℝ
  speed2 : ℝ
  distance2 : ℝ
  returnSpeed : ℝ

/-- Calculates the distance of the first part of the trip -/
def firstPartDistance (trip : CyclistTrip) : ℝ :=
  sorry

/-- Theorem stating that the first part of the trip is 18 miles long -/
theorem first_part_distance_is_18 (trip : CyclistTrip) 
  (h1 : trip.totalTime = 7.2)
  (h2 : trip.speed1 = 9)
  (h3 : trip.speed2 = 10)
  (h4 : trip.distance2 = 12)
  (h5 : trip.returnSpeed = 7.5) :
  firstPartDistance trip = 18 :=
sorry

end NUMINAMATH_CALUDE_first_part_distance_is_18_l3945_394547


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3945_394523

theorem arithmetic_evaluation : (7 + 5 + 8) / 3 - 2 / 3 + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3945_394523


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3945_394586

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_a2 : a 2 = Real.sqrt 2)
  (h_a3 : a 3 = Real.rpow 4 (1/3)) :
  (a 1 + a 15) / (a 7 + a 21) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3945_394586


namespace NUMINAMATH_CALUDE_cylinder_intersection_angle_l3945_394520

theorem cylinder_intersection_angle (r b a : ℝ) (h_r : r = 1) (h_b : b = r) 
  (h_e : (Real.sqrt 5) / 3 = Real.sqrt (1 - (b / a)^2)) :
  Real.arccos (2 / 3) = Real.arccos (b / a) := by sorry

end NUMINAMATH_CALUDE_cylinder_intersection_angle_l3945_394520


namespace NUMINAMATH_CALUDE_inscribed_right_triangle_exists_l3945_394554

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if a point is inside a circle -/
def isInside (c : Circle) (p : Point) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

/-- A right triangle -/
structure RightTriangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point
  is_right_angle : (vertex1.1 - vertex2.1) * (vertex1.1 - vertex3.1) + 
                   (vertex1.2 - vertex2.2) * (vertex1.2 - vertex3.2) = 0

/-- Check if a triangle is inscribed in a circle -/
def isInscribed (c : Circle) (t : RightTriangle) : Prop :=
  let (x1, y1) := t.vertex1
  let (x2, y2) := t.vertex2
  let (x3, y3) := t.vertex3
  let (cx, cy) := c.center
  (x1 - cx)^2 + (y1 - cy)^2 = c.radius^2 ∧
  (x2 - cx)^2 + (y2 - cy)^2 = c.radius^2 ∧
  (x3 - cx)^2 + (y3 - cy)^2 = c.radius^2

/-- Check if a line passes through a point -/
def passesThrough (p1 : Point) (p2 : Point) (p : Point) : Prop :=
  (p.1 - p1.1) * (p2.2 - p1.2) = (p.2 - p1.2) * (p2.1 - p1.1)

theorem inscribed_right_triangle_exists (c : Circle) (A B : Point) 
  (h1 : isInside c A) (h2 : isInside c B) :
  ∃ (t : RightTriangle), isInscribed c t ∧ 
    (passesThrough t.vertex1 t.vertex2 A ∨ passesThrough t.vertex1 t.vertex3 A) ∧
    (passesThrough t.vertex1 t.vertex2 B ∨ passesThrough t.vertex1 t.vertex3 B) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_right_triangle_exists_l3945_394554


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l3945_394515

/-- Given a positive constant a, prove the extreme values of f(x) = sin(2x) + √(sin²(2x) + a cos²(x)) -/
theorem extreme_values_of_f (a : ℝ) (ha : a > 0) :
  let f := fun (x : ℝ) => Real.sin (2 * x) + Real.sqrt ((Real.sin (2 * x))^2 + a * (Real.cos x)^2)
  (∀ x, f x ≥ 0) ∧ 
  (∃ x, f x = 0) ∧
  (∀ x, f x ≤ Real.sqrt (a + 4)) ∧
  (∃ x, f x = Real.sqrt (a + 4)) := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l3945_394515


namespace NUMINAMATH_CALUDE_range_of_a_l3945_394553

def A (a : ℝ) := {x : ℝ | 1 ≤ x ∧ x ≤ a}

def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = 5*x - 6}

def C (a : ℝ) := {m : ℝ | ∃ x ∈ A a, m = x^2}

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ A a → (1 ≤ x ∧ x ≤ a)) → 
  (∀ y : ℝ, y ∈ B a ↔ ∃ x ∈ A a, y = 5*x - 6) → 
  (∀ m : ℝ, m ∈ C a ↔ ∃ x ∈ A a, m = x^2) → 
  (B a ∩ C a = C a) → 
  (2 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3945_394553


namespace NUMINAMATH_CALUDE_M_inter_complement_N_eq_l3945_394542

/-- The universal set U (real numbers) -/
def U : Set ℝ := Set.univ

/-- Set M defined as {x | -2 ≤ x ≤ 2} -/
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

/-- Set N defined as the domain of y = ln(x-1), which is {x | x > 1} -/
def N : Set ℝ := {x | x > 1}

/-- Theorem stating that the intersection of M and the complement of N in U
    is equal to the set {x | -2 ≤ x ≤ 1} -/
theorem M_inter_complement_N_eq :
  M ∩ (U \ N) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_M_inter_complement_N_eq_l3945_394542


namespace NUMINAMATH_CALUDE_box_triples_count_l3945_394511

/-- The number of ordered triples (a, b, c) of positive integers satisfying the box conditions -/
def box_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 4 * (a * b + a * c + b * c))
    (Finset.product (Finset.range 100) (Finset.product (Finset.range 100) (Finset.range 100)))).card

/-- Theorem stating that there are exactly 2 ordered triples satisfying the box conditions -/
theorem box_triples_count : box_triples = 2 := by
  sorry

end NUMINAMATH_CALUDE_box_triples_count_l3945_394511


namespace NUMINAMATH_CALUDE_football_players_count_l3945_394535

/-- Calculates the number of students playing football given the total number of students,
    the number of students playing cricket, the number of students playing neither sport,
    and the number of students playing both sports. -/
def students_playing_football (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) : ℕ :=
  total - neither - cricket + both

/-- Theorem stating that the number of students playing football is 325 -/
theorem football_players_count :
  students_playing_football 450 175 50 100 = 325 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l3945_394535


namespace NUMINAMATH_CALUDE_adam_laundry_theorem_l3945_394577

/-- The number of loads Adam has already washed -/
def washed_loads : ℕ := 8

/-- The number of loads Adam still needs to wash -/
def remaining_loads : ℕ := 6

/-- The total number of loads Adam has to wash -/
def total_loads : ℕ := washed_loads + remaining_loads

theorem adam_laundry_theorem : total_loads = 14 := by sorry

end NUMINAMATH_CALUDE_adam_laundry_theorem_l3945_394577


namespace NUMINAMATH_CALUDE_infiniteLoopDecimal_eq_fraction_l3945_394558

/-- Represents the infinite loop decimal 0.0 ̇1 ̇7 -/
def infiniteLoopDecimal : ℚ := sorry

/-- The infinite loop decimal 0.0 ̇1 ̇7 is equal to 17/990 -/
theorem infiniteLoopDecimal_eq_fraction : infiniteLoopDecimal = 17 / 990 := by sorry

end NUMINAMATH_CALUDE_infiniteLoopDecimal_eq_fraction_l3945_394558


namespace NUMINAMATH_CALUDE_range_of_m_l3945_394592

-- Define the set A as the solution set of |x+m| ≤ 4
def A (m : ℝ) : Set ℝ := {x : ℝ | |x + m| ≤ 4}

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, A m ⊆ {x : ℝ | -2 ≤ x ∧ x ≤ 8}) →
  {m : ℝ | ∃ x : ℝ, x ∈ A m} = {m : ℝ | -4 ≤ m ∧ m ≤ -2} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3945_394592


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3945_394534

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧
  (∃ a, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3945_394534


namespace NUMINAMATH_CALUDE_hex_conversion_and_subtraction_l3945_394528

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | 'B' => 11
  | '1' => 1
  | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : Nat :=
  s.foldl (fun acc c => 16 * acc + hex_to_dec c) 0

/-- The hexadecimal string to be converted -/
def hex_string : String := "B1F"

/-- The value to be subtracted from the converted result -/
def subtrahend : Nat := 432

/-- Theorem stating that converting B1F from base 16 to base 10 and subtracting 432 results in 2415 -/
theorem hex_conversion_and_subtraction :
  hex_string_to_dec hex_string - subtrahend = 2415 := by
  sorry

end NUMINAMATH_CALUDE_hex_conversion_and_subtraction_l3945_394528


namespace NUMINAMATH_CALUDE_pythagorean_triple_l3945_394533

theorem pythagorean_triple (n : ℕ) (h1 : n ≥ 3) (h2 : Odd n) :
  n^2 + ((n^2 - 1) / 2)^2 = ((n^2 + 1) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_l3945_394533


namespace NUMINAMATH_CALUDE_f_sum_2016_2015_l3945_394527

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_sum_2016_2015 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_even : is_even_function (fun x ↦ f (x + 1)))
  (h_f_1 : f 1 = 1) :
  f 2016 + f 2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_2016_2015_l3945_394527


namespace NUMINAMATH_CALUDE_subtraction_addition_result_l3945_394591

/-- The result of subtracting 567.89 from 1234.56 and then adding 300.30 is equal to 966.97 -/
theorem subtraction_addition_result : 
  (1234.56 - 567.89 + 300.30 : ℚ) = 966.97 := by sorry

end NUMINAMATH_CALUDE_subtraction_addition_result_l3945_394591


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l3945_394518

theorem two_numbers_sum_and_difference (x y : ℤ) : 
  x + y = 18 ∧ x - y = 24 → x = 21 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l3945_394518


namespace NUMINAMATH_CALUDE_clairaut_general_solution_l3945_394540

/-- Clairaut's equation -/
def clairaut_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y x = x * (deriv y x) + 1 / (2 * deriv y x)

/-- General solution of Clairaut's equation -/
def is_general_solution (y : ℝ → ℝ) : Prop :=
  ∃ C : ℝ, (∀ x, y x = C * x + 1 / (2 * C)) ∨ (∀ x, (y x)^2 = 2 * x)

/-- Theorem: The general solution satisfies Clairaut's equation -/
theorem clairaut_general_solution :
  ∀ y : ℝ → ℝ, is_general_solution y → ∀ x : ℝ, clairaut_equation y x :=
sorry

end NUMINAMATH_CALUDE_clairaut_general_solution_l3945_394540


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3945_394599

/-- Given the ages of Tony and Belinda, prove that their age ratio is 5/2 -/
theorem age_ratio_proof (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  tony_age + belinda_age = 56 →
  ∃ (k : ℕ), belinda_age = k * tony_age + 8 →
  (belinda_age : ℚ) / (tony_age : ℚ) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3945_394599


namespace NUMINAMATH_CALUDE_steel_making_experiment_l3945_394566

/-- The 0.618 method calculation for steel-making experiment --/
theorem steel_making_experiment (lower upper : ℝ) (h1 : lower = 500) (h2 : upper = 1000) :
  lower + (upper - lower) * 0.618 = 809 :=
by sorry

end NUMINAMATH_CALUDE_steel_making_experiment_l3945_394566


namespace NUMINAMATH_CALUDE_defective_shipped_less_than_one_percent_l3945_394545

/-- Represents the production and shipping process of a product --/
structure ProductionProcess where
  initial_units : ℝ
  prod_defect_rate1 : ℝ
  prod_defect_rate2 : ℝ
  prod_defect_rate3 : ℝ
  ship_defect_rate1 : ℝ
  ship_defect_rate2 : ℝ
  ship_defect_rate3 : ℝ

/-- Calculates the percentage of defective units shipped for sale --/
def defective_shipped_percentage (p : ProductionProcess) : ℝ :=
  sorry

/-- Theorem stating that the percentage of defective units shipped is less than 1% --/
theorem defective_shipped_less_than_one_percent (p : ProductionProcess) 
  (h1 : p.prod_defect_rate1 = 0.06)
  (h2 : p.prod_defect_rate2 = 0.03)
  (h3 : p.prod_defect_rate3 = 0.02)
  (h4 : p.ship_defect_rate1 = 0.04)
  (h5 : p.ship_defect_rate2 = 0.03)
  (h6 : p.ship_defect_rate3 = 0.02)
  (h7 : p.initial_units > 0) :
  defective_shipped_percentage p < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_defective_shipped_less_than_one_percent_l3945_394545


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l3945_394500

theorem doctors_lawyers_ratio (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (40 * (m + n) = 35 * m + 50 * n) → (m : ℚ) / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l3945_394500


namespace NUMINAMATH_CALUDE_power_product_equals_negative_eighth_l3945_394587

theorem power_product_equals_negative_eighth (x : ℝ) (n : ℕ) :
  x = -0.125 → (x^(n+1) * 8^n = -0.125) := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_negative_eighth_l3945_394587


namespace NUMINAMATH_CALUDE_sum_of_roots_l3945_394538

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 5 → b * (b - 4) = 5 → a ≠ b → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3945_394538


namespace NUMINAMATH_CALUDE_park_playgroups_l3945_394529

/-- The number of playgroups formed when a given number of people
    are split into groups of a specific size -/
def num_playgroups (girls boys parents group_size : ℕ) : ℕ :=
  (girls + boys + parents) / group_size

theorem park_playgroups :
  num_playgroups 14 11 50 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_park_playgroups_l3945_394529


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l3945_394513

theorem initial_number_of_persons
  (average_weight_increase : ℝ)
  (weight_difference : ℝ)
  (h1 : average_weight_increase = 2.5)
  (h2 : weight_difference = 20)
  (h3 : average_weight_increase * (initial_persons : ℝ) = weight_difference) :
  initial_persons = 8 := by
sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l3945_394513


namespace NUMINAMATH_CALUDE_distribute_five_objects_l3945_394536

/-- The number of ways to distribute n distinguishable objects into 2 indistinguishable containers,
    such that neither container is empty -/
def distribute (n : ℕ) : ℕ :=
  (2^n - 2) / 2

/-- Theorem: There are 15 ways to distribute 5 distinguishable objects into 2 indistinguishable containers,
    such that neither container is empty -/
theorem distribute_five_objects : distribute 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_objects_l3945_394536


namespace NUMINAMATH_CALUDE_larger_number_problem_l3945_394598

theorem larger_number_problem (x y : ℤ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3945_394598


namespace NUMINAMATH_CALUDE_john_garage_sale_games_l3945_394561

/-- The number of games John bought from a friend -/
def games_from_friend : ℕ := 21

/-- The number of games that didn't work -/
def bad_games : ℕ := 23

/-- The number of good games John ended up with -/
def good_games : ℕ := 6

/-- The number of games John bought at the garage sale -/
def games_from_garage_sale : ℕ := (good_games + bad_games) - games_from_friend

theorem john_garage_sale_games :
  games_from_garage_sale = 8 := by sorry

end NUMINAMATH_CALUDE_john_garage_sale_games_l3945_394561


namespace NUMINAMATH_CALUDE_valid_words_count_l3945_394507

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum length of a word -/
def max_word_length : ℕ := 5

/-- The number of words of length n that do not contain the letter A -/
def words_without_a (n : ℕ) : ℕ := (alphabet_size - 1) ^ n

/-- The total number of possible words of length n -/
def total_words (n : ℕ) : ℕ := alphabet_size ^ n

/-- The number of words of length n that contain the letter A at least once -/
def words_with_a (n : ℕ) : ℕ := total_words n - words_without_a n

/-- The total number of valid words -/
def total_valid_words : ℕ :=
  words_with_a 1 + words_with_a 2 + words_with_a 3 + words_with_a 4 + words_with_a 5

theorem valid_words_count : total_valid_words = 1863701 := by
  sorry

end NUMINAMATH_CALUDE_valid_words_count_l3945_394507


namespace NUMINAMATH_CALUDE_buddy_fraction_l3945_394590

theorem buddy_fraction (t s : ℚ) 
  (h1 : t > 0) 
  (h2 : s > 0) 
  (h3 : (1/4) * t = (3/5) * s) : 
  ((1/4) * t + (3/5) * s) / (t + s) = 6/17 := by
sorry

end NUMINAMATH_CALUDE_buddy_fraction_l3945_394590


namespace NUMINAMATH_CALUDE_min_value_expression_l3945_394584

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 16) / Real.sqrt (x - 4) ≥ 4 * Real.sqrt 5 ∧
  (∃ x₀ : ℝ, x₀ > 4 ∧ (x₀ + 16) / Real.sqrt (x₀ - 4) = 4 * Real.sqrt 5 ∧ x₀ = 24) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3945_394584


namespace NUMINAMATH_CALUDE_tv_screen_area_l3945_394541

theorem tv_screen_area : 
  let trapezoid_short_base : ℝ := 3
  let trapezoid_long_base : ℝ := 5
  let trapezoid_height : ℝ := 2
  let triangle_base : ℝ := trapezoid_long_base
  let triangle_height : ℝ := 4
  let trapezoid_area := (trapezoid_short_base + trapezoid_long_base) * trapezoid_height / 2
  let triangle_area := triangle_base * triangle_height / 2
  trapezoid_area + triangle_area = 18 := by
sorry

end NUMINAMATH_CALUDE_tv_screen_area_l3945_394541


namespace NUMINAMATH_CALUDE_sum_pqrs_equals_32_1_l3945_394596

theorem sum_pqrs_equals_32_1 
  (p q r s : ℝ)
  (hp : p = 2)
  (hpq : p * q = 20)
  (hpqr : p * q * r = 202)
  (hpqrs : p * q * r * s = 2020) :
  p + q + r + s = 32.1 := by
sorry

end NUMINAMATH_CALUDE_sum_pqrs_equals_32_1_l3945_394596


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3945_394576

theorem complex_fraction_equality (u v : ℂ) 
  (h : (u^3 + v^3) / (u^3 - v^3) + (u^3 - v^3) / (u^3 + v^3) = 2) :
  (u^9 + v^9) / (u^9 - v^9) + (u^9 - v^9) / (u^9 + v^9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3945_394576


namespace NUMINAMATH_CALUDE_correct_operation_l3945_394508

theorem correct_operation (a : ℝ) : 2 * a + 3 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3945_394508


namespace NUMINAMATH_CALUDE_solve_equation_l3945_394581

theorem solve_equation : ∃ x : ℝ, x + 1 - 2 + 3 - 4 = 5 - 6 + 7 - 8 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3945_394581


namespace NUMINAMATH_CALUDE_impossible_grouping_l3945_394543

theorem impossible_grouping : ¬ ∃ (partition : List (List Nat)),
  (∀ group ∈ partition, (∀ n ∈ group, 1 ≤ n ∧ n ≤ 77)) ∧
  (∀ group ∈ partition, group.length ≥ 3) ∧
  (∀ group ∈ partition, ∃ n ∈ group, n = (group.sum - n)) ∧
  (partition.join.toFinset = Finset.range 77) :=
by sorry

end NUMINAMATH_CALUDE_impossible_grouping_l3945_394543


namespace NUMINAMATH_CALUDE_basket_count_l3945_394519

theorem basket_count (apples_per_basket : ℕ) (total_apples : ℕ) (h1 : apples_per_basket = 17) (h2 : total_apples = 629) :
  total_apples / apples_per_basket = 37 := by
  sorry

end NUMINAMATH_CALUDE_basket_count_l3945_394519


namespace NUMINAMATH_CALUDE_literacy_test_probabilities_l3945_394505

/-- Scientific literacy test model -/
structure LiteracyTest where
  /-- Probability of answering a question correctly -/
  p_correct : ℝ
  /-- Number of questions in the test -/
  total_questions : ℕ
  /-- Number of correct answers in a row needed for A rating -/
  a_threshold : ℕ
  /-- Number of incorrect answers in a row needed for C rating -/
  c_threshold : ℕ

/-- Probabilities of different outcomes in the literacy test -/
def test_probabilities (test : LiteracyTest) :
  (ℝ × ℝ × ℝ × ℝ) :=
  sorry

/-- The main theorem about the scientific literacy test -/
theorem literacy_test_probabilities :
  let test := LiteracyTest.mk (2/3) 5 4 3
  let (p_a, p_b, p_four, p_five) := test_probabilities test
  p_a = 64/243 ∧ p_b = 158/243 ∧ p_four = 2/9 ∧ p_five = 20/27 :=
sorry

end NUMINAMATH_CALUDE_literacy_test_probabilities_l3945_394505


namespace NUMINAMATH_CALUDE_calculate_speed_l3945_394575

/-- Given two people moving in opposite directions, calculate the unknown speed -/
theorem calculate_speed (known_speed time_minutes distance : ℝ) 
  (h1 : known_speed = 50)
  (h2 : time_minutes = 45)
  (h3 : distance = 60) : 
  ∃ unknown_speed : ℝ, 
    unknown_speed = 30 ∧ 
    (unknown_speed + known_speed) * (time_minutes / 60) = distance :=
by sorry

end NUMINAMATH_CALUDE_calculate_speed_l3945_394575
