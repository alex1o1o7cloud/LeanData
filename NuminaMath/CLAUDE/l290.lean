import Mathlib

namespace NUMINAMATH_CALUDE_speed_in_miles_per_minute_l290_29046

-- Define the speed in kilometers per hour
def speed_km_per_hour : ℝ := 600

-- Define the conversion factor from km to miles
def km_to_miles : ℝ := 0.6

-- Define the number of minutes in an hour
def minutes_per_hour : ℝ := 60

-- Theorem to prove
theorem speed_in_miles_per_minute :
  (speed_km_per_hour * km_to_miles) / minutes_per_hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_miles_per_minute_l290_29046


namespace NUMINAMATH_CALUDE_a_2_equals_4_l290_29088

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem a_2_equals_4 
  (a : ℕ → ℝ) 
  (h : ∀ n, sequence_sum a n = 2 * a n - 2) : 
  a 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_a_2_equals_4_l290_29088


namespace NUMINAMATH_CALUDE_trivia_team_groups_l290_29036

/-- Given a total number of students, number of students not picked, and number of students per group,
    calculate the number of groups formed. -/
def calculate_groups (total : ℕ) (not_picked : ℕ) (per_group : ℕ) : ℕ :=
  (total - not_picked) / per_group

/-- Theorem stating that with 65 total students, 17 not picked, and 6 per group, 8 groups are formed. -/
theorem trivia_team_groups : calculate_groups 65 17 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l290_29036


namespace NUMINAMATH_CALUDE_tan_15_degrees_l290_29013

theorem tan_15_degrees : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_15_degrees_l290_29013


namespace NUMINAMATH_CALUDE_digit_sum_10_2017_position_l290_29072

/-- A sequence of positive integers whose digits sum to 10, arranged in ascending order -/
def digit_sum_10_sequence : ℕ → ℕ := sorry

/-- Predicate to check if a natural number's digits sum to 10 -/
def digits_sum_to_10 (n : ℕ) : Prop := sorry

/-- The sequence digit_sum_10_sequence contains all and only the numbers whose digits sum to 10 -/
axiom digit_sum_10_sequence_property :
  ∀ n : ℕ, digits_sum_to_10 (digit_sum_10_sequence n) ∧
  (∀ m : ℕ, digits_sum_to_10 m → ∃ k : ℕ, digit_sum_10_sequence k = m)

/-- The sequence digit_sum_10_sequence is strictly increasing -/
axiom digit_sum_10_sequence_increasing :
  ∀ n m : ℕ, n < m → digit_sum_10_sequence n < digit_sum_10_sequence m

theorem digit_sum_10_2017_position :
  ∃ n : ℕ, digit_sum_10_sequence n = 2017 ∧ n = 110 := by sorry

end NUMINAMATH_CALUDE_digit_sum_10_2017_position_l290_29072


namespace NUMINAMATH_CALUDE_log_problem_trig_problem_l290_29059

theorem log_problem (k : ℝ) (p : ℝ) 
  (h : Real.log 210 + Real.log k - Real.log 56 + Real.log 40 - Real.log 120 + Real.log 25 = p) : 
  p = 3 := by sorry

theorem trig_problem (A : ℝ) (q : ℝ) 
  (h1 : Real.sin A = 3 / 5) 
  (h2 : Real.cos A / Real.tan A = q / 15) : 
  q = 16 := by sorry

end NUMINAMATH_CALUDE_log_problem_trig_problem_l290_29059


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l290_29058

/-- Given a cube with surface area 294 square centimeters, its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ, s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l290_29058


namespace NUMINAMATH_CALUDE_least_possible_qr_l290_29019

/-- Triangle PQR with side lengths -/
structure TrianglePQR where
  pq : ℝ
  pr : ℝ
  qr : ℝ
  pq_positive : 0 < pq
  pr_positive : 0 < pr
  qr_positive : 0 < qr
  triangle_inequality_1 : qr < pq + pr
  triangle_inequality_2 : pq < qr + pr
  triangle_inequality_3 : pr < pq + qr

/-- Triangle SQR with side lengths -/
structure TriangleSQR where
  sq : ℝ
  sr : ℝ
  qr : ℝ
  sq_positive : 0 < sq
  sr_positive : 0 < sr
  qr_positive : 0 < qr
  triangle_inequality_1 : qr < sq + sr
  triangle_inequality_2 : sq < qr + sr
  triangle_inequality_3 : sr < sq + qr

/-- The theorem stating the least possible integral length of QR -/
theorem least_possible_qr 
  (triangle_pqr : TrianglePQR)
  (triangle_sqr : TriangleSQR)
  (h_pq : triangle_pqr.pq = 7)
  (h_pr : triangle_pqr.pr = 10)
  (h_sq : triangle_sqr.sq = 24)
  (h_sr : triangle_sqr.sr = 15)
  (h_qr_same : triangle_pqr.qr = triangle_sqr.qr)
  (h_qr_int : ∃ n : ℕ, triangle_pqr.qr = n) :
  triangle_pqr.qr = 9 ∧ ∀ m : ℕ, (m : ℝ) = triangle_pqr.qr → m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_least_possible_qr_l290_29019


namespace NUMINAMATH_CALUDE_smallest_positive_equivalent_angle_l290_29042

theorem smallest_positive_equivalent_angle (α : ℝ) : 
  (α > 0 ∧ α < 360 ∧ ∃ k : ℤ, α = 400 - 360 * k) → α = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_equivalent_angle_l290_29042


namespace NUMINAMATH_CALUDE_inequality_system_solution_l290_29028

theorem inequality_system_solution (x : ℝ) :
  (4 * x + 6 > 1 - x) ∧ (3 * (x - 1) ≤ x + 5) → -1 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l290_29028


namespace NUMINAMATH_CALUDE_tangent_circle_center_l290_29065

/-- A circle with radius 1 and center in the first quadrant, tangent to 4x - 3y = 0 and x-axis --/
structure TangentCircle where
  a : ℝ
  b : ℝ
  first_quadrant : 0 < a ∧ 0 < b
  tangent_to_line : |4 * a - 3 * b| / Real.sqrt (4^2 + (-3)^2) = 1
  tangent_to_x_axis : b = 1

/-- The center of the tangent circle is at (2, 1) --/
theorem tangent_circle_center (c : TangentCircle) : c.a = 2 ∧ c.b = 1 := by
  sorry

#check tangent_circle_center

end NUMINAMATH_CALUDE_tangent_circle_center_l290_29065


namespace NUMINAMATH_CALUDE_ellipse_to_circle_l290_29055

theorem ellipse_to_circle 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a ≠ b) 
  (x y : ℝ) 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  x^2 + ((a/b) * y)^2 = a^2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_to_circle_l290_29055


namespace NUMINAMATH_CALUDE_words_per_page_l290_29025

theorem words_per_page (total_pages : ℕ) (word_congruence : ℕ) (max_words_per_page : ℕ)
  (h1 : total_pages = 195)
  (h2 : word_congruence = 221)
  (h3 : max_words_per_page = 120)
  (h4 : ∃ (words_per_page : ℕ), 
    (total_pages * words_per_page) % 251 = word_congruence ∧ 
    words_per_page ≤ max_words_per_page) :
  ∃ (words_per_page : ℕ), words_per_page = 41 ∧ 
    (total_pages * words_per_page) % 251 = word_congruence ∧
    words_per_page ≤ max_words_per_page :=
by sorry

end NUMINAMATH_CALUDE_words_per_page_l290_29025


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l290_29069

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x > 0) (hn : n > 1) :
  (1 + x)^n > 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l290_29069


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_special_terms_l290_29051

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_with_special_terms :
  ∃ (a : ℕ → ℤ) (d : ℤ),
    is_arithmetic_sequence a d ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    a 9 = (a 2) ^ 3 ∧
    (∃ n, a n = (a 2) ^ 2) ∧
    (∃ m, a m = (a 2) ^ 4) →
    a 1 = -24 ∧ a 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_special_terms_l290_29051


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_30_l290_29090

/-- An arithmetic sequence and its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with given partial sums, S_30 can be determined -/
theorem arithmetic_sequence_sum_30 (seq : ArithmeticSequence) 
  (h10 : seq.S 10 = 10) (h20 : seq.S 20 = 30) : seq.S 30 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_30_l290_29090


namespace NUMINAMATH_CALUDE_triangle_similarity_equality_equivalence_l290_29049

/-- Two triangles are similar if their corresponding sides are proportional -/
def SimilarTriangles (a b c a₁ b₁ c₁ : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁

/-- The theorem stating the equivalence between triangle similarity and the given equation -/
theorem triangle_similarity_equality_equivalence
  (a b c a₁ b₁ c₁ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  SimilarTriangles a b c a₁ b₁ c₁ ↔
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) =
  Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) := by
    sorry

#check triangle_similarity_equality_equivalence

end NUMINAMATH_CALUDE_triangle_similarity_equality_equivalence_l290_29049


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_P_l290_29056

/-- If the terminal side of angle α passes through point P(a, 2a) where a < 0, then cos(α) = -√5/5 -/
theorem cos_alpha_for_point_P (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : ∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = a ∧ r * (Real.sin α) = 2*a) : 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_P_l290_29056


namespace NUMINAMATH_CALUDE_committee_size_lower_bound_l290_29052

/-- A structure representing a committee with its meeting details -/
structure Committee where
  total_meetings : ℕ
  members_per_meeting : ℕ
  total_members : ℕ

/-- The property that no two people have met more than once -/
def no_repeated_meetings (c : Committee) : Prop :=
  c.total_meetings * (c.members_per_meeting.choose 2) ≤ c.total_members.choose 2

/-- The theorem to be proved -/
theorem committee_size_lower_bound (c : Committee) 
  (h1 : c.total_meetings = 40)
  (h2 : c.members_per_meeting = 10)
  (h3 : no_repeated_meetings c) :
  c.total_members > 60 := by
  sorry

end NUMINAMATH_CALUDE_committee_size_lower_bound_l290_29052


namespace NUMINAMATH_CALUDE_sin_2x_plus_one_l290_29044

theorem sin_2x_plus_one (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : 
  Real.sin (2 * x) + 1 = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_plus_one_l290_29044


namespace NUMINAMATH_CALUDE_principal_is_12000_l290_29039

/-- Represents the interest calculation problem --/
def InterestProblem (principal : ℝ) : Prop :=
  let rate1 := 0.06  -- 6% for first 2 years
  let rate2 := 0.09  -- 9% for next 3 years
  let rate3 := 0.14  -- 14% for remaining 4 years
  let interest1 := principal * rate1 * 2
  let interest2 := principal * rate2 * 3
  let interest3 := principal * rate3 * 4
  interest1 + interest2 + interest3 = 11400

/-- Theorem stating that the principal amount is 12000 --/
theorem principal_is_12000 : ∃ (principal : ℝ), InterestProblem principal ∧ principal = 12000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_12000_l290_29039


namespace NUMINAMATH_CALUDE_tangent_line_equation_l290_29048

-- Define the function f(x) = x^3 + x
def f (x : ℝ) : ℝ := x^3 + x

-- Define the point of interest
def point_of_interest : ℝ := 1

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (∃ y : ℝ, y = f point_of_interest ∧
      ∀ x : ℝ, abs (f x - (m * x + b)) ≤ abs (m * (x - point_of_interest)) * abs (x - point_of_interest)) ∧
    m * point_of_interest - f point_of_interest + b = 0 ∧
    m = 4 ∧ b = -2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l290_29048


namespace NUMINAMATH_CALUDE_proposition_relationship_l290_29038

theorem proposition_relationship (a b : ℝ) : 
  ¬(((a + b ≠ 4) → (a ≠ 1 ∧ b ≠ 3)) ∧ ((a ≠ 1 ∧ b ≠ 3) → (a + b ≠ 4))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l290_29038


namespace NUMINAMATH_CALUDE_right_triangle_area_l290_29047

theorem right_triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l290_29047


namespace NUMINAMATH_CALUDE_scientific_notation_of_448000_l290_29009

/-- Proves that 448,000 is equal to 4.48 × 10^5 in scientific notation -/
theorem scientific_notation_of_448000 : 
  ∃ (a : ℝ) (n : ℤ), 448000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.48 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_448000_l290_29009


namespace NUMINAMATH_CALUDE_dave_clothes_tickets_l290_29006

def dave_tickets : ℕ := 13
def toys_tickets : ℕ := 8
def clothes_toys_difference : ℕ := 10

theorem dave_clothes_tickets :
  dave_tickets ≥ toys_tickets + clothes_toys_difference →
  toys_tickets + clothes_toys_difference = 18 := by
  sorry

end NUMINAMATH_CALUDE_dave_clothes_tickets_l290_29006


namespace NUMINAMATH_CALUDE_triangle_area_problem_l290_29057

/-- Given a triangle ABC with area 15 and a point D on AB such that AD:DB = 3:2,
    if there exist points E on BC and F on CA forming triangle ABE and quadrilateral DBEF
    with equal areas, then the area of triangle ABE is 9. -/
theorem triangle_area_problem (A B C D E F : ℝ × ℝ) : 
  let triangle_area (P Q R : ℝ × ℝ) := abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2
  triangle_area A B C = 15 →
  D.1 = (3 * B.1 + 2 * A.1) / 5 ∧ D.2 = (3 * B.2 + 2 * A.2) / 5 →
  E.1 = B.1 ∧ E.2 ≤ B.2 ∧ E.2 ≥ C.2 →
  F.1 ≥ C.1 ∧ F.1 ≤ A.1 ∧ F.2 = C.2 →
  triangle_area A B E = triangle_area D B E + triangle_area D E F →
  triangle_area A B E = 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l290_29057


namespace NUMINAMATH_CALUDE_equation_solutions_l290_29087

theorem equation_solutions : 
  let f (x : ℝ) := (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2))
  ∀ x : ℝ, f x = 54 ↔ x = 9 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l290_29087


namespace NUMINAMATH_CALUDE_average_age_of_nine_students_l290_29020

theorem average_age_of_nine_students (total_students : Nat) (total_average : ℝ) 
  (five_students_average : ℝ) (twelfth_student_age : ℝ) :
  total_students = 16 →
  total_average = 16 →
  five_students_average = 14 →
  twelfth_student_age = 42 →
  let remaining_students := total_students - 5 - 1
  let total_age := total_students * total_average
  let five_students_total_age := 5 * five_students_average
  let remaining_total_age := total_age - five_students_total_age - twelfth_student_age
  remaining_total_age / remaining_students = 16 :=
by
  sorry

#check average_age_of_nine_students

end NUMINAMATH_CALUDE_average_age_of_nine_students_l290_29020


namespace NUMINAMATH_CALUDE_circle_line_intersection_min_value_l290_29029

/-- Given a circle with center (m,n) in the first quadrant and radius 3,
    intersected by a line to form a chord of length 4,
    the minimum value of (m+2n)/(mn) is 8/3 -/
theorem circle_line_intersection_min_value (m n : ℝ) :
  m > 0 →
  n > 0 →
  m + 2*n = 3 →
  (∀ x y : ℝ, (x - m)^2 + (y - n)^2 = 9 → x + 2*y + 2 = 0 → 
    ∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ (x' - m)^2 + (y' - n)^2 = 9 ∧ 
    x' + 2*y' + 2 = 0 ∧ (x - x')^2 + (y - y')^2 = 16) →
  (m + 2*n) / (m * n) ≥ 8/3 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_min_value_l290_29029


namespace NUMINAMATH_CALUDE_special_circle_six_radii_l290_29022

/-- A circle with integer radius and specific geometric properties -/
structure SpecialCircle where
  -- Center of the circle
  H : ℝ × ℝ
  -- Radius of the circle (which we want to prove has 6 possible integer values)
  r : ℕ
  -- Point F outside the circle
  F : ℝ × ℝ
  -- Point G on the circle and on line FH
  G : ℝ × ℝ
  -- Point I on the circle where FI is tangent
  I : ℝ × ℝ
  -- Distance FG is an integer
  FG_integer : ∃ (n : ℕ), Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) = n
  -- FI = FG + 6
  FI_eq_FG_plus_6 : Real.sqrt ((F.1 - I.1)^2 + (F.2 - I.2)^2) = 
                    Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2) + 6
  -- G is on circle
  G_on_circle : (G.1 - H.1)^2 + (G.2 - H.2)^2 = r^2
  -- I is on circle
  I_on_circle : (I.1 - H.1)^2 + (I.2 - H.2)^2 = r^2
  -- FI is tangent to circle at I
  FI_tangent : (F.1 - I.1) * (I.1 - H.1) + (F.2 - I.2) * (I.2 - H.2) = 0
  -- G is on line FH
  G_on_FH : ∃ (t : ℝ), G = (F.1 + t * (H.1 - F.1), F.2 + t * (H.2 - F.2))

/-- There are exactly 6 possible values for the radius of a SpecialCircle -/
theorem special_circle_six_radii : ∃! (S : Finset ℕ), (∀ c : SpecialCircle, c.r ∈ S) ∧ S.card = 6 :=
sorry

end NUMINAMATH_CALUDE_special_circle_six_radii_l290_29022


namespace NUMINAMATH_CALUDE_calories_in_one_bar_l290_29074

/-- The number of calories in 11 candy bars -/
def total_calories : ℕ := 341

/-- The number of candy bars -/
def num_bars : ℕ := 11

/-- The number of calories in one candy bar -/
def calories_per_bar : ℕ := total_calories / num_bars

theorem calories_in_one_bar : calories_per_bar = 31 := by
  sorry

end NUMINAMATH_CALUDE_calories_in_one_bar_l290_29074


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l290_29076

/-- If ln x - ax ≤ 2a² - 3 holds for all x > 0, then a ≥ 1 -/
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x > 0, Real.log x - a * x ≤ 2 * a^2 - 3) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l290_29076


namespace NUMINAMATH_CALUDE_rain_in_first_hour_l290_29089

theorem rain_in_first_hour (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by
sorry

end NUMINAMATH_CALUDE_rain_in_first_hour_l290_29089


namespace NUMINAMATH_CALUDE_benny_candy_bars_l290_29071

/-- The number of candy bars Benny bought -/
def candy_bars : ℕ := 5

/-- The cost of each soft drink -/
def soft_drink_cost : ℕ := 4

/-- The number of soft drinks Benny bought -/
def soft_drinks : ℕ := 2

/-- The total amount Benny spent -/
def total_spent : ℕ := 28

/-- The cost of each candy bar -/
def candy_bar_cost : ℕ := 4

theorem benny_candy_bars : 
  candy_bars * candy_bar_cost + soft_drinks * soft_drink_cost = total_spent := by
  sorry

end NUMINAMATH_CALUDE_benny_candy_bars_l290_29071


namespace NUMINAMATH_CALUDE_poetry_society_arrangement_l290_29033

-- Define the number of people in the group
def total_people : ℕ := 8

-- Define the number of people who must not be adjacent
def special_people : ℕ := 3

-- Define the number of remaining people
def remaining_people : ℕ := total_people - special_people

-- Define the number of spaces available for special people
def available_spaces : ℕ := remaining_people + 1

-- Theorem statement
theorem poetry_society_arrangement :
  (remaining_people.factorial) * (available_spaces.factorial / (available_spaces - special_people).factorial) = 14400 :=
sorry

end NUMINAMATH_CALUDE_poetry_society_arrangement_l290_29033


namespace NUMINAMATH_CALUDE_shortest_chord_through_M_l290_29034

/-- The equation of the circle O -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 10 = 0

/-- The coordinates of point M -/
def point_M : ℝ × ℝ := (3, 0)

/-- The equation of the line containing the shortest chord through point M -/
def shortest_chord_line (x y : ℝ) : Prop := x + y - 3 = 0

/-- Theorem stating that the given line equation is indeed the shortest chord through M -/
theorem shortest_chord_through_M :
  ∀ (x y : ℝ), circle_equation x y →
  shortest_chord_line x y ↔ 
  (∀ (l : ℝ → ℝ → Prop), 
    (l point_M.1 point_M.2) → 
    (∃ (p q : ℝ), circle_equation p q ∧ l p q) →
    (∀ (a b : ℝ), circle_equation a b ∧ l a b → 
      (a - point_M.1)^2 + (b - point_M.2)^2 ≥ (x - point_M.1)^2 + (y - point_M.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_through_M_l290_29034


namespace NUMINAMATH_CALUDE_polynomial_simplification_l290_29060

theorem polynomial_simplification (x : ℝ) :
  (2 * x^13 + 3 * x^12 - 4 * x^9 + 5 * x^7) + 
  (8 * x^11 - 2 * x^9 + 3 * x^7 + 6 * x^4 - 7 * x + 9) + 
  (x^13 + 4 * x^12 + x^11 + 9 * x^9) = 
  3 * x^13 + 7 * x^12 + 9 * x^11 + 3 * x^9 + 8 * x^7 + 6 * x^4 - 7 * x + 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l290_29060


namespace NUMINAMATH_CALUDE_total_students_is_44_l290_29077

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : Nat
  one : Nat
  two : Nat
  threeOrMore : Nat

/-- Calculates the total number of students -/
def totalStudents (b : BookBorrowers) : Nat :=
  b.zero + b.one + b.two + b.threeOrMore

/-- Calculates the minimum number of books borrowed -/
def minBooksBorrowed (b : BookBorrowers) : Nat :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- Theorem stating that the total number of students in the class is 44 -/
theorem total_students_is_44 (b : BookBorrowers) : 
  b.zero = 2 → 
  b.one = 12 → 
  b.two = 14 → 
  minBooksBorrowed b = 2 * totalStudents b → 
  totalStudents b = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_44_l290_29077


namespace NUMINAMATH_CALUDE_selling_price_is_80_percent_l290_29032

/-- Represents the money distribution and orange selling scenario -/
structure OrangeSelling where
  cara_ratio : ℚ
  janet_ratio : ℚ
  jerry_ratio : ℚ
  total_money : ℚ
  loss : ℚ

/-- Calculates the selling price as a percentage of the buying price -/
def sellingPricePercentage (scenario : OrangeSelling) : ℚ :=
  let x := scenario.total_money / (scenario.cara_ratio + scenario.janet_ratio + scenario.jerry_ratio)
  let cara_money := scenario.cara_ratio * x
  let janet_money := scenario.janet_ratio * x
  let buying_price := cara_money + janet_money
  let selling_price := buying_price - scenario.loss
  (selling_price / buying_price) * 100

/-- Theorem stating that the selling price is 80% of the buying price -/
theorem selling_price_is_80_percent (scenario : OrangeSelling) 
  (h1 : scenario.cara_ratio = 4)
  (h2 : scenario.janet_ratio = 5)
  (h3 : scenario.jerry_ratio = 6)
  (h4 : scenario.total_money = 75)
  (h5 : scenario.loss = 9) :
  sellingPricePercentage scenario = 80 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_80_percent_l290_29032


namespace NUMINAMATH_CALUDE_savings_ratio_proof_l290_29023

def husband_contribution : ℕ := 335
def wife_contribution : ℕ := 225
def savings_period_months : ℕ := 6
def weeks_per_month : ℕ := 4
def num_children : ℕ := 4
def amount_per_child : ℕ := 1680

def total_weekly_contribution : ℕ := husband_contribution + wife_contribution
def total_weeks : ℕ := savings_period_months * weeks_per_month
def total_savings : ℕ := total_weekly_contribution * total_weeks
def total_divided : ℕ := amount_per_child * num_children

theorem savings_ratio_proof : 
  (total_divided : ℚ) / total_savings = 1/2 := by sorry

end NUMINAMATH_CALUDE_savings_ratio_proof_l290_29023


namespace NUMINAMATH_CALUDE_min_value_inequality_l290_29097

theorem min_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c) * ((a + b + d)⁻¹ + (a + c + d)⁻¹ + (b + c + d)⁻¹) ≥ (9 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l290_29097


namespace NUMINAMATH_CALUDE_shannon_stones_l290_29043

/-- The number of heart-shaped stones Shannon wants in each bracelet -/
def stones_per_bracelet : ℝ := 8.0

/-- The number of bracelets Shannon can make -/
def number_of_bracelets : ℕ := 6

/-- The total number of heart-shaped stones Shannon brought -/
def total_stones : ℝ := stones_per_bracelet * (number_of_bracelets : ℝ)

theorem shannon_stones :
  total_stones = 48.0 := by sorry

end NUMINAMATH_CALUDE_shannon_stones_l290_29043


namespace NUMINAMATH_CALUDE_correct_calculation_l290_29099

theorem correct_calculation (a : ℝ) : a^5 + a^5 = 2*a^5 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l290_29099


namespace NUMINAMATH_CALUDE_equation_solution_l290_29084

theorem equation_solution :
  ∀ x y : ℝ, y = 3 * x →
  (5 * y^2 + 3 * y + 2 = 3 * (8 * x^2 + y + 1)) ↔ 
  (x = 1 / Real.sqrt 21 ∨ x = -(1 / Real.sqrt 21)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l290_29084


namespace NUMINAMATH_CALUDE_joan_family_distance_l290_29080

/-- Calculates the distance traveled given the total time, driving speed, and break times. -/
def distance_traveled (total_time : ℝ) (speed : ℝ) (lunch_break : ℝ) (bathroom_break : ℝ) : ℝ :=
  (total_time - (lunch_break + 2 * bathroom_break)) * speed

/-- Theorem: Given Joan's travel conditions, her family lives 480 miles away. -/
theorem joan_family_distance :
  let total_time : ℝ := 9  -- 9 hours total trip time
  let speed : ℝ := 60      -- 60 mph driving speed
  let lunch_break : ℝ := 0.5  -- 30 minutes = 0.5 hours
  let bathroom_break : ℝ := 0.25  -- 15 minutes = 0.25 hours
  distance_traveled total_time speed lunch_break bathroom_break = 480 := by
  sorry

#eval distance_traveled 9 60 0.5 0.25

end NUMINAMATH_CALUDE_joan_family_distance_l290_29080


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l290_29082

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_difference :
  let C := arithmetic_sequence 20 15
  let D := arithmetic_sequence 20 (-15)
  |C 31 - D 31| = 900 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l290_29082


namespace NUMINAMATH_CALUDE_rowing_round_trip_time_l290_29064

/-- The time taken for a round trip rowing journey given the rowing speed, current velocity, and distance. -/
theorem rowing_round_trip_time
  (rowing_speed : ℝ)
  (current_velocity : ℝ)
  (distance : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : current_velocity = 2)
  (h3 : distance = 72)
  : (distance / (rowing_speed - current_velocity) + distance / (rowing_speed + current_velocity)) = 15 :=
by sorry

end NUMINAMATH_CALUDE_rowing_round_trip_time_l290_29064


namespace NUMINAMATH_CALUDE_arrangements_count_l290_29014

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of positions where A and B can be placed with one person in between -/
def positions : ℕ := 5

/-- The number of ways to arrange the remaining students -/
def remaining_arrangements : ℕ := Nat.factorial (n - 3)

/-- The number of ways A and B can switch places -/
def ab_switch : ℕ := 2

/-- The total number of arrangements for 7 students standing in a row,
    where there must be one person standing between students A and B -/
def total_arrangements : ℕ := positions * remaining_arrangements * ab_switch

theorem arrangements_count : total_arrangements = 1200 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l290_29014


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l290_29066

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℕ) (time : ℕ) : ℚ :=
  (amount - principal : ℚ) * 100 / (principal * time)

/-- Theorem stating that the simple interest rate is 12% given the problem conditions -/
theorem simple_interest_rate_problem :
  simple_interest_rate 750 1200 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l290_29066


namespace NUMINAMATH_CALUDE_cylinder_surface_area_ratio_l290_29045

/-- Given a cylinder with a square side-unfolding, the ratio of its total surface area
    to its side surface area is (1 + 2π) / (2π). -/
theorem cylinder_surface_area_ratio (r : ℝ) (h : r > 0) :
  let height := 2 * π * r
  let side_area := 2 * π * r * height
  let total_area := side_area + 2 * π * r^2
  total_area / side_area = (1 + 2 * π) / (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_ratio_l290_29045


namespace NUMINAMATH_CALUDE_train_crossing_time_l290_29092

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 225 →
  train_speed_kmh = 90 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l290_29092


namespace NUMINAMATH_CALUDE_distance_to_origin_l290_29010

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) (h2 : x = 2 + Real.sqrt 105)
  (h3 : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l290_29010


namespace NUMINAMATH_CALUDE_simplify_expression_l290_29003

theorem simplify_expression (b : ℝ) (h : b ≠ 1) :
  1 - 1 / (2 + b / (1 - b)) = 1 / (2 - b) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l290_29003


namespace NUMINAMATH_CALUDE_B_equals_roster_l290_29012

def A : Set Int := {-2, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_roster : B = {4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_B_equals_roster_l290_29012


namespace NUMINAMATH_CALUDE_gcd_max_digits_l290_29021

theorem gcd_max_digits (a b : ℕ) : 
  1000000 ≤ a ∧ a < 10000000 →
  10000000 ≤ b ∧ b < 100000000 →
  Nat.lcm a b = 1000000000000 →
  Nat.gcd a b < 1000 := by
sorry

end NUMINAMATH_CALUDE_gcd_max_digits_l290_29021


namespace NUMINAMATH_CALUDE_tangent_product_equality_l290_29007

theorem tangent_product_equality : 
  (1 + Real.tan (17 * π / 180)) * 
  (1 + Real.tan (28 * π / 180)) * 
  (1 + Real.tan (27 * π / 180)) * 
  (1 + Real.tan (18 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equality_l290_29007


namespace NUMINAMATH_CALUDE_smaller_cubes_count_l290_29008

theorem smaller_cubes_count (larger_cube_volume : ℝ) (smaller_cube_volume : ℝ) (surface_area_difference : ℝ) :
  larger_cube_volume = 216 →
  smaller_cube_volume = 1 →
  surface_area_difference = 1080 →
  (smaller_cube_volume^(1/3) * 6 * (larger_cube_volume / smaller_cube_volume) - larger_cube_volume^(2/3) * 6 = surface_area_difference) →
  (larger_cube_volume / smaller_cube_volume) = 216 :=
by
  sorry

#check smaller_cubes_count

end NUMINAMATH_CALUDE_smaller_cubes_count_l290_29008


namespace NUMINAMATH_CALUDE_constant_pace_jogging_l290_29040

/-- Represents the time taken to jog a certain distance at a constant pace -/
structure JoggingTime where
  distance : ℝ
  time : ℝ

/-- Given a constant jogging pace, if it takes 24 minutes to jog 3 miles,
    then it will take 12 minutes to jog 1.5 miles -/
theorem constant_pace_jogging 
  (pace : ℝ) 
  (gym : JoggingTime) 
  (park : JoggingTime) 
  (h1 : gym.distance = 3) 
  (h2 : gym.time = 24) 
  (h3 : park.distance = 1.5) 
  (h4 : pace > 0) 
  (h5 : ∀ j : JoggingTime, j.time = j.distance / pace) : 
  park.time = 12 :=
sorry

end NUMINAMATH_CALUDE_constant_pace_jogging_l290_29040


namespace NUMINAMATH_CALUDE_circle_radius_in_triangle_l290_29053

/-- Triangle DEF with specified side lengths -/
structure Triangle where
  de : ℝ
  df : ℝ
  ef : ℝ
  h_de : de = 64
  h_df : df = 64
  h_ef : ef = 72

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Tangency relation between circle and line segment -/
def IsTangent (c : Circle) (a b : ℝ × ℝ) : Prop := sorry

/-- External tangency relation between two circles -/
def IsExternallyTangent (c1 c2 : Circle) : Prop := sorry

/-- A circle is inside a triangle -/
def IsInside (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem circle_radius_in_triangle (t : Triangle) (r s : Circle) :
  t.de = 64 →
  t.df = 64 →
  t.ef = 72 →
  r.radius = 20 →
  IsTangent r (0, 0) (t.df, 0) →
  IsTangent r (t.ef, 0) (0, 0) →
  IsExternallyTangent s r →
  IsTangent s (0, 0) (t.de, 0) →
  IsTangent s (t.ef, 0) (0, 0) →
  IsInside s t →
  s.radius = 52 - 4 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_triangle_l290_29053


namespace NUMINAMATH_CALUDE_final_position_on_number_line_final_position_is_28_l290_29081

/-- Given a number line where the distance from 0 to 40 is divided into 10 equal steps,
    if a person moves forward 8 steps and then back 1 step, their final position will be 28. -/
theorem final_position_on_number_line : ℝ → Prop :=
  fun final_position =>
    let total_distance : ℝ := 40
    let total_steps : ℕ := 10
    let step_size : ℝ := total_distance / total_steps
    let forward_steps : ℕ := 8
    let backward_steps : ℕ := 1
    final_position = (forward_steps - backward_steps : ℕ) * step_size

theorem final_position_is_28 : final_position_on_number_line 28 := by
  sorry

#check final_position_is_28

end NUMINAMATH_CALUDE_final_position_on_number_line_final_position_is_28_l290_29081


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l290_29026

theorem sum_of_three_numbers : 1.48 + 2.32 + 8.45 = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l290_29026


namespace NUMINAMATH_CALUDE_circle_equation_theta_range_l290_29018

theorem circle_equation_theta_range :
  ∀ (x y θ : ℝ),
  (x^2 + y^2 + x + Real.sqrt 3 * y + Real.tan θ = 0) →
  (-π/2 < θ ∧ θ < π/2) →
  (∃ (c : ℝ × ℝ) (r : ℝ), ∀ (p : ℝ × ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ↔ 
    p.1^2 + p.2^2 + p.1 + Real.sqrt 3 * p.2 + Real.tan θ = 0) →
  -π/2 < θ ∧ θ < π/4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_theta_range_l290_29018


namespace NUMINAMATH_CALUDE_second_number_calculation_l290_29095

theorem second_number_calculation (x y z : ℚ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 5 / 8) :
  y = 2400 / 67 := by
sorry

end NUMINAMATH_CALUDE_second_number_calculation_l290_29095


namespace NUMINAMATH_CALUDE_complex_conversion_l290_29075

theorem complex_conversion :
  (2 * Real.sqrt 3) * Complex.exp (Complex.I * (17 * Real.pi / 6)) = -3 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_conversion_l290_29075


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l290_29001

theorem inequality_implies_lower_bound (x y : ℝ) 
  (h : x^4 * y^2 + y^4 + 2 * x^3 * y + 6 * x^2 * y + x^2 + 8 ≤ 0) : 
  x ≥ -1/6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l290_29001


namespace NUMINAMATH_CALUDE_soy_sauce_bottles_l290_29054

/-- Represents the amount of soy sauce in ounces -/
def OuncesPerBottle : ℕ := 16

/-- Represents the number of ounces in a cup -/
def OuncesPerCup : ℕ := 8

/-- Represents the amount of soy sauce needed for recipe 1 in cups -/
def Recipe1Cups : ℕ := 2

/-- Represents the amount of soy sauce needed for recipe 2 in cups -/
def Recipe2Cups : ℕ := 1

/-- Represents the amount of soy sauce needed for recipe 3 in cups -/
def Recipe3Cups : ℕ := 3

/-- Calculates the total number of cups needed for all recipes -/
def TotalCups : ℕ := Recipe1Cups + Recipe2Cups + Recipe3Cups

/-- Calculates the total number of ounces needed for all recipes -/
def TotalOunces : ℕ := TotalCups * OuncesPerCup

/-- Calculates the number of bottles needed -/
def BottlesNeeded : ℕ := (TotalOunces + OuncesPerBottle - 1) / OuncesPerBottle

theorem soy_sauce_bottles : BottlesNeeded = 3 := by
  sorry

end NUMINAMATH_CALUDE_soy_sauce_bottles_l290_29054


namespace NUMINAMATH_CALUDE_x_minus_y_equals_half_l290_29011

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x y : ℝ) : Set ℝ := {1/x, |x|, y/x}

-- State the theorem
theorem x_minus_y_equals_half (x y : ℝ) : A x = B x y → x - y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_half_l290_29011


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l290_29079

/-- Given a cube with edge length 3 cm, if we cut out 6 smaller cubes each with edge length 1 cm,
    the remaining volume is 21 cm³. -/
theorem remaining_cube_volume :
  let large_cube_edge : ℝ := 3
  let small_cube_edge : ℝ := 1
  let num_faces : ℕ := 6
  let original_volume := large_cube_edge ^ 3
  let cut_out_volume := num_faces * small_cube_edge ^ 3
  original_volume - cut_out_volume = 21 := by sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l290_29079


namespace NUMINAMATH_CALUDE_probability_is_half_l290_29027

/-- Represents a circular field with 6 equally spaced roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : Nat)
  (road_angle : ℝ)

/-- Represents a geologist's position after traveling -/
structure GeologistPosition :=
  (road : Nat)
  (distance : ℝ)

/-- Calculates the distance between two geologists -/
def distance_between (field : CircularField) (pos1 pos2 : GeologistPosition) : ℝ :=
  sorry

/-- Determines if two roads are neighboring -/
def are_neighboring (field : CircularField) (road1 road2 : Nat) : Bool :=
  sorry

/-- Calculates the probability of two geologists being more than 8 km apart -/
def probability_more_than_8km (field : CircularField) (speed : ℝ) (time : ℝ) : ℝ :=
  sorry

/-- Main theorem: Probability of geologists being more than 8 km apart is 0.5 -/
theorem probability_is_half (field : CircularField) :
  probability_more_than_8km field 5 1 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l290_29027


namespace NUMINAMATH_CALUDE_equidistant_points_on_line_in_quadrants_I_II_l290_29024

/-- A point (x, y) is in the first quadrant if both x and y are positive -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in the second quadrant if x is negative and y is positive -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is equidistant from the coordinate axes if |x| = |y| -/
def equidistant_from_axes (x y : ℝ) : Prop := abs x = abs y

/-- A point (x, y) is on the line 4x + 6y = 24 -/
def on_line (x y : ℝ) : Prop := 4*x + 6*y = 24

theorem equidistant_points_on_line_in_quadrants_I_II :
  ∃ x y : ℝ, on_line x y ∧ equidistant_from_axes x y ∧ (in_first_quadrant x y ∨ in_second_quadrant x y) ∧
  ∀ x' y' : ℝ, on_line x' y' ∧ equidistant_from_axes x' y' → (in_first_quadrant x' y' ∨ in_second_quadrant x' y') :=
sorry

end NUMINAMATH_CALUDE_equidistant_points_on_line_in_quadrants_I_II_l290_29024


namespace NUMINAMATH_CALUDE_complex_equation_proof_l290_29070

theorem complex_equation_proof (z : ℂ) (h : z = 1 + I) : z^2 - 2*z + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l290_29070


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l290_29093

/-- Calculates the difference in profit shares between two partners given their investments and the profit share of a third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℚ) : ℚ :=
  let total_invest := invest_a + invest_b + invest_c
  let total_profit := (total_invest * b_profit) / invest_b
  let a_share := (invest_a * total_profit) / total_invest
  let c_share := (invest_c * total_profit) / total_invest
  c_share - a_share

/-- Given the investments of A, B, and C as 8000, 10000, and 12000 respectively,
    and B's profit share as 1800, the difference between A's and C's profit shares is 720. -/
theorem profit_share_difference_example :
  profit_share_difference 8000 10000 12000 1800 = 720 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_example_l290_29093


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l290_29031

theorem unique_solution_cube_equation :
  ∀ x y z : ℤ, x^3 - 3*y^3 - 9*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l290_29031


namespace NUMINAMATH_CALUDE_certain_number_proof_l290_29085

theorem certain_number_proof : ∃ x : ℤ, (9823 + x = 13200) ∧ (x = 3377) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l290_29085


namespace NUMINAMATH_CALUDE_gift_bags_total_l290_29086

theorem gift_bags_total (daily_rate : ℕ) (days_needed : ℕ) (h1 : daily_rate = 42) (h2 : days_needed = 13) :
  daily_rate * days_needed = 546 := by
  sorry

end NUMINAMATH_CALUDE_gift_bags_total_l290_29086


namespace NUMINAMATH_CALUDE_specific_solid_surface_area_l290_29068

/-- A solid with specific dimensions -/
structure Solid where
  front_length : ℝ
  front_width : ℝ
  left_length : ℝ
  left_width : ℝ
  top_radius : ℝ

/-- The surface area of the solid -/
def surface_area (s : Solid) : ℝ := sorry

/-- Theorem stating the surface area of the specific solid -/
theorem specific_solid_surface_area :
  ∀ s : Solid,
    s.front_length = 4 ∧
    s.front_width = 2 ∧
    s.left_length = 4 ∧
    s.left_width = 2 ∧
    s.top_radius = 2 →
    surface_area s = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_specific_solid_surface_area_l290_29068


namespace NUMINAMATH_CALUDE_complex_point_location_l290_29002

theorem complex_point_location (x y : ℝ) (h : x / (1 + Complex.I) = 1 - y * Complex.I) :
  x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_location_l290_29002


namespace NUMINAMATH_CALUDE_exists_triangular_numbers_ratio_two_to_one_l290_29062

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: There exist two triangular numbers with a ratio of 2:1 -/
theorem exists_triangular_numbers_ratio_two_to_one :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ triangular_number m = 2 * triangular_number n :=
by
  sorry

end NUMINAMATH_CALUDE_exists_triangular_numbers_ratio_two_to_one_l290_29062


namespace NUMINAMATH_CALUDE_expression_equals_24_l290_29050

theorem expression_equals_24 : 
  2012 * ((3.75 * 1.3 + 3 / 2.6666666666666665) / ((1+3+5+7+9) * 20 + 3)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_24_l290_29050


namespace NUMINAMATH_CALUDE_exactly_one_two_digit_number_satisfies_condition_l290_29067

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : Nat) : Nat :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is a two-digit positive integer -/
def isTwoDigit (n : Nat) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Checks if a number is thrice a perfect square -/
def isThricePerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = 3 * m * m

theorem exactly_one_two_digit_number_satisfies_condition : 
  ∃! n : Nat, isTwoDigit n ∧ 
    isThricePerfectSquare (n + 2 * (reverseDigits n)) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_two_digit_number_satisfies_condition_l290_29067


namespace NUMINAMATH_CALUDE_age_difference_l290_29030

theorem age_difference (son_age father_age : ℕ) 
  (h1 : son_age = 9)
  (h2 : father_age = 36) :
  father_age - son_age = 27 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l290_29030


namespace NUMINAMATH_CALUDE_perspective_square_area_l290_29016

/-- A square whose perspective drawing is a parallelogram with one side of length 4 -/
structure PerspectiveSquare where
  /-- The side length of the original square -/
  side : ℝ
  /-- The side length of the parallelogram in the perspective drawing -/
  perspective_side : ℝ
  /-- The perspective drawing is a parallelogram -/
  is_parallelogram : Bool
  /-- One side of the parallelogram has length 4 -/
  perspective_side_eq_four : perspective_side = 4

/-- The possible areas of the original square -/
def possible_areas (s : PerspectiveSquare) : Set ℝ :=
  {16, 64}

/-- Theorem: The area of the original square is either 16 or 64 -/
theorem perspective_square_area (s : PerspectiveSquare) :
  (s.side ^ 2) ∈ possible_areas s :=
sorry

end NUMINAMATH_CALUDE_perspective_square_area_l290_29016


namespace NUMINAMATH_CALUDE_center_cell_value_l290_29091

/-- Represents a 5x5 square with arithmetic progressions in rows and columns -/
def ArithmeticSquare : Type := Fin 5 → Fin 5 → ℤ

/-- The property that a row forms an arithmetic progression -/
def row_is_arithmetic_progression (s : ArithmeticSquare) (i : Fin 5) : Prop :=
  ∀ j k : Fin 5, s i k - s i j = (k - j) * (s i 1 - s i 0)

/-- The property that a column forms an arithmetic progression -/
def col_is_arithmetic_progression (s : ArithmeticSquare) (j : Fin 5) : Prop :=
  ∀ i k : Fin 5, s k j - s i j = (k - i) * (s 1 j - s 0 j)

/-- The main theorem -/
theorem center_cell_value (s : ArithmeticSquare) 
  (corner_values : s 0 0 = 1 ∧ s 0 4 = 25 ∧ s 4 0 = 81 ∧ s 4 4 = 17)
  (rows_arithmetic : ∀ i : Fin 5, row_is_arithmetic_progression s i)
  (cols_arithmetic : ∀ j : Fin 5, col_is_arithmetic_progression s j) :
  s 2 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_center_cell_value_l290_29091


namespace NUMINAMATH_CALUDE_range_of_a_l290_29098

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc a (a + 2), |x + a| ≥ 2 * |x|) → a ≤ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l290_29098


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l290_29083

/-- Represents the cost function for a caterer -/
structure CatererCost where
  basicFee : ℕ
  perPersonCost : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (c : CatererCost) (people : ℕ) : ℕ :=
  c.basicFee + c.perPersonCost * people

/-- First caterer's cost structure -/
def caterer1 : CatererCost :=
  { basicFee := 50, perPersonCost := 18 }

/-- Second caterer's cost structure -/
def caterer2 : CatererCost :=
  { basicFee := 150, perPersonCost := 15 }

/-- Theorem stating that 34 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost caterer1 n ≤ totalCost caterer2 n) ∧
  (totalCost caterer1 34 > totalCost caterer2 34) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l290_29083


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_integers_l290_29078

theorem largest_of_four_consecutive_integers (a b c d : ℕ) : 
  a > 0 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ a * b * c * d = 840 → d = 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_integers_l290_29078


namespace NUMINAMATH_CALUDE_fraction_evaluation_l290_29005

theorem fraction_evaluation : (20 - 4) / (6 - 3) = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l290_29005


namespace NUMINAMATH_CALUDE_inequality_solution_system_of_equations_solution_l290_29004

-- Part 1: Inequality
theorem inequality_solution (x : ℝ) :
  (5 * x - 12 ≤ 2 * (4 * x - 3)) ↔ (x ≥ -2) := by sorry

-- Part 2: System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (x - y = 5 ∧ 2 * x + y = 4) → (x = 3 ∧ y = -2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_system_of_equations_solution_l290_29004


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l290_29063

/-- Given a constant ratio between (2x - 5) and (y + 20), and the condition that y = 6 when x = 7,
    prove that x = 499/52 when y = 21 -/
theorem constant_ratio_problem (k : ℚ) :
  (∀ x y : ℚ, (2 * x - 5) / (y + 20) = k) →
  ((2 * 7 - 5) / (6 + 20) = k) →
  ∃ x : ℚ, (2 * x - 5) / (21 + 20) = k ∧ x = 499 / 52 :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l290_29063


namespace NUMINAMATH_CALUDE_solve_for_s_l290_29015

theorem solve_for_s (m : ℝ) (s : ℝ) 
  (h1 : 5 = m * (3 ^ s)) 
  (h2 : 45 = m * (9 ^ s)) : 
  s = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_s_l290_29015


namespace NUMINAMATH_CALUDE_even_function_sum_l290_29000

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

theorem even_function_sum (a b : ℝ) : 
  (∀ x ∈ Set.Icc (a^2 - 2) a, f a b x = f a b (-x)) →
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l290_29000


namespace NUMINAMATH_CALUDE_square_minus_circle_area_l290_29094

theorem square_minus_circle_area (r : ℝ) (s : ℝ) : 
  r = 2 → s = 2 * Real.sqrt 2 → 
  s^2 - π * r^2 = 8 - 4 * π := by
sorry

end NUMINAMATH_CALUDE_square_minus_circle_area_l290_29094


namespace NUMINAMATH_CALUDE_ant_movement_l290_29041

theorem ant_movement (initial_position : Int) (move_right1 move_left move_right2 : Int) :
  initial_position = -3 →
  move_right1 = 5 →
  move_left = 9 →
  move_right2 = 1 →
  initial_position + move_right1 - move_left + move_right2 = -6 :=
by sorry

end NUMINAMATH_CALUDE_ant_movement_l290_29041


namespace NUMINAMATH_CALUDE_cars_meet_time_l290_29073

/-- Two cars meet on a highway -/
theorem cars_meet_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 333)
  (h2 : speed1 = 54) (h3 : speed2 = 57) :
  (highway_length / (speed1 + speed2) : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cars_meet_time_l290_29073


namespace NUMINAMATH_CALUDE_gcd_of_polynomials_l290_29096

theorem gcd_of_polynomials (a : ℤ) (h : ∃ k : ℤ, a = 720 * k) :
  Int.gcd (a^2 + 8*a + 18) (a + 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomials_l290_29096


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l290_29061

theorem quadratic_equation_solution (a b : ℝ) : 
  (a * 1^2 + b * 1 + 2 = 0) → (2023 - a - b = 2025) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l290_29061


namespace NUMINAMATH_CALUDE_smaller_sphere_radius_l290_29037

/-- The radius of a smaller sphere when a sphere of radius R is cast into two smaller spheres -/
theorem smaller_sphere_radius (R : ℝ) (R_pos : R > 0) : ℝ :=
  let smaller_radius := R / 3
  let larger_radius := 2 * smaller_radius
  have volume_conservation : (4 / 3) * Real.pi * R^3 = (4 / 3) * Real.pi * smaller_radius^3 + (4 / 3) * Real.pi * larger_radius^3 := by sorry
  have radius_ratio : larger_radius = 2 * smaller_radius := by sorry
  smaller_radius

#check smaller_sphere_radius

end NUMINAMATH_CALUDE_smaller_sphere_radius_l290_29037


namespace NUMINAMATH_CALUDE_pencil_packs_l290_29017

theorem pencil_packs (pencils_per_pack : ℕ) (pencils_per_row : ℕ) (total_rows : ℕ) : 
  pencils_per_pack = 4 →
  pencils_per_row = 2 →
  total_rows = 70 →
  (total_rows * pencils_per_row) / pencils_per_pack = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_packs_l290_29017


namespace NUMINAMATH_CALUDE_fraction_deviation_from_sqrt_l290_29035

theorem fraction_deviation_from_sqrt (x : ℝ) (h : 1 ≤ x ∧ x ≤ 9) : 
  |Real.sqrt x - (6 * x + 6) / (x + 11)| < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_fraction_deviation_from_sqrt_l290_29035
