import Mathlib

namespace NUMINAMATH_CALUDE_similar_triangles_segment_length_l812_81295

/-- Two triangles are similar if they have the same shape but not necessarily the same size. -/
def SimilarTriangles (P Q R X Y Z : ℝ × ℝ) : Prop := sorry

theorem similar_triangles_segment_length 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles P Q R X Y Z)
  (h_PQ : dist P Q = 8)
  (h_QR : dist Q R = 16)
  (h_YZ : dist Y Z = 24) :
  dist X Y = 12 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_segment_length_l812_81295


namespace NUMINAMATH_CALUDE_karen_grooms_one_chihuahua_l812_81275

/-- The time it takes to groom a Rottweiler -/
def rottweiler_time : ℕ := 20

/-- The time it takes to groom a border collie -/
def border_collie_time : ℕ := 10

/-- The time it takes to groom a chihuahua -/
def chihuahua_time : ℕ := 45

/-- The total time Karen spends grooming -/
def total_time : ℕ := 255

/-- The number of Rottweilers Karen grooms -/
def num_rottweilers : ℕ := 6

/-- The number of border collies Karen grooms -/
def num_border_collies : ℕ := 9

/-- The number of chihuahuas Karen grooms -/
def num_chihuahuas : ℕ := 1

theorem karen_grooms_one_chihuahua :
  num_chihuahuas * chihuahua_time =
  total_time - (num_rottweilers * rottweiler_time + num_border_collies * border_collie_time) :=
by sorry

end NUMINAMATH_CALUDE_karen_grooms_one_chihuahua_l812_81275


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l812_81282

/-- The area of a rhombus formed by intersecting equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_height : ℝ := square_side * (Real.sqrt 3) / 2
  let rhombus_diagonal1 : ℝ := 2 * triangle_height - square_side
  let rhombus_diagonal2 : ℝ := square_side
  let rhombus_area : ℝ := (rhombus_diagonal1 * rhombus_diagonal2) / 2
  rhombus_area = 8 * Real.sqrt 3 - 8 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_area_in_square_l812_81282


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l812_81226

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (e : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / a^2 - y₁^2 / b^2 = 1 ∧ 
    x₂^2 / a^2 - y₂^2 / b^2 = 1 ∧
    y₁ - y₂ = x₁ - x₂ ∧
    x₁ > c ∧ x₂ > c) →
  (2 * b^2) / a = (2 * Real.sqrt 2 / 3) * b * e^2 →
  e = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l812_81226


namespace NUMINAMATH_CALUDE_computers_needed_for_expanded_class_l812_81274

/-- Given an initial number of students, a student-to-computer ratio, and additional students,
    calculate the total number of computers needed to maintain the ratio. -/
def total_computers_needed (initial_students : ℕ) (ratio : ℕ) (additional_students : ℕ) : ℕ :=
  (initial_students / ratio) + (additional_students / ratio)

/-- Theorem: Given 82 initial students, a ratio of 2 students per computer, and 16 additional students,
    the total number of computers needed to maintain the same ratio is 49. -/
theorem computers_needed_for_expanded_class : total_computers_needed 82 2 16 = 49 := by
  sorry

end NUMINAMATH_CALUDE_computers_needed_for_expanded_class_l812_81274


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l812_81289

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The problem statement -/
theorem parallel_lines_a_value (a : ℝ) :
  parallel (1 + a) 1 1 2 a 2 → a = 1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l812_81289


namespace NUMINAMATH_CALUDE_part_one_part_two_l812_81270

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - (a + 1/a)*x + 1 < 0

def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Theorem for part (1)
theorem part_one (a x : ℝ) (h1 : a = 2) (h2 : a > 1) (h3 : p x a ∧ q x) :
  1 ≤ x ∧ x < 2 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) (h : a > 1)
  (h_necessary : ∀ x, q x → p x a)
  (h_not_sufficient : ∃ x, p x a ∧ ¬q x) :
  3 < a := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l812_81270


namespace NUMINAMATH_CALUDE_min_lines_for_200_intersections_l812_81213

/-- The number of intersection points for m lines -/
def intersectionPoints (m : ℕ) : ℕ := m * (m - 1) / 2

/-- The minimum number of lines that intersect in exactly 200 points -/
def minLines : ℕ := 21

theorem min_lines_for_200_intersections :
  (intersectionPoints minLines = 200) ∧
  (∀ k : ℕ, k < minLines → intersectionPoints k < 200) := by
  sorry

end NUMINAMATH_CALUDE_min_lines_for_200_intersections_l812_81213


namespace NUMINAMATH_CALUDE_jason_total_games_l812_81232

/-- The number of games Jason attended this month -/
def games_this_month : ℕ := 11

/-- The number of games Jason attended last month -/
def games_last_month : ℕ := 17

/-- The number of games Jason plans to attend next month -/
def games_next_month : ℕ := 16

/-- The total number of games Jason will attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem jason_total_games : total_games = 44 := by sorry

end NUMINAMATH_CALUDE_jason_total_games_l812_81232


namespace NUMINAMATH_CALUDE_min_value_fraction_l812_81225

theorem min_value_fraction (x : ℝ) (h : x > -2) :
  (x^2 + 6*x + 9) / (x + 2) ≥ 4 ∧ ∃ y > -2, (y^2 + 6*y + 9) / (y + 2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l812_81225


namespace NUMINAMATH_CALUDE_rectangle_area_l812_81259

theorem rectangle_area (perimeter width length : ℝ) : 
  perimeter = 72 ∧ 
  2 * (length + width) = perimeter ∧ 
  length = 3 * width → 
  length * width = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l812_81259


namespace NUMINAMATH_CALUDE_attendee_difference_is_25_l812_81227

/-- The number of attendees from Company A -/
def company_A : ℕ := 30

/-- The number of attendees from Company B -/
def company_B : ℕ := 2 * company_A

/-- The number of attendees from Company C -/
def company_C : ℕ := company_A + 10

/-- The number of attendees from Company D -/
def company_D : ℕ := 185 - (company_A + company_B + company_C + 20)

/-- The difference in attendees between Company C and Company D -/
def attendee_difference : ℕ := company_C - company_D

theorem attendee_difference_is_25 : attendee_difference = 25 := by
  sorry

end NUMINAMATH_CALUDE_attendee_difference_is_25_l812_81227


namespace NUMINAMATH_CALUDE_xiaoying_journey_equations_l812_81285

/-- Represents Xiaoying's journey to school --/
structure JourneyToSchool where
  totalDistance : ℝ
  totalTime : ℝ
  uphillSpeed : ℝ
  downhillSpeed : ℝ
  uphillTime : ℝ
  downhillTime : ℝ

/-- The system of equations representing Xiaoying's journey --/
def journeyEquations (j : JourneyToSchool) : Prop :=
  (j.uphillSpeed / 60 * j.uphillTime + j.downhillSpeed / 60 * j.downhillTime = j.totalDistance / 1000) ∧
  (j.uphillTime + j.downhillTime = j.totalTime)

/-- Theorem stating that the given conditions satisfy the journey equations --/
theorem xiaoying_journey_equations :
  ∀ (j : JourneyToSchool),
    j.totalDistance = 1200 ∧
    j.totalTime = 16 ∧
    j.uphillSpeed = 3 ∧
    j.downhillSpeed = 5 →
    journeyEquations j :=
by
  sorry

end NUMINAMATH_CALUDE_xiaoying_journey_equations_l812_81285


namespace NUMINAMATH_CALUDE_nina_weekend_sales_l812_81243

/-- Calculates the total amount Nina made over the weekend from jewelry sales --/
def total_sales (necklace_price bracelet_price earring_price ensemble_price : ℚ)
  (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ)
  (necklace_discount bracelet_discount ensemble_discount tax_rate : ℚ)
  (necklace_custom_fee bracelet_custom_fee : ℚ)
  (necklace_customs bracelet_customs : ℕ) : ℚ :=
  let necklace_total := necklace_price * necklaces_sold * (1 - necklace_discount) + necklace_custom_fee * necklace_customs
  let bracelet_total := bracelet_price * bracelets_sold * (1 - bracelet_discount) + bracelet_custom_fee * bracelet_customs
  let earring_total := earring_price * earrings_sold
  let ensemble_total := ensemble_price * ensembles_sold * (1 - ensemble_discount)
  let subtotal := necklace_total + bracelet_total + earring_total + ensemble_total
  subtotal * (1 + tax_rate)

/-- The total amount Nina made over the weekend is $585.90 --/
theorem nina_weekend_sales :
  total_sales 25 15 10 45 5 10 20 2 (1/10) (1/20) (3/20) (2/25) 5 3 1 2 = 58590/100 := by
  sorry

end NUMINAMATH_CALUDE_nina_weekend_sales_l812_81243


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l812_81250

/-- Calculates the amount John paid out of pocket for a new computer and accessories,
    given the costs and the sale of his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value : ℕ)
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400) :
  computer_cost + accessories_cost - (playstation_value * 80 / 100) = 580 := by
  sorry

#check johns_out_of_pocket_expense

end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l812_81250


namespace NUMINAMATH_CALUDE_committee_probability_l812_81248

/-- The number of members in the Literature club -/
def total_members : ℕ := 24

/-- The number of boys in the Literature club -/
def num_boys : ℕ := 12

/-- The number of girls in the Literature club -/
def num_girls : ℕ := 12

/-- The size of the committee to be chosen -/
def committee_size : ℕ := 5

/-- The probability of choosing a committee with at least 2 boys and at least 2 girls -/
theorem committee_probability : 
  (Nat.choose total_members committee_size - 
   (2 * Nat.choose num_boys committee_size + 
    Nat.choose num_boys 1 * Nat.choose num_girls 4 + 
    Nat.choose num_girls 1 * Nat.choose num_boys 4)) / 
   Nat.choose total_members committee_size = 121 / 177 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l812_81248


namespace NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_72_l812_81284

/-- The sum of N consecutive integers starting from a -/
def sumConsecutiveIntegers (N : ℕ) (a : ℤ) : ℤ := N * (2 * a + N - 1) / 2

/-- The proposition that 144 is the greatest number of consecutive integers summing to 72 -/
theorem greatest_consecutive_integers_sum_72 :
  ∀ N : ℕ, (∃ a : ℤ, sumConsecutiveIntegers N a = 72) → N ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_72_l812_81284


namespace NUMINAMATH_CALUDE_square_sum_divided_l812_81222

theorem square_sum_divided : (10^2 + 6^2) / 2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_divided_l812_81222


namespace NUMINAMATH_CALUDE_melanie_dimes_l812_81258

theorem melanie_dimes (initial : Nat) (dad_gift : Nat) (final_total : Nat) 
  (h1 : initial = 19)
  (h2 : dad_gift = 39)
  (h3 : final_total = 83) :
  final_total - (initial + dad_gift) = 25 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l812_81258


namespace NUMINAMATH_CALUDE_sine_power_five_decomposition_l812_81240

theorem sine_power_five_decomposition (b₁ b₂ b₃ b₄ b₅ : ℝ) : 
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 101 / 256 := by
sorry

end NUMINAMATH_CALUDE_sine_power_five_decomposition_l812_81240


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l812_81200

theorem midpoint_sum_equals_vertex_sum (d e f : ℝ) : 
  let vertex_sum := d + e + f
  let midpoint_sum := (d + e) / 2 + (d + f) / 2 + (e + f) / 2
  vertex_sum = midpoint_sum := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l812_81200


namespace NUMINAMATH_CALUDE_students_in_circle_l812_81215

/-- 
Given a circle of students where the 6th and 16th students are opposite each other,
prove that the total number of students is 18.
-/
theorem students_in_circle (n : ℕ) 
  (h1 : n > 0) -- Ensure there are students in the circle
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ a ≤ n ∧ b ≤ n) -- 6th and 16th students exist
  (h3 : (16 - 6) * 2 + 2 = n) -- Condition for 6th and 16th being opposite
  : n = 18 := by
  sorry

end NUMINAMATH_CALUDE_students_in_circle_l812_81215


namespace NUMINAMATH_CALUDE_percentage_green_shirts_l812_81255

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

end NUMINAMATH_CALUDE_percentage_green_shirts_l812_81255


namespace NUMINAMATH_CALUDE_michael_truck_meetings_l812_81283

/-- Represents the number of meetings between Michael and the garbage truck --/
def number_of_meetings : ℕ := 7

/-- Michael's walking speed in feet per second --/
def michael_speed : ℝ := 6

/-- Distance between trash pails in feet --/
def pail_distance : ℝ := 200

/-- Garbage truck's speed in feet per second --/
def truck_speed : ℝ := 10

/-- Time the truck stops at each pail in seconds --/
def truck_stop_time : ℝ := 40

/-- Initial distance between Michael and the truck in feet --/
def initial_distance : ℝ := 250

/-- Theorem stating that Michael and the truck will meet 7 times --/
theorem michael_truck_meetings :
  ∃ (t : ℝ), t > 0 ∧
  (michael_speed * t = truck_speed * (t - truck_stop_time * (number_of_meetings - 1)) + initial_distance) :=
sorry

end NUMINAMATH_CALUDE_michael_truck_meetings_l812_81283


namespace NUMINAMATH_CALUDE_cats_remaining_l812_81293

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l812_81293


namespace NUMINAMATH_CALUDE_range_of_m_l812_81239

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 2/x + 1/y = 1) (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) :
  m ∈ Set.Ioo (-4 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l812_81239


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l812_81288

theorem rectangle_shorter_side 
  (width : Real) 
  (num_poles : Nat) 
  (pole_distance : Real) 
  (h1 : width = 50) 
  (h2 : num_poles = 24) 
  (h3 : pole_distance = 5) : 
  ∃ length : Real, 
    length = 7.5 ∧ 
    length ≤ width ∧ 
    2 * (length + width) = (num_poles - 1 : Real) * pole_distance := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l812_81288


namespace NUMINAMATH_CALUDE_inverse_15_mod_1003_l812_81218

theorem inverse_15_mod_1003 : ∃ x : ℤ, 0 ≤ x ∧ x < 1003 ∧ (15 * x) % 1003 = 1 :=
by
  use 937
  sorry

end NUMINAMATH_CALUDE_inverse_15_mod_1003_l812_81218


namespace NUMINAMATH_CALUDE_min_length_roots_l812_81252

/-- Given a quadratic function f(x) = a x^2 + (16 - a^3) x - 16 a^2, where a > 0,
    the minimum length of the line segment connecting its roots is 12. -/
theorem min_length_roots (a : ℝ) (ha : a > 0) :
  let f := fun x : ℝ => a * x^2 + (16 - a^3) * x - 16 * a^2
  let roots := {x : ℝ | f x = 0}
  let length := fun (r₁ r₂ : ℝ) => |r₁ - r₂|
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧
    ∀ (s₁ s₂ : ℝ), s₁ ∈ roots → s₂ ∈ roots → s₁ ≠ s₂ →
      length r₁ r₂ ≤ length s₁ s₂ ∧ length r₁ r₂ = 12 :=
by sorry


end NUMINAMATH_CALUDE_min_length_roots_l812_81252


namespace NUMINAMATH_CALUDE_polynomial_expansion_l812_81297

theorem polynomial_expansion :
  ∀ x : ℝ, (2 * x^2 - 3 * x + 5) * (x^2 + 4 * x + 3) = 2 * x^4 + 5 * x^3 - x^2 + 11 * x + 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l812_81297


namespace NUMINAMATH_CALUDE_matts_writing_speed_l812_81223

/-- Matt's writing speed problem -/
theorem matts_writing_speed (right_hand_speed : ℕ) (time : ℕ) (difference : ℕ) : 
  right_hand_speed = 10 →
  time = 5 →
  difference = 15 →
  ∃ (left_hand_speed : ℕ), 
    right_hand_speed * time = left_hand_speed * time + difference ∧
    left_hand_speed = 7 :=
by sorry

end NUMINAMATH_CALUDE_matts_writing_speed_l812_81223


namespace NUMINAMATH_CALUDE_prism_volume_l812_81253

/-- The volume of a right rectangular prism with face areas 30, 50, and 75 -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l812_81253


namespace NUMINAMATH_CALUDE_joe_journey_time_l812_81294

/-- Represents Joe's journey from home to the store -/
structure JoeJourney where
  walk_speed : ℝ
  run_speed : ℝ
  walk_time : ℝ
  total_distance : ℝ

/-- Theorem: Joe's total journey time is 15 minutes -/
theorem joe_journey_time (j : JoeJourney) 
  (h1 : j.run_speed = 2 * j.walk_speed)
  (h2 : j.walk_time = 10)
  (h3 : j.total_distance = 2 * (j.walk_speed * j.walk_time)) : 
  j.walk_time + (j.total_distance / 2) / j.run_speed = 15 := by
  sorry

#check joe_journey_time

end NUMINAMATH_CALUDE_joe_journey_time_l812_81294


namespace NUMINAMATH_CALUDE_pencils_to_yuna_l812_81205

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The number of dozens Jimin initially had -/
def initial_dozens : ℕ := 3

/-- The number of pencils Jimin gave to his younger brother -/
def pencils_to_brother : ℕ := 8

/-- The number of pencils Jimin has left -/
def pencils_left : ℕ := 17

/-- Proves that the number of pencils Jimin gave to Yuna is 11 -/
theorem pencils_to_yuna :
  initial_dozens * pencils_per_dozen - pencils_to_brother - pencils_left = 11 := by
  sorry

end NUMINAMATH_CALUDE_pencils_to_yuna_l812_81205


namespace NUMINAMATH_CALUDE_middle_number_proof_l812_81279

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : a + b = 15) (h4 : a + c = 20) (h5 : b + c = 23) (h6 : c = 2 * a) : 
  b = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l812_81279


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l812_81254

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l812_81254


namespace NUMINAMATH_CALUDE_donation_distribution_l812_81216

theorem donation_distribution (giselle_amount sam_amount isabella_amount : ℕ) :
  giselle_amount = 120 →
  isabella_amount = giselle_amount + 15 →
  isabella_amount = sam_amount + 45 →
  (isabella_amount + giselle_amount + sam_amount) / 3 = 115 :=
by sorry

end NUMINAMATH_CALUDE_donation_distribution_l812_81216


namespace NUMINAMATH_CALUDE_vasya_figure_cells_l812_81292

/-- A figure that can be cut into both 2x2 squares and zigzags of 4 cells -/
structure VasyaFigure where
  cells : ℕ
  divisible_by_4 : 4 ∣ cells
  can_cut_into_2x2 : ∃ n : ℕ, cells = 4 * n
  can_cut_into_zigzags : ∃ m : ℕ, cells = 4 * m

/-- The number of cells in Vasya's figure is a multiple of 8 and is at least 16 -/
theorem vasya_figure_cells (fig : VasyaFigure) : 
  ∃ k : ℕ, fig.cells = 8 * k ∧ fig.cells ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_vasya_figure_cells_l812_81292


namespace NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l812_81251

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The daily water consumption of the first sibling -/
def sibling1_daily_consumption : ℕ := 8

/-- The daily water consumption of the second sibling -/
def sibling2_daily_consumption : ℕ := 7

/-- The daily water consumption of the third sibling -/
def sibling3_daily_consumption : ℕ := 9

/-- The total water consumption of all siblings in one week -/
def total_weekly_consumption : ℕ :=
  (sibling1_daily_consumption + sibling2_daily_consumption + sibling3_daily_consumption) * days_in_week

theorem siblings_weekly_water_consumption :
  total_weekly_consumption = 168 := by
  sorry

end NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l812_81251


namespace NUMINAMATH_CALUDE_pool_perimeter_l812_81204

theorem pool_perimeter (garden_length garden_width pool_area : ℝ) 
  (h1 : garden_length = 8)
  (h2 : garden_width = 6)
  (h3 : pool_area = 24)
  (h4 : ∃ x : ℝ, (garden_length - 2*x) * (garden_width - 2*x) = pool_area ∧ 
                 x > 0 ∧ x < garden_length/2 ∧ x < garden_width/2) :
  ∃ pool_length pool_width : ℝ,
    pool_length * pool_width = pool_area ∧
    pool_length < garden_length ∧
    pool_width < garden_width ∧
    2 * pool_length + 2 * pool_width = 20 :=
by sorry

end NUMINAMATH_CALUDE_pool_perimeter_l812_81204


namespace NUMINAMATH_CALUDE_dinner_time_calculation_l812_81271

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  let newHour := (totalMinutes / 60) % 24
  let newMinute := totalMinutes % 60
  ⟨newHour, newMinute, by sorry, by sorry⟩

theorem dinner_time_calculation (start : Time) 
    (h_start : start = ⟨16, 0, by sorry, by sorry⟩)
    (commute : Nat) (h_commute : commute = 30)
    (grocery : Nat) (h_grocery : grocery = 30)
    (drycleaning : Nat) (h_drycleaning : drycleaning = 10)
    (dog : Nat) (h_dog : dog = 20)
    (cooking : Nat) (h_cooking : cooking = 90) :
  addMinutes start (commute + grocery + drycleaning + dog + cooking) = ⟨19, 0, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_dinner_time_calculation_l812_81271


namespace NUMINAMATH_CALUDE_jellybean_box_capacity_l812_81260

theorem jellybean_box_capacity (tim_capacity : ℕ) (scale_factor : ℕ) : 
  tim_capacity = 150 → scale_factor = 3 → 
  (scale_factor ^ 3 : ℕ) * tim_capacity = 4050 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_box_capacity_l812_81260


namespace NUMINAMATH_CALUDE_unique_function_l812_81214

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x + 1/x else 0

theorem unique_function (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x : ℝ, f (2*x) = a * f x + b * x) ∧
  (∀ x y : ℝ, y ≠ 0 → f x * f y = f (x*y) + f (x/y)) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x : ℝ, g (2*x) = a * g x + b * x) ∧
     (∀ x y : ℝ, y ≠ 0 → g x * g y = g (x*y) + g (x/y)))
    → g = f) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_l812_81214


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l812_81221

/-- A regular polygon with exterior angle of 36 degrees has 7 diagonals from each vertex -/
theorem regular_polygon_diagonals (n : ℕ) (h_regular : n ≥ 3) :
  (360 : ℝ) / 36 = n → n - 3 = 7 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l812_81221


namespace NUMINAMATH_CALUDE_original_price_calculation_l812_81262

/-- Proves that given an article sold for $25 with a gain percent of 150%, the original price of the article was $10. -/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 25 ∧ gain_percent = 150 → 
  ∃ (original_price : ℝ), 
    original_price = 10 ∧ 
    selling_price = original_price + (original_price * (gain_percent / 100)) :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l812_81262


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l812_81281

/-- An arithmetic sequence and its partial sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Partial sum sequence
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.d < 0 → ∃ M, ∀ n, seq.S n ≤ M) ∧
  ((∃ M, ∀ n, seq.S n ≤ M) → seq.d < 0) ∧
  (∃ seq : ArithmeticSequence, (∀ n, seq.S (n + 1) > seq.S n) ∧ ∃ k, seq.S k ≤ 0) ∧
  ((∀ n, seq.S n > 0) → ∀ n, seq.S (n + 1) > seq.S n) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l812_81281


namespace NUMINAMATH_CALUDE_palmer_photos_l812_81273

/-- The number of photos Palmer has after her trip to Bali -/
def total_photos (initial_photos : ℕ) (first_week : ℕ) (third_fourth_week : ℕ) : ℕ :=
  initial_photos + first_week + 2 * first_week + third_fourth_week

/-- Theorem stating the total number of photos Palmer has after her trip -/
theorem palmer_photos : 
  total_photos 100 50 80 = 330 := by
  sorry

#eval total_photos 100 50 80

end NUMINAMATH_CALUDE_palmer_photos_l812_81273


namespace NUMINAMATH_CALUDE_fabric_needed_calculation_l812_81220

/-- Calculates the additional fabric needed for dresses -/
def additional_fabric_needed (yards_per_dress : Float) (dresses : Nat) (available : Float) : Float :=
  yards_per_dress * dresses.toFloat * 3 - available

theorem fabric_needed_calculation (floral_yards_per_dress : Float) 
                                  (striped_yards_per_dress : Float)
                                  (polka_dot_yards_per_dress : Float)
                                  (floral_available : Float)
                                  (striped_available : Float)
                                  (polka_dot_available : Float) :
  floral_yards_per_dress = 5.25 →
  striped_yards_per_dress = 6.75 →
  polka_dot_yards_per_dress = 7.15 →
  floral_available = 12 →
  striped_available = 6 →
  polka_dot_available = 15 →
  additional_fabric_needed floral_yards_per_dress 2 floral_available = 19.5 ∧
  additional_fabric_needed striped_yards_per_dress 2 striped_available = 34.5 ∧
  additional_fabric_needed polka_dot_yards_per_dress 2 polka_dot_available = 27.9 :=
by sorry

end NUMINAMATH_CALUDE_fabric_needed_calculation_l812_81220


namespace NUMINAMATH_CALUDE_polynomial_subtraction_l812_81263

/-- Given two polynomials in a and b, prove that their difference is -a^2*b -/
theorem polynomial_subtraction (a b : ℝ) :
  (3 * a^2 * b - 6 * a * b^2) - (2 * a^2 * b - 3 * a * b^2) = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_l812_81263


namespace NUMINAMATH_CALUDE_mary_circus_change_l812_81247

/-- Calculates the change Mary receives after buying circus tickets for herself and her children -/
theorem mary_circus_change (num_children : ℕ) (adult_price child_price payment : ℚ) : 
  num_children = 3 ∧ 
  adult_price = 2 ∧ 
  child_price = 1 ∧ 
  payment = 20 → 
  payment - (adult_price + num_children * child_price) = 15 := by
  sorry

end NUMINAMATH_CALUDE_mary_circus_change_l812_81247


namespace NUMINAMATH_CALUDE_sequence_inequality_l812_81280

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, n^2 - k*n ≥ 3^2 - k*3) → 
  5 ≤ k ∧ k ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l812_81280


namespace NUMINAMATH_CALUDE_exam_average_l812_81265

theorem exam_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 15 →
  n₂ = 10 →
  avg₁ = 70 / 100 →
  avg_total = 80 / 100 →
  ∃ avg₂ : ℚ, 
    (n₁.cast * avg₁ + n₂.cast * avg₂) / (n₁ + n₂).cast = avg_total ∧
    avg₂ = 95 / 100 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_l812_81265


namespace NUMINAMATH_CALUDE_equation_equivalence_l812_81256

theorem equation_equivalence (a b c : ℝ) (h : a + c = 2 * b) : 
  a^2 + 8 * b * c = (2 * b + c)^2 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l812_81256


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_384_l812_81266

/-- A function that counts the number of 4-digit numbers beginning with 2 
    and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let non_two_digits := digits \ {2}
  let count_with_two_twos := 3 * (Finset.card non_two_digits - 1) * (Finset.card non_two_digits - 1)
  let count_with_non_two_pairs := 3 * (Finset.card non_two_digits) * (Finset.card non_two_digits - 1)
  count_with_two_twos + count_with_non_two_pairs

theorem count_special_numbers_eq_384 : count_special_numbers = 384 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_384_l812_81266


namespace NUMINAMATH_CALUDE_tadd_number_count_l812_81217

theorem tadd_number_count : 
  let n : ℕ := 20  -- number of rounds
  let a : ℕ := 1   -- first term of the sequence
  let d : ℕ := 2   -- common difference
  let l : ℕ := a + d * (n - 1)  -- last term
  (n : ℚ) / 2 * (a + l) = 400 := by
  sorry

end NUMINAMATH_CALUDE_tadd_number_count_l812_81217


namespace NUMINAMATH_CALUDE_exists_digit_satisfying_equation_l812_81210

theorem exists_digit_satisfying_equation : ∃ a : ℕ, 
  0 ≤ a ∧ a ≤ 9 ∧ 1111 * a - 1 = (a - 1) ^ (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_exists_digit_satisfying_equation_l812_81210


namespace NUMINAMATH_CALUDE_product_prime_factors_l812_81201

theorem product_prime_factors (m n : ℕ) : 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    m = p₁ * p₂ * p₃ * p₄) →
  (∃ q₁ q₂ q₃ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ 
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
    n = q₁ * q₂ * q₃) →
  Nat.gcd m n = 15 →
  ∃ r₁ r₂ r₃ r₄ r₅ : ℕ, Prime r₁ ∧ Prime r₂ ∧ Prime r₃ ∧ Prime r₄ ∧ Prime r₅ ∧
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ 
    r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧
    r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧
    r₄ ≠ r₅ ∧
    m * n = r₁ * r₂ * r₃ * r₄ * r₅ :=
by
  sorry

end NUMINAMATH_CALUDE_product_prime_factors_l812_81201


namespace NUMINAMATH_CALUDE_x_zero_value_l812_81291

noncomputable def f (x : ℝ) : ℝ := x * (2014 + Real.log x)

theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2015) → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_zero_value_l812_81291


namespace NUMINAMATH_CALUDE_quadratic_solutions_inequality_solution_set_l812_81212

-- Part 1: Quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

theorem quadratic_solutions : 
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = -1 ∧ 
  ∀ x : ℝ, quadratic_equation x ↔ (x = x1 ∨ x = x2) := by sorry

-- Part 2: Inequality system
def inequality_system (x : ℝ) : Prop := 3*x - 1 ≥ 5 ∧ (1 + 2*x) / 3 > x - 1

theorem inequality_solution_set :
  ∀ x : ℝ, inequality_system x ↔ 2 ≤ x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_solutions_inequality_solution_set_l812_81212


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l812_81237

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l812_81237


namespace NUMINAMATH_CALUDE_total_sandwiches_count_l812_81206

/-- The number of people going to the zoo -/
def people : ℝ := 219.0

/-- The number of sandwiches per person -/
def sandwiches_per_person : ℝ := 3.0

/-- The total number of sandwiches prepared -/
def total_sandwiches : ℝ := people * sandwiches_per_person

/-- Theorem stating that the total number of sandwiches is 657.0 -/
theorem total_sandwiches_count : total_sandwiches = 657.0 := by
  sorry

end NUMINAMATH_CALUDE_total_sandwiches_count_l812_81206


namespace NUMINAMATH_CALUDE_sqrt_three_x_minus_two_lt_x_l812_81264

theorem sqrt_three_x_minus_two_lt_x (x : ℝ) : 
  Real.sqrt 3 * x - 2 < x ↔ x < Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_x_minus_two_lt_x_l812_81264


namespace NUMINAMATH_CALUDE_trigonometric_equation_implications_l812_81276

theorem trigonometric_equation_implications (x : Real) 
  (h : (Real.sin (Real.pi + x) + 2 * Real.cos (3 * Real.pi / 2 + x)) / 
       (Real.cos (Real.pi - x) - Real.sin (Real.pi / 2 - x)) = 1) : 
  Real.tan x = 2/3 ∧ Real.sin (2*x) - Real.cos x ^ 2 = 3/13 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_implications_l812_81276


namespace NUMINAMATH_CALUDE_min_triple_intersection_l812_81241

theorem min_triple_intersection (U : Finset Nat) (A B C : Finset Nat) : 
  Finset.card U = 30 →
  Finset.card A = 26 →
  Finset.card B = 23 →
  Finset.card C = 21 →
  A ⊆ U →
  B ⊆ U →
  C ⊆ U →
  10 ≤ Finset.card (A ∩ B ∩ C) :=
by sorry

end NUMINAMATH_CALUDE_min_triple_intersection_l812_81241


namespace NUMINAMATH_CALUDE_coin_game_probability_l812_81228

def num_players : ℕ := 4
def initial_coins : ℕ := 5
def num_rounds : ℕ := 5
def num_balls : ℕ := 5
def num_green : ℕ := 2
def num_red : ℕ := 1
def num_white : ℕ := 2

def coin_transfer : ℕ := 2

def game_round_probability : ℚ := 1 / 5

theorem coin_game_probability :
  (game_round_probability ^ num_rounds : ℚ) = 1 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_coin_game_probability_l812_81228


namespace NUMINAMATH_CALUDE_phenol_red_identifies_urea_decomposing_bacteria_l812_81268

/-- Represents different types of reagents --/
inductive Reagent
  | PhenolRed
  | EMB
  | SudanIII
  | Biuret

/-- Represents a culture medium --/
structure CultureMedium where
  nitrogenSource : String
  reagent : Reagent

/-- Represents the result of a bacterial identification test --/
inductive TestResult
  | Positive
  | Negative

/-- Function to perform urea decomposition test --/
def ureaDecompositionTest (medium : CultureMedium) : TestResult := sorry

/-- Theorem stating that phenol red is the correct reagent for identifying urea-decomposing bacteria --/
theorem phenol_red_identifies_urea_decomposing_bacteria :
  ∀ (medium : CultureMedium),
    medium.nitrogenSource = "urea" →
    medium.reagent = Reagent.PhenolRed →
    ureaDecompositionTest medium = TestResult.Positive :=
  sorry

end NUMINAMATH_CALUDE_phenol_red_identifies_urea_decomposing_bacteria_l812_81268


namespace NUMINAMATH_CALUDE_area_D_n_formula_l812_81287

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The region D_n -/
def D_n (n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 / (n + 1/2) ≤ p.2 ∧ p.2 ≤ floor (p.1 + 1) - p.1 ∧ p.1 ≥ 0}

/-- The area of D_n -/
noncomputable def area_D_n (n : ℝ) : ℝ := sorry

/-- Theorem: The area of D_n is 1/2 * ((n+3/2)/(n+1/2)) for positive n -/
theorem area_D_n_formula (n : ℝ) (hn : n > 0) :
  area_D_n n = 1/2 * ((n + 3/2) / (n + 1/2)) := by sorry

end NUMINAMATH_CALUDE_area_D_n_formula_l812_81287


namespace NUMINAMATH_CALUDE_quadratic_vertex_l812_81242

/-- The quadratic function f(x) = 3(x+4)^2 - 5 has vertex at (-4, -5) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * (x + 4)^2 - 5
  (∀ x, f x = 3 * (x + 4)^2 - 5) →
  ∃! (h k : ℝ), ∀ x, f x = 3 * (x - h)^2 + k ∧ h = -4 ∧ k = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l812_81242


namespace NUMINAMATH_CALUDE_cube_sum_ge_mixed_product_cube_sum_ge_weighted_square_sum_product_l812_81207

-- Problem 1
theorem cube_sum_ge_mixed_product (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 ≥ x^2*y + x*y^2 := by sorry

-- Problem 2
theorem cube_sum_ge_weighted_square_sum_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := by sorry

end NUMINAMATH_CALUDE_cube_sum_ge_mixed_product_cube_sum_ge_weighted_square_sum_product_l812_81207


namespace NUMINAMATH_CALUDE_circumcircle_equation_l812_81229

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, -1)

-- Define point P on the parabola
def P : ℝ × ℝ := (-4, -4)

-- Define the tangent line at P
def tangent_line (x y : ℝ) : Prop := y = 2*x + 4

-- Define point Q as the intersection of the tangent line and x-axis
def Q : ℝ × ℝ := (-2, 0)

-- Theorem statement
theorem circumcircle_equation :
  ∀ x y : ℝ,
  parabola (P.1) (P.2) →
  tangent_line (Q.1) (Q.2) →
  (x^2 + y^2 + 4*x + 5*y + 4 = 0) ↔ 
  ((x - (-2))^2 + (y - (-5/2))^2 = (5/2)^2) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l812_81229


namespace NUMINAMATH_CALUDE_goats_and_hens_total_amount_l812_81231

/-- The total amount spent on goats and hens -/
def total_amount (num_goats num_hens goat_price hen_price : ℕ) : ℕ :=
  num_goats * goat_price + num_hens * hen_price

/-- Theorem: The total amount spent on 5 goats at Rs. 400 each and 10 hens at Rs. 50 each is Rs. 2500 -/
theorem goats_and_hens_total_amount :
  total_amount 5 10 400 50 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_goats_and_hens_total_amount_l812_81231


namespace NUMINAMATH_CALUDE_total_paper_pieces_l812_81236

theorem total_paper_pieces : 
  let olivia_pieces : ℕ := 127
  let edward_pieces : ℕ := 345
  let sam_pieces : ℕ := 518
  olivia_pieces + edward_pieces + sam_pieces = 990 :=
by sorry

end NUMINAMATH_CALUDE_total_paper_pieces_l812_81236


namespace NUMINAMATH_CALUDE_grid_coloring_count_l812_81238

/-- Represents the number of valid colorings for a 2 × n grid -/
def num_colorings (n : ℕ) : ℕ :=
  3^(n-1)

/-- Theorem stating the number of distinct colorings for the grid -/
theorem grid_coloring_count (n : ℕ) (h : n ≥ 2) :
  let grid_size := 2 * n
  let colored_endpoints := 3
  let vertices_to_color := grid_size - colored_endpoints
  let num_colors := 3
  num_colorings n = num_colors^(n-1) :=
by sorry

end NUMINAMATH_CALUDE_grid_coloring_count_l812_81238


namespace NUMINAMATH_CALUDE_lemon_cupcakes_total_l812_81257

theorem lemon_cupcakes_total (cupcakes_at_home : ℕ) (boxes_given : ℕ) (cupcakes_per_box : ℕ) : 
  cupcakes_at_home = 2 → boxes_given = 17 → cupcakes_per_box = 3 →
  cupcakes_at_home + boxes_given * cupcakes_per_box = 53 := by
  sorry

end NUMINAMATH_CALUDE_lemon_cupcakes_total_l812_81257


namespace NUMINAMATH_CALUDE_sin_2alpha_in_terms_of_k_l812_81203

theorem sin_2alpha_in_terms_of_k (k α : ℝ) (h : Real.cos (π / 4 - α) = k) :
  Real.sin (2 * α) = 2 * k^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_in_terms_of_k_l812_81203


namespace NUMINAMATH_CALUDE_simplify_expression_l812_81224

theorem simplify_expression (a b : ℝ) : 
  (1 : ℝ) * (2 * a) * (3 * b) * (4 * a^2 * b) * (5 * a^3 * b^2) = 120 * a^6 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l812_81224


namespace NUMINAMATH_CALUDE_power_of_negative_product_l812_81233

theorem power_of_negative_product (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l812_81233


namespace NUMINAMATH_CALUDE_smallest_consecutive_primes_sum_after_13_l812_81246

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def consecutive_primes (p q r s : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
  q = p.succ ∧ r = q.succ ∧ s = r.succ

theorem smallest_consecutive_primes_sum_after_13 :
  ∃ (p q r s : ℕ),
    consecutive_primes p q r s ∧
    p > 13 ∧
    4 ∣ (p + q + r + s) ∧
    (p + q + r + s = 88) ∧
    ∀ (a b c d : ℕ),
      consecutive_primes a b c d → a > 13 → 4 ∣ (a + b + c + d) →
      (a + b + c + d ≥ p + q + r + s) :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_primes_sum_after_13_l812_81246


namespace NUMINAMATH_CALUDE_crayon_selection_ways_l812_81202

def total_crayons : ℕ := 15
def red_crayons : ℕ := 4
def selection_size : ℕ := 5

theorem crayon_selection_ways : 
  (Nat.choose total_crayons selection_size) -
  (Nat.choose red_crayons 2 * Nat.choose (total_crayons - red_crayons) (selection_size - 2)) +
  (Nat.choose red_crayons 1 * Nat.choose (total_crayons - red_crayons) (selection_size - 1)) +
  (Nat.choose (total_crayons - red_crayons) selection_size) = 1782 :=
by sorry

end NUMINAMATH_CALUDE_crayon_selection_ways_l812_81202


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l812_81211

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x - m = 0 ∧ y^2 - 6*y - m = 0) ↔ m > -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l812_81211


namespace NUMINAMATH_CALUDE_hiker_distance_l812_81244

theorem hiker_distance (north east south east2 : ℝ) 
  (h1 : north = 15)
  (h2 : east = 8)
  (h3 : south = 9)
  (h4 : east2 = 2) : 
  Real.sqrt ((north - south)^2 + (east + east2)^2) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l812_81244


namespace NUMINAMATH_CALUDE_wendi_chicken_count_l812_81208

def chicken_count (initial : ℕ) : ℕ :=
  let step1 := initial + (initial / 2) ^ 2
  let step2 := step1 - 3
  let step3 := step2 + ((4 * step2 - 28) / 7)
  step3 + 3

theorem wendi_chicken_count : chicken_count 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_wendi_chicken_count_l812_81208


namespace NUMINAMATH_CALUDE_computer_desk_prices_l812_81299

theorem computer_desk_prices :
  ∃ (x y : ℝ),
    (10 * x + 200 * y = 90000) ∧
    (12 * x + 120 * y = 90000) ∧
    (x = 6000) ∧
    (y = 150) := by
  sorry

end NUMINAMATH_CALUDE_computer_desk_prices_l812_81299


namespace NUMINAMATH_CALUDE_nori_crayons_l812_81209

theorem nori_crayons (initial_boxes : ℕ) (crayons_per_box : ℕ) (crayons_left : ℕ) (extra_to_lea : ℕ) :
  initial_boxes = 4 →
  crayons_per_box = 8 →
  crayons_left = 15 →
  extra_to_lea = 7 →
  ∃ (crayons_to_mae : ℕ),
    initial_boxes * crayons_per_box = crayons_left + crayons_to_mae + (crayons_to_mae + extra_to_lea) ∧
    crayons_to_mae = 5 :=
by sorry

end NUMINAMATH_CALUDE_nori_crayons_l812_81209


namespace NUMINAMATH_CALUDE_cube_number_sum_l812_81219

theorem cube_number_sum :
  ∀ (a b c d e f : ℕ),
  -- The numbers are consecutive whole numbers between 15 and 20
  15 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f ≤ 20 →
  -- The sum of opposite faces is equal
  a + f = b + e ∧ b + e = c + d →
  -- The middle number in the range is the largest on one face
  (d = 18 ∨ c = 18) →
  -- The sum of all numbers is 105
  a + b + c + d + e + f = 105 :=
by sorry

end NUMINAMATH_CALUDE_cube_number_sum_l812_81219


namespace NUMINAMATH_CALUDE_P_range_l812_81286

theorem P_range (x : ℝ) (P : ℝ) 
  (h1 : x^2 - 5*x + 6 < 0) 
  (h2 : P = x^2 + 5*x + 6) : 
  20 < P ∧ P < 30 := by
  sorry

end NUMINAMATH_CALUDE_P_range_l812_81286


namespace NUMINAMATH_CALUDE_comic_book_pages_l812_81296

theorem comic_book_pages (total_frames : Nat) (frames_per_page : Nat) 
  (h1 : total_frames = 143)
  (h2 : frames_per_page = 11) :
  (total_frames / frames_per_page = 13) ∧ (total_frames % frames_per_page = 0) := by
  sorry

end NUMINAMATH_CALUDE_comic_book_pages_l812_81296


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l812_81277

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arith_prop : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The main theorem -/
theorem arithmetic_sequence_m_value (seq : ArithmeticSequence) (m : ℕ) 
    (h1 : seq.S (m - 1) = -2)
    (h2 : seq.S m = 0)
    (h3 : seq.S (m + 1) = 3) :
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l812_81277


namespace NUMINAMATH_CALUDE_power_equality_l812_81290

theorem power_equality : 32^5 * 4^3 = 2^31 := by sorry

end NUMINAMATH_CALUDE_power_equality_l812_81290


namespace NUMINAMATH_CALUDE_largest_b_divisible_by_four_l812_81235

theorem largest_b_divisible_by_four :
  let n : ℕ → ℕ := λ b => 4000000 + b * 100000 + 508632
  ∃ (b : ℕ), b ≤ 9 ∧ n b % 4 = 0 ∧ ∀ (x : ℕ), x ≤ 9 ∧ n x % 4 = 0 → x ≤ b :=
by sorry

end NUMINAMATH_CALUDE_largest_b_divisible_by_four_l812_81235


namespace NUMINAMATH_CALUDE_q_value_at_minus_one_l812_81230

-- Define the polynomial q(x)
def q (a b x : ℤ) : ℤ := x^2 + a*x + b

-- Define the two polynomials that q(x) divides
def p1 (x : ℤ) : ℤ := x^4 + 8*x^2 + 49
def p2 (x : ℤ) : ℤ := 2*x^4 + 5*x^2 + 18*x + 3

-- Theorem statement
theorem q_value_at_minus_one 
  (a b : ℤ) 
  (h1 : ∀ x, (p1 x) % (q a b x) = 0)
  (h2 : ∀ x, (p2 x) % (q a b x) = 0) :
  q a b (-1) = 66 := by
  sorry

end NUMINAMATH_CALUDE_q_value_at_minus_one_l812_81230


namespace NUMINAMATH_CALUDE_cuboid_diagonal_l812_81298

/-- Given a cuboid with dimensions a, b, and c, if its surface area is 11
    and the sum of the lengths of its twelve edges is 24,
    then the length of its diagonal is 5. -/
theorem cuboid_diagonal (a b c : ℝ) 
    (h1 : 2 * (a * b + b * c + a * c) = 11)  -- surface area condition
    (h2 : 4 * (a + b + c) = 24) :            -- sum of edges condition
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_l812_81298


namespace NUMINAMATH_CALUDE_inequality_proof_l812_81278

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a * b * (b + 1) * (c + 1)) + 1 / (b * c * (c + 1) * (a + 1)) + 1 / (c * a * (a + 1) * (b + 1)) ≥ 3 / (1 + a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l812_81278


namespace NUMINAMATH_CALUDE_power_sum_product_equals_l812_81267

theorem power_sum_product_equals : (6^3 + 4^2) * 7^5 = 3897624 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_product_equals_l812_81267


namespace NUMINAMATH_CALUDE_inverse_34_mod_47_l812_81272

theorem inverse_34_mod_47 (h : (13⁻¹ : ZMod 47) = 29) : (34⁻¹ : ZMod 47) = 18 := by
  sorry

end NUMINAMATH_CALUDE_inverse_34_mod_47_l812_81272


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l812_81261

theorem new_average_age_with_teacher (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 30 →
  student_avg_age = 14 →
  teacher_age = 45 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_with_teacher_l812_81261


namespace NUMINAMATH_CALUDE_no_identical_lines_l812_81249

theorem no_identical_lines : ¬∃ (a d : ℝ), ∀ (x y : ℝ),
  (5 * x + a * y + d = 0 ↔ 2 * d * x - 3 * y + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_identical_lines_l812_81249


namespace NUMINAMATH_CALUDE_number_of_bs_l812_81269

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

end NUMINAMATH_CALUDE_number_of_bs_l812_81269


namespace NUMINAMATH_CALUDE_train_crossing_time_l812_81245

/-- Time taken for a faster train to cross a man in a slower train -/
theorem train_crossing_time (faster_speed slower_speed : ℝ) (train_length : ℝ) : 
  faster_speed = 54 →
  slower_speed = 36 →
  train_length = 135 →
  (train_length / (faster_speed - slower_speed)) * (3600 / 1000) = 27 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l812_81245


namespace NUMINAMATH_CALUDE_cubic_equation_result_l812_81234

theorem cubic_equation_result (x : ℝ) (h : x^3 + 2*x = 4) : x^7 + 32*x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_result_l812_81234
