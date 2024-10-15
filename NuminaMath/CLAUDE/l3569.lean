import Mathlib

namespace NUMINAMATH_CALUDE_angle_condition_implies_y_range_l3569_356951

/-- Given points A(-1,1) and B(3,y), and vector a = (1,2), if the angle between AB and a is acute, 
    then y ∈ (-1,9) ∪ (9,+∞). -/
theorem angle_condition_implies_y_range (y : ℝ) : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (3, y)
  let a : ℝ × ℝ := (1, 2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (AB.1 * a.1 + AB.2 * a.2 > 0) → -- Dot product > 0 implies acute angle
  (y ∈ Set.Ioo (-1 : ℝ) 9 ∪ Set.Ioi 9) := by
sorry

end NUMINAMATH_CALUDE_angle_condition_implies_y_range_l3569_356951


namespace NUMINAMATH_CALUDE_box_volume_conversion_l3569_356979

theorem box_volume_conversion (box_volume_cubic_feet : ℝ) :
  box_volume_cubic_feet = 216 →
  box_volume_cubic_feet / 27 = 8 :=
by sorry

end NUMINAMATH_CALUDE_box_volume_conversion_l3569_356979


namespace NUMINAMATH_CALUDE_ernest_wire_problem_l3569_356952

theorem ernest_wire_problem (total_parts : ℕ) (used_parts : ℕ) (unused_length : ℝ) :
  total_parts = 5 ∧ used_parts = 3 ∧ unused_length = 20 →
  total_parts * (unused_length / (total_parts - used_parts)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ernest_wire_problem_l3569_356952


namespace NUMINAMATH_CALUDE_sqrt_five_fourth_power_l3569_356928

theorem sqrt_five_fourth_power : (Real.sqrt 5) ^ 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_fourth_power_l3569_356928


namespace NUMINAMATH_CALUDE_minimum_researchers_l3569_356929

theorem minimum_researchers (genetics : ℕ) (microbiology : ℕ) (both : ℕ)
  (h1 : genetics = 120)
  (h2 : microbiology = 90)
  (h3 : both = 40) :
  genetics + microbiology - both = 170 := by
  sorry

end NUMINAMATH_CALUDE_minimum_researchers_l3569_356929


namespace NUMINAMATH_CALUDE_advanced_purchase_tickets_l3569_356926

/-- Proves that the number of advanced-purchase tickets sold is 40 --/
theorem advanced_purchase_tickets (total_tickets : ℕ) (total_amount : ℕ) 
  (advanced_price : ℕ) (door_price : ℕ) (h1 : total_tickets = 140) 
  (h2 : total_amount = 1720) (h3 : advanced_price = 8) (h4 : door_price = 14) :
  ∃ (advanced_tickets : ℕ) (door_tickets : ℕ),
    advanced_tickets + door_tickets = total_tickets ∧
    advanced_price * advanced_tickets + door_price * door_tickets = total_amount ∧
    advanced_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_advanced_purchase_tickets_l3569_356926


namespace NUMINAMATH_CALUDE_total_pears_picked_l3569_356946

theorem total_pears_picked (alyssa_pears nancy_pears : ℕ) 
  (h1 : alyssa_pears = 42) 
  (h2 : nancy_pears = 17) : 
  alyssa_pears + nancy_pears = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l3569_356946


namespace NUMINAMATH_CALUDE_gum_pack_size_l3569_356906

theorem gum_pack_size (mint_gum orange_gum y : ℕ) : 
  mint_gum = 24 → 
  orange_gum = 36 → 
  (mint_gum - 2 * y) / orange_gum = mint_gum / (orange_gum + 4 * y) → 
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_gum_pack_size_l3569_356906


namespace NUMINAMATH_CALUDE_book_categorization_l3569_356927

/-- Proves that given 800 books initially divided into 4 equal categories, 
    then each category divided into 5 groups, the number of final categories 
    when each group is further divided into categories of 20 books each is 40. -/
theorem book_categorization (total_books : Nat) (initial_categories : Nat) 
    (groups_per_category : Nat) (books_per_final_category : Nat) 
    (h1 : total_books = 800)
    (h2 : initial_categories = 4)
    (h3 : groups_per_category = 5)
    (h4 : books_per_final_category = 20) : 
    (total_books / initial_categories / groups_per_category / books_per_final_category) * 
    (initial_categories * groups_per_category) = 40 := by
  sorry

#check book_categorization

end NUMINAMATH_CALUDE_book_categorization_l3569_356927


namespace NUMINAMATH_CALUDE_min_mn_value_l3569_356953

def f (x a : ℝ) : ℝ := |x - a|

theorem min_mn_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f x 1 ≤ 1 ↔ x ∈ Set.Icc 0 2) →
  1 / m + 1 / (2 * n) = 1 →
  ∀ k, m * n ≥ k → k = 2 :=
sorry

end NUMINAMATH_CALUDE_min_mn_value_l3569_356953


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_two_lines_l3569_356943

/-- The maximum number of intersection points between circles -/
def max_circle_intersections (n : ℕ) : ℕ := n * (n - 1)

/-- The maximum number of intersection points between circles and lines -/
def max_circle_line_intersections (circles : ℕ) (lines : ℕ) : ℕ :=
  circles * lines * 2

/-- The maximum number of intersection points between lines -/
def max_line_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total maximum number of intersection points -/
def total_max_intersections (circles : ℕ) (lines : ℕ) : ℕ :=
  max_circle_intersections circles +
  max_circle_line_intersections circles lines +
  max_line_intersections lines

theorem max_intersections_three_circles_two_lines :
  total_max_intersections 3 2 = 19 := by sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_two_lines_l3569_356943


namespace NUMINAMATH_CALUDE_friends_meet_time_l3569_356913

def carl_lap : ℕ := 5
def jenna_lap : ℕ := 8
def marco_lap : ℕ := 9
def leah_lap : ℕ := 10

def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight

theorem friends_meet_time :
  let meeting_time := start_time + Nat.lcm carl_lap (Nat.lcm jenna_lap (Nat.lcm marco_lap leah_lap))
  meeting_time = 15 * 60  -- 3:00 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_friends_meet_time_l3569_356913


namespace NUMINAMATH_CALUDE_plane_speed_l3569_356940

/-- The speed of a plane in still air, given its performance with and against wind. -/
theorem plane_speed (distance_with_wind : ℝ) (distance_against_wind : ℝ) (wind_speed : ℝ) :
  distance_with_wind = 400 →
  distance_against_wind = 320 →
  wind_speed = 20 →
  ∃ (plane_speed : ℝ),
    distance_with_wind / (plane_speed + wind_speed) = distance_against_wind / (plane_speed - wind_speed) ∧
    plane_speed = 180 :=
by sorry

end NUMINAMATH_CALUDE_plane_speed_l3569_356940


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3569_356963

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_subsequence : (a 3) ^ 2 = a 2 * a 7
  initial_condition : 2 * a 1 + a 2 = 1

/-- The general term of the arithmetic sequence is 5/3 - n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n, seq.a n = 5/3 - n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3569_356963


namespace NUMINAMATH_CALUDE_outfit_count_l3569_356949

/-- The number of outfits that can be made with different colored shirts and hats -/
def number_of_outfits : ℕ :=
  let red_shirts := 7
  let blue_shirts := 5
  let green_shirts := 8
  let pants := 10
  let green_hats := 10
  let red_hats := 6
  let blue_hats := 7
  (red_shirts * pants * (green_hats + blue_hats)) +
  (blue_shirts * pants * (green_hats + red_hats)) +
  (green_shirts * pants * (red_hats + blue_hats))

theorem outfit_count : number_of_outfits = 3030 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l3569_356949


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l3569_356914

/-- Given a train that travels at 21 kmph including stoppages and stops for 18 minutes per hour,
    its speed excluding stoppages is 30 kmph. -/
theorem train_speed_excluding_stoppages
  (speed_with_stops : ℝ)
  (stop_time : ℝ)
  (h1 : speed_with_stops = 21)
  (h2 : stop_time = 18)
  : (speed_with_stops * 60) / (60 - stop_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l3569_356914


namespace NUMINAMATH_CALUDE_one_mile_equals_600_rods_l3569_356911

/-- Conversion factor from miles to furlongs -/
def mile_to_furlong : ℚ := 12

/-- Conversion factor from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

/-- Theorem stating that one mile is equal to 600 rods -/
theorem one_mile_equals_600_rods : rods_in_mile = 600 := by
  sorry

end NUMINAMATH_CALUDE_one_mile_equals_600_rods_l3569_356911


namespace NUMINAMATH_CALUDE_common_difference_is_half_l3569_356986

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 1 + a 6 + a 11 = 6
  fourth_term : a 4 = 1

/-- The common difference of an arithmetic sequence is 1/2 given the conditions -/
theorem common_difference_is_half (seq : ArithmeticSequence) : 
  ∃ d : ℚ, (∀ n : ℕ, seq.a (n + 1) - seq.a n = d) ∧ d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_half_l3569_356986


namespace NUMINAMATH_CALUDE_kirsten_stole_14_meatballs_l3569_356972

/-- The number of meatballs Kirsten stole -/
def meatballs_stolen (initial final : ℕ) : ℕ := initial - final

/-- Proof that Kirsten stole 14 meatballs -/
theorem kirsten_stole_14_meatballs (initial final : ℕ) 
  (h_initial : initial = 25)
  (h_final : final = 11) : 
  meatballs_stolen initial final = 14 := by
  sorry

end NUMINAMATH_CALUDE_kirsten_stole_14_meatballs_l3569_356972


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l3569_356904

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, 3 * x - 4 * y = 6 ∧ k * x + 2 * y = 8 ∧ x > 0 ∧ y > 0) ↔ -3/2 < k ∧ k < 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l3569_356904


namespace NUMINAMATH_CALUDE_intersection_A_B_l3569_356912

-- Define the sets A and B
def A : Set ℝ := {x | x ≠ 3 ∧ x ≥ 2}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

-- Define the interval (3,5]
def interval_3_5 : Set ℝ := {x | 3 < x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = interval_3_5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3569_356912


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3569_356915

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ -1) ∧ 
  (∀ x : ℝ, x = -1 → x^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3569_356915


namespace NUMINAMATH_CALUDE_family_change_is_74_l3569_356919

/-- Represents the cost of tickets for a family visit to an amusement park --/
def amusement_park_change (regular_price : ℕ) (child_discount : ℕ) (amount_given : ℕ) : ℕ :=
  let adult_cost := regular_price
  let child_cost := regular_price - child_discount
  let total_cost := 2 * adult_cost + 2 * child_cost
  amount_given - total_cost

/-- Theorem stating that the change received by the family is $74 --/
theorem family_change_is_74 :
  amusement_park_change 109 5 500 = 74 := by
  sorry

end NUMINAMATH_CALUDE_family_change_is_74_l3569_356919


namespace NUMINAMATH_CALUDE_identity_is_unique_divisibility_function_l3569_356969

/-- A function f: ℕ → ℕ satisfying the divisibility condition -/
def DivisibilityFunction (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, (m^2 + n)^2 % (f m^2 + f n) = 0

/-- The theorem stating that the identity function is the only function satisfying the divisibility condition -/
theorem identity_is_unique_divisibility_function :
  ∀ f : ℕ → ℕ, DivisibilityFunction f ↔ ∀ n : ℕ, f n = n :=
sorry

end NUMINAMATH_CALUDE_identity_is_unique_divisibility_function_l3569_356969


namespace NUMINAMATH_CALUDE_cubic_equation_solution_mean_l3569_356935

theorem cubic_equation_solution_mean :
  let f : ℝ → ℝ := λ x => x^3 + 5*x^2 - 14*x
  let solutions := {x : ℝ | f x = 0}
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, f x = 0) ∧ 
    (s.sum id) / s.card = -5/3 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_mean_l3569_356935


namespace NUMINAMATH_CALUDE_remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty_l3569_356978

theorem remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty :
  (19^19 + 19) % 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty_l3569_356978


namespace NUMINAMATH_CALUDE_park_short_trees_after_planting_l3569_356907

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees newly_planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + newly_planted_short_trees

/-- Theorem stating that the total number of short trees after planting is 98 -/
theorem park_short_trees_after_planting :
  total_short_trees 41 57 = 98 := by
  sorry


end NUMINAMATH_CALUDE_park_short_trees_after_planting_l3569_356907


namespace NUMINAMATH_CALUDE_rectangle_length_l3569_356962

/-- The length of a rectangle with width 4 cm and area equal to a square with side length 8 cm -/
theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 8 →
  rect_width = 4 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3569_356962


namespace NUMINAMATH_CALUDE_watson_second_graders_l3569_356942

/-- Represents the number of students in each grade and the total in Ms. Watson's class -/
structure ClassComposition where
  total : Nat
  kindergartners : Nat
  firstGraders : Nat
  thirdGraders : Nat
  absentStudents : Nat

/-- Calculates the number of second graders in the class -/
def secondGraders (c : ClassComposition) : Nat :=
  c.total - (c.kindergartners + c.firstGraders + c.thirdGraders + c.absentStudents)

/-- Theorem stating the number of second graders in Ms. Watson's class -/
theorem watson_second_graders :
  let c : ClassComposition := {
    total := 120,
    kindergartners := 34,
    firstGraders := 48,
    thirdGraders := 5,
    absentStudents := 6
  }
  secondGraders c = 27 := by sorry

end NUMINAMATH_CALUDE_watson_second_graders_l3569_356942


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3569_356916

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (A < π) →
  (B > 0) → (B < π) →
  (C > 0) → (C < π) →
  (A + B + C = π) →
  ((Real.sqrt 3 * a) / (1 + Real.cos A) = c / Real.sin C) →
  (a = Real.sqrt 3) →
  (c - b = (Real.sqrt 6 - Real.sqrt 2) / 2) →
  (A = π / 3 ∧ (1/2 * b * c * Real.sin A = (3 + Real.sqrt 3) / 4)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3569_356916


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3569_356930

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp (a * x)

theorem tangent_line_at_origin (a : ℝ) (h : a ≠ 0) :
  let tangent_line (x : ℝ) := -x - 1
  ∀ x, tangent_line x = f a 0 + (deriv (f a)) 0 * x :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l3569_356930


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l3569_356973

/-- Theorem: Distance between a point on a parabola and its focus
Given a parabola y^2 = 16x with focus F at (4, 0), and a point P on the parabola
that is 12 units away from the x-axis, the distance between P and F is 13 units. -/
theorem parabola_point_focus_distance
  (P : ℝ × ℝ) -- Point P on the parabola
  (h_on_parabola : (P.2)^2 = 16 * P.1) -- P satisfies the parabola equation
  (h_distance_from_x_axis : abs P.2 = 12) -- P is 12 units from x-axis
  : Real.sqrt ((P.1 - 4)^2 + P.2^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l3569_356973


namespace NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l3569_356985

theorem power_of_seven_mod_twelve : 7^203 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l3569_356985


namespace NUMINAMATH_CALUDE_martha_tech_support_ratio_l3569_356922

/-- Proves that the ratio of yelling time to hold time is 1:2 given the conditions of Martha's tech support experience. -/
theorem martha_tech_support_ratio :
  ∀ (router_time hold_time yelling_time : ℕ),
    router_time = 10 →
    hold_time = 6 * router_time →
    router_time + hold_time + yelling_time = 100 →
    (yelling_time : ℚ) / hold_time = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_martha_tech_support_ratio_l3569_356922


namespace NUMINAMATH_CALUDE_symmetry_axis_of_sin_cos_function_l3569_356900

theorem symmetry_axis_of_sin_cos_function :
  ∃ (x : ℝ), x = π / 12 ∧
  ∀ (y : ℝ), y = Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) →
  (∀ (t : ℝ), y = Real.sin (2 * (x + t)) - Real.sqrt 3 * Real.cos (2 * (x + t)) ↔
               y = Real.sin (2 * (x - t)) - Real.sqrt 3 * Real.cos (2 * (x - t))) :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_sin_cos_function_l3569_356900


namespace NUMINAMATH_CALUDE_sum_of_ninth_powers_l3569_356945

theorem sum_of_ninth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^9 + b^9 = 76 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ninth_powers_l3569_356945


namespace NUMINAMATH_CALUDE_range_of_a_l3569_356993

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2) - Real.log (abs x)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f (-a*x + Real.log x + 1) + f (a*x - Real.log x - 1) ≥ 2 * f 1) →
  a ∈ Set.Icc (1 / Real.exp 1) ((2 + Real.log 3) / 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3569_356993


namespace NUMINAMATH_CALUDE_quadratic_points_ordering_l3569_356990

theorem quadratic_points_ordering (m : ℝ) (y₁ y₂ y₃ : ℝ) :
  ((-1)^2 + 2*(-1) + m = y₁) →
  (3^2 + 2*3 + m = y₂) →
  ((1/2)^2 + 2*(1/2) + m = y₃) →
  (y₂ > y₃ ∧ y₃ > y₁) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_points_ordering_l3569_356990


namespace NUMINAMATH_CALUDE_constant_expression_value_l3569_356932

-- Define the triangle DEF
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define the properties of the triangle
def Triangle.sideLengths (t : Triangle) : ℝ × ℝ × ℝ := sorry
def Triangle.circumradius (t : Triangle) : ℝ := sorry
def Triangle.orthocenter (t : Triangle) : ℝ × ℝ := sorry
def Triangle.circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Define the constant expression
def constantExpression (t : Triangle) (Q : ℝ × ℝ) : ℝ :=
  let (d, e, f) := t.sideLengths
  let H := t.orthocenter
  let S := t.circumradius
  (Q.1 - t.D.1)^2 + (Q.2 - t.D.2)^2 +
  (Q.1 - t.E.1)^2 + (Q.2 - t.E.2)^2 +
  (Q.1 - t.F.1)^2 + (Q.2 - t.F.2)^2 -
  ((Q.1 - H.1)^2 + (Q.2 - H.2)^2)

-- State the theorem
theorem constant_expression_value (t : Triangle) :
  ∀ Q ∈ t.circumcircle, constantExpression t Q = 
    let (d, e, f) := t.sideLengths
    let S := t.circumradius
    d^2 + e^2 + f^2 - 4 * S^2 :=
sorry

end NUMINAMATH_CALUDE_constant_expression_value_l3569_356932


namespace NUMINAMATH_CALUDE_min_value_sin_function_l3569_356917

theorem min_value_sin_function (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x - π / 4)) :
  ∃ x ∈ Set.Icc 0 (π / 2), ∀ y ∈ Set.Icc 0 (π / 2), f x ≤ f y ∧ f x = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sin_function_l3569_356917


namespace NUMINAMATH_CALUDE_line_plane_relations_l3569_356950

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_relations 
  (l m : Line) (α : Plane) 
  (h : perpendicular l α) : 
  (perpendicular m α → parallel_lines m l) ∧ 
  (parallel m α → perpendicular_lines m l) ∧ 
  (parallel_lines m l → perpendicular m α) := by
  sorry

end NUMINAMATH_CALUDE_line_plane_relations_l3569_356950


namespace NUMINAMATH_CALUDE_regular_150_sided_polygon_diagonals_l3569_356939

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 150 sides has 11025 diagonals -/
theorem regular_150_sided_polygon_diagonals :
  num_diagonals 150 = 11025 := by sorry

end NUMINAMATH_CALUDE_regular_150_sided_polygon_diagonals_l3569_356939


namespace NUMINAMATH_CALUDE_emma_age_l3569_356977

/-- Represents the ages of the individuals --/
structure Ages where
  oliver : ℕ
  nancy : ℕ
  liam : ℕ
  emma : ℕ

/-- The age relationships between Oliver, Nancy, Liam, and Emma --/
def age_relationships (ages : Ages) : Prop :=
  ages.oliver + 5 = ages.nancy ∧
  ages.nancy = ages.liam + 6 ∧
  ages.emma = ages.liam + 4 ∧
  ages.oliver = 16

/-- Theorem stating that given the age relationships and Oliver's age, Emma is 19 years old --/
theorem emma_age (ages : Ages) : age_relationships ages → ages.emma = 19 := by
  sorry

end NUMINAMATH_CALUDE_emma_age_l3569_356977


namespace NUMINAMATH_CALUDE_product_evaluation_l3569_356960

theorem product_evaluation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^2 + y^2 + z^2)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) * (x^2*y^2 + y^2*z^2 + z^2*x^2)⁻¹ * ((x*y)⁻¹ + (y*z)⁻¹ + (z*x)⁻¹)) =
  ((x*y + y*z + z*x) * (x + y + z)) / (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) :=
by sorry

end NUMINAMATH_CALUDE_product_evaluation_l3569_356960


namespace NUMINAMATH_CALUDE_diagonals_100_sided_polygon_l3569_356987

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 100 sides is 4850 -/
theorem diagonals_100_sided_polygon : num_diagonals 100 = 4850 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_100_sided_polygon_l3569_356987


namespace NUMINAMATH_CALUDE_equal_split_donation_l3569_356967

def total_donation : ℝ := 15000
def donation1 : ℝ := 3500
def donation2 : ℝ := 2750
def donation3 : ℝ := 3870
def donation4 : ℝ := 2475
def num_remaining_homes : ℕ := 4

theorem equal_split_donation :
  let donated_sum := donation1 + donation2 + donation3 + donation4
  let remaining := total_donation - donated_sum
  remaining / num_remaining_homes = 601.25 := by
sorry

end NUMINAMATH_CALUDE_equal_split_donation_l3569_356967


namespace NUMINAMATH_CALUDE_joan_seashells_l3569_356931

def seashell_problem (initial found : ℕ) (given_away : ℕ) (additional_found : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial - given_away + additional_found - traded - lost

theorem joan_seashells :
  seashell_problem 79 63 45 20 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l3569_356931


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3569_356941

/-- A continuous function satisfying the given functional equation -/
structure FunctionalEquation where
  f : ℝ → ℝ
  continuous : Continuous f
  equation : ∀ x y, f (x + y) = f x + f y + f x * f y

/-- The theorem stating the form of the function satisfying the equation -/
theorem functional_equation_solution (fe : FunctionalEquation) :
  ∃ a : ℝ, a ≥ 1 ∧ ∀ x, fe.f x = a^x - 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3569_356941


namespace NUMINAMATH_CALUDE_enrique_commission_is_300_l3569_356944

-- Define the commission rate
def commission_rate : Real := 0.15

-- Define the sales data
def suits_sold : Nat := 2
def suit_price : Real := 700.00
def shirts_sold : Nat := 6
def shirt_price : Real := 50.00
def loafers_sold : Nat := 2
def loafer_price : Real := 150.00

-- Calculate total sales
def total_sales : Real :=
  (suits_sold : Real) * suit_price +
  (shirts_sold : Real) * shirt_price +
  (loafers_sold : Real) * loafer_price

-- Calculate Enrique's commission
def enrique_commission : Real := commission_rate * total_sales

-- Theorem to prove
theorem enrique_commission_is_300 :
  enrique_commission = 300.00 := by sorry

end NUMINAMATH_CALUDE_enrique_commission_is_300_l3569_356944


namespace NUMINAMATH_CALUDE_folded_rectangle_long_side_l3569_356908

/-- A rectangle with a specific folding property -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  folded_congruent : Bool

/-- The theorem stating the relationship between short and long sides in the folded rectangle -/
theorem folded_rectangle_long_side 
  (rect : FoldedRectangle) 
  (h1 : rect.short_side = 8) 
  (h2 : rect.folded_congruent = true) : 
  rect.long_side = 12 := by
  sorry

#check folded_rectangle_long_side

end NUMINAMATH_CALUDE_folded_rectangle_long_side_l3569_356908


namespace NUMINAMATH_CALUDE_chocolate_eggs_weight_l3569_356923

/-- Calculates the total weight of remaining chocolate eggs after discarding one box -/
theorem chocolate_eggs_weight (total_eggs : ℕ) (weight_per_egg : ℕ) (num_boxes : ℕ) :
  total_eggs = 12 →
  weight_per_egg = 10 →
  num_boxes = 4 →
  (total_eggs - (total_eggs / num_boxes)) * weight_per_egg = 90 := by
  sorry

#check chocolate_eggs_weight

end NUMINAMATH_CALUDE_chocolate_eggs_weight_l3569_356923


namespace NUMINAMATH_CALUDE_owl_wings_area_l3569_356921

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Calculates the area of a triangle given three points using the shoelace formula -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p1.x * p2.y + p2.x * p3.y + p3.x * p1.y) - (p1.y * p2.x + p2.y * p3.x + p3.y * p1.x))

/-- Theorem: The area of the shaded region in the specified rectangle is 4 -/
theorem owl_wings_area (rect : Rectangle) 
    (h1 : rect.topRight.x - rect.bottomLeft.x = 4) 
    (h2 : rect.topRight.y - rect.bottomLeft.y = 5) 
    (h3 : rect.topRight.x - rect.bottomLeft.x = rect.topRight.y - rect.bottomLeft.y - 1) :
    ∃ (p1 p2 p3 : Point), triangleArea p1 p2 p3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_owl_wings_area_l3569_356921


namespace NUMINAMATH_CALUDE_binomial_seven_choose_two_l3569_356966

theorem binomial_seven_choose_two : (7 : ℕ).choose 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_choose_two_l3569_356966


namespace NUMINAMATH_CALUDE_symmetry_of_exponential_graphs_l3569_356996

theorem symmetry_of_exponential_graphs :
  ∀ (a b : ℝ), b = 3^a ↔ -b = -(3^(-a)) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_exponential_graphs_l3569_356996


namespace NUMINAMATH_CALUDE_cheries_whistlers_l3569_356976

/-- Represents the number of boxes of fireworks --/
def koby_boxes : ℕ := 2

/-- Represents the number of boxes of fireworks --/
def cherie_boxes : ℕ := 1

/-- Represents the number of sparklers in each of Koby's boxes --/
def koby_sparklers_per_box : ℕ := 3

/-- Represents the number of whistlers in each of Koby's boxes --/
def koby_whistlers_per_box : ℕ := 5

/-- Represents the number of sparklers in Cherie's box --/
def cherie_sparklers : ℕ := 8

/-- Represents the total number of fireworks Koby and Cherie have --/
def total_fireworks : ℕ := 33

/-- Theorem stating that Cherie's box contains 9 whistlers --/
theorem cheries_whistlers :
  (koby_boxes * koby_sparklers_per_box + koby_boxes * koby_whistlers_per_box +
   cherie_sparklers + (total_fireworks - (koby_boxes * koby_sparklers_per_box +
   koby_boxes * koby_whistlers_per_box + cherie_sparklers))) = total_fireworks ∧
  (total_fireworks - (koby_boxes * koby_sparklers_per_box +
   koby_boxes * koby_whistlers_per_box + cherie_sparklers)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_cheries_whistlers_l3569_356976


namespace NUMINAMATH_CALUDE_twelve_students_pairs_l3569_356909

/-- The number of unique pairs in a group of n elements -/
def uniquePairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of unique pairs in a group of 12 students is 66 -/
theorem twelve_students_pairs : uniquePairs 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_students_pairs_l3569_356909


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l3569_356934

/-- The number of times Terrell lifts the 20-pound weights -/
def original_lifts : ℕ := 12

/-- The weight of each dumbbell in the original set (in pounds) -/
def original_weight : ℕ := 20

/-- The weight of each dumbbell in the new set (in pounds) -/
def new_weight : ℕ := 10

/-- The number of dumbbells Terrell lifts each time -/
def num_dumbbells : ℕ := 2

/-- Calculates the total weight lifted -/
def total_weight (weight : ℕ) (lifts : ℕ) : ℕ :=
  num_dumbbells * weight * lifts

/-- The number of times Terrell needs to lift the new weights to achieve the same total weight -/
def required_lifts : ℕ := total_weight original_weight original_lifts / (num_dumbbells * new_weight)

theorem terrell_weight_lifting :
  required_lifts = 24 ∧
  total_weight new_weight required_lifts = total_weight original_weight original_lifts :=
by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l3569_356934


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l3569_356905

theorem lcm_factor_problem (A B : ℕ) (hcf lcm x : ℕ) : 
  A > 0 → B > 0 → 
  Nat.gcd A B = hcf →
  hcf = 20 →
  A = 280 →
  lcm = Nat.lcm A B →
  lcm = 20 * 13 * x →
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l3569_356905


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3569_356937

theorem arithmetic_calculations : 
  (-6 * (-2) + (-5) * 16 = -68) ∧ 
  ((-1)^4 + (1/4) * (2 * (-6) - (-4)^2) = -8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3569_356937


namespace NUMINAMATH_CALUDE_all_judgments_correct_l3569_356959

theorem all_judgments_correct (a b c : ℕ) (ha : a = 2^22) (hb : b = 3^11) (hc : c = 12^9) :
  (a > b) ∧ (a * b > c) ∧ (b < c) := by
  sorry

end NUMINAMATH_CALUDE_all_judgments_correct_l3569_356959


namespace NUMINAMATH_CALUDE_parabolas_common_tangent_l3569_356965

/-- Given two parabolas C₁ and C₂, prove that if they have exactly one common tangent,
    then a = -1/2 and the equation of the common tangent is y = x - 1/4 -/
theorem parabolas_common_tangent (a : ℝ) :
  let C₁ := λ x : ℝ => x^2 + 2*x
  let C₂ := λ x : ℝ => -x^2 + a
  (∃! l : ℝ → ℝ, ∃ x₁ x₂ : ℝ,
    (∀ x, l x = (2*x₁ + 2)*x - x₁^2) ∧
    (∀ x, l x = -2*x₂*x + x₂^2 + a) ∧
    l (C₁ x₁) = C₁ x₁ ∧
    l (C₂ x₂) = C₂ x₂) →
  a = -1/2 ∧ (λ x : ℝ => x - 1/4) = l
  := by sorry

end NUMINAMATH_CALUDE_parabolas_common_tangent_l3569_356965


namespace NUMINAMATH_CALUDE_sum_of_zeros_infimum_l3569_356958

noncomputable section

def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem sum_of_zeros_infimum (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0) →
  ∃ s : ℝ, s = 4 - 2 * Real.log 2 ∧ ∀ x₁ x₂ : ℝ, F m x₁ = 0 → F m x₂ = 0 → x₁ + x₂ ≥ s :=
sorry

end NUMINAMATH_CALUDE_sum_of_zeros_infimum_l3569_356958


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l3569_356988

-- Define the prices and quantities
def pencil_price : ℚ := 0.5
def folder_price : ℚ := 0.9
def notebook_price : ℚ := 1.2
def stapler_price : ℚ := 2.5

def pencil_quantity : ℕ := 24
def folder_quantity : ℕ := 20
def notebook_quantity : ℕ := 15
def stapler_quantity : ℕ := 10

-- Define discount rates
def pencil_discount_rate : ℚ := 0.1
def folder_discount_rate : ℚ := 0.15

-- Define discount conditions
def pencil_discount_threshold : ℕ := 15
def folder_discount_threshold : ℕ := 10

-- Define notebook offer
def notebook_offer : ℕ := 3  -- buy 2 get 1 free

-- Define the total cost function
def total_cost : ℚ :=
  let pencil_cost := pencil_price * pencil_quantity * (1 - pencil_discount_rate)
  let folder_cost := folder_price * folder_quantity * (1 - folder_discount_rate)
  let notebook_cost := notebook_price * (notebook_quantity - notebook_quantity / notebook_offer)
  let stapler_cost := stapler_price * stapler_quantity
  pencil_cost + folder_cost + notebook_cost + stapler_cost

-- Theorem to prove
theorem total_cost_is_correct : total_cost = 63.1 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l3569_356988


namespace NUMINAMATH_CALUDE_rice_cake_slices_l3569_356991

theorem rice_cake_slices (num_cakes : ℕ) (cake_length : ℝ) (overlap : ℝ) (num_slices : ℕ) :
  num_cakes = 5 →
  cake_length = 2.7 →
  overlap = 0.3 →
  num_slices = 6 →
  (num_cakes * cake_length - (num_cakes - 1) * overlap) / num_slices = 2.05 := by
  sorry

end NUMINAMATH_CALUDE_rice_cake_slices_l3569_356991


namespace NUMINAMATH_CALUDE_bag_weight_problem_l3569_356998

theorem bag_weight_problem (w1 w2 w3 : ℝ) : 
  w1 / w2 = 4 / 5 ∧ w2 / w3 = 5 / 6 ∧ w1 + w3 = w2 + 45 → w1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bag_weight_problem_l3569_356998


namespace NUMINAMATH_CALUDE_cannot_end_with_two_l3569_356997

-- Define the initial set of numbers
def initial_numbers : List Nat := List.range 2017

-- Define the operation of taking the difference
def difference_operation (a b : Nat) : Nat := Int.natAbs (a - b)

-- Define the property of maintaining odd sum parity
def maintains_odd_sum_parity (numbers : List Nat) : Prop :=
  List.sum numbers % 2 = 1

-- Define the final state we want to disprove
def final_state (numbers : List Nat) : Prop :=
  numbers = [2]

-- Theorem statement
theorem cannot_end_with_two :
  ¬ ∃ (final_numbers : List Nat),
    (maintains_odd_sum_parity initial_numbers →
     maintains_odd_sum_parity final_numbers) ∧
    final_state final_numbers :=
by sorry

end NUMINAMATH_CALUDE_cannot_end_with_two_l3569_356997


namespace NUMINAMATH_CALUDE_dance_steps_total_time_l3569_356954

/-- The time spent learning seven dance steps -/
def dance_steps_time : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ :=
  fun step1 step2 step3 step4 step5 step6 step7 =>
    step1 + step2 + step3 + step4 + step5 + step6 + step7

theorem dance_steps_total_time :
  ∀ (step1 : ℝ),
    step1 = 50 →
    let step2 := step1 / 3
    let step3 := step1 + step2
    let step4 := 1.75 * step1
    let step5 := step2 + 25
    let step6 := step3 + step5 - 40
    let step7 := step1 + step2 + step4 + 10
    ∃ (ε : ℝ), ε > 0 ∧ 
      |dance_steps_time step1 step2 step3 step4 step5 step6 step7 - 495.02| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_dance_steps_total_time_l3569_356954


namespace NUMINAMATH_CALUDE_rational_solutions_are_integer_l3569_356994

theorem rational_solutions_are_integer (a b : ℤ) :
  ∃ (x y : ℚ), y - 2*x = a ∧ y^2 - x*y + x^2 = b →
  ∃ (x' y' : ℤ), (x' : ℚ) = x ∧ (y' : ℚ) = y := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_are_integer_l3569_356994


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3569_356989

/-- Surface area of a rectangular solid -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: Total surface area of a rectangular solid with given dimensions -/
theorem rectangular_solid_surface_area (a : ℝ) :
  surface_area a (a + 2) (a - 1) = 6 * a^2 + 4 * a - 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3569_356989


namespace NUMINAMATH_CALUDE_wickets_before_last_match_is_175_l3569_356956

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the number of wickets before the last match is 175 -/
theorem wickets_before_last_match_is_175 (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 8)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 175 := by
  sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_is_175_l3569_356956


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3569_356982

/-- 
Given a quadratic expression 4x^2 - 8x + 6, when factorized in the form a(x - h)^2 + k,
the sum of a, h, and k is equal to 7.
-/
theorem quadratic_factorization_sum (x : ℝ) :
  ∃ (a h k : ℝ), 
    (4 * x^2 - 8 * x + 6 = a * (x - h)^2 + k) ∧ 
    (a + h + k = 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3569_356982


namespace NUMINAMATH_CALUDE_sum_of_zeros_less_than_two_ln_a_l3569_356992

/-- Given a function f(x) = e^x - ax + a, where a ∈ ℝ, if f has two zeros, their sum is less than 2 ln a -/
theorem sum_of_zeros_less_than_two_ln_a (a : ℝ) (x₁ x₂ : ℝ) :
  let f := fun x => Real.exp x - a * x + a
  (f x₁ = 0) → (f x₂ = 0) → (x₁ + x₂ < 2 * Real.log a) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_less_than_two_ln_a_l3569_356992


namespace NUMINAMATH_CALUDE_unique_prime_power_equation_l3569_356938

theorem unique_prime_power_equation :
  ∃! (p n : ℕ), Prime p ∧ n > 0 ∧ (1 + p)^n = 1 + p*n + n^p := by sorry

end NUMINAMATH_CALUDE_unique_prime_power_equation_l3569_356938


namespace NUMINAMATH_CALUDE_combined_cost_price_l3569_356975

def usa_stock : ℝ := 100
def uk_stock : ℝ := 150
def germany_stock : ℝ := 200

def usa_discount : ℝ := 0.06
def uk_discount : ℝ := 0.10
def germany_discount : ℝ := 0.07

def usa_brokerage : ℝ := 0.015
def uk_brokerage : ℝ := 0.02
def germany_brokerage : ℝ := 0.025

def usa_transaction : ℝ := 5
def uk_transaction : ℝ := 3
def germany_transaction : ℝ := 2

def maintenance_charge : ℝ := 0.005
def taxation_rate : ℝ := 0.15

def usd_to_gbp : ℝ := 0.75
def usd_to_eur : ℝ := 0.85

theorem combined_cost_price :
  let usa_cost := (usa_stock * (1 - usa_discount) * (1 + usa_brokerage) + usa_transaction) * (1 + maintenance_charge)
  let uk_cost := (uk_stock * (1 - uk_discount) * (1 + uk_brokerage) + uk_transaction) * (1 + maintenance_charge) / usd_to_gbp
  let germany_cost := (germany_stock * (1 - germany_discount) * (1 + germany_brokerage) + germany_transaction) * (1 + maintenance_charge) / usd_to_eur
  let total_cost := usa_cost + uk_cost + germany_cost
  total_cost * (1 + taxation_rate) = 594.75 := by sorry

end NUMINAMATH_CALUDE_combined_cost_price_l3569_356975


namespace NUMINAMATH_CALUDE_wendy_polished_110_glasses_l3569_356902

/-- The number of small glasses Wendy polished -/
def small_glasses : ℕ := 50

/-- The additional number of large glasses compared to small glasses -/
def additional_large_glasses : ℕ := 10

/-- The total number of glasses Wendy polished -/
def total_glasses : ℕ := small_glasses + (small_glasses + additional_large_glasses)

/-- Proves that Wendy polished 110 glasses in total -/
theorem wendy_polished_110_glasses : total_glasses = 110 := by
  sorry

end NUMINAMATH_CALUDE_wendy_polished_110_glasses_l3569_356902


namespace NUMINAMATH_CALUDE_prob_odd_fair_die_l3569_356999

def die_outcomes : Finset Nat := {1, 2, 3, 4, 5, 6}
def odd_outcomes : Finset Nat := {1, 3, 5}

theorem prob_odd_fair_die :
  (Finset.card odd_outcomes : ℚ) / (Finset.card die_outcomes : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_prob_odd_fair_die_l3569_356999


namespace NUMINAMATH_CALUDE_flour_calculation_l3569_356981

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℕ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_extra_flour : ℕ := 2

/-- The total amount of flour needed by both Katie and Sheila -/
def total_flour : ℕ := katie_flour + (katie_flour + sheila_extra_flour)

theorem flour_calculation :
  total_flour = 8 :=
sorry

end NUMINAMATH_CALUDE_flour_calculation_l3569_356981


namespace NUMINAMATH_CALUDE_arithmetic_mean_sequence_l3569_356984

theorem arithmetic_mean_sequence (a b c d e f g : ℝ) 
  (hb : b = (a + c) / 2)
  (hc : c = (b + d) / 2)
  (hd : d = (c + e) / 2)
  (he : e = (d + f) / 2)
  (hf : f = (e + g) / 2) :
  d = (a + g) / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sequence_l3569_356984


namespace NUMINAMATH_CALUDE_book_price_change_l3569_356957

theorem book_price_change (P : ℝ) (h : P > 0) :
  let price_after_decrease : ℝ := P * 0.5
  let final_price : ℝ := P * 1.2
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 140 :=
by sorry

end NUMINAMATH_CALUDE_book_price_change_l3569_356957


namespace NUMINAMATH_CALUDE_polynomial_equality_l3569_356901

theorem polynomial_equality (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3569_356901


namespace NUMINAMATH_CALUDE_equation_solution_l3569_356924

theorem equation_solution : 
  ∃ x : ℝ, (Real.sqrt (x^2 + 6*x + 10) + Real.sqrt (x^2 - 6*x + 10) = 8) ↔ 
  (x = (4 * Real.sqrt 42) / 7 ∨ x = -(4 * Real.sqrt 42) / 7) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3569_356924


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3569_356936

theorem set_intersection_theorem (x : ℝ) :
  { x : ℝ | x ≥ -2 } ∩ ({ x : ℝ | x > 0 }ᶜ) = { x : ℝ | -2 ≤ x ∧ x ≤ 0 } := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3569_356936


namespace NUMINAMATH_CALUDE_solve_for_r_l3569_356961

theorem solve_for_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_r_l3569_356961


namespace NUMINAMATH_CALUDE_palindrome_divisible_by_11_sum_divisible_by_11_divisible_by_11_condition_balanced_sum_of_palindromes_is_palindrome_l3569_356964

/-- A four-digit number is balanced if the sum of its first two digits equals the sum of its last two digits. -/
def IsBalanced (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) = (n / 10 % 10) + n % 10)

/-- A four-digit number is a palindrome if its first digit equals its last digit and its second digit equals its third digit. -/
def IsPalindrome (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- Any four-digit palindrome is divisible by 11. -/
theorem palindrome_divisible_by_11 (n : ℕ) (h : IsPalindrome n) : 11 ∣ n :=
  sorry

/-- The sum of two numbers divisible by 11 is divisible by 11. -/
theorem sum_divisible_by_11 (a b : ℕ) (ha : 11 ∣ a) (hb : 11 ∣ b) : 11 ∣ (a + b) :=
  sorry

/-- A four-digit number divisible by 11 satisfies the condition that twice its first digit minus its last digit is congruent to 0 modulo 11. -/
theorem divisible_by_11_condition (n : ℕ) (h : 11 ∣ n) (h_four_digit : n ≥ 1000 ∧ n < 10000) :
  (2 * (n / 1000) - n % 10) % 11 = 0 :=
  sorry

/-- Main theorem: A four-digit balanced number that is the sum of two palindrome numbers must itself be a palindrome. -/
theorem balanced_sum_of_palindromes_is_palindrome (n : ℕ) 
  (h_balanced : IsBalanced n) 
  (h_sum_of_palindromes : ∃ a b : ℕ, IsPalindrome a ∧ IsPalindrome b ∧ n = a + b) :
  IsPalindrome n :=
  sorry

end NUMINAMATH_CALUDE_palindrome_divisible_by_11_sum_divisible_by_11_divisible_by_11_condition_balanced_sum_of_palindromes_is_palindrome_l3569_356964


namespace NUMINAMATH_CALUDE_solve_system_l3569_356971

theorem solve_system (c d : ℝ) 
  (eq1 : 5 + c = 7 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3569_356971


namespace NUMINAMATH_CALUDE_max_dominoes_8x9_board_l3569_356983

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the maximum number of non-overlapping dominoes on a checkerboard -/
def max_dominoes (board : Checkerboard) (domino : Domino) (initial_placement : ℕ) : ℕ :=
  sorry

theorem max_dominoes_8x9_board :
  let board : Checkerboard := ⟨8, 9⟩
  let domino : Domino := ⟨2, 1⟩
  let initial_placement : ℕ := 6
  max_dominoes board domino initial_placement = 34 :=
by sorry

end NUMINAMATH_CALUDE_max_dominoes_8x9_board_l3569_356983


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3569_356910

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 4 + a 7 = 45)
  (h_sum2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3569_356910


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l3569_356980

theorem rectangular_field_perimeter : ∀ (length breadth : ℝ),
  breadth = 0.6 * length →
  length * breadth = 37500 →
  2 * (length + breadth) = 800 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l3569_356980


namespace NUMINAMATH_CALUDE_fabric_delivery_problem_l3569_356920

/-- Represents the fabric delivery problem for Daniel's textile company -/
theorem fabric_delivery_problem (monday_delivery : ℝ) : 
  monday_delivery * 2 * 3.5 = 140 → monday_delivery = 20 := by
  sorry

#check fabric_delivery_problem

end NUMINAMATH_CALUDE_fabric_delivery_problem_l3569_356920


namespace NUMINAMATH_CALUDE_simplify_expression_l3569_356968

theorem simplify_expression : (45 * 2^10) / (15 * 2^5) * 5 = 480 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3569_356968


namespace NUMINAMATH_CALUDE_center_is_seven_l3569_356918

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two positions in the grid are adjacent or diagonal -/
def adjacent_or_diagonal (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) ∨ (|i - i'| = 1 ∧ |j - j'| = 1)

/-- The main theorem -/
theorem center_is_seven (g : Grid) : 
  (∀ n : ℕ, n ∈ Finset.range 9 → ∃ i j : Fin 3, g i j = n + 1) →
  (g 0 0 + g 0 2 + g 2 0 + g 2 2 = 20) →
  (∀ n : ℕ, n ∈ Finset.range 8 → 
    ∃ i j i' j' : Fin 3, g i j = n + 1 ∧ g i' j' = n + 2 ∧ adjacent_or_diagonal i j i' j') →
  g 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_center_is_seven_l3569_356918


namespace NUMINAMATH_CALUDE_M_always_positive_l3569_356970

theorem M_always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end NUMINAMATH_CALUDE_M_always_positive_l3569_356970


namespace NUMINAMATH_CALUDE_negation_of_implication_l3569_356947

theorem negation_of_implication (x : ℝ) :
  ¬(x > 1 → x^2 > 1) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3569_356947


namespace NUMINAMATH_CALUDE_A_not_perfect_square_l3569_356948

/-- A number formed by 600 times the digit 6 followed by any number of zeros -/
def A (n : ℕ) : ℕ := 666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666 * (10^n)

/-- The 2-adic valuation of a natural number -/
def two_adic_valuation (m : ℕ) : ℕ :=
  if m = 0 then 0 else (m.factors.filter (· = 2)).length

/-- Theorem: A is not a perfect square for any number of trailing zeros -/
theorem A_not_perfect_square (n : ℕ) : ¬ ∃ (k : ℕ), A n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_A_not_perfect_square_l3569_356948


namespace NUMINAMATH_CALUDE_probability_a2_selected_l3569_356995

-- Define the sets of students
def english_students : Finset (Fin 2) := Finset.univ
def japanese_students : Finset (Fin 3) := Finset.univ

-- Define the total number of possible outcomes
def total_outcomes : ℕ := (english_students.card * japanese_students.card)

-- Define the number of outcomes where A₂ is selected
def a2_outcomes : ℕ := japanese_students.card

-- Theorem statement
theorem probability_a2_selected :
  (a2_outcomes : ℚ) / total_outcomes = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_probability_a2_selected_l3569_356995


namespace NUMINAMATH_CALUDE_point_side_line_range_l3569_356925

/-- Given that points (3,1) and (-4,6) are on the same side of the line 3x-2y+a=0,
    the range of values for a is a < -7 or a > 24 -/
theorem point_side_line_range (a : ℝ) : 
  ((3 * 3 - 2 * 1 + a) * (-4 * 3 - 2 * 6 + a) > 0) ↔ (a < -7 ∨ a > 24) := by
  sorry

end NUMINAMATH_CALUDE_point_side_line_range_l3569_356925


namespace NUMINAMATH_CALUDE_unknown_number_divisor_l3569_356974

theorem unknown_number_divisor : ∃ x : ℕ, 
  x > 0 ∧ 
  100 % x = 16 ∧ 
  200 % x = 4 ∧ 
  ∀ y : ℕ, y > 0 → 100 % y = 16 → 200 % y = 4 → y ≤ x :=
sorry

end NUMINAMATH_CALUDE_unknown_number_divisor_l3569_356974


namespace NUMINAMATH_CALUDE_soccer_league_games_l3569_356903

/-- The number of games played in a league with a given number of teams and games per pair of teams. -/
def games_played (n : ℕ) (g : ℕ) : ℕ := n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team, 
    the total number of games played is 180. -/
theorem soccer_league_games : games_played 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l3569_356903


namespace NUMINAMATH_CALUDE_c_grazing_months_l3569_356933

/-- Represents the number of oxen-months for each person -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Represents the total rent of the pasture -/
def total_rent : ℕ := 175

/-- Represents c's share of the rent -/
def c_share : ℕ := 45

/-- Theorem stating that c put his oxen for grazing for 3 months -/
theorem c_grazing_months :
  ∃ (x : ℕ),
    x = 3 ∧
    c_share * (oxen_months 10 7 + oxen_months 12 5 + oxen_months 15 x) =
    total_rent * oxen_months 15 x :=
by sorry

end NUMINAMATH_CALUDE_c_grazing_months_l3569_356933


namespace NUMINAMATH_CALUDE_mia_correctness_rate_l3569_356955

/-- Represents the correctness rate of homework problems -/
structure HomeworkStats where
  individual_ratio : ℚ  -- Ratio of problems solved individually
  together_ratio : ℚ    -- Ratio of problems solved together
  liam_individual_correct : ℚ  -- Liam's correctness rate for individual problems
  liam_total_correct : ℚ       -- Liam's total correctness rate
  mia_individual_correct : ℚ   -- Mia's correctness rate for individual problems

/-- Calculates Mia's overall percentage of correct answers -/
def mia_overall_correct (stats : HomeworkStats) : ℚ :=
  stats.individual_ratio * stats.mia_individual_correct + 
  stats.together_ratio * ((stats.liam_total_correct - stats.individual_ratio * stats.liam_individual_correct) / stats.together_ratio)

/-- Theorem stating Mia's overall correctness rate given the problem conditions -/
theorem mia_correctness_rate (stats : HomeworkStats) 
  (h1 : stats.individual_ratio = 2/3)
  (h2 : stats.together_ratio = 1/3)
  (h3 : stats.liam_individual_correct = 70/100)
  (h4 : stats.liam_total_correct = 82/100)
  (h5 : stats.mia_individual_correct = 85/100) :
  mia_overall_correct stats = 92/100 := by
  sorry  -- Proof omitted

#eval mia_overall_correct {
  individual_ratio := 2/3,
  together_ratio := 1/3,
  liam_individual_correct := 70/100,
  liam_total_correct := 82/100,
  mia_individual_correct := 85/100
}

end NUMINAMATH_CALUDE_mia_correctness_rate_l3569_356955
