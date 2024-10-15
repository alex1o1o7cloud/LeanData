import Mathlib

namespace NUMINAMATH_CALUDE_fishing_result_l3949_394903

/-- The number of fishes Will and Henry have after fishing -/
def total_fishes (will_catfish : ℕ) (will_eels : ℕ) (henry_trout_per_catfish : ℕ) : ℕ :=
  let will_total := will_catfish + will_eels
  let henry_total := will_catfish * henry_trout_per_catfish
  let henry_keeps := henry_total / 2
  will_total + henry_keeps

/-- Theorem stating the total number of fishes Will and Henry have -/
theorem fishing_result : total_fishes 16 10 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fishing_result_l3949_394903


namespace NUMINAMATH_CALUDE_serenas_age_problem_l3949_394991

/-- Proves that in 6 years, Serena's mother will be three times as old as Serena. -/
theorem serenas_age_problem (serena_age : ℕ) (mother_age : ℕ) 
  (h1 : serena_age = 9) (h2 : mother_age = 39) : 
  ∃ (years : ℕ), years = 6 ∧ mother_age + years = 3 * (serena_age + years) := by
  sorry

end NUMINAMATH_CALUDE_serenas_age_problem_l3949_394991


namespace NUMINAMATH_CALUDE_prove_x_equals_three_l3949_394941

/-- Given real numbers a, b, c, d, and x, if (a - b) = (c + d) + 9, 
    (a + b) = (c - d) - x, and a - c = 3, then x = 3. -/
theorem prove_x_equals_three (a b c d x : ℝ) 
  (h1 : a - b = c + d + 9)
  (h2 : a + b = c - d - x)
  (h3 : a - c = 3) : 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_prove_x_equals_three_l3949_394941


namespace NUMINAMATH_CALUDE_sum_first_100_triangular_numbers_l3949_394973

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular_numbers (n : ℕ) : ℕ := 
  (Finset.range n).sum (λ i => triangular_number (i + 1))

/-- Theorem: The sum of the first 100 triangular numbers is 171700 -/
theorem sum_first_100_triangular_numbers : 
  sum_triangular_numbers 100 = 171700 := by
  sorry

#eval sum_triangular_numbers 100

end NUMINAMATH_CALUDE_sum_first_100_triangular_numbers_l3949_394973


namespace NUMINAMATH_CALUDE_parents_years_in_america_before_aziz_birth_l3949_394922

def current_year : ℕ := 2021
def aziz_age : ℕ := 36
def parents_moved_to_america : ℕ := 1982
def parents_return_home : ℕ := 1995
def parents_return_america : ℕ := 1997

def aziz_birth_year : ℕ := current_year - aziz_age

def years_in_america : ℕ := aziz_birth_year - parents_moved_to_america

theorem parents_years_in_america_before_aziz_birth :
  years_in_america = 3 := by sorry

end NUMINAMATH_CALUDE_parents_years_in_america_before_aziz_birth_l3949_394922


namespace NUMINAMATH_CALUDE_probability_different_suits_pinochle_l3949_394989

/-- A pinochle deck of cards -/
structure PinochleDeck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The probability of drawing three cards of different suits from a pinochle deck -/
def probability_different_suits (deck : PinochleDeck) : Rat :=
  let remaining_after_first := deck.cards - 1
  let suitable_for_second := deck.cards - deck.cards_per_suit
  let remaining_after_second := deck.cards - 2
  let suitable_for_third := deck.cards - 2 * deck.cards_per_suit + 1
  (suitable_for_second : Rat) / remaining_after_first *
  (suitable_for_third : Rat) / remaining_after_second

theorem probability_different_suits_pinochle :
  let deck : PinochleDeck := ⟨48, 4, 12, rfl⟩
  probability_different_suits deck = 414 / 1081 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_suits_pinochle_l3949_394989


namespace NUMINAMATH_CALUDE_simplify_fraction_l3949_394988

theorem simplify_fraction (x y z : ℚ) (hx : x = 5) (hy : y = 2) (hz : z = 4) :
  (10 * x^2 * y^3 * z) / (15 * x * y^2 * z^2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3949_394988


namespace NUMINAMATH_CALUDE_opposite_areas_equal_l3949_394990

/-- Represents a rectangle with an interior point connected to midpoints of its sides --/
structure RectangleWithInteriorPoint where
  /-- The rectangle --/
  rectangle : Set (ℝ × ℝ)
  /-- The interior point --/
  interior_point : ℝ × ℝ
  /-- The midpoints of the rectangle's sides --/
  midpoints : Fin 4 → ℝ × ℝ
  /-- The areas of the four polygons formed --/
  polygon_areas : Fin 4 → ℝ

/-- The sum of opposite polygon areas is equal --/
theorem opposite_areas_equal (r : RectangleWithInteriorPoint) : 
  r.polygon_areas 0 + r.polygon_areas 2 = r.polygon_areas 1 + r.polygon_areas 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_areas_equal_l3949_394990


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l3949_394978

theorem square_and_cube_roots : 
  (∀ x : ℝ, x^2 = 4/9 ↔ x = 2/3 ∨ x = -2/3) ∧ 
  (∀ y : ℝ, y^3 = -64 ↔ y = -4) := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l3949_394978


namespace NUMINAMATH_CALUDE_ball_bounces_to_vertex_l3949_394965

/-- The height of the rectangle --/
def rectangle_height : ℕ := 10

/-- The vertical distance covered in one bounce --/
def vertical_distance_per_bounce : ℕ := 2

/-- The number of bounces required to reach the top of the rectangle --/
def number_of_bounces : ℕ := rectangle_height / vertical_distance_per_bounce

theorem ball_bounces_to_vertex :
  number_of_bounces = 5 :=
sorry

end NUMINAMATH_CALUDE_ball_bounces_to_vertex_l3949_394965


namespace NUMINAMATH_CALUDE_original_price_satisfies_conditions_l3949_394953

/-- The original price of a concert ticket -/
def original_price : ℝ := 20

/-- The number of people who received a 40% discount -/
def discount_40_count : ℕ := 10

/-- The number of people who received a 15% discount -/
def discount_15_count : ℕ := 20

/-- The total number of people who bought tickets -/
def total_buyers : ℕ := 45

/-- The total revenue from ticket sales -/
def total_revenue : ℝ := 760

/-- Theorem stating that the original price satisfies the given conditions -/
theorem original_price_satisfies_conditions : 
  discount_40_count * (original_price * 0.6) + 
  discount_15_count * (original_price * 0.85) + 
  (total_buyers - discount_40_count - discount_15_count) * original_price = 
  total_revenue := by sorry

end NUMINAMATH_CALUDE_original_price_satisfies_conditions_l3949_394953


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3949_394915

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + 2*a₁*x + 4*a₂*x^2 + 8*a₃*x^3 + 16*a₄*x^4 + 32*a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3949_394915


namespace NUMINAMATH_CALUDE_minimal_reciprocal_sum_l3949_394940

def satisfies_equation (a b : ℕ+) : Prop := 30 - a.val = 4 * b.val

def reciprocal_sum (a b : ℕ+) : ℚ := 1 / a.val + 1 / b.val

theorem minimal_reciprocal_sum :
  ∀ a b : ℕ+, satisfies_equation a b →
    reciprocal_sum a b ≥ reciprocal_sum 10 5 :=
by sorry

end NUMINAMATH_CALUDE_minimal_reciprocal_sum_l3949_394940


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l3949_394920

-- Define LCM function
def LCM (a b : ℕ) : ℕ := sorry

-- Define GCF function
def GCF (a b : ℕ) : ℕ := sorry

-- Theorem statement
theorem gcf_of_lcms : GCF (LCM 9 15) (LCM 14 25) = 5 := by sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l3949_394920


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3949_394933

theorem fraction_evaluation : (3 : ℚ) / (1 - 2/5) = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3949_394933


namespace NUMINAMATH_CALUDE_hilt_bread_flour_l3949_394997

/-- The amount of flour needed for baking bread -/
def flour_for_bread (loaves : ℕ) (flour_per_loaf : ℚ) : ℚ :=
  loaves * flour_per_loaf

/-- Theorem: Mrs. Hilt needs 5 cups of flour to bake 2 loaves of bread -/
theorem hilt_bread_flour :
  flour_for_bread 2 (5/2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hilt_bread_flour_l3949_394997


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3949_394942

theorem right_triangle_segment_ratio 
  (a b c r s : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_perp : r + s = c) 
  (h_r : r * c = a^2) 
  (h_s : s * c = b^2) 
  (h_ratio : a / b = 2 / 5) : 
  r / s = 4 / 25 := by 
sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3949_394942


namespace NUMINAMATH_CALUDE_dress_price_is_seven_l3949_394952

def total_revenue : ℝ := 69
def num_dresses : ℕ := 7
def num_shirts : ℕ := 4
def price_shirt : ℝ := 5

theorem dress_price_is_seven :
  ∃ (price_dress : ℝ),
    price_dress * num_dresses + price_shirt * num_shirts = total_revenue ∧
    price_dress = 7 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_is_seven_l3949_394952


namespace NUMINAMATH_CALUDE_middle_box_label_l3949_394994

/-- Represents the possible labels on a box. -/
inductive BoxLabel
  | NoPrize : BoxLabel
  | PrizeInNeighbor : BoxLabel

/-- Represents a row of boxes. -/
structure BoxRow :=
  (size : Nat)
  (labels : Fin size → BoxLabel)
  (prizeLocation : Fin size)

/-- The condition that exactly one statement is true. -/
def exactlyOneTrue (row : BoxRow) : Prop :=
  ∃! i : Fin row.size, 
    (row.labels i = BoxLabel.NoPrize ∧ i ≠ row.prizeLocation) ∨
    (row.labels i = BoxLabel.PrizeInNeighbor ∧ 
      (i.val + 1 = row.prizeLocation.val ∨ i.val = row.prizeLocation.val + 1))

/-- The theorem stating the label on the middle box. -/
theorem middle_box_label (row : BoxRow) 
  (h_size : row.size = 23)
  (h_one_true : exactlyOneTrue row) :
  row.labels ⟨11, by {rw [h_size]; simp}⟩ = BoxLabel.PrizeInNeighbor :=
sorry

end NUMINAMATH_CALUDE_middle_box_label_l3949_394994


namespace NUMINAMATH_CALUDE_minor_premise_incorrect_l3949_394906

theorem minor_premise_incorrect : ¬ ∀ x : ℝ, x + 1/x ≥ 2 * Real.sqrt (x * (1/x)) := by
  sorry

end NUMINAMATH_CALUDE_minor_premise_incorrect_l3949_394906


namespace NUMINAMATH_CALUDE_object_speed_mph_l3949_394917

-- Define the distance traveled in feet
def distance_feet : ℝ := 400

-- Define the time traveled in seconds
def time_seconds : ℝ := 4

-- Define the conversion factor from feet to miles
def feet_per_mile : ℝ := 5280

-- Define the conversion factor from seconds to hours
def seconds_per_hour : ℝ := 3600

-- Theorem statement
theorem object_speed_mph :
  let distance_miles := distance_feet / feet_per_mile
  let time_hours := time_seconds / seconds_per_hour
  let speed_mph := distance_miles / time_hours
  ∃ ε > 0, |speed_mph - 68.18| < ε :=
sorry

end NUMINAMATH_CALUDE_object_speed_mph_l3949_394917


namespace NUMINAMATH_CALUDE_boat_speed_l3949_394923

/-- The average speed of a boat in still water, given travel times with and against a current. -/
theorem boat_speed (time_with_current time_against_current current_speed : ℝ)
  (h1 : time_with_current = 2)
  (h2 : time_against_current = 2.5)
  (h3 : current_speed = 3)
  (h4 : time_with_current * (x + current_speed) = time_against_current * (x - current_speed)) :
  x = 27 :=
by sorry


end NUMINAMATH_CALUDE_boat_speed_l3949_394923


namespace NUMINAMATH_CALUDE_book_chapters_l3949_394946

/-- Represents the number of pages in a book with arithmetic progression of chapter lengths -/
def book_pages (n : ℕ) : ℕ := n * (2 * 13 + (n - 1) * 3) / 2

/-- Theorem stating that a book with 95 pages, where the first chapter has 13 pages
    and each subsequent chapter has 3 more pages than the previous one, has 5 chapters -/
theorem book_chapters :
  ∃ (n : ℕ), n > 0 ∧ book_pages n = 95 ∧ n = 5 := by sorry

end NUMINAMATH_CALUDE_book_chapters_l3949_394946


namespace NUMINAMATH_CALUDE_race_distance_l3949_394909

theorem race_distance (a_time b_time lead_distance : ℕ) 
  (ha : a_time = 28)
  (hb : b_time = 32)
  (hl : lead_distance = 28) : 
  (b_time * lead_distance) / (b_time - a_time) = 224 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l3949_394909


namespace NUMINAMATH_CALUDE_midpoint_chain_l3949_394912

/-- Given a line segment AB with several midpoints, prove that AB = 96 -/
theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 3) →        -- AG = 3
  (B - A = 96) :=      -- AB = 96
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l3949_394912


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l3949_394935

theorem charity_ticket_revenue :
  ∀ (f d : ℕ) (p : ℚ),
    f + d = 160 →
    f * p + d * (2/3 * p) = 2800 →
    ∃ (full_revenue : ℚ),
      full_revenue = f * p ∧
      full_revenue = 1680 :=
by sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l3949_394935


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3949_394996

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3949_394996


namespace NUMINAMATH_CALUDE_min_percentage_both_subjects_l3949_394979

theorem min_percentage_both_subjects (total : ℝ) (physics_percentage : ℝ) (chemistry_percentage : ℝ)
  (h_physics : physics_percentage = 68)
  (h_chemistry : chemistry_percentage = 72)
  (h_total : total > 0) :
  (physics_percentage + chemistry_percentage - 100 : ℝ) = 40 := by
sorry

end NUMINAMATH_CALUDE_min_percentage_both_subjects_l3949_394979


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3949_394927

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31 / 13 - (1 / 13) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3949_394927


namespace NUMINAMATH_CALUDE_two_visits_count_l3949_394981

/-- Represents a friend's visiting pattern -/
structure VisitPattern where
  period : Nat
  offset : Nat

/-- Calculates the number of days where exactly two friends visit -/
def exactTwoVisits (alice beatrix claire : VisitPattern) (totalDays : Nat) : Nat :=
  sorry

theorem two_visits_count :
  let alice : VisitPattern := { period := 2, offset := 0 }
  let beatrix : VisitPattern := { period := 6, offset := 1 }
  let claire : VisitPattern := { period := 5, offset := 1 }
  let totalDays : Nat := 400
  exactTwoVisits alice beatrix claire totalDays = 80 := by sorry

end NUMINAMATH_CALUDE_two_visits_count_l3949_394981


namespace NUMINAMATH_CALUDE_tens_digit_of_23_pow_1987_l3949_394948

theorem tens_digit_of_23_pow_1987 : ∃ k : ℕ, 23^1987 ≡ 40 + k [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_pow_1987_l3949_394948


namespace NUMINAMATH_CALUDE_solution_set_f_min_m2_n2_l3949_394960

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f : 
  {x : ℝ | f x > 7} = {x : ℝ | x > 4 ∨ x < -3} := by sorry

-- Theorem for the minimum value of m^2 + n^2
theorem min_m2_n2 (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_min : ∀ x, f x ≥ m + n) (h_eq : m + n = 3) :
  m^2 + n^2 ≥ 9/2 ∧ (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_min_m2_n2_l3949_394960


namespace NUMINAMATH_CALUDE_intersection_complement_and_B_l3949_394999

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

theorem intersection_complement_and_B : 
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_and_B_l3949_394999


namespace NUMINAMATH_CALUDE_probability_calculation_l3949_394982

/-- The number of people with blocks -/
def num_people : ℕ := 3

/-- The number of blocks each person has -/
def blocks_per_person : ℕ := 6

/-- The number of empty boxes -/
def num_boxes : ℕ := 5

/-- The maximum number of blocks a person can place in a box -/
def max_blocks_per_person_per_box : ℕ := 2

/-- The maximum total number of blocks allowed in a box -/
def max_blocks_per_box : ℕ := 4

/-- The number of ways each person can distribute their blocks -/
def distribution_ways : ℕ := (num_boxes + blocks_per_person - 1).choose (blocks_per_person - 1)

/-- The number of favorable distributions for a specific box getting all blocks of the same color -/
def favorable_distributions : ℕ := blocks_per_person

/-- The probability that at least one box has all blocks of the same color -/
def probability_same_color : ℝ := 1.86891e-6

theorem probability_calculation :
  1 - num_boxes * (favorable_distributions : ℝ) / (distribution_ways ^ num_people : ℝ) = probability_same_color := by
  sorry

end NUMINAMATH_CALUDE_probability_calculation_l3949_394982


namespace NUMINAMATH_CALUDE_function_inequality_l3949_394962

theorem function_inequality (m n : ℝ) (hm : m < 0) :
  (∃ x : ℝ, x > 0 ∧ Real.log x + m * x + n ≥ 0) →
  n - 1 ≥ Real.log (-m) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3949_394962


namespace NUMINAMATH_CALUDE_circle_center_correct_l3949_394974

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x + 9 * y^2 + 36 * y + 44 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (1, -2)

/-- Theorem stating that CircleCenter is the center of the circle defined by CircleEquation -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 1/9 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3949_394974


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3949_394975

/-- The line x + (l-m)y + 3 = 0 always passes through the point (-3, 0) for any real number m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (-3 : ℝ) + (1 - m) * (0 : ℝ) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3949_394975


namespace NUMINAMATH_CALUDE_oxygen_weight_value_l3949_394959

/-- The atomic weight of sodium -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of chlorine -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound -/
def compound_weight : ℝ := 74

/-- The atomic weight of oxygen -/
def oxygen_weight : ℝ := compound_weight - (sodium_weight + chlorine_weight)

theorem oxygen_weight_value : oxygen_weight = 15.56 := by
  sorry

end NUMINAMATH_CALUDE_oxygen_weight_value_l3949_394959


namespace NUMINAMATH_CALUDE_gcd_problem_l3949_394900

theorem gcd_problem (b : ℤ) (h : 1632 ∣ b) :
  Int.gcd (b^2 + 11*b + 30) (b + 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3949_394900


namespace NUMINAMATH_CALUDE_no_x_squared_term_l3949_394914

/-- 
Given the algebraic expression (x-2)(ax²-x+1), this theorem states that
the coefficient of x² in the expanded form is zero if and only if a = -1/2.
-/
theorem no_x_squared_term (x a : ℝ) : 
  (x - 2) * (a * x^2 - x + 1) = a * x^3 + 3 * x - 2 ↔ a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l3949_394914


namespace NUMINAMATH_CALUDE_correct_operation_result_l3949_394986

theorem correct_operation_result (x : ℝ) : (x - 9) / 3 = 43 → (x - 3) / 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_result_l3949_394986


namespace NUMINAMATH_CALUDE_b_share_is_108_l3949_394904

/-- Represents the share ratio of partners A, B, and C -/
structure ShareRatio where
  a : Rat
  b : Rat
  c : Rat

/-- Represents the capital contribution of partners over time -/
structure CapitalContribution where
  a : Rat
  b : Rat
  c : Rat

def initial_ratio : ShareRatio :=
  { a := 1/2, b := 1/3, c := 1/4 }

def total_profit : ℚ := 378

def months_before_withdrawal : ℕ := 2
def total_months : ℕ := 12

def capital_contribution (r : ShareRatio) : CapitalContribution :=
  { a := r.a * months_before_withdrawal + (r.a / 2) * (total_months - months_before_withdrawal),
    b := r.b * total_months,
    c := r.c * total_months }

theorem b_share_is_108 (r : ShareRatio) (cc : CapitalContribution) :
  r = initial_ratio →
  cc = capital_contribution r →
  (cc.b / (cc.a + cc.b + cc.c)) * total_profit = 108 :=
by sorry

end NUMINAMATH_CALUDE_b_share_is_108_l3949_394904


namespace NUMINAMATH_CALUDE_sum_of_complex_equation_l3949_394938

/-- Given real numbers x and y satisfying (2+i)x = 4+yi, prove that x + y = 4 -/
theorem sum_of_complex_equation (x y : ℝ) : 
  (Complex.I : ℂ) * x + 2 * x = 4 + (Complex.I : ℂ) * y → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_equation_l3949_394938


namespace NUMINAMATH_CALUDE_initial_milk_cost_initial_milk_cost_is_four_l3949_394908

/-- Calculates the initial cost of milk given the grocery shopping scenario --/
theorem initial_milk_cost (total_money : ℝ) (bread_cost : ℝ) (detergent_cost : ℝ) 
  (banana_cost_per_pound : ℝ) (banana_pounds : ℝ) (detergent_coupon : ℝ) 
  (milk_discount_factor : ℝ) (money_left : ℝ) : ℝ :=
  let total_spent := total_money - money_left
  let banana_cost := banana_cost_per_pound * banana_pounds
  let discounted_detergent_cost := detergent_cost - detergent_coupon
  let non_milk_cost := bread_cost + banana_cost + discounted_detergent_cost
  let milk_cost := total_spent - non_milk_cost
  milk_cost / milk_discount_factor

/-- The initial cost of milk is $4 --/
theorem initial_milk_cost_is_four :
  initial_milk_cost 20 3.5 10.25 0.75 2 1.25 0.5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_cost_initial_milk_cost_is_four_l3949_394908


namespace NUMINAMATH_CALUDE_billy_weight_l3949_394971

theorem billy_weight (carl_weight brad_weight billy_weight : ℕ) 
  (h1 : brad_weight = carl_weight + 5)
  (h2 : billy_weight = brad_weight + 9)
  (h3 : carl_weight = 145) :
  billy_weight = 159 := by
  sorry

end NUMINAMATH_CALUDE_billy_weight_l3949_394971


namespace NUMINAMATH_CALUDE_cubic_curve_triangle_problem_l3949_394969

/-- A point on the curve y = x^3 -/
structure CubicPoint where
  x : ℝ
  y : ℝ
  cubic_cond : y = x^3

/-- The problem statement -/
theorem cubic_curve_triangle_problem :
  ∃ (A B C : CubicPoint),
    -- A, B, C are distinct
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    -- BC is parallel to x-axis
    B.y = C.y ∧
    -- Area condition
    |C.x - B.x| * |A.y - B.y| = 2000 ∧
    -- Sum of digits of A's x-coordinate is 1
    (∃ (n : ℕ), A.x = 10 * n + 1 ∧ 0 ≤ n ∧ n < 10) := by
  sorry

end NUMINAMATH_CALUDE_cubic_curve_triangle_problem_l3949_394969


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l3949_394911

theorem sum_of_x_solutions (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 289) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 289 ∧ x2^2 + y^2 = 289 ∧ x1 + x2 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l3949_394911


namespace NUMINAMATH_CALUDE_min_product_given_sum_l3949_394924

theorem min_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 8 → a * b ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_product_given_sum_l3949_394924


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3949_394939

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 3)
  are_parallel a b → x = 6 := by
    sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3949_394939


namespace NUMINAMATH_CALUDE_x_axis_coefficients_l3949_394980

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a line is the x-axis -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- Theorem stating that if a line is the x-axis, then B ≠ 0 and A = C = 0 -/
theorem x_axis_coefficients (l : Line) :
  is_x_axis l → l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 :=
sorry

end NUMINAMATH_CALUDE_x_axis_coefficients_l3949_394980


namespace NUMINAMATH_CALUDE_probability_two_sunny_days_out_of_five_l3949_394961

theorem probability_two_sunny_days_out_of_five :
  let n : ℕ := 5  -- total number of days
  let k : ℕ := 2  -- number of sunny days we want
  let p : ℚ := 1/4  -- probability of a sunny day (1 - probability of rain)
  let q : ℚ := 3/4  -- probability of a rainy day
  (n.choose k : ℚ) * p^k * q^(n - k) = 135/512 :=
sorry

end NUMINAMATH_CALUDE_probability_two_sunny_days_out_of_five_l3949_394961


namespace NUMINAMATH_CALUDE_subtraction_two_minus_three_l3949_394936

theorem subtraction_two_minus_three : 2 - 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_two_minus_three_l3949_394936


namespace NUMINAMATH_CALUDE_sum_proper_divisors_729_l3949_394947

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x ≠ 0 ∧ n % x = 0)

theorem sum_proper_divisors_729 :
  (proper_divisors 729).sum id = 364 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_729_l3949_394947


namespace NUMINAMATH_CALUDE_cost_for_100km_l3949_394995

/-- Represents the cost of a taxi ride in dollars -/
def taxi_cost (distance : ℝ) : ℝ := sorry

/-- The taxi fare is directly proportional to distance traveled -/
axiom fare_proportional (d1 d2 : ℝ) : d1 ≠ 0 → d2 ≠ 0 → 
  taxi_cost d1 / d1 = taxi_cost d2 / d2

/-- Bob's actual ride: 80 km for $160 -/
axiom bob_ride : taxi_cost 80 = 160

/-- The theorem to prove -/
theorem cost_for_100km : taxi_cost 100 = 200 := by sorry

end NUMINAMATH_CALUDE_cost_for_100km_l3949_394995


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3949_394928

/-- A geometric sequence of positive integers with first term 2 and fourth term 162 has third term 18 -/
theorem geometric_sequence_third_term : 
  ∀ (a : ℕ → ℕ) (r : ℕ),
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 2 →                     -- first term is 2
  a 4 = 162 →                   -- fourth term is 162
  a 3 = 18 :=                   -- third term is 18
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3949_394928


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l3949_394977

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n = 25 →
  initial_mean = 190 →
  incorrect_value = 130 →
  correct_value = 165 →
  (n * initial_mean - incorrect_value + correct_value) / n = 191.4 :=
by sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l3949_394977


namespace NUMINAMATH_CALUDE_incorrect_conclusions_l3949_394954

structure Conclusion where
  correlation : Bool  -- true for positive, false for negative
  coefficient : Real
  constant : Real

def is_correct (c : Conclusion) : Prop :=
  (c.correlation ↔ c.coefficient > 0)

theorem incorrect_conclusions 
  (c1 : Conclusion)
  (c2 : Conclusion)
  (c3 : Conclusion)
  (c4 : Conclusion)
  (h1 : c1 = { correlation := false, coefficient := 2.347, constant := -6.423 })
  (h2 : c2 = { correlation := false, coefficient := -3.476, constant := 5.648 })
  (h3 : c3 = { correlation := true, coefficient := 5.437, constant := 8.493 })
  (h4 : c4 = { correlation := true, coefficient := -4.326, constant := -4.578 }) :
  ¬(is_correct c1) ∧ ¬(is_correct c4) :=
sorry

end NUMINAMATH_CALUDE_incorrect_conclusions_l3949_394954


namespace NUMINAMATH_CALUDE_abs_neg_six_equals_six_l3949_394972

theorem abs_neg_six_equals_six : |(-6 : ℤ)| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_six_equals_six_l3949_394972


namespace NUMINAMATH_CALUDE_book_price_relationship_l3949_394958

/-- Represents a collection of books with linearly increasing prices -/
structure BookCollection where
  basePrice : ℕ
  count : ℕ

/-- Get the price of a book at a specific position -/
def BookCollection.priceAt (bc : BookCollection) (position : ℕ) : ℕ :=
  bc.basePrice + position - 1

/-- The main theorem about the book price relationship -/
theorem book_price_relationship (bc : BookCollection) 
  (h1 : bc.count = 49) : 
  (bc.priceAt 49)^2 = (bc.priceAt 25)^2 + (bc.priceAt 26)^2 := by
  sorry

/-- Helper lemma: The price difference between adjacent books is 1 -/
lemma price_difference (bc : BookCollection) (i : ℕ) 
  (h : i < bc.count) :
  bc.priceAt (i + 1) = bc.priceAt i + 1 := by
  sorry

end NUMINAMATH_CALUDE_book_price_relationship_l3949_394958


namespace NUMINAMATH_CALUDE_composition_result_l3949_394967

/-- Given two functions f and g, prove that f(g(-2)) = 81 -/
theorem composition_result :
  (f : ℝ → ℝ) →
  (g : ℝ → ℝ) →
  (∀ x, f x = x^2) →
  (∀ x, g x = 2*x - 5) →
  f (g (-2)) = 81 := by
sorry

end NUMINAMATH_CALUDE_composition_result_l3949_394967


namespace NUMINAMATH_CALUDE_range_of_m_satisfying_condition_l3949_394916

theorem range_of_m_satisfying_condition :
  {m : ℝ | ∀ x : ℝ, m * x^2 - (3 - m) * x + 1 > 0 ∨ m * x > 0} = {m : ℝ | 1/9 < m ∧ m < 1} := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_satisfying_condition_l3949_394916


namespace NUMINAMATH_CALUDE_investment_rate_proof_l3949_394998

def total_investment : ℝ := 12000
def first_investment : ℝ := 5000
def second_investment : ℝ := 4000
def first_rate : ℝ := 0.03
def second_rate : ℝ := 0.045
def desired_income : ℝ := 600

theorem investment_rate_proof :
  let remaining_investment := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let remaining_income := desired_income - income_from_first - income_from_second
  remaining_income / remaining_investment = 0.09 := by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l3949_394998


namespace NUMINAMATH_CALUDE_f_is_even_g_is_not_odd_even_function_symmetry_odd_function_symmetry_l3949_394985

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the functions
def f (x : ℝ) : ℝ := x^4 + x^2
def g (x : ℝ) : ℝ := x^3 + x^2

-- Theorem statements
theorem f_is_even : IsEven f := by sorry

theorem g_is_not_odd : ¬ IsOdd g := by sorry

theorem even_function_symmetry (f : ℝ → ℝ) (h : IsEven f) :
  ∀ x y, f x = y ↔ f (-x) = y := by sorry

theorem odd_function_symmetry (f : ℝ → ℝ) (h : IsOdd f) :
  ∀ x y, f x = y ↔ f (-x) = -y := by sorry

end NUMINAMATH_CALUDE_f_is_even_g_is_not_odd_even_function_symmetry_odd_function_symmetry_l3949_394985


namespace NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l3949_394950

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 23)
  (lcm_eq : Nat.lcm a b = 23 * 13 * 14) :
  max a b = 322 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l3949_394950


namespace NUMINAMATH_CALUDE_composite_sequence_existence_l3949_394993

theorem composite_sequence_existence (m : ℕ) (hm : m > 0) :
  ∃ n : ℕ, ∀ k : ℤ, |k| ≤ m → 
    (2^n : ℤ) + k > 0 ∧ ¬(Nat.Prime ((2^n : ℤ) + k).natAbs) :=
by sorry

end NUMINAMATH_CALUDE_composite_sequence_existence_l3949_394993


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3949_394976

-- Define the inverse variation relationship
def inverse_variation (a b k : ℝ) : Prop := a * b^3 = k

-- State the theorem
theorem inverse_variation_problem :
  ∀ (a₁ a₂ k : ℝ),
  inverse_variation a₁ 2 k →
  a₁ = 16 →
  inverse_variation a₂ 4 k →
  a₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3949_394976


namespace NUMINAMATH_CALUDE_incandescent_bulbs_on_l3949_394921

/-- Prove that the number of switched-on incandescent bulbs is 420 -/
theorem incandescent_bulbs_on (total_bulbs : ℕ) 
  (incandescent_percent fluorescent_percent led_percent halogen_percent : ℚ)
  (total_on_percent : ℚ)
  (incandescent_on_percent fluorescent_on_percent led_on_percent halogen_on_percent : ℚ) :
  total_bulbs = 3000 →
  incandescent_percent = 40 / 100 →
  fluorescent_percent = 30 / 100 →
  led_percent = 20 / 100 →
  halogen_percent = 10 / 100 →
  total_on_percent = 55 / 100 →
  incandescent_on_percent = 35 / 100 →
  fluorescent_on_percent = 50 / 100 →
  led_on_percent = 80 / 100 →
  halogen_on_percent = 30 / 100 →
  (incandescent_percent * total_bulbs : ℚ) * incandescent_on_percent = 420 :=
by
  sorry


end NUMINAMATH_CALUDE_incandescent_bulbs_on_l3949_394921


namespace NUMINAMATH_CALUDE_glasses_fraction_after_tripling_l3949_394966

theorem glasses_fraction_after_tripling (n : ℝ) (h : n > 0) :
  let initial_with_glasses := (2 / 3 : ℝ) * n
  let initial_without_glasses := (1 / 3 : ℝ) * n
  let new_without_glasses := 3 * initial_without_glasses
  let new_total := initial_with_glasses + new_without_glasses
  initial_with_glasses / new_total = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_glasses_fraction_after_tripling_l3949_394966


namespace NUMINAMATH_CALUDE_camping_payment_difference_l3949_394932

/-- Represents the camping trip expenses and calculations --/
structure CampingExpenses where
  alan_paid : ℝ
  beth_paid : ℝ
  chris_paid : ℝ
  picnic_cost : ℝ
  total_cost : ℝ
  alan_share : ℝ
  beth_share : ℝ
  chris_share : ℝ

/-- Calculates the difference between what Alan and Beth need to pay Chris --/
def payment_difference (expenses : CampingExpenses) : ℝ :=
  (expenses.alan_share - expenses.alan_paid) - (expenses.beth_share - expenses.beth_paid)

/-- Theorem stating that the payment difference is 30 --/
theorem camping_payment_difference :
  ∃ (expenses : CampingExpenses),
    expenses.alan_paid = 110 ∧
    expenses.beth_paid = 140 ∧
    expenses.chris_paid = 190 ∧
    expenses.picnic_cost = 60 ∧
    expenses.total_cost = expenses.alan_paid + expenses.beth_paid + expenses.chris_paid + expenses.picnic_cost ∧
    expenses.alan_share = expenses.total_cost / 3 ∧
    expenses.beth_share = expenses.total_cost / 3 ∧
    expenses.chris_share = expenses.total_cost / 3 ∧
    payment_difference expenses = 30 := by
  sorry

end NUMINAMATH_CALUDE_camping_payment_difference_l3949_394932


namespace NUMINAMATH_CALUDE_pizza_fraction_l3949_394957

theorem pizza_fraction (pieces_per_day : ℕ) (days : ℕ) (whole_pizzas : ℕ) :
  pieces_per_day = 3 →
  days = 72 →
  whole_pizzas = 27 →
  (1 : ℚ) / (pieces_per_day * days / whole_pizzas) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l3949_394957


namespace NUMINAMATH_CALUDE_unbounded_fraction_over_primes_l3949_394902

-- Define the ord_p function
def ord_p (a p : ℕ) : ℕ := sorry

-- State the theorem
theorem unbounded_fraction_over_primes (a : ℕ) (h : a > 1) :
  ∀ M : ℕ, ∃ p : ℕ, Prime p ∧ (p - 1) / ord_p a p > M :=
sorry

end NUMINAMATH_CALUDE_unbounded_fraction_over_primes_l3949_394902


namespace NUMINAMATH_CALUDE_bike_wheel_rotations_l3949_394970

theorem bike_wheel_rotations 
  (rotations_per_block : ℕ) 
  (min_blocks : ℕ) 
  (remaining_rotations : ℕ) 
  (h1 : rotations_per_block = 200)
  (h2 : min_blocks = 8)
  (h3 : remaining_rotations = 1000) :
  min_blocks * rotations_per_block - remaining_rotations = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_bike_wheel_rotations_l3949_394970


namespace NUMINAMATH_CALUDE_triangle_side_length_l3949_394963

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  A = 2 * Real.pi / 3 →
  b = Real.sqrt 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3949_394963


namespace NUMINAMATH_CALUDE_bisection_method_representation_l3949_394901

/-- Represents different types of diagrams --/
inductive DiagramType
  | OrganizationalStructure
  | ProcessFlowchart
  | KnowledgeStructure
  | ProgramFlowchart

/-- Represents the bisection method algorithm --/
structure BisectionMethod where
  hasLoopStructure : Bool
  hasConditionalStructure : Bool

/-- Theorem stating that the bisection method for solving x^2 - 2 = 0 is best represented by a program flowchart --/
theorem bisection_method_representation (bm : BisectionMethod) 
  (h1 : bm.hasLoopStructure = true) 
  (h2 : bm.hasConditionalStructure = true) : 
  DiagramType.ProgramFlowchart = 
    (fun (d : DiagramType) => 
      if bm.hasLoopStructure ∧ bm.hasConditionalStructure 
      then DiagramType.ProgramFlowchart 
      else d) DiagramType.ProgramFlowchart :=
by
  sorry

#check bisection_method_representation

end NUMINAMATH_CALUDE_bisection_method_representation_l3949_394901


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_equality_l3949_394930

theorem rectangle_perimeter_area_equality (k : ℝ) (h : k > 0) :
  (∃ w : ℝ, w > 0 ∧ 
    8 * w = k ∧  -- Perimeter equals k
    3 * w^2 = k) -- Area equals k
  → k = 64 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_area_equality_l3949_394930


namespace NUMINAMATH_CALUDE_complex_unit_circle_ab_range_l3949_394929

theorem complex_unit_circle_ab_range (a b : ℝ) : 
  (Complex.abs (Complex.mk a b) = 1) → 
  (a * b ≥ -1/2 ∧ a * b ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_complex_unit_circle_ab_range_l3949_394929


namespace NUMINAMATH_CALUDE_positive_sequence_existence_l3949_394945

theorem positive_sequence_existence :
  ∃ (a : ℕ → ℝ) (a₁ : ℝ), 
    (∀ n, a n > 0) ∧
    (∀ n, a (n + 2) = a n - a (n + 1)) ∧
    (a₁ > 0) ∧
    (∀ n, a n = a₁ * ((Real.sqrt 5 - 1) / 2) ^ (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_sequence_existence_l3949_394945


namespace NUMINAMATH_CALUDE_train_crossing_time_l3949_394907

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length_m : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length_m = 1250 →
  train_speed_kmh = 300 →
  crossing_time_s = 15 →
  crossing_time_s = (train_length_m / 1000) / (train_speed_kmh / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3949_394907


namespace NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3949_394926

theorem greatest_x_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 4 ∧ 
  (∀ (x : ℝ), -2 * x^2 + 12 * x - 16 ≥ 0 → x ≤ x_max) ∧
  (-2 * x_max^2 + 12 * x_max - 16 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3949_394926


namespace NUMINAMATH_CALUDE_impossibility_of_zero_sum_l3949_394992

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a configuration of signs between numbers 1 to 10 -/
def SignConfiguration := Fin 9 → Bool

/-- Calculates the sum based on a given sign configuration -/
def calculate_sum (config : SignConfiguration) : ℤ :=
  sorry

theorem impossibility_of_zero_sum : ∀ (config : SignConfiguration), 
  calculate_sum config ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_zero_sum_l3949_394992


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_existence_l3949_394919

theorem arithmetic_and_geometric_sequences_existence :
  ∃ (a b c : ℝ) (d r : ℝ),
    d ≠ 0 ∧ r ≠ 0 ∧ r ≠ 1 ∧
    (b - a = d ∧ c - b = d) ∧
    (∃ (x y : ℝ), x * r = y ∧ y * r = a ∧ a * r = b ∧ b * r = c) ∧
    ((a * r = b ∧ b * r = c) ∨ (b * r = a ∧ a * r = c) ∨ (c * r = a ∧ a * r = b)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_existence_l3949_394919


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3949_394937

/-- The x-coordinate of the point on the x-axis equidistant from C(-3, 0) and D(2, 5) is 2 -/
theorem equidistant_point_x_coordinate :
  let C : ℝ × ℝ := (-3, 0)
  let D : ℝ × ℝ := (2, 5)
  ∃ x : ℝ, x = 2 ∧
    (x - C.1)^2 + C.2^2 = (x - D.1)^2 + D.2^2 :=
by sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l3949_394937


namespace NUMINAMATH_CALUDE_inequality_solution_minimum_value_and_points_l3949_394955

-- Part 1
def solution_set := {x : ℝ | 0 < x ∧ x < 2}

theorem inequality_solution : 
  ∀ x : ℝ, |2*x - 1| < |x| + 1 ↔ x ∈ solution_set :=
sorry

-- Part 2
def constraint (x y z : ℝ) := x^2 + y^2 + z^2 = 4

theorem minimum_value_and_points :
  ∃ (x y z : ℝ), constraint x y z ∧
    (∀ (a b c : ℝ), constraint a b c → x - 2*y + 2*z ≤ a - 2*b + 2*c) ∧
    x - 2*y + 2*z = -6 ∧
    x = -2/3 ∧ y = 4/3 ∧ z = -4/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_minimum_value_and_points_l3949_394955


namespace NUMINAMATH_CALUDE_leap_stride_difference_l3949_394968

/-- Represents the number of strides Elmer takes between consecutive poles -/
def elmer_strides : ℕ := 44

/-- Represents the number of leaps Oscar takes between consecutive poles -/
def oscar_leaps : ℕ := 12

/-- Represents the total number of poles -/
def total_poles : ℕ := 41

/-- Represents the total distance in feet between the first and last pole -/
def total_distance : ℕ := 5280

/-- Calculates the length of Elmer's stride in feet -/
def elmer_stride_length : ℚ :=
  total_distance / (elmer_strides * (total_poles - 1))

/-- Calculates the length of Oscar's leap in feet -/
def oscar_leap_length : ℚ :=
  total_distance / (oscar_leaps * (total_poles - 1))

/-- Theorem stating the difference between Oscar's leap and Elmer's stride -/
theorem leap_stride_difference : oscar_leap_length - elmer_stride_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_leap_stride_difference_l3949_394968


namespace NUMINAMATH_CALUDE_rectangle_sides_l3949_394905

theorem rectangle_sides (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) : 
  b = 3 * h → 
  h * b = 2 * (h + b) + Real.sqrt (h^2 + b^2) → 
  h = (8 + Real.sqrt 10) / 3 ∧ b = 8 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sides_l3949_394905


namespace NUMINAMATH_CALUDE_work_completion_time_l3949_394949

/-- A can do a piece of work in some days. A does the work for 5 days only and leaves the job. 
    B does the remaining work in 3 days. B alone can do the work in 4.5 days. 
    This theorem proves that A alone can do the work in 15 days. -/
theorem work_completion_time (W : ℝ) (A_work_per_day B_work_per_day : ℝ) : 
  (B_work_per_day = W / 4.5) →
  (5 * A_work_per_day + 3 * B_work_per_day = W) →
  (A_work_per_day = W / 15) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3949_394949


namespace NUMINAMATH_CALUDE_problem_1_l3949_394934

theorem problem_1 : (1 : ℝ) - 1^4 - 1/2 * (3 - (-3)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3949_394934


namespace NUMINAMATH_CALUDE_no_valid_base_6_digit_for_divisibility_by_7_l3949_394944

theorem no_valid_base_6_digit_for_divisibility_by_7 :
  ∀ d : ℕ, d ≤ 5 → ¬(∃ k : ℤ, 652 + 42 * d = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_base_6_digit_for_divisibility_by_7_l3949_394944


namespace NUMINAMATH_CALUDE_function_composition_equality_l3949_394987

/-- Given a function f and a condition on f[g(x)], prove the form of g(x) -/
theorem function_composition_equality 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f : ∀ x, f x = 3 * x - 1) 
  (h_fg : ∀ x, f (g x) = 2 * x + 3) : 
  ∀ x, g x = (2/3) * x + (4/3) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3949_394987


namespace NUMINAMATH_CALUDE_reginalds_apple_sales_l3949_394951

/-- Represents the problem of calculating the number of apples sold by Reginald --/
theorem reginalds_apple_sales :
  let apple_price : ℚ := 5 / 4  -- $1.25 per apple
  let bike_cost : ℚ := 80
  let repair_cost_ratio : ℚ := 1 / 4  -- 25% of bike cost
  let remaining_ratio : ℚ := 1 / 5  -- 1/5 of earnings remain after repairs
  let apples_per_set : ℕ := 6  -- 5 paid + 1 free
  let paid_apples_per_set : ℕ := 5

  ∃ (total_apples : ℕ),
    total_apples = 120 ∧
    total_apples % apples_per_set = 0 ∧
    let total_sets := total_apples / apples_per_set
    let total_earnings := (total_sets * paid_apples_per_set : ℚ) * apple_price
    let repair_cost := bike_cost * repair_cost_ratio
    total_earnings * remaining_ratio = total_earnings - repair_cost :=
by
  sorry

end NUMINAMATH_CALUDE_reginalds_apple_sales_l3949_394951


namespace NUMINAMATH_CALUDE_more_stable_scores_lower_variance_problem_solution_l3949_394913

/-- Represents an athlete with their test score variance -/
structure Athlete where
  name : String
  variance : ℝ

/-- Determines if an athlete has more stable test scores than another -/
def has_more_stable_scores (a b : Athlete) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with equal average scores, 
    the athlete with lower variance has more stable test scores -/
theorem more_stable_scores_lower_variance 
  (a b : Athlete) 
  (h_avg : ℝ) -- average score of both athletes
  (h_equal_avg : True) -- assumption that both athletes have equal average scores
  : has_more_stable_scores a b ↔ a.variance < b.variance :=
by sorry

/-- Application to the specific problem -/
def athlete_A : Athlete := ⟨"A", 0.024⟩
def athlete_B : Athlete := ⟨"B", 0.008⟩

theorem problem_solution : has_more_stable_scores athlete_B athlete_A :=
by sorry

end NUMINAMATH_CALUDE_more_stable_scores_lower_variance_problem_solution_l3949_394913


namespace NUMINAMATH_CALUDE_focus_of_specific_parabola_l3949_394964

/-- A parabola with equation y = (x - h)^2 + k, where (h, k) is the vertex. -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The focus of a parabola. -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem: The focus of the parabola y = (x - 3)^2 is at (3, 1/8). -/
theorem focus_of_specific_parabola :
  let p : Parabola := { h := 3, k := 0 }
  focus p = (3, 1/8) := by sorry

end NUMINAMATH_CALUDE_focus_of_specific_parabola_l3949_394964


namespace NUMINAMATH_CALUDE_log_base_values_l3949_394918

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_base_values (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 2 4, f a x ∈ Set.Icc (f a 2) (f a 4)) ∧
  (f a 4 - f a 2 = 2 ∨ f a 2 - f a 4 = 2) →
  a = Real.sqrt 2 ∨ a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_log_base_values_l3949_394918


namespace NUMINAMATH_CALUDE_denny_initial_followers_l3949_394943

/-- Calculates the initial number of followers given the daily increase, total unfollows, 
    final follower count, and number of days in a year. -/
def initial_followers (daily_increase : ℕ) (total_unfollows : ℕ) (final_count : ℕ) (days_in_year : ℕ) : ℕ :=
  final_count - (daily_increase * days_in_year) + total_unfollows

/-- Proves that given the specified conditions, the initial number of followers is 100000. -/
theorem denny_initial_followers : 
  initial_followers 1000 20000 445000 365 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_denny_initial_followers_l3949_394943


namespace NUMINAMATH_CALUDE_perpendicular_bisector_covered_l3949_394956

def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

def perpendicular_bisector (O P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | (q.1 - (O.1 + P.1)/2)^2 + (q.2 - (O.2 + P.2)/2)^2 < ((O.1 - P.1)^2 + (O.2 - P.2)^2) / 4}

def plane_region (m : ℝ) : Set (ℝ × ℝ) := {p | |p.1| + |p.2| ≥ m}

theorem perpendicular_bisector_covered (m : ℝ) :
  (∀ P ∈ circle_O, perpendicular_bisector (0, 0) P ⊆ plane_region m) →
  m ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_covered_l3949_394956


namespace NUMINAMATH_CALUDE_sum_of_slopes_on_parabola_l3949_394983

/-- Given three points on a parabola with a focus satisfying certain conditions,
    prove that the sum of the slopes of the lines connecting these points is zero. -/
theorem sum_of_slopes_on_parabola (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  x₁^2 = 4*y₁ →
  x₂^2 = 4*y₂ →
  x₃^2 = 4*y₃ →
  x₁ + x₂ + x₃ = 0 →
  y₁ + y₂ + y₃ = 3 →
  (y₂ - y₁) / (x₂ - x₁) + (y₃ - y₁) / (x₃ - x₁) + (y₃ - y₂) / (x₃ - x₂) = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_slopes_on_parabola_l3949_394983


namespace NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l3949_394925

/-- The volume of an oblique triangular prism -/
theorem oblique_triangular_prism_volume 
  (S d : ℝ) 
  (h_S : S > 0) 
  (h_d : d > 0) : 
  ∃ V : ℝ, V = (1/2) * d * S ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l3949_394925


namespace NUMINAMATH_CALUDE_max_value_bound_max_value_achievable_l3949_394931

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def max_value (a b c : V) : ℝ :=
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2

theorem max_value_bound (a b c : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 4) (hc : ‖c‖ = 2) :
  max_value a b c ≤ 253 :=
sorry

theorem max_value_achievable :
  ∃ (a b c : V), ‖a‖ = 3 ∧ ‖b‖ = 4 ∧ ‖c‖ = 2 ∧ max_value a b c = 253 :=
sorry

end NUMINAMATH_CALUDE_max_value_bound_max_value_achievable_l3949_394931


namespace NUMINAMATH_CALUDE_peanut_cluster_probability_theorem_l3949_394984

/-- Represents the composition of a box of chocolates -/
structure ChocolateBox where
  total : Nat
  caramels : Nat
  nougats : Nat
  truffles : Nat
  peanut_clusters : Nat

/-- Calculates the probability of selecting a peanut cluster -/
def peanut_cluster_probability (box : ChocolateBox) : Rat :=
  box.peanut_clusters / box.total

/-- Theorem stating the probability of selecting a peanut cluster -/
theorem peanut_cluster_probability_theorem (box : ChocolateBox) 
  (h1 : box.total = 50)
  (h2 : box.caramels = 3)
  (h3 : box.nougats = 2 * box.caramels)
  (h4 : box.truffles = box.caramels + 6)
  (h5 : box.peanut_clusters = box.total - box.caramels - box.nougats - box.truffles) :
  peanut_cluster_probability box = 32 / 50 := by
  sorry

#eval (32 : Rat) / 50

end NUMINAMATH_CALUDE_peanut_cluster_probability_theorem_l3949_394984


namespace NUMINAMATH_CALUDE_intersection_M_N_l3949_394910

def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_M_N : M ∩ N = {x | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3949_394910
