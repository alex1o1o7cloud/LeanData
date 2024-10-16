import Mathlib

namespace NUMINAMATH_CALUDE_reporter_average_words_per_minute_l2583_258303

/-- Calculates the average words per minute for a reporter given their pay structure and work conditions. -/
theorem reporter_average_words_per_minute 
  (word_pay : ℝ)
  (article_pay : ℝ)
  (num_articles : ℕ)
  (total_hours : ℝ)
  (hourly_earnings : ℝ)
  (h1 : word_pay = 0.1)
  (h2 : article_pay = 60)
  (h3 : num_articles = 3)
  (h4 : total_hours = 4)
  (h5 : hourly_earnings = 105) :
  (((hourly_earnings * total_hours - article_pay * num_articles) / word_pay) / (total_hours * 60)) = 10 := by
  sorry

#check reporter_average_words_per_minute

end NUMINAMATH_CALUDE_reporter_average_words_per_minute_l2583_258303


namespace NUMINAMATH_CALUDE_coat_price_calculation_l2583_258305

/-- Calculates the total selling price of a coat given its original price, discount percentage, and tax percentage. -/
def totalSellingPrice (originalPrice discount tax : ℚ) : ℚ :=
  let salePrice := originalPrice * (1 - discount)
  salePrice * (1 + tax)

/-- Theorem stating that the total selling price of a coat with original price $120, 30% discount, and 15% tax is $96.60. -/
theorem coat_price_calculation :
  totalSellingPrice 120 (30 / 100) (15 / 100) = 966 / 10 := by
  sorry

#eval totalSellingPrice 120 (30 / 100) (15 / 100)

end NUMINAMATH_CALUDE_coat_price_calculation_l2583_258305


namespace NUMINAMATH_CALUDE_condition_a_sufficient_not_necessary_l2583_258345

theorem condition_a_sufficient_not_necessary :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_condition_a_sufficient_not_necessary_l2583_258345


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_l2583_258375

/-- Given an isosceles triangle and a rectangle with the same area,
    where the base of the triangle equals the width of the rectangle,
    prove that the height of the triangle is twice the length of the rectangle. -/
theorem isosceles_triangle_height (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  (l * w = 1/2 * w * h) →  -- Areas are equal
  (h = 2 * l) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_height_l2583_258375


namespace NUMINAMATH_CALUDE_metallic_sheet_dimension_l2583_258356

/-- Given a rectangular metallic sheet, prove that the second dimension is 36 m -/
theorem metallic_sheet_dimension (sheet_length : ℝ) (sheet_width : ℝ) 
  (cut_length : ℝ) (box_volume : ℝ) :
  sheet_length = 46 →
  cut_length = 8 →
  box_volume = 4800 →
  box_volume = (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length →
  sheet_width = 36 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_dimension_l2583_258356


namespace NUMINAMATH_CALUDE_basketball_preference_percentage_l2583_258334

theorem basketball_preference_percentage :
  let north_students : ℕ := 2200
  let south_students : ℕ := 2600
  let north_basketball_percentage : ℚ := 20 / 100
  let south_basketball_percentage : ℚ := 35 / 100
  let total_students := north_students + south_students
  let north_basketball := (north_students : ℚ) * north_basketball_percentage
  let south_basketball := (south_students : ℚ) * south_basketball_percentage
  let total_basketball := north_basketball + south_basketball
  let combined_percentage := total_basketball / (total_students : ℚ) * 100
  combined_percentage = 28 := by
sorry

end NUMINAMATH_CALUDE_basketball_preference_percentage_l2583_258334


namespace NUMINAMATH_CALUDE_base_10_to_base_3_l2583_258364

theorem base_10_to_base_3 : 
  (2 * 3^6 + 0 * 3^5 + 0 * 3^4 + 1 * 3^3 + 1 * 3^2 + 2 * 3^1 + 2 * 3^0) = 1589 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_3_l2583_258364


namespace NUMINAMATH_CALUDE_disjoint_sets_property_l2583_258353

theorem disjoint_sets_property (A B : Set ℕ) (h1 : A ∩ B = ∅) (h2 : A ∪ B = Set.univ) :
  ∀ n : ℕ, ∃ a b : ℕ, a ≠ b ∧ a > n ∧ b > n ∧
    (({a, b, a + b} : Set ℕ) ⊆ A ∨ ({a, b, a + b} : Set ℕ) ⊆ B) :=
by sorry

end NUMINAMATH_CALUDE_disjoint_sets_property_l2583_258353


namespace NUMINAMATH_CALUDE_article_gain_percentage_l2583_258322

/-- Calculates the percentage gain when selling an article -/
def percentageGain (costPrice sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Calculates the cost price given a selling price and loss percentage -/
def calculateCostPrice (sellingPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercentage / 100)

theorem article_gain_percentage :
  let lossPrice : ℚ := 102
  let gainPrice : ℚ := 144
  let lossPercentage : ℚ := 15
  let costPrice := calculateCostPrice lossPrice lossPercentage
  percentageGain costPrice gainPrice = 20 := by sorry

end NUMINAMATH_CALUDE_article_gain_percentage_l2583_258322


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2583_258300

theorem arithmetic_calculation : 5 * 12 + 6 * 11 + 13 * 5 + 7 * 9 = 254 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2583_258300


namespace NUMINAMATH_CALUDE_max_students_theorem_l2583_258348

/-- The number of subjects offered at the school -/
def num_subjects : ℕ := 6

/-- The maximum number of students taking both or neither of any two subjects -/
def max_students_per_pair : ℕ := 4

/-- The set of all possible combinations of subjects a student can take -/
def subject_combinations : Finset (Finset ℕ) :=
  Finset.powerset (Finset.range num_subjects)

/-- A function that returns the number of subject pairs a student is either taking both or neither -/
def num_pairs_both_or_neither (subjects : Finset ℕ) : ℕ :=
  Nat.choose (subjects.card) 2 + Nat.choose (num_subjects - subjects.card) 2

/-- The theorem stating the maximum number of students -/
theorem max_students_theorem :
  ∃ (max_students : ℕ),
    (∀ (n : ℕ), n ≤ max_students →
      ∃ (student_subjects : Finset (Finset ℕ)),
        student_subjects.card = n ∧
        student_subjects ⊆ subject_combinations ∧
        (∀ (s1 s2 : Finset ℕ), s1 ∈ student_subjects → s2 ∈ student_subjects → s1 ≠ s2 →
          (s1 ∩ s2).card < max_students_per_pair ∧
          (num_subjects - (s1 ∪ s2).card) < max_students_per_pair)) ∧
    (∀ (n : ℕ), n > max_students →
      ¬∃ (student_subjects : Finset (Finset ℕ)),
        student_subjects.card = n ∧
        student_subjects ⊆ subject_combinations ∧
        (∀ (s1 s2 : Finset ℕ), s1 ∈ student_subjects → s2 ∈ student_subjects → s1 ≠ s2 →
          (s1 ∩ s2).card < max_students_per_pair ∧
          (num_subjects - (s1 ∪ s2).card) < max_students_per_pair)) ∧
    max_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_students_theorem_l2583_258348


namespace NUMINAMATH_CALUDE_senior_sample_size_l2583_258361

theorem senior_sample_size (total : ℕ) (freshmen : ℕ) (sophomores : ℕ) (sample : ℕ) 
  (h_total : total = 900)
  (h_freshmen : freshmen = 240)
  (h_sophomores : sophomores = 260)
  (h_sample : sample = 45) :
  let seniors := total - freshmen - sophomores
  let sampling_fraction := sample / total
  seniors * sampling_fraction = 20 := by
sorry

end NUMINAMATH_CALUDE_senior_sample_size_l2583_258361


namespace NUMINAMATH_CALUDE_equation_equivalence_l2583_258366

theorem equation_equivalence (y : ℝ) :
  (8 * y^2 + 90 * y + 5) / (3 * y^2 + 4 * y + 49) = 4 * y + 1 →
  12 * y^3 + 11 * y^2 + 110 * y + 44 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2583_258366


namespace NUMINAMATH_CALUDE_johns_annual_epipen_cost_l2583_258369

/-- Represents the cost of EpiPens for John over a year -/
def annual_epipen_cost (epipen_cost : ℝ) (insurance_coverage : ℝ) (replacements_per_year : ℕ) : ℝ :=
  replacements_per_year * (1 - insurance_coverage) * epipen_cost

/-- Theorem stating that John's annual cost for EpiPens is $250 -/
theorem johns_annual_epipen_cost :
  annual_epipen_cost 500 0.75 2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_johns_annual_epipen_cost_l2583_258369


namespace NUMINAMATH_CALUDE_mans_speed_l2583_258333

theorem mans_speed (time_minutes : ℝ) (distance_km : ℝ) (h1 : time_minutes = 36) (h2 : distance_km = 6) :
  distance_km / (time_minutes / 60) = 10 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_l2583_258333


namespace NUMINAMATH_CALUDE_winnie_min_checks_l2583_258390

/-- Represents the arrangement of jars in Winnie the Pooh's closet -/
structure JarArrangement where
  total : Nat
  jam : Nat
  honey : Nat
  honey_consecutive : Bool

/-- Defines the minimum number of jars to check to find honey -/
def min_checks (arrangement : JarArrangement) : Nat :=
  1

/-- Theorem stating that for Winnie's specific arrangement, the minimum number of checks is 1 -/
theorem winnie_min_checks :
  ∀ (arrangement : JarArrangement),
    arrangement.total = 11 →
    arrangement.jam = 7 →
    arrangement.honey = 4 →
    arrangement.honey_consecutive = true →
    min_checks arrangement = 1 := by
  sorry

end NUMINAMATH_CALUDE_winnie_min_checks_l2583_258390


namespace NUMINAMATH_CALUDE_max_value_expression_l2583_258340

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a * b + b * c + 2 * c * a ≤ 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2583_258340


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l2583_258387

theorem sphere_surface_area_from_volume :
  ∀ (r : ℝ),
  (4 / 3 : ℝ) * π * r^3 = 72 * π →
  4 * π * r^2 = 36 * π * 2^(2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l2583_258387


namespace NUMINAMATH_CALUDE_chemical_mixture_theorem_l2583_258320

/-- Represents a chemical solution with percentages of two components -/
structure Solution :=
  (percent_a : ℝ)
  (percent_b : ℝ)
  (sum_to_one : percent_a + percent_b = 1)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution_x : Solution)
  (solution_y : Solution)
  (percent_x : ℝ)
  (percent_y : ℝ)
  (sum_to_one : percent_x + percent_y = 1)

/-- Calculates the percentage of chemical a in a mixture -/
def percent_a_in_mixture (m : Mixture) : ℝ :=
  m.percent_x * m.solution_x.percent_a + m.percent_y * m.solution_y.percent_a

theorem chemical_mixture_theorem (x y : Solution) 
  (hx : x.percent_a = 0.4) 
  (hy : y.percent_a = 0.5) : 
  let m : Mixture := {
    solution_x := x,
    solution_y := y,
    percent_x := 0.3,
    percent_y := 0.7,
    sum_to_one := by norm_num
  }
  percent_a_in_mixture m = 0.47 := by
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_theorem_l2583_258320


namespace NUMINAMATH_CALUDE_curve_transformation_l2583_258332

/-- Given a curve C in a plane rectangular coordinate system, 
    prove that its equation is 50x^2 + 72y^2 = 1 after an expansion transformation. -/
theorem curve_transformation (x y x' y' : ℝ) : 
  (x' = 5*x ∧ y' = 3*y) →  -- Transformation equations
  (2*x'^2 + 8*y'^2 = 1) →  -- Equation of transformed curve
  (50*x^2 + 72*y^2 = 1)    -- Equation of original curve C
  := by sorry

end NUMINAMATH_CALUDE_curve_transformation_l2583_258332


namespace NUMINAMATH_CALUDE_crackers_per_friend_l2583_258309

theorem crackers_per_friend (initial_crackers : ℕ) (friends : ℕ) (remaining_crackers : ℕ) 
  (h1 : initial_crackers = 15)
  (h2 : friends = 5)
  (h3 : remaining_crackers = 10) :
  (initial_crackers - remaining_crackers) / friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_crackers_per_friend_l2583_258309


namespace NUMINAMATH_CALUDE_mall_product_properties_l2583_258336

/-- Represents the shopping mall's product pricing and sales model -/
structure ProductModel where
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  sales_volume : ℝ → ℝ
  profit : ℝ → ℝ

/-- The specific product model for the shopping mall -/
def mall_product : ProductModel :=
  { purchase_price := 30
    min_selling_price := 30
    max_selling_price := 55
    sales_volume := λ x => -2 * x + 140
    profit := λ x => (x - 30) * (-2 * x + 140) }

theorem mall_product_properties (x : ℝ) :
  let m := mall_product
  (x ≥ m.min_selling_price ∧ x ≤ m.max_selling_price) →
  (m.profit 35 = 350 ∧
   m.profit 40 = 600 ∧
   ∀ y, m.min_selling_price ≤ y ∧ y ≤ m.max_selling_price → m.profit y ≠ 900) :=
by sorry


end NUMINAMATH_CALUDE_mall_product_properties_l2583_258336


namespace NUMINAMATH_CALUDE_road_graveling_cost_l2583_258317

/-- Calculate the cost of graveling two intersecting roads on a rectangular lawn. -/
theorem road_graveling_cost
  (lawn_length : ℕ)
  (lawn_width : ℕ)
  (road_width : ℕ)
  (gravel_cost_per_sqm : ℕ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 40)
  (h3 : road_width = 10)
  (h4 : gravel_cost_per_sqm = 3) :
  lawn_length * road_width + lawn_width * road_width - road_width * road_width * gravel_cost_per_sqm = 3900 :=
by sorry

end NUMINAMATH_CALUDE_road_graveling_cost_l2583_258317


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2583_258352

def set_A : Set ℝ := {x | x^2 - 4 > 0}
def set_B : Set ℝ := {x | x + 2 < 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2583_258352


namespace NUMINAMATH_CALUDE_wheel_marking_theorem_l2583_258382

theorem wheel_marking_theorem :
  ∃ (R : ℝ), R > 0 ∧ 
    ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 360 → 
      ∃ (n : ℕ), 0 ≤ n - R * θ / 360 ∧ n - R * θ / 360 < R / 360 := by
  sorry

end NUMINAMATH_CALUDE_wheel_marking_theorem_l2583_258382


namespace NUMINAMATH_CALUDE_highest_result_l2583_258321

def alice_calc (x : ℕ) : ℕ := x * 3 - 2 + 3

def bob_calc (x : ℕ) : ℕ := (x - 3) * 3 + 4

def carla_calc (x : ℕ) : ℕ := x * 3 + 4 - 2

theorem highest_result (start : ℕ) (h : start = 12) : 
  carla_calc start > alice_calc start ∧ carla_calc start > bob_calc start := by
  sorry

end NUMINAMATH_CALUDE_highest_result_l2583_258321


namespace NUMINAMATH_CALUDE_sqrt_40_between_6_and_7_l2583_258315

theorem sqrt_40_between_6_and_7 :
  ∃ (x : ℝ), x = Real.sqrt 40 ∧ 6 < x ∧ x < 7 :=
by
  have h1 : Real.sqrt 36 < Real.sqrt 40 ∧ Real.sqrt 40 < Real.sqrt 49 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_40_between_6_and_7_l2583_258315


namespace NUMINAMATH_CALUDE_wendys_final_tally_l2583_258384

/-- Calculates Wendy's final point tally for recycling --/
def wendys_points (cans_points newspaper_points cans_total cans_recycled newspapers_recycled penalty_points bonus_points bonus_cans_threshold bonus_newspapers_threshold : ℕ) : ℕ :=
  let points_earned := cans_points * cans_recycled + newspaper_points * newspapers_recycled
  let points_lost := penalty_points * (cans_total - cans_recycled)
  let bonus := if cans_recycled ≥ bonus_cans_threshold ∧ newspapers_recycled ≥ bonus_newspapers_threshold then bonus_points else 0
  points_earned - points_lost + bonus

/-- Theorem stating that Wendy's final point tally is 69 --/
theorem wendys_final_tally :
  wendys_points 5 10 11 9 3 3 15 10 2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_wendys_final_tally_l2583_258384


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l2583_258301

theorem parallelepiped_volume (base_area : ℝ) (angle : ℝ) (lateral_area1 : ℝ) (lateral_area2 : ℝ) :
  base_area = 4 →
  angle = 30 * π / 180 →
  lateral_area1 = 6 →
  lateral_area2 = 12 →
  ∃ (a b c : ℝ),
    a * b * Real.sin angle = base_area ∧
    a * c = lateral_area1 ∧
    b * c = lateral_area2 ∧
    a * b * c = 12 := by
  sorry

#check parallelepiped_volume

end NUMINAMATH_CALUDE_parallelepiped_volume_l2583_258301


namespace NUMINAMATH_CALUDE_problem_polygon_area_l2583_258341

/-- Represents a polygon with right angles at each corner -/
structure RightAnglePolygon where
  -- Define the lengths of the segments
  left_height : ℝ
  bottom_width : ℝ
  middle_height : ℝ
  middle_width : ℝ
  top_right_height : ℝ
  top_right_width : ℝ
  top_left_height : ℝ
  top_left_width : ℝ

/-- Calculates the area of the RightAnglePolygon -/
def area (p : RightAnglePolygon) : ℝ :=
  p.left_height * p.bottom_width +
  p.middle_height * p.middle_width +
  p.top_right_height * p.top_right_width +
  p.top_left_height * p.top_left_width

/-- The specific polygon from the problem -/
def problem_polygon : RightAnglePolygon :=
  { left_height := 7
  , bottom_width := 6
  , middle_height := 5
  , middle_width := 4
  , top_right_height := 6
  , top_right_width := 5
  , top_left_height := 1
  , top_left_width := 2
  }

/-- Theorem stating that the area of the problem_polygon is 94 -/
theorem problem_polygon_area :
  area problem_polygon = 94 := by
  sorry


end NUMINAMATH_CALUDE_problem_polygon_area_l2583_258341


namespace NUMINAMATH_CALUDE_cody_payment_proof_l2583_258312

def initial_purchase : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8
def cody_payment : ℝ := 17

theorem cody_payment_proof :
  cody_payment = (initial_purchase * (1 + tax_rate) - discount) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cody_payment_proof_l2583_258312


namespace NUMINAMATH_CALUDE_bricks_used_in_scenario_l2583_258360

/-- The number of bricks used in a construction project -/
def bricks_used (walls : ℕ) (courses_per_wall : ℕ) (bricks_per_course : ℕ) (uncompleted_courses : ℕ) : ℕ :=
  walls * courses_per_wall * bricks_per_course - uncompleted_courses * bricks_per_course

/-- Theorem stating that the number of bricks used in the given scenario is 220 -/
theorem bricks_used_in_scenario : bricks_used 4 6 10 2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_bricks_used_in_scenario_l2583_258360


namespace NUMINAMATH_CALUDE_horseback_riding_trip_l2583_258302

/-- Calculates the number of hours traveled on the third day of a horseback riding trip -/
theorem horseback_riding_trip (total_distance : ℝ) (day1_speed day1_time day2_speed1 day2_time1 day2_speed2 day2_time2 day3_speed : ℝ) :
  total_distance = 115 ∧
  day1_speed = 5 ∧
  day1_time = 7 ∧
  day2_speed1 = 6 ∧
  day2_time1 = 6 ∧
  day2_speed2 = day2_speed1 / 2 ∧
  day2_time2 = 3 ∧
  day3_speed = 7 →
  (total_distance - (day1_speed * day1_time + day2_speed1 * day2_time1 + day2_speed2 * day2_time2)) / day3_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_horseback_riding_trip_l2583_258302


namespace NUMINAMATH_CALUDE_sqrt_eight_combinable_with_sqrt_two_l2583_258316

theorem sqrt_eight_combinable_with_sqrt_two :
  ∃ (n : ℤ), Real.sqrt 8 = n * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_eight_combinable_with_sqrt_two_l2583_258316


namespace NUMINAMATH_CALUDE_blue_candy_count_l2583_258368

theorem blue_candy_count (total : ℕ) (red : ℕ) (h1 : total = 3409) (h2 : red = 145) :
  total - red = 3264 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l2583_258368


namespace NUMINAMATH_CALUDE_jacksons_entertainment_spending_l2583_258314

/-- The total amount Jackson spent on entertainment -/
def total_spent (computer_game_price movie_ticket_price number_of_tickets : ℕ) : ℕ :=
  computer_game_price + movie_ticket_price * number_of_tickets

/-- Theorem stating that Jackson's total entertainment spending is $102 -/
theorem jacksons_entertainment_spending :
  total_spent 66 12 3 = 102 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_entertainment_spending_l2583_258314


namespace NUMINAMATH_CALUDE_no_snuggly_integers_l2583_258338

/-- A two-digit positive integer is snuggly if it equals the sum of its nonzero tens digit and the cube of its units digit. -/
def is_snuggly (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n = (n / 10) + (n % 10)^3

/-- There are no snuggly two-digit positive integers. -/
theorem no_snuggly_integers : ¬∃ n : ℕ, is_snuggly n := by
  sorry

end NUMINAMATH_CALUDE_no_snuggly_integers_l2583_258338


namespace NUMINAMATH_CALUDE_tangent_line_slope_intersection_line_equation_l2583_258323

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define point Q
def Q : ℝ × ℝ := (0, -2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem for part 1
theorem tangent_line_slope :
  ∃ m : ℝ, m = -3/4 ∧
  (∀ x y : ℝ, y - P.2 = m * (x - P.1) → 
   (∃ t : ℝ, x = t ∧ y = t ∧ circle_C x y)) ∧
  (∀ x y : ℝ, circle_C x y → (y - P.2 ≠ m * (x - P.1) ∨ (x = P.1 ∧ y = P.2))) :=
sorry

-- Theorem for part 2
theorem intersection_line_equation :
  ∃ k : ℝ, (k = 5/3 ∨ k = 1) ∧
  (∀ x y : ℝ, y = k*x - 2 →
   (∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    y = k*x - 2 ∧
    (A.2 / A.1) * (B.2 / B.1) = -1/7)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_intersection_line_equation_l2583_258323


namespace NUMINAMATH_CALUDE_chick_hits_l2583_258319

theorem chick_hits (chick monkey dog : ℕ) : 
  chick * 9 + monkey * 5 + dog * 2 = 61 →
  chick + monkey + dog = 10 →
  chick ≥ 1 →
  monkey ≥ 1 →
  dog ≥ 1 →
  chick = 5 :=
by sorry

end NUMINAMATH_CALUDE_chick_hits_l2583_258319


namespace NUMINAMATH_CALUDE_impossible_sum_240_l2583_258376

theorem impossible_sum_240 : ¬ ∃ (a b c d e f g h i : ℕ), 
  (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (10 ≤ c ∧ c ≤ 99) ∧
  (10 ≤ d ∧ d ≤ 99) ∧ (10 ≤ e ∧ e ≤ 99) ∧ (10 ≤ f ∧ f ≤ 99) ∧
  (10 ≤ g ∧ g ≤ 99) ∧ (10 ≤ h ∧ h ≤ 99) ∧ (10 ≤ i ∧ i ≤ 99) ∧
  (a % 10 = 9 ∨ a / 10 = 9) ∧ (b % 10 = 9 ∨ b / 10 = 9) ∧
  (c % 10 = 9 ∨ c / 10 = 9) ∧ (d % 10 = 9 ∨ d / 10 = 9) ∧
  (e % 10 = 9 ∨ e / 10 = 9) ∧ (f % 10 = 9 ∨ f / 10 = 9) ∧
  (g % 10 = 9 ∨ g / 10 = 9) ∧ (h % 10 = 9 ∨ h / 10 = 9) ∧
  (i % 10 = 9 ∨ i / 10 = 9) ∧
  a + b + c + d + e + f + g + h + i = 240 :=
by sorry

end NUMINAMATH_CALUDE_impossible_sum_240_l2583_258376


namespace NUMINAMATH_CALUDE_sine_cosine_equality_l2583_258325

theorem sine_cosine_equality (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (612 * π / 180) → n = -18 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_equality_l2583_258325


namespace NUMINAMATH_CALUDE_m_range_l2583_258378

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem m_range (m : ℝ) : 
  (m > 0) → 
  (∀ x, ¬(p x) → ¬(q x m)) → 
  (∃ x, ¬(p x) ∧ (q x m)) → 
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2583_258378


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2583_258347

theorem quadratic_inequality_solution (a c : ℝ) :
  (∀ x : ℝ, (a * x^2 + 2 * x + c < 0) ↔ (x < -1 ∨ x > 2)) →
  a + c = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2583_258347


namespace NUMINAMATH_CALUDE_mike_bought_33_books_l2583_258337

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books books_given_away : ℕ) : ℕ :=
  final_books - (initial_books - books_given_away)

/-- Theorem stating that Mike bought 33 books at the yard sale -/
theorem mike_bought_33_books :
  books_bought 35 56 12 = 33 := by
  sorry

end NUMINAMATH_CALUDE_mike_bought_33_books_l2583_258337


namespace NUMINAMATH_CALUDE_geometric_arithmetic_mean_ratio_sum_l2583_258306

/-- Given a geometric sequence a, b, c and their arithmetic means m and n,
    prove that a/m + c/n = 2 -/
theorem geometric_arithmetic_mean_ratio_sum (a b c m n : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric sequence condition
  (h2 : m = (a + b) / 2)  -- arithmetic mean of a and b
  (h3 : n = (b + c) / 2)  -- arithmetic mean of b and c
  : a / m + c / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_mean_ratio_sum_l2583_258306


namespace NUMINAMATH_CALUDE_triangle_side_length_l2583_258358

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (area : ℝ) :
  b = 3 →
  c = 4 →
  area = 3 * Real.sqrt 3 →
  area = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  (a = Real.sqrt 13 ∨ a = Real.sqrt 37) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2583_258358


namespace NUMINAMATH_CALUDE_a_investment_is_400_l2583_258346

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  a_investment : ℝ
  b_investment : ℝ
  total_profit : ℝ
  a_profit : ℝ
  b_investment_time : ℝ
  total_time : ℝ

/-- Theorem stating that given the conditions, A's investment was $400 -/
theorem a_investment_is_400 (scenario : InvestmentScenario) 
  (h1 : scenario.b_investment = 200)
  (h2 : scenario.total_profit = 100)
  (h3 : scenario.a_profit = 80)
  (h4 : scenario.b_investment_time = 6)
  (h5 : scenario.total_time = 12)
  (h6 : scenario.a_investment * scenario.total_time / 
        (scenario.b_investment * scenario.b_investment_time) = 
        scenario.a_profit / (scenario.total_profit - scenario.a_profit)) :
  scenario.a_investment = 400 := by
  sorry


end NUMINAMATH_CALUDE_a_investment_is_400_l2583_258346


namespace NUMINAMATH_CALUDE_exam_average_marks_l2583_258367

theorem exam_average_marks (num_papers : ℕ) (geography_increase : ℕ) (history_increase : ℕ) (new_average : ℕ) :
  num_papers = 11 →
  geography_increase = 20 →
  history_increase = 2 →
  new_average = 65 →
  (num_papers * new_average - geography_increase - history_increase) / num_papers = 63 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_marks_l2583_258367


namespace NUMINAMATH_CALUDE_a_max_value_l2583_258326

def a (n : ℕ+) : ℚ := n / (n^2 + 90)

theorem a_max_value : ∀ n : ℕ+, a n ≤ 1/19 := by
  sorry

end NUMINAMATH_CALUDE_a_max_value_l2583_258326


namespace NUMINAMATH_CALUDE_fraction_simplification_l2583_258359

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  1 / (x - 1) - 2 / (x^2 - 1) = 1 / (x + 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2583_258359


namespace NUMINAMATH_CALUDE_river_flow_speed_l2583_258304

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey --/
theorem river_flow_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 6)
  (h2 : distance = 64)
  (h3 : total_time = 24) :
  ∃ (v : ℝ), v = 2 ∧ 
  (distance / (boat_speed - v) + distance / (boat_speed + v) = total_time) :=
by
  sorry

#check river_flow_speed

end NUMINAMATH_CALUDE_river_flow_speed_l2583_258304


namespace NUMINAMATH_CALUDE_cuboid_sphere_surface_area_l2583_258308

-- Define the cuboid
structure Cuboid where
  face_area1 : ℝ
  face_area2 : ℝ
  face_area3 : ℝ
  vertices_on_sphere : Bool

-- Define the theorem
theorem cuboid_sphere_surface_area 
  (c : Cuboid) 
  (h1 : c.face_area1 = 12) 
  (h2 : c.face_area2 = 15) 
  (h3 : c.face_area3 = 20) 
  (h4 : c.vertices_on_sphere = true) : 
  ∃ (sphere_surface_area : ℝ), sphere_surface_area = 50 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_cuboid_sphere_surface_area_l2583_258308


namespace NUMINAMATH_CALUDE_distance_between_points_l2583_258394

/-- The distance between two points (3,2,0) and (7,6,0) in 3D space is 4√2. -/
theorem distance_between_points : Real.sqrt 32 = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2583_258394


namespace NUMINAMATH_CALUDE_custom_op_identity_l2583_258374

/-- Custom operation ⊗ defined as k ⊗ l = k^2 - l^2 -/
def custom_op (k l : ℝ) : ℝ := k^2 - l^2

/-- Theorem stating that k ⊗ (k ⊗ k) = k^2 -/
theorem custom_op_identity (k : ℝ) : custom_op k (custom_op k k) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_identity_l2583_258374


namespace NUMINAMATH_CALUDE_tabithas_initial_money_l2583_258327

theorem tabithas_initial_money :
  ∀ (initial : ℚ) 
    (given_to_mom : ℚ) 
    (num_items : ℕ) 
    (item_cost : ℚ) 
    (money_left : ℚ),
  given_to_mom = 8 →
  num_items = 5 →
  item_cost = 1/2 →
  money_left = 6 →
  initial - given_to_mom = 2 * ((initial - given_to_mom) / 2 - num_items * item_cost - money_left) →
  initial = 25 := by
sorry

end NUMINAMATH_CALUDE_tabithas_initial_money_l2583_258327


namespace NUMINAMATH_CALUDE_tan_theta_equation_l2583_258330

open Real

theorem tan_theta_equation (θ : ℝ) (h1 : π/4 < θ ∧ θ < π/2) 
  (h2 : tan θ + tan (3*θ) + tan (5*θ) = 0) : tan θ = sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_equation_l2583_258330


namespace NUMINAMATH_CALUDE_markers_leftover_l2583_258392

theorem markers_leftover (total_markers : ℕ) (num_packages : ℕ) (h1 : total_markers = 154) (h2 : num_packages = 13) :
  total_markers % num_packages = 11 := by
sorry

end NUMINAMATH_CALUDE_markers_leftover_l2583_258392


namespace NUMINAMATH_CALUDE_total_children_l2583_258351

/-- The number of children who like cabbage -/
def cabbage_lovers : ℕ := 7

/-- The number of children who like carrots -/
def carrot_lovers : ℕ := 6

/-- The number of children who like peas -/
def pea_lovers : ℕ := 5

/-- The number of children who like both cabbage and carrots -/
def cabbage_carrot_lovers : ℕ := 4

/-- The number of children who like both cabbage and peas -/
def cabbage_pea_lovers : ℕ := 3

/-- The number of children who like both carrots and peas -/
def carrot_pea_lovers : ℕ := 2

/-- The number of children who like all three vegetables -/
def all_veg_lovers : ℕ := 1

/-- The theorem stating the total number of children in the family -/
theorem total_children : 
  cabbage_lovers + carrot_lovers + pea_lovers - 
  cabbage_carrot_lovers - cabbage_pea_lovers - carrot_pea_lovers + 
  all_veg_lovers = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_children_l2583_258351


namespace NUMINAMATH_CALUDE_peanut_distribution_l2583_258311

theorem peanut_distribution (x₁ x₂ x₃ x₄ x₅ : ℕ) : 
  x₁ + x₂ + x₃ + x₄ + x₅ = 100 ∧
  x₁ + x₂ = 52 ∧
  x₂ + x₃ = 43 ∧
  x₃ + x₄ = 34 ∧
  x₄ + x₅ = 30 →
  x₁ = 27 ∧ x₂ = 25 ∧ x₃ = 18 ∧ x₄ = 16 ∧ x₅ = 14 :=
by
  sorry

#check peanut_distribution

end NUMINAMATH_CALUDE_peanut_distribution_l2583_258311


namespace NUMINAMATH_CALUDE_min_distance_sum_of_quadratic_roots_l2583_258396

theorem min_distance_sum_of_quadratic_roots : 
  ∃ (α β : ℝ), (α^2 - 6*α + 5 = 0) ∧ (β^2 - 6*β + 5 = 0) ∧
  (∀ x : ℝ, |x - α| + |x - β| ≥ 4) ∧
  (∃ x : ℝ, |x - α| + |x - β| = 4) := by
sorry

end NUMINAMATH_CALUDE_min_distance_sum_of_quadratic_roots_l2583_258396


namespace NUMINAMATH_CALUDE_monotone_function_implies_increasing_sequence_but_not_converse_l2583_258363

theorem monotone_function_implies_increasing_sequence_but_not_converse 
  (f : ℝ → ℝ) (a : ℕ → ℝ) (h : ∀ n, a n = f n) :
  (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y) →
  (∀ n m, 1 ≤ n ∧ n ≤ m → a n ≤ a m) ∧
  ¬ ((∀ n m, 1 ≤ n ∧ n ≤ m → a n ≤ a m) →
     (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y)) :=
by sorry

end NUMINAMATH_CALUDE_monotone_function_implies_increasing_sequence_but_not_converse_l2583_258363


namespace NUMINAMATH_CALUDE_one_is_not_prime_and_not_composite_l2583_258310

-- Define the properties of natural numbers based on their divisors
def has_only_one_divisor (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

-- Theorem to prove
theorem one_is_not_prime_and_not_composite : 
  ¬(is_prime 1 ∧ ¬is_composite 1) :=
sorry

end NUMINAMATH_CALUDE_one_is_not_prime_and_not_composite_l2583_258310


namespace NUMINAMATH_CALUDE_weaving_increase_proof_l2583_258377

/-- Represents the daily increase in weaving output -/
def daily_increase : ℚ := 16/29

/-- The amount woven on the first day -/
def first_day_output : ℚ := 5

/-- The number of days -/
def num_days : ℕ := 30

/-- The total amount woven in 30 days -/
def total_output : ℚ := 390

theorem weaving_increase_proof :
  first_day_output * num_days + (num_days * (num_days - 1) / 2) * daily_increase = total_output :=
sorry

end NUMINAMATH_CALUDE_weaving_increase_proof_l2583_258377


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2583_258342

theorem sin_2alpha_value (α : ℝ) (h : Real.sin (α - π/4) = -Real.cos (2*α)) :
  Real.sin (2*α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2583_258342


namespace NUMINAMATH_CALUDE_cleaning_event_children_count_l2583_258393

theorem cleaning_event_children_count (total_members : ℕ) 
  (adult_men_percentage : ℚ) (h1 : total_members = 2000) 
  (h2 : adult_men_percentage = 30 / 100) : 
  total_members - (adult_men_percentage * total_members).num - 
  (2 * (adult_men_percentage * total_members).num) = 200 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_event_children_count_l2583_258393


namespace NUMINAMATH_CALUDE_max_value_of_function_max_value_achievable_l2583_258318

theorem max_value_of_function (x : ℝ) (hx : x > 0) : 
  (-2 * x^2 + x - 3) / x ≤ 1 - 2 * Real.sqrt 6 := by sorry

theorem max_value_achievable : 
  ∃ x : ℝ, x > 0 ∧ (-2 * x^2 + x - 3) / x = 1 - 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_max_value_of_function_max_value_achievable_l2583_258318


namespace NUMINAMATH_CALUDE_complex_subtraction_l2583_258335

theorem complex_subtraction : (7 - 3*Complex.I) - (2 + 5*Complex.I) = 5 - 8*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2583_258335


namespace NUMINAMATH_CALUDE_specific_gathering_handshakes_l2583_258388

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : ℕ
  people : ℕ
  circular : Bool
  shake_all : Bool
  no_spouse : Bool
  no_neighbors : Bool

/-- Calculates the number of handshakes in the gathering -/
def handshakes (g : Gathering) : ℕ :=
  (g.people * (g.people - 3)) / 2

/-- Theorem stating the number of handshakes for the specific gathering described in the problem -/
theorem specific_gathering_handshakes :
  let g : Gathering := {
    couples := 8,
    people := 16,
    circular := true,
    shake_all := true,
    no_spouse := true,
    no_neighbors := true
  }
  handshakes g = 96 := by
  sorry

end NUMINAMATH_CALUDE_specific_gathering_handshakes_l2583_258388


namespace NUMINAMATH_CALUDE_max_sum_and_reciprocal_l2583_258379

theorem max_sum_and_reciprocal (nums : Finset ℝ) (x : ℝ) :
  (Finset.card nums = 2023) →
  (∀ y ∈ nums, y > 0) →
  (x ∈ nums) →
  (Finset.sum nums id = 2024) →
  (Finset.sum nums (λ y => 1 / y) = 2024) →
  (x + 1 / x ≤ 4096094 / 2024) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_and_reciprocal_l2583_258379


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l2583_258350

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^2) 
  (h₂ : a₂ = 2^(3/2)) 
  (h₃ : a₃ = 2) 
  (h_gp : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : 
  ∃ a₄ : ℝ, a₄ = a₃ * (a₃ / a₂) ∧ a₄ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l2583_258350


namespace NUMINAMATH_CALUDE_owls_on_fence_l2583_258389

theorem owls_on_fence (initial_owls joining_owls : ℕ) : 
  initial_owls = 3 → joining_owls = 2 → initial_owls + joining_owls = 5 :=
by sorry

end NUMINAMATH_CALUDE_owls_on_fence_l2583_258389


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2583_258380

/-- Given a train of length 140 meters passing a platform of length 260 meters in 23.998080153587715 seconds,
    prove that the speed of the train is 60.0048 kilometers per hour. -/
theorem train_speed_calculation (train_length platform_length time_to_pass : ℝ)
    (h1 : train_length = 140)
    (h2 : platform_length = 260)
    (h3 : time_to_pass = 23.998080153587715) :
    (train_length + platform_length) / time_to_pass * 3.6 = 60.0048 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2583_258380


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l2583_258349

open Real

theorem inclination_angle_of_line (x y : ℝ) :
  let line_equation := x * tan (π / 3) + y + 2 = 0
  let inclination_angle := 2 * π / 3
  line_equation → ∃ α, α = inclination_angle ∧ tan α = -tan (π / 3) ∧ 0 ≤ α ∧ α < π :=
by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l2583_258349


namespace NUMINAMATH_CALUDE_min_value_expression_l2583_258399

theorem min_value_expression (a : ℝ) (h : a > 1) :
  (4 / (a - 1)) + a ≥ 5 ∧ ((4 / (a - 1)) + a = 5 ↔ a = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2583_258399


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2583_258357

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- For any geometric sequence, the average of the squares of the second and fourth terms
    is greater than or equal to the square of the third term. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : IsGeometricSequence a) :
    (a 2)^2 / 2 + (a 4)^2 / 2 ≥ (a 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2583_258357


namespace NUMINAMATH_CALUDE_john_stereo_trade_in_l2583_258397

/-- The cost of John's old stereo system -/
def old_system_cost : ℝ := 250

/-- The trade-in value as a percentage of the old system's cost -/
def trade_in_percentage : ℝ := 0.80

/-- The cost of the new stereo system before discount -/
def new_system_cost : ℝ := 600

/-- The discount percentage on the new system -/
def discount_percentage : ℝ := 0.25

/-- The amount John spent out of pocket -/
def out_of_pocket : ℝ := 250

theorem john_stereo_trade_in :
  old_system_cost * trade_in_percentage + out_of_pocket =
  new_system_cost * (1 - discount_percentage) :=
by sorry

end NUMINAMATH_CALUDE_john_stereo_trade_in_l2583_258397


namespace NUMINAMATH_CALUDE_final_coin_count_l2583_258398

def coin_collection (initial : ℕ) (years : ℕ) : ℕ :=
  let year1 := initial * 2
  let year2 := year1 + 12 * 3
  let year3 := year2 + 12 / 3
  let year4 := year3 - year3 / 4
  year4

theorem final_coin_count : coin_collection 50 4 = 105 := by
  sorry

end NUMINAMATH_CALUDE_final_coin_count_l2583_258398


namespace NUMINAMATH_CALUDE_f_zero_value_l2583_258371

def is_nonneg_int (x : ℤ) : Prop := x ≥ 0

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ m n, is_nonneg_int m → is_nonneg_int n →
    f (m^2 + n^2) = (f m - f n)^2 + f (2*m*n)

theorem f_zero_value (f : ℤ → ℤ) :
  (∀ x, is_nonneg_int (f x)) →
  functional_equation f →
  8 * f 0 + 9 * f 1 = 2006 →
  f 0 = 118 := by sorry

end NUMINAMATH_CALUDE_f_zero_value_l2583_258371


namespace NUMINAMATH_CALUDE_computer_pricing_l2583_258385

theorem computer_pricing (C : ℝ) : 
  C + 0.60 * C = 2560 → C + 0.40 * C = 2240 := by sorry

end NUMINAMATH_CALUDE_computer_pricing_l2583_258385


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2583_258395

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2583_258395


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l2583_258355

/-- Given that 40 cows eat 40 bags of husk in 40 days, prove that one cow will eat one bag of husk in 40 days. -/
theorem one_cow_one_bag_days (cows bags days : ℕ) (h : cows = 40 ∧ bags = 40 ∧ days = 40) : 
  (cows * bags) / (cows * days) = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l2583_258355


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2583_258386

/-- The repeating decimal 0.overline{6} --/
def repeating_six : ℚ := 2/3

/-- The repeating decimal 0.overline{3} --/
def repeating_three : ℚ := 1/3

/-- The sum of 0.overline{6} and 0.overline{3} is equal to 1 --/
theorem sum_of_repeating_decimals : repeating_six + repeating_three = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2583_258386


namespace NUMINAMATH_CALUDE_discounted_notebooks_cost_l2583_258331

/-- The total cost of purchasing discounted notebooks -/
theorem discounted_notebooks_cost 
  (x : ℝ) -- original price of a notebook in yuan
  (y : ℝ) -- discount amount in yuan
  : 5 * (x - y) = 5 * x - 5 * y := by
  sorry

end NUMINAMATH_CALUDE_discounted_notebooks_cost_l2583_258331


namespace NUMINAMATH_CALUDE_study_time_calculation_l2583_258328

theorem study_time_calculation (total_hours : ℝ) (tv_fraction : ℝ) (study_fraction : ℝ) : 
  total_hours = 24 →
  tv_fraction = 1/5 →
  study_fraction = 1/4 →
  (total_hours * (1 - tv_fraction) * study_fraction) * 60 = 288 := by
sorry

end NUMINAMATH_CALUDE_study_time_calculation_l2583_258328


namespace NUMINAMATH_CALUDE_circle_sum_center_radius_l2583_258383

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x - 4 = -y^2 + 2*y

-- Define the center and radius of the circle
def circle_center_radius (a b r : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_center_radius :
  ∃ a b r, circle_center_radius a b r ∧ a + b + r = 5 + Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_circle_sum_center_radius_l2583_258383


namespace NUMINAMATH_CALUDE_eggs_in_two_boxes_l2583_258373

def eggs_per_box : ℕ := 3
def number_of_boxes : ℕ := 2

theorem eggs_in_two_boxes :
  eggs_per_box * number_of_boxes = 6 := by sorry

end NUMINAMATH_CALUDE_eggs_in_two_boxes_l2583_258373


namespace NUMINAMATH_CALUDE_series_sum_l2583_258362

theorem series_sum : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l2583_258362


namespace NUMINAMATH_CALUDE_square_triangulation_100_points_l2583_258381

/-- Represents a triangulation of a square with additional points -/
structure SquareTriangulation where
  n : ℕ  -- number of additional points inside the square
  triangles : ℕ  -- number of triangles in the triangulation

/-- Theorem: A square triangulation with 100 additional points has 202 triangles -/
theorem square_triangulation_100_points :
  ∀ (st : SquareTriangulation), st.n = 100 → st.triangles = 202 := by
  sorry

end NUMINAMATH_CALUDE_square_triangulation_100_points_l2583_258381


namespace NUMINAMATH_CALUDE_clara_triple_anna_age_l2583_258370

def anna_current_age : ℕ := 54
def clara_current_age : ℕ := 80

theorem clara_triple_anna_age :
  ∃ (years_ago : ℕ), 
    clara_current_age - years_ago = 3 * (anna_current_age - years_ago) ∧
    years_ago = 41 :=
by sorry

end NUMINAMATH_CALUDE_clara_triple_anna_age_l2583_258370


namespace NUMINAMATH_CALUDE_box_volume_increase_l2583_258344

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 5670, surface area is 2534, and sum of edges is 252,
    then increasing each dimension by 1 results in a volume of 7001 -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 5670)
  (hs : 2 * (l * w + w * h + h * l) = 2534)
  (he : 4 * (l + w + h) = 252) :
  (l + 1) * (w + 1) * (h + 1) = 7001 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2583_258344


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2583_258391

theorem circle_area_ratio (d : ℝ) (h : d > 0) : 
  (π * ((7 * d) / 2)^2) / (π * (d / 2)^2) = 49 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2583_258391


namespace NUMINAMATH_CALUDE_number_equation_l2583_258329

theorem number_equation : ∃ n : ℝ, (n - 5) * 4 = n * 2 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2583_258329


namespace NUMINAMATH_CALUDE_tens_digit_of_2015_pow_2016_minus_2017_l2583_258372

theorem tens_digit_of_2015_pow_2016_minus_2017 :
  (2015^2016 - 2017) % 100 / 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_2015_pow_2016_minus_2017_l2583_258372


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2583_258313

theorem rationalize_denominator :
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2583_258313


namespace NUMINAMATH_CALUDE_hyperbola_tangent_intersection_product_l2583_258343

/-- The hyperbola with equation x²/4 - y² = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) - p.2^2 = 1}

/-- The asymptotes of the hyperbola -/
def asymptotes : Set (Set (ℝ × ℝ)) :=
  {{p : ℝ × ℝ | p.2 = p.1 / 2}, {p : ℝ × ℝ | p.2 = -p.1 / 2}}

/-- A line tangent to the hyperbola at point P -/
def tangent_line (P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {Q : ℝ × ℝ | ∃ t : ℝ, Q = (P.1 + t, P.2 + t * (P.2 / P.1))}

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem hyperbola_tangent_intersection_product (P : ℝ × ℝ) 
  (h_P : P ∈ hyperbola) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ (tangent_line P ∩ (⋃₀ asymptotes))) 
  (h_N : N ∈ (tangent_line P ∩ (⋃₀ asymptotes))) 
  (h_M_ne_N : M ≠ N) :
  dot_product M N = 3 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_tangent_intersection_product_l2583_258343


namespace NUMINAMATH_CALUDE_range_of_n_l2583_258339

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (n m : ℝ) : Set ℝ := {x | n - m < x ∧ x < n + m}

theorem range_of_n (h : ∀ n : ℝ, (∃ x, x ∈ A ∩ B n 1) → ∃ x, x ∈ A ∩ B n 1) :
  ∀ n : ℝ, n ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_n_l2583_258339


namespace NUMINAMATH_CALUDE_geometric_sequence_range_l2583_258354

theorem geometric_sequence_range (a₁ a₂ a₃ a₄ : ℝ) :
  (0 < a₁ ∧ a₁ < 1) →
  (1 < a₂ ∧ a₂ < 2) →
  (2 < a₃ ∧ a₃ < 3) →
  (∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧ a₄ = a₁ * q^3) →
  (2 * Real.sqrt 2 < a₄ ∧ a₄ < 9) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_range_l2583_258354


namespace NUMINAMATH_CALUDE_fifth_item_equals_one_fifteenth_l2583_258365

-- Define the sequence a_n
def a (n : ℕ) : ℚ := 2 / (n^2 + n : ℚ)

-- Theorem statement
theorem fifth_item_equals_one_fifteenth : a 5 = 1/15 := by
  sorry

end NUMINAMATH_CALUDE_fifth_item_equals_one_fifteenth_l2583_258365


namespace NUMINAMATH_CALUDE_product_zero_in_special_set_l2583_258307

theorem product_zero_in_special_set (n : ℕ) (h : n = 1997) (S : Finset ℝ) 
  (hcard : S.card = n)
  (hsum : ∀ x ∈ S, (S.sum id - x) ∈ S) :
  S.prod id = 0 :=
sorry

end NUMINAMATH_CALUDE_product_zero_in_special_set_l2583_258307


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_l2583_258324

theorem largest_power_dividing_factorial_squared (p : ℕ) (hp : Nat.Prime p) :
  (∀ n : ℕ, (Nat.factorial p)^n ∣ Nat.factorial (p^2)) ↔ n ≤ p + 1 :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_l2583_258324
