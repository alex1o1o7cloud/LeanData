import Mathlib

namespace NUMINAMATH_CALUDE_adjacent_combinations_l2233_223326

def number_of_people : ℕ := 9
def number_of_friends : ℕ := 8
def adjacent_positions : ℕ := 2

theorem adjacent_combinations :
  Nat.choose number_of_friends adjacent_positions = 28 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_combinations_l2233_223326


namespace NUMINAMATH_CALUDE_fraction_equality_l2233_223309

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + (a + 6 * b) / (b + 6 * a) = 2) : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2233_223309


namespace NUMINAMATH_CALUDE_distance_between_cities_l2233_223386

/-- The distance between two cities given the speeds of two cars and their time difference --/
theorem distance_between_cities (v1 v2 : ℝ) (t_diff : ℝ) (h1 : v1 = 60) (h2 : v2 = 70) (h3 : t_diff = 0.25) :
  ∃ d : ℝ, d = 105 ∧ d = v1 * (d / v1) ∧ d = v2 * (d / v2 - t_diff) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2233_223386


namespace NUMINAMATH_CALUDE_women_fair_hair_percentage_l2233_223319

/-- Represents the percentage of fair-haired employees who are women -/
def percent_fair_haired_women : ℝ := 0.40

/-- Represents the percentage of employees who have fair hair -/
def percent_fair_haired : ℝ := 0.25

/-- Represents the percentage of employees who are women with fair hair -/
def percent_women_fair_hair : ℝ := percent_fair_haired_women * percent_fair_haired

theorem women_fair_hair_percentage :
  percent_women_fair_hair = 0.10 := by sorry

end NUMINAMATH_CALUDE_women_fair_hair_percentage_l2233_223319


namespace NUMINAMATH_CALUDE_sum_of_squares_bound_l2233_223353

theorem sum_of_squares_bound 
  (a b c d x y z t : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hz : z ≥ 1) 
  (ht : t ≥ 1) 
  (hsum : a + b + c + d + x + y + z + t = 8) : 
  a^2 + b^2 + c^2 + d^2 + x^2 + y^2 + z^2 + t^2 ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_bound_l2233_223353


namespace NUMINAMATH_CALUDE_smarties_leftover_l2233_223310

theorem smarties_leftover (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_smarties_leftover_l2233_223310


namespace NUMINAMATH_CALUDE_email_subscription_day_l2233_223385

theorem email_subscription_day :
  ∀ (x : ℕ),
  (x ≤ 30) →
  (20 * x + 25 * (30 - x) = 675) →
  x = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_email_subscription_day_l2233_223385


namespace NUMINAMATH_CALUDE_inequality_hold_l2233_223373

theorem inequality_hold (x : ℝ) : 
  x ≥ -1/2 → x ≠ 0 → 
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9 ↔ 
   (-1/2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 24)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_hold_l2233_223373


namespace NUMINAMATH_CALUDE_inequalities_proof_l2233_223356

theorem inequalities_proof :
  (∀ x : ℝ, 2 * x^2 + 5 * x + 3 > x^2 + 3 * x + 1) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → Real.sqrt a > Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2233_223356


namespace NUMINAMATH_CALUDE_childrens_book_balances_weights_l2233_223365

/-- Represents a two-arm scale with items on both sides -/
structure TwoArmScale where
  left_side : ℝ
  right_side : ℝ

/-- Checks if the scale is balanced -/
def is_balanced (scale : TwoArmScale) : Prop :=
  scale.left_side = scale.right_side

/-- The weight of the children's book -/
def childrens_book_weight : ℝ := 1.1

/-- The combined weight of the weights on the right side of the scale -/
def right_side_weight : ℝ := 0.5 + 0.3 + 0.3

/-- Theorem stating that the children's book weight balances the given weights -/
theorem childrens_book_balances_weights :
  is_balanced { left_side := childrens_book_weight, right_side := right_side_weight } :=
by sorry

end NUMINAMATH_CALUDE_childrens_book_balances_weights_l2233_223365


namespace NUMINAMATH_CALUDE_larger_number_proof_l2233_223366

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1360) 
  (h2 : y = 6 * x + 15) : 
  y = 1629 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2233_223366


namespace NUMINAMATH_CALUDE_missing_number_proof_l2233_223307

theorem missing_number_proof : ∃ n : ℝ, n * 120 = 173 * 240 ∧ n = 345.6 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2233_223307


namespace NUMINAMATH_CALUDE_oil_depth_relationship_l2233_223384

/-- Represents a right cylindrical tank -/
structure CylindricalTank where
  height : ℝ
  baseDiameter : ℝ

/-- Represents the oil level in the tank -/
structure OilLevel where
  depthWhenFlat : ℝ
  depthWhenUpright : ℝ

/-- The theorem stating the relationship between oil depths -/
theorem oil_depth_relationship (tank : CylindricalTank) (oil : OilLevel) :
  tank.height = 15 ∧ 
  tank.baseDiameter = 6 ∧ 
  oil.depthWhenFlat = 4 →
  oil.depthWhenUpright = 15 := by
  sorry


end NUMINAMATH_CALUDE_oil_depth_relationship_l2233_223384


namespace NUMINAMATH_CALUDE_vector_parallel_sum_l2233_223339

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

theorem vector_parallel_sum (m : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (3, m)
  parallel a (a.1 + b.1, a.2 + b.2) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_sum_l2233_223339


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l2233_223359

/-- The sequence G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_1000 :
  units_digit (3^(3^1000)) = 1 →
  units_digit (G 1000) = 2 :=
sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l2233_223359


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2233_223397

def point : ℝ × ℝ := (2, -3)

def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_in_fourth_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2233_223397


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l2233_223358

-- Define the function f
def f (x : ℝ) : ℝ := 1 - |x - 2|

-- Theorem for the first part
theorem solution_set_f (x : ℝ) :
  f x > 1 - |x + 4| ↔ x > -1 :=
sorry

-- Theorem for the second part
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 2 (5/2), f x > |x - m|) ↔ m ∈ Set.Ico 2 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l2233_223358


namespace NUMINAMATH_CALUDE_brothers_age_sum_l2233_223371

theorem brothers_age_sum : ∃ x : ℤ, 5 * x - 6 = 89 :=
  sorry

end NUMINAMATH_CALUDE_brothers_age_sum_l2233_223371


namespace NUMINAMATH_CALUDE_number_equation_solution_l2233_223352

theorem number_equation_solution : ∃ x : ℚ, 3 + (1/2) * (1/3) * (1/5) * x = (1/15) * x ∧ x = 90 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2233_223352


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l2233_223300

/-- Given a sequence {aₙ} with sum Sn = (a₁(4ⁿ - 1)) / 3 and a₄ = 32, prove a₁ = 1/2 -/
theorem sequence_sum_problem (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h1 : ∀ n, S n = (a 1 * (4^n - 1)) / 3)
  (h2 : a 4 = 32) :
  a 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l2233_223300


namespace NUMINAMATH_CALUDE_min_overlap_coffee_tea_l2233_223383

theorem min_overlap_coffee_tea (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 0.85)
  (h2 : tea_drinkers = 0.80) :
  0.65 ≤ coffee_drinkers + tea_drinkers - 1 :=
sorry

end NUMINAMATH_CALUDE_min_overlap_coffee_tea_l2233_223383


namespace NUMINAMATH_CALUDE_range_of_m_l2233_223370

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2233_223370


namespace NUMINAMATH_CALUDE_basic_astrophysics_is_108_degrees_l2233_223381

/-- Represents the research and development budget allocation --/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ

/-- The total degrees in a circle --/
def total_degrees : ℝ := 360

/-- Calculate the degrees for basic astrophysics research --/
def basic_astrophysics_degrees (ba : BudgetAllocation) : ℝ :=
  total_degrees * (1 - (ba.microphotonics + ba.home_electronics + ba.food_additives + 
                        ba.genetically_modified_microorganisms + ba.industrial_lubricants))

/-- Theorem stating that the degrees for basic astrophysics research is 108 --/
theorem basic_astrophysics_is_108_degrees (ba : BudgetAllocation) 
    (h1 : ba.microphotonics = 0.09)
    (h2 : ba.home_electronics = 0.14)
    (h3 : ba.food_additives = 0.10)
    (h4 : ba.genetically_modified_microorganisms = 0.29)
    (h5 : ba.industrial_lubricants = 0.08) :
    basic_astrophysics_degrees ba = 108 := by
  sorry

#check basic_astrophysics_is_108_degrees

end NUMINAMATH_CALUDE_basic_astrophysics_is_108_degrees_l2233_223381


namespace NUMINAMATH_CALUDE_sector_area_given_arc_length_l2233_223350

/-- Given a circular sector where the arc length corresponding to a central angle of 2 radians is 4 cm, 
    the area of this sector is 4 cm². -/
theorem sector_area_given_arc_length (r : ℝ) (h : 2 * r = 4) : r * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_given_arc_length_l2233_223350


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2233_223305

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 6 + a 8 + a 10 = 72) :
  2 * a 10 - a 12 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2233_223305


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2233_223393

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y r : ℝ) : Prop := (x - 3)^2 + y^2 = r^2

-- Define the tangency condition
def are_tangent (r : ℝ) : Prop := ∃ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y r

-- Theorem statement
theorem tangent_circles_radius (r : ℝ) (h1 : r > 0) (h2 : are_tangent r) : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l2233_223393


namespace NUMINAMATH_CALUDE_algebraic_expansions_l2233_223390

theorem algebraic_expansions (x y : ℝ) :
  ((x + 2*y - 3) * (x - 2*y + 3) = x^2 - 4*y^2 + 12*y - 9) ∧
  ((2*x^3*y)^2 * (-2*x*y) + (-2*x^3*y)^3 / (2*x^2) = -12*x^7*y^3) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expansions_l2233_223390


namespace NUMINAMATH_CALUDE_english_physical_novels_count_l2233_223378

/-- Represents Iesha's book collection -/
structure BookCollection where
  total : ℕ
  english : ℕ
  school : ℕ
  sports : ℕ
  novels : ℕ
  english_sports : ℕ
  english_school : ℕ
  english_novels : ℕ
  digital_novels : ℕ
  physical_novels : ℕ

/-- Theorem stating the number of English physical format novels in Iesha's collection -/
theorem english_physical_novels_count (c : BookCollection) : c.physical_novels = 135 :=
  by
  have h1 : c.total = 2000 := by sorry
  have h2 : c.english = c.total / 2 := by sorry
  have h3 : c.school = c.total * 30 / 100 := by sorry
  have h4 : c.sports = c.total * 25 / 100 := by sorry
  have h5 : c.novels = c.total - c.school - c.sports := by sorry
  have h6 : c.english_sports = c.english * 10 / 100 := by sorry
  have h7 : c.english_school = c.english * 45 / 100 := by sorry
  have h8 : c.english_novels = c.english - c.english_sports - c.english_school := by sorry
  have h9 : c.digital_novels = c.english_novels * 70 / 100 := by sorry
  have h10 : c.physical_novels = c.english_novels - c.digital_novels := by sorry
  sorry

end NUMINAMATH_CALUDE_english_physical_novels_count_l2233_223378


namespace NUMINAMATH_CALUDE_specific_polygon_perimeter_l2233_223306

/-- A polygon that forms part of a square -/
structure PartialSquarePolygon where
  /-- The length of each visible side of the polygon -/
  visible_side_length : ℝ
  /-- The fraction of the square that the polygon occupies -/
  occupied_fraction : ℝ
  /-- Assumption that the visible side length is positive -/
  visible_side_positive : visible_side_length > 0
  /-- Assumption that the occupied fraction is between 0 and 1 -/
  occupied_fraction_valid : 0 < occupied_fraction ∧ occupied_fraction ≤ 1

/-- The perimeter of a polygon that forms part of a square -/
def perimeter (p : PartialSquarePolygon) : ℝ :=
  4 * p.visible_side_length * p.occupied_fraction

/-- Theorem stating that a polygon occupying three-fourths of a square with visible sides of 5 units has a perimeter of 15 units -/
theorem specific_polygon_perimeter :
  ∀ (p : PartialSquarePolygon),
  p.visible_side_length = 5 →
  p.occupied_fraction = 3/4 →
  perimeter p = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_polygon_perimeter_l2233_223306


namespace NUMINAMATH_CALUDE_julia_tuesday_kids_l2233_223399

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 16

/-- The difference between the number of kids Julia played with on Monday and Tuesday -/
def difference : ℕ := 12

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tuesday_kids : tuesday_kids = 4 := by
  sorry

end NUMINAMATH_CALUDE_julia_tuesday_kids_l2233_223399


namespace NUMINAMATH_CALUDE_number_of_children_l2233_223398

theorem number_of_children (B C : ℕ) : 
  B = 2 * C →
  B = 4 * (C - 160) →
  C = 320 := by
sorry

end NUMINAMATH_CALUDE_number_of_children_l2233_223398


namespace NUMINAMATH_CALUDE_vector_collinearity_l2233_223314

theorem vector_collinearity (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, -2) → ∃ k : ℝ, b = k • a :=
by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2233_223314


namespace NUMINAMATH_CALUDE_smallest_n_for_prob_less_than_half_l2233_223377

def probability_red (n : ℕ) : ℚ :=
  9 / (11 - n)

theorem smallest_n_for_prob_less_than_half :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, 0 < k → k ≤ n → probability_red k < (1/2)) →
    n ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_prob_less_than_half_l2233_223377


namespace NUMINAMATH_CALUDE_average_rate_round_trip_l2233_223313

/-- Calculates the average rate of a round trip given the distance, running speed, and swimming speed. -/
theorem average_rate_round_trip 
  (distance : ℝ) 
  (running_speed : ℝ) 
  (swimming_speed : ℝ) 
  (h1 : distance = 4) 
  (h2 : running_speed = 10) 
  (h3 : swimming_speed = 6) : 
  (2 * distance) / (distance / running_speed + distance / swimming_speed) / 60 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_round_trip_l2233_223313


namespace NUMINAMATH_CALUDE_egg_production_increase_l2233_223361

/-- The number of eggs produced last year -/
def last_year_production : ℕ := 1416

/-- The number of eggs produced this year -/
def this_year_production : ℕ := 4636

/-- The increase in egg production -/
def production_increase : ℕ := this_year_production - last_year_production

theorem egg_production_increase :
  production_increase = 3220 := by sorry

end NUMINAMATH_CALUDE_egg_production_increase_l2233_223361


namespace NUMINAMATH_CALUDE_homework_difference_l2233_223375

def math_pages : ℕ := 5
def reading_pages : ℕ := 2

theorem homework_difference : math_pages - reading_pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_l2233_223375


namespace NUMINAMATH_CALUDE_sara_survey_sara_survey_result_l2233_223308

theorem sara_survey (total : ℕ) 
  (belief_rate : ℚ) 
  (zika_rate : ℚ) 
  (zika_believers : ℕ) : Prop :=
  belief_rate = 753/1000 →
  zika_rate = 602/1000 →
  zika_believers = 37 →
  ∃ (believers : ℕ),
    (believers : ℚ) = zika_believers / zika_rate ∧
    (total : ℚ) = (believers : ℚ) / belief_rate ∧
    total = 81

theorem sara_survey_result : 
  ∃ (total : ℕ), sara_survey total (753/1000) (602/1000) 37 :=
sorry

end NUMINAMATH_CALUDE_sara_survey_sara_survey_result_l2233_223308


namespace NUMINAMATH_CALUDE_expected_value_of_win_l2233_223342

def fair_8_sided_die := Finset.range 8

def win_amount (n : ℕ) : ℝ := 8 - n

theorem expected_value_of_win :
  Finset.sum fair_8_sided_die (λ n => (1 : ℝ) / 8 * win_amount n) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_win_l2233_223342


namespace NUMINAMATH_CALUDE_total_fish_bought_is_89_l2233_223312

/-- Represents the number of fish bought on each visit -/
structure FishPurchase where
  goldfish : Nat
  bluefish : Nat
  greenfish : Nat
  purplefish : Nat
  redfish : Nat

/-- Calculates the total number of fish in a purchase -/
def totalFish (purchase : FishPurchase) : Nat :=
  purchase.goldfish + purchase.bluefish + purchase.greenfish + purchase.purplefish + purchase.redfish

/-- Theorem: The total number of fish Roden bought is 89 -/
theorem total_fish_bought_is_89 
  (visit1 : FishPurchase := { goldfish := 15, bluefish := 7, greenfish := 0, purplefish := 0, redfish := 0 })
  (visit2 : FishPurchase := { goldfish := 10, bluefish := 12, greenfish := 5, purplefish := 0, redfish := 0 })
  (visit3 : FishPurchase := { goldfish := 3, bluefish := 7, greenfish := 9, purplefish := 0, redfish := 0 })
  (visit4 : FishPurchase := { goldfish := 4, bluefish := 8, greenfish := 6, purplefish := 2, redfish := 1 }) :
  totalFish visit1 + totalFish visit2 + totalFish visit3 + totalFish visit4 = 89 := by
  sorry


end NUMINAMATH_CALUDE_total_fish_bought_is_89_l2233_223312


namespace NUMINAMATH_CALUDE_football_team_yardage_l2233_223367

theorem football_team_yardage (initial_loss : ℤ) : 
  (initial_loss < 0) →  -- The team lost some yards initially
  (-initial_loss + 11 = 6) →  -- The team gained 11 yards and ended up with 6 yards progress
  initial_loss = -5 :=  -- The initial loss was 5 yards
by
  sorry

end NUMINAMATH_CALUDE_football_team_yardage_l2233_223367


namespace NUMINAMATH_CALUDE_fraction_problem_l2233_223347

theorem fraction_problem (F : ℝ) :
  (0.4 * F * 150 = 36) → F = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2233_223347


namespace NUMINAMATH_CALUDE_infinite_numbers_with_equal_digit_sum_l2233_223391

/-- Given a natural number, returns the sum of its digits in decimal representation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number contains the digit 0 in its decimal representation -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_numbers_with_equal_digit_sum (k : ℕ) :
  ∃ (T : Set ℕ), Set.Infinite T ∧ ∀ t ∈ T,
    ¬contains_zero t ∧ sum_of_digits t = sum_of_digits (k * t) := by
  sorry

end NUMINAMATH_CALUDE_infinite_numbers_with_equal_digit_sum_l2233_223391


namespace NUMINAMATH_CALUDE_simplify_fraction_l2233_223360

theorem simplify_fraction (a : ℝ) (h : a ≠ 3) :
  1 / (a - 3) - 6 / (a^2 - 9) = 1 / (a + 3) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2233_223360


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2233_223311

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 → Complex.im ((2 : ℂ) + i) / i = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2233_223311


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l2233_223325

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that ax^2 + bx + c = (px + q)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 4 → k = 4 ∨ k = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l2233_223325


namespace NUMINAMATH_CALUDE_complement_of_A_l2233_223372

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2233_223372


namespace NUMINAMATH_CALUDE_average_draw_is_n_plus_one_div_two_l2233_223315

/-- Represents a deck of cards -/
structure Deck :=
  (n : ℕ)  -- Total number of cards
  (ace_count : ℕ)  -- Number of aces in the deck
  (h1 : n > 0)  -- The deck has at least one card
  (h2 : ace_count = 3)  -- There are exactly three aces in the deck

/-- The average number of cards drawn until the second ace -/
def average_draw (d : Deck) : ℚ :=
  (d.n + 1) / 2

/-- Theorem stating that the average number of cards drawn until the second ace is (n + 1) / 2 -/
theorem average_draw_is_n_plus_one_div_two (d : Deck) :
  average_draw d = (d.n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_draw_is_n_plus_one_div_two_l2233_223315


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l2233_223351

theorem cube_sum_inequality (n : ℕ) : 
  (∀ a b c : ℕ, (a + b + c)^3 ≤ n * (a^3 + b^3 + c^3)) ↔ n ≥ 9 := by sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l2233_223351


namespace NUMINAMATH_CALUDE_book_cost_theorem_l2233_223301

/-- Proves that the total cost of books is 600 yuan given the problem conditions -/
theorem book_cost_theorem (total_children : ℕ) (paying_children : ℕ) (extra_payment : ℕ) :
  total_children = 12 →
  paying_children = 10 →
  extra_payment = 10 →
  (paying_children * extra_payment : ℕ) / (total_children - paying_children) * total_children = 600 :=
by
  sorry

#check book_cost_theorem

end NUMINAMATH_CALUDE_book_cost_theorem_l2233_223301


namespace NUMINAMATH_CALUDE_unique_distance_l2233_223334

/-- A two-digit number is represented as 10a + b where a and b are single digits -/
def two_digit_number (a b : ℕ) : Prop := 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- Inserting a zero between digits of a two-digit number -/
def insert_zero (a b : ℕ) : ℕ := 100 * a + b

/-- The property that inserting a zero results in 9 times the original number -/
def nine_times_property (a b : ℕ) : Prop :=
  insert_zero a b = 9 * (10 * a + b)

theorem unique_distance : 
  ∀ a b : ℕ, two_digit_number a b → nine_times_property a b → a = 4 ∧ b = 5 := by
  sorry

#check unique_distance

end NUMINAMATH_CALUDE_unique_distance_l2233_223334


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2233_223346

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2233_223346


namespace NUMINAMATH_CALUDE_intersection_points_count_l2233_223337

theorem intersection_points_count : ∃! (points : Finset (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ points ↔ (9*x^2 + 4*y^2 = 36 ∧ 4*x^2 + 9*y^2 = 36)) ∧
  points.card = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_points_count_l2233_223337


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l2233_223322

theorem polynomial_product_expansion :
  ∀ x : ℝ, (3*x^2 + 2*x + 1) * (2*x^2 + 3*x + 4) = 6*x^4 + 13*x^3 + 20*x^2 + 11*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l2233_223322


namespace NUMINAMATH_CALUDE_dale_max_nuts_l2233_223395

/-- The maximum number of nuts Dale can guarantee to get -/
def max_nuts_dale : ℕ := 71

/-- The total number of nuts -/
def total_nuts : ℕ := 1001

/-- The number of initial piles -/
def initial_piles : ℕ := 3

/-- The number of possible pile configurations -/
def pile_configs : ℕ := 8

theorem dale_max_nuts :
  ∀ (a b c : ℕ) (N : ℕ),
  a + b + c = total_nuts →
  1 ≤ N ∧ N ≤ total_nuts →
  (∃ (moved : ℕ), moved ≤ max_nuts_dale ∧
    (N = 0 ∨ N = a ∨ N = b ∨ N = c ∨ N = a + b ∨ N = b + c ∨ N = c + a ∨ N = total_nuts ∨
     (N < total_nuts ∧ moved = N - min N (min a (min b (min c (min (a + b) (min (b + c) (c + a))))))) ∨
     (N > 0 ∧ moved = min (a - N) (min (b - N) (min (c - N) (min (a + b - N) (min (b + c - N) (c + a - N)))))))) :=
by sorry

end NUMINAMATH_CALUDE_dale_max_nuts_l2233_223395


namespace NUMINAMATH_CALUDE_last_digit_322_369_l2233_223396

theorem last_digit_322_369 : (322^369) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_last_digit_322_369_l2233_223396


namespace NUMINAMATH_CALUDE_planes_divide_space_l2233_223389

/-- The number of regions into which n planes can divide space -/
def R (n : ℕ) : ℚ := (n^3 + 5*n + 6) / 6

/-- Theorem stating that R(n) gives the correct number of regions for n planes -/
theorem planes_divide_space (n : ℕ) : 
  R n = (n^3 + 5*n + 6) / 6 := by sorry

end NUMINAMATH_CALUDE_planes_divide_space_l2233_223389


namespace NUMINAMATH_CALUDE_park_area_change_l2233_223320

theorem park_area_change (original_area : ℝ) (length_decrease_percent : ℝ) (width_increase_percent : ℝ) :
  original_area = 600 →
  length_decrease_percent = 20 →
  width_increase_percent = 30 →
  let new_length_factor := 1 - length_decrease_percent / 100
  let new_width_factor := 1 + width_increase_percent / 100
  let new_area := original_area * new_length_factor * new_width_factor
  new_area = 624 := by sorry

end NUMINAMATH_CALUDE_park_area_change_l2233_223320


namespace NUMINAMATH_CALUDE_derivative_sqrt_l2233_223341

theorem derivative_sqrt (x : ℝ) (h : x > 0) :
  deriv (fun x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by sorry

end NUMINAMATH_CALUDE_derivative_sqrt_l2233_223341


namespace NUMINAMATH_CALUDE_certain_number_problem_l2233_223368

theorem certain_number_problem : ∃ x : ℝ, 11*x + 12*x + 15*x + 11 = 125 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2233_223368


namespace NUMINAMATH_CALUDE_angle_equivalence_same_quadrant_as_2016_l2233_223363

theorem angle_equivalence (θ : ℝ) : 
  θ ≡ (θ % 360) [PMOD 360] :=
sorry

theorem same_quadrant_as_2016 : 
  (2016 : ℝ) % 360 = 216 :=
sorry

end NUMINAMATH_CALUDE_angle_equivalence_same_quadrant_as_2016_l2233_223363


namespace NUMINAMATH_CALUDE_alphabet_size_l2233_223354

theorem alphabet_size (dot_and_line : ℕ) (line_no_dot : ℕ) (dot_no_line : ℕ)
  (h1 : dot_and_line = 20)
  (h2 : line_no_dot = 36)
  (h3 : dot_no_line = 4)
  : dot_and_line + line_no_dot + dot_no_line = 60 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_size_l2233_223354


namespace NUMINAMATH_CALUDE_power_sum_equality_l2233_223364

theorem power_sum_equality : 2^123 + 8^5 / 8^3 = 2^123 + 64 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2233_223364


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_two_l2233_223382

theorem sqrt_expression_equals_two :
  Real.sqrt 4 + Real.sqrt 2 * Real.sqrt 6 - 6 * Real.sqrt (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_two_l2233_223382


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2233_223331

theorem sum_of_coefficients (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 4) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2233_223331


namespace NUMINAMATH_CALUDE_rectangle_circle_square_area_l2233_223338

theorem rectangle_circle_square_area : 
  ∀ (r : ℝ) (l w : ℝ),
    r = 7 →  -- Circle radius
    l = 3 * w →  -- Rectangle length to width ratio
    2 * r = w →  -- Circle diameter equals rectangle width
    l * w + 2 * r^2 = 686 :=  -- Total area of rectangle and square
by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_square_area_l2233_223338


namespace NUMINAMATH_CALUDE_leila_order_proof_l2233_223362

/-- The number of chocolate cakes Leila ordered -/
def chocolate_cakes : ℕ := 3

/-- The cost of each chocolate cake -/
def chocolate_cake_cost : ℕ := 12

/-- The number of strawberry cakes Leila ordered -/
def strawberry_cakes : ℕ := 6

/-- The cost of each strawberry cake -/
def strawberry_cake_cost : ℕ := 22

/-- The total amount Leila should pay -/
def total_amount : ℕ := 168

theorem leila_order_proof :
  chocolate_cakes * chocolate_cake_cost + 
  strawberry_cakes * strawberry_cake_cost = total_amount :=
by sorry

end NUMINAMATH_CALUDE_leila_order_proof_l2233_223362


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l2233_223302

/-- Given an arithmetic sequence of 25 terms with first term 7 and last term 98,
    prove that the 8th term is equal to 343/12. -/
theorem arithmetic_sequence_8th_term :
  ∀ (a : ℕ → ℚ),
    (∀ i j, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
    (a 0 = 7) →                                   -- first term is 7
    (a 24 = 98) →                                 -- last term is 98
    (a 7 = 343 / 12) :=                           -- 8th term (index 7) is 343/12
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l2233_223302


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2233_223348

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h
  , c := p.a * h^2 + k }

theorem parabola_shift_theorem (original : Parabola) (h k : ℝ) :
  original.a = 3 ∧ original.b = 0 ∧ original.c = 0 ∧ h = 1 ∧ k = 2 →
  let shifted := shift_parabola original h k
  shifted.a = 3 ∧ shifted.b = -6 ∧ shifted.c = 5 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2233_223348


namespace NUMINAMATH_CALUDE_sine_squared_equality_l2233_223357

theorem sine_squared_equality (α β : ℝ) 
  (h : (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = 1) :
  (Real.sin α)^2 = (Real.sin β)^2 := by
  sorry

end NUMINAMATH_CALUDE_sine_squared_equality_l2233_223357


namespace NUMINAMATH_CALUDE_distance_to_big_rock_is_4_l2233_223374

/-- Represents the distance to Big Rock in kilometers -/
def distance_to_big_rock : ℝ := sorry

/-- Rower's speed in still water in km/h -/
def rower_speed : ℝ := 6

/-- Current speed to Big Rock in km/h -/
def current_speed_to : ℝ := 2

/-- Current speed from Big Rock in km/h -/
def current_speed_from : ℝ := 3

/-- Rower's speed from Big Rock in km/h -/
def rower_speed_back : ℝ := 7

/-- Total round trip time in hours -/
def total_time : ℝ := 2

theorem distance_to_big_rock_is_4 :
  distance_to_big_rock = 4 ∧
  (distance_to_big_rock / (rower_speed - current_speed_to) +
   distance_to_big_rock / (rower_speed_back - current_speed_from) = total_time) :=
sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_is_4_l2233_223374


namespace NUMINAMATH_CALUDE_power_of_power_l2233_223355

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2233_223355


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2233_223316

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2233_223316


namespace NUMINAMATH_CALUDE_courtyard_ratio_l2233_223388

/-- Given a courtyard with trees, stones, and birds, prove the ratio of trees to stones -/
theorem courtyard_ratio (stones birds : ℕ) (h1 : stones = 40) (h2 : birds = 400)
  (h3 : birds = 2 * (trees + stones)) : (trees : ℚ) / stones = 4 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_courtyard_ratio_l2233_223388


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l2233_223303

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def digit_to_nat (d : ℕ) : ℕ :=
  526000 + d * 100 + 84

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ is_divisible_by_3 (digit_to_nat d) ∧
  ∀ (d' : ℕ), d' < d → ¬is_divisible_by_3 (digit_to_nat d') :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l2233_223303


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l2233_223321

theorem unique_quadratic_root (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → (m = 0 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l2233_223321


namespace NUMINAMATH_CALUDE_dog_count_l2233_223345

theorem dog_count (num_puppies : ℕ) (dog_meal_frequency : ℕ) (dog_meal_amount : ℕ) (total_food : ℕ) : 
  num_puppies = 4 →
  dog_meal_frequency = 3 →
  dog_meal_amount = 4 →
  total_food = 108 →
  (∃ (num_dogs : ℕ),
    num_dogs * (dog_meal_frequency * dog_meal_amount) + 
    num_puppies * (3 * dog_meal_frequency) * (dog_meal_amount / 2) = total_food ∧
    num_dogs = 3) := by
  sorry

end NUMINAMATH_CALUDE_dog_count_l2233_223345


namespace NUMINAMATH_CALUDE_problem_solution_l2233_223318

theorem problem_solution :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (max_value : ℝ), (a + 3*b + 3/a + 4/b = 18) → max_value = 9 + 3*Real.sqrt 6 ∧ a + 3*b ≤ max_value) ∧
  (a > b → ∃ (min_value : ℝ), min_value = 32 ∧ a^2 + 64 / (b*(a-b)) ≥ min_value) ∧
  (∃ (min_value : ℝ), (1/(a+1) + 1/(b+2) = 1/3) → min_value = 14 + 6*Real.sqrt 6 ∧ a*b + a + b ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2233_223318


namespace NUMINAMATH_CALUDE_tree_structure_equation_l2233_223332

/-- Represents the structure of a tree with branches and small branches. -/
structure TreeStructure where
  branches : ℕ
  total_count : ℕ

/-- The equation for the tree structure is correct if it satisfies the given conditions. -/
def is_correct_equation (t : TreeStructure) : Prop :=
  1 + t.branches + t.branches^2 = t.total_count

/-- Theorem stating that the equation correctly represents the tree structure. -/
theorem tree_structure_equation (t : TreeStructure) 
  (h : t.total_count = 57) : is_correct_equation t := by
  sorry

end NUMINAMATH_CALUDE_tree_structure_equation_l2233_223332


namespace NUMINAMATH_CALUDE_possible_k_values_l2233_223329

def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (k : ℝ) : Set ℝ := {x | k*x + 1 = 0}

theorem possible_k_values :
  ∀ k : ℝ, (N k ⊆ M) ↔ (k = 0 ∨ k = -1/2 ∨ k = 1/3) := by sorry

end NUMINAMATH_CALUDE_possible_k_values_l2233_223329


namespace NUMINAMATH_CALUDE_cubic_equation_root_a_value_l2233_223336

theorem cubic_equation_root_a_value :
  ∀ a b : ℚ,
  (∃ x : ℝ, x^3 + a*x^2 + b*x - 48 = 0 ∧ x = 2 - 5*Real.sqrt 3) →
  a = -332/71 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_a_value_l2233_223336


namespace NUMINAMATH_CALUDE_g_neg_one_equals_neg_one_l2233_223330

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Theorem statement
theorem g_neg_one_equals_neg_one
  (h1 : is_odd_function f)
  (h2 : f 1 = 1) :
  g f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_one_equals_neg_one_l2233_223330


namespace NUMINAMATH_CALUDE_transport_cost_calculation_l2233_223317

/-- The problem statement for calculating transport cost --/
theorem transport_cost_calculation (purchase_price installation_cost sell_price : ℚ) : 
  purchase_price = 16500 →
  installation_cost = 250 →
  sell_price = 23100 →
  ∃ (labelled_price transport_cost : ℚ),
    purchase_price = labelled_price * (1 - 0.2) ∧
    sell_price = labelled_price * 1.1 + transport_cost + installation_cost ∧
    transport_cost = 162.5 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_calculation_l2233_223317


namespace NUMINAMATH_CALUDE_game_winnable_iff_game_not_winnable_equal_game_winnable_greater_l2233_223376

/-- Represents a winning strategy for the card game -/
structure WinningStrategy (n k : ℕ) :=
  (moves : ℕ)
  (strategy : Unit)  -- Placeholder for the actual strategy

/-- The existence of a winning strategy for the card game -/
def winnable (n k : ℕ) : Prop :=
  ∃ (s : WinningStrategy n k), true

/-- Main theorem: The game is winnable if and only if n > k, given n ≥ k ≥ 2 -/
theorem game_winnable_iff (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  winnable n k ↔ n > k :=
sorry

/-- The game is not winnable when n = k -/
theorem game_not_winnable_equal (n : ℕ) (h : n ≥ 2) :
  ¬ winnable n n :=
sorry

/-- The game is winnable when n > k -/
theorem game_winnable_greater (n k : ℕ) (h1 : n > k) (h2 : k ≥ 2) :
  winnable n k :=
sorry

end NUMINAMATH_CALUDE_game_winnable_iff_game_not_winnable_equal_game_winnable_greater_l2233_223376


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2233_223344

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → 
  (b^3 - b + 1 = 0) → 
  (c^3 - c + 1 = 0) → 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2233_223344


namespace NUMINAMATH_CALUDE_min_a_for_nonnegative_f_l2233_223387

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (-3 * x^2 + a * x) - a / x

theorem min_a_for_nonnegative_f :
  ∀ a : ℝ, a > 0 →
  (∃ x₀ : ℝ, f a x₀ ≥ 0) →
  a ≥ 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_nonnegative_f_l2233_223387


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l2233_223369

/-- Given a rhombus with one diagonal of length 70 meters and an area of 5600 square meters,
    the other diagonal has a length of 160 meters. -/
theorem rhombus_other_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 70 → area = 5600 → area = (d1 * d2) / 2 → d2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l2233_223369


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l2233_223349

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (x - 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ = 16 ∧ a₂ = 24 ∧ a₁ + a₂ + a₃ + a₄ = -15) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l2233_223349


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_221_l2233_223394

theorem greatest_prime_factor_of_221 : ∃ p : ℕ, p.Prime ∧ p ∣ 221 ∧ ∀ q : ℕ, q.Prime → q ∣ 221 → q ≤ p ∧ p = 17 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_221_l2233_223394


namespace NUMINAMATH_CALUDE_max_value_in_equation_max_value_achievable_l2233_223333

/-- Represents a three-digit number composed of different non-zero digits from 1 to 9 -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9 ∧ n = 100 * x + 10 * y + z) }

/-- The main theorem stating the maximum value of a in the given equation -/
theorem max_value_in_equation (a b c d : ThreeDigitNumber) 
  (h : 1984 - a.val = 2015 - b.val - c.val - d.val) : 
  a.val ≤ 214 := by
  sorry

/-- The theorem proving that 214 is achievable -/
theorem max_value_achievable : 
  ∃ (a b c d : ThreeDigitNumber), 1984 - a.val = 2015 - b.val - c.val - d.val ∧ a.val = 214 := by
  sorry

end NUMINAMATH_CALUDE_max_value_in_equation_max_value_achievable_l2233_223333


namespace NUMINAMATH_CALUDE_product_of_primes_l2233_223327

theorem product_of_primes : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_l2233_223327


namespace NUMINAMATH_CALUDE_cost_of_450_candies_l2233_223379

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box : ℚ) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $112.50, given that a box of 30 costs $7.50 -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = (112.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_cost_of_450_candies_l2233_223379


namespace NUMINAMATH_CALUDE_series_sum_l2233_223324

def series_term (n : ℕ) : ℚ := (2^n : ℚ) / ((3^(3^n) : ℚ) + 1)

theorem series_sum : ∑' n, series_term n = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_l2233_223324


namespace NUMINAMATH_CALUDE_mushroom_trip_theorem_l2233_223328

def mushroom_trip_earnings (day1_earnings day2_price day3_price day4_price day5_price day6_price day7_price : ℝ)
  (day2_mushrooms : ℕ) (day3_increase day4_increase day5_mushrooms day6_decrease day7_mushrooms : ℝ)
  (expenses : ℝ) : Prop :=
  let day2_earnings := day2_mushrooms * day2_price
  let day3_mushrooms := day2_mushrooms + day3_increase
  let day3_earnings := day3_mushrooms * day3_price
  let day4_mushrooms := day3_mushrooms * (1 + day4_increase)
  let day4_earnings := day4_mushrooms * day4_price
  let day5_earnings := day5_mushrooms * day5_price
  let day6_mushrooms := day5_mushrooms * (1 - day6_decrease)
  let day6_earnings := day6_mushrooms * day6_price
  let day7_earnings := day7_mushrooms * day7_price
  let total_earnings := day1_earnings + day2_earnings + day3_earnings + day4_earnings + day5_earnings + day6_earnings + day7_earnings
  total_earnings - expenses = 703.40

theorem mushroom_trip_theorem : 
  mushroom_trip_earnings 120 2.50 1.75 1.30 2.00 2.50 1.80 20 18 0.40 72 0.25 80 25 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_trip_theorem_l2233_223328


namespace NUMINAMATH_CALUDE_expression_equals_one_eighth_l2233_223380

theorem expression_equals_one_eighth :
  let a := 404445
  let b := 202222
  let c := 202223
  let d := 202224
  let e := 12639
  (a^2 / (b * c * d) - c / (b * d) - b / (c * d)) * e = 1/8 := by sorry

end NUMINAMATH_CALUDE_expression_equals_one_eighth_l2233_223380


namespace NUMINAMATH_CALUDE_exists_triangle_101_subdivisions_l2233_223392

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define a function to check if a triangle can be subdivided into n congruent triangles
def can_subdivide (t : Triangle) (n : ℕ) : Prop :=
  ∃ (m : ℕ), m^2 + 1 = n

-- Theorem statement
theorem exists_triangle_101_subdivisions :
  ∃ (t : Triangle), can_subdivide t 101 := by
sorry

end NUMINAMATH_CALUDE_exists_triangle_101_subdivisions_l2233_223392


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l2233_223335

theorem rectangle_area_ratio (large_horizontal small_horizontal large_vertical small_vertical large_area : ℝ)
  (h_horizontal_ratio : large_horizontal / small_horizontal = 8 / 7)
  (h_vertical_ratio : large_vertical / small_vertical = 9 / 4)
  (h_large_area : large_horizontal * large_vertical = large_area)
  (h_large_area_value : large_area = 108) :
  small_horizontal * small_vertical = 42 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l2233_223335


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2233_223304

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 2 → x^2 > 4) ∧ 
  (∃ x, x^2 > 4 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2233_223304


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2233_223343

-- Define the equation of the parabolas
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 3) = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 4)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices :
  parabola_equation vertex1.1 vertex1.2 ∧
  parabola_equation vertex2.1 vertex2.2 →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 5 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_vertices_l2233_223343


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2233_223340

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ  -- coefficient of x

def is_on_parabola (p : Point) (par : Parabola) : Prop :=
  p.y^2 = 4 * par.a * p.x

def is_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

def is_focus (f : Point) (par : Parabola) : Prop :=
  f.x = par.a ∧ f.y = 0

def is_on_circle_diameter (a : Point) (p : Point) (q : Point) : Prop :=
  (p.x - a.x) * (q.x - a.x) + (p.y - a.y) * (q.y - a.y) = 0

theorem parabola_line_intersection 
  (par : Parabola) (l : Line) (f p q : Point) (h_focus : is_focus f par)
  (h_line_through_focus : is_on_line f l)
  (h_p_on_parabola : is_on_parabola p par) (h_p_on_line : is_on_line p l)
  (h_q_on_parabola : is_on_parabola q par) (h_q_on_line : is_on_line q l)
  (h_circle : is_on_circle_diameter ⟨-1, 1⟩ p q) :
  l.m = 1/2 ∧ l.b = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2233_223340


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2233_223323

theorem polynomial_factorization (x : ℝ) : 
  x^9 - 6*x^6 + 12*x^3 - 8 = (x^3 - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2233_223323
