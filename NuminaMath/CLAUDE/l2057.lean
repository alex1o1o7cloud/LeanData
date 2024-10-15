import Mathlib

namespace NUMINAMATH_CALUDE_pizza_size_increase_l2057_205783

theorem pizza_size_increase (r : ℝ) (h : r > 0) :
  let R := r * (1 + 0.5)
  (π * R^2) / (π * r^2) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_pizza_size_increase_l2057_205783


namespace NUMINAMATH_CALUDE_will_baseball_cards_pages_l2057_205776

/-- The number of pages needed to organize baseball cards in a binder -/
def pages_needed (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards) / cards_per_page

/-- Theorem: Will uses 6 pages to organize his baseball cards -/
theorem will_baseball_cards_pages : pages_needed 3 8 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_will_baseball_cards_pages_l2057_205776


namespace NUMINAMATH_CALUDE_solution_value_l2057_205794

theorem solution_value (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ (m / (2 - x)) - (1 / (x - 2)) = 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2057_205794


namespace NUMINAMATH_CALUDE_sodium_bicarbonate_moles_l2057_205711

-- Define the chemical reaction
structure Reaction where
  hcl : ℝ
  nahco3 : ℝ
  nacl : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.hcl = r.nahco3 ∧ r.hcl = r.nacl

-- Theorem statement
theorem sodium_bicarbonate_moles 
  (r : Reaction) 
  (h1 : r.hcl = 1) 
  (h2 : r.nacl = 1) 
  (h3 : balanced_equation r) : 
  r.nahco3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sodium_bicarbonate_moles_l2057_205711


namespace NUMINAMATH_CALUDE_factor_polynomial_l2057_205741

theorem factor_polynomial (x : ℝ) : 
  36 * x^6 - 189 * x^12 + 81 * x^9 = 9 * x^6 * (4 + 9 * x^3 - 21 * x^6) := by sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2057_205741


namespace NUMINAMATH_CALUDE_final_price_is_correct_l2057_205779

-- Define the initial price, discounts, and conversion rate
def initial_price : ℝ := 150
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.05
def usd_to_inr : ℝ := 75

-- Define the function to calculate the final price
def final_price : ℝ :=
  let price1 := initial_price * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * usd_to_inr

-- Theorem statement
theorem final_price_is_correct : final_price = 7267.5 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_correct_l2057_205779


namespace NUMINAMATH_CALUDE_digit_difference_of_63_l2057_205774

theorem digit_difference_of_63 :
  let tens : ℕ := 63 / 10
  let ones : ℕ := 63 % 10
  tens + ones = 9 →
  tens - ones = 3 :=
by sorry

end NUMINAMATH_CALUDE_digit_difference_of_63_l2057_205774


namespace NUMINAMATH_CALUDE_complex_magnitude_l2057_205723

theorem complex_magnitude (z : ℂ) (h : z / (1 - Complex.I)^2 = (1 + Complex.I) / 2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2057_205723


namespace NUMINAMATH_CALUDE_fifteen_times_fifteen_l2057_205799

theorem fifteen_times_fifteen : 
  ∀ n : ℕ, n = 15 → 15 * n = 225 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_times_fifteen_l2057_205799


namespace NUMINAMATH_CALUDE_charlies_metal_storage_l2057_205706

/-- The amount of metal Charlie has in storage -/
def metal_in_storage (total_needed : ℕ) (to_buy : ℕ) : ℕ :=
  total_needed - to_buy

/-- Theorem: Charlie's metal in storage is the difference between total needed and amount to buy -/
theorem charlies_metal_storage :
  metal_in_storage 635 359 = 276 := by
  sorry

end NUMINAMATH_CALUDE_charlies_metal_storage_l2057_205706


namespace NUMINAMATH_CALUDE_square_root_of_1_5625_l2057_205718

theorem square_root_of_1_5625 : Real.sqrt 1.5625 = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1_5625_l2057_205718


namespace NUMINAMATH_CALUDE_sugar_bag_weight_l2057_205705

/-- The weight of a bag of sugar, given the weight of a bag of salt and their combined weight after removing 4 kg. -/
theorem sugar_bag_weight (salt_weight : ℝ) (combined_weight_minus_four : ℝ) 
  (h1 : salt_weight = 30)
  (h2 : combined_weight_minus_four = 42)
  (h3 : combined_weight_minus_four = salt_weight + sugar_weight - 4) :
  sugar_weight = 16 := by
  sorry

#check sugar_bag_weight

end NUMINAMATH_CALUDE_sugar_bag_weight_l2057_205705


namespace NUMINAMATH_CALUDE_probability_above_parabola_l2057_205773

/-- The type of single-digit positive integers -/
def SingleDigitPos := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- The condition for a point (a,b) to be above the parabola y = ax^2 + bx for all x -/
def IsAboveParabola (a b : SingleDigitPos) : Prop :=
  ∀ x : ℝ, (b : ℝ) > (a : ℝ) * x^2 + (b : ℝ) * x

/-- The number of valid (a,b) pairs -/
def NumValidPairs : ℕ := 72

/-- The total number of possible (a,b) pairs -/
def TotalPairs : ℕ := 81

/-- The main theorem: the probability of (a,b) being above the parabola is 8/9 -/
theorem probability_above_parabola :
  (NumValidPairs : ℚ) / (TotalPairs : ℚ) = 8 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_above_parabola_l2057_205773


namespace NUMINAMATH_CALUDE_max_xy_value_l2057_205709

theorem max_xy_value (x y : ℕ) (h : 69 * x + 54 * y ≤ 2008) : x * y ≤ 270 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2057_205709


namespace NUMINAMATH_CALUDE_add_specific_reals_l2057_205755

theorem add_specific_reals : 1.25 + 47.863 = 49.113 := by
  sorry

end NUMINAMATH_CALUDE_add_specific_reals_l2057_205755


namespace NUMINAMATH_CALUDE_automobile_distance_l2057_205729

/-- Proves that an automobile traveling a/4 feet in 2r seconds will travel 25a/r yards in 10 minutes -/
theorem automobile_distance (a r : ℝ) (h : r ≠ 0) : 
  let rate_feet_per_second := a / (4 * 2 * r)
  let rate_yards_per_second := rate_feet_per_second / 3
  let time_seconds := 10 * 60
  rate_yards_per_second * time_seconds = 25 * a / r := by sorry

end NUMINAMATH_CALUDE_automobile_distance_l2057_205729


namespace NUMINAMATH_CALUDE_total_distance_is_15_l2057_205728

def morning_ride : ℕ := 2

def evening_ride (m : ℕ) : ℕ := 5 * m

def third_ride (m : ℕ) : ℕ := 2 * m - 1

def total_distance (m : ℕ) : ℕ := m + evening_ride m + third_ride m

theorem total_distance_is_15 : total_distance morning_ride = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_15_l2057_205728


namespace NUMINAMATH_CALUDE_road_repair_theorem_l2057_205770

/-- Represents the road repair scenario -/
structure RoadRepair where
  initial_workers : ℕ
  initial_days : ℕ
  worked_days : ℕ
  additional_workers : ℕ

/-- Calculates the number of additional days needed to complete the work -/
def additional_days_needed (repair : RoadRepair) : ℚ :=
  let total_work := repair.initial_workers * repair.initial_days
  let work_done := repair.initial_workers * repair.worked_days
  let remaining_work := total_work - work_done
  let new_workforce := repair.initial_workers + repair.additional_workers
  remaining_work / new_workforce

/-- The theorem stating that 6 additional days are needed to complete the work -/
theorem road_repair_theorem (repair : RoadRepair)
  (h1 : repair.initial_workers = 24)
  (h2 : repair.initial_days = 12)
  (h3 : repair.worked_days = 4)
  (h4 : repair.additional_workers = 8) :
  additional_days_needed repair = 6 := by
  sorry

#eval additional_days_needed ⟨24, 12, 4, 8⟩

end NUMINAMATH_CALUDE_road_repair_theorem_l2057_205770


namespace NUMINAMATH_CALUDE_shoe_selection_outcomes_l2057_205726

/-- The number of distinct pairs of shoes -/
def num_pairs : ℕ := 10

/-- The number of shoes drawn -/
def num_drawn : ℕ := 4

/-- The number of ways to select 4 shoes such that none form a pair -/
def no_pairs : ℕ := (Nat.choose num_pairs num_drawn) * (2^num_drawn)

/-- The number of ways to select 4 shoes such that two form a pair and the other two do not form pairs -/
def one_pair : ℕ := (Nat.choose num_pairs 2) * (2^2) * (Nat.choose (num_pairs - 2) 1)

/-- The number of ways to select 4 shoes such that they form two complete pairs -/
def two_pairs : ℕ := Nat.choose num_pairs 2

theorem shoe_selection_outcomes :
  no_pairs = 3360 ∧ one_pair = 1440 ∧ two_pairs = 45 := by
  sorry

end NUMINAMATH_CALUDE_shoe_selection_outcomes_l2057_205726


namespace NUMINAMATH_CALUDE_negation_of_positive_product_l2057_205786

theorem negation_of_positive_product (x y : ℝ) :
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ (x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_positive_product_l2057_205786


namespace NUMINAMATH_CALUDE_trapezoid_max_segment_length_l2057_205797

/-- Given a trapezoid with sum of bases equal to 4, the maximum length of a segment
    passing through the intersection of diagonals and parallel to bases is 2. -/
theorem trapezoid_max_segment_length (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (s : ℝ), s ≤ 2 ∧ 
  ∀ (t : ℝ), (∃ (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4),
    t = (2 * x * y) / (x + y)) → t ≤ s :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_max_segment_length_l2057_205797


namespace NUMINAMATH_CALUDE_downstream_distance_l2057_205701

/-- Proves that the distance traveled downstream is 420 km given the conditions -/
theorem downstream_distance
  (downstream_time : ℝ)
  (upstream_speed : ℝ)
  (total_speed : ℝ)
  (h1 : downstream_time = 20)
  (h2 : upstream_speed = 12)
  (h3 : total_speed = 21) :
  downstream_time * total_speed = 420 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_l2057_205701


namespace NUMINAMATH_CALUDE_correct_answers_for_zero_score_l2057_205714

theorem correct_answers_for_zero_score (total_questions : ℕ) 
  (points_per_correct : ℕ) (points_per_wrong : ℕ) : 
  total_questions = 26 →
  points_per_correct = 8 →
  points_per_wrong = 5 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    points_per_correct * correct_answers = 
      points_per_wrong * (total_questions - correct_answers) ∧
    correct_answers = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_zero_score_l2057_205714


namespace NUMINAMATH_CALUDE_min_calls_for_gossip_l2057_205772

theorem min_calls_for_gossip (n : ℕ) (h : n > 0) : ℕ :=
  2 * (n - 1)

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_min_calls_for_gossip_l2057_205772


namespace NUMINAMATH_CALUDE_video_game_spending_l2057_205752

/-- The total amount spent on video games is the sum of the costs of individual games -/
theorem video_game_spending (basketball_cost racing_cost : ℚ) :
  basketball_cost = 5.2 →
  racing_cost = 4.23 →
  basketball_cost + racing_cost = 9.43 := by
  sorry

end NUMINAMATH_CALUDE_video_game_spending_l2057_205752


namespace NUMINAMATH_CALUDE_unique_multiplication_property_l2057_205724

theorem unique_multiplication_property : ∃! n : ℕ, 
  (n ≥ 10000000 ∧ n < 100000000) ∧  -- 8-digit number
  (n % 10 = 9) ∧                    -- ends in 9
  (∃ k : ℕ, n * 9 = k * 111111111)  -- when multiplied by 9, equals k * 111111111
    := by sorry

end NUMINAMATH_CALUDE_unique_multiplication_property_l2057_205724


namespace NUMINAMATH_CALUDE_josh_candy_count_l2057_205708

def candy_problem (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (shared_candies : ℕ) : ℕ :=
  let remaining_after_siblings := initial_candies - siblings * candies_per_sibling
  let remaining_after_friend := remaining_after_siblings / 2
  remaining_after_friend - shared_candies

theorem josh_candy_count : candy_problem 100 3 10 19 = 16 := by
  sorry

end NUMINAMATH_CALUDE_josh_candy_count_l2057_205708


namespace NUMINAMATH_CALUDE_cross_number_intersection_l2057_205792

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def power_of_three (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

def power_of_seven (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

theorem cross_number_intersection :
  ∃! d : ℕ,
    d < 10 ∧
    ∃ (n m : ℕ),
      is_three_digit n ∧
      is_three_digit m ∧
      power_of_three n ∧
      power_of_seven m ∧
      n % 10 = d ∧
      (m / 100) % 10 = d :=
sorry

end NUMINAMATH_CALUDE_cross_number_intersection_l2057_205792


namespace NUMINAMATH_CALUDE_bisecting_plane_intersects_sixteen_cubes_l2057_205790

/-- Represents a cube composed of unit cubes -/
structure UnitCube where
  side_length : ℕ

/-- Represents a plane that bisects a face diagonal of a cube -/
structure BisectingPlane where
  cube : UnitCube

/-- Counts the number of unit cubes intersected by a bisecting plane -/
def count_intersected_cubes (plane : BisectingPlane) : ℕ :=
  sorry

/-- Theorem stating that a plane bisecting a face diagonal of a 4x4x4 cube intersects 16 unit cubes -/
theorem bisecting_plane_intersects_sixteen_cubes 
  (cube : UnitCube) 
  (plane : BisectingPlane) 
  (h1 : cube.side_length = 4) 
  (h2 : plane.cube = cube) : 
  count_intersected_cubes plane = 16 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_plane_intersects_sixteen_cubes_l2057_205790


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l2057_205782

/-- A circle with center on y = b intersects y = (4/3)x^2 at least thrice, including the origin --/
def CircleIntersectsParabola (b : ℝ) : Prop :=
  ∃ (r : ℝ) (a : ℝ), (a^2 + b^2 = r^2) ∧ 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    ((x₁^2 + ((4/3)*x₁^2 - b)^2 = r^2) ∧
     (x₂^2 + ((4/3)*x₂^2 - b)^2 = r^2)))

/-- Two non-origin intersection points lie on y = (4/3)x + b --/
def IntersectionPointsOnLine (b : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    ((4/3)*x₁^2 = (4/3)*x₁ + b) ∧
    ((4/3)*x₂^2 = (4/3)*x₂ + b)

/-- The theorem to be proved --/
theorem circle_parabola_intersection (b : ℝ) :
  (CircleIntersectsParabola b ∧ IntersectionPointsOnLine b) ↔ b = 25/12 :=
sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l2057_205782


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2057_205716

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 1 / (x - 3) + 1 / (x - 5) - 4 / (x - 4)
  ∃ (sol : ℝ), f sol = 0 ∧ sol = 4 - Real.sqrt 2 ∧ ∀ x, f x = 0 → x ≥ sol :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2057_205716


namespace NUMINAMATH_CALUDE_tan_2theta_minus_pi_6_l2057_205715

theorem tan_2theta_minus_pi_6 (θ : ℝ) 
  (h : 4 * Real.cos (θ + π/3) * Real.cos (θ - π/6) = Real.sin (2*θ)) : 
  Real.tan (2*θ - π/6) = Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_2theta_minus_pi_6_l2057_205715


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2057_205700

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2057_205700


namespace NUMINAMATH_CALUDE_soap_box_width_l2057_205710

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the carton and soap box dimensions, and the maximum number of soap boxes,
    prove that the width of each soap box is 7 inches -/
theorem soap_box_width
  (carton : BoxDimensions)
  (soap_box : BoxDimensions)
  (max_soap_boxes : ℕ)
  (h1 : carton.length = 25)
  (h2 : carton.width = 42)
  (h3 : carton.height = 60)
  (h4 : soap_box.length = 6)
  (h5 : soap_box.height = 6)
  (h6 : max_soap_boxes = 250)
  (h7 : max_soap_boxes * boxVolume soap_box = boxVolume carton) :
  soap_box.width = 7 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_width_l2057_205710


namespace NUMINAMATH_CALUDE_diamond_computation_l2057_205757

-- Define the ⋄ operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_computation :
  (diamond (diamond 4 5) 6) - (diamond 4 (diamond 5 6)) = -139 / 870 := by
  sorry

end NUMINAMATH_CALUDE_diamond_computation_l2057_205757


namespace NUMINAMATH_CALUDE_cows_not_black_l2057_205739

theorem cows_not_black (total : ℕ) (black : ℕ) : total = 18 → black = (total / 2 + 5) → total - black = 4 := by
  sorry

end NUMINAMATH_CALUDE_cows_not_black_l2057_205739


namespace NUMINAMATH_CALUDE_monkey_climb_distance_l2057_205751

/-- Represents the climbing behavior of a monkey -/
structure MonkeyClimb where
  climb_distance : ℝ  -- Distance the monkey climbs in one minute
  slip_distance : ℝ   -- Distance the monkey slips in the next minute
  total_time : ℕ      -- Total time taken to reach the top
  total_height : ℝ    -- Total height reached

/-- Theorem stating that given the monkey's climbing behavior, 
    if it takes 37 minutes to reach 60 meters, then it climbs 6 meters per minute -/
theorem monkey_climb_distance 
  (m : MonkeyClimb) 
  (h1 : m.slip_distance = 3) 
  (h2 : m.total_time = 37) 
  (h3 : m.total_height = 60) : 
  m.climb_distance = 6 := by
  sorry

#check monkey_climb_distance

end NUMINAMATH_CALUDE_monkey_climb_distance_l2057_205751


namespace NUMINAMATH_CALUDE_julie_work_hours_julie_school_year_hours_l2057_205750

/-- Given Julie's work schedule and earnings, calculate her required weekly hours during the school year --/
theorem julie_work_hours (summer_weekly_hours : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℕ :=
  let hourly_rate := summer_earnings / (summer_weekly_hours * summer_weeks)
  let school_year_hours := school_year_target / hourly_rate
  let school_year_weekly_hours := school_year_hours / school_year_weeks
  school_year_weekly_hours

/-- Prove that Julie needs to work 15 hours per week during the school year --/
theorem julie_school_year_hours : 
  julie_work_hours 60 10 8000 50 10000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_julie_work_hours_julie_school_year_hours_l2057_205750


namespace NUMINAMATH_CALUDE_root_one_implies_m_three_l2057_205753

theorem root_one_implies_m_three (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 2 = 0 ∧ x = 1) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_one_implies_m_three_l2057_205753


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l2057_205780

theorem quadratic_equation_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ - m = 0 ∧ x₂^2 + 4*x₂ - m = 0) ↔ m > -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l2057_205780


namespace NUMINAMATH_CALUDE_only_pairC_not_opposite_l2057_205735

-- Define a type for quantities
inductive Quantity
| WinGames (n : ℕ)
| LoseGames (n : ℕ)
| RotateCounterclockwise (n : ℕ)
| RotateClockwise (n : ℕ)
| ReceiveMoney (amount : ℕ)
| IncreaseMoney (amount : ℕ)
| TemperatureRise (degrees : ℕ)
| TemperatureDecrease (degrees : ℕ)

-- Define a function to check if two quantities have opposite meanings
def haveOppositeMeanings (q1 q2 : Quantity) : Prop :=
  match q1, q2 with
  | Quantity.WinGames n, Quantity.LoseGames m => true
  | Quantity.RotateCounterclockwise n, Quantity.RotateClockwise m => true
  | Quantity.ReceiveMoney n, Quantity.IncreaseMoney m => false
  | Quantity.TemperatureRise n, Quantity.TemperatureDecrease m => true
  | _, _ => false

-- Define the pairs of quantities
def pairA := (Quantity.WinGames 3, Quantity.LoseGames 3)
def pairB := (Quantity.RotateCounterclockwise 3, Quantity.RotateClockwise 5)
def pairC := (Quantity.ReceiveMoney 3000, Quantity.IncreaseMoney 3000)
def pairD := (Quantity.TemperatureRise 4, Quantity.TemperatureDecrease 10)

-- Theorem statement
theorem only_pairC_not_opposite : 
  (haveOppositeMeanings pairA.1 pairA.2) ∧
  (haveOppositeMeanings pairB.1 pairB.2) ∧
  ¬(haveOppositeMeanings pairC.1 pairC.2) ∧
  (haveOppositeMeanings pairD.1 pairD.2) :=
by sorry

end NUMINAMATH_CALUDE_only_pairC_not_opposite_l2057_205735


namespace NUMINAMATH_CALUDE_janet_roller_coaster_rides_l2057_205764

/-- The number of tickets required for one roller coaster ride -/
def roller_coaster_tickets : ℕ := 5

/-- The number of tickets required for one giant slide ride -/
def giant_slide_tickets : ℕ := 3

/-- The number of times Janet wants to ride the giant slide -/
def giant_slide_rides : ℕ := 4

/-- The total number of tickets Janet needs -/
def total_tickets : ℕ := 47

/-- The number of times Janet wants to ride the roller coaster -/
def roller_coaster_rides : ℕ := 7

theorem janet_roller_coaster_rides : 
  roller_coaster_tickets * roller_coaster_rides + 
  giant_slide_tickets * giant_slide_rides = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_janet_roller_coaster_rides_l2057_205764


namespace NUMINAMATH_CALUDE_fifth_student_guess_l2057_205767

def jellybeanGuess (first second third fourth fifth : ℕ) : Prop :=
  second = 8 * first ∧
  third = second - 200 ∧
  fourth = ((first + second + third) / 3 + 25 : ℕ) ∧
  fifth = fourth + (fourth * 20 / 100 : ℕ)

theorem fifth_student_guess :
  ∀ first second third fourth fifth : ℕ,
    first = 100 →
    jellybeanGuess first second third fourth fifth →
    fifth = 630 := by
  sorry

end NUMINAMATH_CALUDE_fifth_student_guess_l2057_205767


namespace NUMINAMATH_CALUDE_power_sum_difference_l2057_205730

theorem power_sum_difference : 2^(1+2+3) - (2^1 + 2^2 + 2^3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l2057_205730


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2057_205747

def cost_price : ℝ := 900
def selling_price : ℝ := 1080

theorem gain_percent_calculation :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2057_205747


namespace NUMINAMATH_CALUDE_new_student_weight_l2057_205758

/-- Given a group of 10 students, proves that replacing a 120 kg student with a new student
    that causes the average weight to decrease by 6 kg results in the new student weighing 60 kg. -/
theorem new_student_weight
  (n : ℕ) -- number of students
  (old_avg : ℝ) -- original average weight
  (replaced_weight : ℝ) -- weight of the replaced student
  (new_avg : ℝ) -- new average weight after replacement
  (h1 : n = 10) -- there are 10 students
  (h2 : new_avg = old_avg - 6) -- average weight decreases by 6 kg
  (h3 : replaced_weight = 120) -- replaced student weighs 120 kg
  : n * new_avg + 60 = n * old_avg - replaced_weight := by
  sorry

#check new_student_weight

end NUMINAMATH_CALUDE_new_student_weight_l2057_205758


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2057_205742

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem complement_intersection_equals_set : 
  (Aᶜ : Set ℕ) ∩ B = {1, 4, 5, 6, 7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2057_205742


namespace NUMINAMATH_CALUDE_inequality_proof_l2057_205789

theorem inequality_proof (x y : ℝ) : 
  (∀ x, |x| + |x - 3| < x + 6 ↔ -1 < x ∧ x < 9) →
  x > 0 →
  y > 0 →
  9*x + y - 1 = 0 →
  x + y ≥ 16*x*y := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2057_205789


namespace NUMINAMATH_CALUDE_solve_for_s_l2057_205738

theorem solve_for_s (s t : ℚ) 
  (eq1 : 8 * s + 7 * t = 160) 
  (eq2 : s = t - 3) : 
  s = 139 / 15 := by
sorry

end NUMINAMATH_CALUDE_solve_for_s_l2057_205738


namespace NUMINAMATH_CALUDE_odd_even_function_inequalities_l2057_205778

-- Define the properties of functions f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y
def coincide_nonneg (f g : ℝ → ℝ) : Prop := ∀ x, x ≥ 0 → f x = g x

-- State the theorem
theorem odd_even_function_inequalities
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h_incr : is_increasing f)
  (h_coinc : coincide_nonneg f g)
  {a b : ℝ}
  (h_ab : a > b)
  (h_b_pos : b > 0) :
  (f b - f (-a) > g a - g (-b)) ∧
  (f a - f (-b) > g b - g (-a)) :=
by sorry

end NUMINAMATH_CALUDE_odd_even_function_inequalities_l2057_205778


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2057_205746

/-- Two-dimensional vector -/
def Vector2D := ℝ × ℝ

/-- Parallel vectors are scalar multiples of each other -/
def is_parallel (v w : Vector2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : Vector2D := (1, -2)
  let b : Vector2D := (-2, x)
  is_parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2057_205746


namespace NUMINAMATH_CALUDE_parallelogram_count_l2057_205748

-- Define the parallelogram structure
structure Parallelogram where
  b : ℕ
  d : ℕ
  area_eq : b * d = 1728000
  b_positive : b > 0
  d_positive : d > 0

-- Define the count function
def count_parallelograms : ℕ := sorry

-- Theorem statement
theorem parallelogram_count : count_parallelograms = 56 := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_l2057_205748


namespace NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l2057_205702

/-- Alice's schedule cycle length -/
def alice_cycle : ℕ := 6

/-- Bob's schedule cycle length -/
def bob_cycle : ℕ := 6

/-- Number of days Alice works in her cycle -/
def alice_work_days : ℕ := 4

/-- Number of days Bob works in his cycle -/
def bob_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 800

/-- Function to calculate the number of coinciding rest days -/
def coinciding_rest_days : ℕ := 
  (total_days / alice_cycle) * (alice_cycle - alice_work_days - bob_work_days + 1)

theorem coinciding_rest_days_theorem : 
  coinciding_rest_days = 133 := by sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l2057_205702


namespace NUMINAMATH_CALUDE_unique_prime_base_l2057_205761

theorem unique_prime_base (b : ℕ) : 
  Prime b ∧ (b + 5)^2 = 3*b^2 + 6*b + 1 → b = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_base_l2057_205761


namespace NUMINAMATH_CALUDE_simplify_linear_expression_l2057_205784

theorem simplify_linear_expression (y : ℝ) : 2*y + 3*y + 4*y = 9*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_linear_expression_l2057_205784


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2057_205754

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((3 + i) / i) = -3 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2057_205754


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2057_205763

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2057_205763


namespace NUMINAMATH_CALUDE_cos_arctan_squared_l2057_205737

theorem cos_arctan_squared (x : ℝ) (h1 : x > 0) (h2 : Real.cos (Real.arctan x) = x) :
  x^2 = (Real.sqrt 5 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_cos_arctan_squared_l2057_205737


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l2057_205785

theorem remainder_of_large_number (n : ℕ) (d : ℕ) (h : n = 123456789012 ∧ d = 210) :
  n % d = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l2057_205785


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2057_205732

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) - 3 * (x - y) / (x + y) = 2) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 8320 / 4095 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2057_205732


namespace NUMINAMATH_CALUDE_product_different_from_hundred_l2057_205795

theorem product_different_from_hundred : ∃! (x y : ℚ), 
  ((x = 10 ∧ y = 10) ∨ 
   (x = 20 ∧ y = -5) ∨ 
   (x = -4 ∧ y = -25) ∨ 
   (x = 50 ∧ y = 2) ∨ 
   (x = 5/2 ∧ y = 40)) ∧ 
  x * y ≠ 100 := by
  sorry

end NUMINAMATH_CALUDE_product_different_from_hundred_l2057_205795


namespace NUMINAMATH_CALUDE_f_properties_l2057_205793

/-- The function f(x) = -x^3 + ax^2 - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

theorem f_properties (a : ℝ) :
  /- 1. Tangent line equation when a = 3 at x = 1 -/
  (a = 3 → ∃ (m b : ℝ), m = 3 ∧ b = -5 ∧ ∀ x y, y = f 3 x ↔ m*x - y - b = 0) ∧
  /- 2. Monotonicity depends on a -/
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ f a x1 > f a x2) ∧
  (∃ (x3 x4 : ℝ), x3 < x4 ∧ f a x3 < f a x4) ∧
  /- 3. Condition for f(x0) > 0 -/
  (∃ (x0 : ℝ), x0 > 0 ∧ f a x0 > 0) ↔ a > 3 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2057_205793


namespace NUMINAMATH_CALUDE_g_has_three_zeros_l2057_205733

/-- A function g(x) with a parameter n -/
def g (n : ℕ) (x : ℝ) : ℝ := 2 * x^n + 10 * x^2 - 2 * x - 1

/-- Theorem stating that g(x) has exactly 3 real zeros when n > 3 and n is odd -/
theorem g_has_three_zeros (n : ℕ) (hn : n > 3) (hodd : Odd n) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g n x = 0 :=
sorry

end NUMINAMATH_CALUDE_g_has_three_zeros_l2057_205733


namespace NUMINAMATH_CALUDE_gardener_roses_order_l2057_205788

/-- The number of roses ordered by the gardener -/
def roses : ℕ := 320

/-- The number of tulips ordered -/
def tulips : ℕ := 250

/-- The number of carnations ordered -/
def carnations : ℕ := 375

/-- The cost of each flower in euros -/
def flower_cost : ℕ := 2

/-- The total expenses in euros -/
def total_expenses : ℕ := 1890

theorem gardener_roses_order :
  roses = (total_expenses - (tulips + carnations) * flower_cost) / flower_cost := by
  sorry

end NUMINAMATH_CALUDE_gardener_roses_order_l2057_205788


namespace NUMINAMATH_CALUDE_glorias_turtle_time_l2057_205781

/-- The time it takes for Gloria's turtle to finish the race -/
def glorias_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

/-- Theorem stating that Gloria's turtle finished in 8 minutes -/
theorem glorias_turtle_time :
  let gretas_time := 6
  let georges_time := gretas_time - 2
  glorias_time gretas_time georges_time = 8 := by sorry

end NUMINAMATH_CALUDE_glorias_turtle_time_l2057_205781


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l2057_205717

theorem yellow_marbles_count (total : ℕ) (red : ℕ) 
  (h1 : total = 140)
  (h2 : red = 10)
  (h3 : ∃ blue : ℕ, blue = (5 * red) / 2)
  (h4 : ∃ green : ℕ, green = ((13 * blue) / 10))
  (h5 : ∃ yellow : ℕ, yellow = total - (blue + red + green)) :
  yellow = 73 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l2057_205717


namespace NUMINAMATH_CALUDE_union_equals_A_l2057_205744

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 24 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2*a)*(x - a) < 0}

-- State the theorem
theorem union_equals_A (a : ℝ) : A ∪ B a = A ↔ a ∈ Set.Icc (-3/2) 4 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l2057_205744


namespace NUMINAMATH_CALUDE_nate_age_when_ember_is_14_l2057_205740

-- Define the initial ages
def nate_initial_age : ℕ := 14
def ember_initial_age : ℕ := nate_initial_age / 2

-- Define the target age for Ember
def ember_target_age : ℕ := 14

-- Calculate the age difference
def age_difference : ℕ := ember_target_age - ember_initial_age

-- Theorem statement
theorem nate_age_when_ember_is_14 :
  nate_initial_age + age_difference = 21 :=
sorry

end NUMINAMATH_CALUDE_nate_age_when_ember_is_14_l2057_205740


namespace NUMINAMATH_CALUDE_enough_paint_l2057_205727

/-- Represents the dimensions of the gym --/
structure GymDimensions where
  length : ℝ
  width : ℝ

/-- Represents the paint requirements and availability --/
structure PaintInfo where
  cans : ℕ
  weight_per_can : ℝ
  paint_per_sqm : ℝ

/-- Theorem stating that there is enough paint for the gym floor --/
theorem enough_paint (gym : GymDimensions) (paint : PaintInfo) : 
  gym.length = 65 ∧ 
  gym.width = 32 ∧ 
  paint.cans = 23 ∧ 
  paint.weight_per_can = 25 ∧ 
  paint.paint_per_sqm = 0.25 → 
  (paint.cans : ℝ) * paint.weight_per_can > gym.length * gym.width * paint.paint_per_sqm := by
  sorry

#check enough_paint

end NUMINAMATH_CALUDE_enough_paint_l2057_205727


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2057_205704

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2057_205704


namespace NUMINAMATH_CALUDE_committees_with_president_count_l2057_205771

/-- The number of different five-student committees with an elected president
    that can be chosen from a group of ten students -/
def committees_with_president : ℕ :=
  (Nat.choose 10 5) * 5

/-- Theorem stating that the number of committees with a president is 1260 -/
theorem committees_with_president_count :
  committees_with_president = 1260 := by
  sorry

end NUMINAMATH_CALUDE_committees_with_president_count_l2057_205771


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l2057_205769

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (α β : Plane) (l m : Line) :
  parallel α β → perpendicular l α → line_parallel m β → 
  line_perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l2057_205769


namespace NUMINAMATH_CALUDE_function_equation_solution_l2057_205731

theorem function_equation_solution (a b : ℚ) :
  ∀ f : ℚ → ℚ, (∀ x y : ℚ, f (x + a + f y) = f (x + b) + y) →
  (∀ x : ℚ, f x = x + b - a) ∨ (∀ x : ℚ, f x = -x + b - a) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2057_205731


namespace NUMINAMATH_CALUDE_min_value_expression_l2057_205760

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 25 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2057_205760


namespace NUMINAMATH_CALUDE_square_area_error_l2057_205765

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := 1.25 * x
  let actual_area := x ^ 2
  let calculated_area := measured_side ^ 2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 56.25 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l2057_205765


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l2057_205759

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l2057_205759


namespace NUMINAMATH_CALUDE_equation_solution_l2057_205707

theorem equation_solution : ∃ x : ℚ, x ≠ 0 ∧ (3 / x - (3 / x) / (9 / x) = 1 / 2) ∧ x = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2057_205707


namespace NUMINAMATH_CALUDE_table_height_l2057_205798

/-- Given three rectangular boxes (blue, red, and green) and their height relationships
    with a table, prove that the height of the table is 91 cm. -/
theorem table_height
  (h b r g : ℝ)
  (eq1 : h + b - g = 111)
  (eq2 : h + r - b = 80)
  (eq3 : h + g - r = 82) :
  h = 91 := by
sorry

end NUMINAMATH_CALUDE_table_height_l2057_205798


namespace NUMINAMATH_CALUDE_vector_dot_product_l2057_205743

/-- Given two vectors a and b in ℝ², prove that their dot product is -12
    when their sum is (1, -3) and their difference is (3, 7). -/
theorem vector_dot_product (a b : ℝ × ℝ) 
    (h1 : a.1 + b.1 = 1 ∧ a.2 + b.2 = -3)
    (h2 : a.1 - b.1 = 3 ∧ a.2 - b.2 = 7) :
    a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2057_205743


namespace NUMINAMATH_CALUDE_range_of_a_l2057_205796

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |x - a| < 2) ∧ 
  (∃ x : ℝ, |x - a| < 2 ∧ (x < 1 ∨ x > 3)) ↔ 
  1 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2057_205796


namespace NUMINAMATH_CALUDE_cubic_polynomial_with_coefficient_roots_l2057_205720

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The polynomial function for a CubicPolynomial -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℚ) : ℚ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Predicate for a CubicPolynomial having its coefficients as roots -/
def CubicPolynomial.hasCoefficientsAsRoots (p : CubicPolynomial) : Prop :=
  p.eval p.a = 0 ∧ p.eval p.b = 0 ∧ p.eval p.c = 0

/-- The two specific polynomials mentioned in the problem -/
def f₁ : CubicPolynomial := ⟨1, -2, 0⟩
def f₂ : CubicPolynomial := ⟨1, -1, -1⟩

/-- The main theorem stating that f₁ and f₂ are the only valid polynomials -/
theorem cubic_polynomial_with_coefficient_roots :
  ∀ p : CubicPolynomial, p.hasCoefficientsAsRoots → p = f₁ ∨ p = f₂ := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_with_coefficient_roots_l2057_205720


namespace NUMINAMATH_CALUDE_angle_parallel_lines_l2057_205749

-- Define the types for lines and angles
variable (Line : Type) (Angle : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the angle between two lines
variable (angle_between : Line → Line → Angle)

-- Define equality for angles
variable (angle_eq : Angle → Angle → Prop)

-- Theorem statement
theorem angle_parallel_lines 
  (a b c : Line) (θ : Angle)
  (h1 : parallel a b)
  (h2 : angle_eq (angle_between a c) θ) :
  angle_eq (angle_between b c) θ :=
sorry

end NUMINAMATH_CALUDE_angle_parallel_lines_l2057_205749


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2057_205722

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 4*x - 1
  ∃ x₁ x₂ : ℝ, (x₁ = -2 + Real.sqrt 5 ∧ x₂ = -2 - Real.sqrt 5) ∧ 
              (f x₁ = 0 ∧ f x₂ = 0) ∧
              (∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2057_205722


namespace NUMINAMATH_CALUDE_fruit_store_discount_l2057_205775

/-- Represents the discount policy of a fruit store -/
def discount_policy (lemon_price papaya_price mango_price : ℕ)
                    (lemon_qty papaya_qty mango_qty : ℕ)
                    (total_paid : ℕ) : ℕ :=
  let total_cost := lemon_price * lemon_qty + papaya_price * papaya_qty + mango_price * mango_qty
  let total_fruits := lemon_qty + papaya_qty + mango_qty
  total_cost - total_paid

theorem fruit_store_discount :
  discount_policy 2 1 4 6 4 2 21 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_store_discount_l2057_205775


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2057_205734

theorem simplify_sqrt_expression : 
  (Real.sqrt 192 / Real.sqrt 27) - (Real.sqrt 500 / Real.sqrt 125) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2057_205734


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l2057_205787

/-- Given three square regions A, B, and C with perimeters 16, 32, and 20 units respectively,
    prove that the ratio of the area of region B to the area of region C is 64/25. -/
theorem area_ratio_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (perim_a : 4 * a = 16) (perim_b : 4 * b = 32) (perim_c : 4 * c = 20) :
  (b * b) / (c * c) = 64 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l2057_205787


namespace NUMINAMATH_CALUDE_quadratic_equation_has_two_distinct_real_roots_l2057_205736

/-- Proves that the quadratic equation (2kx^2 + 5kx + 2) = 0, where k = 0.64, has two distinct real roots -/
theorem quadratic_equation_has_two_distinct_real_roots :
  let k : ℝ := 0.64
  let a : ℝ := 2 * k
  let b : ℝ := 5 * k
  let c : ℝ := 2
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_two_distinct_real_roots_l2057_205736


namespace NUMINAMATH_CALUDE_line_perpendicular_slope_l2057_205712

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), perpendicular to a line
    through (-2, 1) with slope -2/3, prove that a = -2/3 -/
theorem line_perpendicular_slope (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = ((1 - t) * (a - 2) + t * (-a - 2), (1 - t) * (-1) + t * 1)}
  let m : ℝ := (1 - (-1)) / ((-a - 2) - (a - 2))
  (∀ p ∈ l, (p.2 - 1) = -2/3 * (p.1 - (-2))) → m * (-2/3) = -1 → a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_line_perpendicular_slope_l2057_205712


namespace NUMINAMATH_CALUDE_remainder_theorem_l2057_205703

/-- Given a polynomial p(x) satisfying p(0) = 2 and p(2) = 6,
    prove that the remainder when p(x) is divided by x(x-2) is 2x + 2 -/
theorem remainder_theorem (p : ℝ → ℝ) (h1 : p 0 = 2) (h2 : p 2 = 6) :
  ∃ (q : ℝ → ℝ), ∀ x, p x = q x * (x * (x - 2)) + (2 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2057_205703


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2057_205791

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 20 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 190. -/
theorem chess_tournament_games :
  num_games 20 = 190 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2057_205791


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2057_205766

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 3 + a 11 = 3 →
  a 5 + a 6 + a 10 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2057_205766


namespace NUMINAMATH_CALUDE_sqrt_factorial_five_squared_l2057_205777

theorem sqrt_factorial_five_squared (n : ℕ) : n = 5 → Real.sqrt ((n.factorial : ℝ) * n.factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_factorial_five_squared_l2057_205777


namespace NUMINAMATH_CALUDE_basketball_team_score_l2057_205725

theorem basketball_team_score :
  ∀ (chandra akiko michiko bailey damien ella : ℕ),
    chandra = 2 * akiko →
    akiko = michiko + 4 →
    michiko * 2 = bailey →
    bailey = 14 →
    damien = 3 * akiko →
    ella = chandra + (chandra / 5) →
    chandra + akiko + michiko + bailey + damien + ella = 113 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_score_l2057_205725


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l2057_205721

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l2057_205721


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2057_205719

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2057_205719


namespace NUMINAMATH_CALUDE_product_469111_9999_l2057_205762

theorem product_469111_9999 : 469111 * 9999 = 4690418889 := by
  sorry

end NUMINAMATH_CALUDE_product_469111_9999_l2057_205762


namespace NUMINAMATH_CALUDE_codecracker_codes_count_l2057_205713

/-- The number of available colors in the CodeCracker game -/
def num_colors : ℕ := 6

/-- The number of slots in a CodeCracker code -/
def code_length : ℕ := 5

/-- The number of possible secret codes in the CodeCracker game -/
def num_codes : ℕ := num_colors * (num_colors - 1)^(code_length - 1)

theorem codecracker_codes_count :
  num_codes = 3750 :=
by sorry

end NUMINAMATH_CALUDE_codecracker_codes_count_l2057_205713


namespace NUMINAMATH_CALUDE_action_figures_earnings_l2057_205756

/-- Calculates the total earnings from selling action figures with discounts -/
def total_earnings (type_a_count type_b_count type_c_count type_d_count : ℕ)
                   (type_a_value type_b_value type_c_value type_d_value : ℕ)
                   (type_a_discount type_b_discount type_c_discount type_d_discount : ℕ) : ℕ :=
  (type_a_count * (type_a_value - type_a_discount)) +
  (type_b_count * (type_b_value - type_b_discount)) +
  (type_c_count * (type_c_value - type_c_discount)) +
  (type_d_count * (type_d_value - type_d_discount))

/-- Theorem stating that the total earnings from selling all action figures is $435 -/
theorem action_figures_earnings :
  total_earnings 6 5 4 5 22 35 45 50 10 14 18 20 = 435 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_earnings_l2057_205756


namespace NUMINAMATH_CALUDE_constant_kill_time_l2057_205768

/-- Represents the time in minutes it takes for a given number of cats to kill the same number of rats -/
def killTime (n : ℕ) : ℝ :=
  3  -- We define this as 3 based on the given condition for 100 cats and 100 rats

theorem constant_kill_time (n : ℕ) (h : n ≥ 3) : killTime n = 3 := by
  sorry

#check constant_kill_time

end NUMINAMATH_CALUDE_constant_kill_time_l2057_205768


namespace NUMINAMATH_CALUDE_decreasing_linear_function_conditions_l2057_205745

/-- A linear function y = kx - b where y decreases as x increases
    and intersects the y-axis above the x-axis -/
def DecreasingLinearFunction (k b : ℝ) : Prop :=
  k < 0 ∧ b > 0

/-- Theorem stating that for a linear function y = kx - b,
    if y decreases as x increases and intersects the y-axis above the x-axis,
    then k < 0 and b > 0 -/
theorem decreasing_linear_function_conditions (k b : ℝ) :
  DecreasingLinearFunction k b ↔ k < 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_conditions_l2057_205745
