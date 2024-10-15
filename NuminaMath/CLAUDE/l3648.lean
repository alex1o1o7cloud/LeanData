import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_m_l3648_364814

theorem solve_for_m (Q t h m : ℝ) (hQ : Q > 0) (ht : t > 0) (hh : h ≥ 0) :
  Q = t / (1 + Real.sqrt h)^m ↔ m = Real.log (t / Q) / Real.log (1 + Real.sqrt h) :=
sorry

end NUMINAMATH_CALUDE_solve_for_m_l3648_364814


namespace NUMINAMATH_CALUDE_tencent_dialectical_materialism_alignment_l3648_364861

/-- Represents the principles of dialectical materialism -/
structure DialecticalMaterialism where
  dialectical_negation : Bool
  innovation : Bool
  development : Bool
  unity_of_opposites : Bool
  unity_of_progressiveness_and_tortuosity : Bool
  unity_of_quantitative_and_qualitative_changes : Bool

/-- Represents Tencent's development characteristics -/
structure TencentDevelopment where
  technological_innovation : Bool
  continuous_growth : Bool
  overcoming_difficulties : Bool
  qualitative_leaps : Bool

/-- Given information about Tencent's development history -/
axiom tencent_history : TencentDevelopment

/-- Theorem stating that Tencent's development aligns with dialectical materialism -/
theorem tencent_dialectical_materialism_alignment :
  ∃ (dm : DialecticalMaterialism),
    dm.dialectical_negation ∧
    dm.innovation ∧
    dm.development ∧
    dm.unity_of_opposites ∧
    dm.unity_of_progressiveness_and_tortuosity ∧
    dm.unity_of_quantitative_and_qualitative_changes ∧
    tencent_history.technological_innovation ∧
    tencent_history.continuous_growth ∧
    tencent_history.overcoming_difficulties ∧
    tencent_history.qualitative_leaps :=
by
  sorry

end NUMINAMATH_CALUDE_tencent_dialectical_materialism_alignment_l3648_364861


namespace NUMINAMATH_CALUDE_num_valid_selections_l3648_364848

/-- Represents the set of volunteers --/
inductive Volunteer
| A
| B
| C
| D
| E

/-- Represents the set of roles --/
inductive Role
| Translator
| TourGuide
| Etiquette
| Driver

/-- Predicate to check if a volunteer can take on a role --/
def canTakeRole (v : Volunteer) (r : Role) : Prop :=
  match v, r with
  | Volunteer.A, Role.Driver => False
  | Volunteer.B, Role.Driver => False
  | _, _ => True

/-- A selection is a function from Role to Volunteer --/
def Selection := Role → Volunteer

/-- Predicate to check if a selection is valid --/
def validSelection (s : Selection) : Prop :=
  (∀ r : Role, canTakeRole (s r) r) ∧
  (∀ v : Volunteer, ∃! r : Role, s r = v)

/-- The number of valid selections --/
def numValidSelections : ℕ := sorry

theorem num_valid_selections :
  numValidSelections = 72 := by sorry

end NUMINAMATH_CALUDE_num_valid_selections_l3648_364848


namespace NUMINAMATH_CALUDE_helga_shoe_shopping_l3648_364863

theorem helga_shoe_shopping (first_store : ℕ) (second_store : ℕ) (third_store : ℕ) :
  first_store = 7 →
  second_store = first_store + 2 →
  third_store = 0 →
  let total_first_three := first_store + second_store + third_store
  let fourth_store := 2 * total_first_three
  first_store + second_store + third_store + fourth_store = 48 :=
by sorry

end NUMINAMATH_CALUDE_helga_shoe_shopping_l3648_364863


namespace NUMINAMATH_CALUDE_base3_to_base10_equality_l3648_364808

/-- Converts a base 3 number represented as a list of digits to its base 10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number we want to convert -/
def base3Number : List Nat := [2, 1, 2, 0, 1]

/-- Theorem stating that the base 3 number 10212₃ is equal to 104 in base 10 -/
theorem base3_to_base10_equality : base3ToBase10 base3Number = 104 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_equality_l3648_364808


namespace NUMINAMATH_CALUDE_probability_of_specific_selection_l3648_364899

/-- A bag containing balls of different colors -/
structure BagOfBalls where
  total : ℕ
  white : ℕ
  red : ℕ
  black : ℕ

/-- The probability of selecting balls with specific conditions -/
def probability_of_selection (bag : BagOfBalls) (selected : ℕ) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem probability_of_specific_selection : 
  let bag : BagOfBalls := ⟨20, 9, 5, 6⟩
  probability_of_selection bag 10 = 7 / 92378 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_selection_l3648_364899


namespace NUMINAMATH_CALUDE_intersection_distance_to_side_l3648_364840

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

def intersectionPoint (c1 c2 : Circle) : Point := sorry

/-- Calculates the distance between a point and a line defined by y = k -/
def distanceToHorizontalLine (p : Point) (k : ℝ) : ℝ := sorry

theorem intersection_distance_to_side (s : Square) 
  (c1 c2 : Circle) (h1 : s.sideLength = 10) 
  (h2 : c1.center = s.A) (h3 : c2.center = s.B) 
  (h4 : c1.radius = 5) (h5 : c2.radius = 5) :
  let X := intersectionPoint c1 c2
  distanceToHorizontalLine X s.sideLength = 10 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_to_side_l3648_364840


namespace NUMINAMATH_CALUDE_percentage_of_number_l3648_364874

theorem percentage_of_number (x : ℝ) (h : x = 16) : x * 0.0025 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l3648_364874


namespace NUMINAMATH_CALUDE_selection_probabilities_l3648_364801

/-- Represents the probabilities of passing selections for a student -/
structure StudentProb where
  first : ℝ  -- Probability of passing first selection
  second : ℝ  -- Probability of passing second selection

/-- Given probabilities for students A, B, and C, prove the required probabilities -/
theorem selection_probabilities (a b c : StudentProb)
  (ha_first : a.first = 0.5) (ha_second : a.second = 0.6)
  (hb_first : b.first = 0.6) (hb_second : b.second = 0.5)
  (hc_first : c.first = 0.4) (hc_second : c.second = 0.5) :
  (a.first * (1 - b.first) = 0.2) ∧
  (a.first * a.second * (1 - b.first * b.second) * (1 - c.first * c.second) +
   (1 - a.first * a.second) * b.first * b.second * (1 - c.first * c.second) +
   (1 - a.first * a.second) * (1 - b.first * b.second) * c.first * c.second = 217 / 500) :=
by sorry


end NUMINAMATH_CALUDE_selection_probabilities_l3648_364801


namespace NUMINAMATH_CALUDE_unit_digit_of_4137_to_754_l3648_364806

theorem unit_digit_of_4137_to_754 : (4137^754) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_4137_to_754_l3648_364806


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3648_364884

-- Define the polynomial
def f (x : ℝ) := 40 * x^3 - 70 * x^2 + 32 * x - 2

-- State the theorem
theorem root_sum_reciprocal (a b c : ℝ) :
  f a = 0 → f b = 0 → f c = 0 →  -- a, b, c are roots of f
  a ≠ b → b ≠ c → a ≠ c →        -- a, b, c are distinct
  0 < a → a < 1 →                -- 0 < a < 1
  0 < b → b < 1 →                -- 0 < b < 1
  0 < c → c < 1 →                -- 0 < c < 1
  1/(1-a) + 1/(1-b) + 1/(1-c) = 11/20 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3648_364884


namespace NUMINAMATH_CALUDE_E_parity_l3648_364831

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 2) + E (n + 1)

def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem E_parity : isEven (E 2023) ∧ isOdd (E 2024) ∧ isOdd (E 2025) := by sorry

end NUMINAMATH_CALUDE_E_parity_l3648_364831


namespace NUMINAMATH_CALUDE_rubber_boat_fall_time_l3648_364851

/-- Represents the speed of the ship in still water -/
def ship_speed : ℝ := sorry

/-- Represents the speed of the water flow -/
def water_flow : ℝ := sorry

/-- Represents the time (in hours) when the rubber boat fell into the water, before 5 PM -/
def fall_time : ℝ := sorry

/-- Represents the fact that the ship catches up with the rubber boat after 1 hour -/
axiom catch_up_condition : (5 - fall_time) * (ship_speed - water_flow) + (6 - fall_time) * water_flow = ship_speed + water_flow

theorem rubber_boat_fall_time : fall_time = 4 := by sorry

end NUMINAMATH_CALUDE_rubber_boat_fall_time_l3648_364851


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2018_l3648_364896

theorem last_two_digits_of_7_pow_2018 : 7^2018 % 100 = 49 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2018_l3648_364896


namespace NUMINAMATH_CALUDE_lisa_flight_time_l3648_364852

/-- 
Given that Lisa flew 500 miles at a speed of 45 miles per hour, 
prove that the time Lisa flew is equal to 500 miles divided by 45 miles per hour.
-/
theorem lisa_flight_time : 
  let distance : ℝ := 500  -- Distance in miles
  let speed : ℝ := 45      -- Speed in miles per hour
  let time : ℝ := distance / speed
  time = 500 / 45 := by sorry

end NUMINAMATH_CALUDE_lisa_flight_time_l3648_364852


namespace NUMINAMATH_CALUDE_log_equation_solution_l3648_364815

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h' : x > 0) :
  Real.log x / Real.log k * Real.log k / Real.log 5 = 3 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3648_364815


namespace NUMINAMATH_CALUDE_sum_of_squares_l3648_364816

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 20)
  (eq2 : y^2 + 5*z = -20)
  (eq3 : z^2 + 7*x = -34) :
  x^2 + y^2 + z^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3648_364816


namespace NUMINAMATH_CALUDE_binomial_600_600_l3648_364802

theorem binomial_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_600_600_l3648_364802


namespace NUMINAMATH_CALUDE_correct_travel_distance_l3648_364853

/-- The distance traveled by Gavril on the electric train -/
def travel_distance : ℝ := 257

/-- The time it takes for the smartphone to fully discharge while watching videos -/
def video_discharge_time : ℝ := 3

/-- The time it takes for the smartphone to fully discharge while playing Tetris -/
def tetris_discharge_time : ℝ := 5

/-- The speed of the train for the first half of the journey -/
def speed_first_half : ℝ := 80

/-- The speed of the train for the second half of the journey -/
def speed_second_half : ℝ := 60

/-- Theorem stating that given the conditions, the travel distance is correct -/
theorem correct_travel_distance :
  let total_time := (video_discharge_time * tetris_discharge_time) / (video_discharge_time / 2 + tetris_discharge_time / 2)
  travel_distance = total_time * (speed_first_half / 2 + speed_second_half / 2) :=
by sorry

end NUMINAMATH_CALUDE_correct_travel_distance_l3648_364853


namespace NUMINAMATH_CALUDE_max_garden_area_l3648_364839

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular garden with one side against a house -/
def gardenPerimeter (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- The available fencing length -/
def availableFencing : ℝ := 400

theorem max_garden_area :
  ∃ (d : GardenDimensions),
    gardenPerimeter d = availableFencing ∧
    ∀ (d' : GardenDimensions),
      gardenPerimeter d' = availableFencing →
      gardenArea d' ≤ gardenArea d ∧
      gardenArea d = 20000 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l3648_364839


namespace NUMINAMATH_CALUDE_saree_pricing_l3648_364880

/-- Calculates the final price of a saree given the original price and discount options --/
def calculate_final_price (original_price : ℚ) : ℚ × ℚ × ℚ := by
  -- Define the discount options
  let option_a : ℚ := (original_price * (1 - 0.18) - 100) * (1 - 0.05) * (1 + 0.0325) + 50
  let option_b : ℚ := original_price * (1 - 0.25) * (1 + 0.0275) * (1 + 0.0175)
  let option_c : ℚ := original_price * (1 - 0.12) * (1 - 0.06) * (1 + 0.035) * (1 + 0.0225)
  
  exact (option_a, option_b, option_c)

/-- Theorem stating the final prices for each option --/
theorem saree_pricing (original_price : ℚ) :
  original_price = 1200 →
  let (price_a, price_b, price_c) := calculate_final_price original_price
  price_a = 917.09 ∧ price_b = 940.93 ∧ price_c = 1050.50 := by
  sorry

end NUMINAMATH_CALUDE_saree_pricing_l3648_364880


namespace NUMINAMATH_CALUDE_correct_performance_calculation_l3648_364871

/-- Represents a batsman's performance in a cricket match -/
structure BatsmanPerformance where
  initialAverage : ℝ
  eleventhInningRuns : ℝ
  averageIncrease : ℝ
  teamHandicap : ℝ

/-- Calculates the new average and total team runs for a batsman -/
def calculatePerformance (performance : BatsmanPerformance) : ℝ × ℝ :=
  let newAverage := performance.initialAverage + performance.averageIncrease
  let totalBatsmanRuns := 11 * newAverage
  let totalTeamRuns := totalBatsmanRuns + performance.teamHandicap
  (newAverage, totalTeamRuns)

/-- Theorem stating the correct calculation of a batsman's performance -/
theorem correct_performance_calculation 
  (performance : BatsmanPerformance) 
  (h1 : performance.eleventhInningRuns = 85)
  (h2 : performance.averageIncrease = 5)
  (h3 : performance.teamHandicap = 75) :
  calculatePerformance performance = (35, 460) := by
  sorry

#check correct_performance_calculation

end NUMINAMATH_CALUDE_correct_performance_calculation_l3648_364871


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_5000_l3648_364844

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def sumOfDigitsUpTo (n : ℕ) : ℕ :=
  (List.range n).map sumOfDigits |>.sum

theorem sum_of_digits_up_to_5000 : 
  sumOfDigitsUpTo 5000 = 167450 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_5000_l3648_364844


namespace NUMINAMATH_CALUDE_intersection_points_count_l3648_364866

/-- The number of intersection points between y = Bx^2 and y^2 + 4y - 2 = x^2 + 5y -/
theorem intersection_points_count (B : ℝ) (hB : B > 0) : 
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (y1 = B * x1^2 ∧ y1^2 + 4*y1 - 2 = x1^2 + 5*y1) ∧
    (y2 = B * x2^2 ∧ y2^2 + 4*y2 - 2 = x2^2 + 5*y2) ∧
    (y3 = B * x3^2 ∧ y3^2 + 4*y3 - 2 = x3^2 + 5*y3) ∧
    (y4 = B * x4^2 ∧ y4^2 + 4*y4 - 2 = x4^2 + 5*y4) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x1 ≠ x4 ∨ y1 ≠ y4) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3) ∧
    (x2 ≠ x4 ∨ y2 ≠ y4) ∧
    (x3 ≠ x4 ∨ y3 ≠ y4) ∧
    ∀ (x y : ℝ), (y = B * x^2 ∧ y^2 + 4*y - 2 = x^2 + 5*y) →
      ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l3648_364866


namespace NUMINAMATH_CALUDE_jessie_weight_before_jogging_l3648_364834

/-- Jessie's weight before jogging, given her current weight and weight loss -/
theorem jessie_weight_before_jogging 
  (current_weight : ℕ) 
  (weight_loss : ℕ) 
  (h1 : current_weight = 67) 
  (h2 : weight_loss = 7) : 
  current_weight + weight_loss = 74 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_before_jogging_l3648_364834


namespace NUMINAMATH_CALUDE_square_root_sum_l3648_364827

theorem square_root_sum (x : ℝ) 
  (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) : 
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l3648_364827


namespace NUMINAMATH_CALUDE_cylindrical_bar_length_l3648_364824

/-- The length of a cylindrical steel bar formed from a rectangular billet -/
theorem cylindrical_bar_length 
  (billet_length : ℝ) 
  (billet_width : ℝ) 
  (billet_height : ℝ) 
  (cylinder_diameter : ℝ) 
  (h1 : billet_length = 12.56)
  (h2 : billet_width = 5)
  (h3 : billet_height = 4)
  (h4 : cylinder_diameter = 4) : 
  (billet_length * billet_width * billet_height) / (π * (cylinder_diameter / 2)^2) = 20 := by
  sorry

#check cylindrical_bar_length

end NUMINAMATH_CALUDE_cylindrical_bar_length_l3648_364824


namespace NUMINAMATH_CALUDE_integer_solution_inequality_l3648_364817

theorem integer_solution_inequality (x : ℤ) : 
  (3 * |2 * x + 1| + 6 < 24) ↔ x ∈ ({-3, -2, -1, 0, 1, 2} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_inequality_l3648_364817


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l3648_364860

theorem smallest_x_with_remainders : ∃ x : ℕ+, 
  (x : ℕ) % 3 = 2 ∧ 
  (x : ℕ) % 7 = 6 ∧ 
  (x : ℕ) % 8 = 7 ∧ 
  (∀ y : ℕ+, y < x → 
    (y : ℕ) % 3 ≠ 2 ∨ 
    (y : ℕ) % 7 ≠ 6 ∨ 
    (y : ℕ) % 8 ≠ 7) ∧
  x = 167 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l3648_364860


namespace NUMINAMATH_CALUDE_equation_implies_m_equals_zero_l3648_364868

theorem equation_implies_m_equals_zero (m n : ℝ) :
  21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_implies_m_equals_zero_l3648_364868


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l3648_364847

/-- The perimeter of the shaded region formed by the segments where three identical touching circles intersect is equal to the circumference of one circle. -/
theorem shaded_region_perimeter (circle_circumference : ℝ) (segment_angle : ℝ) : 
  circle_circumference > 0 →
  segment_angle = 120 →
  (3 * (segment_angle / 360) * circle_circumference) = circle_circumference :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l3648_364847


namespace NUMINAMATH_CALUDE_range_of_m_l3648_364813

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2 * y = 4)
  (h_solution : ∃ m : ℝ, m^2 + (1/3) * m > 2/x + 1/(y+1)) :
  ∃ m : ℝ, (m < -4/3 ∨ m > 1) ∧ m^2 + (1/3) * m > 2/x + 1/(y+1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3648_364813


namespace NUMINAMATH_CALUDE_max_acute_triangles_formula_l3648_364859

/-- Represents a line with marked points -/
structure MarkedLine where
  points : Finset ℝ
  distinct : points.card = 50

/-- The maximum number of acute-angled triangles formed by points on two parallel lines -/
def max_acute_triangles (a b : MarkedLine) : ℕ :=
  (50^3 - 50) / 3

/-- Theorem stating the maximum number of acute-angled triangles -/
theorem max_acute_triangles_formula (a b : MarkedLine) (h : a.points ∩ b.points = ∅) :
  max_acute_triangles a b = 41650 :=
sorry

end NUMINAMATH_CALUDE_max_acute_triangles_formula_l3648_364859


namespace NUMINAMATH_CALUDE_expression_factorization_l3648_364856

theorem expression_factorization (a b c x : ℝ) :
  (x - a)^2 * (b - c) + (x - b)^2 * (c - a) + (x - c)^2 * (a - b) = -(a - b) * (b - c) * (c - a) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3648_364856


namespace NUMINAMATH_CALUDE_meeting_point_relationship_l3648_364818

/-- Represents the scenario of two vehicles meeting on a road --/
structure MeetingScenario where
  S : ℝ  -- Distance between village and city
  x : ℝ  -- Speed of the truck
  y : ℝ  -- Speed of the car
  t : ℝ  -- Time taken to meet under normal conditions
  t1 : ℝ  -- Time taken to meet if truck leaves 45 minutes earlier
  t2 : ℝ  -- Time taken to meet if car leaves 20 minutes earlier

/-- The theorem stating the relationship between the meeting points --/
theorem meeting_point_relationship (scenario : MeetingScenario) :
  scenario.t = scenario.S / (scenario.x + scenario.y) →
  scenario.t1 = (scenario.S - 0.75 * scenario.x) / (scenario.x + scenario.y) →
  scenario.t2 = (scenario.S - scenario.y / 3) / (scenario.x + scenario.y) →
  0.75 * scenario.x + (scenario.S - 0.75 * scenario.x) * scenario.x / (scenario.x + scenario.y) - scenario.S * scenario.x / (scenario.x + scenario.y) = 18 →
  scenario.S * scenario.x / (scenario.x + scenario.y) - (scenario.S - scenario.y / 3) * scenario.x / (scenario.x + scenario.y) = 8 :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_relationship_l3648_364818


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3648_364898

theorem book_pages_calculation (pages_per_night : ℝ) (nights : ℝ) (h1 : pages_per_night = 120.0) (h2 : nights = 10.0) :
  pages_per_night * nights = 1200.0 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l3648_364898


namespace NUMINAMATH_CALUDE_diamond_calculation_l3648_364830

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 3 4) 2) - (diamond 3 (diamond 4 2)) = -13/28 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3648_364830


namespace NUMINAMATH_CALUDE_flour_for_nine_biscuits_l3648_364892

/-- The amount of flour needed to make a certain number of biscuits -/
def flour_needed (num_biscuits : ℕ) : ℝ :=
  sorry

theorem flour_for_nine_biscuits :
  let members : ℕ := 18
  let biscuits_per_member : ℕ := 2
  let total_flour : ℝ := 5
  flour_needed (members * biscuits_per_member) = total_flour →
  flour_needed 9 = 1.25 :=
by sorry

end NUMINAMATH_CALUDE_flour_for_nine_biscuits_l3648_364892


namespace NUMINAMATH_CALUDE_min_area_square_on_parabola_l3648_364877

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a parabola y = x^2 -/
def OnParabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Defines a square with three vertices on a parabola -/
structure SquareOnParabola where
  A : Point
  B : Point
  C : Point
  onParabola : OnParabola A ∧ OnParabola B ∧ OnParabola C
  isSquare : (A.x - B.x)^2 + (A.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2

/-- The area of a square given its side length -/
def SquareArea (sideLength : ℝ) : ℝ :=
  sideLength^2

/-- Theorem: The minimum area of a square with three vertices on the parabola y = x^2 is 2 -/
theorem min_area_square_on_parabola :
  ∀ s : SquareOnParabola, SquareArea (Real.sqrt ((s.A.x - s.B.x)^2 + (s.A.y - s.B.y)^2)) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_area_square_on_parabola_l3648_364877


namespace NUMINAMATH_CALUDE_min_trios_l3648_364849

/-- Represents a group of people in a meeting -/
structure Meeting :=
  (people : Finset Nat)
  (handshakes : Set (Nat × Nat))
  (size_eq : people.card = 5)

/-- Defines a trio in the meeting -/
def is_trio (m : Meeting) (a b c : Nat) : Prop :=
  (a ∈ m.people ∧ b ∈ m.people ∧ c ∈ m.people) ∧
  ((⟨a, b⟩ ∈ m.handshakes ∧ ⟨b, c⟩ ∈ m.handshakes) ∨
   (⟨a, b⟩ ∉ m.handshakes ∧ ⟨b, c⟩ ∉ m.handshakes))

/-- Counts the number of unique trios in the meeting -/
def count_trios (m : Meeting) : Nat :=
  (m.people.powerset.filter (fun s => s.card = 3)).card

/-- The main theorem stating the minimum number of trios -/
theorem min_trios (m : Meeting) : 
  ∃ (handshakes : Set (Nat × Nat)), count_trios { people := m.people, handshakes := handshakes, size_eq := m.size_eq } = 10 ∧ 
  ∀ (other_handshakes : Set (Nat × Nat)), count_trios { people := m.people, handshakes := other_handshakes, size_eq := m.size_eq } ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_trios_l3648_364849


namespace NUMINAMATH_CALUDE_kevins_food_spending_l3648_364895

theorem kevins_food_spending (total_budget : ℕ) (samuels_ticket : ℕ) (samuels_food_drinks : ℕ)
  (kevins_ticket : ℕ) (kevins_drinks : ℕ) (kevins_food : ℕ)
  (h1 : total_budget = 20)
  (h2 : samuels_ticket = 14)
  (h3 : samuels_food_drinks = 6)
  (h4 : kevins_ticket = 14)
  (h5 : kevins_drinks = 2)
  (h6 : samuels_ticket + samuels_food_drinks = total_budget)
  (h7 : kevins_ticket + kevins_drinks + kevins_food = total_budget) :
  kevins_food = 4 := by
  sorry

end NUMINAMATH_CALUDE_kevins_food_spending_l3648_364895


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3648_364881

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 27) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 ∧ x = 24 :=
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3648_364881


namespace NUMINAMATH_CALUDE_hyperbola_and_line_l3648_364812

/-- Hyperbola with center at origin, right focus at (2,0), and distance 1 from focus to asymptote -/
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  right_focus : ℝ × ℝ := (2, 0)
  focus_to_asymptote : ℝ := 1

/-- Line that intersects the hyperbola at two distinct points -/
structure IntersectingLine where
  k : ℝ
  b : ℝ := 2

/-- Theorem about the hyperbola and its intersecting line -/
theorem hyperbola_and_line (C : Hyperbola) (l : IntersectingLine) :
  (∀ A B : ℝ × ℝ, A ≠ B → (A.1^2/3 - A.2^2 = 1 ∧ A.2 = l.k * A.1 + l.b) →
                        (B.1^2/3 - B.2^2 = 1 ∧ B.2 = l.k * B.1 + l.b) →
                        A.1 * B.1 + A.2 * B.2 > 2) →
  (∀ x y : ℝ, x^2/3 - y^2 = 1 ↔ C.center = (0, 0) ∧ C.right_focus = (2, 0) ∧ C.focus_to_asymptote = 1) ∧
  (l.k ∈ Set.Ioo (-Real.sqrt 15 / 3) (-Real.sqrt 3 / 3) ∪ Set.Ioo (Real.sqrt 3 / 3) (Real.sqrt 15 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_l3648_364812


namespace NUMINAMATH_CALUDE_S_intersect_T_characterization_l3648_364821

def S : Set ℝ := {x | |x| < 5}
def T : Set ℝ := {x | x^2 + 4*x - 21 < 0}

theorem S_intersect_T_characterization :
  S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_characterization_l3648_364821


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l3648_364810

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l3648_364810


namespace NUMINAMATH_CALUDE_small_font_words_per_page_l3648_364894

/-- Calculates the number of words per page in the small font given the article constraints -/
theorem small_font_words_per_page 
  (total_words : ℕ) 
  (total_pages : ℕ) 
  (large_font_pages : ℕ) 
  (large_font_words_per_page : ℕ) 
  (h1 : total_words = 48000)
  (h2 : total_pages = 21)
  (h3 : large_font_pages = 4)
  (h4 : large_font_words_per_page = 1800) :
  (total_words - large_font_pages * large_font_words_per_page) / (total_pages - large_font_pages) = 2400 :=
by
  sorry

#check small_font_words_per_page

end NUMINAMATH_CALUDE_small_font_words_per_page_l3648_364894


namespace NUMINAMATH_CALUDE_remainder_of_power_sum_l3648_364878

/-- The remainder when 5^94 + 7^94 is divided by 55 is 29. -/
theorem remainder_of_power_sum : (5^94 + 7^94) % 55 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_sum_l3648_364878


namespace NUMINAMATH_CALUDE_exactly_two_propositions_true_l3648_364841

-- Define the propositions
def proposition1 : Prop := ∀ x : ℝ, (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)
def proposition2 : Prop := (∀ x y : ℝ, x + y = 0 → (x = -y)) ↔ (∀ x y : ℝ, x = -y → x + y = 0)

-- Theorem statement
theorem exactly_two_propositions_true : 
  (proposition1 = true) ∧ (proposition2 = true) ∧
  (¬ proposition1 = false) ∧ (¬ proposition2 = false) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_propositions_true_l3648_364841


namespace NUMINAMATH_CALUDE_medium_lights_count_l3648_364855

/-- Represents the number of medium ceiling lights -/
def M : ℕ := sorry

/-- The number of small ceiling lights -/
def small_lights : ℕ := M + 10

/-- The number of large ceiling lights -/
def large_lights : ℕ := 2 * M

/-- The total number of bulbs needed -/
def total_bulbs : ℕ := 118

/-- Theorem stating that the number of medium ceiling lights is 12 -/
theorem medium_lights_count : M = 12 := by
  have bulb_equation : small_lights * 1 + M * 2 + large_lights * 3 = total_bulbs := by sorry
  sorry

end NUMINAMATH_CALUDE_medium_lights_count_l3648_364855


namespace NUMINAMATH_CALUDE_gcd_8164_2937_l3648_364804

theorem gcd_8164_2937 : Nat.gcd 8164 2937 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8164_2937_l3648_364804


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3648_364828

def f (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 3*x^2 - 8*x + 15

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder : f 4 = 543 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3648_364828


namespace NUMINAMATH_CALUDE_special_triangle_sides_l3648_364829

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℕ+
  b : ℕ+
  c : ℕ+
  -- The perimeter is a natural number (implied by sides being natural numbers)
  perimeter_nat : (a + b + c : ℕ) > 0
  -- The circumradius is 65/8
  circumradius_eq : (a * b * c : ℚ) / (4 * (a + b + c : ℚ)) = 65 / 8
  -- The inradius is 4
  inradius_eq : (a * b * c : ℚ) / ((a + b + c : ℚ) * (a + b + c - 2 * min a (min b c))) = 4

/-- The sides of the special triangle are (13, 14, 15) -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 13 ∧ t.b = 14 ∧ t.c = 15 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_sides_l3648_364829


namespace NUMINAMATH_CALUDE_fifth_derivative_y_l3648_364822

noncomputable def y (x : ℝ) : ℝ := (2 * x^2 - 7) * Real.log (x - 1)

theorem fifth_derivative_y (x : ℝ) (h : x ≠ 1) :
  (deriv^[5] y) x = 8 * (x^2 - 5*x - 11) / (x - 1)^5 :=
by sorry

end NUMINAMATH_CALUDE_fifth_derivative_y_l3648_364822


namespace NUMINAMATH_CALUDE_max_PXQ_value_l3648_364890

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def starts_with (n : ℕ) (d : ℕ) : Prop :=
  (n / 100) = d

def ends_with (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d

theorem max_PXQ_value :
  ∀ XX X PXQ : ℕ,
    is_two_digit_with_equal_digits XX →
    is_one_digit X →
    is_three_digit PXQ →
    XX * X = PXQ →
    starts_with PXQ (PXQ / 100) →
    ends_with PXQ X →
    PXQ ≤ 396 :=
sorry

end NUMINAMATH_CALUDE_max_PXQ_value_l3648_364890


namespace NUMINAMATH_CALUDE_xyz_equals_four_l3648_364865

theorem xyz_equals_four (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_equals_four_l3648_364865


namespace NUMINAMATH_CALUDE_square_diff_product_plus_square_equals_five_l3648_364891

theorem square_diff_product_plus_square_equals_five 
  (a b : ℝ) (ha : a = Real.sqrt 2 + 1) (hb : b = Real.sqrt 2 - 1) : 
  a^2 - a*b + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_square_diff_product_plus_square_equals_five_l3648_364891


namespace NUMINAMATH_CALUDE_grid_path_theorem_l3648_364883

/-- Represents a closed path on a grid that is not self-intersecting -/
structure GridPath (m n : ℕ) where
  -- Add necessary fields to represent the path

/-- Counts the number of points on the path where it does not turn -/
def count_no_turn_points (p : GridPath m n) : ℕ := sorry

/-- Counts the number of squares that the path goes through two non-adjacent sides -/
def count_two_side_squares (p : GridPath m n) : ℕ := sorry

/-- Counts the number of squares with no side in the path -/
def count_empty_squares (p : GridPath m n) : ℕ := sorry

theorem grid_path_theorem {m n : ℕ} (hm : m ≥ 4) (hn : n ≥ 4) (p : GridPath m n) :
  count_no_turn_points p = count_two_side_squares p - count_empty_squares p + m + n - 1 := by
  sorry

end NUMINAMATH_CALUDE_grid_path_theorem_l3648_364883


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3648_364803

theorem right_triangle_hypotenuse (a b c : ℕ) : 
  a * a + b * b = c * c →  -- Pythagorean theorem
  c - b = 1575 →           -- One leg is 1575 units shorter than hypotenuse
  a < 1991 →               -- The other leg is less than 1991 units
  c = 1799 :=              -- The hypotenuse length is 1799
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3648_364803


namespace NUMINAMATH_CALUDE_three_tangent_lines_m_values_l3648_364823

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 2*x^2 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 4*x - 3

-- Define the tangent line equation
def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := 
  f x₀ + f' x₀ * (x - x₀)

-- Define the condition for a point to be on the tangent line
def on_tangent_line (x₀ m : ℝ) : Prop :=
  m = tangent_line x₀ (-1)

-- Theorem statement
theorem three_tangent_lines_m_values :
  ∀ m : ℤ, (∃ x₁ x₂ x₃ : ℝ, 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    on_tangent_line x₁ m ∧
    on_tangent_line x₂ m ∧
    on_tangent_line x₃ m) →
  m = 4 ∨ m = 5 :=
sorry

end NUMINAMATH_CALUDE_three_tangent_lines_m_values_l3648_364823


namespace NUMINAMATH_CALUDE_square_root_equation_l3648_364889

theorem square_root_equation (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l3648_364889


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_nonpositive_l3648_364879

/-- The system of equations has at most one real solution if and only if a ≤ 0 -/
theorem unique_solution_iff_a_nonpositive (a : ℝ) :
  (∃! x y z : ℝ, x^4 = y*z - x^2 + a ∧ y^4 = z*x - y^2 + a ∧ z^4 = x*y - z^2 + a) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_nonpositive_l3648_364879


namespace NUMINAMATH_CALUDE_parallel_lines_length_l3648_364864

-- Define the parallel lines and their lengths
def AB : ℝ := 210
def CD : ℝ := 140
def EF : ℝ := 84

-- Define the parallel relation
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- State the theorem
theorem parallel_lines_length :
  ∀ (ab gh cd ef : ℝ → ℝ → Prop),
    parallel ab gh → parallel gh cd → parallel cd ef →
    AB = 210 → CD = 140 →
    EF = 84 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l3648_364864


namespace NUMINAMATH_CALUDE_banana_basket_count_l3648_364862

theorem banana_basket_count (total_baskets : ℕ) (average_fruits : ℕ) 
  (basket_a : ℕ) (basket_b : ℕ) (basket_c : ℕ) (basket_d : ℕ) :
  total_baskets = 5 →
  average_fruits = 25 →
  basket_a = 15 →
  basket_b = 30 →
  basket_c = 20 →
  basket_d = 25 →
  (total_baskets * average_fruits) - (basket_a + basket_b + basket_c + basket_d) = 35 :=
by sorry

end NUMINAMATH_CALUDE_banana_basket_count_l3648_364862


namespace NUMINAMATH_CALUDE_valid_three_digit_count_l3648_364897

/-- The count of three-digit numbers without exactly two identical adjacent digits -/
def valid_three_digit_numbers : ℕ := 738

/-- The total count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical adjacent digits -/
def invalid_three_digit_numbers : ℕ := 162

theorem valid_three_digit_count :
  valid_three_digit_numbers = total_three_digit_numbers - invalid_three_digit_numbers :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_count_l3648_364897


namespace NUMINAMATH_CALUDE_excess_hour_cost_correct_l3648_364882

/-- The cost per hour in excess of 2 hours for a parking garage -/
def excess_hour_cost : ℝ := 1.75

/-- The cost to park for up to 2 hours -/
def initial_cost : ℝ := 15

/-- The average cost per hour to park for 9 hours -/
def average_cost_9_hours : ℝ := 3.0277777777777777

/-- Theorem stating that the excess hour cost is correct given the initial cost and average cost -/
theorem excess_hour_cost_correct : 
  (initial_cost + 7 * excess_hour_cost) / 9 = average_cost_9_hours :=
by sorry

end NUMINAMATH_CALUDE_excess_hour_cost_correct_l3648_364882


namespace NUMINAMATH_CALUDE_ratio_of_powers_l3648_364838

theorem ratio_of_powers (p q : ℝ) (n : ℕ) (h1 : n > 1) (h2 : p^n / q^n = 7) :
  (p^n + q^n) / (p^n - q^n) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_powers_l3648_364838


namespace NUMINAMATH_CALUDE_job_completion_time_l3648_364837

/-- The number of days it takes for B to do the job alone -/
def B_days : ℕ := 30

/-- The number of days it takes for A and B to do 4 times the job together -/
def AB_days : ℕ := 72

/-- The number of days it takes for A to do the job alone -/
def A_days : ℕ := 45

theorem job_completion_time :
  (1 : ℚ) / A_days + (1 : ℚ) / B_days = 4 / AB_days :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3648_364837


namespace NUMINAMATH_CALUDE_initial_overs_is_ten_l3648_364858

/-- Represents a cricket game scenario --/
structure CricketGame where
  target : ℕ
  initialRunRate : ℚ
  remainingOvers : ℕ
  requiredRunRate : ℚ

/-- Calculates the number of overs played initially in a cricket game --/
def initialOvers (game : CricketGame) : ℚ :=
  (game.target - game.remainingOvers * game.requiredRunRate) / game.initialRunRate

/-- Theorem stating that the number of overs played initially is 10 --/
theorem initial_overs_is_ten (game : CricketGame) 
  (h1 : game.target = 282)
  (h2 : game.initialRunRate = 16/5)
  (h3 : game.remainingOvers = 50)
  (h4 : game.requiredRunRate = 5)
  : initialOvers game = 10 := by
  sorry

#eval initialOvers { target := 282, initialRunRate := 16/5, remainingOvers := 50, requiredRunRate := 5 }

end NUMINAMATH_CALUDE_initial_overs_is_ten_l3648_364858


namespace NUMINAMATH_CALUDE_sin_45_degrees_l3648_364887

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l3648_364887


namespace NUMINAMATH_CALUDE_point_p_coordinates_l3648_364809

/-- A point in the fourth quadrant with specific distances from axes -/
structure PointP where
  x : ℝ
  y : ℝ
  in_fourth_quadrant : x > 0 ∧ y < 0
  distance_to_x_axis : |y| = 1
  distance_to_y_axis : |x| = 2

/-- The coordinates of point P are (2, -1) -/
theorem point_p_coordinates (p : PointP) : p.x = 2 ∧ p.y = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l3648_364809


namespace NUMINAMATH_CALUDE_product_327_3_base9_l3648_364811

/-- Represents a number in base 9 --/
def Base9 := ℕ

/-- Converts a base 9 number to a natural number --/
def to_nat (x : Base9) : ℕ := sorry

/-- Converts a natural number to a base 9 number --/
def from_nat (x : ℕ) : Base9 := sorry

/-- Multiplies two base 9 numbers --/
def mul_base9 (x y : Base9) : Base9 := sorry

theorem product_327_3_base9 : 
  mul_base9 (from_nat 327) (from_nat 3) = from_nat 1083 := by sorry

end NUMINAMATH_CALUDE_product_327_3_base9_l3648_364811


namespace NUMINAMATH_CALUDE_cube_sum_plus_three_l3648_364825

theorem cube_sum_plus_three (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 3 = 973 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_plus_three_l3648_364825


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3648_364832

def S : Set ℝ := {x | (x - 2) * (x + 3) > 0}

def T : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}

theorem intersection_equals_interval : S ∩ T = Set.Ioo 2 3 ∪ Set.singleton 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3648_364832


namespace NUMINAMATH_CALUDE_rainy_day_probability_l3648_364867

theorem rainy_day_probability (A B : Set ℝ) (P : Set ℝ → ℝ) 
  (hA : P A = 0.06)
  (hB : P B = 0.08)
  (hAB : P (A ∩ B) = 0.02) :
  P B / P A = 1/3 :=
sorry

end NUMINAMATH_CALUDE_rainy_day_probability_l3648_364867


namespace NUMINAMATH_CALUDE_cats_favorite_number_l3648_364807

def is_two_digit_positive (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def has_distinct_nonzero_digits (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones ∧ tens ≠ 0 ∧ ones ≠ 0

def digits_are_factors (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  n % tens = 0 ∧ n % ones = 0

def satisfies_four_number_property (a b c d : ℕ) : Prop :=
  a + b - c = d ∧ b + c - a = d ∧ c + d - b = a ∧ d + a - c = b

theorem cats_favorite_number :
  ∃! n : ℕ,
    is_two_digit_positive n ∧
    has_distinct_nonzero_digits n ∧
    digits_are_factors n ∧
    ∃ a b c : ℕ,
      satisfies_four_number_property n a b c ∧
      n^2 = a * b ∧
      (a ≠ n ∧ b ≠ n ∧ c ≠ n) :=
by
  sorry

end NUMINAMATH_CALUDE_cats_favorite_number_l3648_364807


namespace NUMINAMATH_CALUDE_jennifer_initial_oranges_l3648_364846

/-- The number of fruits Jennifer has initially and after giving some away. -/
structure FruitCount where
  initial_pears : ℕ
  initial_apples : ℕ
  initial_oranges : ℕ
  pears_left : ℕ
  apples_left : ℕ
  oranges_left : ℕ
  total_left : ℕ

/-- Theorem stating the number of oranges Jennifer had initially. -/
theorem jennifer_initial_oranges (f : FruitCount) 
  (h1 : f.initial_pears = 10)
  (h2 : f.initial_apples = 2 * f.initial_pears)
  (h3 : f.pears_left = f.initial_pears - 2)
  (h4 : f.apples_left = f.initial_apples - 2)
  (h5 : f.oranges_left = f.initial_oranges - 2)
  (h6 : f.total_left = 44)
  (h7 : f.total_left = f.pears_left + f.apples_left + f.oranges_left) :
  f.initial_oranges = 20 := by
  sorry


end NUMINAMATH_CALUDE_jennifer_initial_oranges_l3648_364846


namespace NUMINAMATH_CALUDE_roger_candies_left_l3648_364826

/-- The number of candies Roger has left after giving some away -/
def candies_left (initial : ℕ) (given_to_stephanie : ℕ) (given_to_john : ℕ) (given_to_emily : ℕ) : ℕ :=
  initial - (given_to_stephanie + given_to_john + given_to_emily)

/-- Theorem stating that Roger has 262 candies left -/
theorem roger_candies_left :
  candies_left 350 45 25 18 = 262 := by
  sorry

end NUMINAMATH_CALUDE_roger_candies_left_l3648_364826


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3648_364800

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ (m > 2 ∨ m < -4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3648_364800


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_minimum_value_l3648_364850

-- Define the function f
def f (m n x : ℝ) : ℝ := |x - m| + |x - n|

-- Part 1
theorem part1_solution_set (x : ℝ) :
  (f 2 (-5) x > 9) ↔ (x < -6 ∨ x > 3) := by sorry

-- Part 2
theorem part2_minimum_value (a : ℝ) (h : a ≠ 0) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x : ℝ), f a (-1/a) x ≥ min := by sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_minimum_value_l3648_364850


namespace NUMINAMATH_CALUDE_sum_of_ages_l3648_364869

theorem sum_of_ages (marie_age marco_age : ℕ) : 
  marie_age = 12 → 
  marco_age = 2 * marie_age + 1 → 
  marie_age + marco_age = 37 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3648_364869


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l3648_364805

/-- Given a line passing through points (1, 3) and (3, 7), prove that m + b = 3 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l3648_364805


namespace NUMINAMATH_CALUDE_gcd_7_factorial_5_factorial_squared_l3648_364893

theorem gcd_7_factorial_5_factorial_squared : Nat.gcd (Nat.factorial 7) ((Nat.factorial 5)^2) = 720 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7_factorial_5_factorial_squared_l3648_364893


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l3648_364875

-- Define the lines L1 and L2
def L1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def L2 (x y : ℝ) : Prop := x - 3 * y + 8 = 0

-- Define the reference line
def ref_line (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := ((-2 : ℝ), (2 : ℝ))

-- Theorem for the parallel line
theorem parallel_line_equation :
  ∀ (x y : ℝ), 2 * x + y + 2 = 0 →
  (L1 (M.1) (M.2) ∧ L2 (M.1) (M.2)) ∧
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), 2 * x + y + 2 = 0 ↔ k * (2 * x + y + 5) = 0 :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line_equation :
  ∀ (x y : ℝ), x - 2 * y + 6 = 0 →
  (L1 (M.1) (M.2) ∧ L2 (M.1) (M.2)) ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x - 2 * y + 6 = 0 → x₁ = x ∧ y₁ = y) →
    (2 * x + y + 5 = 0 → x₂ = x ∧ y₂ = y) →
    (x₂ - x₁) * (x - x₁) + (y₂ - y₁) * (y - y₁) = 0 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l3648_364875


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3648_364857

theorem sum_of_cubes (x y z : ℕ+) : 
  (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 378 → x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3648_364857


namespace NUMINAMATH_CALUDE_jaeho_received_most_notebooks_l3648_364873

def notebooks_given : ℕ := 30
def jaehyuk_notebooks : ℕ := 12
def kyunghwan_notebooks : ℕ := 3
def jaeho_notebooks : ℕ := 15

theorem jaeho_received_most_notebooks :
  jaeho_notebooks > jaehyuk_notebooks ∧ jaeho_notebooks > kyunghwan_notebooks :=
by sorry

end NUMINAMATH_CALUDE_jaeho_received_most_notebooks_l3648_364873


namespace NUMINAMATH_CALUDE_two_special_numbers_exist_l3648_364842

theorem two_special_numbers_exist : ∃ (x y : ℕ), 
  x + y = 2013 ∧ 
  y = 5 * ((x / 100) + 1) ∧ 
  x > y :=
by sorry

end NUMINAMATH_CALUDE_two_special_numbers_exist_l3648_364842


namespace NUMINAMATH_CALUDE_blueberry_zucchini_trade_l3648_364845

/-- The number of bushes needed to obtain a specific number of zucchinis -/
def bushes_needed (total_containers_per_bush : ℕ) (containers_for_jam : ℕ) 
                  (containers_per_trade : ℕ) (zucchinis_per_trade : ℕ) 
                  (target_zucchinis : ℕ) : ℕ :=
  let usable_containers := total_containers_per_bush - containers_for_jam
  let zucchinis_per_container := zucchinis_per_trade / containers_per_trade
  let zucchinis_per_bush := usable_containers * zucchinis_per_container
  target_zucchinis / zucchinis_per_bush

/-- Theorem stating that 18 bushes are needed to obtain 72 zucchinis under given conditions -/
theorem blueberry_zucchini_trade : bushes_needed 10 2 6 3 72 = 18 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_zucchini_trade_l3648_364845


namespace NUMINAMATH_CALUDE_yogurt_combinations_l3648_364836

/- Define the number of flavors and toppings -/
def num_flavors : ℕ := 4
def num_toppings : ℕ := 8

/- Define the function to calculate combinations -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/- Theorem statement -/
theorem yogurt_combinations :
  let no_topping := 1
  let two_toppings := choose num_toppings 2
  let combinations_per_flavor := no_topping + two_toppings
  num_flavors * combinations_per_flavor = 116 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l3648_364836


namespace NUMINAMATH_CALUDE_only_set_A_forms_triangle_l3648_364885

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the sets of line segments
def set_A : List ℝ := [5, 6, 10]
def set_B : List ℝ := [5, 2, 9]
def set_C : List ℝ := [5, 7, 12]
def set_D : List ℝ := [3, 4, 8]

-- Theorem statement
theorem only_set_A_forms_triangle :
  (can_form_triangle 5 6 10) ∧
  ¬(can_form_triangle 5 2 9) ∧
  ¬(can_form_triangle 5 7 12) ∧
  ¬(can_form_triangle 3 4 8) :=
sorry

end NUMINAMATH_CALUDE_only_set_A_forms_triangle_l3648_364885


namespace NUMINAMATH_CALUDE_leahs_coins_value_l3648_364854

/-- Represents the number of coins Leah has -/
def total_coins : ℕ := 15

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents Leah's coin collection -/
structure CoinCollection where
  nickels : ℕ
  pennies : ℕ

/-- The conditions of Leah's coin collection -/
def valid_collection (c : CoinCollection) : Prop :=
  c.nickels + c.pennies = total_coins ∧
  c.nickels + 1 = c.pennies

/-- The total value of a coin collection in cents -/
def collection_value (c : CoinCollection) : ℕ :=
  c.nickels * nickel_value + c.pennies * penny_value

/-- The main theorem stating that Leah's coins are worth 43 cents -/
theorem leahs_coins_value (c : CoinCollection) :
  valid_collection c → collection_value c = 43 := by
  sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l3648_364854


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3648_364833

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 16) and (2, -8) is 9 -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 8
  let y1 : ℝ := 16
  let x2 : ℝ := 2
  let y2 : ℝ := -8
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3648_364833


namespace NUMINAMATH_CALUDE_students_on_left_side_l3648_364872

theorem students_on_left_side (total : ℕ) (right : ℕ) (h1 : total = 63) (h2 : right = 27) :
  total - right = 36 := by
  sorry

end NUMINAMATH_CALUDE_students_on_left_side_l3648_364872


namespace NUMINAMATH_CALUDE_sum_of_seven_step_palindromes_l3648_364870

/-- Reverses a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool := sorry

/-- Performs one step of reversing and adding -/
def reverseAndAdd (n : ℕ) : ℕ := n + reverseNum n

/-- Checks if a number becomes a palindrome after exactly k steps -/
def isPalindromeAfterKSteps (n : ℕ) (k : ℕ) : Bool := sorry

/-- The set of three-digit numbers that become palindromes after exactly 7 steps -/
def sevenStepPalindromes : Finset ℕ := sorry

theorem sum_of_seven_step_palindromes :
  Finset.sum sevenStepPalindromes id = 1160 := by sorry

end NUMINAMATH_CALUDE_sum_of_seven_step_palindromes_l3648_364870


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l3648_364876

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (intersectPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_same_line
  (m n : Line) (α β : Plane)
  (h1 : ¬ intersect m n)
  (h2 : ¬ intersectPlanes α β)
  (h3 : perpendicular m α)
  (h4 : perpendicular m β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l3648_364876


namespace NUMINAMATH_CALUDE_f_2006_equals_1_l3648_364820

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2006_equals_1 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period (fun x ↦ f (3*x + 1)) 3)
  (h_f_1 : f 1 = -1) :
  f 2006 = 1 := by
sorry

end NUMINAMATH_CALUDE_f_2006_equals_1_l3648_364820


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l3648_364819

/-- Given a polynomial g(x) = ax^2 + bx^3 + cx + d where g(-3) = 2, prove that g(3) = 0 -/
theorem polynomial_symmetry (a b c d : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = a * x^2 + b * x^3 + c * x + d)
  (h2 : g (-3) = 2) : 
  g 3 = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l3648_364819


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l3648_364886

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 + 3*x - 4

-- Define the derivative of f(x+1)
def f'_shifted (x : ℝ) : ℝ := (x + 1)^2 + 3*(x + 1) - 4

-- Theorem statement
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-5 : ℝ) 0, f'_shifted x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l3648_364886


namespace NUMINAMATH_CALUDE_max_value_product_sum_l3648_364888

theorem max_value_product_sum (A M C : ℕ) (sum_constraint : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 ∧
  ∃ (A' M' C' : ℕ), A' + M' + C' = 15 ∧ A' * M' * C' + A' * M' + M' * C' + C' * A' = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l3648_364888


namespace NUMINAMATH_CALUDE_subset_condition_l3648_364835

theorem subset_condition (a : ℝ) : 
  ({x : ℝ | 1 ≤ x ∧ x ≤ 2} ⊆ {x : ℝ | a < x}) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3648_364835


namespace NUMINAMATH_CALUDE_trailing_zeros_of_main_expression_l3648_364843

/-- The number of trailing zeros in n -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Prime factorization of 15 -/
def fifteen : ℕ := 3 * 5

/-- Prime factorization of 28 -/
def twentyEight : ℕ := 2^2 * 7

/-- Prime factorization of 55 -/
def fiftyFive : ℕ := 5 * 11

/-- The main expression -/
def mainExpression : ℕ := fifteen^6 * twentyEight^5 * fiftyFive^7

theorem trailing_zeros_of_main_expression :
  trailingZeros mainExpression = 10 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_main_expression_l3648_364843
