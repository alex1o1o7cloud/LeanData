import Mathlib

namespace NUMINAMATH_CALUDE_first_cat_blue_eyed_count_l1393_139347

/-- The number of blue-eyed kittens in the first cat's litter -/
def blue_eyed_first_cat : ℕ := sorry

/-- The number of brown-eyed kittens in the first cat's litter -/
def brown_eyed_first_cat : ℕ := 7

/-- The number of blue-eyed kittens in the second cat's litter -/
def blue_eyed_second_cat : ℕ := 4

/-- The number of brown-eyed kittens in the second cat's litter -/
def brown_eyed_second_cat : ℕ := 6

/-- The percentage of blue-eyed kittens among all kittens -/
def blue_eyed_percentage : ℚ := 35 / 100

theorem first_cat_blue_eyed_count :
  blue_eyed_first_cat = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_first_cat_blue_eyed_count_l1393_139347


namespace NUMINAMATH_CALUDE_ant_beetle_distance_difference_l1393_139364

/-- Calculates the percentage difference in distance traveled between an ant and a beetle -/
theorem ant_beetle_distance_difference :
  let ant_distance : ℝ := 600  -- meters
  let ant_time : ℝ := 12       -- minutes
  let beetle_speed : ℝ := 2.55 -- km/h
  
  let ant_speed : ℝ := (ant_distance / 1000) / (ant_time / 60)
  let beetle_distance : ℝ := (beetle_speed * ant_time) / 60 * 1000
  
  let difference : ℝ := ant_distance - beetle_distance
  let percentage_difference : ℝ := (difference / ant_distance) * 100
  
  percentage_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_ant_beetle_distance_difference_l1393_139364


namespace NUMINAMATH_CALUDE_h_bounds_l1393_139336

/-- The probability that the distance between two randomly chosen points on (0,1) is less than h -/
def probability (h : ℝ) : ℝ := h * (2 - h)

/-- Theorem stating the bounds of h given the probability constraints -/
theorem h_bounds (h : ℝ) (h_pos : 0 < h) (h_lt_one : h < 1) 
  (prob_lower : 1/4 < probability h) (prob_upper : probability h < 3/4) : 
  1/2 - Real.sqrt 3 / 2 < h ∧ h < 1/2 + Real.sqrt 3 / 2 := by
  sorry

#check h_bounds

end NUMINAMATH_CALUDE_h_bounds_l1393_139336


namespace NUMINAMATH_CALUDE_shelf_position_l1393_139359

theorem shelf_position (wall_width : ℝ) (picture_width : ℝ) 
  (hw : wall_width = 26)
  (hp : picture_width = 4) :
  let picture_center := wall_width / 2
  let shelf_left_edge := picture_center + picture_width / 2
  shelf_left_edge = 15 := by
  sorry

end NUMINAMATH_CALUDE_shelf_position_l1393_139359


namespace NUMINAMATH_CALUDE_sin_double_angle_with_tan_l1393_139368

theorem sin_double_angle_with_tan (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_sin_double_angle_with_tan_l1393_139368


namespace NUMINAMATH_CALUDE_det_roots_cubic_l1393_139351

theorem det_roots_cubic (p q r a b c : ℝ) : 
  (a^3 - p*a^2 + q*a - r = 0) →
  (b^3 - p*b^2 + q*b - r = 0) →
  (c^3 - p*c^2 + q*c - r = 0) →
  let matrix := !![2 + a, 1, 1; 1, 2 + b, 1; 1, 1, 2 + c]
  Matrix.det matrix = r + 2*q + 4*p + 4 := by
  sorry

end NUMINAMATH_CALUDE_det_roots_cubic_l1393_139351


namespace NUMINAMATH_CALUDE_team_supporters_equal_positive_responses_l1393_139363

-- Define the four teams
inductive Team
| Spartak
| Dynamo
| Zenit
| Lokomotiv

-- Define the result of a match
inductive MatchResult
| Win
| Lose

-- Define a function to represent fan behavior
def fanResponse (team : Team) (result : MatchResult) : Bool :=
  match result with
  | MatchResult.Win => true
  | MatchResult.Lose => false

-- Define the theorem
theorem team_supporters_equal_positive_responses 
  (match1 : Team → MatchResult) 
  (match2 : Team → MatchResult)
  (positiveResponses : Team → Nat)
  (h1 : ∀ t, (match1 t = MatchResult.Win) ≠ (match2 t = MatchResult.Win))
  (h2 : positiveResponses Team.Spartak = 200)
  (h3 : positiveResponses Team.Dynamo = 300)
  (h4 : positiveResponses Team.Zenit = 500)
  (h5 : positiveResponses Team.Lokomotiv = 600)
  : ∀ t, positiveResponses t = 
    (if fanResponse t (match1 t) then 1 else 0) + 
    (if fanResponse t (match2 t) then 1 else 0) := by
  sorry


end NUMINAMATH_CALUDE_team_supporters_equal_positive_responses_l1393_139363


namespace NUMINAMATH_CALUDE_stationery_box_sheet_count_l1393_139360

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents the usage of a stationery box -/
structure Usage where
  sheetsPerLetter : ℕ
  usedAllEnvelopes : Bool
  usedAllSheets : Bool
  leftoverSheets : ℕ
  leftoverEnvelopes : ℕ

theorem stationery_box_sheet_count (box : StationeryBox) 
  (ann_usage : Usage) (bob_usage : Usage) :
  ann_usage.sheetsPerLetter = 2 →
  bob_usage.sheetsPerLetter = 4 →
  ann_usage.usedAllEnvelopes = true →
  ann_usage.leftoverSheets = 30 →
  bob_usage.usedAllSheets = true →
  bob_usage.leftoverEnvelopes = 20 →
  box.sheets = 40 := by
sorry

end NUMINAMATH_CALUDE_stationery_box_sheet_count_l1393_139360


namespace NUMINAMATH_CALUDE_equation_solution_l1393_139367

theorem equation_solution :
  ∃ x : ℝ, (x^2 + x ≠ 0 ∧ x^2 - x ≠ 0) ∧
  (4 / (x^2 + x) - 3 / (x^2 - x) = 0) ∧
  x = 7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1393_139367


namespace NUMINAMATH_CALUDE_candy_probability_l1393_139301

/-- The probability of picking specific candies from a bag --/
theorem candy_probability : 
  let green : ℕ := 8
  let blue : ℕ := 5
  let red : ℕ := 9
  let yellow : ℕ := 10
  let pink : ℕ := 6
  let total : ℕ := green + blue + red + yellow + pink
  
  -- Probability of picking green first
  let p_green : ℚ := green / total
  
  -- Probability of picking yellow second
  let p_yellow : ℚ := yellow / (total - 1)
  
  -- Probability of picking pink third
  let p_pink : ℚ := pink / (total - 2)
  
  -- Overall probability
  let probability : ℚ := p_green * p_yellow * p_pink
  
  probability = 20 / 2109 := by
  sorry

end NUMINAMATH_CALUDE_candy_probability_l1393_139301


namespace NUMINAMATH_CALUDE_inverse_function_property_l1393_139378

-- Define a function f and its inverse f_inv
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property that f and f_inv are inverse functions
def are_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define the property that f(x+2) and f_inv(x-1) are inverse functions
def special_inverse_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f ((f_inv (x - 1)) + 2) = x ∧ f_inv (f (x + 2) - 1) = x

-- State the theorem
theorem inverse_function_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h1 : are_inverse f f_inv) 
  (h2 : special_inverse_property f f_inv) : 
  f_inv 2007 - f_inv 1 = 4012 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l1393_139378


namespace NUMINAMATH_CALUDE_johns_country_club_payment_l1393_139328

/-- Represents the cost John pays for the country club membership in the first year -/
def johns_payment (num_members : ℕ) (joining_fee_pp : ℕ) (monthly_cost_pp : ℕ) : ℕ :=
  let total_joining_fee := num_members * joining_fee_pp
  let total_monthly_cost := num_members * monthly_cost_pp * 12
  let total_cost := total_joining_fee + total_monthly_cost
  total_cost / 2

/-- Proves that John's payment for the first year is $32000 given the problem conditions -/
theorem johns_country_club_payment :
  johns_payment 4 4000 1000 = 32000 := by
sorry

end NUMINAMATH_CALUDE_johns_country_club_payment_l1393_139328


namespace NUMINAMATH_CALUDE_cube_surface_area_l1393_139314

/-- The surface area of a cube with edge length 8 cm is 384 cm². -/
theorem cube_surface_area : 
  let edge_length : ℝ := 8
  let surface_area : ℝ := 6 * edge_length * edge_length
  surface_area = 384 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1393_139314


namespace NUMINAMATH_CALUDE_total_cost_is_119_l1393_139313

-- Define the number of games and ticket prices for each month
def this_month_games : ℕ := 9
def this_month_price : ℕ := 5
def last_month_games : ℕ := 8
def last_month_price : ℕ := 4
def next_month_games : ℕ := 7
def next_month_price : ℕ := 6

-- Define the total cost function
def total_cost : ℕ :=
  this_month_games * this_month_price +
  last_month_games * last_month_price +
  next_month_games * next_month_price

-- Theorem statement
theorem total_cost_is_119 : total_cost = 119 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_119_l1393_139313


namespace NUMINAMATH_CALUDE_total_bills_is_30_l1393_139379

/-- Represents the number of $10 bills -/
def num_ten_bills : ℕ := 27

/-- Represents the number of $20 bills -/
def num_twenty_bills : ℕ := 3

/-- Represents the total value of all bills in dollars -/
def total_value : ℕ := 330

/-- Theorem stating that the total number of bills is 30 -/
theorem total_bills_is_30 : num_ten_bills + num_twenty_bills = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_bills_is_30_l1393_139379


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1393_139352

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 40 ∧ x - y = 10 → x * y = 375 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1393_139352


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_minus_b_l1393_139355

noncomputable def f (a b x : ℝ) : ℝ := x + a / x + b

theorem tangent_line_implies_a_minus_b (a b : ℝ) :
  (∀ x ≠ 0, HasDerivAt (f a b) (1 - a / (x^2)) x) →
  (f a b 1 = 1 + a + b) →
  (HasDerivAt (f a b) (-2) 1) →
  (∃ c, ∀ x, f a b x = -2 * x + c) →
  a - b = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_minus_b_l1393_139355


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1393_139348

theorem simple_interest_rate_calculation 
  (initial_sum : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : initial_sum = 12500)
  (h2 : final_amount = 15500)
  (h3 : time = 4)
  (h4 : final_amount = initial_sum * (1 + time * (rate / 100))) :
  rate = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1393_139348


namespace NUMINAMATH_CALUDE_sum_10_is_35_l1393_139369

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  third_term : 2 * a 3 = 5
  sum_4_12 : a 4 + a 12 = 9

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1))

/-- The theorem to be proved -/
theorem sum_10_is_35 (seq : ArithmeticSequence) : sum_n seq 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_10_is_35_l1393_139369


namespace NUMINAMATH_CALUDE_hide_and_seek_l1393_139306

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem statement
theorem hide_and_seek : 
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena := by sorry

end NUMINAMATH_CALUDE_hide_and_seek_l1393_139306


namespace NUMINAMATH_CALUDE_fly_can_always_escape_l1393_139397

/-- Represents a bug (fly or spider) in the octahedron -/
structure Bug where
  position : ℝ × ℝ × ℝ
  speed : ℝ

/-- Represents the octahedron -/
structure Octahedron where
  vertices : List (ℝ × ℝ × ℝ)
  edges : List ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ))

/-- Represents the state of the chase -/
structure ChaseState where
  octahedron : Octahedron
  fly : Bug
  spiders : List Bug

/-- Function to determine if the fly can escape -/
def canFlyEscape (state : ChaseState) : Prop :=
  ∃ (nextPosition : ℝ × ℝ × ℝ), nextPosition ∈ state.octahedron.vertices ∧
    ∀ (spider : Bug), spider ∈ state.spiders →
      ‖spider.position - nextPosition‖ > ‖state.fly.position - nextPosition‖ * (spider.speed / state.fly.speed)

/-- The main theorem -/
theorem fly_can_always_escape (r : ℝ) (h : r < 25) :
  ∀ (state : ChaseState),
    state.fly.speed = 50 ∧
    (∀ spider ∈ state.spiders, spider.speed = r) ∧
    state.fly.position ∈ state.octahedron.vertices ∧
    state.spiders.length = 3 →
    canFlyEscape state :=
  sorry

end NUMINAMATH_CALUDE_fly_can_always_escape_l1393_139397


namespace NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l1393_139319

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

-- Theorem statement
theorem intersection_complement_equals_singleton :
  N ∩ (U \ M) = {(2, 3)} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l1393_139319


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1393_139356

/-- Proves that given a 20% reduction in the price of oil, if a housewife can obtain 10 kgs more for Rs. 1500 after the reduction, then the reduced price per kg is Rs. 30. -/
theorem oil_price_reduction (original_price : ℝ) : 
  (1500 / (0.8 * original_price) - 1500 / original_price = 10) → 
  (0.8 * original_price = 30) := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l1393_139356


namespace NUMINAMATH_CALUDE_two_vans_needed_l1393_139358

/-- The number of vans needed for a field trip -/
def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

/-- Proof that 2 vans are needed for the field trip -/
theorem two_vans_needed : vans_needed 4 2 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_vans_needed_l1393_139358


namespace NUMINAMATH_CALUDE_third_boy_age_l1393_139326

theorem third_boy_age (total_age : ℕ) (age_two_boys : ℕ) (num_boys : ℕ) :
  total_age = 29 →
  age_two_boys = 9 →
  num_boys = 3 →
  ∃ (third_boy_age : ℕ), third_boy_age = total_age - 2 * age_two_boys :=
by
  sorry

end NUMINAMATH_CALUDE_third_boy_age_l1393_139326


namespace NUMINAMATH_CALUDE_calculate_expression_l1393_139376

theorem calculate_expression : 3 * 7.5 * (6 + 4) / 2.5 = 90 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l1393_139376


namespace NUMINAMATH_CALUDE_probability_theorem_l1393_139318

/-- A regular hexagon --/
structure RegularHexagon where
  /-- The set of all sides and diagonals --/
  S : Finset ℝ
  /-- Number of sides --/
  num_sides : ℕ
  /-- Number of shorter diagonals --/
  num_shorter_diagonals : ℕ
  /-- Number of longer diagonals --/
  num_longer_diagonals : ℕ
  /-- Total number of segments --/
  total_segments : ℕ
  /-- Condition: num_sides = 6 --/
  sides_eq_six : num_sides = 6
  /-- Condition: num_shorter_diagonals = 6 --/
  shorter_diagonals_eq_six : num_shorter_diagonals = 6
  /-- Condition: num_longer_diagonals = 3 --/
  longer_diagonals_eq_three : num_longer_diagonals = 3
  /-- Condition: total_segments = num_sides + num_shorter_diagonals + num_longer_diagonals --/
  total_segments_eq_sum : total_segments = num_sides + num_shorter_diagonals + num_longer_diagonals

/-- The probability of selecting two segments of the same length --/
def probability_same_length (h : RegularHexagon) : ℚ :=
  33 / 105

/-- Theorem: The probability of selecting two segments of the same length is 33/105 --/
theorem probability_theorem (h : RegularHexagon) : 
  probability_same_length h = 33 / 105 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1393_139318


namespace NUMINAMATH_CALUDE_total_gift_cost_l1393_139341

def engagement_ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def diamond_bracelet_cost : ℕ := 2 * engagement_ring_cost

theorem total_gift_cost : engagement_ring_cost + car_cost + diamond_bracelet_cost = 14000 := by
  sorry

end NUMINAMATH_CALUDE_total_gift_cost_l1393_139341


namespace NUMINAMATH_CALUDE_perpendicular_line_polar_equation_l1393_139320

/-- The polar equation of a line perpendicular to the polar axis and passing through 
    the center of the circle ρ = 6cosθ -/
theorem perpendicular_line_polar_equation (ρ θ : ℝ) : 
  (ρ = 6 * Real.cos θ → ∃ c, c = 3 ∧ ρ * Real.cos θ = c) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_polar_equation_l1393_139320


namespace NUMINAMATH_CALUDE_pebble_ratio_l1393_139337

/-- Prove that the ratio of pebbles Lance threw to pebbles Candy threw is 3:1 -/
theorem pebble_ratio : 
  let candy_pebbles : ℕ := 4
  let lance_pebbles : ℕ := candy_pebbles + 8
  (lance_pebbles : ℚ) / (candy_pebbles : ℚ) = 3 := by sorry

end NUMINAMATH_CALUDE_pebble_ratio_l1393_139337


namespace NUMINAMATH_CALUDE_silver_knights_enchanted_fraction_l1393_139308

structure Kingdom where
  total_knights : ℕ
  silver_knights : ℕ
  gold_knights : ℕ
  enchanted_knights : ℕ
  enchanted_silver : ℕ
  enchanted_gold : ℕ

def is_valid_kingdom (k : Kingdom) : Prop :=
  k.silver_knights + k.gold_knights = k.total_knights ∧
  k.silver_knights = (3 * k.total_knights) / 8 ∧
  k.enchanted_knights = k.total_knights / 8 ∧
  k.enchanted_silver + k.enchanted_gold = k.enchanted_knights ∧
  3 * k.enchanted_gold * k.silver_knights = k.enchanted_silver * k.gold_knights

theorem silver_knights_enchanted_fraction (k : Kingdom) 
  (h : is_valid_kingdom k) : 
  (k.enchanted_silver : ℚ) / k.silver_knights = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_silver_knights_enchanted_fraction_l1393_139308


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1393_139332

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : (2*d)^2 + c*(2*d) + d = 0) : 
  c = (1 : ℝ) / 2 ∧ d = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1393_139332


namespace NUMINAMATH_CALUDE_product_perfect_square_l1393_139327

theorem product_perfect_square (nums : Finset ℕ) : 
  (nums.card = 17) →
  (∀ n ∈ nums, ∃ (a b c d : ℕ), n = 2^a * 3^b * 5^c * 7^d) →
  ∃ (n1 n2 : ℕ), n1 ∈ nums ∧ n2 ∈ nums ∧ n1 ≠ n2 ∧ ∃ (m : ℕ), n1 * n2 = m^2 :=
by sorry

end NUMINAMATH_CALUDE_product_perfect_square_l1393_139327


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_l1393_139377

theorem quadratic_inequality_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) →
  (0 ≤ a ∧ a ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_l1393_139377


namespace NUMINAMATH_CALUDE_function_is_periodic_l1393_139374

-- Define the function f and the constant a
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- State the theorem
theorem function_is_periodic
  (h1 : ∀ x, f x ≠ 0)
  (h2 : a > 0)
  (h3 : ∀ x, f (x - a) = 1 / f x) :
  ∀ x, f x = f (x + 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_function_is_periodic_l1393_139374


namespace NUMINAMATH_CALUDE_lemonade_syrup_parts_l1393_139375

/-- Given a solution with 8 parts water for every L parts lemonade syrup,
    prove that if removing 2.1428571428571423 parts and replacing with water
    results in 25% lemonade syrup, then L = 2.6666666666666665 -/
theorem lemonade_syrup_parts (L : ℝ) : 
  (L = 0.25 * (8 + L)) → L = 2.6666666666666665 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_syrup_parts_l1393_139375


namespace NUMINAMATH_CALUDE_bryan_uninterested_offer_is_one_l1393_139312

/-- Represents the record sale scenario --/
structure RecordSale where
  total_records : ℕ
  sammy_offer : ℚ
  bryan_offer_interested : ℚ
  bryan_interested_fraction : ℚ
  profit_difference : ℚ

/-- Calculates Bryan's offer for uninterested records --/
def bryan_uninterested_offer (sale : RecordSale) : ℚ :=
  let sammy_total := sale.total_records * sale.sammy_offer
  let bryan_interested_records := sale.total_records * sale.bryan_interested_fraction
  let bryan_uninterested_records := sale.total_records - bryan_interested_records
  let bryan_interested_total := bryan_interested_records * sale.bryan_offer_interested
  (sammy_total - bryan_interested_total - sale.profit_difference) / bryan_uninterested_records

/-- Theorem stating Bryan's offer for uninterested records is $1 --/
theorem bryan_uninterested_offer_is_one (sale : RecordSale)
    (h1 : sale.total_records = 200)
    (h2 : sale.sammy_offer = 4)
    (h3 : sale.bryan_offer_interested = 6)
    (h4 : sale.bryan_interested_fraction = 1/2)
    (h5 : sale.profit_difference = 100) :
    bryan_uninterested_offer sale = 1 := by
  sorry

end NUMINAMATH_CALUDE_bryan_uninterested_offer_is_one_l1393_139312


namespace NUMINAMATH_CALUDE_alcohol_concentration_correct_l1393_139340

/-- The concentration of alcohol in the container after n operations --/
def alcohol_concentration (n : ℕ) : ℚ :=
  (12 - 9 * (3/4)^(n-1)) / (32 - 9 * (3/4)^(n-1))

/-- The amount of water in the container after n operations --/
def water_amount (n : ℕ) : ℚ :=
  20/3 * (2/3)^(n-1)

/-- The amount of alcohol in the container after n operations --/
def alcohol_amount (n : ℕ) : ℚ :=
  4 * (2/3)^(n-1) - 6 * (1/2)^n

/-- The theorem stating that the alcohol_concentration function correctly calculates
    the concentration of alcohol in the container after n operations --/
theorem alcohol_concentration_correct (n : ℕ) :
  alcohol_concentration n = alcohol_amount n / (water_amount n + alcohol_amount n) :=
by sorry

/-- The initial amount of water in the container --/
def initial_water : ℚ := 10

/-- The amount of alcohol added in the first step --/
def first_alcohol_addition : ℚ := 1

/-- The amount of alcohol added in the second step --/
def second_alcohol_addition : ℚ := 1/2

/-- The fraction of liquid removed in each step --/
def removal_fraction : ℚ := 1/3

/-- The ratio of alcohol added in each step compared to the previous step --/
def alcohol_addition_ratio : ℚ := 1/2

end NUMINAMATH_CALUDE_alcohol_concentration_correct_l1393_139340


namespace NUMINAMATH_CALUDE_larger_number_proof_l1393_139390

theorem larger_number_proof (L S : ℕ) (hL : L > S) : 
  L - S = 1000 → L = 10 * S + 10 → L = 1110 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1393_139390


namespace NUMINAMATH_CALUDE_track_width_l1393_139335

theorem track_width (r : ℝ) 
  (h1 : 2 * π * (2 * r) - 2 * π * r = 16 * π) 
  (h2 : 2 * r - r = r) : r = 8 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l1393_139335


namespace NUMINAMATH_CALUDE_max_volume_cuboid_l1393_139386

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℕ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℕ :=
  c.length * c.width * c.height

/-- Theorem stating the maximum volume of a cuboid with given conditions -/
theorem max_volume_cuboid :
  ∃ (c : Cuboid), surfaceArea c = 150 ∧
    (∀ (c' : Cuboid), surfaceArea c' = 150 → volume c' ≤ volume c) ∧
    volume c = 125 := by
  sorry


end NUMINAMATH_CALUDE_max_volume_cuboid_l1393_139386


namespace NUMINAMATH_CALUDE_factorization_sum_l1393_139303

theorem factorization_sum (A B C D E F G H J K : ℤ) :
  (∀ x y : ℝ, 27 * x^6 - 512 * y^6 = (A * x + B * y) * (C * x^2 + D * x * y + E * y^2) * 
                                     (F * x + G * y) * (H * x^2 + J * x * y + K * y^2)) →
  A + B + C + D + E + F + G + H + J + K = 32 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l1393_139303


namespace NUMINAMATH_CALUDE_max_value_interval_l1393_139333

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

theorem max_value_interval (a : ℝ) (h1 : a ≤ 4) :
  (∃ (x : ℝ), x ∈ Set.Ioo (3 - a^2) a ∧
   ∀ (y : ℝ), y ∈ Set.Ioo (3 - a^2) a → f y ≤ f x) →
  Real.sqrt 2 < a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_interval_l1393_139333


namespace NUMINAMATH_CALUDE_age_ratio_sandy_molly_l1393_139329

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the passage of time in years -/
def yearsLater (a : Age) (n : ℕ) : Age :=
  ⟨a.years + n⟩

theorem age_ratio_sandy_molly :
  ∀ (sandy_current : Age) (molly_current : Age),
    yearsLater sandy_current 6 = Age.mk 42 →
    molly_current = Age.mk 27 →
    (sandy_current.years : ℚ) / molly_current.years = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sandy_molly_l1393_139329


namespace NUMINAMATH_CALUDE_right_triangle_circle_properties_l1393_139334

/-- Properties of right triangles relating inscribed and circumscribed circles -/
theorem right_triangle_circle_properties (a b c r R p : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → p > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  p = a + b + c →    -- Perimeter
  r = (a + b - c) / 2 →  -- Inradius formula
  R = c / 2 →        -- Circumradius formula
  (p / c - r / R = 2) ∧
  (r / R ≤ 1 / (Real.sqrt 2 + 1)) ∧
  (r / R = 1 / (Real.sqrt 2 + 1) ↔ a = b) :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_circle_properties_l1393_139334


namespace NUMINAMATH_CALUDE_gwen_gave_away_seven_games_l1393_139393

/-- The number of games Gwen gave away -/
def games_given_away (initial_games : ℕ) (remaining_games : ℕ) : ℕ :=
  initial_games - remaining_games

/-- Proof that Gwen gave away 7 games -/
theorem gwen_gave_away_seven_games :
  let initial_games : ℕ := 98
  let remaining_games : ℕ := 91
  games_given_away initial_games remaining_games = 7 := by
  sorry

end NUMINAMATH_CALUDE_gwen_gave_away_seven_games_l1393_139393


namespace NUMINAMATH_CALUDE_average_problem_l1393_139307

theorem average_problem (x : ℝ) : (15 + 25 + x) / 3 = 23 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1393_139307


namespace NUMINAMATH_CALUDE_rectangle_rotation_forms_cylinder_l1393_139371

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_positive : width > 0
  height_positive : height > 0

/-- Represents the solid formed by rotating a rectangle -/
inductive RotatedSolid
  | Cylinder
  | Other

/-- Function that determines the shape of the solid formed by rotating a rectangle -/
def rotate_rectangle (rect : Rectangle) : RotatedSolid := sorry

/-- Theorem stating that rotating a rectangle forms a cylinder -/
theorem rectangle_rotation_forms_cylinder (rect : Rectangle) :
  rotate_rectangle rect = RotatedSolid.Cylinder := by sorry

end NUMINAMATH_CALUDE_rectangle_rotation_forms_cylinder_l1393_139371


namespace NUMINAMATH_CALUDE_temperature_difference_l1393_139304

theorem temperature_difference (M L N : ℝ) : 
  M = L + N →  -- Minneapolis is N degrees warmer than St. Louis at noon
  (∃ (M_4 L_4 : ℝ), 
    M_4 = M - 5 ∧  -- Minneapolis temperature falls by 5 degrees at 4:00
    L_4 = L + 3 ∧  -- St. Louis temperature rises by 3 degrees at 4:00
    abs (M_4 - L_4) = 2) →  -- Temperatures differ by 2 degrees at 4:00
  (N = 10 ∨ N = 6) ∧ N * (16 - N) = 60 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_l1393_139304


namespace NUMINAMATH_CALUDE_first_triangle_height_l1393_139338

/-- Given two triangles where the second has double the area of the first,
    prove that the height of the first triangle is 12 cm. -/
theorem first_triangle_height
  (base1 : ℝ) (base2 : ℝ) (height2 : ℝ)
  (h_base1 : base1 = 15)
  (h_base2 : base2 = 20)
  (h_height2 : height2 = 18)
  (h_area_relation : base2 * height2 = 2 * base1 * (12 : ℝ)) :
  ∃ (height1 : ℝ), height1 = 12 ∧ base1 * height1 = (1/2) * base2 * height2 :=
by sorry

end NUMINAMATH_CALUDE_first_triangle_height_l1393_139338


namespace NUMINAMATH_CALUDE_functional_equation_characterization_l1393_139383

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

/-- The characterization of functions satisfying the functional equation -/
theorem functional_equation_characterization (f : ℕ → ℕ) 
  (h : FunctionalEquation f) : 
  ∃ d : ℕ, d > 0 ∧ ∀ m : ℕ, ∃ k : ℕ, f m = k * d :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_characterization_l1393_139383


namespace NUMINAMATH_CALUDE_days_to_complete_correct_l1393_139396

/-- Represents the number of days required to complete the work -/
def days_to_complete : ℕ := 9

/-- Represents the total number of family members -/
def total_members : ℕ := 15

/-- Represents the number of days it takes for a woman to complete the work -/
def woman_days : ℕ := 180

/-- Represents the number of days it takes for a man to complete the work -/
def man_days : ℕ := 120

/-- Represents the number of women in the family -/
def num_women : ℕ := 3

/-- Represents the number of men in the family -/
def num_men : ℕ := total_members - num_women

/-- Represents the fraction of work done by women in one day -/
def women_work_per_day : ℚ := (1 / woman_days : ℚ) * num_women / 3

/-- Represents the fraction of work done by men in one day -/
def men_work_per_day : ℚ := (1 / man_days : ℚ) * num_men / 2

/-- Represents the total fraction of work done by the family in one day -/
def total_work_per_day : ℚ := women_work_per_day + men_work_per_day

/-- Theorem stating that the calculated number of days to complete the work is correct -/
theorem days_to_complete_correct : 
  ⌈(1 : ℚ) / total_work_per_day⌉ = days_to_complete := by sorry

end NUMINAMATH_CALUDE_days_to_complete_correct_l1393_139396


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1393_139370

theorem sum_of_x_and_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 + y^2 = 1) (h4 : (3*x - 4*x^3) * (3*y - 4*y^3) = -1/2) : 
  x + y = Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1393_139370


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l1393_139300

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0

-- Define proposition q
def q : Prop := ∀ m : ℝ, m > -2 → ∃ x : ℝ, x^2 + 2*x - m = 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l1393_139300


namespace NUMINAMATH_CALUDE_lunchroom_tables_l1393_139349

theorem lunchroom_tables (students_per_table : ℕ) (total_students : ℕ) (h1 : students_per_table = 6) (h2 : total_students = 204) :
  total_students / students_per_table = 34 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_tables_l1393_139349


namespace NUMINAMATH_CALUDE_conference_left_handed_fraction_l1393_139343

theorem conference_left_handed_fraction :
  ∀ (total : ℕ) (red blue left_handed_red left_handed_blue : ℕ),
    red + blue = total →
    red = 7 * (total / 10) →
    blue = 3 * (total / 10) →
    left_handed_red = red / 3 →
    left_handed_blue = 2 * blue / 3 →
    (left_handed_red + left_handed_blue : ℚ) / total = 13 / 30 :=
by sorry

end NUMINAMATH_CALUDE_conference_left_handed_fraction_l1393_139343


namespace NUMINAMATH_CALUDE_binomial_coefficient_26_6_l1393_139373

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_26_6_l1393_139373


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1393_139354

theorem binomial_coefficient_two (n : ℕ+) : (n.val.choose 2) = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1393_139354


namespace NUMINAMATH_CALUDE_max_value_of_g_l1393_139380

/-- The quadratic function f(x, y) -/
def f (x y : ℝ) : ℝ := 10*x - 4*x^2 + 2*x*y

/-- The function g(x) is f(x, 3) -/
def g (x : ℝ) : ℝ := f x 3

theorem max_value_of_g :
  ∃ (m : ℝ), ∀ (x : ℝ), g x ≤ m ∧ ∃ (x₀ : ℝ), g x₀ = m ∧ m = 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l1393_139380


namespace NUMINAMATH_CALUDE_coordinates_of_A_min_length_AB_l1393_139384

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a line passing through the focus
structure LineThruFocus where
  slope : ℝ ⊕ PUnit  -- ℝ for finite slopes, PUnit for vertical line
  passes_thru_focus : True

-- Define the intersection points
def intersectionPoints (l : LineThruFocus) : PointOnParabola × PointOnParabola := sorry

-- Statement for part (1)
theorem coordinates_of_A (l : LineThruFocus) (A B : PointOnParabola) 
  (h : intersectionPoints l = (A, B)) (dist_AF : Real.sqrt ((A.x - 1)^2 + A.y^2) = 4) :
  (A.x = 3 ∧ A.y = 2 * Real.sqrt 3) ∨ (A.x = 3 ∧ A.y = -2 * Real.sqrt 3) := sorry

-- Statement for part (2)
theorem min_length_AB : 
  ∃ (min_length : ℝ), ∀ (l : LineThruFocus) (A B : PointOnParabola),
    intersectionPoints l = (A, B) → 
    Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) ≥ min_length ∧
    min_length = 4 := sorry

end NUMINAMATH_CALUDE_coordinates_of_A_min_length_AB_l1393_139384


namespace NUMINAMATH_CALUDE_minimum_rental_fee_for_class_trip_l1393_139310

/-- Calculates the minimum rental fee for a class trip --/
def minimum_rental_fee (total_students : ℕ) 
  (small_boat_capacity small_boat_cost large_boat_capacity large_boat_cost : ℕ) : ℕ :=
  let large_boats := total_students / large_boat_capacity
  let remaining_students := total_students % large_boat_capacity
  let small_boats := (remaining_students + small_boat_capacity - 1) / small_boat_capacity
  large_boats * large_boat_cost + small_boats * small_boat_cost

theorem minimum_rental_fee_for_class_trip :
  minimum_rental_fee 48 3 16 5 24 = 232 :=
by sorry

end NUMINAMATH_CALUDE_minimum_rental_fee_for_class_trip_l1393_139310


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1393_139388

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  -3 + 23*x - x^2 + 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1393_139388


namespace NUMINAMATH_CALUDE_chinese_dinner_cost_l1393_139322

theorem chinese_dinner_cost (num_people : ℕ) (num_appetizers : ℕ) (appetizer_cost : ℚ)
  (tip_percentage : ℚ) (rush_fee : ℚ) (total_spent : ℚ) :
  num_people = 4 →
  num_appetizers = 2 →
  appetizer_cost = 6 →
  tip_percentage = 0.2 →
  rush_fee = 5 →
  total_spent = 77 →
  ∃ (main_meal_cost : ℚ),
    main_meal_cost * num_people +
    num_appetizers * appetizer_cost +
    (main_meal_cost * num_people + num_appetizers * appetizer_cost) * tip_percentage +
    rush_fee = total_spent ∧
    main_meal_cost = 12 :=
by sorry

end NUMINAMATH_CALUDE_chinese_dinner_cost_l1393_139322


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1393_139305

theorem arithmetic_equality : (469138 * 9999) + (876543 * 12345) = 15512230997 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1393_139305


namespace NUMINAMATH_CALUDE_alok_rice_order_l1393_139392

def chapatis : ℕ := 16
def mixed_vegetable : ℕ := 7
def ice_cream_cups : ℕ := 6
def cost_chapati : ℕ := 6
def cost_rice : ℕ := 45
def cost_mixed_vegetable : ℕ := 70
def cost_ice_cream : ℕ := 40
def total_paid : ℕ := 1051

theorem alok_rice_order :
  ∃ (rice_plates : ℕ),
    rice_plates = 5 ∧
    total_paid = chapatis * cost_chapati +
                 rice_plates * cost_rice +
                 mixed_vegetable * cost_mixed_vegetable +
                 ice_cream_cups * cost_ice_cream :=
by sorry

end NUMINAMATH_CALUDE_alok_rice_order_l1393_139392


namespace NUMINAMATH_CALUDE_Z_in_first_quadrant_l1393_139398

def Z : ℂ := Complex.I * (1 - 2 * Complex.I)

theorem Z_in_first_quadrant : 
  Complex.re Z > 0 ∧ Complex.im Z > 0 := by
  sorry

end NUMINAMATH_CALUDE_Z_in_first_quadrant_l1393_139398


namespace NUMINAMATH_CALUDE_star_negative_two_five_l1393_139372

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := a * b^3 - b^2 + 2

-- Theorem statement
theorem star_negative_two_five : star (-2) 5 = -273 := by
  sorry

end NUMINAMATH_CALUDE_star_negative_two_five_l1393_139372


namespace NUMINAMATH_CALUDE_subcommittee_count_l1393_139309

def total_members : ℕ := 12
def officers : ℕ := 5
def subcommittee_size : ℕ := 5

def subcommittees_with_at_least_two_officers : ℕ :=
  Nat.choose total_members subcommittee_size -
  (Nat.choose (total_members - officers) subcommittee_size +
   Nat.choose officers 1 * Nat.choose (total_members - officers) (subcommittee_size - 1))

theorem subcommittee_count :
  subcommittees_with_at_least_two_officers = 596 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l1393_139309


namespace NUMINAMATH_CALUDE_quadratic_inequality_transformation_l1393_139357

theorem quadratic_inequality_transformation (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c < 0 ↔ x < -2 ∨ x > -1/2) →
  (∀ x : ℝ, c*x^2 - b*x + a > 0 ↔ 1/2 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_transformation_l1393_139357


namespace NUMINAMATH_CALUDE_f_not_valid_mapping_l1393_139342

-- Define the sets M and P
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 6}
def P : Set ℝ := {y | 0 ≤ y ∧ y ≤ 3}

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x

-- Theorem stating that f is not a valid mapping from M to P
theorem f_not_valid_mapping : ¬(∀ x ∈ M, f x ∈ P) := by
  sorry


end NUMINAMATH_CALUDE_f_not_valid_mapping_l1393_139342


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l1393_139302

/-- Represents a 4x4x4 cube composed of smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  diagonal_shaded : Bool

/-- Represents the shading pattern on the faces of the large cube -/
structure ShadingPattern where
  diagonal : Bool
  opposite_faces_identical : Bool

/-- Counts the number of smaller cubes with at least one shaded face -/
def count_shaded_cubes (cube : LargeCube) (pattern : ShadingPattern) : Nat :=
  sorry

/-- The main theorem stating that 32 smaller cubes are shaded -/
theorem shaded_cubes_count (cube : LargeCube) (pattern : ShadingPattern) :
  cube.size = 4 ∧ 
  cube.total_cubes = 64 ∧ 
  cube.diagonal_shaded = true ∧
  pattern.diagonal = true ∧
  pattern.opposite_faces_identical = true →
  count_shaded_cubes cube pattern = 32 :=
sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l1393_139302


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1393_139330

/-- Given real numbers a, b, and c, and polynomials g and f as defined,
    prove that f(-1) = -29041 -/
theorem polynomial_evaluation (a b c : ℝ) : 
  let g := fun (x : ℝ) => x^3 + a*x^2 + x + 20
  let f := fun (x : ℝ) => x^4 + x^3 + b*x^2 + 200*x + c
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0 ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  f (-1) = -29041 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1393_139330


namespace NUMINAMATH_CALUDE_algebraic_equality_l1393_139362

theorem algebraic_equality (a b : ℝ) : 
  (2*a^2 - 4*a*b + b^2 = -3*a^2 + 2*a*b - 5*b^2) → 
  (2*a^2 - 4*a*b + b^2 + 3*a^2 - 2*a*b + 5*b^2 = 5*a^2 + 6*b^2 - 6*a*b) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l1393_139362


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l1393_139399

/-- The perpendicular bisector of a line segment with endpoints (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem perpendicular_bisector_equation :
  perpendicular_bisector 1 3 5 (-1) = {p : ℝ × ℝ | p.1 - p.2 - 2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l1393_139399


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l1393_139394

/-- The number of wheels on a car -/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a bike -/
def wheels_per_bike : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 2

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * wheels_per_car + num_bikes * wheels_per_bike

theorem parking_lot_wheels : total_wheels = 44 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l1393_139394


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1393_139317

def set_A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1393_139317


namespace NUMINAMATH_CALUDE_expression_equals_seventeen_l1393_139346

theorem expression_equals_seventeen : 3 + 4 * 5 - 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seventeen_l1393_139346


namespace NUMINAMATH_CALUDE_axis_of_symmetry_point_relationship_t_range_max_t_value_l1393_139389

-- Define the parabola
def parabola (t x y : ℝ) : Prop := y = x^2 - 2*t*x + 1

-- Theorem for the axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ x y : ℝ, parabola t x y → parabola t (2*t - x) y := by sorry

-- Theorem for point relationship
theorem point_relationship (t m n : ℝ) :
  parabola t (t-2) m → parabola t (t+3) n → m < n := by sorry

-- Theorem for t range
theorem t_range (t : ℝ) :
  (∀ x₁ y₁ y₂ : ℝ, -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ y₁ ∧ parabola t 3 y₂ ∧ y₁ ≤ y₂) 
  → t ≤ 1 := by sorry

-- Theorem for maximum t value
theorem max_t_value :
  ∃ t_max : ℝ, t_max = 5 ∧ 
  ∀ t y₁ y₂ : ℝ, parabola t (t+1) y₁ ∧ parabola t (2*t-4) y₂ ∧ y₁ ≥ y₂ 
  → t ≤ t_max := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_point_relationship_t_range_max_t_value_l1393_139389


namespace NUMINAMATH_CALUDE_in_class_calculation_l1393_139350

theorem in_class_calculation :
  (((4.2 : ℝ) + 2.2) / 0.08 = 80) ∧
  (100 / 0.4 / 2.5 = 100) := by
  sorry

end NUMINAMATH_CALUDE_in_class_calculation_l1393_139350


namespace NUMINAMATH_CALUDE_area_is_60_perimeter_is_40_l1393_139339

/-- Triangle with side lengths 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The area of the right triangle is 60 -/
theorem area_is_60 (t : RightTriangle) : (1/2) * t.a * t.b = 60 := by sorry

/-- The perimeter of the right triangle is 40 -/
theorem perimeter_is_40 (t : RightTriangle) : t.a + t.b + t.c = 40 := by sorry

end NUMINAMATH_CALUDE_area_is_60_perimeter_is_40_l1393_139339


namespace NUMINAMATH_CALUDE_tangent_line_at_one_f_lower_bound_l1393_139344

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp (x - a) - Real.log x - Real.log a

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.exp (x - a) - 1 / x

theorem tangent_line_at_one (a : ℝ) (ha : a > 0) :
  f_prime a 1 = 1 → ∃ m b : ℝ, m = 1 ∧ b = 0 ∧ ∀ x : ℝ, f a x = m * x + b := by sorry

theorem f_lower_bound (a : ℝ) (ha : 0 < a) (ha2 : a < (Real.sqrt 5 - 1) / 2) :
  ∀ x : ℝ, x > 0 → f a x > a / (a + 1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_f_lower_bound_l1393_139344


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1393_139391

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 - 3*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1393_139391


namespace NUMINAMATH_CALUDE_equation_solution_l1393_139365

theorem equation_solution (x : ℕ+) : (x.val - 1) * x.val * (4 * x.val + 1) = 750 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1393_139365


namespace NUMINAMATH_CALUDE_volume_cube_inscribed_sphere_l1393_139361

/-- The volume of a cube inscribed in a sphere of radius R -/
theorem volume_cube_inscribed_sphere (R : ℝ) (R_pos : 0 < R) :
  ∃ (V : ℝ), V = (8 / 9) * Real.sqrt 3 * R^3 :=
sorry

end NUMINAMATH_CALUDE_volume_cube_inscribed_sphere_l1393_139361


namespace NUMINAMATH_CALUDE_three_valid_configurations_l1393_139324

/-- Represents a square in the configuration --/
structure Square :=
  (label : Char)

/-- Represents the F-shaped configuration --/
def FConfiguration : Finset Square := sorry

/-- The set of additional lettered squares --/
def AdditionalSquares : Finset Square := sorry

/-- Predicate to check if a configuration is valid (foldable into a cube with one open non-bottom side) --/
def IsValidConfiguration (config : Finset Square) : Prop := sorry

/-- The number of valid configurations --/
def ValidConfigurationsCount : ℕ := sorry

/-- Theorem stating that there are exactly 3 valid configurations --/
theorem three_valid_configurations :
  ValidConfigurationsCount = 3 := by sorry

end NUMINAMATH_CALUDE_three_valid_configurations_l1393_139324


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l1393_139315

/-- Given two rectangles of equal area, where one rectangle measures 4.5 inches by 19.25 inches,
    and the other rectangle is 3.75 inches long, the width of the second rectangle is 23.1 inches. -/
theorem jordan_rectangle_width (carol_length carol_width jordan_length : ℝ)
  (h1 : carol_length = 4.5)
  (h2 : carol_width = 19.25)
  (h3 : jordan_length = 3.75)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 23.1 :=
by sorry


end NUMINAMATH_CALUDE_jordan_rectangle_width_l1393_139315


namespace NUMINAMATH_CALUDE_fraction_equality_l1393_139387

theorem fraction_equality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h1 : (5*a + b) / (5*c + d) = (6*a + b) / (6*c + d))
  (h2 : (7*a + b) / (7*c + d) = 9) :
  (9*a + b) / (9*c + d) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1393_139387


namespace NUMINAMATH_CALUDE_jack_and_beanstalk_height_l1393_139323

/-- The height of the sky island in Jack and the Beanstalk --/
def sky_island_height (day_climb : ℕ) (night_slide : ℕ) (total_days : ℕ) : ℕ :=
  (total_days - 1) * (day_climb - night_slide) + day_climb

theorem jack_and_beanstalk_height :
  sky_island_height 25 3 64 = 1411 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_beanstalk_height_l1393_139323


namespace NUMINAMATH_CALUDE_certain_number_exists_l1393_139331

theorem certain_number_exists : ∃ x : ℝ, (1.78 * x) / 5.96 = 377.8020134228188 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1393_139331


namespace NUMINAMATH_CALUDE_max_population_teeth_l1393_139366

theorem max_population_teeth (n : ℕ) (h : n = 32) :
  (2 : ℕ) ^ n = 4294967296 :=
sorry

end NUMINAMATH_CALUDE_max_population_teeth_l1393_139366


namespace NUMINAMATH_CALUDE_circle_P_equation_l1393_139316

/-- The curve C defined by the distance ratio condition -/
def C (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 / 2 = 1)

/-- The line l intersecting curve C -/
def l (x y : ℝ) (k : ℝ) : Prop :=
  (y = k * (x - 1) - 1)

/-- Points A and B are on both C and l -/
def A_and_B_on_C_and_l (x₁ y₁ x₂ y₂ k : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ k ∧ l x₂ y₂ k

/-- AB is the diameter of circle P centered at (1, -1) -/
def P_diameter (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = -1

theorem circle_P_equation (x₁ y₁ x₂ y₂ k : ℝ) :
  A_and_B_on_C_and_l x₁ y₁ x₂ y₂ k →
  P_diameter x₁ y₁ x₂ y₂ →
  k = 2/3 →
  ∀ x y, (x - 1)^2 + (y + 1)^2 = 13/30 :=
sorry

end NUMINAMATH_CALUDE_circle_P_equation_l1393_139316


namespace NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l1393_139353

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 10) : 
  a^2 + b^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l1393_139353


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1393_139381

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + 2*a*x + b

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b : ℝ), f a b (-1) = 0 →
  (b = 2*a - 1) ∧
  (a = -1 → ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → f (-1) (-3) x ≤ f (-1) (-3) y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1393_139381


namespace NUMINAMATH_CALUDE_limit_of_function_l1393_139321

theorem limit_of_function (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - π/3| ∧ |x - π/3| < δ →
    |(1 - 2 * Real.cos x) / Real.sin (π - 3 * x) + Real.sqrt 3 / 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_function_l1393_139321


namespace NUMINAMATH_CALUDE_f_g_product_positive_l1393_139395

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x
def monotone_increasing_on (g : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → g x ≤ g y

-- State the theorem
theorem f_g_product_positive
  (h_f_odd : is_odd f)
  (h_f_decr : monotone_decreasing_on f {x | x < 0})
  (h_g_even : is_even g)
  (h_g_incr : monotone_increasing_on g {x | x ≤ 0})
  (h_f_1 : f 1 = 0)
  (h_g_1 : g 1 = 0) :
  {x : ℝ | f x * g x > 0} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x > 1} :=
sorry

end NUMINAMATH_CALUDE_f_g_product_positive_l1393_139395


namespace NUMINAMATH_CALUDE_max_value_theorem_l1393_139382

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  2 * x + y ≤ 46 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1393_139382


namespace NUMINAMATH_CALUDE_art_gallery_pieces_l1393_139385

theorem art_gallery_pieces (total : ℕ) 
  (displayed : ℕ) (sculptures_displayed : ℕ) 
  (paintings_not_displayed : ℕ) (sculptures_not_displayed : ℕ) :
  displayed = total / 3 →
  sculptures_displayed = displayed / 6 →
  paintings_not_displayed = (total - displayed) / 3 →
  sculptures_not_displayed = 1400 →
  total = 3150 :=
by
  sorry

end NUMINAMATH_CALUDE_art_gallery_pieces_l1393_139385


namespace NUMINAMATH_CALUDE_turtle_difference_l1393_139345

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The total number of turtles Marion and Martha received together -/
def total_turtles : ℕ := 100

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := total_turtles - martha_turtles

/-- Marion received more turtles than Martha -/
axiom marion_more : marion_turtles > martha_turtles

theorem turtle_difference : marion_turtles - martha_turtles = 20 := by
  sorry

end NUMINAMATH_CALUDE_turtle_difference_l1393_139345


namespace NUMINAMATH_CALUDE_anna_truck_meet_once_l1393_139325

/-- Represents the movement of Anna and the garbage truck on a path with trash pails. -/
structure TrashCollection where
  annaSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ

/-- Calculates the number of times Anna and the truck meet. -/
def meetingCount (tc : TrashCollection) : ℕ :=
  sorry

/-- The theorem states that Anna and the truck meet exactly once under the given conditions. -/
theorem anna_truck_meet_once :
  ∀ (tc : TrashCollection),
    tc.annaSpeed = 5 ∧
    tc.truckSpeed = 15 ∧
    tc.pailDistance = 300 ∧
    tc.truckStopTime = 40 →
    meetingCount tc = 1 :=
  sorry

end NUMINAMATH_CALUDE_anna_truck_meet_once_l1393_139325


namespace NUMINAMATH_CALUDE_xiao_hong_pen_purchase_l1393_139311

theorem xiao_hong_pen_purchase (total_money : ℝ) (pen_cost : ℝ) (notebook_cost : ℝ) 
  (notebooks_bought : ℕ) (h1 : total_money = 18) (h2 : pen_cost = 3) 
  (h3 : notebook_cost = 3.6) (h4 : notebooks_bought = 2) :
  ∃ (pens : ℕ), pens ∈ ({1, 2, 3} : Set ℕ) ∧ 
  (notebooks_bought : ℝ) * notebook_cost + (pens : ℝ) * pen_cost ≤ total_money :=
sorry

end NUMINAMATH_CALUDE_xiao_hong_pen_purchase_l1393_139311
