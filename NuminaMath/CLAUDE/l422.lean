import Mathlib

namespace complex_multiplication_l422_42203

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2*i := by
  sorry

end complex_multiplication_l422_42203


namespace cube_root_three_equation_l422_42231

theorem cube_root_three_equation : 
  (1 : ℝ) / (2 - Real.rpow 3 (1/3)) = (2 + Real.rpow 3 (1/3)) * (2 + Real.sqrt 3) := by
  sorry

end cube_root_three_equation_l422_42231


namespace care_package_weight_l422_42242

def final_weight (initial_weight : ℝ) : ℝ :=
  let weight_after_chocolate := initial_weight * 1.4
  let weight_after_snacks := weight_after_chocolate + 0.6 - 0.35 + 0.85
  let weight_after_cookies := weight_after_snacks * 1.6
  let weight_after_brownie_removal := weight_after_cookies - 0.45
  5 * initial_weight

theorem care_package_weight :
  let initial_weight := 1.25 + 0.75 + 1.5
  final_weight initial_weight = 17.5 := by
  sorry

end care_package_weight_l422_42242


namespace painted_numbers_theorem_l422_42299

/-- The number of hours on a clock face -/
def clockHours : ℕ := 12

/-- Function to calculate the number of distinct numbers painted on a clock face -/
def distinctPaintedNumbers (paintInterval : ℕ) : ℕ :=
  clockHours / Nat.gcd clockHours paintInterval

theorem painted_numbers_theorem :
  (distinctPaintedNumbers 57 = 4) ∧
  (distinctPaintedNumbers 1913 = 12) := by
  sorry

#eval distinctPaintedNumbers 57  -- Expected: 4
#eval distinctPaintedNumbers 1913  -- Expected: 12

end painted_numbers_theorem_l422_42299


namespace razorback_tshirt_profit_l422_42202

/-- The amount made per t-shirt, given the number of t-shirts sold and the total amount made from t-shirts. -/
def amount_per_tshirt (num_tshirts : ℕ) (total_amount : ℕ) : ℚ :=
  total_amount / num_tshirts

/-- Theorem stating that the amount made per t-shirt is $62. -/
theorem razorback_tshirt_profit : amount_per_tshirt 183 11346 = 62 := by
  sorry

end razorback_tshirt_profit_l422_42202


namespace employed_females_percentage_l422_42293

theorem employed_females_percentage (total_population : ℝ) 
  (h1 : total_population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 72) 
  (employed_males_percentage : ℝ) 
  (h3 : employed_males_percentage = 36) : 
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 50 := by
  sorry

end employed_females_percentage_l422_42293


namespace james_alice_equation_equivalence_l422_42238

theorem james_alice_equation_equivalence (d e : ℝ) : 
  (∀ x, |x - 8| = 3 ↔ x^2 + d*x + e = 0) ↔ (d = -16 ∧ e = 55) := by
  sorry

end james_alice_equation_equivalence_l422_42238


namespace circle_triangle_area_l422_42228

/-- Given a circle C with center (a, 2/a) that passes through the origin (0, 0)
    and intersects the x-axis at (2a, 0) and the y-axis at (0, 4/a),
    prove that the area of the triangle formed by these three points is 4. -/
theorem circle_triangle_area (a : ℝ) (ha : a ≠ 0) : 
  let center : ℝ × ℝ := (a, 2/a)
  let origin : ℝ × ℝ := (0, 0)
  let point_A : ℝ × ℝ := (2*a, 0)
  let point_B : ℝ × ℝ := (0, 4/a)
  let triangle_area := abs ((point_A.1 - origin.1) * (point_B.2 - origin.2)) / 2
  triangle_area = 4 :=
by sorry

end circle_triangle_area_l422_42228


namespace circle_center_and_radius_l422_42214

/-- Given a circle with equation x^2 - 8x + y^2 - 4y = -4, prove its center and radius -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (4, 2) ∧
    radius = 4 ∧
    ∀ (x y : ℝ), x^2 - 8*x + y^2 - 4*y = -4 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l422_42214


namespace star_perimeter_is_160_l422_42223

/-- The radius of each circle in cm -/
def circle_radius : ℝ := 5

/-- The side length of the square in cm -/
def square_side_length : ℝ := 4 * circle_radius

/-- The number of sides in the star -/
def star_sides : ℕ := 8

/-- The perimeter of the star in cm -/
def star_perimeter : ℝ := star_sides * square_side_length

/-- Theorem stating that the perimeter of the star is 160 cm -/
theorem star_perimeter_is_160 : star_perimeter = 160 := by
  sorry

end star_perimeter_is_160_l422_42223


namespace mary_total_time_l422_42272

-- Define the given conditions
def mac_download_time : ℕ := 10
def windows_download_time : ℕ := 3 * mac_download_time
def audio_glitch_time : ℕ := 2 * 4
def video_glitch_time : ℕ := 6
def glitch_time : ℕ := audio_glitch_time + video_glitch_time
def non_glitch_time : ℕ := 2 * glitch_time

-- Theorem statement
theorem mary_total_time :
  mac_download_time + windows_download_time + glitch_time + non_glitch_time = 82 :=
by sorry

end mary_total_time_l422_42272


namespace rectangle_segment_comparison_l422_42264

/-- Given a rectangle ABCD with specific properties, prove AM > BK -/
theorem rectangle_segment_comparison (A B C D M K : ℝ × ℝ) : 
  let AB : ℝ := 2
  let BD : ℝ := Real.sqrt 7
  let AC : ℝ := Real.sqrt (AB^2 + BD^2 - AB^2)
  -- Rectangle properties
  (B.1 - A.1 = AB ∧ B.2 = A.2) →
  (C.1 = B.1 ∧ C.2 - A.2 = AC) →
  (D.1 = A.1 ∧ D.2 = C.2) →
  -- M divides CD in 1:2 ratio
  (M.1 - C.1 = (1/3) * (D.1 - C.1) ∧ M.2 - C.2 = (1/3) * (D.2 - C.2)) →
  -- K is midpoint of AD
  (K.1 = (A.1 + D.1) / 2 ∧ K.2 = (A.2 + D.2) / 2) →
  -- Prove AM > BK
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) > Real.sqrt ((K.1 - B.1)^2 + (K.2 - B.2)^2) :=
by
  sorry


end rectangle_segment_comparison_l422_42264


namespace expected_value_of_new_balls_l422_42278

/-- Represents the outcome of drawing balls in a ping pong match -/
inductive BallDraw
  | zero
  | one
  | two

/-- The probability mass function for the number of new balls drawn -/
def prob_new_balls (draw : BallDraw) : ℚ :=
  match draw with
  | BallDraw.zero => 37/100
  | BallDraw.one  => 54/100
  | BallDraw.two  => 9/100

/-- The number of new balls for each outcome -/
def num_new_balls (draw : BallDraw) : ℕ :=
  match draw with
  | BallDraw.zero => 0
  | BallDraw.one  => 1
  | BallDraw.two  => 2

/-- The expected value of new balls in the second draw -/
def expected_value : ℚ :=
  (prob_new_balls BallDraw.zero * num_new_balls BallDraw.zero) +
  (prob_new_balls BallDraw.one  * num_new_balls BallDraw.one)  +
  (prob_new_balls BallDraw.two  * num_new_balls BallDraw.two)

theorem expected_value_of_new_balls :
  expected_value = 18/25 := by sorry

end expected_value_of_new_balls_l422_42278


namespace correct_calculation_l422_42286

theorem correct_calculation : 
  (67 * 17 ≠ 1649) ∧ 
  (150 * 60 ≠ 900) ∧ 
  (250 * 70 = 17500) ∧ 
  (98 * 36 ≠ 3822) :=
by sorry

end correct_calculation_l422_42286


namespace refrigerator_installation_cost_l422_42247

theorem refrigerator_installation_cost 
  (purchase_price : ℝ) 
  (discount_percentage : ℝ)
  (transport_cost : ℝ)
  (profit_percentage : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 12500)
  (h2 : discount_percentage = 0.20)
  (h3 : transport_cost = 125)
  (h4 : profit_percentage = 0.12)
  (h5 : selling_price = 17920) :
  ∃ (installation_cost : ℝ),
    installation_cost = 295 ∧
    selling_price = 
      (purchase_price / (1 - discount_percentage)) * 
      (1 + profit_percentage) + 
      transport_cost + 
      installation_cost :=
by sorry

end refrigerator_installation_cost_l422_42247


namespace quadratic_function_unique_l422_42287

/-- A quadratic function satisfying certain conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  f (-1) = 0 ∧
  ∀ x, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2

/-- The theorem stating that the quadratic function satisfying the given conditions
    must be f(x) = 1/4(x+1)^2 -/
theorem quadratic_function_unique :
  ∀ f : ℝ → ℝ, QuadraticFunction f → ∀ x, f x = (1/4) * (x + 1)^2 := by
  sorry

end quadratic_function_unique_l422_42287


namespace crates_in_third_trip_is_two_l422_42294

/-- Represents the problem of distributing crates across multiple trips. -/
structure CrateDistribution where
  total_crates : ℕ
  min_crate_weight : ℕ
  max_trip_weight : ℕ

/-- Calculates the number of crates in the third trip. -/
def crates_in_third_trip (cd : CrateDistribution) : ℕ :=
  cd.total_crates - 2 * (cd.max_trip_weight / cd.min_crate_weight)

/-- Theorem stating that for the given conditions, the number of crates in the third trip is 2. -/
theorem crates_in_third_trip_is_two :
  let cd : CrateDistribution := {
    total_crates := 12,
    min_crate_weight := 120,
    max_trip_weight := 600
  }
  crates_in_third_trip cd = 2 := by
  sorry

end crates_in_third_trip_is_two_l422_42294


namespace max_binomial_coeff_expansion_l422_42275

theorem max_binomial_coeff_expansion (m : ℕ) : 
  (∀ x : ℝ, x > 0 → (5 / Real.sqrt x - x)^m = 256) → 
  (∃ k : ℕ, k ≤ m ∧ Nat.choose m k = 6 ∧ ∀ j : ℕ, j ≤ m → Nat.choose m j ≤ 6) := by
  sorry

end max_binomial_coeff_expansion_l422_42275


namespace largest_number_l422_42201

theorem largest_number (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : -1 < b ∧ b < 0) :
  (a - b) = max a (max (a * b) (max (a - b) (a + b))) := by
sorry

end largest_number_l422_42201


namespace age_ratio_proof_l422_42274

-- Define Rahul's and Deepak's ages
def rahul_future_age : ℕ := 26
def years_to_future : ℕ := 10
def deepak_current_age : ℕ := 12

-- Define the ratio we want to prove
def target_ratio : Rat := 4 / 3

-- Theorem statement
theorem age_ratio_proof :
  (rahul_future_age - years_to_future : ℚ) / deepak_current_age = target_ratio := by
  sorry

end age_ratio_proof_l422_42274


namespace olivers_mom_money_l422_42217

/-- Calculates the amount of money Oliver's mom gave him -/
theorem olivers_mom_money (initial : ℕ) (spent : ℕ) (final : ℕ) : 
  initial - spent + (final - (initial - spent)) = final ∧ 
  final - (initial - spent) = 32 :=
by
  sorry

#check olivers_mom_money 33 4 61

end olivers_mom_money_l422_42217


namespace latest_departure_time_l422_42229

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- Represents the flight constraints -/
structure FlightConstraints where
  flightDepartureTime : Time
  checkInTime : Nat
  driveTime : Nat
  parkAndWalkTime : Nat

theorem latest_departure_time (constraints : FlightConstraints) 
  (h1 : constraints.flightDepartureTime = ⟨20, 0⟩)
  (h2 : constraints.checkInTime = 120)
  (h3 : constraints.driveTime = 45)
  (h4 : constraints.parkAndWalkTime = 15) :
  let latestDepartureTime := ⟨17, 0⟩
  timeDifference constraints.flightDepartureTime latestDepartureTime = 
    constraints.checkInTime + constraints.driveTime + constraints.parkAndWalkTime :=
by sorry

end latest_departure_time_l422_42229


namespace max_value_p_l422_42246

theorem max_value_p (p q r s t u v w : ℕ+) : 
  (p + q + r + s = 35) →
  (q + r + s + t = 35) →
  (r + s + t + u = 35) →
  (s + t + u + v = 35) →
  (t + u + v + w = 35) →
  (q + v = 14) →
  (∀ x : ℕ+, x ≤ p → 
    ∃ q' r' s' t' u' v' w' : ℕ+,
      (x + q' + r' + s' = 35) ∧
      (q' + r' + s' + t' = 35) ∧
      (r' + s' + t' + u' = 35) ∧
      (s' + t' + u' + v' = 35) ∧
      (t' + u' + v' + w' = 35) ∧
      (q' + v' = 14)) →
  p = 20 :=
by sorry

end max_value_p_l422_42246


namespace total_coins_remain_odd_cannot_achieve_equal_coins_l422_42236

/-- Represents the state of Petya's coins -/
structure CoinState where
  two_kopeck : ℕ
  ten_kopeck : ℕ

/-- The initial state of Petya's coins -/
def initial_state : CoinState := { two_kopeck := 1, ten_kopeck := 0 }

/-- Represents a coin insertion operation -/
inductive InsertionOperation
  | insert_two_kopeck
  | insert_ten_kopeck

/-- Applies an insertion operation to a coin state -/
def apply_insertion (state : CoinState) (op : InsertionOperation) : CoinState :=
  match op with
  | InsertionOperation.insert_two_kopeck => 
      { two_kopeck := state.two_kopeck - 1, ten_kopeck := state.ten_kopeck + 5 }
  | InsertionOperation.insert_ten_kopeck => 
      { two_kopeck := state.two_kopeck + 5, ten_kopeck := state.ten_kopeck - 1 }

/-- The total number of coins in a given state -/
def total_coins (state : CoinState) : ℕ := state.two_kopeck + state.ten_kopeck

/-- Theorem stating that the total number of coins remains odd after any sequence of insertions -/
theorem total_coins_remain_odd (ops : List InsertionOperation) : 
  Odd (total_coins (ops.foldl apply_insertion initial_state)) := by
  sorry

/-- Theorem stating that Petya cannot achieve an equal number of two-kopeck and ten-kopeck coins -/
theorem cannot_achieve_equal_coins (ops : List InsertionOperation) : 
  let final_state := ops.foldl apply_insertion initial_state
  ¬(final_state.two_kopeck = final_state.ten_kopeck) := by
  sorry

end total_coins_remain_odd_cannot_achieve_equal_coins_l422_42236


namespace bounded_diff_sequence_has_infinite_divisible_pairs_l422_42283

/-- A sequence of positive integers with bounded differences -/
def BoundedDiffSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n ≤ 2001

/-- The property of having infinitely many divisible pairs -/
def InfinitelyManyDivisiblePairs (a : ℕ → ℕ) : Prop :=
  ∀ k, ∃ p q, p > q ∧ q > k ∧ a q ∣ a p

/-- The main theorem -/
theorem bounded_diff_sequence_has_infinite_divisible_pairs
  (a : ℕ → ℕ) (h : BoundedDiffSequence a) :
  InfinitelyManyDivisiblePairs a :=
sorry

end bounded_diff_sequence_has_infinite_divisible_pairs_l422_42283


namespace march_2020_production_theorem_l422_42206

/-- Calculates the total toilet paper production for March 2020 after a production increase -/
def march_2020_toilet_paper_production (initial_production : ℕ) (increase_factor : ℕ) (days : ℕ) : ℕ :=
  (initial_production + initial_production * increase_factor) * days

/-- Theorem stating the total toilet paper production for March 2020 -/
theorem march_2020_production_theorem :
  march_2020_toilet_paper_production 7000 3 31 = 868000 := by
  sorry

#eval march_2020_toilet_paper_production 7000 3 31

end march_2020_production_theorem_l422_42206


namespace goods_train_speed_calculation_l422_42261

/-- The speed of the man's train in km/h -/
def man_train_speed : ℝ := 40

/-- The time it takes for the goods train to pass the man in seconds -/
def passing_time : ℝ := 12

/-- The length of the goods train in meters -/
def goods_train_length : ℝ := 350

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 65

/-- Theorem stating that the given conditions imply the correct speed of the goods train -/
theorem goods_train_speed_calculation :
  man_train_speed = 40 ∧
  passing_time = 12 ∧
  goods_train_length = 350 →
  goods_train_speed = 65 := by
  sorry

#check goods_train_speed_calculation

end goods_train_speed_calculation_l422_42261


namespace deductive_reasoning_properties_l422_42263

-- Define the properties of deductive reasoning
def is_general_to_specific (r : Type) : Prop := sorry
def conclusion_always_correct (r : Type) : Prop := sorry
def has_syllogism_form (r : Type) : Prop := sorry
def correctness_depends_on_premises_and_form (r : Type) : Prop := sorry

-- Define deductive reasoning
def deductive_reasoning : Type := sorry

-- Theorem stating that exactly 3 out of 4 statements are correct
theorem deductive_reasoning_properties :
  is_general_to_specific deductive_reasoning ∧
  ¬conclusion_always_correct deductive_reasoning ∧
  has_syllogism_form deductive_reasoning ∧
  correctness_depends_on_premises_and_form deductive_reasoning :=
sorry

end deductive_reasoning_properties_l422_42263


namespace fern_leaves_count_l422_42254

/-- The number of leaves on all ferns -/
def total_leaves (num_ferns : ℕ) (fronds_per_fern : ℕ) (leaves_per_frond : ℕ) : ℕ :=
  num_ferns * fronds_per_fern * leaves_per_frond

/-- Theorem stating the total number of leaves on all ferns -/
theorem fern_leaves_count :
  total_leaves 6 7 30 = 1260 := by
  sorry

end fern_leaves_count_l422_42254


namespace only_setA_is_pythagorean_triple_l422_42200

/-- A function to check if a triple of integers forms a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a * a + b * b = c * c

/-- The given sets of numbers -/
def setA : List ℤ := [5, 12, 13]
def setB : List ℤ := [7, 9, 11]
def setC : List ℤ := [6, 9, 12]
def setD : List ℚ := [3/10, 4/10, 5/10]

/-- Theorem stating that only setA is a Pythagorean triple -/
theorem only_setA_is_pythagorean_triple :
  (isPythagoreanTriple setA[0]! setA[1]! setA[2]!) ∧
  (¬ isPythagoreanTriple setB[0]! setB[1]! setB[2]!) ∧
  (¬ isPythagoreanTriple setC[0]! setC[1]! setC[2]!) ∧
  (∀ (a b c : ℚ), a ∈ setD → b ∈ setD → c ∈ setD → ¬ isPythagoreanTriple a.num b.num c.num) :=
by sorry


end only_setA_is_pythagorean_triple_l422_42200


namespace smallest_n_purple_candy_l422_42222

def orange_candy : ℕ := 10
def yellow_candy : ℕ := 16
def gray_candy : ℕ := 18
def purple_candy_cost : ℕ := 18

theorem smallest_n_purple_candy : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (total_cost : ℕ), 
    total_cost = orange_candy * n ∧
    total_cost = yellow_candy * n ∧
    total_cost = gray_candy * n ∧
    total_cost = purple_candy_cost * n) ∧
  (∀ (m : ℕ), m < n → 
    ¬(∃ (total_cost : ℕ), 
      total_cost = orange_candy * m ∧
      total_cost = yellow_candy * m ∧
      total_cost = gray_candy * m ∧
      total_cost = purple_candy_cost * m)) ∧
  n = 40 :=
sorry

end smallest_n_purple_candy_l422_42222


namespace right_triangle_area_l422_42207

theorem right_triangle_area (h : ℝ) (h_positive : h > 0) :
  let angle_30 : ℝ := 30 * π / 180
  let angle_60 : ℝ := 60 * π / 180
  let angle_90 : ℝ := 90 * π / 180
  h = 4 →
  (1/2) * (h * (2 * h / Real.sqrt 3)) * (h * Real.sqrt 3) = (16 * Real.sqrt 3) / 3 := by
sorry

end right_triangle_area_l422_42207


namespace midpoint_linear_combination_l422_42230

/-- Given two points A and B in the plane, prove that for their midpoint C,
    a specific linear combination of C's coordinates equals -28. -/
theorem midpoint_linear_combination (A B : ℝ × ℝ) (h : A = (10, 15) ∧ B = (-2, 3)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 4 * C.2 = -28 := by
  sorry

end midpoint_linear_combination_l422_42230


namespace partial_fraction_sum_l422_42239

theorem partial_fraction_sum (P Q R : ℚ) : 
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ -1 ∧ x ≠ 2 → 
    (x^2 + 5*x - 14) / ((x - 3)*(x + 1)*(x - 2)) = 
    P / (x - 3) + Q / (x + 1) + R / (x - 2)) →
  P + Q + R = 11.5 / 3 := by
sorry

end partial_fraction_sum_l422_42239


namespace rocket_coaster_total_cars_l422_42296

/-- Represents a roller coaster with two types of cars -/
structure RollerCoaster where
  four_passenger_cars : ℕ
  six_passenger_cars : ℕ
  total_capacity : ℕ

/-- The Rocket Coaster satisfies the given conditions -/
def rocket_coaster : RollerCoaster :=
  { four_passenger_cars := 9,
    six_passenger_cars := 6,
    total_capacity := 72 }

/-- The total number of cars on the Rocket Coaster -/
def total_cars (rc : RollerCoaster) : ℕ :=
  rc.four_passenger_cars + rc.six_passenger_cars

/-- Theorem stating that the total number of cars on the Rocket Coaster is 15 -/
theorem rocket_coaster_total_cars :
  total_cars rocket_coaster = 15 ∧
  4 * rocket_coaster.four_passenger_cars + 6 * rocket_coaster.six_passenger_cars = rocket_coaster.total_capacity :=
by sorry

#eval total_cars rocket_coaster -- Should output 15

end rocket_coaster_total_cars_l422_42296


namespace volume_of_cut_cube_piece_l422_42237

theorem volume_of_cut_cube_piece (cube_edge : ℝ) (piece_base_side : ℝ) (piece_height : ℝ) : 
  cube_edge = 1 →
  piece_base_side = 1/3 →
  piece_height = 1/3 →
  (1/3) * (piece_base_side^2) * piece_height = 1/81 :=
by sorry

end volume_of_cut_cube_piece_l422_42237


namespace hyperbola_eccentricity_l422_42258

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the slope of one of its asymptotes is 2, then its eccentricity is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 2) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_l422_42258


namespace impossibleIdenticalLongNumbers_l422_42232

/-- Represents a long number formed by concatenating integers -/
def LongNumber := List Nat

/-- Checks if a number is in the valid range [0, 999] -/
def isValidNumber (n : Nat) : Prop := n ≤ 999

/-- Splits a list of numbers into two groups -/
def split (numbers : List Nat) : Prop := ∃ (group1 group2 : List Nat), 
  (group1 ++ group2).Perm numbers ∧ group1 ≠ [] ∧ group2 ≠ []

theorem impossibleIdenticalLongNumbers : 
  ¬∃ (numbers : List Nat), 
    (∀ n ∈ numbers, isValidNumber n) ∧ 
    (∀ n, isValidNumber n → n ∈ numbers) ∧
    (∃ (group1 group2 : LongNumber), 
      split numbers ∧ 
      group1.toString = group2.toString) := by
  sorry

end impossibleIdenticalLongNumbers_l422_42232


namespace prob_A_given_B_l422_42249

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
def P (X : Finset Nat) : ℚ := (X.card : ℚ) / (Ω.card : ℚ)

-- Define conditional probability
def conditional_prob (X Y : Finset Nat) : ℚ := P (X ∩ Y) / P Y

-- Theorem statement
theorem prob_A_given_B : conditional_prob A B = 2/5 := by
  sorry

end prob_A_given_B_l422_42249


namespace common_chord_length_l422_42285

/-- The length of the common chord of two overlapping circles -/
theorem common_chord_length (r : ℝ) (h : r = 15) : 
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 15 * Real.sqrt 3 := by
  sorry

#check common_chord_length

end common_chord_length_l422_42285


namespace min_coins_for_change_l422_42269

/-- Represents the available denominations in cents -/
def denominations : List ℕ := [200, 100, 25, 10, 5, 1]

/-- Calculates the minimum number of bills and coins needed for change -/
def minCoins (amount : ℕ) : ℕ :=
  sorry

/-- The change amount in cents -/
def changeAmount : ℕ := 456

/-- Theorem stating that the minimum number of bills and coins for $4.56 change is 6 -/
theorem min_coins_for_change : minCoins changeAmount = 6 := by
  sorry

end min_coins_for_change_l422_42269


namespace circle_properties_l422_42270

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

/-- Point of tangency -/
def point_of_tangency : ℝ × ℝ := (2, 0)

theorem circle_properties :
  (∃ (x y : ℝ), circle_equation x y ∧ x = 0 ∧ y = 0) ∧  -- Passes through origin
  (∀ (x y : ℝ), circle_equation x y → line_equation x y → (x, y) = point_of_tangency) ∧  -- Tangent at (2, 0)
  circle_equation (point_of_tangency.1) (point_of_tangency.2) :=  -- Point (2, 0) is on the circle
by sorry

end circle_properties_l422_42270


namespace negative_difference_l422_42284

theorem negative_difference (m n : ℝ) : -(m - n) = -m + n := by
  sorry

end negative_difference_l422_42284


namespace gcd_of_squares_sum_l422_42209

theorem gcd_of_squares_sum : Nat.gcd (122^2 + 234^2 + 344^2) (123^2 + 235^2 + 343^2) = 1 := by
  sorry

end gcd_of_squares_sum_l422_42209


namespace basketball_free_throws_l422_42279

theorem basketball_free_throws (deshawn kayla annieka : ℕ) : 
  deshawn = 12 →
  annieka = 14 →
  annieka = kayla - 4 →
  (kayla - deshawn) / deshawn * 100 = 50 := by
  sorry

end basketball_free_throws_l422_42279


namespace julians_debt_l422_42256

/-- The amount Julian owes Jenny after borrowing additional money -/
def total_debt (initial_debt : ℕ) (borrowed_amount : ℕ) : ℕ :=
  initial_debt + borrowed_amount

/-- Theorem stating that Julian's total debt is 28 dollars -/
theorem julians_debt : total_debt 20 8 = 28 := by
  sorry

end julians_debt_l422_42256


namespace castle_provisions_duration_l422_42227

/-- 
Proves that given the conditions of the castle's food provisions,
the initial food supply was meant to last 120 days.
-/
theorem castle_provisions_duration 
  (initial_people : ℕ) 
  (people_left : ℕ) 
  (days_before_leaving : ℕ) 
  (days_after_leaving : ℕ) 
  (h1 : initial_people = 300)
  (h2 : people_left = 100)
  (h3 : days_before_leaving = 30)
  (h4 : days_after_leaving = 90)
  : ℕ := by
  sorry

#check castle_provisions_duration

end castle_provisions_duration_l422_42227


namespace arithmetic_sequence_20th_term_l422_42277

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term 
  (a : ℕ → ℝ) 
  (h_sum : a 1 + a 2 + a 3 = 6) 
  (h_5th : a 5 = 8) 
  (h_arith : arithmetic_sequence a) : 
  a 20 = 38 := by
sorry

end arithmetic_sequence_20th_term_l422_42277


namespace second_quadrant_m_range_l422_42205

theorem second_quadrant_m_range (m : ℝ) : 
  (m^2 - 1 < 0 ∧ m > 0) → (0 < m ∧ m < 1) := by sorry

end second_quadrant_m_range_l422_42205


namespace necessary_but_not_sufficient_l422_42245

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, (-3 ≤ x ∧ x ≤ 1) → (x ≤ 2 ∨ x ≥ 3)) ∧
  (∃ x : ℝ, (x ≤ 2 ∨ x ≥ 3) ∧ ¬(-3 ≤ x ∧ x ≤ 1)) := by
  sorry

end necessary_but_not_sufficient_l422_42245


namespace min_cos_C_in_triangle_l422_42250

theorem min_cos_C_in_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_eq : a^2 + 2*b^2 = 3*c^2) :
  let cos_C := (a^2 + b^2 - c^2) / (2*a*b)
  cos_C ≥ Real.sqrt 2 / 3 := by
sorry

end min_cos_C_in_triangle_l422_42250


namespace inscribed_circle_rectangle_area_l422_42280

theorem inscribed_circle_rectangle_area (r : ℝ) (ratio : ℝ) : 
  r > 0 → 
  ratio > 0 → 
  let width := 2 * r
  let length := ratio * width
  let area := length * width
  r = 8 ∧ ratio = 3 → area = 768 := by
  sorry

end inscribed_circle_rectangle_area_l422_42280


namespace line_segment_point_sum_l422_42273

/-- The line equation y = (5/3)x - 15 -/
def line_equation (x y : ℝ) : Prop := y = (5/3) * x - 15

/-- Point P is where the line crosses the x-axis -/
def point_P (x : ℝ) : Prop := line_equation x 0

/-- Point Q is where the line crosses the y-axis -/
def point_Q (y : ℝ) : Prop := line_equation 0 y

/-- Point T(r, s) is on the line -/
def point_T (r s : ℝ) : Prop := line_equation r s

/-- T is between P and Q on the line segment -/
def T_between_P_Q (r s : ℝ) : Prop := 
  ∃ (px qy : ℝ), point_P px ∧ point_Q qy ∧ 
  ((0 ≤ r ∧ r ≤ px) ∨ (px ≤ r ∧ r ≤ 0)) ∧
  ((qy ≤ s ∧ s ≤ 0) ∨ (0 ≤ s ∧ s ≤ qy))

/-- Area of triangle POQ is twice the area of triangle TOQ -/
def area_condition (r s : ℝ) : Prop :=
  ∃ (px qy : ℝ), point_P px ∧ point_Q qy ∧
  (1/2 * px * abs qy) = 2 * (1/2 * px * abs (s - qy))

theorem line_segment_point_sum : 
  ∀ (r s : ℝ), point_T r s ∧ T_between_P_Q r s ∧ area_condition r s → r + s = -3 := by
  sorry

end line_segment_point_sum_l422_42273


namespace inequality_solution_set_l422_42241

-- Define f as a differentiable function on (0, +∞)
variable (f : ℝ → ℝ)

-- Define the domain of f
variable (hf_diff : Differentiable ℝ f)
variable (hf_domain : ∀ x, x > 0 → f x ≠ 0)

-- Define the condition f(x) > x * f'(x)
variable (hf_cond : ∀ x, x > 0 → f x > x * (deriv f x))

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem inequality_solution_set :
  ∀ x > 0, x^2 * f (1/x) - f x < 0 ↔ x ∈ solution_set :=
sorry

end inequality_solution_set_l422_42241


namespace lines_intersect_l422_42297

/-- Represents a line in the form Ax + By + C = 0 --/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Determines if two lines are intersecting --/
def are_intersecting (l1 l2 : Line) : Prop :=
  l1.A * l2.B ≠ l2.A * l1.B

theorem lines_intersect : 
  let line1 : Line := { A := 3, B := -2, C := 5 }
  let line2 : Line := { A := 1, B := 3, C := 10 }
  are_intersecting line1 line2 := by
  sorry

end lines_intersect_l422_42297


namespace pipe_cut_theorem_l422_42265

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 68 →
  difference = 12 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 28 :=
by
  sorry

end pipe_cut_theorem_l422_42265


namespace product_990_sum_93_l422_42291

theorem product_990_sum_93 : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧ 
  (a * b = 990) ∧ 
  (x * y * z = 990) ∧ 
  (a + b + x + y + z = 93) := by
sorry

end product_990_sum_93_l422_42291


namespace door_challenge_sequences_l422_42251

/-- Represents the number of doors and family members -/
def n : ℕ := 7

/-- Represents the number of binary choices made after the first person -/
def m : ℕ := n - 1

/-- The number of possible sequences given n doors and m binary choices -/
def num_sequences (n m : ℕ) : ℕ := 2^m

theorem door_challenge_sequences :
  n = 7 → m = 6 → num_sequences n m = 64 := by
  sorry

end door_challenge_sequences_l422_42251


namespace remainder_after_adding_2947_l422_42226

theorem remainder_after_adding_2947 (n : ℤ) (h : n % 7 = 3) : (n + 2947) % 7 = 3 := by
  sorry

end remainder_after_adding_2947_l422_42226


namespace partial_fraction_decomposition_l422_42257

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 7 ∧ x ≠ -3 →
  (8 * x - 5) / (x^2 - 4 * x - 21) = (51 / 10) / (x - 7) + (29 / 10) / (x + 3) := by
sorry

end partial_fraction_decomposition_l422_42257


namespace earning_goal_proof_l422_42253

/-- Calculates the total earnings for a salesperson given fixed earnings, commission rate, and sales amount. -/
def totalEarnings (fixedEarnings : ℝ) (commissionRate : ℝ) (sales : ℝ) : ℝ :=
  fixedEarnings + commissionRate * sales

/-- Proves that the earning goal is $500 given the specified conditions. -/
theorem earning_goal_proof :
  let fixedEarnings : ℝ := 190
  let commissionRate : ℝ := 0.04
  let minSales : ℝ := 7750
  totalEarnings fixedEarnings commissionRate minSales = 500 := by
  sorry

#eval totalEarnings 190 0.04 7750

end earning_goal_proof_l422_42253


namespace game_problem_l422_42204

/-- Represents the game setup -/
structure GameSetup :=
  (total_boxes : ℕ)
  (valuable_boxes : ℕ)
  (prob_threshold : ℚ)

/-- Calculates the minimum number of boxes to eliminate -/
def min_boxes_to_eliminate (setup : GameSetup) : ℕ :=
  setup.total_boxes - 2 * setup.valuable_boxes

/-- Theorem statement for the game problem -/
theorem game_problem (setup : GameSetup) 
  (h1 : setup.total_boxes = 30)
  (h2 : setup.valuable_boxes = 5)
  (h3 : setup.prob_threshold = 1/2) :
  min_boxes_to_eliminate setup = 20 := by
  sorry

#eval min_boxes_to_eliminate { total_boxes := 30, valuable_boxes := 5, prob_threshold := 1/2 }

end game_problem_l422_42204


namespace range_of_a_l422_42248

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : even_function f)
  (h_incr : increasing_on_neg f)
  (h_cond : f a ≤ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry

end range_of_a_l422_42248


namespace solve_salary_problem_l422_42219

def salary_problem (salary : ℝ) : Prop :=
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 17000

theorem solve_salary_problem : 
  ∃ (salary : ℝ), salary_problem salary ∧ salary = 170000 := by
  sorry

end solve_salary_problem_l422_42219


namespace max_distance_circle_to_line_l422_42255

/-- The maximum distance from any point on the unit circle to the line x - y + 3 = 0 -/
theorem max_distance_circle_to_line : 
  ∃ (d : ℝ), d = (3 * Real.sqrt 2) / 2 + 1 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 1 → 
  |x - y + 3| / Real.sqrt 2 ≤ d ∧
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 1 ∧ |x₀ - y₀ + 3| / Real.sqrt 2 = d :=
sorry

end max_distance_circle_to_line_l422_42255


namespace sqrt_calculations_l422_42213

theorem sqrt_calculations :
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0) ∧
  ((2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6) = 6) := by
  sorry

end sqrt_calculations_l422_42213


namespace constant_function_theorem_l422_42218

theorem constant_function_theorem (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x - y)) →
  ∃ C : ℝ, ∀ x : ℝ, f x = C :=
by sorry

end constant_function_theorem_l422_42218


namespace smallest_sum_with_gcd_lcm_condition_l422_42221

theorem smallest_sum_with_gcd_lcm_condition (a b : ℕ+) : 
  (Nat.gcd a b + Nat.lcm a b = 3 * (a + b)) → 
  (∀ c d : ℕ+, (Nat.gcd c d + Nat.lcm c d = 3 * (c + d)) → (a + b ≤ c + d)) → 
  a + b = 12 :=
sorry

end smallest_sum_with_gcd_lcm_condition_l422_42221


namespace range_of_t_l422_42292

theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, x ≤ t ∧ x^2 - 4*x + t ≤ 0) → 
  0 ≤ t ∧ t ≤ 4 := by
sorry

end range_of_t_l422_42292


namespace cyclic_quadrilateral_area_l422_42211

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral :=
  (A B C D : ℝ × ℝ)
  (is_cyclic : sorry)

/-- The area of a cyclic quadrilateral. -/
def area (q : CyclicQuadrilateral) : ℝ := sorry

/-- The distance between two points in ℝ². -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem cyclic_quadrilateral_area :
  ∀ (q : CyclicQuadrilateral),
    distance q.A q.B = 1 →
    distance q.B q.C = 3 →
    distance q.C q.D = 2 →
    distance q.D q.A = 2 →
    area q = 2 * Real.sqrt 3 := by
  sorry

end cyclic_quadrilateral_area_l422_42211


namespace quadratic_equation_rewrite_l422_42210

theorem quadratic_equation_rewrite :
  ∀ x : ℝ, (-5 * x^2 = 2 * x + 10) ↔ (x^2 + (2/5) * x + 2 = 0) :=
by sorry

end quadratic_equation_rewrite_l422_42210


namespace normal_block_volume_l422_42282

/-- The volume of a normal block of cheese -/
def normal_volume : ℝ := sorry

/-- The volume of a large block of cheese -/
def large_volume : ℝ := 36

/-- The relationship between large and normal block volumes -/
axiom volume_relationship : large_volume = 12 * normal_volume

theorem normal_block_volume : normal_volume = 3 := by sorry

end normal_block_volume_l422_42282


namespace initially_calculated_average_height_l422_42244

theorem initially_calculated_average_height
  (n : ℕ)
  (wrong_height actual_height : ℝ)
  (actual_average : ℝ)
  (h1 : n = 35)
  (h2 : wrong_height = 166)
  (h3 : actual_height = 106)
  (h4 : actual_average = 183) :
  let initially_calculated_average := 
    (n * actual_average - (wrong_height - actual_height)) / n
  initially_calculated_average = 181 := by
sorry

end initially_calculated_average_height_l422_42244


namespace largest_power_of_two_dividing_difference_of_fourth_powers_l422_42215

theorem largest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, k = 5 ∧ 2^k = (Nat.gcd (15^4 - 9^4) (2^32)) :=
by sorry

end largest_power_of_two_dividing_difference_of_fourth_powers_l422_42215


namespace trig_identity_second_quadrant_l422_42225

theorem trig_identity_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) : 
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) + 
  Real.sqrt (1 - Real.sin α ^ 2) / Real.cos α = 1 := by sorry

end trig_identity_second_quadrant_l422_42225


namespace basketball_team_score_l422_42233

theorem basketball_team_score :
  ∀ (tobee jay sean remy alex : ℕ),
  tobee = 4 →
  jay = 2 * tobee + 6 →
  sean = jay / 2 →
  remy = tobee + jay - 3 →
  alex = sean + remy + 4 →
  tobee + jay + sean + remy + alex = 66 :=
by
  sorry

end basketball_team_score_l422_42233


namespace slope_constraint_implies_a_bound_l422_42289

/-- Given a function f(x) = x ln(x) + ax^2, if there exists a point where the slope is 3,
    then a is greater than or equal to -1 / (2e^3). -/
theorem slope_constraint_implies_a_bound (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (Real.log x + 1 + 2 * a * x = 3)) →
  a ≥ -1 / (2 * Real.exp 3) :=
by sorry

end slope_constraint_implies_a_bound_l422_42289


namespace least_positive_integer_for_multiple_of_five_l422_42271

theorem least_positive_integer_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (528 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 → (528 + m) % 5 = 0 → m ≥ n :=
by sorry

end least_positive_integer_for_multiple_of_five_l422_42271


namespace west_movement_negative_l422_42288

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (d : Direction) (distance : ℝ) : ℝ :=
  match d with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem west_movement_negative (distance : ℝ) :
  movement Direction.East distance = distance →
  movement Direction.West distance = -distance :=
by
  sorry

end west_movement_negative_l422_42288


namespace distance_to_town_l422_42220

theorem distance_to_town (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) :=
by sorry

end distance_to_town_l422_42220


namespace perfect_cube_factors_of_4410_l422_42240

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_cube (n : ℕ) : Prop := sorry

def count_perfect_cube_factors (n : ℕ) : ℕ := sorry

theorem perfect_cube_factors_of_4410 :
  let factorization := prime_factorization 4410
  (factorization = [(2, 1), (3, 2), (5, 1), (7, 2)]) →
  (count_perfect_cube_factors 4410 = 1) := by sorry

end perfect_cube_factors_of_4410_l422_42240


namespace star_properties_l422_42266

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 1) - 3

-- Theorem statement
theorem star_properties :
  (¬ ∀ x y : ℝ, star x y = star y x) ∧ 
  (¬ ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) ∧ 
  (star 0 1 = 1) := by
  sorry


end star_properties_l422_42266


namespace sixth_term_value_l422_42252

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem sixth_term_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 6 = 11 := by
  sorry

end sixth_term_value_l422_42252


namespace jakobs_class_size_l422_42298

theorem jakobs_class_size :
  ∃! b : ℕ, 100 < b ∧ b < 200 ∧
    b % 4 = 2 ∧ b % 5 = 2 ∧ b % 6 = 2 ∧
    b = 122 := by sorry

end jakobs_class_size_l422_42298


namespace skater_speeds_l422_42295

theorem skater_speeds (V₁ V₂ : ℝ) (h1 : V₁ > 0) (h2 : V₂ > 0) 
  (h3 : (V₁ + V₂) / |V₁ - V₂| = 4) (h4 : V₁ = 6 ∨ V₂ = 6) :
  (V₁ = 10 ∧ V₂ = 6) ∨ (V₁ = 6 ∧ V₂ = 3.6) := by
  sorry

end skater_speeds_l422_42295


namespace track_length_l422_42262

/-- The length of a track AB given specific meeting points of two athletes --/
theorem track_length (v₁ v₂ : ℝ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) : 
  let x := (v₁ + v₂) * 300 / v₂
  (300 / v₁ = (x - 300) / v₂) ∧ ((x + 100) / v₁ = (x - 100) / v₂) → x = 500 := by
  sorry

#check track_length

end track_length_l422_42262


namespace x_value_in_equation_l422_42224

theorem x_value_in_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 71) : x = 8 := by
  sorry

end x_value_in_equation_l422_42224


namespace max_hollow_cube_volume_l422_42243

/-- The number of available unit cubes --/
def available_cubes : ℕ := 1000

/-- Function to calculate the number of cubes used for a given side length --/
def cubes_used (x : ℕ) : ℕ :=
  2 * x^2 + 2 * x * (x - 2) + 2 * (x - 2)^2

/-- The maximum side length that can be achieved --/
def max_side_length : ℕ := 13

/-- Theorem stating the maximum volume that can be achieved --/
theorem max_hollow_cube_volume :
  (∀ x : ℕ, cubes_used x ≤ available_cubes → x ≤ max_side_length) ∧
  cubes_used max_side_length ≤ available_cubes ∧
  max_side_length^3 = 2197 :=
sorry

end max_hollow_cube_volume_l422_42243


namespace square_difference_equality_l422_42267

theorem square_difference_equality : (25 + 15 + 8)^2 - (25^2 + 15^2 + 8^2) = 1390 := by
  sorry

end square_difference_equality_l422_42267


namespace complex_equation_solution_l422_42259

theorem complex_equation_solution (z : ℂ) :
  z * (2 - 3*I) = 6 + 4*I → z = 2*I := by
  sorry

end complex_equation_solution_l422_42259


namespace complex_equation_proof_l422_42235

theorem complex_equation_proof (a b : ℝ) : 
  (a + b * Complex.I) / (2 - Complex.I) = (3 : ℂ) + Complex.I → a - b = 8 := by
  sorry

end complex_equation_proof_l422_42235


namespace total_seashells_equation_l422_42212

/-- The number of seashells Fred found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Fred gave to Jessica -/
def seashells_given : ℕ := 25

/-- The number of seashells Fred has left -/
def seashells_left : ℕ := 22

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem total_seashells_equation : total_seashells = seashells_given + seashells_left := by sorry

end total_seashells_equation_l422_42212


namespace parabola_properties_l422_42276

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 9 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ :=
  (1, -3)

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 9*x

theorem parabola_properties :
  (∀ x y, parabola_equation x y → parabola_equation x (-y)) ∧ 
  parabola_equation 0 0 ∧
  parabola_equation (circle_center.1) (circle_center.2) :=
sorry

end parabola_properties_l422_42276


namespace maximum_marks_l422_42234

theorem maximum_marks : ∃ M : ℕ, 
  (M ≥ 434) ∧ 
  (M < 435) ∧ 
  (⌈(0.45 : ℝ) * (M : ℝ)⌉ = 130 + 65) := by
  sorry

end maximum_marks_l422_42234


namespace polynomial_sum_l422_42281

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum (a b c d : ℝ) :
  g a b c d (3*I) = 0 ∧ g a b c d (1 + I) = 0 → a + b + c + d = 9 := by
  sorry

end polynomial_sum_l422_42281


namespace inequality_proof_l422_42260

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) :=
by
  sorry

end inequality_proof_l422_42260


namespace sqrt_12_minus_sqrt_3_l422_42208

theorem sqrt_12_minus_sqrt_3 : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_12_minus_sqrt_3_l422_42208


namespace fraction_comparisons_l422_42216

theorem fraction_comparisons :
  ∀ (a b c : ℚ),
  (0 < b) → (b < 1) → (a * b < a) ∧
  (0 < c) → (c < b) → (a * b > a * c) ∧
  (0 < b) → (b < 1) → (a < a / b) :=
by sorry

end fraction_comparisons_l422_42216


namespace simplify_expression_l422_42268

theorem simplify_expression (w : ℝ) : (5 - 2*w) - (4 + 5*w) = 1 - 7*w := by
  sorry

end simplify_expression_l422_42268


namespace whole_number_between_l422_42290

theorem whole_number_between : 
  ∃ (M : ℕ), (8 : ℚ) < (M : ℚ) / 4 ∧ (M : ℚ) / 4 < 9 → M = 33 :=
by sorry

end whole_number_between_l422_42290
