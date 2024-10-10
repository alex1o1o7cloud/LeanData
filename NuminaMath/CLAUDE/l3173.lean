import Mathlib

namespace quadratic_root_in_unit_interval_l3173_317319

theorem quadratic_root_in_unit_interval (a b : ℝ) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 3 * a * x^2 + 2 * b * x - (a + b) = 0 := by
  sorry

end quadratic_root_in_unit_interval_l3173_317319


namespace ceiling_negative_three_point_seven_l3173_317385

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_negative_three_point_seven_l3173_317385


namespace triangle_inequality_relationships_l3173_317393

/-- A triangle with perimeter, circumradius, and inradius -/
structure Triangle where
  perimeter : ℝ
  circumradius : ℝ
  inradius : ℝ
  perimeter_pos : 0 < perimeter
  circumradius_pos : 0 < circumradius
  inradius_pos : 0 < inradius

/-- Theorem stating that none of the given relationships hold universally for all triangles -/
theorem triangle_inequality_relationships (t : Triangle) : 
  ¬(∀ t : Triangle, t.perimeter > t.circumradius + t.inradius) ∧ 
  ¬(∀ t : Triangle, t.perimeter ≤ t.circumradius + t.inradius) ∧ 
  ¬(∀ t : Triangle, 1/6 < t.circumradius + t.inradius ∧ t.circumradius + t.inradius < 6*t.perimeter) :=
by sorry

end triangle_inequality_relationships_l3173_317393


namespace motorboat_problem_l3173_317371

/-- Represents the problem of calculating the time taken by a motorboat to reach an island in still water -/
theorem motorboat_problem (downstream_distance : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) (island_distance : ℝ) :
  downstream_distance = 160 →
  downstream_time = 8 →
  upstream_time = 16 →
  island_distance = 100 →
  ∃ (boat_speed : ℝ) (current_speed : ℝ),
    boat_speed + current_speed = downstream_distance / downstream_time ∧
    boat_speed - current_speed = downstream_distance / upstream_time ∧
    island_distance / boat_speed = 20 / 3 :=
by sorry

end motorboat_problem_l3173_317371


namespace min_value_x_plus_2y_l3173_317341

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) :
  ∀ z, z = x + 2*y → z ≥ 4 :=
sorry

end min_value_x_plus_2y_l3173_317341


namespace some_number_problem_l3173_317384

theorem some_number_problem (n : ℝ) :
  (∃ x₁ x₂ : ℝ, |x₁ - n| = 50 ∧ |x₂ - n| = 50 ∧ x₁ + x₂ = 50) →
  n = 25 :=
by sorry

end some_number_problem_l3173_317384


namespace route_down_length_l3173_317320

/-- Represents the hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time : ℝ
  rate_down_factor : ℝ

/-- Calculates the distance of a route given rate and time -/
def route_distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: The route down the mountain is 18 miles long -/
theorem route_down_length (hike : MountainHike)
  (h1 : hike.rate_up = 6)
  (h2 : hike.time = 2)
  (h3 : hike.rate_down_factor = 1.5) :
  route_distance (hike.rate_up * hike.rate_down_factor) hike.time = 18 := by
  sorry

#check route_down_length

end route_down_length_l3173_317320


namespace journey_speed_problem_l3173_317317

theorem journey_speed_problem (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 448 →
  total_time = 20 →
  second_half_speed = 24 →
  ∃ first_half_speed : ℝ,
    first_half_speed * (total_time / 2) = total_distance / 2 ∧
    second_half_speed * (total_time / 2) = total_distance / 2 ∧
    first_half_speed = 21 := by
  sorry


end journey_speed_problem_l3173_317317


namespace gcd_490_910_l3173_317303

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end gcd_490_910_l3173_317303


namespace redwood_percentage_increase_l3173_317374

theorem redwood_percentage_increase (num_pines : ℕ) (total_trees : ℕ) : 
  num_pines = 600 → total_trees = 1320 → 
  (total_trees - num_pines : ℚ) / num_pines * 100 = 20 := by
sorry

end redwood_percentage_increase_l3173_317374


namespace arithmetic_sequence_sum_times_three_l3173_317333

theorem arithmetic_sequence_sum_times_three : 
  ∀ (a l d n : ℕ), 
    a = 50 → 
    l = 95 → 
    d = 3 → 
    n * d = l - a + d → 
    3 * (n / 2 * (a + l)) = 3480 := by
  sorry

end arithmetic_sequence_sum_times_three_l3173_317333


namespace ones_digit_of_7_to_53_l3173_317337

theorem ones_digit_of_7_to_53 : (7^53 : ℕ) % 10 = 7 := by sorry

end ones_digit_of_7_to_53_l3173_317337


namespace cab_driver_income_l3173_317355

/-- Theorem: Given a cab driver's income for 5 days where 4 days are known and the average income,
    prove that the income for the unknown day is as calculated. -/
theorem cab_driver_income 
  (day1 day2 day3 day5 : ℕ) 
  (average : ℕ) 
  (h1 : day1 = 300)
  (h2 : day2 = 150)
  (h3 : day3 = 750)
  (h5 : day5 = 500)
  (h_avg : average = 420)
  : ∃ day4 : ℕ, 
    day4 = 400 ∧ 
    (day1 + day2 + day3 + day4 + day5) / 5 = average :=
by
  sorry


end cab_driver_income_l3173_317355


namespace inverse_variation_solution_l3173_317386

/-- The inverse relationship between 5y and x^2 -/
def inverse_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 5 * y = k / (x ^ 2)

theorem inverse_variation_solution (x₀ y₀ x₁ : ℝ) 
  (h₀ : inverse_relation x₀ y₀)
  (h₁ : x₀ = 2)
  (h₂ : y₀ = 4)
  (h₃ : x₁ = 4) :
  ∃ y₁ : ℝ, inverse_relation x₁ y₁ ∧ y₁ = 1 := by
  sorry

end inverse_variation_solution_l3173_317386


namespace height_is_four_l3173_317356

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- A right-angled triangle on a parabola -/
structure RightTriangleOnParabola where
  A : ParabolaPoint
  B : ParabolaPoint
  C : ParabolaPoint
  right_angle_at_C : (B.x - C.x) * (A.x - C.x) + (B.y - C.y) * (A.y - C.y) = 0
  hypotenuse_parallel_to_y : A.x = B.x

/-- The height from the hypotenuse of a right-angled triangle on a parabola -/
def height_from_hypotenuse (t : RightTriangleOnParabola) : ℝ :=
  |t.B.x - t.C.x|

/-- Theorem: The height from the hypotenuse is 4 -/
theorem height_is_four (t : RightTriangleOnParabola) : height_from_hypotenuse t = 4 := by
  sorry

end height_is_four_l3173_317356


namespace solution_sum_l3173_317344

theorem solution_sum (c d : ℝ) : 
  c^2 - 6*c + 15 = 27 →
  d^2 - 6*d + 15 = 27 →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end solution_sum_l3173_317344


namespace seating_arrangements_l3173_317372

def total_people : ℕ := 10
def restricted_group : ℕ := 4

def arrangements_with_restriction (n : ℕ) (k : ℕ) : ℕ :=
  n.factorial - (n - k + 1).factorial * k.factorial

theorem seating_arrangements :
  arrangements_with_restriction total_people restricted_group = 3507840 := by
  sorry

end seating_arrangements_l3173_317372


namespace simplify_fraction_l3173_317387

theorem simplify_fraction (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  (18 * a * b^3 * c^2) / (12 * a^2 * b * c) = 27 := by
  sorry

end simplify_fraction_l3173_317387


namespace container_volume_increase_l3173_317314

/-- Given a container with an initial volume and a volume multiplier, 
    calculate the new volume after applying the multiplier. -/
def new_volume (initial_volume : ℝ) (volume_multiplier : ℝ) : ℝ :=
  initial_volume * volume_multiplier

/-- Theorem: If a container's volume is multiplied by 16, and its original volume was 5 gallons,
    then the new volume is 80 gallons. -/
theorem container_volume_increase :
  let initial_volume : ℝ := 5
  let volume_multiplier : ℝ := 16
  new_volume initial_volume volume_multiplier = 80 := by
  sorry

end container_volume_increase_l3173_317314


namespace pascal_triangle_prob_one_20_l3173_317382

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Set ℕ := sorry

/-- The number of elements in the first n rows of Pascal's Triangle -/
def numElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle -/
def numOnes (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probSelectOne (n : ℕ) : ℚ := (numOnes n : ℚ) / (numElements n : ℚ)

theorem pascal_triangle_prob_one_20 : 
  probSelectOne 20 = 39 / 210 := by sorry

end pascal_triangle_prob_one_20_l3173_317382


namespace marys_max_earnings_l3173_317308

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : Nat
  regularRate : ℕ
  overtimeRate1 : ℕ
  overtimeRate2 : ℕ
  weekendBonus : ℕ
  milestoneBonus : ℕ

/-- Calculates the maximum weekly earnings based on the given work schedule --/
def maxWeeklyEarnings (schedule : WorkSchedule) : ℕ :=
  let regularPay := schedule.regularRate * 40
  let overtimePay1 := schedule.overtimeRate1 * 10
  let overtimePay2 := schedule.overtimeRate2 * 10
  let weekendBonus := schedule.weekendBonus * 2
  regularPay + overtimePay1 + overtimePay2 + weekendBonus + schedule.milestoneBonus

/-- Mary's work schedule --/
def marysSchedule : WorkSchedule := {
  maxHours := 60
  regularRate := 10
  overtimeRate1 := 12
  overtimeRate2 := 15
  weekendBonus := 50
  milestoneBonus := 100
}

/-- Theorem stating that Mary's maximum weekly earnings are $875 --/
theorem marys_max_earnings :
  maxWeeklyEarnings marysSchedule = 875 := by
  sorry


end marys_max_earnings_l3173_317308


namespace imaginary_part_of_iz_l3173_317318

theorem imaginary_part_of_iz (z : ℂ) (h : z^2 - 4*z + 5 = 0) : 
  Complex.im (Complex.I * z) = 2 := by
  sorry

end imaginary_part_of_iz_l3173_317318


namespace joes_total_lift_weight_l3173_317379

/-- The total weight of Joe's two lifts is 1500 pounds, given the conditions of the weight-lifting competition. -/
theorem joes_total_lift_weight :
  let first_lift : ℕ := 600
  let second_lift : ℕ := 2 * first_lift - 300
  first_lift + second_lift = 1500 := by
  sorry

end joes_total_lift_weight_l3173_317379


namespace mikes_painting_area_l3173_317376

/-- The area Mike needs to paint on the wall -/
def area_to_paint (wall_height wall_length window_height window_length painting_side : ℝ) : ℝ :=
  wall_height * wall_length - (window_height * window_length + painting_side * painting_side)

/-- Theorem stating the area Mike needs to paint -/
theorem mikes_painting_area :
  area_to_paint 10 15 3 5 2 = 131 := by
  sorry

end mikes_painting_area_l3173_317376


namespace exam_average_l3173_317396

theorem exam_average (total_candidates : ℕ) (passed_candidates : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) 
  (h1 : total_candidates = 120)
  (h2 : passed_candidates = 100)
  (h3 : passed_avg = 39)
  (h4 : failed_avg = 15) :
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := passed_avg * passed_candidates + failed_avg * failed_candidates
  total_marks / total_candidates = 35 := by
sorry

end exam_average_l3173_317396


namespace charles_whistle_count_l3173_317315

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end charles_whistle_count_l3173_317315


namespace student_decrease_percentage_l3173_317353

theorem student_decrease_percentage
  (initial_students : ℝ)
  (initial_price : ℝ)
  (price_increase : ℝ)
  (consumption_decrease : ℝ)
  (h1 : price_increase = 0.20)
  (h2 : consumption_decrease = 0.074074074074074066)
  (h3 : initial_students > 0)
  (h4 : initial_price > 0) :
  let new_price := initial_price * (1 + price_increase)
  let new_consumption := 1 - consumption_decrease
  let new_students := initial_students * (1 - 0.10)
  initial_students * initial_price = new_students * new_price * new_consumption :=
by sorry

end student_decrease_percentage_l3173_317353


namespace pipe_A_rate_correct_l3173_317368

/-- Represents the rate at which pipe A fills the tank -/
def pipe_A_rate : ℝ := 40

/-- Represents the rate at which pipe B fills the tank -/
def pipe_B_rate : ℝ := 30

/-- Represents the rate at which pipe C drains the tank -/
def pipe_C_rate : ℝ := 20

/-- Represents the capacity of the tank -/
def tank_capacity : ℝ := 850

/-- Represents the time it takes to fill the tank -/
def fill_time : ℝ := 51

/-- Represents the duration of one cycle -/
def cycle_duration : ℝ := 3

/-- Theorem stating that pipe A's rate satisfies the given conditions -/
theorem pipe_A_rate_correct : 
  (fill_time / cycle_duration) * (pipe_A_rate + pipe_B_rate - pipe_C_rate) = tank_capacity :=
by sorry

end pipe_A_rate_correct_l3173_317368


namespace quadratic_polynomial_theorem_l3173_317349

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Divisibility condition for the polynomial -/
def DivisibilityCondition (q : QuadraticPolynomial ℝ) : Prop :=
  ∃ p : ℝ → ℝ, ∀ x, q x ^ 3 - x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_theorem (q : QuadraticPolynomial ℝ) 
  (h : DivisibilityCondition q) : q 10 = 10 := by
  sorry

end quadratic_polynomial_theorem_l3173_317349


namespace fair_coin_probability_difference_l3173_317340

def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

theorem fair_coin_probability_difference :
  let p3 := probability_k_heads 5 3
  let p4 := probability_k_heads 5 4
  |p3 - p4| = 5 / 32 := by
  sorry

end fair_coin_probability_difference_l3173_317340


namespace kaleb_total_score_l3173_317339

/-- Kaleb's score in the first half of the game -/
def first_half_score : ℕ := 43

/-- Kaleb's score in the second half of the game -/
def second_half_score : ℕ := 23

/-- Kaleb's total score in the game -/
def total_score : ℕ := first_half_score + second_half_score

/-- Theorem stating that Kaleb's total score is 66 points -/
theorem kaleb_total_score : total_score = 66 := by
  sorry

end kaleb_total_score_l3173_317339


namespace sqrt_seven_to_six_l3173_317302

theorem sqrt_seven_to_six : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_six_l3173_317302


namespace triangle_side_length_l3173_317334

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- B = 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given condition
  (b^2 = 8) →  -- Equivalent to b = 2√2
  b = 2 * Real.sqrt 2 :=
by sorry

end triangle_side_length_l3173_317334


namespace cave_door_weight_calculation_l3173_317346

/-- The weight already on the switch (in pounds) -/
def weight_on_switch : ℕ := 234

/-- The total weight needed to open the cave doors (in pounds) -/
def total_weight_needed : ℕ := 712

/-- The additional weight needed to open the cave doors (in pounds) -/
def additional_weight_needed : ℕ := total_weight_needed - weight_on_switch

theorem cave_door_weight_calculation :
  additional_weight_needed = 478 := by
  sorry

end cave_door_weight_calculation_l3173_317346


namespace upstream_downstream_time_ratio_l3173_317309

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 --/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 51)
  (h2 : stream_speed = 17) : 
  (boat_speed + stream_speed) / (boat_speed - stream_speed) = 2 := by
  sorry

#check upstream_downstream_time_ratio

end upstream_downstream_time_ratio_l3173_317309


namespace inequality_proof_l3173_317323

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  3/16 ≤ (a/(1+a))^2 + (b/(1+b))^2 + (c/(1+c))^2 ∧ (a/(1+a))^2 + (b/(1+b))^2 + (c/(1+c))^2 ≤ 1/4 := by
  sorry

end inequality_proof_l3173_317323


namespace union_A_B_intersection_complement_A_B_l3173_317331

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x < 10}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | 3 ≤ x ∧ x < 10} := by sorry

-- Theorem for the intersection of complement of A and B
theorem intersection_complement_A_B : (Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x < 10} := by sorry

end union_A_B_intersection_complement_A_B_l3173_317331


namespace tan_product_pi_ninths_l3173_317377

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end tan_product_pi_ninths_l3173_317377


namespace three_greater_than_negative_five_l3173_317307

theorem three_greater_than_negative_five : 3 > -5 := by
  sorry

end three_greater_than_negative_five_l3173_317307


namespace sine_amplitude_l3173_317335

theorem sine_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d) 
  (h6 : ∀ x, a * Real.sin (b * x + c) + d ≤ 5) : a = 4 := by
sorry

end sine_amplitude_l3173_317335


namespace absolute_value_sum_l3173_317395

theorem absolute_value_sum : -2 + |(-3)| = 1 := by
  sorry

end absolute_value_sum_l3173_317395


namespace no_prime_sum_power_four_l3173_317378

theorem no_prime_sum_power_four (n : ℕ+) : ¬ Prime (4^(n : ℕ) + (n : ℕ)^4) := by
  sorry

end no_prime_sum_power_four_l3173_317378


namespace sector_area_l3173_317328

/-- Given a circular sector with central angle 2 radians and circumference 4 cm, its area is 1 cm² -/
theorem sector_area (θ : ℝ) (c : ℝ) (A : ℝ) : 
  θ = 2 → c = 4 → A = 1 → A = (θ * c^2) / (8 * π) := by
  sorry

end sector_area_l3173_317328


namespace least_integer_greater_than_sqrt_500_l3173_317316

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
  sorry

end least_integer_greater_than_sqrt_500_l3173_317316


namespace n_to_b_equals_sixteen_l3173_317326

-- Define n and b
def n : ℝ := 2 ^ (1 / 4)
def b : ℝ := 16.000000000000004

-- Theorem statement
theorem n_to_b_equals_sixteen : n ^ b = 16 := by
  sorry

end n_to_b_equals_sixteen_l3173_317326


namespace omega_sum_equality_l3173_317350

theorem omega_sum_equality (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 = 1 := by
sorry

end omega_sum_equality_l3173_317350


namespace straight_row_not_tetrahedron_l3173_317391

/-- A pattern of squares that can be folded -/
structure FoldablePattern :=
  (squares : ℕ)
  (arrangement : String)

/-- Properties of a regular tetrahedron -/
structure RegularTetrahedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Definition of a straight row pattern -/
def straightRowPattern : FoldablePattern :=
  { squares := 4,
    arrangement := "straight row" }

/-- Definition of a regular tetrahedron -/
def regularTetrahedron : RegularTetrahedron :=
  { faces := 4,
    edges := 6,
    vertices := 4 }

/-- Function to check if a pattern can be folded into a regular tetrahedron -/
def canFoldToTetrahedron (pattern : FoldablePattern) : Prop :=
  ∃ (t : RegularTetrahedron), t = regularTetrahedron

/-- Theorem stating that a straight row pattern cannot be folded into a regular tetrahedron -/
theorem straight_row_not_tetrahedron :
  ¬(canFoldToTetrahedron straightRowPattern) :=
sorry

end straight_row_not_tetrahedron_l3173_317391


namespace photo_arrangements_count_l3173_317304

/-- The number of people in the group --/
def group_size : ℕ := 5

/-- The number of arrangements where two specific people are adjacent --/
def adjacent_arrangements : ℕ := 2 * (group_size - 1).factorial

/-- The number of arrangements where three specific people are adjacent --/
def triple_adjacent_arrangements : ℕ := 2 * (group_size - 2).factorial

/-- The number of valid arrangements --/
def valid_arrangements : ℕ := adjacent_arrangements - triple_adjacent_arrangements

theorem photo_arrangements_count : valid_arrangements = 36 := by
  sorry

end photo_arrangements_count_l3173_317304


namespace race_distance_l3173_317351

/-- The race problem -/
theorem race_distance (speed_A speed_B : ℝ) (head_start win_margin total_distance : ℝ) :
  speed_A > 0 ∧ speed_B > 0 →
  speed_A / speed_B = 3 / 4 →
  head_start = 200 →
  win_margin = 100 →
  total_distance / speed_A = (total_distance - head_start - win_margin) / speed_B →
  total_distance = 900 := by
  sorry

#check race_distance

end race_distance_l3173_317351


namespace buratino_problem_l3173_317362

/-- The sum of a geometric sequence with first term 1 and common ratio 2 -/
def geometricSum (n : ℕ) : ℕ := 2^n - 1

/-- The total payment in kopeks -/
def totalPayment : ℕ := 65535

theorem buratino_problem :
  ∃ n : ℕ, geometricSum n = totalPayment ∧ n = 16 := by
  sorry

end buratino_problem_l3173_317362


namespace singer_hire_duration_l3173_317321

theorem singer_hire_duration (hourly_rate : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h_rate : hourly_rate = 15)
  (h_tip : tip_percentage = 0.20)
  (h_total : total_paid = 54) :
  ∃ (hours : ℝ), hours = 3 ∧ 
    hourly_rate * hours * (1 + tip_percentage) = total_paid :=
by sorry

end singer_hire_duration_l3173_317321


namespace restaurant_bill_with_discounts_l3173_317322

theorem restaurant_bill_with_discounts
  (bob_bill : ℝ) (kate_bill : ℝ) (bob_discount_rate : ℝ) (kate_discount_rate : ℝ)
  (h_bob_bill : bob_bill = 30)
  (h_kate_bill : kate_bill = 25)
  (h_bob_discount : bob_discount_rate = 0.05)
  (h_kate_discount : kate_discount_rate = 0.02) :
  bob_bill * (1 - bob_discount_rate) + kate_bill * (1 - kate_discount_rate) = 53 := by
  sorry

end restaurant_bill_with_discounts_l3173_317322


namespace haruto_tomatoes_l3173_317367

def tomato_problem (initial : ℕ) (eaten : ℕ) (remaining : ℕ) (given : ℕ) (left : ℕ) : Prop :=
  (initial - eaten = remaining) ∧
  (remaining / 2 = given) ∧
  (remaining - given = left)

theorem haruto_tomatoes : tomato_problem 127 19 108 54 54 := by
  sorry

end haruto_tomatoes_l3173_317367


namespace no_non_zero_solutions_l3173_317300

theorem no_non_zero_solutions (a b : ℝ) : 
  (Real.sqrt (a^2 + b^2) = a^2 - b^2 → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = |a - b| → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = (a + b) / 2 → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = a^3 - b^3 → a = 0 ∧ b = 0) := by
  sorry

end no_non_zero_solutions_l3173_317300


namespace right_triangle_inscribed_circle_inequality_l3173_317370

theorem right_triangle_inscribed_circle_inequality (a b r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : r > 0)
  (h_right_triangle : a^2 + b^2 = (a + b)^2 / 2)
  (h_inscribed_circle : r = a * b / (a + b + Real.sqrt (a^2 + b^2))) :
  2 + Real.sqrt 2 ≤ (2 * a * b) / ((a + b) * r) ∧ (2 * a * b) / ((a + b) * r) < 4 := by
  sorry

end right_triangle_inscribed_circle_inequality_l3173_317370


namespace sidewalk_snow_volume_l3173_317363

theorem sidewalk_snow_volume (length width height : ℝ) 
  (h1 : length = 15)
  (h2 : width = 3)
  (h3 : height = 0.6) :
  length * width * height = 27 :=
by
  sorry

end sidewalk_snow_volume_l3173_317363


namespace ratio_x_to_y_l3173_317383

theorem ratio_x_to_y (x y : ℚ) (h : (12*x - 5*y) / (15*x - 4*y) = 4/7) : x/y = 19/24 := by
  sorry

end ratio_x_to_y_l3173_317383


namespace boat_trip_distance_l3173_317342

/-- Proves that given a boat with speed 8 kmph in standing water, a stream with speed 6 kmph,
    and a round trip time of 120 hours, the distance to the destination is 210 km. -/
theorem boat_trip_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : total_time = 120) : 
  (boat_speed + stream_speed) * (boat_speed - stream_speed) * total_time / 
  (2 * (boat_speed + stream_speed) * (boat_speed - stream_speed)) = 210 := by
  sorry

end boat_trip_distance_l3173_317342


namespace point_movement_l3173_317329

def initial_point : ℝ × ℝ := (-2, -3)

def move_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 - units, p.2)

def move_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

theorem point_movement :
  let p := initial_point
  let p' := move_left p 1
  let p'' := move_up p' 3
  p'' = (-3, 0) := by sorry

end point_movement_l3173_317329


namespace black_marble_probability_l3173_317388

theorem black_marble_probability (yellow blue green black : ℕ) 
  (h1 : yellow = 12)
  (h2 : blue = 10)
  (h3 : green = 5)
  (h4 : black = 1) :
  (black * 14000) / (yellow + blue + green + black) = 500 := by
  sorry

end black_marble_probability_l3173_317388


namespace combination_sum_equality_l3173_317357

def combination (n m : ℕ) : ℚ :=
  if n ≥ m then
    (List.range m).foldl (λ acc i => acc * (n - i : ℚ) / (i + 1)) 1
  else 0

theorem combination_sum_equality : combination 9 4 + combination 9 5 = combination 10 5 := by
  sorry

end combination_sum_equality_l3173_317357


namespace new_students_weight_l3173_317343

/-- Given a class of 8 students, prove that when two students weighing 70kg and 80kg 
    are replaced and the average weight decreases by 2 kg, 
    the combined weight of the two new students is 134 kg. -/
theorem new_students_weight (total_weight : ℝ) : 
  (total_weight - 150 + 134) / 8 = total_weight / 8 - 2 := by
  sorry

end new_students_weight_l3173_317343


namespace fractional_equation_solution_l3173_317394

theorem fractional_equation_solution : 
  ∃ (x : ℝ), (x ≠ 0 ∧ x ≠ 2) ∧ (5 / (x - 2) = 3 / x) ∧ (x = -3) := by
  sorry

end fractional_equation_solution_l3173_317394


namespace number_of_boys_l3173_317310

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 300 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 75 := by
sorry

end number_of_boys_l3173_317310


namespace equation_solution_l3173_317389

theorem equation_solution :
  let f (x : ℝ) := 4 / (Real.sqrt (x + 5) - 7) + 3 / (Real.sqrt (x + 5) - 2) +
                   6 / (Real.sqrt (x + 5) + 2) + 9 / (Real.sqrt (x + 5) + 7)
  {x : ℝ | f x = 0} = {-796/169, 383/22} := by
sorry

end equation_solution_l3173_317389


namespace evaluate_expression_l3173_317306

theorem evaluate_expression : (3^2)^2 - (2^3)^3 = -431 := by
  sorry

end evaluate_expression_l3173_317306


namespace equation_solution_l3173_317305

theorem equation_solution :
  ∃! x : ℝ, (1 : ℝ) / (x - 2) = (3 : ℝ) / (x - 5) ∧ x = (1 : ℝ) / 2 := by
  sorry

end equation_solution_l3173_317305


namespace fraction_decomposition_l3173_317325

theorem fraction_decomposition (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (x^2 - 2*x + 5) / (x^3 - x) = (-5)/x + (6*x - 2) / (x^2 - 1) :=
by sorry

end fraction_decomposition_l3173_317325


namespace younger_brother_bricks_l3173_317358

theorem younger_brother_bricks (total_bricks : ℕ) (final_difference : ℕ) : 
  total_bricks = 26 ∧ final_difference = 2 → 
  ∃ (initial_younger : ℕ), 
    initial_younger = 16 ∧
    (total_bricks - initial_younger) + (initial_younger / 2) - 
    ((total_bricks - initial_younger + (initial_younger / 2)) / 2) + 5 = 
    initial_younger - (initial_younger / 2) + 
    ((total_bricks - initial_younger + (initial_younger / 2)) / 2) - 5 + final_difference :=
by
  sorry

#check younger_brother_bricks

end younger_brother_bricks_l3173_317358


namespace reciprocal_inequality_l3173_317366

theorem reciprocal_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 1 / x < 1 / y := by
  sorry

end reciprocal_inequality_l3173_317366


namespace fault_line_current_movement_l3173_317345

/-- The movement of a fault line over two years -/
structure FaultLineMovement where
  total : ℝ  -- Total movement over two years
  previous : ℝ  -- Movement in the previous year
  current : ℝ  -- Movement in the current year

/-- Theorem: Given the total movement and previous year's movement, 
    calculate the current year's movement -/
theorem fault_line_current_movement 
  (f : FaultLineMovement) 
  (h1 : f.total = 6.5) 
  (h2 : f.previous = 5.25) 
  (h3 : f.total = f.previous + f.current) : 
  f.current = 1.25 := by
  sorry

end fault_line_current_movement_l3173_317345


namespace second_bouquet_carnations_proof_l3173_317364

/-- The number of carnations in the second bouquet -/
def second_bouquet_carnations : ℕ := 14

/-- The number of bouquets -/
def num_bouquets : ℕ := 3

/-- The number of carnations in the first bouquet -/
def first_bouquet_carnations : ℕ := 9

/-- The number of carnations in the third bouquet -/
def third_bouquet_carnations : ℕ := 13

/-- The average number of carnations per bouquet -/
def average_carnations : ℕ := 12

theorem second_bouquet_carnations_proof :
  (first_bouquet_carnations + second_bouquet_carnations + third_bouquet_carnations) / num_bouquets = average_carnations :=
by sorry

end second_bouquet_carnations_proof_l3173_317364


namespace people_speaking_both_languages_l3173_317312

/-- Given a group of people with specified language abilities, calculate the number who speak both languages. -/
theorem people_speaking_both_languages 
  (total : ℕ) 
  (latin : ℕ) 
  (french : ℕ) 
  (neither : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_neither : neither = 6) :
  latin + french - (total - neither) = 9 := by
sorry

end people_speaking_both_languages_l3173_317312


namespace greatest_common_divisor_of_three_shared_l3173_317392

theorem greatest_common_divisor_of_three_shared (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
   d1 ∣ 120 ∧ d2 ∣ 120 ∧ d3 ∣ 120 ∧
   d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
   (∀ (x : ℕ), x ∣ 120 ∧ x ∣ n → x = d1 ∨ x = d2 ∨ x = d3)) →
  (∃ (d : ℕ), d ∣ 120 ∧ d ∣ n ∧ d = 9 ∧ 
   (∀ (x : ℕ), x ∣ 120 ∧ x ∣ n → x ≤ d)) :=
by sorry

end greatest_common_divisor_of_three_shared_l3173_317392


namespace distinct_primes_in_product_l3173_317324

theorem distinct_primes_in_product : 
  let n := 12 * 13 * 14 * 15
  Finset.card (Nat.factors n).toFinset = 5 := by
sorry

end distinct_primes_in_product_l3173_317324


namespace max_value_sqrt_sum_l3173_317365

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧ 
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end max_value_sqrt_sum_l3173_317365


namespace max_sum_of_two_integers_l3173_317354

theorem max_sum_of_two_integers (x y : ℕ+) : 
  y = 2 * x → x + y < 100 → (∀ a b : ℕ+, b = 2 * a → a + b < 100 → a + b ≤ x + y) → x + y = 99 :=
by sorry

end max_sum_of_two_integers_l3173_317354


namespace haley_concert_spending_l3173_317361

def ticket_price : ℕ := 4
def tickets_for_self_and_friends : ℕ := 3
def extra_tickets : ℕ := 5

theorem haley_concert_spending :
  (tickets_for_self_and_friends + extra_tickets) * ticket_price = 32 := by
  sorry

end haley_concert_spending_l3173_317361


namespace largest_prime_common_divisor_l3173_317373

theorem largest_prime_common_divisor :
  ∃ (n : ℕ), n.Prime ∧ n ∣ 360 ∧ n ∣ 231 ∧
  ∀ (m : ℕ), m.Prime → m ∣ 360 → m ∣ 231 → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_prime_common_divisor_l3173_317373


namespace quadratic_inequality_range_l3173_317399

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ k ∈ Set.Ioo (-3) 0 := by
  sorry

end quadratic_inequality_range_l3173_317399


namespace cylinder_surface_area_l3173_317352

/-- The surface area of a cylinder with lateral surface net as a rectangle with sides 6π and 4π -/
theorem cylinder_surface_area : 
  ∀ (r h : ℝ), 
  (2 * π * r = 6 * π) → 
  (h = 4 * π) → 
  (2 * π * r * h + 2 * π * r^2 = 24 * π^2 + 18 * π) :=
by sorry

end cylinder_surface_area_l3173_317352


namespace boxes_left_l3173_317313

/-- The number of boxes Jerry started with -/
def initial_boxes : ℕ := 10

/-- The number of boxes Jerry sold -/
def sold_boxes : ℕ := 5

/-- Theorem: Jerry has 5 boxes left after selling -/
theorem boxes_left : initial_boxes - sold_boxes = 5 := by
  sorry

end boxes_left_l3173_317313


namespace polynomial_expansion_equality_l3173_317330

theorem polynomial_expansion_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  10 * p^9 * q = 120 * p^7 * q^3 → 
  p = Real.sqrt (12/13) := by
sorry

end polynomial_expansion_equality_l3173_317330


namespace kittens_given_to_friends_l3173_317390

/-- Given that Joan initially had 8 kittens and now has 6 kittens,
    prove that she gave 2 kittens to her friends. -/
theorem kittens_given_to_friends : 
  ∀ (initial current given : ℕ), 
    initial = 8 → 
    current = 6 → 
    given = initial - current → 
    given = 2 := by
  sorry

end kittens_given_to_friends_l3173_317390


namespace sequence_decomposition_l3173_317398

theorem sequence_decomposition (x : ℕ → ℝ) :
  ∃! (y z : ℕ → ℝ), (∀ n, x n = y n - z n) ∧
    (∀ n, y n ≥ 0) ∧
    (∀ n > 0, z n ≥ z (n - 1)) ∧
    (∀ n > 0, y n * (z n - z (n - 1)) = 0) ∧
    (z 0 = 0) := by
  sorry

end sequence_decomposition_l3173_317398


namespace rectangle_area_l3173_317348

theorem rectangle_area (breadth length perimeter area : ℝ) : 
  length = 3 * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 56 →
  area = length * breadth →
  area = 147 := by
sorry

end rectangle_area_l3173_317348


namespace tg_plus_ctg_values_l3173_317338

-- Define the trigonometric functions
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x
noncomputable def cosec (x : ℝ) : ℝ := 1 / Real.sin x
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem tg_plus_ctg_values (x : ℝ) :
  sec x - cosec x = 4 * Real.sqrt 3 →
  (tg x + ctg x = -6 ∨ tg x + ctg x = 8) :=
by sorry

end tg_plus_ctg_values_l3173_317338


namespace second_number_value_l3173_317301

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 110 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a → 
  b = 30 := by
sorry

end second_number_value_l3173_317301


namespace median_BD_correct_altitude_CE_correct_l3173_317369

/-- Triangle with vertices A(2,3), B(-1,0), and C(5,-1) -/
structure Triangle where
  A : ℝ × ℝ := (2, 3)
  B : ℝ × ℝ := (-1, 0)
  C : ℝ × ℝ := (5, -1)

/-- Line equation in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Median BD of the triangle -/
def median_BD (t : Triangle) : LineEquation :=
  { a := 2, b := -9, c := 2 }

/-- Altitude CE of the triangle -/
def altitude_CE (t : Triangle) : LineEquation :=
  { a := 1, b := 1, c := -4 }

/-- A point (x, y) lies on a line if it satisfies the line equation -/
def point_on_line (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem median_BD_correct (t : Triangle) : 
  point_on_line t.B (median_BD t) ∧ 
  point_on_line ((t.A.1 + t.C.1) / 2, (t.A.2 + t.C.2) / 2) (median_BD t) :=
sorry

theorem altitude_CE_correct (t : Triangle) : 
  point_on_line t.C (altitude_CE t) ∧ 
  (t.A.2 - t.B.2) * (t.C.1 - t.A.1) = (t.A.1 - t.B.1) * (t.C.2 - t.A.2) :=
sorry

end median_BD_correct_altitude_CE_correct_l3173_317369


namespace headphone_cost_l3173_317332

/-- The cost of the headphone set given Amanda's shopping scenario -/
theorem headphone_cost (initial_amount : ℕ) (cassette_cost : ℕ) (num_cassettes : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 50 →
  cassette_cost = 9 →
  num_cassettes = 2 →
  remaining_amount = 7 →
  initial_amount - (num_cassettes * cassette_cost) - remaining_amount = 25 := by
sorry

end headphone_cost_l3173_317332


namespace no_right_triangle_with_sides_13_17_k_l3173_317360

theorem no_right_triangle_with_sides_13_17_k : 
  ¬ ∃ (k : ℕ), k > 0 ∧ 
  ((13 * 13 + 17 * 17 = k * k) ∨ 
   (13 * 13 + k * k = 17 * 17) ∨ 
   (17 * 17 + k * k = 13 * 13)) := by
sorry

end no_right_triangle_with_sides_13_17_k_l3173_317360


namespace college_application_fee_cost_l3173_317359

/-- Proves that the cost of each college application fee is $25.00 -/
theorem college_application_fee_cost 
  (hourly_rate : ℝ) 
  (num_colleges : ℕ) 
  (hours_worked : ℕ) 
  (h1 : hourly_rate = 10)
  (h2 : num_colleges = 6)
  (h3 : hours_worked = 15) :
  (hourly_rate * hours_worked) / num_colleges = 25 := by
sorry

end college_application_fee_cost_l3173_317359


namespace number_of_spiders_l3173_317375

def total_legs : ℕ := 136
def num_ants : ℕ := 12
def spider_legs : ℕ := 8
def ant_legs : ℕ := 6

theorem number_of_spiders :
  ∃ (num_spiders : ℕ), 
    num_spiders * spider_legs + num_ants * ant_legs = total_legs ∧ 
    num_spiders = 8 :=
by sorry

end number_of_spiders_l3173_317375


namespace total_trucks_l3173_317397

/-- The number of trucks Namjoon and Taehyung have together -/
theorem total_trucks (namjoon_trucks taehyung_trucks : ℕ) 
  (h1 : namjoon_trucks = 3) 
  (h2 : taehyung_trucks = 2) : 
  namjoon_trucks + taehyung_trucks = 5 := by
  sorry

end total_trucks_l3173_317397


namespace monotonicity_of_f_range_of_b_l3173_317327

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x / (a * x^2 + b * x + 1)

theorem monotonicity_of_f :
  let f := f 1 1
  ∀ x₁ x₂, (x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂) → f x₁ < f x₂ ∧
           (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1) → f x₁ > f x₂ ∧
           (1 < x₁ ∧ x₁ < x₂) → f x₁ < f x₂ :=
sorry

theorem range_of_b :
  ∀ b : ℝ, (∀ x : ℝ, x ≥ 1 → f 0 b x ≥ 1) ↔ (0 ≤ b ∧ b ≤ Real.exp 1 - 1) :=
sorry

end monotonicity_of_f_range_of_b_l3173_317327


namespace charts_brought_by_associate_prof_l3173_317336

/-- Represents the number of charts brought by each associate professor -/
def charts_per_associate_prof : ℕ := sorry

/-- Represents the number of associate professors -/
def num_associate_profs : ℕ := sorry

/-- Represents the number of assistant professors -/
def num_assistant_profs : ℕ := sorry

theorem charts_brought_by_associate_prof :
  (2 * num_associate_profs + num_assistant_profs = 7) →
  (charts_per_associate_prof * num_associate_profs + 2 * num_assistant_profs = 11) →
  (num_associate_profs + num_assistant_profs = 6) →
  charts_per_associate_prof = 1 := by
    sorry

#check charts_brought_by_associate_prof

end charts_brought_by_associate_prof_l3173_317336


namespace no_integer_coordinate_equilateral_triangle_l3173_317311

theorem no_integer_coordinate_equilateral_triangle :
  ¬ ∃ (A B C : ℤ × ℤ), 
    (A.1 ≠ B.1 ∨ A.2 ≠ B.2) ∧ 
    (B.1 ≠ C.1 ∨ B.2 ≠ C.2) ∧ 
    (C.1 ≠ A.1 ∨ C.2 ≠ A.2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 := by
  sorry


end no_integer_coordinate_equilateral_triangle_l3173_317311


namespace unique_number_solution_l3173_317380

def is_valid_number (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a + b + c = 10 ∧
  b = a + c ∧
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99

theorem unique_number_solution :
  ∃! (a b c : ℕ), is_valid_number a b c ∧ 100 * a + 10 * b + c = 203 :=
sorry

end unique_number_solution_l3173_317380


namespace profit_maximized_at_twelve_point_five_l3173_317381

/-- The profit function for the bookstore -/
def P (p : ℝ) : ℝ := 150 * p - 6 * p^2 - 200

/-- The theorem stating that the profit is maximized at p = 12.5 -/
theorem profit_maximized_at_twelve_point_five :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧ 
  (∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → P p ≥ P q) ∧
  p = 12.5 := by
sorry

end profit_maximized_at_twelve_point_five_l3173_317381


namespace perpendicular_vectors_l3173_317347

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

theorem perpendicular_vectors (k : ℝ) : 
  let c := (a.1 + k * b.1, a.2 + k * b.2)
  (a.1 * c.1 + a.2 * c.2 = 0) → k = -10/3 := by
  sorry

end perpendicular_vectors_l3173_317347
