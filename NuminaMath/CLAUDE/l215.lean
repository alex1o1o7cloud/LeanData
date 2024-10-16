import Mathlib

namespace NUMINAMATH_CALUDE_triangle_trig_expression_l215_21562

theorem triangle_trig_expression (D E F : Real) (DE DF EF : Real) : 
  DE = 8 → DF = 10 → EF = 6 → 
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_expression_l215_21562


namespace NUMINAMATH_CALUDE_mod_nine_power_four_l215_21574

theorem mod_nine_power_four (m : ℕ) : 
  14^4 % 9 = m ∧ 0 ≤ m ∧ m < 9 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_nine_power_four_l215_21574


namespace NUMINAMATH_CALUDE_baker_weekday_hours_l215_21512

/-- Represents the baker's baking schedule and output --/
structure BakingSchedule where
  loavesPerHourPerOven : ℕ
  numOvens : ℕ
  weekendHoursPerDay : ℕ
  totalLoavesIn3Weeks : ℕ

/-- Calculates the number of hours the baker bakes from Monday to Friday each week --/
def weekdayHoursPerWeek (schedule : BakingSchedule) : ℕ :=
  let loavesPerHour := schedule.loavesPerHourPerOven * schedule.numOvens
  let weekendHours := schedule.weekendHoursPerDay * 2  -- 2 weekend days
  let weekendLoavesPerWeek := loavesPerHour * weekendHours
  let weekdayLoavesIn3Weeks := schedule.totalLoavesIn3Weeks - (weekendLoavesPerWeek * 3)
  weekdayLoavesIn3Weeks / (loavesPerHour * 3)

/-- Theorem stating that given the baker's schedule, they bake for 25 hours on weekdays --/
theorem baker_weekday_hours (schedule : BakingSchedule)
  (h1 : schedule.loavesPerHourPerOven = 5)
  (h2 : schedule.numOvens = 4)
  (h3 : schedule.weekendHoursPerDay = 2)
  (h4 : schedule.totalLoavesIn3Weeks = 1740) :
  weekdayHoursPerWeek schedule = 25 := by
  sorry


end NUMINAMATH_CALUDE_baker_weekday_hours_l215_21512


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_not_square_l215_21551

theorem product_of_five_consecutive_not_square (n : ℕ) : 
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_not_square_l215_21551


namespace NUMINAMATH_CALUDE_star_card_probability_l215_21576

theorem star_card_probability (total_cards : ℕ) (num_ranks : ℕ) (num_suits : ℕ) 
  (h1 : total_cards = 65)
  (h2 : num_ranks = 13)
  (h3 : num_suits = 5)
  (h4 : total_cards = num_ranks * num_suits) :
  (num_ranks : ℚ) / total_cards = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_star_card_probability_l215_21576


namespace NUMINAMATH_CALUDE_x_squared_coefficient_in_binomial_expansion_l215_21579

/-- Given a natural number n, returns the binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The exponent n for which the 5th term in (x-1/x)^n has the largest coefficient -/
def n : ℕ := 8

/-- The coefficient of x^2 in the expansion of (x-1/x)^n -/
def coefficient_x_squared : ℤ := -56

theorem x_squared_coefficient_in_binomial_expansion :
  coefficient_x_squared = (-1)^3 * binomial n 3 := by sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_in_binomial_expansion_l215_21579


namespace NUMINAMATH_CALUDE_sin_sum_specific_angles_l215_21580

theorem sin_sum_specific_angles (α β : Real) : 
  0 < α ∧ α < Real.pi → 
  0 < β ∧ β < Real.pi → 
  Real.cos α = -1/2 → 
  Real.sin β = Real.sqrt 3 / 2 → 
  Real.sin (α + β) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_specific_angles_l215_21580


namespace NUMINAMATH_CALUDE_cube_ratios_l215_21589

/-- Given two cubes with edge lengths 4 inches and 24 inches respectively,
    prove that the ratio of their volumes is 1/216 and
    the ratio of their surface areas is 1/36 -/
theorem cube_ratios (edge1 edge2 : ℝ) (h1 : edge1 = 4) (h2 : edge2 = 24) :
  (edge1^3 / edge2^3 = 1 / 216) ∧ ((6 * edge1^2) / (6 * edge2^2) = 1 / 36) := by
  sorry

end NUMINAMATH_CALUDE_cube_ratios_l215_21589


namespace NUMINAMATH_CALUDE_rain_period_end_time_l215_21561

/-- Represents time in 24-hour format -/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Adds hours to a given time -/
def addHours (t : Time) (h : ℕ) : Time :=
  { hour := (t.hour + h) % 24, minute := t.minute }

theorem rain_period_end_time 
  (start : Time)
  (rain_duration : ℕ)
  (no_rain_duration : ℕ)
  (h_start : start = { hour := 9, minute := 0 })
  (h_rain : rain_duration = 2)
  (h_no_rain : no_rain_duration = 6) :
  addHours start (rain_duration + no_rain_duration) = { hour := 17, minute := 0 } :=
sorry

end NUMINAMATH_CALUDE_rain_period_end_time_l215_21561


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l215_21531

theorem sum_remainder_mod_seven : 
  (102345 + 102346 + 102347 + 102348 + 102349 + 102350) % 7 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l215_21531


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l215_21506

/-- Represents a traffic light cycle with durations for each color -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycleDuration (c : TrafficLightCycle) : ℕ :=
  c.green + c.yellow + c.red

/-- Calculates the number of seconds where a color change can be observed in a 3-second interval -/
def changeObservationWindow (c : TrafficLightCycle) : ℕ :=
  3 * 3  -- 3 transitions, each with a 3-second window

/-- The probability of observing a color change in a random 3-second interval -/
def probabilityOfChange (c : TrafficLightCycle) : ℚ :=
  changeObservationWindow c / cycleDuration c

theorem traffic_light_change_probability :
  let c : TrafficLightCycle := ⟨50, 2, 40⟩
  probabilityOfChange c = 9 / 92 := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_change_probability_l215_21506


namespace NUMINAMATH_CALUDE_ordered_pairs_1806_l215_21502

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def count_ordered_pairs (n : ℕ) : ℕ := sorry

theorem ordered_pairs_1806 :
  prime_factorization 1806 = [(2, 1), (3, 2), (101, 1)] →
  count_ordered_pairs 1806 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_1806_l215_21502


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l215_21573

theorem mark_and_carolyn_money_sum : (5 : ℚ) / 8 + (7 : ℚ) / 20 = 0.975 := by
  sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l215_21573


namespace NUMINAMATH_CALUDE_danica_plane_arrangement_l215_21540

theorem danica_plane_arrangement : 
  (∃ n : ℕ, (17 + n) % 7 = 0 ∧ ∀ m : ℕ, m < n → (17 + m) % 7 ≠ 0) → 
  (∃ n : ℕ, (17 + n) % 7 = 0 ∧ ∀ m : ℕ, m < n → (17 + m) % 7 ≠ 0 ∧ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_danica_plane_arrangement_l215_21540


namespace NUMINAMATH_CALUDE_environmental_legislation_support_l215_21597

theorem environmental_legislation_support (men : ℕ) (women : ℕ) 
  (men_support_rate : ℚ) (women_support_rate : ℚ) :
  men = 200 →
  women = 1200 →
  men_support_rate = 70 / 100 →
  women_support_rate = 75 / 100 →
  let total_surveyed := men + women
  let total_supporters := men * men_support_rate + women * women_support_rate
  let overall_support_rate := total_supporters / total_surveyed
  ‖overall_support_rate - 74 / 100‖ < 1 / 100 :=
by sorry

end NUMINAMATH_CALUDE_environmental_legislation_support_l215_21597


namespace NUMINAMATH_CALUDE_range_of_m_l215_21526

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 11

def q (x m : ℝ) : Prop := 1 - 3*m ≤ x ∧ x ≤ 3 + m

theorem range_of_m (h : ∀ x m : ℝ, m > 0 → (¬(p x) → ¬(q x m)) ∧ ∃ x', ¬(q x' m) ∧ p x') :
  ∀ m : ℝ, m ∈ Set.Ici 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l215_21526


namespace NUMINAMATH_CALUDE_hypotenuse_length_hypotenuse_is_four_l215_21535

-- Define a right triangle with one angle of 15 degrees and altitude to hypotenuse of 1 cm
structure RightTriangle where
  -- One angle is 15 degrees (π/12 radians)
  angle : Real
  angle_eq : angle = Real.pi / 12
  -- The altitude to the hypotenuse is 1 cm
  altitude : Real
  altitude_eq : altitude = 1
  -- It's a right triangle (one angle is 90 degrees)
  is_right : Bool
  is_right_eq : is_right = true

-- Theorem: The hypotenuse of this triangle is 4 cm
theorem hypotenuse_length (t : RightTriangle) : Real :=
  4

-- The proof
theorem hypotenuse_is_four (t : RightTriangle) : hypotenuse_length t = 4 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_hypotenuse_is_four_l215_21535


namespace NUMINAMATH_CALUDE_rancher_cattle_movement_l215_21541

/-- A problem about a rancher moving cattle to higher ground. -/
theorem rancher_cattle_movement
  (total_cattle : ℕ)
  (truck_capacity : ℕ)
  (truck_speed : ℝ)
  (total_time : ℝ)
  (h1 : total_cattle = 400)
  (h2 : truck_capacity = 20)
  (h3 : truck_speed = 60)
  (h4 : total_time = 40)
  : (total_time * truck_speed) / (2 * (total_cattle / truck_capacity)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rancher_cattle_movement_l215_21541


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l215_21578

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 3 * 2 = 192 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l215_21578


namespace NUMINAMATH_CALUDE_bob_school_year_hours_l215_21529

/-- Calculates the hours per week Bob needs to work during the school year --/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_hours_needed := school_year_earnings / hourly_wage
  total_hours_needed / school_year_weeks

/-- Theorem stating that Bob needs to work 15 hours per week during the school year --/
theorem bob_school_year_hours : 
  school_year_hours_per_week 8 45 3600 24 3600 = 15 := by sorry

end NUMINAMATH_CALUDE_bob_school_year_hours_l215_21529


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l215_21591

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Slant height from apex to midpoint of a base side -/
  slant_height : ℝ
  /-- Radius of the hemisphere -/
  hemisphere_radius : ℝ

/-- Theorem stating the edge-length of the pyramid's base -/
theorem pyramid_base_edge_length (p : PyramidWithHemisphere) 
  (h1 : p.height = 8)
  (h2 : p.slant_height = 10)
  (h3 : p.hemisphere_radius = 3) :
  ∃ (edge_length : ℝ), edge_length = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l215_21591


namespace NUMINAMATH_CALUDE_double_reflection_result_l215_21516

/-- Reflect a point about the line y=x -/
def reflectYEqualsX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Reflect a point about the line y=-x -/
def reflectYEqualsNegX (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The final position after two reflections -/
def finalPosition (p : ℝ × ℝ) : ℝ × ℝ :=
  reflectYEqualsNegX (reflectYEqualsX p)

theorem double_reflection_result :
  finalPosition (3, -7) = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_result_l215_21516


namespace NUMINAMATH_CALUDE_apples_bought_proof_l215_21507

/-- The price of an orange in reals -/
def orange_price : ℝ := 2

/-- The price of an apple in reals -/
def apple_price : ℝ := 3

/-- An orange costs the same as half an apple plus half a real -/
axiom orange_price_relation : orange_price = apple_price / 2 + 1 / 2

/-- One-third of an apple costs the same as one-quarter of an orange plus half a real -/
axiom apple_price_relation : apple_price / 3 = orange_price / 4 + 1 / 2

/-- The number of apples that can be bought with the value of 5 oranges plus 5 reals -/
def apples_bought : ℕ := 5

theorem apples_bought_proof : 
  (5 * orange_price + 5) / apple_price = apples_bought := by sorry

end NUMINAMATH_CALUDE_apples_bought_proof_l215_21507


namespace NUMINAMATH_CALUDE_same_quotient_remainder_numbers_l215_21572

theorem same_quotient_remainder_numbers : 
  {a : ℕ | ∃ q : ℕ, 0 < q ∧ q < 6 ∧ a = 7 * q} = {7, 14, 21, 28, 35} := by
  sorry

end NUMINAMATH_CALUDE_same_quotient_remainder_numbers_l215_21572


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l215_21595

/-- The common ratio of an infinite geometric series with given first term and sum -/
theorem geometric_series_common_ratio 
  (a : ℝ) 
  (S : ℝ) 
  (h1 : a = 400) 
  (h2 : S = 2500) 
  (h3 : a > 0) 
  (h4 : S > a) : 
  ∃ (r : ℝ), r = 21 / 25 ∧ S = a / (1 - r) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l215_21595


namespace NUMINAMATH_CALUDE_parabola_equation_l215_21588

/-- Given a parabola C: y²=2px (p>0) with focus F, and a point A on C such that the midpoint of AF is (2,2), prove that the equation of C is y² = 8x. -/
theorem parabola_equation (p : ℝ) (F A : ℝ × ℝ) (h1 : p > 0) (h2 : F = (p/2, 0)) 
  (h3 : A.1^2 = 2*p*A.2) (h4 : ((F.1 + A.1)/2, (F.2 + A.2)/2) = (2, 2)) :
  ∀ (x y : ℝ), y^2 = 8*x ↔ y^2 = 2*p*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l215_21588


namespace NUMINAMATH_CALUDE_halloween_goodie_bags_l215_21537

/-- Calculates the minimum cost for buying a given number of items,
    where packs of 5 cost $3 and individual items cost $1 each. -/
def minCost (n : ℕ) : ℕ :=
  (n / 5) * 3 + (n % 5)

/-- The Halloween goodie bag problem -/
theorem halloween_goodie_bags :
  let vampireBags := 11
  let pumpkinBags := 14
  let totalBags := vampireBags + pumpkinBags
  totalBags = 25 →
  minCost vampireBags + minCost pumpkinBags = 17 := by
sorry

end NUMINAMATH_CALUDE_halloween_goodie_bags_l215_21537


namespace NUMINAMATH_CALUDE_remainder_theorem_l215_21523

theorem remainder_theorem :
  (7 * 10^20 + 3^20) % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l215_21523


namespace NUMINAMATH_CALUDE_cakes_per_person_l215_21546

theorem cakes_per_person (total_cakes : ℕ) (num_friends : ℕ) (h1 : total_cakes = 8) (h2 : num_friends = 4) :
  total_cakes / num_friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_cakes_per_person_l215_21546


namespace NUMINAMATH_CALUDE_fraction_simplification_l215_21527

theorem fraction_simplification : (150 : ℚ) / 4500 = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l215_21527


namespace NUMINAMATH_CALUDE_sum_of_odd_symmetric_function_l215_21544

-- Define an odd function with symmetry about x = 1/2
def is_odd_and_symmetric (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (1/2 + x) = f (1/2 - x))

-- Theorem statement
theorem sum_of_odd_symmetric_function (f : ℝ → ℝ) 
  (h : is_odd_and_symmetric f) : 
  f 1 + f 2 + f 3 + f 4 + f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_symmetric_function_l215_21544


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l215_21501

theorem divisibility_by_eleven (a b : ℤ) : 
  (11 ∣ a^2 + b^2) → (11 ∣ a) ∧ (11 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l215_21501


namespace NUMINAMATH_CALUDE_tangent_circle_center_height_l215_21504

/-- A parabola with equation y = 2x^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2 = 2 * p.1 ^ 2}

/-- A circle in the interior of the parabola -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  inside_parabola : center.2 < 2 * center.1 ^ 2
  tangent_points : Set (ℝ × ℝ)
  is_tangent : tangent_points ⊆ Parabola
  on_circle : ∀ p ∈ tangent_points, (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2
  symmetry : ∀ p ∈ tangent_points, (-p.1, p.2) ∈ tangent_points

theorem tangent_circle_center_height (c : TangentCircle) :
  ∃ p ∈ c.tangent_points, c.center.2 - p.2 = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_center_height_l215_21504


namespace NUMINAMATH_CALUDE_triangle_translation_l215_21517

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation2D) (p : Point2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem triangle_translation :
  let A : Point2D := { x := 0, y := 2 }
  let B : Point2D := { x := 2, y := -1 }
  let A' : Point2D := { x := -1, y := 0 }
  let t : Translation2D := { dx := A'.x - A.x, dy := A'.y - A.y }
  let B' : Point2D := applyTranslation t B
  B'.x = 1 ∧ B'.y = -3 := by sorry

end NUMINAMATH_CALUDE_triangle_translation_l215_21517


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l215_21519

-- Define the equation
def equation (θ : Real) : Prop :=
  Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin (θ * Real.pi / 180)

-- State the theorem
theorem least_positive_angle_theorem :
  ∃ (θ : Real), θ > 0 ∧ equation θ ∧ ∀ (φ : Real), φ > 0 ∧ equation φ → θ ≤ φ ∧ θ = 35 :=
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l215_21519


namespace NUMINAMATH_CALUDE_unique_a_value_l215_21577

open Real

theorem unique_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 π, f x = (cos (2 * x) + a) / sin x) →
  (∀ x ∈ Set.Ioo 0 π, |f x| ≤ 3) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_unique_a_value_l215_21577


namespace NUMINAMATH_CALUDE_triangle_neg_three_four_l215_21596

/-- The triangle operation -/
def triangle (a b : ℤ) : ℤ := a * b - a - b + 1

/-- Theorem stating that (-3) △ 4 = -12 -/
theorem triangle_neg_three_four : triangle (-3) 4 = -12 := by sorry

end NUMINAMATH_CALUDE_triangle_neg_three_four_l215_21596


namespace NUMINAMATH_CALUDE_invertible_product_l215_21545

def is_invertible (f : ℕ → Bool) : Prop := f 1 = false ∧ f 2 = true ∧ f 3 = true ∧ f 4 = true

theorem invertible_product (f : ℕ → Bool) (h : is_invertible f) :
  (List.filter (λ i => f i) [1, 2, 3, 4]).prod = 24 := by
  sorry

end NUMINAMATH_CALUDE_invertible_product_l215_21545


namespace NUMINAMATH_CALUDE_organize_blocks_time_l215_21554

/-- Calculates the time in minutes to organize blocks given the specified conditions -/
def timeToOrganizeBlocks (totalBlocks : ℕ) (addedPerCycle : ℕ) (removedPerCycle : ℕ) (cycleTimeSeconds : ℕ) : ℚ :=
  let netIncreasePerCycle := addedPerCycle - removedPerCycle
  let fullCycles := (totalBlocks - addedPerCycle) / netIncreasePerCycle
  let totalSeconds := (fullCycles + 1) * cycleTimeSeconds
  (totalSeconds : ℚ) / 60

/-- The time to organize 45 blocks is 16.5 minutes -/
theorem organize_blocks_time : 
  timeToOrganizeBlocks 45 5 3 45 = 33/2 := by sorry

end NUMINAMATH_CALUDE_organize_blocks_time_l215_21554


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_l215_21538

def S : Finset Int := {10, 30, -12, 15, -8}

theorem smallest_sum_of_three (s : Finset Int) (h : s = S) :
  (Finset.powersetCard 3 s).toList.map (fun t => t.toList.sum)
    |>.minimum?
    |>.map (fun x => x = -10)
    |>.getD False :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_l215_21538


namespace NUMINAMATH_CALUDE_parallel_unit_vector_l215_21505

/-- Given a vector a = (12, 5), prove that its parallel unit vector is (12/13, 5/13) or (-12/13, -5/13) -/
theorem parallel_unit_vector (a : ℝ × ℝ) (h : a = (12, 5)) :
  ∃ u : ℝ × ℝ, (u.1 * u.1 + u.2 * u.2 = 1) ∧ 
  (∃ k : ℝ, u.1 = k * a.1 ∧ u.2 = k * a.2) ∧
  (u = (12/13, 5/13) ∨ u = (-12/13, -5/13)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_unit_vector_l215_21505


namespace NUMINAMATH_CALUDE_sharon_wants_254_supplies_l215_21503

/-- The number of kitchen supplies Sharon wants to buy -/
def sharons_supplies (angela_pots : ℕ) : ℕ :=
  let angela_plates := 3 * angela_pots + 6
  let angela_cutlery := angela_plates / 2
  let sharon_pots := angela_pots / 2
  let sharon_plates := 3 * angela_plates - 20
  let sharon_cutlery := 2 * angela_cutlery
  sharon_pots + sharon_plates + sharon_cutlery

/-- Theorem stating that Sharon wants to buy 254 kitchen supplies -/
theorem sharon_wants_254_supplies : sharons_supplies 20 = 254 := by
  sorry

end NUMINAMATH_CALUDE_sharon_wants_254_supplies_l215_21503


namespace NUMINAMATH_CALUDE_cody_reading_time_l215_21565

def read_series (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let books_first_two_weeks := first_week + second_week
  let remaining_books := total_books - books_first_two_weeks
  let additional_weeks := (remaining_books + subsequent_weeks - 1) / subsequent_weeks
  2 + additional_weeks

theorem cody_reading_time :
  read_series 54 6 3 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cody_reading_time_l215_21565


namespace NUMINAMATH_CALUDE_min_k_value_l215_21530

theorem min_k_value (k : ℕ) : 
  (∃ x₀ : ℝ, x₀ > 2 ∧ k * (x₀ - 2) > x₀ * (Real.log x₀ + 1)) →
  k ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l215_21530


namespace NUMINAMATH_CALUDE_police_can_catch_gangster_police_can_reach_same_side_l215_21510

/-- Represents the setup of the police and gangster problem -/
structure PoliceGangsterSetup where
  a : ℝ  -- side length of the square
  police_speed : ℝ  -- speed of the police officer
  gangster_speed : ℝ  -- speed of the gangster
  h_positive_a : 0 < a  -- side length is positive
  h_positive_police_speed : 0 < police_speed  -- police speed is positive
  h_gangster_speed : gangster_speed = 2.9 * police_speed  -- gangster speed is 2.9 times police speed

/-- Theorem stating that the police officer can always reach a side of the square before the gangster moves more than one side length -/
theorem police_can_catch_gangster (setup : PoliceGangsterSetup) :
  setup.a / (2 * setup.police_speed) < 1.45 * setup.a := by
  sorry

/-- Corollary stating that the police officer can always end up on the same side as the gangster -/
theorem police_can_reach_same_side (setup : PoliceGangsterSetup) :
  ∃ (t : ℝ), t > 0 ∧ t * setup.police_speed ≥ setup.a / 2 ∧ t * setup.gangster_speed < setup.a := by
  sorry

end NUMINAMATH_CALUDE_police_can_catch_gangster_police_can_reach_same_side_l215_21510


namespace NUMINAMATH_CALUDE_sandy_initial_books_l215_21585

/-- The number of books Sandy had initially -/
def sandy_books : ℕ := 10

/-- The number of books Tim has -/
def tim_books : ℕ := 33

/-- The number of books Benny lost -/
def benny_lost : ℕ := 24

/-- The number of books Sandy and Tim have together after Benny lost some -/
def remaining_books : ℕ := 19

/-- Theorem stating that Sandy had 10 books initially -/
theorem sandy_initial_books : 
  sandy_books + tim_books = remaining_books + benny_lost := by sorry

end NUMINAMATH_CALUDE_sandy_initial_books_l215_21585


namespace NUMINAMATH_CALUDE_speed_time_distance_return_trip_time_l215_21539

/-- The distance to Yinping Mountain in kilometers -/
def distance : ℝ := 240

/-- The speed of the car in km/h -/
def speed (v : ℝ) : ℝ := v

/-- The time taken for the trip in hours -/
def time (t : ℝ) : ℝ := t

/-- The relationship between distance, speed, and time -/
theorem speed_time_distance (v t : ℝ) (h : t > 0) :
  speed v * time t = distance → v = distance / t :=
sorry

/-- The time taken for the return trip at 60 km/h -/
theorem return_trip_time :
  ∃ t : ℝ, t > 0 ∧ speed 60 * time t = distance ∧ t = 4 :=
sorry

end NUMINAMATH_CALUDE_speed_time_distance_return_trip_time_l215_21539


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l215_21524

/-- The market value of a machine after two years of depreciation -/
theorem machine_value_after_two_years
  (purchase_price : ℝ)
  (yearly_depreciation_rate : ℝ)
  (h1 : purchase_price = 8000)
  (h2 : yearly_depreciation_rate = 0.1) :
  purchase_price * (1 - yearly_depreciation_rate)^2 = 6480 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_after_two_years_l215_21524


namespace NUMINAMATH_CALUDE_cos_180_degrees_l215_21558

/-- Cosine of 180 degrees is -1 -/
theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l215_21558


namespace NUMINAMATH_CALUDE_book_store_inventory_l215_21550

theorem book_store_inventory (initial_books : ℝ) (first_addition : ℝ) (second_addition : ℝ) :
  initial_books = 41.0 →
  first_addition = 33.0 →
  second_addition = 2.0 →
  initial_books + first_addition + second_addition = 76.0 := by
  sorry

end NUMINAMATH_CALUDE_book_store_inventory_l215_21550


namespace NUMINAMATH_CALUDE_robins_bracelet_cost_l215_21555

/-- Represents a friend's name -/
inductive Friend
| jessica
| tori
| lily
| patrice

/-- Returns the number of letters in a friend's name -/
def nameLength (f : Friend) : Nat :=
  match f with
  | .jessica => 7
  | .tori => 4
  | .lily => 4
  | .patrice => 7

/-- The cost of a single bracelet in dollars -/
def braceletCost : Nat := 2

/-- The list of Robin's friends -/
def friendsList : List Friend := [Friend.jessica, Friend.tori, Friend.lily, Friend.patrice]

/-- Theorem: The total cost for Robin's bracelets is $44 -/
theorem robins_bracelet_cost : 
  (friendsList.map nameLength).sum * braceletCost = 44 := by
  sorry


end NUMINAMATH_CALUDE_robins_bracelet_cost_l215_21555


namespace NUMINAMATH_CALUDE_probability_of_drawing_math_books_l215_21563

/-- The number of Chinese books -/
def chinese_books : ℕ := 10

/-- The number of math books -/
def math_books : ℕ := 2

/-- The total number of books -/
def total_books : ℕ := chinese_books + math_books

/-- The number of books to be drawn -/
def books_drawn : ℕ := 2

theorem probability_of_drawing_math_books :
  (Nat.choose total_books books_drawn - Nat.choose chinese_books books_drawn) / Nat.choose total_books books_drawn = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_math_books_l215_21563


namespace NUMINAMATH_CALUDE_largest_n_factorial_sum_perfect_square_l215_21598

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem largest_n_factorial_sum_perfect_square :
  (∀ n : ℕ, n > 3 → ¬(is_perfect_square (sum_factorials n))) ∧
  (is_perfect_square (sum_factorials 3)) ∧
  (∀ n : ℕ, n > 0 → n < 3 → ¬(is_perfect_square (sum_factorials n))) :=
sorry

end NUMINAMATH_CALUDE_largest_n_factorial_sum_perfect_square_l215_21598


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l215_21548

/-- Given a quadratic polynomial P(x) = x^2 + ax + b, 
    if P(10) + P(30) = 40, then P(20) = -80 -/
theorem quadratic_polynomial_property (a b : ℝ) : 
  let P : ℝ → ℝ := λ x => x^2 + a*x + b
  (P 10 + P 30 = 40) → P 20 = -80 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l215_21548


namespace NUMINAMATH_CALUDE_xy_value_l215_21514

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x / 2 + 2 * y - 2 = Real.log x + Real.log y) : 
  x ^ y = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l215_21514


namespace NUMINAMATH_CALUDE_inequality_proof_l215_21582

theorem inequality_proof (y : ℝ) (h : y > 0) :
  2 * y ≥ 3 - 1 / y^2 ∧ (2 * y = 3 - 1 / y^2 ↔ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l215_21582


namespace NUMINAMATH_CALUDE_henrys_room_books_l215_21522

/-- The number of books Henry had in the room to donate -/
def books_in_room (initial_books : ℕ) (bookshelf_boxes : ℕ) (books_per_box : ℕ)
  (coffee_table_books : ℕ) (kitchen_books : ℕ) (free_books_taken : ℕ) (final_books : ℕ) : ℕ :=
  initial_books - (bookshelf_boxes * books_per_box + coffee_table_books + kitchen_books - free_books_taken)

/-- Theorem stating the number of books Henry had in the room to donate -/
theorem henrys_room_books :
  books_in_room 99 3 15 4 18 12 23 = 44 := by
  sorry

end NUMINAMATH_CALUDE_henrys_room_books_l215_21522


namespace NUMINAMATH_CALUDE_right_triangle_area_l215_21518

/-- 
Given a right triangle with hypotenuse 13 meters and one side 5 meters, 
prove that its area is 30 square meters.
-/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 5) 
  (h3 : a^2 = b^2 + c^2) : (1/2) * b * c = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l215_21518


namespace NUMINAMATH_CALUDE_statement_equivalence_l215_21536

theorem statement_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end NUMINAMATH_CALUDE_statement_equivalence_l215_21536


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l215_21566

theorem difference_of_squares_example : (17 + 10)^2 - (17 - 10)^2 = 680 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l215_21566


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l215_21525

-- Define the two fixed circles
def C1 (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 2
def C2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 2

-- Define a predicate for a point being on the trajectory
def OnTrajectory (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 14 = 1 ∨ x = 0

-- Theorem statement
theorem trajectory_of_moving_circle :
  ∀ (x y r : ℝ),
  (∃ (x1 y1 : ℝ), C1 x1 y1 ∧ (x - x1)^2 + (y - y1)^2 = r^2) →
  (∃ (x2 y2 : ℝ), C2 x2 y2 ∧ (x - x2)^2 + (y - y2)^2 = r^2) →
  OnTrajectory x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l215_21525


namespace NUMINAMATH_CALUDE_average_minutes_run_is_112_div_9_l215_21581

/-- The average number of minutes run per day by all students in an elementary school -/
def average_minutes_run (third_grade_minutes fourth_grade_minutes fifth_grade_minutes : ℕ)
  (third_to_fourth_ratio fourth_to_fifth_ratio : ℕ) : ℚ :=
  let fifth_graders := 1
  let fourth_graders := fourth_to_fifth_ratio * fifth_graders
  let third_graders := third_to_fourth_ratio * fourth_graders
  let total_students := third_graders + fourth_graders + fifth_graders
  let total_minutes := third_grade_minutes * third_graders + 
                       fourth_grade_minutes * fourth_graders + 
                       fifth_grade_minutes * fifth_graders
  (total_minutes : ℚ) / total_students

theorem average_minutes_run_is_112_div_9 :
  average_minutes_run 10 18 16 3 2 = 112 / 9 := by
  sorry

end NUMINAMATH_CALUDE_average_minutes_run_is_112_div_9_l215_21581


namespace NUMINAMATH_CALUDE_closest_to_10_l215_21599

def numbers : List ℝ := [9.998, 10.1, 10.09, 10.001]

def distance_to_10 (x : ℝ) : ℝ := |x - 10|

theorem closest_to_10 : 
  ∀ x ∈ numbers, distance_to_10 10.001 ≤ distance_to_10 x :=
by sorry

end NUMINAMATH_CALUDE_closest_to_10_l215_21599


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l215_21509

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem quadratic_symmetry (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  (m = 2 ∧
   (∀ x y, x < y → f m x < f m y) ∧
   (∀ x, x > 0 → f m x > f m 0) ∧
   (f m 0 = 2 ∧ ∀ x, f m x ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l215_21509


namespace NUMINAMATH_CALUDE_andy_socks_difference_l215_21564

theorem andy_socks_difference (black_socks : ℕ) (white_socks : ℕ) : 
  black_socks = 6 →
  white_socks = 4 * black_socks →
  (white_socks / 2) - black_socks = 6 := by
  sorry

end NUMINAMATH_CALUDE_andy_socks_difference_l215_21564


namespace NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l215_21557

-- Part 1
theorem calculate_expression : 
  (6 - 2 * Real.sqrt 3) * Real.sqrt 3 - Real.sqrt ((2 - Real.sqrt 2)^2) + 1 / Real.sqrt 2 = 
  6 * Real.sqrt 3 - 8 + 3 * Real.sqrt 2 / 2 := by sorry

-- Part 2
theorem solve_system_of_equations (x y : ℝ) :
  5 * x - y = -9 ∧ 3 * x + y = 1 → x = -1 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l215_21557


namespace NUMINAMATH_CALUDE_borrowing_rate_is_four_percent_l215_21593

/-- Calculates the interest rate at which money was borrowed given the following conditions:
  * The borrowed amount is 6000 Rs.
  * The loan duration is 2 years.
  * The money is immediately lent out at 6% per annum.
  * The gain from the transaction is 120 Rs per year.
-/
def calculate_borrowing_rate (borrowed_amount : ℝ) (loan_duration : ℕ) 
                             (lending_rate : ℝ) (gain_per_year : ℝ) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the borrowing rate is 4% -/
theorem borrowing_rate_is_four_percent :
  let borrowed_amount : ℝ := 6000
  let loan_duration : ℕ := 2
  let lending_rate : ℝ := 0.06
  let gain_per_year : ℝ := 120
  calculate_borrowing_rate borrowed_amount loan_duration lending_rate gain_per_year = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_borrowing_rate_is_four_percent_l215_21593


namespace NUMINAMATH_CALUDE_percentage_added_to_a_l215_21552

-- Define the ratio of a to b
def ratio_a_b : ℚ := 4 / 5

-- Define the percentage decrease for m
def decrease_percent : ℚ := 80

-- Define the ratio of m to x
def ratio_m_x : ℚ := 1 / 7

-- Define the function to calculate x given a and P
def x_from_a (a : ℚ) (P : ℚ) : ℚ := a * (1 + P / 100)

-- Define the function to calculate m given b
def m_from_b (b : ℚ) : ℚ := b * (1 - decrease_percent / 100)

-- Theorem statement
theorem percentage_added_to_a (a b : ℚ) (P : ℚ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = ratio_a_b) 
  (h4 : m_from_b b / x_from_a a P = ratio_m_x) : P = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_added_to_a_l215_21552


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l215_21549

theorem smallest_number_with_given_remainders : ∃! n : ℕ,
  (∀ m : ℕ, m < n → ¬(m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5)) ∧
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l215_21549


namespace NUMINAMATH_CALUDE_vertical_translation_equation_translated_line_equation_l215_21532

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateVertically (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + d }

theorem vertical_translation_equation (l : Line) (d : ℝ) :
  (translateVertically l d).slope = l.slope ∧
  (translateVertically l d).intercept = l.intercept + d := by
  sorry

/-- The original line y = -2x + 1 -/
def originalLine : Line :=
  { slope := -2, intercept := 1 }

/-- The translation distance -/
def translationDistance : ℝ := 2

theorem translated_line_equation :
  translateVertically originalLine translationDistance =
  { slope := -2, intercept := 3 } := by
  sorry

end NUMINAMATH_CALUDE_vertical_translation_equation_translated_line_equation_l215_21532


namespace NUMINAMATH_CALUDE_range_of_f_l215_21571

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x) + Real.arccos (Real.sin x)

theorem range_of_f :
  Set.range f = Set.Icc 0 Real.pi :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l215_21571


namespace NUMINAMATH_CALUDE_sector_perimeter_l215_21568

/-- Given a sector with area 2 and central angle 4 radians, its perimeter is 6. -/
theorem sector_perimeter (A : ℝ) (θ : ℝ) (r : ℝ) (P : ℝ) : 
  A = 2 → θ = 4 → A = (1/2) * r^2 * θ → P = r * θ + 2 * r → P = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l215_21568


namespace NUMINAMATH_CALUDE_vector_magnitude_l215_21533

/-- Given vectors a and b in ℝ², where a = (1,3) and (a + b) ⟂ (a - b), prove that |b| = √10 -/
theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0 →
  Real.sqrt ((b.1)^2 + (b.2)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l215_21533


namespace NUMINAMATH_CALUDE_john_plays_two_periods_l215_21583

def points_per_4_minutes : ℕ := 2 * 2 + 1 * 3
def minutes_per_period : ℕ := 12
def total_points : ℕ := 42

theorem john_plays_two_periods :
  (total_points / (points_per_4_minutes * (minutes_per_period / 4))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_plays_two_periods_l215_21583


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l215_21528

/-- The number of eggs in a full container -/
def full_container : ℕ := 15

/-- The number of containers with one missing egg -/
def partial_containers : ℕ := 3

/-- The minimum number of eggs specified in the problem -/
def min_eggs : ℕ := 150

/-- The number of eggs in the solution -/
def solution_eggs : ℕ := 162

/-- Theorem stating that the smallest number of eggs satisfying the conditions is 162 -/
theorem smallest_number_of_eggs :
  ∀ n : ℕ,
  (∃ c : ℕ, n = full_container * c - partial_containers) →
  n > min_eggs →
  n ≥ solution_eggs ∧
  (∀ m : ℕ, m < solution_eggs → 
    (∀ d : ℕ, m ≠ full_container * d - partial_containers) ∨ m ≤ min_eggs) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l215_21528


namespace NUMINAMATH_CALUDE_lecture_room_seating_l215_21556

theorem lecture_room_seating (m n : ℕ) : 
  (∃ boys_per_row girls_per_column unoccupied : ℕ,
    boys_per_row = 6 ∧ 
    girls_per_column = 8 ∧ 
    unoccupied = 15 ∧
    m * n = boys_per_row * m + girls_per_column * n + unoccupied) →
  (m - 8) * (n - 6) = 63 :=
by sorry

end NUMINAMATH_CALUDE_lecture_room_seating_l215_21556


namespace NUMINAMATH_CALUDE_horses_oats_meals_per_day_l215_21559

/-- The number of horses Peter has -/
def num_horses : ℕ := 4

/-- The amount of oats each horse eats per meal (in pounds) -/
def oats_per_meal : ℕ := 4

/-- The amount of grain each horse eats per day (in pounds) -/
def grain_per_day : ℕ := 3

/-- The total amount of food needed for all horses for 3 days (in pounds) -/
def total_food_3days : ℕ := 132

/-- The number of days food is needed for -/
def num_days : ℕ := 3

/-- The number of times horses eat oats per day -/
def oats_meals_per_day : ℕ := 2

theorem horses_oats_meals_per_day : 
  num_days * num_horses * (oats_per_meal * oats_meals_per_day + grain_per_day) = total_food_3days :=
by sorry

end NUMINAMATH_CALUDE_horses_oats_meals_per_day_l215_21559


namespace NUMINAMATH_CALUDE_magician_trick_l215_21508

def is_valid_selection (a d : ℕ) : Prop :=
  2 ≤ a ∧ a ≤ 16 ∧
  2 ≤ d ∧ d ≤ 16 ∧
  a % 2 = 0 ∧ d % 2 = 0 ∧
  a ≠ d ∧
  (d - a) % 16 = 3 ∨ (a - d) % 16 = 3

theorem magician_trick :
  ∃ (a d : ℕ), is_valid_selection a d ∧ a * d = 120 :=
sorry

end NUMINAMATH_CALUDE_magician_trick_l215_21508


namespace NUMINAMATH_CALUDE_carls_flowerbed_area_l215_21575

/-- Represents a rectangular flowerbed with fencing --/
structure Flowerbed where
  short_posts : ℕ  -- Number of posts on the shorter side (including corners)
  long_posts : ℕ   -- Number of posts on the longer side (including corners)
  post_spacing : ℕ -- Spacing between posts in yards

/-- Calculates the area of the flowerbed --/
def Flowerbed.area (fb : Flowerbed) : ℕ :=
  (fb.short_posts - 1) * (fb.long_posts - 1) * fb.post_spacing * fb.post_spacing

/-- Theorem stating the area of Carl's flowerbed --/
theorem carls_flowerbed_area :
  ∃ fb : Flowerbed,
    fb.short_posts + fb.long_posts = 13 ∧
    fb.long_posts = 3 * fb.short_posts - 2 ∧
    fb.post_spacing = 3 ∧
    fb.area = 144 := by
  sorry

end NUMINAMATH_CALUDE_carls_flowerbed_area_l215_21575


namespace NUMINAMATH_CALUDE_f_is_quadratic_l215_21590

/-- A function f : ℝ → ℝ is quadratic if there exist constants a, b, c : ℝ 
    such that f(x) = ax² + bx + c for all x : ℝ, and a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

/-- The function f(x) = x² + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l215_21590


namespace NUMINAMATH_CALUDE_budget_this_year_l215_21569

def cost_supply1 : ℕ := 13
def cost_supply2 : ℕ := 24
def remaining_last_year : ℕ := 6
def remaining_after_purchase : ℕ := 19

theorem budget_this_year :
  (cost_supply1 + cost_supply2 + remaining_after_purchase) - remaining_last_year = 50 := by
  sorry

end NUMINAMATH_CALUDE_budget_this_year_l215_21569


namespace NUMINAMATH_CALUDE_stock_increase_factor_l215_21567

def initial_investment : ℝ := 900
def num_stocks : ℕ := 3
def stock_c_loss_factor : ℝ := 0.5
def final_total_value : ℝ := 1350

theorem stock_increase_factor :
  let initial_per_stock := initial_investment / num_stocks
  let stock_c_final_value := initial_per_stock * stock_c_loss_factor
  let stock_ab_final_value := final_total_value - stock_c_final_value
  let stock_ab_initial_value := initial_per_stock * 2
  stock_ab_final_value / stock_ab_initial_value = 2 := by sorry

end NUMINAMATH_CALUDE_stock_increase_factor_l215_21567


namespace NUMINAMATH_CALUDE_existence_of_function_l215_21520

theorem existence_of_function (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, x + f y = a * (y + f x)) ↔ (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_function_l215_21520


namespace NUMINAMATH_CALUDE_square_sum_equals_thirty_l215_21570

theorem square_sum_equals_thirty (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b = 7) : 
  a^2 + b^2 = 30 := by sorry

end NUMINAMATH_CALUDE_square_sum_equals_thirty_l215_21570


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l215_21521

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism to the volume of the prism -/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * h) / (6 * r^2 * h) = π / 18 := by
  sorry


end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l215_21521


namespace NUMINAMATH_CALUDE_circle_center_sum_l215_21542

/-- Given a circle with equation x^2 + y^2 - 10x + 4y = -40, 
    the sum of the x and y coordinates of its center is 3. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 - 10*x + 4*y = -40 → x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l215_21542


namespace NUMINAMATH_CALUDE_abs_x_y_sum_l215_21584

theorem abs_x_y_sum (x y : ℝ) : 
  (|x| = 7 ∧ |y| = 9 ∧ |x + y| = -(x + y)) → (x - y = 16 ∨ x - y = -16) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_y_sum_l215_21584


namespace NUMINAMATH_CALUDE_cactus_path_problem_l215_21543

theorem cactus_path_problem (num_plants : ℕ) (camel_steps : ℕ) (kangaroo_jumps : ℕ) (total_distance : ℝ) :
  num_plants = 51 →
  camel_steps = 56 →
  kangaroo_jumps = 14 →
  total_distance = 7920 →
  let num_gaps := num_plants - 1
  let total_camel_steps := num_gaps * camel_steps
  let total_kangaroo_jumps := num_gaps * kangaroo_jumps
  let camel_step_length := total_distance / total_camel_steps
  let kangaroo_jump_length := total_distance / total_kangaroo_jumps
  kangaroo_jump_length - camel_step_length = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_cactus_path_problem_l215_21543


namespace NUMINAMATH_CALUDE_triangle_perpendicular_medians_l215_21594

/-- 
A triangle with sides a, b, and c, where the medians corresponding to sides a and c 
are perpendicular, satisfies the equation b² = (a² + c²) / 5.
-/
theorem triangle_perpendicular_medians (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_perpendicular : ∃ (x y : ℝ), x^2 + y^2 = (a/2)^2 ∧ x^2 + (y/2)^2 = (c/2)^2) :
  b^2 = (a^2 + c^2) / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_medians_l215_21594


namespace NUMINAMATH_CALUDE_subcommittee_count_l215_21513

def planning_committee_size : ℕ := 12
def teachers_in_committee : ℕ := 5
def subcommittee_size : ℕ := 5
def min_teachers_in_subcommittee : ℕ := 2

theorem subcommittee_count : 
  (Finset.sum (Finset.range (teachers_in_committee - min_teachers_in_subcommittee + 1))
    (fun k => Nat.choose teachers_in_committee (k + min_teachers_in_subcommittee) * 
              Nat.choose (planning_committee_size - teachers_in_committee) (subcommittee_size - (k + min_teachers_in_subcommittee)))) = 596 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l215_21513


namespace NUMINAMATH_CALUDE_triangle_side_length_l215_21515

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 120 * π / 180 →  -- Convert 120° to radians
  a = 2 * Real.sqrt 3 → 
  b = 2 → 
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) → 
  c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l215_21515


namespace NUMINAMATH_CALUDE_existence_of_linear_bound_l215_21560

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 2 * sin x) * (2^(-x) + 1)

theorem existence_of_linear_bound :
  ∃ (a b m : ℝ), ∀ x > 0, |f x - a * x - b| ≤ m :=
sorry

end NUMINAMATH_CALUDE_existence_of_linear_bound_l215_21560


namespace NUMINAMATH_CALUDE_negation_of_existence_l215_21511

theorem negation_of_existence (x : ℝ) : 
  ¬(∃ x ≥ 0, x^2 - 2*x - 3 = 0) ↔ ∀ x ≥ 0, x^2 - 2*x - 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l215_21511


namespace NUMINAMATH_CALUDE_matthias_balls_without_holes_l215_21553

/-- The number of balls without holes in Matthias' collection -/
def balls_without_holes (total_soccer : ℕ) (total_basketball : ℕ) (soccer_with_holes : ℕ) (basketball_with_holes : ℕ) : ℕ :=
  (total_soccer - soccer_with_holes) + (total_basketball - basketball_with_holes)

/-- Theorem stating the total number of balls without holes in Matthias' collection -/
theorem matthias_balls_without_holes :
  balls_without_holes 180 75 125 49 = 81 := by
  sorry

end NUMINAMATH_CALUDE_matthias_balls_without_holes_l215_21553


namespace NUMINAMATH_CALUDE_square_binomial_divided_by_negative_square_l215_21547

theorem square_binomial_divided_by_negative_square (m : ℝ) (hm : m ≠ 0) :
  (2 * m^2 - m)^2 / (-m^2) = -4 * m^2 + 4 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_divided_by_negative_square_l215_21547


namespace NUMINAMATH_CALUDE_circle_center_l215_21534

/-- The center of a circle with equation x^2 + 4x + y^2 - 6y = 12 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 4*x + y^2 - 6*y = 12) → (x + 2)^2 + (y - 3)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l215_21534


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l215_21587

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def tangent_to_circle (l : Line) (c : Circle) : Prop :=
  (abs (l.a * c.h + l.b * c.k + l.c) / Real.sqrt (l.a^2 + l.b^2)) = c.r

theorem tangent_lines_to_circle (given_line : Line) (c : Circle) :
  given_line = Line.mk 1 2 1 →
  c = Circle.mk 0 0 (Real.sqrt 5) →
  ∃ (l1 l2 : Line),
    parallel l1 given_line ∧
    parallel l2 given_line ∧
    tangent_to_circle l1 c ∧
    tangent_to_circle l2 c ∧
    ((l1 = Line.mk 1 2 5 ∧ l2 = Line.mk 1 2 (-5)) ∨
     (l1 = Line.mk 1 2 (-5) ∧ l2 = Line.mk 1 2 5)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l215_21587


namespace NUMINAMATH_CALUDE_natural_numbers_satisfying_conditions_l215_21500

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def sum_of_digits (n : ℕ) : ℕ := sorry

def num_positive_divisors (n : ℕ) : ℕ := sorry

def has_form_4k_plus_3 (p : ℕ) : Prop := ∃ k : ℕ, p = 4 * k + 3

def has_prime_divisor_with_4_or_more_digits (n : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ p ≥ 1000

theorem natural_numbers_satisfying_conditions (n : ℕ) : 
  (∀ m : ℕ, m > 1 → is_square m → ¬(m ∣ n)) ∧
  (∃! p : ℕ, is_prime p ∧ p ∣ n ∧ has_form_4k_plus_3 p) ∧
  (sum_of_digits n + 2 = num_positive_divisors n) ∧
  (is_square (n + 3)) ∧
  (¬has_prime_divisor_with_4_or_more_digits n) ↔
  (n = 222 ∨ n = 2022) := by sorry

end NUMINAMATH_CALUDE_natural_numbers_satisfying_conditions_l215_21500


namespace NUMINAMATH_CALUDE_football_player_goal_increase_l215_21592

/-- The increase in average goals score after the fifth match -/
def goalScoreIncrease (totalGoals : ℕ) (fifthMatchGoals : ℕ) : ℚ :=
  let firstFourAverage := (totalGoals - fifthMatchGoals : ℚ) / 4
  let newAverage := (totalGoals : ℚ) / 5
  newAverage - firstFourAverage

/-- Theorem stating the increase in average goals score -/
theorem football_player_goal_increase :
  goalScoreIncrease 4 2 = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_football_player_goal_increase_l215_21592


namespace NUMINAMATH_CALUDE_article_sale_profit_loss_l215_21586

theorem article_sale_profit_loss (cost_price selling_price_profit selling_price_25_percent : ℕ)
  (h1 : cost_price = 1400)
  (h2 : selling_price_profit = 1520)
  (h3 : selling_price_25_percent = 1750)
  (h4 : selling_price_25_percent = cost_price + cost_price / 4) :
  ∃ (selling_price_loss : ℕ),
    selling_price_loss = 1280 ∧
    (selling_price_profit - cost_price) / cost_price =
    (cost_price - selling_price_loss) / cost_price :=
by sorry

end NUMINAMATH_CALUDE_article_sale_profit_loss_l215_21586
