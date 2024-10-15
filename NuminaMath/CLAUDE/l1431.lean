import Mathlib

namespace NUMINAMATH_CALUDE_cos_seven_pi_fourths_l1431_143100

theorem cos_seven_pi_fourths : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_fourths_l1431_143100


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1431_143107

theorem triangle_perimeter : 
  ∀ (a b c : ℝ), 
    a = 10 ∧ b = 6 ∧ c = 7 → 
    a + b > c ∧ a + c > b ∧ b + c > a → 
    a + b + c = 23 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1431_143107


namespace NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l1431_143168

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l1431_143168


namespace NUMINAMATH_CALUDE_goldbach_conjecture_false_l1431_143124

/-- Goldbach's conjecture: Every even number greater than 2 can be expressed as the sum of two odd prime numbers -/
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ Odd p ∧ Odd q ∧ n = p + q

/-- Theorem stating that Goldbach's conjecture is false -/
theorem goldbach_conjecture_false : ¬goldbach_conjecture := by
  sorry

/-- Lemma: 4 is a counterexample to Goldbach's conjecture -/
lemma four_is_counterexample : 
  ¬(∃ p q : ℕ, Prime p ∧ Prime q ∧ Odd p ∧ Odd q ∧ 4 = p + q) := by
  sorry

end NUMINAMATH_CALUDE_goldbach_conjecture_false_l1431_143124


namespace NUMINAMATH_CALUDE_mike_lawn_money_l1431_143145

/-- The amount of money Mike made mowing lawns -/
def lawn_money : ℝ := sorry

/-- The amount of money Mike made weed eating -/
def weed_eating_money : ℝ := 26

/-- The number of weeks the money lasted -/
def weeks : ℕ := 8

/-- The amount Mike spent per week -/
def weekly_spending : ℝ := 5

theorem mike_lawn_money :
  lawn_money = 14 :=
by
  have total_spent : ℝ := weekly_spending * weeks
  have total_money : ℝ := lawn_money + weed_eating_money
  have h1 : total_money = total_spent := by sorry
  sorry

end NUMINAMATH_CALUDE_mike_lawn_money_l1431_143145


namespace NUMINAMATH_CALUDE_complex_number_theorem_l1431_143143

theorem complex_number_theorem (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^4 + z^4) = 56 := by
sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l1431_143143


namespace NUMINAMATH_CALUDE_dealer_pricing_theorem_l1431_143198

/-- A dealer's pricing strategy -/
structure DealerPricing where
  cash_discount : ℝ
  profit_percentage : ℝ
  articles_sold : ℕ
  articles_cost_price : ℕ

/-- Calculate the listing percentage above cost price -/
def listing_percentage (d : DealerPricing) : ℝ :=
  -- Define the calculation here
  sorry

/-- Theorem: Under specific conditions, the listing percentage is 60% -/
theorem dealer_pricing_theorem (d : DealerPricing) 
  (h1 : d.cash_discount = 0.15)
  (h2 : d.profit_percentage = 0.36)
  (h3 : d.articles_sold = 25)
  (h4 : d.articles_cost_price = 20) :
  listing_percentage d = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_dealer_pricing_theorem_l1431_143198


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l1431_143155

theorem probability_not_adjacent (n : ℕ) : 
  n = 5 → (36 : ℚ) / (120 : ℚ) = (3 : ℚ) / (10 : ℚ) := by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l1431_143155


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1431_143146

theorem expand_and_simplify (x : ℝ) : 2 * (x + 3) * (x + 8) = 2 * x^2 + 22 * x + 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1431_143146


namespace NUMINAMATH_CALUDE_additional_investment_rate_l1431_143180

theorem additional_investment_rate
  (initial_investment : ℝ)
  (initial_rate : ℝ)
  (additional_investment : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2800)
  (h2 : initial_rate = 0.05)
  (h3 : additional_investment = 1400)
  (h4 : total_rate = 0.06)
  : (initial_investment * initial_rate + additional_investment * (112 / 1400)) / (initial_investment + additional_investment) = total_rate :=
by sorry

end NUMINAMATH_CALUDE_additional_investment_rate_l1431_143180


namespace NUMINAMATH_CALUDE_find_other_number_l1431_143164

theorem find_other_number (x y : ℤ) : 
  (3 * x + 4 * y = 151) → 
  ((x = 19 ∨ y = 19) → (x = 25 ∨ y = 25)) := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l1431_143164


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l1431_143193

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (7, 0)
def circle2_radius : ℝ := 1

-- Define the tangent line (implicitly)
def tangent_line : Set (ℝ × ℝ) := sorry

-- Condition that the tangent line touches both circles in the first quadrant
axiom tangent_touches_circles :
  ∃ (p q : ℝ × ℝ),
    p.1 > 0 ∧ p.2 > 0 ∧
    q.1 > 0 ∧ q.2 > 0 ∧
    p ∈ tangent_line ∧
    q ∈ tangent_line ∧
    (p.1 - circle1_center.1)^2 + (p.2 - circle1_center.2)^2 = circle1_radius^2 ∧
    (q.1 - circle2_center.1)^2 + (q.2 - circle2_center.2)^2 = circle2_radius^2

-- Theorem statement
theorem tangent_y_intercept :
  ∃ (y : ℝ), y = 9 ∧ (0, y) ∈ tangent_line :=
sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l1431_143193


namespace NUMINAMATH_CALUDE_min_abs_z_l1431_143108

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*Complex.I) + Complex.abs (z - 7) = 10) :
  ∃ (w : ℂ), Complex.abs (z - 5*Complex.I) + Complex.abs (z - 7) = 10 → Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 35 / Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_l1431_143108


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1431_143109

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x > 5 ∧ x > a) ↔ x > 5) → a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1431_143109


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l1431_143159

/-- Circle A with equation x^2 + y^2 = 1 -/
def circle_A : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- Circle B with equation (x-3)^2 + (y+4)^2 = 10 -/
def circle_B : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 + 4)^2 = 10}

/-- The point P satisfies the condition that its distances to the tangent points on circles A and B are equal -/
def point_P : Set (ℝ × ℝ) :=
  {p | ∃ d e : ℝ × ℝ, d ∈ circle_A ∧ e ∈ circle_B ∧ 
       (p.1 - d.1)^2 + (p.2 - d.2)^2 = (p.1 - e.1)^2 + (p.2 - e.2)^2}

/-- The minimum distance from point P to the origin is 8/5 -/
theorem min_distance_to_origin : 
  ∀ p ∈ point_P, (∀ q ∈ point_P, p.1^2 + p.2^2 ≤ q.1^2 + q.2^2) → 
  p.1^2 + p.2^2 = (8/5)^2 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l1431_143159


namespace NUMINAMATH_CALUDE_root_conditions_imply_a_range_l1431_143186

theorem root_conditions_imply_a_range (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ > 1 ∧ x₂ < 1 ∧ 
   x₁^2 + a*x₁ + a^2 - a - 2 = 0 ∧
   x₂^2 + a*x₂ + a^2 - a - 2 = 0) →
  -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_a_range_l1431_143186


namespace NUMINAMATH_CALUDE_age_difference_brother_cousin_l1431_143148

/-- Proves that the age difference between Lexie's brother and cousin is 5 years -/
theorem age_difference_brother_cousin : 
  ∀ (lexie_age brother_age sister_age uncle_age grandma_age cousin_age : ℕ),
  lexie_age = 8 →
  grandma_age = 68 →
  lexie_age = brother_age + 6 →
  sister_age = 2 * lexie_age →
  uncle_age + 12 = grandma_age →
  cousin_age = brother_age + 5 →
  cousin_age - brother_age = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_brother_cousin_l1431_143148


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l1431_143165

/-- Represents a rectangular shape with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of tiles that can fit in one dimension -/
def fitInDimension (floorSize tileSize : ℕ) : ℕ :=
  floorSize / tileSize

/-- Calculates the number of tiles that can fit on the floor for a given orientation -/
def tilesForOrientation (floor tile : Rectangle) : ℕ :=
  (fitInDimension floor.width tile.width) * (fitInDimension floor.height tile.height)

/-- Theorem: The maximum number of 50x40 tiles on a 120x150 floor is 9 -/
theorem max_tiles_on_floor :
  let floor : Rectangle := ⟨120, 150⟩
  let tile : Rectangle := ⟨50, 40⟩
  let orientation1 := tilesForOrientation floor tile
  let orientation2 := tilesForOrientation floor ⟨tile.height, tile.width⟩
  max orientation1 orientation2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l1431_143165


namespace NUMINAMATH_CALUDE_tangent_line_proof_l1431_143110

def circle_center : ℝ × ℝ := (6, 3)
def circle_radius : ℝ := 2
def point_p : ℝ × ℝ := (10, 0)

def is_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

def is_on_line (p : ℝ × ℝ) : Prop :=
  4 * p.1 - 3 * p.2 = 19

theorem tangent_line_proof :
  ∃ (q : ℝ × ℝ),
    is_on_circle q ∧
    is_on_line q ∧
    is_on_line point_p ∧
    ∀ (r : ℝ × ℝ), is_on_circle r ∧ is_on_line r → r = q :=
  sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l1431_143110


namespace NUMINAMATH_CALUDE_square_field_area_l1431_143183

/-- Given a square field with two 1-meter wide gates, where the cost of drawing barbed wire
    is 1.10 per meter and the total cost is 732.6, prove that the area of the field is 27889 sq m. -/
theorem square_field_area (side : ℝ) (gate_width : ℝ) (wire_cost_per_meter : ℝ) (total_cost : ℝ)
  (h1 : gate_width = 1)
  (h2 : wire_cost_per_meter = 1.1)
  (h3 : total_cost = 732.6)
  (h4 : wire_cost_per_meter * (4 * side - 2 * gate_width) = total_cost) :
  side^2 = 27889 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l1431_143183


namespace NUMINAMATH_CALUDE_triangle_cos_2C_l1431_143194

theorem triangle_cos_2C (a b : ℝ) (S : ℝ) (C : ℝ) :
  a = 8 →
  b = 5 →
  S = 12 →
  S = (1/2) * a * b * Real.sin C →
  Real.cos (2 * C) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_triangle_cos_2C_l1431_143194


namespace NUMINAMATH_CALUDE_spontaneous_reaction_l1431_143106

theorem spontaneous_reaction (ΔH ΔS : ℝ) (h1 : ΔH = -98.2) (h2 : ΔS = 70.5 / 1000) :
  ∀ T : ℝ, T ≥ 0 → ΔH - T * ΔS < 0 := by
sorry

end NUMINAMATH_CALUDE_spontaneous_reaction_l1431_143106


namespace NUMINAMATH_CALUDE_equation_solution_l1431_143113

theorem equation_solution : ∃ x : ℤ, 121 * x = 75625 ∧ x = 625 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1431_143113


namespace NUMINAMATH_CALUDE_charity_fundraising_l1431_143134

theorem charity_fundraising (donation_percentage : ℚ) (num_organizations : ℕ) (amount_per_org : ℚ) :
  donation_percentage = 80 / 100 →
  num_organizations = 8 →
  amount_per_org = 250 →
  (num_organizations : ℚ) * amount_per_org / donation_percentage = 2500 :=
by sorry

end NUMINAMATH_CALUDE_charity_fundraising_l1431_143134


namespace NUMINAMATH_CALUDE_mr_langsley_arrival_time_l1431_143188

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define addition operation for Time
def Time.add (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hour * 60 + t1.minute + t2.hour * 60 + t2.minute
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

-- Define the problem parameters
def pickup_time : Time := { hour := 6, minute := 0 }
def time_to_first_station : Time := { hour := 0, minute := 40 }
def time_from_first_station_to_work : Time := { hour := 2, minute := 20 }

-- Theorem to prove
theorem mr_langsley_arrival_time :
  (pickup_time.add time_to_first_station).add time_from_first_station_to_work = { hour := 9, minute := 0 } := by
  sorry


end NUMINAMATH_CALUDE_mr_langsley_arrival_time_l1431_143188


namespace NUMINAMATH_CALUDE_number_of_workers_l1431_143132

/-- Proves that the number of men working on the jobs is 3 --/
theorem number_of_workers (time_per_job : ℝ) (num_jobs : ℕ) (hourly_rate : ℝ) (total_earned : ℝ) : ℕ :=
  by
  -- Assume the given conditions
  have h1 : time_per_job = 1 := by sorry
  have h2 : num_jobs = 5 := by sorry
  have h3 : hourly_rate = 10 := by sorry
  have h4 : total_earned = 150 := by sorry

  -- Define the number of workers
  let num_workers : ℕ := 3

  -- Prove that num_workers satisfies the conditions
  have h5 : (↑num_workers : ℝ) * num_jobs * hourly_rate = total_earned := by sorry

  -- Return the number of workers
  exact num_workers

end NUMINAMATH_CALUDE_number_of_workers_l1431_143132


namespace NUMINAMATH_CALUDE_probability_of_two_pairs_and_one_different_l1431_143115

-- Define the number of sides on each die
def numSides : ℕ := 10

-- Define the number of dice rolled
def numDice : ℕ := 5

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := numSides ^ numDice

-- Define the number of ways to choose 2 distinct numbers for pairs
def waysToChoosePairs : ℕ := Nat.choose numSides 2

-- Define the number of choices for the fifth die
def choicesForFifthDie : ℕ := numSides - 2

-- Define the number of ways to arrange the digits
def arrangements : ℕ := Nat.factorial numDice / (2 * 2 * Nat.factorial 1)

-- Define the number of successful outcomes
def successfulOutcomes : ℕ := waysToChoosePairs * choicesForFifthDie * arrangements

-- The theorem to prove
theorem probability_of_two_pairs_and_one_different : 
  (successfulOutcomes : ℚ) / totalOutcomes = 108 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_two_pairs_and_one_different_l1431_143115


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1431_143130

theorem quadratic_equation_roots (k : ℝ) (θ : ℝ) : 
  (∃ x y : ℝ, x = Real.sin θ ∧ y = Real.cos θ ∧ 
    8 * x^2 + 6 * k * x + 2 * k + 1 = 0 ∧
    8 * y^2 + 6 * k * y + 2 * k + 1 = 0) →
  k = -10/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1431_143130


namespace NUMINAMATH_CALUDE_point_upper_left_region_range_l1431_143140

theorem point_upper_left_region_range (t : ℝ) : 
  (2 : ℝ) - 2 * t + 4 ≤ 0 → t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_point_upper_left_region_range_l1431_143140


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1431_143112

theorem min_value_on_circle (a b : ℝ) (h : a^2 + b^2 - 4*a + 3 = 0) :
  2 ≤ Real.sqrt (a^2 + b^2) + 1 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 - 4*a₀ + 3 = 0 ∧ Real.sqrt (a₀^2 + b₀^2) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1431_143112


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l1431_143101

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 3) / (x - 2) - 1 = 3 / x
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation 1
theorem equation1_solution :
  (∃! x : ℝ, equation1 x) ∧ equation1 (3/2) :=
sorry

-- Theorem for equation 2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x :=
sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l1431_143101


namespace NUMINAMATH_CALUDE_store_rooms_problem_l1431_143128

theorem store_rooms_problem (x : ℕ) : 
  (∃ (total_guests : ℕ), 
    total_guests = 7 * x + 7 ∧ 
    total_guests = 9 * (x - 1)) → 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_store_rooms_problem_l1431_143128


namespace NUMINAMATH_CALUDE_jessica_watermelons_l1431_143191

/-- The number of watermelons Jessica has left -/
def watermelons_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Proof that Jessica has 8 watermelons left -/
theorem jessica_watermelons : watermelons_left 35 27 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_watermelons_l1431_143191


namespace NUMINAMATH_CALUDE_bright_numbers_l1431_143121

def isBright (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a^2 + b^3

theorem bright_numbers (r s : ℕ+) :
  (∃ f : ℕ → ℕ, StrictMono f ∧ ∀ i, isBright (r + f i) ∧ isBright (s + f i)) ∧
  (∃ g : ℕ → ℕ, StrictMono g ∧ ∀ i, isBright (r * g i) ∧ isBright (s * g i)) := by
  sorry

end NUMINAMATH_CALUDE_bright_numbers_l1431_143121


namespace NUMINAMATH_CALUDE_point_transformation_theorem_l1431_143185

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectAboutYeqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation_theorem (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 3 2
  let (x₂, y₂) := reflectAboutYeqX x₁ y₁
  (x₂ = 1 ∧ y₂ = -4) → d - c = -9 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_theorem_l1431_143185


namespace NUMINAMATH_CALUDE_number_line_segment_sum_l1431_143147

theorem number_line_segment_sum : 
  ∀ (P V : ℝ) (Q R S T U : ℝ),
  P = 3 →
  V = 33 →
  Q - P = R - Q → R - Q = S - R → S - R = T - S → T - S = U - T → U - T = V - U →
  (S - P) + (V - T) = 25 := by
sorry

end NUMINAMATH_CALUDE_number_line_segment_sum_l1431_143147


namespace NUMINAMATH_CALUDE_bill_vote_change_l1431_143184

theorem bill_vote_change (total_voters : ℕ) (first_for first_against : ℕ) 
  (second_for second_against : ℕ) : 
  total_voters = 400 →
  first_for + first_against = total_voters →
  first_against > first_for →
  second_for + second_against = total_voters →
  second_for > second_against →
  (second_for - second_against) = 2 * (first_against - first_for) →
  second_for = (12 * first_against) / 11 →
  second_for - first_for = 60 := by
sorry

end NUMINAMATH_CALUDE_bill_vote_change_l1431_143184


namespace NUMINAMATH_CALUDE_distinct_cubes_count_l1431_143178

/-- The number of rotational symmetries of a cube -/
def cube_rotational_symmetries : ℕ := 24

/-- The number of unit cubes used to form the 2x2x2 cube -/
def num_unit_cubes : ℕ := 8

/-- The number of distinct 2x2x2 cubes that can be formed -/
def distinct_cubes : ℕ := Nat.factorial num_unit_cubes / cube_rotational_symmetries

theorem distinct_cubes_count :
  distinct_cubes = 1680 := by sorry

end NUMINAMATH_CALUDE_distinct_cubes_count_l1431_143178


namespace NUMINAMATH_CALUDE_min_m_value_l1431_143172

-- Define the points A and B
def A (m : ℝ) : ℝ × ℝ := (1, m)
def B (x : ℝ) : ℝ × ℝ := (-1, 1 - |x|)

-- Define symmetry with respect to the origin
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- State the theorem
theorem min_m_value (m x : ℝ) 
  (h : symmetric_wrt_origin (A m) (B x)) : 
  ∀ k, m ≤ k → -1 ≤ k :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l1431_143172


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1431_143125

/-- The equation of an ellipse in terms of m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1

/-- The range of m for which the equation represents an ellipse -/
def m_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m → m_range m :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1431_143125


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_60_l1431_143136

/-- The cost of paint per kilogram, given the coverage and the cost to paint a cube. -/
theorem paint_cost_per_kg (coverage : Real) (cube_side : Real) (total_cost : Real) : Real :=
  let surface_area := 6 * cube_side * cube_side
  let paint_needed := surface_area / coverage
  total_cost / paint_needed

/-- Proof that the cost of paint per kilogram is $60. -/
theorem paint_cost_is_60 :
  paint_cost_per_kg 20 10 1800 = 60 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_60_l1431_143136


namespace NUMINAMATH_CALUDE_cubic_inequality_l1431_143189

theorem cubic_inequality (x : ℝ) (h : x ≥ 1000000) :
  x^3 + x + 1 ≤ x^4 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1431_143189


namespace NUMINAMATH_CALUDE_parabola_intersection_l1431_143116

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 9 * x - 15
  let g (x : ℝ) := x^2 - 5 * x + 7
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ 
    (x = 1 + 2 * Real.sqrt 3 ∧ y = 19 - 6 * Real.sqrt 3) ∨
    (x = 1 - 2 * Real.sqrt 3 ∧ y = 19 + 6 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1431_143116


namespace NUMINAMATH_CALUDE_one_third_of_one_fourth_l1431_143102

theorem one_third_of_one_fourth (n : ℝ) : (3 / 10 : ℝ) * n = 64.8 → (1 / 3 : ℝ) * (1 / 4 : ℝ) * n = 18 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_one_fourth_l1431_143102


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1431_143195

theorem sufficient_not_necessary (x : ℝ) : 
  (x = 1 → x^2 - 3*x + 2 = 0) ∧ 
  (∃ y : ℝ, y ≠ 1 ∧ y^2 - 3*y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1431_143195


namespace NUMINAMATH_CALUDE_inequality_proof_l1431_143176

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (Real.arctan ((a * d - b * c) / (a * c + b * d)))^2 ≥ 2 * (1 - (a * c + b * d) / Real.sqrt ((a^2 + b^2) * (c^2 + d^2))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1431_143176


namespace NUMINAMATH_CALUDE_intersection_point_a_value_l1431_143179

-- Define the three lines
def line1 (a x y : ℝ) : Prop := a * x + 2 * y + 8 = 0
def line2 (x y : ℝ) : Prop := 4 * x + 3 * y = 10
def line3 (x y : ℝ) : Prop := 2 * x - y = 10

-- Theorem statement
theorem intersection_point_a_value :
  ∃! (a : ℝ), ∃! (x y : ℝ), line1 a x y ∧ line2 x y ∧ line3 x y → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_a_value_l1431_143179


namespace NUMINAMATH_CALUDE_tissue_with_mitotic_and_meiotic_cells_is_gonad_l1431_143133

structure Cell where
  chromosomeCount : ℕ

structure Tissue where
  cells : Set Cell

def isSomaticCell (c : Cell) : Prop := sorry

def isGermCell (c : Cell) (sc : Cell) : Prop :=
  isSomaticCell sc ∧ c.chromosomeCount = sc.chromosomeCount / 2

def containsMitoticCells (t : Tissue) : Prop :=
  ∃ c ∈ t.cells, isSomaticCell c

def containsMeioticCells (t : Tissue) : Prop :=
  ∃ c sc, c ∈ t.cells ∧ isGermCell c sc

def isGonad (t : Tissue) : Prop :=
  containsMitoticCells t ∧ containsMeioticCells t

theorem tissue_with_mitotic_and_meiotic_cells_is_gonad (t : Tissue) :
  containsMitoticCells t → containsMeioticCells t → isGonad t :=
by sorry

end NUMINAMATH_CALUDE_tissue_with_mitotic_and_meiotic_cells_is_gonad_l1431_143133


namespace NUMINAMATH_CALUDE_x_value_proof_l1431_143175

theorem x_value_proof (x : ℝ) (h : 9 / x^2 = x / 25) : x = (225 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1431_143175


namespace NUMINAMATH_CALUDE_snakes_in_pond_l1431_143120

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 56

/-- The number of eyes each alligator has -/
def eyes_per_alligator : ℕ := 2

/-- The number of eyes each snake has -/
def eyes_per_snake : ℕ := 2

/-- The number of snakes in the pond -/
def num_snakes : ℕ := (total_eyes - num_alligators * eyes_per_alligator) / eyes_per_snake

theorem snakes_in_pond : num_snakes = 18 := by
  sorry

end NUMINAMATH_CALUDE_snakes_in_pond_l1431_143120


namespace NUMINAMATH_CALUDE_odd_function_and_monotone_increasing_l1431_143160

/-- An odd function f(x) = x^2 + mx -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- f is an odd function -/
def is_odd (m : ℝ) : Prop := ∀ x, f m (-x) = -(f m x)

/-- f is monotonically increasing on an interval -/
def is_monotone_increasing (m : ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f m x < f m y

theorem odd_function_and_monotone_increasing :
  ∃ m, is_odd m ∧ 
  ∃ a, 1 < a ∧ a ≤ 3 ∧ 
  is_monotone_increasing m (-1) (a - 2) ∧
  m = 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_and_monotone_increasing_l1431_143160


namespace NUMINAMATH_CALUDE_math_club_pair_sequences_l1431_143126

/-- The number of students in the Math Club -/
def num_students : ℕ := 12

/-- The number of sessions per week -/
def sessions_per_week : ℕ := 3

/-- The number of students selected per session -/
def students_per_session : ℕ := 2

/-- The number of different pair sequences that can be selected in one week -/
def pair_sequences_per_week : ℕ := (num_students * (num_students - 1)) ^ sessions_per_week

theorem math_club_pair_sequences :
  pair_sequences_per_week = 2299968 :=
sorry

end NUMINAMATH_CALUDE_math_club_pair_sequences_l1431_143126


namespace NUMINAMATH_CALUDE_base7_sum_equality_l1431_143187

-- Define a type for base 7 digits
def Base7Digit := { n : Nat // n > 0 ∧ n < 7 }

-- Function to convert a three-digit base 7 number to natural number
def base7ToNat (a b c : Base7Digit) : Nat :=
  49 * a.val + 7 * b.val + c.val

-- Statement of the theorem
theorem base7_sum_equality 
  (A B C : Base7Digit) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (hSum : base7ToNat A B C + base7ToNat B C A + base7ToNat C A B = 
          343 * A.val + 49 * A.val + 7 * A.val) : 
  B.val + C.val = 6 := by
sorry

end NUMINAMATH_CALUDE_base7_sum_equality_l1431_143187


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l1431_143118

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 4 5) 6) 8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l1431_143118


namespace NUMINAMATH_CALUDE_catch_up_solution_l1431_143190

/-- Represents the problem of a car catching up to a truck -/
def CatchUpProblem (truckSpeed carInitialSpeed carSpeedIncrease distance : ℝ) : Prop :=
  ∃ (t : ℝ),
    t > 0 ∧
    (carInitialSpeed * t + carSpeedIncrease * t * (t - 1) / 2) = (truckSpeed * t + distance)

/-- The solution to the catch-up problem -/
theorem catch_up_solution :
  CatchUpProblem 40 50 5 135 →
  ∃ (t : ℝ), t = 6 ∧ CatchUpProblem 40 50 5 135 := by sorry

#check catch_up_solution

end NUMINAMATH_CALUDE_catch_up_solution_l1431_143190


namespace NUMINAMATH_CALUDE_five_pairs_l1431_143163

/-- The number of pairs of natural numbers (a, b) satisfying the given conditions -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    a ≥ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 5 pairs of natural numbers satisfying the conditions -/
theorem five_pairs : count_pairs = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_pairs_l1431_143163


namespace NUMINAMATH_CALUDE_ball_max_height_l1431_143154

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem ball_max_height :
  ∃ (max : ℝ), max = 161 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l1431_143154


namespace NUMINAMATH_CALUDE_hockey_league_games_l1431_143192

theorem hockey_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 16) (h2 : total_games = 1200) :
  ∃ x : ℕ, x * n * (n - 1) / 2 = total_games ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1431_143192


namespace NUMINAMATH_CALUDE_outdoor_temp_correction_l1431_143177

/-- Represents a thermometer with a linear error --/
structure Thermometer where
  /-- The slope of the linear relationship between actual and measured temperature --/
  k : ℝ
  /-- The y-intercept of the linear relationship between actual and measured temperature --/
  b : ℝ

/-- Calculates the actual temperature given a thermometer reading --/
def actualTemp (t : Thermometer) (reading : ℝ) : ℝ :=
  t.k * reading + t.b

theorem outdoor_temp_correction (t : Thermometer) 
  (h1 : actualTemp t (-11) = -7)
  (h2 : actualTemp t 32 = 36)
  (h3 : t.k = 1) -- This comes from solving the system of equations in the solution
  (h4 : t.b = -4) -- This comes from solving the system of equations in the solution
  : actualTemp t 22 = 18 := by
  sorry

end NUMINAMATH_CALUDE_outdoor_temp_correction_l1431_143177


namespace NUMINAMATH_CALUDE_six_digit_scrambled_divisibility_l1431_143137

theorem six_digit_scrambled_divisibility (a b c : Nat) 
  (ha : a ∈ Finset.range 10) 
  (hb : b ∈ Finset.range 10) 
  (hc : c ∈ Finset.range 10) 
  (hpos : 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * c + b > 0) :
  let Z := 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * c + b
  ∃ k : Nat, Z = 101 * k := by
  sorry

end NUMINAMATH_CALUDE_six_digit_scrambled_divisibility_l1431_143137


namespace NUMINAMATH_CALUDE_even_quadratic_function_sum_l1431_143138

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_quadratic_function_sum (a b : ℝ) :
  let f := fun x => a * x^2 + b * x
  IsEven f ∧ (∀ x ∈ Set.Icc (a - 1) (2 * a), f x ∈ Set.range f) →
  a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_function_sum_l1431_143138


namespace NUMINAMATH_CALUDE_toy_piles_l1431_143174

theorem toy_piles (total : ℕ) (small : ℕ) (large : ℕ) : 
  total = 120 → 
  large = 2 * small → 
  total = small + large → 
  large = 80 := by
sorry

end NUMINAMATH_CALUDE_toy_piles_l1431_143174


namespace NUMINAMATH_CALUDE_min_fraction_sum_l1431_143135

def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem min_fraction_sum :
  ∃ (W X Y Z : ℕ), W ∈ Digits ∧ X ∈ Digits ∧ Y ∈ Digits ∧ Z ∈ Digits ∧
  W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  ∀ (W' X' Y' Z' : ℕ), W' ∈ Digits → X' ∈ Digits → Y' ∈ Digits → Z' ∈ Digits →
  W' ≠ X' → W' ≠ Y' → W' ≠ Z' → X' ≠ Y' → X' ≠ Z' → Y' ≠ Z' →
  (W : ℚ) / X + (Y : ℚ) / Z ≤ (W' : ℚ) / X' + (Y' : ℚ) / Z' ∧
  (W : ℚ) / X + (Y : ℚ) / Z = 15 / 56 :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l1431_143135


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1431_143162

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, (1 < x ∧ x < 3) → (1 < x ∧ x < 4)) ∧
  ¬(∀ x : ℝ, (1 < x ∧ x < 4) → (1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1431_143162


namespace NUMINAMATH_CALUDE_knight_moves_correct_l1431_143123

/-- The least number of moves for a knight to travel from one corner to the diagonally opposite corner on an n×n chessboard. -/
def knight_moves (n : ℕ) : ℕ := 2 * ((n + 1) / 3)

/-- Theorem: For an n×n chessboard where n ≥ 4, the least number of moves for a knight to travel
    from one corner to the diagonally opposite corner is equal to 2 ⌊(n+1)/3⌋. -/
theorem knight_moves_correct (n : ℕ) (h : n ≥ 4) :
  knight_moves n = 2 * ((n + 1) / 3) :=
by sorry

end NUMINAMATH_CALUDE_knight_moves_correct_l1431_143123


namespace NUMINAMATH_CALUDE_expression_value_proof_l1431_143167

theorem expression_value_proof (a b c k : ℤ) 
  (ha : a = 30) (hb : b = 25) (hc : c = 4) (hk : k = 3) : 
  (a - (b - k * c)) - ((a - k * b) - c) = 66 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_proof_l1431_143167


namespace NUMINAMATH_CALUDE_range_of_function_l1431_143182

theorem range_of_function (y : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ y = x + 4 / x) → y ≤ -4 ∨ y ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l1431_143182


namespace NUMINAMATH_CALUDE_min_squares_for_symmetric_x_l1431_143171

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the state of the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The initial grid with squares at (1,3) and (2,4) shaded -/
def initialGrid : Grid :=
  fun r c => (r = 0 ∧ c = 2) ∨ (r = 1 ∧ c = 3)

/-- Checks if a grid has both vertical and horizontal symmetry -/
def isSymmetric (g : Grid) : Prop :=
  (∀ r c, g r c = g r (3 - c)) ∧  -- Vertical symmetry
  (∀ r c, g r c = g (3 - r) c)    -- Horizontal symmetry

/-- Checks if a grid forms an 'X' shape -/
def formsX (g : Grid) : Prop :=
  (∀ r, g r r = true) ∧ 
  (∀ r, g r (3 - r) = true) ∧
  (∀ r c, r ≠ c ∧ r ≠ (3 - c) → g r c = false)

/-- The main theorem stating that 4 additional squares are needed -/
theorem min_squares_for_symmetric_x : 
  ∃ (finalGrid : Grid),
    (∀ r c, initialGrid r c → finalGrid r c) ∧
    isSymmetric finalGrid ∧
    formsX finalGrid ∧
    (∀ (g : Grid), 
      (∀ r c, initialGrid r c → g r c) → 
      isSymmetric g → 
      formsX g → 
      (∃ (newSquares : List Position),
        newSquares.length = 4 ∧
        (∀ p ∈ newSquares, g p.row p.col ∧ ¬initialGrid p.row p.col))) :=
sorry

end NUMINAMATH_CALUDE_min_squares_for_symmetric_x_l1431_143171


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1431_143152

theorem unknown_number_proof : ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 61 ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1431_143152


namespace NUMINAMATH_CALUDE_circle_equation_line_equation_l1431_143150

/-- A circle C passing through (2,-1), tangent to x+y=1, with center on y=-2x -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : center.1^2 + (center.2 + 1)^2 = radius^2
  tangent_to_line : |center.1 + center.2 - 1| / Real.sqrt 2 = radius
  center_on_line : center.2 = -2 * center.1

/-- A line passing through the origin and cutting a chord of length 2 on CircleC -/
structure LineL (c : CircleC) where
  slope : ℝ
  passes_origin : True
  cuts_chord : (2 * c.radius / Real.sqrt (1 + slope^2))^2 = 4

theorem circle_equation (c : CircleC) :
  ∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

theorem line_equation (c : CircleC) (l : LineL c) :
  (l.slope = 0 ∧ ∀ x y : ℝ, y = l.slope * x ↔ x = 0) ∨
  (l.slope = -3/4 ∧ ∀ x y : ℝ, y = l.slope * x ↔ y = -3/4 * x) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_line_equation_l1431_143150


namespace NUMINAMATH_CALUDE_four_bottles_cost_l1431_143153

/-- The cost of a certain number of bottles of mineral water -/
def cost (bottles : ℕ) : ℚ :=
  if bottles = 3 then 3/2 else (3/2) * (bottles : ℚ) / 3

/-- Theorem: The cost of 4 bottles of mineral water is 2, given that 3 bottles cost 1.50 -/
theorem four_bottles_cost : cost 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_bottles_cost_l1431_143153


namespace NUMINAMATH_CALUDE_difference_of_squares_2023_2022_l1431_143105

theorem difference_of_squares_2023_2022 : 2023^2 - 2022^2 = 4045 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_2023_2022_l1431_143105


namespace NUMINAMATH_CALUDE_f_monotonicity_and_min_value_l1431_143144

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - 2 / x - a * (Real.log x - 1 / x^2)

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2) * (x - a) / x^3

-- Define the minimum value function g
def g (a : ℝ) : ℝ := a - a * Real.log a - 1 / a

-- Theorem statement
theorem f_monotonicity_and_min_value (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Ioo 0 a, f_deriv a x < 0) ∧
  (∀ x ∈ Set.Ioi a, f_deriv a x > 0) ∧
  g a < 1 := by sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_min_value_l1431_143144


namespace NUMINAMATH_CALUDE_plane_equation_3d_l1431_143156

/-- Definition of a line in 2D Cartesian coordinate system -/
def is_line_2d (A B C : ℝ) : Prop :=
  A^2 + B^2 ≠ 0

/-- Definition of a plane in 3D Cartesian coordinate system -/
def is_plane_3d (A B C D : ℝ) : Prop :=
  A^2 + B^2 + C^2 ≠ 0

/-- Theorem stating the equation of a plane in 3D Cartesian coordinate system -/
theorem plane_equation_3d (A B C D : ℝ) :
  is_plane_3d A B C D ↔ ∃ (x y z : ℝ), A*x + B*y + C*z + D = 0 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_3d_l1431_143156


namespace NUMINAMATH_CALUDE_johns_allowance_l1431_143161

/-- Calculates the amount of allowance John received given his initial amount, spending, and final amount -/
def calculate_allowance (initial : ℕ) (spent : ℕ) (final : ℕ) : ℕ :=
  final - (initial - spent)

/-- Proves that John's allowance was 26 dollars given the problem conditions -/
theorem johns_allowance :
  let initial := 5
  let spent := 2
  let final := 29
  calculate_allowance initial spent final = 26 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l1431_143161


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l1431_143129

/-- The line equation y = kx + 1 - 2k -/
def line (k x : ℝ) : ℝ := k * x + 1 - 2 * k

/-- The ellipse equation x²/9 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The point P(2,1) is inside the ellipse -/
def point_inside_ellipse : Prop := 2^2 / 9 + 1^2 / 4 < 1

theorem line_intersects_ellipse :
  ∀ k : ℝ, ∃ x y : ℝ, line k x = y ∧ ellipse x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l1431_143129


namespace NUMINAMATH_CALUDE_four_numbers_sum_l1431_143139

theorem four_numbers_sum (a b c d : ℤ) :
  a + b + c = 21 ∧
  a + b + d = 28 ∧
  a + c + d = 29 ∧
  b + c + d = 30 →
  a = 6 ∧ b = 7 ∧ c = 8 ∧ d = 15 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_sum_l1431_143139


namespace NUMINAMATH_CALUDE_right_triangle_bisector_properties_l1431_143141

/-- Represents a right-angled triangle with an angle bisector --/
structure RightTriangleWithBisector where
  -- AC and BC are the legs, AB is the hypotenuse
  -- D is the point where the angle bisector from A intersects BC
  α : Real  -- angle BAC
  β : Real  -- angle ABC
  k : Real  -- ratio AD/DB

/-- Theorem about properties of a right-angled triangle with angle bisector --/
theorem right_triangle_bisector_properties (t : RightTriangleWithBisector) :
  -- 1. The problem has a solution for all k > 0
  (t.k > 0) →
  -- 2. The triangle is isosceles when k = √(2 + √2)
  (t.α = π/4 ↔ t.k = Real.sqrt (2 + Real.sqrt 2)) ∧
  -- 3. When k = 7/2, α = arccos(7/8) and β = π/2 - arccos(7/8)
  (t.k = 7/2 →
    t.α = Real.arccos (7/8) ∧
    t.β = π/2 - Real.arccos (7/8)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_bisector_properties_l1431_143141


namespace NUMINAMATH_CALUDE_stratified_sample_female_result_l1431_143131

/-- Represents the number of female athletes to be selected in a stratified sample -/
def stratified_sample_female (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_female * sample_size) / (total_male + total_female)

/-- Theorem: In a stratified sampling of 28 people from a population of 98 athletes 
    (56 male and 42 female), the number of female athletes that should be selected is 12 -/
theorem stratified_sample_female_result : 
  stratified_sample_female 56 42 28 = 12 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_female_result_l1431_143131


namespace NUMINAMATH_CALUDE_marathon_distance_theorem_l1431_143122

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Converts a MarathonDistance to total yards -/
def marathonToYards (d : MarathonDistance) : ℕ :=
  d.miles * 1760 + d.yards

/-- The standard marathon distance -/
def standardMarathon : MarathonDistance :=
  { miles := 26, yards := 395 }

/-- Converts total yards to miles and remaining yards -/
def yardsToMilesAndYards (totalYards : ℕ) : MarathonDistance :=
  { miles := totalYards / 1760,
    yards := totalYards % 1760 }

theorem marathon_distance_theorem :
  let totalYards := 15 * marathonToYards standardMarathon
  let result := yardsToMilesAndYards totalYards
  result.yards = 645 := by sorry

end NUMINAMATH_CALUDE_marathon_distance_theorem_l1431_143122


namespace NUMINAMATH_CALUDE_problem_1_l1431_143170

theorem problem_1 : -53 + 21 - (-79) - 37 = 10 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1431_143170


namespace NUMINAMATH_CALUDE_unique_solution_x2024_y3_3y_l1431_143111

theorem unique_solution_x2024_y3_3y :
  ∀ x y : ℤ, x^2024 + y^3 = 3*y ↔ x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x2024_y3_3y_l1431_143111


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1431_143114

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  ((a - 2*b)^2 - (a + 3*b)*(a - 2*b)) / b = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1431_143114


namespace NUMINAMATH_CALUDE_defective_more_likely_from_machine2_l1431_143158

-- Define the probabilities
def p_machine1 : ℝ := 0.8
def p_machine2 : ℝ := 0.2
def p_defect_machine1 : ℝ := 0.01
def p_defect_machine2 : ℝ := 0.05

-- Define the events
def B1 := "part manufactured by first machine"
def B2 := "part manufactured by second machine"
def A := "part is defective"

-- Define the probability of a part being defective
def p_defective : ℝ := p_machine1 * p_defect_machine1 + p_machine2 * p_defect_machine2

-- Theorem to prove
theorem defective_more_likely_from_machine2 :
  (p_machine2 * p_defect_machine2) / p_defective > (p_machine1 * p_defect_machine1) / p_defective :=
sorry

end NUMINAMATH_CALUDE_defective_more_likely_from_machine2_l1431_143158


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l1431_143104

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l1431_143104


namespace NUMINAMATH_CALUDE_train_length_l1431_143117

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ (length : ℝ), abs (length - 250.05) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1431_143117


namespace NUMINAMATH_CALUDE_exactly_three_valid_combinations_l1431_143119

/-- Represents the number of pairs of socks at each price point -/
structure SockCombination :=
  (x : ℕ)  -- Number of 18 yuan socks
  (y : ℕ)  -- Number of 30 yuan socks
  (z : ℕ)  -- Number of 39 yuan socks

/-- Checks if a combination is valid according to the problem constraints -/
def isValidCombination (c : SockCombination) : Prop :=
  18 * c.x + 30 * c.y + 39 * c.z = 100 ∧
  18 * c.x + 30 * c.y + 39 * c.z > 95

/-- The main theorem stating that there are exactly 3 valid combinations -/
theorem exactly_three_valid_combinations :
  ∃! (s : Finset SockCombination), 
    (∀ c ∈ s, isValidCombination c) ∧ 
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_valid_combinations_l1431_143119


namespace NUMINAMATH_CALUDE_middle_integer_is_five_l1431_143151

/-- A function that checks if a number is a one-digit positive integer -/
def isOneDigitPositive (n : ℕ) : Prop := 0 < n ∧ n < 10

/-- A function that checks if a number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- The main theorem -/
theorem middle_integer_is_five :
  ∀ n : ℕ,
  isOneDigitPositive n ∧
  isOdd n ∧
  isOneDigitPositive (n - 2) ∧
  isOdd (n - 2) ∧
  isOneDigitPositive (n + 2) ∧
  isOdd (n + 2) ∧
  ((n - 2) + n + (n + 2)) = ((n - 2) * n * (n + 2)) / 8
  →
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_middle_integer_is_five_l1431_143151


namespace NUMINAMATH_CALUDE_total_pages_read_l1431_143149

-- Define reading speeds for each genre and focus level
def novel_speed : Fin 3 → ℕ
| 0 => 21  -- low focus
| 1 => 25  -- medium focus
| 2 => 30  -- high focus
| _ => 0

def graphic_novel_speed : Fin 3 → ℕ
| 0 => 30  -- low focus
| 1 => 36  -- medium focus
| 2 => 42  -- high focus
| _ => 0

def comic_book_speed : Fin 3 → ℕ
| 0 => 45  -- low focus
| 1 => 54  -- medium focus
| 2 => 60  -- high focus
| _ => 0

def non_fiction_speed : Fin 3 → ℕ
| 0 => 18  -- low focus
| 1 => 22  -- medium focus
| 2 => 28  -- high focus
| _ => 0

def biography_speed : Fin 3 → ℕ
| 0 => 20  -- low focus
| 1 => 24  -- medium focus
| 2 => 29  -- high focus
| _ => 0

-- Define time allocations for each hour
def hour1_allocation : List (ℕ × ℕ × ℕ) := [
  (20, 2, 0),  -- 20 minutes, high focus, novel
  (10, 0, 1),  -- 10 minutes, low focus, graphic novel
  (15, 1, 3),  -- 15 minutes, medium focus, non-fiction
  (15, 0, 4)   -- 15 minutes, low focus, biography
]

def hour2_allocation : List (ℕ × ℕ × ℕ) := [
  (25, 1, 2),  -- 25 minutes, medium focus, comic book
  (15, 2, 1),  -- 15 minutes, high focus, graphic novel
  (20, 0, 0)   -- 20 minutes, low focus, novel
]

def hour3_allocation : List (ℕ × ℕ × ℕ) := [
  (10, 2, 3),  -- 10 minutes, high focus, non-fiction
  (20, 1, 4),  -- 20 minutes, medium focus, biography
  (30, 0, 2)   -- 30 minutes, low focus, comic book
]

-- Function to calculate pages read for a given time, focus, and genre
def pages_read (time : ℕ) (focus : Fin 3) (genre : Fin 5) : ℚ :=
  let speed := match genre with
    | 0 => novel_speed focus
    | 1 => graphic_novel_speed focus
    | 2 => comic_book_speed focus
    | 3 => non_fiction_speed focus
    | 4 => biography_speed focus
    | _ => 0
  (time : ℚ) / 60 * speed

-- Function to calculate total pages read for a list of allocations
def total_pages (allocations : List (ℕ × ℕ × ℕ)) : ℚ :=
  allocations.foldl (fun acc (time, focus, genre) => acc + pages_read time ⟨focus, by sorry⟩ ⟨genre, by sorry⟩) 0

-- Theorem stating the total pages read
theorem total_pages_read :
  ⌊total_pages hour1_allocation + total_pages hour2_allocation + total_pages hour3_allocation⌋ = 100 := by
  sorry


end NUMINAMATH_CALUDE_total_pages_read_l1431_143149


namespace NUMINAMATH_CALUDE_draining_time_is_independent_variable_l1431_143166

/-- Represents the water volume in the reservoir --/
def water_volume (t : ℝ) : ℝ := 50 - 2 * t

theorem draining_time_is_independent_variable :
  ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → water_volume t₁ ≠ water_volume t₂ :=
by sorry

end NUMINAMATH_CALUDE_draining_time_is_independent_variable_l1431_143166


namespace NUMINAMATH_CALUDE_x_power_3a_plus_2b_l1431_143157

theorem x_power_3a_plus_2b (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + 2*b) = 72 := by
  sorry

end NUMINAMATH_CALUDE_x_power_3a_plus_2b_l1431_143157


namespace NUMINAMATH_CALUDE_min_value_expression_l1431_143142

theorem min_value_expression (x : ℝ) (h : x > 10) :
  (x^2 + 36) / (x - 10) ≥ 4 * Real.sqrt 34 + 20 ∧
  (x^2 + 36) / (x - 10) = 4 * Real.sqrt 34 + 20 ↔ x = 10 + 2 * Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1431_143142


namespace NUMINAMATH_CALUDE_house_rooms_l1431_143127

theorem house_rooms (outlets_per_room : ℕ) (total_outlets : ℕ) (h1 : outlets_per_room = 6) (h2 : total_outlets = 42) :
  total_outlets / outlets_per_room = 7 := by
  sorry

end NUMINAMATH_CALUDE_house_rooms_l1431_143127


namespace NUMINAMATH_CALUDE_fruit_shop_apples_l1431_143199

/-- Given the ratio of mangoes : oranges : apples and the number of mangoes,
    calculate the number of apples -/
theorem fruit_shop_apples (ratio_mangoes ratio_oranges ratio_apples num_mangoes : ℕ) 
    (h_ratio : ratio_mangoes = 10 ∧ ratio_oranges = 2 ∧ ratio_apples = 3)
    (h_mangoes : num_mangoes = 120) :
    (num_mangoes / ratio_mangoes) * ratio_apples = 36 := by
  sorry

#check fruit_shop_apples

end NUMINAMATH_CALUDE_fruit_shop_apples_l1431_143199


namespace NUMINAMATH_CALUDE_smallest_M_for_inequality_l1431_143181

open Real

/-- The smallest real number M such that |∑ab(a²-b²)| ≤ M(∑a²)² holds for all real a, b, c -/
theorem smallest_M_for_inequality : 
  ∃ (M : ℝ), M = (9 * Real.sqrt 2) / 32 ∧ 
  (∀ (a b c : ℝ), |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ), |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') :=
sorry


end NUMINAMATH_CALUDE_smallest_M_for_inequality_l1431_143181


namespace NUMINAMATH_CALUDE_letter_count_cycle_exists_l1431_143173

/-- Represents the number of letters in the Russian word for a number -/
def russianWordLength (n : ℕ) : ℕ := sorry

/-- Generates the sequence of letter counts -/
def letterCountSequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => russianWordLength (letterCountSequence start n)

/-- Checks if a sequence has entered a cycle -/
def hasCycle (seq : ℕ → ℕ) (start : ℕ) (length : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i, seq (k + i) = seq (k + i + length)

theorem letter_count_cycle_exists (start : ℕ) :
  ∃ k length : ℕ, hasCycle (letterCountSequence start) k length :=
sorry

end NUMINAMATH_CALUDE_letter_count_cycle_exists_l1431_143173


namespace NUMINAMATH_CALUDE_value_of_r_l1431_143169

theorem value_of_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_r_l1431_143169


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1431_143196

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 100 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1431_143196


namespace NUMINAMATH_CALUDE_square_9801_difference_of_squares_l1431_143197

theorem square_9801_difference_of_squares (x : ℤ) (h : x^2 = 9801) :
  (x + 1) * (x - 1) = 9800 := by
sorry

end NUMINAMATH_CALUDE_square_9801_difference_of_squares_l1431_143197


namespace NUMINAMATH_CALUDE_initial_ratio_proof_l1431_143103

theorem initial_ratio_proof (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 560 →
  new_boarders = 80 →
  ∃ (initial_day_scholars : ℕ),
    (initial_boarders : ℚ) / initial_day_scholars = 7 / 16 ∧
    (initial_boarders + new_boarders : ℚ) / initial_day_scholars = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_ratio_proof_l1431_143103
