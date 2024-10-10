import Mathlib

namespace min_trips_for_elevator_l1916_191696

def weights : List ℕ := [50, 51, 55, 57, 58, 59, 60, 63, 75, 140]
def max_capacity : ℕ := 180

def is_valid_trip (trip : List ℕ) : Prop :=
  trip.sum ≤ max_capacity

def covers_all_weights (trips : List (List ℕ)) : Prop :=
  weights.all (λ w => ∃ t ∈ trips, w ∈ t)

theorem min_trips_for_elevator : 
  ∃ (trips : List (List ℕ)), 
    trips.length = 4 ∧ 
    (∀ t ∈ trips, is_valid_trip t) ∧
    covers_all_weights trips ∧
    (∀ (other_trips : List (List ℕ)), 
      (∀ t ∈ other_trips, is_valid_trip t) → 
      covers_all_weights other_trips → 
      other_trips.length ≥ 4) :=
by sorry

end min_trips_for_elevator_l1916_191696


namespace factorization_problem_1_factorization_problem_2_l1916_191653

-- Problem 1
theorem factorization_problem_1 (a b x y : ℝ) :
  a * (x + y) - 2 * b * (x + y) = (x + y) * (a - 2 * b) := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  a^3 + 2*a^2*b + a*b^2 = a * (a + b)^2 := by sorry

end factorization_problem_1_factorization_problem_2_l1916_191653


namespace man_speed_man_speed_proof_l1916_191663

/-- The speed of a man relative to a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- Proof that the speed of the man is approximately 0.833 m/s given the specified conditions. -/
theorem man_speed_proof :
  let train_length : ℝ := 500
  let train_speed_kmh : ℝ := 63
  let crossing_time : ℝ := 29.997600191984642
  abs (man_speed train_length train_speed_kmh crossing_time - 0.833) < 0.001 := by
  sorry


end man_speed_man_speed_proof_l1916_191663


namespace equidistant_planes_count_l1916_191693

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to check if 4 points are coplanar
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Function to check if a plane is equidistant from 4 points
def is_equidistant_plane (plane : Plane3D) (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Theorem statement
theorem equidistant_planes_count 
  (p1 p2 p3 p4 : Point3D) 
  (h : ¬ are_coplanar p1 p2 p3 p4) : 
  ∃! (planes : Finset Plane3D), 
    (planes.card = 7) ∧ 
    (∀ plane ∈ planes, is_equidistant_plane plane p1 p2 p3 p4) := by
  sorry

end equidistant_planes_count_l1916_191693


namespace inverse_function_solution_l1916_191610

/-- Given a function g(x) = 1 / (2ax + b), where a and b are non-zero constants and a ≠ b,
    prove that the solution to g^(-1)(x) = 0 is x = 1/b. -/
theorem inverse_function_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  let g : ℝ → ℝ := λ x ↦ 1 / (2 * a * x + b)
  ∃! x, Function.invFun g x = 0 ∧ x = 1 / b :=
by sorry

end inverse_function_solution_l1916_191610


namespace middle_card_first_round_l1916_191685

/-- Represents a card with a positive integer value -/
structure Card where
  value : ℕ+
  
/-- Represents a player in the game -/
structure Player where
  totalCounters : ℕ
  lastRoundCard : Card

/-- Represents the game state -/
structure GameState where
  cards : Fin 3 → Card
  players : Fin 3 → Player
  rounds : ℕ

/-- Conditions of the game -/
def gameConditions (g : GameState) : Prop :=
  g.rounds ≥ 2 ∧
  (g.cards 0).value < (g.cards 1).value ∧ (g.cards 1).value < (g.cards 2).value ∧
  (g.players 0).totalCounters + (g.players 1).totalCounters + (g.players 2).totalCounters = 39 ∧
  (∃ i j k : Fin 3, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (g.players i).totalCounters = 20 ∧
    (g.players j).totalCounters = 10 ∧
    (g.players k).totalCounters = 9) ∧
  (∃ i : Fin 3, (g.players i).totalCounters = 10 ∧
    (g.players i).lastRoundCard = g.cards 2)

/-- The theorem to be proved -/
theorem middle_card_first_round (g : GameState) :
  gameConditions g →
  ∃ i : Fin 3, (g.players i).totalCounters = 9 ∧
    (∃ firstRoundCard : Card, firstRoundCard = g.cards 1) :=
sorry

end middle_card_first_round_l1916_191685


namespace red_balls_count_l1916_191655

theorem red_balls_count (total_balls : ℕ) (red_prob : ℚ) (red_balls : ℕ) : 
  total_balls = 15 → red_prob = 1/3 → red_balls = (red_prob * total_balls).num → red_balls = 5 := by
  sorry

end red_balls_count_l1916_191655


namespace simplify_square_roots_l1916_191652

theorem simplify_square_roots : 
  Real.sqrt (10 + 6 * Real.sqrt 2) + Real.sqrt (10 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end simplify_square_roots_l1916_191652


namespace eleven_percent_greater_than_80_l1916_191624

theorem eleven_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 11 / 100) → x = 88.8 := by
sorry

end eleven_percent_greater_than_80_l1916_191624


namespace rod_length_for_given_weight_l1916_191614

/-- Represents the properties of a uniform rod -/
structure UniformRod where
  length_kg_ratio : ℝ  -- Ratio of length to weight

/-- Calculates the length of a uniform rod given its weight -/
def rod_length (rod : UniformRod) (weight : ℝ) : ℝ :=
  weight * rod.length_kg_ratio

theorem rod_length_for_given_weight 
  (rod : UniformRod) 
  (h1 : rod_length rod 42.75 = 11.25)
  (h2 : rod.length_kg_ratio = 11.25 / 42.75) : 
  rod_length rod 26.6 = 7 := by
  sorry

#check rod_length_for_given_weight

end rod_length_for_given_weight_l1916_191614


namespace at_least_one_real_root_l1916_191687

theorem at_least_one_real_root (p₁ p₂ q₁ q₂ : ℝ) (h : p₁ * p₂ = 2 * (q₁ + q₂)) :
  (∃ x : ℝ, x^2 + p₁*x + q₁ = 0) ∨ (∃ x : ℝ, x^2 + p₂*x + q₂ = 0) :=
sorry

end at_least_one_real_root_l1916_191687


namespace donation_scientific_correct_l1916_191630

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The donation amount in yuan -/
def donation_amount : ℝ := 2175000000

/-- The scientific notation of the donation amount -/
def donation_scientific : ScientificNotation := {
  coefficient := 2.175,
  exponent := 9,
  valid := by sorry
}

/-- Theorem stating that the donation amount is correctly represented in scientific notation -/
theorem donation_scientific_correct : 
  donation_amount = donation_scientific.coefficient * (10 : ℝ) ^ donation_scientific.exponent :=
by sorry

end donation_scientific_correct_l1916_191630


namespace set_relation_proof_l1916_191674

theorem set_relation_proof (M P : Set α) (h_nonempty : M.Nonempty) 
  (h_not_subset : ¬(M ⊆ P)) : 
  (∃ x ∈ M, x ∉ P) ∧ ¬(∀ x ∈ M, x ∈ P) := by
  sorry

end set_relation_proof_l1916_191674


namespace expand_polynomial_product_l1916_191651

theorem expand_polynomial_product : ∀ t : ℝ,
  (3 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) = 
  -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12 := by
  sorry

end expand_polynomial_product_l1916_191651


namespace unit_digit_of_3_to_58_l1916_191642

theorem unit_digit_of_3_to_58 : 3^58 % 10 = 9 := by
  sorry

end unit_digit_of_3_to_58_l1916_191642


namespace function_comparison_l1916_191678

theorem function_comparison (x₁ x₂ : ℝ) (h1 : x₁ < x₂) (h2 : x₁ + x₂ = 0) :
  let f := fun x => x^2 + 2*x + 4
  f x₁ < f x₂ := by
sorry

end function_comparison_l1916_191678


namespace max_money_collectible_l1916_191627

-- Define the structure of the land plot
structure LandPlot where
  circles : Fin 36 → ℕ
  -- circles represents the amount of money in each of the 36 circles

-- Define the concept of a valid path
def ValidPath (plot : LandPlot) (path : List (Fin 36)) : Prop :=
  -- A path is valid if it doesn't pass twice along the same straight line
  -- The actual implementation of this condition is complex and omitted here
  sorry

-- Define the sum of money collected along a path
def PathSum (plot : LandPlot) (path : List (Fin 36)) : ℕ :=
  path.map plot.circles |> List.sum

-- The main theorem
theorem max_money_collectible (plot : LandPlot) : 
  (∃ (path : List (Fin 36)), ValidPath plot path ∧ PathSum plot path = 47) ∧
  (∀ (path : List (Fin 36)), ValidPath plot path → PathSum plot path ≤ 47) := by
  sorry

end max_money_collectible_l1916_191627


namespace cost_2005_l1916_191688

/-- Represents the number of songs downloaded in 2004 -/
def songs_2004 : ℕ := 200

/-- Represents the number of songs downloaded in 2005 -/
def songs_2005 : ℕ := 360

/-- Represents the difference in cost per song between 2004 and 2005 in cents -/
def cost_difference : ℕ := 32

/-- Theorem stating that the cost of downloading 360 songs in 2005 was $144.00 -/
theorem cost_2005 (c : ℚ) : 
  (songs_2005 : ℚ) * c = (songs_2004 : ℚ) * (c + cost_difference) → 
  songs_2005 * c = 14400 := by
  sorry

end cost_2005_l1916_191688


namespace water_margin_price_l1916_191649

theorem water_margin_price :
  ∀ (x : ℝ),
    (x > 0) →
    (3600 / (x + 60) = (1 / 2) * (4800 / x)) →
    x = 120 :=
by
  sorry

end water_margin_price_l1916_191649


namespace replaced_person_age_l1916_191638

theorem replaced_person_age 
  (n : ℕ) 
  (original_avg : ℝ) 
  (new_avg : ℝ) 
  (new_person_age : ℝ) 
  (h1 : n = 10)
  (h2 : original_avg = new_avg + 3)
  (h3 : new_person_age = 12) : 
  n * original_avg - (n * new_avg + new_person_age) = 18 := by
sorry

end replaced_person_age_l1916_191638


namespace chloe_treasures_l1916_191600

theorem chloe_treasures (points_per_treasure : ℕ) (second_level_treasures : ℕ) (total_score : ℕ) 
  (h1 : points_per_treasure = 9)
  (h2 : second_level_treasures = 3)
  (h3 : total_score = 81) :
  total_score = points_per_treasure * (second_level_treasures + 6) :=
by sorry

end chloe_treasures_l1916_191600


namespace a_equals_one_sufficient_not_necessary_l1916_191615

def is_parallel (a : ℝ) : Prop :=
  a^2 = 1

theorem a_equals_one_sufficient_not_necessary :
  (∃ a : ℝ, is_parallel a ∧ a ≠ 1) ∧
  (∀ a : ℝ, a = 1 → is_parallel a) :=
by sorry

end a_equals_one_sufficient_not_necessary_l1916_191615


namespace boys_not_adjacent_or_ends_probability_l1916_191616

/-- The number of boys in the lineup -/
def num_boys : ℕ := 2

/-- The number of girls in the lineup -/
def num_girls : ℕ := 4

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of spaces between girls where boys can be placed -/
def available_spaces : ℕ := num_girls - 1

/-- The probability that the boys are neither adjacent nor at the ends in a lineup of boys and girls -/
theorem boys_not_adjacent_or_ends_probability :
  (num_boys.factorial * num_girls.factorial * available_spaces.choose num_boys) / total_people.factorial = 1 / 5 := by
  sorry

end boys_not_adjacent_or_ends_probability_l1916_191616


namespace line_ellipse_intersection_slopes_l1916_191668

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line with slope m and y-intercept -3
def line (m x : ℝ) : ℝ := m * x - 3

-- Define the set of valid slopes
def valid_slopes : Set ℝ := {m : ℝ | m ≤ -Real.sqrt (4/55) ∨ m ≥ Real.sqrt (4/55)}

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x : ℝ, ellipse x (line m x)) ↔ m ∈ valid_slopes :=
sorry

end line_ellipse_intersection_slopes_l1916_191668


namespace train_length_l1916_191664

/-- The length of a train given its speed and the time it takes to cross a platform of known length. -/
theorem train_length (train_speed : Real) (platform_length : Real) (crossing_time : Real) :
  train_speed = 72 / 3.6 →
  platform_length = 50.024 →
  crossing_time = 15 →
  train_speed * crossing_time - platform_length = 249.976 := by
  sorry

#check train_length

end train_length_l1916_191664


namespace stack_map_views_l1916_191670

def StackMap : Type := List (List Nat)

def frontView (sm : StackMap) : List Nat :=
  sm.map (List.foldl max 0)

def rightSideView (sm : StackMap) : List Nat :=
  List.map (List.foldl max 0) (List.transpose sm)

theorem stack_map_views (sm : StackMap) 
  (h1 : sm = [[3, 1, 2], [2, 4, 3], [1, 1, 3]]) : 
  frontView sm = [3, 4, 3] ∧ rightSideView sm = [3, 4, 3] := by
  sorry

end stack_map_views_l1916_191670


namespace max_sum_after_erasing_l1916_191641

-- Define the initial set of numbers
def initial_numbers : List ℕ := List.range 13 |>.map (· + 4)

-- Define a function to check if a list can be divided into groups with equal sums
def can_be_divided_equally (numbers : List ℕ) : Prop :=
  ∃ (k : ℕ) (groups : List (List ℕ)),
    k > 1 ∧
    groups.length = k ∧
    groups.all (λ group ↦ group.sum = (numbers.sum / k)) ∧
    groups.join.toFinset = numbers.toFinset

-- Define the theorem
theorem max_sum_after_erasing (numbers : List ℕ) :
  numbers.sum = 121 →
  numbers ⊆ initial_numbers →
  ¬ can_be_divided_equally numbers →
  ∀ (other_numbers : List ℕ),
    other_numbers ⊆ initial_numbers →
    other_numbers.sum > 121 →
    can_be_divided_equally other_numbers :=
sorry

end max_sum_after_erasing_l1916_191641


namespace average_cost_per_meter_l1916_191631

def silk_length : Real := 9.25
def silk_cost : Real := 416.25
def cotton_length : Real := 7.5
def cotton_cost : Real := 337.50
def wool_length : Real := 6
def wool_cost : Real := 378

def total_length : Real := silk_length + cotton_length + wool_length
def total_cost : Real := silk_cost + cotton_cost + wool_cost

theorem average_cost_per_meter : total_cost / total_length = 49.75 := by sorry

end average_cost_per_meter_l1916_191631


namespace parallel_line_through_point_l1916_191606

/-- 
Given a line L1 with equation 2x - y + 1 = 0, and a point P (1, 1),
prove that the line L2 passing through P and parallel to L1 has the equation 2x - y - 1 = 0.
-/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x - y + 1 = 0) →  -- L1 equation
  (∃ c : ℝ, 2 * x - y + c = 0 ∧ 2 * 1 - 1 + c = 0) →  -- L2 passes through (1, 1)
  (2 * x - y - 1 = 0)  -- L2 equation
:= by sorry

end parallel_line_through_point_l1916_191606


namespace pentagon_angle_measure_l1916_191671

/-- The measure of angle P in a pentagon PQRST where ∠P = 2∠Q = 4∠R = 3∠S = 6∠T is 240° -/
theorem pentagon_angle_measure (P Q R S T : ℝ) : 
  P + Q + R + S + T = 540 → -- sum of angles in a pentagon
  P = 2 * Q →              -- ∠P = 2∠Q
  P = 4 * R →              -- ∠P = 4∠R
  P = 3 * S →              -- ∠P = 3∠S
  P = 6 * T →              -- ∠P = 6∠T
  P = 240 := by            -- ∠P = 240°
sorry


end pentagon_angle_measure_l1916_191671


namespace infinitely_many_solutions_l1916_191613

theorem infinitely_many_solutions (d : ℝ) : 
  (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ d = 5 := by
  sorry

end infinitely_many_solutions_l1916_191613


namespace tangent_addition_subtraction_l1916_191692

theorem tangent_addition_subtraction (γ β : Real) (h1 : Real.tan γ = 5) (h2 : Real.tan β = 3) :
  Real.tan (γ + β) = -4/7 ∧ Real.tan (γ - β) = 1/8 := by
  sorry

end tangent_addition_subtraction_l1916_191692


namespace perpendicular_lines_from_perpendicular_planes_l1916_191644

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (a b : Line) (α β : Plane)
  (h1 : perp_line_plane a α)
  (h2 : perp_line_plane b β)
  (h3 : perp_plane α β) :
  perp_line a b :=
sorry

end perpendicular_lines_from_perpendicular_planes_l1916_191644


namespace min_acquaintances_in_village_l1916_191677

/-- Represents a village with residents and their acquaintances. -/
structure Village where
  residents : Finset ℕ
  acquaintances : Finset (ℕ × ℕ)

/-- Checks if a given set of residents can be seated according to the problem's conditions. -/
def canBeSeatedCircularly (v : Village) (group : Finset ℕ) : Prop :=
  group.card = 6 ∧ ∃ (seating : Fin 6 → ℕ), 
    (∀ i, seating i ∈ group) ∧
    (∀ i, (seating i, seating ((i + 1) % 6)) ∈ v.acquaintances ∧
          (seating i, seating ((i + 5) % 6)) ∈ v.acquaintances)

/-- The main theorem statement. -/
theorem min_acquaintances_in_village (v : Village) :
  v.residents.card = 200 ∧ 
  (∀ group : Finset ℕ, group ⊆ v.residents → canBeSeatedCircularly v group) →
  v.acquaintances.card = 19600 :=
sorry

end min_acquaintances_in_village_l1916_191677


namespace carnation_dozen_cost_carnation_dozen_cost_proof_l1916_191647

theorem carnation_dozen_cost (single_cost : ℚ) (teacher_dozens : ℕ) (friend_singles : ℕ) (total_spent : ℚ) : ℚ :=
  let dozen_cost := (total_spent - single_cost * friend_singles) / teacher_dozens
  dozen_cost

#check carnation_dozen_cost (1/2) 5 14 25 = 18/5

-- The proof is omitted
theorem carnation_dozen_cost_proof :
  carnation_dozen_cost (1/2) 5 14 25 = 18/5 := by sorry

end carnation_dozen_cost_carnation_dozen_cost_proof_l1916_191647


namespace jason_fire_frequency_l1916_191621

/-- Given the conditions of Jason's gameplay in Duty for Ashes, prove that he fires his weapon every 15 seconds on average. -/
theorem jason_fire_frequency
  (flame_duration : ℕ)
  (total_flame_time : ℕ)
  (seconds_per_minute : ℕ)
  (h1 : flame_duration = 5)
  (h2 : total_flame_time = 20)
  (h3 : seconds_per_minute = 60) :
  (seconds_per_minute : ℚ) / ((total_flame_time : ℚ) / (flame_duration : ℚ)) = 15 := by
  sorry

#check jason_fire_frequency

end jason_fire_frequency_l1916_191621


namespace composition_problem_l1916_191632

theorem composition_problem (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (h_comp : ∀ x, f (g x) = 15 * x + d) :
  d = 18 := by
sorry

end composition_problem_l1916_191632


namespace inscribed_squares_area_ratio_l1916_191658

/-- The ratio of the area of a square inscribed in a quarter-circle to the area of a square inscribed in a full circle, both with radius r -/
theorem inscribed_squares_area_ratio (r : ℝ) (hr : r > 0) :
  ∃ (s₁ s₂ : ℝ),
    s₁ > 0 ∧ s₂ > 0 ∧
    s₁^2 + (s₁/2)^2 = r^2 ∧  -- Square inscribed in quarter-circle
    s₂^2 = 2*r^2 ∧           -- Square inscribed in full circle
    s₁^2 / s₂^2 = 2/5 :=
by sorry

end inscribed_squares_area_ratio_l1916_191658


namespace negative_double_less_than_self_l1916_191608

theorem negative_double_less_than_self (a : ℝ) : a < 0 → 2 * a < a := by sorry

end negative_double_less_than_self_l1916_191608


namespace gcd_lcm_sum_8_12_l1916_191654

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l1916_191654


namespace inequality_solution_range_l1916_191656

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end inequality_solution_range_l1916_191656


namespace inverse_iff_horizontal_line_test_l1916_191683

-- Define a function type
def Function := ℝ → ℝ

-- Define what it means for a function to have an inverse
def HasInverse (f : Function) : Prop :=
  ∃ g : Function, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the horizontal line test
def PassesHorizontalLineTest (f : Function) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- Theorem statement
theorem inverse_iff_horizontal_line_test (f : Function) :
  HasInverse f ↔ PassesHorizontalLineTest f :=
sorry

end inverse_iff_horizontal_line_test_l1916_191683


namespace wages_decrease_percentage_l1916_191640

theorem wages_decrease_percentage (W : ℝ) (x : ℝ) 
  (h1 : W > 0)  -- Wages are positive
  (h2 : 0 ≤ x ∧ x ≤ 100)  -- Percentage decrease is between 0 and 100
  (h3 : 0.30 * (W * (1 - x / 100)) = 1.80 * (0.15 * W)) :  -- Condition from the problem
  x = 10 := by sorry

end wages_decrease_percentage_l1916_191640


namespace sum_of_digits_of_sum_of_prime_factors_2310_l1916_191628

def sum_of_digits (n : ℕ) : ℕ := sorry

def prime_factors (n : ℕ) : List ℕ := sorry

theorem sum_of_digits_of_sum_of_prime_factors_2310 : 
  sum_of_digits (List.sum (prime_factors 2310)) = 10 := by sorry

end sum_of_digits_of_sum_of_prime_factors_2310_l1916_191628


namespace peanut_butter_sandwich_days_l1916_191639

/-- Given:
  - There are 5 school days in a week
  - Karen packs ham sandwiches on 3 school days
  - Karen packs cake on one randomly chosen day
  - The probability of packing a ham sandwich and cake on the same day is 12%
  Prove that Karen packs peanut butter sandwiches on 2 days. -/
theorem peanut_butter_sandwich_days :
  ∀ (total_days ham_days cake_days : ℕ) 
    (prob_ham_and_cake : ℚ),
  total_days = 5 →
  ham_days = 3 →
  cake_days = 1 →
  prob_ham_and_cake = 12 / 100 →
  (ham_days : ℚ) / total_days * (cake_days : ℚ) / total_days = prob_ham_and_cake →
  total_days - ham_days = 2 :=
by sorry

end peanut_butter_sandwich_days_l1916_191639


namespace f_properties_l1916_191607

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

theorem f_properties :
  (∀ x, -2 ≤ f x ∧ f x ≤ 2) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ π) ∧
  (∀ x, f (x + π) = f x) :=
sorry

end f_properties_l1916_191607


namespace angle_X_measure_l1916_191601

-- Define the angles in the configuration
def angle_Y : ℝ := 130
def angle_60 : ℝ := 60
def right_angle : ℝ := 90

-- Theorem statement
theorem angle_X_measure :
  ∀ (angle_X : ℝ),
  -- Conditions
  (angle_Y + (180 - angle_Y) = 180) →  -- Y and Z form a linear pair
  (angle_X + angle_60 + right_angle = 180) →  -- Sum of angles in the smaller triangle
  -- Conclusion
  angle_X = 30 := by
  sorry

end angle_X_measure_l1916_191601


namespace product_of_primes_l1916_191672

theorem product_of_primes (p q : ℕ) : 
  Prime p → Prime q → 
  2 < p → p < 6 → 
  8 < q → q < 24 → 
  15 < p * q → p * q < 36 → 
  p * q = 33 := by sorry

end product_of_primes_l1916_191672


namespace square_octagon_tessellation_l1916_191686

-- Define the internal angles of regular polygons
def square_angle : ℝ := 90
def pentagon_angle : ℝ := 108
def hexagon_angle : ℝ := 120
def octagon_angle : ℝ := 135

-- Define a predicate for seamless tessellation
def can_tessellate (angle1 angle2 : ℝ) : Prop :=
  ∃ (n m : ℕ), n * angle1 + m * angle2 = 360

-- Theorem statement
theorem square_octagon_tessellation :
  can_tessellate square_angle octagon_angle ∧
  ¬can_tessellate square_angle hexagon_angle ∧
  ¬can_tessellate square_angle pentagon_angle ∧
  ¬can_tessellate hexagon_angle octagon_angle ∧
  ¬can_tessellate pentagon_angle octagon_angle :=
sorry

end square_octagon_tessellation_l1916_191686


namespace no_real_solutions_l1916_191634

theorem no_real_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 5) (h2 : y + 1 / x = 1 / 6) : False :=
by
  sorry

end no_real_solutions_l1916_191634


namespace solution_set_l1916_191603

-- Define the variables
variable (a b : ℝ)

-- Define the conditions
def condition1 : Prop := ∀ x : ℝ, (a - b) * x + a + 2 * b > 0 ↔ x > 1 / 2
def condition2 : Prop := a > 0

-- Define the theorem
theorem solution_set (h1 : condition1 a b) (h2 : condition2 a) :
  ∀ x : ℝ, a * x < b ↔ x < -1 :=
sorry

end solution_set_l1916_191603


namespace function_domain_condition_l1916_191666

/-- A function f(x) = (kx + 5) / (kx^2 + 4kx + 3) is defined for all real x if and only if 0 ≤ k < 3/4 -/
theorem function_domain_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (k * x + 5) / (k * x^2 + 4 * k * x + 3)) ↔ 
  (0 ≤ k ∧ k < 3/4) :=
by sorry

end function_domain_condition_l1916_191666


namespace sqrt_expression_equals_repeated_sixes_and_seven_l1916_191622

def digits_of_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

def digits_of_fours (n : ℕ) : ℕ :=
  4 * digits_of_ones n * (10^n) + 4 * digits_of_ones n

theorem sqrt_expression_equals_repeated_sixes_and_seven (n : ℕ) :
  let a := digits_of_ones n
  let fours := digits_of_fours n
  let ones := 10 * a + 1
  let sixes := 6 * a
  Real.sqrt (fours / (2 * n * (1/4)) + ones - sixes) = 6 * a + 1 :=
sorry

end sqrt_expression_equals_repeated_sixes_and_seven_l1916_191622


namespace A_D_mutually_exclusive_not_complementary_l1916_191665

/-- Represents the possible outcomes of a fair die toss -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Event A: an odd number is facing up -/
def event_A (face : DieFace) : Prop :=
  face = DieFace.one ∨ face = DieFace.three ∨ face = DieFace.five

/-- Event D: either 2 or 4 is facing up -/
def event_D (face : DieFace) : Prop :=
  face = DieFace.two ∨ face = DieFace.four

/-- The sample space of a fair die toss -/
def sample_space : Set DieFace :=
  {DieFace.one, DieFace.two, DieFace.three, DieFace.four, DieFace.five, DieFace.six}

theorem A_D_mutually_exclusive_not_complementary :
  (∀ (face : DieFace), ¬(event_A face ∧ event_D face)) ∧
  (∃ (face : DieFace), ¬event_A face ∧ ¬event_D face) :=
by sorry

end A_D_mutually_exclusive_not_complementary_l1916_191665


namespace machine_production_time_difference_l1916_191625

/-- Given two machines X and Y that produce widgets, this theorem proves
    that machine X takes 2 days longer than machine Y to produce W widgets. -/
theorem machine_production_time_difference
  (W : ℝ) -- W represents the number of widgets
  (h1 : (W / 6 + W / 4) * 3 = 5 * W / 4) -- Combined production in 3 days
  (h2 : W / 6 * 18 = 3 * W) -- Machine X production in 18 days
  : (W / (W / 6)) - (W / (W / 4)) = 2 :=
sorry

end machine_production_time_difference_l1916_191625


namespace sales_tax_difference_l1916_191689

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.065 → 
  price * tax_rate1 - price * tax_rate2 = 0.5 := by
  sorry

end sales_tax_difference_l1916_191689


namespace number_puzzle_l1916_191684

theorem number_puzzle (x : ℚ) : 
  (((5 * x - (1/3) * (5 * x)) / 10) + (1/3) * x + (1/2) * x + (1/4) * x) = 68 → x = 48 := by
sorry

end number_puzzle_l1916_191684


namespace invalid_period_pair_l1916_191681

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 48) ≡ a n [ZMOD 35]

def least_period_mod5 (a : ℕ → ℤ) (i : ℕ) : Prop :=
  (∀ n, n ≥ 1 → a (n + i) ≡ a n [ZMOD 5]) ∧
  (∀ k, k < i → ∃ n, n ≥ 1 ∧ ¬(a (n + k) ≡ a n [ZMOD 5]))

def least_period_mod7 (a : ℕ → ℤ) (j : ℕ) : Prop :=
  (∀ n, n ≥ 1 → a (n + j) ≡ a n [ZMOD 7]) ∧
  (∀ k, k < j → ∃ n, n ≥ 1 ∧ ¬(a (n + k) ≡ a n [ZMOD 7]))

theorem invalid_period_pair :
  ∀ a : ℕ → ℤ,
  is_valid_sequence a →
  ∀ i j : ℕ,
  least_period_mod5 a i →
  least_period_mod7 a j →
  (i, j) ≠ (16, 4) :=
by sorry

end invalid_period_pair_l1916_191681


namespace unique_x_value_l1916_191620

theorem unique_x_value (x : ℝ) : x^2 ∈ ({0, 1, x} : Set ℝ) → x = -1 := by
  sorry

end unique_x_value_l1916_191620


namespace coat_cost_after_discount_l1916_191682

/-- The cost of Mr. Zubir's purchases --/
structure Purchase where
  pants : ℝ
  shirt : ℝ
  coat : ℝ

/-- The conditions of Mr. Zubir's purchase --/
def purchase_conditions (p : Purchase) : Prop :=
  p.pants + p.shirt = 100 ∧
  p.pants + p.coat = 244 ∧
  p.coat = 5 * p.shirt

/-- The discount rate applied to the purchase --/
def discount_rate : ℝ := 0.1

/-- Theorem stating the cost of the coat after discount --/
theorem coat_cost_after_discount (p : Purchase) 
  (h : purchase_conditions p) : 
  p.coat * (1 - discount_rate) = 162 := by
  sorry

end coat_cost_after_discount_l1916_191682


namespace line_passes_through_fixed_point_l1916_191676

theorem line_passes_through_fixed_point (p q : ℝ) (h : 3 * p - 2 * q = 1) :
  p * (-3/2) + 3 * (1/6) + q = 0 := by
sorry

end line_passes_through_fixed_point_l1916_191676


namespace parallel_tangents_intersection_l1916_191636

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (2 * x₀ = -3 * x₀^2) → (x₀ = 0 ∨ x₀ = -2/3) := by sorry

end parallel_tangents_intersection_l1916_191636


namespace final_savings_calculation_l1916_191637

/-- Calculates the final savings after a given period --/
def calculateFinalSavings (initialSavings : ℕ) (monthlyIncome : ℕ) (monthlyExpenses : ℕ) (months : ℕ) : ℕ :=
  initialSavings + months * monthlyIncome - months * monthlyExpenses

/-- Theorem stating that the final savings will be 1106900 rubles --/
theorem final_savings_calculation :
  let initialSavings : ℕ := 849400
  let monthlyIncome : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
  let monthlyExpenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
  let months : ℕ := 5
  calculateFinalSavings initialSavings monthlyIncome monthlyExpenses months = 1106900 := by
  sorry

#eval calculateFinalSavings 849400 (45000 + 35000 + 7000 + 10000 + 13000) (30000 + 10000 + 5000 + 4500 + 9000) 5

end final_savings_calculation_l1916_191637


namespace extremal_values_sum_l1916_191695

/-- Given real numbers x and y satisfying 4x^2 - 5xy + 4y^2 = 5, 
    S_max and S_min are the maximum and minimum values of x^2 + y^2 respectively. -/
theorem extremal_values_sum (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  let S := x^2 + y^2
  let S_max := (10 : ℝ) / 3
  let S_min := (10 : ℝ) / 13
  (1 / S_max) + (1 / S_min) = 8 / 5 := by
sorry

end extremal_values_sum_l1916_191695


namespace equation_solution_l1916_191680

theorem equation_solution : ∃ x : ℤ, 45 - (5 * 3) = x + 7 ∧ x = 23 := by
  sorry

end equation_solution_l1916_191680


namespace certain_number_minus_one_l1916_191661

theorem certain_number_minus_one (x : ℝ) (h : 15 * x = 45) : x - 1 = 2 := by
  sorry

end certain_number_minus_one_l1916_191661


namespace problem_solution_l1916_191698

theorem problem_solution (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^4 = s^3) 
  (h3 : r - p = 25) : 
  s - q = 73 := by
sorry

end problem_solution_l1916_191698


namespace marks_trees_l1916_191694

theorem marks_trees (initial_trees planted_trees : ℕ) :
  initial_trees = 13 →
  planted_trees = 12 →
  initial_trees + planted_trees = 25 :=
by sorry

end marks_trees_l1916_191694


namespace five_thursdays_in_august_l1916_191648

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def countDayInMonth (startDay : DayOfWeek) (daysInMonth : Nat) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If July has five Tuesdays and both July and August have 31 days, 
    then Thursday must occur five times in August of the same year -/
theorem five_thursdays_in_august 
  (july_start : DayOfWeek) 
  (h1 : countDayInMonth july_start 31 DayOfWeek.Tuesday = 5) 
  : ∃ (august_start : DayOfWeek), 
    countDayInMonth august_start 31 DayOfWeek.Thursday = 5 :=
  sorry

end five_thursdays_in_august_l1916_191648


namespace symmetric_about_x_axis_l1916_191629

-- Define the original function
def g (x : ℝ) : ℝ := x^2 - 3*x

-- Define the symmetric function
def f (x : ℝ) : ℝ := -x^2 + 3*x

-- Theorem statement
theorem symmetric_about_x_axis : 
  ∀ x y : ℝ, g x = y ↔ f x = -y :=
by sorry

end symmetric_about_x_axis_l1916_191629


namespace total_cans_of_peas_l1916_191612

-- Define the number of cans per box
def cans_per_box : ℕ := 4

-- Define the number of boxes ordered
def boxes_ordered : ℕ := 203

-- Theorem to prove
theorem total_cans_of_peas : cans_per_box * boxes_ordered = 812 := by
  sorry

end total_cans_of_peas_l1916_191612


namespace dagger_operation_result_l1916_191619

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_operation_result :
  let result := dagger (5/9) (6/4)
  (result + 1/6) = 27/2 := by
  sorry

end dagger_operation_result_l1916_191619


namespace kangaroo_hop_distance_l1916_191662

theorem kangaroo_hop_distance :
  let a : ℚ := 1/2  -- first term
  let r : ℚ := 3/4  -- common ratio
  let n : ℕ := 7    -- number of terms
  (a * (1 - r^n) / (1 - r) : ℚ) = 14297/2048 := by
  sorry

end kangaroo_hop_distance_l1916_191662


namespace sum_of_powers_l1916_191697

theorem sum_of_powers (k n : ℕ) : 
  (∀ x y : ℝ, 2 * x^k * y^(k+2) + 3 * x^2 * y^n = 5 * x^2 * y^n) → 
  k + n = 6 :=
by sorry

end sum_of_powers_l1916_191697


namespace satisfaction_theorem_l1916_191679

/-- Represents the setup of people around a round table -/
structure TableSetup :=
  (num_men : ℕ)
  (num_women : ℕ)

/-- Defines what it means for a man to be satisfied -/
def is_satisfied (setup : TableSetup) (p : ℝ) : Prop :=
  p = 1 - (setup.num_men - 1) / (setup.num_men + setup.num_women - 1) *
    (setup.num_men - 2) / (setup.num_men + setup.num_women - 2)

/-- The main theorem about the probability of satisfaction and expected number of satisfied men -/
theorem satisfaction_theorem (setup : TableSetup) 
  (h1 : setup.num_men = 50) (h2 : setup.num_women = 50) :
  ∃ (p : ℝ), 
    is_satisfied setup p ∧ 
    p = 25 / 33 ∧
    setup.num_men * p = 1250 / 33 := by
  sorry


end satisfaction_theorem_l1916_191679


namespace olaf_car_collection_l1916_191690

/-- The number of toy cars in Olaf's collection after receiving gifts from his family -/
def total_cars (initial : ℕ) (dad : ℕ) (auntie : ℕ) (uncle : ℕ) : ℕ :=
  let mum := dad + 5
  let grandpa := 2 * uncle
  initial + dad + mum + auntie + uncle + grandpa

/-- Theorem stating the total number of cars in Olaf's collection -/
theorem olaf_car_collection : 
  total_cars 150 10 6 5 = 196 := by
  sorry

end olaf_car_collection_l1916_191690


namespace smallest_fraction_between_l1916_191602

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < 5 / 8 →
  q ≥ 13 ∧ (q = 13 → p = 8) :=
by sorry

end smallest_fraction_between_l1916_191602


namespace construct_axes_l1916_191645

/-- A parabola in a 2D plane -/
structure Parabola where
  f : ℝ → ℝ
  is_parabola : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  is_line : a ≠ 0 ∨ b ≠ 0

/-- Compass and straightedge construction operations -/
inductive Construction
  | point : Point → Construction
  | line : Point → Point → Construction
  | circle : Point → Point → Construction
  | intersect_lines : Line → Line → Construction
  | intersect_line_circle : Line → Point → Point → Construction
  | intersect_circles : Point → Point → Point → Point → Construction

/-- The theorem stating that coordinate axes can be constructed given a parabola -/
theorem construct_axes (p : Parabola) : 
  ∃ (origin : Point) (x_axis y_axis : Line) (constructions : List Construction),
    (∀ x : ℝ, p.f x = x^2) →
    (origin.x = 0 ∧ origin.y = 0) ∧
    (∀ x : ℝ, x_axis.a * x + x_axis.b * 0 + x_axis.c = 0) ∧
    (∀ y : ℝ, y_axis.a * 0 + y_axis.b * y + y_axis.c = 0) :=
sorry

end construct_axes_l1916_191645


namespace root_product_theorem_l1916_191657

-- Define the polynomial f(x)
def f (x : ℂ) : ℂ := x^6 + 2*x^3 + 1

-- Define the function h(x)
def h (x : ℂ) : ℂ := x^3 - 3*x

-- State the theorem
theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ y₆ : ℂ) 
  (hf₁ : f y₁ = 0) (hf₂ : f y₂ = 0) (hf₃ : f y₃ = 0)
  (hf₄ : f y₄ = 0) (hf₅ : f y₅ = 0) (hf₆ : f y₆ = 0) :
  (h y₁) * (h y₂) * (h y₃) * (h y₄) * (h y₅) * (h y₆) = 676 := by
sorry

end root_product_theorem_l1916_191657


namespace lcm_gcf_relation_l1916_191650

theorem lcm_gcf_relation (n : ℕ) (h1 : Nat.lcm n 16 = 52) (h2 : Nat.gcd n 16 = 8) : n = 26 := by
  sorry

end lcm_gcf_relation_l1916_191650


namespace tangent_at_negative_one_a_lower_bound_l1916_191623

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the derivative of g
def g' (x : ℝ) : ℝ := 2*x

-- Define the condition for the tangent line
def tangent_condition (x₁ a : ℝ) : Prop :=
  ∃ x₂, f' x₁ = g' x₂ ∧ f x₁ + f' x₁ * (x₂ - x₁) = g a x₂

-- Theorem 1: When x₁ = -1, a = 3
theorem tangent_at_negative_one :
  tangent_condition (-1) 3 :=
sorry

-- Theorem 2: For all valid x₁, a ≥ -1
theorem a_lower_bound :
  ∀ x₁ a : ℝ, tangent_condition x₁ a → a ≥ -1 :=
sorry

end tangent_at_negative_one_a_lower_bound_l1916_191623


namespace roots_sum_of_sixth_powers_l1916_191618

theorem roots_sum_of_sixth_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r ≠ s →
  r^6 + s^6 = 970 := by
sorry

end roots_sum_of_sixth_powers_l1916_191618


namespace range_of_a_l1916_191604

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ∈ Set.Icc (-1) 1 := by
  sorry

end range_of_a_l1916_191604


namespace right_triangle_angle_bisector_l1916_191609

theorem right_triangle_angle_bisector (DE DF : ℝ) (h_DE : DE = 13) (h_DF : DF = 5) : ∃ XY₁ : ℝ, XY₁ = (10 * Real.sqrt 6) / 17 := by
  sorry


end right_triangle_angle_bisector_l1916_191609


namespace sin_cos_15_deg_l1916_191699

theorem sin_cos_15_deg : 4 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 := by
  sorry

end sin_cos_15_deg_l1916_191699


namespace sixteen_factorial_digit_sum_l1916_191659

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem sixteen_factorial_digit_sum :
  ∃ (X Y : ℕ),
    X < 10 ∧ Y < 10 ∧
    factorial 16 = 2092200000000 + X * 100000000 + 208960000 + Y * 1000000 ∧
    X + Y = 7 := by
  sorry

end sixteen_factorial_digit_sum_l1916_191659


namespace cubic_equation_roots_l1916_191673

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 10*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  p + q = 45 := by
sorry

end cubic_equation_roots_l1916_191673


namespace perfect_score_l1916_191646

theorem perfect_score (perfect_score : ℕ) (h : 3 * perfect_score = 63) : perfect_score = 21 := by
  sorry

end perfect_score_l1916_191646


namespace probability_is_75_1024_l1916_191675

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction of movement -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of moving in any direction -/
def directionProbability : ℚ := 1/4

/-- The starting point -/
def start : Point := ⟨0, 0⟩

/-- The target point -/
def target : Point := ⟨3, 3⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 8

/-- Calculates the probability of reaching the target point from the start point
    in at most maxSteps steps -/
def probabilityToReachTarget (start : Point) (target : Point) (maxSteps : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem probability_is_75_1024 :
  probabilityToReachTarget start target maxSteps = 75/1024 := by
  sorry

end probability_is_75_1024_l1916_191675


namespace total_sodas_sold_l1916_191635

theorem total_sodas_sold (morning_sales afternoon_sales : ℕ) 
  (h1 : morning_sales = 77)
  (h2 : afternoon_sales = 19) :
  morning_sales + afternoon_sales = 96 := by
sorry

end total_sodas_sold_l1916_191635


namespace quadratic_inequality_solution_set_l1916_191633

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
sorry

end quadratic_inequality_solution_set_l1916_191633


namespace james_travel_distance_l1916_191667

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James' travel distance -/
theorem james_travel_distance :
  let speed : ℝ := 80.0
  let time : ℝ := 16.0
  distance speed time = 1280.0 := by
  sorry

end james_travel_distance_l1916_191667


namespace ones_divisible_by_27_l1916_191691

def ones_number (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem ones_divisible_by_27 :
  ∃ k : ℕ, ones_number 27 = 27 * k :=
sorry

end ones_divisible_by_27_l1916_191691


namespace sqrt_a_plus_one_real_iff_a_geq_neg_one_l1916_191669

theorem sqrt_a_plus_one_real_iff_a_geq_neg_one (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 1) ↔ a ≥ -1 := by
  sorry

end sqrt_a_plus_one_real_iff_a_geq_neg_one_l1916_191669


namespace apples_on_tree_l1916_191626

/-- Represents the number of apples in various states -/
structure AppleCount where
  onTree : ℕ
  onGround : ℕ
  eatenByDog : ℕ
  remaining : ℕ

/-- The theorem stating the number of apples on the tree -/
theorem apples_on_tree (a : AppleCount) 
  (h1 : a.onGround = 8)
  (h2 : a.eatenByDog = 3)
  (h3 : a.remaining = 10)
  (h4 : a.onGround = a.remaining + a.eatenByDog) :
  a.onTree = 5 := by
  sorry


end apples_on_tree_l1916_191626


namespace sequence_property_l1916_191605

/-- Given a sequence a_n and S_n where a_{n+1} = 3S_n for all n ≥ 1,
    prove that a_n can be arithmetic but not geometric -/
theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = 3 * S n) :
    (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
    (¬ ∃ r : ℝ, r ≠ 1 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n) :=
by sorry

end sequence_property_l1916_191605


namespace simplify_sqrt_seven_simplify_sqrt_fraction_simplify_sqrt_sum_simplify_sqrt_expression_l1916_191611

-- Problem 1
theorem simplify_sqrt_seven : 2 * Real.sqrt 7 - 6 * Real.sqrt 7 = -4 * Real.sqrt 7 := by sorry

-- Problem 2
theorem simplify_sqrt_fraction : Real.sqrt (2/3) / Real.sqrt (8/27) = 3/2 := by sorry

-- Problem 3
theorem simplify_sqrt_sum : Real.sqrt 18 + Real.sqrt 98 - Real.sqrt 27 = 10 * Real.sqrt 2 - 3 * Real.sqrt 3 := by sorry

-- Problem 4
theorem simplify_sqrt_expression : 
  (Real.sqrt 0.5 + Real.sqrt 6) - (Real.sqrt (1/8) - Real.sqrt 24) = Real.sqrt 2 / 4 + 3 * Real.sqrt 6 := by sorry

end simplify_sqrt_seven_simplify_sqrt_fraction_simplify_sqrt_sum_simplify_sqrt_expression_l1916_191611


namespace triangle_angle_measure_l1916_191660

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 6 →
  b = 2 →
  B = 45 * π / 180 →
  Real.tan A * Real.tan C > 1 →
  A + B + C = π →
  (a / Real.sin A = b / Real.sin B) →
  C = 75 * π / 180 := by
sorry

end triangle_angle_measure_l1916_191660


namespace absolute_value_and_quadratic_equivalence_l1916_191617

theorem absolute_value_and_quadratic_equivalence :
  ∀ (b c : ℝ),
    (∀ x : ℝ, |x - 8| = 3 ↔ x^2 + b*x + c = 0) ↔
    (b = -16 ∧ c = 55) :=
by sorry

end absolute_value_and_quadratic_equivalence_l1916_191617


namespace aaron_remaining_erasers_l1916_191643

def initial_erasers : ℕ := 225
def given_to_doris : ℕ := 75
def given_to_ethan : ℕ := 40
def given_to_fiona : ℕ := 50

theorem aaron_remaining_erasers :
  initial_erasers - (given_to_doris + given_to_ethan + given_to_fiona) = 60 := by
  sorry

end aaron_remaining_erasers_l1916_191643
