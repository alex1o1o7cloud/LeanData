import Mathlib

namespace pokemon_card_ratio_l3064_306412

theorem pokemon_card_ratio : ∀ (nicole cindy rex : ℕ),
  nicole = 400 →
  rex * 4 = 150 * 4 →
  2 * rex = nicole + cindy →
  cindy * 2 = nicole :=
by
  sorry

end pokemon_card_ratio_l3064_306412


namespace percentage_increase_l3064_306486

theorem percentage_increase (x y z : ℝ) : 
  y = 0.4 * z → x = 0.48 * z → (x - y) / y = 0.2 := by sorry

end percentage_increase_l3064_306486


namespace polygon_interior_exterior_angles_equality_l3064_306417

theorem polygon_interior_exterior_angles_equality (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end polygon_interior_exterior_angles_equality_l3064_306417


namespace bs_sequence_bounded_iff_f_null_l3064_306449

def is_bs_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = |a (n + 1) - a (n + 2)|

def is_bounded_sequence (a : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, |a n| ≤ M

def f (a : ℕ → ℝ) (n k : ℕ) : ℝ :=
  a n * a k * (a n - a k)

theorem bs_sequence_bounded_iff_f_null (a : ℕ → ℝ) :
  is_bs_sequence a →
  (is_bounded_sequence a ↔ ∀ n k : ℕ, f a n k = 0) :=
sorry

end bs_sequence_bounded_iff_f_null_l3064_306449


namespace tangent_line_intercept_product_minimum_l3064_306423

/-- The minimum product of x and y intercepts of a tangent line to the unit circle -/
theorem tangent_line_intercept_product_minimum : ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ (x y : ℝ), x^2 + y^2 = 1 → (x / a + y / b = 1 → False) ∨ (x / a + y / b ≠ 1)) →
  a * b ≥ 2 ∧ (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    (∀ (x y : ℝ), x^2 + y^2 = 1 → (x / a₀ + y / b₀ = 1 → False) ∨ (x / a₀ + y / b₀ ≠ 1)) ∧
    a₀ * b₀ = 2) := by
  sorry

end tangent_line_intercept_product_minimum_l3064_306423


namespace leilas_savings_leilas_savings_proof_l3064_306496

theorem leilas_savings : ℝ → Prop :=
  fun savings =>
    let makeup_fraction : ℝ := 3/5
    let sweater_fraction : ℝ := 1/3
    let sweater_cost : ℝ := 40
    let shoes_cost : ℝ := 30
    let remaining_fraction : ℝ := 1 - makeup_fraction - sweater_fraction
    
    (sweater_fraction * savings = sweater_cost) ∧
    (remaining_fraction * savings = shoes_cost) ∧
    (savings = 175)

-- The proof goes here
theorem leilas_savings_proof : ∃ (s : ℝ), leilas_savings s :=
sorry

end leilas_savings_leilas_savings_proof_l3064_306496


namespace inequality_proof_l3064_306478

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b) := by
  sorry

end inequality_proof_l3064_306478


namespace geometric_progression_solution_l3064_306497

/-- 
Given three terms in the form (10 + x), (40 + x), and (90 + x),
prove that x = 35 is the unique solution for which these terms form a geometric progression.
-/
theorem geometric_progression_solution : 
  ∃! x : ℝ, (∃ r : ℝ, r ≠ 0 ∧ (40 + x) = (10 + x) * r ∧ (90 + x) = (40 + x) * r) ∧ x = 35 := by
  sorry

end geometric_progression_solution_l3064_306497


namespace union_equality_implies_a_values_l3064_306489

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a^2}

-- State the theorem
theorem union_equality_implies_a_values (a : ℝ) :
  A a ∪ B a = A a → a = -1 ∨ a = 0 := by
  sorry

end union_equality_implies_a_values_l3064_306489


namespace hash_six_two_l3064_306408

-- Define the # operation
def hash (x y : ℝ) : ℝ := 4*x - 4*y

-- Theorem statement
theorem hash_six_two : hash 6 2 = 16 := by
  sorry

end hash_six_two_l3064_306408


namespace train_schedule_l3064_306445

theorem train_schedule (x y z : ℕ) : 
  x < 24 → y < 24 → z < 24 →
  (60 * y + z) - (60 * x + y) = 60 * z + x →
  x = 0 ∨ x = 12 := by
sorry

end train_schedule_l3064_306445


namespace accidental_multiplication_l3064_306480

theorem accidental_multiplication (x : ℕ) : x * 9 = 153 → x * 6 = 102 := by
  sorry

end accidental_multiplication_l3064_306480


namespace largest_perimeter_l3064_306483

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  h1 : side1 = 7
  h2 : side2 = 9
  h3 : side3 % 3 = 0

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of the triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter --/
theorem largest_perimeter :
  ∀ t : Triangle, is_valid_triangle t →
  ∃ max_t : Triangle, is_valid_triangle max_t ∧
  perimeter max_t = 31 ∧
  ∀ other_t : Triangle, is_valid_triangle other_t →
  perimeter other_t ≤ perimeter max_t :=
sorry

end largest_perimeter_l3064_306483


namespace equation_solution_l3064_306427

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x + 1) - 2 * (x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

#check equation_solution

end equation_solution_l3064_306427


namespace product_expansion_l3064_306482

theorem product_expansion (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end product_expansion_l3064_306482


namespace three_digit_number_proof_l3064_306491

theorem three_digit_number_proof :
  ∀ a b c : ℕ,
  (100 ≤ a * 100 + b * 10 + c) → 
  (a * 100 + b * 10 + c < 1000) →
  (a * (b + c) = 33) →
  (b * (a + c) = 40) →
  (a * 100 + b * 10 + c = 347) :=
by
  sorry

end three_digit_number_proof_l3064_306491


namespace bounded_region_area_l3064_306443

-- Define the equation
def equation (x y : ℝ) : Prop :=
  y^2 + 3*x*y + 60*|x| = 600

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation p.1 p.2}

-- Define the area function
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem bounded_region_area : area bounded_region = 800 := by sorry

end bounded_region_area_l3064_306443


namespace computer_pricing_l3064_306436

/-- Given a computer's cost and selling prices, prove the relationship between different profit percentages. -/
theorem computer_pricing (C : ℝ) : 
  (1.5 * C = 2678.57) → (1.4 * C = 2500.00) := by sorry

end computer_pricing_l3064_306436


namespace intersection_of_P_and_Q_l3064_306415

-- Define the set P
def P : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define the set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = 1/2 * x^2 - 1}

-- Theorem statement
theorem intersection_of_P_and_Q : P ∩ Q = Set.Ici 2 := by sorry

end intersection_of_P_and_Q_l3064_306415


namespace alexander_exhibition_problem_l3064_306468

/-- The number of pictures at each new gallery -/
def pictures_per_new_gallery (
  original_pictures : ℕ
  ) (new_galleries : ℕ
  ) (pencils_per_picture : ℕ
  ) (pencils_for_signing : ℕ
  ) (total_pencils : ℕ
  ) : ℕ :=
  let total_exhibitions := new_galleries + 1
  let pencils_for_drawing := total_pencils - (total_exhibitions * pencils_for_signing)
  let total_pictures := pencils_for_drawing / pencils_per_picture
  let new_gallery_pictures := total_pictures - original_pictures
  new_gallery_pictures / new_galleries

theorem alexander_exhibition_problem :
  pictures_per_new_gallery 9 5 4 2 88 = 2 := by
  sorry

end alexander_exhibition_problem_l3064_306468


namespace existence_of_special_sequence_l3064_306441

/-- A sequence of natural numbers -/
def NatSequence := ℕ → ℕ

/-- The sum of the first k terms of a sequence -/
def PartialSum (a : NatSequence) (k : ℕ) : ℕ :=
  (Finset.range k).sum (fun i => a i)

/-- Predicate for a sequence containing each natural number exactly once -/
def ContainsEachNatOnce (a : NatSequence) : Prop :=
  ∀ n : ℕ, ∃! k : ℕ, a k = n

/-- Predicate for the divisibility condition -/
def DivisibilityCondition (a : NatSequence) : Prop :=
  ∀ k : ℕ, k ∣ PartialSum a k

theorem existence_of_special_sequence :
  ∃ a : NatSequence, ContainsEachNatOnce a ∧ DivisibilityCondition a := by
  sorry

end existence_of_special_sequence_l3064_306441


namespace george_socks_theorem_l3064_306473

/-- The number of socks George initially had -/
def initial_socks : ℕ := 28

/-- The number of socks George threw away -/
def thrown_away : ℕ := 4

/-- The number of new socks George bought -/
def new_socks : ℕ := 36

/-- The total number of socks George would have after the transactions -/
def final_socks : ℕ := 60

/-- Theorem stating that the initial number of socks is correct -/
theorem george_socks_theorem : 
  initial_socks - thrown_away + new_socks = final_socks :=
by sorry

end george_socks_theorem_l3064_306473


namespace cube_volume_problem_l3064_306406

theorem cube_volume_problem (s : ℝ) : 
  (s + 2)^2 * (s - 3) = s^3 + 19 → s^3 = (4 + Real.sqrt 47)^3 :=
by sorry

end cube_volume_problem_l3064_306406


namespace boat_meeting_times_l3064_306428

/-- Represents the meeting time of two boats given their speeds and the river current. -/
def meeting_time (speed_A speed_C current distance : ℝ) : Set ℝ :=
  let effective_speed_A := speed_A + current
  let effective_speed_C_against := speed_C - current
  let effective_speed_C_with := speed_C + current
  let time_opposite := distance / (effective_speed_A + effective_speed_C_against)
  let time_same_direction := distance / (effective_speed_A - effective_speed_C_with)
  {time_opposite, time_same_direction}

/-- The theorem stating the meeting times of the boats under given conditions. -/
theorem boat_meeting_times :
  meeting_time 7 3 2 20 = {2, 5} := by
  sorry

end boat_meeting_times_l3064_306428


namespace fraction_sum_l3064_306440

theorem fraction_sum (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end fraction_sum_l3064_306440


namespace remainder_squared_pred_l3064_306451

theorem remainder_squared_pred (n : ℤ) (h : n % 5 = 3) : (n - 1)^2 % 5 = 4 := by
  sorry

end remainder_squared_pred_l3064_306451


namespace max_revenue_at_18_75_l3064_306448

/-- The revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The theorem stating that 18.75 maximizes the revenue function --/
theorem max_revenue_at_18_75 :
  ∀ p : ℝ, p ≤ 30 → R p ≤ R 18.75 := by
  sorry

#check max_revenue_at_18_75

end max_revenue_at_18_75_l3064_306448


namespace cube_sum_ge_squared_product_sum_l3064_306477

theorem cube_sum_ge_squared_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ a^2*b + b^2*c + c^2*a := by
  sorry

end cube_sum_ge_squared_product_sum_l3064_306477


namespace no_alpha_sequence_exists_l3064_306425

theorem no_alpha_sequence_exists : ¬∃ (α : ℝ) (a : ℕ → ℝ), 
  (0 < α ∧ α < 1) ∧ 
  (∀ n : ℕ, 0 < a n) ∧
  (∀ n : ℕ, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end no_alpha_sequence_exists_l3064_306425


namespace symmetric_points_l3064_306407

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0, 0, 0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Point P with coordinates (1, 3, 5) -/
def P : Point3D := ⟨1, 3, 5⟩

/-- Point P' with coordinates (-1, -3, -5) -/
def P' : Point3D := ⟨-1, -3, -5⟩

/-- Check if two points are symmetric with respect to the origin -/
def isSymmetricToOrigin (a b : Point3D) : Prop :=
  a.x + b.x = 0 ∧ a.y + b.y = 0 ∧ a.z + b.z = 0

/-- Theorem stating that P and P' are symmetric with respect to the origin -/
theorem symmetric_points : isSymmetricToOrigin P P' := by
  sorry

end symmetric_points_l3064_306407


namespace min_races_for_top_three_l3064_306433

/-- Represents a race track with a maximum capacity and a set of horses -/
structure RaceTrack where
  maxCapacity : Nat
  totalHorses : Nat

/-- Represents the minimum number of races needed to find the top n fastest horses -/
def minRacesForTopN (track : RaceTrack) (n : Nat) : Nat :=
  sorry

/-- Theorem stating that for a race track with 5 horse capacity and 25 total horses,
    the minimum number of races to find the top 3 fastest horses is 7 -/
theorem min_races_for_top_three (track : RaceTrack) :
  track.maxCapacity = 5 → track.totalHorses = 25 → minRacesForTopN track 3 = 7 :=
by sorry

end min_races_for_top_three_l3064_306433


namespace equivalent_operations_l3064_306413

theorem equivalent_operations (x : ℚ) : (x * (3/4)) / (3/5) = x * (5/4) := by
  sorry

end equivalent_operations_l3064_306413


namespace passenger_disembark_ways_l3064_306460

theorem passenger_disembark_ways (n : ℕ) (s : ℕ) (h1 : n = 10) (h2 : s = 5) :
  s^n = 5^10 := by
  sorry

end passenger_disembark_ways_l3064_306460


namespace sum_of_squares_l3064_306476

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 10)
  (eq2 : y^2 + 5*z = -10)
  (eq3 : z^2 + 7*x = -20) :
  x^2 + y^2 + z^2 = 20.75 := by sorry

end sum_of_squares_l3064_306476


namespace no_common_real_solution_l3064_306459

theorem no_common_real_solution :
  ¬∃ (x y : ℝ), x^2 + y^2 + 8 = 0 ∧ x^2 - 5*y + 20 = 0 := by
  sorry

end no_common_real_solution_l3064_306459


namespace hexagon_tileable_with_squares_l3064_306446

-- Define a hexagon type
structure Hexagon :=
  (A B C D E F : ℝ × ℝ)

-- Define the property of being convex
def is_convex (h : Hexagon) : Prop := sorry

-- Define the property of being inscribed
def is_inscribed (h : Hexagon) : Prop := sorry

-- Define perpendicularity of segments
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

-- Define equality of segments
def segments_equal (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

-- Define the property of being tileable with squares
def tileable_with_squares (h : Hexagon) : Prop := sorry

theorem hexagon_tileable_with_squares (h : Hexagon) 
  (convex : is_convex h)
  (inscribed : is_inscribed h)
  (perp_AD_CE : perpendicular h.A h.D h.C h.E)
  (eq_AD_CE : segments_equal h.A h.D h.C h.E)
  (perp_BE_AC : perpendicular h.B h.E h.A h.C)
  (eq_BE_AC : segments_equal h.B h.E h.A h.C)
  (perp_CF_EA : perpendicular h.C h.F h.E h.A)
  (eq_CF_EA : segments_equal h.C h.F h.E h.A) :
  tileable_with_squares h := by
  sorry

end hexagon_tileable_with_squares_l3064_306446


namespace pinterest_group_initial_pins_l3064_306421

/-- Calculates the initial number of pins in a Pinterest group --/
def initial_pins (
  daily_contribution : ℕ)  -- Average daily contribution per person
  (weekly_deletion : ℕ)    -- Weekly deletion rate per person
  (group_size : ℕ)         -- Number of people in the group
  (days : ℕ)               -- Number of days
  (final_pins : ℕ)         -- Total pins after the given period
  : ℕ :=
  final_pins - (daily_contribution * group_size * days) + (weekly_deletion * group_size * (days / 7))

theorem pinterest_group_initial_pins :
  initial_pins 10 5 20 30 6600 = 1000 := by
  sorry

end pinterest_group_initial_pins_l3064_306421


namespace sam_age_l3064_306492

theorem sam_age (drew_age : ℕ) (sam_age : ℕ) : 
  drew_age + sam_age = 54 →
  sam_age = drew_age / 2 →
  sam_age = 18 := by
sorry

end sam_age_l3064_306492


namespace ethanol_in_fuel_tank_l3064_306462

/-- Proves that the total amount of ethanol in a fuel tank is 30 gallons given specific conditions -/
theorem ethanol_in_fuel_tank (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) 
  (h1 : tank_capacity = 214)
  (h2 : fuel_a_volume = 106)
  (h3 : fuel_a_ethanol_percent = 0.12)
  (h4 : fuel_b_ethanol_percent = 0.16) :
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let total_ethanol := fuel_a_volume * fuel_a_ethanol_percent + fuel_b_volume * fuel_b_ethanol_percent
  total_ethanol = 30 := by
sorry


end ethanol_in_fuel_tank_l3064_306462


namespace problem_solution_l3064_306469

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≠ 0

def q (m : ℝ) : Prop := m > 0

-- Define the condition that either p or q is true, but not both
def condition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- Define the set of m values that satisfy the condition
def solution_set : Set ℝ := {m | condition m}

-- Theorem statement
theorem problem_solution : 
  solution_set = {m | m ∈ Set.Ioo (-2) 0 ∪ Set.Ioi 2} :=
sorry

end problem_solution_l3064_306469


namespace square_roots_combination_l3064_306465

theorem square_roots_combination : ∃ (a : ℚ), a * Real.sqrt 2 = Real.sqrt 8 ∧
  (∀ (b : ℚ), b * Real.sqrt 3 ≠ Real.sqrt 6) ∧
  (∀ (c : ℚ), c * Real.sqrt 2 ≠ Real.sqrt 12) ∧
  (∀ (d : ℚ), d * Real.sqrt 12 ≠ Real.sqrt 18) := by
  sorry

end square_roots_combination_l3064_306465


namespace complex_equation_sum_l3064_306426

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 + 2 * Complex.I) * (1 - Complex.I) → a + b = 4 := by
  sorry

end complex_equation_sum_l3064_306426


namespace postman_pete_miles_l3064_306474

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer where
  max_reading : Nat
  resets : Nat
  final_reading : Nat

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : Nat :=
  p.resets * (p.max_reading + 1) + p.final_reading

/-- Converts steps to miles given a steps-per-mile rate --/
def steps_to_miles (steps : Nat) (steps_per_mile : Nat) : Nat :=
  steps / steps_per_mile

/-- Theorem stating the total miles walked by Postman Pete --/
theorem postman_pete_miles :
  let p : Pedometer := { max_reading := 99999, resets := 50, final_reading := 25000 }
  let total_miles := steps_to_miles (total_steps p) 1500
  total_miles = 3350 := by
  sorry

end postman_pete_miles_l3064_306474


namespace mat_coverage_fraction_l3064_306430

/-- The fraction of a square tabletop covered by a circular mat -/
theorem mat_coverage_fraction (mat_diameter : ℝ) (table_side : ℝ) 
  (h1 : mat_diameter = 18) (h2 : table_side = 24) : 
  (π * (mat_diameter / 2)^2) / (table_side^2) = π / 7 := by
  sorry

end mat_coverage_fraction_l3064_306430


namespace display_rows_l3064_306487

/-- Represents the number of cans in a row given its position from the top -/
def cans_in_row (n : ℕ) : ℕ := 3 * n - 2

/-- Calculates the total number of cans in the first n rows -/
def total_cans (n : ℕ) : ℕ := n * (3 * n - 1) / 2

theorem display_rows : ∃ n : ℕ, total_cans n = 225 ∧ n = 16 := by
  sorry

end display_rows_l3064_306487


namespace discontinuity_coincidence_l3064_306410

-- Define the functions f, g, and h
variable (f g h : ℝ → ℝ)

-- Define the conditions
variable (hf_diff : Differentiable ℝ f)
variable (hg_mono : Monotone g)
variable (hh_mono : Monotone h)
variable (hf_deriv : ∀ x, deriv f x = f x + g x + h x)

-- State the theorem
theorem discontinuity_coincidence :
  ∀ x : ℝ, ¬(ContinuousAt g x) ↔ ¬(ContinuousAt h x) := by
  sorry

end discontinuity_coincidence_l3064_306410


namespace triple_characterization_l3064_306472

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ (a^n + 203) % (a^m + 1) = 0

def solution_set (k m : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 4*k+1), (2, 3, 6*k+2), (2, 4, 8*k+8), (2, 6, 12*k+9),
   (3, 2, 4*k+3), (4, 2, 4*k+4), (5, 2, 4*k+1), (8, 2, 4*k+3),
   (10, 2, 4*k+2), (203, m, (2*k+1)*m+1)}

theorem triple_characterization :
  ∀ a m n : ℕ, is_valid_triple a m n ↔ ∃ k : ℕ, (a, m, n) ∈ solution_set k m :=
sorry

end triple_characterization_l3064_306472


namespace cube_sum_minus_product_l3064_306420

theorem cube_sum_minus_product (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : a * b + a * c + b * c = 40) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1575 := by
sorry

end cube_sum_minus_product_l3064_306420


namespace shooter_probability_l3064_306402

theorem shooter_probability (p_a p_b : ℝ) 
  (h_p_a : p_a = 3/4)
  (h_p_b : p_b = 2/3)
  (h_p_a_range : 0 ≤ p_a ∧ p_a ≤ 1)
  (h_p_b_range : 0 ≤ p_b ∧ p_b ≤ 1) :
  p_a * (1 - p_b) * (1 - p_b) + (1 - p_a) * p_b * (1 - p_b) + (1 - p_a) * (1 - p_b) * p_b = 7/36 := by
  sorry

end shooter_probability_l3064_306402


namespace trig_inequality_l3064_306438

theorem trig_inequality (x y : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (h_eq : Real.sin x = x * Real.cos y) : 
  x/2 < y ∧ y < x :=
by sorry

end trig_inequality_l3064_306438


namespace point_in_third_quadrant_m_value_l3064_306419

-- Define the point P
def P (m : ℤ) : ℝ × ℝ := (2 - m, m - 4)

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_third_quadrant_m_value :
  ∀ m : ℤ, in_third_quadrant (P m) → m = 3 :=
by sorry

end point_in_third_quadrant_m_value_l3064_306419


namespace power_of_power_l3064_306463

theorem power_of_power (x : ℝ) : (x^5)^2 = x^10 := by
  sorry

end power_of_power_l3064_306463


namespace julia_watch_collection_l3064_306490

theorem julia_watch_collection (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) :
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = (silver_watches + bronze_watches + gold_watches) / 10 →
  silver_watches + bronze_watches + gold_watches = 88 :=
by sorry

end julia_watch_collection_l3064_306490


namespace xor_inequality_iff_even_l3064_306458

-- Define bitwise XOR operation
def bitwise_xor (a b : ℕ) : ℕ := sorry

-- Define the property that needs to be proven
def xor_inequality_property (a : ℕ) : Prop :=
  ∀ x y : ℕ, x > y → y ≥ 0 → bitwise_xor x (a * x) ≠ bitwise_xor y (a * y)

-- Theorem statement
theorem xor_inequality_iff_even (a : ℕ) :
  a > 0 → (xor_inequality_property a ↔ Even a) :=
sorry

end xor_inequality_iff_even_l3064_306458


namespace quadrilateral_side_length_l3064_306439

-- Define the quadrilateral AMOL
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the lengths of sides
def AM (q : Quadrilateral) : ℝ := 10
def MO (q : Quadrilateral) : ℝ := 11
def OL (q : Quadrilateral) : ℝ := 12

-- Define the condition for perpendicular bisectors
def perpendicular_bisectors_condition (q : Quadrilateral) : Prop :=
  ∃ E : ℝ × ℝ, 
    E = ((q.A.1 + q.C.1) / 2, (q.A.2 + q.C.2) / 2) ∧
    (E.1 - q.A.1) * (q.B.1 - q.A.1) + (E.2 - q.A.2) * (q.B.2 - q.A.2) = 0 ∧
    (E.1 - q.C.1) * (q.D.1 - q.C.1) + (E.2 - q.C.2) * (q.D.2 - q.C.2) = 0

-- State the theorem
theorem quadrilateral_side_length (q : Quadrilateral) :
  AM q = 10 ∧ MO q = 11 ∧ OL q = 12 ∧ perpendicular_bisectors_condition q →
  Real.sqrt ((q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2) = Real.sqrt 77 := by
  sorry

end quadrilateral_side_length_l3064_306439


namespace floor_equation_solutions_l3064_306437

theorem floor_equation_solutions (n : ℤ) : 
  (⌊n^2 / 3⌋ : ℤ) - (⌊n / 3⌋ : ℤ)^2 = 3 ↔ n = -8 ∨ n = -4 := by
  sorry

end floor_equation_solutions_l3064_306437


namespace polygon_sides_l3064_306435

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1440 → n = 10 := by
  sorry

end polygon_sides_l3064_306435


namespace power_equation_solution_l3064_306414

theorem power_equation_solution :
  ∃ (x : ℕ), (12 : ℝ)^x * 6^4 / 432 = 5184 ∧ x = 9 := by
  sorry

end power_equation_solution_l3064_306414


namespace abc_sum_product_range_l3064_306403

theorem abc_sum_product_range (a b c : ℝ) (h : a + b + c = 3) :
  ∃ S : Set ℝ, S = Set.Iic 3 ∧ ∀ x : ℝ, x ∈ S ↔ ∃ a' b' c' : ℝ, a' + b' + c' = 3 ∧ a' * b' + a' * c' + b' * c' = x :=
sorry

end abc_sum_product_range_l3064_306403


namespace unique_solution_exists_l3064_306464

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

-- Theorem statement
theorem unique_solution_exists :
  ∃! y : ℝ, otimes 2 y = 20 := by sorry

end unique_solution_exists_l3064_306464


namespace f_two_roots_implies_a_gt_three_l3064_306455

/-- The function f(x) = x³ - ax² + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- f has two distinct positive roots -/
def has_two_distinct_positive_roots (a : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0

theorem f_two_roots_implies_a_gt_three (a : ℝ) :
  has_two_distinct_positive_roots a → a > 3 := by
  sorry

end f_two_roots_implies_a_gt_three_l3064_306455


namespace quadratic_equation_properties_l3064_306444

theorem quadratic_equation_properties (a b c : ℝ) (ha : a ≠ 0) :
  -- Statement 1
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  -- Statement 2
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + c = 0 ∧ a*y^2 + c = 0 →
    ∃ u v : ℝ, u ≠ v ∧ a*u^2 + b*u + c = 0 ∧ a*v^2 + b*v + c = 0) ∧
  -- Statement 4
  ∃ m n : ℝ, m ≠ n ∧ a*m^2 + b*m + c = a*n^2 + b*n + c :=
by sorry

end quadratic_equation_properties_l3064_306444


namespace symmetric_points_range_l3064_306450

open Real

theorem symmetric_points_range (g h : ℝ → ℝ) (a : ℝ) :
  (∀ x, 1/ℯ ≤ x → x ≤ ℯ → g x = a - x^2) →
  (∀ x, h x = 2 * log x) →
  (∃ x, 1/ℯ ≤ x ∧ x ≤ ℯ ∧ g x = -h x) →
  1 ≤ a ∧ a ≤ ℯ^2 - 2 := by
  sorry

end symmetric_points_range_l3064_306450


namespace right_triangle_segment_ratio_l3064_306400

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  r * s = a^2 →      -- Geometric mean theorem for r
  r * s = b^2 →      -- Geometric mean theorem for s
  c = r + s →        -- c is divided into r and s
  a / b = 2 / 5 →    -- Given ratio of a to b
  r / s = 4 / 25 :=  -- Conclusion: ratio of r to s
by sorry

end right_triangle_segment_ratio_l3064_306400


namespace xiaohui_wins_l3064_306411

/-- Represents a student with their scores -/
structure Student where
  name : String
  mandarin : ℕ
  sports : ℕ
  tourism : ℕ

/-- Calculates the weighted score for a student given the weights -/
def weightedScore (s : Student) (w1 w2 w3 : ℕ) : ℚ :=
  (s.mandarin * w1 + s.sports * w2 + s.tourism * w3 : ℚ) / (w1 + w2 + w3 : ℚ)

/-- The theorem stating that Xiaohui wins -/
theorem xiaohui_wins : 
  let xiaocong : Student := ⟨"Xiaocong", 80, 90, 72⟩
  let xiaohui : Student := ⟨"Xiaohui", 90, 80, 70⟩
  weightedScore xiaohui 4 3 3 > weightedScore xiaocong 4 3 3 := by
  sorry


end xiaohui_wins_l3064_306411


namespace sum_of_squares_and_cubes_l3064_306461

theorem sum_of_squares_and_cubes (a b : ℤ) (h : ∃ k : ℤ, a^2 - 4*b = k^2) :
  (∃ x y : ℤ, a^2 - 2*b = x^2 + y^2) ∧
  (∃ u v : ℤ, 3*a*b - a^3 = u^3 + v^3) := by
  sorry

end sum_of_squares_and_cubes_l3064_306461


namespace more_than_three_solutions_l3064_306454

/-- Represents a trapezoid with bases b₁ and b₂, and height h -/
structure Trapezoid where
  b₁ : ℕ
  b₂ : ℕ
  h : ℕ

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℕ :=
  (t.b₁ + t.b₂) * t.h / 2

/-- Predicate for valid trapezoid solutions -/
def isValidSolution (m n : ℕ) : Prop :=
  m + n = 6 ∧
  10 ∣ (10 * m) ∧
  10 ∣ (10 * n) ∧
  area { b₁ := 10 * m, b₂ := 10 * n, h := 60 } = 1800

theorem more_than_three_solutions :
  ∃ (S : Finset (ℕ × ℕ)), S.card > 3 ∧ ∀ (p : ℕ × ℕ), p ∈ S → isValidSolution p.1 p.2 :=
sorry

end more_than_three_solutions_l3064_306454


namespace range_of_a_l3064_306493

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x + 3 ≤ a^2 - 2*a - 1)) → 
  (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l3064_306493


namespace triangle_areas_equal_l3064_306479

theorem triangle_areas_equal :
  let a : ℝ := 24
  let b : ℝ := 24
  let c : ℝ := 34
  let right_triangle_area := (1/2) * a * b
  let s := (a + b + c) / 2
  let general_triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  right_triangle_area = general_triangle_area := by
  sorry

end triangle_areas_equal_l3064_306479


namespace sixteen_is_sixtyfour_percent_of_twentyfive_l3064_306434

theorem sixteen_is_sixtyfour_percent_of_twentyfive :
  ∀ x : ℚ, (16 : ℚ) = 64 / 100 * x → x = 25 := by
  sorry

end sixteen_is_sixtyfour_percent_of_twentyfive_l3064_306434


namespace square_side_ratio_l3064_306418

theorem square_side_ratio (area_ratio : ℚ) : 
  area_ratio = 50 / 98 → 
  ∃ (a b c : ℕ), (a : ℚ) * Real.sqrt (b : ℚ) / (c : ℚ) = Real.sqrt (area_ratio) ∧ 
                  a = 5 ∧ b = 2 ∧ c = 7 := by
  sorry

end square_side_ratio_l3064_306418


namespace computer_contract_probability_l3064_306484

theorem computer_contract_probability (p_not_software : ℝ) (p_at_least_one : ℝ) (p_both : ℝ) 
  (h1 : p_not_software = 3/5)
  (h2 : p_at_least_one = 5/6)
  (h3 : p_both = 0.31666666666666654) :
  let p_software := 1 - p_not_software
  let p_hardware := p_at_least_one + p_both - p_software
  p_hardware = 0.75 := by
sorry

end computer_contract_probability_l3064_306484


namespace chris_soccer_cards_l3064_306495

/-- Chris has some soccer cards. His friend, Charlie, has 32 cards. 
    Chris has 14 fewer cards than Charlie. -/
theorem chris_soccer_cards 
  (charlie_cards : ℕ) 
  (chris_fewer : ℕ)
  (h1 : charlie_cards = 32)
  (h2 : chris_fewer = 14) :
  charlie_cards - chris_fewer = 18 := by
  sorry

end chris_soccer_cards_l3064_306495


namespace climb_nine_flights_l3064_306499

/-- Calculates the number of steps climbed given the number of flights, height per flight, and height per step. -/
def steps_climbed (flights : ℕ) (feet_per_flight : ℕ) (inches_per_step : ℕ) : ℕ :=
  (flights * feet_per_flight * 12) / inches_per_step

/-- Proves that climbing 9 flights of 10-foot stairs with 18-inch steps results in 60 steps. -/
theorem climb_nine_flights : steps_climbed 9 10 18 = 60 := by
  sorry

end climb_nine_flights_l3064_306499


namespace triangle_sine_sum_inequality_l3064_306401

theorem triangle_sine_sum_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_sine_sum_inequality_l3064_306401


namespace same_color_sock_probability_l3064_306452

def total_socks : ℕ := 30
def blue_socks : ℕ := 16
def green_socks : ℕ := 10
def red_socks : ℕ := 4

theorem same_color_sock_probability :
  let total_combinations := total_socks.choose 2
  let blue_combinations := blue_socks.choose 2
  let green_combinations := green_socks.choose 2
  let red_combinations := red_socks.choose 2
  let same_color_combinations := blue_combinations + green_combinations + red_combinations
  (same_color_combinations : ℚ) / total_combinations = 19 / 45 := by
  sorry

end same_color_sock_probability_l3064_306452


namespace unique_four_digit_number_l3064_306405

theorem unique_four_digit_number : ∃! n : ℕ, 
  1000 ≤ n ∧ n < 10000 ∧ 
  n % 131 = 112 ∧ 
  n % 132 = 98 :=
by sorry

end unique_four_digit_number_l3064_306405


namespace boris_books_l3064_306429

theorem boris_books (boris_initial : ℕ) (cameron_initial : ℕ) : 
  cameron_initial = 30 →
  (3 * boris_initial / 4 : ℚ) + (2 * cameron_initial / 3 : ℚ) = 38 →
  boris_initial = 24 :=
by sorry

end boris_books_l3064_306429


namespace select_shoes_result_l3064_306422

/-- The number of ways to select 4 shoes from 5 pairs, with exactly one pair included -/
def select_shoes (n : ℕ) : ℕ :=
  let total_pairs := 5
  let pairs_to_choose := 1
  let single_shoes := n - 2 * pairs_to_choose
  let remaining_pairs := total_pairs - pairs_to_choose
  (total_pairs.choose pairs_to_choose) *
  (remaining_pairs.choose single_shoes) *
  (2^pairs_to_choose * 2^single_shoes)

theorem select_shoes_result : select_shoes 4 = 120 := by
  sorry

end select_shoes_result_l3064_306422


namespace function_minimum_condition_l3064_306471

/-- Given a function f(x) = e^x + ae^(-x) where a is a constant, 
    if f(x) ≥ f(0) for all x in [-1, 1], then a = 1 -/
theorem function_minimum_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, (Real.exp x + a * Real.exp (-x)) ≥ (1 + a)) →
  a = 1 := by
  sorry

end function_minimum_condition_l3064_306471


namespace temperature_84_latest_time_l3064_306431

/-- Temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The time when the temperature is 84 degrees -/
def temperature_84 (t : ℝ) : Prop := temperature t = 84

/-- The latest time when the temperature is 84 degrees -/
def latest_time_84 : ℝ := 11

theorem temperature_84_latest_time :
  temperature_84 latest_time_84 ∧
  ∀ t, t > latest_time_84 → ¬(temperature_84 t) :=
sorry

end temperature_84_latest_time_l3064_306431


namespace max_plus_min_of_f_l3064_306447

noncomputable def f (x : ℝ) : ℝ := 1 + x / (x^2 + 1)

theorem max_plus_min_of_f : 
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
                (∀ x, N ≤ f x) ∧ (∃ x, f x = N) ∧ 
                (M + N = 2) :=
sorry

end max_plus_min_of_f_l3064_306447


namespace percentage_problem_l3064_306453

theorem percentage_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end percentage_problem_l3064_306453


namespace function_upper_bound_l3064_306442

theorem function_upper_bound (x : ℝ) (h : x ≥ 1) : (1 + Real.log x) / x ≤ 1 := by
  sorry

end function_upper_bound_l3064_306442


namespace cyclic_matrix_determinant_zero_l3064_306456

theorem cyclic_matrix_determinant_zero (p q r : ℝ) (a b c d : ℝ) : 
  (a^4 + p*a^2 + q*a + r = 0) → 
  (b^4 + p*b^2 + q*b + r = 0) → 
  (c^4 + p*c^2 + q*c + r = 0) → 
  (d^4 + p*d^2 + q*d + r = 0) → 
  Matrix.det 
    ![![a, b, c, d],
      ![b, c, d, a],
      ![c, d, a, b],
      ![d, a, b, c]] = 0 := by
  sorry

end cyclic_matrix_determinant_zero_l3064_306456


namespace greatest_prime_factor_sum_even_products_l3064_306485

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

def even_product (n : ℕ) : ℕ :=
  double_factorial (2 * (n / 2))

theorem greatest_prime_factor_sum_even_products :
  ∃ (p : ℕ), p.Prime ∧ p = 23 ∧
  ∀ (q : ℕ), q.Prime → q ∣ (even_product 22 + even_product 20) → q ≤ p :=
by sorry

end greatest_prime_factor_sum_even_products_l3064_306485


namespace rachel_homework_l3064_306481

/-- Rachel's homework problem -/
theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 5 → reading_pages = 2 → math_pages + reading_pages = 7 := by
  sorry

end rachel_homework_l3064_306481


namespace grocery_theorem_l3064_306424

def grocery_problem (initial_budget bread_cost candy_cost : ℚ) : ℚ :=
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let turkey_cost := remaining_after_bread_candy / 3
  remaining_after_bread_candy - turkey_cost

theorem grocery_theorem :
  grocery_problem 32 3 2 = 18 := by
  sorry

end grocery_theorem_l3064_306424


namespace marla_horse_purchase_time_l3064_306409

/-- Represents the exchange rates and Marla's scavenging abilities in the post-apocalyptic wasteland -/
structure WastelandEconomy where
  lizard_to_caps : ℕ
  lizards_to_water : ℕ
  water_to_lizards : ℕ
  horse_to_water : ℕ
  daily_scavenge : ℕ
  nightly_cost : ℕ

/-- Calculates the number of days it takes Marla to collect enough bottle caps to buy a horse -/
def days_to_buy_horse (e : WastelandEconomy) : ℕ :=
  let caps_per_lizard := e.lizard_to_caps
  let water_per_horse := e.horse_to_water
  let lizards_per_horse := (water_per_horse * e.water_to_lizards) / e.lizards_to_water
  let caps_per_horse := lizards_per_horse * caps_per_lizard
  let daily_savings := e.daily_scavenge - e.nightly_cost
  caps_per_horse / daily_savings

/-- Theorem stating that it takes Marla 24 days to collect enough bottle caps to buy a horse -/
theorem marla_horse_purchase_time :
  days_to_buy_horse {
    lizard_to_caps := 8,
    lizards_to_water := 3,
    water_to_lizards := 5,
    horse_to_water := 80,
    daily_scavenge := 20,
    nightly_cost := 4
  } = 24 := by
  sorry

end marla_horse_purchase_time_l3064_306409


namespace least_number_divisible_by_five_primes_l3064_306475

theorem least_number_divisible_by_five_primes : 
  ∃ (n : ℕ), n > 0 ∧ (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ),
    Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
    q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
    q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
    q₄ ≠ q₅ ∧
    m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → m ≥ n) ∧
  n = 2310 := by
sorry

end least_number_divisible_by_five_primes_l3064_306475


namespace container_initial_percentage_l3064_306416

theorem container_initial_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 60 →
  added_water = 27 →
  final_fraction = 3/4 →
  (capacity * final_fraction - added_water) / capacity * 100 = 30 := by
  sorry

end container_initial_percentage_l3064_306416


namespace mrs_snyder_income_l3064_306488

/-- Mrs. Snyder's previous monthly income --/
def previous_income : ℝ := 1000

/-- Mrs. Snyder's salary increase --/
def salary_increase : ℝ := 600

/-- Percentage of previous income spent on rent and utilities --/
def previous_percentage : ℝ := 0.40

/-- Percentage of new income spent on rent and utilities --/
def new_percentage : ℝ := 0.25

theorem mrs_snyder_income :
  previous_income * previous_percentage = 
  (previous_income + salary_increase) * new_percentage := by
  sorry

#check mrs_snyder_income

end mrs_snyder_income_l3064_306488


namespace sin_65pi_over_6_l3064_306432

theorem sin_65pi_over_6 : Real.sin (65 * π / 6) = 1 / 2 := by
  sorry

end sin_65pi_over_6_l3064_306432


namespace roots_of_equation_l3064_306467

theorem roots_of_equation (x : ℝ) : (x - 5)^2 = 2*(x - 5) ↔ x = 5 ∨ x = 7 := by
  sorry

end roots_of_equation_l3064_306467


namespace sine_function_value_l3064_306470

/-- Proves that f(π/4) = -4/5 given specific conditions on φ and ω -/
theorem sine_function_value (φ ω : Real) (h1 : (-4 : Real) / 5 = Real.cos φ)
    (h2 : (3 : Real) / 5 = Real.sin φ) (h3 : ω > 0) 
    (h4 : π / (2 * ω) = π / 2) : 
  Real.sin (ω * (π / 4) + φ) = -4/5 := by
  sorry

end sine_function_value_l3064_306470


namespace pet_food_price_l3064_306494

theorem pet_food_price (regular_discount_min regular_discount_max additional_discount lowest_price : Real) 
  (h1 : 0.1 ≤ regular_discount_min ∧ regular_discount_min ≤ regular_discount_max ∧ regular_discount_max ≤ 0.3)
  (h2 : additional_discount = 0.2)
  (h3 : lowest_price = 25.2)
  : ∃ (original_price : Real),
    original_price * (1 - regular_discount_max) * (1 - additional_discount) = lowest_price ∧
    original_price = 45 :=
by
  sorry

end pet_food_price_l3064_306494


namespace logo_enlargement_l3064_306457

/-- Calculates the height of a proportionally enlarged logo -/
def enlargedLogoHeight (originalWidth originalHeight newWidth : ℚ) : ℚ :=
  (newWidth / originalWidth) * originalHeight

/-- Theorem: The enlarged logo height is 8 inches -/
theorem logo_enlargement (originalWidth originalHeight newWidth : ℚ) 
  (h1 : originalWidth = 3)
  (h2 : originalHeight = 2)
  (h3 : newWidth = 12) :
  enlargedLogoHeight originalWidth originalHeight newWidth = 8 := by
  sorry

end logo_enlargement_l3064_306457


namespace line_point_k_value_l3064_306498

/-- A line contains the points (2, -1), (10, k), and (25, 4). The value of k is 17/23. -/
theorem line_point_k_value (k : ℚ) : 
  (∃ (line : ℝ → ℝ), 
    line 2 = -1 ∧ 
    line 10 = k ∧ 
    line 25 = 4) → 
  k = 17/23 := by
sorry

end line_point_k_value_l3064_306498


namespace tetrahedron_edges_form_two_triangles_l3064_306404

-- Define a tetrahedron as a structure with 6 edges
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

-- Define a predicate for valid triangles
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem tetrahedron_edges_form_two_triangles (t : Tetrahedron) :
  ∃ (i₁ i₂ i₃ i₄ i₅ i₆ : Fin 6),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧
    i₅ ≠ i₆ ∧
    is_valid_triangle (t.edges i₁) (t.edges i₂) (t.edges i₃) ∧
    is_valid_triangle (t.edges i₄) (t.edges i₅) (t.edges i₆) :=
by sorry

end tetrahedron_edges_form_two_triangles_l3064_306404


namespace dot_product_when_x_negative_one_parallel_vectors_when_x_eight_l3064_306466

def a : ℝ × ℝ := (4, 7)
def b (x : ℝ) : ℝ × ℝ := (x, x + 6)

theorem dot_product_when_x_negative_one :
  (a.1 * (b (-1)).1 + a.2 * (b (-1)).2) = 31 := by sorry

theorem parallel_vectors_when_x_eight :
  (a.1 / (b 8).1 = a.2 / (b 8).2) := by sorry

end dot_product_when_x_negative_one_parallel_vectors_when_x_eight_l3064_306466
