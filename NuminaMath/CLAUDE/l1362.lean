import Mathlib

namespace complex_number_equality_l1362_136285

theorem complex_number_equality : (1 + Complex.I)^10 / (1 - Complex.I) = -16 + 16 * Complex.I := by
  sorry

end complex_number_equality_l1362_136285


namespace inequality_solution_and_abc_inequality_l1362_136244

theorem inequality_solution_and_abc_inequality :
  let solution_set := {x : ℝ | -1/2 < x ∧ x < 7/2}
  let p : ℝ := -3
  let q : ℝ := -7/4
  (∀ x, x ∈ solution_set ↔ |2*x - 3| < 4) →
  (∀ x, x ∈ solution_set ↔ x^2 + p*x + q < 0) →
  ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c →
    a + b + c = 2*p - 4*q →
    Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 :=
by sorry

end inequality_solution_and_abc_inequality_l1362_136244


namespace triangle_area_l1362_136259

theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = Real.sqrt 2 → 
  A = π / 4 → 
  B = π / 3 → 
  C = π - A - B →
  S = (1 / 2) * a * b * Real.sin C →
  S = (3 + Real.sqrt 3) / 4 := by
  sorry

end triangle_area_l1362_136259


namespace room_length_proof_l1362_136262

/-- Given the width, total cost, and rate of paving a room's floor, 
    prove that the length of the room is 5.5 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) 
    (h1 : width = 3.75)
    (h2 : total_cost = 20625)
    (h3 : paving_rate = 1000) : 
  (total_cost / paving_rate) / width = 5.5 := by
  sorry

#check room_length_proof

end room_length_proof_l1362_136262


namespace sufficient_but_not_necessary_l1362_136200

theorem sufficient_but_not_necessary :
  (∃ p q : Prop, (p ∨ q = False) → (¬p = True)) ∧
  (∃ p q : Prop, (¬p = True) ∧ ¬(p ∨ q = False)) := by
  sorry

end sufficient_but_not_necessary_l1362_136200


namespace cells_reach_1540_in_9_hours_l1362_136209

/-- The number of cells after n hours -/
def cell_count (n : ℕ) : ℕ :=
  3 * 2^(n-1) + 4

/-- The theorem stating that it takes 9 hours to reach 1540 cells -/
theorem cells_reach_1540_in_9_hours :
  cell_count 9 = 1540 ∧
  ∀ k : ℕ, k < 9 → cell_count k < 1540 :=
by sorry

end cells_reach_1540_in_9_hours_l1362_136209


namespace optimal_sampling_theorem_l1362_136277

/-- Represents the different blood types --/
inductive BloodType
| O
| A
| B
| AB

/-- Represents the available sampling methods --/
inductive SamplingMethod
| Random
| Systematic
| Stratified

structure School :=
  (total_students : Nat)
  (blood_type_counts : BloodType → Nat)
  (sample_size : Nat)
  (soccer_team_size : Nat)
  (soccer_sample_size : Nat)

def optimal_sampling_method (school : School) (is_blood_type_study : Bool) : SamplingMethod :=
  if is_blood_type_study then
    SamplingMethod.Stratified
  else
    SamplingMethod.Random

theorem optimal_sampling_theorem (school : School) :
  (school.total_students = 500) →
  (school.blood_type_counts BloodType.O = 200) →
  (school.blood_type_counts BloodType.A = 125) →
  (school.blood_type_counts BloodType.B = 125) →
  (school.blood_type_counts BloodType.AB = 50) →
  (school.sample_size = 20) →
  (school.soccer_team_size = 11) →
  (school.soccer_sample_size = 2) →
  (optimal_sampling_method school true = SamplingMethod.Stratified) ∧
  (optimal_sampling_method school false = SamplingMethod.Random) :=
by
  sorry

end optimal_sampling_theorem_l1362_136277


namespace opera_house_earnings_l1362_136253

/-- Opera house earnings calculation -/
theorem opera_house_earnings : 
  let total_rows : ℕ := 150
  let section_a_rows : ℕ := 50
  let section_b_rows : ℕ := 60
  let section_c_rows : ℕ := 40
  let seats_per_row : ℕ := 10
  let section_a_price : ℕ := 20
  let section_b_price : ℕ := 15
  let section_c_price : ℕ := 10
  let convenience_fee : ℕ := 3
  let section_a_occupancy : ℚ := 9/10
  let section_b_occupancy : ℚ := 3/4
  let section_c_occupancy : ℚ := 7/10

  let section_a_earnings := (section_a_price + convenience_fee) * (section_a_rows * seats_per_row : ℕ) * section_a_occupancy
  let section_b_earnings := (section_b_price + convenience_fee) * (section_b_rows * seats_per_row : ℕ) * section_b_occupancy
  let section_c_earnings := (section_c_price + convenience_fee) * (section_c_rows * seats_per_row : ℕ) * section_c_occupancy

  let total_earnings := section_a_earnings + section_b_earnings + section_c_earnings

  total_earnings = 22090 := by sorry

end opera_house_earnings_l1362_136253


namespace f_composition_nonnegative_iff_a_geq_three_l1362_136281

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 1

theorem f_composition_nonnegative_iff_a_geq_three (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ 3 := by
  sorry

end f_composition_nonnegative_iff_a_geq_three_l1362_136281


namespace wheel_probability_l1362_136220

theorem wheel_probability (p_D p_E p_FG : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_D + p_E + p_FG = 1 → p_FG = 5/12 := by
  sorry

end wheel_probability_l1362_136220


namespace braden_winnings_l1362_136208

/-- Calculates the total amount in Braden's money box after winning a bet -/
def total_amount_after_bet (initial_amount : ℕ) (bet_multiplier : ℕ) : ℕ :=
  initial_amount + bet_multiplier * initial_amount

/-- Theorem stating that given the initial conditions, Braden's final amount is $1200 -/
theorem braden_winnings :
  let initial_amount := 400
  let bet_multiplier := 2
  total_amount_after_bet initial_amount bet_multiplier = 1200 := by
  sorry

end braden_winnings_l1362_136208


namespace parallelepiped_with_surroundings_volume_l1362_136254

/-- The volume of a set consisting of a rectangular parallelepiped and its surrounding elements -/
theorem parallelepiped_with_surroundings_volume 
  (l w h : ℝ) 
  (hl : l = 2) 
  (hw : w = 3) 
  (hh : h = 6) 
  (r : ℝ) 
  (hr : r = 1) : 
  (l * w * h) + 
  (2 * (r * w * h + r * l * h + r * l * w)) + 
  (π * r^2 * (l + w + h)) + 
  (2 * π * r^3) = 
  108 + (41/3) * π := by sorry

end parallelepiped_with_surroundings_volume_l1362_136254


namespace group_payment_l1362_136235

/-- Calculates the total amount paid by a group of moviegoers -/
def total_amount_paid (adult_price child_price : ℚ) (total_people adults : ℕ) : ℚ :=
  adult_price * adults + child_price * (total_people - adults)

/-- Theorem: The group paid $54.50 in total -/
theorem group_payment : total_amount_paid 9.5 6.5 7 3 = 54.5 := by
  sorry

end group_payment_l1362_136235


namespace group_dynamics_index_l1362_136212

theorem group_dynamics_index (n : ℕ) (female_count : ℕ) : 
  n = 25 →
  female_count ≤ n →
  (n - female_count : ℚ) / n - (n - (n - female_count) : ℚ) / n = 9 / 25 →
  female_count = 8 := by
sorry

end group_dynamics_index_l1362_136212


namespace olivias_score_l1362_136207

theorem olivias_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) :
  n = 20 →
  avg_without = 85 →
  avg_with = 86 →
  (n * avg_without + x) / (n + 1) = avg_with →
  x = 106 :=
by sorry

end olivias_score_l1362_136207


namespace triangle_problem_l1362_136255

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement --/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.B * Real.sin t.C + Real.cos t.B + 2 * Real.cos (t.B + t.C) = 0)
  (h2 : Real.sin t.B ≠ 1)
  (h3 : 5 * Real.sin t.B = 3 * Real.sin t.A)
  (h4 : (1/2) * t.a * t.b * Real.sin t.C = 15 * Real.sqrt 3 / 4) :
  t.C = 2 * Real.pi / 3 ∧ t.a + t.b + t.c = 15 := by
  sorry

end triangle_problem_l1362_136255


namespace packing_peanuts_theorem_l1362_136205

/-- Calculates the amount of packing peanuts needed for each small order -/
def packing_peanuts_per_small_order (total_peanuts : ℕ) (large_orders : ℕ) (small_orders : ℕ) (peanuts_per_large_order : ℕ) : ℕ :=
  (total_peanuts - large_orders * peanuts_per_large_order) / small_orders

/-- Theorem: Given the conditions, the amount of packing peanuts needed for each small order is 50g -/
theorem packing_peanuts_theorem :
  packing_peanuts_per_small_order 800 3 4 200 = 50 := by
  sorry

end packing_peanuts_theorem_l1362_136205


namespace octagon_diagonal_length_l1362_136268

/-- The length of a diagonal in a regular octagon inscribed in a circle -/
theorem octagon_diagonal_length (r : ℝ) (h : r = 12) :
  let diagonal_length := Real.sqrt (288 + 144 * Real.sqrt 2)
  ∃ (AC : ℝ), AC = diagonal_length := by sorry

end octagon_diagonal_length_l1362_136268


namespace hyperbola_equation_y_axis_l1362_136275

/-- Given a hyperbola with foci on the y-axis, ratio of real to imaginary axis 2:3, 
    and passing through (√6, 2), prove its equation is y²/1 - x²/3 = 3 -/
theorem hyperbola_equation_y_axis (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 3 * a = 2 * b) (h4 : 4 / a^2 - 6 / b^2 = 1) :
  ∃ (k : ℝ), k * (y^2 / 1 - x^2 / 3) = 3 := by sorry


end hyperbola_equation_y_axis_l1362_136275


namespace distance_between_locations_l1362_136252

theorem distance_between_locations (s : ℝ) : 
  (s > 0) →  -- Distance is positive
  ((s/2 + 12) / (s/2 - 12) = s / (s - 40)) →  -- Condition when cars meet
  (s = 120) :=  -- Distance to prove
by sorry

end distance_between_locations_l1362_136252


namespace a_is_most_suitable_l1362_136264

-- Define the participants
inductive Participant
  | A
  | B
  | C
  | D

-- Define the variance for each participant
def variance (p : Participant) : ℝ :=
  match p with
  | Participant.A => 0.15
  | Participant.B => 0.2
  | Participant.C => 0.4
  | Participant.D => 0.35

-- Define the function to find the most suitable participant
def most_suitable : Participant :=
  Participant.A

-- Theorem to prove A is the most suitable
theorem a_is_most_suitable :
  ∀ p : Participant, variance most_suitable ≤ variance p :=
by sorry

end a_is_most_suitable_l1362_136264


namespace field_trip_arrangements_l1362_136250

/-- The number of grades --/
def num_grades : ℕ := 6

/-- The number of museums --/
def num_museums : ℕ := 6

/-- The number of grades that choose Museum A --/
def grades_choosing_a : ℕ := 2

/-- The number of ways to choose exactly two grades to visit Museum A --/
def ways_to_choose_a : ℕ := Nat.choose num_grades grades_choosing_a

/-- The number of museums excluding Museum A --/
def remaining_museums : ℕ := num_museums - 1

/-- The number of grades not choosing Museum A --/
def grades_not_choosing_a : ℕ := num_grades - grades_choosing_a

/-- The total number of ways to arrange the field trip --/
def total_arrangements : ℕ := ways_to_choose_a * (remaining_museums ^ grades_not_choosing_a)

theorem field_trip_arrangements :
  total_arrangements = Nat.choose num_grades grades_choosing_a * (remaining_museums ^ grades_not_choosing_a) :=
by sorry

end field_trip_arrangements_l1362_136250


namespace complex_magnitude_squared_l1362_136232

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 2 - 3*I) : 
  Complex.abs z ^ 2 = 13/4 := by
sorry

end complex_magnitude_squared_l1362_136232


namespace max_profit_l1362_136203

noncomputable section

def fixed_cost : ℝ := 2.5

def variable_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10*x^2 + 100*x
  else if x ≥ 40 then 701*x + 10000/x - 9450
  else 0

def selling_price : ℝ := 0.7

def profit (x : ℝ) : ℝ :=
  selling_price * x - (fixed_cost + variable_cost x)

def production_quantity : ℝ := 100

theorem max_profit :
  profit production_quantity = 9000 ∧
  ∀ x > 0, profit x ≤ profit production_quantity :=
sorry

end

end max_profit_l1362_136203


namespace complementary_sets_count_l1362_136229

/-- Represents a card in the deck -/
structure Card where
  shape : Fin 3
  color : Fin 3
  shade : Fin 3

/-- The deck of cards -/
def Deck : Finset Card := sorry

/-- A set of three cards -/
def ThreeCardSet : Type := Fin 3 → Card

/-- Checks if a three-card set is complementary -/
def is_complementary (set : ThreeCardSet) : Prop := sorry

/-- The set of all complementary three-card sets -/
def ComplementarySets : Finset ThreeCardSet := sorry

theorem complementary_sets_count : 
  Finset.card ComplementarySets = 702 := by sorry

end complementary_sets_count_l1362_136229


namespace angie_drinks_three_cups_per_day_l1362_136291

/-- Represents the number of cups of coffee per pound -/
def cupsPerPound : ℕ := 40

/-- Represents the number of pounds of coffee bought -/
def poundsBought : ℕ := 3

/-- Represents the number of days the coffee lasts -/
def daysLasting : ℕ := 40

/-- Calculates the number of cups of coffee Angie drinks per day -/
def cupsPerDay : ℕ := (poundsBought * cupsPerPound) / daysLasting

/-- Theorem stating that Angie drinks 3 cups of coffee per day -/
theorem angie_drinks_three_cups_per_day : cupsPerDay = 3 := by
  sorry

end angie_drinks_three_cups_per_day_l1362_136291


namespace sum_of_coefficients_l1362_136276

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 3)^11 = a + a₁*(x - 2) + a₂*(x - 2)^2 + a₃*(x - 2)^3 + 
    a₄*(x - 2)^4 + a₅*(x - 2)^5 + a₆*(x - 2)^6 + a₇*(x - 2)^7 + a₈*(x - 2)^8 + 
    a₉*(x - 2)^9 + a₁₀*(x - 2)^10 + a₁₁*(x - 2)^11 + a₁₂*(x - 2)^12 + a₁₃*(x - 2)^13) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ = 5 := by
sorry

end sum_of_coefficients_l1362_136276


namespace power_function_through_point_l1362_136279

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

-- Theorem statement
theorem power_function_through_point :
  ∀ f : ℝ → ℝ, isPowerFunction f → f 3 = Real.sqrt 3 → f = fun x => Real.sqrt x :=
by sorry


end power_function_through_point_l1362_136279


namespace functional_equation_zero_function_l1362_136236

theorem functional_equation_zero_function 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = x * f x + y * f y) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end functional_equation_zero_function_l1362_136236


namespace union_condition_intersection_condition_l1362_136282

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x ≤ 3}

-- Theorem for A ∪ B = B
theorem union_condition (a : ℝ) : A ∪ B a = B a ↔ a < 2 := by sorry

-- Theorem for A ∩ B = B
theorem intersection_condition (a : ℝ) : A ∩ B a = B a ↔ a ≥ 2 := by sorry

end union_condition_intersection_condition_l1362_136282


namespace kyle_fish_count_l1362_136237

/-- Given that Carla, Kyle, and Tasha caught a total of 36 fish, 
    Carla caught 8 fish, and Kyle and Tasha caught the same number of fish,
    prove that Kyle caught 14 fish. -/
theorem kyle_fish_count (total : ℕ) (carla : ℕ) (kyle : ℕ) (tasha : ℕ)
  (h1 : total = 36)
  (h2 : carla = 8)
  (h3 : kyle = tasha)
  (h4 : total = carla + kyle + tasha) :
  kyle = 14 := by
  sorry

end kyle_fish_count_l1362_136237


namespace angle_between_vectors_l1362_136260

def tangent_of_angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors
  (a b : ℝ × ℝ)
  (h1 : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 5)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1) :
  tangent_of_angle_between_vectors a b = Real.sqrt 3 := by sorry

end angle_between_vectors_l1362_136260


namespace other_endpoint_coordinate_sum_l1362_136204

/-- Given a line segment with midpoint (5, -8) and one endpoint at (9, -6),
    the sum of the coordinates of the other endpoint is -9. -/
theorem other_endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (5 = (9 + x) / 2) →
  (-8 = (-6 + y) / 2) →
  x + y = -9 :=
by sorry

end other_endpoint_coordinate_sum_l1362_136204


namespace sector_area_l1362_136223

/-- Given a sector with perimeter 10 and central angle 2 radians, its area is 25/4 -/
theorem sector_area (r : ℝ) (l : ℝ) (h1 : l + 2*r = 10) (h2 : l = 2*r) : 
  (1/2) * r * l = 25/4 := by
  sorry

end sector_area_l1362_136223


namespace gemstones_for_four_sets_l1362_136263

/-- Calculates the number of gemstones needed for earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let earrings_per_set := 2
  let magnets_per_earring := 2
  let buttons_per_earring := magnets_per_earring / 2
  let gemstones_per_earring := buttons_per_earring * 3
  num_sets * earrings_per_set * gemstones_per_earring

/-- Proves that 4 sets of earrings require 24 gemstones -/
theorem gemstones_for_four_sets : gemstones_needed 4 = 24 := by
  sorry

end gemstones_for_four_sets_l1362_136263


namespace least_zogs_for_dropping_advantage_l1362_136287

/-- Score for dropping n zogs -/
def drop_score (n : ℕ) : ℕ := n * (n + 1)

/-- Score for eating n zogs -/
def eat_score (n : ℕ) : ℕ := 8 * n

/-- Predicate for when dropping earns more points than eating -/
def dropping_beats_eating (n : ℕ) : Prop := drop_score n > eat_score n

theorem least_zogs_for_dropping_advantage : 
  (∀ k < 8, ¬dropping_beats_eating k) ∧ dropping_beats_eating 8 := by sorry

end least_zogs_for_dropping_advantage_l1362_136287


namespace circle_area_ratio_l1362_136296

theorem circle_area_ratio (C D : Real) (r_C r_D : ℝ) :
  (60 / 360) * (2 * Real.pi * r_C) = (40 / 360) * (2 * Real.pi * r_D) →
  (Real.pi * r_C^2) / (Real.pi * r_D^2) = 4 / 9 := by
sorry

end circle_area_ratio_l1362_136296


namespace opposite_of_negative_2023_l1362_136284

theorem opposite_of_negative_2023 : 
  ∃ x : ℤ, x + (-2023) = 0 ∧ x = 2023 :=
by sorry

end opposite_of_negative_2023_l1362_136284


namespace sqrt_difference_equality_l1362_136213

theorem sqrt_difference_equality : 
  Real.sqrt (121 + 81) - Real.sqrt (49 - 36) = Real.sqrt 202 - Real.sqrt 13 := by
  sorry

end sqrt_difference_equality_l1362_136213


namespace mixture_volume_proof_l1362_136278

theorem mixture_volume_proof (initial_water_percent : Real) 
                             (final_water_percent : Real)
                             (added_water : Real) :
  initial_water_percent = 0.20 →
  final_water_percent = 0.25 →
  added_water = 10 →
  ∃ (initial_volume : Real),
    initial_volume * initial_water_percent + added_water = 
    final_water_percent * (initial_volume + added_water) ∧
    initial_volume = 150 := by
  sorry

end mixture_volume_proof_l1362_136278


namespace sixth_power_sum_of_roots_l1362_136286

theorem sixth_power_sum_of_roots (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 5 + 2 = 0 → 
  s^2 - 2*s*Real.sqrt 5 + 2 = 0 → 
  r^6 + s^6 = 3904 := by
  sorry

end sixth_power_sum_of_roots_l1362_136286


namespace bridge_support_cans_l1362_136201

/-- The weight of a full can of soda in ounces -/
def full_can_weight : ℕ := 12 + 2

/-- The weight of an empty can in ounces -/
def empty_can_weight : ℕ := 2

/-- The total weight the bridge must support in ounces -/
def total_bridge_weight : ℕ := 88

/-- The number of additional empty cans -/
def additional_empty_cans : ℕ := 2

/-- The number of full cans of soda the bridge needs to support -/
def num_full_cans : ℕ := (total_bridge_weight - additional_empty_cans * empty_can_weight) / full_can_weight

theorem bridge_support_cans : num_full_cans = 6 := by
  sorry

end bridge_support_cans_l1362_136201


namespace even_function_properties_l1362_136256

def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

def HasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≥ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem even_function_properties (f : ℝ → ℝ) :
  EvenFunction f →
  IncreasingOn f 3 7 →
  HasMinimumOn f 3 7 2 →
  DecreasingOn f (-7) (-3) ∧ HasMaximumOn f (-7) (-3) 2 := by
  sorry

end even_function_properties_l1362_136256


namespace particle_position_after_3000_minutes_l1362_136225

/-- Represents the position of a particle as a pair of integers -/
def Position := ℤ × ℤ

/-- Represents the direction of movement -/
inductive Direction
| Up
| Right
| Down
| Left

/-- Defines the movement pattern of the particle -/
def move_particle (start : Position) (time : ℕ) : Position :=
  sorry

/-- The theorem to be proved -/
theorem particle_position_after_3000_minutes :
  move_particle (0, 0) 3000 = (0, 27) :=
sorry

end particle_position_after_3000_minutes_l1362_136225


namespace planned_goats_addition_l1362_136219

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- Calculates the total number of animals -/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.pigs + farm.goats

/-- The initial number of animals on the farm -/
def initialFarm : FarmAnimals :=
  { cows := 2, pigs := 3, goats := 6 }

/-- The planned additions to the farm -/
def plannedAdditions : FarmAnimals :=
  { cows := 3, pigs := 5, goats := 0 }

/-- The final desired number of animals -/
def finalTotal : ℕ := 21

/-- Theorem: The number of goats the farmer plans to add is 2 -/
theorem planned_goats_addition :
  finalTotal = totalAnimals initialFarm + totalAnimals plannedAdditions + 2 := by
  sorry

end planned_goats_addition_l1362_136219


namespace common_roots_product_l1362_136202

/-- Given two cubic equations with two common roots, prove their product is 8 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (p q r s : ℝ), 
    (p^3 + C*p + 20 = 0) ∧ 
    (q^3 + C*q + 20 = 0) ∧ 
    (r^3 + C*r + 20 = 0) ∧
    (p^3 + D*p^2 + 80 = 0) ∧ 
    (q^3 + D*q^2 + 80 = 0) ∧ 
    (s^3 + D*s^2 + 80 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧
    (p ≠ s) ∧ (q ≠ s) →
    p * q = 8 := by
  sorry

end common_roots_product_l1362_136202


namespace original_balance_l1362_136247

/-- Proves that if a balance incurs a 2% finance charge and results in a total of $153, then the original balance was $150. -/
theorem original_balance (total : ℝ) (finance_charge_rate : ℝ) (h1 : finance_charge_rate = 0.02) (h2 : total = 153) :
  ∃ (original : ℝ), original * (1 + finance_charge_rate) = total ∧ original = 150 :=
by sorry

end original_balance_l1362_136247


namespace mailing_cost_calculation_l1362_136298

/-- Calculates the total cost of mailing letters and packages -/
def total_mailing_cost (letter_cost package_cost : ℚ) (num_letters : ℕ) : ℚ :=
  let num_packages := num_letters - 2
  letter_cost * num_letters + package_cost * num_packages

/-- Theorem: Given the conditions, the total mailing cost is $4.49 -/
theorem mailing_cost_calculation :
  total_mailing_cost (37/100) (88/100) 5 = 449/100 := by
sorry

end mailing_cost_calculation_l1362_136298


namespace gaeun_wins_l1362_136228

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Nana's flight distance in meters -/
def nana_distance_m : ℝ := 1.618

/-- Gaeun's flight distance in centimeters -/
def gaeun_distance_cm : ℝ := 162.3

/-- Theorem stating that Gaeun's flight distance is greater than Nana's by 0.5 cm -/
theorem gaeun_wins :
  gaeun_distance_cm - (nana_distance_m * meters_to_cm) = 0.5 := by
  sorry

end gaeun_wins_l1362_136228


namespace cloth_cost_price_l1362_136238

/-- Given a trader sells cloth with the following conditions:
  * Sells 45 meters of cloth
  * Total selling price is 4500 Rs
  * Profit per meter is 14 Rs
  Prove that the cost price of one meter of cloth is 86 Rs -/
theorem cloth_cost_price 
  (total_meters : ℕ) 
  (selling_price : ℕ) 
  (profit_per_meter : ℕ) 
  (h1 : total_meters = 45)
  (h2 : selling_price = 4500)
  (h3 : profit_per_meter = 14) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 86 := by
sorry

end cloth_cost_price_l1362_136238


namespace initial_time_calculation_l1362_136270

theorem initial_time_calculation (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 180)
  (h2 : new_speed = 20)
  (h3 : time_ratio = 3/2) :
  let new_time := distance / new_speed
  let initial_time := new_time * time_ratio
  initial_time = 13.5 := by
  sorry

end initial_time_calculation_l1362_136270


namespace quadratic_value_theorem_l1362_136222

theorem quadratic_value_theorem (x : ℝ) : 
  x^2 - 2*x - 3 = 0 → 2*x^2 - 4*x + 12 = 18 := by
sorry

end quadratic_value_theorem_l1362_136222


namespace mMobileCheaperByEleven_l1362_136246

-- Define the cost structure for T-Mobile
def tMobileBaseCost : ℕ := 50
def tMobileAdditionalLineCost : ℕ := 16

-- Define the cost structure for M-Mobile
def mMobileBaseCost : ℕ := 45
def mMobileAdditionalLineCost : ℕ := 14

-- Define the number of lines needed
def totalLines : ℕ := 5

-- Define the function to calculate the total cost for a given plan
def calculateTotalCost (baseCost additionalLineCost : ℕ) : ℕ :=
  baseCost + (totalLines - 2) * additionalLineCost

-- Theorem statement
theorem mMobileCheaperByEleven :
  calculateTotalCost tMobileBaseCost tMobileAdditionalLineCost -
  calculateTotalCost mMobileBaseCost mMobileAdditionalLineCost = 11 := by
  sorry

end mMobileCheaperByEleven_l1362_136246


namespace solution_set_f_less_than_8_range_of_m_for_solution_existence_l1362_136267

-- Define the function f
def f (x : ℝ) : ℝ := 45 * abs (2 * x + 3) + abs (2 * x - 1)

-- Theorem for part I
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -5/2 < x ∧ x < 3/2} :=
sorry

-- Theorem for part II
theorem range_of_m_for_solution_existence :
  {m : ℝ | ∃ x, f x ≤ |3 * m + 1|} = {m : ℝ | m ≤ -5/3 ∨ m ≥ 1} :=
sorry

end solution_set_f_less_than_8_range_of_m_for_solution_existence_l1362_136267


namespace profit_is_152_l1362_136288

/-- The profit made from selling jerseys -/
def profit_from_jerseys (profit_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  profit_per_jersey * jerseys_sold

/-- Theorem: The profit from selling jerseys is $152 -/
theorem profit_is_152 :
  profit_from_jerseys 76 2 = 152 := by
  sorry

end profit_is_152_l1362_136288


namespace arithmetic_sequence_sum_l1362_136230

/-- For an arithmetic sequence {a_n} with S_n as the sum of its first n terms, 
    if S_9 = 27, then a_4 + a_6 = 6 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_S9 : S 9 = 27) : 
  a 4 + a 6 = 6 := by
sorry

end arithmetic_sequence_sum_l1362_136230


namespace ratio_equality_l1362_136221

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : y / (x + z) = (x - y) / z ∧ y / (x + z) = x / (y + 2*z)) : 
  x / y = 2 / 3 := by
sorry

end ratio_equality_l1362_136221


namespace multiply_by_special_number_l1362_136265

theorem multiply_by_special_number : ∃ x : ℝ, x * (1/1000) = 0.735 ∧ 10 * x = 7350 := by
  sorry

end multiply_by_special_number_l1362_136265


namespace sum_is_zero_l1362_136269

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Three non-zero vectors with specified properties -/
structure ThreeVectors (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (a b c : V)
  (a_nonzero : a ≠ 0)
  (b_nonzero : b ≠ 0)
  (c_nonzero : c ≠ 0)
  (ab_noncollinear : ¬ ∃ (r : ℝ), a = r • b)
  (bc_noncollinear : ¬ ∃ (r : ℝ), b = r • c)
  (ca_noncollinear : ¬ ∃ (r : ℝ), c = r • a)
  (ab_parallel_c : ∃ (m : ℝ), a + b = m • c)
  (bc_parallel_a : ∃ (n : ℝ), b + c = n • a)

/-- The sum of three vectors with the given properties is zero -/
theorem sum_is_zero (v : ThreeVectors V) : v.a + v.b + v.c = 0 :=
sorry

end sum_is_zero_l1362_136269


namespace x_convergence_bound_l1362_136216

def x : ℕ → ℚ
  | 0 => 3
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

theorem x_convergence_bound :
  ∃ m : ℕ, 31 ≤ m ∧ m ≤ 90 ∧ 
    x m ≤ 5 + 1 / (2^15) ∧
    ∀ k : ℕ, 0 < k → k < m → x k > 5 + 1 / (2^15) := by
  sorry

end x_convergence_bound_l1362_136216


namespace existence_of_special_sequence_l1362_136211

theorem existence_of_special_sequence :
  ∃ (a : Fin 2013 → ℕ), 
    (∀ i j : Fin 2013, i ≠ j → a i ≠ a j) ∧ 
    (∀ k m : Fin 2013, k < m → (a m + a k) % (a m - a k) = 0) :=
sorry

end existence_of_special_sequence_l1362_136211


namespace rational_reachability_l1362_136280

-- Define the operations
def f (x : ℚ) : ℚ := (1 + x) / x
def g (x : ℚ) : ℚ := (1 - x) / x

-- Define a type for sequences of operations
inductive Op
| F : Op
| G : Op

def apply_op (op : Op) (x : ℚ) : ℚ :=
  match op with
  | Op.F => f x
  | Op.G => g x

def apply_ops (ops : List Op) (x : ℚ) : ℚ :=
  ops.foldl (λ acc op => apply_op op acc) x

theorem rational_reachability (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (ops : List Op), apply_ops ops a = b :=
sorry

end rational_reachability_l1362_136280


namespace negation_of_forall_positive_square_plus_x_l1362_136248

theorem negation_of_forall_positive_square_plus_x :
  (¬ ∀ x : ℝ, x^2 + x > 0) ↔ (∃ x : ℝ, x^2 + x ≤ 0) := by
  sorry

end negation_of_forall_positive_square_plus_x_l1362_136248


namespace students_who_like_basketball_l1362_136242

/-- Given a class of students where some play basketball and/or cricket, 
    this theorem proves the number of students who like basketball. -/
theorem students_who_like_basketball 
  (cricket : ℕ)
  (both : ℕ)
  (basketball_or_cricket : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 4)
  (h3 : basketball_or_cricket = 14) :
  basketball_or_cricket = cricket + (basketball_or_cricket - cricket - both) - both :=
by sorry

end students_who_like_basketball_l1362_136242


namespace trigonometric_expression_equals_one_l1362_136233

theorem trigonometric_expression_equals_one :
  (Real.tan (45 * π / 180))^2 - (Real.sin (45 * π / 180))^2 = 
  (Real.tan (45 * π / 180))^2 * (Real.sin (45 * π / 180))^2 := by
  sorry

end trigonometric_expression_equals_one_l1362_136233


namespace johns_father_age_l1362_136283

theorem johns_father_age (john_age father_age : ℕ) : 
  john_age + father_age = 77 →
  father_age = 2 * john_age + 32 →
  john_age = 15 →
  father_age = 62 := by
sorry

end johns_father_age_l1362_136283


namespace smallest_m_is_671_l1362_136271

def is_valid (m n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a = 2015^(3*m+1) ∧
    b = 2015^(6*n+2) ∧
    a < b ∧
    a % 10^2014 = b % 10^2014

theorem smallest_m_is_671 :
  (∃ (n : ℕ), is_valid 671 n) ∧
  (∀ (m : ℕ), m < 671 → ¬∃ (n : ℕ), is_valid m n) :=
sorry

end smallest_m_is_671_l1362_136271


namespace peanut_butter_cookie_probability_l1362_136297

/-- The probability of selecting a peanut butter cookie -/
def peanut_butter_probability (peanut_butter_cookies : ℕ) (chocolate_chip_cookies : ℕ) (lemon_cookies : ℕ) : ℚ :=
  peanut_butter_cookies / (peanut_butter_cookies + chocolate_chip_cookies + lemon_cookies)

theorem peanut_butter_cookie_probability :
  peanut_butter_probability 70 50 20 = 1/2 := by
  sorry

end peanut_butter_cookie_probability_l1362_136297


namespace inequality_proof_l1362_136273

theorem inequality_proof (x y : ℝ) : 2^(-Real.cos x^2) + 2^(-Real.sin x^2) ≥ Real.sin y + Real.cos y := by
  sorry

end inequality_proof_l1362_136273


namespace stating_distribution_schemes_eq_60_l1362_136293

/-- Represents the number of female students -/
def num_female : ℕ := 5

/-- Represents the number of male students -/
def num_male : ℕ := 2

/-- Represents the number of groups -/
def num_groups : ℕ := 2

/-- 
Calculates the number of ways to distribute students into groups
such that each group has at least one female and one male student
-/
def distribution_schemes (f : ℕ) (m : ℕ) (g : ℕ) : ℕ :=
  2 * (2^f - 2)

/-- 
Theorem stating that the number of distribution schemes
for the given problem is 60
-/
theorem distribution_schemes_eq_60 :
  distribution_schemes num_female num_male num_groups = 60 := by
  sorry

end stating_distribution_schemes_eq_60_l1362_136293


namespace cuboid_volume_transformation_l1362_136241

theorem cuboid_volume_transformation (V : ℝ) (h : V = 343) : 
  let s := V^(1/3)
  let L := 3 * s
  let W := 1.5 * s
  let H := 2.5 * s
  L * W * H = 38587.5 := by
  sorry

end cuboid_volume_transformation_l1362_136241


namespace inequality_system_solution_range_l1362_136234

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ 
    (2 * ↑x₁ - 1 ≤ 5 ∧ ↑x₁ - 1 ≥ m) ∧ 
    (2 * ↑x₂ - 1 ≤ 5 ∧ ↑x₂ - 1 ≥ m) ∧
    (∀ x : ℤ, (2 * ↑x - 1 ≤ 5 ∧ ↑x - 1 ≥ m) → (x = x₁ ∨ x = x₂))) ↔
  (-1 < m ∧ m ≤ 0) :=
by sorry

end inequality_system_solution_range_l1362_136234


namespace triangle_angle_value_max_side_sum_l1362_136218

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

theorem triangle_angle_value (t : Triangle) (h : triangle_condition t) : t.A = 2 * Real.pi / 3 := by
  sorry

theorem max_side_sum (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = 4) :
  ∃ (b c : ℝ), t.b = b ∧ t.c = c ∧ b + c ≤ 8 * Real.sqrt 3 / 3 ∧
  ∀ (b' c' : ℝ), t.b = b' ∧ t.c = c' → b' + c' ≤ 8 * Real.sqrt 3 / 3 := by
  sorry

end triangle_angle_value_max_side_sum_l1362_136218


namespace no_three_squares_sum_2015_l1362_136206

theorem no_three_squares_sum_2015 : ¬ ∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2015 := by
  sorry

end no_three_squares_sum_2015_l1362_136206


namespace expression_evaluation_l1362_136249

theorem expression_evaluation :
  let x : ℚ := 7/6
  (x - 1) / x / (x - (2*x - 1) / x) = 6 := by sorry

end expression_evaluation_l1362_136249


namespace pythagorean_sum_inequality_l1362_136272

theorem pythagorean_sum_inequality (a b c x y z : ℕ) 
  (h1 : a^2 + b^2 = c^2) (h2 : x^2 + y^2 = z^2) :
  (a + x)^2 + (b + y)^2 ≤ (c + z)^2 ∧ 
  ((a + x)^2 + (b + y)^2 = (c + z)^2 ↔ (a * z = c * x ∧ b * z = c * y)) :=
sorry

end pythagorean_sum_inequality_l1362_136272


namespace triangle_angles_l1362_136217

theorem triangle_angles (a b c : ℝ) (h1 : a = 2) (h2 : b = 2) (h3 : c = Real.sqrt 6 - Real.sqrt 2) :
  ∃ (α β γ : ℝ),
    α = 30 * π / 180 ∧
    β = 75 * π / 180 ∧
    γ = 75 * π / 180 ∧
    (Real.cos α = (a^2 + b^2 - c^2) / (2 * a * b)) ∧
    (Real.cos β = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
    (Real.cos γ = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
    α + β + γ = π := by
  sorry

end triangle_angles_l1362_136217


namespace roja_work_time_l1362_136226

/-- Given that Malar and Roja combined complete a task in 35 days,
    and Malar alone completes the same work in 60 days,
    prove that Roja alone can complete the work in 210 days. -/
theorem roja_work_time (combined_time malar_time : ℝ)
  (h_combined : combined_time = 35)
  (h_malar : malar_time = 60) :
  let roja_time := (combined_time * malar_time) / (malar_time - combined_time)
  roja_time = 210 := by
sorry

end roja_work_time_l1362_136226


namespace sqrt_fraction_equals_half_l1362_136299

theorem sqrt_fraction_equals_half : 
  Real.sqrt ((16^6 + 8^8) / (16^3 + 8^9)) = 1/2 := by
  sorry

end sqrt_fraction_equals_half_l1362_136299


namespace x_polynomial_equality_l1362_136294

theorem x_polynomial_equality (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^8 + x^4 = 35292*x - 13652 := by
  sorry

end x_polynomial_equality_l1362_136294


namespace boat_speed_l1362_136224

/-- Proves that the speed of a boat in still water is 60 kmph given the conditions of the problem -/
theorem boat_speed (stream_speed : ℝ) (upstream_time downstream_time : ℝ) 
  (h1 : stream_speed = 20)
  (h2 : upstream_time = 2 * downstream_time)
  (h3 : downstream_time > 0)
  (h4 : ∀ (boat_speed : ℝ), 
    (boat_speed + stream_speed) * downstream_time = 
    (boat_speed - stream_speed) * upstream_time → 
    boat_speed = 60) : 
  ∃ (boat_speed : ℝ), boat_speed = 60 := by
  sorry

end boat_speed_l1362_136224


namespace bus_passengers_count_l1362_136227

theorem bus_passengers_count :
  let men_count : ℕ := 18
  let women_count : ℕ := 26
  let children_count : ℕ := 10
  let total_passengers : ℕ := men_count + women_count + children_count
  total_passengers = 54 := by sorry

end bus_passengers_count_l1362_136227


namespace square_area_to_perimeter_ratio_l1362_136289

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁^2 / s₂^2 = 16 / 81) :
  (4 * s₁) / (4 * s₂) = 4 / 9 := by
  sorry

end square_area_to_perimeter_ratio_l1362_136289


namespace real_roots_condition_l1362_136292

theorem real_roots_condition (x : ℝ) :
  (∃ y : ℝ, y^2 + 5*x*y + 2*x + 9 = 0) ↔ (x ≤ -0.6 ∨ x ≥ 0.92) :=
by sorry

end real_roots_condition_l1362_136292


namespace inequality_proof_l1362_136295

theorem inequality_proof (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end inequality_proof_l1362_136295


namespace f_inequality_and_abs_inequality_l1362_136214

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | 1 < x ∧ x < 4}

-- Theorem statement
theorem f_inequality_and_abs_inequality :
  (∀ x, f x < 3 ↔ x ∈ M) ∧
  (∀ a b, a ∈ M → b ∈ M → |a + b| < |1 + a * b|) := by sorry

end f_inequality_and_abs_inequality_l1362_136214


namespace stratified_sampling_first_year_l1362_136266

theorem stratified_sampling_first_year
  (total_sample : ℕ)
  (first_year_ratio second_year_ratio third_year_ratio : ℕ)
  (h_total_sample : total_sample = 56)
  (h_ratios : first_year_ratio = 7 ∧ second_year_ratio = 3 ∧ third_year_ratio = 4) :
  (total_sample * first_year_ratio) / (first_year_ratio + second_year_ratio + third_year_ratio) = 28 := by
  sorry

#check stratified_sampling_first_year

end stratified_sampling_first_year_l1362_136266


namespace find_a₂_l1362_136274

-- Define the equation as a function
def f (x a₀ a₁ a₂ a₃ : ℝ) : ℝ := a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3

-- State the theorem
theorem find_a₂ (a₀ a₁ a₂ a₃ : ℝ) : 
  (∀ x : ℝ, x^3 = f x a₀ a₁ a₂ a₃) → a₂ = 6 := by
  sorry

end find_a₂_l1362_136274


namespace min_value_xy_expression_l1362_136210

theorem min_value_xy_expression :
  (∀ x y : ℝ, (x*y - 2)^2 + (x - y)^2 ≥ 0) ∧
  (∃ x y : ℝ, (x*y - 2)^2 + (x - y)^2 = 0) :=
by sorry

end min_value_xy_expression_l1362_136210


namespace chord_length_range_l1362_136251

/-- The chord length intercepted by the line y = x + t on the circle x + y² = 8 -/
def chordLength (t : ℝ) : ℝ := sorry

theorem chord_length_range (t : ℝ) :
  (∀ x y : ℝ, y = x + t ∧ x + y^2 = 8 → chordLength t ≥ 4 * Real.sqrt 2 / 3) →
  t ∈ Set.Icc (-(8 * Real.sqrt 2 / 3)) (8 * Real.sqrt 2 / 3) :=
sorry

end chord_length_range_l1362_136251


namespace emily_beads_count_l1362_136245

/-- The number of beads per necklace -/
def beads_per_necklace : ℕ := 5

/-- The number of necklaces Emily made -/
def necklaces_made : ℕ := 4

/-- The total number of beads Emily used -/
def total_beads : ℕ := beads_per_necklace * necklaces_made

theorem emily_beads_count : total_beads = 20 := by
  sorry

end emily_beads_count_l1362_136245


namespace min_side_length_two_triangles_l1362_136257

/-- Given two triangles ABC and DBC sharing side BC, with known side lengths,
    prove that the minimum integral length of BC is 16 cm. -/
theorem min_side_length_two_triangles 
  (AB AC DC BD : ℝ) 
  (h_AB : AB = 7)
  (h_AC : AC = 18)
  (h_DC : DC = 10)
  (h_BD : BD = 25) :
  (∃ (BC : ℕ), BC ≥ 16 ∧ ∀ (n : ℕ), n < 16 → 
    (n : ℝ) ≤ AC - AB ∨ (n : ℝ) ≤ BD - DC) :=
by sorry

end min_side_length_two_triangles_l1362_136257


namespace sin_double_angle_l1362_136239

theorem sin_double_angle (α : ℝ) (h : Real.sin (α - π/4) = 3/5) : 
  Real.sin (2 * α) = 7/25 := by
sorry

end sin_double_angle_l1362_136239


namespace page_number_digit_difference_l1362_136215

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between the count of 5's and 3's in page numbers from 1 to 512 -/
theorem page_number_digit_difference :
  let pages := 512
  let start_page := 1
  let end_page := pages
  let digit_five := 5
  let digit_three := 3
  (countDigit digit_five start_page end_page) - (countDigit digit_three start_page end_page) = 22 :=
sorry

end page_number_digit_difference_l1362_136215


namespace window_dimensions_l1362_136231

/-- Represents the dimensions of a rectangular glass pane -/
structure PaneDimensions where
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular window -/
structure WindowDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the dimensions of a window given the pane dimensions and border widths -/
def calculateWindowDimensions (pane : PaneDimensions) (topBorderWidth sideBorderWidth : ℝ) : WindowDimensions :=
  { width := 3 * pane.width + 4 * sideBorderWidth,
    height := 2 * pane.height + 2 * topBorderWidth + sideBorderWidth }

theorem window_dimensions (y : ℝ) :
  let pane : PaneDimensions := { width := 4 * y, height := 3 * y }
  let window := calculateWindowDimensions pane 3 1
  window.width = 12 * y + 4 ∧ window.height = 6 * y + 7 := by
  sorry

#check window_dimensions

end window_dimensions_l1362_136231


namespace geometric_sequence_product_l1362_136258

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_a5 : a 5 = 2)
  (h_a9 : a 9 = 32) :
  a 4 * a 10 = 64 := by
sorry

end geometric_sequence_product_l1362_136258


namespace tony_saturday_sandwiches_l1362_136290

/-- The number of sandwiches Tony made on Saturday -/
def sandwiches_on_saturday (
  slices_per_sandwich : ℕ)
  (days_in_week : ℕ)
  (initial_slices : ℕ)
  (remaining_slices : ℕ)
  (sandwiches_per_day : ℕ) : ℕ :=
  ((initial_slices - remaining_slices) - (days_in_week - 1) * sandwiches_per_day * slices_per_sandwich) / slices_per_sandwich

theorem tony_saturday_sandwiches :
  sandwiches_on_saturday 2 6 22 6 1 = 3 := by
  sorry

end tony_saturday_sandwiches_l1362_136290


namespace remaining_digits_count_l1362_136243

theorem remaining_digits_count (total : ℕ) (avg_total : ℚ) (subset : ℕ) (avg_subset : ℚ) (avg_remaining : ℚ)
  (h1 : total = 10)
  (h2 : avg_total = 80)
  (h3 : subset = 6)
  (h4 : avg_subset = 58)
  (h5 : avg_remaining = 113) :
  total - subset = 4 := by
  sorry

end remaining_digits_count_l1362_136243


namespace cos_two_theta_value_l1362_136240

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) + Real.cos (θ / 2) = 1 / 2) : 
  Real.cos (2 * θ) = -1 / 8 := by
  sorry

end cos_two_theta_value_l1362_136240


namespace bridge_length_l1362_136261

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 250 →
  crossing_time = 32 →
  train_speed_kmh = 45 →
  ∃ (bridge_length : ℝ), bridge_length = 150 := by
  sorry


end bridge_length_l1362_136261
