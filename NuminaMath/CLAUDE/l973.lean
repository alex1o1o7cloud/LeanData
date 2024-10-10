import Mathlib

namespace square_difference_equality_l973_97331

theorem square_difference_equality : 1.99^2 - 1.98 * 1.99 + 0.99^2 = 1 := by
  sorry

end square_difference_equality_l973_97331


namespace cnc_processing_time_l973_97316

/-- The time required for one CNC machine to process a given number of parts, 
    given the rate of multiple machines. -/
theorem cnc_processing_time 
  (machines : ℕ) 
  (parts : ℕ) 
  (hours : ℕ) 
  (target_parts : ℕ) : 
  machines > 0 → 
  parts > 0 → 
  hours > 0 → 
  target_parts > 0 → 
  (3 : ℕ) = machines → 
  (960 : ℕ) = parts → 
  (4 : ℕ) = hours → 
  (400 : ℕ) = target_parts → 
  (5 : ℕ) = (target_parts * machines * hours) / parts := by
  sorry


end cnc_processing_time_l973_97316


namespace parabola_reflection_l973_97322

/-- Given a parabola y = x^2 and a line y = x + 2, prove that the reflection of the parabola about the line is x = y^2 - 4y + 2 -/
theorem parabola_reflection (x y : ℝ) :
  (y = x^2) ∧ (∃ (x' y' : ℝ), y' = x' + 2 ∧ 
    ((x' = y - 2 ∧ y' = x + 2) ∨ (x' = x ∧ y' = y))) →
  x = y^2 - 4*y + 2 :=
by sorry

end parabola_reflection_l973_97322


namespace intersection_of_M_and_N_l973_97347

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l973_97347


namespace chris_savings_proof_l973_97388

def chris_birthday_savings (grandmother_gift aunt_uncle_gift parents_gift chores_money friend_gift_cost total_after : ℝ) : Prop :=
  let total_received := grandmother_gift + aunt_uncle_gift + parents_gift + chores_money
  let additional_amount := total_received - friend_gift_cost
  let savings_before := total_after - additional_amount
  let percentage_increase := (additional_amount / savings_before) * 100
  savings_before = 144 ∧ percentage_increase = 93.75

theorem chris_savings_proof :
  chris_birthday_savings 25 20 75 30 15 279 :=
sorry

end chris_savings_proof_l973_97388


namespace quadratic_inequality_range_l973_97313

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 4 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 16) := by
  sorry

end quadratic_inequality_range_l973_97313


namespace turban_price_turban_price_is_70_l973_97353

/-- The price of a turban given the following conditions:
  * The total salary for one year is Rs. 90 plus the turban
  * The servant works for 9 months (3/4 of a year)
  * The servant receives Rs. 50 and the turban after 9 months
-/
theorem turban_price : ℝ → Prop :=
  fun price =>
    let yearly_salary := 90 + price
    let worked_fraction := 3 / 4
    let received_salary := 50 + price
    worked_fraction * yearly_salary = received_salary

/-- The price of the turban is 70 rupees -/
theorem turban_price_is_70 : turban_price 70 := by
  sorry

end turban_price_turban_price_is_70_l973_97353


namespace age_ratio_six_years_ago_l973_97364

/-- Given Henry and Jill's ages, prove their age ratio 6 years ago -/
theorem age_ratio_six_years_ago 
  (henry_age : ℕ) 
  (jill_age : ℕ) 
  (henry_age_eq : henry_age = 20)
  (jill_age_eq : jill_age = 13)
  (sum_ages : henry_age + jill_age = 33)
  (past_multiple : ∃ k : ℕ, henry_age - 6 = k * (jill_age - 6)) :
  (henry_age - 6) / (jill_age - 6) = 2 := by
sorry

end age_ratio_six_years_ago_l973_97364


namespace opposite_reciprocal_theorem_l973_97369

theorem opposite_reciprocal_theorem (a b c d m : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 2) :
  (a + b) / (4 * m) + m^2 - 3 * c * d = 1 := by sorry

end opposite_reciprocal_theorem_l973_97369


namespace days_to_fulfill_orders_l973_97357

/-- Represents the production details of Wallace's beef jerky company -/
structure JerkyProduction where
  small_batch_time : ℕ
  small_batch_output : ℕ
  large_batch_time : ℕ
  large_batch_output : ℕ
  total_small_bags_ordered : ℕ
  total_large_bags_ordered : ℕ
  small_bags_in_stock : ℕ
  large_bags_in_stock : ℕ
  max_daily_production_hours : ℕ

/-- Calculates the minimum number of days required to fulfill all orders -/
def min_days_to_fulfill_orders (prod : JerkyProduction) : ℕ :=
  let small_bags_to_produce := prod.total_small_bags_ordered - prod.small_bags_in_stock
  let large_bags_to_produce := prod.total_large_bags_ordered - prod.large_bags_in_stock
  let small_batches_needed := (small_bags_to_produce + prod.small_batch_output - 1) / prod.small_batch_output
  let large_batches_needed := (large_bags_to_produce + prod.large_batch_output - 1) / prod.large_batch_output
  let total_hours_needed := small_batches_needed * prod.small_batch_time + large_batches_needed * prod.large_batch_time
  (total_hours_needed + prod.max_daily_production_hours - 1) / prod.max_daily_production_hours

/-- Theorem stating that given the specific conditions, 13 days are required to fulfill all orders -/
theorem days_to_fulfill_orders :
  let prod := JerkyProduction.mk 8 12 12 8 157 97 18 10 18
  min_days_to_fulfill_orders prod = 13 := by
  sorry


end days_to_fulfill_orders_l973_97357


namespace digit_strike_out_theorem_l973_97308

/-- Represents a positive integer as a list of its digits --/
def DigitList := List Nat

/-- Checks if a number represented as a list of digits is divisible by 9 --/
def isDivisibleBy9 (n : DigitList) : Prop :=
  (n.sum % 9 = 0)

/-- Checks if a number can be obtained by striking out one digit from another number --/
def canBeObtainedByStrikingOut (m n : DigitList) : Prop :=
  ∃ (i : Nat), i < n.length ∧ m = (n.take i ++ n.drop (i+1))

/-- The main theorem --/
theorem digit_strike_out_theorem (N : DigitList) :
  (∃ (M : DigitList), N.sum = 9 * M.sum ∧ 
    canBeObtainedByStrikingOut M N ∧ 
    isDivisibleBy9 M) →
  (∀ (K : DigitList), canBeObtainedByStrikingOut K M → isDivisibleBy9 K) ∧
  (N ∈ [[1,0,1,2,5], [2,0,2,5], [3,0,3,7,5], [4,0,5], [5,0,6,2,5], [6,7,5], [7,0,8,7,5]]) :=
by
  sorry


end digit_strike_out_theorem_l973_97308


namespace president_vp_advisory_board_selection_l973_97340

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem president_vp_advisory_board_selection (total_people : ℕ) (h : total_people = 10) :
  (total_people) * (total_people - 1) * (choose (total_people - 2) 2) = 2520 :=
sorry

end president_vp_advisory_board_selection_l973_97340


namespace decrement_calculation_l973_97309

theorem decrement_calculation (n : ℕ) (original_mean updated_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : updated_mean = 194) :
  (n : ℝ) * original_mean - n * updated_mean = 6 * n := by
  sorry

end decrement_calculation_l973_97309


namespace rooms_per_hall_first_wing_is_32_l973_97341

/-- Represents a hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  first_wing_floors : ℕ
  first_wing_halls_per_floor : ℕ
  second_wing_floors : ℕ
  second_wing_halls_per_floor : ℕ
  second_wing_rooms_per_hall : ℕ

/-- Calculates the number of rooms in each hall of the first wing -/
def rooms_per_hall_first_wing (h : Hotel) : ℕ :=
  let second_wing_rooms := h.second_wing_floors * h.second_wing_halls_per_floor * h.second_wing_rooms_per_hall
  let first_wing_rooms := h.total_rooms - second_wing_rooms
  let total_halls_first_wing := h.first_wing_floors * h.first_wing_halls_per_floor
  first_wing_rooms / total_halls_first_wing

/-- Theorem stating that for the given hotel configuration, 
    each hall in the first wing has 32 rooms -/
theorem rooms_per_hall_first_wing_is_32 :
  rooms_per_hall_first_wing {
    total_rooms := 4248,
    first_wing_floors := 9,
    first_wing_halls_per_floor := 6,
    second_wing_floors := 7,
    second_wing_halls_per_floor := 9,
    second_wing_rooms_per_hall := 40
  } = 32 := by
  sorry

end rooms_per_hall_first_wing_is_32_l973_97341


namespace second_pump_rate_l973_97349

/-- Proves that the rate of the second pump is 70 gallons per hour given the conditions -/
theorem second_pump_rate (pump1_rate : ℝ) (total_time : ℝ) (total_volume : ℝ) (pump2_time : ℝ)
  (h1 : pump1_rate = 180)
  (h2 : total_time = 6)
  (h3 : total_volume = 1325)
  (h4 : pump2_time = 3.5) :
  (total_volume - pump1_rate * total_time) / pump2_time = 70 := by
  sorry

end second_pump_rate_l973_97349


namespace half_inequality_l973_97335

theorem half_inequality (a b : ℝ) (h : a > b) : a / 2 > b / 2 := by
  sorry

end half_inequality_l973_97335


namespace divisibility_of_power_plus_minus_one_l973_97392

theorem divisibility_of_power_plus_minus_one (n : ℕ) (h : ¬ 17 ∣ n) :
  17 ∣ (n^8 + 1) ∨ 17 ∣ (n^8 - 1) := by
sorry

end divisibility_of_power_plus_minus_one_l973_97392


namespace exactly_two_linear_functions_l973_97354

/-- Two quadratic trinomials -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Linear function -/
structure LinearFunction where
  m : ℝ
  n : ℝ

/-- Evaluate a quadratic trinomial at a given x -/
def evaluate_quadratic (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Evaluate a linear function at a given x -/
def evaluate_linear (l : LinearFunction) (x : ℝ) : ℝ :=
  l.m * x + l.n

/-- The main theorem -/
theorem exactly_two_linear_functions (P Q : QuadraticTrinomial) :
  ∃! (l₁ l₂ : LinearFunction),
    (∀ x : ℝ, evaluate_quadratic P x = evaluate_quadratic Q (evaluate_linear l₁ x)) ∧
    (∀ x : ℝ, evaluate_quadratic P x = evaluate_quadratic Q (evaluate_linear l₂ x)) ∧
    l₁ ≠ l₂ :=
  sorry

end exactly_two_linear_functions_l973_97354


namespace problem_solution_l973_97307

theorem problem_solution (n : ℕ) (h1 : n > 30) (h2 : (4 * n - 1) ∣ (2002 * n)) : n = 36 := by
  sorry

end problem_solution_l973_97307


namespace cosine_product_equals_half_to_seventh_power_l973_97319

theorem cosine_product_equals_half_to_seventh_power :
  (Real.cos (12 * π / 180)) *
  (Real.cos (24 * π / 180)) *
  (Real.cos (36 * π / 180)) *
  (Real.cos (48 * π / 180)) *
  (Real.cos (60 * π / 180)) *
  (Real.cos (72 * π / 180)) *
  (Real.cos (84 * π / 180)) = (1/2)^7 := by
  sorry

end cosine_product_equals_half_to_seventh_power_l973_97319


namespace triangle_side_length_l973_97370

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  b = 2 :=
by sorry

end triangle_side_length_l973_97370


namespace exists_number_with_reversed_digits_and_middle_zero_l973_97355

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  d : ℕ
  e : ℕ
  f : ℕ
  d_lt_base : d < base
  e_lt_base : e < base
  f_lt_base : f < base

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.d * base^2 + n.e * base + n.f

theorem exists_number_with_reversed_digits_and_middle_zero :
  ∃ (n : ThreeDigitNumber 6) (m : ThreeDigitNumber 8),
    to_nat n = to_nat m ∧
    n.d = m.f ∧
    n.e = 0 ∧
    n.e = m.e ∧
    n.f = m.d :=
sorry

end exists_number_with_reversed_digits_and_middle_zero_l973_97355


namespace nancy_next_month_games_l973_97346

/-- The number of football games Nancy plans to attend next month -/
def games_next_month (games_this_month games_last_month total_games : ℕ) : ℕ :=
  total_games - (games_this_month + games_last_month)

/-- Proof that Nancy plans to attend 7 games next month -/
theorem nancy_next_month_games :
  games_next_month 9 8 24 = 7 := by
  sorry

end nancy_next_month_games_l973_97346


namespace smallest_multiple_l973_97397

theorem smallest_multiple (x : ℕ) : x = 256 ↔ 
  (x > 0 ∧ 900 * x % 1024 = 0 ∧ ∀ y : ℕ, 0 < y ∧ y < x → 900 * y % 1024 ≠ 0) :=
by sorry

end smallest_multiple_l973_97397


namespace crease_lines_form_annulus_l973_97359

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the folding operation
def Fold (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
    p = (center.1 + t * (point.1 - center.1), center.2 + t * (point.2 - center.2))}

-- Define the set of all crease lines
def CreaseLines (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  ⋃ (point ∈ Circle center radius), Fold center radius point

-- Define the annulus
def Annulus (center : ℝ × ℝ) (innerRadius outerRadius : ℝ) : Set (ℝ × ℝ) :=
  {p | innerRadius^2 ≤ (p.1 - center.1)^2 + (p.2 - center.2)^2 ∧ 
       (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ outerRadius^2}

-- The theorem to prove
theorem crease_lines_form_annulus (center : ℝ × ℝ) :
  CreaseLines center 10 = Annulus center 5 10 := by sorry

end crease_lines_form_annulus_l973_97359


namespace window_installation_time_l973_97344

theorem window_installation_time (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) :
  total_windows = 10 →
  installed_windows = 6 →
  time_per_window = 5 →
  (total_windows - installed_windows) * time_per_window = 20 :=
by
  sorry

end window_installation_time_l973_97344


namespace ratio_a_to_c_l973_97380

theorem ratio_a_to_c (a b c : ℚ) 
  (h1 : a / b = 8 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 8 / 15 := by
  sorry

end ratio_a_to_c_l973_97380


namespace positive_integer_pairs_satisfying_equation_l973_97312

theorem positive_integer_pairs_satisfying_equation :
  ∀ x y : ℕ+, 
    (x * y * Nat.gcd x.val y.val = x + y + (Nat.gcd x.val y.val)^2) ↔ 
    ((x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2)) := by
  sorry

end positive_integer_pairs_satisfying_equation_l973_97312


namespace quadratic_solution_difference_squared_l973_97395

theorem quadratic_solution_difference_squared (α β : ℝ) : 
  α ≠ β ∧ 
  α^2 - 3*α + 2 = 0 ∧ 
  β^2 - 3*β + 2 = 0 → 
  (α - β)^2 = 1 := by sorry

end quadratic_solution_difference_squared_l973_97395


namespace rectangle_area_diagonal_relation_l973_97390

theorem rectangle_area_diagonal_relation (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) (h3 : d = 13) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 20 / 41 := by
  sorry

end rectangle_area_diagonal_relation_l973_97390


namespace composite_numbers_equal_if_same_main_divisors_l973_97326

/-- Main divisors of a natural number -/
def main_divisors (n : ℕ) : Set ℕ :=
  {d ∈ Nat.divisors n | d ≠ n ∧ d > 1 ∧ ∀ e ∈ Nat.divisors n, e ≠ n → e ≤ d}

/-- Two largest elements of a finite set of natural numbers -/
def two_largest (s : Set ℕ) : Set ℕ :=
  {x ∈ s | ∀ y ∈ s, y ≤ x ∨ ∃ z ∈ s, z ≠ x ∧ z ≠ y ∧ y ≤ z}

theorem composite_numbers_equal_if_same_main_divisors
  (a b : ℕ) (ha : ¬Nat.Prime a) (hb : ¬Nat.Prime b)
  (h : two_largest (main_divisors a) = two_largest (main_divisors b)) :
  a = b := by
  sorry

end composite_numbers_equal_if_same_main_divisors_l973_97326


namespace time_addition_theorem_l973_97328

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time, wrapping around in 12-hour format -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Calculates the sum of hour, minute, and second components of a time -/
def sumComponents (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_addition_theorem :
  let startTime := Time.mk 3 0 0
  let finalTime := addTime startTime 315 58 36
  finalTime = Time.mk 6 58 36 ∧ sumComponents finalTime = 100 := by sorry

end time_addition_theorem_l973_97328


namespace wall_length_calculation_l973_97342

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 21 →
  wall_width = 28 →
  (mirror_side ^ 2) * 2 = wall_width * (31.5 : ℝ) := by
  sorry

end wall_length_calculation_l973_97342


namespace tank_capacity_l973_97333

theorem tank_capacity (T : ℚ) : 
  (3/4 : ℚ) * T + 4 = (7/8 : ℚ) * T → T = 32 := by
  sorry

end tank_capacity_l973_97333


namespace pole_height_l973_97329

/-- Represents the geometry of a telephone pole with a supporting cable -/
structure TelephonePole where
  /-- Height of the pole in meters -/
  height : ℝ
  /-- Distance from the base of the pole to where the cable touches the ground, in meters -/
  cable_ground_distance : ℝ
  /-- Height of a person touching the cable, in meters -/
  person_height : ℝ
  /-- Distance from the base of the pole to where the person stands, in meters -/
  person_distance : ℝ

/-- Theorem stating the height of the telephone pole -/
theorem pole_height (pole : TelephonePole) 
  (h1 : pole.cable_ground_distance = 3)
  (h2 : pole.person_height = 1.5)
  (h3 : pole.person_distance = 2.5)
  : pole.height = 9 := by
  sorry

/-- Main statement combining the structure and theorem -/
def main : Prop :=
  ∃ pole : TelephonePole, 
    pole.cable_ground_distance = 3 ∧
    pole.person_height = 1.5 ∧
    pole.person_distance = 2.5 ∧
    pole.height = 9

end pole_height_l973_97329


namespace combined_bus_ride_length_l973_97362

theorem combined_bus_ride_length 
  (vince_ride : ℝ) 
  (zachary_ride : ℝ) 
  (alexandra_ride : ℝ) 
  (h1 : vince_ride = 0.62) 
  (h2 : zachary_ride = 0.5) 
  (h3 : alexandra_ride = 0.72) : 
  vince_ride + zachary_ride + alexandra_ride = 1.84 := by
sorry

end combined_bus_ride_length_l973_97362


namespace representatives_selection_theorem_l973_97351

def number_of_students : ℕ := 6
def number_of_representatives : ℕ := 3

def select_representatives (n m : ℕ) (at_least_one_from_set : ℕ) : ℕ :=
  sorry

theorem representatives_selection_theorem :
  select_representatives number_of_students number_of_representatives 2 = 96 :=
sorry

end representatives_selection_theorem_l973_97351


namespace asha_win_probability_l973_97387

theorem asha_win_probability (lose_prob tie_prob : ℚ) 
  (lose_prob_val : lose_prob = 5/12)
  (tie_prob_val : tie_prob = 1/6)
  (total_prob : lose_prob + tie_prob + (1 - lose_prob - tie_prob) = 1) :
  1 - lose_prob - tie_prob = 5/12 := by
  sorry

end asha_win_probability_l973_97387


namespace exponential_equation_solution_l973_97367

theorem exponential_equation_solution (x y : ℝ) :
  (5 : ℝ) ^ (x + y + 4) = 625 ^ x → y = 3 * x - 4 := by
  sorry

end exponential_equation_solution_l973_97367


namespace trajectory_of_P_max_distance_to_L_min_distance_to_L_l973_97311

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M on circle C
def M (x₀ y₀ : ℝ) : Prop := C x₀ y₀

-- Define point N
def N : ℝ × ℝ := (4, 0)

-- Define point P as midpoint of MN
def P (x y x₀ y₀ : ℝ) : Prop := x = (x₀ + 4) / 2 ∧ y = y₀ / 2

-- Theorem for the trajectory of P
theorem trajectory_of_P (x y : ℝ) : 
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → (x - 2)^2 + y^2 = 1 :=
sorry

-- Define the line L: 3x + 4y - 26 = 0
def L (x y : ℝ) : Prop := 3*x + 4*y - 26 = 0

-- Theorem for maximum distance
theorem max_distance_to_L (x y : ℝ) :
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → 
  (∀ x' y', (∃ x₀' y₀', M x₀' y₀' ∧ P x' y' x₀' y₀') → 
    |3*x + 4*y - 26| / Real.sqrt 25 ≤ 5) :=
sorry

-- Theorem for minimum distance
theorem min_distance_to_L (x y : ℝ) :
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → 
  (∀ x' y', (∃ x₀' y₀', M x₀' y₀' ∧ P x' y' x₀' y₀') → 
    |3*x + 4*y - 26| / Real.sqrt 25 ≥ 3) :=
sorry

end trajectory_of_P_max_distance_to_L_min_distance_to_L_l973_97311


namespace restaurant_bill_l973_97358

theorem restaurant_bill (n : ℕ) (extra : ℝ) (discount : ℝ) (original_bill : ℝ) :
  n = 10 →
  extra = 3 →
  discount = 10 →
  (n - 1) * ((original_bill - discount) / n + extra) = original_bill - discount →
  original_bill = 180 := by
sorry

end restaurant_bill_l973_97358


namespace polygon_sides_l973_97345

/-- 
A polygon has n sides. 
The sum of its interior angles is (n - 2) * 180°.
The sum of its exterior angles is 360°.
The sum of its interior angles is three times the sum of its exterior angles.
Prove that n = 8.
-/
theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end polygon_sides_l973_97345


namespace perfect_cube_units_digits_l973_97394

theorem perfect_cube_units_digits : 
  ∃ (S : Finset ℕ), (∀ n : ℕ, ∃ k : ℕ, n ^ 3 % 10 ∈ S) ∧ S.card = 10 :=
by sorry

end perfect_cube_units_digits_l973_97394


namespace g_composition_result_l973_97315

/-- Definition of the function g for complex numbers -/
noncomputable def g (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 - 1 else z^3 + 1

/-- Theorem stating the result of g(g(g(g(2+i)))) -/
theorem g_composition_result :
  g (g (g (g (2 + Complex.I)))) = (-64555 + 70232 * Complex.I)^3 + 1 := by
  sorry

end g_composition_result_l973_97315


namespace prism_lateral_faces_are_parallelograms_l973_97386

/-- A prism is a polyhedron with two congruent and parallel faces (called bases) 
    and all other faces (called lateral faces) are parallelograms. -/
structure Prism where
  -- We don't need to define the internal structure for this problem
  mk :: 

/-- A face of a polyhedron -/
structure Face where
  -- We don't need to define the internal structure for this problem
  mk ::

/-- Predicate to check if a face is a lateral face of a prism -/
def is_lateral_face (p : Prism) (f : Face) : Prop :=
  -- Definition omitted for brevity
  sorry

/-- Predicate to check if a face is a parallelogram -/
def is_parallelogram (f : Face) : Prop :=
  -- Definition omitted for brevity
  sorry

theorem prism_lateral_faces_are_parallelograms (p : Prism) :
  ∀ (f : Face), is_lateral_face p f → is_parallelogram f := by
  sorry

end prism_lateral_faces_are_parallelograms_l973_97386


namespace percentage_of_sikh_boys_l973_97373

theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percentage : ℚ) (hindu_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percentage = 44 / 100 →
  hindu_percentage = 14 / 100 →
  other_boys = 272 →
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1 / 10 := by
  sorry

end percentage_of_sikh_boys_l973_97373


namespace age_problem_l973_97361

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by
sorry

end age_problem_l973_97361


namespace arithmetic_sequence_properties_l973_97383

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  h1 : a 3 + a 4 = 15
  h2 : a 2 * a 5 = 54
  h3 : d < 0

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 11 - n) ∧
  (∃ n : ℕ, sum_n seq n = 55) ∧
  (∀ n : ℕ, sum_n seq n ≤ 55) ∧
  (sum_n seq 11 = 55) :=
sorry

end arithmetic_sequence_properties_l973_97383


namespace solution_value_l973_97317

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 6*x + 11 = 24

-- Define a and b as solutions to the equation
def a_b_solutions (a b : ℝ) : Prop :=
  quadratic_equation a ∧ quadratic_equation b ∧ a ≥ b

-- Theorem statement
theorem solution_value (a b : ℝ) (h : a_b_solutions a b) :
  3*a - b = 6 + 4*Real.sqrt 22 :=
by sorry

end solution_value_l973_97317


namespace rice_left_calculation_l973_97377

/-- Calculates the amount of rice left in grams after cooking -/
def rice_left (initial : ℚ) (morning_cooked : ℚ) (evening_fraction : ℚ) : ℚ :=
  let remaining_after_morning := initial - morning_cooked
  let evening_cooked := remaining_after_morning * evening_fraction
  let final_remaining := remaining_after_morning - evening_cooked
  final_remaining * 1000  -- Convert to grams

/-- Theorem stating the amount of rice left after cooking -/
theorem rice_left_calculation :
  rice_left 10 (9/10 * 10) (1/4) = 750 := by
  sorry

#eval rice_left 10 (9/10 * 10) (1/4)

end rice_left_calculation_l973_97377


namespace prime_difference_divisibility_l973_97338

theorem prime_difference_divisibility (n : ℕ) : 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n ∣ (p - q) := by
  sorry

end prime_difference_divisibility_l973_97338


namespace max_permissible_length_l973_97389

/-- A word is permissible if all adjacent letters are different and 
    it's not possible to obtain a word of the form abab by deleting letters, 
    where a and b are different. -/
def Permissible (word : List Char) (alphabet : List Char) : Prop := sorry

/-- The maximum length of a permissible word for an alphabet with n letters -/
def MaxPermissibleLength (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum length of a permissible word for an alphabet with n letters is 2n - 1 -/
theorem max_permissible_length (n : ℕ) : MaxPermissibleLength n = 2 * n - 1 := by
  sorry

end max_permissible_length_l973_97389


namespace perfect_square_polynomial_l973_97343

theorem perfect_square_polynomial (n : ℤ) : 
  (∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2) ↔ n = 10 := by
sorry

end perfect_square_polynomial_l973_97343


namespace digit_sum_property_l973_97324

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_number (A : ℕ) : Prop :=
  10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)

def solution_set : Finset ℕ := {11, 12, 13, 20, 21, 22, 30, 31, 50}

theorem digit_sum_property :
  ∀ A : ℕ, is_valid_number A ↔ A ∈ solution_set :=
sorry

end digit_sum_property_l973_97324


namespace final_solid_properties_l973_97360

/-- Represents a solid shape with faces, edges, and vertices -/
structure Solid where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Represents a pyramid attached to a face -/
structure Pyramid where
  base_edges : ℕ

/-- Attaches a pyramid to a solid, updating its properties -/
def attach_pyramid (s : Solid) (p : Pyramid) : Solid :=
  { faces := s.faces + p.base_edges - 1
  , edges := s.edges + p.base_edges
  , vertices := s.vertices + 1 }

/-- The initial triangular prism -/
def initial_prism : Solid :=
  { faces := 5, edges := 9, vertices := 6 }

/-- Pyramid attached to triangular face -/
def triangular_pyramid : Pyramid :=
  { base_edges := 3 }

/-- Pyramid attached to quadrilateral face -/
def quadrilateral_pyramid : Pyramid :=
  { base_edges := 4 }

theorem final_solid_properties :
  let s1 := attach_pyramid initial_prism triangular_pyramid
  let final_solid := attach_pyramid s1 quadrilateral_pyramid
  final_solid.faces = 10 ∧
  final_solid.edges = 16 ∧
  final_solid.vertices = 8 ∧
  final_solid.faces + final_solid.edges + final_solid.vertices = 34 := by
  sorry


end final_solid_properties_l973_97360


namespace divisible_by_five_l973_97391

theorem divisible_by_five (a b : ℕ) : 
  (∃ k : ℕ, a * b = 5 * k) → (∃ m : ℕ, a = 5 * m) ∨ (∃ n : ℕ, b = 5 * n) := by
  sorry

end divisible_by_five_l973_97391


namespace ages_when_violet_reaches_thomas_age_l973_97382

def thomas_age : ℕ := 6
def shay_age : ℕ := thomas_age + 13
def james_age : ℕ := shay_age + 5
def violet_age : ℕ := thomas_age - 3
def emily_age : ℕ := shay_age

def years_until_violet_reaches_thomas_age : ℕ := thomas_age - violet_age

theorem ages_when_violet_reaches_thomas_age :
  james_age + years_until_violet_reaches_thomas_age = 27 ∧
  emily_age + years_until_violet_reaches_thomas_age = 22 :=
by sorry

end ages_when_violet_reaches_thomas_age_l973_97382


namespace cake_mix_buyers_l973_97372

theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) :
  total = 100 →
  muffin = 40 →
  both = 18 →
  neither_prob = 28/100 →
  ∃ cake : ℕ, cake = 50 ∧ cake + muffin - both = (1 - neither_prob) * total := by
  sorry

end cake_mix_buyers_l973_97372


namespace dummies_remainder_l973_97375

theorem dummies_remainder (n : ℕ) (h : n % 9 = 7) : (3 * n) % 9 = 3 := by
  sorry

end dummies_remainder_l973_97375


namespace ball_radius_from_hole_dimensions_l973_97314

/-- Given a spherical ball partially submerged in a frozen surface,
    if the hole left after removing the ball is 30 cm across and 10 cm deep,
    then the radius of the ball is 16.25 cm. -/
theorem ball_radius_from_hole_dimensions (hole_width : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) :
  hole_width = 30 →
  hole_depth = 10 →
  ball_radius = 16.25 := by
  sorry

end ball_radius_from_hole_dimensions_l973_97314


namespace first_player_winning_strategy_exists_l973_97398

/-- Represents the state of the chocolate bar game -/
structure ChocolateGame where
  rows : Nat
  cols : Nat

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- The result of the game -/
structure GameResult where
  firstPlayerPieces : Nat
  secondPlayerPieces : Nat

/-- The strategy function type -/
def Strategy := ChocolateGame → Player → Option (Nat × Nat)

/-- Simulates the game given strategies for both players -/
def playGame (firstStrategy : Strategy) (secondStrategy : Strategy) : GameResult :=
  sorry

/-- The main theorem stating the existence of a winning strategy for the first player -/
theorem first_player_winning_strategy_exists :
  ∃ (strategy : Strategy),
    let result := playGame strategy (λ _ _ ↦ none)
    result.firstPlayerPieces ≥ result.secondPlayerPieces + 6 := by
  sorry

end first_player_winning_strategy_exists_l973_97398


namespace james_payment_is_six_l973_97336

/-- Calculates James's share of the payment for stickers -/
def jamesPayment (packs : ℕ) (stickersPerPack : ℕ) (stickerCost : ℚ) (friendSharePercent : ℚ) : ℚ :=
  let totalStickers := packs * stickersPerPack
  let totalCost := totalStickers * stickerCost
  totalCost * (1 - friendSharePercent)

/-- Proves that James pays $6 for his share of the stickers -/
theorem james_payment_is_six :
  jamesPayment 4 30 (1/10) (1/2) = 6 := by
  sorry

end james_payment_is_six_l973_97336


namespace smaller_of_reciprocal_and_sine_interval_length_l973_97306

open Real

theorem smaller_of_reciprocal_and_sine (x : ℝ) :
  (min (1/x) (sin x) > 1/2) ↔ (π/6 < x ∧ x < 5*π/6) :=
sorry

theorem interval_length : 
  (5*π/6 - π/6 : ℝ) = 2*π/3 :=
sorry

end smaller_of_reciprocal_and_sine_interval_length_l973_97306


namespace tan_alpha_value_l973_97384

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/3) = 2 * Real.sqrt 3) :
  Real.tan α = Real.sqrt 3 / 7 := by
  sorry

end tan_alpha_value_l973_97384


namespace min_value_product_l973_97332

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 38 := by
  sorry

end min_value_product_l973_97332


namespace fraction_subtraction_equality_l973_97350

theorem fraction_subtraction_equality : 
  -1/8 - (1 + 1/3) - (-5/8) - (4 + 2/3) = -(11/2) := by sorry

end fraction_subtraction_equality_l973_97350


namespace triangle_extension_l973_97327

/-- Triangle extension theorem -/
theorem triangle_extension (n : ℕ) (a b c t S : ℝ) 
  (h_n : n > 0)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_area : t > 0)
  (h_S : S = a^2 + b^2 + c^2)
  (t_i : Fin (n-1) → ℝ)
  (S_i : Fin (n-1) → ℝ)
  (h_t_i : ∀ i, t_i i > 0)
  (h_S_i : ∀ i, S_i i > 0) :
  (∃ k : ℝ, 
    (S + (Finset.sum Finset.univ S_i) = n^3 * S) ∧ 
    (t + (Finset.sum Finset.univ t_i) = n^3 * t) ∧ 
    (∀ i : Fin (n-1), S_i i / t_i i = k) ∧ 
    (S / t = k)) := by
  sorry

end triangle_extension_l973_97327


namespace f_properties_l973_97323

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x ^ 2)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∃ M, ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ M) ∧
  (∃ m, ∀ x ∈ Set.Icc 0 (Real.pi / 2), m ≤ f x) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ f x₁) ∧
  (∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x₂ ≤ f x) :=
by sorry

end f_properties_l973_97323


namespace inequality_and_equality_condition_l973_97318

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ+) :
  let lhs := (a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2
  let rhs := 4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁)
  (lhs ≥ rhs) ∧ 
  (lhs = rhs ↔ (a₁ : ℚ) / b₁ = (a₂ : ℚ) / b₂ ∧ (a₂ : ℚ) / b₂ = (a₃ : ℚ) / b₃) :=
by sorry

end inequality_and_equality_condition_l973_97318


namespace prime_factorization_sum_l973_97300

theorem prime_factorization_sum (w x y z k : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^k = 2310 → 2*w + 3*x + 5*y + 7*z + 11*k = 28 := by
  sorry

end prime_factorization_sum_l973_97300


namespace hyperbola_eccentricity_l973_97356

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  asymptotes_tangent_to_parabola : Bool

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x - 1

-- Define the eccentricity of a hyperbola
def eccentricity (h : Hyperbola) : ℝ := sorry

-- Theorem statement
theorem hyperbola_eccentricity (C : Hyperbola) :
  C.center = (0, 0) →
  C.foci_on_x_axis = true →
  C.asymptotes_tangent_to_parabola = true →
  eccentricity C = Real.sqrt 5 / 2 := by sorry

end hyperbola_eccentricity_l973_97356


namespace total_age_proof_l973_97339

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 28 years old
  Prove that the total of their ages is 72 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 28 → a = b + 2 → b = 2 * c → a + b + c = 72 := by
  sorry

end total_age_proof_l973_97339


namespace intersection_condition_l973_97368

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (2 * p.1 - p.1^2)}
def N (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * (p.1 + 1)}

-- State the theorem
theorem intersection_condition (k : ℝ) :
  (∃ p, p ∈ M ∩ N k) ↔ 0 ≤ k ∧ k ≤ Real.sqrt 3 / 3 := by
  sorry

end intersection_condition_l973_97368


namespace fraction_sum_equality_l973_97352

theorem fraction_sum_equality (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1 / (b - c) + 1 / (c - a) + 1 / (a - b) := by
  sorry

end fraction_sum_equality_l973_97352


namespace intersection_not_roots_l973_97325

theorem intersection_not_roots : ∀ x : ℝ, 
  (x = x - 3 → x^2 - 3*x ≠ 0) :=
by
  sorry

end intersection_not_roots_l973_97325


namespace seven_balls_two_boxes_l973_97337

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distributeDistinguishableBalls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 128 ways to distribute 7 distinguishable balls into 2 distinguishable boxes -/
theorem seven_balls_two_boxes :
  distributeDistinguishableBalls 7 2 = 128 := by
  sorry

end seven_balls_two_boxes_l973_97337


namespace distinct_pairs_of_twelve_students_l973_97385

-- Define the number of students
def num_students : ℕ := 12

-- Define the function to calculate the number of distinct pairs
def num_distinct_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem distinct_pairs_of_twelve_students :
  num_distinct_pairs num_students = 66 := by
  sorry

end distinct_pairs_of_twelve_students_l973_97385


namespace function_properties_l973_97303

-- Define the function f(x)
def f (x : ℝ) : ℝ := |3*x + 3| - |x - 5|

-- Define the solution set M
def M : Set ℝ := {x | f x > 0}

-- State the theorem
theorem function_properties :
  (M = {x | x < -4 ∨ x > 1/2}) ∧
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 →
    1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3/4) := by
  sorry

end function_properties_l973_97303


namespace slope_at_one_l973_97399

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem slope_at_one (α : ℝ) :
  (deriv f 1 = α) →
  (Real.cos α / (Real.sin α - 4 * Real.cos α) = -1) :=
by sorry

end slope_at_one_l973_97399


namespace fixed_points_for_specific_values_range_for_two_distinct_fixed_points_l973_97379

/-- The function f(x) = ax^2 + (b+1)x + b - 2 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

/-- A point x is a fixed point of f if f(x) = x -/
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

theorem fixed_points_for_specific_values :
  ∀ x : ℝ, is_fixed_point 2 (-2) x ↔ x = -1 ∨ x = 2 := by sorry

theorem range_for_two_distinct_fixed_points :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ is_fixed_point a b x ∧ is_fixed_point a b y) →
  (0 < a ∧ a < 2) := by sorry

end fixed_points_for_specific_values_range_for_two_distinct_fixed_points_l973_97379


namespace fraction_equals_zero_l973_97334

theorem fraction_equals_zero (x : ℝ) :
  (2*x - 4) / (x + 1) = 0 ∧ x + 1 ≠ 0 → x = 2 := by
  sorry

end fraction_equals_zero_l973_97334


namespace male_students_in_school_l973_97374

/-- Represents the number of students in a school population --/
structure SchoolPopulation where
  total : Nat
  sample : Nat
  females_in_sample : Nat

/-- Calculates the number of male students in the school --/
def male_students (pop : SchoolPopulation) : Nat :=
  pop.total - (pop.total * pop.females_in_sample / pop.sample)

/-- Theorem stating the number of male students in the given scenario --/
theorem male_students_in_school (pop : SchoolPopulation) 
  (h1 : pop.total = 1600)
  (h2 : pop.sample = 200)
  (h3 : pop.females_in_sample = 95) :
  male_students pop = 840 := by
  sorry

#eval male_students { total := 1600, sample := 200, females_in_sample := 95 }

end male_students_in_school_l973_97374


namespace b_n_equals_c_1_l973_97301

theorem b_n_equals_c_1 (n : ℕ) (a : ℕ → ℝ) (b c : ℕ → ℝ)
  (h_positive : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h_b_1 : b 1 = a 1)
  (h_b_2 : b 2 = max (a 1) (a 2))
  (h_b_i : ∀ i, 3 ≤ i → i ≤ n → b i = max (b (i - 1)) (b (i - 2) + a i))
  (h_c_n : c n = a n)
  (h_c_n_1 : c (n - 1) = max (a n) (a (n - 1)))
  (h_c_i : ∀ i, 1 ≤ i → i ≤ n - 2 → c i = max (c (i + 1)) (c (i + 2) + a i)) :
  b n = c 1 := by
  sorry


end b_n_equals_c_1_l973_97301


namespace probability_of_drawing_item_l973_97363

/-- Proves that the probability of drawing each item in a sample is 1/5 given the total number of components and sample size -/
theorem probability_of_drawing_item 
  (total_components : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_components = 100) 
  (h2 : sample_size = 20) : 
  (sample_size : ℚ) / (total_components : ℚ) = 1 / 5 := by
  sorry

end probability_of_drawing_item_l973_97363


namespace average_height_combined_groups_l973_97365

theorem average_height_combined_groups (n₁ n₂ : ℕ) (h₁ h₂ : ℝ) :
  n₁ = 35 →
  n₂ = 25 →
  h₁ = 22 →
  h₂ = 18 →
  (n₁ * h₁ + n₂ * h₂) / (n₁ + n₂ : ℝ) = 20.33 :=
by sorry

end average_height_combined_groups_l973_97365


namespace median_in_70_74_interval_l973_97366

/-- Represents a score interval with its lower bound and number of students -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (num_students : ℕ)

/-- Finds the interval containing the median score -/
def median_interval (intervals : List ScoreInterval) : Option ScoreInterval :=
  sorry

theorem median_in_70_74_interval :
  let intervals : List ScoreInterval := [
    ⟨55, 4⟩,
    ⟨60, 8⟩,
    ⟨65, 15⟩,
    ⟨70, 20⟩,
    ⟨75, 18⟩,
    ⟨80, 10⟩
  ]
  let total_students : ℕ := 75
  median_interval intervals = some ⟨70, 20⟩ := by
    sorry

end median_in_70_74_interval_l973_97366


namespace abs_neg_six_l973_97371

theorem abs_neg_six : |(-6 : ℤ)| = 6 := by sorry

end abs_neg_six_l973_97371


namespace tenth_term_of_geometric_sequence_l973_97396

/-- Given a geometric sequence with first term 2 and common ratio 5/3,
    the 10th term is equal to 3906250/19683. -/
theorem tenth_term_of_geometric_sequence :
  let a₁ : ℚ := 2
  let r : ℚ := 5/3
  let n : ℕ := 10
  let aₙ : ℕ → ℚ := λ k => a₁ * r^(k - 1)
  aₙ n = 3906250/19683 := by
sorry

end tenth_term_of_geometric_sequence_l973_97396


namespace hyperbola_focus_l973_97302

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 3 - x^2 / 6 = 1

/-- Definition of a focus of the hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (c : ℝ), c^2 = 3 + 6 ∧ (x = 0 ∧ (y = c ∨ y = -c))

/-- Theorem: One focus of the hyperbola has coordinates (0, 3) -/
theorem hyperbola_focus : ∃ (x y : ℝ), hyperbola_equation x y ∧ is_focus x y ∧ x = 0 ∧ y = 3 := by
  sorry

end hyperbola_focus_l973_97302


namespace min_value_of_function_l973_97378

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x + (x + 1)⁻¹ ≥ 1 ∧ (x + (x + 1)⁻¹ = 1 ↔ x = 0) := by
  sorry

end min_value_of_function_l973_97378


namespace stream_to_meadow_distance_l973_97321

/-- Given a hiking trip with known distances, prove the distance between two points -/
theorem stream_to_meadow_distance 
  (total_distance : ℝ)
  (car_to_stream : ℝ)
  (meadow_to_campsite : ℝ)
  (h1 : total_distance = 0.7)
  (h2 : car_to_stream = 0.2)
  (h3 : meadow_to_campsite = 0.1) :
  total_distance - car_to_stream - meadow_to_campsite = 0.4 := by
  sorry

end stream_to_meadow_distance_l973_97321


namespace total_age_is_32_l973_97320

-- Define the ages of a, b, and c
def age_b : ℕ := 12
def age_a : ℕ := age_b + 2
def age_c : ℕ := age_b / 2

-- Theorem to prove
theorem total_age_is_32 : age_a + age_b + age_c = 32 := by
  sorry


end total_age_is_32_l973_97320


namespace equation_a_is_quadratic_l973_97376

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific equation we want to prove is quadratic
def f (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem equation_a_is_quadratic : is_quadratic_equation f := by
  sorry

end equation_a_is_quadratic_l973_97376


namespace gcd_n_cube_plus_25_and_n_plus_6_l973_97310

theorem gcd_n_cube_plus_25_and_n_plus_6 (n : ℕ) (h : n > 2^5) :
  Nat.gcd (n^3 + 5^2) (n + 6) = 1 := by
  sorry

end gcd_n_cube_plus_25_and_n_plus_6_l973_97310


namespace sine_sqrt_equality_l973_97330

theorem sine_sqrt_equality (a : ℝ) (h1 : a ≥ 0) :
  (∀ x : ℝ, x ≥ 0 → Real.sin (Real.sqrt (x + a)) = Real.sin (Real.sqrt x)) →
  a = 0 := by
  sorry

end sine_sqrt_equality_l973_97330


namespace percentage_of_truth_speakers_l973_97305

theorem percentage_of_truth_speakers (L B : ℝ) (h1 : L = 0.2) (h2 : B = 0.1) 
  (h3 : L + B + (L + B - B) = 0.4) : L + B - B = 0.3 :=
by sorry

end percentage_of_truth_speakers_l973_97305


namespace point_A_coordinates_l973_97393

/-- Given a point A with coordinates (2a-9, 1-2a), prove that if A is moved 5 units
    to the right and lands on the y-axis, then its new coordinates are (-5, -3) -/
theorem point_A_coordinates (a : ℝ) :
  let initial_A : ℝ × ℝ := (2*a - 9, 1 - 2*a)
  let moved_A : ℝ × ℝ := (2*a - 4, 1 - 2*a)  -- Moved 5 units to the right
  moved_A.1 = 0 →  -- Lands on y-axis
  moved_A = (-5, -3) :=
by sorry

end point_A_coordinates_l973_97393


namespace total_books_is_42_l973_97304

-- Define the initial number of books on each shelf
def initial_books_shelf1 : ℕ := 9
def initial_books_shelf2 : ℕ := 0
def initial_books_shelf3 : ℕ := initial_books_shelf1 + (initial_books_shelf1 * 3 / 10)
def initial_books_shelf4 : ℕ := initial_books_shelf3 / 2

-- Define the number of books added to each shelf
def added_books_shelf1 : ℕ := 10
def added_books_shelf4 : ℕ := 5

-- Define the total number of books after additions
def total_books : ℕ := 
  (initial_books_shelf1 + added_books_shelf1) +
  initial_books_shelf2 +
  initial_books_shelf3 +
  (initial_books_shelf4 + added_books_shelf4)

-- Theorem statement
theorem total_books_is_42 : total_books = 42 := by
  sorry

end total_books_is_42_l973_97304


namespace f_plus_g_at_one_equals_two_l973_97348

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_plus_g_at_one_equals_two
  (f g : ℝ → ℝ)
  (h_even : isEven f)
  (h_odd : isOdd g)
  (h_eq : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 2 := by
sorry

end f_plus_g_at_one_equals_two_l973_97348


namespace binomial_10_3_l973_97381

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem binomial_10_3 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_l973_97381
