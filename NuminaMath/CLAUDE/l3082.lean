import Mathlib

namespace tan_4050_undefined_l3082_308294

theorem tan_4050_undefined :
  ∀ x : ℝ, Real.tan (4050 * π / 180) = x → False :=
by
  sorry

end tan_4050_undefined_l3082_308294


namespace prime_binomial_divisibility_l3082_308225

theorem prime_binomial_divisibility (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, m / 3 ≤ n ∧ n ≤ m / 2 → n ∣ Nat.choose n (m - 2*n)) ↔ Nat.Prime m :=
sorry

end prime_binomial_divisibility_l3082_308225


namespace simplify_fraction_l3082_308286

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + 2 * Real.sqrt 18) = 5 * Real.sqrt 2 / 34 := by
  sorry

end simplify_fraction_l3082_308286


namespace marble_arrangement_theorem_l3082_308289

/-- Represents the color of a marble -/
inductive Color
| Blue
| Yellow

/-- Represents an arrangement of marbles -/
def Arrangement := List Color

/-- Counts the number of adjacent pairs with the same color -/
def countSameColorPairs (arr : Arrangement) : Nat :=
  sorry

/-- Counts the number of adjacent pairs with different colors -/
def countDifferentColorPairs (arr : Arrangement) : Nat :=
  sorry

/-- Checks if an arrangement satisfies the equal pairs condition -/
def isValidArrangement (arr : Arrangement) : Prop :=
  countSameColorPairs arr = countDifferentColorPairs arr

/-- Counts the number of blue marbles in an arrangement -/
def countBlueMarbles (arr : Arrangement) : Nat :=
  sorry

/-- Counts the number of yellow marbles in an arrangement -/
def countYellowMarbles (arr : Arrangement) : Nat :=
  sorry

theorem marble_arrangement_theorem :
  ∃ (validArrangements : List Arrangement),
    (∀ arr ∈ validArrangements, isValidArrangement arr) ∧
    (∀ arr ∈ validArrangements, countBlueMarbles arr = 4) ∧
    (∀ arr ∈ validArrangements, countYellowMarbles arr ≤ 11) ∧
    (validArrangements.length = 35) :=
  sorry

end marble_arrangement_theorem_l3082_308289


namespace two_zeros_twelve_divisors_l3082_308254

def endsWithTwoZeros (n : ℕ) : Prop := n % 100 = 0

def countDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem two_zeros_twelve_divisors :
  ∀ n : ℕ, endsWithTwoZeros n ∧ countDivisors n = 12 ↔ n = 200 ∨ n = 500 := by
  sorry

end two_zeros_twelve_divisors_l3082_308254


namespace triangle_angle_measure_l3082_308206

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 80 →  -- Measure of angle D is 80 degrees
  E = 4 * F + 10 →  -- Measure of angle E is 10 degrees more than four times the measure of angle F
  D + E + F = 180 →  -- Sum of angles in a triangle is 180 degrees
  F = 18 :=  -- Measure of angle F is 18 degrees
by sorry

end triangle_angle_measure_l3082_308206


namespace exponent_division_l3082_308259

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^5 / a = a^4 := by
  sorry

end exponent_division_l3082_308259


namespace ellipse_equation_l3082_308253

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the general form of the ellipse
def ellipse (x y A B : ℝ) : Prop := (x^2 / A) + (y^2 / B) = 1

-- State the theorem
theorem ellipse_equation 
  (x y A B : ℝ) 
  (h1 : A > 0) 
  (h2 : B > 0) 
  (h3 : ∃ (xf yf xv yv : ℝ), 
    line xf yf ∧ line xv yv ∧ 
    ellipse xf yf A B ∧ ellipse xv yv A B ∧ 
    ((xf = 0 ∧ xv ≠ 0) ∨ (yf = 0 ∧ yv ≠ 0))) :
  ((A = 5 ∧ B = 4) ∨ (A = 1 ∧ B = 5)) :=
sorry

end ellipse_equation_l3082_308253


namespace line_through_points_l3082_308208

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_through_points :
  let p1 : Point := ⟨2, 10⟩
  let p2 : Point := ⟨6, 26⟩
  let p3 : Point := ⟨10, 42⟩
  let p4 : Point := ⟨45, 182⟩
  collinear p1 p2 p3 → collinear p1 p2 p4 :=
by
  sorry

end line_through_points_l3082_308208


namespace sequence_a_formula_l3082_308274

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_2 : S 2 = 4

axiom a_recursive (n : ℕ) : n ≥ 1 → sequence_a (n + 1) = 2 * S n + 1

theorem sequence_a_formula (n : ℕ) : n ≥ 1 → sequence_a n = 3^(n - 1) := by sorry

end sequence_a_formula_l3082_308274


namespace intersection_A_complement_B_l3082_308204

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {0, 2} := by
  sorry

end intersection_A_complement_B_l3082_308204


namespace fish_value_calculation_l3082_308200

/-- Calculates the total value of non-spoiled fish after sales, spoilage, and new stock arrival --/
def fish_value (initial_trout initial_bass : ℕ) 
               (sold_trout sold_bass : ℕ) 
               (trout_price bass_price : ℚ) 
               (spoil_trout_ratio spoil_bass_ratio : ℚ)
               (new_trout new_bass : ℕ) : ℚ :=
  let remaining_trout := initial_trout - sold_trout
  let remaining_bass := initial_bass - sold_bass
  let spoiled_trout := ⌊remaining_trout * spoil_trout_ratio⌋
  let spoiled_bass := ⌊remaining_bass * spoil_bass_ratio⌋
  let final_trout := remaining_trout - spoiled_trout + new_trout
  let final_bass := remaining_bass - spoiled_bass + new_bass
  final_trout * trout_price + final_bass * bass_price

/-- The theorem statement --/
theorem fish_value_calculation :
  fish_value 120 80 30 20 5 10 (1/4) (1/3) 150 50 = 1990 := by
  sorry

end fish_value_calculation_l3082_308200


namespace amithab_average_expenditure_l3082_308245

/-- Given Amithab's monthly expenses, prove the average expenditure for February to July. -/
theorem amithab_average_expenditure
  (jan_expense : ℕ)
  (jan_to_jun_avg : ℕ)
  (jul_expense : ℕ)
  (h1 : jan_expense = 1200)
  (h2 : jan_to_jun_avg = 4200)
  (h3 : jul_expense = 1500) :
  (6 * jan_to_jun_avg - jan_expense + jul_expense) / 6 = 4250 :=
by sorry

end amithab_average_expenditure_l3082_308245


namespace roots_and_coefficients_l3082_308229

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots
def is_root (a b c x : ℝ) : Prop := quadratic_equation a b c x

-- Theorem statement
theorem roots_and_coefficients (a b c X₁ X₂ : ℝ) 
  (ha : a ≠ 0) 
  (hX₁ : is_root a b c X₁) 
  (hX₂ : is_root a b c X₂) 
  (hX₁₂ : X₁ ≠ X₂) : 
  (X₁ + X₂ = -b / a) ∧ (X₁ * X₂ = c / a) := by
  sorry

end roots_and_coefficients_l3082_308229


namespace circle_equation_l3082_308297

/-- The equation of a circle with center (2, 1) that shares a common chord with another circle,
    where the chord lies on a line passing through a specific point. -/
theorem circle_equation (x y : ℝ) : 
  ∃ (r : ℝ), 
    -- The first circle has center (2, 1) and radius r
    ((x - 2)^2 + (y - 1)^2 = r^2) ∧
    -- The second circle is described by x^2 + y^2 - 3x = 0
    (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 3*x₀ = 0) ∧
    -- The common chord lies on a line passing through (5, -2)
    (∃ (a b c : ℝ), a*5 + b*(-2) + c = 0 ∧ a*x + b*y + c = 0) →
    -- The equation of the first circle is (x-2)^2 + (y-1)^2 = 4
    (x - 2)^2 + (y - 1)^2 = 4 :=
by
  sorry

end circle_equation_l3082_308297


namespace ln2_greatest_l3082_308298

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem ln2_greatest (h1 : ∀ x y : ℝ, x < y → ln x < ln y) (h2 : (2 : ℝ) < Real.exp 1) :
  ln 2 > (ln 2)^2 ∧ ln 2 > ln (ln 2) ∧ ln 2 > ln (Real.sqrt 2) := by
  sorry


end ln2_greatest_l3082_308298


namespace sum_fraction_denominator_form_main_result_l3082_308216

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_fraction (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (double_factorial (2 * (i + 1))) / (double_factorial (2 * (i + 1) + 1)))

theorem sum_fraction_denominator_form (n : ℕ) :
  ∃ (a b : ℕ), b % 2 = 1 ∧ (sum_fraction n).den = 2^a * b := by sorry

theorem main_result : ∃ (a b : ℕ), b % 2 = 1 ∧
  (sum_fraction 2010).den = 2^a * b ∧ (a * b) / 10 = 0 := by sorry

end sum_fraction_denominator_form_main_result_l3082_308216


namespace P_homogeneous_P_symmetry_P_normalization_P_unique_l3082_308247

/-- A binary polynomial that satisfies the given conditions -/
def P (n : ℕ+) (x y : ℝ) : ℝ := x^(n : ℕ) - y^(n : ℕ)

/-- P is homogeneous of degree n -/
theorem P_homogeneous (n : ℕ+) (t x y : ℝ) :
  P n (t * x) (t * y) = t^(n : ℕ) * P n x y := by sorry

/-- P satisfies the symmetry condition -/
theorem P_symmetry (n : ℕ+) (a b c : ℝ) :
  P n (a + b) c + P n (b + c) a + P n (c + a) b = 0 := by sorry

/-- P satisfies the normalization condition -/
theorem P_normalization (n : ℕ+) :
  P n 1 0 = 1 := by sorry

/-- P is the unique polynomial satisfying all conditions -/
theorem P_unique (n : ℕ+) (Q : ℝ → ℝ → ℝ) 
  (h_homogeneous : ∀ t x y, Q (t * x) (t * y) = t^(n : ℕ) * Q x y)
  (h_symmetry : ∀ a b c, Q (a + b) c + Q (b + c) a + Q (c + a) b = 0)
  (h_normalization : Q 1 0 = 1) :
  ∀ x y, Q x y = P n x y := by sorry

end P_homogeneous_P_symmetry_P_normalization_P_unique_l3082_308247


namespace expected_weekly_rainfall_l3082_308214

/-- The expected total rainfall over a week given daily probabilities --/
theorem expected_weekly_rainfall (p_sun p_light p_heavy : ℝ) 
  (rain_light rain_heavy : ℝ) (days : ℕ) :
  p_sun + p_light + p_heavy = 1 →
  p_sun = 0.5 →
  p_light = 0.2 →
  p_heavy = 0.3 →
  rain_light = 2 →
  rain_heavy = 5 →
  days = 7 →
  (p_sun * 0 + p_light * rain_light + p_heavy * rain_heavy) * days = 13.3 := by
  sorry

end expected_weekly_rainfall_l3082_308214


namespace expand_and_simplify_l3082_308215

theorem expand_and_simplify (x : ℝ) : (7 * x + 5) * 3 * x^2 = 21 * x^3 + 15 * x^2 := by
  sorry

end expand_and_simplify_l3082_308215


namespace jack_and_jill_meeting_point_l3082_308227

/-- Represents the hill run by Jack and Jill -/
structure HillRun where
  length : ℝ
  jack_uphill_speed : ℝ
  jack_downhill_speed : ℝ
  jill_uphill_speed : ℝ
  jill_downhill_speed : ℝ
  jack_pause_time : ℝ
  jack_pause_location : ℝ

/-- Calculates the meeting point of Jack and Jill -/
def meeting_point (h : HillRun) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem jack_and_jill_meeting_point (h : HillRun) 
  (h_length : h.length = 6)
  (h_jack_up : h.jack_uphill_speed = 12)
  (h_jack_down : h.jack_downhill_speed = 18)
  (h_jill_up : h.jill_uphill_speed = 15)
  (h_jill_down : h.jill_downhill_speed = 21)
  (h_pause_time : h.jack_pause_time = 0.25)
  (h_pause_loc : h.jack_pause_location = 3) :
  meeting_point h = 63 / 22 := by
  sorry

end jack_and_jill_meeting_point_l3082_308227


namespace final_amount_after_15_years_l3082_308250

/-- Calculate the final amount using simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the final amount after 15 years -/
theorem final_amount_after_15_years :
  simpleInterest 800000 0.07 15 = 1640000 := by
  sorry

end final_amount_after_15_years_l3082_308250


namespace excluded_students_average_mark_l3082_308242

theorem excluded_students_average_mark
  (N : ℕ)  -- Total number of students
  (A : ℚ)  -- Average mark of all students
  (E : ℕ)  -- Number of excluded students
  (AR : ℚ) -- Average mark of remaining students
  (h1 : N = 25)
  (h2 : A = 80)
  (h3 : E = 5)
  (h4 : AR = 90)
  : ∃ AE : ℚ, AE = 40 ∧ N * A - E * AE = (N - E) * AR :=
sorry

end excluded_students_average_mark_l3082_308242


namespace square_difference_equals_eight_xy_l3082_308246

theorem square_difference_equals_eight_xy (x y A : ℝ) :
  (x + 2*y)^2 = (x - 2*y)^2 + A → A = 8*x*y := by
  sorry

end square_difference_equals_eight_xy_l3082_308246


namespace fixed_monthly_costs_l3082_308293

/-- The fixed monthly costs for a computer manufacturer producing electronic components --/
theorem fixed_monthly_costs 
  (production_cost : ℝ) 
  (shipping_cost : ℝ) 
  (units_sold : ℕ) 
  (selling_price : ℝ) 
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 3)
  (h3 : units_sold = 150)
  (h4 : selling_price = 191.67)
  (h5 : selling_price * (units_sold : ℝ) = (production_cost + shipping_cost) * (units_sold : ℝ) + fixed_costs) :
  fixed_costs = 16300.50 := by
  sorry

#check fixed_monthly_costs

end fixed_monthly_costs_l3082_308293


namespace smallest_triangle_side_l3082_308284

theorem smallest_triangle_side : ∃ (s : ℕ), 
  (s : ℝ) ≥ 4 ∧ 
  (∀ (t : ℕ), (t : ℝ) ≥ 4 → 
    (8.5 + (t : ℝ) > 11.5) ∧
    (8.5 + 11.5 > (t : ℝ)) ∧
    (11.5 + (t : ℝ) > 8.5)) ∧
  (∀ (u : ℕ), (u : ℝ) < 4 → 
    ¬((8.5 + (u : ℝ) > 11.5) ∧
      (8.5 + 11.5 > (u : ℝ)) ∧
      (11.5 + (u : ℝ) > 8.5))) :=
by
  sorry

#check smallest_triangle_side

end smallest_triangle_side_l3082_308284


namespace equal_costs_at_60_guests_unique_equal_cost_guests_l3082_308209

/-- Represents the venues for the prom --/
inductive Venue
| caesars_palace
| venus_hall

/-- Calculates the total cost for a given venue and number of guests --/
def total_cost (v : Venue) (guests : ℕ) : ℚ :=
  match v with
  | Venue.caesars_palace => 800 + 34 * guests
  | Venue.venus_hall => 500 + 39 * guests

/-- Proves that the total costs are equal when there are 60 guests --/
theorem equal_costs_at_60_guests :
  total_cost Venue.caesars_palace 60 = total_cost Venue.venus_hall 60 := by
  sorry

/-- Proves that 60 is the unique number of guests for which costs are equal --/
theorem unique_equal_cost_guests :
  ∀ g : ℕ, total_cost Venue.caesars_palace g = total_cost Venue.venus_hall g ↔ g = 60 := by
  sorry

end equal_costs_at_60_guests_unique_equal_cost_guests_l3082_308209


namespace simplify_fraction_l3082_308262

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  12 * x * y^3 / (9 * x^2 * y^2) = 8 / 9 := by
  sorry

end simplify_fraction_l3082_308262


namespace sum_divided_by_ten_l3082_308282

theorem sum_divided_by_ten : (10 + 20 + 30 + 40) / 10 = 10 := by
  sorry

end sum_divided_by_ten_l3082_308282


namespace angle_A_value_max_area_l3082_308240

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 2 ∧
  (1/2) * t.b * t.c * Real.sin t.A = (Real.sqrt 2 / 2) * (t.c * Real.sin t.C + t.b * Real.sin t.B - t.a * Real.sin t.A)

-- Theorem for the value of angle A
theorem angle_A_value (t : Triangle) (h : triangle_conditions t) : t.A = π / 3 := by
  sorry

-- Theorem for the maximum area of triangle ABC
theorem max_area (t : Triangle) (h : triangle_conditions t) : 
  (∀ t' : Triangle, triangle_conditions t' → (1/2) * t'.b * t'.c * Real.sin t'.A ≤ Real.sqrt 3 / 2) ∧
  (∃ t' : Triangle, triangle_conditions t' ∧ (1/2) * t'.b * t'.c * Real.sin t'.A = Real.sqrt 3 / 2) := by
  sorry

end angle_A_value_max_area_l3082_308240


namespace tunnel_length_l3082_308292

/-- The length of a tunnel given a train passing through it. -/
theorem tunnel_length
  (train_length : ℝ)
  (transit_time : ℝ)
  (train_speed : ℝ)
  (h1 : train_length = 2)
  (h2 : transit_time = 4 / 60)  -- 4 minutes converted to hours
  (h3 : train_speed = 90) :
  train_speed * transit_time - train_length = 4 :=
by sorry

end tunnel_length_l3082_308292


namespace shaded_area_five_circles_plus_one_l3082_308238

/-- The area of the shaded region formed by five circles of radius 5 units
    intersecting at the origin, with an additional circle creating 10 similar sectors. -/
theorem shaded_area_five_circles_plus_one (r : ℝ) (n : ℕ) : 
  r = 5 → n = 10 → (n : ℝ) * (π * r^2 / 4 - r^2 / 2) = 62.5 * π - 125 := by
  sorry

#check shaded_area_five_circles_plus_one

end shaded_area_five_circles_plus_one_l3082_308238


namespace simplify_expression_find_value_evaluate_composite_l3082_308244

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3 * (x - y)^2 - 6 * (x - y)^2 + 2 * (x - y)^2 = -(x - y)^2 := by
  sorry

-- Part 2
theorem find_value (a b : ℝ) (h : a^2 - 2*b = 4) :
  3*a^2 - 6*b - 21 = -9 := by
  sorry

-- Part 3
theorem evaluate_composite (a b c d : ℝ) 
  (h1 : a - 5*b = 3) 
  (h2 : 5*b - 3*c = -5) 
  (h3 : 3*c - d = 10) :
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 := by
  sorry

end simplify_expression_find_value_evaluate_composite_l3082_308244


namespace suitcase_waiting_time_l3082_308232

/-- The number of suitcases loaded onto the plane -/
def total_suitcases : ℕ := 200

/-- The number of suitcases belonging to the businesspeople -/
def business_suitcases : ℕ := 10

/-- The time interval between placing suitcases on the conveyor belt (in seconds) -/
def placement_interval : ℕ := 2

/-- The probability of the businesspeople waiting exactly two minutes for their last suitcase -/
def exact_two_minutes_probability : ℚ :=
  (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases : ℚ)

/-- The expected time (in seconds) the businesspeople will wait for their last suitcase -/
def expected_waiting_time : ℚ := 4020 / 11

theorem suitcase_waiting_time :
  (exact_two_minutes_probability = (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases : ℚ)) ∧
  (expected_waiting_time = 4020 / 11) := by
  sorry

end suitcase_waiting_time_l3082_308232


namespace amp_2_neg1_1_l3082_308280

-- Define the & operation
def amp (a b c : ℝ) : ℝ := 3 * b^2 - 4 * a * c

-- Theorem statement
theorem amp_2_neg1_1 : amp 2 (-1) 1 = -5 := by
  sorry

end amp_2_neg1_1_l3082_308280


namespace color_change_probability_is_0_15_l3082_308249

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  blue : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light cycle -/
def probability_of_color_change (cycle : TrafficLightCycle) (observation_window : ℕ) : ℚ :=
  let total_cycle_time := cycle.green + cycle.yellow + cycle.blue + cycle.red
  let favorable_time := 3 * observation_window  -- 3 color transitions
  (favorable_time : ℚ) / total_cycle_time

/-- Theorem stating the probability of observing a color change is 0.15 for the given cycle -/
theorem color_change_probability_is_0_15 :
  let cycle := TrafficLightCycle.mk 45 5 10 40
  let observation_window := 5
  probability_of_color_change cycle observation_window = 15 / 100 := by
  sorry


end color_change_probability_is_0_15_l3082_308249


namespace negation_of_tangent_positive_l3082_308270

open Real

theorem negation_of_tangent_positive :
  (¬ ∀ x : ℝ, x ∈ Set.Ioo (-π/2) (π/2) → tan x > 0) ↔
  (∃ x : ℝ, x ∈ Set.Ioo (-π/2) (π/2) ∧ tan x ≤ 0) :=
by sorry

end negation_of_tangent_positive_l3082_308270


namespace rose_price_is_three_l3082_308201

-- Define the sales data
def tulips_day1 : ℕ := 30
def roses_day1 : ℕ := 20
def tulips_day2 : ℕ := 2 * tulips_day1
def roses_day2 : ℕ := 2 * roses_day1
def tulips_day3 : ℕ := (tulips_day2 * 10) / 100
def roses_day3 : ℕ := 16

-- Define the total sales
def total_tulips : ℕ := tulips_day1 + tulips_day2 + tulips_day3
def total_roses : ℕ := roses_day1 + roses_day2 + roses_day3

-- Define the price of a tulip
def tulip_price : ℚ := 2

-- Define the total earnings
def total_earnings : ℚ := 420

-- Theorem to prove
theorem rose_price_is_three :
  ∃ (rose_price : ℚ), 
    rose_price * total_roses + tulip_price * total_tulips = total_earnings ∧
    rose_price = 3 := by
  sorry


end rose_price_is_three_l3082_308201


namespace polygon_deformable_to_triangle_l3082_308222

/-- A planar polygon represented by its vertices -/
structure PlanarPolygon where
  vertices : List (ℝ × ℝ)
  n : ℕ
  h_n : vertices.length = n

/-- A function that checks if a polygon can be deformed into a triangle -/
def can_deform_to_triangle (p : PlanarPolygon) : Prop :=
  ∃ (v1 v2 v3 : ℝ × ℝ), v1 ∈ p.vertices ∧ v2 ∈ p.vertices ∧ v3 ∈ p.vertices ∧
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- The main theorem stating that any planar polygon with more than 4 vertices
    can be deformed into a triangle -/
theorem polygon_deformable_to_triangle (p : PlanarPolygon) (h : p.n > 4) :
  can_deform_to_triangle p := by
  sorry

end polygon_deformable_to_triangle_l3082_308222


namespace function_growth_l3082_308207

theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x < deriv f x) (a : ℝ) (ha : 0 < a) : 
  f a > Real.exp a * f 0 := by
  sorry

end function_growth_l3082_308207


namespace train_length_l3082_308221

/-- The length of a train given its crossing times over two platforms -/
theorem train_length (platform1_length platform2_length : ℝ)
                     (crossing_time1 crossing_time2 : ℝ)
                     (h1 : platform1_length = 200)
                     (h2 : platform2_length = 300)
                     (h3 : crossing_time1 = 15)
                     (h4 : crossing_time2 = 20) :
  ∃ (train_length : ℝ),
    train_length = 100 ∧
    (train_length + platform1_length) / crossing_time1 =
    (train_length + platform2_length) / crossing_time2 :=
by
  sorry


end train_length_l3082_308221


namespace vector_coordinates_l3082_308224

def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)
def C : ℝ × ℝ := (0, 3)

def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem vector_coordinates :
  vector A B = (-4, 3) ∧
  vector B C = (1, -2) ∧
  vector A C = (-3, 1) := by
  sorry

end vector_coordinates_l3082_308224


namespace magnitude_of_b_magnitude_of_c_and_area_l3082_308219

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 15 ∧ Real.sin t.A = 1/4

-- Theorem 1
theorem magnitude_of_b (t : Triangle) (h : triangle_conditions t) 
  (hcosB : Real.cos t.B = Real.sqrt 5 / 3) :
  t.b = 8 * Real.sqrt 15 / 3 :=
sorry

-- Theorem 2
theorem magnitude_of_c_and_area (t : Triangle) (h : triangle_conditions t) 
  (hb : t.b = 4 * t.a) :
  t.c = 15 ∧ (1/2 * t.b * t.c * Real.sin t.A = 15/2 * Real.sqrt 15) :=
sorry

end magnitude_of_b_magnitude_of_c_and_area_l3082_308219


namespace revenue_change_l3082_308264

theorem revenue_change
  (T : ℝ) -- original tax rate (as a percentage)
  (C : ℝ) -- original consumption
  (h1 : T > 0)
  (h2 : C > 0) :
  let new_tax_rate := T * (1 - 0.16)
  let new_consumption := C * (1 + 0.15)
  let original_revenue := (T / 100) * C
  let new_revenue := (new_tax_rate / 100) * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.034 := by
sorry

end revenue_change_l3082_308264


namespace continuity_properties_l3082_308283

theorem continuity_properties :
  (¬ ∀ a b : ℤ, a < b → ∃ c : ℤ, a < c ∧ c < b) ∧
  (¬ ∀ S : Set ℤ, S.Nonempty → (∃ x : ℤ, ∀ y ∈ S, y ≤ x) → ∃ z : ℤ, ∀ y ∈ S, y ≤ z ∧ ∀ w : ℤ, (∀ y ∈ S, y ≤ w) → z ≤ w) ∧
  (∀ a b : ℚ, a < b → ∃ c : ℚ, a < c ∧ c < b) ∧
  (¬ ∀ S : Set ℚ, S.Nonempty → (∃ x : ℚ, ∀ y ∈ S, y ≤ x) → ∃ z : ℚ, ∀ y ∈ S, y ≤ z ∧ ∀ w : ℚ, (∀ y ∈ S, y ≤ w) → z ≤ w) :=
by sorry

end continuity_properties_l3082_308283


namespace f_decreasing_iff_b_leq_neg_two_l3082_308220

/-- A piecewise function f parameterized by b -/
noncomputable def f (b : ℝ) : ℝ → ℝ := fun x =>
  if x < 0 then x^2 + (2+b)*x - 1 else (2*b-1)*x + b - 2

/-- f is decreasing on ℝ if and only if b ≤ -2 -/
theorem f_decreasing_iff_b_leq_neg_two (b : ℝ) :
  (∀ x y : ℝ, x < y → f b x > f b y) ↔ b ≤ -2 := by sorry

end f_decreasing_iff_b_leq_neg_two_l3082_308220


namespace integer_solutions_equation_l3082_308230

theorem integer_solutions_equation : 
  {(x, y) : ℤ × ℤ | 3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3} = 
  {(0, 0), (6, 6), (-6, -6)} := by
sorry

end integer_solutions_equation_l3082_308230


namespace chess_class_percentage_l3082_308235

theorem chess_class_percentage 
  (total_students : ℕ) 
  (swimming_students : ℕ) 
  (chess_to_swimming_ratio : ℚ) :
  total_students = 1000 →
  swimming_students = 125 →
  chess_to_swimming_ratio = 1/2 →
  (↑swimming_students : ℚ) / (chess_to_swimming_ratio * ↑total_students) = 1/4 := by
  sorry

end chess_class_percentage_l3082_308235


namespace seventh_term_of_geometric_sequence_l3082_308202

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 4) :
  a 7 = 16 := by
sorry

end seventh_term_of_geometric_sequence_l3082_308202


namespace alexis_skirt_time_l3082_308291

/-- The time it takes Alexis to sew a skirt -/
def skirt_time : ℝ := 2

/-- The time it takes Alexis to sew a coat -/
def coat_time : ℝ := 7

/-- The total time it takes Alexis to sew 6 skirts and 4 coats -/
def total_time : ℝ := 40

theorem alexis_skirt_time :
  skirt_time = 2 ∧
  coat_time = 7 ∧
  total_time = 40 ∧
  6 * skirt_time + 4 * coat_time = total_time :=
by sorry

end alexis_skirt_time_l3082_308291


namespace correct_proposition_l3082_308212

-- Define proposition p₁
def p₁ : Prop := ∃ x₀ : ℝ, x₀^2 + x₀ + 1 < 0

-- Define proposition p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem correct_proposition : (¬p₁) ∧ p₂ := by
  sorry

end correct_proposition_l3082_308212


namespace symmetric_polynomial_n_l3082_308281

/-- A polynomial p(x) is symmetric about x = m if p(m + k) = p(m - k) for all real k -/
def is_symmetric_about (p : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ k, p (m + k) = p (m - k)

/-- The polynomial p(x) = x^2 + 2nx + 3 -/
def p (n : ℝ) (x : ℝ) : ℝ := x^2 + 2*n*x + 3

theorem symmetric_polynomial_n (n : ℝ) :
  is_symmetric_about (p n) 5 → n = -5 := by
  sorry

end symmetric_polynomial_n_l3082_308281


namespace probability_sum_greater_than_nine_l3082_308285

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The set of all possible sums when rolling two dice -/
def possible_sums : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

/-- The set of sums greater than 9 -/
def sums_greater_than_nine : Set ℕ := {10, 11, 12}

/-- The number of favorable outcomes (sums greater than 9) -/
def favorable_outcomes : ℕ := 6

/-- Theorem: The probability of rolling two dice and getting a sum greater than 9 is 1/6 -/
theorem probability_sum_greater_than_nine :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by sorry

end probability_sum_greater_than_nine_l3082_308285


namespace soda_distribution_l3082_308233

theorem soda_distribution (boxes : Nat) (cans_per_box : Nat) (discarded : Nat) (cartons : Nat) :
  boxes = 7 →
  cans_per_box = 16 →
  discarded = 13 →
  cartons = 8 →
  (boxes * cans_per_box - discarded) % cartons = 3 := by
  sorry

end soda_distribution_l3082_308233


namespace both_charts_rough_determination_l3082_308237

/-- Represents a chart type -/
inductive ChartType
  | ThreeD_Column
  | TwoD_Bar

/-- Represents the ability to determine relationships between categorical variables -/
inductive RelationshipDetermination
  | Accurate
  | Rough
  | Unable

/-- Function that determines the relationship determination capability of a chart type -/
def chart_relationship_determination : ChartType → RelationshipDetermination
  | ChartType.ThreeD_Column => RelationshipDetermination.Rough
  | ChartType.TwoD_Bar => RelationshipDetermination.Rough

/-- Theorem stating that both 3D column charts and 2D bar charts can roughly determine relationships -/
theorem both_charts_rough_determination :
  (chart_relationship_determination ChartType.ThreeD_Column = RelationshipDetermination.Rough) ∧
  (chart_relationship_determination ChartType.TwoD_Bar = RelationshipDetermination.Rough) :=
by
  sorry

#check both_charts_rough_determination

end both_charts_rough_determination_l3082_308237


namespace jenson_shirts_per_day_l3082_308210

/-- The number of shirts Jenson makes per day -/
def shirts_per_day : ℕ := sorry

/-- The number of pairs of pants Kingsley makes per day -/
def pants_per_day : ℕ := 5

/-- The amount of fabric used for one shirt (in yards) -/
def fabric_per_shirt : ℕ := 2

/-- The amount of fabric used for one pair of pants (in yards) -/
def fabric_per_pants : ℕ := 5

/-- The total amount of fabric needed every 3 days (in yards) -/
def total_fabric_3days : ℕ := 93

theorem jenson_shirts_per_day :
  shirts_per_day = 3 ∧
  shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pants = total_fabric_3days / 3 :=
sorry

end jenson_shirts_per_day_l3082_308210


namespace binomial_10_3_l3082_308205

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l3082_308205


namespace bike_cost_theorem_l3082_308266

theorem bike_cost_theorem (marion_cost : ℕ) : 
  marion_cost = 356 → 
  (marion_cost + 2 * marion_cost + 3 * marion_cost : ℕ) = 2136 := by
  sorry

end bike_cost_theorem_l3082_308266


namespace special_polynomial_sum_l3082_308241

/-- A monic polynomial of degree 3 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y z : ℝ, p x = x^3 + y*x^2 + z*x + (7 - 6*y - 6*z)) ∧ 
  p 1 = 7 ∧ p 2 = 14 ∧ p 3 = 21

theorem special_polynomial_sum (p : ℝ → ℝ) (h : special_polynomial p) : 
  p 0 + p 5 = 53 := by
  sorry

end special_polynomial_sum_l3082_308241


namespace meal_contribution_proof_l3082_308217

/-- Calculates the individual contribution for a shared meal bill -/
def calculate_individual_contribution (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) : ℚ :=
  (total_price - coupon_value) / num_people

/-- Proves that the individual contribution for the given scenario is $21 -/
theorem meal_contribution_proof (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) 
  (h1 : total_price = 67)
  (h2 : coupon_value = 4)
  (h3 : num_people = 3) :
  calculate_individual_contribution total_price coupon_value num_people = 21 := by
  sorry

#eval calculate_individual_contribution 67 4 3

end meal_contribution_proof_l3082_308217


namespace intersection_of_A_and_B_l3082_308213

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l3082_308213


namespace greatest_power_of_two_factor_l3082_308251

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1004 - 4^502) ∧ 
    ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) → 
  (∃ k : ℕ, 2^k ∣ (10^1004 - 4^502) ∧ 
    ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) ∧ 
  n = 1007 :=
by sorry

end greatest_power_of_two_factor_l3082_308251


namespace min_balls_for_three_colors_l3082_308267

/-- Represents the number of balls of a specific color in the box -/
def BallCount := ℕ

/-- Represents the total number of balls in the box -/
def TotalBalls : ℕ := 111

/-- Represents the number of different colors of balls in the box -/
def NumColors : ℕ := 4

/-- Represents the number of balls that guarantees at least four different colors when drawn -/
def GuaranteeFourColors : ℕ := 100

/-- Represents a function that returns the minimum number of balls to draw to ensure at least three different colors -/
def minBallsForThreeColors (total : ℕ) (numColors : ℕ) (guaranteeFour : ℕ) : ℕ := 
  total - guaranteeFour + 1

/-- Theorem stating that the minimum number of balls to draw to ensure at least three different colors is 88 -/
theorem min_balls_for_three_colors : 
  minBallsForThreeColors TotalBalls NumColors GuaranteeFourColors = 88 := by
  sorry

end min_balls_for_three_colors_l3082_308267


namespace fraction_equality_l3082_308223

theorem fraction_equality : (3023 - 2990)^2 / 121 = 9 := by sorry

end fraction_equality_l3082_308223


namespace math_books_count_l3082_308255

/-- Proves that the number of math books bought is 60 given the specified conditions -/
theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℕ) (total_price : ℕ) :
  total_books = 90 →
  math_book_price = 4 →
  history_book_price = 5 →
  total_price = 390 →
  ∃ (math_books : ℕ), math_books = 60 ∧ 
    math_books + (total_books - math_books) = total_books ∧
    math_book_price * math_books + history_book_price * (total_books - math_books) = total_price :=
by
  sorry

end math_books_count_l3082_308255


namespace three_plus_three_cubed_l3082_308260

theorem three_plus_three_cubed : 3 + 3^3 = 30 := by
  sorry

end three_plus_three_cubed_l3082_308260


namespace cubic_expression_value_l3082_308278

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2*x^2 - 7 = -6 := by
  sorry

end cubic_expression_value_l3082_308278


namespace haunted_castle_windows_l3082_308269

theorem haunted_castle_windows (n : ℕ) (h : n = 10) : 
  n * (n - 1) * (n - 2) * (n - 3) = 5040 :=
sorry

end haunted_castle_windows_l3082_308269


namespace vasya_numbers_l3082_308226

theorem vasya_numbers : ∃ (x y : ℝ), x + y = x * y ∧ x + y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end vasya_numbers_l3082_308226


namespace complex_fraction_simplification_l3082_308257

theorem complex_fraction_simplification : 
  (((12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324)) / 
   ((6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324))) = 221 := by
  sorry

end complex_fraction_simplification_l3082_308257


namespace fraction_product_square_l3082_308296

theorem fraction_product_square : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end fraction_product_square_l3082_308296


namespace units_digit_difference_largest_smallest_l3082_308275

theorem units_digit_difference_largest_smallest (a b c d e : ℕ) :
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9 →
  (100 * e + 10 * d + c) - (100 * a + 10 * b + c) ≡ 0 [MOD 10] :=
by sorry

end units_digit_difference_largest_smallest_l3082_308275


namespace total_area_of_three_triangles_l3082_308228

theorem total_area_of_three_triangles (base height : ℝ) (h1 : base = 40) (h2 : height = 20) :
  3 * (1/2 * base * height) = 1200 := by
  sorry

end total_area_of_three_triangles_l3082_308228


namespace prism_18_edges_8_faces_l3082_308231

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ (p : Prism), p.edges = 18 → num_faces p = 8 := by
  sorry

#check prism_18_edges_8_faces

end prism_18_edges_8_faces_l3082_308231


namespace total_lemonade_poured_l3082_308203

def first_intermission : ℝ := 0.25 + 0.125
def second_intermission : ℝ := 0.16666666666666666 + 0.08333333333333333 + 0.16666666666666666
def third_intermission : ℝ := 0.25 + 0.125
def fourth_intermission : ℝ := 0.3333333333333333 + 0.08333333333333333 + 0.16666666666666666

theorem total_lemonade_poured :
  first_intermission + second_intermission + third_intermission + fourth_intermission = 1.75 := by
  sorry

end total_lemonade_poured_l3082_308203


namespace gcd_factorial_problem_l3082_308268

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 720 := by
  sorry

end gcd_factorial_problem_l3082_308268


namespace number_operations_l3082_308261

theorem number_operations (x : ℤ) : 
  (((x + 7) * 3 - 12) / 6 : ℚ) = -8 → x = -19 := by
  sorry

end number_operations_l3082_308261


namespace quadratic_function_max_value_l3082_308252

theorem quadratic_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x ∈ Set.Icc 0 3, a * x^2 - 2 * a * x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 3, a * x^2 - 2 * a * x = 3) →
  a = -3 ∨ a = 1 := by
  sorry

end quadratic_function_max_value_l3082_308252


namespace sin_two_alpha_value_l3082_308277

theorem sin_two_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * (Real.cos α)^2 = Real.sin (π/4 - α)) :
  Real.sin (2*α) = 1 ∨ Real.sin (2*α) = -17/18 := by
  sorry

end sin_two_alpha_value_l3082_308277


namespace prob_at_least_one_in_three_games_l3082_308295

/-- The probability of revealing a golden flower when smashing a single egg -/
def p : ℚ := 1/2

/-- The number of eggs smashed in each game -/
def n : ℕ := 3

/-- The number of games played -/
def games : ℕ := 3

/-- The probability of revealing at least one golden flower in a single game -/
def prob_at_least_one_in_game : ℚ := 1 - (1 - p)^n

/-- Theorem: The probability of revealing at least one golden flower in three games -/
theorem prob_at_least_one_in_three_games :
  (1 : ℚ) - (1 - prob_at_least_one_in_game)^games = 511/512 := by
  sorry

end prob_at_least_one_in_three_games_l3082_308295


namespace some_base_value_l3082_308258

theorem some_base_value (x y some_base : ℝ) 
  (h1 : x * y = 1) 
  (h2 : (some_base^((x + y)^2)) / (some_base^((x - y)^2)) = 2401) : 
  some_base = 7 := by sorry

end some_base_value_l3082_308258


namespace gregs_mom_cookies_l3082_308236

theorem gregs_mom_cookies (greg_halves brad_halves left_halves : ℕ) 
  (h1 : greg_halves = 4)
  (h2 : brad_halves = 6)
  (h3 : left_halves = 18) :
  (greg_halves + brad_halves + left_halves) / 2 = 14 := by
  sorry

end gregs_mom_cookies_l3082_308236


namespace triangle_rotation_l3082_308243

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices O, P, and Q -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point) : ℝ := sorry

/-- Rotates a point 90 degrees counter-clockwise around the origin -/
def rotate90 (p : Point) : Point :=
  { x := -p.y, y := p.x }

/-- The main theorem -/
theorem triangle_rotation (t : Triangle) : 
  t.O = ⟨0, 0⟩ → 
  t.P = ⟨7, 0⟩ → 
  t.Q.x > 0 → 
  t.Q.y > 0 → 
  angle t.P t.Q = π / 2 → 
  angle t.P t.Q - angle t.O t.Q = π / 4 → 
  rotate90 t.Q = ⟨-7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2⟩ := by sorry

end triangle_rotation_l3082_308243


namespace dog_adoption_rate_is_half_l3082_308234

/-- Represents the animal shelter scenario --/
structure AnimalShelter where
  initialDogs : ℕ
  initialCats : ℕ
  initialLizards : ℕ
  catAdoptionRate : ℚ
  lizardAdoptionRate : ℚ
  newPetsPerMonth : ℕ
  totalPetsAfterMonth : ℕ

/-- Calculates the dog adoption rate given the shelter scenario --/
def dogAdoptionRate (shelter : AnimalShelter) : ℚ :=
  let totalInitial := shelter.initialDogs + shelter.initialCats + shelter.initialLizards
  let adoptedCats := shelter.catAdoptionRate * shelter.initialCats
  let adoptedLizards := shelter.lizardAdoptionRate * shelter.initialLizards
  let remainingPets := shelter.totalPetsAfterMonth - shelter.newPetsPerMonth
  ((totalInitial - remainingPets) - (adoptedCats + adoptedLizards)) / shelter.initialDogs

/-- Theorem stating that the dog adoption rate is 50% for the given scenario --/
theorem dog_adoption_rate_is_half (shelter : AnimalShelter) 
  (h1 : shelter.initialDogs = 30)
  (h2 : shelter.initialCats = 28)
  (h3 : shelter.initialLizards = 20)
  (h4 : shelter.catAdoptionRate = 1/4)
  (h5 : shelter.lizardAdoptionRate = 1/5)
  (h6 : shelter.newPetsPerMonth = 13)
  (h7 : shelter.totalPetsAfterMonth = 65) :
  dogAdoptionRate shelter = 1/2 := by
  sorry

end dog_adoption_rate_is_half_l3082_308234


namespace hcl_production_l3082_308273

-- Define the chemical reaction
structure Reaction where
  reactant1 : ℕ  -- moles of NaCl
  reactant2 : ℕ  -- moles of HNO3
  product : ℕ    -- moles of HCl produced

-- Define the stoichiometric relationship
def stoichiometric_ratio (r : Reaction) : Prop :=
  r.product = min r.reactant1 r.reactant2

-- Theorem statement
theorem hcl_production (nacl_moles hno3_moles : ℕ) 
  (h : nacl_moles = 3 ∧ hno3_moles = 3) : 
  ∃ (r : Reaction), r.reactant1 = nacl_moles ∧ r.reactant2 = hno3_moles ∧ 
  stoichiometric_ratio r ∧ r.product = 3 :=
sorry

end hcl_production_l3082_308273


namespace frank_skee_ball_tickets_l3082_308290

/-- The number of tickets Frank won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 33

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 6

/-- The number of candies Frank can buy with his total tickets -/
def candies_bought : ℕ := 7

/-- The number of tickets Frank won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candies_bought * candy_cost - whack_a_mole_tickets

theorem frank_skee_ball_tickets : skee_ball_tickets = 9 := by
  sorry

end frank_skee_ball_tickets_l3082_308290


namespace prob_two_red_marbles_l3082_308279

/-- The probability of selecting two red marbles without replacement from a bag containing 2 red marbles and 3 green marbles is 1/10. -/
theorem prob_two_red_marbles (red : ℕ) (green : ℕ) (h1 : red = 2) (h2 : green = 3) :
  (red / (red + green)) * ((red - 1) / (red + green - 1)) = 1 / 10 := by
  sorry

end prob_two_red_marbles_l3082_308279


namespace arithmetic_geometric_sequence_third_term_l3082_308299

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  first_term : ℚ
  common_diff : ℚ
  seq_def : ∀ n : ℕ, a n = first_term * q^n + common_diff * (1 - q^n) / (1 - q)

/-- Sum of first n terms of an arithmetic-geometric sequence -/
def sum_n (seq : ArithmeticGeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * (1 - seq.q^n) / (1 - seq.q)

theorem arithmetic_geometric_sequence_third_term
  (seq : ArithmeticGeometricSequence)
  (h1 : sum_n seq 6 / sum_n seq 3 = -19/8)
  (h2 : seq.a 4 - seq.a 2 = -15/8) :
  seq.a 3 = 9/4 := by
  sorry

end arithmetic_geometric_sequence_third_term_l3082_308299


namespace v_2008_eq_352_l3082_308287

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ 
| 0 => 1  -- First term
| n + 1 => 
  let group := (Nat.sqrt (8 * (n + 1) + 1) - 1) / 2  -- Determine which group n+1 belongs to
  let groupStart := group * (group + 1) / 2  -- Starting position of the group
  let offset := n + 1 - groupStart  -- Position within the group
  (group + 1) + 3 * ((groupStart - 1) + offset)  -- Calculate the term

/-- The 2008th term of the sequence is 352 -/
theorem v_2008_eq_352 : v 2007 = 352 := by
  sorry

end v_2008_eq_352_l3082_308287


namespace f_of_2_equals_2_l3082_308271

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem f_of_2_equals_2 : f 2 = 2 := by
  sorry

end f_of_2_equals_2_l3082_308271


namespace three_digit_reverse_difference_theorem_l3082_308218

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_do_not_repeat (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

def same_digits (m n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (m = 100 * a + 10 * b + c ∨ m = 100 * a + 10 * c + b ∨
     m = 100 * b + 10 * a + c ∨ m = 100 * b + 10 * c + a ∨
     m = 100 * c + 10 * a + b ∨ m = 100 * c + 10 * b + a) ∧
    (n = 100 * a + 10 * b + c ∨ n = 100 * a + 10 * c + b ∨
     n = 100 * b + 10 * a + c ∨ n = 100 * b + 10 * c + a ∨
     n = 100 * c + 10 * a + b ∨ n = 100 * c + 10 * b + a)

theorem three_digit_reverse_difference_theorem :
  ∀ x : ℕ,
    is_three_digit x ∧
    digits_do_not_repeat x ∧
    is_three_digit (x - reverse_number x) ∧
    same_digits x (x - reverse_number x) →
    x = 954 ∨ x = 459 := by
  sorry


end three_digit_reverse_difference_theorem_l3082_308218


namespace min_points_last_two_games_l3082_308288

theorem min_points_last_two_games 
  (scores : List ℕ)
  (h1 : scores.length = 20)
  (h2 : scores[14] = 26 ∧ scores[15] = 15 ∧ scores[16] = 12 ∧ scores[17] = 24)
  (h3 : (scores.take 18).sum / 18 > (scores.take 14).sum / 14)
  (h4 : scores.sum / 20 > 20) :
  scores[18] + scores[19] ≥ 58 := by
  sorry

end min_points_last_two_games_l3082_308288


namespace f_increasing_iff_l3082_308276

/-- A piecewise function f defined on ℝ --/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 1 then a^x else (3-a)*x + 1

/-- Theorem stating the condition for f to be increasing --/
theorem f_increasing_iff (a : ℝ) :
  StrictMono (f a) ↔ 2 ≤ a ∧ a < 3 :=
sorry

#check f_increasing_iff

end f_increasing_iff_l3082_308276


namespace highest_score_is_242_l3082_308248

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  total_innings : ℕ
  average : ℚ
  score_difference : ℕ
  average_drop : ℚ

/-- Calculates the highest score of a batsman given their statistics -/
def highest_score (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the highest score is 242 -/
theorem highest_score_is_242 (stats : BatsmanStats) 
  (h1 : stats.total_innings = 60)
  (h2 : stats.average = 55)
  (h3 : stats.score_difference = 200)
  (h4 : stats.average_drop = 3) :
  highest_score stats = 242 :=
by sorry

end highest_score_is_242_l3082_308248


namespace john_excess_money_l3082_308265

def earnings_day1 : ℚ := 20
def earnings_day2 : ℚ := 18
def earnings_day3 : ℚ := earnings_day2 / 2
def earnings_day4 : ℚ := earnings_day3 + (earnings_day3 * (25 / 100))
def earnings_day5 : ℚ := earnings_day4 + (earnings_day3 * (25 / 100))
def earnings_day6 : ℚ := earnings_day5 + (earnings_day5 * (15 / 100))
def earnings_day7 : ℚ := earnings_day6 - 10

def daily_increase : ℚ := 1
def pogo_stick_cost : ℚ := 60

def total_earnings : ℚ := 
  earnings_day1 + earnings_day2 + earnings_day3 + earnings_day4 + earnings_day5 + 
  earnings_day6 + earnings_day7 + 
  (earnings_day6 + daily_increase) + 
  (earnings_day6 + 2 * daily_increase) + 
  (earnings_day6 + 3 * daily_increase) + 
  (earnings_day6 + 4 * daily_increase) + 
  (earnings_day6 + 5 * daily_increase) + 
  (earnings_day6 + 6 * daily_increase) + 
  (earnings_day6 + 7 * daily_increase)

theorem john_excess_money : total_earnings - pogo_stick_cost = 170 := by
  sorry

end john_excess_money_l3082_308265


namespace isosceles_triangle_l3082_308263

theorem isosceles_triangle (A B C : Real) (h1 : A + B + C = π) (h2 : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B := by
  sorry

end isosceles_triangle_l3082_308263


namespace square_root_of_49_l3082_308256

theorem square_root_of_49 : 
  {x : ℝ | x^2 = 49} = {7, -7} := by sorry

end square_root_of_49_l3082_308256


namespace perpendicular_vectors_l3082_308239

def a : Fin 2 → ℝ := ![3, 2]

def v1 : Fin 2 → ℝ := ![3, -2]
def v2 : Fin 2 → ℝ := ![2, 3]
def v3 : Fin 2 → ℝ := ![-4, 6]
def v4 : Fin 2 → ℝ := ![-3, 2]

def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

theorem perpendicular_vectors :
  dot_product a v1 ≠ 0 ∧
  dot_product a v2 ≠ 0 ∧
  dot_product a v3 = 0 ∧
  dot_product a v4 ≠ 0 := by
  sorry

end perpendicular_vectors_l3082_308239


namespace equation_solution_l3082_308211

theorem equation_solution : ∀ x : ℚ, (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end equation_solution_l3082_308211


namespace cakes_served_today_l3082_308272

theorem cakes_served_today (dinner_stock : ℕ) (lunch_served : ℚ) (dinner_percentage : ℚ) :
  dinner_stock = 95 →
  lunch_served = 48.5 →
  dinner_percentage = 62.25 →
  ⌈lunch_served + (dinner_percentage / 100) * dinner_stock⌉ = 108 :=
by sorry

end cakes_served_today_l3082_308272
