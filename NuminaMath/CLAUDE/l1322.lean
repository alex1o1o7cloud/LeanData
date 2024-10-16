import Mathlib

namespace NUMINAMATH_CALUDE_max_value_condition_l1322_132233

/-- The function f(x) = -x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x

/-- The maximum value of f(x) on the interval [0, 1] is 2 iff a = -2√2 or a = 3 -/
theorem max_value_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ (∃ x ∈ Set.Icc 0 1, f a x = 2) ↔ 
  (a = -2 * Real.sqrt 2 ∨ a = 3) := by
sorry

end NUMINAMATH_CALUDE_max_value_condition_l1322_132233


namespace NUMINAMATH_CALUDE_ink_cartridge_cost_l1322_132244

theorem ink_cartridge_cost (total_cost : ℕ) (num_cartridges : ℕ) (cost_per_cartridge : ℕ) :
  total_cost = 182 →
  num_cartridges = 13 →
  total_cost = num_cartridges * cost_per_cartridge →
  cost_per_cartridge = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_ink_cartridge_cost_l1322_132244


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1322_132208

theorem trigonometric_simplification (α : ℝ) :
  2 * Real.sin (2 * α) ^ 2 + Real.sqrt 3 * Real.sin (4 * α) -
  (4 * Real.tan (2 * α) * (1 - Real.tan (2 * α) ^ 2)) /
  (Real.sin (8 * α) * (1 + Real.tan (2 * α) ^ 2) ^ 2) =
  2 * Real.sin (4 * α - π / 6) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1322_132208


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1322_132215

theorem inequality_implies_upper_bound (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ a) → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1322_132215


namespace NUMINAMATH_CALUDE_total_bugs_is_63_l1322_132246

/-- The number of bugs eaten by the gecko, lizard, frog, and toad -/
def total_bugs_eaten (gecko_bugs : ℕ) : ℕ :=
  let lizard_bugs := gecko_bugs / 2
  let frog_bugs := lizard_bugs * 3
  let toad_bugs := frog_bugs + frog_bugs / 2
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs

/-- Theorem stating the total number of bugs eaten is 63 -/
theorem total_bugs_is_63 : total_bugs_eaten 12 = 63 := by
  sorry

#eval total_bugs_eaten 12

end NUMINAMATH_CALUDE_total_bugs_is_63_l1322_132246


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_minimum_value_range_l1322_132258

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Part 1: Prove the solution set for f(x) ≤ 5 when a = 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} := by sorry

-- Part 2: Prove the range of a for which f(x) has a minimum value
theorem minimum_value_range :
  {a : ℝ | ∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y} = {a : ℝ | -3 ≤ a ∧ a ≤ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_minimum_value_range_l1322_132258


namespace NUMINAMATH_CALUDE_jameson_medal_count_l1322_132279

/-- Represents the number of medals Jameson has in each category -/
structure MedalCount where
  track : Nat
  swimming : Nat
  badminton : Nat

/-- Calculates the total number of medals -/
def totalMedals (medals : MedalCount) : Nat :=
  medals.track + medals.swimming + medals.badminton

/-- Theorem: Jameson's total medal count is 20 -/
theorem jameson_medal_count :
  ∀ (medals : MedalCount),
    medals.track = 5 →
    medals.swimming = 2 * medals.track →
    medals.badminton = 5 →
    totalMedals medals = 20 := by
  sorry


end NUMINAMATH_CALUDE_jameson_medal_count_l1322_132279


namespace NUMINAMATH_CALUDE_total_CDs_is_448_l1322_132294

/-- The number of shelves in Store A -/
def store_A_shelves : ℕ := 5

/-- The number of CD racks per shelf in Store A -/
def store_A_racks_per_shelf : ℕ := 7

/-- The number of CDs per rack in Store A -/
def store_A_CDs_per_rack : ℕ := 8

/-- The number of shelves in Store B -/
def store_B_shelves : ℕ := 4

/-- The number of CD racks per shelf in Store B -/
def store_B_racks_per_shelf : ℕ := 6

/-- The number of CDs per rack in Store B -/
def store_B_CDs_per_rack : ℕ := 7

/-- The total number of CDs that can be held in Store A and Store B together -/
def total_CDs : ℕ := 
  (store_A_shelves * store_A_racks_per_shelf * store_A_CDs_per_rack) +
  (store_B_shelves * store_B_racks_per_shelf * store_B_CDs_per_rack)

/-- Theorem stating that the total number of CDs that can be held in Store A and Store B together is 448 -/
theorem total_CDs_is_448 : total_CDs = 448 := by
  sorry

end NUMINAMATH_CALUDE_total_CDs_is_448_l1322_132294


namespace NUMINAMATH_CALUDE_samuel_breaks_two_cups_per_box_l1322_132260

theorem samuel_breaks_two_cups_per_box 
  (total_boxes : ℕ) 
  (pan_boxes : ℕ) 
  (cups_per_row : ℕ) 
  (rows_per_box : ℕ) 
  (remaining_cups : ℕ) 
  (h1 : total_boxes = 26)
  (h2 : pan_boxes = 6)
  (h3 : cups_per_row = 4)
  (h4 : rows_per_box = 5)
  (h5 : remaining_cups = 180) :
  let remaining_boxes := total_boxes - pan_boxes
  let decoration_boxes := remaining_boxes / 2
  let teacup_boxes := remaining_boxes - decoration_boxes
  let cups_per_box := cups_per_row * rows_per_box
  let total_cups := teacup_boxes * cups_per_box
  let broken_cups := total_cups - remaining_cups
  2 = broken_cups / teacup_boxes :=
by sorry

end NUMINAMATH_CALUDE_samuel_breaks_two_cups_per_box_l1322_132260


namespace NUMINAMATH_CALUDE_fraction_equality_l1322_132218

theorem fraction_equality : 
  (14/10 : ℚ) = 7/5 ∧ 
  (1 + 2/5 : ℚ) = 7/5 ∧ 
  (1 + 7/35 : ℚ) ≠ 7/5 ∧ 
  (1 + 4/20 : ℚ) ≠ 7/5 ∧ 
  (1 + 3/15 : ℚ) ≠ 7/5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1322_132218


namespace NUMINAMATH_CALUDE_arctan_of_tan_difference_l1322_132221

theorem arctan_of_tan_difference (θ : Real) : 
  θ ≥ 0 ∧ θ ≤ 180 → Real.arctan (Real.tan (75 * π / 180) - 2 * Real.tan (30 * π / 180)) * 180 / π = 75 := by
  sorry

end NUMINAMATH_CALUDE_arctan_of_tan_difference_l1322_132221


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1322_132229

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 3) :
  2^a + 2^b ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1322_132229


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_segments_l1322_132228

/-- 
Theorem: In a trapezoid with bases a and c, and sides b and d, 
the segments AO and OC of diagonal AC divided by diagonal BD are:
AO = (c / (a + c)) * √(ac + (ad² - cb²) / (a - c))
OC = (a / (a + c)) * √(ac + (ad² - cb²) / (a - c))
-/
theorem trapezoid_diagonal_segments 
  (a c b d : ℝ) 
  (ha : a > 0) 
  (hc : c > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hac : a ≠ c) :
  ∃ (AO OC : ℝ),
    AO = (c / (a + c)) * Real.sqrt (a * c + (a * d^2 - c * b^2) / (a - c)) ∧
    OC = (a / (a + c)) * Real.sqrt (a * c + (a * d^2 - c * b^2) / (a - c)) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_segments_l1322_132228


namespace NUMINAMATH_CALUDE_gain_represents_12_meters_l1322_132253

-- Define the total meters of cloth sold
def total_meters : ℝ := 60

-- Define the gain percentage
def gain_percentage : ℝ := 0.20

-- Define the cost price per meter (as a variable)
variable (cost_price : ℝ)

-- Define the selling price per meter
def selling_price (cost_price : ℝ) : ℝ := cost_price * (1 + gain_percentage)

-- Define the total gain
def total_gain (cost_price : ℝ) : ℝ := 
  total_meters * selling_price cost_price - total_meters * cost_price

-- Theorem: The gain represents 12 meters of cloth
theorem gain_represents_12_meters (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  total_gain cost_price = 12 * cost_price := by
  sorry

end NUMINAMATH_CALUDE_gain_represents_12_meters_l1322_132253


namespace NUMINAMATH_CALUDE_angle_solution_l1322_132240

def angle_coincides (α : ℝ) : Prop :=
  ∃ k : ℤ, 9 * α = k * 360 + α

theorem angle_solution :
  ∀ α : ℝ, 0 < α → α < 180 → angle_coincides α → (α = 45 ∨ α = 90) :=
by sorry

end NUMINAMATH_CALUDE_angle_solution_l1322_132240


namespace NUMINAMATH_CALUDE_parallel_not_coincident_condition_l1322_132262

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- Two lines are coincident if they are parallel and have the same y-intercept -/
def coincident (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop := 
  parallel a₁ b₁ a₂ b₂ ∧ (c₁ * b₂ = c₂ * b₁)

/-- The necessary and sufficient condition for the given lines to be parallel and not coincident -/
theorem parallel_not_coincident_condition : 
  ∀ a : ℝ, (parallel a 2 3 (a-1) ∧ 
            ¬coincident a 2 (-3*a) 3 (a-1) (7-a)) ↔ 
           (a = 3) := by sorry

end NUMINAMATH_CALUDE_parallel_not_coincident_condition_l1322_132262


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1322_132272

/-- For a geometric sequence with common ratio 2, the ratio of the 4th term to the 2nd term is 4. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  a 4 / a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1322_132272


namespace NUMINAMATH_CALUDE_equation_solution_l1322_132239

theorem equation_solution : 
  ∃! x : ℝ, x ≠ (1/2) ∧ (5*x + 1) / (2*x^2 + 5*x - 3) = 2*x / (2*x - 1) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1322_132239


namespace NUMINAMATH_CALUDE_job_completion_time_l1322_132285

/-- The time it takes for three workers to complete a job together, given their individual efficiencies -/
theorem job_completion_time 
  (sakshi_time : ℝ) 
  (tanya_efficiency : ℝ) 
  (rahul_efficiency : ℝ) 
  (h1 : sakshi_time = 5) 
  (h2 : tanya_efficiency = 1.25) 
  (h3 : rahul_efficiency = 0.6) : 
  (1 / (1 / sakshi_time + tanya_efficiency / sakshi_time + rahul_efficiency / sakshi_time)) = 100 / 57 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1322_132285


namespace NUMINAMATH_CALUDE_expression_simplification_l1322_132273

theorem expression_simplification (a b c : ℚ) 
  (ha : a = 1/3) (hb : b = 1/2) (hc : c = 1) : 
  (2*a^2 - b) - (a^2 - 4*b) - (b + c) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1322_132273


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1322_132242

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1322_132242


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l1322_132238

/-- A right triangle with side lengths 5, 12, and 13 has an inradius of 2 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 5 ∧ b = 12 ∧ c = 13 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l1322_132238


namespace NUMINAMATH_CALUDE_hyperbola_circle_tangency_l1322_132290

/-- Given a hyperbola and a circle, if one asymptote of the hyperbola is tangent to the circle,
    then the ratio of the hyperbola's parameters is 3/4 -/
theorem hyperbola_circle_tangency (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (y^2 / a^2) - (x^2 / b^2) = 1 ∧ 
   (∃ (t : ℝ), y = (a/b) * x + t) ∧
   (x - 2)^2 + (y - 1)^2 = 1) →
  b / a = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_tangency_l1322_132290


namespace NUMINAMATH_CALUDE_f_decreasing_area_is_one_l1322_132223

/-- A function that is directly proportional to x-1 and passes through (-1, 4) -/
def f (x : ℝ) : ℝ := -2 * x + 2

/-- The property that f is directly proportional to x-1 -/
axiom f_prop (x : ℝ) : ∃ k : ℝ, f x = k * (x - 1)

/-- The property that f(-1) = 4 -/
axiom f_point : f (-1) = 4

/-- For any two x-values, if x1 > x2, then f(x1) < f(x2) -/
theorem f_decreasing (x1 x2 : ℝ) (h : x1 > x2) : f x1 < f x2 := by sorry

/-- The area of the triangle formed by shifting f down by 4 units -/
def triangle_area : ℝ := 1

/-- The area of the triangle formed by shifting f down by 4 units is 1 -/
theorem area_is_one : triangle_area = 1 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_area_is_one_l1322_132223


namespace NUMINAMATH_CALUDE_range_of_f_l1322_132283

def P : Set ℕ := {1, 2, 3}

def f (x : ℕ) : ℕ := 2^x

theorem range_of_f :
  {y | ∃ x ∈ P, f x = y} = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1322_132283


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1322_132282

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! x : ℤ, (x + 2) / (4 - x) < 0 ∧ 2*x^2 + (2*a + 7)*x + 7*a < 0) →
  ((-5 ≤ a ∧ a < 3) ∨ (4 < a ∧ a ≤ 5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1322_132282


namespace NUMINAMATH_CALUDE_soccer_league_teams_l1322_132231

theorem soccer_league_teams (total_games : ℕ) (h_games : total_games = 45) : 
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_teams_l1322_132231


namespace NUMINAMATH_CALUDE_initial_salt_concentration_l1322_132249

theorem initial_salt_concentration
  (initial_volume : ℝ)
  (water_added : ℝ)
  (final_salt_percentage : ℝ)
  (h1 : initial_volume = 56)
  (h2 : water_added = 14)
  (h3 : final_salt_percentage = 0.08)
  (h4 : initial_volume * initial_salt_percentage = (initial_volume + water_added) * final_salt_percentage) :
  initial_salt_percentage = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_initial_salt_concentration_l1322_132249


namespace NUMINAMATH_CALUDE_chord_intersection_ratio_l1322_132236

-- Define a circle
variable (circle : Set ℝ × ℝ)

-- Define points E, F, G, H, Q
variable (E F G H Q : ℝ × ℝ)

-- Define that EF and GH are chords of the circle
variable (chord_EF : Set (ℝ × ℝ))
variable (chord_GH : Set (ℝ × ℝ))

-- Define that Q is the intersection point of EF and GH
variable (intersect_Q : Q ∈ chord_EF ∩ chord_GH)

-- Define lengths
def EQ : ℝ := sorry
def FQ : ℝ := sorry
def GQ : ℝ := sorry
def HQ : ℝ := sorry

-- State the theorem
theorem chord_intersection_ratio 
  (h1 : EQ = 4) 
  (h2 : GQ = 10) : 
  FQ / HQ = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_ratio_l1322_132236


namespace NUMINAMATH_CALUDE_cyclist_minimum_speed_l1322_132209

/-- Minimum speed for a cyclist to intercept a car -/
theorem cyclist_minimum_speed (v a b : ℝ) (hv : v > 0) (ha : a > 0) (hb : b > 0) :
  let min_speed := v * b / a
  ∀ (cyclist_speed : ℝ), cyclist_speed ≥ min_speed → 
  ∃ (t : ℝ), t > 0 ∧ 
    cyclist_speed * t = (a^2 + (v*t)^2).sqrt ∧
    cyclist_speed * t ≥ v * t :=
by sorry

end NUMINAMATH_CALUDE_cyclist_minimum_speed_l1322_132209


namespace NUMINAMATH_CALUDE_min_value_inequality_l1322_132269

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 9) : 
  (x^2 + y^2 + 1) / (x + y) + (x^2 + z^2 + 1) / (x + z) + (y^2 + z^2 + 1) / (y + z) ≥ 4.833 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1322_132269


namespace NUMINAMATH_CALUDE_johns_money_left_l1322_132204

/-- Calculates the money John has left after walking his neighbor's dog, buying books, and giving money to his sister. -/
theorem johns_money_left (days_in_april : ℕ) (sundays_in_april : ℕ) (daily_pay : ℕ) (book_cost : ℕ) (sister_money : ℕ) : 
  days_in_april = 30 →
  sundays_in_april = 4 →
  daily_pay = 10 →
  book_cost = 50 →
  sister_money = 50 →
  (days_in_april - sundays_in_april) * daily_pay - (book_cost + sister_money) = 160 :=
by sorry

end NUMINAMATH_CALUDE_johns_money_left_l1322_132204


namespace NUMINAMATH_CALUDE_max_value_range_l1322_132297

/-- The function f(x) = -x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

/-- The theorem stating the range of m for the maximum value of f(x) -/
theorem max_value_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ f m) →
  0 < m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_range_l1322_132297


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l1322_132295

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the point of tangency
def P : ℝ × ℝ := (2, 0)

-- Define the proposed tangent line
def tangentLine (x : ℝ) : ℝ := 2*x - 4

theorem tangent_line_at_P :
  (∀ x : ℝ, HasDerivAt f (tangentLine P.1) P.1) ∧
  f P.1 = tangentLine P.1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l1322_132295


namespace NUMINAMATH_CALUDE_last_two_digits_product_l1322_132247

def last_two_digits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

theorem last_two_digits_product (n : ℤ) : 
  n % 4 = 0 → 
  (let (a, b) := last_two_digits n; a + b = 12) → 
  (let (a, b) := last_two_digits n; a * b = 32 ∨ a * b = 36) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l1322_132247


namespace NUMINAMATH_CALUDE_arcadia_population_growth_l1322_132255

/-- Represents the population of Arcadia at a given year -/
def population (year : ℕ) : ℕ :=
  if year ≤ 2020 then 250
  else 250 * (3 ^ ((year - 2020) / 25))

/-- The year we're trying to prove -/
def target_year : ℕ := 2095

/-- The population threshold we're trying to exceed -/
def population_threshold : ℕ := 6000

theorem arcadia_population_growth :
  (population target_year > population_threshold) ∧
  (∀ y : ℕ, y < target_year → population y ≤ population_threshold) :=
by sorry

end NUMINAMATH_CALUDE_arcadia_population_growth_l1322_132255


namespace NUMINAMATH_CALUDE_binomial_12_6_l1322_132232

theorem binomial_12_6 : Nat.choose 12 6 = 1848 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_6_l1322_132232


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l1322_132225

theorem set_equality_implies_sum_of_powers (a b : ℝ) : 
  ({b, b/a, 0} : Set ℝ) = {a, a+b, 1} → a^2018 + b^2018 = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l1322_132225


namespace NUMINAMATH_CALUDE_inequality_proof_l1322_132241

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^3 + b^3 = c^3) :
  a^2 + b^2 - c^2 > 6*(c - a)*(c - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1322_132241


namespace NUMINAMATH_CALUDE_concrete_cost_theorem_l1322_132291

/-- Calculates the cost of concrete for home foundations -/
theorem concrete_cost_theorem 
  (num_homes : ℕ) 
  (length width height : ℝ) 
  (density : ℝ) 
  (cost_per_pound : ℝ) : 
  num_homes * length * width * height * density * cost_per_pound = 45000 :=
by
  sorry

#check concrete_cost_theorem 3 100 100 0.5 150 0.02

end NUMINAMATH_CALUDE_concrete_cost_theorem_l1322_132291


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1322_132284

theorem container_volume_ratio :
  ∀ (A B : ℚ),
  A > 0 → B > 0 →
  (3/4 : ℚ) * A + (1/4 : ℚ) * B = (7/8 : ℚ) * B →
  A / B = (5/6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1322_132284


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_three_b_plus_c_range_l1322_132288

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  t.c * (t.a * Real.cos t.B - t.b / 2) = t.a^2 - t.b^2

-- Theorem for part I
theorem angle_A_is_pi_over_three (t : Triangle) 
  (h : satisfies_condition t) : t.A = π / 3 := by
  sorry

-- Theorem for part II
theorem b_plus_c_range (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.a = Real.sqrt 3) : 
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_three_b_plus_c_range_l1322_132288


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_l1322_132296

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the complement of A
theorem complement_A : Aᶜ = {x | x < -1 ∨ 2 ≤ x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_l1322_132296


namespace NUMINAMATH_CALUDE_intersection_is_canonical_line_intersection_is_canonical_line_proof_l1322_132292

/-- Represents a plane in 3D space defined by ax + by + cz + d = 0 --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a line in 3D space in canonical form (x-x₀)/a = (y-y₀)/b = (z-z₀)/c --/
structure CanonicalLine where
  x₀ : ℝ
  y₀ : ℝ
  z₀ : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if the given point (x, y, z) lies on the plane --/
def Plane.contains (p : Plane) (x y z : ℝ) : Prop :=
  p.a * x + p.b * y + p.c * z + p.d = 0

/-- Returns true if the given point (x, y, z) lies on the canonical line --/
def CanonicalLine.contains (l : CanonicalLine) (x y z : ℝ) : Prop :=
  (x - l.x₀) / l.a = (y - l.y₀) / l.b ∧ (y - l.y₀) / l.b = (z - l.z₀) / l.c

/-- The main theorem stating that the intersection of the two given planes
    is exactly the line defined by the given canonical equations --/
theorem intersection_is_canonical_line (p₁ p₂ : Plane) (l : CanonicalLine) : Prop :=
  (p₁.a = 4 ∧ p₁.b = 1 ∧ p₁.c = 1 ∧ p₁.d = 2) →
  (p₂.a = 2 ∧ p₂.b = -1 ∧ p₂.c = -3 ∧ p₂.d = 8) →
  (l.x₀ = 1 ∧ l.y₀ = -6 ∧ l.z₀ = 0 ∧ l.a = -2 ∧ l.b = 14 ∧ l.c = -6) →
  ∀ x y z : ℝ, (p₁.contains x y z ∧ p₂.contains x y z) ↔ l.contains x y z

theorem intersection_is_canonical_line_proof : intersection_is_canonical_line 
  { a := 4, b := 1, c := 1, d := 2 }
  { a := 2, b := -1, c := -3, d := 8 }
  { x₀ := 1, y₀ := -6, z₀ := 0, a := -2, b := 14, c := -6 } := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_canonical_line_intersection_is_canonical_line_proof_l1322_132292


namespace NUMINAMATH_CALUDE_sphere_pyramid_inscribed_radius_l1322_132211

/-- A triangular pyramid arrangement of spheres -/
structure SpherePyramid where
  /-- The number of spheres in the arrangement -/
  num_spheres : ℕ
  /-- The radius of each identical sphere in the arrangement -/
  sphere_radius : ℝ
  /-- The radius of the circumscribing sphere -/
  circumscribing_radius : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ
  /-- Each sphere touches at least three others -/
  touches_three : Prop
  /-- The inscribed sphere touches six identical spheres -/
  inscribed_touches_six : Prop
  /-- The number of spheres is ten -/
  sphere_count : num_spheres = 10

/-- Theorem: In a triangular pyramid arrangement of ten identical spheres where each sphere touches
    at least three others, if the radius of the circumscribing sphere is √6 + 1, then the radius of
    the inscribed sphere that touches six identical spheres is √2 - 1. -/
theorem sphere_pyramid_inscribed_radius
  (p : SpherePyramid)
  (h : p.circumscribing_radius = Real.sqrt 6 + 1) :
  p.inscribed_radius = Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_sphere_pyramid_inscribed_radius_l1322_132211


namespace NUMINAMATH_CALUDE_price_conditions_max_basketballs_l1322_132200

/-- Represents the price of a basketball -/
def basketball_price : ℕ := 80

/-- Represents the price of a soccer ball -/
def soccer_price : ℕ := 50

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 60

/-- The maximum allowed total cost -/
def max_cost : ℕ := 4000

/-- Verifies that the prices satisfy the given conditions -/
theorem price_conditions : 
  2 * basketball_price + 3 * soccer_price = 310 ∧
  5 * basketball_price + 2 * soccer_price = 500 := by sorry

/-- Proves that the maximum number of basketballs that can be purchased is 33 -/
theorem max_basketballs :
  ∀ m : ℕ, 
    m ≤ total_balls ∧ 
    m * basketball_price + (total_balls - m) * soccer_price ≤ max_cost →
    m ≤ 33 := by sorry

end NUMINAMATH_CALUDE_price_conditions_max_basketballs_l1322_132200


namespace NUMINAMATH_CALUDE_PB_equation_l1322_132270

-- Define the points A, B, and P
variable (A B P : ℝ × ℝ)

-- Define the conditions
axiom A_on_x_axis : A.2 = 0
axiom B_on_x_axis : B.2 = 0
axiom P_x_coord : P.1 = 2
axiom PA_PB_equal : (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2
axiom PA_equation : ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1 - p.2 + 1 = 0} ↔ (x - P.1) * (A.2 - P.2) = (y - P.2) * (A.1 - P.1)

-- State the theorem
theorem PB_equation :
  ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1 + p.2 - 5 = 0} ↔ (x - P.1) * (B.2 - P.2) = (y - P.2) * (B.1 - P.1) :=
sorry

end NUMINAMATH_CALUDE_PB_equation_l1322_132270


namespace NUMINAMATH_CALUDE_parallelogram_area_l1322_132256

/-- The area of a parallelogram with vertices at (0, 0), (4, 0), (1, 5), and (5, 5) is 20 square units. -/
theorem parallelogram_area : ℝ := by
  -- Define the vertices of the parallelogram
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (4, 0)
  let v3 : ℝ × ℝ := (1, 5)
  let v4 : ℝ × ℝ := (5, 5)

  -- Calculate the area of the parallelogram
  have area : ℝ := 20

  -- Assert that the calculated area is correct
  exact area

end NUMINAMATH_CALUDE_parallelogram_area_l1322_132256


namespace NUMINAMATH_CALUDE_congruence_solution_l1322_132230

theorem congruence_solution (n : ℤ) : 
  (13 * n) % 47 = 5 % 47 → n % 47 = 4 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_l1322_132230


namespace NUMINAMATH_CALUDE_grandfather_animals_l1322_132227

theorem grandfather_animals (h p k s : ℕ) : 
  h + p + k + s = 40 →
  h = 3 * k →
  s - 8 = h + p →
  40 - (1/4) * h + (3/4) * h = 46 →
  h = 12 ∧ p = 2 ∧ k = 4 ∧ s = 22 := by sorry

end NUMINAMATH_CALUDE_grandfather_animals_l1322_132227


namespace NUMINAMATH_CALUDE_B_alone_time_l1322_132280

/-- The time it takes for A and B together to complete the job -/
def time_AB : ℝ := 3

/-- The time it takes for B and C together to complete the job -/
def time_BC : ℝ := 6

/-- The time it takes for A and C together to complete the job -/
def time_AC : ℝ := 3.6

/-- The rate at which A completes the job -/
def rate_A : ℝ := sorry

/-- The rate at which B completes the job -/
def rate_B : ℝ := sorry

/-- The rate at which C completes the job -/
def rate_C : ℝ := sorry

theorem B_alone_time : 
  rate_A + rate_B = 1 / time_AB ∧ 
  rate_B + rate_C = 1 / time_BC ∧ 
  rate_A + rate_C = 1 / time_AC → 
  1 / rate_B = 9 := by sorry

end NUMINAMATH_CALUDE_B_alone_time_l1322_132280


namespace NUMINAMATH_CALUDE_sms_is_fraudulent_l1322_132250

/-- Represents an SMS message -/
structure SMS where
  claims_prize : Bool
  requests_payment : Bool
  recipient_participated : Bool

/-- Represents characteristics of a legitimate contest -/
structure LegitimateContest where
  requires_payment : Bool

/-- Determines if an SMS is fraudulent based on given conditions -/
def is_fraudulent (sms : SMS) (contest : LegitimateContest) : Prop :=
  sms.claims_prize ∧ 
  sms.requests_payment ∧ 
  ¬sms.recipient_participated ∧
  ¬contest.requires_payment

/-- Theorem stating that an SMS with specific characteristics is fraudulent -/
theorem sms_is_fraudulent (sms : SMS) (contest : LegitimateContest) :
  sms.claims_prize = true →
  sms.requests_payment = true →
  sms.recipient_participated = false →
  contest.requires_payment = false →
  is_fraudulent sms contest := by
  sorry

#check sms_is_fraudulent

end NUMINAMATH_CALUDE_sms_is_fraudulent_l1322_132250


namespace NUMINAMATH_CALUDE_constant_sum_list_difference_l1322_132293

/-- A list of four real numbers where the sum of any two adjacent numbers is constant -/
structure ConstantSumList (a b c d : ℝ) : Prop where
  first_pair : a + b = b + c
  second_pair : b + c = c + d

/-- Theorem: In a list [2, x, y, 5] where the sum of any two adjacent numbers is constant, x - y = 3 -/
theorem constant_sum_list_difference (x y : ℝ) (h : ConstantSumList 2 x y 5) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_list_difference_l1322_132293


namespace NUMINAMATH_CALUDE_problem_solution_l1322_132216

theorem problem_solution (a b c d e : ℕ+) 
  (eq1 : a * b + a + b = 182)
  (eq2 : b * c + b + c = 306)
  (eq3 : c * d + c + d = 210)
  (eq4 : d * e + d + e = 156)
  (prod : a * b * c * d * e = Nat.factorial 10) :
  (a : ℤ) - (e : ℤ) = -154 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1322_132216


namespace NUMINAMATH_CALUDE_max_m_value_l1322_132212

/-- The maximum value of m for which f and g satisfy the given conditions -/
theorem max_m_value : ∀ (n : ℝ), 
  (∀ (m : ℝ), (∀ (t : ℝ), (t^2 + m*t + n^2 ≥ 0) ∨ (t^2 + (m+2)*t + n^2 + m + 1 ≥ 0)) → m ≤ 1) ∧
  (∃ (m : ℝ), m = 1 ∧ ∀ (t : ℝ), (t^2 + m*t + n^2 ≥ 0) ∨ (t^2 + (m+2)*t + n^2 + m + 1 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1322_132212


namespace NUMINAMATH_CALUDE_ducks_in_marsh_l1322_132259

/-- The number of ducks in a marsh, given the total number of birds and the number of geese. -/
def num_ducks (total_birds geese : ℕ) : ℕ := total_birds - geese

/-- Theorem stating that there are 37 ducks in the marsh. -/
theorem ducks_in_marsh : num_ducks 95 58 = 37 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_marsh_l1322_132259


namespace NUMINAMATH_CALUDE_second_shift_widget_fraction_l1322_132235

/-- The fraction of total widgets produced by the second shift in a factory --/
theorem second_shift_widget_fraction :
  -- Define the relative productivity of second shift compared to first shift
  ∀ (second_shift_productivity : ℚ)
  -- Define the relative number of employees in first shift compared to second shift
  (first_shift_employees : ℚ),
  -- Condition: Second shift productivity is 2/3 of first shift
  second_shift_productivity = 2 / 3 →
  -- Condition: First shift has 3/4 as many employees as second shift
  first_shift_employees = 3 / 4 →
  -- Conclusion: The fraction of total widgets produced by second shift is 8/17
  (second_shift_productivity * (1 / first_shift_employees)) /
  (1 + second_shift_productivity * (1 / first_shift_employees)) = 8 / 17 := by
sorry

end NUMINAMATH_CALUDE_second_shift_widget_fraction_l1322_132235


namespace NUMINAMATH_CALUDE_mp_nq_perpendicular_equal_length_l1322_132278

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- Represents a quadrilateral with external isosceles right triangles -/
structure Quadrilateral where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint
  D : ComplexPoint
  M : ComplexPoint
  N : ComplexPoint
  P : ComplexPoint
  Q : ComplexPoint
  is_convex : Bool
  AMB_isosceles_right : Bool
  BNC_isosceles_right : Bool
  CPD_isosceles_right : Bool
  DQA_isosceles_right : Bool

/-- The theorem stating that MP and NQ are perpendicular and of equal length -/
theorem mp_nq_perpendicular_equal_length (quad : Quadrilateral) : 
  quad.is_convex ∧ 
  quad.AMB_isosceles_right ∧ 
  quad.BNC_isosceles_right ∧ 
  quad.CPD_isosceles_right ∧ 
  quad.DQA_isosceles_right → 
  (∃ (angle : ℝ), angle = Real.pi / 2 ∧ 
   Complex.arg (Complex.ofReal (quad.P.re - quad.M.re) + Complex.I * (quad.P.im - quad.M.im)) -
   Complex.arg (Complex.ofReal (quad.Q.re - quad.N.re) + Complex.I * (quad.Q.im - quad.N.im)) = angle) ∧
  (Complex.abs (Complex.ofReal (quad.P.re - quad.M.re) + Complex.I * (quad.P.im - quad.M.im)) =
   Complex.abs (Complex.ofReal (quad.Q.re - quad.N.re) + Complex.I * (quad.Q.im - quad.N.im))) := by
  sorry

end NUMINAMATH_CALUDE_mp_nq_perpendicular_equal_length_l1322_132278


namespace NUMINAMATH_CALUDE_basketball_game_total_points_l1322_132201

theorem basketball_game_total_points : 
  ∀ (adam_2pt adam_3pt mada_2pt mada_3pt : ℕ),
    adam_2pt + adam_3pt = 10 →
    mada_2pt + mada_3pt = 11 →
    adam_2pt = mada_3pt →
    2 * adam_2pt + 3 * adam_3pt = 3 * mada_3pt + 2 * mada_2pt →
    2 * adam_2pt + 3 * adam_3pt + 3 * mada_3pt + 2 * mada_2pt = 52 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_total_points_l1322_132201


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_3_intersection_empty_iff_a_less_than_1_l1322_132274

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- Define the universal set U (assuming it's the real numbers)
def U : Set ℝ := Set.univ

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_3 :
  (A 3 ∩ B = {x | (-1 ≤ x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x ≤ 5)}) ∧
  (A 3 ∪ (U \ B) = {x | -1 ≤ x ∧ x ≤ 5}) := by sorry

-- Theorem for part (2)
theorem intersection_empty_iff_a_less_than_1 :
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_3_intersection_empty_iff_a_less_than_1_l1322_132274


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1322_132287

theorem absolute_value_simplification : |(-5^2 - 6 * 2)| = 37 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1322_132287


namespace NUMINAMATH_CALUDE_dinner_cost_l1322_132202

theorem dinner_cost (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (service_rate : ℝ)
  (h_total : total_bill = 34.5)
  (h_tax : tax_rate = 0.095)
  (h_tip : tip_rate = 0.18)
  (h_service : service_rate = 0.05) :
  ∃ (base_cost : ℝ), 
    base_cost * (1 + tax_rate + tip_rate + service_rate) = total_bill ∧ 
    base_cost = 26 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_l1322_132202


namespace NUMINAMATH_CALUDE_units_digit_of_2_power_10_l1322_132252

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The function to calculate 2 to the power of n -/
def powerOfTwo (n : ℕ) : ℕ := 2^n

theorem units_digit_of_2_power_10 : unitsDigit (powerOfTwo 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_power_10_l1322_132252


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1322_132220

theorem complex_sum_theorem :
  let A : ℂ := 3 + 2*I
  let B : ℂ := -3 + I
  let C : ℂ := 1 - 2*I
  let D : ℂ := 4 + 3*I
  A + B + C + D = 5 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1322_132220


namespace NUMINAMATH_CALUDE_green_curlers_count_l1322_132271

def total_curlers : ℕ := 16

def pink_curlers : ℕ := total_curlers / 4

def blue_curlers : ℕ := 2 * pink_curlers

def green_curlers : ℕ := total_curlers - (pink_curlers + blue_curlers)

theorem green_curlers_count : green_curlers = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_curlers_count_l1322_132271


namespace NUMINAMATH_CALUDE_hotel_loss_calculation_l1322_132275

/-- Calculates the loss incurred by a hotel given its operations expenses and the fraction of expenses covered by client payments. -/
def hotel_loss (expenses : ℝ) (payment_fraction : ℝ) : ℝ :=
  expenses - (payment_fraction * expenses)

/-- Theorem stating that a hotel with $100 in expenses and client payments covering 3/4 of expenses incurs a $25 loss. -/
theorem hotel_loss_calculation :
  hotel_loss 100 (3/4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_hotel_loss_calculation_l1322_132275


namespace NUMINAMATH_CALUDE_lower_limit_of_a_l1322_132210

theorem lower_limit_of_a (a b : ℤ) : 
  (a > 0) → 
  (a < 26) → 
  (b > 14) → 
  (b < 31) → 
  (a / b : ℚ) ≥ 4/3 → 
  a ≥ 20 := by
sorry

end NUMINAMATH_CALUDE_lower_limit_of_a_l1322_132210


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1322_132251

theorem quadratic_root_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 8 = 0 →
  3 * q^2 - 5 * q - 8 = 0 →
  p ≠ q →
  (9 * p^4 - 9 * q^4) / (p - q) = 365 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1322_132251


namespace NUMINAMATH_CALUDE_fruit_difference_l1322_132226

/-- Given the number of apples harvested and the ratio of peaches to apples,
    prove that the difference between the number of peaches and apples is 120. -/
theorem fruit_difference (apples : ℕ) (peach_ratio : ℕ) : apples = 60 → peach_ratio = 3 →
  peach_ratio * apples - apples = 120 := by
  sorry

end NUMINAMATH_CALUDE_fruit_difference_l1322_132226


namespace NUMINAMATH_CALUDE_gcd_x_y_eq_25_l1322_132203

/-- The sum of all even integers between 13 and 63 (inclusive) -/
def x : ℕ := (14 + 62) * 25 / 2

/-- The count of even integers between 13 and 63 (inclusive) -/
def y : ℕ := 25

/-- Theorem stating that the greatest common divisor of x and y is 25 -/
theorem gcd_x_y_eq_25 : Nat.gcd x y = 25 := by sorry

end NUMINAMATH_CALUDE_gcd_x_y_eq_25_l1322_132203


namespace NUMINAMATH_CALUDE_fruit_weights_l1322_132286

/-- Represents the fruits in the problem -/
inductive Fruit
| Orange
| Banana
| Mandarin
| Peach
| Apple

/-- Assigns weights to fruits -/
def weight : Fruit → ℕ
| Fruit.Orange => 280
| Fruit.Banana => 170
| Fruit.Mandarin => 100
| Fruit.Peach => 200
| Fruit.Apple => 150

/-- The set of all possible weights -/
def weightSet : Set ℕ := {100, 150, 170, 200, 280}

theorem fruit_weights :
  (∀ f : Fruit, weight f ∈ weightSet) ∧
  (weight Fruit.Peach < weight Fruit.Orange) ∧
  (weight Fruit.Apple < weight Fruit.Banana) ∧
  (weight Fruit.Banana < weight Fruit.Peach) ∧
  (weight Fruit.Mandarin < weight Fruit.Banana) ∧
  (weight Fruit.Apple + weight Fruit.Banana > weight Fruit.Orange) ∧
  (∀ f g : Fruit, f ≠ g → weight f ≠ weight g) := by
  sorry

end NUMINAMATH_CALUDE_fruit_weights_l1322_132286


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1322_132264

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- Prove that the opposite of -2 is 2. -/
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1322_132264


namespace NUMINAMATH_CALUDE_cade_initial_marbles_l1322_132234

/-- The number of marbles Cade gave away -/
def marbles_given : ℕ := 8

/-- The number of marbles Cade has left -/
def marbles_left : ℕ := 79

/-- The initial number of marbles Cade had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem cade_initial_marbles : initial_marbles = 87 := by
  sorry

end NUMINAMATH_CALUDE_cade_initial_marbles_l1322_132234


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1322_132299

theorem polynomial_factorization (x : ℝ) (h : x^3 ≠ 1) :
  x^12 + x^6 + 1 = (x^6 + x^3 + 1) * (x^6 - x^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1322_132299


namespace NUMINAMATH_CALUDE_odd_function_extension_l1322_132214

/-- An odd function on the real line. -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem odd_function_extension (f : ℝ → ℝ) (h : OddFunction f) 
    (h_neg : ∀ x < 0, f x = x * Real.exp (-x)) :
    ∀ x > 0, f x = x * Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1322_132214


namespace NUMINAMATH_CALUDE_steve_final_marbles_l1322_132268

/-- Represents the initial and final marble counts for each person --/
structure MarbleCounts where
  steve_initial : ℕ
  sam_initial : ℕ
  sally_initial : ℕ
  sarah_initial : ℕ
  steve_final : ℕ

/-- Defines the marble distribution scenario --/
def marble_distribution (counts : MarbleCounts) : Prop :=
  counts.sam_initial = 2 * counts.steve_initial ∧
  counts.sally_initial = counts.sam_initial - 5 ∧
  counts.sarah_initial = counts.steve_initial + 3 ∧
  counts.sam_initial - (3 + 3 + 4) = 6 ∧
  counts.steve_final = counts.steve_initial + 3

theorem steve_final_marbles (counts : MarbleCounts) :
  marble_distribution counts → counts.steve_final = 11 := by
  sorry

end NUMINAMATH_CALUDE_steve_final_marbles_l1322_132268


namespace NUMINAMATH_CALUDE_seven_digit_number_count_l1322_132237

def SevenDigitNumber := Fin 7 → Fin 7

def IsAscending (n : SevenDigitNumber) (start fin : Fin 7) : Prop :=
  ∀ i j, start ≤ i ∧ i < j ∧ j ≤ fin → n i < n j

def IsDescending (n : SevenDigitNumber) (start fin : Fin 7) : Prop :=
  ∀ i j, start ≤ i ∧ i < j ∧ j ≤ fin → n i > n j

def IsValidNumber (n : SevenDigitNumber) : Prop :=
  ∀ i j : Fin 7, i ≠ j → n i ≠ n j

theorem seven_digit_number_count :
  (∃ (S : Finset SevenDigitNumber),
    (∀ n ∈ S, IsValidNumber n ∧ IsAscending n 0 5 ∧ IsDescending n 5 6) ∧
    S.card = 6) ∧
  (∃ (T : Finset SevenDigitNumber),
    (∀ n ∈ T, IsValidNumber n ∧ IsAscending n 0 4 ∧ IsDescending n 4 6) ∧
    T.card = 15) := by sorry

end NUMINAMATH_CALUDE_seven_digit_number_count_l1322_132237


namespace NUMINAMATH_CALUDE_sum_of_four_unit_fractions_l1322_132261

theorem sum_of_four_unit_fractions : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sum_of_four_unit_fractions_l1322_132261


namespace NUMINAMATH_CALUDE_dino_hourly_rate_l1322_132265

/-- Dino's monthly income calculation -/
theorem dino_hourly_rate (hours1 hours2 hours3 : ℕ) (rate2 rate3 : ℚ) 
  (expenses leftover : ℚ) (total_income : ℚ) :
  hours1 = 20 →
  hours2 = 30 →
  hours3 = 5 →
  rate2 = 20 →
  rate3 = 40 →
  expenses = 500 →
  leftover = 500 →
  total_income = expenses + leftover →
  total_income = hours1 * (total_income - hours2 * rate2 - hours3 * rate3) / hours1 + hours2 * rate2 + hours3 * rate3 →
  (total_income - hours2 * rate2 - hours3 * rate3) / hours1 = 10 :=
by sorry

end NUMINAMATH_CALUDE_dino_hourly_rate_l1322_132265


namespace NUMINAMATH_CALUDE_fiftieth_day_of_previous_year_l1322_132224

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℤ
  isLeapYear : Bool

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (dayNumber : ℕ) : DayOfWeek :=
  sorry

/-- Determines if two days of the week are equal -/
def dayOfWeekEqual (d1 d2 : DayOfWeek) : Prop :=
  sorry

theorem fiftieth_day_of_previous_year
  (N : Year)
  (h1 : N.isLeapYear = true)
  (h2 : dayOfWeekEqual (dayOfWeek N 250) DayOfWeek.Monday = true)
  (h3 : dayOfWeekEqual (dayOfWeek (Year.mk (N.number + 1) false) 150) DayOfWeek.Tuesday = true) :
  dayOfWeekEqual (dayOfWeek (Year.mk (N.number - 1) false) 50) DayOfWeek.Wednesday = true :=
sorry

end NUMINAMATH_CALUDE_fiftieth_day_of_previous_year_l1322_132224


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1322_132248

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1322_132248


namespace NUMINAMATH_CALUDE_smallest_degree_poly_div_by_30_l1322_132254

/-- A polynomial with coefficients in {-1, 0, 1} -/
def RestrictedPoly (k : ℕ) := {f : Polynomial ℤ // ∀ i, i < k → f.coeff i ∈ ({-1, 0, 1} : Set ℤ)}

/-- A polynomial is divisible by 30 for all positive integers -/
def DivisibleBy30 (f : Polynomial ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → (30 : ℤ) ∣ f.eval n

/-- The theorem stating the smallest degree of a polynomial satisfying the conditions -/
theorem smallest_degree_poly_div_by_30 :
  ∃ (k : ℕ) (f : RestrictedPoly k),
    DivisibleBy30 f.val ∧
    (∀ (j : ℕ) (g : RestrictedPoly j), DivisibleBy30 g.val → k ≤ j) ∧
    k = 10 := by sorry

end NUMINAMATH_CALUDE_smallest_degree_poly_div_by_30_l1322_132254


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1322_132245

theorem smallest_integer_satisfying_inequality : 
  (∀ y : ℤ, y < 8 → (y : ℚ) / 4 + 3 / 7 ≤ 9 / 4) ∧ 
  (8 : ℚ) / 4 + 3 / 7 > 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1322_132245


namespace NUMINAMATH_CALUDE_original_ghee_quantity_l1322_132263

/-- Proves that the original quantity of ghee is 30 kg given the conditions of the problem. -/
theorem original_ghee_quantity (x : ℝ) : 
  (0.5 * x = 0.3 * (x + 20)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_original_ghee_quantity_l1322_132263


namespace NUMINAMATH_CALUDE_log_101600_div_3_l1322_132243

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_101600_div_3 : log (101600 / 3) = 0.1249 := by
  -- Given conditions
  have h1 : log 102 = 0.3010 := by sorry
  have h2 : log 3 = 0.4771 := by sorry

  -- Proof steps
  sorry

end NUMINAMATH_CALUDE_log_101600_div_3_l1322_132243


namespace NUMINAMATH_CALUDE_pencils_given_l1322_132219

theorem pencils_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 9 → total = 65 → given = total - initial :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_given_l1322_132219


namespace NUMINAMATH_CALUDE_nine_point_circle_triangles_l1322_132266

/-- Given 9 points on a circle, this function calculates the number of distinct triangles
    formed by the intersection points of chords inside the circle. --/
def count_triangles (n : ℕ) : ℕ :=
  Nat.choose n 6

/-- Theorem stating that for 9 points on a circle, with chords connecting every pair of points
    and no three chords intersecting at a single point inside the circle, the number of
    distinct triangles formed by the intersection points of these chords inside the circle is 84. --/
theorem nine_point_circle_triangles :
  count_triangles 9 = 84 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_circle_triangles_l1322_132266


namespace NUMINAMATH_CALUDE_f_even_h_odd_l1322_132267

-- Define the functions f and h
def f (x : ℝ) : ℝ := x^2
def h (x : ℝ) : ℝ := x

-- State the theorem
theorem f_even_h_odd : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, h (-x) = -h x) := by
  sorry

end NUMINAMATH_CALUDE_f_even_h_odd_l1322_132267


namespace NUMINAMATH_CALUDE_inequality_proof_l1322_132205

theorem inequality_proof (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1322_132205


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l1322_132207

/-- The weight of a marble statue after three weeks of carving -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let week1_remainder := initial_weight * (1 - 0.25)
  let week2_remainder := week1_remainder * (1 - 0.15)
  let week3_remainder := week2_remainder * (1 - 0.10)
  week3_remainder

/-- The theorem stating the final weight of the statue -/
theorem statue_weight_calculation :
  final_statue_weight 190 = 109.0125 := by
  sorry

#eval final_statue_weight 190

end NUMINAMATH_CALUDE_statue_weight_calculation_l1322_132207


namespace NUMINAMATH_CALUDE_identify_fake_bag_l1322_132222

/-- Represents the number of bags of coins -/
def num_bags : ℕ := 10

/-- Weight of a real coin in grams -/
def real_coin_weight : ℕ := 10

/-- Weight of a fake coin in grams -/
def fake_coin_weight : ℕ := 9

/-- Calculates the sum of first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the total weight if all coins were real -/
def total_real_weight : ℕ := sum_first_n num_bags * real_coin_weight

/-- Represents the measured weight of selected coins -/
def measured_weight (fake_bag : ℕ) : ℕ := total_real_weight - fake_bag

/-- Theorem stating that the bag number with fake coins equals the difference between total_real_weight and measured_weight -/
theorem identify_fake_bag (fake_bag : ℕ) (h : fake_bag ≤ num_bags) :
  fake_bag = total_real_weight - measured_weight fake_bag := by
  sorry

end NUMINAMATH_CALUDE_identify_fake_bag_l1322_132222


namespace NUMINAMATH_CALUDE_average_sales_per_month_l1322_132217

def sales_data : List ℝ := [120, 90, 50, 110, 80, 100]

theorem average_sales_per_month :
  (List.sum sales_data) / (List.length sales_data) = 91.67 := by
  sorry

end NUMINAMATH_CALUDE_average_sales_per_month_l1322_132217


namespace NUMINAMATH_CALUDE_ratio_problem_l1322_132206

theorem ratio_problem (x y : ℕ) (h1 : x + y = 420) (h2 : x = 180) :
  ∃ (a b : ℕ), a = 3 ∧ b = 4 ∧ x * b = y * a :=
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1322_132206


namespace NUMINAMATH_CALUDE_remainder_theorem_l1322_132281

def q (x : ℝ) : ℝ := 2*x^6 - 3*x^4 + 5*x^2 + 3

theorem remainder_theorem (q : ℝ → ℝ) (a : ℝ) :
  q a = (q 2) → q (-2) = 103 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1322_132281


namespace NUMINAMATH_CALUDE_real_part_of_z_l1322_132277

theorem real_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = -1) : 
  Complex.re z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1322_132277


namespace NUMINAMATH_CALUDE_rohan_salary_calculation_l1322_132289

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 12500

/-- The percentage of salary Rohan spends on food -/
def food_expense_percent : ℝ := 40

/-- The percentage of salary Rohan spends on house rent -/
def rent_expense_percent : ℝ := 20

/-- The percentage of salary Rohan spends on entertainment -/
def entertainment_expense_percent : ℝ := 10

/-- The percentage of salary Rohan spends on conveyance -/
def conveyance_expense_percent : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 2500

/-- Theorem stating that given the conditions, Rohan's monthly salary is Rs. 12500 -/
theorem rohan_salary_calculation :
  (food_expense_percent + rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent) / 100 * monthly_salary + savings = monthly_salary :=
by sorry

end NUMINAMATH_CALUDE_rohan_salary_calculation_l1322_132289


namespace NUMINAMATH_CALUDE_sequence_explicit_formula_l1322_132213

theorem sequence_explicit_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 2 * a n + 1) →
  ∀ n : ℕ, n ≥ 1 → a n = (-2) ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_explicit_formula_l1322_132213


namespace NUMINAMATH_CALUDE_expression_value_l1322_132298

theorem expression_value (a b : ℝ) (h : a + 3*b = 0) : 
  a^3 + 3*a^2*b - 2*a - 6*b - 5 = -5 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1322_132298


namespace NUMINAMATH_CALUDE_equilateral_triangle_circle_radii_l1322_132257

/-- For an equilateral triangle with side length a, prove the radii of circumscribed and inscribed circles. -/
theorem equilateral_triangle_circle_radii (a : ℝ) (h : a > 0) :
  ∃ (R r : ℝ),
    R = a * Real.sqrt 3 / 3 ∧
    r = a * Real.sqrt 3 / 6 ∧
    R = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circle_radii_l1322_132257


namespace NUMINAMATH_CALUDE_financial_equation_balance_l1322_132276

theorem financial_equation_balance (f w p : ℂ) : 
  f = 10 → w = -10 + 250 * I → f * p - w = 8000 → p = 799 + 25 * I := by
  sorry

end NUMINAMATH_CALUDE_financial_equation_balance_l1322_132276
