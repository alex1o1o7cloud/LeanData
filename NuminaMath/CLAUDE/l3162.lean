import Mathlib

namespace gcd_lcm_product_90_150_l3162_316224

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end gcd_lcm_product_90_150_l3162_316224


namespace largest_number_less_than_two_l3162_316227

theorem largest_number_less_than_two : 
  let numbers : Finset ℝ := {0.8, 1/2, 0.5}
  ∀ x ∈ numbers, x < 2 → 
  ∃ max ∈ numbers, ∀ y ∈ numbers, y ≤ max ∧ max = 0.8 := by
  sorry

end largest_number_less_than_two_l3162_316227


namespace envelope_weight_l3162_316291

/-- Given 800 envelopes with a total weight of 6.8 kg, prove that one envelope weighs 8.5 grams. -/
theorem envelope_weight (num_envelopes : ℕ) (total_weight_kg : ℝ) :
  num_envelopes = 800 →
  total_weight_kg = 6.8 →
  (total_weight_kg * 1000) / num_envelopes = 8.5 := by
  sorry

end envelope_weight_l3162_316291


namespace sqrt_inequality_l3162_316270

theorem sqrt_inequality : Real.sqrt 2 + Real.sqrt 7 < Real.sqrt 3 + Real.sqrt 6 := by
  sorry

end sqrt_inequality_l3162_316270


namespace cube_cutting_theorem_l3162_316295

/-- Represents a cube with an integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a set of cubes -/
def CubeSet := List Cube

/-- Calculates the total volume of a set of cubes -/
def totalVolume (cubes : CubeSet) : ℕ :=
  cubes.map (fun c => c.edge ^ 3) |>.sum

/-- Represents a cutting and reassembly solution -/
structure Solution where
  pieces : ℕ
  isValid : Bool

/-- Theorem stating the existence of a valid solution -/
theorem cube_cutting_theorem (original : CubeSet) (target : CubeSet) : 
  (original = [Cube.mk 14, Cube.mk 10] ∧ 
   target = [Cube.mk 13, Cube.mk 11, Cube.mk 6] ∧
   totalVolume original = totalVolume target) →
  ∃ (sol : Solution), sol.pieces = 11 ∧ sol.isValid = true := by
  sorry

#check cube_cutting_theorem

end cube_cutting_theorem_l3162_316295


namespace set_equality_l3162_316232

open Set

def U : Set ℝ := univ
def E : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def F : Set ℝ := {x | -1 < x ∧ x < 5}

theorem set_equality : {x : ℝ | -1 < x ∧ x < 2} = (U \ E) ∩ F := by sorry

end set_equality_l3162_316232


namespace expression_evaluation_l3162_316203

theorem expression_evaluation :
  let x : ℝ := 2
  (2 * (x^2 - 1) - 7*x - (2*x^2 - x + 3)) = -17 := by sorry

end expression_evaluation_l3162_316203


namespace age_ratio_theorem_l3162_316234

/-- Represents the ages of two people A and B -/
structure Ages where
  a : ℕ
  b : ℕ

/-- The ratio between A's and B's present ages is 6:3 -/
def present_ratio (ages : Ages) : Prop :=
  2 * ages.b = ages.a

/-- The ratio between A's age 4 years ago and B's age 4 years hence is 1:1 -/
def past_future_ratio (ages : Ages) : Prop :=
  ages.a - 4 = ages.b + 4

/-- The ratio between A's age 4 years hence and B's age 4 years ago is 5:1 -/
def future_past_ratio (ages : Ages) : Prop :=
  5 * (ages.b - 4) = ages.a + 4

/-- Theorem stating the relationship between the given conditions and the result -/
theorem age_ratio_theorem (ages : Ages) :
  present_ratio ages → past_future_ratio ages → future_past_ratio ages :=
by
  sorry

end age_ratio_theorem_l3162_316234


namespace expression_evaluation_l3162_316239

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := 2
  4 * (2 * a^2 * b - a * b^2) - (3 * a * b^2 + 2 * a^2 * b) = -11 := by
  sorry

end expression_evaluation_l3162_316239


namespace cube_difference_l3162_316201

theorem cube_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l3162_316201


namespace flag_arrangement_theorem_l3162_316207

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def N : ℕ := 858

/-- The number of blue flags -/
def blue_flags : ℕ := 12

/-- The number of green flags -/
def green_flags : ℕ := 11

/-- The total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- The number of flagpoles -/
def flagpoles : ℕ := 2

theorem flag_arrangement_theorem :
  (∀ (arrangement : Fin total_flags → Fin flagpoles),
    (∀ pole : Fin flagpoles, ∃ flag : Fin total_flags, arrangement flag = pole) ∧
    (∀ i j : Fin total_flags, i.val + 1 = j.val →
      (i.val < green_flags ∧ j.val < green_flags → arrangement i ≠ arrangement j) ∧
      (i.val ≥ green_flags ∧ j.val ≥ green_flags → arrangement i ≠ arrangement j)) →
    Fintype.card {arrangement : Fin total_flags → Fin flagpoles //
      (∀ pole : Fin flagpoles, ∃ flag : Fin total_flags, arrangement flag = pole) ∧
      (∀ i j : Fin total_flags, i.val + 1 = j.val →
        (i.val < green_flags ∧ j.val < green_flags → arrangement i ≠ arrangement j) ∧
        (i.val ≥ green_flags ∧ j.val ≥ green_flags → arrangement i ≠ arrangement j))} = N) :=
by sorry

end flag_arrangement_theorem_l3162_316207


namespace additional_toothpicks_for_8_steps_l3162_316265

/-- The number of toothpicks needed for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else toothpicks (n - 1) + 2 + 4 * (n - 1)

theorem additional_toothpicks_for_8_steps :
  toothpicks 4 = 30 →
  toothpicks 8 - toothpicks 4 = 88 :=
by sorry

end additional_toothpicks_for_8_steps_l3162_316265


namespace remaining_balls_for_given_n_l3162_316210

/-- Represents the remaining balls after the removal process -/
def RemainingBalls (n : ℕ) : Finset ℕ := sorry

/-- The rule for removing balls -/
def removalRule (n : ℕ) (i : ℕ) : Bool := sorry

/-- The recurrence relation for the remaining balls -/
def F (n : ℕ) (i : ℕ) : ℕ := sorry

theorem remaining_balls_for_given_n (n : ℕ) (h : n ≥ 56) :
  (RemainingBalls 56 = {10, 20, 29, 37, 56}) →
  (n = 57 → RemainingBalls n = {5, 16, 26, 35, 43}) ∧
  (n = 58 → RemainingBalls n = {11, 22, 32, 41, 49}) ∧
  (n = 59 → RemainingBalls n = {17, 28, 38, 47, 55}) ∧
  (n = 60 → RemainingBalls n = {11, 23, 34, 44, 53}) := by
  sorry

end remaining_balls_for_given_n_l3162_316210


namespace units_digit_of_L_L15_l3162_316249

/-- Lucas numbers sequence -/
def Lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => Lucas (n + 1) + Lucas n

/-- The period of the units digit in the Lucas sequence -/
def LucasPeriod : ℕ := 12

theorem units_digit_of_L_L15 : 
  (Lucas (Lucas 15)) % 10 = 7 := by sorry

end units_digit_of_L_L15_l3162_316249


namespace max_diagonals_same_length_l3162_316241

/-- The number of sides in the regular polygon -/
def n : ℕ := 1000

/-- The number of different diagonal lengths in a regular n-gon -/
def num_diagonal_lengths (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-gon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals of each length in a regular n-gon -/
def diagonals_per_length (n : ℕ) : ℕ := n

/-- Theorem: The maximum number of diagonals that can be selected in a regular 1000-gon 
    such that among any three of the chosen diagonals, at least two have the same length is 2000 -/
theorem max_diagonals_same_length : 
  ∃ (k : ℕ), k = 2000 ∧ 
  k ≤ total_diagonals n ∧
  k = 2 * diagonals_per_length n ∧
  ∀ (m : ℕ), m > k → ¬(∀ (a b c : ℕ), a < m ∧ b < m ∧ c < m → a = b ∨ b = c ∨ a = c) :=
sorry

end max_diagonals_same_length_l3162_316241


namespace equation_four_solutions_l3162_316253

theorem equation_four_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x : ℝ, x ∈ s ↔ (x - 2) * (x + 1) * (x + 4) * (x + 7) = 19) ∧
  (s = {(-5 + Real.sqrt 85) / 2, (-5 - Real.sqrt 85) / 2, 
        (-5 + Real.sqrt 5) / 2, (-5 - Real.sqrt 5) / 2}) :=
by sorry

end equation_four_solutions_l3162_316253


namespace bonsai_cost_proof_l3162_316298

/-- The cost of a small bonsai -/
def small_bonsai_cost : ℝ := 30

/-- The cost of a big bonsai -/
def big_bonsai_cost : ℝ := 20

/-- The number of small bonsai sold -/
def small_bonsai_sold : ℕ := 3

/-- The number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- The total earnings -/
def total_earnings : ℝ := 190

theorem bonsai_cost_proof :
  small_bonsai_cost * small_bonsai_sold + big_bonsai_cost * big_bonsai_sold = total_earnings :=
by sorry

end bonsai_cost_proof_l3162_316298


namespace cone_generatrix_length_l3162_316223

/-- Given a cone with base radius √2 and lateral surface that unfolds into a semicircle,
    the length of the generatrix is 2√2. -/
theorem cone_generatrix_length (r : ℝ) (l : ℝ) : 
  r = Real.sqrt 2 →  -- Base radius is √2
  2 * Real.pi * r = Real.pi * l →  -- Lateral surface unfolds into a semicircle
  l = 2 * Real.sqrt 2 :=  -- Length of generatrix is 2√2
by sorry

end cone_generatrix_length_l3162_316223


namespace total_practice_hours_l3162_316212

def monday_hours : ℕ := 6
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := 5
def thursday_hours : ℕ := 7
def friday_hours : ℕ := 3

def total_scheduled_hours : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
def practice_days : ℕ := 5

def player_a_missed_hours : ℕ := 2
def player_b_missed_hours : ℕ := 3

def rainy_day_hours : ℕ := total_scheduled_hours / practice_days

theorem total_practice_hours :
  total_scheduled_hours - (rainy_day_hours + player_a_missed_hours + player_b_missed_hours) = 15 := by
  sorry

end total_practice_hours_l3162_316212


namespace basis_iff_not_parallel_l3162_316282

def is_basis (e₁ e₂ : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (v : ℝ × ℝ), v = (a * e₁.1 + b * e₂.1, a * e₁.2 + b * e₂.2)

def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 = v₁.2 * v₂.1

theorem basis_iff_not_parallel (e₁ e₂ : ℝ × ℝ) :
  is_basis e₁ e₂ ↔ ¬ are_parallel e₁ e₂ :=
sorry

end basis_iff_not_parallel_l3162_316282


namespace similar_triangle_shortest_side_l3162_316229

theorem similar_triangle_shortest_side 
  (a b c : ℝ) 
  (h1 : a = 24) 
  (h2 : c = 25) 
  (h3 : a^2 + b^2 = c^2) 
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h5 : a ≤ b) 
  (h_hypotenuse : ℝ) 
  (h6 : h_hypotenuse = 100) :
  ∃ (x : ℝ), x = 28 ∧ x = (a * h_hypotenuse) / c :=
by sorry

end similar_triangle_shortest_side_l3162_316229


namespace ralph_tv_watching_hours_l3162_316284

/-- The number of hours Ralph watches TV on a weekday -/
def weekday_hours : ℕ := 4

/-- The number of hours Ralph watches TV on a weekend day -/
def weekend_hours : ℕ := 6

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of hours Ralph watches TV in a week -/
def total_hours : ℕ := weekday_hours * weekdays + weekend_hours * weekend_days

theorem ralph_tv_watching_hours :
  total_hours = 32 := by sorry

end ralph_tv_watching_hours_l3162_316284


namespace vector_operations_l3162_316283

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, -4)

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem vector_operations (c : ℝ × ℝ) 
  (h1 : is_unit_vector c) 
  (h2 : is_perpendicular c (a.1 - b.1, a.2 - b.2)) : 
  (2 * a.1 + 3 * b.1, 2 * a.2 + 3 * b.2) = (-5, -10) ∧
  (a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2 = 145 ∧
  (c = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) ∨ c = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2)) := by
  sorry

end vector_operations_l3162_316283


namespace twentieth_term_of_specific_sequence_l3162_316200

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 5 20 = 97 := by
sorry

end twentieth_term_of_specific_sequence_l3162_316200


namespace inequality_proof_l3162_316213

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end inequality_proof_l3162_316213


namespace balls_placement_count_l3162_316261

-- Define the number of balls and boxes
def num_balls : ℕ := 4
def num_boxes : ℕ := 4

-- Define the function to calculate the number of ways to place the balls
def place_balls : ℕ := sorry

-- Theorem statement
theorem balls_placement_count :
  place_balls = 144 := by sorry

end balls_placement_count_l3162_316261


namespace vacuum_savings_theorem_l3162_316222

/-- Calculate the number of weeks needed to save for a vacuum cleaner. -/
def weeks_to_save (initial_amount : ℕ) (weekly_savings : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_amount) + weekly_savings - 1) / weekly_savings

/-- Theorem: It takes 10 weeks to save for the vacuum cleaner. -/
theorem vacuum_savings_theorem :
  weeks_to_save 20 10 120 = 10 := by
  sorry

#eval weeks_to_save 20 10 120

end vacuum_savings_theorem_l3162_316222


namespace arithmetic_mean_problem_l3162_316221

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = c →
  b * c = 400 →
  d = 28 :=
by sorry

end arithmetic_mean_problem_l3162_316221


namespace right_triangle_area_l3162_316231

/-- The area of a right triangle with hypotenuse 13 and shortest side 5 is 30 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) (h4 : a ≤ b) : (1/2) * a * b = 30 := by
  sorry

end right_triangle_area_l3162_316231


namespace function_satisfying_condition_l3162_316251

/-- A function that satisfies f(a f(b)) = a b for all a and b -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * f b) = a * b

/-- The theorem stating that a function satisfying the condition must be either the identity function or its negation -/
theorem function_satisfying_condition (f : ℝ → ℝ) (h : SatisfiesCondition f) :
    (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end function_satisfying_condition_l3162_316251


namespace pizza_cost_l3162_316206

theorem pizza_cost (total_cost : ℝ) (num_pizzas : ℕ) (h1 : total_cost = 24) (h2 : num_pizzas = 3) :
  total_cost / num_pizzas = 8 := by
  sorry

end pizza_cost_l3162_316206


namespace ann_oatmeal_raisin_cookies_l3162_316254

/-- The number of dozens of oatmeal raisin cookies Ann baked -/
def oatmeal_raisin_dozens : ℝ := sorry

/-- The number of dozens of sugar cookies Ann baked -/
def sugar_dozens : ℝ := 2

/-- The number of dozens of chocolate chip cookies Ann baked -/
def chocolate_chip_dozens : ℝ := 4

/-- The number of dozens of oatmeal raisin cookies Ann gave away -/
def oatmeal_raisin_given : ℝ := 2

/-- The number of dozens of sugar cookies Ann gave away -/
def sugar_given : ℝ := 1.5

/-- The number of dozens of chocolate chip cookies Ann gave away -/
def chocolate_chip_given : ℝ := 2.5

/-- The number of dozens of cookies Ann kept -/
def kept_dozens : ℝ := 3

theorem ann_oatmeal_raisin_cookies :
  oatmeal_raisin_dozens = 3 :=
by sorry

end ann_oatmeal_raisin_cookies_l3162_316254


namespace exists_number_satisfying_equation_l3162_316293

theorem exists_number_satisfying_equation : ∃ N : ℝ, (0.47 * N - 0.36 * 1412) + 66 = 6 := by
  sorry

end exists_number_satisfying_equation_l3162_316293


namespace arithmetic_sequence_proof_l3162_316266

/-- Given a sequence {a_n} defined by a_n = 2n + 5, prove it is an arithmetic sequence with common difference 2 -/
theorem arithmetic_sequence_proof (n : ℕ) : ∃ (d : ℝ), d = 2 ∧ ∀ k, (2 * (k + 1) + 5) - (2 * k + 5) = d := by
  sorry

end arithmetic_sequence_proof_l3162_316266


namespace divide_angle_19_degrees_l3162_316289

theorem divide_angle_19_degrees (angle : ℝ) (n : ℕ) : 
  angle = 19 ∧ n = 19 → (angle / n : ℝ) = 1 := by
  sorry

end divide_angle_19_degrees_l3162_316289


namespace inscribed_trapezoid_area_l3162_316276

/-- Represents a trapezoid with an inscribed circle -/
structure InscribedTrapezoid where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorterBase : ℝ
  /-- Assumption that the center of the circle lies on the longer base -/
  centerOnBase : Bool

/-- 
Given a trapezoid ABCD with an inscribed circle of radius 6,
where the center of the circle lies on the base AD,
and BC = 4, prove that the area of the trapezoid is 24√2.
-/
theorem inscribed_trapezoid_area 
  (t : InscribedTrapezoid) 
  (h1 : t.radius = 6) 
  (h2 : t.shorterBase = 4) 
  (h3 : t.centerOnBase = true) : 
  ∃ (area : ℝ), area = 24 * Real.sqrt 2 := by
  sorry

end inscribed_trapezoid_area_l3162_316276


namespace inscribed_circle_radius_l3162_316202

/-- An isosceles trapezoid with specific dimensions and inscribed circles. -/
structure IsoscelesTrapezoidWithCircles where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of sides BC and DA -/
  bc_da : ℝ
  /-- The length of side CD -/
  cd : ℝ
  /-- The radius of circles centered at A and B -/
  outer_radius_large : ℝ
  /-- The radius of circles centered at C and D -/
  outer_radius_small : ℝ
  /-- Constraint: AB = 8 -/
  ab_eq : ab = 8
  /-- Constraint: BC = DA = 6 -/
  bc_da_eq : bc_da = 6
  /-- Constraint: CD = 5 -/
  cd_eq : cd = 5
  /-- Constraint: Radius of circles at A and B is 4 -/
  outer_radius_large_eq : outer_radius_large = 4
  /-- Constraint: Radius of circles at C and D is 3 -/
  outer_radius_small_eq : outer_radius_small = 3

/-- The theorem stating the radius of the inscribed circle tangent to all four outer circles. -/
theorem inscribed_circle_radius (t : IsoscelesTrapezoidWithCircles) :
  ∃ r : ℝ, r = (-105 + 4 * Real.sqrt 141) / 13 ∧
    r > 0 ∧
    ∃ (x y : ℝ),
      x^2 + y^2 = (r + t.outer_radius_large)^2 ∧
      x^2 + (t.bc_da - y)^2 = (r + t.outer_radius_small)^2 ∧
      2*x = t.ab - 2*t.outer_radius_large :=
by sorry


end inscribed_circle_radius_l3162_316202


namespace angle_sum_proof_l3162_316238

theorem angle_sum_proof (x : ℝ) : 
  (6*x + 3*x + 4*x + 2*x = 360) → x = 24 := by
  sorry

end angle_sum_proof_l3162_316238


namespace problem_1_l3162_316240

theorem problem_1 : (-2)^0 + 1 / Real.sqrt 2 - Real.sqrt 9 = Real.sqrt 2 / 2 - 2 := by sorry

end problem_1_l3162_316240


namespace object_speed_approximation_l3162_316235

/-- Given an object traveling 80 feet in 4 seconds, prove that its speed is approximately 13.64 miles per hour, given that 1 mile equals 5280 feet. -/
theorem object_speed_approximation : 
  let distance_feet : ℝ := 80
  let time_seconds : ℝ := 4
  let feet_per_mile : ℝ := 5280
  let seconds_per_hour : ℝ := 3600
  let speed_mph := (distance_feet / feet_per_mile) / (time_seconds / seconds_per_hour)
  ∃ ε > 0, |speed_mph - 13.64| < ε :=
by
  sorry


end object_speed_approximation_l3162_316235


namespace expected_occurrences_100_rolls_l3162_316208

/-- The expected number of times a specific face appears when rolling a fair die multiple times -/
def expected_occurrences (num_rolls : ℕ) : ℚ :=
  num_rolls * (1 : ℚ) / 6

/-- Theorem: The expected number of times a specific face appears when rolling a fair die 100 times is 50/3 -/
theorem expected_occurrences_100_rolls :
  expected_occurrences 100 = 50 / 3 := by
  sorry

end expected_occurrences_100_rolls_l3162_316208


namespace monotonic_increasing_interval_max_min_values_even_function_l3162_316219

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 3

-- Part 1
theorem monotonic_increasing_interval (a : ℝ) (h : a = 2) :
  ∀ x y, x ≤ y ∧ y ≤ 1 → f a x ≤ f a y :=
sorry

-- Part 2
theorem max_min_values_even_function (a : ℝ) 
  (h : ∀ x, f a x = f a (-x)) :
  (∃ x₀ ∈ Set.Icc (-1) 3, f a x₀ = 3 ∧ 
    ∀ x ∈ Set.Icc (-1) 3, f a x ≤ 3) ∧
  (∃ x₁ ∈ Set.Icc (-1) 3, f a x₁ = -6 ∧ 
    ∀ x ∈ Set.Icc (-1) 3, f a x ≥ -6) :=
sorry

end monotonic_increasing_interval_max_min_values_even_function_l3162_316219


namespace perpendicular_bisector_equation_l3162_316259

def is_perpendicular_bisector (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  a * midpoint.1 + b * midpoint.2 + c = 0

theorem perpendicular_bisector_equation (b : ℝ) :
  is_perpendicular_bisector 1 (-1) (-b) (2, 4) (10, -6) → b = 7 := by
  sorry

end perpendicular_bisector_equation_l3162_316259


namespace vertex_is_correct_l3162_316263

/-- The quadratic function f(x) = 3(x+5)^2 - 2 -/
def f (x : ℝ) : ℝ := 3 * (x + 5)^2 - 2

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-5, -2)

theorem vertex_is_correct : 
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end vertex_is_correct_l3162_316263


namespace expression_value_l3162_316248

theorem expression_value : (5^2 - 5 - 12) / (5 - 4) = 8 := by
  sorry

end expression_value_l3162_316248


namespace smallest_n_square_and_cube_l3162_316260

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

theorem smallest_n_square_and_cube : 
  (∀ n : ℕ, n > 0 ∧ is_perfect_square (5*n) ∧ is_perfect_cube (4*n) → n ≥ 80) ∧
  (is_perfect_square (5*80) ∧ is_perfect_cube (4*80)) :=
sorry

end smallest_n_square_and_cube_l3162_316260


namespace solution_range_l3162_316271

theorem solution_range (k : ℝ) : 
  (∃ x : ℝ, x + 2*k = 4*(x + k) + 1 ∧ x < 0) → k > -1/2 := by
  sorry

end solution_range_l3162_316271


namespace rakesh_salary_rakesh_salary_proof_l3162_316218

theorem rakesh_salary : ℝ → Prop :=
  fun salary =>
    let fixed_deposit := 0.15 * salary
    let remaining_after_deposit := salary - fixed_deposit
    let groceries := 0.30 * remaining_after_deposit
    let cash_in_hand := remaining_after_deposit - groceries
    cash_in_hand = 2380 → salary = 4000

-- Proof
theorem rakesh_salary_proof : rakesh_salary 4000 := by
  sorry

end rakesh_salary_rakesh_salary_proof_l3162_316218


namespace tu_yuan_yuan_theorem_l3162_316280

/-- Represents the purchase and sale of "Tu Yuan Yuan" toys -/
structure ToyPurchase where
  first_cost : ℕ
  second_cost : ℕ
  price_increase : ℕ
  min_profit : ℕ

/-- Calculates the quantity of the first purchase -/
def first_quantity (tp : ToyPurchase) : ℕ :=
  sorry

/-- Calculates the minimum selling price -/
def min_selling_price (tp : ToyPurchase) : ℕ :=
  sorry

/-- Theorem stating the correct quantity and minimum selling price -/
theorem tu_yuan_yuan_theorem (tp : ToyPurchase) 
  (h1 : tp.first_cost = 1500)
  (h2 : tp.second_cost = 3500)
  (h3 : tp.price_increase = 5)
  (h4 : tp.min_profit = 1150) :
  first_quantity tp = 50 ∧ min_selling_price tp = 41 := by
  sorry

end tu_yuan_yuan_theorem_l3162_316280


namespace coefficient_x_squared_zero_l3162_316211

theorem coefficient_x_squared_zero (x : ℝ) (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x ≠ 0, f x = (a + 1/x) * (1 + x)^4 ∧ 
   (∃ c₀ c₁ c₃ c₄ : ℝ, ∀ x ≠ 0, f x = c₀ + c₁*x + 0*x^2 + c₃*x^3 + c₄*x^4)) ↔ 
  a = -2/3 :=
sorry

end coefficient_x_squared_zero_l3162_316211


namespace cyclic_win_sets_count_l3162_316299

/-- A round-robin tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (wins_per_team : ℕ)
  (losses_per_team : ℕ)
  (h_round_robin : wins_per_team + losses_per_team = num_teams - 1)
  (h_no_ties : True)

/-- The number of sets of three teams with cyclic wins -/
def cyclic_win_sets (t : Tournament) : ℕ := sorry

/-- The theorem to be proved -/
theorem cyclic_win_sets_count 
  (t : Tournament) 
  (h_num_teams : t.num_teams = 20) 
  (h_wins : t.wins_per_team = 12) 
  (h_losses : t.losses_per_team = 7) : 
  cyclic_win_sets t = 570 := by sorry

end cyclic_win_sets_count_l3162_316299


namespace group_size_l3162_316220

theorem group_size (average_increase : ℝ) (old_weight new_weight : ℝ) :
  average_increase = 2.5 ∧
  old_weight = 40 ∧
  new_weight = 60 →
  (new_weight - old_weight) / average_increase = 8 := by
sorry

end group_size_l3162_316220


namespace algebra_test_female_count_l3162_316257

theorem algebra_test_female_count :
  ∀ (total_average : ℝ) (male_count : ℕ) (male_average female_average : ℝ),
    total_average = 90 →
    male_count = 8 →
    male_average = 85 →
    female_average = 92 →
    ∃ (female_count : ℕ),
      (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
      female_count = 20 :=
by sorry

end algebra_test_female_count_l3162_316257


namespace book_selection_theorem_l3162_316215

theorem book_selection_theorem (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  (Nat.choose (n - 1) (m - 1)) = 35 := by
  sorry

end book_selection_theorem_l3162_316215


namespace carnival_total_cost_l3162_316214

def carnival_cost (bumper_rides_mara : ℕ) (space_rides_riley : ℕ) (ferris_rides_each : ℕ)
  (bumper_cost : ℕ) (space_cost : ℕ) (ferris_cost : ℕ) : ℕ :=
  bumper_rides_mara * bumper_cost +
  space_rides_riley * space_cost +
  2 * ferris_rides_each * ferris_cost

theorem carnival_total_cost :
  carnival_cost 2 4 3 2 4 5 = 50 := by
  sorry

end carnival_total_cost_l3162_316214


namespace certain_number_equation_l3162_316272

theorem certain_number_equation (x : ℚ) : 4 / (1 + 3 / x) = 1 → x = 1 := by
  sorry

end certain_number_equation_l3162_316272


namespace inequality_range_m_l3162_316244

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := x^2 + |x - 1|

/-- The theorem stating the range of m for which the inequality always holds -/
theorem inequality_range_m :
  (∀ x : ℝ, f x ≥ (m + 2) * x - 1) ↔ m ∈ Set.Icc (-3 - 2 * Real.sqrt 2) 0 :=
sorry

end inequality_range_m_l3162_316244


namespace power_of_negative_product_l3162_316217

theorem power_of_negative_product (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end power_of_negative_product_l3162_316217


namespace reyansh_farm_water_ratio_l3162_316264

/-- Represents the farm owned by Mr. Reyansh -/
structure Farm where
  num_cows : ℕ
  cow_water_daily : ℕ
  sheep_cow_ratio : ℕ
  total_water_weekly : ℕ

/-- Calculates the ratio of daily water consumption of a sheep to a cow -/
def water_consumption_ratio (f : Farm) : Rat :=
  let cow_water_weekly := f.num_cows * f.cow_water_daily * 7
  let sheep_water_weekly := f.total_water_weekly - cow_water_weekly
  let num_sheep := f.num_cows * f.sheep_cow_ratio
  let sheep_water_daily := sheep_water_weekly / (7 * num_sheep)
  sheep_water_daily / f.cow_water_daily

/-- Theorem stating that the water consumption ratio for Mr. Reyansh's farm is 1:4 -/
theorem reyansh_farm_water_ratio :
  let f : Farm := {
    num_cows := 40,
    cow_water_daily := 80,
    sheep_cow_ratio := 10,
    total_water_weekly := 78400
  }
  water_consumption_ratio f = 1 / 4 := by
  sorry


end reyansh_farm_water_ratio_l3162_316264


namespace percentage_sold_first_day_l3162_316262

-- Define the initial number of watermelons
def initial_watermelons : ℕ := 10 * 12

-- Define the number of watermelons left after two days of selling
def remaining_watermelons : ℕ := 54

-- Define the percentage sold on the second day
def second_day_percentage : ℚ := 1 / 4

-- Theorem to prove
theorem percentage_sold_first_day :
  ∃ (p : ℚ), 0 ≤ p ∧ p ≤ 1 ∧
  (1 - second_day_percentage) * ((1 - p) * initial_watermelons) = remaining_watermelons ∧
  p = 2 / 5 := by
  sorry

end percentage_sold_first_day_l3162_316262


namespace t_values_l3162_316267

theorem t_values (t : ℝ) : 
  let M : Set ℝ := {1, 3, t}
  let N : Set ℝ := {t^2 - t + 1}
  (M ∪ N = M) → (t = 0 ∨ t = 2 ∨ t = -1) := by
sorry

end t_values_l3162_316267


namespace point_on_y_axis_l3162_316255

/-- Given that point P(2-a, a-3) lies on the y-axis, prove that a = 2 -/
theorem point_on_y_axis (a : ℝ) : (2 - a = 0) → a = 2 := by
  sorry

end point_on_y_axis_l3162_316255


namespace paper_length_wrapped_around_cylinder_l3162_316296

/-- Calculates the length of paper wrapped around a cylindrical tube. -/
theorem paper_length_wrapped_around_cylinder
  (initial_diameter : ℝ)
  (paper_width : ℝ)
  (num_wraps : ℕ)
  (final_diameter : ℝ)
  (h1 : initial_diameter = 3)
  (h2 : paper_width = 4)
  (h3 : num_wraps = 400)
  (h4 : final_diameter = 11) :
  ∃ (paper_length : ℝ), paper_length = 28 * π ∧ paper_length * 100 = π * (num_wraps * (initial_diameter + final_diameter)) :=
by sorry

end paper_length_wrapped_around_cylinder_l3162_316296


namespace bookshop_inventory_bookshop_current_inventory_l3162_316288

/-- Calculates the current number of books in a bookshop after a weekend of sales and a new shipment. -/
theorem bookshop_inventory (
  initial_inventory : ℕ
  ) (saturday_in_store : ℕ) (saturday_online : ℕ)
  (sunday_in_store_multiplier : ℕ) (sunday_online_increase : ℕ)
  (new_shipment : ℕ) : ℕ :=
  let sunday_in_store := sunday_in_store_multiplier * saturday_in_store
  let sunday_online := saturday_online + sunday_online_increase
  let total_sold := saturday_in_store + saturday_online + sunday_in_store + sunday_online
  let net_change := new_shipment - total_sold
  initial_inventory + net_change

/-- The bookshop currently has 502 books. -/
theorem bookshop_current_inventory :
  bookshop_inventory 743 37 128 2 34 160 = 502 := by
  sorry

end bookshop_inventory_bookshop_current_inventory_l3162_316288


namespace jills_work_days_month3_l3162_316247

def daily_rate_month1 : ℕ := 10
def daily_rate_month2 : ℕ := 2 * daily_rate_month1
def daily_rate_month3 : ℕ := daily_rate_month2
def days_per_month : ℕ := 30
def total_earnings : ℕ := 1200

def earnings_month1 : ℕ := daily_rate_month1 * days_per_month
def earnings_month2 : ℕ := daily_rate_month2 * days_per_month

theorem jills_work_days_month3 :
  (total_earnings - earnings_month1 - earnings_month2) / daily_rate_month3 = 15 := by
  sorry

end jills_work_days_month3_l3162_316247


namespace polynomial_division_remainder_l3162_316242

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ,
  x^5 + 3*x^3 + 1 = (x - 3)^2 * q + (324*x - 488) :=
sorry

end polynomial_division_remainder_l3162_316242


namespace multiples_count_l3162_316228

def count_multiples (n : ℕ) : ℕ := 
  (n.div 3 + n.div 4 - n.div 12) - (n.div 15 + n.div 20 - n.div 60)

theorem multiples_count : count_multiples 2010 = 804 := by sorry

end multiples_count_l3162_316228


namespace smallest_integer_satisfying_inequalities_l3162_316287

theorem smallest_integer_satisfying_inequalities :
  ∀ x : ℤ, x < 3 * x - 12 ∧ x > 0 → x ≥ 7 :=
by sorry

end smallest_integer_satisfying_inequalities_l3162_316287


namespace sqrt_fraction_eval_l3162_316275

theorem sqrt_fraction_eval (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - (2 * x - 3) / (x + 1))) = Complex.I * Real.sqrt (x^2 - 3*x - 4) := by
  sorry

end sqrt_fraction_eval_l3162_316275


namespace range_of_linear_function_l3162_316233

def g (c d x : ℝ) : ℝ := c * x + d

theorem range_of_linear_function (c d : ℝ) (hc : c < 0) :
  ∀ y ∈ Set.range (g c d),
    ∃ x ∈ Set.Icc (-1 : ℝ) 1,
      y = g c d x ∧ c + d ≤ y ∧ y ≤ -c + d :=
by sorry

end range_of_linear_function_l3162_316233


namespace gasohol_calculation_l3162_316230

/-- The amount of gasohol initially in the tank (in liters) -/
def initial_gasohol : ℝ := 27

/-- The fraction of ethanol in the initial mixture -/
def initial_ethanol_fraction : ℝ := 0.05

/-- The fraction of ethanol in the desired mixture -/
def desired_ethanol_fraction : ℝ := 0.10

/-- The amount of pure ethanol added (in liters) -/
def added_ethanol : ℝ := 1.5

theorem gasohol_calculation :
  initial_gasohol * initial_ethanol_fraction + added_ethanol =
  desired_ethanol_fraction * (initial_gasohol + added_ethanol) := by
  sorry

end gasohol_calculation_l3162_316230


namespace smallest_7digit_binary_proof_l3162_316290

/-- The smallest positive integer with a 7-digit binary representation -/
def smallest_7digit_binary : ℕ := 64

/-- The binary representation of a natural number -/
def binary_representation (n : ℕ) : List Bool :=
  sorry

/-- The length of the binary representation of a natural number -/
def binary_length (n : ℕ) : ℕ :=
  (binary_representation n).length

theorem smallest_7digit_binary_proof :
  (∀ m : ℕ, m < smallest_7digit_binary → binary_length m < 7) ∧
  binary_length smallest_7digit_binary = 7 := by
  sorry

end smallest_7digit_binary_proof_l3162_316290


namespace pencil_case_solution_l3162_316279

/-- Represents the cost and quantity of pencil cases --/
structure PencilCases where
  cost_A : ℚ
  cost_B : ℚ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Conditions for the pencil case problem --/
def PencilCaseProblem (p : PencilCases) : Prop :=
  p.cost_B = p.cost_A + 2 ∧
  800 / p.cost_A = 1000 / p.cost_B ∧
  p.quantity_A = 3 * p.quantity_B - 50 ∧
  p.quantity_A + p.quantity_B ≤ 910 ∧
  12 * p.quantity_A + 15 * p.quantity_B - 
  (p.cost_A * p.quantity_A + p.cost_B * p.quantity_B) > 3795

/-- The main theorem to prove --/
theorem pencil_case_solution (p : PencilCases) 
  (h : PencilCaseProblem p) : 
  p.cost_A = 8 ∧ 
  p.cost_B = 10 ∧ 
  p.quantity_B ≤ 240 ∧ 
  (∃ n : ℕ, n = 5 ∧ 
    ∀ m : ℕ, 236 ≤ m ∧ m ≤ 240 → 
      (12 * (3 * m - 50) + 15 * m - (8 * (3 * m - 50) + 10 * m) > 3795)) := by
  sorry


end pencil_case_solution_l3162_316279


namespace perpendicular_conditions_l3162_316294

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (line_perp_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perp_plane : Plane → Plane → Prop)
variable (plane_parallel_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_conditions 
  (a b : Line) (α β : Plane) :
  (line_perp_plane a α ∧ line_perp_plane b β ∧ plane_perp_plane α β → perpendicular a b) ∧
  (line_in_plane a α ∧ line_perp_plane b β ∧ plane_parallel_plane α β → perpendicular a b) ∧
  (line_perp_plane a α ∧ line_parallel_plane b β ∧ plane_parallel_plane α β → perpendicular a b) :=
sorry

end perpendicular_conditions_l3162_316294


namespace probability_at_least_six_sevens_l3162_316246

-- Define the number of sides on the die
def die_sides : ℕ := 8

-- Define the number of rolls
def num_rolls : ℕ := 7

-- Define the minimum number of successful rolls required
def min_successes : ℕ := 6

-- Define the minimum value considered a success
def success_value : ℕ := 7

-- Function to calculate the probability of a single success
def single_success_prob : ℚ := (die_sides - success_value + 1) / die_sides

-- Function to calculate the probability of exactly k successes in n rolls
def exact_success_prob (n k : ℕ) : ℚ :=
  Nat.choose n k * single_success_prob^k * (1 - single_success_prob)^(n - k)

-- Theorem statement
theorem probability_at_least_six_sevens :
  (exact_success_prob num_rolls min_successes + exact_success_prob num_rolls (num_rolls)) = 11 / 2048 := by
  sorry

end probability_at_least_six_sevens_l3162_316246


namespace men_to_women_percentage_l3162_316236

theorem men_to_women_percentage (men women : ℕ) (h : women = men / 2) :
  (men : ℚ) / (women : ℚ) * 100 = 200 := by
  sorry

end men_to_women_percentage_l3162_316236


namespace equation_solution_l3162_316278

theorem equation_solution :
  ∃ x : ℝ, x ≠ -2 ∧ (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 → x = 2 :=
by sorry

end equation_solution_l3162_316278


namespace age_ratio_12_years_ago_l3162_316237

-- Define Neha's current age and her mother's current age
def neha_age : ℕ := sorry
def mother_age : ℕ := 60

-- Define the relationship between their ages 12 years ago
axiom past_relation : ∃ x : ℚ, mother_age - 12 = x * (neha_age - 12)

-- Define the relationship between their ages 12 years from now
axiom future_relation : mother_age + 12 = 2 * (neha_age + 12)

-- Theorem to prove
theorem age_ratio_12_years_ago : 
  (mother_age - 12) / (neha_age - 12) = 4 := by sorry

end age_ratio_12_years_ago_l3162_316237


namespace log_expression_equals_four_l3162_316258

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_four :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 4 := by
  sorry

end log_expression_equals_four_l3162_316258


namespace caiden_roofing_cost_l3162_316226

/-- Calculates the cost of remaining metal roofing needed -/
def roofing_cost (total_required : ℕ) (free_provided : ℕ) (cost_per_foot : ℕ) : ℕ :=
  (total_required - free_provided) * cost_per_foot

/-- Theorem stating the cost calculation for Mr. Caiden's roofing -/
theorem caiden_roofing_cost :
  roofing_cost 300 250 8 = 400 := by
  sorry

end caiden_roofing_cost_l3162_316226


namespace overtake_time_l3162_316274

/-- The time when b starts relative to a's start time. -/
def b_start_time : ℝ := 15

/-- The speed of person a in km/hr. -/
def speed_a : ℝ := 30

/-- The speed of person b in km/hr. -/
def speed_b : ℝ := 40

/-- The speed of person k in km/hr. -/
def speed_k : ℝ := 60

/-- The time when k starts relative to a's start time. -/
def k_start_time : ℝ := 10

theorem overtake_time (t : ℝ) : 
  speed_a * t = speed_b * (t - b_start_time) ∧ 
  speed_a * t = speed_k * (t - k_start_time) → 
  b_start_time = 15 := by sorry

end overtake_time_l3162_316274


namespace card_pack_size_l3162_316204

theorem card_pack_size (prob_not_face : ℝ) (num_face_cards : ℕ) (h1 : prob_not_face = 0.6923076923076923) (h2 : num_face_cards = 12) : 
  ∃ n : ℕ, n = 39 ∧ (n - num_face_cards : ℝ) / n = prob_not_face := by
sorry

end card_pack_size_l3162_316204


namespace dimitri_burger_consumption_l3162_316286

/-- Dimitri's burger consumption problem -/
theorem dimitri_burger_consumption (burgers_per_day : ℕ) 
  (calories_per_burger : ℕ) (total_calories : ℕ) (days : ℕ) :
  burgers_per_day * calories_per_burger * days = total_calories →
  calories_per_burger = 20 →
  total_calories = 120 →
  days = 2 →
  burgers_per_day = 3 := by
sorry

end dimitri_burger_consumption_l3162_316286


namespace peach_apple_pear_pricing_l3162_316209

theorem peach_apple_pear_pricing (x y z : ℝ) 
  (h1 : 7 * x = y + 2 * z)
  (h2 : 7 * y = 10 * z + x) :
  12 * y = 18 * z := by sorry

end peach_apple_pear_pricing_l3162_316209


namespace trig_ratios_for_point_on_terminal_side_l3162_316250

/-- Given a point P(3m, -2m) where m < 0 lying on the terminal side of angle α,
    prove the trigonometric ratios for α. -/
theorem trig_ratios_for_point_on_terminal_side (m : ℝ) (α : ℝ) 
  (h1 : m < 0) 
  (h2 : ∃ (r : ℝ), r > 0 ∧ 3 * m = r * Real.cos α ∧ -2 * m = r * Real.sin α) :
  Real.sin α = 2 * Real.sqrt 13 / 13 ∧ 
  Real.cos α = -3 * Real.sqrt 13 / 13 ∧ 
  Real.tan α = -2 / 3 := by
sorry


end trig_ratios_for_point_on_terminal_side_l3162_316250


namespace expression_evaluation_l3162_316245

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end expression_evaluation_l3162_316245


namespace isosceles_triangle_perimeter_l3162_316273

theorem isosceles_triangle_perimeter (m : ℝ) : 
  (2 : ℝ) ^ 2 - (5 + m) * 2 + 5 * m = 0 →
  ∃ (a b : ℝ), a ^ 2 - (5 + m) * a + 5 * m = 0 ∧
                b ^ 2 - (5 + m) * b + 5 * m = 0 ∧
                a ≠ b ∧
                (a = 2 ∨ b = 2) ∧
                (a + a + b = 12 ∨ a + b + b = 12) := by
  sorry

end isosceles_triangle_perimeter_l3162_316273


namespace problem_proof_l3162_316277

theorem problem_proof : (-1)^2023 - (-1/4)^0 + 2 * Real.cos (π/3) = -1 := by
  sorry

end problem_proof_l3162_316277


namespace inequality_solution_set_l3162_316281

theorem inequality_solution_set : 
  {x : ℝ | x / (x - 1) + (x + 2) / (2 * x) ≥ 3} = 
  {x : ℝ | (0 < x ∧ x ≤ 1/3) ∨ (1 < x ∧ x ≤ 2)} := by sorry

end inequality_solution_set_l3162_316281


namespace parabolic_trajectory_falls_within_interval_l3162_316269

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabolic trajectory -/
structure Trajectory where
  a : ℝ
  c : ℝ

/-- Check if a trajectory passes through a point -/
def passesThrough (t : Trajectory) (p : Point) : Prop :=
  p.y = t.a * p.x^2 + t.c

/-- Check if a trajectory intersects with a given x-coordinate at or below a certain y-coordinate -/
def intersectsAt (t : Trajectory) (x y : ℝ) : Prop :=
  t.a * x^2 + t.c ≤ y

theorem parabolic_trajectory_falls_within_interval 
  (t : Trajectory) 
  (A : Point) 
  (P : Point) 
  (D : Point) :
  t.a < 0 →
  A.x = 0 ∧ A.y = 9 →
  P.x = 2 ∧ P.y = 8.1 →
  D.x = 6 ∧ D.y = 7 →
  passesThrough t A →
  passesThrough t P →
  intersectsAt t D.x D.y :=
by sorry

end parabolic_trajectory_falls_within_interval_l3162_316269


namespace floating_state_exists_l3162_316285

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ

/-- Represents the state of a floating polyhedron -/
structure FloatingState (p : ConvexPolyhedron) where
  submergedVolume : ℝ
  submergedSurfaceArea : ℝ
  volumeRatio : submergedVolume = 0.9 * p.volume
  surfaceAreaRatio : submergedSurfaceArea < 0.5 * p.surfaceArea

/-- Theorem stating that the described floating state is possible -/
theorem floating_state_exists : ∃ (p : ConvexPolyhedron), ∃ (s : FloatingState p), True := by
  sorry

end floating_state_exists_l3162_316285


namespace number_of_boys_l3162_316256

theorem number_of_boys (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  girls = 120 →
  boys + girls = total →
  3 * total = 8 * boys →
  boys = 72 := by
sorry

end number_of_boys_l3162_316256


namespace brians_purchased_animals_ratio_l3162_316268

theorem brians_purchased_animals_ratio (initial_horses : ℕ) (initial_sheep : ℕ) (initial_chickens : ℕ) (gifted_goats : ℕ) (male_animals : ℕ) : 
  initial_horses = 100 →
  initial_sheep = 29 →
  initial_chickens = 9 →
  gifted_goats = 37 →
  male_animals = 53 →
  (initial_horses + initial_sheep + initial_chickens - (2 * male_animals - gifted_goats)) * 2 = initial_horses + initial_sheep + initial_chickens :=
by sorry

end brians_purchased_animals_ratio_l3162_316268


namespace complementary_angles_difference_l3162_316216

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 → x / y = 5 → |x - y| = 60 := by
  sorry

end complementary_angles_difference_l3162_316216


namespace second_caterer_cheaper_at_34_l3162_316205

/-- Caterer pricing structure -/
structure CatererPricing where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculate total cost for a caterer given number of people -/
def totalCost (pricing : CatererPricing) (people : ℕ) : ℕ :=
  pricing.basicFee + pricing.perPersonFee * people

/-- First caterer's pricing -/
def firstCaterer : CatererPricing :=
  { basicFee := 150, perPersonFee := 18 }

/-- Second caterer's pricing -/
def secondCaterer : CatererPricing :=
  { basicFee := 250, perPersonFee := 15 }

/-- Theorem: 34 is the least number of people for which the second caterer is less expensive -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost firstCaterer n ≤ totalCost secondCaterer n) ∧
  (totalCost secondCaterer 34 < totalCost firstCaterer 34) :=
by sorry

end second_caterer_cheaper_at_34_l3162_316205


namespace cubic_sum_equals_36_l3162_316297

theorem cubic_sum_equals_36 (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^3 + b^3 = 36 := by
  sorry

end cubic_sum_equals_36_l3162_316297


namespace class_size_l3162_316292

/-- Represents the number of students in a class with English and German courses -/
structure ClassEnrollment where
  total : ℕ
  bothSubjects : ℕ
  onlyEnglish : ℕ
  onlyGerman : ℕ
  germanTotal : ℕ

/-- Theorem stating the total number of students in the class -/
theorem class_size (c : ClassEnrollment) 
  (h1 : c.bothSubjects = 12)
  (h2 : c.germanTotal = 22)
  (h3 : c.onlyEnglish = 18)
  (h4 : c.germanTotal = c.bothSubjects + c.onlyGerman)
  (h5 : c.total = c.onlyEnglish + c.onlyGerman + c.bothSubjects) :
  c.total = 40 := by
  sorry


end class_size_l3162_316292


namespace total_population_of_two_villages_l3162_316252

/-- Given two villages A and B with the following properties:
    - 90% of Village A's population is 23040
    - 80% of Village B's population is 17280
    - Village A has three times as many children as Village B
    - The adult population is equally distributed between the two villages
    Prove that the total population of both villages combined is 47,200 -/
theorem total_population_of_two_villages :
  ∀ (population_A population_B children_A children_B : ℕ),
    (population_A : ℚ) * (9 / 10) = 23040 →
    (population_B : ℚ) * (4 / 5) = 17280 →
    children_A = 3 * children_B →
    population_A - children_A = population_B - children_B →
    population_A + population_B = 47200 := by
  sorry

#eval 47200

end total_population_of_two_villages_l3162_316252


namespace street_trees_l3162_316243

theorem street_trees (road_length : ℝ) (tree_spacing : ℝ) (h1 : road_length = 268.8) (h2 : tree_spacing = 6.4) : 
  ⌊road_length / tree_spacing⌋ + 2 = 43 := by
  sorry

end street_trees_l3162_316243


namespace triangle_side_length_l3162_316225

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2 / 3 →
  b^2 + c^2 - a^2 = 2 * b * c * Real.cos A →
  b = 3 :=
by sorry

end triangle_side_length_l3162_316225
