import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_l3520_352066

theorem inequality_solution (x y : ℝ) :
  (4 * Real.sin x - Real.sqrt (Real.cos y) - Real.sqrt (Real.cos y - 16 * (Real.cos x)^2 + 12) ≥ 2) ↔
  (∃ (n k : ℤ), x = ((-1)^n * π / 6 + 2 * n * π) ∧ y = (π / 2 + k * π)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3520_352066


namespace NUMINAMATH_CALUDE_meeting_point_2015_l3520_352083

/-- Represents a point on a line segment --/
structure Point where
  position : ℝ
  deriving Inhabited

/-- Represents an object moving on a line segment --/
structure MovingObject where
  startPoint : Point
  speed : ℝ
  startTime : ℝ
  deriving Inhabited

/-- Calculates the meeting point of two objects --/
def meetingPoint (obj1 obj2 : MovingObject) : Point :=
  sorry

/-- Theorem: The 2015th meeting point is the same as the 1st meeting point --/
theorem meeting_point_2015 (A B : Point) (obj1 obj2 : MovingObject) :
  obj1.startPoint = A ∧ obj2.startPoint = B →
  obj1.speed > 0 ∧ obj2.speed > 0 →
  meetingPoint obj1 obj2 = meetingPoint obj1 obj2 :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_2015_l3520_352083


namespace NUMINAMATH_CALUDE_inequality_proof_l3520_352045

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3520_352045


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3520_352001

theorem cube_equation_solution :
  ∃! x : ℝ, (x - 5)^3 = (1/27)⁻¹ :=
by
  -- The unique solution is x = 8
  use 8
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3520_352001


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3520_352057

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if one of its asymptotes is y = 3x/5, then a = 5. -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = 3 * x / 5) →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3520_352057


namespace NUMINAMATH_CALUDE_total_production_cost_l3520_352035

def initial_cost_per_episode : ℕ := 100000
def cost_increase_rate : ℚ := 1.2
def initial_episodes : ℕ := 12
def season_2_increase : ℚ := 1.3
def subsequent_seasons_increase : ℚ := 1.1
def final_season_decrease : ℚ := 0.85
def total_seasons : ℕ := 7

def calculate_total_cost : ℕ := sorry

theorem total_production_cost :
  calculate_total_cost = 25673856 := by sorry

end NUMINAMATH_CALUDE_total_production_cost_l3520_352035


namespace NUMINAMATH_CALUDE_function_shift_theorem_l3520_352058

/-- Given a function f(x) = sin(ωx + π/6) where ω > 0, and the distance between
    adjacent symmetry axes is π/2, prove that the function g(x) obtained by
    shifting f(x) to the left by π/6 units is equal to cos(2x). -/
theorem function_shift_theorem (ω : ℝ) (h1 : ω > 0) :
  let f : ℝ → ℝ := fun x ↦ Real.sin (ω * x + π / 6)
  let symmetry_axis_distance : ℝ := π / 2
  let g : ℝ → ℝ := fun x ↦ f (x + π / 6)
  (∀ x : ℝ, g x = Real.cos (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_function_shift_theorem_l3520_352058


namespace NUMINAMATH_CALUDE_one_thirds_in_eleven_halves_l3520_352041

theorem one_thirds_in_eleven_halves : (11 / 2) / (1 / 3) = 33 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_eleven_halves_l3520_352041


namespace NUMINAMATH_CALUDE_trapezoid_area_is_80_l3520_352067

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid :=
  (long_base : ℝ)
  (base_angle : ℝ)
  (h : 0 < long_base)
  (angle_h : 0 < base_angle ∧ base_angle < π / 2)

/-- The area of an isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem trapezoid_area_is_80 (t : IsoscelesTrapezoid) 
  (h1 : t.long_base = 16)
  (h2 : t.base_angle = Real.arcsin 0.8) :
  trapezoid_area t = 80 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_80_l3520_352067


namespace NUMINAMATH_CALUDE_project_completion_time_l3520_352028

/-- The time taken for teams A and D to complete a project given the completion times of other team combinations -/
theorem project_completion_time (t_AB t_BC t_CD : ℝ) (h_AB : t_AB = 20) (h_BC : t_BC = 60) (h_CD : t_CD = 30) :
  1 / (1 / t_AB + 1 / t_CD - 1 / t_BC) = 15 := by
  sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l3520_352028


namespace NUMINAMATH_CALUDE_congruence_solution_l3520_352061

theorem congruence_solution (n : ℤ) : 
  (15 * n) % 47 = 9 ↔ n % 47 = 18 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l3520_352061


namespace NUMINAMATH_CALUDE_nancy_water_intake_percentage_l3520_352031

/-- Given Nancy's daily water intake and body weight, calculate the percentage of her body weight she drinks in water. -/
theorem nancy_water_intake_percentage 
  (daily_water_intake : ℝ) 
  (body_weight : ℝ) 
  (h1 : daily_water_intake = 54) 
  (h2 : body_weight = 90) : 
  (daily_water_intake / body_weight) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_nancy_water_intake_percentage_l3520_352031


namespace NUMINAMATH_CALUDE_new_energy_vehicle_sales_growth_rate_l3520_352096

theorem new_energy_vehicle_sales_growth_rate 
  (january_sales : ℕ) 
  (march_sales : ℕ) 
  (growth_rate : ℝ) : 
  january_sales = 25 → 
  march_sales = 36 → 
  (1 + growth_rate)^2 = march_sales / january_sales → 
  growth_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_new_energy_vehicle_sales_growth_rate_l3520_352096


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3520_352009

theorem polynomial_division_theorem (z : ℝ) :
  4 * z^4 - 6 * z^3 + 7 * z^2 - 17 * z + 3 =
  (5 * z + 4) * (z^3 - (26/5) * z^2 + (1/5) * z - 67/25) + 331/25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3520_352009


namespace NUMINAMATH_CALUDE_square_difference_l3520_352049

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3520_352049


namespace NUMINAMATH_CALUDE_fraction_sum_equals_thirteen_fourths_l3520_352004

theorem fraction_sum_equals_thirteen_fourths (a b : ℝ) (h1 : a = 3) (h2 : b = 1) :
  5 / (a + b) + 2 = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_thirteen_fourths_l3520_352004


namespace NUMINAMATH_CALUDE_ralphs_cards_l3520_352013

/-- 
Given that Ralph initially collected some cards and his father gave him additional cards,
this theorem proves the total number of cards Ralph has.
-/
theorem ralphs_cards (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : additional_cards = 8) : 
  initial_cards + additional_cards = 12 := by
  sorry

end NUMINAMATH_CALUDE_ralphs_cards_l3520_352013


namespace NUMINAMATH_CALUDE_rectangle_ribbon_length_l3520_352055

/-- The length of ribbon needed to form a rectangle -/
def ribbon_length (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: The length of ribbon needed to form a rectangle with length 20 feet and width 15 feet is 70 feet -/
theorem rectangle_ribbon_length : 
  ribbon_length 20 15 = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ribbon_length_l3520_352055


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3520_352037

/-- Represents the number of employees in each job category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  middle : ℕ
  general : ℕ

/-- Represents the number of sampled employees in each job category -/
structure SampledCount where
  senior : ℕ
  middle : ℕ
  general : ℕ

/-- Checks if the sampling is stratified correctly -/
def is_stratified_sampling (ec : EmployeeCount) (sample_size : ℕ) (sc : SampledCount) : Prop :=
  sc.senior * ec.total = sample_size * ec.senior ∧
  sc.middle * ec.total = sample_size * ec.middle ∧
  sc.general * ec.total = sample_size * ec.general

theorem stratified_sampling_theorem (ec : EmployeeCount) (sample_size : ℕ) :
  ec.total = ec.senior + ec.middle + ec.general →
  ∃ (sc : SampledCount), 
    sc.senior + sc.middle + sc.general = sample_size ∧
    is_stratified_sampling ec sample_size sc := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3520_352037


namespace NUMINAMATH_CALUDE_seven_consecutive_beautiful_numbers_odd_numbers_beautiful_divisible_by_four_beautiful_not_beautiful_mod_eight_six_l3520_352033

def is_beautiful (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + b^2 ∨ n = a^2 - b^2

theorem seven_consecutive_beautiful_numbers (k : ℤ) (hk : k ≥ 0) :
  ∃ n : ℤ, n ≥ k ∧ 
    (∀ i : ℤ, 0 ≤ i ∧ i < 7 → is_beautiful (8*n + i - 1)) ∧
    ¬(∀ i : ℤ, 0 ≤ i ∧ i < 8 → is_beautiful (8*n + i - 1)) :=
sorry

theorem odd_numbers_beautiful (n : ℤ) :
  n % 2 = 1 → is_beautiful n :=
sorry

theorem divisible_by_four_beautiful (n : ℤ) :
  n % 4 = 0 → is_beautiful n :=
sorry

theorem not_beautiful_mod_eight_six (n : ℤ) :
  n % 8 = 6 → ¬is_beautiful n :=
sorry

end NUMINAMATH_CALUDE_seven_consecutive_beautiful_numbers_odd_numbers_beautiful_divisible_by_four_beautiful_not_beautiful_mod_eight_six_l3520_352033


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l3520_352060

/-- The probability of drawing 3 marbles of the same color in the last draw -/
theorem cheryl_same_color_probability :
  let total_marbles : ℕ := 9
  let marbles_per_color : ℕ := 3
  let colors : ℕ := 3
  let draws : ℕ := 3
  let total_ways : ℕ := (total_marbles.choose draws) * ((total_marbles - draws).choose draws) * ((total_marbles - 2*draws).choose draws)
  let favorable_ways : ℕ := colors * ((total_marbles - 2*draws).choose draws)
  (favorable_ways : ℚ) / total_ways = 1 / 28 := by
sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l3520_352060


namespace NUMINAMATH_CALUDE_mencius_reading_problem_l3520_352042

theorem mencius_reading_problem (total_chars : ℕ) (days : ℕ) (first_day_chars : ℕ) : 
  total_chars = 34685 →
  days = 3 →
  first_day_chars + 2 * first_day_chars + 4 * first_day_chars = total_chars →
  first_day_chars = 4955 := by
sorry

end NUMINAMATH_CALUDE_mencius_reading_problem_l3520_352042


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3520_352046

def set_A : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}
def set_B : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3520_352046


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l3520_352078

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.5
  expected_potato_yield garden step_length yield_per_sqft = 1406.25 := by
  sorry


end NUMINAMATH_CALUDE_mr_green_potato_yield_l3520_352078


namespace NUMINAMATH_CALUDE_range_of_a_l3520_352054

def S : Set ℝ := {x : ℝ | (x - 2)^2 > 9}
def T (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 8}

theorem range_of_a : ∀ a : ℝ, (S ∪ T a = Set.univ) ↔ (-3 < a ∧ a < -1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3520_352054


namespace NUMINAMATH_CALUDE_student_fail_marks_l3520_352036

theorem student_fail_marks (pass_percentage : ℝ) (max_score : ℕ) (obtained_score : ℕ) : 
  pass_percentage = 36 / 100 → 
  max_score = 400 → 
  obtained_score = 130 → 
  ⌈pass_percentage * max_score⌉ - obtained_score = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_fail_marks_l3520_352036


namespace NUMINAMATH_CALUDE_yellow_beads_count_l3520_352017

theorem yellow_beads_count (blue_beads : ℕ) (total_parts : ℕ) (removed_per_part : ℕ) (final_per_part : ℕ) : 
  blue_beads = 23 →
  total_parts = 3 →
  removed_per_part = 10 →
  final_per_part = 6 →
  (∃ (yellow_beads : ℕ),
    let total_beads := blue_beads + yellow_beads
    let remaining_per_part := (total_beads / total_parts) - removed_per_part
    2 * remaining_per_part = final_per_part ∧
    yellow_beads = 16) :=
by
  sorry

#check yellow_beads_count

end NUMINAMATH_CALUDE_yellow_beads_count_l3520_352017


namespace NUMINAMATH_CALUDE_spheres_touching_triangle_and_other_spheres_l3520_352018

/-- Given a scalene triangle ABC with sides a, b, c and circumradius R,
    prove the existence of two spheres with radii r and ρ (ρ > r) that touch
    the plane of the triangle and three other spheres (with radii r_A, r_B, r_C)
    that touch the triangle at its vertices, such that 1/r - 1/ρ = 2√3/R. -/
theorem spheres_touching_triangle_and_other_spheres
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hscalene : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (R : ℝ) (hR : R > 0)
  (r_A r_B r_C : ℝ)
  (hr_A : r_A = b * c / (2 * a))
  (hr_B : r_B = c * a / (2 * b))
  (hr_C : r_C = a * b / (2 * c)) :
  ∃ (r ρ : ℝ), r > 0 ∧ ρ > r ∧ 1/r - 1/ρ = 2 * Real.sqrt 3 / R :=
sorry

end NUMINAMATH_CALUDE_spheres_touching_triangle_and_other_spheres_l3520_352018


namespace NUMINAMATH_CALUDE_sum_of_intercepts_l3520_352048

-- Define the parabola function
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define the x-intercept
def a : ℝ := parabola 0

-- Define the y-intercepts as roots of the equation 0 = 3y^2 - 9y + 4
def y_intercepts : Set ℝ := {y : ℝ | parabola y = 0}

-- Theorem statement
theorem sum_of_intercepts :
  ∃ (b c : ℝ), y_intercepts = {b, c} ∧ a + b + c = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_intercepts_l3520_352048


namespace NUMINAMATH_CALUDE_frog_escape_probability_l3520_352062

/-- Probability of frog escaping from pad N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- Total number of lily pads -/
def total_pads : ℕ := 15

/-- Starting pad for the frog -/
def start_pad : ℕ := 2

theorem frog_escape_probability :
  (∀ N, 0 < N → N < total_pads - 1 →
    P N = (N : ℝ) / total_pads * P (N - 1) + (1 - (N : ℝ) / total_pads) * P (N + 1)) →
  P 0 = 0 →
  P (total_pads - 1) = 1 →
  P start_pad = 163 / 377 :=
sorry

end NUMINAMATH_CALUDE_frog_escape_probability_l3520_352062


namespace NUMINAMATH_CALUDE_probability_of_asian_card_l3520_352016

-- Define the set of cards
inductive Card : Type
| China : Card
| USA : Card
| UK : Card
| SouthKorea : Card

-- Define a function to check if a card corresponds to an Asian country
def isAsian : Card → Bool
| Card.China => true
| Card.SouthKorea => true
| _ => false

-- Define the total number of cards
def totalCards : ℕ := 4

-- Define the number of Asian countries
def asianCards : ℕ := 2

-- Theorem statement
theorem probability_of_asian_card :
  (asianCards : ℚ) / totalCards = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_asian_card_l3520_352016


namespace NUMINAMATH_CALUDE_triangle_inequality_l3520_352059

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) : 
  a ≥ b ∧ b ≥ c ∧ c > 0 ∧ 
  0 < x ∧ x < π ∧ 0 < y ∧ y < π ∧ 0 < z ∧ z < π ∧ x + y + z = π → 
  b * c + c * a - a * b < b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ (1/2) * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3520_352059


namespace NUMINAMATH_CALUDE_root_cube_relation_l3520_352075

/-- The polynomial f(x) = x^3 + 2x^2 + 3x + 4 -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

/-- The polynomial h(x) = x^3 + bx^2 + cx + d -/
def h (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- Theorem stating the relationship between f and h, and the values of b, c, and d -/
theorem root_cube_relation (b c d : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧
    (∀ x : ℝ, h x b c d = 0 ↔ ∃ r : ℝ, f r = 0 ∧ x = r^3)) →
  b = 6 ∧ c = -8 ∧ d = 16 := by
sorry

end NUMINAMATH_CALUDE_root_cube_relation_l3520_352075


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l3520_352068

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 18) 
  (h3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 54 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l3520_352068


namespace NUMINAMATH_CALUDE_complex_subtraction_l3520_352056

theorem complex_subtraction : (6 : ℂ) + 2*I - (3 - 5*I) = 3 + 7*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3520_352056


namespace NUMINAMATH_CALUDE_three_propositions_imply_l3520_352090

theorem three_propositions_imply (p q r : Prop) : 
  (((p ∨ (¬q ∧ r)) → ((p → q) → r)) ∧
   ((¬p ∨ (¬q ∧ r)) → ((p → q) → r)) ∧
   ((p ∨ (¬q ∧ ¬r)) → ¬((p → q) → r)) ∧
   ((¬p ∨ (q ∧ r)) → ((p → q) → r))) := by
  sorry

end NUMINAMATH_CALUDE_three_propositions_imply_l3520_352090


namespace NUMINAMATH_CALUDE_sum_of_five_and_seven_l3520_352022

theorem sum_of_five_and_seven : 5 + 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_and_seven_l3520_352022


namespace NUMINAMATH_CALUDE_cos_540_degrees_l3520_352003

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_540_degrees_l3520_352003


namespace NUMINAMATH_CALUDE_fundraising_goal_l3520_352081

/-- Fundraising problem -/
theorem fundraising_goal (ken mary scott goal : ℕ) : 
  ken = 600 →
  mary = 5 * ken →
  mary = 3 * scott →
  ken + mary + scott = goal + 600 →
  goal = 4000 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_goal_l3520_352081


namespace NUMINAMATH_CALUDE_largest_value_l3520_352021

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (max (max 1 (2*a*b)) (a^2 + b^2)) a = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l3520_352021


namespace NUMINAMATH_CALUDE_parallel_transitivity_parallel_planes_imply_parallel_line_perpendicular_implies_parallel_perpendicular_planes_imply_parallel_line_l3520_352089

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- State the conditions
variable (h1 : ¬in_plane m α)
variable (h2 : ¬in_plane m β)
variable (h3 : ¬in_plane n α)
variable (h4 : ¬in_plane n β)

-- State the theorems to be proved
theorem parallel_transitivity 
  (h5 : parallel m n) (h6 : parallel_plane n α) : 
  parallel_plane m α := by sorry

theorem parallel_planes_imply_parallel_line 
  (h5 : parallel_plane m β) (h6 : parallel_planes α β) : 
  parallel_plane m α := by sorry

theorem perpendicular_implies_parallel 
  (h5 : perpendicular m n) (h6 : perpendicular_plane n α) : 
  parallel_plane m α := by sorry

theorem perpendicular_planes_imply_parallel_line 
  (h5 : perpendicular_plane m β) (h6 : perpendicular_planes α β) : 
  parallel_plane m α := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_parallel_planes_imply_parallel_line_perpendicular_implies_parallel_perpendicular_planes_imply_parallel_line_l3520_352089


namespace NUMINAMATH_CALUDE_inequality_holds_l3520_352012

theorem inequality_holds (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : b < 3) :
  b * 2^a < a * 2^b := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3520_352012


namespace NUMINAMATH_CALUDE_work_completion_l3520_352099

theorem work_completion (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 6 → absent_men = 4 → final_days = 12 →
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_l3520_352099


namespace NUMINAMATH_CALUDE_difference_of_squares_262_258_l3520_352095

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_262_258_l3520_352095


namespace NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l3520_352027

theorem fourth_power_nested_sqrt : 
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 2))))^4 = 
  2 + 2 * Real.sqrt (1 + Real.sqrt 2) + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_sqrt_l3520_352027


namespace NUMINAMATH_CALUDE_jeff_shelter_cats_l3520_352079

def cat_shelter_problem (initial_cats : ℕ) 
  (monday_added : ℕ) (tuesday_added : ℕ) 
  (wednesday_adopted wednesday_added : ℕ) 
  (thursday_adopted thursday_added : ℕ)
  (friday_adopted friday_added : ℕ) : Prop :=
  let after_monday := initial_cats + monday_added
  let after_tuesday := after_monday + tuesday_added
  let after_wednesday := after_tuesday + wednesday_added - wednesday_adopted
  let after_thursday := after_wednesday + thursday_added - thursday_adopted
  let final_count := after_thursday + friday_added - friday_adopted
  final_count = 30

theorem jeff_shelter_cats : 
  cat_shelter_problem 20 9 6 8 2 3 3 2 3 :=
by sorry

end NUMINAMATH_CALUDE_jeff_shelter_cats_l3520_352079


namespace NUMINAMATH_CALUDE_determinant_calculation_l3520_352085

def determinant (a b c d : Int) : Int :=
  a * d - b * c

theorem determinant_calculation : determinant 2 3 (-6) (-5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_determinant_calculation_l3520_352085


namespace NUMINAMATH_CALUDE_expected_red_pairs_in_standard_deck_l3520_352092

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardCount : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def redAdjacentProbability : ℚ := 25 / 51

/-- The expected number of pairs of adjacent red cards in a standard deck dealt in a circle -/
def expectedRedPairs : ℚ := redCardCount * redAdjacentProbability

theorem expected_red_pairs_in_standard_deck :
  expectedRedPairs = 650 / 51 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_in_standard_deck_l3520_352092


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l3520_352047

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_expectation_and_variance :
  ∃ (X : BinomialRV), X.n = 10 ∧ X.p = 0.6 ∧ expected_value X = 6 ∧ variance X = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l3520_352047


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3520_352032

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3520_352032


namespace NUMINAMATH_CALUDE_intersection_line_equation_l3520_352084

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x + 4)^2 + (y + 3)^2 = 8

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 3*y + 13 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l3520_352084


namespace NUMINAMATH_CALUDE_even_function_shift_l3520_352044

/-- Given a function f and a real number a, proves that if f(x+a) is even and a is in (0,π/2), then a = 5π/12 -/
theorem even_function_shift (f : ℝ → ℝ) (a : ℝ) : 
  (f = λ x => 3 * Real.sin (2 * x - π/3)) →
  (∀ x, f (x + a) = f (-x - a)) →
  (0 < a) →
  (a < π/2) →
  a = 5*π/12 := by
sorry

end NUMINAMATH_CALUDE_even_function_shift_l3520_352044


namespace NUMINAMATH_CALUDE_x1_value_l3520_352011

theorem x1_value (x₁ x₂ x₃ : ℝ) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h2 : (1-x₁)^3 + (x₁-x₂)^3 + (x₂-x₃)^3 + x₃^3 = 1/8) : 
  x₁ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l3520_352011


namespace NUMINAMATH_CALUDE_not_all_bisecting_diameters_perpendicular_l3520_352025

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord of a circle -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- A diameter of a circle -/
structure Diameter (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Predicate to check if a diameter bisects a chord -/
def bisects (d : Diameter c) (ch : Chord c) : Prop :=
  sorry

/-- Predicate to check if a diameter is perpendicular to a chord -/
def perpendicular (d : Diameter c) (ch : Chord c) : Prop :=
  sorry

/-- Theorem stating that it's not always true that a diameter bisecting a chord is perpendicular to it -/
theorem not_all_bisecting_diameters_perpendicular (c : Circle) :
  ∃ (d : Diameter c) (ch : Chord c), bisects d ch ∧ ¬perpendicular d ch :=
sorry

end NUMINAMATH_CALUDE_not_all_bisecting_diameters_perpendicular_l3520_352025


namespace NUMINAMATH_CALUDE_sum_squares_five_consecutive_not_perfect_square_l3520_352019

theorem sum_squares_five_consecutive_not_perfect_square (n : ℤ) :
  ¬∃ m : ℤ, (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = m^2 :=
sorry


end NUMINAMATH_CALUDE_sum_squares_five_consecutive_not_perfect_square_l3520_352019


namespace NUMINAMATH_CALUDE_sum_of_xyz_l3520_352077

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 120) :
  x + y + z = 22 * Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l3520_352077


namespace NUMINAMATH_CALUDE_total_toys_l3520_352052

/-- The number of toys each child has -/
structure ToyCount where
  jerry : ℕ
  gabriel : ℕ
  jaxon : ℕ

/-- The conditions of the problem -/
def toy_conditions (t : ToyCount) : Prop :=
  t.jerry = t.gabriel + 8 ∧
  t.gabriel = 2 * t.jaxon ∧
  t.jaxon = 15

/-- The theorem stating the total number of toys -/
theorem total_toys (t : ToyCount) (h : toy_conditions t) : 
  t.jerry + t.gabriel + t.jaxon = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_l3520_352052


namespace NUMINAMATH_CALUDE_distance_AB_is_336_l3520_352076

/-- The distance between two points A and B, given the conditions of the problem. -/
def distance_AB : ℝ :=
  let t_total := 3.5  -- Total time in hours
  let t_car3 := 3     -- Time for Car 3 to reach A
  let d_car1_left := 84  -- Distance left for Car 1 at 10:30 AM
  let d_car2_fraction := 3/8  -- Fraction of total distance Car 2 has traveled when Car 1 and 3 meet
  336

/-- The theorem stating that the distance between A and B is 336 km. -/
theorem distance_AB_is_336 :
  let d := distance_AB
  let v1 := d / 3.5 - 24  -- Speed of Car 1
  let v2 := d / 3.5       -- Speed of Car 2
  let v3 := d / 6         -- Speed of Car 3
  (v1 + v3 = 8/3 * v2) ∧  -- Condition when Car 1 and 3 meet
  (v3 * 3 = d / 2) ∧      -- Car 3 reaches A at 10:00 AM
  (v2 * 3.5 = d) ∧        -- Car 2 reaches A at 10:30 AM
  (d - v1 * 3.5 = 84) →   -- Car 1 is 84 km from B at 10:30 AM
  d = 336 := by
  sorry


end NUMINAMATH_CALUDE_distance_AB_is_336_l3520_352076


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_m_range_l3520_352010

/-- Represents an ellipse with the given equation -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1

/-- Indicates that the ellipse has foci on the y-axis -/
def has_foci_on_y_axis (e : Ellipse m) : Prop :=
  2 - m > |m| - 1 ∧ |m| - 1 > 0

/-- The range of m for which the ellipse has foci on the y-axis -/
def m_range (m : ℝ) : Prop :=
  m < -1 ∨ (1 < m ∧ m < 3/2)

/-- Theorem stating the range of m for an ellipse with foci on the y-axis -/
theorem ellipse_foci_y_axis_m_range (m : ℝ) :
  (∃ e : Ellipse m, has_foci_on_y_axis e) ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_m_range_l3520_352010


namespace NUMINAMATH_CALUDE_five_people_handshakes_l3520_352094

/-- The number of handshakes in a group of n people where each person
    shakes hands with every other person exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 5 people, where each person shakes hands with
    every other person exactly once, the total number of handshakes is 10 -/
theorem five_people_handshakes :
  handshakes 5 = 10 := by
  sorry

#eval handshakes 5  -- To verify the result

end NUMINAMATH_CALUDE_five_people_handshakes_l3520_352094


namespace NUMINAMATH_CALUDE_khalil_dogs_count_l3520_352051

/-- Represents the veterinary clinic problem -/
def veterinary_clinic_problem (dog_cost cat_cost : ℕ) (num_cats total_cost : ℕ) : Prop :=
  ∃ (num_dogs : ℕ), 
    dog_cost * num_dogs + cat_cost * num_cats = total_cost

/-- Proves that the number of dogs Khalil took to the clinic is 20 -/
theorem khalil_dogs_count : veterinary_clinic_problem 60 40 60 3600 → 
  ∃ (num_dogs : ℕ), num_dogs = 20 ∧ 60 * num_dogs + 40 * 60 = 3600 :=
by
  sorry


end NUMINAMATH_CALUDE_khalil_dogs_count_l3520_352051


namespace NUMINAMATH_CALUDE_min_time_for_all_flickers_l3520_352006

/-- The number of colored lights -/
def num_lights : ℕ := 5

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The time taken for one flicker (in seconds) -/
def flicker_time : ℕ := 5

/-- The interval time between flickers (in seconds) -/
def interval_time : ℕ := 5

/-- The total number of possible flickers -/
def total_flickers : ℕ := Nat.factorial num_lights

theorem min_time_for_all_flickers :
  (total_flickers * flicker_time) + ((total_flickers - 1) * interval_time) = 1195 := by
  sorry

end NUMINAMATH_CALUDE_min_time_for_all_flickers_l3520_352006


namespace NUMINAMATH_CALUDE_canal_construction_efficiency_l3520_352005

theorem canal_construction_efficiency (total_length : ℝ) (efficiency_multiplier : ℝ) (days_ahead : ℝ) 
  (original_daily_plan : ℝ) : 
  total_length = 3600 ∧ 
  efficiency_multiplier = 1.8 ∧ 
  days_ahead = 20 ∧
  (total_length / original_daily_plan - total_length / (efficiency_multiplier * original_daily_plan) = days_ahead) →
  original_daily_plan = 20 := by
sorry

end NUMINAMATH_CALUDE_canal_construction_efficiency_l3520_352005


namespace NUMINAMATH_CALUDE_fraction_with_buddies_l3520_352043

/-- Represents the number of students in each grade --/
structure StudentCounts where
  ninth : ℚ
  sixth : ℚ
  seventh : ℚ

/-- Represents the pairing ratios --/
structure PairingRatios where
  ninth : ℚ
  sixth : ℚ

/-- Represents the school mentoring program --/
structure MentoringProgram where
  counts : StudentCounts
  ratios : PairingRatios

/-- The main theorem about the fraction of students with buddies --/
theorem fraction_with_buddies (program : MentoringProgram) 
  (h1 : program.counts.ninth = 5 * program.counts.sixth / 4)
  (h2 : program.counts.seventh = 3 * program.counts.sixth / 4)
  (h3 : program.ratios.ninth = 1/4)
  (h4 : program.ratios.sixth = 1/3)
  (h5 : program.ratios.ninth * program.counts.ninth = program.ratios.sixth * program.counts.sixth) :
  (program.ratios.ninth * program.counts.ninth) / 
  (program.counts.ninth + program.counts.sixth + program.counts.seventh) = 5/48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_with_buddies_l3520_352043


namespace NUMINAMATH_CALUDE_kamal_chemistry_marks_l3520_352002

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  biology : ℕ
  chemistry : ℕ

/-- Calculates the average marks for a student -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.biology + marks.chemistry) / 5

/-- Theorem: Given Kamal's marks and average, his Chemistry marks must be 62 -/
theorem kamal_chemistry_marks :
  ∀ (kamal : StudentMarks),
    kamal.english = 66 →
    kamal.mathematics = 65 →
    kamal.physics = 77 →
    kamal.biology = 75 →
    average kamal = 69 →
    kamal.chemistry = 62 := by
  sorry

end NUMINAMATH_CALUDE_kamal_chemistry_marks_l3520_352002


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3520_352038

theorem quadratic_equation_roots :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ 
  (∀ x : ℝ, x^2 + x - 1 = 0 ↔ x = r1 ∨ x = r2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3520_352038


namespace NUMINAMATH_CALUDE_school_bus_capacity_l3520_352098

theorem school_bus_capacity 
  (columns_per_bus : ℕ) 
  (rows_per_bus : ℕ) 
  (number_of_buses : ℕ) 
  (h1 : columns_per_bus = 4) 
  (h2 : rows_per_bus = 10) 
  (h3 : number_of_buses = 6) : 
  columns_per_bus * rows_per_bus * number_of_buses = 240 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_capacity_l3520_352098


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3520_352093

/-- Given two people p and q, where 8 years ago p was half of q's age,
    and the total of their present ages is 28,
    prove that the ratio of their present ages is 3:4 -/
theorem age_ratio_problem (p q : ℕ) : 
  (p - 8 = (q - 8) / 2) → 
  (p + q = 28) → 
  (p : ℚ) / q = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3520_352093


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3520_352070

theorem unique_number_satisfying_conditions : ∃! n : ℕ,
  35 < n ∧ n < 70 ∧ (n - 3) % 6 = 0 ∧ (n - 1) % 8 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3520_352070


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l3520_352091

/-- Represents the possible weather outcomes for a day -/
inductive Weather
  | Sun
  | Rain2Inches
  | Rain8Inches

/-- The probability of each weather outcome -/
def weather_probability : Weather → ℝ
  | Weather.Sun => 0.35
  | Weather.Rain2Inches => 0.40
  | Weather.Rain8Inches => 0.25

/-- The amount of rainfall for each weather outcome -/
def rainfall_amount : Weather → ℝ
  | Weather.Sun => 0
  | Weather.Rain2Inches => 2
  | Weather.Rain8Inches => 8

/-- The number of days in the week -/
def days_in_week : ℕ := 7

/-- The expected rainfall for a single day -/
def expected_daily_rainfall : ℝ :=
  (weather_probability Weather.Sun * rainfall_amount Weather.Sun) +
  (weather_probability Weather.Rain2Inches * rainfall_amount Weather.Rain2Inches) +
  (weather_probability Weather.Rain8Inches * rainfall_amount Weather.Rain8Inches)

/-- Theorem: The expected total rainfall for the week is 19.6 inches -/
theorem expected_weekly_rainfall :
  (days_in_week : ℝ) * expected_daily_rainfall = 19.6 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l3520_352091


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l3520_352088

def original_expression (x : ℝ) : ℝ :=
  2 * (x - 6) + 5 * (10 - 3 * x^2 + 4 * x) - 7 * (3 * x^2 - 2 * x + 1)

theorem coefficient_of_x_squared :
  ∃ (a b c : ℝ), ∀ x, original_expression x = a * x^2 + b * x + c ∧ a = -36 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l3520_352088


namespace NUMINAMATH_CALUDE_max_package_volume_l3520_352020

/-- Represents the dimensions of a rectangular package. -/
structure PackageDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular package. -/
def volume (d : PackageDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the rope length required for a package. -/
def ropeLength (d : PackageDimensions) : ℝ :=
  2 * d.length + 4 * d.width + 6 * d.height

/-- The total available rope length in centimeters. -/
def totalRopeLength : ℝ := 360

/-- Theorem stating that the maximum volume of a package is 36000 cubic centimeters
    given the rope length and wrapping constraints. -/
theorem max_package_volume :
  ∃ (d : PackageDimensions),
    ropeLength d = totalRopeLength ∧
    ∀ (d' : PackageDimensions),
      ropeLength d' = totalRopeLength →
      volume d' ≤ volume d ∧
      volume d = 36000 :=
sorry

end NUMINAMATH_CALUDE_max_package_volume_l3520_352020


namespace NUMINAMATH_CALUDE_prob_heart_then_king_is_one_52_l3520_352000

def standard_deck : ℕ := 52
def hearts_in_deck : ℕ := 13
def kings_in_deck : ℕ := 4

def prob_heart_then_king : ℚ :=
  (hearts_in_deck / standard_deck) * (kings_in_deck / (standard_deck - 1))

theorem prob_heart_then_king_is_one_52 :
  prob_heart_then_king = 1 / 52 := by sorry

end NUMINAMATH_CALUDE_prob_heart_then_king_is_one_52_l3520_352000


namespace NUMINAMATH_CALUDE_jimmy_change_l3520_352015

def pen_cost : ℕ := 1
def notebook_cost : ℕ := 3
def folder_cost : ℕ := 5
def num_pens : ℕ := 3
def num_notebooks : ℕ := 4
def num_folders : ℕ := 2
def paid_amount : ℕ := 50

def total_cost : ℕ := 
  num_pens * pen_cost + num_notebooks * notebook_cost + num_folders * folder_cost

theorem jimmy_change : paid_amount - total_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_change_l3520_352015


namespace NUMINAMATH_CALUDE_f_of_3_equals_9_l3520_352007

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1) + 1

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_9_l3520_352007


namespace NUMINAMATH_CALUDE_pascal_row_20_sum_l3520_352065

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sum of the third, fourth, and fifth elements in Row 20 of Pascal's triangle -/
def pascalSum : ℕ := binomial 20 2 + binomial 20 3 + binomial 20 4

/-- Theorem stating that the sum of the third, fourth, and fifth elements 
    in Row 20 of Pascal's triangle is equal to 6175 -/
theorem pascal_row_20_sum : pascalSum = 6175 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_20_sum_l3520_352065


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3520_352008

theorem smallest_number_proof (x y z : ℝ) 
  (sum_xy : x + y = 23)
  (sum_xz : x + z = 31)
  (sum_yz : y + z = 11) :
  min x (min y z) = 21.5 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3520_352008


namespace NUMINAMATH_CALUDE_banana_cost_theorem_l3520_352029

def cost_of_fruit (apple_cost banana_cost orange_cost : ℚ)
                  (apple_count banana_count orange_count : ℕ)
                  (average_cost : ℚ) : Prop :=
  let total_count := apple_count + banana_count + orange_count
  let total_cost := apple_cost * apple_count + banana_cost * banana_count + orange_cost * orange_count
  total_cost = average_cost * total_count

theorem banana_cost_theorem :
  ∀ (banana_cost : ℚ),
    cost_of_fruit 2 banana_cost 3 12 4 4 2 →
    banana_cost = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_cost_theorem_l3520_352029


namespace NUMINAMATH_CALUDE_positive_integers_satisfying_conditions_l3520_352069

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_8 n ∧ sum_of_digits n = 7 ∧ product_of_digits n = 6

theorem positive_integers_satisfying_conditions :
  {n : ℕ | n > 0 ∧ satisfies_conditions n} = {1312, 3112} :=
by sorry

end NUMINAMATH_CALUDE_positive_integers_satisfying_conditions_l3520_352069


namespace NUMINAMATH_CALUDE_prob_two_red_in_three_draws_l3520_352024

def total_balls : ℕ := 8
def red_balls : ℕ := 3
def white_balls : ℕ := 5

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem prob_two_red_in_three_draws :
  (prob_white * prob_red * prob_red) + (prob_red * prob_white * prob_red) = 45 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_in_three_draws_l3520_352024


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l3520_352040

/-- Given six points on a circle, the probability of forming a convex quadrilateral
    by selecting any four chords from these points is 1/91. -/
theorem convex_quadrilateral_probability (n : ℕ) (h : n = 6) :
  (Nat.choose n 4 : ℚ) / (Nat.choose (Nat.choose n 2) 4) = 1 / 91 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l3520_352040


namespace NUMINAMATH_CALUDE_frank_meets_quota_l3520_352023

/-- Represents the sales data for Frank's car sales challenge -/
structure SalesData where
  quota : Nat
  days : Nat
  first3DaysSales : Nat
  next4DaysSales : Nat
  bonusCars : Nat
  remainingInventory : Nat
  oddDaySales : Nat
  evenDaySales : Nat

/-- Calculates the total sales and remaining inventory based on the given sales data -/
def calculateSales (data : SalesData) : (Nat × Nat) :=
  let initialSales := data.first3DaysSales * 3 + data.next4DaysSales * 4 + data.bonusCars
  let remainingDays := data.days - 7
  let oddDays := remainingDays / 2
  let evenDays := remainingDays - oddDays
  let potentialRemainingDaySales := data.oddDaySales * oddDays + data.evenDaySales * evenDays
  let actualRemainingDaySales := min potentialRemainingDaySales data.remainingInventory
  let totalSales := min (initialSales + actualRemainingDaySales) data.quota
  let remainingInventory := data.remainingInventory - (totalSales - initialSales)
  (totalSales, remainingInventory)

/-- Theorem stating that Frank will meet his quota and have 22 cars left in inventory -/
theorem frank_meets_quota (data : SalesData)
  (h1 : data.quota = 50)
  (h2 : data.days = 30)
  (h3 : data.first3DaysSales = 5)
  (h4 : data.next4DaysSales = 3)
  (h5 : data.bonusCars = 5)
  (h6 : data.remainingInventory = 40)
  (h7 : data.oddDaySales = 2)
  (h8 : data.evenDaySales = 3) :
  calculateSales data = (50, 22) := by
  sorry


end NUMINAMATH_CALUDE_frank_meets_quota_l3520_352023


namespace NUMINAMATH_CALUDE_mary_age_proof_l3520_352073

/-- Represents a person's age --/
structure Person where
  age : ℕ

/-- Represents the passage of time in years --/
def years_passed : ℕ := 4

/-- Suzy's current age --/
def suzy_now : Person := ⟨20⟩

/-- Mary's current age (to be proved) --/
def mary_now : Person := ⟨8⟩

/-- Theorem stating Mary's current age given the conditions --/
theorem mary_age_proof :
  (suzy_now.age + years_passed = 2 * (mary_now.age + years_passed)) →
  mary_now.age = 8 := by
  sorry


end NUMINAMATH_CALUDE_mary_age_proof_l3520_352073


namespace NUMINAMATH_CALUDE_gcd_459_357_l3520_352080

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3520_352080


namespace NUMINAMATH_CALUDE_max_sum_of_product_48_l3520_352014

theorem max_sum_of_product_48 :
  (∃ (x y : ℕ+), x * y = 48 ∧ x + y = 49) ∧
  (∀ (a b : ℕ+), a * b = 48 → a + b ≤ 49) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_product_48_l3520_352014


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3520_352053

theorem least_n_satisfying_inequality : 
  (∀ k : ℕ, k > 0 ∧ k < 4 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 15) ∧ 
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < 1 / 15) := by
  sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3520_352053


namespace NUMINAMATH_CALUDE_divisibility_of_2_power_n_minus_1_l3520_352072

theorem divisibility_of_2_power_n_minus_1 :
  ∃ (n : ℕ+), ∃ (k : ℕ), 2^n.val - 1 = 17 * k ∧
  ∀ (m : ℕ), 10 ≤ m → m ≤ 20 → m ≠ 17 → ¬∃ (l : ℕ+), ∃ (j : ℕ), 2^l.val - 1 = m * j :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_2_power_n_minus_1_l3520_352072


namespace NUMINAMATH_CALUDE_crabapple_sequences_l3520_352030

/-- The number of students in the class -/
def num_students : ℕ := 11

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 4

/-- The number of possible sequences of crabapple recipients in a week -/
def possible_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  possible_sequences = 14641 :=
sorry

end NUMINAMATH_CALUDE_crabapple_sequences_l3520_352030


namespace NUMINAMATH_CALUDE_justice_plants_l3520_352050

/-- The number of plants Justice wants in her home -/
def desired_plants (ferns palms succulents additional : ℕ) : ℕ :=
  ferns + palms + succulents + additional

/-- Theorem stating the total number of plants Justice wants -/
theorem justice_plants : 
  desired_plants 3 5 7 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_justice_plants_l3520_352050


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l3520_352039

/-- The Ferris wheel problem -/
theorem ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) 
  (h1 : total_people = 20) 
  (h2 : people_per_seat = 5) : 
  total_people / people_per_seat = 4 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l3520_352039


namespace NUMINAMATH_CALUDE_range_of_m_l3520_352086

/-- The function f(x) = x^2 - 3x + 4 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (7/4) 4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 7/4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 4) →
  m ∈ Set.Icc (3/2) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3520_352086


namespace NUMINAMATH_CALUDE_triangular_array_coins_l3520_352064

theorem triangular_array_coins (N : ℕ) : 
  (N * (N + 1)) / 2 = 2485 → N = 70 ∧ (N / 10 * (N % 10)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coins_l3520_352064


namespace NUMINAMATH_CALUDE_min_sum_arithmetic_sequence_l3520_352063

/-- An arithmetic sequence with a_n = 2n - 19 -/
def a (n : ℕ) : ℤ := 2 * n - 19

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n^2 - 18 * n

theorem min_sum_arithmetic_sequence :
  ∃ k : ℕ, k > 0 ∧ 
  (∀ n : ℕ, n > 0 → S n ≥ S k) ∧
  S k = -81 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_arithmetic_sequence_l3520_352063


namespace NUMINAMATH_CALUDE_square_root_expressions_l3520_352097

theorem square_root_expressions :
  (∃ x : ℝ, x^2 = 12) ∧ 
  (∃ y : ℝ, y^2 = 8) ∧ 
  (∃ z : ℝ, z^2 = 6) ∧ 
  (∃ w : ℝ, w^2 = 3) ∧ 
  (∃ v : ℝ, v^2 = 2) →
  (∃ a b : ℝ, a^2 = 12 ∧ b^2 = 8 ∧ 
    a + b * Real.sqrt 6 = 6 * Real.sqrt 3) ∧
  (∃ c d e : ℝ, c^2 = 12 ∧ d^2 = 3 ∧ e^2 = 2 ∧
    c + 1 / (Real.sqrt 3 - Real.sqrt 2) - Real.sqrt 6 * d = 3 * Real.sqrt 3 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_square_root_expressions_l3520_352097


namespace NUMINAMATH_CALUDE_clothing_selection_probability_l3520_352034

/-- The probability of selecting exactly one shirt, one pair of shorts, one pair of socks, and one hat
    when randomly choosing 4 articles of clothing from a drawer containing 6 shirts, 7 pairs of shorts,
    8 pairs of socks, and 3 hats. -/
theorem clothing_selection_probability :
  let num_shirts : ℕ := 6
  let num_shorts : ℕ := 7
  let num_socks : ℕ := 8
  let num_hats : ℕ := 3
  let total_items : ℕ := num_shirts + num_shorts + num_socks + num_hats
  let favorable_outcomes : ℕ := num_shirts * num_shorts * num_socks * num_hats
  let total_outcomes : ℕ := Nat.choose total_items 4
  (favorable_outcomes : ℚ) / total_outcomes = 144 / 1815 := by
  sorry


end NUMINAMATH_CALUDE_clothing_selection_probability_l3520_352034


namespace NUMINAMATH_CALUDE_dart_score_is_75_l3520_352071

/-- The final score of three dart throws -/
def final_score (bullseye : ℕ) (half_bullseye : ℕ) (miss : ℕ) : ℕ :=
  bullseye + half_bullseye + miss

/-- Theorem stating that the final score is 75 points -/
theorem dart_score_is_75 :
  ∃ (bullseye half_bullseye miss : ℕ),
    bullseye = 50 ∧
    half_bullseye = bullseye / 2 ∧
    miss = 0 ∧
    final_score bullseye half_bullseye miss = 75 := by
  sorry

end NUMINAMATH_CALUDE_dart_score_is_75_l3520_352071


namespace NUMINAMATH_CALUDE_polynomial_value_l3520_352074

theorem polynomial_value (a b : ℝ) (h : 2*a + 3*b - 5 = 0) : 
  6*a + 9*b - 12 = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l3520_352074


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3520_352087

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3520_352087


namespace NUMINAMATH_CALUDE_arthur_winning_strategy_l3520_352082

theorem arthur_winning_strategy (n : ℕ) (hn : n ≥ 2) :
  ∃ (A B : Finset ℕ), 
    A.card = n ∧ B.card = n ∧
    ∀ (k : ℕ) (hk : k ≤ 2*n - 2),
      ∃ (x : Fin k → ℕ),
        (∀ i : Fin k, ∃ (a : A) (b : B), x i = a * b) ∧
        (∀ i j l : Fin k, i ≠ j ∧ j ≠ l ∧ i ≠ l → ¬(x i ∣ x j * x l)) :=
by sorry

end NUMINAMATH_CALUDE_arthur_winning_strategy_l3520_352082


namespace NUMINAMATH_CALUDE_rachel_apples_remaining_l3520_352026

/-- The number of apples remaining on a tree after some are picked. -/
def applesRemaining (initial : ℕ) (picked : ℕ) : ℕ :=
  initial - picked

/-- Theorem: There are 3 apples remaining on Rachel's tree. -/
theorem rachel_apples_remaining :
  applesRemaining 7 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apples_remaining_l3520_352026
