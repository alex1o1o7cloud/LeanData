import Mathlib

namespace NUMINAMATH_CALUDE_first_marvelous_monday_l923_92321

/-- Represents a date in October --/
structure OctoberDate :=
  (day : ℕ)
  (is_monday : Bool)

/-- The number of days in October --/
def october_days : ℕ := 31

/-- The first day of school --/
def school_start : OctoberDate :=
  { day := 2, is_monday := true }

/-- A function to find the next Monday given a current date --/
def next_monday (d : OctoberDate) : OctoberDate :=
  { day := d.day + 7, is_monday := true }

/-- The definition of a Marvelous Monday --/
def is_marvelous_monday (d : OctoberDate) : Prop :=
  d.is_monday ∧ d.day ≤ october_days ∧ 
  (∀ m : OctoberDate, m.is_monday ∧ m.day > d.day → m.day > october_days)

/-- The theorem to prove --/
theorem first_marvelous_monday : 
  ∃ d : OctoberDate, d.day = 30 ∧ is_marvelous_monday d :=
sorry

end NUMINAMATH_CALUDE_first_marvelous_monday_l923_92321


namespace NUMINAMATH_CALUDE_parabola_through_point_l923_92358

-- Define a parabola
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax² + by² = c

-- Define the point (1, -2)
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem parabola_through_point :
  ∃ (p1 p2 : Parabola),
    (p1.a = 0 ∧ p1.b = 1 ∧ p1.c = 4 ∧ p1.a * point.1^2 + p1.b * point.2^2 = p1.c) ∨
    (p2.a = 1 ∧ p2.b = -1/2 ∧ p2.c = 0 ∧ p2.a * point.1^2 + p2.b * point.2 = p2.c) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_point_l923_92358


namespace NUMINAMATH_CALUDE_money_distribution_l923_92308

theorem money_distribution (A B C : ℝ) 
  (total : A + B + C = 450)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 350) :
  C = 100 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l923_92308


namespace NUMINAMATH_CALUDE_marbles_remainder_l923_92387

theorem marbles_remainder (r p : ℕ) 
  (h1 : r % 8 = 5) 
  (h2 : p % 8 = 7) 
  (h3 : (r + p) % 10 = 0) :
  (r + p) % 8 = 4 := by sorry

end NUMINAMATH_CALUDE_marbles_remainder_l923_92387


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l923_92316

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 + 3 * a 8 + a 15 = 120) :
  3 * a 9 - a 11 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l923_92316


namespace NUMINAMATH_CALUDE_adam_magnets_l923_92327

theorem adam_magnets (peter_magnets : ℕ) (adam_remaining : ℕ) (adam_initial : ℕ) : 
  peter_magnets = 24 →
  adam_remaining = peter_magnets / 2 →
  adam_remaining = adam_initial * 2 / 3 →
  adam_initial = 18 := by
sorry

end NUMINAMATH_CALUDE_adam_magnets_l923_92327


namespace NUMINAMATH_CALUDE_green_mm_probability_l923_92347

-- Define the initial state and actions
def initial_green : ℕ := 20
def initial_red : ℕ := 20
def green_eaten : ℕ := 12
def red_eaten : ℕ := initial_red / 2
def yellow_added : ℕ := 14

-- Calculate the final numbers
def final_green : ℕ := initial_green - green_eaten
def final_red : ℕ := initial_red - red_eaten
def final_yellow : ℕ := yellow_added

-- Calculate the total number of M&Ms after all actions
def total_mms : ℕ := final_green + final_red + final_yellow

-- Define the probability of selecting a green M&M
def prob_green : ℚ := final_green / total_mms

-- Theorem statement
theorem green_mm_probability : prob_green = 1/4 := by sorry

end NUMINAMATH_CALUDE_green_mm_probability_l923_92347


namespace NUMINAMATH_CALUDE_circle_intersection_condition_l923_92317

def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 9

def line_l (x y k : ℝ) : Prop := y = k * x + 3

def point_on_chord (M : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

def circle_intersects (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 ≤ radius^2

theorem circle_intersection_condition (k : ℝ) :
  (∀ x y : ℝ, circle_C x y → line_l x y k → 
    ∀ A B : ℝ × ℝ, circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 k ∧ line_l B.1 B.2 k →
      ∀ M : ℝ × ℝ, point_on_chord M A B →
        ∀ x y : ℝ, circle_C x y → circle_intersects M 2 x y) →
  k ≥ -3/4 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_condition_l923_92317


namespace NUMINAMATH_CALUDE_power_of_81_l923_92311

theorem power_of_81 : (81 : ℝ) ^ (5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l923_92311


namespace NUMINAMATH_CALUDE_series_sum_inequality_l923_92369

theorem series_sum_inequality (S : ℝ) (h : S = 2^(1/4)) : 
  ∃ n : ℕ, 2^n < S^2007 ∧ S^2007 < 2^(n+1) ∧ n = 501 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_inequality_l923_92369


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l923_92310

theorem employee_pay_calculation (total_pay : ℝ) (percentage : ℝ) (y : ℝ) :
  total_pay = 880 →
  percentage = 120 →
  total_pay = y + (percentage / 100) * y →
  y = 400 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l923_92310


namespace NUMINAMATH_CALUDE_y_derivative_l923_92323

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt (49 * x^2 + 1) * Real.arctan (7 * x) - Real.log (7 * x + Real.sqrt (49 * x^2 + 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = (7 * Real.arctan (7 * x)) / (2 * Real.sqrt (49 * x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l923_92323


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l923_92382

theorem factor_implies_d_value (d : ℚ) :
  (∀ x : ℚ, (x - 4) ∣ (d * x^4 + 11 * x^3 + 5 * d * x^2 - 28 * x + 72)) →
  d = -83/42 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l923_92382


namespace NUMINAMATH_CALUDE_line_circle_intersection_l923_92371

/-- A line with equation y = kx + 1 always intersects a circle with equation
    x^2 + y^2 - 2ax + a^2 - 2a - 4 = 0 if and only if -1 ≤ a ≤ 3 -/
theorem line_circle_intersection (k a : ℝ) :
  (∀ x y : ℝ, y = k * x + 1 → x^2 + y^2 - 2*a*x + a^2 - 2*a - 4 = 0 → 
    ∃ x' y' : ℝ, y' = k * x' + 1 ∧ x'^2 + y'^2 - 2*a*x' + a^2 - 2*a - 4 = 0) ↔ 
  -1 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l923_92371


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_l923_92312

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_x_for_cube : 
  (∀ y : ℕ, y > 0 ∧ y < 7350 → ¬ is_perfect_cube (1260 * y)) ∧ 
  is_perfect_cube (1260 * 7350) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_l923_92312


namespace NUMINAMATH_CALUDE_geometric_sequence_206th_term_l923_92342

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem geometric_sequence_206th_term :
  let a₁ := 4
  let a₂ := -12
  let r := a₂ / a₁
  geometric_sequence a₁ r 206 = -4 * 3^204 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_206th_term_l923_92342


namespace NUMINAMATH_CALUDE_homework_reading_assignment_l923_92380

theorem homework_reading_assignment (sam_pages pam_pages harrison_pages assigned_pages : ℕ) : 
  sam_pages = 100 →
  sam_pages = 2 * pam_pages →
  pam_pages = harrison_pages + 15 →
  harrison_pages = assigned_pages + 10 →
  assigned_pages = 25 := by
sorry

end NUMINAMATH_CALUDE_homework_reading_assignment_l923_92380


namespace NUMINAMATH_CALUDE_scientific_notation_32000000_l923_92309

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_32000000 :
  toScientificNotation 32000000 = ScientificNotation.mk 3.2 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_32000000_l923_92309


namespace NUMINAMATH_CALUDE_order_of_a_b_c_l923_92318

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_a_b_c : a > b ∧ a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_a_b_c_l923_92318


namespace NUMINAMATH_CALUDE_dinner_cost_l923_92365

/-- Proves that given a meal with a 10% sales tax, a 15% tip on the pre-tax amount,
    and a total cost of $27.50, the original cost of the meal before tax and tip is $22. -/
theorem dinner_cost (original_cost : ℝ) : 
  (original_cost * (1 + 0.1 + 0.15) = 27.5) → original_cost = 22 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_l923_92365


namespace NUMINAMATH_CALUDE_equal_boy_girl_division_theorem_l923_92393

/-- Represents a student arrangement as a list of integers, where 1 represents a boy and -1 represents a girl -/
def StudentArrangement := List Int

/-- Checks if a given arrangement can be divided into two parts with equal number of boys and girls -/
def canBeDivided (arrangement : StudentArrangement) : Bool :=
  sorry

/-- Counts the number of arrangements where division is impossible -/
def countImpossibleDivisions (n : Nat) : Nat :=
  sorry

/-- Counts the number of arrangements where exactly one division is possible -/
def countSingleDivisions (n : Nat) : Nat :=
  sorry

theorem equal_boy_girl_division_theorem (n : Nat) (h : n ≥ 2) :
  countSingleDivisions (2 * n) = 2 * countImpossibleDivisions (2 * n) :=
by
  sorry

end NUMINAMATH_CALUDE_equal_boy_girl_division_theorem_l923_92393


namespace NUMINAMATH_CALUDE_f_min_value_is_4_l923_92326

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem f_min_value_is_4 : ∀ x : ℝ, f x ≥ 4 ∧ ∃ x₀ : ℝ, f x₀ = 4 := by sorry

end NUMINAMATH_CALUDE_f_min_value_is_4_l923_92326


namespace NUMINAMATH_CALUDE_three_W_four_l923_92353

-- Define the operation W
def W (a b : ℤ) : ℤ := b + 5*a - 3*a^2

-- Theorem statement
theorem three_W_four : W 3 4 = -8 := by sorry

end NUMINAMATH_CALUDE_three_W_four_l923_92353


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l923_92399

/-- A function f(x) = ax + 3 has a zero point in the interval [-1, 2] -/
def has_zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ a * x + 3 = 0

/-- The condition a < -3 is sufficient but not necessary for the function to have a zero point in [-1, 2] -/
theorem sufficient_not_necessary :
  (∀ a : ℝ, a < -3 → has_zero_point a) ∧
  ¬(∀ a : ℝ, has_zero_point a → a < -3) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l923_92399


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l923_92329

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area :
  let h : ℝ := 8  -- height in inches
  let r : ℝ := 3  -- radius in inches
  let lateral_area : ℝ := 2 * π * r * h
  let base_area : ℝ := π * r^2
  let total_surface_area : ℝ := lateral_area + 2 * base_area
  total_surface_area = 66 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l923_92329


namespace NUMINAMATH_CALUDE_nancy_pears_l923_92379

/-- Given that Alyssa picked 42 pears and the total number of pears picked was 59,
    prove that Nancy picked 17 pears. -/
theorem nancy_pears (alyssa_pears total_pears : ℕ) 
  (h1 : alyssa_pears = 42)
  (h2 : total_pears = 59) :
  total_pears - alyssa_pears = 17 := by
  sorry

end NUMINAMATH_CALUDE_nancy_pears_l923_92379


namespace NUMINAMATH_CALUDE_pumpkin_weight_sum_total_pumpkin_weight_l923_92303

theorem pumpkin_weight_sum : ℝ → ℝ → ℝ
  | weight1, weight2 => weight1 + weight2

theorem total_pumpkin_weight :
  let weight1 : ℝ := 4
  let weight2 : ℝ := 8.7
  pumpkin_weight_sum weight1 weight2 = 12.7 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_weight_sum_total_pumpkin_weight_l923_92303


namespace NUMINAMATH_CALUDE_prob_exact_tails_l923_92383

def coin_flips : ℕ := 8
def p_tails : ℚ := 4/5
def p_heads : ℚ := 1/5
def exact_tails : ℕ := 3

theorem prob_exact_tails :
  (Nat.choose coin_flips exact_tails : ℚ) * p_tails ^ exact_tails * p_heads ^ (coin_flips - exact_tails) = 3584/390625 := by
  sorry

end NUMINAMATH_CALUDE_prob_exact_tails_l923_92383


namespace NUMINAMATH_CALUDE_waiter_tables_l923_92348

theorem waiter_tables (initial_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) : 
  initial_customers = 44 → left_customers = 12 → people_per_table = 8 → 
  (initial_customers - left_customers) / people_per_table = 4 := by
sorry

end NUMINAMATH_CALUDE_waiter_tables_l923_92348


namespace NUMINAMATH_CALUDE_max_value_is_nine_l923_92302

-- Define the set of possible values
def S : Finset ℕ := {1, 2, 4, 5}

-- Define the expression to be maximized
def f (x y z w : ℕ) : ℤ := x * y - y * z + z * w - w * x

-- Theorem statement
theorem max_value_is_nine :
  ∃ (x y z w : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ w ∈ S ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  f x y z w = 9 ∧
  ∀ (a b c d : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  f a b c d ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_nine_l923_92302


namespace NUMINAMATH_CALUDE_sum_exterior_angles_regular_hexagon_l923_92376

/-- A regular hexagon is a polygon with 6 sides of equal length and 6 angles of equal measure. -/
def RegularHexagon : Type := Unit

/-- The sum of the exterior angles of a polygon. -/
def SumExteriorAngles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a regular hexagon is 360°. -/
theorem sum_exterior_angles_regular_hexagon :
  SumExteriorAngles RegularHexagon = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_regular_hexagon_l923_92376


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l923_92346

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l923_92346


namespace NUMINAMATH_CALUDE_expression_equals_three_l923_92398

theorem expression_equals_three : (-1)^2 + Real.sqrt 16 - |(-3)| + 2 + (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l923_92398


namespace NUMINAMATH_CALUDE_average_height_problem_l923_92352

/-- Given the heights of four people with specific relationships, prove their average height. -/
theorem average_height_problem (reese daisy parker giselle : ℝ) : 
  reese = 60 →
  daisy = reese + 8 →
  parker = daisy - 4 →
  giselle = parker - 2 →
  (reese + daisy + parker + giselle) / 4 = 63.5 := by
  sorry

end NUMINAMATH_CALUDE_average_height_problem_l923_92352


namespace NUMINAMATH_CALUDE_bird_nest_babies_six_babies_in_nest_l923_92363

/-- The number of babies in a bird's nest given the worm requirements and available worms. -/
theorem bird_nest_babies (worms_per_baby_per_day : ℕ) (papa_worms : ℕ) (mama_worms : ℕ) 
  (stolen_worms : ℕ) (additional_worms_needed : ℕ) (days : ℕ) : ℕ :=
  let total_worms := papa_worms + mama_worms - stolen_worms + additional_worms_needed
  let worms_per_baby := worms_per_baby_per_day * days
  total_worms / worms_per_baby

/-- There are 6 babies in the nest given the specific conditions. -/
theorem six_babies_in_nest : 
  bird_nest_babies 3 9 13 2 34 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_babies_six_babies_in_nest_l923_92363


namespace NUMINAMATH_CALUDE_towel_bleaching_l923_92378

theorem towel_bleaching (original_length original_breadth : ℝ) 
  (h_positive : original_length > 0 ∧ original_breadth > 0) :
  let new_length := 0.7 * original_length
  let new_area := 0.42 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.6 * original_breadth ∧
    new_length * new_breadth = new_area :=
by sorry

end NUMINAMATH_CALUDE_towel_bleaching_l923_92378


namespace NUMINAMATH_CALUDE_colleen_paid_more_than_joy_l923_92304

/-- The amount of money Colleen paid more than Joy for pencils -/
def extra_cost (joy_pencils colleen_pencils price_per_pencil : ℕ) : ℕ :=
  (colleen_pencils - joy_pencils) * price_per_pencil

/-- Proof that Colleen paid $80 more than Joy for pencils -/
theorem colleen_paid_more_than_joy :
  extra_cost 30 50 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_colleen_paid_more_than_joy_l923_92304


namespace NUMINAMATH_CALUDE_sum_of_roots_l923_92390

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x + 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l923_92390


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l923_92397

theorem product_of_sum_and_difference (a b : ℝ) : (a + b) * (a - b) = (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l923_92397


namespace NUMINAMATH_CALUDE_popsicle_stick_cost_l923_92337

/-- Represents the cost of popsicle supplies in dollars -/
structure PopsicleSupplies where
  total_budget : ℚ
  mold_cost : ℚ
  juice_cost_per_bottle : ℚ
  popsicles_per_bottle : ℕ
  total_sticks : ℕ
  remaining_sticks : ℕ

/-- Calculates the cost of the pack of popsicle sticks -/
def stick_pack_cost (supplies : PopsicleSupplies) : ℚ :=
  supplies.total_budget - supplies.mold_cost - 
  (supplies.juice_cost_per_bottle * ((supplies.total_sticks - supplies.remaining_sticks) / supplies.popsicles_per_bottle))

/-- Theorem stating that the cost of the pack of popsicle sticks is $1 -/
theorem popsicle_stick_cost (supplies : PopsicleSupplies) 
  (h1 : supplies.total_budget = 10)
  (h2 : supplies.mold_cost = 3)
  (h3 : supplies.juice_cost_per_bottle = 2)
  (h4 : supplies.popsicles_per_bottle = 20)
  (h5 : supplies.total_sticks = 100)
  (h6 : supplies.remaining_sticks = 40) :
  stick_pack_cost supplies = 1 := by
  sorry

#eval stick_pack_cost { 
  total_budget := 10, 
  mold_cost := 3, 
  juice_cost_per_bottle := 2, 
  popsicles_per_bottle := 20, 
  total_sticks := 100, 
  remaining_sticks := 40 
}

end NUMINAMATH_CALUDE_popsicle_stick_cost_l923_92337


namespace NUMINAMATH_CALUDE_second_number_proof_l923_92338

theorem second_number_proof (a b : ℝ) (h1 : a = 50) (h2 : 0.6 * a - 0.3 * b = 27) : b = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l923_92338


namespace NUMINAMATH_CALUDE_no_prime_satisfies_condition_l923_92336

theorem no_prime_satisfies_condition : ¬∃ (P : ℕ), Prime P ∧ (100 : ℚ) * P = P + (1386 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_condition_l923_92336


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l923_92339

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept) + 0)

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l := Line.mk (-3) 7
  y_intercept l = (0, 21) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l923_92339


namespace NUMINAMATH_CALUDE_power_negative_cube_fourth_l923_92350

theorem power_negative_cube_fourth (a : ℝ) : (-a^3)^4 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_negative_cube_fourth_l923_92350


namespace NUMINAMATH_CALUDE_semicircle_radius_is_ten_l923_92340

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The height of the triangle, which is equal to the length of its legs -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base of the triangle is 20 units -/
  base_eq : base = 20
  /-- The semicircle's diameter is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base

/-- The radius of the inscribed semicircle is 10 units -/
theorem semicircle_radius_is_ten (t : IsoscelesTriangleWithSemicircle) : t.radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_is_ten_l923_92340


namespace NUMINAMATH_CALUDE_trigonometric_evaluations_l923_92356

open Real

theorem trigonometric_evaluations :
  (∃ (x : ℝ), x = sin (18 * π / 180) ∧ x = (Real.sqrt 5 - 1) / 4) ∧
  sin (18 * π / 180) * sin (54 * π / 180) = 1 / 4 ∧
  sin (36 * π / 180) * sin (72 * π / 180) = Real.sqrt 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_evaluations_l923_92356


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l923_92372

theorem regular_polygon_sides (n : ℕ) : n ≥ 3 → (n * (n - 3) / 2 + 2 * n = 36) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l923_92372


namespace NUMINAMATH_CALUDE_bank_deposit_calculation_l923_92313

theorem bank_deposit_calculation (initial_amount : ℝ) : 
  (0.20 * 0.25 * 0.30 * initial_amount = 750) → initial_amount = 50000 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_calculation_l923_92313


namespace NUMINAMATH_CALUDE_class_size_is_fifteen_l923_92315

/-- Given a class of students with the following properties:
  1. The average age of all students is 15 years
  2. The average age of 6 students is 14 years
  3. The average age of 8 students is 16 years
  4. The age of the 15th student is 13 years
  Prove that the total number of students in the class is 15 -/
theorem class_size_is_fifteen (N : ℕ) 
  (h1 : (N : ℚ) * 15 = (6 : ℚ) * 14 + (8 : ℚ) * 16 + 13)
  (h2 : N ≥ 15) : N = 15 := by
  sorry


end NUMINAMATH_CALUDE_class_size_is_fifteen_l923_92315


namespace NUMINAMATH_CALUDE_farm_animals_l923_92332

theorem farm_animals (total_legs : ℕ) (chicken_count : ℕ) : 
  total_legs = 38 → chicken_count = 5 → ∃ (sheep_count : ℕ), 
    chicken_count + sheep_count = 12 ∧ 
    2 * chicken_count + 4 * sheep_count = total_legs :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l923_92332


namespace NUMINAMATH_CALUDE_remaining_income_calculation_l923_92306

def remaining_income (food_percent : ℝ) (education_percent : ℝ) (rent_percent : ℝ) 
  (utilities_percent : ℝ) (transportation_percent : ℝ) (insurance_percent : ℝ) 
  (emergency_fund_percent : ℝ) : ℝ :=
  let initial_remaining := 1 - (food_percent + education_percent + transportation_percent)
  let rent_amount := rent_percent * initial_remaining
  let post_rent_remaining := initial_remaining - rent_amount
  let utilities_amount := utilities_percent * rent_amount
  let post_utilities_remaining := post_rent_remaining - utilities_amount
  let insurance_amount := insurance_percent * post_utilities_remaining
  let pre_emergency_remaining := post_utilities_remaining - insurance_amount
  let emergency_fund_amount := emergency_fund_percent * pre_emergency_remaining
  pre_emergency_remaining - emergency_fund_amount

theorem remaining_income_calculation :
  remaining_income 0.42 0.18 0.30 0.25 0.12 0.15 0.06 = 0.139825 := by
  sorry

#eval remaining_income 0.42 0.18 0.30 0.25 0.12 0.15 0.06

end NUMINAMATH_CALUDE_remaining_income_calculation_l923_92306


namespace NUMINAMATH_CALUDE_rectangle_width_equals_four_l923_92366

theorem rectangle_width_equals_four (square_side : ℝ) (rectangle_length : ℝ) (rectangle_width : ℝ) :
  square_side = 8 →
  rectangle_length = 16 →
  square_side * square_side = rectangle_length * rectangle_width →
  rectangle_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_four_l923_92366


namespace NUMINAMATH_CALUDE_matrix_from_eigenvectors_l923_92359

theorem matrix_from_eigenvectors (A : Matrix (Fin 2) (Fin 2) ℝ) :
  (A.mulVec (![1, -3]) = ![-1, 3]) →
  (A.mulVec (![1, 1]) = ![3, 3]) →
  A = !![2, 1; 3, 0] := by
sorry

end NUMINAMATH_CALUDE_matrix_from_eigenvectors_l923_92359


namespace NUMINAMATH_CALUDE_extra_flowers_l923_92396

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 36 → roses = 37 → used = 70 → tulips + roses - used = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_flowers_l923_92396


namespace NUMINAMATH_CALUDE_area_ABCGDE_value_l923_92374

/-- Shape ABCGDE formed by an equilateral triangle ABC and a square DEFG -/
structure ShapeABCGDE where
  /-- Side length of equilateral triangle ABC -/
  triangle_side : ℝ
  /-- Side length of square DEFG -/
  square_side : ℝ
  /-- Point D is at the midpoint of BC -/
  d_midpoint : Bool

/-- Calculate the area of shape ABCGDE -/
def area_ABCGDE (shape : ShapeABCGDE) : ℝ :=
  sorry

/-- Theorem: The area of shape ABCGDE is 27 + 9√3 -/
theorem area_ABCGDE_value :
  ∀ (shape : ShapeABCGDE),
  shape.triangle_side = 6 ∧ 
  shape.square_side = 6 ∧ 
  shape.d_midpoint = true →
  area_ABCGDE shape = 27 + 9 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_area_ABCGDE_value_l923_92374


namespace NUMINAMATH_CALUDE_class_average_score_l923_92300

theorem class_average_score (total_students : ℕ) (present_students : ℕ) (initial_average : ℚ) (makeup_score : ℚ) :
  total_students = 40 →
  present_students = 38 →
  initial_average = 92 →
  makeup_score = 100 →
  ((initial_average * present_students + makeup_score * (total_students - present_students)) / total_students) = 92.4 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l923_92300


namespace NUMINAMATH_CALUDE_apple_sorting_probability_l923_92319

def ratio_large_to_small : ℚ := 9 / 1
def prob_large_to_small : ℚ := 5 / 100
def prob_small_to_large : ℚ := 2 / 100

theorem apple_sorting_probability : 
  let total_apples := ratio_large_to_small + 1
  let prob_large := ratio_large_to_small / total_apples
  let prob_small := 1 / total_apples
  let prob_large_sorted_large := 1 - prob_large_to_small
  let prob_small_sorted_large := prob_small_to_large
  let prob_sorted_large := prob_large * prob_large_sorted_large + prob_small * prob_small_sorted_large
  let prob_large_and_sorted_large := prob_large * prob_large_sorted_large
  (prob_large_and_sorted_large / prob_sorted_large) = 855 / 857 :=
by sorry

end NUMINAMATH_CALUDE_apple_sorting_probability_l923_92319


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l923_92307

theorem product_of_sum_and_sum_of_cubes (c d : ℝ) 
  (h1 : c + d = 10) 
  (h2 : c^3 + d^3 = 370) : 
  c * d = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l923_92307


namespace NUMINAMATH_CALUDE_smallest_cube_ending_888_l923_92377

theorem smallest_cube_ending_888 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 888 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 888 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_888_l923_92377


namespace NUMINAMATH_CALUDE_apple_orchard_problem_l923_92351

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧ 
  o.pure_fuji + o.cross_pollinated = 153 ∧
  o.pure_fuji = (o.total * 3) / 4 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

theorem apple_orchard_problem :
  ∀ o : Orchard, orchard_conditions o → o.pure_gala = 27 := by
  sorry

end NUMINAMATH_CALUDE_apple_orchard_problem_l923_92351


namespace NUMINAMATH_CALUDE_minsu_age_proof_l923_92333

/-- Minsu's current age in years -/
def minsu_current_age : ℕ := 8

/-- Years in the future when Minsu's age will be four times his current age -/
def years_in_future : ℕ := 24

/-- Theorem stating that Minsu's current age is 8, given the condition -/
theorem minsu_age_proof :
  minsu_current_age = 8 ∧
  minsu_current_age + years_in_future = 4 * minsu_current_age :=
by sorry

end NUMINAMATH_CALUDE_minsu_age_proof_l923_92333


namespace NUMINAMATH_CALUDE_initial_children_on_bus_prove_initial_children_on_bus_l923_92388

theorem initial_children_on_bus : ℕ → Prop :=
  fun initial_children =>
    ∀ (added_children total_children : ℕ),
      added_children = 7 →
      total_children = 25 →
      initial_children + added_children = total_children →
      initial_children = 18

-- Proof
theorem prove_initial_children_on_bus :
  ∃ (initial_children : ℕ), initial_children_on_bus initial_children :=
by
  sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_prove_initial_children_on_bus_l923_92388


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l923_92362

/-- Given three lines that pass through the same point, prove the value of k -/
theorem intersection_of_three_lines (t : ℝ) (h_t : t = 6) :
  ∃ (x y : ℝ), (x + t * y + 8 = 0 ∧ 5 * x - t * y + 4 = 0 ∧ 3 * x - 5 * y + 1 = 0) →
  ∀ k : ℝ, (x + t * y + 8 = 0 ∧ 5 * x - t * y + 4 = 0 ∧ 3 * x - k * y + 1 = 0) →
  k = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l923_92362


namespace NUMINAMATH_CALUDE_number_problem_l923_92345

theorem number_problem (x : ℕ) (h1 : x + 3927 = 13800) : x = 9873 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l923_92345


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l923_92394

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | 3 * x + 5 ≥ -1 ∧ 3 - x > (1/2) * x}
  S = {x | -2 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l923_92394


namespace NUMINAMATH_CALUDE_helen_raisin_cookie_difference_l923_92392

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := 300

/-- The number of raisin cookies Helen baked today -/
def raisin_cookies_today : ℕ := 280

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_cookie_difference : ℕ := raisin_cookies_yesterday - raisin_cookies_today

theorem helen_raisin_cookie_difference : raisin_cookie_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_helen_raisin_cookie_difference_l923_92392


namespace NUMINAMATH_CALUDE_rectangle_cannot_fit_in_square_l923_92361

theorem rectangle_cannot_fit_in_square :
  ∀ (rect_length rect_width square_side : ℝ),
  rect_length > 0 ∧ rect_width > 0 ∧ square_side > 0 →
  rect_length * rect_width = 90 →
  rect_length / rect_width = 5 / 3 →
  square_side * square_side = 100 →
  rect_length > square_side :=
by sorry

end NUMINAMATH_CALUDE_rectangle_cannot_fit_in_square_l923_92361


namespace NUMINAMATH_CALUDE_canoe_weight_with_dog_l923_92367

/-- Calculates the total weight carried by Penny's canoe with her dog -/
theorem canoe_weight_with_dog (normal_capacity : ℕ) (person_weight : ℝ) : 
  normal_capacity = 6 →
  person_weight = 140 →
  (2 : ℝ) / 3 * normal_capacity * person_weight + 1 / 4 * person_weight = 595 :=
by
  sorry

#check canoe_weight_with_dog

end NUMINAMATH_CALUDE_canoe_weight_with_dog_l923_92367


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l923_92373

theorem dress_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  (d * (100 - x) / 100) * 0.7 = 0.455 * d ↔ x = 35 := by
sorry

end NUMINAMATH_CALUDE_dress_discount_percentage_l923_92373


namespace NUMINAMATH_CALUDE_tadpole_survival_fraction_l923_92391

/-- Represents the frog pond ecosystem --/
structure FrogPond where
  num_frogs : ℕ
  num_tadpoles : ℕ
  max_capacity : ℕ
  frogs_to_relocate : ℕ

/-- Calculates the fraction of tadpoles that will survive to maturity as frogs --/
def survival_fraction (pond : FrogPond) : ℚ :=
  let surviving_tadpoles := pond.max_capacity - pond.num_frogs
  ↑surviving_tadpoles / ↑pond.num_tadpoles

/-- Theorem stating the fraction of tadpoles that will survive to maturity as frogs --/
theorem tadpole_survival_fraction (pond : FrogPond) 
  (h1 : pond.num_frogs = 5)
  (h2 : pond.num_tadpoles = 3 * pond.num_frogs)
  (h3 : pond.max_capacity = 8)
  (h4 : pond.frogs_to_relocate = 7) :
  survival_fraction pond = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tadpole_survival_fraction_l923_92391


namespace NUMINAMATH_CALUDE_provisions_last_20_days_l923_92325

/-- Calculates the number of days provisions will last after reinforcement arrives -/
def days_after_reinforcement (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) (reinforcement : ℕ) : ℕ :=
  let initial_man_days := initial_men * initial_days
  let used_man_days := initial_men * days_before_reinforcement
  let remaining_man_days := initial_man_days - used_man_days
  let total_men_after_reinforcement := initial_men + reinforcement
  remaining_man_days / total_men_after_reinforcement

/-- Theorem stating that given the problem conditions, the provisions will last 20 more days after reinforcement -/
theorem provisions_last_20_days :
  days_after_reinforcement 2000 65 15 3000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_provisions_last_20_days_l923_92325


namespace NUMINAMATH_CALUDE_remaining_area_in_square_l923_92305

theorem remaining_area_in_square : 
  let large_square_side : ℝ := 3.5
  let small_square_side : ℝ := 2
  let rectangle_length : ℝ := 2
  let rectangle_width : ℝ := 1.5
  let triangle_leg : ℝ := 1
  let large_square_area := large_square_side ^ 2
  let small_square_area := small_square_side ^ 2
  let rectangle_area := rectangle_length * rectangle_width
  let triangle_area := 0.5 * triangle_leg * triangle_leg
  let occupied_area := small_square_area + rectangle_area + triangle_area
  large_square_area - occupied_area = 4.75 := by
sorry

end NUMINAMATH_CALUDE_remaining_area_in_square_l923_92305


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l923_92389

/-- The symmetric point of M(2, -3, 1) with respect to the origin is (-2, 3, -1). -/
theorem symmetric_point_wrt_origin :
  let M : ℝ × ℝ × ℝ := (2, -3, 1)
  let symmetric_point : ℝ × ℝ × ℝ := (-2, 3, -1)
  ∀ (x y z : ℝ), (x, y, z) = M → (-x, -y, -z) = symmetric_point :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l923_92389


namespace NUMINAMATH_CALUDE_stapler_equation_l923_92357

theorem stapler_equation (sheets : ℕ) (time_first time_combined : ℝ) (time_second : ℝ) :
  sheets > 0 ∧ time_first > 0 ∧ time_combined > 0 ∧ time_second > 0 →
  (sheets / time_first + sheets / time_second = sheets / time_combined) ↔
  (1 / time_first + 1 / time_second = 1 / time_combined) :=
by sorry

end NUMINAMATH_CALUDE_stapler_equation_l923_92357


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l923_92360

theorem min_sum_absolute_values (x : ℝ) : 
  |x + 3| + |x + 6| + |x + 7| + 2 ≥ 8 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 6| + |y + 7| + 2 = 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l923_92360


namespace NUMINAMATH_CALUDE_second_day_charge_l923_92301

theorem second_day_charge (day1_charge : ℝ) (day3_charge : ℝ) (attendance_ratio : Fin 3 → ℝ) (average_charge : ℝ) :
  day1_charge = 15 →
  day3_charge = 2.5 →
  attendance_ratio 0 = 2 →
  attendance_ratio 1 = 5 →
  attendance_ratio 2 = 13 →
  average_charge = 5 →
  ∃ day2_charge : ℝ,
    day2_charge = 7.5 ∧
    average_charge * (attendance_ratio 0 + attendance_ratio 1 + attendance_ratio 2) =
      day1_charge * attendance_ratio 0 + day2_charge * attendance_ratio 1 + day3_charge * attendance_ratio 2 :=
by
  sorry


end NUMINAMATH_CALUDE_second_day_charge_l923_92301


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l923_92355

theorem smallest_number_divisible (h : ℕ) : 
  (∀ n : ℕ, n < 259 → ¬(((n + 5) % 8 = 0) ∧ ((n + 5) % 11 = 0) ∧ ((n + 5) % 24 = 0))) ∧
  ((259 + 5) % 8 = 0) ∧ ((259 + 5) % 11 = 0) ∧ ((259 + 5) % 24 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l923_92355


namespace NUMINAMATH_CALUDE_plum_problem_l923_92349

theorem plum_problem (x : ℕ) : 
  (4 * x / 5 : ℚ) = (5 * x / 6 : ℚ) - 1 → 2 * x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_plum_problem_l923_92349


namespace NUMINAMATH_CALUDE_eight_b_value_l923_92320

theorem eight_b_value (a b : ℝ) (h1 : 6 * a + 3 * b = 3) (h2 : a = 2 * b + 3) : 8 * b = -8 := by
  sorry

end NUMINAMATH_CALUDE_eight_b_value_l923_92320


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l923_92343

theorem regular_polygon_sides (n : ℕ) (interior_angle exterior_angle : ℝ) : 
  n > 2 →
  interior_angle / exterior_angle = 5 →
  interior_angle + exterior_angle = 180 →
  n * exterior_angle = 360 →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l923_92343


namespace NUMINAMATH_CALUDE_remainder_31_pow_31_plus_31_mod_32_l923_92330

theorem remainder_31_pow_31_plus_31_mod_32 : (31^31 + 31) % 32 = 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_31_pow_31_plus_31_mod_32_l923_92330


namespace NUMINAMATH_CALUDE_additional_steps_day3_l923_92322

def day1_steps : ℕ := 200 + 300

def day2_steps : ℕ := 2 * day1_steps

def total_steps : ℕ := 1600

theorem additional_steps_day3 : 
  total_steps - (day1_steps + day2_steps) = 100 := by sorry

end NUMINAMATH_CALUDE_additional_steps_day3_l923_92322


namespace NUMINAMATH_CALUDE_product_of_brackets_l923_92334

def bracket_a (a : ℕ) : ℕ := a^2 + 3

def bracket_b (b : ℕ) : ℕ := 2*b - 4

theorem product_of_brackets (p q : ℕ) (h1 : p = 7) (h2 : q = 10) :
  bracket_a p * bracket_b q = 832 := by
  sorry

end NUMINAMATH_CALUDE_product_of_brackets_l923_92334


namespace NUMINAMATH_CALUDE_mady_balls_theorem_l923_92364

def to_nonary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_nonary_aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_nonary_aux (m / 9) ((m % 9) :: acc)
    to_nonary_aux n []

def sum_of_digits (digits : List ℕ) : ℕ :=
  digits.sum

theorem mady_balls_theorem (step : ℕ) (h : step = 2500) :
  sum_of_digits (to_nonary step) = 20 :=
sorry

end NUMINAMATH_CALUDE_mady_balls_theorem_l923_92364


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l923_92375

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the foci
def is_focus (F : ℝ × ℝ) : Prop := 
  ∃ (c : ℝ), F.1^2 + F.2^2 = c^2 ∧ c^2 = 4 - 3

-- Define a point on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the property of points on the ellipse
def ellipse_property (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  on_ellipse A ∧ on_ellipse B ∧ is_focus F1 ∧ is_focus F2 →
  dist A F1 + dist A F2 = dist B F1 + dist B F2

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :
  ellipse_property A B F1 F2 →
  (∃ (t : ℝ), A = F1 + t • (B - F1)) →
  dist A B + dist A F2 + dist B F2 = 8 := by
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l923_92375


namespace NUMINAMATH_CALUDE_max_value_on_curve_l923_92314

theorem max_value_on_curve (b : ℝ) (h : b > 0) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 + 2*y
  let S : Set (ℝ × ℝ) := {(x, y) | x^2/4 + y^2/b^2 = 1}
  (∃ (M : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) ∧
  (0 < b ∧ b ≤ 4 → ∀ (M : ℝ), (∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) → b^2/4 + 4 ≤ M) ∧
  (b > 4 → ∀ (M : ℝ), (∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) → 2*b ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l923_92314


namespace NUMINAMATH_CALUDE_symmetric_line_proof_l923_92370

/-- The fixed point M through which all lines ax+y+3a-1=0 pass -/
def M : ℝ × ℝ := (-3, 1)

/-- The original line -/
def original_line (x y : ℝ) : Prop := 2*x + 3*y - 6 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 2*x + 3*y + 12 = 0

/-- The family of lines passing through M -/
def family_line (a x y : ℝ) : Prop := a*x + y + 3*a - 1 = 0

theorem symmetric_line_proof :
  ∀ (a : ℝ), family_line a M.1 M.2 →
  ∀ (x y : ℝ), symmetric_line x y ↔ 
    (x - M.1 = M.1 - x' ∧ y - M.2 = M.2 - y' ∧ original_line x' y') :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_proof_l923_92370


namespace NUMINAMATH_CALUDE_percent_relation_l923_92354

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 
  2 * b / a = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_percent_relation_l923_92354


namespace NUMINAMATH_CALUDE_trigonometric_identity_l923_92381

theorem trigonometric_identity (α : ℝ) :
  -Real.cos (5 * α) * Real.cos (4 * α) - Real.cos (4 * α) * Real.cos (3 * α) + 2 * (Real.cos (2 * α))^2 * Real.cos α
  = 2 * Real.cos α * Real.sin (2 * α) * Real.sin (6 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l923_92381


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l923_92386

def sum_of_range (a b : ℕ) : ℕ := 
  ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  let x := sum_of_range 20 30
  let y := count_even_in_range 20 30
  x + y = 281 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l923_92386


namespace NUMINAMATH_CALUDE_apple_bags_count_l923_92335

/-- The number of bags of apples loaded onto a lorry -/
def number_of_bags (empty_weight loaded_weight bag_weight : ℕ) : ℕ :=
  (loaded_weight - empty_weight) / bag_weight

/-- Theorem stating that the number of bags of apples is 20 -/
theorem apple_bags_count : 
  let empty_weight : ℕ := 500
  let loaded_weight : ℕ := 1700
  let bag_weight : ℕ := 60
  number_of_bags empty_weight loaded_weight bag_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_apple_bags_count_l923_92335


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l923_92385

theorem polynomial_division_remainder (x : ℂ) : 
  (x^75 + x^60 + x^45 + x^30 + x^15 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l923_92385


namespace NUMINAMATH_CALUDE_child_ticket_cost_l923_92328

/-- Calculates the cost of a child movie ticket given the following information:
  * Adult ticket cost is $9.50
  * Total group size is 7
  * Number of adults is 3
  * Total amount paid is $54.50
-/
theorem child_ticket_cost : 
  let adult_cost : ℝ := 9.50
  let total_group : ℕ := 7
  let num_adults : ℕ := 3
  let total_paid : ℝ := 54.50
  let num_children : ℕ := total_group - num_adults
  let child_cost : ℝ := (total_paid - (adult_cost * num_adults)) / num_children
  child_cost = 6.50 := by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l923_92328


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l923_92331

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_A : (center.1 - 1)^2 + (center.2 - 2)^2 = radius^2
  passes_through_B : (center.1 - 1)^2 + (center.2 - 10)^2 = radius^2
  tangent_to_line : |center.1 - 2*center.2 - 1| / Real.sqrt 5 = radius

/-- The theorem stating that a circle passing through (1, 2) and (1, 10) and 
    tangent to x - 2y - 1 = 0 must have one of two specific equations -/
theorem tangent_circle_equation : 
  ∀ c : TangentCircle, 
    ((c.center.1 = 3 ∧ c.center.2 = 6 ∧ c.radius^2 = 20) ∨
     (c.center.1 = -7 ∧ c.center.2 = 6 ∧ c.radius^2 = 80)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l923_92331


namespace NUMINAMATH_CALUDE_complex_magnitude_constraint_l923_92368

theorem complex_magnitude_constraint (a : ℝ) :
  let z : ℂ := 1 + a * I
  (Complex.abs z < 2) → (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_constraint_l923_92368


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l923_92384

theorem compare_negative_fractions : -4/5 > -5/6 := by sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l923_92384


namespace NUMINAMATH_CALUDE_triangle_relations_l923_92324

/-- Given a triangle with area S, inradius r, exradii r_a, r_b, r_c, 
    side lengths a, b, c, circumradius R, and semiperimeter p -/
theorem triangle_relations (S r r_a r_b r_c a b c R : ℝ) 
  (h_positive : S > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_semiperimeter : ∃ p, p = (a + b + c) / 2) :
  (1 / r^3 - 1 / r_a^3 - 1 / r_b^3 - 1 / r_c^3 = 12 * R / S^2) ∧
  (a * (b + c) = (r + r_a) * (4 * R + r - r_a)) ∧
  (a * (b - c) = (r_b - r_c) * (4 * R - r_b - r_c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_relations_l923_92324


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l923_92341

theorem quadratic_equation_solution :
  let f (x : ℝ) := x^2 - 5*x + 1
  ∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧
               x₂ = (5 - Real.sqrt 21) / 2 ∧
               f x₁ = 0 ∧ f x₂ = 0 ∧
               ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l923_92341


namespace NUMINAMATH_CALUDE_power_sum_equals_four_sqrt_three_over_three_logarithm_equation_solution_l923_92344

-- Define a as log_4(3)
noncomputable def a : ℝ := Real.log 3 / Real.log 4

-- Theorem 1: 2^a + 2^(-a) = (4 * sqrt(3)) / 3
theorem power_sum_equals_four_sqrt_three_over_three :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := by sorry

-- Theorem 2: The solution to log_2(9^(x-1) - 5) = log_2(3^(x-1) - 2) + 2 is x = 2
theorem logarithm_equation_solution :
  ∃! x : ℝ, (x > 1 ∧ Real.log (9^(x-1) - 5) / Real.log 2 = Real.log (3^(x-1) - 2) / Real.log 2 + 2) ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_power_sum_equals_four_sqrt_three_over_three_logarithm_equation_solution_l923_92344


namespace NUMINAMATH_CALUDE_smallest_valid_perfect_square_l923_92395

def is_valid (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 10, n % (k + 2) = k + 1

theorem smallest_valid_perfect_square : 
  ∃ n : ℕ, n = 2782559 ∧ 
    is_valid n ∧ 
    ∃ m : ℕ, n = m^2 ∧ 
    ∀ k : ℕ, k < n → ¬(is_valid k ∧ ∃ m : ℕ, k = m^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_perfect_square_l923_92395
