import Mathlib

namespace tangent_parallel_condition_extreme_values_max_k_no_intersection_l3554_355415

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + a / Real.exp x

theorem tangent_parallel_condition (a : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, f a x = f a 1 + k * (x - 1)) ↔ a = Real.exp 1 := by sorry

theorem extreme_values (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > 0 → ∃ x : ℝ, x = Real.log a ∧ ∀ y : ℝ, y ≠ x → f a x < f a y) := by sorry

theorem max_k_no_intersection :
  ∃ k : ℝ, k = 1 ∧
    (∀ k' : ℝ, (∀ x : ℝ, f 1 x ≠ k' * x - 1) → k' ≤ k) := by sorry

end tangent_parallel_condition_extreme_values_max_k_no_intersection_l3554_355415


namespace shampoo_duration_l3554_355443

theorem shampoo_duration (rose_shampoo : Rat) (jasmine_shampoo : Rat) (daily_usage : Rat) : 
  rose_shampoo = 1/3 → jasmine_shampoo = 1/4 → daily_usage = 1/12 →
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end shampoo_duration_l3554_355443


namespace decimal_to_base_conversion_l3554_355437

theorem decimal_to_base_conversion (x : ℕ) : 
  (4 * x + 7 = 71) → x = 16 := by
  sorry

end decimal_to_base_conversion_l3554_355437


namespace eighth_term_is_negative_22_l3554_355487

/-- An arithmetic sequence with a2 = -4 and common difference -3 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  -4 + (n - 2) * (-3)

/-- Theorem: The 8th term of the arithmetic sequence is -22 -/
theorem eighth_term_is_negative_22 : arithmetic_sequence 8 = -22 := by
  sorry

end eighth_term_is_negative_22_l3554_355487


namespace function_extrema_sum_l3554_355419

def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

theorem function_extrema_sum (m : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-3 : ℝ) 0, f m x ≤ max) ∧ 
    (∃ x ∈ Set.Icc (-3 : ℝ) 0, f m x = max) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 0, f m x ≥ min) ∧ 
    (∃ x ∈ Set.Icc (-3 : ℝ) 0, f m x = min) ∧
    (max + min = -1)) →
  m = 7.5 := by
sorry

end function_extrema_sum_l3554_355419


namespace shaded_area_theorem_l3554_355476

theorem shaded_area_theorem (square_side : ℝ) (total_beans : ℕ) (shaded_beans : ℕ) :
  square_side = 2 →
  total_beans = 200 →
  shaded_beans = 120 →
  (shaded_beans : ℝ) / (total_beans : ℝ) * (square_side ^ 2) = 12 / 5 :=
by sorry

end shaded_area_theorem_l3554_355476


namespace unique_solution_sin_system_l3554_355424

theorem unique_solution_sin_system (a b c d : Real) 
  (h_sum : a + b + c + d = Real.pi) :
  ∃! (x y z w : Real),
    x = Real.sin (a + b) ∧
    y = Real.sin (b + c) ∧
    z = Real.sin (c + d) ∧
    w = Real.sin (d + a) :=
by sorry

end unique_solution_sin_system_l3554_355424


namespace equation_one_solution_equation_two_no_solution_l3554_355496

/-- The first equation has a unique solution x = 4 -/
theorem equation_one_solution :
  ∃! x : ℝ, (5 / (x + 1) = 1 / (x - 3)) ∧ (x + 1 ≠ 0) ∧ (x - 3 ≠ 0) :=
sorry

/-- The second equation has no solution -/
theorem equation_two_no_solution :
  ¬∃ x : ℝ, ((3 - x) / (x - 4) = 1 / (4 - x) - 2) ∧ (x - 4 ≠ 0) ∧ (4 - x ≠ 0) :=
sorry

end equation_one_solution_equation_two_no_solution_l3554_355496


namespace complex_sum_real_parts_l3554_355474

theorem complex_sum_real_parts (a b : ℝ) (h : Complex.mk a b = Complex.I * (1 - Complex.I)) : a + b = 2 := by
  sorry

end complex_sum_real_parts_l3554_355474


namespace largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l3554_355470

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 13*n + 40 < 0 → n ≤ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  7^2 - 13*7 + 40 < 0 :=
by
  sorry

theorem eight_does_not_satisfy_inequality :
  8^2 - 13*8 + 40 ≥ 0 :=
by
  sorry

end largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l3554_355470


namespace quadratic_real_root_condition_l3554_355442

-- Define the quadratic equation
def quadratic (b x : ℝ) : ℝ := x^2 + b*x + 25

-- Define the condition for real roots
def has_real_root (b : ℝ) : Prop := ∃ x : ℝ, quadratic b x = 0

-- Theorem statement
theorem quadratic_real_root_condition (b : ℝ) :
  has_real_root b ↔ b ≤ -10 ∨ b ≥ 10 :=
by sorry

end quadratic_real_root_condition_l3554_355442


namespace flag_arrangement_remainder_l3554_355472

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  13 * Nat.choose 14 10 - 2 * Nat.choose 13 10

/-- The theorem stating the remainder when M is divided by 1000 -/
theorem flag_arrangement_remainder :
  M % 1000 = 441 := by
  sorry

end flag_arrangement_remainder_l3554_355472


namespace lemonade_stand_problem_l3554_355445

/-- Represents the lemonade stand problem -/
theorem lemonade_stand_problem 
  (total_days : ℕ) 
  (hot_days : ℕ) 
  (cups_per_day : ℕ) 
  (total_profit : ℚ) 
  (cost_per_cup : ℚ) 
  (hot_day_price_increase : ℚ) :
  total_days = 10 →
  hot_days = 4 →
  cups_per_day = 32 →
  total_profit = 350 →
  cost_per_cup = 3/4 →
  hot_day_price_increase = 1/4 →
  ∃ (regular_price : ℚ),
    regular_price > 0 ∧
    (total_days - hot_days) * cups_per_day * regular_price +
    hot_days * cups_per_day * (regular_price * (1 + hot_day_price_increase)) -
    total_days * cups_per_day * cost_per_cup = total_profit ∧
    regular_price * (1 + hot_day_price_increase) = 15/8 :=
by sorry

end lemonade_stand_problem_l3554_355445


namespace probability_one_blue_one_white_l3554_355473

def total_marbles : ℕ := 8
def blue_marbles : ℕ := 3
def white_marbles : ℕ := 5
def marbles_left : ℕ := 2

def favorable_outcomes : ℕ := blue_marbles * white_marbles
def total_outcomes : ℕ := Nat.choose total_marbles marbles_left

theorem probability_one_blue_one_white :
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 28 := by sorry

end probability_one_blue_one_white_l3554_355473


namespace digit_sum_puzzle_l3554_355455

theorem digit_sum_puzzle (c d : ℕ) : 
  c < 10 → d < 10 → 
  (40 + c) * (10 * d + 5) = 215 →
  (40 + c) * 5 = 20 →
  (40 + c) * d * 10 = 180 →
  c + d = 5 := by
sorry

end digit_sum_puzzle_l3554_355455


namespace only_jia_can_formulate_quadratic_l3554_355497

/-- Represents a person in the problem -/
inductive Person
  | jia
  | yi
  | bing
  | ding

/-- Checks if a number is congruent to 1 modulo 3 -/
def is_cong_1_mod_3 (n : ℤ) : Prop := n % 3 = 1

/-- Checks if a number is congruent to 2 modulo 3 -/
def is_cong_2_mod_3 (n : ℤ) : Prop := n % 3 = 2

/-- Represents the conditions for each person's quadratic equation -/
def satisfies_conditions (person : Person) (p q x₁ x₂ : ℤ) : Prop :=
  match person with
  | Person.jia => is_cong_1_mod_3 p ∧ is_cong_1_mod_3 q ∧ is_cong_1_mod_3 x₁ ∧ is_cong_1_mod_3 x₂
  | Person.yi => is_cong_2_mod_3 p ∧ is_cong_2_mod_3 q ∧ is_cong_2_mod_3 x₁ ∧ is_cong_2_mod_3 x₂
  | Person.bing => is_cong_1_mod_3 p ∧ is_cong_1_mod_3 q ∧ is_cong_2_mod_3 x₁ ∧ is_cong_2_mod_3 x₂
  | Person.ding => is_cong_2_mod_3 p ∧ is_cong_2_mod_3 q ∧ is_cong_1_mod_3 x₁ ∧ is_cong_1_mod_3 x₂

/-- Represents a valid quadratic equation -/
def is_valid_quadratic (p q x₁ x₂ : ℤ) : Prop :=
  x₁ + x₂ = -p ∧ x₁ * x₂ = q

/-- The main theorem stating that only Jia can formulate a valid quadratic equation -/
theorem only_jia_can_formulate_quadratic :
  ∀ person : Person,
    (∃ p q x₁ x₂ : ℤ, satisfies_conditions person p q x₁ x₂ ∧ is_valid_quadratic p q x₁ x₂) ↔
    person = Person.jia :=
sorry


end only_jia_can_formulate_quadratic_l3554_355497


namespace hours_to_seconds_l3554_355434

-- Define the conversion factors
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Define the problem
def hours : ℚ := 3.5

-- Theorem to prove
theorem hours_to_seconds : 
  (hours * minutes_per_hour * seconds_per_minute : ℚ) = 12600 := by
  sorry

end hours_to_seconds_l3554_355434


namespace mode_of_team_ages_l3554_355479

def team_ages : List Nat := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_team_ages :
  mode team_ages = 18 := by
  sorry

end mode_of_team_ages_l3554_355479


namespace g_expression_l3554_355493

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g using the given condition
def g (x : ℝ) : ℝ := f (x - 3)

-- Theorem statement
theorem g_expression : ∀ x : ℝ, g x = 2 * x - 3 := by
  sorry

end g_expression_l3554_355493


namespace tangent_curve_sum_l3554_355461

/-- A curve y = -2x^2 + bx + c is tangent to the line y = x - 3 at the point (2, -1). -/
theorem tangent_curve_sum (b c : ℝ) : 
  (∀ x, -2 * x^2 + b * x + c = x - 3 → x = 2) → 
  (-2 * 2^2 + b * 2 + c = -1) →
  ((-4 * 2 + b) = 1) →
  b + c = -2 := by sorry

end tangent_curve_sum_l3554_355461


namespace new_parabola_equation_l3554_355491

/-- The original quadratic function -/
def original_function (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The vertex of the original parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- The axis of symmetry of the original parabola -/
def axis_of_symmetry : ℝ := 1

/-- The line that intersects the new parabola -/
def intersecting_line (m : ℝ) (x : ℝ) : ℝ := m * x - 2

/-- The point of intersection between the new parabola and the line -/
def intersection_point : ℝ × ℝ := (2, 4)

/-- The equation of the new parabola -/
def new_parabola (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 4

theorem new_parabola_equation :
  ∃ (t : ℝ), ∀ (x : ℝ),
    new_parabola x = -3 * (x - axis_of_symmetry)^2 + vertex.2 + t ∧
    new_parabola intersection_point.1 = intersection_point.2 :=
by sorry

end new_parabola_equation_l3554_355491


namespace consecutive_integers_product_l3554_355407

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 10 := by
sorry

end consecutive_integers_product_l3554_355407


namespace non_negativity_and_extrema_l3554_355447

theorem non_negativity_and_extrema :
  (∀ x y : ℝ, (x - 1)^2 ≥ 0 ∧ x^2 + 1 > 0 ∧ |3*x + 2*y| ≥ 0) ∧
  (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 ∧ (∃ x₀ : ℝ, x₀^2 - 2*x₀ + 1 = 0)) ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 + x*y →
    (x - 3*y)^2 + 4*(y + x)*(x - y) ≤ 6) :=
by sorry

end non_negativity_and_extrema_l3554_355447


namespace abs_diff_inequality_l3554_355440

theorem abs_diff_inequality (x : ℝ) : |x + 3| - |x - 1| > 0 ↔ x > -1 := by
  sorry

end abs_diff_inequality_l3554_355440


namespace expand_and_simplify_l3554_355412

theorem expand_and_simplify (x : ℝ) : (5*x - 3)*(2*x + 4) = 10*x^2 + 14*x - 12 := by
  sorry

end expand_and_simplify_l3554_355412


namespace min_omega_value_l3554_355463

/-- Given a function f(x) = sin(ω(x - π/4)) where ω > 0, if f(3π/4) = 0, then the minimum value of ω is 2. -/
theorem min_omega_value (ω : ℝ) (h₁ : ω > 0) :
  (fun x => Real.sin (ω * (x - π / 4))) (3 * π / 4) = 0 → ω ≥ 2 ∧ ∀ ω' > 0, (fun x => Real.sin (ω' * (x - π / 4))) (3 * π / 4) = 0 → ω' ≥ ω :=
by sorry

end min_omega_value_l3554_355463


namespace equation_solution_l3554_355409

theorem equation_solution :
  ∃! x : ℝ, (x^2 + 4*x + 5) / (x + 3) = x + 7 ∧ x ≠ -3 :=
by
  use (-8/3)
  sorry

end equation_solution_l3554_355409


namespace max_value_constraint_l3554_355414

theorem max_value_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 2) :
  x + y^3 + z^2 ≤ 8 ∧ ∃ (a b c : ℝ), a + b^3 + c^2 = 8 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 2 := by
sorry

end max_value_constraint_l3554_355414


namespace quadratic_symmetry_l3554_355482

/-- A quadratic function with axis of symmetry at x = 9.5 and p(1) = 2 -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = p a b c (19 - x)) →  -- symmetry about x = 9.5
  p a b c 1 = 2 →                            -- p(1) = 2
  p a b c 18 = 2 :=                          -- p(18) = 2
by
  sorry

#check quadratic_symmetry

end quadratic_symmetry_l3554_355482


namespace jiangsu_income_scientific_notation_l3554_355485

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Rounds a real number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original amount in yuan -/
def originalAmount : ℝ := 26341

/-- The number of significant figures required -/
def requiredSigFigs : ℕ := 3

theorem jiangsu_income_scientific_notation :
  toScientificNotation originalAmount requiredSigFigs =
    ScientificNotation.mk 2.63 4 := by sorry

end jiangsu_income_scientific_notation_l3554_355485


namespace average_speed_calculation_l3554_355429

theorem average_speed_calculation (distance1 : ℝ) (distance2 : ℝ) (time1 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 100) 
  (h2 : distance2 = 60) 
  (h3 : time1 = 1) 
  (h4 : time2 = 1) : 
  (distance1 + distance2) / (time1 + time2) = 80 := by
  sorry

end average_speed_calculation_l3554_355429


namespace johns_umbrella_cost_l3554_355425

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * cost_per_umbrella

/-- Proof that John's total cost for umbrellas is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end johns_umbrella_cost_l3554_355425


namespace enclosed_area_calculation_l3554_355486

/-- The area enclosed by a curve consisting of 9 congruent circular arcs, 
    each of length π/2, whose centers are at the vertices of a regular hexagon 
    with side length 3. -/
def enclosed_area (num_arcs : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) : ℝ :=
  sorry

/-- Theorem stating the enclosed area for the specific problem -/
theorem enclosed_area_calculation : 
  enclosed_area 9 (π/2) 3 = (27 * Real.sqrt 3) / 2 + 9 * π / 8 :=
sorry

end enclosed_area_calculation_l3554_355486


namespace inequality_solutions_range_l3554_355484

theorem inequality_solutions_range (a : ℝ) : 
  (∃! (x₁ x₂ : ℕ), x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
   (∀ (x : ℕ), x > 0 → (3 * ↑x + a ≤ 2 ↔ (x = x₁ ∨ x = x₂)))) →
  -7 < a ∧ a ≤ -4 :=
by sorry

end inequality_solutions_range_l3554_355484


namespace exactly_three_correct_implies_B_false_l3554_355492

-- Define the function f over ℝ
variable (f : ℝ → ℝ)

-- Define the properties stated by each student
def property_A (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

def property_B (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

def property_C (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + (1 - x)) = f x

def property_D (f : ℝ → ℝ) : Prop :=
  ∃ x, f x < f 0

-- Theorem stating that if exactly three properties are true, then B must be false
theorem exactly_three_correct_implies_B_false (f : ℝ → ℝ) :
  ((property_A f ∧ property_C f ∧ property_D f) ∨
   (property_A f ∧ property_B f ∧ property_C f) ∨
   (property_A f ∧ property_B f ∧ property_D f) ∨
   (property_B f ∧ property_C f ∧ property_D f)) →
  ¬ property_B f :=
sorry

end exactly_three_correct_implies_B_false_l3554_355492


namespace inequality_system_solution_l3554_355488

theorem inequality_system_solution :
  ∀ x : ℝ, (x - 7 < 5 * (x - 1) ∧ 4/3 * x + 3 ≥ 1 - 2/3 * x) ↔ x > -1/2 := by
  sorry

end inequality_system_solution_l3554_355488


namespace hydrangea_price_l3554_355475

def pansy_price : ℝ := 2.50
def petunia_price : ℝ := 1.00
def num_pansies : ℕ := 5
def num_petunias : ℕ := 5
def discount_rate : ℝ := 0.10
def paid_amount : ℝ := 50.00
def change_received : ℝ := 23.00

theorem hydrangea_price (hydrangea_cost : ℝ) : hydrangea_cost = 12.50 := by
  sorry

#check hydrangea_price

end hydrangea_price_l3554_355475


namespace total_pizzas_eaten_l3554_355465

/-- The number of pizzas eaten by class A -/
def pizzas_class_a : ℕ := 8

/-- The number of pizzas eaten by class B -/
def pizzas_class_b : ℕ := 7

/-- The total number of pizzas eaten by both classes -/
def total_pizzas : ℕ := pizzas_class_a + pizzas_class_b

theorem total_pizzas_eaten : total_pizzas = 15 := by
  sorry

end total_pizzas_eaten_l3554_355465


namespace one_fifth_of_five_times_nine_l3554_355421

theorem one_fifth_of_five_times_nine : (1 / 5 : ℚ) * (5 * 9) = 9 := by
  sorry

end one_fifth_of_five_times_nine_l3554_355421


namespace boat_speed_ratio_l3554_355438

theorem boat_speed_ratio (boat_speed : ℝ) (current_speed : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : current_speed = 5)
  (h3 : boat_speed > current_speed) :
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let avg_speed := 2 / (1 / downstream_speed + 1 / upstream_speed)
  avg_speed / boat_speed = 8 / 9 := by
sorry

end boat_speed_ratio_l3554_355438


namespace log_product_plus_exp_equals_seven_l3554_355457

theorem log_product_plus_exp_equals_seven :
  Real.log 9 / Real.log 2 * (Real.log 4 / Real.log 3) + (2 : ℝ) ^ (Real.log 3 / Real.log 2) = 7 := by
  sorry

end log_product_plus_exp_equals_seven_l3554_355457


namespace dollar_three_neg_two_l3554_355498

-- Define the operation $
def dollar (a b : ℤ) : ℤ := a * (b - 1) + a * b

-- Theorem statement
theorem dollar_three_neg_two : dollar 3 (-2) = -15 := by
  sorry

end dollar_three_neg_two_l3554_355498


namespace adjacent_sides_equal_not_imply_rhombus_l3554_355403

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, dist (q.vertices i) (q.vertices j) = dist (q.vertices 0) (q.vertices 1)

-- Define adjacent sides
def adjacent_sides_equal (q : Quadrilateral) : Prop :=
  ∃ i : Fin 4, dist (q.vertices i) (q.vertices (i + 1)) = dist (q.vertices ((i + 1) % 4)) (q.vertices ((i + 2) % 4))

-- Theorem to prove
theorem adjacent_sides_equal_not_imply_rhombus :
  ¬(∀ q : Quadrilateral, adjacent_sides_equal q → is_rhombus q) :=
sorry

end adjacent_sides_equal_not_imply_rhombus_l3554_355403


namespace collinear_vectors_perpendicular_vectors_l3554_355467

-- Problem 1
def point_A : ℝ × ℝ := (5, 4)
def point_C : ℝ × ℝ := (12, -2)

def vector_AB (k : ℝ) : ℝ × ℝ := (k - 5, 6)
def vector_BC (k : ℝ) : ℝ × ℝ := (12 - k, -12)

def are_collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem collinear_vectors :
  are_collinear (vector_AB (-2)) (vector_BC (-2)) := by sorry

-- Problem 2
def vector_OA : ℝ × ℝ := (-7, 6)
def vector_OC : ℝ × ℝ := (5, 7)

def vector_AB' (k : ℝ) : ℝ × ℝ := (10, k - 6)
def vector_BC' (k : ℝ) : ℝ × ℝ := (2, 7 - k)

def are_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem perpendicular_vectors :
  (are_perpendicular (vector_AB' 2) (vector_BC' 2)) ∧
  (are_perpendicular (vector_AB' 11) (vector_BC' 11)) := by sorry

end collinear_vectors_perpendicular_vectors_l3554_355467


namespace intersection_of_A_and_B_l3554_355494

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l3554_355494


namespace function_properties_l3554_355433

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (f a 1 + f a (-1) = 5/2 → f a 2 + f a (-2) = 17/4) ∧
  (∃ (max min : ℝ), (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ max ∧ f a x ≥ min) ∧
    max - min = 8/3 → a = 3 ∨ a = 1/3) :=
sorry

end function_properties_l3554_355433


namespace product_ab_l3554_355435

theorem product_ab (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a * b = 2256 := by
  sorry

end product_ab_l3554_355435


namespace krystiana_monthly_earnings_l3554_355451

/-- Represents the apartment building owned by Krystiana -/
structure ApartmentBuilding where
  firstFloorRate : ℕ
  secondFloorRate : ℕ
  thirdFloorRate : ℕ
  roomsPerFloor : ℕ
  occupiedThirdFloorRooms : ℕ

/-- Calculates the monthly earnings from Krystiana's apartment building -/
def calculateMonthlyEarnings (building : ApartmentBuilding) : ℕ :=
  building.firstFloorRate * building.roomsPerFloor +
  building.secondFloorRate * building.roomsPerFloor +
  building.thirdFloorRate * building.occupiedThirdFloorRooms

/-- Theorem stating that Krystiana's monthly earnings are $165 -/
theorem krystiana_monthly_earnings :
  ∀ (building : ApartmentBuilding),
    building.firstFloorRate = 15 →
    building.secondFloorRate = 20 →
    building.thirdFloorRate = 2 * building.firstFloorRate →
    building.roomsPerFloor = 3 →
    building.occupiedThirdFloorRooms = 2 →
    calculateMonthlyEarnings building = 165 := by
  sorry

#eval calculateMonthlyEarnings {
  firstFloorRate := 15,
  secondFloorRate := 20,
  thirdFloorRate := 30,
  roomsPerFloor := 3,
  occupiedThirdFloorRooms := 2
}

end krystiana_monthly_earnings_l3554_355451


namespace generalized_schur_inequality_l3554_355458

theorem generalized_schur_inequality (a b c t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^t * (a - b) * (a - c) + b^t * (b - c) * (b - a) + c^t * (c - a) * (c - b) ≥ 0 := by
  sorry

end generalized_schur_inequality_l3554_355458


namespace chessboard_tiling_l3554_355430

/-- A chessboard of size a × b can be tiled with n-ominoes of size 1 × n if and only if n divides a or n divides b. -/
theorem chessboard_tiling (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (∃ (tiling : ℕ → ℕ → Fin n), ∀ (i j : ℕ), i < a ∧ j < b → 
    (∀ (k : ℕ), k < n → tiling i (j + k) = tiling i j + k ∨ tiling (i + k) j = tiling i j + k)) ↔ 
  (n ∣ a ∨ n ∣ b) :=
by sorry

end chessboard_tiling_l3554_355430


namespace hyperbola_equation_l3554_355481

/-- The equation of a hyperbola with foci at (-3, 0) and (3, 0), and |MA| - |MB| = 4 -/
theorem hyperbola_equation (x y : ℝ) :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (3, 0)
  let M : ℝ × ℝ := (x, y)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist M A - dist M B = 4) → (x > 0) →
  (x^2 / 4 - y^2 / 5 = 1) := by
sorry

end hyperbola_equation_l3554_355481


namespace quadratic_congruences_equivalence_l3554_355401

theorem quadratic_congruences_equivalence (p : Nat) (h : Nat.Prime p) :
  (∃ x, (x^2 + x + 3) % p = 0 → ∃ y, (y^2 + y + 25) % p = 0) ∧
  (¬∃ x, (x^2 + x + 3) % p = 0 → ¬∃ y, (y^2 + y + 25) % p = 0) ∧
  (∃ y, (y^2 + y + 25) % p = 0 → ∃ x, (x^2 + x + 3) % p = 0) ∧
  (¬∃ y, (y^2 + y + 25) % p = 0 → ¬∃ x, (x^2 + x + 3) % p = 0) :=
by sorry


end quadratic_congruences_equivalence_l3554_355401


namespace arithmetic_sum_10_l3554_355413

/-- An arithmetic sequence with given first and second terms -/
def arithmetic_sequence (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * (a₂ - a₁)

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + arithmetic_sequence a₁ a₂ n) / 2

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence
    with a₁ = 1 and a₂ = -3 is -170 -/
theorem arithmetic_sum_10 :
  arithmetic_sum 1 (-3) 10 = -170 := by
  sorry

end arithmetic_sum_10_l3554_355413


namespace area_T_prime_l3554_355489

/-- A transformation matrix -/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -2, 5]

/-- The area of the original region T -/
def area_T : ℝ := 9

/-- The theorem stating the area of the transformed region T' -/
theorem area_T_prime : 
  let det_A := Matrix.det A
  area_T * det_A = 207 := by sorry

end area_T_prime_l3554_355489


namespace find_w_l3554_355400

/-- Given that ( √ 1.21 ) / ( √ 0.81 ) + ( √ 1.44 ) / ( √ w ) = 2.9365079365079367, prove that w = 0.49 -/
theorem find_w (w : ℝ) (h : Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt w = 2.9365079365079367) : 
  w = 0.49 := by
  sorry

end find_w_l3554_355400


namespace max_value_when_m_3_solution_f_geq_0_l3554_355499

-- Define the function f(x, m)
def f (x m : ℝ) : ℝ := |x - m| - 2 * |x - 1|

-- Theorem for the maximum value when m = 3
theorem max_value_when_m_3 :
  ∃ (max : ℝ), max = 2 ∧ ∀ x, f x 3 ≤ max :=
sorry

-- Theorem for the solution of f(x) ≥ 0
theorem solution_f_geq_0 (m : ℝ) :
  (m > 1 → ∀ x, f x m ≥ 0 ↔ 2 - m ≤ x ∧ x ≤ (2 + m) / 3) ∧
  (m = 1 → ∀ x, f x m ≥ 0 ↔ x = 1) ∧
  (m < 1 → ∀ x, f x m ≥ 0 ↔ (2 + m) / 3 ≤ x ∧ x ≤ 2 - m) :=
sorry

end max_value_when_m_3_solution_f_geq_0_l3554_355499


namespace clock_hands_angle_l3554_355411

theorem clock_hands_angle (n : ℕ) : 0 < n ∧ n < 720 → (∃ k : ℤ, |11 * n / 2 % 360 - 360 * k| = 1) ↔ n = 262 ∨ n = 458 := by
  sorry

end clock_hands_angle_l3554_355411


namespace boa_constrictor_length_l3554_355490

/-- The length of a boa constrictor given the length of a garden snake and their relative sizes -/
theorem boa_constrictor_length 
  (garden_snake_length : ℝ) 
  (relative_size : ℝ) 
  (h1 : garden_snake_length = 10.0)
  (h2 : relative_size = 7.0) : 
  garden_snake_length / relative_size = 10.0 / 7.0 := by sorry

end boa_constrictor_length_l3554_355490


namespace cats_sold_during_sale_l3554_355446

theorem cats_sold_during_sale
  (initial_siamese : ℕ)
  (initial_house : ℕ)
  (cats_remaining : ℕ)
  (h1 : initial_siamese = 12)
  (h2 : initial_house = 20)
  (h3 : cats_remaining = 12) :
  initial_siamese + initial_house - cats_remaining = 20 :=
by sorry

end cats_sold_during_sale_l3554_355446


namespace guess_number_in_seven_questions_l3554_355436

theorem guess_number_in_seven_questions :
  ∃ (f : Fin 7 → (Nat × Nat)),
    (∀ i, (f i).1 < 100 ∧ (f i).2 < 100) →
    ∀ X ≤ 100,
      ∀ Y ≤ 100,
      (∀ i, Nat.gcd (X + (f i).1) (f i).2 = Nat.gcd (Y + (f i).1) (f i).2) →
      X = Y :=
by sorry

end guess_number_in_seven_questions_l3554_355436


namespace intersection_M_N_l3554_355428

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end intersection_M_N_l3554_355428


namespace gcd_factorial_seven_eight_l3554_355432

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_factorial_seven_eight_l3554_355432


namespace derivative_symmetry_l3554_355453

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem derivative_symmetry (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end derivative_symmetry_l3554_355453


namespace shopkeeper_profit_l3554_355452

theorem shopkeeper_profit (CP : ℝ) (CP_pos : CP > 0) : 
  let LP := CP * 1.3
  let SP := LP * 0.9
  let profit := SP - CP
  let percent_profit := (profit / CP) * 100
  percent_profit = 17 := by sorry

end shopkeeper_profit_l3554_355452


namespace removed_triangles_area_l3554_355477

theorem removed_triangles_area (s : ℝ) (h1 : s > 0) : 
  let x := (s - 8) / 2
  4 * (1/2 * x^2) = 8 :=
by sorry

end removed_triangles_area_l3554_355477


namespace abc_sum_sqrt_l3554_355469

theorem abc_sum_sqrt (a b c : ℝ) 
  (eq1 : b + c = 17) 
  (eq2 : c + a = 18) 
  (eq3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 60 * Real.sqrt 10 := by
  sorry

end abc_sum_sqrt_l3554_355469


namespace equation_solutions_l3554_355450

theorem equation_solutions :
  (∀ x : ℝ, (x + 1) / (x - 1) - 4 / (x^2 - 1) ≠ 1) ∧
  (∀ x : ℝ, x^2 + 3*x - 2 = 0 ↔ x = -3/2 - Real.sqrt 17/2 ∨ x = -3/2 + Real.sqrt 17/2) :=
by sorry

end equation_solutions_l3554_355450


namespace y_derivative_l3554_355439

/-- The function y in terms of x -/
def y (x : ℝ) : ℝ := (3 * x - 2) ^ 2

/-- The derivative of y with respect to x -/
def y' (x : ℝ) : ℝ := 6 * (3 * x - 2)

/-- Theorem stating that y' is the derivative of y -/
theorem y_derivative : ∀ x, deriv y x = y' x := by sorry

end y_derivative_l3554_355439


namespace problem_solution_l3554_355495

theorem problem_solution (a b m n : ℚ) 
  (ha_neg : a < 0) 
  (ha_abs : |a| = 7/4)
  (hb_recip : 1/b = -3/2)
  (hmn_opp : m = -n) :
  4*a / b + 3*(m + n) = 21/2 := by
  sorry

end problem_solution_l3554_355495


namespace sum_of_coefficients_l3554_355483

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 512 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 60 :=
by sorry

end sum_of_coefficients_l3554_355483


namespace raul_shopping_spree_l3554_355480

def initial_amount : ℚ := 87
def comic_price : ℚ := 4
def comic_quantity : ℕ := 8
def novel_price : ℚ := 7
def novel_quantity : ℕ := 3
def magazine_price : ℚ := 5.5
def magazine_quantity : ℕ := 2

def total_spent : ℚ :=
  comic_price * comic_quantity +
  novel_price * novel_quantity +
  magazine_price * magazine_quantity

def remaining_amount : ℚ := initial_amount - total_spent

theorem raul_shopping_spree :
  remaining_amount = 23 := by sorry

end raul_shopping_spree_l3554_355480


namespace parallel_lines_m_value_l3554_355426

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂ ∧ a₁ / b₁ ≠ c₁ / c₂

/-- The value of m for which the lines x + my + 6 = 0 and (m-2)x + 3y + 2m = 0 are parallel -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel 1 m 6 (m-2) 3 (2*m) → m = -1 :=
by sorry

end parallel_lines_m_value_l3554_355426


namespace tshirt_shop_weekly_earnings_l3554_355471

/-- Represents the T-shirt shop's operations and calculates weekly earnings -/
def TShirtShopEarnings : ℕ :=
  let women_shirt_price : ℕ := 18
  let men_shirt_price : ℕ := 15
  let women_shirt_interval : ℕ := 30  -- in minutes
  let men_shirt_interval : ℕ := 40    -- in minutes
  let daily_open_hours : ℕ := 12
  let days_per_week : ℕ := 7
  let minutes_per_hour : ℕ := 60

  let women_shirts_per_day : ℕ := (minutes_per_hour / women_shirt_interval) * daily_open_hours
  let men_shirts_per_day : ℕ := (minutes_per_hour / men_shirt_interval) * daily_open_hours
  
  let daily_earnings : ℕ := women_shirts_per_day * women_shirt_price + men_shirts_per_day * men_shirt_price
  
  daily_earnings * days_per_week

/-- Theorem stating that the weekly earnings of the T-shirt shop is $4914 -/
theorem tshirt_shop_weekly_earnings : TShirtShopEarnings = 4914 := by
  sorry

end tshirt_shop_weekly_earnings_l3554_355471


namespace books_written_proof_l3554_355444

/-- The number of books Zig wrote -/
def zig_books : ℕ := 60

/-- The number of books Flo wrote -/
def flo_books : ℕ := zig_books / 4

/-- The total number of books written by Zig and Flo -/
def total_books : ℕ := zig_books + flo_books

theorem books_written_proof :
  (zig_books = 4 * flo_books) → total_books = 75 := by
  sorry

end books_written_proof_l3554_355444


namespace exist_three_aliens_common_language_l3554_355462

/-- The number of aliens -/
def num_aliens : ℕ := 3 * Nat.factorial 2005

/-- The number of languages -/
def num_languages : ℕ := 2005

/-- A function representing the language used between two aliens -/
def communication_language : Fin num_aliens → Fin num_aliens → Fin num_languages := sorry

/-- The main theorem -/
theorem exist_three_aliens_common_language :
  ∃ (a b c : Fin num_aliens),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    communication_language a b = communication_language b c ∧
    communication_language b c = communication_language a c :=
by sorry

end exist_three_aliens_common_language_l3554_355462


namespace company_picnic_attendance_l3554_355406

theorem company_picnic_attendance (men_attendance : Real) (women_attendance : Real) 
  (total_attendance : Real) (men_percentage : Real) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.31000000000000007 →
  men_attendance * men_percentage + women_attendance * (1 - men_percentage) = total_attendance →
  men_percentage = 0.45 := by
sorry

end company_picnic_attendance_l3554_355406


namespace parallel_line_slope_l3554_355448

/-- Given a line with equation 3x + 6y = -12, this theorem states that
    the slope of any line parallel to it is -1/2. -/
theorem parallel_line_slope (x y : ℝ) :
  (3 : ℝ) * x + 6 * y = -12 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = -1/2 :=
by sorry

end parallel_line_slope_l3554_355448


namespace rectangular_prism_edge_sum_l3554_355441

theorem rectangular_prism_edge_sum (l w h : ℝ) : 
  l * w * h = 8 →                   -- Volume condition
  2 * (l * w + w * h + h * l) = 32 → -- Surface area condition
  ∃ q : ℝ, l = 2 / q ∧ w = 2 ∧ h = 2 * q → -- Geometric progression condition
  4 * (l + w + h) = 28 :=           -- Conclusion: sum of edge lengths
by sorry

end rectangular_prism_edge_sum_l3554_355441


namespace tree_height_l3554_355422

theorem tree_height (hop_distance : ℕ) (slip_distance : ℕ) (total_hours : ℕ) (tree_height : ℕ) : 
  hop_distance = 3 →
  slip_distance = 2 →
  total_hours = 17 →
  tree_height = (total_hours - 1) * (hop_distance - slip_distance) + hop_distance := by
sorry

#eval (17 - 1) * (3 - 2) + 3

end tree_height_l3554_355422


namespace slope_of_cutting_line_l3554_355417

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℚ

/-- Checks if a line cuts a parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The specific parallelogram from the problem -/
def specificParallelogram : Parallelogram :=
  { v1 := ⟨4, 20⟩
  , v2 := ⟨4, 56⟩
  , v3 := ⟨13, 81⟩
  , v4 := ⟨13, 45⟩ }

/-- The theorem to be proved -/
theorem slope_of_cutting_line :
  ∃ (l : Line), cutsIntoCongruentPolygons specificParallelogram l ∧ l.slope = 53 / 9 :=
sorry

end slope_of_cutting_line_l3554_355417


namespace binomial_product_nine_two_seven_two_l3554_355427

theorem binomial_product_nine_two_seven_two :
  Nat.choose 9 2 * Nat.choose 7 2 = 756 := by
  sorry

end binomial_product_nine_two_seven_two_l3554_355427


namespace line_segment_parameterization_l3554_355420

/-- Given a line segment connecting points (1,3) and (4,9), parameterized by x = at + b and y = ct + d,
    where t = 0 corresponds to (1,3) and t = 1 corresponds to (4,9),
    prove that a^2 + b^2 + c^2 + d^2 = 55. -/
theorem line_segment_parameterization (a b c d : ℝ) : 
  (∀ t : ℝ, (a * t + b, c * t + d) = (1 - t, 3 - 3*t) + t • (4, 9)) →
  a^2 + b^2 + c^2 + d^2 = 55 := by
sorry

end line_segment_parameterization_l3554_355420


namespace ellipse_eccentricity_theorem_l3554_355466

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- The right focus of an ellipse -/
def right_focus (e : Ellipse a b) : Point := sorry

/-- The intersection points of a line and an ellipse -/
def intersection_points (e : Ellipse a b) (l : Line) : (Point × Point) := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if two line segments are perpendicular -/
def perpendicular (p1 p2 p3 : Point) : Prop := sorry

theorem ellipse_eccentricity_theorem 
  (a b : ℝ) (e : Ellipse a b) (l : Line) :
  let F := right_focus e
  let (A, B) := intersection_points e l
  perpendicular A F B ∧ distance A F = 3 * distance B F →
  eccentricity e = Real.sqrt 10 / 4 := by
  sorry

end ellipse_eccentricity_theorem_l3554_355466


namespace symmetry_composition_iff_intersection_l3554_355454

-- Define a type for lines in a plane
structure Line where
  -- Add necessary properties for a line

-- Define a type for points in a plane
structure Point where
  -- Add necessary properties for a point

-- Define a symmetry operation
def symmetry (l : Line) : Point → Point := sorry

-- Define composition of symmetries
def compose_symmetries (a b c : Line) : Point → Point :=
  symmetry c ∘ symmetry b ∘ symmetry a

-- Define a predicate for three lines intersecting at a single point
def intersect_at_single_point (a b c : Line) : Prop := sorry

-- The main theorem
theorem symmetry_composition_iff_intersection (a b c : Line) :
  (∃ l : Line, compose_symmetries a b c = symmetry l) ↔ intersect_at_single_point a b c := by
  sorry

end symmetry_composition_iff_intersection_l3554_355454


namespace power_expression_l3554_355464

theorem power_expression (a x y : ℝ) (ha : a > 0) (hx : a^x = 3) (hy : a^y = 5) :
  a^(2*x + y/2) = 9 * Real.sqrt 5 := by
  sorry

end power_expression_l3554_355464


namespace complete_square_result_l3554_355404

/-- Given a quadratic equation x^2 - 6x + 5 = 0, prove that when completing the square, 
    the resulting equation (x + c)^2 = d has d = 4 -/
theorem complete_square_result (c : ℝ) : 
  ∃ d : ℝ, (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + c)^2 = d) ∧ d = 4 := by
  sorry

end complete_square_result_l3554_355404


namespace apple_fractions_l3554_355459

/-- Given that Simone ate 1/2 of an apple each day for 16 days,
    Lauri ate x fraction of an apple each day for 15 days,
    and the total number of apples eaten by both girls is 13,
    prove that x = 1/3 -/
theorem apple_fractions (x : ℚ) : 
  (16 * (1/2 : ℚ)) + (15 * x) = 13 → x = 1/3 := by sorry

end apple_fractions_l3554_355459


namespace punger_baseball_cards_l3554_355402

/-- Given the number of packs, cards per pack, and cards per page, 
    calculate the number of pages needed to store all cards. -/
def pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) : ℕ :=
  (packs * cards_per_pack + cards_per_page - 1) / cards_per_page

theorem punger_baseball_cards : 
  pages_needed 60 7 10 = 42 := by
  sorry

end punger_baseball_cards_l3554_355402


namespace repair_cost_is_5000_l3554_355456

/-- Calculates the repair cost of a machine given its purchase price, transportation charges,
    profit percentage, and final selling price. -/
def repair_cost (purchase_price : ℤ) (transportation_charges : ℤ) (profit_percentage : ℚ)
                (selling_price : ℤ) : ℚ :=
  ((selling_price : ℚ) - (1 + profit_percentage) * ((purchase_price + transportation_charges) : ℚ)) /
  (1 + profit_percentage)

/-- Theorem stating that the repair cost is 5000 given the specific conditions -/
theorem repair_cost_is_5000 :
  repair_cost 13000 1000 (1/2) 28500 = 5000 := by
  sorry

end repair_cost_is_5000_l3554_355456


namespace solve_car_price_l3554_355410

def car_price_problem (total_payment loan_amount interest_rate : ℝ) : Prop :=
  let interest := loan_amount * interest_rate
  let car_price := total_payment - interest
  car_price = 35000

theorem solve_car_price :
  car_price_problem 38000 20000 0.15 :=
sorry

end solve_car_price_l3554_355410


namespace fuel_cost_calculation_l3554_355468

theorem fuel_cost_calculation (original_cost : ℝ) : 
  (2 * original_cost * 1.2 = 480) → original_cost = 200 := by
  sorry

end fuel_cost_calculation_l3554_355468


namespace trivia_team_size_l3554_355416

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) 
  (h1 : absent_members = 2)
  (h2 : points_per_member = 6)
  (h3 : total_points = 18) :
  ∃ (original_size : ℕ), 
    original_size * points_per_member - absent_members * points_per_member = total_points ∧ 
    original_size = 5 := by
  sorry

end trivia_team_size_l3554_355416


namespace undefined_rational_expression_l3554_355418

theorem undefined_rational_expression (x : ℝ) :
  (x^2 - 16*x + 64 = 0) ↔ (x = 8) :=
by sorry

end undefined_rational_expression_l3554_355418


namespace factorial_sum_calculation_l3554_355460

theorem factorial_sum_calculation : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 2 * Nat.factorial 5 = 5160 := by
  sorry

end factorial_sum_calculation_l3554_355460


namespace lineup_combinations_l3554_355423

def total_members : ℕ := 12
def offensive_linemen : ℕ := 5
def positions_to_fill : ℕ := 5

def choose_lineup : ℕ := offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

theorem lineup_combinations :
  choose_lineup = 39600 := by
  sorry

end lineup_combinations_l3554_355423


namespace equation_transformation_l3554_355431

theorem equation_transformation (x y : ℚ) : 
  5 * x - 6 * y = 4 → y = (5/6) * x - 2/3 := by
  sorry

end equation_transformation_l3554_355431


namespace inequality_proof_l3554_355408

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) +
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) +
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
    sorry

end inequality_proof_l3554_355408


namespace solve_z_l3554_355405

-- Define the complex number i
def i : ℂ := Complex.I

-- Define a predicate for purely imaginary numbers
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define the theorem
theorem solve_z (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : ((z + 2) / (1 - i)).im = 0) : 
  z = -2 * i := by
  sorry

end solve_z_l3554_355405


namespace total_meows_in_five_minutes_l3554_355478

/-- The number of meows per minute for the first cat -/
def first_cat_meows : ℕ := 3

/-- The number of meows per minute for the second cat -/
def second_cat_meows : ℕ := 2 * first_cat_meows

/-- The number of meows per minute for the third cat -/
def third_cat_meows : ℕ := second_cat_meows / 3

/-- The duration in minutes -/
def duration : ℕ := 5

/-- Theorem: The total number of meows from all three cats in 5 minutes is 55 -/
theorem total_meows_in_five_minutes :
  first_cat_meows * duration + second_cat_meows * duration + third_cat_meows * duration = 55 := by
  sorry

end total_meows_in_five_minutes_l3554_355478


namespace john_remaining_money_l3554_355449

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let initial_money : ℝ := 50
  let drink_cost : ℝ := q
  let small_pizza_cost : ℝ := q
  let large_pizza_cost : ℝ := 4 * q
  let num_drinks : ℕ := 4
  let num_small_pizzas : ℕ := 2
  let num_large_pizzas : ℕ := 1
  initial_money - (num_drinks * drink_cost + num_small_pizzas * small_pizza_cost + num_large_pizzas * large_pizza_cost)

/-- Theorem stating that John's remaining money is equal to 50 - 10q -/
theorem john_remaining_money (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end john_remaining_money_l3554_355449
