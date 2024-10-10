import Mathlib

namespace meaningful_sqrt_range_l4114_411464

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by
sorry

end meaningful_sqrt_range_l4114_411464


namespace largest_number_l4114_411439

theorem largest_number (a b c d : ℝ) (h1 : a = 3) (h2 : b = -7) (h3 : c = 0) (h4 : d = 1/9) :
  a = max a (max b (max c d)) :=
sorry

end largest_number_l4114_411439


namespace exam_maximum_marks_l4114_411470

/-- Given the conditions of a student's exam performance, 
    prove that the maximum marks are 500. -/
theorem exam_maximum_marks :
  let pass_percentage : ℚ := 33 / 100
  let student_marks : ℕ := 125
  let fail_margin : ℕ := 40
  ∃ (max_marks : ℕ), 
    (pass_percentage * max_marks : ℚ) = (student_marks + fail_margin : ℕ) ∧ 
    max_marks = 500 := by
  sorry

end exam_maximum_marks_l4114_411470


namespace problem_statement_l4114_411467

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_prod : a * b * c = 1)
  (h_eq1 : a + 1 / c = 8)
  (h_eq2 : b + 1 / a = 20) :
  c + 1 / b = 10 / 53 := by
sorry

end problem_statement_l4114_411467


namespace first_train_crossing_time_l4114_411442

/-- Two trains running in opposite directions with equal speeds -/
structure TwoTrains where
  v₁ : ℝ  -- Speed of the first train
  v₂ : ℝ  -- Speed of the second train
  L₁ : ℝ  -- Length of the first train
  L₂ : ℝ  -- Length of the second train
  t₂ : ℝ  -- Time taken by the second train to cross the man
  cross_time : ℝ  -- Time taken for the trains to cross each other

/-- The conditions given in the problem -/
def problem_conditions (trains : TwoTrains) : Prop :=
  trains.v₁ > 0 ∧ 
  trains.v₂ > 0 ∧ 
  trains.L₁ > 0 ∧ 
  trains.L₂ > 0 ∧ 
  trains.v₁ = trains.v₂ ∧  -- Ratio of speeds is 1
  trains.t₂ = 17 ∧  -- Second train crosses the man in 17 seconds
  trains.cross_time = 22 ∧  -- Trains cross each other in 22 seconds
  (trains.L₁ + trains.L₂) / (trains.v₁ + trains.v₂) = trains.cross_time

/-- The theorem to be proved -/
theorem first_train_crossing_time (trains : TwoTrains) 
  (h : problem_conditions trains) : 
  trains.L₁ / trains.v₁ = 27 := by
  sorry


end first_train_crossing_time_l4114_411442


namespace difference_calculation_l4114_411434

theorem difference_calculation : 
  (1 / 10 : ℚ) * 8000 - (1 / 20 : ℚ) / 100 * 8000 = 796 := by
  sorry

end difference_calculation_l4114_411434


namespace halfway_fraction_l4114_411405

theorem halfway_fraction (a b : ℚ) (ha : a = 1/6) (hb : b = 1/4) :
  (a + b) / 2 = 5/24 := by
  sorry

end halfway_fraction_l4114_411405


namespace sin_equality_proof_l4114_411426

theorem sin_equality_proof (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.sin (-474 * π / 180) → n = 66 := by
  sorry

end sin_equality_proof_l4114_411426


namespace sum_of_reciprocals_of_roots_l4114_411468

/-- Given a quadratic equation 6x^2 + 5x + 7, prove that the sum of the reciprocals of its roots is -5/7 -/
theorem sum_of_reciprocals_of_roots (x : ℝ) (γ δ : ℝ) :
  (6 * x^2 + 5 * x + 7 = 0) →
  (∃ p q : ℝ, 6 * p^2 + 5 * p + 7 = 0 ∧ 6 * q^2 + 5 * q + 7 = 0 ∧ γ = 1/p ∧ δ = 1/q) →
  γ + δ = -5/7 := by
sorry

end sum_of_reciprocals_of_roots_l4114_411468


namespace f_negative_implies_a_range_l4114_411498

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * (x - 1)

theorem f_negative_implies_a_range (a : ℝ) :
  (∀ x > 1, f a x < 0) → a ≥ 2 := by sorry

end f_negative_implies_a_range_l4114_411498


namespace quadratic_product_zero_l4114_411450

/-- Given a quadratic polynomial f(x) = ax^2 + bx + c, 
    if f((a - b - c)/(2a)) = 0 and f((c - a - b)/(2a)) = 0, 
    then f(-1) * f(1) = 0 -/
theorem quadratic_product_zero 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f ((a - b - c) / (2 * a)) = 0)
  (h3 : f ((c - a - b) / (2 * a)) = 0)
  : f (-1) * f 1 = 0 := by
  sorry

end quadratic_product_zero_l4114_411450


namespace exponential_characterization_l4114_411447

/-- A continuous function satisfying f(x+y) = f(x)f(y) is of the form aˣ for some a > 0 -/
theorem exponential_characterization (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_nonzero : ∃ x₀, f x₀ ≠ 0) 
  (h_mult : ∀ x y, f (x + y) = f x * f y) : 
  ∃ a > 0, ∀ x, f x = Real.exp (x * Real.log a) := by
sorry

end exponential_characterization_l4114_411447


namespace complex_cube_root_product_l4114_411471

theorem complex_cube_root_product (w : ℂ) (hw : w^3 = 1) :
  (1 - w + w^2) * (1 + w - w^2) = 4 := by
  sorry

end complex_cube_root_product_l4114_411471


namespace blue_bicycle_selection_count_l4114_411459

/-- The number of ways to select at least two blue shared bicycles -/
def select_blue_bicycles : ℕ :=
  (Nat.choose 4 2 * Nat.choose 6 2) + (Nat.choose 4 3 * Nat.choose 6 1) + Nat.choose 4 4

/-- Theorem stating that the number of ways to select at least two blue shared bicycles is 115 -/
theorem blue_bicycle_selection_count :
  select_blue_bicycles = 115 := by sorry

end blue_bicycle_selection_count_l4114_411459


namespace jake_arrival_delay_l4114_411457

/-- Represents the problem of Austin and Jake descending a building --/
structure DescentProblem where
  floors : ℕ               -- Number of floors to descend
  steps_per_floor : ℕ      -- Number of steps per floor
  jake_speed : ℕ           -- Jake's speed in steps per second
  elevator_time : ℕ        -- Time taken by elevator in seconds

/-- Calculates the time difference between Jake's arrival and the elevator's arrival --/
def time_difference (p : DescentProblem) : ℤ :=
  let total_steps := p.floors * p.steps_per_floor
  let jake_time := total_steps / p.jake_speed
  jake_time - p.elevator_time

/-- The main theorem stating that Jake will arrive 20 seconds after the elevator --/
theorem jake_arrival_delay (p : DescentProblem) 
  (h1 : p.floors = 8)
  (h2 : p.steps_per_floor = 30)
  (h3 : p.jake_speed = 3)
  (h4 : p.elevator_time = 60) : 
  time_difference p = 20 := by
  sorry

#eval time_difference ⟨8, 30, 3, 60⟩

end jake_arrival_delay_l4114_411457


namespace power_function_value_l4114_411490

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f (1/2) = 8) : f 2 = 1/8 := by
  sorry

end power_function_value_l4114_411490


namespace simplify_expression_1_l4114_411418

theorem simplify_expression_1 (x : ℝ) : 
  5*x^2 + x + 3 + 4*x - 8*x^2 - 2 = -3*x^2 + 5*x + 1 := by sorry

end simplify_expression_1_l4114_411418


namespace twenty_five_percent_less_than_eighty_is_half_more_than_forty_l4114_411454

theorem twenty_five_percent_less_than_eighty_is_half_more_than_forty : 
  ∃ x : ℝ, (80 - 0.25 * 80 = x + 0.5 * x) ∧ x = 40 := by
  sorry

end twenty_five_percent_less_than_eighty_is_half_more_than_forty_l4114_411454


namespace inequality_proof_l4114_411431

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) 
  (hab : a + b = 1) : 
  (a*x + b*y) * (b*x + a*y) ≥ x*y := by
sorry

end inequality_proof_l4114_411431


namespace complex_equation_l4114_411429

/-- Given x ∈ ℝ, y is a pure imaginary number, and (x-y)i = 2-i, then x+y = -1+2i -/
theorem complex_equation (x : ℝ) (y : ℂ) (h1 : y.re = 0) (h2 : (x - y) * Complex.I = 2 - Complex.I) : 
  x + y = -1 + 2 * Complex.I := by
sorry

end complex_equation_l4114_411429


namespace parts_from_64_blanks_l4114_411424

/-- Calculates the total number of parts that can be produced from a given number of initial blanks,
    where shavings from a certain number of parts can be remelted into one new blank. -/
def total_parts (initial_blanks : ℕ) (parts_per_remelted_blank : ℕ) : ℕ :=
  let first_batch := initial_blanks
  let second_batch := initial_blanks / parts_per_remelted_blank
  let third_batch := second_batch / parts_per_remelted_blank
  first_batch + second_batch + third_batch

/-- Theorem stating that given 64 initial blanks and the ability to remelt shavings 
    from 8 parts into one new blank, the total number of parts that can be produced is 73. -/
theorem parts_from_64_blanks : total_parts 64 8 = 73 := by
  sorry

end parts_from_64_blanks_l4114_411424


namespace smallest_angle_trig_equation_l4114_411472

theorem smallest_angle_trig_equation : 
  (∃ (x : ℝ), x > 0 ∧ x < 10 * π / 180 ∧ Real.sin (4*x) * Real.sin (5*x) = Real.cos (4*x) * Real.cos (5*x)) ∨
  (∀ (x : ℝ), x > 0 ∧ Real.sin (4*x) * Real.sin (5*x) = Real.cos (4*x) * Real.cos (5*x) → x ≥ 10 * π / 180) :=
by sorry

end smallest_angle_trig_equation_l4114_411472


namespace equation_solutions_l4114_411413

theorem equation_solutions :
  (∃ x : ℚ, x / (3/4) = 2 / (9/10) ∧ x = 5/3) ∧
  (∃ x : ℚ, 0.5 / x = 0.75 / 6 ∧ x = 4) ∧
  (∃ x : ℚ, x / 20 = 2/5 ∧ x = 8) ∧
  (∃ x : ℚ, (3/4 * x) / 15 = 2/3 ∧ x = 40/3) :=
by sorry

end equation_solutions_l4114_411413


namespace zoo_guide_problem_l4114_411495

/-- Given two zoo guides speaking to groups of children, where one guide spoke to 25 children
    and the total number of children spoken to is 44, prove that the number of children
    the other guide spoke to is 19. -/
theorem zoo_guide_problem (total_children : ℕ) (second_guide_children : ℕ) :
  total_children = 44 →
  second_guide_children = 25 →
  total_children - second_guide_children = 19 :=
by sorry

end zoo_guide_problem_l4114_411495


namespace orange_savings_l4114_411407

theorem orange_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ) :
  liam_oranges = 40 →
  liam_price = 5/2 →
  claire_oranges = 30 →
  claire_price = 6/5 →
  (liam_oranges / 2 * liam_price + claire_oranges * claire_price : ℚ) = 86 := by
  sorry

end orange_savings_l4114_411407


namespace triangle_area_is_integer_l4114_411483

theorem triangle_area_is_integer (a b c : ℝ) (h1 : a^2 = 377) (h2 : b^2 = 153) (h3 : c^2 = 80)
  (h4 : ∃ (w h : ℤ), ∃ (x y z : ℝ),
    (x^2 + y^2 = w^2) ∧ (x^2 + z^2 = h^2) ∧
    (y + z = a ∨ y + z = b ∨ y + z = c) ∧
    (∃ (d1 d2 : ℤ), d1 ≥ 0 ∧ d2 ≥ 0 ∧ d1 + d2 + x = w ∧ d1 + d2 + y = h)) :
  ∃ (A : ℤ), A = 42 ∧ 16 * A^2 = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b) :=
by sorry

end triangle_area_is_integer_l4114_411483


namespace volume_of_region_l4114_411473

-- Define the region
def Region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   |x - y + z| + |x - y - z| ≤ 10 ∧
                   x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0}

-- State the theorem
theorem volume_of_region : 
  MeasureTheory.volume Region = 125 := by sorry

end volume_of_region_l4114_411473


namespace contest_finish_orders_l4114_411489

def number_of_participants : ℕ := 3

theorem contest_finish_orders :
  (Nat.factorial number_of_participants) = 6 := by
  sorry

end contest_finish_orders_l4114_411489


namespace product_three_consecutive_divisibility_l4114_411486

theorem product_three_consecutive_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 5 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 15 * m) ∧
  (∃ m : ℤ, n = 30 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 60 * m) :=
by sorry

end product_three_consecutive_divisibility_l4114_411486


namespace solution_equivalence_l4114_411461

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {((1 : ℝ) / Real.rpow 6 (1/6), Real.sqrt 2 / Real.rpow 6 (1/6), Real.sqrt 3 / Real.rpow 6 (1/6)),
   (-(1 : ℝ) / Real.rpow 6 (1/6), -Real.sqrt 2 / Real.rpow 6 (1/6), Real.sqrt 3 / Real.rpow 6 (1/6)),
   (-(1 : ℝ) / Real.rpow 6 (1/6), Real.sqrt 2 / Real.rpow 6 (1/6), -Real.sqrt 3 / Real.rpow 6 (1/6)),
   ((1 : ℝ) / Real.rpow 6 (1/6), -Real.sqrt 2 / Real.rpow 6 (1/6), -Real.sqrt 3 / Real.rpow 6 (1/6))}

def satisfies_equations (x y z : ℝ) : Prop :=
  x^3 * y^3 * z^3 = 1 ∧ x * y^5 * z^3 = 2 ∧ x * y^3 * z^5 = 3

theorem solution_equivalence :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end solution_equivalence_l4114_411461


namespace at_least_one_less_than_or_equal_to_one_l4114_411453

theorem at_least_one_less_than_or_equal_to_one 
  (x y z : ℝ) 
  (pos_x : 0 < x) 
  (pos_y : 0 < y) 
  (pos_z : 0 < z) 
  (sum_eq_three : x + y + z = 3) :
  (x * (x + y - z) ≤ 1) ∨ (y * (y + z - x) ≤ 1) ∨ (z * (z + x - y) ≤ 1) := by
  sorry

end at_least_one_less_than_or_equal_to_one_l4114_411453


namespace fundamental_disagreement_essence_l4114_411448

-- Define philosophical viewpoints
def materialist_viewpoint : String := "Without scenery, where does emotion come from?"
def idealist_viewpoint : String := "Without emotion, where does scenery come from?"

-- Define the concept of fundamental disagreement
def fundamental_disagreement (v1 v2 : String) : Prop := sorry

-- Define the essence of the world
inductive WorldEssence
| Material
| Consciousness

-- Theorem statement
theorem fundamental_disagreement_essence :
  fundamental_disagreement materialist_viewpoint idealist_viewpoint ↔
  ∃ (e : WorldEssence), (e = WorldEssence.Material ∨ e = WorldEssence.Consciousness) :=
sorry

end fundamental_disagreement_essence_l4114_411448


namespace science_club_membership_l4114_411411

theorem science_club_membership (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : biology = 50)
  (h3 : chemistry = 40)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 15 :=
by sorry

end science_club_membership_l4114_411411


namespace johns_remaining_money_l4114_411492

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after subtracting flight cost --/
def remaining_money (savings : ℕ) (flight_cost : ℕ) : ℕ :=
  octal_to_decimal savings - flight_cost

/-- Theorem stating that John's remaining money is 1725 in decimal --/
theorem johns_remaining_money :
  remaining_money 5555 1200 = 1725 := by sorry

end johns_remaining_money_l4114_411492


namespace average_first_15_even_numbers_l4114_411416

theorem average_first_15_even_numbers : 
  let first_15_even : List ℕ := List.range 15 |>.map (fun n => 2 * (n + 1))
  (first_15_even.sum / first_15_even.length : ℚ) = 16 := by
  sorry

end average_first_15_even_numbers_l4114_411416


namespace tan_alpha_two_implies_fraction_equals_negative_three_l4114_411460

theorem tan_alpha_two_implies_fraction_equals_negative_three (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -3 := by
  sorry

end tan_alpha_two_implies_fraction_equals_negative_three_l4114_411460


namespace triangle_equilateral_conditions_l4114_411414

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (h_a h_b h_c : ℝ)
  (ha_pos : h_a > 0)
  (hb_pos : h_b > 0)
  (hc_pos : h_c > 0)

-- Define the property of having equal sums of side and height
def equal_side_height_sums (t : Triangle) : Prop :=
  t.a + t.h_a = t.b + t.h_b ∧ t.b + t.h_b = t.c + t.h_c

-- Define the property of having equal inscribed squares
def equal_inscribed_squares (t : Triangle) : Prop :=
  (2 * t.a * t.h_a) / (t.a + t.h_a) = (2 * t.b * t.h_b) / (t.b + t.h_b) ∧
  (2 * t.b * t.h_b) / (t.b + t.h_b) = (2 * t.c * t.h_c) / (t.c + t.h_c)

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem triangle_equilateral_conditions (t : Triangle) :
  (equal_side_height_sums t ∨ equal_inscribed_squares t) → is_equilateral t :=
by sorry

end triangle_equilateral_conditions_l4114_411414


namespace leo_weight_l4114_411499

/-- Given that Leo's weight plus 10 pounds is 1.5 times Kendra's weight,
    and that their combined weight is 180 pounds,
    prove that Leo's current weight is 104 pounds. -/
theorem leo_weight (leo kendra : ℝ) 
  (h1 : leo + 10 = 1.5 * kendra) 
  (h2 : leo + kendra = 180) : 
  leo = 104 := by sorry

end leo_weight_l4114_411499


namespace austin_picked_24_bags_l4114_411443

/-- The number of bags of apples Dallas picked -/
def dallas_apples : ℕ := 14

/-- The number of bags of pears Dallas picked -/
def dallas_pears : ℕ := 9

/-- The number of additional bags of apples Austin picked compared to Dallas -/
def austin_extra_apples : ℕ := 6

/-- The number of fewer bags of pears Austin picked compared to Dallas -/
def austin_fewer_pears : ℕ := 5

/-- The total number of bags of fruit Austin picked -/
def austin_total : ℕ := (dallas_apples + austin_extra_apples) + (dallas_pears - austin_fewer_pears)

theorem austin_picked_24_bags :
  austin_total = 24 := by sorry

end austin_picked_24_bags_l4114_411443


namespace w_squared_values_l4114_411496

theorem w_squared_values (w : ℝ) :
  (2 * w + 17)^2 = (4 * w + 9) * (3 * w + 6) →
  w^2 = 19.69140625 ∨ w^2 = 43.06640625 := by
  sorry

end w_squared_values_l4114_411496


namespace percentage_increase_l4114_411462

theorem percentage_increase (original : ℝ) (final : ℝ) (increase : ℝ) :
  original = 90 →
  final = 135 →
  increase = (final - original) / original * 100 →
  increase = 50 := by
sorry

end percentage_increase_l4114_411462


namespace not_right_triangle_l4114_411475

theorem not_right_triangle (a b c : ℕ) (h : a = 7 ∧ b = 9 ∧ c = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end not_right_triangle_l4114_411475


namespace scale_drawing_conversion_l4114_411481

/-- Converts a length in inches to miles, given a scale where 1 inch represents 1000 feet --/
def inches_to_miles (inches : ℚ) : ℚ :=
  inches * 1000 / 5280

/-- Theorem stating that 7.5 inches on the given scale represents 125/88 miles --/
theorem scale_drawing_conversion :
  inches_to_miles (7.5) = 125 / 88 := by
  sorry

end scale_drawing_conversion_l4114_411481


namespace painted_cubes_count_l4114_411444

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes with one face painted in a rectangular solid -/
def cubes_with_one_face_painted (solid : RectangularSolid) : ℕ :=
  2 * ((solid.length - 2) * (solid.width - 2) +
       (solid.length - 2) * (solid.height - 2) +
       (solid.width - 2) * (solid.height - 2))

/-- Theorem: In a 9x10x11 rectangular solid, 382 cubes have exactly one face painted -/
theorem painted_cubes_count :
  let solid : RectangularSolid := ⟨9, 10, 11⟩
  cubes_with_one_face_painted solid = 382 := by
  sorry

end painted_cubes_count_l4114_411444


namespace milk_mixture_problem_l4114_411415

/-- Proves that the butterfat percentage of the added milk is 10% given the conditions of the problem -/
theorem milk_mixture_problem (final_percentage : ℝ) (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  final_percentage = 20 →
  initial_volume = 8 →
  initial_percentage = 30 →
  added_volume = 8 →
  (initial_volume * initial_percentage + added_volume * (100 * final_percentage - initial_volume * initial_percentage) / added_volume) / (initial_volume + added_volume) = 10 := by
sorry

end milk_mixture_problem_l4114_411415


namespace sqrt_equality_implies_t_value_l4114_411422

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → t = 3.7 := by
  sorry

end sqrt_equality_implies_t_value_l4114_411422


namespace cos_96_cos_24_minus_sin_96_sin_24_l4114_411400

theorem cos_96_cos_24_minus_sin_96_sin_24 :
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (24 * π / 180) = -1/2 := by
  sorry

end cos_96_cos_24_minus_sin_96_sin_24_l4114_411400


namespace max_value_polynomial_l4114_411441

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 → x^3*y + x^2*y + x*y + x*y^2 + x*y^3 ≤ max) ∧
  (x^3*y + x^2*y + x*y + x*y^2 + x*y^3 ≤ 961/8) :=
sorry

end max_value_polynomial_l4114_411441


namespace real_roots_condition_zero_sum_of_squares_l4114_411420

-- Statement 1
theorem real_roots_condition (q : ℝ) :
  q < 1 → ∃ x : ℝ, x^2 + 2*x + q = 0 :=
sorry

-- Statement 2
theorem zero_sum_of_squares (x y : ℝ) :
  x^2 + y^2 = 0 → x = 0 ∧ y = 0 :=
sorry

end real_roots_condition_zero_sum_of_squares_l4114_411420


namespace number_of_topics_six_students_three_groups_ninety_arrangements_l4114_411402

theorem number_of_topics (num_students : Nat) (num_groups : Nat) (num_arrangements : Nat) : Nat :=
  let students_per_group := num_students / num_groups
  let ways_to_divide := num_arrangements / (num_groups^students_per_group)
  ways_to_divide

theorem six_students_three_groups_ninety_arrangements :
  number_of_topics 6 3 90 = 1 := by sorry

end number_of_topics_six_students_three_groups_ninety_arrangements_l4114_411402


namespace max_profit_l4114_411494

/-- Annual sales revenue function -/
noncomputable def Q (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then -x^2 + 1040*x + 1200
  else if x > 30 then 998*x - 2048/(x-2) + 1800
  else 0

/-- Annual total profit function (in million yuan) -/
noncomputable def W (x : ℝ) : ℝ :=
  (Q x - (1000*x + 600)) / 1000

/-- The maximum profit is 1068 million yuan -/
theorem max_profit :
  ∃ x : ℝ, x > 0 ∧ W x = 1068 ∧ ∀ y : ℝ, y > 0 → W y ≤ W x :=
sorry

end max_profit_l4114_411494


namespace fraction_ratio_equality_l4114_411485

theorem fraction_ratio_equality : 
  (15 / 8) / (2 / 5) = (3 / 8) / (1 / 5) := by
  sorry

end fraction_ratio_equality_l4114_411485


namespace record_listening_time_l4114_411451

/-- The number of days required to listen to a record collection --/
def days_to_listen (initial_records : ℕ) (gift_records : ℕ) (purchased_records : ℕ) (days_per_record : ℕ) : ℕ :=
  (initial_records + gift_records + purchased_records) * days_per_record

/-- Theorem: Given the initial conditions, it takes 100 days to listen to the entire record collection --/
theorem record_listening_time : days_to_listen 8 12 30 2 = 100 := by
  sorry

end record_listening_time_l4114_411451


namespace dartboard_section_angle_l4114_411417

/-- Represents a circular dartboard divided into sections -/
structure Dartboard where
  /-- The probability of a dart landing in a particular section -/
  section_probability : ℝ
  /-- The central angle of the section in degrees -/
  section_angle : ℝ

/-- 
Theorem: For a circular dartboard divided into sections by radius lines, 
if the probability of a dart landing in a particular section is 1/4, 
then the central angle of that section is 90 degrees.
-/
theorem dartboard_section_angle (d : Dartboard) 
  (h_prob : d.section_probability = 1/4) : 
  d.section_angle = 90 := by
  sorry

end dartboard_section_angle_l4114_411417


namespace two_digit_number_theorem_l4114_411479

/-- Given a two-digit number, return its tens digit -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Given a two-digit number, return its ones digit -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- The product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ := (tens_digit n) * (ones_digit n)

/-- The sum of digits of a two-digit number -/
def digit_sum (n : ℕ) : ℕ := (tens_digit n) + (ones_digit n)

theorem two_digit_number_theorem (x : ℕ) 
  (h1 : is_two_digit x)
  (h2 : digit_product (x + 46) = 6)
  (h3 : digit_sum x = 14) :
  x = 77 ∨ x = 86 := by
sorry

end two_digit_number_theorem_l4114_411479


namespace race_probability_l4114_411487

theorem race_probability (pA pB pC pD pE : ℚ) 
  (hA : pA = 1/4) 
  (hB : pB = 1/8) 
  (hC : pC = 1/12) 
  (hD : pD = 1/20) 
  (hE : pE = 1/30) : 
  pA + pB + pC + pD + pE = 65/120 := by
sorry

end race_probability_l4114_411487


namespace function_intersection_l4114_411456

/-- Two functions f and g have exactly one common point if and only if (a-c)(b-d) = 2,
    given that they are centrally symmetric with respect to the point ((b+d)/2, a+c) -/
theorem function_intersection (a b c d : ℝ) :
  let f (x : ℝ) := 2*a + 1/(x-b)
  let g (x : ℝ) := 2*c + 1/(x-d)
  let center : ℝ × ℝ := ((b+d)/2, a+c)
  (∃! p, f p = g p ∧ 
   ∀ x y, f x = y ↔ g (b+d-x) = 2*(a+c)-y) ↔ 
  (a-c)*(b-d) = 2 := by
  sorry

end function_intersection_l4114_411456


namespace sandy_grew_six_carrots_l4114_411408

/-- The number of carrots grown by Sandy -/
def sandy_carrots : ℕ := sorry

/-- The number of carrots grown by Sam -/
def sam_carrots : ℕ := 3

/-- The total number of carrots grown by Sandy and Sam -/
def total_carrots : ℕ := 9

/-- Theorem stating that Sandy grew 6 carrots -/
theorem sandy_grew_six_carrots : sandy_carrots = 6 := by sorry

end sandy_grew_six_carrots_l4114_411408


namespace book_selling_price_l4114_411436

theorem book_selling_price 
  (num_books : ℕ) 
  (buying_price : ℚ) 
  (price_difference : ℚ) 
  (h1 : num_books = 15)
  (h2 : buying_price = 11)
  (h3 : price_difference = 210) :
  ∃ (selling_price : ℚ), 
    selling_price * num_books - buying_price * num_books = price_difference ∧ 
    selling_price = 25 :=
by sorry

end book_selling_price_l4114_411436


namespace intersection_distance_zero_l4114_411474

/-- The distance between the intersection points of x^2 + y^2 = 18 and x + y = 6 is 0 -/
theorem intersection_distance_zero : 
  let S := {p : ℝ × ℝ | p.1^2 + p.2^2 = 18 ∧ p.1 + p.2 = 6}
  ∃! p : ℝ × ℝ, p ∈ S :=
by
  sorry

end intersection_distance_zero_l4114_411474


namespace fifth_pattern_white_tiles_l4114_411445

/-- The number of white tiles in the n-th pattern of a hexagonal tile sequence -/
def white_tiles (n : ℕ) : ℕ := 4 * n + 2

/-- Theorem: The number of white tiles in the fifth pattern is 22 -/
theorem fifth_pattern_white_tiles : white_tiles 5 = 22 := by
  sorry

end fifth_pattern_white_tiles_l4114_411445


namespace union_of_A_and_B_l4114_411406

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x : ℝ | 3 < x ∧ x < 9}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 9} := by
  sorry

end union_of_A_and_B_l4114_411406


namespace qingming_festival_probability_l4114_411419

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade7 : ℕ
  grade8 : ℕ
  grade9 : ℕ

/-- Represents the participation methods for each grade --/
structure ParticipationMethods where
  memorial_hall : GradeDistribution
  online : GradeDistribution

/-- The main theorem to prove --/
theorem qingming_festival_probability 
  (total_students : GradeDistribution)
  (participation : ParticipationMethods)
  (h1 : total_students.grade7 = 4 * k)
  (h2 : total_students.grade8 = 5 * k)
  (h3 : total_students.grade9 = 6 * k)
  (h4 : participation.memorial_hall.grade7 = 2 * a - 1)
  (h5 : participation.memorial_hall.grade8 = 8)
  (h6 : participation.memorial_hall.grade9 = 10)
  (h7 : participation.online.grade7 = a)
  (h8 : participation.online.grade8 = b)
  (h9 : participation.online.grade9 = 2)
  (h10 : total_students.grade7 = participation.memorial_hall.grade7 + participation.online.grade7)
  (h11 : total_students.grade8 = participation.memorial_hall.grade8 + participation.online.grade8)
  (h12 : total_students.grade9 = participation.memorial_hall.grade9 + participation.online.grade9)
  : ℚ :=
  5/21

/-- Auxiliary function to calculate combinations --/
def combinations (n : ℕ) (r : ℕ) : ℕ := sorry

#check qingming_festival_probability

end qingming_festival_probability_l4114_411419


namespace exist_three_distinct_naturals_sum_product_squares_l4114_411421

theorem exist_three_distinct_naturals_sum_product_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ (m : ℕ), a + b + c = m^2) ∧
  (∃ (n : ℕ), a * b * c = n^2) := by
  sorry

end exist_three_distinct_naturals_sum_product_squares_l4114_411421


namespace ellipse_k_value_l4114_411427

/-- The equation of an ellipse with a parameter k -/
def ellipse_equation (x y k : ℝ) : Prop := x^2 + (k*y^2)/5 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (0, 2)

/-- Theorem: For an ellipse with the given equation and focus, k equals 1 -/
theorem ellipse_k_value :
  ∃ k : ℝ, (∀ x y : ℝ, ellipse_equation x y k) ∧ 
  (focus.1 = 0 ∧ focus.2 = 2) → k = 1 :=
sorry

end ellipse_k_value_l4114_411427


namespace stadium_entrance_count_l4114_411497

/-- The number of placards initially in the basket -/
def initial_placards : ℕ := 5682

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The number of people who entered the stadium -/
def people_entered : ℕ := initial_placards / placards_per_person

theorem stadium_entrance_count :
  people_entered = 2841 :=
sorry

end stadium_entrance_count_l4114_411497


namespace binomial_15_3_l4114_411480

theorem binomial_15_3 : Nat.choose 15 3 = 455 := by
  sorry

end binomial_15_3_l4114_411480


namespace negation_of_existence_negation_of_factorial_squared_gt_power_of_two_l4114_411428

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬ ∃ n, P n) ↔ (∀ n, ¬ P n) :=
by sorry

theorem negation_of_factorial_squared_gt_power_of_two : 
  (¬ ∃ n : ℕ, (n.factorial ^ 2 : ℝ) > 2^n) ↔ 
  (∀ n : ℕ, (n.factorial ^ 2 : ℝ) ≤ 2^n) :=
by sorry

end negation_of_existence_negation_of_factorial_squared_gt_power_of_two_l4114_411428


namespace even_function_implies_a_equals_one_l4114_411446

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function y = (x+1)(x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one :
  ∃ a : ℝ, IsEven (f a) → a = 1 := by sorry

end even_function_implies_a_equals_one_l4114_411446


namespace student_council_committees_l4114_411409

theorem student_council_committees (n : ℕ) (k : ℕ) (m : ℕ) (p : ℕ) (w : ℕ) :
  n = 15 →  -- Total number of student council members
  k = 3 →   -- Size of welcoming committee
  m = 4 →   -- Size of planning committee
  p = 2 →   -- Size of finance committee
  w = 20 →  -- Number of ways to select welcoming committee
  (n.choose m) * (k.choose p) = 4095 :=
by sorry

end student_council_committees_l4114_411409


namespace factories_unchecked_l4114_411463

/-- The number of unchecked factories given the total number of factories and the number checked by two groups -/
def unchecked_factories (total : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  total - (group1 + group2)

/-- Theorem stating that 67 factories remain unchecked -/
theorem factories_unchecked :
  unchecked_factories 259 105 87 = 67 := by
  sorry

end factories_unchecked_l4114_411463


namespace rachel_homework_l4114_411435

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 3 → reading_pages = math_pages + 1 → reading_pages = 4 := by
  sorry

end rachel_homework_l4114_411435


namespace factor_expression_l4114_411488

theorem factor_expression (x : ℝ) : 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end factor_expression_l4114_411488


namespace employee_discount_percentage_l4114_411433

/-- Proves that the employee discount percentage is 10% given the problem conditions --/
theorem employee_discount_percentage 
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 216) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 10 := by sorry

end employee_discount_percentage_l4114_411433


namespace white_balls_count_l4114_411476

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_or_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 8 →
  red = 9 →
  purple = 3 →
  prob_not_red_or_purple = 88/100 →
  total - (green + yellow + red + purple) = 50 :=
by sorry

end white_balls_count_l4114_411476


namespace largest_common_divisor_of_difference_of_squares_l4114_411403

theorem largest_common_divisor_of_difference_of_squares (m n : ℤ) 
  (h_m_even : Even m) (h_n_odd : Odd n) (h_n_lt_m : n < m) :
  (∀ k : ℤ, k ∣ (m^2 - n^2) → k ≤ 2) ∧ 2 ∣ (m^2 - n^2) := by
  sorry

#check largest_common_divisor_of_difference_of_squares

end largest_common_divisor_of_difference_of_squares_l4114_411403


namespace geometric_sequence_first_term_l4114_411412

theorem geometric_sequence_first_term (a b c d : ℚ) :
  (∃ r : ℚ, r ≠ 0 ∧ 
    a * r^4 = 48 ∧ 
    a * r^5 = 192) →
  a = 3/16 := by
sorry

end geometric_sequence_first_term_l4114_411412


namespace product_remainder_l4114_411440

theorem product_remainder (a b m : ℕ) (h : a * b = 145 * 155) (hm : m = 12) : 
  (a * b) % m = 11 := by
  sorry

end product_remainder_l4114_411440


namespace f_properties_l4114_411458

-- Define the function f
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

-- State the theorem
theorem f_properties (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (h3 : f a a b c = 0) (h4 : f b a b c = 0) (h5 : f c a b c = 0) :
  (f 0 a b c) * (f 1 a b c) < 0 ∧ (f 0 a b c) * (f 3 a b c) > 0 := by
  sorry

end f_properties_l4114_411458


namespace no_unique_solution_l4114_411482

/-- 
Given a system of two linear equations:
  3(3x + 4y) = 36
  kx + cy = 30
where k = 9, prove that the system does not have a unique solution when c = 12.
-/
theorem no_unique_solution (x y : ℝ) : 
  (3 * (3 * x + 4 * y) = 36) →
  (9 * x + 12 * y = 30) →
  ¬ (∃! (x y : ℝ), 3 * (3 * x + 4 * y) = 36 ∧ 9 * x + 12 * y = 30) :=
by sorry

end no_unique_solution_l4114_411482


namespace scientific_notation_equivalence_l4114_411493

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.0000037 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.7 ∧ n = -6 := by
  sorry

end scientific_notation_equivalence_l4114_411493


namespace probability_not_greater_than_2_78_l4114_411469

def digits : Finset ℕ := {7, 1, 8}

def valid_combinations : Finset (ℕ × ℕ) :=
  {(1, 7), (1, 8), (7, 1), (7, 8)}

theorem probability_not_greater_than_2_78 :
  (Finset.card valid_combinations) / (Finset.card (digits.product digits)) = 2 / 3 := by
  sorry

end probability_not_greater_than_2_78_l4114_411469


namespace solution_range_l4114_411437

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (x + m) / (x - 2) - 3 = (x - 1) / (2 - x)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ equation x m) ↔ (m ≥ -5 ∧ m ≠ -3) :=
sorry

end solution_range_l4114_411437


namespace ice_cube_distribution_l4114_411401

theorem ice_cube_distribution (total_cubes : ℕ) (num_chests : ℕ) (cubes_per_chest : ℕ) 
  (h1 : total_cubes = 294)
  (h2 : num_chests = 7)
  (h3 : total_cubes = num_chests * cubes_per_chest) :
  cubes_per_chest = 42 := by
  sorry

end ice_cube_distribution_l4114_411401


namespace phil_coin_collection_l4114_411484

def coin_collection (initial : ℕ) (year1 : ℕ → ℕ) (year2 : ℕ) (year3 : ℕ) (year4 : ℕ) 
                    (year5 : ℕ → ℕ) (year6 : ℕ) (year7 : ℕ) (year8 : ℕ) (year9 : ℕ → ℕ) : ℕ :=
  let after_year1 := year1 initial
  let after_year2 := after_year1 + year2
  let after_year3 := after_year2 + year3
  let after_year4 := after_year3 + year4
  let after_year5 := year5 after_year4
  let after_year6 := after_year5 + year6
  let after_year7 := after_year6 + year7
  let after_year8 := after_year7 + year8
  year9 after_year8

theorem phil_coin_collection :
  coin_collection 1000 (λ x => x * 4) (7 * 52) (3 * 182) (2 * 52) 
                  (λ x => x - (x * 2 / 5)) (5 * 91) (20 * 12) (10 * 52)
                  (λ x => x - (x / 3)) = 2816 := by
  sorry

end phil_coin_collection_l4114_411484


namespace triangle_angle_extension_l4114_411410

theorem triangle_angle_extension (a b c x : Real) : 
  a = 50 → b = 60 → c = 180 - a - b → 
  x = 180 - (180 - c) → x = 70 := by
  sorry

end triangle_angle_extension_l4114_411410


namespace mary_max_earnings_l4114_411452

/-- Calculates the maximum weekly earnings for a worker with given parameters. -/
def maxWeeklyEarnings (maxHours : ℕ) (regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  let regularEarnings := (regularHours.min maxHours : ℚ) * regularRate
  let overtimeHours := maxHours - regularHours
  let overtimeEarnings := (overtimeHours.max 0 : ℚ) * overtimeRate
  regularEarnings + overtimeEarnings

/-- Theorem stating Mary's maximum weekly earnings -/
theorem mary_max_earnings :
  maxWeeklyEarnings 80 20 8 (1/4) = 760 := by
  sorry

end mary_max_earnings_l4114_411452


namespace expression_evaluation_l4114_411466

theorem expression_evaluation :
  (18^40 : ℕ) / (54^20) * 2^10 = 2^30 * 3^20 :=
by sorry

end expression_evaluation_l4114_411466


namespace fourth_year_afforestation_l4114_411491

/-- Calculates the area afforested in a given year, given the initial area and annual increase rate. -/
def area_afforested (initial_area : ℝ) (annual_increase : ℝ) (year : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ (year - 1)

/-- Theorem stating that given an initial area of 10,000 acres and an annual increase of 20%,
    the area afforested in the fourth year is 17,280 acres. -/
theorem fourth_year_afforestation :
  area_afforested 10000 0.2 4 = 17280 := by
  sorry

end fourth_year_afforestation_l4114_411491


namespace trig_identity_simplification_l4114_411432

theorem trig_identity_simplification (x y : ℝ) : 
  Real.cos (x + y) * Real.sin y - Real.sin (x + y) * Real.cos y = -Real.sin (x + y) := by
  sorry

end trig_identity_simplification_l4114_411432


namespace evaluate_expression_l4114_411477

theorem evaluate_expression (b : ℝ) : 
  let x := 2 * b + 9
  x - 2 * b + 5 = 14 := by sorry

end evaluate_expression_l4114_411477


namespace opposite_unit_vector_l4114_411423

/-- Given a vector a = (-3, 4), prove that the unit vector a₀ in the opposite direction of a has coordinates (3/5, -4/5). -/
theorem opposite_unit_vector (a : ℝ × ℝ) (h : a = (-3, 4)) :
  let a₀ := (-(a.1) / Real.sqrt ((a.1)^2 + (a.2)^2), -(a.2) / Real.sqrt ((a.1)^2 + (a.2)^2))
  a₀ = (3/5, -4/5) := by
sorry


end opposite_unit_vector_l4114_411423


namespace discrete_probability_distribution_l4114_411404

theorem discrete_probability_distribution (p₁ p₃ : ℝ) : 
  p₃ = 4 * p₁ →
  p₁ + 0.15 + p₃ + 0.25 + 0.35 = 1 →
  p₁ = 0.05 ∧ p₃ = 0.20 := by
sorry

end discrete_probability_distribution_l4114_411404


namespace horner_rule_v3_l4114_411449

def horner_rule (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def polynomial (x : ℝ) : ℝ :=
  3 * x^4 - x^2 + 2 * x + 1

theorem horner_rule_v3 (x : ℝ) (h : x = 2) :
  let a := [1, 2, 0, -1, 3]
  let v₃ := horner_rule (a.take 4) x
  v₃ = 20 := by sorry

end horner_rule_v3_l4114_411449


namespace largest_integer_from_averages_l4114_411465

theorem largest_integer_from_averages : 
  ∀ w x y z : ℤ,
  (w + x + y) / 3 = 32 →
  (w + x + z) / 3 = 39 →
  (w + y + z) / 3 = 40 →
  (x + y + z) / 3 = 44 →
  max w (max x (max y z)) = 59 := by
  sorry

end largest_integer_from_averages_l4114_411465


namespace janice_purchase_l4114_411438

theorem janice_purchase (x y z : ℕ) : 
  x + y + z = 40 ∧ 
  50 * x + 150 * y + 300 * z = 4500 →
  x = 30 := by
  sorry

end janice_purchase_l4114_411438


namespace x_minus_y_value_l4114_411455

theorem x_minus_y_value (x y : ℝ) (h : |x + y + 1| + Real.sqrt (2 * x - y) = 0) : 
  x - y = 1/3 := by
sorry

end x_minus_y_value_l4114_411455


namespace output_for_15_l4114_411478

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 25 then step1 + 10 else step1 - 7

theorem output_for_15 : function_machine 15 = 38 := by sorry

end output_for_15_l4114_411478


namespace blue_balls_count_l4114_411430

theorem blue_balls_count (B : ℕ) : 
  (5 : ℚ) * 4 / (2 * ((7 + B : ℚ) * (6 + B))) = 0.1282051282051282 → B = 6 := by
  sorry

end blue_balls_count_l4114_411430


namespace factorial_expression_is_perfect_square_l4114_411425

theorem factorial_expression_is_perfect_square (n : ℕ) (h : n ≥ 10) :
  (Nat.factorial (n + 3) - Nat.factorial (n + 2)) / Nat.factorial (n + 1) = (n + 2) ^ 2 := by
  sorry

#check factorial_expression_is_perfect_square

end factorial_expression_is_perfect_square_l4114_411425
