import Mathlib

namespace inequality_system_solution_range_l716_71649

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ x : ℝ, (1 + x > a ∧ 2 * x - 4 ≤ 0)) → a < 3 := by
  sorry

end inequality_system_solution_range_l716_71649


namespace fraction_equation_solution_l716_71653

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 5) / (x - 3) = (x - 4) / (x + 2) ↔ x = 1 / 7 :=
by sorry

end fraction_equation_solution_l716_71653


namespace smartphone_price_decrease_l716_71601

/-- The average percentage decrease in price for a smartphone that underwent two price reductions -/
theorem smartphone_price_decrease (original_price final_price : ℝ) 
  (h1 : original_price = 2000)
  (h2 : final_price = 1280) : 
  (original_price - final_price) / original_price / 2 * 100 = 18 := by
  sorry

end smartphone_price_decrease_l716_71601


namespace square_sum_from_difference_and_product_l716_71670

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = 15) : 
  x^2 + y^2 = 79 := by
  sorry

end square_sum_from_difference_and_product_l716_71670


namespace set_equality_implies_sum_l716_71622

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2016 + b^2017 = 1 := by
sorry

end set_equality_implies_sum_l716_71622


namespace eighth_row_interior_sum_l716_71668

/-- Sum of all elements in the n-th row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem eighth_row_interior_sum :
  pascal_interior_sum 8 = 126 := by sorry

end eighth_row_interior_sum_l716_71668


namespace equation_solution_l716_71689

theorem equation_solution (x : ℝ) : 3 / (x + 10) = 1 / (2 * x) → x = 2 := by
  sorry

end equation_solution_l716_71689


namespace amy_garden_problem_l716_71632

/-- Amy's gardening problem -/
theorem amy_garden_problem (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ)
  (h1 : total_seeds = 101)
  (h2 : small_gardens = 9)
  (h3 : seeds_per_small_garden = 6) :
  total_seeds - (small_gardens * seeds_per_small_garden) = 47 := by
  sorry

end amy_garden_problem_l716_71632


namespace purely_imaginary_iff_a_eq_one_l716_71659

theorem purely_imaginary_iff_a_eq_one (a : ℝ) : 
  (Complex.I * (a * Complex.I) = (a^2 - a) + a * Complex.I) ↔ a = 1 := by sorry

end purely_imaginary_iff_a_eq_one_l716_71659


namespace repeating_decimal_to_fraction_l716_71693

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = 0.56) ∧ x = 56/99 := by
  sorry

end repeating_decimal_to_fraction_l716_71693


namespace f_composition_equals_sqrt2_over_2_minus_1_l716_71660

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 1 else Real.log x / Real.log 3 + 1

theorem f_composition_equals_sqrt2_over_2_minus_1 :
  f (f (Real.sqrt 3 / 9)) = Real.sqrt 2 / 2 - 1 := by sorry

end f_composition_equals_sqrt2_over_2_minus_1_l716_71660


namespace sum_difference_is_60_l716_71663

def sum_even_2_to_120 : ℕ := (Finset.range 60).sum (fun i => 2 * (i + 1))

def sum_odd_1_to_119 : ℕ := (Finset.range 60).sum (fun i => 2 * i + 1)

theorem sum_difference_is_60 : sum_even_2_to_120 - sum_odd_1_to_119 = 60 := by
  sorry

end sum_difference_is_60_l716_71663


namespace proportion_equality_l716_71687

theorem proportion_equality (a b c x : ℝ) (h : a / x = 4 * a * b / (17.5 * c)) :
  x = 17.5 * c / (4 * b) := by
sorry

end proportion_equality_l716_71687


namespace line_properties_l716_71690

-- Define the lines l₁ and l₂
def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := a * (a - 1) - b = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := a / b = 1 - a

-- Define point M
def point_M (a b : ℝ) : Prop := l₁ a b (-3) (-1)

-- Define equal distance from origin to both lines
def equal_distance (b : ℝ) : Prop := 4 / b = b

theorem line_properties (a b : ℝ) :
  (perpendicular a b ∧ point_M a b → a = 2 ∧ b = 2) ∧
  (parallel a b ∧ equal_distance b → (a = 2 ∧ b = -2) ∨ (a = 2/3 ∧ b = 2)) :=
sorry

end line_properties_l716_71690


namespace imaginary_part_of_product_l716_71688

theorem imaginary_part_of_product (i : ℂ) : 
  i * i = -1 →
  Complex.im ((1 + 2*i) * (2 - i)) = 3 := by
sorry

end imaginary_part_of_product_l716_71688


namespace digits_of_8_pow_20_times_5_pow_18_l716_71680

/-- The number of digits in a positive integer n in base b -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log b n + 1

/-- Theorem: The number of digits in 8^20 * 5^18 in base 10 is 31 -/
theorem digits_of_8_pow_20_times_5_pow_18 :
  num_digits (8^20 * 5^18) 10 = 31 := by
  sorry

end digits_of_8_pow_20_times_5_pow_18_l716_71680


namespace probability_at_least_one_boy_l716_71624

/-- The probability of selecting at least one boy from a group of 5 girls and 2 boys,
    given that girl A is already selected and a total of 3 people are to be selected. -/
theorem probability_at_least_one_boy (total_girls : ℕ) (total_boys : ℕ) 
  (h_girls : total_girls = 5) (h_boys : total_boys = 2) (selection_size : ℕ) 
  (h_selection : selection_size = 3) :
  (Nat.choose (total_boys + total_girls - 1) (selection_size - 1) - 
   Nat.choose (total_girls - 1) (selection_size - 1)) / 
  Nat.choose (total_boys + total_girls - 1) (selection_size - 1) = 3 / 5 := by
  sorry

end probability_at_least_one_boy_l716_71624


namespace solution_set_theorem_a_range_theorem_l716_71623

/-- The function f(x) defined as |x| + 2|x-a| where a > 0 -/
def f (a : ℝ) (x : ℝ) : ℝ := |x| + 2 * |x - a|

/-- The solution set of f(x) ≤ 4 when a = 1 -/
def solution_set : Set ℝ := {x : ℝ | x ∈ Set.Icc (-2/3) 2}

/-- The range of a for which f(x) ≥ 4 always holds -/
def a_range : Set ℝ := {a : ℝ | a ∈ Set.Ici 4}

/-- Theorem stating the solution set of f(x) ≤ 4 when a = 1 -/
theorem solution_set_theorem :
  ∀ x : ℝ, f 1 x ≤ 4 ↔ x ∈ solution_set := by sorry

/-- Theorem stating the range of a for which f(x) ≥ 4 always holds -/
theorem a_range_theorem :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4) ↔ a ∈ a_range := by sorry

end solution_set_theorem_a_range_theorem_l716_71623


namespace anniversary_day_theorem_l716_71697

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years in a 300-year span -/
def leapYearsIn300Years : Nat := 73

/-- Calculates the number of regular years in a 300-year span -/
def regularYearsIn300Years : Nat := 300 - leapYearsIn300Years

/-- Calculates the total days to move backward in 300 years -/
def totalDaysBackward : Nat :=
  regularYearsIn300Years + 2 * leapYearsIn300Years

/-- Theorem: If a 300th anniversary falls on a Thursday, the original date was a Tuesday -/
theorem anniversary_day_theorem (anniversaryDay : DayOfWeek) :
  anniversaryDay = DayOfWeek.Thursday →
  (totalDaysBackward % 7 : Nat) = 2 →
  ∃ (originalDay : DayOfWeek), originalDay = DayOfWeek.Tuesday :=
sorry

end anniversary_day_theorem_l716_71697


namespace distance_in_scientific_notation_l716_71699

/-- Given a distance of 14,000,000 meters between two mountain peaks,
    prove that its representation in scientific notation is 1.4 × 10^7 -/
theorem distance_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    14000000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧ 
    a = 1.4 ∧ 
    n = 7 := by
  sorry

end distance_in_scientific_notation_l716_71699


namespace exists_arrangement_for_23_l716_71633

/-- Define a Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 0 = 0 ∧ F 1 = 1 ∧ (∀ n ≥ 2, F n = 3 * F (n - 1) - F (n - 2)) ∧ F 12 % 23 = 0 := by
  sorry

end exists_arrangement_for_23_l716_71633


namespace correct_transformation_l716_71600

theorem correct_transformation (a x y : ℝ) : 
  ax = ay → 3 - ax = 3 - ay := by
sorry

end correct_transformation_l716_71600


namespace range_of_a_l716_71648

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 > 2*x - 1) : a ≠ 1 :=
by
  sorry

end range_of_a_l716_71648


namespace polar_to_cartesian_circle_l716_71640

/-- Given a polar equation ρ = 4sin(θ), prove its equivalence to the Cartesian equation x² + (y-2)² = 4 -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

end polar_to_cartesian_circle_l716_71640


namespace alternating_sum_equals_eight_l716_71673

theorem alternating_sum_equals_eight :
  43 - 41 + 39 - 37 + 35 - 33 + 31 - 29 = 8 := by
  sorry

end alternating_sum_equals_eight_l716_71673


namespace intersection_of_M_and_N_l716_71644

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x ≥ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_M_and_N_l716_71644


namespace sufficient_not_necessary_l716_71664

theorem sufficient_not_necessary :
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) ∧
  (∀ x : ℝ, x > 2 → x^2 > 4) :=
by sorry

end sufficient_not_necessary_l716_71664


namespace triangle_area_l716_71619

/-- The area of a triangle with vertices at (2, 2), (7, 2), and (4, 9) is 17.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (7, 2)
  let C : ℝ × ℝ := (4, 9)
  let base := |B.1 - A.1|
  let height := |C.2 - A.2|
  let area := (1/2) * base * height
  area = 17.5 := by sorry

end triangle_area_l716_71619


namespace quadratic_complete_square_sum_l716_71694

/-- Given a quadratic equation x^2 - 2x + m = 0 that can be written as (x-1)^2 = n
    after completing the square, prove that m + n = 1 -/
theorem quadratic_complete_square_sum (m n : ℝ) : 
  (∀ x, x^2 - 2*x + m = 0 ↔ (x - 1)^2 = n) → m + n = 1 := by
  sorry

end quadratic_complete_square_sum_l716_71694


namespace line_parametric_equations_l716_71684

/-- Parametric equations of a line passing through M(1,5) with inclination angle 2π/3 -/
theorem line_parametric_equations (t : ℝ) : 
  let M : ℝ × ℝ := (1, 5)
  let angle : ℝ := 2 * Real.pi / 3
  let P : ℝ × ℝ := (1 - (1/2) * t, 5 + (Real.sqrt 3 / 2) * t)
  (P.1 - M.1 = t * Real.cos angle) ∧ (P.2 - M.2 = t * Real.sin angle) := by
  sorry

end line_parametric_equations_l716_71684


namespace no_prime_5n_plus_3_l716_71642

theorem no_prime_5n_plus_3 : ¬∃ (n : ℕ+), 
  (∃ k : ℕ, (2 : ℤ) * n + 1 = k^2) ∧ 
  (∃ l : ℕ, (3 : ℤ) * n + 1 = l^2) ∧ 
  Nat.Prime ((5 : ℤ) * n + 3).toNat :=
by sorry

end no_prime_5n_plus_3_l716_71642


namespace truck_distance_l716_71635

theorem truck_distance (truck_time car_time : ℝ) (speed_difference : ℝ) :
  truck_time = 8 →
  car_time = 5 →
  speed_difference = 18 →
  ∃ (truck_speed : ℝ),
    truck_speed * truck_time = (truck_speed + speed_difference) * car_time ∧
    truck_speed * truck_time = 240 :=
by sorry

end truck_distance_l716_71635


namespace congruence_square_implies_congruence_or_negative_l716_71682

theorem congruence_square_implies_congruence_or_negative (x y : ℤ) :
  x^2 ≡ y^2 [ZMOD 239] → (x ≡ y [ZMOD 239] ∨ x ≡ -y [ZMOD 239]) := by
  sorry

end congruence_square_implies_congruence_or_negative_l716_71682


namespace cylinder_ellipse_intersection_l716_71651

/-- Given a right circular cylinder with radius 2 and a plane intersecting it to form an ellipse,
    if the major axis of the ellipse is 25% longer than its minor axis,
    then the length of the major axis is 5. -/
theorem cylinder_ellipse_intersection (cylinder_radius : ℝ) (minor_axis major_axis : ℝ) :
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = 1.25 * minor_axis →
  major_axis = 5 := by
  sorry

end cylinder_ellipse_intersection_l716_71651


namespace five_balls_three_boxes_l716_71638

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 66 ways to put 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : ways_to_distribute 5 3 = 66 := by
  sorry

end five_balls_three_boxes_l716_71638


namespace unique_solution_condition_l716_71634

theorem unique_solution_condition (a b : ℤ) : 
  (∃! (x y z : ℤ), x + y = a - 1 ∧ x * (y + 1) - z^2 = b) ↔ 4 * b = a^2 := by
sorry

end unique_solution_condition_l716_71634


namespace least_number_with_remainder_least_number_is_205517_least_number_unique_l716_71627

theorem least_number_with_remainder (n : ℕ) : 
  (n % 45 = 2 ∧ n % 59 = 2 ∧ n % 77 = 2) → n ≥ 205517 :=
by
  sorry

theorem least_number_is_205517 : 
  205517 % 45 = 2 ∧ 205517 % 59 = 2 ∧ 205517 % 77 = 2 :=
by
  sorry

theorem least_number_unique : 
  ∃! n : ℕ, (n % 45 = 2 ∧ n % 59 = 2 ∧ n % 77 = 2) ∧ 
  ∀ m : ℕ, (m % 45 = 2 ∧ m % 59 = 2 ∧ m % 77 = 2) → n ≤ m :=
by
  sorry

end least_number_with_remainder_least_number_is_205517_least_number_unique_l716_71627


namespace race_outcomes_l716_71647

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_positions : ℕ := 3

/-- The number of different podium outcomes in a race with no ties -/
def num_outcomes : ℕ := num_participants * (num_participants - 1) * (num_participants - 2)

theorem race_outcomes :
  num_outcomes = 120 :=
by sorry

end race_outcomes_l716_71647


namespace distance_between_towns_l716_71626

/-- The distance between two towns given train speeds and meeting time -/
theorem distance_between_towns (express_speed : ℝ) (speed_difference : ℝ) (meeting_time : ℝ) : 
  express_speed = 80 →
  speed_difference = 30 →
  meeting_time = 3 →
  (express_speed + (express_speed - speed_difference)) * meeting_time = 390 := by
sorry

end distance_between_towns_l716_71626


namespace scientific_notation_equivalence_l716_71612

def kilowatt_hours : ℝ := 448000

theorem scientific_notation_equivalence : 
  kilowatt_hours = 4.48 * (10 : ℝ)^5 := by sorry

end scientific_notation_equivalence_l716_71612


namespace sufficient_not_necessary_l716_71646

/-- The line l is defined by the equation x + y - 1 = 0 --/
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

/-- A point P lies on line l if its coordinates satisfy the line equation --/
def point_on_line_l (x y : ℝ) : Prop := line_l x y

/-- The specific condition we're examining --/
def specific_condition (x y : ℝ) : Prop := x = 2 ∧ y = -1

theorem sufficient_not_necessary :
  (∀ x y : ℝ, specific_condition x y → point_on_line_l x y) ∧
  ¬(∀ x y : ℝ, point_on_line_l x y → specific_condition x y) :=
sorry

end sufficient_not_necessary_l716_71646


namespace sum_bc_equals_nine_l716_71681

theorem sum_bc_equals_nine 
  (h1 : a + b = 16) 
  (h2 : c + d = 3) 
  (h3 : a + d = 10) : 
  b + c = 9 := by
sorry

end sum_bc_equals_nine_l716_71681


namespace parabola_focus_coordinates_focus_coordinates_y_eq_2x_squared_l716_71641

/-- The focus of a parabola y = ax^2 has coordinates (0, 1/(4a)) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - a * x^2
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1 / (4 * a) ∧ 
    ∀ (x y : ℝ), f (x, y) = 0 → (x - p.1)^2 + (y - p.2)^2 = (y - p.2 + 1 / (4 * a))^2 :=
sorry

/-- The focus of the parabola y = 2x^2 has coordinates (0, 1/8) -/
theorem focus_coordinates_y_eq_2x_squared :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - 2 * x^2
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1/8 ∧ 
    ∀ (x y : ℝ), f (x, y) = 0 → (x - p.1)^2 + (y - p.2)^2 = (y - p.2 + 1/8)^2 :=
sorry

end parabola_focus_coordinates_focus_coordinates_y_eq_2x_squared_l716_71641


namespace janet_hires_four_warehouse_workers_l716_71617

/-- Represents the employment scenario for Janet's company --/
structure EmploymentScenario where
  total_employees : ℕ
  managers : ℕ
  warehouse_wage : ℝ
  manager_wage : ℝ
  fica_tax_rate : ℝ
  work_days : ℕ
  work_hours : ℕ
  total_cost : ℝ

/-- Calculates the number of warehouse workers in Janet's company --/
def calculate_warehouse_workers (scenario : EmploymentScenario) : ℕ :=
  scenario.total_employees - scenario.managers

/-- Theorem stating that Janet hires 4 warehouse workers --/
theorem janet_hires_four_warehouse_workers :
  let scenario : EmploymentScenario := {
    total_employees := 6,
    managers := 2,
    warehouse_wage := 15,
    manager_wage := 20,
    fica_tax_rate := 0.1,
    work_days := 25,
    work_hours := 8,
    total_cost := 22000
  }
  calculate_warehouse_workers scenario = 4 := by
  sorry


end janet_hires_four_warehouse_workers_l716_71617


namespace prime_sum_2003_l716_71675

theorem prime_sum_2003 (a b : ℕ) (ha : Prime a) (hb : Prime b) (h : a^2 + b = 2003) : 
  a + b = 2001 := by sorry

end prime_sum_2003_l716_71675


namespace total_payment_l716_71662

def payment_structure (year1 year2 year3 year4 : ℕ) : Prop :=
  year1 = 20 ∧ 
  year2 = year1 + 2 ∧ 
  year3 = year2 + 3 ∧ 
  year4 = year3 + 4

theorem total_payment (year1 year2 year3 year4 : ℕ) :
  payment_structure year1 year2 year3 year4 →
  year1 + year2 + year3 + year4 = 96 := by
  sorry

end total_payment_l716_71662


namespace election_votes_theorem_l716_71669

theorem election_votes_theorem (total_votes : ℕ) : 
  (75 : ℝ) / 100 * ((100 : ℝ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 := by
  sorry

end election_votes_theorem_l716_71669


namespace solution_system_trigonometric_equations_l716_71671

theorem solution_system_trigonometric_equations :
  ∀ x y : ℝ,
  (Real.sin x)^2 = Real.sin y ∧ (Real.cos x)^4 = Real.cos y →
  (∃ l m : ℤ, x = l * Real.pi ∧ y = 2 * m * Real.pi) ∨
  (∃ l m : ℤ, x = l * Real.pi + Real.pi / 2 ∧ y = 2 * m * Real.pi + Real.pi / 2) :=
by sorry

end solution_system_trigonometric_equations_l716_71671


namespace cos_sin_inequality_solution_set_l716_71615

open Real

theorem cos_sin_inequality_solution_set (x : ℝ) : 
  (cos x)^4 - 2 * sin x * cos x - (sin x)^4 - 1 > 0 ↔ 
  ∃ k : ℤ, x ∈ Set.Ioo (k * π - π/4) (k * π) := by sorry

end cos_sin_inequality_solution_set_l716_71615


namespace initial_peanuts_l716_71611

theorem initial_peanuts (added : ℕ) (final : ℕ) (h1 : added = 6) (h2 : final = 10) :
  final - added = 4 := by
  sorry

end initial_peanuts_l716_71611


namespace chetan_score_percentage_l716_71658

theorem chetan_score_percentage (max_score : ℕ) (amar_percent : ℚ) (bhavan_percent : ℚ) (average_mark : ℕ) :
  max_score = 900 →
  amar_percent = 64/100 →
  bhavan_percent = 36/100 →
  average_mark = 432 →
  ∃ (chetan_percent : ℚ), 
    (amar_percent + bhavan_percent + chetan_percent) * max_score / 3 = average_mark ∧
    chetan_percent = 44/100 :=
by sorry

end chetan_score_percentage_l716_71658


namespace probability_of_specific_arrangement_l716_71679

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'O', 'X', 'O', 'X', 'O']

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 
  (1 : ℚ) / 35 :=
sorry

end probability_of_specific_arrangement_l716_71679


namespace perpendicular_m_value_parallel_distance_l716_71674

-- Define the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + (m - 3) * y + m = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := m * x - 2 * y + 4 = 0

-- Define perpendicularity condition
def perpendicular (m : ℝ) : Prop := (-1 : ℝ) / (m - 3) * (m / 2) = -1

-- Define parallelism condition
def parallel (m : ℝ) : Prop := 1 * (-2) = m * (m - 3)

-- Theorem for perpendicular case
theorem perpendicular_m_value : ∃ m : ℝ, perpendicular m ∧ m = 6 := by sorry

-- Theorem for parallel case
theorem parallel_distance : 
  ∃ m : ℝ, parallel m ∧ 
  (let d := |4 - 1| / Real.sqrt (1^2 + (-2)^2);
   d = 3 * Real.sqrt 5 / 5) := by sorry

end perpendicular_m_value_parallel_distance_l716_71674


namespace min_distance_for_ten_trees_l716_71610

/-- Calculates the minimum distance to water trees in a row -/
def min_watering_distance (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  let well_to_tree := tree_distance
  let tree_to_well := tree_distance
  let full_trips := (num_trees - 1) / 2
  let full_trip_distance := full_trips * (well_to_tree + tree_to_well)
  let remaining_trees := (num_trees - 1) % 2
  let last_trip_distance := remaining_trees * (well_to_tree + tree_to_well)
  full_trip_distance + last_trip_distance + (num_trees - 1) * tree_distance

theorem min_distance_for_ten_trees :
  min_watering_distance 10 10 = 410 :=
by sorry

end min_distance_for_ten_trees_l716_71610


namespace tom_apple_purchase_l716_71683

/-- The problem of determining how many kg of apples Tom purchased -/
theorem tom_apple_purchase (apple_price mango_price : ℕ) (mango_amount total_paid : ℕ) 
  (h1 : apple_price = 70)
  (h2 : mango_price = 70)
  (h3 : mango_amount = 9)
  (h4 : total_paid = 1190) :
  ∃ (apple_amount : ℕ), apple_amount * apple_price + mango_amount * mango_price = total_paid ∧ apple_amount = 8 := by
  sorry

end tom_apple_purchase_l716_71683


namespace cubic_function_extrema_l716_71696

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem cubic_function_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ (y : ℝ), f a y ≤ f a x₁) ∧ 
    (∀ (y : ℝ), f a y ≥ f a x₂)) →
  (a < -3 ∨ a > 6) :=
by sorry

end cubic_function_extrema_l716_71696


namespace tan_45_degrees_l716_71652

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l716_71652


namespace binomial_16_choose_5_l716_71639

theorem binomial_16_choose_5 : Nat.choose 16 5 = 4368 := by
  sorry

end binomial_16_choose_5_l716_71639


namespace hulk_jump_exceeds_1000_l716_71650

def hulk_jump (n : ℕ) : ℝ := 3^n

theorem hulk_jump_exceeds_1000 : 
  (∀ k < 7, hulk_jump k ≤ 1000) ∧ hulk_jump 7 > 1000 := by
  sorry

end hulk_jump_exceeds_1000_l716_71650


namespace value_of_y_l716_71698

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end value_of_y_l716_71698


namespace hawks_win_rate_theorem_l716_71618

/-- The minimum number of additional games needed for the Hawks to reach 90% win rate -/
def min_additional_games : ℕ := 25

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The number of games initially won by the Hawks -/
def initial_hawks_wins : ℕ := 2

/-- The target win percentage as a fraction -/
def target_win_rate : ℚ := 9/10

theorem hawks_win_rate_theorem :
  ∀ n : ℕ, 
    (initial_hawks_wins + n : ℚ) / (initial_games + n) ≥ target_win_rate ↔ 
    n ≥ min_additional_games := by
  sorry

end hawks_win_rate_theorem_l716_71618


namespace triangle_angle_measure_l716_71628

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 74 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 17.6 := by
  sorry

end triangle_angle_measure_l716_71628


namespace sqrt_450_simplification_l716_71630

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l716_71630


namespace b_work_time_l716_71620

-- Define the work completion time for A
def a_time : ℝ := 6

-- Define the total payment for A and B
def total_payment : ℝ := 3200

-- Define the time taken with C's help
def time_with_c : ℝ := 3

-- Define C's payment
def c_payment : ℝ := 400.0000000000002

-- Define B's work completion time (to be proved)
def b_time : ℝ := 8

-- Theorem statement
theorem b_work_time : 
  1 / a_time + 1 / b_time + (c_payment / total_payment) * (1 / time_with_c) = 1 / time_with_c :=
sorry

end b_work_time_l716_71620


namespace max_distance_complex_l716_71657

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z - 1) = 1) :
  ∃ (max_val : ℝ), max_val = 3 ∧ ∀ w, Complex.abs (w - 1) = 1 → Complex.abs (w - (2 * Complex.I + 1)) ≤ max_val :=
by sorry

end max_distance_complex_l716_71657


namespace geometric_progression_product_l716_71605

/-- For a geometric progression with n terms, first term a, and common ratio r,
    where P is the product of the n terms and T is the sum of the squares of the terms,
    the following equation holds. -/
theorem geometric_progression_product (n : ℕ) (a r : ℝ) (P T : ℝ) 
    (h1 : P = a^n * r^(n * (n - 1) / 2))
    (h2 : T = a^2 * (1 - r^(2*n)) / (1 - r^2)) 
    (h3 : r ≠ 1) : 
  P = T^(n/2) * ((1 - r^2) / (1 - r^(2*n)))^(n/2) * r^(n*(n-1)/2) := by
  sorry

end geometric_progression_product_l716_71605


namespace fraction_subtraction_l716_71667

theorem fraction_subtraction (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3 * x^2 - 2 * x + 1) / ((x + 2) * (x - 3)) - (x^2 - 5 * x + 6) / ((x + 2) * (x - 3)) =
  (2 * x^2 + 3 * x - 5) / ((x + 2) * (x - 3)) := by
  sorry

end fraction_subtraction_l716_71667


namespace prime_relation_l716_71616

theorem prime_relation (P Q : ℕ) (hP : Nat.Prime P) (hQ : Nat.Prime Q)
  (h1 : P ∣ (Q^3 - 1)) (h2 : Q ∣ (P - 1)) : P = 1 + Q + Q^2 := by
  sorry

end prime_relation_l716_71616


namespace hash_difference_l716_71677

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 8 5 - hash 5 8 = -12 := by
  sorry

end hash_difference_l716_71677


namespace curve_transformation_l716_71607

/-- The matrix A --/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; 0, 1]

/-- The original curve C --/
def C (x y : ℝ) : Prop := (x - y)^2 + y^2 = 1

/-- The transformed curve C' --/
def C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Theorem stating that C' is the result of transforming C under A --/
theorem curve_transformation (x y : ℝ) : 
  C' x y ↔ ∃ x₀ y₀ : ℝ, C x₀ y₀ ∧ A.mulVec ![x₀, y₀] = ![x, y] :=
sorry

end curve_transformation_l716_71607


namespace paper_tearing_impossibility_l716_71686

theorem paper_tearing_impossibility : ∀ n : ℕ, 
  n % 3 = 2 → 
  ¬ (∃ (sequence : ℕ → ℕ), 
    sequence 0 = 1 ∧ 
    (∀ i : ℕ, sequence (i + 1) = sequence i + 3 ∨ sequence (i + 1) = sequence i + 9) ∧
    (∃ k : ℕ, sequence k = n)) :=
by sorry

end paper_tearing_impossibility_l716_71686


namespace base_7_to_decimal_l716_71637

/-- Converts a list of digits in base b to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The base 7 number 3206 -/
def base_7_number : List Nat := [3, 2, 0, 6]

/-- Theorem stating that 3206 in base 7 is equal to 1133 in base 10 -/
theorem base_7_to_decimal :
  to_decimal base_7_number 7 = 1133 := by
  sorry

end base_7_to_decimal_l716_71637


namespace complex_dot_product_l716_71636

theorem complex_dot_product (z : ℂ) (h1 : Complex.abs z = Real.sqrt 2) (h2 : Complex.im (z^2) = 2) :
  (z + z^2) • (z - z^2) = -2 ∨ (z + z^2) • (z - z^2) = 8 := by
  sorry

end complex_dot_product_l716_71636


namespace second_project_questions_l716_71678

/-- Calculates the number of questions for the second project given the total questions per day,
    number of days, and questions for the first project. -/
def questions_for_second_project (questions_per_day : ℕ) (days : ℕ) (questions_first_project : ℕ) : ℕ :=
  questions_per_day * days - questions_first_project

/-- Proves that given the specified conditions, the number of questions for the second project is 476. -/
theorem second_project_questions :
  questions_for_second_project 142 7 518 = 476 := by
  sorry

end second_project_questions_l716_71678


namespace unknown_number_in_set_l716_71655

theorem unknown_number_in_set (x : ℝ) : 
  let set1 : List ℝ := [12, 32, 56, 78, 91]
  let set2 : List ℝ := [7, 47, 67, 105, x]
  (set1.sum / set1.length : ℝ) = (set2.sum / set2.length : ℝ) + 10 →
  x = -7 := by
sorry

end unknown_number_in_set_l716_71655


namespace activity_popularity_ranking_l716_71665

/-- Represents the popularity of an activity as a fraction --/
structure ActivityPopularity where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- The three activities in the festival --/
inductive Activity
  | Dance
  | Painting
  | ClayModeling

/-- Given popularity data for the activities --/
def popularity : Activity → ActivityPopularity
  | Activity.Dance => ⟨3, 8, by norm_num⟩
  | Activity.Painting => ⟨5, 16, by norm_num⟩
  | Activity.ClayModeling => ⟨9, 24, by norm_num⟩

/-- Convert a fraction to a common denominator --/
def toCommonDenominator (ap : ActivityPopularity) (lcd : ℕ) : ℚ :=
  (ap.numerator : ℚ) * (lcd / ap.denominator) / lcd

/-- The least common denominator of all activities' fractions --/
def leastCommonDenominator : ℕ := 48

theorem activity_popularity_ranking :
  let commonDance := toCommonDenominator (popularity Activity.Dance) leastCommonDenominator
  let commonPainting := toCommonDenominator (popularity Activity.Painting) leastCommonDenominator
  let commonClayModeling := toCommonDenominator (popularity Activity.ClayModeling) leastCommonDenominator
  (commonDance = commonClayModeling) ∧ (commonDance > commonPainting) := by
  sorry

#check activity_popularity_ranking

end activity_popularity_ranking_l716_71665


namespace problem_one_problem_two_l716_71625

-- Problem 1
theorem problem_one : 
  Real.rpow 0.064 (-1/3) - Real.rpow (-1/8) 0 + Real.rpow 16 (3/4) + Real.rpow 0.25 (1/2) = 10 := by
  sorry

-- Problem 2
theorem problem_two :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 5/4 := by
  sorry

end problem_one_problem_two_l716_71625


namespace twenty_percent_of_number_is_fifty_l716_71602

theorem twenty_percent_of_number_is_fifty (x : ℝ) : (20 / 100) * x = 50 → x = 250 := by
  sorry

end twenty_percent_of_number_is_fifty_l716_71602


namespace oranges_distribution_l716_71654

theorem oranges_distribution (total : ℕ) (boxes : ℕ) (difference : ℕ) (first_box : ℕ) : 
  total = 120 →
  boxes = 7 →
  difference = 2 →
  (first_box * boxes + (boxes * (boxes - 1) * difference) / 2 = total) →
  first_box = 11 := by
sorry

end oranges_distribution_l716_71654


namespace square_difference_sum_l716_71676

theorem square_difference_sum : 
  20^2 - 18^2 + 16^2 - 14^2 + 12^2 - 10^2 + 8^2 - 6^2 + 4^2 - 2^2 = 220 := by
  sorry

end square_difference_sum_l716_71676


namespace fraction_comparison_l716_71656

theorem fraction_comparison : ((3 / 5 : ℚ) * 320 + (5 / 9 : ℚ) * 540) - ((7 / 12 : ℚ) * 450) = 229.5 := by
  sorry

end fraction_comparison_l716_71656


namespace total_marbles_l716_71631

theorem total_marbles (dohyun_pockets : Nat) (dohyun_per_pocket : Nat)
                      (joohyun_bags : Nat) (joohyun_per_bag : Nat) :
  dohyun_pockets = 7 →
  dohyun_per_pocket = 16 →
  joohyun_bags = 6 →
  joohyun_per_bag = 25 →
  dohyun_pockets * dohyun_per_pocket + joohyun_bags * joohyun_per_bag = 262 := by
  sorry

end total_marbles_l716_71631


namespace expression_equals_one_l716_71614

theorem expression_equals_one (a b c : ℝ) :
  ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2) / ((a - b)^2 + (b - c)^2 + (c - a)^2) = 1 := by
  sorry

end expression_equals_one_l716_71614


namespace inequality_and_equality_conditions_l716_71604

theorem inequality_and_equality_conditions (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 2) :
  3*x + 8*x*y + 16*x*y*z ≤ 12 ∧ 
  (3*x + 8*x*y + 16*x*y*z = 12 ↔ x = 1 ∧ y = 3/4 ∧ z = 1/4) :=
by sorry

end inequality_and_equality_conditions_l716_71604


namespace count_five_ruble_coins_l716_71695

theorem count_five_ruble_coins 
  (total_coins : ℕ) 
  (not_two_ruble : ℕ) 
  (not_ten_ruble : ℕ) 
  (not_one_ruble : ℕ) 
  (h1 : total_coins = 25)
  (h2 : not_two_ruble = 19)
  (h3 : not_ten_ruble = 20)
  (h4 : not_one_ruble = 16) :
  total_coins - ((total_coins - not_two_ruble) + (total_coins - not_ten_ruble) + (total_coins - not_one_ruble)) = 5 := by
sorry

end count_five_ruble_coins_l716_71695


namespace parallel_line_equation_l716_71613

/-- Given a line with slope 2/3 and y-intercept 5, 
    prove that a parallel line 5 units away has the equation 
    y = (2/3)x + (5 ± (5√13)/3) -/
theorem parallel_line_equation (x y : ℝ) : 
  let given_line := λ x : ℝ => (2/3) * x + 5
  let distance := 5
  let parallel_line := λ x : ℝ => (2/3) * x + c
  let c_diff := |c - 5|
  (∀ x, |parallel_line x - given_line x| = distance) →
  (c = 5 + (5 * Real.sqrt 13) / 3 ∨ c = 5 - (5 * Real.sqrt 13) / 3) :=
by sorry

end parallel_line_equation_l716_71613


namespace soris_population_2080_l716_71685

/-- The population growth function for Soris island -/
def soris_population (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (2 ^ (years_passed / 20))

/-- Theorem stating the population of Soris in 2080 -/
theorem soris_population_2080 :
  soris_population 500 80 = 8000 := by
  sorry

end soris_population_2080_l716_71685


namespace jellybean_count_l716_71645

theorem jellybean_count (nephews nieces jellybeans_per_child : ℕ) 
  (h1 : nephews = 3)
  (h2 : nieces = 2)
  (h3 : jellybeans_per_child = 14) :
  (nephews + nieces) * jellybeans_per_child = 70 := by
  sorry

end jellybean_count_l716_71645


namespace exists_multiple_of_E_l716_71606

def E (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * (i + 1))

def D (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * i + 1)

theorem exists_multiple_of_E (n : ℕ) : ∃ m : ℕ, ∃ k : ℕ, D n * 2^m = k * E n := by
  sorry

end exists_multiple_of_E_l716_71606


namespace total_spent_is_684_l716_71621

/-- Calculates the total amount spent by Christy and Tanya on face moisturizers and body lotions with discounts applied. -/
def total_spent (face_moisturizer_price : ℝ) (body_lotion_price : ℝ)
                (tanya_face : ℕ) (tanya_body : ℕ)
                (christy_face : ℕ) (christy_body : ℕ)
                (face_discount : ℝ) (body_discount : ℝ) : ℝ :=
  let tanya_total := (1 - face_discount) * (face_moisturizer_price * tanya_face) +
                     (1 - body_discount) * (body_lotion_price * tanya_body)
  let christy_total := (1 - face_discount) * (face_moisturizer_price * christy_face) +
                       (1 - body_discount) * (body_lotion_price * christy_body)
  tanya_total + christy_total

/-- Theorem stating that the total amount spent by Christy and Tanya is $684 under the given conditions. -/
theorem total_spent_is_684 :
  total_spent 50 60 2 4 3 5 0.1 0.15 = 684 ∧
  total_spent 50 60 2 4 3 5 0.1 0.15 = 2 * total_spent 50 60 2 4 2 4 0.1 0.15 :=
by sorry


end total_spent_is_684_l716_71621


namespace square_root_sum_implies_product_l716_71629

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) := by
sorry

end square_root_sum_implies_product_l716_71629


namespace correct_answer_calculation_l716_71661

theorem correct_answer_calculation (x y : ℝ) : 
  (y = x + 2 * 0.42) → (x = y - 2 * 0.42) :=
by
  sorry

#eval (0.9 : ℝ) - 2 * 0.42

end correct_answer_calculation_l716_71661


namespace hyperbola_ellipse_foci_parabola_equation_l716_71692

-- Define the hyperbola and ellipse
def hyperbola (m : ℝ) := {(x, y) : ℝ × ℝ | x^2 - y^2 = m}
def ellipse := {(x, y) : ℝ × ℝ | 2*x^2 + 3*y^2 = 72}

-- Define the condition of same foci
def same_foci (h : Set (ℝ × ℝ)) (e : Set (ℝ × ℝ)) : Prop := sorry

-- Define a parabola
def parabola (p : ℝ) := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}

-- Define the condition for focus on positive x-axis
def focus_on_positive_x (p : Set (ℝ × ℝ)) : Prop := sorry

-- Define the condition for passing through a point
def passes_through (p : Set (ℝ × ℝ)) (point : ℝ × ℝ) : Prop := point ∈ p

-- Theorem 1
theorem hyperbola_ellipse_foci (m : ℝ) : 
  same_foci (hyperbola m) ellipse → m = 6 := sorry

-- Theorem 2
theorem parabola_equation : 
  ∃ p : ℝ, focus_on_positive_x (parabola p) ∧ 
  passes_through (parabola p) (2, -4) ∧ 
  p = 4 := sorry

end hyperbola_ellipse_foci_parabola_equation_l716_71692


namespace distinct_triangles_count_l716_71643

/-- Given a triangle ABC with n1 points on side AB (excluding A and B),
    n2 points on side BC (excluding B and C), and n3 points on side AC (excluding A and C),
    the number of distinct triangles formed by choosing one point from each side
    is equal to n1 * n2 * n3. -/
theorem distinct_triangles_count (n1 n2 n3 : ℕ) : ℕ :=
  n1 * n2 * n3

#check distinct_triangles_count

end distinct_triangles_count_l716_71643


namespace water_speed_calculation_l716_71603

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  still_water_speed = 12 →
  distance = 8 →
  time = 4 →
  ∃ water_speed : ℝ, water_speed = still_water_speed - (distance / time) :=
by
  sorry

end water_speed_calculation_l716_71603


namespace goldfish_remaining_l716_71691

/-- Given Finn's initial number of goldfish and the number that die,
    prove the number of goldfish left. -/
theorem goldfish_remaining (initial : ℕ) (died : ℕ) :
  initial ≥ died →
  initial - died = initial - died :=
by sorry

end goldfish_remaining_l716_71691


namespace factorial_solutions_l716_71672

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_solutions :
  ∀ x y : ℕ, factorial x + 2^y = factorial (x + 1) ↔ (x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 1) := by
  sorry

end factorial_solutions_l716_71672


namespace trains_meeting_time_l716_71608

/-- Two trains meeting problem -/
theorem trains_meeting_time (distance : ℝ) (express_speed : ℝ) (speed_difference : ℝ) : 
  distance = 390 →
  express_speed = 80 →
  speed_difference = 30 →
  (distance / (express_speed + (express_speed - speed_difference))) = 3 := by
  sorry

end trains_meeting_time_l716_71608


namespace junior_percentage_is_22_l716_71609

/-- Represents the number of students in each grade --/
structure StudentCount where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The total number of students in the sample --/
def totalStudents : ℕ := 800

/-- The conditions of the problem --/
def sampleConditions (s : StudentCount) : Prop :=
  s.freshmen + s.sophomores + s.juniors + s.seniors = totalStudents ∧
  s.sophomores = totalStudents / 4 ∧
  s.seniors = 160 ∧
  s.freshmen = s.sophomores + 64

/-- The percentage of juniors in the sample --/
def juniorPercentage (s : StudentCount) : ℚ :=
  s.juniors * 100 / totalStudents

/-- Theorem stating that the percentage of juniors is 22% --/
theorem junior_percentage_is_22 (s : StudentCount) 
  (h : sampleConditions s) : juniorPercentage s = 22 := by
  sorry

end junior_percentage_is_22_l716_71609


namespace john_business_venture_result_l716_71666

structure Currency where
  name : String
  exchange_rate : ℚ

structure Item where
  name : String
  currency : Currency
  purchase_price : ℚ
  sale_percentage : ℚ
  tax_rate : ℚ

def calculate_profit_or_loss (items : List Item) : ℚ :=
  sorry

theorem john_business_venture_result 
  (grinder : Item)
  (mobile_phone : Item)
  (refrigerator : Item)
  (television : Item)
  (h_grinder : grinder = { 
    name := "Grinder", 
    currency := { name := "INR", exchange_rate := 1 },
    purchase_price := 15000,
    sale_percentage := -4/100,
    tax_rate := 5/100
  })
  (h_mobile_phone : mobile_phone = {
    name := "Mobile Phone",
    currency := { name := "USD", exchange_rate := 75 },
    purchase_price := 100,
    sale_percentage := 10/100,
    tax_rate := 7/100
  })
  (h_refrigerator : refrigerator = {
    name := "Refrigerator",
    currency := { name := "GBP", exchange_rate := 101 },
    purchase_price := 200,
    sale_percentage := 8/100,
    tax_rate := 6/100
  })
  (h_television : television = {
    name := "Television",
    currency := { name := "EUR", exchange_rate := 90 },
    purchase_price := 300,
    sale_percentage := -6/100,
    tax_rate := 9/100
  }) :
  calculate_profit_or_loss [grinder, mobile_phone, refrigerator, television] = -346/100 :=
sorry

end john_business_venture_result_l716_71666
