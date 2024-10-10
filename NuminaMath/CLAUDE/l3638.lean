import Mathlib

namespace equation_solution_l3638_363879

theorem equation_solution (x y : ℝ) : 3 * x - 4 * y = 5 → x = (1/3) * (5 + 4 * y) := by
  sorry

end equation_solution_l3638_363879


namespace smallest_integer_l3638_363863

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 24) :
  b ≥ 360 :=
sorry

end smallest_integer_l3638_363863


namespace area_of_quadrilateral_AFCH_l3638_363805

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the cross-shaped figure formed by two intersecting rectangles -/
structure CrossFigure where
  rect1 : Rectangle
  rect2 : Rectangle

/-- Theorem: Area of quadrilateral AFCH in the cross-shaped figure -/
theorem area_of_quadrilateral_AFCH (cf : CrossFigure)
  (h1 : cf.rect1.width = 9)
  (h2 : cf.rect1.height = 5)
  (h3 : cf.rect2.width = 3)
  (h4 : cf.rect2.height = 10) :
  area (Rectangle.mk 9 10) - (area cf.rect1 + area cf.rect2 - area (Rectangle.mk 3 5)) / 2 = 52.5 := by
  sorry

end area_of_quadrilateral_AFCH_l3638_363805


namespace function_properties_l3638_363816

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) (f'' : ℝ → ℝ) 
  (h_even : is_even f)
  (h_deriv : ∀ x, HasDerivAt f (f'' x) x)
  (h_eq : ∀ x, f (x - 1/2) + f (x + 1) = 0)
  (h_val : Real.exp 3 * f 2018 = 1)
  (h_ineq : ∀ x, f x > f'' (-x)) :
  {x : ℝ | f (x - 1) > 1 / Real.exp x} = {x : ℝ | x > 3} := by
  sorry

end function_properties_l3638_363816


namespace product_of_numbers_l3638_363818

theorem product_of_numbers (x y : ℝ) : 
  x + y = 25 → x - y = 7 → x * y = 144 := by sorry

end product_of_numbers_l3638_363818


namespace unique_solution_linear_system_l3638_363860

theorem unique_solution_linear_system :
  ∃! (x y : ℝ), (2 * x - y = 1) ∧ (x + y = 2) :=
by
  -- The proof would go here
  sorry

end unique_solution_linear_system_l3638_363860


namespace sin_15_cos_15_eq_quarter_l3638_363885

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end sin_15_cos_15_eq_quarter_l3638_363885


namespace red_hair_ratio_l3638_363839

theorem red_hair_ratio (red_hair_count : ℕ) (total_count : ℕ) 
  (h1 : red_hair_count = 9) 
  (h2 : total_count = 48) : 
  (red_hair_count : ℚ) / total_count = 3 / 16 := by
  sorry

end red_hair_ratio_l3638_363839


namespace complex_modulus_sum_difference_l3638_363876

theorem complex_modulus_sum_difference :
  let z₁ : ℂ := 3 - 5*I
  let z₂ : ℂ := 3 + 5*I
  let z₃ : ℂ := -2 + 6*I
  Complex.abs z₁ + Complex.abs z₂ - Real.sqrt (Complex.abs z₃) = 2 * Real.sqrt 34 - Real.sqrt (2 * Real.sqrt 10) :=
by sorry

end complex_modulus_sum_difference_l3638_363876


namespace arithmetic_sequence_problem_l3638_363849

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₆ = 6 and a₉ = 9, prove that a₃ = 3 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a6 : a 6 = 6) 
  (h_a9 : a 9 = 9) : 
  a 3 = 3 := by
  sorry

end arithmetic_sequence_problem_l3638_363849


namespace paperclip_capacity_l3638_363890

theorem paperclip_capacity (small_volume small_capacity large_volume efficiency : ℝ) 
  (h1 : small_volume = 12)
  (h2 : small_capacity = 40)
  (h3 : large_volume = 60)
  (h4 : efficiency = 0.8)
  : (large_volume * efficiency * small_capacity) / small_volume = 160 := by
  sorry

end paperclip_capacity_l3638_363890


namespace quadrilateral_angle_combinations_l3638_363861

/-- Represents the type of an angle in a quadrilateral -/
inductive AngleType
| Acute
| Right
| Obtuse

/-- Represents a combination of angles in a quadrilateral -/
structure AngleCombination :=
  (acute : Nat)
  (right : Nat)
  (obtuse : Nat)

/-- A convex quadrilateral has exactly four angles -/
def total_angles : Nat := 4

/-- The sum of interior angles in a quadrilateral is 360 degrees -/
def angle_sum : Nat := 360

/-- Theorem: The only possible combinations of internal angles in a convex quadrilateral
    are the seven combinations listed. -/
theorem quadrilateral_angle_combinations :
  ∃ (valid_combinations : List AngleCombination),
    (valid_combinations.length = 7) ∧
    (∀ combo : AngleCombination,
      (combo.acute + combo.right + combo.obtuse = total_angles) →
      (combo.right * 90 + combo.acute * 89 + combo.obtuse * 91 ≤ angle_sum) →
      (combo.right * 90 + combo.acute * 1 + combo.obtuse * 91 ≥ angle_sum) →
      (combo ∈ valid_combinations)) ∧
    (∀ combo : AngleCombination,
      combo ∈ valid_combinations →
      (combo.acute + combo.right + combo.obtuse = total_angles) ∧
      (combo.right * 90 + combo.acute * 89 + combo.obtuse * 91 ≤ angle_sum) ∧
      (combo.right * 90 + combo.acute * 1 + combo.obtuse * 91 ≥ angle_sum)) :=
sorry

end quadrilateral_angle_combinations_l3638_363861


namespace inscribed_square_area_l3638_363870

/-- A square inscribed in a right triangle -/
structure InscribedSquare where
  /-- The side length of the inscribed square -/
  side : ℝ
  /-- The distance from one vertex of the right triangle to where the square touches the hypotenuse -/
  dist1 : ℝ
  /-- The distance from the other vertex of the right triangle to where the square touches the hypotenuse -/
  dist2 : ℝ
  /-- The constraint that the square is properly inscribed in the right triangle -/
  inscribed : side * side = dist1 * dist2

/-- The theorem stating that a square inscribed in a right triangle with specific measurements has an area of 975 -/
theorem inscribed_square_area (s : InscribedSquare) 
    (h1 : s.dist1 = 15) 
    (h2 : s.dist2 = 65) : 
  s.side * s.side = 975 := by
  sorry

end inscribed_square_area_l3638_363870


namespace number_value_l3638_363804

theorem number_value (x number : ℝ) 
  (h1 : (x + 5) * (number - 5) = 0)
  (h2 : ∀ y z : ℝ, (y + 5) * (z - 5) = 0 → x^2 + number^2 ≤ y^2 + z^2) :
  number = 5 := by
sorry

end number_value_l3638_363804


namespace greg_and_sarah_apples_l3638_363832

/-- Represents the number of apples each person has -/
structure AppleDistribution where
  greg : ℕ
  sarah : ℕ
  susan : ℕ
  mark : ℕ
  mom : ℕ

/-- Checks if the apple distribution satisfies the given conditions -/
def is_valid_distribution (d : AppleDistribution) : Prop :=
  d.greg = d.sarah ∧
  d.susan = 2 * d.greg ∧
  d.mark = d.susan - 5 ∧
  d.mom = 49

/-- Theorem stating that Greg and Sarah have 18 apples in total -/
theorem greg_and_sarah_apples (d : AppleDistribution) 
  (h : is_valid_distribution d) : d.greg + d.sarah = 18 := by
  sorry

end greg_and_sarah_apples_l3638_363832


namespace milk_cost_percentage_l3638_363882

-- Define the costs and total amount
def sandwich_cost : ℝ := 4
def juice_cost : ℝ := 2 * sandwich_cost
def total_paid : ℝ := 21

-- Define the total cost of sandwich and juice
def sandwich_juice_total : ℝ := sandwich_cost + juice_cost

-- Define the cost of milk
def milk_cost : ℝ := total_paid - sandwich_juice_total

-- The theorem to prove
theorem milk_cost_percentage : 
  (milk_cost / sandwich_juice_total) * 100 = 75 := by sorry

end milk_cost_percentage_l3638_363882


namespace trigonometric_problem_l3638_363845

open Real

theorem trigonometric_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : sin α = 2 * Real.sqrt 5 / 5) :
  (tan α = 2) ∧ 
  ((4 * sin (π - α) + 2 * cos (2 * π - α)) / (sin (π/2 - α) - sin α) = -10) := by
  sorry

end trigonometric_problem_l3638_363845


namespace new_ratio_is_25_to_1_l3638_363827

/-- Represents the ratio of students to teachers -/
structure Ratio where
  students : ℕ
  teachers : ℕ

def initial_ratio : Ratio := { students := 50, teachers := 1 }
def initial_teachers : ℕ := 3
def student_increase : ℕ := 50
def teacher_increase : ℕ := 5

def new_ratio : Ratio :=
  { students := initial_ratio.students * initial_teachers + student_increase,
    teachers := initial_teachers + teacher_increase }

theorem new_ratio_is_25_to_1 : new_ratio = { students := 25, teachers := 1 } := by
  sorry

end new_ratio_is_25_to_1_l3638_363827


namespace cuboid_height_calculation_l3638_363897

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: A cuboid with length 4 cm, breadth 6 cm, and surface area 120 cm² has a height of 3.6 cm -/
theorem cuboid_height_calculation (h : ℝ) :
  cuboidSurfaceArea 4 6 h = 120 → h = 3.6 := by
  sorry

end cuboid_height_calculation_l3638_363897


namespace min_value_quadratic_l3638_363826

theorem min_value_quadratic :
  let f (x : ℝ) := x^2 + 14*x + 10
  ∃ (y_min : ℝ), (∀ (x : ℝ), f x ≥ y_min) ∧ (∃ (x : ℝ), f x = y_min) ∧ y_min = -39 := by
  sorry

end min_value_quadratic_l3638_363826


namespace train_length_is_200_emily_steps_l3638_363813

/-- Represents the movement of Emily relative to a train -/
structure EmilyAndTrain where
  emily_step : ℝ
  train_step : ℝ
  train_length : ℝ

/-- The conditions of Emily's run relative to the train -/
def emily_run_conditions (et : EmilyAndTrain) : Prop :=
  ∃ (e t : ℝ),
    et.emily_step = e ∧
    et.train_step = t ∧
    et.train_length = 300 * e + 300 * t ∧
    et.train_length = 90 * e - 90 * t

/-- The theorem stating that under the given conditions, 
    the train length is 200 times Emily's step length -/
theorem train_length_is_200_emily_steps 
  (et : EmilyAndTrain) 
  (h : emily_run_conditions et) : 
  et.train_length = 200 * et.emily_step := by
  sorry

end train_length_is_200_emily_steps_l3638_363813


namespace abs_equals_diff_exists_l3638_363887

theorem abs_equals_diff_exists : ∃ x : ℝ, |x - 1| = x - 1 := by
  sorry

end abs_equals_diff_exists_l3638_363887


namespace overtaking_points_l3638_363814

theorem overtaking_points (track_length : ℕ) (pedestrian_speed : ℝ) (cyclist_speed : ℝ) : 
  track_length = 55 →
  cyclist_speed = 1.55 * pedestrian_speed →
  pedestrian_speed > 0 →
  (∃ n : ℕ, n * (cyclist_speed - pedestrian_speed) = track_length ∧ n = 11) :=
by sorry

end overtaking_points_l3638_363814


namespace max_value_parabola_l3638_363837

/-- The maximum value of y = -3x^2 + 7, where x is a real number, is 7. -/
theorem max_value_parabola :
  ∀ x : ℝ, -3 * x^2 + 7 ≤ 7 ∧ ∃ x₀ : ℝ, -3 * x₀^2 + 7 = 7 :=
by sorry

end max_value_parabola_l3638_363837


namespace fixed_point_on_line_l3638_363842

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  k * x + (1 - k) * y - 3 = 0

/-- Theorem stating that the line passes through (3, 3) for all k -/
theorem fixed_point_on_line :
  ∀ (k : ℝ), line_equation k 3 3 :=
by sorry

end fixed_point_on_line_l3638_363842


namespace paper_boutique_sales_l3638_363893

theorem paper_boutique_sales (notebook_sales : ℝ) (marker_sales : ℝ) (stapler_sales : ℝ)
  (h1 : notebook_sales = 25)
  (h2 : marker_sales = 40)
  (h3 : stapler_sales = 15)
  (h4 : notebook_sales + marker_sales + stapler_sales + (100 - notebook_sales - marker_sales - stapler_sales) = 100) :
  100 - notebook_sales - marker_sales = 35 := by
sorry

end paper_boutique_sales_l3638_363893


namespace factorization_proof_l3638_363829

theorem factorization_proof (b : ℝ) : 2 * b^2 - 8 * b + 8 = 2 * (b - 2)^2 := by
  sorry

end factorization_proof_l3638_363829


namespace work_ratio_women_to_men_l3638_363836

/-- The ratio of work done by women to men given specific work conditions -/
theorem work_ratio_women_to_men :
  let men_count : ℕ := 15
  let men_days : ℕ := 21
  let men_hours_per_day : ℕ := 8
  let women_count : ℕ := 21
  let women_days : ℕ := 36
  let women_hours_per_day : ℕ := 5
  let men_total_hours : ℕ := men_count * men_days * men_hours_per_day
  let women_total_hours : ℕ := women_count * women_days * women_hours_per_day
  (men_total_hours : ℚ) / women_total_hours = 2 / 3 :=
by
  sorry


end work_ratio_women_to_men_l3638_363836


namespace negative_three_x_squared_times_two_x_l3638_363873

theorem negative_three_x_squared_times_two_x (x : ℝ) : (-3 * x)^2 * (2 * x) = 18 * x^3 := by
  sorry

end negative_three_x_squared_times_two_x_l3638_363873


namespace intersection_P_complement_Q_l3638_363809

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_P_complement_Q : P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_P_complement_Q_l3638_363809


namespace no_solutions_for_equation_l3638_363866

theorem no_solutions_for_equation : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (2 / a + 2 / b = 1 / (a + b)) := by
  sorry

end no_solutions_for_equation_l3638_363866


namespace median_exists_l3638_363895

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem median_exists : ∃ a : ℝ, is_median {a, 2, 4, 0, 5} 4 := by
  sorry

end median_exists_l3638_363895


namespace plane_perpendicular_condition_l3638_363807

/-- The normal vector of a plane -/
structure NormalVector where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Dot product of two normal vectors -/
def dot_product (v1 v2 : NormalVector) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

/-- Two planes are perpendicular if their normal vectors are orthogonal -/
def perpendicular (v1 v2 : NormalVector) : Prop :=
  dot_product v1 v2 = 0

theorem plane_perpendicular_condition (k : ℝ) :
  let α : NormalVector := ⟨3, 1, -2⟩
  let β : NormalVector := ⟨-1, 1, k⟩
  perpendicular α β → k = -1 :=
by sorry

end plane_perpendicular_condition_l3638_363807


namespace largest_integer_inequality_l3638_363855

theorem largest_integer_inequality (y : ℤ) : (y / 4 : ℚ) + 3 / 7 < 7 / 4 ↔ y ≤ 5 := by
  sorry

#check largest_integer_inequality

end largest_integer_inequality_l3638_363855


namespace circle_radius_exists_l3638_363828

theorem circle_radius_exists : ∃ r : ℝ, r > 0 ∧ π * r^2 + 2 * r - 2 * π * r = 12 := by
  sorry

end circle_radius_exists_l3638_363828


namespace crouton_calories_l3638_363822

def salad_calories : ℕ := 350
def lettuce_calories : ℕ := 30
def cucumber_calories : ℕ := 80
def num_croutons : ℕ := 12

theorem crouton_calories : 
  (salad_calories - lettuce_calories - cucumber_calories) / num_croutons = 20 := by
  sorry

end crouton_calories_l3638_363822


namespace earning_amount_l3638_363843

/-- Represents the earning and spending pattern over 60 days -/
def pattern_result (E : ℚ) : ℚ :=
  30 * (E - 15)

/-- Proves that the earning amount must be 17 given the conditions -/
theorem earning_amount : ∃ E : ℚ, pattern_result E = 60 ∧ E = 17 := by
  sorry

end earning_amount_l3638_363843


namespace count_equals_60_l3638_363835

/-- A function that generates all 5-digit numbers composed of digits 1, 2, 3, 4, and 5 without repetition -/
def generate_numbers : List Nat := sorry

/-- A function that checks if a number is greater than 23145 and less than 43521 -/
def is_in_range (n : Nat) : Bool := 23145 < n && n < 43521

/-- The count of numbers in the specified range -/
def count_in_range : Nat :=
  (generate_numbers.filter is_in_range).length

theorem count_equals_60 : count_in_range = 60 := by sorry

end count_equals_60_l3638_363835


namespace angle_terminal_side_l3638_363838

open Real

theorem angle_terminal_side (α : Real) :
  (tan α < 0 ∧ cos α < 0) → 
  (π / 2 < α ∧ α < π) :=
by sorry

end angle_terminal_side_l3638_363838


namespace weight_meets_standard_l3638_363801

/-- The nominal weight of the strawberry box in kilograms -/
def nominal_weight : ℝ := 5

/-- The allowed deviation from the nominal weight in kilograms -/
def allowed_deviation : ℝ := 0.03

/-- The actual weight of the strawberry box in kilograms -/
def actual_weight : ℝ := 4.98

/-- Theorem stating that the actual weight meets the standard -/
theorem weight_meets_standard : 
  nominal_weight - allowed_deviation ≤ actual_weight ∧ 
  actual_weight ≤ nominal_weight + allowed_deviation := by
  sorry

end weight_meets_standard_l3638_363801


namespace adam_bought_seven_boxes_l3638_363833

/-- The number of boxes Adam gave away -/
def boxes_given_away : ℕ := 7

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- The number of pieces Adam still has -/
def remaining_pieces : ℕ := 36

/-- The number of boxes Adam bought initially -/
def initial_boxes : ℕ := 7

theorem adam_bought_seven_boxes :
  initial_boxes * pieces_per_box = boxes_given_away * pieces_per_box + remaining_pieces :=
by sorry

end adam_bought_seven_boxes_l3638_363833


namespace quadratic_distinct_roots_condition_l3638_363817

/-- A quadratic equation with coefficients a, b, and c has two distinct real roots if and only if its discriminant is positive. -/
axiom quadratic_two_distinct_roots (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- For the quadratic equation x^2 + 2x + k = 0 to have two distinct real roots, k must be less than 1. -/
theorem quadratic_distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0) ↔ k < 1 := by
  sorry

end quadratic_distinct_roots_condition_l3638_363817


namespace reciprocal_sum_greater_than_four_l3638_363851

theorem reciprocal_sum_greater_than_four 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (sum_of_squares : a^2 + b^2 + c^2 = 1) : 
  1/a + 1/b + 1/c > 4 := by
sorry

end reciprocal_sum_greater_than_four_l3638_363851


namespace divisibility_of_power_plus_one_l3638_363853

theorem divisibility_of_power_plus_one (n : ℕ) : 
  ∃ k : ℤ, (2 : ℤ) ^ (3 ^ n) + 1 = k * (3 : ℤ) ^ (n + 1) := by
sorry

end divisibility_of_power_plus_one_l3638_363853


namespace student_multiplication_factor_l3638_363823

theorem student_multiplication_factor : ∃ (x : ℚ), 121 * x - 138 = 104 ∧ x = 2 := by
  sorry

end student_multiplication_factor_l3638_363823


namespace eliminate_first_power_term_l3638_363811

theorem eliminate_first_power_term (a m : ℝ) : 
  (∀ k, (a + m) * (a + 1/2) = k * a^2 + c) ↔ m = -1/2 := by sorry

end eliminate_first_power_term_l3638_363811


namespace attic_junk_percentage_l3638_363847

theorem attic_junk_percentage :
  ∀ (total useful heirlooms junk : ℕ),
    useful = (20 : ℕ) * total / 100 →
    heirlooms = (10 : ℕ) * total / 100 →
    useful = 8 →
    junk = 28 →
    total = useful + heirlooms + junk →
    (junk : ℚ) / (total : ℚ) = 7 / 10 := by
  sorry

end attic_junk_percentage_l3638_363847


namespace cheryl_material_usage_l3638_363896

theorem cheryl_material_usage
  (material1 : ℚ)
  (material2 : ℚ)
  (leftover : ℚ)
  (h1 : material1 = 4 / 19)
  (h2 : material2 = 2 / 13)
  (h3 : leftover = 4 / 26)
  : material1 + material2 - leftover = 52 / 247 :=
by sorry

end cheryl_material_usage_l3638_363896


namespace computer_price_after_15_years_l3638_363812

/-- Proves that a computer's price after 15 years of depreciation is 2400 yuan,
    given an initial price of 8100 yuan and a 1/3 price decrease every 5 years. -/
theorem computer_price_after_15_years
  (initial_price : ℝ)
  (price_decrease_ratio : ℝ)
  (price_decrease_period : ℕ)
  (total_time : ℕ)
  (h1 : initial_price = 8100)
  (h2 : price_decrease_ratio = 1 / 3)
  (h3 : price_decrease_period = 5)
  (h4 : total_time = 15)
  : initial_price * (1 - price_decrease_ratio) ^ (total_time / price_decrease_period) = 2400 :=
sorry

end computer_price_after_15_years_l3638_363812


namespace shopping_equation_system_l3638_363802

theorem shopping_equation_system (x y : ℤ) : 
  (∀ (coins_per_person excess : ℤ), coins_per_person * x - y = excess → 
    ((coins_per_person = 8 ∧ excess = 3) ∨ (coins_per_person = 7 ∧ excess = -4))) → 
  (8 * x - y = 3 ∧ y - 7 * x = 4) := by
sorry

end shopping_equation_system_l3638_363802


namespace divides_two_pow_minus_one_l3638_363884

theorem divides_two_pow_minus_one (n : ℕ) : n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by
  sorry

end divides_two_pow_minus_one_l3638_363884


namespace valid_configurations_l3638_363872

/-- A configuration of lines and points on a plane -/
structure PlaneConfiguration where
  n : ℕ  -- number of points
  lines : Fin 3 → Set (ℝ × ℝ)  -- three lines represented as sets of points
  points : Fin n → ℝ × ℝ  -- n points

/-- Predicate to check if a point is on a line -/
def isOnLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  p ∈ l

/-- Predicate to check if a point is on either side of a line -/
def isOnEitherSide (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  ¬(isOnLine p l)

/-- The main theorem stating the possible values of n -/
theorem valid_configurations (c : PlaneConfiguration) :
  (∀ l : Fin 3, ∃! (s₁ s₂ : Finset (Fin c.n)),
    s₁.card = 2 ∧ s₂.card = 2 ∧ 
    (∀ i ∈ s₁, isOnEitherSide (c.points i) (c.lines l)) ∧
    (∀ i ∈ s₂, isOnEitherSide (c.points i) (c.lines l)) ∧
    (∀ i : Fin c.n, i ∉ s₁ ∧ i ∉ s₂ → isOnLine (c.points i) (c.lines l))) →
  c.n = 0 ∨ c.n = 1 ∨ c.n = 3 ∨ c.n = 4 ∨ c.n = 6 ∨ c.n = 7 :=
by sorry

end valid_configurations_l3638_363872


namespace representatives_selection_theorem_l3638_363880

/-- The number of ways to select representatives from a group of students -/
def select_representatives (total_students : ℕ) (num_representatives : ℕ) (restricted_student : ℕ) : ℕ :=
  (total_students - 1) * (total_students - 1) * (total_students - 2)

/-- Theorem stating the number of ways to select 3 representatives from 5 students,
    with one student restricted from being the Mathematics representative -/
theorem representatives_selection_theorem :
  select_representatives 5 3 1 = 48 := by
  sorry

end representatives_selection_theorem_l3638_363880


namespace difference_of_squares_numbers_l3638_363878

def is_difference_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > b ∧ n = a * a - b * b

theorem difference_of_squares_numbers :
  is_difference_of_squares 2020 ∧
  is_difference_of_squares 2022 ∧
  is_difference_of_squares 2023 ∧
  is_difference_of_squares 2024 ∧
  ¬is_difference_of_squares 2021 :=
sorry

end difference_of_squares_numbers_l3638_363878


namespace prob_odd_sum_is_two_thirds_l3638_363859

/-- A card is represented by a natural number between 1 and 4 -/
def Card := {n : ℕ // 1 ≤ n ∧ n ≤ 4}

/-- The set of all possible cards -/
def allCards : Finset Card := sorry

/-- A function to check if the sum of two cards is odd -/
def isOddSum (c1 c2 : Card) : Bool := sorry

/-- The set of all pairs of cards -/
def allPairs : Finset (Card × Card) := sorry

/-- The set of all pairs of cards with odd sum -/
def oddSumPairs : Finset (Card × Card) := sorry

/-- The probability of drawing two cards with odd sum -/
def probOddSum : ℚ := (Finset.card oddSumPairs : ℚ) / (Finset.card allPairs : ℚ)

theorem prob_odd_sum_is_two_thirds : probOddSum = 2/3 := by sorry

end prob_odd_sum_is_two_thirds_l3638_363859


namespace balloon_difference_l3638_363825

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 5

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 3

/-- Theorem stating the difference in number of balloons between Allan and Jake -/
theorem balloon_difference : allan_balloons - jake_balloons = 2 := by
  sorry

end balloon_difference_l3638_363825


namespace min_abs_z_complex_l3638_363846

theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - 5) = 7) :
  ∃ (w : ℂ), Complex.abs w = 10/7 ∧ ∀ z', Complex.abs (z' - 2*I) + Complex.abs (z' - 5) = 7 → Complex.abs w ≤ Complex.abs z' :=
sorry

end min_abs_z_complex_l3638_363846


namespace arithmetic_sequence_common_difference_l3638_363894

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a5 : a 5 = 10)
  (h_a10 : a 10 = -5) :
  CommonDifference a = -3 := by
sorry

end arithmetic_sequence_common_difference_l3638_363894


namespace bug_on_square_probability_l3638_363867

/-- Represents the probability of the bug being at its starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 1 - Q n

/-- The bug's movement on a square -/
theorem bug_on_square_probability : Q 8 = 1 := by
  sorry

end bug_on_square_probability_l3638_363867


namespace four_digit_square_completion_l3638_363886

theorem four_digit_square_completion : 
  ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧ 
    ∃ k : ℕ, (400 * 10000 + n) = k^2 :=
sorry

end four_digit_square_completion_l3638_363886


namespace edward_money_problem_l3638_363841

def initial_amount : ℕ := 14
def spent_amount : ℕ := 17
def received_amount : ℕ := 10
def final_amount : ℕ := 7

theorem edward_money_problem :
  initial_amount - spent_amount + received_amount = final_amount :=
by sorry

end edward_money_problem_l3638_363841


namespace repeating_decimal_to_fraction_l3638_363883

/-- Proves that the repeating decimal 0.35̄ is equal to 5/14 -/
theorem repeating_decimal_to_fraction : 
  ∀ x : ℚ, (∃ n : ℕ, x = (35 : ℚ) / (100^n - 1)) → x = 5/14 := by
  sorry

end repeating_decimal_to_fraction_l3638_363883


namespace total_books_calculation_l3638_363854

def initial_books : ℕ := 9
def added_books : ℕ := 10

theorem total_books_calculation :
  initial_books + added_books = 19 :=
by sorry

end total_books_calculation_l3638_363854


namespace eggs_to_market_l3638_363848

/-- Represents the number of dozens of eggs collected on each collection day -/
def eggs_collected_per_day : ℕ := 8

/-- Represents the number of collection days per week -/
def collection_days_per_week : ℕ := 2

/-- Represents the number of dozens of eggs delivered to the mall -/
def eggs_to_mall : ℕ := 5

/-- Represents the number of dozens of eggs used for pie -/
def eggs_for_pie : ℕ := 4

/-- Represents the number of dozens of eggs donated to charity -/
def eggs_to_charity : ℕ := 4

/-- Represents the total number of dozens of eggs collected in a week -/
def total_eggs_collected : ℕ := eggs_collected_per_day * collection_days_per_week

/-- Represents the total number of dozens of eggs used or given away -/
def total_eggs_used : ℕ := eggs_to_mall + eggs_for_pie + eggs_to_charity

/-- Proves that the number of dozens of eggs delivered to the market is 3 -/
theorem eggs_to_market : total_eggs_collected - total_eggs_used = 3 := by
  sorry

end eggs_to_market_l3638_363848


namespace a_range_if_increasing_l3638_363877

/-- The sequence defined by a_n = an^2 + n -/
def a_seq (a : ℝ) (n : ℕ) : ℝ := a * n^2 + n

/-- The theorem stating that if the sequence is increasing, then a is non-negative -/
theorem a_range_if_increasing (a : ℝ) :
  (∀ n : ℕ, a_seq a n < a_seq a (n + 1)) → a ≥ 0 :=
by sorry

end a_range_if_increasing_l3638_363877


namespace medium_pizza_slices_l3638_363858

theorem medium_pizza_slices :
  -- Define the number of slices for small and large pizzas
  let small_slices : ℕ := 6
  let large_slices : ℕ := 12
  -- Define the total number of pizzas and the number of each size
  let total_pizzas : ℕ := 15
  let small_pizzas : ℕ := 4
  let medium_pizzas : ℕ := 5
  -- Define the total number of slices
  let total_slices : ℕ := 136
  -- Calculate the number of large pizzas
  let large_pizzas : ℕ := total_pizzas - small_pizzas - medium_pizzas
  -- Define the number of slices in a medium pizza as a variable
  ∀ medium_slices : ℕ,
  -- If the total slices equation holds
  (small_pizzas * small_slices + medium_pizzas * medium_slices + large_pizzas * large_slices = total_slices) →
  -- Then the number of slices in a medium pizza must be 8
  medium_slices = 8 := by
sorry

end medium_pizza_slices_l3638_363858


namespace garden_length_l3638_363831

/-- Proves that a rectangular garden with length twice its width and 300 yards of fencing has a length of 100 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- length is twice the width
  2 * length + 2 * width = 300 →  -- 300 yards of fencing encloses the garden
  length = 100 := by
  sorry

end garden_length_l3638_363831


namespace increasing_order_abc_l3638_363819

theorem increasing_order_abc (a b c : ℝ) : 
  a = 2^(4/3) → b = 3^(2/3) → c = 25^(1/3) → b < a ∧ a < c := by
  sorry

end increasing_order_abc_l3638_363819


namespace walking_speed_equation_l3638_363808

theorem walking_speed_equation (x : ℝ) 
  (h1 : x > 0) -- Xiao Wang's speed is positive
  (h2 : x + 1 > 0) -- Xiao Zhang's speed is positive
  : 
  (15 / x - 15 / (x + 1) = 1 / 2) ↔ 
  (15 / x = 15 / (x + 1) + 1 / 2) :=
by sorry

end walking_speed_equation_l3638_363808


namespace stratified_sampling_proportion_l3638_363868

theorem stratified_sampling_proportion (second_year_total : ℕ) (third_year_total : ℕ) (third_year_sample : ℕ) :
  second_year_total = 1600 →
  third_year_total = 1400 →
  third_year_sample = 70 →
  (third_year_sample : ℚ) / third_year_total = 80 / second_year_total :=
by sorry

end stratified_sampling_proportion_l3638_363868


namespace min_a_for_increasing_f_l3638_363803

/-- The function f(x) defined as x² + (a-2)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 2) * x - 1

/-- The property that f is increasing on the interval [2, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → x < y → f a x < f a y

/-- The theorem stating the minimum value of a for which f is increasing on [2, +∞) -/
theorem min_a_for_increasing_f :
  (∃ a_min : ℝ, (∀ a : ℝ, is_increasing_on_interval a ↔ a_min ≤ a) ∧ a_min = -2) :=
sorry

end min_a_for_increasing_f_l3638_363803


namespace circle_positions_l3638_363871

theorem circle_positions (a b d : ℝ) (h1 : a = 4) (h2 : b = 10) (h3 : b > a) :
  (∃ d, d = b - a) ∧
  (∃ d, d = b + a) ∧
  (∃ d, d > b + a) ∧
  (∃ d, d > b - a) :=
by sorry

end circle_positions_l3638_363871


namespace simplify_expression_l3638_363864

theorem simplify_expression (y : ℝ) : 7 * y + 8 - 3 * y + 16 = 4 * y + 24 := by
  sorry

end simplify_expression_l3638_363864


namespace perfect_square_sum_in_partition_l3638_363844

theorem perfect_square_sum_in_partition (n : ℕ) (A B : Set ℕ) 
  (h1 : n ≥ 15)
  (h2 : A ⊆ Finset.range (n + 1))
  (h3 : B ⊆ Finset.range (n + 1))
  (h4 : A ∩ B = ∅)
  (h5 : A ∪ B = Finset.range (n + 1))
  (h6 : A ≠ Finset.range (n + 1))
  (h7 : B ≠ Finset.range (n + 1)) :
  ∃ (x y : ℕ), (x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ ∃ (k : ℕ), x + y = k^2) ∨
               (x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ ∃ (k : ℕ), x + y = k^2) :=
by sorry

end perfect_square_sum_in_partition_l3638_363844


namespace torch_relay_probability_l3638_363852

/-- The number of torchbearers -/
def n : ℕ := 5

/-- The number of torchbearers to be selected -/
def k : ℕ := 2

/-- The total number of ways to select k torchbearers from n torchbearers -/
def total_combinations : ℕ := n.choose k

/-- The number of ways to select k consecutive torchbearers from n torchbearers -/
def consecutive_combinations : ℕ := n - k + 1

/-- The probability of selecting consecutive torchbearers -/
def probability : ℚ := consecutive_combinations / total_combinations

theorem torch_relay_probability : probability = 2 / 5 := by
  sorry

end torch_relay_probability_l3638_363852


namespace age_difference_l3638_363856

theorem age_difference : ∃ (a b : ℕ), 
  (a ≤ 9 ∧ b ≤ 9) ∧ 
  (10 * a + b + 5 = 2 * (10 * b + a + 5)) ∧
  ((10 * a + b) - (10 * b + a) = 18) := by
  sorry

end age_difference_l3638_363856


namespace star_sum_equals_396_l3638_363850

def star (a b : ℕ) : ℕ := a * a - b * b

theorem star_sum_equals_396 : 
  (List.range 18).foldl (λ acc i => acc + star (i + 3) (i + 2)) 0 = 396 := by
  sorry

end star_sum_equals_396_l3638_363850


namespace remaining_money_l3638_363857

def base_8_to_10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def savings : ℕ := 5555

def ticket_cost : ℕ := 1200

theorem remaining_money :
  base_8_to_10 savings - ticket_cost = 1725 := by sorry

end remaining_money_l3638_363857


namespace function_value_theorem_l3638_363899

theorem function_value_theorem (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f ((1/2) * x - 1) = 2 * x + 3) →
  f m = 6 →
  m = -(1/4) := by
sorry

end function_value_theorem_l3638_363899


namespace negation_of_existence_proposition_l3638_363891

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 2*x + 1 ≥ 0) := by
  sorry

end negation_of_existence_proposition_l3638_363891


namespace expansion_simplification_l3638_363862

theorem expansion_simplification (a b : ℝ) : (a + b) * (3 * a - b) - b * (a - b) = 3 * a^2 + a * b := by
  sorry

end expansion_simplification_l3638_363862


namespace handshake_count_l3638_363800

/-- Represents a company at the convention -/
inductive Company
| A | B | C | D | E

/-- The number of companies at the convention -/
def num_companies : Nat := 5

/-- The number of representatives per company -/
def reps_per_company : Nat := 4

/-- The total number of attendees at the convention -/
def total_attendees : Nat := num_companies * reps_per_company

/-- Determines if two companies are the same -/
def same_company (c1 c2 : Company) : Bool :=
  match c1, c2 with
  | Company.A, Company.A => true
  | Company.B, Company.B => true
  | Company.C, Company.C => true
  | Company.D, Company.D => true
  | Company.E, Company.E => true
  | _, _ => false

/-- Determines if a company is Company A -/
def is_company_a (c : Company) : Bool :=
  match c with
  | Company.A => true
  | _ => false

/-- Calculates the number of handshakes for a given company -/
def handshakes_for_company (c : Company) : Nat :=
  if is_company_a c then
    reps_per_company * (total_attendees - reps_per_company)
  else
    reps_per_company * (total_attendees - 2 * reps_per_company)

/-- The total number of handshakes at the convention -/
def total_handshakes : Nat :=
  (handshakes_for_company Company.A +
   handshakes_for_company Company.B +
   handshakes_for_company Company.C +
   handshakes_for_company Company.D +
   handshakes_for_company Company.E) / 2

/-- The main theorem stating that the total number of handshakes is 128 -/
theorem handshake_count : total_handshakes = 128 := by
  sorry


end handshake_count_l3638_363800


namespace purely_imaginary_complex_number_l3638_363834

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := (x^2 - 1) + (x + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
  sorry

end purely_imaginary_complex_number_l3638_363834


namespace floor_abs_sum_l3638_363820

theorem floor_abs_sum : ⌊|(-3.1 : ℝ)|⌋ + |⌊(-3.1 : ℝ)⌋| = 7 := by
  sorry

end floor_abs_sum_l3638_363820


namespace four_digit_cubes_divisible_by_16_l3638_363810

theorem four_digit_cubes_divisible_by_16 :
  (∃! (s : Finset ℕ), s = {n : ℕ | 1000 ≤ (2*n)^3 ∧ (2*n)^3 ≤ 9999} ∧ Finset.card s = 3) := by
  sorry

end four_digit_cubes_divisible_by_16_l3638_363810


namespace inequality_proof_l3638_363874

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : (x^2 / (1 + x^2)) + (y^2 / (1 + y^2)) + (z^2 / (1 + z^2)) = 2) : 
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ Real.sqrt 2 := by
  sorry

end inequality_proof_l3638_363874


namespace smallest_n_for_inequality_l3638_363806

theorem smallest_n_for_inequality : ∃ (n : ℕ+),
  (∀ (m : ℕ), 0 < m → m < 2001 → 
    ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / (n : ℚ) ∧ (k : ℚ) / (n : ℚ) < ((m + 1) : ℚ) / 2002) ∧
  (∀ (n' : ℕ+), 
    (∀ (m : ℕ), 0 < m → m < 2001 → 
      ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / (n' : ℚ) ∧ (k : ℚ) / (n' : ℚ) < ((m + 1) : ℚ) / 2002) →
    n ≤ n') ∧
  n = 4003 :=
sorry

end smallest_n_for_inequality_l3638_363806


namespace no_solution_to_double_inequality_l3638_363898

theorem no_solution_to_double_inequality :
  ¬ ∃ x : ℝ, (4 * x - 3 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 5) := by
  sorry

end no_solution_to_double_inequality_l3638_363898


namespace subtraction_in_third_quadrant_l3638_363830

/-- Given complex numbers z₁ and z₂, prove that z₁ - z₂ is in the third quadrant -/
theorem subtraction_in_third_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = -2 + I) 
  (h₂ : z₂ = 1 + 2*I) : 
  let z := z₁ - z₂
  (z.re < 0 ∧ z.im < 0) := by
  sorry

end subtraction_in_third_quadrant_l3638_363830


namespace certain_number_proof_l3638_363821

theorem certain_number_proof (x : ℝ) : 
  (0.02: ℝ)^2 + x^2 + (0.035 : ℝ)^2 = 100 * ((0.002 : ℝ)^2 + (0.052 : ℝ)^2 + (0.0035 : ℝ)^2) → 
  x = 0.52 := by
sorry

end certain_number_proof_l3638_363821


namespace parabola_c_value_l3638_363888

/-- A parabola with vertex at (-2, 3) passing through (2, 7) has c = 4 in its equation y = ax^2 + bx + c -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (3 = a * (-2)^2 + b * (-2) + c) →       -- Condition 2 (vertex)
  (3 = a * 4 + b * (-2) + c) →            -- Condition 2 (vertex)
  (7 = a * 2^2 + b * 2 + c) →             -- Condition 3
  c = 4 := by
sorry


end parabola_c_value_l3638_363888


namespace min_sum_box_dimensions_l3638_363892

theorem min_sum_box_dimensions : 
  ∀ (l w h : ℕ+), 
  l * w * h = 3003 → 
  ∀ (a b c : ℕ+), 
  a * b * c = 3003 → 
  l + w + h ≤ a + b + c ∧
  ∃ (x y z : ℕ+), x * y * z = 3003 ∧ x + y + z = 45 := by
sorry

end min_sum_box_dimensions_l3638_363892


namespace function_properties_l3638_363875

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := Real.log (a^x - b^x) / Real.log 10

-- State the theorem
theorem function_properties (a b : ℝ) (ha : a > 1) (hb : 0 < b) (hab : b < 1) :
  -- 1. Domain of f is (0, +∞)
  (∀ x : ℝ, x > 0 → (a^x - b^x > 0)) ∧
  -- 2. No two distinct points with same y-value
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a b x₁ ≠ f a b x₂) ∧
  -- 3. Condition for f to be positive on (1, +∞)
  (a ≥ b + 1 → ∀ x : ℝ, x > 1 → f a b x > 0) :=
by sorry

end function_properties_l3638_363875


namespace large_lemonhead_doll_cost_l3638_363889

/-- The cost of a large lemonhead doll satisfies the given conditions -/
theorem large_lemonhead_doll_cost :
  ∃ (L : ℝ), 
    (L > 0) ∧ 
    (350 / (L - 2) = 350 / L + 20) ∧ 
    (L = 7) := by
  sorry

end large_lemonhead_doll_cost_l3638_363889


namespace crayon_count_l3638_363815

theorem crayon_count (initial_crayons added_crayons : ℕ) 
  (h1 : initial_crayons = 9)
  (h2 : added_crayons = 3) :
  initial_crayons + added_crayons = 12 := by
sorry

end crayon_count_l3638_363815


namespace product_units_digit_of_first_five_composite_l3638_363840

def first_five_composite_numbers : List Nat := [4, 6, 8, 9, 10]

def units_digit (n : Nat) : Nat := n % 10

def product_units_digit (numbers : List Nat) : Nat :=
  units_digit (numbers.foldl (·*·) 1)

theorem product_units_digit_of_first_five_composite : 
  product_units_digit first_five_composite_numbers = 0 := by
  sorry

end product_units_digit_of_first_five_composite_l3638_363840


namespace tan_half_implies_expression_eight_l3638_363824

theorem tan_half_implies_expression_eight (x : ℝ) (h : Real.tan x = 1 / 2) :
  (2 * Real.sin x + 3 * Real.cos x) / (Real.cos x - Real.sin x) = 8 := by
  sorry

end tan_half_implies_expression_eight_l3638_363824


namespace right_triangle_inequality_right_triangle_inequality_optimal_l3638_363865

theorem right_triangle_inequality (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (side_order : a ≤ b ∧ b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3 * Real.sqrt 2) * a * b * c :=
by sorry

theorem right_triangle_inequality_optimal (k : ℝ) 
  (h : ∀ (a b c : ℝ), a^2 + b^2 = c^2 → a ≤ b → b < c → 
    a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ k * a * b * c) :
  k ≤ 2 + 3 * Real.sqrt 2 :=
by sorry

end right_triangle_inequality_right_triangle_inequality_optimal_l3638_363865


namespace repeated_root_condition_l3638_363869

/-- The equation has a repeated root if and only if m = -1 -/
theorem repeated_root_condition (m : ℝ) : 
  (∃ x : ℝ, (x - 6) / (x - 5) + 1 = m / (x - 5) ∧ 
   ∀ y : ℝ, y ≠ x → (y - 6) / (y - 5) + 1 ≠ m / (y - 5)) ↔ 
  m = -1 := by
  sorry

end repeated_root_condition_l3638_363869


namespace geometric_sequence_sixth_term_l3638_363881

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) := ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_third_term : a 3 = 27)
  (h_ninth_term : a 9 = 3) :
  a 6 = 9 := by
sorry

end geometric_sequence_sixth_term_l3638_363881
