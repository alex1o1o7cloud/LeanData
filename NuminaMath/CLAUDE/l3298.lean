import Mathlib

namespace root_in_interval_l3298_329824

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  ∃ (root : ℝ), root ∈ Set.Icc 2 2.5 ∧ f root = 0 :=
by
  have h1 : f 2 < 0 := by sorry
  have h2 : f 2.5 > 0 := by sorry
  have h3 : f 3 > 0 := by sorry
  sorry

end root_in_interval_l3298_329824


namespace cube_side_length_l3298_329856

theorem cube_side_length (surface_area : ℝ) (h : surface_area = 600) :
  ∃ (side_length : ℝ), side_length > 0 ∧ 6 * side_length^2 = surface_area ∧ side_length = 10 := by
  sorry

end cube_side_length_l3298_329856


namespace driver_weekly_distance_l3298_329852

def weekday_distance (speed1 speed2 speed3 time1 time2 time3 : ℕ) : ℕ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3

def sunday_distance (speed time : ℕ) : ℕ :=
  speed * time

def weekly_distance (weekday_dist sunday_dist days_per_week : ℕ) : ℕ :=
  weekday_dist * days_per_week + sunday_dist

theorem driver_weekly_distance :
  let weekday_dist := weekday_distance 30 25 40 3 4 2
  let sunday_dist := sunday_distance 35 5
  weekly_distance weekday_dist sunday_dist 6 = 1795 := by sorry

end driver_weekly_distance_l3298_329852


namespace equation_solution_l3298_329839

theorem equation_solution : 
  ∃! x : ℚ, (x - 17) / 3 = (3 * x + 4) / 8 ∧ x = -148 := by
  sorry

end equation_solution_l3298_329839


namespace line_equation_through_A_and_B_l3298_329803

/-- Two-point form equation of a line passing through two points -/
def two_point_form (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1)

/-- Theorem: The two-point form equation of the line passing through A(1,2) and B(-1,1) -/
theorem line_equation_through_A_and_B :
  two_point_form 1 2 (-1) 1 x y ↔ (x - 1) / (-2) = (y - 2) / (-1) := by
  sorry

end line_equation_through_A_and_B_l3298_329803


namespace lassis_and_smoothies_count_l3298_329879

/-- Represents the number of lassis that can be made from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (15 * mangoes) / 3

/-- Represents the number of smoothies that can be made from given numbers of mangoes and bananas -/
def smoothies_from_ingredients (mangoes bananas : ℕ) : ℕ :=
  min mangoes (bananas / 2)

/-- Theorem stating the number of lassis and smoothies that can be made -/
theorem lassis_and_smoothies_count :
  lassis_from_mangoes 18 = 90 ∧ smoothies_from_ingredients 18 36 = 18 :=
by sorry

end lassis_and_smoothies_count_l3298_329879


namespace trajectory_E_equation_max_area_AMBN_l3298_329821

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P on circle C
def P (x y : ℝ) : Prop := C x y

-- Define the point H as the foot of the perpendicular from P to x-axis
def H (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the point Q
def Q (x y : ℝ) : Prop := ∃ (px py : ℝ), P px py ∧ x = (px + (H px).1) / 2 ∧ y = (py + (H px).2) / 2

-- Define the trajectory E
def E (x y : ℝ) : Prop := Q x y

-- Define the line y = kx
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * x ∧ k > 0

-- Theorem for the equation of trajectory E
theorem trajectory_E_equation : ∀ x y : ℝ, E x y ↔ x^2/4 + y^2 = 1 :=
sorry

-- Theorem for the maximum area of quadrilateral AMBN
theorem max_area_AMBN : ∃ (max_area : ℝ), 
  (∀ k x1 y1 x2 y2 : ℝ, E x1 y1 ∧ E x2 y2 ∧ Line k x1 y1 ∧ Line k x2 y2 → 
    abs (x1 * y2 - x2 * y1) ≤ max_area) ∧
  max_area = 2 * Real.sqrt 2 :=
sorry

end trajectory_E_equation_max_area_AMBN_l3298_329821


namespace triangle_angle_sum_l3298_329865

theorem triangle_angle_sum 
  (A B C : ℝ) 
  (h_acute_A : 0 < A ∧ A < π/2) 
  (h_acute_B : 0 < B ∧ B < π/2)
  (h_sin_A : Real.sin A = Real.sqrt 5 / 5)
  (h_sin_B : Real.sin B = Real.sqrt 10 / 10)
  : Real.cos (A + B) = Real.sqrt 2 / 2 ∧ C = 3 * π / 4 := by
  sorry

end triangle_angle_sum_l3298_329865


namespace closest_point_is_correct_l3298_329859

/-- The point on the line y = -4x - 8 that is closest to (3, 6) -/
def closest_point : ℚ × ℚ := (-53/17, 76/17)

/-- The line y = -4x - 8 -/
def mouse_trajectory (x : ℚ) : ℚ := -4 * x - 8

theorem closest_point_is_correct :
  let (a, b) := closest_point
  -- The point is on the line
  (mouse_trajectory a = b) ∧
  -- It's the closest point to (3, 6)
  (∀ x y, mouse_trajectory x = y →
    (x - 3)^2 + (y - 6)^2 ≥ (a - 3)^2 + (b - 6)^2) ∧
  -- The sum of its coordinates is 23/17
  (a + b = 23/17) := by sorry


end closest_point_is_correct_l3298_329859


namespace trapezoid_segment_length_l3298_329893

/-- Represents a trapezoid ABCD with side lengths AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem stating that if the area ratio of triangles ABC to ADC is 8:2
    and AB + CD = 250, then AB = 200 -/
theorem trapezoid_segment_length (t : Trapezoid) 
    (h1 : (t.AB / t.CD) = 4)  -- Ratio of areas is equivalent to ratio of bases
    (h2 : t.AB + t.CD = 250) : 
  t.AB = 200 := by sorry

end trapezoid_segment_length_l3298_329893


namespace sqrt_equation_solution_l3298_329815

theorem sqrt_equation_solution :
  let f : ℝ → ℝ := λ x => Real.sqrt (x + 9) - 2 * Real.sqrt (x - 2) + 3
  ∃ x₁ x₂ : ℝ, x₁ = 8 + 4 * Real.sqrt 2 ∧ x₂ = 8 - 4 * Real.sqrt 2 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end sqrt_equation_solution_l3298_329815


namespace no_overlap_in_intervals_l3298_329872

theorem no_overlap_in_intervals (x : ℝ) : 
  50 ≤ x ∧ x ≤ 150 ∧ Int.floor (Real.sqrt x) = 11 → 
  Int.floor (Real.sqrt (50 * x)) ≠ 110 := by
sorry

end no_overlap_in_intervals_l3298_329872


namespace oak_trees_in_park_l3298_329863

theorem oak_trees_in_park (initial_trees : ℕ) (planted_trees : ℕ) : initial_trees = 5 → planted_trees = 4 → initial_trees + planted_trees = 9 := by
  sorry

end oak_trees_in_park_l3298_329863


namespace square_sum_eq_841_times_product_plus_one_l3298_329816

theorem square_sum_eq_841_times_product_plus_one :
  ∀ a b : ℕ, a^2 + b^2 = 841 * (a * b + 1) ↔ (a = 0 ∧ b = 29) ∨ (a = 29 ∧ b = 0) :=
by sorry

end square_sum_eq_841_times_product_plus_one_l3298_329816


namespace dice_sum_not_23_l3298_329827

theorem dice_sum_not_23 (a b c d e : ℕ) : 
  a ≥ 1 ∧ a ≤ 6 ∧
  b ≥ 1 ∧ b ≤ 6 ∧
  c ≥ 1 ∧ c ≤ 6 ∧
  d ≥ 1 ∧ d ≤ 6 ∧
  e ≥ 1 ∧ e ≤ 6 ∧
  a * b * c * d * e = 720 →
  a + b + c + d + e ≠ 23 :=
by sorry

end dice_sum_not_23_l3298_329827


namespace theater_tickets_proof_l3298_329868

theorem theater_tickets_proof (reduced_first_week : ℕ) 
  (h1 : reduced_first_week > 0)
  (h2 : 5 * reduced_first_week = 16500)
  (h3 : reduced_first_week + 16500 = 25200) : 
  reduced_first_week = 8700 := by
  sorry

end theater_tickets_proof_l3298_329868


namespace square_pyramid_volume_l3298_329883

/-- The volume of a regular square pyramid with base edge length 1 and height 3 is 1 -/
theorem square_pyramid_volume :
  let base_edge : ℝ := 1
  let height : ℝ := 3
  let base_area : ℝ := base_edge ^ 2
  let volume : ℝ := (1 / 3) * base_area * height
  volume = 1 := by
  sorry

end square_pyramid_volume_l3298_329883


namespace negation_of_absolute_value_statement_l3298_329812

theorem negation_of_absolute_value_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| ≥ 3) ↔ (∃ x ∈ S, |x| < 3) := by
  sorry

end negation_of_absolute_value_statement_l3298_329812


namespace smallest_block_with_399_hidden_cubes_l3298_329869

/-- A rectangular block made of identical cubes -/
structure RectangularBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of cubes in a rectangular block -/
def RectangularBlock.volume (b : RectangularBlock) : ℕ :=
  b.length * b.width * b.height

/-- The number of hidden cubes when three faces are visible -/
def RectangularBlock.hiddenCubes (b : RectangularBlock) : ℕ :=
  (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- The theorem stating the smallest possible value of N -/
theorem smallest_block_with_399_hidden_cubes :
  ∀ b : RectangularBlock,
    b.hiddenCubes = 399 →
    b.volume ≥ 640 ∧
    ∃ b' : RectangularBlock, b'.hiddenCubes = 399 ∧ b'.volume = 640 :=
by sorry

end smallest_block_with_399_hidden_cubes_l3298_329869


namespace right_triangle_area_l3298_329897

theorem right_triangle_area (a b c : ℝ) (h1 : b = (2/3) * a) (h2 : b = (2/3) * c) 
  (h3 : a^2 + b^2 = c^2) : (1/2) * a * b = 32/9 := by
  sorry

end right_triangle_area_l3298_329897


namespace ice_cream_sales_for_games_l3298_329819

/-- The number of ice creams needed to be sold to buy two games -/
def ice_creams_needed (game_cost : ℕ) (ice_cream_price : ℕ) : ℕ :=
  2 * game_cost / ice_cream_price

/-- Proof that 24 ice creams are needed to buy two $60 games when each ice cream is $5 -/
theorem ice_cream_sales_for_games : ice_creams_needed 60 5 = 24 := by
  sorry

end ice_cream_sales_for_games_l3298_329819


namespace only_zero_solution_l3298_329891

theorem only_zero_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end only_zero_solution_l3298_329891


namespace solutions_of_equation_l3298_329899

theorem solutions_of_equation : 
  {z : ℂ | z^6 - 9*z^3 + 8 = 0} = {2, 1} := by sorry

end solutions_of_equation_l3298_329899


namespace product_of_three_integers_summing_to_seven_l3298_329801

theorem product_of_three_integers_summing_to_seven (a b c : ℕ) :
  a > 0 → b > 0 → c > 0 →
  a ≠ b → b ≠ c → a ≠ c →
  a + b + c = 7 →
  a * b * c = 8 := by
sorry

end product_of_three_integers_summing_to_seven_l3298_329801


namespace expression_equality_l3298_329857

theorem expression_equality : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_equality_l3298_329857


namespace worker_problem_l3298_329875

/-- The number of workers in the problem -/
def num_workers : ℕ := 30

/-- The total amount of money -/
def total_money : ℤ := 5 * num_workers + 30

theorem worker_problem :
  (total_money - 5 * num_workers = 30) ∧
  (total_money - 7 * num_workers = -30) :=
by sorry

end worker_problem_l3298_329875


namespace sum_of_numbers_l3298_329862

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 375) (h4 : 1 / x + 1 / y = 0.10666666666666667) : 
  x + y = 40 := by
  sorry

end sum_of_numbers_l3298_329862


namespace early_movie_savings_l3298_329814

/-- Calculates the savings for going to an earlier movie given ticket and food combo prices and discounts --/
theorem early_movie_savings 
  (evening_ticket_price : ℚ)
  (evening_combo_price : ℚ)
  (ticket_discount_percent : ℚ)
  (combo_discount_percent : ℚ)
  (h1 : evening_ticket_price = 10)
  (h2 : evening_combo_price = 10)
  (h3 : ticket_discount_percent = 20 / 100)
  (h4 : combo_discount_percent = 50 / 100)
  : evening_ticket_price * ticket_discount_percent + 
    evening_combo_price * combo_discount_percent = 7 := by
  sorry

end early_movie_savings_l3298_329814


namespace f_is_odd_l3298_329818

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Theorem: f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end f_is_odd_l3298_329818


namespace exists_decreasing_then_increasing_not_exists_increasing_then_decreasing_l3298_329849

-- Define the sequence type
def PowerSumSequence (originalNumbers : List ℝ) : ℕ → ℝ :=
  λ n => (originalNumbers.map (λ x => x ^ n)).sum

-- Theorem for part (a)
theorem exists_decreasing_then_increasing :
  ∃ (originalNumbers : List ℝ),
    (∀ x ∈ originalNumbers, x > 0) ∧
    (let a := PowerSumSequence originalNumbers
     a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 > a 4 ∧ a 4 > a 5 ∧
     ∀ n ≥ 5, a n < a (n + 1)) := by
  sorry

-- Theorem for part (b)
theorem not_exists_increasing_then_decreasing :
  ¬ ∃ (originalNumbers : List ℝ),
    (∀ x ∈ originalNumbers, x > 0) ∧
    (let a := PowerSumSequence originalNumbers
     a 1 < a 2 ∧ a 2 < a 3 ∧ a 3 < a 4 ∧ a 4 < a 5 ∧
     ∀ n ≥ 5, a n > a (n + 1)) := by
  sorry

end exists_decreasing_then_increasing_not_exists_increasing_then_decreasing_l3298_329849


namespace not_p_and_q_implies_not_p_or_not_q_l3298_329823

theorem not_p_and_q_implies_not_p_or_not_q (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end not_p_and_q_implies_not_p_or_not_q_l3298_329823


namespace rational_equation_implies_c_zero_l3298_329843

theorem rational_equation_implies_c_zero (a b c : ℚ) 
  (h : (a + b + c) * (a + b - c) = 2 * c^2) : c = 0 := by
  sorry

end rational_equation_implies_c_zero_l3298_329843


namespace min_value_expression_l3298_329847

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) ≥ 24 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 6 ∧ (9 / a₀ + 16 / b₀ + 25 / c₀) = 24 :=
by sorry

end min_value_expression_l3298_329847


namespace correct_operation_l3298_329845

theorem correct_operation (a : ℝ) : 3 * a^2 - 4 * a^2 = -a^2 := by
  sorry

end correct_operation_l3298_329845


namespace quadratic_points_relationship_l3298_329861

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * (x + 1)^2 - 8

-- Define the points on the graph
def y₁ : ℝ := f 1
def y₂ : ℝ := f 2
def y₃ : ℝ := f (-2)

-- Theorem statement
theorem quadratic_points_relationship : y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end quadratic_points_relationship_l3298_329861


namespace feathers_count_l3298_329825

/-- The number of animals in the first group -/
def group1_animals : ℕ := 934

/-- The number of feathers in crowns for the first group -/
def group1_feathers : ℕ := 7

/-- The number of animals in the second group -/
def group2_animals : ℕ := 425

/-- The number of colored feathers in crowns for the second group -/
def group2_colored_feathers : ℕ := 7

/-- The number of golden feathers in crowns for the second group -/
def group2_golden_feathers : ℕ := 5

/-- The number of animals in the third group -/
def group3_animals : ℕ := 289

/-- The number of colored feathers in crowns for the third group -/
def group3_colored_feathers : ℕ := 4

/-- The number of golden feathers in crowns for the third group -/
def group3_golden_feathers : ℕ := 10

/-- The total number of feathers needed for all animals -/
def total_feathers : ℕ := 15684

theorem feathers_count :
  group1_animals * group1_feathers +
  group2_animals * (group2_colored_feathers + group2_golden_feathers) +
  group3_animals * (group3_colored_feathers + group3_golden_feathers) =
  total_feathers := by
  sorry

end feathers_count_l3298_329825


namespace geometric_sequence_third_term_l3298_329850

/-- Given a geometric sequence with first term 2 and fifth term 18, the third term is 6 -/
theorem geometric_sequence_third_term :
  ∀ (x y z : ℝ), 
  (∃ q : ℝ, q ≠ 0 ∧ x = 2 * q ∧ y = 2 * q^2 ∧ z = 2 * q^3 ∧ 18 = 2 * q^4) →
  y = 6 :=
by sorry

end geometric_sequence_third_term_l3298_329850


namespace game_draw_probability_l3298_329870

theorem game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3)
  (h_not_lose : p_not_lose = 0.8) :
  p_not_lose - p_win = 0.5 := by
sorry

end game_draw_probability_l3298_329870


namespace jack_marbles_remaining_l3298_329833

/-- 
Given that Jack starts with a certain number of marbles and shares some with Rebecca,
this theorem proves how many marbles Jack ends up with.
-/
theorem jack_marbles_remaining (initial : ℕ) (shared : ℕ) (remaining : ℕ) 
  (h1 : initial = 62)
  (h2 : shared = 33)
  (h3 : remaining = initial - shared) : 
  remaining = 29 := by
  sorry

end jack_marbles_remaining_l3298_329833


namespace fiftieth_day_previous_year_is_wednesday_l3298_329854

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : Int
  dayNumber : Nat

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek :=
  sorry

theorem fiftieth_day_previous_year_is_wednesday
  (N : Int)
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Friday)
  (h2 : dayOfWeek ⟨N + 1, 150⟩ = DayOfWeek.Friday) :
  dayOfWeek ⟨N - 1, 50⟩ = DayOfWeek.Wednesday :=
sorry

end fiftieth_day_previous_year_is_wednesday_l3298_329854


namespace difference_between_B_and_C_l3298_329876

theorem difference_between_B_and_C (A B C : ℤ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h2 : C - 8 = 1)
  (h3 : A + 5 = B)
  (h4 : A = 9 - 4) :
  B - C = 1 := by
  sorry

end difference_between_B_and_C_l3298_329876


namespace mikes_marbles_l3298_329882

/-- Given that Mike initially has 8 orange marbles and gives 4 to Sam,
    prove that Mike now has 4 orange marbles. -/
theorem mikes_marbles (initial_marbles : ℕ) (marbles_given : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 8 →
  marbles_given = 4 →
  remaining_marbles = initial_marbles - marbles_given →
  remaining_marbles = 4 := by
  sorry

end mikes_marbles_l3298_329882


namespace fraction_problem_l3298_329874

theorem fraction_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
sorry

end fraction_problem_l3298_329874


namespace fraction_equality_l3298_329840

theorem fraction_equality (y : ℝ) (h : y > 0) : (9 * y) / 20 + (3 * y) / 10 = 0.75 * y := by
  sorry

end fraction_equality_l3298_329840


namespace wage_increase_hours_reduction_l3298_329834

/-- Proves that when an employee's hourly wage increases by 10% and they want to maintain
    the same total weekly income, the percent reduction in hours worked is (1 - 1/1.10) * 100% -/
theorem wage_increase_hours_reduction (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let new_wage := 1.1 * w
  let new_hours := h * w / new_wage
  let percent_reduction := (h - new_hours) / h * 100
  percent_reduction = (1 - 1 / 1.1) * 100 := by
  sorry


end wage_increase_hours_reduction_l3298_329834


namespace cone_height_calculation_l3298_329830

theorem cone_height_calculation (cylinder_base_area cone_base_area cylinder_height : ℝ) 
  (h1 : cylinder_base_area * cylinder_height = (1/3) * cone_base_area * cone_height)
  (h2 : cylinder_base_area / cone_base_area = 3/5)
  (h3 : cylinder_height = 8) : 
  cone_height = 14.4 := by
  sorry

#check cone_height_calculation

end cone_height_calculation_l3298_329830


namespace words_with_vowel_count_l3298_329828

/-- The set of all letters used to construct words -/
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of vowels -/
def vowels : Finset Char := {'A', 'E'}

/-- The set of consonants -/
def consonants : Finset Char := letters \ vowels

/-- The length of words we're considering -/
def wordLength : Nat := 5

/-- The number of 5-letter words with at least one vowel -/
def numWordsWithVowel : Nat :=
  letters.card ^ wordLength - consonants.card ^ wordLength

theorem words_with_vowel_count :
  numWordsWithVowel = 6752 := by
  sorry

end words_with_vowel_count_l3298_329828


namespace customer_difference_l3298_329832

theorem customer_difference (initial_customers remaining_customers : ℕ) 
  (h1 : initial_customers = 19) 
  (h2 : remaining_customers = 4) : 
  initial_customers - remaining_customers = 15 := by
sorry

end customer_difference_l3298_329832


namespace shifted_parabola_vertex_l3298_329844

/-- The vertex of a parabola y = 3x^2 shifted 2 units left and 3 units up is at (-2,3) -/
theorem shifted_parabola_vertex :
  let f (x : ℝ) := 3 * (x + 2)^2 + 3
  ∃! (a b : ℝ), (∀ x, f x ≥ f a) ∧ f a = b ∧ a = -2 ∧ b = 3 := by
  sorry

end shifted_parabola_vertex_l3298_329844


namespace circle_line_intersection_l3298_329817

theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 16 ∧ p.1 = 4 :=
by sorry

end circle_line_intersection_l3298_329817


namespace integer_decimal_parts_of_2_plus_sqrt_6_l3298_329877

theorem integer_decimal_parts_of_2_plus_sqrt_6 :
  let x := Int.floor (2 + Real.sqrt 6)
  let y := (2 + Real.sqrt 6) - x
  (x = 4 ∧ y = Real.sqrt 6 - 2 ∧ Real.sqrt (x - 1) = Real.sqrt 3) := by
  sorry

end integer_decimal_parts_of_2_plus_sqrt_6_l3298_329877


namespace sufficient_not_necessary_l3298_329829

theorem sufficient_not_necessary (x y : ℝ) :
  (((x < 0 ∧ y < 0) → (x + y - 4 < 0)) ∧
   ∃ x y : ℝ, (x + y - 4 < 0) ∧ ¬(x < 0 ∧ y < 0)) :=
by sorry

end sufficient_not_necessary_l3298_329829


namespace number_problem_l3298_329806

theorem number_problem : ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end number_problem_l3298_329806


namespace game_winner_l3298_329802

/-- Given a game with three players and three cards, prove who received q marbles in the first round -/
theorem game_winner (p q r : ℕ) (total_rounds : ℕ) : 
  0 < p → p < q → q < r →
  total_rounds > 1 →
  total_rounds * (p + q + r) = 39 →
  2 * p + r = 10 →
  2 * q + p = 9 →
  q = 4 →
  (∃ (x : ℕ), x = total_rounds ∧ x = 3) →
  (∃ (player : String), player = "A" ∧ 
    (∀ (other : String), other ≠ "A" → 
      (other = "B" → (∃ (y : ℕ), y = r ∧ y = 8)) ∧ 
      (other = "C" → (∃ (z : ℕ), z = p ∧ z = 1)))) :=
by sorry

end game_winner_l3298_329802


namespace tangent_line_slope_l3298_329894

/-- A line passing through the origin and tangent to the circle (x - √3)² + (y - 1)² = 1 has a slope of either 0 or √3 -/
theorem tangent_line_slope :
  ∀ k : ℝ,
  (∃ x y : ℝ, y = k * x ∧ (x - Real.sqrt 3)^2 + (y - 1)^2 = 1) →
  (k = 0 ∨ k = Real.sqrt 3) :=
by sorry

end tangent_line_slope_l3298_329894


namespace sale_price_lower_than_original_l3298_329841

theorem sale_price_lower_than_original : ∀ x : ℝ, x > 0 → 0.75 * (1.3 * x) < x := by
  sorry

end sale_price_lower_than_original_l3298_329841


namespace prob_at_least_as_many_females_l3298_329853

/-- The probability of selecting at least as many females as males when randomly choosing 2 students from a group of 5 students with 2 females and 3 males is 7/10. -/
theorem prob_at_least_as_many_females (total : ℕ) (females : ℕ) (males : ℕ) :
  total = 5 →
  females = 2 →
  males = 3 →
  females + males = total →
  (Nat.choose total 2 : ℚ) ≠ 0 →
  (Nat.choose females 2 + Nat.choose females 1 * Nat.choose males 1 : ℚ) / Nat.choose total 2 = 7 / 10 := by
  sorry

#check prob_at_least_as_many_females

end prob_at_least_as_many_females_l3298_329853


namespace collinear_vectors_l3298_329807

/-- Given vectors a and b in ℝ², if a + b is collinear with a, then the second component of a is 1. -/
theorem collinear_vectors (k : ℝ) : 
  let a : Fin 2 → ℝ := ![1, k]
  let b : Fin 2 → ℝ := ![2, 2]
  (∃ (t : ℝ), (a + b) = t • a) → k = 1 := by
  sorry

end collinear_vectors_l3298_329807


namespace largest_difference_l3298_329836

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 1005^1006)
  (hB : B = 1005^1006)
  (hC : C = 1004 * 1005^1005)
  (hD : D = 3 * 1005^1005)
  (hE : E = 1005^1005)
  (hF : F = 1005^1004) :
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) :=
by sorry

end largest_difference_l3298_329836


namespace find_V_l3298_329890

-- Define the relationship between R, V, and W
def relationship (R V W : ℚ) : Prop :=
  ∃ c : ℚ, c ≠ 0 ∧ R * W = c * V

-- State the theorem
theorem find_V : 
  (∃ R₀ V₀ W₀ : ℚ, R₀ = 6 ∧ V₀ = 2 ∧ W₀ = 3 ∧ relationship R₀ V₀ W₀) →
  (∃ R₁ V₁ W₁ : ℚ, R₁ = 25 ∧ W₁ = 5 ∧ relationship R₁ V₁ W₁ ∧ V₁ = 125 / 9) :=
by sorry

end find_V_l3298_329890


namespace vector_operation_proof_l3298_329867

def v1 : ℝ × ℝ × ℝ := (-3, 2, -5)
def v2 : ℝ × ℝ × ℝ := (1, 7, -3)

theorem vector_operation_proof :
  v1 + (2 : ℝ) • v2 = (-1, 16, -11) := by sorry

end vector_operation_proof_l3298_329867


namespace sum_of_a_and_d_l3298_329881

theorem sum_of_a_and_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 := by
sorry

end sum_of_a_and_d_l3298_329881


namespace min_value_a_squared_plus_b_squared_l3298_329831

/-- Given a quadratic function f(x) = x^2 + ax + b - 3 that passes through the point (2,0),
    the minimum value of a^2 + b^2 is 1. -/
theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b - 3 = 0 → x = 2) → 
  (∃ m : ℝ, ∀ a' b' : ℝ, (∀ x : ℝ, x^2 + a'*x + b' - 3 = 0 → x = 2) → a'^2 + b'^2 ≥ m) ∧
  (a^2 + b^2 = 1) :=
by sorry

end min_value_a_squared_plus_b_squared_l3298_329831


namespace pages_left_to_read_l3298_329866

theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read_day1 : ℕ) 
  (pages_read_day2 : ℕ) 
  (h1 : total_pages = 95) 
  (h2 : pages_read_day1 = 18) 
  (h3 : pages_read_day2 = 58) : 
  total_pages - (pages_read_day1 + pages_read_day2) = 19 := by
  sorry

end pages_left_to_read_l3298_329866


namespace rectangular_garden_width_l3298_329880

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 588 →
  width = 14 := by
sorry

end rectangular_garden_width_l3298_329880


namespace james_cryptocurrency_investment_l3298_329896

theorem james_cryptocurrency_investment (C : ℕ) : 
  (C * 15 = 12 * (15 + 15 * (2/3))) → C = 20 := by
  sorry

end james_cryptocurrency_investment_l3298_329896


namespace prob_different_colors_is_three_fourths_l3298_329858

/-- The number of color options for shorts -/
def shorts_colors : ℕ := 3

/-- The number of color options for jerseys -/
def jersey_colors : ℕ := 4

/-- The total number of possible combinations of shorts and jerseys -/
def total_combinations : ℕ := shorts_colors * jersey_colors

/-- The number of combinations where shorts and jerseys have different colors -/
def different_color_combinations : ℕ := shorts_colors * (jersey_colors - 1)

/-- The probability of choosing different colors for shorts and jersey -/
def prob_different_colors : ℚ := different_color_combinations / total_combinations

/-- Theorem stating that the probability of choosing different colors for shorts and jersey is 3/4 -/
theorem prob_different_colors_is_three_fourths : prob_different_colors = 3/4 := by
  sorry

end prob_different_colors_is_three_fourths_l3298_329858


namespace symmetrical_line_sum_l3298_329855

/-- Given a line y = mx + b that is symmetrical to the line x - 3y + 11 = 0
    with respect to the x-axis, prove that m + b = -4 -/
theorem symmetrical_line_sum (m b : ℝ) : 
  (∀ x y, y = m * x + b ↔ x + 3 * y + 11 = 0) → m + b = -4 := by
  sorry

end symmetrical_line_sum_l3298_329855


namespace power_sum_inequality_l3298_329871

theorem power_sum_inequality (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end power_sum_inequality_l3298_329871


namespace standard_deviation_of_dataset_l3298_329800

def dataset : List ℝ := [3, 4, 5, 5, 6, 7]

theorem standard_deviation_of_dataset :
  let n : ℕ := dataset.length
  let mean : ℝ := (dataset.sum) / n
  let variance : ℝ := (dataset.map (fun x => (x - mean)^2)).sum / n
  Real.sqrt variance = Real.sqrt (5/3) := by sorry

end standard_deviation_of_dataset_l3298_329800


namespace indian_teepee_proportion_l3298_329895

/-- Represents the fraction of drawings with a specific combination of person and dwelling -/
structure DrawingFraction :=
  (eskimo_teepee : ℚ)
  (eskimo_igloo : ℚ)
  (indian_igloo : ℚ)
  (indian_teepee : ℚ)

/-- The conditions given in the problem -/
def problem_conditions (df : DrawingFraction) : Prop :=
  df.eskimo_teepee + df.eskimo_igloo + df.indian_igloo + df.indian_teepee = 1 ∧
  df.indian_teepee + df.indian_igloo = 2 * (df.eskimo_teepee + df.eskimo_igloo) ∧
  df.indian_igloo = df.eskimo_teepee ∧
  df.eskimo_igloo = 3 * df.eskimo_teepee

/-- The theorem to be proved -/
theorem indian_teepee_proportion (df : DrawingFraction) :
  problem_conditions df →
  df.indian_teepee / (df.indian_teepee + df.eskimo_teepee) = 7/8 :=
by sorry

end indian_teepee_proportion_l3298_329895


namespace calculation_proof_l3298_329842

theorem calculation_proof : 5^2 * 3 + (7 * 2 - 15) / 3 = 74 + 2/3 := by
  sorry

end calculation_proof_l3298_329842


namespace taxi_speed_theorem_l3298_329888

/-- The speed of the taxi in mph -/
def taxi_speed : ℝ := 60

/-- The speed of the bus in mph -/
def bus_speed : ℝ := taxi_speed - 30

/-- The time difference between the taxi and bus departure in hours -/
def time_difference : ℝ := 3

/-- The time it takes for the taxi to overtake the bus in hours -/
def overtake_time : ℝ := 3

theorem taxi_speed_theorem :
  taxi_speed * overtake_time = (taxi_speed - 30) * (overtake_time + time_difference) :=
by sorry

end taxi_speed_theorem_l3298_329888


namespace curve_equation_k_value_l3298_329809

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) +
  Real.sqrt ((x - 0)^2 + (y + Real.sqrt 3)^2) = 4

-- Define the line that intersects C
def Line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define the perpendicularity condition
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem for the equation of C
theorem curve_equation :
  ∀ x y : ℝ, C x y ↔ x^2 + y^2/4 = 1 :=
sorry

-- Theorem for the value of k
theorem k_value (k : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    C x₁ y₁ ∧ C x₂ y₂ ∧
    Line k x₁ y₁ ∧ Line k x₂ y₂ ∧
    Perpendicular x₁ y₁ x₂ y₂) →
  k = 1/2 ∨ k = -1/2 :=
sorry

end curve_equation_k_value_l3298_329809


namespace hexagon_side_count_l3298_329885

/-- A convex hexagon with two distinct side lengths -/
structure ConvexHexagon where
  side1 : ℕ  -- Length of the first type of side
  side2 : ℕ  -- Length of the second type of side
  count1 : ℕ -- Number of sides with length side1
  count2 : ℕ -- Number of sides with length side2
  distinct : side1 ≠ side2
  total_sides : count1 + count2 = 6
  perimeter : side1 * count1 + side2 * count2 = 38

theorem hexagon_side_count (h : ConvexHexagon) (h_side1 : h.side1 = 7) (h_side2 : h.side2 = 4) :
  h.count2 = 1 := by
  sorry

end hexagon_side_count_l3298_329885


namespace trigonometric_equality_l3298_329892

theorem trigonometric_equality : 2 * Real.tan (π / 3) + Real.tan (π / 4) - 4 * Real.cos (π / 6) = 1 := by
  sorry

end trigonometric_equality_l3298_329892


namespace all_natural_numbers_have_P_structure_l3298_329884

/-- The set of all squares of positive integers -/
def P : Set ℕ := {n : ℕ | ∃ k : ℕ+, n = k^2}

/-- A number n has a P structure if it can be expressed as a sum of some distinct elements from P -/
def has_P_structure (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), (∀ s ∈ S, s ∈ P) ∧ (S.sum id = n)

/-- Every natural number has a P structure -/
theorem all_natural_numbers_have_P_structure :
  ∀ n : ℕ, has_P_structure n :=
sorry

end all_natural_numbers_have_P_structure_l3298_329884


namespace expression_simplification_l3298_329864

theorem expression_simplification (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y^2) - 5 * (2 + 3 * y) = -4 * y^2 - 17 * y - 8 := by
  sorry

end expression_simplification_l3298_329864


namespace five_integers_sum_20_product_420_l3298_329804

theorem five_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    a + b + c + d + e = 20 ∧
    a * b * c * d * e = 420 :=
by
  sorry

end five_integers_sum_20_product_420_l3298_329804


namespace triangle_folding_theorem_l3298_329878

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Represents a folding method for a triangle -/
structure FoldingMethod where
  apply : Triangle → ℕ

/-- Represents the result of applying a folding method to a triangle -/
structure FoldedTriangle where
  original : Triangle
  method : FoldingMethod
  layers : ℕ

/-- A folded triangle has uniform thickness if all points have the same number of layers -/
def hasUniformThickness (ft : FoldedTriangle) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ ft.original.a :: ft.original.b :: ft.original.c :: [] → 
    ft.method.apply ft.original = ft.layers

theorem triangle_folding_theorem :
  ∀ t : Triangle, ∃ fm : FoldingMethod, 
    let ft := FoldedTriangle.mk t fm 2020
    hasUniformThickness ft ∧ ft.layers = 2020 := by
  sorry

end triangle_folding_theorem_l3298_329878


namespace sqrt_15_simplest_l3298_329805

-- Define what it means for a square root to be in its simplest form
def is_simplest_sqrt (n : ℝ) : Prop :=
  ∀ (a b : ℝ), a > 0 ∧ b > 0 → n ≠ a * b^2

-- Theorem statement
theorem sqrt_15_simplest : is_simplest_sqrt 15 := by
  sorry

end sqrt_15_simplest_l3298_329805


namespace max_xy_collinear_vectors_l3298_329898

def vector_a (x : ℝ) : ℝ × ℝ := (1, x^2)
def vector_b (y : ℝ) : ℝ × ℝ := (-2, y^2 - 2)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (k * a.1 = b.1 ∧ k * a.2 = b.2)

theorem max_xy_collinear_vectors (x y : ℝ) :
  collinear (vector_a x) (vector_b y) →
  x * y ≤ Real.sqrt 2 / 2 :=
sorry

end max_xy_collinear_vectors_l3298_329898


namespace mark_kate_difference_l3298_329860

/-- Represents the hours charged by each person on the project -/
structure ProjectHours where
  kate : ℝ
  pat : ℝ
  ravi : ℝ
  sarah : ℝ
  mark : ℝ

/-- Defines the conditions of the project hours problem -/
def validProjectHours (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.ravi = 1.5 * h.kate ∧
  h.sarah = 4 * h.ravi ∧
  h.sarah = 2/3 * h.mark ∧
  h.kate + h.pat + h.ravi + h.sarah + h.mark = 310

/-- Theorem stating the difference between Mark's and Kate's hours -/
theorem mark_kate_difference (h : ProjectHours) (hvalid : validProjectHours h) :
  ∃ ε > 0, |h.mark - h.kate - 127.2| < ε :=
sorry

end mark_kate_difference_l3298_329860


namespace exam_pass_percentage_l3298_329889

/-- Calculates the pass percentage for a group of students -/
def passPercentage (totalStudents : ℕ) (passedStudents : ℕ) : ℚ :=
  (passedStudents : ℚ) / (totalStudents : ℚ) * 100

theorem exam_pass_percentage :
  let set1 := 40
  let set2 := 50
  let set3 := 60
  let pass1 := 40  -- 100% of 40
  let pass2 := 45  -- 90% of 50
  let pass3 := 48  -- 80% of 60
  let totalStudents := set1 + set2 + set3
  let totalPassed := pass1 + pass2 + pass3
  abs (passPercentage totalStudents totalPassed - 88.67) < 0.01 := by
  sorry

#eval passPercentage (40 + 50 + 60) (40 + 45 + 48)

end exam_pass_percentage_l3298_329889


namespace resulting_angle_25_2_5_turns_l3298_329887

/-- Given an initial angle and a number of clockwise turns, calculate the resulting angle -/
def resulting_angle (initial_angle : ℝ) (clockwise_turns : ℝ) : ℝ :=
  initial_angle - 360 * clockwise_turns

/-- Theorem: The resulting angle after rotating 25° clockwise by 2.5 turns is -875° -/
theorem resulting_angle_25_2_5_turns :
  resulting_angle 25 2.5 = -875 := by
  sorry

end resulting_angle_25_2_5_turns_l3298_329887


namespace train_speed_calculation_l3298_329886

/-- The speed of a train given the lengths of two trains, the speed of one train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length1 length2 speed1 time_to_cross : ℝ) :
  length1 = 280 →
  length2 = 220.04 →
  speed1 = 120 →
  time_to_cross = 9 →
  ∃ speed2 : ℝ, 
    (length1 + length2) = (speed1 + speed2) * (5 / 18) * time_to_cross ∧ 
    abs (speed2 - 80.016) < 0.001 := by
  sorry

end train_speed_calculation_l3298_329886


namespace circular_field_diameter_circular_field_diameter_approx_42_l3298_329808

/-- The diameter of a circular field given the fencing cost per meter and total fencing cost -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- Proof that the diameter of the circular field is approximately 42 meters -/
theorem circular_field_diameter_approx_42 :
  ∃ ε > 0, |circular_field_diameter 5 659.73 - 42| < ε :=
sorry

end circular_field_diameter_circular_field_diameter_approx_42_l3298_329808


namespace first_equation_is_root_multiplying_root_multiplying_with_root_two_l3298_329811

/-- A quadratic equation ax^2 + bx + c = 0 is root-multiplying if it has two real roots and one root is twice the other -/
def is_root_multiplying (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

/-- The first part of the theorem -/
theorem first_equation_is_root_multiplying :
  is_root_multiplying 1 (-3) 2 :=
sorry

/-- The second part of the theorem -/
theorem root_multiplying_with_root_two (a b : ℝ) :
  is_root_multiplying a b (-6) ∧ (∃ x : ℝ, a * x^2 + b * x - 6 = 0 ∧ x = 2) →
  (a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9) :=
sorry

end first_equation_is_root_multiplying_root_multiplying_with_root_two_l3298_329811


namespace work_completion_l3298_329810

/-- The number of men in the first group that can complete a work in 18 days, 
    working 7 hours a day, given that 12 men can complete the same work in 12 days,
    also working 7 hours a day. -/
def number_of_men : ℕ := 8

theorem work_completion :
  ∀ (hours_per_day : ℕ) (days_first_group : ℕ) (days_second_group : ℕ),
    hours_per_day > 0 →
    days_first_group > 0 →
    days_second_group > 0 →
    number_of_men * hours_per_day * days_first_group = 12 * hours_per_day * days_second_group →
    hours_per_day = 7 →
    days_first_group = 18 →
    days_second_group = 12 →
    number_of_men = 8 := by
  sorry

end work_completion_l3298_329810


namespace triangle_formation_l3298_329848

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_formation :
  can_form_triangle 13 12 20 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 ∧
  ¬ can_form_triangle 3 4 8 :=
by sorry

end triangle_formation_l3298_329848


namespace no_intersection_l3298_329813

theorem no_intersection : ¬ ∃ x : ℝ, |3 * x + 6| = -|2 * x - 4| := by
  sorry

end no_intersection_l3298_329813


namespace pauline_car_count_l3298_329837

/-- Represents the total number of matchbox cars Pauline has. -/
def total_cars : ℕ := 125

/-- Represents the number of convertible cars Pauline has. -/
def convertibles : ℕ := 35

/-- Represents the percentage of regular cars as a rational number. -/
def regular_cars_percent : ℚ := 64 / 100

/-- Represents the percentage of trucks as a rational number. -/
def trucks_percent : ℚ := 8 / 100

/-- Theorem stating that given the conditions, Pauline has 125 matchbox cars in total. -/
theorem pauline_car_count : 
  (regular_cars_percent + trucks_percent) * total_cars + convertibles = total_cars :=
sorry

end pauline_car_count_l3298_329837


namespace condition_necessary_not_sufficient_l3298_329846

def complex (a b : ℝ) := a + b * Complex.I

theorem condition_necessary_not_sufficient :
  ∃ a b : ℝ, (complex a b)^2 = 2 * Complex.I ∧ (a ≠ 1 ∨ b ≠ 1) ∧
  ∀ a b : ℝ, (complex a b)^2 = 2 * Complex.I → a = 1 ∧ b = 1 :=
sorry

end condition_necessary_not_sufficient_l3298_329846


namespace stating_circle_implies_a_eq_neg_one_l3298_329873

/-- 
A function that represents the equation of a potential circle.
-/
def potential_circle (a : ℝ) (x y : ℝ) : ℝ :=
  x^2 + (a + 2) * y^2 + 2 * a * x + a

/-- 
A predicate that determines if an equation represents a circle.
This is a simplified representation and may need to be adjusted based on the specific criteria for a circle.
-/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), f x y = (x - h)^2 + (y - k)^2 - r^2

/-- 
Theorem stating that if the given equation represents a circle, then a = -1.
-/
theorem circle_implies_a_eq_neg_one :
  is_circle (potential_circle a) → a = -1 := by
  sorry


end stating_circle_implies_a_eq_neg_one_l3298_329873


namespace series_end_probability_l3298_329822

/-- Probability of Mathletes winning a single game -/
def p : ℚ := 2/3

/-- Probability of the opponent winning a single game -/
def q : ℚ := 1 - p

/-- Number of games in the series before the final game -/
def n : ℕ := 6

/-- Number of wins required to end the series -/
def k : ℕ := 5

/-- Probability of the series ending in exactly 7 games -/
def prob_series_end_7 : ℚ := 
  (Nat.choose n (k-1)) * (p^(k-1) * q^(n-(k-1)) * p + p^(n-(k-1)) * q^(k-1) * q)

theorem series_end_probability :
  prob_series_end_7 = 20/81 := by
  sorry

end series_end_probability_l3298_329822


namespace side_length_is_seven_l3298_329835

noncomputable def triangle_side_length (a c : ℝ) (B : ℝ) : ℝ :=
  Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))

theorem side_length_is_seven :
  let a : ℝ := 3 * Real.sqrt 3
  let c : ℝ := 2
  let B : ℝ := 150 * π / 180
  triangle_side_length a c B = 7 := by
sorry

end side_length_is_seven_l3298_329835


namespace sufficient_not_necessary_condition_l3298_329851

/-- A quadratic function f(x) = x^2 + 2ax - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 3

/-- The condition a > -1 is sufficient but not necessary for f to be monotonically increasing on (1, +∞) -/
theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > -1 → ∀ x y, 1 < x → x < y → f a x < f a y) ∧
  ¬(∀ x y, 1 < x → x < y → f a x < f a y → a > -1) :=
sorry

end sufficient_not_necessary_condition_l3298_329851


namespace inequality_solution_l3298_329838

def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then Set.univ
  else if m > 0 then {x | -3/m < x ∧ x < 1/m}
  else {x | 1/m < x ∧ x < -3/m}

theorem inequality_solution (m : ℝ) :
  {x : ℝ | m^2 * x^2 + 2*m*x - 3 < 0} = solution_set m :=
by sorry

end inequality_solution_l3298_329838


namespace average_speed_calculation_l3298_329826

-- Define the variables
def distance_day1 : ℝ := 240
def distance_day2 : ℝ := 420
def time_difference : ℝ := 3

-- Define the theorem
theorem average_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧
  distance_day2 / v = distance_day1 / v + time_difference ∧
  v = 60 := by
  sorry

end average_speed_calculation_l3298_329826


namespace f_iterated_four_times_l3298_329820

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^2 else z^2

theorem f_iterated_four_times :
  f (f (f (f (1 + 2*I)))) = 165633 - 112896*I := by sorry

end f_iterated_four_times_l3298_329820
