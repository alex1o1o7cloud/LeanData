import Mathlib

namespace NUMINAMATH_CALUDE_alley_width_l3796_379646

/-- Given a ladder of length l placed in an alley, touching one wall at a 60° angle
    and the other wall at a 30° angle with the ground, the width w of the alley
    is equal to l(√3 + 1)/2. -/
theorem alley_width (l : ℝ) (h : l > 0) :
  let w := l * (Real.sqrt 3 + 1) / 2
  let angle_A := 60 * π / 180
  let angle_B := 30 * π / 180
  ∃ (m : ℝ), m > 0 ∧ w = l * Real.sin angle_A + l * Real.sin angle_B :=
sorry

end NUMINAMATH_CALUDE_alley_width_l3796_379646


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_parabola_l3796_379602

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 16*x

/-- 
Given a hyperbola with equation x^2 - y^2/3 = 1 and right focus F(2,0),
the standard equation of the parabola with focus F is y^2 = 16x.
-/
theorem hyperbola_focus_to_parabola :
  ∀ x y : ℝ, hyperbola x y → parabola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_to_parabola_l3796_379602


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3796_379624

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log x

theorem tangent_line_equation :
  let p : ℝ × ℝ := (2, f 2)
  let m : ℝ := (deriv f) 2
  let tangent_eq (x y : ℝ) : Prop := x - y + 2 * log 2 - 2 = 0
  tangent_eq p.1 p.2 ∧ ∀ x y, tangent_eq x y ↔ y - p.2 = m * (x - p.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3796_379624


namespace NUMINAMATH_CALUDE_largest_integer_in_inequality_l3796_379614

theorem largest_integer_in_inequality : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/11 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11) → y ≤ x :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_inequality_l3796_379614


namespace NUMINAMATH_CALUDE_total_candy_cases_is_80_l3796_379668

/-- The Sweet Shop gets a new candy shipment every 35 days. -/
def shipment_interval : ℕ := 35

/-- The number of cases of chocolate bars. -/
def chocolate_cases : ℕ := 25

/-- The number of cases of lollipops. -/
def lollipop_cases : ℕ := 55

/-- The total number of candy cases. -/
def total_candy_cases : ℕ := chocolate_cases + lollipop_cases

/-- Theorem stating that the total number of candy cases is 80. -/
theorem total_candy_cases_is_80 : total_candy_cases = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_cases_is_80_l3796_379668


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3796_379605

theorem smallest_angle_in_triangle (a b c : ℝ) (C : ℝ) : 
  a = 2 →
  b = 2 →
  c ≥ 4 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  C ≥ 120 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3796_379605


namespace NUMINAMATH_CALUDE_second_fraction_base_l3796_379630

theorem second_fraction_base (x k : ℝ) (h1 : (1/2)^18 * (1/x)^k = 1/18^18) (h2 : k = 9) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_second_fraction_base_l3796_379630


namespace NUMINAMATH_CALUDE_smallest_integer_l3796_379656

theorem smallest_integer (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 3) →
  (Nat.lcm a b = x * (x + 3)) →
  (b = 36) →
  (∀ y : ℕ+, y < x → ¬(∃ c : ℕ+, 
    Nat.gcd c 36 = y + 3 ∧ 
    Nat.lcm c 36 = y * (y + 3))) →
  (a = 108) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_l3796_379656


namespace NUMINAMATH_CALUDE_range_of_p_l3796_379617

-- Define the set A
def A (p : ℝ) : Set ℝ := {x : ℝ | |x| * x^2 + (p + 2) * x + 1 = 0}

-- Define the theorem
theorem range_of_p (p : ℝ) :
  (A p ∩ Set.Ici 0 = ∅) ↔ -4 < p ∧ p < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l3796_379617


namespace NUMINAMATH_CALUDE_vacation_costs_l3796_379631

/-- Vacation expenses for Xiao Ming's family --/
structure VacationExpenses where
  tickets : ℕ
  meals : ℕ
  accommodation : ℕ

/-- The actual expenses for Xiao Ming's family's vacation --/
def actualExpenses : VacationExpenses :=
  { tickets := 456
  , meals := 385
  , accommodation := 396 }

/-- The total cost of the vacation --/
def totalCost (e : VacationExpenses) : ℕ :=
  e.tickets + e.meals + e.accommodation

/-- The approximate total cost of the vacation --/
def approximateTotalCost (e : VacationExpenses) : ℕ :=
  ((e.tickets + 50) / 100 * 100) +
  ((e.meals + 50) / 100 * 100) +
  ((e.accommodation + 50) / 100 * 100)

theorem vacation_costs (e : VacationExpenses) 
  (h : e = actualExpenses) : 
  approximateTotalCost e = 1300 ∧ totalCost e = 1237 := by
  sorry

end NUMINAMATH_CALUDE_vacation_costs_l3796_379631


namespace NUMINAMATH_CALUDE_investment_ratio_is_two_thirds_l3796_379629

/-- Represents the investments and profit shares of three partners A, B, and C. -/
structure Partnership where
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  b_profit_share : ℝ

/-- Theorem stating that under given conditions, the ratio of B's investment to C's investment is 2:3 -/
theorem investment_ratio_is_two_thirds (p : Partnership) 
  (h1 : p.b_investment > 0)
  (h2 : p.c_investment > 0)
  (h3 : p.total_profit = 4400)
  (h4 : p.b_profit_share = 800)
  (h5 : 3 * p.b_investment = p.b_investment + p.c_investment) :
  p.b_investment / p.c_investment = 2 / 3 := by
  sorry

#check investment_ratio_is_two_thirds

end NUMINAMATH_CALUDE_investment_ratio_is_two_thirds_l3796_379629


namespace NUMINAMATH_CALUDE_weight_difference_l3796_379659

/-- The weight difference between two metal pieces -/
theorem weight_difference (iron_weight aluminum_weight : ℝ) 
  (h1 : iron_weight = 11.17)
  (h2 : aluminum_weight = 0.83) : 
  iron_weight - aluminum_weight = 10.34 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3796_379659


namespace NUMINAMATH_CALUDE_log_stack_sum_l3796_379620

theorem log_stack_sum (n : ℕ) (a l : ℕ) (h1 : n = 12) (h2 : a = 4) (h3 : l = 15) :
  n * (a + l) / 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l3796_379620


namespace NUMINAMATH_CALUDE_lowest_class_size_class_size_120_lowest_class_size_is_120_l3796_379641

theorem lowest_class_size (n : ℕ) : n > 0 ∧ 6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n → n ≥ 120 := by
  sorry

theorem class_size_120 : 6 ∣ 120 ∧ 8 ∣ 120 ∧ 12 ∣ 120 ∧ 15 ∣ 120 := by
  sorry

theorem lowest_class_size_is_120 : ∃! n : ℕ, n > 0 ∧ 6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 6 ∣ m ∧ 8 ∣ m ∧ 12 ∣ m ∧ 15 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_lowest_class_size_class_size_120_lowest_class_size_is_120_l3796_379641


namespace NUMINAMATH_CALUDE_hyperbola_satisfies_conditions_l3796_379684

/-- A hyperbola with the equation 4x² - 9y² = -32 -/
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = -32

/-- The asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop := (2 * x + 3 * y = 0) ∨ (2 * x - 3 * y = 0)

theorem hyperbola_satisfies_conditions :
  (∀ x y : ℝ, asymptotes x y ↔ (4 * x^2 - 9 * y^2 = 0)) ∧
  hyperbola 1 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_satisfies_conditions_l3796_379684


namespace NUMINAMATH_CALUDE_three_intersection_points_k_value_l3796_379623

/-- Curve C1 -/
def C1 (k : ℝ) (x y : ℝ) : Prop :=
  y = k * abs x + 2

/-- Curve C2 -/
def C2 (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4

/-- Number of intersection points between C1 and C2 -/
def intersectionPoints (k : ℝ) : ℕ :=
  sorry -- This would require a complex implementation to count intersection points

theorem three_intersection_points_k_value :
  ∀ k : ℝ, intersectionPoints k = 3 → k = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_three_intersection_points_k_value_l3796_379623


namespace NUMINAMATH_CALUDE_average_first_10_even_numbers_l3796_379666

theorem average_first_10_even_numbers : 
  let first_10_even : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
  (first_10_even.sum / first_10_even.length : ℚ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_first_10_even_numbers_l3796_379666


namespace NUMINAMATH_CALUDE_wednesday_production_l3796_379673

/-- The number of clay pots Nancy created on each day of the week --/
structure ClayPotProduction where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- The conditions of Nancy's clay pot production --/
def nancysProduction : ClayPotProduction where
  monday := 12
  tuesday := 12 * 2
  wednesday := 50 - (12 + 12 * 2)

/-- Theorem stating that Nancy created 14 clay pots on Wednesday --/
theorem wednesday_production : nancysProduction.wednesday = 14 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_production_l3796_379673


namespace NUMINAMATH_CALUDE_inequality_1_solution_inequality_2_solution_l3796_379653

-- Define the solution sets
def solution_set_1 : Set ℝ := {x | -2 < x ∧ x < 1}
def solution_set_2 : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem for the first inequality
theorem inequality_1_solution (x : ℝ) : 
  |2*x + 1| < 3 ↔ x ∈ solution_set_1 :=
sorry

-- Theorem for the second inequality
theorem inequality_2_solution (x : ℝ) :
  |x - 2| + |x - 3| > 3 ↔ x ∈ solution_set_2 :=
sorry

end NUMINAMATH_CALUDE_inequality_1_solution_inequality_2_solution_l3796_379653


namespace NUMINAMATH_CALUDE_cut_square_theorem_l3796_379622

/-- Represents the dimensions of the original square -/
def original_size : ℕ := 8

/-- Represents the total length of cuts -/
def total_cut_length : ℕ := 54

/-- Represents the width of a rectangular piece -/
def rect_width : ℕ := 1

/-- Represents the length of a rectangular piece -/
def rect_length : ℕ := 4

/-- Represents the side length of a square piece -/
def square_side : ℕ := 2

/-- Represents the perimeter of the original square -/
def original_perimeter : ℕ := 4 * original_size

/-- Represents the total number of cells in the original square -/
def total_cells : ℕ := original_size * original_size

/-- Represents the number of cells covered by each piece (both rectangle and square) -/
def cells_per_piece : ℕ := square_side * square_side

theorem cut_square_theorem (num_rectangles num_squares : ℕ) :
  (num_rectangles + num_squares = total_cells / cells_per_piece) ∧
  (2 * total_cut_length + original_perimeter = 
   num_rectangles * (2 * (rect_width + rect_length)) + 
   num_squares * (4 * square_side)) →
  num_rectangles = 6 ∧ num_squares = 10 := by
  sorry

end NUMINAMATH_CALUDE_cut_square_theorem_l3796_379622


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3796_379642

theorem isosceles_triangle_side_length 
  (equilateral_side : ℝ) 
  (isosceles_base : ℝ) 
  (equilateral_area : ℝ) 
  (isosceles_area : ℝ) :
  equilateral_side = 1 →
  isosceles_base = 1/3 →
  equilateral_area = Real.sqrt 3 / 4 →
  isosceles_area = equilateral_area / 3 →
  ∃ (isosceles_side : ℝ), 
    isosceles_side = Real.sqrt 3 / 3 ∧ 
    isosceles_side^2 = (isosceles_base/2)^2 + (2 * isosceles_area / isosceles_base)^2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3796_379642


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l3796_379626

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number :
  triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l3796_379626


namespace NUMINAMATH_CALUDE_washington_goat_count_l3796_379698

/-- The number of goats Washington has -/
def washington_goats : ℕ := 140

/-- The number of goats Paddington has -/
def paddington_goats : ℕ := washington_goats + 40

/-- The total number of goats -/
def total_goats : ℕ := 320

theorem washington_goat_count : washington_goats = 140 :=
  by sorry

end NUMINAMATH_CALUDE_washington_goat_count_l3796_379698


namespace NUMINAMATH_CALUDE_promotions_recipients_l3796_379672

def stadium_capacity : ℕ := 4500
def cap_interval : ℕ := 90
def shirt_interval : ℕ := 45
def sunglasses_interval : ℕ := 60

theorem promotions_recipients : 
  (∀ n : ℕ, n ≤ stadium_capacity → 
    (n % cap_interval = 0 ∧ n % shirt_interval = 0 ∧ n % sunglasses_interval = 0) ↔ 
    n % 180 = 0) →
  (stadium_capacity / 180 = 25) :=
by sorry

end NUMINAMATH_CALUDE_promotions_recipients_l3796_379672


namespace NUMINAMATH_CALUDE_tree_planting_equation_holds_l3796_379658

/-- Represents the tree planting project with increased efficiency -/
structure TreePlantingProject where
  total_trees : ℕ
  efficiency_increase : ℝ
  days_ahead : ℕ
  trees_per_day : ℝ

/-- The equation holds for the given tree planting project -/
theorem tree_planting_equation_holds (project : TreePlantingProject) 
  (h1 : project.total_trees = 20000)
  (h2 : project.efficiency_increase = 0.25)
  (h3 : project.days_ahead = 5) :
  project.total_trees / project.trees_per_day - 
  project.total_trees / (project.trees_per_day * (1 + project.efficiency_increase)) = 
  project.days_ahead := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_equation_holds_l3796_379658


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l3796_379611

theorem geometric_sequence_eighth_term 
  (a : ℝ) (r : ℝ) 
  (positive_sequence : ∀ n : ℕ, a * r^n > 0)
  (fourth_term : a * r^3 = 12)
  (twelfth_term : a * r^11 = 3) :
  a * r^7 = 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l3796_379611


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3796_379683

theorem trigonometric_inequality (θ : Real) (h : 0 < θ ∧ θ < π/4) :
  3 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 3 * Real.tan θ + 2 * Real.sin θ ≥ 4 * (3 * Real.sqrt 3) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3796_379683


namespace NUMINAMATH_CALUDE_probability_is_one_fourth_l3796_379660

/-- Represents a right triangle XYZ with given side lengths -/
structure RightTriangle where
  xy : ℝ
  xz : ℝ
  angle_x_is_right : xy > 0 ∧ xz > 0

/-- Calculates the probability of a randomly chosen point P inside the right triangle XYZ
    forming a triangle PYZ with an area less than one-third of the area of XYZ -/
def probability_small_area (t : RightTriangle) : ℝ :=
  sorry

/-- The main theorem stating that for a right triangle with sides 6 and 8,
    the probability of forming a smaller triangle with area less than one-third
    of the original triangle's area is 1/4 -/
theorem probability_is_one_fourth :
  let t : RightTriangle := ⟨6, 8, by norm_num⟩
  probability_small_area t = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_is_one_fourth_l3796_379660


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_6_l3796_379689

theorem largest_integer_less_than_100_with_remainder_5_mod_6 :
  ∀ n : ℕ, n < 100 ∧ n % 6 = 5 → n ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_6_l3796_379689


namespace NUMINAMATH_CALUDE_lemons_removed_equals_30_l3796_379616

/-- The number of lemons Sally picked -/
def sally_lemons : ℕ := 7

/-- The number of lemons Mary picked -/
def mary_lemons : ℕ := 9

/-- The number of lemons Tom picked -/
def tom_lemons : ℕ := 12

/-- The number of lemons that fell and were eaten by animals -/
def fallen_lemons : ℕ := 2

/-- The total number of lemons removed from the tree -/
def total_removed_lemons : ℕ := sally_lemons + mary_lemons + tom_lemons + fallen_lemons

theorem lemons_removed_equals_30 : total_removed_lemons = 30 := by
  sorry

end NUMINAMATH_CALUDE_lemons_removed_equals_30_l3796_379616


namespace NUMINAMATH_CALUDE_pigeonhole_principle_on_sequence_l3796_379696

theorem pigeonhole_principle_on_sequence (n : ℕ) : 
  ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 2*n ∧ (i + i) % (2*n) = (j + j) % (2*n) := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_on_sequence_l3796_379696


namespace NUMINAMATH_CALUDE_apples_for_pies_l3796_379674

/-- Calculates the number of apples needed to make a given number of pies. -/
def apples_needed (apples_per_pie : ℝ) (num_pies : ℕ) : ℝ :=
  apples_per_pie * (num_pies : ℝ)

/-- Theorem stating that 504 apples are needed to make 126 pies,
    given that it takes 4.0 apples to make 1.0 pie. -/
theorem apples_for_pies :
  apples_needed 4.0 126 = 504 := by
  sorry

end NUMINAMATH_CALUDE_apples_for_pies_l3796_379674


namespace NUMINAMATH_CALUDE_passenger_catches_train_l3796_379600

/-- Represents the problem of a passenger trying to catch a train --/
theorem passenger_catches_train 
  (train_delay : ℝ) 
  (train_speed : ℝ) 
  (distance : ℝ) 
  (train_stop_time : ℝ) 
  (passenger_delay : ℝ) 
  (passenger_speed : ℝ) 
  (h1 : train_delay = 11) 
  (h2 : train_speed = 10) 
  (h3 : distance = 1.5) 
  (h4 : train_stop_time = 14.5) 
  (h5 : passenger_delay = 12) 
  (h6 : passenger_speed = 4) :
  passenger_delay + distance / passenger_speed * 60 ≤ 
  train_delay + distance / train_speed * 60 + train_stop_time := by
  sorry

#check passenger_catches_train

end NUMINAMATH_CALUDE_passenger_catches_train_l3796_379600


namespace NUMINAMATH_CALUDE_f_min_value_solution_set_characterization_l3796_379650

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem 1: The minimum value of f(x) is -3
theorem f_min_value : ∀ x : ℝ, f x ≥ -3 := by sorry

-- Theorem 2: Characterization of the solution set for the inequality
theorem solution_set_characterization :
  ∀ x : ℝ, x^2 - 8*x + 15 + f x < 0 ↔ (5 - Real.sqrt 3 < x ∧ x < 5) ∨ (5 < x ∧ x < 6) := by sorry

end NUMINAMATH_CALUDE_f_min_value_solution_set_characterization_l3796_379650


namespace NUMINAMATH_CALUDE_mary_potatoes_l3796_379649

/-- The number of potatoes Mary initially had -/
def initial_potatoes : ℕ := 8

/-- The number of potatoes eaten by rabbits -/
def eaten_potatoes : ℕ := 3

/-- The number of potatoes Mary has now -/
def remaining_potatoes : ℕ := initial_potatoes - eaten_potatoes

theorem mary_potatoes : remaining_potatoes = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_potatoes_l3796_379649


namespace NUMINAMATH_CALUDE_sneeze_interval_l3796_379663

/-- Given a sneezing fit lasting 2 minutes with 40 sneezes in total, 
    prove that the time between each sneeze is 3 seconds. -/
theorem sneeze_interval (duration_minutes : ℕ) (total_sneezes : ℕ) 
  (h1 : duration_minutes = 2) 
  (h2 : total_sneezes = 40) : 
  (duration_minutes * 60) / total_sneezes = 3 := by
  sorry

end NUMINAMATH_CALUDE_sneeze_interval_l3796_379663


namespace NUMINAMATH_CALUDE_population_change_l3796_379647

/-- Represents the population changes over 5 years -/
structure PopulationChange where
  year1 : Real
  year2 : Real
  year3 : Real
  year4 : Real
  year5 : Real

/-- Calculates the final population given an initial population and population changes -/
def finalPopulation (initialPop : Real) (changes : PopulationChange) : Real :=
  initialPop * (1 + changes.year1) * (1 + changes.year2) * (1 + changes.year3) * (1 + changes.year4) * (1 + changes.year5)

/-- The theorem to be proved -/
theorem population_change (changes : PopulationChange) 
  (h1 : changes.year1 = 0.10)
  (h2 : changes.year2 = -0.08)
  (h3 : changes.year3 = 0.15)
  (h4 : changes.year4 = -0.06)
  (h5 : changes.year5 = 0.12)
  (h6 : finalPopulation 13440 changes = 16875) : 
  ∃ (initialPop : Real), abs (initialPop - 13440) < 1 ∧ finalPopulation initialPop changes = 16875 := by
  sorry

end NUMINAMATH_CALUDE_population_change_l3796_379647


namespace NUMINAMATH_CALUDE_expression_equals_159_l3796_379675

def numerator : List ℕ := [12, 24, 36, 48, 60]
def denominator : List ℕ := [6, 18, 30, 42, 54]

def term (x : ℕ) : ℕ := x^4 + 375

def expression : ℚ :=
  (numerator.map term).prod / (denominator.map term).prod

theorem expression_equals_159 : expression = 159 := by sorry

end NUMINAMATH_CALUDE_expression_equals_159_l3796_379675


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3796_379634

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  is_in_second_quadrant (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3796_379634


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l3796_379682

/-- The number of years until a man's age is twice his son's age -/
def yearsUntilDoubleAge (sonAge manAge : ℕ) : ℕ :=
  if manAge ≤ sonAge then 0
  else (manAge - sonAge)

theorem double_age_in_two_years (sonAge manAge : ℕ) 
  (h1 : manAge = sonAge + 24)
  (h2 : sonAge = 22) :
  yearsUntilDoubleAge sonAge manAge = 2 := by
sorry

end NUMINAMATH_CALUDE_double_age_in_two_years_l3796_379682


namespace NUMINAMATH_CALUDE_exists_same_color_neighbors_l3796_379648

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- Represents a position in the grid -/
structure Position where
  x : Fin 50
  y : Fin 50

/-- Represents the coloring of the grid -/
def GridColoring := Position → Color

/-- Checks if a position is valid in the 50x50 grid -/
def isValidPosition (p : Position) : Prop :=
  p.x < 50 ∧ p.y < 50

/-- Gets the color of a cell at a given position -/
def getColor (g : GridColoring) (p : Position) : Color :=
  g p

/-- Checks if a cell has the same color as its four adjacent cells -/
def hasSameColorNeighbors (g : GridColoring) (p : Position) : Prop :=
  isValidPosition p ∧
  isValidPosition ⟨p.x - 1, p.y⟩ ∧
  isValidPosition ⟨p.x + 1, p.y⟩ ∧
  isValidPosition ⟨p.x, p.y - 1⟩ ∧
  isValidPosition ⟨p.x, p.y + 1⟩ ∧
  getColor g p = getColor g ⟨p.x - 1, p.y⟩ ∧
  getColor g p = getColor g ⟨p.x + 1, p.y⟩ ∧
  getColor g p = getColor g ⟨p.x, p.y - 1⟩ ∧
  getColor g p = getColor g ⟨p.x, p.y + 1⟩

/-- Theorem: There exists a cell with four cells on its sides of the same color -/
theorem exists_same_color_neighbors :
  ∀ (g : GridColoring), ∃ (p : Position), hasSameColorNeighbors g p :=
by sorry

end NUMINAMATH_CALUDE_exists_same_color_neighbors_l3796_379648


namespace NUMINAMATH_CALUDE_theater_performance_duration_l3796_379635

/-- The duration of a theater performance in hours -/
def performance_duration : ℝ := 3

/-- The number of weeks Mark visits the theater -/
def weeks : ℕ := 6

/-- The price per hour for a theater ticket in dollars -/
def price_per_hour : ℝ := 5

/-- The total amount spent on theater visits in dollars -/
def total_spent : ℝ := 90

theorem theater_performance_duration :
  performance_duration * price_per_hour * weeks = total_spent :=
by sorry

end NUMINAMATH_CALUDE_theater_performance_duration_l3796_379635


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l3796_379627

theorem sum_smallest_largest_prime_1_to_50 : ∃ (p q : Nat), 
  (p.Prime ∧ q.Prime) ∧ 
  (∀ r, r.Prime → 1 < r ∧ r ≤ 50 → p ≤ r ∧ r ≤ q) ∧ 
  p + q = 49 := by
sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l3796_379627


namespace NUMINAMATH_CALUDE_bellas_age_l3796_379609

theorem bellas_age : 
  ∀ (bella_age : ℕ), 
  (bella_age + (bella_age + 9) + (bella_age / 2) = 27) → 
  bella_age = 6 := by
sorry

end NUMINAMATH_CALUDE_bellas_age_l3796_379609


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3796_379678

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 4*x + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3796_379678


namespace NUMINAMATH_CALUDE_oldest_child_daily_cheese_is_two_l3796_379690

/-- The number of string cheeses Kelly's oldest child wants every day. -/
def oldest_child_daily_cheese : ℕ := 
  let days_per_week : ℕ := 5
  let total_weeks : ℕ := 4
  let cheeses_per_package : ℕ := 30
  let packages_needed : ℕ := 2
  let youngest_child_daily_cheese : ℕ := 1
  let total_days : ℕ := days_per_week * total_weeks
  let total_cheeses : ℕ := packages_needed * cheeses_per_package
  let youngest_total_cheeses : ℕ := youngest_child_daily_cheese * total_days
  let oldest_total_cheeses : ℕ := total_cheeses - youngest_total_cheeses
  oldest_total_cheeses / total_days

theorem oldest_child_daily_cheese_is_two : oldest_child_daily_cheese = 2 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_daily_cheese_is_two_l3796_379690


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_57_l3796_379693

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define divisibility
def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

-- State the theorem
theorem no_primes_divisible_by_57 :
  ¬∃ p : ℕ, isPrime p ∧ divides 57 p :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_57_l3796_379693


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_is_two_min_value_achieved_l3796_379692

theorem min_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + x*y + 3*y^2 = 10) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + a*b + 3*b^2 = 10 → x^2 - x*y + y^2 ≤ a^2 - a*b + b^2 :=
by sorry

theorem min_value_is_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + x*y + 3*y^2 = 10) : 
  x^2 - x*y + y^2 ≥ 2 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + x*y + 3*y^2 = 10 ∧ x^2 - x*y + y^2 < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_is_two_min_value_achieved_l3796_379692


namespace NUMINAMATH_CALUDE_binary_arithmetic_l3796_379640

-- Define binary numbers as natural numbers
def bin_10110 : ℕ := 22  -- 10110 in binary is 22 in decimal
def bin_1011 : ℕ := 11   -- 1011 in binary is 11 in decimal
def bin_11100 : ℕ := 28  -- 11100 in binary is 28 in decimal
def bin_11101 : ℕ := 29  -- 11101 in binary is 29 in decimal
def bin_100010 : ℕ := 34 -- 100010 in binary is 34 in decimal

-- Define a function to convert a natural number to its binary representation
def to_binary (n : ℕ) : List ℕ := sorry

-- Theorem statement
theorem binary_arithmetic :
  to_binary (bin_10110 + bin_1011 - bin_11100 + bin_11101) = to_binary bin_100010 :=
sorry

end NUMINAMATH_CALUDE_binary_arithmetic_l3796_379640


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3796_379643

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 3 → Odd c → a + b > c → b + c > a → c + a > b → a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3796_379643


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_l3796_379664

theorem sin_50_plus_sqrt3_tan_10 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_l3796_379664


namespace NUMINAMATH_CALUDE_steve_coins_problem_l3796_379610

/-- Represents the number of coins of each type -/
structure CoinCount where
  dimes : ℕ
  nickels : ℕ

/-- Represents the value of coins in cents -/
def coinValue (c : CoinCount) : ℕ := c.dimes * 10 + c.nickels * 5

theorem steve_coins_problem :
  ∃ (c : CoinCount),
    c.dimes + c.nickels = 36 ∧
    coinValue c = 310 ∧
    c.dimes = 26 := by
  sorry

end NUMINAMATH_CALUDE_steve_coins_problem_l3796_379610


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3796_379607

theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![(-3), 1]
  let b : Fin 2 → ℝ := ![x, 6]
  (∀ (i j : Fin 2), i.val + j.val = 1 → a i * b j = 0) →
  x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3796_379607


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l3796_379670

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l3796_379670


namespace NUMINAMATH_CALUDE_unique_fish_count_l3796_379691

/-- The number of unique fish owned by four friends given specific conditions -/
theorem unique_fish_count :
  let micah_fish : ℕ := 7
  let kenneth_fish : ℕ := 3 * micah_fish
  let matthias_fish : ℕ := kenneth_fish - 15
  let gabrielle_fish : ℕ := 2 * (micah_fish + kenneth_fish + matthias_fish)
  let micah_matthias_shared : ℕ := 4
  let kenneth_gabrielle_shared : ℕ := 6
  (micah_fish + kenneth_fish + matthias_fish + gabrielle_fish) - 
  (micah_matthias_shared + kenneth_gabrielle_shared) = 92 :=
by sorry

end NUMINAMATH_CALUDE_unique_fish_count_l3796_379691


namespace NUMINAMATH_CALUDE_pump_x_portion_l3796_379619

/-- Represents the pumping scenario with two pumps -/
structure PumpingScenario where
  total_water : ℝ
  pump_x_rate : ℝ
  pump_y_rate : ℝ

/-- The conditions of the pumping scenario -/
def pumping_conditions (s : PumpingScenario) : Prop :=
  s.pump_x_rate > 0 ∧
  s.pump_y_rate > 0 ∧
  3 * s.pump_x_rate + 3 * (s.pump_x_rate + s.pump_y_rate) = s.total_water ∧
  20 * s.pump_y_rate = s.total_water

/-- The theorem stating that Pump X pumps out 17/40 of the total water in the first 3 hours -/
theorem pump_x_portion (s : PumpingScenario) 
  (h : pumping_conditions s) : 
  3 * s.pump_x_rate = (17 / 40) * s.total_water := by
  sorry

end NUMINAMATH_CALUDE_pump_x_portion_l3796_379619


namespace NUMINAMATH_CALUDE_inequality_proof_l3796_379644

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 + x / y)^3 + (1 + y / x)^3 ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3796_379644


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3796_379636

theorem polygon_sides_count : ∃ n : ℕ, 
  n > 2 ∧ 
  (n * (n - 3) / 2 : ℚ) = 2 * n ∧ 
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3796_379636


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3796_379679

/-- The area of the region between two concentric circles -/
theorem area_between_concentric_circles
  (r : ℝ) -- radius of the inner circle
  (h : r > 0) -- assumption that r is positive
  (width : ℝ) -- width of the region between circles
  (h_width : width = 3 * r - r) -- definition of width
  (h_width_value : width = 4) -- given width value
  : (π * (3 * r)^2 - π * r^2) = 8 * π * r^2 := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3796_379679


namespace NUMINAMATH_CALUDE_second_month_sale_l3796_379688

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def first_month_sale : ℕ := 6535
def third_month_sale : ℕ := 6855
def fourth_month_sale : ℕ := 7230
def fifth_month_sale : ℕ := 6562
def sixth_month_sale : ℕ := 4891

theorem second_month_sale :
  ∃ (second_month_sale : ℕ),
    second_month_sale = average_sale * num_months - 
      (first_month_sale + third_month_sale + fourth_month_sale + 
       fifth_month_sale + sixth_month_sale) ∧
    second_month_sale = 6927 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l3796_379688


namespace NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l3796_379628

theorem sum_reciprocals_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1) :
  4 ≤ (1/a + 1/b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀^2 + b₀^2 = 1 ∧ 1/a₀ + 1/b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l3796_379628


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3796_379686

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : x > y) :
  x + y = 5 ∨ x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3796_379686


namespace NUMINAMATH_CALUDE_certain_number_proof_l3796_379608

theorem certain_number_proof : ∃ X : ℝ, 
  0.8 * X = 0.7 * 60.00000000000001 + 30 ∧ X = 90.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3796_379608


namespace NUMINAMATH_CALUDE_math_contest_problem_l3796_379694

theorem math_contest_problem (a b c d e f g : ℕ) : 
  a + b + c + d + e + f + g = 25 →
  b + d = 2 * (c + d) →
  a = 1 + (e + f + g) →
  a = b + c →
  b = 6 :=
by sorry

end NUMINAMATH_CALUDE_math_contest_problem_l3796_379694


namespace NUMINAMATH_CALUDE_equation_solution_l3796_379655

theorem equation_solution : ∃! x : ℝ, 5 * x - 3 * x = 210 + 6 * (x + 4) ∧ x = -58.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3796_379655


namespace NUMINAMATH_CALUDE_balls_in_boxes_with_empty_l3796_379657

/-- The number of ways to put n distinguishable balls in k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to put n distinguishable balls in k distinguishable boxes
    with at least one box empty -/
def ways_with_empty_box (n : ℕ) (k : ℕ) : ℕ :=
  ways_to_put_balls_in_boxes n k -
  (Nat.choose k 1) * ways_to_put_balls_in_boxes n (k-1) +
  (Nat.choose k 2) * ways_to_put_balls_in_boxes n (k-2) -
  (Nat.choose k 3) * ways_to_put_balls_in_boxes n (k-3)

/-- Theorem: There are 240 ways to put 5 distinguishable balls in 4 distinguishable boxes
    with at least one box remaining empty -/
theorem balls_in_boxes_with_empty : ways_with_empty_box 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_with_empty_l3796_379657


namespace NUMINAMATH_CALUDE_work_completion_theorem_l3796_379680

theorem work_completion_theorem 
  (total_work : ℕ) 
  (days_group1 : ℕ) 
  (men_group1 : ℕ) 
  (days_group2 : ℕ) : 
  men_group1 * days_group1 = total_work → 
  total_work = days_group2 * (total_work / days_group2) → 
  men_group1 = 10 → 
  days_group1 = 35 → 
  days_group2 = 50 → 
  total_work / days_group2 = 7 :=
by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_work_completion_theorem_l3796_379680


namespace NUMINAMATH_CALUDE_handbag_price_adjustment_l3796_379637

/-- Calculates the final price of a handbag after a price increase followed by a discount -/
theorem handbag_price_adjustment (initial_price : ℝ) : 
  initial_price = 50 →
  (initial_price * 1.2) * 0.8 = 48 := by sorry

end NUMINAMATH_CALUDE_handbag_price_adjustment_l3796_379637


namespace NUMINAMATH_CALUDE_smallest_circle_equation_l3796_379613

/-- A parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- A circle with center on the parabola and passing through the focus -/
def CircleOnParabola (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - Focus.1)^2 + (center.2 - Focus.2)^2}

/-- The theorem stating that the circle with smallest radius has equation x^2 + y^2 = 1 -/
theorem smallest_circle_equation :
  ∃ (center : ℝ × ℝ),
    center ∈ Parabola ∧
    Focus ∈ CircleOnParabola center ∧
    (∀ (other_center : ℝ × ℝ),
      other_center ∈ Parabola →
      Focus ∈ CircleOnParabola other_center →
      (center.1 - Focus.1)^2 + (center.2 - Focus.2)^2 ≤ (other_center.1 - Focus.1)^2 + (other_center.2 - Focus.2)^2) →
    CircleOnParabola center = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_equation_l3796_379613


namespace NUMINAMATH_CALUDE_integer_root_implies_specific_m_l3796_379618

/-- Defines a quadratic equation with coefficient m -/
def quadratic_equation (m : ℤ) (x : ℤ) : ℤ := m * x^2 + 2*(m-5)*x + (m-4)

/-- Checks if the equation has an integer root -/
def has_integer_root (m : ℤ) : Prop := ∃ x : ℤ, quadratic_equation m x = 0

/-- The main theorem to be proved -/
theorem integer_root_implies_specific_m :
  ∀ m : ℤ, has_integer_root m → m = -4 ∨ m = 4 ∨ m = -16 := by sorry

end NUMINAMATH_CALUDE_integer_root_implies_specific_m_l3796_379618


namespace NUMINAMATH_CALUDE_number_of_girls_l3796_379676

theorem number_of_girls (total_children happy_children sad_children neutral_children boys happy_boys sad_girls neutral_boys : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : boys = 22)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 10)
  (h9 : happy_children + sad_children + neutral_children = total_children)
  : total_children - boys = 38 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l3796_379676


namespace NUMINAMATH_CALUDE_regular_price_is_100_l3796_379697

/-- The regular price of one bag -/
def regular_price : ℝ := 100

/-- The promotional price of the fourth bag -/
def fourth_bag_price : ℝ := 5

/-- The total cost for four bags -/
def total_cost : ℝ := 305

/-- Theorem stating that the regular price of one bag is $100 -/
theorem regular_price_is_100 :
  3 * regular_price + fourth_bag_price = total_cost :=
by sorry

end NUMINAMATH_CALUDE_regular_price_is_100_l3796_379697


namespace NUMINAMATH_CALUDE_solution_of_system_l3796_379638

theorem solution_of_system (x y : ℚ) :
  (x + 5) / (x - 4) = (x - 7) / (x + 3) ∧ x + y = 20 →
  x = 13 / 19 ∧ y = 367 / 19 := by
sorry


end NUMINAMATH_CALUDE_solution_of_system_l3796_379638


namespace NUMINAMATH_CALUDE_justin_age_proof_l3796_379621

/-- Angelina's age in 5 years -/
def angelina_future_age : ℕ := 40

/-- Number of years until Angelina reaches her future age -/
def years_until_future : ℕ := 5

/-- Age difference between Angelina and Justin -/
def age_difference : ℕ := 4

/-- Justin's current age -/
def justin_current_age : ℕ := angelina_future_age - years_until_future - age_difference

theorem justin_age_proof : justin_current_age = 31 := by
  sorry

end NUMINAMATH_CALUDE_justin_age_proof_l3796_379621


namespace NUMINAMATH_CALUDE_function_periodic_l3796_379681

/-- A function satisfying certain symmetry properties is periodic -/
theorem function_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 3) = f (3 - x))
  (h2 : ∀ x : ℝ, f (x + 11) = f (11 - x)) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
sorry

end NUMINAMATH_CALUDE_function_periodic_l3796_379681


namespace NUMINAMATH_CALUDE_percent_calculation_l3796_379612

theorem percent_calculation (x : ℝ) (h : 0.4 * x = 160) : 0.2 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l3796_379612


namespace NUMINAMATH_CALUDE_lance_cents_l3796_379677

/-- Represents the amount of cents each person has -/
structure Cents where
  lance : ℕ
  margaret : ℕ
  guy : ℕ
  bill : ℕ

/-- The problem statement -/
theorem lance_cents (c : Cents) : 
  c.margaret = 75 → -- Margaret has three-fourths of a dollar (75 cents)
  c.guy = 60 → -- Guy has two quarters (50 cents) and a dime (10 cents)
  c.bill = 60 → -- Bill has six dimes (6 * 10 cents)
  c.lance + c.margaret + c.guy + c.bill = 265 → -- Total combined cents
  c.lance = 70 := by
  sorry


end NUMINAMATH_CALUDE_lance_cents_l3796_379677


namespace NUMINAMATH_CALUDE_chess_tournament_green_teams_l3796_379665

theorem chess_tournament_green_teams (red_players green_players total_players total_teams red_red_teams : ℕ)
  (h1 : red_players = 64)
  (h2 : green_players = 68)
  (h3 : total_players = red_players + green_players)
  (h4 : total_teams = 66)
  (h5 : total_players = 2 * total_teams)
  (h6 : red_red_teams = 20) :
  ∃ green_green_teams : ℕ, green_green_teams = 22 ∧ 
  green_green_teams = total_teams - red_red_teams - (red_players - 2 * red_red_teams) := by
  sorry

#check chess_tournament_green_teams

end NUMINAMATH_CALUDE_chess_tournament_green_teams_l3796_379665


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l3796_379639

theorem smallest_value_of_expression (a b c : ℤ) (ω : ℂ) 
  (h1 : ω^4 = 1) 
  (h2 : ω ≠ 1) 
  (h3 : a = 2*b - c) : 
  ∃ (a₀ b₀ c₀ : ℤ), ∀ (a' b' c' : ℤ), 
    |Complex.abs (a₀ + b₀*ω + c₀*ω^3)| ≤ |Complex.abs (a' + b'*ω + c'*ω^3)| ∧ 
    |Complex.abs (a₀ + b₀*ω + c₀*ω^3)| = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l3796_379639


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3796_379687

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - (1 / 2)⌋ = ⌊x + 3⌋ ↔ x ∈ Set.Icc (3.5 : ℝ) (4.5 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3796_379687


namespace NUMINAMATH_CALUDE_ping_pong_balls_sold_l3796_379671

/-- Calculates the number of ping pong balls sold in a shop -/
theorem ping_pong_balls_sold
  (initial_baseballs : ℕ)
  (initial_ping_pong : ℕ)
  (baseballs_sold : ℕ)
  (total_left : ℕ)
  (h1 : initial_baseballs = 2754)
  (h2 : initial_ping_pong = 1938)
  (h3 : baseballs_sold = 1095)
  (h4 : total_left = 3021)
  (h5 : total_left = initial_baseballs + initial_ping_pong - baseballs_sold - (initial_ping_pong - ping_pong_sold))
  : ping_pong_sold = 576 :=
by
  sorry

#check ping_pong_balls_sold

end NUMINAMATH_CALUDE_ping_pong_balls_sold_l3796_379671


namespace NUMINAMATH_CALUDE_min_sqrt_equality_l3796_379604

theorem min_sqrt_equality {a b c : ℝ} (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a*b + 1)/(a*b*c))) (min (Real.sqrt ((b*c + 1)/(a*b*c))) (Real.sqrt ((c*a + 1)/(a*b*c)))) =
    Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔
  ∃ w : ℝ, w > 0 ∧ a = w^2/(1+(w^2+1)^2) ∧ b = w^2/(1+w^2) ∧ c = 1/(1+w^2) :=
by sorry

end NUMINAMATH_CALUDE_min_sqrt_equality_l3796_379604


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l3796_379662

theorem negation_of_existential_proposition :
  (¬ ∃ n : ℕ, n + 10 / n < 4) ↔ (∀ n : ℕ, n + 10 / n ≥ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l3796_379662


namespace NUMINAMATH_CALUDE_cost_equalization_l3796_379652

theorem cost_equalization (X Y Z : ℝ) (h : X < Y ∧ Y < Z) :
  let E := (X + Y + Z) / 3
  let payment_to_bernardo := E - X - (Z - Y) / 2
  let payment_to_carlos := (Z - Y) / 2
  (X + payment_to_bernardo + payment_to_carlos = E) ∧
  (Y - payment_to_bernardo = E) ∧
  (Z - payment_to_carlos = E) := by
sorry

end NUMINAMATH_CALUDE_cost_equalization_l3796_379652


namespace NUMINAMATH_CALUDE_sum_seven_probability_l3796_379603

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when tossing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to get a sum of 7 when tossing two dice -/
def favorableOutcomes : ℕ := 6

/-- The probability of getting a sum of 7 when tossing two dice -/
def probabilitySumSeven : ℚ := favorableOutcomes / totalOutcomes

theorem sum_seven_probability :
  probabilitySumSeven = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_probability_l3796_379603


namespace NUMINAMATH_CALUDE_ratio_greater_than_one_but_numerator_not_greater_l3796_379601

theorem ratio_greater_than_one_but_numerator_not_greater : 
  ∃ (a b : ℝ), a / b > 1 ∧ ¬(a > b) := by sorry

end NUMINAMATH_CALUDE_ratio_greater_than_one_but_numerator_not_greater_l3796_379601


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3796_379669

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

def domain (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

theorem f_increasing_on_interval :
  ∀ x y, x < y → x < -1 → y < -1 → domain x → domain y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3796_379669


namespace NUMINAMATH_CALUDE_sally_peaches_l3796_379651

/-- The number of peaches Sally picked from the orchard -/
def picked_peaches : ℕ := 42

/-- The total number of peaches at the stand after picking -/
def total_peaches : ℕ := 55

/-- The number of peaches Sally had before picking more -/
def initial_peaches : ℕ := total_peaches - picked_peaches

theorem sally_peaches : initial_peaches = 13 := by
  sorry

end NUMINAMATH_CALUDE_sally_peaches_l3796_379651


namespace NUMINAMATH_CALUDE_station_entry_problem_l3796_379645

/-- The number of ways for n people to enter through k gates, where each gate must have at least one person -/
def enterWays (n k : ℕ) : ℕ :=
  sorry

/-- The condition that the number of people is greater than the number of gates -/
def validInput (n k : ℕ) : Prop :=
  n > k ∧ k > 0

theorem station_entry_problem :
  ∀ n k : ℕ, validInput n k → (n = 5 ∧ k = 3) → enterWays n k = 720 :=
sorry

end NUMINAMATH_CALUDE_station_entry_problem_l3796_379645


namespace NUMINAMATH_CALUDE_prime_divisor_congruent_to_one_l3796_379615

theorem prime_divisor_congruent_to_one (p : ℕ) (hp : Prime p) :
  ∃ q : ℕ, Prime q ∧ q ∣ (p^p - 1) ∧ q % p = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_congruent_to_one_l3796_379615


namespace NUMINAMATH_CALUDE_cone_surface_area_l3796_379654

theorem cone_surface_area (θ : Real) (S_lateral : Real) (S_total : Real) : 
  θ = 2 * Real.pi / 3 →  -- 120° in radians
  S_lateral = 3 * Real.pi →
  S_total = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3796_379654


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l3796_379685

theorem quadratic_roots_nature (x : ℝ) : 
  (x^2 - 6*x + 9 = 0) → (∃ r : ℝ, x = r ∧ x^2 - 6*x + 9 = 0) ∧ 
  (∃! r : ℝ, x^2 - 6*x + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l3796_379685


namespace NUMINAMATH_CALUDE_quadratic_inequality_not_always_negative_l3796_379661

theorem quadratic_inequality_not_always_negative :
  ¬ (∀ x : ℝ, x^2 + x - 1 < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_not_always_negative_l3796_379661


namespace NUMINAMATH_CALUDE_grid_with_sequence_exists_l3796_379667

-- Define the grid type
def Grid := Matrix (Fin 6) (Fin 6) (Fin 4)

-- Define a predicate for valid subgrids
def valid_subgrid (g : Grid) (i j : Fin 2) : Prop :=
  ∀ n : Fin 4, ∃! x y : Fin 2, g (2 * i + x) (2 * j + y) = n

-- Define a predicate for adjacent cells being different
def adjacent_different (g : Grid) : Prop :=
  ∀ i j i' j' : Fin 6, 
    (i = i' ∧ |j - j'| = 1) ∨ 
    (j = j' ∧ |i - i'| = 1) ∨ 
    (|i - i'| = 1 ∧ |j - j'| = 1) → 
    g i j ≠ g i' j'

-- Define the existence of the sequence 3521 in the grid
def sequence_exists (g : Grid) : Prop :=
  ∃ i₁ j₁ i₂ j₂ i₃ j₃ i₄ j₄ : Fin 6,
    g i₁ j₁ = 3 ∧ g i₂ j₂ = 5 ∧ g i₃ j₃ = 2 ∧ g i₄ j₄ = 1

-- The main theorem
theorem grid_with_sequence_exists : 
  ∃ g : Grid, 
    (∀ i j : Fin 2, valid_subgrid g i j) ∧ 
    adjacent_different g ∧
    sequence_exists g :=
sorry

end NUMINAMATH_CALUDE_grid_with_sequence_exists_l3796_379667


namespace NUMINAMATH_CALUDE_rachel_math_homework_l3796_379633

/-- The number of pages of reading homework Rachel had to complete -/
def reading_homework : ℕ := 3

/-- The additional pages of math homework compared to reading homework -/
def additional_math_pages : ℕ := 4

/-- The total number of pages of math homework Rachel had to complete -/
def math_homework : ℕ := reading_homework + additional_math_pages

theorem rachel_math_homework :
  math_homework = 7 :=
by sorry

end NUMINAMATH_CALUDE_rachel_math_homework_l3796_379633


namespace NUMINAMATH_CALUDE_jack_queen_king_prob_in_standard_deck_l3796_379695

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (jacks : ℕ)
  (queens : ℕ)
  (kings : ℕ)

/-- Calculates the probability of drawing a specific card from a deck -/
def draw_probability (n : ℕ) (total : ℕ) : ℚ :=
  n / total

/-- Calculates the probability of drawing a Jack, then a Queen, then a King -/
def jack_queen_king_probability (d : Deck) : ℚ :=
  (draw_probability d.jacks d.total_cards) *
  (draw_probability d.queens (d.total_cards - 1)) *
  (draw_probability d.kings (d.total_cards - 2))

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , jacks := 4
  , queens := 4
  , kings := 4 }

theorem jack_queen_king_prob_in_standard_deck :
  jack_queen_king_probability standard_deck = 8 / 16575 :=
by sorry

end NUMINAMATH_CALUDE_jack_queen_king_prob_in_standard_deck_l3796_379695


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3796_379625

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2

-- Define the properties of function f
def is_quadratic (f : ℝ → ℝ) : Prop := ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

def is_odd_sum (f : ℝ → ℝ) : Prop := ∀ x, f x + g x = -(f (-x) + g (-x))

def has_equal_roots (f : ℝ → ℝ) : Prop := ∃ x : ℝ, f x = 3 * x + 2 ∧ 
  ∀ y : ℝ, f y = 3 * y + 2 → y = x

-- Main theorem
theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h1 : is_quadratic f)
  (h2 : is_odd_sum f)
  (h3 : has_equal_roots f) :
  (∀ x, f x = -x^2 + 3*x + 2) ∧ 
  (∀ x, (3 - Real.sqrt 41) / 4 < x ∧ x < (3 + Real.sqrt 41) / 4 → f x > g x) ∧
  (∃ m n : ℝ, m = -1 ∧ n = 17/8 ∧ 
    (∀ x, f x ∈ Set.Icc (-2) (247/64) ↔ x ∈ Set.Icc m n)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3796_379625


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l3796_379606

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l3796_379606


namespace NUMINAMATH_CALUDE_shaded_fraction_is_one_eighth_l3796_379699

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

theorem shaded_fraction_is_one_eighth (r : Rectangle) 
  (h1 : r.width = 15)
  (h2 : r.height = 20)
  (h3 : ∃ (shaded_area : ℝ), shaded_area = (1/4) * ((1/2) * r.area)) :
  ∃ (shaded_area : ℝ), shaded_area / r.area = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_one_eighth_l3796_379699


namespace NUMINAMATH_CALUDE_planes_perpendicular_parallel_l3796_379632

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_parallel 
  (a b : Line) (α β γ : Plane) 
  (h1 : perpendicular α γ) 
  (h2 : parallel β γ) : 
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_parallel_l3796_379632
