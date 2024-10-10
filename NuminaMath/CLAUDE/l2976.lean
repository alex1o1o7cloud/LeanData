import Mathlib

namespace function_extrema_l2976_297665

open Real

theorem function_extrema (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x => (sin x + a) / sin x
  ∃ (m : ℝ), (∀ x, 0 < x → x < π → f x ≥ m) ∧
  (∀ M : ℝ, ∃ x, 0 < x ∧ x < π ∧ f x > M) := by
  sorry

end function_extrema_l2976_297665


namespace range_of_m_l2976_297617

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 1) →
  m ∈ Set.Icc 2 4 :=
by sorry

end range_of_m_l2976_297617


namespace unique_quadratic_solution_l2976_297679

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 20 * x + c = 0) →  -- exactly one solution
  (a + c = 29) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 4 ∧ c = 25) := by              -- conclusion
sorry

end unique_quadratic_solution_l2976_297679


namespace stock_investment_percentage_l2976_297635

theorem stock_investment_percentage (investment : ℝ) (earnings : ℝ) (percentage : ℝ) :
  investment = 5760 →
  earnings = 1900 →
  percentage = (earnings * 100) / investment →
  percentage = 33 :=
by sorry

end stock_investment_percentage_l2976_297635


namespace set_operations_and_range_l2976_297664

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}
def B : Set ℝ := {x | 2 < x ∧ x < 5}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem set_operations_and_range (a : ℝ) : 
  (A ∪ B = {x | 2 < x ∧ x ≤ 9}) ∧ 
  (B ∩ C a = ∅ → a ≥ 5) := by
  sorry

end set_operations_and_range_l2976_297664


namespace exactlyTwoVisitCount_l2976_297619

/-- Represents a visitor with a visiting frequency -/
structure Visitor where
  frequency : ℕ

/-- Calculates the number of days when exactly two out of three visitors visit -/
def exactlyTwoVisit (v1 v2 v3 : Visitor) (days : ℕ) : ℕ :=
  sorry

theorem exactlyTwoVisitCount :
  let alice : Visitor := ⟨2⟩
  let beatrix : Visitor := ⟨5⟩
  let claire : Visitor := ⟨7⟩
  exactlyTwoVisit alice beatrix claire 365 = 55 := by sorry

end exactlyTwoVisitCount_l2976_297619


namespace two_stamps_theorem_l2976_297648

/-- The cost of a single stamp in dollars -/
def single_stamp_cost : ℚ := 34/100

/-- The cost of three stamps in dollars -/
def three_stamps_cost : ℚ := 102/100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68/100

theorem two_stamps_theorem :
  (single_stamp_cost * 2 = two_stamps_cost) ∧
  (single_stamp_cost * 3 = three_stamps_cost) := by
  sorry

end two_stamps_theorem_l2976_297648


namespace equation_solution_l2976_297640

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => 1/((x - 3)*(x - 4)) + 1/((x - 4)*(x - 5)) + 1/((x - 5)*(x - 6))
  ∀ x : ℝ, f x = 1/8 ↔ x = (9 + Real.sqrt 57)/2 ∨ x = (9 - Real.sqrt 57)/2 :=
by sorry

end equation_solution_l2976_297640


namespace other_communities_count_l2976_297633

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) :
  total = 1500 →
  muslim_percent = 37.5 / 100 →
  hindu_percent = 25.6 / 100 →
  sikh_percent = 8.4 / 100 →
  ↑(round ((1 - (muslim_percent + hindu_percent + sikh_percent)) * total)) = 428 :=
by sorry

end other_communities_count_l2976_297633


namespace cost_of_pens_l2976_297608

/-- Given a box of 150 pens costing $45, prove that the cost of 3600 pens is $1080 -/
theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (total_pens : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  total_pens = 3600 →
  (total_pens : ℚ) / box_size * box_cost = 1080 :=
by
  sorry


end cost_of_pens_l2976_297608


namespace cube_sum_preceding_integers_l2976_297614

theorem cube_sum_preceding_integers : ∃ n : ℤ, n = 6 ∧ n^3 = (n-1)^3 + (n-2)^3 + (n-3)^3 := by
  sorry

end cube_sum_preceding_integers_l2976_297614


namespace alex_total_marbles_l2976_297687

/-- The number of marbles each person has -/
structure MarbleCount where
  lorin_black : ℕ
  jimmy_yellow : ℕ
  alex_black : ℕ
  alex_yellow : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.lorin_black = 4 ∧
  m.jimmy_yellow = 22 ∧
  m.alex_black = 2 * m.lorin_black ∧
  m.alex_yellow = m.jimmy_yellow / 2

/-- The theorem stating that Alex has 19 marbles in total -/
theorem alex_total_marbles (m : MarbleCount) 
  (h : marble_problem m) : m.alex_black + m.alex_yellow = 19 := by
  sorry


end alex_total_marbles_l2976_297687


namespace chord_sum_l2976_297658

/-- Definition of the circle --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 20 = 0

/-- The point (1, -1) lies on the circle --/
axiom point_on_circle : circle_equation 1 (-1)

/-- Definition of the longest chord length --/
def longest_chord_length : ℝ := sorry

/-- Definition of the shortest chord length --/
def shortest_chord_length : ℝ := sorry

/-- Theorem: The sum of the longest and shortest chord lengths is 18 --/
theorem chord_sum :
  longest_chord_length + shortest_chord_length = 18 := by sorry

end chord_sum_l2976_297658


namespace function_value_at_inverse_point_l2976_297616

noncomputable def log_log_2_10 : ℝ := Real.log (Real.log 10 / Real.log 2)

theorem function_value_at_inverse_point 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h : ∀ x, f x = a * x^3 + b * Real.sin x + 4) 
  (h1 : f log_log_2_10 = 5) : 
  f (- log_log_2_10) = 3 := by
sorry

end function_value_at_inverse_point_l2976_297616


namespace shoes_alteration_problem_l2976_297643

theorem shoes_alteration_problem (cost_per_shoe : ℕ) (total_cost : ℕ) (num_pairs : ℕ) :
  cost_per_shoe = 29 →
  total_cost = 986 →
  num_pairs = total_cost / (2 * cost_per_shoe) →
  num_pairs = 17 :=
by sorry

end shoes_alteration_problem_l2976_297643


namespace negation_existence_real_gt_one_l2976_297621

theorem negation_existence_real_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end negation_existence_real_gt_one_l2976_297621


namespace fixed_point_of_linear_function_l2976_297656

theorem fixed_point_of_linear_function (k : ℝ) : 
  2 = k * 1 - k + 2 := by sorry

end fixed_point_of_linear_function_l2976_297656


namespace remaining_family_member_age_l2976_297610

/-- Represents the ages of family members -/
structure FamilyAges where
  total : ℕ
  father : ℕ
  mother : ℕ
  brother : ℕ
  sister : ℕ
  remaining : ℕ

/-- Theorem stating the age of the remaining family member -/
theorem remaining_family_member_age 
  (family : FamilyAges)
  (h_total : family.total = 200)
  (h_father : family.father = 60)
  (h_mother : family.mother = family.father - 2)
  (h_brother : family.brother = family.father / 2)
  (h_sister : family.sister = 40)
  (h_sum : family.total = family.father + family.mother + family.brother + family.sister + family.remaining) :
  family.remaining = 12 :=
by sorry

end remaining_family_member_age_l2976_297610


namespace min_value_of_x_plus_3y_min_value_is_16_min_value_achieved_l2976_297605

theorem min_value_of_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a * b = 3 * a + b → x + 3 * y ≤ a + 3 * b :=
by sorry

theorem min_value_is_16 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  x + 3 * y ≥ 16 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 3 * x + y) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 3 * a + b ∧ a + 3 * b = 16 :=
by sorry

end min_value_of_x_plus_3y_min_value_is_16_min_value_achieved_l2976_297605


namespace production_increase_l2976_297685

/-- Calculates the number of units produced today given the previous average, 
    number of days, and new average including today's production. -/
def units_produced_today (prev_avg : ℝ) (prev_days : ℕ) (new_avg : ℝ) : ℝ :=
  (new_avg * (prev_days + 1)) - (prev_avg * prev_days)

/-- Proves that given the conditions, the number of units produced today is 90. -/
theorem production_increase (prev_avg : ℝ) (prev_days : ℕ) (new_avg : ℝ) 
  (h1 : prev_avg = 60)
  (h2 : prev_days = 5)
  (h3 : new_avg = 65) :
  units_produced_today prev_avg prev_days new_avg = 90 := by
  sorry

#eval units_produced_today 60 5 65

end production_increase_l2976_297685


namespace apple_tv_cost_l2976_297606

theorem apple_tv_cost (iphone_count : ℕ) (iphone_cost : ℝ)
                      (ipad_count : ℕ) (ipad_cost : ℝ)
                      (apple_tv_count : ℕ)
                      (total_avg_cost : ℝ) :
  iphone_count = 100 →
  iphone_cost = 1000 →
  ipad_count = 20 →
  ipad_cost = 900 →
  apple_tv_count = 80 →
  total_avg_cost = 670 →
  (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * 200) / (iphone_count + ipad_count + apple_tv_count)) / (iphone_count + ipad_count + apple_tv_count) = total_avg_cost →
  (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * 200) / (iphone_count + ipad_count + apple_tv_count) = total_avg_cost :=
by sorry

#check apple_tv_cost

end apple_tv_cost_l2976_297606


namespace intersection_with_complement_l2976_297659

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end intersection_with_complement_l2976_297659


namespace difference_of_squares_70_30_l2976_297622

theorem difference_of_squares_70_30 : 70^2 - 30^2 = 4000 := by sorry

end difference_of_squares_70_30_l2976_297622


namespace geometric_sequence_sum_l2976_297673

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), r > 0 ∧ a₁ > 0 ∧ ∀ n, a n = a₁ * r ^ (n - 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 3 + a 2 * a 6 + 2 * a 3 ^ 2 = 36) →
  (a 2 + a 4 = 6) := by
  sorry

end geometric_sequence_sum_l2976_297673


namespace grains_per_teaspoon_l2976_297634

/-- Represents the number of grains of rice in a cup -/
def grains_per_cup : ℕ := 480

/-- Represents the number of tablespoons in half a cup -/
def tablespoons_per_half_cup : ℕ := 8

/-- Represents the number of teaspoons in a tablespoon -/
def teaspoons_per_tablespoon : ℕ := 3

/-- Theorem stating that there are 10 grains of rice in a teaspoon -/
theorem grains_per_teaspoon : 
  (grains_per_cup : ℚ) / ((2 * tablespoons_per_half_cup) * teaspoons_per_tablespoon) = 10 := by
  sorry

end grains_per_teaspoon_l2976_297634


namespace total_quantities_l2976_297692

theorem total_quantities (total_avg : ℝ) (subset1_count : ℕ) (subset1_avg : ℝ) (subset2_count : ℕ) (subset2_avg : ℝ) :
  total_avg = 6 →
  subset1_count = 3 →
  subset1_avg = 4 →
  subset2_count = 2 →
  subset2_avg = 33 →
  ∃ (n : ℕ), n = subset1_count + subset2_count ∧ n = 13 :=
by
  sorry

end total_quantities_l2976_297692


namespace angle_measure_in_acute_triangle_l2976_297683

-- Define an acute triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

-- State the theorem
theorem angle_measure_in_acute_triangle (t : AcuteTriangle) :
  (t.b^2 + t.c^2 - t.a^2) * Real.tan t.A = t.b * t.c → t.A = π/6 := by
  sorry

end angle_measure_in_acute_triangle_l2976_297683


namespace cookie_chip_ratio_l2976_297604

/-- Proves that the ratio of cookie tins to chip bags is 4:1 given the problem conditions -/
theorem cookie_chip_ratio :
  let chip_weight : ℕ := 20  -- weight of a bag of chips in ounces
  let cookie_weight : ℕ := 9  -- weight of a tin of cookies in ounces
  let chip_bags : ℕ := 6  -- number of bags of chips Jasmine buys
  let total_weight : ℕ := 21 * 16  -- total weight Jasmine carries in ounces

  let cookie_tins : ℕ := (total_weight - chip_weight * chip_bags) / cookie_weight

  (cookie_tins : ℚ) / chip_bags = 4 / 1 :=
by sorry

end cookie_chip_ratio_l2976_297604


namespace perpendicular_lines_k_l2976_297612

/-- Two lines in the plane given by their equations -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_k (k : ℝ) :
  let l1 : Line := { a := k, b := -1, c := -3 }
  let l2 : Line := { a := 1, b := 2*k+3, c := -2 }
  perpendicular l1 l2 → k = -3 := by
sorry

end perpendicular_lines_k_l2976_297612


namespace delta_value_l2976_297682

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ + 5 → Δ = -17 := by
  sorry

end delta_value_l2976_297682


namespace p_value_l2976_297657

theorem p_value (p : ℝ) : (∀ x : ℝ, (x - 1) * (x + 2) = x^2 + p*x - 2) → p = 1 := by
  sorry

end p_value_l2976_297657


namespace cats_owned_by_olly_l2976_297674

def shoes_per_animal : ℕ := 4

def num_dogs : ℕ := 3

def num_ferrets : ℕ := 1

def total_shoes : ℕ := 24

def num_cats : ℕ := (total_shoes - (num_dogs + num_ferrets) * shoes_per_animal) / shoes_per_animal

theorem cats_owned_by_olly :
  num_cats = 2 := by sorry

end cats_owned_by_olly_l2976_297674


namespace increasing_function_inequality_l2976_297624

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
  sorry

end increasing_function_inequality_l2976_297624


namespace pig_feed_per_day_l2976_297671

/-- Given that Randy has 2 pigs and they are fed 140 pounds of pig feed per week,
    prove that each pig is fed 10 pounds of feed per day. -/
theorem pig_feed_per_day (num_pigs : ℕ) (total_feed_per_week : ℕ) (days_per_week : ℕ) :
  num_pigs = 2 →
  total_feed_per_week = 140 →
  days_per_week = 7 →
  (total_feed_per_week / num_pigs) / days_per_week = 10 :=
by
  sorry

end pig_feed_per_day_l2976_297671


namespace perpendicular_vectors_k_l2976_297631

/-- Two planar vectors a and b are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_k (k : ℝ) :
  let a : ℝ × ℝ := (k, 3)
  let b : ℝ × ℝ := (1, 4)
  perpendicular a b → k = -12 := by
sorry

end perpendicular_vectors_k_l2976_297631


namespace new_vessel_capacity_l2976_297691

/-- Given two vessels with different alcohol concentrations, prove the capacity of a new vessel that contains their combined contents plus water to achieve a specific concentration. -/
theorem new_vessel_capacity
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percent : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percent : ℝ)
  (total_liquid : ℝ)
  (new_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percent = 0.25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percent = 0.40)
  (h5 : total_liquid = 8)
  (h6 : new_concentration = 0.29000000000000004) :
  (vessel1_capacity * vessel1_alcohol_percent + vessel2_capacity * vessel2_alcohol_percent) / new_concentration = 10 := by
  sorry

#eval (2 * 0.25 + 6 * 0.40) / 0.29000000000000004

end new_vessel_capacity_l2976_297691


namespace age_difference_l2976_297644

-- Define the ages of A and B
def A : ℕ := sorry
def B : ℕ := 95

-- State the theorem
theorem age_difference : A - B = 5 := by
  -- The condition that in 30 years, A will be twice as old as B was 30 years ago
  have h : A + 30 = 2 * (B - 30) := by sorry
  sorry

end age_difference_l2976_297644


namespace tower_height_difference_l2976_297646

theorem tower_height_difference (grace_height clyde_height : ℕ) :
  grace_height = 40 ∧ grace_height = 8 * clyde_height →
  grace_height - clyde_height = 35 := by
  sorry

end tower_height_difference_l2976_297646


namespace min_marked_price_for_profit_l2976_297695

theorem min_marked_price_for_profit (num_sets : ℕ) (purchase_price : ℝ) (discount_rate : ℝ) (desired_profit : ℝ) :
  let marked_price := (desired_profit + num_sets * purchase_price) / (num_sets * (1 - discount_rate))
  marked_price ≥ 200 ∧ 
  num_sets * (1 - discount_rate) * marked_price - num_sets * purchase_price ≥ desired_profit ∧
  ∀ x < marked_price, num_sets * (1 - discount_rate) * x - num_sets * purchase_price < desired_profit :=
by sorry

end min_marked_price_for_profit_l2976_297695


namespace circle_cutting_terminates_l2976_297628

-- Define the circle-cutting process
def circle_cutting_process (m : ℕ) (n : ℕ) : Prop :=
  m ≥ 2 ∧ ∃ (remaining_area : ℝ), 
    remaining_area > 0 ∧
    remaining_area < (1 - 1/m)^n

-- Theorem statement
theorem circle_cutting_terminates (m : ℕ) :
  m ≥ 2 → ∃ n : ℕ, ∀ k : ℕ, k ≥ n → ¬(circle_cutting_process m k) :=
sorry

end circle_cutting_terminates_l2976_297628


namespace mean_proportional_of_segments_l2976_297603

theorem mean_proportional_of_segments (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∃ c : ℝ, c > 0 ∧ c^2 = a * b ∧ c = 2 * Real.sqrt 3 := by
  sorry

end mean_proportional_of_segments_l2976_297603


namespace missing_angle_in_polygon_l2976_297625

theorem missing_angle_in_polygon (n : ℕ) (sum_angles : ℝ) (common_angle : ℝ) : 
  sum_angles = 3420 →
  common_angle = 150 →
  n > 2 →
  (n - 1) * common_angle + (sum_angles - (n - 1) * common_angle) = sum_angles →
  sum_angles - (n - 1) * common_angle = 420 :=
by sorry

end missing_angle_in_polygon_l2976_297625


namespace equation_solution_l2976_297653

theorem equation_solution : 
  ∃! x : ℝ, 45 - (28 - (37 - (15 - x))) = 58 ∧ x = 19 := by sorry

end equation_solution_l2976_297653


namespace largest_integer_for_negative_quadratic_l2976_297660

theorem largest_integer_for_negative_quadratic : 
  ∃ n : ℤ, n^2 - 11*n + 28 < 0 ∧ 
  ∀ m : ℤ, m^2 - 11*m + 28 < 0 → m ≤ n ∧ 
  n = 6 := by sorry

end largest_integer_for_negative_quadratic_l2976_297660


namespace range_of_sin_minus_cos_l2976_297623

open Real

theorem range_of_sin_minus_cos (x : ℝ) : 
  -Real.sqrt 3 ≤ sin (x + 18 * π / 180) - cos (x + 48 * π / 180) ∧
  sin (x + 18 * π / 180) - cos (x + 48 * π / 180) ≤ Real.sqrt 3 := by
  sorry

end range_of_sin_minus_cos_l2976_297623


namespace number_times_x_minus_3y_l2976_297618

/-- Given that 2x - y = 4 and kx - 3y = 12, prove that k = 6 -/
theorem number_times_x_minus_3y (x y k : ℝ) 
  (h1 : 2 * x - y = 4) 
  (h2 : k * x - 3 * y = 12) : 
  k = 6 := by
  sorry

end number_times_x_minus_3y_l2976_297618


namespace class_average_approx_76_percent_l2976_297637

def class_average (group1_percent : ℝ) (group1_score : ℝ) 
                  (group2_percent : ℝ) (group2_score : ℝ) 
                  (group3_percent : ℝ) (group3_score : ℝ) : ℝ :=
  group1_percent * group1_score + group2_percent * group2_score + group3_percent * group3_score

theorem class_average_approx_76_percent :
  let group1_percent : ℝ := 0.15
  let group1_score : ℝ := 100
  let group2_percent : ℝ := 0.50
  let group2_score : ℝ := 78
  let group3_percent : ℝ := 0.35
  let group3_score : ℝ := 63
  let average := class_average group1_percent group1_score group2_percent group2_score group3_percent group3_score
  ∃ ε > 0, |average - 76| < ε :=
by
  sorry


end class_average_approx_76_percent_l2976_297637


namespace quadratic_perfect_square_l2976_297654

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + c = (x + a)^2) → c = 5625 := by
  sorry

end quadratic_perfect_square_l2976_297654


namespace oranges_per_bag_l2976_297641

theorem oranges_per_bag (total_bags : ℕ) (rotten_oranges : ℕ) (juice_oranges : ℕ) (sold_oranges : ℕ)
  (h1 : total_bags = 10)
  (h2 : rotten_oranges = 50)
  (h3 : juice_oranges = 30)
  (h4 : sold_oranges = 220) :
  (rotten_oranges + juice_oranges + sold_oranges) / total_bags = 30 :=
by sorry

end oranges_per_bag_l2976_297641


namespace intersection_condition_l2976_297666

/-- The parabola equation: x = -3y^2 - 4y + 10 -/
def parabola (x y : ℝ) : Prop := x = -3 * y^2 - 4 * y + 10

/-- The line equation: x = k -/
def line (x k : ℝ) : Prop := x = k

/-- The condition for exactly one intersection point -/
def unique_intersection (k : ℝ) : Prop :=
  ∃! y, parabola k y

theorem intersection_condition (k : ℝ) :
  unique_intersection k ↔ k = 34 / 3 := by sorry

end intersection_condition_l2976_297666


namespace circle_range_theta_l2976_297620

/-- The range of θ for a circle with center (2cos θ, 2sin θ) and radius 1,
    where all points (x,y) on the circle satisfy x ≤ y -/
theorem circle_range_theta :
  ∀ θ : ℝ,
  (∀ x y : ℝ, (x - 2 * Real.cos θ)^2 + (y - 2 * Real.sin θ)^2 = 1 → x ≤ y) →
  0 ≤ θ →
  θ ≤ 2 * Real.pi →
  5 * Real.pi / 12 ≤ θ ∧ θ ≤ 13 * Real.pi / 12 :=
by sorry

end circle_range_theta_l2976_297620


namespace problem_sample_is_valid_problem_sample_sequence_correct_l2976_297651

/-- Represents a systematic sample -/
structure SystematicSample where
  first : ℕ
  interval : ℕ
  size : ℕ
  population : ℕ

/-- Checks if a systematic sample is valid -/
def isValidSystematicSample (s : SystematicSample) : Prop :=
  s.first > 0 ∧
  s.first ≤ s.population ∧
  s.interval > 0 ∧
  s.size > 0 ∧
  s.population ≥ s.size ∧
  ∀ i : ℕ, i < s.size → s.first + i * s.interval ≤ s.population

/-- The specific systematic sample from the problem -/
def problemSample : SystematicSample :=
  { first := 3
    interval := 10
    size := 6
    population := 60 }

/-- Theorem stating that the problem's sample is valid -/
theorem problem_sample_is_valid : isValidSystematicSample problemSample := by
  sorry

/-- The sequence of numbers in the systematic sample -/
def sampleSequence (s : SystematicSample) : List ℕ :=
  List.range s.size |>.map (λ i => s.first + i * s.interval)

/-- Theorem stating that the sample sequence matches the given answer -/
theorem problem_sample_sequence_correct :
  sampleSequence problemSample = [3, 13, 23, 33, 43, 53] := by
  sorry

end problem_sample_is_valid_problem_sample_sequence_correct_l2976_297651


namespace sprint_no_wind_time_l2976_297672

/-- A sprinter's performance under different wind conditions -/
structure SprintPerformance where
  with_wind_distance : ℝ
  against_wind_distance : ℝ
  time_with_wind : ℝ
  time_against_wind : ℝ
  wind_speed : ℝ
  no_wind_speed : ℝ

/-- Theorem stating the time taken to run 100 meters in no wind condition -/
theorem sprint_no_wind_time (perf : SprintPerformance) 
  (h1 : perf.with_wind_distance = 90)
  (h2 : perf.against_wind_distance = 70)
  (h3 : perf.time_with_wind = 10)
  (h4 : perf.time_against_wind = 10)
  (h5 : perf.with_wind_distance / (perf.no_wind_speed + perf.wind_speed) = perf.time_with_wind)
  (h6 : perf.against_wind_distance / (perf.no_wind_speed - perf.wind_speed) = perf.time_against_wind)
  : (100 : ℝ) / perf.no_wind_speed = 12.5 := by
  sorry

end sprint_no_wind_time_l2976_297672


namespace cos_A_minus_B_l2976_297661

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1.5) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 0.625 := by
  sorry

end cos_A_minus_B_l2976_297661


namespace zoo_animals_count_l2976_297638

/-- Represents the number of four-legged birds -/
def num_birds : ℕ := 14

/-- Represents the number of six-legged calves -/
def num_calves : ℕ := 22

/-- The total number of heads -/
def total_heads : ℕ := 36

/-- The total number of legs -/
def total_legs : ℕ := 100

/-- The number of legs each bird has -/
def bird_legs : ℕ := 4

/-- The number of legs each calf has -/
def calf_legs : ℕ := 6

theorem zoo_animals_count :
  (num_birds + num_calves = total_heads) ∧
  (num_birds * bird_legs + num_calves * calf_legs = total_legs) := by
  sorry

end zoo_animals_count_l2976_297638


namespace sum_of_digits_of_five_to_23_l2976_297677

/-- The sum of the tens digit and the ones digit of (2+3)^23 is 7 -/
theorem sum_of_digits_of_five_to_23 :
  let n : ℕ := (2 + 3)^23
  (n / 10 % 10) + (n % 10) = 7 := by
  sorry

end sum_of_digits_of_five_to_23_l2976_297677


namespace A_intersect_B_l2976_297607

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | (x + 1) * (x - 2) < 0}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end A_intersect_B_l2976_297607


namespace orthocenter_on_line_l2976_297639

/-
  Define the necessary geometric objects and properties
-/

-- Define a Point type
structure Point := (x y : ℝ)

-- Define a Line type
structure Line := (a b c : ℝ)

-- Define a Circle type
structure Circle := (center : Point) (radius : ℝ)

-- Define a Triangle type
structure Triangle := (A B C : Point)

-- Function to check if a triangle is acute-angled
def is_acute_triangle (t : Triangle) : Prop := sorry

-- Function to get the circumcenter of a triangle
def circumcenter (t : Triangle) : Point := sorry

-- Function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Function to get the orthocenter of a triangle
def orthocenter (t : Triangle) : Point := sorry

-- Function to check if a circle passes through a point
def circle_passes_through (c : Circle) (p : Point) : Prop := sorry

-- Function to get the intersection points of a circle and a line segment
def circle_line_intersection (c : Circle) (l : Line) : List Point := sorry

-- Main theorem
theorem orthocenter_on_line 
  (A B C : Point) 
  (O : Point) 
  (c : Circle) 
  (P Q : Point) :
  is_acute_triangle (Triangle.mk A B C) →
  O = circumcenter (Triangle.mk A B C) →
  circle_passes_through c B →
  circle_passes_through c O →
  P ∈ circle_line_intersection c (Line.mk 0 1 0) → -- Assuming BC is on y-axis
  Q ∈ circle_line_intersection c (Line.mk 1 0 0) → -- Assuming BA is on x-axis
  point_on_line (orthocenter (Triangle.mk P O Q)) (Line.mk 1 1 0) -- Assuming AC is y = x
  := by sorry

end orthocenter_on_line_l2976_297639


namespace original_number_proof_l2976_297675

theorem original_number_proof (t : ℝ) : 
  t * (1 + 0.125) - t * (1 - 0.25) = 30 → t = 80 := by
  sorry

end original_number_proof_l2976_297675


namespace solution_relationship_l2976_297690

theorem solution_relationship (c c' d d' : ℝ) 
  (hc : c ≠ 0) (hc' : c' ≠ 0)
  (h : -d / (2 * c) = 2 * (-d' / (3 * c'))) :
  d / (2 * c) = 2 * d' / (3 * c') := by
  sorry

end solution_relationship_l2976_297690


namespace equation_transformation_l2976_297689

theorem equation_transformation (x y : ℝ) : x - 3 = y - 3 → x - y = 0 := by
  sorry

end equation_transformation_l2976_297689


namespace yoque_borrowed_amount_l2976_297636

/-- The amount Yoque borrowed -/
def borrowed_amount : ℝ := 150

/-- The number of months for repayment -/
def repayment_period : ℕ := 11

/-- The monthly payment amount -/
def monthly_payment : ℝ := 15

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.1

theorem yoque_borrowed_amount :
  borrowed_amount = (monthly_payment * repayment_period) / (1 + interest_rate) :=
by sorry

end yoque_borrowed_amount_l2976_297636


namespace stratified_sampling_ratio_l2976_297650

theorem stratified_sampling_ratio 
  (total_first : ℕ) 
  (total_second : ℕ) 
  (sample_first : ℕ) 
  (sample_second : ℕ) 
  (h1 : total_first = 400) 
  (h2 : total_second = 360) 
  (h3 : sample_first = 60) : 
  (sample_first : ℚ) / total_first = (sample_second : ℚ) / total_second → 
  sample_second = 54 := by
sorry

end stratified_sampling_ratio_l2976_297650


namespace hexagon_angle_measure_l2976_297647

/-- Given a convex hexagon ABCDEF with the following properties:
  - Angles A, B, and C are congruent
  - Angles D and E are congruent
  - Angle A is 30° less than angle D
  - Angle F is equal to angle A
  Prove that the measure of angle D is 140° -/
theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  A = B ∧ B = C ∧                      -- Angles A, B, and C are congruent
  D = E ∧                              -- Angles D and E are congruent
  A = D - 30 ∧                         -- Angle A is 30° less than angle D
  F = A ∧                              -- Angle F is equal to angle A
  A + B + C + D + E + F = 720          -- Sum of angles in a hexagon
  → D = 140 := by
  sorry

end hexagon_angle_measure_l2976_297647


namespace semicircle_radius_l2976_297676

theorem semicircle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 2 = 12.5 * π) (h_xz_arc : π * y = 9 * π) :
  z / 2 = Real.sqrt 424 / 2 := by
  sorry

end semicircle_radius_l2976_297676


namespace circle_point_range_l2976_297642

theorem circle_point_range (a : ℝ) : 
  ((-1 + a)^2 + (-1 - a)^2 < 4) → (-1 < a ∧ a < 1) :=
by sorry

end circle_point_range_l2976_297642


namespace projection_theorem_l2976_297681

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ := sorry

theorem projection_theorem (Q : Plane) :
  project (7, 1, 8) Q = (6, 3, 2) →
  project (6, 2, 9) Q = (9/2, 5, 9/2) := by sorry

end projection_theorem_l2976_297681


namespace number_problem_l2976_297652

theorem number_problem (x : ℝ) : 35 + 3 * x = 56 → x = 7 := by
  sorry

end number_problem_l2976_297652


namespace total_amount_calculation_l2976_297668

theorem total_amount_calculation (part1 : ℝ) (part2 : ℝ) (total_interest : ℝ) :
  part1 = 1500.0000000000007 →
  part1 * 0.05 + part2 * 0.06 = 135 →
  part1 + part2 = 2500.000000000000 :=
by
  sorry

end total_amount_calculation_l2976_297668


namespace matchsticks_20th_stage_l2976_297613

def matchsticks (n : ℕ) : ℕ :=
  5 + 3 * (n - 1) + (n - 1) / 5

theorem matchsticks_20th_stage :
  matchsticks 20 = 66 := by
  sorry

end matchsticks_20th_stage_l2976_297613


namespace no_solution_for_equation_l2976_297686

theorem no_solution_for_equation : ¬∃ x : ℝ, 1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4) := by
  sorry

end no_solution_for_equation_l2976_297686


namespace largest_number_is_sqrt5_l2976_297601

theorem largest_number_is_sqrt5 (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x*y + x*z + y*z = -11)
  (prod_eq : x*y*z = 15) :
  max x (max y z) = Real.sqrt 5 := by
sorry

end largest_number_is_sqrt5_l2976_297601


namespace rotten_bananas_percentage_l2976_297627

theorem rotten_bananas_percentage (total_oranges total_bananas : ℕ)
  (rotten_oranges_percent good_fruits_percent : ℚ) :
  total_oranges = 600 →
  total_bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  good_fruits_percent = 898 / 1000 →
  (total_bananas - (good_fruits_percent * (total_oranges + total_bananas : ℚ) -
    ((1 - rotten_oranges_percent) * total_oranges))) / total_bananas = 3 / 100 := by
  sorry

end rotten_bananas_percentage_l2976_297627


namespace equation_system_solution_l2976_297615

theorem equation_system_solution (a b c d : ℝ) :
  (a + b = c + d) →
  (a^3 + b^3 = c^3 + d^3) →
  ((a = c ∧ b = d) ∨ (a = d ∧ b = c)) :=
by sorry

end equation_system_solution_l2976_297615


namespace total_crayons_l2976_297669

theorem total_crayons (billy_crayons jane_crayons : Float) 
  (h1 : billy_crayons = 62.0) 
  (h2 : jane_crayons = 52.0) : 
  billy_crayons + jane_crayons = 114.0 := by
  sorry

end total_crayons_l2976_297669


namespace larger_integer_value_l2976_297632

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 3 / 2)
  (h_product : (a : ℕ) * b = 180) :
  (a : ℝ) = 3 * Real.sqrt 30 := by
  sorry

end larger_integer_value_l2976_297632


namespace triangle_lines_theorem_l2976_297600

-- Define the triangle vertices
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- Define the line equation type
def LineEquation := ℝ → ℝ → ℝ

-- Define the line AC
def line_AC : LineEquation := fun x y => 3 * x + 4 * y - 12

-- Define the altitude from B to AB
def altitude_B : LineEquation := fun x y => 2 * x + 7 * y - 21

-- Theorem statement
theorem triangle_lines_theorem :
  (∀ x y, line_AC x y = 0 ↔ (x - A.1) * (C.2 - A.2) = (y - A.2) * (C.1 - A.1)) ∧
  (∀ x y, altitude_B x y = 0 ↔ (x - B.1) * (B.1 - A.1) + (y - B.2) * (B.2 - A.2) = 0) :=
sorry

end triangle_lines_theorem_l2976_297600


namespace g_composition_of_six_l2976_297667

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 2 * x + 1

theorem g_composition_of_six : g (g (g (g 6))) = 23 := by
  sorry

end g_composition_of_six_l2976_297667


namespace part_one_part_two_l2976_297696

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m-9)*x + m^2 - 9*m ≥ 0}

-- Define the complement of B in ℝ
def C_R_B (m : ℝ) : Set ℝ := (Set.univ : Set ℝ) \ B m

-- Part 1: If A ∩ B = [-3, 3], then m = 12
theorem part_one (m : ℝ) : A ∩ B m = Set.Icc (-3) 3 → m = 12 := by sorry

-- Part 2: If A ⊆ C_ℝB, then 5 < m < 6
theorem part_two (m : ℝ) : A ⊆ C_R_B m → 5 < m ∧ m < 6 := by sorry

end part_one_part_two_l2976_297696


namespace percentage_error_multiplication_l2976_297670

theorem percentage_error_multiplication : 
  let correct_factor : ℚ := 5 / 3
  let incorrect_factor : ℚ := 3 / 5
  let percentage_error := (correct_factor - incorrect_factor) / correct_factor * 100
  percentage_error = 64 := by
sorry

end percentage_error_multiplication_l2976_297670


namespace carol_invitation_packs_l2976_297662

/-- The number of friends Carol is sending invitations to -/
def num_friends : ℕ := 12

/-- The number of invitations in each pack -/
def invitations_per_pack : ℕ := 4

/-- The number of packs Carol bought -/
def num_packs : ℕ := num_friends / invitations_per_pack

theorem carol_invitation_packs : num_packs = 3 := by
  sorry

end carol_invitation_packs_l2976_297662


namespace tim_total_amount_l2976_297649

-- Define the value of each coin type
def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10
def half_dollar_value : ℚ := 0.50

-- Define the number of each coin type Tim received
def nickels_from_shining : ℕ := 3
def dimes_from_shining : ℕ := 13
def dimes_from_tip_jar : ℕ := 7
def half_dollars_from_tip_jar : ℕ := 9

-- Calculate the total amount Tim received
def total_amount : ℚ :=
  nickels_from_shining * nickel_value +
  (dimes_from_shining + dimes_from_tip_jar) * dime_value +
  half_dollars_from_tip_jar * half_dollar_value

-- Theorem statement
theorem tim_total_amount : total_amount = 6.65 := by
  sorry

end tim_total_amount_l2976_297649


namespace average_mark_is_35_l2976_297694

/-- The average mark obtained by candidates in an examination. -/
def average_mark (total_marks : ℕ) (num_candidates : ℕ) : ℚ :=
  total_marks / num_candidates

/-- Theorem stating that the average mark is 35 given the conditions. -/
theorem average_mark_is_35 :
  average_mark 4200 120 = 35 := by
  sorry

end average_mark_is_35_l2976_297694


namespace minimum_value_theorem_l2976_297629

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end minimum_value_theorem_l2976_297629


namespace code_problem_l2976_297630

theorem code_problem (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 →
  B > A →
  A < C →
  11 * B + 11 * A + 11 * C = 242 →
  ((A = 5 ∧ B = 8 ∧ C = 9) ∨ (A = 5 ∧ B = 9 ∧ C = 8)) :=
by sorry

end code_problem_l2976_297630


namespace cost_of_stationery_l2976_297611

/-- Given the cost of different combinations of erasers, pens, and markers,
    prove that 3 erasers, 4 pens, and 6 markers cost 520 rubles. -/
theorem cost_of_stationery (E P M : ℕ) : 
  (E + 3 * P + 2 * M = 240) →
  (2 * E + 5 * P + 4 * M = 440) →
  (3 * E + 4 * P + 6 * M = 520) :=
by sorry

end cost_of_stationery_l2976_297611


namespace number_ratio_problem_l2976_297699

theorem number_ratio_problem (N : ℝ) (x : ℝ) 
  (h1 : N = 280) 
  (h2 : (1/5) * N + 4 = x * N - 10) : 
  x = 1/4 := by sorry

end number_ratio_problem_l2976_297699


namespace fraction_product_squared_l2976_297609

theorem fraction_product_squared :
  (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end fraction_product_squared_l2976_297609


namespace group_ratio_theorem_l2976_297697

/-- Represents the group composition and average ages -/
structure GroupComposition where
  avg_age : ℝ
  doc_age : ℝ
  law_age : ℝ
  eng_age : ℝ
  doc_count : ℝ
  law_count : ℝ
  eng_count : ℝ

/-- Theorem stating the ratios of group members based on given average ages -/
theorem group_ratio_theorem (g : GroupComposition) 
  (h1 : g.avg_age = 45)
  (h2 : g.doc_age = 40)
  (h3 : g.law_age = 55)
  (h4 : g.eng_age = 35)
  (h5 : g.avg_age * (g.doc_count + g.law_count + g.eng_count) = 
        g.doc_age * g.doc_count + g.law_age * g.law_count + g.eng_age * g.eng_count) :
  g.doc_count / g.law_count = 1 ∧ g.eng_count / g.law_count = 2 := by
  sorry

#check group_ratio_theorem

end group_ratio_theorem_l2976_297697


namespace solution_set_absolute_value_inequality_l2976_297680

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 3| + |x - 5| ≥ 4} = {x : ℝ | x ≤ 2 ∨ x ≥ 6} := by sorry

end solution_set_absolute_value_inequality_l2976_297680


namespace boundary_length_special_square_l2976_297698

/-- The length of the boundary of a special figure constructed from a square --/
theorem boundary_length_special_square : 
  ∀ (s : Real) (a : Real),
    s * s = 64 →  -- area of the square is 64
    a = s / 4 →   -- length of each arc segment
    (16 : Real) + 14 * Real.pi = 
      4 * s +     -- sum of straight segments
      12 * (a * Real.pi / 2) +  -- sum of side arcs
      4 * (a * Real.pi / 2)     -- sum of corner arcs
    := by sorry

end boundary_length_special_square_l2976_297698


namespace all_propositions_true_l2976_297663

theorem all_propositions_true (a b : ℝ) :
  (a > b → a * |a| > b * |b|) ∧
  (a * |a| > b * |b| → a > b) ∧
  (a ≤ b → a * |a| ≤ b * |b|) ∧
  (a * |a| ≤ b * |b| → a ≤ b) :=
sorry

end all_propositions_true_l2976_297663


namespace second_day_distance_l2976_297645

-- Define the constants
def first_day_distance : ℝ := 250
def average_speed : ℝ := 33.333333333333336
def time_difference : ℝ := 3

-- Define the theorem
theorem second_day_distance :
  let first_day_time := first_day_distance / average_speed
  let second_day_time := first_day_time + time_difference
  second_day_time * average_speed = 350 := by
  sorry

end second_day_distance_l2976_297645


namespace triangle_30_60_90_divisible_l2976_297655

/-- A triangle with angles 30°, 60°, and 90° -/
structure Triangle30_60_90 where
  -- We define the triangle using its angles
  angle1 : Real
  angle2 : Real
  angle3 : Real
  angle1_eq : angle1 = 30
  angle2_eq : angle2 = 60
  angle3_eq : angle3 = 90
  sum_angles : angle1 + angle2 + angle3 = 180

/-- A representation of three equal triangles -/
structure ThreeEqualTriangles where
  -- We define three triangles and their equality
  triangle1 : Triangle30_60_90
  triangle2 : Triangle30_60_90
  triangle3 : Triangle30_60_90
  equality12 : triangle1 = triangle2
  equality23 : triangle2 = triangle3

/-- Theorem stating that a 30-60-90 triangle can be divided into three equal triangles -/
theorem triangle_30_60_90_divisible (t : Triangle30_60_90) : 
  ∃ (et : ThreeEqualTriangles), True :=
sorry

end triangle_30_60_90_divisible_l2976_297655


namespace k_set_characterization_l2976_297678

theorem k_set_characterization (r : ℕ) :
  let h := 2^r
  let k_set := {k : ℕ | ∃ (m n : ℕ), 
    Odd m ∧ m > 1 ∧
    k ∣ m^k - 1 ∧
    m ∣ n^((m^k - 1)/k) + 1}
  k_set = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ ¬ Even t} :=
by sorry

end k_set_characterization_l2976_297678


namespace regular_polygon_sides_l2976_297602

theorem regular_polygon_sides (n : ℕ) : n ≥ 3 →
  (n : ℝ) - (n * (n - 3) / 2) = 2 → n = 4 := by
  sorry

end regular_polygon_sides_l2976_297602


namespace inequality_solution_set_l2976_297684

theorem inequality_solution_set (x : ℝ) : 
  x^2 - |x - 1| - 1 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 1 := by sorry

end inequality_solution_set_l2976_297684


namespace farmers_wheat_harvest_l2976_297626

/-- The farmer's wheat harvest problem -/
theorem farmers_wheat_harvest 
  (estimated_harvest : ℕ) 
  (additional_harvest : ℕ) 
  (h1 : estimated_harvest = 48097)
  (h2 : additional_harvest = 684) :
  estimated_harvest + additional_harvest = 48781 :=
by sorry

end farmers_wheat_harvest_l2976_297626


namespace envelope_ratio_l2976_297693

theorem envelope_ratio (blue_envelopes : ℕ) (yellow_diff : ℕ) (total_envelopes : ℕ)
  (h1 : blue_envelopes = 14)
  (h2 : yellow_diff = 6)
  (h3 : total_envelopes = 46) :
  ∃ (green_envelopes yellow_envelopes : ℕ),
    yellow_envelopes = blue_envelopes - yellow_diff ∧
    green_envelopes = 3 * yellow_envelopes ∧
    blue_envelopes + yellow_envelopes + green_envelopes = total_envelopes ∧
    green_envelopes / yellow_envelopes = 3 := by
  sorry

end envelope_ratio_l2976_297693


namespace definite_integral_x_x_squared_sin_x_l2976_297688

theorem definite_integral_x_x_squared_sin_x : 
  ∫ x in (-1)..1, (x + x^2 + Real.sin x) = 2/3 := by sorry

end definite_integral_x_x_squared_sin_x_l2976_297688
