import Mathlib

namespace poster_distance_is_18cm_l1048_104849

/-- The number of posters -/
def num_posters : ℕ := 8

/-- The width of each poster in centimeters -/
def poster_width : ℝ := 29.05

/-- The width of the wall in meters -/
def wall_width_m : ℝ := 3.944

/-- The width of the wall in centimeters -/
def wall_width_cm : ℝ := wall_width_m * 100

/-- The number of gaps between posters and wall ends -/
def num_gaps : ℕ := num_posters + 1

/-- The theorem stating that the distance between posters is 18 cm -/
theorem poster_distance_is_18cm : 
  (wall_width_cm - num_posters * poster_width) / num_gaps = 18 := by
  sorry

end poster_distance_is_18cm_l1048_104849


namespace train_speed_calculation_l1048_104841

theorem train_speed_calculation (t m s : ℝ) (ht : t > 0) (hm : m > 0) (hs : s > 0) :
  let m₁ := (Real.sqrt (t * m * (4 * s + t * m)) - t * m) / (2 * t)
  ∃ (t₁ : ℝ), t₁ > 0 ∧ m₁ * t₁ = s ∧ (m₁ + m) * (t₁ - t) = s :=
by sorry

end train_speed_calculation_l1048_104841


namespace equation_solutions_l1048_104891

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 11*x - 8) + 1 / (x^2 + 2*x - 8) + 1 / (x^2 - 13*x - 8) = 0)} = 
  {8, 1, -1, -8} := by sorry

end equation_solutions_l1048_104891


namespace probability_same_group_four_people_prove_probability_same_group_four_people_l1048_104869

/-- The probability that two specific people are in the same group when four people are divided into two groups. -/
theorem probability_same_group_four_people : ℚ :=
  5 / 6

/-- Proof that the probability of two specific people being in the same group when four people are divided into two groups is 5/6. -/
theorem prove_probability_same_group_four_people :
  probability_same_group_four_people = 5 / 6 := by
  sorry

end probability_same_group_four_people_prove_probability_same_group_four_people_l1048_104869


namespace max_leftover_candy_l1048_104863

theorem max_leftover_candy (y : ℕ) : ∃ (q r : ℕ), y = 6 * q + r ∧ r < 6 ∧ r ≤ 5 :=
by sorry

end max_leftover_candy_l1048_104863


namespace sum_of_coefficients_l1048_104837

def polynomial (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9 * x^3 - 6) + 4 * (7 * x^6 - 2 * x^3 + 8)

theorem sum_of_coefficients : 
  polynomial 1 = 62 := by sorry

end sum_of_coefficients_l1048_104837


namespace clara_pill_cost_l1048_104814

/-- The cost of pills for Clara's treatment --/
def pill_cost (blue_cost : ℚ) : Prop :=
  let days : ℕ := 10
  let red_cost : ℚ := blue_cost - 2
  let daily_cost : ℚ := blue_cost + red_cost
  let total_cost : ℚ := 480
  (days : ℚ) * daily_cost = total_cost ∧ blue_cost = 25

theorem clara_pill_cost : ∃ (blue_cost : ℚ), pill_cost blue_cost := by
  sorry

end clara_pill_cost_l1048_104814


namespace solve_bus_problem_l1048_104800

def bus_problem (initial : ℕ) 
                (first_off : ℕ) 
                (second_off second_on : ℕ) 
                (third_off third_on : ℕ) : Prop :=
  let after_first := initial - first_off
  let after_second := after_first - second_off + second_on
  let after_third := after_second - third_off + third_on
  after_third = 28

theorem solve_bus_problem : 
  bus_problem 50 15 8 2 4 3 := by
  sorry

end solve_bus_problem_l1048_104800


namespace distribution_difference_l1048_104843

theorem distribution_difference (total : ℕ) (p q r s : ℕ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  q = r →
  p + q + r + s = total →
  s - p = 250 := by
sorry

end distribution_difference_l1048_104843


namespace iains_pennies_l1048_104893

/-- The number of pennies Iain had initially -/
def initial_pennies : ℕ := 200

/-- The number of old pennies removed -/
def old_pennies : ℕ := 30

/-- The percentage of remaining pennies kept after throwing out -/
def kept_percentage : ℚ := 80 / 100

/-- The number of pennies left after removing old pennies and throwing out some -/
def remaining_pennies : ℕ := 136

theorem iains_pennies :
  (kept_percentage * (initial_pennies - old_pennies : ℚ)).floor = remaining_pennies :=
sorry

end iains_pennies_l1048_104893


namespace cleaning_time_with_doubled_speed_l1048_104844

-- Define the cleaning rates
def anne_rate : ℚ := 1 / 12
def bruce_rate : ℚ := 1 / 4 - anne_rate

-- Define the time it takes for both to clean at normal speed
def normal_time : ℚ := 4

-- Define Anne's doubled rate
def anne_doubled_rate : ℚ := 2 * anne_rate

-- Theorem statement
theorem cleaning_time_with_doubled_speed :
  (bruce_rate + anne_doubled_rate)⁻¹ = 3 := by
  sorry

end cleaning_time_with_doubled_speed_l1048_104844


namespace shaded_cubes_count_l1048_104821

/-- Represents a 4x4x4 cube with a specific shading pattern -/
structure ShadedCube where
  /-- The number of smaller cubes along each edge of the large cube -/
  size : Nat
  /-- The shading pattern on one face of the cube -/
  shading_pattern : Fin 4 → Fin 4 → Bool
  /-- Assertion that the cube is 4x4x4 -/
  size_is_four : size = 4
  /-- The shading pattern includes the entire top row -/
  top_row_shaded : ∀ j, shading_pattern 0 j = true
  /-- The shading pattern includes the entire bottom row -/
  bottom_row_shaded : ∀ j, shading_pattern 3 j = true
  /-- The shading pattern includes one cube in each corner of the second and third rows -/
  corners_shaded : (shading_pattern 1 0 = true) ∧ (shading_pattern 1 3 = true) ∧
                   (shading_pattern 2 0 = true) ∧ (shading_pattern 2 3 = true)

/-- The total number of smaller cubes with at least one face shaded -/
def count_shaded_cubes (cube : ShadedCube) : Nat :=
  sorry

/-- Theorem stating that the number of shaded cubes is 32 -/
theorem shaded_cubes_count (cube : ShadedCube) : count_shaded_cubes cube = 32 := by
  sorry

end shaded_cubes_count_l1048_104821


namespace triangle_properties_l1048_104885

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (1/2 * b * c * Real.sin A = 3 * Real.sin A) →
  (a + b + c = 4 * (Real.sqrt 2 + 1)) →
  (Real.sin B + Real.sin C = Real.sqrt 2 * Real.sin A) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a = 4 ∧ 
   Real.cos A = 1/3 ∧
   Real.cos (2*A - π/3) = (4*Real.sqrt 6 - 7) / 18) := by
sorry

end triangle_properties_l1048_104885


namespace expenditure_savings_ratio_l1048_104806

/-- Given an income, expenditure, and savings, proves that the ratio of expenditure to savings is 1.5:1 -/
theorem expenditure_savings_ratio 
  (income expenditure savings : ℝ) 
  (h1 : income = expenditure + savings)
  (h2 : 1.15 * income = 1.21 * expenditure + 1.06 * savings) : 
  expenditure = 1.5 * savings := by
  sorry

end expenditure_savings_ratio_l1048_104806


namespace cross_section_area_is_21_over_8_l1048_104842

/-- Right prism with specific properties -/
structure RightPrism where
  -- Base triangle
  base_hypotenuse : ℝ
  base_angle_B : ℝ
  base_angle_C : ℝ
  -- Cutting plane properties
  distance_C_to_plane : ℝ

/-- The cross-section of the prism -/
def cross_section_area (prism : RightPrism) : ℝ :=
  sorry

/-- Main theorem: The area of the cross-section is 21/8 -/
theorem cross_section_area_is_21_over_8 (prism : RightPrism) 
  (h1 : prism.base_hypotenuse = Real.sqrt 14)
  (h2 : prism.base_angle_B = 90)
  (h3 : prism.base_angle_C = 30)
  (h4 : prism.distance_C_to_plane = 2) :
  cross_section_area prism = 21 / 8 :=
sorry

end cross_section_area_is_21_over_8_l1048_104842


namespace log_2_base_10_l1048_104881

theorem log_2_base_10 (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) (h3 : 2^9 = 512) (h4 : 2^12 = 4096) :
  Real.log 2 / Real.log 10 = 1/3 := by
  sorry

end log_2_base_10_l1048_104881


namespace farmer_feed_expenditure_l1048_104812

theorem farmer_feed_expenditure (initial_amount : ℝ) :
  (initial_amount * 0.4 / 0.5) + (initial_amount * 0.6) = 49 →
  initial_amount = 35 := by
sorry

end farmer_feed_expenditure_l1048_104812


namespace sqrt_fifteen_over_two_equals_half_sqrt_thirty_l1048_104852

theorem sqrt_fifteen_over_two_equals_half_sqrt_thirty :
  Real.sqrt (15 / 2) = (1 / 2) * Real.sqrt 30 := by
  sorry

end sqrt_fifteen_over_two_equals_half_sqrt_thirty_l1048_104852


namespace cupcake_ratio_l1048_104898

/-- Proves that the ratio of gluten-free cupcakes to total cupcakes is 3/20 given the specified conditions --/
theorem cupcake_ratio : 
  ∀ (total vegan non_vegan gluten_free : ℕ),
    total = 80 →
    vegan = 24 →
    non_vegan = 28 →
    gluten_free = vegan / 2 →
    (gluten_free : ℚ) / total = 3 / 20 := by
  sorry

end cupcake_ratio_l1048_104898


namespace largest_decimal_l1048_104865

theorem largest_decimal : 
  let a := 0.987
  let b := 0.9861
  let c := 0.98709
  let d := 0.968
  let e := 0.96989
  (c ≥ a) ∧ (c ≥ b) ∧ (c ≥ d) ∧ (c ≥ e) := by
  sorry

end largest_decimal_l1048_104865


namespace odd_square_minus_one_multiple_of_24_and_101_case_l1048_104839

theorem odd_square_minus_one_multiple_of_24_and_101_case : 
  (∀ n : ℕ, n > 1 → (2*n + 1)^2 - 1 = 24 * (n * (n + 1) / 2)) ∧ 
  (101^2 - 1 = 10200) := by
  sorry

end odd_square_minus_one_multiple_of_24_and_101_case_l1048_104839


namespace negation_of_proposition_l1048_104895

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := by sorry

end negation_of_proposition_l1048_104895


namespace iphone_price_drop_l1048_104804

/-- Calculates the final price of an iPhone after two consecutive price drops -/
theorem iphone_price_drop (initial_price : ℝ) (first_drop : ℝ) (second_drop : ℝ) :
  initial_price = 1000 ∧ first_drop = 0.1 ∧ second_drop = 0.2 →
  initial_price * (1 - first_drop) * (1 - second_drop) = 720 := by
  sorry


end iphone_price_drop_l1048_104804


namespace ellipse_hyperbola_eccentricity_product_l1048_104883

/-- Represents a conic section (ellipse or hyperbola) -/
structure Conic where
  center : ℝ × ℝ
  foci : ℝ × ℝ
  eccentricity : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem ellipse_hyperbola_eccentricity_product (C₁ C₂ : Conic) (P : Point) :
  C₁.center = (0, 0) →
  C₂.center = (0, 0) →
  C₁.foci.1 < 0 →
  C₁.foci.2 > 0 →
  C₂.foci = C₁.foci →
  P.x > 0 →
  P.y > 0 →
  (P.x - C₁.foci.1)^2 + P.y^2 = (P.x - C₁.foci.2)^2 + P.y^2 →
  C₁.eccentricity * C₂.eccentricity > 1/3 := by
  sorry

end ellipse_hyperbola_eccentricity_product_l1048_104883


namespace simplify_2A_minus_3B_value_2A_minus_3B_special_case_l1048_104875

/-- Given two real numbers a and b, we define A and B as follows: -/
def A (a b : ℝ) : ℝ := 3 * b^2 - 2 * a^2 + 5 * a * b

def B (a b : ℝ) : ℝ := 4 * a * b + 2 * b^2 - a^2

/-- Theorem stating that 2A - 3B simplifies to -a² - 2ab for any real a and b -/
theorem simplify_2A_minus_3B (a b : ℝ) : 2 * A a b - 3 * B a b = -a^2 - 2*a*b := by
  sorry

/-- Theorem stating that when a = -1 and b = 2, the value of 2A - 3B is 3 -/
theorem value_2A_minus_3B_special_case : 2 * A (-1) 2 - 3 * B (-1) 2 = 3 := by
  sorry

end simplify_2A_minus_3B_value_2A_minus_3B_special_case_l1048_104875


namespace lcm_1560_1040_l1048_104818

theorem lcm_1560_1040 : Nat.lcm 1560 1040 = 3120 := by
  sorry

end lcm_1560_1040_l1048_104818


namespace parallelepiped_dimensions_l1048_104859

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) → n = 18 := by
sorry

end parallelepiped_dimensions_l1048_104859


namespace function_value_at_three_l1048_104871

/-- Given a function f : ℝ → ℝ satisfying f(x) + 2f(1 - x) = 3x^2 for all real x,
    prove that f(3) = -1 -/
theorem function_value_at_three (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2) : 
    f 3 = -1 := by
  sorry

end function_value_at_three_l1048_104871


namespace cylinder_height_calculation_l1048_104889

/-- Given a cylinder with radius 3 units, if increasing the radius by 4 units
    and increasing the height by 10 units both result in the same volume increase,
    then the original height of the cylinder is 2.25 units. -/
theorem cylinder_height_calculation (h : ℝ) : 
  let r := 3
  let new_r := r + 4
  let new_h := h + 10
  let volume := π * r^2 * h
  let volume_after_radius_increase := π * new_r^2 * h
  let volume_after_height_increase := π * r^2 * new_h
  (volume_after_radius_increase - volume = volume_after_height_increase - volume) →
  h = 2.25 :=
by sorry

end cylinder_height_calculation_l1048_104889


namespace seating_theorem_l1048_104830

/-- The number of ways to seat 2 students in a row of 5 desks with at least one empty desk between them -/
def seating_arrangements : ℕ := 12

/-- The number of desks in the row -/
def num_desks : ℕ := 5

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- Minimum number of empty desks between students -/
def min_empty_desks : ℕ := 1

theorem seating_theorem :
  seating_arrangements = 12 ∧
  num_desks = 5 ∧
  num_students = 2 ∧
  min_empty_desks = 1 →
  seating_arrangements = 12 :=
by
  sorry

end seating_theorem_l1048_104830


namespace arithmetic_sequence_properties_l1048_104874

def a (n : ℕ) : ℤ := 2^n - (-1)^n

theorem arithmetic_sequence_properties :
  (∃ (n₁ n₂ n₃ : ℕ), n₁ < n₂ ∧ n₂ < n₃ ∧ 
    n₂ = n₁ + 1 ∧ n₃ = n₂ + 1 ∧
    2 * a n₂ = a n₁ + a n₃ ∧ n₁ = 2) ∧
  (∀ n₂ n₃ : ℕ, 1 < n₂ ∧ n₂ < n₃ ∧ 
    2 * a n₂ = a 1 + a n₃ → n₃ - n₂ = 1) ∧
  (∀ t : ℕ, t > 3 → 
    ¬∃ (s : ℕ → ℕ), Monotone s ∧ 
      (∀ i j : Fin t, i < j → s i < s j) ∧
      (∀ i : Fin (t - 1), 
        2 * a (s (i + 1)) = a (s i) + a (s (i + 2)))) :=
by sorry

end arithmetic_sequence_properties_l1048_104874


namespace first_divisible_correct_l1048_104845

/-- The first 4-digit number divisible by 25, 40, and 75 -/
def first_divisible : ℕ := 1200

/-- The greatest 4-digit number divisible by 25, 40, and 75 -/
def greatest_divisible : ℕ := 9600

/-- Theorem stating that first_divisible is the first 4-digit number divisible by 25, 40, and 75 -/
theorem first_divisible_correct :
  (first_divisible ≥ 1000) ∧
  (first_divisible ≤ 9999) ∧
  (first_divisible % 25 = 0) ∧
  (first_divisible % 40 = 0) ∧
  (first_divisible % 75 = 0) ∧
  (∀ n : ℕ, 1000 ≤ n ∧ n < first_divisible →
    ¬(n % 25 = 0 ∧ n % 40 = 0 ∧ n % 75 = 0)) ∧
  (greatest_divisible = 9600) ∧
  (greatest_divisible % 25 = 0) ∧
  (greatest_divisible % 40 = 0) ∧
  (greatest_divisible % 75 = 0) ∧
  (∀ m : ℕ, m > greatest_divisible → m > 9999 ∨ ¬(m % 25 = 0 ∧ m % 40 = 0 ∧ m % 75 = 0)) :=
by sorry

end first_divisible_correct_l1048_104845


namespace evaluate_expression_l1048_104838

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 + 2 = 288 := by
  sorry

end evaluate_expression_l1048_104838


namespace dumbbell_system_weight_l1048_104836

/-- Represents the weight of a dumbbell pair in pounds -/
structure DumbbellPair where
  weight : ℕ

/-- Represents a multi-level dumbbell system -/
structure DumbbellSystem where
  pairs : List DumbbellPair

def total_weight (system : DumbbellSystem) : ℕ :=
  system.pairs.map (λ pair => 2 * pair.weight) |>.sum

theorem dumbbell_system_weight :
  ∀ (system : DumbbellSystem),
    system.pairs = [
      DumbbellPair.mk 3,
      DumbbellPair.mk 5,
      DumbbellPair.mk 8
    ] →
    total_weight system = 32 := by
  sorry

end dumbbell_system_weight_l1048_104836


namespace round_trip_average_speed_l1048_104840

/-- Calculates the average speed of a round trip flight with wind effects -/
theorem round_trip_average_speed
  (up_airspeed : ℝ)
  (up_tailwind : ℝ)
  (down_airspeed : ℝ)
  (down_headwind : ℝ)
  (h1 : up_airspeed = 110)
  (h2 : up_tailwind = 20)
  (h3 : down_airspeed = 88)
  (h4 : down_headwind = 15) :
  (up_airspeed + up_tailwind + (down_airspeed - down_headwind)) / 2 = 101.5 := by
sorry

end round_trip_average_speed_l1048_104840


namespace angle_c_in_triangle_l1048_104827

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end angle_c_in_triangle_l1048_104827


namespace lcm_of_10_14_20_l1048_104848

theorem lcm_of_10_14_20 : Nat.lcm (Nat.lcm 10 14) 20 = 140 := by
  sorry

end lcm_of_10_14_20_l1048_104848


namespace first_grade_enrollment_l1048_104876

theorem first_grade_enrollment (a : ℕ) : 
  (200 ≤ a ∧ a ≤ 300) →
  (∃ R : ℕ, a = 25 * R + 10) →
  (∃ L : ℕ, a = 30 * L - 15) →
  a = 285 := by
sorry

end first_grade_enrollment_l1048_104876


namespace quadratic_through_origin_l1048_104822

/-- A quadratic function of the form f(x) = ax² - 3x + a² - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + a^2 - 1

/-- Theorem: If f(0) = 0 and a > 0, then a = 1 -/
theorem quadratic_through_origin (a : ℝ) (h1 : f a 0 = 0) (h2 : a > 0) : a = 1 := by
  sorry

end quadratic_through_origin_l1048_104822


namespace vehicle_tire_usage_l1048_104886

/-- Calculates the miles each tire is used given the total miles traveled, 
    number of tires, and number of tires used at a time. -/
def milesPerTire (totalMiles : ℕ) (numTires : ℕ) (tiresUsed : ℕ) : ℚ :=
  (totalMiles * tiresUsed : ℚ) / numTires

/-- Proves that given the conditions of the problem, each tire is used for 32,000 miles -/
theorem vehicle_tire_usage :
  let totalMiles : ℕ := 48000
  let numTires : ℕ := 6
  let tiresUsed : ℕ := 4
  milesPerTire totalMiles numTires tiresUsed = 32000 := by
  sorry

#eval milesPerTire 48000 6 4

end vehicle_tire_usage_l1048_104886


namespace disjoint_subsets_remainder_l1048_104887

def S : Finset Nat := Finset.range 12

def count_disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) :
  count_disjoint_subsets S % 1000 = 625 := by
  sorry

#eval count_disjoint_subsets S % 1000

end disjoint_subsets_remainder_l1048_104887


namespace area_relationship_l1048_104858

/-- The area of an isosceles triangle with sides 17, 17, and 16. -/
def P : ℝ := sorry

/-- The area of a right triangle with legs 15 and 20. -/
def Q : ℝ := sorry

/-- Theorem stating the relationship between P and Q. -/
theorem area_relationship : P = (4/5) * Q := by sorry

end area_relationship_l1048_104858


namespace remaining_money_after_tickets_l1048_104882

/-- Calculates the remaining money after buying tickets -/
def remaining_money (olivia_money : ℕ) (nigel_money : ℕ) (num_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  olivia_money + nigel_money - num_tickets * ticket_price

/-- Proves that Olivia and Nigel have $83 left after buying tickets -/
theorem remaining_money_after_tickets : remaining_money 112 139 6 28 = 83 := by
  sorry

end remaining_money_after_tickets_l1048_104882


namespace probability_all_female_committee_l1048_104880

def total_group_size : ℕ := 8
def num_females : ℕ := 5
def num_males : ℕ := 3
def committee_size : ℕ := 3

theorem probability_all_female_committee :
  (Nat.choose num_females committee_size : ℚ) / (Nat.choose total_group_size committee_size) = 5 / 28 := by
  sorry

end probability_all_female_committee_l1048_104880


namespace carlas_chickens_l1048_104826

theorem carlas_chickens (initial_chickens : ℕ) : 
  (initial_chickens : ℝ) - 0.4 * initial_chickens + 10 * (0.4 * initial_chickens) = 1840 →
  initial_chickens = 400 := by
  sorry

end carlas_chickens_l1048_104826


namespace max_profit_at_11_l1048_104879

/-- The cost price of each item in yuan -/
def cost_price : ℝ := 8

/-- The initial selling price in yuan -/
def initial_price : ℝ := 9

/-- The initial daily sales volume at the initial price -/
def initial_volume : ℝ := 20

/-- The rate at which sales volume decreases per yuan increase in price -/
def volume_decrease_rate : ℝ := 4

/-- The daily sales volume as a function of the selling price -/
def sales_volume (price : ℝ) : ℝ :=
  initial_volume - volume_decrease_rate * (price - initial_price)

/-- The daily profit as a function of the selling price -/
def daily_profit (price : ℝ) : ℝ :=
  sales_volume price * (price - cost_price)

/-- The theorem stating that the daily profit is maximized at 11 yuan -/
theorem max_profit_at_11 :
  ∃ (max_price : ℝ), max_price = 11 ∧
  ∀ (price : ℝ), price ≥ initial_price →
  daily_profit price ≤ daily_profit max_price :=
sorry

end max_profit_at_11_l1048_104879


namespace f_sin_pi_12_l1048_104860

theorem f_sin_pi_12 (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = - (Real.sqrt 3) / 2 := by
  sorry

end f_sin_pi_12_l1048_104860


namespace football_shoes_cost_l1048_104834

-- Define the costs and amounts
def football_cost : ℚ := 3.75
def shorts_cost : ℚ := 2.40
def zachary_has : ℚ := 10
def zachary_needs : ℚ := 8

-- Define the theorem
theorem football_shoes_cost :
  let total_cost := zachary_has + zachary_needs
  let other_items_cost := football_cost + shorts_cost
  total_cost - other_items_cost = 11.85 := by sorry

end football_shoes_cost_l1048_104834


namespace f_of_f_10_eq_2_l1048_104820

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else Real.log x / Real.log 2

theorem f_of_f_10_eq_2 : f (f 10) = 2 := by
  sorry

end f_of_f_10_eq_2_l1048_104820


namespace total_jelly_beans_l1048_104801

/-- The number of jelly beans needed to fill a large drinking glass -/
def large_glass_beans : ℕ := 50

/-- The number of jelly beans needed to fill a small drinking glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- Theorem stating the total number of jelly beans needed to fill all glasses -/
theorem total_jelly_beans :
  num_large_glasses * large_glass_beans + num_small_glasses * small_glass_beans = 325 := by
  sorry

end total_jelly_beans_l1048_104801


namespace triangle_side_length_l1048_104829

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = Real.sqrt 13 :=
by sorry

end triangle_side_length_l1048_104829


namespace engineering_collections_l1048_104896

/-- Represents the count of each letter in "ENGINEERING" -/
structure LetterCount where
  e : Nat -- vowel
  n : Nat -- consonant
  g : Nat -- consonant
  r : Nat -- consonant
  i : Nat -- consonant

/-- Represents a collection of letters -/
structure LetterCollection where
  vowels : Nat
  consonants : Nat

/-- Checks if a letter collection is valid -/
def isValidCollection (lc : LetterCollection) : Prop :=
  lc.vowels = 3 ∧ lc.consonants = 3

/-- Counts the number of distinct letter collections -/
noncomputable def countDistinctCollections (word : LetterCount) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem engineering_collections (word : LetterCount) 
  (h1 : word.e = 5) -- number of E's
  (h2 : word.n = 2) -- number of N's
  (h3 : word.g = 3) -- number of G's
  (h4 : word.r = 1) -- number of R's
  (h5 : word.i = 1) -- number of I's
  : countDistinctCollections word = 13 := by sorry

end engineering_collections_l1048_104896


namespace circles_intersect_l1048_104835

/-- Circle C1 with equation x^2 + y^2 - 2x - 3 = 0 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

/-- Circle C2 with equation x^2 + y^2 - 4x + 2y + 4 = 0 -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

/-- Two circles are intersecting if they have at least one point in common -/
def intersecting (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, C1 x y ∧ C2 x y

/-- The circles C1 and C2 are intersecting -/
theorem circles_intersect : intersecting C1 C2 := by
  sorry

end circles_intersect_l1048_104835


namespace gold_coin_distribution_l1048_104817

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.ierema + 70 = w.yuliy) ∧ (w.foma - 40 = w.yuliy)

/-- The theorem to prove -/
theorem gold_coin_distribution (w : MerchantWealth) 
  (h : problem_conditions w) : w.foma - 55 = w.ierema + 55 := by
  sorry

#check gold_coin_distribution

end gold_coin_distribution_l1048_104817


namespace race_finish_order_l1048_104872

theorem race_finish_order (total_students : ℕ) (before_yoongi : ℕ) (after_yoongi : ℕ) : 
  total_students = 20 → before_yoongi = 11 → after_yoongi = total_students - (before_yoongi + 1) → after_yoongi = 8 := by
  sorry

end race_finish_order_l1048_104872


namespace both_parents_single_eyelids_sufficient_not_necessary_l1048_104802

-- Define the possible genotypes
inductive Genotype
  | AA
  | Aa
  | aa

-- Define the phenotype (eyelid type)
inductive Phenotype
  | Double
  | Single

-- Function to determine phenotype from genotype
def phenotype (g : Genotype) : Phenotype :=
  match g with
  | Genotype.AA => Phenotype.Double
  | Genotype.Aa => Phenotype.Double
  | Genotype.aa => Phenotype.Single

-- Function to model gene inheritance
def inheritGene (parent1 : Genotype) (parent2 : Genotype) : Genotype :=
  sorry

-- Define what it means for both parents to have single eyelids
def bothParentsSingleEyelids (parent1 : Genotype) (parent2 : Genotype) : Prop :=
  phenotype parent1 = Phenotype.Single ∧ phenotype parent2 = Phenotype.Single

-- Define what it means for a child to have single eyelids
def childSingleEyelids (child : Genotype) : Prop :=
  phenotype child = Phenotype.Single

-- Theorem stating that "both parents have single eyelids" is sufficient but not necessary
theorem both_parents_single_eyelids_sufficient_not_necessary :
  (∀ (parent1 parent2 : Genotype),
    bothParentsSingleEyelids parent1 parent2 →
    childSingleEyelids (inheritGene parent1 parent2)) ∧
  (∃ (parent1 parent2 : Genotype),
    childSingleEyelids (inheritGene parent1 parent2) ∧
    ¬bothParentsSingleEyelids parent1 parent2) :=
  sorry

end both_parents_single_eyelids_sufficient_not_necessary_l1048_104802


namespace cookie_drop_count_l1048_104811

/-- Represents the number of cookies of each type made by Alice and Bob --/
structure CookieCount where
  chocolate_chip : ℕ
  sugar : ℕ
  oatmeal_raisin : ℕ
  peanut_butter : ℕ
  snickerdoodle : ℕ
  white_chocolate_macadamia : ℕ

/-- Calculates the total number of cookies --/
def total_cookies (c : CookieCount) : ℕ :=
  c.chocolate_chip + c.sugar + c.oatmeal_raisin + c.peanut_butter + c.snickerdoodle + c.white_chocolate_macadamia

theorem cookie_drop_count 
  (initial_cookies : CookieCount)
  (initial_dropped : CookieCount)
  (additional_cookies : CookieCount)
  (final_edible_cookies : ℕ) :
  total_cookies initial_cookies + total_cookies additional_cookies - final_edible_cookies = 139 :=
by sorry

end cookie_drop_count_l1048_104811


namespace beach_trip_duration_l1048_104846

-- Define the variables
def seashells_per_day : ℕ := 7
def total_seashells : ℕ := 35

-- Define the function to calculate the number of days
def days_at_beach : ℕ := total_seashells / seashells_per_day

-- Theorem statement
theorem beach_trip_duration : days_at_beach = 5 := by
  sorry

end beach_trip_duration_l1048_104846


namespace coefficient_x6_sum_binomial_expansions_l1048_104808

theorem coefficient_x6_sum_binomial_expansions :
  let f (n : ℕ) := (1 + X : Polynomial ℚ)^n
  let expansion := f 5 + f 6 + f 7
  (expansion.coeff 6 : ℚ) = 8 := by
  sorry

end coefficient_x6_sum_binomial_expansions_l1048_104808


namespace solve_selinas_shirt_sales_l1048_104892

/-- Represents the problem of determining how many shirts Selina sold. -/
def SelinasShirtSales : Prop :=
  let pants_price : ℕ := 5
  let shorts_price : ℕ := 3
  let shirt_price : ℕ := 4
  let pants_sold : ℕ := 3
  let shorts_sold : ℕ := 5
  let shirts_bought : ℕ := 2
  let shirt_buy_price : ℕ := 10
  let remaining_money : ℕ := 30
  ∃ (shirts_sold : ℕ),
    shirts_sold * shirt_price + 
    pants_sold * pants_price + 
    shorts_sold * shorts_price = 
    remaining_money + shirts_bought * shirt_buy_price ∧
    shirts_sold = 5

theorem solve_selinas_shirt_sales : SelinasShirtSales := by
  sorry

#check solve_selinas_shirt_sales

end solve_selinas_shirt_sales_l1048_104892


namespace problem_statement_l1048_104856

theorem problem_statement :
  (∀ (x : ℕ), x > 0 → (1/2 : ℝ)^x ≥ (1/3 : ℝ)^x) ∧
  ¬(∃ (x : ℕ), x > 0 ∧ (2 : ℝ)^x + (2 : ℝ)^(1-x) = 2 * Real.sqrt 2) :=
by sorry

end problem_statement_l1048_104856


namespace cubic_sum_inequality_l1048_104890

theorem cubic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 * b + b^3 * c + c^3 * a ≥ a * b * c * (a + b + c) := by
  sorry

end cubic_sum_inequality_l1048_104890


namespace binomial_150_150_l1048_104853

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_l1048_104853


namespace subset_quadratic_linear_l1048_104870

theorem subset_quadratic_linear (a : ℝ) : 
  let M : Set ℝ := {x | x^2 + x - 6 = 0}
  let N : Set ℝ := {x | a*x - 1 = 0}
  N ⊆ M → (a = 1/2 ∨ a = -1/3) :=
by
  sorry

#check subset_quadratic_linear

end subset_quadratic_linear_l1048_104870


namespace frank_candy_count_l1048_104866

theorem frank_candy_count (bags : ℕ) (pieces_per_bag : ℕ) 
  (h1 : bags = 26) (h2 : pieces_per_bag = 33) : 
  bags * pieces_per_bag = 858 := by
  sorry

end frank_candy_count_l1048_104866


namespace q_necessary_not_sufficient_for_p_l1048_104897

theorem q_necessary_not_sufficient_for_p :
  (∀ x : ℝ, |x| < 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ |x| ≥ 1) := by
  sorry

end q_necessary_not_sufficient_for_p_l1048_104897


namespace mans_rate_in_still_water_l1048_104855

/-- Given a man's downstream and upstream speeds, and the rate of current,
    calculate the man's rate in still water. -/
theorem mans_rate_in_still_water
  (downstream_speed : ℝ)
  (upstream_speed : ℝ)
  (current_rate : ℝ)
  (h1 : downstream_speed = 45)
  (h2 : upstream_speed = 23)
  (h3 : current_rate = 11) :
  (downstream_speed + upstream_speed) / 2 = 34 := by
sorry

end mans_rate_in_still_water_l1048_104855


namespace computation_problems_count_l1048_104864

theorem computation_problems_count (total_problems : ℕ) (comp_points : ℕ) (word_points : ℕ) (total_points : ℕ) :
  total_problems = 30 →
  comp_points = 3 →
  word_points = 5 →
  total_points = 110 →
  ∃ (comp_count : ℕ) (word_count : ℕ),
    comp_count + word_count = total_problems ∧
    comp_count * comp_points + word_count * word_points = total_points ∧
    comp_count = 20 :=
by sorry

end computation_problems_count_l1048_104864


namespace penalty_kicks_required_l1048_104888

theorem penalty_kicks_required (total_players : ℕ) (goalies : ℕ) (h1 : total_players = 18) (h2 : goalies = 4) : 
  (total_players - goalies) * goalies = 68 := by
  sorry

end penalty_kicks_required_l1048_104888


namespace simplify_trig_expression_l1048_104828

theorem simplify_trig_expression : 
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) / 
  (Real.cos (20 * π / 180) - Real.sqrt (1 - Real.cos (160 * π / 180) ^ 2)) = 1 := by
  sorry

end simplify_trig_expression_l1048_104828


namespace nba_schedule_impossibility_l1048_104868

theorem nba_schedule_impossibility :
  ∀ (n k : ℕ) (x y z : ℕ),
    n = 30 →  -- Total number of teams
    k ≤ n →   -- Number of teams in one conference
    x + y + z = (n * 82) / 2 →  -- Total number of games
    82 * k = 2 * x + z →  -- Games played by teams in one conference
    82 * (n - k) = 2 * y + z →  -- Games played by teams in the other conference
    2 * z = x + y + z →  -- Inter-conference games are half of total games
    False :=
by sorry

end nba_schedule_impossibility_l1048_104868


namespace ice_cream_combinations_l1048_104805

/-- The number of ways to distribute n indistinguishable objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem ice_cream_combinations : 
  distribute 5 4 = combinations 8 3 := by sorry

end ice_cream_combinations_l1048_104805


namespace quadratic_inequalities_condition_l1048_104884

theorem quadratic_inequalities_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x, a₁ * x^2 + b₁ * x + c₁ > 0 ↔ a₂ * x^2 + b₂ * x + c₂ > 0) ↔
    (a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂) :=
by sorry

end quadratic_inequalities_condition_l1048_104884


namespace f_is_quadratic_l1048_104894

/-- A quadratic equation in terms of x is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² - x + 1 --/
def f (x : ℝ) : ℝ := x^2 - x + 1

/-- Theorem: f(x) = x² - x + 1 is a quadratic equation in terms of x --/
theorem f_is_quadratic : is_quadratic_in_x f := by
  sorry


end f_is_quadratic_l1048_104894


namespace melanie_dimes_count_l1048_104877

def final_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

theorem melanie_dimes_count : final_dimes 7 4 = 11 := by
  sorry

end melanie_dimes_count_l1048_104877


namespace max_pen_area_l1048_104832

/-- The maximum area of a rectangular pen with one side against a wall,
    given 30 meters of fencing for the other three sides. -/
theorem max_pen_area (total_fence : ℝ) (h_total_fence : total_fence = 30) :
  ∃ (width height : ℝ),
    width > 0 ∧
    height > 0 ∧
    width + 2 * height = total_fence ∧
    ∀ (w h : ℝ), w > 0 → h > 0 → w + 2 * h = total_fence →
      w * h ≤ width * height ∧
      width * height = 112 :=
by sorry

end max_pen_area_l1048_104832


namespace remaining_integers_l1048_104833

/-- The number of integers remaining in a set of 1 to 80 after removing multiples of 4 and 5 -/
theorem remaining_integers (n : ℕ) (hn : n = 80) : 
  n - (n / 4 + n / 5 - n / 20) = 48 := by
  sorry

end remaining_integers_l1048_104833


namespace possible_values_of_a_l1048_104862

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define the theorem
theorem possible_values_of_a :
  ∀ a : ℝ, (M ∩ N a = N a) → (a = -1 ∨ a = 0 ∨ a = 2/3) :=
by sorry

end possible_values_of_a_l1048_104862


namespace line_equation_and_range_l1048_104867

/-- A line passing through two points -/
structure Line where
  k : ℝ
  b : ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def Line.y_at (l : Line) (x : ℝ) : ℝ := l.k * x + l.b

theorem line_equation_and_range (l : Line) 
  (h1 : l.y_at (-1) = 2)
  (h2 : l.y_at 2 = 5) :
  (∀ x, l.y_at x = x + 3) ∧ 
  (∀ x, l.y_at x > 0 ↔ x > -3) := by
  sorry


end line_equation_and_range_l1048_104867


namespace grandson_age_l1048_104878

/-- Given the ages of three family members satisfying certain conditions,
    prove that the youngest member (grandson) is 20 years old. -/
theorem grandson_age (grandson_age son_age markus_age : ℕ) : 
  son_age = 2 * grandson_age →
  markus_age = 2 * son_age →
  grandson_age + son_age + markus_age = 140 →
  grandson_age = 20 := by
  sorry

end grandson_age_l1048_104878


namespace root_sum_of_coefficients_l1048_104851

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def isRoot (z : ℂ) (b c : ℝ) : Prop :=
  z^2 + b * z + c = 0

-- Theorem statement
theorem root_sum_of_coefficients :
  ∀ (b c : ℝ), isRoot (2 + i) b c → b + c = 1 :=
by sorry

end root_sum_of_coefficients_l1048_104851


namespace expected_total_cost_is_350_l1048_104861

/-- Represents the outcome of a single test -/
inductive TestResult
| Defective
| NonDefective

/-- Represents the possible total costs of the testing process -/
inductive TotalCost
| Cost200
| Cost300
| Cost400

/-- The probability of getting a specific test result -/
def testProbability (result : TestResult) : ℚ :=
  match result with
  | TestResult.Defective => 2/5
  | TestResult.NonDefective => 3/5

/-- The probability of getting a specific total cost -/
def costProbability (cost : TotalCost) : ℚ :=
  match cost with
  | TotalCost.Cost200 => 1/10
  | TotalCost.Cost300 => 3/10
  | TotalCost.Cost400 => 3/5

/-- The cost in yuan for a specific total cost outcome -/
def costValue (cost : TotalCost) : ℚ :=
  match cost with
  | TotalCost.Cost200 => 200
  | TotalCost.Cost300 => 300
  | TotalCost.Cost400 => 400

/-- The expected value of the total cost -/
def expectedTotalCost : ℚ :=
  (costValue TotalCost.Cost200 * costProbability TotalCost.Cost200) +
  (costValue TotalCost.Cost300 * costProbability TotalCost.Cost300) +
  (costValue TotalCost.Cost400 * costProbability TotalCost.Cost400)

theorem expected_total_cost_is_350 :
  expectedTotalCost = 350 := by sorry

end expected_total_cost_is_350_l1048_104861


namespace yadav_expenditure_l1048_104873

/-- Represents Mr. Yadav's monthly salary in some monetary unit -/
def monthly_salary : ℝ := sorry

/-- Represents the percentage of salary spent on consumable items -/
def consumable_percentage : ℝ := 0.6

/-- Represents the percentage of remaining salary spent on clothes and transport -/
def clothes_transport_percentage : ℝ := 0.5

/-- Represents the yearly savings -/
def yearly_savings : ℝ := 24624

theorem yadav_expenditure :
  let remaining_after_consumables := monthly_salary * (1 - consumable_percentage)
  let clothes_transport_expenditure := remaining_after_consumables * clothes_transport_percentage
  let monthly_savings := yearly_savings / 12
  clothes_transport_expenditure = 2052 := by sorry

end yadav_expenditure_l1048_104873


namespace min_value_theorem_l1048_104854

/-- The minimum value of 1/m + 2/n given the constraints -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_theorem_l1048_104854


namespace total_material_is_correct_l1048_104819

/-- The amount of sand required for the renovation project in truck-loads -/
def sand : ℚ := 0.16666666666666666

/-- The amount of dirt required for the renovation project in truck-loads -/
def dirt : ℚ := 0.3333333333333333

/-- The amount of cement required for the renovation project in truck-loads -/
def cement : ℚ := 0.16666666666666666

/-- The total amount of material required for the renovation project in truck-loads -/
def total_material : ℚ := sand + dirt + cement

/-- Theorem stating that the total amount of material required is 0.6666666666666666 truck-loads -/
theorem total_material_is_correct : total_material = 0.6666666666666666 := by
  sorry

end total_material_is_correct_l1048_104819


namespace cube_folding_preserves_adjacency_l1048_104807

/-- Represents a face of the cube -/
inductive Face : Type
| One
| Two
| Three
| Four
| Five
| Six

/-- Represents the net of the cube -/
structure CubeNet :=
(faces : List Face)
(adjacent : Face → Face → Bool)

/-- Represents the folded cube -/
structure FoldedCube :=
(faces : List Face)
(adjacent : Face → Face → Bool)

/-- Theorem stating that the face adjacencies in the folded cube
    must match the adjacencies in the original net -/
theorem cube_folding_preserves_adjacency (net : CubeNet) (cube : FoldedCube) :
  (net.faces = cube.faces) →
  (∀ (f1 f2 : Face), net.adjacent f1 f2 = cube.adjacent f1 f2) :=
sorry

end cube_folding_preserves_adjacency_l1048_104807


namespace band_competition_l1048_104813

theorem band_competition (flute trumpet trombone drummer clarinet french_horn : ℕ) : 
  trumpet = 3 * flute ∧ 
  trombone = trumpet - 8 ∧ 
  drummer = trombone + 11 ∧ 
  clarinet = 2 * flute ∧ 
  french_horn = trombone + 3 ∧ 
  flute + trumpet + trombone + drummer + clarinet + french_horn = 65 → 
  flute = 6 := by sorry

end band_competition_l1048_104813


namespace triangle_with_semiprime_angles_l1048_104823

/-- A number is semi-prime if it's a product of exactly two primes (not necessarily distinct) -/
def IsSemiPrime (n : ℕ) : Prop := ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q

/-- The smallest semi-prime number -/
def SmallestSemiPrime : ℕ := 4

theorem triangle_with_semiprime_angles (p q : ℕ) :
  p = 2 * q →
  IsSemiPrime p →
  IsSemiPrime q →
  (p = SmallestSemiPrime ∨ q = SmallestSemiPrime) →
  ∃ x : ℕ, x = 168 ∧ p + q + x = 180 :=
sorry

end triangle_with_semiprime_angles_l1048_104823


namespace train_speed_calculation_l1048_104850

/-- Proves that a train of length 60 m crossing an electric pole in 1.4998800095992322 seconds has a speed of approximately 11.112 km/hr -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) 
  (h1 : train_length = 60) 
  (h2 : crossing_time = 1.4998800095992322) : 
  ∃ (speed : Real), abs (speed - 11.112) < 0.001 ∧ 
  speed = (train_length / crossing_time) * (3600 / 1000) := by
  sorry

end train_speed_calculation_l1048_104850


namespace max_abc_value_l1048_104809

theorem max_abc_value (a b c : ℕ+) 
  (h1 : a * b + b * c = 518)
  (h2 : a * b - a * c = 360) :
  (∀ x y z : ℕ+, x * y + y * z = 518 → x * y - x * z = 360 → a * b * c ≥ x * y * z) ∧ 
  a * b * c = 1008 :=
sorry

end max_abc_value_l1048_104809


namespace problem_statement_l1048_104825

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → 
  a^2009 + b^2009 = -1 := by
  sorry

end problem_statement_l1048_104825


namespace sin_45_degrees_l1048_104899

/-- The sine of 45 degrees is equal to √2/2 -/
theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by sorry

end sin_45_degrees_l1048_104899


namespace sandy_dog_puppies_l1048_104831

/-- The number of puppies Sandy now has -/
def total_puppies : ℕ := 12

/-- The number of puppies Sandy's friend gave her -/
def friend_puppies : ℕ := 4

/-- The number of puppies Sandy's dog initially had -/
def initial_puppies : ℕ := total_puppies - friend_puppies

theorem sandy_dog_puppies : initial_puppies = 8 := by
  sorry

end sandy_dog_puppies_l1048_104831


namespace johnny_age_puzzle_l1048_104847

/-- Represents Johnny's age now -/
def current_age : ℕ := 8

/-- Represents the number of years into the future Johnny is referring to -/
def future_years : ℕ := 2

/-- Theorem stating that the number of years into the future Johnny was referring to is correct -/
theorem johnny_age_puzzle :
  (current_age + future_years = 2 * (current_age - 3)) ∧
  (future_years = 2) := by
  sorry

end johnny_age_puzzle_l1048_104847


namespace question_mark_value_l1048_104810

theorem question_mark_value : ∃ x : ℚ, (786 * x) / 30 = 1938.8 ∧ x = 74 := by sorry

end question_mark_value_l1048_104810


namespace streamer_profit_formula_l1048_104816

/-- Streamer's daily profit function -/
def daily_profit (x : ℝ) : ℝ :=
  (x - 50) * (300 + 3 * (99 - x))

/-- Initial selling price -/
def initial_price : ℝ := 99

/-- Initial daily sales volume -/
def initial_sales : ℝ := 300

/-- Sales volume increase per yuan price decrease -/
def sales_increase_rate : ℝ := 3

/-- Cost and expenses per item -/
def cost_per_item : ℝ := 50

theorem streamer_profit_formula (x : ℝ) :
  daily_profit x = (x - cost_per_item) * (initial_sales + sales_increase_rate * (initial_price - x)) :=
by sorry

end streamer_profit_formula_l1048_104816


namespace basketball_score_proof_l1048_104857

theorem basketball_score_proof (total_score : ℕ) (two_point_shots : ℕ) (three_point_shots : ℕ) :
  total_score = 16 ∧
  two_point_shots = three_point_shots + 3 ∧
  2 * two_point_shots + 3 * three_point_shots = total_score →
  three_point_shots = 2 := by
sorry

end basketball_score_proof_l1048_104857


namespace system_solution_range_l1048_104824

theorem system_solution_range (x y m : ℝ) : 
  (x + 2*y = m + 4) →
  (2*x + y = 2*m - 1) →
  (x + y < 2) →
  (x - y < 4) →
  m < 1 := by
  sorry

end system_solution_range_l1048_104824


namespace sequence_is_arithmetic_l1048_104815

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

theorem sequence_is_arithmetic (a : ℕ → ℝ) 
    (h : ∀ n, 3 * a (n + 1) = 3 * a n + 1) : 
    is_arithmetic_sequence a (1/3) := by
  sorry

end sequence_is_arithmetic_l1048_104815


namespace negation_of_universal_proposition_l1048_104803

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x - 6 > 0) ↔ (∃ x : ℝ, x^2 + 2*x - 6 ≤ 0) := by sorry

end negation_of_universal_proposition_l1048_104803
