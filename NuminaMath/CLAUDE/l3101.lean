import Mathlib

namespace NUMINAMATH_CALUDE_pen_profit_percentage_l3101_310175

/-- Calculates the profit percentage for a retailer selling pens --/
theorem pen_profit_percentage
  (num_pens : ℕ)
  (price_pens : ℕ)
  (discount_percent : ℚ)
  (h1 : num_pens = 120)
  (h2 : price_pens = 36)
  (h3 : discount_percent = 1/100)
  : ∃ (profit_percent : ℚ), profit_percent = 230/100 :=
by sorry

end NUMINAMATH_CALUDE_pen_profit_percentage_l3101_310175


namespace NUMINAMATH_CALUDE_star_3_5_l3101_310138

-- Define the star operation
def star (a b : ℝ) : ℝ := (a + b)^2 + (a - b)^2

-- Theorem statement
theorem star_3_5 : star 3 5 = 68 := by sorry

end NUMINAMATH_CALUDE_star_3_5_l3101_310138


namespace NUMINAMATH_CALUDE_sector_area_l3101_310136

theorem sector_area (angle : Real) (radius : Real) (area : Real) : 
  angle = 120 * (π / 180) →  -- Convert 120° to radians
  radius = 10 →
  area = (angle / (2 * π)) * π * radius^2 →
  area = 100 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l3101_310136


namespace NUMINAMATH_CALUDE_tangent_segments_area_l3101_310108

theorem tangent_segments_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 4) :
  let inner_radius := r
  let outer_radius := Real.sqrt (r^2 + (l/2)^2)
  (π * outer_radius^2 - π * inner_radius^2) = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_tangent_segments_area_l3101_310108


namespace NUMINAMATH_CALUDE_johns_allowance_l3101_310112

/-- John's weekly allowance problem -/
theorem johns_allowance :
  ∀ (A : ℚ),
  (A > 0) →
  (3/5 * A + 1/3 * (A - 3/5 * A) + 88/100 = A) →
  A = 33/10 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l3101_310112


namespace NUMINAMATH_CALUDE_total_population_l3101_310123

theorem total_population (b g t : ℕ) : 
  b = 4 * g ∧ g = 8 * t → b + g + t = 41 * t :=
by sorry

end NUMINAMATH_CALUDE_total_population_l3101_310123


namespace NUMINAMATH_CALUDE_unique_solution_fourth_root_equation_l3101_310133

/-- The equation √⁴(58 - 3x) + √⁴(26 + 3x) = 5 has a unique solution -/
theorem unique_solution_fourth_root_equation :
  ∃! x : ℝ, (58 - 3*x)^(1/4) + (26 + 3*x)^(1/4) = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_fourth_root_equation_l3101_310133


namespace NUMINAMATH_CALUDE_f_difference_l3101_310107

/-- The function f(x) = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Theorem: For all real x and h, f(x + h) - f(x) = h(6x + 3h - 4) -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h - 4) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l3101_310107


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l3101_310143

theorem largest_integer_inequality : 
  (∀ x : ℤ, x ≤ 3 → (x : ℚ) / 5 + 6 / 7 < 8 / 5) ∧ 
  (4 : ℚ) / 5 + 6 / 7 ≥ 8 / 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l3101_310143


namespace NUMINAMATH_CALUDE_parent_teacher_night_duration_l3101_310176

def time_to_school : ℕ := 20
def time_from_school : ℕ := 20
def total_time : ℕ := 110

theorem parent_teacher_night_duration :
  total_time - (time_to_school + time_from_school) = 70 :=
by sorry

end NUMINAMATH_CALUDE_parent_teacher_night_duration_l3101_310176


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3101_310172

theorem greatest_divisor_with_remainders : 
  let a := 150 - 50
  let b := 230 - 5
  let c := 175 - 25
  Nat.gcd a (Nat.gcd b c) = 25 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3101_310172


namespace NUMINAMATH_CALUDE_x_coordinate_of_Q_l3101_310161

/-- A line through the origin equidistant from two points -/
structure EquidistantLine where
  slope : ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  is_equidistant : ∀ (x y : ℝ), y = slope * x → 
    (x - P.1)^2 + (y - P.2)^2 = (x - Q.1)^2 + (y - Q.2)^2

/-- Theorem: Given the conditions, the x-coordinate of Q is 2.5 -/
theorem x_coordinate_of_Q (L : EquidistantLine) 
  (h_slope : L.slope = 0.8)
  (h_Q_y : L.Q.2 = 2) :
  L.Q.1 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_of_Q_l3101_310161


namespace NUMINAMATH_CALUDE_largest_n_with_conditions_l3101_310131

theorem largest_n_with_conditions : 
  ∃ (n : ℕ), n = 4513 ∧ 
  (∃ (m : ℕ), n^2 = (m+1)^3 - m^3) ∧
  (∃ (k : ℕ), 2*n + 99 = k^2) ∧
  (∀ (n' : ℕ), n' > n → 
    (¬∃ (m : ℕ), n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℕ), 2*n' + 99 = k^2)) := by
  sorry

#check largest_n_with_conditions

end NUMINAMATH_CALUDE_largest_n_with_conditions_l3101_310131


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3101_310192

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (1/3) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3101_310192


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1023_l3101_310152

theorem largest_prime_factor_of_1023 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1023 ∧ ∀ q, Nat.Prime q → q ∣ 1023 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1023_l3101_310152


namespace NUMINAMATH_CALUDE_inequalities_solution_l3101_310178

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 < 0
def inequality2 (x : ℝ) : Prop := (2 * x) / (x + 1) ≥ 1

-- State the theorem
theorem inequalities_solution :
  (∀ x : ℝ, inequality1 x ↔ (1/2 < x ∧ x < 1)) ∧
  (∀ x : ℝ, inequality2 x ↔ (x < -1 ∨ x ≥ 1)) :=
sorry

end NUMINAMATH_CALUDE_inequalities_solution_l3101_310178


namespace NUMINAMATH_CALUDE_mariels_dogs_count_l3101_310183

/-- The number of dogs Mariel is walking -/
def mariels_dogs : ℕ :=
  let total_legs : ℕ := 36
  let num_walkers : ℕ := 2
  let other_walker_dogs : ℕ := 3
  let human_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let total_dogs : ℕ := (total_legs - num_walkers * human_legs) / dog_legs
  total_dogs - other_walker_dogs

theorem mariels_dogs_count : mariels_dogs = 5 := by
  sorry

end NUMINAMATH_CALUDE_mariels_dogs_count_l3101_310183


namespace NUMINAMATH_CALUDE_divisibility_implication_l3101_310105

/-- Represents a three-digit number with non-zero digits -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digit_sum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- The product of digits of a three-digit number -/
def digit_product (n : ThreeDigitNumber) : Nat :=
  n.a * n.b * n.c

theorem divisibility_implication (n : ThreeDigitNumber) :
  (value n % digit_sum n = 0) ∧ (value n % digit_product n = 0) →
  90 * n.a % digit_sum n = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l3101_310105


namespace NUMINAMATH_CALUDE_even_function_properties_l3101_310181

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define what it means for a function to be symmetric about a vertical line
def symmetric_about_vertical_line (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Define what it means for a function to be symmetric about the y-axis
def symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem even_function_properties (h : is_even (fun x => f (x + 1))) :
  (symmetric_about_vertical_line f 1) ∧
  (symmetric_about_y_axis (fun x => f (x + 1))) ∧
  (∀ x, f (1 + x) = f (1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_even_function_properties_l3101_310181


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l3101_310141

theorem rectangle_cylinder_volume_ratio :
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 9
  let cylinder_a_radius : ℝ := rectangle_width / (2 * Real.pi)
  let cylinder_a_height : ℝ := rectangle_height
  let cylinder_b_radius : ℝ := rectangle_height / (2 * Real.pi)
  let cylinder_b_height : ℝ := rectangle_width
  let volume_a : ℝ := Real.pi * cylinder_a_radius^2 * cylinder_a_height
  let volume_b : ℝ := Real.pi * cylinder_b_radius^2 * cylinder_b_height
  volume_b / volume_a = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l3101_310141


namespace NUMINAMATH_CALUDE_circle_tangency_problem_l3101_310157

theorem circle_tangency_problem (r : ℕ) : 
  (0 < r ∧ r < 60 ∧ 120 % r = 0) → 
  (∃ (S : Finset ℕ), S = {x : ℕ | 0 < x ∧ x < 60 ∧ 120 % x = 0} ∧ Finset.card S = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_problem_l3101_310157


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3101_310186

/-- Properties of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  angle_F1PF2 : ℝ
  area_F1PF2 : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  e_eq : e = 2
  angle_eq : angle_F1PF2 = Real.pi / 2
  area_eq : area_F1PF2 = 3

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ := 2 * h.a

/-- Theorem: The length of the real axis of the given hyperbola is 2 -/
theorem hyperbola_real_axis_length (h : Hyperbola) : real_axis_length h = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3101_310186


namespace NUMINAMATH_CALUDE_amanda_camila_hike_ratio_l3101_310148

/-- Proves that the ratio of Amanda's hikes to Camila's hikes is 8:1 --/
theorem amanda_camila_hike_ratio :
  let camila_hikes : ℕ := 7
  let steven_hikes : ℕ := camila_hikes + 4 * 16
  let amanda_hikes : ℕ := steven_hikes - 15
  amanda_hikes / camila_hikes = 8 := by
sorry

end NUMINAMATH_CALUDE_amanda_camila_hike_ratio_l3101_310148


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l3101_310117

noncomputable def f (m n x : ℝ) : ℝ := 2^x * m + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, f m n x = 0) ∧ 
  (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l3101_310117


namespace NUMINAMATH_CALUDE_soap_survey_households_l3101_310169

theorem soap_survey_households (total : ℕ) (neither : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 160 ∧
  neither = 80 ∧
  only_A = 60 ∧
  both = 5 →
  total = neither + only_A + both + 3 * both :=
by sorry

end NUMINAMATH_CALUDE_soap_survey_households_l3101_310169


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l3101_310149

/-- The amount of money Gwen received from her mom -/
def mom_money : ℕ := 8

/-- The difference between the money Gwen received from her mom and dad -/
def difference : ℕ := 3

/-- The amount of money Gwen received from her dad -/
def dad_money : ℕ := mom_money - difference

theorem gwen_birthday_money : dad_money = 5 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l3101_310149


namespace NUMINAMATH_CALUDE_car_speed_problem_l3101_310196

/-- Proves that the speed of Car A is 70 km/h given the conditions of the problem -/
theorem car_speed_problem (time : ℝ) (speed_B : ℝ) (ratio : ℝ) :
  time = 10 →
  speed_B = 35 →
  ratio = 2 →
  let distance_A := time * (ratio * speed_B)
  let distance_B := time * speed_B
  (distance_A / distance_B = ratio) →
  (ratio * speed_B = 70) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3101_310196


namespace NUMINAMATH_CALUDE_solve_average_salary_l3101_310151

def average_salary_problem (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) : Prop :=
  let total_salary := num_employees * (manager_salary / (num_employees + 1) - avg_increase)
  let new_total_salary := total_salary + manager_salary
  let new_average := new_total_salary / (num_employees + 1)
  (manager_salary / (num_employees + 1) - avg_increase) = 2400 ∧
  new_average = (manager_salary / (num_employees + 1) - avg_increase) + avg_increase

theorem solve_average_salary :
  average_salary_problem 24 4900 100 := by
  sorry

end NUMINAMATH_CALUDE_solve_average_salary_l3101_310151


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3101_310199

theorem inequality_system_solution : 
  let S := {x : ℤ | x > 0 ∧ 5 + 3*x < 13 ∧ (x+2)/3 - (x-1)/2 ≤ 2}
  S = {1, 2} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3101_310199


namespace NUMINAMATH_CALUDE_kittens_from_friends_proof_l3101_310137

/-- The number of kittens Joan's cat had initially -/
def initial_kittens : ℕ := 8

/-- The total number of kittens Joan has now -/
def total_kittens : ℕ := 10

/-- The number of kittens Joan got from her friends -/
def kittens_from_friends : ℕ := total_kittens - initial_kittens

theorem kittens_from_friends_proof :
  kittens_from_friends = total_kittens - initial_kittens :=
by sorry

end NUMINAMATH_CALUDE_kittens_from_friends_proof_l3101_310137


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3101_310165

-- Part 1
theorem problem_1 : 8 - (-4) / (2^2) * 3 = 11 := by sorry

-- Part 2
theorem problem_2 (x : ℝ) : 2 * x^2 + 3 * (2*x - x^2) = -x^2 + 6*x := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3101_310165


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l3101_310154

theorem magnitude_of_complex_power : 
  Complex.abs ((5 : ℂ) - (2 * Real.sqrt 3) * Complex.I) ^ 4 = 1369 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l3101_310154


namespace NUMINAMATH_CALUDE_exists_all_met_l3101_310121

-- Define a type for participants
variable (Participant : Type)

-- Define a relation for "has met"
variable (has_met : Participant → Participant → Prop)

-- Define the number of participants
variable (n : ℕ)

-- Assume there are at least 4 participants
variable (h_n : n ≥ 4)

-- Define the set of all participants
variable (participants : Finset Participant)

-- Assume the number of participants matches n
variable (h_card : participants.card = n)

-- State the condition that among any 4 participants, one has met the other 3
variable (h_four_met : ∀ (a b c d : Participant), a ∈ participants → b ∈ participants → c ∈ participants → d ∈ participants →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (has_met a b ∧ has_met a c ∧ has_met a d) ∨
  (has_met b a ∧ has_met b c ∧ has_met b d) ∨
  (has_met c a ∧ has_met c b ∧ has_met c d) ∨
  (has_met d a ∧ has_met d b ∧ has_met d c))

-- Theorem statement
theorem exists_all_met :
  ∃ (x : Participant), x ∈ participants ∧ ∀ (y : Participant), y ∈ participants → y ≠ x → has_met x y :=
sorry

end NUMINAMATH_CALUDE_exists_all_met_l3101_310121


namespace NUMINAMATH_CALUDE_max_shot_radius_l3101_310135

/-- Given a sphere of radius 3 cm from which 27 shots can be made, 
    prove that the maximum radius of each shot is 1 cm. -/
theorem max_shot_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 3 → n = 27 → (4 / 3 * Real.pi * R^3 = n * (4 / 3 * Real.pi * r^3)) → r ≤ 1 := by
  sorry

#check max_shot_radius

end NUMINAMATH_CALUDE_max_shot_radius_l3101_310135


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3101_310166

theorem modulus_of_complex_number (z : ℂ) (h : z = 3 - 2*Complex.I) : Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3101_310166


namespace NUMINAMATH_CALUDE_hollow_block_length_l3101_310146

/-- Represents a hollow rectangular block made of small cubes -/
structure HollowBlock where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the number of small cubes used in a hollow rectangular block -/
def cubesUsed (block : HollowBlock) : ℕ :=
  2 * (block.length * block.width + block.width * block.depth + block.length * block.depth) -
  4 * (block.length + block.width + block.depth) + 8 -
  ((block.length - 2) * (block.width - 2) * (block.depth - 2))

/-- Theorem stating that a hollow block with given dimensions uses 114 cubes and has a length of 10 -/
theorem hollow_block_length :
  ∃ (block : HollowBlock), block.width = 9 ∧ block.depth = 5 ∧ cubesUsed block = 114 ∧ block.length = 10 :=
by sorry

end NUMINAMATH_CALUDE_hollow_block_length_l3101_310146


namespace NUMINAMATH_CALUDE_prob_X_eq_three_l3101_310193

/-- A random variable X following a binomial distribution B(6, 1/2) -/
def X : ℕ → ℝ := sorry

/-- The probability mass function for X -/
def pmf (k : ℕ) : ℝ := sorry

/-- Theorem: The probability of X = 3 is 5/16 -/
theorem prob_X_eq_three : pmf 3 = 5/16 := by sorry

end NUMINAMATH_CALUDE_prob_X_eq_three_l3101_310193


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3101_310109

theorem fixed_point_of_exponential_function (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x-2) + 4
  f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3101_310109


namespace NUMINAMATH_CALUDE_combinatorial_equality_l3101_310153

theorem combinatorial_equality (n : ℕ) : 
  (Nat.choose n 3 = Nat.choose n 5) → n = 8 := by
sorry

end NUMINAMATH_CALUDE_combinatorial_equality_l3101_310153


namespace NUMINAMATH_CALUDE_lizard_spot_wrinkle_ratio_l3101_310125

/-- Represents a three-eyed lizard with wrinkles and spots. -/
structure Lizard where
  eyes : ℕ
  wrinkles : ℕ
  spots : ℕ

/-- The properties of our specific lizard. -/
def specialLizard : Lizard where
  eyes := 3
  wrinkles := 3 * 3
  spots := 3 + (3 * 3) - 69 + 3

theorem lizard_spot_wrinkle_ratio (l : Lizard) 
  (h1 : l.eyes = 3)
  (h2 : l.wrinkles = 3 * l.eyes)
  (h3 : l.eyes = l.spots + l.wrinkles - 69) :
  l.spots / l.wrinkles = 7 := by
  sorry

#eval specialLizard.spots / specialLizard.wrinkles

end NUMINAMATH_CALUDE_lizard_spot_wrinkle_ratio_l3101_310125


namespace NUMINAMATH_CALUDE_greatest_x_value_l3101_310195

theorem greatest_x_value : ∃ (x_max : ℚ), 
  (∀ x : ℚ, ((4*x - 16) / (3*x - 4))^2 + ((4*x - 16) / (3*x - 4)) = 6 → x ≤ x_max) ∧
  ((4*x_max - 16) / (3*x_max - 4))^2 + ((4*x_max - 16) / (3*x_max - 4)) = 6 ∧
  x_max = 28/13 := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3101_310195


namespace NUMINAMATH_CALUDE_potato_flour_weight_l3101_310179

theorem potato_flour_weight (potato_bags flour_bags total_weight weight_difference : ℕ) 
  (h1 : potato_bags = 15)
  (h2 : flour_bags = 12)
  (h3 : total_weight = 1710)
  (h4 : weight_difference = 30) :
  ∃ (potato_weight flour_weight : ℕ),
    potato_weight * potato_bags + flour_weight * flour_bags = total_weight ∧
    flour_weight = potato_weight + weight_difference ∧
    potato_weight = 50 ∧
    flour_weight = 80 := by
  sorry

end NUMINAMATH_CALUDE_potato_flour_weight_l3101_310179


namespace NUMINAMATH_CALUDE_tan_315_degrees_l3101_310188

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l3101_310188


namespace NUMINAMATH_CALUDE_total_passengers_l3101_310127

def bus_problem (initial_a initial_b new_a new_b : ℕ) : ℕ :=
  (initial_a + new_a) + (initial_b + new_b)

theorem total_passengers :
  bus_problem 4 7 13 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_l3101_310127


namespace NUMINAMATH_CALUDE_best_fit_highest_abs_r_model1_best_fit_l3101_310110

/-- Represents a linear regression model with its correlation coefficient -/
structure RegressionModel where
  r : ℝ

/-- Determines if a model is the best fit among a list of models -/
def isBestFit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, |model.r| ≥ |m.r|

theorem best_fit_highest_abs_r (models : List RegressionModel) (model : RegressionModel) 
    (h : model ∈ models) :
    isBestFit model models ↔ ∀ m ∈ models, |model.r| ≥ |m.r| := by
  sorry

/-- The four models from the problem -/
def model1 : RegressionModel := ⟨0.98⟩
def model2 : RegressionModel := ⟨0.80⟩
def model3 : RegressionModel := ⟨0.50⟩
def model4 : RegressionModel := ⟨0.25⟩

def allModels : List RegressionModel := [model1, model2, model3, model4]

theorem model1_best_fit : isBestFit model1 allModels := by
  sorry

end NUMINAMATH_CALUDE_best_fit_highest_abs_r_model1_best_fit_l3101_310110


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l3101_310126

theorem arctan_tan_difference (x y : Real) :
  Real.arctan (Real.tan (65 * π / 180) - 2 * Real.tan (40 * π / 180)) = 25 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l3101_310126


namespace NUMINAMATH_CALUDE_set_cardinality_lower_bound_l3101_310142

theorem set_cardinality_lower_bound
  (m : ℕ)
  (A : Finset ℤ)
  (B : Fin m → Finset ℤ)
  (h_m : m ≥ 2)
  (h_subset : ∀ k : Fin m, B k ⊆ A)
  (h_sum : ∀ k : Fin m, (B k).sum id = m ^ (k : ℕ).succ) :
  A.card ≥ m / 2 :=
sorry

end NUMINAMATH_CALUDE_set_cardinality_lower_bound_l3101_310142


namespace NUMINAMATH_CALUDE_smallest_positive_angle_2015_l3101_310160

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem smallest_positive_angle_2015 :
  ∃! θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ (-2015) ∧
  ∀ φ, 0 ≤ φ ∧ φ < 360 ∧ same_terminal_side φ (-2015) → θ ≤ φ :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_2015_l3101_310160


namespace NUMINAMATH_CALUDE_x_plus_q_equals_five_plus_two_q_l3101_310134

theorem x_plus_q_equals_five_plus_two_q (x q : ℝ) 
  (h1 : |x - 5| = q) 
  (h2 : x > 5) : 
  x + q = 5 + 2*q := by
sorry

end NUMINAMATH_CALUDE_x_plus_q_equals_five_plus_two_q_l3101_310134


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l3101_310128

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 50*x + c = (x + a)^2) → c = 625 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l3101_310128


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_l3101_310164

theorem sum_of_A_and_C (A B C : ℕ) : A = 238 → A = B + 143 → C = B + 304 → A + C = 637 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_l3101_310164


namespace NUMINAMATH_CALUDE_calculate_second_discount_other_discount_percentage_l3101_310159

/-- Given an article with a list price and two successive discounts, 
    calculate the second discount percentage. -/
theorem calculate_second_discount 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (first_discount : ℝ) : ℝ :=
  let price_after_first_discount := list_price * (1 - first_discount / 100)
  let second_discount := (price_after_first_discount - final_price) / price_after_first_discount * 100
  second_discount

/-- Prove that for an article with a list price of 70 units, 
    after applying two successive discounts, one of which is 10%, 
    resulting in a final price of 56.16 units, 
    the other discount percentage is approximately 10.857%. -/
theorem other_discount_percentage : 
  let result := calculate_second_discount 70 56.16 10
  ∃ ε > 0, abs (result - 10.857) < ε :=
sorry

end NUMINAMATH_CALUDE_calculate_second_discount_other_discount_percentage_l3101_310159


namespace NUMINAMATH_CALUDE_bike_truck_travel_time_indeterminate_equal_travel_time_l3101_310118

/-- Given a bike and a truck with the same speed covering the same distance,
    prove that their travel times are equal but indeterminate without knowing the speed. -/
theorem bike_truck_travel_time (distance : ℝ) (speed : ℝ) : 
  distance > 0 → speed > 0 → ∃ (time : ℝ), 
    time = distance / speed ∧ 
    (∀ (bike_time truck_time : ℝ), 
      bike_time = distance / speed → 
      truck_time = distance / speed → 
      bike_time = truck_time) :=
by sorry

/-- The specific distance covered by both vehicles -/
def covered_distance : ℝ := 72

/-- The speed difference between the bike and the truck -/
def speed_difference : ℝ := 0

/-- Theorem stating that the travel time for both vehicles is the same 
    but cannot be determined without knowing the speed -/
theorem indeterminate_equal_travel_time :
  ∃ (time : ℝ), 
    (∀ (bike_speed : ℝ), bike_speed > 0 →
      time = covered_distance / bike_speed) ∧
    (∀ (truck_speed : ℝ), truck_speed > 0 →
      time = covered_distance / truck_speed) ∧
    (∀ (bike_time truck_time : ℝ),
      bike_time = covered_distance / bike_speed →
      truck_time = covered_distance / truck_speed →
      bike_time = truck_time) :=
by sorry

end NUMINAMATH_CALUDE_bike_truck_travel_time_indeterminate_equal_travel_time_l3101_310118


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_one_l3101_310101

theorem product_of_fractions_equals_one :
  (5 / 3 : ℚ) * (6 / 10 : ℚ) * (15 / 9 : ℚ) * (12 / 20 : ℚ) *
  (25 / 15 : ℚ) * (18 / 30 : ℚ) * (35 / 21 : ℚ) * (24 / 40 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_one_l3101_310101


namespace NUMINAMATH_CALUDE_integral_bound_for_differentiable_function_l3101_310194

open Set
open MeasureTheory
open Interval
open Real

theorem integral_bound_for_differentiable_function 
  (f : ℝ → ℝ) 
  (hf_diff : DifferentiableOn ℝ f (Icc 0 1))
  (hf_zero : f 0 = 0 ∧ f 1 = 0)
  (hf_deriv_bound : ∀ x ∈ Icc 0 1, abs (deriv f x) ≤ 1) :
  abs (∫ x in Icc 0 1, f x) < (1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_integral_bound_for_differentiable_function_l3101_310194


namespace NUMINAMATH_CALUDE_trapezium_height_l3101_310102

theorem trapezium_height (a b h : ℝ) : 
  a = 20 → b = 18 → (1/2) * (a + b) * h = 247 → h = 13 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l3101_310102


namespace NUMINAMATH_CALUDE_system_solution_l3101_310155

theorem system_solution :
  ∃ (x y : ℝ), 
    (x + Real.sqrt (x + 2*y) - 2*y = 7/2) ∧
    (x^2 + x + 2*y - 4*y^2 = 27/2) ∧
    (x = 19/4) ∧ (y = 17/8) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3101_310155


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3101_310158

theorem min_value_of_expression (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 6) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3101_310158


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3101_310100

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ a ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3101_310100


namespace NUMINAMATH_CALUDE_fourth_group_size_l3101_310122

theorem fourth_group_size (total : ℕ) (group1 group2 group3 : ℕ) 
  (h1 : total = 24) 
  (h2 : group1 = 5) 
  (h3 : group2 = 8) 
  (h4 : group3 = 7) : 
  total - (group1 + group2 + group3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_size_l3101_310122


namespace NUMINAMATH_CALUDE_straight_line_angle_l3101_310144

/-- 
Given a straight line segment PQ with angle measurements of 90°, x°, and 20° along it,
prove that x = 70°.
-/
theorem straight_line_angle (x : ℝ) : 
  (90 : ℝ) + x + 20 = 180 → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_straight_line_angle_l3101_310144


namespace NUMINAMATH_CALUDE_carrie_vegetable_revenue_l3101_310156

/-- Represents the revenue calculation for Carrie's vegetable sales --/
theorem carrie_vegetable_revenue : 
  let tomatoes := 200
  let carrots := 350
  let eggplants := 120
  let cucumbers := 75
  let tomato_price := 1
  let carrot_price := 1.5
  let eggplant_price := 2.5
  let cucumber_price := 1.75
  let tomato_discount := 0.05
  let carrot_discount_price := 1.25
  let eggplant_free_per := 10
  let cucumber_discount := 0.1
  
  let tomato_revenue := tomatoes * (tomato_price * (1 - tomato_discount))
  let carrot_revenue := carrots * carrot_discount_price
  let eggplant_revenue := (eggplants - (eggplants / eggplant_free_per)) * eggplant_price
  let cucumber_revenue := cucumbers * (cucumber_price * (1 - cucumber_discount))
  
  tomato_revenue + carrot_revenue + eggplant_revenue + cucumber_revenue = 1015.625 := by
  sorry


end NUMINAMATH_CALUDE_carrie_vegetable_revenue_l3101_310156


namespace NUMINAMATH_CALUDE_exercise_goal_theorem_l3101_310187

/-- Calculates the total exercise time given daily exercise time, days per week, and number of weeks. -/
def totalExerciseTime (dailyTime : ℕ) (daysPerWeek : ℕ) (numWeeks : ℕ) : ℕ :=
  dailyTime * daysPerWeek * numWeeks

/-- Proves that exercising 1 hour a day for 5 days a week over 8 weeks results in 40 hours of total exercise time. -/
theorem exercise_goal_theorem :
  totalExerciseTime 1 5 8 = 40 := by
  sorry

#eval totalExerciseTime 1 5 8

end NUMINAMATH_CALUDE_exercise_goal_theorem_l3101_310187


namespace NUMINAMATH_CALUDE_inequality_proof_l3101_310184

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / (a + b)) * ((a + 2*b) / (a + 3*b)) < Real.sqrt (a / (a + 4*b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3101_310184


namespace NUMINAMATH_CALUDE_at_least_one_hit_probability_l3101_310115

theorem at_least_one_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.7) (h2 : p2 = 0.8) :
  p1 + p2 - p1 * p2 = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_hit_probability_l3101_310115


namespace NUMINAMATH_CALUDE_triple_angle_sine_sin_18_degrees_l3101_310129

open Real

-- Define the sum of sines formula
axiom sum_of_sines (α β : ℝ) : sin (α + β) = sin α * cos β + cos α * sin β

-- Define the double angle formula for sine
axiom double_angle_sine (α : ℝ) : sin (2 * α) = 2 * sin α * cos α

-- Define the relation between sine and cosine
axiom sine_cosine_relation (α : ℝ) : sin α = cos (π / 2 - α)

-- Theorem 1: Triple angle formula for sine
theorem triple_angle_sine (α : ℝ) : sin (3 * α) = 3 * sin α - 4 * (sin α)^3 := by sorry

-- Theorem 2: Value of sin 18°
theorem sin_18_degrees : sin (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by sorry

end NUMINAMATH_CALUDE_triple_angle_sine_sin_18_degrees_l3101_310129


namespace NUMINAMATH_CALUDE_division_ratio_l3101_310177

theorem division_ratio (dividend quotient divisor remainder : ℕ) : 
  dividend = 5290 →
  remainder = 46 →
  divisor = 10 * quotient →
  dividend = divisor * quotient + remainder →
  (divisor : ℚ) / remainder = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_ratio_l3101_310177


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3101_310191

theorem polynomial_expansion (t : ℝ) : 
  (2*t^2 - 3*t + 2) * (-3*t^2 + t - 5) = -6*t^4 + 11*t^3 - 19*t^2 + 17*t - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3101_310191


namespace NUMINAMATH_CALUDE_multiplication_simplification_l3101_310132

theorem multiplication_simplification : 12 * (1 / 26) * 52 * 4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_simplification_l3101_310132


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3101_310171

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3101_310171


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3101_310170

def A : Set ℝ := {x | x ≥ -4}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x | -4 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3101_310170


namespace NUMINAMATH_CALUDE_quadrilateral_equation_implies_rhombus_l3101_310140

-- Define a quadrilateral
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_d : d > 0

-- Define the condition from the problem
def satisfiesEquation (q : Quadrilateral) : Prop :=
  q.a^4 + q.b^4 + q.c^4 + q.d^4 = 4 * q.a * q.b * q.c * q.d

-- Define a rhombus
def isRhombus (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem statement
theorem quadrilateral_equation_implies_rhombus (q : Quadrilateral) :
  satisfiesEquation q → isRhombus q :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_equation_implies_rhombus_l3101_310140


namespace NUMINAMATH_CALUDE_exam_average_l3101_310145

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (avg_passed : ℕ) (avg_failed : ℕ)
  (h1 : total_boys = 120)
  (h2 : passed_boys = 100)
  (h3 : avg_passed = 39)
  (h4 : avg_failed = 15) :
  (avg_passed * passed_boys + avg_failed * (total_boys - passed_boys)) / total_boys = 35 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l3101_310145


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3101_310190

theorem sin_cos_identity : Real.sin (15 * π / 180) * Real.sin (105 * π / 180) - 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3101_310190


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l3101_310114

theorem rectangular_plot_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 24 →
  length = 2 * width →
  2 * (length + width) = perimeter →
  length * width = 32 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l3101_310114


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l3101_310168

/-- Given a right pyramid with a square base, proves that the side length of the base is 10 meters 
    when the area of one lateral face is 120 square meters and the slant height is 24 meters. -/
theorem pyramid_base_side_length : 
  ∀ (base_side_length slant_height lateral_face_area : ℝ),
  slant_height = 24 →
  lateral_face_area = 120 →
  lateral_face_area = (1/2) * base_side_length * slant_height →
  base_side_length = 10 := by
sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l3101_310168


namespace NUMINAMATH_CALUDE_narcissus_count_is_75_l3101_310124

/-- The number of narcissus flowers in a florist's inventory -/
def narcissus_count : ℕ := 75

/-- The number of chrysanthemums in the florist's inventory -/
def chrysanthemum_count : ℕ := 90

/-- The number of bouquets that can be made -/
def bouquet_count : ℕ := 33

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- Theorem stating that the number of narcissus flowers is 75 -/
theorem narcissus_count_is_75 : 
  narcissus_count = bouquet_count * flowers_per_bouquet - chrysanthemum_count :=
by
  sorry

#eval narcissus_count -- Should output 75

end NUMINAMATH_CALUDE_narcissus_count_is_75_l3101_310124


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3101_310113

theorem remainder_divisibility (N : ℤ) : N % 13 = 5 → N % 39 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3101_310113


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l3101_310103

-- Define the total number of boys
def total_boys : ℕ := 700

-- Define the percentage of Muslim boys
def muslim_percentage : ℚ := 44 / 100

-- Define the percentage of Sikh boys
def sikh_percentage : ℚ := 10 / 100

-- Define the number of boys from other communities
def other_boys : ℕ := 126

-- Define the percentage of Hindu boys
def hindu_percentage : ℚ := 28 / 100

-- Theorem statement
theorem percentage_of_hindu_boys :
  hindu_percentage * total_boys =
    total_boys - (muslim_percentage * total_boys + sikh_percentage * total_boys + other_boys) :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l3101_310103


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l3101_310198

/-- A dodecahedron is a 3-dimensional figure with specific properties -/
structure Dodecahedron where
  faces : ℕ
  vertices : ℕ
  faces_per_vertex : ℕ
  faces_are_pentagonal : Prop
  h_faces : faces = 12
  h_vertices : vertices = 20
  h_faces_per_vertex : faces_per_vertex = 3

/-- The number of interior diagonals in a dodecahedron -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - 1 - d.faces_per_vertex)) / 2

/-- Theorem stating the number of interior diagonals in a dodecahedron -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l3101_310198


namespace NUMINAMATH_CALUDE_dip_amount_is_twenty_l3101_310106

/-- Represents the amount of dip that can be made given a budget and artichoke-to-dip ratio --/
def dip_amount (budget : ℚ) (artichokes_per_batch : ℕ) (ounces_per_batch : ℚ) (total_ounces : ℚ) : ℚ :=
  let price_per_artichoke : ℚ := budget / (total_ounces / ounces_per_batch * artichokes_per_batch)
  let artichokes_bought : ℚ := budget / price_per_artichoke
  (artichokes_bought / artichokes_per_batch) * ounces_per_batch

/-- Theorem stating that under given conditions, 20 ounces of dip can be made --/
theorem dip_amount_is_twenty :
  dip_amount 15 3 5 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dip_amount_is_twenty_l3101_310106


namespace NUMINAMATH_CALUDE_perry_phil_difference_l3101_310116

/-- The number of games won by each player -/
structure GolfWins where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- The conditions of the golf game -/
def golf_game (g : GolfWins) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie = g.dana - 2 ∧
  g.phil = g.charlie + 3 ∧
  g.phil = 12

theorem perry_phil_difference (g : GolfWins) (h : golf_game g) : 
  g.perry - g.phil = 4 := by
  sorry

#check perry_phil_difference

end NUMINAMATH_CALUDE_perry_phil_difference_l3101_310116


namespace NUMINAMATH_CALUDE_square_minus_five_equals_two_l3101_310189

theorem square_minus_five_equals_two : ∃ x : ℤ, x - 5 = 2 := by
  use 7
  sorry

end NUMINAMATH_CALUDE_square_minus_five_equals_two_l3101_310189


namespace NUMINAMATH_CALUDE_no_positive_integer_perfect_square_l3101_310167

theorem no_positive_integer_perfect_square : 
  ∀ n : ℕ+, ¬∃ y : ℤ, (n : ℤ)^2 - 21*(n : ℤ) + 110 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_perfect_square_l3101_310167


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3101_310162

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3101_310162


namespace NUMINAMATH_CALUDE_sixteen_divisors_problem_l3101_310147

theorem sixteen_divisors_problem (n : ℕ+) : 
  (∃ (d : Fin 16 → ℕ+), 
    (∀ i : Fin 16, d i ∣ n) ∧ 
    (∀ i j : Fin 16, i < j → d i < d j) ∧
    (d 0 = 1) ∧ 
    (d 15 = n) ∧ 
    (d 5 = 18) ∧ 
    (d 8 - d 7 = 17) ∧
    (∀ m : ℕ+, m ∣ n → ∃ i : Fin 16, d i = m)) →
  n = 1998 ∨ n = 3834 := by
sorry

end NUMINAMATH_CALUDE_sixteen_divisors_problem_l3101_310147


namespace NUMINAMATH_CALUDE_exponent_simplification_l3101_310111

theorem exponent_simplification : 8^6 * 27^6 * 8^27 * 27^8 = 216^14 * 8^19 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l3101_310111


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3101_310150

theorem quadratic_rewrite (b : ℝ) : 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 72 = (x + n)^2 + 20) → 
  (b = 4 * Real.sqrt 13 ∨ b = -4 * Real.sqrt 13) := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3101_310150


namespace NUMINAMATH_CALUDE_least_likely_score_l3101_310119

def class_size : ℕ := 50
def average_score : ℝ := 82
def score_variance : ℝ := 8.2

def score_options : List ℝ := [60, 70, 80, 100]

def distance_from_mean (score : ℝ) : ℝ :=
  |score - average_score|

theorem least_likely_score :
  ∃ (score : ℝ), score ∈ score_options ∧
    ∀ (other : ℝ), other ∈ score_options → other ≠ score →
      distance_from_mean score > distance_from_mean other :=
by sorry

end NUMINAMATH_CALUDE_least_likely_score_l3101_310119


namespace NUMINAMATH_CALUDE_blue_square_area_ratio_l3101_310174

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  cross_area_ratio : ℝ
  symmetric : Bool

/-- The area of the flag -/
def flag_area (flag : CrossFlag) : ℝ := flag.side ^ 2

/-- The area of the cross -/
def cross_area (flag : CrossFlag) : ℝ := flag.cross_area_ratio * flag_area flag

/-- The theorem stating the relationship between the blue square area and the flag area -/
theorem blue_square_area_ratio (flag : CrossFlag) 
  (h1 : flag.cross_area_ratio = 0.36)
  (h2 : flag.symmetric = true) : 
  (flag.side * 0.2) ^ 2 / flag_area flag = 0.04 := by
  sorry

#check blue_square_area_ratio

end NUMINAMATH_CALUDE_blue_square_area_ratio_l3101_310174


namespace NUMINAMATH_CALUDE_vector_properties_l3101_310130

-- Define plane vectors a and b
variable (a b : ℝ × ℝ)

-- Define the conditions
def condition1 : Prop := norm a = 1
def condition2 : Prop := norm b = 1
def condition3 : Prop := norm (2 • a + b) = Real.sqrt 6

-- Define the theorem
theorem vector_properties (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) :
  (a • b = 1/4) ∧ (norm (a + b) = Real.sqrt 10 / 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l3101_310130


namespace NUMINAMATH_CALUDE_trains_meeting_point_l3101_310104

/-- Proves that two trains traveling towards each other on a 200 km track,
    with train A moving at 60 km/h and train B moving at 90 km/h,
    will meet at a distance of 80 km from train A's starting point. -/
theorem trains_meeting_point (distance : ℝ) (speed_A : ℝ) (speed_B : ℝ)
  (h1 : distance = 200)
  (h2 : speed_A = 60)
  (h3 : speed_B = 90) :
  speed_A * (distance / (speed_A + speed_B)) = 80 :=
by sorry

end NUMINAMATH_CALUDE_trains_meeting_point_l3101_310104


namespace NUMINAMATH_CALUDE_total_amount_calculation_l3101_310197

theorem total_amount_calculation (two_won_bills : ℕ) (one_won_bills : ℕ) : 
  two_won_bills = 8 → one_won_bills = 2 → two_won_bills * 2 + one_won_bills * 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l3101_310197


namespace NUMINAMATH_CALUDE_twice_a_minus_four_nonnegative_l3101_310185

theorem twice_a_minus_four_nonnegative (a : ℝ) :
  (2 * a - 4 ≥ 0) ↔ (∃ (x : ℝ), x ≥ 0 ∧ x = 2 * a - 4) :=
by sorry

end NUMINAMATH_CALUDE_twice_a_minus_four_nonnegative_l3101_310185


namespace NUMINAMATH_CALUDE_olivias_initial_money_l3101_310120

/-- Calculates the initial amount of money Olivia had given the number of card packs, their prices, and the change received. -/
def initialMoney (basketballPacks : ℕ) (basketballPrice : ℕ) (baseballDecks : ℕ) (baseballPrice : ℕ) (change : ℕ) : ℕ :=
  basketballPacks * basketballPrice + baseballDecks * baseballPrice + change

/-- Proves that Olivia's initial amount of money was $50 given the problem conditions. -/
theorem olivias_initial_money :
  initialMoney 2 3 5 4 24 = 50 := by
  sorry

end NUMINAMATH_CALUDE_olivias_initial_money_l3101_310120


namespace NUMINAMATH_CALUDE_sqrt_real_range_l3101_310173

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 4 + 2 * x) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l3101_310173


namespace NUMINAMATH_CALUDE_client_cost_is_3400_l3101_310139

def ladders_cost (num_ladders1 : ℕ) (rungs_per_ladder1 : ℕ) 
                 (num_ladders2 : ℕ) (rungs_per_ladder2 : ℕ) 
                 (cost_per_rung : ℕ) : ℕ :=
  (num_ladders1 * rungs_per_ladder1 + num_ladders2 * rungs_per_ladder2) * cost_per_rung

theorem client_cost_is_3400 :
  ladders_cost 10 50 20 60 2 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_client_cost_is_3400_l3101_310139


namespace NUMINAMATH_CALUDE_parabola_max_area_l3101_310163

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x + 1

-- Define the condition for p
def p_condition (p : ℝ) : Prop := p > 0

-- Define the points A and B on the parabola
def point_on_parabola (p x y : ℝ) : Prop := parabola p x y

-- Define the condition for x₁ and x₂
def x_condition (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂ ∧ x₁ + x₂ = 4

-- Define the theorem
theorem parabola_max_area (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  p_condition p →
  parabola p x₁ y₁ →
  parabola p x₂ y₂ →
  x_condition x₁ x₂ →
  (∃ (x y : ℝ), parabola p x y ∧ tangent_line x y) →
  (∃ (area : ℝ), area ≤ 8 ∧ 
    (∀ (other_area : ℝ), other_area ≤ area)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_area_l3101_310163


namespace NUMINAMATH_CALUDE_negative_a_fifth_squared_l3101_310182

theorem negative_a_fifth_squared (a : ℝ) : (-a^5)^2 = a^10 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_fifth_squared_l3101_310182


namespace NUMINAMATH_CALUDE_chicken_feed_requirement_l3101_310180

/-- Represents the problem of calculating chicken feed requirements --/
theorem chicken_feed_requirement 
  (chicken_price : ℝ) 
  (feed_price : ℝ) 
  (feed_weight : ℝ) 
  (num_chickens : ℕ) 
  (profit : ℝ) 
  (h1 : chicken_price = 1.5)
  (h2 : feed_price = 2)
  (h3 : feed_weight = 20)
  (h4 : num_chickens = 50)
  (h5 : profit = 65) :
  (num_chickens * chicken_price - profit) / feed_price * feed_weight / num_chickens = 2 := by
  sorry

#check chicken_feed_requirement

end NUMINAMATH_CALUDE_chicken_feed_requirement_l3101_310180
