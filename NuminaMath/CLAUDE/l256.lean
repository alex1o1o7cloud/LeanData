import Mathlib

namespace NUMINAMATH_CALUDE_dumbbell_system_weight_l256_25623

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

end NUMINAMATH_CALUDE_dumbbell_system_weight_l256_25623


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l256_25603

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

end NUMINAMATH_CALUDE_round_trip_average_speed_l256_25603


namespace NUMINAMATH_CALUDE_chef_potato_problem_l256_25693

/-- The number of potatoes already cooked -/
def potatoes_cooked : ℕ := 7

/-- The time it takes to cook one potato (in minutes) -/
def cooking_time_per_potato : ℕ := 5

/-- The time it takes to cook the remaining potatoes (in minutes) -/
def remaining_cooking_time : ℕ := 45

/-- The total number of potatoes the chef needs to cook -/
def total_potatoes : ℕ := 16

theorem chef_potato_problem :
  total_potatoes = potatoes_cooked + remaining_cooking_time / cooking_time_per_potato :=
by sorry

end NUMINAMATH_CALUDE_chef_potato_problem_l256_25693


namespace NUMINAMATH_CALUDE_lcm_of_10_14_20_l256_25629

theorem lcm_of_10_14_20 : Nat.lcm (Nat.lcm 10 14) 20 = 140 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_10_14_20_l256_25629


namespace NUMINAMATH_CALUDE_computation_problems_count_l256_25617

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

end NUMINAMATH_CALUDE_computation_problems_count_l256_25617


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l256_25647

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l256_25647


namespace NUMINAMATH_CALUDE_arrangements_eq_36_l256_25609

/-- The number of students in the row -/
def n : ℕ := 5

/-- A function that calculates the number of arrangements given the conditions -/
def arrangements (n : ℕ) : ℕ :=
  let positions := n - 1  -- Possible positions for A (excluding ends)
  let pairs := 2  -- A and B can be arranged in 2 ways next to each other
  let others := n - 2  -- Remaining students to arrange
  positions * pairs * (others.factorial)

/-- The theorem stating that the number of arrangements is 36 -/
theorem arrangements_eq_36 : arrangements n = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_eq_36_l256_25609


namespace NUMINAMATH_CALUDE_largest_decimal_l256_25618

theorem largest_decimal : 
  let a := 0.987
  let b := 0.9861
  let c := 0.98709
  let d := 0.968
  let e := 0.96989
  (c ≥ a) ∧ (c ≥ b) ∧ (c ≥ d) ∧ (c ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l256_25618


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_three_l256_25656

theorem smallest_digit_divisible_by_three : 
  ∃ (x : ℕ), x < 10 ∧ 
  (526000 + x * 100 + 18) % 3 = 0 ∧
  ∀ (y : ℕ), y < x → y < 10 → (526000 + y * 100 + 18) % 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_three_l256_25656


namespace NUMINAMATH_CALUDE_cleaning_time_with_doubled_speed_l256_25604

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

end NUMINAMATH_CALUDE_cleaning_time_with_doubled_speed_l256_25604


namespace NUMINAMATH_CALUDE_carlos_initial_blocks_l256_25637

/-- The number of blocks Carlos gave to Rachel -/
def blocks_given : ℕ := 21

/-- The number of blocks Carlos had left -/
def blocks_left : ℕ := 37

/-- The initial number of blocks Carlos had -/
def initial_blocks : ℕ := blocks_given + blocks_left

theorem carlos_initial_blocks : initial_blocks = 58 := by sorry

end NUMINAMATH_CALUDE_carlos_initial_blocks_l256_25637


namespace NUMINAMATH_CALUDE_cosine_arithmetic_sequence_product_l256_25659

theorem cosine_arithmetic_sequence_product (a : ℕ → ℝ) (S : Set ℝ) (a₀ b₀ : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + 2 * π / 3) →
  S = {x | ∃ n : ℕ, x = Real.cos (a n)} →
  S = {a₀, b₀} →
  a₀ * b₀ = -1/2 :=
sorry

end NUMINAMATH_CALUDE_cosine_arithmetic_sequence_product_l256_25659


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_a_range_l256_25645

/-- The function f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ if and only if a ∈ [-√3, √3] -/
theorem monotonic_cubic_function_a_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_a_range_l256_25645


namespace NUMINAMATH_CALUDE_quadratic_minimum_l256_25651

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l256_25651


namespace NUMINAMATH_CALUDE_unique_B_for_divisibility_l256_25661

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def number_4BBB2 (B : ℕ) : ℕ := 40000 + 1000 * B + 100 * B + 10 * B + 2

theorem unique_B_for_divisibility :
  ∃! B : ℕ, digit B ∧ is_divisible_by_9 (number_4BBB2 B) ∧ B = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_B_for_divisibility_l256_25661


namespace NUMINAMATH_CALUDE_odd_square_minus_one_multiple_of_24_and_101_case_l256_25602

theorem odd_square_minus_one_multiple_of_24_and_101_case : 
  (∀ n : ℕ, n > 1 → (2*n + 1)^2 - 1 = 24 * (n * (n + 1) / 2)) ∧ 
  (101^2 - 1 = 10200) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_multiple_of_24_and_101_case_l256_25602


namespace NUMINAMATH_CALUDE_cake_recipe_salt_l256_25677

theorem cake_recipe_salt (sugar_total : ℕ) (salt : ℕ) : 
  sugar_total = 8 → 
  sugar_total = salt + 1 → 
  salt = 7 := by
sorry

end NUMINAMATH_CALUDE_cake_recipe_salt_l256_25677


namespace NUMINAMATH_CALUDE_inequality_proof_l256_25685

theorem inequality_proof (a b c d : ℝ) 
  (h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d)
  (h_product : (a-b)*(b-c)*(c-d)*(d-a) = -3) :
  (a + b + c + d = 6 → d < 0.36) ∧
  (a^2 + b^2 + c^2 + d^2 = 14 → (a+c)*(b+d) ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l256_25685


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l256_25658

/-- An isosceles triangle with side lengths a, b, and c, where two sides are 11 and one side is 5 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_isosceles : (a = b ∧ a = 11 ∧ c = 5) ∨ (a = c ∧ a = 11 ∧ b = 5) ∨ (b = c ∧ b = 11 ∧ a = 5)
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of an isosceles triangle with two sides of length 11 and one side of length 5 is 27 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : t.a + t.b + t.c = 27 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l256_25658


namespace NUMINAMATH_CALUDE_inequality_proof_l256_25678

def M : Set ℝ := {x : ℝ | |x + 1| + |x - 1| ≤ 2}

theorem inequality_proof (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ 1/6) (hz : |z| ≤ 1/9) :
  |x + 2*y - 3*z| ≤ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l256_25678


namespace NUMINAMATH_CALUDE_inequality_solution_set_range_of_m_l256_25675

-- Define the function f(x) = |x+1|
def f (x : ℝ) : ℝ := |x + 1|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 2 * x + 1} = {x : ℝ | x ≤ 0} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, f (x - 2) - f (x + 6) < m) ↔ m > -8 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_range_of_m_l256_25675


namespace NUMINAMATH_CALUDE_both_parents_single_eyelids_sufficient_not_necessary_l256_25631

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

end NUMINAMATH_CALUDE_both_parents_single_eyelids_sufficient_not_necessary_l256_25631


namespace NUMINAMATH_CALUDE_student_arrangements_eq_20_l256_25638

/-- The number of ways to arrange 7 students of different heights in a row,
    with the tallest in the middle and the others decreasing in height towards both ends. -/
def student_arrangements : ℕ :=
  Nat.choose 6 3

/-- Theorem stating that the number of student arrangements is 20. -/
theorem student_arrangements_eq_20 : student_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangements_eq_20_l256_25638


namespace NUMINAMATH_CALUDE_blue_candles_l256_25650

/-- The number of blue candles on a birthday cake -/
theorem blue_candles (total : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ)
  (h1 : total = 79)
  (h2 : yellow = 27)
  (h3 : red = 14)
  (h4 : blue = total - yellow - red) :
  blue = 38 := by
  sorry

end NUMINAMATH_CALUDE_blue_candles_l256_25650


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l256_25615

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

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l256_25615


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l256_25696

def f (x : ℝ) := x^3 - 15*x^2 - 33*x + 6

theorem monotonic_decreasing_interval :
  {x : ℝ | ∀ y, x < y → f x > f y} = {x | -1 < x ∧ x < 11} := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l256_25696


namespace NUMINAMATH_CALUDE_tangent_line_sum_l256_25684

/-- Given a function f: ℝ → ℝ with a tangent line 2x - y - 3 = 0 at x = 2,
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f 2 → 2*x - y - 3 = 0 ↔ y = f x) : 
    f 2 + deriv f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l256_25684


namespace NUMINAMATH_CALUDE_f_range_contains_interval_f_range_may_extend_l256_25666

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 + 2*Real.pi * Real.arcsin (x/3) - 3*(Real.arcsin (x/3))^2 + 
  (Real.pi^2/4)*(x^3 - x^2 + 4*x - 8)

theorem f_range_contains_interval :
  ∀ y ∈ Set.Icc (Real.pi^2/4) ((9*Real.pi^2)/4),
  ∃ x ∈ Set.Icc (-3) 3, f x = y :=
by sorry

theorem f_range_may_extend :
  ∃ y, (y < Real.pi^2/4 ∨ y > (9*Real.pi^2)/4) ∧
  ∃ x ∈ Set.Icc (-3) 3, f x = y :=
by sorry

end NUMINAMATH_CALUDE_f_range_contains_interval_f_range_may_extend_l256_25666


namespace NUMINAMATH_CALUDE_tan_double_angle_l256_25691

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l256_25691


namespace NUMINAMATH_CALUDE_mary_warmth_hours_l256_25674

/-- The number of sticks of wood produced by chopping up furniture and the number of hours Mary can keep warm. -/
def furniture_to_warmth (chair_sticks table_sticks cabinet_sticks stool_sticks : ℕ)
  (chairs tables cabinets stools : ℕ) (sticks_per_hour : ℕ) : ℕ :=
  let total_sticks := chair_sticks * chairs + table_sticks * tables + 
                      cabinet_sticks * cabinets + stool_sticks * stools
  total_sticks / sticks_per_hour

/-- Theorem stating that Mary can keep warm for 64 hours given the specified conditions. -/
theorem mary_warmth_hours : 
  furniture_to_warmth 8 12 16 3 25 12 5 8 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_mary_warmth_hours_l256_25674


namespace NUMINAMATH_CALUDE_plum_picking_l256_25644

/-- The number of plums picked by Melanie -/
def melanie_plums : ℕ := 4

/-- The number of plums picked by Dan -/
def dan_plums : ℕ := 9

/-- The number of plums picked by Sally -/
def sally_plums : ℕ := 3

/-- The total number of plums picked -/
def total_plums : ℕ := melanie_plums + dan_plums + sally_plums

theorem plum_picking :
  total_plums = 16 := by sorry

end NUMINAMATH_CALUDE_plum_picking_l256_25644


namespace NUMINAMATH_CALUDE_first_divisible_correct_l256_25605

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

end NUMINAMATH_CALUDE_first_divisible_correct_l256_25605


namespace NUMINAMATH_CALUDE_modulus_of_z_l256_25681

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l256_25681


namespace NUMINAMATH_CALUDE_rotation_direction_undetermined_l256_25642

-- Define a type for rotation direction
inductive RotationDirection
| Clockwise
| Counterclockwise

-- Define a type for a quadrilateral
structure Quadrilateral where
  -- We don't need to specify the exact structure of a quadrilateral for this problem
  mk :: 

-- Define a point Z
def Z : ℝ × ℝ := sorry

-- Define the rotation transformation
def rotate (q : Quadrilateral) (center : ℝ × ℝ) (angle : ℝ) : Quadrilateral := sorry

-- State the theorem
theorem rotation_direction_undetermined 
  (q1 q2 : Quadrilateral) 
  (h1 : rotate q1 Z (270 : ℝ) = q2) : 
  ¬ ∃ (d : RotationDirection), d = RotationDirection.Clockwise ∨ d = RotationDirection.Counterclockwise := 
sorry

end NUMINAMATH_CALUDE_rotation_direction_undetermined_l256_25642


namespace NUMINAMATH_CALUDE_triangle_inequalities_l256_25641

/-- Theorem about triangle inequalities involving area, side lengths, altitudes, and excircle radii -/
theorem triangle_inequalities (a b c : ℝ) (S : ℝ) (h_a h_b h_c : ℝ) (r_a r_b r_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_area : S > 0)
  (h_altitudes : h_a > 0 ∧ h_b > 0 ∧ c > 0)
  (h_radii : r_a > 0 ∧ r_b > 0 ∧ r_c > 0) :
  S^3 ≤ (Real.sqrt 3 / 4)^3 * (a * b * c)^2 ∧ 
  3 * h_a * h_b * h_c ≤ 3 * Real.sqrt 3 * S ∧
  3 * Real.sqrt 3 * S ≤ 3 * r_a * r_b * r_c := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequalities_l256_25641


namespace NUMINAMATH_CALUDE_sin_plus_cos_special_angle_l256_25697

/-- Given a point P(-3,4) on the terminal side of angle α, prove that sin α + cos α = 1/5 -/
theorem sin_plus_cos_special_angle (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  (∃ t : ℝ, t > 0 ∧ P.1 = t * Real.cos α ∧ P.2 = t * Real.sin α) →
  Real.sin α + Real.cos α = 1/5 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_special_angle_l256_25697


namespace NUMINAMATH_CALUDE_ticket_price_ratio_l256_25670

/-- Proves that the ratio of adult to child ticket prices is 2:1 given the problem conditions --/
theorem ticket_price_ratio :
  ∀ (adult_price child_price : ℚ),
    adult_price = 32 →
    400 * adult_price + 200 * child_price = 16000 →
    adult_price / child_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_ratio_l256_25670


namespace NUMINAMATH_CALUDE_hundreds_digit_of_factorial_difference_l256_25671

theorem hundreds_digit_of_factorial_difference : (25 - 20).factorial ≡ 0 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_factorial_difference_l256_25671


namespace NUMINAMATH_CALUDE_tunnel_length_proof_l256_25640

/-- The length of a train in miles -/
def train_length : ℝ := 1.5

/-- The time difference in minutes between the front of the train entering the tunnel and the tail exiting -/
def time_difference : ℝ := 4

/-- The speed of the train in miles per hour -/
def train_speed : ℝ := 45

/-- The length of the tunnel in miles -/
def tunnel_length : ℝ := 1.5

theorem tunnel_length_proof :
  tunnel_length = train_speed * (time_difference / 60) - train_length :=
by sorry

end NUMINAMATH_CALUDE_tunnel_length_proof_l256_25640


namespace NUMINAMATH_CALUDE_carlas_chickens_l256_25628

theorem carlas_chickens (initial_chickens : ℕ) : 
  (initial_chickens : ℝ) - 0.4 * initial_chickens + 10 * (0.4 * initial_chickens) = 1840 →
  initial_chickens = 400 := by
  sorry

end NUMINAMATH_CALUDE_carlas_chickens_l256_25628


namespace NUMINAMATH_CALUDE_dice_rolling_expectation_l256_25672

/-- The expected value of 6^D after n steps in the dice rolling process -/
def expected_value (n : ℕ) : ℝ :=
  6 + 5 * n

/-- The number of steps in the process -/
def num_steps : ℕ := 2013

theorem dice_rolling_expectation :
  expected_value num_steps = 10071 := by
  sorry

end NUMINAMATH_CALUDE_dice_rolling_expectation_l256_25672


namespace NUMINAMATH_CALUDE_diamond_calculation_l256_25673

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation : 
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l256_25673


namespace NUMINAMATH_CALUDE_ice_cream_cost_l256_25694

theorem ice_cream_cost (pierre_scoops mom_scoops : ℕ) (total_bill : ℚ) 
  (h1 : pierre_scoops = 3)
  (h2 : mom_scoops = 4)
  (h3 : total_bill = 14) :
  ∃ (scoop_cost : ℚ), scoop_cost * (pierre_scoops + mom_scoops : ℚ) = total_bill ∧ scoop_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l256_25694


namespace NUMINAMATH_CALUDE_football_shoes_cost_l256_25621

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

end NUMINAMATH_CALUDE_football_shoes_cost_l256_25621


namespace NUMINAMATH_CALUDE_problem_statement_l256_25616

theorem problem_statement :
  (∀ (x : ℕ), x > 0 → (1/2 : ℝ)^x ≥ (1/3 : ℝ)^x) ∧
  ¬(∃ (x : ℕ), x > 0 ∧ (2 : ℝ)^x + (2 : ℝ)^(1-x) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l256_25616


namespace NUMINAMATH_CALUDE_line_l_equation_no_symmetric_points_l256_25676

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2*x + y + 1 = 0
def l₂ (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define line l
def l (x y : ℝ) : Prop := x + y = 0

-- Define the parabola
def parabola (a x y : ℝ) : Prop := y = a*x^2 - 1

-- Theorem 1: Prove that l is the correct line given the midpoint condition
theorem line_l_equation : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ + x₂)/2 = 0 ∧ (y₁ + y₂)/2 = 0) →
  (∀ x y : ℝ, l x y ↔ x + y = 0) :=
sorry

-- Theorem 2: Prove the condition for non-existence of symmetric points
theorem no_symmetric_points (a : ℝ) :
  (a ≠ 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (parabola a x₁ y₁ ∧ parabola a x₂ y₂ ∧ 
     (x₁ + x₂)/2 + (y₁ + y₂)/2 = 0) → x₁ = x₂ ∧ y₁ = y₂) ↔ 
  (a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_no_symmetric_points_l256_25676


namespace NUMINAMATH_CALUDE_reduced_price_per_dozen_l256_25648

/-- Represents the price reduction percentage -/
def price_reduction : ℝ := 0.40

/-- Represents the additional number of apples that can be bought after the price reduction -/
def additional_apples : ℕ := 64

/-- Represents the fixed amount of money spent on apples -/
def fixed_amount : ℝ := 40

/-- Represents the number of apples in a dozen -/
def apples_per_dozen : ℕ := 12

/-- Theorem stating that given the conditions, the reduced price per dozen apples is Rs. 3 -/
theorem reduced_price_per_dozen (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := original_price * (1 - price_reduction)
  let original_quantity := fixed_amount / original_price
  let new_quantity := original_quantity + additional_apples
  (new_quantity : ℝ) * reduced_price = fixed_amount →
  apples_per_dozen * (fixed_amount / new_quantity) = 3 :=
by sorry

end NUMINAMATH_CALUDE_reduced_price_per_dozen_l256_25648


namespace NUMINAMATH_CALUDE_fraction_equality_l256_25699

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + 2 * y) / (x - 5 * y) = 3) : 
  (x + 5 * y) / (5 * x - y) = 7 / 87 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l256_25699


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l256_25652

/-- The ratio of volumes of two cylinders formed from a 6x9 rectangle -/
theorem cylinder_volume_ratio : 
  let rect_width : ℝ := 6
  let rect_height : ℝ := 9
  let cylinder1_height : ℝ := rect_height
  let cylinder1_circumference : ℝ := rect_width
  let cylinder2_height : ℝ := rect_width
  let cylinder2_circumference : ℝ := rect_height
  let cylinder1_volume : ℝ := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let cylinder2_volume : ℝ := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  let max_volume : ℝ := max cylinder1_volume cylinder2_volume
  let min_volume : ℝ := min cylinder1_volume cylinder2_volume
  (max_volume / min_volume) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l256_25652


namespace NUMINAMATH_CALUDE_root_between_roots_l256_25689

theorem root_between_roots (a b : ℝ) (α β : ℝ) 
  (hα : α^2 + a*α + b = 0) 
  (hβ : β^2 - a*β - b = 0) : 
  ∃ x, x ∈ Set.Icc α β ∧ x^2 - 2*a*x - 2*b = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_between_roots_l256_25689


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l256_25607

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

end NUMINAMATH_CALUDE_shaded_cubes_count_l256_25607


namespace NUMINAMATH_CALUDE_k_value_is_four_thirds_l256_25680

/-- The function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The function g(x) = kx^2 - x - (k+1) -/
def g (k : ℝ) (x : ℝ) : ℝ := k * x^2 - x - (k + 1)

/-- The theorem stating that k = 4/3 given the conditions -/
theorem k_value_is_four_thirds (k : ℝ) (h1 : k > 1) :
  (∀ x₁ ∈ Set.Icc 2 4, ∃ x₂ ∈ Set.Icc 2 4, f x₁ / g k x₁ = g k x₂ / f x₂) →
  k = 4/3 := by sorry

end NUMINAMATH_CALUDE_k_value_is_four_thirds_l256_25680


namespace NUMINAMATH_CALUDE_zero_to_positive_power_l256_25643

theorem zero_to_positive_power (n : ℕ+) : 0 ^ (n : ℕ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_to_positive_power_l256_25643


namespace NUMINAMATH_CALUDE_root_equation_problem_l256_25665

/-- Given two constants p and q, if the specified equations have the given number of distinct roots
    and q = 8, then 50p - 10q = 20 -/
theorem root_equation_problem (p q : ℝ) : 
  (∃! x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + p) * (x + q) * (x - 8) / (x + 4)^2 = 0) →
  (∃! x y, x ≠ y ∧ 
    (x + 4*p) * (x - 4) * (x - 10) / ((x + q) * (x - 8)) = 0) →
  q = 8 →
  50 * p - 10 * q = 20 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l256_25665


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l256_25632

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x - 6 > 0) ↔ (∃ x : ℝ, x^2 + 2*x - 6 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l256_25632


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l256_25601

theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (π / 6) (π / 3), 
    Monotone (fun x => (a - Real.sin x) / Real.cos x)) →
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l256_25601


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l256_25682

theorem solve_exponential_equation :
  ∃ n : ℕ, 4^n * 4^n * 4^n = 16^3 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l256_25682


namespace NUMINAMATH_CALUDE_function_properties_l256_25610

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →  -- g is an odd function
  (∃ C, ∀ x, f a b x = -1/3 * x^3 + x^2 + C) ∧ 
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≤ 4 * Real.sqrt 2 / 3) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≥ 4 / 3) ∧
  (g (-1/3) 0 (Real.sqrt 2) = 4 * Real.sqrt 2 / 3) ∧
  (g (-1/3) 0 2 = 4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l256_25610


namespace NUMINAMATH_CALUDE_teacher_health_survey_l256_25669

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : high_bp = 80)
  (h3 : heart_trouble = 50)
  (h4 : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 100/3 :=
sorry

end NUMINAMATH_CALUDE_teacher_health_survey_l256_25669


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l256_25679

theorem square_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0) (h' : s₂ > 0) :
  s₂ * Real.sqrt 2 = 1.5 * (s₁ * Real.sqrt 2) →
  (4 * s₂) / (4 * s₁) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l256_25679


namespace NUMINAMATH_CALUDE_cut_rectangle_corners_l256_25649

/-- A shape created by cutting off one corner of a rectangle --/
structure CutRectangle where
  originalCorners : Nat
  cutCorners : Nat
  newCorners : Nat

/-- Properties of a rectangle with one corner cut off --/
def isValidCutRectangle (r : CutRectangle) : Prop :=
  r.originalCorners = 4 ∧
  r.cutCorners = 1 ∧
  r.newCorners = r.originalCorners + r.cutCorners

/-- Theorem: A rectangle with one corner cut off has 5 corners --/
theorem cut_rectangle_corners (r : CutRectangle) (h : isValidCutRectangle r) :
  r.newCorners = 5 := by
  sorry

#check cut_rectangle_corners

end NUMINAMATH_CALUDE_cut_rectangle_corners_l256_25649


namespace NUMINAMATH_CALUDE_triangle_side_length_l256_25635

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l256_25635


namespace NUMINAMATH_CALUDE_B_power_101_l256_25611

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 0]]

theorem B_power_101 : B ^ 101 = B := by sorry

end NUMINAMATH_CALUDE_B_power_101_l256_25611


namespace NUMINAMATH_CALUDE_equation_solution_l256_25612

theorem equation_solution : 
  ∃! x : ℚ, (x - 27) / 3 = (3 * x + 6) / 8 ∧ x = -234 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l256_25612


namespace NUMINAMATH_CALUDE_circles_intersect_l256_25622

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

end NUMINAMATH_CALUDE_circles_intersect_l256_25622


namespace NUMINAMATH_CALUDE_set_A_range_l256_25687

theorem set_A_range (a : ℝ) : 
  let A := {x : ℝ | a * x^2 - 3 * x - 4 = 0}
  (∀ x y : ℝ, x ∈ A → y ∈ A → x = y) → 
  (a ≤ -9/16 ∨ a = 0) := by
sorry

end NUMINAMATH_CALUDE_set_A_range_l256_25687


namespace NUMINAMATH_CALUDE_laborer_wage_calculation_l256_25613

/-- Proves that the daily wage for a laborer is 2.00 rupees given the problem conditions --/
theorem laborer_wage_calculation (total_days : ℕ) (absent_days : ℕ) (fine_per_day : ℚ) (total_received : ℚ) :
  total_days = 25 →
  absent_days = 5 →
  fine_per_day = 1/2 →
  total_received = 75/2 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - absent_days : ℚ) - (fine_per_day * absent_days) = total_received ∧
    daily_wage = 2 := by
  sorry

#eval (2 : ℚ)

end NUMINAMATH_CALUDE_laborer_wage_calculation_l256_25613


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l256_25683

theorem sqrt_equation_solution : 
  ∀ x : ℝ, (Real.sqrt (5 * x - 4) + 12 / Real.sqrt (5 * x - 4) = 9) ↔ (x = 13/5 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l256_25683


namespace NUMINAMATH_CALUDE_soap_brands_survey_l256_25646

theorem soap_brands_survey (total : Nat) (neither : Nat) (only_a : Nat) (both : Nat) : 
  total = 300 →
  neither = 80 →
  only_a = 60 →
  total = neither + only_a + 3 * both + both →
  both = 40 := by
sorry

end NUMINAMATH_CALUDE_soap_brands_survey_l256_25646


namespace NUMINAMATH_CALUDE_solve_bus_problem_l256_25634

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

end NUMINAMATH_CALUDE_solve_bus_problem_l256_25634


namespace NUMINAMATH_CALUDE_largest_c_value_l256_25633

theorem largest_c_value (c d e : ℤ) 
  (eq : 5 * c + (d - 12)^2 + e^3 = 235)
  (c_lt_d : c < d) : 
  c ≤ 22 ∧ ∃ (c' d' e' : ℤ), c' = 22 ∧ c' < d' ∧ 5 * c' + (d' - 12)^2 + e'^3 = 235 :=
sorry

end NUMINAMATH_CALUDE_largest_c_value_l256_25633


namespace NUMINAMATH_CALUDE_polygon_diagonals_l256_25654

/-- 
For an n-sided polygon, if 6 diagonals can be drawn from a single vertex, then n = 9.
-/
theorem polygon_diagonals (n : ℕ) : (n - 3 = 6) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l256_25654


namespace NUMINAMATH_CALUDE_stating_min_messages_proof_l256_25660

/-- Represents the minimum number of messages needed for information distribution -/
def min_messages (n : ℕ) : ℕ := 2 * (n - 1)

/-- 
Theorem stating that the minimum number of messages needed for n people 
to share all information is 2(n-1)
-/
theorem min_messages_proof (n : ℕ) (h : n > 0) : 
  ∀ (f : ℕ → ℕ), 
  (∀ i : ℕ, i < n → f i ≥ min_messages n) → 
  (∃ g : ℕ → ℕ → Bool, 
    (∀ i j : ℕ, i < n ∧ j < n → g i j = true) ∧ 
    (∀ i : ℕ, i < n → ∃ k : ℕ, k < f i ∧ 
      (∀ j : ℕ, j < n → ∃ m : ℕ, m ≤ k ∧ g i j = true))) :=
sorry

#check min_messages_proof

end NUMINAMATH_CALUDE_stating_min_messages_proof_l256_25660


namespace NUMINAMATH_CALUDE_remaining_integers_l256_25620

/-- The number of integers remaining in a set of 1 to 80 after removing multiples of 4 and 5 -/
theorem remaining_integers (n : ℕ) (hn : n = 80) : 
  n - (n / 4 + n / 5 - n / 20) = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_integers_l256_25620


namespace NUMINAMATH_CALUDE_infinitely_many_composite_generating_numbers_l256_25692

theorem infinitely_many_composite_generating_numbers :
  ∃ f : ℕ → ℕ, Infinite {k | ∀ n : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n^4 + f k = x * y} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_generating_numbers_l256_25692


namespace NUMINAMATH_CALUDE_fourth_power_sum_l256_25688

theorem fourth_power_sum (a : ℝ) (h : a^2 - 3*a + 1 = 0) : a^4 + 1/a^4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l256_25688


namespace NUMINAMATH_CALUDE_basketball_score_proof_l256_25606

theorem basketball_score_proof (total_score : ℕ) (two_point_shots : ℕ) (three_point_shots : ℕ) :
  total_score = 16 ∧
  two_point_shots = three_point_shots + 3 ∧
  2 * two_point_shots + 3 * three_point_shots = total_score →
  three_point_shots = 2 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l256_25606


namespace NUMINAMATH_CALUDE_sum_of_even_and_multiples_of_five_l256_25698

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit multiples of 5 -/
def B : ℕ := 1800

/-- The sum of four-digit even numbers and four-digit multiples of 5 is 6300 -/
theorem sum_of_even_and_multiples_of_five : C + B = 6300 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_and_multiples_of_five_l256_25698


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l256_25625

def polynomial (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9 * x^3 - 6) + 4 * (7 * x^6 - 2 * x^3 + 8)

theorem sum_of_coefficients : 
  polynomial 1 = 62 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l256_25625


namespace NUMINAMATH_CALUDE_flyers_left_to_hand_out_l256_25655

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed : ℕ) 
  (rose_handed : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed = 120)
  (h3 : rose_handed = 320) : 
  total_flyers - (jack_handed + rose_handed) = 796 :=
by sorry

end NUMINAMATH_CALUDE_flyers_left_to_hand_out_l256_25655


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l256_25668

/-- Simple interest calculation for a given principal, rate, and interest amount -/
theorem simple_interest_time_calculation 
  (P : ℝ) (R : ℝ) (SI : ℝ) (h1 : P = 10000) (h2 : R = 5) (h3 : SI = 500) : 
  (SI * 100) / (P * R) * 12 = 12 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l256_25668


namespace NUMINAMATH_CALUDE_poster_distance_is_18cm_l256_25630

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

end NUMINAMATH_CALUDE_poster_distance_is_18cm_l256_25630


namespace NUMINAMATH_CALUDE_largest_x_floor_div_l256_25664

theorem largest_x_floor_div (x : ℝ) : 
  (⌊x⌋ : ℝ) / x = 9 / 10 → x ≤ 80 / 9 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_div_l256_25664


namespace NUMINAMATH_CALUDE_unique_triple_l256_25686

theorem unique_triple : 
  ∃! (A B C : ℕ), A^2 + B - C = 100 ∧ A + B^2 - C = 124 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_triple_l256_25686


namespace NUMINAMATH_CALUDE_city_population_increase_l256_25653

/-- Represents the net population increase in a city over one day -/
def net_population_increase (birth_rate : ℕ) (death_rate : ℕ) (time_interval : ℕ) (seconds_per_day : ℕ) : ℕ :=
  let net_rate_per_interval := birth_rate - death_rate
  let net_rate_per_second := net_rate_per_interval / time_interval
  net_rate_per_second * seconds_per_day

/-- Theorem stating the net population increase in a day given specific birth and death rates -/
theorem city_population_increase : 
  net_population_increase 6 2 2 86400 = 172800 := by
  sorry

#eval net_population_increase 6 2 2 86400

end NUMINAMATH_CALUDE_city_population_increase_l256_25653


namespace NUMINAMATH_CALUDE_evaluate_expression_l256_25626

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 + 2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l256_25626


namespace NUMINAMATH_CALUDE_question_mark_value_l256_25614

theorem question_mark_value : ∃ x : ℚ, (786 * x) / 30 = 1938.8 ∧ x = 74 := by sorry

end NUMINAMATH_CALUDE_question_mark_value_l256_25614


namespace NUMINAMATH_CALUDE_polynomial_problem_l256_25636

/-- Given polynomial P = 2(ax-3) - 3(bx+5) -/
def P (a b x : ℝ) : ℝ := 2*(a*x - 3) - 3*(b*x + 5)

theorem polynomial_problem (a b : ℝ) (h1 : P a b 2 = -31) (h2 : a + b = 0) :
  (a = -1 ∧ b = 1) ∧ 
  (∀ x : ℤ, P a b x > 0 → x ≤ -5) ∧
  (P a b (-5 : ℝ) > 0) :=
sorry

end NUMINAMATH_CALUDE_polynomial_problem_l256_25636


namespace NUMINAMATH_CALUDE_equation_solution_l256_25657

theorem equation_solution :
  ∃ x : ℝ, (5 * x - 8 * (2 * x + 3) = 4 * (x - 3 * (2 * x - 5)) + 7 * (2 * x - 5)) ∧ x = -9.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l256_25657


namespace NUMINAMATH_CALUDE_streamer_profit_formula_l256_25624

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

end NUMINAMATH_CALUDE_streamer_profit_formula_l256_25624


namespace NUMINAMATH_CALUDE_frank_candy_count_l256_25619

theorem frank_candy_count (bags : ℕ) (pieces_per_bag : ℕ) 
  (h1 : bags = 26) (h2 : pieces_per_bag = 33) : 
  bags * pieces_per_bag = 858 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_count_l256_25619


namespace NUMINAMATH_CALUDE_shorter_leg_of_second_triangle_l256_25662

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse_eq : hypotenuse = 2 * shorterLeg
  longerLeg_eq : longerLeg = shorterLeg * Real.sqrt 3

/-- A sequence of two 30-60-90 triangles where the hypotenuse of the first is the longer leg of the second -/
def TwoTriangles (t1 t2 : Triangle30_60_90) :=
  t1.hypotenuse = 12 ∧ t1.longerLeg = t2.hypotenuse

theorem shorter_leg_of_second_triangle (t1 t2 : Triangle30_60_90) 
  (h : TwoTriangles t1 t2) : t2.shorterLeg = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_shorter_leg_of_second_triangle_l256_25662


namespace NUMINAMATH_CALUDE_problem_statement_l256_25627

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → 
  a^2009 + b^2009 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l256_25627


namespace NUMINAMATH_CALUDE_trumpet_players_count_l256_25663

def orchestra_size : ℕ := 21
def drummer_count : ℕ := 1
def trombone_count : ℕ := 4
def french_horn_count : ℕ := 1
def violinist_count : ℕ := 3
def cellist_count : ℕ := 1
def contrabassist_count : ℕ := 1
def clarinet_count : ℕ := 3
def flute_count : ℕ := 4
def maestro_count : ℕ := 1

theorem trumpet_players_count :
  orchestra_size - (drummer_count + trombone_count + french_horn_count + 
    violinist_count + cellist_count + contrabassist_count + 
    clarinet_count + flute_count + maestro_count) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_players_count_l256_25663


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l256_25695

-- Define the complex number z
variable (z : ℂ)

-- Define the condition
def condition (z : ℂ) : Prop := 1 + z = 2 + 3 * Complex.I

-- Theorem statement
theorem imaginary_part_of_z (h : condition z) : z.im = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l256_25695


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l256_25667

theorem quadratic_inequality_solution (a b : ℝ) (h : ∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :
  (a = 1 ∧ b = 2) ∧
  (∀ m : ℝ,
    (m = 2 → ∀ x, ¬(x^2 - (m + 2) * x + 2 * m < 0)) ∧
    (m < 2 → ∀ x, x^2 - (m + 2) * x + 2 * m < 0 ↔ m < x ∧ x < 2) ∧
    (m > 2 → ∀ x, x^2 - (m + 2) * x + 2 * m < 0 ↔ 2 < x ∧ x < m)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l256_25667


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l256_25639

/-- A hyperbola is represented by an equation of the form a*x² + b*y² = c, where a and b have opposite signs and c ≠ 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0
  h3 : c ≠ 0
  h4 : a * b < 0

/-- The equation x²/(9-k) + y²/(k-4) = 1 -/
def equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (9 - k) + y^2 / (k - 4) = 1

/-- The condition k > 9 is sufficient but not necessary for the equation to represent a hyperbola -/
theorem sufficient_not_necessary (k : ℝ) :
  (k > 9 → ∃ h : Hyperbola, equation k = λ x y ↦ h.a * x^2 + h.b * y^2 = h.c) ∧
  ¬(∀ h : Hyperbola, equation k = λ x y ↦ h.a * x^2 + h.b * y^2 = h.c → k > 9) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l256_25639


namespace NUMINAMATH_CALUDE_consecutive_non_primes_l256_25690

theorem consecutive_non_primes (n : ℕ) : 
  ∃ k : ℕ, ∀ i : ℕ, i ∈ Finset.range n → ¬ Prime (k + i + 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_l256_25690


namespace NUMINAMATH_CALUDE_nba_schedule_impossibility_l256_25600

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

end NUMINAMATH_CALUDE_nba_schedule_impossibility_l256_25600


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l256_25608

/-- A quadratic function of the form f(x) = ax² - 3x + a² - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + a^2 - 1

/-- Theorem: If f(0) = 0 and a > 0, then a = 1 -/
theorem quadratic_through_origin (a : ℝ) (h1 : f a 0 = 0) (h2 : a > 0) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l256_25608
