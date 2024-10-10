import Mathlib

namespace car_speed_l3238_323831

/-- Calculates the speed of a car given distance and time -/
theorem car_speed (distance : ℝ) (time : ℝ) (h1 : distance = 624) (h2 : time = 2 + 2/5) :
  distance / time = 260 := by
  sorry

end car_speed_l3238_323831


namespace simplify_sqrt_expression_l3238_323806

theorem simplify_sqrt_expression : 
  (Real.sqrt 448 / Real.sqrt 32) - (Real.sqrt 245 / Real.sqrt 49) = Real.sqrt 2 * Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end simplify_sqrt_expression_l3238_323806


namespace prob_A_and_B_l3238_323838

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.85

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.60

/-- The theorem stating that the probability of both A and B occurring simultaneously
    is equal to the product of their individual probabilities -/
theorem prob_A_and_B : prob_A * prob_B = 0.51 := by sorry

end prob_A_and_B_l3238_323838


namespace sqrt_meaningful_iff_l3238_323878

theorem sqrt_meaningful_iff (x : ℝ) : Real.sqrt (x - 1/2) ≥ 0 ↔ x ≥ 1/2 := by sorry

end sqrt_meaningful_iff_l3238_323878


namespace cubic_odd_and_increasing_negative_l3238_323841

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing_negative : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) :=
by sorry

end cubic_odd_and_increasing_negative_l3238_323841


namespace milk_storage_theorem_l3238_323857

def initial_milk : ℕ := 30000
def pump_out_rate : ℕ := 2880
def pump_out_hours : ℕ := 4
def add_milk_hours : ℕ := 7
def initial_add_rate : ℕ := 1200
def add_rate_increase : ℕ := 200

def final_milk_amount : ℕ := 31080

theorem milk_storage_theorem :
  let milk_after_pump_out := initial_milk - pump_out_rate * pump_out_hours
  let milk_added := add_milk_hours * (initial_add_rate + (initial_add_rate + (add_milk_hours - 1) * add_rate_increase)) / 2
  milk_after_pump_out + milk_added = final_milk_amount := by sorry

end milk_storage_theorem_l3238_323857


namespace external_diagonals_inequality_five_seven_ten_not_valid_l3238_323849

/-- External diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_b : a ≤ b
  b_le_c : b ≤ c
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- Theorem: For valid external diagonals of a right regular prism, a² + b² > c² -/
theorem external_diagonals_inequality (d : ExternalDiagonals) : d.a^2 + d.b^2 > d.c^2 := by
  sorry

/-- The set {5, 7, 10} cannot be the lengths of external diagonals of a right regular prism -/
theorem five_seven_ten_not_valid : ¬∃ (d : ExternalDiagonals), d.a = 5 ∧ d.b = 7 ∧ d.c = 10 := by
  sorry

end external_diagonals_inequality_five_seven_ten_not_valid_l3238_323849


namespace parallel_line_length_l3238_323880

/-- Given a triangle with base 15 inches and two parallel lines dividing it into three equal areas,
    the length of the parallel line closer to the base is 5√3 inches. -/
theorem parallel_line_length (base : ℝ) (parallel_line : ℝ) : 
  base = 15 →
  (parallel_line / base)^2 = 1/3 →
  parallel_line = 5 * Real.sqrt 3 :=
by sorry

end parallel_line_length_l3238_323880


namespace masking_tape_length_l3238_323808

/-- The total length of masking tape needed for four walls -/
def total_tape_length (wall_width1 : ℝ) (wall_width2 : ℝ) : ℝ :=
  2 * wall_width1 + 2 * wall_width2

/-- Theorem: The total length of masking tape needed is 20 meters -/
theorem masking_tape_length :
  total_tape_length 4 6 = 20 :=
by sorry

end masking_tape_length_l3238_323808


namespace complex_exponential_sum_l3238_323818

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (2/3 : ℂ) + (5/8 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (2/3 : ℂ) - (5/8 : ℂ) * Complex.I :=
by sorry

end complex_exponential_sum_l3238_323818


namespace cube_edge_length_l3238_323858

theorem cube_edge_length (box_edge : ℝ) (num_cubes : ℕ) (h1 : box_edge = 1) (h2 : num_cubes = 1000) :
  ∃ (cube_edge : ℝ), cube_edge^3 * num_cubes = box_edge^3 ∧ cube_edge = 0.1 := by
  sorry

end cube_edge_length_l3238_323858


namespace prob_both_selected_l3238_323853

/-- The probability of both Ram and Ravi being selected in an exam -/
theorem prob_both_selected (prob_ram prob_ravi : ℚ) 
  (h_ram : prob_ram = 3/7)
  (h_ravi : prob_ravi = 1/5) :
  prob_ram * prob_ravi = 3/35 := by
  sorry

end prob_both_selected_l3238_323853


namespace or_and_not_implies_false_and_true_l3238_323895

theorem or_and_not_implies_false_and_true (p q : Prop) :
  (p ∨ q) → (¬p) → (¬p ∧ q) := by
  sorry

end or_and_not_implies_false_and_true_l3238_323895


namespace power_equation_solution_l3238_323832

theorem power_equation_solution (n b : ℝ) : n = 2^(1/4) → n^b = 16 → b = 16 := by
  sorry

end power_equation_solution_l3238_323832


namespace sector_arc_length_l3238_323851

theorem sector_arc_length (θ : Real) (A : Real) (l : Real) : 
  θ = 120 * π / 180 →  -- Convert 120° to radians
  A = π →              -- Area of the sector
  l = 2 * Real.sqrt 3 * π / 3 → 
  l = θ * Real.sqrt (2 * A / θ) := by
  sorry

end sector_arc_length_l3238_323851


namespace monica_has_eight_cookies_left_l3238_323897

/-- The number of cookies left for Monica --/
def cookies_left_for_monica (total_cookies : ℕ) (father_cookies : ℕ) : ℕ :=
  let mother_cookies := father_cookies / 2
  let brother_cookies := mother_cookies + 2
  total_cookies - (father_cookies + mother_cookies + brother_cookies)

/-- Theorem stating that Monica has 8 cookies left --/
theorem monica_has_eight_cookies_left :
  cookies_left_for_monica 30 10 = 8 := by
  sorry

end monica_has_eight_cookies_left_l3238_323897


namespace maximize_x_cube_y_fourth_l3238_323891

theorem maximize_x_cube_y_fourth (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 40) :
  x^3 * y^4 ≤ 24^3 * 32^4 ∧ x^3 * y^4 = 24^3 * 32^4 ↔ x = 24 ∧ y = 32 := by
  sorry

end maximize_x_cube_y_fourth_l3238_323891


namespace division_problem_l3238_323889

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by sorry

end division_problem_l3238_323889


namespace division_problem_l3238_323893

theorem division_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 725 →
  quotient = 20 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 36 := by
  sorry

end division_problem_l3238_323893


namespace max_value_theorem_l3238_323884

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 7 + 9 * y * z ≤ (1/2) * Real.sqrt 88 := by
sorry

end max_value_theorem_l3238_323884


namespace unique_A_exists_l3238_323861

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def satisfies_conditions (A : ℕ) : Prop :=
  is_single_digit A ∧
  72 % A = 0 ∧
  (354100 + 10 * A + 6) % 4 = 0 ∧
  (354100 + 10 * A + 6) % 9 = 0

theorem unique_A_exists :
  ∃! A, satisfies_conditions A :=
sorry

end unique_A_exists_l3238_323861


namespace coin_distribution_l3238_323892

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem coin_distribution : distribute 5 3 = 21 := by sorry

end coin_distribution_l3238_323892


namespace june_net_income_l3238_323836

def daily_milk_production : ℕ := 200
def milk_price : ℚ := 355/100
def monthly_expenses : ℕ := 3000
def days_in_june : ℕ := 30

def daily_income : ℚ := daily_milk_production * milk_price

def total_income : ℚ := daily_income * days_in_june

def net_income : ℚ := total_income - monthly_expenses

theorem june_net_income : net_income = 18300 := by
  sorry

end june_net_income_l3238_323836


namespace password_unique_l3238_323873

def is_valid_password (n : ℕ) : Prop :=
  -- The password is an eight-digit number
  100000000 > n ∧ n ≥ 10000000 ∧
  -- The password is a multiple of both 3 and 25
  n % 3 = 0 ∧ n % 25 = 0 ∧
  -- The password is between 20,000,000 and 30,000,000
  n > 20000000 ∧ n < 30000000 ∧
  -- The millions place and the hundred thousands place digits are the same
  (n / 1000000) % 10 = (n / 100000) % 10 ∧
  -- The hundreds digit is 2 less than the ten thousands digit
  (n / 100) % 10 + 2 = (n / 10000) % 10 ∧
  -- The digits in the hundred thousands, ten thousands, and thousands places form a three-digit number
  -- which, when divided by the two-digit number formed by the digits in the ten millions and millions places,
  -- gives a quotient of 25
  ((n / 100000) % 1000) / ((n / 1000000) % 100) = 25

theorem password_unique : ∀ n : ℕ, is_valid_password n ↔ n = 26650350 := by
  sorry

end password_unique_l3238_323873


namespace symmetry_point_xoz_l3238_323807

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xoz plane -/
def symmetry_xoz (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetry_point_xoz :
  let p : Point3D := ⟨1, 2, 3⟩
  symmetry_xoz p = ⟨1, -2, 3⟩ := by
  sorry

end symmetry_point_xoz_l3238_323807


namespace grid_separation_impossible_l3238_323815

/-- Represents a point on the grid -/
structure GridPoint where
  x : Fin 8
  y : Fin 8

/-- Represents a line on the grid -/
structure GridLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line passes through a point -/
def line_passes_through (l : GridLine) (p : GridPoint) : Prop :=
  l.a * p.x.val + l.b * p.y.val + l.c = 0

/-- Checks if two points are separated by a line -/
def points_separated_by_line (l : GridLine) (p1 p2 : GridPoint) : Prop :=
  (l.a * p1.x.val + l.b * p1.y.val + l.c) * (l.a * p2.x.val + l.b * p2.y.val + l.c) < 0

/-- The main theorem stating the impossibility of the grid separation -/
theorem grid_separation_impossible :
  ¬ ∃ (lines : Fin 13 → GridLine),
    (∀ (l : Fin 13) (p : GridPoint), ¬ line_passes_through (lines l) p) ∧
    (∀ (p1 p2 : GridPoint), p1 ≠ p2 → ∃ (l : Fin 13), points_separated_by_line (lines l) p1 p2) :=
by sorry

end grid_separation_impossible_l3238_323815


namespace exam_score_standard_deviations_l3238_323860

/-- Given an exam with mean score and standard deviation, prove the number of standard deviations above the mean for a specific score -/
theorem exam_score_standard_deviations (mean sd : ℝ) (x : ℝ) 
  (h1 : mean - 2 * sd = 58)
  (h2 : mean = 74)
  (h3 : mean + x * sd = 98) :
  x = 3 := by
sorry

end exam_score_standard_deviations_l3238_323860


namespace exact_two_fours_probability_l3238_323825

-- Define the number of dice
def num_dice : ℕ := 15

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the target number we're looking for
def target_number : ℕ := 4

-- Define the number of dice we want to show the target number
def target_count : ℕ := 2

-- Define the probability of rolling the target number on a single die
def single_prob : ℚ := 1 / num_sides

-- Define the probability of not rolling the target number on a single die
def single_prob_complement : ℚ := 1 - single_prob

-- Theorem statement
theorem exact_two_fours_probability :
  (Nat.choose num_dice target_count : ℚ) * single_prob ^ target_count * single_prob_complement ^ (num_dice - target_count) =
  (105 : ℚ) * 5^13 / 6^15 := by
  sorry

end exact_two_fours_probability_l3238_323825


namespace x_gt_one_sufficient_not_necessary_for_x_gt_zero_l3238_323859

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ ¬(x > 1)) :=
by sorry

end x_gt_one_sufficient_not_necessary_for_x_gt_zero_l3238_323859


namespace geometric_sequence_problem_l3238_323867

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function

/-- Conditions for the geometric sequence -/
def GeometricSequenceConditions (seq : GeometricSequence) : Prop :=
  seq.a 3 = 3/2 ∧ seq.S 3 = 9/2

/-- The value m forms a geometric sequence with a₃ and S₃ -/
def FormsGeometricSequence (seq : GeometricSequence) (m : ℝ) : Prop :=
  ∃ q : ℝ, seq.a 3 * q = m ∧ m * q = seq.S 3

theorem geometric_sequence_problem (seq : GeometricSequence) 
  (h : GeometricSequenceConditions seq) :
  (∀ m : ℝ, FormsGeometricSequence seq m → m = 3*Real.sqrt 3/2 ∨ m = -3*Real.sqrt 3/2) ∧
  (seq.a 1 = 3/2 ∨ seq.a 1 = 6) :=
sorry

end geometric_sequence_problem_l3238_323867


namespace function_passes_through_point_l3238_323840

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 := by sorry

end function_passes_through_point_l3238_323840


namespace escalator_length_l3238_323835

/-- The length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
theorem escalator_length 
  (escalator_speed : ℝ) 
  (walking_speed : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15) 
  (h2 : walking_speed = 3) 
  (h3 : time_taken = 10) : 
  escalator_speed * time_taken + walking_speed * time_taken = 180 := by
  sorry

end escalator_length_l3238_323835


namespace jimmy_wins_l3238_323812

/-- Represents a fan with four blades -/
structure Fan :=
  (rotation_speed : ℝ)
  (blade_count : ℕ)

/-- Represents a bullet trajectory -/
structure Trajectory :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Checks if a trajectory intersects a blade at a given position and time -/
def intersects_blade (f : Fan) (t : Trajectory) (position : ℕ) (time : ℝ) : Prop :=
  sorry

/-- The main theorem stating that there exists a trajectory that intersects all blades -/
theorem jimmy_wins (f : Fan) : 
  f.rotation_speed = 50 ∧ f.blade_count = 4 → 
  ∃ t : Trajectory, ∀ p : ℕ, p < f.blade_count → 
    ∃ time : ℝ, intersects_blade f t p time :=
sorry

end jimmy_wins_l3238_323812


namespace exists_function_satisfying_conditions_l3238_323852

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_derivative_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, deriv f (-x) - deriv f x = 2 * Real.sqrt 2 * Real.sin x

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x, x > -3 * Real.pi / 2 → f x ≤ Real.exp (x + Real.pi / 4) - Real.pi / 4

theorem exists_function_satisfying_conditions :
  ∃ f : ℝ → ℝ,
    is_even f ∧
    satisfies_derivative_condition f ∧
    satisfies_inequality f ∧
    f = fun x ↦ Real.sqrt 2 * Real.cos x - 10 := by sorry

end exists_function_satisfying_conditions_l3238_323852


namespace remainder_problem_l3238_323898

theorem remainder_problem (a b : ℕ) (h1 : 3 * a > b) (h2 : a % 5 = 1) (h3 : b % 5 = 4) :
  (3 * a - b) % 5 = 4 := by
  sorry

end remainder_problem_l3238_323898


namespace x_power_2188_minus_reciprocal_l3238_323826

theorem x_power_2188_minus_reciprocal (x : ℂ) :
  x - (1 / x) = Complex.I * Real.sqrt 3 →
  x^2188 - (1 / x^2188) = -1 := by sorry

end x_power_2188_minus_reciprocal_l3238_323826


namespace platform_length_l3238_323833

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 10 seconds, prove that the platform length is 870 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_time = 39)
  (h3 : pole_time = 10) :
  let train_speed := train_length / pole_time
  let platform_length := train_speed * platform_time - train_length
  platform_length = 870 := by sorry

end platform_length_l3238_323833


namespace smallest_N_with_g_geq_10_N_mod_1000_l3238_323814

/-- Sum of digits in base b representation of n -/
def digitSum (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-5 representation of n -/
def f (n : ℕ) : ℕ := digitSum n 5

/-- g(n) is the sum of digits in base-7 representation of f(n) -/
def g (n : ℕ) : ℕ := digitSum (f n) 7

theorem smallest_N_with_g_geq_10 :
  ∃ N : ℕ, (∀ k < N, g k < 10) ∧ g N ≥ 10 ∧ N = 610 := by sorry

theorem N_mod_1000 :
  ∃ N : ℕ, (∀ k < N, g k < 10) ∧ g N ≥ 10 ∧ N ≡ 610 [MOD 1000] := by sorry

end smallest_N_with_g_geq_10_N_mod_1000_l3238_323814


namespace quadratic_max_value_quadratic_max_value_achieved_l3238_323862

theorem quadratic_max_value (s : ℝ) : -3 * s^2 + 24 * s - 7 ≤ 41 := by sorry

theorem quadratic_max_value_achieved : ∃ s : ℝ, -3 * s^2 + 24 * s - 7 = 41 := by sorry

end quadratic_max_value_quadratic_max_value_achieved_l3238_323862


namespace complex_equation_solution_l3238_323801

theorem complex_equation_solution (z : ℂ) :
  (1 + 2 * Complex.I) * z = -3 + 4 * Complex.I →
  z = 3/5 + 12/5 * Complex.I :=
by
  sorry

end complex_equation_solution_l3238_323801


namespace woman_birth_year_l3238_323890

/-- A woman born in the second half of the 19th century was x years old in the year x^2. -/
theorem woman_birth_year :
  ∃ x : ℕ,
    (1850 ≤ x^2 - x) ∧
    (x^2 - x < 1900) ∧
    (x^2 = x + 1892) :=
by sorry

end woman_birth_year_l3238_323890


namespace bug_meeting_point_l3238_323844

/-- Triangle with side lengths 7, 8, and 9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (RP : ℝ)
  (h_PQ : PQ = 7)
  (h_QR : QR = 8)
  (h_RP : RP = 9)

/-- The meeting point of two bugs crawling from P in opposite directions -/
def meetingPoint (t : Triangle) : ℝ := sorry

/-- Theorem stating that QS = 5 in the given triangle -/
theorem bug_meeting_point (t : Triangle) : meetingPoint t = 5 := by sorry

end bug_meeting_point_l3238_323844


namespace quadratic_root_range_l3238_323817

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + a*x - 2 = 0 ↔ x = x₁ ∨ x = x₂) →  -- equation has exactly two roots
  x₁ ≠ x₂ →  -- roots are distinct
  x₁ < -1 →
  x₂ > 1 →
  -1 < a ∧ a < 1 := by
sorry

end quadratic_root_range_l3238_323817


namespace perpendicular_vectors_x_equals_one_l3238_323810

def a : ℝ × ℝ := (-2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, x^2 + 1)

theorem perpendicular_vectors_x_equals_one :
  ∀ x : ℝ, (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 1 := by
  sorry

end perpendicular_vectors_x_equals_one_l3238_323810


namespace joe_fruit_probability_l3238_323888

/-- The number of fruit types available to Joe -/
def num_fruits : ℕ := 4

/-- The number of meals Joe has per day -/
def num_meals : ℕ := 3

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruits

/-- The probability of choosing the same fruit for all meals -/
def prob_same_fruit : ℚ := prob_one_fruit ^ num_meals

/-- The probability of not eating at least two different kinds of fruits -/
def prob_not_varied : ℚ := num_fruits * prob_same_fruit

/-- The probability of eating at least two different kinds of fruits -/
def prob_varied : ℚ := 1 - prob_not_varied

theorem joe_fruit_probability : prob_varied = 15 / 16 := by
  sorry

end joe_fruit_probability_l3238_323888


namespace last_three_digits_of_7_to_123_l3238_323827

theorem last_three_digits_of_7_to_123 : 7^123 % 1000 = 773 := by
  sorry

end last_three_digits_of_7_to_123_l3238_323827


namespace max_value_of_one_minus_cos_l3238_323819

open Real

theorem max_value_of_one_minus_cos (x : ℝ) :
  ∃ (k : ℤ), (∀ y : ℝ, 1 - cos y ≤ 1 - cos (π + 2 * π * ↑k)) ∧
              (1 - cos x = 1 - cos (π + 2 * π * ↑k) ↔ ∃ m : ℤ, x = π + 2 * π * ↑m) := by
  sorry

end max_value_of_one_minus_cos_l3238_323819


namespace equation_solution_l3238_323843

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 0 ∧ x ≠ -3 ∧ (2 / x + x / (x + 3) = 1) ∧ x = 6 := by
  sorry

end equation_solution_l3238_323843


namespace root_sum_reciprocal_l3238_323855

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x : ℝ, x^3 - 18*x^2 + 77*x - 120 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 18*s^2 + 77*s - 120) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 196 := by
sorry

end root_sum_reciprocal_l3238_323855


namespace train_distance_theorem_l3238_323866

/-- Represents the train journey with given conditions -/
structure TrainJourney where
  speed : ℝ
  stop_interval : ℝ
  regular_stop_duration : ℝ
  fifth_stop_duration : ℝ
  total_travel_time : ℝ

/-- Calculates the total distance traveled by the train -/
def total_distance (journey : TrainJourney) : ℝ :=
  sorry

/-- Theorem stating the total distance traveled by the train -/
theorem train_distance_theorem (journey : TrainJourney) 
  (h1 : journey.speed = 60)
  (h2 : journey.stop_interval = 48)
  (h3 : journey.regular_stop_duration = 1/6)
  (h4 : journey.fifth_stop_duration = 1/2)
  (h5 : journey.total_travel_time = 58) :
  total_distance journey = 2870 := by
  sorry

end train_distance_theorem_l3238_323866


namespace remainder_problem_l3238_323846

theorem remainder_problem (d r : ℤ) : 
  d > 1 ∧ 
  1012 % d = r ∧ 
  1548 % d = r ∧ 
  2860 % d = r → 
  d - r = 4 := by
sorry

end remainder_problem_l3238_323846


namespace hexagon_diagonal_intersection_theorem_l3238_323869

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon defined by its six vertices -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- The intersection point of the diagonals -/
def G (h : Hexagon) : Point := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a hexagon is convex -/
def isConvex (h : Hexagon) : Prop := sorry

/-- Checks if a hexagon is inscribed in a circle -/
def isInscribed (h : Hexagon) : Prop := sorry

/-- Checks if three lines intersect at a point forming 60° angles -/
def intersectAt60Degrees (p1 p2 p3 p4 p5 p6 : Point) : Prop := sorry

/-- The main theorem -/
theorem hexagon_diagonal_intersection_theorem (h : Hexagon) 
  (convex : isConvex h)
  (inscribed : isInscribed h)
  (intersect : intersectAt60Degrees h.A h.D h.B h.E h.C h.F) :
  distance (G h) h.A + distance (G h) h.C + distance (G h) h.E =
  distance (G h) h.B + distance (G h) h.D + distance (G h) h.F := by
  sorry

end hexagon_diagonal_intersection_theorem_l3238_323869


namespace parabola_through_point_l3238_323824

theorem parabola_through_point (x y : ℝ) :
  (x = 1 ∧ y = 2) →
  (y^2 = 4*x ∨ x^2 = (1/2)*y) :=
by sorry

end parabola_through_point_l3238_323824


namespace sum_divisible_by_3_probability_l3238_323850

/-- Represents the outcome of rolling a fair 6-sided die -/
def DieRoll : Type := Fin 6

/-- The sample space of rolling a fair die three times -/
def SampleSpace : Type := DieRoll × DieRoll × DieRoll

/-- The number of possible outcomes in the sample space -/
def totalOutcomes : Nat := 216

/-- Predicate for outcomes where the sum is divisible by 3 -/
def sumDivisibleBy3 (outcome : SampleSpace) : Prop :=
  (outcome.1.val + outcome.2.1.val + outcome.2.2.val + 3) % 3 = 0

/-- The number of favorable outcomes (sum divisible by 3) -/
def favorableOutcomes : Nat := 72

/-- The probability of the sum being divisible by 3 -/
def probability : ℚ := favorableOutcomes / totalOutcomes

theorem sum_divisible_by_3_probability :
  probability = 1 / 3 := by sorry

end sum_divisible_by_3_probability_l3238_323850


namespace correct_average_after_error_correction_l3238_323816

theorem correct_average_after_error_correction (n : ℕ) (incorrect_sum correct_sum : ℝ) :
  n = 10 →
  incorrect_sum = 46 * n →
  correct_sum = incorrect_sum + 50 →
  correct_sum / n = 51 := by
  sorry

end correct_average_after_error_correction_l3238_323816


namespace average_of_three_numbers_l3238_323896

theorem average_of_three_numbers (N : ℕ) : 
  15 < N ∧ N < 23 ∧ Even N → 
  (∃ x, x = (8 + 12 + N) / 3 ∧ (x = 12 ∨ x = 14)) :=
sorry

end average_of_three_numbers_l3238_323896


namespace number_of_employees_l3238_323872

-- Define the given constants
def average_salary : ℚ := 1500
def salary_increase : ℚ := 100
def manager_salary : ℚ := 3600

-- Define the number of employees (excluding manager) as a variable
variable (n : ℚ)

-- Define the theorem
theorem number_of_employees : 
  (n * average_salary + manager_salary) / (n + 1) = average_salary + salary_increase →
  n = 20 := by
  sorry

end number_of_employees_l3238_323872


namespace min_sum_a7_a14_l3238_323883

/-- An arithmetic sequence of positive real numbers -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 0 < a n ∧ ∃ d : ℝ, a (n + 1) = a n + d

/-- The theorem stating the minimum value of a_7 + a_14 in the given sequence -/
theorem min_sum_a7_a14 (a : ℕ → ℝ) (h_arith : ArithmeticSequence a) (h_prod : a 1 * a 20 = 100) :
  ∀ x y : ℝ, x = a 7 ∧ y = a 14 → x + y ≥ 20 :=
sorry

end min_sum_a7_a14_l3238_323883


namespace current_rate_calculation_l3238_323804

/-- Given a boat with speed in still water and distance travelled downstream in a specific time,
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (time_minutes : ℝ) :
  boat_speed = 24 ∧ downstream_distance = 6.75 ∧ time_minutes = 15 →
  ∃ (current_rate : ℝ), current_rate = 3 ∧
    boat_speed + current_rate = downstream_distance / (time_minutes / 60) :=
by sorry

end current_rate_calculation_l3238_323804


namespace contractor_absence_l3238_323813

/-- A contractor's work problem -/
theorem contractor_absence (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) :
  total_days = 30 ∧ 
  daily_wage = 25 ∧ 
  daily_fine = (15/2) ∧ 
  total_amount = 425 →
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    daily_wage * days_worked - daily_fine * days_absent = total_amount ∧
    days_absent = 10 := by
  sorry

end contractor_absence_l3238_323813


namespace gear_angular_speed_relationship_l3238_323885

theorem gear_angular_speed_relationship 
  (x y z : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (ω₁ ω₂ ω₃ : ℝ) 
  (h₁ : 2 * x * ω₁ = 3 * y * ω₂) 
  (h₂ : 3 * y * ω₂ = 4 * z * ω₃) :
  ∃ (k : ℝ), k > 0 ∧ 
    ω₁ = k * (2 * z / x) ∧
    ω₂ = k * (4 * z / (3 * y)) ∧
    ω₃ = k :=
by sorry

end gear_angular_speed_relationship_l3238_323885


namespace quadratic_inequality_range_l3238_323876

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + a > 0) → (0 < a ∧ a < 4) := by
  sorry

end quadratic_inequality_range_l3238_323876


namespace negative_integer_square_plus_self_l3238_323828

theorem negative_integer_square_plus_self (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by sorry

end negative_integer_square_plus_self_l3238_323828


namespace proposition_four_l3238_323894

theorem proposition_four (p q : Prop) :
  (p → q) ∧ ¬(q → p) → (¬p → ¬q) ∧ ¬(¬q → ¬p) :=
sorry


end proposition_four_l3238_323894


namespace house_rent_fraction_l3238_323839

theorem house_rent_fraction (salary : ℚ) (food_fraction : ℚ) (clothes_fraction : ℚ) (remaining : ℚ) :
  salary = 180000 →
  food_fraction = 1/5 →
  clothes_fraction = 3/5 →
  remaining = 18000 →
  ∃ (house_rent_fraction : ℚ),
    house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining = salary ∧
    house_rent_fraction = 1/10 :=
by sorry

end house_rent_fraction_l3238_323839


namespace smallest_square_area_l3238_323803

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The smallest square that can contain two rectangles without overlap -/
def smallest_containing_square (r1 r2 : Rectangle) : ℕ :=
  (min r1.width r1.height + min r2.width r2.height) ^ 2

/-- Theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle) 
  (h1 : r1 = ⟨2, 3⟩) 
  (h2 : r2 = ⟨3, 4⟩) : 
  smallest_containing_square r1 r2 = 25 := by
  sorry

#eval smallest_containing_square ⟨2, 3⟩ ⟨3, 4⟩

end smallest_square_area_l3238_323803


namespace min_committee_size_l3238_323834

/-- Represents a committee with the given properties -/
structure Committee where
  meetings : Nat
  attendees_per_meeting : Nat
  total_members : Nat
  attendance : Fin meetings → Finset (Fin total_members)
  ten_per_meeting : ∀ m, (attendance m).card = attendees_per_meeting
  at_most_once : ∀ i j m₁ m₂, i ≠ j → m₁ ≠ m₂ → 
    (i ∈ attendance m₁ ∧ i ∈ attendance m₂) → 
    (j ∉ attendance m₁ ∨ j ∉ attendance m₂)

/-- The main theorem stating the minimum number of members -/
theorem min_committee_size :
  ∀ c : Committee, c.meetings = 12 → c.attendees_per_meeting = 10 → c.total_members ≥ 58 :=
by sorry

end min_committee_size_l3238_323834


namespace range_of_m_for_increasing_function_l3238_323800

-- Define an increasing function on an open interval
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x ∈ Set.Ioo (-2) 2 → y ∈ Set.Ioo (-2) 2 → f x < f y

-- State the theorem
theorem range_of_m_for_increasing_function 
  (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : IncreasingFunction f) 
  (h_inequality : f (m - 1) < f (1 - 2*m)) :
  m ∈ Set.Ioo (-1/2) (2/3) := by
  sorry

end range_of_m_for_increasing_function_l3238_323800


namespace parabola_properties_l3238_323870

/-- Represents a quadratic function of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

def Parabola.opensDownward (p : Parabola) : Prop := p.a < 0

def Parabola.axisOfSymmetry (p : Parabola) : ℝ := p.h

def Parabola.vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

def Parabola.increasingOnInterval (p : Parabola) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → p.a * (x - p.h)^2 + p.k < p.a * (y - p.h)^2 + p.k

theorem parabola_properties (p : Parabola) (h1 : p.a = -1) (h2 : p.h = -1) (h3 : p.k = 3) :
  (p.opensDownward ∧ 
   p.vertex = (-1, 3) ∧ 
   ¬(p.axisOfSymmetry = 1) ∧ 
   ¬(p.increasingOnInterval 0 (-p.h))) := by sorry

end parabola_properties_l3238_323870


namespace remainder_theorem_l3238_323877

theorem remainder_theorem (A : ℕ) 
  (h1 : A % 1981 = 35) 
  (h2 : A % 1982 = 35) : 
  A % 14 = 7 := by
sorry

end remainder_theorem_l3238_323877


namespace age_puzzle_solution_l3238_323865

theorem age_puzzle_solution :
  ∃! x : ℕ, 6 * (x + 6) - 6 * (x - 6) = x ∧ x = 72 :=
by sorry

end age_puzzle_solution_l3238_323865


namespace quartic_root_product_l3238_323863

theorem quartic_root_product (a : ℝ) : 
  (∃ x y : ℝ, x * y = -32 ∧ 
   x^4 - 18*x^3 + a*x^2 + 200*x - 1984 = 0 ∧
   y^4 - 18*y^3 + a*y^2 + 200*y - 1984 = 0) →
  a = 86 := by
sorry

end quartic_root_product_l3238_323863


namespace shaded_area_equals_circle_area_l3238_323868

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- Configuration of the diagram -/
structure DiagramConfig where
  x : ℝ
  A : Point
  B : Point
  C : Point
  D : Point
  semicircleAB : Semicircle
  semicircleAC : Semicircle
  semicircleCB : Semicircle

/-- The main theorem -/
theorem shaded_area_equals_circle_area (config : DiagramConfig) : 
  (config.A.x - config.B.x = 8 * config.x) →
  (config.A.x - config.C.x = 6 * config.x) →
  (config.C.x - config.B.x = 2 * config.x) →
  (config.D.y - config.C.y = Real.sqrt 3 * config.x) →
  (config.semicircleAB.radius = 4 * config.x) →
  (config.semicircleAC.radius = 3 * config.x) →
  (config.semicircleCB.radius = config.x) →
  (config.semicircleAB.center.x = (config.A.x + config.B.x) / 2) →
  (config.semicircleAC.center.x = (config.A.x + config.C.x) / 2) →
  (config.semicircleCB.center.x = (config.C.x + config.B.x) / 2) →
  (config.A.y = config.B.y) →
  (config.C.y = config.A.y) →
  (config.D.x = config.C.x) →
  (π * (4 * config.x)^2 / 2 - π * (3 * config.x)^2 / 2 - π * config.x^2 / 2 = π * (Real.sqrt 3 * config.x)^2) :=
by sorry

end shaded_area_equals_circle_area_l3238_323868


namespace three_teacher_student_pairs_arrangements_l3238_323879

def teacher_student_arrangements (n : ℕ) : ℕ :=
  n.factorial * (2^n)

theorem three_teacher_student_pairs_arrangements :
  teacher_student_arrangements 3 = 48 := by
sorry

end three_teacher_student_pairs_arrangements_l3238_323879


namespace train_length_calculation_l3238_323886

/-- The length of the train in meters -/
def train_length : ℝ := 1200

/-- The time (in seconds) it takes for the train to cross a tree -/
def tree_crossing_time : ℝ := 120

/-- The time (in seconds) it takes for the train to pass a platform -/
def platform_crossing_time : ℝ := 160

/-- The length of the platform in meters -/
def platform_length : ℝ := 400

theorem train_length_calculation :
  train_length = 1200 ∧
  (train_length / tree_crossing_time = (train_length + platform_length) / platform_crossing_time) :=
sorry

end train_length_calculation_l3238_323886


namespace small_prism_surface_area_l3238_323820

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (d : Dimensions) : ℕ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Theorem: Surface area of small prism in arrangement of 9 identical prisms -/
theorem small_prism_surface_area 
  (small : Dimensions) 
  (large_surface_area : ℕ) 
  (h1 : large_surface_area = 360) 
  (h2 : 3 * small.width = 2 * small.length) 
  (h3 : small.length = 3 * small.height) 
  (h4 : surfaceArea { length := 3 * small.width, 
                      width := 3 * small.width, 
                      height := small.length + small.height } = large_surface_area) : 
  surfaceArea small = 88 := by
sorry

end small_prism_surface_area_l3238_323820


namespace problem_solution_l3238_323882

theorem problem_solution : ∃ N : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  N / sum = 2 * diff ∧ N % sum = 80 ∧ N = 220080 := by
  sorry

end problem_solution_l3238_323882


namespace rectangle_perimeter_l3238_323847

theorem rectangle_perimeter (x y : ℝ) 
  (h1 : 2*x + 2*y = x/2 + 2*y + 18) 
  (h2 : x*y = x*y/4 + 18) : 
  2*x + 2*y = 28 := by
sorry

end rectangle_perimeter_l3238_323847


namespace alternating_draw_probability_l3238_323854

def total_balls : ℕ := 9
def white_balls : ℕ := 5
def black_balls : ℕ := 4

def alternating_sequence_probability : ℚ :=
  1 / (total_balls.choose black_balls)

theorem alternating_draw_probability :
  alternating_sequence_probability = 1 / 126 :=
by sorry

end alternating_draw_probability_l3238_323854


namespace remainder_problem_l3238_323848

theorem remainder_problem : (1989 * 1990 * 1991 + 1992^2) % 7 = 0 := by
  sorry

end remainder_problem_l3238_323848


namespace tile_arrangements_l3238_323875

/-- The number of distinguishable arrangements of tiles -/
def num_arrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 brown, 1 purple, 2 green, and 3 yellow tiles is 420 -/
theorem tile_arrangements :
  num_arrangements 1 1 2 3 = 420 := by
  sorry

end tile_arrangements_l3238_323875


namespace point_in_fourth_quadrant_l3238_323809

theorem point_in_fourth_quadrant (a b : ℝ) (z z₁ z₂ : ℂ) :
  z = a + b * Complex.I ∧
  z₁ = 1 + Complex.I ∧
  z₂ = 3 - Complex.I ∧
  z = z₁ * z₂ →
  a > 0 ∧ b < 0 := by
  sorry

end point_in_fourth_quadrant_l3238_323809


namespace complement_of_union_A_B_l3238_323845

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_of_union_A_B : (U \ (A ∪ B)) = {-2, 0} := by sorry

end complement_of_union_A_B_l3238_323845


namespace polynomial_simplification_l3238_323887

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 2 * x - 5) - (2 * x^3 + x^2 + 3 * x + 7) = x^3 + 3 * x^2 - x - 12 := by
  sorry

end polynomial_simplification_l3238_323887


namespace stanley_tire_cost_l3238_323821

/-- The total cost of tires purchased by Stanley -/
def total_cost (num_tires : ℕ) (cost_per_tire : ℚ) : ℚ :=
  num_tires * cost_per_tire

/-- Proof that Stanley's total cost for tires is $240.00 -/
theorem stanley_tire_cost :
  let num_tires : ℕ := 4
  let cost_per_tire : ℚ := 60
  total_cost num_tires cost_per_tire = 240 := by
  sorry

#eval total_cost 4 60

end stanley_tire_cost_l3238_323821


namespace molecular_properties_l3238_323856

structure MolecularSystem where
  surface_distance : ℝ
  internal_distance : ℝ
  surface_attraction : Bool

structure IdealGas where
  temperature : ℝ
  collision_frequency : ℝ

structure OleicAcid where
  diameter : ℝ
  molar_volume : ℝ

def surface_tension (ms : MolecularSystem) : Prop :=
  ms.surface_distance > ms.internal_distance ∧ ms.surface_attraction

def gas_collision_frequency (ig : IdealGas) : Prop :=
  ig.collision_frequency = ig.temperature

def avogadro_estimation (oa : OleicAcid) : Prop :=
  oa.diameter > 0 ∧ oa.molar_volume > 0

theorem molecular_properties 
  (ms : MolecularSystem) 
  (ig : IdealGas) 
  (oa : OleicAcid) : 
  surface_tension ms ∧ 
  gas_collision_frequency ig ∧ 
  avogadro_estimation oa :=
sorry

end molecular_properties_l3238_323856


namespace even_odd_handshakers_l3238_323864

theorem even_odd_handshakers (population : ℕ) : ∃ (even_shakers odd_shakers : ℕ),
  even_shakers + odd_shakers = population ∧ 
  Even odd_shakers := by
  sorry

end even_odd_handshakers_l3238_323864


namespace sequence_classification_l3238_323899

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = aⁿ - 1 (a is a non-zero real number),
    the sequence {aₙ} is either an arithmetic sequence or a geometric sequence. -/
theorem sequence_classification (a : ℝ) (ha : a ≠ 0) :
  let S : ℕ → ℝ := λ n => a^n - 1
  let a_seq : ℕ → ℝ := λ n => S n - S (n-1)
  (∀ n : ℕ, n > 1 → a_seq (n+1) - a_seq n = 0) ∨
  (∀ n : ℕ, n > 2 → a_seq (n+1) / a_seq n = a) :=
by sorry

end sequence_classification_l3238_323899


namespace geometric_sequence_sum_l3238_323871

/-- 
Given a geometric sequence with:
- First term a = 5
- Common ratio r = 2
- Number of terms n = 5

Prove that the sum of this sequence is 155.
-/
theorem geometric_sequence_sum : 
  let a : ℕ := 5  -- first term
  let r : ℕ := 2  -- common ratio
  let n : ℕ := 5  -- number of terms
  (a * (r^n - 1)) / (r - 1) = 155 := by
  sorry

end geometric_sequence_sum_l3238_323871


namespace total_sales_correct_l3238_323874

/-- Represents the sales data for a house --/
structure HouseSale where
  boxes : Nat
  price_per_box : Float
  discount_rate : Float
  discount_threshold : Nat

/-- Represents the sales data for the neighbor --/
structure NeighborSale where
  boxes : Nat
  price_per_box : Float
  exchange_rate : Float

/-- Calculates the total sales in US dollars --/
def calculate_total_sales (green : HouseSale) (yellow : HouseSale) (brown : HouseSale) (neighbor : NeighborSale) (tax_rate : Float) : Float :=
  sorry

/-- The main theorem to prove --/
theorem total_sales_correct (green : HouseSale) (yellow : HouseSale) (brown : HouseSale) (neighbor : NeighborSale) (tax_rate : Float) :
  let green_sale := HouseSale.mk 3 4 0.1 2
  let yellow_sale := HouseSale.mk 3 (13/3) 0 0
  let brown_sale := HouseSale.mk 9 2 0.05 3
  let neighbor_sale := NeighborSale.mk 3 4.5 1.1
  calculate_total_sales green_sale yellow_sale brown_sale neighbor_sale 0.07 = 57.543 :=
  sorry

end total_sales_correct_l3238_323874


namespace linear_function_quadrant_slope_l3238_323881

/-- A linear function passing through the first, second, and third quadrants has a slope between 0 and 2 -/
theorem linear_function_quadrant_slope (k : ℝ) :
  (∀ x y : ℝ, y = k * x + (2 - k)) →
  (∃ x₁ y₁ : ℝ, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = k * x₁ + (2 - k)) →
  (∃ x₂ y₂ : ℝ, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = k * x₂ + (2 - k)) →
  (∃ x₃ y₃ : ℝ, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = k * x₃ + (2 - k)) →
  0 < k ∧ k < 2 :=
by sorry

end linear_function_quadrant_slope_l3238_323881


namespace simplest_radical_among_options_l3238_323842

def is_simplest_radical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ 
  (∀ m : ℕ, m ^ 2 ≤ n → m ^ 2 = n ∨ m ^ 2 < n) ∧
  (∀ a b : ℕ, n = a * b → (a = 1 ∨ b = 1 ∨ ¬ ∃ k : ℕ, k ^ 2 = a))

theorem simplest_radical_among_options :
  is_simplest_radical (Real.sqrt 7) ∧
  ¬ is_simplest_radical (Real.sqrt 9) ∧
  ¬ is_simplest_radical (Real.sqrt 20) ∧
  ¬ is_simplest_radical (Real.sqrt (1/3)) :=
by sorry

end simplest_radical_among_options_l3238_323842


namespace exists_k_good_iff_k_ge_two_l3238_323805

/-- A function is k-good if the GCD of f(m) + n and f(n) + m is at most k for all m ≠ n -/
def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ (m n : ℕ+), m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

/-- There exists a k-good function if and only if k ≥ 2 -/
theorem exists_k_good_iff_k_ge_two :
  ∀ k : ℕ, (∃ f : ℕ+ → ℕ+, IsKGood k f) ↔ k ≥ 2 :=
sorry

end exists_k_good_iff_k_ge_two_l3238_323805


namespace find_x_l3238_323811

theorem find_x : ∃ x : ℕ, 
  (∃ k : ℕ, x = 9 * k) ∧ 
  x^2 > 120 ∧ 
  x < 25 ∧ 
  x % 2 = 1 ∧
  x = 9 := by
sorry

end find_x_l3238_323811


namespace hyperbola_asymptote_relation_l3238_323830

/-- A hyperbola with equation x^2/a + y^2/9 = 1 and asymptotes 3x ± 2y = 0 has a = -4 -/
theorem hyperbola_asymptote_relation (a : ℝ) :
  (∀ x y : ℝ, x^2/a + y^2/9 = 1 ↔ (3*x - 2*y = 0 ∨ 3*x + 2*y = 0)) →
  a = -4 := by
  sorry

end hyperbola_asymptote_relation_l3238_323830


namespace solution_difference_l3238_323837

theorem solution_difference (p q : ℝ) : 
  ((p - 5) * (p + 5) = 17 * p - 85) →
  ((q - 5) * (q + 5) = 17 * q - 85) →
  p ≠ q →
  p > q →
  p - q = 7 := by
sorry

end solution_difference_l3238_323837


namespace consecutive_integers_sum_l3238_323829

theorem consecutive_integers_sum (n : ℤ) : 
  (∀ k : ℤ, n - 4 ≤ k ∧ k ≤ n + 4 → k > 0) →
  (n - 4) + (n - 3) + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 99 →
  n + 4 = 15 :=
by sorry

end consecutive_integers_sum_l3238_323829


namespace extremum_sum_l3238_323823

theorem extremum_sum (a b : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = x^3 + a*x^2 + b*x + a^2) ∧ 
   (f 1 = 10) ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ 10)) →
  a + b = -7 := by
sorry

end extremum_sum_l3238_323823


namespace min_values_and_corresponding_points_l3238_323822

theorem min_values_and_corresponding_points (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 2) : 
  (∃ (min_ab : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 → x*y ≥ min_ab ∧ a*b = min_ab) ∧
  (∃ (min_sum : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 → x + 2*y ≥ min_sum ∧ a + 2*b = min_sum) ∧
  a = (1 + Real.sqrt 2) / 2 ∧ b = (2 + Real.sqrt 2) / 4 := by
  sorry

end min_values_and_corresponding_points_l3238_323822


namespace pete_backward_speed_calculation_l3238_323802

/-- Pete's backward walking speed in miles per hour -/
def pete_backward_speed : ℝ := 12

/-- Susan's forward walking speed in miles per hour -/
def susan_forward_speed : ℝ := 4

/-- Tracy's cartwheel speed in miles per hour -/
def tracy_cartwheel_speed : ℝ := 8

/-- Pete's hand-walking speed in miles per hour -/
def pete_hand_speed : ℝ := 2

theorem pete_backward_speed_calculation :
  (pete_backward_speed = 3 * susan_forward_speed) ∧
  (tracy_cartwheel_speed = 2 * susan_forward_speed) ∧
  (pete_hand_speed = (1/4) * tracy_cartwheel_speed) ∧
  (pete_hand_speed = 2) →
  pete_backward_speed = 12 := by
sorry

end pete_backward_speed_calculation_l3238_323802
