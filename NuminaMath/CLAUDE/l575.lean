import Mathlib

namespace marathon_average_time_l575_57536

-- Define the marathon distance in miles
def marathonDistance : ℕ := 24

-- Define the total time in minutes (3 hours and 36 minutes = 216 minutes)
def totalTimeMinutes : ℕ := 3 * 60 + 36

-- Define the average time per mile
def averageTimePerMile : ℚ := totalTimeMinutes / marathonDistance

-- Theorem statement
theorem marathon_average_time :
  averageTimePerMile = 9 := by sorry

end marathon_average_time_l575_57536


namespace prob_one_rectification_prob_at_least_one_closed_l575_57575

-- Define the number of canteens
def num_canteens : ℕ := 4

-- Define the probability of passing inspection before rectification
def prob_pass_before : ℝ := 0.5

-- Define the probability of passing inspection after rectification
def prob_pass_after : ℝ := 0.8

-- Theorem for the probability that exactly one canteen needs rectification
theorem prob_one_rectification :
  (num_canteens.choose 1 : ℝ) * prob_pass_before^(num_canteens - 1) * (1 - prob_pass_before) = 0.25 := by
  sorry

-- Theorem for the probability that at least one canteen is closed
theorem prob_at_least_one_closed :
  1 - (1 - (1 - prob_pass_before) * (1 - prob_pass_after))^num_canteens = 0.34 := by
  sorry

end prob_one_rectification_prob_at_least_one_closed_l575_57575


namespace prism_volume_l575_57590

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 10)
  (h2 : b * c = 15)
  (h3 : c * a = 18) :
  a * b * c = 30 * Real.sqrt 3 := by
  sorry

end prism_volume_l575_57590


namespace inscribed_circle_radius_for_given_triangle_l575_57539

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Distance from vertex A to side BC -/
  h_a : ℝ
  /-- Sum of distances from B to AC and from C to AB -/
  h_b_plus_h_c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The radius satisfies the relationship with heights -/
  radius_height_relation : 1 / r = 1 / h_a + 2 / h_b_plus_h_c

/-- The theorem stating the radius of the inscribed circle for the given triangle -/
theorem inscribed_circle_radius_for_given_triangle :
  ∀ (t : TriangleWithInscribedCircle),
    t.h_a = 100 ∧ t.h_b_plus_h_c = 300 →
    t.r = 300 / 7 := by
  sorry

end inscribed_circle_radius_for_given_triangle_l575_57539


namespace circle_equation_k_value_l575_57517

theorem circle_equation_k_value (x y k : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 64) → 
  k = 44 :=
sorry

end circle_equation_k_value_l575_57517


namespace noon_temperature_l575_57589

def morning_temp : ℤ := 4
def temp_drop : ℤ := 10

theorem noon_temperature :
  morning_temp - temp_drop = -6 := by
  sorry

end noon_temperature_l575_57589


namespace final_brand_z_percentage_l575_57580

/-- Represents the state of the fuel tank -/
structure TankState where
  brandZ : ℚ  -- Amount of Brand Z gasoline
  brandX : ℚ  -- Amount of Brand X gasoline

/-- Fills the tank with Brand Z gasoline -/
def fillWithZ (s : TankState) : TankState :=
  { brandZ := s.brandZ + (1 - s.brandZ - s.brandX), brandX := s.brandX }

/-- Fills the tank with Brand X gasoline -/
def fillWithX (s : TankState) : TankState :=
  { brandZ := s.brandZ, brandX := s.brandX + (1 - s.brandZ - s.brandX) }

/-- Empties the tank by the given fraction -/
def emptyTank (s : TankState) (fraction : ℚ) : TankState :=
  { brandZ := s.brandZ * (1 - fraction), brandX := s.brandX * (1 - fraction) }

/-- The main theorem stating the final percentage of Brand Z gasoline -/
theorem final_brand_z_percentage : 
  let s0 := TankState.mk 1 0  -- Initial state: full of Brand Z
  let s1 := fillWithX (emptyTank s0 (3/4))  -- 3/4 empty, fill with X
  let s2 := fillWithZ (emptyTank s1 (1/2))  -- 1/2 empty, fill with Z
  let s3 := fillWithX (emptyTank s2 (1/2))  -- 1/2 empty, fill with X
  s3.brandZ / (s3.brandZ + s3.brandX) = 5/16 := by
  sorry

#eval (5/16 : ℚ) * 100  -- Should evaluate to 31.25

end final_brand_z_percentage_l575_57580


namespace angle_measure_proof_l575_57585

theorem angle_measure_proof : ∃! x : ℝ, 0 < x ∧ x < 90 ∧ x + (3 * x^2 + 10) = 90 := by
  sorry

end angle_measure_proof_l575_57585


namespace divisible_by_six_l575_57545

theorem divisible_by_six (m : ℕ) : ∃ k : ℤ, (m : ℤ)^3 + 11 * m = 6 * k := by
  sorry

end divisible_by_six_l575_57545


namespace mod_twelve_equiv_nine_l575_57542

theorem mod_twelve_equiv_nine : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ -2187 [ZMOD 12] ∧ n = 9 := by
  sorry

end mod_twelve_equiv_nine_l575_57542


namespace no_consistent_solution_l575_57592

theorem no_consistent_solution :
  ¬ ∃ (x y : ℕ+) (z : ℤ),
    (∃ (q : ℕ), x = 11 * y + 4 ∧ x = 11 * q + 4) ∧
    (∃ (q : ℕ), 2 * x = 8 * (3 * y) + 3 ∧ 2 * x = 8 * q + 3) ∧
    (∃ (q : ℕ), x + z = 17 * (2 * y) + 5 ∧ x + z = 17 * q + 5) :=
by sorry

end no_consistent_solution_l575_57592


namespace function_range_l575_57584

theorem function_range (m : ℝ) : 
  (∀ x : ℝ, (2 * m * x^2 - 2 * (4 - m) * x + 1 > 0) ∨ (m * x > 0)) → 
  (m > 0 ∧ m < 8) := by
sorry

end function_range_l575_57584


namespace circular_course_circumference_l575_57576

/-- The circumference of a circular course where two people walking at different speeds meet after a certain time. -/
theorem circular_course_circumference
  (speed_a speed_b : ℝ)
  (meeting_time : ℝ)
  (h1 : speed_a = 4)
  (h2 : speed_b = 5)
  (h3 : meeting_time = 115)
  (h4 : speed_b > speed_a) :
  (speed_b - speed_a) * meeting_time = 115 :=
by sorry

end circular_course_circumference_l575_57576


namespace min_value_on_circle_l575_57518

theorem min_value_on_circle (x y : ℝ) : 
  (x - 1)^2 + (y - 2)^2 = 9 → y ≥ 2 → x + Real.sqrt 3 * y ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end min_value_on_circle_l575_57518


namespace line_not_in_first_quadrant_l575_57507

/-- A line that does not pass through the first quadrant has a non-positive slope -/
def not_in_first_quadrant (t : ℝ) : Prop :=
  3 - 2 * t ≤ 0

/-- The range of t for which the line (2t-3)x + y + 6 = 0 does not pass through the first quadrant -/
def t_range : Set ℝ :=
  {t : ℝ | t ≥ 3/2}

theorem line_not_in_first_quadrant :
  ∀ t : ℝ, not_in_first_quadrant t ↔ t ∈ t_range :=
sorry

end line_not_in_first_quadrant_l575_57507


namespace no_member_divisible_by_4_or_5_l575_57512

def T : Set Int := {x | ∃ n : Int, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2}

theorem no_member_divisible_by_4_or_5 : ∀ x ∈ T, ¬(x % 4 = 0 ∨ x % 5 = 0) := by
  sorry

end no_member_divisible_by_4_or_5_l575_57512


namespace max_a_value_l575_57572

theorem max_a_value (x y a : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 17) 
  (h4 : (3/4) * x = (5/6) * y + a) (h5 : a > 0) : a < 51/4 := by
  sorry

end max_a_value_l575_57572


namespace no_function_satisfies_inequality_l575_57582

theorem no_function_satisfies_inequality :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y z : ℝ), f (x * y) + f (x * z) - f x * f (y * z) > 1 := by
  sorry

end no_function_satisfies_inequality_l575_57582


namespace jam_cost_proof_l575_57571

/-- The cost of jam used for all sandwiches --/
def jam_cost (N B J H : ℕ+) : ℚ :=
  (N * J * 7 : ℚ) / 100

/-- The total cost of ingredients for all sandwiches --/
def total_cost (N B J H : ℕ+) : ℚ :=
  (N * (6 * B + 7 * J + 4 * H) : ℚ) / 100

theorem jam_cost_proof (N B J H : ℕ+) (h1 : N > 1) (h2 : total_cost N B J H = 462/100) :
  jam_cost N B J H = 462/100 := by
  sorry

end jam_cost_proof_l575_57571


namespace max_area_prime_sides_l575_57513

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- The perimeter of the rectangle is 40 meters. -/
def perimeter : ℕ := 40

/-- The theorem stating that the maximum area of a rectangular enclosure with prime side lengths and a perimeter of 40 meters is 91 square meters. -/
theorem max_area_prime_sides : 
  ∀ l w : ℕ, 
    isPrime l → 
    isPrime w → 
    l + w = perimeter / 2 → 
    l * w ≤ 91 :=
sorry

end max_area_prime_sides_l575_57513


namespace common_root_pairs_l575_57540

theorem common_root_pairs (n : ℕ) (hn : n > 1) :
  ∀ (a b : ℤ), (∃ (x : ℝ), x^n + a*x - 2008 = 0 ∧ x^n + b*x - 2009 = 0) ↔
    ((a = 2007 ∧ b = 2008) ∨ (a = (-1)^(n-1) - 2008 ∧ b = (-1)^(n-1) - 2009)) :=
by sorry

end common_root_pairs_l575_57540


namespace trig_simplification_l575_57516

theorem trig_simplification :
  let x : Real := 40 * π / 180
  let y : Real := 50 * π / 180
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.sin y ^ 2)) = 1 := by
  sorry

end trig_simplification_l575_57516


namespace solve_for_y_l575_57548

theorem solve_for_y (x y : ℝ) 
  (eq1 : 9823 + x = 13200) 
  (eq2 : x = y / 3 + 37.5) : 
  y = 10018.5 := by
sorry

end solve_for_y_l575_57548


namespace binary_11010_equals_octal_32_l575_57568

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal [true, true, false, true, false]) = [3, 2] := by
  sorry

end binary_11010_equals_octal_32_l575_57568


namespace line_solution_l575_57525

/-- Given a line y = ax + b (a ≠ 0) passing through points (0,4) and (-3,0),
    the solution to ax + b = 0 is x = -3. -/
theorem line_solution (a b : ℝ) (ha : a ≠ 0) :
  (4 = b) →                        -- Line passes through (0,4)
  (0 = -3*a + b) →                 -- Line passes through (-3,0)
  (∀ x, a*x + b = 0 ↔ x = -3) :=   -- Solution to ax + b = 0 is x = -3
by
  sorry

end line_solution_l575_57525


namespace ratio_problem_l575_57581

theorem ratio_problem (a b c : ℚ) 
  (h1 : c / b = 4)
  (h2 : b / a = 2)
  (h3 : c = 20 - 7 * b) :
  a = 10 / 11 := by
sorry

end ratio_problem_l575_57581


namespace decreasing_f_implies_a_leq_neg_three_l575_57538

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem decreasing_f_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) → a ≤ -3 := by sorry

end decreasing_f_implies_a_leq_neg_three_l575_57538


namespace price_of_car_is_five_l575_57543

/-- Calculates the price of one little car given the total earnings, cost of Legos, and number of cars sold. -/
def price_of_one_car (total_earnings : ℕ) (legos_cost : ℕ) (num_cars : ℕ) : ℚ :=
  (total_earnings - legos_cost : ℚ) / num_cars

/-- Theorem stating that the price of one little car is $5 given the problem conditions. -/
theorem price_of_car_is_five :
  price_of_one_car 45 30 3 = 5 := by
  sorry

#eval price_of_one_car 45 30 3

end price_of_car_is_five_l575_57543


namespace complex_modulus_problem_l575_57528

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 4 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l575_57528


namespace or_true_if_one_true_l575_57591

theorem or_true_if_one_true (p q : Prop) (h : p ∨ q) : p ∨ q := by
  sorry

end or_true_if_one_true_l575_57591


namespace consecutive_integers_square_difference_l575_57535

theorem consecutive_integers_square_difference :
  ∃ n : ℕ, 
    (n > 0) ∧ 
    (n + (n + 1) + (n + 2) < 150) ∧ 
    ((n + 2)^2 - n^2 = 144) :=
by sorry

end consecutive_integers_square_difference_l575_57535


namespace sqrt_x_squared_plus_6x_plus_9_l575_57562

theorem sqrt_x_squared_plus_6x_plus_9 (x : ℝ) (h : x = Real.sqrt 5 - 3) :
  Real.sqrt (x^2 + 6*x + 9) = Real.sqrt 5 := by
  sorry

end sqrt_x_squared_plus_6x_plus_9_l575_57562


namespace beckys_necklaces_l575_57508

theorem beckys_necklaces (initial_count : ℕ) (broken : ℕ) (new_purchases : ℕ) (final_count : ℕ)
  (h1 : initial_count = 50)
  (h2 : broken = 3)
  (h3 : new_purchases = 5)
  (h4 : final_count = 37) :
  initial_count - broken + new_purchases - final_count = 15 := by
  sorry

end beckys_necklaces_l575_57508


namespace sequence_formula_l575_57520

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = (1/3) * (a n - 1)) :
  ∀ n : ℕ+, a n = n + 1 := by sorry

end sequence_formula_l575_57520


namespace sum_of_specific_terms_l575_57558

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 + 3n,
    prove that the sum of the 6th, 7th, and 8th terms is 48. -/
theorem sum_of_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h : ∀ n, S n = n^2 + 3*n) :
  a 6 + a 7 + a 8 = 48 := by
sorry

end sum_of_specific_terms_l575_57558


namespace cube_volume_problem_l575_57522

theorem cube_volume_problem (x : ℝ) (h : x > 0) :
  (x - 2) * x * (x + 2) = x^3 - 10 → x^3 = 15.625 := by
  sorry

end cube_volume_problem_l575_57522


namespace arithmetic_geometric_sequence_l575_57524

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a1, a3, and a4 form a geometric sequence
  a 2 = -6 := by sorry

end arithmetic_geometric_sequence_l575_57524


namespace triangle_angle_problem_l575_57527

theorem triangle_angle_problem (a b : ℝ) (B : ℝ) (A : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  B = 30 * π / 180 →
  0 < A →
  A < π →
  (A = π / 3 ∨ A = 2 * π / 3) := by
  sorry

end triangle_angle_problem_l575_57527


namespace walking_speed_problem_l575_57554

/-- Proves that given the conditions of the walking problem, Deepak's speed is 4.5 km/hr -/
theorem walking_speed_problem (track_circumference : ℝ) (wife_speed : ℝ) (meeting_time : ℝ) :
  track_circumference = 528 →
  wife_speed = 3.75 →
  meeting_time = 3.84 →
  ∃ (deepak_speed : ℝ),
    deepak_speed = 4.5 ∧
    (wife_speed * 1000 / 60) * meeting_time + deepak_speed * 1000 / 60 * meeting_time = track_circumference :=
by sorry

end walking_speed_problem_l575_57554


namespace coffee_doughnut_problem_l575_57599

theorem coffee_doughnut_problem :
  ∀ (c d : ℕ),
    c + d = 7 →
    (90 * c + 60 * d) % 100 = 0 →
    c = 6 := by
  sorry

end coffee_doughnut_problem_l575_57599


namespace max_large_chips_l575_57530

theorem max_large_chips (total : ℕ) (small large : ℕ → ℕ) (composite : ℕ → ℕ) : 
  total = 72 →
  (∀ n, total = small n + large n) →
  (∀ n, small n = large n + composite n) →
  (∀ n, composite n ≥ 4) →
  (∃ max_large : ℕ, ∀ n, large n ≤ max_large ∧ (∃ m, large m = max_large)) →
  (∃ max_large : ℕ, max_large = 34 ∧ ∀ n, large n ≤ max_large) :=
by sorry

end max_large_chips_l575_57530


namespace fred_car_wash_earnings_l575_57579

/-- The amount of money Fred made washing cars -/
def fred_earnings (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem: Fred made 63 dollars washing cars -/
theorem fred_car_wash_earnings : fred_earnings 23 86 = 63 := by
  sorry

end fred_car_wash_earnings_l575_57579


namespace all_contradictions_valid_l575_57506

/-- A type representing the different kinds of contradictions in a proof by contradiction -/
inductive ContradictionType
  | KnownFact
  | Assumption
  | DefinitionTheoremAxiomLaw
  | Fact

/-- Definition of a valid contradiction in a proof by contradiction -/
def is_valid_contradiction (c : ContradictionType) : Prop :=
  match c with
  | ContradictionType.KnownFact => True
  | ContradictionType.Assumption => True
  | ContradictionType.DefinitionTheoremAxiomLaw => True
  | ContradictionType.Fact => True

/-- Theorem stating that all types of contradictions are valid in a proof by contradiction -/
theorem all_contradictions_valid :
  ∀ (c : ContradictionType), is_valid_contradiction c :=
by sorry

end all_contradictions_valid_l575_57506


namespace series_sum_equals_three_fourths_l575_57537

/-- The sum of the series 1/(n(n+2)) from n=1 to infinity equals 3/4 -/
theorem series_sum_equals_three_fourths :
  ∑' n, (1 : ℝ) / (n * (n + 2)) = 3/4 := by sorry

end series_sum_equals_three_fourths_l575_57537


namespace arjun_has_largest_result_l575_57501

def initial_number : ℕ := 15

def liam_result : ℕ := ((initial_number - 2) * 3) + 3

def maya_result : ℕ := ((initial_number * 3) - 4) + 5

def arjun_result : ℕ := ((initial_number - 3) + 4) * 3

theorem arjun_has_largest_result :
  arjun_result > liam_result ∧ arjun_result > maya_result :=
by sorry

end arjun_has_largest_result_l575_57501


namespace largest_number_in_ratio_l575_57587

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  (a + b + c) / 3 = 20 →
  c = 24 := by
sorry

end largest_number_in_ratio_l575_57587


namespace permutation_count_modulo_l575_57546

/-- The number of characters in the string -/
def string_length : ℕ := 15

/-- The number of A's in the string -/
def num_A : ℕ := 4

/-- The number of B's in the string -/
def num_B : ℕ := 5

/-- The number of C's in the string -/
def num_C : ℕ := 5

/-- The number of D's in the string -/
def num_D : ℕ := 2

/-- The length of the first segment where A's are not allowed -/
def first_segment : ℕ := 4

/-- The length of the second segment where B's are not allowed -/
def second_segment : ℕ := 5

/-- The length of the third segment where C's and D's are not allowed -/
def third_segment : ℕ := 6

/-- The function to calculate the number of valid permutations -/
def num_permutations : ℕ := sorry

theorem permutation_count_modulo :
  num_permutations ≡ 715 [MOD 1000] := by sorry

end permutation_count_modulo_l575_57546


namespace slope_range_l575_57551

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l passing through (0,2) with slope k
def line (x y k : ℝ) : Prop := y = k * x + 2

-- Define the condition for intersection points
def intersects (k : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  x₁ ≠ x₂ ∧ ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ k ∧ line x₂ y₂ k

-- Define the acute angle condition
def acute_angle (k : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂ : ℝ, 
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ k ∧ line x₂ y₂ k → 
  x₁ * x₂ + y₁ * y₂ > 0

-- Main theorem
theorem slope_range : 
  ∀ k : ℝ, intersects k ∧ acute_angle k ↔ 
  (k > Real.sqrt 3 / 2 ∧ k < 2) ∨ (k < -Real.sqrt 3 / 2 ∧ k > -2) := by
  sorry

end slope_range_l575_57551


namespace min_packs_for_120_cans_l575_57515

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize  -- 9 cans
  | medium : PackSize -- 18 cans
  | large : PackSize  -- 30 cans

/-- Calculates the number of cans in a given pack -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | .small => 9
  | .medium => 18
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Checks if a combination qualifies for the promotion -/
def qualifiesForPromotion (c : PackCombination) : Bool :=
  c.large ≥ 2

/-- Represents the store's promotion rule -/
def applyPromotion (c : PackCombination) : PackCombination :=
  if qualifiesForPromotion c then
    { c with small := c.small + 1 }
  else
    c

/-- Calculates the total number of packs in a combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- The main theorem to prove -/
theorem min_packs_for_120_cans :
  ∃ (c : PackCombination),
    totalCans (applyPromotion c) = 120 ∧
    totalPacks c = 4 ∧
    (∀ (c' : PackCombination),
      totalCans (applyPromotion c') = 120 →
      totalPacks c' ≥ totalPacks c) :=
  sorry


end min_packs_for_120_cans_l575_57515


namespace no_valid_ab_pairs_l575_57534

theorem no_valid_ab_pairs : 
  ¬∃ (a b : ℝ), ∃ (x y : ℤ), 
    (3 * a * x + 7 * b * y = 3) ∧ 
    (x^2 + y^2 = 85) ∧ 
    (x % 5 = 0 ∨ y % 5 = 0) :=
sorry

end no_valid_ab_pairs_l575_57534


namespace correct_factorization_l575_57597

theorem correct_factorization (x y : ℝ) : x^2 - 2*x*y + x = x*(x - 2*y + 1) := by
  sorry

end correct_factorization_l575_57597


namespace points_per_correct_answer_l575_57509

theorem points_per_correct_answer 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (final_score : ℚ) 
  (incorrect_penalty : ℚ) 
  (h1 : total_questions = 120)
  (h2 : correct_answers = 104)
  (h3 : final_score = 100)
  (h4 : incorrect_penalty = -1/4)
  (h5 : correct_answers ≤ total_questions) :
  ∃ (points_per_correct : ℚ), 
    points_per_correct * correct_answers + 
    incorrect_penalty * (total_questions - correct_answers) = final_score ∧
    points_per_correct = 1 :=
by sorry

end points_per_correct_answer_l575_57509


namespace distribution_plans_for_given_conditions_l575_57514

/-- The number of ways to distribute employees between two departments --/
def distribution_plans (total_employees : ℕ) (translators : ℕ) (programmers : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of distribution plans for the given conditions --/
theorem distribution_plans_for_given_conditions :
  distribution_plans 8 2 3 = 36 :=
sorry

end distribution_plans_for_given_conditions_l575_57514


namespace one_face_colored_count_l575_57594

/-- Represents a cube that has been painted and cut into smaller cubes -/
structure PaintedCube where
  edge_count : Nat
  is_painted : Bool

/-- Counts the number of small cubes with exactly one face colored -/
def count_one_face_colored (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem: A cube painted on all faces and cut into 5x5x5 smaller cubes
    will have 54 small cubes with exactly one face colored -/
theorem one_face_colored_count (cube : PaintedCube) :
  cube.edge_count = 5 → cube.is_painted → count_one_face_colored cube = 54 := by
  sorry

end one_face_colored_count_l575_57594


namespace johns_speed_l575_57510

/-- Prove that John's speed during his final push was 4.2 m/s given the race conditions --/
theorem johns_speed (initial_gap : ℝ) (steve_speed : ℝ) (final_gap : ℝ) (push_duration : ℝ) : 
  initial_gap = 14 →
  steve_speed = 3.7 →
  final_gap = 2 →
  push_duration = 32 →
  (initial_gap + final_gap) / push_duration + steve_speed = 4.2 := by
sorry

end johns_speed_l575_57510


namespace parallel_tangent_length_l575_57553

/-- Represents an isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  base : ℝ
  height : ℝ
  inscribed_circle : Circle

/-- Represents a tangent line to the inscribed circle, parallel to the base -/
structure ParallelTangent where
  triangle : IsoscelesTriangleWithInscribedCircle
  length : ℝ

/-- The theorem statement -/
theorem parallel_tangent_length 
  (triangle : IsoscelesTriangleWithInscribedCircle) 
  (tangent : ParallelTangent) 
  (h1 : triangle.base = 12)
  (h2 : triangle.height = 8)
  (h3 : tangent.triangle = triangle) : 
  tangent.length = 3 := by sorry

end parallel_tangent_length_l575_57553


namespace prime_diff_perfect_square_pairs_l575_57521

theorem prime_diff_perfect_square_pairs (m n : ℕ+) (p : ℕ) :
  p.Prime →
  m - n = p →
  ∃ k : ℕ, m * n = k^2 →
  p % 2 = 1 ∧ m = ((p + 1)^2 / 4 : ℕ) ∧ n = ((p - 1)^2 / 4 : ℕ) := by
  sorry

end prime_diff_perfect_square_pairs_l575_57521


namespace sequence_difference_l575_57559

theorem sequence_difference (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a (n + 1) - a n - n = 0) : 
  a 2017 - a 2016 = 2016 := by
  sorry

end sequence_difference_l575_57559


namespace collinear_points_sum_l575_57529

/-- Three points in ℝ³ are collinear if they lie on the same line. -/
def collinear (A B C : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C - A = t • (B - A)

/-- The theorem states that if A(1,3,-2), B(2,5,1), and C(p,7,q-2) are collinear in ℝ³, 
    then p+q = 9. -/
theorem collinear_points_sum (p q : ℝ) : 
  let A : ℝ × ℝ × ℝ := (1, 3, -2)
  let B : ℝ × ℝ × ℝ := (2, 5, 1)
  let C : ℝ × ℝ × ℝ := (p, 7, q-2)
  collinear A B C → p + q = 9 := by
  sorry

end collinear_points_sum_l575_57529


namespace larger_number_problem_l575_57583

theorem larger_number_problem (x y : ℝ) : 
  x - y = 7 → x + y = 35 → max x y = 21 := by
  sorry

end larger_number_problem_l575_57583


namespace value_of_a_value_of_b_when_perpendicular_distance_when_parallel_l575_57547

-- Define the lines
def l1 (a : ℝ) : ℝ → ℝ → Prop := λ x y => a * x + 2 * y - 1 = 0
def l2 (b : ℝ) : ℝ → ℝ → Prop := λ x y => x + b * y - 3 = 0

-- Define the angle of inclination
def angle_of_inclination (l : ℝ → ℝ → Prop) : ℝ := sorry

-- Define perpendicularity
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define parallelism
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define distance between parallel lines
def distance_between_parallel_lines (l1 l2 : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statements
theorem value_of_a (a : ℝ) : 
  angle_of_inclination (l1 a) = π / 4 → a = -2 := by sorry

theorem value_of_b_when_perpendicular (b : ℝ) : 
  perpendicular (l1 (-2)) (l2 b) → b = 1 := by sorry

theorem distance_when_parallel (b : ℝ) : 
  parallel (l1 (-2)) (l2 b) → 
  distance_between_parallel_lines (l1 (-2)) (l2 b) = 7 * Real.sqrt 2 / 4 := by sorry

end value_of_a_value_of_b_when_perpendicular_distance_when_parallel_l575_57547


namespace ice_cream_permutations_l575_57505

theorem ice_cream_permutations :
  Finset.card (Finset.univ.image (fun σ : Equiv.Perm (Fin 5) => σ)) = 120 := by
  sorry

end ice_cream_permutations_l575_57505


namespace alcohol_dilution_l575_57588

theorem alcohol_dilution (initial_volume : ℝ) (initial_alcohol_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_alcohol_percentage = 20 →
  added_water = 5 →
  let initial_alcohol := initial_volume * (initial_alcohol_percentage / 100)
  let new_volume := initial_volume + added_water
  let new_alcohol_percentage := (initial_alcohol / new_volume) * 100
  new_alcohol_percentage = 15 := by
  sorry

end alcohol_dilution_l575_57588


namespace assignment_ways_theorem_l575_57532

/-- The number of ways to assign 7 friends to 7 rooms with at most 3 friends per room -/
def assignment_ways : ℕ := 17640

/-- The number of rooms in the inn -/
def num_rooms : ℕ := 7

/-- The number of friends arriving -/
def num_friends : ℕ := 7

/-- The maximum number of friends allowed per room -/
def max_per_room : ℕ := 3

/-- Theorem stating that the number of ways to assign 7 friends to 7 rooms,
    with at most 3 friends per room, is equal to 17640 -/
theorem assignment_ways_theorem :
  ∃ (ways : ℕ → ℕ → ℕ → ℕ),
    ways num_rooms num_friends max_per_room = assignment_ways :=
by sorry

end assignment_ways_theorem_l575_57532


namespace x_squared_mod_25_l575_57595

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 2 * x ≡ 22 [ZMOD 25]) :
  x^2 ≡ 9 [ZMOD 25] := by
  sorry

end x_squared_mod_25_l575_57595


namespace cone_sphere_volume_ratio_l575_57557

theorem cone_sphere_volume_ratio (r : ℝ) (h : ℝ) : 
  r > 0 → h = 2 * r → 
  (1 / 3 * π * r^2 * h) / (4 / 3 * π * r^3) = 1 / 2 := by
sorry

end cone_sphere_volume_ratio_l575_57557


namespace queen_spade_probability_l575_57555

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)

/-- Represents a Queen card -/
def is_queen (card : Nat × Nat) : Prop := card.1 = 12

/-- Represents a Spade card -/
def is_spade (card : Nat × Nat) : Prop := card.2 = 3

/-- The probability of drawing a Queen as the first card and a Spade as the second card -/
def queen_spade_prob (d : Deck) : ℚ :=
  18 / 221

theorem queen_spade_probability (d : Deck) :
  queen_spade_prob d = 18 / 221 :=
sorry

end queen_spade_probability_l575_57555


namespace lcm_of_ratio_and_hcf_l575_57561

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → Nat.gcd a b = 5 → Nat.lcm a b = 60 := by
  sorry

end lcm_of_ratio_and_hcf_l575_57561


namespace dad_age_is_36_l575_57502

-- Define the current ages
def talia_age : ℕ := 13
def mom_age : ℕ := 39
def dad_age : ℕ := 36
def grandpa_age : ℕ := 18

-- Define the theorem
theorem dad_age_is_36 :
  (talia_age + 7 = 20) ∧
  (mom_age = 3 * talia_age) ∧
  (dad_age + 2 = grandpa_age + 2 + 5) ∧
  (dad_age + 3 = mom_age) ∧
  (grandpa_age + 3 = (mom_age + 3) / 2) →
  dad_age = 36 := by
  sorry

end dad_age_is_36_l575_57502


namespace concert_ticket_revenue_l575_57564

/-- Calculate the total revenue from concert ticket sales --/
theorem concert_ticket_revenue : 
  let original_price : ℚ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let third_group_size : ℕ := 15
  let first_discount : ℚ := 0.4
  let second_discount : ℚ := 0.15
  let third_premium : ℚ := 0.1
  let first_group_revenue := first_group_size * (original_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (original_price * (1 - second_discount))
  let third_group_revenue := third_group_size * (original_price * (1 + third_premium))
  let total_revenue := first_group_revenue + second_group_revenue + third_group_revenue
  total_revenue = 790 := by
  sorry


end concert_ticket_revenue_l575_57564


namespace candidate_count_l575_57565

theorem candidate_count (total : ℕ) (selected_A selected_B : ℕ) : 
  selected_A = (6 * total) / 100 →
  selected_B = (7 * total) / 100 →
  selected_B = selected_A + 81 →
  total = 8100 := by
sorry

end candidate_count_l575_57565


namespace bank_deposit_theorem_l575_57552

/-- Calculates the actual amount of principal and interest after one year,
    given an initial deposit, annual interest rate, and interest tax rate. -/
def actual_amount (initial_deposit : ℝ) (interest_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  initial_deposit + (1 - tax_rate) * interest_rate * initial_deposit

theorem bank_deposit_theorem (x : ℝ) :
  actual_amount x 0.0225 0.2 = (0.8 * 0.0225 * x + x) := by sorry

end bank_deposit_theorem_l575_57552


namespace increasing_f_range_of_a_l575_57544

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 4 * a
  else Real.log x / Real.log a

-- Theorem statement
theorem increasing_f_range_of_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (6/5) 6 ∧ a ≠ 6 :=
sorry

end increasing_f_range_of_a_l575_57544


namespace tim_zoo_cost_l575_57578

/-- The total cost of Tim's animals for his zoo -/
def total_cost (num_goats : ℕ) (goat_cost : ℚ) : ℚ :=
  let num_llamas := 2 * num_goats
  let llama_cost := goat_cost * (1 + 1/2)
  num_goats * goat_cost + num_llamas * llama_cost

/-- Theorem stating that Tim's total cost for animals is $4800 -/
theorem tim_zoo_cost : total_cost 3 400 = 4800 := by
  sorry

end tim_zoo_cost_l575_57578


namespace power_sum_of_i_l575_57570

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^47 = -2*i := by sorry

end power_sum_of_i_l575_57570


namespace f_neg_five_eq_twelve_l575_57503

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem f_neg_five_eq_twelve : f (-5) = 12 := by
  sorry

end f_neg_five_eq_twelve_l575_57503


namespace fiction_books_count_l575_57550

theorem fiction_books_count (total : ℕ) (picture_books : ℕ) : 
  total = 35 → picture_books = 11 → ∃ (fiction : ℕ), 
    fiction + (fiction + 4) + 2 * fiction + picture_books = total ∧ fiction = 5 := by
  sorry

end fiction_books_count_l575_57550


namespace only_four_points_l575_57560

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- Three points are collinear if the area of the triangle they form is zero -/
def collinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop :=
  triangleArea p₁ p₂ p₃ = 0

/-- A valid configuration satisfies the problem conditions -/
def validConfiguration {n : ℕ} (config : PointConfiguration n) : Prop :=
  (n > 3) ∧
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬collinear (config.points i) (config.points j) (config.points k)) ∧
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) =
      config.r i + config.r j + config.r k)

/-- The main theorem: The only valid configuration is for n = 4 -/
theorem only_four_points :
  ∀ n : ℕ, (∃ config : PointConfiguration n, validConfiguration config) → n = 4 :=
sorry

end only_four_points_l575_57560


namespace max_value_product_sum_l575_57541

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 15 →
    A' * M' * C' + A' * M' + M' * C' + C' * A' ≤ A * M * C + A * M + M * C + C * A) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end max_value_product_sum_l575_57541


namespace rectangle_DC_length_l575_57573

/-- Represents a rectangle ABCF with points E and D on FC -/
structure Rectangle :=
  (AB : ℝ)
  (AF : ℝ)
  (FE : ℝ)
  (area_ABDE : ℝ)

/-- The length of DC in the rectangle -/
def length_DC (r : Rectangle) : ℝ :=
  -- Definition of DC length
  sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem rectangle_DC_length (r : Rectangle) 
  (h1 : r.AB = 30)
  (h2 : r.AF = 14)
  (h3 : r.FE = 5)
  (h4 : r.area_ABDE = 266) :
  length_DC r = 17 :=
sorry

end rectangle_DC_length_l575_57573


namespace negative_product_expression_B_l575_57574

theorem negative_product_expression_B : 
  let a : ℚ := -9
  let b : ℚ := 1/8
  let c : ℚ := -4/7
  let d : ℚ := 7
  let e : ℚ := -1/3
  a * b * c * d * e < 0 := by
  sorry

end negative_product_expression_B_l575_57574


namespace larger_number_of_pair_l575_57533

theorem larger_number_of_pair (x y : ℝ) (h1 : x + y = 29) (h2 : x - y = 5) : 
  max x y = 17 := by
sorry

end larger_number_of_pair_l575_57533


namespace car_speed_problem_l575_57556

/-- Proves that given the conditions of the car problem, the speed of Car X is approximately 33.87 mph -/
theorem car_speed_problem (speed_y : ℝ) (time_diff : ℝ) (distance_x : ℝ) :
  speed_y = 42 →
  time_diff = 72 / 60 →
  distance_x = 210 →
  ∃ (speed_x : ℝ), 
    speed_x > 0 ∧ 
    speed_x * (distance_x / speed_y + time_diff) = distance_x ∧ 
    (abs (speed_x - 33.87) < 0.01) := by
  sorry

end car_speed_problem_l575_57556


namespace count_equations_l575_57567

-- Define a function to check if an expression is an equation
def is_equation (expr : String) : Bool :=
  match expr with
  | "5 + 3 = 8" => false
  | "a = 0" => true
  | "y^2 - 2y" => false
  | "x - 3 = 8" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["5 + 3 = 8", "a = 0", "y^2 - 2y", "x - 3 = 8"]

-- Theorem to prove
theorem count_equations :
  (expressions.filter is_equation).length = 2 := by
  sorry

end count_equations_l575_57567


namespace translation_of_complex_plane_l575_57523

theorem translation_of_complex_plane (t : ℂ → ℂ) :
  (t (1 + 3*I) = 4 - 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  (t (2 - I) = 5 - 6*I) := by
sorry

end translation_of_complex_plane_l575_57523


namespace fraction_value_l575_57593

theorem fraction_value : (0.5 ^ 4) / (0.05 ^ 2.5) = 559.06 := by sorry

end fraction_value_l575_57593


namespace system_solution_l575_57586

theorem system_solution (x y : ℝ) : 
  (x - y = 2 ∧ 3 * x + y = 4) ↔ (x = 1.5 ∧ y = -0.5) := by
sorry

end system_solution_l575_57586


namespace min_value_expression_l575_57569

theorem min_value_expression (x y : ℝ) :
  x^2 - 6*x*Real.sin y - 9*(Real.cos y)^2 ≥ -9 ∧
  ∃ (x y : ℝ), x^2 - 6*x*Real.sin y - 9*(Real.cos y)^2 = -9 := by
sorry

end min_value_expression_l575_57569


namespace min_value_expression_l575_57504

/-- Given positive real numbers m and n, vectors a and b, where a is parallel to b,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (a b : Fin 2 → ℝ) 
  (ha : a = ![m, 1]) 
  (hb : b = ![1-n, 1]) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y ≥ 1/m + 2/n) → 
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l575_57504


namespace hawks_first_half_score_l575_57596

/-- Represents the score of a basketball team in a game with two halves -/
structure TeamScore where
  first_half : ℕ
  second_half : ℕ

/-- Represents the final scores of two teams in a basketball game -/
structure GameScore where
  eagles : TeamScore
  hawks : TeamScore

/-- The conditions of the basketball game -/
def game_conditions (game : GameScore) : Prop :=
  let eagles_total := game.eagles.first_half + game.eagles.second_half
  let hawks_total := game.hawks.first_half + game.hawks.second_half
  eagles_total + hawks_total = 120 ∧
  eagles_total = hawks_total + 16 ∧
  game.hawks.second_half = game.hawks.first_half + 8

theorem hawks_first_half_score (game : GameScore) :
  game_conditions game → game.hawks.first_half = 22 := by
  sorry

#check hawks_first_half_score

end hawks_first_half_score_l575_57596


namespace sum_of_divisors_of_29_l575_57598

theorem sum_of_divisors_of_29 (h : Nat.Prime 29) : 
  (Finset.filter (· ∣ 29) (Finset.range 30)).sum id = 30 := by
  sorry

end sum_of_divisors_of_29_l575_57598


namespace circle_area_circumference_ratio_l575_57563

theorem circle_area_circumference_ratio (r₁ r₂ : ℝ) (h : π * r₁^2 / (π * r₂^2) = 49 / 64) :
  (2 * π * r₁) / (2 * π * r₂) = 7 / 8 := by
sorry

end circle_area_circumference_ratio_l575_57563


namespace max_sum_of_four_numbers_l575_57526

theorem max_sum_of_four_numbers (a b c d : ℕ) : 
  a < b → b < c → c < d → 
  (b + d) + (c + d) + (a + b + c) + (a + b + d) = 2017 →
  a + b + c + d ≤ 1006 := by
sorry

end max_sum_of_four_numbers_l575_57526


namespace lake_circumference_diameter_ratio_l575_57531

/-- For a circular lake with given diameter and circumference, 
    prove that the ratio of circumference to diameter is 3.14 -/
theorem lake_circumference_diameter_ratio :
  ∀ (diameter circumference : ℝ),
    diameter = 100 →
    circumference = 314 →
    circumference / diameter = 3.14 := by
  sorry

end lake_circumference_diameter_ratio_l575_57531


namespace simplify_and_rationalize_l575_57577

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 3) * (Real.sqrt 4 / Real.sqrt 5) * (Real.sqrt 6 / Real.sqrt 7) = 4 * Real.sqrt 35 / 35 := by
  sorry

end simplify_and_rationalize_l575_57577


namespace parentheses_removal_l575_57519

theorem parentheses_removal (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end parentheses_removal_l575_57519


namespace base6_subtraction_addition_l575_57566

-- Define a function to convert base-6 to decimal
def base6ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert decimal to base-6
def decimalToBase6 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base6_subtraction_addition :
  decimalToBase6 (base6ToDecimal 655 - base6ToDecimal 222 + base6ToDecimal 111) = 544 := by
  sorry

end base6_subtraction_addition_l575_57566


namespace valerie_light_bulb_purchase_l575_57500

/-- Valerie's light bulb purchase problem -/
theorem valerie_light_bulb_purchase (small_bulb_cost large_bulb_cost small_bulb_count large_bulb_count leftover_money : ℕ) :
  small_bulb_cost = 8 →
  large_bulb_cost = 12 →
  small_bulb_count = 3 →
  large_bulb_count = 1 →
  leftover_money = 24 →
  small_bulb_cost * small_bulb_count + large_bulb_cost * large_bulb_count + leftover_money = 60 :=
by sorry

end valerie_light_bulb_purchase_l575_57500


namespace square_root_of_ten_thousand_l575_57549

theorem square_root_of_ten_thousand : Real.sqrt 10000 = 100 := by
  sorry

end square_root_of_ten_thousand_l575_57549


namespace probability_all_white_balls_l575_57511

def white_balls : ℕ := 7
def black_balls : ℕ := 7
def total_balls : ℕ := white_balls + black_balls
def num_draws : ℕ := 6

theorem probability_all_white_balls :
  (white_balls : ℚ) / total_balls ^ num_draws = 1 / 64 := by sorry

end probability_all_white_balls_l575_57511
