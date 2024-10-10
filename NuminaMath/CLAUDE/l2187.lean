import Mathlib

namespace evaluate_expression_l2187_218794

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 3/4) (hz : z = 8) :
  x^2 * y^3 * z = 27/128 := by sorry

end evaluate_expression_l2187_218794


namespace gcd_111_1850_l2187_218780

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by
  sorry

end gcd_111_1850_l2187_218780


namespace chord_length_hyperbola_equation_l2187_218708

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line with slope 1 passing through the right focus
def line (x y : ℝ) : Prop := y = x - Real.sqrt 3

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Theorem 1: Length of chord AB
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8/5 :=
sorry

-- Theorem 2: Equation of the hyperbola
theorem hyperbola_equation :
  ∃ (a b : ℝ), 
    hyperbola 3 4 (-3) (2 * Real.sqrt 3) ∧
    ∀ (x y : ℝ), hyperbola a b x y ↔ 4*x^2/9 - y^2/4 = 1 :=
sorry

end chord_length_hyperbola_equation_l2187_218708


namespace regular_polygon_center_containment_l2187_218739

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ
  center : ℝ × ℝ

/-- M1 is situated inside M2 -/
def isInside (M1 M2 : RegularPolygon n) : Prop :=
  sorry

/-- The center of a polygon -/
def centerOf (M : RegularPolygon n) : ℝ × ℝ :=
  M.center

/-- A point is contained in a polygon -/
def contains (M : RegularPolygon n) (p : ℝ × ℝ) : Prop :=
  sorry

theorem regular_polygon_center_containment (n : ℕ) (M1 M2 : RegularPolygon n) 
  (h1 : M1.sideLength = a)
  (h2 : M2.sideLength = 2 * a)
  (h3 : isInside M1 M2)
  : contains M1 (centerOf M2) :=
sorry

end regular_polygon_center_containment_l2187_218739


namespace geometric_sequence_sum_l2187_218765

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- a_n is a geometric sequence
  a 1 + a 2 = 3 →                           -- a_1 + a_2 = 3
  a 2 + a 3 = 6 →                           -- a_2 + a_3 = 6
  a 4 + a 5 = 24 :=                         -- a_4 + a_5 = 24
by sorry

end geometric_sequence_sum_l2187_218765


namespace max_sum_of_diagonals_l2187_218766

/-- A rhombus with side length 5 and diagonals d1 and d2 where d1 ≤ 6 and d2 ≥ 6 -/
structure Rhombus where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  side_is_5 : side_length = 5
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The maximum sum of diagonals in the given rhombus is 14 -/
theorem max_sum_of_diagonals (r : Rhombus) : (r.d1 + r.d2 ≤ 14) ∧ (∃ (s : Rhombus), s.d1 + s.d2 = 14) := by
  sorry


end max_sum_of_diagonals_l2187_218766


namespace f_inequality_l2187_218785

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * exp x - k * x^2

theorem f_inequality (k : ℝ) (h1 : k > 1/2) 
  (h2 : ∀ x > 0, f k x + (log (2*k))^2 + 2*k * log (exp 1 / (2*k)) > 0) :
  f k (k - 1 + log 2) < f k k := by
sorry

end f_inequality_l2187_218785


namespace obtuse_triangle_from_altitudes_l2187_218746

theorem obtuse_triangle_from_altitudes (h₁ h₂ h₃ : ℝ) 
  (h_pos : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0) 
  (h_ineq : 1/h₁ + 1/h₂ > 1/h₃ ∧ 1/h₂ + 1/h₃ > 1/h₁ ∧ 1/h₃ + 1/h₁ > 1/h₂) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    h₁ = (2 * (a * b * c / (a + b + c))) / a ∧
    h₂ = (2 * (a * b * c / (a + b + c))) / b ∧
    h₃ = (2 * (a * b * c / (a + b + c))) / c ∧
    a^2 + b^2 < c^2 :=
sorry

end obtuse_triangle_from_altitudes_l2187_218746


namespace certain_number_sum_l2187_218747

theorem certain_number_sum (x : ℤ) : 47 + x = 30 → x = -17 := by
  sorry

end certain_number_sum_l2187_218747


namespace solution_existence_l2187_218729

theorem solution_existence (k : ℝ) : 
  (∃ x ∈ Set.Icc 0 2, k * 9^x - k * 3^(x + 1) + 6 * (k - 5) = 0) ↔ k ∈ Set.Icc (1/2) 8 := by
  sorry

end solution_existence_l2187_218729


namespace find_y_value_l2187_218702

theorem find_y_value (x : ℝ) (y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 20) : y = 80 := by
  sorry

end find_y_value_l2187_218702


namespace students_not_enrolled_l2187_218752

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 6 := by
sorry

end students_not_enrolled_l2187_218752


namespace wasted_meat_pounds_l2187_218709

def minimum_wage : ℝ := 8
def fruit_veg_cost_per_pound : ℝ := 4
def fruit_veg_wasted : ℝ := 15
def bread_cost_per_pound : ℝ := 1.5
def bread_wasted : ℝ := 60
def janitor_normal_wage : ℝ := 10
def janitor_hours : ℝ := 10
def meat_cost_per_pound : ℝ := 5
def james_work_hours : ℝ := 50

def total_cost : ℝ := james_work_hours * minimum_wage

def fruit_veg_cost : ℝ := fruit_veg_cost_per_pound * fruit_veg_wasted
def bread_cost : ℝ := bread_cost_per_pound * bread_wasted
def janitor_cost : ℝ := janitor_normal_wage * 1.5 * janitor_hours

def known_costs : ℝ := fruit_veg_cost + bread_cost + janitor_cost
def meat_cost : ℝ := total_cost - known_costs

theorem wasted_meat_pounds : meat_cost / meat_cost_per_pound = 20 := by
  sorry

end wasted_meat_pounds_l2187_218709


namespace nancy_homework_time_l2187_218745

/-- The time required to finish all problems -/
def time_to_finish (math_problems : Float) (spelling_problems : Float) (problems_per_hour : Float) : Float :=
  (math_problems + spelling_problems) / problems_per_hour

/-- Proof that Nancy will take 4.0 hours to finish all problems -/
theorem nancy_homework_time : 
  time_to_finish 17.0 15.0 8.0 = 4.0 := by
  sorry

end nancy_homework_time_l2187_218745


namespace square_roots_of_sqrt_256_is_correct_l2187_218774

-- Define the set of square roots of √256
def square_roots_of_sqrt_256 : Set ℝ :=
  {x : ℝ | x ^ 2 = Real.sqrt 256}

-- Theorem statement
theorem square_roots_of_sqrt_256_is_correct :
  square_roots_of_sqrt_256 = {-4, 4} := by
sorry

end square_roots_of_sqrt_256_is_correct_l2187_218774


namespace sum_of_roots_zero_l2187_218799

theorem sum_of_roots_zero (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 6*x^2 - x + 6
  ∃ a b c d : ℝ, (∀ x, f x = (x^2 + a*x + b) * (x^2 + c*x + d)) →
  (a + b + c + d = 0) :=
by sorry

end sum_of_roots_zero_l2187_218799


namespace escalator_length_l2187_218744

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time length : ℝ),
  escalator_speed = 12 →
  person_speed = 3 →
  time = 10 →
  length = (escalator_speed + person_speed) * time →
  length = 150 := by
sorry

end escalator_length_l2187_218744


namespace arithmetic_mean_of_numbers_l2187_218776

def numbers : List ℝ := [15, 23, 37, 45]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 30 := by sorry

end arithmetic_mean_of_numbers_l2187_218776


namespace shirts_bought_l2187_218786

/-- Given John's initial and final shirt counts, prove the number of shirts bought. -/
theorem shirts_bought (initial_shirts final_shirts : ℕ) 
  (h1 : initial_shirts = 12)
  (h2 : final_shirts = 16)
  : final_shirts - initial_shirts = 4 := by
  sorry

end shirts_bought_l2187_218786


namespace quadratic_symmetry_inequality_l2187_218713

/-- Given real numbers a, b, c, and a quadratic function f(x) = ax^2 + bx + c
    that is symmetric about x = 1, prove that f(1-a) < f(1-2a) < f(1) is impossible. -/
theorem quadratic_symmetry_inequality (a b c : ℝ) 
    (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 + b * x + c) 
    (h_sym : ∀ x, f x = f (2 - x)) : 
  ¬(f (1 - a) < f (1 - 2*a) ∧ f (1 - 2*a) < f 1) := by
  sorry

end quadratic_symmetry_inequality_l2187_218713


namespace product_OA_OC_constant_C_trajectory_l2187_218772

-- Define the rhombus ABCD
structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_rhombus : side_length = 4)
  (OB_length : ℝ)
  (OD_length : ℝ)
  (OB_OD_equal : OB_length = 6 ∧ OD_length = 6)

-- Define the function for |OA| * |OC|
def product_OA_OC (r : Rhombus) : ℝ := sorry

-- Define the function for the coordinates of C
def C_coordinates (r : Rhombus) (A_x A_y : ℝ) : ℝ × ℝ := sorry

-- Theorem 1: |OA| * |OC| is constant
theorem product_OA_OC_constant (r : Rhombus) : 
  product_OA_OC r = 20 := by sorry

-- Theorem 2: Trajectory of C
theorem C_trajectory (r : Rhombus) (A_x A_y : ℝ) 
  (h1 : (A_x - 2)^2 + A_y^2 = 4) (h2 : 2 ≤ A_x ∧ A_x ≤ 4) :
  ∃ (y : ℝ), C_coordinates r A_x A_y = (5, y) ∧ -5 ≤ y ∧ y ≤ 5 := by sorry

end product_OA_OC_constant_C_trajectory_l2187_218772


namespace arctan_tan_difference_l2187_218728

/-- Prove that arctan(tan 75° - 2 tan 30°) = 75° --/
theorem arctan_tan_difference (π : Real) : 
  let deg_to_rad : Real → Real := (· * π / 180)
  Real.arctan (Real.tan (deg_to_rad 75) - 2 * Real.tan (deg_to_rad 30)) = deg_to_rad 75 := by
  sorry

end arctan_tan_difference_l2187_218728


namespace no_perfect_squares_in_sequence_l2187_218712

/-- Represents a number in the sequence -/
def SequenceNumber (n : ℕ) : ℕ := 20142015 + n * 10^6

/-- The sum of digits for any number in the sequence -/
def DigitSum : ℕ := 15

/-- A number is a candidate for being a perfect square if its digit sum is 0, 1, 4, 7, or 9 mod 9 -/
def IsPerfectSquareCandidate (n : ℕ) : Prop :=
  n % 9 = 0 ∨ n % 9 = 1 ∨ n % 9 = 4 ∨ n % 9 = 7 ∨ n % 9 = 9

theorem no_perfect_squares_in_sequence :
  ∀ n : ℕ, ¬ ∃ m : ℕ, (SequenceNumber n) = m^2 :=
sorry

end no_perfect_squares_in_sequence_l2187_218712


namespace tim_cantaloupes_count_l2187_218789

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

theorem tim_cantaloupes_count : tim_cantaloupes = 44 := by
  sorry

end tim_cantaloupes_count_l2187_218789


namespace train_length_calculation_train_length_proof_l2187_218761

theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) 
  (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed - jogger_speed
  let distance_traveled := relative_speed * passing_time
  let train_length := distance_traveled - initial_distance
  train_length

theorem train_length_proof :
  train_length_calculation 2.5 12.5 240 36 = 120 := by
  sorry

end train_length_calculation_train_length_proof_l2187_218761


namespace arithmetic_sequence_sum_l2187_218731

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  (a 1)^2 - 10*(a 1) + 16 = 0 →
  (a 2015)^2 - 10*(a 2015) + 16 = 0 →
  a 2 + a 1008 + a 2014 = 15 := by
  sorry

end arithmetic_sequence_sum_l2187_218731


namespace system_solution_l2187_218751

theorem system_solution (x y : ℝ) : 
  (4 * (x - y) = 8 - 3 * y) ∧ 
  (x / 2 + y / 3 = 1) ↔ 
  (x = 2 ∧ y = 0) := by
sorry

end system_solution_l2187_218751


namespace at_least_one_fraction_less_than_two_l2187_218791

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end at_least_one_fraction_less_than_two_l2187_218791


namespace westward_plane_speed_l2187_218767

/-- Given two planes traveling in opposite directions, this theorem calculates
    the speed of the westward-traveling plane. -/
theorem westward_plane_speed
  (east_speed : ℝ)
  (time : ℝ)
  (total_distance : ℝ)
  (h1 : east_speed = 325)
  (h2 : time = 3.5)
  (h3 : total_distance = 2100)
  : ∃ (west_speed : ℝ),
    west_speed = 275 ∧
    total_distance = (east_speed + west_speed) * time :=
by sorry

end westward_plane_speed_l2187_218767


namespace angles_around_point_l2187_218788

theorem angles_around_point (a b c : ℝ) : 
  a + b + c = 360 →  -- sum of angles around a point is 360°
  c = 120 →          -- one angle is 120°
  a = b →            -- the other two angles are equal
  a = 120 :=         -- prove that each of the equal angles is 120°
by sorry

end angles_around_point_l2187_218788


namespace flag_combinations_l2187_218750

def num_colors : ℕ := 2
def num_stripes : ℕ := 3

theorem flag_combinations : (num_colors ^ num_stripes : ℕ) = 8 := by
  sorry

end flag_combinations_l2187_218750


namespace greatest_difference_l2187_218706

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem greatest_difference (x y : ℕ) 
  (hx_lower : 6 < x) (hx_upper : x < 10)
  (hy_lower : 10 < y) (hy_upper : y < 17)
  (hx_prime : is_prime x)
  (hy_square : is_perfect_square y) :
  (∀ x' y' : ℕ, 
    6 < x' → x' < 10 → 10 < y' → y' < 17 → 
    is_prime x' → is_perfect_square y' → 
    y' - x' ≤ y - x) ∧
  y - x = 9 :=
sorry

end greatest_difference_l2187_218706


namespace cube_volume_from_surface_area_l2187_218742

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 486 → volume = 729 := by
  sorry

end cube_volume_from_surface_area_l2187_218742


namespace group_size_l2187_218724

theorem group_size (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 35)
  (h2 : norway = 23)
  (h3 : both = 31)
  (h4 : neither = 33) :
  iceland + norway - both + neither = 60 := by
  sorry

end group_size_l2187_218724


namespace max_diagonal_value_l2187_218726

/-- Represents a table with n rows and columns where the first column contains 1s
    and each row k is an arithmetic sequence with common difference k -/
def specialTable (n : ℕ) : ℕ → ℕ → ℕ :=
  fun k j => if j = 1 then 1 else 1 + (j - 1) * k

/-- The value on the diagonal from bottom-left to top-right at row k -/
def diagonalValue (n : ℕ) (k : ℕ) : ℕ :=
  specialTable n k (n - k + 1)

theorem max_diagonal_value :
  ∃ k, k ≤ 100 ∧ diagonalValue 100 k = 2501 ∧
  ∀ m, m ≤ 100 → diagonalValue 100 m ≤ 2501 := by
  sorry

end max_diagonal_value_l2187_218726


namespace pencil_eraser_combinations_l2187_218719

/-- The number of possible combinations when choosing one item from each of two sets -/
def combinations (set1 : ℕ) (set2 : ℕ) : ℕ := set1 * set2

/-- Theorem: The number of combinations when choosing one pencil from 2 types
    and one eraser from 3 types is equal to 6 -/
theorem pencil_eraser_combinations :
  combinations 2 3 = 6 := by
  sorry

end pencil_eraser_combinations_l2187_218719


namespace problem_solution_l2187_218792

theorem problem_solution (A B C D : ℕ+) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 72 →
  C * D = 72 →
  A - B = C * D →
  A = 3 := by
sorry

end problem_solution_l2187_218792


namespace bob_water_usage_percentage_l2187_218707

/-- Represents a farmer with their crop acreages -/
structure Farmer where
  corn_acres : ℝ
  cotton_acres : ℝ
  bean_acres : ℝ

/-- Represents water requirements for different crops -/
structure WaterRequirements where
  corn_gallons_per_acre : ℝ
  cotton_gallons_per_acre : ℝ
  bean_gallons_per_acre : ℝ

/-- Calculates the total water usage for a farmer -/
def water_usage (f : Farmer) (w : WaterRequirements) : ℝ :=
  f.corn_acres * w.corn_gallons_per_acre +
  f.cotton_acres * w.cotton_gallons_per_acre +
  f.bean_acres * w.bean_gallons_per_acre

/-- Theorem: The percentage of total water used by Farmer Bob is 36% -/
theorem bob_water_usage_percentage
  (bob : Farmer)
  (brenda : Farmer)
  (bernie : Farmer)
  (water_req : WaterRequirements)
  (h1 : bob.corn_acres = 3 ∧ bob.cotton_acres = 9 ∧ bob.bean_acres = 12)
  (h2 : brenda.corn_acres = 6 ∧ brenda.cotton_acres = 7 ∧ brenda.bean_acres = 14)
  (h3 : bernie.corn_acres = 2 ∧ bernie.cotton_acres = 12 ∧ bernie.bean_acres = 0)
  (h4 : water_req.corn_gallons_per_acre = 20)
  (h5 : water_req.cotton_gallons_per_acre = 80)
  (h6 : water_req.bean_gallons_per_acre = 2 * water_req.corn_gallons_per_acre) :
  (water_usage bob water_req) / (water_usage bob water_req + water_usage brenda water_req + water_usage bernie water_req) = 0.36 := by
  sorry


end bob_water_usage_percentage_l2187_218707


namespace triangle_angle_calculation_l2187_218721

/-- Given a triangle ABC where:
  * The side opposite to angle A is 2
  * The side opposite to angle B is √2
  * Angle A measures 45°
Prove that angle B measures 30° -/
theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = Real.sqrt 2 →
  A = π / 4 →
  B = π / 6 :=
by sorry

end triangle_angle_calculation_l2187_218721


namespace complex_fraction_calculation_l2187_218711

theorem complex_fraction_calculation : 
  (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -10/3 := by
  sorry

end complex_fraction_calculation_l2187_218711


namespace at_most_two_solutions_l2187_218759

theorem at_most_two_solutions (m : ℕ) : 
  ∃ (a₁ a₂ : ℤ), ∀ (a : ℤ), 
    (⌊(a : ℝ) - Real.sqrt (a : ℝ)⌋ = m) → (a = a₁ ∨ a = a₂) :=
sorry

end at_most_two_solutions_l2187_218759


namespace train_passing_pole_time_l2187_218773

/-- Proves that a train 150 meters long running at 90 km/hr takes 6 seconds to pass a pole. -/
theorem train_passing_pole_time (train_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 150 ∧ train_speed_kmh = 90 →
  (train_length / (train_speed_kmh * (1000 / 3600))) = 6 := by
  sorry

end train_passing_pole_time_l2187_218773


namespace floor_painting_cost_l2187_218723

theorem floor_painting_cost 
  (length : ℝ) 
  (paint_rate : ℝ) 
  (length_ratio : ℝ) : 
  length = 12.24744871391589 →
  paint_rate = 2 →
  length_ratio = 3 →
  (length * (length / length_ratio)) * paint_rate = 100 := by
  sorry

end floor_painting_cost_l2187_218723


namespace max_sum_with_condition_l2187_218790

/-- Given positive integers a and b not exceeding 100 satisfying the condition,
    the maximum value of a + b is 78. -/
theorem max_sum_with_condition (a b : ℕ) : 
  0 < a ∧ 0 < b ∧ a ≤ 100 ∧ b ≤ 100 →
  a * b = (Nat.lcm a b / Nat.gcd a b) ^ 2 →
  ∀ (x y : ℕ), 0 < x ∧ 0 < y ∧ x ≤ 100 ∧ y ≤ 100 →
    x * y = (Nat.lcm x y / Nat.gcd x y) ^ 2 →
    a + b ≤ 78 ∧ (∃ (a' b' : ℕ), a' + b' = 78 ∧ 
      0 < a' ∧ 0 < b' ∧ a' ≤ 100 ∧ b' ≤ 100 ∧
      a' * b' = (Nat.lcm a' b' / Nat.gcd a' b') ^ 2) :=
by sorry

end max_sum_with_condition_l2187_218790


namespace perpendicular_bisectors_intersection_l2187_218763

-- Define a triangle as a structure with three points in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem perpendicular_bisectors_intersection (t : Triangle) :
  ∃! O : ℝ × ℝ, distance O t.A = distance O t.B ∧ distance O t.A = distance O t.C := by
  sorry

end perpendicular_bisectors_intersection_l2187_218763


namespace circles_are_tangent_l2187_218749

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let x1 := -c1.b / 2
  let y1 := -c1.c / 2
  let r1 := Real.sqrt (x1^2 + y1^2 - c1.e)
  let x2 := -c2.b / 2
  let y2 := -c2.c / 2
  let r2 := Real.sqrt (x2^2 + y2^2 - c2.e)
  let d := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)
  d = r1 + r2 ∨ d = abs (r1 - r2)

theorem circles_are_tangent : 
  let c1 : Circle := ⟨1, -6, 4, 1, 12⟩
  let c2 : Circle := ⟨1, -14, -2, 1, 14⟩
  are_tangent c1 c2 := by
  sorry

end circles_are_tangent_l2187_218749


namespace sheet_reduction_percentage_l2187_218783

def original_sheets : ℕ := 20
def original_lines_per_sheet : ℕ := 55
def original_chars_per_line : ℕ := 65

def retyped_lines_per_sheet : ℕ := 65
def retyped_chars_per_line : ℕ := 70

def total_chars : ℕ := original_sheets * original_lines_per_sheet * original_chars_per_line

def chars_per_retyped_sheet : ℕ := retyped_lines_per_sheet * retyped_chars_per_line

def retyped_sheets : ℕ := (total_chars + chars_per_retyped_sheet - 1) / chars_per_retyped_sheet

theorem sheet_reduction_percentage : 
  (original_sheets - retyped_sheets) * 100 / original_sheets = 20 := by
  sorry

end sheet_reduction_percentage_l2187_218783


namespace ball_reaches_top_left_pocket_l2187_218720

/-- Represents a point on the billiard table or its reflections -/
structure TablePoint where
  x : Int
  y : Int

/-- Represents the dimensions of the billiard table -/
structure TableDimensions where
  width : Nat
  height : Nat

/-- Checks if a point is a top-left pocket in the reflected grid -/
def isTopLeftPocket (p : TablePoint) (dim : TableDimensions) : Prop :=
  ∃ (m n : Int), p.x = dim.width * m ∧ p.y = dim.height * n ∧ m % 2 = 0 ∧ n % 2 = 1

/-- The theorem stating that the ball will reach the top-left pocket -/
theorem ball_reaches_top_left_pocket (dim : TableDimensions) 
  (h_dim : dim.width = 1965 ∧ dim.height = 26) :
  ∃ (p : TablePoint), p.y = p.x ∧ isTopLeftPocket p dim := by
  sorry

end ball_reaches_top_left_pocket_l2187_218720


namespace discretionary_income_ratio_l2187_218748

/-- Represents Jill's financial situation --/
structure JillFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFundPercent : ℝ
  savingsPercent : ℝ
  socializingPercent : ℝ
  giftsAmount : ℝ

/-- The conditions of Jill's finances --/
def jillFinancesConditions (j : JillFinances) : Prop :=
  j.netSalary = 3300 ∧
  j.vacationFundPercent = 0.3 ∧
  j.savingsPercent = 0.2 ∧
  j.socializingPercent = 0.35 ∧
  j.giftsAmount = 99 ∧
  j.giftsAmount = (1 - (j.vacationFundPercent + j.savingsPercent + j.socializingPercent)) * j.discretionaryIncome

/-- The theorem stating the ratio of discretionary income to net salary --/
theorem discretionary_income_ratio (j : JillFinances) 
  (h : jillFinancesConditions j) : 
  j.discretionaryIncome / j.netSalary = 1 / 5 := by
  sorry


end discretionary_income_ratio_l2187_218748


namespace complex_ratio_theorem_l2187_218730

theorem complex_ratio_theorem (a b : ℝ) (z : ℂ) (h1 : z = Complex.mk a b) 
  (h2 : ∃ (k : ℝ), z / Complex.mk 2 1 = Complex.mk 0 k) : b / a = -2 := by
  sorry

end complex_ratio_theorem_l2187_218730


namespace smallest_natural_number_square_cube_seventy_two_satisfies_conditions_smallest_natural_number_is_72_l2187_218781

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_natural_number_square_cube : 
  ∀ x : ℕ, (is_perfect_square (2 * x) ∧ is_perfect_cube (3 * x)) → x ≥ 72 :=
by sorry

theorem seventy_two_satisfies_conditions : 
  is_perfect_square (2 * 72) ∧ is_perfect_cube (3 * 72) :=
by sorry

theorem smallest_natural_number_is_72 : 
  ∃! x : ℕ, x = 72 ∧ 
    (∀ y : ℕ, (is_perfect_square (2 * y) ∧ is_perfect_cube (3 * y)) → y ≥ x) :=
by sorry

end smallest_natural_number_square_cube_seventy_two_satisfies_conditions_smallest_natural_number_is_72_l2187_218781


namespace moores_law_transistor_count_l2187_218770

/-- Moore's law calculation for transistor count --/
theorem moores_law_transistor_count 
  (initial_year : Nat) 
  (final_year : Nat) 
  (initial_transistors : Nat) 
  (doubling_period : Nat) 
  (h1 : initial_year = 1985)
  (h2 : final_year = 2010)
  (h3 : initial_transistors = 300000)
  (h4 : doubling_period = 2) :
  let years_passed := final_year - initial_year
  let doublings := years_passed / doubling_period
  initial_transistors * (2 ^ doublings) = 1228800000 :=
by sorry

end moores_law_transistor_count_l2187_218770


namespace ball_bounce_height_l2187_218782

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (h₁ : h₀ = 1000) (h₂ : r = 1/2) :
  ∃ k : ℕ, k > 0 ∧ h₀ * r^k < 1 ∧ ∀ j : ℕ, 0 < j → j < k → h₀ * r^j ≥ 1 :=
by sorry

end ball_bounce_height_l2187_218782


namespace a_minus_c_value_l2187_218797

theorem a_minus_c_value (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180)
  (h3 : d = 2 * (a - b)) : 
  a - c = -200 := by
sorry

end a_minus_c_value_l2187_218797


namespace train_speed_problem_l2187_218716

/-- Proves that the original speed of a train is 60 km/h given the specified conditions -/
theorem train_speed_problem (delay : Real) (distance : Real) (speed_increase : Real) :
  delay = 0.2 ∧ distance = 60 ∧ speed_increase = 15 →
  ∃ original_speed : Real,
    original_speed > 0 ∧
    distance / original_speed - distance / (original_speed + speed_increase) = delay ∧
    original_speed = 60 := by
  sorry

end train_speed_problem_l2187_218716


namespace adventure_team_probabilities_l2187_218784

def team_size : ℕ := 8
def medical_staff : ℕ := 3
def group_size : ℕ := 4

def probability_one_medical_in_one_group : ℚ := 6/7
def probability_at_least_two_medical_in_group : ℚ := 1/2
def expected_medical_in_group : ℚ := 3/2

theorem adventure_team_probabilities :
  (team_size = 8) →
  (medical_staff = 3) →
  (group_size = 4) →
  (probability_one_medical_in_one_group = 6/7) ∧
  (probability_at_least_two_medical_in_group = 1/2) ∧
  (expected_medical_in_group = 3/2) :=
by sorry

end adventure_team_probabilities_l2187_218784


namespace watermelon_pricing_l2187_218795

/-- Represents the number of watermelons sold by each student in the morning -/
structure MorningSales where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the prices of watermelons -/
structure Prices where
  morning : ℚ
  afternoon : ℚ

/-- Theorem statement for the watermelon pricing problem -/
theorem watermelon_pricing
  (sales : MorningSales)
  (prices : Prices)
  (h1 : prices.morning > prices.afternoon)
  (h2 : prices.afternoon > 0)
  (h3 : sales.first < 10)
  (h4 : sales.second < 16)
  (h5 : sales.third < 26)
  (h6 : prices.morning * sales.first + prices.afternoon * (10 - sales.first) = 42)
  (h7 : prices.morning * sales.second + prices.afternoon * (16 - sales.second) = 42)
  (h8 : prices.morning * sales.third + prices.afternoon * (26 - sales.third) = 42)
  : prices.morning = 4.5 ∧ prices.afternoon = 1.5 := by
  sorry

#check watermelon_pricing

end watermelon_pricing_l2187_218795


namespace exponent_sum_theorem_l2187_218764

theorem exponent_sum_theorem : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end exponent_sum_theorem_l2187_218764


namespace factors_of_2520_l2187_218733

theorem factors_of_2520 : Nat.card (Nat.divisors 2520) = 48 := by
  sorry

end factors_of_2520_l2187_218733


namespace brand_preference_survey_l2187_218704

theorem brand_preference_survey (total : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) (h_total : total = 80) (h_ratio : ratio_x = 3 ∧ ratio_y = 1) : 
  (total * ratio_x) / (ratio_x + ratio_y) = 60 := by
  sorry

end brand_preference_survey_l2187_218704


namespace first_term_of_sequence_l2187_218736

theorem first_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 1) : a 1 = 2 := by
  sorry

end first_term_of_sequence_l2187_218736


namespace not_always_y_equals_a_when_x_zero_l2187_218722

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope
  x_bar : ℝ  -- mean of x
  y_bar : ℝ  -- mean of y

/-- Predicted y value for a given x -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.b * x + model.a

/-- The regression line passes through the point (x_bar, y_bar) -/
axiom passes_through_mean (model : LinearRegression) :
  predict model model.x_bar = model.y_bar

/-- b represents the average change in y for a unit increase in x -/
axiom slope_interpretation (model : LinearRegression) (x₁ x₂ : ℝ) :
  predict model x₂ - predict model x₁ = model.b * (x₂ - x₁)

/-- Sample data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Theorem: It is not necessarily true that y = a when x = 0 in the sample data -/
theorem not_always_y_equals_a_when_x_zero (model : LinearRegression) :
  ∃ (data : DataPoint), data.x = 0 ∧ data.y ≠ model.a :=
sorry

end not_always_y_equals_a_when_x_zero_l2187_218722


namespace pebble_collection_sum_l2187_218725

theorem pebble_collection_sum (n : ℕ) (h : n = 20) : 
  (List.range n).sum = 210 := by
  sorry

end pebble_collection_sum_l2187_218725


namespace watermelon_seeds_count_l2187_218762

/-- The number of seeds in each watermelon -/
def seeds_per_watermelon : ℕ := 345

/-- The number of watermelons -/
def number_of_watermelons : ℕ := 27

/-- The total number of seeds in all watermelons -/
def total_seeds : ℕ := seeds_per_watermelon * number_of_watermelons

theorem watermelon_seeds_count : total_seeds = 9315 := by
  sorry

end watermelon_seeds_count_l2187_218762


namespace total_fruits_eq_137_l2187_218718

/-- The number of fruits picked by George, Amelia, and Olivia --/
def total_fruits (george_oranges amelia_apples olivia_time olivia_rate_time olivia_rate_oranges olivia_rate_apples : ℕ) : ℕ :=
  let george_apples := amelia_apples + 5
  let amelia_oranges := george_oranges - 18
  let olivia_sets := olivia_time / olivia_rate_time
  let olivia_oranges := olivia_sets * olivia_rate_oranges
  let olivia_apples := olivia_sets * olivia_rate_apples
  (george_oranges + george_apples) + (amelia_oranges + amelia_apples) + (olivia_oranges + olivia_apples)

/-- Theorem stating the total number of fruits picked --/
theorem total_fruits_eq_137 :
  total_fruits 45 15 30 5 3 2 = 137 := by
  sorry

end total_fruits_eq_137_l2187_218718


namespace soccer_lineup_combinations_l2187_218755

def team_size : ℕ := 16
def non_goalkeeper : ℕ := 1
def lineup_positions : ℕ := 4

theorem soccer_lineup_combinations :
  (team_size - non_goalkeeper) *
  (team_size - 1) *
  (team_size - 2) *
  (team_size - 3) = 42210 :=
by sorry

end soccer_lineup_combinations_l2187_218755


namespace simplify_linear_expression_l2187_218705

theorem simplify_linear_expression (y : ℝ) : 5*y + 2*y + 7*y = 14*y := by
  sorry

end simplify_linear_expression_l2187_218705


namespace anya_additional_biscuits_l2187_218741

/-- Represents the distribution of biscuits and payments among three sisters. -/
structure BiscuitDistribution where
  total_biscuits : ℕ
  total_payment : ℕ
  anya_payment : ℕ
  berini_payment : ℕ
  carla_payment : ℕ

/-- Calculates the number of additional biscuits Anya would receive if distributed proportionally to payments. -/
def additional_biscuits_for_anya (bd : BiscuitDistribution) : ℕ :=
  let equal_share := bd.total_biscuits / 3
  let proportional_share := (bd.anya_payment * bd.total_biscuits) / bd.total_payment
  proportional_share - equal_share

/-- Theorem stating that Anya would receive 6 more biscuits in a proportional distribution. -/
theorem anya_additional_biscuits :
  ∀ (bd : BiscuitDistribution),
  bd.total_biscuits = 30 ∧
  bd.total_payment = 150 ∧
  bd.anya_payment = 80 ∧
  bd.berini_payment = 50 ∧
  bd.carla_payment = 20 →
  additional_biscuits_for_anya bd = 6 := by
  sorry

end anya_additional_biscuits_l2187_218741


namespace larger_number_is_322_l2187_218769

def is_hcf (a b h : ℕ) : Prop := h ∣ a ∧ h ∣ b ∧ ∀ d : ℕ, d ∣ a → d ∣ b → d ≤ h

def is_lcm (a b l : ℕ) : Prop := a ∣ l ∧ b ∣ l ∧ ∀ m : ℕ, a ∣ m → b ∣ m → l ∣ m

theorem larger_number_is_322 (a b : ℕ) (h : a > 0 ∧ b > 0) :
  is_hcf a b 23 → is_lcm a b (23 * 13 * 14) → max a b = 322 := by
  sorry

end larger_number_is_322_l2187_218769


namespace teenas_speed_l2187_218740

theorem teenas_speed (initial_distance : ℝ) (poes_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 7.5 →
  poes_speed = 40 →
  time = 1.5 →
  final_distance = 15 →
  (initial_distance + poes_speed * time + final_distance) / time = 55 := by
sorry

end teenas_speed_l2187_218740


namespace brochure_printing_problem_l2187_218798

/-- Represents the number of pages printed for the spreads for which the press prints a block of 4 ads -/
def pages_per_ad_block : ℕ := by sorry

theorem brochure_printing_problem :
  let single_page_spreads : ℕ := 20
  let double_page_spreads : ℕ := 2 * single_page_spreads
  let pages_per_brochure : ℕ := 5
  let total_brochures : ℕ := 25
  let total_pages : ℕ := total_brochures * pages_per_brochure
  let pages_from_double_spreads : ℕ := double_page_spreads * 2
  let remaining_pages : ℕ := total_pages - pages_from_double_spreads
  let unused_single_spreads : ℕ := single_page_spreads - remaining_pages
  pages_per_ad_block = unused_single_spreads := by sorry

end brochure_printing_problem_l2187_218798


namespace num_valid_selections_eq_twenty_l2187_218732

/-- Represents a volunteer --/
inductive Volunteer : Type
  | A | B | C | D | E

/-- Represents a role --/
inductive Role : Type
  | Translator | TourGuide | Etiquette | Driver

/-- Predicate to check if a volunteer can perform a given role --/
def can_perform (v : Volunteer) (r : Role) : Prop :=
  match v, r with
  | Volunteer.A, Role.Translator => True
  | Volunteer.A, Role.TourGuide => True
  | Volunteer.B, Role.Translator => True
  | Volunteer.B, Role.TourGuide => True
  | Volunteer.C, _ => True
  | Volunteer.D, _ => True
  | Volunteer.E, _ => True
  | _, _ => False

/-- A valid selection is a function from Role to Volunteer satisfying the constraints --/
def ValidSelection : Type :=
  { f : Role → Volunteer // ∀ r, can_perform (f r) r ∧ ∀ r' ≠ r, f r ≠ f r' }

/-- The number of valid selections --/
def num_valid_selections : ℕ := sorry

/-- Theorem stating that the number of valid selections is 20 --/
theorem num_valid_selections_eq_twenty : num_valid_selections = 20 := by sorry

end num_valid_selections_eq_twenty_l2187_218732


namespace condition_relationship_l2187_218758

theorem condition_relationship (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
sorry

end condition_relationship_l2187_218758


namespace missing_number_proof_l2187_218777

theorem missing_number_proof (x : ℝ) : 
  11 + Real.sqrt (-4 + 6 * 4 / x) = 13 → x = 4 := by
  sorry

end missing_number_proof_l2187_218777


namespace payroll_threshold_proof_l2187_218703

/-- Proves that the payroll threshold is $200,000 given the problem conditions --/
theorem payroll_threshold_proof 
  (total_payroll : ℝ) 
  (tax_paid : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_payroll = 400000)
  (h2 : tax_paid = 400)
  (h3 : tax_rate = 0.002) : 
  ∃ threshold : ℝ, 
    threshold = 200000 ∧ 
    tax_rate * (total_payroll - threshold) = tax_paid :=
by sorry

end payroll_threshold_proof_l2187_218703


namespace complex_number_conditions_l2187_218700

theorem complex_number_conditions (z : ℂ) : 
  (∃ (a : ℝ), a > 0 ∧ (z - 3*I) / (z + I) = -a) ∧ 
  (∃ (b : ℝ), b ≠ 0 ∧ (z - 3) / (z + 1) = b*I) → 
  z = -1 + 2*I :=
by sorry

end complex_number_conditions_l2187_218700


namespace complex_square_plus_self_l2187_218743

theorem complex_square_plus_self : 
  let z : ℂ := 1 + Complex.I
  z^2 + z = 1 + 3 * Complex.I := by sorry

end complex_square_plus_self_l2187_218743


namespace solution_set_characterization_range_of_a_characterization_l2187_218735

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Part 1: Characterize the solution set of f(x) ≥ 4
theorem solution_set_characterization :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 0} :=
sorry

-- Part 2: Characterize the range of a
theorem range_of_a_characterization :
  {a : ℝ | ∀ x > 0, f x + a * x - 1 > 0} = {a : ℝ | a > -5/2} :=
sorry

end solution_set_characterization_range_of_a_characterization_l2187_218735


namespace no_solution_condition_l2187_218787

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (3 - 2*x)/(x - 3) - (m*x - 2)/(3 - x) ≠ -1) ↔ 
  (m = 5/3 ∨ m = 1) :=
by sorry

end no_solution_condition_l2187_218787


namespace smallest_square_containing_rectangles_l2187_218771

/-- The smallest square containing two non-overlapping rectangles -/
theorem smallest_square_containing_rectangles :
  ∀ (w₁ h₁ w₂ h₂ : ℕ),
  w₁ = 3 ∧ h₁ = 5 ∧ w₂ = 4 ∧ h₂ = 6 →
  ∃ (s : ℕ),
    s ≥ w₁ ∧ s ≥ h₁ ∧ s ≥ w₂ ∧ s ≥ h₂ ∧
    s ≥ w₁ + w₂ ∧ s ≥ h₁ ∧ s ≥ h₂ ∧
    (∀ (t : ℕ),
      t ≥ w₁ ∧ t ≥ h₁ ∧ t ≥ w₂ ∧ t ≥ h₂ ∧
      t ≥ w₁ + w₂ ∧ t ≥ h₁ ∧ t ≥ h₂ →
      t ≥ s) ∧
    s^2 = 49 :=
by sorry

end smallest_square_containing_rectangles_l2187_218771


namespace no_integer_solutions_l2187_218757

theorem no_integer_solutions : ¬∃ (x y : ℤ), 3 * x^2 = 16 * y^2 + 8 * y + 5 := by
  sorry

end no_integer_solutions_l2187_218757


namespace max_red_balls_l2187_218701

theorem max_red_balls (n : ℕ) : 
  (∃ y : ℕ, 
    n = 90 + 9 * y ∧
    (89 + 8 * y : ℚ) / (90 + 9 * y) ≥ 92 / 100 ∧
    ∀ m > n, (∃ z : ℕ, m = 90 + 9 * z) → 
      (89 + 8 * z : ℚ) / (90 + 9 * z) < 92 / 100) →
  n = 288 :=
sorry

end max_red_balls_l2187_218701


namespace equation_solution_l2187_218796

theorem equation_solution :
  ∃ x : ℝ, (x^2 + x ≠ 0) ∧ (x^2 - x ≠ 0) ∧
  (4 / (x^2 + x) - 3 / (x^2 - x) = 0) ∧ 
  (x = 7) := by
  sorry

end equation_solution_l2187_218796


namespace quadratic_equation_magnitude_l2187_218738

theorem quadratic_equation_magnitude (z : ℂ) : 
  z^2 - 10*z + 28 = 0 → ∃! m : ℝ, ∃ z : ℂ, z^2 - 10*z + 28 = 0 ∧ Complex.abs z = m :=
by sorry

end quadratic_equation_magnitude_l2187_218738


namespace basketball_game_points_l2187_218793

/-- The total points scored by three players in a basketball game. -/
def total_points (jon_points jack_points tom_points : ℕ) : ℕ :=
  jon_points + jack_points + tom_points

/-- Theorem stating the total points scored by Jon, Jack, and Tom. -/
theorem basketball_game_points : ∃ (jack_points tom_points : ℕ),
  let jon_points := 3
  jack_points = jon_points + 5 ∧
  tom_points = (jon_points + jack_points) - 4 ∧
  total_points jon_points jack_points tom_points = 18 := by
  sorry

end basketball_game_points_l2187_218793


namespace seventh_person_age_l2187_218737

theorem seventh_person_age
  (n : ℕ)
  (initial_people : ℕ)
  (future_average : ℕ)
  (new_average : ℕ)
  (years_passed : ℕ)
  (h1 : initial_people = 6)
  (h2 : future_average = 43)
  (h3 : new_average = 45)
  (h4 : years_passed = 2)
  (h5 : n = initial_people + 1) :
  (n * new_average) - (initial_people * (future_average + years_passed)) = 69 := by
  sorry

end seventh_person_age_l2187_218737


namespace karen_walnuts_l2187_218717

/-- The amount of nuts in cups added to the trail mix -/
def total_nuts : ℝ := 0.5

/-- The amount of almonds in cups added to the trail mix -/
def almonds : ℝ := 0.25

/-- The amount of walnuts in cups added to the trail mix -/
def walnuts : ℝ := total_nuts - almonds

theorem karen_walnuts : walnuts = 0.25 := by
  sorry

end karen_walnuts_l2187_218717


namespace hyperbola_focus_l2187_218727

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the focus coordinates
def focus_coordinates : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ focus_coordinates = (x, y) :=
sorry

end hyperbola_focus_l2187_218727


namespace circle_properties_l2187_218710

noncomputable def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 1

theorem circle_properties :
  ∃ (c : ℝ × ℝ),
    (c.1 = c.2) ∧  -- Center is on the line y = x
    (∀ x y : ℝ, circle_equation x y → (x - c.1)^2 + (y - c.2)^2 = 1) ∧  -- Equation represents a circle
    (circle_equation 1 0) ∧  -- Circle passes through (1,0)
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → ¬(circle_equation x 0)) :=
by sorry

end circle_properties_l2187_218710


namespace missing_number_proof_l2187_218715

theorem missing_number_proof :
  ∃ x : ℝ, 0.72 * 0.43 + x * 0.34 = 0.3504 ∧ abs (x - 0.12) < 0.0001 := by
  sorry

end missing_number_proof_l2187_218715


namespace melanie_caught_ten_l2187_218778

/-- The number of trout Sara caught -/
def sara_trout : ℕ := 5

/-- The factor by which Melanie's catch exceeds Sara's -/
def melanie_factor : ℕ := 2

/-- The number of trout Melanie caught -/
def melanie_trout : ℕ := melanie_factor * sara_trout

theorem melanie_caught_ten : melanie_trout = 10 := by
  sorry

end melanie_caught_ten_l2187_218778


namespace equilateral_triangle_area_decrease_l2187_218756

theorem equilateral_triangle_area_decrease :
  ∀ s : ℝ,
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 81 * Real.sqrt 3 →
  let s' := s - 3
  let new_area := (s'^2 * Real.sqrt 3) / 4
  let area_decrease := 81 * Real.sqrt 3 - new_area
  area_decrease = 24.75 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_area_decrease_l2187_218756


namespace right_triangle_5_12_13_l2187_218775

/-- A right triangle with sides a, b, and c satisfies the Pythagorean theorem --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The triple (5, 12, 13) forms a right triangle --/
theorem right_triangle_5_12_13 :
  is_right_triangle 5 12 13 := by
  sorry

end right_triangle_5_12_13_l2187_218775


namespace a_in_S_l2187_218753

def S : Set ℤ := {n | ∃ x y : ℤ, n = x^2 + 2*y^2}

theorem a_in_S (a : ℤ) (h : 3*a ∈ S) : a ∈ S := by
  sorry

end a_in_S_l2187_218753


namespace inscribed_rectangle_exists_l2187_218714

/-- Represents a right triangle with sides 3, 4, and 5 -/
structure EgyptianTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 3
  hb : b = 4
  hc : c = 5
  right_angle : a^2 + b^2 = c^2

/-- Represents a rectangle inscribed in the Egyptian triangle -/
structure InscribedRectangle (t : EgyptianTriangle) where
  width : ℝ
  height : ℝ
  ratio : width * 3 = height
  fits_in_triangle : width ≤ t.a ∧ width ≤ t.b ∧ height ≤ t.b ∧ height ≤ t.c

/-- The theorem stating the existence and dimensions of the inscribed rectangle -/
theorem inscribed_rectangle_exists (t : EgyptianTriangle) :
  ∃ (r : InscribedRectangle t), r.width = 20/29 ∧ r.height = 60/29 := by
  sorry

end inscribed_rectangle_exists_l2187_218714


namespace least_acute_triangle_side_l2187_218760

/-- A function that checks if three side lengths form an acute triangle -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

/-- The least positive integer A such that an acute triangle with side lengths 5, A, and 8 exists -/
theorem least_acute_triangle_side : ∃ (A : ℕ), 
  (∀ (k : ℕ), k < A → ¬is_acute_triangle 5 (k : ℝ) 8) ∧ 
  is_acute_triangle 5 A 8 ∧
  A = 7 := by
  sorry

end least_acute_triangle_side_l2187_218760


namespace pure_imaginary_value_l2187_218779

def complex_number (a : ℝ) : ℂ := (a^2 + 2*a - 3 : ℝ) + (a^2 - 4*a + 3 : ℝ) * Complex.I

theorem pure_imaginary_value (a : ℝ) :
  (complex_number a).re = 0 ∧ (complex_number a).im ≠ 0 → a = -3 :=
by sorry

end pure_imaginary_value_l2187_218779


namespace water_left_in_cooler_l2187_218768

/-- Calculates the remaining water in a cooler after filling Dixie cups for a meeting --/
theorem water_left_in_cooler 
  (initial_gallons : ℕ) 
  (ounces_per_cup : ℕ) 
  (rows : ℕ) 
  (chairs_per_row : ℕ) 
  (ounces_per_gallon : ℕ) 
  (h1 : initial_gallons = 3)
  (h2 : ounces_per_cup = 6)
  (h3 : rows = 5)
  (h4 : chairs_per_row = 10)
  (h5 : ounces_per_gallon = 128) : 
  initial_gallons * ounces_per_gallon - rows * chairs_per_row * ounces_per_cup = 84 := by
  sorry

end water_left_in_cooler_l2187_218768


namespace tan_beta_value_l2187_218754

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/7) 
  (h2 : Real.tan (α + β) = 1/3) : 
  Real.tan β = 2/11 := by
sorry

end tan_beta_value_l2187_218754


namespace remainder_3_pow_2024_mod_17_l2187_218734

theorem remainder_3_pow_2024_mod_17 : 3^2024 % 17 = 13 := by
  sorry

end remainder_3_pow_2024_mod_17_l2187_218734
