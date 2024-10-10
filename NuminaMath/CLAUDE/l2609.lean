import Mathlib

namespace unique_contributions_exist_l2609_260955

/-- Represents the contributions of five friends to a project -/
structure Contributions where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (c : Contributions) : Prop :=
  c.A = 1.1 * c.B ∧
  c.C = 0.8 * c.A ∧
  c.D = 2 * c.B ∧
  c.E = c.D - 200 ∧
  c.A + c.B + c.C + c.D + c.E = 1500

/-- Theorem stating that there exists a unique set of contributions satisfying the conditions -/
theorem unique_contributions_exist : ∃! c : Contributions, satisfies_conditions c :=
sorry

end unique_contributions_exist_l2609_260955


namespace rectangle_area_l2609_260957

/-- A rectangle with length four times its width and perimeter 200 cm has an area of 1600 cm² --/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 → l * w = 1600 := by
  sorry

end rectangle_area_l2609_260957


namespace ellipse_iff_k_range_l2609_260907

/-- The curve equation -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (4 - k) + y^2 / (k - 1) = 1

/-- Conditions for the curve to be an ellipse -/
def is_ellipse (k : ℝ) : Prop :=
  4 - k > 0 ∧ k - 1 > 0 ∧ 4 - k ≠ k - 1

/-- The range of k for which the curve is an ellipse -/
def k_range (k : ℝ) : Prop :=
  1 < k ∧ k < 4 ∧ k ≠ 5/2

/-- Theorem: The curve is an ellipse if and only if k is in the specified range -/
theorem ellipse_iff_k_range (k : ℝ) :
  is_ellipse k ↔ k_range k :=
sorry

end ellipse_iff_k_range_l2609_260907


namespace f_derivative_at_two_l2609_260923

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_two
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : deriv (f a b) 1 = 0) :
  deriv (f a b) 2 = -1/2 := by
sorry

end f_derivative_at_two_l2609_260923


namespace inequality_proof_l2609_260932

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^4 ≥ x^2*y^2 + y^2*z^2 + z^2*x^2 ∧ 
  x^2*y^2 + y^2*z^2 + z^2*x^2 ≥ x*y*z*(x+y+z) := by
  sorry

end inequality_proof_l2609_260932


namespace equation_solution_and_difference_l2609_260965

theorem equation_solution_and_difference :
  (∃ x : ℚ, 11 * x + 4 = 7) ∧
  (let x : ℚ := 3 / 11; 11 * x + 4 = 7) ∧
  (12 / 11 - 3 / 11 = 9 / 11) := by
  sorry

end equation_solution_and_difference_l2609_260965


namespace cricket_run_rate_theorem_l2609_260948

/-- Represents a cricket game with given parameters -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let first_part_runs := game.first_part_run_rate * game.first_part_overs
  let remaining_runs := game.target_runs - first_part_runs
  remaining_runs / remaining_overs

/-- The main theorem stating the required run rate for the given game parameters -/
theorem cricket_run_rate_theorem (game : CricketGame) 
    (h_total_overs : game.total_overs = 50)
    (h_first_part_overs : game.first_part_overs = 10)
    (h_first_part_run_rate : game.first_part_run_rate = 3.2)
    (h_target_runs : game.target_runs = 242) :
    required_run_rate game = 5.25 := by
  sorry

#eval required_run_rate {
  total_overs := 50,
  first_part_overs := 10,
  first_part_run_rate := 3.2,
  target_runs := 242
}

end cricket_run_rate_theorem_l2609_260948


namespace boat_upstream_distance_l2609_260935

/-- Proves that given a boat with speed 18 kmph in still water and a stream with speed 6 kmph,
    if the boat can cover 48 km downstream or a certain distance upstream in the same time,
    then the distance the boat can cover upstream is 24 km. -/
theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) :
  boat_speed = 18 →
  stream_speed = 6 →
  downstream_distance = 48 →
  (downstream_distance / (boat_speed + stream_speed) = 
   (boat_speed - stream_speed) * (downstream_distance / (boat_speed + stream_speed))) →
  (boat_speed - stream_speed) * (downstream_distance / (boat_speed + stream_speed)) = 24 :=
by sorry

end boat_upstream_distance_l2609_260935


namespace geometric_sequence_condition_l2609_260904

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the property of being a positive sequence
def IsPositive (a : Sequence) : Prop :=
  ∀ n : ℕ, a n > 0

-- Define the recurrence relation
def SatisfiesRecurrence (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 2) = a n * a (n + 1)

-- Define the property of being a geometric sequence
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_condition (a : Sequence) 
  (h_pos : IsPositive a) (h_rec : SatisfiesRecurrence a) :
  IsGeometric a ↔ a 1 = 1 ∧ a 2 = 1 :=
sorry

end geometric_sequence_condition_l2609_260904


namespace sequence_3078th_term_l2609_260973

/-- Calculates the sum of cubes of digits of a natural number -/
def sumOfCubesOfDigits (n : ℕ) : ℕ := sorry

/-- Generates the next term in the sequence -/
def nextTerm (n : ℕ) : ℕ := sumOfCubesOfDigits n

/-- Generates the nth term of the sequence starting with the given initial term -/
def nthTerm (initial : ℕ) (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem sequence_3078th_term (initial : ℕ) (h : initial = 3078) : 
  nthTerm initial 3078 = 153 := by sorry

end sequence_3078th_term_l2609_260973


namespace lines_intersect_at_one_point_l2609_260971

-- Define the basic geometric objects
variable (A B C D E F P Q M O : Point)

-- Define the convex quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the extension relationships
def lies_on_extension (P X Y Z : Point) : Prop := sorry

-- Define the midpoint relationship
def is_midpoint (M X Y : Point) : Prop := sorry

-- Define when a point lies on a line
def point_on_line (P X Y : Point) : Prop := sorry

-- Main theorem
theorem lines_intersect_at_one_point 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_E_ext : lies_on_extension E A B B)
  (h_F_ext : lies_on_extension F C D D)
  (h_M_mid_AD : is_midpoint M A D)
  (h_P_on_BE : point_on_line P B E)
  (h_Q_on_DF : point_on_line Q D F)
  (h_M_mid_PQ : is_midpoint M P Q) :
  ∃ O, point_on_line O A B ∧ point_on_line O C D ∧ point_on_line O P Q :=
sorry

end lines_intersect_at_one_point_l2609_260971


namespace square_sum_geq_product_sum_l2609_260972

theorem square_sum_geq_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end square_sum_geq_product_sum_l2609_260972


namespace quadratic_coefficient_positive_l2609_260924

theorem quadratic_coefficient_positive (a c : ℝ) (h₁ : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2*a*x + c
  f (-1) = 1 ∧ f (-5) = 5 → a > 0 := by
  sorry

end quadratic_coefficient_positive_l2609_260924


namespace positive_integer_M_satisfying_equation_l2609_260962

theorem positive_integer_M_satisfying_equation : ∃ M : ℕ+, (12^2 * 60^2 : ℕ) = 30^2 * M^2 ∧ M = 12 := by
  sorry

end positive_integer_M_satisfying_equation_l2609_260962


namespace complex_magnitude_two_thirds_minus_four_fifths_i_l2609_260931

theorem complex_magnitude_two_thirds_minus_four_fifths_i :
  Complex.abs (⟨2/3, -4/5⟩ : ℂ) = Real.sqrt 244 / 15 := by
  sorry

end complex_magnitude_two_thirds_minus_four_fifths_i_l2609_260931


namespace triangle_sine_sum_inequality_l2609_260974

theorem triangle_sine_sum_inequality (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_sine_sum_inequality_l2609_260974


namespace not_equivalent_polar_point_l2609_260917

def is_equivalent_polar (r : ℝ) (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem not_equivalent_polar_point :
  ¬ is_equivalent_polar 2 (π/6) (11*π/6) := by
  sorry

end not_equivalent_polar_point_l2609_260917


namespace resource_sum_theorem_l2609_260958

/-- Converts a base 6 number to base 10 -/
def base6_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The amount of mineral X in base 6 -/
def mineral_x : List Nat := [2, 3, 4, 1]

/-- The amount of mineral Y in base 6 -/
def mineral_y : List Nat := [4, 1, 2, 3]

/-- The amount of water in base 6 -/
def water : List Nat := [4, 1, 2]

theorem resource_sum_theorem :
  base6_to_base10 mineral_x + base6_to_base10 mineral_y + base6_to_base10 water = 868 := by
  sorry

end resource_sum_theorem_l2609_260958


namespace jaden_final_cars_l2609_260993

/-- The number of toy cars Jaden has after various changes --/
def final_car_count (initial : ℕ) (bought : ℕ) (birthday : ℕ) (sister : ℕ) (friend : ℕ) : ℕ :=
  initial + bought + birthday - sister - friend

/-- Theorem stating that Jaden's final car count is 43 --/
theorem jaden_final_cars : 
  final_car_count 14 28 12 8 3 = 43 := by
  sorry

end jaden_final_cars_l2609_260993


namespace age_ratio_simplified_l2609_260989

theorem age_ratio_simplified (kul_age saras_age : ℕ) 
  (h1 : kul_age = 22) 
  (h2 : saras_age = 33) : 
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ saras_age * b = kul_age * a :=
by
  sorry

end age_ratio_simplified_l2609_260989


namespace square_root_meaningful_range_l2609_260949

theorem square_root_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 6 * x + 12) ↔ x ≥ -2 := by
  sorry

end square_root_meaningful_range_l2609_260949


namespace binomial_expansion_largest_coefficient_l2609_260915

theorem binomial_expansion_largest_coefficient (n : ℕ) : 
  (∃ k, k = 5 ∧ 
    (∀ j, j ≠ k → Nat.choose n k > Nat.choose n j) ∧
    (∀ j, j < k → Nat.choose n j < Nat.choose n (j+1)) ∧
    (∀ j, k < j ∧ j ≤ n → Nat.choose n j < Nat.choose n (j-1))) →
  n = 8 := by
sorry

end binomial_expansion_largest_coefficient_l2609_260915


namespace min_sum_squares_l2609_260978

theorem min_sum_squares (a b c : ℝ) (h : a^3 + b^3 + c^3 - 3*a*b*c = 8) :
  ∃ (m : ℝ), (∀ x y z : ℝ, x^3 + y^3 + z^3 - 3*x*y*z = 8 → x^2 + y^2 + z^2 ≥ m) ∧
             (a^2 + b^2 + c^2 ≥ m) ∧
             (m = 4) :=
sorry

end min_sum_squares_l2609_260978


namespace inequality_and_equality_condition_l2609_260998

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_product : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + b^2*c^2 ≥ 15/16 ∧
  (a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + b^2*c^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end inequality_and_equality_condition_l2609_260998


namespace eggs_per_basket_l2609_260939

theorem eggs_per_basket (purple_eggs teal_eggs min_eggs : ℕ) 
  (h1 : purple_eggs = 30)
  (h2 : teal_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (n : ℕ), n ≥ min_eggs ∧ purple_eggs % n = 0 ∧ teal_eggs % n = 0 ∧ 
  ∀ (m : ℕ), m ≥ min_eggs ∧ purple_eggs % m = 0 ∧ teal_eggs % m = 0 → m ≤ n :=
by sorry

end eggs_per_basket_l2609_260939


namespace composite_function_solution_l2609_260954

def δ (x : ℝ) : ℝ := 5 * x + 6
def φ (x : ℝ) : ℝ := 9 * x + 4

theorem composite_function_solution :
  ∀ x : ℝ, δ (φ x) = 14 → x = -4/15 := by
  sorry

end composite_function_solution_l2609_260954


namespace min_value_sum_reciprocals_l2609_260914

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x + y + z = 3) :
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) > 3 / 2 := by
sorry

end min_value_sum_reciprocals_l2609_260914


namespace marys_max_earnings_l2609_260975

/-- Calculates the maximum weekly earnings for a worker with the given parameters. -/
def maxWeeklyEarnings (maxHours regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let regularEarnings := regularRate * regularHours
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  let overtimeHours := maxHours - regularHours
  let overtimeEarnings := overtimeRate * overtimeHours
  regularEarnings + overtimeEarnings

/-- Theorem stating that Mary's maximum weekly earnings are $460 -/
theorem marys_max_earnings :
  maxWeeklyEarnings 50 20 8 (1/4) = 460 := by
  sorry

#eval maxWeeklyEarnings 50 20 8 (1/4)

end marys_max_earnings_l2609_260975


namespace smallest_a_inequality_l2609_260960

theorem smallest_a_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  (2/9 : ℝ) * (x^2 + y^2 + z^2) + x*y*z ≥ 10/27 ∧
  ∀ a < 2/9, ∃ x' y' z' : ℝ, x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧
    a * (x'^2 + y'^2 + z'^2) + x'*y'*z' < 10/27 :=
by sorry

end smallest_a_inequality_l2609_260960


namespace negative_point_two_fifth_times_five_fifth_equals_negative_one_l2609_260916

theorem negative_point_two_fifth_times_five_fifth_equals_negative_one :
  (-0.2)^5 * 5^5 = -1 := by
  sorry

end negative_point_two_fifth_times_five_fifth_equals_negative_one_l2609_260916


namespace smallest_x_satisfying_abs_equation_l2609_260999

theorem smallest_x_satisfying_abs_equation : 
  ∃ x : ℝ, (∀ y : ℝ, |5*y + 2| = 28 → x ≤ y) ∧ |5*x + 2| = 28 := by
  sorry

end smallest_x_satisfying_abs_equation_l2609_260999


namespace constant_term_proof_l2609_260929

/-- The constant term in the expansion of (x^2 + 3)(x - 2/x)^6 -/
def constantTerm : ℤ := -240

/-- The expression (x^2 + 3)(x - 2/x)^6 -/
def expression (x : ℚ) : ℚ := (x^2 + 3) * (x - 2/x)^6

theorem constant_term_proof :
  ∃ (f : ℚ → ℚ), (∀ x ≠ 0, f x = expression x) ∧
  (∃ c : ℚ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (c : ℤ) = constantTerm :=
sorry

end constant_term_proof_l2609_260929


namespace symmetric_point_wrt_origin_l2609_260909

/-- Given a point (2, -2), its symmetric point with respect to the origin has coordinates (-2, 2) -/
theorem symmetric_point_wrt_origin :
  let original_point : ℝ × ℝ := (2, -2)
  let symmetric_point : ℝ × ℝ := (-2, 2)
  (∀ (x y : ℝ), (x, y) = original_point → (-x, -y) = symmetric_point) :=
by sorry

end symmetric_point_wrt_origin_l2609_260909


namespace remaining_oranges_l2609_260927

-- Define the initial number of oranges Mildred collects
def initial_oranges : ℝ := 77.0

-- Define the number of oranges Mildred's father eats
def eaten_oranges : ℝ := 2.0

-- Theorem stating the number of oranges Mildred has after her father eats some
theorem remaining_oranges : initial_oranges - eaten_oranges = 75.0 := by
  sorry

end remaining_oranges_l2609_260927


namespace probability_even_distinct_digits_l2609_260952

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

def count_valid_numbers : ℕ := sorry

theorem probability_even_distinct_digits :
  (count_valid_numbers : ℚ) / (9999 - 1000 + 1 : ℚ) = 343 / 1125 := by sorry

end probability_even_distinct_digits_l2609_260952


namespace collinear_points_b_value_l2609_260983

/-- Three points are collinear if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_b_value :
  ∀ b : ℚ, collinear 5 (-3) (-b + 3) 5 (3*b + 1) 4 → b = 18/31 := by
  sorry

end collinear_points_b_value_l2609_260983


namespace function_product_l2609_260970

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem function_product (y z : ℝ) 
  (h1 : -1 < y ∧ y < 1) 
  (h2 : -1 < z ∧ z < 1) 
  (h3 : f ((y + z) / (1 + y * z)) = 1) 
  (h4 : f ((y - z) / (1 - y * z)) = 2) : 
  f y * f z = -3/4 := by
sorry

end function_product_l2609_260970


namespace triangle_properties_l2609_260979

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- a, b, c are sides opposite to A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * (Real.sin ((B + C) / 2))^2 →
  -- Radius of circumcircle
  2 * Real.sqrt 3 = 2 * a / Real.sin A →
  -- Prove A = π/3
  A = π/3 ∧
  -- Prove maximum area is 9√3
  (1/2) * b * c * Real.sin A ≤ 9 * Real.sqrt 3 :=
by sorry

end triangle_properties_l2609_260979


namespace train_speed_problem_l2609_260946

theorem train_speed_problem (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 44 →
  passing_time = 36 →
  train_length = 40 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (5/18) * passing_time = 2 * train_length :=
by sorry

end train_speed_problem_l2609_260946


namespace smallest_number_l2609_260913

theorem smallest_number : 
  ∀ (a b c : ℝ), a = -Real.sqrt 2 ∧ b = 3.14 ∧ c = 2021 → 
    a < 0 ∧ a < b ∧ a < c :=
by sorry

end smallest_number_l2609_260913


namespace school_relationship_l2609_260984

/-- In a school with teachers and students, prove the relationship between
    the number of teachers, students, students per teacher, and teachers per student. -/
theorem school_relationship (m n k ℓ : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : k > 0) 
  (h4 : ℓ > 0) 
  (teacher_students : ∀ t, t ≤ m → (∃ s, s ≤ n ∧ s = k))
  (student_teachers : ∀ s, s ≤ n → (∃ t, t ≤ m ∧ t = ℓ)) :
  m * k = n * ℓ := by
  sorry


end school_relationship_l2609_260984


namespace subtraction_problem_l2609_260933

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end subtraction_problem_l2609_260933


namespace juan_tire_count_l2609_260956

/-- The number of tires on the vehicles Juan saw --/
def total_tires (cars trucks bicycles tricycles : ℕ) : ℕ :=
  4 * (cars + trucks) + 2 * bicycles + 3 * tricycles

/-- Theorem stating the total number of tires Juan saw --/
theorem juan_tire_count : total_tires 15 8 3 1 = 101 := by
  sorry

end juan_tire_count_l2609_260956


namespace hyperbola_equation_l2609_260966

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  hpos : 0 < a ∧ 0 < b
  hc : c = Real.sqrt 5
  hasymptote : b / a = 1 / 2

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 / 4 - y^2 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1

theorem hyperbola_equation (h : Hyperbola) : standard_equation h := by
  sorry

end hyperbola_equation_l2609_260966


namespace vehicle_speed_problem_l2609_260944

/-- Proves that the average speed of vehicle X is 36 miles per hour given the conditions -/
theorem vehicle_speed_problem (initial_distance : ℝ) (y_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 22 →
  y_speed = 45 →
  time = 5 →
  final_distance = 23 →
  let x_distance := y_speed * time - (initial_distance + final_distance)
  let x_speed := x_distance / time
  x_speed = 36 := by sorry

end vehicle_speed_problem_l2609_260944


namespace circle_ratio_l2609_260925

theorem circle_ratio (r R a b : ℝ) (hr : r > 0) (hR : R > r) (hab : a > b) (hb : b > 0) 
  (h : R^2 = (a/b) * (R^2 - r^2)) : 
  R/r = Real.sqrt (a/(a-b)) := by
  sorry

end circle_ratio_l2609_260925


namespace total_weight_is_103_2_l2609_260903

/-- The total weight of all books owned by Sandy, Benny, and Tim -/
def total_weight : ℝ :=
  let sandy_books := 10
  let sandy_weight := 1.5
  let benny_books := 24
  let benny_weight := 1.2
  let tim_books := 33
  let tim_weight := 1.8
  sandy_books * sandy_weight + benny_books * benny_weight + tim_books * tim_weight

/-- Theorem stating that the total weight of all books is 103.2 pounds -/
theorem total_weight_is_103_2 : total_weight = 103.2 := by
  sorry

end total_weight_is_103_2_l2609_260903


namespace conditional_probability_B_given_A_l2609_260988

/-- Two fair six-sided dice are thrown. -/
def dice_space : Type := Fin 6 × Fin 6

/-- Event A: "the number of points on die A is greater than 4" -/
def event_A : Set dice_space :=
  {x | x.1 > 4}

/-- Event B: "the sum of the number of points on dice A and B is equal to 7" -/
def event_B : Set dice_space :=
  {x | x.1.val + x.2.val = 7}

/-- The probability measure on the dice space -/
def P : Set dice_space → ℝ :=
  sorry

/-- Theorem: The conditional probability P(B|A) = 1/6 -/
theorem conditional_probability_B_given_A :
  P event_B / P event_A = 1 / 6 := by
  sorry

end conditional_probability_B_given_A_l2609_260988


namespace park_creatures_l2609_260921

theorem park_creatures (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300)
  (h2 : total_legs = 686) : ∃ (birds mammals imaginary : ℕ),
  birds + mammals + imaginary = total_heads ∧
  2 * birds + 4 * mammals + 3 * imaginary = total_legs ∧
  birds = 214 := by
  sorry

end park_creatures_l2609_260921


namespace complex_fraction_equals_negative_i_negative_i_coordinates_l2609_260900

theorem complex_fraction_equals_negative_i :
  let z : ℂ := (1 - 2*I) / (2 + I)
  z = -I :=
by sorry

theorem negative_i_coordinates :
  let z : ℂ := -I
  Complex.re z = 0 ∧ Complex.im z = -1 :=
by sorry

end complex_fraction_equals_negative_i_negative_i_coordinates_l2609_260900


namespace cubic_factorization_l2609_260926

theorem cubic_factorization (a : ℝ) : a^3 - 9*a = a*(a+3)*(a-3) := by sorry

end cubic_factorization_l2609_260926


namespace power_of_two_divisibility_l2609_260934

theorem power_of_two_divisibility (n : ℕ+) :
  (∀ (n : ℕ+), ∃ (m : ℤ), (2^n.val - 1) ∣ (m^2 + 9)) ↔
  ∃ (s : ℕ), n.val = 2^s :=
sorry

end power_of_two_divisibility_l2609_260934


namespace perpendicular_condition_l2609_260990

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the "contained in" relation
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_condition 
  (a : Line) (α β : Plane) 
  (h_contained : contained_in a α) :
  (∀ β, perpendicular a β → perpendicular_planes α β) ∧ 
  (∃ β, perpendicular_planes α β ∧ ¬perpendicular a β) :=
sorry

end perpendicular_condition_l2609_260990


namespace rectangle_area_increase_l2609_260940

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let original_area := L * W
  let new_area := (1.2 * L) * (1.2 * W)
  (new_area - original_area) / original_area * 100 = 44 := by
  sorry

end rectangle_area_increase_l2609_260940


namespace student_count_problem_l2609_260906

theorem student_count_problem : ∃! n : ℕ, n < 500 ∧ 
  n % 17 = 15 ∧ 
  n % 19 = 18 ∧ 
  n % 16 = 7 ∧ 
  n = 417 := by
sorry

end student_count_problem_l2609_260906


namespace ellipse_intersection_max_y_intercept_l2609_260930

/-- An ellipse with major axis 2√2 times the minor axis, passing through (2, √2/2) --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : a = 2 * Real.sqrt 2 * b
  h4 : (2 / a)^2 + ((Real.sqrt 2 / 2) / b)^2 = 1

/-- A line intersecting the ellipse at two points --/
structure IntersectingLine (e : Ellipse) where
  k : ℝ
  m : ℝ
  h1 : k ≠ 0  -- Line is not parallel to coordinate axes

/-- The distance between intersection points is 2√2 --/
def intersection_distance (e : Ellipse) (l : IntersectingLine e) : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1^2 / e.a^2) + ((l.k * x1 + l.m)^2 / e.b^2) = 1 ∧
    (x2^2 / e.a^2) + ((l.k * x2 + l.m)^2 / e.b^2) = 1 ∧
    (x2 - x1)^2 + (l.k * (x2 - x1))^2 = 8

/-- The theorem to be proved --/
theorem ellipse_intersection_max_y_intercept (e : Ellipse) :
  ∃ (max_m : ℝ), max_m = Real.sqrt 14 - Real.sqrt 7 ∧
  ∀ (l : IntersectingLine e), intersection_distance e l →
    l.m ≤ max_m ∧
    ∃ (l' : IntersectingLine e), intersection_distance e l' ∧ l'.m = max_m :=
sorry

end ellipse_intersection_max_y_intercept_l2609_260930


namespace unique_magnitude_for_quadratic_roots_l2609_260986

theorem unique_magnitude_for_quadratic_roots (z : ℂ) : 
  z^2 - 8*z + 37 = 0 → ∃! m : ℝ, ∃ w : ℂ, w^2 - 8*w + 37 = 0 ∧ Complex.abs w = m :=
by sorry

end unique_magnitude_for_quadratic_roots_l2609_260986


namespace mechanics_total_charge_l2609_260953

/-- Calculates the total charge for two mechanics given their hourly rates, total combined work time, and the second mechanic's work time. -/
theorem mechanics_total_charge 
  (rate1 : ℕ) 
  (rate2 : ℕ) 
  (total_hours : ℕ) 
  (hours2 : ℕ) : 
  rate1 = 45 → 
  rate2 = 85 → 
  total_hours = 20 → 
  hours2 = 5 → 
  rate1 * (total_hours - hours2) + rate2 * hours2 = 1100 := by
sorry

end mechanics_total_charge_l2609_260953


namespace f_properties_l2609_260985

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_properties (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  (x₂ * f x₁ < x₁ * f x₂) ∧
  (x₁ > Real.exp (-1) → x₁ * f x₁ + x₂ * f x₂ > x₂ * f x₁ + x₁ * f x₂) := by
  sorry

end f_properties_l2609_260985


namespace arcsin_sin_eq_x_div_3_l2609_260919

theorem arcsin_sin_eq_x_div_3 :
  ∃! x : ℝ, x ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2) ∧ 
    Real.arcsin (Real.sin x) = x / 3 := by
  sorry

end arcsin_sin_eq_x_div_3_l2609_260919


namespace least_five_digit_square_cube_l2609_260942

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n = 15625) ∧
  (∀ m : ℕ, m < n → m < 10000 ∨ m > 99999 ∨ ¬∃ a : ℕ, m = a^2 ∨ ¬∃ b : ℕ, m = b^3) ∧
  (∃ x : ℕ, n = x^2) ∧
  (∃ y : ℕ, n = y^3) ∧
  (n ≥ 10000) ∧
  (n ≤ 99999) :=
by sorry

end least_five_digit_square_cube_l2609_260942


namespace frisbee_sales_theorem_l2609_260996

/-- Represents the total number of frisbees sold -/
def total_frisbees : ℕ := 60

/-- Represents the number of $3 frisbees sold -/
def frisbees_3 : ℕ := 36

/-- Represents the number of $4 frisbees sold -/
def frisbees_4 : ℕ := 24

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 204

/-- Theorem stating that the total number of frisbees sold is 60 -/
theorem frisbee_sales_theorem :
  (frisbees_3 * 3 + frisbees_4 * 4 = total_receipts) ∧
  (frisbees_4 ≥ 24) ∧
  (total_frisbees = frisbees_3 + frisbees_4) :=
by sorry

end frisbee_sales_theorem_l2609_260996


namespace basketball_free_throws_l2609_260991

theorem basketball_free_throws (two_points three_points free_throws : ℕ) : 
  (3 * three_points = 2 * two_points) →  -- Points from three-point shots are twice the points from two-point shots
  (free_throws = 2 * two_points - 1) →   -- Number of free throws is twice the number of two-point shots minus one
  (2 * two_points + 3 * three_points + free_throws = 89) →  -- Total score is 89 points
  free_throws = 29 := by
  sorry

end basketball_free_throws_l2609_260991


namespace no_arithmetic_sqrt_neg_nine_l2609_260995

-- Define the concept of an arithmetic square root
def arithmetic_sqrt (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x ∧ y ≥ 0

-- Theorem stating that the arithmetic square root of -9 does not exist
theorem no_arithmetic_sqrt_neg_nine :
  ¬ arithmetic_sqrt (-9) :=
sorry

end no_arithmetic_sqrt_neg_nine_l2609_260995


namespace domain_of_f_squared_l2609_260928

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem domain_of_f_squared :
  {x : ℝ | ∃ y ∈ dom_f, x^2 = y} = Set.Icc (-1) 1 := by sorry

end domain_of_f_squared_l2609_260928


namespace ratio_equality_l2609_260943

theorem ratio_equality {a₁ a₂ a₃ b₁ b₂ b₃ p₁ p₂ p₃ : ℝ} (h1 : a₁ / b₁ = a₂ / b₂) (h2 : a₁ / b₁ = a₃ / b₃)
    (h3 : ¬(p₁ = 0 ∧ p₂ = 0 ∧ p₃ = 0)) :
  ∀ n : ℕ, (a₁ / b₁) ^ n = (p₁ * a₁^n + p₂ * a₂^n + p₃ * a₃^n) / (p₁ * b₁^n + p₂ * b₂^n + p₃ * b₃^n) := by
  sorry

end ratio_equality_l2609_260943


namespace olivia_coins_left_l2609_260922

/-- The number of coins Olivia has left after buying a soda -/
def coins_left (initial_quarters : ℕ) (spent_quarters : ℕ) : ℕ :=
  initial_quarters - spent_quarters

/-- Theorem: Olivia has 7 coins left after buying a soda -/
theorem olivia_coins_left : coins_left 11 4 = 7 := by
  sorry

end olivia_coins_left_l2609_260922


namespace ratio_adjustment_l2609_260980

theorem ratio_adjustment (x : ℚ) : x = 29 ↔ (4 + x) / (15 + x) = 3 / 4 := by
  sorry

end ratio_adjustment_l2609_260980


namespace multiplication_addition_equality_l2609_260905

theorem multiplication_addition_equality : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end multiplication_addition_equality_l2609_260905


namespace remainder_50_power_50_mod_7_l2609_260947

theorem remainder_50_power_50_mod_7 : 50^50 % 7 = 1 := by
  sorry

end remainder_50_power_50_mod_7_l2609_260947


namespace arithmetic_sequence_a10_l2609_260951

/-- An arithmetic sequence with a_2 = 2 and a_3 = 4 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 3 - a 2

theorem arithmetic_sequence_a10 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end arithmetic_sequence_a10_l2609_260951


namespace set_operation_result_l2609_260936

def U : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem set_operation_result :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end set_operation_result_l2609_260936


namespace initial_apples_in_pile_l2609_260964

def apple_pile (initial : ℕ) (added : ℕ) (final : ℕ) : Prop :=
  initial + added = final

def package_size : ℕ := 11

theorem initial_apples_in_pile : 
  ∃ (initial : ℕ), apple_pile initial 5 13 ∧ initial = 8 := by
  sorry

end initial_apples_in_pile_l2609_260964


namespace area_difference_square_rectangle_l2609_260992

/-- Given a square and a rectangle with the same perimeter, this theorem proves
    the difference between their areas when specific dimensions are provided. -/
theorem area_difference_square_rectangle (square_perimeter : ℝ) (rect_perimeter : ℝ) (rect_length : ℝ)
  (h1 : square_perimeter = 52)
  (h2 : rect_perimeter = 52)
  (h3 : rect_length = 15) :
  (square_perimeter / 4) ^ 2 - rect_length * ((rect_perimeter / 2) - rect_length) = 4 :=
by sorry


end area_difference_square_rectangle_l2609_260992


namespace raisin_distribution_l2609_260967

theorem raisin_distribution (total_raisins total_boxes box1_raisins box2_raisins : ℕ) 
  (h1 : total_raisins = 437)
  (h2 : total_boxes = 5)
  (h3 : box1_raisins = 72)
  (h4 : box2_raisins = 74)
  (h5 : ∃ (equal_box_raisins : ℕ), 
    total_raisins = box1_raisins + box2_raisins + 3 * equal_box_raisins) :
  ∃ (equal_box_raisins : ℕ), equal_box_raisins = 97 ∧ 
    total_raisins = box1_raisins + box2_raisins + 3 * equal_box_raisins :=
by sorry

end raisin_distribution_l2609_260967


namespace increasing_cubic_function_parameter_negative_l2609_260918

/-- Given a function y = a(x^3 - 3x) that is increasing on the interval (-1, 1), prove that a < 0 --/
theorem increasing_cubic_function_parameter_negative
  (a : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = a * (x^3 - 3*x))
  (h2 : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, StrictMono y):
  a < 0 :=
sorry

end increasing_cubic_function_parameter_negative_l2609_260918


namespace x_squared_mod_24_l2609_260908

theorem x_squared_mod_24 (x : ℤ) 
  (h1 : 6 * x ≡ 12 [ZMOD 24])
  (h2 : 4 * x ≡ 20 [ZMOD 24]) : 
  x^2 ≡ 12 [ZMOD 24] := by
sorry

end x_squared_mod_24_l2609_260908


namespace rice_wheat_division_l2609_260941

/-- Calculates the approximate amount of wheat grains in a large quantity of mixed grains,
    given a sample ratio. -/
def approximate_wheat_amount (total_amount : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) : ℕ :=
  (total_amount * wheat_in_sample) / sample_size

/-- The rice and wheat division problem from "Jiuzhang Suanshu" -/
theorem rice_wheat_division :
  let total_amount : ℕ := 1536
  let sample_size : ℕ := 224
  let wheat_in_sample : ℕ := 28
  approximate_wheat_amount total_amount sample_size wheat_in_sample = 192 := by
  sorry

#eval approximate_wheat_amount 1536 224 28

end rice_wheat_division_l2609_260941


namespace circle_tangent_to_line_l2609_260950

/-- The value of m for which the circle x^2 + y^2 = m^2 is tangent to the line x + y = m -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = m^2 ∧ x + y = m) → m = 0 := by
  sorry


end circle_tangent_to_line_l2609_260950


namespace class_vision_only_comprehensive_l2609_260912

-- Define the concept of a survey
structure Survey where
  population : Type
  data_collection : population → Bool

-- Define what makes a survey comprehensive
def is_comprehensive (s : Survey) : Prop :=
  ∀ x : s.population, s.data_collection x

-- Define the specific surveys
def bulb_survey : Survey := sorry
def class_vision_survey : Survey := sorry
def food_preservative_survey : Survey := sorry
def river_water_quality_survey : Survey := sorry

-- State the theorem
theorem class_vision_only_comprehensive :
  is_comprehensive class_vision_survey ∧
  ¬is_comprehensive bulb_survey ∧
  ¬is_comprehensive food_preservative_survey ∧
  ¬is_comprehensive river_water_quality_survey :=
sorry

end class_vision_only_comprehensive_l2609_260912


namespace quadratic_inequality_solution_l2609_260910

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 - 5 * x - 2 < 0 ↔ -1/3 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_solution_l2609_260910


namespace no_integer_solution_l2609_260945

theorem no_integer_solution : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end no_integer_solution_l2609_260945


namespace quadratic_composition_theorem_l2609_260920

/-- A unitary quadratic trinomial -/
structure UnitaryQuadratic where
  b : ℝ
  c : ℝ

/-- Evaluate a unitary quadratic trinomial at a point -/
def evaluate (f : UnitaryQuadratic) (x : ℝ) : ℝ :=
  x^2 + f.b * x + f.c

/-- Composition of two unitary quadratic trinomials -/
def compose (f g : UnitaryQuadratic) : UnitaryQuadratic :=
  { b := g.b^2 + f.b * (1 + g.b) + g.c * f.b
    c := g.c^2 + f.b * g.c + f.c }

/-- A polynomial has no real roots -/
def hasNoRealRoots (f : UnitaryQuadratic) : Prop :=
  ∀ x : ℝ, evaluate f x ≠ 0

theorem quadratic_composition_theorem (f g : UnitaryQuadratic) 
    (h1 : hasNoRealRoots (compose f g))
    (h2 : hasNoRealRoots (compose g f)) :
    hasNoRealRoots (compose f f) ∨ hasNoRealRoots (compose g g) := by
  sorry

end quadratic_composition_theorem_l2609_260920


namespace prob_at_least_three_heads_is_half_l2609_260963

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The probability of getting at least three heads when flipping five coins -/
def prob_at_least_three_heads : ℚ := 1/2

/-- Theorem stating that the probability of getting at least three heads 
    when flipping five coins simultaneously is 1/2 -/
theorem prob_at_least_three_heads_is_half : 
  prob_at_least_three_heads = 1/2 := by sorry

end prob_at_least_three_heads_is_half_l2609_260963


namespace non_positive_sequence_l2609_260976

theorem non_positive_sequence (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h_ineq : ∀ k : ℕ, k ∈ Finset.range (n - 1) → a k - 2 * a (k + 1) + a (k + 2) ≥ 0) :
  ∀ i : ℕ, i ≤ n → a i ≤ 0 :=
sorry

end non_positive_sequence_l2609_260976


namespace smallest_prime_factor_of_3063_l2609_260977

theorem smallest_prime_factor_of_3063 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3063 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3063 → p ≤ q :=
sorry

end smallest_prime_factor_of_3063_l2609_260977


namespace game_download_time_l2609_260969

/-- Calculates the remaining download time for a game -/
theorem game_download_time (total_size : ℕ) (downloaded : ℕ) (speed : ℕ) : 
  total_size = 880 ∧ downloaded = 310 ∧ speed = 3 → 
  (total_size - downloaded) / speed = 190 := by
  sorry

end game_download_time_l2609_260969


namespace zoo_guide_count_l2609_260968

/-- Represents the number of children spoken to by a guide on a specific day -/
structure DailyGuideCount where
  english : ℕ
  french : ℕ
  spanish : ℕ

/-- Represents the count of guides for each language -/
structure GuideCount where
  total : ℕ
  english : ℕ
  french : ℕ

def weekend_count (guides : GuideCount) (friday saturday sunday : DailyGuideCount) : ℕ :=
  let spanish_guides := guides.total - guides.english - guides.french
  let friday_total := guides.english * friday.english + guides.french * friday.french + spanish_guides * friday.spanish
  let saturday_total := guides.english * saturday.english + guides.french * saturday.french + spanish_guides * saturday.spanish
  let sunday_total := guides.english * sunday.english + guides.french * sunday.french + spanish_guides * sunday.spanish
  friday_total + saturday_total + sunday_total

theorem zoo_guide_count :
  let guides : GuideCount := { total := 22, english := 10, french := 6 }
  let friday : DailyGuideCount := { english := 20, french := 25, spanish := 30 }
  let saturday : DailyGuideCount := { english := 22, french := 24, spanish := 32 }
  let sunday : DailyGuideCount := { english := 24, french := 23, spanish := 35 }
  weekend_count guides friday saturday sunday = 1674 := by
  sorry


end zoo_guide_count_l2609_260968


namespace journey_fraction_by_rail_l2609_260982

theorem journey_fraction_by_rail 
  (total_journey : ℝ) 
  (bus_fraction : ℝ) 
  (foot_distance : ℝ) : 
  total_journey = 130 ∧ 
  bus_fraction = 17/20 ∧ 
  foot_distance = 6.5 → 
  (total_journey - bus_fraction * total_journey - foot_distance) / total_journey = 1/10 := by
sorry

end journey_fraction_by_rail_l2609_260982


namespace apple_distribution_l2609_260994

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 3

/-- The number of apples Kevin has -/
def kevins_apples : ℕ := 2 * jackies_apples

/-- The number of apples Adam has -/
def adams_apples : ℕ := jackies_apples + 8

/-- The total number of apples Adam, Jackie, and Kevin have -/
def total_apples : ℕ := jackies_apples + kevins_apples + adams_apples

/-- The number of apples He has -/
def his_apples : ℕ := 3 * total_apples

theorem apple_distribution :
  total_apples = 20 ∧ his_apples = 60 :=
sorry

end apple_distribution_l2609_260994


namespace nancy_shoe_count_nancy_has_168_shoes_l2609_260911

/-- Calculates the total number of individual shoes Nancy has given her shoe collection. -/
theorem nancy_shoe_count (boots : ℕ) (slippers : ℕ) (heels : ℕ) : ℕ :=
  let total_pairs := boots + slippers + heels
  2 * total_pairs

/-- Proves that Nancy has 168 individual shoes given the conditions of her shoe collection. -/
theorem nancy_has_168_shoes : nancy_shoe_count 6 15 63 = 168 := by
  sorry

#check nancy_has_168_shoes

end nancy_shoe_count_nancy_has_168_shoes_l2609_260911


namespace cookie_accident_l2609_260901

/-- Problem: Cookie Baking Accident -/
theorem cookie_accident (alice_initial bob_initial alice_additional bob_additional final_edible : ℕ) :
  alice_initial = 74 →
  bob_initial = 7 →
  alice_additional = 5 →
  bob_additional = 36 →
  final_edible = 93 →
  (alice_initial + bob_initial + alice_additional + bob_additional) - final_edible = 29 :=
by sorry

end cookie_accident_l2609_260901


namespace polynomial_root_magnitude_implies_a_range_l2609_260987

/-- A polynomial of degree 4 with real coefficients -/
structure Polynomial4 (a : ℝ) where
  coeff : Fin 5 → ℝ
  coeff_0 : coeff 0 = 2
  coeff_1 : coeff 1 = a
  coeff_2 : coeff 2 = 9
  coeff_3 : coeff 3 = a
  coeff_4 : coeff 4 = 2

/-- The roots of a polynomial -/
def roots (p : Polynomial4 a) : Finset ℂ := sorry

/-- Predicate to check if all roots are complex -/
def allRootsComplex (p : Polynomial4 a) : Prop :=
  ∀ r ∈ roots p, r.im ≠ 0

/-- Predicate to check if all root magnitudes are not equal to 1 -/
def allRootMagnitudesNotOne (p : Polynomial4 a) : Prop :=
  ∀ r ∈ roots p, Complex.abs r ≠ 1

/-- The main theorem -/
theorem polynomial_root_magnitude_implies_a_range (a : ℝ) (p : Polynomial4 a) 
    (h1 : allRootsComplex p) (h2 : allRootMagnitudesNotOne p) : 
    a ∈ Set.Ioo (-2 * Real.sqrt 10) (2 * Real.sqrt 10) := by
  sorry

end polynomial_root_magnitude_implies_a_range_l2609_260987


namespace repeating_not_necessarily_periodic_l2609_260902

/-- Definition of the sequence property --/
def has_repeating_property (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, ∃ t : ℕ, t > 0 ∧ ∀ n : ℕ, a k = a (k + n * t)

/-- Definition of periodicity --/
def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ k : ℕ, a k = a (k + T)

/-- Theorem stating that a sequence with the repeating property is not necessarily periodic --/
theorem repeating_not_necessarily_periodic :
  ∃ a : ℕ → ℕ, has_repeating_property a ∧ ¬ is_periodic a := by
  sorry

end repeating_not_necessarily_periodic_l2609_260902


namespace biased_coin_expected_value_l2609_260959

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (win_heads : ℚ) (win_tails : ℚ) : ℚ :=
  p_heads * win_heads + p_tails * win_tails

theorem biased_coin_expected_value :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_heads : ℚ := 5
  let win_tails : ℚ := -4
  coin_flip_expected_value p_heads p_tails win_heads win_tails = -2/5 := by
sorry

end biased_coin_expected_value_l2609_260959


namespace range_of_m_l2609_260961

-- Define the propositions p and q
def p (m : ℝ) : Prop := (1 + 1 - 2*m + 2*m + 2*m^2 - 4) < 0

def q (m : ℝ) : Prop := m ≥ 0 ∧ 2*m + 1 ≥ 0

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∀ m : ℝ, (-1 < m ∧ m < 0) ∨ m ≥ 1) :=
sorry

end range_of_m_l2609_260961


namespace digit_difference_in_base_d_l2609_260981

/-- Given two digits A and B in base d > 6, if AB_d + AA_d = 172_d, then A_d - B_d = 3_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h_d : d > 6)
  (h_digits : A < d ∧ B < d)
  (h_sum : d * B + A + d * A + A = d^2 + 7 * d + 2) :
  A - B = 3 :=
sorry

end digit_difference_in_base_d_l2609_260981


namespace drevlandia_roads_l2609_260997

-- Define the number of cities
def num_cities : ℕ := 101

-- Define the function to calculate the number of roads
def num_roads (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem drevlandia_roads : num_roads num_cities = 5050 := by
  sorry

end drevlandia_roads_l2609_260997


namespace paper_folding_ratio_l2609_260938

theorem paper_folding_ratio : 
  let square_side : ℝ := 1
  let large_rect_length : ℝ := square_side
  let large_rect_width : ℝ := square_side / 2
  let small_rect_length : ℝ := square_side
  let small_rect_width : ℝ := square_side / 4
  let large_rect_perimeter : ℝ := 2 * (large_rect_length + large_rect_width)
  let small_rect_perimeter : ℝ := 2 * (small_rect_length + small_rect_width)
  small_rect_perimeter / large_rect_perimeter = 5 / 6 := by
sorry

end paper_folding_ratio_l2609_260938


namespace distribute_seven_balls_three_boxes_l2609_260937

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end distribute_seven_balls_three_boxes_l2609_260937
