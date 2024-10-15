import Mathlib

namespace NUMINAMATH_CALUDE_geometric_progression_condition_l3858_385840

/-- Given real numbers a, b, c with b < 0, prove that b^2 = ac is necessary and 
    sufficient for a, b, c to form a geometric progression -/
theorem geometric_progression_condition (a b c : ℝ) (h : b < 0) :
  (b^2 = a*c) ↔ ∃ r : ℝ, (r ≠ 0 ∧ b = a*r ∧ c = b*r) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_condition_l3858_385840


namespace NUMINAMATH_CALUDE_distance_MN_equals_5_l3858_385876

def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 5)

theorem distance_MN_equals_5 : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_MN_equals_5_l3858_385876


namespace NUMINAMATH_CALUDE_m_is_smallest_l3858_385833

/-- The smallest positive integer satisfying the given divisibility conditions -/
def m : ℕ := 60

theorem m_is_smallest : m = 60 ∧ 
  (∃ (n : ℕ), m = 13 * n + 8 ∧ m = 15 * n) ∧
  (∀ (k : ℕ), k > 0 ∧ k < m → 
    ¬(∃ (n : ℕ), k = 13 * n + 8 ∧ k = 15 * n)) := by
  sorry

#check m_is_smallest

end NUMINAMATH_CALUDE_m_is_smallest_l3858_385833


namespace NUMINAMATH_CALUDE_abs_one_minus_i_l3858_385862

theorem abs_one_minus_i : Complex.abs (1 - Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_one_minus_i_l3858_385862


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3858_385835

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a < 0 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3858_385835


namespace NUMINAMATH_CALUDE_number_of_model_X_computers_prove_number_of_model_X_computers_l3858_385885

/-- Represents the time (in minutes) for a model X computer to complete the task -/
def modelXTime : ℝ := 72

/-- Represents the time (in minutes) for a model Y computer to complete the task -/
def modelYTime : ℝ := 36

/-- Represents the total time (in minutes) for the combined computers to complete the task -/
def totalTime : ℝ := 1

/-- Theorem stating that the number of model X computers used is 24 -/
theorem number_of_model_X_computers : ℕ :=
  24

/-- Proof that the number of model X computers used is 24 -/
theorem prove_number_of_model_X_computers :
  (modelXTime : ℝ) > 0 ∧ (modelYTime : ℝ) > 0 ∧ totalTime > 0 →
  ∃ (n : ℕ), n > 0 ∧ n = number_of_model_X_computers ∧
  (n : ℝ) * (1 / modelXTime + 1 / modelYTime) = 1 / totalTime :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_model_X_computers_prove_number_of_model_X_computers_l3858_385885


namespace NUMINAMATH_CALUDE_sin_270_degrees_l3858_385837

theorem sin_270_degrees : Real.sin (270 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_270_degrees_l3858_385837


namespace NUMINAMATH_CALUDE_tan_eleven_pi_thirds_l3858_385896

theorem tan_eleven_pi_thirds : Real.tan (11 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_thirds_l3858_385896


namespace NUMINAMATH_CALUDE_total_cups_in_trays_l3858_385898

theorem total_cups_in_trays (first_tray second_tray : ℕ) 
  (h1 : second_tray = first_tray - 20) 
  (h2 : second_tray = 240) : 
  first_tray + second_tray = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_in_trays_l3858_385898


namespace NUMINAMATH_CALUDE_system_solution_l3858_385873

theorem system_solution : ∃ x y : ℤ, (3 * x - 14 * y = 2) ∧ (4 * y - x = 6) ∧ x = -46 ∧ y = -10 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3858_385873


namespace NUMINAMATH_CALUDE_gcf_of_21_and_12_l3858_385856

theorem gcf_of_21_and_12 (h : Nat.lcm 21 12 = 42) : Nat.gcd 21 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_21_and_12_l3858_385856


namespace NUMINAMATH_CALUDE_binomial_probability_5_to_7_successes_in_8_trials_l3858_385806

theorem binomial_probability_5_to_7_successes_in_8_trials :
  let n : ℕ := 8
  let p : ℝ := 1/2
  let q : ℝ := 1 - p
  let X : ℕ → ℝ := λ k => Nat.choose n k * p^k * q^(n-k)
  (X 5 + X 6 + X 7) = 23/64 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_5_to_7_successes_in_8_trials_l3858_385806


namespace NUMINAMATH_CALUDE_silverware_probability_l3858_385814

theorem silverware_probability (forks spoons knives : ℕ) (h1 : forks = 6) (h2 : spoons = 6) (h3 : knives = 6) :
  let total := forks + spoons + knives
  let ways_to_choose_three := Nat.choose total 3
  let ways_to_choose_one_each := forks * spoons * knives
  (ways_to_choose_one_each : ℚ) / ways_to_choose_three = 9 / 34 :=
by
  sorry

end NUMINAMATH_CALUDE_silverware_probability_l3858_385814


namespace NUMINAMATH_CALUDE_factors_imply_unique_h_k_l3858_385886

-- Define the polynomial
def P (h k : ℝ) (x : ℝ) : ℝ := 3 * x^4 - h * x^2 + k * x - 12

-- State the theorem
theorem factors_imply_unique_h_k :
  ∀ h k : ℝ,
  (∀ x : ℝ, P h k x = 0 ↔ x = 3 ∨ x = -4) →
  ∃! (h' k' : ℝ), P h' k' = P h k :=
by sorry

end NUMINAMATH_CALUDE_factors_imply_unique_h_k_l3858_385886


namespace NUMINAMATH_CALUDE_parallelepiped_diagonal_l3858_385828

/-- The diagonal of a rectangular parallelepiped given its face diagonals -/
theorem parallelepiped_diagonal (m n p : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = (m^2 + n^2 + p^2) / 2 := by
  sorry

#check parallelepiped_diagonal

end NUMINAMATH_CALUDE_parallelepiped_diagonal_l3858_385828


namespace NUMINAMATH_CALUDE_bread_baking_time_l3858_385889

/-- The time it takes for one ball of dough to rise -/
def rise_time : ℕ := 3

/-- The time it takes to bake one ball of dough -/
def bake_time : ℕ := 2

/-- The number of balls of dough Ellen makes -/
def num_balls : ℕ := 4

/-- The total time taken to make and bake all balls of dough -/
def total_time : ℕ := rise_time + (num_balls - 1) * rise_time + num_balls * bake_time

theorem bread_baking_time :
  total_time = 14 :=
sorry

end NUMINAMATH_CALUDE_bread_baking_time_l3858_385889


namespace NUMINAMATH_CALUDE_g_at_negative_three_l3858_385836

-- Define the property of g being a rational function satisfying the given equation
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2

-- State the theorem
theorem g_at_negative_three (g : ℝ → ℝ) (h : is_valid_g g) : g (-3) = 247 / 39 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_three_l3858_385836


namespace NUMINAMATH_CALUDE_stationery_prices_l3858_385831

theorem stationery_prices (pen_price notebook_price : ℝ) : 
  pen_price + notebook_price = 3.6 →
  pen_price + 4 * notebook_price = 10.5 →
  pen_price = 1.3 ∧ notebook_price = 2.3 := by
sorry

end NUMINAMATH_CALUDE_stationery_prices_l3858_385831


namespace NUMINAMATH_CALUDE_lesser_number_problem_l3858_385822

theorem lesser_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 612) :
  min x y = 21.395 := by sorry

end NUMINAMATH_CALUDE_lesser_number_problem_l3858_385822


namespace NUMINAMATH_CALUDE_triangle_properties_l3858_385859

/-- Given a triangle ABC with angle B = 150°, side a = √3c, and side b = 2√7,
    prove that its area is √3 and if sin A + √3 sin C = √2/2, then C = 15° --/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  B = 150 * π / 180 →
  a = Real.sqrt 3 * c →
  b = 2 * Real.sqrt 7 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 ∧
  (Real.sin A + Real.sqrt 3 * Real.sin C = Real.sqrt 2 / 2 → C = 15 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3858_385859


namespace NUMINAMATH_CALUDE_evening_to_morning_ratio_l3858_385815

def morning_miles : ℝ := 2
def total_miles : ℝ := 12

def evening_miles : ℝ := total_miles - morning_miles

theorem evening_to_morning_ratio :
  evening_miles / morning_miles = 5 := by sorry

end NUMINAMATH_CALUDE_evening_to_morning_ratio_l3858_385815


namespace NUMINAMATH_CALUDE_square_root_sum_l3858_385800

theorem square_root_sum (y : ℝ) : 
  Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4 →
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l3858_385800


namespace NUMINAMATH_CALUDE_basketball_weight_proof_l3858_385852

/-- The weight of one basketball in pounds -/
def basketball_weight : ℚ := 125 / 9

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℚ := 25

theorem basketball_weight_proof :
  (9 : ℚ) * basketball_weight = (5 : ℚ) * bicycle_weight ∧
  (3 : ℚ) * bicycle_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_basketball_weight_proof_l3858_385852


namespace NUMINAMATH_CALUDE_triangle_equilateral_l3858_385845

theorem triangle_equilateral (m n p : ℝ) (h1 : m + n + p = 180) 
  (h2 : |m - n| + (n - p)^2 = 0) : m = n ∧ n = p := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l3858_385845


namespace NUMINAMATH_CALUDE_diagonal_cubes_200_420_480_l3858_385824

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: The number of cubes an internal diagonal passes through in a 200×420×480 rectangular solid is 1000 -/
theorem diagonal_cubes_200_420_480 :
  diagonal_cubes 200 420 480 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cubes_200_420_480_l3858_385824


namespace NUMINAMATH_CALUDE_certain_number_base_l3858_385861

theorem certain_number_base (x y : ℕ) (a : ℝ) 
  (h1 : 3^x * a^y = 3^12) 
  (h2 : x - y = 12) 
  (h3 : x = 12) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_base_l3858_385861


namespace NUMINAMATH_CALUDE_frac_x_div_2_gt_0_is_linear_inequality_l3858_385874

/-- A linear inequality in one variable is of the form ax + b > 0 or ax + b < 0,
    where a and b are constants and a ≠ 0. --/
def IsLinearInequality (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ) (rel : ℝ → ℝ → Prop), a ≠ 0 ∧
    (rel = (· > ·) ∨ rel = (· < ·)) ∧
    (∀ x, f x ↔ rel (a * x + b) 0)

/-- The function f(x) = x/2 > 0 is a linear inequality in one variable. --/
theorem frac_x_div_2_gt_0_is_linear_inequality :
  IsLinearInequality (fun x => x / 2 > 0) :=
sorry

end NUMINAMATH_CALUDE_frac_x_div_2_gt_0_is_linear_inequality_l3858_385874


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3858_385865

theorem possible_values_of_a (x y a : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : x^3 + y^3 = a) 
  (eq3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3858_385865


namespace NUMINAMATH_CALUDE_volume_is_304_l3858_385841

/-- The volume of the described set of points -/
def total_volume (central_box : ℝ × ℝ × ℝ) (extension : ℝ) : ℝ :=
  let (l, w, h) := central_box
  let box_volume := l * w * h
  let bounding_boxes_volume := 2 * (l * w + l * h + w * h) * extension
  let edge_prism_volume := 2 * (l + w + h) * extension * extension
  box_volume + bounding_boxes_volume + edge_prism_volume

/-- The theorem stating that the total volume is 304 cubic units -/
theorem volume_is_304 :
  total_volume (2, 3, 4) 2 = 304 := by sorry

end NUMINAMATH_CALUDE_volume_is_304_l3858_385841


namespace NUMINAMATH_CALUDE_student_sample_size_l3858_385880

theorem student_sample_size :
  ∀ (total juniors sophomores freshmen seniors : ℕ),
  juniors = (26 * total) / 100 →
  sophomores = (25 * total) / 100 →
  seniors = 160 →
  freshmen = sophomores + 32 →
  total = freshmen + sophomores + juniors + seniors →
  total = 800 := by
sorry

end NUMINAMATH_CALUDE_student_sample_size_l3858_385880


namespace NUMINAMATH_CALUDE_average_temperature_proof_l3858_385813

theorem average_temperature_proof (tuesday wednesday thursday friday : ℝ) : 
  (tuesday + wednesday + thursday) / 3 = 32 →
  friday = 44 →
  tuesday = 38 →
  (wednesday + thursday + friday) / 3 = 34 := by
sorry

end NUMINAMATH_CALUDE_average_temperature_proof_l3858_385813


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l3858_385875

-- Define the points
def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (4, 1)
def D : ℝ × ℝ := (-2, 4)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AD : ℝ × ℝ := (D.1 - A.1, D.2 - A.2)
def DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicular
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Define parallel
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- Define trapezoid
def is_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  parallel (B.1 - A.1, B.2 - A.2) (D.1 - C.1, D.2 - C.2) ∧
  ¬parallel (A.1 - D.1, A.2 - D.2) (B.1 - C.1, B.2 - C.2)

theorem quadrilateral_properties :
  perpendicular AB AD ∧ parallel AB DC ∧ is_trapezoid A B C D := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l3858_385875


namespace NUMINAMATH_CALUDE_unique_solution_3x_eq_12_l3858_385807

theorem unique_solution_3x_eq_12 : 
  ∀ (x₁ x₂ : ℝ), (3 : ℝ) ^ x₁ = 12 → (3 : ℝ) ^ x₂ = 12 → x₁ = x₂ := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3x_eq_12_l3858_385807


namespace NUMINAMATH_CALUDE_smallest_n_for_equal_candy_costs_l3858_385899

theorem smallest_n_for_equal_candy_costs : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p y o : ℕ), p > 0 ∧ y > 0 ∧ o > 0 ∧ 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * n) ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    ¬∃ (p y o : ℕ), p > 0 ∧ y > 0 ∧ o > 0 ∧ 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * m) ∧
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_equal_candy_costs_l3858_385899


namespace NUMINAMATH_CALUDE_no_factors_l3858_385894

def f (x : ℝ) : ℝ := x^4 + 2*x^2 + 9

def g₁ (x : ℝ) : ℝ := x^2 + 3
def g₂ (x : ℝ) : ℝ := x + 1
def g₃ (x : ℝ) : ℝ := x^2 - 3
def g₄ (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem no_factors : 
  (∀ x : ℝ, f x ≠ 0 → g₁ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₂ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₃ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₄ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_factors_l3858_385894


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3858_385804

theorem angle_measure_proof (x : ℝ) : 
  x + (3 * x - 10) = 180 → x = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3858_385804


namespace NUMINAMATH_CALUDE_jim_paycheck_amount_l3858_385843

/-- Calculates the final amount on a paycheck after retirement and tax deductions -/
def final_paycheck_amount (gross_pay : ℝ) (retirement_rate : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_rate) - tax_deduction

/-- Theorem stating that given the specific conditions, the final paycheck amount is $740 -/
theorem jim_paycheck_amount :
  final_paycheck_amount 1120 0.25 100 = 740 := by
  sorry

#eval final_paycheck_amount 1120 0.25 100

end NUMINAMATH_CALUDE_jim_paycheck_amount_l3858_385843


namespace NUMINAMATH_CALUDE_decimal_to_binary_conversion_l3858_385826

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- The decimal number to be converted -/
def decimal_number : ℕ := 2016

/-- The expected binary representation -/
def expected_binary : List Bool := [true, true, true, true, true, false, false, false, false, false, false]

theorem decimal_to_binary_conversion :
  to_binary decimal_number = expected_binary := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_conversion_l3858_385826


namespace NUMINAMATH_CALUDE_solve_for_t_l3858_385825

theorem solve_for_t (s t u : ℚ) 
  (eq1 : 12 * s + 6 * t + 3 * u = 180)
  (eq2 : t = s + 2)
  (eq3 : t = u + 3) :
  t = 213 / 21 := by
sorry

end NUMINAMATH_CALUDE_solve_for_t_l3858_385825


namespace NUMINAMATH_CALUDE_family_reunion_handshakes_count_l3858_385827

/-- Represents the number of handshakes at a family reunion --/
def family_reunion_handshakes : ℕ :=
  let twin_sets := 7
  let triplet_sets := 4
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2)
  let triplet_handshakes := triplets * (triplets - 3)
  let cross_handshakes := twins * (triplets / 3) + triplets * (twins / 4)
  (twin_handshakes + triplet_handshakes + cross_handshakes) / 2

/-- Theorem stating that the number of handshakes at the family reunion is 184 --/
theorem family_reunion_handshakes_count : family_reunion_handshakes = 184 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_handshakes_count_l3858_385827


namespace NUMINAMATH_CALUDE_min_omega_for_cos_symmetry_l3858_385820

theorem min_omega_for_cos_symmetry (ω : ℕ+) : 
  (∃ k : ℤ, ω = 6 * k + 2) → 
  (∀ ω' : ℕ+, (∃ k' : ℤ, ω' = 6 * k' + 2) → ω ≤ ω') → 
  ω = 2 := by sorry

end NUMINAMATH_CALUDE_min_omega_for_cos_symmetry_l3858_385820


namespace NUMINAMATH_CALUDE_reciprocal_of_two_l3858_385893

theorem reciprocal_of_two (m : ℚ) : m - 3 = 1 / 2 → m = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_l3858_385893


namespace NUMINAMATH_CALUDE_polynomial_is_perfect_square_l3858_385878

theorem polynomial_is_perfect_square (x : ℝ) : 
  ∃ (t u : ℝ), (49/4 : ℝ) * x^2 + 21 * x + 9 = (t * x + u)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_is_perfect_square_l3858_385878


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_l3858_385849

-- Problem 1
theorem problem_1 : (1) - 27 + (-32) + (-8) + 27 = -40 := by sorry

-- Problem 2
theorem problem_2 : (2) * (-5) + |(-3)| = -2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℤ) (h1 : -x = 3) (h2 : |y| = 5) : 
  x + y = 2 ∨ x + y = -8 := by sorry

-- Problem 4
theorem problem_4 : (-1 - 1/2) + (1 + 1/4) + (-2 - 1/2) - (-3 - 1/4) - (1 + 1/4) = -3/4 := by sorry

-- Problem 5
theorem problem_5 (a b : ℝ) (h : |a - 4| + |b + 5| = 0) : a - b = 9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_l3858_385849


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l3858_385832

/-- Represents the game setup with total boxes and high-value boxes -/
structure GameSetup where
  total_boxes : ℕ
  high_value_boxes : ℕ

/-- Calculates the probability of holding a high-value box -/
def probability_high_value (g : GameSetup) (eliminated : ℕ) : ℚ :=
  g.high_value_boxes / (g.total_boxes - eliminated)

/-- Theorem stating that eliminating 7 boxes results in at least 50% chance of high-value box -/
theorem deal_or_no_deal_probability 
  (g : GameSetup) 
  (h1 : g.total_boxes = 30) 
  (h2 : g.high_value_boxes = 8) : 
  probability_high_value g 7 ≥ 1/2 := by
  sorry

#eval probability_high_value ⟨30, 8⟩ 7

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l3858_385832


namespace NUMINAMATH_CALUDE_expression_equality_l3858_385844

theorem expression_equality (x y : ℝ) (h : 2 * x - 3 * y = 1) : 
  10 - 4 * x + 6 * y = 8 := by
sorry

end NUMINAMATH_CALUDE_expression_equality_l3858_385844


namespace NUMINAMATH_CALUDE_set_equality_implies_m_zero_l3858_385853

theorem set_equality_implies_m_zero (m : ℝ) : 
  ({3, m} : Set ℝ) = ({3 * m, 3} : Set ℝ) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_zero_l3858_385853


namespace NUMINAMATH_CALUDE_snake_shedding_decimal_l3858_385879

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

/-- The number of ways a snake can shed its skin in octal --/
def snake_shedding_octal : ℕ := 453

theorem snake_shedding_decimal :
  octal_to_decimal snake_shedding_octal = 299 := by
  sorry

end NUMINAMATH_CALUDE_snake_shedding_decimal_l3858_385879


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3858_385847

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n}, if a_1 + a_19 = 10, then a_10 = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 1 + a 19 = 10) : 
  a 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3858_385847


namespace NUMINAMATH_CALUDE_sum_of_modified_numbers_l3858_385850

theorem sum_of_modified_numbers (R x y : ℝ) (h : x + y = R) :
  2 * (x + 4) + 2 * (y + 5) = 2 * R + 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_modified_numbers_l3858_385850


namespace NUMINAMATH_CALUDE_quadrilateral_JMIT_cyclic_l3858_385848

structure Triangle (α : Type*) [Field α] where
  a : α
  b : α
  c : α

def incenter {α : Type*} [Field α] (t : Triangle α) : α :=
  -(t.a * t.b + t.b * t.c + t.c * t.a)

def excenter {α : Type*} [Field α] (t : Triangle α) : α :=
  t.a * t.b - t.b * t.c + t.c * t.a

def midpoint_BC {α : Type*} [Field α] (t : Triangle α) : α :=
  (t.b^2 + t.c^2) / 2

def symmetric_point {α : Type*} [Field α] (t : Triangle α) : α :=
  2 * t.a^2 - t.b * t.c

def is_cyclic {α : Type*} [Field α] (a b c d : α) : Prop :=
  ∃ (k : α), k ≠ 0 ∧ (b - a) * (d - c) = k * (c - a) * (d - b)

theorem quadrilateral_JMIT_cyclic {α : Type*} [Field α] (t : Triangle α) 
  (h1 : t.a^2 ≠ 0) (h2 : t.b^2 ≠ 0) (h3 : t.c^2 ≠ 0)
  (h4 : t.a^2 * t.a^2 = 1) (h5 : t.b^2 * t.b^2 = 1) (h6 : t.c^2 * t.c^2 = 1) :
  is_cyclic (excenter t) (midpoint_BC t) (incenter t) (symmetric_point t) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_JMIT_cyclic_l3858_385848


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3858_385869

theorem arithmetic_calculation : 28 * 7 * 25 + 12 * 7 * 25 + 7 * 11 * 3 + 44 = 7275 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3858_385869


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l3858_385851

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l3858_385851


namespace NUMINAMATH_CALUDE_rice_grains_difference_l3858_385891

def grains_on_square (k : ℕ) : ℕ := 2^k

def sum_of_first_n_squares (n : ℕ) : ℕ :=
  (List.range n).map grains_on_square |>.sum

theorem rice_grains_difference :
  grains_on_square 12 - sum_of_first_n_squares 10 = 2050 := by
  sorry

end NUMINAMATH_CALUDE_rice_grains_difference_l3858_385891


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l3858_385866

/-- Given two parabolas, prove that their intersection points lie on a circle with radius squared equal to 16 -/
theorem intersection_points_on_circle (x y : ℝ) : 
  y = (x - 2)^2 ∧ x = (y - 5)^2 - 1 → (x - 2)^2 + (y - 5)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l3858_385866


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l3858_385805

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 6*x - 5 < 3 - 2*x :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l3858_385805


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l3858_385882

theorem quadratic_function_a_range (a b c : ℝ) : 
  a ≠ 0 →
  a * (-1)^2 + b * (-1) + c = 3 →
  a * 1^2 + b * 1 + c = 1 →
  0 < c →
  c < 1 →
  1 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l3858_385882


namespace NUMINAMATH_CALUDE_johann_oranges_l3858_385809

def orange_problem (initial_oranges eaten_oranges stolen_fraction returned_oranges : ℕ) : Prop :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen := (remaining_after_eating / 2 : ℕ)
  let final_count := remaining_after_eating - stolen + returned_oranges
  final_count = 30

theorem johann_oranges :
  orange_problem 60 10 2 5 := by sorry

end NUMINAMATH_CALUDE_johann_oranges_l3858_385809


namespace NUMINAMATH_CALUDE_fayes_rows_l3858_385881

theorem fayes_rows (pencils_per_row : ℕ) (crayons_per_row : ℕ) (total_items : ℕ) : 
  pencils_per_row = 31 →
  crayons_per_row = 27 →
  total_items = 638 →
  total_items / (pencils_per_row + crayons_per_row) = 11 := by
sorry

end NUMINAMATH_CALUDE_fayes_rows_l3858_385881


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l3858_385801

theorem polynomial_perfect_square (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5*x + 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l3858_385801


namespace NUMINAMATH_CALUDE_remainder_7645_div_9_l3858_385890

theorem remainder_7645_div_9 : 7645 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7645_div_9_l3858_385890


namespace NUMINAMATH_CALUDE_junior_score_l3858_385830

theorem junior_score (n : ℝ) (h_pos : n > 0) : 
  let junior_ratio : ℝ := 0.2
  let senior_ratio : ℝ := 0.8
  let class_average : ℝ := 78
  let senior_average : ℝ := 75
  let junior_count : ℝ := junior_ratio * n
  let senior_count : ℝ := senior_ratio * n
  let total_score : ℝ := class_average * n
  let senior_total_score : ℝ := senior_average * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_junior_score_l3858_385830


namespace NUMINAMATH_CALUDE_min_value_is_neg_one_l3858_385803

/-- The system of equations and inequalities -/
def system (x y : ℝ) : Prop :=
  3^(-x) * y^4 - 2*y^2 + 3^x ≤ 0 ∧ 27^x + y^4 - 3^x - 1 = 0

/-- The expression to be minimized -/
def expression (x y : ℝ) : ℝ := x^3 + y^3

/-- The theorem stating that the minimum value of the expression is -1 -/
theorem min_value_is_neg_one :
  ∃ (x y : ℝ), system x y ∧
  ∀ (a b : ℝ), system a b → expression x y ≤ expression a b ∧
  expression x y = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_is_neg_one_l3858_385803


namespace NUMINAMATH_CALUDE_arctan_tan_sum_equals_angle_l3858_385818

theorem arctan_tan_sum_equals_angle (θ : Real) : 
  θ ≥ 0 ∧ θ ≤ π / 2 → Real.arctan (Real.tan θ + 3 * Real.tan (π / 12)) = θ := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_sum_equals_angle_l3858_385818


namespace NUMINAMATH_CALUDE_safe_combination_l3858_385864

/-- Represents a digit in base 10 -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (n i m a : Digit) : Prop :=
  n ≠ i ∧ n ≠ m ∧ n ≠ a ∧ i ≠ m ∧ i ≠ a ∧ m ≠ a

/-- Converts a three-digit number in base 10 to its decimal value -/
def toDecimal (n i m : Digit) : Nat :=
  100 * n.val + 10 * i.val + m.val

/-- Checks if the equation NIM + AM + MIA = MINA holds in base 10 -/
def equationHolds (n i m a : Digit) : Prop :=
  (100 * n.val + 10 * i.val + m.val) +
  (10 * a.val + m.val) +
  (100 * m.val + 10 * i.val + a.val) =
  (1000 * m.val + 100 * i.val + 10 * n.val + a.val)

theorem safe_combination :
  ∃! (n i m a : Digit), distinct n i m a ∧
  equationHolds n i m a ∧
  toDecimal n i m = 845 := by
sorry

end NUMINAMATH_CALUDE_safe_combination_l3858_385864


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l3858_385858

theorem smallest_b_in_arithmetic_sequence (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- All terms are positive
  a = b - d →              -- a is the first term
  c = b + d →              -- c is the third term
  a * b * c = 125 →        -- Product of terms is 125
  ∀ x : ℝ, x > 0 ∧ x < b → ¬∃ y : ℝ, 
    (x - y) > 0 ∧ (x + y) > 0 ∧ (x - y) * x * (x + y) = 125 →
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l3858_385858


namespace NUMINAMATH_CALUDE_equivalent_operation_l3858_385839

theorem equivalent_operation (x : ℝ) : (x / (5/6)) * (4/7) = x * (24/35) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l3858_385839


namespace NUMINAMATH_CALUDE_car_speed_problem_l3858_385810

/-- Proves that if a car traveling at 94.73684210526315 km/h takes 2 seconds longer to travel 1 kilometer
    compared to a certain faster speed, then that faster speed is 90 km/h. -/
theorem car_speed_problem (current_speed : ℝ) (faster_speed : ℝ) : 
  current_speed = 94.73684210526315 →
  (1 / current_speed) * 3600 = (1 / faster_speed) * 3600 + 2 →
  faster_speed = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3858_385810


namespace NUMINAMATH_CALUDE_stock_worth_calculation_l3858_385817

theorem stock_worth_calculation (W : ℝ) 
  (h1 : 0.02 * W - 0.024 * W = -400) : W = 100000 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_calculation_l3858_385817


namespace NUMINAMATH_CALUDE_high_correlation_implies_r_close_to_one_l3858_385860

/-- Represents the correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Represents the degree of linear correlation between two variables -/
def linear_correlation_degree (x y : ℝ → ℝ) : ℝ := sorry

/-- A high degree of linear correlation -/
def high_correlation : ℝ := sorry

theorem high_correlation_implies_r_close_to_one (x y : ℝ → ℝ) :
  linear_correlation_degree x y ≥ high_correlation →
  ∀ ε > 0, ∃ δ > 0, linear_correlation_degree x y > 1 - δ →
  |correlation_coefficient x y| > 1 - ε :=
sorry

end NUMINAMATH_CALUDE_high_correlation_implies_r_close_to_one_l3858_385860


namespace NUMINAMATH_CALUDE_special_trapezoid_not_isosceles_l3858_385819

/-- A trapezoid with the given properties --/
structure SpecialTrapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal : ℝ
  is_trapezoid : base1 ≠ base2
  base_values : base1 = 3 ∧ base2 = 4
  diagonal_length : diagonal = 6
  diagonal_bisects_angle : Bool

/-- Theorem stating that a trapezoid with the given properties cannot be isosceles --/
theorem special_trapezoid_not_isosceles (t : SpecialTrapezoid) : 
  ¬(∃ (side : ℝ), side > 0 ∧ t.base1 < t.base2 → 
    (side = t.diagonal ∧ side^2 = (t.base2 - t.base1)^2 / 4 + side^2 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_not_isosceles_l3858_385819


namespace NUMINAMATH_CALUDE_team_composition_proof_l3858_385823

theorem team_composition_proof (x y : ℕ) (h1 : x > 0) (h2 : y > 0) : 
  (22 * x + 47 * y) / (x + y) = 41 → x / (x + y) = 6 / 25 :=
by
  sorry

end NUMINAMATH_CALUDE_team_composition_proof_l3858_385823


namespace NUMINAMATH_CALUDE_abs_negative_six_l3858_385811

theorem abs_negative_six : |(-6 : ℤ)| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_six_l3858_385811


namespace NUMINAMATH_CALUDE_multiple_of_24_multiple_of_3_and_8_six_hundred_is_multiple_of_24_l3858_385812

theorem multiple_of_24 : ∃ (n : ℕ), 600 = 24 * n := by
  sorry

theorem multiple_of_3_and_8 (x : ℕ) : x % 24 = 0 ↔ x % 3 = 0 ∧ x % 8 = 0 := by
  sorry

theorem six_hundred_is_multiple_of_24 : 600 % 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_24_multiple_of_3_and_8_six_hundred_is_multiple_of_24_l3858_385812


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l3858_385829

theorem expression_equals_negative_one
  (a b y : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ a)
  (hy1 : y ≠ a)
  (hy2 : y ≠ -a) :
  (((a + b) / (a + y) + y / (a - y)) /
   ((y + b) / (a + y) - a / (a - y)) = -1) ↔
  (y = a - b) :=
sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l3858_385829


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l3858_385884

theorem triangle_area_inequality (a b c α β γ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α > 0) (hβ : β > 0) (hγ : γ > 0)
  (hα_def : α = 2 * Real.sqrt (b * c))
  (hβ_def : β = 2 * Real.sqrt (c * a))
  (hγ_def : γ = 2 * Real.sqrt (a * b)) :
  a / α + b / β + c / γ ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l3858_385884


namespace NUMINAMATH_CALUDE_equal_temperature_proof_l3858_385871

/-- The temperature at which Fahrenheit and Celsius scales are equal -/
def equal_temperature : ℚ := -40

/-- The relation between Fahrenheit (f) and Celsius (c) temperatures -/
def fahrenheit_celsius_relation (c : ℚ) : ℚ := (9/5) * c + 32

/-- Theorem stating that the equal_temperature is the point where Fahrenheit and Celsius scales meet -/
theorem equal_temperature_proof :
  fahrenheit_celsius_relation equal_temperature = equal_temperature := by
  sorry

end NUMINAMATH_CALUDE_equal_temperature_proof_l3858_385871


namespace NUMINAMATH_CALUDE_exp_ln_one_equals_one_l3858_385867

theorem exp_ln_one_equals_one : Real.exp (Real.log 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_exp_ln_one_equals_one_l3858_385867


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l3858_385857

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 -/
def linear_coefficient (a b c : ℚ) : ℚ := b

/-- The quadratic equation 2x^2 - 3x - 4 = 0 -/
def quadratic_equation (x : ℚ) : Prop := 2 * x^2 - 3 * x - 4 = 0

theorem linear_coefficient_of_quadratic :
  linear_coefficient 2 (-3) (-4) = -3 := by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l3858_385857


namespace NUMINAMATH_CALUDE_gratuity_calculation_correct_l3858_385838

/-- Calculates the gratuity for a restaurant bill given the individual dish prices,
    discount rate, sales tax rate, and tip rate. -/
def calculate_gratuity (prices : List ℝ) (discount_rate sales_tax_rate tip_rate : ℝ) : ℝ :=
  let total_before_discount := prices.sum
  let discounted_total := total_before_discount * (1 - discount_rate)
  let total_with_tax := discounted_total * (1 + sales_tax_rate)
  total_with_tax * tip_rate

/-- The gratuity calculated for the given restaurant bill is correct. -/
theorem gratuity_calculation_correct :
  let prices := [21, 15, 26, 13, 20]
  let discount_rate := 0.15
  let sales_tax_rate := 0.08
  let tip_rate := 0.18
  calculate_gratuity prices discount_rate sales_tax_rate tip_rate = 15.70 := by
  sorry

#eval calculate_gratuity [21, 15, 26, 13, 20] 0.15 0.08 0.18

end NUMINAMATH_CALUDE_gratuity_calculation_correct_l3858_385838


namespace NUMINAMATH_CALUDE_even_composition_l3858_385863

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_composition (g : ℝ → ℝ) (h : IsEven g) : IsEven (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_even_composition_l3858_385863


namespace NUMINAMATH_CALUDE_fraction_simplification_l3858_385883

theorem fraction_simplification : 48 / (7 - 3/4) = 192/25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3858_385883


namespace NUMINAMATH_CALUDE_present_cost_difference_l3858_385895

theorem present_cost_difference (cost_first cost_second cost_third : ℕ) : 
  cost_first = 18 →
  cost_third = cost_first - 11 →
  cost_first + cost_second + cost_third = 50 →
  cost_second > cost_first →
  cost_second - cost_first = 7 := by
sorry

end NUMINAMATH_CALUDE_present_cost_difference_l3858_385895


namespace NUMINAMATH_CALUDE_paint_set_cost_l3858_385834

def total_cost (has : ℝ) (needs : ℝ) : ℝ := has + needs
def paintbrush_cost : ℝ := 1.50
def easel_cost : ℝ := 12.65
def albert_has : ℝ := 6.50
def albert_needs : ℝ := 12.00

theorem paint_set_cost :
  total_cost albert_has albert_needs - (paintbrush_cost + easel_cost) = 4.35 := by
  sorry

end NUMINAMATH_CALUDE_paint_set_cost_l3858_385834


namespace NUMINAMATH_CALUDE_hash_four_one_l3858_385892

-- Define the # operation
def hash (a b : ℤ) : ℤ := (a + b + 2) * (a - b - 2)

-- Theorem statement
theorem hash_four_one : hash 4 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_one_l3858_385892


namespace NUMINAMATH_CALUDE_fourth_house_number_l3858_385870

theorem fourth_house_number (x : ℕ) (k : ℕ) : 
  k ≥ 4 → 
  (k + 1) * (x + k) = 78 → 
  x + 6 = 14 :=
by sorry

end NUMINAMATH_CALUDE_fourth_house_number_l3858_385870


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3858_385846

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1 ∧ c = b + 1 ∧ a * b * c = 990) → a + b + c = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3858_385846


namespace NUMINAMATH_CALUDE_chicken_eggs_l3858_385872

/-- The number of eggs laid by a chicken over two days -/
def total_eggs (today : ℕ) (yesterday : ℕ) : ℕ := today + yesterday

/-- Theorem: The chicken laid 49 eggs in total over two days -/
theorem chicken_eggs : total_eggs 30 19 = 49 := by
  sorry

end NUMINAMATH_CALUDE_chicken_eggs_l3858_385872


namespace NUMINAMATH_CALUDE_number_equation_l3858_385855

theorem number_equation (y : ℝ) : y = (1 / y) * (-y) - 5 → y = -6 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3858_385855


namespace NUMINAMATH_CALUDE_a_3_range_l3858_385888

def is_convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a n + a (n + 2)) / 2 ≤ a (n + 1)

def b (n : ℕ) : ℝ := n^2 - 6*n + 10

theorem a_3_range (a : ℕ → ℝ) :
  is_convex_sequence a →
  a 1 = 1 →
  a 10 = 28 →
  (∀ n : ℕ, 1 ≤ n → n < 10 → |a n - b n| ≤ 20) →
  7 ≤ a 3 ∧ a 3 ≤ 19 := by
sorry

end NUMINAMATH_CALUDE_a_3_range_l3858_385888


namespace NUMINAMATH_CALUDE_smallest_integer_l3858_385868

theorem smallest_integer (a b : ℕ) (x : ℕ) (h1 : b = 18) (h2 : x > 0)
  (h3 : Nat.gcd a b = x + 3) (h4 : Nat.lcm a b = x * (x + 3)) :
  ∃ (a_min : ℕ), a_min = 6 ∧ ∀ (a' : ℕ), (∃ (x' : ℕ), x' > 0 ∧
    Nat.gcd a' b = x' + 3 ∧ Nat.lcm a' b = x' * (x' + 3)) → a' ≥ a_min :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l3858_385868


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l3858_385821

/-- Given a triangle XYZ with sides x, y, and z satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_of_triangle (x y z : ℝ) (h1 : x + 3*y + 4*z = x^2) (h2 : x + 3*y - 4*z = -7) :
  ∃ (X Y Z : ℝ), X + Y + Z = 180 ∧ 0 < X ∧ 0 < Y ∧ 0 < Z ∧ max X (max Y Z) = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l3858_385821


namespace NUMINAMATH_CALUDE_triangle_area_from_intersecting_lines_triangle_area_from_intersecting_lines_proof_l3858_385854

/-- Given two lines intersecting at P(1,6), one with slope 1 and the other with slope 2,
    the area of the triangle formed by P and the x-intercepts of these lines is 9 square units. -/
theorem triangle_area_from_intersecting_lines : ℝ → Prop :=
  fun area =>
    let P : ℝ × ℝ := (1, 6)
    let slope1 : ℝ := 1
    let slope2 : ℝ := 2
    let Q : ℝ × ℝ := (P.1 - P.2 / slope1, 0)  -- x-intercept of line with slope 1
    let R : ℝ × ℝ := (P.1 - P.2 / slope2, 0)  -- x-intercept of line with slope 2
    let base : ℝ := R.1 - Q.1
    let height : ℝ := P.2
    area = (1/2) * base * height ∧ area = 9

/-- Proof of the theorem -/
theorem triangle_area_from_intersecting_lines_proof : 
  ∃ (area : ℝ), triangle_area_from_intersecting_lines area :=
by
  sorry

#check triangle_area_from_intersecting_lines
#check triangle_area_from_intersecting_lines_proof

end NUMINAMATH_CALUDE_triangle_area_from_intersecting_lines_triangle_area_from_intersecting_lines_proof_l3858_385854


namespace NUMINAMATH_CALUDE_correct_propositions_l3858_385802

-- Define the planes
variable (α β : Set (Point))

-- Define the property of being a plane
def is_plane (p : Set (Point)) : Prop := sorry

-- Define the property of being distinct
def distinct (p q : Set (Point)) : Prop := p ≠ q

-- Define line
def Line : Type := sorry

-- Define the property of a line being within a plane
def line_in_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define perpendicularity between lines
def perp_lines (l1 l2 : Line) : Prop := sorry

-- Define perpendicularity between a line and a plane
def perp_line_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define perpendicularity between planes
def perp_planes (p q : Set (Point)) : Prop := sorry

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define parallelism between planes
def parallel_planes (p q : Set (Point)) : Prop := sorry

-- State the theorem
theorem correct_propositions 
  (h_planes : is_plane α ∧ is_plane β) 
  (h_distinct : distinct α β) :
  (∀ (l : Line), line_in_plane l α → 
    (∀ (m : Line), line_in_plane m β → perp_lines l m) → 
    perp_planes α β) ∧ 
  (∀ (l : Line), line_in_plane l α → 
    parallel_line_plane l β → 
    parallel_planes α β) ∧ 
  (perp_planes α β → 
    ∃ (l : Line), line_in_plane l α ∧ ¬(perp_line_plane l β)) ∧
  (parallel_planes α β → 
    ∀ (l : Line), line_in_plane l α → 
    parallel_line_plane l β) :=
sorry

end NUMINAMATH_CALUDE_correct_propositions_l3858_385802


namespace NUMINAMATH_CALUDE_largest_number_l3858_385816

theorem largest_number : Real.sqrt 2 = max (max (max (-3) 0) 1) (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3858_385816


namespace NUMINAMATH_CALUDE_blue_garden_yield_l3858_385877

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected carrot yield from a rectangular garden -/
def expectedCarrotYield (garden : GardenDimensions) (stepLength : ℝ) (yieldPerSqFt : ℝ) : ℝ :=
  (garden.length : ℝ) * stepLength * (garden.width : ℝ) * stepLength * yieldPerSqFt

/-- Theorem stating the expected carrot yield for Mr. Blue's garden -/
theorem blue_garden_yield :
  let garden : GardenDimensions := ⟨18, 25⟩
  let stepLength : ℝ := 3
  let yieldPerSqFt : ℝ := 3 / 4
  expectedCarrotYield garden stepLength yieldPerSqFt = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_blue_garden_yield_l3858_385877


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l3858_385808

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_34 : ℚ := 34 / 99

theorem product_of_repeating_decimals :
  repeating_decimal_12 * repeating_decimal_34 = 136 / 3267 :=
by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l3858_385808


namespace NUMINAMATH_CALUDE_point_outside_circle_l3858_385842

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 24}
  P ∉ circle ∧ ∀ (x y : ℝ), (x, y) ∈ circle → (m^2 - x)^2 + (5 - y)^2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3858_385842


namespace NUMINAMATH_CALUDE_factory_production_rate_l3858_385897

/-- Represents a chocolate factory's production parameters and calculates the hourly production rate. -/
def ChocolateFactory (total_candies : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  total_candies / (days * hours_per_day)

/-- Theorem stating that for the given production parameters, the factory produces 50 candies per hour. -/
theorem factory_production_rate :
  ChocolateFactory 4000 8 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_rate_l3858_385897


namespace NUMINAMATH_CALUDE_playground_fence_posts_l3858_385887

/-- Calculates the number of fence posts required for a rectangular playground -/
def fence_posts (width : ℕ) (length : ℕ) (post_interval : ℕ) : ℕ :=
  let long_side_posts := length / post_interval + 2
  let short_side_posts := width / post_interval + 1
  long_side_posts + 2 * short_side_posts

/-- Theorem stating the number of fence posts for a 50m by 90m playground -/
theorem playground_fence_posts :
  fence_posts 50 90 10 = 25 := by
  sorry

#eval fence_posts 50 90 10

end NUMINAMATH_CALUDE_playground_fence_posts_l3858_385887
