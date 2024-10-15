import Mathlib

namespace NUMINAMATH_CALUDE_greatest_prime_factor_f_24_l1668_166811

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2)) (fun i => 2 * (i + 1))

theorem greatest_prime_factor_f_24 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ f 24 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ f 24 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_f_24_l1668_166811


namespace NUMINAMATH_CALUDE_point_on_inverse_proportion_graph_l1668_166877

theorem point_on_inverse_proportion_graph :
  let f : ℝ → ℝ := λ x => 6 / x
  f 2 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_inverse_proportion_graph_l1668_166877


namespace NUMINAMATH_CALUDE_rectangle_garden_length_l1668_166857

/-- The perimeter of a rectangle -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangular garden with perimeter 1800 m and breadth 400 m, the length is 500 m -/
theorem rectangle_garden_length (p b : ℝ) (h1 : p = 1800) (h2 : b = 400) :
  ∃ l : ℝ, perimeter l b = p ∧ l = 500 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_garden_length_l1668_166857


namespace NUMINAMATH_CALUDE_vector_magnitude_l1668_166847

theorem vector_magnitude (a b : ℝ × ℝ) : 
  (3 • a - 2 • b) • (5 • a + b) = 0 → 
  a • b = 1/7 → 
  ‖a‖ = 1 → 
  ‖b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1668_166847


namespace NUMINAMATH_CALUDE_manoj_lending_problem_l1668_166831

/-- Calculates simple interest -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem manoj_lending_problem (borrowed : ℚ) (borrowRate : ℚ) (lendRate : ℚ) (time : ℚ) (totalGain : ℚ)
  (h1 : borrowed = 3900)
  (h2 : borrowRate = 6)
  (h3 : lendRate = 9)
  (h4 : time = 3)
  (h5 : totalGain = 824.85)
  : ∃ (lentSum : ℚ), 
    lentSum = 5655 ∧ 
    simpleInterest lentSum lendRate time - simpleInterest borrowed borrowRate time = totalGain :=
sorry

end NUMINAMATH_CALUDE_manoj_lending_problem_l1668_166831


namespace NUMINAMATH_CALUDE_rahul_share_l1668_166862

/-- Calculates the share of a worker given the total payment and the time taken by both workers --/
def calculateShare (totalPayment : ℚ) (time1 : ℚ) (time2 : ℚ) : ℚ :=
  let combinedRate := 1 / time1 + 1 / time2
  let share := (1 / time1) / combinedRate
  share * totalPayment

/-- Proves that Rahul's share is $60 given the problem conditions --/
theorem rahul_share :
  let rahulTime : ℚ := 3
  let rajeshTime : ℚ := 2
  let totalPayment : ℚ := 150
  calculateShare totalPayment rahulTime rajeshTime = 60 := by
sorry

#eval calculateShare 150 3 2

end NUMINAMATH_CALUDE_rahul_share_l1668_166862


namespace NUMINAMATH_CALUDE_paint_mixing_l1668_166869

/-- Represents the mixing of two paints to achieve a target yellow percentage -/
theorem paint_mixing (light_green_volume : ℝ) (light_green_yellow_percent : ℝ)
  (dark_green_yellow_percent : ℝ) (target_yellow_percent : ℝ) :
  light_green_volume = 5 →
  light_green_yellow_percent = 0.2 →
  dark_green_yellow_percent = 0.4 →
  target_yellow_percent = 0.25 →
  ∃ dark_green_volume : ℝ,
    dark_green_volume = 5 / 3 ∧
    (light_green_volume * light_green_yellow_percent + dark_green_volume * dark_green_yellow_percent) /
      (light_green_volume + dark_green_volume) = target_yellow_percent :=
by sorry

end NUMINAMATH_CALUDE_paint_mixing_l1668_166869


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1668_166852

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1668_166852


namespace NUMINAMATH_CALUDE_ram_price_decrease_l1668_166838

theorem ram_price_decrease (initial_price increased_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : increased_price = initial_price * 1.3)
  (h3 : final_price = 52) :
  (increased_price - final_price) / increased_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_ram_price_decrease_l1668_166838


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1668_166823

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of a line in the form y = mx + b is m -/
def slope_of_line (m b : ℝ) : ℝ := m

/-- The slope of a line in the form ax + y + c = 0 is -a -/
def slope_of_general_line (a c : ℝ) : ℝ := -a

theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope_of_general_line a (-5)) (slope_of_line 7 (-2)) → a = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1668_166823


namespace NUMINAMATH_CALUDE_second_number_proof_l1668_166897

theorem second_number_proof (first second third : ℝ) : 
  first = 6 → 
  third = 22 → 
  (first + second + third) / 3 = 13 → 
  second = 11 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l1668_166897


namespace NUMINAMATH_CALUDE_craigs_remaining_apples_l1668_166809

/-- Calculates the number of apples Craig has after sharing -/
def craigs_apples_after_sharing (initial_apples : ℕ) (shared_apples : ℕ) : ℕ :=
  initial_apples - shared_apples

theorem craigs_remaining_apples :
  craigs_apples_after_sharing 20 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_craigs_remaining_apples_l1668_166809


namespace NUMINAMATH_CALUDE_fourth_root_of_46656000_l1668_166868

theorem fourth_root_of_46656000 : (46656000 : ℝ) ^ (1/4 : ℝ) = 216 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_46656000_l1668_166868


namespace NUMINAMATH_CALUDE_star_properties_l1668_166807

/-- Custom binary operation ※ -/
def star (x y : ℚ) : ℚ := x * y + 1

/-- Theorem stating the properties of the ※ operation -/
theorem star_properties :
  (star 2 4 = 9) ∧
  (star (star 1 4) (-2) = -9) ∧
  (∀ a b c : ℚ, star a (b + c) + 1 = star a b + star a c) :=
by sorry

end NUMINAMATH_CALUDE_star_properties_l1668_166807


namespace NUMINAMATH_CALUDE_evaluate_nested_fraction_l1668_166864

theorem evaluate_nested_fraction : 1 - (1 / (1 - (1 / (1 + 2)))) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_evaluate_nested_fraction_l1668_166864


namespace NUMINAMATH_CALUDE_existence_of_xyz_l1668_166898

theorem existence_of_xyz (n : ℕ) (hn : n > 0) :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^(n-1) + y^n = z^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xyz_l1668_166898


namespace NUMINAMATH_CALUDE_perimeter_specific_midpoint_triangle_l1668_166850

/-- A solid right prism with regular hexagonal bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  edge : String

/-- Triangle formed by three midpoints -/
structure MidpointTriangle where
  point1 : Midpoint
  point2 : Midpoint
  point3 : Midpoint

/-- Calculate the perimeter of the midpoint triangle -/
def perimeter_midpoint_triangle (prism : RightPrism) (triangle : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific midpoint triangle -/
theorem perimeter_specific_midpoint_triangle :
  ∀ (prism : RightPrism) (triangle : MidpointTriangle),
  prism.height = 20 ∧ 
  prism.base_side_length = 10 ∧
  triangle.point1 = Midpoint.mk "AB" ∧
  triangle.point2 = Midpoint.mk "BC" ∧
  triangle.point3 = Midpoint.mk "EF" →
  perimeter_midpoint_triangle prism triangle = 45 :=
sorry

end NUMINAMATH_CALUDE_perimeter_specific_midpoint_triangle_l1668_166850


namespace NUMINAMATH_CALUDE_twenty_thousand_scientific_notation_l1668_166891

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  prop : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_thousand_scientific_notation :
  toScientificNotation 20000 = ScientificNotation.mk 2 4 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_twenty_thousand_scientific_notation_l1668_166891


namespace NUMINAMATH_CALUDE_solution_equation_l1668_166865

theorem solution_equation (x : ℝ) (k : ℤ) : 
  (8.492 * (Real.log (Real.sin x) / Real.log (Real.sin x * Real.cos x)) * 
           (Real.log (Real.cos x) / Real.log (Real.sin x * Real.cos x)) = 1/4) →
  (Real.sin x > 0) →
  (x = π/4 * (8 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_equation_l1668_166865


namespace NUMINAMATH_CALUDE_question_selection_ways_l1668_166856

theorem question_selection_ways : 
  (Nat.choose 10 8) * (Nat.choose 10 5) = 11340 := by sorry

end NUMINAMATH_CALUDE_question_selection_ways_l1668_166856


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l1668_166851

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | |x| < 5}
def T : Set ℝ := {x : ℝ | (x + 7) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l1668_166851


namespace NUMINAMATH_CALUDE_constant_for_max_n_l1668_166804

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, c * n^2 ≤ 6400 → n ≤ 7) ∧ 
  (∃ n : ℤ, c * n^2 ≤ 6400 ∧ n = 7) →
  c = 6400 / 49 :=
sorry

end NUMINAMATH_CALUDE_constant_for_max_n_l1668_166804


namespace NUMINAMATH_CALUDE_fast_site_selection_probability_l1668_166827

theorem fast_site_selection_probability (total : ℕ) (guizhou : ℕ) (selected : ℕ)
  (h1 : total = 8)
  (h2 : guizhou = 3)
  (h3 : selected = 2)
  (h4 : guizhou ≤ total) :
  (Nat.choose guizhou 1 * Nat.choose (total - guizhou) 1 + Nat.choose guizhou 2) / Nat.choose total selected = 9 / 14 :=
by sorry

end NUMINAMATH_CALUDE_fast_site_selection_probability_l1668_166827


namespace NUMINAMATH_CALUDE_film_festival_selection_l1668_166854

/-- Given a film festival selection process, prove that the fraction of color films
    selected by the subcommittee is 20/21. -/
theorem film_festival_selection (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let total_films := 30 * x + 6 * y
  let bw_selected := (y / x) * (30 * x) / 100
  let color_selected := 6 * y
  let total_selected := bw_selected + color_selected
  color_selected / total_selected = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_film_festival_selection_l1668_166854


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1668_166816

/-- Given a hyperbola C and an ellipse with the following properties:
    1. C has the form x²/a² - y²/b² = 1 where a > 0 and b > 0
    2. C has an asymptote with equation y = (√5/2)x
    3. C shares a common focus with the ellipse x²/12 + y²/3 = 1
    Then the equation of C is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), y = (Real.sqrt 5 / 2) * x) ∧
  (∃ (c : ℝ), c^2 = 3^2 ∧ c^2 = a^2 + b^2) →
  a^2 = 4 ∧ b^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1668_166816


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1668_166822

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1668_166822


namespace NUMINAMATH_CALUDE_sequences_properties_l1668_166806

def a (n : ℕ) : ℤ := (-3) ^ n
def b (n : ℕ) : ℤ := (-3) ^ n - 3
def c (n : ℕ) : ℤ := -(-3) ^ n - 1

def m (n : ℕ) : ℤ := a n + b n + c n

theorem sequences_properties :
  (a 5 = -243 ∧ b 5 = -246 ∧ c 5 = 242) ∧
  (∃ k : ℕ, a k + a (k + 1) + a (k + 2) = -1701) ∧
  (∀ n : ℕ,
    (n % 2 = 1 → max (a n) (max (b n) (c n)) - min (a n) (min (b n) (c n)) = -2 * m n - 6) ∧
    (n % 2 = 0 → max (a n) (max (b n) (c n)) - min (a n) (min (b n) (c n)) = 2 * m n + 9)) :=
by sorry

end NUMINAMATH_CALUDE_sequences_properties_l1668_166806


namespace NUMINAMATH_CALUDE_rational_square_plus_one_positive_l1668_166871

theorem rational_square_plus_one_positive (x : ℚ) : x^2 + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_plus_one_positive_l1668_166871


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1668_166808

theorem unique_positive_solution : ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The unique positive solution is 5/3
  use 5/3
  constructor
  · -- Prove that 5/3 satisfies the conditions
    constructor
    · -- Prove 5/3 > 0
      sorry
    · -- Prove 3 * (5/3)^2 + 7 * (5/3) - 20 = 0
      sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1668_166808


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1668_166836

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1668_166836


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_nonempty_solution_l1668_166846

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = {x : ℝ | 1/2 < x ∧ x < 5/2} :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a) ↔ a > 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_nonempty_solution_l1668_166846


namespace NUMINAMATH_CALUDE_cosine_of_arithmetic_sequence_l1668_166832

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem cosine_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = Real.pi) : 
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_arithmetic_sequence_l1668_166832


namespace NUMINAMATH_CALUDE_average_weight_increase_l1668_166876

/-- Proves that replacing a person weighing 45 kg with a person weighing 65 kg
    in a group of 8 people increases the average weight by 2.5 kg -/
theorem average_weight_increase (initial_group_size : ℕ) 
                                 (old_weight new_weight : ℝ) : 
  initial_group_size = 8 →
  old_weight = 45 →
  new_weight = 65 →
  (new_weight - old_weight) / initial_group_size = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1668_166876


namespace NUMINAMATH_CALUDE_function_properties_l1668_166872

noncomputable def m (a : ℝ) (t : ℝ) : ℝ := (1/2) * a * t^2 + t - a

noncomputable def g (a : ℝ) : ℝ :=
  if a > -1/2 then a + 2
  else if a > -Real.sqrt 2 / 2 then -a - 1 / (2 * a)
  else Real.sqrt 2

theorem function_properties (a : ℝ) :
  (∀ t : ℝ, Real.sqrt 2 ≤ t ∧ t ≤ 2 → 
    ∃ x : ℝ, m a t = a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)) ∧
  (∀ t : ℝ, Real.sqrt 2 ≤ t ∧ t ≤ 2 → m a t ≤ g a) ∧
  (a ≥ -Real.sqrt 2 → (g a = g (1/a) ↔ (-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2) ∨ a = 1)) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1668_166872


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l1668_166894

theorem quadratic_form_k_value : 
  ∃ (a h k : ℚ), ∀ x, x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l1668_166894


namespace NUMINAMATH_CALUDE_friday_temperature_l1668_166870

theorem friday_temperature
  (temp : Fin 5 → ℝ)
  (avg_mon_to_thu : (temp 0 + temp 1 + temp 2 + temp 3) / 4 = 48)
  (avg_tue_to_fri : (temp 1 + temp 2 + temp 3 + temp 4) / 4 = 46)
  (monday_temp : temp 0 = 42) :
  temp 4 = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l1668_166870


namespace NUMINAMATH_CALUDE_domain_of_g_l1668_166866

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the original domain of f
def original_domain : Set ℝ := Set.Icc 1 5

-- Define the new function g(x) = f(2x - 3)
def g (x : ℝ) : ℝ := f (2 * x - 3)

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l1668_166866


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_m_greater_than_one_l1668_166875

theorem quadratic_always_positive_implies_m_greater_than_one (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_m_greater_than_one_l1668_166875


namespace NUMINAMATH_CALUDE_problem_solution_l1668_166863

theorem problem_solution (x y : ℚ) : 
  x / y = 15 / 3 → y = 27 → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1668_166863


namespace NUMINAMATH_CALUDE_second_digit_of_n_l1668_166890

theorem second_digit_of_n (n : ℕ) : 
  (10^99 ≤ 8*n) ∧ (8*n < 10^100) ∧ 
  (10^101 ≤ 81*n - 102) ∧ (81*n - 102 < 10^102) →
  (n / 10^97) % 10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_second_digit_of_n_l1668_166890


namespace NUMINAMATH_CALUDE_intersection_condition_l1668_166896

/-- Curve C in the xy-plane -/
def C (x y : ℝ) : Prop := y^2 = 6*x - 2 ∧ y ≥ 0

/-- Line l in the xy-plane -/
def L (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + 2*m

/-- Intersection points of C and L -/
def Intersection (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C p.1 p.2 ∧ L p.1 p.2 m}

/-- Two distinct intersection points exist -/
def HasTwoDistinctIntersections (m : ℝ) : Prop :=
  ∃ p q : ℝ × ℝ, p ∈ Intersection m ∧ q ∈ Intersection m ∧ p ≠ q

theorem intersection_condition (m : ℝ) :
  HasTwoDistinctIntersections m ↔ -Real.sqrt 3 / 6 ≤ m ∧ m < Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1668_166896


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1668_166878

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → a = Real.sqrt 3 → b = 1 →
  (0 < A ∧ A < π) → (0 < B ∧ B < π) → (0 < C ∧ C < π) →
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  B = π/6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1668_166878


namespace NUMINAMATH_CALUDE_scientific_notation_of_111_3_billion_l1668_166861

theorem scientific_notation_of_111_3_billion : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 111300000000 = a * (10 : ℝ) ^ n ∧ a = 1.113 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_111_3_billion_l1668_166861


namespace NUMINAMATH_CALUDE_correct_ages_l1668_166886

/-- Represents the ages of family members -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  mother : ℕ

/-- Calculates the correct ages given the problem conditions -/
def calculateAges : FamilyAges :=
  let father := 44
  let son := father / 2
  let mother := son + 5
  { father := father, son := son, mother := mother }

/-- Theorem stating that the calculated ages satisfy the given conditions -/
theorem correct_ages (ages : FamilyAges := calculateAges) :
  ages.father = 44 ∧
  ages.father = ages.son + ages.son ∧
  ages.son - 5 = ages.mother - 10 ∧
  ages.father = 44 ∧
  ages.son = 22 ∧
  ages.mother = 27 :=
by sorry

end NUMINAMATH_CALUDE_correct_ages_l1668_166886


namespace NUMINAMATH_CALUDE_abs_equal_necessary_not_sufficient_l1668_166883

theorem abs_equal_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_abs_equal_necessary_not_sufficient_l1668_166883


namespace NUMINAMATH_CALUDE_no_divisible_by_seven_l1668_166805

theorem no_divisible_by_seven : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2009 → ¬(7 ∣ (4 * n^6 + n^3 + 5)) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_seven_l1668_166805


namespace NUMINAMATH_CALUDE_remainder_theorem_l1668_166859

-- Define the polynomial p(x) = 4x^3 - 12x^2 + 16x - 20
def p (x : ℝ) : ℝ := 4 * x^3 - 12 * x^2 + 16 * x - 20

-- Define the divisor d(x) = x - 3
def d (x : ℝ) : ℝ := x - 3

-- Theorem statement
theorem remainder_theorem :
  (p 3 : ℝ) = 28 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1668_166859


namespace NUMINAMATH_CALUDE_log_relation_l1668_166834

theorem log_relation (y : ℝ) (k : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 625 / Real.log 2 = k * y) → 
  k = 12 := by sorry

end NUMINAMATH_CALUDE_log_relation_l1668_166834


namespace NUMINAMATH_CALUDE_complement_of_A_l1668_166814

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 + 2*x ≥ 0}

-- State the theorem
theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1668_166814


namespace NUMINAMATH_CALUDE_max_a_value_l1668_166873

theorem max_a_value (a b c d : ℕ+) 
  (h1 : a < 2 * b + 1)
  (h2 : b < 3 * c + 1)
  (h3 : c < 4 * d + 1)
  (h4 : d^2 < 10000) :
  a ≤ 2376 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1668_166873


namespace NUMINAMATH_CALUDE_smallest_K_for_inequality_l1668_166818

theorem smallest_K_for_inequality : 
  ∃ (K : ℝ), K = Real.sqrt 6 / 3 ∧ 
  (∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 → 
    K + (a + b + c) / 3 ≥ (K + 1) * Real.sqrt ((a^2 + b^2 + c^2) / 3)) ∧
  (∀ (K' : ℝ), K' > 0 ∧ K' < K → 
    ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧
      K' + (a + b + c) / 3 < (K' + 1) * Real.sqrt ((a^2 + b^2 + c^2) / 3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_K_for_inequality_l1668_166818


namespace NUMINAMATH_CALUDE_cos_equality_problem_l1668_166813

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l1668_166813


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l1668_166855

theorem cricket_team_captain_age (team_size : ℕ) (captain_age wicket_keeper_age : ℕ) 
  (team_average : ℚ) (remaining_average : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  team_average = 22 →
  remaining_average = team_average - 1 →
  (team_size : ℚ) * team_average = 
    captain_age + wicket_keeper_age + (team_size - 2 : ℚ) * remaining_average →
  captain_age = 25 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l1668_166855


namespace NUMINAMATH_CALUDE_min_value_function_extremum_function_l1668_166803

-- Part 1
theorem min_value_function (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) + 6 ≥ 9 ∧
  (x + 4 / (x + 1) + 6 = 9 ↔ x = 1) :=
sorry

-- Part 2
theorem extremum_function (x : ℝ) (h : x > 1) :
  (x^2 + 8) / (x - 1) ≥ 8 ∧
  ((x^2 + 8) / (x - 1) = 8 ↔ x = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_function_extremum_function_l1668_166803


namespace NUMINAMATH_CALUDE_pasture_rental_problem_l1668_166884

/-- The pasture rental problem -/
theorem pasture_rental_problem 
  (total_cost : ℕ) 
  (a_horses b_horses c_horses : ℕ) 
  (b_months c_months : ℕ) 
  (b_payment : ℕ) 
  (h_total_cost : total_cost = 870)
  (h_a_horses : a_horses = 12)
  (h_b_horses : b_horses = 16)
  (h_c_horses : c_horses = 18)
  (h_b_months : b_months = 9)
  (h_c_months : c_months = 6)
  (h_b_payment : b_payment = 360)
  : ∃ (a_months : ℕ), 
    a_horses * a_months * b_payment = b_horses * b_months * (total_cost - b_payment - c_horses * c_months * b_payment / (b_horses * b_months)) ∧ 
    a_months = 8 :=
by sorry

end NUMINAMATH_CALUDE_pasture_rental_problem_l1668_166884


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l1668_166892

/-- Given four real numbers a, b, c, d forming a geometric progression,
    prove that (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 -/
theorem geometric_progression_equality (a b c d : ℝ) 
  (h1 : c^2 = b * d) 
  (h2 : b^2 = a * c) 
  (h3 : a * d = b * c) : 
  (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l1668_166892


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l1668_166819

theorem popped_kernel_probability (p_white p_yellow p_red : ℝ)
  (pop_white pop_yellow pop_red : ℝ) :
  p_white = 1/2 →
  p_yellow = 1/3 →
  p_red = 1/6 →
  pop_white = 1/2 →
  pop_yellow = 2/3 →
  pop_red = 1/3 →
  (p_white * pop_white) / (p_white * pop_white + p_yellow * pop_yellow + p_red * pop_red) = 9/19 := by
sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l1668_166819


namespace NUMINAMATH_CALUDE_cube_sum_equation_l1668_166853

theorem cube_sum_equation (a b : ℝ) 
  (h1 : a^5 - a^4*b - a^4 + a - b - 1 = 0)
  (h2 : 2*a - 3*b = 1) : 
  a^3 + b^3 = 9 := by sorry

end NUMINAMATH_CALUDE_cube_sum_equation_l1668_166853


namespace NUMINAMATH_CALUDE_symmetric_points_on_circumcircle_l1668_166881

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a point symmetric to another point with respect to a line
def symmetric_point (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem symmetric_points_on_circumcircle (t : Triangle) :
  let H := orthocenter t
  let A1 := symmetric_point H (t.B, t.C)
  let B1 := symmetric_point H (t.C, t.A)
  let C1 := symmetric_point H (t.A, t.B)
  A1 ∈ circumcircle t ∧ B1 ∈ circumcircle t ∧ C1 ∈ circumcircle t := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_on_circumcircle_l1668_166881


namespace NUMINAMATH_CALUDE_barn_painted_area_l1668_166844

/-- Calculates the total area to be painted for a rectangular barn --/
def total_painted_area (width length height : ℝ) : ℝ :=
  let wall_area := 2 * (width * height + length * height)
  let floor_ceiling_area := 2 * (width * length)
  wall_area + floor_ceiling_area

/-- Theorem stating that the total area to be painted for the given barn is 1002 sq yd --/
theorem barn_painted_area :
  total_painted_area 15 18 7 = 1002 := by
  sorry

end NUMINAMATH_CALUDE_barn_painted_area_l1668_166844


namespace NUMINAMATH_CALUDE_vector_AB_l1668_166893

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define the vector between two points
def vector (p q : Point) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem vector_AB : 
  let A : Point := (-2, 3)
  let B : Point := (3, 2)
  vector A B = (5, -1) := by sorry

end NUMINAMATH_CALUDE_vector_AB_l1668_166893


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_value_l1668_166860

theorem quadratic_roots_imply_m_value (m : ℝ) : 
  (∃ x : ℂ, 5 * x^2 - 4 * x + m = 0 ∧ 
   (x = (2 + Complex.I * Real.sqrt 143) / 5 ∨ 
    x = (2 - Complex.I * Real.sqrt 143) / 5)) → 
  m = 7.95 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_value_l1668_166860


namespace NUMINAMATH_CALUDE_simplified_expression_equals_negative_sqrt_three_l1668_166867

theorem simplified_expression_equals_negative_sqrt_three :
  let a := 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180)
  let b := 3
  1 - (a - b) / (a + 2 * b) / ((a^2 - b^2) / (a^2 + 4 * a * b + 4 * b^2)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_negative_sqrt_three_l1668_166867


namespace NUMINAMATH_CALUDE_angle_conversion_l1668_166895

theorem angle_conversion (α k : ℤ) : 
  α = 195 ∧ k = -3 → 
  0 ≤ α ∧ α < 360 ∧ 
  -885 = α + k * 360 := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l1668_166895


namespace NUMINAMATH_CALUDE_max_annual_average_profit_l1668_166885

def profit_function (x : ℕ+) : ℚ := -x^2 + 18*x - 25

def annual_average_profit (x : ℕ+) : ℚ := (profit_function x) / x

theorem max_annual_average_profit :
  ∃ (x : ℕ+), (∀ (y : ℕ+), annual_average_profit y ≤ annual_average_profit x) ∧
              x = 5 ∧
              annual_average_profit x = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_annual_average_profit_l1668_166885


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l1668_166845

/-- The value of m for which the line x = -2 is tangent to the circle x^2 + y^2 + 6x + m = 0 -/
theorem parabola_circle_tangency (m : ℝ) : 
  (∀ y : ℝ, ((-2)^2 + y^2 + 6*(-2) + m = 0) → 
   (∀ x : ℝ, x ≠ -2 → x^2 + y^2 + 6*x + m ≠ 0)) → 
  m = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l1668_166845


namespace NUMINAMATH_CALUDE_correct_average_marks_l1668_166888

theorem correct_average_marks (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  (n * initial_avg - (wrong_mark - correct_mark)) / n = 98 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l1668_166888


namespace NUMINAMATH_CALUDE_candy_distribution_l1668_166842

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) (h1 : total_candy = 30) (h2 : num_friends = 4) :
  total_candy - (total_candy / num_friends) * num_friends = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1668_166842


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1668_166848

theorem sum_of_cubes (a b s p : ℝ) (h1 : s = a + b) (h2 : p = a * b) : 
  a^3 + b^3 = s^3 - 3*s*p := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1668_166848


namespace NUMINAMATH_CALUDE_two_digit_product_555_sum_l1668_166849

theorem two_digit_product_555_sum (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 555 → x + y = 52 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_product_555_sum_l1668_166849


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l1668_166879

/-- Calculates the total grocery bill for Ray's purchase with a rewards discount --/
theorem rays_grocery_bill :
  let hamburger_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let total_before_discount : ℚ := 
    hamburger_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  
  let discount_amount : ℚ := total_before_discount * discount_rate
  
  let final_bill : ℚ := total_before_discount - discount_amount

  final_bill = 18 := by sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l1668_166879


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1668_166887

def f (x : ℝ) := x^3 + x

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1668_166887


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1668_166874

theorem quadratic_coefficient (c : ℝ) : (5 : ℝ)^2 + c * 5 + 45 = 0 → c = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1668_166874


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1668_166828

/-- Theorem: If a line y = mx + 3 is tangent to the ellipse x² + 9y² = 9, then m² = 8/9 -/
theorem line_tangent_to_ellipse (m : ℝ) : 
  (∃! x y : ℝ, y = m * x + 3 ∧ x^2 + 9 * y^2 = 9) → m^2 = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1668_166828


namespace NUMINAMATH_CALUDE_marble_probability_l1668_166837

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  total = 84 →
  p_white = 1/4 →
  p_green = 1/7 →
  1 - (p_white + p_green) = 17/28 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1668_166837


namespace NUMINAMATH_CALUDE_inequality_system_sum_l1668_166833

theorem inequality_system_sum (a b : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x + 2*a > 4 ∧ 2*x < b)) → 
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_sum_l1668_166833


namespace NUMINAMATH_CALUDE_integer_fractions_l1668_166825

theorem integer_fractions (x : ℤ) : 
  (∃ k : ℤ, (5 * x^3 - x + 17) = 15 * k) ∧ 
  (∃ m : ℤ, (2 * x^2 + x - 3) = 7 * m) ↔ 
  (∃ t : ℤ, x = 105 * t + 22 ∨ x = 105 * t + 37) :=
sorry

end NUMINAMATH_CALUDE_integer_fractions_l1668_166825


namespace NUMINAMATH_CALUDE_volume_increase_rectangular_prism_l1668_166839

/-- Theorem: Increase in volume of a rectangular prism -/
theorem volume_increase_rectangular_prism 
  (L B H : ℝ) 
  (h_positive : L > 0 ∧ B > 0 ∧ H > 0) :
  let V_original := L * B * H
  let V_new := (L * 1.15) * (B * 1.30) * (H * 1.20)
  (V_new - V_original) / V_original = 0.794 := by
  sorry

end NUMINAMATH_CALUDE_volume_increase_rectangular_prism_l1668_166839


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1668_166820

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : Fin 6 → Bool

/-- Counts the number of unit cubes with at least two painted faces in a painted cube -/
def count_multi_painted_cubes (c : Cube 4) : ℕ :=
  sorry

/-- The theorem stating that a 4x4x4 painted cube has 56 unit cubes with at least two painted faces -/
theorem painted_cube_theorem (c : Cube 4) 
  (h : ∀ (f : Fin 6), c.painted_faces f = true) : 
  count_multi_painted_cubes c = 56 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l1668_166820


namespace NUMINAMATH_CALUDE_max_value_theorem_l1668_166817

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 15) : 
  x + 2 * y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1668_166817


namespace NUMINAMATH_CALUDE_invisible_dots_count_l1668_166815

/-- The sum of numbers on a standard six-sided die -/
def dieSumOfFaces : ℕ := 21

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 4

/-- The list of visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [1, 1, 2, 3, 4, 4, 5, 6]

/-- The theorem stating that the number of invisible dots is 58 -/
theorem invisible_dots_count : 
  numberOfDice * dieSumOfFaces - visibleNumbers.sum = 58 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l1668_166815


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1668_166830

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d) 
  (h3 : (a 3 - a 1) * (a 9 - a 3) = (a 3 - a 1)^2) : 
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13/16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1668_166830


namespace NUMINAMATH_CALUDE_new_employee_age_l1668_166826

theorem new_employee_age 
  (initial_employees : ℕ) 
  (initial_avg_age : ℝ) 
  (final_employees : ℕ) 
  (final_avg_age : ℝ) : 
  initial_employees = 13 → 
  initial_avg_age = 35 → 
  final_employees = initial_employees + 1 → 
  final_avg_age = 34 → 
  (final_employees * final_avg_age - initial_employees * initial_avg_age : ℝ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_new_employee_age_l1668_166826


namespace NUMINAMATH_CALUDE_divisibility_condition_l1668_166802

/-- s_n is the sum of all integers in [1,n] that are mutually prime to n -/
def s_n (n : ℕ) : ℕ := sorry

/-- t_n is the sum of the remaining integers in [1,n] -/
def t_n (n : ℕ) : ℕ := sorry

/-- Theorem: For all integers n ≥ 2, n divides (s_n - t_n) if and only if n is odd -/
theorem divisibility_condition (n : ℕ) (h : n ≥ 2) :
  n ∣ (s_n n - t_n n) ↔ Odd n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1668_166802


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1668_166800

/-- Given a complex number z = (10-5ai)/(1-2i) where a is a real number,
    and the sum of its real and imaginary parts is 4,
    prove that its real part is negative and its imaginary part is positive. -/
theorem complex_number_in_second_quadrant (a : ℝ) :
  let z : ℂ := (10 - 5*a*Complex.I) / (1 - 2*Complex.I)
  (z.re + z.im = 4) →
  (z.re < 0 ∧ z.im > 0) :=
by sorry


end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1668_166800


namespace NUMINAMATH_CALUDE_triangle_area_l1668_166824

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (3, 5)

theorem triangle_area : 
  (1/2 : ℝ) * |a.1 * b.2 - a.2 * b.1| = 23/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1668_166824


namespace NUMINAMATH_CALUDE_intersection_points_l1668_166810

theorem intersection_points (x : ℝ) : 
  (∃ y : ℝ, y = 10 / (x^2 + 1) ∧ x^2 + y = 3) ↔ 
  (x = Real.sqrt (1 + 2 * Real.sqrt 2) ∨ x = -Real.sqrt (1 + 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l1668_166810


namespace NUMINAMATH_CALUDE_travel_distance_l1668_166858

theorem travel_distance (d : ℝ) (h1 : d > 0) :
  d / 4 + d / 8 + d / 12 = 11 / 60 →
  3 * d = 1.2 := by
sorry

end NUMINAMATH_CALUDE_travel_distance_l1668_166858


namespace NUMINAMATH_CALUDE_sam_tuna_change_sam_change_proof_l1668_166835

/-- Calculates the change Sam received when buying tuna cans. -/
theorem sam_tuna_change (num_cans : ℕ) (num_coupons : ℕ) (coupon_value : ℕ) 
  (can_cost : ℕ) (paid_amount : ℕ) : ℕ :=
  let total_discount := num_coupons * coupon_value
  let total_cost := num_cans * can_cost
  let actual_paid := total_cost - total_discount
  paid_amount - actual_paid

/-- Proves that Sam received $5.50 in change. -/
theorem sam_change_proof : 
  sam_tuna_change 9 5 25 175 2000 = 550 := by
  sorry

end NUMINAMATH_CALUDE_sam_tuna_change_sam_change_proof_l1668_166835


namespace NUMINAMATH_CALUDE_evaluate_expression_l1668_166843

theorem evaluate_expression : -(20 / 2 * (6^2 + 10) - 120 + 5 * 6) = -370 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1668_166843


namespace NUMINAMATH_CALUDE_circle_condition_l1668_166880

theorem circle_condition (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*k*x - 2*y + 5*k = 0 ∧ 
   ∀ (x' y' : ℝ), x'^2 + y'^2 + 4*k*x' - 2*y' + 5*k = 0 → 
   (x' - x)^2 + (y' - y)^2 = (x - x)^2 + (y - y)^2) ↔ 
  (k > 1 ∨ k < 1/4) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1668_166880


namespace NUMINAMATH_CALUDE_factor_expression_l1668_166801

theorem factor_expression (x : ℝ) : 72 * x^4 - 252 * x^9 = 36 * x^4 * (2 - 7 * x^5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1668_166801


namespace NUMINAMATH_CALUDE_group_frequency_l1668_166829

theorem group_frequency (sample_capacity : ℕ) (group_frequency : ℚ) :
  sample_capacity = 80 →
  group_frequency = 0.125 →
  (sample_capacity : ℚ) * group_frequency = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_frequency_l1668_166829


namespace NUMINAMATH_CALUDE_largest_value_l1668_166821

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 6 * (6 ^ (1 / 6)))
  (hb : b = 6 ^ (1 / 3))
  (hc : c = 6 ^ (1 / 4))
  (hd : d = 2 * (6 ^ (1 / 3)))
  (he : e = 3 * (4 ^ (1 / 3))) :
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e :=
sorry

end NUMINAMATH_CALUDE_largest_value_l1668_166821


namespace NUMINAMATH_CALUDE_only_crop_yield_fertilizer_correlational_l1668_166889

-- Define the types of relationships
inductive Relationship
| Functional
| Correlational

-- Define the variables for each relationship
def height_age_relation : Relationship := sorry
def cube_volume_edge_relation : Relationship := sorry
def pencils_money_relation : Relationship := sorry
def crop_yield_fertilizer_relation : Relationship := sorry

-- Theorem stating that only the crop yield and fertilizer relationship is correlational
theorem only_crop_yield_fertilizer_correlational :
  (height_age_relation = Relationship.Functional) ∧
  (cube_volume_edge_relation = Relationship.Functional) ∧
  (pencils_money_relation = Relationship.Functional) ∧
  (crop_yield_fertilizer_relation = Relationship.Correlational) := by sorry

end NUMINAMATH_CALUDE_only_crop_yield_fertilizer_correlational_l1668_166889


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1668_166841

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1668_166841


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l1668_166840

open Real

theorem tangent_equation_solution (x : ℝ) :
  tan x + tan (50 * π / 180) + tan (70 * π / 180) = tan x * tan (50 * π / 180) * tan (70 * π / 180) →
  ∃ n : ℤ, x = (60 + 180 * n) * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l1668_166840


namespace NUMINAMATH_CALUDE_slow_car_speed_is_correct_l1668_166899

/-- The speed of the slow car in km/h -/
def slow_car_speed : ℝ := 40

/-- The speed of the fast car in km/h -/
def fast_car_speed : ℝ := 1.5 * slow_car_speed

/-- The distance to the memorial hall in km -/
def distance : ℝ := 60

/-- The time difference between departures in hours -/
def time_difference : ℝ := 0.5

theorem slow_car_speed_is_correct :
  (distance / slow_car_speed) - (distance / fast_car_speed) = time_difference :=
sorry

end NUMINAMATH_CALUDE_slow_car_speed_is_correct_l1668_166899


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1668_166812

theorem smallest_k_no_real_roots :
  ∃ (k : ℤ),
    (∀ (j : ℤ), j < k → ∃ (x : ℝ), 3 * x * (j * x - 5) - 2 * x^2 + 9 = 0) ∧
    (∀ (x : ℝ), 3 * x * (k * x - 5) - 2 * x^2 + 9 ≠ 0) ∧
    k = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1668_166812


namespace NUMINAMATH_CALUDE_sticker_difference_l1668_166882

/-- The number of stickers each person has -/
structure StickerCount where
  jerry : ℕ
  george : ℕ
  fred : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : StickerCount) : Prop :=
  s.jerry = 3 * s.george ∧
  s.george < s.fred ∧
  s.fred = 18 ∧
  s.jerry = 36

/-- The theorem to prove -/
theorem sticker_difference (s : StickerCount) 
  (h : problem_conditions s) : s.fred - s.george = 6 := by
  sorry

end NUMINAMATH_CALUDE_sticker_difference_l1668_166882
