import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_properties_l1894_189495

def A1 : ℝ × ℝ × ℝ := (3, 10, -1)
def A2 : ℝ × ℝ × ℝ := (-2, 3, -5)
def A3 : ℝ × ℝ × ℝ := (-6, 0, -3)
def A4 : ℝ × ℝ × ℝ := (1, -1, 2)

def tetrahedron_volume (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ := sorry

def tetrahedron_height (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  tetrahedron_volume A1 A2 A3 A4 = 45.5 ∧
  tetrahedron_height A1 A2 A3 A4 = 7 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l1894_189495


namespace NUMINAMATH_CALUDE_percentage_problem_l1894_189443

theorem percentage_problem (P : ℝ) : P = 20 := by
  have h1 : 50 = P / 100 * 15 + 47 := by sorry
  have h2 : 15 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1894_189443


namespace NUMINAMATH_CALUDE_pool_game_tie_l1894_189473

theorem pool_game_tie (calvin_score : ℕ) (paislee_score : ℕ) : 
  calvin_score = 500 →
  paislee_score = (3 * calvin_score) / 4 →
  calvin_score - paislee_score = 125 :=
by sorry

end NUMINAMATH_CALUDE_pool_game_tie_l1894_189473


namespace NUMINAMATH_CALUDE_max_degree_theorem_l1894_189464

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The degree of a polynomial -/
def degree (p : RealPolynomial) : ℕ := sorry

/-- The number of coefficients equal to 1 in a polynomial -/
def num_coeff_one (p : RealPolynomial) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def num_real_roots (p : RealPolynomial) : ℕ := sorry

/-- The maximum degree of a polynomial satisfying the given conditions -/
def max_degree : ℕ := 4

theorem max_degree_theorem :
  ∀ (p : RealPolynomial),
    num_coeff_one p ≥ degree p →
    num_real_roots p = degree p →
    degree p ≤ max_degree :=
by sorry

end NUMINAMATH_CALUDE_max_degree_theorem_l1894_189464


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1894_189497

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |6 - (250 : ℝ)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1894_189497


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l1894_189479

-- Define the propositions p and q
def p (x : ℝ) : Prop := 4 / (x - 1) ≤ -1
def q (x a : ℝ) : Prop := x^2 - x < a^2 - a

-- Define the condition that ¬q is sufficient but not necessary for ¬p
def sufficient_not_necessary (a : ℝ) : Prop :=
  ∀ x, ¬(q x a) → ¬(p x) ∧ ∃ y, ¬(p y) ∧ q y a

-- Define the range of a
def range_of_a : Set ℝ := {a | a ∈ [0, 1] ∧ a ≠ 1/2}

-- State the theorem
theorem range_of_a_theorem :
  ∀ a, sufficient_not_necessary a ↔ a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l1894_189479


namespace NUMINAMATH_CALUDE_am_gm_inequality_special_case_l1894_189441

theorem am_gm_inequality_special_case (x : ℝ) (h : x > 0) :
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_special_case_l1894_189441


namespace NUMINAMATH_CALUDE_new_person_weight_l1894_189416

/-- Given 4 persons where one weighing 65 kg is replaced, causing the average weight to increase by 1.5 kg, prove the new person weighs 71 kg. -/
theorem new_person_weight (initial_total : ℝ) (h1 : initial_total > 0) : 
  let final_total := initial_total - 65 + (initial_total / 4 + 1.5) * 4
  final_total - initial_total = 6 → 71 = final_total - (initial_total - 65) :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1894_189416


namespace NUMINAMATH_CALUDE_expression_exists_l1894_189408

/-- Represents an expression formed by ones and operators --/
inductive Expression
  | One : Expression
  | Add : Expression → Expression → Expression
  | Mul : Expression → Expression → Expression

/-- Evaluates the expression --/
def evaluate : Expression → ℕ
  | Expression.One => 1
  | Expression.Add e1 e2 => evaluate e1 + evaluate e2
  | Expression.Mul e1 e2 => evaluate e1 * evaluate e2

/-- Swaps the operators in the expression --/
def swap_operators : Expression → Expression
  | Expression.One => Expression.One
  | Expression.Add e1 e2 => Expression.Mul (swap_operators e1) (swap_operators e2)
  | Expression.Mul e1 e2 => Expression.Add (swap_operators e1) (swap_operators e2)

/-- Theorem stating the existence of the required expression --/
theorem expression_exists : ∃ (e : Expression), 
  evaluate e = 2014 ∧ evaluate (swap_operators e) = 2014 := by
  sorry


end NUMINAMATH_CALUDE_expression_exists_l1894_189408


namespace NUMINAMATH_CALUDE_merchant_loss_l1894_189428

theorem merchant_loss (C S : ℝ) (h : C > 0) :
  40 * C = 25 * S → (S - C) / C * 100 = -20 := by
  sorry

end NUMINAMATH_CALUDE_merchant_loss_l1894_189428


namespace NUMINAMATH_CALUDE_problem_solution_l1894_189405

theorem problem_solution (x y : ℝ) : 
  (0.40 * x = (1/3) * y + 110) → 
  (y = (2/3) * x) → 
  (x = 618.75 ∧ y = 412.5) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1894_189405


namespace NUMINAMATH_CALUDE_semicircle_in_right_triangle_l1894_189462

/-- Given a right-angled triangle with an inscribed semicircle, where:
    - The semicircle has radius r
    - The shorter edges of the triangle are tangent to the semicircle and have lengths a and b
    - The diameter of the semicircle lies on the hypotenuse of the triangle
    Then: 1/r = 1/a + 1/b -/
theorem semicircle_in_right_triangle (r a b : ℝ) 
    (hr : r > 0) (ha : a > 0) (hb : b > 0)
    (h_right_triangle : ∃ c, a^2 + b^2 = c^2)
    (h_tangent : ∃ p q : ℝ × ℝ, 
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = (2*r)^2 ∧
      (p.1 - 0)^2 + (p.2 - 0)^2 = a^2 ∧
      (q.1 - 0)^2 + (q.2 - 0)^2 = b^2) :
  1/r = 1/a + 1/b := by
    sorry

end NUMINAMATH_CALUDE_semicircle_in_right_triangle_l1894_189462


namespace NUMINAMATH_CALUDE_cubic_real_root_existence_l1894_189471

theorem cubic_real_root_existence (a₀ a₁ a₂ a₃ : ℝ) (ha₀ : a₀ ≠ 0) :
  ∃ x : ℝ, a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_real_root_existence_l1894_189471


namespace NUMINAMATH_CALUDE_solve_system_l1894_189417

-- Define the system of equations and the condition
def system_of_equations (x y m : ℝ) : Prop :=
  (4 * x + 2 * y = 3 * m) ∧ (3 * x + y = m + 2)

def opposite_sign (x y : ℝ) : Prop :=
  y = -x

-- Theorem statement
theorem solve_system :
  ∀ (x y m : ℝ), system_of_equations x y m → opposite_sign x y → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1894_189417


namespace NUMINAMATH_CALUDE_complement_A_B_l1894_189447

def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

theorem complement_A_B : (A \ B) = {0, 2, 6, 10} := by sorry

end NUMINAMATH_CALUDE_complement_A_B_l1894_189447


namespace NUMINAMATH_CALUDE_expected_replant_is_200_l1894_189440

/-- The expected number of seeds to be replanted -/
def expected_replant (p : ℝ) (n : ℕ) (r : ℕ) : ℝ :=
  n * (1 - p) * r

/-- Theorem: The expected number of seeds to be replanted is 200 -/
theorem expected_replant_is_200 :
  expected_replant 0.9 1000 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_expected_replant_is_200_l1894_189440


namespace NUMINAMATH_CALUDE_ef_is_one_eighth_of_gh_l1894_189487

/-- Given a line segment GH with points E and F on it, prove that EF is 1/8 of GH -/
theorem ef_is_one_eighth_of_gh (G E F H : Real) :
  (E ≥ G) → (F ≥ G) → (H ≥ E) → (H ≥ F) →  -- E and F lie on GH
  (E - G = 3 * (H - E)) →  -- GE = 3EH
  (F - G = 7 * (H - F)) →  -- GF = 7FH
  abs (E - F) = (1/8) * (H - G) := by sorry

end NUMINAMATH_CALUDE_ef_is_one_eighth_of_gh_l1894_189487


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l1894_189483

theorem isosceles_triangle_quadratic_roots (m n : ℝ) (k : ℝ) : 
  (m > 0 ∧ n > 0) →  -- positive side lengths
  (m = n ∨ m = 4 ∨ n = 4) →  -- isosceles condition
  (m ≠ n ∨ m ≠ 4) →  -- not equilateral
  (m + n > 4 ∧ m + 4 > n ∧ n + 4 > m) →  -- triangle inequality
  (m^2 - 6*m + k + 2 = 0) →  -- m is a root
  (n^2 - 6*n + k + 2 = 0) →  -- n is a root
  (k = 6 ∨ k = 7) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l1894_189483


namespace NUMINAMATH_CALUDE_base_conversion_difference_l1894_189414

-- Define a function to convert a number from base b to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [3, 0, 5]
def base1 : Nat := 8

def num2 : List Nat := [1, 6, 5]
def base2 : Nat := 7

-- Theorem statement
theorem base_conversion_difference :
  to_base_10 num1 base1 - to_base_10 num2 base2 = 101 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_difference_l1894_189414


namespace NUMINAMATH_CALUDE_paper_towel_case_rolls_l1894_189491

/-- The number of rolls in a case of paper towels -/
def number_of_rolls : ℕ := 12

/-- The price of the case in dollars -/
def case_price : ℚ := 9

/-- The price of an individual roll in dollars -/
def individual_roll_price : ℚ := 1

/-- The savings percentage per roll when buying the case -/
def savings_percentage : ℚ := 25 / 100

theorem paper_towel_case_rolls :
  case_price = number_of_rolls * (individual_roll_price * (1 - savings_percentage)) :=
sorry

end NUMINAMATH_CALUDE_paper_towel_case_rolls_l1894_189491


namespace NUMINAMATH_CALUDE_third_year_sample_size_l1894_189404

/-- Calculates the number of students to be sampled from the third year in a stratified sampling -/
theorem third_year_sample_size 
  (total_students : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 900) 
  (h2 : first_year = 240) 
  (h3 : second_year = 260) 
  (h4 : sample_size = 45) :
  (sample_size * (total_students - first_year - second_year)) / total_students = 20 := by
  sorry

#check third_year_sample_size

end NUMINAMATH_CALUDE_third_year_sample_size_l1894_189404


namespace NUMINAMATH_CALUDE_slope_of_line_l1894_189469

/-- The slope of a line represented by the equation 4x + 5y = 20 is -4/5 -/
theorem slope_of_line (x y : ℝ) : 4 * x + 5 * y = 20 → (y - 4) / x = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1894_189469


namespace NUMINAMATH_CALUDE_megan_total_songs_l1894_189442

/-- The number of country albums Megan bought -/
def country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def pop_albums : ℕ := 8

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := (country_albums + pop_albums) * songs_per_album

theorem megan_total_songs : total_songs = 70 := by
  sorry

end NUMINAMATH_CALUDE_megan_total_songs_l1894_189442


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_three_l1894_189481

theorem gcd_of_powers_of_three : Nat.gcd (3^1001 - 1) (3^1010 - 1) = 19682 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_three_l1894_189481


namespace NUMINAMATH_CALUDE_cave_depth_l1894_189474

theorem cave_depth (current_depth remaining_distance : ℕ) 
  (h1 : current_depth = 849)
  (h2 : remaining_distance = 369) : 
  current_depth + remaining_distance = 1218 := by
sorry

end NUMINAMATH_CALUDE_cave_depth_l1894_189474


namespace NUMINAMATH_CALUDE_large_circle_radius_l1894_189484

/-- Configuration of circles -/
structure CircleConfiguration where
  small_radius : ℝ
  chord_length : ℝ
  small_circle_count : ℕ

/-- Theorem: If five identical circles are placed in a line inside a larger circle,
    and the chord connecting the endpoints of the line of circles has length 16,
    then the radius of the large circle is 8. -/
theorem large_circle_radius
  (config : CircleConfiguration)
  (h1 : config.small_circle_count = 5)
  (h2 : config.chord_length = 16) :
  4 * config.small_radius = 8 := by
  sorry

#check large_circle_radius

end NUMINAMATH_CALUDE_large_circle_radius_l1894_189484


namespace NUMINAMATH_CALUDE_gcf_of_60_180_150_l1894_189450

theorem gcf_of_60_180_150 : Nat.gcd 60 (Nat.gcd 180 150) = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_180_150_l1894_189450


namespace NUMINAMATH_CALUDE_solve_for_a_l1894_189406

-- Define the function f(x) = x^2 + ax
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem solve_for_a (a : ℝ) (h1 : a < -1) :
  (∀ x : ℝ, f a x ≤ -x) ∧ 
  (∃ x : ℝ, f a x = -x) ∧
  (∀ x : ℝ, f a x ≥ -1/2) ∧
  (∃ x : ℝ, f a x = -1/2) →
  a = -3/2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l1894_189406


namespace NUMINAMATH_CALUDE_max_min_difference_l1894_189419

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

-- Define the interval
def interval : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem max_min_difference (a : ℝ) :
  ∃ (M N : ℝ),
    (∀ x ∈ interval, f a x ≤ M) ∧
    (∃ x ∈ interval, f a x = M) ∧
    (∀ x ∈ interval, N ≤ f a x) ∧
    (∃ x ∈ interval, f a x = N) ∧
    M - N = 18 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l1894_189419


namespace NUMINAMATH_CALUDE_pencil_remainder_l1894_189480

theorem pencil_remainder (a b : ℕ) 
  (ha : a % 8 = 5) 
  (hb : b % 8 = 6) : 
  (a + b) % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_pencil_remainder_l1894_189480


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l1894_189455

theorem range_of_a_for_quadratic_inequality :
  {a : ℝ | ∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0} = {a : ℝ | -8 ≤ a ∧ a ≤ 0} := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l1894_189455


namespace NUMINAMATH_CALUDE_three_digit_number_sum_l1894_189465

theorem three_digit_number_sum (a b c : ℕ) : 
  (100 * a + 10 * b + c) % 5 = 0 →
  a = 2 * b →
  a * b * c = 40 →
  a + b + c = 11 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_sum_l1894_189465


namespace NUMINAMATH_CALUDE_sum_equals_1332_l1894_189456

/-- Converts a base 4 number (represented as a list of digits) to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a decimal number to its base 4 representation (as a list of digits) -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The sum of 232₄, 121₄, and 313₄ in base 4 -/
def sumInBase4 : List Nat :=
  decimalToBase4 (base4ToDecimal [2,3,2] + base4ToDecimal [1,2,1] + base4ToDecimal [3,1,3])

theorem sum_equals_1332 : sumInBase4 = [1,3,3,2] := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_1332_l1894_189456


namespace NUMINAMATH_CALUDE_complex_product_QED_l1894_189452

theorem complex_product_QED (Q E D : ℂ) : 
  Q = 4 + 3*I ∧ E = 2*I ∧ D = 4 - 3*I → Q * E * D = 50*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_QED_l1894_189452


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1894_189436

theorem fixed_point_on_line (m : ℝ) : 
  m * (-2) - 1 + 2 * m + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1894_189436


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1894_189445

-- Define an isosceles triangle with side lengths 4 and 7
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 4 ∧ b = 7 ∧ a = c) ∨ (a = 7 ∧ b = 4 ∧ a = c)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 15 ∨ Perimeter a b c = 18 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1894_189445


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1894_189463

/-- A geometric sequence with the given property has common ratio 2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (h : ∀ n, a n * a (n + 1) = 4^n)  -- The given condition
  : (∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = q * a n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1894_189463


namespace NUMINAMATH_CALUDE_system_solution_l1894_189438

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 5 * x^2 - 14 * x * y + 10 * y^2 = 17
def equation2 (x y : ℝ) : Prop := 4 * x^2 - 10 * x * y + 6 * y^2 = 8

-- Define the solution set
def solutions : List (ℝ × ℝ) := [(-1, -2), (11, 7), (-11, -7), (1, 2)]

-- Theorem statement
theorem system_solution :
  ∀ (p : ℝ × ℝ), p ∈ solutions → equation1 p.1 p.2 ∧ equation2 p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1894_189438


namespace NUMINAMATH_CALUDE_wrapping_paper_rolls_l1894_189413

/-- The number of rolls of wrapping paper Savannah bought -/
def rolls_bought : ℕ := 3

/-- The total number of gifts Savannah has to wrap -/
def total_gifts : ℕ := 12

/-- The number of gifts wrapped with the first roll -/
def gifts_first_roll : ℕ := 3

/-- The number of gifts wrapped with the second roll -/
def gifts_second_roll : ℕ := 5

/-- The number of gifts wrapped with the third roll -/
def gifts_third_roll : ℕ := 4

theorem wrapping_paper_rolls :
  rolls_bought = 3 ∧
  total_gifts = 12 ∧
  gifts_first_roll = 3 ∧
  gifts_second_roll = 5 ∧
  gifts_third_roll = 4 ∧
  total_gifts = gifts_first_roll + gifts_second_roll + gifts_third_roll :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_rolls_l1894_189413


namespace NUMINAMATH_CALUDE_point_in_triangle_property_l1894_189427

/-- Triangle in the xy-plane with vertices (0,0), (4,0), and (4,10) -/
def Triangle : Set (ℝ × ℝ) :=
  {p | ∃ (t1 t2 : ℝ), 0 ≤ t1 ∧ 0 ≤ t2 ∧ t1 + t2 ≤ 1 ∧
       p.1 = 4 * t1 + 4 * t2 ∧
       p.2 = 10 * t2}

/-- The theorem states that for any point (a, b) in the defined triangle, a - b ≤ 0 -/
theorem point_in_triangle_property (p : ℝ × ℝ) (h : p ∈ Triangle) : p.1 - p.2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_triangle_property_l1894_189427


namespace NUMINAMATH_CALUDE_ayen_exercise_time_l1894_189454

/-- Represents the total exercise time in minutes for a week -/
def weekly_exercise (
  weekday_jog : ℕ
  ) (tuesday_extra : ℕ) (friday_extra : ℕ) (saturday_jog : ℕ) (sunday_swim : ℕ) : ℚ :=
  let weekday_total := 3 * weekday_jog + (weekday_jog + tuesday_extra) + (weekday_jog + friday_extra)
  let jogging_total := weekday_total + saturday_jog
  let swimming_equivalent := (3 / 2) * sunday_swim
  (jogging_total + swimming_equivalent) / 60

/-- The theorem stating Ayen's total exercise time for the week -/
theorem ayen_exercise_time : 
  weekly_exercise 30 5 25 45 60 = (23 / 4) := by sorry

end NUMINAMATH_CALUDE_ayen_exercise_time_l1894_189454


namespace NUMINAMATH_CALUDE_toy_purchase_cost_l1894_189493

theorem toy_purchase_cost (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) : 
  num_toys = 5 → 
  cost_per_toy = 3 → 
  discount_percent = 20 →
  (num_toys * cost_per_toy) * (1 - discount_percent / 100) = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_cost_l1894_189493


namespace NUMINAMATH_CALUDE_train_speeds_l1894_189490

theorem train_speeds (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 450)
  (h2 : time = 5)
  (h3 : speed_difference = 6) :
  ∃ (speed1 speed2 : ℝ),
    speed2 = speed1 + speed_difference ∧
    distance = (speed1 + speed2) * time ∧
    speed1 = 42 ∧
    speed2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_train_speeds_l1894_189490


namespace NUMINAMATH_CALUDE_plane_through_points_l1894_189492

def point1 : ℝ × ℝ × ℝ := (2, -3, 5)
def point2 : ℝ × ℝ × ℝ := (4, -3, 6)
def point3 : ℝ × ℝ × ℝ := (6, -4, 8)

def plane_equation (x y z : ℝ) : ℝ := x - 2*y + 2*z - 18

theorem plane_through_points :
  (plane_equation point1.1 point1.2.1 point1.2.2 = 0) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2 = 0) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2 = 0) ∧
  (1 > 0) ∧
  (Nat.gcd (Nat.gcd 1 2) (Nat.gcd 2 18) = 1) := by
  sorry

end NUMINAMATH_CALUDE_plane_through_points_l1894_189492


namespace NUMINAMATH_CALUDE_smallest_number_l1894_189429

theorem smallest_number (S : Finset ℕ) (h : S = {5, 8, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1894_189429


namespace NUMINAMATH_CALUDE_square_root_problem_l1894_189498

-- Define the variables
variable (a b c : ℝ)

-- Define the conditions
def condition1 : Prop := (5 * a + 2) ^ (1/3 : ℝ) = 3
def condition2 : Prop := (3 * a + b - 1).sqrt = 4
def condition3 : Prop := c = ⌊(13 : ℝ).sqrt⌋

-- State the theorem
theorem square_root_problem (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 c) :
  (3 * a - b + c).sqrt = 4 ∨ (3 * a - b + c).sqrt = -4 :=
sorry

end NUMINAMATH_CALUDE_square_root_problem_l1894_189498


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l1894_189476

def total_students : ℕ := 500
def cat_owners : ℕ := 75

theorem percentage_of_cat_owners : 
  (cat_owners : ℚ) / total_students * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l1894_189476


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l1894_189411

theorem cubic_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (17*x - 2)^(1/3) + (11*x + 2)^(1/3) - 2*(9*x)^(1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = (2 + Real.sqrt 35) / 31 ∨ x = (2 - Real.sqrt 35) / 31 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l1894_189411


namespace NUMINAMATH_CALUDE_board_numbers_product_l1894_189421

def pairwise_sums (a b c d e : ℤ) : Finset ℤ :=
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e}

theorem board_numbers_product (a b c d e : ℤ) :
  pairwise_sums a b c d e = {-1, 4, 6, 9, 10, 11, 15, 16, 20, 22} →
  a * b * c * d * e = -4914 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_product_l1894_189421


namespace NUMINAMATH_CALUDE_other_sales_percentage_l1894_189448

/-- Represents the sales distribution of the Dreamy Bookstore for April -/
structure SalesDistribution where
  notebooks : ℝ
  bookmarks : ℝ
  other : ℝ

/-- The sales distribution for the Dreamy Bookstore in April -/
def april_sales : SalesDistribution where
  notebooks := 45
  bookmarks := 25
  other := 100 - (45 + 25)

/-- Theorem stating that the percentage of sales that were neither notebooks nor bookmarks is 30% -/
theorem other_sales_percentage (s : SalesDistribution) 
  (h1 : s.notebooks = 45)
  (h2 : s.bookmarks = 25)
  (h3 : s.notebooks + s.bookmarks + s.other = 100) :
  s.other = 30 := by
  sorry

#eval april_sales.other

end NUMINAMATH_CALUDE_other_sales_percentage_l1894_189448


namespace NUMINAMATH_CALUDE_triangle_properties_l1894_189468

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c * Real.cos t.B = 2 * t.a + t.b) 
  (h2 : t.a = t.b) 
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (Real.sqrt 3 / 2) * t.c) :
  (t.C = 2 * Real.pi / 3) ∧ 
  (t.a + t.b + t.c = 6 + 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1894_189468


namespace NUMINAMATH_CALUDE_tina_fruit_difference_l1894_189434

/-- Represents the number of fruits in Tina's bag -/
structure FruitBag where
  oranges : ℕ
  tangerines : ℕ

/-- Calculates the difference between tangerines and oranges after removal -/
def tangerine_orange_difference (bag : FruitBag) (oranges_removed : ℕ) (tangerines_removed : ℕ) : ℤ :=
  (bag.tangerines - tangerines_removed) - (bag.oranges - oranges_removed)

theorem tina_fruit_difference :
  let initial_bag : FruitBag := { oranges := 5, tangerines := 17 }
  let oranges_removed := 2
  let tangerines_removed := 10
  tangerine_orange_difference initial_bag oranges_removed tangerines_removed = 4 := by
  sorry

end NUMINAMATH_CALUDE_tina_fruit_difference_l1894_189434


namespace NUMINAMATH_CALUDE_system_solution_l1894_189459

theorem system_solution :
  ∃ (x y : ℚ), 
    (7 * x = -9 - 3 * y) ∧ 
    (4 * x = 5 * y - 34) ∧ 
    (x = -413/235) ∧ 
    (y = -202/47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1894_189459


namespace NUMINAMATH_CALUDE_unique_solution_tan_cos_equation_l1894_189402

theorem unique_solution_tan_cos_equation : 
  ∃! (n : ℕ), n > 0 ∧ Real.tan (π / (2 * n)) + Real.cos (π / (2 * n)) = n / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_tan_cos_equation_l1894_189402


namespace NUMINAMATH_CALUDE_car_wheels_count_l1894_189451

theorem car_wheels_count (cars : ℕ) (motorcycles : ℕ) (total_wheels : ℕ) 
  (h1 : cars = 19)
  (h2 : motorcycles = 11)
  (h3 : total_wheels = 117)
  (h4 : ∀ m : ℕ, m ≤ motorcycles → 2 * m ≤ total_wheels) :
  ∃ (wheels_per_car : ℕ), wheels_per_car * cars + 2 * motorcycles = total_wheels ∧ wheels_per_car = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_wheels_count_l1894_189451


namespace NUMINAMATH_CALUDE_manuscript_cost_is_1350_l1894_189446

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscript_typing_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (first_time_cost : ℕ) (revision_cost : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let first_time_total := total_pages * first_time_cost
  let revision_once_total := pages_revised_once * revision_cost
  let revision_twice_total := pages_revised_twice * revision_cost * 2
  first_time_total + revision_once_total + revision_twice_total

/-- The total cost of typing the manuscript is $1350. -/
theorem manuscript_cost_is_1350 : 
  manuscript_typing_cost 100 30 20 10 5 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_is_1350_l1894_189446


namespace NUMINAMATH_CALUDE_candy_game_solution_l1894_189437

/-- Represents the game state and rules --/
structure CandyGame where
  totalCandies : Nat
  xiaomingEat : Nat
  xiaomingKeep : Nat
  xiaoliangEat : Nat
  xiaoliangKeep : Nat

/-- Represents the result of the game --/
structure GameResult where
  xiaomingWins : Nat
  xiaoliangWins : Nat
  xiaomingPocket : Nat
  xiaoliangPocket : Nat
  totalEaten : Nat

/-- The theorem to prove --/
theorem candy_game_solution (game : CandyGame)
  (h1 : game.totalCandies = 50)
  (h2 : game.xiaomingEat + game.xiaomingKeep = 5)
  (h3 : game.xiaoliangEat + game.xiaoliangKeep = 5)
  (h4 : game.xiaomingKeep = 1)
  (h5 : game.xiaoliangKeep = 2)
  : ∃ (result : GameResult),
    result.xiaomingWins + result.xiaoliangWins = game.totalCandies / 5 ∧
    result.xiaomingPocket = result.xiaomingWins * game.xiaomingKeep ∧
    result.xiaoliangPocket = result.xiaoliangWins * game.xiaoliangKeep ∧
    result.xiaoliangPocket = 3 * result.xiaomingPocket ∧
    result.totalEaten = result.xiaomingWins * game.xiaomingEat + result.xiaoliangWins * game.xiaoliangEat ∧
    result.totalEaten = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_candy_game_solution_l1894_189437


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l1894_189460

theorem quadratic_roots_theorem (a₁ a₂ a₃ : ℝ) : 
  a₁ > 0 → a₂ > 0 → a₃ > 0 →  -- a_1, a_2, a_3 are positive real numbers
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) →  -- a_1, a_2, a_3 form a geometric sequence
  (∃ x : ℝ, x^2 + a₁*x + 1 = 0) →  -- equation (1) has real roots
  (∀ x : ℝ, x^2 + a₂*x + 2 ≠ 0) →  -- equation (2) has no real roots
  (∀ x : ℝ, x^2 + a₃*x + 4 ≠ 0) :=  -- equation (3) has no real roots
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l1894_189460


namespace NUMINAMATH_CALUDE_factorial_ratio_squared_l1894_189494

theorem factorial_ratio_squared : (Nat.factorial 10 / Nat.factorial 9) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_squared_l1894_189494


namespace NUMINAMATH_CALUDE_internally_tangent_circles_distance_l1894_189410

theorem internally_tangent_circles_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 4) :
  let d := (r₁ - r₂)^2 + r₂^2
  d = (4 * Real.sqrt 10)^2 :=
by sorry

end NUMINAMATH_CALUDE_internally_tangent_circles_distance_l1894_189410


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l1894_189401

theorem fourth_root_equivalence (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 * y^(1/3))^(1/4) = x^(1/2) * y^(1/12) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l1894_189401


namespace NUMINAMATH_CALUDE_exam_students_count_l1894_189409

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) (excluded_count : ℕ) :
  total_average = 80 →
  excluded_average = 20 →
  remaining_average = 92 →
  excluded_count = 5 →
  ∃ N : ℕ, 
    N * total_average = (N - excluded_count) * remaining_average + excluded_count * excluded_average ∧
    N = 30 := by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l1894_189409


namespace NUMINAMATH_CALUDE_second_oldest_age_l1894_189477

/-- Represents the ages of three brothers -/
structure BrothersAges where
  youngest : ℕ
  secondOldest : ℕ
  oldest : ℕ

/-- Defines the conditions for the brothers' ages -/
def validAges (ages : BrothersAges) : Prop :=
  ages.youngest + ages.secondOldest + ages.oldest = 34 ∧
  ages.oldest = 3 * ages.youngest ∧
  ages.secondOldest = 2 * ages.youngest - 2

/-- Theorem stating that the second oldest brother is 10 years old -/
theorem second_oldest_age (ages : BrothersAges) (h : validAges ages) : ages.secondOldest = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_oldest_age_l1894_189477


namespace NUMINAMATH_CALUDE_repair_easier_than_thermometer_l1894_189496

def word1 : String := "термометр"
def word2 : String := "ремонт"

def uniqueLetters (s : String) : Finset Char :=
  s.toList.toFinset

theorem repair_easier_than_thermometer :
  (uniqueLetters word2).card > (uniqueLetters word1).card := by
  sorry

end NUMINAMATH_CALUDE_repair_easier_than_thermometer_l1894_189496


namespace NUMINAMATH_CALUDE_root_sequence_difference_l1894_189432

theorem root_sequence_difference (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a = 1) ∧
    (a * d = b * c) ∧
    ({a, b, c, d} = {x : ℝ | (x^2 - m*x + 27 = 0) ∨ (x^2 - n*x + 27 = 0)}) ∧
    (∃ q : ℝ, b = a*q ∧ c = b*q ∧ d = c*q)) →
  |m - n| = 16 :=
by sorry

end NUMINAMATH_CALUDE_root_sequence_difference_l1894_189432


namespace NUMINAMATH_CALUDE_eleventh_term_value_l1894_189412

/-- An arithmetic progression is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

/-- The theorem states that for an arithmetic progression satisfying
    certain conditions, the 11th term is 109. -/
theorem eleventh_term_value
    (a : ℕ → ℝ)
    (h_ap : ArithmeticProgression a)
    (h_sum1 : a 4 + a 7 + a 10 = 207)
    (h_sum2 : a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 553) :
    a 11 = 109 := by
  sorry


end NUMINAMATH_CALUDE_eleventh_term_value_l1894_189412


namespace NUMINAMATH_CALUDE_system_two_solutions_iff_l1894_189415

def system_has_two_solutions (a : ℝ) : Prop :=
  ∃! (s₁ s₂ : ℝ × ℝ), s₁ ≠ s₂ ∧
    (∀ (x y : ℝ), (x, y) = s₁ ∨ (x, y) = s₂ →
      a^2 - 2*a*x + 10*y + x^2 + y^2 = 0 ∧
      (|x| - 12)^2 + (|y| - 5)^2 = 169)

theorem system_two_solutions_iff (a : ℝ) :
  system_has_two_solutions a ↔ 
  (a > -30 ∧ a < -20) ∨ a = 0 ∨ (a > 20 ∧ a < 30) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_iff_l1894_189415


namespace NUMINAMATH_CALUDE_special_pie_crust_flour_amount_l1894_189499

/-- The amount of flour used in each special pie crust when the total flour amount remains constant but the number of crusts changes. -/
theorem special_pie_crust_flour_amount 
  (typical_crusts : ℕ) 
  (typical_flour_per_crust : ℚ) 
  (special_crusts : ℕ) 
  (h1 : typical_crusts = 50)
  (h2 : typical_flour_per_crust = 1 / 10)
  (h3 : special_crusts = 25)
  (h4 : typical_crusts * typical_flour_per_crust = special_crusts * (special_flour_per_crust : ℚ)) :
  special_flour_per_crust = 1 / 5 := by
  sorry

#check special_pie_crust_flour_amount

end NUMINAMATH_CALUDE_special_pie_crust_flour_amount_l1894_189499


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_137_l1894_189424

theorem first_nonzero_digit_of_one_over_137 :
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d < 10 ∧ 
  (∀ (k : ℕ), k < n → (10^(k+1) / 137 % 10 = 0)) ∧
  (10^(n+1) / 137 % 10 = d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_137_l1894_189424


namespace NUMINAMATH_CALUDE_mixture_composition_l1894_189478

theorem mixture_composition (alcohol_volume : ℚ) (water_volume : ℚ) 
  (h1 : alcohol_volume = 3/5)
  (h2 : alcohol_volume / water_volume = 3/4) :
  water_volume = 4/5 := by
sorry

end NUMINAMATH_CALUDE_mixture_composition_l1894_189478


namespace NUMINAMATH_CALUDE_next_skipped_perfect_square_l1894_189449

theorem next_skipped_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, m^2 = n) ∧
  (∀ y : ℕ, y > x ∧ y < n → ¬∃ m : ℕ, m^2 = y) ∧
  (∃ m : ℕ, m^2 = x + 4 * Real.sqrt x + 4) :=
sorry

end NUMINAMATH_CALUDE_next_skipped_perfect_square_l1894_189449


namespace NUMINAMATH_CALUDE_parabola_intersection_locus_locus_nature_l1894_189403

/-- Given a parabola and a point in its plane, this theorem describes the locus of 
    intersection points formed by certain lines related to the parabola. -/
theorem parabola_intersection_locus 
  (p : ℝ) -- Parameter of the parabola
  (α β : ℝ) -- Coordinates of point A
  (x y : ℝ) -- Coordinates of the locus point M
  (h_parabola : y^2 = 2*p*x) -- Equation of the parabola
  : 2*p*x^2 - β*x*y + α*y^2 - 2*p*α*x = 0 := by
  sorry

/-- This theorem characterizes the nature of the locus based on the position of point A 
    relative to the parabola. -/
theorem locus_nature 
  (p : ℝ) -- Parameter of the parabola
  (α β : ℝ) -- Coordinates of point A
  : (β^2 = 8*p*α → IsParabola) ∧ 
    (β^2 < 8*p*α → IsEllipse) ∧ 
    (β^2 > 8*p*α → IsHyperbola) := by
  sorry

-- We need to define these predicates
axiom IsParabola : Prop
axiom IsEllipse : Prop
axiom IsHyperbola : Prop

end NUMINAMATH_CALUDE_parabola_intersection_locus_locus_nature_l1894_189403


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1894_189425

def U : Set Nat := {2, 3, 6, 8}
def A : Set Nat := {2, 3}
def B : Set Nat := {2, 6, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {6, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1894_189425


namespace NUMINAMATH_CALUDE_z_remainder_when_z_plus_3_div_9_is_integer_l1894_189422

theorem z_remainder_when_z_plus_3_div_9_is_integer (z : ℤ) :
  (∃ k : ℤ, (z + 3) / 9 = k) → z ≡ 6 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_z_remainder_when_z_plus_3_div_9_is_integer_l1894_189422


namespace NUMINAMATH_CALUDE_grocer_average_sale_l1894_189444

theorem grocer_average_sale 
  (sales : List ℕ) 
  (h1 : sales = [5266, 5744, 5864, 6122, 6588, 4916]) :
  (sales.sum / sales.length : ℚ) = 5750 := by
  sorry

end NUMINAMATH_CALUDE_grocer_average_sale_l1894_189444


namespace NUMINAMATH_CALUDE_long_letter_time_ratio_l1894_189466

/-- Represents the letter writing schedule and times for Steve --/
structure LetterWriting where
  days_between_letters : ℕ
  regular_letter_time : ℕ
  time_per_page : ℕ
  long_letter_time : ℕ
  total_pages_per_month : ℕ

/-- Calculates the ratio of time spent per page for the long letter compared to a regular letter --/
def time_ratio (lw : LetterWriting) : ℚ :=
  let regular_letters_per_month := 30 / lw.days_between_letters
  let pages_per_regular_letter := lw.regular_letter_time / lw.time_per_page
  let regular_letter_pages := regular_letters_per_month * pages_per_regular_letter
  let long_letter_pages := lw.total_pages_per_month - regular_letter_pages
  let long_letter_time_per_page := lw.long_letter_time / long_letter_pages
  long_letter_time_per_page / lw.time_per_page

/-- Theorem stating that the ratio of time spent per page for the long letter compared to a regular letter is 2:1 --/
theorem long_letter_time_ratio (lw : LetterWriting) 
  (h1 : lw.days_between_letters = 3)
  (h2 : lw.regular_letter_time = 20)
  (h3 : lw.time_per_page = 10)
  (h4 : lw.long_letter_time = 80)
  (h5 : lw.total_pages_per_month = 24) : 
  time_ratio lw = 2 := by
  sorry


end NUMINAMATH_CALUDE_long_letter_time_ratio_l1894_189466


namespace NUMINAMATH_CALUDE_system_solution_l1894_189453

theorem system_solution (x y : ℝ) (dot star : ℝ) : 
  (2 * x + y = dot ∧ 2 * x - y = 12 ∧ x = 5 ∧ y = star) → 
  (dot = 8 ∧ star = -2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1894_189453


namespace NUMINAMATH_CALUDE_no_real_solutions_l1894_189400

theorem no_real_solutions : ∀ x : ℝ, ¬(Real.sqrt (9 - 3*x) = x * Real.sqrt (9 - 9*x)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1894_189400


namespace NUMINAMATH_CALUDE_range_of_m_symmetrical_circle_equation_existence_of_m_for_circle_through_origin_l1894_189457

-- Define the circle C and line l
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem 1: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_C x y m) → m < 37/4 :=
sorry

-- Theorem 2: Equation of symmetrical circle
theorem symmetrical_circle_equation :
  ∀ m : ℝ, (∃ x y : ℝ, circle_C x y m ∧ line_l x y) →
  (∀ x y : ℝ, x^2 + (y - 7/2)^2 = 1/8) :=
sorry

-- Theorem 3: Existence of m for circle through origin
theorem existence_of_m_for_circle_through_origin :
  ∃ m : ℝ, m = -3/2 ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C x₁ y₁ m ∧ circle_C x₂ y₂ m ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∃ a b r : ℝ, (x₁ - a)^2 + (y₁ - b)^2 = r^2 ∧
                  (x₂ - a)^2 + (y₂ - b)^2 = r^2 ∧
                  a^2 + b^2 = r^2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_symmetrical_circle_equation_existence_of_m_for_circle_through_origin_l1894_189457


namespace NUMINAMATH_CALUDE_mean_equality_implies_sum_l1894_189407

theorem mean_equality_implies_sum (x y : ℝ) : 
  (4 + 10 + 16 + 24) / 4 = (14 + x + y) / 3 → x + y = 26.5 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_sum_l1894_189407


namespace NUMINAMATH_CALUDE_extremum_and_intersection_implies_m_range_l1894_189488

def f (x : ℝ) := x^3 - 3*x - 1

theorem extremum_and_intersection_implies_m_range :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1) ∨ f x ≥ f (-1)) →
  (∃ m : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) →
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) → 
  -3 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_extremum_and_intersection_implies_m_range_l1894_189488


namespace NUMINAMATH_CALUDE_equation_solution_l1894_189482

theorem equation_solution : ∃ x : ℚ, 
  x = 81 / 16 ∧ 
  Real.sqrt x + 4 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 2*x :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1894_189482


namespace NUMINAMATH_CALUDE_greatest_missable_problems_l1894_189433

theorem greatest_missable_problems (total_problems : ℕ) (passing_percentage : ℚ) 
  (h1 : total_problems = 50)
  (h2 : passing_percentage = 85 / 100) :
  ∃ (max_missable : ℕ), 
    max_missable = 7 ∧ 
    (total_problems - max_missable : ℚ) / total_problems ≥ passing_percentage ∧
    ∀ (n : ℕ), n > max_missable → (total_problems - n : ℚ) / total_problems < passing_percentage :=
by sorry

end NUMINAMATH_CALUDE_greatest_missable_problems_l1894_189433


namespace NUMINAMATH_CALUDE_ellipse_axes_sum_l1894_189472

/-- Given a cylinder and two spheres with specific dimensions, prove that the sum of the major and minor axes of the ellipse formed by a tangent plane is 25. -/
theorem ellipse_axes_sum (cylinder_radius sphere_radius : ℝ) (sphere_distance : ℝ) : 
  cylinder_radius = 6 →
  sphere_radius = 6 →
  sphere_distance = 13 →
  ∃ (major_axis minor_axis : ℝ),
    (major_axis + minor_axis = 25 ∧
     minor_axis = 2 * cylinder_radius ∧
     major_axis = sphere_distance) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axes_sum_l1894_189472


namespace NUMINAMATH_CALUDE_part_one_part_two_l1894_189470

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part_one (t : Triangle) (h1 : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b) 
    (h2 : 0 < t.A ∧ t.A < Real.pi / 2) : t.A = Real.pi / 3 := by
  sorry

-- Part 2
theorem part_two (t : Triangle) (h1 : t.b = 5) (h2 : t.c = Real.sqrt 5) 
    (h3 : Real.cos t.C = 9/10) : t.a = 4 ∨ t.a = 5 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1894_189470


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_total_volume_l1894_189475

-- Define the conversion factor
def meters_to_centimeters : ℝ := 100

-- Theorem 1: One cubic meter is equal to 1,000,000 cubic centimeters
theorem cubic_meter_to_cubic_centimeters : 
  (meters_to_centimeters ^ 3 : ℝ) = 1000000 := by sorry

-- Theorem 2: The sum of one cubic meter and 500 cubic centimeters is equal to 1,000,500 cubic centimeters
theorem total_volume (cubic_cm_to_add : ℝ) : 
  cubic_cm_to_add = 500 → 
  (meters_to_centimeters ^ 3 + cubic_cm_to_add : ℝ) = 1000500 := by sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_total_volume_l1894_189475


namespace NUMINAMATH_CALUDE_zongzi_survey_measure_l1894_189430

-- Define the types of statistical measures
inductive StatMeasure
| Variance
| Mean
| Median
| Mode

-- Define a function that determines the most appropriate measure
def most_appropriate_measure (survey_goal : String) (data_type : String) : StatMeasure :=
  if survey_goal = "determine most preferred" && data_type = "categorical" then
    StatMeasure.Mode
  else
    StatMeasure.Mean  -- Default to mean for other cases

-- Theorem statement
theorem zongzi_survey_measure :
  most_appropriate_measure "determine most preferred" "categorical" = StatMeasure.Mode :=
by sorry

end NUMINAMATH_CALUDE_zongzi_survey_measure_l1894_189430


namespace NUMINAMATH_CALUDE_reporters_not_covering_politics_l1894_189435

/-- The percentage of reporters who cover local politics in country X -/
def local_politics_coverage : ℝ := 5

/-- The percentage of reporters who cover politics but not local politics in country X -/
def non_local_politics_coverage : ℝ := 30

/-- The percentage of reporters who cover politics and local politics in country X -/
def local_politics_ratio : ℝ := 100 - non_local_politics_coverage

theorem reporters_not_covering_politics (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (p : ℝ), abs (p - 92.86) < ε ∧ 
  p = 100 - (local_politics_coverage * 100 / local_politics_ratio) :=
sorry

end NUMINAMATH_CALUDE_reporters_not_covering_politics_l1894_189435


namespace NUMINAMATH_CALUDE_number_to_add_for_divisibility_l1894_189418

theorem number_to_add_for_divisibility (n m k : ℕ) (h1 : n = 956734) (h2 : m = 412) (h3 : k = 390) :
  (n + k) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_number_to_add_for_divisibility_l1894_189418


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1894_189458

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  F₁.1 = 4 ∧ F₁.2 = 0 ∧ F₂.1 = -4 ∧ F₂.2 = 0

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 → foci F₁ F₂ →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) +
  Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) = 18 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1894_189458


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l1894_189467

/-- Given a pet store with bird cages, each containing parrots and parakeets,
    calculate the total number of birds. -/
theorem pet_store_bird_count (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) :
  num_cages = 6 →
  parrots_per_cage = 2 →
  parakeets_per_cage = 7 →
  num_cages * (parrots_per_cage + parakeets_per_cage) = 54 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l1894_189467


namespace NUMINAMATH_CALUDE_adqr_is_cyclic_l1894_189461

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point lies on a line segment between two other points -/
def point_on_segment (P Q R : Point) : Prop := sorry

/-- Checks if two line segments have equal length -/
def segments_equal (A B C D : Point) : Prop := sorry

/-- Checks if a quadrilateral is cyclic (can be inscribed in a circle) -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- Main theorem -/
theorem adqr_is_cyclic 
  (A B C D P Q R T : Point)
  (h_convex : is_convex ⟨A, B, C, D⟩)
  (h_equal1 : segments_equal A P P T)
  (h_equal2 : segments_equal P T T D)
  (h_equal3 : segments_equal Q B B C)
  (h_equal4 : segments_equal B C C R)
  (h_on_AB1 : point_on_segment A P B)
  (h_on_AB2 : point_on_segment A Q B)
  (h_on_CD1 : point_on_segment C R D)
  (h_on_CD2 : point_on_segment C T D)
  (h_bctp_cyclic : is_cyclic ⟨B, C, T, P⟩) :
  is_cyclic ⟨A, D, Q, R⟩ :=
sorry

end NUMINAMATH_CALUDE_adqr_is_cyclic_l1894_189461


namespace NUMINAMATH_CALUDE_intersection_M_N_l1894_189420

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 2*x + 8)}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = abs x + 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | -2 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1894_189420


namespace NUMINAMATH_CALUDE_equal_distribution_contribution_l1894_189423

def earnings : List ℕ := [18, 22, 30, 35, 45]

theorem equal_distribution_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  match max_earner with
  | some max => max - equal_share = 15
  | none => False
  := by sorry

end NUMINAMATH_CALUDE_equal_distribution_contribution_l1894_189423


namespace NUMINAMATH_CALUDE_doubled_container_volume_l1894_189439

/-- The volume of a container after doubling its dimensions -/
def doubled_volume (original_volume : ℝ) : ℝ := 8 * original_volume

/-- Theorem: Doubling the dimensions of a 4-gallon container results in a 32-gallon container -/
theorem doubled_container_volume : doubled_volume 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_doubled_container_volume_l1894_189439


namespace NUMINAMATH_CALUDE_solve_equation_l1894_189426

theorem solve_equation : 
  ∃ x : ℚ, 64 + 5 * x / (180 / 3) = 65 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1894_189426


namespace NUMINAMATH_CALUDE_ball_pit_problem_l1894_189486

theorem ball_pit_problem (total : ℕ) (red_fraction : ℚ) (blue_fraction : ℚ) : 
  total = 360 →
  red_fraction = 1/4 →
  blue_fraction = 1/5 →
  ∃ (red blue neither : ℕ),
    red = total * red_fraction ∧
    blue = (total - red) * blue_fraction ∧
    neither = total - red - blue ∧
    neither = 216 :=
by sorry

end NUMINAMATH_CALUDE_ball_pit_problem_l1894_189486


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l1894_189431

theorem tripled_base_and_exponent (a b x : ℝ) (hb : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ (2 * b) → x = 3 * Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l1894_189431


namespace NUMINAMATH_CALUDE_hotel_room_charges_l1894_189485

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * 0.8)  -- P is 20% less than R
  (h2 : P = G * 0.9)  -- P is 10% less than G
  : R = G * 1.125 :=  -- R is 12.5% greater than G
by sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l1894_189485


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1894_189489

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + a

-- Define the theorem
theorem solution_set_implies_a_equals_one :
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) →
  (∃ (a : ℝ), a = 1 ∧ ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1894_189489
