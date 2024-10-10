import Mathlib

namespace real_part_of_x_l288_28826

-- Define the variables and their types
variable (x : ℂ) -- x is a complex number
variable (y z : ℝ) -- y and z are real numbers
variable (p q : ℕ) -- p and q are natural numbers (we'll define them as prime later)
variable (n m : ℕ) -- n and m are non-negative integers
variable (k : ℕ) -- k is a natural number (we'll define it as odd prime later)

-- Define the conditions
axiom p_prime : Nat.Prime p
axiom q_prime : Nat.Prime q
axiom p_ne_q : p ≠ q
axiom k_odd_prime : Nat.Prime k ∧ k % 2 = 1
axiom least_p_q : ∀ p' q', Nat.Prime p' → Nat.Prime q' → p' ≠ q' → (p < p' ∨ q < q')

-- Define the specific values
axiom n_val : n = 2
axiom m_val : m = 3
axiom y_val : y = 5
axiom z_val : z = 10

-- Define the system of equations
axiom eq1 : x^n / (12 * ↑p * ↑q) = ↑k
axiom eq2 : x^m + y = z

-- Theorem to prove
theorem real_part_of_x :
  ∃ r : ℝ, (r = 6 * Real.sqrt 6 ∨ r = -6 * Real.sqrt 6) ∧ x.re = r :=
sorry

end real_part_of_x_l288_28826


namespace f_is_linear_l288_28864

/-- A function representing the total price of masks based on quantity -/
def f (x : ℝ) : ℝ := 0.9 * x

/-- The unit price of a mask in yuan -/
def unit_price : ℝ := 0.9

/-- Theorem stating that f is a linear function -/
theorem f_is_linear : 
  ∃ (m b : ℝ), ∀ x, f x = m * x + b :=
sorry

end f_is_linear_l288_28864


namespace series_sum_after_removal_equals_neg_3026_l288_28824

def series_sum (n : ℕ) : ℤ :=
  if n % 4 = 0 then
    (n - 3) - (n - 2) - (n - 1) + n
  else if n % 4 = 1 then
    n - (n + 1) - (n + 2)
  else
    0

def remove_multiples_of_10 (n : ℤ) : ℤ :=
  if n % 10 = 0 then 0 else n

def final_sum : ℤ :=
  (List.range 2015).foldl (λ acc i => acc + remove_multiples_of_10 (series_sum (i + 1))) 0

theorem series_sum_after_removal_equals_neg_3026 :
  final_sum = -3026 :=
sorry

end series_sum_after_removal_equals_neg_3026_l288_28824


namespace percentage_difference_l288_28886

theorem percentage_difference : 
  (60 * (50 / 100) * (40 / 100)) - (70 * (60 / 100) * (50 / 100)) = 9 := by
  sorry

end percentage_difference_l288_28886


namespace range_of_a_l288_28846

theorem range_of_a (a : ℝ) : 
  (∀ x, 2*x^2 - x - 1 ≤ 0 → x^2 - (2*a-1)*x + a*(a-1) ≤ 0) ∧ 
  (∃ x, 2*x^2 - x - 1 ≤ 0 ∧ x^2 - (2*a-1)*x + a*(a-1) > 0) →
  1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end range_of_a_l288_28846


namespace q_div_p_equals_225_l288_28810

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability of drawing 5 cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing 4 cards with one number and 1 card with a different number -/
def q : ℚ := ((distinct_numbers * (distinct_numbers - 1) * cards_per_number * cards_per_number) : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The theorem stating that q/p equals 225 -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end q_div_p_equals_225_l288_28810


namespace simplify_expression_l288_28831

theorem simplify_expression (x : ℝ) : 3 * x^5 * (4 * x^3) = 12 * x^8 := by
  sorry

end simplify_expression_l288_28831


namespace rectangle_dimensions_l288_28853

theorem rectangle_dimensions (perimeter : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 26) (h_area : area = 42) :
  ∃ (length width : ℝ),
    length + width = perimeter / 2 ∧
    length * width = area ∧
    length = 7 ∧
    width = 6 := by
  sorry

end rectangle_dimensions_l288_28853


namespace swimming_time_difference_l288_28856

theorem swimming_time_difference 
  (distance : ℝ) 
  (jack_speed : ℝ) 
  (jill_speed : ℝ) 
  (h1 : distance = 1) 
  (h2 : jack_speed = 10) 
  (h3 : jill_speed = 4) : 
  (distance / jill_speed - distance / jack_speed) * 60 = 9 := by
  sorry

end swimming_time_difference_l288_28856


namespace existence_of_special_fractions_l288_28894

theorem existence_of_special_fractions : 
  ∃ (a b c d : ℕ), (a : ℚ) / b + (c : ℚ) / d = 1 ∧ (a : ℚ) / d + (d : ℚ) / b = 2008 := by
  sorry

end existence_of_special_fractions_l288_28894


namespace train_meeting_time_l288_28849

theorem train_meeting_time (distance : ℝ) (speed_diff : ℝ) (final_speed : ℝ) :
  distance = 450 →
  speed_diff = 6 →
  final_speed = 48 →
  (distance / (final_speed + (final_speed + speed_diff))) = 75 / 17 := by
  sorry

end train_meeting_time_l288_28849


namespace transform_equivalence_l288_28899

-- Define the original function
def f : ℝ → ℝ := sorry

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*x - 2) + 1

-- Define the horizontal shift
def shift_right (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*(x - 1))

-- Define the vertical shift
def shift_up (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1

-- Theorem statement
theorem transform_equivalence (x : ℝ) : 
  transform f x = shift_up (shift_right (f ∘ (fun x => 2*x))) x := by sorry

end transform_equivalence_l288_28899


namespace rectangle_area_with_inscribed_circle_l288_28865

/-- Given a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1, 
    prove that its area is 588. -/
theorem rectangle_area_with_inscribed_circle (r w l : ℝ) : 
  r = 7 ∧ w = 2 * r ∧ l = 3 * w → l * w = 588 := by
  sorry

end rectangle_area_with_inscribed_circle_l288_28865


namespace average_of_z_multiples_l288_28870

/-- The average of z, 4z, 10z, 22z, and 46z is 16.6z -/
theorem average_of_z_multiples (z : ℝ) : 
  (z + 4*z + 10*z + 22*z + 46*z) / 5 = 16.6 * z := by
  sorry

end average_of_z_multiples_l288_28870


namespace inscribed_cube_side_length_is_sqrt6_div_2_l288_28817

/-- Represents a pyramid with a regular hexagonal base and equilateral triangle lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_is_equilateral : Bool

/-- Represents a cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  bottom_covers_base : Bool
  top_touches_midpoints : Bool

/-- Calculates the side length of an inscribed cube in a hexagonal pyramid -/
def inscribed_cube_side_length (cube : InscribedCube) : ℝ :=
  sorry

/-- Theorem stating that the side length of the inscribed cube is √6/2 -/
theorem inscribed_cube_side_length_is_sqrt6_div_2 
  (cube : InscribedCube) 
  (h1 : cube.pyramid.base_side_length = 2)
  (h2 : cube.pyramid.lateral_face_is_equilateral = true)
  (h3 : cube.bottom_covers_base = true)
  (h4 : cube.top_touches_midpoints = true) :
  inscribed_cube_side_length cube = Real.sqrt 6 / 2 :=
sorry

end inscribed_cube_side_length_is_sqrt6_div_2_l288_28817


namespace green_peaches_count_l288_28801

/-- The number of peaches in a basket --/
structure Basket :=
  (red : ℕ)
  (yellow : ℕ)
  (green : ℕ)

/-- The basket with the given conditions --/
def my_basket : Basket :=
  { red := 2
  , yellow := 6
  , green := 6 + 8 }

/-- Theorem stating that the number of green peaches is 14 --/
theorem green_peaches_count (b : Basket) 
  (h1 : b.red = 2) 
  (h2 : b.yellow = 6) 
  (h3 : b.green = b.yellow + 8) : 
  b.green = 14 := by
  sorry

end green_peaches_count_l288_28801


namespace intersection_values_l288_28890

-- Define the complex plane
variable (z : ℂ)

-- Define the equation |z - 4| = 3|z + 4|
def equation (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 4)

-- Define the intersection condition
def intersects_once (k : ℝ) : Prop :=
  ∃! z, equation z ∧ Complex.abs z = k

-- Theorem statement
theorem intersection_values :
  ∀ k, intersects_once k → k = 2 ∨ k = 14 :=
sorry

end intersection_values_l288_28890


namespace pentagon_x_coordinate_l288_28834

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a horizontal line of symmetry -/
def hasHorizontalSymmetry (p : Pentagon) : Prop := sorry

theorem pentagon_x_coordinate :
  ∀ (p : Pentagon) (xc : ℝ),
    p.A = (0, 0) →
    p.B = (0, 6) →
    p.C = (xc, 12) →
    p.D = (6, 6) →
    p.E = (6, 0) →
    hasHorizontalSymmetry p →
    pentagonArea p = 60 →
    xc = 8 := by
  sorry

end pentagon_x_coordinate_l288_28834


namespace equilateral_triangle_area_perimeter_ratio_l288_28814

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l288_28814


namespace polynomial_coefficient_sum_l288_28804

theorem polynomial_coefficient_sum :
  ∀ (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ),
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end polynomial_coefficient_sum_l288_28804


namespace cosA_value_triangle_area_l288_28891

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part I
theorem cosA_value (t : Triangle) 
  (h1 : t.a^2 = 3 * t.b * t.c) 
  (h2 : Real.sin t.A = Real.sin t.C) : 
  Real.cos t.A = 1/6 := by
sorry

-- Part II
theorem triangle_area (t : Triangle) 
  (h1 : t.a^2 = 3 * t.b * t.c) 
  (h2 : t.A = π/4) 
  (h3 : t.a = 3) : 
  (1/2) * t.b * t.c * Real.sin t.A = (3/4) * Real.sqrt 2 := by
sorry

end cosA_value_triangle_area_l288_28891


namespace problem_1_problem_2_problem_3_problem_4_l288_28872

-- Problem 1
theorem problem_1 : (-10) - (-4) + 5 = -1 := by sorry

-- Problem 2
theorem problem_2 : (-72) * (2/3 - 1/4 - 5/6) = 30 := by sorry

-- Problem 3
theorem problem_3 : -3^2 - (-2)^3 * (-1)^4 + Real.rpow 27 (1/3) = 2 := by sorry

-- Problem 4
theorem problem_4 : 5 + 4 * (Real.sqrt 6 - 2) - 4 * (Real.sqrt 6 - 1) = 1 := by sorry

end problem_1_problem_2_problem_3_problem_4_l288_28872


namespace derivative_of_f_l288_28850

noncomputable def f (x : ℝ) : ℝ := (2^x * (Real.sin x + Real.cos x * Real.log 2)) / (1 + Real.log 2 ^ 2)

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2^x * Real.cos x :=
by sorry

end derivative_of_f_l288_28850


namespace wheel_turns_l288_28836

/-- A wheel makes 6 turns every 30 seconds. This theorem proves that it makes 1440 turns in 2 hours. -/
theorem wheel_turns (turns_per_30_sec : ℕ) (hours : ℕ) : 
  turns_per_30_sec = 6 → hours = 2 → turns_per_30_sec * 240 * hours = 1440 := by
  sorry

end wheel_turns_l288_28836


namespace sin_alpha_for_point_l288_28889

/-- If the terminal side of angle α passes through the point (-1, 2) in the Cartesian coordinate system, then sin α = (2√5) / 5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
  sorry

end sin_alpha_for_point_l288_28889


namespace solutions_equation1_solutions_equation2_l288_28896

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 3 = 0
def equation2 (y : ℝ) : Prop := 4*(2*y - 5)^2 = (3*y - 1)^2

-- Theorem for the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 1 ∧ equation1 3) :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  (∃ y : ℝ, equation2 y) ↔ (equation2 9 ∧ equation2 (11/7)) :=
sorry

end solutions_equation1_solutions_equation2_l288_28896


namespace max_lessons_l288_28892

/-- Represents the number of shirts the teacher has. -/
def s : ℕ := sorry

/-- Represents the number of pairs of pants the teacher has. -/
def p : ℕ := sorry

/-- Represents the number of pairs of shoes the teacher has. -/
def b : ℕ := sorry

/-- Represents the number of jackets the teacher has. -/
def jackets : ℕ := 2

/-- Represents the total number of possible lessons. -/
def total_lessons : ℕ := 2 * s * p * b

/-- States that one more shirt would allow 36 more lessons. -/
axiom shirt_condition : 2 * (s + 1) * p * b = total_lessons + 36

/-- States that one more pair of pants would allow 72 more lessons. -/
axiom pants_condition : 2 * s * (p + 1) * b = total_lessons + 72

/-- States that one more pair of shoes would allow 54 more lessons. -/
axiom shoes_condition : 2 * s * p * (b + 1) = total_lessons + 54

/-- Theorem stating the maximum number of lessons the teacher could have conducted. -/
theorem max_lessons : total_lessons = 216 := by sorry

end max_lessons_l288_28892


namespace min_omega_value_l288_28893

/-- Given a function f(x) = 2 sin(ωx) with a minimum value of 2 on the interval [-π/3, π/4],
    the minimum value of ω is 3/2 -/
theorem min_omega_value (ω : ℝ) (h : ∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) ≥ 2) :
  ω ≥ 3/2 := by
  sorry

end min_omega_value_l288_28893


namespace number_comparison_l288_28841

def A : ℕ := 888888888888888888888  -- 19 eights
def B : ℕ := 333333333333333333333333333333333333333333333333333333333333333333333  -- 68 threes
def C : ℕ := 444444444444444444444  -- 19 fours
def D : ℕ := 666666666666666666666666666666666666666666666666666666666666666666667  -- 67 sixes and one seven

theorem number_comparison : C * D - A * B = 444444444444444444444 := by
  sorry

end number_comparison_l288_28841


namespace sin_315_degrees_l288_28847

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_315_degrees_l288_28847


namespace subtraction_of_fractions_l288_28812

theorem subtraction_of_fractions : (1 : ℚ) / 6 - (5 : ℚ) / 12 = (-1 : ℚ) / 4 := by sorry

end subtraction_of_fractions_l288_28812


namespace dot_product_AB_BC_l288_28866

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 7 ∧ BC = 5 ∧ CA = 6

-- Define the dot product of two 2D vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Theorem statement
theorem dot_product_AB_BC (A B C : ℝ × ℝ) :
  triangle_ABC A B C →
  dot_product (B.1 - A.1, B.2 - A.2) (C.1 - B.1, C.2 - B.2) = -19 :=
sorry

end dot_product_AB_BC_l288_28866


namespace inequality_proof_l288_28823

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y)) ≥ 3/4 := by
  sorry

end inequality_proof_l288_28823


namespace total_blue_balloons_l288_28897

theorem total_blue_balloons (joan_balloons sally_balloons jessica_balloons : ℕ) 
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : jessica_balloons = 2) :
  joan_balloons + sally_balloons + jessica_balloons = 16 := by
  sorry

end total_blue_balloons_l288_28897


namespace duty_arrangement_for_three_leaders_l288_28858

/-- The number of ways to arrange n leaders for duty over d days, 
    with each leader on duty for m days. -/
def dutyArrangements (n d m : ℕ) : ℕ := sorry

/-- The number of combinations of n items taken k at a time. -/
def nCk (n k : ℕ) : ℕ := sorry

theorem duty_arrangement_for_three_leaders :
  dutyArrangements 3 6 2 = 90 :=
by
  sorry

end duty_arrangement_for_three_leaders_l288_28858


namespace max_value_w_l288_28805

theorem max_value_w (p q : ℝ) 
  (h1 : 2 * p - q ≥ 0) 
  (h2 : 3 * q - 2 * p ≥ 0) 
  (h3 : 6 - 2 * q ≥ 0) : 
  Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q) ≤ 3 * Real.sqrt 2 ∧
  (Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q) = 3 * Real.sqrt 2 ↔ p = 2 ∧ q = 2) :=
by sorry

end max_value_w_l288_28805


namespace remainder_101_103_div_11_l288_28819

theorem remainder_101_103_div_11 : (101 * 103) % 11 = 8 := by
  sorry

end remainder_101_103_div_11_l288_28819


namespace sundae_booth_packs_l288_28845

/-- Calculates the number of packs needed for a given topping -/
def packs_needed (total_items : ℕ) (items_per_pack : ℕ) : ℕ :=
  (total_items + items_per_pack - 1) / items_per_pack

/-- Represents the sundae booth problem -/
theorem sundae_booth_packs (monday_sundaes tuesday_sundaes : ℕ)
  (monday_mms monday_gummy monday_marsh : ℕ)
  (tuesday_mms tuesday_gummy tuesday_marsh : ℕ)
  (mms_per_pack gummy_per_pack marsh_per_pack : ℕ)
  (h_monday : monday_sundaes = 40)
  (h_tuesday : tuesday_sundaes = 20)
  (h_monday_mms : monday_mms = 6)
  (h_monday_gummy : monday_gummy = 4)
  (h_monday_marsh : monday_marsh = 8)
  (h_tuesday_mms : tuesday_mms = 10)
  (h_tuesday_gummy : tuesday_gummy = 5)
  (h_tuesday_marsh : tuesday_marsh = 12)
  (h_mms_pack : mms_per_pack = 40)
  (h_gummy_pack : gummy_per_pack = 30)
  (h_marsh_pack : marsh_per_pack = 50) :
  (packs_needed (monday_sundaes * monday_mms + tuesday_sundaes * tuesday_mms) mms_per_pack = 11) ∧
  (packs_needed (monday_sundaes * monday_gummy + tuesday_sundaes * tuesday_gummy) gummy_per_pack = 9) ∧
  (packs_needed (monday_sundaes * monday_marsh + tuesday_sundaes * tuesday_marsh) marsh_per_pack = 12) :=
by sorry

end sundae_booth_packs_l288_28845


namespace scooter_gain_percent_l288_28898

/-- Calculate the gain percent from a scooter sale -/
theorem scooter_gain_percent 
  (purchase_price : ℝ) 
  (repair_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 800)
  (h2 : repair_cost = 200)
  (h3 : selling_price = 1200) : 
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 20 := by
  sorry

end scooter_gain_percent_l288_28898


namespace quadratic_root_existence_l288_28895

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_root_existence (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x : ℝ, x > 0.6 ∧ x < 0.7 ∧ quadratic_function a b c x = 0 :=
by
  have h1 : quadratic_function a b c 0.6 < 0 := by sorry
  have h2 : quadratic_function a b c 0.7 > 0 := by sorry
  sorry

#check quadratic_root_existence

end quadratic_root_existence_l288_28895


namespace quadratic_inequality_l288_28840

theorem quadratic_inequality (x : ℝ) : x^2 - 10*x + 21 < 0 ↔ 3 < x ∧ x < 7 := by
  sorry

end quadratic_inequality_l288_28840


namespace stone_statue_cost_is_20_l288_28832

/-- The cost of a stone statue -/
def stone_statue_cost : ℚ := 20

/-- The number of stone statues produced monthly -/
def stone_statues_per_month : ℕ := 10

/-- The number of wooden statues produced monthly -/
def wooden_statues_per_month : ℕ := 20

/-- The cost of a wooden statue -/
def wooden_statue_cost : ℚ := 5

/-- The tax rate as a decimal -/
def tax_rate : ℚ := 1/10

/-- The monthly earnings after taxes -/
def monthly_earnings_after_taxes : ℚ := 270

/-- Theorem stating that the cost of a stone statue is $20 -/
theorem stone_statue_cost_is_20 :
  stone_statue_cost * stone_statues_per_month +
  wooden_statue_cost * wooden_statues_per_month =
  monthly_earnings_after_taxes / (1 - tax_rate) :=
by sorry

end stone_statue_cost_is_20_l288_28832


namespace vector_equation_l288_28822

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_equation : 4 • a - 3 • (a + b) = a - 3 • b := by sorry

end vector_equation_l288_28822


namespace four_fold_f_application_l288_28857

-- Define the function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z ^ 2 else -(z ^ 2)

-- State the theorem
theorem four_fold_f_application :
  f (f (f (f (2 + 2*I)))) = -16777216 := by
  sorry

end four_fold_f_application_l288_28857


namespace nina_payment_l288_28803

theorem nina_payment (x y z w : ℕ) : 
  x + y + z + w = 27 →  -- Total number of coins
  y = 2 * z →           -- Number of 5 kopek coins is twice the number of 2 kopek coins
  z = 2 * x →           -- Number of 2 kopek coins is twice the number of 10 kopek coins
  7 < w →               -- Number of 3 kopek coins is more than 7
  w < 20 →              -- Number of 3 kopek coins is less than 20
  10 * x + 5 * y + 2 * z + 3 * w = 107 := by
sorry

end nina_payment_l288_28803


namespace probability_square_or_circle_l288_28879

theorem probability_square_or_circle (total : ℕ) (triangles squares circles : ℕ) : 
  total = triangles + squares + circles →
  triangles = 4 →
  squares = 3 →
  circles = 5 →
  (squares + circles : ℚ) / total = 2 / 3 := by
  sorry

end probability_square_or_circle_l288_28879


namespace sum_with_radical_conjugate_l288_28811

theorem sum_with_radical_conjugate :
  let x : ℝ := 5 - Real.sqrt 500
  let y : ℝ := 5 + Real.sqrt 500
  x + y = 10 := by sorry

end sum_with_radical_conjugate_l288_28811


namespace expected_remaining_bullets_value_l288_28807

/-- The probability of hitting the target -/
def p : ℝ := 0.6

/-- The total number of bullets -/
def n : ℕ := 4

/-- The expected number of remaining bullets -/
def expected_remaining_bullets : ℝ :=
  (n - 1) * p + (n - 2) * (1 - p) * p + (n - 3) * (1 - p)^2 * p + 0 * (1 - p)^3 * p

/-- Theorem stating the expected number of remaining bullets -/
theorem expected_remaining_bullets_value :
  expected_remaining_bullets = 2.376 := by sorry

end expected_remaining_bullets_value_l288_28807


namespace isabelle_concert_savings_l288_28861

/-- Calculates the number of weeks Isabelle must work to afford concert tickets for herself and her brothers. -/
theorem isabelle_concert_savings (isabelle_ticket : ℕ) (brother_ticket : ℕ) (isabelle_savings : ℕ) (brothers_savings : ℕ) (weekly_earnings : ℕ) : 
  isabelle_ticket = 20 →
  brother_ticket = 10 →
  isabelle_savings = 5 →
  brothers_savings = 5 →
  weekly_earnings = 3 →
  (isabelle_ticket + 2 * brother_ticket - isabelle_savings - brothers_savings) / weekly_earnings = 10 := by
sorry

end isabelle_concert_savings_l288_28861


namespace no_solution_system_l288_28873

/-- Proves that the system of equations 3x - 4y = 5 and 6x - 8y = 7 has no solution -/
theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 5) ∧ (6 * x - 8 * y = 7) := by
sorry

end no_solution_system_l288_28873


namespace sum_of_squares_l288_28828

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 12)
  (eq2 : y^2 + 5*z = -15)
  (eq3 : z^2 + 7*x = -21) :
  x^2 + y^2 + z^2 = 83/4 := by
sorry

end sum_of_squares_l288_28828


namespace two_color_theorem_l288_28815

/-- A line in a plane --/
structure Line where
  -- We don't need to define the specifics of a line for this problem

/-- A region in a plane formed by intersecting lines --/
structure Region where
  -- We don't need to define the specifics of a region for this problem

/-- A color used for coloring regions --/
inductive Color
  | Red
  | Blue

/-- A configuration of lines in a plane --/
def Configuration := List Line

/-- A coloring of regions --/
def Coloring := Region → Color

/-- Check if two regions are adjacent --/
def adjacent (r1 r2 : Region) : Prop :=
  sorry -- Definition of adjacency

/-- A valid coloring ensures no adjacent regions have the same color --/
def valid_coloring (c : Configuration) (coloring : Coloring) : Prop :=
  ∀ r1 r2 : Region, adjacent r1 r2 → coloring r1 ≠ coloring r2

/-- The main theorem: for any configuration of lines, there exists a valid coloring --/
theorem two_color_theorem (c : Configuration) : 
  ∃ coloring : Coloring, valid_coloring c coloring :=
sorry

end two_color_theorem_l288_28815


namespace yellow_marbles_fraction_l288_28806

theorem yellow_marbles_fraction (total : ℝ) (h : total > 0) :
  let initial_green := (2/3) * total
  let initial_yellow := total - initial_green
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = 3/5 := by sorry

end yellow_marbles_fraction_l288_28806


namespace candy_distribution_l288_28838

theorem candy_distribution (initial_candy : ℕ) (eaten : ℕ) (bowls : ℕ) (taken : ℕ) : 
  initial_candy = 100 →
  eaten = 8 →
  bowls = 4 →
  taken = 3 →
  (initial_candy - eaten) / bowls - taken = 20 :=
by
  sorry

end candy_distribution_l288_28838


namespace stick_cutting_theorem_l288_28887

/-- Represents a marked stick with cuts -/
structure MarkedStick :=
  (length : ℕ)
  (left_interval : ℕ)
  (right_interval : ℕ)

/-- Counts the number of segments of a given length in a marked stick -/
def count_segments (stick : MarkedStick) (segment_length : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that a 240 cm stick marked as described yields 12 pieces of 3 cm -/
theorem stick_cutting_theorem :
  let stick : MarkedStick := ⟨240, 7, 6⟩
  count_segments stick 3 = 12 := by sorry

end stick_cutting_theorem_l288_28887


namespace triangle_angle_sum_l288_28827

theorem triangle_angle_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 90) (h3 : b = 58) : c = 32 := by
  sorry

#check triangle_angle_sum

end triangle_angle_sum_l288_28827


namespace glorias_turtle_finish_time_l288_28809

/-- The finish time of Gloria's turtle in the Key West Turtle Race -/
def glorias_turtle_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

/-- Theorem stating that Gloria's turtle finish time is 8 minutes -/
theorem glorias_turtle_finish_time :
  ∃ (gretas_time georges_time : ℕ),
    gretas_time = 6 ∧
    georges_time = gretas_time - 2 ∧
    glorias_turtle_time gretas_time georges_time = 8 :=
by
  sorry


end glorias_turtle_finish_time_l288_28809


namespace triangle_roots_range_l288_28802

theorem triangle_roots_range (m : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ - 2) * (x₁^2 - 4*x₁ + m) = 0 ∧
    (x₂ - 2) * (x₂^2 - 4*x₂ + m) = 0 ∧
    (x₃ - 2) * (x₃^2 - 4*x₃ + m) = 0 ∧
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    x₁ + x₂ > x₃ ∧ x₂ + x₃ > x₁ ∧ x₃ + x₁ > x₂) →
  3 < m ∧ m < 4 :=
by sorry

end triangle_roots_range_l288_28802


namespace incorrect_step_is_count_bacteria_l288_28855

/-- Represents a step in the bacterial counting experiment -/
inductive ExperimentStep
  | PrepMedium
  | SpreadSamples
  | Incubate
  | CountBacteria

/-- Represents a range of bacterial counts -/
structure CountRange where
  lower : ℕ
  upper : ℕ

/-- Defines the correct count range for bacterial counting -/
def correct_count_range : CountRange := { lower := 30, upper := 300 }

/-- Defines whether a step is correct in the experiment -/
def is_correct_step (step : ExperimentStep) : Prop :=
  match step with
  | ExperimentStep.PrepMedium => True
  | ExperimentStep.SpreadSamples => True
  | ExperimentStep.Incubate => True
  | ExperimentStep.CountBacteria => False

/-- Theorem stating that the CountBacteria step is the incorrect one -/
theorem incorrect_step_is_count_bacteria :
  ∃ (step : ExperimentStep), ¬(is_correct_step step) ↔ step = ExperimentStep.CountBacteria :=
sorry

end incorrect_step_is_count_bacteria_l288_28855


namespace max_consecutive_sum_15_l288_28884

/-- The sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A sequence of n consecutive positive integers starting from k -/
def consecutive_sum (n k : ℕ) : ℕ := n * k + triangular_number n

theorem max_consecutive_sum_15 :
  (∃ (n : ℕ), n > 0 ∧ consecutive_sum n 1 = 15) ∧
  (∀ (m : ℕ), m > 5 → consecutive_sum m 1 > 15) :=
sorry

end max_consecutive_sum_15_l288_28884


namespace intersection_area_theorem_l288_28871

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by its vertices -/
structure Rectangle where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a circle defined by its center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the area of intersection between a rectangle and a circle -/
def intersectionArea (r : Rectangle) (c : Circle) : ℝ := sorry

/-- The main theorem stating the area of intersection -/
theorem intersection_area_theorem (r : Rectangle) (c : Circle) : 
  r.v1 = ⟨3, 9⟩ → 
  r.v2 = ⟨20, 9⟩ → 
  r.v3 = ⟨20, -6⟩ → 
  r.v4 = ⟨3, -6⟩ → 
  c.center = ⟨3, -6⟩ → 
  c.radius = 5 → 
  intersectionArea r c = 25 * Real.pi / 4 := by sorry

end intersection_area_theorem_l288_28871


namespace subset_sum_divisible_by_2n_l288_28848

theorem subset_sum_divisible_by_2n (n : ℕ) (a : Fin n → ℕ) 
  (h1 : n ≥ 4)
  (h2 : ∀ i j, i ≠ j → a i ≠ a j)
  (h3 : ∀ i, 0 < a i ∧ a i < 2*n) :
  ∃ (i j : Fin n), i < j ∧ (2*n) ∣ (a i + a j) :=
sorry

end subset_sum_divisible_by_2n_l288_28848


namespace danes_daughters_flowers_l288_28818

theorem danes_daughters_flowers (total_baskets : Nat) (flowers_per_basket : Nat) 
  (growth : Nat) (died : Nat) (num_daughters : Nat) :
  total_baskets = 5 →
  flowers_per_basket = 4 →
  growth = 20 →
  died = 10 →
  num_daughters = 2 →
  (total_baskets * flowers_per_basket + died - growth) / num_daughters = 5 := by
  sorry

end danes_daughters_flowers_l288_28818


namespace quadratic_is_perfect_square_l288_28839

/-- 
For a quadratic expression of the form x^2 - 16x + k to be the square of a binomial,
k must equal 64.
-/
theorem quadratic_is_perfect_square (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 16*x + k = (a*x + b)^2) ↔ k = 64 := by
  sorry

end quadratic_is_perfect_square_l288_28839


namespace prob_even_and_divisible_by_three_on_two_dice_l288_28821

/-- The probability of rolling an even number on a six-sided die -/
def prob_even_on_six_sided_die : ℚ := 1/2

/-- The probability of rolling a number divisible by three on a six-sided die -/
def prob_divisible_by_three_on_six_sided_die : ℚ := 1/3

/-- The probability of rolling an even number on one six-sided die
    and a number divisible by three on another six-sided die -/
theorem prob_even_and_divisible_by_three_on_two_dice :
  prob_even_on_six_sided_die * prob_divisible_by_three_on_six_sided_die = 1/6 := by
  sorry

end prob_even_and_divisible_by_three_on_two_dice_l288_28821


namespace complex_equation_solution_l288_28825

theorem complex_equation_solution (x : ℝ) :
  (1 - 2*Complex.I) * (x + Complex.I) = 4 - 3*Complex.I → x = 2 :=
by
  sorry

end complex_equation_solution_l288_28825


namespace boys_playing_cards_l288_28816

theorem boys_playing_cards (total_marble_boys : ℕ) (total_marbles : ℕ) (marbles_per_boy : ℕ) :
  total_marble_boys = 13 →
  total_marbles = 26 →
  marbles_per_boy = 2 →
  total_marbles = total_marble_boys * marbles_per_boy →
  (total_marble_boys : ℤ) - (total_marbles / marbles_per_boy : ℤ) = 0 :=
by sorry

end boys_playing_cards_l288_28816


namespace bella_items_after_purchase_l288_28877

def total_items (marbles frisbees deck_cards action_figures : ℕ) : ℕ :=
  marbles + frisbees + deck_cards + action_figures

theorem bella_items_after_purchase : 
  ∀ (marbles frisbees deck_cards action_figures : ℕ),
    marbles = 60 →
    marbles = 2 * frisbees →
    frisbees = deck_cards + 20 →
    marbles = 5 * action_figures →
    total_items (marbles + (2 * marbles) / 5)
                (frisbees + (2 * frisbees) / 5)
                (deck_cards + (2 * deck_cards) / 5)
                (action_figures + action_figures / 3) = 156 := by
  sorry

#check bella_items_after_purchase

end bella_items_after_purchase_l288_28877


namespace shelter_new_pets_l288_28880

theorem shelter_new_pets (initial_dogs : ℕ) (initial_cats : ℕ) (initial_lizards : ℕ)
  (dog_adoption_rate : ℚ) (cat_adoption_rate : ℚ) (lizard_adoption_rate : ℚ)
  (pets_after_month : ℕ) :
  initial_dogs = 30 →
  initial_cats = 28 →
  initial_lizards = 20 →
  dog_adoption_rate = 1/2 →
  cat_adoption_rate = 1/4 →
  lizard_adoption_rate = 1/5 →
  pets_after_month = 65 →
  ∃ new_pets : ℕ,
    new_pets = 13 ∧
    pets_after_month = 
      (initial_dogs - initial_dogs * dog_adoption_rate).floor +
      (initial_cats - initial_cats * cat_adoption_rate).floor +
      (initial_lizards - initial_lizards * lizard_adoption_rate).floor +
      new_pets :=
by
  sorry

end shelter_new_pets_l288_28880


namespace simplify_trig_expression_l288_28837

theorem simplify_trig_expression :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 
    (1 / 2) * Real.cos (10 * π / 180) := by
  sorry

end simplify_trig_expression_l288_28837


namespace purple_balls_count_l288_28881

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 30 ∧
  yellow = 8 ∧
  red = 9 ∧
  prob = 88/100 ∧
  prob = (white + green + yellow : ℚ) / total →
  total - (white + green + yellow + red) = 0 :=
by sorry

end purple_balls_count_l288_28881


namespace freight_train_speed_l288_28860

/-- Proves that the speed of the freight train is 50 km/hr given the problem conditions --/
theorem freight_train_speed 
  (distance : ℝ) 
  (speed_difference : ℝ) 
  (express_speed : ℝ) 
  (time : ℝ) 
  (h1 : distance = 390) 
  (h2 : speed_difference = 30) 
  (h3 : express_speed = 80) 
  (h4 : time = 3) 
  (h5 : distance = (express_speed * time) + ((express_speed - speed_difference) * time)) : 
  express_speed - speed_difference = 50 := by
  sorry

end freight_train_speed_l288_28860


namespace xy_value_l288_28843

theorem xy_value (x y : ℝ) : y = Real.sqrt (x - 1/2) + Real.sqrt (1/2 - x) - 6 → x * y = -3 := by
  sorry

end xy_value_l288_28843


namespace company_production_l288_28835

/-- The number of bottles a case can hold -/
def bottles_per_case : ℕ := 13

/-- The number of cases required for one-day production -/
def cases_per_day : ℕ := 5000

/-- The total number of bottles produced in one day -/
def bottles_per_day : ℕ := bottles_per_case * cases_per_day

/-- Theorem stating that the company produces 65,000 bottles per day -/
theorem company_production : bottles_per_day = 65000 := by
  sorry

end company_production_l288_28835


namespace total_length_is_16cm_l288_28876

/-- Represents the dimensions of a rectangle in centimeters -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the segments removed from the rectangle -/
structure RemovedSegments where
  long_side : ℝ
  short_side_ends : ℝ

/-- Represents the split in the remaining short side -/
structure SplitSegment where
  distance_from_middle : ℝ

/-- Calculates the total length of segments after modifications -/
def total_length_after_modifications (rect : Rectangle) (removed : RemovedSegments) (split : SplitSegment) : ℝ :=
  let remaining_long_side := rect.length - removed.long_side
  let remaining_short_side := rect.width - 2 * removed.short_side_ends
  let split_segment := min split.distance_from_middle (remaining_short_side / 2)
  remaining_long_side + remaining_short_side + 2 * removed.short_side_ends

/-- Theorem stating that the total length of segments after modifications is 16 cm -/
theorem total_length_is_16cm (rect : Rectangle) (removed : RemovedSegments) (split : SplitSegment)
    (h1 : rect.length = 10)
    (h2 : rect.width = 5)
    (h3 : removed.long_side = 3)
    (h4 : removed.short_side_ends = 2)
    (h5 : split.distance_from_middle = 1) :
  total_length_after_modifications rect removed split = 16 := by
  sorry

end total_length_is_16cm_l288_28876


namespace window_width_is_24_inches_l288_28820

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure Window where
  pane : GlassPane
  num_columns : ℕ
  num_rows : ℕ
  border_width : ℝ

/-- Calculates the total width of the window -/
def total_width (w : Window) : ℝ :=
  w.num_columns * w.pane.width + (w.num_columns + 1) * w.border_width

/-- Theorem stating that the total width of the window is 24 inches -/
theorem window_width_is_24_inches (w : Window) 
  (h1 : w.pane.height / w.pane.width = 3 / 4)
  (h2 : w.border_width = 3)
  (h3 : w.num_columns = 3)
  (h4 : w.num_rows = 2) :
  total_width w = 24 := by
  sorry


end window_width_is_24_inches_l288_28820


namespace train_journey_equation_l288_28874

/-- Represents the equation for a train journey where:
    - x is the distance in km
    - The speed increases from 160 km/h to 200 km/h
    - The travel time reduces by 2.5 hours
-/
theorem train_journey_equation (x : ℝ) : x / 160 - x / 200 = 2.5 := by
  sorry

end train_journey_equation_l288_28874


namespace interest_payment_time_l288_28882

-- Define the principal amount
def principal : ℝ := 8000

-- Define the interest rates
def rate1 : ℝ := 0.08
def rate2 : ℝ := 0.10
def rate3 : ℝ := 0.12

-- Define the time periods
def time1 : ℝ := 4
def time2 : ℝ := 6

-- Define the total interest paid
def totalInterest : ℝ := 12160

-- Function to calculate interest
def calculateInterest (p : ℝ) (r : ℝ) (t : ℝ) : ℝ := p * r * t

-- Theorem statement
theorem interest_payment_time :
  ∃ t : ℝ, 
    calculateInterest principal rate1 time1 +
    calculateInterest principal rate2 time2 +
    calculateInterest principal rate3 (t - (time1 + time2)) = totalInterest ∧
    t = 15 := by sorry

end interest_payment_time_l288_28882


namespace solve_seashells_problem_l288_28883

def seashells_problem (initial_seashells current_seashells : ℕ) : Prop :=
  ∃ (given_seashells : ℕ), 
    initial_seashells = current_seashells + given_seashells

theorem solve_seashells_problem (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_problem initial_seashells current_seashells →
  ∃ (given_seashells : ℕ), given_seashells = initial_seashells - current_seashells :=
by
  sorry

end solve_seashells_problem_l288_28883


namespace tom_found_four_seashells_today_l288_28885

/-- The number of seashells Tom found yesterday -/
def yesterdays_seashells : ℕ := 7

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := 11

/-- The number of seashells Tom found today -/
def todays_seashells : ℕ := total_seashells - yesterdays_seashells

theorem tom_found_four_seashells_today : todays_seashells = 4 := by
  sorry

end tom_found_four_seashells_today_l288_28885


namespace defective_units_shipped_l288_28830

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.09)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * 100) = 0.36 := by
  sorry

end defective_units_shipped_l288_28830


namespace larger_number_is_eight_l288_28833

theorem larger_number_is_eight (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 := by
  sorry

end larger_number_is_eight_l288_28833


namespace arithmetic_sequence_sum_l288_28862

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 12 and S_20 = 17, prove S_30 = 15 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence)
    (h1 : a.S 10 = 12)
    (h2 : a.S 20 = 17) :
  a.S 30 = 15 := by
  sorry

end arithmetic_sequence_sum_l288_28862


namespace chrysler_has_23_floors_l288_28878

/-- The number of floors in the Leeward Center -/
def leeward_floors : ℕ := sorry

/-- The number of floors in the Chrysler Building -/
def chrysler_floors : ℕ := sorry

/-- The Chrysler Building has 11 more floors than the Leeward Center -/
axiom chrysler_leeward_difference : chrysler_floors = leeward_floors + 11

/-- The total number of floors in both buildings is 35 -/
axiom total_floors : leeward_floors + chrysler_floors = 35

/-- Theorem: The Chrysler Building has 23 floors -/
theorem chrysler_has_23_floors : chrysler_floors = 23 := by sorry

end chrysler_has_23_floors_l288_28878


namespace houses_on_block_l288_28854

/-- Given a block of houses where:
  * The total number of pieces of junk mail for the block is 24
  * Each house receives 4 pieces of junk mail
  This theorem proves that there are 6 houses on the block. -/
theorem houses_on_block (total_mail : ℕ) (mail_per_house : ℕ) 
  (h1 : total_mail = 24) 
  (h2 : mail_per_house = 4) : 
  total_mail / mail_per_house = 6 := by
  sorry

end houses_on_block_l288_28854


namespace correct_product_with_decimals_l288_28808

theorem correct_product_with_decimals :
  let x : ℚ := 0.85
  let y : ℚ := 3.25
  let product_without_decimals : ℕ := 27625
  x * y = 2.7625 :=
by sorry

end correct_product_with_decimals_l288_28808


namespace random_selection_more_representative_l288_28888

/-- Represents a student in the school -/
structure Student where
  grade : ℕ
  gender : Bool

/-- Represents the entire student population of the school -/
def StudentPopulation := List Student

/-- Represents a sample of students -/
def StudentSample := List Student

/-- Function to check if a sample is representative of the population -/
def isRepresentative (population : StudentPopulation) (sample : StudentSample) : Prop :=
  -- Definition of what makes a sample representative
  sorry

/-- Function to randomly select students from various grades -/
def randomSelectFromGrades (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of random selection from various grades
  sorry

/-- Function to select students from a single class -/
def selectFromSingleClass (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of selection from a single class
  sorry

/-- Function to select students of a single gender -/
def selectSingleGender (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of selection of a single gender
  sorry

/-- Theorem stating that random selection from various grades is more representative -/
theorem random_selection_more_representative 
  (population : StudentPopulation) (sampleSize : ℕ) : 
  isRepresentative population (randomSelectFromGrades population sampleSize) ∧
  ¬isRepresentative population (selectFromSingleClass population sampleSize) ∧
  ¬isRepresentative population (selectSingleGender population sampleSize) :=
by
  sorry


end random_selection_more_representative_l288_28888


namespace problem_solution_l288_28829

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 5/x + 1/x^2 = 40)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 11 := by sorry

end problem_solution_l288_28829


namespace safe_mushrooms_l288_28813

/-- Given the following conditions about mushroom foraging:
  * The total number of mushrooms is 32
  * The number of poisonous mushrooms is twice the number of safe mushrooms
  * There are 5 uncertain mushrooms
  * The sum of safe, poisonous, and uncertain mushrooms equals the total
  Prove that the number of safe mushrooms is 9. -/
theorem safe_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) (uncertain : ℕ) 
  (h1 : total = 32)
  (h2 : poisonous = 2 * safe)
  (h3 : uncertain = 5)
  (h4 : safe + poisonous + uncertain = total) :
  safe = 9 := by sorry

end safe_mushrooms_l288_28813


namespace no_real_roots_implies_nonzero_sum_l288_28867

theorem no_real_roots_implies_nonzero_sum (a b c : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) → 
  a^3 + a * b + c ≠ 0 := by
sorry

end no_real_roots_implies_nonzero_sum_l288_28867


namespace transformed_curve_equation_l288_28875

/-- Given a curve and a scaling transformation, prove the equation of the transformed curve -/
theorem transformed_curve_equation (x y x' y' : ℝ) :
  (x^2 / 4 - y^2 = 1) →
  (x' = x / 2) →
  (y' = 2 * y) →
  (x'^2 - y'^2 / 4 = 1) :=
by sorry

end transformed_curve_equation_l288_28875


namespace candy_final_temperature_l288_28800

/-- Calculates the final temperature of a candy mixture given the initial conditions and rates. -/
theorem candy_final_temperature 
  (initial_temp : ℝ) 
  (max_temp : ℝ) 
  (heating_rate : ℝ) 
  (cooling_rate : ℝ) 
  (total_time : ℝ) 
  (h1 : initial_temp = 60)
  (h2 : max_temp = 240)
  (h3 : heating_rate = 5)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46) :
  let heating_time := (max_temp - initial_temp) / heating_rate
  let cooling_time := total_time - heating_time
  let temp_drop := cooling_rate * cooling_time
  max_temp - temp_drop = 170 := by
  sorry

end candy_final_temperature_l288_28800


namespace friend_team_assignment_l288_28844

theorem friend_team_assignment (n : ℕ) (k : ℕ) : 
  n = 6 → k = 3 → k ^ n = 729 := by sorry

end friend_team_assignment_l288_28844


namespace will_chocolate_pieces_l288_28852

theorem will_chocolate_pieces : 
  ∀ (total_boxes given_boxes pieces_per_box : ℕ),
  total_boxes = 7 →
  given_boxes = 3 →
  pieces_per_box = 4 →
  (total_boxes - given_boxes) * pieces_per_box = 16 :=
by
  sorry

end will_chocolate_pieces_l288_28852


namespace imaginary_power_sum_l288_28868

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^48 + i^96 + i^144 = 3 := by
  sorry

end imaginary_power_sum_l288_28868


namespace closest_fraction_l288_28869

def medals_won : ℚ := 24 / 150

def fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧
  ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
  f = 1/6 :=
sorry

end closest_fraction_l288_28869


namespace geometric_sequence_middle_term_l288_28859

theorem geometric_sequence_middle_term (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- a, b, c form a geometric sequence
  a = 5 + 2 * Real.sqrt 6 →
  c = 5 - 2 * Real.sqrt 6 →
  b = 1 ∨ b = -1 := by
sorry

end geometric_sequence_middle_term_l288_28859


namespace total_amount_paid_l288_28851

def grape_quantity : ℕ := 7
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 55

theorem total_amount_paid :
  grape_quantity * grape_rate + mango_quantity * mango_rate = 985 := by
  sorry

end total_amount_paid_l288_28851


namespace fraction_simplification_l288_28863

theorem fraction_simplification : (252 : ℚ) / 8820 * 21 = 3 / 5 := by sorry

end fraction_simplification_l288_28863


namespace village_population_l288_28842

theorem village_population (P : ℝ) : 
  (P > 0) →
  (0.8 * (0.9 * P) = 4500) →
  P = 6250 := by
sorry

end village_population_l288_28842
