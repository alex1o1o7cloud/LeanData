import Mathlib

namespace total_age_proof_l1769_176964

noncomputable def total_age : ℕ :=
  let susan := 15
  let arthur := susan + 2
  let bob := 11
  let tom := bob - 3
  let emily := susan / 2
  let david := (arthur + tom + emily) / 3
  susan + arthur + tom + bob + emily + david

theorem total_age_proof : total_age = 70 := by
  unfold total_age
  sorry

end total_age_proof_l1769_176964


namespace line_through_circle_center_l1769_176968

theorem line_through_circle_center {m : ℝ} :
  (∃ (x y : ℝ), x - 2*y + m = 0 ∧ x^2 + y^2 + 2*x - 4*y = 0) → m = 5 :=
by
  sorry

end line_through_circle_center_l1769_176968


namespace no_integer_solutions_l1769_176945

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
  sorry

end no_integer_solutions_l1769_176945


namespace find_m_for_parallel_lines_l1769_176960

open Real

theorem find_m_for_parallel_lines :
  ∀ (m : ℝ),
    (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0 → 3 * m = 4 * 6) →
    m = 8 :=
by
  intro m h
  have H : 3 * m = 4 * 6 := h 0 0 sorry sorry
  linarith

end find_m_for_parallel_lines_l1769_176960


namespace line_parabola_one_point_l1769_176939

theorem line_parabola_one_point (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 2 ∧ y^2 = 8 * x) 
  → (k = 0 ∨ k = 1) := 
by 
  sorry

end line_parabola_one_point_l1769_176939


namespace pipes_fill_time_l1769_176971

noncomputable def filling_time (P X Y Z : ℝ) : ℝ :=
  P / (X + Y + Z)

theorem pipes_fill_time (P : ℝ) (X Y Z : ℝ)
  (h1 : X + Y = P / 3) 
  (h2 : X + Z = P / 6) 
  (h3 : Y + Z = P / 4.5) :
  filling_time P X Y Z = 36 / 13 := by
  sorry

end pipes_fill_time_l1769_176971


namespace outfits_count_l1769_176957

-- Definitions of various clothing counts
def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 3
def numPants : ℕ := 8
def numBlueShoes : ℕ := 5
def numRedShoes : ℕ := 5
def numGreenHats : ℕ := 10
def numRedHats : ℕ := 6

-- Statement of the theorem based on the problem description
theorem outfits_count :
  (numRedShirts * numPants * numBlueShoes * numGreenHats) + 
  (numGreenShirts * numPants * (numBlueShoes + numRedShoes) * numRedHats) = 4240 := 
by
  -- No proof required, only the statement is needed
  sorry

end outfits_count_l1769_176957


namespace collinear_points_min_value_l1769_176905

open Real

/-- Let \(\overrightarrow{e_{1}}\) and \(\overrightarrow{e_{2}}\) be two non-collinear vectors in a plane,
    \(\overrightarrow{AB} = (a-1) \overrightarrow{e_{1}} + \overrightarrow{e_{2}}\),
    \(\overrightarrow{AC} = b \overrightarrow{e_{1}} - 2 \overrightarrow{e_{2}}\),
    with \(a > 0\) and \(b > 0\). 
    If points \(A\), \(B\), and \(C\) are collinear, then the minimum value of \(\frac{1}{a} + \frac{2}{b}\) is \(4\). -/
theorem collinear_points_min_value 
  (e1 e2 : ℝ) 
  (H_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0))
  (a b : ℝ) 
  (H_a_pos : a > 0) 
  (H_b_pos : b > 0)
  (H_collinear : ∃ x : ℝ, (a - 1) * e1 + e2 = x * (b * e1 - 2 * e2)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + (1/2) * b = 1 ∧ (∀ a b : ℝ, (1/a) + (2/b) ≥ 4) :=
sorry

end collinear_points_min_value_l1769_176905


namespace solution_set_I_range_of_a_l1769_176922

-- Define the function f(x) = |x + a| - |x + 1|
def f (x a : ℝ) : ℝ := abs (x + a) - abs (x + 1)

-- Part (I)
theorem solution_set_I (a : ℝ) : 
  (f a a > 1) ↔ (a < -2/3 ∨ a > 2) := by
  sorry

-- Part (II)
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 2 * a) ↔ (a ≥ 1/3) := by
  sorry

end solution_set_I_range_of_a_l1769_176922


namespace range_of_a_l1769_176937

open Real

theorem range_of_a {a : ℝ} :
  (∃ x : ℝ, sqrt (3 * x + 6) + sqrt (14 - x) > a) → a < 8 :=
by
  intro h
  sorry

end range_of_a_l1769_176937


namespace inscribed_circle_radius_l1769_176976

/-- Define a square SEAN with side length 2. -/
def square_side_length : ℝ := 2

/-- Define a quarter-circle of radius 1. -/
def quarter_circle_radius : ℝ := 1

/-- Hypothesis: The radius of the largest circle that can be inscribed in the remaining figure. -/
theorem inscribed_circle_radius :
  let S : ℝ := square_side_length
  let R : ℝ := quarter_circle_radius
  ∃ (r : ℝ), (r = 5 - 3 * Real.sqrt 2) := 
sorry

end inscribed_circle_radius_l1769_176976


namespace compute_value_l1769_176955

theorem compute_value : 12 - 4 * (5 - 10)^3 = 512 :=
by
  sorry

end compute_value_l1769_176955


namespace number_of_outfits_l1769_176909

theorem number_of_outfits (shirts pants : ℕ) (h_shirts : shirts = 5) (h_pants : pants = 3) 
    : shirts * pants = 15 := by
  sorry

end number_of_outfits_l1769_176909


namespace actual_average_speed_l1769_176943

variable {t : ℝ} (h₁ : t > 0) -- ensure that time is positive
variable {v : ℝ} 

theorem actual_average_speed (h₂ : v > 0)
  (h3 : v * t = (v + 12) * (3 / 4 * t)) : v = 36 :=
by
  sorry

end actual_average_speed_l1769_176943


namespace union_of_A_and_B_l1769_176989

namespace SetProof

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x > 0}
def expectedUnion : Set ℝ := {x | -2 ≤ x}

theorem union_of_A_and_B : (A ∪ B) = expectedUnion := by
  sorry

end SetProof

end union_of_A_and_B_l1769_176989


namespace prove_axisymmetric_char4_l1769_176936

-- Predicates representing whether a character is an axisymmetric figure
def is_axisymmetric (ch : Char) : Prop := sorry

-- Definitions for the conditions given in the problem
def char1 := '月'
def char2 := '右'
def char3 := '同'
def char4 := '干'

-- Statement that needs to be proven
theorem prove_axisymmetric_char4 (h1 : ¬ is_axisymmetric char1) 
                                  (h2 : ¬ is_axisymmetric char2) 
                                  (h3 : ¬ is_axisymmetric char3) : 
                                  is_axisymmetric char4 :=
sorry

end prove_axisymmetric_char4_l1769_176936


namespace right_angled_triangle_l1769_176995

-- Define the lengths of the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- State the theorem using the Pythagorean theorem
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l1769_176995


namespace range_of_a_l1769_176940

noncomputable def f (a x : ℝ) : ℝ := Real.sin x + 0.5 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici 0, f a x ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l1769_176940


namespace max_b_no_lattice_point_l1769_176993

theorem max_b_no_lattice_point (m : ℚ) (x : ℤ) (b : ℚ) :
  (y = mx + 3) → (0 < x ∧ x ≤ 50) → (2/5 < m ∧ m < b) → 
  ∀ (x : ℕ), y ≠ m * x + 3 →
  b = 11/51 :=
sorry

end max_b_no_lattice_point_l1769_176993


namespace minimum_point_translation_l1769_176923

theorem minimum_point_translation (x y : ℝ) : 
  (∀ (x : ℝ), y = 2 * |x| - 4) →
  x = 0 →
  y = -4 →
  (∀ (x y : ℝ), x_new = x + 3 ∧ y_new = y + 4) →
  (x_new, y_new) = (3, 0) :=
sorry

end minimum_point_translation_l1769_176923


namespace gasoline_storage_l1769_176911

noncomputable def total_distance : ℕ := 280 * 2

noncomputable def miles_per_segment : ℕ := 40

noncomputable def gasoline_consumption : ℕ := 8

noncomputable def total_segments : ℕ := total_distance / miles_per_segment

noncomputable def total_gasoline : ℕ := total_segments * gasoline_consumption

noncomputable def number_of_refills : ℕ := 14

theorem gasoline_storage (storage_capacity : ℕ) (h : number_of_refills * storage_capacity = total_gasoline) :
  storage_capacity = 8 :=
by
  sorry

end gasoline_storage_l1769_176911


namespace drums_filled_per_day_l1769_176958

-- Definition of given conditions
def pickers : ℕ := 266
def total_drums : ℕ := 90
def total_days : ℕ := 5

-- Statement to prove
theorem drums_filled_per_day : (total_drums / total_days) = 18 := by
  sorry

end drums_filled_per_day_l1769_176958


namespace solve_for_b_l1769_176966

theorem solve_for_b (b x : ℚ)
  (h₁ : 3 * x + 5 = 1)
  (h₂ : b * x + 6 = 0) :
  b = 9 / 2 :=
sorry   -- The proof is omitted as per instruction.

end solve_for_b_l1769_176966


namespace correct_expression_must_hold_l1769_176946

variable {f : ℝ → ℝ}

-- Conditions
axiom increasing_function : ∀ x y : ℝ, x < y → f x < f y
axiom positive_function : ∀ x : ℝ, f x > 0

-- Problem Statement
theorem correct_expression_must_hold : 3 * f (-2) > 2 * f (-3) := by
  sorry

end correct_expression_must_hold_l1769_176946


namespace bottles_per_day_l1769_176965

theorem bottles_per_day (b d : ℕ) (h1 : b = 8066) (h2 : d = 74) : b / d = 109 :=
by {
  sorry
}

end bottles_per_day_l1769_176965


namespace find_n_l1769_176916

theorem find_n
    (h : Real.arctan (1 / 2) + Real.arctan (1 / 3) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2) :
    n = 46 :=
sorry

end find_n_l1769_176916


namespace solve_digits_l1769_176974

variables (h t u : ℕ)

theorem solve_digits :
  (u = h + 6) →
  (u + h = 16) →
  (∀ (x y z : ℕ), 100 * h + 10 * t + u + 100 * u + 10 * t + h = 100 * x + 10 * y + z ∧ y = 9 ∧ z = 6) →
  (h = 5 ∧ t = 4 ∧ u = 11) :=
sorry

end solve_digits_l1769_176974


namespace michelle_drives_294_miles_l1769_176991

theorem michelle_drives_294_miles
  (total_distance : ℕ)
  (michelle_drives : ℕ)
  (katie_drives : ℕ)
  (tracy_drives : ℕ)
  (h1 : total_distance = 1000)
  (h2 : michelle_drives = 3 * katie_drives)
  (h3 : tracy_drives = 2 * michelle_drives + 20)
  (h4 : katie_drives + michelle_drives + tracy_drives = total_distance) :
  michelle_drives = 294 := by
  sorry

end michelle_drives_294_miles_l1769_176991


namespace find_g_g2_l1769_176956

def g (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 1

theorem find_g_g2 : g (g 2) = 2630 := by
  sorry

end find_g_g2_l1769_176956


namespace timothy_movies_count_l1769_176972

variable (T : ℕ)

def timothy_movies_previous_year (T : ℕ) :=
  let timothy_2010 := T + 7
  let theresa_2010 := 2 * (T + 7)
  let theresa_previous := T / 2
  T + timothy_2010 + theresa_2010 + theresa_previous = 129

theorem timothy_movies_count (T : ℕ) (h : timothy_movies_previous_year T) : T = 24 := 
by 
  sorry

end timothy_movies_count_l1769_176972


namespace frisbee_total_distance_correct_l1769_176910

-- Define the conditions
def bess_distance_per_throw : ℕ := 20 * 2 -- 20 meters out and 20 meters back = 40 meters
def bess_number_of_throws : ℕ := 4
def holly_distance_per_throw : ℕ := 8
def holly_number_of_throws : ℕ := 5

-- Calculate total distances
def bess_total_distance : ℕ := bess_distance_per_throw * bess_number_of_throws
def holly_total_distance : ℕ := holly_distance_per_throw * holly_number_of_throws
def total_distance : ℕ := bess_total_distance + holly_total_distance

-- The proof statement
theorem frisbee_total_distance_correct :
  total_distance = 200 :=
by
  -- proof goes here (we use sorry to skip the proof)
  sorry

end frisbee_total_distance_correct_l1769_176910


namespace repeated_root_cubic_l1769_176932

theorem repeated_root_cubic (p : ℝ) :
  (∃ x : ℝ, (3 * x^3 - (p + 1) * x^2 + 4 * x - 12 = 0) ∧ (9 * x^2 - 2 * (p + 1) * x + 4 = 0)) →
  (p = 5 ∨ p = -7) :=
by
  sorry

end repeated_root_cubic_l1769_176932


namespace parabola_standard_equation_l1769_176954

theorem parabola_standard_equation :
  ∃ m : ℝ, (∀ x y : ℝ, (x^2 = 2 * m * y ↔ (0, -6) ∈ ({p | 3 * p.1 - 4 * p.2 - 24 = 0}))) → 
  (x^2 = -24 * y) := 
by {
  sorry
}

end parabola_standard_equation_l1769_176954


namespace inequality_subtraction_l1769_176977

theorem inequality_subtraction (a b : ℝ) (h : a < b) : a - 5 < b - 5 := 
by {
  sorry
}

end inequality_subtraction_l1769_176977


namespace magical_stack_130_cards_l1769_176947

theorem magical_stack_130_cards (n : ℕ) (h1 : 2 * n > 0) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ 2 * (n - k + 1) = 131 ∨ 
                                   (n + 1) ≤ k ∧ k ≤ 2 * n ∧ 2 * k - 1 = 131) : 2 * n = 130 :=
by
  sorry

end magical_stack_130_cards_l1769_176947


namespace relationship_y1_y2_y3_l1769_176949

-- Define the quadratic function
def quadratic (x : ℝ) (k : ℝ) : ℝ :=
  -(x - 2) ^ 2 + k

-- Define the points A, B, and C
def A (y1 k : ℝ) := ∃ y1, quadratic (-1 / 2) k = y1
def B (y2 k : ℝ) := ∃ y2, quadratic (1) k = y2
def C (y3 k : ℝ) := ∃ y3, quadratic (4) k = y3

theorem relationship_y1_y2_y3 (y1 y2 y3 k: ℝ)
  (hA : A y1 k)
  (hB : B y2 k)
  (hC : C y3 k) :
  y1 < y3 ∧ y3 < y2 :=
  sorry

end relationship_y1_y2_y3_l1769_176949


namespace smallest_nat_divisible_by_225_l1769_176917

def has_digits_0_or_1 (n : ℕ) : Prop := 
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 1

def divisible_by_225 (n : ℕ) : Prop := 225 ∣ n

theorem smallest_nat_divisible_by_225 :
  ∃ (n : ℕ), has_digits_0_or_1 n ∧ divisible_by_225 n 
    ∧ ∀ (m : ℕ), has_digits_0_or_1 m ∧ divisible_by_225 m → n ≤ m 
    ∧ n = 11111111100 := 
  sorry

end smallest_nat_divisible_by_225_l1769_176917


namespace number_of_real_roots_l1769_176982

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 2010 * x + Real.log x / Real.log 2010
  else if x < 0 then - (2010 * (-x) + Real.log (-x) / Real.log 2010)
  else 0

theorem number_of_real_roots : 
  (∃ x1 x2 x3 : ℝ, 
    f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ∧ 
    ∀ x y z : ℝ, 
    (f x = 0 ∧ f y = 0 ∧ f z = 0 → 
    (x = y ∨ x = z ∨ y = z)) 
  :=
by
  sorry

end number_of_real_roots_l1769_176982


namespace pizza_share_l1769_176953

theorem pizza_share :
  forall (friends : ℕ) (leftover_pizza : ℚ), friends = 4 -> leftover_pizza = 5/6 -> (leftover_pizza / friends) = (5 / 24) :=
by
  intros friends leftover_pizza h_friends h_leftover_pizza
  sorry

end pizza_share_l1769_176953


namespace parabola_translation_l1769_176981

theorem parabola_translation :
  ∀ (x y : ℝ), (y = 2 * (x - 3) ^ 2) ↔ ∃ t : ℝ, t = x - 3 ∧ y = 2 * t ^ 2 :=
by sorry

end parabola_translation_l1769_176981


namespace multiplication_correct_l1769_176901

theorem multiplication_correct :
  375680169467 * 4565579427629 = 1715110767607750737263 :=
  by sorry

end multiplication_correct_l1769_176901


namespace find_f_neg3_l1769_176906

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x^2 - 2 * x else -(x^2 - 2 * -x)

theorem find_f_neg3 (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, 0 < x → f x = x^2 - 2 * x) : f (-3) = -3 :=
by
  sorry

end find_f_neg3_l1769_176906


namespace cone_altitude_ratio_l1769_176904

variable (r h : ℝ)
variable (radius_condition : r > 0)
variable (volume_condition : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3)

theorem cone_altitude_ratio {r h : ℝ}
  (radius_condition : r > 0) 
  (volume_condition : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := by
  sorry

end cone_altitude_ratio_l1769_176904


namespace percentage_decrease_in_savings_l1769_176903

theorem percentage_decrease_in_savings (I : ℝ) (F : ℝ) (IncPercent : ℝ) (decPercent : ℝ)
  (h1 : I = 125) (h2 : IncPercent = 0.25) (h3 : F = 125) :
  let P := (I * (1 + IncPercent))
  ∃ decPercent, decPercent = ((P - F) / P) * 100 ∧ decPercent = 20 := 
by
  sorry

end percentage_decrease_in_savings_l1769_176903


namespace factor_polynomial_int_l1769_176999

theorem factor_polynomial_int : 
  ∀ x : ℤ, 5 * (x + 3) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 = 
           (5 * x^2 + 81 * x + 315) * (x^2 + 16 * x + 213) := 
by
  intros
  norm_num
  sorry

end factor_polynomial_int_l1769_176999


namespace percent_covered_by_larger_triangles_l1769_176900

-- Define the number of small triangles in one large hexagon
def total_small_triangles := 16

-- Define the number of small triangles that are part of the larger triangles within one hexagon
def small_triangles_in_larger_triangles := 9

-- Calculate the fraction of the area of the hexagon covered by larger triangles
def fraction_covered_by_larger_triangles := 
  small_triangles_in_larger_triangles / total_small_triangles

-- Define the expected result as a fraction of the total area
def expected_fraction := 56 / 100

-- The proof problem in Lean 4 statement:
theorem percent_covered_by_larger_triangles
  (h1 : fraction_covered_by_larger_triangles = 9 / 16) :
  fraction_covered_by_larger_triangles = expected_fraction :=
  by
    sorry

end percent_covered_by_larger_triangles_l1769_176900


namespace number_of_teams_l1769_176907

theorem number_of_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : x = 8 :=
sorry

end number_of_teams_l1769_176907


namespace sequence_k_value_l1769_176912

theorem sequence_k_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ m n : ℕ, a (m + n) = a m * a n)
  (hk1 : ∀ k : ℕ, a (k + 1) = 1024) :
  ∃ k : ℕ, k = 9 :=
by {
  sorry
}

end sequence_k_value_l1769_176912


namespace original_distance_cycled_l1769_176921

theorem original_distance_cycled
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1/4) * (3/4 * t))
  (h3 : d = (x - 1/4) * (t + 3)) :
  d = 4.5 := 
sorry

end original_distance_cycled_l1769_176921


namespace solve_x_l1769_176933

variable (x : ℝ)

def vector_a := (2, 1)
def vector_b := (1, x)

def vectors_parallel : Prop :=
  let a_plus_b := (2 + 1, 1 + x)
  let a_minus_b := (2 - 1, 1 - x)
  a_plus_b.1 * a_minus_b.2 = a_plus_b.2 * a_minus_b.1

theorem solve_x (hx : vectors_parallel x) : x = 1/2 := by
  sorry

end solve_x_l1769_176933


namespace find_m_value_l1769_176990

-- Defining the hyperbola equation and the conditions
def hyperbola_eq (x y : ℝ) (m : ℝ) : Prop :=
  (x^2 / m) - (y^2 / 4) = 1

-- Definition of the focal distance
def focal_distance (c : ℝ) :=
  2 * c = 6

-- Definition of the relationship c^2 = a^2 + b^2 for hyperbolas
def hyperbola_focal_distance_eq (m : ℝ) (c b : ℝ) : Prop :=
  c^2 = m + b^2

-- Stating that the hyperbola has the given focal distance
def given_focal_distance : Prop :=
  focal_distance 3

-- Stating the given condition on b²
def given_b_squared : Prop :=
  4 = 4

-- The main theorem stating that m = 5 given the conditions.
theorem find_m_value (m : ℝ) : 
  (hyperbola_eq 1 1 m) → given_focal_distance → given_b_squared → m = 5 :=
by
  sorry

end find_m_value_l1769_176990


namespace students_before_intersection_equal_l1769_176920

-- Define the conditions
def students_after_stop : Nat := 58
def percentage : Real := 0.40
def percentage_students_entered : Real := 12

-- Define the target number of students before stopping
def students_before_stop (total_after : Nat) (entered : Nat) : Nat :=
  total_after - entered

-- State the proof problem
theorem students_before_intersection_equal :
  ∃ (x : Nat), 
  percentage * (x : Real) = percentage_students_entered ∧ 
  students_before_stop students_after_stop x = 28 :=
by
  sorry

end students_before_intersection_equal_l1769_176920


namespace rectangle_area_increase_l1769_176961

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let A_original := L * W
  let A_new := (2 * L) * (2 * W)
  (A_new - A_original) / A_original * 100 = 300 := by
  sorry

end rectangle_area_increase_l1769_176961


namespace greatest_root_of_g_l1769_176975

noncomputable def g (x : ℝ) : ℝ := 16 * x^4 - 20 * x^2 + 5

theorem greatest_root_of_g :
  ∃ r : ℝ, r = Real.sqrt 5 / 2 ∧ (forall x, g x ≤ g r) :=
sorry

end greatest_root_of_g_l1769_176975


namespace equation_transformation_l1769_176941

theorem equation_transformation (x y: ℝ) (h : 2 * x - 3 * y = 6) : 
  y = (2 * x - 6) / 3 := 
by
  sorry

end equation_transformation_l1769_176941


namespace non_planar_characterization_l1769_176984

-- Definitions:
structure Graph where
  V : ℕ
  E : ℕ
  F : ℕ

def is_planar (G : Graph) : Prop :=
  G.V - G.E + G.F = 2

def edge_inequality (G : Graph) : Prop :=
  G.E ≤ 3 * G.V - 6

def has_subgraph_K5_or_K33 (G : Graph) : Prop := sorry -- Placeholder for the complex subgraph check

-- Theorem statement:
theorem non_planar_characterization (G : Graph) (hV : G.V ≥ 3) :
  ¬ is_planar G ↔ ¬ edge_inequality G ∨ has_subgraph_K5_or_K33 G := sorry

end non_planar_characterization_l1769_176984


namespace mark_brought_in_4_times_more_cans_l1769_176951

theorem mark_brought_in_4_times_more_cans (M J R : ℕ) (h1 : M = 100) 
  (h2 : J = 2 * R + 5) (h3 : M + J + R = 135) : M / J = 4 :=
by sorry

end mark_brought_in_4_times_more_cans_l1769_176951


namespace middle_number_is_40_l1769_176969

theorem middle_number_is_40 (A B C : ℕ) (h1 : C = 56) (h2 : C - A = 32) (h3 : B / C = 5 / 7) : B = 40 :=
  sorry

end middle_number_is_40_l1769_176969


namespace modular_inverse_sum_correct_l1769_176930

theorem modular_inverse_sum_correct :
  (3 * 8 + 9 * 13) % 56 = 29 :=
by
  sorry

end modular_inverse_sum_correct_l1769_176930


namespace total_birds_correct_l1769_176928

-- Define the conditions
def number_of_trees : ℕ := 7
def blackbirds_per_tree : ℕ := 3
def magpies : ℕ := 13

-- Define the total number of blackbirds using the conditions
def total_blackbirds : ℕ := number_of_trees * blackbirds_per_tree

-- Define the total number of birds using the total number of blackbirds and the number of magpies
def total_birds : ℕ := total_blackbirds + magpies

-- The theorem statement that should be proven
theorem total_birds_correct : total_birds = 34 := 
sorry

end total_birds_correct_l1769_176928


namespace determine_r_l1769_176935

theorem determine_r (S : ℕ → ℤ) (r : ℤ) (n : ℕ) (h1 : 2 ≤ n) (h2 : ∀ k, S k = 2^k + r) : 
  r = -1 :=
sorry

end determine_r_l1769_176935


namespace isosceles_triangle_base_length_l1769_176998

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : b + 2 * a = 25) : b = 11 := by
  sorry

end isosceles_triangle_base_length_l1769_176998


namespace ms_lee_class_difference_l1769_176973

noncomputable def boys_and_girls_difference (ratio_b : ℕ) (ratio_g : ℕ) (total_students : ℕ) : ℕ :=
  let x := total_students / (ratio_b + ratio_g)
  let boys := ratio_b * x
  let girls := ratio_g * x
  girls - boys

theorem ms_lee_class_difference :
  boys_and_girls_difference 3 4 42 = 6 :=
by
  sorry

end ms_lee_class_difference_l1769_176973


namespace find_n_l1769_176919

theorem find_n (n : ℕ) (h : 2^n = 2 * 16^2 * 4^3) : n = 15 :=
by
  sorry

end find_n_l1769_176919


namespace time_for_first_three_workers_l1769_176927

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l1769_176927


namespace abs_sum_neq_zero_iff_or_neq_zero_l1769_176948

variable {x y : ℝ}

theorem abs_sum_neq_zero_iff_or_neq_zero (x y : ℝ) :
  (|x| + |y| ≠ 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end abs_sum_neq_zero_iff_or_neq_zero_l1769_176948


namespace choose_3_from_12_l1769_176967

theorem choose_3_from_12 : (Nat.choose 12 3) = 220 := by
  sorry

end choose_3_from_12_l1769_176967


namespace quadratic_inequality_solution_l1769_176914

theorem quadratic_inequality_solution (a b c : ℝ) (h_solution_set : ∀ x, ax^2 + bx + c < 0 ↔ x < -1 ∨ x > 3) :
  (a < 0) ∧
  (a + b + c > 0) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l1769_176914


namespace sum_of_roots_is_zero_l1769_176996

-- Definitions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem Statement
theorem sum_of_roots_is_zero (f : ℝ → ℝ) (h_even : is_even f) (h_intersects : ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) : 
  x1 + x2 + x3 + x4 = 0 :=
by 
  sorry -- Proof can be provided here

end sum_of_roots_is_zero_l1769_176996


namespace sally_eggs_l1769_176929

def dozen := 12
def total_eggs := 48

theorem sally_eggs : total_eggs / dozen = 4 := by
  -- Normally a proof would follow here, but we will use sorry to skip it
  sorry

end sally_eggs_l1769_176929


namespace equal_sums_of_squares_l1769_176985

-- Define the coordinates of a rectangle in a 3D space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define vertices of the rectangle.
def A : Point3D := ⟨0, 0, 0⟩
def B (a : ℝ) : Point3D := ⟨a, 0, 0⟩
def C (a b : ℝ) : Point3D := ⟨a, b, 0⟩
def D (b : ℝ) : Point3D := ⟨0, b, 0⟩

-- Distance squared between two points in 3D space.
def distance_squared (M N : Point3D) : ℝ :=
  (M.x - N.x)^2 + (M.y - N.y)^2 + (M.z - N.z)^2

-- Prove that the sums of the squares of the distances between an arbitrary point M and opposite vertices of the rectangle are equal.
theorem equal_sums_of_squares (a b : ℝ) (M : Point3D) :
  distance_squared M A + distance_squared M (C a b) = distance_squared M (B a) + distance_squared M (D b) :=
by
  sorry

end equal_sums_of_squares_l1769_176985


namespace proof_volume_l1769_176931

noncomputable def volume_set (a b c h r : ℝ) : ℝ := 
  let v_box := a * b * c
  let v_extensions := 2 * (a * b * h) + 2 * (a * c * h) + 2 * (b * c * h)
  let v_cylinder := Real.pi * r^2 * h
  let v_spheres := 8 * (1/6) * (Real.pi * r^3)
  v_box + v_extensions + v_cylinder + v_spheres

theorem proof_volume : 
  let a := 2; let b := 3; let c := 6
  let r := 2; let h := 3
  volume_set a b c h r = (540 + 48 * Real.pi) / 3 ∧ (540 + 48 + 3) = 591 :=
by 
  sorry

end proof_volume_l1769_176931


namespace triangle_arithmetic_angles_l1769_176902

/-- The angles in a triangle are in arithmetic progression and the side lengths are 6, 7, and y.
    The sum of the possible values of y equals a + sqrt b + sqrt c,
    where a, b, and c are positive integers. Prove that a + b + c = 68. -/
theorem triangle_arithmetic_angles (y : ℝ) (a b c : ℕ) (h1 : a = 3) (h2 : b = 22) (h3 : c = 43) :
    (∃ y1 y2 : ℝ, y1 = 3 + Real.sqrt 22 ∧ y2 = Real.sqrt 43 ∧ (y = y1 ∨ y = y2))
    → a + b + c = 68 :=
by
  sorry

end triangle_arithmetic_angles_l1769_176902


namespace friends_bought_boxes_l1769_176908

def rainbow_colors : ℕ := 7
def total_pencils : ℕ := 56
def pencils_per_box : ℕ := rainbow_colors

theorem friends_bought_boxes (emily_box : ℕ := 1) :
  (total_pencils / pencils_per_box) - emily_box = 7 := by
  sorry

end friends_bought_boxes_l1769_176908


namespace neg_p_true_l1769_176986

theorem neg_p_true :
  ∀ (x : ℝ), -2 < x ∧ x < 2 → |x - 1| + |x + 2| < 6 :=
by
  sorry

end neg_p_true_l1769_176986


namespace simplify_fraction_l1769_176952

theorem simplify_fraction (a b : ℕ) (h : a ≠ b) : (2 * a) / (2 * b) = a / b :=
sorry

end simplify_fraction_l1769_176952


namespace susan_homework_time_l1769_176913

theorem susan_homework_time :
  ∀ (start finish practice : ℕ),
  start = 119 ->
  practice = 240 ->
  finish = practice - 25 ->
  (start < finish) ->
  (finish - start) = 96 :=
by
  intros start finish practice h_start h_practice h_finish h_lt
  sorry

end susan_homework_time_l1769_176913


namespace check_line_properties_l1769_176963

-- Define the conditions
def line_equation (x y : ℝ) : Prop := y + 7 = -x - 3

-- Define the point and slope
def point_and_slope (x y : ℝ) (m : ℝ) : Prop := (x, y) = (-3, -7) ∧ m = -1

-- State the theorem to prove
theorem check_line_properties :
  ∃ x y m, line_equation x y ∧ point_and_slope x y m :=
sorry

end check_line_properties_l1769_176963


namespace find_b_for_square_binomial_l1769_176925

theorem find_b_for_square_binomial 
  (b : ℝ)
  (u t : ℝ)
  (h₁ : u^2 = 4)
  (h₂ : 2 * t * u = 8)
  (h₃ : b = t^2) : b = 4 := 
  sorry

end find_b_for_square_binomial_l1769_176925


namespace value_of_f1_l1769_176915

variable (f : ℝ → ℝ)
open Function

theorem value_of_f1
  (h : ∀ x y : ℝ, f (f (x - y)) = f x * f y - f x + f y - 2 * x * y + 2 * x - 2 * y) :
  f 1 = -1 :=
sorry

end value_of_f1_l1769_176915


namespace inequality_range_l1769_176938

theorem inequality_range (a : ℝ) : 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 ≤ y ∧ y ≤ 3 → x * y ≤ a * x^2 + 2 * y^2) ↔ a ≥ -1 := by 
  sorry

end inequality_range_l1769_176938


namespace student_score_in_first_subject_l1769_176918

theorem student_score_in_first_subject 
  (x : ℝ)  -- Percentage in the first subject
  (w : ℝ)  -- Constant weight (as all subjects have same weight)
  (S2_score : ℝ)  -- Score in the second subject
  (S3_score : ℝ)  -- Score in the third subject
  (target_avg : ℝ) -- Target average score
  (hS2 : S2_score = 70)  -- Second subject score is 70%
  (hS3 : S3_score = 80)  -- Third subject score is 80%
  (havg : (x + S2_score + S3_score) / 3 = target_avg) :  -- The desired average is equal to the target average
  target_avg = 70 → x = 60 :=   -- Target average score is 70%
by
  sorry

end student_score_in_first_subject_l1769_176918


namespace coprime_unique_residues_non_coprime_same_residue_l1769_176926

-- Part (a)

theorem coprime_unique_residues (m k : ℕ) (h : m.gcd k = 1) : 
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∀ (i : Fin m) (j : Fin k), 
      ∀ (i' : Fin m) (j' : Fin k), 
        (i, j) ≠ (i', j') → (a i * b j) % (m * k) ≠ (a i' * b j') % (m * k) := 
sorry

-- Part (b)

theorem non_coprime_same_residue (m k : ℕ) (h : m.gcd k > 1) : 
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∃ (i : Fin m) (j : Fin k) (i' : Fin m) (j' : Fin k), 
      (i, j) ≠ (i', j') ∧ (a i * b j) % (m * k) = (a i' * b j') % (m * k) := 
sorry

end coprime_unique_residues_non_coprime_same_residue_l1769_176926


namespace common_ratio_of_geometric_series_l1769_176942

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l1769_176942


namespace geometric_sequence_15th_term_l1769_176980

theorem geometric_sequence_15th_term :
  let a_1 := 27
  let r := (1 : ℚ) / 6
  let a_15 := a_1 * r ^ 14
  a_15 = 1 / 14155776 := by
  sorry

end geometric_sequence_15th_term_l1769_176980


namespace regular_nonagon_diagonals_correct_l1769_176983

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l1769_176983


namespace increase_80_by_50_percent_l1769_176988

theorem increase_80_by_50_percent :
  let initial_number : ℕ := 80
  let increase_percentage : ℝ := 0.5
  initial_number + (initial_number * increase_percentage) = 120 :=
by
  sorry

end increase_80_by_50_percent_l1769_176988


namespace anne_wandered_hours_l1769_176970

noncomputable def speed : ℝ := 2 -- miles per hour
noncomputable def distance : ℝ := 6 -- miles

theorem anne_wandered_hours (t : ℝ) (h : distance = speed * t) : t = 3 := by
  sorry

end anne_wandered_hours_l1769_176970


namespace leak_empties_cistern_in_12_hours_l1769_176934

theorem leak_empties_cistern_in_12_hours 
  (R : ℝ) (L : ℝ)
  (h1 : R = 1 / 4) 
  (h2 : R - L = 1 / 6) : 
  1 / L = 12 := 
by
  -- proof will go here
  sorry

end leak_empties_cistern_in_12_hours_l1769_176934


namespace min_sum_six_l1769_176997

theorem min_sum_six (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 :=
sorry

end min_sum_six_l1769_176997


namespace algebraic_expression_result_l1769_176979

theorem algebraic_expression_result (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 12 = -11 :=
by
  sorry

end algebraic_expression_result_l1769_176979


namespace geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l1769_176994

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n ≠ 0 ∧ (a (n + 1) = a n * (a (n + 1) / a n))

theorem geometric_sequence_implies_condition (a : ℕ → ℝ) :
  is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1) := sorry

theorem counterexample_condition_does_not_imply_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a := sorry

theorem geometric_sequence_sufficient_not_necessary (a : ℕ → ℝ) :
  (is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1)) ∧
  ((∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a) := by
  exact ⟨geometric_sequence_implies_condition a, counterexample_condition_does_not_imply_geometric_sequence a⟩

end geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l1769_176994


namespace parabola_y1_gt_y2_l1769_176992

variable {x1 x2 y1 y2 : ℝ}

theorem parabola_y1_gt_y2 
  (hx1 : -4 < x1 ∧ x1 < -2) 
  (hx2 : 0 < x2 ∧ x2 < 2) 
  (hy1 : y1 = x1^2) 
  (hy2 : y2 = x2^2) : 
  y1 > y2 :=
by 
  sorry

end parabola_y1_gt_y2_l1769_176992


namespace TimSpentTotal_l1769_176959

variable (LunchCost : ℝ) (TipPercentage : ℝ)

def TotalAmountSpent (LunchCost : ℝ) (TipPercentage : ℝ) : ℝ := 
  LunchCost + (LunchCost * TipPercentage)

theorem TimSpentTotal (h1 : LunchCost = 50.50) (h2 : TipPercentage = 0.20) :
  TotalAmountSpent LunchCost TipPercentage = 60.60 := by
  sorry

end TimSpentTotal_l1769_176959


namespace sphere_radius_equal_l1769_176987

theorem sphere_radius_equal (r : ℝ) 
  (hvol : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_equal_l1769_176987


namespace exponent_of_term_on_right_side_l1769_176924

theorem exponent_of_term_on_right_side
  (s m : ℕ) 
  (h1 : (2^16) * (25^s) = 5 * (10^m))
  (h2 : m = 16) : m = 16 := 
by
  sorry

end exponent_of_term_on_right_side_l1769_176924


namespace Nick_raising_money_l1769_176962

theorem Nick_raising_money :
  let chocolate_oranges := 20
  let oranges_price := 10
  let candy_bars := 160
  let bars_price := 5
  let total_amount := chocolate_oranges * oranges_price + candy_bars * bars_price
  total_amount = 1000 := 
by
  sorry

end Nick_raising_money_l1769_176962


namespace no_valid_middle_number_l1769_176978

theorem no_valid_middle_number
    (x : ℤ)
    (h1 : (x % 2 = 1))
    (h2 : 3 * x + 12 = x^2 + 20) :
    false :=
by
    sorry

end no_valid_middle_number_l1769_176978


namespace dryer_sheets_per_load_l1769_176944

theorem dryer_sheets_per_load (loads_per_week : ℕ) (cost_of_box : ℝ) (sheets_per_box : ℕ)
  (annual_savings : ℝ) (weeks_in_year : ℕ) (x : ℕ)
  (h1 : loads_per_week = 4)
  (h2 : cost_of_box = 5.50)
  (h3 : sheets_per_box = 104)
  (h4 : annual_savings = 11)
  (h5 : weeks_in_year = 52)
  (h6 : annual_savings = 2 * cost_of_box)
  (h7 : sheets_per_box * 2 = weeks_in_year * (loads_per_week * x)):
  x = 1 :=
by
  sorry

end dryer_sheets_per_load_l1769_176944


namespace sum_of_arithmetic_sequence_l1769_176950

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ):
  (S 4 = S 8 - S 4) →
  (S 4 = S 12 - S 8) →
  (S 4 = S 16 - S 12) →
  S 16 / S 4 = 10 :=
by
  intros h1 h2 h3
  sorry

end sum_of_arithmetic_sequence_l1769_176950
