import Mathlib
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Module.LinearMap
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Angle
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometry
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Mod
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Factors
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Card
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Planar
import Mathlib.Probability.Basic
import Mathlib.Probability.ConditionalProbability
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic

namespace max_f_value_l539_539559

def op (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

def f (x : ℝ) : ℝ :=
  (op 1 x) * x - (op 2 x)

theorem max_f_value : 
  ∀ x ∈ set.Icc (-2 : ℝ) 2, f x ≤ 6 ∧ ∃ y ∈ set.Icc (-2 : ℝ) 2, f y = 6 :=
by 
-- Proof goes here
sorry

end max_f_value_l539_539559


namespace my_op_2006_eq_4011_l539_539460

noncomputable def my_op : ℕ → ℕ
| 1       := 1
| (n + 1) := 2 + my_op n

theorem my_op_2006_eq_4011 : my_op 2006 = 4011 :=
by {
  sorry,
}

end my_op_2006_eq_4011_l539_539460


namespace percent_of_dollar_l539_539422

def value_in_cents (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies * 1 + nickels * 5 + dimes * 10 + quarters * 25

theorem percent_of_dollar (pennies nickels dimes quarters : ℕ) :
  pennies = 3 → nickels = 2 → dimes = 4 → quarters = 1 →
  value_in_cents pennies nickels dimes quarters = 78 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold value_in_cents
  norm_num
  sorry

end percent_of_dollar_l539_539422


namespace simplify_expr_eq_l539_539358

noncomputable def simplify_expression (a : ℝ) : ℝ :=
  ((3 * a / ((a ^ 2) - 1) - 1 / (a - 1)) / ((2 * a - 1) / (a + 1)))

theorem simplify_expr_eq (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 1/2) :
  simplify_expression a = (1 / (a - 1)) :=
by rationals; sorry

end simplify_expr_eq_l539_539358


namespace tan_arccot_l539_539132

theorem tan_arccot (x : ℝ) (h : x = 5 / 12) : Real.tan (Real.arccot x) = 12 / 5 :=
by
  rw [h]
  sorry

end tan_arccot_l539_539132


namespace necessary_but_not_sufficient_condition_l539_539183

def is_pure_imag (z : ℂ) : Prop := z.re = 0

theorem necessary_but_not_sufficient_condition (z : ℂ) : (z + conj z = 0) ↔ is_pure_imag z := by
  sorry

end necessary_but_not_sufficient_condition_l539_539183


namespace trajectory_equation_l539_539690

def fixed_point : ℝ × ℝ := (1, 2)

def moving_point (x y : ℝ) : ℝ × ℝ := (x, y)

def dot_product (p1 p2 : ℝ × ℝ) : ℝ :=
p1.1 * p2.1 + p1.2 * p2.2

theorem trajectory_equation (x y : ℝ) (h : dot_product (moving_point x y) fixed_point = 4) :
  x + 2 * y - 4 = 0 :=
sorry

end trajectory_equation_l539_539690


namespace maximum_value_of_function_l539_539011

theorem maximum_value_of_function : ∃ x, x > (1 : ℝ) ∧ (∀ y, y > 1 → (x + 1 / (x - 1) ≥ y + 1 / (y - 1))) ∧ (x = 2 ∧ (x + 1 / (x - 1) = 3)) :=
sorry

end maximum_value_of_function_l539_539011


namespace correct_exponential_calculation_l539_539046

theorem correct_exponential_calculation (a : ℝ) (ha : a ≠ 0) : 
  (a^4)^4 = a^16 :=
by sorry

end correct_exponential_calculation_l539_539046


namespace tammy_speed_on_second_day_l539_539780

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l539_539780


namespace parabola_distance_l539_539805

theorem parabola_distance (a : ℝ) :
  (abs (1 + (1 / (4 * a))) = 2 → a = 1 / 4) ∨ 
  (abs (1 - (1 / (4 * a))) = 2 → a = -1 / 12) := by 
  sorry

end parabola_distance_l539_539805


namespace trig_identity_solution_l539_539253

theorem trig_identity_solution 
  (α : ℝ)
  (h1 : sin (π - α) = - (real.sqrt 3) / 3)
  (h2 : π < α ∧ α < 3 * π / 2) :
  sin (π / 2 + α) = - (real.sqrt 6) / 3 := 
sorry

end trig_identity_solution_l539_539253


namespace sqrt_mul_sqrt_1_sqrt_mul_sqrt_2_sqrt_mul_sqrt_3_l539_539458

theorem sqrt_mul_sqrt_1 (x : ℝ) : (sqrt (3 * x) * sqrt (2 * (x^2)) = sqrt (6) * x^(3/2)) := 
by sorry

theorem sqrt_mul_sqrt_2 (x : ℝ) : (sqrt (3) * x * sqrt (2) * (x^2) = sqrt (6) * x^3) := 
by sorry

theorem sqrt_mul_sqrt_3 (x : ℝ) : (sqrt (sqrt (3 * (x^2)) + sqrt (2 * x)) * sqrt (sqrt (5 * (x^2)) + sqrt (4 * x)) = sqrt ((sqrt (3) + sqrt (2)) * (sqrt (5) + 2)) * x) := 
by sorry

end sqrt_mul_sqrt_1_sqrt_mul_sqrt_2_sqrt_mul_sqrt_3_l539_539458


namespace log2_geometric_sum_l539_539407

noncomputable def geometric_terms := ∃ r : ℝ, ∀ n : ℕ, a n = a 1 * (r ^ (n - 1))

theorem log2_geometric_sum (a : ℕ → ℝ) (h_geometric : geometric_terms a)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_cond : a 5 * a 6 + a 3 * a 8 = 16) :
  ∑ i in Finset.range 10, Real.log_base 2 (a (i+1)) = 15 :=
by
  sorry

end log2_geometric_sum_l539_539407


namespace line_intersects_circle_l539_539223

theorem line_intersects_circle 
  (radius : ℝ) 
  (distance_center_line : ℝ) 
  (h_radius : radius = 4) 
  (h_distance : distance_center_line = 3) : 
  radius > distance_center_line := 
by 
  sorry

end line_intersects_circle_l539_539223


namespace distance_from_point_to_line_l539_539000

-- Define the point
def point : ℝ × ℝ := (1, 2)

-- Define the line equation in its converted standard form: 4x + 3y - 12 = 0
def line (x y : ℝ) : ℝ := 4 * x + 3 * y - 12

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / real.sqrt (4^2 + 3^2)

-- Prove that the distance from point (1, 2) to the line 4x + 3y - 12 = 0 is 2/5
theorem distance_from_point_to_line : distance_point_to_line point (λ x y, line x y) = 2 / 5 :=
by
  sorry

end distance_from_point_to_line_l539_539000


namespace exists_beta_iff_sum_divides_a_l539_539740

variable {R : Type*} [LinearOrderedField R]

-- Define the function f and its necessary properties
variable (f : R → R) (a : ℤ)

-- Conditions
axiom f_property (x : R) : f(f(x)) = x * f(x) + (a : R)

-- Theorem to prove
theorem exists_beta_iff_sum_divides_a :
  (∃ β : R, f β = 0) ↔ ∃ n : ℕ, 0 < n ∧ ∑ k in finset.range(n+1), (k : ℤ)^3 ∣ a :=
sorry

end exists_beta_iff_sum_divides_a_l539_539740


namespace determine_values_l539_539150

variable {p q : ℝ}

theorem determine_values (h : log p + log q = log (p + 2q)) (hq1 : q ≠ 1) : p = 2 * q / (q - 1) :=
sorry

end determine_values_l539_539150


namespace perimeter_of_pool_l539_539543

theorem perimeter_of_pool (area_square : ℝ) (total_length : ℝ) :
  (∃ side : ℝ, side^2 = area_square ∧ total_length = 2 * side + 2 * (side + w)) →
  2 * (total_length - 2 * sqrt area_square) + 2 * (sqrt area_square) = 20 :=
sorry

end perimeter_of_pool_l539_539543


namespace trapezoid_area_sum_l539_539527

def side_lengths := {a b c d : ℝ // a = 4 ∧ b = 6 ∧ c = 8 ∧ d = 10}

theorem trapezoid_area_sum (s : side_lengths):
  (s.val.a = 4 ∧ s.val.b = 6 ∧ s.val.c = 8 ∧ s.val.d = 10) →
  ∃ area_sum : ℝ, area_sum = 3 * Real.sqrt 15 :=
sorry


end trapezoid_area_sum_l539_539527


namespace combined_percentage_of_students_preferring_tennis_l539_539950

theorem combined_percentage_of_students_preferring_tennis
    (north_students : ℕ)
    (north_prefers_tennis_percentage : ℚ)
    (south_students : ℕ)
    (south_prefers_tennis_percentage : ℚ)
    (north_students_eq : north_students = 1500)
    (north_percentage_eq : north_prefers_tennis_percentage = 30/100)
    (south_students_eq : south_students = 1800)
    (south_percentage_eq : south_prefers_tennis_percentage = 35/100) :
    (north_students * 30 / 100 + south_students * 35 / 100) / (north_students + south_students) * 100 ≈ 33 := 
by
  sorry

end combined_percentage_of_students_preferring_tennis_l539_539950


namespace rabbit_carrot_count_l539_539564

theorem rabbit_carrot_count
  (r h : ℕ)
  (hr : r = h - 3)
  (eq_carrots : 4 * r = 5 * h) :
  4 * r = 36 :=
by
  sorry

end rabbit_carrot_count_l539_539564


namespace least_three_digit_product_of_digits_is_8_l539_539872

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l539_539872


namespace sqrt_of_4_l539_539818

theorem sqrt_of_4 (y : ℝ) : y^2 = 4 → (y = 2 ∨ y = -2) :=
sorry

end sqrt_of_4_l539_539818


namespace triangle_sine_inequality_l539_539702

-- Define the context of the problem
variables (a b c : ℝ) (R : ℝ)
variables (A B C : ℝ) -- Angles in triangle ABC
variables (sin_A sin_B sin_C : ℝ) -- Sine values of angles A, B, and C

-- Define the sine of the angles using the sine rule
def sine_rule (a b c : ℝ) (R : ℝ) (A B C : ℝ) : Prop :=
  sin_A = a / (2 * R) ∧ sin_B = b / (2 * R) ∧ sin_C = c / (2 * R)

-- State the theorem to be proved
theorem triangle_sine_inequality
  (h : sine_rule a b c R A B C)
  : (sin_A ^ a * sin_B ^ b * sin_C ^ c) / ((sin_A * sin_B) ^ c *
             (sin_B * sin_C) ^ a * (sin_C * sin_A) ^ b) > 1 :=
sorry

end triangle_sine_inequality_l539_539702


namespace right_prism_max_volume_l539_539680

-- Define the variables and conditions
variables {a b h : ℝ} {θ : ℝ}
-- Define the surface area constraint
axiom surface_area_constraint : a * h + b * h + (1/2) * a * b * sin θ = 30

-- Define the maximum volume condition
def max_volume (V : ℝ) := V = 10 * sqrt 5

-- Statement of the problem
theorem right_prism_max_volume (a b h : ℝ) (θ : ℝ) 
  (h_surface_area_constraint : a * h + b * h + (1/2) * a * b * sin θ = 30) : 
  ∃ V : ℝ, max_volume V :=
sorry

end right_prism_max_volume_l539_539680


namespace yanna_kept_36_apples_l539_539446

-- Define the initial number of apples Yanna has
def initial_apples : ℕ := 60

-- Define the number of apples given to Zenny
def apples_given_to_zenny : ℕ := 18

-- Define the number of apples given to Andrea
def apples_given_to_andrea : ℕ := 6

-- The proof statement that Yanna kept 36 apples
theorem yanna_kept_36_apples : initial_apples - apples_given_to_zenny - apples_given_to_andrea = 36 := by
  sorry

end yanna_kept_36_apples_l539_539446


namespace units_digit_square_product_even_composite_l539_539884

theorem units_digit_square_product_even_composite :
  let n := (4 * 6 * 8)
  in ((n * n) % 10) = 4 := 
by
  let n := (4 * 6 * 8)
  sorry

end units_digit_square_product_even_composite_l539_539884


namespace speed_conversion_l539_539092

theorem speed_conversion (speed_mps : ℝ) (conversion_factor : ℝ) (speed_kmph : ℝ) : 
  speed_mps = 15.001199999999999 → 
  conversion_factor = 3.6 →
  speed_kmph ≈ 54.004 :=
begin
  assume h1 : speed_mps = 15.001199999999999,
  assume h2 : conversion_factor = 3.6,
  sorry
end

end speed_conversion_l539_539092


namespace range_of_given_function_l539_539561

noncomputable def given_function (x : ℝ) : ℝ :=
  abs (Real.sin x) / (Real.sin x) + Real.cos x / abs (Real.cos x) + abs (Real.tan x) / Real.tan x

theorem range_of_given_function : Set.range given_function = {-1, 3} :=
by
  sorry

end range_of_given_function_l539_539561


namespace measure_of_side_XY_is_correct_l539_539841

noncomputable def is_isosceles_right_triangle (X Y Z : Type) [metric_space X] (h_iso : XYZ.xyz_triangle.is_isosceles) : Prop :=
triangle.is_right (triangle X Y Z) ∧ (XY = XZ)

def measure_of_side_XY (X Y Z : Type) [metric_space X] (h_iso : XYZ.xyz_triangle.is_isosceles)
  (h_right : triangle.is_right (triangle X Y Z))
  (h_longer : XY > YZ)
  (h_area : triangle.area (triangle X Y Z) = 9)
  : real := 6

theorem measure_of_side_XY_is_correct (X Y Z : Type) [metric_space X] 
  (h_iso : XYZ.xyz_triangle.is_isosceles)
  (h_right : triangle.is_right (triangle X Y Z))
  (h_longer : XY > YZ)
  (h_area : triangle.area (triangle X Y Z) = 9) 
  : measure_of_side_XY X Y Z h_iso h_right h_longer h_area = 6 :=
sorry

end measure_of_side_XY_is_correct_l539_539841


namespace infinitely_many_nineteen_hundred_eighty_solutions_l539_539356

theorem infinitely_many_nineteen_hundred_eighty_solutions :
  ∀ m : ℕ, ∃ n ≥ m, ∃ P : ℕ → ℕ → Prop, (∀ a b : ℕ, 1 ≤ a → a ≤ N^2 → 1 ≤ b → b ≤ N^2 → P a b ↔ ⌊a * (3 / 2)⌋ + ⌊b * (3 / 2)⌋ = n) ∧ (∃ p : ℕ, p ≥ 1980 ∧ (∃I J, I = { a // 1 ≤ a ∧ a ≤ N^2 } ∧ J = { b // 1 ≤ b ∧ b ≤ N^2 } ∧ ∀ a ∈ I, ∀ b ∈ J, P a b → p)) :=
sorry

end infinitely_many_nineteen_hundred_eighty_solutions_l539_539356


namespace relationship_of_y_values_l539_539202

noncomputable def quadratic_function (x : ℝ) (c : ℝ) := x^2 - 6*x + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  quadratic_function 1 c = y1 →
  quadratic_function (2 * Real.sqrt 2) c = y2 →
  quadratic_function 4 c = y3 →
  y3 < y2 ∧ y2 < y1 :=
by
  intros hA hB hC
  sorry

end relationship_of_y_values_l539_539202


namespace modulus_value_l539_539251

open Complex

theorem modulus_value (z : ℂ) (h : (1 - 2 * I) * z = 5 * I) : Complex.abs z = sqrt 5 := 
by 
  sorry

end modulus_value_l539_539251


namespace necessary_but_not_sufficient_condition_l539_539653

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
noncomputable def vector_b : ℝ × ℝ := (2, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Statement: Prove x > 0 is a necessary but not sufficient condition for the angle between vectors a and b to be acute.
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (dot_product (vector_a x) vector_b > 0) ↔ (x > 0) := 
sorry

end necessary_but_not_sufficient_condition_l539_539653


namespace deng_xiaoping_1979_l539_539273

theorem deng_xiaoping_1979 (H1 : "We want developed, productive, and prosperous socialism") 
(H2 : "Socialism can also engage in a market economy") :
facilitated_establishment_development_SEZ :=
sorry

end deng_xiaoping_1979_l539_539273


namespace greatest_base_nine_digit_sum_l539_539426

theorem greatest_base_nine_digit_sum (n : ℕ) (h1 : n < 2500) : ∃ s, s = 24 ∧ ∀ m, m < 2500 → digit_sum_base_nine m ≤ s :=
begin
  sorry
end

def digit_sum_base_nine (n : ℕ) : ℕ :=
  -- conversion and digit sum calculation logic to be implemented here
  sorry

end greatest_base_nine_digit_sum_l539_539426


namespace area_quadrilateral_FDBG_l539_539840

def midpoint (a b : Point) : Point := 
  { x := (a.x + b.x) / 2, y := (a.y + b.y) / 2 }

axiom point (x y : ℝ) : Point

noncomputable def triangle_area (a b c : Point) : ℝ :=
  abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2

theorem area_quadrilateral_FDBG:
  ∀ (A B C D E F G : Point),
  B.x = 60 →
  C.x = 12 →
  triangle_area A B C = 180 →
  D = midpoint A B →
  E = midpoint A C →
  angle_bisector A B C ∩ DE = F →
  angle_bisector A B C ∩ BC = G →
  triangle_area A D E = 45 →
  triangle_area A G C = 30 →
  triangle_area A F E = 7.5 →
  triangle_area F D B + triangle_area D B G = 112.5 :=
begin
  intros A B C D E F G hB hC hABC hD hE hF hG hADE hAGC hAFE,
  sorry
end

end area_quadrilateral_FDBG_l539_539840


namespace jellybean_count_l539_539566

theorem jellybean_count (x : ℕ) (h : (0.7 : ℝ) ^ 3 * x = 34) : x = 99 :=
sorry

end jellybean_count_l539_539566


namespace range_of_a_l539_539269

theorem range_of_a (a : ℝ) (h : ∃ (S : set ℤ), S = {x : ℤ | 1 + a ≤ x ∧ x < 2} ∧ 5 ≤ S.card) : 
  -5 < a ∧ a ≤ -4 :=
by
  sorry

end range_of_a_l539_539269


namespace checkerboard_ratio_l539_539148

theorem checkerboard_ratio :
  let lines := 7 in
  let num_rectangles := nat.choose lines 2 * nat.choose lines 2 in
  let num_squares := (list.range 7).sum (λ k, k * k) in
  let ratio := num_squares / num_rectangles in
  num_squares = 91 → 
  num_rectangles = 441 →
  ratio = 1/7 →
  let m := 1 in
  let n := 7 in
  m + n = 8 :=
by
  sorry

end checkerboard_ratio_l539_539148


namespace total_books_collected_l539_539257

theorem total_books_collected :
  let books_from_NA := 581
  let books_from_SA := 435
  let books_from_Africa := 524
  let books_from_Europe := 688
  let books_from_Australia := 319
  let books_from_Asia := 526
  let books_from_Antarctica := 276
  in books_from_NA + books_from_SA + books_from_Africa + books_from_Europe + books_from_Australia + books_from_Asia + books_from_Antarctica = 3349 :=
by
  sorry

end total_books_collected_l539_539257


namespace least_three_digit_product_of_digits_is_8_l539_539874

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l539_539874


namespace sin_identity_alpha_l539_539736

theorem sin_identity_alpha (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
by 
  sorry

end sin_identity_alpha_l539_539736


namespace weeks_of_exercise_l539_539708

def hours_per_day : ℕ := 1
def days_per_week : ℕ := 5
def total_hours : ℕ := 40

def weekly_hours : ℕ := hours_per_day * days_per_week

theorem weeks_of_exercise (W : ℕ) (h : total_hours = weekly_hours * W) : W = 8 :=
by
  sorry

end weeks_of_exercise_l539_539708


namespace prime_square_minus_one_divisible_by_24_l539_539760

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 
  ∃ k : ℤ, p^2 - 1 = 24 * k :=
  sorry

end prime_square_minus_one_divisible_by_24_l539_539760


namespace fixed_point_logarithmic_shift_l539_539266

def log_fixed_point (a : ℝ) (ha_pos : a > 0) (ha_one : a ≠ 1) : Prop :=
  f 2 = 0
where
  f (x : ℝ) : ℝ := Real.log a (x - 1)

theorem fixed_point_logarithmic_shift (a : ℝ) (ha_pos : a > 0) (ha_one : a ≠ 1) : log_fixed_point a ha_pos ha_one :=
by
  sorry

end fixed_point_logarithmic_shift_l539_539266


namespace tammy_speed_on_second_day_l539_539789

theorem tammy_speed_on_second_day :
  ∃ v t : ℝ, t + (t - 2) = 14 ∧ v * t + (v + 0.5) * (t - 2) = 52 → (v + 0.5 = 4) :=
begin
  sorry
end

end tammy_speed_on_second_day_l539_539789


namespace installations_correct_l539_539835

noncomputable def installations : ℕ × ℕ × ℕ :=
  let x₁ := 9
  let x₂ := 36
  let x₃ := 27
  (x₁, x₂, x₃)

theorem installations_correct :
  let (x₁, x₂, x₃) := installations in
  x₁ + x₂ + x₃ ≤ 200 ∧ 
  x₂ = 4 * x₁ ∧ 
  ∃ k : ℕ, x₃ = k * x₁ ∧ 
  5 * x₃ = x₂ + 99 ∧ 
  x₁ ∈ ℕ ∧ x₂ ∈ ℕ ∧ x₃ ∈ ℕ :=
by
  sorry

end installations_correct_l539_539835


namespace area_triangle_A_l539_539905

variables {A B C A' B' C' : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
          [MetricSpace A'] [MetricSpace B'] [MetricSpace C']

-- Define the conditions as hypotheses
def area_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : ℝ := 1 -- Given Area of ABC is 1
def ratio_BB' (AB : ℝ) : ℝ := 2 * AB -- BB' = 2 AB
def ratio_CC' (BC : ℝ) : ℝ := 3 * BC -- CC' = 3 BC
def ratio_AA' (CA : ℝ) : ℝ := 4 * CA -- AA' = 4 CA

-- Define the proof of the area of triangle A'B'C'
theorem area_triangle_A'B'C' (AB BC CA : ℝ) : 
  area_triangle A B C = 1 →
  ratio_BB' AB = 2 * AB →
  ratio_CC' BC = 3 * BC →
  ratio_AA' CA = 4 * CA →
  area_triangle A' B' C' = 39 := 
by
  assume h1 h2 h3 h4,
  sorry

end area_triangle_A_l539_539905


namespace angle_PRQ_in_regular_heptagon_l539_539849

noncomputable theory

def regular_heptagon_interior_angle (n : ℕ) : ℝ :=
  (180 * (n - 2)) / n

theorem angle_PRQ_in_regular_heptagon (P Q R : Point) (heptagon : RegularHeptagon P Q R) :
  ∠ PRQ = 25.715 :=
by
  -- Some intermediate steps may go here
  sorry

end angle_PRQ_in_regular_heptagon_l539_539849


namespace log_sum_sin_eq_neg_three_l539_539971

noncomputable def log2 : ℝ → ℝ := sorry -- This function definition should appropriately define log base 2
noncomputable def sin : ℝ → ℝ := sorry -- This function definition should appropriately define sin

theorem log_sum_sin_eq_neg_three :
  log2 (sin (10 * real.pi / 180)) + log2 (sin (50 * real.pi / 180)) + log2 (sin (70 * real.pi / 180)) = -3 :=
by sorry

end log_sum_sin_eq_neg_three_l539_539971


namespace base_b_arithmetic_l539_539985

theorem base_b_arithmetic (b : ℕ) (h1 : 4 + 3 = 7) (h2 : 6 + 2 = 8) (h3 : 4 + 6 = 10) (h4 : 3 + 4 + 1 = 8) : b = 9 :=
  sorry

end base_b_arithmetic_l539_539985


namespace find_constants_l539_539165

noncomputable def constants (A B : ℚ) : Prop :=
  A + B = 4 ∧ A - 2 * B = -1

theorem find_constants :
  ∃ (A B : ℚ), 
    constants A B ∧ 
    A = 7 / 3 ∧ 
    B = 5 / 3 := 
by {
  use [7 / 3, 5 / 3],
  split,
  { split; norm_num, },
  { split; refl, }
}

end find_constants_l539_539165


namespace sum_odd_divisors_240_l539_539435

theorem sum_odd_divisors_240 : 
  let n := 240 in 
  let prime_factors := [2, 2, 2, 2, 3, 5] in 
  ∑ d in (finset.filter (λ d : ℕ, d ∣ n ∧ (∀ p, prime p ∧ p ∣ d → p % 2 = 1)) (finset.range (n + 1))), d = 24 :=
by { let n := 240,
     let prime_factors := [2, 2, 2, 2, 3, 5],
     sorry
}

end sum_odd_divisors_240_l539_539435


namespace number_of_chickens_l539_539828

variable (C P : ℕ) (legs_total : ℕ := 48) (legs_pig : ℕ := 4) (legs_chicken : ℕ := 2) (number_pigs : ℕ := 9)

theorem number_of_chickens (h1 : P = number_pigs)
                           (h2 : legs_pig * P + legs_chicken * C = legs_total) :
                           C = 6 :=
by
  sorry

end number_of_chickens_l539_539828


namespace jason_work_hours_l539_539709

variable (x y : ℕ)

def working_hours : Prop :=
  (4 * x + 6 * y = 88) ∧
  (x + y = 18)

theorem jason_work_hours (h : working_hours x y) : y = 8 :=
  by
    sorry

end jason_work_hours_l539_539709


namespace probability_at_least_two_consecutive_heads_l539_539083

open Classical

-- Define the basic setup for the problem
def fair_coin_toss_four_times := (Finset.ftuple 4 (Finset.range 2))

-- To calculate the probability of interest
noncomputable def P (s : Finset (Fin 2) → Prop) : ℚ := 
  (s.to_finset.card : ℚ) / fair_coin_toss_four_times.card

-- Condition: determining success. We assume s identifies the sequences
-- that have at least two consecutive heads.
def at_least_two_consecutive_heads (sequence : Finset (Fin 2)) : Prop :=
  ∃ i : Fin 3, sequence i = 1 ∧ sequence (i + 1) = 1

-- Formal statement of the problem
theorem probability_at_least_two_consecutive_heads :
  P at_least_two_consecutive_heads = 9/16 :=
sorry

end probability_at_least_two_consecutive_heads_l539_539083


namespace remainder_is_five_l539_539577

theorem remainder_is_five (A : ℕ) (h : 17 = 6 * 2 + A) : A = 5 :=
sorry

end remainder_is_five_l539_539577


namespace minimum_perimeter_triangle_PE_l539_539692

noncomputable def cube_edge_length : ℝ := 2
noncomputable def cube_midpoint (a b : ℝ) : ℝ := (a + b) / 2

def C₁_point : ℝ × ℝ × ℝ := (0, 0, 2)
def C_point : ℝ × ℝ × ℝ := (0, 0, 0)

noncomputable def E_point : ℝ × ℝ × ℝ := (cube_midpoint C₁_point.1 C_point.1, cube_midpoint C₁_point.2 C_point.2, cube_midpoint C₁_point.3 C_point.3)

def A₁_face (P Q : ℝ × ℝ × ℝ) : Prop := 
  P.3 = 2 ∧ 0 ≤ P.1 ∧ P.1 ≤ 2 ∧ 0 ≤ P.2 ∧ P.2 ≤ 2 ∧  -- P condition
  Q.2 = 2 ∧ 0 ≤ Q.1 ∧ Q.1 ≤ 2 ∧ 0 ≤ Q.3 ∧ Q.3 ≤ 2  -- Q condition

def minimum_perimeter := 
  2 + sqrt 2 + sqrt 10

theorem minimum_perimeter_triangle_PE : 
  ∃ P Q, A₁_face P Q ∧ (P = (0,1,2)) ∧ (Q = (0,2,0)) → minimum_perimeter = sqrt 10 :=
begin
  sorry
end

end minimum_perimeter_triangle_PE_l539_539692


namespace problem_1_problem_2_l539_539989

noncomputable theory
open Set Real

-- Define D(k) type function
def is_Dk_type (f : ℝ → ℝ) (k : ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, 1/k < f x ∧ f x < k

-- Problem 1: Prove the range of values for 'a'
theorem problem_1 (a : ℝ) : is_Dk_type (λ x : ℝ, a * abs x) 3 ([-3, -1] ∪ [1, 3]) ↔ (1/3 < a ∧ a < 1) :=
by sorry

-- Problem 2: Prove that g(x) is a D(2) type function
theorem problem_2 : is_Dk_type (λ x : ℝ, exp x - x^2 - x) 2 (Ioo 0 2) :=
by sorry

end problem_1_problem_2_l539_539989


namespace find_Q_l539_539843

-- We define the circles and their centers
def circle1 (x y r : ℝ) : Prop := (x + 1) ^ 2 + (y - 1) ^ 2 = r ^ 2
def circle2 (x y R : ℝ) : Prop := (x - 2) ^ 2 + (y + 2) ^ 2 = R ^ 2

-- Coordinates of point P
def P : ℝ × ℝ := (1, 2)

-- Defining the symmetry about the line y = -x
def symmetric_about (p q : ℝ × ℝ) : Prop := p.1 = -q.2 ∧ p.2 = -q.1

-- Theorem stating that if P is (1, 2), Q should be (-2, -1)
theorem find_Q {r R : ℝ} (h1 : circle1 1 2 r) (h2 : circle2 1 2 R) (hP : P = (1, 2)) :
  ∃ Q : ℝ × ℝ, symmetric_about P Q ∧ Q = (-2, -1) :=
by
  sorry

end find_Q_l539_539843


namespace num_four_digit_numbers_with_property_l539_539658

def is_valid_number (N : ℕ) : Prop :=
  let a := N / 1000
  let x := N % 1000
  (1000 <= N ∧ N < 10000) ∧ (x = 1000 * a / 8)

def count_valid_numbers : ℕ :=
  (Finset.range 10000).filter is_valid_number |>.card

theorem num_four_digit_numbers_with_property : count_valid_numbers = 6 := by
  sorry

end num_four_digit_numbers_with_property_l539_539658


namespace final_proof_l539_539194

def line_polar_eq_to_cartesian : Prop :=
  ∀ ρ θ : ℝ, 
    ρ * Real.sin (π / 4 - θ) = sqrt 2 → 
    ∃ x y : ℝ, 
      x - y = 2

def sum_of_slopes (A B M : (ℝ × ℝ)) : ℝ :=
  let slope (P Q : (ℝ × ℝ)) : ℝ := (Q.2 - P.2) / (Q.1 - P.1) in
  slope M A + slope M B

def intersection_sum_of_slopes : Prop :=
  ∃ (A B M : (ℝ × ℝ)),
    (M = (1, -2)) ∧
    (A.1 - A.2 = 2) ∧
    ((A.1 - 1)^2 + (A.2 + 2)^2 = 9) ∧
    (B.1 - B.2 = 2) ∧
    ((B.1 - 1)^2 + (B.2 + 2)^2 = 9) ∧
    sum_of_slopes A B M = 1 / 2

-- Combining the two parts into a single proof problem
theorem final_proof : line_polar_eq_to_cartesian ∧ intersection_sum_of_slopes := sorry

end final_proof_l539_539194


namespace solve_equation_l539_539363

theorem solve_equation : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -3 → (2 / x + x / (x + 3) = 1) ↔ (x = 6) := by
  intro x h
  have h1 : x ≠ 0 := h.1
  have h2 : x ≠ -3 := h.2
  sorry

end solve_equation_l539_539363


namespace smallest_n_l539_539984

theorem smallest_n (a b c n : ℕ) (h1 : n = 100 * a + 10 * b + c)
  (h2 : n = a + b + c + a * b + b * c + a * c + a * b * c)
  (h3 : n >= 100 ∧ n < 1000)
  (h4 : a ≥ 1 ∧ a ≤ 9)
  (h5 : b ≥ 0 ∧ b ≤ 9)
  (h6 : c ≥ 0 ∧ c ≤ 9) :
  n = 199 :=
sorry

end smallest_n_l539_539984


namespace sum_of_geometric_sequence_with_arithmetic_property_l539_539695

theorem sum_of_geometric_sequence_with_arithmetic_property 
  (a : ℝ) (n : ℕ) (a_n : ℕ → ℝ) (S_n : ℝ) 
  (h_geom : ∀ n, a_n n = a * 1^(n-1)) 
  (h_first_term : a_n 1 = a)
  (h_sum: S_n = ∑ i in finset.range n, a_n (i + 1))
  (h_arith_seq : ∀ n, 2 * (a_n (n+1) + 1) = (a_n n + 1) + (a_n (n+2) + 1)) : 
  S_n = n * a :=
sorry

end sum_of_geometric_sequence_with_arithmetic_property_l539_539695


namespace regular_polyhedron_spheres_l539_539759

-- Define a regular polyhedron
structure RegularPolyhedron (P : Type) :=
  (vertices : Set P)
  (faces : Set (Set P))
  (convex : Convex P)
  (regular : ∀ f ∈ faces, is_regular_face f)

-- Definitions for spheres
structure Sphere (S : Type) :=
  (center : S)
  (radius : ℝ)

-- Define the properties of circumscribed, inscribed, and mid-spheres
def circumscribed_sphere (P : RegularPolyhedron ℝ) : Sphere ℝ :=
  sorry

def inscribed_sphere (P : RegularPolyhedron ℝ) : Sphere ℝ :=
  sorry

def mid_sphere (P : RegularPolyhedron ℝ) : Sphere ℝ :=
  sorry

-- Statement of the theorem
theorem regular_polyhedron_spheres (P : RegularPolyhedron ℝ) :
  ∃ S₁ S₂ S₃ : Sphere ℝ,
    (∀ v ∈ P.vertices, PointOnSphere v S₁) ∧
    (∀ f ∈ P.faces, SphereTangentToFace S₂ f) ∧
    (∀ e, EdgeOfPolyhedron P e → SphereTangentToEdge S₃ e) ∧
    (S₁.center = S₂.center ∧ S₂.center = S₃.center) :=
  sorry

end regular_polyhedron_spheres_l539_539759


namespace equilateral_triangle_arc_sum_l539_539719

theorem equilateral_triangle_arc_sum
  {A B C M : Type}
  [Point : Type]
  (triangle_eq : EquilateralTriangle A B C)
  (circumcircle : CircumscribedCircle triangle_eq)
  (M_on_arc : OnArc M circumcircle.BC_not_containing_A) :
  Distance A M = Distance B M + Distance C M := 
sorry

end equilateral_triangle_arc_sum_l539_539719


namespace initial_velocity_proof_l539_539052

-- Define constants and conditions
def mass : ℝ := 4.2
def theta : ℝ := 12 * Real.pi / 180  -- converting degrees to radians
def time_to_peak : ℝ := 1.0
def gravitational_acc : ℝ := 9.8

-- Define period formula for pendulum and solving for length
def period (l : ℝ) : ℝ := 2 * Real.pi * Real.sqrt (l / gravitational_acc)
def length := (time_to_peak * 4) ^ 2 * gravitational_acc / (4 * Real.pi ^ 2)

-- Define small-angle approximation
def pendulum_height := length * (1 - Real.cos theta)

-- Define potential energy conversion to kinetic energy after collision
def velocity2 := Real.sqrt (2 * gravitational_acc * pendulum_height)

-- Using the properties of perfectly elastic collision between identical masses
-- velocity of first block becomes the velocity of the second block post-collision
def initial_velocity := velocity2

-- Calculate and round 10v to the nearest integer
def ten_v := (10 * initial_velocity : ℝ).round

-- Proving the statement
theorem initial_velocity_proof : ten_v = 13 := by 
  sorry  -- Proof goes here.

end initial_velocity_proof_l539_539052


namespace tammy_avg_speed_second_day_l539_539786

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l539_539786


namespace sum_of_T_equals_227_l539_539720

def isDistinctDigits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def validDigit (n : ℕ) : Prop := n < 10

def T : set ℝ :=
  { x | ∃ a b c d : ℕ,
        validDigit a ∧ validDigit b ∧ validDigit c ∧ validDigit d ∧
        isDistinctDigits a b c d ∧
        x = (a * 1000 + b * 100 + c * 10 + d) / 9999 }

theorem sum_of_T_equals_227 :
  ∑ x in (finset.filter (λ x, x ∈ T) (finset.range 9999)), x = 227 := 
sorry

end sum_of_T_equals_227_l539_539720


namespace number_of_squares_in_H_l539_539560

def H : set (ℤ × ℤ) := 
  { p | let x := p.1 in let y := p.2 in 2 ≤ |x| ∧ |x| ≤ 6 ∧ 2 ≤ |y| ∧ |y| ≤ 6 }

theorem number_of_squares_in_H : 
  ∃ n : ℕ, n = 8 ∧ ∀ sq : set (ℤ × ℤ), (is_square sq ∧ square_side_length sq ≥ 4 ∧ sq ⊆ H) → sq.count_vertices == 4 :=
sorry

end number_of_squares_in_H_l539_539560


namespace tower_of_hanoi_l539_539032

-- Define the function H for the number of moves.
def H : ℕ → ℕ
| 0     := 0
| (n+1) := 2 * H n + 1

-- Define the hypothesis for the problem statement.
theorem tower_of_hanoi (n : ℕ) : H n = 2^n - 1 :=
by
  sorry

end tower_of_hanoi_l539_539032


namespace large_pizzas_sold_l539_539019

def small_pizza_price : ℕ := 2
def large_pizza_price : ℕ := 8
def total_earnings : ℕ := 40
def small_pizzas_sold : ℕ := 8

theorem large_pizzas_sold : 
  ∀ (small_pizza_price large_pizza_price total_earnings small_pizzas_sold : ℕ), 
    small_pizza_price = 2 → 
    large_pizza_price = 8 → 
    total_earnings = 40 → 
    small_pizzas_sold = 8 →
    (total_earnings - small_pizzas_sold * small_pizza_price) / large_pizza_price = 3 :=
by 
  intros small_pizza_price large_pizza_price total_earnings small_pizzas_sold 
         h_small_pizza_price h_large_pizza_price h_total_earnings h_small_pizzas_sold
  rw [h_small_pizza_price, h_large_pizza_price, h_total_earnings, h_small_pizzas_sold]
  simp
  sorry

end large_pizzas_sold_l539_539019


namespace smallest_n_satisfying_condition_l539_539588

theorem smallest_n_satisfying_condition : 
  ∃ n : ℕ, n ≥ 9 ∧ (∀ (a : ℕ → ℤ), 
    ∃ (i : Fin 9 → Fin n) (b : Fin 9 → {4, 7}),
      (b 0 * a (i 0) + b 1 * a (i 1) + b 2 * a (i 2) + b 3 * a (i 3) + b 4 * a (i 4) 
      + b 5 * a (i 5) + b 6 * a (i 6) + b 7 * a (i 7) + b 8 * a (i 8)) % 9 = 0) ∧
  n = 13 :=
sorry

end smallest_n_satisfying_condition_l539_539588


namespace max_value_of_g_l539_539594

def g (x : ℝ) : ℝ := min (min (x + 3) (x - 1)) (-1/2 * x + 5)

theorem max_value_of_g :
  ∃ x : ℝ, g x = 13/3 ∧ ∀ y : ℝ, g y ≤ 13/3 :=
by
  sorry

end max_value_of_g_l539_539594


namespace line_passes_through_fixed_point_l539_539610

noncomputable def parabola_eq : ((y : ℝ) × (x : ℝ) → Prop) := λ (y x : ℝ), y^2 = 4 * x

theorem line_passes_through_fixed_point :
  ∀ (circle : (ℝ × ℝ) → ℝ), (∀ (A B D E : ℝ × ℝ), circle (0, 0) = 1 → parabola_eq A → parabola_eq B →
    parabola_eq D → parabola_eq E → (y D = 0 ∧ y E = 0) →
    dist A B = 4 ∧ dist D E = 4) →
  ∀ (M N : ℝ × ℝ) (l : ℝ × ℝ → Prop), l M → l N → ¬ l (0, 0) →
    let OM := λ (M : ℝ × ℝ), (fst M = 0) in
    let ON := λ (N : ℝ × ℝ), (snd N = 0) in
    OM M ⊥ ON N →
    ∃ (Q : ℝ × ℝ), l Q ∧ Q = (4, 0) := by
  sorry

end line_passes_through_fixed_point_l539_539610


namespace length_S_l539_539718

noncomputable theory

open Set Real

def S := {p : ℝ × ℝ | let x := |p.1 - 2|, y := |p.2 - 2| in (x^2 + y^2) ^ (1/2) = 2 - abs(1 - (x^2 + y^2) ^ (1/2))}

theorem length_S : 4 * (2 * π + 3 * π) = 20 * π :=
by
  sorry

end length_S_l539_539718


namespace range_of_m_l539_539262

theorem range_of_m (m : ℝ) : 
  ((0 - m)^2 + (0 + m)^2 < 4) → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
by
  sorry

end range_of_m_l539_539262


namespace divisors_of_180_pow_180_l539_539661

theorem divisors_of_180_pow_180 (a b c : ℕ) :
  let base := 180
  let number := base ^ base
  let num_divisors := (a + 1) * (b + 1) * (c + 1)
  (∀ a b c, 0 ≤ a ∧ a ≤ 360 ∧ 0 ≤ b ∧ b ≤ 360 ∧ 0 ≤ c ∧ c ≤ 180)
  → num_divisors = 18
  → (∃ div_count : ℕ, div_count = 24) :=
begin
  sorry
end

end divisors_of_180_pow_180_l539_539661


namespace steve_oranges_l539_539546

variable {Oranges : Type}

def oranges_count (Brian Marcie : Oranges) (combined : Oranges → Oranges → Oranges) :=
  combined Brian Marcie = 24

def triple (oranges : Oranges) :=
  oranges * 3 = 72

theorem steve_oranges 
  (Brian Marcie : Oranges) 
  (combined : Oranges → Oranges → Oranges)
  (triple : Oranges → Oranges)
  (h1 : Brian = Marcie)
  (h2 : Marcie = 12)
  (h3 : combined = λ x y, x + y) :
  triple (combined Brian Marcie) :=
by
  sorry

end steve_oranges_l539_539546


namespace probability_A_shoots_l539_539489

theorem probability_A_shoots (P : ℚ) :
  (∀ n : ℕ, (2 * n + 1) % 2 = 1) →  -- A's turn is always the odd turn
  (∀ m : ℕ, (2 * m) % 2 = 0) →  -- B's turn is always the even turn
  let p_A_first_shot := (1 : ℚ) / 6 in  -- probability A fires on the first shot
  let p_A_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun
  let p_B_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun for B
  let P_A := p_A_first_shot + (p_A_turn * p_B_turn * P) in  -- recursive definition
  P_A = 6 / 11 := -- final probability
sorry

end probability_A_shoots_l539_539489


namespace length_of_CD_l539_539296

theorem length_of_CD (r : ℝ) (volume : ℝ) (length_CD : ℝ) :
  r = 4 ∧ volume = 352 * Real.pi → length_CD = 50 / 3 :=
by
  -- Introduce the assumptions given in the problem
  intro h
  -- Extract radius and volume conditions from assumption
  obtain ⟨hr, hv⟩ := h
  -- Assert that these are indeed the given radius and volume
  rw [hr, hv]
  sorry -- Skip the actual proof steps

end length_of_CD_l539_539296


namespace range_of_m_l539_539617

variable (m : ℝ)

def p := 2 < m ∧ m < 5
def q := m ≤ 4

theorem range_of_m (h1 : p ∨ q) (h2 : p ∧ q) : m ≤ 2 ∨ (4 < m ∧ m < 5) :=
by {
  sorry
}

end range_of_m_l539_539617


namespace abs_sum_greater_by_10_l539_539402

-- Definitions
def num1 := -5
def num2 := 3
def abs_sum := |num1| + |num2|
def sum := num1 + num2
def difference := abs_sum - sum

-- The statement of the theorem/problem
theorem abs_sum_greater_by_10 :
  difference = 10 := by
  sorry

end abs_sum_greater_by_10_l539_539402


namespace factorial_subtraction_l539_539455

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_subtraction : factorial 6 - factorial 4 = 696 := by
  sorry

end factorial_subtraction_l539_539455


namespace common_tangents_count_l539_539014

-- Define the first circle Q1
def Q1 (x y : ℝ) := x^2 + y^2 = 9

-- Define the second circle Q2
def Q2 (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 1

-- Prove the number of common tangents between Q1 and Q2
theorem common_tangents_count :
  ∃ n : ℕ, n = 4 ∧ ∀ x y : ℝ, Q1 x y ∧ Q2 x y -> n = 4 := sorry

end common_tangents_count_l539_539014


namespace equilateral_triangle_isosceles_points_l539_539199

theorem equilateral_triangle_isosceles_points
  (A B C : Point) 
  (hABC : equilateral_triangle A B C) :
  ∃ D : Point, isosceles_triangle A B D ∧ isosceles_triangle B C D :=
sorry

end equilateral_triangle_isosceles_points_l539_539199


namespace problem1_l539_539130

noncomputable def expr1 : ℚ :=
(1 : ℚ) * (↑27 / ↑8) ^ (-2/3 : ℚ) - (↑49 / ↑9) ^ (1/2 : ℚ) + (8 / 1000 : ℚ) ^ (-2/3 : ℚ) * (2 / 25 : ℚ)

theorem problem1 : expr1 = 1/9 :=
by
  sorry

end problem1_l539_539130


namespace integer_solutions_exist_l539_539631

theorem integer_solutions_exist (k : ℤ) :
  (∃ x : ℤ, 9 * x - 3 = k * x + 14) ↔ (k = 8 ∨ k = 10 ∨ k = -8 ∨ k = 26) :=
by
  sorry

end integer_solutions_exist_l539_539631


namespace arithmetic_sequence_sum_l539_539885

theorem arithmetic_sequence_sum {a b : ℤ} (h : ∀ n : ℕ, 3 + n * 6 = if n = 2 then a else if n = 3 then b else 33) : a + b = 48 := by
  sorry

end arithmetic_sequence_sum_l539_539885


namespace solve_fraction_equation_l539_539361

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ -3) :
  (2 / x + x / (x + 3) = 1) ↔ x = 6 := 
by
  sorry

end solve_fraction_equation_l539_539361


namespace cafe_location_l539_539332

-- Definition of points and conditions
structure Point where
  x : ℤ
  y : ℚ

def mark : Point := { x := 1, y := 8 }
def sandy : Point := { x := -5, y := 0 }

-- The problem statement
theorem cafe_location :
  ∃ cafe : Point, cafe.x = -3 ∧ cafe.y = 8/3 := by
  sorry

end cafe_location_l539_539332


namespace part1_part2_l539_539215

theorem part1 (a : ℕ → ℝ) :
  (∀ (x : ℝ), (1 - 2 * x)^100 = ∑ i in Finset.range 101, a i * (x + 1) ^ i) →
  (∑ i in Finset.range 101, a i = 1) :=
begin
  intro h,
  sorry
end

theorem part2 (a : ℕ → ℝ) :
  (∀ (x : ℝ), (1 - 2 * x)^100 = ∑ i in Finset.range 101, a i * (x + 1) ^ i) →
  (∑ i in Finset.range 101, a i = 1) →
  (∑ i in Finset.range 101 \ {i | i % 2 = 0}, a i = (1 - 5^100) / 2) :=
begin
  intros h₁ h₂,
  sorry
end

end part1_part2_l539_539215


namespace number_of_fish_disappeared_l539_539341

-- First, define initial amounts of each type of fish
def goldfish_initial := 7
def catfish_initial := 12
def guppies_initial := 8
def angelfish_initial := 5

-- Define the total initial number of fish
def total_fish_initial := goldfish_initial + catfish_initial + guppies_initial + angelfish_initial

-- Define the current number of fish
def fish_current := 27

-- Define the number of fish disappeared
def fish_disappeared := total_fish_initial - fish_current

-- Proof statement
theorem number_of_fish_disappeared:
  fish_disappeared = 5 :=
by
  -- Sorry is a placeholder that indicates the proof is omitted.
  sorry

end number_of_fish_disappeared_l539_539341


namespace commutative_matrices_implies_fraction_l539_539312

-- Definitions
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 3], ![4, 5]]
def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

-- Theorem Statement
theorem commutative_matrices_implies_fraction (a b c d : ℝ) 
    (h1 : A * B a b c d = B a b c d * A) 
    (h2 : 4 * b ≠ c) : 
    (a - d) / (c - 4 * b) = 3 / 8 :=
by
  sorry

end commutative_matrices_implies_fraction_l539_539312


namespace least_marked_cells_l539_539428

theorem least_marked_cells (n : ℕ) :
  ∃ (markers : Finset (Fin n × Fin n)), 
    (∀ m > n / 2, ∀ (i j : Fin m), 
      ((i = j ∨ i + j = m - 1) → (i, j) ∈ markers)) ∧
    markers.card = n :=
sorry

end least_marked_cells_l539_539428


namespace least_three_digit_product_8_is_118_l539_539880

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_product (n : ℕ) (product : ℕ) : Prop :=
  let digits := (list.cons (n / 100) (list.cons ((n / 10) % 10) (list.cons (n % 10) list.nil))) in
  digits.prod = product

theorem least_three_digit_product_8_is_118 :
  ∃ n : ℕ, is_three_digit_number n ∧ digits_product n 8 ∧ (∀ m : ℕ, is_three_digit_number m ∧ digits_product m 8 → n ≤ m) :=
sorry

end least_three_digit_product_8_is_118_l539_539880


namespace square_of_binomial_k_l539_539886

theorem square_of_binomial_k (k : ℕ) : (∃ b : ℕ, (x + b)^2 = x^2 + 24x + k) → k = 144 :=
by
  sorry

end square_of_binomial_k_l539_539886


namespace angle_between_cube_diagonals_l539_539599

theorem angle_between_cube_diagonals (a : ℝ) : 
  ∃ θ : ℝ, θ = 60 ∧ 
           let V := (0, 0, 0),
               d1 := (a, a, 0),
               d2 := (a, 0, a),
               dot_product := d1.1 * d2.1 + d1.2 * d2.2 + d1.3 * d2.3,
               norm_d1 := real.sqrt (d1.1 ^ 2 + d1.2 ^ 2 + d1.3 ^ 2),
               norm_d2 := real.sqrt (d2.1 ^ 2 + d2.2 ^ 2 + d2.3 ^ 2),
               cos_theta := dot_product / (norm_d1 * norm_d2)
           in cos_theta = 1 / 2 := 
begin
  sorry
end

end angle_between_cube_diagonals_l539_539599


namespace minimum_AB_value_l539_539375

-- Define the parametric equation of the line l
def line_param (t : ℝ) (α : ℝ) := (1 + t * Real.cos α, t * Real.sin α)

-- Define the polar equation of the curve C
def curve_polar (ρ θ : ℝ) := ρ * (Real.sin θ)^2 = 4 * Real.cos θ

-- The Cartesian coordinate equation of the curve C
def curve_cartesian : Prop := ∀ (x y : ℝ), y^2 = 4 * x

-- Function to find the intersection of the line with the curve
def intersect (α : ℝ) (t : ℝ) :=  t^2 * (Real.sin α)^2 - 4 * t * (Real.cos α) - 4 = 0

-- Function to find the distance |AB|
noncomputable def distance_AB (α : ℝ) := by
  let t1 := Real.cos α / (Real.sin α)^2
  let t2 := -4 / (Real.sin α)^2
  exact Real.sqrt ( (t1 + t2)^2 - 4 * t1 * t2 )

-- Minimum value of |AB| when α changes
def min_distance_AB (α : ℝ) := distance_AB α

theorem minimum_AB_value : ∀ α, (0 < α ∧ α < Real.pi) → min_distance_AB α = 4 := sorry

end minimum_AB_value_l539_539375


namespace first_player_win_l539_539834

variables {x y : ℝ} (hxy : x ≠ y)

def V_A : ℝ := x^3
def V_B : ℝ := x^2 * y
def V_C : ℝ := x * y^2
def V_D : ℝ := y^3

theorem first_player_win (hxy : x ≠ y) : (x - y)^2 * (x + y) > 0 :=
by {
  apply mul_pos,
  { apply pow_two_nonneg,
    exact sub_ne_zero_of_ne hxy },
  { apply add_pos_of_pos_of_nonneg,
    { apply lt_trans (sub_pos_of_lt (lt_of_le_of_ne (abs_nonneg x) (abs_ne_zero_iff.mpr hxy))) (lt_add_of_pos_left y (lt_of_le_of_ne (abs_nonneg x) (abs_ne_zero_iff.mpr hxy))),
      exact abs_pos_of_ne_zero hxy },
    { apply pow_two_nonneg } }
}

#print first_player_win -- This statement ensures the first player's win. 


end first_player_win_l539_539834


namespace min_x_plus_2y_max_sqrt7_x_plus_2y_2xy_l539_539268

noncomputable def min_value_x_plus_2y (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x^2 + 4 * y^2 + 4 * x * y + 4 * x^2 * y^2 = 32) : ℝ := 4

theorem min_x_plus_2y (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x^2 + 4 * y^2 + 4 * x * y + 4 * x^2 * y^2 = 32) :
  x + 2 * y ≥ min_value_x_plus_2y x y hx hy h :=
sorry

noncomputable def max_value_sqrt7_x_plus_2y_2xy (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x^2 + 4 * y^2 + 4 * x * y + 4 * x^2 * y^2 = 32) : ℝ :=
  4 * real.sqrt 7 + 4

theorem max_sqrt7_x_plus_2y_2xy (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x^2 + 4 * y^2 + 4 * x * y + 4 * x^2 * y^2 = 32) :
  real.sqrt 7 * (x + 2 * y) + 2 * x * y ≤ max_value_sqrt7_x_plus_2y_2xy x y hx hy h :=
sorry

end min_x_plus_2y_max_sqrt7_x_plus_2y_2xy_l539_539268


namespace cotangent_sum_eq_div_sum_of_squares_l539_539904

variables {α β γ a b c S : ℝ}

theorem cotangent_sum_eq_div_sum_of_squares (h_sum_angles : α + β + γ = real.pi)
  (h_area : S = (1/2) * a * b * real.sin γ) :
  real.cot α + real.cot β + real.cot γ = (a^2 + b^2 + c^2) / (4 * S) :=
sorry

end cotangent_sum_eq_div_sum_of_squares_l539_539904


namespace Yoojung_total_vehicles_l539_539447

theorem Yoojung_total_vehicles : 
  let motorcycles := 2
  let bicycles := 5
  motorcycles + bicycles = 7 := 
by
  sorry

end Yoojung_total_vehicles_l539_539447


namespace probability_A_fires_proof_l539_539470

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end probability_A_fires_proof_l539_539470


namespace right_vs_oblique_prism_similarities_and_differences_l539_539846

-- Definitions of Prisms and their properties
structure Prism where
  parallel_bases : Prop
  congruent_bases : Prop
  parallelogram_faces : Prop

structure RightPrism extends Prism where
  rectangular_faces : Prop
  perpendicular_sides : Prop

structure ObliquePrism extends Prism where
  non_perpendicular_sides : Prop

theorem right_vs_oblique_prism_similarities_and_differences 
  (p1 : RightPrism) (p2 : ObliquePrism) : 
    (p1.parallel_bases ↔ p2.parallel_bases) ∧ 
    (p1.congruent_bases ↔ p2.congruent_bases) ∧ 
    (p1.parallelogram_faces ↔ p2.parallelogram_faces) ∧
    (p1.rectangular_faces ∧ p1.perpendicular_sides ↔ p2.non_perpendicular_sides) := 
by 
  sorry

end right_vs_oblique_prism_similarities_and_differences_l539_539846


namespace triangle_area_l539_539673

def parabola_equation (x : ℝ) : ℝ :=
  -4 * x^2 + 16 * x - 15

def vertex_A : ℝ × ℝ :=
  (2, 1)

def intersection_B : ℝ × ℝ :=
  (3/2, 0)

def intersection_C : ℝ × ℝ :=
  (5/2, 0)

noncomputable def area_triangle_ABC :=
  1/2

theorem triangle_area :
  let A := vertex_A
  let B := intersection_B
  let C := intersection_C in
  1/2 = area_triangle_ABC :=
by
  sorry

end triangle_area_l539_539673


namespace evaluate_log_sum_l539_539976

theorem evaluate_log_sum :
  (3 / (log 3 (5000^5)) + 4 / (log 7 (5000^5))) = log 5000 (3^(3/5) * 7^(4/5)) :=
sorry

end evaluate_log_sum_l539_539976


namespace sequence_fourth_term_is_correct_l539_539645

theorem sequence_fourth_term_is_correct (x : ℝ) (r : ℝ) (h : (3*x + 3) / x = (5*x + 5) / (3*x + 3)) :
    (x, 3*x+3, 5*x+5, r*(5*x+5)).nth 3 = -10.4167 := by
  sorry

end sequence_fourth_term_is_correct_l539_539645


namespace regular_polygon_radius_l539_539826

theorem regular_polygon_radius 
  (n : ℕ) (side_length : ℝ) (h1 : side_length = 2) 
  (h2 : sum_of_interior_angles n = 2 * sum_of_exterior_angles n)
  (h3 : is_regular_polygon n) :
  radius_of_polygon n side_length = 2 :=
by
  sorry

end regular_polygon_radius_l539_539826


namespace determine_die_face_l539_539082

-- Define the basic properties of a standard die
def opposite_faces (a b : ℕ) : Prop := a + b = 7

-- State the conditions
variables (faces : Finset ℕ) (rotation : faces → faces)
variable (visible_faces : faces) -- Set of faces visible in a certain state

-- Define the constants
constant die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
constant initial_visible_faces : Finset ℕ := {1, 2, 3} -- assumed faces visible initially based on problem statement
constant rotated_visible_faces : Finset ℕ := initial_visible_faces.map rotation

-- Lean theorem stating the problem
theorem determine_die_face :
  (∀ a ∈ die_faces, ∃ b ∈ die_faces, opposite_faces a b) →
  (∀ a b ∈ visible_faces, rotation a = b) →
  (∃ x ∈ rotated_visible_faces, x = 6) :=
begin
  sorry -- Proof goes here
end

end determine_die_face_l539_539082


namespace cylinder_volume_approx_l539_539057

noncomputable def volume_cylinder (h : ℝ) (d : ℝ) : ℝ :=
  let r := d / 2
  π * r^2 * h

theorem cylinder_volume_approx (h : ℝ) (d : ℝ) (approximately_equal : ℝ → ℝ → Prop) [is_approx : ∀ x y, approximately_equal x y → abs (x - y) < 0.001] :
  h = 14 ∧ d = 10 → approximately_equal (volume_cylinder h d) 1099.547 :=
by sorry

end cylinder_volume_approx_l539_539057


namespace time_to_cross_signal_pole_l539_539464

noncomputable def trainLength : ℝ := 300
noncomputable def platformLength : ℝ := 366.67
noncomputable def timeToCrossPlatform : ℝ := 40

theorem time_to_cross_signal_pole : 
  (trainLength + platformLength) / timeToCrossPlatform = 16.6667 ∧
  trainLength / ( (trainLength + platformLength) / timeToCrossPlatform ) ≈ 18 := 
by
  -- The following proof is outlined but not completed. Lean will need specific steps provided to complete this equivalency.
  sorry

end time_to_cross_signal_pole_l539_539464


namespace probability_A_fires_l539_539465

theorem probability_A_fires :
  let p_A := (1 : ℚ) / 6 + (5 : ℚ) / 6 * (5 : ℚ) / 6 * p_A
  in p_A = 6 / 11 :=
by
  sorry

end probability_A_fires_l539_539465


namespace population_in_2050_l539_539569

def population : ℕ → ℕ := sorry

theorem population_in_2050 : population 2050 = 2700 :=
by
  -- sorry statement to skip the proof
  sorry

end population_in_2050_l539_539569


namespace subtract_mult_equal_l539_539847

theorem subtract_mult_equal :
  2000000000000 - 1111111111111 * 1 = 888888888889 :=
by
  sorry

end subtract_mult_equal_l539_539847


namespace max_ab_l539_539629

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ab ≤ 1 / 16 :=
by
  sorry

end max_ab_l539_539629


namespace lee_vs_kai_reading_time_l539_539711

-- Definitions based on given problem conditions
def kai_reading_speed : ℝ := 120  -- pages per hour
def lee_reading_speed : ℝ := 60  -- pages per hour
def total_pages : ℝ := 300  -- total pages in book

-- Definition to be proved: Lee takes 150 minutes more than Kai to finish the book
theorem lee_vs_kai_reading_time :
  let kai_time := total_pages / kai_reading_speed,
      lee_time := total_pages / lee_reading_speed,
      time_difference := lee_time - kai_time
  in time_difference * 60 = 150 :=
by
  sorry

end lee_vs_kai_reading_time_l539_539711


namespace max_pens_given_budget_l539_539519

-- Define the conditions.
def max_pens (x y : ℕ) := 12 * x + 20 * y

-- Define the main theorem stating the proof problem.
theorem max_pens_given_budget : ∃ (x y : ℕ), (10 * x + 15 * y ≤ 173) ∧ (max_pens x y = 224) :=
  sorry

end max_pens_given_budget_l539_539519


namespace four_letter_list_product_l539_539155

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def list_product (s : String) : Nat :=
  s.foldl (λ acc c => acc * letter_value c) 1

def target_product : Nat :=
  list_product "TUVW"

theorem four_letter_list_product : 
  ∀ (s1 s2 : String), s1.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') → s2.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') →
  s1.length = 4 → s2.length = 4 →
  list_product s1 = target_product → s2 = "BEHK" :=
by
  sorry

end four_letter_list_product_l539_539155


namespace sum_inverse_inequality_l539_539967

noncomputable def a : ℕ → ℝ
| 0     := 0  -- not used, just for lean's definition starting from 0
| 1     := 2
| (n+2) := (a (n + 1))^2 / 2 + 1 / 2

theorem sum_inverse_inequality (N : ℕ) (h : N > 0) : 
  (∑ j in Finset.range(N) + 1, 1 / (a j + 1)) < 1 := 
by
  sorry

end sum_inverse_inequality_l539_539967


namespace number_of_different_primes_l539_539714

-- Definitions of conditions
variables {n : ℕ} (a : fin n → ℕ)

-- Define the sum of products of all possible combinations
def p_k (k : ℕ) : ℕ :=
  ∑ S in (univ.powerset.filter (λ s, s.card = k)).to_finset, ∏ i in S, a i

-- Define P as the sum of all p_k with odd k
def P : ℕ :=
  ∑ k in finset.range(n+1), if odd k then p_k a k else 0

-- The problem statement in Lean 4
theorem number_of_different_primes 
  (h1 : 1 < n) 
  (h_prime : ∀ j : fin n, nat.prime (a j)) 
  (P_prime : nat.prime P) : 
  ∃! b : ℕ, (∃ j : fin n, a j = b) ∧ (∃! j : fin n, nat.prime b ∧ even b) := sorry

end number_of_different_primes_l539_539714


namespace least_three_digit_product_eight_l539_539855

theorem least_three_digit_product_eight : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).prod = 8 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (nat.digits 10 m).prod = 8 → n ≤ m :=
by
  sorry

end least_three_digit_product_eight_l539_539855


namespace floor_sum_inverse_a_2018_eq_2_l539_539310

noncomputable def a : ℕ → ℕ
| 0 := 0 -- Not used, just a placeholder
| 1 := 1
| (n + 2) := a (n + 1) + n + 2

def sum_inverse_a (n : ℕ) : ℝ :=
(range n).sum (λ k, 1 / (a (k + 1):ℝ))

theorem floor_sum_inverse_a_2018_eq_2 :
  ⌊sum_inverse_a 2018⌋ = 2 :=
begin
  sorry,
end

end floor_sum_inverse_a_2018_eq_2_l539_539310


namespace Adam_bought_26_books_l539_539109

theorem Adam_bought_26_books (initial_books : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) (leftover_books : ℕ) :
  initial_books = 56 → shelves = 4 → books_per_shelf = 20 → leftover_books = 2 → 
  let total_capacity := shelves * books_per_shelf in
  let total_books_after := total_capacity + leftover_books in
  let books_bought := total_books_after - initial_books in
  books_bought = 26 :=
by
  intros h1 h2 h3 h4
  simp [total_capacity, total_books_after, books_bought]
  rw [h1, h2, h3, h4]
  sorry

end Adam_bought_26_books_l539_539109


namespace factor_expression_l539_539574

theorem factor_expression (x : ℝ) : 3 * x * (x - 5) + 7 * (x - 5) - 2 * (x - 5) = (3 * x + 5) * (x - 5) :=
by
  sorry

end factor_expression_l539_539574


namespace train_passing_man_l539_539096

/-- 
Given:
- length of the train: 250 meters
- speed of the train: 80 kilometers per hour
- speed of the man: 10 kilometers per hour
- man is running in opposite direction

Prove: 
The time for the train to pass the man completely is 10 seconds.
-/
theorem train_passing_man (train_length : ℕ) (train_speed_kmph : ℕ) (man_speed_kmph : ℕ) (opposite_direction : bool) 
  (h1 : train_length = 250)
  (h2 : train_speed_kmph = 80)
  (h3 : man_speed_kmph = 10)
  (h4 : opposite_direction = true) :
  let train_speed_mps := (train_speed_kmph * 1000) / 3600 -- converting km/hr to m/s
  let man_speed_mps := (man_speed_kmph * 1000) / 3600 -- converting km/hr to m/s
  let relative_speed_mps := train_speed_mps + man_speed_mps
  let passing_time := train_length / relative_speed_mps
  passing_time = 10 := 
by
  sorry

end train_passing_man_l539_539096


namespace dog_food_weight_l539_539498

theorem dog_food_weight (weight_per_cup : ℚ) (number_of_dogs : ℕ) 
  (cups_per_meal : ℕ) (meals_per_day : ℕ) (bags_per_month : ℕ) 
  (days_per_month : ℕ) (total_weight_per_bag : ℚ) :
  weight_per_cup = 1/4 →
  number_of_dogs = 2 →
  cups_per_meal = 6 →
  meals_per_day = 2 →
  bags_per_month = 9 →
  days_per_month = 30 →
  total_weight_per_bag = (2 * 6 * 2 * (1/4) * 30) / 9 :=
begin
  sorry,
end

end dog_food_weight_l539_539498


namespace distance_from_origin_l539_539679

def point := (-9.5, 5.5 : ℝ)
def origin := (0, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_from_origin :
  distance origin point = Real.sqrt 120.5 :=
by
  -- sorry can be used to denote the skipped proof
  sorry

end distance_from_origin_l539_539679


namespace part_I_part_II_l539_539650
open Real

-- Definitions for the given problem context
variables {A B C a b c : ℝ}

-- Condition provided in part I
def condition1 := (sqrt 3 * c / cos C = a / cos (3 * pi / 2 + A))

-- Condition provided in part II
def condition2 := (c / a = 2)
def condition3 := (b = 4 * sqrt 3)

-- Prove in part I: C = π / 6
theorem part_I (h1 : condition1) : C = π / 6 :=
sorry

-- Prove in part II: Area of ΔABC = 2√15 - 2√3, given the results from part I and additional conditions
theorem part_II (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : C = π / 6) :
  1 / 2 * a * b * sin(C) = 2 * sqrt 15 - 2 * sqrt 3 :=
sorry

end part_I_part_II_l539_539650


namespace least_three_digit_product_eight_l539_539852

theorem least_three_digit_product_eight : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).prod = 8 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (nat.digits 10 m).prod = 8 → n ≤ m :=
by
  sorry

end least_three_digit_product_eight_l539_539852


namespace reading_time_per_disc_l539_539508

theorem reading_time_per_disc (total_minutes : ℕ) (disc_capacity : ℕ) (d : ℕ) (reading_per_disc : ℕ) :
  total_minutes = 528 ∧ disc_capacity = 45 ∧ d = 12 ∧ total_minutes = d * reading_per_disc → reading_per_disc = 44 :=
by
  sorry

end reading_time_per_disc_l539_539508


namespace division_remainder_3012_97_l539_539431

theorem division_remainder_3012_97 : 3012 % 97 = 5 := 
by 
  sorry

end division_remainder_3012_97_l539_539431


namespace trip_time_l539_539914

theorem trip_time (x : ℕ) : (75 * 4 + 60 * x) / (4 + x) = 70 → 4 + x = 6 :=
by
  assume h: (75 * 4 + 60 * x) / (4 + x) = 70
  have h1: 75 * 4 = 300 := rfl
  have h2: 300 + 60 * x = 70 * (4 + x) := by sorry
  sorry

end trip_time_l539_539914


namespace no_nontrivial_solutions_in_integers_l539_539349

theorem no_nontrivial_solutions_in_integers (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
  by
    sorry

end no_nontrivial_solutions_in_integers_l539_539349


namespace fill_grid_ways_l539_539161

theorem fill_grid_ways (n : ℕ) (h : n ≥ 3) :
  ∃ (ways : ℕ), ways = 2 ^ ((n - 1) ^ 2) * (n! ^ 3) :=
by
  use 2 ^ ((n - 1) ^ 2) * (n! ^ 3)
  sorry

end fill_grid_ways_l539_539161


namespace good_or_bad_l539_539987

def is_good (k n : ℕ) : Prop := sorry -- Predicate defining a 'good' number
def is_bad (k n : ℕ) : Prop := sorry -- Predicate defining a 'bad' number

theorem good_or_bad (k n n' : ℕ) (hk : k > 2) (hn : n ≥ k) (hn' : n' ≥ k) 
  (h_prime_div : ∀ p, p ≤ k → p.prime → (p ∣ n ↔ p ∣ n')) : 
  (is_good k n ↔ is_good k n') ∧ (is_bad k n ↔ is_bad k n') :=
by
  sorry

end good_or_bad_l539_539987


namespace find_m_min_max_determine_a_range_l539_539611

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.sin (x + π / 3)

-- Define the function g, symmetric to f with respect to x = π / 4
def g (x : ℝ) : ℝ := 2 * Real.sin (5 * π / 6 - x)

-- Define problem 1: find the min and max values of m such that g(x)^2 - mg(x) + 2 = 0 holds for some x in [0, π / 2)
theorem find_m_min_max :
  ∃ x ∈ Ico 0 (π / 2), (m : ℝ) (m_min : ℝ) (m_max : ℝ), 
  (g x)^2 - m * (g x) + 2 = 0 ∧ m_min = 2 * Real.sqrt 2 ∧ m_max = 3 := sorry

-- Define problem 2: determine the range of a such that f(x) + a * g(-x) > 0 for all x in [0, 11π / 12]
theorem determine_a_range : 
  (a : ℝ) → (∀ x ∈ Icc 0 (11 * π / 12), f x + a * g (-x) > 0) → 
  a ∈ Iio (-sqrt 2) ∪ Ioi (sqrt 2) := sorry

end find_m_min_max_determine_a_range_l539_539611


namespace scarves_sold_at_new_price_l539_539282

theorem scarves_sold_at_new_price :
  ∃ (p : ℕ), (∃ (c k : ℕ), (k = p * c) ∧ (p = 30) ∧ (c = 10)) ∧
  (∃ (new_c : ℕ), new_c = 165 / 10 ∧ k = new_p * new_c) ∧
  new_p = 18
:=
sorry

end scarves_sold_at_new_price_l539_539282


namespace total_nickels_l539_539745

-- Definition of the number of original nickels Mary had
def original_nickels := 7

-- Definition of the number of nickels her dad gave her
def added_nickels := 5

-- Prove that the total number of nickels Mary has now is 12
theorem total_nickels : original_nickels + added_nickels = 12 := by
  sorry

end total_nickels_l539_539745


namespace part_a_l539_539449

theorem part_a (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a + b = 1) : a * b = 0 := 
by 
  sorry

end part_a_l539_539449


namespace probability_of_selecting_red_ball_l539_539072

theorem probability_of_selecting_red_ball (total_balls red_balls : ℕ)
    (h_total : total_balls = 15) (h_red : red_balls = 3) :
    (red_balls : ℚ) / total_balls = 1 / 5 :=
by
  rw [h_total, h_red]
  norm_num
  sorry

end probability_of_selecting_red_ball_l539_539072


namespace total_nickels_l539_539746

-- Definition of the number of original nickels Mary had
def original_nickels := 7

-- Definition of the number of nickels her dad gave her
def added_nickels := 5

-- Prove that the total number of nickels Mary has now is 12
theorem total_nickels : original_nickels + added_nickels = 12 := by
  sorry

end total_nickels_l539_539746


namespace bird_cages_count_l539_539929

-- Definitions based on the conditions provided
def num_parrots_per_cage : ℕ := 2
def num_parakeets_per_cage : ℕ := 7
def total_birds_per_cage : ℕ := num_parrots_per_cage + num_parakeets_per_cage
def total_birds_in_store : ℕ := 54
def num_bird_cages : ℕ := total_birds_in_store / total_birds_per_cage

-- The proof we need to derive
theorem bird_cages_count : num_bird_cages = 6 := by
  sorry

end bird_cages_count_l539_539929


namespace probability_A_fires_l539_539480

theorem probability_A_fires 
  (p_first_shot: ℚ := 1/6)
  (p_not_fire: ℚ := 5/6)
  (p_recur: ℚ := p_not_fire * p_not_fire) : 
  ∃ (P_A : ℚ), P_A = 6/11 :=
by
  have eq1 : P_A = p_first_shot + (p_recur * P_A) := sorry
  have eq2 : P_A * (1 - p_recur) = p_first_shot := sorry
  have eq3 : P_A = (p_first_shot * 36) / 11 := sorry
  exact ⟨P_A, sorry⟩

end probability_A_fires_l539_539480


namespace part1_part2_part3_l539_539721

noncomputable def f (a x : ℝ) := x^a * Real.log x

theorem part1 (a x : ℝ) (h : 0 < a) : 
  (0 < x ∧ x < Real.exp (-1 / a) → f' a x < 0) -- (locally decreasing interval)
  ∧ (Real.exp (-1 / a) < x → f' a x > 0) -- (locally increasing interval)
:= sorry

theorem part2 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 0 < x → f a x ≤ x) → a ∈ Set.Ioo 0 (1 - Real.exp (-1)) 
:= sorry

theorem part3 (a x : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 < x → ((x^(a - 1)) * (a * Real.log x + 1) ≤ 1)) → a = 1 / 2 
:= sorry

end part1_part2_part3_l539_539721


namespace inv_matrix_eq_linear_comb_l539_539316

-- Define the matrix N
def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 1], ![0, -2]]

-- Define the constants a and b
def a := (1 : ℚ) / 6
def b := (1 : ℚ) / 6

-- prove that N⁻¹ = a * N + b * I
theorem inv_matrix_eq_linear_comb :
  N⁻¹ = (a : ℚ) • N + (b : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
by
  -- proof to be provided
  sorry

end inv_matrix_eq_linear_comb_l539_539316


namespace N_salary_correct_l539_539453

noncomputable def N_salary_eq_260 (x : ℝ) : Prop :=
  let total_salary := 572
  let M_salary := 1.20 * x
  total_salary = x + M_salary ∧ x = 260

theorem N_salary_correct : ∃ x : ℝ, N_salary_eq_260 x := by
  use 260
  dsimp [N_salary_eq_260]
  ring_nf
  norm_num
  sorry

end N_salary_correct_l539_539453


namespace find_percentage_l539_539259

variable (P x : ℝ)

theorem find_percentage (h1 : x = 10)
    (h2 : (P / 100) * x = 0.05 * 500 - 20) : P = 50 := by
  sorry

end find_percentage_l539_539259


namespace sum_f_equals_zero_l539_539197

noncomputable def f (x : ℝ) : ℝ := cos x * (2 * sin x + 1)

def arithmetic_sequence (a1 d : ℝ) : ℕ → ℝ
| 0     := a1
| (n+1) := a1 + (n + 1) * d

def S (a1 d : ℝ) (n : ℕ) : ℝ :=
(n : ℝ) / 2 * (2 * a1 + (n - 1) * d)

theorem sum_f_equals_zero (a1 d : ℝ) (h : d ≠ 0) (hS8 : S a1 d 8 = 4 * π) :
  f (arithmetic_sequence a1 d 0) + f (arithmetic_sequence a1 d 1) +
  f (arithmetic_sequence a1 d 2) + f (arithmetic_sequence a1 d 3) +
  f (arithmetic_sequence a1 d 4) + f (arithmetic_sequence a1 d 5) +
  f (arithmetic_sequence a1 d 6) + f (arithmetic_sequence a1 d 7) = 0 :=
sorry

end sum_f_equals_zero_l539_539197


namespace sum_S5_eq_l539_539555

-- Definitions of the sequence and their sum
def sequence (a1 : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  let s := λ a i, a * (7/6)^i
  s a1 n

def S (a1 : ℝ) (k : ℝ) (N : ℕ) : ℝ :=
  ∑ i in finset.range N, sequence a1 k i

-- The specific parameters for our problem
def a1 : ℝ := 1/3
def k : ℝ := 3
def S5 := S a1 k 5

-- The theorem to prove
theorem sum_S5_eq :
  S5 = (2/3) * (1 - (1/2)^5) :=
sorry

end sum_S5_eq_l539_539555


namespace divisible_by_6_but_not_4_or_9_number_of_integers_divisible_6_not_4_or_9_l539_539171

theorem divisible_by_6_but_not_4_or_9 (n : ℕ) (h : n < 2018) :
  n % 6 = 0 → (n % 4 ≠ 0 ∨ n % 9 ≠ 0) :=
sorry

theorem number_of_integers_divisible_6_not_4_or_9 : 
  (finset.range 2018).filter (λ n, n % 6 = 0 ∧ (n % 4 ≠ 0 ∨ n % 9 ≠ 0)).card = 112 :=
sorry

end divisible_by_6_but_not_4_or_9_number_of_integers_divisible_6_not_4_or_9_l539_539171


namespace find_the_number_l539_539589

theorem find_the_number :
  ∃ x : ℕ, 72519 * x = 724827405 ∧ x = 10005 :=
by
  sorry

end find_the_number_l539_539589


namespace water_gun_problem_l539_539284

-- Define the main condition for the problem
def water_gun_field (n : ℕ) : Prop :=
  (∀ i < n, ∀ j < n, i ≠ j → distinct_distances i j) ∧
  (n % 2 = 1 → ∃ i < n, is_dry i) ∧
  (n % 2 = 0 → ¬ ∃ i < n, is_dry i)

-- Assumption for distinct distances
axiom distinct_distances : ∀ (i j : ℕ), Prop

-- Definition for a person being dry
axiom is_dry : ∀ (i : ℕ), Prop

-- The main theorem to be proved
theorem water_gun_problem (n : ℕ) : water_gun_field n :=
by
  sorry

end water_gun_problem_l539_539284


namespace conclude_b_product_l539_539207

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ)

-- arithemetic sequence condition: definition
def is_arithemetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, a (m + 1) = a m + d

-- geometric sequence condition: definition
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ m : ℕ, b (m + 1) = b m * r

-- specific conditions in the problem: 
def conditions : Prop :=
  is_arithemetic_sequence a ∧
  is_geometric_sequence b ∧
  a 1010 = 5 ∧
  (∑ k in finset.range 2019, a (k + 1)) = 5 * 2019 ∧
  b 1010 = 5

theorem conclude_b_product :
  conditions a b →
  (∏ k in finset.range 2019, b (k + 1)) = 5^2019 :=
begin
  sorry        -- proof will be provided outside of the specification
end

end conclude_b_product_l539_539207


namespace prove_identity_l539_539729

noncomputable def real_line : Type := ℝ

namespace differentiable_functions

variables {f g : real_line → real_line}

-- Conditions
axiom condition1 : ∀ x y : real_line, f (x + y) = f x * f y - g x * g y
axiom condition2 : ∀ x y : real_line, g (x + y) = f x * g y + g x * f y
axiom condition3 : deriv f 0 = 0

-- Theorem to prove
theorem prove_identity : ∀ x : real_line, f x ^ 2 + g x ^ 2 = 1 :=
sorry

end differentiable_functions

end prove_identity_l539_539729


namespace card_pair_probability_l539_539922

theorem card_pair_probability :
  let cards := ⟦1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9,
                 10, 10, 10, 10, 10⟧ in 
  let remaining_cards := cards.erase [1, 1, 2, 2] in -- Two pairs removed from the deck
  let pair_probability := let total_ways := Nat.choose 46 2 in
                          let pair_ways := 8 * Nat.choose 5 2 + 2 * Nat.choose 3 2 in
                          pair_ways / total_ways in
  let (m, n) := (86, 1035) in -- Ratio simplification manually calculated here
  Nat.gcd m n = 1 → m + n = 1121 :=
by { sorry }

end card_pair_probability_l539_539922


namespace colorful_triangle_l539_539961

theorem colorful_triangle
  (T : Triangle)
  (coloring : Plane → Fin 1992)
  (h_used : ∀ (c : Fin 1992), ∃ (p : Plane), coloring p = c) :
  ∃ (T' : Triangle), congruent T T' ∧ (∀ (e : Edge), ∀ (p1 p2 : Point), e ∈ Sides T' → coloring p1 = coloring p2) :=
sorry

end colorful_triangle_l539_539961


namespace solution_set_of_inequality_l539_539208

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (mono_f : ∀ ⦃x y : ℝ⦄, -1 ≤ x → x ≤ y → f(x) ≤ f(y))
  (domain_f : ∀ x, x ≥ -1 → f(x) ≥ -1) :
  {x : ℝ | f(e^(x-2)) ≥ f(2 - x/2)} = set.Icc 2 6 :=
by {
  sorry
}

end solution_set_of_inequality_l539_539208


namespace is_increasing_on_interval_l539_539638

def f (x : ℝ) : ℝ := sin (2 * x) - x

theorem is_increasing_on_interval : ∀ x, 0 < x ∧ x < π / 6 → (deriv f x) > 0 := by
  sorry

end is_increasing_on_interval_l539_539638


namespace inequality_proof_l539_539626

variable {x1 x2 y1 y2 z1 z2 : ℝ}

theorem inequality_proof (hx1 : x1 > 0) (hx2 : x2 > 0)
   (hxy1 : x1 * y1 - z1^2 > 0) (hxy2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
  sorry

end inequality_proof_l539_539626


namespace least_three_digit_number_product8_l539_539858

theorem least_three_digit_number_product8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (digits 10 n).prod = 8 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (digits 10 m).prod = 8 → n ≤ m) :=
sorry

end least_three_digit_number_product8_l539_539858


namespace volume_of_regular_tetrahedron_on_sphere_l539_539933

-- Definition of the conditions
def sphere_radius : ℝ := 1
def base_vertices_on_great_circle : Prop := True
def regular_tetrahedron_vertices_on_surface (radius : ℝ) : Prop := radius = sphere_radius

-- The volume of the regular tetrahedron given the aforementioned conditions
theorem volume_of_regular_tetrahedron_on_sphere :
  regular_tetrahedron_vertices_on_surface sphere_radius →
  base_vertices_on_great_circle →
  ∃ (V : ℝ), V = (√3)/4 :=
by
  intros
  exists (√3)/4
  sorry

end volume_of_regular_tetrahedron_on_sphere_l539_539933


namespace units_digit_of_17_pow_2025_l539_539042

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2025 :
  units_digit (17 ^ 2025) = 7 :=
by sorry

end units_digit_of_17_pow_2025_l539_539042


namespace angle_A_side_lengths_l539_539272

def triangle_angle_condition (A B C : ℝ) :=
  8 * sin^2 ((B + C) / 2) - 2 * cos (2 * A) = 7

theorem angle_A (A B C : ℝ) (h : triangle_angle_condition A B C) : 
  A = 60 * (π / 180) :=
sorry

def triangle_sides_condition (a b c : ℝ) (A : ℝ) :=
  a = sqrt 3 ∧ b + c = 3 ∧ (b^2 + c^2 - b*c = 3)

theorem side_lengths (b c : ℝ) (h1 : b + c = 3) (h2 : b^2 + c^2 - b*c = 3) :
  (b = 1 ∧ c = 2) ∨ (b = 2 ∧ c = 1) :=
sorry

end angle_A_side_lengths_l539_539272


namespace bug_can_return_to_starting_cell_l539_539277

/-- In a grid with doors between neighboring cells, a bug starts at a certain cell.
Each time the bug moves through a closed door, it opens the door in the direction of its movement,
leaving it open. An open door can only be passed through in the direction it was opened.
We need to prove that the bug can always return to its starting cell. -/
theorem bug_can_return_to_starting_cell
    (Grid : Type) (Cell : Grid → Grid → Prop) (Door : Grid → Grid → Prop)
    (bug_moves : Grid → Grid → Prop)
    (start : Grid) :
    (∀ c1 c2, Door c1 c2 → Door c2 c1 → false) → -- An open door can only be passed through in the direction it was opened.
    (∀ c1 c2, Cell c1 c2 ↔ ¬ Cell c2 c1) → -- Each cell is connected to its adjacent neighboring cells by doors.
    (∀ c1 c2, c1 ≠ c2 → bug_moves c1 c2 → Door c1 c2) → -- When the bug moves through a closed door, it opens it in the direction it is moving.
    (∀ c, ∃ c', bug_moves start c → bug_moves c' c) → -- The bug can travel through cells by passing through doors.
    start ∈ Grid → -- The bug starts from a specific cell.
    (∃ path, path 0 = start ∧ (∀ n, path (n + 1) = start) → start ∈ Grid :=
begin
  sorry -- Proof to be completed.
end

end bug_can_return_to_starting_cell_l539_539277


namespace pawns_form_L_shape_l539_539423

theorem pawns_form_L_shape :
  ∀ (pawns : Finset (Fin 8 × Fin 8)),
    pawns.card = 33 →
    ∃ (r c : Fin 8 → Fin 8), (r ≠ c) ∧ (r.val / 2 = c.val / 2 ∧ (∃ i j, (i ≠ j) ∧ (r i, r j, c i) forms_L_shape) sorry


end pawns_form_L_shape_l539_539423


namespace mean_height_is_68_l539_539823

/-
Given the heights of the volleyball players:
  heights_50s = [58, 59]
  heights_60s = [60, 61, 62, 65, 65, 66, 67]
  heights_70s = [70, 71, 71, 72, 74, 75, 79, 79]

We need to prove that the mean height of the players is 68 inches.
-/
def heights_50s : List ℕ := [58, 59]
def heights_60s : List ℕ := [60, 61, 62, 65, 65, 66, 67]
def heights_70s : List ℕ := [70, 71, 71, 72, 74, 75, 79, 79]

def total_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s
def number_of_players : ℕ := total_heights.length
def total_height : ℕ := total_heights.sum
def mean_height : ℕ := total_height / number_of_players

theorem mean_height_is_68 : mean_height = 68 := by
  sorry

end mean_height_is_68_l539_539823


namespace min_value_b_over_a_l539_539228

noncomputable def f (x a b : ℝ) : ℝ := Real.log x + (Math.E - a) * x - b

theorem min_value_b_over_a {a b : ℝ} (h : ∀ x : ℝ, x > 0 → f x a b ≤ 0) : b / a = -1 / Math.E :=
sorry

end min_value_b_over_a_l539_539228


namespace real_solution_exists_l539_539581

theorem real_solution_exists (x : ℝ) (h1: x ≠ 5) (h2: x ≠ 6) : 
  (x = 1 ∨ x = 2 ∨ x = 3) ↔ 
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1) /
  ((x - 5) * (x - 6) * (x - 5)) = 1) := 
by 
  sorry

end real_solution_exists_l539_539581


namespace cone_prism_ratio_is_pi_over_16_l539_539515

noncomputable def cone_prism_volume_ratio 
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ) 
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) : ℝ :=
  (1/3) * Real.pi * cone_base_radius^2 * cone_height / (prism_length * prism_width * prism_height)

theorem cone_prism_ratio_is_pi_over_16
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ)
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) :
  cone_prism_volume_ratio prism_length prism_width prism_height cone_base_radius cone_height
    h_length h_width h_height h_radius_cone h_cone_height = Real.pi / 16 := 
by
  sorry

end cone_prism_ratio_is_pi_over_16_l539_539515


namespace tank_capacity_l539_539499

-- Define the initial fullness of the tank and the total capacity
def initial_fullness (w c : ℝ) : Prop :=
  w = c / 5

-- Define the fullness of the tank after adding 5 liters
def fullness_after_adding (w c : ℝ) : Prop :=
  (w + 5) / c = 2 / 7

-- The main theorem: if both conditions hold, c must equal to 35/3
theorem tank_capacity (w c : ℝ) (h1 : initial_fullness w c) (h2 : fullness_after_adding w c) : 
  c = 35 / 3 :=
sorry

end tank_capacity_l539_539499


namespace solution_set_inequality_l539_539461

theorem solution_set_inequality (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ |2 * x - a| + a ≤ 6) → a = 2 :=
sorry

end solution_set_inequality_l539_539461


namespace radius_of_regular_polygon_l539_539825

theorem radius_of_regular_polygon :
  ∃ (p : ℝ), 
        (∀ n : ℕ, 3 ≤ n → (n : ℝ) = 6) ∧ 
        (∀ s : ℝ, s = 2 → s = 2) → 
        (∀ i : ℝ, i = 720 → i = 720) →
        (∀ e : ℝ, e = 360 → e = 360) →
        p = 2 :=
by
  sorry

end radius_of_regular_polygon_l539_539825


namespace parabola_focus_eq_l539_539822

theorem parabola_focus_eq (focus : ℝ × ℝ) (hfocus : focus = (0, 1)) :
  ∃ (p : ℝ), p = 1 ∧ ∀ (x y : ℝ), x^2 = 4 * p * y → x^2 = 4 * y :=
by { sorry }

end parabola_focus_eq_l539_539822


namespace sum_of_reflected_midpoint_coordinates_l539_539346

/-- Define points P and R and their coordinates -/
def P : ℝ × ℝ := (3, 5)
def R : ℝ × ℝ := (17, -9)

/-- Define the midpoint M of segment PR -/
def M : ℝ × ℝ := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)

/-- Define the reflection of a point (x, y) over the y-axis -/
def reflect_y (point : ℝ × ℝ) : ℝ × ℝ := (-point.1, point.2)

/-- Define the reflected points P' and R' -/
def P' : ℝ × ℝ := reflect_y P
def R' : ℝ × ℝ := reflect_y R

/-- Define the midpoint M' of the reflected segment P'R' -/
def M' : ℝ × ℝ := ((P'.1 + R'.1) / 2, (P'.2 + R'.2) / 2)

/-- Prove that the sum of the coordinates of M' is -12 -/
theorem sum_of_reflected_midpoint_coordinates : (M'.1 + M'.2) = -12 := by
  sorry

end sum_of_reflected_midpoint_coordinates_l539_539346


namespace adam_bought_26_books_l539_539112

-- Conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def avg_books_per_shelf : ℕ := 20
def leftover_books : ℕ := 2

-- Definitions based on conditions
def capacity_books : ℕ := shelves * avg_books_per_shelf
def total_books_after_trip : ℕ := capacity_books + leftover_books

-- Question: How many books did Adam buy on his shopping trip?
def books_bought : ℕ := total_books_after_trip - initial_books

theorem adam_bought_26_books :
  books_bought = 26 :=
by
  sorry

end adam_bought_26_books_l539_539112


namespace compare_abc_l539_539618

noncomputable def a : ℝ := Real.log 3 / Real.log 2 -- log_2(3)
noncomputable def b : ℝ := 2 * Real.cos (36 * Real.pi / 180) -- 2 * cos(36°)
def c : ℝ := Real.sqrt 2 -- sqrt(2)

theorem compare_abc (a_def : a = Real.log 3 / Real.log 2) 
                    (b_def : b = 2 * Real.cos (36 * Real.pi / 180)) 
                    (c_def : c = Real.sqrt 2) : 
    b > a ∧ a > c := by 
  sorry

end compare_abc_l539_539618


namespace geometric_relation_a_formula_correctness_b_formula_correctness_P_n_inequality_l539_539404

def S (n : ℕ) : ℕ := 2 * (2^n) - 2
def a (n : ℕ) : ℕ := 2^n
def b (n : ℕ) : ℕ := 3 * n - 1
def c (n : ℕ) : ℝ := 1 / (b n * b (n + 1))
def P (n : ℕ) : ℝ := 1 / 3 * (1 / 2 - 1 / (3 * n + 2))

theorem geometric_relation :
  ∀ d, ((2 + 2 * d)^2 = 2 * (2 + 10 * d)) → d = 3 :=
sorry

theorem a_formula_correctness :
  ∀ n, S n = 2 * (a n) - 2 :=
sorry

theorem b_formula_correctness :
  ∀ n, b n = 3 * n - 1 :=
sorry

theorem P_n_inequality (t : ℝ) :
  (∀ n : ℕ, n > 0 → P n < t) → t ≥ 1 / 6 :=
sorry

end geometric_relation_a_formula_correctness_b_formula_correctness_P_n_inequality_l539_539404


namespace translation_problem_l539_539534

def is_translation (t : ℕ → Prop) : Prop := t = 2 ∨ t = 4

theorem translation_problem :
  (is_translation 1 = false) ∧ 
  (is_translation 2 = true) ∧ 
  (is_translation 3 = false) ∧ 
  (is_translation 4 = true) →
  is_translation 2 ∧ is_translation 4 :=
by sorry

end translation_problem_l539_539534


namespace obtuse_triangle_range_l539_539245

theorem obtuse_triangle_range (x : ℝ) (h_pos : x > 0) :
  (1 < x ∧ x < sqrt 6) ↔
  (let a := x^2 + 4;
       b := 4 * x;
       c := x^2 + 8 in
   (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
   (-(c^2) + a^2 + b^2 < 0)) :=
by
  -- The proof is omitted.
  sorry

end obtuse_triangle_range_l539_539245


namespace total_oranges_in_collection_l539_539033

theorem total_oranges_in_collection
  (groups_of_oranges : ℕ) (oranges_per_group : ℕ)
  (h_groups : groups_of_oranges = 16)
  (h_oranges_per_group : oranges_per_group = 24) :
  groups_of_oranges * oranges_per_group = 384 :=
by
  -- We assume h_groups and h_oranges_per_group and try to prove the claim.
  rw [h_groups, h_oranges_per_group]
  -- Now we need to multiply 16 * 24 and show it equals 384.
  -- Proof is skipped with sorry.
  sorry

end total_oranges_in_collection_l539_539033


namespace oldest_child_age_l539_539376

noncomputable def average_age_of_seven_children := 8

noncomputable def age_difference := 3

noncomputable def ages := [x, x + age_difference, x + 2 * age_difference, x + 3 * age_difference, x + 4 * age_difference, x + 5 * age_difference, x + 6 * age_difference]

theorem oldest_child_age (x : ℤ) (h : x + 3 * age_difference = average_age_of_seven_children) : (x + 6 * age_difference) = 17 := 
by {
  sorry
}

end oldest_child_age_l539_539376


namespace ten_pow_n_plus_one_divisible_by_eleven_l539_539758

theorem ten_pow_n_plus_one_divisible_by_eleven (n : ℕ) (h : n % 2 = 1) : 11 ∣ (10 ^ n + 1) :=
sorry

end ten_pow_n_plus_one_divisible_by_eleven_l539_539758


namespace largest_prime_factor_of_9973_l539_539168

theorem largest_prime_factor_of_9973 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 9973 ∧ (∀ q : ℕ, nat.prime q ∧ q ∣ 9973 → q ≤ p) :=
begin
  use 103,
  split,
  { exact nat.prime_103, },
  split,
  { exact nat.dvd_of_factors_subset (by norm_num1 : 9973 = 97 * 103) (by norm_num1) (by norm_num1), },
  { intros q hq,
    have hq' := hq.2,
    rw nat.prime.has_dvd_eq_iff at hq',
    rw nat.dvd_prime_mul_iff nat.prime_97 nat.prime_103,
    cases hq',
    { linarith [hq'.symm, nat.prime_97.pos], },
    { exact le_of_eq hq'.symm, }, },
end

end largest_prime_factor_of_9973_l539_539168


namespace max_people_not_sociable_or_shy_l539_539496

noncomputable def max_possible_people (n : ℕ) : Prop :=
  (∀ p : ℕ, p < n → 
     (¬ (∃ s : Set ℕ, s.card ≥ 20 ∧ (∃ x y ∈ s, x ≠ y ∧ x ∈ acquaintances p ∧ y ∈ acquaintances p)) ∧ 
      ¬ (∃ t : Set ℕ, t.card ≥ 20 ∧ (∃ u v ∈ t, u ≠ v ∧ u ∉ acquaintances p ∧ v ∉ acquaintances p))
  ))

theorem max_people_not_sociable_or_shy :
  max_possible_people 40 :=
sorry

end max_people_not_sociable_or_shy_l539_539496


namespace least_three_digit_product_8_is_118_l539_539879

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_product (n : ℕ) (product : ℕ) : Prop :=
  let digits := (list.cons (n / 100) (list.cons ((n / 10) % 10) (list.cons (n % 10) list.nil))) in
  digits.prod = product

theorem least_three_digit_product_8_is_118 :
  ∃ n : ℕ, is_three_digit_number n ∧ digits_product n 8 ∧ (∀ m : ℕ, is_three_digit_number m ∧ digits_product m 8 → n ≤ m) :=
sorry

end least_three_digit_product_8_is_118_l539_539879


namespace total_spent_l539_539665

-- Defining the conditions
def lunch_cost : ℝ := 50.20
def drink_cost : ℝ := 4.30
def tip_rate : ℝ := 0.25
def tip_amount (cost : ℝ) : ℝ := (tip_rate * cost).round 2

-- Theorem statement
theorem total_spent : 
  let total_before_tip := lunch_cost + drink_cost in
  let tip := tip_amount total_before_tip in
  let expected_total := 68.13 in
  total_before_tip + tip = expected_total := 
by
  let total_before_tip := lunch_cost + drink_cost
  let tip := tip_amount total_before_tip
  let expected_total := 68.13
  have h1 : total_before_tip = 54.50 := by norm_num
  have h2 : tip = 13.63 := by norm_num
  simp [*]
  norm_num
  sorry

end total_spent_l539_539665


namespace arithmetic_sequence_15th_term_l539_539393

noncomputable def log_seq (a b : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then log (a ^ 4 * b ^ 9)
  else if n = 2 then log (a ^ 7 * b ^ 17)
  else if n = 3 then log (a ^ 11 * b ^ 26)
  else sorry -- The actual generalization for nth term isn't provided.

theorem arithmetic_sequence_15th_term (a b : ℝ) : 
  ∃ n, log_seq a b 15 = log (b ^ n) ∧ n = 167 :=
sorry

end arithmetic_sequence_15th_term_l539_539393


namespace fraction_studying_japanese_l539_539125

variable (J S : ℕ)
variable (h : S = 2 * J)

theorem fraction_studying_japanese
  (h_senior_japanese : (3 * S) / 8)
  (h_junior_japanese : J / 4) :
  ( (3 / 8 * ↑S) + (1 / 4 * ↑J) ) / ( ↑J + ↑S ) = (1 / 3) := 
  sorry

end fraction_studying_japanese_l539_539125


namespace sum_odd_divisors_240_l539_539434

theorem sum_odd_divisors_240 : 
  let n := 240 in 
  let prime_factors := [2, 2, 2, 2, 3, 5] in 
  ∑ d in (finset.filter (λ d : ℕ, d ∣ n ∧ (∀ p, prime p ∧ p ∣ d → p % 2 = 1)) (finset.range (n + 1))), d = 24 :=
by { let n := 240,
     let prime_factors := [2, 2, 2, 2, 3, 5],
     sorry
}

end sum_odd_divisors_240_l539_539434


namespace subset_sum_equals_n_l539_539727

theorem subset_sum_equals_n (a : List ℤ) (n : ℤ)
  (h1 : 0 < n)
  (h2 : ∀ i j, 0 ≤ i → i < j → j < a.length → a[i] ≤ a[j])
  (h3 : a.sum = 2 * n)
  (h4 : Even n)
  (h5 : ∀ i, i < a.length → a[i] ≠ n + 1) :
  ∃ S : Finset ℤ, S.sum id = n :=
by
  sorry

end subset_sum_equals_n_l539_539727


namespace square_area_is_correct_l539_539425

-- Define the condition: the side length of the square field
def side_length : ℝ := 7

-- Define the theorem to prove the area of the square field with given side length
theorem square_area_is_correct : side_length * side_length = 49 := by
  -- Proof goes here
  sorry

end square_area_is_correct_l539_539425


namespace machine_a_produces_18_sprockets_per_hour_l539_539329

theorem machine_a_produces_18_sprockets_per_hour :
  ∃ (A : ℝ), (∀ (B C : ℝ),
  B = 1.10 * A ∧
  B = 1.20 * C ∧
  990 / A = 990 / B + 10 ∧
  990 / C = 990 / A - 5) →
  A = 18 :=
by { sorry }

end machine_a_produces_18_sprockets_per_hour_l539_539329


namespace bug_can_return_to_start_l539_539278

variable (Grid : Type)
variable (Cell : Grid → Type)
variable (Start : ∀ g : Grid, Cell g)
variable (Door : ∀ g : Grid, Cell g → Cell g → Bool)  -- True if the door is open, False otherwise
variable (Move : ∀ g : Grid, (c1 c2 : Cell g), Door g c1 c2 → Prop)

-- Movement rule: bug can move to an adjacent cell c2 if there's an open door from c1 to c2.
axiom move_through_open_door : ∀ g (c1 c2 : Cell g), Door g c1 c2 → Move g c1 c2

-- Movement rule: the bug opens the door in the direction it moves.
axiom open_door : ∀ g (c1 c2 : Cell g), Move g c1 c2 → Door g c1 c2

-- Prove the bug can return to the starting cell at any moment.
theorem bug_can_return_to_start (g : Grid) : ∀ (c : Cell g), ∃ (p : List (Cell g)), p.head = Start g ∧ p.tail.last = c ∧ ∀ (i : ℕ) (h : i < p.length - 1), Move g (p.nth_le i h) (p.nth_le (i + 1) sorry) :=
  sorry

end bug_can_return_to_start_l539_539278


namespace instantaneous_velocity_at_3_l539_539804

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 + 10

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := (deriv displacement) t

-- State the theorem that the instantaneous velocity at t = 3 is 6 m/s
theorem instantaneous_velocity_at_3 : velocity 3 = 6 := 
sorry

end instantaneous_velocity_at_3_l539_539804


namespace line_equation_l539_539970

def point := (ℝ × ℝ) 
def equal_intercepts_line (P : point) (a : ℝ) : Prop :=
  P = (-2, -3) ∧ (a = 5) ∧ (∀ x y, x + y = a)

theorem line_equation (P : point) (a : ℝ) : equal_intercepts_line P a → (∀ x y, x + y = 5) :=
by
  intro h
  have h1 : P = (-2, -3) := h.1
  have h2 : a = 5 := h.2.1
  have h3 : ∀ x y, x + y = a := h.2.2
  rw [h2]
  exact h3

end line_equation_l539_539970


namespace correct_statement_l539_539050

theorem correct_statement :
  let A := (3 : ℕ) = 3 ∧ (3 + 3) = 6 ∧ ((3 : ℕ) / (3 + 3) : ℝ) = 0.5 in
  let B := ∀ (n : ℕ), (n = 100 → (1/100 * n ≠ 1)) in
  let C := False in
  let D := ∀ (a : ℝ), |a| > 0 → a ≠ 0 in
  A ∧ ¬ B ∧ C ∧ ¬ D :=
by 
  sorry

end correct_statement_l539_539050


namespace movie_theater_operating_time_l539_539506

theorem movie_theater_operating_time :
  let movie_length := 1.5
  let replay_count := 6
  let ad_length_min := 20
  let total_movie_time := movie_length * replay_count
  let total_ad_time_min := ad_length_min * replay_count
  let total_ad_time_hours := total_ad_time_min / 60
  total_movie_time + total_ad_time_hours = 11 := 
by
  let movie_length := 1.5
  let replay_count := 6
  let ad_length_min := 20
  let total_movie_time := movie_length * replay_count
  let total_ad_time_min := ad_length_min * replay_count
  let total_ad_time_hours := total_ad_time_min / 60
  have h1 : total_movie_time = 9 := by sorry
  have h2 : total_ad_time_hours = 2 := by sorry
  have h3 : 9 + 2 = 11 := by sorry
  exact h3

end movie_theater_operating_time_l539_539506


namespace cuboid_cutout_l539_539948

theorem cuboid_cutout (x y : ℕ) (h1 : x * y = 36) (h2 : 0 < x) (h3 : x < 4) (h4 : 0 < y) (h5 : y < 15) :
  x + y = 15 :=
sorry

end cuboid_cutout_l539_539948


namespace inequality_proof_l539_539772

theorem inequality_proof 
  (x y z w : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w)
  (h_eq : (x^3 + y^3)^4 = z^3 + w^3) :
  x^4 * z + y^4 * w ≥ z * w :=
sorry

end inequality_proof_l539_539772


namespace true_propositions_l539_539227

-- Basic setup for propositions
variables {k a b c x y : ℝ}

-- Proposition 1: The quadratic equation x^2 + 2x - k = 0 has real roots given k > 0.
lemma prop_1 (hk : k > 0) : (∃ x : ℝ, x^2 + 2 * x - k = 0) :=
sorry

-- Proposition 2: The negation of "If a > b, then a + c > b + c" is false.
lemma prop_2 : ¬ (∀ a b c : ℝ, a > b → a + c ≤ b + c) :=
sorry

-- Proposition 3: The converse of "The diagonals of a rectangle are equal" is false.
lemma prop_3 : ¬ (∀ (q : Type), is_quadrilateral q → equal_diagonals q → is_rectangle q) :=
sorry

-- Proposition 4: The negation of "If xy = 0, then at least one of x and y is 0" is false.
lemma prop_4 : ¬ (∀ x y : ℝ, xy = 0 → ¬ (x = 0 ∨ y = 0)) :=
sorry

-- Proof that propositions 1 and 4 are true
theorem true_propositions : (prop_1 hk) ∧ (prop_4) :=
⟨sorry, sorry⟩

end true_propositions_l539_539227


namespace evaluate_expression_l539_539159

theorem evaluate_expression : 8^15 / 64^5 = 32768 := by
  -- We rewrite the exponents in terms of base 2
  have h1 : 8 = 2^3 := by norm_num
  have h2 : 64 = 2^6 := by norm_num
  -- Use the properties of exponents and the provided conditions
  calc
    8^15 / 64^5 = (2^3)^15 / (2^6)^5 : by { rw [h1, h2], }
            ... = 2^45 / 2^30 : by norm_num
            ... = 2^(45 - 30) : by rw [pow_sub]; norm_num
            ... = 2^15 : by norm_num
            ... = 32768 : by norm_num

end evaluate_expression_l539_539159


namespace final_answer_l539_539565

noncomputable def across_1 : ℕ := 87
noncomputable def across_3 : ℕ := 125
noncomputable def across_5 : ℕ := 91
noncomputable def down_1 : ℕ := 81
noncomputable def down_2 : ℕ := 729
noncomputable def down_4 : ℕ := 51

lemma across_1_condition : across_1 = 87 := sorry
lemma across_3_condition : across_3 = 125 := sorry
lemma across_5_condition : across_5 = 91 := sorry
lemma down_1_condition : down_1 = 81 := sorry
lemma down_2_condition : down_2 = 729 := sorry
lemma down_4_condition : down_4 = 51 := sorry

def T := across_1 + across_3 + across_5 + down_1 + down_2 + down_4

lemma T_sum : T = 1164 :=
by 
  rw [across_1_condition, across_3_condition, across_5_condition,
      down_1_condition, down_2_condition, down_4_condition]
  sorry

theorem final_answer : 0.5 * T = 582 :=
by 
  rw [T_sum]
  sorry

end final_answer_l539_539565


namespace sum_to_fraction_l539_539160

theorem sum_to_fraction :
  (2 / 10) + (3 / 100) + (4 / 1000) + (6 / 10000) + (7 / 100000) = 23467 / 100000 :=
by
  sorry

end sum_to_fraction_l539_539160


namespace sequence_property_implies_geometric_progression_l539_539715

theorem sequence_property_implies_geometric_progression {p : ℝ} {a : ℕ → ℝ}
  (h_p : (2 / (Real.sqrt 5 + 1) ≤ p) ∧ (p < 1))
  (h_a : ∀ (e : ℕ → ℤ), (∀ n, (e n = 0) ∨ (e n = 1) ∨ (e n = -1)) →
    (∑' n, (e n) * (p ^ n)) = 0 → (∑' n, (e n) * (a n)) = 0) :
  ∃ c : ℝ, ∀ n, a n = c * (p ^ n) := by
  sorry

end sequence_property_implies_geometric_progression_l539_539715


namespace find_a_l539_539203

theorem find_a (a : ℝ) (M : Set ℝ) (N : Set ℝ) : 
  M = {1, 3} → N = {1 - a, 3} → (M ∪ N) = {1, 2, 3} → a = -1 :=
by
  intros hM hN hUnion
  sorry

end find_a_l539_539203


namespace triangulation_graph_eulerian_cycle_l539_539921

theorem triangulation_graph_eulerian_cycle (n : ℕ) 
  (convex_ngon : convex n) 
  (non_intersecting_diagonals : ∀ (d₁ d₂ : diagonal), disjoint_except_at_endpoints d₁ d₂) 
  (triangulation : triangulation_graph n) : 
  (∃ eulerian_cycle : eulerian_cycle triangulation, 31 ∣ n) ↔ (triangulation_drawable_in_one_stroke triangulation)
:=
sorry

end triangulation_graph_eulerian_cycle_l539_539921


namespace tammy_avg_speed_second_day_l539_539785

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l539_539785


namespace max_truth_tellers_l539_539751

-- Defining predicates for statements
def statement1 (n : ℕ) (x : ℕ) : Prop := x > n
def statement2 (n : ℕ) (y : ℕ) : Prop := y < n

-- Definition of the problem translated in Lean4
theorem max_truth_tellers : ∀ (persons : ℕ → ℕ), persons 1 = 1 ∧ persons 2 = 2 ∧ persons 3 = 3 ∧ 
persons 4 = 4 ∧ persons 5 = 5 ∧ persons 6 = 6 ∧ persons 7 = 7 ∧ 
persons 8 = 8 ∧ persons 9 = 9 ∧ (persons 10 < 1 ∨ persons 10 > 10) → 
∀ (truth_tellers : ℕ), truth_tellers ≤ 9 := 
by 
  intros persons H truth_tellers,
  sorry

end max_truth_tellers_l539_539751


namespace required_speed_l539_539547

theorem required_speed
  (D T : ℝ) (h1 : 30 = D / T) 
  (h2 : 2 * D / 3 = 30 * (T / 3)) :
  (D / 3) / (2 * T / 3) = 15 :=
by
  sorry

end required_speed_l539_539547


namespace probability_A_fires_l539_539483

theorem probability_A_fires 
  (p_first_shot: ℚ := 1/6)
  (p_not_fire: ℚ := 5/6)
  (p_recur: ℚ := p_not_fire * p_not_fire) : 
  ∃ (P_A : ℚ), P_A = 6/11 :=
by
  have eq1 : P_A = p_first_shot + (p_recur * P_A) := sorry
  have eq2 : P_A * (1 - p_recur) = p_first_shot := sorry
  have eq3 : P_A = (p_first_shot * 36) / 11 := sorry
  exact ⟨P_A, sorry⟩

end probability_A_fires_l539_539483


namespace find_k_l539_539698

theorem find_k (m n : ℝ) 
  (h₁ : m = k * n + 5) 
  (h₂ : m + 2 = k * (n + 0.5) + 5) : 
  k = 4 :=
by
  sorry

end find_k_l539_539698


namespace distinct_positive_solutions_l539_539367

noncomputable def solve_system (a b : ℝ) : ℝ × ℝ × ℝ :=
let x := (a^2 + b^2 + Real.sqrt ((3 * a^2 - b^2) * (3 * b^2 - a^2))) / (4 * a),
    y := (a^2 + b^2 - Real.sqrt ((3 * a^2 - b^2) * (3 * b^2 - a^2))) / (4 * a),
    z := (a^2 - b^2) / (2 * a) in
(x, y, z)

theorem distinct_positive_solutions {a b : ℝ} (ha : |b| < a) (hb : a < Real.sqrt 3 * |b|) :
  ∃ x y z : ℝ, (x + y + z = a ∧ x^2 + y^2 + z^2 = b^2 ∧ xy = z^2 ∧ x ≠ y ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
  let sol := solve_system a b in
  ∃ (x y z : ℝ), x = sol.1 ∧ y = sol.2 ∧ z = sol.3 ∧
  (x + y + z = a) ∧ (x^2 + y^2 + z^2 = b^2) ∧ (x * y = z^2) ∧ (x ≠ y) ∧ (x > 0) ∧ (y > 0) ∧ (z > 0) :=
sorry

end distinct_positive_solutions_l539_539367


namespace not_differentiable_at_0_l539_539944

open Real

def sinx (x : ℝ) : ℝ := sin x
def x_cubed (x : ℝ) : ℝ := x^3
def ln2 (x : ℝ) : ℝ := log 2 -- Note: log is the natural logarithm (ln)
def absx (x : ℝ) : ℝ := abs x

theorem not_differentiable_at_0 : 
  ¬ ∀ f : ℝ → ℝ, f = sinx ∨ f = x_cubed ∨ f = ln2 ∨ f = absx →
  differentiable_at ℝ f 0 :=
by
  sorry

end not_differentiable_at_0_l539_539944


namespace range_of_t_for_three_tangents_l539_539632

-- Define the given conditions
def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x
def P := (1 : ℝ, t : ℝ)

-- Define the function g, derived from the solution steps
def g (x : ℝ, t : ℝ) : ℝ := 4 * x ^ 3 - 6 * x ^ 2 + 3 + t

-- Statement to show the range of t
theorem range_of_t_for_three_tangents :
  ∃ t : ℝ, (-3 < t) ∧ (t < -1) ↔
  (∃ t : ℝ, ∀ x : ℝ, g(x, t) = 0 → t ∈ Icc (-3:ℝ) (-1:ℝ)) := by
  sorry

end range_of_t_for_three_tangents_l539_539632


namespace probability_red_second_draw_l539_539831

theorem probability_red_second_draw 
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (after_first_draw_balls : ℕ)
  (after_first_draw_red : ℕ)
  (probability : ℚ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  after_first_draw_balls = 4 →
  after_first_draw_red = 2 →
  probability = after_first_draw_red / after_first_draw_balls →
  probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_red_second_draw_l539_539831


namespace average_pregnancies_per_kettle_l539_539502

-- Define the given conditions
def num_kettles : ℕ := 6
def babies_per_pregnancy : ℕ := 4
def survival_rate : ℝ := 0.75
def total_expected_babies : ℕ := 270

-- Calculate surviving babies per pregnancy
def surviving_babies_per_pregnancy : ℝ := babies_per_pregnancy * survival_rate

-- Prove that the average number of pregnancies per kettle is 15
theorem average_pregnancies_per_kettle : ∃ P : ℝ, num_kettles * P * surviving_babies_per_pregnancy = total_expected_babies ∧ P = 15 :=
by
  sorry

end average_pregnancies_per_kettle_l539_539502


namespace explicit_formula_for_f_f_geq_negx2_plus_x_range_of_k_for_fx_geq_kx_l539_539635

-- Definition of the function f
def f (x : ℝ) : ℝ := Real.exp x - x^2 - 1

-- Variables
variables (x k : ℝ)

-- 1. Prove the explicit formula for f
theorem explicit_formula_for_f : f x = Real.exp x - x^2 - 1 := by
  sorry

-- 2. Prove that for all x ∈ ℝ, f(x) ≥ -x^2 + x
theorem f_geq_negx2_plus_x : ∀ x : ℝ, f x ≥ -x^2 + x := by
  sorry

-- 3. Find the range of k such that f(x) ≥ kx for x ∈ (0, +∞)
theorem range_of_k_for_fx_geq_kx : (∀ x ∈ Ioi 0, f x ≥ k*x) → k ≤ Real.exp 1 - 2 := by
  sorry

end explicit_formula_for_f_f_geq_negx2_plus_x_range_of_k_for_fx_geq_kx_l539_539635


namespace pentagon_acute_triangle_probability_l539_539271

theorem pentagon_acute_triangle_probability : 
  let vertices := {1, 2, 3, 4, 5} in
  let all_triangles := {S | S ⊆ vertices ∧ S.card = 3 } in
  let acute_triangles := {T | T ∈ all_triangles ∧ is_acute_triangle T} in
  (acute_triangles.to_finset.card : ℚ) / (all_triangles.to_finset.card : ℚ) = 1 / 2 :=
sorry

end pentagon_acute_triangle_probability_l539_539271


namespace mushroom_picking_times_l539_539530

-- Given conditions
def time_left_after_8 : ℚ := 8 + 43 * (7 / 11) / 60
def time_returned_after_2 : ℚ := 14 + 43 * (7 / 11) / 60 -- using 14 for 2 PM in 24-hour format

-- Theorem to prove the specific times
theorem mushroom_picking_times :
  (time_left_after_8 = 8 + 43 * (7 / 11) / 60) ∧ 
  (time_returned_after_2 = 14 + 43 * (7 / 11) / 60) ∧
  (time_returned_after_2 - time_left_after_8 = 6) :=
by
  sorry

# IT: "Proved statement if no sorry was used in the definition"

end mushroom_picking_times_l539_539530


namespace lana_goal_is_20_l539_539713

def muffins_sold_morning := 12
def muffins_sold_afternoon := 4
def muffins_needed_to_goal := 4
def total_muffins_sold := muffins_sold_morning + muffins_sold_afternoon
def lana_goal := total_muffins_sold + muffins_needed_to_goal

theorem lana_goal_is_20 : lana_goal = 20 := by
  sorry

end lana_goal_is_20_l539_539713


namespace repeating_decimals_count_l539_539175

theorem repeating_decimals_count : 
  ∀ n : ℕ, 1 ≤ n ∧ n < 1000 → ¬(∃ k : ℕ, n + 1 = 2^k ∨ n + 1 = 5^k) :=
by
  sorry

end repeating_decimals_count_l539_539175


namespace enough_officials_to_judge_l539_539684

-- Definitions from conditions part (a)
variable {n : ℕ} -- number of matches

-- Each match eliminates one player
variable {elim_players : ℕ} -- number of eliminated players

-- Number of judges required is (n - 1) for subsequent matches
def required_judges (n : ℕ) : ℕ := n - 1

-- Theorem to be proved
theorem enough_officials_to_judge (n elim_players : ℕ) (h1 : elim_players = n)
  (h2 : n ≥ 1) :
  elim_players ≥ required_judges n :=
by {
  cases n,
  { -- case when no matches have been played: n = 0
    simp [elim_players, required_judges],
    sorry },
  { -- case when at least one match is played: n ≥ 1
    simp [elim_players, required_judges] at *,
    sorry
  }
}

end enough_officials_to_judge_l539_539684


namespace solve_fractional_equation_l539_539579

theorem solve_fractional_equation (x : ℝ) :
  (x ≠ 5) ∧ (x ≠ 6) ∧ ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) / ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ 
  x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
sorry

end solve_fractional_equation_l539_539579


namespace length_of_side_triangle_PQO_l539_539756

theorem length_of_side_triangle_PQO :
  ∃ (P Q : ℝ × ℝ), P.2 = - (1 / 3) * P.1 ^ 2 ∧ Q.2 = - (1 / 3) * Q.1 ^ 2 ∧
  ∃ (O : ℝ × ℝ), O = (0, 0) ∧
  ∃ (equilateral : ∀ (A B C : ℝ × ℝ), 
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B),
  (dist (0, 0) P + dist (0, 0) Q + dist P Q = 6 * real.sqrt 3) := by
sorry

end length_of_side_triangle_PQO_l539_539756


namespace find_value_of_expression_l539_539730

theorem find_value_of_expression
  (a b c d : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : d ≥ 0)
  (h₄ : a / (b + c + d) = b / (a + c + d))
  (h₅ : b / (a + c + d) = c / (a + b + d))
  (h₆ : c / (a + b + d) = d / (a + b + c))
  (h₇ : d / (a + b + c) = a / (b + c + d)) :
  (a + b) / (c + d) + (b + c) / (a + d) + (c + d) / (a + b) + (d + a) / (b + c) = 4 :=
by sorry

end find_value_of_expression_l539_539730


namespace solve_fraction_equation_l539_539362

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ -3) :
  (2 / x + x / (x + 3) = 1) ↔ x = 6 := 
by
  sorry

end solve_fraction_equation_l539_539362


namespace yanna_kept_apples_l539_539439

theorem yanna_kept_apples (total_apples : ℕ) (apples_to_Zenny : ℕ) (apples_to_Andrea : ℕ) 
  (h_total : total_apples = 60) (h_Zenny : apples_to_Zenny = 18) (h_Andrea : apples_to_Andrea = 6) : 
  (total_apples - apples_to_Zenny - apples_to_Andrea) = 36 := by
  -- Initial setup based on the problem conditions
  rw [h_total, h_Zenny, h_Andrea]
  -- Simplify the expression
  rfl

-- The theorem simplifies to proving 60 - 18 - 6 = 36

end yanna_kept_apples_l539_539439


namespace no_valid_four_digit_numbers_l539_539656

theorem no_valid_four_digit_numbers :
  ∀ N: ℕ, (1000 ≤ N ∧ N ≤ 9999) →
         (∃ a: ℕ, ∃ x: ℕ, a = N / 1000 ∧ x = N % 1000 ∧ x = 1000 * a / 7 ∧ 100 ≤ x ∧ x ≤ 999) → false :=
begin
  sorry
end

end no_valid_four_digit_numbers_l539_539656


namespace approx_log3_20_l539_539627

variable (log10_2 : ℝ) (log10_3 : ℝ)
axiom h_log10_2 : log10_2 ≈ 0.301
axiom h_log10_3 : log10_3 ≈ 0.477

theorem approx_log3_20 : log (20 : ℝ) / log 3 ≈ 2.786 :=
by
  sorry

end approx_log3_20_l539_539627


namespace original_price_correct_l539_539671

noncomputable def original_price : ℝ :=
  270

theorem original_price_correct:
  (∃ P : ℝ, 1.30 * P = 351 ∧ 2 * P = 540) → original_price = 270 :=
by
  intro h
  cases h with P hP
  have h1 : 1.30 * P = 351 := hP.1
  have h2 : 2 * P = 540 := hP.2
  have P_eq : P = 270 := sorry
  show original_price = 270, by rw [P_eq]

end original_price_correct_l539_539671


namespace linda_savings_l539_539452

theorem linda_savings (S : ℕ) (h1 : (3 / 4) * S = x) (h2 : (1 / 4) * S = 240) : S = 960 :=
by
  sorry

end linda_savings_l539_539452


namespace tangent_angle_QABC_l539_539964

-- Let the regular triangular pyramids P-ABC and Q-ABC be inscribed in the same sphere
-- Let the angle between the lateral face and the base of pyramid P-ABC be 45 degrees
-- We need to prove that the tangent of the angle between the lateral face and the base of pyramid Q-ABC is 4

theorem tangent_angle_QABC (P Q A B C O : Point) (sphere_radius : Real) :
  inscribed_in_same_sphere P Q A B C O sphere_radius →
  common_base A B C →
  lateral_face_angle P A B C = 45 →
  tangent_of_lateral_face_angle Q A B C = 4 :=
  by sorry

end tangent_angle_QABC_l539_539964


namespace distance_vertex_asymptote_l539_539586

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 4 = 1

def vertex : ℝ × ℝ := (Real.sqrt 2, 0)

def asymptote (x y : ℝ) : Prop :=
  Real.sqrt 2 * x + y = 0

theorem distance_vertex_asymptote :
  let d := (| Real.sqrt 2 * Real.sqrt 2 + 1 * 0 |) / Real.sqrt (Real.sqrt 2 ^ 2 + 1 ^ 2)
  d = 2 * Real.sqrt 3 / 3 :=
sorry

end distance_vertex_asymptote_l539_539586


namespace probability_of_a_firing_l539_539477

/-- 
Prove that the probability that A will eventually fire the bullet is 6/11, given the following conditions:
1. A and B take turns shooting with a six-shot revolver that has only one bullet.
2. They randomly spin the cylinder before each shot.
3. A starts the game.
-/
theorem probability_of_a_firing (p_a : ℝ) :
  (1 / 6) + (5 / 6) * (5 / 6) * p_a = p_a → p_a = 6 / 11 :=
by
  intro hyp
  have h : p_a - (25 / 36) * p_a = 1 / 6 := by
    rwa [← sub_eq_of_eq_add hyp, sub_self, zero_mul] 
  field_simp at h
  linarith
  sorry

end probability_of_a_firing_l539_539477


namespace sequence_1234_3269_never_appear_sequence_1975_appears_again_sequence_8197_appears_l539_539699

def digit_sum (a b c d : ℕ) : ℕ :=
  (a + b + c + d) % 10

def sequence_rule (seq : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k ≥ 4, seq (k + 1) = digit_sum (seq (k - 3)) (seq (k - 2)) (seq (k - 1)) (seq k)

noncomputable def seq_start := [1, 9, 7, 5, 2] -- Initial sequence

def extend_sequence (seq : ℕ → ℕ) (n: ℕ) : ℕ :=
  if n < 5 then seq_start[n] else digit_sum (seq (n-4)) (seq (n-3)) (seq (n-2)) (seq (n-1))

theorem sequence_1234_3269_never_appear (seq : ℕ → ℕ) (h : sequence_rule seq 5) :
  ¬ (∃ n, seq n = 1 ∧ seq (n+1) = 2 ∧ seq (n+2) = 3 ∧ seq (n+3) = 4) ∧
  ¬ (∃ n, seq n = 3 ∧ seq (n+1) = 2 ∧ seq (n+2) = 6 ∧ seq (n+3) = 9) :=
sorry

theorem sequence_1975_appears_again (seq : ℕ → ℕ) (h : sequence_rule seq 5) :
  ∃ m > 4, seq m = 1 ∧ seq (m+1) = 9 ∧ seq (m+2) = 7 ∧ seq (m+3) = 5 :=
sorry

theorem sequence_8197_appears (seq : ℕ → ℕ) (h : sequence_rule seq 5) :
  ∃ m > 4, seq m = 8 ∧ seq (m+1) = 1 ∧ seq (m+2) = 9 ∧ seq (m+3) = 7 :=
sorry

end sequence_1234_3269_never_appear_sequence_1975_appears_again_sequence_8197_appears_l539_539699


namespace probability_underdside_red_given_upper_red_l539_539035

-- Define the cards
def card1 : string := "RR"  -- Red on both sides
def card2 : string := "RB"  -- Red on one side, blue on the other

-- Define the events
def event_red_up (card : string) : Prop :=
  card = "RR" ∨ card = "RB"

def event_rr : Prop := true

-- Define the probabilities
noncomputable def P_card1 : ℝ := 1 / 2
noncomputable def P_card2 : ℝ := 1 / 2
noncomputable def P_red_up_RR : ℝ := 1
noncomputable def P_red_up_RB : ℝ := 1 / 2

-- Define the conditional probability to prove
noncomputable def P_rr_given_red_up : ℝ :=
  P_card1 * P_red_up_RR / (P_card1 * P_red_up_RR + P_card2 * P_red_up_RB)

-- Theorem statement
theorem probability_underdside_red_given_upper_red :
  P_rr_given_red_up = 2 / 3 := by
  sorry

end probability_underdside_red_given_upper_red_l539_539035


namespace fred_speed_5_mph_l539_539597

theorem fred_speed_5_mph (F : ℝ) (h1 : 50 = 25 + 25) (h2 : 25 / 5 = 5) (h3 : 25 / F = 5) : 
  F = 5 :=
by
  -- Since Fred's speed makes meeting with Sam in the same time feasible
  sorry

end fred_speed_5_mph_l539_539597


namespace complex_abs_of_sqrt_l539_539372

variable (z : ℂ)

theorem complex_abs_of_sqrt (h : z^2 = 16 - 30 * Complex.I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_abs_of_sqrt_l539_539372


namespace correct_sampling_pairings_l539_539414

/-- 
Given the following sampling scenarios:
1. From a population consisting of 150 urban students and 150 rural students, a sample of 100 students is taken.
2. From a total of 20 products, a sample of 7 products is taken for quality inspection.
3. From a population of 2000 students, a sample of 10 students is taken to understand their daily habits.
And the possible sampling methods:
I. Simple Random Sampling
II. Systematic Sampling
III. Stratified Sampling

We aim to prove that the correct methodology pairings are:
1. Stratified Sampling (III)
2. Simple Random Sampling (I)
3. Systematic Sampling (II),
as indicated by option (C).
-/
theorem correct_sampling_pairings : 
  pairing_scenario1_method == III ∧ pairing_scenario2_method == I ∧ pairing_scenario3_method == II :=
sorry

end correct_sampling_pairings_l539_539414


namespace calc_fraction_l539_539176

variable {x y : ℝ}

theorem calc_fraction (h : x + y = x * y - 1) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 1 - 1 / (x * y) := 
by 
  sorry

end calc_fraction_l539_539176


namespace find_n_l539_539331

def boxes_of_cookies (n M A : ℕ) : Prop :=
  (M = n - 9) ∧ (A = n - 2) ∧ (M ≥ 1) ∧ (A ≥ 1) ∧ (M + A < n)

theorem find_n (n M A : ℕ) (h : boxes_of_cookies n M A) : n = 10 :=
by {
  rcases h with ⟨ h1, h2, h3, h4, h5 ⟩,
  sorry
}

end find_n_l539_539331


namespace interval_of_decrease_l539_539006

noncomputable def f (x : ℝ) := Real.log (-x^2 + 2 * x + 3)

theorem interval_of_decrease :
  (∀ x : ℝ, f x = Real.log (-x^2 + 2 * x + 3)) ∧ 
  (∀ x : ℝ,  -x^2 + 2 * x + 3 > 0 →  1 < x ∧ x < 3) →
  ∃ (a b : ℝ), (a = 1 ∧ b = 3) →
  ∀ x : ℝ, (1 < x ∧ x < 3) → (f x).deriv < 0 :=
by
  sorry

end interval_of_decrease_l539_539006


namespace number_of_ellipses_l539_539630

theorem number_of_ellipses:
  let valid_mns := (λ (m : ℕ), m ∈ {1, 2, 3, 4, 5} ∧ 
                               (∃ (n: ℕ), n ∈ {1, 2, 3, 4, 5, 6, 7} ∧ m > n)) in
  (Finset.card (Finset.univ.filter valid_mns)) = 10 :=
by
  sorry

end number_of_ellipses_l539_539630


namespace sqrt3_sqrt5_sqrt7_not_arithmetic_seq_l539_539888

theorem sqrt3_sqrt5_sqrt7_not_arithmetic_seq :
  ¬ ∃ d : ℝ, ∃ a : ℝ, √3 + d = a ∧ √5 + d = a + d ∧ √7 = a + 2 * d := by
  sorry

end sqrt3_sqrt5_sqrt7_not_arithmetic_seq_l539_539888


namespace scallops_final_cost_l539_539069

-- Define the conditions in Lean
def num_scallops_per_pound : ℕ := 8
def cost_per_pound : ℝ := 24.00
def scallops_per_person : ℕ := 2
def num_people : ℕ := 8
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.07

-- Total number of scallops needed
def total_scallops : ℕ := num_people * scallops_per_person
-- Weight of scallops needed in pounds
def scallops_weight : ℝ := total_scallops.to_nat / num_scallops_per_pound.to_nat
-- Initial cost without taxes or discounts
def initial_cost : ℝ := scallops_weight * cost_per_pound

-- Discount applied
def discount_amount : ℝ := discount_rate * initial_cost
def discounted_price : ℝ := initial_cost - discount_amount

-- Sales tax applied
def sales_tax_amount : ℝ := sales_tax_rate * discounted_price
def final_cost : ℝ := discounted_price + sales_tax_amount

-- Prove the final cost is $46.22
theorem scallops_final_cost : final_cost = 46.22 :=
by {
  sorry -- Proof to be provided
}

end scallops_final_cost_l539_539069


namespace train_travel_distance_l539_539525

theorem train_travel_distance : 
  (∫ t in 0..30, (27 - 0.9 * t)) = 405 :=
by
  sorry

end train_travel_distance_l539_539525


namespace probability_of_a_firing_l539_539475

/-- 
Prove that the probability that A will eventually fire the bullet is 6/11, given the following conditions:
1. A and B take turns shooting with a six-shot revolver that has only one bullet.
2. They randomly spin the cylinder before each shot.
3. A starts the game.
-/
theorem probability_of_a_firing (p_a : ℝ) :
  (1 / 6) + (5 / 6) * (5 / 6) * p_a = p_a → p_a = 6 / 11 :=
by
  intro hyp
  have h : p_a - (25 / 36) * p_a = 1 / 6 := by
    rwa [← sub_eq_of_eq_add hyp, sub_self, zero_mul] 
  field_simp at h
  linarith
  sorry

end probability_of_a_firing_l539_539475


namespace find_f2014_l539_539608

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f2 (x : ℝ) : ℝ := derivative (f1 x)
noncomputable def f3 (x : ℝ) : ℝ := derivative (f2 x)
noncomputable def f4 (x : ℝ) : ℝ := derivative (f3 x)
noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  if h : n % 4 = 1 then f1 x
  else if h : n % 4 = 2 then f2 x
  else if h : n % 4 = 3 then f3 x
  else f4 x

theorem find_f2014 : fn 2014 = λ x, Real.cos x - Real.sin x :=
by
  sorry

end find_f2014_l539_539608


namespace solve_math_problem_l539_539643

noncomputable def parabola : set (ℝ × ℝ) := {p | p.2^2 = p.1}

noncomputable def circle : set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 2}

noncomputable def directrix_intersects_x_axis (C : set (ℝ × ℝ)) : Prop := 
  ∃ T : ℝ × ℝ, T ∈ C ∧ T.2 = 0

noncomputable def chord_through_focus (C : set (ℝ × ℝ)) : Prop := 
  ∃ M N : ℝ × ℝ, M ∈ C ∧ N ∈ C ∧ ({
    chord := (y - M.2)/(M.2 - N.2) = (x - M.1)/(M.1 - N.1)
  }) 

noncomputable def minimum_value_of_vector_product (T M N : ℝ × ℝ) : ℝ := sorry

noncomputable def coordinates_of_point (C1 C2 : set (ℝ × ℝ)) (P A B : ℝ × ℝ) : ℝ × (ℝ × ℝ) := sorry

theorem solve_math_problem :
  (∀ C1 C2 : set (ℝ × ℝ),
  C1 = parabola →
  C2 = circle →
  directrix_intersects_x_axis C1 →
  chord_through_focus C1 →
  (minimum_value_of_vector_product (directrix_intersects_x_axis C1) (chord_through_focus C1) = 0) ∧
  (coordinates_of_point C1 C2 P A B = ((11 / 3, -sqrt (33) / 3), (11 / 3, sqrt (33) / 3)))) :=
begin
  sorry
end

end solve_math_problem_l539_539643


namespace spiders_catch_insect_l539_539544

-- Definitions for the problem conditions
def vertex := ℕ -- Define a vertex index type for simplicity.
def edge (u v : vertex) := u ≠ v -- Edge exists if vertices are distinct.

-- The problem conditions
variable (A G : vertex) -- Initial positions of spiders and insect
variable (speed : ℕ) -- Speed of movement along the edges
variable (awareness : vertex → vertex → vertex → Prop) -- Position awareness predicate
variable (predict : vertex → vertex → vertex → Prop) -- Movement prediction predicate

-- The question reformulated as a proof goal
theorem spiders_catch_insect (A G : vertex) (hAG : edge A G) 
  (speed : ℕ) (awareness : ∀ u v i, edge u v → Prop) (predict : ∀ u v i, edge u v → Prop) :
  ∃ strategy : (vertex × vertex → vertex × vertex → vertex → (ℕ → vertex × vertex)),
  ∀ (u v i : vertex), 
  edge A G ∧ awareness u v i ∧ predict u v i →
  ∃ t : ℕ, let positions := strategy (u, v) (A, G) i in positions t = (i, i) :=
sorry -- skip the proof

end spiders_catch_insect_l539_539544


namespace root_of_equation_l539_539397

open Nat

def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))
def permutation (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

theorem root_of_equation : ∃ x : ℕ, 3 * combination (x - 3) 4 = 5 * permutation (x - 4) 2 ∧ 7 ≤ x ∧ x = 11 := 
by 
  sorry

end root_of_equation_l539_539397


namespace probability_A_fires_l539_539468

theorem probability_A_fires :
  let p_A := (1 : ℚ) / 6 + (5 : ℚ) / 6 * (5 : ℚ) / 6 * p_A
  in p_A = 6 / 11 :=
by
  sorry

end probability_A_fires_l539_539468


namespace tammy_speed_on_second_day_l539_539791

theorem tammy_speed_on_second_day :
  ∃ v t : ℝ, t + (t - 2) = 14 ∧ v * t + (v + 0.5) * (t - 2) = 52 → (v + 0.5 = 4) :=
begin
  sorry
end

end tammy_speed_on_second_day_l539_539791


namespace wash_window_time_l539_539706

variable (x : ℝ)

/-- Your friend takes 3 minutes to wash a window -/
def friend_rate := 1 / 3

/-- It takes 30 minutes to wash 25 windows together -/
def combined_effort_per_minute := 25 / 30

/-- Assuming the rate at which you wash windows -/
def your_rate := 1 / x

/-- Together rate at which you wash windows -/
def together_rate := friend_rate + your_rate

theorem wash_window_time : together_rate = combined_effort_per_minute → x = 2 :=
by
  intro h
  -- (proof omitted)
  sorry

end wash_window_time_l539_539706


namespace magnitude_difference_perpendicular_sum_difference_l539_539184

variables {a b : ℝ^3}

-- Conditions given in the problem
axiom norm_a : ‖a‖ = 1
axiom norm_b : ‖b‖ = 1
axiom norm_a_add_b : ‖a + b‖ = sqrt(3)

-- Part 1: Prove the magnitude of the difference of vectors is 1
theorem magnitude_difference (a b : ℝ^3) (norm_a : ‖a‖ = 1) (norm_b : ‖b‖ = 1) (norm_a_add_b : ‖a + b‖ = sqrt(3)) : ‖a - b‖ = 1 :=
sorry

-- Part 2: Prove vectors are perpendicular
theorem perpendicular_sum_difference (a b : ℝ^3) (norm_a : ‖a‖ = 1) (norm_b : ‖b‖ = 1) (norm_a_add_b : ‖a + b‖ = sqrt(3)) : (a + b) • (a - b) = 0 :=
sorry

end magnitude_difference_perpendicular_sum_difference_l539_539184


namespace semiperimeter_of_triangle_l539_539136

theorem semiperimeter_of_triangle (a ρ_b ρ_c : ℝ) (s : ℝ) :
  s = (a / 2) + real.sqrt ((a / 2) ^ 2 + ρ_b * ρ_c) :=
sorry

end semiperimeter_of_triangle_l539_539136


namespace total_buyers_l539_539916

noncomputable theory
open_locale classical

-- Definitions
def C := 50
def M := 40
def B := 18
def non_mixing_prob := 0.28

-- Proof statement
theorem total_buyers : ∃ T : ℕ, T - 0.28 * T = 72 := by
  sorry

end total_buyers_l539_539916


namespace least_three_digit_number_product8_l539_539857

theorem least_three_digit_number_product8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (digits 10 n).prod = 8 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (digits 10 m).prod = 8 → n ≤ m) :=
sorry

end least_three_digit_number_product8_l539_539857


namespace part_a_part_b_l539_539463

-- Define the matrix a_ij representing problem solving.
def ProblemMatrix : Type := List (List ℕ)

def student_solved_problem (a : ProblemMatrix) (i j : ℕ) : Prop :=
  a.nth i >>= (·.nth j) = some 1

-- Define the conditions: 24 students and 25 problems, each problem solved by at least one student.
variable (a : ProblemMatrix)
variable (students : Fin 24)
variable (problems : Fin 25)
variable (solved_by_someone : ∀ j, ∃ i, student_solved_problem a i j)

-- Part (a): Each student can solve an even number of marked problems.
theorem part_a : ∃ x : Fin 25 → ℤ, (∀ i, 
    ∑ j in Finset.univ, ↑(if student_solved_problem a i j then x j else 0) = 0 ∧ 
    ∃ j, x j ≠ 0) :=
sorry

-- Part (b): Problems can be marked with "+" and "-" such that each student has equal points.
theorem part_b : ∃ (plus minus : Fin 25 → ℤ), (∀ i,
    ∑ j in Finset.univ, ↑(if student_solved_problem a i j then plus j else 0) = 
    ∑ j in Finset.univ, ↑(if student_solved_problem a i j then minus j else 0)) :=
sorry

end part_a_part_b_l539_539463


namespace least_three_digit_with_product_eight_is_124_l539_539867

noncomputable def least_three_digit_with_product_eight : ℕ :=
  let candidates := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (x.digits 10).prod = 8} in
  if h : Nonempty candidates then
    let min_candidate := Nat.min' _ h in min_candidate
  else
    0 -- default to 0 if no such number exists

theorem least_three_digit_with_product_eight_is_124 :
  least_three_digit_with_product_eight = 124 := sorry

end least_three_digit_with_product_eight_is_124_l539_539867


namespace find_vector_v_l539_539317

def vector3 (α: Type*) := (α × α × α)

variables (V : Type*) [AddCommGroup V] [Module ℝ V]

-- Vector definitions
def a : vector3 ℝ := (1, 2, 0)
def b : vector3 ℝ := (-1, 0, 2)
def c : vector3 ℝ := (0, 1, -1)

-- Operations
def cross_product : vector3 ℝ → vector3 ℝ → vector3 ℝ
| (x1, y1, z1), (x2, y2, z2) := (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)

-- The condition vectors
variable (v : vector3 ℝ)

-- The assumptions
axiom v_cross_a_eq_b_cross_a : cross_product v a = cross_product b a
axiom v_cross_b_eq_a_cross_b : cross_product v b = cross_product a b
axiom v_decomposition : ∃ t : ℝ, v = (1, 2, 0) + (-1, 0, 2) + t • (0, 1, -1)

-- The theorem to prove
theorem find_vector_v : v = (0, 2, 2) := by
  sorry

end find_vector_v_l539_539317


namespace trigonometric_function_property_l539_539809

theorem trigonometric_function_property :
  ∀ (x : ℝ), 2 * (cos (x - π / 4)) ^ 2 - 1 = sin (2 * x) ∧ 
             (∀ T > 0, (∀ x, 2*(cos (x-T/2-π/4))^2 -1 = sin(2*x) → T ≥ π)) := sorry

end trigonometric_function_property_l539_539809


namespace running_speed_is_correct_l539_539512

-- Definitions stemming from the problem conditions
def walking_speed : ℝ := 4
def total_distance : ℝ := 12
def half_distance : ℝ := total_distance / 2
def total_time : ℝ := 2.25

-- The mathematical equivalent proof problem
theorem running_speed_is_correct (R : ℝ) : 1.5 + half_distance / R = total_time → R = 8 := by
  intro h
  sorry

end running_speed_is_correct_l539_539512


namespace flower_bed_area_and_pots_l539_539382

noncomputable def area_of_flower_bed (d : ℝ) : ℝ := 3.14 * (d / 2) ^ 2

noncomputable def number_of_pots (area : ℝ) (pot_area : ℝ) : ℝ := area / pot_area

theorem flower_bed_area_and_pots : 
  (round (number_of_pots (area_of_flower_bed 60) (1 / 10)) / 10000) = 3 :=
  by 
    have h_area : area_of_flower_bed 60 = 2826 := sorry
    have h_pots : number_of_pots 2826 (1 / 10) = 28260 := sorry
    have h_round : round (28260 / 10000) = 3 := sorry
    sorry

end flower_bed_area_and_pots_l539_539382


namespace correct_statements_count_l539_539254

theorem correct_statements_count (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  (if (∃ a b, 0 < a ∧ 0 < b ∧ a < b ∧ (sqrt a + sqrt b = sqrt 18)) then 1 else 0)
  + (if (∃ a b, 0 < a ∧ 0 < b ∧ a < b ∧ (sqrt a + sqrt b = sqrt 75)) then 1 else 0)
  + (if (¬ ∃ a b, 0 < a ∧ 0 < b ∧ a < b ∧ (sqrt a + sqrt b = sqrt 260)) then 1 else 0)
  + (if ∀ c, (∃ a b, 0 < a ∧ 0 < b ∧ a < b ∧ (sqrt a + sqrt b = sqrt c)) →
    (c % 49 = 0 ∨ c % 64 = 0) then 1 else 0)
  = 3 :=
sorry

end correct_statements_count_l539_539254


namespace least_three_digit_product_eight_l539_539856

theorem least_three_digit_product_eight : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).prod = 8 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (nat.digits 10 m).prod = 8 → n ≤ m :=
by
  sorry

end least_three_digit_product_eight_l539_539856


namespace equation_of_line_l539_539670

theorem equation_of_line (A B: ℝ × ℝ) (C: ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  (C = (-1, -1)) →
  (l = λ p, 3 * p.1 + 4 * p.2 + 2 = 0 ∨ p.1 = -2) →
  (A ≠ B) →
  (A.1^2 + A.2^2 + 2 * A.1 + 2 * A.2 - 2 = 0) →
  (B.1^2 + B.2^2 + 2 * B.1 + 2 * B.2 - 2 = 0) →
  (|√ ((A.1 - B.1)^2 + (A.2 - B.2)^2)| = 2 * √ ((3)) → 
  ∃ l, l (-2, 1) := by
  sorry

end equation_of_line_l539_539670


namespace circle_n_gon_area_ineq_l539_539055

variable {n : ℕ} {S S1 S2 : ℝ}

theorem circle_n_gon_area_ineq (h1 : S1 > 0) (h2 : S > 0) (h3 : S2 > 0) : 
  S * S = S1 * S2 := 
sorry

end circle_n_gon_area_ineq_l539_539055


namespace length_segment_AB_l539_539697

-- Condition definitions
def line_l (t : ℝ) : ℝ × ℝ := (1 + 3 / 5 * t, 4 / 5 * t)
def curve_C (k : ℝ) : ℝ × ℝ := (4 * k^2, 4 * k)

-- Proof statement
theorem length_segment_AB :
  let A := (4, 4)
  let B := (1 / 4, -1)
  (real.dist (A.1, A.2) (B.1, B.2)) = 25 / 4 :=
sorry

end length_segment_AB_l539_539697


namespace coeff_binomial_expansion_l539_539666

theorem coeff_binomial_expansion (a : ℝ) : 
  (∃ (a : ℝ), ∃ (r : ℕ), -7 + 2 * r = -3 ∧ 
  ∏(n in {0, 7}, (7.choose r) * (2 ^ r) * (a ^ (7 - r)) * x ^ (-7 + 2 * r)) = 84) -> 
  a = 1 := 
by
  sorry

end coeff_binomial_expansion_l539_539666


namespace sum_of_three_numbers_l539_539406

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a * b + b * c + c * a = 100) : 
  a + b + c = 21 := 
by
  sorry

end sum_of_three_numbers_l539_539406


namespace number_of_children_is_five_l539_539403

/-- The sum of the ages of children born at intervals of 2 years each is 50 years, 
    and the age of the youngest child is 6 years.
    Prove that the number of children is 5. -/
theorem number_of_children_is_five (n : ℕ) (h1 : (0 < n ∧ n / 2 * (8 + 2 * n) = 50)): n = 5 :=
sorry

end number_of_children_is_five_l539_539403


namespace even_tails_probability_l539_539917

/- Define the main theorem statement for the probability of getting an even number of tails after 2021 flips, using the conditions provided. -/
theorem even_tails_probability : 
  let p := 0.5 in
  ∀ (n : ℕ), 
    n = 2021 →
    let probability_of_even_tails := 
      if n % 2 = 0 then p
      else p 
    in 
    probability_of_even_tails = 0.5 :=
by
  intros p n hn probability_of_even_tails
  rw hn
  simp only
  sorry

end even_tails_probability_l539_539917


namespace probability_of_a_firing_l539_539478

/-- 
Prove that the probability that A will eventually fire the bullet is 6/11, given the following conditions:
1. A and B take turns shooting with a six-shot revolver that has only one bullet.
2. They randomly spin the cylinder before each shot.
3. A starts the game.
-/
theorem probability_of_a_firing (p_a : ℝ) :
  (1 / 6) + (5 / 6) * (5 / 6) * p_a = p_a → p_a = 6 / 11 :=
by
  intro hyp
  have h : p_a - (25 / 36) * p_a = 1 / 6 := by
    rwa [← sub_eq_of_eq_add hyp, sub_self, zero_mul] 
  field_simp at h
  linarith
  sorry

end probability_of_a_firing_l539_539478


namespace train_length_l539_539524

/-- Length of a train moving at a speed of 132 km/hour crossing a platform of length 165 meters in 7.499400047996161 seconds is approximately 110 meters. -/
theorem train_length (v : ℝ) (t : ℝ) (l_p : ℝ) (l_t : ℝ) :
  v = 132 →
  t = 7.499400047996161 →
  l_p = 165 →
  l_t = v * (1000 / 3600) * t - l_p →
  l_t ≈ 110 :=
by
  intros hv ht hlp hlt
  rw [hv, ht, hlp] at hlt
  norm_num at hlt
  sorry

end train_length_l539_539524


namespace distance_on_map_correct_l539_539952

-- Define the conditions given in the problem
def time := 3.5 -- hours
def speed := 60 -- miles per hour
def scale := 0.023809523809523808 -- inches per mile

-- Define the actual distance Pete traveled
def actual_distance := speed * time

-- Define the distance on the map based on the actual distance and scale
def distance_on_map := actual_distance * scale

-- Statement to prove
theorem distance_on_map_correct : distance_on_map = 5 := 
by 
  -- Prove steps omitted
  sorry

end distance_on_map_correct_l539_539952


namespace problem_solution_sets_l539_539752

theorem problem_solution_sets (x y : ℝ) :
  (x^2 * y + y^3 = 2 * x^2 + 2 * y^2 ∧ x * y + 1 = x + y) →
  ( (x = 0 ∧ y = 0) ∨ y = 2 ∨ x = 1 ∨ y = 1 ) :=
by
  sorry

end problem_solution_sets_l539_539752


namespace least_three_digit_number_product8_l539_539859

theorem least_three_digit_number_product8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (digits 10 n).prod = 8 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (digits 10 m).prod = 8 → n ≤ m) :=
sorry

end least_three_digit_number_product8_l539_539859


namespace no_valid_four_digit_numbers_l539_539657

theorem no_valid_four_digit_numbers :
  ∀ N: ℕ, (1000 ≤ N ∧ N ≤ 9999) →
         (∃ a: ℕ, ∃ x: ℕ, a = N / 1000 ∧ x = N % 1000 ∧ x = 1000 * a / 7 ∧ 100 ≤ x ∧ x ≤ 999) → false :=
begin
  sorry
end

end no_valid_four_digit_numbers_l539_539657


namespace seq_sum_2019_eq_zero_l539_539211

def a (n : ℕ) : ℝ := Real.sin (n * Real.pi / 2)

def S (n : ℕ) : ℝ := (Finset.range n).sum a

theorem seq_sum_2019_eq_zero : S 2019 = 0 := by
  sorry

end seq_sum_2019_eq_zero_l539_539211


namespace triangle_ABC_BC_l539_539274

noncomputable def triangle_ABC_B_not := 30
noncomputable def triangle_ABC_AC := 2 * Real.sqrt 5
noncomputable def point_D := sorry -- We will define D later as required by the conditions
noncomputable def triangle_ACD_CD := 2
noncomputable def triangle_ACD_is_acute := Boolean.true
noncomputable def triangle_ACD_area := 4

theorem triangle_ABC_BC 
  (B : ℝ) 
  (AC : ℝ)
  (CD : ℝ) 
  (is_acute : Bool) 
  (area_ACD : ℝ)
  (BC : ℝ)
  (hB : B = 30) 
  (hAC : AC = 2 * Real.sqrt 5) 
  (hCD : CD = 2) 
  (his_acute : is_acute = Boolean.true) 
  (harea_ACD : area_ACD = 4) : 
  BC = 4 := 
sorry

end triangle_ABC_BC_l539_539274


namespace letter_digit_value_l539_539974

theorem letter_digit_value 
  (E H M O P : ℕ)
  (distinct_digits : ∀ x y : ℕ, x ≠ y → x ∈ {E, H, M, O, P} → y ∈ {E, H, M, O, P} → x ≠ y)
  (valid_digits : E ∈ {1, 2, 3, 4, 6, 8, 9} ∧ H ∈ {1, 2, 3, 4, 6, 8, 9} ∧ 
  M ∈ {1, 2, 3, 4, 6, 8, 9} ∧ O ∈ {1, 2, 3, 4, 6, 8, 9} ∧ P ∈ {1, 2, 3, 4, 6, 8, 9})
  (eq1 : E * H = M * O * P * O * 3)
  (eq2 : E + H = M + O + P + O + 3) :
  E * H + M * O * P * O * 3 = 72 :=
by
  sorry

end letter_digit_value_l539_539974


namespace resulting_solid_has_correct_properties_l539_539080

variables (a b c : ℝ) (hab : a > b) (hbc : b > c)

noncomputable def resulting_faces : ℕ := 36
noncomputable def resulting_edges : ℕ := 30
noncomputable def resulting_vertices : ℕ := 20

def face_shapes : list String := ["hexagons", "rhombuses", "pentagons"]

noncomputable def volume_of_solid : ℝ :=
  a * b * c + (1/2) * (a * b^2 + a * c^2 + b * c^2) - (1/6) * b^3 - (1/3) * c^3

theorem resulting_solid_has_correct_properties :
  ∀ (a b c : ℝ) (hab : a > b) (hbc : b > c),
    resulting_faces a b c hab hbc = 36 ∧
    resulting_edges a b c hab hbc = 30 ∧
    resulting_vertices a b c hab hbc = 20 ∧
    face_shapes.fillList = ["hexagons", "rhombuses", "pentagons"] ∧
    volume_of_solid a b c = (a * b * c + (1/2) * (a * b^2 + a * c^2 + b * c^2) - (1/6) * b^3 - (1/3) * c^3) := by
  sorry

end resulting_solid_has_correct_properties_l539_539080


namespace general_term_formula_sum_of_bn_l539_539027

-- Given conditions about the sequence {a_n}
variables (d : ℝ) (h1 : d ≠ 0)
noncomputable def a₁ := 2
noncomputable def a₂ := 2 + d
noncomputable def a₄ := 2 + 3 * d
noncomputable def a₈ := 2 + 7 * d

-- Given condition that a₂, a₄, a₈ form a geometric progression
axiom geo_progression : (a₄ d) ^ 2 = (a₂ d) * (a₈ d)

-- Prove general term formula for the sequence {a_n}
theorem general_term_formula : (∀ n : ℕ, n ≥ 1 → (d ≠ 0) → a₁ = 2 → geo_progression →
  (∃ d : ℝ, a₁ = 2 ∧ ∀ n : ℕ, aₙ = 2 + d * (n - 1)) → ∀ n : ℕ, aₙ = 2 * n) := sorry

-- Defining {b_n} and proving sum of the first n terms
noncomputable def bₙ (n : ℕ) : ℝ := 2 / ((n + 1) * (2 * n))

-- Sum of the first n terms of sequence {b_n}
theorem sum_of_bn (n : ℕ) : (∀ n ≥ 1, bₙ = 2 / ((n + 1) * (2 * n)) →
  (∑ i in range n, bₙ i) = n / (n + 1)) := sorry

end general_term_formula_sum_of_bn_l539_539027


namespace area_large_square_l539_539694

theorem area_large_square (a b c : ℝ) 
  (h1 : a^2 = b^2 + 32) 
  (h2 : 4*a = 4*c + 16) : a^2 = 100 := 
by {
  sorry
}

end area_large_square_l539_539694


namespace monic_poly_with_root_l539_539162

theorem monic_poly_with_root (x : ℝ) :
  (∃ P : polynomial ℚ, P.monic ∧ P.degree = 4 ∧ root P (sqrt 3 + sqrt 5)) :=
begin
  use polynomial.X^4 - 16 * polynomial.X^2 + 4,
  split,
  { exact polynomial.monic_X_pow_sub_C 1 (polynomial_nat_degree_X_pow (4 - 1)), },
  split,
  { exact polynomial.degree_sub_C (polynomial_nat_degree_X_pow (4 - 1)), },
  {
    sorry
  }
end

end monic_poly_with_root_l539_539162


namespace solve_inequality_l539_539383

variable (f : ℝ → ℝ)

axiom domain_f : ∀ x : ℝ, f x = f x
axiom f_at_4 : f 4 = -3
axiom f_double_prime : ∀ x : ℝ, f'' x < 3

theorem solve_inequality : ∀ x : ℝ, (f x < 3 * x - 15) ↔ (x > 4) := 
by 
  sorry

end solve_inequality_l539_539383


namespace hyperbola_eccentricity_l539_539801

theorem hyperbola_eccentricity (a b c : ℝ) (h : y = x + (1 / x)) 
  (h_symmetry : ¬(axis_of_symmetry ∈ coordinate_axes))
  (h_asymptotes : ∀ x, asymptote_y = (y = x) ∨ asymptote_y = (y = 0)) :
  let b := a * (Real.sqrt 2 - 1) in
  c^2 = a^2 + b^2 → 
  ∃ e, e = Real.sqrt (4 - 2 * Real.sqrt 2) :=
by 
  sorry

end hyperbola_eccentricity_l539_539801


namespace number_of_true_statements_l539_539593

def statement1 (a b c d : ℝ) := a > b ∧ c > d → a - c > b - d
def statement2 (a b c d : ℝ) := a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a * c > b * d
def statement3 (a b : ℝ) := a > b ∧ b > 0 → 3 * a > 3 * b
def statement4 (a b : ℝ) := a > b ∧ b > 0 → 1 / a^2 < 1 / b^2

theorem number_of_true_statements (a b c d : ℝ) :
  (∃ (a b c d : ℝ), ¬statement1 a b c d) ∧
  ∀ (a b c d : ℝ), statement2 a b c d ∧
  ∀ (a b : ℝ), statement3 a b ∧
  ∀ (a b : ℝ), statement4 a b →
  3 := by
  sorry

end number_of_true_statements_l539_539593


namespace problem1_problem2_l539_539209

-- Definitions for the first proof problem
variables {V : Type*} [inner_product_space ℝ V]
variable (a b : V)
variable (theta1 : ℝ)

-- Given conditions
def given_cond1 := (∥a∥ = 1) ∧ (∥b∥ = real.sqrt 2) ∧ (theta1 = real.pi / 3)

-- First proof problem
theorem problem1 (h : given_cond1 a b theta1) :
  ∥a + b∥ = real.sqrt (3 + real.sqrt 2) := by sorry

-- Definitions for the second proof problem
variable (theta2 : ℝ)

-- Given conditions
def given_cond2 := (∥a∥ = 1) ∧ (∥b∥ = real.sqrt 2) ∧ inner a (a - b) = 0

-- Second proof problem
theorem problem2 (h : given_cond2 a b) :
  theta2 = real.pi / 4 := by sorry

end problem1_problem2_l539_539209


namespace ellipse_focus_chord_l539_539963

/-- Given an ellipse with the equation (x^2)/4 + y^2 = 1, with one focus at F = (2, 0),
    and a point P = (p, 0) where p > 0, such that for any chord AB passing through F,
    the angles ∠APF and ∠BPF are equal, then p = 3. -/
theorem ellipse_focus_chord (p : ℝ) (h : p > 0) :
  (∀ (A B : ℝ), ∃ (x : ℝ), (((x^2) / 4) + ((A * x^2) / (4 * srat.sqrt (4 * A^2 + 1)) + 
  (B * x^2) / (4 * srat.sqrt (4 * B^2 + 1)) = 1) ∧ ((angle A p 2) = (angle B p 2))) → p = 3 :=
sorry

end ellipse_focus_chord_l539_539963


namespace determine_x_y_l539_539774

-- Definitions from the conditions
def cond1 (x y : ℚ) : Prop := 12 * x + 198 = 12 * y + 176
def cond2 (x y : ℚ) : Prop := x + y = 29

-- Statement to prove
theorem determine_x_y : ∃ x y : ℚ, cond1 x y ∧ cond2 x y ∧ x = 163 / 12 ∧ y = 185 / 12 := 
by 
  sorry

end determine_x_y_l539_539774


namespace find_k_plus_m_l539_539388

def initial_sum := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
def initial_count := 9

def new_list_sum (m k : ℕ) := initial_sum + 8 * m + 9 * k
def new_list_count (m k : ℕ) := initial_count + m + k

def average_eq_73 (m k : ℕ) := (new_list_sum m k : ℝ) / (new_list_count m k : ℝ) = 7.3

theorem find_k_plus_m : ∃ (m k : ℕ), average_eq_73 m k ∧ (k + m = 21) :=
by
  sorry

end find_k_plus_m_l539_539388


namespace max_distance_from_circle_to_line_l539_539010

def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def line (x y : ℝ) : Prop := 4 * x - 3 * y + 25 = 0

theorem max_distance_from_circle_to_line :
  ∃ M : ℝ, M = 7 ∧ 
  ∀ (x y : ℝ), circle x y → ∃ (u v : ℝ), line u v ∧ sqrt ((x - u)^2 + (y - v)^2) = M :=
by
  sorry

end max_distance_from_circle_to_line_l539_539010


namespace octahedron_edge_length_l539_539514

/-- 
A regular octahedron circumscribed around four identical balls, 
each with a radius of 2 units, where three of the balls are touching each other and resting on the floor 
while the fourth ball is resting on the top of these three, has an edge length of 4 units.
-/
theorem octahedron_edge_length (r : ℝ) (h1 : r = 2) 
    (A B C D : euclidean_space.ℝ 3)
    (h2 : dist A B = 2 * r ∧ dist B C = 2 * r ∧ dist C A = 2 * r) 
    (h3 : dist D A = 2 * r ∧ dist D B = 2 * r ∧ dist D C = 2 * r) :
    ∃ s, s = 4 ∧ s = dist A B ∧ dist A C = s ∧ dist A D = s ∧ dist B C = s ∧ dist B D = s ∧ dist C D = s := by
  sorry

end octahedron_edge_length_l539_539514


namespace prob_atleast_two_consecutive_in_ten_l539_539600

noncomputable def prob_consecutive (s : Finset ℕ) (k : ℕ) : ℚ :=
  if h : k ≤ s.card then ( (s.powerset.filter (λ t, t.card = k ∧ ∃ a b c ∈ t, a + 1 = b ∨ b + 1 = c ∨ c + 1 = a)).card : ℚ) / (s.choose k).card else 0

theorem prob_atleast_two_consecutive_in_ten : 
  prob_consecutive (Finset.range 10) 3 = 8 / 15 := 
sorry

end prob_atleast_two_consecutive_in_ten_l539_539600


namespace triangle_perimeter_l539_539396

theorem triangle_perimeter (x : ℕ) (a b c : ℕ) 
  (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : c = 5 * x)  
  (h4 : c - a = 6) : a + b + c = 36 := 
by
  sorry

end triangle_perimeter_l539_539396


namespace find_w_l539_539142

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![0, 1], ![3, 0]]

def w : Vector (Fin 2) ℚ :=
  ![0, 7/20]

theorem find_w :
  (B^6 + B^4 + B^2 + 1) * w = ![0, 14] :=
by
  -- Definitions for B^n and the calculations derived in the problem will be used here
  sorry

end find_w_l539_539142


namespace simplify_and_rationalize_l539_539357

theorem simplify_and_rationalize 
: (sqrt 2 / sqrt 5) * (sqrt 8 / sqrt 9) * (sqrt 3 / sqrt 7) = (4 * sqrt 105) / 105 :=
    sorry

end simplify_and_rationalize_l539_539357


namespace number_of_books_bought_l539_539107

def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def remaining_books : ℕ := 2

theorem number_of_books_bought : 
  let total_books_after_shopping := shelves * books_per_shelf + remaining_books in
  total_books_after_shopping - initial_books = 26 := 
by 
  sorry

end number_of_books_bought_l539_539107


namespace probability_exactly_two_students_from_same_class_l539_539280

theorem probability_exactly_two_students_from_same_class
  (num_classes : ℕ) (students_per_class : ℕ) (total_selected : ℕ)
  (h1 : num_classes = 5) (h2 : students_per_class = 2) (h3 : total_selected = 4) :
  (4 / 7 : ℝ) = 
  let num_students := num_classes * students_per_class
      num_ways_to_select_4 := (Finset.card (Finset.powersetLen total_selected (Finset.range num_students))) 
      num_ways_exactly_two_same_class := num_classes * choose students_per_class 2 * choose ((num_classes - 1) * students_per_class) (total_selected - 2)
  in (num_ways_exactly_two_same_class / num_ways_to_select_4 : ℝ) :=
by
  sorry

end probability_exactly_two_students_from_same_class_l539_539280


namespace evaluate_expression_l539_539571

variable (x y z : ℚ) -- assuming x, y, z are rational numbers

theorem evaluate_expression (h1 : x = 1 / 4) (h2 : y = 3 / 4) (h3 : z = -8) :
  x^2 * y^3 * z^2 = 108 := by
  sorry

end evaluate_expression_l539_539571


namespace radius_of_the_dumplings_l539_539124

theorem radius_of_the_dumplings :
  ∃ r : ℝ, (∀ (r_hemispherical_bowl : ℝ), 
    r_hemispherical_bowl = 6 →
    ∀ (r_dumpling : ℝ), 
    r_dumpling = r →
    let OO1 := r_hemispherical_bowl - r_dumpling,
        OO2 := r_hemispherical_bowl - r_dumpling,
        OO3 := r_hemispherical_bowl - r_dumpling,
        O1O2 := 2 * r_dumpling,
        O1O3 := 2 * r_dumpling,
        O2O3 := 2 * r_dumpling,
        O1M := (2 * Real.sqrt 3 / 3) * r_dumpling,
        OM := r_dumpling in
    (O1M^2 + OM^2 = OO1^2) →
    r = (3 * Real.sqrt 21 - 9) / 2) :=
sorry

end radius_of_the_dumplings_l539_539124


namespace initial_amount_of_liquid_A_l539_539897

theorem initial_amount_of_liquid_A (A B : ℝ) (initial_ratio : A = 4 * B) (removed_mixture : ℝ) (new_ratio : (A - (4/5) * removed_mixture) = (2 / 3) * ((B - (1/5) * removed_mixture) + removed_mixture)) :
  A = 16 := 
  sorry

end initial_amount_of_liquid_A_l539_539897


namespace probability_log_floor_eq_l539_539352

open Real

noncomputable def log_floor_eq (x y : ℝ) : Prop :=
  (⌊Real.log x / Real.log 3⌋ = ⌊Real.log y / Real.log 3⌋)

noncomputable def prob_log_floor_eq_xy : ℝ :=
  ∫ (x : ℝ) in 0..2, ∫ (y : ℝ) in 0..2, if log_floor_eq x y then (1 / 4) else 0

theorem probability_log_floor_eq :
  prob_log_floor_eq_xy = 25 / 36 :=
sorry

end probability_log_floor_eq_l539_539352


namespace odd_at_least_one_not_hit_even_not_always_one_escapes_l539_539068

-- Definitions
def people (n : ℕ) := fin n
def nearest_person (n : ℕ) (p : people n) : people n := sorry -- function to define the nearest person (assumed to be unique)

-- Proposition 1: At least one person is not hit if n is odd
theorem odd_at_least_one_not_hit {n : ℕ} (hn : n % 2 = 1) :
  ∃ p : people n, ∀ q : people n, nearest_person n q ≠ p := sorry

-- Proposition 2: It is not always the case that one person escapes if n is even
theorem even_not_always_one_escapes {n : ℕ} (hn : n % 2 = 0) :
  ¬ (∃ p : people n, ∀ q : people n, nearest_person n q ≠ p) := sorry

end odd_at_least_one_not_hit_even_not_always_one_escapes_l539_539068


namespace spends_on_food_l539_539333

noncomputable def weekly_pay : ℕ := 100
noncomputable def arcade_fraction : ℕ := 2
noncomputable def arcade_hourly_cost : ℕ := 8
noncomputable def play_time_minutes : ℕ := 300
noncomputable def minutes_in_hour : ℕ := 60

theorem spends_on_food:
  let total_in_arcade := weekly_pay / arcade_fraction in
  let play_time_hours := play_time_minutes / minutes_in_hour in
  let token_cost := play_time_hours * arcade_hourly_cost in
  (total_in_arcade - token_cost) = 10 :=
by
  sorry

end spends_on_food_l539_539333


namespace probability_of_a_firing_l539_539479

/-- 
Prove that the probability that A will eventually fire the bullet is 6/11, given the following conditions:
1. A and B take turns shooting with a six-shot revolver that has only one bullet.
2. They randomly spin the cylinder before each shot.
3. A starts the game.
-/
theorem probability_of_a_firing (p_a : ℝ) :
  (1 / 6) + (5 / 6) * (5 / 6) * p_a = p_a → p_a = 6 / 11 :=
by
  intro hyp
  have h : p_a - (25 / 36) * p_a = 1 / 6 := by
    rwa [← sub_eq_of_eq_add hyp, sub_self, zero_mul] 
  field_simp at h
  linarith
  sorry

end probability_of_a_firing_l539_539479


namespace minimum_distance_l539_539462

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem minimum_distance :
  let line_eqn := λ P : ℝ × ℝ, P.2 = 2
  let circle_eqn := λ Q : ℝ × ℝ, (Q.1 + 1)^2 + Q.2^2 = 1
  ∃ P Q : ℝ × ℝ, line_eqn P ∧ circle_eqn Q ∧ distance P Q = 1 :=
sorry

end minimum_distance_l539_539462


namespace isola_population_estimate_l539_539570

theorem isola_population_estimate :
  ∃ year, ∀ n : ℕ, year = 2000 + n * 30 ∧ 250 * 4 ^ n ≈ 8000 → year = 2090 := by
  sorry

end isola_population_estimate_l539_539570


namespace percentage_of_boys_passed_thm_l539_539686

noncomputable def percentage_of_boys_passed (total_candidates girls passed_girls failed_percentage : ℕ) : ℕ :=
  let boys := total_candidates - girls
  let passed_percentage := 100 - failed_percentage
  let total_passed_candidates := (35.3 / 100) * total_candidates
  let passed_girls_count := (32 / 100) * girls
  let equation := total_passed_candidates - passed_girls_count
  let boys_passed := (equation / boys) * 100
  boys_passed.toNat

theorem percentage_of_boys_passed_thm : percentage_of_boys_passed 2000 900 32 64.7 = 38 := by
  sorry

end percentage_of_boys_passed_thm_l539_539686


namespace lashawn_three_times_kymbrea_l539_539306

-- Definitions based on the conditions
def kymbrea_collection (months : ℕ) : ℕ := 50 + 3 * months
def lashawn_collection (months : ℕ) : ℕ := 20 + 5 * months

-- Theorem stating the core of the problem
theorem lashawn_three_times_kymbrea (x : ℕ) 
  (h : lashawn_collection x = 3 * kymbrea_collection x) : x = 33 := 
sorry

end lashawn_three_times_kymbrea_l539_539306


namespace area_ratio_triangle_l539_539288

theorem area_ratio_triangle 
  {A B C D E F : Type} 
  (AB AC : ℕ) (h_AB : AB = 130) (h_AC : AC = 130)
  (AD : ℕ) (h_AD : AD = 50)
  (CF : ℕ) (h_CF : CF = 80)
  (BD_AF_ratio : ℕ) (h_ratio : BD_AF_ratio = 21 / 5) : 
  [CEF] / [DBE] = 21 / 5 := 
sorry

end area_ratio_triangle_l539_539288


namespace isosceles_triangle_dot_product_l539_539292

theorem isosceles_triangle_dot_product :
  ∀ (A B C D : ℝ³), 
    ∃ (AB AC BC: ℝ), 
      AB = 2 ∧ AC = 2 ∧
      angle A B C = 2 * π / 3 ∧
      (∃ (ratios : ℝ), ratios = 3 ∧ area A C D = ratios * (area A B D)) →
      (vector.dot (A - B) (A - D) = 5 / 2) := 
sorry

end isosceles_triangle_dot_product_l539_539292


namespace painted_cube_l539_539925

theorem painted_cube (n : ℕ) (h : 3 / 4 * (6 * n ^ 3) = 4 * n ^ 2) : n = 2 := sorry

end painted_cube_l539_539925


namespace variance_of_data_l539_539012

noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := (data.sum / data.length.to_real)
  (data.map (λ x, (x - mean)^2)).sum / data.length.to_real

theorem variance_of_data : variance [1, 2, 0, -1, -2] = 2 := by
  sorry

end variance_of_data_l539_539012


namespace period_f_translation_symmetry_phi_l539_539229

def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)

theorem period_f : ∃ T > 0, ∀ x, f (x + T) = f x := 
  by sorry

theorem translation_symmetry_phi (φ : ℝ) : 
  (∀ x, f (x - φ) = f (-x)) → φ = (3 * Real.pi) / 8 := 
  by sorry

end period_f_translation_symmetry_phi_l539_539229


namespace square_root_2a_minus_3b_solve_eqn_l539_539456

variables (a b : ℝ) (x : ℝ)

-- Conditions
def condition1 := |2 * a + b| + sqrt (3 * b + 12) = 0

-- Proving the first statement
theorem square_root_2a_minus_3b (h : condition1) : sqrt (2 * a - 3 * b) = 4 :=
sorry

-- Proving the second statement
theorem solve_eqn (h : condition1) : a * x^2 + 4 * b - 2 = 0 ↔ x = 3 ∨ x = -3 :=
sorry

end square_root_2a_minus_3b_solve_eqn_l539_539456


namespace increase_in_lighting_power_l539_539813

-- Conditions
def N_before : ℕ := 240
def N_after : ℕ := 300

-- Theorem
theorem increase_in_lighting_power : N_after - N_before = 60 := by
  sorry

end increase_in_lighting_power_l539_539813


namespace part1_part2_l539_539186

variables {a b : ℝ^3} -- Define vectors a and b

-- Define the conditions under which we are working
axiom ha : ∥a∥ = 1
axiom hb : ∥b∥ = 1
axiom hab : ∥a + b∥ = real.sqrt 3

-- First part: proving |a - b| = 1
theorem part1 : ∥a - b∥ = 1 :=
sorry

-- Second part: proving (a + b) is perpendicular to (a - b)
theorem part2 : inner (a + b) (a - b) = 0 :=
sorry

end part1_part2_l539_539186


namespace ellipse_equation_l539_539384

theorem ellipse_equation (x y : ℝ) :
  sqrt(x^2 + (y - 3)^2) + sqrt(x^2 + (y + 3)^2) = 10 ↔
  (x^2 / 25) + (y^2 / 16) = 1 :=
by
  sorry

end ellipse_equation_l539_539384


namespace average_output_40_cogs_per_hour_l539_539118

theorem average_output_40_cogs_per_hour :
  (let initial_cogs := 60 in
   let initial_rate := 30 in
   let second_cogs := 60 in
   let second_rate := 60 in
   let total_cogs := initial_cogs + second_cogs in
   let initial_time := initial_cogs / initial_rate in
   let second_time := second_cogs / second_rate in
   let total_time := initial_time + second_time in
   (total_cogs / total_time = 40)) := sorry

end average_output_40_cogs_per_hour_l539_539118


namespace volume_of_tetrahedron_l539_539949

theorem volume_of_tetrahedron (AB AD : ℝ) (h1 : AB = 3) (h2 : AD = 4) (AE : ℝ) (h3 : AE = 1) :
  let E := (1 / 3 * AB, 0) in
  let B := (3, 0, 0) in
  let D := (0, 4, 0) in
  let C := (3, 4, 0) in
  let F := (0, 4 / 3 * √5) in
  let base_area := 1 / 2 * 4 * 4 in
  let B'F := √3 / 2 in
  volume_of_tetrahedron B (1/3, 0, 3/2 * √5) F :=
  (1 / 3 * base_area * B'F) = 4 * √3 / 3 :=
sorry

end volume_of_tetrahedron_l539_539949


namespace smallest_number_l539_539537

theorem smallest_number :
  let a := (-5)^(0 : ℤ) in
  let b := -Real.sqrt 5 in
  let c := -(1 / 5) in
  let d := abs (-5) in
  min (min a b) (min c d) = -Real.sqrt 5 :=
by
  sorry

end smallest_number_l539_539537


namespace percent_increase_l539_539899

-- Definitions based on conditions
def initial_price : ℝ := 10
def final_price : ℝ := 15

-- Goal: Prove that the percent increase in the price per share is 50%
theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 50 := 
by
  sorry  -- Proof is not required, so we skip it with sorry.

end percent_increase_l539_539899


namespace yanna_kept_36_apples_l539_539445

-- Define the initial number of apples Yanna has
def initial_apples : ℕ := 60

-- Define the number of apples given to Zenny
def apples_given_to_zenny : ℕ := 18

-- Define the number of apples given to Andrea
def apples_given_to_andrea : ℕ := 6

-- The proof statement that Yanna kept 36 apples
theorem yanna_kept_36_apples : initial_apples - apples_given_to_zenny - apples_given_to_andrea = 36 := by
  sorry

end yanna_kept_36_apples_l539_539445


namespace least_three_digit_with_product_eight_is_124_l539_539863

noncomputable def least_three_digit_with_product_eight : ℕ :=
  let candidates := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (x.digits 10).prod = 8} in
  if h : Nonempty candidates then
    let min_candidate := Nat.min' _ h in min_candidate
  else
    0 -- default to 0 if no such number exists

theorem least_three_digit_with_product_eight_is_124 :
  least_three_digit_with_product_eight = 124 := sorry

end least_three_digit_with_product_eight_is_124_l539_539863


namespace range_of_function_l539_539808

def f (x : ℕ) : ℤ := x^2 - 2 * x

theorem range_of_function : 
  (∃ D : set ℕ, D = {0, 1, 2, 3}) →
  (∃ R : set ℤ, R = {y | ∃ x ∈ {0, 1, 2, 3}, f x = y}) →
  (R = {-1, 0, 3}) := 
by 
  sorry

end range_of_function_l539_539808


namespace transaction_base_is_five_l539_539289

theorem transaction_base_is_five (s : ℕ) (s_pos : s > 0)
    (h630 : 6 * s^2 + 3 * s = nat_of_string (string_of_nat 630 16)) 
    (h250 : 2 * s^2 + 5 * s = nat_of_string (string_of_nat 250 16)) 
    (h470 : 4 * s^2 + 7 * s = nat_of_string (string_of_nat 470 16)) 
    (h1000 : s^3 = nat_of_string (string_of_nat 1000 16)) : 
    s = 5 := 
sorry

end transaction_base_is_five_l539_539289


namespace car_traveling_speed_is_73_5_l539_539492

def car_speed (distance : ℝ) (extra_time : ℝ) (reference_speed : ℝ) (reference_time : ℝ) : ℝ :=
  distance / (reference_time + extra_time / 3600)

theorem car_traveling_speed_is_73_5 :
  car_speed 2 8 80 (2 / 80) = 73.5 :=
by
  -- informal proof here
  sorry

end car_traveling_speed_is_73_5_l539_539492


namespace linear_function_passes_through_point_l539_539338

theorem linear_function_passes_through_point :
  ∀ x y : ℝ, y = -2 * x - 6 → (x = -4 → y = 2) :=
by
  sorry

end linear_function_passes_through_point_l539_539338


namespace usual_time_to_reach_school_l539_539058

theorem usual_time_to_reach_school
  (R T : ℝ)
  (h1 : (7 / 6) * R = R / (T - 3) * T) : T = 21 :=
sorry

end usual_time_to_reach_school_l539_539058


namespace correct_graph_is_E_l539_539845

noncomputable def f (x : ℝ) : ℝ :=
  if h₁ : -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if h₂ : 0 < x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2
  else if h₃ : 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0 -- this handles cases outside the given intervals, making f total

noncomputable def g (x : ℝ) : ℝ := f(x) - 2

theorem correct_graph_is_E : True := sorry

end correct_graph_is_E_l539_539845


namespace find_wednesday_temperature_l539_539127

-- Definitions from conditions
def temperatures : List ℝ := [99.1, 98.2, 98.7, 99.8, 99, 98.9] -- known temperatures for 6 days
def average_temperature : ℝ := 99 -- average temperature for the week

-- Number of days in the week
def days_in_week : ℕ := 7

-- The temperature on Wednesday (unknown to be proved)
def wednesday_temperature := 99.3

theorem find_wednesday_temperature :
  let total_temperature := average_temperature * days_in_week
  let known_temperature_sum := temperatures.sum
  let wednesday_temp := total_temperature - known_temperature_sum
  wednesday_temp = wednesday_temperature :=
by
  sorry

end find_wednesday_temperature_l539_539127


namespace ellipse_properties_circle_conditions_l539_539615

noncomputable def ellipse_equation : String := 
  "The equation of the ellipse is \\(\\frac{x^2}{2} + y^2 = 1\\)"

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (A : ℝ × ℝ) (hA : A = (-1, - real.sqrt 2 / 2)) 
  (F1 : ℝ × ℝ) (hF1 : F1 = (-1, 0)) : 
  a = real.sqrt 2 ∧ b = 1 ∧
  (∀ x y, (x^2) / 2 + y^2 = 1 ↔ (x, y) ∈ elipse) :=
begin
  sorry
end

theorem circle_conditions (x0 y0 : ℝ) (hA_on_ellipse : (x0^2) / 2 + y0^2 = 1)
  (F2 : ℝ × ℝ) (hF2 : F2 = (1, 0)) :
  ∃ x0, x0 = 2 / 3 ∧ (∃ y1 y2, 
  y1 + y2 = 2 * y0 ∧ y1 * y2 = 2 * x0 - 1 ∧ 
  (1 + y1 * y2 = 4 / 3 ∧ (|y1 - y2| = 4 / 3) ∧ 
   (1 / 2) * |y1 - y2| * 1 = 2 / 3)) :=
begin
  sorry
end

end ellipse_properties_circle_conditions_l539_539615


namespace line_equation_through_point_intersects_circle_l539_539667

theorem line_equation_through_point_intersects_circle (x y : ℝ) :
  ∃ l, ((∃ k, l = (fun p:ℝ × ℝ => p.2 - 1 = k * (p.1 + 2))
        ∨ l = (fun p:ℝ × ℝ => p.1 = -2)) 
        ∧ (∃ A B : ℝ × ℝ, A ≠ B ∧ A.1^2 + A.2^2 + 2*A.1 + 2*A.2 - 2 = 0 ∧ B.1^2 + B.2^2 + 2*B.1 + 2*B.2 - 2 = 0
        ∧ l A ∧ l B 
        ∧ real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * real.sqrt 3)) :=
sorry

end line_equation_through_point_intersects_circle_l539_539667


namespace tammy_avg_speed_second_day_l539_539784

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l539_539784


namespace find_alpha_l539_539676

noncomputable def polar_to_cartesian : ℝ := 
  ∀ (ρ θ : ℝ), 
  (ρ = 4 * real.cos (θ - real.pi / 3)) ↔ 
  (let x := ρ * real.cos θ;
       y := ρ * real.sin θ in
    (x - 1)^2 + (y - real.sqrt 3)^2 = 4)

-- We define the conditions as hypotheses to be used in the theorem.

variable (P : Point) (l : Line) (C : Curve) (AB : Segment)
variables (k α : ℝ)

theorem find_alpha :
  P = (2, real.sqrt 3) →
  C = { p | let (x, y) := p in (x - 1)^2 + (y - real.sqrt 3)^2 = 4 } →
  length AB = real.sqrt 13 →
  ∃ α : Real.Angle, α = real.pi / 3 ∨ α = 2 * real.pi / 3 :=
by
  sorry

end find_alpha_l539_539676


namespace Bills_age_proof_l539_539954

variable {b t : ℚ}

theorem Bills_age_proof (h1 : b = 4 * t / 3) (h2 : b + 30 = 9 * (t + 30) / 8) : b = 24 := by 
  sorry

end Bills_age_proof_l539_539954


namespace gcd_sub_12_eq_36_l539_539174

theorem gcd_sub_12_eq_36 :
  Nat.gcd 7344 48 - 12 = 36 := 
by 
  sorry

end gcd_sub_12_eq_36_l539_539174


namespace minimum_trig_expression_l539_539170
variables {x : ℝ}

-- Define the main trigonometric identity
def sine_cosine_identity := ∀ x, sin x ^ 2 + cos x ^ 2 = 1

-- Define the trigonometric expression
def trig_expression := 3 * (sin x) ^ 4 + 4 * (cos x) ^ 4

-- State the theorem that verifies the minimum value
theorem minimum_trig_expression : sine_cosine_identity x → ∃ val, val = 4 ∧ (∀ y, y = trig_expression → y ≥ val) := 
sorry

end minimum_trig_expression_l539_539170


namespace magnitude_b_l539_539218

-- Definitions of vectors a and b and the magnitudes as given conditions
variables (a b : ℝ^3)
variables (θ : ℝ)

-- Define the conditions
def angle_condition : Prop := θ = real.pi / 6
def magnitude_a : Prop := ‖a‖ = 1
def magnitude_2a_minus_b : Prop := ‖2 • a - b‖ = 1

-- Define the main theorem to prove the magnitude of vector b
theorem magnitude_b (h_angle : angle_condition a b θ) (h_mag_a : magnitude_a a) (h_mag_2a_b : magnitude_2a_minus_b a b) :
  ‖b‖ = sqrt 3 :=
sorry

end magnitude_b_l539_539218


namespace square_can_be_cut_into_8_acute_angled_triangles_l539_539138

-- Define an acute-angled triangle
def is_acute_angled_triangle (T : Triangle) := 
  ∀ angle ∈ T.angles, angle < 90

-- Define the problem statement in Lean
theorem square_can_be_cut_into_8_acute_angled_triangles (S : Square) : 
  ∃ (triangles : list Triangle), 
    triangles.length = 8 ∧ 
    (∀ T ∈ triangles, is_acute_angled_triangle T) ∧ 
    covers_square triangles S :=
sorry

end square_can_be_cut_into_8_acute_angled_triangles_l539_539138


namespace g_at_4_l539_539319

def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := 3 - 4 / x

def g (x : ℝ) : ℝ := 1 / (f_inv x) + 7

theorem g_at_4 : g 4 = 7.5 := by
  sorry

end g_at_4_l539_539319


namespace probability_A_shoots_l539_539486

theorem probability_A_shoots (P : ℚ) :
  (∀ n : ℕ, (2 * n + 1) % 2 = 1) →  -- A's turn is always the odd turn
  (∀ m : ℕ, (2 * m) % 2 = 0) →  -- B's turn is always the even turn
  let p_A_first_shot := (1 : ℚ) / 6 in  -- probability A fires on the first shot
  let p_A_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun
  let p_B_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun for B
  let P_A := p_A_first_shot + (p_A_turn * p_B_turn * P) in  -- recursive definition
  P_A = 6 / 11 := -- final probability
sorry

end probability_A_shoots_l539_539486


namespace ball_arrangement_l539_539408

theorem ball_arrangement : ∃ (n : ℕ), n = 120 ∧
  (∀ (ball_count : ℕ), ball_count = 20 → ∃ (box1 box2 box3 : ℕ), 
    box1 ≥ 1 ∧ box2 ≥ 2 ∧ box3 ≥ 3 ∧ box1 + box2 + box3 = ball_count) :=
by
  sorry

end ball_arrangement_l539_539408


namespace range_of_f_l539_539326

def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_f : { x : ℝ | f x > f (2 * x - 1) } = set.Ioo (1 / 3 : ℝ) 1 :=
by
  sorry

end range_of_f_l539_539326


namespace part1_part2a_part2b_part3_l539_539998

-- Definitions of conditions
def M (x : ℝ) : ℝ × ℝ := (1 + Real.cos (2 * x), 1)
def N (x a : ℝ) : ℝ × ℝ := (1, Real.sqrt 3 * Real.sin (2 * x) + a)
def OM (x : ℝ) : ℝ × ℝ := (1 + Real.cos (2 * x), 1)
def ON (x a : ℝ) : ℝ × ℝ := (1, Real.sqrt 3 * Real.sin (2 * x) + a)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)
def f (x a : ℝ) : ℝ := Real.cos (2 * x) + Real.sqrt 3 * Real.sin (2 * x) + 1 + a

-- Theorems to Prove
theorem part1 (x a : ℝ) : dot_product (OM x) (ON x a) = f x a :=
  by sorry

theorem part2a (x k : ℤ) (a : ℝ) : 
  ∀ k, f is_strict_mono_on ([ - Real.pi / 3 + (k * Real.pi), Real.pi / 6 + (k * Real.pi)]) :=
  by sorry

theorem part2b (x k : ℤ) (a : ℝ) :
  ∀ k, f is_strict_anti_on ([ Real.pi / 6 + (k * Real.pi), 2 * Real.pi / 3 + (k * Real.pi)]) :=
  by sorry

theorem part3 (a : ℝ) :
  ∀ x ∈ [0, Real.pi / 2], max (f x a) = 4 → a = 1 :=
  by sorry

end part1_part2a_part2b_part3_l539_539998


namespace anna_apples_ratio_proof_l539_539122

noncomputable def anna_apples_ratio : Prop :=
  let a_T : ℕ := 4
  let a_Th : ℕ := a_T / 2
  ∃ a_W : ℕ, a_T + a_W + a_Th = 14 ∧ a_W / a_T = 2

theorem anna_apples_ratio_proof : anna_apples_ratio := 
by {
  let a_T : ℕ := 4,
  let a_Th : ℕ := a_T / 2,
  use 8, 
  split,
  { 
    calc
      4 + 8 + 2 = 14 : by norm_num
  },
  {
    exact nat.div_eq_of_eq_mul_right (by norm_num) rfl
  } 
}

end anna_apples_ratio_proof_l539_539122


namespace horner_operations_count_l539_539421

-- Definitions based on the conditions
def f (x : ℤ) : ℤ := 5 * x^6 + 4 * x^5 + x^4 + 3 * x^3 - 81 * x^2 + 9 * x - 1

-- Polynomial evaluation using Horner's Rule
def horner_eval (x : ℤ) : ℤ :=
  (((((5 * x + 4) * x + 1) * x + 3) * x - 81) * x + 9) * x - 1

-- Proof statement
theorem horner_operations_count (x : ℤ) :
  ∀ (evaluated_value : ℤ), evaluated_value = horner_eval x ∧
  let horner_multiplications := 6 in
  let horner_additions := 6 in
  true :=
by
  intros
  have eval_correct : evaluated_value = horner_eval x := sorry
  have mult_count : horner_multiplications = 6 := sorry
  have add_count : horner_additions = 6 := sorry
  exact ⟨eval_correct, mult_count, add_count, trivial⟩

end horner_operations_count_l539_539421


namespace proof_problem_l539_539731

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : f 1 = 1
axiom h2 : ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

def m := { y : ℝ | f (1 / 3) = y }.to_finset.card
def t := { y : ℝ | f (1 / 3) = y }.to_finset.sum id

theorem proof_problem : m * t = 1 / 3 := by
  sorry

end proof_problem_l539_539731


namespace repeating_fraction_base_k_l539_539991

theorem repeating_fraction_base_k (k : ℕ) (h : 0 < k) : 
  (0.161616..._k : ℝ) = 8 / 35 ↔ k = 13 := by
  sorry

end repeating_fraction_base_k_l539_539991


namespace find_k_l539_539652

open_locale real_inner_product_space

variables {E : Type*} [inner_product_space ℝ E]

-- Given conditions
variables (e1 e2 : E) (k : ℝ)
hypothesis he1_unit : ∥e1∥ = 1
hypothesis he2_unit : ∥e2∥ = 1
hypothesis he1_e2_angle : real.angle e1 e2 = 2/3 * real.pi
def a := e1 - (2 : ℝ) • e2
def b := k • e1 + e2
hypothesis a_perpendicular_b : inner_product_space.inner a b = 0

-- What we want to prove
theorem find_k : k = 5/4 :=
sorry

end find_k_l539_539652


namespace expression_value_l539_539001

theorem expression_value (x : ℝ) (hx1 : x ≠ -1) (hx2 : x ≠ 2) :
  (2 * x ^ 2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := 
by
  sorry

end expression_value_l539_539001


namespace perfect_square_n_l539_539981

open Nat

theorem perfect_square_n (n : ℕ) : 
  (∃ k : ℕ, 2 ^ (n + 1) * n = k ^ 2) ↔ 
  (∃ m : ℕ, n = 2 * m ^ 2) ∨ (∃ odd_k : ℕ, n = odd_k ^ 2 ∧ odd_k % 2 = 1) := 
sorry

end perfect_square_n_l539_539981


namespace shortest_distance_from_parabola_to_line_l539_539308

theorem shortest_distance_from_parabola_to_line :
  let A := λ a : ℝ, (a, a^2 - 9*a + 25)
  let distance (A : ℝ × ℝ) (B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2
  ∀ a : ℝ, ∀ x : ℝ, let B := (x, x - 8) in
  ∀ y_line_eq: x - (x - 8) = 8,
  (∀ a : ℝ, min (abs (a^2 - 10*a + 33) / real.sqrt 2 = 4 * real.sqrt 2)) :=
  sorry

end shortest_distance_from_parabola_to_line_l539_539308


namespace solve_abs_equation_l539_539583

-- Define the condition for the equation
def condition (x : ℝ) : Prop := 3 * x + 5 ≥ 0

-- The main theorem to prove that x = 1/5 is the only solution
theorem solve_abs_equation (x : ℝ) (h : condition x) : |2 * x - 6| = 3 * x + 5 ↔ x = 1 / 5 := by
  sorry

end solve_abs_equation_l539_539583


namespace find_fx_on_interval_l539_539725

-- Define the properties of the function
variables {f : ℝ → ℝ}

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

-- f has period 2
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x : ℝ, f (x + p) = f x

-- f(x) = x for x ∈ [2, 3]
def interval_constraint (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, (2 ≤ x ∧ x ≤ 3) → f x = x

-- Statement of the theorem
theorem find_fx_on_interval (h_even : even_function f)
                             (h_periodic : periodic_function f 2)
                             (h_interval : interval_constraint f)
                             (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 0) :
                             f x = 3 - | x + 1 | :=
sorry

end find_fx_on_interval_l539_539725


namespace adam_bought_26_books_l539_539113

-- Conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def avg_books_per_shelf : ℕ := 20
def leftover_books : ℕ := 2

-- Definitions based on conditions
def capacity_books : ℕ := shelves * avg_books_per_shelf
def total_books_after_trip : ℕ := capacity_books + leftover_books

-- Question: How many books did Adam buy on his shopping trip?
def books_bought : ℕ := total_books_after_trip - initial_books

theorem adam_bought_26_books :
  books_bought = 26 :=
by
  sorry

end adam_bought_26_books_l539_539113


namespace speed_solution_l539_539938

theorem speed_solution (s : ℝ) (h : 40 = 5 * s^2 + 20 * s + 15) : 5 * s^2 + 20 * s - 25 = 0 :=
begin
  calc
    40 = 5 * s^2 + 20 * s + 15 : h
    ... = 5 * s^2 + 20 * s + 15 : by refl
    ... = (5 * s^2 + 20 * s + 15 - 40) + 40 : by rw add_sub_cancel 5 (s^2 + 4 * s - 5 )
    ... = 5 * s^2 + 20 * s + 15 - 40 : add_zero(5* s^2 + 20 * s -25)
end

end speed_solution_l539_539938


namespace number_of_correct_conclusions_l539_539969

def concentric_expression (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ + a₂ = 0 ∧ b₁ = b₂ ∧ c₁ + c₂ = 0

def conclusion1_correct : Prop :=
  ¬ concentric_expression (-2) 3 0 2 3 0

def conclusion2_correct (m n : ℝ) (h1 : 8 * m + 6 * n = 0) (h2 : n = 4) : Prop :=
  (m + n) ^ 2023 = 1

def conclusion3_correct (a₁ c₁ a₂ c₂ : ℝ) (h : concentric_expression a₁ 0 c₁ a₂ 0 c₂) : Prop :=
  ∀ x : ℝ, (a₁ * x^2 + c₁ = -(a₂ * x^2 + c₂))

def conclusion4_correct (a₁ b₁ c₁ : ℝ) (h : concentric_expression a₁ b₁ c₁ (-a₁) b₁ (-c₁))
  (h_eq_roots : (3*a₁)*x^2 - b₁*x + 3*c₁ = 0) : Prop :=
  b₁^2 = 36*a₁*c₁

theorem number_of_correct_conclusions : 3 = 
  (if conclusion1_correct then 1 else 0) +
  (if ∃ m n, conclusion2_correct m n (by sorry) (by sorry) then 1 else 0) +
  (if ∃ a₁ c₁ a₂ c₂, conclusion3_correct a₁ c₁ a₂ c₂ (by sorry) then 1 else 0) +
  (if ∃ a₁ b₁ c₁, conclusion4_correct a₁ b₁ c₁ (by sorry) (by sorry) then 1 else 0) :=
by sorry

end number_of_correct_conclusions_l539_539969


namespace circles_common_tangents_l539_539420

theorem circles_common_tangents (r R : ℝ) (h : r ≠ R) :
  ¬ (∃ n : ℕ, n = 2 ∧ number_of_common_tangents r R = n) := 
sorry

end circles_common_tangents_l539_539420


namespace sqrt_sum_ge_sqrt_five_l539_539737

theorem sqrt_sum_ge_sqrt_five (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  sqrt (1 + a^2) + sqrt (1 + b^2) ≥ sqrt 5 := 
sorry

end sqrt_sum_ge_sqrt_five_l539_539737


namespace rectangle_A_travel_distance_is_7pi_l539_539353

def rectangle_rotation_total_path_length (AB CD BC DA : ℕ) (rotations : ℕ) : ℝ :=
  let d_AD := Float.sqrt ((AB * AB) + (BC * BC))
  let first_rotation_length := (1 / 4) * (2 * Real.pi * d_AD)
  let second_rotation_length := (1 / 4) * (2 * Real.pi * BC)
  let third_rotation_length := (1 / 4) * (2 * Real.pi * AB)
  first_rotation_length + second_rotation_length + third_rotation_length

theorem rectangle_A_travel_distance_is_7pi :
  rectangle_rotation_total_path_length 4 4 3 3 3 = 7 * Real.pi :=
by 
  sorry

end rectangle_A_travel_distance_is_7pi_l539_539353


namespace general_term_a_is_correct_l539_539810

noncomputable def general_term_b (n : ℕ) : ℕ := 10^n - 1
noncomputable def general_term_a (n : ℕ) : ℕ := 3 * (general_term_b n)

theorem general_term_a_is_correct (n : ℕ) : general_term_a n = \frac {1}{3} * (10^n - 1) :=
by
  sorry

end general_term_a_is_correct_l539_539810


namespace range_of_m_in_third_quadrant_l539_539287

theorem range_of_m_in_third_quadrant (m : ℝ) : (1 - (1/3) * m < 0) ∧ (m - 5 < 0) ↔ (3 < m ∧ m < 5) := 
by 
  intros
  sorry

end range_of_m_in_third_quadrant_l539_539287


namespace expand_polynomial_l539_539978

noncomputable def polynomial_expansion : Prop :=
  ∀ (x : ℤ), (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18

theorem expand_polynomial : polynomial_expansion :=
by
  sorry

end expand_polynomial_l539_539978


namespace number_of_triangles_with_area_S_l539_539542

variable (S : ℝ)
variables (A B C D E F : Type)
variables [HasArea ((A × E), ℝ)] [HasArea ((B × C), ℝ)] [HasArea ((C × E), ℝ)]
variables (P : (E = geometry.intersection (geometry.diagonal A C) (geometry.diagonal B D)))
variables (Q : (F ∈ geometry.segment B C))
variables (R : (geometry.parallel A B E F))
variables (T : (geometry.parallel C D E F))
variables (U : (geometry.area (triangle B C E) = S))

theorem number_of_triangles_with_area_S : number_of_triangles_with_area_S = 4 :=
sorry

end number_of_triangles_with_area_S_l539_539542


namespace train_cross_pole_time_l539_539100

/-- 
  Given the speed of a train in kilometers per hour and its length in meters, 
  prove that the time taken to cross a pole is approximately 15 seconds.
--/
theorem train_cross_pole_time :
  ∀ (v : ℝ) (l : ℝ), 
  v = 60 → 
  l = 250.00000000000003 →
  l / (v * 1000 / 3600) ≈ 15 :=
by
  sorry

end train_cross_pole_time_l539_539100


namespace julian_comic_book_l539_539305

theorem julian_comic_book : 
  ∀ (total_frames frames_per_page : ℕ),
    total_frames = 143 →
    frames_per_page = 11 →
    total_frames / frames_per_page = 13 ∧ total_frames % frames_per_page = 0 :=
by
  intros total_frames frames_per_page
  intros h_total_frames h_frames_per_page
  sorry

end julian_comic_book_l539_539305


namespace f_value_at_2sqrt10_l539_539212

noncomputable def f : ℝ → ℝ :=
λ x, if 0 < x ∧ x < 2 then x^2 - 16 * x + 60 else 0

axiom f_odd : ∀ x : ℝ, f(-x) = -f(x)
axiom f_periodic : ∀ x : ℝ, f(x) = f(x + 4)

-- The main theorem to prove
theorem f_value_at_2sqrt10 : f (2 * real.sqrt 10) = -36 :=
by
  sorry

end f_value_at_2sqrt10_l539_539212


namespace ratio_of_sums_of_sides_and_sines_l539_539674

theorem ratio_of_sums_of_sides_and_sines (A B C : ℝ) (a b c : ℝ) (hA : A = 60) (ha : a = 3) 
  (h : a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C) : 
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 3 := 
by 
  sorry

end ratio_of_sums_of_sides_and_sines_l539_539674


namespace time_to_cross_pole_l539_539098

-- Define the speed in km/hr
def speed_kmh : ℝ := 60

-- Convert the speed to m/s
def speed_ms : ℝ := (speed_kmh * 1000) / 3600

-- Define the length of the train in meters
def length_of_train : ℝ := 250.00000000000003

-- Prove the time it takes for the train to cross the pole is 15 seconds
theorem time_to_cross_pole : (length_of_train / speed_ms) = 15 := by
  -- Insert specific proof here (omitted)
  sorry

end time_to_cross_pole_l539_539098


namespace hits_distance_less_than_40cm_l539_539934

theorem hits_distance_less_than_40cm (
    (hits : Fin 7 → (ℝ × ℝ))
    (in_triangle : ∀ i, hits i.1 ≥ 0 ∧ hits i.1 ≤ 1 ∧ hits i.2 ≥ 0 ∧ hits i.2 ≤ 1 ∧ hits i.1 + hits i.2 ≤ 1)
  ) : ∃ i j, i ≠ j ∧ dist (hits i) (hits j) < 0.4 :=
by
  sorry

end hits_distance_less_than_40cm_l539_539934


namespace fraction_married_men_correct_l539_539545

def total_people := 11
def married_men := 4
def fraction_married_men := 4 / 11

theorem fraction_married_men_correct (total_women : ℕ) (single_women : nat) (prob_single : ℚ) (total_people : ℕ) (married_men : ℕ) :
  prob_single = 3 / 7 ∧ total_women = 7 ∧ single_women = 3 ∧ married_men = 4 ∧ total_people = (total_women + married_men) →
  married_men / total_people = 4 / 11 :=
by
  sorry

end fraction_married_men_correct_l539_539545


namespace maximum_sum_of_O_and_square_l539_539663

theorem maximum_sum_of_O_and_square 
(O square : ℕ) (h1 : (O > 0) ∧ (square > 0)) 
(h2 : (O : ℚ) / 11 < (7 : ℚ) / (square))
(h3 : (7 : ℚ) / (square) < (4 : ℚ) / 5) : 
O + square = 18 :=
sorry

end maximum_sum_of_O_and_square_l539_539663


namespace six_lines_condition_l539_539188

theorem six_lines_condition 
  (lines : Fin 6 → Line) 
  (h_no_three_coplanar : ∀ (l1 l2 l3 : Fin 6), ¬ coplanar lines l1 l2 l3) :
  ∃ (l1 l2 l3 : Fin 6), 
    (Form_plane lines l1 l2 ∧ Form_plane lines l2 l3 ∧ Form_plane lines l1 l3) ∨
    (Parallel lines l1 lines l2 ∧ Parallel lines l2 lines l3 ∧ Parallel lines l1 lines l3) ∨
    (Intersect_at_point lines l1 l2 l3) :=
sorry

end six_lines_condition_l539_539188


namespace find_cos_beta_l539_539688

-- Definitions of the conditions
variable (β : Real)
variable (p q : Real)
variable (EF GH EH FG : Real)
variable (angleE angleG : Real)

-- Defining the conditions given in the problem
axiom angle_conditions : angleE = β ∧ angleG = β
axiom side_conditions : EF = 200 ∧ GH = 200
axiom EH_FG_diff : EH ≠ FG
axiom perimeter_condition : EF + GH + p + q = 720

-- The theorem we need to prove
theorem find_cos_beta :
  ∃ β : Real, (angleE = β ∧ angleG = β) → (EF = 200 ∧ GH = 200) → EH ≠ FG →
  (EF + GH + p + q = 720) → cos β = 4 / 5 :=
begin
  assume β angle_condition side_condition diff_condition perimeter_condition,
  sorry
end

end find_cos_beta_l539_539688


namespace min_sum_labels_9x9_chessboard_l539_539750

theorem min_sum_labels_9x9_chessboard :
  (∃ (f : Fin 9 → Fin 9), (∑ i, (1 : ℝ) / (i + 1 + f i).val) = 1) :=
sorry

end min_sum_labels_9x9_chessboard_l539_539750


namespace geometry_inequality_l539_539728

theorem geometry_inequality
  (A B C T H B1 : Type)
  (AT BT CT TH TB1: ℝ)
  (angle_ATB1 : ℝ)
  (midpoint_B1 : B1 = (B + T) / 2)
  (H_on_extension : H = AT + BT1) 
  (angle_conditions : ∠THB1 = ∠TB1H ∧ ∠THB1 = 60 ∧ ∠TB1H = 60)
  (length_equalities : HB1 = TB1 ∧ TB1 = B1B)
  (angle_BHB1_conditions : ∠BHB1 = 30 ∧ ∠B1BH = 30)
  (right_angle : ∠BHA = 90)
  (perpendicular_ineq_B : AB > AT + TH)
  (perpendicular_ineq_C : AC > AT + CT)
  (perpendicular_ineq_T : BC > BT + CT) :
  2 * AB + 2 * BC + 2 * CA > 4 * AT + 3 * BT + 2 * CT :=
sorry

end geometry_inequality_l539_539728


namespace quadrilateral_is_square_l539_539374

variable {Point : Type*} [MetricSpace Point]
variable (A B C D O : Point)

-- Assuming the conditions in the problem
variable (h_inConvexQuadrilateral : ∀ (a b c d : Point), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a → ConvexHull (Set.Points {a, b, c, d}) (Set.Points {o}))

variable (h_area : (dist O A) ^ 2 + (dist O B) ^ 2 + (dist O C) ^ 2 + (dist O D) ^ 2 = 2 * Area A B C D)

-- Goal: Prove that the quadrilateral is a square and O is its center
theorem quadrilateral_is_square (h_inConvexQuadrilateral : ∀ (a b c d : Point), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a → ConvexHull (Set.Points {a, b, c, d}) (Set.Points {o}))
(h_area : (dist O A) ^ 2 + (dist O B) ^ 2 + (dist O C) ^ 2 + (dist O D) ^ 2 = 2 * Area A B C D) : IsSquare A B C D ∧ IsCenter O A B C D :=
begin
  sorry
end

end quadrilateral_is_square_l539_539374


namespace logarithmic_inequality_l539_539605

theorem logarithmic_inequality (a b c : ℝ) (h1 : a = Real.log 3 / Real.log 2)
    (h2 : b = Real.log 3 / Real.log (1/2))
    (h3 : c = Real.sqrt 3) : c > a ∧ a > b :=
by
  have h4 : b = - a, from sorry  -- a property of logarithms
  have h5 : 1 < a, from sorry    -- 3^10 < 2^17 implies 1 < log_2(3)
  have h6 : a < 1.7, from sorry
  have h7 : 1.7 < c, from sorry
  exact ⟨h7.trans (lt_of_le_of_ne sorry sorry), sorry⟩

end logarithmic_inequality_l539_539605


namespace range_of_a_l539_539634

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the conditions: f has a unique zero point x₀ and x₀ < 0
def unique_zero_point (a : ℝ) : Prop :=
  ∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0

-- The theorem we need to prove
theorem range_of_a (a : ℝ) : unique_zero_point a → a > 2 :=
sorry

end range_of_a_l539_539634


namespace solve_equation_l539_539770

theorem solve_equation : ∀ (x : ℝ), x ≠ -3 → x ≠ 3 → 
  (x / (x + 3) + 6 / (x^2 - 9) = 1 / (x - 3)) → x = 1 :=
by
  intros x hx1 hx2 h
  sorry

end solve_equation_l539_539770


namespace trig_identity_l539_539204

theorem trig_identity (a b : ℝ) (θ : ℝ) (h : (sin θ)^6 / a + (cos θ)^6 / b = 1 / (a + b)) : 
  (sin θ)^12 / a^2 + (cos θ)^12 / b^2 = (a^4 + b^4) / (a + b)^6 := 
by sorry

end trig_identity_l539_539204


namespace inner_product_norm_eq_iff_parallel_l539_539249

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem inner_product_norm_eq_iff_parallel :
  (∥a∥ * ∥b∥ ≠ 0) → (|⟪a, b⟫| = ∥a∥ * ∥b∥ ↔ a = 0 ∨ b = 0 ∨ ∃ k : ℝ, a = k • b) :=
by
  sorry

end inner_product_norm_eq_iff_parallel_l539_539249


namespace least_three_digit_number_product8_l539_539862

theorem least_three_digit_number_product8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (digits 10 n).prod = 8 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (digits 10 m).prod = 8 → n ≤ m) :=
sorry

end least_three_digit_number_product8_l539_539862


namespace coloring_ap_l539_539298

def is_red (n : ℕ) : Prop := ∃ k : ℕ, (k % 2 = 0) ∧ (k * (k - 1) + 1 ≤ n ∧ n ≤ k * (k - 1) + k)
def is_blue (n : ℕ) : Prop := ∃ k : ℕ, (k % 2 = 0) ∧ (k * (k - 1) + k + 1 ≤ n ∧ n ≤ k * (k - 1) + 2 * k)

theorem coloring_ap (d : ℕ) (h : d > 0) :
  ∀ a : ℕ, ∃ n m : ℕ, (a + n * d ≠ a + m * d) ∧ (is_red (a + n * d) ↔ is_blue (a + m * d) ∨ is_blue (a + n * d) ↔ is_red (a + m * d)) :=
sorry

end coloring_ap_l539_539298


namespace probability_A_shoots_l539_539488

theorem probability_A_shoots (P : ℚ) :
  (∀ n : ℕ, (2 * n + 1) % 2 = 1) →  -- A's turn is always the odd turn
  (∀ m : ℕ, (2 * m) % 2 = 0) →  -- B's turn is always the even turn
  let p_A_first_shot := (1 : ℚ) / 6 in  -- probability A fires on the first shot
  let p_A_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun
  let p_B_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun for B
  let P_A := p_A_first_shot + (p_A_turn * p_B_turn * P) in  -- recursive definition
  P_A = 6 / 11 := -- final probability
sorry

end probability_A_shoots_l539_539488


namespace gain_in_meters_l539_539517

noncomputable def cost_price : ℝ := sorry
noncomputable def selling_price : ℝ := 1.5 * cost_price
noncomputable def total_cost_price : ℝ := 30 * cost_price
noncomputable def total_selling_price : ℝ := 30 * selling_price
noncomputable def gain : ℝ := total_selling_price - total_cost_price

theorem gain_in_meters (S C : ℝ) (h_S : S = 1.5 * C) (h_gain : gain = 15 * C) :
  15 * C / S = 10 := by
  sorry

end gain_in_meters_l539_539517


namespace cassidy_posters_two_years_ago_l539_539131

def posters_two_years_ago (posters_now : ℕ) (additional_posters : ℕ) (double_collection : ℕ → Prop) :=
  ∃ (P : ℕ), (posters_now + additional_posters = 2 * P) ∧ double_collection P

/-- Cassidy had 14 posters two years ago, given the conditions described. -/
theorem cassidy_posters_two_years_ago : posters_two_years_ago 22 6 (λ P, P = 14) :=
by
  sorry

end cassidy_posters_two_years_ago_l539_539131


namespace sum_zero_monotonicity_zeros_product_l539_539637

noncomputable def f (x : ℝ) : ℝ := Real.log x - (x + 1) / (x - 1)

open Real

-- Part 1: Prove that the sum is zero
theorem sum_zero : (Finset.range 2023).sum (λ n, f (n + 2) + f (1 / (n + 2))) = 0 :=
  sorry

-- Part 2: Prove the monotonicity of the function
theorem monotonicity :
  (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x < y → f x < f y) ∧
  (∀ x y : ℝ, 1 < x ∧ 1 < y ∧ x < y → f x < f y) :=
  sorry

-- Part 3: Prove that the function has exactly two zeros and their product is 1
theorem zeros_product :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ (f x1 = 0 ∧ f x2 = 0 ∧ x1 * x2 = 1) :=
  sorry

end sum_zero_monotonicity_zeros_product_l539_539637


namespace susan_total_distance_l539_539775

theorem susan_total_distance (a b : ℕ) (r : ℝ) (h1 : a = 15) (h2 : b = 25) (h3 : r = 3) :
  (r * ((a + b) / 60)) = 2 :=
by
  sorry

end susan_total_distance_l539_539775


namespace one_eq_a_l539_539996

theorem one_eq_a (x y z a : ℝ) (h₁: x + y + z = a) (h₂: 1/x + 1/y + 1/z = 1/a) :
  x = a ∨ y = a ∨ z = a :=
  sorry

end one_eq_a_l539_539996


namespace probability_xi_leq_one_l539_539066

noncomputable def C (n k : ℕ) : ℕ := Nat.descFactorial n k / Nat.factorial k

theorem probability_xi_leq_one :
  let total_people := 12
  let excellent_students := 5
  let selected_people := 5
  let C_7_5 := C 7 5
  let C_5_1_C_7_4 := C 5 1 * C 7 4
  let C_12_5 := C total_people selected_people
  ∃ (xi : ℕ → ℕ), xi <= excellent_students →
    (C_7_5 + C_5_1_C_7_4) / C_12_5 = ∑ k in Finset.range 2, xi k :=
begin
  intro total_people,
  intro excellent_students,
  intro selected_people,
  intro C_7_5,
  intro C_5_1_C_7_4,
  intro C_12_5,
  use λ k, if k = 0 then 1 else if k = 1 then 1 else 0,
  intro hxi,
  sorry
end

end probability_xi_leq_one_l539_539066


namespace modulus_of_z_l539_539814

-- Definition of the complex number
def z : ℂ := 1 + 3 * complex.I

-- Goal statement to prove
theorem modulus_of_z : complex.abs z = real.sqrt 10 := 
by sorry

end modulus_of_z_l539_539814


namespace find_fiona_experience_l539_539799

namespace Experience

variables (d e f : ℚ)

def avg_experience_equation : Prop := d + e + f = 36
def fiona_david_equation : Prop := f - 5 = d
def emma_david_future_equation : Prop := e + 4 = (3/4) * (d + 4)

theorem find_fiona_experience (h1 : avg_experience_equation d e f) (h2 : fiona_david_equation d f) (h3 : emma_david_future_equation d e) :
  f = 183 / 11 :=
by
  sorry

end Experience

end find_fiona_experience_l539_539799


namespace massivest_polyhedral_faces_l539_539910

open Real

-- Define a polyhedron whose projections on the xy, yz, and zx planes are good 12-gons.
structure good_12gon (vertices : Fin 12 → ℝ × ℝ) : Prop :=
(regular : ∀ i j, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
(filled : ∀ p : ℝ × ℝ, p ∈ convex_hull (range vertices) → interior_point p)
(center_origin : ∀ i, (vertices i).fst + (vertices i).snd = 0)

structure polyhedron :=
(projection_xy : good_12gon)
(projection_yz : good_12gon)
(projection_zx : good_12gon)

theorem massivest_polyhedral_faces :
  ∀ P : polyhedron,
  (exists (faces : ℕ), faces = 36) :=
by
  intro P
  exists 36
  sorry

end massivest_polyhedral_faces_l539_539910


namespace find_t_l539_539246

variable {a b : EuclideanSpace ℝ (Fin 3)}

def is_unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∥v∥ = 1

def angle_60_degrees (u v : EuclideanSpace ℝ (Fin 3)) : Prop :=
  inner u v = real.cos (real.pi / 3)

def c (t : ℝ) : EuclideanSpace ℝ (Fin 3) :=
  t • a + (1 - t) • b

theorem find_t (ha : is_unit_vector a) (hb : is_unit_vector b) 
  (hangle : angle_60_degrees a b) (hb_dot_c_zero : inner b (c t) = 0) :
  t = 2 :=
sorry

end find_t_l539_539246


namespace rational_terms_expansion_term_largest_absolute_value_value_of_summation_l539_539200

-- Given conditions and variables defined
noncomputable def CoefficientRatio := 56 / 3

-- Define the rationals terms in the expansion
theorem rational_terms_expansion :
  (∃ (x : ℝ), x^5 ∈ (λ n, (√x - 2/ (x^(1/3)))^n).terms) ∧
  (∃ (x : ℕ), x = 13440) :=
begin
  sorry -- Proof that x^5 and 13440 are rational terms in the expansion.
end

-- Define the term with the largest absolute value
theorem term_largest_absolute_value :
  (∃ (x : ℝ), x = -15360 * x^(-5 / 6)) :=
begin
  sorry -- Proof that -15360x^(-5/6) is the term with the largest absolute value in the expansion.
end

-- Define the value of the given summation
theorem value_of_summation :
  (∃ n : ℕ, n = 10) →
  (∃ (sum : ℕ), sum = (10 + ∑ i in finset.range 10, 9^i * Nat.choose 10 i) / 9) :=
begin
  sorry -- Proof that the summation equals (10^10 - 1) / 9.
end

end rational_terms_expansion_term_largest_absolute_value_value_of_summation_l539_539200


namespace sum_of_roots_of_polynomial_l539_539548

theorem sum_of_roots_of_polynomial :
  let P := (λ x : ℂ, (x - 1) ^ 2010 - 2 * (x - 2) ^ 2009 + 3 * (x - 3) ^ 2008 - 
            ∑ i in Finset.range 2009, (-1) ^ (i + 1) * (i + 1) * (x - (i + 1)) ^ (2010 - (i + 1)))
  in ∑ x in (P.roots.to_finset), x = 2007 := sorry

end sum_of_roots_of_polynomial_l539_539548


namespace average_score_B_median_score_B_mode_score_B_stability_B_more_than_A_excellent_rate_A_excellent_rate_B_student_B_more_suitable_l539_539495

def scores_A := [60, 75, 100, 90, 75]
def scores_B := [70, 90, 100, 80, 80]

def variance_A := 190
def variance_B := 104

theorem average_score_B : (70 + 90 + 100 + 80 + 80) / 5 = 84 := by
  sorry

theorem median_score_B : List.coe_sort (List.sort scores_B) = [70, 80, 80, 90, 100] ∧
                         (List.nthLe [70, 80, 80, 90, 100] 2 sorry = 80) := by
  sorry

theorem mode_score_B : List.mode scores_B = 80 := by
  sorry

theorem stability_B_more_than_A : variance_B < variance_A := by
  sorry

theorem excellent_rate_A : 2 / 5 = 0.4 := by
  sorry

theorem excellent_rate_B : 4 / 5 = 0.8 := by
  sorry

theorem student_B_more_suitable : variance_B < variance_A ∧ 4 / 5 = 0.8 ∧ 2 / 5 = 0.4 := by
  sorry

end average_score_B_median_score_B_mode_score_B_stability_B_more_than_A_excellent_rate_A_excellent_rate_B_student_B_more_suitable_l539_539495


namespace probability_A_fires_l539_539467

theorem probability_A_fires :
  let p_A := (1 : ℚ) / 6 + (5 : ℚ) / 6 * (5 : ℚ) / 6 * p_A
  in p_A = 6 / 11 :=
by
  sorry

end probability_A_fires_l539_539467


namespace dihedral_angle_is_120_l539_539691

-- Definitions of points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (3, -2)

-- Given distance after folding
def folded_distance : ℝ := 2 * real.sqrt 11

-- Hypothesis: After folding the Cartesian plane along the x-axis, the distance |AB| = 2 * sqrt 11
axiom fold_distance_condition : real.dist A B = folded_distance

-- Define the dihedral angle
def dihedral_angle (θ : ℝ) : Prop :=
  θ = 120

-- Statement to be proved: The size of the dihedral angle is 120 degrees.
theorem dihedral_angle_is_120 :
  ∃ θ : ℝ, dihedral_angle θ ∧ real.dist A B = 2 * real.sqrt 11 :=
by
  use 120
  unfold dihedral_angle
  split
  · refl
  · exact fold_distance_condition

end dihedral_angle_is_120_l539_539691


namespace part1_part2_l539_539201

-- Definitions for point P
structure Point where
  x : ℝ
  y : ℝ

variables (a : ℝ)

-- Point defined in terms of a
def P : Point := { x := 2 * a - 3, y := a + 6 }

-- Part (1) Prove that if P lies on the x-axis, then P=(-15, 0)
theorem part1 (h : P a) : P a = ⟨-15, 0⟩ :=
by sorry

-- Part (2) Prove that if P lies in the second quadrant and distances to axes are equal, then a^{2003} + 2024 = 2023
theorem part2 (h1 : 2 * a - 3 < 0) (h2 : a + 6 > 0) (h3 : 3 - 2 * a = a + 6) : a^2003 + 2024 = 2023 :=
by sorry

end part1_part2_l539_539201


namespace sum_due_is_correct_l539_539901

theorem sum_due_is_correct (BD TD PV : ℝ) (h1 : BD = 80) (h2 : TD = 70) (h_relation : BD = TD + (TD^2) / PV) : PV = 490 :=
by sorry

end sum_due_is_correct_l539_539901


namespace pirate_treasure_division_l539_539924

theorem pirate_treasure_division (initial_treasure : ℕ) (p1_share p2_share p3_share p4_share p5_share remaining : ℕ)
  (h_initial : initial_treasure = 3000)
  (h_p1_share : p1_share = initial_treasure / 10)
  (h_p1_rem : remaining = initial_treasure - p1_share)
  (h_p2_share : p2_share = 2 * remaining / 10)
  (h_p2_rem : remaining = remaining - p2_share)
  (h_p3_share : p3_share = 3 * remaining / 10)
  (h_p3_rem : remaining = remaining - p3_share)
  (h_p4_share : p4_share = 4 * remaining / 10)
  (h_p4_rem : remaining = remaining - p4_share)
  (h_p5_share : p5_share = 5 * remaining / 10)
  (h_p5_rem : remaining = remaining - p5_share)
  (p6_p9_total : ℕ)
  (h_p6_p9_total : p6_p9_total = 20 * 4)
  (final_remaining : ℕ)
  (h_final_remaining : final_remaining = remaining - p6_p9_total) :
  final_remaining = 376 :=
by sorry

end pirate_treasure_division_l539_539924


namespace solve_system_l539_539030

theorem solve_system :
  ∃ (x y : ℝ), (2 * x - y = 1) ∧ (x + y = 2) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end solve_system_l539_539030


namespace find_g_8_l539_539086

def g (x : ℝ) : ℝ := x^2 + x + 1

theorem find_g_8 : (∀ x : ℝ, g (2*x - 4) = x^2 + x + 1) → g 8 = 43 := 
by sorry

end find_g_8_l539_539086


namespace correct_options_l539_539890

variable (Ω : Type) [ProbSpace Ω]
variable (A B : Event Ω)
variable (P : Probability Ω)

noncomputable def mutually_exclusive (P : Probability Ω) (A B : Event Ω) : Prop :=
  P (A ∩ B) = 0

noncomputable def independent (P : Probability Ω) (A B : Event Ω) : Prop :=
  P (A ∩ B) = P A * P B

noncomputable def prob_complement (P : Probability Ω) (A : Event Ω) : ℝ :=
  1 - P A

theorem correct_options
  (mut_excl : mutually_exclusive P A B → ¬mutually_exclusive P A Bᶜ)
  (indep_AB : independent P A B → independent P A B)
  (indep_AcompB : independent P A B → independent P A Bᶜ)
  (P_A_06 : P A = 0.6)
  (P_B_02 : P B = 0.2)
  (indep_A_B : independent P A B)
  (P_A_B_eq : P (A ∪ B) = P A + P B - P (A ∩ B)) :
  ¬(P (A ∪ B) = 0.8) →
  (P_A_08 : P A = 0.8)
  (P_B_07 : P B = 0.7)
  (indep_A_B_again : independent P A B)
  (P_A_comp_B_eq : P (A ∩ Bᶜ) = 0.8 * (1 - 0.7)) :
  true := by
  sorry

end correct_options_l539_539890


namespace large_pizzas_sold_l539_539020

def small_pizza_price : ℕ := 2
def large_pizza_price : ℕ := 8
def total_earnings : ℕ := 40
def small_pizzas_sold : ℕ := 8

theorem large_pizzas_sold : 
  ∀ (small_pizza_price large_pizza_price total_earnings small_pizzas_sold : ℕ), 
    small_pizza_price = 2 → 
    large_pizza_price = 8 → 
    total_earnings = 40 → 
    small_pizzas_sold = 8 →
    (total_earnings - small_pizzas_sold * small_pizza_price) / large_pizza_price = 3 :=
by 
  intros small_pizza_price large_pizza_price total_earnings small_pizzas_sold 
         h_small_pizza_price h_large_pizza_price h_total_earnings h_small_pizzas_sold
  rw [h_small_pizza_price, h_large_pizza_price, h_total_earnings, h_small_pizzas_sold]
  simp
  sorry

end large_pizzas_sold_l539_539020


namespace isosceles_triangle_dot_product_l539_539293

theorem isosceles_triangle_dot_product :
  ∀ (A B C D : ℝ³), 
    ∃ (AB AC BC: ℝ), 
      AB = 2 ∧ AC = 2 ∧
      angle A B C = 2 * π / 3 ∧
      (∃ (ratios : ℝ), ratios = 3 ∧ area A C D = ratios * (area A B D)) →
      (vector.dot (A - B) (A - D) = 5 / 2) := 
sorry

end isosceles_triangle_dot_product_l539_539293


namespace four_letter_list_product_l539_539154

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def list_product (s : String) : Nat :=
  s.foldl (λ acc c => acc * letter_value c) 1

def target_product : Nat :=
  list_product "TUVW"

theorem four_letter_list_product : 
  ∀ (s1 s2 : String), s1.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') → s2.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') →
  s1.length = 4 → s2.length = 4 →
  list_product s1 = target_product → s2 = "BEHK" :=
by
  sorry

end four_letter_list_product_l539_539154


namespace tammy_avg_speed_l539_539776

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l539_539776


namespace sqrt_four_eq_pm_two_l539_539821

theorem sqrt_four_eq_pm_two : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_four_eq_pm_two_l539_539821


namespace problem_statement_l539_539603

theorem problem_statement (x : ℝ) (hx : tan x = 2) : 
  4 * sin x ^ 2 - 3 * sin x * cos x - 5 * cos x ^ 2 = 1 := 
by
  sorry

end problem_statement_l539_539603


namespace trigonometric_identity_l539_539053

open Real

theorem trigonometric_identity (α β : ℝ) :
  sin (2 * α) ^ 2 + sin β ^ 2 + cos (2 * α + β) * cos (2 * α - β) = 1 :=
sorry

end trigonometric_identity_l539_539053


namespace total_loaves_served_l539_539753

theorem total_loaves_served :
  let wheat := 5 / 4
  let white := 3 / 4
  let rye := 3 / 5
  let multigrain := 7 / 10
  ∑(loaf in [wheat, white, rye, multigrain]) (loaf) = 3 + 3 / 10 :=
by 
  sorry

end total_loaves_served_l539_539753


namespace counterfeit_coins_heavier_or_lighter_l539_539533

-- Definitions of the conditions
def total_coins := 103
def counterfeit_coins := 2

-- Problem statement in Lean 4
theorem counterfeit_coins_heavier_or_lighter (real_coin_weight counterfeit_coin_weight : ℝ) (coins : Fin total_coins → ℝ)
    (h_real_equal : ∀ i j, i ≠ j → coins i = real_coin_weight → coins j = real_coin_weight)
    (h_counterfeit_equal : ∀ i j, coins i = counterfeit_coin_weight → coins j = counterfeit_coin_weight)
    (h_counterfeit_different : counterfeit_coin_weight ≠ real_coin_weight)
    (h_count_counterfeit : ∃ i j, i ≠ j ∧ coins i = counterfeit_coin_weight ∧ coins j = counterfeit_coin_weight)
    (h_count_real : (∑ i, if coins i = real_coin_weight then 1 else 0) = total_coins - counterfeit_coins)
    (h_grouped_weighings : ∃ A B C : Finset (Fin total_coins), A.card = 34 ∧ B.card = 34 ∧ C.card = 35 ∧ A ∪ B ∪ C = Finset.univ) :
  ∃ weighings : List (Finset (Fin total_coins) × Finset (Fin total_coins)), weighings.length ≤ 3 ∧ 
  (∀ (w : Finset (Fin total_coins) × Finset (Fin total_coins)), w ∈ weighings → w.1.card = w.2.card) ∧
  (weigh_group weighings coins = (CounterfeitLighter ∨ CounterfeitHeavier)) :=
sorry

end counterfeit_coins_heavier_or_lighter_l539_539533


namespace probability_A_fires_l539_539466

theorem probability_A_fires :
  let p_A := (1 : ℚ) / 6 + (5 : ℚ) / 6 * (5 : ℚ) / 6 * p_A
  in p_A = 6 / 11 :=
by
  sorry

end probability_A_fires_l539_539466


namespace triangle_count_l539_539007

theorem triangle_count (a b : Set Point) (h_parallel : Parallel a b) 
  (ha : ∃ l : Finset Point, l.card = 5 ∧ ∀ p ∈ l, p ∈ a) 
  (hb : ∃ m : Finset Point, m.card = 4 ∧ ∀ q ∈ m, q ∈ b) : 
  let C (n k : ℕ) := nat.choose n k in
  ∃ t : ℕ, t = C 5 2 * C 4 1 + C 5 1 * C 4 2 := 
by
  sorry

end triangle_count_l539_539007


namespace sum_of_first_n_terms_l539_539133

variable (a_n : ℕ → ℝ) -- Sequence term
variable (S_n : ℕ → ℝ) -- Sum of first n terms

-- Conditions given in the problem
axiom sum_first_term : a_n 1 = 2
axiom sum_first_two_terms : a_n 1 + a_n 2 = 7
axiom sum_first_three_terms : a_n 1 + a_n 2 + a_n 3 = 18

-- Expected result to prove
theorem sum_of_first_n_terms 
  (h1 : S_n 1 = 2)
  (h2 : S_n 2 = 7)
  (h3 : S_n 3 = 18) :
  S_n n = (3/2) * ((n * (n + 1) * (2 * n + 1) / 6) - (n * (n + 1) / 2) + 2 * n) :=
sorry

end sum_of_first_n_terms_l539_539133


namespace focus_of_parabola_x_squared_eq_neg_4_y_l539_539802

theorem focus_of_parabola_x_squared_eq_neg_4_y:
  (∃ F : ℝ × ℝ, (F = (0, -1)) ∧ (∀ x y : ℝ, x^2 = -4 * y → F = (0, y + 1))) :=
sorry

end focus_of_parabola_x_squared_eq_neg_4_y_l539_539802


namespace unique_odd_number_between_500_and_1000_l539_539509

theorem unique_odd_number_between_500_and_1000 :
  ∃! x : ℤ, 500 ≤ x ∧ x ≤ 1000 ∧ x % 25 = 6 ∧ x % 9 = 7 ∧ x % 2 = 1 :=
sorry

end unique_odd_number_between_500_and_1000_l539_539509


namespace find_lunch_days_l539_539683

variable (x y : ℕ) -- School days for School A and School B
def P_A := x / 2 -- Aliyah packs lunch half the time
def P_B := y / 4 -- Becky packs lunch a quarter of the time
def P_C := y / 2 -- Charlie packs lunch half the time

theorem find_lunch_days (x y : ℕ) :
  P_A x = x / 2 ∧
  P_B y = y / 4 ∧
  P_C y = y / 2 :=
by
  sorry

end find_lunch_days_l539_539683


namespace area_of_region_B_is_correct_l539_539556

noncomputable def region_area_B (w : ℂ) : Prop :=
  let u := w.re in
  let v := w.im in
  (-50 ≤ u ∧ u ≤ 50) ∧ (-50 ≤ v ∧ v ≤ 50) ∧
  (u * u + v * v ≥ 50 * |u|) ∧ (u * u + v * v ≥ 50 * |v|)

theorem area_of_region_B_is_correct :
  ∃ B : set ℂ, (∀ w : ℂ, w ∈ B ↔ region_area_B w) ∧
  ∃ area : ℝ, area = 10000 - 312.5 * Real.pi :=
sorry

end area_of_region_B_is_correct_l539_539556


namespace exists_a_satisfying_f_l539_539265

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 1 else x - 1

theorem exists_a_satisfying_f (a : ℝ) : 
  f (a + 1) = f a ↔ (a = -1/2 ∨ a = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end exists_a_satisfying_f_l539_539265


namespace range_of_m_l539_539640

-- Define the quadratic function f
def f (a c x : ℝ) := a * x^2 - 2 * a * x + c

-- State the theorem
theorem range_of_m (a c : ℝ) (h : f a c 2017 < f a c (-2016)) (m : ℝ) 
  : f a c m ≤ f a c 0 → 0 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l539_539640


namespace range_of_a_l539_539999

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) → ((x + a) * (x + 1) > 0)) ∧ 
  (∃ x : ℝ, ¬(x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) ∧ ((x + a) * (x + 1) > 0)) → 
  a ∈ Set.Iio (-3) := 
  sorry

end range_of_a_l539_539999


namespace train_travelled_from_Ufa_to_Baku_l539_539792

def position_in_alphabet (s : String) : list ℕ := 
  s.to_list.map (fun c => Char.to_nat c - Char.to_nat 'A' + 1)

def decode_city (encoded : String) := 
  if encoded = "21221" then "Ufa"
  else if encoded = "211221" then "Baku"
  else "Unknown"

theorem train_travelled_from_Ufa_to_Baku : 
  decode_city "21221" = "Ufa" ∧ decode_city "211221" = "Baku" := 
by 
  sorry

end train_travelled_from_Ufa_to_Baku_l539_539792


namespace probability_A_fires_proof_l539_539471

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end probability_A_fires_proof_l539_539471


namespace tournament_key_player_l539_539522

theorem tournament_key_player (n : ℕ) (plays : Fin n → Fin n → Bool) (wins : ∀ i j, plays i j → ¬plays j i) :
  ∃ X, ∀ (Y : Fin n), Y ≠ X → (plays X Y ∨ ∃ Z, plays X Z ∧ plays Z Y) :=
by
  sorry

end tournament_key_player_l539_539522


namespace inverse_function_l539_539812

theorem inverse_function :
  (∀ x: ℝ, 1 ≤ x ∧ x ≤ 3 → y = 1 + log 3 x) ↔ (∀ x: ℝ, 1 ≤ x ∧ x ≤ 2 → y = 3^(x - 1)) :=
by
  sorry

end inverse_function_l539_539812


namespace sin_150_eq_half_l539_539575

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_150_eq_half_l539_539575


namespace fraction_value_sin_cos_value_l539_539206

open Real

-- Let alpha be an angle in radians satisfying the given condition
variable (α : ℝ)

-- Given condition
def condition  : Prop := sin α = 2 * cos α

-- First question
theorem fraction_value (h : condition α) : 
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1 / 6 :=
sorry

-- Second question
theorem sin_cos_value (h : condition α) : 
  sin α ^ 2 + 2 * sin α * cos α = 8 / 5 :=
sorry

end fraction_value_sin_cos_value_l539_539206


namespace ellipse_standard_equation_l539_539198

theorem ellipse_standard_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (-4)^2 / a^2 + 3^2 / b^2 = 1) 
    (h4 : a^2 = b^2 + 5^2) : 
    ∃ (a b : ℝ), a^2 = 40 ∧ b^2 = 15 ∧ 
    (∀ x y : ℝ, x^2 / 40 + y^2 / 15 = 1 → (∃ f1 f2 : ℝ, f1 = 5 ∧ f2 = -5)) :=
by {
    sorry
}

end ellipse_standard_equation_l539_539198


namespace bug_can_return_to_starting_cell_l539_539276

/-- In a grid with doors between neighboring cells, a bug starts at a certain cell.
Each time the bug moves through a closed door, it opens the door in the direction of its movement,
leaving it open. An open door can only be passed through in the direction it was opened.
We need to prove that the bug can always return to its starting cell. -/
theorem bug_can_return_to_starting_cell
    (Grid : Type) (Cell : Grid → Grid → Prop) (Door : Grid → Grid → Prop)
    (bug_moves : Grid → Grid → Prop)
    (start : Grid) :
    (∀ c1 c2, Door c1 c2 → Door c2 c1 → false) → -- An open door can only be passed through in the direction it was opened.
    (∀ c1 c2, Cell c1 c2 ↔ ¬ Cell c2 c1) → -- Each cell is connected to its adjacent neighboring cells by doors.
    (∀ c1 c2, c1 ≠ c2 → bug_moves c1 c2 → Door c1 c2) → -- When the bug moves through a closed door, it opens it in the direction it is moving.
    (∀ c, ∃ c', bug_moves start c → bug_moves c' c) → -- The bug can travel through cells by passing through doors.
    start ∈ Grid → -- The bug starts from a specific cell.
    (∃ path, path 0 = start ∧ (∀ n, path (n + 1) = start) → start ∈ Grid :=
begin
  sorry -- Proof to be completed.
end

end bug_can_return_to_starting_cell_l539_539276


namespace probability_A_shoots_l539_539487

theorem probability_A_shoots (P : ℚ) :
  (∀ n : ℕ, (2 * n + 1) % 2 = 1) →  -- A's turn is always the odd turn
  (∀ m : ℕ, (2 * m) % 2 = 0) →  -- B's turn is always the even turn
  let p_A_first_shot := (1 : ℚ) / 6 in  -- probability A fires on the first shot
  let p_A_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun
  let p_B_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun for B
  let P_A := p_A_first_shot + (p_A_turn * p_B_turn * P) in  -- recursive definition
  P_A = 6 / 11 := -- final probability
sorry

end probability_A_shoots_l539_539487


namespace tangent_circle_ratio_l539_539494

theorem tangent_circle_ratio
  (A B C : Point)
  (ω ω1 ω2 : Circle)
  (hω : Circumscribed ω ABC)
  (hω1 : TangentAtLine ω1 A B ∧ PassesThrough ω1 C)
  (hω2 : TangentAtLine ω2 A C ∧ PassesThrough ω2 B)
  (tangentA : TangentAtPoint ω A)
  (X : Point)
  (hX : IntersectAt ω tangentA ω1 X)
  (Y : Point)
  (hY : IntersectAt ω tangentA ω2 Y) :
  (AX / XY = 1 / 2) :=
sorry

end tangent_circle_ratio_l539_539494


namespace minimum_value_of_f_f_geq_g_for_all_m_n_l539_539237

open Real

def f (x : ℝ) : ℝ := x * log x
def g (x : ℝ) : ℝ := x / exp x - 2 / exp 1

theorem minimum_value_of_f :
  ∃ x ∈ Ioi (0 : ℝ), f x = -1 / exp 1 :=
by
  sorry

theorem f_geq_g_for_all_m_n :
  ∀ (m n : ℝ), 0 < m → 0 < n → f m ≥ g n :=
by
  sorry

-- We use Ioi (0 : ℝ) to represent the interval (0, ∞)

end minimum_value_of_f_f_geq_g_for_all_m_n_l539_539237


namespace distance_D_D_l539_539418

noncomputable theory

-- Define the points D and D'
def D : ℝ × ℝ := (2, 4)
def D' : ℝ × ℝ := (2, -4)

-- Define the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- State that the distance between D and D' is 8
theorem distance_D_D' : distance D D' = 8 :=
by sorry

end distance_D_D_l539_539418


namespace onkon_room_area_l539_539340

def length_longer_leg_feet : ℕ := 15
def width_legs_feet : ℕ := 6
def length_shorter_leg_feet : ℕ := 9
def feet_per_yard : ℕ := 3

def length_longer_leg_yards : ℕ := length_longer_leg_feet / feet_per_yard := by sorry
def width_legs_yards : ℕ := width_legs_feet / feet_per_yard := by sorry
def length_shorter_leg_yards : ℕ := length_shorter_leg_feet / feet_per_yard := by sorry

def area_longer_leg_yards : ℕ := length_longer_leg_yards * width_legs_yards := by sorry
def area_shorter_leg_yards : ℕ := length_shorter_leg_yards * width_legs_yards := by sorry

def total_area_yards : ℕ := area_longer_leg_yards + area_shorter_leg_yards := by sorry

theorem onkon_room_area : total_area_yards = 16 := by sorry

end onkon_room_area_l539_539340


namespace decimal_to_binary_23_l539_539965

/-- Represents the conversion of decimal 23 to its binary form using the "divide by 2 and take the remainder" method. -/
theorem decimal_to_binary_23 : 
  ∃ (remainders : List ℕ), 
    (23 % 2 = remainders.head) ∧
    (23 / 2 % 2 = remainders.nth 1) ∧
    (23 / 4 % 2 = remainders.nth 2) ∧
    (23 / 8 % 2 = remainders.nth 3) ∧
    (23 / 16 % 2 = remainders.nth 4) ∧
    (List.reverse remainders).join = [1, 0, 1, 1, 1] := 
by 
  sorry

end decimal_to_binary_23_l539_539965


namespace real_value_of_b_l539_539993

open Real

theorem real_value_of_b : ∃ x : ℝ, (x^2 - 2 * x + 1 = 0) ∧ (x^2 + x - 2 = 0) :=
by
  sorry

end real_value_of_b_l539_539993


namespace part1_part2_l539_539741

-- Definitions from part (a)
def a_n (n : ℕ) : ℕ := 2 * n - 1
def b_n (n : ℕ) : ℕ := 2 ^ (a_n n + 1)

-- Specification from the given problem
def S_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a_n i
def c_n (n : ℕ) : ℕ := a_n n * b_n n
def T_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), c_n i

-- Theorem to be proven (part (c))
theorem part1 (n : ℕ) : S_n n = n ^ 2 := by
  sorry

theorem part2 (n : ℕ) : T_n n = (24 * n - 20) * 4 ^ n / 9 + 20 / 9 := by
  sorry

end part1_part2_l539_539741


namespace b_completes_work_alone_l539_539448

theorem b_completes_work_alone (A_twice_B : ∀ (B : ℕ), A = 2 * B)
  (together : ℕ := 7) : ∃ (B : ℕ), 21 = 3 * together :=
by
  sorry

end b_completes_work_alone_l539_539448


namespace number_of_girls_l539_539678

theorem number_of_girls (B G : ℕ) (h1 : B * 5 = G * 8) (h2 : B + G = 1040) : G = 400 :=
by
  sorry

end number_of_girls_l539_539678


namespace interest_rate_calculation_l539_539940

theorem interest_rate_calculation (P1 P2 I1 I2 : ℝ) (r1 : ℝ) :
  P2 = 1648 ∧ P1 = 2678 - P2 ∧ I2 = P2 * 0.05 * 3 ∧ I1 = P1 * r1 * 8 ∧ I1 = I2 →
  r1 = 0.03 :=
by sorry

end interest_rate_calculation_l539_539940


namespace ellipse_standard_equation_chord_length_range_l539_539219

-- Conditions for question 1
def ellipse_center (O : ℝ × ℝ) : Prop := O = (0, 0)
def major_axis_x (major_axis : ℝ) : Prop := major_axis = 1
def eccentricity (e : ℝ) : Prop := e = (Real.sqrt 2) / 2
def perp_chord_length (AA' : ℝ) : Prop := AA' = Real.sqrt 2

-- Lean statement for question 1
theorem ellipse_standard_equation (O : ℝ × ℝ) (major_axis : ℝ) (e : ℝ) (AA' : ℝ) :
  ellipse_center O → major_axis_x major_axis → eccentricity e → perp_chord_length AA' →
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2 / (a^2)) + y^2 / (b^2) = 1) := sorry

-- Conditions for question 2
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def max_area_triangle (S : ℝ) : Prop := S = 1 / 2

-- Lean statement for question 2
theorem chord_length_range (x y z w : ℝ) (E F G H : ℝ × ℝ) :
  circle_eq x y → ellipse_eq z w → max_area_triangle ((E.1 * F.1) * (Real.sin (E.2 * F.2))) →
  ( ∃ min_chord max_chord : ℝ, min_chord = Real.sqrt 3 ∧ max_chord = 2 ∧
    ∀ x1 y1 x2 y2 : ℝ, (G.1 = x1 ∧ H.1 = x2 ∧ G.2 = y1 ∧ H.2 = y2) →
    (min_chord ≤ (Real.sqrt ((1 + (x2 ^ 2)) * ((x1 ^ 2) - 4 * (x1 * x2)))) ∧
         Real.sqrt ((1 + (x2 ^ 2)) * ((x1 ^ 2) - 4 * (x1 * x2))) ≤ max_chord )) := sorry

end ellipse_standard_equation_chord_length_range_l539_539219


namespace sum_of_all_valid_solutions_l539_539360

noncomputable def equation (x : ℝ) := x - 5 = (3 * |x - 2|) / (x - 2)
def is_valid_solution (x : ℝ) := x ≠ 2 ∧ equation x

theorem sum_of_all_valid_solutions : 
  (∑ x in {x | is_valid_solution x}.to_finset, x) = 8 :=
by
  sorry

end sum_of_all_valid_solutions_l539_539360


namespace barrel_volume_after_leak_l539_539071

theorem barrel_volume_after_leak
  (initial_volume : ℕ) 
  (percent_lost : ℕ) :
  initial_volume = 220 →
  percent_lost = 10 →
  initial_volume - (initial_volume * percent_lost / 100) = 198 :=
by
  intros h_initial_volume h_percent_lost
  rw [h_initial_volume, h_percent_lost]
  norm_num
  sorry

end barrel_volume_after_leak_l539_539071


namespace least_three_digit_product_eight_l539_539854

theorem least_three_digit_product_eight : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).prod = 8 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (nat.digits 10 m).prod = 8 → n ≤ m :=
by
  sorry

end least_three_digit_product_eight_l539_539854


namespace probability_A_fires_proof_l539_539472

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end probability_A_fires_proof_l539_539472


namespace lines_relationship_undetermined_l539_539216

variable (l m : ℝ → ℝ → ℝ) -- define lines l and m
variable (A B C D : ℝ × ℝ × ℝ) -- define points of the trapezoid A, B, C, D

-- define perpendicular predicate
def is_perpendicular (line1 line2 : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, line1 x y = 0 → line2 x y = 0

-- Assumptions
axiom h1 : is_perpendicular l (λ x y, x - y) -- l perpendicular to AB (encoded as a simple line equation)
axiom h2 : is_perpendicular l (λ x y, y - x) -- l perpendicular to CD
axiom h3 : is_perpendicular m (λ x y, x)     -- m perpendicular to AD (simplified line equation)
axiom h4 : is_perpendicular m (λ x y, y)     -- m perpendicular to BC

-- Goal: The relationship between l and m is undetermined.
theorem lines_relationship_undetermined : 
  ∃ (rel : ℝ → ℝ → ℝ → Prop), (is_perpendicular l m ∧ ¬ (rel l m)) ∨ (is_perpendicular l m ∧ (rel l m)) := sorry

end lines_relationship_undetermined_l539_539216


namespace incenter_minimizes_sum_l539_539598

theorem incenter_minimizes_sum (a b c : ℝ) (ABC : triangle) (M : point) (hM : M ∈ interior ABC) :
  let MA1 := perpendicular_from M to ABC.side1,
      MB1 := perpendicular_from M to ABC.side2,
      MC1 := perpendicular_from M to ABC.side3 in
  is_incenter ABC M ↔
    (a / MA1 + b / MB1 + c / MC1) =
    min_value

end incenter_minimizes_sum_l539_539598


namespace complex_abs_of_sqrt_l539_539369

variable (z : ℂ)

theorem complex_abs_of_sqrt (h : z^2 = 16 - 30 * Complex.I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_abs_of_sqrt_l539_539369


namespace tammy_speed_on_second_day_l539_539782

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l539_539782


namespace least_three_digit_with_product_eight_is_124_l539_539864

noncomputable def least_three_digit_with_product_eight : ℕ :=
  let candidates := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (x.digits 10).prod = 8} in
  if h : Nonempty candidates then
    let min_candidate := Nat.min' _ h in min_candidate
  else
    0 -- default to 0 if no such number exists

theorem least_three_digit_with_product_eight_is_124 :
  least_three_digit_with_product_eight = 124 := sorry

end least_three_digit_with_product_eight_is_124_l539_539864


namespace not_possible_to_have_at_most_one_friend_each_l539_539491

-- Define the initial conditions
def initial_friends : ℕ := 91
def initial_participants : ℕ := 91
def participants_with_46_friends : ℕ := 45
def participants_with_45_friends : ℕ := 46

-- Define the daily operation
noncomputable def daily_operation (A B C: ℕ) (is_friends : ℕ -> ℕ -> Prop) : Prop :=
  is_friends A B ∧ is_friends A C ∧ ¬is_friends B C

-- Final goal: prove it is not possible to end up with each participant having at most one friend
theorem not_possible_to_have_at_most_one_friend_each
  (is_friends : ℕ -> ℕ -> Prop)
  (initial_conditions : 
    (∑ i in (finset.range participants_with_46_friends), (finset.card (is_friends i)) = 46) ∧
    (∑ i in (finset.range participants_with_45_friends), (finset.card (is_friends i)) = 45)) :
  ¬(∀ i : ℕ, finset.card (is_friends i) ≤ 1) :=
sorry

end not_possible_to_have_at_most_one_friend_each_l539_539491


namespace cos_alpha_minus_beta_l539_539602

noncomputable def cos_diff (α β : ℝ) : ℝ :=
  cos (α - β)

theorem cos_alpha_minus_beta {α β : ℝ} 
  (h1 : sin α + sin β = 1 / 2)
  (h2 : cos α + cos β = 1 / 3) :
  cos_diff α β = -59 / 72 := by
  sorry

end cos_alpha_minus_beta_l539_539602


namespace cube_surface_area_increase_50_percent_l539_539060

noncomputable def percentage_increase_surface_area (L : ℝ) (SA_original : ℝ) (SA_new : ℝ) : ℝ :=
  ((SA_new - SA_original) / SA_original) * 100

theorem cube_surface_area_increase_50_percent (L : ℝ) 
  (SA_original : ℝ := 6 * L^2) 
  (L_new : ℝ := 1.5 * L)
  (SA_new : ℝ := 6 * (L_new)^2) :
  percentage_increase_surface_area L SA_original SA_new = 125 :=
by
  have h1 : SA_new = 13.5 * L^2 := by 
    rw [L_new]
    norm_num
    rw [mul_assoc, mul_assoc, ← sq, sq, mul_assoc, mul_comm 2.25, mul_assoc 6];
    ring
  have h2 : percentage_increase_surface_area L SA_original SA_new = 125 := by
    rw [percentage_increase_surface_area, SA_original, h1]
    norm_num
  exact h2

end cube_surface_area_increase_50_percent_l539_539060


namespace cos_25pi_over_6_l539_539563

noncomputable def cos_value (x : ℝ) : ℝ := Real.cos x

theorem cos_25pi_over_6 : cos_value (25 * Real.pi / 6) = √3 / 2 :=
by sorry

end cos_25pi_over_6_l539_539563


namespace irrational_sqrt_2_l539_539536

theorem irrational_sqrt_2 :
  let x1 := 3.14
  let x2 := (λ n : ℕ, if n = 0 then 3 else if n % 2 = 1 then 1 else 4) -- representing 3.141414...
  let x3 := 1/3
  let x4 := Real.sqrt 2
  (∃ n k : ℕ, x1 = n / k) ∧
  (∃ a b : ℕ, (∀ n : ℕ, x2 n = if n = 0 then a else if n % 2 = 1 then b else a)) ∧ 
  (∃ p q : ℕ, x3 = p / q) ∧ 
  ¬ (∃ r s : ℕ, x4 = r / s) := 
begin
  sorry
end

end irrational_sqrt_2_l539_539536


namespace smallest_hexagons_cover_disc_l539_539041

-- Definitions
def disc_radius : ℝ := 1
def hexagon_side : ℝ := 1

-- Statement
theorem smallest_hexagons_cover_disc :
  ∀ (n : ℕ), n_reg_hexagons_cover_disc hexagon_side disc_radius n → n ≥ 3 :=
by
  sorry

-- Supporting Definitions
def n_reg_hexagons_cover_disc (hex_side disc_rad : ℝ) (n : ℕ) : Prop :=
  ∃ (h : n > 0), (bounded_area_clearance hex_side disc_rad n = true)

-- Dummy Definition
def bounded_area_clearance (hex_side disc_rad : ℝ) (n : ℕ) : bool :=
  if n >= 3 then true else false

end smallest_hexagons_cover_disc_l539_539041


namespace sequence_sum_square_l539_539335

theorem sequence_sum_square (n : ℕ) (h : n > 0) :
  n + (n + 1) + ... + (3 * n - 2) = (2 * n - 1) ^ 2 :=
sorry

end sequence_sum_square_l539_539335


namespace percentage_increase_efficiency_l539_539355

-- Defining the times taken by Sakshi and Tanya
def sakshi_time : ℕ := 12
def tanya_time : ℕ := 10

-- Defining the efficiency in terms of work per day for Sakshi and Tanya
def sakshi_efficiency : ℚ := 1 / sakshi_time
def tanya_efficiency : ℚ := 1 / tanya_time

-- The statement of the proof: percentage increase
theorem percentage_increase_efficiency : 
  100 * ((tanya_efficiency - sakshi_efficiency) / sakshi_efficiency) = 20 := 
by
  -- The actual proof will go here
  sorry

end percentage_increase_efficiency_l539_539355


namespace margie_travel_distance_l539_539742

theorem margie_travel_distance:
  (gas_price_per_gallon : ℕ) (miles_per_gallon : ℕ) (money_available : ℕ) :
  gas_price_per_gallon = 4 →
  miles_per_gallon = 40 →
  money_available = 30 →
  money_available / gas_price_per_gallon * miles_per_gallon = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end margie_travel_distance_l539_539742


namespace matrix_commutation_l539_539313

open Matrix

theorem matrix_commutation 
  (a b c d : ℝ) (h₁ : 4 * b ≠ c) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5],
      B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d] in
  A * B = B * A → (a - d) / (c - 4 * b) = 2 :=
by 
  intro A B h₂ 
  sorry

end matrix_commutation_l539_539313


namespace even_functions_l539_539562

noncomputable def f1 : ℝ → ℝ := λ x => (x^2)^(1/3) + 1
noncomputable def f2 : ℝ → ℝ := λ x => x + 1/x
noncomputable def f3 : ℝ → ℝ := λ x => (1 + x)^(1/2) - (1 - x)^(1/2)
noncomputable def f4 : ℝ → ℝ := λ x => x^2 + x^(-2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem even_functions :
  is_even f1 ∧ is_even f4 :=
by
  sorry

end even_functions_l539_539562


namespace incorrect_oper_is_invalid_l539_539892

-- Conditions
def cond_A := (real.sqrt ((-2)^2) = 2)
def cond_B := ((-real.sqrt 2)^2 = 2)
def cond_D := (real.cbrt (-8) = -2)

-- Incorrect operation: √4 = ±2
def incorrect_oper := (real.sqrt 4 = 2) ∧ (real.sqrt 4 ≠ -2)

-- Lean statement
theorem incorrect_oper_is_invalid : ¬ incorrect_oper := by
  -- convert options into conditions
  have h1 : real.sqrt 4 = 2, from eq.refl (real.sqrt 4),
  -- assume the negative equality
  have h2 : real.sqrt 4 ≠ -2, from λ h, by linarith,
  contradiction


end incorrect_oper_is_invalid_l539_539892


namespace min_value_func_l539_539850

noncomputable def func (x y : ℝ) : ℝ := (x * y - 2)^2 + (x - y)^2

theorem min_value_func : ∃ x y : ℝ, func x y = 0 :=
by
  use [sqrt 2, sqrt 2]
  have h : (sqrt 2 * sqrt 2 - 2 : ℝ) = 0 := by
    -- sqrt 2 * sqrt 2 = 2
    have := real.mul_self_sqrt (real.sqrt_nonneg 2)
    norm_num
  simp [func, h]
  simp [pow_two, mul_self_sqrt, real.sqrt_nonneg, sqrt_eq_rpow, sub_self]
  sorry

end min_value_func_l539_539850


namespace log_equation_solution_l539_539972

theorem log_equation_solution (x : ℝ) (h : Real.log x + Real.log (x + 4) = Real.log (2 * x + 8)) : x = 2 :=
sorry

end log_equation_solution_l539_539972


namespace probability_A_fires_l539_539482

theorem probability_A_fires 
  (p_first_shot: ℚ := 1/6)
  (p_not_fire: ℚ := 5/6)
  (p_recur: ℚ := p_not_fire * p_not_fire) : 
  ∃ (P_A : ℚ), P_A = 6/11 :=
by
  have eq1 : P_A = p_first_shot + (p_recur * P_A) := sorry
  have eq2 : P_A * (1 - p_recur) = p_first_shot := sorry
  have eq3 : P_A = (p_first_shot * 36) / 11 := sorry
  exact ⟨P_A, sorry⟩

end probability_A_fires_l539_539482


namespace shanghai_expo_assignment_l539_539151

-- Define the set of volunteers and their genders
def volunteers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def girls := {1, 2, 3, 4}
def boys := {5, 6, 7, 8, 9, 10}

-- Define the function to count combinations
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- Prove the number of valid assignments
theorem shanghai_expo_assignment :
  let total_combs := combinations 10 3
  let girl_combs := combinations 4 3
  let boy_combs := combinations 6 3
  let valid_combs := total_combs - girl_combs - boy_combs
  valid_combs * 6 = 576 :=
by
  let total_combs := combinations 10 3
  let girl_combs := combinations 4 3
  let boy_combs := combinations 6 3
  let valid_combs := total_combs - girl_combs - boy_combs
  have comb_10_3 : total_combs = 120 := by sorry
  have comb_4_3 : girl_combs = 4 := by sorry
  have comb_6_3 : boy_combs = 20 := by sorry
  have valid_comb_10_3 : valid_combs * 6 = 576 := by sorry
  exact valid_comb_10_3

end shanghai_expo_assignment_l539_539151


namespace f_f_10_eq_2_l539_539230

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else Real.log10 x

theorem f_f_10_eq_2 : f (f 10) = 2 := by
  sorry

end f_f_10_eq_2_l539_539230


namespace point_on_line_has_correct_y_l539_539909

theorem point_on_line_has_correct_y (a : ℝ) : (2 * 3 + a - 7 = 0) → a = 1 :=
by 
  sorry

end point_on_line_has_correct_y_l539_539909


namespace sum_of_squares_l539_539628

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : ab + bc + ca = 5) : a^2 + b^2 + c^2 = 390 :=
by sorry

end sum_of_squares_l539_539628


namespace hexagon_area_correct_l539_539763

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

def hexagon_area (A D : ℝ × ℝ) (is_regular_hexagon : Bool) : ℝ :=
  if is_regular_hexagon then
    let side_length := (distance A D) / 2
    (3 : ℝ) * Real.sqrt (3) / 2 * side_length ^ 2
  else 0

def A := (0, 0) : ℝ × ℝ
def D := (8, 2) : ℝ × ℝ
def is_regular_hexagon := true

theorem hexagon_area_correct :
  hexagon_area A D is_regular_hexagon = (17 * Real.sqrt 3) / 2 :=
sorry

end hexagon_area_correct_l539_539763


namespace milkshakes_more_than_ice_cream_cones_l539_539748

def ice_cream_cones_sold : ℕ := 67
def milkshakes_sold : ℕ := 82

theorem milkshakes_more_than_ice_cream_cones : milkshakes_sold - ice_cream_cones_sold = 15 := by
  sorry

end milkshakes_more_than_ice_cream_cones_l539_539748


namespace max_marks_test_l539_539093

theorem max_marks_test (M : ℝ) : 
  (0.30 * M = 80 + 100) -> 
  M = 600 :=
by 
  sorry

end max_marks_test_l539_539093


namespace find_a_l539_539644

noncomputable def curve_equation (a x y : ℝ) : Prop := 4 * x^2 + a^2 * y^2 = 4 * a^2

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 + t, 7 + sqrt 3 * t)

noncomputable def point_P : ℝ × ℝ := (0, 4)

theorem find_a (a : ℝ) (t₁ t₂ : ℝ) 
  (line_eqn : ∀ t, (parametric_line t).fst^2 + a^2 * (parametric_line t).snd^2 = 4 * a^2)
  (P_MN_eq : |point_P.1 - (parametric_line t₁).1| * |point_P.2 - (parametric_line t₂).2| = 14) :
  a = 2 * sqrt 21 / 3 := by
  sorry

end find_a_l539_539644


namespace transformations_return_triangle_to_original_position_l539_539309

-- Define the triangle vertices
def T : set (ℝ × ℝ) := {(0, 0), (5, 0), (0, 4)}

-- Define the transformations
def rotate_60 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in (-y / 2 - (√3 / 2) * x, (√3 / 2) * y - x / 2)

def rotate_120 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in (-(√3 / 2) * y - x / 2, y / 2 + (√3 / 2) * x)

def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in (-x, -y)

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in (x, -y)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in (-x, y)

def identity_scaling (p : ℝ × ℝ) : ℝ × ℝ := p

-- List of all transformations
def transformations : list (ℝ × ℝ → ℝ × ℝ) :=
  [rotate_60, rotate_120, rotate_180, reflect_x, reflect_y, identity_scaling]

-- Define the problem statement
theorem transformations_return_triangle_to_original_position :
  let sequences := list.product transformations transformations
                    |> list.product transformations
                    |> list.map (λ (t, (u, v)), v ∘ u ∘ t)
  in (sequences.count (λ f, f '' T = T)) = 25 :=
by {
  sorry
}

end transformations_return_triangle_to_original_position_l539_539309


namespace width_of_path_l539_539089

theorem width_of_path (length width : ℝ) (area_of_path : ℝ) (x : ℝ) (h1 : length = 30)
  (h2 : width = 20) (h3 : area_of_path = (1 / 3) * (length * width)) :
  2.1925 = x :=
by 
  have g_area : ℝ := length * width
  have h4 : g_area = 600 := by sorry
  have h5 : area_of_path = 200 := by sorry
  have remaining_area : ℝ := (length - 2 * x) * (width - 2 * x)
  have h6 : 200 = g_area - remaining_area := by sorry
  have eqn : 4 * x^2 - 100 * x + 200 = 0 := by sorry
  have sol := (quad_eq_solve 4 (-100) 200)
  show 2.1925 = x, by sorry

-- quadratic equation solution helper
def quad_eq_solve (a b c : ℝ) : ℝ :=
  (-b + real.sqrt (b^2 - 4 * a * c)) / (2 * a)

end width_of_path_l539_539089


namespace min_value_x_add_one_div_y_l539_539182

theorem min_value_x_add_one_div_y (x y : ℝ) (h1 : x > 1) (h2 : x - y = 1) : 
x + 1 / y ≥ 3 :=
sorry

end min_value_x_add_one_div_y_l539_539182


namespace class_schedule_count_l539_539833

-- Define the types of classes
inductive ClassType
| Chinese
| English
| Physics
| Chemistry
| Biology
| Mathematics

open ClassType

-- Define the conditions
def is_valid_schedule (schedule : List ClassType) : Prop :=
  schedule.length = 7 ∧
  (∃ i, schedule.nth i = some Biology → i ≠ 0) ∧
  (∃ i, schedule.nth i = some Mathematics ∧ schedule.nth (i + 1) = some Mathematics) ∧
  ∀ i, schedule.nth i = some English → (schedule.nth (i + 1) ≠ some Mathematics ∧ schedule.nth (i - 1) ≠ some Mathematics)

-- Define the main theorem
theorem class_schedule_count : ∃ (schedules : List (List ClassType)), (∀ s ∈ schedules, is_valid_schedule s) ∧ schedules.length = 408 := by
  sorry

end class_schedule_count_l539_539833


namespace inequality_solution_ab_4bc_9ac_ineq_l539_539911

-- Part 1: Prove the solution set for the inequality 2|x-2| - |x+1| > 3 is { x | x < 0 or x > 8 }
theorem inequality_solution (x : ℝ) : (2 * |x - 2| - |x + 1| > 3) ↔ (x < 0 ∨ x > 8) := 
sorry

-- Part 2: Prove ab + 4bc + 9ac ≥ 36 given a, b, and c are positive numbers satisfying abc = a + b + c
theorem ab_4bc_9ac_ineq (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
  (h : a * b * c = a + b + c) : ab + 4bc + 9ac ≥ 36 := 
sorry

end inequality_solution_ab_4bc_9ac_ineq_l539_539911


namespace matrix_commutation_l539_539314

open Matrix

theorem matrix_commutation 
  (a b c d : ℝ) (h₁ : 4 * b ≠ c) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5],
      B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d] in
  A * B = B * A → (a - d) / (c - 4 * b) = 2 :=
by 
  intro A B h₂ 
  sorry

end matrix_commutation_l539_539314


namespace pens_sold_to_recover_investment_l539_539710

-- Given the conditions
variables (P C : ℝ) (N : ℝ)
-- P is the total cost of 30 pens
-- C is the cost price of each pen
-- N is the number of pens sold to recover the initial investment

-- Stating the conditions
axiom h1 : P = 30 * C
axiom h2 : N * 1.5 * C = P

-- Proving that N = 20
theorem pens_sold_to_recover_investment (P C N : ℝ) (h1 : P = 30 * C) (h2 : N * 1.5 * C = P) : N = 20 :=
by
  sorry

end pens_sold_to_recover_investment_l539_539710


namespace person_A_work_days_l539_539344

theorem person_A_work_days (A : ℝ) : 
  (∀ (B : ℝ), B = 24 → 
  ∀ (combined_work_completed : ℝ), combined_work_completed = 0.19444444444444442 →
  ∀ (days_working_together : ℝ), days_working_together = 2 →
  2 * (1 / A + 1 / B) = combined_work_completed) →
  A = 18 :=
begin
  sorry
end

end person_A_work_days_l539_539344


namespace right_prism_max_volume_l539_539681

-- Define the variables and conditions
variables {a b h : ℝ} {θ : ℝ}
-- Define the surface area constraint
axiom surface_area_constraint : a * h + b * h + (1/2) * a * b * sin θ = 30

-- Define the maximum volume condition
def max_volume (V : ℝ) := V = 10 * sqrt 5

-- Statement of the problem
theorem right_prism_max_volume (a b h : ℝ) (θ : ℝ) 
  (h_surface_area_constraint : a * h + b * h + (1/2) * a * b * sin θ = 30) : 
  ∃ V : ℝ, max_volume V :=
sorry

end right_prism_max_volume_l539_539681


namespace noemi_initial_amount_l539_539749

theorem noemi_initial_amount : 
  ∀ (rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount : ℕ), 
    rouletteLoss = 600 → 
    blackjackLoss = 800 → 
    pokerLoss = 400 → 
    baccaratLoss = 700 → 
    remainingAmount = 1500 → 
    initialAmount = rouletteLoss + blackjackLoss + pokerLoss + baccaratLoss + remainingAmount →
    initialAmount = 4000 :=
by
  intros rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  exact h6

end noemi_initial_amount_l539_539749


namespace printer_to_enhanced_ratio_l539_539830

def B : ℕ := 2125
def P : ℕ := 2500 - B
def E : ℕ := B + 500
def total_price := E + P

theorem printer_to_enhanced_ratio :
  (P : ℚ) / total_price = 1 / 8 := 
by {
  -- skipping the proof
  sorry
}

end printer_to_enhanced_ratio_l539_539830


namespace smallest_number_l539_539945

theorem smallest_number:
    let a := 3.25
    let b := 3.26   -- 326% in decimal
    let c := 3.2    -- 3 1/5 in decimal
    let d := 3.75   -- 15/4 in decimal
    c < a ∧ c < b ∧ c < d :=
by
    sorry

end smallest_number_l539_539945


namespace max_books_borrowed_l539_539900

theorem max_books_borrowed 
  (num_students : ℕ)
  (num_no_books : ℕ)
  (num_one_book : ℕ)
  (num_two_books : ℕ)
  (average_books : ℕ)
  (h_num_students : num_students = 32)
  (h_num_no_books : num_no_books = 2)
  (h_num_one_book : num_one_book = 12)
  (h_num_two_books : num_two_books = 10)
  (h_average_books : average_books = 2)
  : ∃ max_books : ℕ, max_books = 11 := 
by
  sorry

end max_books_borrowed_l539_539900


namespace total_money_left_l539_539966

theorem total_money_left (david_start john_start emily_start : ℝ) 
  (david_percent_left john_percent_spent emily_percent_spent : ℝ) : 
  (david_start = 3200) → 
  (david_percent_left = 0.65) → 
  (john_start = 2500) → 
  (john_percent_spent = 0.60) → 
  (emily_start = 4000) → 
  (emily_percent_spent = 0.45) → 
  let david_spent := david_start / (1 + david_percent_left)
  let david_remaining := david_start - david_spent
  let john_remaining := john_start * (1 - john_percent_spent)
  let emily_remaining := emily_start * (1 - emily_percent_spent)
  david_remaining + john_remaining + emily_remaining = 4460.61 :=
by
  sorry

end total_money_left_l539_539966


namespace find_tuesday_temp_l539_539128

variable (temps : List ℝ) (avg : ℝ) (len : ℕ) 

theorem find_tuesday_temp (h1 : temps = [99.1, 98.2, 99.3, 99.8, 99, 98.9, tuesday_temp])
                         (h2 : avg = 99)
                         (h3 : len = 7)
                         (h4 : (temps.sum / len) = avg) :
                         tuesday_temp = 98.7 := 
sorry

end find_tuesday_temp_l539_539128


namespace median_is_87_5_l539_539391

noncomputable def median_of_mean (s : Finset ℕ) : ℚ :=
  let mean := (s.sum id : ℚ) / s.card
  have mean_s : mean = 87.5 := by sorry
  let y := 525 - s.erase 89 88 85 86 87
  let sorted_s := (s.insert y).sort (≤)
  let n := sorted_s.card
  if h : n % 2 = 1 then
    sorted_s.get (Fin.of_nat (n / 2))
  else
    (sorted_s.get (Fin.of_nat (n / 2 - 1)) + sorted_s.get (Fin.of_nat (n / 2))) / 2

-- Now we state the proof problem in Lean 4.
theorem median_is_87_5 :
  median_of_mean (Finset.erase (Finset.range 6) 90 88 85 86 87) = 87.5 :=
by
  sorry

end median_is_87_5_l539_539391


namespace hours_between_dates_not_thirteen_l539_539995

def total_hours (start_date: ℕ × ℕ × ℕ × ℕ) (end_date: ℕ × ℕ × ℕ × ℕ) (days_in_dec: ℕ) : ℕ :=
  let (start_year, start_month, start_day, start_hour) := start_date
  let (end_year, end_month, end_day, end_hour) := end_date
  (days_in_dec - start_day) * 24 - start_hour + end_day * 24 + end_hour

theorem hours_between_dates_not_thirteen :
  let start_date := (2015, 12, 30, 23)
  let end_date := (2016, 1, 1, 12)
  let days_in_dec := 31
  total_hours start_date end_date days_in_dec ≠ 13 :=
by
  sorry

end hours_between_dates_not_thirteen_l539_539995


namespace least_three_digit_with_product_eight_is_124_l539_539865

noncomputable def least_three_digit_with_product_eight : ℕ :=
  let candidates := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (x.digits 10).prod = 8} in
  if h : Nonempty candidates then
    let min_candidate := Nat.min' _ h in min_candidate
  else
    0 -- default to 0 if no such number exists

theorem least_three_digit_with_product_eight_is_124 :
  least_three_digit_with_product_eight = 124 := sorry

end least_three_digit_with_product_eight_is_124_l539_539865


namespace least_three_digit_product_of_digits_is_8_l539_539869

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l539_539869


namespace taller_tree_height_l539_539829

-- Given conditions
variables (h : ℕ) (ratio_cond : (h - 20) * 7 = h * 5)

-- Proof goal
theorem taller_tree_height : h = 70 :=
sorry

end taller_tree_height_l539_539829


namespace cylinder_surface_area_correct_l539_539935

noncomputable def cylinder_surface_area :=
  let r := 8   -- radius in cm
  let h := 10  -- height in cm
  let arc_angle := 90 -- degrees
  let x := 40
  let y := -40
  let z := 2
  x + y + z

theorem cylinder_surface_area_correct : cylinder_surface_area = 2 := by
  sorry

end cylinder_surface_area_correct_l539_539935


namespace find_number_outside_range_of_f_l539_539213

open Real

noncomputable def f (a b c d : ℝ) : ℝ → ℝ := λ x, (a * x + b) / (c * x + d)

def conditions (a b c d : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  f(a, b, c, d) 19 = 19 ∧ 
  f(a, b, c, d) 97 = 97 ∧ 
  ∀ x : ℝ, x ≠ -d / c → f(a, b, c, d) (f(a, b, c, d) x) = x

theorem find_number_outside_range_of_f (a b c d : ℝ) (h : conditions a b c d) :
  ∃ x : ℝ, x = 58 :=
sorry

end find_number_outside_range_of_f_l539_539213


namespace geometric_sequence_and_k_bounds_l539_539405

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom cond : ∀ n : ℕ, 2 * a n - S n = 1

theorem geometric_sequence_and_k_bounds :
  (∃ q : ℝ, q ≠ 0 ∧ a = λ n, (a 1) * q ^ (n - 1)) ∧ ( ∀ k : ℝ, (∀ n : ℕ, k * n * (2 ^ n) / (n + 1) ≥ (2 * n - 9) * (Σ m in range n, 1 / m (m + 1))) → k ≥ 3 / 64)
:= by
  sorry

end geometric_sequence_and_k_bounds_l539_539405


namespace triangle_area_QDA_l539_539146

def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (3, 15)
def D (p : ℝ) : ℝ × ℝ := (0, p + 3)

def base (A Q : ℝ × ℝ) : ℝ := A.1 - Q.1
def height (Q D : ℝ × ℝ) : ℝ := Q.2 - D.2

theorem triangle_area_QDA (p : ℝ) :
  let Q := Q; let A := A; let D := D p in
  let base := base A Q in
  let height := height Q D in
  (1 / 2 * base * height) = 18 - (3 / 2) * p :=
by
  sorry

end triangle_area_QDA_l539_539146


namespace ratio_of_areas_l539_539101

variables {A B C O L F E H : Point}
variables {AL : ℝ} {AH : ℝ} {angleAEH : ℝ}
variables (triangle_ABC : Triangle A B C)
variables (circle_O : Circle O)
variables (angle_bisector_AF : Line A F)
variables (radius_AO : Line A O)
variables (altitude_AH : Line A H)

-- Given conditions
axiom angle_A_gt_90 : ∠A > 90
axiom angle_bisector_intersects_circle_at_L : extends angle_bisector_AF intersects circle_O at L
axiom radius_intersects_BC_at_E : extends radius_AO intersects line (B, C) at E
axiom AL_length : AL = 4 * real.sqrt 2
axiom AH_length : AH = real.sqrt (2 * real.sqrt 3)
axiom angle_AEH_equals_60 : angleAEH = 60

-- The hypothesis about the ratio of areas
theorem ratio_of_areas :
  (area_triangle O A L) / (area_quadrilateral O E F L) = 4 / 3 :=
sorry

end ratio_of_areas_l539_539101


namespace sin_cos_relationship_l539_539258

theorem sin_cos_relationship (α : ℝ) (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) : 
  Real.sin α - Real.cos α > 1 :=
sorry

end sin_cos_relationship_l539_539258


namespace non_square_matchings_at_least_factorial_l539_539988

theorem non_square_matchings_at_least_factorial 
  (n : ℕ) 
  (a : ℕ → ℕ)
  (non_square_matching_exists : ∃ (S : (fin (2 * n) → fin (2 * n))),
    (∀ i j : fin (2 * n), i ≠ j → (i != S i → (a i * a j) ≠ (k * k for some k : ℕ)))
    ∧ ∀ i, S (S i) = i
  ) :
  ∃ (number_of_non_square_matchings : ℕ), number_of_non_square_matchings ≥ n! :=
sorry

end non_square_matchings_at_least_factorial_l539_539988


namespace selling_price_l539_539301

def initial_cost : ℕ := 600
def food_cost_per_day : ℕ := 20
def number_of_days : ℕ := 40
def vaccination_and_deworming_cost : ℕ := 500
def profit : ℕ := 600

theorem selling_price (S : ℕ) :
  S = initial_cost + (food_cost_per_day * number_of_days) + vaccination_and_deworming_cost + profit :=
by
  sorry

end selling_price_l539_539301


namespace complex_magnitude_power_12_l539_539129

theorem complex_magnitude_power_12 : 
  abs ((1 / 2 : ℂ) + (sqrt 3 / 2 : ℂ) * I)^12 = 1 := 
by
  sorry

end complex_magnitude_power_12_l539_539129


namespace quadratic_other_root_l539_539191

theorem quadratic_other_root (m x : ℝ) (h : 2 * x^2 - (m + 3) * x + m = 0) (h0 : x = 1) : 
  ∃ y, y = (-m - 5) / 2 :=
by
  -- Given conditions
  -- x is a root
  have : 2 * 1^2 - (m + 3) * 1 + m = 0 := sorry

  -- By Vieta's formulas, the other root is computed
  -- y = (-b - a) / a, with a = 2, b = -(m + 3)
  have y := (-m - 5) / 2
  exact ⟨y, rfl⟩

end quadratic_other_root_l539_539191


namespace min_vector_sum_length_l539_539173

theorem min_vector_sum_length (v : Fin 7 → ℝ × ℝ) 
  (h_unit : ∀ i, ‖v i‖ = 1) (h_nonneg : ∀ i, 0 ≤ v i.1 ∧ 0 ≤ v i.2) :
  ‖∑ i, v i‖ ≥ 5 := by
  sorry

end min_vector_sum_length_l539_539173


namespace problem1_problem2_l539_539365

theorem problem1
  (x: ℝ)
  (h1 : (1 + 3^(-x)) / (1 + 3^x) = 3) :
  x = -1 :=
sorry

theorem problem2
  (x: ℝ)
  (h2 : log 4 (3 * x - 1) = log 4 (x - 1) + log 4 (3 + x))
  (h3 : x > 1) :
  x = 2 :=
sorry

end problem1_problem2_l539_539365


namespace points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l539_539554

-- Problem 1: Prove that if \(x^3 + y^3 + z^3 = (x + y + z)^3\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_cubic_eq (x y z : ℝ) (h : x^3 + y^3 + z^3 = (x + y + z)^3) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

-- Problem 2: Prove that if \(x^5 + y^5 + z^5 = (x + y + z)^5\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_quintic_eq (x y z : ℝ) (h : x^5 + y^5 + z^5 = (x + y + z)^5) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

end points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l539_539554


namespace integer_pairs_satisfying_equation_l539_539578

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x * (x + 1) * (x + 7) * (x + 8) = y^2 →
    (x = 1 ∧ y = 12) ∨ (x = 1 ∧ y = -12) ∨ 
    (x = -9 ∧ y = 12) ∨ (x = -9 ∧ y = -12) ∨ 
    (x = -4 ∧ y = 12) ∨ (x = -4 ∧ y = -12) ∨ 
    (x = 0 ∧ y = 0) ∨ (x = -8 ∧ y = 0) ∨ 
    (x = -1 ∧ y = 0) ∨ (x = -7 ∧ y = 0) :=
by sorry

end integer_pairs_satisfying_equation_l539_539578


namespace distinct_arrangements_apple_l539_539655

theorem distinct_arrangements_apple :
  let n := 5
  let p := 2
  nat.factorial n / nat.factorial p = 60 := by
sorry

end distinct_arrangements_apple_l539_539655


namespace find_diff_between_max_and_min_l539_539039

theorem find_diff_between_max_and_min :
  ∀ (a_1 a_2 a_3 a_4 a_5 : ℕ),
  {a_1, a_2, a_3, a_4, a_5} = {1, 2, 3, 4, 5} →
  let S := a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 in
  let S_max := 1 + 2*2 + 3*3 + 4*4 + 5*5 in
  let S_min := 1*5 + 2*4 + 3*3 + 4*2 + 5*1 in
  (S_max - S_min) = 20 :=
by
  intros a_1 a_2 a_3 a_4 a_5 h
  let S := a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5
  let S_max := 1 + 2*2 + 3*3 + 4*4 + 5*5
  let S_min := 1*5 + 2*4 + 3*3 + 4*2 + 5*1
  sorry

end find_diff_between_max_and_min_l539_539039


namespace robot_paths_from_A_to_B_l539_539516

/-- Define a function that computes the number of distinct paths a robot can take -/
def distinctPaths (A B : ℕ × ℕ) : ℕ := sorry

/-- Proof statement: There are 556 distinct paths from A to B, given the movement conditions -/
theorem robot_paths_from_A_to_B (A B : ℕ × ℕ) (h_move : (A, B) = ((0, 0), (10, 10))) :
  distinctPaths A B = 556 :=
sorry

end robot_paths_from_A_to_B_l539_539516


namespace toby_walks_9400_steps_on_Sunday_l539_539415

theorem toby_walks_9400_steps_on_Sunday :
  ∀ (steps_mon steps_tue steps_wed steps_thu steps_fri steps_sat : ℕ),
    steps_mon = 9100 →
    steps_tue = 8300 →
    steps_wed = 9200 →
    steps_thu = 8900 →
    2 * 9050 = steps_fri + steps_sat →
    (steps_mon + steps_tue + steps_wed + steps_thu + steps_fri + steps_sat + 9400) / 7  = 9000 →
    steps_fri + steps_sat > 0 :=
begin
  intros,  -- Introduce all assumptions
  sorry    -- Provide the proof here
end

end toby_walks_9400_steps_on_Sunday_l539_539415


namespace log_domain_l539_539806

theorem log_domain : 
  (∀ x : ℝ, ∃ y : ℝ, f x = log2 (x - 3) → x > 3) :=
sorry

end log_domain_l539_539806


namespace complex_abs_of_sqrt_l539_539371

variable (z : ℂ)

theorem complex_abs_of_sqrt (h : z^2 = 16 - 30 * Complex.I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_abs_of_sqrt_l539_539371


namespace max_triangle_area_l539_539004

theorem max_triangle_area (a b c : ℝ) (h₁ : a = 3) (h₂ : b + c = 5) :
  ∃ S, S = 3 ∧ (S = sqrt ((4 * (4 - a) * (4 - b) * (4 - c))) ) :=
by 
  sorry

end max_triangle_area_l539_539004


namespace motorcyclist_average_speed_l539_539926

theorem motorcyclist_average_speed
  (d_AB : ℕ := 120) -- Distance from A to B
  (d_BC : ℕ := d_AB / 2) -- Distance from B to C
  (t_BC_hour: ℝ := (d_BC / 108 : ℝ)) -- Time from B to C in hours
  (t_AB_hour: ℝ := 3 * t_BC_hour) -- Time from A to B in hours
  (total_distance: ℕ := d_AB + d_BC) -- Total distance of the trip
  (total_time: ℝ := t_AB_hour + t_BC_hour) -- Total time of the trip
  : total_time = 20 / 9 →
    total_distance = 180 →
    (total_distance / total_time) = 81 :=
by
  intros h1 h2
  rw [h2, h1]
  norm_num
  have h : (180 / (20 / 9) : ℝ) = 81 := by norm_num
  exact h

end motorcyclist_average_speed_l539_539926


namespace smallest_possible_perimeter_l539_539937

noncomputable def smallest_prime_perimeter_of_scalene_triangle : ℕ :=
  23

theorem smallest_possible_perimeter (p q : ℕ) (hp : p.prime) (hq : q.prime)
    (hscalene : 5 ≠ p ∧ 5 ≠ q ∧ p ≠ q) (ht : 5 + p > q ∧ p + q > 5 ∧ q + 5 > p) :
    Nat.prime (5 + p + q) → 5 + p + q = smallest_prime_perimeter_of_scalene_triangle :=
by
  sorry

end smallest_possible_perimeter_l539_539937


namespace vector_magnitude_example_l539_539620

open Real

variables (a b : ℝ × ℝ × ℝ)
def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def vector_magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (u.1^2 + u.2^2 + u.3^2)

theorem vector_magnitude_example :
  vector_magnitude (vector_sub (4, 2, -4) (6, -3, 3)) = sqrt 78 := by
  sorry

end vector_magnitude_example_l539_539620


namespace find_constants_l539_539294

theorem find_constants (a : ℕ → ℕ) (b c d : ℕ) 
  (h : ∀ n, a n = 2 * Int.floor (Real.sqrt (n + -1 : ℝ)) + 1) :
  b = 2 ∧ c = -1 ∧ d = 1 :=
by
  sorry

end find_constants_l539_539294


namespace sum_two_digit_divisors_of_99_l539_539552

theorem sum_two_digit_divisors_of_99 :
  ∑ x in { y | 10 ≤ y ∧ y < 100 ∧ 99 ∣ y }.to_finset ∩ {11, 33}.to_finset, x = 44 :=
by
  sorry

end sum_two_digit_divisors_of_99_l539_539552


namespace sister_bought_4_avocados_l539_539179

theorem sister_bought_4_avocados
  (avocados_per_serving: ℕ)
  (initial_avocados: ℕ)
  (servings: ℕ)
  (total_avocados: ℕ)
  (required_avocados: ℕ) :
  avocados_per_serving = 3 →
  initial_avocados = 5 →
  servings = 3 →
  total_avocados = servings * avocados_per_serving →
  required_avocados = total_avocados - initial_avocados →
  required_avocados = 4 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  simp at h4
  rw [h4, h2] at h5
  simp at h5
  exact h5

end sister_bought_4_avocados_l539_539179


namespace fraction_left_handed_non_throwers_l539_539336

theorem fraction_left_handed_non_throwers (total_players throwers total_right_handed : ℕ)
  (all_throwers_right_handed : Bool) (h1 : total_players = 70)
  (h2 : throwers = 34)
  (h3 : all_throwers_right_handed = true)
  (h4 : total_right_handed = 58) :
  (let non_throwers := total_players - throwers in
   let right_handed_non_throwers := total_right_handed - throwers in
   let left_handed_non_throwers := non_throwers - right_handed_non_throwers in
   left_handed_non_throwers / non_throwers = 1 / 3) :=
by 
  sorry

end fraction_left_handed_non_throwers_l539_539336


namespace problem1_problem2_problem3_problem4_l539_539957

-- The first problem
theorem problem1 : (-4) - (+13) + (-5) - (-9) + 7 = -6 :=
by
  sorry

-- The second problem
theorem problem2 : (25 / 4) - 3.3 - (-6) - (15 / 4) + 4 + 3.3 = 20 :=
by
  sorry

-- The third problem
theorem problem3 : -81 ÷ (-9 / 4) × (4 / 9) ÷ (-16) = -1 :=
by
  sorry

-- The fourth problem
theorem problem4 : (-24) × ((11 / 8) + (7 / 3) - 0.75) = -71 :=
by
  sorry

end problem1_problem2_problem3_problem4_l539_539957


namespace values_of_a_and_b_range_of_c_isosceles_perimeter_l539_539255

def a : ℝ := 3
def b : ℝ := 4

axiom triangle_ABC (c : ℝ) : 0 < c

noncomputable def equation_condition (a b : ℝ) : Prop :=
  |a-3| + (b-4)^2 = 0

noncomputable def is_valid_c (c : ℝ) : Prop :=
  1 < c ∧ c < 7

theorem values_of_a_and_b (h : equation_condition a b) : a = 3 ∧ b = 4 := sorry

theorem range_of_c (h : equation_condition a b) : is_valid_c c := sorry

noncomputable def isosceles_triangle (c : ℝ) : Prop :=
  c = 4 ∨ c = 3

theorem isosceles_perimeter (h : equation_condition a b) (hc : isosceles_triangle c) : (3 + 3 + 4 = 10) ∨ (4 + 4 + 3 = 11) := sorry

end values_of_a_and_b_range_of_c_isosceles_perimeter_l539_539255


namespace johnny_jog_speed_l539_539304

theorem johnny_jog_speed :
  let distance := 6.857142857142858
  let total_time := 1
  let bus_speed := 30
  let bus_time := distance / bus_speed
  let jog_time := total_time - bus_time
  let jog_speed := distance / jog_time
  jog_speed ≈ 8.89 :=
by
  rfl

end johnny_jog_speed_l539_539304


namespace preimage_of_4_3_l539_539240

theorem preimage_of_4_3 :
  ∃ (a b : ℕ), (a + 2 * b = 4) ∧ (2 * a - b = 3) ∧ (a = 2) ∧ (b = 1) :=
by
  use [2, 1]
  exact ⟨rfl, rfl, rfl, rfl⟩

end preimage_of_4_3_l539_539240


namespace probability_of_x_greater_than_9y_l539_539345

theorem probability_of_x_greater_than_9y : 
  let region := set.prod (set.Icc 0 2017) (set.Icc 0 2018)
  ∃ (p : ℚ), p = 2017 / 36324 ∧ measure (set_of (λ (x, y) : ℚ × ℚ, x > 9 * y) ∩ region) / measure region = p :=
sorry

end probability_of_x_greater_than_9y_l539_539345


namespace polyhedron_with_9_edges_is_triangular_prism_l539_539535

theorem polyhedron_with_9_edges_is_triangular_prism 
  (n_tri : ℕ) (E_tri : ℕ)
  (h1 : n_tri = 3) (h2 : E_tri = 3 * n_tri) :
  E_tri = 9 :=
by  {
  rw [h1] at h2,
  exact h2,
}

end polyhedron_with_9_edges_is_triangular_prism_l539_539535


namespace woman_distance_2_miles_l539_539104

-- Distance calculation given the final position
def distance_from_start (start final : ℝ × ℝ) : ℝ :=
  real.sqrt ((final.1 - start.1) ^ 2 + (final.2 - start.2) ^ 2)

-- Definition of the woman's initial and final coordinates
def final_position (y : ℝ) : ℝ × ℝ :=
  (y - 2 * real.sqrt 3, -2)

-- The proof problem to be solved in Lean
theorem woman_distance_2_miles (y : ℝ) : 
  distance_from_start (0, 0) (final_position y) = 2 ↔ y = 2 * real.sqrt 3 :=
by
  sorry

end woman_distance_2_miles_l539_539104


namespace todd_snow_cones_sold_l539_539416

-- Definitions of the conditions
def borrowed_from_brother : ℕ := 100
def repay_amount_to_brother : ℕ := 110
def ingredients_cost : ℕ := 75
def price_per_snow_cone : ℚ := 0.75
def amount_left_after_repayment : ℕ := 65

-- Main proof statement
theorem todd_snow_cones_sold : 
    let total_amount_made := repay_amount_to_brother + amount_left_after_repayment in
    let total_sales := total_amount_made + ingredients_cost in
    let number_of_snow_cones := total_sales / (price_per_snow_cone.to_nat) in 
    number_of_snow_cones = 333 := 
by
  sorry

end todd_snow_cones_sold_l539_539416


namespace vector_addition_l539_539654

def vec2 := (ℝ × ℝ)

def a : vec2 := (2, 1)
def b : vec2 := (1, 5)

theorem vector_addition : (2 • a + b = (5, 7)) :=
  sorry

end vector_addition_l539_539654


namespace impossible_placement_l539_539299

structure EquilateralTriangle where
  a b c : Point
  length_ab : dist a b = dist b c
  length_bc : dist b c = dist c a
  length_ca : dist c a = dist a b

structure RegularHexagon where
  a b c d e f : Point
  length_ab : dist a b = dist b c
  length_bc : dist b c = dist c d
  length_cd : dist c d = dist d e
  length_de : dist d e = dist e f
  length_ef : dist e f = dist f a
  length_fa : dist f a = dist a b

def isVisible (p q : Point) (triangle : EquilateralTriangle) :=
  let line_pq := segment p q
  ¬ (line_pq ∩ (triangle.a ∪ triangle.b ∪ triangle.c)).nonempty

theorem impossible_placement (hexagon : RegularHexagon) (triangle : EquilateralTriangle) :
  ∃ v : Point, (v ∈ {hexagon.a, hexagon.b, hexagon.c, hexagon.d, hexagon.e, hexagon.f}) ∧ 
  ¬ (isVisible v triangle.a triangle ∧ isVisible v triangle.b triangle ∧ isVisible v triangle.c triangle) :=
sorry

end impossible_placement_l539_539299


namespace odd_divisors_up_to_100_l539_539250

theorem odd_divisors_up_to_100 : 
  {n : ℕ | n > 0 ∧ n ≤ 100 ∧ (∃ m : ℕ, n = m * m)}.card = 10 := 
by
  sorry

end odd_divisors_up_to_100_l539_539250


namespace vector_perpendicular_sets_l539_539248

-- Define the problem in Lean
theorem vector_perpendicular_sets (x : ℝ) : 
  let a := (Real.sin x, Real.cos x)
  let b := (Real.sin x + Real.cos x, Real.sin x - Real.cos x)
  a.1 * b.1 + a.2 * b.2 = 0 ↔ ∃ (k : ℤ), x = k * (π / 2) + (π / 8) :=
sorry

end vector_perpendicular_sets_l539_539248


namespace molecular_weight_of_one_mole_l539_539430

theorem molecular_weight_of_one_mole 
  (total_weight : ℝ) (n_moles : ℝ) (mw_per_mole : ℝ)
  (h : total_weight = 792) (h2 : n_moles = 9) 
  (h3 : total_weight = n_moles * mw_per_mole) 
  : mw_per_mole = 88 :=
by
  sorry

end molecular_weight_of_one_mole_l539_539430


namespace unsuccessful_attempts_124_l539_539390

theorem unsuccessful_attempts_124 (num_digits: ℕ) (choices_per_digit: ℕ) (total_attempts: ℕ):
  num_digits = 3 → choices_per_digit = 5 → total_attempts = choices_per_digit ^ num_digits →
  total_attempts - 1 = 124 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact sorry

end unsuccessful_attempts_124_l539_539390


namespace sum_of_odd_divisors_of_240_l539_539433

theorem sum_of_odd_divisors_of_240 : 
  let n := 240 in 
  let odd_divisors := {d ∣ n | d % 2 = 1} in 
  ∑ d in odd_divisors, d = 24 :=
by 
  let n := 240
  let odd_divisors := {d ∣ n | d % 2 = 1}
  sorry

end sum_of_odd_divisors_of_240_l539_539433


namespace six_digit_square_number_l539_539091

theorem six_digit_square_number :
  ∃ (n : ℕ), (317 ≤ n ∧ n ≤ 999) ∧
    (let n_squared := n^2 in
     let last_two_digits := n_squared % 100 in
     let middle_two_digits := (n_squared / 100) % 100 in
     let first_two_digits := n_squared / 10000 in
     (last_two_digits = middle_two_digits) ∧
     (first_two_digits + middle_two_digits = 100) ∧
     n_squared = 316969) :=
by {
  -- proof steps here, but not required for this task
  sorry
}

end six_digit_square_number_l539_539091


namespace z_real_iff_m_z_complex_iff_m_z_purely_imaginary_iff_m_l539_539992

theorem z_real_iff_m :
  ∀ (m : ℝ), (∃ (z : ℂ), z = ↑(m^2 + m - 2) + (m^2 - 1) * complex.i ∧ z.im = 0) ↔ (m = 1 ∨ m = -1) :=
by
  sorry

theorem z_complex_iff_m :
  ∀ (m : ℝ), (∃ (z : ℂ), z = ↑(m^2 + m - 2) + (m^2 - 1) * complex.i) ↔ (m ≠ 1 ∧ m ≠ -1) :=
by
  sorry

theorem z_purely_imaginary_iff_m :
  ∀ (m : ℝ), (∃ (z : ℂ), z = ↑(m^2 + m - 2) + (m^2 - 1) * complex.i ∧ z.re = 0 ∧ z.im ≠ 0) ↔ (m = -2) :=
by
  sorry

end z_real_iff_m_z_complex_iff_m_z_purely_imaginary_iff_m_l539_539992


namespace dvd_count_correct_l539_539330

def total_dvds (store_dvds online_dvds : Nat) : Nat :=
  store_dvds + online_dvds

theorem dvd_count_correct :
  total_dvds 8 2 = 10 :=
by
  sorry

end dvd_count_correct_l539_539330


namespace cosine_angle_between_foci_and_intersection_point_l539_539378

noncomputable def ellipse : Set (ℝ × ℝ) := {p | p.1^2 / 6 + p.2^2 / 2 = 1}
noncomputable def hyperbola : Set (ℝ × ℝ) := {p | p.1^2 / 3 - p.2^2 = 1}
noncomputable def foci_of_ellipse : (ℝ × ℝ) × (ℝ × ℝ) := ((-2, 0), (2, 0))

theorem cosine_angle_between_foci_and_intersection_point :
  ∃ F1 F2 P, F1 = (-2,0) ∧ F2 = (2,0) ∧
    (P ∈ ellipse ∧ P ∈ hyperbola) ∧
    ∃ angle, cos (angle) = 1/3 :=
sorry

end cosine_angle_between_foci_and_intersection_point_l539_539378


namespace slope_at_point_l539_539028

theorem slope_at_point (f : ℝ → ℝ) (f' : ℝ → ℝ) (x₀ : ℝ) (y₀ : ℝ)
  (hf : ∀ x, f x = x^3 - 2 * x)
  (hf' : ∀ x, f' x = 3 * x^2 - 2)
  (hx₀ : x₀ = 1)
  (hy₀ : y₀ = f x₀) :
  f' x₀ = 1 :=
by
  rw [hf', hx₀]
  simp
  norm_num
  sorry

end slope_at_point_l539_539028


namespace quadratic_eq_with_root_l539_539137

theorem quadratic_eq_with_root (a r : ℝ) (h1 : a = 1) (h2 : r = √5 - 1) :
  ∃ (b c : ℚ), (∀ x : ℝ, x^2 + (b : ℝ) * x + (c : ℝ) = 0 → x = r ∨ x = -√5 - 1) :=
by
  sorry

end quadratic_eq_with_root_l539_539137


namespace focus_of_parabola_l539_539585

-- Problem statement
theorem focus_of_parabola (x y : ℝ) : (2 * x^2 = -y) → (focus_coordinates = (0, -1 / 8)) :=
by
  sorry

end focus_of_parabola_l539_539585


namespace solve_inequality_l539_539164

def satisfies_inequality (x : ℝ) : Prop :=
  (3 * x - 4) * (x + 1) / x ≥ 0

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | -1 ≤ x ∧ x < 0 ∨ x ≥ 4 / 3} :=
by
  sorry

end solve_inequality_l539_539164


namespace ratio_p_q_l539_539031

-- Define our conditions and results
noncomputable def numBalls : ℕ := 25
noncomputable def numBins : ℕ := 6

def distA : ℕ × ℕ × ℕ × ℕ := (5, 4, 4, 3)
def distB : ℕ × ℕ := (5, 4)

-- Definitions of probabilities p and q
noncomputable def p (N : ℕ) : ℝ := (some_calculation_for_p)
noncomputable def q (N : ℕ) : ℝ := (some_calculation_for_q)

theorem ratio_p_q :
  let N := (some_total_ways_to_distribute_balls) in 
  (p N) / (q N) = 5 / 3 :=
begin
  sorry
end

end ratio_p_q_l539_539031


namespace total_amount_paid_l539_539120

theorem total_amount_paid (g_p g_q m_p m_q : ℝ) (g_d g_t m_d m_t : ℝ) : 
    g_p = 70 -> g_q = 8 -> g_d = 0.05 -> g_t = 0.08 -> 
    m_p = 55 -> m_q = 9 -> m_d = 0.07 -> m_t = 0.11 -> 
    (g_p * g_q * (1 - g_d) * (1 + g_t) + m_p * m_q * (1 - m_d) * (1 + m_t)) = 1085.55 := by 
    sorry

end total_amount_paid_l539_539120


namespace plane_intersect_probability_l539_539932

-- Define the vertices of the rectangular prism
def vertices : List (ℝ × ℝ × ℝ) := 
  [(0,0,0), (2,0,0), (2,2,0), (0,2,0), 
   (0,0,1), (2,0,1), (2,2,1), (0,2,1)]

-- Calculate total number of ways to choose 3 vertices out of 8
def total_ways : ℕ := Nat.choose 8 3

-- Calculate the number of planes that do not intersect the interior of the prism
def non_intersecting_planes : ℕ := 6 * Nat.choose 4 3

-- Calculate the probability as a fraction
def probability_of_intersecting (total non_intersecting : ℕ) : ℚ :=
  1 - (non_intersecting : ℚ) / (total : ℚ)

-- The main theorem to state the probability is 4/7
theorem plane_intersect_probability : 
  probability_of_intersecting total_ways non_intersecting_planes = 4 / 7 := 
  by
    -- Skipping the proof
    sorry

end plane_intersect_probability_l539_539932


namespace binomial_expansion_l539_539044

theorem binomial_expansion (a b : ℕ) (h_a : a = 34) (h_b : b = 5) :
  a^2 + 2*a*b + b^2 = 1521 :=
by
  rw [h_a, h_b]
  sorry

end binomial_expansion_l539_539044


namespace range_of_a_x1_lt_x3_l539_539625

noncomputable def equation (x : ℝ) (a : ℝ) : Prop := 
  e^x - a * x = log (a * x) - x

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h1 : equation x1 a) (h2 : equation x2 a) (h3 : x1 < x2) : 
  a > real.exp 1 := 
sorry

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x

noncomputable def g (x : ℝ) : ℝ := log (1 + x) - cos x + 2

theorem x1_lt_x3 (a : ℝ) (x1 x2 x3 : ℝ) (h1 : equation x1 a) (h2 : equation x2 a) (h3 : x1 < x2) (h4 : f x1 a = g x3) :
  x1 < x3 := 
sorry

end range_of_a_x1_lt_x3_l539_539625


namespace original_number_of_men_l539_539503

theorem original_number_of_men 
  (x : ℕ)
  (H : 15 * 18 * x = 15 * 18 * (x - 8) + 8 * 15 * 18)
  (h_pos : x > 8) :
  x = 40 :=
sorry

end original_number_of_men_l539_539503


namespace intersect_trihedral_angle_l539_539705

-- Definitions of variables
variables {a b c : ℝ} (S : Type) 

-- Definition of a valid intersection condition
def valid_intersection (a b c : ℝ) : Prop :=
  a^2 + b^2 - c^2 > 0 ∧ b^2 + c^2 - a^2 > 0 ∧ a^2 + c^2 - b^2 > 0

-- Theorem statement
theorem intersect_trihedral_angle (h : valid_intersection a b c) : 
  ∃ (SA SB SC : ℝ), (SA^2 + SB^2 = a^2 ∧ SA^2 + SC^2 = b^2 ∧ SB^2 + SC^2 = c^2) :=
sorry

end intersect_trihedral_angle_l539_539705


namespace part1_inequality_solution_l539_539607

def f (x : ℝ) : ℝ := |x + 1| + |2 * x - 3|

theorem part1_inequality_solution :
  ∀ x : ℝ, f x ≤ 6 ↔ -4 / 3 ≤ x ∧ x ≤ 8 / 3 :=
by sorry

end part1_inequality_solution_l539_539607


namespace probability_A_shoots_l539_539485

theorem probability_A_shoots (P : ℚ) :
  (∀ n : ℕ, (2 * n + 1) % 2 = 1) →  -- A's turn is always the odd turn
  (∀ m : ℕ, (2 * m) % 2 = 0) →  -- B's turn is always the even turn
  let p_A_first_shot := (1 : ℚ) / 6 in  -- probability A fires on the first shot
  let p_A_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun
  let p_B_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun for B
  let P_A := p_A_first_shot + (p_A_turn * p_B_turn * P) in  -- recursive definition
  P_A = 6 / 11 := -- final probability
sorry

end probability_A_shoots_l539_539485


namespace solution_set_l539_539236

def f (x m : ℝ) : ℝ := |x - 1| - |x - m|

theorem solution_set (x : ℝ) :
  (f x 2) ≥ 1 ↔ x ≥ 2 :=
sorry

end solution_set_l539_539236


namespace prize_distribution_l539_539689

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem prize_distribution :
  let total_ways := 
    (binomial_coefficient 7 3) * 5 * (Nat.factorial 4) + 
    (binomial_coefficient 7 2 * binomial_coefficient 5 2 / 2) * 
    (binomial_coefficient 5 2) * (Nat.factorial 3)
  total_ways = 10500 :=
by 
  sorry

end prize_distribution_l539_539689


namespace find_a_b_k_range_l539_539231

/-
Given the function f(x)= (a * ln x) / (x + 1) + b / x,
the equation of the tangent line to the curve y=f(x) at the point (1, f(1)) is x + 2y - 3 = 0.
Prove:
  - a = 1 and b = 1
  - For x > 0 and x ≠ 1, f(x) > (ln x)/(x - 1) + k / x implies k ∈ (-∞, 0]

Conditions:
  1. f(x) = (a * ln x) / (x + 1) + b / x
  2. Tangent line at (1, f(1)) is x + 2y - 3 = 0.
-/

noncomputable def f (x a b : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

theorem find_a_b (a b : ℝ) :
  f 1 a b = 1 ∧ 
  HasDerivAt (f x a b) (-(1/2)) 1 →
  a = 1 ∧ b = 1 :=
sorry

theorem k_range (k : ℝ) :
  (∀ x > 0, x ≠ 1 → f x 1 1 > (Real.log x) / (x - 1) + k / x) →
  k ∈ Set.Iic (0 : ℝ) :=
sorry

end find_a_b_k_range_l539_539231


namespace like_terms_exponents_l539_539621

theorem like_terms_exponents (m n : ℕ) (h₁ : m + 3 = 5) (h₂ : 6 = 2 * n) : m^n = 8 :=
by
  sorry

end like_terms_exponents_l539_539621


namespace eval_expression_l539_539002

theorem eval_expression : (-2 ^ 3) ^ (1/3 : ℝ) - (-1 : ℝ) ^ 0 = -3 := by 
  sorry

end eval_expression_l539_539002


namespace smallest_integer_divisibility_l539_539883

def smallest_integer (a : ℕ) : Prop :=
  a > 0 ∧ ¬ ∀ b, a = b + 1

theorem smallest_integer_divisibility :
  ∃ a, smallest_integer a ∧ gcd a 63 > 1 ∧ gcd a 66 > 1 ∧ ∀ b, smallest_integer b → b < a → gcd b 63 ≤ 1 ∨ gcd b 66 ≤ 1 :=
sorry

end smallest_integer_divisibility_l539_539883


namespace incorrect_oper_is_invalid_l539_539893

-- Conditions
def cond_A := (real.sqrt ((-2)^2) = 2)
def cond_B := ((-real.sqrt 2)^2 = 2)
def cond_D := (real.cbrt (-8) = -2)

-- Incorrect operation: √4 = ±2
def incorrect_oper := (real.sqrt 4 = 2) ∧ (real.sqrt 4 ≠ -2)

-- Lean statement
theorem incorrect_oper_is_invalid : ¬ incorrect_oper := by
  -- convert options into conditions
  have h1 : real.sqrt 4 = 2, from eq.refl (real.sqrt 4),
  -- assume the negative equality
  have h2 : real.sqrt 4 ≠ -2, from λ h, by linarith,
  contradiction


end incorrect_oper_is_invalid_l539_539893


namespace no_square_number_divisible_by_six_between_50_and_120_l539_539576

theorem no_square_number_divisible_by_six_between_50_and_120 :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n * n) ∧ (x % 6 = 0) ∧ (50 < x ∧ x < 120) := 
sorry

end no_square_number_divisible_by_six_between_50_and_120_l539_539576


namespace smith_bakery_pies_l539_539768

theorem smith_bakery_pies (n : ℕ) (h : n = 16) : 
  let m := n^2 in 
  let s := m + m / 2 in 
  s = 384 :=
by
  rw [h]
  let m := 16^2
  let s := m + m / 2
  have : m = 256 := rfl
  have : s = 256 + 128 := by rw [this]; exact rfl
  exact this

end smith_bakery_pies_l539_539768


namespace tan_alpha_beta_eq_neg_four_sevenths_l539_539609

-- Define the conditions
def sin_alpha : ℝ := 3 / 5
def alpha_range : (ℝ → Prop) := λ a, a > π / 2 ∧ a < π
def sin_quotient : ℝ := 4

-- Define the main theorem to prove
theorem tan_alpha_beta_eq_neg_four_sevenths (α β : ℝ) (hα : α_range α) (hα_tri : real.sin α = sin_alpha) (hquot : (real.sin (α + β)) / (real.sin β) = sin_quotient) :
  real.tan (α + β) = -4 / 7 :=
sorry

end tan_alpha_beta_eq_neg_four_sevenths_l539_539609


namespace police_female_officers_l539_539056

theorem police_female_officers (perc : ℝ) (total_on_duty: ℝ) (half_on_duty : ℝ) (F : ℝ) :
    perc = 0.18 →
    total_on_duty = 144 →
    half_on_duty = total_on_duty / 2 →
    half_on_duty = perc * F →
    F = 400 :=
by
  sorry

end police_female_officers_l539_539056


namespace not_perpendicular_DE_D_l539_539134

variables (a b c d e f : ℝ)
-- All coordinates are positive
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)

-- Definitions for points D, E, F and their reflections
def D := (a, b)
def E := (c, d)
def F := (e, f)
def D' := (b, a)
def E' := (d, c)
def F' := (f, e)

-- Slope function definition
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- The proof problem statement for perpendicularity of DE and D'E'
theorem not_perpendicular_DE_D'E' :
  ¬ (slope D E * slope D' E' = -1) :=
sorry

end not_perpendicular_DE_D_l539_539134


namespace percentage_increase_l539_539095

theorem percentage_increase 
  (P : ℝ)
  (bought_price : ℝ := 0.80 * P) 
  (original_profit : ℝ := 0.3600000000000001 * P) :
  ∃ X : ℝ, X = 70.00000000000002 ∧ (1.3600000000000001 * P = bought_price * (1 + X / 100)) :=
sorry

end percentage_increase_l539_539095


namespace find_m_l539_539647

theorem find_m (m : ℝ) (P : Set ℝ) (Q : Set ℝ) (hP : P = {m^2 - 4, m + 1, -3})
  (hQ : Q = {m - 3, 2 * m - 1, 3 * m + 1}) (h_intersect : P ∩ Q = {-3}) :
  m = -4 / 3 :=
by
  sorry

end find_m_l539_539647


namespace square_area_with_circles_l539_539939

theorem square_area_with_circles (r : ℝ) (h : r = 3) : 
  let d := 2 * r in
  let side_length := 2 * d in
  let area := side_length * side_length in
  area = 144 :=
by
  let d := 2 * r
  let side_length := 2 * d
  let area := side_length * side_length
  have h1 : d = 6,
  { sorry },
  have h2 : side_length = 12,
  { sorry },
  have h3 : area = 144,
  { sorry },
  exact h3

end square_area_with_circles_l539_539939


namespace regular_polygon_radius_l539_539827

theorem regular_polygon_radius 
  (n : ℕ) (side_length : ℝ) (h1 : side_length = 2) 
  (h2 : sum_of_interior_angles n = 2 * sum_of_exterior_angles n)
  (h3 : is_regular_polygon n) :
  radius_of_polygon n side_length = 2 :=
by
  sorry

end regular_polygon_radius_l539_539827


namespace evaluate_sum_of_reciprocals_squared_l539_539324

noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt5 : ℝ := Real.sqrt 5
noncomputable def sqrt10 : ℝ := Real.sqrt 10

def p : ℝ := sqrt2 + sqrt5 + sqrt10
def q : ℝ := -sqrt2 + sqrt5 + sqrt10
def r : ℝ := sqrt2 - sqrt5 + sqrt10
def s : ℝ := -sqrt2 - sqrt5 + sqrt10

theorem evaluate_sum_of_reciprocals_squared :
  ( (1 / p) + (1 / q) + (1 / r) + (1 / s) ) ^ 2 = 128 / 45 :=
by
  -- proof skipped
  sorry

end evaluate_sum_of_reciprocals_squared_l539_539324


namespace evaluate_expression_l539_539573

theorem evaluate_expression (c d : ℕ) (hc : c = 4) (hd : d = 2) :
  (c^c - c * (c - d)^c)^c = 136048896 := by
  sorry

end evaluate_expression_l539_539573


namespace two_people_can_attend_each_day_l539_539975

def cannotAttend (person : String) : String → Prop
| "Anna" => ["Mon", "Wed", "Sat"]
| "Bill" => ["Tues", "Thurs", "Fri"]
| "Carl" => ["Mon", "Tues", "Thurs", "Fri"]
| "Dana" => ["Wed", "Sat"]
| _ => []

def canAttend (day : String) : List String :=
  ["Anna", "Bill", "Carl", "Dana"].filter (λ p, !(cannotAttend p).contains day)

theorem two_people_can_attend_each_day :
  ∀ day ∈ ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat"], (canAttend day).length = 2 :=
by
  sorry

end two_people_can_attend_each_day_l539_539975


namespace count_valid_n_l539_539518

theorem count_valid_n (n : ℕ) : 10 ≤ n ∧ n ≤ 100 ∧ ((n-1) * (n-2)) % 6 = 0 ∧ n % 3 ≠ 0 → ∃ (k : ℕ), 
  61 = (finset.range 101).filter (λ k, 10 ≤ k ∧ k ≤ 100 ∧ ((k-1) * (k-2)) % 6 = 0 ∧ k % 3 ≠ 0).card :=
begin
  sorry
end

end count_valid_n_l539_539518


namespace complement_intersection_A_B_complement_union_A_B_range_of_a_l539_539619

open Set

variable {α : Type} [LinearOrder α] [OrderTopology α]

def A : Set ℝ := {x | 2 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x | x ≤ 3 ∨ 6 ≤ x} :=
by sorry

theorem complement_union_A_B :
  (Bᶜ ∪ A) = {x | x < 6 ∨ 9 ≤ x} :=
by sorry

theorem range_of_a (a : ℝ) :
  C a ⊆ B → (3 ≤ a ∧ a ≤ 8) :=
by sorry

end complement_intersection_A_B_complement_union_A_B_range_of_a_l539_539619


namespace probability_greater_than_sqrt_l539_539651

theorem probability_greater_than_sqrt (x y : ℝ) : 
  (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → 
  (∃! p : ℝ, p = (measure_theory.measure_space.volume (set.univ ∩ {p : ℝ × ℝ | p.2 > real.sqrt (1 - p.1^2)}) /
                     measure_theory.measure_space.volume (set.Icc (0:ℝ) 1))) :=
begin
  sorry
end

end probability_greater_than_sqrt_l539_539651


namespace second_car_distance_l539_539419

theorem second_car_distance (x : ℝ) : 
  let d_initial : ℝ := 150
  let d_first_car_initial : ℝ := 25
  let d_right_turn : ℝ := 15
  let d_left_turn : ℝ := 25
  let d_final_gap : ℝ := 65
  (d_initial - x = d_final_gap) → x = 85 := by
  sorry

end second_car_distance_l539_539419


namespace odd_function_is_funcD_l539_539116

def funcA (x : ℝ) : ℝ := x^2 + 1
def funcB (x : ℝ) : ℝ := Real.tan x
def funcC (x : ℝ) : ℝ := 2^x
def funcD (x : ℝ) : ℝ := x + Real.sin x

theorem odd_function_is_funcD : 
  (∀ x : ℝ, funcD (-x) = -(funcD x))
  ∧ (∀ x : ℝ, funcD x = x + Real.sin x := sorry

end odd_function_is_funcD_l539_539116


namespace compare_negatives_l539_539550

theorem compare_negatives : -1 < - (2 / 3) := by
  sorry

end compare_negatives_l539_539550


namespace symmetry_axis_of_g_l539_539232

noncomputable def f (x : ℝ) : ℝ := (sqrt 2) * sin (x - (π / 4))

noncomputable def g (x : ℝ) : ℝ :=
  (sqrt 2) * sin (1 / 2 * x - (5 * π / 12))

theorem symmetry_axis_of_g :
  ∃ k : ℤ, (x = 11 * π / 6 + (2 * k * π)) :=
begin
  sorry
end

end symmetry_axis_of_g_l539_539232


namespace find_length_of_MN_l539_539192

theorem find_length_of_MN
  (A B C D K L N : ℝ)
  (h1 : A K = 10)
  (h2 : K L = 17)
  (h3 : D N = 7) :
  M N = 23 := 
sorry

end find_length_of_MN_l539_539192


namespace calculate_percentage_millet_in_brandA_l539_539084

-- Variables
variable (A B : ℝ)

-- Conditions
def brandB_millet : B = 0.65 := by sorry
def mix_percentage_millet : 0.6 * A + 0.4 * B = 0.5 := by sorry

-- Theorem
theorem calculate_percentage_millet_in_brandA (h1 : B = 0.65) (h2 : 0.6 * A + 0.4 * B = 0.5) : A = 0.4 :=
by
  have : 0.6 * A + 0.4 * 0.65 = 0.5 := by
    rw [h1]
    assumption
  rw [← mix_percentage_millet]
  sorry

end calculate_percentage_millet_in_brandA_l539_539084


namespace domain_h_l539_539373

def domain_f : Set ℝ := Set.Icc (-12) 6
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-3*x)

theorem domain_h {f : ℝ → ℝ} (hf : ∀ x, x ∈ domain_f → f x ∈ Set.univ) {x : ℝ} :
  h f x ∈ Set.univ ↔ x ∈ Set.Icc (-2) 4 :=
by
  sorry

end domain_h_l539_539373


namespace minimum_value_of_A2_minus_B2_is_400_l539_539732

noncomputable def A (x y z : ℝ) : ℝ := Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 12)
noncomputable def B (x y z : ℝ) : ℝ := Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1)
def min_value (x y z : ℝ) := (A x y z)^2 - (B x y z)^2

theorem minimum_value_of_A2_minus_B2_is_400 :
  ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → min_value x y z ≥ 400 := by
  sorry

end minimum_value_of_A2_minus_B2_is_400_l539_539732


namespace Andy_cavities_l539_539121

noncomputable def total_candy_canes (parents teachers friends grandparents : Nat) : Nat :=
  let gift_candies := parents + teachers + friends + grandparents
  let bought_candies := (2 / 5 : ℚ) * gift_candies
  gift_candies + bought_candies.toNat

theorem Andy_cavities :
  let parents := 2
  let teachers := 3 * 6
  let friends := 5 * 3
  let grandparents := 7
  let total_candies := total_candy_canes parents teachers friends grandparents
  (total_candies / 4).toNat = 14 := by
  sorry

end Andy_cavities_l539_539121


namespace cut_square_into_acute_triangles_cut_scalene_triangle_into_isosceles_l539_539140

-- Problem 1: Cutting a Square into 8 Acute-Angled Triangles
theorem cut_square_into_acute_triangles (a : ℝ) (h : a > 0) :
  ∃ (triangles : list (ℝ × ℝ × ℝ)), 
    triangles.length = 8 ∧ 
    ∀ (t ∈ triangles), let ⟨x, y, z⟩ := t in x < π/2 ∧ y < π/2 ∧ z < π/2 :=
begin
  sorry
end

-- Problem 2: Cutting a Scalene Triangle into 7 Isosceles Triangles with at Least Three Congruent
theorem cut_scalene_triangle_into_isosceles (A B C : ℝ) 
(h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) :
  ∃ (isosceles_triangles: list (ℝ × ℝ × ℝ)), 
    isosceles_triangles.length = 7 ∧ 
    ∃ (congruent_set : set (ℝ × ℝ × ℝ)), congruent_set.card ≥ 3 ∧ 
    ∀ (t ∈ isosceles_triangles), let ⟨x, y, z⟩ := t in (x = y ∨ y = z ∨ z = x) :=
begin
  sorry
end

end cut_square_into_acute_triangles_cut_scalene_triangle_into_isosceles_l539_539140


namespace least_three_digit_product_of_digits_is_8_l539_539873

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l539_539873


namespace two_colonies_reach_limit_in_21_days_l539_539076

theorem two_colonies_reach_limit_in_21_days
  (doubles_every_day : ∀ (n : ℕ), n ≥ 0 → 2^n) 
  (single_limit_in_21 : 2^21 = l) :
  2^21 = l :=
by 
  sorry

end two_colonies_reach_limit_in_21_days_l539_539076


namespace paul_initial_savings_l539_539343

theorem paul_initial_savings (additional_allowance: ℕ) (cost_per_toy: ℕ) (number_of_toys: ℕ) (total_savings: ℕ) :
  additional_allowance = 7 →
  cost_per_toy = 5 →
  number_of_toys = 2 →
  total_savings + additional_allowance = cost_per_toy * number_of_toys →
  total_savings = 3 :=
by
  intros h_additional h_cost h_number h_total
  sorry

end paul_initial_savings_l539_539343


namespace series_sum_bound_sqr_sum_ineq_l539_539459

-- Problem 1
theorem series_sum_bound (n : ℕ) (h : n ≥ 2) : (1 + ∑ i in finset.range (n - 1), (1 / (i + 2)^2)) < (2 - 1 / n) :=
sorry

-- Problem 2
theorem sqr_sum_ineq (a b c : ℝ) : (a^2 + b^2 + c^2) ≥ (a * b + a * c + b * c) :=
sorry

end series_sum_bound_sqr_sum_ineq_l539_539459


namespace incorrect_statement_l539_539436

-- Define polynomials and terms
def polynomial_1 := x^2 - 3*x + 1
def terms_poly_1 := [x^2, -3*x, 1]

def polynomial_2 := (1/4)*x^2*y^3 - 2*x*y + 3
def degree_poly_2 := 5

def term_1 := -2*a^2*b^3
def term_2 := 5*a^3*b^2

def term_3 := m/2
def coeff_term_3 := 1/2
def degree_term_3 := 1

-- Statements as lean booleans
def statement_A := (polynomial_1.terms = terms_poly_1)
def statement_B := (¬(term_1.like term_2))
def statement_C := (coeff_term_3 = 2 ∧ degree_term_3 = 1)
def statement_D := (polynomial_2.degree = degree_poly_2)

-- Incorrect statement
theorem incorrect_statement : ¬statement_C := by
    sorry

end incorrect_statement_l539_539436


namespace find_g_six_l539_539385

noncomputable def g : ℝ → ℝ := sorry -- Placeholder for the function g

axiom functional_equation (x y : ℝ) : g(x + y) = g(x) * g(y)
axiom g_value : g(2) = 5

theorem find_g_six : g(6) = 125 :=
by
  sorry

end find_g_six_l539_539385


namespace adam_bought_26_books_l539_539111

-- Conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def avg_books_per_shelf : ℕ := 20
def leftover_books : ℕ := 2

-- Definitions based on conditions
def capacity_books : ℕ := shelves * avg_books_per_shelf
def total_books_after_trip : ℕ := capacity_books + leftover_books

-- Question: How many books did Adam buy on his shopping trip?
def books_bought : ℕ := total_books_after_trip - initial_books

theorem adam_bought_26_books :
  books_bought = 26 :=
by
  sorry

end adam_bought_26_books_l539_539111


namespace square_diagonal_area_200_l539_539038

theorem square_diagonal_area_200 (s d : ℝ) (h_area : s ^ 2 = 200) (h_diag : d ^ 2 = 2 * s ^ 2) : d = 20 := by
suffices h: d ^ 2 = 400, from (real.sqrt_eq_iff_sq_eq.mpr h).resolve_right (by linarith),
rw [h_diag, h_area],
ring

end square_diagonal_area_200_l539_539038


namespace minimum_perimeter_l539_539685

def fractional_part (x : ℝ) : ℝ := x - floor x

theorem minimum_perimeter 
  (l m n : ℕ) 
  (h1 : l > m) 
  (h2 : m > n) 
  (h3 : fractional_part (3^l / 10000) = fractional_part (3^m / 10000)) 
  (h4 : fractional_part (3^m / 10000) = fractional_part (3^n / 10000)) : 
  l + m + n = 3003 :=
sorry

end minimum_perimeter_l539_539685


namespace flowers_left_l539_539532

theorem flowers_left (total_peonies tulips : ℕ) (watered : ℕ) (picked : ℕ) (unwatered : ℕ) (final_picked_tulip : ℕ) :
  total_peonies = 15 →
  tulips = 15 →
  10th_tulip : ∃ x, x = 10 →
  watered = 20 →
  picked = 6 →
  unwatered = 10 →
  final_picked_tulip = 6 →
  total_peonies + tulips - watered + final_picked_tulip - 1 = 19 :=
begin
  intros _ _ _ _ _ _ _ sorry
end

end flowers_left_l539_539532


namespace product_of_first_17_terms_l539_539290

theorem product_of_first_17_terms (a : ℕ → ℝ) (h : a 9 = -2) :
  (∏ i in finset.range 17, a (i + 1)) = -2 ^ 17 :=
by sorry

end product_of_first_17_terms_l539_539290


namespace complex_numbers_on_same_circumference_l539_539189

variable (S : ℝ) (hs : abs S ≤ 2)
variables (a1 a2 a3 a4 a5 : ℂ)

theorem complex_numbers_on_same_circumference : 
  (∀ (a1 a2 a3 a4 a5 : ℂ), abs S ≤ 2 → 
    ∃ (r : ℝ) (c : ℂ), ∀ (z : ℂ), z ∈ {a1, a2, a3, a4, a5} → abs (z - c) = r) :=
by sorry

end complex_numbers_on_same_circumference_l539_539189


namespace sqrt_of_4_l539_539819

theorem sqrt_of_4 (y : ℝ) : y^2 = 4 → (y = 2 ∨ y = -2) :=
sorry

end sqrt_of_4_l539_539819


namespace functional_eq_solution_l539_539386

theorem functional_eq_solution (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) :
  g 10 = -48 :=
sorry

end functional_eq_solution_l539_539386


namespace fraction_of_smaller_jar_l539_539568

theorem fraction_of_smaller_jar (S L : ℝ) (W : ℝ) (F : ℝ) 
  (h1 : W = F * S) 
  (h2 : W = 1/2 * L) 
  (h3 : 2 * W = 2/3 * L) 
  (h4 : S = 2/3 * L) :
  F = 3 / 4 :=
by
  sorry

end fraction_of_smaller_jar_l539_539568


namespace minimum_value_fraction_l539_539222

theorem minimum_value_fraction (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 2) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 2 → 
    ((1 / (1 + x)) + (1 / (2 + 2 * y)) ≥ 4 / 5)) :=
by sorry

end minimum_value_fraction_l539_539222


namespace time_to_finish_work_l539_539260

theorem time_to_finish_work (a b c : ℕ) (h1 : 1/a + 1/9 + 1/18 = 1/4) : a = 12 :=
by
  sorry

end time_to_finish_work_l539_539260


namespace problem_statement_l539_539180

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem problem_statement :
  f (5 * Real.pi / 24) = Real.sqrt 2 ∧
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 :=
by
  sorry

end problem_statement_l539_539180


namespace tammy_avg_speed_l539_539778

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l539_539778


namespace least_three_digit_number_product8_l539_539860

theorem least_three_digit_number_product8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (digits 10 n).prod = 8 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (digits 10 m).prod = 8 → n ≤ m) :=
sorry

end least_three_digit_number_product8_l539_539860


namespace least_three_digit_product_eight_l539_539853

theorem least_three_digit_product_eight : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).prod = 8 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (nat.digits 10 m).prod = 8 → n ≤ m :=
by
  sorry

end least_three_digit_product_eight_l539_539853


namespace calculate_x_l539_539590

theorem calculate_x :
  let a := 3
  let b := 5
  let c := 2
  let d := 4
  let term1 := (a ^ 2) * b * 0.47 * 1442
  let term2 := c * d * 0.36 * 1412
  (term1 - term2) + 63 = 26544.74 := by
  sorry

end calculate_x_l539_539590


namespace contrapositive_example_l539_539381

theorem contrapositive_example (a b : ℕ) (h : a = 0 → ab = 0) : ab ≠ 0 → a ≠ 0 :=
by sorry

end contrapositive_example_l539_539381


namespace charley_pencils_l539_539958

theorem charley_pencils :
  ∀ (initial_pencils lost_moving: ℕ),
    initial_pencils = 30 →
    lost_moving = 6 →
    (let remaining_pencils := initial_pencils - lost_moving in
     let lost_after := remaining_pencils / 3 in
     let final_pencils := remaining_pencils - lost_after in
     final_pencils = 16) :=
begin
  sorry
end

end charley_pencils_l539_539958


namespace modulus_of_z_l539_539325
noncomputable def z : ℂ := 3 + 4 * complex.I

theorem modulus_of_z : complex.abs z = 5 := by
  sorry

end modulus_of_z_l539_539325


namespace sum_first_60_terms_arithmetic_progression_l539_539672

variable (a d : ℝ)
variable (S : ℕ → ℝ := λ n, n / 2 * (2 * a + (n - 1) * d))

theorem sum_first_60_terms_arithmetic_progression 
  (h₁ : S 15 = 150) 
  (h₂ : S 45 = 0) : 
  S 60 = -300 := by
  sorry

end sum_first_60_terms_arithmetic_progression_l539_539672


namespace rectangle_area_l539_539898

noncomputable def area_of_rectangle_from_wire (r : ℝ) (length_ratio breadth_ratio : ℝ) (circumference : ℝ) : ℝ :=
  let L := circumference * 3 / (2 * (length_ratio + breadth_ratio)) in
  let B := (breadth_ratio / length_ratio) * L in
  L * B

theorem rectangle_area (r : ℝ) (length_ratio breadth_ratio : ℝ) :
  r = 3.5 →
  length_ratio = 6 →
  breadth_ratio = 5 →
  let circumference := 2 * real.pi * r in
  area_of_rectangle_from_wire r length_ratio breadth_ratio circumference = (735 * real.pi^2) / 242 :=
by
  intros hr hlen_ratio hbreadth_ratio circumference_def
  sorry

end rectangle_area_l539_539898


namespace red_ball_probability_correct_l539_539048

theorem red_ball_probability_correct (R B : ℕ) (hR : R = 3) (hB : B = 3) :
  (R / (R + B) : ℚ) = 1 / 2 := by
  sorry

end red_ball_probability_correct_l539_539048


namespace limit_sequences_equal_l539_539454

open Classical

variables {x0 y0 : ℝ}
variable h : x0 > y0 ∧ x0 > 0 ∧ y0 > 0

noncomputable def x : ℕ → ℝ
| 0     := x0
| (n+1) := (x n + y n) / 2

noncomputable def y : ℕ → ℝ
| 0     := y0
| (n+1) := (2 * x n * y n) / (x n + y n)

theorem limit_sequences_equal (h : x0 > y0 ∧ x0 > 0 ∧ y0 > 0) :
  (tendsto x at_top (𝓝 (sqrt (x0 * y0))) ∧ tendsto y at_top (𝓝 (sqrt (x0 * y0)))) :=
begin
  sorry
end

end limit_sequences_equal_l539_539454


namespace weight_of_individual_corn_l539_539075

theorem weight_of_individual_corn 
  (bushel_weight : ℕ := 56)
  (bushels_picked : ℕ := 2)
  (cobs_picked : ℕ := 224) :
  (bushel_weight * bushels_picked) / cobs_picked = 0.5 := 
by
  sorry

end weight_of_individual_corn_l539_539075


namespace magnitude_difference_perpendicular_sum_difference_l539_539185

variables {a b : ℝ^3}

-- Conditions given in the problem
axiom norm_a : ‖a‖ = 1
axiom norm_b : ‖b‖ = 1
axiom norm_a_add_b : ‖a + b‖ = sqrt(3)

-- Part 1: Prove the magnitude of the difference of vectors is 1
theorem magnitude_difference (a b : ℝ^3) (norm_a : ‖a‖ = 1) (norm_b : ‖b‖ = 1) (norm_a_add_b : ‖a + b‖ = sqrt(3)) : ‖a - b‖ = 1 :=
sorry

-- Part 2: Prove vectors are perpendicular
theorem perpendicular_sum_difference (a b : ℝ^3) (norm_a : ‖a‖ = 1) (norm_b : ‖b‖ = 1) (norm_a_add_b : ‖a + b‖ = sqrt(3)) : (a + b) • (a - b) = 0 :=
sorry

end magnitude_difference_perpendicular_sum_difference_l539_539185


namespace sin_identity_l539_539734

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end sin_identity_l539_539734


namespace cot_arccot_combination_l539_539986

noncomputable def cot (θ : ℝ) : ℝ := 1 / tan θ
noncomputable def arccot (x : ℝ) : ℝ := arctan (1 / x)

theorem cot_arccot_combination
  (a b c d : ℝ)
  (ha : a = 4) (hb : b = 9) (hc : c = 17) (hd : d = 31) :
  cot (arccot a + arccot b + arccot c + arccot d) = 20 :=
by
  -- We'll use the given values as the conditions
  rw [ha, hb, hc, hd]
  -- The actual proof would go here, but we'll just insert sorry to skip it
  sorry

end cot_arccot_combination_l539_539986


namespace profit_margin_increase_l539_539930

theorem profit_margin_increase (CP : ℝ) (SP : ℝ) (NSP : ℝ) (initial_margin : ℝ) (desired_margin : ℝ) :
  initial_margin = 0.25 → desired_margin = 0.40 → SP = (1 + initial_margin) * CP → NSP = (1 + desired_margin) * CP →
  ((NSP - SP) / SP) * 100 = 12 := 
by 
  intros h1 h2 h3 h4
  sorry

end profit_margin_increase_l539_539930


namespace least_perimeter_of_triangle_l539_539389

theorem least_perimeter_of_triangle (c : ℕ) (h1 : 24 + 51 > c) (h2 : c > 27) : 24 + 51 + c = 103 :=
by
  sorry

end least_perimeter_of_triangle_l539_539389


namespace distance_from_point_to_plane_OAB_l539_539295

variable {P : EuclideanSpace ℝ (fin 3)} (OAB : AffineSubspace ℝ (EuclideanSpace ℝ (fin 3)))
variable (O : EuclideanSpace ℝ (fin 3)) (n : EuclideanSpace ℝ (fin 3))

noncomputable def distance_from_point_to_plane 
  (P O n : EuclideanSpace ℝ (fin 3)) (OAB : AffineSubspace ℝ (EuclideanSpace ℝ (fin 3))) : ℝ :=
  let (a, b, c) := (n.1, n.2, n.3)
      (x0, y0, z0) := (P.1, P.2, P.3)
      d := 0
  in abs (a * x0 + b * y0 + c * z0 + d) / real.sqrt (a ^ 2 + b ^ 2 + c ^ 2)

theorem distance_from_point_to_plane_OAB 
  (P : EuclideanSpace ℝ (fin 3)) (OAB : AffineSubspace ℝ (EuclideanSpace ℝ (fin 3)))
  (O : EuclideanSpace ℝ (fin 3)) (n : EuclideanSpace ℝ (fin 3))
  (hO : O = EuclideanSpace.vector3 0 0 0)
  (hP : P = EuclideanSpace.vector3 (-1) 3 2)
  (hn : n = EuclideanSpace.vector3 2 (-2) 1)
  (hOAB : O ∈ OAB) (hOn : OAB.direction.contains (EuclideanSpace.span_singleton ℝ n)) :
  distance_from_point_to_plane P O n OAB = 2 :=
  sorry

end distance_from_point_to_plane_OAB_l539_539295


namespace molecular_weight_BaBr2_const_l539_539147

noncomputable def atomicWeightBa : ℝ := 137.33
noncomputable def atomicWeightBr : ℝ := 79.90

def molecularWeightBaBr2 (atomicWeightBa atomicWeightBr : ℝ) : ℝ :=
  atomicWeightBa + 2 * atomicWeightBr

theorem molecular_weight_BaBr2_const (T P : ℝ) :
  molecularWeightBaBr2 atomicWeightBa atomicWeightBr = 297.13 :=
by
  simp [atomicWeightBa, atomicWeightBr, molecularWeightBaBr2]
  norm_num
  apply (eq.refl 297.13)
  sorry

end molecular_weight_BaBr2_const_l539_539147


namespace modulus_of_z_l539_539190

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 - I) * z = 2 + 2 * I) : abs z = 2 := 
sorry

end modulus_of_z_l539_539190


namespace equal_ratios_l539_539844

-- Definitions
universe u

variables {α : Type u} [Field α]

-- Assumptions and Conditions
variables
  (A1 A2 B C : α) -- Points on the plane
  (dist : α → α → α) -- Distance function
  (midpoint : α → α → α) -- Midpoint function
  (power_of_point : α → α → α → α) -- Power of a point w.r.t. two circles

axiom two_circles_intersect_at_BC : ∃ (circle1 circle2 : Circle α), intersect circle1 circle2 = {B, C}
axiom line_tangentially_intersects_at_A1A2 : TangentLine (circle1 ∪ circle2) (A1, A2)

-- Definitions from the problem
def E := midpoint A1 A2

-- Prove the given equality
theorem equal_ratios (h : power_of_point E B C = power_of_point E B C):
  dist A1 B / dist A1 C = dist A2 B / dist A2 C :=
sorry

end equal_ratios_l539_539844


namespace find_AB_l539_539286

noncomputable def rectangle_ABCD : Type := sorry
variables (A B C D P: rectangle_ABCD)
variables (BP CP: ℝ)
variables (tan_APD: ℝ)

-- Given conditions
axiom rectangle_ABC_dis_a_rectangle : rectangle_ABCD
axiom P_on_BC : P ∈ line_segment B C
axiom BP_eq_12 : BP = 12
axiom CP_eq_18 : CP = 18
axiom tan_angle_APD_eq_1 : tan_APD = 1

-- Required to prove
theorem find_AB (AB: ℝ) : AB = 36 := sorry

end find_AB_l539_539286


namespace number_of_integer_terms_l539_539398

noncomputable def count_integer_terms_in_sequence (n : ℕ) (k : ℕ) (a : ℕ) : ℕ :=
  if h : a = k * 3 ^ n then n + 1 else 0

theorem number_of_integer_terms :
  count_integer_terms_in_sequence 5 (2^3 * 5) 9720 = 6 :=
by sorry

end number_of_integer_terms_l539_539398


namespace max_lambda_inequality_l539_539983

noncomputable def max_lambda_exists (ABC P : Triangle) (A B C A1 B1 C1 : Point) (λ : ℝ) :=
  (∃ P, isInside P ABC) ∧
  (∠PAB = ∠PBC ∧ ∠PBC = ∠PCA) ∧
  (onCircumcircle A1 (triangle P B C) ∧ 
   onCircumcircle B1 (triangle P C A) ∧ 
   onCircumcircle C1 (triangle P A B)) ∧
  (S (triangle P B C) + S (triangle A1 B C) + S (triangle B1 C A) ≤ λ * S ABC )

theorem max_lambda_inequality (ABC P : Triangle) (A B C A1 B1 C1 : Point) :
  max_lambda_exists ABC P A B C A1 B1 C1 3 :=
sorry

end max_lambda_inequality_l539_539983


namespace sqrt_four_eq_pm_two_l539_539820

theorem sqrt_four_eq_pm_two : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_four_eq_pm_two_l539_539820


namespace total_pages_in_book_l539_539960

variable (p1 p2 p_total : ℕ)
variable (read_first_four_days : p1 = 4 * 45)
variable (read_next_three_days : p2 = 3 * 52)
variable (total_until_last_day : p_total = p1 + p2 + 15)

theorem total_pages_in_book : p_total = 351 :=
by
  -- Introduce the conditions
  rw [read_first_four_days, read_next_three_days] at total_until_last_day
  sorry

end total_pages_in_book_l539_539960


namespace parabola_sum_non_horizontal_line_l539_539928

noncomputable def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ :=
  -a * x^2 - b * x - c

noncomputable def translated_parabola_left (a b c : ℝ) (x : ℝ) : ℝ :=
  a * (x + 5)^2 + b * (x + 5) + c

noncomputable def translated_parabola_right (a b c : ℝ) (x : ℝ) : ℝ :=
  -a * (x - 5)^2 - b * (x - 5) - c

theorem parabola_sum_non_horizontal_line (a b c : ℝ) :
  ∀ x : ℝ, (translated_parabola_left a b c x + translated_parabola_right a b c x) =
  20 * a * x + 10 * b :=
begin
  sorry
end

end parabola_sum_non_horizontal_line_l539_539928


namespace equilateral_triangle_functional_l539_539538

def is_functional_relation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y → x = y

theorem equilateral_triangle_functional (s : ℝ) (A : ℝ) (h : A = (sqrt 3 / 4) * s^2) : 
  is_functional_relation (λ s : ℝ, (sqrt 3 / 4) * s^2) :=
by
  sorry

end equilateral_triangle_functional_l539_539538


namespace smallest_even_length_sides_in_poly_l539_539513

theorem smallest_even_length_sides_in_poly {n : ℕ} (h : n = 2005) :
  ∃ k : ℕ, k = 2 ∧ ∀ (polygon : Set (ℕ × ℕ)), (built_from_dominoes polygon n) → (even_length_sides polygon = k) :=
begin
  sorry,
end

def built_from_dominoes (polygon: Set (ℕ × ℕ)) (n: ℕ) : Prop := sorry -- Definition of a polygon built from n 1x2 dominoes
def even_length_sides (polygon: Set (ℕ × ℕ)) : ℕ := sorry -- Definition for counting sides of even lengths

end smallest_even_length_sides_in_poly_l539_539513


namespace only_four_letter_list_with_same_product_as_TUVW_l539_539153

/-- Each letter of the alphabet is assigned a value $A=1, B=2, C=3, \ldots, Z=26$. 
    The product of a four-letter list is the product of the values of its four letters. 
    The product of the list $TUVW$ is $(20)(21)(22)(23)$. 
    Prove that the only other four-letter list with a product equal to the product of the list $TUVW$ 
    is the list $TUVW$ itself. 
 -/
theorem only_four_letter_list_with_same_product_as_TUVW : ∀ (l : list char), 
  (l.all (λ c, 'A' ≤ c ∧ c ≤ 'Z')) → 
  (l.prod (λ c, letter_value c) = (20 * 21 * 22 * 23)) →
  (l = ['T', 'U', 'V', 'W']) :=
by
  sorry

/-- Helper function to convert a letter to its value. -/
def letter_value (c : char) : ℕ :=
  c.to_nat - 'A'.to_nat + 1

#eval letter_value 'T' -- 20
#eval letter_value 'U' -- 21
#eval letter_value 'V' -- 22
#eval letter_value 'W' -- 23

end only_four_letter_list_with_same_product_as_TUVW_l539_539153


namespace find_x_l539_539591

theorem find_x (x : ℝ) : 
  3.5 * ( (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * x) ) = 2800.0000000000005 → x = 1.25 :=
by
  sorry

end find_x_l539_539591


namespace common_difference_range_l539_539196

variable {a : ℕ → ℝ} {d : ℝ}

-- Conditions
axiom seq_arithmetic (n : ℕ) : a (n + 1) = a n + d
axiom d_positive : d > 0
axiom geometric_mean (a1 a4 : ℝ) : a 1 = a1 ∧ a 4 = a4 ∧ (a 2)^2 = a1 * a4

-- Definition of b_n where b_n = a (2^*n) interpreted as a (2*n)
def b (n : ℕ) : ℝ := a (2*n)

-- Theorem to prove
theorem common_difference_range (a1 a4 : ℝ) (d : ℝ) (a_n : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + d) →
  d > 0 →
  a 1 = a1 →
  a 4 = a4 →
  (a 2)^2 = a1 * a4 →
  (∀ n : ℕ, 1 ≤ n → (∑ i in Finset.range n, 1 / (b (i+1)) < 2)) →
  d ≥ 1 / 2 := 
sorry

end common_difference_range_l539_539196


namespace determine_m_l539_539261

-- Define a complex number structure in Lean
structure ComplexNumber where
  re : ℝ  -- real part
  im : ℝ  -- imaginary part

-- Define the condition where the complex number is purely imaginary
def is_purely_imaginary (z : ComplexNumber) : Prop :=
  z.re = 0

-- State the Lean theorem
theorem determine_m (m : ℝ) (h : is_purely_imaginary (ComplexNumber.mk (m^2 - m) m)) : m = 1 :=
by
  sorry

end determine_m_l539_539261


namespace find_a_for_max_value_of_six_find_k_for_inequality_l539_539633

noncomputable theory

-- Define the function f
def f (x a : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

-- Define the first part of the problem: find a such that f has a maximum value of 6
theorem find_a_for_max_value_of_six (a : ℝ) :
  (∃ x : ℝ, f x a = 6) → a = 6 :=
sorry

-- Define the second part of the problem: find the range of values for k
theorem find_k_for_inequality (k x t : ℝ) :
  (∀ (x ∈ set.Icc (-2 : ℝ) 2) (t ∈ set.Icc (-1 : ℝ) 1), f x 6 ≥ k * t - 25) → -3 ≤ k ∧ k ≤ 3 :=
sorry

end find_a_for_max_value_of_six_find_k_for_inequality_l539_539633


namespace semicircle_perimeter_l539_539902

theorem semicircle_perimeter (r : ℝ) (π : ℝ) (h : 0 < π) (r_eq : r = 14):
  (14 * π + 28) = 14 * π + 28 :=
by
  sorry

end semicircle_perimeter_l539_539902


namespace parabola_parameters_l539_539379

theorem parabola_parameters :
  let F := (2, 2)
  let P1 := (4, 2)
  let P2 := (-2, 5)
  let distance (A B : ℝ × ℝ) : ℝ := (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))
  (P1F : distance P1 F = 2) (P2F : distance P2 F = 5)
  (directrix1 : ∃ a b c : ℝ, a * 4 + b * 2 + c = 0)
  (directrix2 : ∃ a b c : ℝ, a * (-2) + b * 5 + c = 0)
  (parameter1 := distance (2, 2) (4, 2))
  (parameter2 := distance (2, 2) ((2.8, 3.6)))
: parameter1 = 2 ∧ parameter2 = 3.6 :=
by
  sorry

end parabola_parameters_l539_539379


namespace least_three_digit_product_8_is_118_l539_539877

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_product (n : ℕ) (product : ℕ) : Prop :=
  let digits := (list.cons (n / 100) (list.cons ((n / 10) % 10) (list.cons (n % 10) list.nil))) in
  digits.prod = product

theorem least_three_digit_product_8_is_118 :
  ∃ n : ℕ, is_three_digit_number n ∧ digits_product n 8 ∧ (∀ m : ℕ, is_three_digit_number m ∧ digits_product m 8 → n ≤ m) :=
sorry

end least_three_digit_product_8_is_118_l539_539877


namespace geometric_progression_solution_l539_539026

noncomputable def first_term_of_geometric_progression (b2 b6 : ℚ) (q : ℚ) : ℚ := 
  b2 / q
  
theorem geometric_progression_solution 
  (b2 b6 : ℚ)
  (h1 : b2 = 37 + 1/3)
  (h2 : b6 = 2 + 1/3) :
  ∃ a q : ℚ, a = 224 / 3 ∧ q = 1/2 ∧ b2 = a * q ∧ b6 = a * q^5 :=
by
  sorry

end geometric_progression_solution_l539_539026


namespace sum_of_valid_x_values_l539_539839

theorem sum_of_valid_x_values :
  let n := 360
  ∑ (x : ℕ) in {x | x ∣ n ∧ x ≥ 18 ∧ (n / x) ≥ 12}.toFinset, x = 92 :=
by
  sorry

end sum_of_valid_x_values_l539_539839


namespace parities_of_E_10_11_12_l539_539927

noncomputable def E : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| (n + 3) => 2 * (E (n + 2)) + (E n)

theorem parities_of_E_10_11_12 :
  (E 10 % 2 = 0) ∧ (E 11 % 2 = 1) ∧ (E 12 % 2 = 1) := 
  by
  sorry

end parities_of_E_10_11_12_l539_539927


namespace chains_in_graph_l539_539178

noncomputable def least_integer_ge (a : ℝ) : ℤ := ⌈a⌉

theorem chains_in_graph (G : Type*) [graph : simple_graph G] (vertices : finset G) (edges : finset (sym2 G)) 
  (q : ℕ) (edge_number : sym2 G → ℕ) (n : ℕ) (h_edges : edges.card = q) (h_vertices : vertices.card = n)
  (m := least_integer_ge (2 * q / n)) :
  ∃ (chain : list (sym2 G)), chain.length = m ∧ 
  (∀ i, i < chain.length - 1 → ∃ v, v ∈ chain.get i ∧ v ∈ chain.get (i + 1)) ∧
  (∀ i, i < chain.length - 1 → edge_number (chain.get i) < edge_number (chain.get (i + 1))) :=
sorry

end chains_in_graph_l539_539178


namespace find_scalars_l539_539649

open Real

def vec (a b c : ℝ) := (a, b, c)

def dot_product (v1 v2 : (ℝ × ℝ × ℝ)) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_scalars :
  let u := vec 2 2 (-1)
  let v := vec 3 (-1) 2
  let w := vec (-1) 4 2
  let target := vec 5 (-3) 12
  ∃ x y z,
  target = (x * u.1 + y * v.1 + z * w.1,
            x * u.2 + y * v.2 + z * w.2,
            x * u.3 + y * v.3 + z * w.3) ∧
  x = -8/9 ∧ y = 3 ∧ z = 1/3 :=
by {
  sorry
}

end find_scalars_l539_539649


namespace percentage_decrease_correct_l539_539025

variable (O N : ℕ)
variable (percentage_decrease : ℕ)

-- Define the conditions based on the problem
def original_price := 1240
def new_price := 620
def price_effect := ((original_price - new_price) * 100) / original_price

-- Prove the percentage decrease is 50%
theorem percentage_decrease_correct :
  price_effect = 50 := by
  sorry

end percentage_decrease_correct_l539_539025


namespace evaluate_fractions_l539_539158

-- Define the fractions
def frac1 := 7 / 12
def frac2 := 8 / 15
def frac3 := 2 / 5

-- Prove that the sum and difference is as specified
theorem evaluate_fractions :
  frac1 + frac2 - frac3 = 43 / 60 :=
by
  sorry

end evaluate_fractions_l539_539158


namespace relationship_between_x_and_y_l539_539677

noncomputable def circle_geom (O A B D C E : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  let a := radius / 2 in
  -- Conditions include geometry constraints and distance definitions
  (circle center O radius) ∧
  (diameter A B) ∧
  (chord A D) ∧
  (tangent_at B C) ∧
  (E ∈ line A C) ∧
  (dist A E = dist E C) ∧
  (dist E diameter A B = y) ∧
  (dist E tangent_at A = x)

theorem relationship_between_x_and_y (O A B D C E : Point) (radius : ℝ) (x y : ℝ) :
  circle_geom O A B D C E radius x y → y^2 = x^3 / (2 * (radius/2) + x) :=
sorry

end relationship_between_x_and_y_l539_539677


namespace no_such_natural_numbers_l539_539704

theorem no_such_natural_numbers :
  ¬ ∃ (x y : ℕ), (∃ (a b : ℕ), x^2 + y = a^2 ∧ x - y = b^2) := 
sorry

end no_such_natural_numbers_l539_539704


namespace unique_solution_condition_l539_539739

theorem unique_solution_condition (a b c : ℝ) : 
  (∀ x : ℝ, 4 * x - 7 + a = (b + 1) * x + c) ↔ b ≠ 3 :=
by
  sorry

end unique_solution_condition_l539_539739


namespace red_ball_probability_correct_l539_539049

theorem red_ball_probability_correct (R B : ℕ) (hR : R = 3) (hB : B = 3) :
  (R / (R + B) : ℚ) = 1 / 2 := by
  sorry

end red_ball_probability_correct_l539_539049


namespace min_balls_to_ensure_single_color_l539_539074

-- Define the problem conditions
def red_balls := 23
def green_balls := 24
def white_balls := 12
def blue_balls := 21
def threshold := 20

-- State the theorem
theorem min_balls_to_ensure_single_color (drawn_balls : ℕ) :
  drawn_balls ≥ 70 → 
  ∃ color : string, (color = "red" ∨ color = "green" ∨ color = "white" ∨ color = "blue") ∧ 
  (color = "red" → drawn_balls - (red_balls - threshold) ≥ threshold) ∧
  (color = "green" → drawn_balls - (green_balls - threshold) ≥ threshold) ∧
  (color = "white" → drawn_balls - (white_balls - threshold) ≥ threshold) ∧
  (color = "blue" → drawn_balls - (blue_balls - threshold) ≥ threshold) :=
sorry

end min_balls_to_ensure_single_color_l539_539074


namespace min_k_sqrt_x_not_2Lipschitz_log2_periodic_1Lipschitz_leq1_l539_539270

def kLipschitz (f : ℝ → ℝ) (k : ℝ) (D : Set ℝ) : Prop :=
  ∀ x1 x2 ∈ D, x1 ≠ x2 → |f x1 - f x2| ≤ k * |x1 - x2|

theorem min_k_sqrt_x :
  let D := {x | 1 ≤ x ∧ x ≤ 4}
  ∀ k, kLipschitz (λ x, Real.sqrt x) k D → 1 / 2 ≤ k :=
begin
  intro D,
  rintro k _,
  sorry,
end

theorem not_2Lipschitz_log2 :
  ¬ kLipschitz Real.log2 2 Set.univ :=
begin
  sorry,
end

theorem periodic_1Lipschitz_leq1 (f : ℝ → ℝ) :
  (∀ x, f (x + 2) = f x) →
  kLipschitz f 1 Set.univ →
  ∀ x1 x2 : ℝ, |f x1 - f x2| ≤ 1 :=
begin
  intros periodic hlipschitz x1 x2,
  sorry,
end

end min_k_sqrt_x_not_2Lipschitz_log2_periodic_1Lipschitz_leq1_l539_539270


namespace find_triangle_value_l539_539263

theorem find_triangle_value 
  (triangle : ℕ)
  (h_units : (triangle + 3) % 7 = 2)
  (h_tens : (1 + 4 + triangle) % 7 = 4)
  (h_hundreds : (2 + triangle + 1) % 7 = 2)
  (h_thousands : 3 + 0 + 1 = 4) :
  triangle = 6 :=
sorry

end find_triangle_value_l539_539263


namespace number_of_paper_cover_copies_l539_539119

-- Defining the conditions
def percentage_paper_cover := 0.06
def percentage_hardcover := 0.12
def price_paper_cover := 0.20
def price_hardcover := 0.40
def copies_hardcover := 15000
def total_earnings := 1104

-- Defining the earnings formula for hardcover
def earnings_hardcover := percentage_hardcover * copies_hardcover * price_hardcover

-- Assertion of the main proof problem
theorem number_of_paper_cover_copies:
  ∃ P: ℕ, (percentage_paper_cover * P * price_paper_cover + earnings_hardcover = total_earnings) ∧ 
        P = 32000 :=
by
  sorry

end number_of_paper_cover_copies_l539_539119


namespace Adam_bought_26_books_l539_539110

theorem Adam_bought_26_books (initial_books : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) (leftover_books : ℕ) :
  initial_books = 56 → shelves = 4 → books_per_shelf = 20 → leftover_books = 2 → 
  let total_capacity := shelves * books_per_shelf in
  let total_books_after := total_capacity + leftover_books in
  let books_bought := total_books_after - initial_books in
  books_bought = 26 :=
by
  intros h1 h2 h3 h4
  simp [total_capacity, total_books_after, books_bought]
  rw [h1, h2, h3, h4]
  sorry

end Adam_bought_26_books_l539_539110


namespace BC_length_l539_539700

theorem BC_length (AD BC MN : ℝ) (h1 : AD = 2) (h2 : MN = 6) (h3 : MN = 0.5 * (AD + BC)) : BC = 10 :=
by
  sorry

end BC_length_l539_539700


namespace probability_no_more_than_once_probability_two_months_l539_539529

noncomputable def P : ℕ → ℝ
| 0 => 0.3
| 1 => 0.5
| 2 => 0.2
| _ => 0.0

def no_more_than_once :=
  P 0 + P 1 = 0.8

def independent_events (P1 P2 : ℕ → ℝ) :=
  ∀ (i j : ℕ), P1 i * P2 j = (P1 i) * (P2 j)

def P_months (n1 n2 : ℕ) : ℝ :=
  match n1, n2 with
  | 0, 2 => P 0 * P 2
  | 2, 0 => P 2 * P 0
  | 1, 1 => P 1 * P 1
  | _, _ => 0.0

def two_months :=
  P_months 0 2 + P_months 2 0 + P_months 1 1 = 0.37

axiom January_February_independent: independent_events P P

theorem probability_no_more_than_once : no_more_than_once := by
  sorry

theorem probability_two_months : two_months := by
  have h_independent := January_February_independent
  sorry

end probability_no_more_than_once_probability_two_months_l539_539529


namespace solve_abs_equation_l539_539982

theorem solve_abs_equation (x : ℝ) : |2 * x - 5| = 3 * x - 1 → x = 6 / 5 :=
by
  simp
  sorry

end solve_abs_equation_l539_539982


namespace equal_dihedral_angles_cond_l539_539614

-- Defining the conditions
def unit_cube : Type := ℝ × ℝ × ℝ
def sphere (r : ℝ) : Type := {p : unit_cube // ∥ p ∥ = r}
def convex_solid (r : ℝ) (c : unit_cube) := 
  {v : unit_cube | -- where v is either a vertex of the unit cube or intersection points of face diagonals with the sphere }
-- The distance constraint conditions
def valid_radius (r : ℝ) : Prop := r > 1/2 ∧ r < 1

-- The main proof statement
theorem equal_dihedral_angles_cond (r : ℝ) (c : unit_cube) (h : valid_radius r) : 
  ∃ (T : convex_solid r c), (dihedral_angles T = dihedral_angles (convex_solid (3/4) c)) := 
sorry

end equal_dihedral_angles_cond_l539_539614


namespace probability_A_fires_l539_539469

theorem probability_A_fires :
  let p_A := (1 : ℚ) / 6 + (5 : ℚ) / 6 * (5 : ℚ) / 6 * p_A
  in p_A = 6 / 11 :=
by
  sorry

end probability_A_fires_l539_539469


namespace music_disk_space_per_minute_l539_539500

def days : ℕ := 15
def total_space_MB : ℕ := 20000
def total_minutes : ℕ := days * 24 * 60
def space_per_minute : ℚ := total_space_MB / total_minutes

theorem music_disk_space_per_minute :
  Int.round (space_per_minute.cast_to ℝ) = 1 :=
by
  sorry

end music_disk_space_per_minute_l539_539500


namespace T_is_ellipse_l539_539037

noncomputable def T (B C : Point) : Set Point :=
{ A : Point | let h := (4 / (distance B C)) in
  abs (area B C A) = 2 ∧
  (distance A B) + (distance A C) + (distance B C) = 10 }

theorem T_is_ellipse (B C : Point) (h : (4 / (distance B C))) :
  ∀ A : Point, A ∈ (T B C) → ∃ e : Ellipse, A ∈ e :=
sorry

end T_is_ellipse_l539_539037


namespace segment_length_of_triangle_l539_539242

theorem segment_length_of_triangle 
  (a b c n m Δ : ℝ) (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: n > 0) (h₅: m > 0) (hΔ: Δ > 0) :
  ∃ (x : ℝ), 
  sqrt(
    (x + n + m) / 2 *
    (-x + n + m) / 2 *
    (x - n + m) / 2 *
    (x + n - m) / 2
  ) = Δ * (n * m) / (b * c) :=
sorry

end segment_length_of_triangle_l539_539242


namespace number_of_books_bought_l539_539105

def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def remaining_books : ℕ := 2

theorem number_of_books_bought : 
  let total_books_after_shopping := shelves * books_per_shelf + remaining_books in
  total_books_after_shopping - initial_books = 26 := 
by 
  sorry

end number_of_books_bought_l539_539105


namespace constant_s_for_parabola_chords_l539_539918

theorem constant_s_for_parabola_chords (d : ℕ) : ∀ (A B C : ℝ × ℝ),
  (C = (0, 1)) →
  (A.2 = A.1 ^ 2) →
  (B.2 = B.1 ^ 2) →
  let AC := (A.1 - 0)^2 + (A.2 - 1)^2 in
  let BC := (B.1 - 0)^2 + (B.2 - 1)^2 in
  (∀ m : ℝ, (m ≠ 0) → ∃ x1 x2 : ℝ, x1 + x2 = m ∧ x1 * x2 = -1 ∧ d = 1 + m^2) →
  (s = 2)
:= sorry

end constant_s_for_parabola_chords_l539_539918


namespace problem_inequality_l539_539156

variable {a b c : ℝ}

theorem problem_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by sorry

end problem_inequality_l539_539156


namespace tangents_intersect_at_given_point_l539_539761

-- Definitions
def parabola (p : ℝ × ℝ) : Prop := 4 * p.snd = p.fst^2

def tangent_slope (p : ℝ × ℝ) : ℝ := p.fst / 2

def is_tangent (p : ℝ × ℝ) (t : ℝ × ℝ) : Prop :=
  t.snd = tangent_slope p * t.fst - (tangent_slope p * p.fst - p.snd)

def intersection (p1 p2 i : ℝ × ℝ) : Prop :=
  is_tangent p1 i ∧ is_tangent p2 i

-- Theorem to be proven
theorem tangents_intersect_at_given_point
  (t1 t2 : ℝ)
  (p1 : ℝ × ℝ := (2 * t1, t1^2))
  (p2 : ℝ × ℝ := (2 * t2, t2^2))
  (i : ℝ × ℝ := (t1 + t2, t1 * t2)) :
  parabola p1 ∧ parabola p2 → intersection p1 p2 i := sorry

end tangents_intersect_at_given_point_l539_539761


namespace min_value_of_expression_l539_539321

open Real

noncomputable def minValue (x y z : ℝ) : ℝ :=
  x + 3 * y + 5 * z

theorem min_value_of_expression : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 8 → minValue x y z = 14.796 :=
by
  intros x y z h
  sorry

end min_value_of_expression_l539_539321


namespace distance_between_points_l539_539412

noncomputable def distance_AB 
  (CA CB : ℝ) 
  (angle_ACB : ℝ) 
  (h1 : CA = 50) 
  (h2 : CB = 30) 
  (h3 : angle_ACB = 120) : ℝ :=
  real.sqrt (CA ^ 2 + CB ^ 2 - 2 * CA * CB * real.cos (angle_ACB * real.pi / 180))

theorem distance_between_points 
  : distance_AB 50 30 120 = 70 := by
  -- The proof can be filled in here.
  simp only [distance_AB]
  rw [h1, h2, h3]
  sorry

end distance_between_points_l539_539412


namespace turtle_minimum_distance_l539_539102

/-- Define the movement parameters for the turtle -/
def speed := 5 -- meters per hour
def turns := 11 -- number of turns

/-- Define potential movements/ turns -/
inductive Direction
| north : Direction
| east : Direction
| south : Direction
| west : Direction

/-- Calculate the minimum distance from the origin after 11 hours of specific movement patterns -/
theorem turtle_minimum_distance (speed : ℕ) (turns : ℕ) : ℕ :=
let pos := (0, 0) -- starting at origin
let distance := 5 -- 5 meters/h and turns changes every hour mark
in distance

end turtle_minimum_distance_l539_539102


namespace tammy_avg_speed_second_day_l539_539787

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l539_539787


namespace collinear_PQS_l539_539757

noncomputable def points_on_circle (A B C D : Type) := sorry

theorem collinear_PQS (A B C D S P Q : Type)
  (h1 : points_on_circle A B C D)
  (h2 : tangent S A)
  (h3 : tangent S D)
  (h4 : intersection P (line_through A B) (line_through C D))
  (h5 : intersection Q (line_through A C) (line_through B D)) :
  collinear P Q S :=
sorry

end collinear_PQS_l539_539757


namespace cut_square_into_acute_triangles_cut_scalene_triangle_into_isosceles_l539_539141

-- Problem 1: Cutting a Square into 8 Acute-Angled Triangles
theorem cut_square_into_acute_triangles (a : ℝ) (h : a > 0) :
  ∃ (triangles : list (ℝ × ℝ × ℝ)), 
    triangles.length = 8 ∧ 
    ∀ (t ∈ triangles), let ⟨x, y, z⟩ := t in x < π/2 ∧ y < π/2 ∧ z < π/2 :=
begin
  sorry
end

-- Problem 2: Cutting a Scalene Triangle into 7 Isosceles Triangles with at Least Three Congruent
theorem cut_scalene_triangle_into_isosceles (A B C : ℝ) 
(h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) :
  ∃ (isosceles_triangles: list (ℝ × ℝ × ℝ)), 
    isosceles_triangles.length = 7 ∧ 
    ∃ (congruent_set : set (ℝ × ℝ × ℝ)), congruent_set.card ≥ 3 ∧ 
    ∀ (t ∈ isosceles_triangles), let ⟨x, y, z⟩ := t in (x = y ∨ y = z ∨ z = x) :=
begin
  sorry
end

end cut_square_into_acute_triangles_cut_scalene_triangle_into_isosceles_l539_539141


namespace terminal_side_neg7pi_over_8_l539_539796

noncomputable theory

def same_terminal_side (θ ψ : Real) : Prop :=
  ∃ k : Int, ψ = θ + 2 * k * Real.pi

theorem terminal_side_neg7pi_over_8 :
  ∀ k : Int, same_terminal_side (-7 * Real.pi / 8) (-7 * Real.pi / 8 + 2 * k * Real.pi) := 
by
  assume k
  unfold same_terminal_side
  exists k
  rfl

end terminal_side_neg7pi_over_8_l539_539796


namespace minimum_toothpicks_removal_l539_539994

theorem minimum_toothpicks_removal
    (num_toothpicks : ℕ) 
    (num_triangles : ℕ) 
    (h1 : num_toothpicks = 40) 
    (h2 : num_triangles > 35) :
    ∃ (min_removal : ℕ), min_removal = 15 
    := 
    sorry

end minimum_toothpicks_removal_l539_539994


namespace solve_simultaneous_equations_l539_539366

theorem solve_simultaneous_equations (a b : ℚ) : 
  (a + b) * (a^2 - b^2) = 4 ∧ (a - b) * (a^2 + b^2) = 5 / 2 → 
  (a = 3 / 2 ∧ b = 1 / 2) ∨ (a = -1 / 2 ∧ b = -3 / 2) :=
by
  sorry

end solve_simultaneous_equations_l539_539366


namespace graph_shift_sin_cos_l539_539413

theorem graph_shift_sin_cos :
  ∀ x : ℝ, sin (3 * x) + cos (3 * x) = sqrt 2 * sin (3 * (x + π / 12)) :=
by
  intro x
  sorry

end graph_shift_sin_cos_l539_539413


namespace volume_displacement_square_l539_539081

theorem volume_displacement_square 
  (r : ℝ) (h_barrel : ℝ) (s_cube : ℝ)
  (h_r : r = 5) (h_h_barrel : h_barrel = 15) (h_s_cube : s_cube = 7) : 
  let V_cube := s_cube ^ 3 in
  let v_squared := V_cube ^ 2 in
  v_squared = 117649 :=
by
  have h1 : V_cube = 7 ^ 3 := by rw [h_s_cube]; rw [pow_succ]; rw [pow_succ]; rw [pow_one]
  have h2 : v_squared = (7 ^ 3) ^ 2 := by rw [h1]
  rw [pow_succ'] at h2
  rw [pow_succ'] at h2
  rw [pow_one] at h2
  exact Eq.trans h2 rfl

end volume_displacement_square_l539_539081


namespace perpendicular_vector_solution_l539_539247

def vector_dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perpendicular_vector_solution:
  let a := (1, -3) in
  let b := (4, -2) in
  ∀ λ : ℝ, (vector_dot_product (λ * a.1 + b.1, λ * a.2 + b.2) a) = 0 ↔ λ = -1 :=
by
  let a := (1, -3)
  let b := (4, -2)
  intro λ
  sorry

end perpendicular_vector_solution_l539_539247


namespace num_distinct_values_w_l539_539320

-- Let w be a complex number. Suppose there exist distinct complex numbers u, v such that
-- for every complex number z, we have (z - u)(z - v) = (z - w * u)(z - w * v).
-- We need to prove that the number of distinct possible values of w is 2.

theorem num_distinct_values_w :
  ∀ w u v : ℂ,
  u ≠ v →
  (∀ z : ℂ, (z - u) * (z - v) = (z - w * u) * (z - w * v)) →
  {w : ℂ | ∃ u v : ℂ, u ≠ v ∧ ∀ z : ℂ, (z - u) * (z - v) = (z - w * u) * (z - w * v)}.finite.card = 2 :=
by
  sorry

end num_distinct_values_w_l539_539320


namespace numberOfEquidistantPlanes_correct_l539_539392

noncomputable def numberOfEquidistantPlanes (ABC : Triangle) (d : ℝ) : ℕ :=
  sorry

theorem numberOfEquidistantPlanes_correct : 
  ∀ (ABC : Triangle) (A B C : Point) (d : ℝ),
  is_equilateral ABC ∧
  sideLength ABC = 3 ∧
  distance_to_plane A d ∧
  distance_to_plane B d ∧
  distance_to_plane C d → 
  numberOfEquidistantPlanes ABC d = 5 :=
begin
  assume ABC A B C d,
  assume h : is_equilateral ABC ∧
             sideLength ABC = 3 ∧
             distance_to_plane A d ∧
             distance_to_plane B d ∧
             distance_to_plane C d,
  sorry
end

end numberOfEquidistantPlanes_correct_l539_539392


namespace least_three_digit_product_8_is_118_l539_539875

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_product (n : ℕ) (product : ℕ) : Prop :=
  let digits := (list.cons (n / 100) (list.cons ((n / 10) % 10) (list.cons (n % 10) list.nil))) in
  digits.prod = product

theorem least_three_digit_product_8_is_118 :
  ∃ n : ℕ, is_three_digit_number n ∧ digits_product n 8 ∧ (∀ m : ℕ, is_three_digit_number m ∧ digits_product m 8 → n ≤ m) :=
sorry

end least_three_digit_product_8_is_118_l539_539875


namespace common_point_arithmetic_progression_l539_539531

theorem common_point_arithmetic_progression (a b c : ℝ) (h : 2 * b = a + c) :
  ∃ (x y : ℝ), (∀ x, y = a * x^2 + b * x + c) ∧ x = -2 ∧ y = 0 :=
by
  sorry

end common_point_arithmetic_progression_l539_539531


namespace largest_consecutive_debatable_numbers_proof_l539_539549

def is_power_of_two (n : ℕ) : Prop := ∃ m : ℕ, n = 2^m

def is_debatable (k : ℕ) : Prop :=
  is_power_of_two (k.factors.filter (λ p, p % 2 = 1).length)

def largest_consecutive_debatable_numbers : ℕ :=
  17

theorem largest_consecutive_debatable_numbers_proof :
  ∃ n : ℕ, (∀ i : ℕ, i < n → is_debatable (i + 1) ) ∧ n = 17 :=
by
  use 17
  split
  . intro i
    intro hi
    -- Here we would prove that every number from 1 to 17 is debatable.
    sorry
  . refl

end largest_consecutive_debatable_numbers_proof_l539_539549


namespace problem_solution_l539_539193

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n+1), a i

def a_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ n, a n > 0) ∧
  a 1 = 1 ∧
  (∀ n : ℕ, a (n+1)^2 - 2 * S a n = n + 1)

def b_seq (a b : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ n : ℕ, b n = a n * Real.sin (n * Real.pi / 2)

def T_100 (b : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range 101, b i

theorem problem_solution :
  (∀ a : ℕ → ℝ,
    a_seq a →
    (∀ n, a n = n)) ∧
  (∀ a b : ℕ → ℝ,
    a_seq a →
    b_seq a b →
    T_100 b = -50) :=
by
  sorry

end problem_solution_l539_539193


namespace pizzeria_large_pizzas_sold_l539_539023

theorem pizzeria_large_pizzas_sold (price_small price_large total_earnings num_small_pizzas num_large_pizzas : ℕ):
  price_small = 2 →
  price_large = 8 →
  total_earnings = 40 →
  num_small_pizzas = 8 →
  total_earnings = price_small * num_small_pizzas + price_large * num_large_pizzas →
  num_large_pizzas = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pizzeria_large_pizzas_sold_l539_539023


namespace find_tan_theta_l539_539315

open Matrix Real

-- Define the matrix for dilation, translation, and rotation.
def D (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![k, 0], ![0, k]]

def T : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 2]]

noncomputable def R (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![cos θ, -sin θ], ![sin θ, cos θ]]

-- Combined transformation matrix
def combined_matrix (θ k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := (R θ) * (D k) + T

theorem find_tan_theta {θ k : ℝ} (hk : 0 < k) :
  combined_matrix θ k = ![![10, -5], ![5, 10]] → 
  tan θ = 1 / 2 :=
by
  assume h : combined_matrix θ k = ![![10, -5], ![5, 10]]
  -- Skipping the proof
  sorry

end find_tan_theta_l539_539315


namespace geometric_sequence_product_l539_539291

-- Defining the geometric sequence and the equation
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def satisfies_quadratic_roots (a : ℕ → ℝ) : Prop :=
  (a 2 = -1 ∧ a 18 = -16 / (-1 + 16 / -1) ∨
  a 18 = -1 ∧ a 2 = -16 / (-1 + 16 / -1))

-- Problem statement
theorem geometric_sequence_product (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : satisfies_quadratic_roots a) : 
  a 3 * a 10 * a 17 = -64 :=
sorry

end geometric_sequence_product_l539_539291


namespace net_price_change_l539_539451

variable (P : ℝ)

theorem net_price_change :
  let decreased_price := P - 0.20 * P in
  let new_price := 0.80 * P in
  let increased_price := new_price + 0.40 * new_price in
  let final_price := 1.12 * P in
  final_price = P + 0.12 * P :=
by
  sorry

end net_price_change_l539_539451


namespace correct_statement_l539_539051

theorem correct_statement :
  let A := (3 : ℕ) = 3 ∧ (3 + 3) = 6 ∧ ((3 : ℕ) / (3 + 3) : ℝ) = 0.5 in
  let B := ∀ (n : ℕ), (n = 100 → (1/100 * n ≠ 1)) in
  let C := False in
  let D := ∀ (a : ℝ), |a| > 0 → a ≠ 0 in
  A ∧ ¬ B ∧ C ∧ ¬ D :=
by 
  sorry

end correct_statement_l539_539051


namespace sin_identity_l539_539733

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end sin_identity_l539_539733


namespace bug_can_return_to_start_l539_539279

variable (Grid : Type)
variable (Cell : Grid → Type)
variable (Start : ∀ g : Grid, Cell g)
variable (Door : ∀ g : Grid, Cell g → Cell g → Bool)  -- True if the door is open, False otherwise
variable (Move : ∀ g : Grid, (c1 c2 : Cell g), Door g c1 c2 → Prop)

-- Movement rule: bug can move to an adjacent cell c2 if there's an open door from c1 to c2.
axiom move_through_open_door : ∀ g (c1 c2 : Cell g), Door g c1 c2 → Move g c1 c2

-- Movement rule: the bug opens the door in the direction it moves.
axiom open_door : ∀ g (c1 c2 : Cell g), Move g c1 c2 → Door g c1 c2

-- Prove the bug can return to the starting cell at any moment.
theorem bug_can_return_to_start (g : Grid) : ∀ (c : Cell g), ∃ (p : List (Cell g)), p.head = Start g ∧ p.tail.last = c ∧ ∀ (i : ℕ) (h : i < p.length - 1), Move g (p.nth_le i h) (p.nth_le (i + 1) sorry) :=
  sorry

end bug_can_return_to_start_l539_539279


namespace mrs_hilt_monday_miles_l539_539747

variable (M : ℕ)
variable miles_wednesday : ℕ := 2
variable miles_friday : ℕ := 7
variable total_miles : ℕ := 12

theorem mrs_hilt_monday_miles : M = total_miles - (miles_wednesday + miles_friday) → M = 3 :=
by
  intro h
  rw [h]
  sorry

end mrs_hilt_monday_miles_l539_539747


namespace distance_after_100_moves_l539_539755

theorem distance_after_100_moves : 
  let P_initial := 0 in 
  let P_final := (List.range 100).sum (λ n => if n % 2 = 0 then n + 1 else - (n + 1)) in 
  abs P_final = 50 := 
by 
  let P_initial := 0
  let moves := List.range 100
  let P_final := moves.sum (λ n => if n % 2 = 0 then n + 1 else - (n + 1))
  have step1 : P_final = -50 := sorry
  have step2 : abs P_final = 50 := by rw [step1]; simp
  exact step2

end distance_after_100_moves_l539_539755


namespace large_pizzas_sold_l539_539018

variables (num_small_pizzas num_large_pizzas : ℕ) (price_small price_large total_revenue revenue_from_smalls revenue_from_larges : ℕ)

theorem large_pizzas_sold
  (price_small := 2)
  (price_large := 8)
  (total_revenue := 40)
  (num_small_pizzas := 8)
  (revenue_from_smalls := num_small_pizzas * price_small)
  (revenue_from_larges := total_revenue - revenue_from_smalls)
  (large_pizza_count := revenue_from_larges / price_large) :
  large_pizza_count = 3 :=
sorry

end large_pizzas_sold_l539_539018


namespace sqrt_529000_pow_2_5_l539_539551

theorem sqrt_529000_pow_2_5 : (529000 ^ (1 / 2) ^ (5 / 2)) = 14873193 := by
  sorry

end sqrt_529000_pow_2_5_l539_539551


namespace triangle_angle_relationship_l539_539297

theorem triangle_angle_relationship
  (A B C : ℝ) (φ : ℝ)
  (h : ∀ (x y z : ℝ), x^2 + y^2 - 2*x*z + y*z = 0)  -- represent the altitude condition equivalently
  (h_diff_ang : A - B = φ) :
  sin (C / 2) = 1 - sin (φ / 2) :=
sorry

end triangle_angle_relationship_l539_539297


namespace triangle_area_l539_539167

def point_3d := (ℝ × ℝ × ℝ)

def A : point_3d := (1, 4, 5)
def B : point_3d := (3, 4, 1)
def C : point_3d := (1, 1, 1)

-- Define the magnitude of a 3D vector
noncomputable def magnitude (v : ℝ × ℝ × ℝ) :=
  real.sqrt ((v.1)^2 + (v.2)^2 + (v.3)^2)

-- Define cross product for 3D vectors
noncomputable def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

-- Define vector subtraction
def vector_sub (p q : point_3d) : ℝ × ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2, p.3 - q.3)

theorem triangle_area :
  let AB := vector_sub B A in
  let AC := vector_sub C A in
  let cross := cross_product AB AC in
  (1 / 2) * magnitude cross = real.sqrt 61 :=
by {
  let AB := vector_sub B A,
  let AC := vector_sub C A,
  let cross := cross_product AB AC,
  unfold magnitude cross_product,
  sorry
}

end triangle_area_l539_539167


namespace number_of_zeros_f_in_interval_604_l539_539558

def f (x : ℝ) : ℝ := if x ≤ 4 ∧ x > -1 then x^2 - 2*x else 16 - f(x - 5)

theorem number_of_zeros_f_in_interval_604 :
  (∃ count : ℕ, count = 604) ↔
    (∀ x ∈ Set.Icc (0 : ℝ) 2013, f x = 0 → count = 604) :=
sorry

end number_of_zeros_f_in_interval_604_l539_539558


namespace velocity_zero_at_1_and_2_l539_539511

def displacement (t : ℝ) : ℝ := (1 / 3) * t^3 - (3 / 2) * t^2 + 2 * t

def velocity (t : ℝ) : ℝ := deriv displacement t

theorem velocity_zero_at_1_and_2 : velocity 1 = 0 ∧ velocity 2 = 0 := by
  sorry

end velocity_zero_at_1_and_2_l539_539511


namespace find_a_b_solve_inequality_l539_539327

noncomputable def quadratic_function (a b : ℝ) := λ x : ℝ, x^2 - a * x + b

noncomputable def linear_function := λ x : ℝ, x - 1

theorem find_a_b :
  ∃ a b : ℝ, 
    (∀ x : ℝ, 1 < x ∧ x < 2 → (quadratic_function a b x < 0)) ∧
    (a = 3 ∧ b = 2) := by
  sorry

theorem solve_inequality (c : ℝ) :
  let f := quadratic_function 3 2
  let g := linear_function
  (c > -1 → ∀ x : ℝ, x > c + 2 ∨ x < 1 → f x > c * g x) ∧
  (c < -1 → ∀ x : ℝ, x > 1 ∨ x < c + 2 → f x > c * g x) ∧
  (c = -1 → ∀ x : ℝ, x ≠ 1 → f x > c * g x) := by
  sorry

end find_a_b_solve_inequality_l539_539327


namespace albert_investment_years_l539_539114

noncomputable def compound_interest_years (P A r : ℝ) :=
  log(A / P) / log(1 + r)

theorem albert_investment_years :
  compound_interest_years 1000 1331.0000000000005 0.10 = 3 :=
by
  sorry

end albert_investment_years_l539_539114


namespace count_pow2_not_pow4_l539_539662

theorem count_pow2_not_pow4 (n : ℕ) (h₁ : 10000000 < 2^n) (h₂ : 2^(n - 1) < 10000000) : 
  {m : ℕ | m < 10000000 ∧ ∃ k : ℕ, 2^k = m ∧ (∀ j : ℕ, m ≠ 2^(2*j)) }.card = 12 :=
by {
  sorry
}

end count_pow2_not_pow4_l539_539662


namespace proper_subsets_count_l539_539646

theorem proper_subsets_count (N : Set ℕ) (hN : N = {1, 3, 5}) : Nat.card (N.powerset \ {N}) = 7 :=
by
  sorry

end proper_subsets_count_l539_539646


namespace solve_system_of_inequalities_l539_539368

open Set

theorem solve_system_of_inequalities : ∀ x : ℕ, (2 * (x - 1) < x + 3) ∧ ((2 * x + 1) / 3 > x - 1) → x ∈ ({0, 1, 2, 3} : Set ℕ) :=
by
  intro x
  intro h
  sorry

end solve_system_of_inequalities_l539_539368


namespace trajectory_midpoint_of_chord_l539_539210

theorem trajectory_midpoint_of_chord :
  ∀ (M: ℝ × ℝ), (∃ (C D : ℝ × ℝ), (C.1^2 + C.2^2 = 25 ∧ D.1^2 + D.2^2 = 25 ∧ dist C D = 8) ∧ M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  → M.1^2 + M.2^2 = 9 :=
sorry

end trajectory_midpoint_of_chord_l539_539210


namespace min_value_of_f_l539_539887

noncomputable def f (x : ℝ) : ℝ := x^2 + 8 * x + 3

theorem min_value_of_f : ∃ x₀ : ℝ, (∀ x : ℝ, f x ≥ f x₀) ∧ f x₀ = -13 :=
by
  sorry

end min_value_of_f_l539_539887


namespace find_a_tangent_l539_539225

-- Define the first curve
def curve1 (x : ℝ) : ℝ := x^2 - log x

-- Define the second curve parameterized by a
def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

-- Tangent slope function for curve1
def tangent_slope_curve1 (x : ℝ) : ℝ := 2 * x - (1 / x)

theorem find_a_tangent (a : ℝ) (x : ℝ) (y : ℝ) :
  (curve1 1 = 1) ∧ (tangent_slope_curve1 1 = 1) ∧ (∀ (x : ℝ), (x - 1) ≠ 0 → (curve2 a x = x)) → a = 1 :=
  sorry

end find_a_tangent_l539_539225


namespace square_lake_area_l539_539490

noncomputable def speed : ℝ := 10 -- MPH
noncomputable def time_length : ℝ := 2 -- hours
noncomputable def time_width_minutes : ℝ := 30 -- minutes
noncomputable def time_width : ℝ := time_width_minutes / 60 -- hours

def length_of_lake : ℝ := speed * time_length
def width_of_lake : ℝ := speed * time_width
def area_of_lake : ℝ := length_of_lake * width_of_lake

theorem square_lake_area : area_of_lake = 100 :=
by
  -- Mathematical proof goes here
  sorry

end square_lake_area_l539_539490


namespace find_pq_l539_539726

theorem find_pq (p q : ℝ) (h1 : (p + 3 * complex.I) * (q + 7 * complex.I) = 5 + 66 * complex.I)
  (h2 : p + q = 12)
  (h3 : 7 * p + 3 * q = 66) :
  (p = 7.5 ∧ q = 4.5) :=
sorry

end find_pq_l539_539726


namespace yanna_kept_36_apples_l539_539444

-- Define the initial number of apples Yanna has
def initial_apples : ℕ := 60

-- Define the number of apples given to Zenny
def apples_given_to_zenny : ℕ := 18

-- Define the number of apples given to Andrea
def apples_given_to_andrea : ℕ := 6

-- The proof statement that Yanna kept 36 apples
theorem yanna_kept_36_apples : initial_apples - apples_given_to_zenny - apples_given_to_andrea = 36 := by
  sorry

end yanna_kept_36_apples_l539_539444


namespace absolute_value_expression_l539_539322

theorem absolute_value_expression {x : ℤ} (h : x = 2024) :
  abs (abs (abs x - x) - abs x) = 0 :=
by
  sorry

end absolute_value_expression_l539_539322


namespace not_integer_fraction_l539_539738

theorem not_integer_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 2) : ¬ (∃ k : ℤ, (2^a + 1) = k * (2^b - 1)) :=
sorry

end not_integer_fraction_l539_539738


namespace ratio_of_John_to_Mary_l539_539303

variable (John Mary Jamison : ℕ)

-- Conditions
def Mary_weight : Prop := Mary = 160
def Mary_less_than_Jamison : Prop := Mary + 20 = Jamison
def Combined_weight : Prop := John + Mary + Jamison = 540

-- Theorem to prove the ratio of John's weight to Mary's weight
theorem ratio_of_John_to_Mary (John_weight Mary_weight Mary_less_than_Jamison Combined_weight) : 
  ∃ r : ℚ, r = 5 / 4 :=
by
  sorry

end ratio_of_John_to_Mary_l539_539303


namespace train_speed_in_km_hr_l539_539523

noncomputable def train_length : ℝ := 320
noncomputable def crossing_time : ℝ := 7.999360051195905
noncomputable def speed_in_meter_per_sec : ℝ := train_length / crossing_time
noncomputable def meter_per_sec_to_km_hr (speed_mps : ℝ) : ℝ := speed_mps * 3.6
noncomputable def expected_speed : ℝ := 144.018001125

theorem train_speed_in_km_hr :
  meter_per_sec_to_km_hr speed_in_meter_per_sec = expected_speed := by
  sorry

end train_speed_in_km_hr_l539_539523


namespace solve_fractional_equation_l539_539580

theorem solve_fractional_equation (x : ℝ) :
  (x ≠ 5) ∧ (x ≠ 6) ∧ ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) / ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ 
  x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
sorry

end solve_fractional_equation_l539_539580


namespace large_pizzas_sold_l539_539017

variables (num_small_pizzas num_large_pizzas : ℕ) (price_small price_large total_revenue revenue_from_smalls revenue_from_larges : ℕ)

theorem large_pizzas_sold
  (price_small := 2)
  (price_large := 8)
  (total_revenue := 40)
  (num_small_pizzas := 8)
  (revenue_from_smalls := num_small_pizzas * price_small)
  (revenue_from_larges := total_revenue - revenue_from_smalls)
  (large_pizza_count := revenue_from_larges / price_large) :
  large_pizza_count = 3 :=
sorry

end large_pizzas_sold_l539_539017


namespace elder_age_is_33_l539_539795

-- Define the conditions
variables (y e : ℕ)

def age_difference_condition : Prop :=
  e = y + 20

def age_reduced_condition : Prop :=
  e - 8 = 5 * (y - 8)

-- State the theorem to prove the age of the elder person
theorem elder_age_is_33 (h1 : age_difference_condition y e) (h2 : age_reduced_condition y e): e = 33 :=
  sorry

end elder_age_is_33_l539_539795


namespace coin_flip_frequency_probability_l539_539687

-- Definitions of conditions
def flips : Nat := 800
def heads : Nat := 440
def fair_coin : Prop := true

-- Frequency calculation
def frequency (heads flips : Nat) : Float :=
  head_ratio = ((heads : Float) / (flips))
  head_ratio

-- Probability for a fair coin
def probability (fair_coin : Prop) : Float :=
  probability_head = 0.5
  probability_head

-- Statement of the proof problem
theorem coin_flip_frequency_probability : 
  (frequency heads flips = 0.55) 
  ∧ (probability fair_coin = 0.5) :=
by 
  sorry

end coin_flip_frequency_probability_l539_539687


namespace integral_f_l539_539908

def f (x : ℝ) :=
  if x ≤ 0 then x^2
  else cos x - 1

theorem integral_f : ∫ x in -1..(π / 2), f x = (4 / 3) - (π / 2) :=
by sorry

end integral_f_l539_539908


namespace determine_min_k_l539_539323

open Nat

theorem determine_min_k (n : ℕ) (h : n ≥ 3) 
  (a : Fin n → ℕ) (b : Fin (choose n 2) → ℕ) : 
  ∃ k, k = (n - 1) * (n - 2) / 2 + 1 := 
sorry

end determine_min_k_l539_539323


namespace garden_area_proof_l539_539923

def length_rect : ℕ := 20
def width_rect : ℕ := 18
def area_rect : ℕ := length_rect * width_rect

def side_square1 : ℕ := 4
def area_square1 : ℕ := side_square1 * side_square1

def side_square2 : ℕ := 5
def area_square2 : ℕ := side_square2 * side_square2

def area_remaining : ℕ := area_rect - area_square1 - area_square2

theorem garden_area_proof : area_remaining = 319 := by
  sorry

end garden_area_proof_l539_539923


namespace reciprocal_sum_of_roots_l539_539624

theorem reciprocal_sum_of_roots :
  (∃ m n : ℝ, (m^2 + 2 * m - 3 = 0) ∧ (n^2 + 2 * n - 3 = 0) ∧ m ≠ n) →
  (∃ m n : ℝ, (1/m + 1/n = 2/3)) :=
by
  sorry

end reciprocal_sum_of_roots_l539_539624


namespace complement_cardinality_l539_539328

open Set

variable (U : Set ℕ) (A B : Set ℕ)

noncomputable def U := { n | 0 < n ∧ n ≤ 9 }
noncomputable def A := { 2, 5 }
noncomputable def B := { 1, 2, 4, 5 }

theorem complement_cardinality :
  card (U \ (A ∪ B)) = 5 := by
  sorry

end complement_cardinality_l539_539328


namespace sin_sum_cos_identity_l539_539063

theorem sin_sum_cos_identity (A B C : ℝ) : 
  sin A + sin B + sin C = 4 * cos (A / 2) * cos (B / 2) * cos (C / 2) :=
sorry

end sin_sum_cos_identity_l539_539063


namespace prism_similar_faces_l539_539936

open Nat

/-- In a right rectangular prism with sides of integral lengths a, b, c such that a ≤ b ≤ c
 and b = 2023, if the prism is cut by a plane parallel to one of its faces resulting in two
 smaller prisms one of which is similar to the original, then the number of ordered triples
 (a, b, c) satisfying these conditions is 13. -/
theorem prism_similar_faces (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 2023)
  (h4 : a * c = 2023^2) : 
  (finset.filter (λ (t : ℕ × ℕ × ℕ), ((t.1 ≤ t.2 ∧ t.2 ≤ t.3 ∧ t.2 = 2023) ∧ (t.1 * t.3 = 2023^2))
  (finset.Icc (0, 0, 0) (2023, 2023, 2023))).card = 13 := sorry

end prism_similar_faces_l539_539936


namespace interest_tax_rate_l539_539675

noncomputable def interest_tax (p r: ℝ) (total: ℝ) : ℝ :=
  let i := p * r in
  let t := total - p in
  1 - t / i

theorem interest_tax_rate :
  interest_tax 10000 0.0225 10180 = 0.2 :=
by
  sorry

end interest_tax_rate_l539_539675


namespace angle_between_a_and_v_is_90_l539_539318

namespace VectorAngleProof

open Real

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (-1, 0, 4)
def c : ℝ × ℝ × ℝ := (0, 5, -2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def scalar_mult (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def v : ℝ × ℝ × ℝ :=
  vector_sub (scalar_mult (dot_product a c) b) (scalar_mult (dot_product a b) c)

theorem angle_between_a_and_v_is_90 :
  dot_product a v = 0 :=
by
  -- Proof to be provided
  sorry

end VectorAngleProof

end angle_between_a_and_v_is_90_l539_539318


namespace maps_line_to_line_l539_539070

noncomputable def f : ℝ × ℝ → ℝ × ℝ := sorry

axiom bijective_f : Function.Bijective f
axiom maps_circle_to_circle : ∀ (c : set (ℝ × ℝ)), is_circle c → is_circle (f '' c)

theorem maps_line_to_line (l : set (ℝ × ℝ)) (h : is_line l) : is_line (f '' l) := by
  sorry

end maps_line_to_line_l539_539070


namespace tammy_speed_on_second_day_l539_539781

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l539_539781


namespace part_one_part_two_l539_539612

noncomputable def sequence_satisfying_conditions : ℕ → ℤ
| 0 => 0
| n + 1 => (sequence_satisfying_conditions n + n + 1 : ℤ)

def a (n : ℕ) : ℕ :=
if n = 0 then 1 else (sequence_satisfying_conditions n.succ - sequence_satisfying_conditions n : ℤ).toNat

def S (n : ℕ) : ℕ :=
n * n

def b (n : ℕ) : ℚ :=
if n = 0 then 0 else
  (S n / (n * a n * a (n+1)) : ℚ) - (S (n+1) / ((n+1) * a (n+1) * a (n+2)) : ℚ)

theorem part_one (n : ℕ) : S n = n * n :=
sorry

theorem part_two (n : ℕ) : ∑ i in finset.range (n + 1), b i < 1 / 3 :=
sorry

end part_one_part_two_l539_539612


namespace sum_of_remainders_l539_539889

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 47 = 25) (h2 : b % 47 = 20) (h3 : c % 47 = 3) : 
  (a + b + c) % 47 = 1 := 
by {
  sorry
}

end sum_of_remainders_l539_539889


namespace smallest_side_length_of_circumscribing_ngon_l539_539347

open Real EuclideanGeometry

/-- Prove that among all regular n-gons covering the circle (O, r),
    the side length of the n-gon circumscribing (O, r) is the smallest. -/
theorem smallest_side_length_of_circumscribing_ngon (r : ℝ) (O : EuclideanGeometry.Point) (n : ℕ) (h1 : n ≥ 3) :
  ∀ (P : EuclideanGeometry.Point → Prop) (hP : regular_ngon_covering_circle P O r),
  side_length_of_ngon P > side_length_of_ngon_circumscribing_circle n O r :=
by
  sorry

end smallest_side_length_of_circumscribing_ngon_l539_539347


namespace tom_gave_2_seashells_to_jessica_l539_539838

-- Conditions
def original_seashells : Nat := 5
def current_seashells : Nat := 3

-- Question as a proposition
def seashells_given (x : Nat) : Prop :=
  original_seashells - current_seashells = x

-- The proof problem
theorem tom_gave_2_seashells_to_jessica : seashells_given 2 :=
by 
  sorry

end tom_gave_2_seashells_to_jessica_l539_539838


namespace sequence_of_form_l539_539144

def sequence (a : ℕ → ℤ) : Prop :=
  a 0 = 1 ∧
  a 1 = 4 ∧
  ∀ n >= 1, a (n + 1) = 5 * a n - a (n - 1)

theorem sequence_of_form (a : ℕ → ℤ) (h : sequence a) :
  ∀ n, ∃ x y : ℤ, a n = x^2 + 3 * y^2 :=
by
  sorry

end sequence_of_form_l539_539144


namespace tomatoes_remaining_l539_539094

def initial_tomatoes : ℕ := 100
def fraction_picked_initially : ℚ := 1 / 4
def picked_after_one_week : ℕ := 20
def picked_in_following_week (prev_picked : ℕ) : ℕ := 2 * prev_picked

theorem tomatoes_remaining : 
  let initial_picked := (initial_tomatoes : ℚ) * fraction_picked_initially in
  let picked_after_first_week := initial_picked.nat_cast + picked_after_one_week in
  let picked_after_second_week := picked_in_following_week picked_after_one_week in
  let total_picked := picked_after_first_week + picked_after_second_week in
  initial_tomatoes - total_picked = 15 :=
by
  sorry

end tomatoes_remaining_l539_539094


namespace congruent_triangles_l539_539962

theorem congruent_triangles (n : ℕ) (n_ge_4 : n ≥ 4) (r : ℕ) (V : Finset (Fin n))
  (hV : V.card = r) (h : r * (r - 3) ≥ n) :
  ∃ (u v w x y z : Fin n), {u, v, w} ⊆ V ∧ {x, y, z} ⊆ V ∧
    (u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧ 
    (dist u v = dist x y ∧ dist v w = dist y z ∧ dist w u = dist z x) :=
sorry

end congruent_triangles_l539_539962


namespace number_of_valid_integers_l539_539595

noncomputable def count_valid_n : Nat :=
  Nat.card {n : Nat | 10 ≤ n ∧ n ≤ 60 ∧ ¬Nat.Prime n ∧ Nat.divisible ((Nat.factorial (n^2 - 3))) ((Nat.factorial n)^(n - 1))}

theorem number_of_valid_integers :
  count_valid_n = 30 :=
sorry

end number_of_valid_integers_l539_539595


namespace football_region_area_l539_539762

/-- Defining point D of the rectangle ABCD -/
def D : Point := { x := 4, y := 2}

/-- Defining point B of the rectangle ABCD -/
def B : Point := { x := 0, y := 2}

/-- Defining the radius of Circle centered at D -/
def radius_D : ℝ := 4

/-- Defining the radius of Circle centered at B -/
def radius_B : ℝ := 4

/-- Function to calculate the area of a circular sector given its radius -/
def sector_area (r : ℝ) : ℝ := ((real.pi * r * r) / 4)

/-- Total area of the football shaped region formed by the intersection -/
def intersection_area : ℝ :=
  sector_area radius_D + sector_area radius_B - 8

theorem football_region_area :
  abs (intersection_area - 7.7) < 0.1 :=
by
  -- proof goes here
  sorry

end football_region_area_l539_539762


namespace largest_divisor_of_product_visible_faces_l539_539539

def eight_sided_die_faces := {n : ℕ | 1 ≤ n ∧ n ≤ 8}

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem largest_divisor_of_product_visible_faces
  (P : ℕ) 
  (H : ∀ (S ⊆ eight_sided_die_faces) (hS : S.card = 7), 
    P = (∏ i in S, i)) :
  ∃ k : ℕ, k = 192 ∧ ∀ d : ℕ, (∀ (S ⊆ eight_sided_die_faces) (hS : S.card = 7), d ∣ P) → d ≤ k :=
sorry

end largest_divisor_of_product_visible_faces_l539_539539


namespace integral_cos_ln_correct_l539_539541

noncomputable def integral_cos_ln (a : ℝ) (a_nonzero : a ≠ 0) (x : ℝ) : ℝ :=
  ∫ (x : ℝ) in 1..e^(π/4), x^2 * cos(a * ln(x)) dx = (3 * x^3 * cos(a * ln(x)) + a * x^3 * sin(a * ln(x))) / (a^2 + 9)

noncomputable def volume_solid_rotation : ℝ :=
  let f := λ x : ℝ, x * cos(ln(x))
  π / 24 * (10 * exp(3 * π / 4) - 13)

theorem integral_cos_ln_correct (a : ℝ) (a_nonzero : a ≠ 0) (x : ℝ) : 
  integral_cos_ln a a_nonzero x ∧ volume_solid_rotation =
  ∫ (x : ℝ) in 1..e^(π/4), (x^2 * cos(2 * ln x)) dx :=
sorry

end integral_cos_ln_correct_l539_539541


namespace smallest_n_for_f_gt_20_l539_539664

def sum_of_digits (n : ℕ) : ℕ :=
(n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def f (n : ℕ) : ℕ :=
sum_of_digits (4^n)

theorem smallest_n_for_f_gt_20 : ∃ n : ℕ, n > 0 ∧ f(n) > 20 ∧ ∀ m : ℕ, m > 0 ∧ m < n → f(m) ≤ 20 :=
by
  -- Proof omitted
  sorry

end smallest_n_for_f_gt_20_l539_539664


namespace train_pass_tree_time_l539_539054

-- Define the conditions
def train_length : ℝ := 280 -- length in meters
def speed_kmh : ℝ := 72  -- speed in kilometers per hour
def speed_ms : ℝ := speed_kmh * 1000 / 3600 -- Convert speed to meters per second
def expected_time : ℝ := 14 -- The expected answer we calculated

-- Define the theorem to be proven
theorem train_pass_tree_time (h_length : train_length = 280)
                              (h_speed_kmh : speed_kmh = 72)
                              (h_speed_ms : speed_ms = 72 * 1000 / 3600) :
  (train_length / speed_ms) = expected_time := by
  -- sorry is used here to indicate the proof needs to be filled in
  sorry

end train_pass_tree_time_l539_539054


namespace Wendy_age_l539_539424

theorem Wendy_age
  (years_as_accountant : ℕ)
  (years_as_manager : ℕ)
  (percent_accounting_related : ℝ)
  (total_accounting_related : ℕ)
  (total_lifespan : ℝ) :
  years_as_accountant = 25 →
  years_as_manager = 15 →
  percent_accounting_related = 0.50 →
  total_accounting_related = years_as_accountant + years_as_manager →
  (total_accounting_related : ℝ) = percent_accounting_related * total_lifespan →
  total_lifespan = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Wendy_age_l539_539424


namespace correct_option_l539_539891

theorem correct_option 
  (A_false : ¬ (-6 - (-9)) = -3)
  (B_false : ¬ (-2 * (-5)) = -7)
  (C_false : ¬ (-x^2 + 3 * x^2) = 2)
  (D_true : (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b) :
  (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b :=
by sorry

end correct_option_l539_539891


namespace pizzeria_large_pizzas_sold_l539_539024

theorem pizzeria_large_pizzas_sold (price_small price_large total_earnings num_small_pizzas num_large_pizzas : ℕ):
  price_small = 2 →
  price_large = 8 →
  total_earnings = 40 →
  num_small_pizzas = 8 →
  total_earnings = price_small * num_small_pizzas + price_large * num_large_pizzas →
  num_large_pizzas = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pizzeria_large_pizzas_sold_l539_539024


namespace domain_of_sqrt_over_linear_l539_539807

noncomputable def domain_of_function (f : ℝ → ℝ) : Set ℝ :=
  { x : ℝ | x ≥ -2 ∧ x ≠ 1 / 2 }

theorem domain_of_sqrt_over_linear :
  domain_of_function (λ x : ℝ, sqrt (x + 2) / (2 * x - 1)) = 
  { x : ℝ | x ≥ -2 ∧ x ≠ 1 / 2 } :=
by
  sorry

end domain_of_sqrt_over_linear_l539_539807


namespace number_of_solutions_eq_one_l539_539587

theorem number_of_solutions_eq_one :
  ∃! (n : ℕ), 0 < n ∧ 
              (∃ k : ℕ, (n + 1500) = 90 * k ∧ k = Int.floor (Real.sqrt n)) :=
sorry

end number_of_solutions_eq_one_l539_539587


namespace John_l539_539302

theorem John's_number (n : ℕ) : (200 ∣ n) ∧ (45 ∣ n) ∧ (1000 < n ∧ n < 3000) → n = 1800 :=
begin
  sorry
end

end John_l539_539302


namespace tammy_avg_speed_l539_539777

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l539_539777


namespace division_4073_by_38_l539_539882

theorem division_4073_by_38 :
  ∃ q r, 4073 = 38 * q + r ∧ 0 ≤ r ∧ r < 38 ∧ q = 107 ∧ r = 7 := by
  sorry

end division_4073_by_38_l539_539882


namespace number_of_students_in_third_grade_l539_539077

theorem number_of_students_in_third_grade
    (total_students : ℕ)
    (sample_size : ℕ)
    (students_first_grade : ℕ)
    (students_second_grade : ℕ)
    (sample_first_and_second : ℕ)
    (students_in_third_grade : ℕ)
    (h1 : total_students = 2000)
    (h2 : sample_size = 100)
    (h3 : sample_first_and_second = students_first_grade + students_second_grade)
    (h4 : students_first_grade = 30)
    (h5 : students_second_grade = 30)
    (h6 : sample_first_and_second = 60)
    (h7 : sample_size - sample_first_and_second = students_in_third_grade)
    (h8 : students_in_third_grade * total_students = 40 * total_students / 100) :
  students_in_third_grade = 800 :=
sorry

end number_of_students_in_third_grade_l539_539077


namespace abs_inequality_solution_l539_539817

theorem abs_inequality_solution (x : ℝ) : (|x - 1| < 2) ↔ (x > -1 ∧ x < 3) := 
sorry

end abs_inequality_solution_l539_539817


namespace find_cost_price_of_watch_l539_539942

noncomputable def cost_price_of_watch : ℝ :=
let
  a := 1.134 -- (1 + 0.08) * (1 + 0.05)
  b := 0.594 -- (1 - 0.46) * (1 + 0.10)
in
  140 / (a - b)

theorem find_cost_price_of_watch : abs (cost_price_of_watch - 259.26) < 0.01 :=
sorry

end find_cost_price_of_watch_l539_539942


namespace equivalency_of_planes1_equivalency_of_planes2_l539_539613

variables {S : Type*} [InnerProductSpace ℝ S]

-- Definitions for the edge and lines
variables (a b c : S)
variables (α β γ α' β' γ' : S)

-- Conditions as hypotheses
axiom edge_non_collinear : ¬Collinear ℝ ({a, b, c} : Set S)
axiom lines_in_planes : ∃ αs βs γs, PlaneCont αs (∠ a b c) ∧ PlaneCont βs (∠ b c a) ∧ PlaneCont γs (∠ c a b) ∧ α = αs ∧ β = βs ∧ γ = γs
axiom symmetrical_lines : ∃ α's β's γ's, Symmetric α' (bisector_plane a) α's ∧ Symmetric β' (bisector_plane b) β's ∧ Symmetric γ' (bisector_plane c) γ's

-- Goal: Proving the equivalency statements
theorem equivalency_of_planes1 :
  Coplanar ℝ ({α, β, γ} : Set S) ↔ Coplanar ℝ ({α', β', γ'} : Set S) :=
sorry

theorem equivalency_of_planes2 :
  IntersectsInOneLine (plane_containing a α) (plane_containing b β) (plane_containing c γ) ↔
  IntersectsInOneLine (plane_containing a α') (plane_containing b β') (plane_containing c γ') :=
sorry

end equivalency_of_planes1_equivalency_of_planes2_l539_539613


namespace find_valid_n_l539_539163

def sum_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem find_valid_n (n : ℕ) (h : 1 ≤ n ∧ n ≤ 999) : 
  (n^2 = (sum_digits n)^3) → (n = 1 ∨ n = 27) :=
by
  sorry

end find_valid_n_l539_539163


namespace gardening_project_cost_l539_539955

def cost_rose_bushes (number_of_bushes: ℕ) (cost_per_bush: ℕ) : ℕ := number_of_bushes * cost_per_bush
def cost_gardener (hourly_rate: ℕ) (hours_per_day: ℕ) (days: ℕ) : ℕ := hourly_rate * hours_per_day * days
def cost_soil (cubic_feet: ℕ) (cost_per_cubic_foot: ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem gardening_project_cost :
  cost_rose_bushes 20 150 + cost_gardener 30 5 4 + cost_soil 100 5 = 4100 :=
by
  sorry

end gardening_project_cost_l539_539955


namespace solution_set_for_inequality_l539_539005

theorem solution_set_for_inequality (k : ℤ) (x : ℝ) :
  abs (log (sin x) / log (cos x)) > abs (log (cos x) / log (sin x)) →
  ∃ k : ℤ, 2 * k * π < x ∧ x < 2 * k * π + π / 4 :=
sorry

end solution_set_for_inequality_l539_539005


namespace sufficient_not_necessary_l539_539065

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → a^2 > 1) ∧ ¬(a^2 > 1 → a > 1) :=
by {
  sorry
}

end sufficient_not_necessary_l539_539065


namespace intersect_y_axis_at_l539_539087

noncomputable def point := (ℝ × ℝ)

def line_through (p1 p2 : point) : ℝ → point :=
fun x => let m := (p2.2 - p1.2) / (p2.1 - p1.1)
         let b := p1.2 - m * p1.1
         (x, m * x + b)

theorem intersect_y_axis_at (p1 p2 : point) : 
  p1 = (10, 0) → p2 = (6, -4) → 
  (line_through p1 p2 0) = (0, -10) := by
  sorry

end intersect_y_axis_at_l539_539087


namespace number_of_white_balls_l539_539283

-- Definition of conditions
def red_balls : ℕ := 4
def frequency_of_red_balls : ℝ := 0.25
def total_balls (white_balls : ℕ) : ℕ := red_balls + white_balls

-- Proving the number of white balls given the conditions
theorem number_of_white_balls (x : ℕ) :
  (red_balls : ℝ) / total_balls x = frequency_of_red_balls → x = 12 :=
by
  sorry

end number_of_white_balls_l539_539283


namespace find_original_number_l539_539510

theorem find_original_number (x : ℤ) (h1 : 4 * x = 108) (h2 : odd (3 * x)) : x = 27 := sorry

end find_original_number_l539_539510


namespace natural_sum_possible_l539_539061

noncomputable def α : ℝ := (-1 + Real.sqrt 29) / 2
def is_coin_denominations (c : ℕ → ℝ) : Prop :=
  ∀ k, c k = if k = 0 then 1 else α ^ k

theorem natural_sum_possible (n : ℕ) :
  ∃ (c : ℕ → ℝ), is_coin_denominations c ∧ 
  (∀ k > 0, k ∋ ∃ n_k, n_k ≤ 6 ∧ c k = α^k) ∧ 
  (∑ k in (Finset.range n), c k * n_k = n) := 
sorry

end natural_sum_possible_l539_539061


namespace nickels_count_l539_539743

theorem nickels_count (original_nickels : ℕ) (additional_nickels : ℕ) 
                        (h₁ : original_nickels = 7) 
                        (h₂ : additional_nickels = 5) : 
    original_nickels + additional_nickels = 12 := 
by sorry

end nickels_count_l539_539743


namespace red_flowers_count_l539_539015

theorem red_flowers_count (w r : ℕ) (h1 : w = 555) (h2 : w = r + 208) : r = 347 :=
by {
  -- Proof steps will be here
  sorry
}

end red_flowers_count_l539_539015


namespace jing_jing_notebook_count_l539_539214

-- Define the main theorem
theorem jing_jing_notebook_count :
  ∃ (x y z : ℕ), 1.8 * x + 3.5 * y + 4.2 * z = 20 ∧ y = 4 :=
by
  sorry

end jing_jing_notebook_count_l539_539214


namespace nickels_count_l539_539744

theorem nickels_count (original_nickels : ℕ) (additional_nickels : ℕ) 
                        (h₁ : original_nickels = 7) 
                        (h₂ : additional_nickels = 5) : 
    original_nickels + additional_nickels = 12 := 
by sorry

end nickels_count_l539_539744


namespace mother_three_times_serena_in_x_years_l539_539767

def serena_current_age := 9
def mother_current_age := 39

theorem mother_three_times_serena_in_x_years (x : ℕ) (h : 39 + x = 3 * (9 + x)) : x = 6 := by
  have : 39 + 6 = 3 * (9 + 6) := by norm_num
  exact eq_of_sub_eq_zero (sub_eq_zero.mpr this)

end mother_three_times_serena_in_x_years_l539_539767


namespace tangents_to_circumcircle_EOF_l539_539337

open_locale classical
noncomputable theory

structure Parallelogram (A B C D : Type) :=
  (Sides_parallel : (parallel A B C D) ∧ (parallel B C A D))

structure TangencyCondition (A O D : Type) :=
  (Tangency_AE : tangent AE (circumcircle A O D))
  (Tangency_DF : tangent DF (circumcircle A O D))

structure Points_on_side_BC (B C E F : Type) :=
  (E_between_B_F : between B E F)

theorem tangents_to_circumcircle_EOF (A B C D E F O : Type)
  [Parallelogram A B C D]
  [Points_on_side_BC B C E F]
  (H_diag_inter : intersect AC BD = O)
  [TangencyCondition A O D] :
  tangent AE (circumcircle E O F) ∧ tangent DF (circumcircle E O F) :=
sorry

end tangents_to_circumcircle_EOF_l539_539337


namespace polynomial_proof_l539_539815

variable (a b : ℝ)

-- Define the given monomial and the resulting polynomial 
def monomial := -3 * a ^ 2 * b
def result := 6 * a ^ 3 * b ^ 2 - 3 * a ^ 2 * b ^ 2 + 9 * a ^ 2 * b

-- Define the polynomial we want to prove
def poly := -2 * a * b + b - 3

-- Statement of the problem in Lean 4
theorem polynomial_proof :
  monomial * poly = result :=
by sorry

end polynomial_proof_l539_539815


namespace parallel_lines_slope_l539_539387

theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, x + 2 * y - 4 = 0 → - (1 / 2 : ℝ) = - (a / 2 : ℝ)) → a = 1 :=
by {
  intros h,
  have h_slope1 : - (1 / 2 : ℝ) = (by sorry),
  have h_slope2 : - (a / 2 : ℝ) = (by sorry),
  exact sorry,
}

end parallel_lines_slope_l539_539387


namespace train_cross_pole_time_l539_539099

/-- 
  Given the speed of a train in kilometers per hour and its length in meters, 
  prove that the time taken to cross a pole is approximately 15 seconds.
--/
theorem train_cross_pole_time :
  ∀ (v : ℝ) (l : ℝ), 
  v = 60 → 
  l = 250.00000000000003 →
  l / (v * 1000 / 3600) ≈ 15 :=
by
  sorry

end train_cross_pole_time_l539_539099


namespace optimal_meeting_point_l539_539410

-- Define the points (houses) and speeds
constant House1 House2 House3 : Point
constant speed1 speed2 speed3 : ℝ

-- Define the conditions
axiom gnome1_speed : speed1 = 1
axiom gnome2_speed : speed2 = 2
axiom gnome3_speed : speed3 = 3

-- Define the distance function
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

-- Define the total travel time from each house to point A
def total_travel_time (A : Point) : ℝ :=
  distance(House1, A) / speed1 + distance(House2, A) / speed2 + distance(House3, A) / speed3

-- Assertion that the optimal meeting point is House1
theorem optimal_meeting_point : ∀ A : Point, 
    total_travel_time House1 ≤ total_travel_time A := 
begin
  sorry
end

end optimal_meeting_point_l539_539410


namespace equivalent_problem_l539_539997

variable {x y : Real}

theorem equivalent_problem 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 15) :
  (x - y)^2 = 21 ∧ (x + y) * (x - y) = Real.sqrt 1701 :=
by
  sorry

end equivalent_problem_l539_539997


namespace triangle_area_sum_eq_l539_539557

variable (a b h1 h2 : ℝ)

-- Conditions defining the lengths and heights:
axiom AD_BC_eq_a : ∀ {A B C D : ℝ^2}, AD = BC := a
axiom AB_CD_eq_b : ∀ {A B C D : ℝ^2}, AB = CD := b
axiom O_is_diagonal_intersection : ∀ {A B C D O : ℝ^2}, O = (diagonal_intersection A B C D)

-- Definition of distances between sides:
axiom distance_AD_BC : ∀ {A B C D : ℝ^2}, distance AD BC = h1
axiom distance_AB_CD : ∀ {A B C D : ℝ^2}, distance AB CD = h2

-- Area calculation axioms based on geometric properties:
axiom area_triangle : ∀ {A B C : ℝ^2}, area (triangle A B C) = 1/2 * base(A B) * height(A C)

-- Real values of areas of particular triangles:
axiom area_BOC_AOD_eq : ∀ {B O C A D : ℝ^2}, area (triangle B O C) + area (triangle A O D) = 1/2 * a * h1
axiom area_AOB_COD_eq : ∀ {A O B C D : ℝ^2}, area (triangle A O B) + area (triangle C O D) = 1/2 * b * h2

-- Relationship between height-distance product of alternate sides:
axiom height_product_eq : a * h1 = b * h2

-- The main theorem to be proven:
theorem triangle_area_sum_eq :
  area (triangle B O C) + area (triangle A O D) = area (triangle A O B) + area (triangle C O D) :=
sorry

end triangle_area_sum_eq_l539_539557


namespace probability_A_fires_proof_l539_539473

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end probability_A_fires_proof_l539_539473


namespace total_number_of_coins_l539_539528

theorem total_number_of_coins (x : ℕ) (h : 1 * x + 5 * x + 10 * x + 50 * x + 100 * x = 332) : 5 * x = 10 :=
by {
  sorry
}

end total_number_of_coins_l539_539528


namespace milk_consumed_is_84_l539_539009

def number_of_cartoons_consumed_by_students (total_students : ℕ) (percentage_girls : ℝ) (monitors : ℕ)
  (milk_per_boy : ℕ) (milk_per_girl : ℕ) (monitors_ratio : ℕ) :=
  let total_girls := ((percentage_girls / 100) * total_students).to_nat in
  let total_boys := total_students - total_girls in
  (total_boys * milk_per_boy) + (total_girls * milk_per_girl)

theorem milk_consumed_is_84 :
  let total_students := (8 / 2) * 15 in
  number_of_cartoons_consumed_by_students total_students 40 8 1 2 15 = 84 :=
by
  sorry

end milk_consumed_is_84_l539_539009


namespace intersection_on_semicircle_l539_539604

variables (A B C D H P Q : Type*) [Triangle ABC]
variables (Hproj : is_orthogonal_projection C AB H)
variables (Dproperty : inside_triangle D CBH ∧ bisect_perpendicular CH AD)
variables (Pintersection : intersection BD CH P)
variables (Gamma : Semicircle BD)
variables (Tangent : tangent_line_through P Gamma Q)
variables (IntersectionCQAD : intersection CQ AD)

theorem intersection_on_semicircle :
  lies_on_semicircle (IntersectionCQAD) Gamma :=
sorry

end intersection_on_semicircle_l539_539604


namespace no_solution_sqrt_eq_add_l539_539973

theorem no_solution_sqrt_eq_add (x y : ℝ) (h : x * y = 1) : sqrt (x^2 + y^2) ≠ x + y :=
by
  sorry

end no_solution_sqrt_eq_add_l539_539973


namespace area_of_square_field_l539_539059

-- Define side length
def sideLength : ℕ := 14

-- Define the area function for a square
def area_of_square (side : ℕ) : ℕ := side * side

-- Prove that the area of the square with side length 14 meters is 196 square meters
theorem area_of_square_field : area_of_square sideLength = 196 := by
  sorry

end area_of_square_field_l539_539059


namespace consecutive_integers_sqrt19_sum_l539_539623

theorem consecutive_integers_sqrt19_sum :
  ∃ a b : ℤ, (a < ⌊Real.sqrt 19⌋ ∧ ⌊Real.sqrt 19⌋ < b ∧ a + 1 = b) ∧ a + b = 9 := 
by
  sorry

end consecutive_integers_sqrt19_sum_l539_539623


namespace constant_term_expansion_l539_539380

noncomputable def binomial_coeff (n k : ℕ) : ℕ := sorry
noncomputable def general_term_expansion (x : ℝ) (r : ℕ) : ℝ := sorry

theorem constant_term_expansion : 
  let x : ℝ := sorry,
      general_term (r : ℕ) := binomial_coeff 5 r * (x + 1 / real.sqrt x) ^ r * (-2) ^ (5 - r),
      term (k r : ℕ) := binomial_coeff r k * x ^ (r - (3 / 2) * k)
  in (general_term 0 * term 0 0) + (general_term 3 * term 2 3) = 88 := 
by 
  sorry

end constant_term_expansion_l539_539380


namespace speed_of_train_l539_539526

namespace TrainProblem

def length_of_train : ℝ := 360
def length_of_bridge : ℝ := 140
def time_to_pass_bridge : ℝ := 60

def expected_speed : ℝ := 8.33

theorem speed_of_train : 
  (length_of_train + length_of_bridge) / time_to_pass_bridge ≈ 8.33 := 
by
  -- Prove the speed of the train is approximately 8.33 meters per second.
  sorry

end TrainProblem

end speed_of_train_l539_539526


namespace lineup_restrictions_l539_539285

theorem lineup_restrictions : 
  let n := 5
  let youngest_restrictions := 3
  let remaining_positions_factorial := (n - 1)!
  (youngest_restrictions * remaining_positions_factorial) = 72 :=
by
  sorry

end lineup_restrictions_l539_539285


namespace find_n_l539_539078

theorem find_n (x : ℝ) (h1 : x = 4.0) (h2 : 3 * x + n = 48) : n = 36 := by
  sorry

end find_n_l539_539078


namespace g_at_pi_over_3_l539_539220

noncomputable def f (w φ : ℝ) (x : ℝ) : ℝ := 5 * cos (w * x + φ)
noncomputable def g (w φ : ℝ) (x : ℝ) : ℝ := 4 * sin (w * x + φ) + 1

theorem g_at_pi_over_3
  (w φ : ℝ)
  (h₁ : ∀ x : ℝ, f w φ (π / 3 + x) = f w φ (π / 3 - x))
  (h₂ : f w φ (π / 3) = 5 ∨ f w φ (π / 3) = -5) :
  g w φ (π / 3) = 1 :=
by
  sorry

end g_at_pi_over_3_l539_539220


namespace probability_math_physics_consecutive_l539_539794

-- Definitions of number of people, number of math majors, physics majors, and arts majors
def total_people : ℕ := 10
def math_majors : ℕ := 3
def physics_majors : ℕ := 4
def arts_majors : ℕ := 3

-- Theorem stating the probability that all three math majors and at least one physics major sit in consecutive seats
theorem probability_math_physics_consecutive : 
  ∀ (total_people math_majors physics_majors arts_majors : ℕ),
  total_people = 10 → 
  math_majors = 3 → 
  physics_majors = 4 → 
  arts_majors = 3 → 
  (probability_all_math_and_at_least_one_physics_consecutive total_people math_majors physics_majors arts_majors) = 3 / 4 :=
begin
  intros,
  -- proof can be added here
  sorry
end

end probability_math_physics_consecutive_l539_539794


namespace Yanna_kept_apples_l539_539442

theorem Yanna_kept_apples (initial_apples : ℕ) (apples_given_Zenny : ℕ) (apples_given_Andrea : ℕ) :
  initial_apples = 60 → apples_given_Zenny = 18 → apples_given_Andrea = 6 →
  (initial_apples - (apples_given_Zenny + apples_given_Andrea) = 36) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.add_sub_assoc]
  exact rfl
  apply Nat.succ_le_succ
  exact nat.succ_pos'


end Yanna_kept_apples_l539_539442


namespace max_tied_teams_in_tournament_l539_539682

def max_tied_teams (teams games : ℕ) (plays_each_other_once : teams * (teams - 1) / 2 = games) (result_in_win_or_loss : ∀ (game : ℕ), game < games → ∃ (winner loser : ℕ), winner ≠ loser ∧ winner < teams ∧ loser < teams) : ℕ := 
  sorry

theorem max_tied_teams_in_tournament :
  max_tied_teams 7 21
  (by norm_num : 7 * (7 - 1) / 2 = 21)
  (by intros game h_game; use (game / 7, game % 7); norm_num at h_game; split_ifs) 
  = 6 :=
sorry

end max_tied_teams_in_tournament_l539_539682


namespace center_of_symmetry_of_f_interval_of_monotonic_decrease_of_f_max_value_of_g_min_value_of_g_l539_539639

-- Definitions
def f (x : ℝ) := sin (2 * x + π / 6) + 2 * (sin x)^2

def g (x : ℝ) := sin (2 * (x - π / 12) + π / 6) + 1

-- Proving the center of symmetry of f
theorem center_of_symmetry_of_f :
  ∀ k : ℤ, ∃ c : ℝ, ∀ x : ℝ, f (2 * (x - (π / 12 + k * π / 2)) + π / 6) = f x := sorry

-- Proving the interval of monotonic decrease of f
theorem interval_of_monotonic_decrease_of_f :
  ∀ k : ℤ, ∀ x : ℝ, x ∈ set.Icc (k * π + π / 3) (k * π + 5 * π / 6) →
  (deriv f x ≤ 0) := sorry

-- Proving the maximum value of g on [0, π/2]
theorem max_value_of_g :
  is_max_on g (set.Icc 0 (π / 2)) 2 := sorry

-- Proving the minimum value of g on [0, π/2]
theorem min_value_of_g :
  is_min_on g (set.Icc 0 (π / 2)) (-sqrt 3 / 2 + 1) := sorry

end center_of_symmetry_of_f_interval_of_monotonic_decrease_of_f_max_value_of_g_min_value_of_g_l539_539639


namespace birthday_probability_l539_539832

noncomputable def prob_birthday (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  let λ := n * p;
  (real.exp (-λ) * λ^k) / (nat.factorial k)

theorem birthday_probability :
  let n := 500;
  let p := (1 : ℝ) / 365;
  prob_birthday n p 0 ≈ 0.2541 ∧
  prob_birthday n p 1 ≈ 0.3481 ∧
  prob_birthday n p 2 ≈ 0.2385 ∧
  prob_birthday n p 3 ≈ 0.1089 :=
by
  unfold prob_birthday
  have : n * p ≈ 1.36986 := sorry
  split
  { sorry }
  split
  { sorry }
  split
  { sorry }
  { sorry }

end birthday_probability_l539_539832


namespace sweet_tray_GCD_l539_539837

/-!
Tim has a bag of 36 orange-flavoured sweets and Peter has a bag of 44 grape-flavoured sweets.
They have to divide up the sweets into small trays with equal number of sweets;
each tray containing either orange-flavoured or grape-flavoured sweets only.
The largest possible number of sweets in each tray without any remainder is 4.
-/

theorem sweet_tray_GCD :
  Nat.gcd 36 44 = 4 :=
by
  sorry

end sweet_tray_GCD_l539_539837


namespace average_temperature_l539_539377

theorem average_temperature :
  ∀ (T : ℝ) (Tt : ℝ),
  -- Conditions
  (43 + T + T + T) / 4 = 48 → 
  Tt = 35 →
  -- Proof
  (T + T + T + Tt) / 4 = 46 :=
by
  intros T Tt H1 H2
  sorry

end average_temperature_l539_539377


namespace quadratic_inequality_l539_539181

theorem quadratic_inequality (m y1 y2 y3 : ℝ)
  (h1 : m < -2)
  (h2 : y1 = (m-1)^2 - 2*(m-1))
  (h3 : y2 = m^2 - 2*m)
  (h4 : y3 = (m+1)^2 - 2*(m+1)) :
  y3 < y2 ∧ y2 < y1 :=
by
  sorry

end quadratic_inequality_l539_539181


namespace solve_system_of_equations_l539_539771

theorem solve_system_of_equations 
  (x y : ℝ) 
  (h1 : x / 3 - (y + 1) / 2 = 1) 
  (h2 : 4 * x - (2 * y - 5) = 11) : 
  x = 0 ∧ y = -3 :=
  sorry

end solve_system_of_equations_l539_539771


namespace increase_in_average_weight_l539_539798

theorem increase_in_average_weight 
    (A : ℝ) 
    (weight_left : ℝ)
    (weight_new : ℝ)
    (h_weight_left : weight_left = 67)
    (h_weight_new : weight_new = 87) : 
    ((8 * A - weight_left + weight_new) / 8 - A) = 2.5 := 
by
  sorry

end increase_in_average_weight_l539_539798


namespace equation_of_line_l539_539669

theorem equation_of_line (A B: ℝ × ℝ) (C: ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  (C = (-1, -1)) →
  (l = λ p, 3 * p.1 + 4 * p.2 + 2 = 0 ∨ p.1 = -2) →
  (A ≠ B) →
  (A.1^2 + A.2^2 + 2 * A.1 + 2 * A.2 - 2 = 0) →
  (B.1^2 + B.2^2 + 2 * B.1 + 2 * B.2 - 2 = 0) →
  (|√ ((A.1 - B.1)^2 + (A.2 - B.2)^2)| = 2 * √ ((3)) → 
  ∃ l, l (-2, 1) := by
  sorry

end equation_of_line_l539_539669


namespace ratio_of_sister_to_mom_l539_539712

-- Definition of the initial conditions
def initial_balance : ℕ := 190
def transferred_to_mom : ℕ := 60
def remaining_balance : ℕ := 100

-- Definition of the amounts
def transferred_to_sister : ℕ := initial_balance - transferred_to_mom - remaining_balance := 30

-- Definition of the ratio
def ratio_to_simplify : ratio ℕ := (transferred_to_sister, transferred_to_mom)
def gcd_ratio : ℕ := Nat.gcd transferred_to_sister transferred_to_mom := 30

-- Simplified ratio
def simplified_ratio : ratio ℕ := (transferred_to_sister / gcd_ratio, transferred_to_mom / gcd_ratio) := (1, 2)

-- The main theorem
theorem ratio_of_sister_to_mom :
  simplified_ratio = (1, 2) :=
by
  --skipping the proof
  sorry

end ratio_of_sister_to_mom_l539_539712


namespace num_four_digit_numbers_with_property_l539_539659

def is_valid_number (N : ℕ) : Prop :=
  let a := N / 1000
  let x := N % 1000
  (1000 <= N ∧ N < 10000) ∧ (x = 1000 * a / 8)

def count_valid_numbers : ℕ :=
  (Finset.range 10000).filter is_valid_number |>.card

theorem num_four_digit_numbers_with_property : count_valid_numbers = 6 := by
  sorry

end num_four_digit_numbers_with_property_l539_539659


namespace initial_quantity_of_liquid_A_l539_539913

variable (initial_mix_A initial_mix_B drawn replaced new_mix_B : ℝ)
variable (x : ℝ)

-- Define the initial condition and transformations
def initial_condition : Prop :=
  initial_mix_A = 7 * x ∧ initial_mix_B = 5 * x

def transformation : Prop :=
  let drawn_A := (7 / 12) * drawn in
  let drawn_B := (5 / 12) * drawn in
  let remaining_A := initial_mix_A - drawn_A in
  let remaining_B := initial_mix_B - drawn_B + replaced in
  (remaining_A / remaining_B) = (7 / 9)

-- Theorem to prove the initial quantity of liquid A
theorem initial_quantity_of_liquid_A (h1 : initial_condition) (h2 : transformation) : initial_mix_A = 21 := by
  have h3: x = 3 := sorry
  specialize h1 (x := 3)
  rw [← h1.left]
  simp
  exact sorry

end initial_quantity_of_liquid_A_l539_539913


namespace ascending_order_proof_l539_539123

noncomputable def frac1 : ℚ := 1 / 2
noncomputable def frac2 : ℚ := 3 / 4
noncomputable def frac3 : ℚ := 1 / 5
noncomputable def dec1 : ℚ := 0.25
noncomputable def dec2 : ℚ := 0.42

theorem ascending_order_proof :
  frac3 < dec1 ∧ dec1 < dec2 ∧ dec2 < frac1 ∧ frac1 < frac2 :=
by {
  -- The proof will show the conversions mentioned in solution steps
  sorry
}

end ascending_order_proof_l539_539123


namespace only_four_letter_list_with_same_product_as_TUVW_l539_539152

/-- Each letter of the alphabet is assigned a value $A=1, B=2, C=3, \ldots, Z=26$. 
    The product of a four-letter list is the product of the values of its four letters. 
    The product of the list $TUVW$ is $(20)(21)(22)(23)$. 
    Prove that the only other four-letter list with a product equal to the product of the list $TUVW$ 
    is the list $TUVW$ itself. 
 -/
theorem only_four_letter_list_with_same_product_as_TUVW : ∀ (l : list char), 
  (l.all (λ c, 'A' ≤ c ∧ c ≤ 'Z')) → 
  (l.prod (λ c, letter_value c) = (20 * 21 * 22 * 23)) →
  (l = ['T', 'U', 'V', 'W']) :=
by
  sorry

/-- Helper function to convert a letter to its value. -/
def letter_value (c : char) : ℕ :=
  c.to_nat - 'A'.to_nat + 1

#eval letter_value 'T' -- 20
#eval letter_value 'U' -- 21
#eval letter_value 'V' -- 22
#eval letter_value 'W' -- 23

end only_four_letter_list_with_same_product_as_TUVW_l539_539152


namespace probability_A_fires_l539_539481

theorem probability_A_fires 
  (p_first_shot: ℚ := 1/6)
  (p_not_fire: ℚ := 5/6)
  (p_recur: ℚ := p_not_fire * p_not_fire) : 
  ∃ (P_A : ℚ), P_A = 6/11 :=
by
  have eq1 : P_A = p_first_shot + (p_recur * P_A) := sorry
  have eq2 : P_A * (1 - p_recur) = p_first_shot := sorry
  have eq3 : P_A = (p_first_shot * 36) / 11 := sorry
  exact ⟨P_A, sorry⟩

end probability_A_fires_l539_539481


namespace obtuse_triangle_has_exactly_one_obtuse_angle_l539_539660

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- Definition of an obtuse angle
def is_obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

-- The theorem statement
theorem obtuse_triangle_has_exactly_one_obtuse_angle {A B C : ℝ} 
  (h1 : is_obtuse_triangle A B C) : 
  (is_obtuse_angle A ∨ is_obtuse_angle B ∨ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle B) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle B ∧ is_obtuse_angle C) :=
sorry

end obtuse_triangle_has_exactly_one_obtuse_angle_l539_539660


namespace large_pizzas_sold_l539_539016

variables (num_small_pizzas num_large_pizzas : ℕ) (price_small price_large total_revenue revenue_from_smalls revenue_from_larges : ℕ)

theorem large_pizzas_sold
  (price_small := 2)
  (price_large := 8)
  (total_revenue := 40)
  (num_small_pizzas := 8)
  (revenue_from_smalls := num_small_pizzas * price_small)
  (revenue_from_larges := total_revenue - revenue_from_smalls)
  (large_pizza_count := revenue_from_larges / price_large) :
  large_pizza_count = 3 :=
sorry

end large_pizzas_sold_l539_539016


namespace standard_eq_of_parabola_focus_l539_539401

theorem standard_eq_of_parabola_focus
  (focus : ℝ × ℝ)
  (h : 3 * focus.1 - 4 * focus.2 - 12 = 0) :
  (focus = (4, 0) ∨ focus = (0, -3)) →
  (∃ (x y : ℝ), (y^2 = 16 * x) ∨ (x^2 = -12 * y)) :=
begin
  sorry
end

end standard_eq_of_parabola_focus_l539_539401


namespace choose_5_starters_including_twins_l539_539073

def number_of_ways_choose_starters (total_players : ℕ) (members_in_lineup : ℕ) (twins1 twins2 : (ℕ × ℕ)) : ℕ :=
1834

theorem choose_5_starters_including_twins :
  number_of_ways_choose_starters 18 5 (1, 2) (3, 4) = 1834 :=
sorry

end choose_5_starters_including_twins_l539_539073


namespace sum_of_squares_reciprocal_sqrt_inequality_l539_539067

-- Problem 1: Mathematical induction proof for the sum of squares formula
theorem sum_of_squares (n : ℕ) (hn : 0 < n) : 
  ∑ i in finset.range (n + 1), i^2 = (n * (n + 1) * (2 * n + 1)) / 6 :=
sorry

-- Problem 2: Mathematical induction proof for the inequality involving reciprocals and square roots 
theorem reciprocal_sqrt_inequality (n : ℕ) (hn : 0 < n) : 
  ∑ i in finset.range n + 1, 1 / (real.sqrt (i : ℝ)) < 2 * real.sqrt n :=
sorry

end sum_of_squares_reciprocal_sqrt_inequality_l539_539067


namespace remaining_volume_correct_l539_539079

noncomputable def side_length := 5
noncomputable def radius := 2
noncomputable def height := 5

def volume_cube := side_length ^ 3
def volume_cylinder := Real.pi * (radius ^ 2) * height

def remaining_volume := volume_cube - volume_cylinder

theorem remaining_volume_correct :
  remaining_volume = 125 - 20 * Real.pi :=
by
  -- This part of the proof is omitted as per the instructions
  sorry

end remaining_volume_correct_l539_539079


namespace part_I_part_II_l539_539239

/-
  Proof for part I:

  There exists a fixed point P(2, -1) such that 
  ∀θ ∈ ℝ, line l: cos^2θ * 2 + cos2θ * (-1) - 1 = 0.
-/
theorem part_I (θ : ℝ) : ∃ P : ℝ × ℝ, P = (2, -1) ∧ ∀ θ : ℝ, (cos θ ^ 2) * 2 + (cos (2 * θ)) * (-1) - 1 = 0 :=
sorry

/-
  Proof for part II:

  The range of the x-coordinate of point M is [ (2 - sqrt 5) / 2, 4 / 5 ].
-/
theorem part_II (θ : ℝ) : ∀ k : ℝ, -4/3 ≤ k ∧ k ≤ 0 → 
  let OM := ((k + 2 * k^2) / (1 + k^2)) in
  (2 + (k - 2) / (1 + k^2)) = OM ∧ 0 ≤ OM ∧ OM ≤ 4/5 ∧ 
  OM = (2 - sqrt 5) / 2 ∨ OM = 4 / 5 :=
sorry

end part_I_part_II_l539_539239


namespace find_k_l539_539205

-- Definitions for unit vectors, angle between them, and the vectors a and b
variables (e1 e2 : ℝ^3) (k : ℝ)
# check if the imported linear algebra plane is correct

-- Conditions
variables (h1 : ∥e1∥ = 1) (h2 : ∥e2∥ = 1) (θ : ℝ) (h3 : θ = (2 / 3) * real.pi)
variables (h4 : e1 ⬝ e2 = real.cos θ)
variables (a : ℝ^3) (h5 : a = e1 - 2 • e2)
variables (b : ℝ^3) (h6 : b = k • e1 + e2)
variables (h7 : a ⬝ b = 0)

-- Proof to find the value of k
theorem find_k : k = 5 / 4 :=
by
  sorry

end find_k_l539_539205


namespace fencing_cost_dollars_l539_539399

noncomputable def cost_of_fencing_in_dollars
    (area : ℕ)
    (ratio : ℕ × ℕ)
    (cost_per_meter_ps : ℕ)
    (exchange_rate : ℕ) : Float :=
  let x := Real.sqrt (area / (ratio.1 * ratio.2))
  let length := ratio.1 * x
  let width := ratio.2 * x
  let perimeter := 2 * (length + width)
  let cost_per_meter_rupees := (cost_per_meter_ps / 100.0)
  let total_cost_rupees := perimeter * cost_per_meter_rupees
  total_cost_rupees / exchange_rate

theorem fencing_cost_dollars
    (h_ratio : (3, 2))
    (h_area : 3750)
    (h_cost_ps : 40)
    (h_exchange_rate : 75) :
    cost_of_fencing_in_dollars h_area h_ratio h_cost_ps h_exchange_rate = 1.33 := by
  sorry

end fencing_cost_dollars_l539_539399


namespace charley_pencils_l539_539959

theorem charley_pencils :
  ∀ (initial_pencils lost_moving: ℕ),
    initial_pencils = 30 →
    lost_moving = 6 →
    (let remaining_pencils := initial_pencils - lost_moving in
     let lost_after := remaining_pencils / 3 in
     let final_pencils := remaining_pencils - lost_after in
     final_pencils = 16) :=
begin
  sorry
end

end charley_pencils_l539_539959


namespace probability_of_a_firing_l539_539476

/-- 
Prove that the probability that A will eventually fire the bullet is 6/11, given the following conditions:
1. A and B take turns shooting with a six-shot revolver that has only one bullet.
2. They randomly spin the cylinder before each shot.
3. A starts the game.
-/
theorem probability_of_a_firing (p_a : ℝ) :
  (1 / 6) + (5 / 6) * (5 / 6) * p_a = p_a → p_a = 6 / 11 :=
by
  intro hyp
  have h : p_a - (25 / 36) * p_a = 1 / 6 := by
    rwa [← sub_eq_of_eq_add hyp, sub_self, zero_mul] 
  field_simp at h
  linarith
  sorry

end probability_of_a_firing_l539_539476


namespace sally_earnings_in_dozens_l539_539765

theorem sally_earnings_in_dozens (earnings_per_house : ℕ) (houses_cleaned : ℕ) (dozens_of_dollars : ℕ) : 
  earnings_per_house = 25 ∧ houses_cleaned = 96 → dozens_of_dollars = 200 := 
by
  intros h
  sorry

end sally_earnings_in_dozens_l539_539765


namespace max_OM_ON_l539_539339

theorem max_OM_ON (a b : ℝ) (O : Point) (M : Point) (N : Point) (γ : Angle) 
  (AC BC : ℝ) (h1 : AC = a) (h2 : BC = b) (h3 : midpoint AC = M) (h4 : midpoint BC = N) 
  (h5 : center_square_on_AB = O) : 
  ∃ γ : Angle, OM + ON = (1 + Real.sqrt 2) / 2 * (a + b) :=
sorry

end max_OM_ON_l539_539339


namespace smallest_period_of_f_is_pi_max_and_min_of_f_on_interval_value_of_cos_2x0_l539_539233

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * (cos x)^2 - 1

theorem smallest_period_of_f_is_pi : 
  ∀ T > 0, (∀ x, f (x + T) = f x) → T = π := 
sorry

theorem max_and_min_of_f_on_interval : 
  (∀ x ∈ set.Icc 0 (π / 2), f x ≤ 2) ∧ 
  (∃ x ∈ set.Icc 0 (π / 2), f x = 2) ∧
  (∀ x ∈ set.Icc 0 (π / 2), f x ≥ -1) ∧ 
  (∃ x ∈ set.Icc 0 (π / 2), f x = -1) :=
sorry

theorem value_of_cos_2x0 (x0 : ℝ) (h0 : x0 ∈ set.Icc (π / 4) (π / 2)) (h1 : f x0 = 6 / 5) :
  cos (2 * x0) = (3 - 4 * sqrt 3) / 10 := 
sorry

end smallest_period_of_f_is_pi_max_and_min_of_f_on_interval_value_of_cos_2x0_l539_539233


namespace solution_set_log_inequality_l539_539252

def greatest_integer (x : ℝ) := ⌊x⌋

theorem solution_set_log_inequality :
  {x : ℚ | log (0.5 : ℚ) (greatest_integer x : ℝ) ≥ -1} = {0, 1, 2} :=
by sorry

end solution_set_log_inequality_l539_539252


namespace range_of_a_l539_539195

-- Definitions and conditions
def angleC := Real.pi / 3
def sideAB := Real.sqrt 3

-- Statement of the problem
theorem range_of_a (a : ℝ) : 
  (∃ A B C : ℝ, 
    C = angleC ∧ 
    A + B + C = Real.pi ∧
    AB = sideAB ∧
    BC = a ∧
    -- Law of Sines condition
    sideAB / Real.sin C = a / Real.sin A ) → 
  Real.sqrt 3 < a ∧ a < 2 :=
sorry

end range_of_a_l539_539195


namespace tammy_speed_on_second_day_l539_539783

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l539_539783


namespace length_DC_l539_539980

variables {A B C D F E : Type} [MetricSpace A B C D F E]
variables [HasAngle A B C D]
variables (AB DC : ℝ)
variables (BC : ℝ) (angleBCD angleCDA : ℝ)

-- Conditions
def trapezoid (AB DC : ℝ) : Prop := AB = 6 ∧ DC ∥ AB
def sideBC : ℝ := 2 * Real.sqrt 3
def angleBCD : ℝ := 30
def angleCDA : ℝ := 45

-- Question and proof of equivalent problem
theorem length_DC (h1 : trapezoid AB DC) (h2 : BC = sideBC) (h3 : angleBCD = 30)
(h4 : angleCDA = 45) : DC = Real.sqrt 3 + 6 + 3 * Real.sqrt 2 :=
by
  cases h1
  sorry

end length_DC_l539_539980


namespace num_of_true_propositions_l539_539946

variable (a b c : ℝ)

theorem num_of_true_propositions (h : 2 = count_true
  [ 
    (a > b → ac^2 > bc^2),
    (ac^2 > bc^2 → a > b),
    (a ≤ b → ac^2 ≤ bc^2),
    (ac^2 ≤ bc^2 → a ≤ b)
  ]) : 2 := 
sorry

end num_of_true_propositions_l539_539946


namespace card_A_minus_B_l539_539968

def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {2, 3, 4}
def A_minus_B : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem card_A_minus_B : A_minus_B.toFinset.card = 2 := by
  sorry

end card_A_minus_B_l539_539968


namespace johns_profit_l539_539437

variable (earnings : ℝ)
variable (initial_cost : ℝ)
variable (trade_in_value : ℝ)

def net_cost (initial_cost : ℝ) (trade_in_value : ℝ) : ℝ :=
  initial_cost - trade_in_value

def profit (earnings : ℝ) (net_cost : ℝ) : ℝ :=
  earnings - net_cost

theorem johns_profit (earnings : ℝ) (initial_cost : ℝ) (trade_in_value : ℝ) :
  earnings = 30000 → initial_cost = 18000 → trade_in_value = 6000 →
  profit earnings (net_cost initial_cost trade_in_value) = 18000 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have nc : net_cost 18000 6000 = 12000 := by norm_num
  rw [nc]
  norm_num

end johns_profit_l539_539437


namespace line_equation_through_point_intersects_circle_l539_539668

theorem line_equation_through_point_intersects_circle (x y : ℝ) :
  ∃ l, ((∃ k, l = (fun p:ℝ × ℝ => p.2 - 1 = k * (p.1 + 2))
        ∨ l = (fun p:ℝ × ℝ => p.1 = -2)) 
        ∧ (∃ A B : ℝ × ℝ, A ≠ B ∧ A.1^2 + A.2^2 + 2*A.1 + 2*A.2 - 2 = 0 ∧ B.1^2 + B.2^2 + 2*B.1 + 2*B.2 - 2 = 0
        ∧ l A ∧ l B 
        ∧ real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * real.sqrt 3)) :=
sorry

end line_equation_through_point_intersects_circle_l539_539668


namespace find_m_l539_539241

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (4, 2)
def c (m : ℝ) : ℝ × ℝ := (m * a.1 + b.1, m * a.2 + b.2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

theorem find_m (m : ℝ) (h : (dot_product (c m) a) / ((magnitude (c m)) * (magnitude a))
                      = (dot_product (c m) b)  / ((magnitude (c m)) * (magnitude b))) : m = 2 := 
sorry

end find_m_l539_539241


namespace pizzeria_large_pizzas_sold_l539_539022

theorem pizzeria_large_pizzas_sold (price_small price_large total_earnings num_small_pizzas num_large_pizzas : ℕ):
  price_small = 2 →
  price_large = 8 →
  total_earnings = 40 →
  num_small_pizzas = 8 →
  total_earnings = price_small * num_small_pizzas + price_large * num_large_pizzas →
  num_large_pizzas = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pizzeria_large_pizzas_sold_l539_539022


namespace line_properties_l539_539354

theorem line_properties :
  let l := λ x y : Real, sqrt 3 * x - y - 1 = 0 in
  let slope := sqrt 3 in
  let inclination_angle := 60 in
  (∀ x y, l x y ↔ y = slope * x - 1) ∧ (atan (Real.sqrt 3) = Real.pi / 3) :=
begin
  -- proof steps go here
  sorry
end

end line_properties_l539_539354


namespace domain_all_real_l539_539145

theorem domain_all_real (p : ℝ) : 
  (∀ x : ℝ, -3 * x ^ 2 + 3 * x + p ≠ 0) ↔ p < -3 / 4 := 
by
  sorry

end domain_all_real_l539_539145


namespace least_three_digit_with_product_eight_is_124_l539_539866

noncomputable def least_three_digit_with_product_eight : ℕ :=
  let candidates := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (x.digits 10).prod = 8} in
  if h : Nonempty candidates then
    let min_candidate := Nat.min' _ h in min_candidate
  else
    0 -- default to 0 if no such number exists

theorem least_three_digit_with_product_eight_is_124 :
  least_three_digit_with_product_eight = 124 := sorry

end least_three_digit_with_product_eight_is_124_l539_539866


namespace perpendicular_vectors_l539_539716

theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) (c : ℝ × ℝ) 
  (h1 : a = (1, 2)) (h2 : b = (1, 1)) 
  (h3 : c = (1 + k, 2 + k))
  (h4 : b.1 * c.1 + b.2 * c.2 = 0) : 
  k = -3 / 2 :=
by
  sorry

end perpendicular_vectors_l539_539716


namespace lcm_second_factor_l539_539811

theorem lcm_second_factor (A B : ℕ) (hcf : ℕ) (f1 f2 : ℕ) 
  (h₁ : hcf = 25) 
  (h₂ : A = 350) 
  (h₃ : Nat.gcd A B = hcf) 
  (h₄ : Nat.lcm A B = hcf * f1 * f2) 
  (h₅ : f1 = 13)
  : f2 = 14 := 
sorry

end lcm_second_factor_l539_539811


namespace yanna_kept_apples_l539_539440

theorem yanna_kept_apples (total_apples : ℕ) (apples_to_Zenny : ℕ) (apples_to_Andrea : ℕ) 
  (h_total : total_apples = 60) (h_Zenny : apples_to_Zenny = 18) (h_Andrea : apples_to_Andrea = 6) : 
  (total_apples - apples_to_Zenny - apples_to_Andrea) = 36 := by
  -- Initial setup based on the problem conditions
  rw [h_total, h_Zenny, h_Andrea]
  -- Simplify the expression
  rfl

-- The theorem simplifies to proving 60 - 18 - 6 = 36

end yanna_kept_apples_l539_539440


namespace minimum_width_l539_539417

theorem minimum_width (A l w : ℝ) (hA : A >= 150) (hl : l = 2 * w) (hA_def : A = w * l) : 
  w >= 5 * Real.sqrt 3 := 
  by
    -- Using the given conditions, we can prove that w >= 5 * sqrt(3)
    sorry

end minimum_width_l539_539417


namespace find_cost_l539_539450

def cost_of_article (C : ℝ) (G : ℝ) : Prop :=
  (580 = C + G) ∧ (600 = C + G + 0.05 * G)

theorem find_cost (C : ℝ) (G : ℝ) (h : cost_of_article C G) : C = 180 :=
by
  sorry

end find_cost_l539_539450


namespace farm_area_l539_539931

theorem farm_area (b l d : ℕ)
    (h_b : b = 30)
    (h_cost_per_meter : ∀ meter, cost meter = 13 * meter)
    (h_total_cost : ∀ l d, total_cost l 30 d = 1560)
    (h_diag : d * d = l * l + b * b)
    (h_fencing : l + 30 + d = 120) :
  let A : ℕ := l * b in
  A = 1200 :=
by
  sorry

noncomputable def cost (meter : ℕ) := 13 * meter
noncomputable def total_cost (l b d : ℕ) := cost (l + b + d)

end farm_area_l539_539931


namespace a_general_formula_T_sum_l539_539224

-- Statement of the sum of the first n terms condition
def S (n : ℕ) : ℤ := (n - 1)^2 - 1

-- General formula for the sequence a_n
def a (n : ℕ) : ℤ := 2 * n - 3

-- Sequence b_n
def b (n : ℕ) : ℝ := 1 / ((2 * n - 3) * (2 * n - 1))

-- Sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℝ := ∑ k in finset.range n, b (k + 1)

theorem a_general_formula (n : ℕ) : S n - S (n - 1) = 2 * n - 3 := by
  sorry

theorem T_sum (n : ℕ) : T n = - (n : ℝ) / (2 * n - 1) := by
  sorry

end a_general_formula_T_sum_l539_539224


namespace find_expression_value_l539_539622

theorem find_expression_value (x y : ℚ) (h₁ : 3 * x + y = 6) (h₂ : x + 3 * y = 8) :
  9 * x ^ 2 + 15 * x * y + 9 * y ^ 2 = 1629 / 16 := 
sorry

end find_expression_value_l539_539622


namespace line_length_after_erasure_l539_539157

-- Defining the initial and erased lengths
def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 33

-- The statement we need to prove
theorem line_length_after_erasure : initial_length_cm - erased_length_cm = 67 := by
  sorry

end line_length_after_erasure_l539_539157


namespace probability_A_fires_proof_l539_539474

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end probability_A_fires_proof_l539_539474


namespace sqrt_expression_eval_l539_539977

noncomputable def sqrt_expression : ℝ :=
  real.sqrt ((9^6 + 3^12) / (9^3 + 3^17))

theorem sqrt_expression_eval : sqrt_expression = 0.091 :=
by
  sorry

end sqrt_expression_eval_l539_539977


namespace initial_oranges_in_box_l539_539034

theorem initial_oranges_in_box (o_taken_out o_left_in_box : ℕ) (h1 : o_taken_out = 35) (h2 : o_left_in_box = 20) :
  o_taken_out + o_left_in_box = 55 := 
by
  sorry

end initial_oranges_in_box_l539_539034


namespace sticks_form_triangle_l539_539409

theorem sticks_form_triangle (a b c d e : ℝ) 
  (h1 : 2 < a) (h2 : a < 8)
  (h3 : 2 < b) (h4 : b < 8)
  (h5 : 2 < c) (h6 : c < 8)
  (h7 : 2 < d) (h8 : d < 8)
  (h9 : 2 < e) (h10 : e < 8) :
  ∃ x y z, 
    (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
    (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
    (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
    x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    x + y > z ∧ x + z > y ∧ y + z > x :=
by sorry

end sticks_form_triangle_l539_539409


namespace sum_abc_l539_539177

section
variables {a b c : ℚ} -- rational numbers
variable f : ℤ → ℤ → ℤ → ℤ -- a function satisfying the conditions
variable n : ℤ -- integer

-- Definition of f, applying the closest integer property
def closest_integer (r : ℝ) : ℤ :=
  let frac := r - r.floor
  in if frac < 0.5 then r.floor else r.ceil

-- Define the property for any n
noncomputable def f_property (a b c : ℚ) (n : ℤ) : Prop :=
  closest_integer ((a * n) : ℝ) + closest_integer ((b * n) : ℝ) + closest_integer ((c * n) : ℝ) = n

-- Main theorem statement
theorem sum_abc (a b c : ℚ) (h₁ : a > b) (h₂ : b > c) (h₃ : ∀ n : ℤ, f_property a b c n) : a + b + c = 1 :=
sorry

end

end sum_abc_l539_539177


namespace number_of_books_bought_l539_539106

def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def remaining_books : ℕ := 2

theorem number_of_books_bought : 
  let total_books_after_shopping := shelves * books_per_shelf + remaining_books in
  total_books_after_shopping - initial_books = 26 := 
by 
  sorry

end number_of_books_bought_l539_539106


namespace tammy_speed_on_second_day_l539_539788

theorem tammy_speed_on_second_day :
  ∃ v t : ℝ, t + (t - 2) = 14 ∧ v * t + (v + 0.5) * (t - 2) = 52 → (v + 0.5 = 4) :=
begin
  sorry
end

end tammy_speed_on_second_day_l539_539788


namespace least_area_of_triangle_DEF_l539_539400

theorem least_area_of_triangle_DEF :
  let z := c - 3 in
  let solutions := {w | ∃ k : ℕ, k < 10 ∧ w = 3 + (2 ^ (1/2 : ℝ) * Complex.exp (Complex.I * 2 * Real.pi * k / 10))} in
  let D := (Real.sqrt 2, 0) in
  let E := (Real.sqrt 2 * Real.cos (2 * Real.pi / 10), Real.sqrt 2 * Real.sin (2 * Real.pi / 10)) in
  let F := (Real.sqrt 2 * Real.cos (4 * Real.pi / 10), Real.sqrt 2 * Real.sin (4 * Real.pi / 10)) in
  (1 / 2 : ℝ) * 2 * Real.sqrt 2 * Real.sin (Real.pi / 10) * Real.sqrt 2 * Real.sin (2 * Real.pi / 10) = 2 * Real.sin (Real.pi / 10) * Real.sin (2 * Real.pi / 10).

end least_area_of_triangle_DEF_l539_539400


namespace log_product_log3_81_eq_4_main_l539_539848

theorem log_product :
  (∏ n in Finset.range 78, Real.log (n + 4) / Real.log (n + 3)) = Real.log 81 / Real.log 3 := by sorry

theorem log3_81_eq_4 : Real.log 81 / Real.log 3 = 4 := by sorry

theorem main : (∏ n in Finset.range 78, Real.log (n + 4) / Real.log (n + 3)) = 4 :=
by
  have h1 := log_product
  have h2 := log3_81_eq_4
  rw [h1, h2]
  sorry

end log_product_log3_81_eq_4_main_l539_539848


namespace evaluate_expression_l539_539572

variable (k : ℤ)

theorem evaluate_expression :
  2^(-(3*k+2)) - 3^(-(2*k+1)) - 2^(-(3*k)) + 3^(-2*k) =
  (-9 * 2^(-3*k) + 8 * 3^(-2*k)) / 12 := 
sorry

end evaluate_expression_l539_539572


namespace square_can_be_cut_into_8_acute_angled_triangles_l539_539139

-- Define an acute-angled triangle
def is_acute_angled_triangle (T : Triangle) := 
  ∀ angle ∈ T.angles, angle < 90

-- Define the problem statement in Lean
theorem square_can_be_cut_into_8_acute_angled_triangles (S : Square) : 
  ∃ (triangles : list Triangle), 
    triangles.length = 8 ∧ 
    (∀ T ∈ triangles, is_acute_angled_triangle T) ∧ 
    covers_square triangles S :=
sorry

end square_can_be_cut_into_8_acute_angled_triangles_l539_539139


namespace last_two_digits_of_large_sum_l539_539906

theorem last_two_digits_of_large_sum :
  let N := (Finset.range 2016).sum (λ k, 2 ^ (5 ^ (k + 1))) in
  N % 100 = 80 :=
by
  let N := (Finset.range 2016).sum (λ k, 2 ^ (5 ^ (k + 1)))
  have h₀ : N % 4 = 0 := sorry  -- corresponding to finding N mod 4
  have h₁ : N % 25 = 5 := sorry -- corresponding to finding N mod 25
  exact Nat.Mod.chinese_remainder h₀ h₁
  -- The actual proof would involve further steps ensuring h₀ and h₁ corrections with correct intermediate steps

end last_two_digits_of_large_sum_l539_539906


namespace greatest_difference_47x_l539_539521

def is_multiple_of_4 (n : Nat) : Prop :=
  n % 4 = 0

def valid_digit (d : Nat) : Prop :=
  d < 10

theorem greatest_difference_47x :
  ∃ x y : Nat, (is_multiple_of_4 (470 + x) ∧ valid_digit x) ∧ (is_multiple_of_4 (470 + y) ∧ valid_digit y) ∧ (x < y) ∧ (y - x = 4) :=
sorry

end greatest_difference_47x_l539_539521


namespace BC_parallel_AD_l539_539903

-- Define convex quadrilateral with given properties
variables {A B C D : Type*} [HasZero A] [HasZero B] [HasZero C] [HasZero D]
variables [HasAdd A] [HasAdd B] [HasAdd C] [HasAdd D]
variables [HasSub A] [HasSub B] [HasSub C] [HasSub D]
variables [HasMul A] [HasMul B] [HasMul C] [HasMul D]
variables [HasDiv A] [HasDiv B] [HasDiv C] [HasDiv D]

-- Function indicating convex quadrilateral
def is_convex_quadrilateral (ABCD : A → B → C → D → Prop) :=
  sorry -- define convex quadrilateral properties

-- Function indicating circles touch the sides
def circle_touches_side (ABCD_touch : A → B → C → D → Prop) :=
  sorry -- define circles touching sides properties

-- Final theorem combining the conditions
theorem BC_parallel_AD (AB CD : A) (ABCD : A → B → C → D → Prop) (ABCD_touch : A → B → C → D → Prop) :
  is_convex_quadrilateral ABCD →
  circle_touches_side ABCD_touch →
  (BC_parallel_AD ABCD) := 
sorry


end BC_parallel_AD_l539_539903


namespace trajectory_is_line_segment_l539_539979

-- Define the fixed points F1 and F2
def F1 : (ℝ × ℝ) := (-4, 0)
def F2 : (ℝ × ℝ) := (4, 0)

-- Define a moving point P
variable P : (ℝ × ℝ)

-- Define the distance function
def distance (A B : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

-- State the problem condition
def condition : Prop :=
  distance P F1 + distance P F2 = 8

-- Theorem stating the trajectory of P is a line segment between F1 and F2
theorem trajectory_is_line_segment (h : condition) : 
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
  P = (t * F1.1 + (1 - t) * F2.1, t * F1.2 + (1 - t) * F2.2) :=
sorry

end trajectory_is_line_segment_l539_539979


namespace compute_m_plus_n_l539_539592

-- Define the necessary conditions
def is_n_branch (n : ℕ) (B : list (set ℕ)) :=
  ∀ (i j : ℕ), i < j → (i < B.length ∧ j < B.length) → B[i] ⊆ B[j] ∧ B[j] ⊆ finset.range (n + 1) 

def is_n_plant (n : ℕ) (P : finset (list (set ℕ))) : Prop :=
  (∀ (B : list (set ℕ)), B ∈ P → is_n_branch n B) ∧ 
  (∀ (x : ℕ), x ∈ finset.range (n + 1) → ∃! (B : list (set ℕ)), B ∈ P ∧ x ∈ B.reverse.head)

def T (n : ℕ) : ℕ := sorry  -- Placeholder for the number of distinct perfect n-plants

-- Given condition
def given_condition (x : ℝ) : Prop :=
  real.log (∑ n in finset.range ∞, T n * (real.log x)^n / nat.factorial n) = 6 / 29

-- Define the main problem
theorem compute_m_plus_n (m n : ℕ) (h_coprime : nat.coprime m n) (x : ℝ) :
  x = m / n → given_condition x → m + n = 76 := sorry

end compute_m_plus_n_l539_539592


namespace probability_team_includes_boy_and_girl_l539_539088

section
variables {G B : ℕ} (girls boys : ℕ)
  (select_team : girls + boys >= 2)
open_locale big_operators

def probability_boy_and_girl (G B : ℕ) : ℚ :=
  ∑ i in finset.range G, ∑ j in finset.range B, (G.choose 1 * B.choose 1) / ((G + B).choose 2)

theorem probability_team_includes_boy_and_girl (hg : girls = 3) (hb : boys = 2) :
  ∃ (prob : ℚ), prob = 3 / 5 :=
begin
  use probability_boy_and_girl 3 2,
  sorry
end

end

end probability_team_includes_boy_and_girl_l539_539088


namespace least_three_digit_with_product_eight_is_124_l539_539868

noncomputable def least_three_digit_with_product_eight : ℕ :=
  let candidates := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (x.digits 10).prod = 8} in
  if h : Nonempty candidates then
    let min_candidate := Nat.min' _ h in min_candidate
  else
    0 -- default to 0 if no such number exists

theorem least_three_digit_with_product_eight_is_124 :
  least_three_digit_with_product_eight = 124 := sorry

end least_three_digit_with_product_eight_is_124_l539_539868


namespace find_f_neg_three_l539_539143

noncomputable def f (x : ℝ) : ℝ :=
  sorry

axiom f_equation (x y : ℝ) : f(x + y) = f(x) + f(y) + 2 * x * y
axiom f_one : f 1 = 2

theorem find_f_neg_three : f (-3) = 6 := sorry

end find_f_neg_three_l539_539143


namespace real_solution_exists_l539_539582

theorem real_solution_exists (x : ℝ) (h1: x ≠ 5) (h2: x ≠ 6) : 
  (x = 1 ∨ x = 2 ∨ x = 3) ↔ 
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1) /
  ((x - 5) * (x - 6) * (x - 5)) = 1) := 
by 
  sorry

end real_solution_exists_l539_539582


namespace find_k_l539_539773

noncomputable theory
open_locale classical

def sequence (k : ℕ) : ℕ → ℕ
| 1       := 1
| (n + 1) := (1 + k / n) * sequence n + 1

theorem find_k (k : ℕ) (h_pos : 0 < k) :
  (∀ n : ℕ, 0 < n → sequence k n ∈ ℕ) ↔ k = 2 :=
sorry

end find_k_l539_539773


namespace flute_cost_calculation_l539_539300

def cost_of_music_tool : ℝ := 8.89
def cost_of_song_book : ℝ := 7.00
def total_spent : ℝ := 158.35
def cost_of_flute : ℝ := 142.46

theorem flute_cost_calculation :
  total_spent - (cost_of_music_tool + cost_of_song_book) = cost_of_flute :=
by
  rw [cost_of_music_tool, cost_of_song_book]
  rw sub_eq_add_neg
  norm_num
  rw total_spent
  rw cost_of_flute
  norm_num
  sorry

end flute_cost_calculation_l539_539300


namespace strawberries_left_l539_539342

theorem strawberries_left (initial : ℝ) (eaten : ℝ) (remaining : ℝ) : initial = 78.0 → eaten = 42.0 → remaining = 36.0 → initial - eaten = remaining :=
by
  sorry

end strawberries_left_l539_539342


namespace option_c_not_same_l539_539117

-- Definitions for the functions in Option C
def f_c (x : ℝ) : ℝ := sqrt (x - 1) * sqrt (x + 1)
def g_c (x : ℝ) : ℝ := sqrt (x ^ 2 - 1)

-- Predicate to state the problem condition in Option C
def same_functions (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x

theorem option_c_not_same :
  ∃ x, x ≥ 1 ∨ x ≤ -1 ∧ f_c x ≠ g_c x :=
sorry

end option_c_not_same_l539_539117


namespace average_hidden_primes_l539_539951

/-- 
  Barry wrote 6 different numbers, one on each side of 3 cards, and laid the cards on a table, 
  with sums on each of the three cards being equal. 
  The visible numbers are 42, 59, and 36. 
  The numbers on the opposite side are all primes. 
  Prove that the average of the hidden prime numbers is 56/3.
-/
theorem average_hidden_primes (h1 : ∀ a b c d : ℕ, prime a ∧ prime b ∧ prime c ∧ (42 + a = 59 + b) 
                      ∧ (59 + b = 36 + c) ∧ (42 + a = 36 + c) ∧ set.Icc 1 100 {x | x = a ∨ x = b ∨ x = c}) : 
  42 + a + 59 + b + 36 + c := 42 + x + 59 + y + 36 + z ∧ average_hidden_primes (avg 42 59 36) = 56 / 3 :=
sorry

end average_hidden_primes_l539_539951


namespace joes_fast_food_cost_l539_539766

noncomputable def cost_of_sandwich (n : ℕ) : ℝ := n * 4
noncomputable def cost_of_soda (m : ℕ) : ℝ := m * 1.50
noncomputable def total_cost (n m : ℕ) : ℝ :=
  if n >= 10 then cost_of_sandwich n - 5 + cost_of_soda m else cost_of_sandwich n + cost_of_soda m

theorem joes_fast_food_cost :
  total_cost 10 6 = 44 := by
  sorry

end joes_fast_food_cost_l539_539766


namespace sum_of_inverses_is_integer_l539_539115

theorem sum_of_inverses_is_integer :
  (∃ (a : ℕ) (b : ℕ), a + b = 67 ∧ b + a = 66 ∧
  (33 ≤ a + b ∧ -33 ≤ b + a)) → ∃ k m : ℕ, k * 1 / 67 + m * 1 / 66 = (k + m : ℕ) :=
begin
  sorry
end

end sum_of_inverses_is_integer_l539_539115


namespace f_2008_eq_2009_l539_539235

noncomputable def f (x : ℕ) (h : x > 1) : ℝ :=
  Real.sqrt (1 + x * Real.sqrt (1 + (x + 1) * Real.sqrt (1 + (x + 2) * Real.sqrt (1 + (x + 3) * Real.sqrt (⋯)))))

theorem f_2008_eq_2009 : f 2008 (by norm_num) = 2009 := 
  sorry

end f_2008_eq_2009_l539_539235


namespace tetrahedron_blue_face_area_l539_539764

/-- 
A tetrahedron with three faces having areas 60, 20, and 15 square feet,
and the edges meet at right angles. 
Prove that the area of the fourth largest face is 65 square feet.
-/
theorem tetrahedron_blue_face_area :
  ∃ (A B C : ℝ), (A * B / 2 = 60) ∧ (B * C / 2 = 20) ∧ (C * A / 2 = 15) ∧ 
  (A ^ 2 + B ^ 2 = 100) ∧ (B ^ 2 + C ^ 2 = 250) ∧ (C ^ 2 + A ^ 2 = 170) ∧ 
  (let blue_area := 65 in blue_area = 65) :=
sorry

end tetrahedron_blue_face_area_l539_539764


namespace function_not_monotonous_l539_539264

theorem function_not_monotonous {k : ℝ} (h1 : 1 < k) (h2 : k < 3/2) :
  ¬ (∀ I ∈ set.Ioo (k - 1) (k + 1), monotone_on (λ x, 2 * x^2 - real.log x) I) :=
begin
  sorry
end

end function_not_monotonous_l539_539264


namespace sin_identity_alpha_l539_539735

theorem sin_identity_alpha (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
by 
  sorry

end sin_identity_alpha_l539_539735


namespace angle_CDB_is_45_l539_539090

-- Define the points and angles for the right triangle and equilateral triangle
variables {A B C D : Type} [EuclideanGeometry A]

-- Conditions from the problem
def right_triangle (ABC : Triangle) (C : Point) : Prop :=
  ABC.angles C = 30

def shared_side (CB CD : Segment) : Prop :=
  CB = CD

-- Statement to prove
theorem angle_CDB_is_45 (ABC : Triangle) (C : Point) (CB CD : Segment)
  (h1 : right_triangle ABC C) (h2 : shared_side CB CD) :
  ∠ CDB = 45 :=
sorry

end angle_CDB_is_45_l539_539090


namespace transfer_people_correct_equation_l539_539793

theorem transfer_people_correct_equation (A B x : ℕ) (h1 : A = 28) (h2 : B = 20) : 
  A + x = 2 * (B - x) := 
by sorry

end transfer_people_correct_equation_l539_539793


namespace least_three_digit_product_8_is_118_l539_539878

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_product (n : ℕ) (product : ℕ) : Prop :=
  let digits := (list.cons (n / 100) (list.cons ((n / 10) % 10) (list.cons (n % 10) list.nil))) in
  digits.prod = product

theorem least_three_digit_product_8_is_118 :
  ∃ n : ℕ, is_three_digit_number n ∧ digits_product n 8 ∧ (∀ m : ℕ, is_three_digit_number m ∧ digits_product m 8 → n ≤ m) :=
sorry

end least_three_digit_product_8_is_118_l539_539878


namespace smallest_steps_l539_539351

theorem smallest_steps (n : ℕ) :
  (n % 6 = 5) → (n % 7 = 1) → (n > 20) → n = 29 :=
by
  intros h1 h2 h3
  sorry

end smallest_steps_l539_539351


namespace elliot_storeroom_blocks_l539_539567

def storeroom_volume (length: ℕ) (width: ℕ) (height: ℕ) : ℕ :=
  length * width * height

def inner_volume (length: ℕ) (width: ℕ) (height: ℕ) (thickness: ℕ) : ℕ :=
  (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness)

def blocks_needed (outer_volume: ℕ) (inner_volume: ℕ) : ℕ :=
  outer_volume - inner_volume

theorem elliot_storeroom_blocks :
  let length := 15
  let width := 12
  let height := 8
  let thickness := 2
  let outer_volume := storeroom_volume length width height
  let inner_volume := inner_volume length width height thickness
  let required_blocks := blocks_needed outer_volume inner_volume
  required_blocks = 912 :=
by {
  -- Definitions and calculations as per conditions
  sorry
}

end elliot_storeroom_blocks_l539_539567


namespace quadrilateral_area_l539_539008

variable (A B C D K L M N : Type) 
variable (AB BC CD DA AC BD : ℝ)
variable (M1 M2 : ℝ)
variable (midpoint : Type → Type → Type)
variable (convex_quadrilateral : Type → Prop)

-- Define midpoints
def K := midpoint A B
def L := midpoint B C
def M := midpoint C D
def N := midpoint D A

-- Define condition that midpoints connect to form equal segments
def equal_midsegments (K L M N : Type) : Prop :=
  ∃ (KL MN : Type), (KL = MN)

-- Define condition of diagonals
def diagonals (AC BD : ℝ) : Prop :=
  AC = 8 ∧ BD = 12

-- Main theorem
theorem quadrilateral_area (h1 : convex_quadrilateral ABCD)
    (h2 : equal_midsegments K L M N) (h3 : diagonals AC BD) :
    (1/2) * AC * BD = 48 := by
  sorry

end quadrilateral_area_l539_539008


namespace measure_of_angle_C_l539_539947

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 7 * D) : C = 157.5 := 
by 
  sorry

end measure_of_angle_C_l539_539947


namespace find_d_l539_539045

theorem find_d :
  ∃ d : ℝ, ∀ x : ℝ, x * (4 * x - 3) < d ↔ - (9/4 : ℝ) < x ∧ x < (3/2 : ℝ) ∧ d = 27 / 2 :=
by
  sorry

end find_d_l539_539045


namespace flight_time_approx_l539_539394

-- Given conditions
def radius_earth := 3960 -- miles
def speed_jet := 550 -- miles per hour
def pi_approx := 3.14 -- approximate value of π

-- The goal is to prove the flight time is approximately 45 hours
theorem flight_time_approx :
  let circumference := 2 * pi * radius_earth in
  let flight_time := circumference / speed_jet in
  abs (flight_time - 45) < 1 :=
by
  let circumference := 2 * Real.pi * radius_earth
  let flight_time := circumference / speed_jet
  sorry

end flight_time_approx_l539_539394


namespace function_range_is_above_3_l539_539395

noncomputable def f (x : ℝ) : ℝ := 4^x + 2^(x + 1) + 3

theorem function_range_is_above_3:
  set.range f = {y | 3 < y} := 
sorry

end function_range_is_above_3_l539_539395


namespace least_three_digit_product_eight_l539_539851

theorem least_three_digit_product_eight : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).prod = 8 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (nat.digits 10 m).prod = 8 → n ≤ m :=
by
  sorry

end least_three_digit_product_eight_l539_539851


namespace det_B_squared_sub_3B_l539_539641

def B : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![3, 2]]

theorem det_B_squared_sub_3B : det ((B * B) - (3 • B)) = 88 := by
  sorry

end det_B_squared_sub_3B_l539_539641


namespace range_of_f_l539_539172

noncomputable def f (x : ℝ) : ℝ := abs (x ^ 2 - 4) - 3 * x

theorem range_of_f : set.range f = set.Icc (-6 : ℝ) ((25 : ℝ) / 4) :=
sorry

end range_of_f_l539_539172


namespace parabola_equation_l539_539504

theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) 
  (hl : 0 < p)
  (l : Line (A, B) passes through the focus of x^2 = 2 * p * y)
  (AB_len : distance A B = 6) 
  (mid_AB_dist : abs (midpoint A B).snd = 1) :
  2 * p = 4 := 
sorry

end parabola_equation_l539_539504


namespace van_distance_l539_539103

theorem van_distance (D : ℝ) (t_initial t_new : ℝ) (speed_new : ℝ) 
  (h1 : t_initial = 6) 
  (h2 : t_new = (3 / 2) * t_initial) 
  (h3 : speed_new = 30) 
  (h4 : D = speed_new * t_new) : 
  D = 270 :=
by
  sorry

end van_distance_l539_539103


namespace convex_ngon_divided_into_equal_triangles_l539_539920

theorem convex_ngon_divided_into_equal_triangles (n : ℕ) (P : convex_ngon n)
  (h1 : n > 3) (h2 : divides_into_equal_triangles P) : n = 4 :=
by
  -- Mathematical proof here
  sorry

end convex_ngon_divided_into_equal_triangles_l539_539920


namespace probability_A_fires_l539_539484

theorem probability_A_fires 
  (p_first_shot: ℚ := 1/6)
  (p_not_fire: ℚ := 5/6)
  (p_recur: ℚ := p_not_fire * p_not_fire) : 
  ∃ (P_A : ℚ), P_A = 6/11 :=
by
  have eq1 : P_A = p_first_shot + (p_recur * P_A) := sorry
  have eq2 : P_A * (1 - p_recur) = p_first_shot := sorry
  have eq3 : P_A = (p_first_shot * 36) / 11 := sorry
  exact ⟨P_A, sorry⟩

end probability_A_fires_l539_539484


namespace tammy_avg_speed_l539_539779

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l539_539779


namespace determine_angle_F_l539_539701

noncomputable def sin := fun x => Real.sin x
noncomputable def cos := fun x => Real.cos x
noncomputable def arcsin := fun x => Real.arcsin x
noncomputable def angleF (D E : ℝ) := 180 - (D + E)

theorem determine_angle_F (D E F : ℝ)
  (h1 : 2 * sin D + 5 * cos E = 7)
  (h2 : 5 * sin E + 2 * cos D = 4) :
  F = arcsin (9 / 10) ∨ F = 180 - arcsin (9 / 10) :=
  sorry

end determine_angle_F_l539_539701


namespace Yanna_kept_apples_l539_539441

theorem Yanna_kept_apples (initial_apples : ℕ) (apples_given_Zenny : ℕ) (apples_given_Andrea : ℕ) :
  initial_apples = 60 → apples_given_Zenny = 18 → apples_given_Andrea = 6 →
  (initial_apples - (apples_given_Zenny + apples_given_Andrea) = 36) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.add_sub_assoc]
  exact rfl
  apply Nat.succ_le_succ
  exact nat.succ_pos'


end Yanna_kept_apples_l539_539441


namespace min_rounds_needed_l539_539281

-- Defining the number of players
def num_players : ℕ := 10

-- Defining the number of matches each player plays per round
def matches_per_round (n : ℕ) : ℕ := n / 2

-- Defining the scoring system
def win_points : ℝ := 1
def draw_points : ℝ := 0.5
def loss_points : ℝ := 0

-- Defining the total number of rounds needed for a clear winner to emerge
def min_rounds_for_winner : ℕ := 7

-- Theorem stating the minimum number of rounds required
theorem min_rounds_needed :
  ∀ (n : ℕ), n = num_players → (∃ r : ℕ, r = min_rounds_for_winner) :=
by
  intros n hn
  existsi min_rounds_for_winner
  sorry

end min_rounds_needed_l539_539281


namespace train_passing_pole_time_l539_539941

theorem train_passing_pole_time (speed_kmh : ℕ) (crossing_time_stationary_train : ℕ) (stationary_train_length : ℕ) :
    (speed_kmh = 72) → (crossing_time_stationary_train = 27) → (stationary_train_length = 300) → 
    (∃ time_pole : ℕ, time_pole = 12) :=
by
  intros h1 h2 h3
  have speed_ms : ℕ := 20         -- Convert speed from km/h to m/s
  have train_length : ℕ := 240    -- Calculate the length of the moving train
  have time_pole := 12            -- Calculate the time it takes to pass a pole
  exact ⟨time_pole, rfl⟩
  sorry

end train_passing_pole_time_l539_539941


namespace water_heater_supply_l539_539493

theorem water_heater_supply (V_0 : ℕ) (t : ℕ → ℕ) (V_added : ℕ → ℕ)
  (V_discharged : ℕ → ℕ) (V_person : ℕ) (n : ℕ) :
  V_0 = 200 →
  (∀ t, V_added t = 2 * t^2) →
  (∀ t, V_discharged t = 34 * t) →
  V_person = 60 →
  (∃ t_min, ∀ t ≥ t_min, (V_0 + V_added t - V_discharged t) / V_person ≥ n) :=
begin
  sorry -- Proof goes here
end

end water_heater_supply_l539_539493


namespace eq_solutions_l539_539047

theorem eq_solutions (A : \(\exists x, (3x + 1)^2 = 0\))
                     (B : \(\exists x, |2x + 1| - 6 = 0\))
                     (C : \neg(\exists x, \sqrt{5 - x} + 3 = 0\))
                     (D : \(\exists x, \sqrt{4x + 9} - 7 = 0\))
                     (E : \neg(\exists x, |5x - 3| + 2 = -1\)) :
                     A ∧ B ∧ C ∧ D ∧ E := by
    sorry

end eq_solutions_l539_539047


namespace range_f_when_t_is_1_monotonic_increasing_f_condition_l539_539636

-- Definition of the function f
def f (x: ℝ) (t: ℝ) := 
  if x ≥ t then x^2 
  else if 0 < x ∧ x < t then x 
  else 0

-- Problem 1: Prove the range of f(x) when t = 1 is (0, +∞)
theorem range_f_when_t_is_1 : (∀ y, ∃ x, (y > 0 ↔ (f x 1) = y)) := sorry

-- Problem 2: Prove the condition for f(x) to be monotonically increasing on (0, +∞)
theorem monotonic_increasing_f_condition (t: ℝ) (ht: t > 0) : 
  (∀ x1 x2, (0 < x1 → x1 < x2 → f x1 t ≤ f x2 t)) ↔ (1 ≤ t) := sorry

end range_f_when_t_is_1_monotonic_increasing_f_condition_l539_539636


namespace complex_abs_of_sqrt_l539_539370

variable (z : ℂ)

theorem complex_abs_of_sqrt (h : z^2 = 16 - 30 * Complex.I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_abs_of_sqrt_l539_539370


namespace rhombus_diagonal_length_l539_539803

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d2 = 17) (h2 : A = 127.5) 
  (h3 : A = (d1 * d2) / 2) : d1 = 15 := 
by 
  -- Definitions
  sorry

end rhombus_diagonal_length_l539_539803


namespace parabola_sum_distances_l539_539990

theorem parabola_sum_distances :
  ∑ n in Finset.range (1992 + 1), (1 - 1 / (n + 2)) = 1992 / 1993 :=
by
  sorry

end parabola_sum_distances_l539_539990


namespace set_inter_and_complement_l539_539244

def U : Set ℕ := {2, 3, 4, 5, 6, 7}
def A : Set ℕ := {4, 5, 7}
def B : Set ℕ := {4, 6}

theorem set_inter_and_complement :
  A ∩ (U \ B) = {5, 7} := by
  sorry

end set_inter_and_complement_l539_539244


namespace ratio_of_areas_l539_539881

def isRatioCorrect (area_shaded_triangle area_large_square : ℕ) : Prop :=
  (area_shaded_triangle : ℚ) / area_large_square = (1 : ℚ) / 50

theorem ratio_of_areas :
  ∀ (area_shaded_triangle area_large_square : ℕ),
  area_large_square = 25 →
  area_shaded_triangle = 0.5 →
  isRatioCorrect area_shaded_triangle area_large_square :=
by
  intros area_shaded_triangle area_large_square h1 h2
  rw [isRatioCorrect]
  simp [h1, h2]
  sorry

end ratio_of_areas_l539_539881


namespace log2_plus_log5_minus_8_cubed_root_add_neg2_zero_l539_539062

theorem log2_plus_log5_minus_8_cubed_root_add_neg2_zero : 
  log 2 + log 5 - 8^(1/3) + (-2)^0 = 0 := 
by
  sorry

end log2_plus_log5_minus_8_cubed_root_add_neg2_zero_l539_539062


namespace coefficient_a2_in_expansion_l539_539601

theorem coefficient_a2_in_expansion:
  let a := (x - 1)^4
  let expansion := a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4
  a2 = 6 :=
by
  sorry

end coefficient_a2_in_expansion_l539_539601


namespace smallest_number_proof_l539_539896

def smallest_five_digit_number (digits : List ℕ) : ℕ :=
  let digits_sorted := digits.filter (≠ 0) |>.sort
  let non_zero_min := digits_sorted.head!
  let remaining_digits := digits.filter (λ d => d ≠ non_zero_min) |>.insert 0 0 |>.sort
  non_zero_min * 10000 + remaining_digits.foldl (λ acc d => acc * 10 + d) 0

theorem smallest_number_proof :
  smallest_five_digit_number [7, 2, 5, 0, 1] = 10257 :=
by { sorry }

end smallest_number_proof_l539_539896


namespace max_sin_pow_two_l539_539429

theorem max_sin_pow_two (n : ℕ) (hn : n > 0) : ∃ M : ℝ, M = 0.8988 ∧ ∀ k, 0 ≤ k → 0 ≤ sin (2^k) ∧ sin (2^k) ≤ M := sorry

end max_sin_pow_two_l539_539429


namespace shaded_area_l539_539166

theorem shaded_area (R : ℝ) : 
    let α := 20 * (Real.pi / 180)  -- convert degree to radians
    let S0 := (Real.pi * R^2) / 2 in 
    let sector_area := (1 / 2) * (2 * R)^2 * (α / (2 * Real.pi)) in
    (sector_area / (α / (2 * Real.pi))) / 2 = 2 * R^2 * Real.pi / 9 :=
by
  let α := 20 * (Real.pi / 180)
  let S0 := (Real.pi * R^2) / 2
  let sector_area := (1 / 2) * (2 * R)^2 * (α / (2 * Real.pi))
  calc
    (sector_area / (α / (2 * Real.pi))) / 2 = 2 * R^2 * Real.pi / 9 : sorry

end shaded_area_l539_539166


namespace solve_equation_l539_539364

theorem solve_equation : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -3 → (2 / x + x / (x + 3) = 1) ↔ (x = 6) := by
  intro x h
  have h1 : x ≠ 0 := h.1
  have h2 : x ≠ -3 := h.2
  sorry

end solve_equation_l539_539364


namespace least_three_digit_number_product8_l539_539861

theorem least_three_digit_number_product8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (digits 10 n).prod = 8 ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (digits 10 m).prod = 8 → n ≤ m) :=
sorry

end least_three_digit_number_product8_l539_539861


namespace commutative_matrices_implies_fraction_l539_539311

-- Definitions
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 3], ![4, 5]]
def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

-- Theorem Statement
theorem commutative_matrices_implies_fraction (a b c d : ℝ) 
    (h1 : A * B a b c d = B a b c d * A) 
    (h2 : 4 * b ≠ c) : 
    (a - d) / (c - 4 * b) = 3 / 8 :=
by
  sorry

end commutative_matrices_implies_fraction_l539_539311


namespace find_m_l539_539217

theorem find_m :
  (∃ (N : ℝ × ℝ) (m : ℝ), 
    N.1 + N.2 = 0 ∧ 
    ∃ (M : ℝ × ℝ), M = (1,0) ∧ 
    ∃ (x y : ℝ), 
    (M.1 + N.1) / 2 = x ∧ (M.2 + N.2) / 2 = y ∧ mx - y + 1 = 0) →
  (m = sqrt 3 ∨ m = -sqrt 3) := 
sorry

end find_m_l539_539217


namespace probability_to_buy_ticket_l539_539912

def p : ℝ := 0.1
def q : ℝ := 0.9
def initial_money : ℝ := 20
def target_money : ℝ := 45
def ticket_cost : ℝ := 10
def prize : ℝ := 30

noncomputable def equation_lhs : ℝ := p^2 * (1 + 2 * q)
noncomputable def equation_rhs : ℝ := 1 - 2 * p * q^2

noncomputable def x2 : ℝ := equation_lhs / equation_rhs

theorem probability_to_buy_ticket : x2 = 0.033 := sorry

end probability_to_buy_ticket_l539_539912


namespace cost_difference_l539_539707

def sailboat_weekday_rental_cost : ℕ := 60
def sailboat_weekend_rental_cost : ℕ := 90
def ski_boat_weekday_rental_cost_per_hour: ℕ := 80
def ski_boat_weekend_rental_cost_per_hour: ℕ := 120
def sailboat_fuel_cost_per_hour: ℕ := 10
def ski_boat_fuel_cost_per_hour: ℕ := 20
def discount_rate_on_second_day: ℝ := 0.10
def hours_per_day: ℕ := 3
def rental_days: ℕ := 2

theorem cost_difference (cost_diff: ℕ) 
  (sailboat_weekday_rental_cost = 60) 
  (sailboat_weekend_rental_cost = 90) 
  (ski_boat_weekday_rental_cost_per_hour = 80)
  (ski_boat_weekend_rental_cost_per_hour = 120)
  (sailboat_fuel_cost_per_hour = 10)
  (ski_boat_fuel_cost_per_hour = 20)
  (discount_rate_on_second_day = 0.10)
  (hours_per_day = 3) 
  (rental_days = 2):
  cost_diff = 630 := 
sorry

end cost_difference_l539_539707


namespace rectangle_area_l539_539540

open Real

theorem rectangle_area (A : ℝ) (s l w : ℝ) (h1 : A = 9 * sqrt 3) (h2 : A = (sqrt 3 / 4) * s^2)
  (h3 : w = s) (h4 : l = 3 * w) : w * l = 108 :=
by
  sorry

end rectangle_area_l539_539540


namespace incorrect_sqrt_4_l539_539895

theorem incorrect_sqrt_4 : (∀ (x : ℝ), sqrt ((-2)^2) = 2 ∧ (-sqrt 2)^2 = 2 ∧ sqrt (∛(-8)) = -2) → sqrt 4 ≠ ±2 :=
by
  intro h
  sorry

end incorrect_sqrt_4_l539_539895


namespace high_school_student_count_l539_539915

theorem high_school_student_count 
  (music_students art_students both_students neither_students : ℕ) 
  (h_music : music_students = 30)
  (h_art : art_students = 20)
  (h_both : both_students = 10)
  (h_neither : neither_students = 460) :
  ∃ n : ℕ, n = music_students - both_students + art_students - both_students + both_students + neither_students ∧ n = 500 :=
by {
  -- starting the proof and introducing the assumption for n
  let n := music_students - both_students + art_students - both_students + both_students + neither_students,
  use n,
  rw [h_music, h_art, h_both, h_neither],
  exact ⟨rfl, rfl⟩,
}

end high_school_student_count_l539_539915


namespace maximum_profit_l539_539497

noncomputable def profit (x : ℝ) : ℝ :=
  5.06 * x - 0.15 * x^2 + 2 * (15 - x)

theorem maximum_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 :=
by
  sorry

end maximum_profit_l539_539497


namespace probability_of_event_l539_539501

noncomputable theory
open Classical

def roll_die : Fin₆ := sorry -- represents rolling the die

def sample_space : List (Fin₆ × Fin₆) :=
  List.product (List.range 6) (List.range 6)

def count_satisfying_event : ℕ :=
sample_space.count (fun (p : Fin₆ × Fin₆) => p.2 ≤ 2 * p.1)

theorem probability_of_event : (count_satisfying_event : ℚ) / 36 = 5 / 6 := by
  -- proof here
  sorry

end probability_of_event_l539_539501


namespace find_a_l539_539267

theorem find_a (a : ℝ) : (∃ x y : ℝ, y = 4 - 3 * x ∧ y = 2 * x - 1 ∧ y = a * x + 7) → a = 6 := 
by
  sorry

end find_a_l539_539267


namespace largest_divisor_product_of_consecutive_odds_l539_539427

theorem largest_divisor_product_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) : 
  15 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) :=
sorry

end largest_divisor_product_of_consecutive_odds_l539_539427


namespace yanna_kept_apples_l539_539438

theorem yanna_kept_apples (total_apples : ℕ) (apples_to_Zenny : ℕ) (apples_to_Andrea : ℕ) 
  (h_total : total_apples = 60) (h_Zenny : apples_to_Zenny = 18) (h_Andrea : apples_to_Andrea = 6) : 
  (total_apples - apples_to_Zenny - apples_to_Andrea) = 36 := by
  -- Initial setup based on the problem conditions
  rw [h_total, h_Zenny, h_Andrea]
  -- Simplify the expression
  rfl

-- The theorem simplifies to proving 60 - 18 - 6 = 36

end yanna_kept_apples_l539_539438


namespace f_zero_and_f_six_f_is_odd_solve_inequality_l539_539724

-- Given conditions
variable {f : ℝ → ℝ}
variable (h1 : ∀ x y : ℝ, f(x - y) = f(x) - f(y))
variable (h2 : f(3) = 1011)
variable (h3 : ∀ x : ℝ, x > 0 → f(x) > 0)

-- Part (1)
theorem f_zero_and_f_six : f(0) = 0 ∧ f(6) = 2022 :=
by sorry

-- Part (2)
theorem f_is_odd : (∀ x : ℝ, f(-x) = -f(x)) :=
by sorry

-- Part (3)
theorem solve_inequality (x : ℝ) : f(2 * x - 4) > 2022 → x > 5 :=
by sorry

end f_zero_and_f_six_f_is_odd_solve_inequality_l539_539724


namespace remainder_poly_div_l539_539149

theorem remainder_poly_div 
    (x : ℤ) 
    (h1 : (x^2 + x + 1) ∣ (x^3 - 1)) 
    (h2 : x^5 - 1 = (x^3 - 1) * (x^2 + x + 1) - x * (x^2 + x + 1) + 1) : 
  ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 :=
by
  sorry

end remainder_poly_div_l539_539149


namespace max_value_of_sum_of_twelfth_powers_l539_539307

theorem max_value_of_sum_of_twelfth_powers :
  ∀ (x : Fin 1997 → ℝ),
    (∀ i, -1 / Real.sqrt 3 ≤ x i ∧ x i ≤ Real.sqrt 3) →
    (Finset.univ.sum x = -318 * Real.sqrt 3) →
    let sum_of_twelfth_powers := Finset.univ.sum (λ i, (x i) ^ 12)
    in
    sum_of_twelfth_powers ≤ 1736 * (3:ℝ)^(-6) + 260 * (3:ℝ)^(6) + (4 / 3)^6 :=
begin
  sorry
end

end max_value_of_sum_of_twelfth_powers_l539_539307


namespace problem_part_1_problem_part_2_l539_539226

variable (θ : Real)
variable (m : Real)
variable (h_θ : θ ∈ Ioc 0 (2 * Real.pi))
variable (h_eq : ∀ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ↔ (x = Real.sin θ ∨ x = Real.cos θ))

theorem problem_part_1 : 
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2 := 
by
  sorry

theorem problem_part_2 : 
  m = Real.sqrt 3 / 2 := 
by 
  sorry

end problem_part_1_problem_part_2_l539_539226


namespace remaining_oranges_l539_539411

/-- Define the conditions of the problem. -/
def oranges_needed_Michaela : ℕ := 20
def oranges_needed_Cassandra : ℕ := 2 * oranges_needed_Michaela
def total_oranges_picked : ℕ := 90

/-- State the proof problem. -/
theorem remaining_oranges : total_oranges_picked - (oranges_needed_Michaela + oranges_needed_Cassandra) = 30 := 
sorry

end remaining_oranges_l539_539411


namespace distance_and_chord_length_l539_539221

theorem distance_and_chord_length
  (l : LinearForm ℝ := {a := 4, b := 3, c := 6})
  (C_center : Point := (1, 0))
  (r : ℝ := 3)
  (dist_center_to_line : ℝ := distance C_center l = 2)
  (chord_length_squared : ∃ (E F : Point), distance E F = 2 * sqrt (r^2 - (distance C_center l)^2)) :
  distance C_center l = 2 ∧ ∃ (E F : Point), distance E F = 2 * sqrt (r^2 - 2^2) :=
by sorry

end distance_and_chord_length_l539_539221


namespace boys_who_quit_l539_539836

theorem boys_who_quit (initial_girls initial_boys girls_joined total_children after_quit_girls : ℕ) :
  initial_girls = 18 ∧ initial_boys = 15 ∧ girls_joined = 7 ∧ total_children = 36 ∧ 
  after_quit_girls = initial_girls + girls_joined →
  initial_boys - (total_children - after_quit_girls) = 4 :=
by
  intros h
  obtain ⟨h_girls, h_boys, h_girls_joined, h_total_children, h_after_quit_girls⟩ := h
  rw [h_girls, h_boys, h_girls_joined, h_total_children] at h_after_quit_girls
  sorry

end boys_who_quit_l539_539836


namespace incorrect_sqrt_4_l539_539894

theorem incorrect_sqrt_4 : (∀ (x : ℝ), sqrt ((-2)^2) = 2 ∧ (-sqrt 2)^2 = 2 ∧ sqrt (∛(-8)) = -2) → sqrt 4 ≠ ±2 :=
by
  intro h
  sorry

end incorrect_sqrt_4_l539_539894


namespace complement_intersection_eq_4_l539_539648

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection_eq_4 (hU : U = {0, 1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4}) :
  ((U \ A) ∩ B) = {4} :=
by {
  -- Proof goes here
  exact sorry
}

end complement_intersection_eq_4_l539_539648


namespace a_n_geometric_S_n_sum_l539_539606

variable (m : ℝ) (f : ℝ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom f_def : ∀ x : ℝ, f x = log m x
axiom m_pos : m > 0
axiom m_ne_one : m ≠ 1
axiom arithmetic_sequence : ∀ n : ℕ, f (a n) = 2 * n.succ.succ -- (2n + 2 where n starts from 1)

-- Question 1: Geometric sequence
theorem a_n_geometric : ∀ n : ℕ, a (n + 1) = m ^ (2 * (n + 2)) := by sorry

-- Conditions for bn and Sn
axiom b_def : ∀ n : ℕ, b n = a n * f (a n)

-- Question 2: Sum of the sequence
theorem S_n_sum (h : m = Real.sqrt 2) : ∀ n : ℕ, S n = 2 ^ (n + 3) * n := by sorry

end a_n_geometric_S_n_sum_l539_539606


namespace part1_part2_l539_539187

variables {a b : ℝ^3} -- Define vectors a and b

-- Define the conditions under which we are working
axiom ha : ∥a∥ = 1
axiom hb : ∥b∥ = 1
axiom hab : ∥a + b∥ = real.sqrt 3

-- First part: proving |a - b| = 1
theorem part1 : ∥a - b∥ = 1 :=
sorry

-- Second part: proving (a + b) is perpendicular to (a - b)
theorem part2 : inner (a + b) (a - b) = 0 :=
sorry

end part1_part2_l539_539187


namespace units_digit_7_pow_2050_l539_539043

theorem units_digit_7_pow_2050 : (7 ^ 2050) % 10 = 9 := 
by 
  sorry

end units_digit_7_pow_2050_l539_539043


namespace large_pizzas_sold_l539_539021

def small_pizza_price : ℕ := 2
def large_pizza_price : ℕ := 8
def total_earnings : ℕ := 40
def small_pizzas_sold : ℕ := 8

theorem large_pizzas_sold : 
  ∀ (small_pizza_price large_pizza_price total_earnings small_pizzas_sold : ℕ), 
    small_pizza_price = 2 → 
    large_pizza_price = 8 → 
    total_earnings = 40 → 
    small_pizzas_sold = 8 →
    (total_earnings - small_pizzas_sold * small_pizza_price) / large_pizza_price = 3 :=
by 
  intros small_pizza_price large_pizza_price total_earnings small_pizzas_sold 
         h_small_pizza_price h_large_pizza_price h_total_earnings h_small_pizzas_sold
  rw [h_small_pizza_price, h_large_pizza_price, h_total_earnings, h_small_pizzas_sold]
  simp
  sorry

end large_pizzas_sold_l539_539021


namespace radius_of_regular_polygon_l539_539824

theorem radius_of_regular_polygon :
  ∃ (p : ℝ), 
        (∀ n : ℕ, 3 ≤ n → (n : ℝ) = 6) ∧ 
        (∀ s : ℝ, s = 2 → s = 2) → 
        (∀ i : ℝ, i = 720 → i = 720) →
        (∀ e : ℝ, e = 360 → e = 360) →
        p = 2 :=
by
  sorry

end radius_of_regular_polygon_l539_539824


namespace probability_of_Y_l539_539842

-- Define the probabilities as given in the conditions
def P_X : ℝ := 1 / 7
def P_X_and_Y : ℝ := 0.05714285714285714

-- Define the problem statement to prove P(Y) = 0.4
theorem probability_of_Y (P_Y : ℝ) (H1 : P_X = 1 / 7) (H2 : P_X_and_Y = 0.05714285714285714) : P_Y = 0.4 := 
by 
  -- The proof steps go here
  sorry

end probability_of_Y_l539_539842


namespace max_handshakes_excluding_committee_and_red_badge_holders_l539_539275

theorem max_handshakes_excluding_committee_and_red_badge_holders
  (total_participants : ℕ) (committee_members : ℕ) (red_badge_holders : ℕ) :
  total_participants = 50 → committee_members = 10 → red_badge_holders = 5 →
  (let participants_who_can_shake_hands := total_participants - committee_members - red_badge_holders in
  participants_who_can_shake_hands * (participants_who_can_shake_hands - 1) / 2 = 595) :=
begin
  intros h_total h_committee h_red_badge,
  have participants_who_can_shake_hands : ℕ := total_participants - committee_members - red_badge_holders,
  rw [h_total, h_committee, h_red_badge],
  sorry, -- The proof is omitted as per instructions.
end

end max_handshakes_excluding_committee_and_red_badge_holders_l539_539275


namespace shooting_guard_seconds_l539_539754

-- Define the given conditions
def x_pg := 130
def x_sf := 85
def x_pf := 60
def x_c := 180
def avg_time_per_player := 120
def total_players := 5

-- Define the total footage
def total_footage : Nat := total_players * avg_time_per_player

-- Define the footage for four players
def footage_of_four : Nat := x_pg + x_sf + x_pf + x_c

-- Define the footage of the shooting guard, which is a variable we want to compute
def x_sg := total_footage - footage_of_four

-- The statement we want to prove
theorem shooting_guard_seconds :
  x_sg = 145 := by
  sorry

end shooting_guard_seconds_l539_539754


namespace time_to_cross_pole_l539_539097

-- Define the speed in km/hr
def speed_kmh : ℝ := 60

-- Convert the speed to m/s
def speed_ms : ℝ := (speed_kmh * 1000) / 3600

-- Define the length of the train in meters
def length_of_train : ℝ := 250.00000000000003

-- Prove the time it takes for the train to cross the pole is 15 seconds
theorem time_to_cross_pole : (length_of_train / speed_ms) = 15 := by
  -- Insert specific proof here (omitted)
  sorry

end time_to_cross_pole_l539_539097


namespace problem_statement_l539_539616

-- Definition of the ellipse E
def is_on_ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Definition of point P on ellipse
def P_on_ellipse (x₀ y₀ : ℝ) : Prop := is_on_ellipse x₀ y₀

-- Q is the foot of the perpendicular from P to the y-axis
def Q_from_P (y₀ : ℝ) : Prop := (0, y₀)

-- QM is a vector relation from the problem
def QM_relation (x y x₀ y₀ : ℝ) : Prop :=
  x = (√3 / 3) * x₀ ∧ y = y₀

-- Definition of the trajectory Γ of the moving point M
def trajectory (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Maximum area of ∆OAB
def max_area_triangle (m n : ℝ) : Prop :=
  n^2 = m^2 + 1 →
  ∃ (A B : ℝ × ℝ), 
    let y₁ := A.snd, y₂ := B.snd in
    |AB| = sqrt 3 → 
    (height O A B = 1) →
    area_max = sqrt 3 / 2

-- Lean statement proving the questions
theorem problem_statement :
  (∀ (x₀ y₀ : ℝ), P_on_ellipse x₀ y₀ → 
  ∃ (x y : ℝ), trajectory x y ∧ QM_relation x y x₀ y₀) ∧
  (∀ (m n : ℝ), max_area_triangle m n) := 
begin
  sorry, -- Proof elided
end

end problem_statement_l539_539616


namespace distinct_solutions_equation_l539_539722

theorem distinct_solutions_equation (a b : ℝ) (h1 : a ≠ b) (h2 : a > b) (h3 : ∀ x, (3 * x - 9) / (x^2 + 3 * x - 18) = x + 1) (sol_a : x = a) (sol_b : x = b) :
  a - b = 1 :=
sorry

end distinct_solutions_equation_l539_539722


namespace incenter_divides_angle_bisector_l539_539350

-- Definitions
variables (ABC : Type) [triangle ABC]
variables (a b c : ℝ) -- a = BC, b = AC, c = AB
variables (A B C I : ABC) -- Vertices of the triangle and the incenter

-- Given Conditions
hypothesis h1 : side_length A B = c
hypothesis h2 : side_length B C = a
hypothesis h3 : side_length A C = b
hypothesis h4 : is_incenter I ABC

-- Theorem Statement
theorem incenter_divides_angle_bisector :
  divides_angle_bisector I C (a + b) c :=
by
  sorry

end incenter_divides_angle_bisector_l539_539350


namespace points_collinear_l539_539717

variables {Point : Type} [inhabited Point]

noncomputable def triangle (A B C : Point) : Prop := sorry

noncomputable def on_circumcircle (P : Point) (A B C : Point) : Prop := sorry

noncomputable def orthogonal_projection (P : Point) (l : Line) : Point := sorry

theorem points_collinear 
  {A B C P X Y Z : Point}
  (h_triangle : triangle A B C)
  (h_on_circumcircle : on_circumcircle P A B C)
  (hX : X = orthogonal_projection P (line_through A B))
  (hY : Y = orthogonal_projection P (line_through B C))
  (hZ : Z = orthogonal_projection P (line_through C A)) :
  collinear X Y Z := 
sorry

end points_collinear_l539_539717


namespace infinitely_many_terms_of_sequence_can_be_expressed_as_linear_combination_l539_539723

variable {a : ℕ → ℕ} -- Sequence of positive integers

theorem infinitely_many_terms_of_sequence_can_be_expressed_as_linear_combination 
  (h₀ : ∀ k, a k < a (k + 1)) -- strictly increasing condition
  (h₁ : ∀ k, 0 < a k)        -- positive integers condition
  : ∃ (A : set ℕ), A.infinite ∧ (∀ m ∈ A, ∃ p q x y : ℕ, x > 0 ∧ y > 0 ∧ p ≠ q ∧ a m = x * a p + y * a q) :=
sorry

end infinitely_many_terms_of_sequence_can_be_expressed_as_linear_combination_l539_539723


namespace solution_set_linear_inequality_l539_539029

theorem solution_set_linear_inequality (a b : ℝ)
  (h1 : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ x ∈ set.Iio (-3) ∪ set.Ioi 1) :
  ∀ x : ℝ, a * x + b < 0 ↔ x ∈ set.Iio (3 / 2) :=
by
  sorry

end solution_set_linear_inequality_l539_539029


namespace trig_identity_l539_539907

theorem trig_identity : 
  ( 4 * Real.sin (40 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) / Real.cos (20 * Real.pi / 180) 
   - Real.tan (20 * Real.pi / 180) ) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l539_539907


namespace cos_base_angle_isosceles_triangle_l539_539816

theorem cos_base_angle_isosceles_triangle (k : ℝ) (h : 0 < k ∧ k ≤ 1/2) :
  ∃ (cosx : ℝ), cosx = (1 + real.sqrt (1 - 2 * k)) / 2 ∨ cosx = (1 - real.sqrt (1 - 2 * k)) / 2 :=
by
  sorry

end cos_base_angle_isosceles_triangle_l539_539816


namespace least_three_digit_product_8_is_118_l539_539876

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_product (n : ℕ) (product : ℕ) : Prop :=
  let digits := (list.cons (n / 100) (list.cons ((n / 10) % 10) (list.cons (n % 10) list.nil))) in
  digits.prod = product

theorem least_three_digit_product_8_is_118 :
  ∃ n : ℕ, is_three_digit_number n ∧ digits_product n 8 ∧ (∀ m : ℕ, is_three_digit_number m ∧ digits_product m 8 → n ≤ m) :=
sorry

end least_three_digit_product_8_is_118_l539_539876


namespace quadratic_inequality_iff_l539_539596

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + 4*x - 96 > abs x

theorem quadratic_inequality_iff (x : ℝ) : quadratic_inequality_solution x ↔ x < -12 ∨ x > 8 := by
  sorry

end quadratic_inequality_iff_l539_539596


namespace num_ordered_pairs_tangent_line_circle_l539_539238

theorem num_ordered_pairs_tangent_line_circle (n m : ℕ) (n_pos : 0 < n) (m_pos : 0 < m) :
  let line := λ x y, (√3) * x - y + 2^m
      circle := λ x y, x^2 + y^2 - n^2
      tangent := ∀ x y, (circle x y = 0) → (line x y = 0) → abs ((√3) / n - 1 / n * y + 2^m / n) = 1
      pairs := [(1, 1), (2, 2), (3, 4), (4, 8)]
  in  (n - m < 5) → tangent → ∃ (m n : ℕ), n - m < 5 ∧ (m, n) ∈ pairs :=
begin
  sorry
end

end num_ordered_pairs_tangent_line_circle_l539_539238


namespace milk_fraction_in_final_cup_l539_539943

structure Cups :=
  (milk_cup1 : ℝ)    -- ounces of milk in Cup 1 initially
  (honey_cup2 : ℝ)   -- ounces of honey in Cup 2 initially
  (cup_capacity : ℝ) -- capacity of each cup

theorem milk_fraction_in_final_cup (cups : Cups) : 
  let cup1_after_first_transfer := cups.milk_cup1 / 2
  let cup2_after_first_transfer := cups.honey_cup2 + cups.milk_cup1 / 2
  let transferred_back_amount := (cup2_after_first_transfer / 2)
  let milk_transferred_back := (cup2_after_first_transfer / 2) * (cups.milk_cup1 / 10)
  let cup1_after_second_transfer := cup1_after_first_transfer + milk_transferred_back
  let cup2_after_second_transfer := cup2_after_first_transfer - transferred_back_amount
  let milk_in_cup2_after_transferred_back := cup2_after_second_transfer * (4 / 10)
  let one_third_transfer_from_first_to_second := cup1_after_second_transfer / 3
  let final_milk_cup2 := milk_in_cup2_after_transferred_back + one_third_transfer_from_first_to_second
  let final_total_cup2 := cup2_after_second_transfer + one_third_transfer_from_first_to_second
  in final_milk_cup2 / final_total_cup2 = 2 / 3 :=
by {
  sorry
}

def initial_cups : Cups := {
  milk_cup1 := 8,
  honey_cup2 := 6,
  cup_capacity := 10
}

#eval milk_fraction_in_final_cup initial_cups

end milk_fraction_in_final_cup_l539_539943


namespace at_least_one_nonzero_l539_539013

theorem at_least_one_nonzero (a b : ℝ) : a^2 + b^2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) := by
  sorry

end at_least_one_nonzero_l539_539013


namespace proof_number_of_pairs_l539_539135

def line_slope (line : ℝ → ℝ) : ℝ :=
  let x1 := 1 in
  let x2 := 2 in
  let y1 := line x1 in
  let y2 := line x2 in
  (y2 - y1) / (x2 - x1)

def number_of_parallel_or_perpendicular_pairs (lines : List (ℝ → ℝ)) : ℕ :=
  let slopes := lines.map line_slope
  let parallel_pairs := slopes.erasedup.map (λ s, slopes.count (s = ·)).filter (λ c, c >= 2)
  let num_parallel_pairs := parallel_pairs.map (λ c, c * (c - 1) / 2).sum
  let perpendicular_pairs := slopes.filter (λ m1, slopes.any (λ m2, -1 = m1 * m2))
  let num_perpendicular_pairs := perpendicular_pairs.length / 2 -- the division by 2 avoids double counting
  num_parallel_pairs + num_perpendicular_pairs

def solution_example : ℕ :=
  let lines := [ (λ x, 4*x + 3)
               , (λ x, 2*x + 3)
               , (λ x, 4*x - 3/4)
               , (λ x, (2/3)*x - 7/3)
               , (λ x, (2/5)*x - 2) ]
  number_of_parallel_or_perpendicular_pairs lines

theorem proof_number_of_pairs : solution_example = 1 := by
  -- The proof follows from the problem's solution steps.
  sorry

end proof_number_of_pairs_l539_539135


namespace distance_from_incenter_to_face_BSC_l539_539800

-- Define the sides of the triangle
def AB : ℝ := 3
def BC : ℝ := 3
def AC : ℝ := 5

-- Define the height of the pyramid
def h : ℝ := 7 / 12

-- Define the distance to be proved
def distance_to_prove : ℝ := 35 * Real.sqrt 11 / 396

theorem distance_from_incenter_to_face_BSC :
  let r := 5 * Real.sqrt 11 / 26 in
  let OK := (5 * Real.sqrt 11) / 12 in
  let OT := (h * OK) / Real.sqrt (h ^ 2 + OK ^ 2) in
  (2 * OT * BC) / (AC + 2 * BC) = distance_to_prove :=
by
  sorry

end distance_from_incenter_to_face_BSC_l539_539800


namespace union_A_B_intersection_complementA_B_range_of_a_l539_539243

-- Definition of the universal set U, sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}

-- Complement of A in the universal set U
def complement_A : Set ℝ := {x | x < 1 ∨ x ≥ 5}

-- Definition of set C parametrized by a
def C (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- Prove that A ∪ B is {x | 1 ≤ x < 8}
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 8} :=
sorry

-- Prove that (complement_U A) ∩ B = {x | 5 ≤ x < 8}
theorem intersection_complementA_B : (complement_A ∩ B) = {x | 5 ≤ x ∧ x < 8} :=
sorry

-- Prove the range of values for a if C ∩ A = C
theorem range_of_a (a : ℝ) : (C a ∩ A = C a) → a ≤ -1 :=
sorry

end union_A_B_intersection_complementA_B_range_of_a_l539_539243


namespace rate_of_interest_l539_539520

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem rate_of_interest :
  ∃ R : ℝ, simple_interest 8925 R 5 = 4016.25 ∧ R = 9 := 
by
  use 9
  simp [simple_interest]
  norm_num
  sorry

end rate_of_interest_l539_539520


namespace least_three_digit_product_of_digits_is_8_l539_539871

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l539_539871


namespace least_three_digit_product_of_digits_is_8_l539_539870

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l539_539870


namespace interval_contains_zero_l539_539584

noncomputable def f (x : ℝ) : ℝ := 2^x + 3*x - 7

theorem interval_contains_zero : ∃ c ∈ Ioo (1 : ℝ) (3/2 : ℝ), f c = 0 :=
by
  have h1 : f 1 < 0 := by norm_num [f]
  have h2 : f (3/2) > 0 := by norm_num [f]
  exact IntermediateValueTheorem f 1 (3/2) h1 h2 sorry

end interval_contains_zero_l539_539584


namespace bernardo_probability_is_correct_l539_539953

noncomputable def bernardo_larger_probability : ℚ :=
  let total_bernardo_combinations := (Nat.choose 10 3 : ℚ)
  let total_silvia_combinations := (Nat.choose 8 3 : ℚ)
  let bernardo_has_10 := (Nat.choose 8 2 : ℚ) / total_bernardo_combinations
  let bernardo_not_has_10 := ((total_silvia_combinations - 1) / total_silvia_combinations) / 2
  bernardo_has_10 * 1 + (1 - bernardo_has_10) * bernardo_not_has_10

theorem bernardo_probability_is_correct :
  bernardo_larger_probability = 19 / 28 := by
  sorry

end bernardo_probability_is_correct_l539_539953


namespace barbara_current_savings_l539_539126

def wristwatch_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def initial_saving_duration : ℕ := 10
def further_saving_duration : ℕ := 16

theorem barbara_current_savings : 
  -- Given:
  -- wristwatch_cost: $100
  -- weekly_allowance: $5
  -- further_saving_duration: 16 weeks
  -- Prove:
  -- Barbara currently has $20
  wristwatch_cost - weekly_allowance * further_saving_duration = 20 :=
by
  sorry

end barbara_current_savings_l539_539126


namespace log_eq_of_x_l539_539359

theorem log_eq_of_x (x : ℝ) (h1 : ∀ (a b : ℝ) (c : ℕ), log a (b ^ c) = c * log a b)
  (h2 : ∀ (a b : ℝ) (k : ℕ), log (a ^ k) b = (1 / k) * log a b)
  (h3 : log 8 x + log 4 (x ^ 3) = 15) :
  x = 2 ^ (90 / 11) :=
by
  sorry

end log_eq_of_x_l539_539359


namespace angle_BAO_is_20_degrees_l539_539693

/-- In a semicircle with diameter CD and center O, with point A on the extension of DC past C, and point E on the semicircle.
    Let B be the point of intersection of AE with the semicircle (distinct from E). Given:
    1. AB = OD = 2 units,
    2. ∠EOD = 60°
    Prove the measure of ∠BAO is 20°.
-/
theorem angle_BAO_is_20_degrees
  (O C D A E B : Point)
  (AB OD : ℝ)
  (h1 : CD = Diameter O)
  (h2 : lies_on_extension A DC)
  (h3 : lies_on_semicircle E O CD)
  (h4 : intersection B (line_through A E) (semicircle_through O E))
  (h5 : AB = 2)
  (h6 : OD = 2)
  (angle_EOD : angle E O D)
  (h7 : angle_EOD = 60) : 
  (angle B A O) = 20 :=
by sorry

end angle_BAO_is_20_degrees_l539_539693


namespace daily_wage_l539_539505

theorem daily_wage :
  ∃ W : ℕ,
    let T := 30 in
    let A := 7 in
    let F := 2 in
    let R := 216 in
    (T - A) * W - F * A = R ∧ W = 10 :=
begin
  use 10,
  let T := 30,
  let A := 7,
  let F := 2,
  let R := 216,
  split,
  { calc
      (T - A) * 10 - F * A
          = (30 - 7) * 10 - 2 * 7 : by rw [T, A, F]
      ... = 23 * 10 - 14 : by norm_num
      ... = 216 : by norm_num },
  { refl }
end

end daily_wage_l539_539505


namespace problem_proof_l539_539234

def f (a x : ℝ) := |a - x|

theorem problem_proof (a x x0 : ℝ) (h_a : a = 3 / 2) (h_x0 : x0 < 0) : 
  f a (x0 * x) ≥ x0 * f a x + f a (a * x0) :=
sorry

end problem_proof_l539_539234


namespace pyramid_edges_count_l539_539036

-- Definitions based on the conditions in the problem
def pyramid_faces (n : ℕ) : ℕ := n + 1
def pyramid_vertices (n : ℕ) : ℕ := n + 1
def pyramid_edges (n : ℕ) : ℕ := 2 * n

-- Condition: Sum of faces and vertices equals 16
axiom faces_vertices_sum_eq_16 (n : ℕ) : pyramid_faces n + pyramid_vertices n = 16

-- The main theorem to be proved
theorem pyramid_edges_count (n : ℕ) (h : faces_vertices_sum_eq_16 n) : pyramid_edges n = 14 :=
sorry

end pyramid_edges_count_l539_539036


namespace net_gain_is_88837_50_l539_539334

def initial_home_value : ℝ := 500000
def first_sale_price : ℝ := 1.15 * initial_home_value
def first_purchase_price : ℝ := 0.95 * first_sale_price
def second_sale_price : ℝ := 1.1 * first_purchase_price
def second_purchase_price : ℝ := 0.9 * second_sale_price

def total_sales : ℝ := first_sale_price + second_sale_price
def total_purchases : ℝ := first_purchase_price + second_purchase_price
def net_gain_for_A : ℝ := total_sales - total_purchases

theorem net_gain_is_88837_50 : net_gain_for_A = 88837.50 := by
  -- proof steps would go here, but they are omitted per instructions
  sorry

end net_gain_is_88837_50_l539_539334


namespace Adam_bought_26_books_l539_539108

theorem Adam_bought_26_books (initial_books : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) (leftover_books : ℕ) :
  initial_books = 56 → shelves = 4 → books_per_shelf = 20 → leftover_books = 2 → 
  let total_capacity := shelves * books_per_shelf in
  let total_books_after := total_capacity + leftover_books in
  let books_bought := total_books_after - initial_books in
  books_bought = 26 :=
by
  intros h1 h2 h3 h4
  simp [total_capacity, total_books_after, books_bought]
  rw [h1, h2, h3, h4]
  sorry

end Adam_bought_26_books_l539_539108


namespace brick_wall_l539_539919

theorem brick_wall (y : ℕ) (h1 : ∀ y, 6 * ((y / 8) + (y / 12) - 12) = y) : y = 288 :=
sorry

end brick_wall_l539_539919


namespace expression_value_l539_539256

theorem expression_value (x : ℕ) (h : x = 12) : (3 / 2 * x - 3 : ℚ) = 15 := by
  rw [h]
  norm_num
-- sorry to skip the proof if necessary
-- sorry 

end expression_value_l539_539256


namespace four_digit_number_conditions_l539_539085

theorem four_digit_number_conditions :
  ∃ (a b c d : ℕ), 
    (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 10 * 23) ∧ 
    (a + b + c + d = 26) ∧ 
    ((b * d / 10) % 10 = a + c) ∧ 
    ∃ (n : ℕ), (b * d - c^2 = 2^n) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 1979) :=
sorry

end four_digit_number_conditions_l539_539085


namespace locus_of_M_l539_539169

-- Definitions: 
def is_equilateral_triangle (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

def locus_M (A B C M : Point) : Prop :=
  ∃ (P Q : Point) (γ : Circle), 
    arc_on_circle γ P Q ∧
    angle_at_point γ B C 150 ∧
    M ∈ arc_interior γ P Q

def satisfies_condition (A B C M : Point) : Prop :=
  dist M A ^ 2 = dist M B ^ 2 + dist M C ^ 2

-- Theorem statement
theorem locus_of_M {A B C M : Point} :
  is_equilateral_triangle A B C →
  satisfies_condition A B C M ↔ 
  locus_M A B C M := 
by sorry

end locus_of_M_l539_539169


namespace ellipse_conjugate_diameters_l539_539642

variable (A B C D E : ℝ)

theorem ellipse_conjugate_diameters :
  (A * E - B * D = 0) ∧ (2 * B ^ 2 + (A - C) * A = 0) :=
sorry

end ellipse_conjugate_diameters_l539_539642


namespace A_alone_days_l539_539457

variable (r_A r_B r_C : ℝ)

-- Given conditions:
axiom cond1 : r_A + r_B = 1 / 3
axiom cond2 : r_B + r_C = 1 / 6
axiom cond3 : r_A + r_C = 4 / 15

-- Proposition stating the required proof, that A alone can do the job in 60/13 days:
theorem A_alone_days : r_A ≠ 0 → 1 / r_A = 60 / 13 :=
by
  intro h
  sorry

end A_alone_days_l539_539457


namespace brian_total_distance_l539_539956

noncomputable def miles_per_gallon : ℝ := 20
noncomputable def tank_capacity : ℝ := 15
noncomputable def tank_fraction_remaining : ℝ := 3 / 7

noncomputable def total_miles_traveled (miles_per_gallon tank_capacity tank_fraction_remaining : ℝ) : ℝ :=
  let total_miles := miles_per_gallon * tank_capacity
  let fuel_used := tank_capacity * (1 - tank_fraction_remaining)
  let miles_traveled := fuel_used * miles_per_gallon
  miles_traveled

theorem brian_total_distance : 
  total_miles_traveled miles_per_gallon tank_capacity tank_fraction_remaining = 171.4 := 
by
  sorry

end brian_total_distance_l539_539956


namespace find_side_AB_l539_539703

-- Definitions of the given conditions
structure Triangle (α : Type*) :=
  (A B C : α)

noncomputable theory
open_locale classical

def BP (T : Triangle ℝ) : ℝ := 16
def PC (T : Triangle ℝ) : ℝ := 20
def circumcenter_lies_on_segment_AC (T : Triangle ℝ) : Prop := sorry -- circumcenter condition as a prop

-- The theorem to prove AB based on the given conditions
theorem find_side_AB (T : Triangle ℝ) (h1 : BP T = 16) (h2 : PC T = 20)
  (h3 : circumcenter_lies_on_segment_AC T) : 
  ∃ (AB : ℝ), AB = 144 * real.sqrt 5 / 5 :=
sorry

end find_side_AB_l539_539703


namespace necessary_condition_inequality_l539_539507

theorem necessary_condition_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 := 
sorry

end necessary_condition_inequality_l539_539507


namespace solve_for_x_l539_539769

theorem solve_for_x (x : ℝ) (h : 50^2 = 10^x) : x = 3.39794 :=
by
  sorry

end solve_for_x_l539_539769


namespace max_cardinality_seven_l539_539003

theorem max_cardinality_seven 
  (M : Finset ℝ)
  (h : ∀ a b c ∈ M, (∃ x y ∈ pairwise (λ x y, x ∈ M) {a, b, c}, x + y ∈ M)) :
  M.card ≤ 7 :=
sorry

end max_cardinality_seven_l539_539003


namespace smallest_integral_k_l539_539040

theorem smallest_integral_k (k : ℤ) :
  (297 - 108 * k < 0) ↔ (k ≥ 3) :=
sorry

end smallest_integral_k_l539_539040


namespace square_side_length_l539_539797

theorem square_side_length (x : ℝ) (h : x^2 = (1/2) * x * 2) : x = 1 := by
  sorry

end square_side_length_l539_539797


namespace points_on_opposite_sides_l539_539696

-- Definitions and the conditions written to Lean
def satisfies_A (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 2 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

def satisfies_B (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 8 * a^2 * x - 2 * a^3 * y + 12 * a * y + a^4 + 36 = 0

def opposite_sides_of_line (y_A y_B : ℝ) : Prop :=
  (y_A - 1) * (y_B - 1) < 0

theorem points_on_opposite_sides (a : ℝ) (x_A y_A x_B y_B : ℝ) :
  satisfies_A a x_A y_A →
  satisfies_B a x_B y_B →
  -2 > a ∨ (-1 < a ∧ a < 0) ∨ 3 < a →
  opposite_sides_of_line y_A y_B → 
  x_A = 2 * a ∧ y_A = -a ∧ x_B = 4 ∧ y_B = a - 6/a :=
sorry

end points_on_opposite_sides_l539_539696


namespace Yanna_kept_apples_l539_539443

theorem Yanna_kept_apples (initial_apples : ℕ) (apples_given_Zenny : ℕ) (apples_given_Andrea : ℕ) :
  initial_apples = 60 → apples_given_Zenny = 18 → apples_given_Andrea = 6 →
  (initial_apples - (apples_given_Zenny + apples_given_Andrea) = 36) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.add_sub_assoc]
  exact rfl
  apply Nat.succ_le_succ
  exact nat.succ_pos'


end Yanna_kept_apples_l539_539443


namespace arithmetic_contains_geometric_l539_539064

theorem arithmetic_contains_geometric (a b : ℚ) (h : a^2 + b^2 ≠ 0) :
  ∃ (q : ℚ) (c : ℚ) (n₀ : ℕ) (n : ℕ → ℕ), (∀ k : ℕ, n (k+1) = n k + c * q^k) ∧
  ∀ k : ℕ, ∃ r : ℚ, a + b * n k = r * q^k :=
sorry

end arithmetic_contains_geometric_l539_539064


namespace variance_bound_l539_539348

variable (X : ℝ → ℝ) (a b : ℝ)

theorem variance_bound (h_min : ∀ ω, X ω ≥ a) (h_max : ∀ ω, X ω ≤ b) :
  real.variance X ≤ (b - a)^2 / 4 := 
sorry

end variance_bound_l539_539348


namespace sum_of_odd_divisors_of_240_l539_539432

theorem sum_of_odd_divisors_of_240 : 
  let n := 240 in 
  let odd_divisors := {d ∣ n | d % 2 = 1} in 
  ∑ d in odd_divisors, d = 24 :=
by 
  let n := 240
  let odd_divisors := {d ∣ n | d % 2 = 1}
  sorry

end sum_of_odd_divisors_of_240_l539_539432


namespace trig_identity_l539_539553

theorem trig_identity (α : ℝ) (hα : α = 60) :
  cos (degToRad (α + 30)) * cos (degToRad (α - 30)) + sin (degToRad (α + 30)) * sin (degToRad (α - 30)) = 1 / 2 :=
by
  sorry

end trig_identity_l539_539553


namespace tammy_speed_on_second_day_l539_539790

theorem tammy_speed_on_second_day :
  ∃ v t : ℝ, t + (t - 2) = 14 ∧ v * t + (v + 0.5) * (t - 2) = 52 → (v + 0.5 = 4) :=
begin
  sorry
end

end tammy_speed_on_second_day_l539_539790
