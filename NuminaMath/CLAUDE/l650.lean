import Mathlib

namespace NUMINAMATH_CALUDE_inverse_composition_l650_65003

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Assumption: f and g are bijective (to ensure inverses exist)
variable (hf : Function.Bijective f)
variable (hg : Function.Bijective g)

-- Define the relationship between f_inv and g
axiom relation (x : ℝ) : f_inv (g x) = 4 * x - 2

-- Theorem to prove
theorem inverse_composition :
  g_inv (f 5) = 7/4 :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_l650_65003


namespace NUMINAMATH_CALUDE_constant_segments_am_plus_bn_equals_11_am_equals_bn_l650_65017

-- Define the points on the number line
def A (t : ℝ) : ℝ := -1 + 2*t
def M (t : ℝ) : ℝ := t
def N (t : ℝ) : ℝ := t + 2
def B (t : ℝ) : ℝ := 11 - t

-- Theorem for part 1
theorem constant_segments :
  ∀ x t : ℝ, abs (B t - A t) = 12 ∧ abs (N t - M t) = 2 :=
sorry

-- Theorem for part 2, question 1
theorem am_plus_bn_equals_11 :
  ∃ t : ℝ, abs (M t - A t) + abs (B t - N t) = 11 ∧ t = 9.5 :=
sorry

-- Theorem for part 2, question 2
theorem am_equals_bn :
  ∃ t₁ t₂ : ℝ, 
    abs (M t₁ - A t₁) = abs (B t₁ - N t₁) ∧
    abs (M t₂ - A t₂) = abs (B t₂ - N t₂) ∧
    t₁ = 10/3 ∧ t₂ = 8 :=
sorry

end NUMINAMATH_CALUDE_constant_segments_am_plus_bn_equals_11_am_equals_bn_l650_65017


namespace NUMINAMATH_CALUDE_two_digit_sum_problem_l650_65000

theorem two_digit_sum_problem :
  ∃! (x y z : ℕ), 
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    11 * x + 11 * y + 11 * z = 100 * x + 10 * y + z :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_problem_l650_65000


namespace NUMINAMATH_CALUDE_sum_is_linear_l650_65048

/-- The original parabola function -/
def original_parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

/-- The function f(x) derived from the original parabola -/
def f (a h k x : ℝ) : ℝ := -a * (x - h - 3)^2 - k

/-- The function g(x) derived from the original parabola -/
def g (a h k x : ℝ) : ℝ := a * (x - h + 7)^2 + k

/-- The sum of f(x) and g(x) -/
def f_plus_g (a h k x : ℝ) : ℝ := f a h k x + g a h k x

theorem sum_is_linear (a h k : ℝ) (ha : a ≠ 0) :
  ∃ m b : ℝ, (∀ x : ℝ, f_plus_g a h k x = m * x + b) ∧ m ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_sum_is_linear_l650_65048


namespace NUMINAMATH_CALUDE_inequality_solution_set_l650_65064

theorem inequality_solution_set :
  let S := {x : ℝ | (x + 5) * (3 - 2*x) ≤ 6}
  S = {x : ℝ | -9 ≤ x ∧ x ≤ 1/2} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l650_65064


namespace NUMINAMATH_CALUDE_prime_simultaneous_l650_65033

theorem prime_simultaneous (p : ℕ) : 
  Nat.Prime p ∧ Nat.Prime (8 * p^2 + 1) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_simultaneous_l650_65033


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_six_l650_65071

/-- Given a non-isosceles triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its perimeter is 6 under certain conditions. -/
theorem triangle_perimeter_is_six 
  (a b c A B C : ℝ) 
  (h_non_isosceles : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a * (Real.cos (C / 2))^2 + c * (Real.cos (A / 2))^2 = 3 * c / 2)
  (h_sines : 2 * Real.sin (A - B) + b * Real.sin B = a * Real.sin A) :
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_six_l650_65071


namespace NUMINAMATH_CALUDE_percentage_theorem_l650_65074

theorem percentage_theorem (x y : ℝ) (P : ℝ) 
  (h1 : 0.6 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.5 * x) : 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_theorem_l650_65074


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l650_65029

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 22

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 46

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 1012 := by
  sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l650_65029


namespace NUMINAMATH_CALUDE_ab_value_l650_65001

/-- Given that p and q are integers satisfying the equation and other conditions, prove that ab = 10^324 -/
theorem ab_value (p q : ℤ) (a b : ℝ) 
  (hp : p = Real.sqrt (Real.log a))
  (hq : q = Real.sqrt (Real.log b))
  (ha : a = 10^(p^2))
  (hb : b = 10^(q^2))
  (heq : 2*p + 2*q + (Real.log a)/2 + (Real.log b)/2 + p * q = 200) :
  a * b = 10^324 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l650_65001


namespace NUMINAMATH_CALUDE_cubes_fill_box_completely_l650_65038

def box_length : ℕ := 12
def box_width : ℕ := 6
def box_height : ℕ := 9
def cube_side : ℕ := 3

def cubes_per_length : ℕ := box_length / cube_side
def cubes_per_width : ℕ := box_width / cube_side
def cubes_per_height : ℕ := box_height / cube_side

def total_cubes : ℕ := cubes_per_length * cubes_per_width * cubes_per_height

def box_volume : ℕ := box_length * box_width * box_height
def cube_volume : ℕ := cube_side ^ 3
def total_cube_volume : ℕ := total_cubes * cube_volume

theorem cubes_fill_box_completely :
  total_cube_volume = box_volume := by sorry

end NUMINAMATH_CALUDE_cubes_fill_box_completely_l650_65038


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_after_2023_seconds_l650_65008

/-- Represents the number of positions in the cat's path -/
def cat_path_length : ℕ := 8

/-- Represents the number of positions in the mouse's path -/
def mouse_path_length : ℕ := 12

/-- Calculates the position of an object after a given number of seconds,
    given the length of its path -/
def position_after_time (path_length : ℕ) (time : ℕ) : ℕ :=
  time % path_length

/-- The main theorem stating the positions of the cat and mouse after 2023 seconds -/
theorem cat_and_mouse_positions_after_2023_seconds :
  position_after_time cat_path_length 2023 = 7 ∧
  position_after_time mouse_path_length 2023 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cat_and_mouse_positions_after_2023_seconds_l650_65008


namespace NUMINAMATH_CALUDE_chords_from_eight_points_l650_65069

/-- The number of chords that can be drawn from n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 8 points on a circle's circumference is 28 -/
theorem chords_from_eight_points : num_chords 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_chords_from_eight_points_l650_65069


namespace NUMINAMATH_CALUDE_max_sum_of_visible_faces_l650_65021

/-- Represents a standard six-sided die with opposite faces summing to 7 -/
structure Die :=
  (faces : Fin 6 → Nat)
  (valid : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

/-- The maximum sum of n visible faces on a standard die -/
def maxSum (n : Nat) : Nat :=
  if n ≤ 6 then
    List.sum (List.take n [6, 5, 4, 3, 2, 1])
  else
    0

/-- The configuration of visible faces for each die in the stack -/
def visibleFaces : List Nat := [5, 3, 3, 4, 4, 2]

/-- Theorem stating the maximum sum of visible faces in the given configuration -/
theorem max_sum_of_visible_faces :
  List.sum (List.map maxSum visibleFaces) = 89 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_visible_faces_l650_65021


namespace NUMINAMATH_CALUDE_cole_drive_time_to_work_l650_65070

/-- Proves that given the conditions of Cole's round trip, it took him 210 minutes to drive to work. -/
theorem cole_drive_time_to_work (speed_to_work : ℝ) (speed_to_home : ℝ) (total_time : ℝ) :
  speed_to_work = 75 →
  speed_to_home = 105 →
  total_time = 6 →
  (total_time * speed_to_work * speed_to_home) / (speed_to_work + speed_to_home) * (60 / speed_to_work) = 210 := by
  sorry

#check cole_drive_time_to_work

end NUMINAMATH_CALUDE_cole_drive_time_to_work_l650_65070


namespace NUMINAMATH_CALUDE_gcd_4004_10010_l650_65097

theorem gcd_4004_10010 : Nat.gcd 4004 10010 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4004_10010_l650_65097


namespace NUMINAMATH_CALUDE_larger_integer_value_l650_65045

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  a = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l650_65045


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l650_65026

/-- Given that a is a real number and i is the imaginary unit, 
    if (a+3i)/(1-2i) is a pure imaginary number, then a = 6 -/
theorem complex_fraction_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ b : ℝ, (a + 3 * Complex.I) / (1 - 2 * Complex.I) = b * Complex.I) →
  a = 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l650_65026


namespace NUMINAMATH_CALUDE_negative_difference_l650_65088

theorem negative_difference (m n : ℝ) : -(m - n) = -m + n := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l650_65088


namespace NUMINAMATH_CALUDE_problem_solution_l650_65096

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3*a

-- Define the conditions
def condition1 (a : ℝ) (m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ 1 < x ∧ x < m

def condition2 (a : ℝ) : Prop :=
  ∀ x, f a x > 0

def condition3 (a : ℝ) (k : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → a^(k+3) < a^(x^2-k*x) ∧ a^(x^2-k*x) < a^(k-3)

-- State the theorem
theorem problem_solution (a m k : ℝ) :
  condition1 a m →
  condition2 a →
  condition3 a k →
  (a = 1 ∧ m = 3) ∧
  (-1 < k ∧ k < -2 + Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l650_65096


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l650_65095

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 3 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l650_65095


namespace NUMINAMATH_CALUDE_mn_inequality_characterization_l650_65091

theorem mn_inequality_characterization :
  ∀ m n : ℕ+, 
    (1 ≤ m^n.val - n^m.val ∧ m^n.val - n^m.val ≤ m.val * n.val) ↔ 
    ((m ≥ 2 ∧ n = 1) ∨ (m = 2 ∧ n = 5) ∨ (m = 3 ∧ n = 2)) := by
  sorry

end NUMINAMATH_CALUDE_mn_inequality_characterization_l650_65091


namespace NUMINAMATH_CALUDE_expression_equality_l650_65015

theorem expression_equality (x : ℝ) : 
  (3*x + 1)^2 + 2*(3*x + 1)*(x - 3) + (x - 3)^2 = 16*x^2 - 16*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l650_65015


namespace NUMINAMATH_CALUDE_second_cat_brown_kittens_count_l650_65062

/-- The number of brown-eyed kittens the second cat has -/
def second_cat_brown_kittens : ℕ := sorry

/-- The total number of kittens from both cats -/
def total_kittens : ℕ := 14 + second_cat_brown_kittens

/-- The total number of blue-eyed kittens from both cats -/
def blue_eyed_kittens : ℕ := 7

/-- The percentage of blue-eyed kittens -/
def blue_eyed_percentage : ℚ := 35 / 100

theorem second_cat_brown_kittens_count : second_cat_brown_kittens = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_cat_brown_kittens_count_l650_65062


namespace NUMINAMATH_CALUDE_all_rationals_same_color_l650_65065

-- Define a color type
def Color := Nat

-- Define a coloring function
def coloring : ℚ → Color := sorry

-- Define the main theorem
theorem all_rationals_same_color (n : Nat) 
  (h : ∀ a b : ℚ, coloring a ≠ coloring b → 
       coloring ((a + b) / 2) ≠ coloring a ∧ 
       coloring ((a + b) / 2) ≠ coloring b) : 
  ∀ x y : ℚ, coloring x = coloring y := by sorry

end NUMINAMATH_CALUDE_all_rationals_same_color_l650_65065


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l650_65039

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to distribute 7 indistinguishable balls into 4 indistinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 11 := by sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l650_65039


namespace NUMINAMATH_CALUDE_adams_sandwiches_l650_65052

/-- The number of sandwiches Adam bought -/
def num_sandwiches : ℕ := 3

/-- The cost of each sandwich in dollars -/
def sandwich_cost : ℕ := 3

/-- The cost of the water bottle in dollars -/
def water_cost : ℕ := 2

/-- The total cost of Adam's shopping in dollars -/
def total_cost : ℕ := 11

theorem adams_sandwiches :
  num_sandwiches * sandwich_cost + water_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_adams_sandwiches_l650_65052


namespace NUMINAMATH_CALUDE_q_min_at_two_l650_65032

/-- The function q(x) defined as (x - 5)^2 + (x + 1)^2 - 6 -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

/-- Theorem stating that q(x) has a minimum value when x = 2 -/
theorem q_min_at_two : 
  ∀ x : ℝ, q 2 ≤ q x := by sorry

end NUMINAMATH_CALUDE_q_min_at_two_l650_65032


namespace NUMINAMATH_CALUDE_linear_function_monotonicity_and_inequality_l650_65035

variables (a b c : ℝ)

def f (x : ℝ) := a * x + b

theorem linear_function_monotonicity_and_inequality (a b c : ℝ) :
  (a > 0 → Monotone (f a b)) ∧
  (b^2 - 4*a*c < 0 → a^3 + a*b + c ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_monotonicity_and_inequality_l650_65035


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l650_65024

open Set

/-- The set of real numbers x where x³ - 2x > 0 -/
def S : Set ℝ := {x | x^3 - 2*x > 0}

/-- The set of real numbers x where |x + 1| > 3 -/
def T : Set ℝ := {x | |x + 1| > 3}

theorem not_sufficient_not_necessary : ¬(S ⊆ T) ∧ ¬(T ⊆ S) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l650_65024


namespace NUMINAMATH_CALUDE_inequality_proof_l650_65061

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l650_65061


namespace NUMINAMATH_CALUDE_proposition_A_sufficient_not_necessary_l650_65014

/-- Defines a geometric sequence of three real numbers -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b ^ 2 = a * c) ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

theorem proposition_A_sufficient_not_necessary :
  (∀ a b c : ℝ, b ^ 2 ≠ a * c → ¬ is_geometric_sequence a b c) ∧
  (∃ a b c : ℝ, ¬ is_geometric_sequence a b c ∧ b ^ 2 = a * c) :=
by sorry

end NUMINAMATH_CALUDE_proposition_A_sufficient_not_necessary_l650_65014


namespace NUMINAMATH_CALUDE_average_of_combined_results_l650_65075

theorem average_of_combined_results :
  let n₁ : ℕ := 40
  let avg₁ : ℚ := 30
  let n₂ : ℕ := 30
  let avg₂ : ℚ := 40
  let total_sum := n₁ * avg₁ + n₂ * avg₂
  let total_count := n₁ + n₂
  (total_sum / total_count : ℚ) = 2400 / 70 := by
sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l650_65075


namespace NUMINAMATH_CALUDE_combined_salaries_l650_65079

/-- The combined salaries of four employees given the salary of the fifth and the average of all five -/
theorem combined_salaries 
  (c_salary : ℕ) 
  (average_salary : ℕ) 
  (h1 : c_salary = 15000)
  (h2 : average_salary = 8800) :
  c_salary + 4 * average_salary - 5 * average_salary = 29000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l650_65079


namespace NUMINAMATH_CALUDE_m_range_l650_65012

-- Define the original statement
def original_statement (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1 < x ∧ x < m + 1) → (1 < x ∧ x < 2)

-- Define the converse statement
def converse_statement (m : ℝ) : Prop :=
  ∀ x : ℝ, (1 < x ∧ x < 2) → (m - 1 < x ∧ x < m + 1)

-- Theorem: If the converse statement is true, then m is in [1, 2]
theorem m_range (h : ∀ m : ℝ, converse_statement m) :
  ∀ m : ℝ, (converse_statement m) ↔ (1 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l650_65012


namespace NUMINAMATH_CALUDE_set_A_properties_l650_65010

-- Define set A
def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k ∧ k > 0}

-- Define the properties
theorem set_A_properties :
  (∀ a ∈ A, ∀ b : ℕ, b > 0 → b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a : ℕ, a > 1 → a ∉ A → ∃ b : ℕ, b > 0 ∧ b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
by
  sorry


end NUMINAMATH_CALUDE_set_A_properties_l650_65010


namespace NUMINAMATH_CALUDE_bell_size_ratio_l650_65076

theorem bell_size_ratio (first_bell : ℝ) (second_bell : ℝ) (third_bell : ℝ) 
  (h1 : first_bell = 50)
  (h2 : third_bell = 4 * second_bell)
  (h3 : first_bell + second_bell + third_bell = 550) :
  second_bell / first_bell = 2 := by
sorry

end NUMINAMATH_CALUDE_bell_size_ratio_l650_65076


namespace NUMINAMATH_CALUDE_temp_rise_negative_equals_decrease_l650_65013

/-- Represents a temperature change in degrees Celsius -/
structure TemperatureChange where
  value : ℝ
  unit : String

/-- Defines a temperature rise -/
def temperature_rise (t : ℝ) : TemperatureChange :=
  { value := t, unit := "°C" }

/-- Defines a temperature decrease -/
def temperature_decrease (t : ℝ) : TemperatureChange :=
  { value := t, unit := "°C" }

/-- Theorem stating that a temperature rise of -2°C is equivalent to a temperature decrease of 2°C -/
theorem temp_rise_negative_equals_decrease :
  temperature_rise (-2) = temperature_decrease 2 := by
  sorry

end NUMINAMATH_CALUDE_temp_rise_negative_equals_decrease_l650_65013


namespace NUMINAMATH_CALUDE_function_equation_solution_l650_65036

theorem function_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l650_65036


namespace NUMINAMATH_CALUDE_remainder_3_250_mod_11_l650_65002

theorem remainder_3_250_mod_11 : 3^250 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_250_mod_11_l650_65002


namespace NUMINAMATH_CALUDE_triangle_inequalities_l650_65077

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ -- semi-perimeter
  R : ℝ -- circumradius
  r : ℝ -- inradius
  S : ℝ -- area

-- State the theorem
theorem triangle_inequalities (t : Triangle) : 
  (Real.cos t.A + Real.cos t.B + Real.cos t.C ≤ 3/2) ∧
  (Real.sin (t.A/2) * Real.sin (t.B/2) * Real.sin (t.C/2) ≤ 1/8) ∧
  (t.a * t.b * t.c ≥ 8 * (t.p - t.a) * (t.p - t.b) * (t.p - t.c)) ∧
  (t.R ≥ 2 * t.r) ∧
  (t.S ≤ (1/2) * t.R * t.p) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l650_65077


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l650_65082

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l650_65082


namespace NUMINAMATH_CALUDE_king_total_payment_l650_65081

def crown_cost : ℚ := 20000
def architect_cost : ℚ := 50000
def chef_cost : ℚ := 10000

def crown_tip_percent : ℚ := 10 / 100
def architect_tip_percent : ℚ := 5 / 100
def chef_tip_percent : ℚ := 15 / 100

def total_cost : ℚ := crown_cost * (1 + crown_tip_percent) + 
                       architect_cost * (1 + architect_tip_percent) + 
                       chef_cost * (1 + chef_tip_percent)

theorem king_total_payment : total_cost = 86000 :=
by sorry

end NUMINAMATH_CALUDE_king_total_payment_l650_65081


namespace NUMINAMATH_CALUDE_custom_mul_theorem_l650_65089

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem stating that if a * 6 = -3 using the custom multiplication, then a = 23 -/
theorem custom_mul_theorem (a : ℝ) (h : custom_mul a 6 = -3) : a = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_theorem_l650_65089


namespace NUMINAMATH_CALUDE_hcf_problem_l650_65018

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2460) (h2 : Nat.lcm a b = 205) :
  Nat.gcd a b = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l650_65018


namespace NUMINAMATH_CALUDE_revenue_change_l650_65019

/-- Given a projected revenue increase and the ratio of actual to projected revenue,
    calculate the actual percent change in revenue. -/
theorem revenue_change
  (projected_increase : ℝ)
  (actual_to_projected_ratio : ℝ)
  (h1 : projected_increase = 0.20)
  (h2 : actual_to_projected_ratio = 0.75) :
  (1 + projected_increase) * actual_to_projected_ratio - 1 = -0.10 := by
  sorry

#check revenue_change

end NUMINAMATH_CALUDE_revenue_change_l650_65019


namespace NUMINAMATH_CALUDE_darren_tshirts_l650_65056

/-- The number of packs of white t-shirts Darren bought -/
def white_packs : ℕ := 5

/-- The number of t-shirts in each pack of white t-shirts -/
def white_per_pack : ℕ := 6

/-- The number of packs of blue t-shirts Darren bought -/
def blue_packs : ℕ := 3

/-- The number of t-shirts in each pack of blue t-shirts -/
def blue_per_pack : ℕ := 9

/-- The total number of t-shirts Darren bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem darren_tshirts : total_tshirts = 57 := by
  sorry

end NUMINAMATH_CALUDE_darren_tshirts_l650_65056


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l650_65042

def systematic_sampling (total : ℕ) (sample_size : ℕ) (interval_start : ℕ) (interval_end : ℕ) : ℕ :=
  let sampling_interval := total / sample_size
  let interval_size := interval_end - interval_start + 1
  interval_size / sampling_interval

theorem systematic_sampling_result :
  systematic_sampling 420 21 281 420 = 7 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l650_65042


namespace NUMINAMATH_CALUDE_sum_x_y_equals_22_over_5_l650_65020

theorem sum_x_y_equals_22_over_5 (x y : ℝ) 
  (eq1 : |x| + x + y = 12)
  (eq2 : x + |y| - y = 14) : 
  x + y = 22 / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_22_over_5_l650_65020


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l650_65034

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (remaining_players_age_diff : ℕ) : 
  team_size = 11 → 
  captain_age = 28 → 
  wicket_keeper_age_diff = 3 → 
  remaining_players_age_diff = 1 → 
  ∃ (team_avg_age : ℚ), 
    team_avg_age = 25 ∧ 
    team_size * team_avg_age = 
      captain_age + (captain_age + wicket_keeper_age_diff) + 
      (team_size - 2) * (team_avg_age - remaining_players_age_diff) :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l650_65034


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_terms_l650_65006

theorem arithmetic_sequence_max_terms 
  (a : ℝ) (n : ℕ) 
  (h1 : a^2 + (n - 1) * (a + 2 * (n - 1)) ≤ 100) : n ≤ 8 := by
  sorry

#check arithmetic_sequence_max_terms

end NUMINAMATH_CALUDE_arithmetic_sequence_max_terms_l650_65006


namespace NUMINAMATH_CALUDE_distance_traveled_l650_65028

-- Define the velocity function
def velocity (t : ℝ) : ℝ := t^2 + 1

-- Define the theorem
theorem distance_traveled (v : ℝ → ℝ) (a b : ℝ) : 
  (v = velocity) → (a = 0) → (b = 3) → ∫ x in a..b, v x = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l650_65028


namespace NUMINAMATH_CALUDE_reasoning_classification_l650_65049

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive

-- Define the reasoning methods
def method1 : String := "Inferring the properties of a ball from the properties of a circle"
def method2 : String := "Inducing that the sum of the internal angles of all triangles is 180° from the sum of the internal angles of right triangles, isosceles triangles, and equilateral triangles"
def method3 : String := "Deducing that f(x) = sinx is an odd function from f(-x) = -f(x), x ∈ R"
def method4 : String := "Inducing that the sum of the internal angles of a convex polygon is (n-2)•180° from the sum of the internal angles of a triangle, quadrilateral, and pentagon"

-- Define a function to classify reasoning methods
def classifyReasoning (method : String) : ReasoningType := sorry

-- Theorem to prove
theorem reasoning_classification :
  (classifyReasoning method1 = ReasoningType.Analogical) ∧
  (classifyReasoning method2 = ReasoningType.Inductive) ∧
  (classifyReasoning method3 = ReasoningType.Deductive) ∧
  (classifyReasoning method4 = ReasoningType.Inductive) := by
  sorry

end NUMINAMATH_CALUDE_reasoning_classification_l650_65049


namespace NUMINAMATH_CALUDE_drum_sticks_per_show_l650_65085

/-- Proves that the number of drum stick sets used per show for playing is 5 --/
theorem drum_sticks_per_show 
  (total_shows : ℕ) 
  (tossed_per_show : ℕ) 
  (total_sets : ℕ) 
  (h1 : total_shows = 30) 
  (h2 : tossed_per_show = 6) 
  (h3 : total_sets = 330) : 
  (total_sets - total_shows * tossed_per_show) / total_shows = 5 := by
  sorry

#check drum_sticks_per_show

end NUMINAMATH_CALUDE_drum_sticks_per_show_l650_65085


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l650_65040

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + k * x + 18 = 0 ∧ x = 2 - 3*I) → k = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l650_65040


namespace NUMINAMATH_CALUDE_five_person_circle_greetings_l650_65068

/-- Represents a circular arrangement of people --/
structure CircularArrangement (n : ℕ) where
  people : Fin n

/-- Number of greetings in a circular arrangement --/
def greetings (c : CircularArrangement 5) : ℕ := sorry

theorem five_person_circle_greetings :
  ∀ c : CircularArrangement 5, greetings c = 5 := by sorry

end NUMINAMATH_CALUDE_five_person_circle_greetings_l650_65068


namespace NUMINAMATH_CALUDE_absolute_value_equals_negative_l650_65009

theorem absolute_value_equals_negative (a : ℝ) : 
  (abs a = -a) → a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negative_l650_65009


namespace NUMINAMATH_CALUDE_no_further_simplification_l650_65037

theorem no_further_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  ∀ (f : ℝ → ℝ), f (a/b - b/a + a^2*b^2) = a/b - b/a + a^2*b^2 → f = id := by
  sorry

end NUMINAMATH_CALUDE_no_further_simplification_l650_65037


namespace NUMINAMATH_CALUDE_number_of_boys_number_of_boys_is_17_l650_65023

theorem number_of_boys (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neither_children : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) 
  (neither_boys : ℕ) : ℕ :=
  by
  have h1 : total_children = 60 := by sorry
  have h2 : happy_children = 30 := by sorry
  have h3 : sad_children = 10 := by sorry
  have h4 : neither_children = 20 := by sorry
  have h5 : girls = 43 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 5 := by sorry
  
  exact total_children - girls

theorem number_of_boys_is_17 : number_of_boys 60 30 10 20 43 6 4 5 = 17 := by sorry

end NUMINAMATH_CALUDE_number_of_boys_number_of_boys_is_17_l650_65023


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l650_65072

-- Define sets A and B
def A : Set ℝ := {x : ℝ | |x| > 4}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 6}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_interval : A_intersect_B = Set.Ioo 4 6 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l650_65072


namespace NUMINAMATH_CALUDE_prob_five_odd_in_seven_rolls_prob_five_odd_in_seven_rolls_proof_l650_65087

/-- The probability of getting exactly 5 odd numbers in 7 rolls of a fair 6-sided die -/
theorem prob_five_odd_in_seven_rolls : ℚ :=
  21 / 128

/-- A fair 6-sided die has equal probability for each outcome -/
axiom fair_die : ∀ (outcome : Fin 6), ℚ

/-- The probability of rolling an odd number on a fair 6-sided die is 1/2 -/
axiom prob_odd : (fair_die 1 + fair_die 3 + fair_die 5 : ℚ) = 1 / 2

/-- The rolls are independent -/
axiom independent_rolls : ∀ (n : ℕ), ℚ

/-- The probability of exactly k successes in n independent Bernoulli trials 
    with success probability p is given by the binomial probability formula -/
axiom binomial_probability : 
  ∀ (n k : ℕ) (p : ℚ), 
  0 ≤ p ∧ p ≤ 1 → 
  independent_rolls n = (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_five_odd_in_seven_rolls_proof : 
  prob_five_odd_in_seven_rolls = independent_rolls 7 :=
sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_seven_rolls_prob_five_odd_in_seven_rolls_proof_l650_65087


namespace NUMINAMATH_CALUDE_sqrt_inequality_l650_65090

theorem sqrt_inequality (a b : ℝ) : Real.sqrt a < Real.sqrt b → a < b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l650_65090


namespace NUMINAMATH_CALUDE_classroom_ratio_simplification_l650_65011

/-- The ratio of girls to boys in a classroom -/
def classroom_ratio (girls boys : ℕ) : ℚ := girls / boys

/-- The simplified ratio of girls to boys in the classroom -/
def simplified_ratio : ℚ := 1 / 2

theorem classroom_ratio_simplification :
  classroom_ratio 10 20 = simplified_ratio := by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_simplification_l650_65011


namespace NUMINAMATH_CALUDE_carbon_dioxide_formation_l650_65078

-- Define the chemical reaction
def chemical_reaction (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 = 1 ∧ NaHCO3 = 1 ∧ NaNO3 = 1 ∧ CO2 = 1 ∧ H2O = 1

-- Theorem statement
theorem carbon_dioxide_formation :
  ∀ (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ),
    chemical_reaction HNO3 NaHCO3 NaNO3 CO2 H2O →
    CO2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_carbon_dioxide_formation_l650_65078


namespace NUMINAMATH_CALUDE_transformation_maps_points_l650_65066

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Scales a point by a factor about the origin -/
def scale (p : Point) (factor : ℝ) : Point :=
  { x := p.x * factor, y := p.y * factor }

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Applies scaling followed by reflection across x-axis -/
def scaleAndReflect (p : Point) (factor : ℝ) : Point :=
  reflectX (scale p factor)

theorem transformation_maps_points :
  let C : Point := { x := -5, y := 2 }
  let D : Point := { x := 0, y := 3 }
  let C' : Point := { x := 10, y := -4 }
  let D' : Point := { x := 0, y := -6 }
  (scaleAndReflect C 2 = C') ∧ (scaleAndReflect D 2 = D') := by
  sorry

end NUMINAMATH_CALUDE_transformation_maps_points_l650_65066


namespace NUMINAMATH_CALUDE_kevins_cards_l650_65043

/-- Kevin's card problem -/
theorem kevins_cards (initial_cards found_cards : ℕ) : 
  initial_cards = 7 → found_cards = 47 → initial_cards + found_cards = 54 := by
  sorry

end NUMINAMATH_CALUDE_kevins_cards_l650_65043


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l650_65044

theorem simplify_sqrt_expression :
  Real.sqrt 768 / Real.sqrt 192 - Real.sqrt 98 / Real.sqrt 49 = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l650_65044


namespace NUMINAMATH_CALUDE_rectangle_to_square_cut_l650_65004

theorem rectangle_to_square_cut (rectangle_length : ℝ) (rectangle_width : ℝ) (num_parts : ℕ) :
  rectangle_length = 2 ∧ rectangle_width = 1 ∧ num_parts = 3 →
  ∃ (square_side : ℝ), square_side = Real.sqrt 2 ∧
    rectangle_length * rectangle_width = square_side * square_side :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_cut_l650_65004


namespace NUMINAMATH_CALUDE_student_arrangement_theorem_l650_65067

/-- The number of ways to arrange 3 male and 2 female students in a row -/
def total_arrangements : ℕ := 120

/-- The number of arrangements where exactly two male students are adjacent -/
def two_male_adjacent : ℕ := 72

/-- The number of arrangements where 3 male students of different heights 
    are arranged in descending order of height -/
def male_descending_height : ℕ := 20

/-- Given 3 male students and 2 female students, prove:
    1. The total number of arrangements
    2. The number of arrangements with exactly two male students adjacent
    3. The number of arrangements with male students in descending height order -/
theorem student_arrangement_theorem 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (h1 : male_count = 3) 
  (h2 : female_count = 2) :
  (total_arrangements = 120) ∧ 
  (two_male_adjacent = 72) ∧ 
  (male_descending_height = 20) := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_theorem_l650_65067


namespace NUMINAMATH_CALUDE_function_properties_l650_65054

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x + b

theorem function_properties (a b : ℝ) :
  (∀ x y : ℝ, x = 1 → y = f a b x → (3 * x - y - 3 = 0) → (a = -2 ∧ b = -1/2)) ∧
  ((∀ x : ℝ, x ≠ 0 → (deriv (f a b) x = 0 ↔ x = 1)) → a = 1) ∧
  ((-2 ≤ a ∧ a < 0) →
    (∃ m : ℝ, m = 12 ∧
      (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ ≤ 2 ∧ 0 < x₂ ∧ x₂ ≤ 2 →
        |f a b x₁ - f a b x₂| ≤ m * |1/x₁ - 1/x₂|) ∧
      (∀ m' : ℝ, m' < m →
        ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ ≤ 2 ∧ 0 < x₂ ∧ x₂ ≤ 2 ∧
          |f a b x₁ - f a b x₂| > m' * |1/x₁ - 1/x₂|))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l650_65054


namespace NUMINAMATH_CALUDE_matrix_vector_product_l650_65098

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; -6, 5]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![2; -3]

theorem matrix_vector_product :
  A * v = !![14; -27] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_product_l650_65098


namespace NUMINAMATH_CALUDE_least_k_factorial_multiple_of_315_l650_65080

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem least_k_factorial_multiple_of_315 (k : ℕ) (h1 : k > 1) (h2 : 315 ∣ factorial k) :
  k ≥ 7 ∧ 315 ∣ factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_least_k_factorial_multiple_of_315_l650_65080


namespace NUMINAMATH_CALUDE_john_payment_amount_l650_65084

/-- The final amount John needs to pay after late charges -/
def final_amount (original_bill : ℝ) (first_charge : ℝ) (second_charge : ℝ) (third_charge : ℝ) : ℝ :=
  original_bill * (1 + first_charge) * (1 + second_charge) * (1 + third_charge)

/-- Theorem stating the final amount John needs to pay -/
theorem john_payment_amount :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |final_amount 500 0.02 0.03 0.025 - 538.43| < ε :=
sorry

end NUMINAMATH_CALUDE_john_payment_amount_l650_65084


namespace NUMINAMATH_CALUDE_intersection_probability_odd_polygon_l650_65055

/-- The probability that two randomly chosen diagonals intersect inside a convex polygon with 2n+1 vertices -/
theorem intersection_probability_odd_polygon (n : ℕ) :
  let vertices := 2 * n + 1
  let diagonals := n * (2 * n + 1) - (2 * n + 1)
  let ways_to_choose_diagonals := (diagonals.choose 2 : ℚ)
  let ways_to_choose_vertices := ((2 * n + 1).choose 4 : ℚ)
  ways_to_choose_vertices / ways_to_choose_diagonals = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_probability_odd_polygon_l650_65055


namespace NUMINAMATH_CALUDE_inequality_solutions_l650_65086

theorem inequality_solutions : 
  ∃! (s : Finset Int), 
    (∀ y ∈ s, (2 * y ≤ -y + 4 ∧ 5 * y ≥ -10 ∧ 3 * y ≤ -2 * y + 20)) ∧ 
    s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l650_65086


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l650_65016

/-- Theorem: Relationship between heights of two cylinders with equal volume and different radii -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l650_65016


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_l650_65046

/-- The complement of an angle in degrees -/
def complement (α : ℝ) : ℝ := 90 - α

/-- The supplement of an angle in degrees -/
def supplement (α : ℝ) : ℝ := 180 - α

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125 degrees -/
theorem supplement_of_complement_of_35 :
  supplement (complement 35) = 125 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_l650_65046


namespace NUMINAMATH_CALUDE_base_10_to_base_6_l650_65025

theorem base_10_to_base_6 : 
  (1 * 6^4 + 3 * 6^3 + 0 * 6^2 + 5 * 6^1 + 4 * 6^0 : ℕ) = 1978 := by
  sorry

#eval 1 * 6^4 + 3 * 6^3 + 0 * 6^2 + 5 * 6^1 + 4 * 6^0

end NUMINAMATH_CALUDE_base_10_to_base_6_l650_65025


namespace NUMINAMATH_CALUDE_cone_height_l650_65099

/-- Given a cone with base radius 1 and central angle of the unfolded side view 2/3π,
    the height of the cone is 2√2. -/
theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ) : 
  r = 1 → θ = (2/3) * Real.pi → h = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l650_65099


namespace NUMINAMATH_CALUDE_max_value_on_circle_l650_65063

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 10 →
  ∃ (max : ℝ), max = 5 * Real.sqrt 10 ∧ ∀ (a b : ℝ), a^2 + b^2 = 10 → 3*a + 4*b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l650_65063


namespace NUMINAMATH_CALUDE_age_problem_l650_65007

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l650_65007


namespace NUMINAMATH_CALUDE_age_difference_l650_65059

theorem age_difference (A B : ℕ) : B = 48 → A + 10 = 2 * (B - 10) → A - B = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l650_65059


namespace NUMINAMATH_CALUDE_flight_duration_sum_l650_65092

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + t2.minutes - t1.minutes

/-- Theorem: The flight duration sum is 39 -/
theorem flight_duration_sum : 
  let departureTime : Time := ⟨9, 47, by sorry⟩
  let arrivalTime : Time := ⟨12, 25, by sorry⟩  -- Adjusted for timezone difference
  let durationMinutes := timeDiffMinutes departureTime arrivalTime
  let h := durationMinutes / 60
  let m := durationMinutes % 60
  0 < m ∧ m < 60 → h + m = 39 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l650_65092


namespace NUMINAMATH_CALUDE_min_value_of_f_l650_65093

/-- The quadratic function f(x) = x^2 + 6x + 13 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 13

theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 4 ∧ ∃ x₀ : ℝ, f x₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l650_65093


namespace NUMINAMATH_CALUDE_find_b_l650_65047

/-- Given the conditions, prove that b = -2 --/
theorem find_b (a : ℕ) (b : ℝ) : 
  (2 * (a.choose 2) - (a.choose 1 - 1) * 6 = 0) →  -- Condition 1
  (b ≠ 0) →                                        -- Condition 3
  (a.choose 1 * b = -12) →                         -- Condition 2 (simplified)
  b = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l650_65047


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l650_65057

theorem reciprocal_of_sum (y : ℚ) : y = 6 + 1/6 → 1/y = 6/37 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l650_65057


namespace NUMINAMATH_CALUDE_prime_divisibility_l650_65051

theorem prime_divisibility (p q r : ℕ) : 
  Prime p → Prime q → Prime r → Odd p → (p ∣ q^r + 1) → 
  (2*r ∣ p - 1) ∨ (p ∣ q^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l650_65051


namespace NUMINAMATH_CALUDE_selene_purchase_l650_65094

theorem selene_purchase (camera_price : ℝ) (frame_price : ℝ) (discount_rate : ℝ) (total_paid : ℝ) :
  camera_price = 110 →
  frame_price = 120 →
  discount_rate = 0.05 →
  total_paid = 551 →
  ∃ num_frames : ℕ,
    (1 - discount_rate) * (2 * camera_price + num_frames * frame_price) = total_paid ∧
    num_frames = 3 :=
by sorry

end NUMINAMATH_CALUDE_selene_purchase_l650_65094


namespace NUMINAMATH_CALUDE_polygon_sides_count_l650_65041

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (2 * 360 : ℝ) = (n - 2 : ℝ) * 180 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l650_65041


namespace NUMINAMATH_CALUDE_restaurant_menu_combinations_l650_65083

theorem restaurant_menu_combinations (n : ℕ) (h : n = 12) :
  n * (n - 1) = 132 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_menu_combinations_l650_65083


namespace NUMINAMATH_CALUDE_two_digit_number_value_l650_65030

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Theorem: The value of a two-digit number is 10a + b, where a is the tens digit and b is the ones digit -/
theorem two_digit_number_value (n : TwoDigitNumber) : 
  n.value = 10 * n.tens + n.ones := by sorry

end NUMINAMATH_CALUDE_two_digit_number_value_l650_65030


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l650_65031

theorem minimum_value_theorem (x : ℝ) (h : x > 3) :
  (x + 18) / Real.sqrt (x - 3) ≥ 2 * Real.sqrt 21 ∧
  (∃ x₀ : ℝ, x₀ > 3 ∧ (x₀ + 18) / Real.sqrt (x₀ - 3) = 2 * Real.sqrt 21 ∧ x₀ = 24) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l650_65031


namespace NUMINAMATH_CALUDE_divide_by_four_theorem_l650_65053

theorem divide_by_four_theorem (x : ℝ) (h : 812 / x = 25) : x / 4 = 8.12 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_four_theorem_l650_65053


namespace NUMINAMATH_CALUDE_initial_ratio_of_liquids_l650_65005

/-- Given a mixture of two liquids p and q with total volume 40 liters,
    if adding 15 liters of q results in a ratio of 5:6 for p:q,
    then the initial ratio of p:q was 5:3. -/
theorem initial_ratio_of_liquids (p q : ℝ) : 
  p + q = 40 →
  p / (q + 15) = 5 / 6 →
  p / q = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_of_liquids_l650_65005


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l650_65058

theorem least_addition_for_divisibility : 
  (∃ (n : ℕ), 25 ∣ (1019 + n) ∧ ∀ (m : ℕ), m < n → ¬(25 ∣ (1019 + m))) ∧ 
  (∃ (n : ℕ), n = 6 ∧ 25 ∣ (1019 + n) ∧ ∀ (m : ℕ), m < n → ¬(25 ∣ (1019 + m))) :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l650_65058


namespace NUMINAMATH_CALUDE_max_planes_from_parallel_lines_max_planes_is_six_l650_65060

/-- Given four parallel lines, the maximum number of unique planes formed by selecting two lines -/
theorem max_planes_from_parallel_lines : ℕ :=
  -- Define the number of lines
  let num_lines : ℕ := 4

  -- Define the function to calculate combinations
  let combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

  -- Calculate the number of ways to select 2 lines out of 4
  combinations num_lines 2

/-- Proof that the maximum number of planes is 6 -/
theorem max_planes_is_six : max_planes_from_parallel_lines = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_planes_from_parallel_lines_max_planes_is_six_l650_65060


namespace NUMINAMATH_CALUDE_sin_double_theta_l650_65027

theorem sin_double_theta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (2 * θ) = 6 * Real.sqrt 8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_theta_l650_65027


namespace NUMINAMATH_CALUDE_min_n_for_S_greater_than_1020_l650_65050

def S (n : ℕ) : ℕ := 2 * (2^n - 1) - n

theorem min_n_for_S_greater_than_1020 :
  ∀ k : ℕ, k < 10 → S k ≤ 1020 ∧ S 10 > 1020 := by sorry

end NUMINAMATH_CALUDE_min_n_for_S_greater_than_1020_l650_65050


namespace NUMINAMATH_CALUDE_fixed_point_satisfies_function_l650_65073

/-- A linear function of the form y = kx + k + 2 -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + k + 2

/-- The fixed point of the linear function -/
def fixedPoint : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the linear function for all k -/
theorem fixed_point_satisfies_function :
  ∀ k : ℝ, linearFunction k (fixedPoint.1) = fixedPoint.2 := by
  sorry

#check fixed_point_satisfies_function

end NUMINAMATH_CALUDE_fixed_point_satisfies_function_l650_65073


namespace NUMINAMATH_CALUDE_fibonacci_6_l650_65022

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_6 : fibonacci 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_6_l650_65022
