import Mathlib

namespace gcd_of_45_and_75_l2633_263306

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_of_45_and_75_l2633_263306


namespace count_multiples_of_ten_l2633_263314

theorem count_multiples_of_ten : ∃ n : ℕ, n = (Finset.filter (λ x => x % 10 = 0 ∧ x > 9 ∧ x < 101) (Finset.range 101)).card ∧ n = 10 := by
  sorry

end count_multiples_of_ten_l2633_263314


namespace vertex_locus_is_circle_l2633_263308

/-- A triangle with a fixed base and a median of constant length --/
structure TriangleWithMedian where
  /-- The length of the fixed base AB --/
  base_length : ℝ
  /-- The length of the median from A to side BC --/
  median_length : ℝ

/-- The locus of vertex C in a triangle with a fixed base and constant median length --/
def vertex_locus (t : TriangleWithMedian) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p | ∃ (A B : EuclideanSpace ℝ (Fin 2)), 
    ‖B - A‖ = t.base_length ∧ 
    ‖p - A‖ = t.median_length}

/-- The theorem stating that the locus of vertex C is a circle --/
theorem vertex_locus_is_circle (t : TriangleWithMedian) 
  (h : t.base_length = 6 ∧ t.median_length = 3) : 
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    vertex_locus t = {p | ‖p - center‖ = radius} ∧ radius = 3 :=
sorry

end vertex_locus_is_circle_l2633_263308


namespace novel_writing_speed_l2633_263315

/-- Given a novel with 40,000 words written in 80 hours, 
    the average number of words written per hour is 500. -/
theorem novel_writing_speed (total_words : ℕ) (total_hours : ℕ) 
  (h1 : total_words = 40000) (h2 : total_hours = 80) :
  total_words / total_hours = 500 := by
  sorry

#check novel_writing_speed

end novel_writing_speed_l2633_263315


namespace triangle_side_length_l2633_263367

/-- Represents a triangle with sides a, b, c and heights ha, hb, hc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ

/-- Theorem: In a triangle with sides AC = 6 cm and BC = 3 cm, 
    if the half-sum of heights to AC and BC equals the height to AB, 
    then AB = 4 cm -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.b = 6)
  (h2 : t.c = 3)
  (h3 : (t.ha + t.hb) / 2 = t.hc) : 
  t.a = 4 := by
  sorry

end triangle_side_length_l2633_263367


namespace log_sum_condition_l2633_263335

theorem log_sum_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b, a > 1 ∧ b > 1 → Real.log a + Real.log b > 0) ∧
  (∃ a b, Real.log a + Real.log b > 0 ∧ ¬(a > 1 ∧ b > 1)) := by
  sorry

end log_sum_condition_l2633_263335


namespace quadratic_inequality_solution_l2633_263381

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a + b = -1 := by
sorry

end quadratic_inequality_solution_l2633_263381


namespace sphere_remaining_volume_l2633_263354

/-- The remaining volume of a sphere after drilling a cylindrical hole -/
theorem sphere_remaining_volume (R : ℝ) (h : R > 3) : 
  (4 / 3 * π * R^3) - (6 * π * (R^2 - 9)) - (2 * π * 3^2 * (R - 3 / 3)) = 36 * π :=
sorry

end sphere_remaining_volume_l2633_263354


namespace improve_shooting_average_l2633_263358

/-- Represents a basketball player's shooting statistics -/
structure ShootingStats :=
  (initial_shots : ℕ)
  (initial_made : ℕ)
  (additional_shots : ℕ)
  (additional_made : ℕ)

/-- Calculates the shooting average as a rational number -/
def shooting_average (stats : ShootingStats) : ℚ :=
  (stats.initial_made + stats.additional_made : ℚ) / (stats.initial_shots + stats.additional_shots)

theorem improve_shooting_average 
  (stats : ShootingStats) 
  (h1 : stats.initial_shots = 40)
  (h2 : stats.initial_made = 18)
  (h3 : stats.additional_shots = 15)
  (h4 : shooting_average {initial_shots := stats.initial_shots, 
                          initial_made := stats.initial_made, 
                          additional_shots := 0, 
                          additional_made := 0} = 45/100)
  : shooting_average {initial_shots := stats.initial_shots,
                      initial_made := stats.initial_made,
                      additional_shots := stats.additional_shots,
                      additional_made := 12} = 55/100 := by
  sorry

end improve_shooting_average_l2633_263358


namespace expand_product_l2633_263339

theorem expand_product (x : ℝ) : (x + 6) * (x + 8) * (x - 3) = x^3 + 11*x^2 + 6*x - 144 := by
  sorry

end expand_product_l2633_263339


namespace average_of_7_12_and_M_l2633_263373

theorem average_of_7_12_and_M :
  ∃ M : ℝ, 10 < M ∧ M < 20 ∧ ((7 + 12 + M) / 3 = 11 ∨ (7 + 12 + M) / 3 = 13) := by
  sorry

end average_of_7_12_and_M_l2633_263373


namespace intersection_point_l2633_263300

/-- The system of linear equations representing two lines -/
def system (x y : ℝ) : Prop :=
  8 * x + 5 * y = 40 ∧ 3 * x - 10 * y = 15

/-- The theorem stating that (5, 0) is the unique solution to the system -/
theorem intersection_point : ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (5, 0) := by sorry

end intersection_point_l2633_263300


namespace chord_length_l2633_263396

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (x y : ℝ) : 
  (x - y + 2 = 0) →  -- Line equation
  ((x - 1)^2 + (y - 2)^2 = 4) →  -- Circle equation
  ∃ A B : ℝ × ℝ,  -- Points of intersection
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14  -- Length of AB squared
  := by sorry

end chord_length_l2633_263396


namespace equilateral_triangle_sum_product_l2633_263345

def is_equilateral_triangle (x y z : ℂ) : Prop :=
  Complex.abs (y - x) = Complex.abs (z - y) ∧ 
  Complex.abs (z - y) = Complex.abs (x - z) ∧
  Complex.abs (x - z) = Complex.abs (y - x)

theorem equilateral_triangle_sum_product (x y z : ℂ) :
  is_equilateral_triangle x y z →
  Complex.abs (y - x) = 24 →
  Complex.abs (x + y + z) = 72 →
  Complex.abs (x * y + x * z + y * z) = 1728 := by
  sorry

end equilateral_triangle_sum_product_l2633_263345


namespace area_triangle_OAB_l2633_263337

/-- Given a polar coordinate system with pole O, point A(1, π/6), and point B(2, π/2),
    the area of triangle OAB is √3/2. -/
theorem area_triangle_OAB :
  let r₁ : ℝ := 1
  let θ₁ : ℝ := π / 6
  let r₂ : ℝ := 2
  let θ₂ : ℝ := π / 2
  let area := (1 / 2) * r₁ * r₂ * Real.sin (θ₂ - θ₁)
  area = Real.sqrt 3 / 2 := by
  sorry

end area_triangle_OAB_l2633_263337


namespace smallest_divisible_by_1_to_10_l2633_263368

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ m) → n ≤ m) := by
  sorry

end smallest_divisible_by_1_to_10_l2633_263368


namespace set_representation_l2633_263309

theorem set_representation :
  {x : ℕ | 8 < x ∧ x < 12} = {9, 10, 11} := by
  sorry

end set_representation_l2633_263309


namespace system_solution_l2633_263398

/-- Given a system of equations x + y = 2a and xy(x^2 + y^2) = 2b^4,
    this theorem states the condition for real solutions and
    provides the solutions for specific values of a and b. -/
theorem system_solution (a b : ℝ) (h : b^4 = 9375) :
  (∀ x y : ℝ, x + y = 2*a ∧ x*y*(x^2 + y^2) = 2*b^4 → a^2 ≥ b^2) ∧
  (a = 10 → ∃ x y : ℝ, (x = 15 ∧ y = 5 ∨ x = 5 ∧ y = 15) ∧
                       x + y = 2*a ∧ x*y*(x^2 + y^2) = 2*b^4) :=
by sorry

end system_solution_l2633_263398


namespace intersection_and_complement_when_m_eq_2_existence_of_m_for_subset_l2633_263374

-- Define sets A and B
def A : Set ℝ := {x | (x + 3) / (x - 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x - 2*m^2 ≤ 0}

-- Theorem for part (1)
theorem intersection_and_complement_when_m_eq_2 :
  (A ∩ B 2 = {x | -2 ≤ x ∧ x < 1}) ∧
  (Set.univ \ B 2 = {x | x < -2 ∨ x > 4}) := by sorry

-- Theorem for part (2)
theorem existence_of_m_for_subset :
  (∃ m : ℝ, m ≥ 3 ∧ A ⊆ B m) ∧
  (∀ m : ℝ, m < 3 → ¬(A ⊆ B m)) := by sorry

end intersection_and_complement_when_m_eq_2_existence_of_m_for_subset_l2633_263374


namespace parabola_translation_l2633_263355

-- Define the original parabola
def original_parabola (x y : ℝ) : Prop := y = x^2 + 3

-- Define the translated parabola
def translated_parabola (x y : ℝ) : Prop := y = (x + 1)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, original_parabola (x + 1) y ↔ translated_parabola x y :=
by sorry

end parabola_translation_l2633_263355


namespace tom_marble_combinations_l2633_263394

/-- Represents the number of marbles of each color -/
structure MarbleSet :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- Calculates the number of ways to choose 2 marbles from a given set -/
def chooseTwo (s : MarbleSet) : ℕ :=
  sorry

/-- Tom's marble set -/
def tomMarbles : MarbleSet :=
  { red := 1, blue := 1, green := 2, yellow := 3 }

theorem tom_marble_combinations :
  chooseTwo tomMarbles = 19 :=
sorry

end tom_marble_combinations_l2633_263394


namespace oplus_inequality_solutions_l2633_263385

def oplus (a b : ℤ) : ℤ := 1 - a * b

theorem oplus_inequality_solutions :
  (∃! (n : ℕ), ∀ (x : ℕ), oplus x 2 ≥ -3 ↔ x ≤ n) ∧
  (∃ (s : Finset ℕ), s.card = 3 ∧ ∀ (x : ℕ), x ∈ s ↔ oplus x 2 ≥ -3) :=
by sorry

end oplus_inequality_solutions_l2633_263385


namespace remaining_three_digit_numbers_l2633_263325

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical adjacent digits and a different third digit -/
def excluded_numbers : ℕ := 162

/-- The remaining count of three-digit numbers after exclusion -/
def remaining_numbers : ℕ := total_three_digit_numbers - excluded_numbers

theorem remaining_three_digit_numbers : remaining_numbers = 738 := by
  sorry

end remaining_three_digit_numbers_l2633_263325


namespace sufficient_not_necessary_condition_l2633_263366

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ 
  (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end sufficient_not_necessary_condition_l2633_263366


namespace right_triangle_integer_area_l2633_263371

theorem right_triangle_integer_area (a b : ℕ) :
  (∃ A : ℕ, A = a * b / 2) ↔ (Even a ∨ Even b) :=
sorry

end right_triangle_integer_area_l2633_263371


namespace z_in_first_quadrant_l2633_263384

theorem z_in_first_quadrant : 
  ∃ (z : ℂ), (Complex.I + 1) * z = Complex.I^2013 ∧ 
  z.re > 0 ∧ z.im > 0 :=
by sorry

end z_in_first_quadrant_l2633_263384


namespace min_value_a_plus_b_l2633_263370

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 4 * a + b = a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 4 * x + y = x * y → a + b ≤ x + y ∧ a + b = 9 :=
sorry

end min_value_a_plus_b_l2633_263370


namespace abs_ab_minus_cd_le_quarter_l2633_263378

theorem abs_ab_minus_cd_le_quarter 
  (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) : 
  |a * b - c * d| ≤ 1/4 := by
  sorry

end abs_ab_minus_cd_le_quarter_l2633_263378


namespace trailer_cost_is_120000_l2633_263395

/-- Represents the cost of a house in dollars -/
def house_cost : ℕ := 480000

/-- Represents the loan period in months -/
def loan_period : ℕ := 240

/-- Represents the additional monthly payment for the house compared to the trailer in dollars -/
def additional_house_payment : ℕ := 1500

/-- Calculates the cost of the trailer given the house cost, loan period, and additional house payment -/
def trailer_cost (h : ℕ) (l : ℕ) (a : ℕ) : ℕ := 
  h - l * a

/-- Theorem stating that the cost of the trailer is $120,000 -/
theorem trailer_cost_is_120000 : 
  trailer_cost house_cost loan_period additional_house_payment = 120000 := by
  sorry

end trailer_cost_is_120000_l2633_263395


namespace all_positive_integers_are_dapper_l2633_263301

/-- A positive integer is dapper if at least one of its multiples begins with 2008. -/
def is_dapper (n : ℕ+) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), k * n.val = 2008 * 10^m + m ∧ m < 10^m

/-- Every positive integer is dapper. -/
theorem all_positive_integers_are_dapper : ∀ (n : ℕ+), is_dapper n := by
  sorry

end all_positive_integers_are_dapper_l2633_263301


namespace image_of_3_4_preimages_of_1_neg6_l2633_263330

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (3, 4)
theorem image_of_3_4 : f (3, 4) = (7, 12) := by sorry

-- Theorem for the pre-images of (1, -6)
theorem preimages_of_1_neg6 : 
  {p : ℝ × ℝ | f p = (1, -6)} = {(-2, 3), (3, -2)} := by sorry

end image_of_3_4_preimages_of_1_neg6_l2633_263330


namespace fish_pond_estimation_l2633_263391

def fish_estimation (initial_catch : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) : ℕ :=
  (initial_catch * second_catch) / marked_in_second

theorem fish_pond_estimation :
  let initial_catch := 200
  let second_catch := 200
  let marked_in_second := 8
  fish_estimation initial_catch second_catch marked_in_second = 5000 := by
  sorry

end fish_pond_estimation_l2633_263391


namespace a_equals_three_iff_parallel_not_coincident_l2633_263302

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + 2 * y + 3 * a = 0
  line2 : ℝ × ℝ → Prop := fun (x, y) ↦ 3 * x + (a - 1) * y + 7 - a = 0

/-- Condition for two lines to be parallel and not coincident -/
def parallel_not_coincident (l : TwoLines) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (∀ (x y : ℝ), l.line1 (x, y) ↔ l.line2 (k * x + l.a, k * y + 2))

/-- The main theorem -/
theorem a_equals_three_iff_parallel_not_coincident (l : TwoLines) :
  l.a = 3 ↔ parallel_not_coincident l :=
sorry

end a_equals_three_iff_parallel_not_coincident_l2633_263302


namespace quadratic_root_ratio_l2633_263312

theorem quadratic_root_ratio (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = 2 * x ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  2 * b^2 = 9 * a * c :=
by sorry

end quadratic_root_ratio_l2633_263312


namespace kate_keyboard_cost_l2633_263364

/-- The amount Kate spent on the keyboard -/
def keyboard_cost (march_savings april_savings may_savings mouse_cost remaining : ℕ) : ℕ :=
  (march_savings + april_savings + may_savings) - (mouse_cost + remaining)

theorem kate_keyboard_cost :
  keyboard_cost 27 13 28 5 14 = 49 := by
  sorry

end kate_keyboard_cost_l2633_263364


namespace families_with_car_or_ebike_l2633_263323

theorem families_with_car_or_ebike (total_car : ℕ) (total_ebike : ℕ) (both : ℕ) :
  total_car = 35 → total_ebike = 65 → both = 20 →
  total_car + total_ebike - both = 80 := by
  sorry

end families_with_car_or_ebike_l2633_263323


namespace odd_m_triple_g_eq_5_l2633_263319

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n - 4 else n / 3

theorem odd_m_triple_g_eq_5 (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 5) : m = 17 := by
  sorry

end odd_m_triple_g_eq_5_l2633_263319


namespace psychology_lecture_first_probability_l2633_263340

-- Define the type for lectures
inductive Lecture
| Morality
| Psychology
| Safety

-- Define a function to calculate the number of permutations
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the theorem
theorem psychology_lecture_first_probability :
  let total_arrangements := factorial 3
  let favorable_arrangements := factorial 2
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3 := by
sorry


end psychology_lecture_first_probability_l2633_263340


namespace ian_painted_cuboids_l2633_263305

/-- The number of cuboids painted by Ian -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted -/
def total_faces : ℕ := 48

/-- The number of faces on one cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem: The number of cuboids painted is equal to 8 -/
theorem ian_painted_cuboids : 
  num_cuboids = total_faces / faces_per_cuboid :=
sorry

end ian_painted_cuboids_l2633_263305


namespace quadratic_root_existence_l2633_263336

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by
  sorry

end quadratic_root_existence_l2633_263336


namespace restaurant_cooks_count_l2633_263341

/-- Proves that the number of cooks is 9 given the initial and final ratios of cooks to waiters -/
theorem restaurant_cooks_count (cooks waiters : ℕ) : 
  (cooks : ℚ) / waiters = 3 / 10 →
  cooks / (waiters + 12) = 3 / 14 →
  cooks = 9 := by
sorry

end restaurant_cooks_count_l2633_263341


namespace max_sum_squared_distances_l2633_263317

open InnerProductSpace

theorem max_sum_squared_distances {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c d : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 ∧
  ∃ (a' b' c' d' : V), ‖a'‖ = 1 ∧ ‖b'‖ = 1 ∧ ‖c'‖ = 1 ∧ ‖d'‖ = 1 ∧
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖a' - d'‖^2 + ‖b' - c'‖^2 + ‖b' - d'‖^2 + ‖c' - d'‖^2 = 16 :=
by
  sorry

end max_sum_squared_distances_l2633_263317


namespace range_of_a_l2633_263324

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a : 
  (∀ x, p x → (∀ a, q x a)) ∧ 
  (∃ x a, ¬(p x) ∧ q x a) → 
  ∀ a, 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_a_l2633_263324


namespace quadratic_function_properties_l2633_263377

def f (x : ℝ) : ℝ := 2 * (x + 1) * (x - 3)

theorem quadratic_function_properties :
  (∀ x, f x = 2 * (x + 1) * (x - 3)) ∧
  f (-1) = 0 ∧ f 3 = 0 ∧ f 1 = -8 ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ -8 ∧ f x ≤ 0) ∧
  (∀ x, f x ≥ 0 ↔ x ≤ -1 ∨ x ≥ 3) :=
by sorry

#check quadratic_function_properties

end quadratic_function_properties_l2633_263377


namespace square_diff_sum_l2633_263356

theorem square_diff_sum : 1010^2 - 990^2 - 1005^2 + 995^2 + 1012^2 - 988^2 = 68000 := by
  sorry

end square_diff_sum_l2633_263356


namespace regular_polygon_sides_l2633_263304

theorem regular_polygon_sides (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 108) (h2 : side_length = 12) :
  (perimeter / side_length : ℝ) = 9 := by
  sorry

end regular_polygon_sides_l2633_263304


namespace birth_rate_calculation_l2633_263362

/-- The average birth rate in people per two seconds -/
def average_birth_rate : ℝ := 5

/-- The death rate in people per two seconds -/
def death_rate : ℝ := 3

/-- The number of two-second intervals in a day -/
def intervals_per_day : ℝ := 43200

/-- The net population increase in one day -/
def net_increase_per_day : ℝ := 86400

theorem birth_rate_calculation :
  (average_birth_rate - death_rate) * intervals_per_day = net_increase_per_day :=
by sorry

end birth_rate_calculation_l2633_263362


namespace rectangle_width_l2633_263376

/-- A rectangle with a perimeter of 20 cm and length 2 cm more than its width has a width of 4 cm. -/
theorem rectangle_width (w : ℝ) (h1 : 2 * (w + 2) + 2 * w = 20) : w = 4 := by
  sorry

end rectangle_width_l2633_263376


namespace mary_seth_age_difference_l2633_263350

/-- Represents the age difference between Mary and Seth -/
def age_difference : ℝ → ℝ → ℝ := λ m s => m - s

/-- Mary's age after one year will be three times Seth's age after one year -/
def future_age_relation (m : ℝ) (s : ℝ) : Prop := m + 1 = 3 * (s + 1)

theorem mary_seth_age_difference :
  ∀ (m s : ℝ),
  m > s →
  future_age_relation m s →
  m + s = 3.5 →
  age_difference m s = 2.75 := by
sorry

end mary_seth_age_difference_l2633_263350


namespace subtract_linear_equations_l2633_263310

theorem subtract_linear_equations :
  let eq1 : ℝ → ℝ → ℝ := λ x y => 2 * x + 3 * y
  let eq2 : ℝ → ℝ → ℝ := λ x y => 5 * x + 3 * y
  let result : ℝ → ℝ := λ x => -3 * x
  (∀ x y, eq1 x y = 11) →
  (∀ x y, eq2 x y = -7) →
  (∀ x, result x = 18) →
  ∀ x y, eq1 x y - eq2 x y = result x :=
by
  sorry

end subtract_linear_equations_l2633_263310


namespace cubic_roots_sum_cubes_l2633_263347

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (3 * a^3 + 5 * a^2 - 150 * a + 7 = 0) →
  (3 * b^3 + 5 * b^2 - 150 * b + 7 = 0) →
  (3 * c^3 + 5 * c^2 - 150 * c + 7 = 0) →
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 303 := by
sorry

end cubic_roots_sum_cubes_l2633_263347


namespace money_left_l2633_263333

/-- Proves that if a person has 15 cents and spends 11 cents, they will have 4 cents left. -/
theorem money_left (initial_amount spent_amount : ℕ) : 
  initial_amount = 15 → spent_amount = 11 → initial_amount - spent_amount = 4 := by
  sorry

end money_left_l2633_263333


namespace mean_exercise_days_jenkins_class_l2633_263321

/-- Represents the exercise data for a group of students -/
structure ExerciseData where
  students : List (Nat × Float)

/-- Calculates the mean number of days exercised -/
def calculateMean (data : ExerciseData) : Float :=
  let totalDays := data.students.foldl (fun acc (n, d) => acc + n.toFloat * d) 0
  let totalStudents := data.students.foldl (fun acc (n, _) => acc + n) 0
  totalDays / totalStudents.toFloat

/-- Rounds a float to the nearest hundredth -/
def roundToHundredth (x : Float) : Float :=
  (x * 100).round / 100

theorem mean_exercise_days_jenkins_class :
  let jenkinsData : ExerciseData := {
    students := [
      (2, 0.5),
      (4, 1),
      (5, 3),
      (3, 4),
      (7, 6),
      (2, 7)
    ]
  }
  roundToHundredth (calculateMean jenkinsData) = 3.83 := by
  sorry

end mean_exercise_days_jenkins_class_l2633_263321


namespace family_size_family_size_is_four_l2633_263360

theorem family_size (current_avg_age : ℝ) (youngest_age : ℝ) (birth_avg_age : ℝ) : ℝ :=
  let n := (youngest_age * birth_avg_age) / (current_avg_age - birth_avg_age - youngest_age)
  n

#check family_size 20 10 12.5

theorem family_size_is_four :
  family_size 20 10 12.5 = 4 := by sorry

end family_size_family_size_is_four_l2633_263360


namespace triangle_existence_l2633_263307

theorem triangle_existence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end triangle_existence_l2633_263307


namespace larry_gave_brother_l2633_263357

def larry_problem (initial_amount lunch_expense final_amount : ℕ) : Prop :=
  initial_amount - lunch_expense - final_amount = 2

theorem larry_gave_brother : 
  larry_problem 22 5 15 := by sorry

end larry_gave_brother_l2633_263357


namespace distinct_plants_count_l2633_263399

/-- Represents a flower bed -/
structure FlowerBed where
  plants : Finset ℕ

/-- The total number of distinct plants in three intersecting flower beds -/
def total_distinct_plants (X Y Z : FlowerBed) : ℕ :=
  (X.plants ∪ Y.plants ∪ Z.plants).card

/-- The theorem stating the total number of distinct plants in the given scenario -/
theorem distinct_plants_count (X Y Z : FlowerBed)
  (hX : X.plants.card = 600)
  (hY : Y.plants.card = 500)
  (hZ : Z.plants.card = 400)
  (hXY : (X.plants ∩ Y.plants).card = 100)
  (hYZ : (Y.plants ∩ Z.plants).card = 80)
  (hXZ : (X.plants ∩ Z.plants).card = 120)
  (hXYZ : (X.plants ∩ Y.plants ∩ Z.plants).card = 30) :
  total_distinct_plants X Y Z = 1230 := by
  sorry


end distinct_plants_count_l2633_263399


namespace Q_times_E_times_D_l2633_263392

def Q : ℂ := 3 + 4 * Complex.I
def E : ℂ := Complex.I ^ 2
def D : ℂ := 3 - 4 * Complex.I

theorem Q_times_E_times_D : Q * E * D = -25 := by
  sorry

end Q_times_E_times_D_l2633_263392


namespace books_sold_in_garage_sale_l2633_263388

theorem books_sold_in_garage_sale 
  (initial_books : ℕ) 
  (books_given_to_friend : ℕ) 
  (remaining_books : ℕ) 
  (h1 : initial_books = 108) 
  (h2 : books_given_to_friend = 35) 
  (h3 : remaining_books = 62) :
  initial_books - books_given_to_friend - remaining_books = 11 := by
sorry

end books_sold_in_garage_sale_l2633_263388


namespace prob_sum_24_four_dice_l2633_263334

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The target sum we're aiming for -/
def target_sum : ℕ := 24

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Probability of rolling a specific number on a fair, standard six-sided die -/
def single_die_prob : ℚ := 1 / standard_die_sides

/-- Theorem: The probability of rolling a sum of 24 with four fair, standard six-sided dice is 1/1296 -/
theorem prob_sum_24_four_dice : 
  (single_die_prob ^ num_dice : ℚ) = 1 / 1296 := by
  sorry

end prob_sum_24_four_dice_l2633_263334


namespace arrangement_count_is_correct_l2633_263320

/-- The number of ways to arrange 5 students in a row with specific constraints -/
def arrangement_count : ℕ := 36

/-- A function that calculates the number of valid arrangements -/
def calculate_arrangements : ℕ :=
  let total_students : ℕ := 5
  let ab_pair_arrangements : ℕ := 3 * 2  -- 3! for AB pair and 2 others, 2! for AB swap
  let c_placement_options : ℕ := 3       -- C always has 3 valid positions
  ab_pair_arrangements * c_placement_options

/-- Theorem stating that the number of valid arrangements is 36 -/
theorem arrangement_count_is_correct : arrangement_count = calculate_arrangements := by
  sorry

end arrangement_count_is_correct_l2633_263320


namespace product_abc_value_l2633_263379

theorem product_abc_value
  (h1 : b * c * d = 65)
  (h2 : c * d * e = 750)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.6666666666666666)
  : a * b * c = 130 := by
  sorry

end product_abc_value_l2633_263379


namespace subcubes_two_plus_painted_faces_count_l2633_263343

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : ℕ
  h_side : side_length = n
  h_painted : painted_faces = 6

/-- Represents a subcube of a larger cube --/
structure Subcube (n : ℕ) where
  painted_faces : ℕ
  h_painted : painted_faces ≤ 3

/-- The number of subcubes with at least two painted faces in a painted cube --/
def subcubes_with_two_plus_painted_faces (c : Cube 4) : ℕ := sorry

/-- Theorem stating that the number of 1x1x1 subcubes with at least two painted faces
    in a 4x4x4 fully painted cube is 32 --/
theorem subcubes_two_plus_painted_faces_count (c : Cube 4) :
  subcubes_with_two_plus_painted_faces c = 32 := by sorry

end subcubes_two_plus_painted_faces_count_l2633_263343


namespace wax_remaining_after_detailing_l2633_263383

-- Define the initial amounts of wax
def waxA_initial : ℕ := 10
def waxB_initial : ℕ := 15

-- Define the amounts required for each vehicle
def waxA_car : ℕ := 4
def waxA_suv : ℕ := 6
def waxB_car : ℕ := 3
def waxB_suv : ℕ := 5

-- Define the amounts spilled
def waxA_spilled : ℕ := 3
def waxB_spilled : ℕ := 4

-- Theorem to prove
theorem wax_remaining_after_detailing :
  (waxA_initial - waxA_spilled - waxA_car) + (waxB_initial - waxB_spilled - waxB_suv) = 9 := by
  sorry

end wax_remaining_after_detailing_l2633_263383


namespace inequality_preservation_l2633_263318

theorem inequality_preservation (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, a * (2 : ℝ)^x > b * (2 : ℝ)^x := by
sorry


end inequality_preservation_l2633_263318


namespace smallest_d_for_3150_square_l2633_263329

/-- The smallest positive integer d such that 3150 * d is a perfect square is 14 -/
theorem smallest_d_for_3150_square : ∃ (n : ℕ), 
  (3150 * 14 = n ^ 2) ∧ 
  (∀ (d : ℕ), d > 0 ∧ d < 14 → ¬∃ (m : ℕ), 3150 * d = m ^ 2) := by
  sorry

end smallest_d_for_3150_square_l2633_263329


namespace tangent_slope_at_one_l2633_263344

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_differentiable : Differentiable ℝ f
axiom limit_condition : ∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, 
  |((f (x + 1) - f 1) / (2 * x)) - 3| < ε

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f 1) = 6 := by sorry

end tangent_slope_at_one_l2633_263344


namespace smallest_binary_multiple_of_60_l2633_263359

def is_binary (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_60 :
  ∃ (X : ℕ), X > 0 ∧ is_binary (60 * X) ∧
  (∀ (Y : ℕ), Y > 0 → is_binary (60 * Y) → X ≤ Y) ∧
  X = 185 := by
sorry

end smallest_binary_multiple_of_60_l2633_263359


namespace product_9_to_11_l2633_263316

theorem product_9_to_11 : (List.range 3).foldl (·*·) 1 * 9 = 990 := by
  sorry

end product_9_to_11_l2633_263316


namespace repeating_decimal_equals_fraction_l2633_263365

/-- The repeating decimal 4.252525... -/
def repeating_decimal : ℚ := 4 + 25 / 99

/-- The fraction 421/99 -/
def target_fraction : ℚ := 421 / 99

/-- Theorem stating that the repeating decimal 4.252525... is equal to 421/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end repeating_decimal_equals_fraction_l2633_263365


namespace midpoint_x_coordinate_l2633_263328

/-- Given two points P and Q, prove that if their midpoint has x-coordinate 18, then the x-coordinate of P is 6. -/
theorem midpoint_x_coordinate (a : ℝ) : 
  let P : ℝ × ℝ := (a, 2)
  let Q : ℝ × ℝ := (30, -6)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  midpoint.1 = 18 → a = 6 := by
  sorry

#check midpoint_x_coordinate

end midpoint_x_coordinate_l2633_263328


namespace prob_at_least_one_success_l2633_263313

/-- The probability of success for a single attempt -/
def p : ℝ := 0.5

/-- The number of attempts for each athlete -/
def attempts_per_athlete : ℕ := 2

/-- The total number of attempts -/
def total_attempts : ℕ := 2 * attempts_per_athlete

/-- The probability of at least one successful attempt out of the total attempts -/
theorem prob_at_least_one_success :
  1 - (1 - p) ^ total_attempts = 0.9375 := by
  sorry


end prob_at_least_one_success_l2633_263313


namespace square_diff_cube_seven_six_l2633_263386

theorem square_diff_cube_seven_six : (7^2 - 6^2)^3 = 2197 := by
  sorry

end square_diff_cube_seven_six_l2633_263386


namespace lcm_36_84_l2633_263353

theorem lcm_36_84 : Nat.lcm 36 84 = 252 := by
  sorry

end lcm_36_84_l2633_263353


namespace x_minus_q_equals_three_l2633_263342

theorem x_minus_q_equals_three (x q : ℝ) (h1 : |x - 3| = q) (h2 : x > 3) :
  x - q = 3 := by sorry

end x_minus_q_equals_three_l2633_263342


namespace cubic_equation_root_l2633_263351

theorem cubic_equation_root (k : ℚ) : 
  (∃ x : ℚ, 10 * k * x^3 - x - 9 = 0 ∧ x = -1) → k = -4/5 := by
  sorry

end cubic_equation_root_l2633_263351


namespace percentage_of_french_speakers_l2633_263303

theorem percentage_of_french_speakers (total_employees : ℝ) (h1 : total_employees > 0) :
  let men_percentage : ℝ := 70
  let women_percentage : ℝ := 100 - men_percentage
  let men_french_speakers_percentage : ℝ := 50
  let women_non_french_speakers_percentage : ℝ := 83.33333333333331
  let men : ℝ := (men_percentage / 100) * total_employees
  let women : ℝ := (women_percentage / 100) * total_employees
  let men_french_speakers : ℝ := (men_french_speakers_percentage / 100) * men
  let women_french_speakers : ℝ := (1 - women_non_french_speakers_percentage / 100) * women
  let total_french_speakers : ℝ := men_french_speakers + women_french_speakers
  (total_french_speakers / total_employees) * 100 = 40 := by
sorry

end percentage_of_french_speakers_l2633_263303


namespace sqrt_3_times_sqrt_12_l2633_263380

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l2633_263380


namespace largest_C_gap_l2633_263375

/-- Represents a square on the chessboard -/
structure Square :=
  (row : Fin 8)
  (col : Fin 8)

/-- The chessboard is an 8x8 grid of squares -/
def Chessboard := Fin 8 → Fin 8 → Square

/-- Two squares are adjacent if they share a side or vertex -/
def adjacent (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row = s2.row ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col = s2.col) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col = s2.col) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val = s2.col.val + 1)

/-- A numbering of the chessboard is a function assigning each square a unique number from 1 to 64 -/
def Numbering := Square → Fin 64

/-- A C-gap is a number g such that for every numbering, there exist two adjacent squares whose numbers differ by at least g -/
def is_C_gap (g : ℕ) : Prop :=
  ∀ (n : Numbering), ∃ (s1 s2 : Square), 
    adjacent s1 s2 ∧ |n s1 - n s2| ≥ g

/-- The theorem stating that the largest C-gap for an 8x8 chessboard is 9 -/
theorem largest_C_gap : 
  (is_C_gap 9) ∧ (∀ g : ℕ, g > 9 → ¬(is_C_gap g)) :=
sorry

end largest_C_gap_l2633_263375


namespace sin_inequality_solution_set_l2633_263397

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | Real.sin x < a} = {x : ℝ | ∃ n : ℤ, (2 * n - 1) * π - θ < x ∧ x < 2 * n * π + θ} := by
  sorry

end sin_inequality_solution_set_l2633_263397


namespace game_solution_l2633_263331

def game_result (x y z : ℚ) : Prop :=
  let a1 := x + y/3 + z/3
  let b1 := 2*y/3
  let c1 := 2*z/3
  let a2 := 2*a1/3
  let b2 := b1 + c1/3
  let c2 := 2*c1/3
  let a3 := 2*a2/3
  let b3 := 2*b2/3
  let c3 := c2 + b2/3 + a2/3
  x - a3 = 2 ∧ c3 - z = 2*z + 8 ∧ x + y + z < 1000

theorem game_solution :
  ∃ x y z : ℚ, game_result x y z ∧ x = 54 ∧ y = 162 ∧ z = 27 :=
by sorry

end game_solution_l2633_263331


namespace triangle_properties_l2633_263332

/-- Given a triangle ABC with vertices A(0,1), B(0,-1), and C(-2,1) -/
def triangle_ABC : Set (ℝ × ℝ) := {(0, 1), (0, -1), (-2, 1)}

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure CircleEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Theorem stating the equations of altitude, midline, and circumcircle -/
theorem triangle_properties (ABC : Set (ℝ × ℝ)) 
  (h : ABC = triangle_ABC) : 
  ∃ (altitude_eq : LineEquation) 
    (midline_eq : LineEquation) 
    (circumcircle_eq : CircleEquation),
  altitude_eq = ⟨1, -1, 1⟩ ∧ 
  midline_eq = ⟨1, 0, 1⟩ ∧
  circumcircle_eq = ⟨2, 0, -1⟩ := by
  sorry

end triangle_properties_l2633_263332


namespace quadratic_increasing_after_vertex_l2633_263327

def f (x : ℝ) : ℝ := (x - 1)^2 + 5

theorem quadratic_increasing_after_vertex (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > x1) : 
  f x2 > f x1 := by
  sorry

end quadratic_increasing_after_vertex_l2633_263327


namespace no_nonzero_integer_solution_l2633_263361

theorem no_nonzero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end no_nonzero_integer_solution_l2633_263361


namespace inverse_f_at_negative_three_over_128_l2633_263393

noncomputable def f (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_f_at_negative_three_over_128 :
  f⁻¹ (-3/128) = (29/32)^(1/7) := by
  sorry

end inverse_f_at_negative_three_over_128_l2633_263393


namespace equation_solution_l2633_263338

theorem equation_solution (x : ℂ) : 
  (x^2 + 4*x + 8) / (x - 3) = 2 ↔ x = -1 + (7*Real.sqrt 2/2)*I ∨ x = -1 - (7*Real.sqrt 2/2)*I :=
sorry

end equation_solution_l2633_263338


namespace negation_of_proposition_l2633_263326

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, Real.log (x^2 + 1) > 0) ↔ (∃ x : ℝ, Real.log (x^2 + 1) ≤ 0) :=
by sorry

end negation_of_proposition_l2633_263326


namespace part_one_part_two_l2633_263387

/-- The function f(x) defined in the problem -/
def f (a c x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

/-- Part 1 of the problem -/
theorem part_one (a : ℝ) :
  f a 19 1 > 0 ↔ -2 < a ∧ a < 8 := by sorry

/-- Part 2 of the problem -/
theorem part_two (a c : ℝ) :
  (∀ x : ℝ, f a c x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9) := by sorry

end part_one_part_two_l2633_263387


namespace break_even_circus_production_l2633_263322

/-- Calculates the number of sold-out performances needed to break even for a circus production -/
def break_even_performances (overhead : ℕ) (production_cost : ℕ) (revenue : ℕ) : ℕ :=
  let total_cost (x : ℕ) := overhead + production_cost * x
  let total_revenue (x : ℕ) := revenue * x
  (overhead / (revenue - production_cost) : ℕ)

/-- Proves that 9 sold-out performances are needed to break even given the specific costs and revenue -/
theorem break_even_circus_production :
  break_even_performances 81000 7000 16000 = 9 := by
  sorry

end break_even_circus_production_l2633_263322


namespace quadratic_factorization_l2633_263349

theorem quadratic_factorization (a : ℝ) : a^2 - 6*a + 9 = (a - 3)^2 := by
  sorry

end quadratic_factorization_l2633_263349


namespace vacation_tents_l2633_263372

/-- Calculates the number of tents needed given the total number of people,
    the number of people the house can accommodate, and the capacity of each tent. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (tent_capacity : ℕ) : ℕ :=
  ((total_people - house_capacity) + tent_capacity - 1) / tent_capacity

theorem vacation_tents :
  tents_needed 14 4 2 = 5 := by
  sorry

end vacation_tents_l2633_263372


namespace valid_numbers_characterization_l2633_263352

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧  -- even
  (n / 10 + n % 10) > 6 ∧  -- sum of digits greater than 6
  (n / 10) ≥ (n % 10 + 4)  -- tens digit at least 4 greater than units digit

theorem valid_numbers_characterization :
  {n : ℕ | is_valid_number n} = {70, 80, 90, 62, 72, 82, 92, 84, 94} :=
by sorry

end valid_numbers_characterization_l2633_263352


namespace increase_per_page_correct_l2633_263363

/-- The increase in drawings per page -/
def increase_per_page : ℕ := 5

/-- The number of drawings on the first page -/
def first_page_drawings : ℕ := 5

/-- The number of pages we're considering -/
def num_pages : ℕ := 5

/-- The total number of drawings on the first five pages -/
def total_drawings : ℕ := 75

/-- Theorem stating that the increase per page is correct -/
theorem increase_per_page_correct : 
  first_page_drawings + 
  (first_page_drawings + increase_per_page) + 
  (first_page_drawings + 2 * increase_per_page) + 
  (first_page_drawings + 3 * increase_per_page) + 
  (first_page_drawings + 4 * increase_per_page) = total_drawings :=
by sorry

end increase_per_page_correct_l2633_263363


namespace grandma_inheritance_l2633_263369

theorem grandma_inheritance (total : ℕ) (shelby_share : ℕ) (remaining_grandchildren : ℕ) :
  total = 124600 →
  shelby_share = total / 2 →
  remaining_grandchildren = 10 →
  (total - shelby_share) / remaining_grandchildren = 6230 :=
by sorry

end grandma_inheritance_l2633_263369


namespace power_five_sum_greater_than_mixed_products_l2633_263311

theorem power_five_sum_greater_than_mixed_products {a b : ℝ} 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^5 + b^5 > a^3 * b^2 + a^2 * b^3 := by
  sorry

end power_five_sum_greater_than_mixed_products_l2633_263311


namespace days_to_build_floor_l2633_263348

-- Define the daily pay rate for a builder
def builder_daily_pay : ℕ := 100

-- Define the total cost for the project
def total_project_cost : ℕ := 270000

-- Define the number of builders for the project
def project_builders : ℕ := 6

-- Define the number of houses in the project
def project_houses : ℕ := 5

-- Define the number of floors per house in the project
def floors_per_house : ℕ := 6

-- Theorem to prove
theorem days_to_build_floor (builders : ℕ) (days : ℕ) : 
  builders = 3 → days = 30 → 
  (builders * builder_daily_pay * days = 
   total_project_cost * builders / project_builders / 
   (project_houses * floors_per_house)) := by sorry

end days_to_build_floor_l2633_263348


namespace first_tap_fill_time_l2633_263390

/-- Represents the time (in hours) it takes for the first tap to fill the cistern -/
def T : ℝ := 3

/-- Represents the time (in hours) it takes for the second tap to empty the cistern -/
def empty_time : ℝ := 8

/-- Represents the time (in hours) it takes to fill the cistern when both taps are open -/
def both_open_time : ℝ := 4.8

/-- Proves that T is the correct time for the first tap to fill the cistern -/
theorem first_tap_fill_time :
  (1 / T - 1 / empty_time = 1 / both_open_time) ∧ T > 0 := by
  sorry

end first_tap_fill_time_l2633_263390


namespace exam_score_deviation_l2633_263346

/-- Given an exam with mean score 74 and standard deviation σ,
    prove that 58 is 2σ below the mean when 98 is 3σ above the mean. -/
theorem exam_score_deviation (σ : ℝ) : 
  (74 + 3 * σ = 98) → (74 - 2 * σ = 58) := by
  sorry

end exam_score_deviation_l2633_263346


namespace sequence_non_positive_l2633_263389

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k, k < n → a k.pred - 2 * a k + a k.succ ≥ 0) :
  ∀ i, i ≤ n → a i ≤ 0 := by
  sorry

end sequence_non_positive_l2633_263389


namespace coefficient_x4_is_1120_l2633_263382

open BigOperators

/-- The coefficient of x^4 in the expansion of (x^2 + 2/x)^8 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 8 4) * 2^4

/-- Theorem stating that the coefficient of x^4 in (x^2 + 2/x)^8 is 1120 -/
theorem coefficient_x4_is_1120 : coefficient_x4 = 1120 := by
  sorry

#eval coefficient_x4  -- This will evaluate the expression and show the result

end coefficient_x4_is_1120_l2633_263382
