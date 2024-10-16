import Mathlib

namespace NUMINAMATH_CALUDE_dancing_preference_fraction_l1251_125195

def total_students : ℕ := 200
def like_dancing_percent : ℚ := 70 / 100
def dislike_dancing_percent : ℚ := 30 / 100
def honest_like_percent : ℚ := 85 / 100
def dishonest_like_percent : ℚ := 15 / 100
def honest_dislike_percent : ℚ := 80 / 100
def dishonest_dislike_percent : ℚ := 20 / 100

theorem dancing_preference_fraction :
  let like_dancing := (like_dancing_percent * total_students : ℚ)
  let dislike_dancing := (dislike_dancing_percent * total_students : ℚ)
  let say_like := (honest_like_percent * like_dancing + dishonest_dislike_percent * dislike_dancing : ℚ)
  let actually_dislike_say_like := (dishonest_dislike_percent * dislike_dancing : ℚ)
  actually_dislike_say_like / say_like = 12 / 131 := by
  sorry

end NUMINAMATH_CALUDE_dancing_preference_fraction_l1251_125195


namespace NUMINAMATH_CALUDE_more_philosophers_than_mathematicians_l1251_125133

theorem more_philosophers_than_mathematicians
  (m p : ℕ+)
  (h : (m : ℚ) / 7 = (p : ℚ) / 9) :
  p > m :=
sorry

end NUMINAMATH_CALUDE_more_philosophers_than_mathematicians_l1251_125133


namespace NUMINAMATH_CALUDE_gift_box_weight_l1251_125186

/-- The weight of an empty gift box -/
def empty_box_weight (num_tangerines : ℕ) (tangerine_weight : ℝ) (total_weight : ℝ) : ℝ :=
  total_weight - (num_tangerines : ℝ) * tangerine_weight

/-- Theorem: The weight of the empty gift box is 0.46 kg -/
theorem gift_box_weight :
  empty_box_weight 30 0.36 11.26 = 0.46 := by sorry

end NUMINAMATH_CALUDE_gift_box_weight_l1251_125186


namespace NUMINAMATH_CALUDE_probability_alice_has_ball_after_three_turns_l1251_125150

-- Define the probabilities
def alice_pass : ℚ := 1/3
def alice_keep : ℚ := 2/3
def bob_pass : ℚ := 1/4
def bob_keep : ℚ := 3/4

-- Define the game state after three turns
def alice_has_ball_after_three_turns : ℚ :=
  alice_keep^3 + alice_pass * bob_pass * alice_keep + alice_keep * alice_pass * bob_pass

-- Theorem statement
theorem probability_alice_has_ball_after_three_turns :
  alice_has_ball_after_three_turns = 11/27 := by sorry

end NUMINAMATH_CALUDE_probability_alice_has_ball_after_three_turns_l1251_125150


namespace NUMINAMATH_CALUDE_hexagon_circle_comparison_l1251_125139

theorem hexagon_circle_comparison : ∃ (h r : ℝ),
  h > 0 ∧ r > 0 ∧
  (3 * Real.sqrt 3 / 2) * h^2 = 6 * h ∧  -- Hexagon area equals perimeter
  π * r^2 = 2 * π * r ∧                  -- Circle area equals perimeter
  (Real.sqrt 3 / 2) * h = r ∧            -- Apothem equals radius
  r = 2 := by sorry

end NUMINAMATH_CALUDE_hexagon_circle_comparison_l1251_125139


namespace NUMINAMATH_CALUDE_square_area_error_l1251_125107

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1251_125107


namespace NUMINAMATH_CALUDE_max_m_value_l1251_125119

theorem max_m_value (m : ℝ) (h1 : m > 1) 
  (h2 : ∃ x : ℝ, x ∈ Set.Icc (-2) 0 ∧ x^2 + 2*m*x + m^2 - m ≤ 0) : 
  (∀ n : ℝ, (n > 1 ∧ ∃ y : ℝ, y ∈ Set.Icc (-2) 0 ∧ y^2 + 2*n*y + n^2 - n ≤ 0) → n ≤ m) →
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l1251_125119


namespace NUMINAMATH_CALUDE_largest_common_divisor_342_285_l1251_125147

theorem largest_common_divisor_342_285 : ∃ (n : ℕ), n > 0 ∧ n ∣ 342 ∧ n ∣ 285 ∧ ∀ (m : ℕ), m > n → (m ∣ 342 ∧ m ∣ 285 → False) :=
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_342_285_l1251_125147


namespace NUMINAMATH_CALUDE_product_evaluation_l1251_125137

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n^2 + 1) = 7200 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1251_125137


namespace NUMINAMATH_CALUDE_bryan_mineral_samples_l1251_125180

/-- The number of mineral samples per shelf -/
def samples_per_shelf : ℕ := 65

/-- The number of shelves -/
def number_of_shelves : ℕ := 7

/-- The total number of mineral samples -/
def total_samples : ℕ := samples_per_shelf * number_of_shelves

theorem bryan_mineral_samples :
  total_samples = 455 :=
sorry

end NUMINAMATH_CALUDE_bryan_mineral_samples_l1251_125180


namespace NUMINAMATH_CALUDE_book_pages_count_l1251_125117

/-- The number of pages Lance read on the first day -/
def pages_day1 : ℕ := 35

/-- The number of pages Lance read on the second day -/
def pages_day2 : ℕ := pages_day1 - 5

/-- The number of pages Lance will read on the third day -/
def pages_day3 : ℕ := 35

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_day1 + pages_day2 + pages_day3

theorem book_pages_count : total_pages = 100 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1251_125117


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1251_125114

theorem unknown_number_proof (x : ℝ) : 1.75 * x = 63 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1251_125114


namespace NUMINAMATH_CALUDE_units_digit_of_power_of_three_l1251_125140

theorem units_digit_of_power_of_three (n : ℕ) : (3^(4*n + 2) % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_of_three_l1251_125140


namespace NUMINAMATH_CALUDE_smallest_sum_of_c_and_d_l1251_125122

theorem smallest_sum_of_c_and_d (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ (4*Real.sqrt 3 + 4/3) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_c_and_d_l1251_125122


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l1251_125128

theorem least_sum_of_bases (a b : ℕ+) : 
  (6 * a.val + 3 = 3 * b.val + 6) →
  (∀ (a' b' : ℕ+), (6 * a'.val + 3 = 3 * b'.val + 6) → (a'.val + b'.val ≥ a.val + b.val)) →
  a.val + b.val = 20 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l1251_125128


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1251_125125

def U : Set Nat := {1,2,3,4,5,6,7,8}
def M : Set Nat := {1,3,5,7}
def N : Set Nat := {5,6,7}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1251_125125


namespace NUMINAMATH_CALUDE_trivia_team_size_l1251_125153

/-- The original number of members in a trivia team -/
def original_members (absent : ℕ) (points_per_member : ℕ) (total_points : ℕ) : ℕ :=
  (total_points / points_per_member) + absent

theorem trivia_team_size :
  original_members 3 2 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_size_l1251_125153


namespace NUMINAMATH_CALUDE_inverse_function_b_value_l1251_125161

theorem inverse_function_b_value (f : ℝ → ℝ) (b : ℝ) :
  (∀ x, f x = 5 - b * x) →
  (∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f ∧ g (-3) = 3) →
  b = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_b_value_l1251_125161


namespace NUMINAMATH_CALUDE_decreasing_function_condition_l1251_125116

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x

-- State the theorem
theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (1 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_condition_l1251_125116


namespace NUMINAMATH_CALUDE_train_crossing_tree_time_l1251_125169

/-- Given a train and platform with specified lengths and time to pass the platform,
    calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_tree_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 1000)
  (h3 : time_to_pass_platform = 146.67) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_tree_time_l1251_125169


namespace NUMINAMATH_CALUDE_at_least_one_by_cellini_son_not_both_by_cellini_not_both_by_other_l1251_125105

-- Define the possible makers of the caskets
inductive Maker
| Cellini
| CelliniSon
| Other

-- Define the caskets
structure Casket where
  material : String
  inscription : String
  maker : Maker

-- Define the problem setup
def goldenCasket : Casket := {
  material := "golden"
  inscription := "The silver casket was made by Cellini."
  maker := Maker.Other -- Initial assumption, will be proved
}

def silverCasket : Casket := {
  material := "silver"
  inscription := "The golden casket was made by someone other than Cellini."
  maker := Maker.Other -- Initial assumption, will be proved
}

-- The main theorem to prove
theorem at_least_one_by_cellini_son (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  g.maker = Maker.CelliniSon ∨ s.maker = Maker.CelliniSon := by
  sorry

-- Additional helper theorems if needed
theorem not_both_by_cellini (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  ¬(g.maker = Maker.Cellini ∧ s.maker = Maker.Cellini) := by
  sorry

theorem not_both_by_other (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  ¬(g.maker = Maker.Other ∧ s.maker = Maker.Other) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_by_cellini_son_not_both_by_cellini_not_both_by_other_l1251_125105


namespace NUMINAMATH_CALUDE_function_solution_set_l1251_125170

/-- Given a function f(x) = 2x / (x^2 + 6), if the solution set of f(x) > k 
    is {x | x < -3 or x > -2}, then k = -2/5 -/
theorem function_solution_set (f : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 2 * x / (x^2 + 6)) →
  (∀ x, f x > k ↔ x < -3 ∨ x > -2) →
  k = -2/5 := by
sorry

end NUMINAMATH_CALUDE_function_solution_set_l1251_125170


namespace NUMINAMATH_CALUDE_square_area_relation_l1251_125158

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := a + b
  let area_I := (diagonal_I^2) / 2
  let area_II := 2 * area_I
  area_II = (a + b)^2 := by
sorry

end NUMINAMATH_CALUDE_square_area_relation_l1251_125158


namespace NUMINAMATH_CALUDE_modulus_z_l1251_125165

theorem modulus_z (r k : ℝ) (z : ℂ) 
  (hr : |r| < 2) 
  (hk : |k| < 3) 
  (hz : z + k * z⁻¹ = r) : 
  Complex.abs z = Real.sqrt ((r^2 - 2*k) / 2) := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_l1251_125165


namespace NUMINAMATH_CALUDE_vectors_are_coplanar_l1251_125174

/-- Prove that vectors a, b, and c are coplanar -/
theorem vectors_are_coplanar :
  let a : ℝ × ℝ × ℝ := (1, 2, -3)
  let b : ℝ × ℝ × ℝ := (-2, -4, 6)
  let c : ℝ × ℝ × ℝ := (1, 0, 5)
  ∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_vectors_are_coplanar_l1251_125174


namespace NUMINAMATH_CALUDE_unicorn_journey_flowers_l1251_125157

/-- Calculates the number of flowers that bloom when unicorns walk across a forest -/
def unicorn_flowers (num_unicorns : ℕ) (journey_km : ℕ) (step_meters : ℕ) (flowers_per_step : ℕ) : ℕ :=
  let journey_meters := journey_km * 1000
  let num_steps := journey_meters / step_meters
  let flowers_per_unicorn := num_steps * flowers_per_step
  num_unicorns * flowers_per_unicorn

/-- Theorem stating that 6 unicorns walking 9 km with 3-meter steps, each causing 4 flowers to bloom, results in 72000 flowers -/
theorem unicorn_journey_flowers :
  unicorn_flowers 6 9 3 4 = 72000 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_journey_flowers_l1251_125157


namespace NUMINAMATH_CALUDE_complex_multiplication_l1251_125154

theorem complex_multiplication : ∃ (i : ℂ), i^2 = -1 ∧ (3 - 4*i) * (-7 + 2*i) = -13 + 34*i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1251_125154


namespace NUMINAMATH_CALUDE_three_percentage_problem_l1251_125120

theorem three_percentage_problem (x y : ℝ) 
  (h1 : 3 = 0.25 * x) 
  (h2 : 3 = 0.50 * y) : 
  x - y = 6 ∧ x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_three_percentage_problem_l1251_125120


namespace NUMINAMATH_CALUDE_sarah_cookies_count_l1251_125184

/-- The number of cookies Sarah took -/
def cookies_sarah_took (total_cookies : ℕ) (num_neighbors : ℕ) (cookies_per_neighbor : ℕ) (cookies_left : ℕ) : ℕ :=
  total_cookies - cookies_left - (num_neighbors - 1) * cookies_per_neighbor

theorem sarah_cookies_count :
  cookies_sarah_took 150 15 10 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sarah_cookies_count_l1251_125184


namespace NUMINAMATH_CALUDE_tangency_condition_l1251_125101

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 4

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 4

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
  ∀ x' y' : ℝ, parabola x' y' ∧ hyperbola m x' y' → (x', y') = (x, y)

/-- The theorem statement -/
theorem tangency_condition (m : ℝ) :
  are_tangent m ↔ m = 8 + 4 * Real.sqrt 3 ∨ m = 8 - 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tangency_condition_l1251_125101


namespace NUMINAMATH_CALUDE_rectangle_area_l1251_125188

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 2025
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_breadth : ℝ := b
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area = 18 * b :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1251_125188


namespace NUMINAMATH_CALUDE_triangle_has_inside_altitude_l1251_125132

-- Define a triangle
def Triangle : Type := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

-- Define an altitude of a triangle
def Altitude (t : Triangle) : Type := ℝ × ℝ × ℝ × ℝ

-- Define what it means for an altitude to be inside a triangle
def IsInside (a : Altitude t) (t : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_has_inside_altitude (t : Triangle) : 
  ∃ (a : Altitude t), IsInside a t := sorry

end NUMINAMATH_CALUDE_triangle_has_inside_altitude_l1251_125132


namespace NUMINAMATH_CALUDE_factor_x4_minus_81_l1251_125130

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_81_l1251_125130


namespace NUMINAMATH_CALUDE_jeff_phone_storage_capacity_l1251_125156

theorem jeff_phone_storage_capacity :
  let storage_used : ℕ := 4
  let song_size : ℕ := 30
  let max_songs : ℕ := 400
  let mb_per_gb : ℕ := 1000
  let total_storage : ℕ := 
    storage_used + (song_size * max_songs) / mb_per_gb
  total_storage = 16 := by
  sorry

end NUMINAMATH_CALUDE_jeff_phone_storage_capacity_l1251_125156


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1251_125124

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- Define the statement to be proved
theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, q x a → p x) ∧ (∃ x, p x ∧ ¬q x a) → a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1251_125124


namespace NUMINAMATH_CALUDE_school_student_count_l1251_125129

/-- The number of classrooms in the school -/
def num_classrooms : ℕ := 24

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 5

/-- The total number of students in the school -/
def total_students : ℕ := num_classrooms * students_per_classroom

theorem school_student_count : total_students = 120 := by
  sorry

end NUMINAMATH_CALUDE_school_student_count_l1251_125129


namespace NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l1251_125163

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 9*x^2 + 18*x + 38

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, 
    (p.1 = g p.2 ∧ p.2 = g p.1) ∧ 
    p = (-2, -2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l1251_125163


namespace NUMINAMATH_CALUDE_incorrect_value_calculation_l1251_125141

/-- Given a set of values with an incorrect mean due to a copying error,
    calculate the incorrect value that was used. -/
theorem incorrect_value_calculation
  (n : ℕ)
  (initial_mean correct_mean : ℚ)
  (correct_value : ℚ)
  (h_n : n = 30)
  (h_initial_mean : initial_mean = 250)
  (h_correct_mean : correct_mean = 251)
  (h_correct_value : correct_value = 165) :
  ∃ (incorrect_value : ℚ),
    incorrect_value = 195 ∧
    n * correct_mean = n * initial_mean - correct_value + incorrect_value :=
by sorry

end NUMINAMATH_CALUDE_incorrect_value_calculation_l1251_125141


namespace NUMINAMATH_CALUDE_degree_of_h_l1251_125192

/-- Given a polynomial f(x) = -5x^5 + 2x^4 + 7x - 8 and a polynomial h(x) such that
    the degree of f(x) - h(x) is 3, prove that the degree of h(x) is 5. -/
theorem degree_of_h (f h : Polynomial ℝ) : 
  f = -5 * X^5 + 2 * X^4 + 7 * X - 8 →
  Polynomial.degree (f - h) = 3 →
  Polynomial.degree h = 5 :=
by sorry

end NUMINAMATH_CALUDE_degree_of_h_l1251_125192


namespace NUMINAMATH_CALUDE_fraction_of_fraction_l1251_125112

theorem fraction_of_fraction : 
  (2 / 5 : ℚ) * (1 / 3 : ℚ) / (3 / 4 : ℚ) = 8 / 45 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_l1251_125112


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l1251_125113

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 9

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 8

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := 22

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := total_highlighters - (pink_highlighters + yellow_highlighters)

theorem blue_highlighters_count : blue_highlighters = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_highlighters_count_l1251_125113


namespace NUMINAMATH_CALUDE_fred_seashells_l1251_125143

theorem fred_seashells (initial_seashells : ℕ) (given_seashells : ℕ) : 
  initial_seashells = 47 → given_seashells = 25 → initial_seashells - given_seashells = 22 := by
  sorry

end NUMINAMATH_CALUDE_fred_seashells_l1251_125143


namespace NUMINAMATH_CALUDE_cube_sum_is_18_l1251_125135

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 9

/-- The sum of numbers on a face of the cube -/
def face_sum (arrangement : CubeArrangement) (face : Finset (Fin 8)) : ℕ :=
  (face.sum fun v => arrangement v).val + 1

/-- Predicate for a valid cube arrangement -/
def is_valid_arrangement (arrangement : CubeArrangement) : Prop :=
  ∀ (face1 face2 : Finset (Fin 8)), face1.card = 4 → face2.card = 4 → 
    face_sum arrangement face1 = face_sum arrangement face2

theorem cube_sum_is_18 :
  ∀ (arrangement : CubeArrangement), is_valid_arrangement arrangement →
    ∃ (face : Finset (Fin 8)), face.card = 4 ∧ face_sum arrangement face = 18 :=
sorry

end NUMINAMATH_CALUDE_cube_sum_is_18_l1251_125135


namespace NUMINAMATH_CALUDE_problem_polygon_area_l1251_125126

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Represents a polygon composed of rectangles -/
structure Polygon where
  rectangles : List Rectangle

/-- Calculates the total area of a polygon -/
def polygonArea (p : Polygon) : ℕ :=
  p.rectangles.map rectangleArea |>.sum

/-- The polygon in the problem -/
def problemPolygon : Polygon :=
  { rectangles := [
      { width := 2, height := 2 },  -- 2x2 square
      { width := 1, height := 2 },  -- 1x2 rectangle
      { width := 1, height := 2 }   -- 1x2 rectangle
    ] 
  }

theorem problem_polygon_area : polygonArea problemPolygon = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l1251_125126


namespace NUMINAMATH_CALUDE_complex_norm_squared_l1251_125191

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 5 + 4*Complex.I) : 
  Complex.abs z^2 = 41/10 := by
sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l1251_125191


namespace NUMINAMATH_CALUDE_pi_digits_ratio_l1251_125197

/-- The number of digits of pi memorized by Carlos -/
def carlos_digits : ℕ := sorry

/-- The number of digits of pi memorized by Sam -/
def sam_digits : ℕ := sorry

/-- The number of digits of pi memorized by Mina -/
def mina_digits : ℕ := sorry

/-- The ratio of digits memorized by Mina to Carlos -/
def mina_carlos_ratio : ℚ := sorry

theorem pi_digits_ratio :
  sam_digits = carlos_digits + 6 ∧
  mina_digits = 24 ∧
  sam_digits = 10 ∧
  ∃ k : ℕ, mina_digits = k * carlos_digits →
  mina_carlos_ratio = 6 := by sorry

end NUMINAMATH_CALUDE_pi_digits_ratio_l1251_125197


namespace NUMINAMATH_CALUDE_mode_of_interest_groups_l1251_125172

def interest_groups : List Nat := [4, 7, 5, 4, 6, 4, 5]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_interest_groups :
  mode interest_groups = 4 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_interest_groups_l1251_125172


namespace NUMINAMATH_CALUDE_percentage_equation_l1251_125194

theorem percentage_equation (x : ℝ) : 0.65 * x = 0.20 * 552.50 → x = 170 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l1251_125194


namespace NUMINAMATH_CALUDE_last_remaining_number_l1251_125189

/-- The function that determines the next position of a number after one round of erasure -/
def nextPosition (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 * (n / 3)
  else if n % 3 = 2 then 2 * (n / 3) + 1
  else 0  -- This case (n % 3 = 1) corresponds to erased numbers

/-- The function that determines the original position given a final position -/
def originalPosition (finalPos : ℕ) : ℕ :=
  if finalPos = 1 then 1458 else 0  -- We only care about the winning position

/-- The theorem stating that 1458 is the last remaining number -/
theorem last_remaining_number :
  ∃ (n : ℕ), n ≤ 2002 ∧ 
  (∀ (m : ℕ), m ≤ 2002 → m ≠ n → 
    ∃ (k : ℕ), originalPosition m = 3 * k + 1 ∨ 
    ∃ (j : ℕ), nextPosition (originalPosition m) = originalPosition j ∧ j < m) ∧
  originalPosition n = n ∧ n = 1458 :=
sorry

end NUMINAMATH_CALUDE_last_remaining_number_l1251_125189


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1251_125151

/-- For an ellipse with given eccentricity and focal length, prove the length of its minor axis. -/
theorem ellipse_minor_axis_length 
  (e : ℝ) -- eccentricity
  (f : ℝ) -- focal length
  (h_e : e = 1/2)
  (h_f : f = 2) :
  ∃ (minor_axis : ℝ), minor_axis = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1251_125151


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1251_125127

theorem complex_fraction_equality : (2 : ℂ) / (1 - I) = 1 + I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1251_125127


namespace NUMINAMATH_CALUDE_summer_program_undergrads_l1251_125109

theorem summer_program_undergrads (total_students : ℕ) 
  (coding_team_ugrad_percent : ℚ) (coding_team_grad_percent : ℚ) :
  total_students = 36 →
  coding_team_ugrad_percent = 1/5 →
  coding_team_grad_percent = 1/4 →
  ∃ (undergrads grads coding_team_size : ℕ),
    undergrads + grads = total_students ∧
    coding_team_size * 2 = coding_team_ugrad_percent * undergrads + coding_team_grad_percent * grads ∧
    undergrads = 20 := by
  sorry

end NUMINAMATH_CALUDE_summer_program_undergrads_l1251_125109


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l1251_125131

theorem lcm_of_20_45_75 : Nat.lcm (Nat.lcm 20 45) 75 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l1251_125131


namespace NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1251_125162

theorem geometric_progression_common_ratio 
  (x y z w : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0) 
  (h_geom_prog : ∃ (a r : ℝ), r ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2 ∧ 
    w * (x - y) = a * r^3) : 
  ∃ r : ℝ, r^3 + r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1251_125162


namespace NUMINAMATH_CALUDE_lewis_harvest_weeks_l1251_125159

/-- The number of weeks Lewis works during the harvest -/
def harvest_weeks (total_earnings weekly_earnings : ℕ) : ℕ :=
  total_earnings / weekly_earnings

/-- Proof that Lewis works 5 weeks during the harvest -/
theorem lewis_harvest_weeks :
  harvest_weeks 460 92 = 5 := by
  sorry

end NUMINAMATH_CALUDE_lewis_harvest_weeks_l1251_125159


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_implies_both_rational_l1251_125166

theorem sqrt_sum_rational_implies_both_rational 
  (a b : ℚ) 
  (h : ∃ (q : ℚ), q = Real.sqrt a + Real.sqrt b) : 
  (∃ (r : ℚ), r = Real.sqrt a) ∧ (∃ (s : ℚ), s = Real.sqrt b) := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_implies_both_rational_l1251_125166


namespace NUMINAMATH_CALUDE_area_of_special_quadrilateral_in_cube_l1251_125136

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents a quadrilateral in 3D space -/
structure Quadrilateral where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

/-- Calculate the area of a quadrilateral given its vertices -/
def areaOfQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is a vertex of a cube -/
def isVertexOfCube (p : Point3D) (cube : Cube) : Prop := sorry

/-- Check if a point is a midpoint of an edge of a cube -/
def isMidpointOfCubeEdge (p : Point3D) (cube : Cube) : Prop := sorry

/-- Check if two points are diagonally opposite vertices of a cube -/
def areDiagonallyOppositeVertices (p1 p2 : Point3D) (cube : Cube) : Prop := sorry

/-- Main theorem -/
theorem area_of_special_quadrilateral_in_cube (cube : Cube) (a b c d : Point3D) :
  cube.sideLength = 2 →
  isVertexOfCube a cube →
  isVertexOfCube c cube →
  isMidpointOfCubeEdge b cube →
  isMidpointOfCubeEdge d cube →
  areDiagonallyOppositeVertices a c cube →
  areaOfQuadrilateral ⟨a, b, c, d⟩ = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_quadrilateral_in_cube_l1251_125136


namespace NUMINAMATH_CALUDE_red_light_probability_l1251_125144

-- Define the durations of each light
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

-- Define the total cycle time
def total_cycle_time : ℕ := red_duration + yellow_duration + green_duration

-- Define the probability of seeing a red light
def probability_red_light : ℚ := red_duration / total_cycle_time

-- Theorem statement
theorem red_light_probability :
  probability_red_light = 30 / 75 :=
by sorry

end NUMINAMATH_CALUDE_red_light_probability_l1251_125144


namespace NUMINAMATH_CALUDE_speed_difference_proof_l1251_125106

/-- Prove that given a distance of 8 miles, if person A travels for 40 minutes
    and person B travels for 1 hour, the difference in their average speeds is 4 mph. -/
theorem speed_difference_proof (distance : ℝ) (time_A : ℝ) (time_B : ℝ) :
  distance = 8 →
  time_A = 40 / 60 →
  time_B = 1 →
  (distance / time_A) - (distance / time_B) = 4 := by
sorry

end NUMINAMATH_CALUDE_speed_difference_proof_l1251_125106


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1251_125183

theorem cube_root_of_negative_eight : 
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1251_125183


namespace NUMINAMATH_CALUDE_lee_test_probability_l1251_125178

theorem lee_test_probability (p_physics : ℝ) (p_chem_given_no_physics : ℝ) 
  (h1 : p_physics = 5/8)
  (h2 : p_chem_given_no_physics = 2/3) :
  (1 - p_physics) * p_chem_given_no_physics = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_lee_test_probability_l1251_125178


namespace NUMINAMATH_CALUDE_runners_meet_at_6000_seconds_l1251_125104

/-- The time at which three runners meet again on a circular track -/
def runners_meeting_time (track_length : ℝ) (speed1 speed2 speed3 : ℝ) : ℝ :=
  let t := 6000
  t

/-- Theorem stating that the runners meet after 6000 seconds -/
theorem runners_meet_at_6000_seconds (track_length : ℝ) (speed1 speed2 speed3 : ℝ)
  (h_track : track_length = 600)
  (h_speed1 : speed1 = 4.4)
  (h_speed2 : speed2 = 4.9)
  (h_speed3 : speed3 = 5.1) :
  runners_meeting_time track_length speed1 speed2 speed3 = 6000 := by
  sorry

#check runners_meet_at_6000_seconds

end NUMINAMATH_CALUDE_runners_meet_at_6000_seconds_l1251_125104


namespace NUMINAMATH_CALUDE_unique_n_for_consecutive_prime_products_l1251_125115

def x (n : ℕ) : ℕ := 2 * n + 49

def is_product_of_two_distinct_primes_with_same_difference (m : ℕ) : Prop :=
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ ∃ (d : ℕ), m = p * q ∧ q - p = d

theorem unique_n_for_consecutive_prime_products : 
  ∃! (n : ℕ), n > 0 ∧ 
    is_product_of_two_distinct_primes_with_same_difference (x n) ∧
    is_product_of_two_distinct_primes_with_same_difference (x (n + 1)) ∧
    n = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_n_for_consecutive_prime_products_l1251_125115


namespace NUMINAMATH_CALUDE_paulines_potatoes_l1251_125176

/-- Represents Pauline's garden -/
structure Garden where
  rows : Nat
  spaces_per_row : Nat
  tomato_kinds : Nat
  tomatoes_per_kind : Nat
  cucumber_kinds : Nat
  cucumbers_per_kind : Nat
  empty_spaces : Nat

/-- Calculates the number of potatoes in the garden -/
def count_potatoes (g : Garden) : Nat :=
  let total_spaces := g.rows * g.spaces_per_row
  let tomatoes := g.tomato_kinds * g.tomatoes_per_kind
  let cucumbers := g.cucumber_kinds * g.cucumbers_per_kind
  let occupied_spaces := total_spaces - g.empty_spaces
  occupied_spaces - (tomatoes + cucumbers)

/-- Theorem stating the number of potatoes in Pauline's garden -/
theorem paulines_potatoes :
  let g : Garden := {
    rows := 10,
    spaces_per_row := 15,
    tomato_kinds := 3,
    tomatoes_per_kind := 5,
    cucumber_kinds := 5,
    cucumbers_per_kind := 4,
    empty_spaces := 85
  }
  count_potatoes g = 30 := by sorry

end NUMINAMATH_CALUDE_paulines_potatoes_l1251_125176


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_sum_l1251_125181

-- Define a parallelogram
structure Parallelogram (V : Type*) [AddCommGroup V] :=
  (A B C D : V)
  (parallelogram_property : (C - B) = (D - A) ∧ (D - C) = (B - A))

-- Theorem statement
theorem parallelogram_diagonal_sum 
  {V : Type*} [AddCommGroup V] (ABCD : Parallelogram V) :
  ABCD.B - ABCD.A + (ABCD.D - ABCD.A) = ABCD.C - ABCD.A :=
sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_sum_l1251_125181


namespace NUMINAMATH_CALUDE_min_shortest_side_is_12_l1251_125152

/-- Represents a triangle with integer side lengths and given altitudes -/
structure Triangle where
  -- Side lengths
  AB : ℕ
  BC : ℕ
  CA : ℕ
  -- Altitude lengths
  AD : ℕ
  BE : ℕ
  CF : ℕ
  -- Conditions
  altitude_AD : AD = 3
  altitude_BE : BE = 4
  altitude_CF : CF = 5
  -- Area equality conditions
  area_eq_1 : BC * AD = CA * BE
  area_eq_2 : CA * BE = AB * CF

/-- The minimum possible length of the shortest side of the triangle -/
def min_shortest_side (t : Triangle) : ℕ := min t.AB (min t.BC t.CA)

/-- Theorem stating the minimum possible length of the shortest side is 12 -/
theorem min_shortest_side_is_12 (t : Triangle) : min_shortest_side t = 12 := by
  sorry

#check min_shortest_side_is_12

end NUMINAMATH_CALUDE_min_shortest_side_is_12_l1251_125152


namespace NUMINAMATH_CALUDE_least_power_congruence_l1251_125138

theorem least_power_congruence (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 195 → 3^m % 143^2 ≠ 1) ∧ 
  3^195 % 143^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_power_congruence_l1251_125138


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l1251_125187

/-- A perfect square trinomial in the form ax^2 + bx + c -/
structure PerfectSquareTrinomial (a b c : ℝ) : Prop where
  is_perfect_square : ∃ (p q : ℝ), a * x^2 + b * x + c = (p * x + q)^2

/-- The main theorem -/
theorem perfect_square_trinomial_m_values (m : ℝ) :
  PerfectSquareTrinomial 1 (m - 1) 9 → m = -5 ∨ m = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l1251_125187


namespace NUMINAMATH_CALUDE_complex_sum_argument_l1251_125108

/-- The argument of the sum of five complex exponentials -/
theorem complex_sum_argument :
  let z₁ := Complex.exp (11 * π * Complex.I / 120)
  let z₂ := Complex.exp (31 * π * Complex.I / 120)
  let z₃ := Complex.exp (51 * π * Complex.I / 120)
  let z₄ := Complex.exp (71 * π * Complex.I / 120)
  let z₅ := Complex.exp (91 * π * Complex.I / 120)
  Complex.arg (z₁ + z₂ + z₃ + z₄ + z₅) = 17 * π / 40 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_argument_l1251_125108


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l1251_125198

theorem baker_cakes_problem (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (pastry_cake_difference : ℕ) :
  pastries_made = 131 →
  cakes_sold = 70 →
  pastries_sold = 88 →
  pastry_cake_difference = 112 →
  ∃ cakes_made : ℕ, 
    cakes_made + pastry_cake_difference = pastries_made ∧
    cakes_made = 107 :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l1251_125198


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l1251_125145

/-- If the terminal side of angle α passes through the point (-4, 3), then sin α = 3/5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = -4 ∧ r * Real.sin α = 3) → 
  Real.sin α = 3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l1251_125145


namespace NUMINAMATH_CALUDE_stone_piles_total_l1251_125199

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- The conditions of the stone pile problem -/
def satisfiesConditions (piles : StonePiles) : Prop :=
  piles.pile5 = 6 * piles.pile3 ∧
  piles.pile2 = 2 * (piles.pile3 + piles.pile5) ∧
  piles.pile1 = piles.pile5 / 3 ∧
  piles.pile1 = piles.pile4 - 10 ∧
  piles.pile4 = piles.pile2 / 2

/-- The theorem stating that any StonePiles satisfying the conditions will have a total of 60 stones -/
theorem stone_piles_total (piles : StonePiles) :
  satisfiesConditions piles →
  piles.pile1 + piles.pile2 + piles.pile3 + piles.pile4 + piles.pile5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stone_piles_total_l1251_125199


namespace NUMINAMATH_CALUDE_last_digit_of_multiple_of_six_l1251_125171

theorem last_digit_of_multiple_of_six (x : ℕ) :
  x < 10 →
  (43560 + x) % 6 = 0 →
  x = 0 ∨ x = 6 := by
sorry

end NUMINAMATH_CALUDE_last_digit_of_multiple_of_six_l1251_125171


namespace NUMINAMATH_CALUDE_parameterization_validity_l1251_125177

/-- The slope of the line -/
def m : ℚ := 7/4

/-- The y-intercept of the line -/
def b : ℚ := -14/4

/-- The line equation -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- Vector parameterization A -/
def param_A (t : ℚ) : ℚ × ℚ := (3 + 4*t, -5/4 + 7*t)

/-- Vector parameterization B -/
def param_B (t : ℚ) : ℚ × ℚ := (7 + 8*t, 7/4 + 14*t)

/-- Vector parameterization C -/
def param_C (t : ℚ) : ℚ × ℚ := (2 + 14*t, 1/2 + 7*t)

/-- Vector parameterization D -/
def param_D (t : ℚ) : ℚ × ℚ := (-1 + 8*t, -27/4 - 15*t)

/-- Vector parameterization E -/
def param_E (t : ℚ) : ℚ × ℚ := (4 - 7*t, 9/2 + 5*t)

theorem parameterization_validity :
  (∀ t, line_eq (param_A t).1 (param_A t).2) ∧
  (∀ t, line_eq (param_B t).1 (param_B t).2) ∧
  ¬(∀ t, line_eq (param_C t).1 (param_C t).2) ∧
  ¬(∀ t, line_eq (param_D t).1 (param_D t).2) ∧
  ¬(∀ t, line_eq (param_E t).1 (param_E t).2) :=
by sorry

end NUMINAMATH_CALUDE_parameterization_validity_l1251_125177


namespace NUMINAMATH_CALUDE_five_Y_three_equals_64_l1251_125118

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_64 : Y 5 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_equals_64_l1251_125118


namespace NUMINAMATH_CALUDE_joshuas_bottle_caps_l1251_125168

/-- The total number of bottle caps after buying more -/
def total_bottle_caps (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Joshua's total bottle caps -/
theorem joshuas_bottle_caps : total_bottle_caps 40 7 = 47 := by
  sorry

end NUMINAMATH_CALUDE_joshuas_bottle_caps_l1251_125168


namespace NUMINAMATH_CALUDE_solve_g_inequality_range_of_a_l1251_125111

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + |2*x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- Theorem for part (1)
theorem solve_g_inequality :
  ∀ x : ℝ, |g x| < 5 ↔ -2 < x ∧ x < 4 :=
sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  (a ≥ -1 ∨ a ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_solve_g_inequality_range_of_a_l1251_125111


namespace NUMINAMATH_CALUDE_total_edges_after_ten_cuts_l1251_125123

/-- Represents the number of edges after a certain number of cuts -/
def num_edges (n : ℕ) : ℕ :=
  4 + 3 * n

/-- The theorem stating that after 10 cuts, the total number of edges is 34 -/
theorem total_edges_after_ten_cuts :
  num_edges 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_edges_after_ten_cuts_l1251_125123


namespace NUMINAMATH_CALUDE_milk_water_mixture_volume_l1251_125167

/-- Proves that given a mixture of milk and water with an initial ratio of 3:2,
    if adding 46 liters of water changes the ratio to 3:4,
    then the initial volume of the mixture was 115 liters. -/
theorem milk_water_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (h1 : initial_milk / initial_water = 3 / 2)
  (h2 : initial_milk / (initial_water + 46) = 3 / 4) :
  initial_milk + initial_water = 115 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_mixture_volume_l1251_125167


namespace NUMINAMATH_CALUDE_inequality_properties_l1251_125173

theorem inequality_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  (∀ c, a + c > b + c) ∧
  (a^2 > b^2) ∧
  (Real.sqrt a > Real.sqrt b) ∧
  (∃ c, a * c ≤ b * c) := by
sorry

end NUMINAMATH_CALUDE_inequality_properties_l1251_125173


namespace NUMINAMATH_CALUDE_shooter_probability_l1251_125182

theorem shooter_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.20) (h2 : p9 = 0.30) (h3 : p8 = 0.10) :
  1 - (p10 + p9 + p8) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l1251_125182


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l1251_125134

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hit_ground_time (y t : ℝ) : 
  y = -9.8 * t^2 + 5.6 * t + 10 →
  y = 0 →
  t = 131 / 98 := by sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l1251_125134


namespace NUMINAMATH_CALUDE_defective_pens_l1251_125164

theorem defective_pens (total_pens : ℕ) (prob_non_defective : ℚ) (defective_pens : ℕ) : 
  total_pens = 9 →
  prob_non_defective = 5 / 12 →
  (total_pens - defective_pens : ℚ) / total_pens * ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = prob_non_defective →
  defective_pens = 3 := by
sorry

end NUMINAMATH_CALUDE_defective_pens_l1251_125164


namespace NUMINAMATH_CALUDE_complex_simplification_l1251_125121

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the simplification of the given complex expression equals 30 -/
theorem complex_simplification : 6 * (4 - 2 * i) + 2 * i * (6 - 3 * i) = 30 := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l1251_125121


namespace NUMINAMATH_CALUDE_difference_from_sum_and_difference_of_squares_l1251_125175

theorem difference_from_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_from_sum_and_difference_of_squares_l1251_125175


namespace NUMINAMATH_CALUDE_opposite_number_theorem_l1251_125142

theorem opposite_number_theorem (a : ℝ) : -a = -1 → a + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_theorem_l1251_125142


namespace NUMINAMATH_CALUDE_jills_salary_l1251_125190

/-- Represents a person's monthly financial allocation --/
structure MonthlyFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFund : ℝ
  savings : ℝ
  socializing : ℝ
  charitable : ℝ

/-- Conditions for Jill's financial allocation --/
def JillsFinances (m : MonthlyFinances) : Prop :=
  m.discretionaryIncome = m.netSalary / 5 ∧
  m.vacationFund = 0.3 * m.discretionaryIncome ∧
  m.savings = 0.2 * m.discretionaryIncome ∧
  m.socializing = 0.35 * m.discretionaryIncome ∧
  m.charitable = 99

/-- Theorem stating that under the given conditions, Jill's net monthly salary is $3300 --/
theorem jills_salary (m : MonthlyFinances) (h : JillsFinances m) : m.netSalary = 3300 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l1251_125190


namespace NUMINAMATH_CALUDE_no_gcd_solution_l1251_125100

theorem no_gcd_solution : ¬∃ (a b c : ℕ), 
  (Nat.gcd a b = Nat.factorial 30 + 111) ∧ 
  (Nat.gcd b c = Nat.factorial 40 + 234) ∧ 
  (Nat.gcd c a = Nat.factorial 50 + 666) := by
sorry

end NUMINAMATH_CALUDE_no_gcd_solution_l1251_125100


namespace NUMINAMATH_CALUDE_definite_integral_evaluation_l1251_125102

theorem definite_integral_evaluation :
  ∫ x in (1 : ℝ)..3, (2 * x - 1 / (x^2)) = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_evaluation_l1251_125102


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_103_l1251_125179

theorem units_digit_of_7_to_103 : ∃ n : ℕ, 7^103 ≡ 3 [ZMOD 10] :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_103_l1251_125179


namespace NUMINAMATH_CALUDE_select_at_least_one_first_class_l1251_125185

theorem select_at_least_one_first_class :
  let total_parts : ℕ := 10
  let first_class_parts : ℕ := 6
  let second_class_parts : ℕ := 4
  let parts_to_select : ℕ := 3
  let total_combinations := Nat.choose total_parts parts_to_select
  let all_second_class := Nat.choose second_class_parts parts_to_select
  total_combinations - all_second_class = 116 := by
  sorry

end NUMINAMATH_CALUDE_select_at_least_one_first_class_l1251_125185


namespace NUMINAMATH_CALUDE_grape_juice_mixture_problem_l1251_125193

theorem grape_juice_mixture_problem (initial_volume : ℝ) (added_pure_juice : ℝ) (final_percentage : ℝ) :
  initial_volume = 30 →
  added_pure_juice = 10 →
  final_percentage = 0.325 →
  ∃ initial_percentage : ℝ,
    initial_percentage * initial_volume + added_pure_juice = 
    (initial_volume + added_pure_juice) * final_percentage ∧
    initial_percentage = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_problem_l1251_125193


namespace NUMINAMATH_CALUDE_vector_collinearity_l1251_125149

def a (k : ℝ) : Fin 2 → ℝ := ![1, k]
def b : Fin 2 → ℝ := ![2, 2]

theorem vector_collinearity (k : ℝ) :
  (∀ (i : Fin 2), (a k + b) i = (a k) i * (3 : ℝ)) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1251_125149


namespace NUMINAMATH_CALUDE_parabola_equation_l1251_125160

-- Define the parabola
def Parabola (p : ℝ) := {(x, y) : ℝ × ℝ | y^2 = 4 * p * x}

-- Define the focus of the parabola
def Focus (p : ℝ) : ℝ × ℝ := (p, 0)

-- Theorem statement
theorem parabola_equation (p : ℝ) (h : p = 2) :
  Parabola p = {(x, y) : ℝ × ℝ | y^2 = 8 * x} :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1251_125160


namespace NUMINAMATH_CALUDE_books_purchased_with_grant_l1251_125110

/-- The number of books purchased by Silvergrove Public Library using a grant --/
theorem books_purchased_with_grant 
  (total_books : Nat) 
  (books_before_grant : Nat) 
  (h1 : total_books = 8582)
  (h2 : books_before_grant = 5935) :
  total_books - books_before_grant = 2647 := by
  sorry

end NUMINAMATH_CALUDE_books_purchased_with_grant_l1251_125110


namespace NUMINAMATH_CALUDE_type_B_first_is_better_l1251_125148

/-- Represents the score distribution for a two-question quiz -/
structure ScoreDistribution where
  p0 : ℝ  -- Probability of scoring 0
  p1 : ℝ  -- Probability of scoring the first question's points
  p2 : ℝ  -- Probability of scoring both questions' points
  sum_to_one : p0 + p1 + p2 = 1

/-- Calculates the expected score given a score distribution and point values -/
def expectedScore (d : ScoreDistribution) (points1 points2 : ℝ) : ℝ :=
  d.p1 * points1 + d.p2 * (points1 + points2)

/-- Represents the quiz setup -/
structure QuizSetup where
  probA : ℝ  -- Probability of correctly answering type A
  probB : ℝ  -- Probability of correctly answering type B
  pointsA : ℝ  -- Points for correct answer in type A
  pointsB : ℝ  -- Points for correct answer in type B
  probA_bounds : 0 ≤ probA ∧ probA ≤ 1
  probB_bounds : 0 ≤ probB ∧ probB ≤ 1
  positive_points : pointsA > 0 ∧ pointsB > 0

/-- Theorem: Starting with type B questions yields a higher expected score -/
theorem type_B_first_is_better (q : QuizSetup) : 
  let distA : ScoreDistribution := {
    p0 := 1 - q.probA,
    p1 := q.probA * (1 - q.probB),
    p2 := q.probA * q.probB,
    sum_to_one := by sorry
  }
  let distB : ScoreDistribution := {
    p0 := 1 - q.probB,
    p1 := q.probB * (1 - q.probA),
    p2 := q.probB * q.probA,
    sum_to_one := by sorry
  }
  expectedScore distB q.pointsB q.pointsA > expectedScore distA q.pointsA q.pointsB :=
by sorry

end NUMINAMATH_CALUDE_type_B_first_is_better_l1251_125148


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l1251_125103

theorem three_digit_number_problem : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 6 * n = 41 * 18 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l1251_125103


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_one_minus_i_l1251_125146

theorem complex_fraction_equals_neg_one_minus_i :
  let i : ℂ := Complex.I
  (1 + i)^3 / (1 - i)^2 = -1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_one_minus_i_l1251_125146


namespace NUMINAMATH_CALUDE_diamond_equation_l1251_125196

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_equation : 
  (diamond (diamond 2 3) 5) - (diamond 2 (diamond 3 5)) = -37 / 210 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_l1251_125196


namespace NUMINAMATH_CALUDE_flower_cost_ratio_l1251_125155

/-- Given the conditions of Nadia's flower purchase, prove the ratio of lily cost to rose cost. -/
theorem flower_cost_ratio :
  ∀ (roses : ℕ) (lilies : ℚ) (rose_cost lily_cost total_cost : ℚ),
    roses = 20 →
    lilies = (3 / 4) * roses →
    rose_cost = 5 →
    total_cost = 250 →
    total_cost = roses * rose_cost + lilies * lily_cost →
    lily_cost / rose_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_flower_cost_ratio_l1251_125155
