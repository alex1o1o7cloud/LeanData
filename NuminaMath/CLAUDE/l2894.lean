import Mathlib

namespace NUMINAMATH_CALUDE_smallest_set_size_for_divisibility_by_20_l2894_289417

theorem smallest_set_size_for_divisibility_by_20 :
  ∃ (n : ℕ), n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T →
    a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d ∨
    ¬(20 ∣ (a + b - c - d))) :=
by
  sorry

#check smallest_set_size_for_divisibility_by_20

end NUMINAMATH_CALUDE_smallest_set_size_for_divisibility_by_20_l2894_289417


namespace NUMINAMATH_CALUDE_first_digit_base_9_l2894_289443

def base_3_digits : List Nat := [2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1]

def y : Nat := (List.reverse base_3_digits).enum.foldl (fun acc (i, digit) => acc + digit * (3 ^ i)) 0

theorem first_digit_base_9 : ∃ (k : Nat), 4 * (9 ^ k) ≤ y ∧ y < 5 * (9 ^ k) ∧ (∀ m, m > k → y < 4 * (9 ^ m)) :=
  sorry

end NUMINAMATH_CALUDE_first_digit_base_9_l2894_289443


namespace NUMINAMATH_CALUDE_find_divisor_l2894_289476

theorem find_divisor (divisor : ℕ) : 
  (127 / divisor = 9) ∧ (127 % divisor = 1) → divisor = 14 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2894_289476


namespace NUMINAMATH_CALUDE_q_is_false_l2894_289485

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p)) : ¬q :=
sorry

end NUMINAMATH_CALUDE_q_is_false_l2894_289485


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2894_289496

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2894_289496


namespace NUMINAMATH_CALUDE_students_using_green_l2894_289484

theorem students_using_green (total : ℕ) (both : ℕ) (red : ℕ) : 
  total = 70 → both = 38 → red = 56 → 
  total = (total - both) + red → 
  (total - both) = 52 := by sorry

end NUMINAMATH_CALUDE_students_using_green_l2894_289484


namespace NUMINAMATH_CALUDE_Q_representation_exists_zero_polynomial_l2894_289442

variable (x₁ x₂ x₃ x₄ : ℝ)

def Q (x₁ x₂ x₃ x₄ : ℝ) : ℝ := 4 * (x₁^2 + x₂^2 + x₃^2 + x₄^2) - (x₁ + x₂ + x₃ + x₄)^2

def P₁ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ + x₂ - x₃ - x₄
def P₂ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ + x₃ - x₄
def P₃ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ - x₃ + x₄
def P₄ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := 0

theorem Q_representation (x₁ x₂ x₃ x₄ : ℝ) :
  Q x₁ x₂ x₃ x₄ = (P₁ x₁ x₂ x₃ x₄)^2 + (P₂ x₁ x₂ x₃ x₄)^2 + (P₃ x₁ x₂ x₃ x₄)^2 + (P₄ x₁ x₂ x₃ x₄)^2 :=
sorry

theorem exists_zero_polynomial (f g h k : ℝ → ℝ → ℝ → ℝ → ℝ) 
  (hQ : ∀ x₁ x₂ x₃ x₄, Q x₁ x₂ x₃ x₄ = (f x₁ x₂ x₃ x₄)^2 + (g x₁ x₂ x₃ x₄)^2 + (h x₁ x₂ x₃ x₄)^2 + (k x₁ x₂ x₃ x₄)^2) :
  (f = λ _ _ _ _ => 0) ∨ (g = λ _ _ _ _ => 0) ∨ (h = λ _ _ _ _ => 0) ∨ (k = λ _ _ _ _ => 0) :=
sorry

end NUMINAMATH_CALUDE_Q_representation_exists_zero_polynomial_l2894_289442


namespace NUMINAMATH_CALUDE_quadratic_vertex_condition_l2894_289441

theorem quadratic_vertex_condition (a b c x₀ y₀ : ℝ) (h_a : a ≠ 0) :
  (∀ m n : ℝ, n = a * m^2 + b * m + c → a * (y₀ - n) ≤ 0) →
  y₀ = a * x₀^2 + b * x₀ + c →
  2 * a * x₀ + b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_condition_l2894_289441


namespace NUMINAMATH_CALUDE_sum_consecutive_odds_not_even_not_div_four_l2894_289439

theorem sum_consecutive_odds_not_even_not_div_four (n : ℤ) (m : ℤ) :
  ¬(∃ (k : ℤ), 4 * (n + 1) = 2 * k ∧ ¬(∃ (l : ℤ), 2 * k = 4 * l)) :=
by sorry

end NUMINAMATH_CALUDE_sum_consecutive_odds_not_even_not_div_four_l2894_289439


namespace NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l2894_289471

-- Define the necessary types
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the given information
def B : Point := sorry
def G : Point := sorry  -- centroid
def L : Point := sorry  -- intersection of symmedian from B with circumcircle

-- Define the necessary concepts
def isCentroid (G : Point) (t : Triangle) : Prop := sorry
def isSymmedianIntersection (L : Point) (t : Triangle) : Prop := sorry

-- The theorem statement
theorem triangle_reconstruction_uniqueness :
  ∃! (t : Triangle), 
    t.B = B ∧ 
    isCentroid G t ∧ 
    isSymmedianIntersection L t :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l2894_289471


namespace NUMINAMATH_CALUDE_system_solution_l2894_289422

variable (y : ℝ)
variable (x₁ x₂ x₃ x₄ x₅ : ℝ)

def system_equations (y x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₅ + x₂ = y * x₁) ∧
  (x₁ + x₃ = y * x₂) ∧
  (x₂ + x₄ = y * x₃) ∧
  (x₃ + x₅ = y * x₄) ∧
  (x₄ + x₁ = y * x₅)

theorem system_solution :
  (system_equations y x₁ x₂ x₃ x₄ x₅) →
  ((y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
   (y = 2 → ∃ t, x₁ = t ∧ x₂ = t ∧ x₃ = t ∧ x₄ = t ∧ x₅ = t) ∧
   (y^2 + y - 1 = 0 → ∃ u v, x₁ = u ∧ x₅ = v ∧ x₂ = y * u - v ∧ x₃ = -y * (u + v) ∧ x₄ = y * v - u ∧
                            (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2))) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l2894_289422


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2894_289458

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

-- Theorem for part (2)
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2894_289458


namespace NUMINAMATH_CALUDE_farmer_vegetable_difference_l2894_289470

/-- Calculates the total difference between initial and remaining tomatoes and carrots --/
def total_difference (initial_tomatoes initial_carrots picked_tomatoes picked_carrots given_tomatoes given_carrots : ℕ) : ℕ :=
  (initial_tomatoes - (initial_tomatoes - picked_tomatoes + given_tomatoes)) +
  (initial_carrots - (initial_carrots - picked_carrots + given_carrots))

/-- Theorem stating the total difference for the given problem --/
theorem farmer_vegetable_difference :
  total_difference 17 13 5 6 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_farmer_vegetable_difference_l2894_289470


namespace NUMINAMATH_CALUDE_prime_multiple_all_ones_l2894_289409

theorem prime_multiple_all_ones (p : ℕ) (hp : Prime p) (hp_not_two : p ≠ 2) (hp_not_five : p ≠ 5) :
  ∃ k : ℕ, ∃ n : ℕ, p * k = 10^n - 1 :=
sorry

end NUMINAMATH_CALUDE_prime_multiple_all_ones_l2894_289409


namespace NUMINAMATH_CALUDE_driver_license_exam_results_l2894_289488

/-- Represents the probabilities of passing each subject in the driver's license exam -/
structure ExamProbabilities where
  subject1 : ℝ
  subject2 : ℝ
  subject3 : ℝ

/-- Calculates the probability of obtaining a driver's license -/
def probabilityOfObtainingLicense (p : ExamProbabilities) : ℝ :=
  p.subject1 * p.subject2 * p.subject3

/-- Calculates the expected number of attempts during the application process -/
def expectedAttempts (p : ExamProbabilities) : ℝ :=
  1 * (1 - p.subject1) +
  2 * (p.subject1 * (1 - p.subject2)) +
  3 * (p.subject1 * p.subject2)

/-- Theorem stating the probability of obtaining a license and expected attempts -/
theorem driver_license_exam_results (p : ExamProbabilities)
  (h1 : p.subject1 = 0.9)
  (h2 : p.subject2 = 0.7)
  (h3 : p.subject3 = 0.6) :
  probabilityOfObtainingLicense p = 0.378 ∧
  expectedAttempts p = 2.53 := by
  sorry


end NUMINAMATH_CALUDE_driver_license_exam_results_l2894_289488


namespace NUMINAMATH_CALUDE_dividend_calculation_l2894_289431

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.05) :
  let actual_share_price := share_value * (1 + premium_rate)
  let num_shares := investment / actual_share_price
  let dividend_per_share := share_value * dividend_rate
  dividend_per_share * num_shares = 600 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2894_289431


namespace NUMINAMATH_CALUDE_min_snakes_is_three_l2894_289489

/-- Represents the number of people owning a specific combination of pets -/
structure PetOwnership :=
  (total : ℕ)
  (onlyDogs : ℕ)
  (onlyCats : ℕ)
  (catsAndDogs : ℕ)
  (catsDogsSnakes : ℕ)

/-- The minimum number of snakes given the pet ownership information -/
def minSnakes (po : PetOwnership) : ℕ := po.catsDogsSnakes

/-- Theorem stating that the minimum number of snakes is 3 given the problem conditions -/
theorem min_snakes_is_three (po : PetOwnership)
  (h1 : po.total = 89)
  (h2 : po.onlyDogs = 15)
  (h3 : po.onlyCats = 10)
  (h4 : po.catsAndDogs = 5)
  (h5 : po.catsDogsSnakes = 3) :
  minSnakes po = 3 := by sorry

end NUMINAMATH_CALUDE_min_snakes_is_three_l2894_289489


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2894_289486

theorem complex_equation_sum (x y : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (x - 2 : ℂ) + y * i = -1 + i) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2894_289486


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l2894_289402

/-- The surface area of a cuboid given its three edge lengths -/
def cuboidSurfaceArea (x y z : ℝ) : ℝ := 2 * (x * y + x * z + y * z)

/-- Theorem stating that if a cuboid with edges x, 5, and 6 has surface area 148, then x = 4 -/
theorem cuboid_edge_length (x : ℝ) :
  cuboidSurfaceArea x 5 6 = 148 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l2894_289402


namespace NUMINAMATH_CALUDE_permutation_sum_squares_values_l2894_289472

theorem permutation_sum_squares_values (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) : 
  ∃! (s : Finset ℝ), 
    s.card = 3 ∧ 
    (∀ (x y z t : ℝ), ({x, y, z, t} : Finset ℝ) = {a, b, c, d} → 
      ((x - y)^2 + (y - z)^2 + (z - t)^2 + (t - x)^2) ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_permutation_sum_squares_values_l2894_289472


namespace NUMINAMATH_CALUDE_english_only_students_l2894_289466

theorem english_only_students (total : Nat) (max_liz : Nat) (english : Nat) (french : Nat) : 
  total = 25 → 
  max_liz = 2 → 
  total = english + french - max_liz → 
  english = 2 * french → 
  english - french = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_english_only_students_l2894_289466


namespace NUMINAMATH_CALUDE_smallest_positive_e_value_l2894_289447

theorem smallest_positive_e_value (a b c d e : ℤ) : 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    (x = -3 ∨ x = 7 ∨ x = 8 ∨ x = -1/4)) →
  (∀ e' : ℤ, e' > 0 → e' ≥ 168) →
  e = 168 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_e_value_l2894_289447


namespace NUMINAMATH_CALUDE_triangulation_count_equals_catalan_l2894_289404

/-- The number of ways to triangulate a convex polygon -/
def triangulationCount (n : ℕ) : ℕ := sorry

/-- The n-th Catalan number -/
def catalanNumber (n : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to triangulate a convex (n+2)-gon
    is equal to the (n-1)-th Catalan number -/
theorem triangulation_count_equals_catalan (n : ℕ) :
  triangulationCount (n + 2) = catalanNumber (n - 1) := by sorry

end NUMINAMATH_CALUDE_triangulation_count_equals_catalan_l2894_289404


namespace NUMINAMATH_CALUDE_expand_expression_l2894_289479

theorem expand_expression (x : ℝ) : 5 * (x + 3) * (x + 6) = 5 * x^2 + 45 * x + 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2894_289479


namespace NUMINAMATH_CALUDE_godzilla_stitches_proof_l2894_289490

/-- The number of stitches Carolyn can sew per minute -/
def stitches_per_minute : ℕ := 4

/-- The number of stitches required to embroider a flower -/
def stitches_per_flower : ℕ := 60

/-- The number of stitches required to embroider a unicorn -/
def stitches_per_unicorn : ℕ := 180

/-- The number of unicorns in the embroidery -/
def num_unicorns : ℕ := 3

/-- The number of flowers in the embroidery -/
def num_flowers : ℕ := 50

/-- The total time Carolyn spends embroidering (in minutes) -/
def total_time : ℕ := 1085

/-- The number of stitches required to embroider Godzilla -/
def stitches_for_godzilla : ℕ := 800

theorem godzilla_stitches_proof : 
  stitches_for_godzilla = 
    total_time * stitches_per_minute - 
    (num_unicorns * stitches_per_unicorn + num_flowers * stitches_per_flower) := by
  sorry

end NUMINAMATH_CALUDE_godzilla_stitches_proof_l2894_289490


namespace NUMINAMATH_CALUDE_solve_probability_problem_l2894_289405

def probability_problem (p_man : ℚ) (p_wife : ℚ) : Prop :=
  p_man = 1/4 ∧ p_wife = 1/3 →
  (1 - p_man) * (1 - p_wife) = 1/2

theorem solve_probability_problem : probability_problem (1/4) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_solve_probability_problem_l2894_289405


namespace NUMINAMATH_CALUDE_completing_square_transform_l2894_289445

theorem completing_square_transform (x : ℝ) :
  x^2 - 2*x - 5 = 0 ↔ (x - 1)^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transform_l2894_289445


namespace NUMINAMATH_CALUDE_total_paintable_area_l2894_289420

/-- Calculate the total square feet of walls to be painted in bedrooms and hallway -/
theorem total_paintable_area (
  num_bedrooms : ℕ)
  (bedroom_length bedroom_width bedroom_height : ℝ)
  (hallway_length hallway_width hallway_height : ℝ)
  (unpaintable_area_per_bedroom : ℝ)
  (h1 : num_bedrooms = 4)
  (h2 : bedroom_length = 14)
  (h3 : bedroom_width = 11)
  (h4 : bedroom_height = 9)
  (h5 : hallway_length = 20)
  (h6 : hallway_width = 7)
  (h7 : hallway_height = 9)
  (h8 : unpaintable_area_per_bedroom = 70) :
  (num_bedrooms * (2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area_per_bedroom)) +
  (2 * (hallway_length * hallway_height + hallway_width * hallway_height)) = 2006 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l2894_289420


namespace NUMINAMATH_CALUDE_surface_area_of_sawed_cube_l2894_289452

/-- The total surface area of rectangular blocks obtained by sawing a unit cube -/
def total_surface_area (length_cuts width_cuts height_cuts : ℕ) : ℝ :=
  let original_surface := 6
  let new_surface := (length_cuts + 1) * (width_cuts + 1) * 2 +
                     (length_cuts + 1) * (height_cuts + 1) * 2 +
                     (width_cuts + 1) * (height_cuts + 1) * 2
  original_surface + new_surface - 6

/-- Theorem: The total surface area of 24 rectangular blocks obtained by sawing a unit cube
    1 time along length, 2 times along width, and 3 times along height is 18 square meters -/
theorem surface_area_of_sawed_cube : total_surface_area 1 2 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_sawed_cube_l2894_289452


namespace NUMINAMATH_CALUDE_trucks_given_to_jeff_l2894_289415

-- Define the variables
def initial_trucks : ℕ := 51
def remaining_trucks : ℕ := 38

-- Define the theorem
theorem trucks_given_to_jeff : 
  initial_trucks - remaining_trucks = 13 := by
  sorry

end NUMINAMATH_CALUDE_trucks_given_to_jeff_l2894_289415


namespace NUMINAMATH_CALUDE_angle_reflection_l2894_289424

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 180 + 360 * (k : Real) < α ∧ α < 270 + 360 * (k : Real)

def is_in_fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 270 + 360 * (k : Real) < α ∧ α < 360 + 360 * (k : Real)

theorem angle_reflection (α : Real) :
  is_in_third_quadrant α → is_in_fourth_quadrant (180 - α) := by
  sorry

end NUMINAMATH_CALUDE_angle_reflection_l2894_289424


namespace NUMINAMATH_CALUDE_num_factors_48_mult_6_eq_4_l2894_289411

/-- The number of positive factors of 48 that are also multiples of 6 -/
def num_factors_48_mult_6 : ℕ :=
  (Finset.filter (λ x => x ∣ 48 ∧ 6 ∣ x) (Finset.range 49)).card

/-- Theorem stating that the number of positive factors of 48 that are also multiples of 6 is 4 -/
theorem num_factors_48_mult_6_eq_4 : num_factors_48_mult_6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_num_factors_48_mult_6_eq_4_l2894_289411


namespace NUMINAMATH_CALUDE_min_y_max_x_l2894_289427

theorem min_y_max_x (x y : ℝ) (h : x^2 + y^2 = 18*x + 40*y) : 
  (∀ y' : ℝ, x^2 + y'^2 = 18*x + 40*y' → y ≤ y') ∧ 
  (∀ x' : ℝ, x'^2 + y^2 = 18*x' + 40*y → x' ≤ x) → 
  y = 20 - Real.sqrt 481 ∧ x = 9 + Real.sqrt 481 :=
by sorry

end NUMINAMATH_CALUDE_min_y_max_x_l2894_289427


namespace NUMINAMATH_CALUDE_cards_left_after_distribution_l2894_289416

/-- Given the initial number of cards, number of cards given to each student,
    and number of students, prove that the number of cards left is 12. -/
theorem cards_left_after_distribution (initial_cards : ℕ) (cards_per_student : ℕ) (num_students : ℕ)
    (h1 : initial_cards = 357)
    (h2 : cards_per_student = 23)
    (h3 : num_students = 15) :
  initial_cards - (cards_per_student * num_students) = 12 := by
  sorry

#check cards_left_after_distribution

end NUMINAMATH_CALUDE_cards_left_after_distribution_l2894_289416


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l2894_289455

theorem rectangle_area_perimeter_relation (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := x + 7
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → (x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l2894_289455


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2894_289437

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - I) = abs (1 - I) + I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2894_289437


namespace NUMINAMATH_CALUDE_subtraction_division_equality_l2894_289453

theorem subtraction_division_equality : 5020 - (502 / 100.4) = 5014.998 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_equality_l2894_289453


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2894_289449

theorem fraction_sum_equality (p q m n : ℕ+) (x : ℚ) 
  (h1 : (p : ℚ) / q = (m : ℚ) / n)
  (h2 : (p : ℚ) / q = 4 / 5)
  (h3 : x = 1 / 7) :
  x + ((2 * q - p + 3 * m - 2 * n) : ℚ) / (2 * q + p - m + n) = 71 / 105 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2894_289449


namespace NUMINAMATH_CALUDE_min_value_theorem_l2894_289423

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = 9 ∧ (1/x + 4/y ≥ min) ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1/x₀ + 4/y₀ = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2894_289423


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2894_289448

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 141 →
  divisor = 17 →
  quotient = 8 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2894_289448


namespace NUMINAMATH_CALUDE_unique_factorization_1386_l2894_289403

/-- Two-digit numbers are natural numbers between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A factorization of 1386 into two two-digit numbers -/
structure Factorization :=
  (a b : ℕ)
  (h1 : TwoDigitNumber a)
  (h2 : TwoDigitNumber b)
  (h3 : a * b = 1386)

/-- Two factorizations are considered the same if they have the same factors (in any order) -/
def Factorization.equiv (f g : Factorization) : Prop :=
  (f.a = g.a ∧ f.b = g.b) ∨ (f.a = g.b ∧ f.b = g.a)

/-- The main theorem stating that there is exactly one factorization of 1386 into two-digit numbers -/
theorem unique_factorization_1386 : 
  ∃! (f : Factorization), True :=
sorry

end NUMINAMATH_CALUDE_unique_factorization_1386_l2894_289403


namespace NUMINAMATH_CALUDE_card_probability_and_combinations_l2894_289425

theorem card_probability_and_combinations : 
  -- Part 1: Probability of drawing two hearts
  (Nat.choose 13 2 : ℚ) / (Nat.choose 52 2) = 1 / 17 ∧ 
  -- Part 2: Number of ways to choose 15 from 17
  Nat.choose 17 15 = 136 ∧ 
  -- Part 3: Number of non-empty subsets of a 4-element set
  (2^4 - 1 : ℕ) = 15 ∧ 
  -- Part 4: Probability of drawing a red ball
  (3 : ℚ) / 15 = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_card_probability_and_combinations_l2894_289425


namespace NUMINAMATH_CALUDE_angle_terminal_side_l2894_289487

theorem angle_terminal_side (x : ℝ) (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (x, 4) ∧ P.1 = x ∧ P.2 = 4) → 
  Real.sin α = 4/5 → 
  x = 3 ∨ x = -3 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l2894_289487


namespace NUMINAMATH_CALUDE_brownies_made_next_morning_l2894_289459

def initial_brownies : ℕ := 2 * 12
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def total_next_morning : ℕ := 36

theorem brownies_made_next_morning :
  total_next_morning - (initial_brownies - father_ate - mooney_ate) = 24 := by
  sorry

end NUMINAMATH_CALUDE_brownies_made_next_morning_l2894_289459


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l2894_289495

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 13)
  (h2 : z + x = 14)
  (h3 : x + y = 15) :
  Real.sqrt (x * y * z * (x + y + z)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l2894_289495


namespace NUMINAMATH_CALUDE_other_coin_denomination_l2894_289414

/-- Proves that given the problem conditions, the denomination of the other type of coin is 25 paise -/
theorem other_coin_denomination (total_coins : ℕ) (total_value : ℕ) (twenty_paise_coins : ℕ) 
  (h1 : total_coins = 324)
  (h2 : total_value = 7100)  -- 71 Rs in paise
  (h3 : twenty_paise_coins = 200) :
  (total_value - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

end NUMINAMATH_CALUDE_other_coin_denomination_l2894_289414


namespace NUMINAMATH_CALUDE_point_outside_circle_l2894_289407

theorem point_outside_circle (r OA : ℝ) (h1 : r = 3) (h2 : OA = 5) :
  OA > r := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2894_289407


namespace NUMINAMATH_CALUDE_reflection_about_y_eq_neg_x_l2894_289436

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

/-- The original point -/
def original_point : ℝ × ℝ := (3, -7)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (7, -3)

theorem reflection_about_y_eq_neg_x :
  reflect_about_y_eq_neg_x original_point = reflected_point := by
  sorry

end NUMINAMATH_CALUDE_reflection_about_y_eq_neg_x_l2894_289436


namespace NUMINAMATH_CALUDE_grapes_per_day_calculation_l2894_289428

/-- The number of pickers -/
def num_pickers : ℕ := 235

/-- The number of drums of raspberries filled per day -/
def raspberries_per_day : ℕ := 100

/-- The number of days -/
def num_days : ℕ := 77

/-- The total number of drums filled in 77 days -/
def total_drums : ℕ := 17017

/-- The number of drums of grapes filled per day -/
def grapes_per_day : ℕ := 121

theorem grapes_per_day_calculation :
  grapes_per_day = (total_drums - raspberries_per_day * num_days) / num_days :=
by sorry

end NUMINAMATH_CALUDE_grapes_per_day_calculation_l2894_289428


namespace NUMINAMATH_CALUDE_ryan_got_seven_books_l2894_289480

/-- Calculates the number of books Ryan got from the library given the conditions -/
def ryans_books (ryan_total_pages : ℕ) (brother_daily_pages : ℕ) (days : ℕ) (ryan_extra_daily_pages : ℕ) : ℕ :=
  ryan_total_pages / (brother_daily_pages + ryan_extra_daily_pages)

/-- Theorem stating that Ryan got 7 books from the library -/
theorem ryan_got_seven_books :
  ryans_books 2100 200 7 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_got_seven_books_l2894_289480


namespace NUMINAMATH_CALUDE_roger_cookie_price_l2894_289413

/-- Represents a cookie batch -/
structure CookieBatch where
  shape : String
  count : ℕ
  price : ℚ

/-- Calculates the total earnings from a cookie batch -/
def totalEarnings (batch : CookieBatch) : ℚ :=
  batch.count * batch.price

theorem roger_cookie_price (art_batch roger_batch : CookieBatch) 
  (h1 : art_batch.shape = "rectangle")
  (h2 : roger_batch.shape = "square")
  (h3 : art_batch.count = 15)
  (h4 : roger_batch.count = 20)
  (h5 : art_batch.price = 75/100)
  (h6 : totalEarnings art_batch = totalEarnings roger_batch) :
  roger_batch.price = 5625/10000 := by
  sorry

#eval (5625 : ℚ) / 10000  -- Expected output: 0.5625

end NUMINAMATH_CALUDE_roger_cookie_price_l2894_289413


namespace NUMINAMATH_CALUDE_inequality_range_l2894_289429

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ a ≤ -1 ∨ a ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2894_289429


namespace NUMINAMATH_CALUDE_sqrt_pattern_l2894_289421

theorem sqrt_pattern (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 + (2 * n - 1) / (n^2 : ℝ)) = (n + 1 : ℝ) / n :=
by sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l2894_289421


namespace NUMINAMATH_CALUDE_max_self_intersections_correct_max_self_intersections_13_max_self_intersections_1950_l2894_289491

/-- The maximum number of self-intersection points in a closed polygonal line with n segments -/
def max_self_intersections (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 3) / 2
  else
    n * (n - 4) / 2 + 1

theorem max_self_intersections_correct (n : ℕ) (h : n ≥ 3) :
  max_self_intersections n =
    if n % 2 = 1 then
      n * (n - 3) / 2
    else
      n * (n - 4) / 2 + 1 :=
by sorry

-- Specific cases
theorem max_self_intersections_13 :
  max_self_intersections 13 = 65 :=
by sorry

theorem max_self_intersections_1950 :
  max_self_intersections 1950 = 1897851 :=
by sorry

end NUMINAMATH_CALUDE_max_self_intersections_correct_max_self_intersections_13_max_self_intersections_1950_l2894_289491


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2894_289406

-- Define the equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (3 - k) + y^2 / (k - 2) = 1

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y, hyperbola_equation x y k ∧ (3 - k) * (k - 2) < 0

-- Theorem statement
theorem hyperbola_condition (k : ℝ) :
  is_hyperbola k ↔ k < 2 ∨ k > 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2894_289406


namespace NUMINAMATH_CALUDE_polynomial_sum_coefficients_l2894_289451

theorem polynomial_sum_coefficients (d : ℝ) (a b c e : ℤ) : 
  d ≠ 0 → 
  (16 * d + 15 + 18 * d^2 + 5 * d^3) + (4 * d + 3 + 3 * d^3) = a * d + b + c * d^2 + e * d^3 →
  a + b + c + e = 64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_coefficients_l2894_289451


namespace NUMINAMATH_CALUDE_reflection_of_point_p_l2894_289468

/-- The coordinates of a point with respect to the center of the coordinate origin -/
def reflection_through_origin (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- Theorem: The coordinates of P(-3,1) with respect to the center of the coordinate origin are (3,-1) -/
theorem reflection_of_point_p : reflection_through_origin (-3) 1 = (3, -1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_p_l2894_289468


namespace NUMINAMATH_CALUDE_count_pairs_eq_12_l2894_289483

def count_pairs : ℕ :=
  Finset.sum (Finset.range 3) (fun x =>
    Finset.card (Finset.filter (fun y => x + 1 + y < 7) (Finset.range 6)))

theorem count_pairs_eq_12 : count_pairs = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_12_l2894_289483


namespace NUMINAMATH_CALUDE_eve_last_student_l2894_289462

/-- Represents the students in the circle -/
inductive Student
| Alan
| Bob
| Cara
| Dan
| Eve

/-- The order of students in the circle -/
def initialOrder : List Student := [Student.Alan, Student.Bob, Student.Cara, Student.Dan, Student.Eve]

/-- Checks if a number is a multiple of 7 or contains the digit 6 -/
def isEliminationNumber (n : Nat) : Bool :=
  n % 7 == 0 || n.repr.contains '6'

/-- Simulates the elimination process and returns the last student remaining -/
def lastStudent (order : List Student) : Student :=
  sorry

/-- Theorem stating that Eve is the last student remaining -/
theorem eve_last_student : lastStudent initialOrder = Student.Eve :=
  sorry

end NUMINAMATH_CALUDE_eve_last_student_l2894_289462


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2894_289464

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, x^2 - m*x - 6*n < 0 ↔ -3 < x ∧ x < 6) → m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2894_289464


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l2894_289469

def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}

theorem subset_implies_a_values :
  ∀ a : ℝ, B a ⊆ A → a ∈ ({0, 1, -1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l2894_289469


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l2894_289400

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) := 2 * a^2 + 3 * b - a * b

/-- Theorem stating that 4 * 3 = 29 under the custom multiplication -/
theorem custom_mult_four_three : custom_mult 4 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l2894_289400


namespace NUMINAMATH_CALUDE_range_of_expression_l2894_289440

theorem range_of_expression (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (∀ x y : ℝ, y = x^2 + 2*b*x + 1 → y ≠ 2*a*(x + b)) →
  ∀ θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) →
    1 < (a - Real.cos θ)^2 + (b - Real.sin θ)^2 ∧
    (a - Real.cos θ)^2 + (b - Real.sin θ)^2 < 4 := by
  sorry


end NUMINAMATH_CALUDE_range_of_expression_l2894_289440


namespace NUMINAMATH_CALUDE_correct_mark_l2894_289410

theorem correct_mark (wrong_mark : ℕ) (class_size : ℕ) (average_increase : ℚ) 
  (h1 : wrong_mark = 79)
  (h2 : class_size = 68)
  (h3 : average_increase = 1/2) : 
  ∃ (correct_mark : ℕ), 
    (wrong_mark : ℚ) - correct_mark = average_increase * class_size ∧ 
    correct_mark = 45 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_l2894_289410


namespace NUMINAMATH_CALUDE_remainder_of_2456789_div_7_l2894_289454

theorem remainder_of_2456789_div_7 : 2456789 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2456789_div_7_l2894_289454


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l2894_289438

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 10 = 36 → Nat.gcd n 10 = 5 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l2894_289438


namespace NUMINAMATH_CALUDE_worker_wages_l2894_289435

theorem worker_wages (workers1 workers2 days1 days2 total_wages2 : ℕ)
  (hw1 : workers1 = 15)
  (hw2 : workers2 = 19)
  (hd1 : days1 = 6)
  (hd2 : days2 = 5)
  (ht2 : total_wages2 = 9975) :
  workers1 * days1 * (total_wages2 / (workers2 * days2)) = 9450 := by
  sorry

end NUMINAMATH_CALUDE_worker_wages_l2894_289435


namespace NUMINAMATH_CALUDE_prime_in_sequence_l2894_289408

theorem prime_in_sequence (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∃ n : ℕ, p = Int.sqrt (24 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_in_sequence_l2894_289408


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2894_289474

/-- The speed of the goods train in kilometers per hour -/
def train_speed : ℝ := 72

/-- The time taken by the train to cross the platform in seconds -/
def crossing_time : ℝ := 26

/-- The length of the goods train in meters -/
def train_length : ℝ := 170.0416

/-- The length of the platform in meters -/
def platform_length : ℝ := 349.9584

/-- Theorem stating that the calculated platform length is correct -/
theorem platform_length_calculation :
  platform_length = (train_speed * 1000 / 3600 * crossing_time) - train_length := by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l2894_289474


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l2894_289493

theorem abs_sum_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -13/2 < x ∧ x < 7/2 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l2894_289493


namespace NUMINAMATH_CALUDE_second_to_first_ratio_l2894_289473

/-- Represents the guesses of four students for the number of jellybeans in a jar. -/
structure JellybeanGuesses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Defines the conditions for the jellybean guessing problem. -/
def valid_guesses (g : JellybeanGuesses) : Prop :=
  g.first = 100 ∧
  g.third = g.second - 200 ∧
  g.fourth = (g.first + g.second + g.third) / 3 + 25 ∧
  g.fourth = 525

/-- Theorem stating that for valid guesses, the ratio of the second to the first guess is 8:1. -/
theorem second_to_first_ratio (g : JellybeanGuesses) (h : valid_guesses g) :
  g.second / g.first = 8 := by
  sorry

#check second_to_first_ratio

end NUMINAMATH_CALUDE_second_to_first_ratio_l2894_289473


namespace NUMINAMATH_CALUDE_elevator_optimal_stop_l2894_289498

def total_floors : ℕ := 12
def num_people : ℕ := 11

def dissatisfaction (n : ℕ) : ℕ :=
  let down_sum := (n - 2) * (n - 1) / 2
  let up_sum := (total_floors - n) * (total_floors - n + 1)
  down_sum + 2 * up_sum

theorem elevator_optimal_stop :
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ total_floors →
    dissatisfaction 9 ≤ dissatisfaction k :=
sorry

end NUMINAMATH_CALUDE_elevator_optimal_stop_l2894_289498


namespace NUMINAMATH_CALUDE_chessboard_dark_light_difference_l2894_289430

/-- Represents a square on the chessboard -/
inductive Square
| Dark
| Light

/-- Represents a row on the chessboard -/
def Row := Vector Square 9

/-- Generates a row starting with the given square color -/
def generateRow (startSquare : Square) : Row := sorry

/-- The chessboard, consisting of 9 rows -/
def Chessboard := Vector Row 9

/-- Generates the chessboard with alternating row starts -/
def generateChessboard : Chessboard := sorry

/-- Counts the number of dark squares in a row -/
def countDarkSquares (row : Row) : Nat := sorry

/-- Counts the number of light squares in a row -/
def countLightSquares (row : Row) : Nat := sorry

/-- Counts the total number of dark squares on the chessboard -/
def totalDarkSquares (board : Chessboard) : Nat := sorry

/-- Counts the total number of light squares on the chessboard -/
def totalLightSquares (board : Chessboard) : Nat := sorry

theorem chessboard_dark_light_difference :
  let board := generateChessboard
  totalDarkSquares board = totalLightSquares board + 1 := by sorry

end NUMINAMATH_CALUDE_chessboard_dark_light_difference_l2894_289430


namespace NUMINAMATH_CALUDE_special_ellipse_major_twice_minor_l2894_289433

/-- An ellipse where one focus and two vertices form an equilateral triangle -/
structure SpecialEllipse where
  -- Major axis length
  a : ℝ
  -- Minor axis length
  b : ℝ
  -- Distance from center to focus
  c : ℝ
  -- Constraint that one focus and two vertices form an equilateral triangle
  equilateral_triangle : c = a / 2
  -- Standard ellipse equation
  ellipse_equation : a^2 = b^2 + c^2

/-- The major axis is twice the minor axis in a special ellipse -/
theorem special_ellipse_major_twice_minor (e : SpecialEllipse) : e.a = 2 * e.b := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_major_twice_minor_l2894_289433


namespace NUMINAMATH_CALUDE_borrowed_amount_l2894_289463

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  amount : ℝ  -- The amount borrowed/lent
  borrowRate : ℝ  -- Borrowing interest rate (as a decimal)
  lendRate : ℝ  -- Lending interest rate (as a decimal)
  years : ℝ  -- Duration of the transaction in years
  yearlyGain : ℝ  -- Gain per year

/-- Calculates the total gain over the entire period -/
def totalGain (t : Transaction) : ℝ :=
  (t.lendRate - t.borrowRate) * t.amount * t.years

/-- The main theorem that proves the borrowed amount given the conditions -/
theorem borrowed_amount (t : Transaction) 
    (h1 : t.years = 2)
    (h2 : t.borrowRate = 0.04)
    (h3 : t.lendRate = 0.06)
    (h4 : t.yearlyGain = 80) :
    t.amount = 2000 := by
  sorry

#check borrowed_amount

end NUMINAMATH_CALUDE_borrowed_amount_l2894_289463


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_prop_l2894_289477

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n : ℕ, p n) ↔ (∀ n : ℕ, ¬ p n) := by sorry

theorem negation_of_greater_than_prop :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_prop_l2894_289477


namespace NUMINAMATH_CALUDE_yw_equals_two_l2894_289460

/-- A right triangle with specific side lengths and a median -/
structure RightTriangleWithMedian where
  /-- The length of side XY -/
  xy : ℝ
  /-- The length of side YZ -/
  yz : ℝ
  /-- The point where the median from X meets YZ -/
  w : ℝ
  /-- XY equals 3 -/
  xy_eq : xy = 3
  /-- YZ equals 4 -/
  yz_eq : yz = 4

/-- The length YW in a right triangle with specific side lengths and median -/
def yw (t : RightTriangleWithMedian) : ℝ := t.w

/-- Theorem: In a right triangle XYZ with XY = 3 and YZ = 4, 
    if W is where the median from X meets YZ, then YW = 2 -/
theorem yw_equals_two (t : RightTriangleWithMedian) : yw t = 2 := by
  sorry

end NUMINAMATH_CALUDE_yw_equals_two_l2894_289460


namespace NUMINAMATH_CALUDE_find_other_number_l2894_289475

theorem find_other_number (A B : ℕ+) (h1 : A = 24) (h2 : Nat.gcd A B = 15) (h3 : Nat.lcm A B = 312) :
  B = 195 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2894_289475


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2894_289494

theorem tangent_line_to_circle (θ : Real) (h1 : 0 < θ ∧ θ < π) :
  (∃ t : Real, ∀ x y : Real,
    (x = t * Real.cos θ ∧ y = t * Real.sin θ) →
    (∃ α : Real, x = 4 + 2 * Real.cos α ∧ y = 2 * Real.sin α) →
    (∀ x' y' : Real, (x' - 4)^2 + y'^2 = 4 →
      (y' - y) * Real.cos θ = (x' - x) * Real.sin θ)) →
  θ = π/6 ∨ θ = 5*π/6 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2894_289494


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2894_289497

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | x ≥ 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2894_289497


namespace NUMINAMATH_CALUDE_complex_problem_l2894_289450

theorem complex_problem (b : ℝ) (z : ℂ) (h1 : z = 3 + b * I) 
  (h2 : (Complex.I * (Complex.I * ((1 + 3 * I) * z))).re = 0) : 
  z = 3 + I ∧ Complex.abs (z / (2 + I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_problem_l2894_289450


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2894_289481

/-- Proves that given the conditions of the problem, the principal amount is 2800 --/
theorem simple_interest_problem (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2240 → P = 2800 := by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2894_289481


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2894_289482

/-- Given a geometric sequence {a_n} with common ratio 2 and a_1 * a_3 = 6 * a_2, prove that a_4 = 24 -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  a 1 * a 3 = 6 * a 2 →         -- given condition
  a 4 = 24 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2894_289482


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2894_289446

theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 3) ↔ (∃ x y : ℝ, y = 3 * m * x + 6 ∧ y = (4 * m - 3) * x + 9) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2894_289446


namespace NUMINAMATH_CALUDE_painting_price_increase_l2894_289467

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 20 / 100) = 104 / 100 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l2894_289467


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_sixth_l2894_289478

theorem opposite_of_negative_one_sixth (x : ℚ) : x = -1/6 → -x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_sixth_l2894_289478


namespace NUMINAMATH_CALUDE_range_of_f_real_l2894_289492

def f (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem range_of_f_real : Set.range f = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_real_l2894_289492


namespace NUMINAMATH_CALUDE_boxes_ordered_correct_l2894_289401

/-- Represents the number of apples in each box -/
def apples_per_box : ℕ := 300

/-- Represents the fraction of stock sold -/
def fraction_sold : ℚ := 3/4

/-- Represents the number of unsold apples -/
def unsold_apples : ℕ := 750

/-- Calculates the number of boxes ordered each week -/
def boxes_ordered : ℕ := 10

/-- Proves that the number of boxes ordered is correct given the conditions -/
theorem boxes_ordered_correct :
  (1 - fraction_sold) * (apples_per_box * boxes_ordered) = unsold_apples := by sorry

end NUMINAMATH_CALUDE_boxes_ordered_correct_l2894_289401


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2894_289418

theorem quadratic_inequality (x : ℝ) : x^2 - 40*x + 400 ≤ 10 ↔ 20 - Real.sqrt 10 ≤ x ∧ x ≤ 20 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2894_289418


namespace NUMINAMATH_CALUDE_semicircle_covering_l2894_289457

theorem semicircle_covering (N : ℕ) (r : ℝ) : 
  N > 0 → 
  r > 0 → 
  let A := N * (π * r^2 / 2)
  let B := (π * (N * r)^2 / 2) - A
  A / B = 1 / 3 → 
  N = 4 := by
sorry

end NUMINAMATH_CALUDE_semicircle_covering_l2894_289457


namespace NUMINAMATH_CALUDE_volume_of_specific_cuboid_l2894_289456

/-- The volume of a cuboid formed by two identical cubes in a line --/
def cuboid_volume (edge_length : ℝ) : ℝ :=
  2 * (edge_length ^ 3)

/-- Theorem: The volume of a cuboid formed by two cubes with edge length 5 cm is 250 cm³ --/
theorem volume_of_specific_cuboid :
  cuboid_volume 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_cuboid_l2894_289456


namespace NUMINAMATH_CALUDE_only_math_scores_need_census_l2894_289426

-- Define the survey types
inductive SurveyType
  | Sampling
  | Census

-- Define the survey options
inductive SurveyOption
  | WeeklyAllowance
  | MathTestScores
  | TVWatchTime
  | ExtracurricularReading

-- Function to determine the appropriate survey type for each option
def appropriateSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.MathTestScores => SurveyType.Census
  | _ => SurveyType.Sampling

-- Theorem stating that only the MathTestScores option requires a census
theorem only_math_scores_need_census :
  ∀ (option : SurveyOption),
    appropriateSurveyType option = SurveyType.Census ↔ option = SurveyOption.MathTestScores :=
by sorry


end NUMINAMATH_CALUDE_only_math_scores_need_census_l2894_289426


namespace NUMINAMATH_CALUDE_tree_planting_theorem_l2894_289465

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The number of trees planted by 6th graders -/
def trees_6th : ℕ := 3 * trees_5th - 30

/-- The total number of trees planted by all three grades -/
def total_trees : ℕ := trees_4th + trees_5th + trees_6th

theorem tree_planting_theorem : total_trees = 240 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_theorem_l2894_289465


namespace NUMINAMATH_CALUDE_prob_two_hearts_one_spade_l2894_289499

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Fin 52)
  (ranks : Fin 13)
  (suits : Fin 4)

/-- Represents the suits in a deck -/
inductive Suit
| hearts
| diamonds
| clubs
| spades

/-- Defines the color of a suit -/
def suitColor (s : Suit) : Bool :=
  match s with
  | Suit.hearts | Suit.diamonds => true  -- Red
  | Suit.clubs | Suit.spades => false    -- Black

/-- Calculates the probability of drawing two hearts followed by a spade -/
def probabilityTwoHeartsOneSpade (d : Deck) : ℚ :=
  13 / 850

/-- Theorem stating the probability of drawing two hearts followed by a spade -/
theorem prob_two_hearts_one_spade (d : Deck) :
  probabilityTwoHeartsOneSpade d = 13 / 850 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_hearts_one_spade_l2894_289499


namespace NUMINAMATH_CALUDE_circular_garden_ratio_l2894_289412

theorem circular_garden_ratio (r : ℝ) (h : r = 10) : 
  (2 * π * r) / (π * r^2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_ratio_l2894_289412


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2894_289434

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- The condition "x = 2" for the given vectors -/
def condition (x : ℝ) : Prop := x = 2

/-- The vectors a and b as functions of x -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, condition x → are_parallel (a x) (b x)) ∧
  ¬(∀ x : ℝ, are_parallel (a x) (b x) → condition x) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2894_289434


namespace NUMINAMATH_CALUDE_fraction_of_72_l2894_289432

theorem fraction_of_72 : (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 6 : ℚ) * 72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_72_l2894_289432


namespace NUMINAMATH_CALUDE_watch_cost_price_l2894_289419

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (C : ℝ), 
  (C > 0) ∧ 
  (0.64 * C + 140 = 1.04 * C) ∧ 
  (C = 350) := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2894_289419


namespace NUMINAMATH_CALUDE_probability_one_of_each_type_l2894_289444

def total_silverware : ℕ := 30
def forks : ℕ := 10
def spoons : ℕ := 10
def knives : ℕ := 10

theorem probability_one_of_each_type (total_silverware forks spoons knives : ℕ) :
  total_silverware = forks + spoons + knives →
  (Nat.choose total_silverware 3 : ℚ) ≠ 0 →
  (forks * spoons * knives : ℚ) / Nat.choose total_silverware 3 = 500 / 203 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_type_l2894_289444


namespace NUMINAMATH_CALUDE_real_part_reciprocal_l2894_289461

theorem real_part_reciprocal (z : ℂ) (h : z = 1 - 2*I) : 
  (1 / z).re = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_real_part_reciprocal_l2894_289461
