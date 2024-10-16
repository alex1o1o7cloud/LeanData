import Mathlib

namespace NUMINAMATH_CALUDE_equality_condition_l1038_103836

theorem equality_condition (a b c d : ℝ) :
  a + b * c * d = (a + b) * (a + c) * (a + d) ↔ a^2 + a * (b + c + d) + b * c + b * d + c * d = 1 :=
sorry

end NUMINAMATH_CALUDE_equality_condition_l1038_103836


namespace NUMINAMATH_CALUDE_cos_forty_five_degrees_l1038_103863

theorem cos_forty_five_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_forty_five_degrees_l1038_103863


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1038_103860

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) :
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1038_103860


namespace NUMINAMATH_CALUDE_sum_of_c_values_l1038_103841

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ (x y : ℚ), y = x^2 - 11*x - c ∧ 
    ∀ z : ℚ, z^2 - 11*z - c = 0 ↔ (z = x ∨ z = y)) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ (x y : ℚ), y = x^2 - 11*x - c ∧ 
    ∀ z : ℚ, z^2 - 11*z - c = 0 ↔ (z = x ∨ z = y)) → 
    c ∈ S) ∧
  (S.sum id = 38) :=
sorry

end NUMINAMATH_CALUDE_sum_of_c_values_l1038_103841


namespace NUMINAMATH_CALUDE_gcd_180_450_l1038_103819

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_450_l1038_103819


namespace NUMINAMATH_CALUDE_johnson_family_seating_l1038_103892

/-- The number of ways to seat 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

theorem johnson_family_seating :
  seating_arrangements 5 4 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l1038_103892


namespace NUMINAMATH_CALUDE_right_angle_triangle_identification_l1038_103834

/-- Checks if three lengths can form a right-angled triangle -/
def isRightAngleTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_angle_triangle_identification :
  (¬ isRightAngleTriangle 2 3 4) ∧
  (¬ isRightAngleTriangle 3 3 4) ∧
  (isRightAngleTriangle 9 12 15) ∧
  (¬ isRightAngleTriangle 4 5 6) := by
  sorry

#check right_angle_triangle_identification

end NUMINAMATH_CALUDE_right_angle_triangle_identification_l1038_103834


namespace NUMINAMATH_CALUDE_min_y_value_l1038_103809

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 40*y) :
  ∃ (y_min : ℝ), y_min = 20 - Real.sqrt 464 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 16*x' + 40*y' → y' ≥ y_min := by
sorry

end NUMINAMATH_CALUDE_min_y_value_l1038_103809


namespace NUMINAMATH_CALUDE_cristinas_pace_is_5_l1038_103826

/-- Cristina's pace in meters per second -/
def cristinas_pace : ℝ := 5

/-- Nicky's head start in meters -/
def head_start : ℝ := 54

/-- Nicky's pace in meters per second -/
def nickys_pace : ℝ := 3

/-- Time in seconds before Cristina catches up to Nicky -/
def catch_up_time : ℝ := 27

/-- Theorem stating that Cristina's pace is 5 meters per second -/
theorem cristinas_pace_is_5 : 
  cristinas_pace * catch_up_time = nickys_pace * catch_up_time + head_start :=
by sorry

end NUMINAMATH_CALUDE_cristinas_pace_is_5_l1038_103826


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1038_103856

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b are parallel, prove that x = -6 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = -6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1038_103856


namespace NUMINAMATH_CALUDE_replaced_person_weight_l1038_103884

theorem replaced_person_weight
  (num_persons : ℕ)
  (avg_weight_increase : ℝ)
  (new_person_weight : ℝ) :
  num_persons = 5 →
  avg_weight_increase = 1.5 →
  new_person_weight = 72.5 →
  new_person_weight - (num_persons : ℝ) * avg_weight_increase = 65 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l1038_103884


namespace NUMINAMATH_CALUDE_proposition_p_and_not_q_l1038_103812

open Real

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 + 2*x + 5 ≤ 4) ∧ 
  (∀ x ∈ Set.Ioo 0 (π/2), sin x + 4/(sin x) > 4) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_and_not_q_l1038_103812


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l1038_103823

theorem solve_cubic_equation (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  (x + 1)^3 = x^3 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l1038_103823


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l1038_103853

/-- Given an ellipse with eccentricity e and focal length 2c, 
    prove that its standard equation is of the form x²/a² + y²/b² = 1 
    where a and b are the semi-major and semi-minor axes respectively. -/
theorem ellipse_standard_equation 
  (e : ℝ) 
  (c : ℝ) 
  (h_e : e = 2/3) 
  (h_c : 2*c = 16) : 
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ 
      (x^2/144 + y^2/80 = 1 ∨ x^2/80 + y^2/144 = 1)) := by
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l1038_103853


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_arithmetic_sequence_l1038_103873

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

/-- The 10th term of the arithmetic sequence with first term 10 and common difference -2 is -8 -/
theorem tenth_term_of_specific_arithmetic_sequence :
  arithmeticSequence 10 (-2) 10 = -8 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_arithmetic_sequence_l1038_103873


namespace NUMINAMATH_CALUDE_axis_of_symmetry_shifted_cosine_l1038_103857

open Real

theorem axis_of_symmetry_shifted_cosine :
  let f : ℝ → ℝ := λ x ↦ Real.sin (π / 2 - x)
  let g : ℝ → ℝ := λ x ↦ Real.cos (x + π / 6)
  ∀ x : ℝ, g (5 * π / 6 + x) = g (5 * π / 6 - x) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_shifted_cosine_l1038_103857


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l1038_103813

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 7 → a 5 = 13 → a 7 = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l1038_103813


namespace NUMINAMATH_CALUDE_max_diff_of_squares_l1038_103859

theorem max_diff_of_squares (n : ℕ) (h1 : n > 0) (h2 : n + (n + 1) < 150) :
  (∃ (m : ℕ), m > 0 ∧ m + (m + 1) < 150 ∧ (m + 1)^2 - m^2 > (n + 1)^2 - n^2) →
  (n + 1)^2 - n^2 ≤ 149 :=
sorry

end NUMINAMATH_CALUDE_max_diff_of_squares_l1038_103859


namespace NUMINAMATH_CALUDE_smallest_power_l1038_103846

theorem smallest_power : 2^55 < 3^44 ∧ 2^55 < 5^33 ∧ 2^55 < 6^22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_l1038_103846


namespace NUMINAMATH_CALUDE_twelve_chairs_subsets_l1038_103867

/-- The number of chairs in the circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets with at least four adjacent chairs -/
def subsetsWithAdjacentChairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs, the number of subsets with at least four adjacent chairs is 1776 -/
theorem twelve_chairs_subsets : subsetsWithAdjacentChairs n = 1776 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_subsets_l1038_103867


namespace NUMINAMATH_CALUDE_bird_sale_ratio_is_half_l1038_103804

/-- Represents the initial counts of animals in the pet store -/
structure InitialCounts where
  birds : ℕ
  puppies : ℕ
  cats : ℕ
  spiders : ℕ

/-- Represents the changes in animal counts -/
structure Changes where
  puppies_adopted : ℕ
  spiders_loose : ℕ

/-- Calculates the ratio of birds sold to initial birds -/
def bird_sale_ratio (initial : InitialCounts) (changes : Changes) (final_count : ℕ) : ℚ :=
  let total_initial := initial.birds + initial.puppies + initial.cats + initial.spiders
  let birds_sold := total_initial - changes.puppies_adopted - changes.spiders_loose - final_count
  birds_sold / initial.birds

/-- Theorem stating the ratio of birds sold to initial birds is 1:2 -/
theorem bird_sale_ratio_is_half 
  (initial : InitialCounts)
  (changes : Changes)
  (final_count : ℕ)
  (h_initial : initial = ⟨12, 9, 5, 15⟩)
  (h_changes : changes = ⟨3, 7⟩)
  (h_final : final_count = 25) :
  bird_sale_ratio initial changes final_count = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bird_sale_ratio_is_half_l1038_103804


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l1038_103816

theorem smallest_n_for_exact_tax : ∃ (x : ℕ+), (↑x : ℚ) * 105 / 100 = 21 ∧ 
  ∀ (n : ℕ+), n < 21 → ¬∃ (y : ℕ+), (↑y : ℚ) * 105 / 100 = ↑n :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l1038_103816


namespace NUMINAMATH_CALUDE_triangle_conversion_cost_l1038_103820

theorem triangle_conversion_cost 
  (side1 : ℝ) (side2 : ℝ) (angle : ℝ) (cost_per_sqm : ℝ) :
  side1 = 32 →
  side2 = 68 →
  angle = 30 * π / 180 →
  cost_per_sqm = 50 →
  (1/2 * side1 * side2 * Real.sin angle) * cost_per_sqm = 54400 :=
by sorry

end NUMINAMATH_CALUDE_triangle_conversion_cost_l1038_103820


namespace NUMINAMATH_CALUDE_marbles_distribution_l1038_103801

theorem marbles_distribution (x : ℚ) 
  (h1 : x > 0) 
  (h2 : (4 * x + 2) + (2 * x + 1) + 3 * x = 62) : 
  (4 * x + 2 = 254 / 9) ∧ (2 * x + 1 = 127 / 9) ∧ (3 * x = 177 / 9) :=
by sorry

end NUMINAMATH_CALUDE_marbles_distribution_l1038_103801


namespace NUMINAMATH_CALUDE_tangent_ratio_range_l1038_103800

open Real

-- Define the function f(x) = |e^x - 1|
noncomputable def f (x : ℝ) : ℝ := abs (exp x - 1)

-- Define the theorem
theorem tangent_ratio_range 
  (x₁ x₂ : ℝ) 
  (h₁ : x₁ < 0) 
  (h₂ : x₂ > 0) 
  (h_perp : (deriv f x₁) * (deriv f x₂) = -1) :
  ∃ (AM BN : ℝ), 
    AM > 0 ∧ BN > 0 ∧ 
    0 < AM / BN ∧ AM / BN < 1 :=
by sorry


end NUMINAMATH_CALUDE_tangent_ratio_range_l1038_103800


namespace NUMINAMATH_CALUDE_least_number_of_pennies_l1038_103803

theorem least_number_of_pennies :
  ∃ (p : ℕ), p > 0 ∧ p % 7 = 3 ∧ p % 4 = 1 ∧
  ∀ (q : ℕ), q > 0 ∧ q % 7 = 3 ∧ q % 4 = 1 → p ≤ q :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_number_of_pennies_l1038_103803


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1038_103864

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 7/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1038_103864


namespace NUMINAMATH_CALUDE_color_film_fraction_l1038_103897

theorem color_film_fraction (x y : ℝ) (h1 : x ≠ 0) : 
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (1 / 100) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected : ℝ) = 6 / 31 := by
  sorry

end NUMINAMATH_CALUDE_color_film_fraction_l1038_103897


namespace NUMINAMATH_CALUDE_fish_count_l1038_103839

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 8

/-- The number of fish Max has -/
def max_fish : ℕ := 15

/-- The total number of fish Lilly, Rosy, and Max have -/
def total_fish : ℕ := lilly_fish + rosy_fish + max_fish

theorem fish_count : total_fish = 33 := by sorry

end NUMINAMATH_CALUDE_fish_count_l1038_103839


namespace NUMINAMATH_CALUDE_base7_sum_theorem_l1038_103861

/-- Represents a single digit in base 7 --/
def Base7Digit := Fin 7

/-- Converts a base 7 number to base 10 --/
def toBase10 (x : Base7Digit) : Nat := x.val

/-- The equation 5XY₇ + 32₇ = 62X₇ in base 7 --/
def base7Equation (X Y : Base7Digit) : Prop :=
  (5 * 7 + toBase10 X) * 7 + toBase10 Y + 32 = (6 * 7 + 2) * 7 + toBase10 X

/-- Theorem stating that if X and Y satisfy the base 7 equation, then X + Y = 10 in base 10 --/
theorem base7_sum_theorem (X Y : Base7Digit) : 
  base7Equation X Y → toBase10 X + toBase10 Y = 10 := by
  sorry

end NUMINAMATH_CALUDE_base7_sum_theorem_l1038_103861


namespace NUMINAMATH_CALUDE_vanya_correct_answers_l1038_103830

/-- The number of questions Sasha asked Vanya -/
def total_questions : ℕ := 50

/-- The number of candies Vanya receives for a correct answer -/
def correct_reward : ℕ := 7

/-- The number of candies Vanya gives for an incorrect answer -/
def incorrect_penalty : ℕ := 3

/-- The number of questions Vanya answered correctly -/
def correct_answers : ℕ := 15

theorem vanya_correct_answers :
  correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty :=
by sorry

end NUMINAMATH_CALUDE_vanya_correct_answers_l1038_103830


namespace NUMINAMATH_CALUDE_smallest_with_property_l1038_103891

/-- A function that returns the list of digits of a natural number. -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if two lists of natural numbers are permutations of each other. -/
def is_permutation (l1 l2 : List ℕ) : Prop := sorry

/-- The property we're looking for: when multiplied by 9, the result has the same digits in a different order. -/
def has_property (n : ℕ) : Prop :=
  is_permutation (digits n) (digits (9 * n))

/-- The theorem stating that 1089 is the smallest natural number with the desired property. -/
theorem smallest_with_property :
  has_property 1089 ∧ ∀ m : ℕ, m < 1089 → ¬(has_property m) := by sorry

end NUMINAMATH_CALUDE_smallest_with_property_l1038_103891


namespace NUMINAMATH_CALUDE_max_dot_product_on_ellipse_l1038_103854

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

/-- Definition of the center O -/
def O : ℝ × ℝ := (0, 0)

/-- Definition of the left focus F -/
def F : ℝ × ℝ := (-1, 0)

/-- Definition of the dot product of OP and FP -/
def dot_product (x y : ℝ) : ℝ := x^2 + x + y^2

theorem max_dot_product_on_ellipse :
  ∃ (max : ℝ), max = 6 ∧
  ∀ (x y : ℝ), is_on_ellipse x y →
  dot_product x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_on_ellipse_l1038_103854


namespace NUMINAMATH_CALUDE_polynomial_division_condition_l1038_103810

theorem polynomial_division_condition (a b : ℝ) : 
  (∀ x : ℝ, (x - 1)^2 ∣ (a * x^4 + b * x^3 + 1)) ↔ (a = 3 ∧ b = -4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_condition_l1038_103810


namespace NUMINAMATH_CALUDE_power_two_mod_four_l1038_103874

theorem power_two_mod_four : 2^300 % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_two_mod_four_l1038_103874


namespace NUMINAMATH_CALUDE_sqrt_5_minus_1_over_2_gt_half_l1038_103840

theorem sqrt_5_minus_1_over_2_gt_half : 
  (4 < 5) → (5 < 9) → (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_minus_1_over_2_gt_half_l1038_103840


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l1038_103847

theorem sum_of_squares_problem (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l1038_103847


namespace NUMINAMATH_CALUDE_subset_condition_l1038_103849

def A (x : ℝ) : Prop := |2 * x - 1| < 1

def B (a x : ℝ) : Prop := x^2 - 2*a*x + a^2 - 1 > 0

theorem subset_condition (a : ℝ) :
  (∀ x, A x → B a x) ↔ (a ≤ -1 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_subset_condition_l1038_103849


namespace NUMINAMATH_CALUDE_area_inside_circle_outside_square_l1038_103895

/-- The area inside a circle of radius √3/3 but outside a square of side length 1, 
    when they share the same center. -/
theorem area_inside_circle_outside_square : 
  let square_side : ℝ := 1
  let circle_radius : ℝ := Real.sqrt 3 / 3
  let circle_area : ℝ := π * circle_radius^2
  let square_area : ℝ := square_side^2
  let area_difference : ℝ := circle_area - square_area
  area_difference = 2 * π / 9 - Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_inside_circle_outside_square_l1038_103895


namespace NUMINAMATH_CALUDE_book_pages_count_l1038_103862

/-- Given a book with pages numbered consecutively starting from 1,
    this function calculates the total number of digits used to number the pages. -/
def totalDigits (n : ℕ) : ℕ :=
  (n.min 9) + 
  (n - 9).max 0 * 2 + 
  (n - 99).max 0 * 3

/-- Theorem stating that a book has 369 pages if the total number of digits
    used in numbering is 999. -/
theorem book_pages_count : totalDigits 369 = 999 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1038_103862


namespace NUMINAMATH_CALUDE_twelve_roll_prob_l1038_103890

/-- Probability of a specific outcome on a standard six-sided die -/
def die_prob : ℚ := 1 / 6

/-- Probability of rolling any number except the previous one -/
def diff_prob : ℚ := 5 / 6

/-- Number of rolls before the 8th roll -/
def pre_8th_rolls : ℕ := 6

/-- Number of rolls between 8th and 12th (exclusive) -/
def post_8th_rolls : ℕ := 3

/-- The probability that the 12th roll is the last roll, given the 8th roll is a 4 -/
theorem twelve_roll_prob : 
  (1 : ℚ) * diff_prob ^ pre_8th_rolls * die_prob * diff_prob ^ post_8th_rolls * die_prob = 5^9 / 6^11 :=
by sorry

end NUMINAMATH_CALUDE_twelve_roll_prob_l1038_103890


namespace NUMINAMATH_CALUDE_group_commutativity_l1038_103893

theorem group_commutativity (G : Type*) [Group G] (m n : ℕ) 
  (coprime_mn : Nat.Coprime m n)
  (surj_m : Function.Surjective (fun x : G => x^(m+1)))
  (surj_n : Function.Surjective (fun x : G => x^(n+1)))
  (endo_m : ∀ (x y : G), (x*y)^(m+1) = x^(m+1) * y^(m+1))
  (endo_n : ∀ (x y : G), (x*y)^(n+1) = x^(n+1) * y^(n+1)) :
  ∀ (a b : G), a * b = b * a := by
  sorry

end NUMINAMATH_CALUDE_group_commutativity_l1038_103893


namespace NUMINAMATH_CALUDE_third_angle_is_90_l1038_103814

-- Define a triangle with two known angles
def Triangle (angle1 angle2 : ℝ) :=
  { angle3 : ℝ // angle1 + angle2 + angle3 = 180 }

-- Theorem: In a triangle with angles of 30 and 60 degrees, the third angle is 90 degrees
theorem third_angle_is_90 :
  ∀ (t : Triangle 30 60), t.val = 90 := by
  sorry

end NUMINAMATH_CALUDE_third_angle_is_90_l1038_103814


namespace NUMINAMATH_CALUDE_initial_solution_volume_l1038_103831

theorem initial_solution_volume 
  (V : ℝ)  -- Initial volume in liters
  (h1 : 0.20 * V + 3.6 = 0.50 * (V + 3.6))  -- Equation representing the alcohol balance
  : V = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_solution_volume_l1038_103831


namespace NUMINAMATH_CALUDE_sons_age_l1038_103848

theorem sons_age (father_age son_age : ℕ) : 
  (father_age + 6 + son_age + 6 = 68) → 
  (father_age = 6 * son_age) → 
  son_age = 8 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1038_103848


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1038_103845

theorem cistern_filling_time (p q : ℝ) (h1 : q = 15) (h2 : 2/p + 2/q + 10.5/q = 1) : p = 12 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1038_103845


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1038_103868

theorem trigonometric_identity (α : Real) 
  (h1 : Real.tan (α + π/4) = 1/2) 
  (h2 : -π/2 < α) 
  (h3 : α < 0) : 
  Real.sin (2*α) + 2 * (Real.sin α)^2 = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1038_103868


namespace NUMINAMATH_CALUDE_A_equals_B_l1038_103802

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem A_equals_B : A = B := by
  sorry

end NUMINAMATH_CALUDE_A_equals_B_l1038_103802


namespace NUMINAMATH_CALUDE_triangle_inequality_l1038_103880

/-- Given a triangle ABC with sides a, b, c, and inradius r, 
    prove that 24√3 r³ ≤ (-a+b+c)(a-b+c)(a+b-c) --/
theorem triangle_inequality (a b c r : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_r : 0 < r) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  24 * Real.sqrt 3 * r^3 ≤ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1038_103880


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_iff_l1038_103824

/-- Represents an ellipse equation of the form mx^2 + ny^2 = 1 --/
structure EllipseEquation (m n : ℝ) where
  eq : ∀ x y : ℝ, m * x^2 + n * y^2 = 1

/-- Predicate to check if an ellipse has foci on the y-axis --/
def hasFociOnYAxis (m n : ℝ) : Prop :=
  m > n ∧ n > 0

/-- Theorem stating that m > n > 0 is necessary and sufficient for 
    mx^2 + ny^2 = 1 to represent an ellipse with foci on the y-axis --/
theorem ellipse_foci_on_y_axis_iff (m n : ℝ) :
  hasFociOnYAxis m n ↔ ∃ (e : EllipseEquation m n), True :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_iff_l1038_103824


namespace NUMINAMATH_CALUDE_total_dry_grapes_weight_l1038_103828

/-- Calculates the total weight of Dry Grapes after dehydrating Fresh Grapes Type A and B -/
theorem total_dry_grapes_weight 
  (water_content_A : Real) 
  (water_content_B : Real)
  (weight_A : Real) 
  (weight_B : Real) :
  water_content_A = 0.92 →
  water_content_B = 0.88 →
  weight_A = 30 →
  weight_B = 50 →
  (1 - water_content_A) * weight_A + (1 - water_content_B) * weight_B = 8.4 :=
by sorry

end NUMINAMATH_CALUDE_total_dry_grapes_weight_l1038_103828


namespace NUMINAMATH_CALUDE_binomial_difference_divisibility_l1038_103899

theorem binomial_difference_divisibility (p k : ℕ) (h_prime : Nat.Prime p) (h_k_lower : 2 ≤ k) (h_k_upper : k ≤ p - 2) :
  ∃ m : ℤ, (Nat.choose (p - k + 1) k : ℤ) - (Nat.choose (p - k - 1) (k - 2) : ℤ) = m * p := by
  sorry

end NUMINAMATH_CALUDE_binomial_difference_divisibility_l1038_103899


namespace NUMINAMATH_CALUDE_eighth_root_unity_l1038_103885

theorem eighth_root_unity : ∃ n : ℕ, n ∈ Finset.range 8 ∧
  (Complex.I + Complex.tan (π / 8)) / (Complex.tan (π / 8) - Complex.I) =
  Complex.exp (2 * n * π * Complex.I / 8) := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_unity_l1038_103885


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1038_103822

theorem constant_term_expansion (x : ℝ) : 
  let expression := (x - 4 + 4 / x)^3
  ∃ (a b c : ℝ), expression = a * x^3 + b * x^2 + c * x - 160
  := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1038_103822


namespace NUMINAMATH_CALUDE_marbles_with_red_count_l1038_103827

def total_marbles : ℕ := 10
def red_marbles : ℕ := 1
def marbles_to_choose : ℕ := 5

theorem marbles_with_red_count :
  (Nat.choose total_marbles marbles_to_choose) - 
  (Nat.choose (total_marbles - red_marbles) marbles_to_choose) = 126 := by
  sorry

end NUMINAMATH_CALUDE_marbles_with_red_count_l1038_103827


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l1038_103879

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  sideLength : ℕ
  smallCubeSize : ℕ
  largeCubeSize : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (c : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 4 with specific corner removals has 48 edges -/
theorem modified_cube_edge_count :
  let c : ModifiedCube := ⟨4, 1, 2⟩
  edgeCount c = 48 := by sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l1038_103879


namespace NUMINAMATH_CALUDE_different_color_probability_l1038_103842

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_different := (blue_chips * (total_chips - blue_chips) +
                      red_chips * (total_chips - red_chips) +
                      yellow_chips * (total_chips - yellow_chips) +
                      green_chips * (total_chips - green_chips)) /
                     (total_chips * total_chips)
  p_different = 119 / 162 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l1038_103842


namespace NUMINAMATH_CALUDE_circle_tangent_intersection_theorem_l1038_103825

def circle_tangent_intersection (t₁ t₂ : ℂ) : Prop :=
  let r : ℝ := Complex.abs t₁  -- radius of the circle
  (Complex.abs t₁ = r) →  -- t₁ is on the circle
  (Complex.abs t₂ = r) →  -- t₂ is on the circle
  (t₁ ≠ t₂) →  -- t₁ and t₂ are distinct points
  ∃ (P : ℂ), P = (2 * t₁ * t₂) / (t₁ + t₂)  -- P is the intersection point

theorem circle_tangent_intersection_theorem (t₁ t₂ : ℂ) :
  circle_tangent_intersection t₁ t₂ := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_intersection_theorem_l1038_103825


namespace NUMINAMATH_CALUDE_mike_car_parts_cost_l1038_103878

-- Define the cost of speakers
def speaker_cost : ℚ := 118.54

-- Define the cost of tires
def tire_cost : ℚ := 106.33

-- Define the total cost
def total_cost : ℚ := speaker_cost + tire_cost

-- Theorem to prove
theorem mike_car_parts_cost : total_cost = 224.87 := by
  sorry

end NUMINAMATH_CALUDE_mike_car_parts_cost_l1038_103878


namespace NUMINAMATH_CALUDE_triangle_circumcircle_distance_sum_bounds_l1038_103855

theorem triangle_circumcircle_distance_sum_bounds :
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (2 * Real.sqrt 3, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ → ℝ × ℝ := fun θ ↦ (Real.sqrt 3 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)
  ∀ θ : ℝ,
    let P := C θ
    let dist_squared (X Y : ℝ × ℝ) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2
    let sum := dist_squared P O + dist_squared P A + dist_squared P B
    sum ≤ 32 ∧ sum ≥ 16 ∧ (∃ θ₁ θ₂, C θ₁ = 32 ∧ C θ₂ = 16) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_circumcircle_distance_sum_bounds_l1038_103855


namespace NUMINAMATH_CALUDE_employee_salary_problem_l1038_103807

theorem employee_salary_problem (num_employees : ℕ) (salary_increase : ℕ) (manager_salary : ℕ) :
  num_employees = 24 →
  salary_increase = 400 →
  manager_salary = 11500 →
  ∃ (avg_salary : ℕ),
    avg_salary * num_employees + manager_salary = (avg_salary + salary_increase) * (num_employees + 1) ∧
    avg_salary = 1500 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l1038_103807


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l1038_103882

open Complex

theorem max_abs_z_on_circle (z : ℂ) : 
  (abs (z - I) = abs (3 - 4*I)) → (abs z ≤ 6) ∧ (∃ w : ℂ, abs (w - I) = abs (3 - 4*I) ∧ abs w = 6) := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l1038_103882


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l1038_103883

def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

theorem hamburgers_left_over : hamburgers_made - hamburgers_served = 6 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l1038_103883


namespace NUMINAMATH_CALUDE_candle_weight_theorem_l1038_103837

/-- The weight of beeswax used in each candle, in ounces. -/
def beeswax_weight : ℕ := 8

/-- The weight of coconut oil used in each candle, in ounces. -/
def coconut_oil_weight : ℕ := 1

/-- The number of candles Ethan makes. -/
def num_candles : ℕ := 10 - 3

/-- The total weight of one candle, in ounces. -/
def candle_weight : ℕ := beeswax_weight + coconut_oil_weight

/-- The combined weight of all candles, in ounces. -/
def total_weight : ℕ := num_candles * candle_weight

theorem candle_weight_theorem : total_weight = 63 := by
  sorry

end NUMINAMATH_CALUDE_candle_weight_theorem_l1038_103837


namespace NUMINAMATH_CALUDE_minimum_rental_fee_is_3520_l1038_103894

/-- Represents a bus type with its seat capacity and rental fee. -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting a given number of people
    using two types of buses. -/
def minimumRentalFee (people : ℕ) (busA : BusType) (busB : BusType) : ℕ :=
  let totalBuses := 8
  let x := 4  -- number of Type A buses
  x * busA.fee + (totalBuses - x) * busB.fee

/-- Theorem stating that the minimum rental fee for 360 people using the given bus types is 3520 yuan. -/
theorem minimum_rental_fee_is_3520 :
  let people := 360
  let busA := BusType.mk 40 400
  let busB := BusType.mk 50 480
  minimumRentalFee people busA busB = 3520 := by
  sorry

#eval minimumRentalFee 360 (BusType.mk 40 400) (BusType.mk 50 480)

end NUMINAMATH_CALUDE_minimum_rental_fee_is_3520_l1038_103894


namespace NUMINAMATH_CALUDE_distinct_sequences_count_l1038_103888

/-- The number of sides on the die -/
def die_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 10

/-- The number of distinct sequences when rolling a die -/
def num_sequences : ℕ := die_sides ^ num_rolls

theorem distinct_sequences_count : num_sequences = 60466176 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sequences_count_l1038_103888


namespace NUMINAMATH_CALUDE_sin_taylor_expansion_at_3_l1038_103886

open Complex

/-- Taylor series expansion of sine function around z = 3 -/
theorem sin_taylor_expansion_at_3 (z : ℂ) : 
  sin z = (sin 3 * (∑' n, ((-1)^n / (2*n).factorial : ℂ) * (z - 3)^(2*n))) + 
          (cos 3 * (∑' n, ((-1)^n / (2*n + 1).factorial : ℂ) * (z - 3)^(2*n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_sin_taylor_expansion_at_3_l1038_103886


namespace NUMINAMATH_CALUDE_claire_male_pets_l1038_103858

theorem claire_male_pets (total_pets : ℕ) (gerbils : ℕ) (hamsters : ℕ)
  (h_total : total_pets = 92)
  (h_only_gerbils_hamsters : total_pets = gerbils + hamsters)
  (h_gerbils : gerbils = 68)
  (h_male_gerbils : ℕ → ℕ := λ x => x / 4)
  (h_male_hamsters : ℕ → ℕ := λ x => x / 3)
  : h_male_gerbils gerbils + h_male_hamsters hamsters = 25 := by
  sorry

end NUMINAMATH_CALUDE_claire_male_pets_l1038_103858


namespace NUMINAMATH_CALUDE_largest_angle_of_inclination_l1038_103838

-- Define the angle of inclination for a line given its slope
noncomputable def angle_of_inclination (slope : ℝ) : ℝ :=
  Real.arctan slope * (180 / Real.pi)

-- Define the lines
def line_A : ℝ → ℝ := λ x => -x + 1
def line_B : ℝ → ℝ := λ x => x + 1
def line_C : ℝ → ℝ := λ x => 2*x + 1
def line_D : ℝ → ℝ := λ _ => 1

-- Theorem statement
theorem largest_angle_of_inclination :
  let angle_A := angle_of_inclination (-1)
  let angle_B := angle_of_inclination 1
  let angle_C := angle_of_inclination 2
  let angle_D := 90
  angle_A > angle_B ∧ angle_A > angle_C ∧ angle_A > angle_D :=
by sorry


end NUMINAMATH_CALUDE_largest_angle_of_inclination_l1038_103838


namespace NUMINAMATH_CALUDE_triangular_prism_skew_lines_l1038_103872

/-- A triangular prism -/
structure TriangularPrism where
  vertices : Finset (ℝ × ℝ × ℝ)
  edges : Finset (Finset (ℝ × ℝ × ℝ))
  is_valid : vertices.card = 6 ∧ edges.card = 9

/-- A line in 3D space -/
def Line3D := Finset (ℝ × ℝ × ℝ)

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- The set of all lines passing through any two vertices of the prism -/
def all_lines (p : TriangularPrism) : Finset Line3D := sorry

/-- The set of all pairs of skew lines in the prism -/
def skew_line_pairs (p : TriangularPrism) : Finset (Line3D × Line3D) := sorry

theorem triangular_prism_skew_lines (p : TriangularPrism) :
  (all_lines p).card = 15 → (skew_line_pairs p).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_skew_lines_l1038_103872


namespace NUMINAMATH_CALUDE_remainder_problem_l1038_103881

theorem remainder_problem (t : ℕ) :
  let n : ℤ := 209 * t + 23
  (n % 19 = 4) ∧ (n % 11 = 1) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1038_103881


namespace NUMINAMATH_CALUDE_f_upper_bound_f_max_value_condition_l1038_103805

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3 - abs (2 * x - 1)

-- Part 1
theorem f_upper_bound :
  ∀ x : ℝ, (x ≤ 0 ∨ x ≥ 2) → f 1 x ≤ 2 :=
sorry

-- Part 2
theorem f_max_value_condition :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_f_max_value_condition_l1038_103805


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1038_103808

theorem fractional_equation_solution :
  ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1038_103808


namespace NUMINAMATH_CALUDE_correspondence_theorem_l1038_103843

theorem correspondence_theorem (m n : ℕ) (l : ℕ) 
  (h1 : l ≥ m * (n / 2))
  (h2 : l ≤ n * (m / 2)) :
  l = m * (n / 2) ∧ l = n * (m / 2) :=
sorry

end NUMINAMATH_CALUDE_correspondence_theorem_l1038_103843


namespace NUMINAMATH_CALUDE_soccer_ball_cost_l1038_103844

theorem soccer_ball_cost (ball_cost shirt_cost : ℝ) : 
  ball_cost + shirt_cost = 100 →
  2 * ball_cost + 3 * shirt_cost = 262 →
  ball_cost = 38 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_l1038_103844


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1038_103811

theorem quadratic_discriminant (a b c : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  |x₂ - x₁| = 2 →
  b^2 - 4*a*c = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1038_103811


namespace NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l1038_103852

/-- Given a quadratic function f(x) = x^2 + ax + b - 3 that passes through (2, 0),
    the minimum value of a^2 + b^2 is 1/5 -/
theorem min_value_of_a2_plus_b2 (a b : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + b - 3 = 0) → x = 2) → 
  (∃ m : ℝ, m = (1 : ℝ) / 5 ∧ ∀ a' b' : ℝ, (∀ x : ℝ, (x^2 + a'*x + b' - 3 = 0) → x = 2) → a'^2 + b'^2 ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l1038_103852


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1038_103870

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 + I) / (1 + 2*I) ∧ z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1038_103870


namespace NUMINAMATH_CALUDE_mary_juan_income_ratio_l1038_103829

/-- Given that Mary's income is 60% more than Tim's income, and Tim's income is 60% less than Juan's income,
    prove that Mary's income is 64% of Juan's income. -/
theorem mary_juan_income_ratio (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan)
  (h2 : mary = 1.6 * tim) : 
  mary = 0.64 * juan := by
  sorry

end NUMINAMATH_CALUDE_mary_juan_income_ratio_l1038_103829


namespace NUMINAMATH_CALUDE_point_transformation_l1038_103887

def rotate_z (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def initial_point : ℝ × ℝ × ℝ := (2, 2, -1)

def final_point : ℝ × ℝ × ℝ := (-2, 2, -1)

theorem point_transformation :
  (reflect_xy ∘ rotate_z ∘ reflect_yz ∘ reflect_xy ∘ rotate_z) initial_point = final_point := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1038_103887


namespace NUMINAMATH_CALUDE_max_distance_between_C1_and_C2_l1038_103866

-- Define the curves C1 and C2
def C1 (ρ θ : ℝ) : Prop := ρ + 6 * Real.sin θ + 8 / ρ = 0
def C2 (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define a point on C1
def point_on_C1 (x y : ℝ) : Prop :=
  ∃ (ρ θ : ℝ), C1 ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Define a point on C2
def point_on_C2 (x y : ℝ) : Prop := C2 x y

-- State the theorem
theorem max_distance_between_C1_and_C2 :
  ∃ (max_dist : ℝ),
    max_dist = Real.sqrt 65 / 2 + 1 ∧
    (∀ (x1 y1 x2 y2 : ℝ),
      point_on_C1 x1 y1 → point_on_C2 x2 y2 →
      Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) ≤ max_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      point_on_C1 x1 y1 ∧ point_on_C2 x2 y2 ∧
      Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = max_dist) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_C1_and_C2_l1038_103866


namespace NUMINAMATH_CALUDE_no_cross_sum_2018_l1038_103806

theorem no_cross_sum_2018 (n : ℕ) (h : n ∈ Finset.range 4901) : 5 * n ≠ 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_cross_sum_2018_l1038_103806


namespace NUMINAMATH_CALUDE_rectangle_width_l1038_103869

theorem rectangle_width (area : ℝ) (length width : ℝ) : 
  area = 63 →
  width = length - 2 →
  area = length * width →
  width = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1038_103869


namespace NUMINAMATH_CALUDE_perpendicular_vectors_trig_equality_l1038_103832

/-- Given two perpendicular vectors a and b, prove that 
    (sin³α + cos³α) / (sinα - cosα) = 9/5 -/
theorem perpendicular_vectors_trig_equality 
  (a b : ℝ × ℝ) 
  (h1 : a = (4, -2)) 
  (h2 : ∃ α : ℝ, b = (Real.cos α, Real.sin α)) 
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ∃ α : ℝ, (Real.sin α)^3 + (Real.cos α)^3 = 9/5 * ((Real.sin α) - (Real.cos α)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_trig_equality_l1038_103832


namespace NUMINAMATH_CALUDE_variables_positively_correlated_l1038_103835

/-- Represents a simple linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Defines positive correlation between variables in a linear regression model -/
def positively_correlated (model : LinearRegression) : Prop :=
  model.slope > 0

/-- The specific linear regression model given in the problem -/
def given_model : LinearRegression :=
  { slope := 0.5, intercept := 2 }

/-- Theorem stating that the variables in the given model are positively correlated -/
theorem variables_positively_correlated : 
  positively_correlated given_model := by sorry

end NUMINAMATH_CALUDE_variables_positively_correlated_l1038_103835


namespace NUMINAMATH_CALUDE_intersection_and_complement_union_l1038_103871

-- Define the universe U as the real numbers
def U := ℝ

-- Define set M
def M : Set ℝ := {x | x ≥ 1}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_and_complement_union :
  (M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 5}) ∧
  ((Mᶜ ∪ Nᶜ) = {x : ℝ | x < 1 ∨ x ≥ 5}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_union_l1038_103871


namespace NUMINAMATH_CALUDE_restaurant_bill_entree_cost_l1038_103851

/-- Given the conditions of a restaurant bill, prove the cost of each entree -/
theorem restaurant_bill_entree_cost 
  (appetizer_cost : ℝ)
  (tip_percentage : ℝ)
  (total_spent : ℝ)
  (num_entrees : ℕ)
  (h_appetizer : appetizer_cost = 10)
  (h_tip : tip_percentage = 0.2)
  (h_total : total_spent = 108)
  (h_num_entrees : num_entrees = 4) :
  ∃ (entree_cost : ℝ), 
    entree_cost * num_entrees + appetizer_cost + 
    (entree_cost * num_entrees + appetizer_cost) * tip_percentage = total_spent ∧
    entree_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_entree_cost_l1038_103851


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1038_103815

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = Real.sqrt 3 - 1) : 
  (a^2 + a) * ((a + 1) / a) = 3 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = 1/2) : 
  (a + 1) / (a^2 - 1) - (a + 1) / (1 - a) = -5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1038_103815


namespace NUMINAMATH_CALUDE_area_of_shaded_region_l1038_103898

/-- A line defined by two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The area bounded by two lines -/
def area_between_lines (l1 l2 : Line) : ℝ := sorry

/-- The first line passing through (0,3) and (10,2) -/
def line1 : Line := { x1 := 0, y1 := 3, x2 := 10, y2 := 2 }

/-- The second line passing through (0,5) and (5,0) -/
def line2 : Line := { x1 := 0, y1 := 5, x2 := 5, y2 := 0 }

theorem area_of_shaded_region :
  area_between_lines line1 line2 = 5/4 := by sorry

end NUMINAMATH_CALUDE_area_of_shaded_region_l1038_103898


namespace NUMINAMATH_CALUDE_pencil_cost_is_13_l1038_103865

/-- Represents the data for the pencil purchase problem -/
structure PencilPurchaseData where
  total_students : ℕ
  buyers : ℕ
  total_cost : ℕ
  pencil_cost : ℕ
  pencils_per_student : ℕ

/-- The conditions of the pencil purchase problem -/
def pencil_purchase_conditions (data : PencilPurchaseData) : Prop :=
  data.total_students = 50 ∧
  data.buyers > data.total_students / 2 ∧
  data.pencil_cost > data.pencils_per_student ∧
  data.buyers * data.pencil_cost * data.pencils_per_student = data.total_cost ∧
  data.total_cost = 2275

/-- The theorem stating that under the given conditions, the pencil cost is 13 cents -/
theorem pencil_cost_is_13 (data : PencilPurchaseData) :
  pencil_purchase_conditions data → data.pencil_cost = 13 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_is_13_l1038_103865


namespace NUMINAMATH_CALUDE_permutations_formula_l1038_103817

-- Define the number of permutations
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem permutations_formula {n k : ℕ} (h : 1 ≤ k ∧ k ≤ n) :
  permutations n k = (Nat.factorial n) / (Nat.factorial (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_permutations_formula_l1038_103817


namespace NUMINAMATH_CALUDE_program_output_l1038_103821

theorem program_output : 
  let a₀ := 10
  let b := a₀ - 8
  let a₁ := a₀ - b
  a₁ = 8 := by sorry

end NUMINAMATH_CALUDE_program_output_l1038_103821


namespace NUMINAMATH_CALUDE_car_speed_problem_l1038_103876

/-- Given a car traveling for two hours, where its speed in the second hour is 30 km/h
    and its average speed over the two hours is 25 km/h, prove that the speed of the car
    in the first hour must be 20 km/h. -/
theorem car_speed_problem (first_hour_speed : ℝ) : 
  (first_hour_speed + 30) / 2 = 25 → first_hour_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1038_103876


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l1038_103889

theorem ellipse_hyperbola_product (a b : ℝ) 
  (h_ellipse : b^2 - a^2 = 25)
  (h_hyperbola : a^2 + b^2 = 64) : 
  |a * b| = Real.sqrt (3461 / 4) := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l1038_103889


namespace NUMINAMATH_CALUDE_prob_different_suits_expanded_deck_l1038_103877

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Calculates the probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  let remaining_cards := d.total_cards - 1
  let different_suit_cards := d.total_cards - d.cards_per_suit
  different_suit_cards / remaining_cards

/-- Theorem: The probability of drawing two cards of different suits
    from a 78-card deck with 6 suits of 13 cards each is 65/77 -/
theorem prob_different_suits_expanded_deck :
  let d : Deck := ⟨78, 6, 13, rfl⟩
  prob_different_suits d = 65 / 77 := by sorry

end NUMINAMATH_CALUDE_prob_different_suits_expanded_deck_l1038_103877


namespace NUMINAMATH_CALUDE_square_shape_side_length_l1038_103818

theorem square_shape_side_length (x : ℝ) :
  x > 0 →
  x - 3 > 0 →
  (x + (x - 1)) = ((x - 2) + (x - 3) + 4) →
  1 = x - 3 →
  4 = (x + (x - 1)) - (2 * x - 5) := by
sorry

end NUMINAMATH_CALUDE_square_shape_side_length_l1038_103818


namespace NUMINAMATH_CALUDE_relationship_abc_l1038_103896

open Real

theorem relationship_abc (a b c : ℝ) (ha : a = 2^(log 2)) (hb : b = 2 + 2*log 2) (hc : c = (log 2)^2) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1038_103896


namespace NUMINAMATH_CALUDE_height_classification_groups_l1038_103850

/-- Given the heights of students in a class, calculate the number of groups needed for classification --/
theorem height_classification_groups 
  (tallest_height : ℕ) 
  (shortest_height : ℕ) 
  (class_width : ℕ) 
  (h1 : tallest_height = 175) 
  (h2 : shortest_height = 150) 
  (h3 : class_width = 3) : 
  ℕ := by
  sorry

#check height_classification_groups

end NUMINAMATH_CALUDE_height_classification_groups_l1038_103850


namespace NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l1038_103833

-- Define the number of songs
def num_songs : ℕ := 8

-- Define the length of the shortest song
def shortest_song_length : ℕ := 1

-- Define the length of the favorite song
def favorite_song_length : ℕ := 5

-- Define the duration we're considering
def considered_duration : ℕ := 7

-- Function to calculate song length based on its position
def song_length (position : ℕ) : ℕ :=
  shortest_song_length + position - 1

-- Theorem stating the probability of not hearing every second of the favorite song
theorem probability_not_hearing_favorite_song :
  let total_arrangements := num_songs.factorial
  let favorable_arrangements := (num_songs - 1).factorial + (num_songs - 2).factorial
  (total_arrangements - favorable_arrangements) / total_arrangements = 6 / 7 :=
sorry

end NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l1038_103833


namespace NUMINAMATH_CALUDE_sandy_fingernail_record_l1038_103875

/-- Calculates the length of fingernails after a given number of years -/
def fingernail_length (current_age : ℕ) (target_age : ℕ) (current_length : ℝ) (growth_rate : ℝ) : ℝ :=
  current_length + (target_age - current_age) * 12 * growth_rate

/-- Proves that Sandy's fingernails will be 26 inches long at age 32, given the initial conditions -/
theorem sandy_fingernail_record :
  fingernail_length 12 32 2 0.1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fingernail_record_l1038_103875
