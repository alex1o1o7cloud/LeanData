import Mathlib

namespace NUMINAMATH_CALUDE_smallest_other_integer_l2587_258714

theorem smallest_other_integer (a b x : ℕ+) : 
  (a = 36 ∨ b = 36) →
  Nat.gcd a b = x + 6 →
  Nat.lcm a b = x * (x + 6) →
  (a ≠ 36 → a ≥ 24) ∧ (b ≠ 36 → b ≥ 24) :=
sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l2587_258714


namespace NUMINAMATH_CALUDE_cuboid_volume_l2587_258718

/-- 
Theorem: Volume of a specific cuboid

Given a cuboid with the following properties:
1. The length and width are equal.
2. The length is 2 cm more than the height.
3. When the height is increased by 2 cm (making it equal to the length and width),
   the surface area increases by 56 square centimeters.

This theorem proves that the volume of the original cuboid is 245 cubic centimeters.
-/
theorem cuboid_volume (l w h : ℝ) : 
  l = w → -- length equals width
  l = h + 2 → -- length is 2 more than height
  6 * l^2 - 2 * (l^2 - (l-2)^2) = 56 → -- surface area increase condition
  l * w * h = 245 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l2587_258718


namespace NUMINAMATH_CALUDE_oranges_left_to_sell_l2587_258753

theorem oranges_left_to_sell (x : ℕ) (h : x ≥ 7) :
  let total := 12 * x
  let friend1 := (1 / 4 : ℚ) * total
  let friend2 := (1 / 6 : ℚ) * total
  let charity := (1 / 8 : ℚ) * total
  let remaining_after_giving := total - friend1 - friend2 - charity
  let sold_yesterday := (3 / 7 : ℚ) * remaining_after_giving
  let remaining_after_selling := remaining_after_giving - sold_yesterday
  let eaten_by_birds := (1 / 10 : ℚ) * remaining_after_selling
  let remaining_after_birds := remaining_after_selling - eaten_by_birds
  remaining_after_birds - 4 = (3.0214287 : ℚ) * x - 4 := by sorry

end NUMINAMATH_CALUDE_oranges_left_to_sell_l2587_258753


namespace NUMINAMATH_CALUDE_inverse_proportion_through_neg_one_three_l2587_258782

/-- An inverse proportion function passing through (-1, 3) has k = -3 --/
theorem inverse_proportion_through_neg_one_three (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (k / x = 3 ↔ x = -1)) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_neg_one_three_l2587_258782


namespace NUMINAMATH_CALUDE_expansion_equality_l2587_258771

theorem expansion_equality (x : ℝ) : (1 + x^2) * (1 - x^4) = 1 + x^2 - x^4 - x^6 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l2587_258771


namespace NUMINAMATH_CALUDE_polynomial_real_root_l2587_258779

/-- The polynomial in question -/
def P (a x : ℝ) : ℝ := x^4 + a*x^3 - 2*x^2 + a*x + 2

/-- The theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, P a x = 0) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l2587_258779


namespace NUMINAMATH_CALUDE_right_triangle_inequality_right_triangle_inequality_equality_l2587_258785

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3*a + 4*b ≤ 5*c :=
by sorry

theorem right_triangle_inequality_equality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3*a + 4*b = 5*c ↔ ∃ (k : ℝ), k > 0 ∧ a = 3*k ∧ b = 4*k ∧ c = 5*k :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_right_triangle_inequality_equality_l2587_258785


namespace NUMINAMATH_CALUDE_quarters_to_nickels_difference_l2587_258762

/-- The difference in money (in nickels) between two people with different numbers of quarters -/
theorem quarters_to_nickels_difference (q : ℚ) : 
  5 * ((7 * q + 2) - (3 * q + 7)) = 20 * (q - 1.25) := by
  sorry

end NUMINAMATH_CALUDE_quarters_to_nickels_difference_l2587_258762


namespace NUMINAMATH_CALUDE_root_range_implies_k_range_l2587_258710

theorem root_range_implies_k_range :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, 
    x₁^2 + (k-3)*x₁ + k^2 = 0 ∧
    x₂^2 + (k-3)*x₂ + k^2 = 0 ∧
    x₁ < 1 ∧ x₂ > 1 ∧ x₁ ≠ x₂) →
  k > -2 ∧ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_root_range_implies_k_range_l2587_258710


namespace NUMINAMATH_CALUDE_cone_volume_l2587_258781

-- Define the right triangle
structure RightTriangle where
  area : ℝ
  centroidCircumference : ℝ

-- Define the cone formed by rotating the right triangle
structure Cone where
  triangle : RightTriangle

-- Define the volume of the cone
def volume (c : Cone) : ℝ := c.triangle.area * c.triangle.centroidCircumference

-- Theorem statement
theorem cone_volume (c : Cone) : volume c = c.triangle.area * c.triangle.centroidCircumference := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2587_258781


namespace NUMINAMATH_CALUDE_smallest_student_group_l2587_258744

theorem smallest_student_group (n : ℕ) : n = 46 ↔ 
  (n > 0) ∧ 
  (n % 3 = 1) ∧ 
  (n % 6 = 4) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 1 → m % 6 = 4 → m % 8 = 5 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_student_group_l2587_258744


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2587_258732

theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  interest = 4016.25 →
  rate = 0.01 →
  time = 5 →
  interest = principal * rate * time →
  principal = 80325 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2587_258732


namespace NUMINAMATH_CALUDE_odd_function_unique_m_l2587_258757

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The given function f(x) parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 6)

/-- Theorem stating that m = 6 is the only value that makes f an odd function -/
theorem odd_function_unique_m :
  ∃! m : ℝ, IsOddFunction (f m) ∧ m = 6 :=
sorry

end NUMINAMATH_CALUDE_odd_function_unique_m_l2587_258757


namespace NUMINAMATH_CALUDE_sum_difference_equals_product_l2587_258705

-- Define the sequence
def seq : ℕ → ℕ
  | 0 => 0
  | n + 1 => n / 2 + 1

-- Define f(n) as the sum of the first n terms of the sequence
def f (n : ℕ) : ℕ := (List.range n).map seq |>.sum

-- Theorem statement
theorem sum_difference_equals_product {s t : ℕ} (hs : s > 0) (ht : t > 0) (hst : s > t) :
  f (s + t) - f (s - t) = s * t := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_product_l2587_258705


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l2587_258777

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and intersection relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- State the theorem
theorem line_parallel_to_intersection
  (m n : Line)
  (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_intersection : intersection α β n)
  (h_m_parallel_α : parallel_plane m α)
  (h_m_parallel_β : parallel_plane m β) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l2587_258777


namespace NUMINAMATH_CALUDE_average_height_is_10_8_l2587_258775

def tree_heights (h1 h2 h3 h4 h5 : ℕ) : Prop :=
  h2 = 18 ∧
  (h1 = 3 * h2 ∨ h1 * 3 = h2) ∧
  (h2 = 3 * h3 ∨ h2 * 3 = h3) ∧
  (h3 = 3 * h4 ∨ h3 * 3 = h4) ∧
  (h4 = 3 * h5 ∨ h4 * 3 = h5)

theorem average_height_is_10_8 :
  ∃ (h1 h2 h3 h4 h5 : ℕ), tree_heights h1 h2 h3 h4 h5 ∧
  (h1 + h2 + h3 + h4 + h5) / 5 = 54 / 5 :=
sorry

end NUMINAMATH_CALUDE_average_height_is_10_8_l2587_258775


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2587_258709

/-- Proves that mixing 300 mL of 10% alcohol solution with 200 mL of 30% alcohol solution results in 18% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 300
  let y_volume : ℝ := 200
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let final_concentration : ℝ := 0.18
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = final_concentration := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2587_258709


namespace NUMINAMATH_CALUDE_rational_solutions_k_l2587_258721

/-- A function that checks if a given positive integer k results in rational solutions for the equation kx^2 + 20x + k = 0 -/
def has_rational_solutions (k : ℕ+) : Prop :=
  ∃ n : ℕ, (100 - k.val^2 : ℤ) = n^2

/-- The theorem stating that the positive integer values of k for which kx^2 + 20x + k = 0 has rational solutions are exactly 6, 8, and 10 -/
theorem rational_solutions_k :
  ∀ k : ℕ+, has_rational_solutions k ↔ k.val ∈ ({6, 8, 10} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_rational_solutions_k_l2587_258721


namespace NUMINAMATH_CALUDE_fifteenth_term_equals_44_l2587_258780

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- The 15th term of the specific arithmetic progression -/
def fifteenthTerm : ℝ :=
  arithmeticProgressionTerm 2 3 15

theorem fifteenth_term_equals_44 : fifteenthTerm = 44 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_equals_44_l2587_258780


namespace NUMINAMATH_CALUDE_expression_simplification_l2587_258729

theorem expression_simplification (a b : ℤ) : 
  (a = 1) → (b = -a) → 3*a^2*b + 2*(a*b - 3/2*a^2*b) - (2*a*b^2 - (3*a*b^2 - a*b)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2587_258729


namespace NUMINAMATH_CALUDE_max_value_of_product_l2587_258795

/-- Given real numbers x and y that satisfy x + y = 1, 
    the maximum value of (x^3 + 1)(y^3 + 1) is 4. -/
theorem max_value_of_product (x y : ℝ) (h : x + y = 1) :
  ∃ M : ℝ, M = 4 ∧ ∀ x y : ℝ, x + y = 1 → (x^3 + 1) * (y^3 + 1) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_product_l2587_258795


namespace NUMINAMATH_CALUDE_ones_divisible_by_l_l2587_258738

theorem ones_divisible_by_l (l : ℕ) (h1 : ¬ 2 ∣ l) (h2 : ¬ 5 ∣ l) :
  ∃ n : ℕ, l ∣ n ∧ ∀ d : ℕ, d ∈ (n.digits 10) → d = 1 :=
sorry

end NUMINAMATH_CALUDE_ones_divisible_by_l_l2587_258738


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l2587_258743

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l2587_258743


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2587_258799

theorem initial_money_calculation (initial_money : ℚ) : 
  (initial_money * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 400) → 
  initial_money = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2587_258799


namespace NUMINAMATH_CALUDE_cubic_equation_implies_specific_value_l2587_258722

theorem cubic_equation_implies_specific_value :
  ∀ x : ℝ, x^3 - 3 * Real.sqrt 2 * x^2 + 6 * x - 2 * Real.sqrt 2 - 8 = 0 →
  x^5 - 41 * x^2 + 2012 = 1998 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_implies_specific_value_l2587_258722


namespace NUMINAMATH_CALUDE_ab_value_l2587_258706

theorem ab_value (a b : ℤ) (h1 : |a| = 7) (h2 : b = 5) (h3 : a + b < 0) : a * b = -35 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2587_258706


namespace NUMINAMATH_CALUDE_order_of_equations_l2587_258786

def order_of_diff_eq (eq : String) : ℕ :=
  match eq with
  | "y' + 2x = 0" => 1
  | "y'' + 3y' - 4 = 0" => 2
  | "2dy - 3x dx = 0" => 1
  | "y'' = cos x" => 2
  | _ => 0

theorem order_of_equations :
  (order_of_diff_eq "y' + 2x = 0" = 1) ∧
  (order_of_diff_eq "y'' + 3y' - 4 = 0" = 2) ∧
  (order_of_diff_eq "2dy - 3x dx = 0" = 1) ∧
  (order_of_diff_eq "y'' = cos x" = 2) := by
  sorry

end NUMINAMATH_CALUDE_order_of_equations_l2587_258786


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2587_258737

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 4}
def B : Set Nat := {4, 5}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2587_258737


namespace NUMINAMATH_CALUDE_marcy_lip_gloss_tubs_l2587_258747

/-- The number of tubs of lip gloss Marcy needs to bring for a wedding -/
def tubs_of_lip_gloss (people : ℕ) (people_per_tube : ℕ) (tubes_per_tub : ℕ) : ℕ :=
  (people / people_per_tube) / tubes_per_tub

/-- Theorem: Marcy needs to bring 6 tubs of lip gloss for 36 people -/
theorem marcy_lip_gloss_tubs : tubs_of_lip_gloss 36 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_marcy_lip_gloss_tubs_l2587_258747


namespace NUMINAMATH_CALUDE_specific_enclosed_area_l2587_258791

/-- The area enclosed by a curve composed of 9 congruent circular arcs, where the centers of the
    corresponding circles are among the vertices of a regular hexagon. -/
def enclosed_area (arc_length : ℝ) (hexagon_side : ℝ) : ℝ :=
  sorry

/-- The theorem stating that the area enclosed by the specific curve described in the problem
    is equal to (27√3)/2 + (1125π²)/96. -/
theorem specific_enclosed_area :
  enclosed_area (5 * π / 6) 3 = (27 * Real.sqrt 3) / 2 + (1125 * π^2) / 96 := by
  sorry

end NUMINAMATH_CALUDE_specific_enclosed_area_l2587_258791


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2587_258717

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line_equation x y := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2587_258717


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_equation_l2587_258735

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by
  sorry

theorem negation_of_exponential_equation : 
  (¬ ∃ x : ℝ, Real.exp x = x - 1) ↔ (∀ x : ℝ, Real.exp x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_equation_l2587_258735


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l2587_258712

/-- The volume of a sphere with diameter 24 cm minus the volume of three cylindrical holes
    (with depths 6 cm and diameters 1.5 cm, 2.5 cm, and 3 cm respectively) is equal to 2239.5π cubic cm. -/
theorem bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let hole_diameter1 : ℝ := 1.5
  let hole_diameter2 : ℝ := 2.5
  let hole_diameter3 : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let hole_volume1 := π * (hole_diameter1 / 2) ^ 2 * hole_depth
  let hole_volume2 := π * (hole_diameter2 / 2) ^ 2 * hole_depth
  let hole_volume3 := π * (hole_diameter3 / 2) ^ 2 * hole_depth
  let remaining_volume := sphere_volume - (hole_volume1 + hole_volume2 + hole_volume3)
  remaining_volume = 2239.5 * π :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_volume_l2587_258712


namespace NUMINAMATH_CALUDE_pages_left_to_read_l2587_258740

/-- Given a book with 400 pages, prove that if a person has read 20% of it,
    they need to read 320 pages to finish the book. -/
theorem pages_left_to_read (total_pages : ℕ) (percentage_read : ℚ) 
  (h1 : total_pages = 400)
  (h2 : percentage_read = 20 / 100) :
  total_pages - (total_pages * percentage_read).floor = 320 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l2587_258740


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l2587_258723

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 0, 3, -1; -1, 3, 2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 0; 2, 1, -2; 3, 0, 1]
def c : ℤ := 2

theorem matrix_multiplication_result :
  c • (A * B) = !![(-14:ℤ), -4, -6; 6, 6, -14; 22, 8, -8] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l2587_258723


namespace NUMINAMATH_CALUDE_french_exam_vocabulary_l2587_258704

theorem french_exam_vocabulary (total_words : ℕ) (guess_rate : ℚ) (target_score : ℚ) : 
  total_words = 600 → 
  guess_rate = 5 / 100 → 
  target_score = 90 / 100 → 
  ∃ (words_to_learn : ℕ), 
    words_to_learn ≥ 537 ∧ 
    (words_to_learn : ℚ) / total_words + 
      guess_rate * ((total_words - words_to_learn) : ℚ) / total_words ≥ target_score ∧
    ∀ (x : ℕ), x < 537 → 
      (x : ℚ) / total_words + 
        guess_rate * ((total_words - x) : ℚ) / total_words < target_score :=
by sorry

end NUMINAMATH_CALUDE_french_exam_vocabulary_l2587_258704


namespace NUMINAMATH_CALUDE_complex_magnitude_eighth_power_l2587_258798

theorem complex_magnitude_eighth_power : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^8 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_eighth_power_l2587_258798


namespace NUMINAMATH_CALUDE_equation_solution_l2587_258734

theorem equation_solution :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2587_258734


namespace NUMINAMATH_CALUDE_picture_book_shelves_l2587_258741

theorem picture_book_shelves 
  (total_books : ℕ) 
  (books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) 
  (h1 : total_books = 72)
  (h2 : books_per_shelf = 9)
  (h3 : mystery_shelves = 3) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 5 := by
sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l2587_258741


namespace NUMINAMATH_CALUDE_marble_probability_l2587_258769

theorem marble_probability (blue green white : ℝ) 
  (prob_sum : blue + green + white = 1)
  (prob_blue : blue = 0.25)
  (prob_green : green = 0.4) :
  white = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2587_258769


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2587_258758

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 5/6
  let a₂ : ℚ := -4/9
  let a₃ : ℚ := 32/135
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → (a₁ * r^(n-1) : ℚ) = if n = 2 then a₂ else if n = 3 then a₃ else 0) →
  r = -8/15 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2587_258758


namespace NUMINAMATH_CALUDE_ratio_problem_l2587_258703

theorem ratio_problem (x y : ℝ) (h : (2*x - y) / (x + y) = 2/3) : x / y = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2587_258703


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_power_2017_l2587_258792

theorem imaginary_part_of_i_power_2017 : Complex.im (Complex.I ^ 2017) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_power_2017_l2587_258792


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_is_linear_in_two_variables_l2587_258711

-- Define the solution point
def solution_x : ℝ := 2
def solution_y : ℝ := -3

-- Define the linear equation
def linear_equation (x y : ℝ) : Prop := x + y = -1

-- Theorem statement
theorem solution_satisfies_equation : 
  linear_equation solution_x solution_y := by
  sorry

-- Theorem to prove the equation is linear in two variables
theorem is_linear_in_two_variables : 
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ 
  ∀ (x y : ℝ), linear_equation x y ↔ a * x + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_is_linear_in_two_variables_l2587_258711


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2587_258773

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.tan (x^3 + x^2 * Real.sin (2/x))
  else 0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2587_258773


namespace NUMINAMATH_CALUDE_permutation_combination_problem_l2587_258742

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem permutation_combination_problem :
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 + A 9 5) = 5 / 11 ∧
  C 200 192 + C 200 196 + 2 * C 200 197 = 67331650 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_problem_l2587_258742


namespace NUMINAMATH_CALUDE_probability_hit_at_least_once_l2587_258750

-- Define the probability of hitting the target in a single shot
def hit_probability : ℚ := 2/3

-- Define the number of shots
def num_shots : ℕ := 3

-- Theorem statement
theorem probability_hit_at_least_once :
  1 - (1 - hit_probability) ^ num_shots = 26/27 := by
  sorry

end NUMINAMATH_CALUDE_probability_hit_at_least_once_l2587_258750


namespace NUMINAMATH_CALUDE_divisible_by_4_or_6_count_l2587_258720

def count_divisible (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

theorem divisible_by_4_or_6_count :
  (count_divisible 51 4) + (count_divisible 51 6) - (count_divisible 51 12) = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_4_or_6_count_l2587_258720


namespace NUMINAMATH_CALUDE_henry_payment_l2587_258788

/-- The payment Henry receives for painting a bike -/
def paint_payment : ℕ := 5

/-- The additional payment Henry receives for selling a bike compared to painting it -/
def sell_additional_payment : ℕ := 8

/-- The number of bikes Henry sells and paints -/
def num_bikes : ℕ := 8

/-- The total payment Henry receives for selling and painting the given number of bikes -/
def total_payment (paint : ℕ) (sell_additional : ℕ) (bikes : ℕ) : ℕ :=
  bikes * (paint + sell_additional + paint)

theorem henry_payment :
  total_payment paint_payment sell_additional_payment num_bikes = 144 := by
sorry

end NUMINAMATH_CALUDE_henry_payment_l2587_258788


namespace NUMINAMATH_CALUDE_distance_A_l2587_258728

-- Define the points
def A : ℝ × ℝ := (0, 11)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the condition that AA' and BB' intersect at C
def intersect_at_C (A' B' : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    C = (t₁ * A'.1 + (1 - t₁) * A.1, t₁ * A'.2 + (1 - t₁) * A.2) ∧
    C = (t₂ * B'.1 + (1 - t₂) * B.1, t₂ * B'.2 + (1 - t₂) * B.2)

-- Main theorem
theorem distance_A'B'_is_2_26 :
  ∃ A' B' : ℝ × ℝ, 
    line_y_eq_x A' ∧ 
    line_y_eq_x B' ∧ 
    intersect_at_C A' B' ∧ 
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2.26 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_l2587_258728


namespace NUMINAMATH_CALUDE_modulus_of_z_l2587_258756

-- Define the complex number z
def z : ℂ := (3 - Complex.I)^2 * Complex.I

-- State the theorem
theorem modulus_of_z : Complex.abs z = 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2587_258756


namespace NUMINAMATH_CALUDE_horner_v4_value_l2587_258784

def horner_step (v : ℤ) (a : ℤ) (x : ℤ) : ℤ := v * x + a

def horner_method (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc coeff => horner_step acc coeff x) 0

theorem horner_v4_value :
  let coeffs := [3, 5, 6, 20, -8, 35, 12]
  let x := -2
  let v0 := 3
  let v1 := horner_step v0 5 x
  let v2 := horner_step v1 6 x
  let v3 := horner_step v2 20 x
  let v4 := horner_step v3 (-8) x
  v4 = -16 := by sorry

end NUMINAMATH_CALUDE_horner_v4_value_l2587_258784


namespace NUMINAMATH_CALUDE_binomial_inequality_l2587_258715

theorem binomial_inequality (n : ℕ) : 2 ≤ (1 + 1 / n : ℝ) ^ n ∧ (1 + 1 / n : ℝ) ^ n < 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l2587_258715


namespace NUMINAMATH_CALUDE_regression_line_intercept_l2587_258702

theorem regression_line_intercept (x_bar m : ℝ) (y_bar : ℝ) :
  x_bar = m → y_bar = 6 → y_bar = 2 * x_bar + m → m = 2 := by sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l2587_258702


namespace NUMINAMATH_CALUDE_jack_baseball_cards_l2587_258751

theorem jack_baseball_cards :
  ∀ (total_cards baseball_cards football_cards : ℕ),
  total_cards = 125 →
  baseball_cards = 3 * football_cards + 5 →
  total_cards = baseball_cards + football_cards →
  baseball_cards = 95 := by
sorry

end NUMINAMATH_CALUDE_jack_baseball_cards_l2587_258751


namespace NUMINAMATH_CALUDE_x_convergence_l2587_258770

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 8 * x n + 9) / (x n + 7)

theorem x_convergence :
  ∃ m : ℕ, m ≥ 81 ∧ m ≤ 242 ∧ 
    x m ≤ 5 + 1 / 2^15 ∧ 
    ∀ k : ℕ, k > 0 ∧ k < m → x k > 5 + 1 / 2^15 :=
by sorry

end NUMINAMATH_CALUDE_x_convergence_l2587_258770


namespace NUMINAMATH_CALUDE_expression_value_l2587_258793

theorem expression_value : 3 * (15 + 7)^2 - (15^2 + 7^2) = 1178 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2587_258793


namespace NUMINAMATH_CALUDE_min_tiles_for_coverage_l2587_258760

-- Define the grid size
def grid_size : ℕ := 8

-- Define the size of small squares
def small_square_size : ℕ := 2

-- Define the number of cells covered by each L-shaped tile
def cells_per_tile : ℕ := 3

-- Calculate the number of small squares in the grid
def num_small_squares : ℕ := (grid_size * grid_size) / (small_square_size * small_square_size)

-- Define the minimum number of cells that need to be covered
def min_cells_to_cover : ℕ := 2 * num_small_squares

-- Define the minimum number of L-shaped tiles needed
def min_tiles_needed : ℕ := (min_cells_to_cover + cells_per_tile - 1) / cells_per_tile

-- Theorem statement
theorem min_tiles_for_coverage : min_tiles_needed = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_coverage_l2587_258760


namespace NUMINAMATH_CALUDE_fraction_sum_l2587_258730

theorem fraction_sum (p q : ℝ) (h : p ≠ 0 ∧ q ≠ 0) 
  (h1 : 1/p + 1/q = 1/(p+q)) : p/q + q/p = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2587_258730


namespace NUMINAMATH_CALUDE_school_children_count_l2587_258768

/-- Proves that the number of children in the school is 320 given the banana distribution conditions --/
theorem school_children_count : ∃ (C : ℕ) (B : ℕ), 
  B = 2 * C ∧                   -- Total bananas if each child gets 2
  B = 4 * (C - 160) ∧           -- Total bananas distributed to present children
  C = 320                       -- The number we want to prove
  := by sorry

end NUMINAMATH_CALUDE_school_children_count_l2587_258768


namespace NUMINAMATH_CALUDE_cubic_roots_nature_l2587_258716

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 4

-- Theorem statement
theorem cubic_roots_nature :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c < 0 ∧
    (∀ x : ℝ, cubic_poly x = 0 ↔ (x = a ∨ x = b ∨ x = c))) :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_nature_l2587_258716


namespace NUMINAMATH_CALUDE_correct_num_selections_l2587_258731

/-- The number of pairs of gloves -/
def num_pairs : ℕ := 5

/-- The number of gloves to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select 3 gloves of different colors from 5 pairs of gloves -/
def num_selections : ℕ := 80

/-- Theorem stating that the number of selections is correct -/
theorem correct_num_selections :
  (num_pairs.choose num_selected) * (2^num_selected) = num_selections :=
by sorry

end NUMINAMATH_CALUDE_correct_num_selections_l2587_258731


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l2587_258774

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500) 
  (h2 : music_students = 20) 
  (h3 : art_students = 20) 
  (h4 : both_students = 10) : 
  total_students - (music_students + art_students - both_students) = 470 := by
  sorry

#check students_taking_neither_music_nor_art

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l2587_258774


namespace NUMINAMATH_CALUDE_complex_number_problem_l2587_258708

theorem complex_number_problem (a : ℝ) : 
  (∃ (z₁ : ℂ), z₁ = a + (2 / (1 - Complex.I)) ∧ z₁.re < 0 ∧ z₁.im > 0) ∧ 
  Complex.abs (a - Complex.I) = 2 → 
  a = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2587_258708


namespace NUMINAMATH_CALUDE_base_conversion_sum_l2587_258749

/-- Converts a number from given base to base 10 -/
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- The main theorem -/
theorem base_conversion_sum :
  let a := to_base_10 [2, 1, 3] 8
  let b := to_base_10 [1, 2] 3
  let c := to_base_10 [2, 3, 4] 5
  let d := to_base_10 [3, 2] 4
  (a / b : Rat) + (c / d : Rat) = 31 + 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l2587_258749


namespace NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l2587_258755

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + b = 0) :
  ∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ 
    (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ m) ∧
    (∃ x : ℝ, (a^2 + b^2) / (a - b) = m) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l2587_258755


namespace NUMINAMATH_CALUDE_photocopy_cost_calculation_l2587_258767

/-- The cost of a single photocopy --/
def photocopy_cost : ℝ := sorry

/-- The discount rate for orders over 100 photocopies --/
def discount_rate : ℝ := 0.25

/-- The number of copies each person needs --/
def copies_per_person : ℕ := 80

/-- The amount saved per person when ordering together --/
def savings_per_person : ℝ := 0.40

/-- Theorem stating the cost of a single photocopy --/
theorem photocopy_cost_calculation : photocopy_cost = 0.02 := by
  have h1 : 2 * copies_per_person * photocopy_cost - 
    (2 * copies_per_person * photocopy_cost * (1 - discount_rate)) = 
    2 * savings_per_person := by sorry
  
  -- The rest of the proof steps would go here
  sorry

end NUMINAMATH_CALUDE_photocopy_cost_calculation_l2587_258767


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l2587_258797

/-- Parabola with focus F and equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Circle intersecting y-axis -/
structure IntersectingCircle (M : PointOnParabola C) where
  radius : ℝ
  chord_length : ℝ
  h_chord : chord_length = 2 * Real.sqrt 5
  h_radius_eq : radius^2 = M.x^2 + 5

/-- Line intersecting parabola -/
structure IntersectingLine (C : Parabola) where
  slope : ℝ
  x_intercept : ℝ
  h_slope : slope = Real.pi / 4
  h_intercept : x_intercept = 2

/-- Intersection points of line and parabola -/
structure IntersectionPoints (C : Parabola) (l : IntersectingLine C) where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h_on_parabola₁ : y₁^2 = 2 * C.p * x₁
  h_on_parabola₂ : y₂^2 = 2 * C.p * x₂
  h_on_line₁ : y₁ = l.slope * (x₁ - l.x_intercept)
  h_on_line₂ : y₂ = l.slope * (x₂ - l.x_intercept)

theorem parabola_circle_intersection
  (C : Parabola)
  (M : PointOnParabola C)
  (circle : IntersectingCircle M)
  (l : IntersectingLine C)
  (points : IntersectionPoints C l) :
  circle.radius = 3 ∧ x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l2587_258797


namespace NUMINAMATH_CALUDE_tangent_fraction_equals_one_l2587_258765

theorem tangent_fraction_equals_one (θ : Real) (h : Real.tan θ = -2 * Real.sqrt 2) :
  (2 * (Real.cos (θ / 2))^2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + π/4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_fraction_equals_one_l2587_258765


namespace NUMINAMATH_CALUDE_local_extremum_implies_b_minus_a_l2587_258761

/-- A function with a local extremum -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - b

theorem local_extremum_implies_b_minus_a (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → b - a = 15 := by
  sorry

end NUMINAMATH_CALUDE_local_extremum_implies_b_minus_a_l2587_258761


namespace NUMINAMATH_CALUDE_bonus_sector_area_l2587_258746

/-- Given a circular spinner with radius 15 cm and a "Bonus" sector with a 
    probability of 1/3 of being landed on, the area of the "Bonus" sector 
    is 75π square centimeters. -/
theorem bonus_sector_area (radius : ℝ) (probability : ℝ) (bonus_area : ℝ) : 
  radius = 15 →
  probability = 1 / 3 →
  bonus_area = probability * π * radius^2 →
  bonus_area = 75 * π := by
  sorry


end NUMINAMATH_CALUDE_bonus_sector_area_l2587_258746


namespace NUMINAMATH_CALUDE_range_of_a_for_p_and_q_l2587_258752

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + a*x

def is_monotonically_increasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

def represents_hyperbola (a : ℝ) : Prop :=
  (a + 2) * (a - 2) < 0

theorem range_of_a_for_p_and_q :
  {a : ℝ | is_monotonically_increasing (f a) ∧ represents_hyperbola a} = Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_p_and_q_l2587_258752


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_theorem_l2587_258727

/-- An isosceles trapezoid with given midline length and height -/
structure IsoscelesTrapezoid where
  midline : ℝ
  height : ℝ

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := t.midline * t.height

/-- Theorem: The area of an isosceles trapezoid with midline 15 and height 3 is 45 -/
theorem isosceles_trapezoid_area_theorem :
  ∀ t : IsoscelesTrapezoid, t.midline = 15 ∧ t.height = 3 → area t = 45 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_theorem_l2587_258727


namespace NUMINAMATH_CALUDE_equation_has_root_minus_one_l2587_258726

theorem equation_has_root_minus_one : ∃ x : ℝ, x = -1 ∧ x^2 - x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_root_minus_one_l2587_258726


namespace NUMINAMATH_CALUDE_min_sum_grid_l2587_258787

theorem min_sum_grid (a b c d : ℕ+) (h : a * b + c * d + a * c + b * d = 2015) :
  a + b + c + d ≥ 88 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_grid_l2587_258787


namespace NUMINAMATH_CALUDE_special_circle_equation_l2587_258790

/-- A circle with center on the y-axis passing through (3, 1) and tangent to x-axis -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  passes_through_point : (3 - center.1)^2 + (1 - center.2)^2 = radius^2
  tangent_to_x_axis : center.2 = radius

/-- The equation of the special circle is x^2 + y^2 - 10y = 0 -/
theorem special_circle_equation (c : SpecialCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔ x^2 + y^2 - 10*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_special_circle_equation_l2587_258790


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l2587_258754

/-- The number of walnut trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of trees planted. -/
theorem walnut_trees_after_planting 
  (initial_trees : ℕ) 
  (planted_trees : ℕ) : 
  initial_trees + planted_trees = 
    initial_trees + planted_trees :=
by sorry

/-- Specific instance of the theorem with 4 initial trees and 6 planted trees -/
example : 4 + 6 = 10 :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l2587_258754


namespace NUMINAMATH_CALUDE_incorrect_calculation_l2587_258713

theorem incorrect_calculation : 
  ((-11) + (-17) = -28) ∧ 
  ((-3/4 : ℚ) + (1/2 : ℚ) = -1/4) ∧ 
  ((-9) + 9 = 0) ∧ 
  ((5/8 : ℚ) + (-7/12 : ℚ) ≠ -1/24) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l2587_258713


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2587_258794

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2587_258794


namespace NUMINAMATH_CALUDE_parking_capacity_l2587_258733

/-- Represents a parking garage with four levels --/
structure ParkingGarage :=
  (level1 : ℕ)
  (level2 : ℕ)
  (level3 : ℕ)
  (level4 : ℕ)

/-- Calculates the total number of parking spaces in the garage --/
def total_spaces (g : ParkingGarage) : ℕ :=
  g.level1 + g.level2 + g.level3 + g.level4

/-- Theorem: Given the parking garage conditions, 299 more cars can be accommodated --/
theorem parking_capacity 
  (g : ParkingGarage)
  (h1 : g.level1 = 90)
  (h2 : g.level2 = g.level1 + 8)
  (h3 : g.level3 = g.level2 + 12)
  (h4 : g.level4 = g.level3 - 9)
  (h5 : total_spaces g - 100 = 299) : 
  ∃ (n : ℕ), n = 299 ∧ n = total_spaces g - 100 :=
by
  sorry

#check parking_capacity

end NUMINAMATH_CALUDE_parking_capacity_l2587_258733


namespace NUMINAMATH_CALUDE_car_journey_speed_l2587_258766

theorem car_journey_speed (s : ℝ) (h : s > 0) : 
  let first_part := 0.4 * s
  let second_part := 0.6 * s
  let first_speed := 40
  let average_speed := 100
  let first_time := first_part / first_speed
  let total_time := s / average_speed
  ∃ d : ℝ, d > 0 ∧ second_part / d = total_time - first_time ∧ d = 120 := by
sorry

end NUMINAMATH_CALUDE_car_journey_speed_l2587_258766


namespace NUMINAMATH_CALUDE_circle_numbers_solution_l2587_258725

def CircleNumbers (a b c d e f : ℚ) : Prop :=
  a + b + c + d + e + f = 1 ∧
  a = |b - c| ∧
  b = |c - d| ∧
  c = |d - e| ∧
  d = |e - f| ∧
  e = |f - a| ∧
  f = |a - b|

theorem circle_numbers_solution :
  ∀ a b c d e f : ℚ, CircleNumbers a b c d e f →
  ((a = 1/4 ∧ b = 1/4 ∧ c = 0 ∧ d = 1/4 ∧ e = 1/4 ∧ f = 0) ∨
   (a = 1/4 ∧ b = 0 ∧ c = 1/4 ∧ d = 1/4 ∧ e = 0 ∧ f = 1/4) ∨
   (a = 0 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 0 ∧ e = 1/4 ∧ f = 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_numbers_solution_l2587_258725


namespace NUMINAMATH_CALUDE_loaves_needed_l2587_258776

/-- The number of first-year students -/
def first_year_students : ℕ := 247

/-- The difference between the number of sophomores and first-year students -/
def sophomore_difference : ℕ := 131

/-- The number of sophomores -/
def sophomores : ℕ := first_year_students + sophomore_difference

/-- The total number of students (first-year and sophomores) -/
def total_students : ℕ := first_year_students + sophomores

theorem loaves_needed : total_students = 625 := by sorry

end NUMINAMATH_CALUDE_loaves_needed_l2587_258776


namespace NUMINAMATH_CALUDE_quadratic_vertex_l2587_258789

/-- The quadratic function f(x) = 2x^2 - 4x + 5 has its vertex at (1, 3) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 5
  (∀ x, f x ≥ f 1) ∧ f 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l2587_258789


namespace NUMINAMATH_CALUDE_hash_six_two_l2587_258719

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + a / b

-- Theorem statement
theorem hash_six_two : hash 6 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hash_six_two_l2587_258719


namespace NUMINAMATH_CALUDE_x_value_l2587_258707

theorem x_value (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2587_258707


namespace NUMINAMATH_CALUDE_sum_of_W_and_Y_l2587_258772

def problem (W X Y Z : ℕ) : Prop :=
  W ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  X ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  Y ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  Z ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  (W * X : ℚ) / (Y * Z) + (Y : ℚ) / Z = 3

theorem sum_of_W_and_Y (W X Y Z : ℕ) :
  problem W X Y Z → W + Y = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_W_and_Y_l2587_258772


namespace NUMINAMATH_CALUDE_alex_jane_pen_difference_l2587_258764

/-- The number of pens Alex has after n weeks, given she starts with 4 pens and triples her collection each week -/
def alexPens (n : ℕ) : ℕ := 4 * 3^(n - 1)

/-- The number of pens Jane has after a month -/
def janePens : ℕ := 50

/-- The number of weeks in a month -/
def weeksInMonth : ℕ := 4

theorem alex_jane_pen_difference :
  alexPens weeksInMonth - janePens = 58 := by sorry

end NUMINAMATH_CALUDE_alex_jane_pen_difference_l2587_258764


namespace NUMINAMATH_CALUDE_fraction_inequality_counterexample_l2587_258778

theorem fraction_inequality_counterexample : 
  ∃ (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
    a₁ > 0 ∧ a₂ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ c₁ > 0 ∧ c₂ > 0 ∧ d₁ > 0 ∧ d₂ > 0 ∧
    (a₁ : ℚ) / b₁ < a₂ / b₂ ∧
    (c₁ : ℚ) / d₁ < c₂ / d₂ ∧
    (a₁ + c₁ : ℚ) / (b₁ + d₁) ≥ (a₂ + c₂) / (b₂ + d₂) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_counterexample_l2587_258778


namespace NUMINAMATH_CALUDE_factorization_equality_l2587_258796

theorem factorization_equality (m a b : ℝ) : 3*m*a^2 - 6*m*a*b + 3*m*b^2 = 3*m*(a-b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2587_258796


namespace NUMINAMATH_CALUDE_f_inequality_l2587_258748

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition f'(x) > f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv f) x > f x)

-- Theorem statement
theorem f_inequality : f 2 > Real.exp 2 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2587_258748


namespace NUMINAMATH_CALUDE_largest_house_number_l2587_258700

def phone_number : List Nat := [3, 4, 6, 2, 8, 9, 0]

def sum_digits (num : List Nat) : Nat :=
  num.foldl (· + ·) 0

def is_distinct (num : List Nat) : Prop :=
  num.length = num.toFinset.card

def is_valid_house_number (num : List Nat) : Prop :=
  num.length = 4 ∧ is_distinct num ∧ sum_digits num = sum_digits phone_number

theorem largest_house_number :
  ∀ (house_num : List Nat),
    is_valid_house_number house_num →
    house_num.foldl (fun acc d => acc * 10 + d) 0 ≤ 9876 :=
by sorry

end NUMINAMATH_CALUDE_largest_house_number_l2587_258700


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l2587_258745

/-- The distance from the origin to the line x + 2y - 5 = 0 is √5 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + 2*y - 5 = 0}
  abs (5) / Real.sqrt (1^2 + 2^2) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l2587_258745


namespace NUMINAMATH_CALUDE_fair_cake_distribution_l2587_258739

/-- Represents a cake flavor -/
inductive Flavor
  | Chocolate
  | Strawberry
  | Vanilla

/-- Represents a child's flavor preferences -/
structure ChildPreference where
  flavor1 : Flavor
  flavor2 : Flavor
  different : flavor1 ≠ flavor2

/-- Represents the distribution of cakes -/
structure CakeDistribution where
  totalCakes : Nat
  numChildren : Nat
  numFlavors : Nat
  childPreferences : Fin numChildren → ChildPreference
  cakesPerChild : Nat
  cakesPerFlavor : Fin numFlavors → Nat

/-- Theorem stating that a fair distribution is possible -/
theorem fair_cake_distribution 
  (d : CakeDistribution) 
  (h_total : d.totalCakes = 18) 
  (h_children : d.numChildren = 3) 
  (h_flavors : d.numFlavors = 3) 
  (h_preferences : ∀ i, (d.childPreferences i).flavor1 ≠ (d.childPreferences i).flavor2) 
  (h_distribution : ∀ i, d.cakesPerFlavor i = 6) :
  d.cakesPerChild = 6 ∧ 
  (∀ i : Fin d.numChildren, ∃ f1 f2 : Fin d.numFlavors, 
    f1 ≠ f2 ∧ 
    d.cakesPerFlavor f1 + d.cakesPerFlavor f2 = d.cakesPerChild) :=
by sorry

end NUMINAMATH_CALUDE_fair_cake_distribution_l2587_258739


namespace NUMINAMATH_CALUDE_pencil_distribution_l2587_258783

/-- Given a total number of pencils and students, calculate the number of pencils per student -/
def pencils_per_student (total_pencils : ℕ) (total_students : ℕ) : ℕ :=
  total_pencils / total_students

theorem pencil_distribution (total_pencils : ℕ) (total_students : ℕ) 
  (h1 : total_pencils = 195)
  (h2 : total_students = 65) :
  pencils_per_student total_pencils total_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2587_258783


namespace NUMINAMATH_CALUDE_unique_dot_product_solution_l2587_258763

theorem unique_dot_product_solution (a : ℝ) : 
  (∃! x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ 
    (-Real.sin x * Real.sin (3 * x) + Real.sin (2 * x) * Real.sin (4 * x) = a)) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_dot_product_solution_l2587_258763


namespace NUMINAMATH_CALUDE_hyperbola_condition_ellipse_x_major_condition_l2587_258724

-- Define the curve C
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) := ∀ (x y : ℝ), (x, y) ∈ C t → (4 - t) * (t - 1) < 0

-- Define what it means for C to be an ellipse with major axis on x-axis
def is_ellipse_x_major (t : ℝ) := ∀ (x y : ℝ), (x, y) ∈ C t → (4 - t) > (t - 1) ∧ (t - 1) > 0

-- Theorem statements
theorem hyperbola_condition (t : ℝ) :
  is_hyperbola t → t > 4 ∨ t < 1 := by sorry

theorem ellipse_x_major_condition (t : ℝ) :
  is_ellipse_x_major t → 1 < t ∧ t < 5/2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_ellipse_x_major_condition_l2587_258724


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2587_258759

/-- If z = (a + i) / (1 - i) is a pure imaginary number and a is real, then a = 1 -/
theorem pure_imaginary_complex_fraction (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ b : ℝ, z = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2587_258759


namespace NUMINAMATH_CALUDE_fair_attendance_l2587_258736

/-- Represents the number of children attending the fair -/
def num_children : ℕ := sorry

/-- Represents the number of adults attending the fair -/
def num_adults : ℕ := sorry

/-- The admission fee for children in cents -/
def child_fee : ℕ := 150

/-- The admission fee for adults in cents -/
def adult_fee : ℕ := 400

/-- The total number of people attending the fair -/
def total_people : ℕ := 2200

/-- The total amount collected in cents -/
def total_amount : ℕ := 505000

theorem fair_attendance : 
  num_children + num_adults = total_people ∧
  num_children * child_fee + num_adults * adult_fee = total_amount →
  num_children = 1500 :=
sorry

end NUMINAMATH_CALUDE_fair_attendance_l2587_258736


namespace NUMINAMATH_CALUDE_total_wall_area_l2587_258701

/-- Represents the properties of tiles and the wall they cover -/
structure TileWall where
  regularTileArea : ℝ
  regularTileCount : ℝ
  jumboTileCount : ℝ
  jumboTileLengthRatio : ℝ

/-- The theorem stating the total wall area given the tile properties -/
theorem total_wall_area (w : TileWall)
  (h1 : w.regularTileArea * w.regularTileCount = 60)
  (h2 : w.jumboTileCount = w.regularTileCount / 3)
  (h3 : w.jumboTileLengthRatio = 3) :
  w.regularTileArea * w.regularTileCount + 
  (w.jumboTileLengthRatio * w.regularTileArea) * w.jumboTileCount = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_wall_area_l2587_258701
