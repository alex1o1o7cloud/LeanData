import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2867_286710

/-- An isosceles triangle with two sides of length 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = b ∧ (a = 3 ∧ c = 7 ∨ a = 7 ∧ c = 3)) →
  a + b + c = 17 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2867_286710


namespace NUMINAMATH_CALUDE_no_special_subset_exists_l2867_286740

theorem no_special_subset_exists : ¬∃ (M : Set ℝ), 
  (M.Nonempty) ∧ 
  (∀ (r : ℝ) (a : ℝ), r > 0 → a ∈ M → 
    ∃! (b : ℝ), b ∈ M ∧ |a - b| = r) := by
  sorry

end NUMINAMATH_CALUDE_no_special_subset_exists_l2867_286740


namespace NUMINAMATH_CALUDE_henry_shells_l2867_286757

theorem henry_shells (broken_shells : ℕ) (perfect_non_spiral : ℕ) (spiral_difference : ℕ) :
  broken_shells = 52 →
  perfect_non_spiral = 12 →
  spiral_difference = 21 →
  ∃ (total_perfect : ℕ),
    total_perfect = (broken_shells / 2 - spiral_difference) + perfect_non_spiral ∧
    total_perfect = 17 := by
  sorry

end NUMINAMATH_CALUDE_henry_shells_l2867_286757


namespace NUMINAMATH_CALUDE_solve_for_S_l2867_286799

theorem solve_for_S : ∃ S : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 180 ∧ S = 180 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_S_l2867_286799


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_l2867_286711

theorem sum_of_prime_factors (n : ℕ) : 
  n > 0 ∧ n < 1000 ∧ (∃ k : ℤ, 42 * n = 180 * k) →
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_l2867_286711


namespace NUMINAMATH_CALUDE_table_problem_l2867_286751

theorem table_problem :
  (∀ x : ℤ, (2 * x - 7 : ℤ) = -5 ↔ x = 1) ∧
  (∀ x : ℤ, (-3 * x - 1 : ℤ) = 5 ↔ x = -2) ∧
  (∀ x : ℤ, (3 * x + 2 : ℤ) - (-2 * x + 5) = 7 ↔ x = 2) ∧
  (∀ m n : ℤ, (∀ x : ℤ, (m * (x + 1) + n : ℤ) - (m * x + n) = -4) →
              (m * 3 + n : ℤ) = -5 →
              (m * 7 + n : ℤ) = -21) :=
by sorry

end NUMINAMATH_CALUDE_table_problem_l2867_286751


namespace NUMINAMATH_CALUDE_cousins_assignment_count_l2867_286713

/-- The number of ways to assign n indistinguishable objects to k indistinguishable containers -/
def assign_indistinguishable (n k : ℕ) : ℕ :=
  sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to assign -/
def num_cousins : ℕ := 5

/-- The number of ways to assign the cousins to the rooms is 51 -/
theorem cousins_assignment_count : assign_indistinguishable num_cousins num_rooms = 51 := by
  sorry

end NUMINAMATH_CALUDE_cousins_assignment_count_l2867_286713


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2867_286734

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 1038239 → ¬(618 ∣ (m + 1) ∧ 3648 ∣ (m + 1) ∧ 60 ∣ (m + 1))) ∧ 
  (618 ∣ (1038239 + 1) ∧ 3648 ∣ (1038239 + 1) ∧ 60 ∣ (1038239 + 1)) := by
  sorry


end NUMINAMATH_CALUDE_smallest_number_divisible_l2867_286734


namespace NUMINAMATH_CALUDE_expand_product_l2867_286737

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2867_286737


namespace NUMINAMATH_CALUDE_flour_needed_l2867_286798

/-- Given a recipe requiring 12 cups of flour and 10 cups already added,
    prove that the additional cups of flour needed is 2. -/
theorem flour_needed (recipe_flour : ℕ) (added_flour : ℕ) : 
  recipe_flour = 12 → added_flour = 10 → recipe_flour - added_flour = 2 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_l2867_286798


namespace NUMINAMATH_CALUDE_vector_equality_l2867_286720

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, -1)
def c : ℝ × ℝ := (-1, 2)

theorem vector_equality : c = a - b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l2867_286720


namespace NUMINAMATH_CALUDE_orange_difference_l2867_286716

/-- The number of oranges and apples picked by George and Amelia -/
structure FruitPicking where
  george_oranges : ℕ
  george_apples : ℕ
  amelia_oranges : ℕ
  amelia_apples : ℕ

/-- The conditions of the fruit picking problem -/
def fruit_picking_conditions (fp : FruitPicking) : Prop :=
  fp.george_oranges = 45 ∧
  fp.george_apples = fp.amelia_apples + 5 ∧
  fp.amelia_oranges < fp.george_oranges ∧
  fp.amelia_apples = 15 ∧
  fp.george_oranges + fp.george_apples + fp.amelia_oranges + fp.amelia_apples = 107

/-- The theorem stating the difference in orange count -/
theorem orange_difference (fp : FruitPicking) 
  (h : fruit_picking_conditions fp) : 
  fp.george_oranges - fp.amelia_oranges = 18 := by
  sorry

end NUMINAMATH_CALUDE_orange_difference_l2867_286716


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_5020030_l2867_286712

def numbers : List Nat := [1000, 1001, 1002, 1003, 1004]

theorem sum_of_squares_equals_5020030 :
  (numbers.map (λ x => x * x)).sum = 5020030 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_5020030_l2867_286712


namespace NUMINAMATH_CALUDE_bobby_total_candy_and_chocolate_l2867_286706

def candy_initial : ℕ := 33
def candy_additional : ℕ := 4
def chocolate : ℕ := 14

theorem bobby_total_candy_and_chocolate :
  candy_initial + candy_additional + chocolate = 51 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_candy_and_chocolate_l2867_286706


namespace NUMINAMATH_CALUDE_max_transitions_to_wiki_l2867_286722

theorem max_transitions_to_wiki (channel_a channel_b channel_c : ℕ) :
  channel_a = 850 * 6 / 100 ∧
  channel_b = 1500 * 42 / 1000 ∧
  channel_c = 4536 / 72 →
  max channel_b channel_c = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_max_transitions_to_wiki_l2867_286722


namespace NUMINAMATH_CALUDE_power_product_eq_7776_l2867_286755

theorem power_product_eq_7776 : 3^5 * 2^5 = 7776 := by
  sorry

end NUMINAMATH_CALUDE_power_product_eq_7776_l2867_286755


namespace NUMINAMATH_CALUDE_inscribed_sphere_slant_angle_l2867_286719

/-- A sphere inscribed in a cone with ratio k of tangency circle radius to base radius -/
structure InscribedSphere (k : ℝ) where
  /-- The ratio of the radius of the circle of tangency to the radius of the base of the cone -/
  ratio : k > 0 ∧ k < 1

/-- The cosine of the angle between the slant height and the base of the cone -/
def slant_base_angle_cosine (s : InscribedSphere k) : ℝ := 1 - k

/-- Theorem: The cosine of the angle between the slant height and the base of the cone
    for a sphere inscribed in a cone with ratio k is 1 - k -/
theorem inscribed_sphere_slant_angle (k : ℝ) (s : InscribedSphere k) :
  slant_base_angle_cosine s = 1 - k := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_slant_angle_l2867_286719


namespace NUMINAMATH_CALUDE_probability_is_half_l2867_286736

/-- A circular field with six equally spaced radial roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : ℕ)
  (h_num_roads : num_roads = 6)

/-- A geologist traveling on one of the roads -/
structure Geologist :=
  (speed : ℝ)
  (road : ℕ)
  (h_speed : speed = 5)
  (h_road : road ∈ Finset.range 6)

/-- The distance between two geologists after one hour -/
def distance (field : CircularField) (g1 g2 : Geologist) : ℝ :=
  sorry

/-- The probability of two geologists being more than 8 km apart -/
def probability (field : CircularField) : ℝ :=
  sorry

/-- Main theorem: The probability is 0.5 -/
theorem probability_is_half (field : CircularField) :
  probability field = 0.5 :=
sorry

end NUMINAMATH_CALUDE_probability_is_half_l2867_286736


namespace NUMINAMATH_CALUDE_consecutive_sum_39_l2867_286789

theorem consecutive_sum_39 (n m : ℕ) : 
  m = n + 1 → n + m = 39 → m = 20 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_39_l2867_286789


namespace NUMINAMATH_CALUDE_commercial_length_l2867_286768

theorem commercial_length 
  (total_time : ℕ) 
  (long_commercial_count : ℕ) 
  (long_commercial_length : ℕ) 
  (short_commercial_count : ℕ) : 
  total_time = 37 ∧ 
  long_commercial_count = 3 ∧ 
  long_commercial_length = 5 ∧ 
  short_commercial_count = 11 → 
  (total_time - long_commercial_count * long_commercial_length) / short_commercial_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_commercial_length_l2867_286768


namespace NUMINAMATH_CALUDE_team_point_difference_l2867_286791

/-- The difference in points between two teams -/
def pointDifference (beth_score jan_score judy_score angel_score : ℕ) : ℕ :=
  (beth_score + jan_score) - (judy_score + angel_score)

/-- Theorem stating the point difference between the two teams -/
theorem team_point_difference :
  pointDifference 12 10 8 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_team_point_difference_l2867_286791


namespace NUMINAMATH_CALUDE_min_stamps_l2867_286745

def stamp_problem (n_010 n_020 n_050 n_200 : ℕ) (total : ℚ) : Prop :=
  n_010 ≥ 2 ∧
  n_020 ≥ 5 ∧
  n_050 ≥ 3 ∧
  n_200 ≥ 1 ∧
  total = 10 ∧
  0.1 * n_010 + 0.2 * n_020 + 0.5 * n_050 + 2 * n_200 = total

theorem min_stamps :
  ∃ (n_010 n_020 n_050 n_200 : ℕ),
    stamp_problem n_010 n_020 n_050 n_200 10 ∧
    (∀ (m_010 m_020 m_050 m_200 : ℕ),
      stamp_problem m_010 m_020 m_050 m_200 10 →
      n_010 + n_020 + n_050 + n_200 ≤ m_010 + m_020 + m_050 + m_200) ∧
    n_010 + n_020 + n_050 + n_200 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_min_stamps_l2867_286745


namespace NUMINAMATH_CALUDE_girls_fraction_at_joint_event_l2867_286774

/-- Represents a middle school with a given number of students and boy-to-girl ratio --/
structure MiddleSchool where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a middle school --/
def girls_count (school : MiddleSchool) : ℚ :=
  (school.total_students : ℚ) * school.girl_ratio / (school.boy_ratio + school.girl_ratio)

/-- The fraction of girls at a joint event of two middle schools --/
def girls_fraction (school1 school2 : MiddleSchool) : ℚ :=
  (girls_count school1 + girls_count school2) / (school1.total_students + school2.total_students)

theorem girls_fraction_at_joint_event :
  let jasper_creek : MiddleSchool := { total_students := 360, boy_ratio := 7, girl_ratio := 5 }
  let brookstone : MiddleSchool := { total_students := 240, boy_ratio := 3, girl_ratio := 5 }
  girls_fraction jasper_creek brookstone = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_girls_fraction_at_joint_event_l2867_286774


namespace NUMINAMATH_CALUDE_towel_area_decrease_l2867_286709

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let original_area := L * B
  let new_length := L * (1 - 0.2)
  let new_breadth := B * (1 - 0.1)
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area * 100 = 28 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l2867_286709


namespace NUMINAMATH_CALUDE_largest_decimal_l2867_286754

theorem largest_decimal : ∀ (a b c d e : ℚ),
  a = 0.936 ∧ b = 0.9358 ∧ c = 0.9361 ∧ d = 0.935 ∧ e = 0.921 →
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l2867_286754


namespace NUMINAMATH_CALUDE_book_reading_time_l2867_286792

/-- Given a book with a certain number of pages and initial reading pace,
    calculate the number of days needed to finish the book with an increased reading pace. -/
theorem book_reading_time (total_pages : ℕ) (initial_pages_per_day : ℕ) (initial_days : ℕ) 
    (increase : ℕ) (h1 : total_pages = initial_pages_per_day * initial_days)
    (h2 : initial_pages_per_day = 15) (h3 : initial_days = 24) (h4 : increase = 3) : 
    total_pages / (initial_pages_per_day + increase) = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l2867_286792


namespace NUMINAMATH_CALUDE_new_boys_in_classroom_l2867_286730

/-- The number of new boys that joined a classroom --/
def new_boys (initial_size : ℕ) (initial_girls_percent : ℚ) (final_girls_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of new boys that joined the classroom --/
theorem new_boys_in_classroom :
  new_boys 20 (40 / 100) (32 / 100) = 5 := by sorry

end NUMINAMATH_CALUDE_new_boys_in_classroom_l2867_286730


namespace NUMINAMATH_CALUDE_problem_statement_l2867_286790

theorem problem_statement (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : 1 / a^3 = 512 / b^3 ∧ 1 / a^3 = 125 / c^3 ∧ 1 / a^3 = d / (a + b + c)^3) : 
  d = 2744 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2867_286790


namespace NUMINAMATH_CALUDE_inequality_proof_l2867_286780

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1) / (a + b)^2 + (b*c + 1) / (b + c)^2 + (c*a + 1) / (c + a)^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2867_286780


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2867_286704

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 4) / 3 > x / 2 + 1 ∧ x + a < 0 → x < 2) →
  (∀ x : ℝ, x < 2 → x + a < 0) →
  a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2867_286704


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2867_286714

/-- 
Given two cubic polynomials x^3 + ax^2 + 17x + 10 and x^3 + bx^2 + 20x + 12 that have two distinct
common roots, prove that a = -6 and b = -7.
-/
theorem cubic_polynomials_common_roots (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 17*r + 10 = 0 ∧ 
    r^3 + b*r^2 + 20*r + 12 = 0 ∧
    s^3 + a*s^2 + 17*s + 10 = 0 ∧ 
    s^3 + b*s^2 + 20*s + 12 = 0) →
  a = -6 ∧ b = -7 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2867_286714


namespace NUMINAMATH_CALUDE_relationship_A_and_p_l2867_286732

theorem relationship_A_and_p (x y p : ℝ) (A : ℝ) 
  (h1 : A = (x^2 - 3*y^2) / (3*x^2 + y^2))
  (h2 : p*x*y / (x^2 - (2+p)*x*y + 2*p*y^2) - y / (x - 2*y) = 1/2)
  (h3 : x ≠ 0)
  (h4 : y ≠ 0)
  (h5 : x ≠ 2*y)
  (h6 : x ≠ p*y) :
  A = (9*p^2 - 3) / (27*p^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_relationship_A_and_p_l2867_286732


namespace NUMINAMATH_CALUDE_square_triangle_area_ratio_l2867_286759

theorem square_triangle_area_ratio : 
  ∀ (s t : ℝ), s > 0 → t > 0 → 
  s^2 = (t^2 * Real.sqrt 3) / 4 → 
  s / t = (Real.sqrt (Real.sqrt 3)) / 2 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_area_ratio_l2867_286759


namespace NUMINAMATH_CALUDE_euler_line_equation_l2867_286724

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)

-- Define the property that AC = BC (isosceles triangle)
def is_isosceles (C : ℝ × ℝ) : Prop := dist A C = dist B C

-- Define the Euler line
def euler_line (C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | P.1 - 2 * P.2 + 3 = 0}

-- Theorem statement
theorem euler_line_equation (C : ℝ × ℝ) (h : is_isosceles C) :
  euler_line C = {P : ℝ × ℝ | P.1 - 2 * P.2 + 3 = 0} :=
sorry

end NUMINAMATH_CALUDE_euler_line_equation_l2867_286724


namespace NUMINAMATH_CALUDE_point_order_on_line_l2867_286727

/-- Given points on a line, prove their y-coordinates are ordered. -/
theorem point_order_on_line (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 3 * (-3) - b)
  (h₂ : y₂ = 3 * 1 - b)
  (h₃ : y₃ = 3 * (-1) - b) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end NUMINAMATH_CALUDE_point_order_on_line_l2867_286727


namespace NUMINAMATH_CALUDE_path_area_calculation_l2867_286715

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

theorem path_area_calculation (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.8) :
  path_area field_length field_width path_width = 759.36 := by
  sorry

#eval path_area 75 55 2.8

end NUMINAMATH_CALUDE_path_area_calculation_l2867_286715


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2867_286703

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x + 2| - |x - 1| < a) → a > -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2867_286703


namespace NUMINAMATH_CALUDE_sum_of_abc_l2867_286721

theorem sum_of_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 48) (h5 : a * b + b * c + c * a = 26) (h6 : a = 2 * b) :
  a + b + c = 6 + 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_abc_l2867_286721


namespace NUMINAMATH_CALUDE_no_valid_subset_exists_l2867_286788

/-- The set M defined as the intersection of (0,1) and ℚ -/
def M : Set ℚ := Set.Ioo 0 1 ∩ Set.range Rat.cast

/-- Definition of a valid subset A -/
def is_valid_subset (A : Set ℚ) : Prop :=
  A ⊆ M ∧
  ∀ x ∈ M, ∃! (S : Finset ℚ), (S : Set ℚ) ⊆ A ∧ x = S.sum id

/-- Theorem stating that no valid subset A exists -/
theorem no_valid_subset_exists : ¬∃ A : Set ℚ, is_valid_subset A := by
  sorry

end NUMINAMATH_CALUDE_no_valid_subset_exists_l2867_286788


namespace NUMINAMATH_CALUDE_f_derivative_at_two_l2867_286794

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_two (a b : ℝ) :
  (f a b 1 = -2) →
  (∀ x, HasDerivAt (f a b) ((a * x + b) / x^2) x) →
  (HasDerivAt (f a b) 0 1) →
  (∀ x, HasDerivAt (f a b) ((-2 * x + 2) / x^2) x) →
  HasDerivAt (f a b) (-1/2) 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_two_l2867_286794


namespace NUMINAMATH_CALUDE_cubic_stone_weight_l2867_286760

/-- The weight of a cubic stone -/
def stone_weight (edge_length : ℝ) (weight_per_unit : ℝ) : ℝ :=
  edge_length ^ 3 * weight_per_unit

/-- Theorem: The weight of a cubic stone with edge length 8 decimeters,
    where each cubic decimeter weighs 3.5 kilograms, is 1792 kilograms. -/
theorem cubic_stone_weight :
  stone_weight 8 3.5 = 1792 := by
  sorry

end NUMINAMATH_CALUDE_cubic_stone_weight_l2867_286760


namespace NUMINAMATH_CALUDE_equation_three_holds_l2867_286766

theorem equation_three_holds (square : ℚ) (h : square = 3 + 1/20) : 
  ((6.5 - 2/3) / (3 + 1/2) - (1 + 8/15)) * (square + 71.95) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_three_holds_l2867_286766


namespace NUMINAMATH_CALUDE_three_solutions_at_plus_minus_one_two_solutions_at_plus_minus_sqrt_two_four_solutions_between_neg_sqrt_two_and_sqrt_two_no_solutions_outside_sqrt_two_l2867_286753

/-- The number of solutions to the system of equations
    x^2 - y^2 = 0 and (x-a)^2 + y^2 = 1 -/
def num_solutions (a : ℝ) : ℕ :=
  sorry

/-- The system has three solutions when a = ±1 -/
theorem three_solutions_at_plus_minus_one :
  (num_solutions 1 = 3) ∧ (num_solutions (-1) = 3) :=
sorry

/-- The system has two solutions when a = ±√2 -/
theorem two_solutions_at_plus_minus_sqrt_two :
  (num_solutions (Real.sqrt 2) = 2) ∧ (num_solutions (-(Real.sqrt 2)) = 2) :=
sorry

/-- The system has four solutions for all other values of a in (-√2, √2) except ±1 -/
theorem four_solutions_between_neg_sqrt_two_and_sqrt_two (a : ℝ) :
  a ∈ Set.Ioo (-(Real.sqrt 2)) (Real.sqrt 2) ∧ a ≠ 1 ∧ a ≠ -1 →
  num_solutions a = 4 :=
sorry

/-- The system has no solutions for |a| > √2 -/
theorem no_solutions_outside_sqrt_two (a : ℝ) :
  |a| > Real.sqrt 2 → num_solutions a = 0 :=
sorry

end NUMINAMATH_CALUDE_three_solutions_at_plus_minus_one_two_solutions_at_plus_minus_sqrt_two_four_solutions_between_neg_sqrt_two_and_sqrt_two_no_solutions_outside_sqrt_two_l2867_286753


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2867_286717

theorem simplify_and_evaluate_expression (m : ℚ) (h : m = 5) :
  (m + 2 - 5 / (m - 2)) / ((3 * m - m^2) / (m - 2)) = -8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2867_286717


namespace NUMINAMATH_CALUDE_simplest_form_sqrt_l2867_286787

/-- A square root is in its simplest form if the number under the root has no perfect square factors other than 1. -/
def is_simplest_form (x : ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → (n^2 : ℝ) ∣ x → False

/-- Given four square roots, prove that √15 is in its simplest form while the others are not. -/
theorem simplest_form_sqrt : 
  is_simplest_form 15 ∧ 
  ¬is_simplest_form 24 ∧ 
  ¬is_simplest_form (7/3) ∧ 
  ¬is_simplest_form 0.9 :=
sorry

end NUMINAMATH_CALUDE_simplest_form_sqrt_l2867_286787


namespace NUMINAMATH_CALUDE_intersection_eq_interval_l2867_286771

open Set

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1, 2)
def interval_1_2 : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_eq_interval : M ∩ N = interval_1_2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_eq_interval_l2867_286771


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2867_286747

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) :
  side_length = 7 ∧ exterior_angle = 45 →
  (360 / exterior_angle) * side_length = 56 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2867_286747


namespace NUMINAMATH_CALUDE_calculate_expression_l2867_286758

theorem calculate_expression : (-3)^25 + 2^(4^2 + 5^2 - 7^2) + 3^3 = -3^25 + 27 + 1/256 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2867_286758


namespace NUMINAMATH_CALUDE_sequence_properties_l2867_286761

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

def sequence_T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sum_S n = 2 * sequence_a n - sequence_a 1) ∧
  (sequence_a 1 + sequence_a 3 = 2 * (sequence_a 2 + 1)) →
  (∀ n : ℕ, n > 0 → sequence_a n = 2^n) ∧
  (∀ n : ℕ, n > 0 → sequence_T n = 2 - (n + 2) / 2^n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2867_286761


namespace NUMINAMATH_CALUDE_solution_in_interval_l2867_286750

theorem solution_in_interval (x : ℝ) : 
  (Real.log x + x = 2) → (1.5 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2867_286750


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l2867_286701

theorem modulus_of_complex_expression :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l2867_286701


namespace NUMINAMATH_CALUDE_min_value_of_function_l2867_286793

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  4 * x + 9 / x^2 ≥ 3 * (36 : ℝ)^(1/3) ∧ 
  ∃ y > 0, 4 * y + 9 / y^2 = 3 * (36 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2867_286793


namespace NUMINAMATH_CALUDE_double_wardrobe_socks_l2867_286784

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe :=
  (socks : ℕ)
  (shoes : ℕ)
  (pants : ℕ)
  (tshirts : ℕ)

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  w.socks * 2 + w.shoes * 2 + w.pants + w.tshirts

/-- Calculates the number of sock pairs needed to double the wardrobe -/
def sockPairsNeeded (w : Wardrobe) : ℕ :=
  totalItems w

theorem double_wardrobe_socks (w : Wardrobe) 
  (h1 : w.socks = 20)
  (h2 : w.shoes = 5)
  (h3 : w.pants = 10)
  (h4 : w.tshirts = 10) :
  sockPairsNeeded w = 35 := by
  sorry

#eval sockPairsNeeded { socks := 20, shoes := 5, pants := 10, tshirts := 10 }

end NUMINAMATH_CALUDE_double_wardrobe_socks_l2867_286784


namespace NUMINAMATH_CALUDE_blackboard_number_increase_l2867_286731

theorem blackboard_number_increase (n k : ℕ+) :
  let new_k := k + Nat.gcd k n
  (new_k - k = 1) ∨ (Nat.Prime (new_k - k)) :=
sorry

end NUMINAMATH_CALUDE_blackboard_number_increase_l2867_286731


namespace NUMINAMATH_CALUDE_log_sum_equality_l2867_286767

theorem log_sum_equality : 21 * Real.log 2 + Real.log 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2867_286767


namespace NUMINAMATH_CALUDE_min_value_bound_l2867_286746

noncomputable section

variable (a : ℝ) (x₀ : ℝ)

def f (x : ℝ) := Real.exp x - (a * x) / (x + 1)

theorem min_value_bound (h1 : a > 0) (h2 : x₀ > -1) 
  (h3 : ∀ x > -1, f a x ≥ f a x₀) : f a x₀ ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_min_value_bound_l2867_286746


namespace NUMINAMATH_CALUDE_geometric_progression_value_l2867_286763

def is_geometric_progression (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

theorem geometric_progression_value :
  ∃ x : ℝ, is_geometric_progression (30 + x) (70 + x) (150 + x) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_value_l2867_286763


namespace NUMINAMATH_CALUDE_john_garage_sale_games_l2867_286743

/-- The number of games John bought from a friend -/
def games_from_friend : ℕ := 21

/-- The number of games that didn't work -/
def bad_games : ℕ := 23

/-- The number of good games John ended up with -/
def good_games : ℕ := 6

/-- The number of games John bought at the garage sale -/
def games_from_garage_sale : ℕ := (good_games + bad_games) - games_from_friend

theorem john_garage_sale_games :
  games_from_garage_sale = 8 := by sorry

end NUMINAMATH_CALUDE_john_garage_sale_games_l2867_286743


namespace NUMINAMATH_CALUDE_log_equality_l2867_286708

theorem log_equality (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_log_equality_l2867_286708


namespace NUMINAMATH_CALUDE_train_length_calculation_l2867_286775

/-- Prove that a train traveling at 60 km/hour that takes 30 seconds to pass a bridge of 140 meters in length has a length of approximately 360.1 meters. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_pass_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 →
  bridge_pass_time = 30 →
  bridge_length = 140 →
  ∃ (train_length : ℝ), abs (train_length - 360.1) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2867_286775


namespace NUMINAMATH_CALUDE_oliver_video_games_l2867_286777

/-- The number of working video games Oliver bought -/
def working_games : ℕ := 6

/-- The number of bad video games Oliver bought -/
def bad_games : ℕ := 5

/-- The total number of video games Oliver bought -/
def total_games : ℕ := working_games + bad_games

theorem oliver_video_games : 
  total_games = working_games + bad_games := by sorry

end NUMINAMATH_CALUDE_oliver_video_games_l2867_286777


namespace NUMINAMATH_CALUDE_h2so4_equals_khso4_l2867_286781

/-- Represents the balanced chemical equation for the reaction between KOH and H2SO4 to form KHSO4 -/
structure ChemicalReaction where
  koh : ℝ
  h2so4 : ℝ
  khso4 : ℝ

/-- The theorem states that the number of moles of H2SO4 needed is equal to the number of moles of KHSO4 formed,
    given that the number of moles of KOH initially present is equal to the number of moles of KHSO4 formed -/
theorem h2so4_equals_khso4 (reaction : ChemicalReaction) 
    (h : reaction.koh = reaction.khso4) : reaction.h2so4 = reaction.khso4 := by
  sorry

#check h2so4_equals_khso4

end NUMINAMATH_CALUDE_h2so4_equals_khso4_l2867_286781


namespace NUMINAMATH_CALUDE_moss_pollen_radius_scientific_notation_l2867_286725

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem moss_pollen_radius_scientific_notation :
  let r := 0.0000042
  let sn := toScientificNotation r
  sn.significand = 4.2 ∧ sn.exponent = -6 := by sorry

end NUMINAMATH_CALUDE_moss_pollen_radius_scientific_notation_l2867_286725


namespace NUMINAMATH_CALUDE_negation_equivalence_exists_false_conjunction_true_component_negation_implication_l2867_286702

-- Define the propositions
def p : Prop := ∃ x : ℝ, x^2 + x - 1 < 0
def q : Prop := ∃ x : ℝ, x^2 - 3*x + 2 = 0

-- Statement 1
theorem negation_equivalence : (¬p) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

-- Statement 2
theorem exists_false_conjunction_true_component :
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) := by sorry

-- Statement 3
theorem negation_implication :
  ¬(q → ∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x = 2) ≠
  (q → ∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 2) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_exists_false_conjunction_true_component_negation_implication_l2867_286702


namespace NUMINAMATH_CALUDE_inequality_theorem_l2867_286726

theorem inequality_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  c / a > c / b :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2867_286726


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2867_286776

theorem triangle_angle_proof (a b c : ℝ) (A : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  0 < A ∧ A < π →  -- Angle A is between 0 and π
  a^2 = b^2 + Real.sqrt 3 * b * c + c^2 →  -- Given condition
  A = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2867_286776


namespace NUMINAMATH_CALUDE_card_passing_game_theorem_l2867_286749

/-- Represents the state of the card-passing game -/
structure GameState where
  num_students : ℕ
  num_cards : ℕ
  card_distribution : List ℕ

/-- Defines a valid game state -/
def valid_game_state (state : GameState) : Prop :=
  state.num_students = 1994 ∧
  state.card_distribution.length = state.num_students ∧
  state.card_distribution.sum = state.num_cards

/-- Defines the condition for the game to end -/
def game_ends (state : GameState) : Prop :=
  ∀ n, n ∈ state.card_distribution → n ≤ 1

/-- Defines the ability to continue the game -/
def can_continue (state : GameState) : Prop :=
  ∃ n, n ∈ state.card_distribution ∧ n ≥ 2

/-- Main theorem about the card-passing game -/
theorem card_passing_game_theorem (state : GameState) 
  (h_valid : valid_game_state state) :
  (state.num_cards ≥ state.num_students → 
    ∃ (game_sequence : ℕ → GameState), ∀ n, can_continue (game_sequence n)) ∧
  (state.num_cards < state.num_students → 
    ∃ (game_sequence : ℕ → GameState) (end_state : ℕ), game_ends (game_sequence end_state)) :=
sorry

end NUMINAMATH_CALUDE_card_passing_game_theorem_l2867_286749


namespace NUMINAMATH_CALUDE_william_max_moves_l2867_286778

/-- Represents a player in the game -/
inductive Player : Type
| Mark : Player
| William : Player

/-- Represents a move in the game -/
inductive Move : Type
| Double : Move  -- Multiply by 2 and add 1
| Quadruple : Move  -- Multiply by 4 and add 3

/-- Applies a move to the current value -/
def applyMove (value : ℕ) (move : Move) : ℕ :=
  match move with
  | Move.Double => 2 * value + 1
  | Move.Quadruple => 4 * value + 3

/-- Checks if the game is over -/
def isGameOver (value : ℕ) : Prop :=
  value > 2^100

/-- Represents the state of the game -/
structure GameState :=
  (value : ℕ)
  (currentPlayer : Player)

/-- Represents an optimal strategy for the game -/
def OptimalStrategy : Type :=
  GameState → Move

/-- The maximum number of moves William can make -/
def maxWilliamMoves : ℕ := 33

/-- The main theorem to be proved -/
theorem william_max_moves 
  (strategy : OptimalStrategy) : 
  ∃ (game : List Move), 
    game.length = 2 * maxWilliamMoves + 1 ∧ 
    isGameOver (game.foldl applyMove 1) ∧
    ∀ (game' : List Move), 
      game'.length > 2 * maxWilliamMoves + 1 → 
      ¬isGameOver (game'.foldl applyMove 1) :=
sorry

end NUMINAMATH_CALUDE_william_max_moves_l2867_286778


namespace NUMINAMATH_CALUDE_prove_weights_l2867_286795

/-- Represents a weighing device that signals when the total weight is 46 kg -/
def WeighingDevice (weights : List Nat) : Bool :=
  weights.sum = 46

/-- Represents the set of ingots with weights from 1 to 13 kg -/
def Ingots : List Nat := List.range 13 |>.map (· + 1)

/-- Checks if a given list of weights is a subset of the Ingots -/
def IsValidSelection (selection : List Nat) : Bool :=
  selection.all (· ∈ Ingots) ∧ selection.length ≤ Ingots.length

theorem prove_weights :
  ∃ (selection1 selection2 : List Nat),
    IsValidSelection selection1 ∧
    IsValidSelection selection2 ∧
    WeighingDevice selection1 ∧
    WeighingDevice selection2 ∧
    (9 ∈ selection1 ∨ 9 ∈ selection2) ∧
    (10 ∈ selection1 ∨ 10 ∈ selection2) :=
  sorry

end NUMINAMATH_CALUDE_prove_weights_l2867_286795


namespace NUMINAMATH_CALUDE_quadratic_radical_sum_l2867_286756

theorem quadratic_radical_sum (m n : ℕ) : 
  (∃ k : ℕ, (m - 1 : ℕ) = 2 ∧ 7^k = 7) ∧ 
  (∃ l : ℕ, 4*n - 1 = 7^l) ∧
  (m - 1 : ℕ) = 2 → 
  m + n = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_radical_sum_l2867_286756


namespace NUMINAMATH_CALUDE_button_fraction_proof_l2867_286785

theorem button_fraction_proof (mari kendra sue : ℕ) : 
  mari = 5 * kendra + 4 →
  mari = 64 →
  sue = 6 →
  sue / kendra = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_button_fraction_proof_l2867_286785


namespace NUMINAMATH_CALUDE_expression_evaluation_l2867_286786

theorem expression_evaluation (a b : ℝ) 
  (h : (a + 1/2)^2 + |b - 2| = 0) : 
  5*(3*a^2*b - a*b^2) - (a*b^2 + 3*a^2*b) = 18 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2867_286786


namespace NUMINAMATH_CALUDE_right_triangle_area_l2867_286782

theorem right_triangle_area (a b : ℝ) (h1 : a = 3) (h2 : b = 5) : 
  (1/2) * a * b = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2867_286782


namespace NUMINAMATH_CALUDE_two_squares_share_vertices_l2867_286741

/-- A square in a plane. -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ

/-- An isosceles right triangle in a plane. -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a square shares two vertices with a triangle. -/
def SharesTwoVertices (s : Square) (t : IsoscelesRightTriangle) : Prop :=
  ∃ (i j : Fin 4) (v w : ℝ × ℝ), i ≠ j ∧
    s.vertices i = v ∧ s.vertices j = w ∧
    (v = t.A ∨ v = t.B ∨ v = t.C) ∧
    (w = t.A ∨ w = t.B ∨ w = t.C)

/-- The main theorem stating that there are exactly two squares sharing two vertices
    with an isosceles right triangle. -/
theorem two_squares_share_vertices (t : IsoscelesRightTriangle) :
  ∃! (n : ℕ), ∃ (squares : Fin n → Square),
    (∀ i, SharesTwoVertices (squares i) t) ∧
    (∀ s, SharesTwoVertices s t → ∃ i, s = squares i) ∧
    n = 2 :=
sorry

end NUMINAMATH_CALUDE_two_squares_share_vertices_l2867_286741


namespace NUMINAMATH_CALUDE_age_ratio_l2867_286718

/-- Represents the ages of Sam, Sue, and Kendra -/
structure Ages where
  sam : ℕ
  sue : ℕ
  kendra : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.sam = 2 * ages.sue ∧
  ages.kendra = 18 ∧
  ages.sam + ages.sue + ages.kendra + 9 = 36

/-- The theorem to prove -/
theorem age_ratio (ages : Ages) (h : satisfiesConditions ages) : 
  ages.kendra / ages.sam = 3 := by
  sorry

/-- Auxiliary lemma to help with division -/
lemma div_eq_of_mul_eq {a b c : ℕ} (hb : b ≠ 0) (h : a = b * c) : a / b = c := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l2867_286718


namespace NUMINAMATH_CALUDE_composition_of_linear_functions_l2867_286705

theorem composition_of_linear_functions (a b : ℝ) : 
  (∀ x : ℝ, (3 * (a * x + b) - 4) = 4 * x + 2) → 
  a + b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_composition_of_linear_functions_l2867_286705


namespace NUMINAMATH_CALUDE_product_of_sines_equals_one_eighth_l2867_286769

theorem product_of_sines_equals_one_eighth :
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) *
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_one_eighth_l2867_286769


namespace NUMINAMATH_CALUDE_bride_age_at_silver_anniversary_l2867_286739

theorem bride_age_at_silver_anniversary (age_difference : ℕ) (combined_age : ℕ) : 
  age_difference = 19 → combined_age = 185 → ∃ (bride_age groom_age : ℕ), 
    bride_age = groom_age + age_difference ∧ 
    bride_age + groom_age = combined_age ∧ 
    bride_age = 102 := by
  sorry

end NUMINAMATH_CALUDE_bride_age_at_silver_anniversary_l2867_286739


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l2867_286744

theorem smallest_common_multiple_of_9_and_6 :
  ∃ n : ℕ+, (n : ℕ) % 9 = 0 ∧ (n : ℕ) % 6 = 0 ∧
  ∀ m : ℕ+, (m : ℕ) % 9 = 0 → (m : ℕ) % 6 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l2867_286744


namespace NUMINAMATH_CALUDE_partnership_investment_l2867_286773

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) (m : ℝ) : 
  x > 0 ∧ m > 0 →
  18900 * (12 * x) / (12 * x + 2 * x * (12 - m) + 3 * x * 4) = 6300 →
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l2867_286773


namespace NUMINAMATH_CALUDE_custom_op_example_l2867_286748

/-- Custom operation ⊕ for rational numbers -/
def custom_op (a b : ℚ) : ℚ := a * b + (a - b)

/-- Theorem stating that (-5) ⊕ 4 = -29 -/
theorem custom_op_example : custom_op (-5) 4 = -29 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l2867_286748


namespace NUMINAMATH_CALUDE_inequality_proof_l2867_286772

theorem inequality_proof (x : ℝ) (hx : x ≠ 0) :
  max 0 (Real.log (abs x)) ≥ 
    ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
    (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
    (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ∧
  (max 0 (Real.log (abs x)) = 
    ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
    (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
    (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
    x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ 
    x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2867_286772


namespace NUMINAMATH_CALUDE_fort_blocks_count_l2867_286700

/-- Represents the dimensions of a fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to construct a fort --/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) (floorThickness : ℕ) : ℕ :=
  let outerVolume := d.length * d.width * d.height
  let innerLength := d.length - 2 * wallThickness
  let innerWidth := d.width - 2 * wallThickness
  let innerHeight := d.height - floorThickness
  let innerVolume := innerLength * innerWidth * innerHeight
  let topLayerVolume := d.length * d.width
  outerVolume - innerVolume + topLayerVolume

/-- Theorem stating that the number of blocks needed for the given fort is 912 --/
theorem fort_blocks_count :
  let fortDims : FortDimensions := ⟨15, 12, 7⟩
  blocksNeeded fortDims 2 1 = 912 := by
  sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l2867_286700


namespace NUMINAMATH_CALUDE_initial_people_is_ten_l2867_286762

/-- Represents the job completion scenario with given conditions -/
structure JobCompletion where
  initialDays : ℕ
  initialWorkDone : ℚ
  daysBeforeFiring : ℕ
  peopleFired : ℕ
  remainingDays : ℕ

/-- Calculates the initial number of people hired -/
def initialPeopleHired (job : JobCompletion) : ℕ :=
  sorry

/-- Theorem stating that the initial number of people hired is 10 -/
theorem initial_people_is_ten (job : JobCompletion) 
  (h1 : job.initialDays = 100)
  (h2 : job.initialWorkDone = 1/4)
  (h3 : job.daysBeforeFiring = 20)
  (h4 : job.peopleFired = 2)
  (h5 : job.remainingDays = 75) :
  initialPeopleHired job = 10 :=
sorry

end NUMINAMATH_CALUDE_initial_people_is_ten_l2867_286762


namespace NUMINAMATH_CALUDE_reciprocal_of_point_two_l2867_286752

theorem reciprocal_of_point_two (x : ℝ) : x = 0.2 → 1 / x = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_point_two_l2867_286752


namespace NUMINAMATH_CALUDE_quadratic_factor_value_l2867_286707

-- Define the polynomials
def f (x : ℝ) : ℝ := x^4 + 8*x^3 + 18*x^2 + 8*x + 35
def g (x : ℝ) : ℝ := 2*x^4 - 4*x^3 + x^2 + 26*x + 10

-- Define the quadratic polynomial q
def q (d e : ℤ) (x : ℝ) : ℝ := x^2 + d*x + e

-- State the theorem
theorem quadratic_factor_value (d e : ℤ) :
  (∃ (p₁ p₂ : ℝ → ℝ), f = q d e * p₁ ∧ g = q d e * p₂) →
  q d e 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factor_value_l2867_286707


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2867_286729

/-- Definition of a hyperbola with foci and points -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The conditions of the problem -/
def hyperbola_conditions (Γ : Hyperbola) : Prop :=
  Γ.a > 0 ∧ Γ.b > 0 ∧
  (Γ.C.2 = 0) ∧  -- C is on x-axis
  (Γ.C.1 - Γ.B.1, Γ.C.2 - Γ.B.2) = 3 • (Γ.F₂.1 - Γ.A.1, Γ.F₂.2 - Γ.A.2) ∧  -- CB = 3F₂A
  (∃ t : ℝ, t > 0 ∧ Γ.B.1 - Γ.F₂.1 = t * (Γ.F₁.1 - Γ.C.1) ∧ Γ.B.2 - Γ.F₂.2 = t * (Γ.F₁.2 - Γ.C.2))  -- BF₂ bisects ∠F₁BC

/-- The theorem to be proved -/
theorem hyperbola_eccentricity (Γ : Hyperbola) :
  hyperbola_conditions Γ → (Real.sqrt ((Γ.F₁.1 - Γ.F₂.1)^2 + (Γ.F₁.2 - Γ.F₂.2)^2) / (2 * Γ.a) = Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2867_286729


namespace NUMINAMATH_CALUDE_total_available_seats_l2867_286735

/-- Represents a bus with its seating configuration and broken seats -/
structure Bus where
  columns : ℕ
  rows_left : ℕ
  rows_right : ℕ
  broken_seats : ℕ

/-- Calculates the number of available seats in a bus -/
def available_seats (bus : Bus) : ℕ :=
  bus.columns * (bus.rows_left + bus.rows_right) - bus.broken_seats

/-- The list of buses with their configurations -/
def buses : List Bus := [
  ⟨4, 10, 0, 2⟩,   -- Bus 1
  ⟨5, 8, 0, 4⟩,    -- Bus 2
  ⟨3, 12, 0, 3⟩,   -- Bus 3
  ⟨4, 6, 8, 1⟩,    -- Bus 4
  ⟨6, 8, 10, 5⟩,   -- Bus 5
  ⟨5, 8, 2, 4⟩     -- Bus 6 (2 rows with 2 seats each unavailable)
]

/-- Theorem stating that the total number of available seats is 311 -/
theorem total_available_seats :
  (buses.map available_seats).sum = 311 := by
  sorry


end NUMINAMATH_CALUDE_total_available_seats_l2867_286735


namespace NUMINAMATH_CALUDE_line_equation_l2867_286779

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

/-- The main theorem -/
theorem line_equation (l : Line) (m a b : Point) : 
  pointOnLine m l → 
  m.x = -1 ∧ m.y = 2 →
  a.y = 0 →
  b.x = 0 →
  isMidpoint m a b →
  l.a = 2 ∧ l.b = -1 ∧ l.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2867_286779


namespace NUMINAMATH_CALUDE_infinitely_many_triangular_pentagonal_pairs_l2867_286723

/-- A pair of positive integers (n, m) is a triangular-pentagonal pair if n(n+1) = m(3m-1) -/
def IsTriangularPentagonalPair (n m : ℕ) : Prop :=
  n > 0 ∧ m > 0 ∧ n * (n + 1) = m * (3 * m - 1)

/-- There exist infinitely many triangular-pentagonal pairs -/
theorem infinitely_many_triangular_pentagonal_pairs :
  ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ m > k ∧ IsTriangularPentagonalPair n m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_triangular_pentagonal_pairs_l2867_286723


namespace NUMINAMATH_CALUDE_cycle_price_proof_l2867_286797

/-- Proves that given a cycle sold at a 20% loss for Rs. 1280, the original price was Rs. 1600 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1280)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1600 :=
by sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l2867_286797


namespace NUMINAMATH_CALUDE_shekar_biology_score_l2867_286770

/-- Represents a student's scores in five subjects -/
structure StudentScores where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average score for a student -/
def averageScore (scores : StudentScores) : ℚ :=
  (scores.mathematics + scores.science + scores.socialStudies + scores.english + scores.biology) / 5

/-- Theorem: Given Shekar's scores in four subjects and the average, his Biology score must be 75 -/
theorem shekar_biology_score :
  ∀ (scores : StudentScores),
    scores.mathematics = 76 →
    scores.science = 65 →
    scores.socialStudies = 82 →
    scores.english = 67 →
    averageScore scores = 73 →
    scores.biology = 75 := by
  sorry

end NUMINAMATH_CALUDE_shekar_biology_score_l2867_286770


namespace NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l2867_286733

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_to_same_line_are_parallel
  (a b c : Line) :
  parallel a c → parallel b c → parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l2867_286733


namespace NUMINAMATH_CALUDE_card_area_theorem_l2867_286728

/-- Represents a rectangular card with length and width in inches -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card in square inches -/
def area (c : Card) : ℝ := c.length * c.width

/-- The original card -/
def original_card : Card := { length := 5, width := 7 }

/-- Theorem: If shortening one side of the original 5x7 card by 2 inches
    results in an area of 21 square inches, then shortening the other side
    by 1 inch results in an area of 30 square inches -/
theorem card_area_theorem :
  (∃ (c : Card), (c.length = original_card.length - 2 ∨ c.width = original_card.width - 2) ∧
                 area c = 21) →
  area { length := original_card.length,
         width := original_card.width - 1 } = 30 := by
  sorry

end NUMINAMATH_CALUDE_card_area_theorem_l2867_286728


namespace NUMINAMATH_CALUDE_parking_lot_problem_l2867_286783

/-- Represents the number of wheels for each vehicle type -/
structure VehicleWheels where
  car : Nat
  bicycle : Nat
  motorcycle : Nat

/-- Represents the count of each vehicle type in the parking lot -/
structure VehicleCount where
  cars : Nat
  bicycles : Nat
  motorcycles : Nat

/-- The theorem stating the relationship between the number of cars and motorcycles -/
theorem parking_lot_problem (wheels : VehicleWheels) (count : VehicleCount) :
  wheels.car = 4 →
  wheels.bicycle = 2 →
  wheels.motorcycle = 2 →
  count.bicycles = 2 * count.motorcycles →
  wheels.car * count.cars + wheels.bicycle * count.bicycles + wheels.motorcycle * count.motorcycles = 196 →
  count.cars = (98 - 3 * count.motorcycles) / 2 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l2867_286783


namespace NUMINAMATH_CALUDE_beef_weight_problem_l2867_286765

theorem beef_weight_problem (initial_weight : ℝ) : 
  initial_weight > 0 →
  initial_weight * (1 - 0.3) * (1 - 0.2) * (1 - 0.5) = 315 →
  initial_weight = 1125 := by
sorry

end NUMINAMATH_CALUDE_beef_weight_problem_l2867_286765


namespace NUMINAMATH_CALUDE_cookies_left_after_sales_l2867_286742

/-- Calculates the number of cookies left after sales throughout the day -/
theorem cookies_left_after_sales (initial : ℕ) (morning_dozens : ℕ) (lunch : ℕ) (afternoon : ℕ) :
  initial = 120 →
  morning_dozens = 3 →
  lunch = 57 →
  afternoon = 16 →
  initial - (morning_dozens * 12 + lunch + afternoon) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_left_after_sales_l2867_286742


namespace NUMINAMATH_CALUDE_mary_apple_expense_l2867_286738

theorem mary_apple_expense (total_spent berries_cost peaches_cost : ℚ)
  (h1 : total_spent = 34.72)
  (h2 : berries_cost = 11.08)
  (h3 : peaches_cost = 9.31) :
  total_spent - (berries_cost + peaches_cost) = 14.33 := by
sorry

end NUMINAMATH_CALUDE_mary_apple_expense_l2867_286738


namespace NUMINAMATH_CALUDE_fib_div_by_five_l2867_286796

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_div_by_five (n : ℕ) : 5 ∣ n → 5 ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_fib_div_by_five_l2867_286796


namespace NUMINAMATH_CALUDE_slower_time_is_692_l2867_286764

/-- The number of stories in the building -/
def num_stories : ℕ := 50

/-- The time Lola takes to run up one story (in seconds) -/
def lola_time_per_story : ℕ := 12

/-- The time the elevator takes to go up one story (in seconds) -/
def elevator_time_per_story : ℕ := 10

/-- The time the elevator stops on each floor (in seconds) -/
def elevator_stop_time : ℕ := 4

/-- The number of floors where the elevator stops -/
def num_elevator_stops : ℕ := num_stories - 2

/-- The time Lola takes to reach the top floor -/
def lola_total_time : ℕ := num_stories * lola_time_per_story

/-- The time Tara takes to reach the top floor -/
def tara_total_time : ℕ := num_stories * elevator_time_per_story + num_elevator_stops * elevator_stop_time

theorem slower_time_is_692 : max lola_total_time tara_total_time = 692 := by sorry

end NUMINAMATH_CALUDE_slower_time_is_692_l2867_286764
