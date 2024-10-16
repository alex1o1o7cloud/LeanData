import Mathlib

namespace NUMINAMATH_CALUDE_exists_same_type_quadratic_surd_with_three_l3306_330657

/-- Two square roots are of the same type of quadratic surd if one can be expressed as a rational multiple of the other. -/
def same_type_quadratic_surd (x y : ℝ) : Prop :=
  ∃ (q : ℚ), x = q * y ∨ y = q * x

theorem exists_same_type_quadratic_surd_with_three :
  ∃ (a : ℕ), a > 0 ∧ same_type_quadratic_surd (Real.sqrt a) (Real.sqrt 3) ∧ a = 12 := by
  sorry

end NUMINAMATH_CALUDE_exists_same_type_quadratic_surd_with_three_l3306_330657


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3306_330673

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 15 * x - 15
def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 12

-- Define the intersection points
def point1 : ℝ × ℝ := (-3, 57)
def point2 : ℝ × ℝ := (12, 237)

theorem parabolas_intersection :
  (∀ x y : ℝ, parabola1 x = parabola2 x ∧ parabola1 x = y → (x, y) = point1 ∨ (x, y) = point2) ∧
  parabola1 (point1.1) = point1.2 ∧
  parabola2 (point1.1) = point1.2 ∧
  parabola1 (point2.1) = point2.2 ∧
  parabola2 (point2.1) = point2.2 ∧
  point1.1 < point2.1 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3306_330673


namespace NUMINAMATH_CALUDE_exponential_fixed_point_l3306_330631

/-- The function f(x) = a^(x-1) + 2 passes through the point (1, 3) for all a > 0 and a ≠ 1 -/
theorem exponential_fixed_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_fixed_point_l3306_330631


namespace NUMINAMATH_CALUDE_cone_volume_l3306_330661

/-- Given a cone with base radius √3 cm and lateral area 6π cm², its volume is 3π cm³ -/
theorem cone_volume (r h : ℝ) : 
  r = Real.sqrt 3 → 
  2 * π * r * (Real.sqrt (h^2 + r^2)) = 6 * π → 
  (1/3) * π * r^2 * h = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3306_330661


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_is_96_l3306_330643

/-- The surface area of a cube with small cubes removed from its corners -/
def modified_cube_surface_area (cube_side : ℝ) (removed_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_side^2
  let removed_area := 8 * 3 * removed_side^2
  let new_exposed_area := 8 * 3 * removed_side^2
  original_surface_area - removed_area + new_exposed_area

/-- Theorem: The surface area of a 4 cm cube with 1 cm cubes removed from each corner is 96 sq.cm -/
theorem modified_cube_surface_area_is_96 :
  modified_cube_surface_area 4 1 = 96 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_is_96_l3306_330643


namespace NUMINAMATH_CALUDE_cookie_batches_for_workshop_l3306_330675

/-- Calculates the minimum number of full batches of cookies needed for a math competition workshop --/
def min_cookie_batches (base_students : ℕ) (additional_students : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  let total_students := base_students + additional_students
  let total_cookies_needed := total_students * cookies_per_student
  (total_cookies_needed + cookies_per_batch - 1) / cookies_per_batch

/-- Proves that 16 batches are needed for the given conditions --/
theorem cookie_batches_for_workshop : 
  min_cookie_batches 90 15 3 20 = 16 := by
sorry

end NUMINAMATH_CALUDE_cookie_batches_for_workshop_l3306_330675


namespace NUMINAMATH_CALUDE_rectangle_existence_l3306_330676

theorem rectangle_existence (s d : ℝ) (hs : s > 0) (hd : d > 0) :
  ∃ (a b : ℝ), 2 * (a + b) = s ∧ a^2 + b^2 = d^2 ∧ a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_l3306_330676


namespace NUMINAMATH_CALUDE_point_inside_circle_l3306_330611

theorem point_inside_circle (a b : ℝ) : 
  a ≠ b → 
  a^2 - a - Real.sqrt 2 = 0 → 
  b^2 - b - Real.sqrt 2 = 0 → 
  a^2 + b^2 < 8 := by
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3306_330611


namespace NUMINAMATH_CALUDE_prob_at_least_one_on_l3306_330660

/-- The probability that at least one of three independent electronic components is on,
    given that each component has a probability of 1/2 of being on. -/
theorem prob_at_least_one_on (n : Nat) (p : ℝ) (h1 : n = 3) (h2 : p = 1 / 2) :
  1 - (1 - p) ^ n = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_on_l3306_330660


namespace NUMINAMATH_CALUDE_domino_set_0_to_12_l3306_330619

/-- The number of tiles in a domino set with values from 0 to n -/
def dominoCount (n : ℕ) : ℕ := Nat.choose (n + 1) 2

/-- The number of tiles in a standard domino set (0 to 6) -/
def standardDominoCount : ℕ := 28

theorem domino_set_0_to_12 : dominoCount 12 = 91 := by sorry

end NUMINAMATH_CALUDE_domino_set_0_to_12_l3306_330619


namespace NUMINAMATH_CALUDE_expand_expression_l3306_330667

theorem expand_expression (x y z : ℝ) : 
  (x + 5) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 15 * y + 10 * z + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3306_330667


namespace NUMINAMATH_CALUDE_ceiling_sqrt_count_l3306_330682

theorem ceiling_sqrt_count : 
  (Finset.range 325 \ Finset.range 290).card = 35 := by sorry

#check ceiling_sqrt_count

end NUMINAMATH_CALUDE_ceiling_sqrt_count_l3306_330682


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3306_330636

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3306_330636


namespace NUMINAMATH_CALUDE_largest_product_bound_l3306_330606

theorem largest_product_bound (a : Fin 1985 → Fin 1985) (h : Function.Bijective a) :
  (Finset.range 1985).sup (λ k => (k + 1) * a (k + 1)) ≥ 993^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_product_bound_l3306_330606


namespace NUMINAMATH_CALUDE_certain_amount_problem_l3306_330691

theorem certain_amount_problem : ∃ x : ℤ, 7 * 5 - 15 = 2 * 5 + x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_problem_l3306_330691


namespace NUMINAMATH_CALUDE_some_number_value_l3306_330622

theorem some_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3306_330622


namespace NUMINAMATH_CALUDE_greatest_integer_pi_plus_three_l3306_330616

theorem greatest_integer_pi_plus_three :
  ∀ π : ℝ, 3 < π ∧ π < 4 → ⌊π + 3⌋ = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_pi_plus_three_l3306_330616


namespace NUMINAMATH_CALUDE_maryann_working_time_l3306_330600

/-- Maryann's working time calculation -/
theorem maryann_working_time 
  (time_calling : ℕ) 
  (accounting_ratio : ℕ) 
  (h1 : time_calling = 70) 
  (h2 : accounting_ratio = 7) : 
  time_calling + accounting_ratio * time_calling = 560 := by
  sorry

end NUMINAMATH_CALUDE_maryann_working_time_l3306_330600


namespace NUMINAMATH_CALUDE_eel_count_l3306_330694

theorem eel_count (electric moray freshwater : ℕ) 
  (h1 : moray + freshwater = 12)
  (h2 : electric + freshwater = 14)
  (h3 : electric + moray = 16) :
  electric + moray + freshwater = 21 := by
sorry

end NUMINAMATH_CALUDE_eel_count_l3306_330694


namespace NUMINAMATH_CALUDE_probability_concentric_circles_l3306_330683

/-- The probability of a randomly chosen point from a circle with radius 3 
    lying within a concentric circle with radius 1 is 1/9. -/
theorem probability_concentric_circles : 
  let outer_radius : ℝ := 3
  let inner_radius : ℝ := 1
  let outer_area := π * outer_radius^2
  let inner_area := π * inner_radius^2
  (inner_area / outer_area : ℝ) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_probability_concentric_circles_l3306_330683


namespace NUMINAMATH_CALUDE_bleacher_runs_theorem_l3306_330608

/-- The number of times a player runs up and down the bleachers -/
def number_of_trips (stairs_one_way : ℕ) (calories_per_stair : ℕ) (total_calories_burned : ℕ) : ℕ :=
  total_calories_burned / (4 * stairs_one_way * calories_per_stair)

/-- Theorem stating the number of times players run up and down the bleachers -/
theorem bleacher_runs_theorem (stairs_one_way : ℕ) (calories_per_stair : ℕ) (total_calories_burned : ℕ)
    (h1 : stairs_one_way = 32)
    (h2 : calories_per_stair = 2)
    (h3 : total_calories_burned = 5120) :
    number_of_trips stairs_one_way calories_per_stair total_calories_burned = 40 := by
  sorry

end NUMINAMATH_CALUDE_bleacher_runs_theorem_l3306_330608


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_10_l3306_330634

theorem x_plus_2y_equals_10 (x y : ℝ) (h1 : x + y = 19) (h2 : x + 3*y = 1) : 
  x + 2*y = 10 := by
sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_10_l3306_330634


namespace NUMINAMATH_CALUDE_departure_representation_l3306_330692

/-- Represents the change in grain quantity -/
inductive GrainChange
| Arrival (amount : ℕ)
| Departure (amount : ℕ)

/-- Records the change in grain quantity -/
def record (change : GrainChange) : ℤ :=
  match change with
  | GrainChange.Arrival amount => amount
  | GrainChange.Departure amount => -amount

/-- Theorem: If arrival of 30 tons is recorded as +30, then -30 represents departure of 30 tons -/
theorem departure_representation :
  (record (GrainChange.Arrival 30) = 30) →
  (record (GrainChange.Departure 30) = -30) :=
by
  sorry

end NUMINAMATH_CALUDE_departure_representation_l3306_330692


namespace NUMINAMATH_CALUDE_valid_solutions_l3306_330656

def is_valid_solution (xyz : ℕ) : Prop :=
  xyz ≥ 100 ∧ xyz ≤ 999 ∧ (456000 + xyz) % 504 = 0

theorem valid_solutions :
  ∀ xyz : ℕ, is_valid_solution xyz ↔ (xyz = 120 ∨ xyz = 624) :=
sorry

end NUMINAMATH_CALUDE_valid_solutions_l3306_330656


namespace NUMINAMATH_CALUDE_p_or_q_iff_m_in_range_l3306_330677

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 3/2 > 0

def q (m : ℝ) : Prop := 
  (m - 1 > 0) ∧ (3 - m > 0) ∧ 
  ∃ c : ℝ, c^2 = (m - 1)*(3 - m) ∧ 
  ∀ x y : ℝ, x^2/(m-1) + y^2/(3-m) = 1 → x^2 + y^2 = (m-1)^2/(m-1) ∨ x^2 + y^2 = (3-m)^2/(3-m)

theorem p_or_q_iff_m_in_range (m : ℝ) : 
  p m ∨ q m ↔ m > -Real.sqrt 6 ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_p_or_q_iff_m_in_range_l3306_330677


namespace NUMINAMATH_CALUDE_smallest_all_ones_divisible_by_d_is_correct_l3306_330684

def d : ℕ := 3 * (10^100 - 1) / 9

def is_all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

def smallest_all_ones_divisible_by_d : ℕ := (10^300 - 1) / 9

theorem smallest_all_ones_divisible_by_d_is_correct :
  is_all_ones smallest_all_ones_divisible_by_d ∧
  smallest_all_ones_divisible_by_d % d = 0 ∧
  ∀ n : ℕ, is_all_ones n → n % d = 0 → n ≥ smallest_all_ones_divisible_by_d :=
by sorry

end NUMINAMATH_CALUDE_smallest_all_ones_divisible_by_d_is_correct_l3306_330684


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l3306_330651

/-- Calculates the sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the sum of an arithmetic sequence --/
def sum_arithmetic_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

/-- The number of days Murtha collects pebbles --/
def total_days : ℕ := 15

/-- The number of days Murtha skips collecting pebbles --/
def skipped_days : ℕ := total_days / 3

/-- Theorem: Murtha's pebble collection after 15 days --/
theorem murtha_pebble_collection :
  sum_first_n total_days - sum_arithmetic_seq 3 3 skipped_days = 75 := by
  sorry

#eval sum_first_n total_days - sum_arithmetic_seq 3 3 skipped_days

end NUMINAMATH_CALUDE_murtha_pebble_collection_l3306_330651


namespace NUMINAMATH_CALUDE_inequality_solution_l3306_330629

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 2 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3306_330629


namespace NUMINAMATH_CALUDE_largest_band_formation_l3306_330610

/-- Represents a band formation --/
structure BandFormation where
  totalMembers : ℕ
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (bf : BandFormation) : Prop :=
  bf.totalMembers < 120 ∧
  bf.totalMembers = bf.rows * bf.membersPerRow + 3 ∧
  bf.totalMembers = (bf.rows - 1) * (bf.membersPerRow + 2)

/-- Theorem stating that 231 is the largest number of band members satisfying the conditions --/
theorem largest_band_formation :
  ∀ bf : BandFormation, isValidFormation bf → bf.totalMembers ≤ 231 :=
by
  sorry

#check largest_band_formation

end NUMINAMATH_CALUDE_largest_band_formation_l3306_330610


namespace NUMINAMATH_CALUDE_probability_grape_star_l3306_330663

/-- A tablet shape -/
inductive Shape
| Square
| Triangle
| Star

/-- A tablet flavor -/
inductive Flavor
| Strawberry
| Grape
| Orange

/-- The number of tablets of each shape -/
def tablets_per_shape : ℕ := 60

/-- The number of flavors -/
def num_flavors : ℕ := 3

/-- The total number of tablets -/
def total_tablets : ℕ := tablets_per_shape * 3

/-- The number of grape star tablets -/
def grape_star_tablets : ℕ := tablets_per_shape / num_flavors

theorem probability_grape_star :
  (grape_star_tablets : ℚ) / total_tablets = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_grape_star_l3306_330663


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l3306_330647

open Set

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 4}

-- Define set N
def N : Finset Nat := {2, 3}

-- Theorem to prove
theorem complement_intersection_problem :
  (U \ M) ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l3306_330647


namespace NUMINAMATH_CALUDE_exam_attendance_calculation_l3306_330652

theorem exam_attendance_calculation (total_topics : ℕ) 
  (all_topics_pass_percent : ℚ) (no_topic_pass_percent : ℚ)
  (one_topic_pass_percent : ℚ) (two_topics_pass_percent : ℚ)
  (four_topics_pass_percent : ℚ) (three_topics_pass_count : ℕ)
  (h1 : total_topics = 5)
  (h2 : all_topics_pass_percent = 1/10)
  (h3 : no_topic_pass_percent = 1/10)
  (h4 : one_topic_pass_percent = 1/5)
  (h5 : two_topics_pass_percent = 1/4)
  (h6 : four_topics_pass_percent = 6/25)
  (h7 : three_topics_pass_count = 500) :
  ∃ total_students : ℕ, total_students = 4546 ∧
  (all_topics_pass_percent + no_topic_pass_percent + one_topic_pass_percent + 
   two_topics_pass_percent + four_topics_pass_percent) * total_students + 
   three_topics_pass_count = total_students :=
by sorry

end NUMINAMATH_CALUDE_exam_attendance_calculation_l3306_330652


namespace NUMINAMATH_CALUDE_mango_purchase_l3306_330689

/-- The amount of grapes purchased in kg -/
def grapes : ℕ := 8

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 60

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 1100

/-- The amount of mangoes purchased in kg -/
def mangoes : ℕ := (total_paid - grapes * grape_price) / mango_price

theorem mango_purchase : mangoes = 9 := by
  sorry

end NUMINAMATH_CALUDE_mango_purchase_l3306_330689


namespace NUMINAMATH_CALUDE_unique_irreverent_polynomial_exists_l3306_330632

/-- A quadratic polynomial of the form x^2 - px + q -/
structure QuadraticPolynomial where
  p : ℝ
  q : ℝ

/-- The number of distinct real solutions to q(q(x)) = 0 -/
noncomputable def numSolutions (poly : QuadraticPolynomial) : ℕ := sorry

/-- The product of the roots of q(q(x)) = 0 -/
noncomputable def rootProduct (poly : QuadraticPolynomial) : ℝ := sorry

/-- Evaluates q(1) for a given quadratic polynomial -/
def evalAtOne (poly : QuadraticPolynomial) : ℝ :=
  1 - poly.p + poly.q

/-- A quadratic polynomial is irreverent if q(q(x)) = 0 has exactly four distinct real solutions -/
def isIrreverent (poly : QuadraticPolynomial) : Prop :=
  numSolutions poly = 4

theorem unique_irreverent_polynomial_exists :
  ∃! poly : QuadraticPolynomial,
    isIrreverent poly ∧
    (∀ other : QuadraticPolynomial, isIrreverent other → rootProduct poly ≤ rootProduct other) ∧
    ∃ y : ℝ, evalAtOne poly = y :=
  sorry


end NUMINAMATH_CALUDE_unique_irreverent_polynomial_exists_l3306_330632


namespace NUMINAMATH_CALUDE_sin_shift_left_l3306_330637

theorem sin_shift_left (x : ℝ) : 
  Real.sin (x + π/4) = Real.sin (x - (-π/4)) := by sorry

end NUMINAMATH_CALUDE_sin_shift_left_l3306_330637


namespace NUMINAMATH_CALUDE_age_when_billy_was_born_l3306_330699

/-- Proves the age when Billy was born given the current ages -/
theorem age_when_billy_was_born
  (my_current_age billy_current_age : ℕ)
  (h1 : my_current_age = 4 * billy_current_age)
  (h2 : billy_current_age = 4)
  : my_current_age - billy_current_age = my_current_age - billy_current_age :=
by sorry

end NUMINAMATH_CALUDE_age_when_billy_was_born_l3306_330699


namespace NUMINAMATH_CALUDE_solve_for_x_l3306_330665

theorem solve_for_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 1) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3306_330665


namespace NUMINAMATH_CALUDE_horner_method_v₂_l3306_330658

/-- Horner's method for a polynomial of degree 6 -/
def horner (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) (x : ℤ) : ℤ × ℤ × ℤ :=
  let v₀ := a₆
  let v₁ := v₀ * x + a₅
  let v₂ := v₁ * x + a₄
  (v₀, v₁, v₂)

/-- The polynomial f(x) = 208 + 9x² + 6x⁴ + x⁶ -/
def f (x : ℤ) : ℤ := 208 + 9*x^2 + 6*x^4 + x^6

theorem horner_method_v₂ : 
  (horner 208 0 9 0 6 0 1 (-4)).2.2 = 22 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₂_l3306_330658


namespace NUMINAMATH_CALUDE_tangent_parallel_points_main_theorem_l3306_330609

/-- The function f(x) = x³ + x - 2 --/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_points :
  {x : ℝ | f' x = 4} = {1, -1} :=
sorry

theorem main_theorem :
  {p : ℝ × ℝ | p.1 ∈ {x : ℝ | f' x = 4} ∧ p.2 = f p.1} = {(1, 0), (-1, -4)} :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_main_theorem_l3306_330609


namespace NUMINAMATH_CALUDE_only_students_far_from_school_not_set_l3306_330686

-- Define the groups of objects
def right_angled_triangles : Set (Set ℝ) := sorry
def points_on_unit_circle : Set (ℝ × ℝ) := sorry
def students_far_from_school : Set String := sorry
def homeroom_teachers : Set String := sorry

-- Define a predicate for well-defined sets
def is_well_defined_set (S : Set α) : Prop := sorry

-- Theorem statement
theorem only_students_far_from_school_not_set :
  is_well_defined_set right_angled_triangles ∧
  is_well_defined_set points_on_unit_circle ∧
  ¬ is_well_defined_set students_far_from_school ∧
  is_well_defined_set homeroom_teachers :=
sorry

end NUMINAMATH_CALUDE_only_students_far_from_school_not_set_l3306_330686


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l3306_330648

/-- Given a point (3, -4), prove that after reflecting it across the x-axis
    and translating it 3 units to the right, the resulting coordinates are (6, 4). -/
theorem circle_reflection_translation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point := (initial_point.1, -initial_point.2)
  let translated_point := (reflected_point.1 + 3, reflected_point.2)
  translated_point = (6, 4) := by sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l3306_330648


namespace NUMINAMATH_CALUDE_pascal_triangle_20th_number_in_25_number_row_l3306_330688

theorem pascal_triangle_20th_number_in_25_number_row : 
  let n : ℕ := 24  -- The row number (0-indexed) for a row with 25 numbers
  let k : ℕ := 19  -- The 0-indexed position of the 20th number
  Nat.choose n k = 4252 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_20th_number_in_25_number_row_l3306_330688


namespace NUMINAMATH_CALUDE_expensive_fluid_price_is_30_l3306_330602

/-- Represents the cost of cleaning fluids and drums --/
structure CleaningSupplies where
  total_drums : ℕ
  expensive_drums : ℕ
  cheap_price : ℕ
  total_cost : ℕ

/-- Calculates the price of the more expensive fluid per drum --/
def expensive_fluid_price (supplies : CleaningSupplies) : ℕ :=
  (supplies.total_cost - (supplies.total_drums - supplies.expensive_drums) * supplies.cheap_price) / supplies.expensive_drums

/-- Theorem stating that the price of the more expensive fluid is $30 per drum --/
theorem expensive_fluid_price_is_30 (supplies : CleaningSupplies) 
    (h1 : supplies.total_drums = 7)
    (h2 : supplies.expensive_drums = 2)
    (h3 : supplies.cheap_price = 20)
    (h4 : supplies.total_cost = 160) :
  expensive_fluid_price supplies = 30 := by
  sorry

#eval expensive_fluid_price { total_drums := 7, expensive_drums := 2, cheap_price := 20, total_cost := 160 }

end NUMINAMATH_CALUDE_expensive_fluid_price_is_30_l3306_330602


namespace NUMINAMATH_CALUDE_max_value_constraint_l3306_330617

theorem max_value_constraint (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) : x^2 + y^2 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3306_330617


namespace NUMINAMATH_CALUDE_circle_area_special_condition_l3306_330662

theorem circle_area_special_condition (r : ℝ) (h : (2 * r)^2 = 8 * (2 * π * r)) :
  π * r^2 = 16 * π^3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_special_condition_l3306_330662


namespace NUMINAMATH_CALUDE_equation_solution_l3306_330645

theorem equation_solution :
  ∃ x : ℚ, (x - 1) / 2 - (2 - 3*x) / 3 = 1 ∧ x = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3306_330645


namespace NUMINAMATH_CALUDE_third_number_value_l3306_330624

theorem third_number_value (a b c : ℝ) : 
  a + b + c = 500 →
  a = 200 →
  b = 2 * c →
  c = 100 := by
sorry

end NUMINAMATH_CALUDE_third_number_value_l3306_330624


namespace NUMINAMATH_CALUDE_opinion_change_difference_l3306_330697

theorem opinion_change_difference (initial_yes initial_no final_yes final_no : ℝ) :
  initial_yes = 30 →
  initial_no = 70 →
  final_yes = 60 →
  final_no = 40 →
  initial_yes + initial_no = 100 →
  final_yes + final_no = 100 →
  ∃ (min_change max_change : ℝ),
    (min_change ≤ max_change) ∧
    (∀ (change : ℝ), change ≥ min_change ∧ change ≤ max_change →
      ∃ (yes_to_no no_to_yes : ℝ),
        yes_to_no ≥ 0 ∧
        no_to_yes ≥ 0 ∧
        yes_to_no + no_to_yes = change ∧
        initial_yes - yes_to_no + no_to_yes = final_yes) ∧
    (max_change - min_change = 30) :=
by sorry

end NUMINAMATH_CALUDE_opinion_change_difference_l3306_330697


namespace NUMINAMATH_CALUDE_lily_lemur_hops_l3306_330672

theorem lily_lemur_hops : 
  let hop_fraction : ℚ := 1/4
  let num_hops : ℕ := 4
  let total_distance : ℚ := (1 - (1 - hop_fraction)^num_hops) / hop_fraction
  total_distance = 175/256 := by sorry

end NUMINAMATH_CALUDE_lily_lemur_hops_l3306_330672


namespace NUMINAMATH_CALUDE_fraction_problem_l3306_330630

theorem fraction_problem (n : ℝ) (f : ℝ) : 
  n = 630.0000000000009 →
  (4/15 * 5/7 * n) > (4/9 * f * n + 8) →
  f = 0.4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3306_330630


namespace NUMINAMATH_CALUDE_price_reduction_equivalence_l3306_330696

theorem price_reduction_equivalence : 
  let first_reduction : ℝ := 0.25
  let second_reduction : ℝ := 0.20
  let equivalent_reduction : ℝ := 1 - (1 - first_reduction) * (1 - second_reduction)
  equivalent_reduction = 0.40
  := by sorry

end NUMINAMATH_CALUDE_price_reduction_equivalence_l3306_330696


namespace NUMINAMATH_CALUDE_lines_parallel_perpendicular_l3306_330650

/-- Two lines l₁ and l₂ in the plane --/
structure Lines (m : ℝ) where
  l₁ : ℝ → ℝ → ℝ := λ x y => 2*x + (m+1)*y + 4
  l₂ : ℝ → ℝ → ℝ := λ x y => m*x + 3*y - 6

/-- The lines are parallel --/
def parallel (m : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 2 = k * m ∧ m + 1 = k * 3 ∧ 4 ≠ k * (-6)

/-- The lines are perpendicular --/
def perpendicular (m : ℝ) : Prop :=
  2 * m + 3 * (m + 1) = 0

/-- Main theorem --/
theorem lines_parallel_perpendicular (m : ℝ) :
  (parallel m ↔ m = 2) ∧ (perpendicular m ↔ m = -3/5) := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_perpendicular_l3306_330650


namespace NUMINAMATH_CALUDE_black_or_white_probability_l3306_330621

/-- The probability of drawing a red ball from the box -/
def prob_red : ℝ := 0.45

/-- The probability of drawing a white ball from the box -/
def prob_white : ℝ := 0.25

/-- The probability of drawing either a black ball or a white ball from the box -/
def prob_black_or_white : ℝ := 1 - prob_red

theorem black_or_white_probability : prob_black_or_white = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_black_or_white_probability_l3306_330621


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3306_330671

-- Problem 1
theorem problem_1 : (π - 2023) ^ 0 - 3 * Real.tan (π / 6) + |1 - Real.sqrt 3| = 0 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  ((2 * x + 1) / (x - 1) - 1) / ((2 * x + x^2) / (x^2 - 2 * x + 1)) = (x - 1) / x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3306_330671


namespace NUMINAMATH_CALUDE_abc_fraction_value_l3306_330628

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 1 / 3)
  (h2 : b * c / (b + c) = 1 / 4)
  (h3 : c * a / (c + a) = 1 / 5) :
  a * b * c / (a * b + b * c + c * a) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l3306_330628


namespace NUMINAMATH_CALUDE_investment_problem_l3306_330690

/-- Investment problem -/
theorem investment_problem (x y : ℕ) (profit_ratio : Rat) (y_investment : ℕ) : 
  profit_ratio = 2 / 6 →
  y_investment = 15000 →
  x = 5000 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3306_330690


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l3306_330653

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (78/19, 41/19)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3*x - 2*y = 8

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 5*x + 3*y = 27

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l3306_330653


namespace NUMINAMATH_CALUDE_principal_correct_l3306_330693

/-- Calculates the final amount after compound interest with varying rates and additional investments -/
def final_amount (principal : ℝ) (initial_rate : ℝ) (rate_increase : ℝ) (years : ℝ) (annual_investment : ℝ) : ℝ :=
  let first_year := principal * (1 + initial_rate) + annual_investment
  let second_year := first_year * (1 + (initial_rate + rate_increase)) + annual_investment
  second_year * (1 + (initial_rate + 2 * rate_increase) * (years - 2))

/-- The principal amount is correct if it results in the expected final amount -/
theorem principal_correct (principal : ℝ) : 
  abs (final_amount principal 0.07 0.02 2.4 200 - 1120) < 0.01 → 
  abs (principal - 556.25) < 0.01 := by
  sorry

#eval final_amount 556.25 0.07 0.02 2.4 200

end NUMINAMATH_CALUDE_principal_correct_l3306_330693


namespace NUMINAMATH_CALUDE_smaller_city_size_l3306_330655

/-- Proves that given a population density of 80 people per cubic yard, 
    if a larger city with 9000 cubic yards has 208000 more people than a smaller city, 
    then the smaller city has 6400 cubic yards. -/
theorem smaller_city_size (density : ℕ) (larger_city_size : ℕ) (population_difference : ℕ) :
  density = 80 →
  larger_city_size = 9000 →
  population_difference = 208000 →
  (larger_city_size * density) - (population_difference) = 6400 * density :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_city_size_l3306_330655


namespace NUMINAMATH_CALUDE_boat_trip_time_l3306_330670

theorem boat_trip_time (v : ℝ) :
  (90 = (v - 3) * (T + 0.5)) →
  (90 = (v + 3) * T) →
  (T > 0) →
  T = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_boat_trip_time_l3306_330670


namespace NUMINAMATH_CALUDE_min_boxes_to_eliminate_l3306_330623

/-- Represents the game setup with total boxes and valuable boxes -/
structure GameSetup :=
  (total_boxes : ℕ)
  (valuable_boxes : ℕ)

/-- Calculates the probability of holding a valuable box -/
def probability (setup : GameSetup) (eliminated : ℕ) : ℚ :=
  setup.valuable_boxes / (setup.total_boxes - eliminated)

/-- Theorem stating the minimum number of boxes to eliminate -/
theorem min_boxes_to_eliminate (setup : GameSetup) 
  (h1 : setup.total_boxes = 30)
  (h2 : setup.valuable_boxes = 9) :
  ∃ (n : ℕ), 
    (n = 3) ∧ 
    (probability setup n ≥ 1/3) ∧ 
    (∀ m : ℕ, m < n → probability setup m < 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_boxes_to_eliminate_l3306_330623


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3306_330638

/-- The quadratic function f(x) = -x^2 + 2x has a maximum value of 1. -/
theorem quadratic_max_value (x : ℝ) : 
  (∀ y : ℝ, -y^2 + 2*y ≤ 1) ∧ (∃ z : ℝ, -z^2 + 2*z = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3306_330638


namespace NUMINAMATH_CALUDE_divisibility_condition_l3306_330605

theorem divisibility_condition (M : ℕ) : 
  M > 0 ∧ M < 10 →
  (5 ∣ (1989^M + M^1989)) ↔ (M = 1 ∨ M = 4) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3306_330605


namespace NUMINAMATH_CALUDE_factorization_equality_l3306_330664

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3306_330664


namespace NUMINAMATH_CALUDE_sum_mod_five_equals_four_l3306_330646

/-- Given positive integers a, b, c less than 5 satisfying certain congruences,
    prove that their sum modulo 5 is 4. -/
theorem sum_mod_five_equals_four
  (a b c : ℕ)
  (ha : 0 < a ∧ a < 5)
  (hb : 0 < b ∧ b < 5)
  (hc : 0 < c ∧ c < 5)
  (h1 : a * b * c % 5 = 1)
  (h2 : 3 * c % 5 = 2)
  (h3 : 4 * b % 5 = (3 + b) % 5) :
  (a + b + c) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_five_equals_four_l3306_330646


namespace NUMINAMATH_CALUDE_red_cards_count_l3306_330678

theorem red_cards_count (total_cards : ℕ) (total_credits : ℕ) 
  (red_card_cost blue_card_cost : ℕ) :
  total_cards = 20 →
  total_credits = 84 →
  red_card_cost = 3 →
  blue_card_cost = 5 →
  ∃ (red_cards blue_cards : ℕ),
    red_cards + blue_cards = total_cards ∧
    red_cards * red_card_cost + blue_cards * blue_card_cost = total_credits ∧
    red_cards = 8 :=
by sorry

end NUMINAMATH_CALUDE_red_cards_count_l3306_330678


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3306_330679

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  (Nat.choose 6 3 : ℝ) * p^3 * (1 - p)^3 = (1 : ℝ) / 20 →
  p = 0.125 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3306_330679


namespace NUMINAMATH_CALUDE_both_languages_students_l3306_330640

/-- The number of students taking both French and Spanish classes -/
def students_taking_both (french_class : ℕ) (spanish_class : ℕ) (total_students : ℕ) (students_one_language : ℕ) : ℕ :=
  french_class + spanish_class - total_students

theorem both_languages_students :
  let french_class : ℕ := 21
  let spanish_class : ℕ := 21
  let students_one_language : ℕ := 30
  let total_students : ℕ := students_one_language + students_taking_both french_class spanish_class total_students students_one_language
  students_taking_both french_class spanish_class total_students students_one_language = 6 := by
  sorry

end NUMINAMATH_CALUDE_both_languages_students_l3306_330640


namespace NUMINAMATH_CALUDE_parallel_transitive_l3306_330687

-- Define the type for lines in space
structure Line3D where
  -- We don't need to specify the internal structure of a line
  -- as we're only concerned with their relationships

-- Define the parallelism relation
def parallel (l1 l2 : Line3D) : Prop :=
  sorry  -- The actual definition is not important for this statement

-- State the theorem
theorem parallel_transitive (a b c : Line3D) 
  (hab : parallel a b) (hbc : parallel b c) : 
  parallel a c :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_l3306_330687


namespace NUMINAMATH_CALUDE_door_height_is_eight_l3306_330627

/-- Represents the dimensions of a rectangular door and a pole. -/
structure DoorPole where
  pole_length : ℝ
  door_width : ℝ
  door_height : ℝ
  door_diagonal : ℝ

/-- The conditions of the door and pole problem. -/
def door_pole_conditions (d : DoorPole) : Prop :=
  d.pole_length = d.door_width + 4 ∧
  d.pole_length = d.door_height + 2 ∧
  d.pole_length = d.door_diagonal ∧
  d.door_diagonal^2 = d.door_width^2 + d.door_height^2

/-- The theorem stating that under the given conditions, the door height is 8 feet. -/
theorem door_height_is_eight (d : DoorPole) 
  (h : door_pole_conditions d) : d.door_height = 8 := by
  sorry

end NUMINAMATH_CALUDE_door_height_is_eight_l3306_330627


namespace NUMINAMATH_CALUDE_red_bowl_possible_values_l3306_330644

/-- Represents the distribution of balls in the three bowls -/
structure BallDistribution where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Checks if a ball distribution is valid according to the problem conditions -/
def is_valid_distribution (d : BallDistribution) : Prop :=
  d.red + d.blue + d.yellow = 27 ∧
  ∃ (red_sum blue_sum yellow_sum : ℕ),
    red_sum + blue_sum + yellow_sum = (27 * 28) / 2 ∧
    d.red > 0 → red_sum / d.red = 15 ∧
    d.blue > 0 → blue_sum / d.blue = 3 ∧
    d.yellow > 0 → yellow_sum / d.yellow = 18

/-- The theorem stating the possible values for the number of balls in the red bowl -/
theorem red_bowl_possible_values :
  ∀ d : BallDistribution, is_valid_distribution d → d.red ∈ ({11, 16, 21} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_red_bowl_possible_values_l3306_330644


namespace NUMINAMATH_CALUDE_absolute_value_fraction_calculation_l3306_330633

theorem absolute_value_fraction_calculation : 
  |(-7)| / ((2/3) - (1/5)) - (1/2) * ((-4)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_calculation_l3306_330633


namespace NUMINAMATH_CALUDE_cos_150_degrees_l3306_330620

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l3306_330620


namespace NUMINAMATH_CALUDE_meeting_unexpectedly_is_random_l3306_330601

/-- Represents an event --/
inductive Event
| WinterToSpring
| FishingMoonInWater
| SeekingFishOnTree
| MeetingUnexpectedly

/-- Defines whether an event is certain --/
def isCertain : Event → Prop
| Event.WinterToSpring => True
| _ => False

/-- Defines whether an event is impossible --/
def isImpossible : Event → Prop
| Event.FishingMoonInWater => True
| Event.SeekingFishOnTree => True
| _ => False

/-- Defines a random event --/
def isRandom (e : Event) : Prop :=
  ¬(isCertain e) ∧ ¬(isImpossible e)

/-- Theorem: Meeting unexpectedly is a random event --/
theorem meeting_unexpectedly_is_random :
  isRandom Event.MeetingUnexpectedly :=
by sorry

end NUMINAMATH_CALUDE_meeting_unexpectedly_is_random_l3306_330601


namespace NUMINAMATH_CALUDE_total_candy_cases_l3306_330612

/-- The Sweet Shop's candy inventory -/
structure CandyInventory where
  chocolate_cases : ℕ
  lollipop_cases : ℕ

/-- The total number of candy cases in the inventory -/
def total_cases (inventory : CandyInventory) : ℕ :=
  inventory.chocolate_cases + inventory.lollipop_cases

/-- Theorem: The total number of candy cases is 80 -/
theorem total_candy_cases :
  ∃ (inventory : CandyInventory),
    inventory.chocolate_cases = 25 ∧
    inventory.lollipop_cases = 55 ∧
    total_cases inventory = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_cases_l3306_330612


namespace NUMINAMATH_CALUDE_not_all_prime_in_sequence_l3306_330639

-- Define the recursive sequence
def x (n : ℕ) (x₀ a b : ℕ) : ℕ :=
  match n with
  | 0 => x₀
  | n + 1 => x n x₀ a b * a + b

-- Theorem statement
theorem not_all_prime_in_sequence (x₀ a b : ℕ) :
  ∃ n : ℕ, ¬ Nat.Prime (x n x₀ a b) :=
by sorry

end NUMINAMATH_CALUDE_not_all_prime_in_sequence_l3306_330639


namespace NUMINAMATH_CALUDE_range_of_a_l3306_330604

def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}

def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_a (a : ℝ) :
  (M a ∪ N = N) ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3306_330604


namespace NUMINAMATH_CALUDE_profit_maximization_l3306_330659

noncomputable def y (x : ℝ) : ℝ := 20 * x - 3 * x^2 + 96 * Real.log x - 90

theorem profit_maximization :
  ∃ (x_max : ℝ), 
    (4 ≤ x_max ∧ x_max ≤ 12) ∧
    (∀ x, 4 ≤ x ∧ x ≤ 12 → y x ≤ y x_max) ∧
    x_max = 6 ∧
    y x_max = 96 * Real.log 6 - 78 :=
sorry

end NUMINAMATH_CALUDE_profit_maximization_l3306_330659


namespace NUMINAMATH_CALUDE_concentric_circles_radii_product_l3306_330613

theorem concentric_circles_radii_product (r₁ r₂ r₃ : ℝ) : 
  r₁ = 2 →
  (r₂^2 - r₁^2 = r₁^2) →
  (r₃^2 - r₂^2 = r₁^2) →
  (r₁ * r₂ * r₃)^2 = 384 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_product_l3306_330613


namespace NUMINAMATH_CALUDE_coin_arrangement_l3306_330685

theorem coin_arrangement (n : ℕ) : 
  n ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) → 
  (n * 4 = 12 ↔ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_coin_arrangement_l3306_330685


namespace NUMINAMATH_CALUDE_perimeter_difference_is_six_l3306_330625

/-- Calculate the perimeter of a rectangle --/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculate the perimeter of a divided rectangle --/
def divided_rectangle_perimeter (length width divisions : ℕ) : ℕ :=
  rectangle_perimeter length width + 2 * divisions

/-- The positive difference between the perimeters of two specific rectangles --/
def perimeter_difference : ℕ :=
  Int.natAbs (rectangle_perimeter 7 3 - divided_rectangle_perimeter 6 2 5)

theorem perimeter_difference_is_six : perimeter_difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_is_six_l3306_330625


namespace NUMINAMATH_CALUDE_zachary_did_more_pushups_l3306_330618

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference in push-ups between Zachary and David -/
def pushup_difference : ℕ := zachary_pushups - david_pushups

theorem zachary_did_more_pushups : pushup_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_zachary_did_more_pushups_l3306_330618


namespace NUMINAMATH_CALUDE_team_selection_count_l3306_330669

def boys : ℕ := 7
def girls : ℕ := 9
def team_size : ℕ := 6
def min_boys : ℕ := 2

theorem team_selection_count :
  (Finset.sum (Finset.range (boys - min_boys + 1))
    (fun k => Nat.choose boys (k + min_boys) * Nat.choose girls (team_size - (k + min_boys)))) = 6846 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l3306_330669


namespace NUMINAMATH_CALUDE_organize_objects_groups_l3306_330674

/-- The total number of groups created when organizing objects into groups -/
def totalGroups (eggs bananas marbles : ℕ) (eggGroupSize bananaGroupSize marbleGroupSize : ℕ) : ℕ :=
  (eggs / eggGroupSize) + (bananas / bananaGroupSize) + (marbles / marbleGroupSize)

/-- Theorem stating that organizing 57 eggs in groups of 7, 120 bananas in groups of 10,
    and 248 marbles in groups of 8 results in 51 total groups -/
theorem organize_objects_groups : totalGroups 57 120 248 7 10 8 = 51 := by
  sorry

end NUMINAMATH_CALUDE_organize_objects_groups_l3306_330674


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l3306_330666

theorem largest_power_of_two_dividing_difference : ∃ (k : ℕ), k = 13 ∧ 
  (∀ (n : ℕ), 2^n ∣ (10^10 - 2^10) → n ≤ k) ∧ 
  (2^k ∣ (10^10 - 2^10)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l3306_330666


namespace NUMINAMATH_CALUDE_cos_300_degrees_l3306_330603

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l3306_330603


namespace NUMINAMATH_CALUDE_intercept_sum_l3306_330635

/-- The modulus for the congruence equation -/
def m : ℕ := 11

/-- The congruence equation -/
def congruence_eq (x y : ℤ) : Prop :=
  (2 * x) % m = (3 * y + 1) % m

/-- The x-intercept of the congruence equation -/
def x_intercept : ℤ :=
  6

/-- The y-intercept of the congruence equation -/
def y_intercept : ℤ :=
  7

/-- Theorem stating that the sum of x-intercept and y-intercept is 13 -/
theorem intercept_sum :
  x_intercept + y_intercept = 13 ∧
  congruence_eq x_intercept 0 ∧
  congruence_eq 0 y_intercept :=
sorry

end NUMINAMATH_CALUDE_intercept_sum_l3306_330635


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3306_330626

theorem isosceles_right_triangle_area (leg : ℝ) (h_leg : leg = 3) :
  let triangle_area := (1 / 2) * leg * leg
  triangle_area = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3306_330626


namespace NUMINAMATH_CALUDE_medium_pizzas_ordered_l3306_330649

/-- Represents the number of slices in different pizza sizes --/
structure PizzaSlices where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of pizzas ordered --/
structure PizzaOrder where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of slices for a given order --/
def totalSlices (slices : PizzaSlices) (order : PizzaOrder) : Nat :=
  slices.small * order.small + slices.medium * order.medium + slices.large * order.large

/-- The main theorem to prove --/
theorem medium_pizzas_ordered 
  (slices : PizzaSlices) 
  (order : PizzaOrder) 
  (h1 : slices.small = 6)
  (h2 : slices.medium = 8)
  (h3 : slices.large = 12)
  (h4 : order.small + order.medium + order.large = 15)
  (h5 : order.small = 4)
  (h6 : totalSlices slices order = 136) :
  order.medium = 5 := by
  sorry

end NUMINAMATH_CALUDE_medium_pizzas_ordered_l3306_330649


namespace NUMINAMATH_CALUDE_roots_cubic_sum_l3306_330615

theorem roots_cubic_sum (a b c : ℝ) (r s : ℝ) (h : a ≠ 0) (h1 : c ≠ 0) : 
  (a * r^2 + b * r + c = 0) → 
  (a * s^2 + b * s + c = 0) → 
  (1 / r^3 + 1 / s^3 = (-b^3 + 3*a*b*c) / c^3) :=
by sorry

end NUMINAMATH_CALUDE_roots_cubic_sum_l3306_330615


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l3306_330641

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : ∃ (k : ℤ), n^4 - n^2 = 12 * k ∧ ∀ (m : ℤ), (∀ (n : ℤ), ∃ (l : ℤ), n^4 - n^2 = m * l) → m ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l3306_330641


namespace NUMINAMATH_CALUDE_new_stationary_points_order_l3306_330695

-- Define the "new stationary point" for each function
def alpha : ℝ := 1

-- β is implicitly defined by the equation ln(β+1) = 1/(β+1)
def beta_equation (x : ℝ) : Prop := Real.log (x + 1) = 1 / (x + 1) ∧ x > 0

-- γ is implicitly defined by the equation γ³ - 1 = 3γ²
def gamma_equation (x : ℝ) : Prop := x^3 - 1 = 3 * x^2 ∧ x > 0

-- State the theorem
theorem new_stationary_points_order 
  (beta : ℝ) (h_beta : beta_equation beta)
  (gamma : ℝ) (h_gamma : gamma_equation gamma) :
  gamma > alpha ∧ alpha > beta := by
  sorry

end NUMINAMATH_CALUDE_new_stationary_points_order_l3306_330695


namespace NUMINAMATH_CALUDE_girls_in_school_l3306_330607

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girl_boy_diff : ℕ) :
  total_students = 1600 →
  sample_size = 200 →
  girl_boy_diff = 20 →
  ∃ (girls : ℕ) (boys : ℕ),
    girls + boys = total_students ∧
    girls * sample_size = (total_students - girls) * (sample_size - girl_boy_diff) ∧
    girls = 720 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_school_l3306_330607


namespace NUMINAMATH_CALUDE_n_div_f_n_equals_5_for_625_n_div_f_n_equals_1_solutions_l3306_330698

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Defines the function f as described in the problem -/
def f (n : ThreeDigitNumber) : Nat :=
  let a := n.hundreds
  let b := n.tens
  let c := n.ones
  a + b + c + a * b + b * c + c * a + a * b * c

theorem n_div_f_n_equals_5_for_625 :
  let n : ThreeDigitNumber := ⟨6, 2, 5, by simp, by simp, by simp⟩
  (n.toNat : ℚ) / f n = 5 := by sorry

theorem n_div_f_n_equals_1_solutions :
  {n : ThreeDigitNumber | (n.toNat : ℚ) / f n = 1} =
  {⟨1, 9, 9, by simp, by simp, by simp⟩,
   ⟨2, 9, 9, by simp, by simp, by simp⟩,
   ⟨3, 9, 9, by simp, by simp, by simp⟩,
   ⟨4, 9, 9, by simp, by simp, by simp⟩,
   ⟨5, 9, 9, by simp, by simp, by simp⟩,
   ⟨6, 9, 9, by simp, by simp, by simp⟩,
   ⟨7, 9, 9, by simp, by simp, by simp⟩,
   ⟨8, 9, 9, by simp, by simp, by simp⟩,
   ⟨9, 9, 9, by simp, by simp, by simp⟩} := by sorry

end NUMINAMATH_CALUDE_n_div_f_n_equals_5_for_625_n_div_f_n_equals_1_solutions_l3306_330698


namespace NUMINAMATH_CALUDE_power_of_four_l3306_330680

theorem power_of_four (n : ℕ) : 
  (2 * n + 7 + 2 = 31) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_l3306_330680


namespace NUMINAMATH_CALUDE_sum_of_max_min_l3306_330654

theorem sum_of_max_min (a b c d : ℚ) : 
  a = 11/100 ∧ b = 98/100 ∧ c = 3/4 ∧ d = 2/3 →
  max a (max b (max c d)) + min a (min b (min c d)) = 109/100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_l3306_330654


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l3306_330681

/-- Represents the number of matching pairs of shoes after losing some shoes. -/
def MaxPairsAfterLoss (totalPairs : ℕ) (colors : ℕ) (sizes : ℕ) (shoesLost : ℕ) : ℕ :=
  min (totalPairs - shoesLost) (colors * sizes)

/-- Theorem stating the maximum number of matching pairs after losing shoes. -/
theorem max_pairs_after_loss :
  MaxPairsAfterLoss 23 6 3 9 = 14 := by
  sorry

#eval MaxPairsAfterLoss 23 6 3 9

end NUMINAMATH_CALUDE_max_pairs_after_loss_l3306_330681


namespace NUMINAMATH_CALUDE_target_hit_probability_l3306_330614

theorem target_hit_probability (total_groups : ℕ) (hit_groups : ℕ) : 
  total_groups = 20 → hit_groups = 5 → 
  (hit_groups : ℚ) / total_groups = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3306_330614


namespace NUMINAMATH_CALUDE_special_function_properties_l3306_330668

/-- An increasing function f: ℝ₊ → ℝ₊ satisfying f(xy) = f(x)f(y) for all x, y > 0, and f(2) = 4 -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 0 → y > 0 → f (x * y) = f x * f y) ∧
  (∀ x y, x > y → x > 0 → y > 0 → f x > f y) ∧
  f 2 = 4

theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 1 = 1 ∧ f 8 = 64 ∧ Set.Ioo 3 (7/2) = {x | 16 * f (1 / (x - 3)) ≥ f (2 * x + 1)} := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l3306_330668


namespace NUMINAMATH_CALUDE_boric_acid_weight_l3306_330642

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Boron in g/mol -/
def atomic_weight_B : ℝ := 10.81

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Hydrogen atoms in Boric acid -/
def num_H : ℕ := 3

/-- The number of Boron atoms in Boric acid -/
def num_B : ℕ := 1

/-- The number of Oxygen atoms in Boric acid -/
def num_O : ℕ := 3

/-- The molecular weight of Boric acid (H3BO3) in g/mol -/
def molecular_weight_boric_acid : ℝ :=
  num_H * atomic_weight_H + num_B * atomic_weight_B + num_O * atomic_weight_O

theorem boric_acid_weight :
  molecular_weight_boric_acid = 61.834 := by sorry

end NUMINAMATH_CALUDE_boric_acid_weight_l3306_330642
