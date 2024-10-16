import Mathlib

namespace NUMINAMATH_CALUDE_vertical_translation_by_two_l2020_202010

/-- For any real-valued function f and any real number x,
    f(x) + 2 is equal to a vertical translation of f(x) by 2 units upward -/
theorem vertical_translation_by_two (f : ℝ → ℝ) (x : ℝ) :
  f x + 2 = (fun y ↦ f y + 2) x :=
by sorry

end NUMINAMATH_CALUDE_vertical_translation_by_two_l2020_202010


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2020_202028

theorem polynomial_factorization (x : ℝ) :
  x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2020_202028


namespace NUMINAMATH_CALUDE_probability_one_or_two_in_pascal_l2020_202049

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def ones_in_pascal (n : ℕ) : ℕ := if n = 0 then 0 else 2 * n - 1

/-- The number of 2's in the first n rows of Pascal's Triangle -/
def twos_in_pascal (n : ℕ) : ℕ := if n ≤ 2 then 0 else 2 * (n - 2)

/-- The probability of selecting 1 or 2 from the first 20 rows of Pascal's Triangle -/
theorem probability_one_or_two_in_pascal : 
  (ones_in_pascal 20 + twos_in_pascal 20 : ℚ) / pascal_triangle_elements 20 = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_or_two_in_pascal_l2020_202049


namespace NUMINAMATH_CALUDE_exists_integer_between_sqrt2_and_sqrt11_l2020_202019

theorem exists_integer_between_sqrt2_and_sqrt11 :
  ∃ m : ℤ, Real.sqrt 2 < m ∧ m < Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_between_sqrt2_and_sqrt11_l2020_202019


namespace NUMINAMATH_CALUDE_no_factors_l2020_202073

def f (x : ℝ) : ℝ := x^4 + 2*x^2 + 9

def g₁ (x : ℝ) : ℝ := x^2 + 3
def g₂ (x : ℝ) : ℝ := x + 1
def g₃ (x : ℝ) : ℝ := x^2 - 3
def g₄ (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem no_factors : 
  (∀ x : ℝ, f x ≠ 0 → g₁ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₂ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₃ x ≠ 0) ∧
  (∀ x : ℝ, f x ≠ 0 → g₄ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_factors_l2020_202073


namespace NUMINAMATH_CALUDE_prob_same_color_is_34_100_l2020_202056

/-- Represents an urn with balls of different colors -/
structure Urn :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Calculates the total number of balls in an urn -/
def Urn.total (u : Urn) : ℕ := u.blue + u.red + u.green

/-- Calculates the probability of drawing a ball of a specific color from an urn -/
def prob_color (u : Urn) (color : ℕ) : ℚ :=
  color / u.total

/-- Calculates the probability of drawing balls of the same color from two urns -/
def prob_same_color (u1 u2 : Urn) : ℚ :=
  prob_color u1 u1.blue * prob_color u2 u2.blue +
  prob_color u1 u1.red * prob_color u2 u2.red +
  prob_color u1 u1.green * prob_color u2 u2.green

/-- The main theorem stating that the probability of drawing balls of the same color
    from the given urns is 0.34 -/
theorem prob_same_color_is_34_100 :
  let u1 : Urn := ⟨2, 3, 5⟩
  let u2 : Urn := ⟨4, 2, 4⟩
  prob_same_color u1 u2 = 34/100 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_34_100_l2020_202056


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2020_202003

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_composite_no_small_factors : 
  (∀ n < 289, is_composite n → ∃ p, p < 15 ∧ Nat.Prime p ∧ p ∣ n) ∧ 
  is_composite 289 ∧
  (∀ p, Nat.Prime p → p ∣ 289 → p ≥ 15) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2020_202003


namespace NUMINAMATH_CALUDE_egg_yolk_count_l2020_202083

theorem egg_yolk_count (total_eggs : ℕ) (double_yolk_eggs : ℕ) : 
  total_eggs = 12 → double_yolk_eggs = 5 → 
  (total_eggs - double_yolk_eggs) + 2 * double_yolk_eggs = 17 := by
  sorry

#check egg_yolk_count

end NUMINAMATH_CALUDE_egg_yolk_count_l2020_202083


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l2020_202005

theorem larger_number_in_ratio (a b : ℕ+) : 
  a.val * 5 = b.val * 2 →  -- ratio condition
  Nat.lcm a.val b.val = 160 →  -- LCM condition
  b = 160 := by  -- conclusion: larger number is 160
sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l2020_202005


namespace NUMINAMATH_CALUDE_product_ab_equals_negative_one_l2020_202023

theorem product_ab_equals_negative_one (a b : ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) : 
  a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_product_ab_equals_negative_one_l2020_202023


namespace NUMINAMATH_CALUDE_james_huskies_count_l2020_202035

/-- The number of huskies James has -/
def num_huskies : ℕ := sorry

/-- The number of pitbulls James has -/
def num_pitbulls : ℕ := 2

/-- The number of golden retrievers James has -/
def num_golden_retrievers : ℕ := 4

/-- The number of pups each husky and pitbull has -/
def pups_per_husky_pitbull : ℕ := 3

/-- The additional number of pups each golden retriever has compared to huskies -/
def additional_pups_golden : ℕ := 2

/-- The difference between total pups and adult dogs -/
def pup_adult_difference : ℕ := 30

theorem james_huskies_count :
  num_huskies = 5 ∧
  num_huskies * pups_per_husky_pitbull +
  num_pitbulls * pups_per_husky_pitbull +
  num_golden_retrievers * (pups_per_husky_pitbull + additional_pups_golden) =
  num_huskies + num_pitbulls + num_golden_retrievers + pup_adult_difference :=
sorry

end NUMINAMATH_CALUDE_james_huskies_count_l2020_202035


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l2020_202091

theorem polynomial_perfect_square (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5*x + 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l2020_202091


namespace NUMINAMATH_CALUDE_complement_of_B_l2020_202014

def A : Set ℕ := {1, 2, 3}
def B (a : ℕ) : Set ℕ := {a + 2, a}

theorem complement_of_B (a : ℕ) (h : A ∩ B a = B a) : 
  (A \ B a) = {2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_B_l2020_202014


namespace NUMINAMATH_CALUDE_equation_solution_l2020_202013

theorem equation_solution : ∃ r : ℝ, (24 - 5 = 3 * r + 7) ∧ (r = 4) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2020_202013


namespace NUMINAMATH_CALUDE_perfect_power_arithmetic_sequence_l2020_202015

/-- A perfect power is a number that can be expressed as an integer raised to a positive integer exponent. -/
def IsPerfectPower (x : ℕ) : Prop :=
  ∃ (b e : ℕ), e > 0 ∧ x = b ^ e

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def IsArithmeticSequence (s : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ i, i < n → s i = a + i * d

/-- A sequence is non-constant if it has at least two distinct terms. -/
def IsNonConstant (s : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ i j, i < n ∧ j < n ∧ i ≠ j ∧ s i ≠ s j

/-- For any positive integer n, there exists a non-constant arithmetic sequence of length n
    where all terms are perfect powers. -/
theorem perfect_power_arithmetic_sequence (n : ℕ) (hn : n > 0) :
  ∃ (s : ℕ → ℕ),
    IsArithmeticSequence s n ∧
    IsNonConstant s n ∧
    (∀ i, i < n → IsPerfectPower (s i)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_power_arithmetic_sequence_l2020_202015


namespace NUMINAMATH_CALUDE_biggest_measure_for_containers_l2020_202065

theorem biggest_measure_for_containers (a b c : ℕ) 
  (ha : a = 496) (hb : b = 403) (hc : c = 713) : 
  Nat.gcd a (Nat.gcd b c) = 31 := by
  sorry

end NUMINAMATH_CALUDE_biggest_measure_for_containers_l2020_202065


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2014th_term_l2020_202046

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_1_eq_1 : a 1 = 1
  d : ℝ
  d_ne_0 : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  is_geometric : (a 2) ^ 2 = a 1 * a 5

/-- The 2014th term of the arithmetic sequence is 4027 -/
theorem arithmetic_sequence_2014th_term (seq : ArithmeticSequence) : seq.a 2014 = 4027 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2014th_term_l2020_202046


namespace NUMINAMATH_CALUDE_line_intersection_point_sum_l2020_202075

/-- The line equation y = -1/2x + 8 -/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (16, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 8)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs (P.1 * Q.2 - Q.1 * P.2) / 2 = 4 * abs (r * P.2 - P.1 * s) / 2

theorem line_intersection_point_sum : 
  ∀ r s : ℝ, line_equation r s → T_on_PQ r s → area_condition r s → r + s = 14 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_point_sum_l2020_202075


namespace NUMINAMATH_CALUDE_romeo_chocolate_profit_l2020_202069

theorem romeo_chocolate_profit :
  let num_bars : ℕ := 20
  let cost_per_bar : ℕ := 8
  let total_sales : ℕ := 240
  let packaging_cost_per_bar : ℕ := 3
  let advertising_cost : ℕ := 15
  
  let total_cost : ℕ := num_bars * cost_per_bar + num_bars * packaging_cost_per_bar + advertising_cost
  let profit : ℤ := total_sales - total_cost
  
  profit = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_romeo_chocolate_profit_l2020_202069


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2020_202038

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_group_number : ℕ

/-- The theorem for systematic sampling -/
theorem systematic_sampling_theorem (s : SystematicSampling)
  (h1 : s.total_students = 300)
  (h2 : s.sample_size = 20)
  (h3 : s.group_size = s.total_students / s.sample_size)
  (h4 : s.first_group_number < s.group_size)
  (h5 : 231 = s.first_group_number + 15 * s.group_size) :
  s.first_group_number = 6 := by
  sorry

#check systematic_sampling_theorem

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2020_202038


namespace NUMINAMATH_CALUDE_student_percentage_theorem_l2020_202031

theorem student_percentage_theorem (total : ℝ) (h_total_pos : total > 0) : 
  let third_year_percent : ℝ := 0.30
  let not_third_second_ratio : ℝ := 1/7
  let third_year : ℝ := third_year_percent * total
  let not_third_year : ℝ := total - third_year
  let second_year_not_third : ℝ := not_third_second_ratio * not_third_year
  let not_second_year : ℝ := total - second_year_not_third
  (not_second_year / total) * 100 = 90
:= by sorry

end NUMINAMATH_CALUDE_student_percentage_theorem_l2020_202031


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2020_202082

theorem fixed_point_exponential_function :
  ∀ (a : ℝ), a > 0 → ((-2 : ℝ)^((-2 : ℝ) + 2) - 3 = -2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2020_202082


namespace NUMINAMATH_CALUDE_cyclists_circular_track_l2020_202060

/-- Given two cyclists on a circular track starting from the same point in opposite directions
    with speeds of 7 m/s and 8 m/s, meeting at the starting point after 20 seconds,
    the circumference of the track is 300 meters. -/
theorem cyclists_circular_track (speed1 speed2 time : ℝ) (circumference : ℝ) : 
  speed1 = 7 → 
  speed2 = 8 → 
  time = 20 → 
  circumference = (speed1 + speed2) * time → 
  circumference = 300 := by sorry

end NUMINAMATH_CALUDE_cyclists_circular_track_l2020_202060


namespace NUMINAMATH_CALUDE_crow_eating_time_l2020_202017

/-- Represents the time it takes for a crow to eat a certain fraction of nuts -/
def eating_time (fraction : ℚ) : ℚ :=
  7.5 / (1/4) * fraction

theorem crow_eating_time :
  eating_time (1/5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_crow_eating_time_l2020_202017


namespace NUMINAMATH_CALUDE_empty_vessel_mass_l2020_202079

/-- The mass of an empty vessel given the masses when filled with kerosene and water, and the densities of kerosene and water. -/
theorem empty_vessel_mass
  (mass_with_kerosene : ℝ)
  (mass_with_water : ℝ)
  (density_water : ℝ)
  (density_kerosene : ℝ)
  (h1 : mass_with_kerosene = 31)
  (h2 : mass_with_water = 33)
  (h3 : density_water = 1000)
  (h4 : density_kerosene = 800) :
  ∃ (empty_mass : ℝ) (volume : ℝ),
    empty_mass = 23 ∧
    mass_with_kerosene = empty_mass + density_kerosene * volume ∧
    mass_with_water = empty_mass + density_water * volume :=
by sorry

end NUMINAMATH_CALUDE_empty_vessel_mass_l2020_202079


namespace NUMINAMATH_CALUDE_correct_propositions_l2020_202092

-- Define the planes
variable (α β : Set (Point))

-- Define the property of being a plane
def is_plane (p : Set (Point)) : Prop := sorry

-- Define the property of being distinct
def distinct (p q : Set (Point)) : Prop := p ≠ q

-- Define line
def Line : Type := sorry

-- Define the property of a line being within a plane
def line_in_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define perpendicularity between lines
def perp_lines (l1 l2 : Line) : Prop := sorry

-- Define perpendicularity between a line and a plane
def perp_line_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define perpendicularity between planes
def perp_planes (p q : Set (Point)) : Prop := sorry

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Line) (p : Set (Point)) : Prop := sorry

-- Define parallelism between planes
def parallel_planes (p q : Set (Point)) : Prop := sorry

-- State the theorem
theorem correct_propositions 
  (h_planes : is_plane α ∧ is_plane β) 
  (h_distinct : distinct α β) :
  (∀ (l : Line), line_in_plane l α → 
    (∀ (m : Line), line_in_plane m β → perp_lines l m) → 
    perp_planes α β) ∧ 
  (∀ (l : Line), line_in_plane l α → 
    parallel_line_plane l β → 
    parallel_planes α β) ∧ 
  (perp_planes α β → 
    ∃ (l : Line), line_in_plane l α ∧ ¬(perp_line_plane l β)) ∧
  (parallel_planes α β → 
    ∀ (l : Line), line_in_plane l α → 
    parallel_line_plane l β) :=
sorry

end NUMINAMATH_CALUDE_correct_propositions_l2020_202092


namespace NUMINAMATH_CALUDE_trig_expression_equals_seven_l2020_202029

theorem trig_expression_equals_seven :
  2 * Real.sin (390 * π / 180) - Real.tan (-45 * π / 180) + 5 * Real.cos (360 * π / 180) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_seven_l2020_202029


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2020_202036

/-- Calculates the total wet surface area of a rectangular cistern --/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of a specific cistern --/
theorem cistern_wet_surface_area :
  let length : ℝ := 5
  let width : ℝ := 4
  let depth : ℝ := 1.25
  total_wet_surface_area length width depth = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2020_202036


namespace NUMINAMATH_CALUDE_bert_pencil_usage_l2020_202099

/-- The number of days it takes to use up a pencil given the total words per pencil and words per puzzle -/
def days_to_use_pencil (total_words_per_pencil : ℕ) (words_per_puzzle : ℕ) : ℕ :=
  total_words_per_pencil / words_per_puzzle

/-- Theorem stating that it takes Bert 14 days to use up a pencil -/
theorem bert_pencil_usage : days_to_use_pencil 1050 75 = 14 := by
  sorry

#eval days_to_use_pencil 1050 75

end NUMINAMATH_CALUDE_bert_pencil_usage_l2020_202099


namespace NUMINAMATH_CALUDE_quadratic_form_with_factor_l2020_202001

/-- A quadratic expression with (x + 3) as a factor and m = 2 -/
def quadratic_expression (c : ℝ) (x : ℝ) : ℝ :=
  2 * (x + 3) * (x + c)

/-- Theorem stating the form of the quadratic expression -/
theorem quadratic_form_with_factor (f : ℝ → ℝ) :
  (∃ (g : ℝ → ℝ), ∀ x, f x = (x + 3) * g x) →  -- (x + 3) is a factor
  (∃ c, ∀ x, f x = quadratic_expression c x) :=
by
  sorry

#check quadratic_form_with_factor

end NUMINAMATH_CALUDE_quadratic_form_with_factor_l2020_202001


namespace NUMINAMATH_CALUDE_work_completion_time_l2020_202027

/-- Given workers A and B, where A can complete a job in 15 days and B in 9 days,
    if A works for 5 days and then leaves, B will complete the remaining work in 6 days. -/
theorem work_completion_time (a_total_days b_total_days a_worked_days : ℕ) 
    (ha : a_total_days = 15)
    (hb : b_total_days = 9)
    (hw : a_worked_days = 5) : 
    (b_total_days : ℚ) * (1 - (a_worked_days : ℚ) / (a_total_days : ℚ)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2020_202027


namespace NUMINAMATH_CALUDE_fraction_difference_equals_one_l2020_202062

theorem fraction_difference_equals_one (x y : ℝ) (h : x * y = x - y) (h_nonzero : x * y ≠ 0) :
  1 / y - 1 / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_one_l2020_202062


namespace NUMINAMATH_CALUDE_octahedron_triangles_l2020_202025

/-- The number of vertices in a regular octahedron -/
def octahedron_vertices : ℕ := 8

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles that can be constructed by connecting three different vertices of a regular octahedron -/
def distinct_triangles : ℕ := Nat.choose octahedron_vertices triangle_vertices

theorem octahedron_triangles : distinct_triangles = 56 := by sorry

end NUMINAMATH_CALUDE_octahedron_triangles_l2020_202025


namespace NUMINAMATH_CALUDE_football_game_spectators_l2020_202052

theorem football_game_spectators (total_wristbands : ℕ) 
  (wristbands_per_person : ℕ) (h1 : total_wristbands = 234) 
  (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 117 := by
  sorry

end NUMINAMATH_CALUDE_football_game_spectators_l2020_202052


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2020_202022

/-- A cubic equation with parameter p has three natural number roots -/
def has_three_natural_roots (p : ℝ) : Prop :=
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p) ∧
    (5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p) ∧
    (5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p-1)*(z : ℝ) + 1 = 66*p)

/-- If a cubic equation with parameter p has three natural number roots, then p = 76 -/
theorem cubic_equation_roots (p : ℝ) :
  has_three_natural_roots p → p = 76 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2020_202022


namespace NUMINAMATH_CALUDE_sara_height_l2020_202087

/-- Given the heights of Mark, Roy, Joe, and Sara, prove Sara's height is 45 inches -/
theorem sara_height (mark_height joe_height roy_height sara_height : ℕ) 
  (h1 : mark_height = 34)
  (h2 : roy_height = mark_height + 2)
  (h3 : joe_height = roy_height + 3)
  (h4 : sara_height = joe_height + 6) :
  sara_height = 45 := by
  sorry

end NUMINAMATH_CALUDE_sara_height_l2020_202087


namespace NUMINAMATH_CALUDE_propositions_truth_l2020_202051

theorem propositions_truth : 
  (∀ x : ℝ, x < 0 → abs x > x) ∧ 
  (∀ a b : ℝ, a * b < 0 ↔ a / b < 0) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l2020_202051


namespace NUMINAMATH_CALUDE_certain_number_is_five_l2020_202097

/-- The certain number that satisfies the equation given the number of Doberman puppies and Schnauzers -/
def certain_number (D S : ℕ) : ℤ :=
  3 * D - 90 - (D - S)

/-- Theorem stating that the certain number is 5 given the specified conditions -/
theorem certain_number_is_five :
  certain_number 20 55 = 5 := by sorry

end NUMINAMATH_CALUDE_certain_number_is_five_l2020_202097


namespace NUMINAMATH_CALUDE_total_travel_time_l2020_202006

theorem total_travel_time (total_distance : ℝ) (initial_time : ℝ) (lunch_time : ℝ) 
  (h1 : total_distance = 200)
  (h2 : initial_time = 1)
  (h3 : lunch_time = 1)
  (h4 : initial_time * 4 * total_distance / 4 = total_distance) :
  initial_time + lunch_time + (total_distance - total_distance / 4) / (total_distance / 4 / initial_time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_l2020_202006


namespace NUMINAMATH_CALUDE_ratio_equality_l2020_202004

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (x + y - z) / (2 * x - y + z) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2020_202004


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2020_202064

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 + x^2 - 7 * x - 6) = x^3 + 3 * x^2 + 2 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2020_202064


namespace NUMINAMATH_CALUDE_line_segment_ratio_l2020_202077

theorem line_segment_ratio (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l2020_202077


namespace NUMINAMATH_CALUDE_courtyard_length_l2020_202016

/-- Proves that a rectangular courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length (width : ℝ) (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) :
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  num_bricks = 20000 →
  (width * (num_bricks * brick_length * brick_width / width)) = 25 := by
  sorry

#check courtyard_length

end NUMINAMATH_CALUDE_courtyard_length_l2020_202016


namespace NUMINAMATH_CALUDE_inconsistent_weight_problem_l2020_202089

theorem inconsistent_weight_problem :
  ∀ (initial_students : ℕ) (initial_avg_weight : ℝ) 
    (new_students : ℕ) (new_avg_weight : ℝ) 
    (first_new_student_weight : ℝ) (second_new_student_min_weight : ℝ),
  initial_students = 19 →
  initial_avg_weight = 15 →
  new_students = 2 →
  new_avg_weight = 14.6 →
  first_new_student_weight = 12 →
  second_new_student_min_weight = 14 →
  ¬∃ (second_new_student_weight : ℝ),
    (initial_students * initial_avg_weight + first_new_student_weight + second_new_student_weight) / 
      (initial_students + new_students) = new_avg_weight ∧
    second_new_student_weight ≥ second_new_student_min_weight :=
by sorry

end NUMINAMATH_CALUDE_inconsistent_weight_problem_l2020_202089


namespace NUMINAMATH_CALUDE_odd_prime_divisibility_l2020_202009

theorem odd_prime_divisibility (a b n : ℕ) (ha : a > b) (hb : b > 1) (hodd : Odd b) 
  (hn : n > 0) (hdiv : (b^n : ℕ) ∣ (a^n - 1)) : 
  (a^b : ℝ) > (3^n : ℝ) / n := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_divisibility_l2020_202009


namespace NUMINAMATH_CALUDE_fair_hair_percentage_l2020_202039

/-- Given that 10% of employees are women with fair hair and 40% of fair-haired employees
    are women, prove that 25% of employees have fair hair. -/
theorem fair_hair_percentage
  (total_employees : ℕ)
  (women_fair_hair_percentage : ℚ)
  (women_percentage_of_fair_hair : ℚ)
  (h1 : women_fair_hair_percentage = 1 / 10)
  (h2 : women_percentage_of_fair_hair = 2 / 5)
  : (total_employees : ℚ) * 1 / 4 = (total_employees : ℚ) * women_fair_hair_percentage / women_percentage_of_fair_hair :=
by sorry

end NUMINAMATH_CALUDE_fair_hair_percentage_l2020_202039


namespace NUMINAMATH_CALUDE_gasoline_expense_gasoline_expense_proof_l2020_202068

/-- Calculates the amount spent on gasoline given the initial amount, known expenses, money received, and the amount left for the return trip. -/
theorem gasoline_expense (initial_amount : ℝ) (lunch_expense : ℝ) (gift_expense : ℝ) 
  (money_from_grandma : ℝ) (return_trip_money : ℝ) : ℝ :=
  let total_amount := initial_amount + money_from_grandma
  let known_expenses := lunch_expense + gift_expense
  let remaining_after_known_expenses := total_amount - known_expenses
  remaining_after_known_expenses - return_trip_money

/-- Proves that the amount spent on gasoline is $8 given the specific values from the problem. -/
theorem gasoline_expense_proof :
  gasoline_expense 50 15.65 10 20 36.35 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_expense_gasoline_expense_proof_l2020_202068


namespace NUMINAMATH_CALUDE_expression_simplification_l2020_202086

theorem expression_simplification :
  Real.sqrt 12 - 2 * Real.cos (30 * π / 180) - (1/3)⁻¹ = Real.sqrt 3 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2020_202086


namespace NUMINAMATH_CALUDE_fraction_equality_l2020_202084

theorem fraction_equality : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2020_202084


namespace NUMINAMATH_CALUDE_sampling_plans_correct_l2020_202024

/-- Represents a canning factory with its production rate and operating hours. -/
structure CanningFactory where
  production_rate : ℕ  -- cans per hour
  operating_hours : ℕ

/-- Represents a sampling plan with the number of cans sampled and the interval between samples. -/
structure SamplingPlan where
  cans_per_sample : ℕ
  sample_interval : ℕ  -- in minutes

/-- Calculates the total number of cans sampled in a day given a factory and a sampling plan. -/
def total_sampled_cans (factory : CanningFactory) (plan : SamplingPlan) : ℕ :=
  (factory.operating_hours * 60 / plan.sample_interval) * plan.cans_per_sample

/-- Theorem stating that the given sampling plans result in the required number of sampled cans. -/
theorem sampling_plans_correct (factory : CanningFactory) :
  factory.production_rate = 120000 ∧ factory.operating_hours = 12 →
  (∃ plan1200 : SamplingPlan, total_sampled_cans factory plan1200 = 1200 ∧ 
    plan1200.cans_per_sample = 10 ∧ plan1200.sample_interval = 6) ∧
  (∃ plan980 : SamplingPlan, total_sampled_cans factory plan980 = 980 ∧ 
    plan980.cans_per_sample = 49 ∧ plan980.sample_interval = 36) := by
  sorry

end NUMINAMATH_CALUDE_sampling_plans_correct_l2020_202024


namespace NUMINAMATH_CALUDE_dartboard_probability_l2020_202030

theorem dartboard_probability :
  -- Define the probabilities for each sector
  ∀ (prob_E prob_F prob_G prob_H prob_I : ℚ),
  -- Conditions
  prob_E = 1/5 →
  prob_F = 2/5 →
  prob_G = prob_H →
  prob_G = prob_I →
  -- Sum of all probabilities is 1
  prob_E + prob_F + prob_G + prob_H + prob_I = 1 →
  -- Conclusion: probability of landing on sector G is 2/15
  prob_G = 2/15 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_probability_l2020_202030


namespace NUMINAMATH_CALUDE_decimal_to_binary_conversion_l2020_202098

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- The decimal number to be converted -/
def decimal_number : ℕ := 2016

/-- The expected binary representation -/
def expected_binary : List Bool := [true, true, true, true, true, false, false, false, false, false, false]

theorem decimal_to_binary_conversion :
  to_binary decimal_number = expected_binary := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_conversion_l2020_202098


namespace NUMINAMATH_CALUDE_tan_sqrt3_sin_equality_l2020_202094

theorem tan_sqrt3_sin_equality : (Real.tan (10 * π / 180) - Real.sqrt 3) * Real.sin (40 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sqrt3_sin_equality_l2020_202094


namespace NUMINAMATH_CALUDE_right_triangle_area_l2020_202032

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- The angle at C is a right angle -/
  right_angle_at_C : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  /-- The length of hypotenuse AB is 50 -/
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 50^2
  /-- The median through A lies along the line y = x - 2 -/
  median_A : ∃ (t : ℝ), A.2 = A.1 - 2 ∧ ((B.1 + C.1) / 2 = A.1 + t) ∧ ((B.2 + C.2) / 2 = A.2 + t)
  /-- The median through B lies along the line y = 3x + 1 -/
  median_B : ∃ (t : ℝ), B.2 = 3 * B.1 + 1 ∧ ((A.1 + C.1) / 2 = B.1 + t) ∧ ((A.2 + C.2) / 2 = B.2 + 3 * t)

/-- The area of a right triangle ABC with the given properties is 3750/59 -/
theorem right_triangle_area (t : RightTriangle) : 
  abs ((t.A.1 - t.C.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.C.2)) / 2 = 3750 / 59 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2020_202032


namespace NUMINAMATH_CALUDE_min_nSn_is_neg_nine_l2020_202020

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_correct : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The theorem statement -/
theorem min_nSn_is_neg_nine (seq : ArithmeticSequence) (m : ℕ) (h_m : m ≥ 2) 
    (h_Sm_minus_one : seq.S (m - 1) = -2)
    (h_Sm : seq.S m = 0)
    (h_Sm_plus_one : seq.S (m + 1) = 3) :
    (∃ n : ℕ, seq.S n * n = -9) ∧ (∀ n : ℕ, seq.S n * n ≥ -9) := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_is_neg_nine_l2020_202020


namespace NUMINAMATH_CALUDE_smallest_equal_prob_sum_l2020_202072

/-- The number of faces on a standard die -/
def faces : ℕ := 6

/-- The target sum we're comparing to -/
def target_sum : ℕ := 2001

/-- The smallest number of dice needed to potentially reach the target sum -/
def min_dice : ℕ := (target_sum + faces - 1) / faces

/-- The function that transforms a die roll -/
def transform (x : ℕ) : ℕ := faces + 1 - x

/-- The smallest value S with equal probability to the target sum -/
def smallest_S : ℕ := (faces + 1) * min_dice - target_sum

theorem smallest_equal_prob_sum :
  smallest_S = 337 :=
sorry

end NUMINAMATH_CALUDE_smallest_equal_prob_sum_l2020_202072


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2020_202054

theorem system_of_equations_solution (x y z : ℝ) : 
  x + 3*y = 4*y^3 ∧ 
  y + 3*z = 4*z^3 ∧ 
  z + 3*x = 4*x^3 →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) ∨
  (x = Real.cos (π/14) ∧ y = -Real.cos (5*π/14) ∧ z = Real.cos (3*π/14)) ∨
  (x = -Real.cos (π/14) ∧ y = Real.cos (5*π/14) ∧ z = -Real.cos (3*π/14)) ∨
  (x = Real.cos (π/7) ∧ y = -Real.cos (2*π/7) ∧ z = Real.cos (3*π/7)) ∨
  (x = -Real.cos (π/7) ∧ y = Real.cos (2*π/7) ∧ z = -Real.cos (3*π/7)) ∨
  (x = Real.cos (π/13) ∧ y = -Real.cos (π/13) ∧ z = Real.cos (3*π/13)) ∨
  (x = -Real.cos (π/13) ∧ y = Real.cos (π/13) ∧ z = -Real.cos (3*π/13)) :=
by sorry


end NUMINAMATH_CALUDE_system_of_equations_solution_l2020_202054


namespace NUMINAMATH_CALUDE_quiz_probabilities_l2020_202095

/-- Represents the quiz with multiple-choice and true/false questions -/
structure Quiz where
  total_questions : ℕ
  multiple_choice : ℕ
  true_false : ℕ

/-- Calculates the probability of A drawing a multiple-choice question and B drawing a true/false question -/
def prob_a_multiple_b_true_false (q : Quiz) : ℚ :=
  (q.multiple_choice * q.true_false) / (q.total_questions * (q.total_questions - 1))

/-- Calculates the probability of at least one of A or B drawing a multiple-choice question -/
def prob_at_least_one_multiple (q : Quiz) : ℚ :=
  1 - (q.true_false * (q.true_false - 1)) / (q.total_questions * (q.total_questions - 1))

theorem quiz_probabilities (q : Quiz) 
  (h1 : q.total_questions = 10)
  (h2 : q.multiple_choice = 6)
  (h3 : q.true_false = 4) :
  prob_a_multiple_b_true_false q = 4 / 15 ∧ 
  prob_at_least_one_multiple q = 13 / 15 := by
  sorry


end NUMINAMATH_CALUDE_quiz_probabilities_l2020_202095


namespace NUMINAMATH_CALUDE_graph_behavior_l2020_202008

def g (x : ℝ) := x^2 - 2*x - 8

theorem graph_behavior (x : ℝ) :
  (∃ M : ℝ, ∀ x > M, g x > g M) ∧
  (∃ N : ℝ, ∀ x < N, g x > g N) :=
sorry

end NUMINAMATH_CALUDE_graph_behavior_l2020_202008


namespace NUMINAMATH_CALUDE_min_distance_sum_l2020_202043

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define the theorem
theorem min_distance_sum (M N P : ℝ × ℝ) :
  C₁ M.1 M.2 →
  C₂ N.1 N.2 →
  P.2 = 0 →
  ∃ (M' N' P' : ℝ × ℝ),
    C₁ M'.1 M'.2 ∧
    C₂ N'.1 N'.2 ∧
    P'.2 = 0 ∧
    Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) +
    Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) ≥
    Real.sqrt ((M'.1 - P'.1)^2 + (M'.2 - P'.2)^2) +
    Real.sqrt ((N'.1 - P'.1)^2 + (N'.2 - P'.2)^2) ∧
    Real.sqrt ((M'.1 - P'.1)^2 + (M'.2 - P'.2)^2) +
    Real.sqrt ((N'.1 - P'.1)^2 + (N'.2 - P'.2)^2) =
    5 * Real.sqrt 2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l2020_202043


namespace NUMINAMATH_CALUDE_solve_y_l2020_202041

theorem solve_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 14) : y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_y_l2020_202041


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2020_202074

/-- Given a complex number z = (3ai)/(1-2i) where a < 0 and i is the imaginary unit,
    if |z| = √5, then a = -5/3 -/
theorem complex_number_modulus (a : ℝ) (h1 : a < 0) :
  let z : ℂ := (3 * a * Complex.I) / (1 - 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 → a = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2020_202074


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2020_202093

theorem geometric_series_sum : 
  let a : ℚ := 1/5
  let r : ℚ := -1/3
  let n : ℕ := 7
  let series_sum := a * (1 - r^n) / (1 - r)
  series_sum = 1641/10935 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2020_202093


namespace NUMINAMATH_CALUDE_three_digit_remainder_problem_l2020_202018

theorem three_digit_remainder_problem :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 3 ∧ n % 8 = 6 ∧ n % 12 = 8) ∧
    (∀ n, 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 3 ∧ n % 8 = 6 ∧ n % 12 = 8 → n ∈ s) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_remainder_problem_l2020_202018


namespace NUMINAMATH_CALUDE_parallelogram_external_bisectors_rectangle_diagonal_l2020_202000

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is a rectangle -/
def isRectangle (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the length of a line segment between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Represents the intersection points of external angle bisectors -/
structure ExternalBisectorPoints :=
  (P Q R S : Point)

/-- Checks if given points are formed by intersection of external angle bisectors -/
def areExternalBisectorPoints (q : Quadrilateral) (e : ExternalBisectorPoints) : Prop :=
  sorry

/-- Main theorem -/
theorem parallelogram_external_bisectors_rectangle_diagonal
  (ABCD : Quadrilateral)
  (PQRS : ExternalBisectorPoints) :
  isParallelogram ABCD →
  areExternalBisectorPoints ABCD PQRS →
  isRectangle ⟨PQRS.P, PQRS.Q, PQRS.R, PQRS.S⟩ →
  distance PQRS.P PQRS.R = distance ABCD.A ABCD.B + distance ABCD.B ABCD.C :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_external_bisectors_rectangle_diagonal_l2020_202000


namespace NUMINAMATH_CALUDE_ines_initial_amount_l2020_202061

/-- The amount of money Ines had in her purse initially -/
def initial_amount : ℕ := 20

/-- The number of pounds of peaches Ines bought -/
def peaches_bought : ℕ := 3

/-- The cost per pound of peaches -/
def cost_per_pound : ℕ := 2

/-- The amount of money Ines had left after buying peaches -/
def amount_left : ℕ := 14

/-- Theorem stating that Ines had $20 in her purse initially -/
theorem ines_initial_amount :
  initial_amount = peaches_bought * cost_per_pound + amount_left :=
by sorry

end NUMINAMATH_CALUDE_ines_initial_amount_l2020_202061


namespace NUMINAMATH_CALUDE_cosine_equality_proof_l2020_202058

theorem cosine_equality_proof : ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 360 ∧ Real.cos (n * π / 180) = Real.cos (1234 * π / 180) ∧ n = 154 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_proof_l2020_202058


namespace NUMINAMATH_CALUDE_abs_one_minus_i_l2020_202067

theorem abs_one_minus_i : Complex.abs (1 - Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_one_minus_i_l2020_202067


namespace NUMINAMATH_CALUDE_intersection_area_l2020_202037

/-- A regular cube with side length 2 units -/
structure Cube where
  side_length : ℝ
  is_regular : side_length = 2

/-- A plane that cuts the cube -/
structure IntersectingPlane where
  parallel_to_face : Bool
  at_middle : Bool

/-- The polygon formed by the intersection of the plane and the cube -/
def intersection_polygon (c : Cube) (p : IntersectingPlane) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem intersection_area (c : Cube) (p : IntersectingPlane) :
  p.parallel_to_face ∧ p.at_middle →
  area (intersection_polygon c p) = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_area_l2020_202037


namespace NUMINAMATH_CALUDE_system_implies_sum_l2020_202055

theorem system_implies_sum (x y m : ℝ) : x + m = 4 → y - 5 = m → x + y = 9 := by sorry

end NUMINAMATH_CALUDE_system_implies_sum_l2020_202055


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l2020_202012

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hyperbola xy = 1 -/
def hyperbola (p : Point) : Prop := p.x * p.y = 1

/-- Four points lie on the same circle -/
def on_same_circle (p1 p2 p3 p4 : Point) : Prop := 
  ∃ (h k s : ℝ), 
    (p1.x - h)^2 + (p1.y - k)^2 = s^2 ∧
    (p2.x - h)^2 + (p2.y - k)^2 = s^2 ∧
    (p3.x - h)^2 + (p3.y - k)^2 = s^2 ∧
    (p4.x - h)^2 + (p4.y - k)^2 = s^2

theorem fourth_intersection_point : 
  let p1 : Point := ⟨3, 1/3⟩
  let p2 : Point := ⟨-4, -1/4⟩
  let p3 : Point := ⟨1/6, 6⟩
  let p4 : Point := ⟨-1/2, -2⟩
  hyperbola p1 ∧ hyperbola p2 ∧ hyperbola p3 ∧ hyperbola p4 ∧
  on_same_circle p1 p2 p3 p4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l2020_202012


namespace NUMINAMATH_CALUDE_distance_not_half_radius_l2020_202080

/-- Two circles with radii p and p/2, whose centers are a non-zero distance d apart -/
structure TwoCircles (p : ℝ) where
  d : ℝ
  d_pos : d > 0

/-- Theorem: The distance between the centers cannot be p/2 -/
theorem distance_not_half_radius (p : ℝ) (circles : TwoCircles p) :
  circles.d ≠ p / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_not_half_radius_l2020_202080


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l2020_202040

theorem circle_diameter_ratio (R S : Real) (harea : R^2 = 0.64 * S^2) : 
  R = 0.8 * S := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l2020_202040


namespace NUMINAMATH_CALUDE_subtraction_of_large_numbers_l2020_202096

theorem subtraction_of_large_numbers :
  10000000000000 - (5555555555555 * 2) = -1111111111110 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_large_numbers_l2020_202096


namespace NUMINAMATH_CALUDE_george_christopher_age_difference_l2020_202047

theorem george_christopher_age_difference :
  ∀ (G C F : ℕ),
    C = 18 →
    F = C - 2 →
    G + C + F = 60 →
    G > C →
    G - C = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_george_christopher_age_difference_l2020_202047


namespace NUMINAMATH_CALUDE_number_exchange_ratio_l2020_202070

theorem number_exchange_ratio (a b p q : ℝ) (h : p * q ≠ 1) :
  ∃ z : ℝ, (z + a - a) + ((p * z - a) + a) = q * ((z + a + b) - ((p * z - a) - b)) →
  z = (a + b) * (q + 1) / (p * q - 1) :=
by sorry

end NUMINAMATH_CALUDE_number_exchange_ratio_l2020_202070


namespace NUMINAMATH_CALUDE_tangent_line_to_quartic_curve_l2020_202045

/-- Given that y = 4x + b is a tangent line to y = x^4 - 1, prove that b = -4 -/
theorem tangent_line_to_quartic_curve (b : ℝ) : 
  (∃ x₀ : ℝ, (4 * x₀ + b = x₀^4 - 1) ∧ 
             (∀ x : ℝ, 4 * x + b ≥ x^4 - 1) ∧ 
             (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| ∧ |x - x₀| < δ → 4 * x + b > x^4 - 1)) → 
  b = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_quartic_curve_l2020_202045


namespace NUMINAMATH_CALUDE_mike_books_equal_sum_l2020_202048

/-- The number of books Bobby has -/
def bobby_books : Nat := 142

/-- The number of books Kristi has -/
def kristi_books : Nat := 78

/-- The number of books Mike needs to have -/
def mike_books : Nat := bobby_books + kristi_books

theorem mike_books_equal_sum :
  mike_books = bobby_books + kristi_books := by
  sorry

end NUMINAMATH_CALUDE_mike_books_equal_sum_l2020_202048


namespace NUMINAMATH_CALUDE_tomato_difference_l2020_202026

theorem tomato_difference (initial_tomatoes picked_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 17)
  (h2 : picked_tomatoes = 9) :
  initial_tomatoes - picked_tomatoes = 8 := by
  sorry

end NUMINAMATH_CALUDE_tomato_difference_l2020_202026


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2020_202033

/-- Given a line with slope -3 passing through the point (2, 4), prove that m + b = 7 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -3 ∧ 4 = m * 2 + b → m + b = 7 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2020_202033


namespace NUMINAMATH_CALUDE_largest_number_l2020_202078

theorem largest_number : Real.sqrt 2 = max (max (max (-3) 0) 1) (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2020_202078


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2020_202063

/-- The number of terms between 400 and 600 in an arithmetic sequence -/
theorem arithmetic_sequence_terms (a₁ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 110 →
  d = 6 →
  (∃ k₁ k₂ : ℕ, 
    a₁ + (k₁ - 1) * d ≥ 400 ∧
    a₁ + (k₁ - 1) * d < a₁ + k₁ * d ∧
    a₁ + (k₂ - 1) * d ≤ 600 ∧
    a₁ + k₂ * d > 600 ∧
    k₂ - k₁ + 1 = 33) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2020_202063


namespace NUMINAMATH_CALUDE_difference_exists_l2020_202002

def sequence_property (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ ∀ n, x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n

theorem difference_exists (x : ℕ → ℕ) (h : sequence_property x) :
  ∀ k : ℕ, k > 0 → ∃ r s, x r - x s = k := by
  sorry

end NUMINAMATH_CALUDE_difference_exists_l2020_202002


namespace NUMINAMATH_CALUDE_trajectory_of_point_l2020_202044

/-- The trajectory of a point M, given specific conditions -/
theorem trajectory_of_point (M : ℝ × ℝ) :
  (∀ (x y : ℝ), M = (x, y) →
    (x^2 + (y + 3)^2)^(1/2) = |y - 3|) →  -- M is equidistant from (0, -3) and y = 3
  (∃ (a b c : ℝ), ∀ (x y : ℝ), M = (x, y) → 
    a*x^2 + b*y + c = 0) →  -- Trajectory of M is a conic section (which includes parabolas)
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 = -12*y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_l2020_202044


namespace NUMINAMATH_CALUDE_triple_sum_diverges_l2020_202053

open Real BigOperators

theorem triple_sum_diverges :
  let f (m n k : ℕ) := (1 : ℝ) / (m * (m + n + k) * (n + 1))
  ∃ (S : ℝ), ∀ (M N K : ℕ), (∑ m in Finset.range M, ∑ n in Finset.range N, ∑ k in Finset.range K, f m n k) ≤ S
  → false :=
sorry

end NUMINAMATH_CALUDE_triple_sum_diverges_l2020_202053


namespace NUMINAMATH_CALUDE_mistaken_calculation_l2020_202071

theorem mistaken_calculation (x : ℕ) : 423 - x = 421 → (423 * x) + (423 - x) = 1267 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l2020_202071


namespace NUMINAMATH_CALUDE_correct_probability_l2020_202076

def first_three_digits : ℕ := 3
def last_four_digits : ℕ := 24
def total_combinations : ℕ := first_three_digits * last_four_digits
def correct_combinations : ℕ := 1

theorem correct_probability : 
  (correct_combinations : ℚ) / total_combinations = 1 / 72 := by sorry

end NUMINAMATH_CALUDE_correct_probability_l2020_202076


namespace NUMINAMATH_CALUDE_first_graders_count_l2020_202034

/-- The number of Kindergartners -/
def kindergartners : ℕ := 101

/-- The cost of an orange shirt for Kindergartners -/
def orange_shirt_cost : ℚ := 29/5

/-- The cost of a yellow shirt for first graders -/
def yellow_shirt_cost : ℚ := 5

/-- The number of second graders -/
def second_graders : ℕ := 107

/-- The cost of a blue shirt for second graders -/
def blue_shirt_cost : ℚ := 28/5

/-- The number of third graders -/
def third_graders : ℕ := 108

/-- The cost of a green shirt for third graders -/
def green_shirt_cost : ℚ := 21/4

/-- The total amount spent by the P.T.O. -/
def total_spent : ℚ := 2317

/-- The number of first graders wearing yellow shirts -/
def first_graders : ℕ := 113

theorem first_graders_count : 
  first_graders * yellow_shirt_cost + 
  kindergartners * orange_shirt_cost + 
  second_graders * blue_shirt_cost + 
  third_graders * green_shirt_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_first_graders_count_l2020_202034


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2020_202066

theorem arithmetic_sequence_problem (a b c : ℤ) :
  (∃ d : ℤ, -1 = a - d ∧ a = b - d ∧ b = c - d ∧ c = -9 + d) →
  b = -5 ∧ a * c = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2020_202066


namespace NUMINAMATH_CALUDE_triangle_on_parabola_ef_length_l2020_202050

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

/-- Triangle DEF -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The theorem to be proved -/
theorem triangle_on_parabola_ef_length (t : Triangle) :
  t.D = vertex ∧
  (∀ x, (x, parabola x) = t.D ∨ (x, parabola x) = t.E ∨ (x, parabola x) = t.F) ∧
  t.E.2 = t.F.2 ∧
  (1/2 * (t.F.1 - t.E.1) * (t.E.2 - t.D.2) = 32) →
  t.F.1 - t.E.1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_on_parabola_ef_length_l2020_202050


namespace NUMINAMATH_CALUDE_star_calculation_l2020_202090

-- Define the ⋆ operation
def star (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem star_calculation : star (star 2 1) 4 = 259 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l2020_202090


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l2020_202057

theorem min_value_sum_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + 1) / c + (a + c + 1) / b + (b + c + 1) / a ≥ 9 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    (a₀ + b₀ + 1) / c₀ + (a₀ + c₀ + 1) / b₀ + (b₀ + c₀ + 1) / a₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l2020_202057


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2020_202042

theorem quadratic_solution_sum (a b : ℚ) : 
  (∃ x : ℂ, x = a + b * I ∧ 5 * x^2 - 2 * x + 17 = 0) →
  a + b^2 = 89/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2020_202042


namespace NUMINAMATH_CALUDE_store_discount_l2020_202085

theorem store_discount (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_discount := 0.4
  let second_discount := 0.1
  let claimed_discount := 0.5
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  let actual_discount := 1 - (final_price / original_price)
  actual_discount = 0.46 ∧ claimed_discount - actual_discount = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_l2020_202085


namespace NUMINAMATH_CALUDE_distance_origin_to_point_l2020_202021

/-- The distance between the origin (0,0) and the point (1, √3) in a Cartesian coordinate system is 2. -/
theorem distance_origin_to_point :
  let A : ℝ × ℝ := (1, Real.sqrt 3)
  Real.sqrt ((A.1 - 0)^2 + (A.2 - 0)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_origin_to_point_l2020_202021


namespace NUMINAMATH_CALUDE_pentagon_cannot_tile_floor_l2020_202088

-- Define a function to calculate the interior angle of a regular polygon
def interior_angle (n : ℕ) : ℚ :=
  180 - 360 / n

-- Define a function to check if an angle can divide 360° evenly
def divides_360 (angle : ℚ) : Prop :=
  ∃ k : ℕ, k * angle = 360

-- Theorem statement
theorem pentagon_cannot_tile_floor :
  divides_360 (interior_angle 6) ∧
  divides_360 90 ∧
  divides_360 60 ∧
  ¬ divides_360 (interior_angle 5) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tile_floor_l2020_202088


namespace NUMINAMATH_CALUDE_hcf_of_48_and_99_l2020_202059

theorem hcf_of_48_and_99 : Nat.gcd 48 99 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_48_and_99_l2020_202059


namespace NUMINAMATH_CALUDE_parallelepiped_height_l2020_202011

/-- The surface area of a rectangular parallelepiped -/
def surface_area (l w h : ℝ) : ℝ := 2*l*w + 2*l*h + 2*w*h

/-- Theorem: The height of a rectangular parallelepiped with given dimensions -/
theorem parallelepiped_height (w l : ℝ) (h : ℝ) :
  w = 7 → l = 8 → surface_area l w h = 442 → h = 11 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_height_l2020_202011


namespace NUMINAMATH_CALUDE_john_good_games_l2020_202081

/-- The number of good games John ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (broken_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - broken_games

/-- Theorem stating that John ended up with 6 good games -/
theorem john_good_games :
  good_games 21 8 23 = 6 := by
  sorry

end NUMINAMATH_CALUDE_john_good_games_l2020_202081


namespace NUMINAMATH_CALUDE_javier_has_four_children_l2020_202007

/-- The number of children Javier has -/
def num_children : ℕ :=
  let total_legs : ℕ := 22
  let num_dogs : ℕ := 2
  let num_cats : ℕ := 1
  let javier_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let cat_legs : ℕ := 4
  (total_legs - (num_dogs * dog_legs + num_cats * cat_legs + javier_legs)) / 2

theorem javier_has_four_children : num_children = 4 := by
  sorry

end NUMINAMATH_CALUDE_javier_has_four_children_l2020_202007
