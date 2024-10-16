import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l2557_255792

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_four_terms
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_5th : a 5 = 11)
  (h_6th : a 6 = 17)
  (h_7th : a 7 = 23) :
  a 1 + a 2 + a 3 + a 4 = -16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l2557_255792


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2557_255786

theorem subset_implies_m_equals_one (m : ℝ) : 
  let A : Set ℝ := {3, m^2}
  let B : Set ℝ := {-1, 3, 2*m - 1}
  A ⊆ B → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2557_255786


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2557_255793

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2557_255793


namespace NUMINAMATH_CALUDE_inverse_equals_original_at_three_l2557_255799

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 9

-- Define the property of being an inverse function
def is_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

-- Theorem statement
theorem inverse_equals_original_at_three :
  ∃ g_inv : ℝ → ℝ, is_inverse g g_inv ∧
  ∀ x : ℝ, g x = g_inv x ↔ x = 3 :=
sorry

end NUMINAMATH_CALUDE_inverse_equals_original_at_three_l2557_255799


namespace NUMINAMATH_CALUDE_appliance_price_ratio_l2557_255753

theorem appliance_price_ratio : 
  ∀ (c p q : ℝ), 
  p = 0.8 * c →  -- 20% loss
  q = 1.25 * c → -- 25% profit
  q / p = 25 / 16 := by
sorry

end NUMINAMATH_CALUDE_appliance_price_ratio_l2557_255753


namespace NUMINAMATH_CALUDE_even_odd_periodic_properties_l2557_255797

-- Define the properties of even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of periodic functions
def IsPeriodic (f : ℝ → ℝ) : Prop := ∃ t : ℝ, t ≠ 0 ∧ ∀ x, f (x + t) = f x

-- State the theorem
theorem even_odd_periodic_properties 
  (f g : ℝ → ℝ) 
  (hf_even : IsEven f) 
  (hg_odd : IsOdd g) 
  (hf_periodic : IsPeriodic f) 
  (hg_periodic : IsPeriodic g) : 
  IsOdd (λ x ↦ g (g x)) ∧ IsPeriodic (λ x ↦ f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_even_odd_periodic_properties_l2557_255797


namespace NUMINAMATH_CALUDE_base_equation_solution_l2557_255735

/-- Represents a number in a given base -/
def toBase (n : ℕ) (base : ℕ) : ℕ → ℕ 
| 0 => 0
| (d+1) => (toBase n base d) * base + n % base

/-- The main theorem -/
theorem base_equation_solution (A B : ℕ) (h1 : B = A + 2) 
  (h2 : toBase 216 A 3 + toBase 52 B 2 = toBase 75 (A + B + 1) 2) : 
  A + B + 1 = 15 := by
  sorry

#eval toBase 216 6 3  -- Should output 90
#eval toBase 52 8 2   -- Should output 42
#eval toBase 75 15 2  -- Should output 132

end NUMINAMATH_CALUDE_base_equation_solution_l2557_255735


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_div_by_three_l2557_255781

theorem n_times_n_plus_one_div_by_three (n : ℤ) (h : 1 ≤ n ∧ n ≤ 99) :
  ∃ k : ℤ, n * (n + 1) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_div_by_three_l2557_255781


namespace NUMINAMATH_CALUDE_pattern_perimeter_is_24_l2557_255774

/-- A pattern formed by squares, triangles, and a hexagon -/
structure Pattern where
  num_squares : ℕ
  num_triangles : ℕ
  square_side_length : ℝ
  triangle_perimeter_contribution : ℕ
  square_perimeter_contribution : ℕ

/-- Calculate the perimeter of the pattern -/
def pattern_perimeter (p : Pattern) : ℝ :=
  (p.num_triangles * p.triangle_perimeter_contribution +
   p.num_squares * p.square_perimeter_contribution) * p.square_side_length

/-- The specific pattern described in the problem -/
def specific_pattern : Pattern := {
  num_squares := 6,
  num_triangles := 6,
  square_side_length := 2,
  triangle_perimeter_contribution := 2,
  square_perimeter_contribution := 2
}

theorem pattern_perimeter_is_24 :
  pattern_perimeter specific_pattern = 24 := by
  sorry

end NUMINAMATH_CALUDE_pattern_perimeter_is_24_l2557_255774


namespace NUMINAMATH_CALUDE_sons_age_is_correct_l2557_255704

/-- The age of the son -/
def sons_age : ℕ := 23

/-- The age of the father -/
def fathers_age : ℕ := sons_age + 25

theorem sons_age_is_correct : 
  (fathers_age + 2 = 2 * (sons_age + 2)) ∧ 
  (fathers_age = sons_age + 25) ∧ 
  (sons_age = 23) := by
  sorry

end NUMINAMATH_CALUDE_sons_age_is_correct_l2557_255704


namespace NUMINAMATH_CALUDE_symmetric_line_l2557_255782

/-- Given a line L1 with equation 2x-y+3=0 and a point M(-1,2),
    prove that the line L2 symmetric to L1 with respect to M
    has the equation 2x-y+5=0 -/
theorem symmetric_line (x y : ℝ) :
  (2 * x - y + 3 = 0) →
  (2 * (-2 - x) - (4 - y) + 3 = 0) →
  (2 * x - y + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2557_255782


namespace NUMINAMATH_CALUDE_matches_for_15_squares_bottom_l2557_255716

/-- The number of matches needed for the nth layer -/
def matches_for_layer (n : ℕ) : ℕ := 4 + 9 * (n - 1)

/-- The total number of matches needed up to the nth layer -/
def total_matches (n : ℕ) : ℕ := 
  if n = 0 then 0 else (List.range n).map matches_for_layer |>.sum

/-- The number of squares in the bottom layer for the nth structure -/
def squares_in_bottom_layer (n : ℕ) : ℕ := n * (n + 1) / 2

theorem matches_for_15_squares_bottom : 
  ∃ n : ℕ, squares_in_bottom_layer n = 15 ∧ total_matches n = 151 :=
sorry

end NUMINAMATH_CALUDE_matches_for_15_squares_bottom_l2557_255716


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2557_255719

def M : Set ℝ := {x | 1 < x ∧ x < 4}

theorem sqrt_inequality (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  |Real.sqrt (a * b) - 2| < |2 * Real.sqrt a - Real.sqrt b| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2557_255719


namespace NUMINAMATH_CALUDE_range_of_f_set_where_g_less_than_f_l2557_255794

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + x
def g (x : ℝ) : ℝ := |x + 1|

-- Statement for the range of f
theorem range_of_f : Set.range f = Set.Ici 2 := by sorry

-- Statement for the set where g(x) < f(x)
theorem set_where_g_less_than_f : 
  {x : ℝ | g x < f x} = Set.union (Set.Ioo (-3) 1) (Set.Ioi 3) := by sorry

end NUMINAMATH_CALUDE_range_of_f_set_where_g_less_than_f_l2557_255794


namespace NUMINAMATH_CALUDE_decimal_calculation_l2557_255771

theorem decimal_calculation : (0.25 * 0.8) - 0.12 = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_decimal_calculation_l2557_255771


namespace NUMINAMATH_CALUDE_donation_theorem_l2557_255784

def donation_problem (total_donation : ℚ) 
  (community_pantry_fraction : ℚ) 
  (crisis_fund_fraction : ℚ) 
  (contingency_amount : ℚ) : Prop :=
  let remaining := total_donation - (community_pantry_fraction * total_donation) - (crisis_fund_fraction * total_donation)
  let livelihood_amount := remaining - contingency_amount
  livelihood_amount / remaining = 1 / 4

theorem donation_theorem : 
  donation_problem 240 (1/3) (1/2) 30 := by
  sorry

end NUMINAMATH_CALUDE_donation_theorem_l2557_255784


namespace NUMINAMATH_CALUDE_negative_three_is_square_mod_p_l2557_255710

theorem negative_three_is_square_mod_p (p q : ℕ) (h_prime : Nat.Prime p) (h_form : p = 3 * q + 1) :
  ∃ x : ZMod p, x^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_is_square_mod_p_l2557_255710


namespace NUMINAMATH_CALUDE_girls_in_first_grade_l2557_255702

/-- Represents the first grade class configuration -/
structure FirstGrade where
  classrooms : ℕ
  boys : ℕ
  students_per_classroom : ℕ

/-- Calculates the number of girls in the first grade -/
def girls_count (fg : FirstGrade) : ℕ :=
  fg.classrooms * fg.students_per_classroom - fg.boys

/-- Theorem stating the number of girls in the first grade -/
theorem girls_in_first_grade (fg : FirstGrade) 
  (h1 : fg.classrooms = 4)
  (h2 : fg.boys = 56)
  (h3 : fg.students_per_classroom = 25)
  (h4 : ∀ c, c ≤ fg.classrooms → fg.boys / fg.classrooms = (girls_count fg) / fg.classrooms) :
  girls_count fg = 44 := by
  sorry

#eval girls_count ⟨4, 56, 25⟩

end NUMINAMATH_CALUDE_girls_in_first_grade_l2557_255702


namespace NUMINAMATH_CALUDE_tire_promotion_price_l2557_255754

/-- The regular price of a tire under the given promotion -/
def regular_price : ℝ := 105

/-- The total cost of five tires under the promotion -/
def total_cost : ℝ := 421

/-- The promotion: buy four tires at regular price, get the fifth for $1 -/
theorem tire_promotion_price : 
  4 * regular_price + 1 = total_cost := by sorry

end NUMINAMATH_CALUDE_tire_promotion_price_l2557_255754


namespace NUMINAMATH_CALUDE_intersection_of_solutions_range_of_a_l2557_255778

-- Define the conditions
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 ≥ 0

-- Part 1: Intersection of solutions when a = 1
theorem intersection_of_solutions :
  {x : ℝ | p x 1 ∧ q x} = {x : ℝ | 2 ≤ x ∧ x < 3} :=
sorry

-- Part 2: Range of a for which ¬p ↔ ¬q
theorem range_of_a :
  {a : ℝ | a > 0 ∧ ∀ x, ¬(p x a) ↔ ¬(q x)} = {a : ℝ | 1 < a ∧ a < 2} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_solutions_range_of_a_l2557_255778


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l2557_255744

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 24) (h2 : b = 18) : 
  (a + b + x = 87) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l2557_255744


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l2557_255726

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l2557_255726


namespace NUMINAMATH_CALUDE_problem_solution_l2557_255700

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 1) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2557_255700


namespace NUMINAMATH_CALUDE_second_to_last_digit_even_for_valid_numbers_l2557_255756

def ends_in_valid_digit (k : ℕ) : Prop :=
  k % 10 = 1 ∨ k % 10 = 3 ∨ k % 10 = 7 ∨ k % 10 = 9 ∨ k % 10 = 5 ∨ k % 10 = 0

def second_to_last_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem second_to_last_digit_even_for_valid_numbers (k n : ℕ) 
  (h : ends_in_valid_digit k) : 
  Even (second_to_last_digit (k^n)) :=
sorry

end NUMINAMATH_CALUDE_second_to_last_digit_even_for_valid_numbers_l2557_255756


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2557_255707

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 9 ∧ (x - y)^2 + (y - z)^2 + (z - x)^2 > (a - b)^2 + (b - c)^2 + (c - a)^2) →
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2557_255707


namespace NUMINAMATH_CALUDE_large_pots_delivered_l2557_255748

/-- The number of boxes delivered -/
def num_boxes : ℕ := 32

/-- The number of small pots in each box -/
def small_pots_per_box : ℕ := 36

/-- The number of large pots in each box -/
def large_pots_per_box : ℕ := 12

/-- The total number of large pots delivered -/
def total_large_pots : ℕ := num_boxes * large_pots_per_box

/-- The number of boxes used for comparison -/
def comparison_boxes : ℕ := 8

theorem large_pots_delivered :
  total_large_pots = 384 ∧
  total_large_pots = comparison_boxes * (small_pots_per_box + large_pots_per_box) :=
by sorry

end NUMINAMATH_CALUDE_large_pots_delivered_l2557_255748


namespace NUMINAMATH_CALUDE_fgh_supermarket_count_l2557_255725

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 41

/-- The difference between the number of US and Canadian supermarkets -/
def difference : ℕ := 22

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - difference

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := us_supermarkets + canada_supermarkets

theorem fgh_supermarket_count : total_supermarkets = 60 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_count_l2557_255725


namespace NUMINAMATH_CALUDE_xy_value_l2557_255727

theorem xy_value (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 + 6*y + 9 = 0) : x*y = 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2557_255727


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l2557_255798

/-- Represents a stratified sample from a population -/
structure StratifiedSample where
  total_population : ℕ
  male_population : ℕ
  female_population : ℕ
  female_sample : ℕ
  male_sample : ℕ

/-- Checks if a stratified sample is valid according to the stratified sampling principle -/
def is_valid_stratified_sample (s : StratifiedSample) : Prop :=
  s.female_population * s.male_sample = s.male_population * s.female_sample

theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_population = 680)
  (h2 : s.male_population = 360)
  (h3 : s.female_population = 320)
  (h4 : s.female_sample = 16)
  (h5 : is_valid_stratified_sample s) :
  s.male_sample = 18 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l2557_255798


namespace NUMINAMATH_CALUDE_price_decrease_proof_l2557_255779

theorem price_decrease_proof (original_price : ℝ) (decrease_percentage : ℝ) (new_price : ℝ) :
  decrease_percentage = 24 →
  new_price = 421.05263157894734 →
  new_price = original_price * (1 - decrease_percentage / 100) :=
by
  sorry

#eval 421.05263157894734 -- To show the exact value used in the problem

end NUMINAMATH_CALUDE_price_decrease_proof_l2557_255779


namespace NUMINAMATH_CALUDE_min_width_proof_l2557_255728

/-- The minimum width of a rectangular area with given constraints -/
def min_width : ℝ := 10

/-- The length of the rectangular area -/
def length (w : ℝ) : ℝ := w + 20

/-- The area of the rectangular area -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 150 → w ≥ min_width) ∧
  (area min_width ≥ 150) ∧
  (min_width > 0) :=
sorry

end NUMINAMATH_CALUDE_min_width_proof_l2557_255728


namespace NUMINAMATH_CALUDE_open_sets_l2557_255733

-- Define the concept of an open set in a plane
def is_open_set (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ A → 
    ∃ (r : ℝ), r > 0 ∧ {(x, y) | (x - x₀)^2 + (y - y₀)^2 < r^2} ⊆ A

-- Define the four sets
def set1 : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 1}
def set2 : Set (ℝ × ℝ) := {(x, y) | |x + y + 2| ≥ 1}
def set3 : Set (ℝ × ℝ) := {(x, y) | |x| + |y| < 1}
def set4 : Set (ℝ × ℝ) := {(x, y) | 0 < x^2 + (y - 1)^2 ∧ x^2 + (y - 1)^2 < 1}

-- State the theorem
theorem open_sets : 
  ¬(is_open_set set1) ∧ 
  ¬(is_open_set set2) ∧ 
  (is_open_set set3) ∧ 
  (is_open_set set4) := by
  sorry

end NUMINAMATH_CALUDE_open_sets_l2557_255733


namespace NUMINAMATH_CALUDE_beaus_age_is_42_l2557_255706

/-- Beau's age today given his triplet sons' ages and a condition from the past -/
def beaus_age_today (sons_age_today : ℕ) : ℕ :=
  let sons_age_past := sons_age_today - 3
  let beaus_age_past := 3 * sons_age_past
  beaus_age_past + 3

theorem beaus_age_is_42 :
  beaus_age_today 16 = 42 := by sorry

end NUMINAMATH_CALUDE_beaus_age_is_42_l2557_255706


namespace NUMINAMATH_CALUDE_perimeter_ratio_from_area_ratio_l2557_255787

theorem perimeter_ratio_from_area_ratio (s1 s2 : ℝ) (h : s1 ^ 2 / s2 ^ 2 = 49 / 64) :
  (4 * s1) / (4 * s2) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_from_area_ratio_l2557_255787


namespace NUMINAMATH_CALUDE_wolf_does_not_catch_hare_l2557_255759

/-- Represents the step length of the hare -/
def hare_step : ℝ := 1

/-- Represents the step length of the wolf -/
def wolf_step : ℝ := 2 * hare_step

/-- Represents the number of steps the hare takes in a time unit -/
def hare_frequency : ℕ := 3

/-- Represents the number of steps the wolf takes in a time unit -/
def wolf_frequency : ℕ := 1

/-- Theorem stating that the wolf will not catch the hare -/
theorem wolf_does_not_catch_hare : 
  (hare_step * hare_frequency) > (wolf_step * wolf_frequency) := by
  sorry


end NUMINAMATH_CALUDE_wolf_does_not_catch_hare_l2557_255759


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l2557_255752

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

theorem thirty_factorial_trailing_zeros :
  trailing_zeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l2557_255752


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_angle_l2557_255777

/-- The equation of a line passing through a given point with a given slope angle -/
theorem line_equation_through_point_with_slope_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) :
  x₀ = Real.sqrt 3 →
  y₀ = 1 →
  θ = π / 3 →
  ∃ (a b c : ℝ), 
    a * Real.sqrt 3 + b * 1 + c = 0 ∧
    a * x + b * y + c = 0 ∧
    a = Real.sqrt 3 ∧
    b = -1 ∧
    c = -2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_angle_l2557_255777


namespace NUMINAMATH_CALUDE_inequality_solution_l2557_255751

theorem inequality_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) < 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2557_255751


namespace NUMINAMATH_CALUDE_power_calculation_l2557_255783

theorem power_calculation : 3^15 * 9^5 / 27^6 = 3^7 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2557_255783


namespace NUMINAMATH_CALUDE_car_speed_proof_l2557_255732

/-- Proves that a car traveling at speed v km/h takes 20 seconds longer to travel 1 kilometer 
    than it would at 36 km/h if and only if v = 30 km/h. -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 36) * 3600 = 20 ↔ v = 30 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l2557_255732


namespace NUMINAMATH_CALUDE_f_monotone_iff_m_condition_possible_m_values_l2557_255796

/-- A function f is monotonically increasing on an interval [a, b] -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

/-- The function f(x) = x^2 - mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

/-- Theorem stating the conditions for f to be monotonically increasing on [3, 8] -/
theorem f_monotone_iff_m_condition (m : ℝ) :
  MonotonicallyIncreasing (f m) 3 8 ↔ m ≤ 6 ∨ m ≥ 16 := by
  sorry

/-- Corollary: The possible integer values of m for f to be monotonically increasing on [3, 8] -/
theorem possible_m_values :
  ∃ (S : Set ℤ), S = {0, 16, 20} ∧
    ∀ m : ℤ, m ∈ S ↔ MonotonicallyIncreasing (f (m : ℝ)) 3 8 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_iff_m_condition_possible_m_values_l2557_255796


namespace NUMINAMATH_CALUDE_x_value_when_y_is_half_l2557_255721

theorem x_value_when_y_is_half :
  ∀ x y : ℚ, y = 2 / (4 * x + 2) → y = 1 / 2 → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_half_l2557_255721


namespace NUMINAMATH_CALUDE_cubic_polynomial_with_coefficient_roots_l2557_255746

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The polynomial function for a CubicPolynomial -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℚ) : ℚ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Predicate for a CubicPolynomial having its coefficients as roots -/
def CubicPolynomial.hasCoefficientsAsRoots (p : CubicPolynomial) : Prop :=
  p.eval p.a = 0 ∧ p.eval p.b = 0 ∧ p.eval p.c = 0

/-- The two specific polynomials mentioned in the problem -/
def f₁ : CubicPolynomial := ⟨1, -2, 0⟩
def f₂ : CubicPolynomial := ⟨1, -1, -1⟩

/-- The main theorem stating that f₁ and f₂ are the only valid polynomials -/
theorem cubic_polynomial_with_coefficient_roots :
  ∀ p : CubicPolynomial, p.hasCoefficientsAsRoots → p = f₁ ∨ p = f₂ := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_with_coefficient_roots_l2557_255746


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l2557_255724

theorem probability_of_y_selection (p_x p_both : ℝ) (h1 : p_x = 1/7) 
  (h2 : p_both = 0.031746031746031744) : 
  ∃ p_y : ℝ, p_y = 0.2222222222222222 ∧ p_both = p_x * p_y :=
sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l2557_255724


namespace NUMINAMATH_CALUDE_stating_min_sides_for_rotation_l2557_255742

/-- The rotation angle in degrees -/
def rotation_angle : ℚ := 25 + 30 / 60

/-- The fraction of a full circle that the rotation represents -/
def rotation_fraction : ℚ := rotation_angle / 360

/-- The minimum number of sides for the polygons -/
def min_sides : ℕ := 240

/-- 
  Theorem stating that the minimum number of sides for two identical polygons
  that coincide when one is rotated by 25°30' is 240
-/
theorem min_sides_for_rotation :
  ∀ n : ℕ, 
    (n > 0 ∧ (rotation_fraction * n).den = 1) → 
    n ≥ min_sides :=
sorry

end NUMINAMATH_CALUDE_stating_min_sides_for_rotation_l2557_255742


namespace NUMINAMATH_CALUDE_candy_seller_problem_l2557_255701

/-- The number of candies the seller had initially, given the number of clowns,
    children, candies per person, and candies left after selling. -/
def initial_candies (clowns children candies_per_person candies_left : ℕ) : ℕ :=
  (clowns + children) * candies_per_person + candies_left

/-- Theorem stating that given the specific conditions in the problem,
    the initial number of candies is 700. -/
theorem candy_seller_problem :
  initial_candies 4 30 20 20 = 700 := by
  sorry

end NUMINAMATH_CALUDE_candy_seller_problem_l2557_255701


namespace NUMINAMATH_CALUDE_new_encoding_of_old_message_l2557_255722

/-- Represents the old encoding system --/
def OldEncoding : Type := String

/-- Represents the new encoding system --/
def NewEncoding : Type := String

/-- Decodes a message from the old encoding system --/
def decode (msg : OldEncoding) : String :=
  sorry

/-- Encodes a message using the new encoding system --/
def encode (msg : String) : NewEncoding :=
  sorry

/-- The new encoding rules --/
def newEncodingRules : List (Char × String) :=
  [('A', "21"), ('B', "122"), ('C', "1")]

/-- The theorem to be proved --/
theorem new_encoding_of_old_message :
  let oldMsg : OldEncoding := "011011010011"
  let decodedMsg := decode oldMsg
  encode decodedMsg = "211221121" :=
by sorry

end NUMINAMATH_CALUDE_new_encoding_of_old_message_l2557_255722


namespace NUMINAMATH_CALUDE_harry_hours_worked_l2557_255741

/-- Represents the payment structure and hours worked for an employee -/
structure Employee where
  baseHours : ℕ  -- Number of hours paid at base rate
  baseRate : ℝ   -- Base hourly rate
  overtimeRate : ℝ  -- Overtime hourly rate
  hoursWorked : ℕ  -- Total hours worked

/-- Calculates the total pay for an employee -/
def totalPay (e : Employee) : ℝ :=
  let baseAmount := min e.hoursWorked e.baseHours * e.baseRate
  let overtimeHours := max (e.hoursWorked - e.baseHours) 0
  baseAmount + overtimeHours * e.overtimeRate

/-- The main theorem to prove -/
theorem harry_hours_worked 
  (x : ℝ) 
  (harry : Employee) 
  (james : Employee) :
  harry.baseHours = 12 ∧ 
  harry.baseRate = x ∧ 
  harry.overtimeRate = 1.5 * x ∧
  james.baseHours = 40 ∧ 
  james.baseRate = x ∧ 
  james.overtimeRate = 2 * x ∧
  james.hoursWorked = 41 ∧
  totalPay harry = totalPay james →
  harry.hoursWorked = 32 := by
  sorry


end NUMINAMATH_CALUDE_harry_hours_worked_l2557_255741


namespace NUMINAMATH_CALUDE_min_value_theorem_l2557_255772

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  (∃ (x : ℝ), x = 1 / (2 * abs a) + abs a / b ∧
    (∀ (y : ℝ), y = 1 / (2 * abs a) + abs a / b → x ≤ y)) →
  (∃ (min_val : ℝ), min_val = 3/4 ∧
    (∃ (x : ℝ), x = 1 / (2 * abs a) + abs a / b ∧ x = min_val) ∧
    a = -2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2557_255772


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2557_255743

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2557_255743


namespace NUMINAMATH_CALUDE_even_a_iff_xor_inequality_l2557_255711

-- Define bitwise XOR operation
def bitwiseXOR (a b : ℕ) : ℕ :=
  Nat.rec 0 (fun k res => 
    if (a / 2^k + b / 2^k - res / 2^k) % 2 = 0 
    then res 
    else res + 2^k) a

-- State the theorem
theorem even_a_iff_xor_inequality (a : ℕ) : 
  (a > 0 ∧ a % 2 = 0) ↔ 
  (∀ x y : ℕ, x > y → bitwiseXOR x (a * x) ≠ bitwiseXOR y (a * y)) := by
sorry

end NUMINAMATH_CALUDE_even_a_iff_xor_inequality_l2557_255711


namespace NUMINAMATH_CALUDE_fraction_of_men_left_l2557_255760

/-- Represents the movie screening scenario -/
structure MovieScreening where
  total_guests : ℕ
  women : ℕ
  men : ℕ
  children : ℕ
  children_left : ℕ
  people_stayed : ℕ

/-- The specific movie screening instance from the problem -/
def problem_screening : MovieScreening :=
  { total_guests := 50
  , women := 25
  , men := 15
  , children := 10
  , children_left := 4
  , people_stayed := 43
  }

/-- Theorem stating that the fraction of men who left is 1/5 -/
theorem fraction_of_men_left (s : MovieScreening) 
  (h1 : s.total_guests = 50)
  (h2 : s.women = s.total_guests / 2)
  (h3 : s.men = 15)
  (h4 : s.children = s.total_guests - s.women - s.men)
  (h5 : s.children_left = 4)
  (h6 : s.people_stayed = 43) :
  (s.total_guests - s.people_stayed - s.children_left) / s.men = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_men_left_l2557_255760


namespace NUMINAMATH_CALUDE_digit_sum_last_digit_match_l2557_255708

theorem digit_sum_last_digit_match 
  (digits : Finset ℕ) 
  (h_digits_size : digits.card = 7) 
  (h_digits_distinct : ∀ (a b : ℕ), a ∈ digits → b ∈ digits → a ≠ b → a ≠ b) 
  (h_digits_range : ∀ d ∈ digits, d < 10) :
  ∀ n : ℕ, ∃ (a b : ℕ), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (a + b) % 10 = n % 10 := by
  sorry


end NUMINAMATH_CALUDE_digit_sum_last_digit_match_l2557_255708


namespace NUMINAMATH_CALUDE_g_neg_one_eq_neg_one_l2557_255738

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_eq_neg_one
  (h1 : is_odd_function f)
  (h2 : f 1 = 1) :
  g f (-1) = -1 := by
    sorry

end NUMINAMATH_CALUDE_g_neg_one_eq_neg_one_l2557_255738


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l2557_255723

/-- A random variable following a normal distribution with mean 3 and standard deviation σ -/
def X (σ : ℝ) : Type := Unit

/-- The probability that X is less than 2 -/
def prob_X_less_than_2 (σ : ℝ) : ℝ := 0.3

/-- The probability that X is between 2 and 4 -/
def prob_X_between_2_and_4 (σ : ℝ) : ℝ := 1 - 2 * prob_X_less_than_2 σ

theorem normal_distribution_symmetry (σ : ℝ) (h : σ > 0) :
  prob_X_between_2_and_4 σ = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l2557_255723


namespace NUMINAMATH_CALUDE_basketball_free_throws_l2557_255791

theorem basketball_free_throws 
  (two_point_shots three_point_shots free_throws : ℕ) :
  (3 * three_point_shots = 2 * two_point_shots) →
  (three_point_shots = two_point_shots - 2) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 68) →
  free_throws = 44 := by
sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l2557_255791


namespace NUMINAMATH_CALUDE_square_area_with_rectangle_division_l2557_255713

theorem square_area_with_rectangle_division (x : ℝ) (h1 : x > 0) : 
  let rectangle_area := 14
  let square_side := 4 * x
  let rectangle_width := x
  let rectangle_length := 3 * x
  rectangle_area = rectangle_width * rectangle_length →
  (square_side)^2 = 224/3 := by
sorry

end NUMINAMATH_CALUDE_square_area_with_rectangle_division_l2557_255713


namespace NUMINAMATH_CALUDE_days_in_year_l2557_255775

/-- The number of days in a year, given the number of hours in a year and hours in a day -/
theorem days_in_year (hours_in_year : ℕ) (hours_in_day : ℕ) 
  (h1 : hours_in_year = 8760) (h2 : hours_in_day = 24) : 
  hours_in_year / hours_in_day = 365 := by
  sorry

end NUMINAMATH_CALUDE_days_in_year_l2557_255775


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2557_255785

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 4 * x - 3 + a = b * x + 2) ↔ b ≠ 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2557_255785


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2557_255745

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2557_255745


namespace NUMINAMATH_CALUDE_y_minimum_range_l2557_255758

def y (x : ℝ) : ℝ := |x^2 - 1| + |2*x^2 - 1| + |3*x^2 - 1|

theorem y_minimum_range :
  ∀ x : ℝ, y x ≥ 1 ∧
  (y x = 1 ↔ (x ∈ Set.Icc (-Real.sqrt (1/2)) (-Real.sqrt (1/3)) ∪ 
              Set.Icc (Real.sqrt (1/3)) (Real.sqrt (1/2)))) :=
sorry

end NUMINAMATH_CALUDE_y_minimum_range_l2557_255758


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2557_255764

theorem complex_fraction_simplification :
  (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2557_255764


namespace NUMINAMATH_CALUDE_remainder_x7_plus_2_div_x_plus_1_l2557_255795

theorem remainder_x7_plus_2_div_x_plus_1 :
  ∃ q : Polynomial ℤ, (X ^ 7 + 2 : Polynomial ℤ) = (X + 1) * q + 1 :=
sorry

end NUMINAMATH_CALUDE_remainder_x7_plus_2_div_x_plus_1_l2557_255795


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l2557_255720

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l2557_255720


namespace NUMINAMATH_CALUDE_books_remaining_correct_l2557_255729

/-- Calculates the number of books remaining on the shelf by the evening. -/
def books_remaining (initial : ℕ) (borrowed_lunch : ℕ) (added : ℕ) (borrowed_evening : ℕ) : ℕ :=
  initial - borrowed_lunch + added - borrowed_evening

/-- Proves that the number of books remaining on the shelf by the evening is correct. -/
theorem books_remaining_correct (initial : ℕ) (borrowed_lunch : ℕ) (added : ℕ) (borrowed_evening : ℕ)
    (h1 : initial = 100)
    (h2 : borrowed_lunch = 50)
    (h3 : added = 40)
    (h4 : borrowed_evening = 30) :
    books_remaining initial borrowed_lunch added borrowed_evening = 60 := by
  sorry

end NUMINAMATH_CALUDE_books_remaining_correct_l2557_255729


namespace NUMINAMATH_CALUDE_power_division_l2557_255703

theorem power_division (a : ℝ) (h : a ≠ 0) : a^6 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2557_255703


namespace NUMINAMATH_CALUDE_storage_house_blocks_l2557_255763

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular prism -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the specifications of the storage house -/
structure StorageHouse where
  outerDimensions : Dimensions
  wallThickness : ℕ

/-- Calculates the inner dimensions of the storage house -/
def innerDimensions (s : StorageHouse) : Dimensions :=
  { length := s.outerDimensions.length - 2 * s.wallThickness,
    width := s.outerDimensions.width - 2 * s.wallThickness,
    height := s.outerDimensions.height - s.wallThickness }

/-- Calculates the number of blocks needed for the storage house -/
def blocksNeeded (s : StorageHouse) : ℕ :=
  volume s.outerDimensions - volume (innerDimensions s)

theorem storage_house_blocks :
  let s : StorageHouse :=
    { outerDimensions := { length := 15, width := 12, height := 8 },
      wallThickness := 2 }
  blocksNeeded s = 912 := by sorry

end NUMINAMATH_CALUDE_storage_house_blocks_l2557_255763


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l2557_255734

theorem cube_sum_minus_product_eq_2003 : 
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} = 
  {(668, 668, 667), (668, 667, 668), (667, 668, 668)} := by
sorry

end NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l2557_255734


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l2557_255731

/-- Given a line defined by (2, -1) · ((x, y) - (4, -3)) = 0, 
    prove that it's equivalent to y = 2x - 11 -/
theorem line_equation_equivalence :
  ∀ (x y : ℝ), 
  (2 * (x - 4) + (-1) * (y - (-3)) = 0) ↔ (y = 2 * x - 11) := by
sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l2557_255731


namespace NUMINAMATH_CALUDE_gold_bars_remaining_l2557_255717

theorem gold_bars_remaining (initial_bars : ℕ) (tax_rate : ℚ) (divorce_loss_rate : ℚ) : 
  initial_bars = 60 ∧ tax_rate = 1/10 ∧ divorce_loss_rate = 1/2 →
  ↑initial_bars * (1 - tax_rate) * (1 - divorce_loss_rate) = 27 := by
sorry

end NUMINAMATH_CALUDE_gold_bars_remaining_l2557_255717


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l2557_255705

/-- The height difference between Janet's sandcastle and her sister's sandcastle --/
def height_difference : ℝ :=
  let janet_height : ℝ := 3.6666666666666665
  let sister_height : ℝ := 2.3333333333333335
  janet_height - sister_height

/-- Theorem stating that the height difference is 1.333333333333333 feet --/
theorem sandcastle_height_difference :
  height_difference = 1.333333333333333 := by sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l2557_255705


namespace NUMINAMATH_CALUDE_existence_of_equal_modulus_unequal_squares_l2557_255766

theorem existence_of_equal_modulus_unequal_squares : ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_equal_modulus_unequal_squares_l2557_255766


namespace NUMINAMATH_CALUDE_Q_has_negative_root_l2557_255773

/-- The polynomial Q(x) = x^7 - 4x^6 + 2x^5 - 9x^3 + 2x + 16 -/
def Q (x : ℝ) : ℝ := x^7 - 4*x^6 + 2*x^5 - 9*x^3 + 2*x + 16

/-- The polynomial Q(x) has at least one negative root -/
theorem Q_has_negative_root : ∃ x : ℝ, x < 0 ∧ Q x = 0 := by sorry

end NUMINAMATH_CALUDE_Q_has_negative_root_l2557_255773


namespace NUMINAMATH_CALUDE_equation_solutions_l2557_255709

variable (a : ℝ)

theorem equation_solutions :
  {x : ℝ | x * (x + a)^3 * (5 - x) = 0} = {0, -a, 5} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2557_255709


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l2557_255757

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l2557_255757


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l2557_255761

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic_mean : (a + b) / 2 = 10)
  (h_geometric_mean : Real.sqrt (a * b) = 24) :
  ∀ x, x^2 - 20*x + 576 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l2557_255761


namespace NUMINAMATH_CALUDE_problem_solution_l2557_255762

theorem problem_solution : 
  (∀ x : ℝ, (Real.sqrt 24 - Real.sqrt 6) / Real.sqrt 3 - (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 2 - 1) ∧ 
  (∀ x : ℝ, 2 * x^3 - 16 = 0 ↔ x = 2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2557_255762


namespace NUMINAMATH_CALUDE_square_sum_from_system_l2557_255788

theorem square_sum_from_system (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : x^2 * y + x * y^2 + x + y = 63) : 
  x^2 + y^2 = 69 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_system_l2557_255788


namespace NUMINAMATH_CALUDE_expression_simplification_l2557_255780

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 4) * (9 * p - 12) = 89 * p - 84 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2557_255780


namespace NUMINAMATH_CALUDE_ramsey_theorem_for_interns_l2557_255755

/-- Represents the relationship between two interns -/
inductive Relationship
  | Knows
  | DoesNotKnow

/-- Defines a group of interns and their relationships -/
structure InternGroup :=
  (size : Nat)
  (relationships : Fin size → Fin size → Relationship)

/-- The main theorem -/
theorem ramsey_theorem_for_interns (group : InternGroup) (h : group.size = 6) :
  ∃ (a b c : Fin group.size),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    ((group.relationships a b = Relationship.Knows ∧
      group.relationships b c = Relationship.Knows ∧
      group.relationships a c = Relationship.Knows) ∨
     (group.relationships a b = Relationship.DoesNotKnow ∧
      group.relationships b c = Relationship.DoesNotKnow ∧
      group.relationships a c = Relationship.DoesNotKnow)) :=
sorry

end NUMINAMATH_CALUDE_ramsey_theorem_for_interns_l2557_255755


namespace NUMINAMATH_CALUDE_fraction_simplification_l2557_255768

theorem fraction_simplification (x : ℝ) (hx : x ≠ 0) :
  (42 * x^3) / (63 * x^5) = 2 / (3 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2557_255768


namespace NUMINAMATH_CALUDE_num_positive_divisors_1386_l2557_255715

/-- The number of positive divisors of a natural number -/
def numPositiveDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 1386 is 24 -/
theorem num_positive_divisors_1386 : numPositiveDivisors 1386 = 24 := by sorry

end NUMINAMATH_CALUDE_num_positive_divisors_1386_l2557_255715


namespace NUMINAMATH_CALUDE_rational_equation_implication_l2557_255765

theorem rational_equation_implication (a b : ℚ) 
  (h : Real.sqrt (a + 4) + (b - 2)^2 = 0) : a - b = -6 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_implication_l2557_255765


namespace NUMINAMATH_CALUDE_table_formula_proof_l2557_255770

def f (x : ℕ) : ℕ := x^2 + 3*x + 1

theorem table_formula_proof :
  (f 1 = 5) ∧ (f 2 = 11) ∧ (f 3 = 19) ∧ (f 4 = 29) ∧ (f 5 = 41) :=
by sorry

end NUMINAMATH_CALUDE_table_formula_proof_l2557_255770


namespace NUMINAMATH_CALUDE_correct_answers_statistics_probability_two_multiple_choice_A_l2557_255747

-- Define the data for schools A and B
def school_A_students : ℕ := 12
def school_A_mean : ℚ := 1
def school_A_variance : ℚ := 1

def school_B_students : ℕ := 8
def school_B_mean : ℚ := 3/2
def school_B_variance : ℚ := 1/4

-- Define the boxes
def box_A_multiple_choice : ℕ := 4
def box_A_fill_blank : ℕ := 2
def box_B_multiple_choice : ℕ := 3
def box_B_fill_blank : ℕ := 3

-- Part 1: Mean and Variance Calculation
def total_students : ℕ := school_A_students + school_B_students

theorem correct_answers_statistics : 
  let total_mean : ℚ := (school_A_students * school_A_mean + school_B_students * school_B_mean) / total_students
  let total_variance : ℚ := (school_A_students * (school_A_variance + (school_A_mean - total_mean)^2) + 
                             school_B_students * (school_B_variance + (school_B_mean - total_mean)^2)) / total_students
  total_mean = 6/5 ∧ total_variance = 19/25 := by sorry

-- Part 2: Probability Calculation
def prob_two_multiple_choice_A : ℚ := 2/5
def prob_one_multiple_one_fill_A : ℚ := 8/15
def prob_two_fill_A : ℚ := 1/15

def prob_B_multiple_given_A_two_multiple : ℚ := 5/8
def prob_B_multiple_given_A_one_each : ℚ := 8/15
def prob_B_multiple_given_A_two_fill : ℚ := 3/8

theorem probability_two_multiple_choice_A : 
  let prob_B_multiple : ℚ := prob_two_multiple_choice_A * prob_B_multiple_given_A_two_multiple + 
                              prob_one_multiple_one_fill_A * prob_B_multiple_given_A_one_each + 
                              prob_two_fill_A * prob_B_multiple_given_A_two_fill
  let prob_A_two_multiple_given_B_multiple : ℚ := (prob_two_multiple_choice_A * prob_B_multiple_given_A_two_multiple) / prob_B_multiple
  prob_A_two_multiple_given_B_multiple = 6/13 := by sorry

end NUMINAMATH_CALUDE_correct_answers_statistics_probability_two_multiple_choice_A_l2557_255747


namespace NUMINAMATH_CALUDE_distance_between_points_l2557_255776

theorem distance_between_points : 
  let point1 : ℝ × ℝ := (0, 6)
  let point2 : ℝ × ℝ := (4, 0)
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2557_255776


namespace NUMINAMATH_CALUDE_fred_current_money_l2557_255750

/-- Fred's money situation --/
def fred_money_problem (initial_amount earned_amount : ℕ) : Prop :=
  initial_amount + earned_amount = 40

/-- Theorem: Fred now has 40 dollars --/
theorem fred_current_money :
  fred_money_problem 19 21 :=
by
  sorry

end NUMINAMATH_CALUDE_fred_current_money_l2557_255750


namespace NUMINAMATH_CALUDE_function_with_two_zeros_m_range_l2557_255767

theorem function_with_two_zeros_m_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m < -2 ∨ m > 2 :=
sorry

end NUMINAMATH_CALUDE_function_with_two_zeros_m_range_l2557_255767


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_l2557_255718

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A circle with radius r -/
structure Circle where
  r : ℝ
  h : r > 0

/-- The theorem statement -/
theorem ellipse_circle_tangent (C : Ellipse) (O : Circle) :
  C.a = 2 * Real.sqrt 2 →  -- Left vertex at (-2√2, 0)
  O.r = 2 →  -- Circle equation: x² + y² = 4
  (∃ F : ℝ × ℝ, F.1 = -Real.sqrt 2 ∧ F.2 = 0 ∧
    ∃ A B : ℝ × ℝ, 
      -- A and B are on the circle
      (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧
      -- Line AB passes through F
      (B.2 - A.2) * (F.1 - A.1) = (F.2 - A.2) * (B.1 - A.1)) →
  C.a^2 + C.b^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_l2557_255718


namespace NUMINAMATH_CALUDE_system_solution_l2557_255714

theorem system_solution (x y m n : ℝ) 
  (eq1 : x + y = m)
  (eq2 : x - y = n + 1)
  (sol_x : x = 3)
  (sol_y : y = 2) :
  m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2557_255714


namespace NUMINAMATH_CALUDE_amanda_notebooks_l2557_255749

/-- Represents the number of notebooks Amanda ordered -/
def ordered_notebooks : ℕ := 6

/-- Amanda's initial number of notebooks -/
def initial_notebooks : ℕ := 10

/-- Number of notebooks Amanda lost -/
def lost_notebooks : ℕ := 2

/-- Amanda's final number of notebooks -/
def final_notebooks : ℕ := 14

theorem amanda_notebooks :
  initial_notebooks + ordered_notebooks - lost_notebooks = final_notebooks :=
by sorry

end NUMINAMATH_CALUDE_amanda_notebooks_l2557_255749


namespace NUMINAMATH_CALUDE_walter_school_expenses_l2557_255790

/-- Represents Walter's weekly work schedule and earnings --/
structure WalterSchedule where
  job1_weekday_hours : ℝ
  job1_weekend_hours : ℝ
  job1_hourly_rate : ℝ
  job1_weekly_bonus : ℝ
  job1_tax_rate : ℝ
  job2_hours : ℝ
  job2_hourly_rate : ℝ
  job2_tax_rate : ℝ
  job3_hours : ℝ
  job3_hourly_rate : ℝ
  school_allocation_rate : ℝ

/-- Calculates Walter's weekly school expense allocation --/
def calculateSchoolExpenses (schedule : WalterSchedule) : ℝ :=
  let job1_earnings := (schedule.job1_weekday_hours * 5 + schedule.job1_weekend_hours * 2) * schedule.job1_hourly_rate + schedule.job1_weekly_bonus
  let job1_after_tax := job1_earnings * (1 - schedule.job1_tax_rate)
  let job2_earnings := schedule.job2_hours * schedule.job2_hourly_rate
  let job2_after_tax := job2_earnings * (1 - schedule.job2_tax_rate)
  let job3_earnings := schedule.job3_hours * schedule.job3_hourly_rate
  let total_earnings := job1_after_tax + job2_after_tax + job3_earnings
  total_earnings * schedule.school_allocation_rate

/-- Theorem stating that Walter's weekly school expense allocation is approximately $211.69 --/
theorem walter_school_expenses (schedule : WalterSchedule) 
  (h1 : schedule.job1_weekday_hours = 4)
  (h2 : schedule.job1_weekend_hours = 6)
  (h3 : schedule.job1_hourly_rate = 5)
  (h4 : schedule.job1_weekly_bonus = 50)
  (h5 : schedule.job1_tax_rate = 0.1)
  (h6 : schedule.job2_hours = 5)
  (h7 : schedule.job2_hourly_rate = 7)
  (h8 : schedule.job2_tax_rate = 0.05)
  (h9 : schedule.job3_hours = 6)
  (h10 : schedule.job3_hourly_rate = 10)
  (h11 : schedule.school_allocation_rate = 0.75) :
  ∃ ε > 0, |calculateSchoolExpenses schedule - 211.69| < ε := by
  sorry

end NUMINAMATH_CALUDE_walter_school_expenses_l2557_255790


namespace NUMINAMATH_CALUDE_seventeenth_group_number_l2557_255737

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstGroup : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := totalStudents / sampleSize
  firstGroup + (groupNumber - 1) * interval

/-- Theorem: The 17th group number in the given systematic sampling is 264 -/
theorem seventeenth_group_number :
  systematicSample 800 50 8 17 = 264 := by
  sorry

end NUMINAMATH_CALUDE_seventeenth_group_number_l2557_255737


namespace NUMINAMATH_CALUDE_percentage_of_democrats_l2557_255739

theorem percentage_of_democrats (D R : ℝ) : 
  D + R = 100 →
  0.7 * D + 0.2 * R = 50 →
  D = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_democrats_l2557_255739


namespace NUMINAMATH_CALUDE_equation_solution_l2557_255736

theorem equation_solution (y : ℝ) : 
  ∃ z : ℝ, 19 * (1 + y) + z = 19 * (-1 + y) - 21 ∧ z = -59 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2557_255736


namespace NUMINAMATH_CALUDE_second_player_wins_l2557_255769

/-- Represents the state of the game -/
structure GameState where
  grid_size : Nat
  piece1_pos : Nat
  piece2_pos : Nat

/-- Defines a valid move in the game -/
inductive Move
  | One
  | Two

/-- Applies a move to a game state -/
def apply_move (state : GameState) (player : Nat) (move : Move) : GameState :=
  match player, move with
  | 1, Move.One => { state with piece1_pos := state.piece1_pos + 1 }
  | 1, Move.Two => { state with piece1_pos := state.piece1_pos + 2 }
  | 2, Move.One => { state with piece2_pos := state.piece2_pos - 1 }
  | 2, Move.Two => { state with piece2_pos := state.piece2_pos - 2 }
  | _, _ => state

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (player : Nat) (move : Move) : Prop :=
  match player, move with
  | 1, Move.One => state.piece1_pos + 1 < state.piece2_pos
  | 1, Move.Two => state.piece1_pos + 2 < state.piece2_pos
  | 2, Move.One => state.piece1_pos < state.piece2_pos - 1
  | 2, Move.Two => state.piece1_pos < state.piece2_pos - 2
  | _, _ => False

/-- Checks if the game is over -/
def is_game_over (state : GameState) : Prop :=
  state.piece2_pos - state.piece1_pos <= 1

/-- Checks if the number of empty squares between pieces is a multiple of 3 -/
def is_multiple_of_three (state : GameState) : Prop :=
  (state.piece2_pos - state.piece1_pos - 1) % 3 = 0

/-- Theorem: The second player has a winning strategy if and only if
    the number of empty squares between pieces is always a multiple of 3
    after the second player's move -/
theorem second_player_wins (initial_state : GameState)
  (h_initial : initial_state.grid_size = 20 ∧
               initial_state.piece1_pos = 1 ∧
               initial_state.piece2_pos = 20) :
  (∀ (game_state : GameState),
   ∀ (move1 : Move),
   is_valid_move game_state 1 move1 →
   ∃ (move2 : Move),
   is_valid_move (apply_move game_state 1 move1) 2 move2 ∧
   is_multiple_of_three (apply_move (apply_move game_state 1 move1) 2 move2)) ↔
  (∃ (strategy : GameState → Move),
   ∀ (game_state : GameState),
   ¬is_game_over game_state →
   is_valid_move game_state 2 (strategy game_state) ∧
   is_multiple_of_three (apply_move game_state 2 (strategy game_state))) :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l2557_255769


namespace NUMINAMATH_CALUDE_last_two_digits_sum_sum_of_last_two_digits_main_result_l2557_255789

theorem last_two_digits_sum (n : ℕ) : ∃ (k : ℕ), 11^2004 - 5 = k * 100 + 36 :=
by sorry

theorem sum_of_last_two_digits : (11^2004 - 5) % 100 = 36 :=
by sorry

theorem main_result : (((11^2004 - 5) / 10) % 10) + ((11^2004 - 5) % 10) = 9 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_sum_of_last_two_digits_main_result_l2557_255789


namespace NUMINAMATH_CALUDE_expression_simplification_l2557_255730

theorem expression_simplification (a : ℚ) (h : a = -1/2) :
  (a + 2)^2 + (a + 2) * (2 - a) - 6 * a = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2557_255730


namespace NUMINAMATH_CALUDE_reciprocal_of_opposite_negative_two_thirds_l2557_255740

theorem reciprocal_of_opposite_negative_two_thirds :
  (-(- (2 : ℚ) / 3))⁻¹ = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_opposite_negative_two_thirds_l2557_255740


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2557_255712

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero. -/
def symmetricToOrigin (p q : Point) : Prop :=
  p.x + q.x = 0 ∧ p.y + q.y = 0

theorem symmetric_point_coordinates :
  let A : Point := ⟨2, 4⟩
  let B : Point := ⟨-2, -4⟩
  symmetricToOrigin A B → B = ⟨-2, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2557_255712
