import Mathlib

namespace NUMINAMATH_CALUDE_probability_ratio_l866_86696

def total_cards : ℕ := 50
def numbers_range : ℕ := 10
def cards_per_number : ℕ := 5
def cards_drawn : ℕ := 5

def probability_all_same (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / (total.choose drawn)

def probability_four_and_one (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  ((range * (range - 1)) * (per_num.choose (drawn - 1)) * (per_num.choose 1) : ℚ) / (total.choose drawn)

theorem probability_ratio :
  (probability_four_and_one total_cards numbers_range cards_per_number cards_drawn) /
  (probability_all_same total_cards numbers_range cards_per_number cards_drawn) = 225 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l866_86696


namespace NUMINAMATH_CALUDE_product_signs_l866_86683

theorem product_signs (a b c d e : ℝ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 →
  (∃ (pos neg : Finset (ℝ × ℝ)), 
    pos.card = 5 ∧ 
    neg.card = 5 ∧ 
    (∀ p ∈ pos, p.1 + p.2 > 0) ∧
    (∀ p ∈ neg, p.1 + p.2 < 0) ∧
    pos ∩ neg = ∅ ∧
    pos ∪ neg = {(a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)}) →
  (∃ (pos_prod neg_prod : Finset (ℝ × ℝ)),
    pos_prod.card = 4 ∧
    neg_prod.card = 6 ∧
    (∀ p ∈ pos_prod, p.1 * p.2 > 0) ∧
    (∀ p ∈ neg_prod, p.1 * p.2 < 0) ∧
    pos_prod ∩ neg_prod = ∅ ∧
    pos_prod ∪ neg_prod = {(a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)}) :=
by sorry

end NUMINAMATH_CALUDE_product_signs_l866_86683


namespace NUMINAMATH_CALUDE_equals_2022_l866_86613

theorem equals_2022 : 1 - (-2021) = 2022 := by
  sorry

#check equals_2022

end NUMINAMATH_CALUDE_equals_2022_l866_86613


namespace NUMINAMATH_CALUDE_factor_implies_p_value_l866_86653

theorem factor_implies_p_value (m p : ℤ) : 
  (∃ k : ℤ, m^2 - p*m - 24 = (m - 8) * k) → p = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_p_value_l866_86653


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l866_86663

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l866_86663


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l866_86624

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a - 8 > b - 8) → ¬(a > b)) ↔ (a - 8 ≤ b - 8 → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l866_86624


namespace NUMINAMATH_CALUDE_double_symmetry_quadratic_l866_86632

/-- Given a quadratic function f(x) = ax^2 + bx + c, 
    this function returns the quadratic function 
    that results from applying y-axis symmetry 
    followed by x-axis symmetry -/
def double_symmetry (a b c : ℝ) : ℝ → ℝ := 
  fun x => -a * x^2 + b * x - c

/-- Theorem stating that the double symmetry operation 
    on a quadratic function results in the expected 
    transformed function -/
theorem double_symmetry_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  ∀ x, double_symmetry a b c x = -(a * x^2 + b * x + c) :=
by
  sorry

#check double_symmetry_quadratic

end NUMINAMATH_CALUDE_double_symmetry_quadratic_l866_86632


namespace NUMINAMATH_CALUDE_sugar_cube_theorem_l866_86676

/-- Represents a box of sugar cubes -/
structure SugarBox where
  height : Nat
  width : Nat
  depth : Nat

/-- Calculates the number of remaining cubes in a sugar box after eating layers -/
def remaining_cubes (box : SugarBox) : Set Nat :=
  if box.width * box.depth = 77 ∧ box.height * box.depth = 55 then
    if box.depth = 1 then {0}
    else if box.depth = 11 then {300}
    else ∅
  else ∅

/-- Theorem stating that the number of remaining cubes is either 300 or 0 -/
theorem sugar_cube_theorem (box : SugarBox) :
  remaining_cubes box ⊆ {0, 300} :=
by sorry

end NUMINAMATH_CALUDE_sugar_cube_theorem_l866_86676


namespace NUMINAMATH_CALUDE_ellipse_k_range_l866_86685

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- The origin is inside the ellipse -/
def origin_inside (k : ℝ) : Prop :=
  ∃ ε > 0, ∀ x y : ℝ, x^2 + y^2 < ε^2 → ellipse k x y

/-- The theorem stating the range of k -/
theorem ellipse_k_range :
  ∀ k : ℝ, origin_inside k → 0 < |k| ∧ |k| < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l866_86685


namespace NUMINAMATH_CALUDE_first_day_is_thursday_l866_86630

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure MonthDay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the previous day of the week -/
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

/-- Theorem: If the 24th day of a month is a Saturday, then the 1st day of that month is a Thursday -/
theorem first_day_is_thursday (m : MonthDay) (h : m.day = 24 ∧ m.dayOfWeek = DayOfWeek.Saturday) :
  ∃ (firstDay : MonthDay), firstDay.day = 1 ∧ firstDay.dayOfWeek = DayOfWeek.Thursday :=
by sorry

end NUMINAMATH_CALUDE_first_day_is_thursday_l866_86630


namespace NUMINAMATH_CALUDE_fraction_simplification_l866_86668

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l866_86668


namespace NUMINAMATH_CALUDE_nabla_calculation_l866_86657

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_calculation : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l866_86657


namespace NUMINAMATH_CALUDE_even_function_value_l866_86641

def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

theorem even_function_value (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-b) (2*b - 2), f a b x = f a b (-x)) →
  f a b (b/2) = 2 := by
sorry

end NUMINAMATH_CALUDE_even_function_value_l866_86641


namespace NUMINAMATH_CALUDE_smallest_x_value_l866_86643

theorem smallest_x_value (y : ℕ+) (x : ℕ) 
  (h : (857 : ℚ) / 1000 = (y : ℚ) / ((210 : ℚ) + x)) : 
  ∀ x' : ℕ, x' ≥ x → x = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l866_86643


namespace NUMINAMATH_CALUDE_license_combinations_l866_86633

/-- Represents the number of choices for the letter in a license -/
def letter_choices : ℕ := 3

/-- Represents the number of choices for each digit in a license -/
def digit_choices : ℕ := 10

/-- Represents the number of digits in a license -/
def num_digits : ℕ := 4

/-- Calculates the total number of possible license combinations -/
def total_combinations : ℕ := letter_choices * digit_choices ^ num_digits

/-- Proves that the number of unique license combinations is 30000 -/
theorem license_combinations : total_combinations = 30000 := by
  sorry

end NUMINAMATH_CALUDE_license_combinations_l866_86633


namespace NUMINAMATH_CALUDE_vector_square_difference_l866_86618

theorem vector_square_difference (a b : ℝ × ℝ) 
  (h1 : a + b = (-3, 6)) 
  (h2 : a - b = (-3, 2)) 
  (h3 : a ≠ (0, 0)) 
  (h4 : b ≠ (0, 0)) : 
  (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_square_difference_l866_86618


namespace NUMINAMATH_CALUDE_item_a_price_correct_l866_86689

/-- The price of item (a) in won -/
def item_a_price : ℕ := 7 * 1000 + 4 * 100 + 5 * 10

/-- The number of 1000 won coins used -/
def coins_1000 : ℕ := 7

/-- The number of 100 won coins used -/
def coins_100 : ℕ := 4

/-- The number of 10 won coins used -/
def coins_10 : ℕ := 5

theorem item_a_price_correct : item_a_price = 7450 := by
  sorry

end NUMINAMATH_CALUDE_item_a_price_correct_l866_86689


namespace NUMINAMATH_CALUDE_galaxy_planets_l866_86664

theorem galaxy_planets (total : ℕ) (ratio : ℕ) (h1 : total = 200) (h2 : ratio = 8) : 
  ∃ (planets : ℕ), planets * (ratio + 1) = total ∧ planets = 22 := by
  sorry

end NUMINAMATH_CALUDE_galaxy_planets_l866_86664


namespace NUMINAMATH_CALUDE_stratified_sampling_junior_count_l866_86607

theorem stratified_sampling_junior_count 
  (total_employees : ℕ) 
  (junior_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 150) 
  (h2 : junior_employees = 90) 
  (h3 : sample_size = 30) :
  (junior_employees : ℚ) * (sample_size : ℚ) / (total_employees : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_junior_count_l866_86607


namespace NUMINAMATH_CALUDE_largest_valid_number_l866_86609

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- Four-digit number
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧  -- All digits are different
  (∀ i j, i < j → (n / 10^i) % 10 ≤ (n / 10^j) % 10)  -- No two digits can be swapped to form a smaller number

theorem largest_valid_number : 
  is_valid_number 7089 ∧ ∀ m, is_valid_number m → m ≤ 7089 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l866_86609


namespace NUMINAMATH_CALUDE_domain_of_function_1_l866_86655

theorem domain_of_function_1 (x : ℝ) : 
  (x ≥ 1 ∨ x < -1) ↔ (x ≠ -1 ∧ (x - 1) / (x + 1) ≥ 0) :=
sorry

#check domain_of_function_1

end NUMINAMATH_CALUDE_domain_of_function_1_l866_86655


namespace NUMINAMATH_CALUDE_initial_population_proof_l866_86628

/-- The population change function over 5 years -/
def population_change (P : ℝ) : ℝ :=
  P * 0.9 * 1.1 * 0.9 * 1.15 * 0.75

/-- Theorem stating the initial population given the final population -/
theorem initial_population_proof : 
  ∃ P : ℕ, population_change (P : ℝ) = 4455 ∧ P = 5798 :=
sorry

end NUMINAMATH_CALUDE_initial_population_proof_l866_86628


namespace NUMINAMATH_CALUDE_exponential_equation_l866_86612

theorem exponential_equation (x y : ℝ) (h : 2*x + 4*y - 3 = 0) : 
  (4 : ℝ)^x * (16 : ℝ)^y = 8 := by
sorry

end NUMINAMATH_CALUDE_exponential_equation_l866_86612


namespace NUMINAMATH_CALUDE_tan_theta_value_l866_86649

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_theta_value (θ : Real) (h : ∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l866_86649


namespace NUMINAMATH_CALUDE_commercial_time_l866_86661

theorem commercial_time (p : ℝ) (h : p = 0.9) : (1 - p) * 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_commercial_time_l866_86661


namespace NUMINAMATH_CALUDE_marble_sculpture_weight_l866_86645

theorem marble_sculpture_weight (W : ℝ) : 
  W > 0 →
  (1 - 0.3) * (1 - 0.2) * (1 - 0.25) * W = 105 →
  W = 250 := by
sorry

end NUMINAMATH_CALUDE_marble_sculpture_weight_l866_86645


namespace NUMINAMATH_CALUDE_gingerbread_red_hat_percentage_l866_86687

/-- Calculates the percentage of gingerbread men with red hats -/
theorem gingerbread_red_hat_percentage
  (red_hats : ℕ)
  (blue_boots : ℕ)
  (both : ℕ)
  (h1 : red_hats = 6)
  (h2 : blue_boots = 9)
  (h3 : both = 3) :
  (red_hats : ℚ) / ((red_hats + blue_boots - both) : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_gingerbread_red_hat_percentage_l866_86687


namespace NUMINAMATH_CALUDE_same_terminal_side_as_405_degrees_l866_86690

theorem same_terminal_side_as_405_degrees :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 + 45) ↔ (∃ n : ℤ, θ = 405 + n * 360) :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_405_degrees_l866_86690


namespace NUMINAMATH_CALUDE_midpoint_pentagon_perimeter_l866_86656

/-- A convex pentagon in a 2D plane. -/
structure ConvexPentagon where
  -- We don't need to define the specific properties of a convex pentagon for this problem

/-- The sum of all diagonals of a convex pentagon. -/
def sum_of_diagonals (p : ConvexPentagon) : ℝ := sorry

/-- The pentagon formed by connecting the midpoints of the sides of a convex pentagon. -/
def midpoint_pentagon (p : ConvexPentagon) : ConvexPentagon := sorry

/-- The perimeter of a pentagon. -/
def perimeter (p : ConvexPentagon) : ℝ := sorry

/-- 
Theorem: The perimeter of the pentagon formed by connecting the midpoints 
of the sides of a convex pentagon is equal to half the sum of all diagonals 
of the original pentagon.
-/
theorem midpoint_pentagon_perimeter (p : ConvexPentagon) : 
  perimeter (midpoint_pentagon p) = (1/2) * sum_of_diagonals p := by
  sorry

end NUMINAMATH_CALUDE_midpoint_pentagon_perimeter_l866_86656


namespace NUMINAMATH_CALUDE_exists_non_intersecting_circle_exists_regular_polygon_M_properties_l866_86623

/-- Line system M: x cos θ + (y-1) sin θ = 1, where 0 ≤ θ ≤ 2π -/
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ p.1 * Real.cos θ + (p.2 - 1) * Real.sin θ = 1}

/-- There exists a circle that does not intersect any of the lines in M -/
theorem exists_non_intersecting_circle (M : Set (ℝ × ℝ)) : 
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ p ∈ M, (p.1 - c.1)^2 + (p.2 - c.2)^2 > r^2 := by sorry

/-- For any integer n ≥ 3, there exists a regular n-sided polygon whose edges all lie on lines in M -/
theorem exists_regular_polygon (M : Set (ℝ × ℝ)) (n : ℕ) (hn : n ≥ 3) :
  ∃ (polygon : Fin n → ℝ × ℝ), 
    (∀ i : Fin n, ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ 
      (polygon i).1 * Real.cos θ + ((polygon i).2 - 1) * Real.sin θ = 1) ∧
    (∀ i j : Fin n, (polygon i).1^2 + (polygon i).2^2 = (polygon j).1^2 + (polygon j).2^2) := by sorry

/-- Main theorem combining the two properties -/
theorem M_properties : 
  (∃ (c : ℝ × ℝ) (r : ℝ), ∀ p ∈ M, (p.1 - c.1)^2 + (p.2 - c.2)^2 > r^2) ∧
  (∀ (n : ℕ), n ≥ 3 → 
    ∃ (polygon : Fin n → ℝ × ℝ), 
      (∀ i : Fin n, ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ 
        (polygon i).1 * Real.cos θ + ((polygon i).2 - 1) * Real.sin θ = 1) ∧
      (∀ i j : Fin n, (polygon i).1^2 + (polygon i).2^2 = (polygon j).1^2 + (polygon j).2^2)) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_intersecting_circle_exists_regular_polygon_M_properties_l866_86623


namespace NUMINAMATH_CALUDE_min_overlap_beethoven_vivaldi_l866_86692

theorem min_overlap_beethoven_vivaldi (total : ℕ) (beethoven : ℕ) (vivaldi : ℕ) 
  (h1 : total = 200) 
  (h2 : beethoven = 160) 
  (h3 : vivaldi = 130) : 
  ∃ (both : ℕ), both ≥ 90 ∧ 
    (∀ (x : ℕ), x < 90 → ¬(x ≤ beethoven ∧ x ≤ vivaldi ∧ beethoven + vivaldi - x ≤ total)) :=
by
  sorry

#check min_overlap_beethoven_vivaldi

end NUMINAMATH_CALUDE_min_overlap_beethoven_vivaldi_l866_86692


namespace NUMINAMATH_CALUDE_multiply_powers_l866_86606

theorem multiply_powers (a : ℝ) : 2 * a^3 * 3 * a^2 = 6 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_l866_86606


namespace NUMINAMATH_CALUDE_one_and_one_third_problem_l866_86635

theorem one_and_one_third_problem :
  ∃ x : ℚ, (4 / 3) * x = 45 ∧ x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_problem_l866_86635


namespace NUMINAMATH_CALUDE_log_relation_difference_l866_86684

theorem log_relation_difference (a b c d : ℤ) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) : 
  b - d = 93 := by
sorry

end NUMINAMATH_CALUDE_log_relation_difference_l866_86684


namespace NUMINAMATH_CALUDE_marble_fraction_l866_86682

theorem marble_fraction (total : ℝ) (h : total > 0) : 
  let initial_blue := (2/3) * total
  let initial_red := total - initial_blue
  let new_blue := 2 * initial_blue
  let new_red := 3 * initial_red
  let new_total := new_blue + new_red
  new_red / new_total = 3/7 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_l866_86682


namespace NUMINAMATH_CALUDE_sum_remainder_by_eight_l866_86631

theorem sum_remainder_by_eight (n : ℤ) : (8 - n + (n + 5)) % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_by_eight_l866_86631


namespace NUMINAMATH_CALUDE_cos_2_sum_of_tan_roots_l866_86698

theorem cos_2_sum_of_tan_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 + 5*x - 6 = 0 ∧ y^2 + 5*y - 6 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.cos (2 * (α + β)) = 12 / 37 := by
sorry

end NUMINAMATH_CALUDE_cos_2_sum_of_tan_roots_l866_86698


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l866_86652

theorem imaginary_part_of_complex_division : 
  let z : ℂ := -3 + 4*I
  let w : ℂ := 1 + I
  (z / w).im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l866_86652


namespace NUMINAMATH_CALUDE_expression_between_two_and_three_l866_86651

theorem expression_between_two_and_three (a b : ℝ) (h : 3 * a = 5 * b) :
  2 < |a + b| / b ∧ |a + b| / b < 3 := by sorry

end NUMINAMATH_CALUDE_expression_between_two_and_three_l866_86651


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l866_86621

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l866_86621


namespace NUMINAMATH_CALUDE_distinct_positive_numbers_properties_l866_86659

theorem distinct_positive_numbers_properties (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧ 
  (a ≠ c ∧ b ≠ c ∧ a ≠ b) := by
  sorry

end NUMINAMATH_CALUDE_distinct_positive_numbers_properties_l866_86659


namespace NUMINAMATH_CALUDE_scientific_notation_1300000_l866_86600

theorem scientific_notation_1300000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1300000 = a * (10 : ℝ) ^ n ∧ a = 1.3 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_1300000_l866_86600


namespace NUMINAMATH_CALUDE_square_area_triple_l866_86667

/-- Given a square I with diagonal 2a, prove that a square II with triple the area of square I has an area of 6a² -/
theorem square_area_triple (a : ℝ) :
  let diagonal_I : ℝ := 2 * a
  let area_I : ℝ := (diagonal_I ^ 2) / 2
  let area_II : ℝ := 3 * area_I
  area_II = 6 * a ^ 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_triple_l866_86667


namespace NUMINAMATH_CALUDE_four_digit_diff_divisible_iff_middle_digits_same_l866_86638

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9
  h4 : d ≥ 0 ∧ d ≤ 9

/-- Calculates the value of a four-digit number -/
def fourDigitValue (n : FourDigitNumber) : ℕ :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Calculates the value of the reversed four-digit number -/
def reversedValue (n : FourDigitNumber) : ℕ :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- Theorem: For a four-digit number, the difference between the number and its reverse
    is divisible by 37 if and only if the two middle digits are the same -/
theorem four_digit_diff_divisible_iff_middle_digits_same (n : FourDigitNumber) :
  (fourDigitValue n - reversedValue n) % 37 = 0 ↔ n.b = n.c := by
  sorry

end NUMINAMATH_CALUDE_four_digit_diff_divisible_iff_middle_digits_same_l866_86638


namespace NUMINAMATH_CALUDE_softball_team_ratio_l866_86675

theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 6 →
  men + women = 24 →
  (men : ℚ) / women = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l866_86675


namespace NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l866_86694

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of being an extreme value point
def is_extreme_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- Define the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x₀ : ℝ, is_extreme_point f x₀ → (deriv f) x₀ = 0) ∧
  (∃ x₀ : ℝ, (deriv f) x₀ = 0 ∧ ¬(is_extreme_point f x₀)) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l866_86694


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l866_86677

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let q := MonicCubicPolynomial a b c
  (q (2 - 3*I) = 0 ∧ q 0 = -30) →
  (a = -82/13 ∧ b = 277/13 ∧ c = -390/13) :=
by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l866_86677


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l866_86681

/-- Two arithmetic sequences and their sum sequences -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n / 2) * (a 1 + a n) ∧ T n = (n / 2) * (b 1 + b n)

/-- The ratio of sums condition -/
def sum_ratio_condition (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n) / (n + 3)

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h1 : arithmetic_sequences a b S T)
  (h2 : sum_ratio_condition S T) :
  a 5 / b 5 = 21 / 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l866_86681


namespace NUMINAMATH_CALUDE_corrected_mean_l866_86610

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * (36.5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l866_86610


namespace NUMINAMATH_CALUDE_min_value_m_min_value_m_tight_l866_86617

theorem min_value_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 (π/3), m ≥ 2 * Real.tan x) → m ≥ 2 * Real.sqrt 3 :=
by sorry

theorem min_value_m_tight : 
  ∃ m : ℝ, (∀ x ∈ Set.Icc 0 (π/3), m ≥ 2 * Real.tan x) ∧ m = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_min_value_m_tight_l866_86617


namespace NUMINAMATH_CALUDE_abs_diff_opposite_for_negative_l866_86686

theorem abs_diff_opposite_for_negative (x : ℝ) (h : x < 0) : |x - (-x)| = -2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_opposite_for_negative_l866_86686


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l866_86679

theorem impossible_coin_probabilities :
  ¬∃ (p₁ p₂ : ℝ), 0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧
    (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
    p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l866_86679


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l866_86640

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≥ 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- Define the complement of A and B with respect to ℝ
def C_UA : Set ℝ := {x : ℝ | x ∉ A}
def C_UB : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem complement_intersection_theorem :
  (C_UA ∩ C_UB) = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l866_86640


namespace NUMINAMATH_CALUDE_shaded_area_semicircle_arrangement_l866_86608

/-- The area of the shaded region in a semicircle arrangement -/
theorem shaded_area_semicircle_arrangement (n : ℕ) (d : ℝ) (h : n = 8 ∧ d = 5) :
  let large_diameter := n * d
  let large_semicircle_area := π * (large_diameter / 2)^2 / 2
  let small_semicircle_area := n * (π * d^2 / 8)
  large_semicircle_area - small_semicircle_area = 175 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_semicircle_arrangement_l866_86608


namespace NUMINAMATH_CALUDE_y_value_l866_86601

theorem y_value (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = -5) : y = 28 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l866_86601


namespace NUMINAMATH_CALUDE_subtraction_divisibility_implies_sum_l866_86669

/-- Represents a three-digit number in the form xyz --/
structure ThreeDigitNumber where
  x : Nat
  y : Nat
  z : Nat
  x_nonzero : x ≠ 0
  digits_bound : x < 10 ∧ y < 10 ∧ z < 10

/-- Converts a ThreeDigitNumber to its numerical value --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.x + 10 * n.y + n.z

theorem subtraction_divisibility_implies_sum (a b : Nat) :
  ∃ (num1 num2 : ThreeDigitNumber),
    num1.toNat = 407 + 10 * a ∧
    num2.toNat = 304 + 10 * b ∧
    830 - num1.toNat = num2.toNat ∧
    num2.toNat % 7 = 0 →
    a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_divisibility_implies_sum_l866_86669


namespace NUMINAMATH_CALUDE_complex_magnitude_l866_86603

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 4 - 2 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l866_86603


namespace NUMINAMATH_CALUDE_mathland_license_plate_probability_l866_86691

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of possible two-digit numbers --/
def two_digit_numbers : ℕ := 100

/-- The total number of possible license plates in Mathland --/
def total_license_plates : ℕ := alphabet_size * (alphabet_size - 1) * (alphabet_size - 2) * two_digit_numbers

/-- The probability of a specific license plate configuration in Mathland --/
def specific_plate_probability : ℚ := 1 / total_license_plates

theorem mathland_license_plate_probability :
  specific_plate_probability = 1 / 1560000 := by sorry

end NUMINAMATH_CALUDE_mathland_license_plate_probability_l866_86691


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l866_86611

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that in an arithmetic progression where the sum of the 3rd
    to 7th terms is 45, the 5th term is equal to 9. -/
theorem arithmetic_progression_sum (a : ℕ → ℝ) :
  is_arithmetic_progression a →
  a 3 + a 4 + a 5 + a 6 + a 7 = 45 →
  a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l866_86611


namespace NUMINAMATH_CALUDE_equator_arc_length_equals_radius_l866_86688

/-- The radius of the Earth's equator in kilometers -/
def earth_radius : ℝ := 6370

/-- The length of an arc on a circle, given its radius and angle in radians -/
def arc_length (radius : ℝ) (angle : ℝ) : ℝ := radius * angle

/-- Theorem: The length of an arc on the Earth's equator corresponding to 1 radian 
    is equal to the radius of the Earth's equator -/
theorem equator_arc_length_equals_radius : 
  arc_length earth_radius 1 = earth_radius := by sorry

end NUMINAMATH_CALUDE_equator_arc_length_equals_radius_l866_86688


namespace NUMINAMATH_CALUDE_solution_of_equation_l866_86654

theorem solution_of_equation (x : ℝ) : 
  (Real.sqrt (3 * x + 1) + Real.sqrt (3 * x + 6) = Real.sqrt (4 * x - 2) + Real.sqrt (4 * x + 3)) → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l866_86654


namespace NUMINAMATH_CALUDE_yard_area_l866_86648

/-- The area of a rectangular yard with a square cut out -/
theorem yard_area (length width cut_side : ℝ) 
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : cut_side = 3) : 
  length * width - cut_side^2 = 171 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l866_86648


namespace NUMINAMATH_CALUDE_volume_P_5_l866_86671

/-- Represents the volume of the dodecahedron after i iterations -/
def P (i : ℕ) : ℚ :=
  sorry

/-- The height of the tetrahedra at step i -/
def r (i : ℕ) : ℚ :=
  (1 / 2) ^ i

/-- The initial dodecahedron has volume 1 -/
axiom P_0 : P 0 = 1

/-- The recursive definition of P(i+1) based on P(i) and r(i) -/
axiom P_step (i : ℕ) : P (i + 1) = P i + 6 * (1 / 3) * (r i)^3

/-- The main theorem: the volume of P₅ is 8929/4096 -/
theorem volume_P_5 : P 5 = 8929 / 4096 :=
  sorry

end NUMINAMATH_CALUDE_volume_P_5_l866_86671


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l866_86658

theorem negation_of_universal_positive_square_plus_one (p : Prop) :
  (p ↔ ∀ x : ℝ, x^2 + 1 > 0) →
  (¬p ↔ ∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l866_86658


namespace NUMINAMATH_CALUDE_sequence_properties_l866_86650

-- Define the sequence S_n
def S (n : ℕ+) : ℚ := n^2 + n

-- Define the sequence a_n
def a (n : ℕ+) : ℚ := 2 * n

-- Define the sequence T_n
def T (n : ℕ+) : ℚ := (1 / 2) * (1 - 1 / (n + 1))

theorem sequence_properties :
  (∀ n : ℕ+, S n = n^2 + n) →
  (∀ n : ℕ+, a n = 2 * n) ∧
  (∀ n : ℕ+, T n = (1 / 2) * (1 - 1 / (n + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l866_86650


namespace NUMINAMATH_CALUDE_factor_210_into_four_l866_86697

def prime_factors : Multiset ℕ := {2, 3, 5, 7}

/-- The number of ways to partition a multiset of 4 distinct elements into 4 non-empty subsets -/
def partition_count (m : Multiset ℕ) : ℕ := sorry

theorem factor_210_into_four : partition_count prime_factors = 15 := by sorry

end NUMINAMATH_CALUDE_factor_210_into_four_l866_86697


namespace NUMINAMATH_CALUDE_people_owning_cats_and_dogs_l866_86647

theorem people_owning_cats_and_dogs (
  total_pet_owners : ℕ)
  (only_dog_owners : ℕ)
  (only_cat_owners : ℕ)
  (cat_dog_snake_owners : ℕ)
  (h1 : total_pet_owners = 69)
  (h2 : only_dog_owners = 15)
  (h3 : only_cat_owners = 10)
  (h4 : cat_dog_snake_owners = 3) :
  total_pet_owners = only_dog_owners + only_cat_owners + 41 + cat_dog_snake_owners :=
by
  sorry

end NUMINAMATH_CALUDE_people_owning_cats_and_dogs_l866_86647


namespace NUMINAMATH_CALUDE_magical_stack_size_l866_86674

/-- A magical stack is a stack of cards where at least one card from each pile retains its original position after restacking. -/
def MagicalStack (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b ≤ n ∧ a ≠ b

/-- The position of a card in the original stack. -/
def OriginalPosition (card : ℕ) (n : ℕ) : ℕ :=
  if card ≤ n then card else card - n

/-- The position of a card in the restacked stack. -/
def RestackedPosition (card : ℕ) (n : ℕ) : ℕ :=
  2 * card - 1 - (if card ≤ n then 0 else 1)

/-- A card retains its position if its original position equals its restacked position. -/
def RetainsPosition (card : ℕ) (n : ℕ) : Prop :=
  OriginalPosition card n = RestackedPosition card n

theorem magical_stack_size :
  ∀ n : ℕ,
    MagicalStack n →
    RetainsPosition 111 n →
    RetainsPosition 90 n →
    2 * n ≥ 332 →
    2 * n = 332 :=
sorry

end NUMINAMATH_CALUDE_magical_stack_size_l866_86674


namespace NUMINAMATH_CALUDE_vector_sum_equals_one_five_l866_86699

/-- Given vectors a and b in R², prove that their sum is (1, 5) -/
theorem vector_sum_equals_one_five :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  (a + b) = ![1, 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equals_one_five_l866_86699


namespace NUMINAMATH_CALUDE_new_person_weight_is_87_l866_86678

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_group_size : ℕ) (average_increase : ℝ) (replaced_person_weight : ℝ) : ℝ :=
  replaced_person_weight + initial_group_size * average_increase

/-- Theorem stating that the weight of the new person is 87 kg -/
theorem new_person_weight_is_87 :
  new_person_weight 8 4 55 = 87 := by
  sorry

#eval new_person_weight 8 4 55

end NUMINAMATH_CALUDE_new_person_weight_is_87_l866_86678


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l866_86665

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1 ≤ x^2 ∧ x^2 ≤ 16) ∧ 
  (∃ x : ℝ, 1 ≤ x^2 ∧ x^2 ≤ 16 ∧ ¬(1 ≤ x ∧ x ≤ 4)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l866_86665


namespace NUMINAMATH_CALUDE_sam_seashells_l866_86605

def seashells_problem (yesterday_found : ℕ) (given_to_joan : ℕ) (today_found : ℕ) (given_to_tom : ℕ) : ℕ :=
  yesterday_found - given_to_joan + today_found - given_to_tom

theorem sam_seashells : seashells_problem 35 18 20 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l866_86605


namespace NUMINAMATH_CALUDE_number_of_female_democrats_l866_86673

/-- Given a meeting with male and female participants, prove the number of female Democrats --/
theorem number_of_female_democrats 
  (total_participants : ℕ) 
  (female_participants : ℕ) 
  (male_participants : ℕ) 
  (h1 : total_participants = 780)
  (h2 : female_participants + male_participants = total_participants)
  (h3 : 2 * (total_participants / 3) = female_participants / 2 + male_participants / 4) :
  female_participants / 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_number_of_female_democrats_l866_86673


namespace NUMINAMATH_CALUDE_f_three_point_five_l866_86622

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_neg (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def identity_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 < x → x < 1 → f x = x

theorem f_three_point_five 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : periodic_neg f) 
  (h_identity : identity_on_interval f) : 
  f 3.5 = -0.5 := by
sorry

end NUMINAMATH_CALUDE_f_three_point_five_l866_86622


namespace NUMINAMATH_CALUDE_triangle_properties_l866_86620

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  Real.cos C = 1 / 3 →
  c = 4 * Real.sqrt 2 →
  A = π / 3 ∧ 
  (1 / 2 * a * c * Real.sin B) = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l866_86620


namespace NUMINAMATH_CALUDE_import_tax_threshold_l866_86660

/-- Proves that the amount in excess of which a 7% import tax was applied is $1,000,
    given that the tax paid was $111.30 on an item with a total value of $2,590. -/
theorem import_tax_threshold (tax_rate : ℝ) (tax_paid : ℝ) (total_value : ℝ) :
  tax_rate = 0.07 →
  tax_paid = 111.30 →
  total_value = 2590 →
  ∃ (threshold : ℝ), threshold = 1000 ∧ tax_rate * (total_value - threshold) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_threshold_l866_86660


namespace NUMINAMATH_CALUDE_borrowed_amount_correct_l866_86695

/-- The amount of money Yoque borrowed -/
def borrowed_amount : ℝ := 150

/-- The number of months for repayment -/
def repayment_months : ℕ := 11

/-- The additional percentage added to the repayment -/
def additional_percentage : ℝ := 0.1

/-- The monthly payment amount -/
def monthly_payment : ℝ := 15

/-- Theorem stating that the borrowed amount satisfies the given conditions -/
theorem borrowed_amount_correct : 
  borrowed_amount * (1 + additional_percentage) = repayment_months * monthly_payment := by
  sorry


end NUMINAMATH_CALUDE_borrowed_amount_correct_l866_86695


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_1029_l866_86626

theorem least_n_factorial_divisible_by_1029 : 
  ∃ n : ℕ, n = 21 ∧ 
  (∀ k : ℕ, k < n → ¬(1029 ∣ k!)) ∧ 
  (1029 ∣ n!) := by
  sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_1029_l866_86626


namespace NUMINAMATH_CALUDE_goats_gifted_count_l866_86636

/-- Represents the number of goats gifted by Jeremy to Fred -/
def goats_gifted (initial_horses initial_sheep initial_chickens : ℕ) 
  (male_animals : ℕ) : ℕ :=
  let initial_total := initial_horses + initial_sheep + initial_chickens
  let after_brian_sale := initial_total - initial_total / 2
  let final_total := male_animals * 2
  final_total - after_brian_sale

/-- Theorem stating the number of goats gifted by Jeremy -/
theorem goats_gifted_count : 
  goats_gifted 100 29 9 53 = 37 := by
  sorry

#eval goats_gifted 100 29 9 53

end NUMINAMATH_CALUDE_goats_gifted_count_l866_86636


namespace NUMINAMATH_CALUDE_derivative_at_zero_l866_86625

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan ((3 * x / 2) - x^2 * Real.sin (1 / x))
  else 0

-- State the theorem
theorem derivative_at_zero (h : HasDerivAt f (3/2) 0) : 
  deriv f 0 = 3/2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l866_86625


namespace NUMINAMATH_CALUDE_sum_of_surds_simplification_l866_86642

theorem sum_of_surds_simplification : ∃ (a b d c : ℕ+),
  (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 + Real.sqrt 2 + 1 / Real.sqrt 2 =
   (a.val * Real.sqrt 3 + b.val * Real.sqrt 11 + d.val * Real.sqrt 2) / c.val) ∧
  (∀ (a' b' d' c' : ℕ+),
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 + Real.sqrt 2 + 1 / Real.sqrt 2 =
     (a'.val * Real.sqrt 3 + b'.val * Real.sqrt 11 + d'.val * Real.sqrt 2) / c'.val) →
    c'.val ≥ c.val) ∧
  (a.val + b.val + d.val + c.val = 325) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_surds_simplification_l866_86642


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l866_86614

/-- The sum of the series Σ(k=1 to ∞) 3^k / (9^k - 1) is equal to 1/2 -/
theorem series_sum_equals_half :
  ∑' k, (3 : ℝ)^k / (9^k - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l866_86614


namespace NUMINAMATH_CALUDE_berries_needed_for_cobbler_l866_86627

/-- Given that Maria needs a certain number of cartons of berries for a cobbler,
    and she already has some cartons of strawberries and blueberries,
    this theorem calculates how many more cartons she needs to buy. -/
theorem berries_needed_for_cobbler 
  (total_needed : ℕ) 
  (strawberries : ℕ) 
  (blueberries : ℕ) : 
  total_needed = 21 → 
  strawberries = 4 → 
  blueberries = 8 → 
  total_needed - (strawberries + blueberries) = 9 :=
by sorry

end NUMINAMATH_CALUDE_berries_needed_for_cobbler_l866_86627


namespace NUMINAMATH_CALUDE_blackjack_bet_l866_86602

theorem blackjack_bet (payout_ratio : Rat) (received_amount : ℚ) (original_bet : ℚ) : 
  payout_ratio = 3/2 →
  received_amount = 60 →
  received_amount = payout_ratio * original_bet →
  original_bet = 40 := by
sorry

end NUMINAMATH_CALUDE_blackjack_bet_l866_86602


namespace NUMINAMATH_CALUDE_plain_pancakes_count_l866_86646

/-- Given a total of 67 pancakes, with 20 having blueberries and 24 having bananas,
    prove that there are 23 plain pancakes. -/
theorem plain_pancakes_count (total : ℕ) (blueberry : ℕ) (banana : ℕ) 
  (h1 : total = 67) 
  (h2 : blueberry = 20) 
  (h3 : banana = 24) :
  total - (blueberry + banana) = 23 := by
  sorry

#check plain_pancakes_count

end NUMINAMATH_CALUDE_plain_pancakes_count_l866_86646


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_1320_l866_86634

theorem sum_of_distinct_prime_factors_1320 :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (1320 + 1)))
    (fun p => if p ∣ 1320 then p else 0)) = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_1320_l866_86634


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l866_86644

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47, the 8th term is 71. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h4 : a 4 = 23) (h6 : a 6 = 47) : a 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l866_86644


namespace NUMINAMATH_CALUDE_morgan_change_is_eleven_l866_86615

/-- The change Morgan receives after buying lunch -/
def morgan_change (hamburger_cost onion_rings_cost smoothie_cost bill_amount : ℕ) : ℕ :=
  bill_amount - (hamburger_cost + onion_rings_cost + smoothie_cost)

/-- Theorem stating that Morgan receives $11 in change -/
theorem morgan_change_is_eleven :
  morgan_change 4 2 3 20 = 11 := by
  sorry

end NUMINAMATH_CALUDE_morgan_change_is_eleven_l866_86615


namespace NUMINAMATH_CALUDE_parallelogram_base_l866_86693

theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 120 →
  height = 10 →
  area = base * height →
  base = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l866_86693


namespace NUMINAMATH_CALUDE_derivative_of_fraction_l866_86666

theorem derivative_of_fraction (x : ℝ) :
  let y : ℝ → ℝ := λ x => (1 - Real.cos (2 * x)) / (1 + Real.cos (2 * x))
  HasDerivAt y (4 * Real.sin (2 * x) / (1 + Real.cos (2 * x))^2) x :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_of_fraction_l866_86666


namespace NUMINAMATH_CALUDE_probability_of_white_marble_l866_86670

/-- Given a box of marbles with four colors, prove the probability of drawing a white marble. -/
theorem probability_of_white_marble (total_marbles : ℕ) 
  (p_green p_red_or_blue p_white : ℝ) : 
  total_marbles = 100 →
  p_green = 1/5 →
  p_red_or_blue = 0.55 →
  p_green + p_red_or_blue + p_white = 1 →
  p_white = 0.25 := by
  sorry

#check probability_of_white_marble

end NUMINAMATH_CALUDE_probability_of_white_marble_l866_86670


namespace NUMINAMATH_CALUDE_circle_line_intersection_l866_86616

theorem circle_line_intersection (α β : ℝ) (n k : ℤ) : 
  (∃ A B : ℝ × ℝ, 
    A = (Real.cos (2 * α), Real.cos (2 * β)) ∧ 
    B = (Real.cos (2 * β), Real.cos α) ∧
    (A = (-1/2, 0) ∧ B = (0, -1/2) ∨ A = (0, -1/2) ∧ B = (-1/2, 0))) →
  (α = 2 * Real.pi / 3 + 2 * Real.pi * ↑n ∨ 
   α = -2 * Real.pi / 3 + 2 * Real.pi * ↑n) ∧
  β = Real.pi / 4 + Real.pi / 2 * ↑k :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l866_86616


namespace NUMINAMATH_CALUDE_solve_system_l866_86637

theorem solve_system (c d : ℝ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 9 + c) : 
  5 - c = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l866_86637


namespace NUMINAMATH_CALUDE_edges_after_cutting_l866_86662

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  edges : ℕ

/-- Result of cutting off pyramids from a convex polyhedron. -/
def cutOffPyramids (P : ConvexPolyhedron) : ConvexPolyhedron :=
  ConvexPolyhedron.mk (3 * P.edges)

/-- Theorem stating the number of edges in the new polyhedron after cutting off pyramids. -/
theorem edges_after_cutting (P : ConvexPolyhedron) 
  (h : P.edges = 2021) : 
  (cutOffPyramids P).edges = 6063 := by
  sorry

end NUMINAMATH_CALUDE_edges_after_cutting_l866_86662


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_23_20_l866_86619

theorem sum_of_solutions_eq_23_20 : 
  let f : ℝ → ℝ := λ x => (5*x + 3) * (4*x - 7)
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ x + y = 23/20) := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_23_20_l866_86619


namespace NUMINAMATH_CALUDE_first_year_exceeding_2_million_is_correct_l866_86639

/-- The year when the R&D investment first exceeds 2 million yuan -/
def first_year_exceeding_2_million : ℕ := 2020

/-- The initial R&D investment in 2016 (in millions of yuan) -/
def initial_investment : ℝ := 1.3

/-- The annual increase rate of R&D investment -/
def annual_increase_rate : ℝ := 0.12

/-- The target R&D investment (in millions of yuan) -/
def target_investment : ℝ := 2.0

/-- Function to calculate the R&D investment for a given year -/
def investment_for_year (year : ℕ) : ℝ :=
  initial_investment * (1 + annual_increase_rate) ^ (year - 2016)

theorem first_year_exceeding_2_million_is_correct :
  (∀ y : ℕ, y < first_year_exceeding_2_million → investment_for_year y ≤ target_investment) ∧
  investment_for_year first_year_exceeding_2_million > target_investment :=
by sorry

end NUMINAMATH_CALUDE_first_year_exceeding_2_million_is_correct_l866_86639


namespace NUMINAMATH_CALUDE_function_is_constant_one_l866_86629

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ a b : ℕ, a > 0 ∧ b > 0 → f (a^2 + b^2) = f a * f b) ∧
  (∀ a : ℕ, a > 0 → f (a^2) = (f a)^2)

theorem function_is_constant_one (f : ℕ → ℕ) (h : is_valid_function f) :
  ∀ n : ℕ, n > 0 → f n = 1 :=
by sorry

end NUMINAMATH_CALUDE_function_is_constant_one_l866_86629


namespace NUMINAMATH_CALUDE_total_distance_triangle_l866_86672

theorem total_distance_triangle (XZ XY : ℝ) (h1 : XZ = 5000) (h2 : XY = 5200) :
  let YZ := Real.sqrt (XY ^ 2 - XZ ^ 2)
  XZ + XY + YZ = 11628 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_triangle_l866_86672


namespace NUMINAMATH_CALUDE_sourball_theorem_l866_86604

/-- The number of sourball candies Nellie can eat before crying -/
def nellies_candies : ℕ := 12

/-- The initial number of candies in the bucket -/
def initial_candies : ℕ := 30

/-- The number of candies each person gets after dividing the remaining candies -/
def remaining_candies_per_person : ℕ := 3

/-- The number of sourball candies Jacob can eat before crying -/
def jacobs_candies (n : ℕ) : ℕ := n / 2

/-- The number of sourball candies Lana can eat before crying -/
def lanas_candies (n : ℕ) : ℕ := jacobs_candies n - 3

theorem sourball_theorem : 
  nellies_candies + jacobs_candies nellies_candies + lanas_candies nellies_candies = 
  initial_candies - 3 * remaining_candies_per_person := by
  sorry

end NUMINAMATH_CALUDE_sourball_theorem_l866_86604


namespace NUMINAMATH_CALUDE_red_notebook_cost_l866_86680

/-- Proves that the cost of each red notebook is 4 dollars --/
theorem red_notebook_cost (total_spent : ℕ) (total_notebooks : ℕ) (red_notebooks : ℕ) 
  (green_notebooks : ℕ) (green_cost : ℕ) (blue_cost : ℕ) :
  total_spent = 37 →
  total_notebooks = 12 →
  red_notebooks = 3 →
  green_notebooks = 2 →
  green_cost = 2 →
  blue_cost = 3 →
  (total_spent - (green_notebooks * green_cost + (total_notebooks - red_notebooks - green_notebooks) * blue_cost)) / red_notebooks = 4 := by
sorry

end NUMINAMATH_CALUDE_red_notebook_cost_l866_86680
