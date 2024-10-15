import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l3963_396328

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (-2, 6)
def C : ℝ × ℝ := (8, 2)

-- Define the median from A to BC
def median_A_BC (x y : ℝ) : Prop :=
  y = 4

-- Define the perpendicular bisector of AC
def perp_bisector_AC (x y : ℝ) : Prop :=
  y = 4 * x - 13

-- Theorem statement
theorem triangle_properties :
  (∀ x y, median_A_BC x y ↔ y = 4) ∧
  (∀ x y, perp_bisector_AC x y ↔ y = 4 * x - 13) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3963_396328


namespace NUMINAMATH_CALUDE_min_portraits_theorem_l3963_396345

def min_year := 1600
def max_year := 2008
def max_age := 80

def ScientistData := {birth : ℕ // min_year ≤ birth ∧ birth ≤ max_year}

def death_year (s : ScientistData) : ℕ := s.val + (Nat.min max_age (max_year - s.val))

def product_ratio (scientists : List ScientistData) : ℚ :=
  (scientists.map death_year).prod / (scientists.map (λ s => s.val)).prod

theorem min_portraits_theorem :
  ∃ (scientists : List ScientistData),
    scientists.length = 5 ∧
    product_ratio scientists = 5/4 ∧
    ∀ (smaller_list : List ScientistData),
      smaller_list.length < 5 →
      product_ratio smaller_list < 5/4 :=
sorry

end NUMINAMATH_CALUDE_min_portraits_theorem_l3963_396345


namespace NUMINAMATH_CALUDE_power_of_ten_zeros_l3963_396356

theorem power_of_ten_zeros (n : ℕ) : ∃ k : ℕ, (5000^50) * 100^2 = k * 10^154 ∧ 10^154 ≤ k ∧ k < 10^155 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_zeros_l3963_396356


namespace NUMINAMATH_CALUDE_max_points_top_four_l3963_396333

/-- Represents a tournament with 8 teams -/
structure Tournament :=
  (teams : Fin 8)
  (games : Fin 8 → Fin 8 → Nat)
  (points : Fin 8 → Nat)

/-- The scoring system for the tournament -/
def score (result : Nat) : Nat :=
  match result with
  | 0 => 3  -- win
  | 1 => 1  -- draw
  | _ => 0  -- loss

/-- The theorem stating the maximum possible points for the top four teams -/
theorem max_points_top_four (t : Tournament) : 
  ∃ (a b c d : Fin 8), 
    (∀ i : Fin 8, t.points i ≤ t.points a) ∧
    (t.points a = t.points b) ∧
    (t.points b = t.points c) ∧
    (t.points c = t.points d) ∧
    (t.points a ≤ 33) :=
sorry

end NUMINAMATH_CALUDE_max_points_top_four_l3963_396333


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3963_396364

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| = x^2 - 5*x + 6 ∧ x = 2 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3963_396364


namespace NUMINAMATH_CALUDE_pharmacist_weights_exist_l3963_396361

theorem pharmacist_weights_exist : ∃ (w₁ w₂ w₃ : ℝ),
  w₁ < 90 ∧ w₂ < 90 ∧ w₃ < 90 ∧
  w₁ + w₂ + w₃ = 100 ∧
  w₁ + w₂ + (w₃ + 1) = 101 ∧
  w₂ + w₃ + (w₃ + 1) = 102 :=
by sorry

end NUMINAMATH_CALUDE_pharmacist_weights_exist_l3963_396361


namespace NUMINAMATH_CALUDE_T_divisibility_l3963_396319

def T : Set ℤ := {t | ∃ n : ℤ, t = n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2}

theorem T_divisibility :
  (∀ t ∈ T, ¬(9 ∣ t)) ∧ (∃ t ∈ T, 5 ∣ t) := by
  sorry

end NUMINAMATH_CALUDE_T_divisibility_l3963_396319


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l3963_396331

/-- Piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 2 * a else a^x

/-- Theorem stating the range of a for which f is increasing on ℝ -/
theorem f_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3/2 ≤ a ∧ a < 6) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l3963_396331


namespace NUMINAMATH_CALUDE_problem_solution_l3963_396354

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |e| = 4) : 
  e^2 - (a + b)^2022 + (-c * d)^2021 = 15 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3963_396354


namespace NUMINAMATH_CALUDE_expression_simplification_l3963_396386

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 - Real.sqrt 11) 
  (hb : b = Real.sqrt 3 + Real.sqrt 11) : 
  (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3963_396386


namespace NUMINAMATH_CALUDE_equilibrium_constant_temperature_relation_l3963_396351

-- Define the chemical equilibrium constant
variable (K : ℝ)

-- Define temperature
variable (T : ℝ)

-- Define a relation between K and T
def related_to_temperature (K T : ℝ) : Prop := sorry

-- Theorem stating that K is related to temperature
theorem equilibrium_constant_temperature_relation :
  related_to_temperature K T :=
sorry

end NUMINAMATH_CALUDE_equilibrium_constant_temperature_relation_l3963_396351


namespace NUMINAMATH_CALUDE_unique_magnitude_quadratic_l3963_396340

theorem unique_magnitude_quadratic :
  ∃! m : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = m := by sorry

end NUMINAMATH_CALUDE_unique_magnitude_quadratic_l3963_396340


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_l3963_396318

/-- Represents a club with members of different handedness and music preferences -/
structure Club where
  total : ℕ
  leftHanded : ℕ
  ambidextrous : ℕ
  rightHanded : ℕ
  jazzLovers : ℕ
  rightHandedJazzDislikers : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club) 
  (h1 : c.total = 30)
  (h2 : c.leftHanded = 12)
  (h3 : c.ambidextrous = 3)
  (h4 : c.rightHanded = c.total - c.leftHanded - c.ambidextrous)
  (h5 : c.jazzLovers = 20)
  (h6 : c.rightHandedJazzDislikers = 4) :
  ∃ x : ℕ, x = 6 ∧ 
    x ≤ c.leftHanded ∧ 
    x + (c.rightHanded - c.rightHandedJazzDislikers) + c.ambidextrous = c.jazzLovers :=
  sorry


end NUMINAMATH_CALUDE_left_handed_jazz_lovers_l3963_396318


namespace NUMINAMATH_CALUDE_sqrt_720_equals_12_sqrt_5_l3963_396341

theorem sqrt_720_equals_12_sqrt_5 : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_equals_12_sqrt_5_l3963_396341


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3963_396395

-- Define an isosceles triangle with side lengths 5, 5, and 2
def isoscelesTriangle (a b c : ℝ) : Prop :=
  a = 5 ∧ b = 5 ∧ c = 2

-- Define the perimeter of a triangle
def trianglePerimeter (a b c : ℝ) : ℝ :=
  a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, isoscelesTriangle a b c → trianglePerimeter a b c = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3963_396395


namespace NUMINAMATH_CALUDE_probability_negative_product_l3963_396311

def S : Finset Int := {-6, -3, -1, 2, 5, 8}

def negative_product_pairs (S : Finset Int) : Finset (Int × Int) :=
  S.product S |>.filter (fun (a, b) => a ≠ b ∧ a * b < 0)

def total_pairs (S : Finset Int) : Finset (Int × Int) :=
  S.product S |>.filter (fun (a, b) => a ≠ b)

theorem probability_negative_product :
  (negative_product_pairs S).card / (total_pairs S).card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_negative_product_l3963_396311


namespace NUMINAMATH_CALUDE_candy_distribution_l3963_396301

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (kiss_bags : ℕ) (jelly_bags : ℕ) :
  total_candy = 260 →
  total_bags = 13 →
  heart_bags = 4 →
  kiss_bags = 5 →
  jelly_bags = 3 →
  total_candy % total_bags = 0 →
  let pieces_per_bag := total_candy / total_bags
  let chew_bags := total_bags - heart_bags - kiss_bags - jelly_bags
  heart_bags * pieces_per_bag + chew_bags * pieces_per_bag + jelly_bags * pieces_per_bag = total_candy :=
by sorry

#check candy_distribution

end NUMINAMATH_CALUDE_candy_distribution_l3963_396301


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l3963_396342

theorem count_ordered_pairs : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 128) (Finset.product (Finset.range 129) (Finset.range 129))).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l3963_396342


namespace NUMINAMATH_CALUDE_max_areas_formula_l3963_396396

/-- Represents a circular disk with n equally spaced radii and one off-center chord -/
structure DividedDisk where
  n : ℕ  -- number of equally spaced radii
  has_off_center_chord : Bool

/-- Calculates the maximum number of non-overlapping areas in a divided disk -/
def max_areas (d : DividedDisk) : ℕ :=
  2 * d.n + 2

/-- Theorem: The maximum number of non-overlapping areas in a divided disk is 2n + 2 -/
theorem max_areas_formula (d : DividedDisk) (h : d.has_off_center_chord = true) :
  max_areas d = 2 * d.n + 2 := by
  sorry

#check max_areas_formula

end NUMINAMATH_CALUDE_max_areas_formula_l3963_396396


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3963_396374

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 4) + Real.sqrt (x + 6) = 12 ∧ x = 4465 / 144 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3963_396374


namespace NUMINAMATH_CALUDE_fewer_baseball_cards_l3963_396314

theorem fewer_baseball_cards (hockey football baseball : ℕ) 
  (h1 : baseball < football)
  (h2 : football = 4 * hockey)
  (h3 : hockey = 200)
  (h4 : baseball + football + hockey = 1750) :
  football - baseball = 50 := by
sorry

end NUMINAMATH_CALUDE_fewer_baseball_cards_l3963_396314


namespace NUMINAMATH_CALUDE_smallest_four_digit_arithmetic_sequence_l3963_396338

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  ∃ r : ℤ, b = a + r ∧ c = b + r ∧ d = c + r

def digits_are_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_arithmetic_sequence :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
    digits_are_distinct n ∧
    is_arithmetic_sequence (n / 1000 % 10) (n / 100 % 10) (n / 10 % 10) (n % 10) →
  1234 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_arithmetic_sequence_l3963_396338


namespace NUMINAMATH_CALUDE_velvet_for_hats_and_cloaks_l3963_396303

/-- The amount of velvet needed for hats and cloaks -/
def velvet_needed (hats_per_yard : ℚ) (yards_per_cloak : ℚ) (num_hats : ℚ) (num_cloaks : ℚ) : ℚ :=
  (num_hats / hats_per_yard) + (num_cloaks * yards_per_cloak)

/-- Theorem stating the total amount of velvet needed for 6 cloaks and 12 hats -/
theorem velvet_for_hats_and_cloaks :
  velvet_needed 4 3 12 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_velvet_for_hats_and_cloaks_l3963_396303


namespace NUMINAMATH_CALUDE_fifth_match_goals_is_five_l3963_396308

/-- The number of goals scored in the fifth match -/
def fifth_match_goals : ℕ := 5

/-- The total number of matches played -/
def total_matches : ℕ := 5

/-- The increase in average goals after the fifth match -/
def average_increase : ℚ := 1/5

/-- The total number of goals in all matches -/
def total_goals : ℕ := 21

/-- Theorem stating that the number of goals scored in the fifth match is 5 -/
theorem fifth_match_goals_is_five :
  fifth_match_goals = 5 ∧
  (fifth_match_goals : ℚ) + (total_goals - fifth_match_goals) = total_goals ∧
  (total_goals : ℚ) / total_matches = 
    ((total_goals - fifth_match_goals) : ℚ) / (total_matches - 1) + average_increase :=
by sorry

end NUMINAMATH_CALUDE_fifth_match_goals_is_five_l3963_396308


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3963_396383

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_common_difference 
  (a₁ d : ℚ) :
  arithmetic_sequence a₁ d 5 + arithmetic_sequence a₁ d 6 = -10 ∧
  sum_arithmetic_sequence a₁ d 14 = -14 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3963_396383


namespace NUMINAMATH_CALUDE_daycare_peas_preference_l3963_396339

theorem daycare_peas_preference (total : ℕ) (peas carrots corn : ℕ) : 
  total > 0 ∧
  carrots = 9 ∧
  corn = 5 ∧
  corn = (25 : ℕ) * total / 100 ∧
  total = peas + carrots + corn →
  peas = 6 := by
  sorry

end NUMINAMATH_CALUDE_daycare_peas_preference_l3963_396339


namespace NUMINAMATH_CALUDE_log_sum_equality_l3963_396352

theorem log_sum_equality : Real.log 0.01 / Real.log 10 + Real.log 16 / Real.log 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l3963_396352


namespace NUMINAMATH_CALUDE_identity_function_divisibility_l3963_396304

def is_divisible (a b : ℕ) : Prop := ∃ k, b = a * k

theorem identity_function_divisibility :
  ∀ f : ℕ+ → ℕ+, 
  (∀ x y : ℕ+, is_divisible (x.val * f x + y.val * f y) ((x.val^2 + y.val^2)^2022)) → 
  (∀ x : ℕ+, f x = x) := by sorry

end NUMINAMATH_CALUDE_identity_function_divisibility_l3963_396304


namespace NUMINAMATH_CALUDE_cube_holes_surface_area_222_l3963_396391

/-- Calculates the surface area of a cube with square holes cut through each face. -/
def cube_with_holes_surface_area (cube_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_edge^2
  let new_exposed_area := 6 * 4 * hole_edge^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 5 and square holes of side length 2
    has a total surface area of 222 square meters. -/
theorem cube_holes_surface_area_222 :
  cube_with_holes_surface_area 5 2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cube_holes_surface_area_222_l3963_396391


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l3963_396397

theorem simplify_product_of_square_roots (x : ℝ) (hx : x > 0) :
  Real.sqrt (56 * x^3) * Real.sqrt (10 * x^2) * Real.sqrt (63 * x^4) = 84 * x^4 * Real.sqrt (5 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l3963_396397


namespace NUMINAMATH_CALUDE_power_multiplication_l3963_396337

theorem power_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3963_396337


namespace NUMINAMATH_CALUDE_fair_haired_employees_percentage_l3963_396358

theorem fair_haired_employees_percentage :
  -- Define the total number of employees
  ∀ (total_employees : ℕ),
  total_employees > 0 →
  -- Define the number of women with fair hair
  ∀ (women_fair_hair : ℕ),
  women_fair_hair = (28 * total_employees) / 100 →
  -- Define the number of fair-haired employees
  ∀ (fair_haired_employees : ℕ),
  women_fair_hair = (40 * fair_haired_employees) / 100 →
  -- The percentage of employees with fair hair is 70%
  (fair_haired_employees : ℚ) / total_employees = 70 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_fair_haired_employees_percentage_l3963_396358


namespace NUMINAMATH_CALUDE_horizontal_cut_length_l3963_396326

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  area : ℝ

/-- A horizontal cut in an isosceles triangle -/
structure HorizontalCut where
  triangle : IsoscelesTriangle
  trapezoidArea : ℝ
  cutLength : ℝ

/-- The main theorem -/
theorem horizontal_cut_length 
  (triangle : IsoscelesTriangle)
  (cut : HorizontalCut)
  (h1 : triangle.area = 144)
  (h2 : triangle.height = 24)
  (h3 : cut.triangle = triangle)
  (h4 : cut.trapezoidArea = 108) :
  cut.cutLength = 6 := by
  sorry

end NUMINAMATH_CALUDE_horizontal_cut_length_l3963_396326


namespace NUMINAMATH_CALUDE_race_head_start_l3963_396355

/-- Proves that if A's speed is 32/27 times B's speed, then A needs to give B
    a head start of 5/32 of the race length for the race to end in a dead heat. -/
theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (32/27) * Vb) :
  (L / Va = (L - (5/32) * L) / Vb) := by
  sorry

end NUMINAMATH_CALUDE_race_head_start_l3963_396355


namespace NUMINAMATH_CALUDE_tangent_perpendicular_and_minimum_l3963_396362

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / (2 * x) + (3 / 2) * x + 1

theorem tangent_perpendicular_and_minimum (a : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a) ((a / x) - 1 / (2 * x^2) + 3 / 2) x) →
  (HasDerivAt (f a) 0 1) →
  a = -1 ∧
  ∀ x > 0, f (-1) x ≥ 3 ∧ f (-1) 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_and_minimum_l3963_396362


namespace NUMINAMATH_CALUDE_smallest_consecutive_sum_divisible_by_17_l3963_396325

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if two natural numbers are consecutive -/
def areConsecutive (a b : ℕ) : Prop := b = a + 1

/-- Check if there exist smaller consecutive numbers satisfying the condition -/
def existSmallerPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), x < a ∧ areConsecutive x y ∧ 
    (sumOfDigits x % 17 = 0) ∧ (sumOfDigits y % 17 = 0)

theorem smallest_consecutive_sum_divisible_by_17 :
  areConsecutive 8899 8900 ∧
  (sumOfDigits 8899 % 17 = 0) ∧
  (sumOfDigits 8900 % 17 = 0) ∧
  ¬(existSmallerPair 8899 8900) :=
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_sum_divisible_by_17_l3963_396325


namespace NUMINAMATH_CALUDE_floor_square_minus_floor_product_l3963_396357

theorem floor_square_minus_floor_product (x : ℝ) : x = 13.2 →
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_square_minus_floor_product_l3963_396357


namespace NUMINAMATH_CALUDE_fixed_point_on_all_lines_l3963_396315

/-- The fixed point through which all lines of a certain form pass -/
def fixed_point : ℝ × ℝ := (2, 1)

/-- The line equation parameterized by k -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 - 2 * k = 0

/-- Theorem stating that the fixed point lies on all lines of the given form -/
theorem fixed_point_on_all_lines :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) :=
by
  sorry

#check fixed_point_on_all_lines

end NUMINAMATH_CALUDE_fixed_point_on_all_lines_l3963_396315


namespace NUMINAMATH_CALUDE_edgar_cookies_count_l3963_396306

/-- The number of cookies a paper bag can hold -/
def cookies_per_bag : ℕ := 16

/-- The number of paper bags Edgar needs -/
def bags_needed : ℕ := 19

/-- The total number of cookies Edgar bought -/
def total_cookies : ℕ := cookies_per_bag * bags_needed

theorem edgar_cookies_count : total_cookies = 304 := by
  sorry

end NUMINAMATH_CALUDE_edgar_cookies_count_l3963_396306


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_simplify_with_condition_l3963_396312

-- Part 1
theorem simplify_and_evaluate (x y : ℤ) (h1 : x = -2) (h2 : y = -3) :
  x^2 - 2*(x^2 - 3*y) - 3*(2*x^2 + 5*y) = -1 := by sorry

-- Part 2
theorem simplify_with_condition (a b : ℝ) (h : a - b = 2*b^2) :
  2*(a^3 - 2*b^2) - (2*b - a) + a - 2*a^3 = 0 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_simplify_with_condition_l3963_396312


namespace NUMINAMATH_CALUDE_largest_gcd_of_ten_numbers_summing_to_1001_l3963_396344

theorem largest_gcd_of_ten_numbers_summing_to_1001 :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ),
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1001 ∧
    (∀ d : ℕ, d > 0 → d ∣ a₁ ∧ d ∣ a₂ ∧ d ∣ a₃ ∧ d ∣ a₄ ∧ d ∣ a₅ ∧
                      d ∣ a₆ ∧ d ∣ a₇ ∧ d ∣ a₈ ∧ d ∣ a₉ ∧ d ∣ a₁₀ → d ≤ 91) ∧
    91 ∣ a₁ ∧ 91 ∣ a₂ ∧ 91 ∣ a₃ ∧ 91 ∣ a₄ ∧ 91 ∣ a₅ ∧
    91 ∣ a₆ ∧ 91 ∣ a₇ ∧ 91 ∣ a₈ ∧ 91 ∣ a₉ ∧ 91 ∣ a₁₀ := by
  sorry

end NUMINAMATH_CALUDE_largest_gcd_of_ten_numbers_summing_to_1001_l3963_396344


namespace NUMINAMATH_CALUDE_alice_bob_meeting_l3963_396348

/-- The number of points on the circle -/
def n : ℕ := 12

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 4

/-- The relative movement of Alice to Bob per turn -/
def relative_move : ℤ := alice_move - (n - bob_move)

/-- The number of turns required for Alice and Bob to meet -/
def meeting_turns : ℕ := n

theorem alice_bob_meeting :
  (relative_move * meeting_turns) % n = 0 ∧
  ∀ k : ℕ, k < meeting_turns → (relative_move * k) % n ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meeting_l3963_396348


namespace NUMINAMATH_CALUDE_chairs_left_is_54_l3963_396363

/-- The number of chairs left in Rodrigo's classroom after borrowing -/
def chairs_left : ℕ :=
  let red_chairs : ℕ := 4
  let yellow_chairs : ℕ := 2 * red_chairs
  let blue_chairs : ℕ := 3 * yellow_chairs
  let green_chairs : ℕ := blue_chairs / 2
  let orange_chairs : ℕ := green_chairs + 2
  let total_chairs : ℕ := red_chairs + yellow_chairs + blue_chairs + green_chairs + orange_chairs
  let borrowed_chairs : ℕ := 5 + 3
  total_chairs - borrowed_chairs

theorem chairs_left_is_54 : chairs_left = 54 := by
  sorry

end NUMINAMATH_CALUDE_chairs_left_is_54_l3963_396363


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_nine_l3963_396322

/-- A triangle with consecutive integer side lengths where the smallest side is even. -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  is_even : Even a
  satisfies_triangle_inequality : a + (a + 1) > (a + 2) ∧ a + (a + 2) > (a + 1) ∧ (a + 1) + (a + 2) > a

/-- The perimeter of a ConsecutiveIntegerTriangle. -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ := t.a + (t.a + 1) + (t.a + 2)

/-- The smallest possible perimeter of a ConsecutiveIntegerTriangle is 9. -/
theorem smallest_perimeter_is_nine :
  ∀ t : ConsecutiveIntegerTriangle, perimeter t ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_nine_l3963_396322


namespace NUMINAMATH_CALUDE_equation_has_solution_in_interval_l3963_396371

theorem equation_has_solution_in_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^3 = 2^x := by sorry

end NUMINAMATH_CALUDE_equation_has_solution_in_interval_l3963_396371


namespace NUMINAMATH_CALUDE_haley_trees_l3963_396387

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 2

/-- The difference between survived trees and dead trees -/
def survival_difference : ℕ := 7

/-- The total number of trees Haley initially grew -/
def total_trees : ℕ := dead_trees + (dead_trees + survival_difference)

theorem haley_trees : total_trees = 11 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_l3963_396387


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l3963_396393

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The minimum distance between opposite edges -/
  d : ℝ
  /-- The length of the shortest height -/
  h : ℝ
  /-- d is positive -/
  d_pos : d > 0
  /-- h is positive -/
  h_pos : h > 0

/-- 
For any tetrahedron, twice the minimum distance between 
opposite edges is greater than the length of the shortest height
-/
theorem tetrahedron_inequality (t : Tetrahedron) : 2 * t.d > t.h := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l3963_396393


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3963_396365

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (9 * p^3 - 9 * q^3) * (p - q)⁻¹ = 49 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3963_396365


namespace NUMINAMATH_CALUDE_f_neg_two_eq_zero_l3963_396394

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2007 + b * x + 1

theorem f_neg_two_eq_zero (a b : ℝ) : f a b 2 = 2 → f a b (-2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_zero_l3963_396394


namespace NUMINAMATH_CALUDE_square_area_sum_l3963_396399

theorem square_area_sum (a b : ℕ) (h : a^2 + b^2 = 400) : a + b = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_area_sum_l3963_396399


namespace NUMINAMATH_CALUDE_students_exceed_pets_l3963_396302

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of rabbits in each classroom -/
def rabbits_per_classroom : ℕ := 3

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 5

/-- The theorem stating the difference between students and pets -/
theorem students_exceed_pets : 
  (num_classrooms * students_per_classroom) - 
  (num_classrooms * (rabbits_per_classroom + hamsters_per_classroom)) = 70 := by
  sorry

end NUMINAMATH_CALUDE_students_exceed_pets_l3963_396302


namespace NUMINAMATH_CALUDE_richsWalkDistance_l3963_396385

/-- Calculates the total distance Rich walks given his walking pattern --/
def richsWalk (houseToSidewalk : ℕ) (sidewalkToRoadEnd : ℕ) : ℕ :=
  let initialDistance := houseToSidewalk + sidewalkToRoadEnd
  let toIntersection := initialDistance * 2
  let toEndOfRoute := (initialDistance + toIntersection) / 2
  let oneWayDistance := initialDistance + toIntersection + toEndOfRoute
  oneWayDistance * 2

theorem richsWalkDistance :
  richsWalk 20 200 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_richsWalkDistance_l3963_396385


namespace NUMINAMATH_CALUDE_simplify_expression_l3963_396370

theorem simplify_expression (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^(2/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3963_396370


namespace NUMINAMATH_CALUDE_extended_segment_coordinates_l3963_396353

/-- Given two points A and B on a plane, and a point C such that BC = 2/3 * AB,
    this theorem proves that the coordinates of C can be determined. -/
theorem extended_segment_coordinates
  (A B : ℝ × ℝ)
  (hA : A = (-1, 3))
  (hB : B = (11, 7))
  (hC : ∃ C : ℝ × ℝ, (C.1 - B.1)^2 + (C.2 - B.2)^2 = (2/3)^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)) :
  ∃ C : ℝ × ℝ, C = (19, 29/3) :=
sorry

end NUMINAMATH_CALUDE_extended_segment_coordinates_l3963_396353


namespace NUMINAMATH_CALUDE_ryan_english_study_hours_l3963_396388

/-- Ryan's daily study schedule -/
structure StudySchedule where
  chinese_hours : ℕ
  english_hours : ℕ
  hours_difference : ℕ

/-- Theorem: Ryan's English study hours -/
theorem ryan_english_study_hours (schedule : StudySchedule)
  (h1 : schedule.chinese_hours = 5)
  (h2 : schedule.hours_difference = 2)
  (h3 : schedule.english_hours = schedule.chinese_hours + schedule.hours_difference) :
  schedule.english_hours = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_study_hours_l3963_396388


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l3963_396327

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | |x| < 2}

def set_difference (X Y : Set ℤ) : Set ℤ := {x : ℤ | x ∈ X ∧ x ∉ Y}
def symmetric_difference (X Y : Set ℤ) : Set ℤ := (set_difference X Y) ∪ (set_difference Y X)

theorem symmetric_difference_A_B :
  symmetric_difference A B = {-1, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l3963_396327


namespace NUMINAMATH_CALUDE_linda_college_applications_l3963_396360

def number_of_colleges (hourly_rate : ℚ) (application_fee : ℚ) (hours_worked : ℚ) : ℚ :=
  (hourly_rate * hours_worked) / application_fee

theorem linda_college_applications : number_of_colleges 10 25 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_linda_college_applications_l3963_396360


namespace NUMINAMATH_CALUDE_difference_of_squares_303_297_l3963_396323

theorem difference_of_squares_303_297 : 303^2 - 297^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_303_297_l3963_396323


namespace NUMINAMATH_CALUDE_log_difference_equals_negative_one_l3963_396384

theorem log_difference_equals_negative_one :
  ∀ (log : ℝ → ℝ → ℝ),
    (∀ (a N : ℝ), a > 0 → a ≠ 1 → ∃ b, N = a^b → log a N = b) →
    9 = 3^2 →
    125 = 5^3 →
    log 3 9 - log 5 125 = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_negative_one_l3963_396384


namespace NUMINAMATH_CALUDE_girls_in_class_l3963_396316

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (num_girls : ℕ) :
  total = 35 →
  ratio_girls = 3 →
  ratio_boys = 4 →
  ratio_girls + ratio_boys = num_girls + (total - num_girls) →
  num_girls * ratio_boys = (total - num_girls) * ratio_girls →
  num_girls = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l3963_396316


namespace NUMINAMATH_CALUDE_f_is_integer_l3963_396307

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def f (m n : ℕ) : ℚ :=
  (factorial (2 * m) * factorial (2 * n)) / (factorial m * factorial n * factorial (m + n))

theorem f_is_integer (m n : ℕ) : ∃ k : ℤ, f m n = k :=
sorry

end NUMINAMATH_CALUDE_f_is_integer_l3963_396307


namespace NUMINAMATH_CALUDE_pool_attendance_difference_l3963_396347

theorem pool_attendance_difference (total : ℕ) (day1 : ℕ) (day3 : ℕ) 
  (h1 : total = 246)
  (h2 : day1 = 79)
  (h3 : day3 = 120)
  (h4 : total = day1 + day3 + (total - day1 - day3)) :
  (total - day1 - day3) - day3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_pool_attendance_difference_l3963_396347


namespace NUMINAMATH_CALUDE_hundred_to_fifty_zeros_l3963_396309

theorem hundred_to_fifty_zeros (n : ℕ) : 100^50 = 10^100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_to_fifty_zeros_l3963_396309


namespace NUMINAMATH_CALUDE_sequence_and_max_sum_l3963_396378

def f (x : ℝ) := -x^2 + 7*x

def S (n : ℕ) := f n

def a (n : ℕ+) := S n - S (n-1)

theorem sequence_and_max_sum :
  (∀ n : ℕ+, a n = -2*(n:ℝ) + 8) ∧
  (∃ n : ℕ+, S n = 12 ∧ ∀ m : ℕ+, S m ≤ 12) :=
sorry

end NUMINAMATH_CALUDE_sequence_and_max_sum_l3963_396378


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l3963_396368

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  1 / (a + 1) + 1 / b ≥ 1 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧ 1 / (a₀ + 1) + 1 / b₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l3963_396368


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3963_396330

theorem geometric_sequence_seventh_term
  (a : ℝ) (r : ℝ)
  (positive_sequence : ∀ n : ℕ, a * r ^ (n - 1) > 0)
  (fourth_term : a * r^3 = 16)
  (tenth_term : a * r^9 = 2) :
  a * r^6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3963_396330


namespace NUMINAMATH_CALUDE_john_light_bulbs_left_l3963_396359

/-- The number of light bulbs John has left after using some and giving away half of the remainder --/
def lightBulbsLeft (initial : ℕ) (used : ℕ) : ℕ :=
  let remaining := initial - used
  remaining - remaining / 2

/-- Theorem stating that John has 12 light bulbs left --/
theorem john_light_bulbs_left :
  lightBulbsLeft 40 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_light_bulbs_left_l3963_396359


namespace NUMINAMATH_CALUDE_f_increasing_and_range_l3963_396392

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_increasing_and_range :
  (∀ a : ℝ, Monotone (f a)) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x = -(f a (-x))) →
    Set.range (f a) = Set.Ioo (-1/2 : ℝ) (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_and_range_l3963_396392


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3963_396373

/-- The system of equations has a nontrivial solution with the given ratio -/
theorem system_solution_ratio :
  ∃ (x y z : ℚ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  x + (95/9)*y + 4*z = 0 ∧
  4*x + (95/9)*y - 3*z = 0 ∧
  3*x + 5*y - 4*z = 0 ∧
  x*z / (y^2) = 175/81 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3963_396373


namespace NUMINAMATH_CALUDE_salary_problem_l3963_396390

theorem salary_problem (A B : ℝ) 
  (h1 : A + B = 4000)
  (h2 : A * 0.05 = B * 0.15) :
  A = 3000 := by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l3963_396390


namespace NUMINAMATH_CALUDE_max_sum_with_lcm_gcd_l3963_396350

theorem max_sum_with_lcm_gcd (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 140) 
  (h_gcd : Nat.gcd a b = 5) : 
  a + b ≤ 145 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_lcm_gcd_l3963_396350


namespace NUMINAMATH_CALUDE_dice_sum_product_l3963_396313

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 180 →
  a + b + c + d ≠ 14 ∧ a + b + c + d ≠ 17 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_product_l3963_396313


namespace NUMINAMATH_CALUDE_roots_of_unity_quadratic_equation_l3963_396329

theorem roots_of_unity_quadratic_equation :
  ∃! (S : Finset ℂ),
    (∀ z ∈ S, (Complex.abs z = 1) ∧
      (∃ a : ℤ, z ^ 2 + a * z + 1 = 0 ∧
        -2 ≤ a ∧ a ≤ 2 ∧
        ∃ k : ℤ, a = k * Real.cos (k * π / 6))) ∧
    Finset.card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_unity_quadratic_equation_l3963_396329


namespace NUMINAMATH_CALUDE_dave_derek_money_difference_l3963_396380

/-- Calculates the difference between Dave's and Derek's remaining money after expenses -/
def moneyDifference (derekInitial : ℕ) (derekExpenses : List ℕ) (daveInitial : ℕ) (daveExpenses : List ℕ) : ℕ :=
  let derekRemaining := derekInitial - derekExpenses.sum
  let daveRemaining := daveInitial - daveExpenses.sum
  daveRemaining - derekRemaining

/-- Proves that Dave has $20 more left than Derek after expenses -/
theorem dave_derek_money_difference :
  moneyDifference 40 [14, 11, 5, 8] 50 [7, 12, 9] = 20 := by
  sorry

#eval moneyDifference 40 [14, 11, 5, 8] 50 [7, 12, 9]

end NUMINAMATH_CALUDE_dave_derek_money_difference_l3963_396380


namespace NUMINAMATH_CALUDE_intersection_condition_l3963_396349

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x + 1 < 3}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem intersection_condition (a : ℝ) : M ∩ N a = N a → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3963_396349


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l3963_396381

theorem factorization_of_quadratic (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l3963_396381


namespace NUMINAMATH_CALUDE_intersection_line_slope_l3963_396324

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 12 = 0

-- Define the intersection points
def intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem stating that the slope of the line connecting intersection points is 1
theorem intersection_line_slope :
  ∃ (x1 y1 x2 y2 : ℝ),
    intersection_points x1 y1 ∧
    intersection_points x2 y2 ∧
    x1 ≠ x2 →
    (y2 - y1) / (x2 - x1) = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l3963_396324


namespace NUMINAMATH_CALUDE_candy_distribution_totals_l3963_396317

/-- Represents the number of candies each child has -/
structure CandyDistribution where
  vitya : Nat
  masha : Nat
  sasha : Nat

/-- Checks if a candy distribution satisfies the given conditions -/
def isValidDistribution (d : CandyDistribution) : Prop :=
  d.vitya = 5 ∧ d.masha < d.vitya ∧ d.sasha = d.vitya + d.masha

/-- Calculates the total number of candies for a distribution -/
def totalCandies (d : CandyDistribution) : Nat :=
  d.vitya + d.masha + d.sasha

/-- The set of possible total numbers of candies -/
def possibleTotals : Set Nat := {18, 16, 14, 12}

/-- Theorem stating that the possible total numbers of candies are 18, 16, 14, and 12 -/
theorem candy_distribution_totals :
  ∀ d : CandyDistribution, isValidDistribution d →
    totalCandies d ∈ possibleTotals := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_totals_l3963_396317


namespace NUMINAMATH_CALUDE_function_transformation_l3963_396389

/-- Given a function f(x) = 3sin(2x + φ) where φ ∈ (0, π/2), if the graph of f(x) is translated
    left by π/6 units and is symmetric about the y-axis, then f(x) = 3sin(2x + π/6). -/
theorem function_transformation (φ : Real) (h1 : φ > 0) (h2 : φ < π/2) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (2 * x + φ)
  let g : ℝ → ℝ := λ x ↦ f (x + π/6)
  (∀ x, g x = g (-x)) →  -- Symmetry about y-axis
  (∀ x, f x = 3 * Real.sin (2 * x + π/6)) := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l3963_396389


namespace NUMINAMATH_CALUDE_potion_price_l3963_396332

theorem potion_price (current_price : ℚ) (original_price : ℚ) : 
  current_price = 9 → current_price = (1 / 15) * original_price → original_price = 135 := by
  sorry

end NUMINAMATH_CALUDE_potion_price_l3963_396332


namespace NUMINAMATH_CALUDE_pear_sales_ratio_l3963_396320

/-- Given the total amount of pears sold in a day and the amount sold in the afternoon,
    prove that the ratio of pears sold in the afternoon to pears sold in the morning is 2:1. -/
theorem pear_sales_ratio (total : ℕ) (afternoon : ℕ) 
    (h1 : total = 510)
    (h2 : afternoon = 340) : 
  (afternoon : ℚ) / ((total - afternoon) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pear_sales_ratio_l3963_396320


namespace NUMINAMATH_CALUDE_vacation_photos_count_l3963_396336

/-- The number of photos Alyssa took on vacation -/
def total_photos : ℕ := 100

/-- The total number of pages in the album -/
def total_pages : ℕ := 30

/-- The number of photos that can be placed on each of the first 10 pages -/
def photos_per_page_first : ℕ := 3

/-- The number of photos that can be placed on each of the next 10 pages -/
def photos_per_page_second : ℕ := 4

/-- The number of photos that can be placed on each of the remaining pages -/
def photos_per_page_last : ℕ := 3

/-- The number of pages in the first section -/
def pages_first_section : ℕ := 10

/-- The number of pages in the second section -/
def pages_second_section : ℕ := 10

theorem vacation_photos_count : 
  total_photos = 
    photos_per_page_first * pages_first_section + 
    photos_per_page_second * pages_second_section + 
    photos_per_page_last * (total_pages - pages_first_section - pages_second_section) :=
by sorry

end NUMINAMATH_CALUDE_vacation_photos_count_l3963_396336


namespace NUMINAMATH_CALUDE_min_value_expression_l3963_396321

theorem min_value_expression (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - b)^2) / c^2 ≥ 0 ∧
  ∃ a b, ((a + b)^2 + (b - c)^2 + (c - b)^2) / c^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3963_396321


namespace NUMINAMATH_CALUDE_base_eight_sum_l3963_396372

theorem base_eight_sum (A B C : ℕ) : 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A < 8 ∧ B < 8 ∧ C < 8 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (A * 8^2 + B * 8 + C) + (B * 8^2 + C * 8 + A) + (C * 8^2 + A * 8 + B) = A * (8^3 + 8^2 + 8) →
  B + C = 7 :=
by sorry

end NUMINAMATH_CALUDE_base_eight_sum_l3963_396372


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3963_396300

/-- Given a cylinder with volume 72π cm³ and a cone with the same radius as the cylinder 
    and half its height, the volume of the cone is 12π cm³ -/
theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  π * r^2 * h = 72 * π → (1/3) * π * r^2 * (h/2) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3963_396300


namespace NUMINAMATH_CALUDE_total_vegetables_collected_schoolchildren_vegetable_collection_l3963_396377

/-- Represents the amount of vegetables collected by each grade -/
structure VegetableCollection where
  fourth_cabbage : ℕ
  fourth_carrots : ℕ
  fifth_cucumbers : ℕ
  sixth_cucumbers : ℕ
  sixth_onions : ℕ

/-- Theorem stating the total amount of vegetables collected -/
theorem total_vegetables_collected (vc : VegetableCollection) : ℕ :=
  vc.fourth_cabbage + vc.fourth_carrots + vc.fifth_cucumbers + vc.sixth_cucumbers + vc.sixth_onions

/-- Main theorem proving the total amount of vegetables collected is 49 centners -/
theorem schoolchildren_vegetable_collection : 
  ∃ (vc : VegetableCollection), 
    vc.fourth_cabbage = 18 ∧ 
    vc.fourth_carrots = vc.sixth_onions ∧
    vc.fifth_cucumbers < vc.sixth_cucumbers ∧
    vc.fifth_cucumbers > vc.fourth_carrots ∧
    vc.sixth_onions = 7 ∧
    vc.sixth_cucumbers = vc.fourth_cabbage / 2 ∧
    total_vegetables_collected vc = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetables_collected_schoolchildren_vegetable_collection_l3963_396377


namespace NUMINAMATH_CALUDE_a_between_3_and_5_necessary_not_sufficient_l3963_396375

/-- The equation of a potential ellipse -/
def ellipse_equation (a x y : ℝ) : Prop :=
  x^2 / (a - 3) + y^2 / (5 - a) = 1

/-- The condition that a is between 3 and 5 -/
def a_between_3_and_5 (a : ℝ) : Prop :=
  3 < a ∧ a < 5

/-- The statement that the equation represents an ellipse -/
def is_ellipse (a : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_equation a x y ∧ (x ≠ 0 ∨ y ≠ 0)

/-- The main theorem: a_between_3_and_5 is necessary but not sufficient for is_ellipse -/
theorem a_between_3_and_5_necessary_not_sufficient :
  (∀ a : ℝ, is_ellipse a → a_between_3_and_5 a) ∧
  ¬(∀ a : ℝ, a_between_3_and_5 a → is_ellipse a) :=
sorry

end NUMINAMATH_CALUDE_a_between_3_and_5_necessary_not_sufficient_l3963_396375


namespace NUMINAMATH_CALUDE_ExistsFourDigitNumberDivisibleBy11WithDigitSum10_l3963_396310

-- Define a four-digit number
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- Define the sum of digits
def SumOfDigits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

-- Theorem statement
theorem ExistsFourDigitNumberDivisibleBy11WithDigitSum10 :
  ∃ n : ℕ, FourDigitNumber n ∧ SumOfDigits n = 10 ∧ n % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ExistsFourDigitNumberDivisibleBy11WithDigitSum10_l3963_396310


namespace NUMINAMATH_CALUDE_safe_access_theorem_access_conditions_l3963_396382

/-- Represents the number of members in the commission -/
def commission_size : ℕ := 11

/-- Represents the minimum number of members needed for access -/
def min_access : ℕ := 6

/-- Calculates the number of locks needed -/
def num_locks : ℕ := Nat.choose commission_size (min_access - 1)

/-- Calculates the number of keys each member should have -/
def keys_per_member : ℕ := num_locks * min_access / commission_size

/-- Theorem stating the correct number of locks and keys per member -/
theorem safe_access_theorem :
  num_locks = 462 ∧ keys_per_member = 252 :=
sorry

/-- Theorem proving that the arrangement satisfies the access conditions -/
theorem access_conditions (members : Finset (Fin commission_size)) :
  (members.card ≥ min_access → ∃ (lock : Fin num_locks), ∀ k ∈ members, k.val < keys_per_member) ∧
  (members.card < min_access → ∃ (lock : Fin num_locks), ∀ k ∈ members, k.val ≥ keys_per_member) :=
sorry

end NUMINAMATH_CALUDE_safe_access_theorem_access_conditions_l3963_396382


namespace NUMINAMATH_CALUDE_inequality_chain_l3963_396367

theorem inequality_chain (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3963_396367


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3963_396346

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) : 
  leg = 15 → angle = 45 → ∃ (hypotenuse : ℝ), hypotenuse = 15 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3963_396346


namespace NUMINAMATH_CALUDE_range_of_t_l3963_396334

theorem range_of_t (x y a : ℝ) 
  (eq1 : x + 3 * y + a = 4)
  (eq2 : x - y - 3 * a = 0)
  (bounds : -1 ≤ a ∧ a ≤ 1) :
  let t := x + y
  1 ≤ t ∧ t ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_t_l3963_396334


namespace NUMINAMATH_CALUDE_river_width_l3963_396343

/-- Two ferries traveling between opposite banks of a river -/
structure FerrySystem where
  /-- Width of the river -/
  width : ℝ
  /-- Distance from one bank where ferries first meet -/
  first_meeting : ℝ
  /-- Distance from the other bank where ferries second meet -/
  second_meeting : ℝ

/-- Theorem stating the width of the river given the meeting points -/
theorem river_width (fs : FerrySystem) 
    (h1 : fs.first_meeting = 700)
    (h2 : fs.second_meeting = 400) : 
    fs.width = 1700 := by
  sorry

#check river_width

end NUMINAMATH_CALUDE_river_width_l3963_396343


namespace NUMINAMATH_CALUDE_acute_triangle_sine_cosine_inequality_l3963_396335

theorem acute_triangle_sine_cosine_inequality 
  (A B C : Real) 
  (h_acute : A ∈ Set.Ioo 0 (π/2) ∧ B ∈ Set.Ioo 0 (π/2) ∧ C ∈ Set.Ioo 0 (π/2)) 
  (h_sum : A + B + C = π) : 
  Real.sin A + Real.sin B > Real.cos A + Real.cos B + Real.cos C := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_cosine_inequality_l3963_396335


namespace NUMINAMATH_CALUDE_square_area_perimeter_ratio_l3963_396305

theorem square_area_perimeter_ratio : 
  ∀ (a b : ℝ), a > 0 → b > 0 → (a^2 / b^2 = 49 / 64) → (4*a / (4*b) = 7 / 8) := by
  sorry

end NUMINAMATH_CALUDE_square_area_perimeter_ratio_l3963_396305


namespace NUMINAMATH_CALUDE_existence_of_divalent_radical_with_bounded_growth_l3963_396369

/-- A set of positive integers is a divalent radical if any sufficiently large positive integer
    can be expressed as the sum of two elements in the set. -/
def IsDivalentRadical (A : Set ℕ+) : Prop :=
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → ∃ a b : ℕ+, a ∈ A ∧ b ∈ A ∧ (a : ℕ) + b = n

/-- A(x) is the set of all elements in A that do not exceed x -/
def ASubset (A : Set ℕ+) (x : ℝ) : Set ℕ+ :=
  {a ∈ A | (a : ℝ) ≤ x}

theorem existence_of_divalent_radical_with_bounded_growth :
  ∃ (A : Set ℕ+) (C : ℝ), A.Nonempty ∧ IsDivalentRadical A ∧
    ∀ x : ℝ, x ≥ 1 → (ASubset A x).ncard ≤ C * Real.sqrt x :=
sorry

end NUMINAMATH_CALUDE_existence_of_divalent_radical_with_bounded_growth_l3963_396369


namespace NUMINAMATH_CALUDE_speed_limit_violation_percentage_l3963_396366

theorem speed_limit_violation_percentage
  (total_motorists : ℝ)
  (h1 : total_motorists > 0)
  (ticketed_percentage : ℝ)
  (h2 : ticketed_percentage = 40)
  (unticketed_speeders_percentage : ℝ)
  (h3 : unticketed_speeders_percentage = 20)
  : (ticketed_percentage / (100 - unticketed_speeders_percentage)) * 100 = 50 := by
  sorry

#check speed_limit_violation_percentage

end NUMINAMATH_CALUDE_speed_limit_violation_percentage_l3963_396366


namespace NUMINAMATH_CALUDE_arithmetic_progression_max_first_term_l3963_396379

theorem arithmetic_progression_max_first_term 
  (b₁ : ℚ) 
  (d : ℚ) 
  (S₄ S₉ : ℕ) :
  (4 * b₁ + 6 * d = S₄) →
  (9 * b₁ + 36 * d = S₉) →
  (b₁ ≤ 3/4) →
  (∀ b₁' d' S₄' S₉' : ℚ, 
    (4 * b₁' + 6 * d' = S₄') →
    (9 * b₁' + 36 * d' = S₉') →
    (b₁' ≤ 3/4) →
    (b₁' ≤ b₁)) →
  b₁ = 11/15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_max_first_term_l3963_396379


namespace NUMINAMATH_CALUDE_number_multiples_l3963_396398

def is_valid_number (n : ℕ) : Prop :=
  ∃ (A B C D E F : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    n = A * 100000 + B * 10000 + C * 1000 + D * 100 + E * 10 + F

def satisfies_conditions (n : ℕ) : Prop :=
  is_valid_number n ∧
  ∃ (A B C D E F : ℕ),
    4 * n = A * 100000 + B * 10000 + C * 1000 + D * 100 + E * 10 + F ∧
    13 * n = F * 100000 + A * 10000 + B * 1000 + C * 100 + D * 10 + E ∧
    22 * n = C * 100000 + D * 10000 + E * 1000 + F * 100 + A * 10 + B

theorem number_multiples (n : ℕ) (h : satisfies_conditions n) :
  (∃ k : ℕ, n * k = 984126) ∧
  (∀ k : ℕ, n * k ≠ 269841) ∧
  (∀ k : ℕ, n * k ≠ 841269) :=
sorry

end NUMINAMATH_CALUDE_number_multiples_l3963_396398


namespace NUMINAMATH_CALUDE_nonnegative_inequality_l3963_396376

theorem nonnegative_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a * (a - b) * (a - 2 * b) + b * (b - c) * (b - 2 * c) + c * (c - a) * (c - 2 * a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_inequality_l3963_396376
