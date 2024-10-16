import Mathlib

namespace NUMINAMATH_CALUDE_max_brand_A_is_15_l2118_211838

/-- The price difference between brand A and brand B soccer balls -/
def price_difference : ℕ := 10

/-- The number of brand A soccer balls in the initial purchase -/
def initial_brand_A : ℕ := 20

/-- The number of brand B soccer balls in the initial purchase -/
def initial_brand_B : ℕ := 15

/-- The total cost of the initial purchase -/
def initial_total_cost : ℕ := 3350

/-- The total number of soccer balls to be purchased -/
def total_balls : ℕ := 50

/-- The maximum total cost for the new purchase -/
def max_total_cost : ℕ := 4650

/-- The price of a brand A soccer ball -/
def price_A : ℕ := initial_total_cost / (initial_brand_A + initial_brand_B)

/-- The price of a brand B soccer ball -/
def price_B : ℕ := price_A - price_difference

/-- The maximum number of brand A soccer balls that can be purchased -/
def max_brand_A : ℕ := (max_total_cost - price_B * total_balls) / (price_A - price_B)

theorem max_brand_A_is_15 : max_brand_A = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_brand_A_is_15_l2118_211838


namespace NUMINAMATH_CALUDE_planning_committee_subcommittees_l2118_211888

theorem planning_committee_subcommittees (total_members : ℕ) (professor_count : ℕ) (subcommittee_size : ℕ) : 
  total_members = 12 →
  professor_count = 5 →
  subcommittee_size = 5 →
  (Nat.choose total_members subcommittee_size) - (Nat.choose (total_members - professor_count) subcommittee_size) = 771 :=
by sorry

end NUMINAMATH_CALUDE_planning_committee_subcommittees_l2118_211888


namespace NUMINAMATH_CALUDE_colored_paper_problem_l2118_211810

/-- The number of pieces of colored paper Yuna had initially -/
def yunas_initial_paper : ℕ := 150

/-- The number of pieces of colored paper Namjoon had initially -/
def namjoons_initial_paper : ℕ := 250

/-- The number of pieces of colored paper Namjoon gave to Yuna -/
def paper_given : ℕ := 60

/-- The difference in paper count between Yuna and Namjoon after the exchange -/
def paper_difference : ℕ := 20

theorem colored_paper_problem :
  yunas_initial_paper = 150 ∧
  namjoons_initial_paper = 250 ∧
  paper_given = 60 ∧
  paper_difference = 20 →
  yunas_initial_paper + paper_given = namjoons_initial_paper - paper_given + paper_difference :=
by sorry

end NUMINAMATH_CALUDE_colored_paper_problem_l2118_211810


namespace NUMINAMATH_CALUDE_tank_dimension_proof_l2118_211871

/-- Proves that for a rectangular tank with given dimensions and insulation cost,
    the third dimension is 3 feet. -/
theorem tank_dimension_proof (x : ℝ) : 
  let length : ℝ := 4
  let width : ℝ := 5
  let cost_per_sqft : ℝ := 20
  let total_cost : ℝ := 1880
  let surface_area : ℝ := 2 * (length * width + length * x + width * x)
  surface_area * cost_per_sqft = total_cost → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_tank_dimension_proof_l2118_211871


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l2118_211870

theorem no_function_satisfies_condition :
  ∀ (f : ℝ → ℝ), ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ f (x + y^2) < f x + y :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l2118_211870


namespace NUMINAMATH_CALUDE_star_property_counterexample_l2118_211895

/-- Definition of the star operation -/
def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

/-- Theorem stating that 2(x ★ y) ≠ (2x) ★ (2y) for some real x and y -/
theorem star_property_counterexample : ∃ x y : ℝ, 2 * (star x y) ≠ star (2*x) (2*y) := by
  sorry

end NUMINAMATH_CALUDE_star_property_counterexample_l2118_211895


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l2118_211827

theorem missing_fraction_sum (x : ℚ) : 
  x = 7/15 → 
  (1/3 : ℚ) + (1/2 : ℚ) + (-5/6 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + x = 
  (13333333333333333 : ℚ) / (100000000000000000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l2118_211827


namespace NUMINAMATH_CALUDE_solution_product_l2118_211833

theorem solution_product (a b : ℝ) : 
  (a - 3) * (2 * a + 7) = a^2 - 11 * a + 28 →
  (b - 3) * (2 * b + 7) = b^2 - 11 * b + 28 →
  a ≠ b →
  (a + 2) * (b + 2) = -66 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2118_211833


namespace NUMINAMATH_CALUDE_set_operations_l2118_211819

def A (x : ℝ) : Set ℝ := {0, |x|}
def B : Set ℝ := {1, 0, -1}

theorem set_operations (x : ℝ) (h : A x ⊆ B) :
  (A x ∩ B = {0, 1}) ∧
  (A x ∪ B = {-1, 0, 1}) ∧
  (B \ A x = {-1}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l2118_211819


namespace NUMINAMATH_CALUDE_rotated_semicircle_area_l2118_211824

/-- The area of a figure formed by rotating a semicircle around one of its ends by 45 degrees -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let α : Real := π / 4  -- 45 degrees in radians
  let semicircle_area := π * R^2 / 2
  let rotated_area := (2 * R)^2 * α / 2
  rotated_area = semicircle_area :=
by sorry

end NUMINAMATH_CALUDE_rotated_semicircle_area_l2118_211824


namespace NUMINAMATH_CALUDE_triangle_side_mod_three_l2118_211812

/-- 
Given two triangles with the same perimeter, where the first is equilateral
with integer side lengths and the second has integer side lengths with one
side of length 1 and another of length d, then d ≡ 1 (mod 3).
-/
theorem triangle_side_mod_three (a d : ℕ) : 
  (3 * a = 2 * d + 1) → (d % 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_mod_three_l2118_211812


namespace NUMINAMATH_CALUDE_three_zero_points_range_l2118_211840

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

-- State the theorem
theorem three_zero_points_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a > -2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_zero_points_range_l2118_211840


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l2118_211825

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 1

theorem unique_number_with_properties : ∃! n : ℕ,
  n < 200 ∧
  all_digits_odd n ∧
  ∃ a b : ℕ, is_two_digit a ∧ is_two_digit b ∧ n = a * b :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l2118_211825


namespace NUMINAMATH_CALUDE_production_calculation_l2118_211839

/-- Calculates the production given the number of workers, hours per day, 
    number of days, efficiency factor, and base production rate -/
def calculate_production (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
                         (efficiency_factor : ℚ) (base_rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours_per_day : ℚ) * (days : ℚ) * efficiency_factor * base_rate

theorem production_calculation :
  let initial_workers : ℕ := 10
  let initial_hours : ℕ := 6
  let initial_days : ℕ := 5
  let initial_production : ℕ := 200
  let new_workers : ℕ := 8
  let new_hours : ℕ := 7
  let new_days : ℕ := 4
  let efficiency_increase : ℚ := 11/10

  let base_rate : ℚ := (initial_production : ℚ) / 
    ((initial_workers : ℚ) * (initial_hours : ℚ) * (initial_days : ℚ))

  let new_production : ℚ := calculate_production new_workers new_hours new_days 
                            efficiency_increase base_rate

  new_production = 198 :=
by sorry

end NUMINAMATH_CALUDE_production_calculation_l2118_211839


namespace NUMINAMATH_CALUDE_circle_distance_l2118_211829

theorem circle_distance (R r : ℝ) : 
  R^2 - 4*R + 2 = 0 → 
  r^2 - 4*r + 2 = 0 → 
  R ≠ r → 
  (∃ d : ℝ, d = abs (R - r) ∧ (d = 4 ∨ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_l2118_211829


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l2118_211834

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (abs x - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l2118_211834


namespace NUMINAMATH_CALUDE_factor_sum_relation_l2118_211844

theorem factor_sum_relation (P Q R : ℝ) : 
  (∃ b c : ℝ, x^4 + P*x^2 + R*x + Q = (x^2 + 3*x + 7) * (x^2 + b*x + c)) →
  P + Q + R = 11*P - 1 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_relation_l2118_211844


namespace NUMINAMATH_CALUDE_square_difference_l2118_211821

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 80) (h2 : x * y = 12) : 
  (x - y)^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2118_211821


namespace NUMINAMATH_CALUDE_equation_solution_l2118_211875

theorem equation_solution : ∃ x : ℝ, 10111 - 10 * 2 * (5 + x) = 0 ∧ x = 500.55 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2118_211875


namespace NUMINAMATH_CALUDE_unique_solution_is_sqrt_2_l2118_211899

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

theorem unique_solution_is_sqrt_2 :
  ∃! x, x > 1 ∧ f x = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_sqrt_2_l2118_211899


namespace NUMINAMATH_CALUDE_quadratic_root_transformations_l2118_211897

/-- Given a quadratic equation x^2 + px + q = 0, this theorem proves the equations
    with roots differing by sign and reciprocal roots. -/
theorem quadratic_root_transformations (p q : ℝ) :
  let original := fun x : ℝ => x^2 + p*x + q
  let opposite_sign := fun x : ℝ => x^2 - p*x + q
  let reciprocal := fun x : ℝ => q*x^2 + p*x + 1
  (∀ x, original x = 0 → ∃ y, opposite_sign y = 0 ∧ y = -x) ∧
  (∀ x, original x = 0 → x ≠ 0 → ∃ y, reciprocal y = 0 ∧ y = 1/x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformations_l2118_211897


namespace NUMINAMATH_CALUDE_min_rice_purchase_exact_min_rice_purchase_l2118_211809

/-- The minimum amount of rice Maria could purchase, given the constraints on oats and rice. -/
theorem min_rice_purchase (o r : ℝ) 
  (h1 : o ≥ 4 + r / 3)  -- Condition 1: oats ≥ 4 + 1/3 * rice
  (h2 : o ≤ 3 * r)      -- Condition 2: oats ≤ 3 * rice
  : r ≥ 3/2 := by sorry

/-- The exact minimum amount of rice Maria could purchase is 1.5 kg. -/
theorem exact_min_rice_purchase : 
  ∃ (o r : ℝ), r = 3/2 ∧ o = 4.5 ∧ o ≥ 4 + r / 3 ∧ o ≤ 3 * r := by sorry

end NUMINAMATH_CALUDE_min_rice_purchase_exact_min_rice_purchase_l2118_211809


namespace NUMINAMATH_CALUDE_pm25_scientific_notation_correct_l2118_211881

/-- PM2.5 diameter in meters -/
def pm25_diameter : ℝ := 0.0000025

/-- Scientific notation representation of PM2.5 diameter -/
def pm25_scientific : ℝ × ℤ := (2.5, -6)

/-- Theorem stating that the PM2.5 diameter is correctly expressed in scientific notation -/
theorem pm25_scientific_notation_correct :
  pm25_diameter = pm25_scientific.1 * (10 : ℝ) ^ pm25_scientific.2 :=
by sorry

end NUMINAMATH_CALUDE_pm25_scientific_notation_correct_l2118_211881


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2118_211859

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2118_211859


namespace NUMINAMATH_CALUDE_intersection_distance_l2118_211823

/-- The distance between the points of intersection of three lines -/
theorem intersection_distance (x₁ x₂ : ℝ) (y : ℝ) : 
  x₁ = 1975 / 3 ∧ 
  x₂ = 1981 / 3 ∧ 
  y = 1975 ∧ 
  (3 * x₁ = y) ∧ 
  (3 * x₂ - 6 = y) →
  Real.sqrt ((x₂ - x₁)^2 + (y - y)^2) = 2 := by
  sorry

#check intersection_distance

end NUMINAMATH_CALUDE_intersection_distance_l2118_211823


namespace NUMINAMATH_CALUDE_smallest_prime_with_prime_digit_sum_l2118_211898

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_prime_digit_sum :
  ∃ (p : Nat), is_prime p ∧ 
               is_prime (digit_sum p) ∧ 
               digit_sum p > 10 ∧ 
               (∀ q : Nat, q < p → ¬(is_prime q ∧ is_prime (digit_sum q) ∧ digit_sum q > 10)) ∧
               p = 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_with_prime_digit_sum_l2118_211898


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2118_211802

/-- The volume of a tetrahedron with given edge lengths -/
def tetrahedron_volume (AB BC CD DA AC BD : ℝ) : ℝ :=
  -- Definition of volume calculation goes here
  sorry

/-- Theorem: The volume of the specific tetrahedron is √66/2 -/
theorem specific_tetrahedron_volume :
  tetrahedron_volume 1 (2 * Real.sqrt 6) 5 7 5 7 = Real.sqrt 66 / 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2118_211802


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2118_211878

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 19 * n ≡ 2701 [ZMOD 9] ∧ ∀ (m : ℕ), m > 0 → 19 * m ≡ 2701 [ZMOD 9] → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2118_211878


namespace NUMINAMATH_CALUDE_complex_function_property_l2118_211866

/-- A complex function f(z) = (a+bi)z with certain properties -/
def f (a b : ℝ) (z : ℂ) : ℂ := (Complex.mk a b) * z

/-- The theorem statement -/
theorem complex_function_property (a b c : ℝ) :
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z - Complex.I * c)) →
  (Complex.abs (Complex.mk a b) = 9) →
  (b^2 = 323/4) := by
  sorry

end NUMINAMATH_CALUDE_complex_function_property_l2118_211866


namespace NUMINAMATH_CALUDE_polynomial_root_product_l2118_211845

theorem polynomial_root_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b*r - c = 0) → b*c = 110 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l2118_211845


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2118_211874

-- Define the slope k and inclination angle α
variable (k α : ℝ)

-- Define the relationship between k and α
def slope_angle_relation (k α : ℝ) : Prop := k = Real.tan α

-- Define the range of k
def slope_range (k : ℝ) : Prop := -1 ≤ k ∧ k < Real.sqrt 3

-- Define the range of α
def angle_range (α : ℝ) : Prop := 
  (0 ≤ α ∧ α < Real.pi/3) ∨ (3*Real.pi/4 ≤ α ∧ α < Real.pi)

-- State the theorem
theorem inclination_angle_range :
  ∀ k α, slope_angle_relation k α → slope_range k → angle_range α :=
sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l2118_211874


namespace NUMINAMATH_CALUDE_mixed_groups_count_l2118_211848

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ), 
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos) :=
by sorry

end NUMINAMATH_CALUDE_mixed_groups_count_l2118_211848


namespace NUMINAMATH_CALUDE_wrapping_paper_problem_l2118_211814

theorem wrapping_paper_problem (x : ℝ) 
  (h1 : x + (3/4 * x) + (x + 3/4 * x) = 7) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_problem_l2118_211814


namespace NUMINAMATH_CALUDE_line_l_passes_through_A_and_B_l2118_211822

/-- The line l passes through points A(-1, 0) and B(1, 4) -/
def line_l (x y : ℝ) : Prop := y = 2 * x + 2

/-- Point A has coordinates (-1, 0) -/
def point_A : ℝ × ℝ := (-1, 0)

/-- Point B has coordinates (1, 4) -/
def point_B : ℝ × ℝ := (1, 4)

/-- The line l passes through points A and B -/
theorem line_l_passes_through_A_and_B : 
  line_l point_A.1 point_A.2 ∧ line_l point_B.1 point_B.2 := by sorry

end NUMINAMATH_CALUDE_line_l_passes_through_A_and_B_l2118_211822


namespace NUMINAMATH_CALUDE_triangle_theorem_l2118_211862

noncomputable def triangle_proof (A B C : ℝ) (a b c : ℝ) : Prop :=
  let perimeter := a + b + c
  let area := (1/2) * b * c * Real.sin A
  perimeter = 4 * (Real.sqrt 2 + 1) ∧
  Real.sin B + Real.sin C = Real.sqrt 2 * Real.sin A ∧
  area = 3 * Real.sin A ∧
  a = 4 ∧
  A = Real.arccos (1/3)

theorem triangle_theorem :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_proof A B C a b c :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2118_211862


namespace NUMINAMATH_CALUDE_cuboidal_box_surface_area_l2118_211807

/-- A cuboidal box with given face areas has a specific total surface area -/
theorem cuboidal_box_surface_area (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 120 → w * h = 72 → l * h = 60 →
  2 * (l * w + w * h + l * h) = 504 := by
sorry

end NUMINAMATH_CALUDE_cuboidal_box_surface_area_l2118_211807


namespace NUMINAMATH_CALUDE_problem_solution_l2118_211893

theorem problem_solution : 
  (1/2 - 2/3 - 3/4) * 12 = -11 ∧ 
  -(1^6) + |-2/3| - (1 - 5/9) + 2/3 = -1/9 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2118_211893


namespace NUMINAMATH_CALUDE_square_sum_equals_73_l2118_211865

theorem square_sum_equals_73 (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 24) : 
  a^2 + b^2 = 73 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_73_l2118_211865


namespace NUMINAMATH_CALUDE_peters_speed_is_five_l2118_211864

/-- Peter's speed in miles per hour -/
def peter_speed : ℝ := sorry

/-- Juan's speed in miles per hour -/
def juan_speed : ℝ := peter_speed + 3

/-- Time traveled in hours -/
def time : ℝ := 1.5

/-- Total distance between Juan and Peter after traveling -/
def total_distance : ℝ := 19.5

/-- Theorem stating that Peter's speed is 5 miles per hour -/
theorem peters_speed_is_five :
  peter_speed = 5 :=
by
  have h1 : time * peter_speed + time * juan_speed = total_distance := sorry
  sorry

end NUMINAMATH_CALUDE_peters_speed_is_five_l2118_211864


namespace NUMINAMATH_CALUDE_sqrt_sum_representation_l2118_211837

theorem sqrt_sum_representation : ∃ (a b c : ℕ+),
  (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) = 
   (a.val * Real.sqrt 3 + b.val * Real.sqrt 11) / c.val) ∧
  (∀ (a' b' c' : ℕ+),
    (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) = 
     (a'.val * Real.sqrt 3 + b'.val * Real.sqrt 11) / c'.val) →
    c'.val ≥ c.val) ∧
  a.val = 44 ∧ b.val = 36 ∧ c.val = 33 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_representation_l2118_211837


namespace NUMINAMATH_CALUDE_female_employees_percentage_l2118_211873

/-- The percentage of female employees in an office -/
def percentage_female_employees (total_employees : ℕ) 
  (percent_computer_literate : ℚ) 
  (female_computer_literate : ℕ) 
  (percent_male_computer_literate : ℚ) : ℚ :=
  sorry

/-- Theorem stating the percentage of female employees is 60% -/
theorem female_employees_percentage 
  (h1 : total_employees = 1500)
  (h2 : percent_computer_literate = 62 / 100)
  (h3 : female_computer_literate = 630)
  (h4 : percent_male_computer_literate = 1 / 2) :
  percentage_female_employees total_employees percent_computer_literate 
    female_computer_literate percent_male_computer_literate = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_female_employees_percentage_l2118_211873


namespace NUMINAMATH_CALUDE_carousel_seating_arrangement_l2118_211867

-- Define the friends
inductive Friend
| Alan
| Bella
| Chloe
| David
| Emma

-- Define the seats
inductive Seat
| One
| Two
| Three
| Four
| Five

-- Define the seating arrangement
def SeatingArrangement := Friend → Seat

-- Define the condition of being opposite
def isOpposite (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Three) ∨ (s1 = Seat.Two ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Five) ∨ (s1 = Seat.Four ∧ s2 = Seat.One) ∨
  (s1 = Seat.Five ∧ s2 = Seat.Two)

-- Define the condition of being two seats away
def isTwoSeatsAway (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Three) ∨ (s1 = Seat.Two ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Five) ∨ (s1 = Seat.Four ∧ s2 = Seat.One) ∨
  (s1 = Seat.Five ∧ s2 = Seat.Two)

-- Define the condition of being next to each other
def isNextTo (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ (s2 = Seat.Two ∨ s2 = Seat.Five)) ∨
  (s1 = Seat.Two ∧ (s2 = Seat.One ∨ s2 = Seat.Three)) ∨
  (s1 = Seat.Three ∧ (s2 = Seat.Two ∨ s2 = Seat.Four)) ∨
  (s1 = Seat.Four ∧ (s2 = Seat.Three ∨ s2 = Seat.Five)) ∨
  (s1 = Seat.Five ∧ (s2 = Seat.Four ∨ s2 = Seat.One))

-- Define the condition of being to the immediate left
def isImmediateLeft (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Two) ∨
  (s1 = Seat.Two ∧ s2 = Seat.Three) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Four ∧ s2 = Seat.Five) ∨
  (s1 = Seat.Five ∧ s2 = Seat.One)

theorem carousel_seating_arrangement 
  (seating : SeatingArrangement)
  (h1 : isOpposite (seating Friend.Chloe) (seating Friend.Emma))
  (h2 : isTwoSeatsAway (seating Friend.David) (seating Friend.Alan))
  (h3 : ¬isNextTo (seating Friend.Alan) (seating Friend.Emma))
  (h4 : isNextTo (seating Friend.Bella) (seating Friend.Emma))
  : isImmediateLeft (seating Friend.Chloe) (seating Friend.Alan) :=
sorry

end NUMINAMATH_CALUDE_carousel_seating_arrangement_l2118_211867


namespace NUMINAMATH_CALUDE_dress_shoes_count_l2118_211816

theorem dress_shoes_count (polished_percent : ℚ) (remaining : ℕ) : 
  polished_percent = 45/100 → remaining = 11 → (1 - polished_percent) * (2 * remaining / (1 - polished_percent)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dress_shoes_count_l2118_211816


namespace NUMINAMATH_CALUDE_village_assistant_selection_l2118_211887

theorem village_assistant_selection (n : ℕ) (k : ℕ) : 
  n = 10 → k = 3 → (Nat.choose 9 3) - (Nat.choose 7 3) = 49 := by
  sorry

end NUMINAMATH_CALUDE_village_assistant_selection_l2118_211887


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l2118_211815

theorem opposite_sides_line_range (a : ℝ) : 
  (((3 : ℝ) * 3 - 2 * 1 + a > 0 ∧ (3 : ℝ) * (-4) - 2 * 6 + a < 0) ∨
   ((3 : ℝ) * 3 - 2 * 1 + a < 0 ∧ (3 : ℝ) * (-4) - 2 * 6 + a > 0)) →
  -7 < a ∧ a < 24 := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l2118_211815


namespace NUMINAMATH_CALUDE_berry_theorem_l2118_211808

def berry_problem (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) (raspberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries + raspberries)

theorem berry_theorem : berry_problem 50 18 12 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_berry_theorem_l2118_211808


namespace NUMINAMATH_CALUDE_citizenship_test_study_time_l2118_211849

/-- Calculates the total study time in hours for a citizenship test -/
theorem citizenship_test_study_time :
  let total_questions : ℕ := 90
  let multiple_choice_questions : ℕ := 30
  let fill_in_blank_questions : ℕ := 30
  let essay_questions : ℕ := 30
  let multiple_choice_time : ℕ := 15  -- minutes per question
  let fill_in_blank_time : ℕ := 25    -- minutes per question
  let essay_time : ℕ := 45            -- minutes per question
  
  let total_time_minutes : ℕ := 
    multiple_choice_questions * multiple_choice_time +
    fill_in_blank_questions * fill_in_blank_time +
    essay_questions * essay_time

  let total_time_hours : ℚ := (total_time_minutes : ℚ) / 60

  total_questions = multiple_choice_questions + fill_in_blank_questions + essay_questions →
  total_time_hours = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_citizenship_test_study_time_l2118_211849


namespace NUMINAMATH_CALUDE_number_at_21_21_l2118_211861

/-- Represents the number at a given position in the matrix -/
def matrixNumber (row : ℕ) (col : ℕ) : ℕ :=
  row^2 - (col - 1)

/-- The theorem stating that the number in the 21st row and 21st column is 421 -/
theorem number_at_21_21 : matrixNumber 21 21 = 421 := by
  sorry

end NUMINAMATH_CALUDE_number_at_21_21_l2118_211861


namespace NUMINAMATH_CALUDE_fraction_relation_l2118_211850

theorem fraction_relation (a b c d : ℚ) 
  (h1 : a / b = 8)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 3) :
  d / a = 15 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l2118_211850


namespace NUMINAMATH_CALUDE_rs_value_l2118_211851

theorem rs_value (r s : ℝ) (hr : r > 0) (hs : s > 0)
  (h1 : r^3 + s^3 = 1) (h2 : r^6 + s^6 = 15/16) :
  r * s = 1 / Real.rpow 48 (1/3) :=
sorry

end NUMINAMATH_CALUDE_rs_value_l2118_211851


namespace NUMINAMATH_CALUDE_acute_angle_inequality_l2118_211820

theorem acute_angle_inequality (α : Real) (h : 0 < α ∧ α < π / 2) :
  α < (Real.sin α + Real.tan α) / 2 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_inequality_l2118_211820


namespace NUMINAMATH_CALUDE_increasing_function_condition_l2118_211800

def f (a b x : ℝ) : ℝ := (a + 1) * x + b

theorem increasing_function_condition (a b : ℝ) :
  (∀ x y : ℝ, x < y → f a b x < f a b y) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l2118_211800


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2118_211817

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 - 4) :
  (4 - x) / (x - 2) / (x + 2 - 12 / (x - 2)) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2118_211817


namespace NUMINAMATH_CALUDE_proposition_analysis_l2118_211891

def p : Prop := 6 ∣ 12
def q : Prop := 6 ∣ 24

theorem proposition_analysis :
  (p ∨ q) ∧ (p ∧ q) ∧ (¬¬p) := by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l2118_211891


namespace NUMINAMATH_CALUDE_two_pow_gt_square_l2118_211879

theorem two_pow_gt_square (n : ℕ) (h : n > 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_two_pow_gt_square_l2118_211879


namespace NUMINAMATH_CALUDE_smallest_n_for_rectangle_l2118_211855

/-- A function that checks if it's possible to form a rectangle with given pieces --/
def can_form_rectangle (pieces : List Nat) : Prop :=
  ∃ (w h : Nat), w * 2 + h * 2 = pieces.sum ∧ w > 0 ∧ h > 0

/-- The main theorem stating that 102 is the smallest N that satisfies the conditions --/
theorem smallest_n_for_rectangle : 
  (∀ n < 102, ¬∃ (pieces : List Nat), 
    pieces.length = n ∧ 
    pieces.sum = 200 ∧ 
    (∀ p ∈ pieces, p > 0) ∧
    can_form_rectangle pieces) ∧
  (∃ (pieces : List Nat), 
    pieces.length = 102 ∧ 
    pieces.sum = 200 ∧ 
    (∀ p ∈ pieces, p > 0) ∧
    can_form_rectangle pieces) :=
by sorry

#check smallest_n_for_rectangle

end NUMINAMATH_CALUDE_smallest_n_for_rectangle_l2118_211855


namespace NUMINAMATH_CALUDE_cosine_of_angle_l2118_211843

def a : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -3)

theorem cosine_of_angle (t : ℝ) : 
  let b : ℝ × ℝ := (3, t)
  (b.1 * c.1 + b.2 * c.2 = 0) →  -- b ⊥ c
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_of_angle_l2118_211843


namespace NUMINAMATH_CALUDE_parabola_equation_l2118_211860

/-- A parabola with vertex at the origin, coordinate axes as axes of symmetry, 
    and passing through point (-4, -2) has a standard equation of either 
    x^2 = -8y or y^2 = -x -/
theorem parabola_equation (f : ℝ → ℝ) : 
  (∀ x y, f x = y ↔ (x^2 = -8*y ∨ y^2 = -x)) ↔ 
  (f 0 = 0 ∧ 
   (∀ x, f x = f (-x)) ∧ 
   (∀ y, f (f y) = y) ∧
   f (-4) = -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2118_211860


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l2118_211803

theorem yellow_marbles_count (total red blue green yellow : ℕ) : 
  total = 110 →
  red = 8 →
  blue = 4 * red →
  green = 2 * blue →
  yellow = total - (red + blue + green) →
  yellow = 6 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l2118_211803


namespace NUMINAMATH_CALUDE_base_of_negative_four_cubed_l2118_211806

def base_of_power (x : ℤ) (n : ℕ) : ℤ := x

theorem base_of_negative_four_cubed :
  base_of_power (-4) 3 = -4 := by sorry

end NUMINAMATH_CALUDE_base_of_negative_four_cubed_l2118_211806


namespace NUMINAMATH_CALUDE_paint_needed_l2118_211889

theorem paint_needed (total_needed existing_paint new_paint : ℕ) 
  (h1 : total_needed = 70)
  (h2 : existing_paint = 36)
  (h3 : new_paint = 23) :
  total_needed - (existing_paint + new_paint) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_needed_l2118_211889


namespace NUMINAMATH_CALUDE_domain_of_f_l2118_211884

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 3)) / (x^2 + 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 < x ∧ x ≠ -1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2118_211884


namespace NUMINAMATH_CALUDE_drivers_distance_difference_l2118_211805

/-- Calculates the difference in distance traveled between two drivers meeting on a highway --/
theorem drivers_distance_difference
  (initial_distance : ℝ)
  (speed_a : ℝ)
  (speed_b : ℝ)
  (delay : ℝ)
  (h1 : initial_distance = 787)
  (h2 : speed_a = 90)
  (h3 : speed_b = 80)
  (h4 : delay = 1) :
  let remaining_distance := initial_distance - speed_a * delay
  let relative_speed := speed_a + speed_b
  let meeting_time := remaining_distance / relative_speed
  let distance_a := speed_a * (meeting_time + delay)
  let distance_b := speed_b * meeting_time
  distance_a - distance_b = 131 := by sorry

end NUMINAMATH_CALUDE_drivers_distance_difference_l2118_211805


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2118_211830

theorem solution_set_equivalence :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2118_211830


namespace NUMINAMATH_CALUDE_kim_thursday_sales_l2118_211853

/-- The number of boxes Kim sold on Tuesday -/
def tuesday_sales : ℕ := 4800

/-- The number of boxes Kim sold on Wednesday -/
def wednesday_sales : ℕ := tuesday_sales / 2

/-- The number of boxes Kim sold on Thursday -/
def thursday_sales : ℕ := wednesday_sales / 2

/-- Theorem stating that Kim sold 1200 boxes on Thursday -/
theorem kim_thursday_sales : thursday_sales = 1200 := by
  sorry

end NUMINAMATH_CALUDE_kim_thursday_sales_l2118_211853


namespace NUMINAMATH_CALUDE_dot_product_zero_on_diagonal_l2118_211892

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_unit_square : A.1 + 1 = B.1 ∧ A.2 + 1 = B.2 ∧
                   C.1 = B.1 ∧ C.2 = B.2 + 1 ∧
                   D.1 = A.1 ∧ D.2 = C.2

/-- A point on the diagonal AC of a unit square -/
def PointOnDiagonal (square : UnitSquare) : Type :=
  {P : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    P.1 = square.A.1 + t * (square.C.1 - square.A.1) ∧
    P.2 = square.A.2 + t * (square.C.2 - square.A.2)}

/-- Vector from point A to point P -/
def vec_AP (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (P.val.1 - square.A.1, P.val.2 - square.A.2)

/-- Vector from point P to point B -/
def vec_PB (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (square.B.1 - P.val.1, square.B.2 - P.val.2)

/-- Vector from point P to point D -/
def vec_PD (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (square.D.1 - P.val.1, square.D.2 - P.val.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_zero_on_diagonal (square : UnitSquare) (P : PointOnDiagonal square) :
  dot_product (vec_AP square P) (vec_PB square P + vec_PD square P) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_zero_on_diagonal_l2118_211892


namespace NUMINAMATH_CALUDE_symmetry_of_functions_l2118_211877

theorem symmetry_of_functions (f : ℝ → ℝ) : 
  ∀ x y : ℝ, f (1 - x) = y ↔ f (x - 1) = y :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_functions_l2118_211877


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2118_211894

theorem no_positive_integer_solution :
  ¬ ∃ (a b c : ℕ+), a^2 + b^2 = 4 * c + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2118_211894


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2118_211801

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then
    Real.sqrt (1 + Real.log (1 + 3 * x^2 * Real.cos (2 / x))) - 1
  else
    0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2118_211801


namespace NUMINAMATH_CALUDE_number_ordering_l2118_211876

theorem number_ordering : 6^10 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l2118_211876


namespace NUMINAMATH_CALUDE_smaller_cone_height_equals_frustum_height_l2118_211856

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  height : ℝ
  larger_base_area : ℝ
  smaller_base_area : ℝ

/-- Calculates the height of the smaller cone removed to form the frustum -/
def smaller_cone_height (f : Frustum) : ℝ :=
  f.height

/-- Theorem stating that the height of the smaller cone is equal to the frustum's height -/
theorem smaller_cone_height_equals_frustum_height (f : Frustum)
  (h1 : f.height = 18)
  (h2 : f.larger_base_area = 400 * Real.pi)
  (h3 : f.smaller_base_area = 100 * Real.pi) :
  smaller_cone_height f = f.height :=
by sorry

end NUMINAMATH_CALUDE_smaller_cone_height_equals_frustum_height_l2118_211856


namespace NUMINAMATH_CALUDE_number_of_ways_to_choose_officials_l2118_211858

-- Define the number of people in the group
def group_size : ℕ := 8

-- Define the number of positions to be filled
def num_positions : ℕ := 3

-- Theorem stating the number of ways to choose the officials
theorem number_of_ways_to_choose_officials :
  (group_size * (group_size - 1) * (group_size - 2)) = 336 := by
  sorry

end NUMINAMATH_CALUDE_number_of_ways_to_choose_officials_l2118_211858


namespace NUMINAMATH_CALUDE_common_factor_is_gcf_l2118_211880

noncomputable def p1 (x y z : ℝ) : ℝ := 3 * x^2 * y^3 * z + 9 * x^3 * y^3 * z
noncomputable def p2 (x y z : ℝ) : ℝ := 6 * x^4 * y * z^2
noncomputable def common_factor (x y z : ℝ) : ℝ := 3 * x^2 * y * z

theorem common_factor_is_gcf :
  ∀ x y z : ℝ,
  (∃ k1 k2 : ℝ, p1 x y z = common_factor x y z * k1 ∧ p2 x y z = common_factor x y z * k2) ∧
  (∀ f : ℝ → ℝ → ℝ → ℝ, (∃ l1 l2 : ℝ, p1 x y z = f x y z * l1 ∧ p2 x y z = f x y z * l2) →
    ∃ m : ℝ, f x y z = common_factor x y z * m) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_is_gcf_l2118_211880


namespace NUMINAMATH_CALUDE_no_integer_a_with_one_integer_solution_l2118_211836

theorem no_integer_a_with_one_integer_solution :
  ¬ ∃ (a : ℤ), ∃! (x : ℤ), x^3 - a*x^2 - 6*a*x + a^2 - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_a_with_one_integer_solution_l2118_211836


namespace NUMINAMATH_CALUDE_constant_derivative_implies_linear_l2118_211847

/-- If a function's derivative is zero everywhere, then its graph is a straight line -/
theorem constant_derivative_implies_linear (f : ℝ → ℝ) :
  (∀ x : ℝ, deriv f x = 0) → ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_constant_derivative_implies_linear_l2118_211847


namespace NUMINAMATH_CALUDE_decimal_place_150_of_5_11_l2118_211868

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The period of the decimal representation of a rational number -/
def decimal_period (q : ℚ) : ℕ := sorry

theorem decimal_place_150_of_5_11 :
  decimal_representation (5/11) 150 = 5 := by sorry

end NUMINAMATH_CALUDE_decimal_place_150_of_5_11_l2118_211868


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2118_211886

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x | x - 1 ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2118_211886


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2118_211863

-- Define the circles
def circle1 : ℝ × ℝ := (0, 0)
def circle2 : ℝ × ℝ := (20, 0)
def radius1 : ℝ := 3
def radius2 : ℝ := 9

-- Define the tangent line intersection point
def intersection_point : ℝ := 5

-- Theorem statement
theorem tangent_line_intersection :
  let d := circle2.1 - circle1.1  -- Distance between circle centers
  ∃ (t : ℝ), 
    t > 0 ∧ 
    intersection_point = circle1.1 + t * radius1 ∧
    intersection_point = circle2.1 - t * radius2 ∧
    t * (radius1 + radius2) = d :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2118_211863


namespace NUMINAMATH_CALUDE_triangle_rectangle_perimeter_equality_l2118_211835

/-- The perimeter of an isosceles triangle with two sides of 12 cm and one side of 14 cm 
    is equal to the perimeter of a rectangle with width 8 cm and length x cm. -/
theorem triangle_rectangle_perimeter_equality (x : ℝ) : 
  (12 : ℝ) + 12 + 14 = 2 * (x + 8) → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_rectangle_perimeter_equality_l2118_211835


namespace NUMINAMATH_CALUDE_geometric_mean_of_square_sides_l2118_211828

theorem geometric_mean_of_square_sides (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 64) (h₂ : a₂ = 81) (h₃ : a₃ = 144) :
  (((a₁.sqrt * a₂.sqrt * a₃.sqrt) ^ (1/3 : ℝ)) : ℝ) = 6 * (4 ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_square_sides_l2118_211828


namespace NUMINAMATH_CALUDE_hexagon_perimeter_hexagon_perimeter_proof_l2118_211883

/-- The perimeter of a regular hexagon with side length 2 inches is 12 inches. -/
theorem hexagon_perimeter : ℝ → Prop :=
  fun (side_length : ℝ) =>
    side_length = 2 →
    6 * side_length = 12

/-- Proof of the theorem -/
theorem hexagon_perimeter_proof : hexagon_perimeter 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_hexagon_perimeter_proof_l2118_211883


namespace NUMINAMATH_CALUDE_intersection_is_empty_l2118_211832

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * p.1 - 3}

theorem intersection_is_empty : set_A ∩ set_B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l2118_211832


namespace NUMINAMATH_CALUDE_rectangle_problem_l2118_211872

/-- Prove that A = 4 given the conditions of the rectangle problem -/
theorem rectangle_problem (A : ℝ) : A = 4 :=
  let original_side : ℝ := 12
  let rectangle_width : ℝ := original_side + 3
  let rectangle_length : ℝ := original_side - A
  let rectangle_area : ℝ := 120
  have h1 : rectangle_area = rectangle_width * rectangle_length :=
    by sorry
  have h2 : rectangle_area = (original_side + 3) * (original_side - A) :=
    by sorry
  have h3 : 120 = (12 + 3) * (12 - A) :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l2118_211872


namespace NUMINAMATH_CALUDE_rectangle_to_square_side_half_length_l2118_211804

/-- Given a rectangle with dimensions 7 × 21 that is cut into two congruent shapes
    and rearranged into a square, half the length of a side of the resulting square
    is equal to 7√3/2. -/
theorem rectangle_to_square_side_half_length :
  let rectangle_length : ℝ := 21
  let rectangle_width : ℝ := 7
  let rectangle_area := rectangle_length * rectangle_width
  let square_side := Real.sqrt rectangle_area
  let y := square_side / 2
  y = 7 * Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_side_half_length_l2118_211804


namespace NUMINAMATH_CALUDE_equation_root_implies_m_equals_three_l2118_211882

theorem equation_root_implies_m_equals_three (x m : ℝ) :
  (x ≠ 3) →
  (x / (x - 3) = 2 - m / (3 - x)) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_equals_three_l2118_211882


namespace NUMINAMATH_CALUDE_money_ratio_l2118_211842

theorem money_ratio (rodney ian jessica : ℕ) : 
  rodney = ian + 35 →
  jessica = 100 →
  jessica = rodney + 15 →
  ian * 2 = jessica := by sorry

end NUMINAMATH_CALUDE_money_ratio_l2118_211842


namespace NUMINAMATH_CALUDE_muscovy_duck_percentage_l2118_211896

theorem muscovy_duck_percentage (total_ducks : ℕ) (female_muscovy : ℕ) 
  (h1 : total_ducks = 40)
  (h2 : female_muscovy = 6)
  (h3 : (female_muscovy : ℝ) / ((total_ducks : ℝ) * 0.5) = 0.3) :
  (total_ducks : ℝ) * 0.5 = (total_ducks : ℝ) * 0.5 := by
  sorry

#check muscovy_duck_percentage

end NUMINAMATH_CALUDE_muscovy_duck_percentage_l2118_211896


namespace NUMINAMATH_CALUDE_quadratic_solution_form_l2118_211811

theorem quadratic_solution_form (x : ℝ) : 
  (5 * x^2 - 11 * x + 2 = 0) →
  ∃ (m n p : ℕ), 
    x = (m + Real.sqrt n) / p ∧ 
    m = 20 ∧ 
    n = 0 ∧ 
    p = 10 ∧
    m + n + p = 30 ∧
    Nat.gcd m (Nat.gcd n p) = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_form_l2118_211811


namespace NUMINAMATH_CALUDE_at_least_one_outstanding_equiv_l2118_211846

/-- Represents whether a person is an outstanding student -/
def IsOutstandingStudent (person : Prop) : Prop := person

/-- The statement "At least one of person A and person B is an outstanding student" -/
def AtLeastOneOutstanding (A B : Prop) : Prop :=
  IsOutstandingStudent A ∨ IsOutstandingStudent B

theorem at_least_one_outstanding_equiv (A B : Prop) :
  AtLeastOneOutstanding A B ↔ (IsOutstandingStudent A ∨ IsOutstandingStudent B) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_outstanding_equiv_l2118_211846


namespace NUMINAMATH_CALUDE_gear_rotation_problem_l2118_211885

/-- The number of revolutions per minute of gear q -/
def q_rpm : ℝ := 40

/-- The time elapsed in minutes -/
def time : ℝ := 1.5

/-- The difference in revolutions between gears q and p after 90 seconds -/
def rev_diff : ℝ := 45

/-- The number of revolutions per minute of gear p -/
def p_rpm : ℝ := 10

theorem gear_rotation_problem :
  p_rpm * time + rev_diff = q_rpm * time := by sorry

end NUMINAMATH_CALUDE_gear_rotation_problem_l2118_211885


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2118_211841

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(3*x+2) * (4 : ℝ)^(2*x+1) = (8 : ℝ)^(3*x+4) ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2118_211841


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2118_211826

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) : 
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7*x^6 + 1)) / (3 * x^3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2118_211826


namespace NUMINAMATH_CALUDE_bottles_sold_eq_60_l2118_211831

/-- Represents the sales data for Wal-Mart's thermometers and hot-water bottles --/
structure SalesData where
  thermometer_price : ℕ
  bottle_price : ℕ
  total_sales : ℕ
  thermometer_to_bottle_ratio : ℕ

/-- Calculates the number of hot-water bottles sold given the sales data --/
def bottles_sold (data : SalesData) : ℕ :=
  data.total_sales / (data.bottle_price + data.thermometer_price * data.thermometer_to_bottle_ratio)

/-- Theorem stating that given the specific sales data, 60 hot-water bottles were sold --/
theorem bottles_sold_eq_60 (data : SalesData) 
    (h1 : data.thermometer_price = 2)
    (h2 : data.bottle_price = 6)
    (h3 : data.total_sales = 1200)
    (h4 : data.thermometer_to_bottle_ratio = 7) : 
  bottles_sold data = 60 := by
  sorry

#eval bottles_sold { thermometer_price := 2, bottle_price := 6, total_sales := 1200, thermometer_to_bottle_ratio := 7 }

end NUMINAMATH_CALUDE_bottles_sold_eq_60_l2118_211831


namespace NUMINAMATH_CALUDE_union_of_sets_l2118_211854

theorem union_of_sets : 
  let M : Set ℕ := {1, 2, 4, 5}
  let N : Set ℕ := {2, 3, 4}
  M ∪ N = {1, 2, 3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2118_211854


namespace NUMINAMATH_CALUDE_max_perfect_matchings_20gon_l2118_211857

/-- Represents a convex polygon with 2n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin (2 * n) → ℝ × ℝ

/-- Represents a triangulation of a convex polygon -/
structure Triangulation (n : ℕ) where
  polygon : ConvexPolygon n
  diagonals : Fin (2 * n - 3) → Fin (2 * n) × Fin (2 * n)

/-- Represents a perfect matching in a triangulation -/
structure PerfectMatching (n : ℕ) where
  triangulation : Triangulation n
  edges : Fin n → Fin (4 * n - 3)

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => fib (n + 1) + fib n

/-- The maximum number of perfect matchings for a convex 2n-gon -/
def maxPerfectMatchings (n : ℕ) : ℕ := fib n

/-- The theorem statement -/
theorem max_perfect_matchings_20gon :
  maxPerfectMatchings 10 = 89 := by sorry

end NUMINAMATH_CALUDE_max_perfect_matchings_20gon_l2118_211857


namespace NUMINAMATH_CALUDE_triangle_height_l2118_211890

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 12 → area = 30 → area = (base * height) / 2 → height = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l2118_211890


namespace NUMINAMATH_CALUDE_souvenir_problem_l2118_211813

/-- Represents the cost and selling prices of souvenirs -/
structure SouvenirPrices where
  costA : ℝ
  costB : ℝ
  sellingB : ℝ

/-- Represents the quantity and profit of souvenirs -/
structure SouvenirQuantities where
  totalQuantity : ℕ
  minQuantityA : ℕ

/-- Theorem stating the properties of the souvenir problem -/
theorem souvenir_problem 
  (prices : SouvenirPrices) 
  (quantities : SouvenirQuantities) 
  (h1 : prices.costA = prices.costB + 30)
  (h2 : 1000 / prices.costA = 400 / prices.costB)
  (h3 : quantities.totalQuantity = 200)
  (h4 : quantities.minQuantityA ≥ 60)
  (h5 : quantities.minQuantityA < quantities.totalQuantity - quantities.minQuantityA)
  (h6 : prices.sellingB = 30) :
  prices.costA = 50 ∧ 
  prices.costB = 20 ∧
  (∃ x : ℝ, x = 65 ∧ (x - prices.costA) * (400 - 5*x) = 1125) ∧
  (∃ y : ℝ, y = 2480 ∧ 
    y = (68 - prices.costA) * (400 - 5*68) + 
        (prices.sellingB - prices.costB) * (quantities.totalQuantity - (400 - 5*68))) :=
by sorry

end NUMINAMATH_CALUDE_souvenir_problem_l2118_211813


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2118_211818

/-- The magnitude of the complex number z = (1+i)/i is equal to √2 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2118_211818


namespace NUMINAMATH_CALUDE_function_is_linear_l2118_211869

/-- Given a function f: ℝ → ℝ satisfying f(x²-y²) = x f(x) - y f(y) for all x, y ∈ ℝ,
    prove that f is a linear function. -/
theorem function_is_linear (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) :
    ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end NUMINAMATH_CALUDE_function_is_linear_l2118_211869


namespace NUMINAMATH_CALUDE_salary_raise_percentage_l2118_211852

/-- Calculates the percentage raise given the original and new salaries. -/
def percentage_raise (original : ℚ) (new : ℚ) : ℚ :=
  (new - original) / original * 100

/-- Proves that the percentage raise from $500 to $530 is 6%. -/
theorem salary_raise_percentage :
  percentage_raise 500 530 = 6 := by
  sorry

end NUMINAMATH_CALUDE_salary_raise_percentage_l2118_211852
