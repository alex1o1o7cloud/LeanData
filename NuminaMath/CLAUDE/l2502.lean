import Mathlib

namespace NUMINAMATH_CALUDE_triangle_coloring_theorem_l2502_250296

/-- The number of ways to color 6 circles in a fixed triangular arrangement with 4 blue, 1 green, and 1 red circle -/
def triangle_coloring_ways : ℕ := 30

/-- Theorem stating that the number of ways to color the triangular arrangement is 30 -/
theorem triangle_coloring_theorem : triangle_coloring_ways = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_coloring_theorem_l2502_250296


namespace NUMINAMATH_CALUDE_total_equipment_cost_l2502_250234

def num_players : ℕ := 16
def jersey_cost : ℚ := 25
def shorts_cost : ℚ := 15.20
def socks_cost : ℚ := 6.80

theorem total_equipment_cost :
  (num_players : ℚ) * (jersey_cost + shorts_cost + socks_cost) = 752 := by
  sorry

end NUMINAMATH_CALUDE_total_equipment_cost_l2502_250234


namespace NUMINAMATH_CALUDE_max_sum_cubes_l2502_250270

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (M : ℝ), M = 5 * Real.sqrt 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 ≤ M ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l2502_250270


namespace NUMINAMATH_CALUDE_log_expression_equals_four_l2502_250227

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_four :
  4 * log10 2 + 3 * log10 5 - log10 (1/5) = 4 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_four_l2502_250227


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_seven_l2502_250246

theorem same_solution_implies_a_equals_seven (a : ℝ) : 
  (∃ x : ℝ, 6 * (x + 8) = 18 * x ∧ 6 * x - 2 * (a - x) = 2 * a + x) → 
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_seven_l2502_250246


namespace NUMINAMATH_CALUDE_solve_for_m_l2502_250238

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (2 * x + 2) + 3

-- State the theorem
theorem solve_for_m (m : ℝ) : f (m) = 6 → m = -1/4 := by sorry

end NUMINAMATH_CALUDE_solve_for_m_l2502_250238


namespace NUMINAMATH_CALUDE_apples_on_tree_l2502_250201

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 9

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := initial_apples - picked_apples

theorem apples_on_tree : remaining_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_on_tree_l2502_250201


namespace NUMINAMATH_CALUDE_unique_base_conversion_l2502_250278

/-- Convert a base-5 number to decimal --/
def base5ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number in base b to decimal --/
def baseBToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The unique positive solution for b in the equation 32₅ = 121ᵦ --/
theorem unique_base_conversion : 
  ∃! (b : ℕ), b > 0 ∧ base5ToDecimal 32 = baseBToDecimal 121 b := by sorry

end NUMINAMATH_CALUDE_unique_base_conversion_l2502_250278


namespace NUMINAMATH_CALUDE_ice_cream_problem_l2502_250208

/-- Ice cream purchase and profit maximization problem -/
theorem ice_cream_problem 
  (cost_equation1 : ℝ → ℝ → Prop) 
  (cost_equation2 : ℝ → ℝ → Prop)
  (total_budget : ℝ)
  (total_ice_creams : ℕ)
  (brand_a_constraint : ℕ → ℕ → Prop)
  (selling_price_a : ℝ)
  (selling_price_b : ℝ) :
  -- Part 1: Purchase prices
  ∃ (price_a price_b : ℝ),
    cost_equation1 price_a price_b ∧
    cost_equation2 price_a price_b ∧
    price_a = 12 ∧
    price_b = 15 ∧
  -- Part 2: Profit maximization
  ∃ (brand_a brand_b : ℕ),
    brand_a + brand_b = total_ice_creams ∧
    brand_a_constraint brand_a brand_b ∧
    price_a * brand_a + price_b * brand_b ≤ total_budget ∧
    brand_a = 20 ∧
    brand_b = 20 ∧
    ∀ (m n : ℕ), 
      m + n = total_ice_creams →
      brand_a_constraint m n →
      price_a * m + price_b * n ≤ total_budget →
      (selling_price_a - price_a) * brand_a + (selling_price_b - price_b) * brand_b ≥
      (selling_price_a - price_a) * m + (selling_price_b - price_b) * n :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_problem_l2502_250208


namespace NUMINAMATH_CALUDE_overlapping_circles_area_l2502_250236

/-- The area of the common part of two equal circles with radius R,
    where the circumference of each circle passes through the center of the other. -/
theorem overlapping_circles_area (R : ℝ) (R_pos : R > 0) :
  ∃ (A : ℝ), A = R^2 * (4 * Real.pi - 3 * Real.sqrt 3) / 6 ∧
  A = 2 * (1/3 * Real.pi * R^2 - R^2 * Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_overlapping_circles_area_l2502_250236


namespace NUMINAMATH_CALUDE_selection_methods_count_l2502_250210

/-- The number of teachers in each department -/
def teachers_per_dept : ℕ := 4

/-- The total number of departments -/
def total_depts : ℕ := 4

/-- The number of leaders to be selected -/
def leaders_to_select : ℕ := 3

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to select leaders satisfying the given conditions -/
def selection_methods : ℕ :=
  -- One from admin, two from same other dept
  choose teachers_per_dept 1 * choose (total_depts - 1) 1 * choose teachers_per_dept 2 +
  -- One from admin, two from different other depts
  choose teachers_per_dept 1 * choose (total_depts - 1) 2 * choose teachers_per_dept 1 * choose teachers_per_dept 1 +
  -- Two from admin, one from any other dept
  choose teachers_per_dept 2 * choose (total_depts - 1) 1 * choose teachers_per_dept 1

theorem selection_methods_count :
  selection_methods = 336 :=
by sorry

end NUMINAMATH_CALUDE_selection_methods_count_l2502_250210


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2502_250219

/-- If the set A = {x ∈ ℝ | ax² + ax + 1 = 0} has only one element, then a = 4 -/
theorem unique_quadratic_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2502_250219


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l2502_250253

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmeticSequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 8) :
  a 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l2502_250253


namespace NUMINAMATH_CALUDE_isosceles_triangle_relationship_l2502_250211

-- Define the isosceles triangle
structure IsoscelesTriangle where
  x : ℝ  -- leg length
  y : ℝ  -- base length

-- Define the properties of the isosceles triangle
def validIsoscelesTriangle (t : IsoscelesTriangle) : Prop :=
  t.x > 0 ∧ t.y > 0 ∧ 2 * t.x > t.y ∧ t.x + t.y > t.x

-- Define the perimeter constraint
def hasPerimeter30 (t : IsoscelesTriangle) : Prop :=
  2 * t.x + t.y = 30

-- Define the relationship between x and y
def relationshipXY (t : IsoscelesTriangle) : Prop :=
  t.y = 30 - 2 * t.x

-- Define the constraints on x
def xConstraints (t : IsoscelesTriangle) : Prop :=
  15 / 2 < t.x ∧ t.x < 15

-- Theorem stating the relationship between x and y for the isosceles triangle
theorem isosceles_triangle_relationship (t : IsoscelesTriangle) :
  validIsoscelesTriangle t → hasPerimeter30 t → relationshipXY t ∧ xConstraints t :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_relationship_l2502_250211


namespace NUMINAMATH_CALUDE_sum_of_transformed_roots_equals_one_l2502_250232

theorem sum_of_transformed_roots_equals_one : 
  ∀ α β γ : ℂ, 
  (α^3 = α + 1) → (β^3 = β + 1) → (γ^3 = γ + 1) →
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_transformed_roots_equals_one_l2502_250232


namespace NUMINAMATH_CALUDE_people_who_got_off_l2502_250218

theorem people_who_got_off (initial_people : ℕ) (remaining_people : ℕ) (h1 : initial_people = 48) (h2 : remaining_people = 31) :
  initial_people - remaining_people = 17 := by
  sorry

end NUMINAMATH_CALUDE_people_who_got_off_l2502_250218


namespace NUMINAMATH_CALUDE_problem_body_surface_area_l2502_250214

/-- Represents a three-dimensional geometric body -/
structure GeometricBody where
  -- Add necessary fields to represent the geometric body
  -- This is a placeholder as we don't have specific information about the structure

/-- Calculates the surface area of a geometric body -/
noncomputable def surfaceArea (body : GeometricBody) : ℝ :=
  sorry -- Actual calculation would go here

/-- Represents the specific geometric body from the problem -/
def problemBody : GeometricBody :=
  sorry -- Construction of the specific body would go here

/-- Theorem stating that the surface area of the problem's geometric body is 40 -/
theorem problem_body_surface_area :
    surfaceArea problemBody = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_body_surface_area_l2502_250214


namespace NUMINAMATH_CALUDE_arithmetic_expressions_correctness_l2502_250216

theorem arithmetic_expressions_correctness :
  (∀ a b c : ℚ, (a + b) + c = a + (b + c)) ∧
  (∃ a b c : ℚ, (a - b) - c ≠ a - (b - c)) ∧
  (∃ a b c : ℚ, (a + b) / c ≠ a + (b / c)) ∧
  (∃ a b c : ℚ, (a / b) / c ≠ a / (b / c)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_correctness_l2502_250216


namespace NUMINAMATH_CALUDE_gcf_72_90_l2502_250258

theorem gcf_72_90 : Nat.gcd 72 90 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_90_l2502_250258


namespace NUMINAMATH_CALUDE_glass_volume_l2502_250288

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = volume_pessimist)
  (h2 : 0.6 * V = volume_optimist)
  (h3 : volume_optimist - volume_pessimist = 46) :
  V = 230 :=
by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l2502_250288


namespace NUMINAMATH_CALUDE_perpendicular_line_unique_l2502_250299

-- Define a line by its coefficients (a, b, c) in the equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Line.throughPoint (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_unique :
  ∃! l : Line, l.throughPoint (3, 0) ∧
                l.perpendicular { a := 2, b := 1, c := -5 } ∧
                l = { a := 1, b := -2, c := -3 } := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_unique_l2502_250299


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2502_250256

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 9 →
  a 3 = 15 →
  a 7 = 33 →
  a 4 + a 5 + a 6 = 81 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2502_250256


namespace NUMINAMATH_CALUDE_unique_solution_is_zero_l2502_250203

theorem unique_solution_is_zero : 
  ∃! x : ℝ, (3 : ℝ) / (x - 3) = (5 : ℝ) / (x - 5) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_zero_l2502_250203


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_2024_l2502_250254

/-- Given a quadratic equation with roots satisfying specific conditions, 
    the maximum value of the sum of their reciprocals raised to the 2024th power is 2. -/
theorem max_reciprocal_sum_2024 (s p r₁ r₂ : ℝ) : 
  (r₁^2 - s*r₁ + p = 0) →
  (r₂^2 - s*r₂ + p = 0) →
  (∀ (n : ℕ), n ≤ 2023 → r₁^n + r₂^n = s) →
  (∃ (max : ℝ), max = 2 ∧ 
    ∀ (s' p' r₁' r₂' : ℝ), 
      (r₁'^2 - s'*r₁' + p' = 0) →
      (r₂'^2 - s'*r₂' + p' = 0) →
      (∀ (n : ℕ), n ≤ 2023 → r₁'^n + r₂'^n = s') →
      1/r₁'^2024 + 1/r₂'^2024 ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_2024_l2502_250254


namespace NUMINAMATH_CALUDE_two_cakes_left_l2502_250237

/-- The number of cakes left at a restaurant -/
def cakes_left (baked_today baked_yesterday sold : ℕ) : ℕ :=
  baked_today + baked_yesterday - sold

/-- Theorem: Given the conditions, prove that 2 cakes are left -/
theorem two_cakes_left : cakes_left 5 3 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_cakes_left_l2502_250237


namespace NUMINAMATH_CALUDE_f_difference_l2502_250221

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 7*x

-- State the theorem
theorem f_difference : f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2502_250221


namespace NUMINAMATH_CALUDE_cos_sum_less_than_sum_of_cos_l2502_250233

theorem cos_sum_less_than_sum_of_cos (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) : 
  Real.cos (α + β) < Real.cos α + Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_less_than_sum_of_cos_l2502_250233


namespace NUMINAMATH_CALUDE_train_speed_l2502_250294

/-- The speed of a train given its length, time to cross a walking man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 400 →
  crossing_time = 23.998 →
  man_speed_kmh = 3 →
  ∃ (train_speed_kmh : ℝ), 
    (train_speed_kmh ≥ 63.004 ∧ train_speed_kmh ≤ 63.006) ∧
    train_speed_kmh * 1000 / 3600 = 
      train_length / crossing_time + man_speed_kmh * 1000 / 3600 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l2502_250294


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2502_250271

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ x^2 - 4*a*x + 3*a^2 = 0 ∧ y^2 - 4*a*y + 3*a^2 = 0) →
  (1/3 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2502_250271


namespace NUMINAMATH_CALUDE_g_value_l2502_250200

-- Define the polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^3 - 2*x - 2
axiom sum_eq : ∀ x, f x + g x = -2 + x

-- State the theorem
theorem g_value : g = fun x ↦ -x^3 + 3*x := by sorry

end NUMINAMATH_CALUDE_g_value_l2502_250200


namespace NUMINAMATH_CALUDE_negative_fraction_multiplication_l2502_250226

theorem negative_fraction_multiplication :
  ((-144 : ℤ) / (-36 : ℤ)) * 3 = 12 := by sorry

end NUMINAMATH_CALUDE_negative_fraction_multiplication_l2502_250226


namespace NUMINAMATH_CALUDE_first_half_speed_l2502_250290

/-- Given a journey with the following properties:
  * The total distance is 224 km
  * The total time is 10 hours
  * The second half of the journey is traveled at 24 km/hr
  Prove that the speed during the first half of the journey is 21 km/hr -/
theorem first_half_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (second_half_speed : ℝ)
  (h1 : total_distance = 224)
  (h2 : total_time = 10)
  (h3 : second_half_speed = 24)
  : (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_half_speed_l2502_250290


namespace NUMINAMATH_CALUDE_box_weights_sum_l2502_250239

theorem box_weights_sum (box1 box2 box3 box4 box5 : ℝ) 
  (h1 : box1 = 2.5)
  (h2 : box2 = 11.3)
  (h3 : box3 = 5.75)
  (h4 : box4 = 7.2)
  (h5 : box5 = 3.25) :
  box1 + box2 + box3 + box4 + box5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_box_weights_sum_l2502_250239


namespace NUMINAMATH_CALUDE_part_one_part_two_l2502_250250

-- Define propositions p and q
def p (k : ℝ) : Prop := k^2 - 2*k - 24 ≤ 0

def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b < 0 ∧ a = 3 - k ∧ b = 3 + k

-- Part 1
theorem part_one (k : ℝ) : q k → k ∈ Set.Iio (-3) := by
  sorry

-- Part 2
theorem part_two (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) → k ∈ Set.Iio (-4) ∪ Set.Icc (-3) 6 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2502_250250


namespace NUMINAMATH_CALUDE_positive_number_square_root_l2502_250262

theorem positive_number_square_root (x : ℝ) : 
  x > 0 → Real.sqrt ((7 * x) / 3) = x → x = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_positive_number_square_root_l2502_250262


namespace NUMINAMATH_CALUDE_app_security_theorem_all_measures_secure_l2502_250207

/-- Represents a security measure for protecting credit card data -/
inductive SecurityMeasure
  | avoidStoringCardData
  | encryptStoredData
  | encryptDataInTransit
  | codeObfuscation
  | restrictRootedDevices
  | antivirusProtection

/-- Represents an online store app with credit card payment and home delivery -/
structure OnlineStoreApp :=
  (implementedMeasures : List SecurityMeasure)

/-- Defines what it means for an app to be secure -/
def isSecure (app : OnlineStoreApp) : Prop :=
  app.implementedMeasures.length ≥ 3

/-- Theorem stating that implementing at least three security measures 
    ensures the app is secure -/
theorem app_security_theorem (app : OnlineStoreApp) :
  app.implementedMeasures.length ≥ 3 → isSecure app :=
by
  sorry

/-- Corollary: An app with all six security measures is secure -/
theorem all_measures_secure (app : OnlineStoreApp) :
  app.implementedMeasures.length = 6 → isSecure app :=
by
  sorry

end NUMINAMATH_CALUDE_app_security_theorem_all_measures_secure_l2502_250207


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l2502_250255

theorem amusement_park_tickets :
  ∀ (a b c : ℕ),
  a + b + c = 85 →
  7 * a + 4 * b + 2 * c = 500 →
  a = b + 31 →
  a = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_tickets_l2502_250255


namespace NUMINAMATH_CALUDE_train_speed_l2502_250295

/-- The speed of a train given its length and time to pass a stationary point. -/
theorem train_speed (length time : ℝ) (h1 : length = 300) (h2 : time = 6) :
  length / time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2502_250295


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l2502_250284

/-- The surface area of a rectangular prism formed by three cubes -/
def surface_area_rectangular_prism (a : ℝ) : ℝ := 14 * a^2

/-- The surface area of a single cube -/
def surface_area_cube (a : ℝ) : ℝ := 6 * a^2

theorem rectangular_prism_surface_area (a : ℝ) (h : a > 0) :
  surface_area_rectangular_prism a = 3 * surface_area_cube a - 4 * a^2 :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l2502_250284


namespace NUMINAMATH_CALUDE_chord_length_is_7_exists_unique_P_l2502_250240

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 16

-- Define point F
def F : ℝ × ℝ := (-2, 0)

-- Define the line x = -4
def line_x_eq_neg_4 (x y : ℝ) : Prop := x = -4

-- Define the property that G is the midpoint of GT
def is_midpoint_GT (G T : ℝ × ℝ) : Prop :=
  G.1 = (G.1 + T.1) / 2 ∧ G.2 = (G.2 + T.2) / 2

-- Theorem 1: The length of the chord cut by FG on C₁ is 7
theorem chord_length_is_7 (G : ℝ × ℝ) (T : ℝ × ℝ) :
  C₁ G.1 G.2 →
  line_x_eq_neg_4 T.1 T.2 →
  is_midpoint_GT G T →
  ∃ (A B : ℝ × ℝ), C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 49 :=
sorry

-- Theorem 2: There exists a unique point P(4, 0) such that |GP| = 2|GF| for all G on C₁
theorem exists_unique_P (P : ℝ × ℝ) :
  P = (4, 0) ↔
  ∀ (G : ℝ × ℝ), C₁ G.1 G.2 →
    (G.1 - P.1)^2 + (G.2 - P.2)^2 = 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_is_7_exists_unique_P_l2502_250240


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2502_250289

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.sin x

theorem derivative_f_at_one :
  deriv f 1 = 3 * Real.sin 1 + Real.cos 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2502_250289


namespace NUMINAMATH_CALUDE_tan_identity_l2502_250286

theorem tan_identity (α : ℝ) (h : Real.tan (π / 3 - α) = 1 / 3) :
  Real.tan (2 * π / 3 + α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_identity_l2502_250286


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2502_250222

theorem complex_modulus_problem (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : 
  Complex.abs z = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2502_250222


namespace NUMINAMATH_CALUDE_caterpillars_on_tree_l2502_250293

theorem caterpillars_on_tree (initial : ℕ) (hatched : ℕ) (left : ℕ) : 
  initial = 14 → hatched = 4 → left = 8 → 
  initial + hatched - left = 10 := by sorry

end NUMINAMATH_CALUDE_caterpillars_on_tree_l2502_250293


namespace NUMINAMATH_CALUDE_larger_number_problem_l2502_250276

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1000) (h3 : L = 10 * S + 10) : L = 1110 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2502_250276


namespace NUMINAMATH_CALUDE_q_value_l2502_250287

theorem q_value (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1/p + 1/q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_q_value_l2502_250287


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2502_250266

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y^2 = 4) :
  x + 2*y ≥ 3 * Real.rpow 4 (1/3) := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2502_250266


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2502_250245

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2502_250245


namespace NUMINAMATH_CALUDE_line_does_not_intersect_circle_l2502_250231

/-- Proves that a line does not intersect a circle given the radius and distance from center to line -/
theorem line_does_not_intersect_circle (r d : ℝ) (hr : r = 10) (hd : d = 13) :
  d > r → ¬ (∃ (p : ℝ × ℝ), p.1^2 + p.2^2 = r^2 ∧ d = |p.1|) :=
by sorry

end NUMINAMATH_CALUDE_line_does_not_intersect_circle_l2502_250231


namespace NUMINAMATH_CALUDE_two_middle_zeros_in_quotient_l2502_250268

/-- Count the number of zeros in the middle of a positive integer -/
def count_middle_zeros (n : ℕ) : ℕ :=
  sorry

/-- The quotient when 2010 is divided by 2 -/
def quotient : ℕ := 2010 / 2

theorem two_middle_zeros_in_quotient : count_middle_zeros quotient = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_middle_zeros_in_quotient_l2502_250268


namespace NUMINAMATH_CALUDE_total_packages_l2502_250263

theorem total_packages (num_trucks : ℕ) (packages_per_truck : ℕ) 
  (h1 : num_trucks = 7) 
  (h2 : packages_per_truck = 70) : 
  num_trucks * packages_per_truck = 490 := by
  sorry

end NUMINAMATH_CALUDE_total_packages_l2502_250263


namespace NUMINAMATH_CALUDE_inequality_solution_l2502_250243

theorem inequality_solution (x : ℝ) : 
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ 
  (0 < x ∧ x ≤ 1/4) ∨ (1 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2502_250243


namespace NUMINAMATH_CALUDE_scrap_rate_cost_relationship_l2502_250281

/-- Represents the regression line equation for pig iron cost -/
def regression_line (x : ℝ) : ℝ := 256 + 2 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_relationship :
  ∀ x : ℝ, regression_line (x + 1) = regression_line x + 2 :=
by sorry

end NUMINAMATH_CALUDE_scrap_rate_cost_relationship_l2502_250281


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l2502_250282

/-- Given points A and B, where A is at (0, 0) and B is on the line y = 3,
    if the slope of segment AB is 3/4, then the sum of the x- and y-coordinates of B is 7. -/
theorem point_coordinate_sum (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 → x + 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l2502_250282


namespace NUMINAMATH_CALUDE_sod_coverage_theorem_l2502_250229

/-- The number of square sod pieces needed to cover two rectangular areas -/
def sod_squares_needed (length1 width1 length2 width2 sod_size : ℕ) : ℕ :=
  ((length1 * width1 + length2 * width2) : ℕ) / (sod_size * sod_size)

/-- Theorem stating that 1500 squares of 2x2-foot sod are needed to cover two areas of 30x40 feet and 60x80 feet -/
theorem sod_coverage_theorem :
  sod_squares_needed 30 40 60 80 2 = 1500 :=
by sorry

end NUMINAMATH_CALUDE_sod_coverage_theorem_l2502_250229


namespace NUMINAMATH_CALUDE_inverse_proportion_l2502_250204

theorem inverse_proportion (x y : ℝ → ℝ) (k : ℝ) :
  (∀ t, x t * y t = k) →  -- x is inversely proportional to y
  x 2 = 4 →               -- x = 4 when y = 2
  y 2 = 2 →               -- y = 2 when x = 4
  y (-5) = -5 →           -- y = -5
  x (-5) = -8/5 :=        -- x = -8/5 when y = -5
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_l2502_250204


namespace NUMINAMATH_CALUDE_pete_miles_walked_l2502_250212

/-- Represents a pedometer with a maximum step count --/
structure Pedometer :=
  (max_steps : ℕ)

/-- Calculates the total number of steps given the number of resets and final reading --/
def total_steps (p : Pedometer) (resets : ℕ) (final_reading : ℕ) : ℕ :=
  resets * (p.max_steps + 1) + final_reading

/-- Converts steps to miles --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℚ :=
  (steps : ℚ) / (steps_per_mile : ℚ)

/-- Theorem stating the approximate number of miles Pete walked --/
theorem pete_miles_walked :
  let p : Pedometer := ⟨99999⟩
  let resets : ℕ := 44
  let final_reading : ℕ := 50000
  let steps_per_mile : ℕ := 1800
  let total_steps := total_steps p resets final_reading
  let miles_walked := steps_to_miles total_steps steps_per_mile
  ∃ ε > 0, abs (miles_walked - 2472.22) < ε := by
  sorry

end NUMINAMATH_CALUDE_pete_miles_walked_l2502_250212


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2502_250274

/-- An arithmetic sequence with the given properties has 13 terms -/
theorem arithmetic_sequence_terms (a d : ℝ) (n : ℕ) 
  (h1 : 3 * a + 3 * d = 34)
  (h2 : 3 * a + 3 * (n - 1) * d = 146)
  (h3 : n * (2 * a + (n - 1) * d) / 2 = 390)
  : n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2502_250274


namespace NUMINAMATH_CALUDE_volleyball_betting_strategy_exists_l2502_250267

theorem volleyball_betting_strategy_exists : ∃ (x₁ x₂ x₃ x₄ : ℝ),
  x₁ + x₂ + x₃ + x₄ = 1 ∧
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
  6 * x₁ ≥ 1 ∧ 2 * x₂ ≥ 1 ∧ 6 * x₃ ≥ 1 ∧ 7 * x₄ ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_betting_strategy_exists_l2502_250267


namespace NUMINAMATH_CALUDE_theta_value_l2502_250298

theorem theta_value (θ : Real)
  (h1 : 3 * Real.pi ≤ θ ∧ θ ≤ 4 * Real.pi)
  (h2 : Real.sqrt ((1 + Real.cos θ) / 2) + Real.sqrt ((1 - Real.cos θ) / 2) = Real.sqrt 6 / 2) :
  θ = 19 * Real.pi / 6 ∨ θ = 23 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_theta_value_l2502_250298


namespace NUMINAMATH_CALUDE_equal_numbers_l2502_250297

theorem equal_numbers (x y z : ℕ) 
  (h1 : x ∣ Nat.gcd y z)
  (h2 : y ∣ Nat.gcd x z)
  (h3 : z ∣ Nat.gcd x y)
  (h4 : x ∣ Nat.lcm y z)
  (h5 : y ∣ Nat.lcm x z)
  (h6 : z ∣ Nat.lcm x y) :
  x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_l2502_250297


namespace NUMINAMATH_CALUDE_function_value_comparison_l2502_250244

theorem function_value_comparison (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 1 - a) : 
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
sorry

end NUMINAMATH_CALUDE_function_value_comparison_l2502_250244


namespace NUMINAMATH_CALUDE_team_size_l2502_250261

theorem team_size (first_day_per_person : ℕ) (second_day_multiplier : ℕ) (third_day_total : ℕ) (total_blankets : ℕ) :
  first_day_per_person = 2 →
  second_day_multiplier = 3 →
  third_day_total = 22 →
  total_blankets = 142 →
  ∃ team_size : ℕ, 
    team_size * first_day_per_person + 
    team_size * first_day_per_person * second_day_multiplier + 
    third_day_total = total_blankets ∧
    team_size = 15 :=
by sorry

end NUMINAMATH_CALUDE_team_size_l2502_250261


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2502_250215

/-- Given a triangle with inradius 3 cm and area 30 cm², its perimeter is 20 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 3 → A = 30 → A = r * (p / 2) → p = 20 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2502_250215


namespace NUMINAMATH_CALUDE_maximize_electronic_thermometers_l2502_250228

/-- Represents the problem of maximizing electronic thermometers purchase --/
theorem maximize_electronic_thermometers
  (total_budget : ℕ)
  (mercury_cost : ℕ)
  (electronic_cost : ℕ)
  (total_students : ℕ)
  (h1 : total_budget = 300)
  (h2 : mercury_cost = 3)
  (h3 : electronic_cost = 10)
  (h4 : total_students = 53) :
  ∃ (x : ℕ), x ≤ total_students ∧
             x * electronic_cost + (total_students - x) * mercury_cost ≤ total_budget ∧
             ∀ (y : ℕ), y ≤ total_students →
                        y * electronic_cost + (total_students - y) * mercury_cost ≤ total_budget →
                        y ≤ x ∧
             x = 20 :=
by sorry

end NUMINAMATH_CALUDE_maximize_electronic_thermometers_l2502_250228


namespace NUMINAMATH_CALUDE_negative_cube_squared_l2502_250252

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l2502_250252


namespace NUMINAMATH_CALUDE_unique_prime_with_remainder_l2502_250273

theorem unique_prime_with_remainder : ∃! n : ℕ,
  20 < n ∧ n < 30 ∧
  Prime n ∧
  n % 8 = 5 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_remainder_l2502_250273


namespace NUMINAMATH_CALUDE_unique_valid_denomination_l2502_250220

def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 104 → ∃ a b c : ℕ, k = 7 * a + n * b + (n + 2) * c ∧
  ¬∃ a b c : ℕ, 104 = 7 * a + n * b + (n + 2) * c

theorem unique_valid_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_denomination_l2502_250220


namespace NUMINAMATH_CALUDE_no_real_solution_complex_roots_l2502_250283

theorem no_real_solution_complex_roots :
  ∀ x : ℂ, (2 * x - 36) / 3 = (3 * x^2 + 6 * x + 1) / 4 →
  (∃ b : ℝ, x = -5/9 + b * I ∨ x = -5/9 - b * I) ∧
  (∀ y : ℝ, (2 * y - 36) / 3 ≠ (3 * y^2 + 6 * y + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_complex_roots_l2502_250283


namespace NUMINAMATH_CALUDE_state_A_selection_percentage_l2502_250272

theorem state_A_selection_percentage : 
  ∀ (total_candidates : ℕ) (state_B_percentage : ℚ) (extra_selected : ℕ),
    total_candidates = 8000 →
    state_B_percentage = 7 / 100 →
    extra_selected = 80 →
    ∃ (state_A_percentage : ℚ),
      state_A_percentage * total_candidates + extra_selected = state_B_percentage * total_candidates ∧
      state_A_percentage = 6 / 100 := by
  sorry

end NUMINAMATH_CALUDE_state_A_selection_percentage_l2502_250272


namespace NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l2502_250217

def selling_price : ℝ := 670
def original_cost : ℝ := 536

theorem profit_percentage_is_25_percent : 
  (selling_price - original_cost) / original_cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l2502_250217


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l2502_250251

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x - 1| - a

-- Theorem 1
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f a x - 2 * |x - 7| ≤ 0) → a ≥ -12 := by sorry

-- Theorem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f 1 x + |x + 7| ≥ m) → m ≤ 7 := by sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l2502_250251


namespace NUMINAMATH_CALUDE_quadratic_polynomial_from_roots_and_point_l2502_250242

/-- Given a quadratic polynomial q(x) with roots at x = -2 and x = 3, and q(1) = -10,
    prove that q(x) = 5/3x^2 - 5/3x - 10 -/
theorem quadratic_polynomial_from_roots_and_point (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = -2 ∨ x = 3) →  -- roots at x = -2 and x = 3
  (∃ a b c, ∀ x, q x = a * x^2 + b * x + c) →  -- q is a quadratic polynomial
  q 1 = -10 →  -- q(1) = -10
  ∀ x, q x = 5/3 * x^2 - 5/3 * x - 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_from_roots_and_point_l2502_250242


namespace NUMINAMATH_CALUDE_breadth_is_five_l2502_250248

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 15 * breadth
  length_diff : length = breadth + 10

/-- The breadth of a rectangular plot with given properties is 5 meters -/
theorem breadth_is_five (plot : RectangularPlot) : plot.breadth = 5 := by
  sorry

end NUMINAMATH_CALUDE_breadth_is_five_l2502_250248


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2502_250241

/-- A quadratic function f(x) = ax² + bx satisfying specific conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (a b : ℝ) :
  (f a b 2 = 0) →
  (∃ (x : ℝ), f a b x = x ∧ (∀ y : ℝ, f a b y = y → y = x)) →
  (∀ x : ℝ, f a b x = -1/2 * x^2 + x) ∧
  (Set.Icc 0 3).image (f a b) = Set.Icc (-3/2) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2502_250241


namespace NUMINAMATH_CALUDE_pairings_equal_25_l2502_250269

/-- The number of bowls and glasses -/
def n : ℕ := 5

/-- The total number of possible pairings of bowls and glasses -/
def total_pairings : ℕ := n * n

/-- Theorem stating that the total number of pairings is 25 -/
theorem pairings_equal_25 : total_pairings = 25 := by
  sorry

end NUMINAMATH_CALUDE_pairings_equal_25_l2502_250269


namespace NUMINAMATH_CALUDE_set_membership_properties_l2502_250249

def A : Set Int := {x | ∃ k, x = 3 * k - 1}
def B : Set Int := {x | ∃ k, x = 3 * k + 1}
def C : Set Int := {x | ∃ k, x = 3 * k}

theorem set_membership_properties (a b c : Int) (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  (2 * a ∈ B) ∧ (2 * b ∈ A) ∧ (a + b ∈ C) := by
  sorry

end NUMINAMATH_CALUDE_set_membership_properties_l2502_250249


namespace NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l2502_250223

theorem spinsters_to_cats_ratio : 
  ∀ (S C : ℕ),
  S = 22 →
  C = S + 55 →
  (S : ℚ) / C = 2 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l2502_250223


namespace NUMINAMATH_CALUDE_bryan_work_hours_l2502_250291

/-- Represents Bryan's daily work schedule --/
structure WorkSchedule where
  customer_outreach : ℝ
  advertisement : ℝ
  marketing : ℝ

/-- Calculates the total working hours given a work schedule --/
def total_hours (schedule : WorkSchedule) : ℝ :=
  schedule.customer_outreach + schedule.advertisement + schedule.marketing

/-- Theorem stating Bryan's total working hours --/
theorem bryan_work_hours :
  ∀ (schedule : WorkSchedule),
    schedule.customer_outreach = 4 →
    schedule.advertisement = schedule.customer_outreach / 2 →
    schedule.marketing = 2 →
    total_hours schedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_bryan_work_hours_l2502_250291


namespace NUMINAMATH_CALUDE_student_number_problem_l2502_250285

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 106 → x = 122 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2502_250285


namespace NUMINAMATH_CALUDE_equation_solution_l2502_250206

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x + 1) - 2 * (x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2502_250206


namespace NUMINAMATH_CALUDE_sequence_formula_l2502_250279

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n > 1, a n - a (n-1) = n) :
  ∀ n : ℕ, n > 0 → a n = n * (n + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_sequence_formula_l2502_250279


namespace NUMINAMATH_CALUDE_min_buses_needed_l2502_250247

/-- The number of students to be transported -/
def total_students : ℕ := 540

/-- The maximum number of students each bus can hold -/
def bus_capacity : ℕ := 45

/-- The minimum number of buses needed is the ceiling of the quotient of total students divided by bus capacity -/
theorem min_buses_needed : 
  (total_students + bus_capacity - 1) / bus_capacity = 12 := by sorry

end NUMINAMATH_CALUDE_min_buses_needed_l2502_250247


namespace NUMINAMATH_CALUDE_karl_net_income_l2502_250259

/-- Represents the sale of boots and subsequent transactions -/
structure BootSale where
  initial_price : ℚ
  actual_sale_price : ℚ
  reduced_price : ℚ
  refund_amount : ℚ
  candy_expense : ℚ
  actual_refund : ℚ

/-- Calculates the net income from a boot sale -/
def net_income (sale : BootSale) : ℚ :=
  sale.actual_sale_price * 2 - sale.refund_amount

/-- Theorem stating that Karl's net income is 20 talers -/
theorem karl_net_income (sale : BootSale) 
  (h1 : sale.initial_price = 25)
  (h2 : sale.actual_sale_price = 12.5)
  (h3 : sale.reduced_price = 10)
  (h4 : sale.refund_amount = 5)
  (h5 : sale.candy_expense = 3)
  (h6 : sale.actual_refund = 1) :
  net_income sale = 20 := by
  sorry


end NUMINAMATH_CALUDE_karl_net_income_l2502_250259


namespace NUMINAMATH_CALUDE_triangle_inequality_l2502_250209

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  a * b = Real.sqrt 2 →
  a^2 + b^2 + c^2 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2502_250209


namespace NUMINAMATH_CALUDE_unique_digit_equation_l2502_250235

/-- Represents a mapping from symbols to digits -/
def SymbolMap := Char → Fin 10

/-- Checks if a SymbolMap assigns unique digits to different symbols -/
def isValidMap (m : SymbolMap) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Represents the equation "华 ÷ (3 * 好) = 杯赛" -/
def equationHolds (m : SymbolMap) : Prop :=
  (m '华').val = (m '杯').val * 100 + (m '赛').val * 10 + (m '赛').val

theorem unique_digit_equation :
  ∀ m : SymbolMap,
    isValidMap m →
    equationHolds m →
    (m '好').val = 2 := by sorry

end NUMINAMATH_CALUDE_unique_digit_equation_l2502_250235


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2502_250213

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 969 := by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2502_250213


namespace NUMINAMATH_CALUDE_factorization_sum_l2502_250277

theorem factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 19*x + 88 = (x + d)*(x + e)) →
  (∀ x : ℝ, x^2 - 23*x + 120 = (x - e)*(x - f)) →
  d + e + f = 31 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l2502_250277


namespace NUMINAMATH_CALUDE_parabola_curve_intersection_l2502_250202

/-- The parabola defined by y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The curve defined by y = k/x where k > 0 -/
def curve (k x y : ℝ) : Prop := k > 0 ∧ y = k / x

/-- The focus of the parabola y = (1/4)x^2 -/
def focus : ℝ × ℝ := (0, 1)

/-- A point is on both the parabola and the curve -/
def intersection_point (k x y : ℝ) : Prop :=
  parabola x y ∧ curve k x y

/-- The line from a point to the focus is perpendicular to the y-axis -/
def perpendicular_to_y_axis (x y : ℝ) : Prop :=
  x = (focus.1 - x)

theorem parabola_curve_intersection (k : ℝ) :
  (∃ x y : ℝ, intersection_point k x y ∧ perpendicular_to_y_axis x y) →
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_curve_intersection_l2502_250202


namespace NUMINAMATH_CALUDE_remainder_sum_modulo_l2502_250225

theorem remainder_sum_modulo (c d : ℤ) 
  (hc : c % 80 = 74)
  (hd : d % 120 = 114) : 
  (c + d) % 40 = 28 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_modulo_l2502_250225


namespace NUMINAMATH_CALUDE_brendas_age_l2502_250280

/-- Proves that Brenda's age is 8/3 years given the conditions in the problem. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)   -- Addison's age is four times Brenda's age
  (h2 : J = B + 8)   -- Janet is eight years older than Brenda
  (h3 : A = J)       -- Addison and Janet are twins
  : B = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l2502_250280


namespace NUMINAMATH_CALUDE_collinear_vectors_solution_l2502_250275

/-- Two vectors in R² -/
def m (x : ℝ) : ℝ × ℝ := (x, x + 2)
def n (x : ℝ) : ℝ × ℝ := (1, 3*x)

/-- Collinearity condition for two vectors -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem collinear_vectors_solution :
  ∀ x : ℝ, collinear (m x) (n x) → x = -2/3 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_collinear_vectors_solution_l2502_250275


namespace NUMINAMATH_CALUDE_initial_average_calculation_l2502_250230

theorem initial_average_calculation (n : ℕ) (correct_sum wrong_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 24)
  (h3 : wrong_sum = correct_sum - 10) :
  wrong_sum / n = 23 := by
sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l2502_250230


namespace NUMINAMATH_CALUDE_distance_AB_is_360_l2502_250205

/-- The distance between two points A and B --/
def distance_AB : ℝ := sorry

/-- The initial speed of the passenger train --/
def v_pass : ℝ := sorry

/-- The initial speed of the freight train --/
def v_freight : ℝ := sorry

/-- The time taken by the freight train to travel from A to B --/
def t_freight : ℝ := sorry

/-- The time difference between the passenger and freight trains --/
def time_diff : ℝ := 3.2

/-- The additional distance traveled by the passenger train --/
def additional_distance : ℝ := 288

/-- The speed increase for both trains --/
def speed_increase : ℝ := 10

/-- The new time difference after speed increase --/
def new_time_diff : ℝ := 2.4

theorem distance_AB_is_360 :
  v_pass * (t_freight - time_diff) = v_freight * t_freight + additional_distance ∧
  distance_AB / (v_freight + speed_increase) - distance_AB / (v_pass + speed_increase) = new_time_diff ∧
  distance_AB = v_freight * t_freight →
  distance_AB = 360 := by sorry

end NUMINAMATH_CALUDE_distance_AB_is_360_l2502_250205


namespace NUMINAMATH_CALUDE_class_size_l2502_250265

theorem class_size :
  ∀ (m d : ℕ),
  (m + d > 30) →
  (m + d < 40) →
  (3 * m = 5 * d) →
  (m + d = 32) :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l2502_250265


namespace NUMINAMATH_CALUDE_scientific_notation_of_41800000000_l2502_250292

theorem scientific_notation_of_41800000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 41800000000 = a * (10 : ℝ) ^ n ∧ a = 4.18 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_41800000000_l2502_250292


namespace NUMINAMATH_CALUDE_symmetric_points_m_value_l2502_250257

/-- Two points are symmetric about the origin if their coordinates are negatives of each other -/
def symmetric_about_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Given that point A(2, -1) is symmetric with point B(-2, m) about the origin, prove that m = 1 -/
theorem symmetric_points_m_value :
  let A : ℝ × ℝ := (2, -1)
  let B : ℝ × ℝ := (-2, m)
  symmetric_about_origin A B → m = 1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_m_value_l2502_250257


namespace NUMINAMATH_CALUDE_mary_anne_sparkling_water_cost_l2502_250260

/-- The amount Mary Anne spends on sparkling water in a year -/
def sparkling_water_cost (daily_consumption : ℚ) (bottle_cost : ℚ) : ℚ :=
  (365 : ℚ) * daily_consumption * bottle_cost

theorem mary_anne_sparkling_water_cost :
  sparkling_water_cost (1/5) 2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_mary_anne_sparkling_water_cost_l2502_250260


namespace NUMINAMATH_CALUDE_non_opaque_arrangements_l2502_250224

/-- Represents the number of glasses in the stack -/
def num_glasses : ℕ := 5

/-- Represents the number of possible rotations for each glass -/
def num_rotations : ℕ := 3

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ := num_glasses.factorial * num_rotations ^ (num_glasses - 1)

/-- Calculates the number of opaque arrangements -/
def opaque_arrangements : ℕ := 50 * num_glasses.factorial

/-- Theorem stating the number of non-opaque arrangements -/
theorem non_opaque_arrangements :
  total_arrangements - opaque_arrangements = 3720 :=
sorry

end NUMINAMATH_CALUDE_non_opaque_arrangements_l2502_250224


namespace NUMINAMATH_CALUDE_min_sum_of_product_72_l2502_250264

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_72_l2502_250264
