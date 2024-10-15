import Mathlib

namespace NUMINAMATH_CALUDE_shaded_ratio_is_one_ninth_l3393_339331

-- Define the structure of our square grid
def SquareGrid :=
  { n : ℕ // n > 0 }

-- Define the large square
def LargeSquare : SquareGrid :=
  ⟨6, by norm_num⟩

-- Define the number of squares in the shaded region
def ShadedSquares : ℕ := 4

-- Define the ratio of shaded area to total area
def ShadedRatio (grid : SquareGrid) (shaded : ℕ) : ℚ :=
  shaded / (grid.val ^ 2 : ℚ)

-- Theorem statement
theorem shaded_ratio_is_one_ninth :
  ShadedRatio LargeSquare ShadedSquares = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_ratio_is_one_ninth_l3393_339331


namespace NUMINAMATH_CALUDE_exterior_angle_sum_is_360_l3393_339313

/-- A convex polygon with n sides and equilateral triangles attached to each side -/
structure ConvexPolygonWithTriangles where
  n : ℕ  -- number of sides of the original polygon
  [n_pos : Fact (n > 0)]

/-- The sum of exterior angles of a convex polygon with attached equilateral triangles -/
def exterior_angle_sum (p : ConvexPolygonWithTriangles) : ℝ :=
  360

/-- Theorem: The sum of all exterior angles in a convex polygon with attached equilateral triangles is 360° -/
theorem exterior_angle_sum_is_360 (p : ConvexPolygonWithTriangles) :
  exterior_angle_sum p = 360 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_sum_is_360_l3393_339313


namespace NUMINAMATH_CALUDE_monomial_equality_l3393_339340

-- Define variables
variable (a b : ℝ)
variable (x : ℝ)

-- Define the theorem
theorem monomial_equality (h : x * (2 * a^2 * b) = 2 * a^3 * b) : x = a := by
  sorry

end NUMINAMATH_CALUDE_monomial_equality_l3393_339340


namespace NUMINAMATH_CALUDE_value_of_m_l3393_339359

theorem value_of_m (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l3393_339359


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l3393_339322

theorem technician_round_trip_completion (distance : ℝ) (h : distance > 0) :
  let one_way := distance
  let return_portion := 0.4 * distance
  let total_distance := 2 * distance
  let completed_distance := one_way + return_portion
  (completed_distance / total_distance) * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l3393_339322


namespace NUMINAMATH_CALUDE_earth_surface_usage_l3393_339388

/-- The fraction of the Earth's surface that is land -/
def land_fraction : ℚ := 1/3

/-- The fraction of land that is inhabitable -/
def inhabitable_fraction : ℚ := 2/3

/-- The fraction of inhabitable land used for agriculture and urban development -/
def used_fraction : ℚ := 3/4

/-- The fraction of the Earth's surface used for agriculture or urban purposes -/
def agriculture_urban_fraction : ℚ := land_fraction * inhabitable_fraction * used_fraction

theorem earth_surface_usage :
  agriculture_urban_fraction = 1/6 := by sorry

end NUMINAMATH_CALUDE_earth_surface_usage_l3393_339388


namespace NUMINAMATH_CALUDE_factorization_equality_l3393_339378

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3393_339378


namespace NUMINAMATH_CALUDE_complex_magnitude_l3393_339392

theorem complex_magnitude (z : ℂ) (a b : ℝ) (h1 : z = Complex.mk a b) (h2 : a ≠ 0) 
  (h3 : Complex.abs z ^ 2 - 2 * z = Complex.mk 1 2) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3393_339392


namespace NUMINAMATH_CALUDE_probability_range_for_event_A_l3393_339387

theorem probability_range_for_event_A (p : ℝ) : 
  (0 ≤ p ∧ p < 1) →
  (4 * p * (1 - p)^3 ≤ 6 * p^2 * (1 - p)^2) →
  0.4 ≤ p ∧ p < 1 :=
sorry

end NUMINAMATH_CALUDE_probability_range_for_event_A_l3393_339387


namespace NUMINAMATH_CALUDE_polynomial_roots_l3393_339349

theorem polynomial_roots : ∃ (x₁ x₂ x₃ x₄ : ℂ),
  (x₁ = (7 + Real.sqrt 37) / 6) ∧
  (x₂ = (7 - Real.sqrt 37) / 6) ∧
  (x₃ = (-3 + Real.sqrt 5) / 2) ∧
  (x₄ = (-3 - Real.sqrt 5) / 2) ∧
  (∀ x : ℂ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3393_339349


namespace NUMINAMATH_CALUDE_log_equation_solution_l3393_339337

-- Define the logarithm function
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x = 17 ∧ log2 ((3*x + 9) / (5*x - 3)) + log2 ((5*x - 3) / (x - 2)) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3393_339337


namespace NUMINAMATH_CALUDE_shoe_probability_theorem_l3393_339341

/-- Represents the number of pairs of shoes of a specific color -/
structure ColorPairs :=
  (count : ℕ)

/-- Represents Sue's shoe collection -/
structure ShoeCollection :=
  (total_pairs : ℕ)
  (black : ColorPairs)
  (brown : ColorPairs)
  (gray : ColorPairs)

/-- Calculates the probability of picking two shoes of the same color, one left and one right -/
def probability_same_color_different_feet (collection : ShoeCollection) : ℚ :=
  let total_shoes := 2 * collection.total_pairs
  let black_prob := (2 * collection.black.count : ℚ) / total_shoes * collection.black.count / (total_shoes - 1)
  let brown_prob := (2 * collection.brown.count : ℚ) / total_shoes * collection.brown.count / (total_shoes - 1)
  let gray_prob := (2 * collection.gray.count : ℚ) / total_shoes * collection.gray.count / (total_shoes - 1)
  black_prob + brown_prob + gray_prob

theorem shoe_probability_theorem (sue_collection : ShoeCollection) 
  (h1 : sue_collection.total_pairs = 12)
  (h2 : sue_collection.black.count = 7)
  (h3 : sue_collection.brown.count = 3)
  (h4 : sue_collection.gray.count = 2) :
  probability_same_color_different_feet sue_collection = 31 / 138 := by
  sorry

end NUMINAMATH_CALUDE_shoe_probability_theorem_l3393_339341


namespace NUMINAMATH_CALUDE_solution_set_xfx_less_than_zero_l3393_339355

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

theorem solution_set_xfx_less_than_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_on_positive f)
  (h_f_neg_three : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = {x | x < -3 ∨ x > 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_xfx_less_than_zero_l3393_339355


namespace NUMINAMATH_CALUDE_sin_225_degrees_l3393_339346

theorem sin_225_degrees : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l3393_339346


namespace NUMINAMATH_CALUDE_ultramindmaster_codes_l3393_339358

/-- The number of available colors in UltraMindmaster -/
def num_colors : ℕ := 8

/-- The number of slots in each secret code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in UltraMindmaster -/
def num_codes : ℕ := num_colors ^ num_slots

theorem ultramindmaster_codes :
  num_codes = 32768 := by
  sorry

end NUMINAMATH_CALUDE_ultramindmaster_codes_l3393_339358


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l3393_339363

theorem cubic_roots_sum_of_squares (α β γ : ℂ) : 
  (α^3 - 6*α^2 + 11*α - 6 = 0) → 
  (β^3 - 6*β^2 + 11*β - 6 = 0) → 
  (γ^3 - 6*γ^2 + 11*γ - 6 = 0) → 
  α^2 + β^2 + γ^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l3393_339363


namespace NUMINAMATH_CALUDE_right_triangle_third_side_square_l3393_339367

theorem right_triangle_third_side_square (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ a^2 + b^2 = c^2) ∨ (a = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) →
  c^2 = 25 ∨ b^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_square_l3393_339367


namespace NUMINAMATH_CALUDE_barry_cycling_time_difference_barry_cycling_proof_l3393_339321

theorem barry_cycling_time_difference : ℝ → Prop :=
  λ time_diff : ℝ =>
    let total_distance : ℝ := 4 * 3
    let time_at_varying_speeds : ℝ := 2 * (3 / 6) + 1 * (3 / 3) + 1 * (3 / 5)
    let time_at_constant_speed : ℝ := total_distance / 5
    let time_diff_hours : ℝ := time_at_varying_speeds - time_at_constant_speed
    time_diff = time_diff_hours * 60 ∧ time_diff = 42

theorem barry_cycling_proof : barry_cycling_time_difference 42 := by
  sorry

end NUMINAMATH_CALUDE_barry_cycling_time_difference_barry_cycling_proof_l3393_339321


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_nine_l3393_339316

theorem eight_digit_divisible_by_nine (n : Nat) : 
  n ≤ 9 →
  (854 * 10^7 + n * 10^6 + 5 * 10^5 + 2 * 10^4 + 6 * 10^3 + 8 * 10^2 + 6 * 10 + 8) % 9 = 0 →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_nine_l3393_339316


namespace NUMINAMATH_CALUDE_blackboard_numbers_l3393_339374

theorem blackboard_numbers (n : ℕ) (S : ℕ) (x : ℕ) : 
  S / n = 30 →
  (S + 100) / (n + 1) = 40 →
  (S + 100 + x) / (n + 2) = 50 →
  x = 120 := by
sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l3393_339374


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3393_339390

-- Define the vector space
variable (V : Type) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors and their properties
variable (a b : V)
variable (h1 : a = (1 : ℝ) • (1, 0))
variable (h2 : ‖b‖ = 1)
variable (h3 : inner a b = -(1/2 : ℝ) * ‖a‖ * ‖b‖)

-- State the theorem
theorem vector_sum_magnitude :
  ‖a + 2 • b‖ = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3393_339390


namespace NUMINAMATH_CALUDE_min_even_integers_l3393_339350

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 26 → 
  a + b + c + d = 41 → 
  a + b + c + d + e + f = 57 → 
  ∃ (n : ℕ), n ≥ 1 ∧ 
  (∀ (m : ℕ), m < n → 
    ¬∃ (evens : Finset ℤ), evens.card = m ∧ 
    (∀ x ∈ evens, Even x) ∧ 
    evens ⊆ {a, b, c, d, e, f}) :=
sorry

end NUMINAMATH_CALUDE_min_even_integers_l3393_339350


namespace NUMINAMATH_CALUDE_f_properties_l3393_339308

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi/6) + 1

theorem f_properties :
  let T := Real.pi
  let interval := Set.Icc (Real.pi/4) ((2*Real.pi)/3)
  (∀ x, f (x + T) = f x) ∧  -- Smallest positive period
  (∀ x ∈ interval, f x ≤ 2) ∧  -- Maximum value
  (∃ x ∈ interval, f x = 2) ∧  -- Maximum value is attained
  (∀ x ∈ interval, f x ≥ -1) ∧  -- Minimum value
  (∃ x ∈ interval, f x = -1) :=  -- Minimum value is attained
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3393_339308


namespace NUMINAMATH_CALUDE_weight_gain_difference_l3393_339396

def weight_gain_problem (orlando_gain jose_gain fernando_gain : ℕ) : Prop :=
  orlando_gain = 5 ∧
  jose_gain = 2 * orlando_gain + 2 ∧
  fernando_gain < jose_gain / 2 ∧
  orlando_gain + jose_gain + fernando_gain = 20

theorem weight_gain_difference (orlando_gain jose_gain fernando_gain : ℕ) 
  (h : weight_gain_problem orlando_gain jose_gain fernando_gain) :
  jose_gain / 2 - fernando_gain = 3 := by
  sorry

end NUMINAMATH_CALUDE_weight_gain_difference_l3393_339396


namespace NUMINAMATH_CALUDE_consecutive_triangular_not_square_infinitely_many_square_products_l3393_339303

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Statement: The product of two consecutive triangular numbers is not a perfect square -/
theorem consecutive_triangular_not_square (n : ℕ) (h : n > 1) :
  ¬ ∃ m : ℕ, triangular_number (n - 1) * triangular_number n = m^2 := by sorry

/-- Statement: For each triangular number, there exist infinitely many larger triangular numbers
    such that their product is a perfect square -/
theorem infinitely_many_square_products (n : ℕ) :
  ∃ f : ℕ → ℕ, Monotone f ∧ (∀ k : ℕ, f k > n) ∧
  (∀ k : ℕ, ∃ m : ℕ, triangular_number n * triangular_number (f k) = m^2) := by sorry

end NUMINAMATH_CALUDE_consecutive_triangular_not_square_infinitely_many_square_products_l3393_339303


namespace NUMINAMATH_CALUDE_albert_betty_age_ratio_l3393_339336

/-- Represents the ages of Albert, Mary, and Betty -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.mary = ages.albert - 22 ∧
  ages.betty = 11

/-- The theorem to prove -/
theorem albert_betty_age_ratio (ages : Ages) :
  age_conditions ages → (ages.albert : ℚ) / ages.betty = 4 := by
  sorry

#check albert_betty_age_ratio

end NUMINAMATH_CALUDE_albert_betty_age_ratio_l3393_339336


namespace NUMINAMATH_CALUDE_triangle_formation_l3393_339309

-- Define the triangle formation condition
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_formation (a : ℝ) : 
  can_form_triangle 5 a 9 ↔ a = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3393_339309


namespace NUMINAMATH_CALUDE_intersection_equals_S_l3393_339393

def S : Set ℝ := {y | ∃ x : ℝ, y = 3 * x}
def T : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

theorem intersection_equals_S : S ∩ T = S := by sorry

end NUMINAMATH_CALUDE_intersection_equals_S_l3393_339393


namespace NUMINAMATH_CALUDE_second_number_value_l3393_339381

theorem second_number_value (x : ℝ) (h : 8000 * x = 480 * (10^5)) : x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l3393_339381


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l3393_339320

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 8) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l3393_339320


namespace NUMINAMATH_CALUDE_teaching_years_sum_l3393_339304

/-- The combined years of teaching for Virginia, Adrienne, and Dennis -/
def combined_years (virginia adrienne dennis : ℕ) : ℕ := virginia + adrienne + dennis

/-- Theorem stating the combined years of teaching given the conditions -/
theorem teaching_years_sum :
  ∀ (virginia adrienne dennis : ℕ),
  virginia = adrienne + 9 →
  virginia = dennis - 9 →
  dennis = 40 →
  combined_years virginia adrienne dennis = 93 := by
sorry

end NUMINAMATH_CALUDE_teaching_years_sum_l3393_339304


namespace NUMINAMATH_CALUDE_abs_T_equals_1024_l3393_339382

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^19 - (1 - i)^19

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by sorry

end NUMINAMATH_CALUDE_abs_T_equals_1024_l3393_339382


namespace NUMINAMATH_CALUDE_complex_cube_root_sum_l3393_339317

theorem complex_cube_root_sum (a b : ℤ) (z : ℂ) : 
  z = a + b * Complex.I ∧ z^3 = 2 + 11 * Complex.I → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_sum_l3393_339317


namespace NUMINAMATH_CALUDE_team_selection_count_l3393_339311

def boys := 10
def girls := 10
def team_size := 8
def min_boys := 3

def select_team (b g : ℕ) (t m : ℕ) : ℕ :=
  (Nat.choose b 3 * Nat.choose g 5) +
  (Nat.choose b 4 * Nat.choose g 4) +
  (Nat.choose b 5 * Nat.choose g 3) +
  (Nat.choose b 6 * Nat.choose g 2) +
  (Nat.choose b 7 * Nat.choose g 1) +
  (Nat.choose b 8 * Nat.choose g 0)

theorem team_selection_count :
  select_team boys girls team_size min_boys = 114275 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l3393_339311


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l3393_339318

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1050 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l3393_339318


namespace NUMINAMATH_CALUDE_exactly_one_not_through_origin_l3393_339366

def f₁ (x : ℝ) : ℝ := x^4 + 1
def f₂ (x : ℝ) : ℝ := x^4 + x
def f₃ (x : ℝ) : ℝ := x^4 + x^2
def f₄ (x : ℝ) : ℝ := x^4 + x^3

def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

theorem exactly_one_not_through_origin :
  ∃! i : Fin 4, ¬passes_through_origin (match i with
    | 0 => f₁
    | 1 => f₂
    | 2 => f₃
    | 3 => f₄) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_not_through_origin_l3393_339366


namespace NUMINAMATH_CALUDE_probability_third_key_opens_door_l3393_339398

/-- The probability of opening a door with the third key, given 5 keys with only one correct key --/
theorem probability_third_key_opens_door : 
  ∀ (n : ℕ) (p : ℝ),
    n = 5 →  -- There are 5 keys
    p = 1 / n →  -- The probability of selecting the correct key is 1/n
    p = 1 / 5  -- The probability of opening the door on the third attempt is 1/5
    := by sorry

end NUMINAMATH_CALUDE_probability_third_key_opens_door_l3393_339398


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l3393_339376

/-- A line parallel to the X-axis passing through the point (3, -2) has the equation y = -2 -/
theorem line_parallel_to_x_axis (line : Set (ℝ × ℝ)) : 
  ((3 : ℝ), -2) ∈ line →  -- The line passes through the point (3, -2)
  (∀ (x y₁ y₂ : ℝ), ((x, y₁) ∈ line ∧ (x, y₂) ∈ line) → y₁ = y₂) →  -- The line is parallel to the X-axis
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = -2 :=  -- The equation of the line is y = -2
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l3393_339376


namespace NUMINAMATH_CALUDE_earliest_retirement_year_l3393_339351

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Employee's age in a given year -/
def age_in_year (hire_year : ℕ) (hire_age : ℕ) (current_year : ℕ) : ℕ :=
  (current_year - hire_year) + hire_age

/-- Employee's years of employment in a given year -/
def years_employed (hire_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - hire_year

theorem earliest_retirement_year 
  (hire_year : ℕ) 
  (hire_age : ℕ) 
  (retirement_year : ℕ) :
  hire_year = 1988 →
  hire_age = 32 →
  retirement_year = 2007 →
  rule_of_70 (age_in_year hire_year hire_age retirement_year) 
             (years_employed hire_year retirement_year) ∧
  ∀ y : ℕ, y < retirement_year →
    ¬(rule_of_70 (age_in_year hire_year hire_age y) 
                 (years_employed hire_year y)) :=
by sorry

end NUMINAMATH_CALUDE_earliest_retirement_year_l3393_339351


namespace NUMINAMATH_CALUDE_ednas_neighbors_l3393_339354

/-- The number of cookies Edna made -/
def total_cookies : ℕ := 150

/-- The number of cookies each neighbor (except Sarah) took -/
def cookies_per_neighbor : ℕ := 10

/-- The number of cookies Sarah took -/
def sarah_cookies : ℕ := 12

/-- The number of cookies left for the last neighbor -/
def cookies_left : ℕ := 8

/-- The number of Edna's neighbors -/
def num_neighbors : ℕ := 14

theorem ednas_neighbors :
  total_cookies = num_neighbors * cookies_per_neighbor + (sarah_cookies - cookies_per_neighbor) + cookies_left :=
by sorry

end NUMINAMATH_CALUDE_ednas_neighbors_l3393_339354


namespace NUMINAMATH_CALUDE_choose_computers_l3393_339360

theorem choose_computers (n : ℕ) : 
  (Nat.choose 3 2 * Nat.choose 3 1) + (Nat.choose 3 1 * Nat.choose 3 2) = 18 :=
by sorry

end NUMINAMATH_CALUDE_choose_computers_l3393_339360


namespace NUMINAMATH_CALUDE_no_geometric_progression_with_11_12_13_l3393_339338

theorem no_geometric_progression_with_11_12_13 :
  ¬ ∃ (a q : ℝ) (k l n : ℕ), 
    (k < l ∧ l < n) ∧
    (a * q ^ k = 11) ∧
    (a * q ^ l = 12) ∧
    (a * q ^ n = 13) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_progression_with_11_12_13_l3393_339338


namespace NUMINAMATH_CALUDE_milk_percentage_after_three_replacements_l3393_339353

/-- Represents the percentage of milk remaining after one replacement operation -/
def milk_after_one_replacement (initial_milk_percentage : Real) : Real :=
  initial_milk_percentage * 0.8

/-- Represents the percentage of milk remaining after three replacement operations -/
def milk_after_three_replacements (initial_milk_percentage : Real) : Real :=
  milk_after_one_replacement (milk_after_one_replacement (milk_after_one_replacement initial_milk_percentage))

theorem milk_percentage_after_three_replacements :
  milk_after_three_replacements 100 = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_milk_percentage_after_three_replacements_l3393_339353


namespace NUMINAMATH_CALUDE_min_value_ab_l3393_339324

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b - 2 * a - b = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y - 2 * x - y = 0 → a * b ≤ x * y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y - 2 * x - y = 0 ∧ x * y = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l3393_339324


namespace NUMINAMATH_CALUDE_square_root_of_four_l3393_339327

theorem square_root_of_four : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3393_339327


namespace NUMINAMATH_CALUDE_special_polygon_area_l3393_339379

/-- A polygon with special properties -/
structure SpecialPolygon where
  sides : ℕ
  perimeter : ℝ
  is_perpendicular : Bool
  is_equal_length : Bool

/-- The area of a special polygon -/
def area (p : SpecialPolygon) : ℝ := sorry

/-- Theorem: The area of a special polygon with 36 sides and perimeter 72 is 144 -/
theorem special_polygon_area :
  ∀ (p : SpecialPolygon),
    p.sides = 36 ∧
    p.perimeter = 72 ∧
    p.is_perpendicular ∧
    p.is_equal_length →
    area p = 144 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_area_l3393_339379


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_f_non_positive_iff_m_eq_one_l3393_339362

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem f_monotonicity_and_range (m : ℝ) :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ∨
  (∃ c, 0 < c ∧ 
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < c → f m x₁ < f m x₂) ∧
    (∀ x₁ x₂, c < x₁ ∧ x₁ < x₂ → f m x₁ > f m x₂)) :=
sorry

theorem f_non_positive_iff_m_eq_one (m : ℝ) :
  (∀ x, 0 < x → f m x ≤ 0) ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_f_non_positive_iff_m_eq_one_l3393_339362


namespace NUMINAMATH_CALUDE_abby_and_damon_weight_l3393_339371

/-- Given the weights of pairs of people, prove that Abby and Damon's combined weight is 285 pounds. -/
theorem abby_and_damon_weight
  (a b c d : ℝ)  -- Weights of Abby, Bart, Cindy, and Damon
  (h1 : a + b = 260)  -- Abby and Bart's combined weight
  (h2 : b + c = 245)  -- Bart and Cindy's combined weight
  (h3 : c + d = 270)  -- Cindy and Damon's combined weight
  : a + d = 285 := by
  sorry

#check abby_and_damon_weight

end NUMINAMATH_CALUDE_abby_and_damon_weight_l3393_339371


namespace NUMINAMATH_CALUDE_ellipse_equation_and_sum_l3393_339345

theorem ellipse_equation_and_sum (t : ℝ) :
  let x := (3 * (Real.sin t - 2)) / (3 - Real.cos t)
  let y := (4 * (Real.cos t - 6)) / (3 - Real.cos t)
  ∃ (A B C D E F : ℤ),
    (144 : ℝ) * x^2 - 96 * x * y + 25 * y^2 + 192 * x - 400 * y + 400 = 0 ∧
    Int.gcd A (Int.gcd B (Int.gcd C (Int.gcd D (Int.gcd E F)))) = 1 ∧
    Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1257 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_sum_l3393_339345


namespace NUMINAMATH_CALUDE_books_loaned_out_l3393_339397

/-- Proves the number of books loaned out given initial and final book counts and return rate -/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 150)
  (h2 : final_books = 122)
  (h3 : return_rate = 65 / 100) :
  ∃ (loaned_books : ℕ), 
    (initial_books : ℚ) - (loaned_books : ℚ) * (1 - return_rate) = final_books ∧ 
    loaned_books = 80 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_l3393_339397


namespace NUMINAMATH_CALUDE_circle_coloring_theorem_l3393_339357

/-- Represents a circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a coloring of the plane -/
def Coloring := ℝ × ℝ → Bool

/-- Checks if two points are on opposite sides of a circle -/
def oppositeSides (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2)
  let d2 := Real.sqrt ((p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2)
  (d1 < c.radius ∧ d2 > c.radius) ∨ (d1 > c.radius ∧ d2 < c.radius)

/-- Checks if a coloring is valid for a given set of circles -/
def validColoring (circles : List Circle) (coloring : Coloring) : Prop :=
  ∀ c ∈ circles, ∀ p1 p2 : ℝ × ℝ, oppositeSides c p1 p2 → coloring p1 ≠ coloring p2

theorem circle_coloring_theorem (n : ℕ) (hn : n > 0) (circles : List Circle) 
    (hc : circles.length = n) : 
    ∃ coloring : Coloring, validColoring circles coloring := by
  sorry

end NUMINAMATH_CALUDE_circle_coloring_theorem_l3393_339357


namespace NUMINAMATH_CALUDE_dog_distance_l3393_339394

theorem dog_distance (s : ℝ) (ivan_speed dog_speed : ℝ) : 
  s > 0 → 
  ivan_speed > 0 → 
  dog_speed > 0 → 
  s = 3 → 
  dog_speed = 3 * ivan_speed → 
  (∃ t : ℝ, t > 0 ∧ ivan_speed * t = s / 4 ∧ dog_speed * t = 3 * s / 4) → 
  (dog_speed * (s / ivan_speed)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_dog_distance_l3393_339394


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_alpha_l3393_339343

/-- Given two parallel vectors a and b, where a = (6, 8) and b = (sinα, cosα), prove that tanα = 3/4 -/
theorem parallel_vectors_tan_alpha (α : Real) : 
  let a : Fin 2 → Real := ![6, 8]
  let b : Fin 2 → Real := ![Real.sin α, Real.cos α]
  (∃ (k : Real), k ≠ 0 ∧ (∀ i, a i = k * b i)) → 
  Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_alpha_l3393_339343


namespace NUMINAMATH_CALUDE_parallelogram_bisector_slope_l3393_339319

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Checks if a line through the origin bisects a parallelogram into two congruent polygons -/
def bisects_parallelogram (m n : ℕ) (p : Parallelogram) : Prop :=
  -- This is a placeholder for the actual condition
  True

/-- The main theorem -/
theorem parallelogram_bisector_slope (p : Parallelogram) :
  p.v1 = ⟨20, 90⟩ ∧
  p.v2 = ⟨20, 228⟩ ∧
  p.v3 = ⟨56, 306⟩ ∧
  p.v4 = ⟨56, 168⟩ ∧
  bisects_parallelogram 369 76 p →
  369 / 76 = (p.v3.y - p.v1.y) / (p.v3.x - p.v1.x) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_slope_l3393_339319


namespace NUMINAMATH_CALUDE_solution_set_implies_a_range_l3393_339315

theorem solution_set_implies_a_range :
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) → a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_range_l3393_339315


namespace NUMINAMATH_CALUDE_lara_age_proof_l3393_339330

/-- Lara's age 10 years from now, given her age 7 years ago -/
def lara_future_age (age_7_years_ago : ℕ) : ℕ :=
  age_7_years_ago + 7 + 10

/-- Theorem stating Lara's age 10 years from now -/
theorem lara_age_proof :
  lara_future_age 9 = 26 := by
  sorry

end NUMINAMATH_CALUDE_lara_age_proof_l3393_339330


namespace NUMINAMATH_CALUDE_sin_4x_eq_sin_2x_solution_l3393_339368

open Set
open Real

def solution_set : Set ℝ := {π/6, π/2, π, 5*π/6, 7*π/6}

theorem sin_4x_eq_sin_2x_solution (x : ℝ) :
  x ∈ Ioo 0 (3*π/2) →
  (sin (4*x) = sin (2*x)) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_sin_4x_eq_sin_2x_solution_l3393_339368


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l3393_339377

/-- The growth factor of bacteria per cycle -/
def growth_factor : ℕ := 4

/-- The duration of one growth cycle in hours -/
def cycle_duration : ℕ := 5

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 1000

/-- The final number of bacteria -/
def final_bacteria : ℕ := 256000

/-- The number of cycles needed to reach the final bacteria count -/
def num_cycles : ℕ := 4

theorem bacteria_growth_time :
  cycle_duration * num_cycles =
    (final_bacteria / initial_bacteria).log growth_factor * cycle_duration :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l3393_339377


namespace NUMINAMATH_CALUDE_inequality_solution_l3393_339329

theorem inequality_solution (x : ℕ) : 5 * x + 3 < 3 * (2 + x) ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3393_339329


namespace NUMINAMATH_CALUDE_increase_by_fifty_percent_l3393_339375

theorem increase_by_fifty_percent (initial : ℝ) (increase : ℝ) (result : ℝ) : 
  initial = 350 → increase = 0.5 → result = initial * (1 + increase) → result = 525 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_fifty_percent_l3393_339375


namespace NUMINAMATH_CALUDE_f_inv_composition_l3393_339380

-- Define the function f
def f : ℕ → ℕ
| 2 => 5
| 3 => 7
| 4 => 11
| 5 => 17
| 6 => 23
| 7 => 40  -- Extended definition
| _ => 0   -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 5 => 2
| 7 => 3
| 11 => 4
| 17 => 5
| 23 => 6
| 40 => 7  -- Extended definition
| _ => 0   -- Default case for other inputs

-- Theorem statement
theorem f_inv_composition : f_inv ((f_inv 23)^2 + (f_inv 5)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_inv_composition_l3393_339380


namespace NUMINAMATH_CALUDE_plot_length_is_sixty_l3393_339389

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ
  lengthExcess : ℝ
  lengthIsTwentyMoreThanBreadth : length = breadth + lengthExcess
  fencingCostEquation : fencingCostPerMeter * (2 * (length + breadth)) = totalFencingCost

/-- Theorem stating that under given conditions, the length of the plot is 60 meters -/
theorem plot_length_is_sixty (plot : RectangularPlot)
  (h1 : plot.lengthExcess = 20)
  (h2 : plot.fencingCostPerMeter = 26.5)
  (h3 : plot.totalFencingCost = 5300) :
  plot.length = 60 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_is_sixty_l3393_339389


namespace NUMINAMATH_CALUDE_remainder_seven_twelfth_mod_hundred_l3393_339365

theorem remainder_seven_twelfth_mod_hundred : 7^12 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_twelfth_mod_hundred_l3393_339365


namespace NUMINAMATH_CALUDE_f_difference_l3393_339372

/-- Sum of all positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℚ := (sigma n + n) / n

/-- Theorem stating the result of f(540) - f(180) -/
theorem f_difference : f 540 - f 180 = 7 / 90 := by sorry

end NUMINAMATH_CALUDE_f_difference_l3393_339372


namespace NUMINAMATH_CALUDE_thirteenth_result_l3393_339314

theorem thirteenth_result (results : List ℝ) 
  (h1 : results.length = 25)
  (h2 : results.sum / 25 = 19)
  (h3 : (results.take 12).sum / 12 = 14)
  (h4 : (results.drop 13).sum / 12 = 17) :
  results[12] = 103 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_result_l3393_339314


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l3393_339332

/-- Represents a ball with a color and label -/
structure Ball where
  color : String
  label : Char

/-- Represents the bag of balls -/
def bag : List Ball := [
  { color := "yellow", label := 'a' },
  { color := "yellow", label := 'b' },
  { color := "red", label := 'c' },
  { color := "red", label := 'd' }
]

/-- Calculates the probability of drawing a yellow ball on the first draw -/
def probYellowFirst (bag : List Ball) : ℚ :=
  (bag.filter (fun b => b.color = "yellow")).length / bag.length

/-- Calculates the probability of drawing a yellow ball on the second draw -/
def probYellowSecond (bag : List Ball) : ℚ :=
  let totalOutcomes := bag.length * (bag.length - 1)
  let favorableOutcomes := 2 * (bag.length - 2)
  favorableOutcomes / totalOutcomes

theorem yellow_ball_probability (bag : List Ball) :
  probYellowFirst bag = 1/2 ∧ probYellowSecond bag = 1/2 :=
sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l3393_339332


namespace NUMINAMATH_CALUDE_joystick_payment_ratio_l3393_339364

/-- Proves that the ratio of Frank's payment for the joystick to the total cost of the joystick is 1:4 -/
theorem joystick_payment_ratio :
  ∀ (computer_table computer_chair joystick frank_joystick eman_joystick : ℕ),
    computer_table = 140 →
    computer_chair = 100 →
    joystick = 20 →
    frank_joystick + eman_joystick = joystick →
    computer_table + frank_joystick = computer_chair + eman_joystick + 30 →
    frank_joystick * 4 = joystick := by
  sorry

end NUMINAMATH_CALUDE_joystick_payment_ratio_l3393_339364


namespace NUMINAMATH_CALUDE_rectangular_box_diagonals_l3393_339348

theorem rectangular_box_diagonals 
  (x y z : ℝ) 
  (surface_area : 2 * (x*y + y*z + z*x) = 106) 
  (edge_sum : 4 * (x + y + z) = 52) :
  4 * Real.sqrt (x^2 + y^2 + z^2) = 12 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonals_l3393_339348


namespace NUMINAMATH_CALUDE_cube_with_cylindrical_hole_l3393_339395

/-- The surface area of a cube with a cylindrical hole --/
def surface_area (cube_edge : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) (π : ℝ) : ℝ :=
  6 * cube_edge^2 - 2 * π * cylinder_radius^2 + 2 * π * cylinder_radius * cylinder_height

/-- The volume of a cube with a cylindrical hole --/
def volume (cube_edge : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) (π : ℝ) : ℝ :=
  cube_edge^3 - π * cylinder_radius^2 * cylinder_height

/-- Theorem stating the surface area and volume of the resulting geometric figure --/
theorem cube_with_cylindrical_hole :
  let cube_edge : ℝ := 10
  let cylinder_radius : ℝ := 2
  let cylinder_height : ℝ := 10
  let π : ℝ := 3
  surface_area cube_edge cylinder_radius cylinder_height π = 696 ∧
  volume cube_edge cylinder_radius cylinder_height π = 880 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_cylindrical_hole_l3393_339395


namespace NUMINAMATH_CALUDE_y_takes_70_days_l3393_339326

-- Define the work completion rates
def mahesh_rate : ℚ := 1 / 35
def rajesh_rate : ℚ := 1 / 30

-- Define the amount of work Mahesh completes
def mahesh_work : ℚ := mahesh_rate * 20

-- Define the amount of work Rajesh completes
def rajesh_work : ℚ := 1 - mahesh_work

-- Define Y's completion time
def y_completion_time : ℚ := 70

-- Theorem statement
theorem y_takes_70_days :
  y_completion_time = 70 := by sorry

end NUMINAMATH_CALUDE_y_takes_70_days_l3393_339326


namespace NUMINAMATH_CALUDE_correct_train_process_l3393_339384

-- Define the actions as an inductive type
inductive TrainAction
  | BuyTicket
  | WaitForTrain
  | CheckTicket
  | BoardTrain

-- Define a type for a sequence of actions
def ActionSequence := List TrainAction

-- Define the correct sequence
def correctSequence : ActionSequence :=
  [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.CheckTicket, TrainAction.BoardTrain]

-- Define a predicate for a valid train-taking process
def isValidProcess (sequence : ActionSequence) : Prop :=
  sequence = correctSequence

-- Theorem statement
theorem correct_train_process :
  isValidProcess correctSequence :=
sorry

end NUMINAMATH_CALUDE_correct_train_process_l3393_339384


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l3393_339333

/-- An arithmetic-geometric sequence -/
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r q : ℝ, ∀ n : ℕ, a (n + 1) = r * a n + q

/-- The statement to be proven -/
theorem arithmetic_geometric_sum (a : ℕ → ℝ) :
  arithmetic_geometric_sequence a →
  a 4 + a 6 = 5 →
  a 4 * a 6 = 6 →
  a 3 * a 5 + a 5 * a 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l3393_339333


namespace NUMINAMATH_CALUDE_inequality_solution_l3393_339385

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (2 - x)) ∧
  (∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → (x₁ - x₂) / (f x₁ - f x₂) > 0)

/-- The solution set of the inequality -/
def solution_set (x : ℝ) : Prop :=
  x ≤ 0 ∨ x ≥ 4/3

/-- Theorem stating the solution of the inequality -/
theorem inequality_solution (f : ℝ → ℝ) (h : special_function f) :
  ∀ x, f (2*x - 1) - f (3 - x) ≥ 0 ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3393_339385


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3393_339386

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_3 = 8 and a_6 = 5, a_9 = 2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a3 : a 3 = 8) 
  (h_a6 : a 6 = 5) : 
  a 9 = 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3393_339386


namespace NUMINAMATH_CALUDE_remaining_fabric_is_294_l3393_339339

/-- Represents the flag-making scenario with given initial conditions -/
structure FlagScenario where
  totalFabric : ℕ
  squareFlagSize : ℕ
  wideFlagWidth : ℕ
  wideFlagHeight : ℕ
  tallFlagWidth : ℕ
  tallFlagHeight : ℕ
  squareFlagsMade : ℕ
  wideFlagsMade : ℕ
  tallFlagsMade : ℕ

/-- Calculates the remaining fabric after making flags -/
def remainingFabric (scenario : FlagScenario) : ℕ :=
  scenario.totalFabric -
  (scenario.squareFlagSize * scenario.squareFlagSize * scenario.squareFlagsMade +
   scenario.wideFlagWidth * scenario.wideFlagHeight * scenario.wideFlagsMade +
   scenario.tallFlagWidth * scenario.tallFlagHeight * scenario.tallFlagsMade)

/-- Theorem stating that the remaining fabric is 294 square feet -/
theorem remaining_fabric_is_294 (scenario : FlagScenario)
  (h1 : scenario.totalFabric = 1000)
  (h2 : scenario.squareFlagSize = 4)
  (h3 : scenario.wideFlagWidth = 5)
  (h4 : scenario.wideFlagHeight = 3)
  (h5 : scenario.tallFlagWidth = 3)
  (h6 : scenario.tallFlagHeight = 5)
  (h7 : scenario.squareFlagsMade = 16)
  (h8 : scenario.wideFlagsMade = 20)
  (h9 : scenario.tallFlagsMade = 10) :
  remainingFabric scenario = 294 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fabric_is_294_l3393_339339


namespace NUMINAMATH_CALUDE_power_relation_l3393_339312

theorem power_relation (x m n : ℝ) (hm : x^m = 3) (hn : x^n = 5) :
  x^(2*m - 3*n) = 9/125 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3393_339312


namespace NUMINAMATH_CALUDE_final_card_count_l3393_339335

def baseball_card_problem (initial_cards : ℕ) (maria_takes : ℕ → ℕ) (peter_takes : ℕ) (paul_multiplies : ℕ → ℕ) : ℕ :=
  let after_maria := initial_cards - maria_takes initial_cards
  let after_peter := after_maria - peter_takes
  paul_multiplies after_peter

theorem final_card_count :
  baseball_card_problem 15 (fun n => (n + 1) / 2) 1 (fun n => 3 * n) = 18 := by
  sorry

end NUMINAMATH_CALUDE_final_card_count_l3393_339335


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3393_339361

theorem algebraic_expression_equality (x y : ℝ) (h : x + 2*y = 2) : 1 - 2*x - 4*y = -3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3393_339361


namespace NUMINAMATH_CALUDE_range_interval_length_l3393_339307

-- Define the geometric sequence and its sum
def a (n : ℕ) : ℚ := 3/2 * (-1/2)^(n-1)
def S (n : ℕ) : ℚ := 1 - (-1/2)^n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := S n + 1 / S n

-- State the theorem
theorem range_interval_length :
  (∀ n : ℕ, n > 0 → -2 * S 2 + 4 * S 4 = 2 * S 3) →
  (∃ L : ℚ, L > 0 ∧ ∀ n : ℕ, n > 0 → ∃ x y : ℚ, x < y ∧ y - x = L ∧ b n ∈ Set.Icc x y) ∧
  (∀ L' : ℚ, L' > 0 → (∀ n : ℕ, n > 0 → ∃ x y : ℚ, x < y ∧ y - x = L' ∧ b n ∈ Set.Icc x y) → L' ≥ 1/6) :=
sorry

end NUMINAMATH_CALUDE_range_interval_length_l3393_339307


namespace NUMINAMATH_CALUDE_betty_daughter_age_difference_l3393_339328

/-- Proves that Betty's daughter is 40% younger than Betty given the specified conditions -/
theorem betty_daughter_age_difference (betty_age : ℕ) (granddaughter_age : ℕ) : 
  betty_age = 60 →
  granddaughter_age = 12 →
  granddaughter_age = (betty_age - (betty_age - granddaughter_age * 3)) / 3 →
  (betty_age - granddaughter_age * 3) / betty_age * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_betty_daughter_age_difference_l3393_339328


namespace NUMINAMATH_CALUDE_concert_revenue_calculation_l3393_339310

def ticket_revenue (student_price : ℕ) (non_student_price : ℕ) (total_tickets : ℕ) (student_tickets : ℕ) : ℕ :=
  let non_student_tickets := total_tickets - student_tickets
  student_price * student_tickets + non_student_price * non_student_tickets

theorem concert_revenue_calculation :
  ticket_revenue 9 11 2000 520 = 20960 := by
  sorry

end NUMINAMATH_CALUDE_concert_revenue_calculation_l3393_339310


namespace NUMINAMATH_CALUDE_binomial_expansion_103_l3393_339352

theorem binomial_expansion_103 : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_103_l3393_339352


namespace NUMINAMATH_CALUDE_sixth_power_sum_l3393_339399

theorem sixth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 23)
  (h4 : a * x^4 + b * y^4 = 50)
  (h5 : a * x^5 + b * y^5 = 106) :
  a * x^6 + b * y^6 = 238 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l3393_339399


namespace NUMINAMATH_CALUDE_orthocenter_symmetry_and_equal_circles_l3393_339347

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A point in a plane -/
def Point := ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (A B C : Point) : Point := sorry

/-- Checks if four points are on the same circle -/
def on_same_circle (A B C D : Point) (S : Circle) : Prop := sorry

/-- Checks if two quadrilaterals are symmetric with respect to a point -/
def symmetric_quadrilaterals (A B C D A' B' C' D' H : Point) : Prop := sorry

/-- Checks if four points are on a circle with the same radius as another circle -/
def on_equal_circle (A B C D : Point) (S : Circle) : Prop := sorry

theorem orthocenter_symmetry_and_equal_circles 
  (A₁ A₂ A₃ A₄ : Point) (S : Circle)
  (h_same_circle : on_same_circle A₁ A₂ A₃ A₄ S)
  (H₁ := orthocenter A₂ A₃ A₄)
  (H₂ := orthocenter A₁ A₃ A₄)
  (H₃ := orthocenter A₁ A₂ A₄)
  (H₄ := orthocenter A₁ A₂ A₃) :
  ∃ (H : Point),
    (symmetric_quadrilaterals A₁ A₂ A₃ A₄ H₁ H₂ H₃ H₄ H) ∧
    (on_equal_circle A₁ A₂ H₃ H₄ S) ∧
    (on_equal_circle A₁ A₃ H₂ H₄ S) ∧
    (on_equal_circle A₁ A₄ H₂ H₃ S) ∧
    (on_equal_circle A₂ A₃ H₁ H₄ S) ∧
    (on_equal_circle A₂ A₄ H₁ H₃ S) ∧
    (on_equal_circle A₃ A₄ H₁ H₂ S) ∧
    (on_equal_circle H₁ H₂ H₃ H₄ S) :=
  sorry

end NUMINAMATH_CALUDE_orthocenter_symmetry_and_equal_circles_l3393_339347


namespace NUMINAMATH_CALUDE_emilys_cards_l3393_339383

theorem emilys_cards (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 63)
  (h2 : final_cards = 70) :
  final_cards - initial_cards = 7 := by
  sorry

end NUMINAMATH_CALUDE_emilys_cards_l3393_339383


namespace NUMINAMATH_CALUDE_ellipse_range_l3393_339305

theorem ellipse_range (m n : ℝ) : 
  (m^2 / 3 + n^2 / 8 = 1) → 
  ∃ x : ℝ, x = Real.sqrt 3 * m ∧ -3 ≤ x ∧ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_range_l3393_339305


namespace NUMINAMATH_CALUDE_complement_of_union_l3393_339391

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {4, 5}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (A ∪ B)ᶜ = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3393_339391


namespace NUMINAMATH_CALUDE_mary_nancy_balloon_ratio_l3393_339325

def nancy_balloons : ℕ := 7
def mary_balloons : ℕ := 28

theorem mary_nancy_balloon_ratio :
  mary_balloons / nancy_balloons = 4 := by sorry

end NUMINAMATH_CALUDE_mary_nancy_balloon_ratio_l3393_339325


namespace NUMINAMATH_CALUDE_power_of_half_equals_one_l3393_339370

theorem power_of_half_equals_one (a b : ℕ) : 
  (2^a : ℕ) ∣ 300 ∧ 
  (∀ k : ℕ, k > a → ¬((2^k : ℕ) ∣ 300)) ∧ 
  (3^b : ℕ) ∣ 300 ∧ 
  (∀ k : ℕ, k > b → ¬((3^k : ℕ) ∣ 300)) → 
  (1/2 : ℚ)^(b - a + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_half_equals_one_l3393_339370


namespace NUMINAMATH_CALUDE_tangents_and_line_of_tangency_l3393_339302

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

-- Define point P
def P : ℝ × ℝ := (-2, 3)

-- Define the tangent lines
def tangent1 (x y : ℝ) : Prop := (Real.sqrt 3 + 6) * x - 4 * y + 2 * Real.sqrt 3 - 3 = 0
def tangent2 (x y : ℝ) : Prop := (3 + Real.sqrt 3) * x + 4 * y - 6 + 2 * Real.sqrt 3 = 0

-- Define the line passing through points of tangency
def tangencyLine (x y : ℝ) : Prop := 3 * x - 2 * y - 3 = 0

theorem tangents_and_line_of_tangency :
  ∃ (M N : ℝ × ℝ),
    M ∈ C ∧ N ∈ C ∧
    (tangent1 M.1 M.2 ∨ tangent2 M.1 M.2) ∧
    (tangent1 N.1 N.2 ∨ tangent2 N.1 N.2) ∧
    tangencyLine M.1 M.2 ∧
    tangencyLine N.1 N.2 :=
by sorry

end NUMINAMATH_CALUDE_tangents_and_line_of_tangency_l3393_339302


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_l3393_339301

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_wrt_origin (2*a + 1) 4 1 (3*b - 1) → 2*a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_l3393_339301


namespace NUMINAMATH_CALUDE_circle_diameter_twice_radius_l3393_339306

/-- A circle with a center, radius, and diameter. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  diameter : ℝ

/-- The diameter of a circle is twice its radius. -/
theorem circle_diameter_twice_radius (c : Circle) : c.diameter = 2 * c.radius := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_twice_radius_l3393_339306


namespace NUMINAMATH_CALUDE_line_through_point_l3393_339373

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → (2 = 2 * 1 + b) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3393_339373


namespace NUMINAMATH_CALUDE_decreasing_implies_positive_a_l3393_339300

/-- The function f(x) = a(x^3 - 3x) is decreasing on the interval (-1, 1) --/
def is_decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

/-- The main theorem: if f(x) = a(x^3 - 3x) is decreasing on (-1, 1), then a > 0 --/
theorem decreasing_implies_positive_a (a : ℝ) :
  is_decreasing_on_interval (fun x => a * (x^3 - 3*x)) → a > 0 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_implies_positive_a_l3393_339300


namespace NUMINAMATH_CALUDE_intersection_points_max_distance_values_l3393_339344

-- Define the line l
def line_l (a t : ℝ) : ℝ × ℝ := (a + 2*t, 1 - t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Part 1: Intersection points
theorem intersection_points :
  ∃ (t₁ t₂ : ℝ),
    let (x₁, y₁) := line_l (-2) t₁
    let (x₂, y₂) := line_l (-2) t₂
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    x₁ = -4*Real.sqrt 5/5 ∧ y₁ = 2*Real.sqrt 5/5 ∧
    x₂ = 4*Real.sqrt 5/5 ∧ y₂ = -2*Real.sqrt 5/5 :=
sorry

-- Part 2: Values of a
theorem max_distance_values :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), curve_C x y →
      (|x + 2*y - 2 - a| / Real.sqrt 5 ≤ 2 * Real.sqrt 5)) ∧
    (∃ (x y : ℝ), curve_C x y ∧
      |x + 2*y - 2 - a| / Real.sqrt 5 = 2 * Real.sqrt 5) →
    (a = 8 - 2*Real.sqrt 5 ∨ a = 2*Real.sqrt 5 - 12) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_max_distance_values_l3393_339344


namespace NUMINAMATH_CALUDE_integral_3x_plus_sinx_l3393_339334

theorem integral_3x_plus_sinx (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x + Real.sin x) :
  ∫ x in (0)..(Real.pi / 2), f x = (3 / 8) * Real.pi^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_3x_plus_sinx_l3393_339334


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l3393_339356

/-- The length of the tangent from the origin to a circle passing through specific points -/
theorem tangent_length_to_circle (A B C : ℝ × ℝ) : 
  A = (4, 5) → B = (8, 10) → C = (7, 17) → 
  ∃ (circle : Set (ℝ × ℝ)) (tangent : ℝ × ℝ → ℝ),
    (A ∈ circle ∧ B ∈ circle ∧ C ∈ circle) ∧
    (tangent (0, 0) = 2 * Real.sqrt 41) := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l3393_339356


namespace NUMINAMATH_CALUDE_circles_intersect_l3393_339369

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- Definition of intersection for two circles -/
def intersect (c : TwoCircles) : Prop :=
  let d := Real.sqrt ((c.center1.1 - c.center2.1)^2 + (c.center1.2 - c.center2.2)^2)
  d < c.radius1 + c.radius2 ∧ d > abs (c.radius1 - c.radius2)

/-- The main theorem: the given circles intersect -/
theorem circles_intersect : 
  let c := TwoCircles.mk (0, 0) 2 (-3, 4) 4
  intersect c := by sorry


end NUMINAMATH_CALUDE_circles_intersect_l3393_339369


namespace NUMINAMATH_CALUDE_sarah_marriage_age_l3393_339323

/-- The age at which a person will get married according to the game -/
def marriage_age (current_age : ℕ) (name_length : ℕ) : ℕ :=
  name_length + 2 * current_age

/-- Sarah's current age -/
def sarah_age : ℕ := 9

/-- The number of letters in Sarah's name -/
def sarah_name_length : ℕ := 5

/-- Theorem stating that Sarah will get married at age 23 according to the game -/
theorem sarah_marriage_age : marriage_age sarah_age sarah_name_length = 23 := by
  sorry

end NUMINAMATH_CALUDE_sarah_marriage_age_l3393_339323


namespace NUMINAMATH_CALUDE_det_A_eq_16_l3393_339342

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2, -4; 6, -1, 3; 2, -3, 5]

theorem det_A_eq_16 : Matrix.det A = 16 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_16_l3393_339342
