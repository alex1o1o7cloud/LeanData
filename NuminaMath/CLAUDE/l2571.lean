import Mathlib

namespace NUMINAMATH_CALUDE_additional_lawn_to_mow_l2571_257144

/-- The problem of calculating additional square feet to mow -/
theorem additional_lawn_to_mow 
  (rate : ℚ) 
  (book_cost : ℚ) 
  (lawns_mowed : ℕ) 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) : 
  rate = 1/10 → 
  book_cost = 150 → 
  lawns_mowed = 3 → 
  lawn_length = 20 → 
  lawn_width = 15 → 
  (book_cost - lawns_mowed * lawn_length * lawn_width * rate) / rate = 600 := by
  sorry

#check additional_lawn_to_mow

end NUMINAMATH_CALUDE_additional_lawn_to_mow_l2571_257144


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2571_257196

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 245 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l2571_257196


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2571_257158

theorem geometric_sequence_formula (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a (k + 1) = 3 * a k) →  -- Geometric sequence with common ratio 3
  a 1 = 4 →                     -- First term is 4
  a n = 4 * 3^(n - 1) :=        -- General formula
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2571_257158


namespace NUMINAMATH_CALUDE_system_solution_l2571_257151

theorem system_solution (x y z : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x^2 + y^2 = -x + 3*y + z ∧
  y^2 + z^2 = x + 3*y - z ∧
  z^2 + x^2 = 2*x + 2*y - z →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2571_257151


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l2571_257126

theorem cow_chicken_problem (cows chickens : ℕ) : 
  4 * cows + 2 * chickens = 14 + 2 * (cows + chickens) → cows = 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l2571_257126


namespace NUMINAMATH_CALUDE_son_age_is_22_l2571_257166

/-- Given a man and his son, where:
    1. The man is 24 years older than his son
    2. In two years, the man's age will be twice the age of his son
    This theorem proves that the present age of the son is 22 years. -/
theorem son_age_is_22 (man_age son_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_son_age_is_22_l2571_257166


namespace NUMINAMATH_CALUDE_xyz_value_l2571_257136

theorem xyz_value (x y z : ℝ) : 
  4 * (Real.sqrt x + Real.sqrt (y - 1) + Real.sqrt (z - 2)) = x + y + z + 9 →
  x * y * z = 120 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2571_257136


namespace NUMINAMATH_CALUDE_inequalities_hold_for_all_reals_l2571_257124

-- Define the two quadratic functions
def f (x : ℝ) := x^2 + 6*x + 10
def g (x : ℝ) := -x^2 + x - 2

-- Theorem stating that both inequalities hold for all real numbers
theorem inequalities_hold_for_all_reals :
  (∀ x : ℝ, f x > 0) ∧ (∀ x : ℝ, g x < 0) :=
sorry

end NUMINAMATH_CALUDE_inequalities_hold_for_all_reals_l2571_257124


namespace NUMINAMATH_CALUDE_odd_function_property_l2571_257147

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem odd_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h1 : OddFunction f) 
  (h2 : a = 2) 
  (h3 : f (-2) = 11) : 
  f a = -11 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2571_257147


namespace NUMINAMATH_CALUDE_divide_by_reciprocal_twelve_divided_by_one_twelfth_l2571_257159

theorem divide_by_reciprocal (x y : ℚ) (h : y ≠ 0) : x / y = x * (1 / y) := by sorry

theorem twelve_divided_by_one_twelfth : 12 / (1 / 12) = 144 := by sorry

end NUMINAMATH_CALUDE_divide_by_reciprocal_twelve_divided_by_one_twelfth_l2571_257159


namespace NUMINAMATH_CALUDE_pumpkins_left_l2571_257105

theorem pumpkins_left (grown : ℕ) (eaten : ℕ) (h1 : grown = 43) (h2 : eaten = 23) :
  grown - eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_pumpkins_left_l2571_257105


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_twenty_two_l2571_257198

/-- The function f(x) = x^3 - 3x^2 + x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + x

/-- Theorem: The value of f(-2) is -22 -/
theorem f_neg_two_eq_neg_twenty_two : f (-2) = -22 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_twenty_two_l2571_257198


namespace NUMINAMATH_CALUDE_negation_of_existence_l2571_257190

theorem negation_of_existence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2571_257190


namespace NUMINAMATH_CALUDE_book_price_proof_l2571_257110

theorem book_price_proof (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 63)
  (h2 : profit_percentage = 5) :
  ∃ original_price : ℝ, 
    original_price * (1 + profit_percentage / 100) = selling_price ∧ 
    original_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_price_proof_l2571_257110


namespace NUMINAMATH_CALUDE_number_of_books_a_l2571_257165

/-- Proves that the number of books (a) is 12, given the conditions -/
theorem number_of_books_a (total : ℕ) (diff : ℕ) : 
  (total = 20) → (diff = 4) → ∃ (a b : ℕ), (a + b = total) ∧ (a = b + diff) ∧ (a = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_books_a_l2571_257165


namespace NUMINAMATH_CALUDE_molecular_weight_BaF2_is_175_l2571_257179

/-- The molecular weight of BaF2 in grams per mole. -/
def molecular_weight_BaF2 : ℝ := 175

/-- The number of moles of BaF2 in the given condition. -/
def moles_BaF2 : ℝ := 6

/-- The total weight of the given moles of BaF2 in grams. -/
def total_weight_BaF2 : ℝ := 1050

/-- Theorem stating that the molecular weight of BaF2 is 175 grams/mole. -/
theorem molecular_weight_BaF2_is_175 :
  molecular_weight_BaF2 = total_weight_BaF2 / moles_BaF2 :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_BaF2_is_175_l2571_257179


namespace NUMINAMATH_CALUDE_square_difference_l2571_257146

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2571_257146


namespace NUMINAMATH_CALUDE_number_count_l2571_257142

theorem number_count (n : ℕ) 
  (h1 : (n : ℝ) * 30 = (4 : ℝ) * 25 + (3 : ℝ) * 35 - 25)
  (h2 : n > 4) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_count_l2571_257142


namespace NUMINAMATH_CALUDE_triangle_with_unequal_angle_l2571_257186

theorem triangle_with_unequal_angle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal
  c = a - 10 →       -- Third angle is 10° less than the others
  c = 53.33 :=       -- Measure of the smallest angle
by sorry

end NUMINAMATH_CALUDE_triangle_with_unequal_angle_l2571_257186


namespace NUMINAMATH_CALUDE_power_function_m_equals_four_l2571_257178

/-- A function f is a power function if it has the form f(x) = ax^b where a and b are constants and a ≠ 0 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ b

/-- Given f(x) = (m^2 - 3m - 3)x^(√m) is a power function, prove that m = 4 -/
theorem power_function_m_equals_four (m : ℝ) 
  (h : is_power_function (fun x ↦ (m^2 - 3*m - 3) * x^(Real.sqrt m))) : 
  m = 4 := by
  sorry


end NUMINAMATH_CALUDE_power_function_m_equals_four_l2571_257178


namespace NUMINAMATH_CALUDE_coprime_2013_in_32nd_group_l2571_257193

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def group_size (n : ℕ) : ℕ := 2 * n - 1

def cumulative_group_size (n : ℕ) : ℕ := n^2

def coprime_count (n : ℕ) : ℕ := n - (n.div 2 + n.div 503 - n.div 1006)

theorem coprime_2013_in_32nd_group :
  ∃ k : ℕ, k = 32 ∧
    coprime_count 2012 < cumulative_group_size (k - 1) ∧
    coprime_count 2012 + 1 ≤ cumulative_group_size k ∧
    is_coprime 2013 2012 := by
  sorry

end NUMINAMATH_CALUDE_coprime_2013_in_32nd_group_l2571_257193


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l2571_257191

/-- A parallelogram with given side lengths -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ → ℝ
  side4 : ℝ → ℝ

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram where
  side1 := 12
  side2 := 15
  side3 := fun y => 10 * y - 3
  side4 := fun x => 3 * x + 6

/-- The theorem stating the solution to the problem -/
theorem parallelogram_side_sum (p : Parallelogram) 
  (h1 : p.side1 = p.side4 1)
  (h2 : p.side2 = p.side3 2)
  (h3 : p = problem_parallelogram) :
  ∃ (x y : ℝ), x + y = 3.8 ∧ p.side3 y = 15 ∧ p.side4 x = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l2571_257191


namespace NUMINAMATH_CALUDE_total_boys_count_l2571_257174

theorem total_boys_count (average_all : ℝ) (average_passed : ℝ) (average_failed : ℝ) (passed_count : ℕ) :
  average_all = 37 →
  average_passed = 39 →
  average_failed = 15 →
  passed_count = 110 →
  ∃ (total_count : ℕ), 
    total_count = passed_count + (total_count - passed_count) ∧
    (average_all * total_count : ℝ) = average_passed * passed_count + average_failed * (total_count - passed_count) ∧
    total_count = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_total_boys_count_l2571_257174


namespace NUMINAMATH_CALUDE_decimal_132_to_binary_l2571_257177

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

-- Theorem statement
theorem decimal_132_to_binary :
  decimalToBinary 132 = [true, false, false, false, false, true, false, false] := by
  sorry

#eval decimalToBinary 132

end NUMINAMATH_CALUDE_decimal_132_to_binary_l2571_257177


namespace NUMINAMATH_CALUDE_point_on_x_axis_point_in_second_quadrant_l2571_257122

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Part 1
theorem point_on_x_axis (a : ℝ) :
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Part 2
theorem point_in_second_quadrant (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| →
  a^2023 + 2023 = 2022 :=
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_point_in_second_quadrant_l2571_257122


namespace NUMINAMATH_CALUDE_product_expansion_l2571_257132

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 1) = x^4 - 5*x^2 + 6*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2571_257132


namespace NUMINAMATH_CALUDE_person_age_in_1900_l2571_257154

theorem person_age_in_1900 (birth_year : ℕ) (death_year : ℕ) (age_at_death : ℕ) :
  (age_at_death = birth_year / 29) →
  (birth_year < 1900) →
  (1901 ≤ death_year) →
  (death_year ≤ 1930) →
  (death_year = birth_year + age_at_death) →
  (1900 - birth_year = 44) :=
by sorry

end NUMINAMATH_CALUDE_person_age_in_1900_l2571_257154


namespace NUMINAMATH_CALUDE_village_population_problem_l2571_257197

theorem village_population_problem (original : ℕ) : 
  (original : ℝ) * 0.9 * 0.75 = 5130 → original = 7600 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l2571_257197


namespace NUMINAMATH_CALUDE_relationship_abc_l2571_257155

theorem relationship_abc (a b c : ℕ) : 
  a = 2^555 → b = 3^444 → c = 6^222 → a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2571_257155


namespace NUMINAMATH_CALUDE_car_dealer_sales_l2571_257123

theorem car_dealer_sales (x : ℕ) (a b : ℤ) : 
  x > 0 ∧ 
  (7 : ℚ) = (x : ℚ)⁻¹ * (7 * x : ℚ) ∧ 
  (8 : ℚ) = ((x - 1) : ℚ)⁻¹ * ((7 * x - a) : ℚ) ∧ 
  (5 : ℚ) = ((x - 1) : ℚ)⁻¹ * ((7 * x - b) : ℚ) ∧ 
  (23 : ℚ) / 4 = ((x - 2) : ℚ)⁻¹ * ((7 * x - a - b) : ℚ) →
  7 * x = 42 := by
  sorry

end NUMINAMATH_CALUDE_car_dealer_sales_l2571_257123


namespace NUMINAMATH_CALUDE_sum_in_base4_l2571_257141

-- Define a function to convert from base 10 to base 4
def toBase4 (n : ℕ) : List ℕ :=
  sorry

-- Define a function to convert from base 4 to base 10
def fromBase4 (l : List ℕ) : ℕ :=
  sorry

theorem sum_in_base4 : 
  toBase4 (195 + 61) = [1, 0, 0, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base4_l2571_257141


namespace NUMINAMATH_CALUDE_nonshaded_perimeter_is_64_l2571_257195

/-- A structure representing the geometric configuration described in the problem -/
structure GeometricConfig where
  outer_length : ℝ
  outer_width : ℝ
  inner_length : ℝ
  inner_width : ℝ
  extension : ℝ
  shaded_area : ℝ

/-- The perimeter of the non-shaded region given the geometric configuration -/
def nonshaded_perimeter (config : GeometricConfig) : ℝ :=
  2 * (config.outer_width + (config.outer_length + config.extension - config.inner_length))

/-- Theorem stating that given the specific geometric configuration, 
    the perimeter of the non-shaded region is 64 inches -/
theorem nonshaded_perimeter_is_64 (config : GeometricConfig) 
  (h1 : config.outer_length = 12)
  (h2 : config.outer_width = 10)
  (h3 : config.inner_length = 3)
  (h4 : config.inner_width = 4)
  (h5 : config.extension = 3)
  (h6 : config.shaded_area = 120) :
  nonshaded_perimeter config = 64 := by
  sorry

end NUMINAMATH_CALUDE_nonshaded_perimeter_is_64_l2571_257195


namespace NUMINAMATH_CALUDE_problem_solution_l2571_257131

-- Define the set of integers
def Z : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n}

-- Define the set of x satisfying the conditions
def S : Set ℝ := {x : ℝ | (|x - 1| < 2 ∨ x ∉ Z) ∧ x ∈ Z}

-- Define the target set
def T : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem problem_solution : S = T := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2571_257131


namespace NUMINAMATH_CALUDE_point_on_x_axis_point_neg_two_zero_on_x_axis_l2571_257139

/-- A point lies on the x-axis if and only if its y-coordinate is 0 -/
theorem point_on_x_axis (x y : ℝ) : 
  (x, y) ∈ {p : ℝ × ℝ | p.2 = 0} ↔ y = 0 := by sorry

/-- The point (-2, 0) lies on the x-axis -/
theorem point_neg_two_zero_on_x_axis : 
  (-2, 0) ∈ {p : ℝ × ℝ | p.2 = 0} := by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_point_neg_two_zero_on_x_axis_l2571_257139


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l2571_257130

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ y => 6 * y^2 - 29 * y + 24
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → y ≤ z ∧ y = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l2571_257130


namespace NUMINAMATH_CALUDE_all_statements_false_l2571_257101

-- Define the concepts of lines and planes
variable (Line Plane : Type)

-- Define the concept of parallelism between lines
variable (parallel_lines : Line → Line → Prop)

-- Define the concept of parallelism between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the concept of perpendicularity between lines
variable (perpendicular : Line → Line → Prop)

-- Define the concept of a line having no common points with another line
variable (no_common_points : Line → Line → Prop)

-- Define the concept of a line having no common points with countless lines in a plane
variable (no_common_points_with_plane_lines : Line → Plane → Prop)

theorem all_statements_false :
  (∀ (l₁ l₂ : Line) (p : Plane), parallel_line_plane l₁ p → parallel_line_plane l₂ p → parallel_lines l₁ l₂) = False ∧
  (∀ (l₁ l₂ : Line), no_common_points l₁ l₂ → parallel_lines l₁ l₂) = False ∧
  (∀ (l₁ l₂ l₃ : Line), perpendicular l₁ l₃ → perpendicular l₂ l₃ → parallel_lines l₁ l₂) = False ∧
  (∀ (l : Line) (p : Plane), no_common_points_with_plane_lines l p → parallel_line_plane l p) = False :=
sorry

end NUMINAMATH_CALUDE_all_statements_false_l2571_257101


namespace NUMINAMATH_CALUDE_rational_equation_proof_l2571_257143

theorem rational_equation_proof (m n : ℚ) 
  (h1 : 3 * m + 2 * n = 0) 
  (h2 : m * n ≠ 0) : 
  m / n - n / m = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_proof_l2571_257143


namespace NUMINAMATH_CALUDE_frame_width_l2571_257182

/-- Given a frame with three square photo openings, this theorem proves that
    the width of the frame is 5 cm under the specified conditions. -/
theorem frame_width (s : ℝ) (d : ℝ) : 
  s > 0 →  -- side length of square opening is positive
  d > 0 →  -- frame width is positive
  4 * s = 60 →  -- perimeter of one photo opening
  2 * ((3 * s + 4 * d) + (s + 2 * d)) = 180 →  -- total perimeter of the frame
  d = 5 := by
  sorry

end NUMINAMATH_CALUDE_frame_width_l2571_257182


namespace NUMINAMATH_CALUDE_max_value_quadratic_sum_l2571_257187

theorem max_value_quadratic_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - x*y + y^2 = 9) : 
  x^2 + x*y + y^2 ≤ 27 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 - a*b + b^2 = 9 ∧ a^2 + a*b + b^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_sum_l2571_257187


namespace NUMINAMATH_CALUDE_min_victory_points_l2571_257102

/-- Represents the point system for a football competition --/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team's performance --/
structure TeamPerformance where
  total_matches : ℕ
  played_matches : ℕ
  current_points : ℕ
  target_points : ℕ
  min_victories_needed : ℕ

/-- The theorem to prove --/
theorem min_victory_points (ps : PointSystem) (tp : TeamPerformance) : 
  ps.draw_points = 1 ∧ 
  ps.defeat_points = 0 ∧
  tp.total_matches = 20 ∧ 
  tp.played_matches = 5 ∧
  tp.current_points = 14 ∧
  tp.target_points = 40 ∧
  tp.min_victories_needed = 6 →
  ps.victory_points ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_victory_points_l2571_257102


namespace NUMINAMATH_CALUDE_glen_animals_theorem_l2571_257171

theorem glen_animals_theorem (f t c r : ℕ) : 
  f = (5 * t) / 2 → 
  c = 3 * f → 
  r = 4 * c → 
  ∀ t : ℕ, f + t + c + r ≠ 108 :=
by
  sorry

end NUMINAMATH_CALUDE_glen_animals_theorem_l2571_257171


namespace NUMINAMATH_CALUDE_certain_number_solution_l2571_257148

theorem certain_number_solution : 
  ∃ x : ℝ, (0.02^2 + 0.52^2 + x^2) = 100 * (0.002^2 + 0.052^2 + 0.0035^2) ∧ x = 0.035 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l2571_257148


namespace NUMINAMATH_CALUDE_bicycle_price_calculation_l2571_257145

theorem bicycle_price_calculation (initial_cost : ℝ) (profit1 : ℝ) (profit2 : ℝ) : 
  initial_cost = 150 ∧ profit1 = 0.20 ∧ profit2 = 0.25 →
  (initial_cost * (1 + profit1)) * (1 + profit2) = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_calculation_l2571_257145


namespace NUMINAMATH_CALUDE_class_average_problem_l2571_257156

/-- Given a class of 50 students with an overall average of 92 and the first 30 students
    having an average of 90, the average of the remaining 20 students is 95. -/
theorem class_average_problem :
  ∀ (total_score first_group_score last_group_score : ℝ),
  (50 : ℝ) * 92 = total_score →
  (30 : ℝ) * 90 = first_group_score →
  total_score = first_group_score + last_group_score →
  last_group_score / (20 : ℝ) = 95 := by
sorry

end NUMINAMATH_CALUDE_class_average_problem_l2571_257156


namespace NUMINAMATH_CALUDE_ratio_BL_LC_l2571_257162

/-- A square with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (5, 0) ∧ C = (5, 5) ∧ D = (0, 5))

/-- A point K on side AB of the square -/
def K : ℝ × ℝ := (3, 0)

/-- A point L on side BC of the square -/
def L (y : ℝ) : ℝ × ℝ := (5, y)

/-- Distance function between a point and a line -/
def distance_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := sorry

/-- The theorem to be proved -/
theorem ratio_BL_LC (ABCD : Square) :
  ∃ y : ℝ, 0 ≤ y ∧ y ≤ 5 ∧
  distance_to_line K (fun p => p.2 = (y - 5) / 5 * p.1 + 5) = 3 →
  (5 - y) / y = 8 / 7 := by sorry

end NUMINAMATH_CALUDE_ratio_BL_LC_l2571_257162


namespace NUMINAMATH_CALUDE_number_problem_l2571_257184

theorem number_problem (x : ℝ) : 0.4 * x - 30 = 50 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2571_257184


namespace NUMINAMATH_CALUDE_danny_travel_time_l2571_257140

/-- The time it takes Danny to reach Steve's house -/
def danny_time : ℝ := 31

/-- The time it takes Steve to reach Danny's house -/
def steve_time (t : ℝ) : ℝ := 2 * t

/-- The time difference between Steve and Danny reaching the halfway point -/
def halfway_time_difference : ℝ := 15.5

theorem danny_travel_time :
  ∀ t : ℝ,
  (steve_time t / 2 - t / 2 = halfway_time_difference) →
  t = danny_time :=
by
  sorry


end NUMINAMATH_CALUDE_danny_travel_time_l2571_257140


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2571_257106

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 4 * x > 25) → x ≥ -5 ∧ (7 - 4 * (-5) > 25) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2571_257106


namespace NUMINAMATH_CALUDE_simplify_expression_l2571_257120

theorem simplify_expression (a : ℝ) : (a^2)^3 + 3*a^4*a^2 - a^8/a^2 = 3*a^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2571_257120


namespace NUMINAMATH_CALUDE_davids_age_l2571_257111

/-- Given the age relationships between Anna, Ben, Carla, and David, prove David's age. -/
theorem davids_age 
  (anna ben carla david : ℕ)  -- Define variables for ages
  (h1 : anna = ben - 5)       -- Anna is five years younger than Ben
  (h2 : ben = carla + 2)      -- Ben is two years older than Carla
  (h3 : david = carla + 4)    -- David is four years older than Carla
  (h4 : anna = 12)            -- Anna is 12 years old
  : david = 19 :=             -- Prove David is 19 years old
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_davids_age_l2571_257111


namespace NUMINAMATH_CALUDE_arithmetic_to_harmonic_progression_l2571_257149

/-- Three non-zero real numbers form an arithmetic progression if and only if
    the difference between the second and first is equal to the difference between the third and second. -/
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- Three non-zero real numbers form a harmonic progression if and only if
    the reciprocal of the middle term is the arithmetic mean of the reciprocals of the other two terms. -/
def is_harmonic_progression (a b c : ℝ) : Prop :=
  2 / b = 1 / a + 1 / c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- If three non-zero real numbers form an arithmetic progression,
    then their reciprocals form a harmonic progression. -/
theorem arithmetic_to_harmonic_progression (a b c : ℝ) :
  is_arithmetic_progression a b c → is_harmonic_progression (1/a) (1/b) (1/c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_to_harmonic_progression_l2571_257149


namespace NUMINAMATH_CALUDE_fruit_difference_l2571_257112

theorem fruit_difference (apples : ℕ) (peach_multiplier : ℕ) : 
  apples = 60 → peach_multiplier = 3 → 
  (peach_multiplier * apples) - apples = 120 := by
  sorry

end NUMINAMATH_CALUDE_fruit_difference_l2571_257112


namespace NUMINAMATH_CALUDE_intersection_points_determine_a_l2571_257180

def curve_C₁ (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = a^2 ∧ 0 ≤ x ∧ x ≤ a

def curve_C₂ (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

theorem intersection_points_determine_a :
  ∀ a : ℝ, a > 0 →
  ∃ A B : ℝ × ℝ,
    curve_C₁ a A.1 A.2 ∧
    curve_C₁ a B.1 B.2 ∧
    curve_C₂ A.1 A.2 ∧
    curve_C₂ B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * Real.sqrt 2 / 3)^2 →
    a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_determine_a_l2571_257180


namespace NUMINAMATH_CALUDE_f_ratio_range_l2571_257133

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- State the theorem
theorem f_ratio_range :
  (∀ x : ℝ, f' x - f x = 2 * x * Real.exp x) →
  f 0 = 1 →
  ∀ x : ℝ, x > 0 → 1 < (f' x) / (f x) ∧ (f' x) / (f x) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_ratio_range_l2571_257133


namespace NUMINAMATH_CALUDE_flag_distribution_l2571_257183

theorem flag_distribution (F : ℕ) (blue_flags red_flags : ℕ) :
  F % 2 = 0 →
  F = blue_flags + red_flags →
  blue_flags ≥ (3 * F) / 10 →
  red_flags ≥ F / 4 →
  (F / 2 - (3 * F) / 10 - F / 4) / (F / 2) = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_l2571_257183


namespace NUMINAMATH_CALUDE_cars_in_parking_lot_l2571_257176

theorem cars_in_parking_lot (total_wheels : ℕ) (wheels_per_car : ℕ) (h1 : total_wheels = 48) (h2 : wheels_per_car = 4) :
  total_wheels / wheels_per_car = 12 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_parking_lot_l2571_257176


namespace NUMINAMATH_CALUDE_uf_games_before_championship_l2571_257192

/-- The number of games UF played before the championship game -/
def n : ℕ := sorry

/-- The total points UF scored in previous games -/
def total_points : ℕ := 720

/-- UF's score in the championship game -/
def championship_score : ℕ := total_points / (2 * n) - 2

/-- UF's opponent's score in the championship game -/
def opponent_score : ℕ := 11

theorem uf_games_before_championship : 
  (total_points / n = championship_score + 2) ∧ 
  (championship_score = opponent_score + 2) ∧
  (n = 24) := by sorry

end NUMINAMATH_CALUDE_uf_games_before_championship_l2571_257192


namespace NUMINAMATH_CALUDE_paint_remaining_after_three_days_paint_problem_solution_l2571_257199

/-- Represents the amount of paint remaining after a certain number of days -/
def paint_remaining (initial_amount : ℚ) (days : ℕ) : ℚ :=
  initial_amount * (1 / 2) ^ days

/-- Theorem stating that after 3 days of using half the remaining paint each day, 
    1/4 of the original amount remains -/
theorem paint_remaining_after_three_days (initial_amount : ℚ) :
  paint_remaining initial_amount 3 = initial_amount / 4 := by
  sorry

/-- Theorem proving that starting with 2 gallons and using half the remaining paint 
    for three consecutive days leaves 1/4 of the original amount -/
theorem paint_problem_solution :
  paint_remaining 2 3 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_after_three_days_paint_problem_solution_l2571_257199


namespace NUMINAMATH_CALUDE_no_three_intersections_l2571_257163

-- Define a circle in Euclidean space
structure EuclideanCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an intersection point
def IntersectionPoint (c1 c2 : EuclideanCircle) := 
  {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
               (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2}

-- Theorem statement
theorem no_three_intersections 
  (c1 c2 : EuclideanCircle) 
  (h_distinct : c1 ≠ c2) : 
  ¬∃ (p1 p2 p3 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 ∈ IntersectionPoint c1 c2 ∧
    p2 ∈ IntersectionPoint c1 c2 ∧
    p3 ∈ IntersectionPoint c1 c2 :=
sorry

end NUMINAMATH_CALUDE_no_three_intersections_l2571_257163


namespace NUMINAMATH_CALUDE_sushi_eating_orders_l2571_257160

/-- Represents a 2 × 3 grid of sushi pieces -/
structure SushiGrid :=
  (pieces : Fin 6 → Bool)

/-- Checks if a piece is adjacent to at most two other pieces -/
def isEatable (grid : SushiGrid) (pos : Fin 6) : Bool :=
  sorry

/-- Generates all valid eating orders for a given SushiGrid -/
def validEatingOrders (grid : SushiGrid) : List (List (Fin 6)) :=
  sorry

/-- The number of valid eating orders for a 2 × 3 sushi grid -/
def numValidOrders : Nat :=
  sorry

theorem sushi_eating_orders :
  numValidOrders = 360 :=
sorry

end NUMINAMATH_CALUDE_sushi_eating_orders_l2571_257160


namespace NUMINAMATH_CALUDE_managers_salary_proof_l2571_257129

def prove_managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : Prop :=
  let total_salary := num_employees * avg_salary
  let new_avg := avg_salary + avg_increase
  let new_total := (num_employees + 1) * new_avg
  new_total - total_salary = 3800

theorem managers_salary_proof :
  prove_managers_salary 20 1700 100 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_proof_l2571_257129


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_45_l2571_257109

/-- The equation of a line passing through (-4, 3) with a slope angle of 45° -/
theorem line_equation_through_point_with_angle_45 :
  ∃ (f : ℝ → ℝ),
    (∀ x y, f x = y ↔ x - y + 7 = 0) ∧
    f (-4) = 3 ∧
    (∀ x, (f x - f (-4)) / (x - (-4)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_45_l2571_257109


namespace NUMINAMATH_CALUDE_special_function_properties_l2571_257185

def I : Set ℝ := Set.Icc (-1) 1

structure SpecialFunction (f : ℝ → ℝ) : Prop where
  domain : ∀ x, x ∈ I → f x ≠ 0 → True
  additive : ∀ x y, x ∈ I → y ∈ I → f (x + y) = f x + f y
  positive : ∀ x, x > 0 → x ∈ I → f x > 0

theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  (∀ x, x ∈ I → f (-x) = -f x) ∧
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l2571_257185


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l2571_257116

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x ^ 2 ≡ 0 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l2571_257116


namespace NUMINAMATH_CALUDE_max_arrangement_is_eight_l2571_257135

/-- Represents a valid arrangement of balls -/
def ValidArrangement (arrangement : List Nat) : Prop :=
  (∀ n ∈ arrangement, 1 ≤ n ∧ n ≤ 9) ∧
  (5 ∈ arrangement → (arrangement.indexOf 5).pred = arrangement.indexOf 1 ∨ 
                     (arrangement.indexOf 5).succ = arrangement.indexOf 1) ∧
  (7 ∈ arrangement → (arrangement.indexOf 7).pred = arrangement.indexOf 1 ∨ 
                     (arrangement.indexOf 7).succ = arrangement.indexOf 1)

/-- The maximum number of balls that can be arranged -/
def MaxArrangement : Nat := 8

/-- Theorem stating that the maximum number of balls that can be arranged is 8 -/
theorem max_arrangement_is_eight :
  (∃ arrangement : List Nat, arrangement.length = MaxArrangement ∧ ValidArrangement arrangement) ∧
  (∀ arrangement : List Nat, arrangement.length > MaxArrangement → ¬ValidArrangement arrangement) := by
  sorry

end NUMINAMATH_CALUDE_max_arrangement_is_eight_l2571_257135


namespace NUMINAMATH_CALUDE_equation_solutions_l2571_257103

theorem equation_solutions :
  ∀ x : ℝ, 
    Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6 ↔ 
    x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2571_257103


namespace NUMINAMATH_CALUDE_edward_garage_sale_games_l2571_257157

/-- The number of games Edward bought at the garage sale -/
def garage_sale_games : ℕ := 14

/-- The number of games Edward bought from a friend -/
def friend_games : ℕ := 41

/-- The number of games that didn't work -/
def bad_games : ℕ := 31

/-- The number of good games Edward ended up with -/
def good_games : ℕ := 24

theorem edward_garage_sale_games :
  garage_sale_games = (good_games + bad_games) - friend_games :=
by sorry

end NUMINAMATH_CALUDE_edward_garage_sale_games_l2571_257157


namespace NUMINAMATH_CALUDE_polynomial_roots_nature_l2571_257119

def P (x : ℝ) : ℝ := x^6 - 5*x^5 + 3*x^2 - 8*x + 16

theorem polynomial_roots_nature :
  (∀ x < 0, P x > 0) ∧ 
  (∃ a b, 0 < a ∧ a < b ∧ P a * P b < 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_nature_l2571_257119


namespace NUMINAMATH_CALUDE_min_lcm_x_z_l2571_257169

theorem min_lcm_x_z (x y z : ℕ) (h1 : Nat.lcm x y = 18) (h2 : Nat.lcm y z = 20) :
  ∃ (x' z' : ℕ), Nat.lcm x' z' = 90 ∧ ∀ (x'' z'' : ℕ), 
    Nat.lcm x'' y = 18 → Nat.lcm y z'' = 20 → Nat.lcm x'' z'' ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_min_lcm_x_z_l2571_257169


namespace NUMINAMATH_CALUDE_equivalent_expression_l2571_257167

theorem equivalent_expression (x : ℝ) (h : x < 0) : 
  Real.sqrt (x / (1 - (x - 2) / x)) = -x / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_l2571_257167


namespace NUMINAMATH_CALUDE_min_garden_cost_is_108_l2571_257117

/-- Represents the cost of each flower type in dollars -/
structure FlowerCost where
  asters : ℝ
  begonias : ℝ
  cannas : ℝ
  dahlias : ℝ
  easterLilies : ℝ

/-- Represents the dimensions of each region in the flower bed -/
structure RegionDimensions where
  region1 : ℝ × ℝ
  region2 : ℝ × ℝ
  region3 : ℝ × ℝ
  region4 : ℝ × ℝ
  region5 : ℝ × ℝ

/-- Calculates the minimum cost of the garden given the flower costs and region dimensions -/
def minGardenCost (costs : FlowerCost) (dimensions : RegionDimensions) : ℝ :=
  sorry

/-- Theorem stating that the minimum cost of the garden is $108 -/
theorem min_garden_cost_is_108 (costs : FlowerCost) (dimensions : RegionDimensions) :
  costs.asters = 1 ∧ 
  costs.begonias = 1.5 ∧ 
  costs.cannas = 2 ∧ 
  costs.dahlias = 2.5 ∧ 
  costs.easterLilies = 3 ∧
  dimensions.region1 = (3, 4) ∧
  dimensions.region2 = (2, 3) ∧
  dimensions.region3 = (3, 5) ∧
  dimensions.region4 = (4, 5) ∧
  dimensions.region5 = (3, 7) →
  minGardenCost costs dimensions = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_min_garden_cost_is_108_l2571_257117


namespace NUMINAMATH_CALUDE_vector_addition_result_l2571_257150

theorem vector_addition_result :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-3, 4)
  2 • a + b = (1, 2) := by
sorry

end NUMINAMATH_CALUDE_vector_addition_result_l2571_257150


namespace NUMINAMATH_CALUDE_g_of_three_equals_five_l2571_257137

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g in terms of f
def g (x : ℝ) : ℝ := f (x - 2)

-- Theorem to prove
theorem g_of_three_equals_five : g 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_equals_five_l2571_257137


namespace NUMINAMATH_CALUDE_pencils_theorem_l2571_257114

def pencils_problem (monday tuesday wednesday thursday friday : ℕ) : Prop :=
  let total_tuesday := monday + tuesday
  let total_wednesday := total_tuesday + 3 * tuesday - 20
  let total_thursday := total_wednesday + wednesday / 2
  let total_friday := total_thursday + 2 * monday
  let final_total := total_friday - 50
  (monday = 35) ∧
  (tuesday = 42) ∧
  (wednesday = 3 * tuesday) ∧
  (thursday = wednesday / 2) ∧
  (friday = 2 * monday) ∧
  (final_total = 266)

theorem pencils_theorem :
  ∃ (monday tuesday wednesday thursday friday : ℕ),
    pencils_problem monday tuesday wednesday thursday friday :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_theorem_l2571_257114


namespace NUMINAMATH_CALUDE_sam_distance_l2571_257107

/-- Given that Marguerite drove 150 miles in 3 hours, and Sam drove for 4 hours at the same average rate as Marguerite, prove that Sam drove 200 miles. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 200 :=
by sorry

end NUMINAMATH_CALUDE_sam_distance_l2571_257107


namespace NUMINAMATH_CALUDE_tenth_term_is_512_l2571_257189

/-- A sequence where each term is twice the previous term, starting with 1 -/
def doubling_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * doubling_sequence n

/-- The 10th term of the doubling sequence is 512 -/
theorem tenth_term_is_512 : doubling_sequence 9 = 512 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_512_l2571_257189


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l2571_257170

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({4, 5, 7, 8, 10} : Set ℤ) →
  (3 * b^3 - b^2 + b - 1) % 5 ≠ 0 ↔ b = 4 ∨ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l2571_257170


namespace NUMINAMATH_CALUDE_square_area_proof_l2571_257125

/-- Given a square with side length equal to both 5x - 21 and 29 - 2x,
    prove that its area is 10609/49 square meters. -/
theorem square_area_proof (x : ℝ) (h : 5 * x - 21 = 29 - 2 * x) :
  (5 * x - 21) ^ 2 = 10609 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l2571_257125


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2571_257194

/-- The standard equation of a hyperbola with given foci and asymptotes -/
theorem hyperbola_standard_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) →
  (a^2 + b^2 = 10) →
  (b / a = 1 / 2) →
  (a^2 = 8 ∧ b^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2571_257194


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2571_257181

theorem fractional_equation_solution :
  ∃ x : ℚ, (3 / 2 : ℚ) - (2 * x) / (3 * x - 1) = 7 / (6 * x - 2) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2571_257181


namespace NUMINAMATH_CALUDE_intersection_equality_condition_l2571_257108

def A : Set ℝ := {x | 1 < x ∧ x ≤ 2}

def B (a : ℝ) : Set ℝ := {x | (2 : ℝ)^(2*a*x) < (2 : ℝ)^(a+x)}

theorem intersection_equality_condition (a : ℝ) :
  A ∩ B a = A ↔ a < 2/3 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_condition_l2571_257108


namespace NUMINAMATH_CALUDE_candy_distribution_l2571_257118

theorem candy_distribution (total_candy : ℕ) (num_people : ℕ) (bags_per_person : ℕ) : 
  total_candy = 648 → num_people = 4 → bags_per_person = 8 →
  (total_candy / num_people / bags_per_person : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2571_257118


namespace NUMINAMATH_CALUDE_expression_value_l2571_257168

theorem expression_value (x : ℝ) (h : x = -2) : (3 * x - 4)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2571_257168


namespace NUMINAMATH_CALUDE_shopkeeper_oranges_l2571_257188

/-- The number of oranges bought by a shopkeeper -/
def oranges : ℕ := sorry

/-- The number of bananas bought by the shopkeeper -/
def bananas : ℕ := 400

/-- The percentage of oranges that are not rotten -/
def good_orange_percentage : ℚ := 85 / 100

/-- The percentage of bananas that are not rotten -/
def good_banana_percentage : ℚ := 92 / 100

/-- The percentage of all fruits that are in good condition -/
def total_good_percentage : ℚ := 878 / 1000

theorem shopkeeper_oranges :
  (↑oranges * good_orange_percentage + ↑bananas * good_banana_percentage) / (↑oranges + ↑bananas) = total_good_percentage ∧
  oranges = 600 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_oranges_l2571_257188


namespace NUMINAMATH_CALUDE_b_remaining_work_days_l2571_257138

-- Define the work rates and time periods
def a_rate : ℚ := 1 / 4
def b_rate : ℚ := 1 / 14
def initial_work_days : ℕ := 2

-- Theorem statement
theorem b_remaining_work_days :
  let total_work : ℚ := 1
  let combined_rate : ℚ := a_rate + b_rate
  let work_done_together : ℚ := combined_rate * initial_work_days
  let remaining_work : ℚ := total_work - work_done_together
  (remaining_work / b_rate : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_b_remaining_work_days_l2571_257138


namespace NUMINAMATH_CALUDE_shopkeeper_red_cards_l2571_257134

/-- Represents the number of decks for each type of playing cards --/
structure DeckCounts where
  standard : Nat
  special : Nat
  custom : Nat

/-- Represents the number of red cards in each type of deck --/
structure RedCardCounts where
  standard : Nat
  special : Nat
  custom : Nat

/-- Calculates the total number of red cards given the deck counts and red card counts --/
def totalRedCards (decks : DeckCounts) (redCards : RedCardCounts) : Nat :=
  decks.standard * redCards.standard +
  decks.special * redCards.special +
  decks.custom * redCards.custom

/-- Theorem stating that the shopkeeper has 178 red cards in total --/
theorem shopkeeper_red_cards :
  let decks : DeckCounts := { standard := 3, special := 2, custom := 2 }
  let redCards : RedCardCounts := { standard := 26, special := 30, custom := 20 }
  totalRedCards decks redCards = 178 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_red_cards_l2571_257134


namespace NUMINAMATH_CALUDE_simplify_expression_l2571_257173

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2571_257173


namespace NUMINAMATH_CALUDE_jays_family_female_guests_l2571_257121

theorem jays_family_female_guests 
  (total_guests : ℕ) 
  (female_percentage : ℚ) 
  (jays_family_percentage : ℚ) 
  (h1 : total_guests = 240)
  (h2 : female_percentage = 60 / 100)
  (h3 : jays_family_percentage = 50 / 100) :
  ↑total_guests * female_percentage * jays_family_percentage = 72 := by
  sorry

end NUMINAMATH_CALUDE_jays_family_female_guests_l2571_257121


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l2571_257153

theorem power_equality_implies_exponent (m : ℝ) : (81 : ℝ) ^ (1/4 : ℝ) = 3^m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l2571_257153


namespace NUMINAMATH_CALUDE_multiplication_simplification_l2571_257152

theorem multiplication_simplification :
  2000 * 2992 * 0.2992 * 20 = 4 * 2992^2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_simplification_l2571_257152


namespace NUMINAMATH_CALUDE_andrea_rhinestone_ratio_l2571_257164

/-- Proves that the ratio of rhinestones Andrea bought to the total rhinestones needed is 1:3 -/
theorem andrea_rhinestone_ratio :
  let total_needed : ℕ := 45
  let found_in_supplies : ℕ := total_needed / 5
  let still_needed : ℕ := 21
  let bought : ℕ := total_needed - found_in_supplies - still_needed
  (bought : ℚ) / total_needed = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_andrea_rhinestone_ratio_l2571_257164


namespace NUMINAMATH_CALUDE_cube_split_73_l2571_257175

/-- The first "split number" of m^3 -/
def firstSplitNumber (m : ℕ) : ℕ := m^2 - m + 1

/-- Predicate to check if a number is one of the "split numbers" of m^3 -/
def isSplitNumber (m : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < m ∧ n = firstSplitNumber m + 2 * k

theorem cube_split_73 (m : ℕ) (h1 : m > 1) (h2 : isSplitNumber m 73) : m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_73_l2571_257175


namespace NUMINAMATH_CALUDE_tangent_line_condition_function_upper_bound_inequality_for_reciprocal_product_l2571_257113

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x - a * x * Real.log x

theorem tangent_line_condition (a : ℝ) : 
  (deriv (f a)) 1 = -1 → a = 1 := by sorry

theorem function_upper_bound (x : ℝ) (hx : x > 0) : 
  x / Real.exp x - x * Real.log x < 2 / Real.exp 1 := by sorry

theorem inequality_for_reciprocal_product (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n = 1) :
  1 / Real.exp m + 1 / Real.exp n < 2 * (m + n) := by sorry

end NUMINAMATH_CALUDE_tangent_line_condition_function_upper_bound_inequality_for_reciprocal_product_l2571_257113


namespace NUMINAMATH_CALUDE_ellipse_cartesian_eq_l2571_257161

def ellipse_eq (x y : ℝ) : Prop :=
  ∃ t : ℝ, x = (3 * (Real.sin t - 2)) / (3 - Real.cos t) ∧
            y = (4 * (Real.cos t - 6)) / (3 - Real.cos t)

theorem ellipse_cartesian_eq :
  ∀ x y : ℝ, ellipse_eq x y ↔ 9*x^2 + 36*x*y + 9*y^2 + 216*x + 432*y + 1440 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_cartesian_eq_l2571_257161


namespace NUMINAMATH_CALUDE_scaling_transformation_maps_line_l2571_257100

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- The original line equation -/
def original_line (x y : ℝ) : Prop := x + y + 2 = 0

/-- The transformed line equation -/
def transformed_line (x y : ℝ) : Prop := 8*x + y + 8 = 0

/-- Theorem stating that the given scaling transformation maps the original line to the transformed line -/
theorem scaling_transformation_maps_line :
  ∃ (t : ScalingTransformation),
    (∀ (x y : ℝ), original_line x y ↔ transformed_line (t.x_scale * x) (t.y_scale * y)) ∧
    t.x_scale = 1/2 ∧ t.y_scale = 4 := by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_maps_line_l2571_257100


namespace NUMINAMATH_CALUDE_range_of_m_l2571_257115

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * log x + x + (1 - a) / x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := exp x + m * x^2 - 2 * exp 2 - 3

theorem range_of_m :
  ∀ m : ℝ, (∃ x₂ : ℝ, x₂ ≥ 1 ∧ ∀ x₁ : ℝ, x₁ ≥ 1 → g m x₂ ≤ f (exp 2 + 1) x₁) ↔ m ≤ exp 2 - exp 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2571_257115


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l2571_257172

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / (x - 2)) ↔ (x ≥ -1 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l2571_257172


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l2571_257104

theorem inequality_solution_implies_m_value (m : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2*x > m*x)) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l2571_257104


namespace NUMINAMATH_CALUDE_dividend_remainder_proof_l2571_257128

theorem dividend_remainder_proof (D d q r : ℕ) : 
  D = 18972 → d = 526 → q = 36 → D = d * q + r → r = 36 := by
  sorry

end NUMINAMATH_CALUDE_dividend_remainder_proof_l2571_257128


namespace NUMINAMATH_CALUDE_football_field_area_l2571_257127

-- Define the football field and fertilizer properties
def total_fertilizer : ℝ := 800
def partial_fertilizer : ℝ := 300
def partial_area : ℝ := 3600

-- Define the theorem
theorem football_field_area :
  (total_fertilizer * partial_area) / partial_fertilizer = 9600 := by
  sorry

end NUMINAMATH_CALUDE_football_field_area_l2571_257127
