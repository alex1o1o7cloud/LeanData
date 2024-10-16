import Mathlib

namespace NUMINAMATH_CALUDE_initial_tickets_correct_l969_96992

/-- The number of tickets Adam initially bought at the fair -/
def initial_tickets : ℕ := 13

/-- The number of tickets left after riding the ferris wheel -/
def tickets_left : ℕ := 4

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 9

/-- The total amount spent on the ferris wheel in dollars -/
def ferris_wheel_cost : ℕ := 81

/-- Theorem stating that the initial number of tickets is correct -/
theorem initial_tickets_correct : 
  initial_tickets = (ferris_wheel_cost / ticket_cost) + tickets_left := by
  sorry

end NUMINAMATH_CALUDE_initial_tickets_correct_l969_96992


namespace NUMINAMATH_CALUDE_quadratic_unique_root_l969_96977

/-- Given real numbers p, q, r forming an arithmetic sequence with p ≥ q ≥ r ≥ 0,
    if the quadratic px^2 + qx + r has exactly one root, then this root is equal to 1 - √6/2 -/
theorem quadratic_unique_root (p q r : ℝ) 
  (arith_seq : ∃ k, q = p - k ∧ r = p - 2*k)
  (order : p ≥ q ∧ q ≥ r ∧ r ≥ 0)
  (unique_root : ∃! x, p*x^2 + q*x + r = 0) :
  ∃ x, p*x^2 + q*x + r = 0 ∧ x = 1 - Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_unique_root_l969_96977


namespace NUMINAMATH_CALUDE_initial_balloons_l969_96922

theorem initial_balloons (x : ℕ) : 
  Odd x ∧ 
  (x / 3 : ℚ) + 10 = 45 → 
  x = 105 := by
sorry

end NUMINAMATH_CALUDE_initial_balloons_l969_96922


namespace NUMINAMATH_CALUDE_four_numbers_theorem_l969_96993

def satisfies_condition (x y z t : ℝ) : Prop :=
  x + y * z * t = 2 ∧
  y + x * z * t = 2 ∧
  z + x * y * t = 2 ∧
  t + x * y * z = 2

theorem four_numbers_theorem :
  ∀ x y z t : ℝ,
    satisfies_condition x y z t ↔
      ((x = 1 ∧ y = 1 ∧ z = 1 ∧ t = 1) ∨
       (x = -1 ∧ y = -1 ∧ z = -1 ∧ t = 3) ∨
       (x = -1 ∧ y = -1 ∧ z = 3 ∧ t = -1) ∨
       (x = -1 ∧ y = 3 ∧ z = -1 ∧ t = -1) ∨
       (x = 3 ∧ y = -1 ∧ z = -1 ∧ t = -1)) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_theorem_l969_96993


namespace NUMINAMATH_CALUDE_postage_calculation_l969_96901

/-- Calculates the postage cost for a letter based on its weight and given rates. -/
def calculatePostage (weight : ℚ) (baseRate : ℚ) (additionalRate : ℚ) : ℚ :=
  let additionalWeight := max (weight - 1) 0
  let additionalCharges := ⌈additionalWeight⌉
  baseRate + additionalCharges * additionalRate

/-- Theorem stating that the postage for a 4.5-ounce letter is 1.18 dollars 
    given the specified rates. -/
theorem postage_calculation :
  let weight : ℚ := 4.5
  let baseRate : ℚ := 0.30
  let additionalRate : ℚ := 0.22
  calculatePostage weight baseRate additionalRate = 1.18 := by
  sorry


end NUMINAMATH_CALUDE_postage_calculation_l969_96901


namespace NUMINAMATH_CALUDE_functional_inequality_domain_l969_96937

-- Define the function f
def f (n : ℕ) (x : ℝ) : ℝ := x^n

-- Define the theorem
theorem functional_inequality_domain (n : ℕ) (h_n : n > 1) :
  ∀ x : ℝ, (f n x + f n (1 - x) > 1) ↔ (x < 0 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_functional_inequality_domain_l969_96937


namespace NUMINAMATH_CALUDE_non_negative_integer_solutions_of_inequality_l969_96983

theorem non_negative_integer_solutions_of_inequality :
  {x : ℕ | x + 1 < 4} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_non_negative_integer_solutions_of_inequality_l969_96983


namespace NUMINAMATH_CALUDE_min_value_inequality_l969_96932

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + 3 * c) + b / (8 * c + 4 * a) + 9 * c / (3 * a + 2 * b) ≥ 47 / 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l969_96932


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l969_96925

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families * total_average) / (total_families - childless_families) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l969_96925


namespace NUMINAMATH_CALUDE_fixed_point_on_moving_line_intersecting_parabola_l969_96978

/-- Theorem: Fixed point on a moving line intersecting a parabola -/
theorem fixed_point_on_moving_line_intersecting_parabola
  (p : ℝ) (k : ℝ) (b : ℝ)
  (hp : p > 0)
  (hk : k ≠ 0)
  (hb : b ≠ 0)
  (h_slope_product : ∀ x₁ y₁ x₂ y₂ : ℝ,
    y₁^2 = 2*p*x₁ → y₂^2 = 2*p*x₂ →
    y₁ = k*x₁ + b → y₂ = k*x₂ + b →
    (y₁ / x₁) * (y₂ / x₂) = Real.sqrt 3) :
  let fixed_point : ℝ × ℝ := (-2*p/Real.sqrt 3, 0)
  ∃ b' : ℝ, k * fixed_point.1 + b' = fixed_point.2 ∧ b' = 2*p*k/Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_fixed_point_on_moving_line_intersecting_parabola_l969_96978


namespace NUMINAMATH_CALUDE_no_solution_equation_l969_96979

theorem no_solution_equation : ¬ ∃ (x y z : ℤ), x^3 + y^6 = 7*z + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l969_96979


namespace NUMINAMATH_CALUDE_rectangle_ratio_l969_96928

/-- Represents the configuration of squares and a rectangle forming a larger square -/
structure SquareConfiguration where
  small_square_side : ℝ
  large_square_side : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The theorem stating the ratio of the rectangle's length to its width -/
theorem rectangle_ratio (config : SquareConfiguration) 
  (h1 : config.large_square_side = 4 * config.small_square_side)
  (h2 : config.rectangle_length = config.large_square_side)
  (h3 : config.rectangle_width = config.large_square_side - 3 * config.small_square_side) :
  config.rectangle_length / config.rectangle_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l969_96928


namespace NUMINAMATH_CALUDE_valid_tiling_characterization_l969_96920

/-- A tetromino type -/
inductive Tetromino
| T
| Square

/-- Represents a tiling of an n × n field -/
structure Tiling (n : ℕ) where
  pieces : List (Tetromino × ℕ × ℕ)  -- List of (type, row, col) for each piece
  no_gaps : Sorry
  no_overlaps : Sorry
  covers_field : Sorry
  odd_squares : Sorry  -- The number of square tetrominoes is odd

/-- Main theorem: Characterization of valid n for tiling -/
theorem valid_tiling_characterization (n : ℕ) :
  (∃ (t : Tiling n), True) ↔ (∃ (k : ℕ), n = 2 * k ∧ k % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_valid_tiling_characterization_l969_96920


namespace NUMINAMATH_CALUDE_product_equation_sum_l969_96969

theorem product_equation_sum (p q r s : ℤ) : 
  (∀ x, (x^2 + p*x + q) * (x^2 + r*x + s) = x^4 - x^3 + 3*x^2 - 4*x + 4) →
  p + q + r + s = -1 := by
sorry

end NUMINAMATH_CALUDE_product_equation_sum_l969_96969


namespace NUMINAMATH_CALUDE_complex_equality_implies_values_l969_96949

theorem complex_equality_implies_values (x y : ℝ) : 
  (Complex.mk (x - 1) y = Complex.mk 0 1 - Complex.mk (3*x) 0) → 
  (x = 1/4 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_values_l969_96949


namespace NUMINAMATH_CALUDE_first_term_is_seven_l969_96966

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 2) + (-1)^n * a n = 3 * n - 1

/-- The sum of the first 16 terms of the sequence equals 540 -/
def SumCondition (a : ℕ → ℚ) : Prop :=
  (Finset.range 16).sum a = 540

/-- The theorem stating that a₁ = 7 for the given conditions -/
theorem first_term_is_seven
    (a : ℕ → ℚ)
    (h_recurrence : RecurrenceSequence a)
    (h_sum : SumCondition a) :
    a 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_seven_l969_96966


namespace NUMINAMATH_CALUDE_table_runner_coverage_l969_96917

theorem table_runner_coverage (
  total_runner_area : ℝ)
  (table_area : ℝ)
  (two_layer_area : ℝ)
  (three_layer_area : ℝ)
  (h1 : total_runner_area = 212)
  (h2 : table_area = 175)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 24)
  : ∃ (coverage_percentage : ℝ),
    abs (coverage_percentage - 52.57) < 0.01 ∧
    coverage_percentage = (total_runner_area - 2 * two_layer_area - 3 * three_layer_area) / table_area * 100 := by
  sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l969_96917


namespace NUMINAMATH_CALUDE_megan_math_problems_l969_96944

/-- Proves that Megan had 36 math problems given the conditions of the problem -/
theorem megan_math_problems :
  ∀ (total_problems math_problems spelling_problems : ℕ)
    (problems_per_hour hours_taken : ℕ),
  spelling_problems = 28 →
  problems_per_hour = 8 →
  hours_taken = 8 →
  total_problems = math_problems + spelling_problems →
  total_problems = problems_per_hour * hours_taken →
  math_problems = 36 := by
  sorry

end NUMINAMATH_CALUDE_megan_math_problems_l969_96944


namespace NUMINAMATH_CALUDE_integer_representation_l969_96924

theorem integer_representation (k : ℤ) (h : -1985 ≤ k ∧ k ≤ 1985) :
  ∃ (a : Fin 8 → ℤ), (∀ i, a i ∈ ({-1, 0, 1} : Set ℤ)) ∧
    k = (a 0) * 1 + (a 1) * 3 + (a 2) * 9 + (a 3) * 27 +
        (a 4) * 81 + (a 5) * 243 + (a 6) * 729 + (a 7) * 2187 :=
by sorry

end NUMINAMATH_CALUDE_integer_representation_l969_96924


namespace NUMINAMATH_CALUDE_lines_sum_l969_96942

-- Define the lines
def l₀ (x y : ℝ) : Prop := x - y + 1 = 0
def l₁ (a x y : ℝ) : Prop := a * x - 2 * y + 1 = 0
def l₂ (b x y : ℝ) : Prop := x + b * y + 3 = 0

-- Define perpendicularity and parallelism
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, f x₁ y₁ → f x₂ y₂ → g x₁ y₁ → g x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (y₂ - y₁) - (y₂ - y₁) * (x₂ - x₁) = 0)

def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    f x₁ y₁ → f x₂ y₂ → g x₃ y₃ → g x₄ y₄ →
    (x₂ - x₁) * (y₄ - y₃) = (y₂ - y₁) * (x₄ - x₃)

-- Theorem statement
theorem lines_sum (a b : ℝ) : 
  perpendicular (l₀) (l₁ a) → parallel (l₀) (l₂ b) → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_lines_sum_l969_96942


namespace NUMINAMATH_CALUDE_problem_solution_l969_96904

-- Define the conditions
def is_square_root_of_same_number (x y : ℝ) : Prop := ∃ z : ℝ, z > 0 ∧ x^2 = z ∧ y^2 = z

-- Main theorem
theorem problem_solution :
  ∀ (a b c : ℝ),
  (is_square_root_of_same_number (a + 3) (2*a - 15)) →
  (b^(1/3 : ℝ) = -2) →
  (c ≥ 0 ∧ c^(1/2 : ℝ) = c) →
  ((c = 0 → a + b - 2*c = -4) ∧ (c = 1 → a + b - 2*c = -6)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l969_96904


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l969_96954

theorem inequality_and_equality_condition (a b c : ℝ) (h : a * b * c = 1 / 8) :
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 ∧
  (a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = 15 / 16 ↔ a = 1 / 2 ∧ b = 1 / 2 ∧ c = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l969_96954


namespace NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l969_96961

-- Define the type for spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the standard representation constraints
def isStandardRepresentation (coord : SphericalCoord) : Prop :=
  coord.ρ > 0 ∧ 0 ≤ coord.θ ∧ coord.θ < 2 * Real.pi ∧ 0 ≤ coord.φ ∧ coord.φ ≤ Real.pi

-- Define the equivalence relation between spherical coordinates
def sphericalEquivalent (coord1 coord2 : SphericalCoord) : Prop :=
  coord1.ρ = coord2.ρ ∧
  (coord1.θ % (2 * Real.pi) = coord2.θ % (2 * Real.pi)) ∧
  ((coord1.φ % (2 * Real.pi) = coord2.φ % (2 * Real.pi)) ∨
   (coord1.φ % (2 * Real.pi) = 2 * Real.pi - (coord2.φ % (2 * Real.pi))))

-- Theorem statement
theorem spherical_coordinate_equivalence :
  let original := SphericalCoord.mk 5 (5 * Real.pi / 6) (9 * Real.pi / 5)
  let standard := SphericalCoord.mk 5 (11 * Real.pi / 6) (Real.pi / 5)
  sphericalEquivalent original standard ∧ isStandardRepresentation standard :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l969_96961


namespace NUMINAMATH_CALUDE_fraction_enlargement_l969_96964

theorem fraction_enlargement (x y : ℝ) (h : x ≠ y) :
  (3 * x) * (3 * y) / ((3 * x) - (3 * y)) = 3 * (x * y / (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_enlargement_l969_96964


namespace NUMINAMATH_CALUDE_barbecue_packages_l969_96990

/-- Represents the number of items in each package type -/
structure PackageSizes where
  hotDogs : Nat
  buns : Nat
  soda : Nat

/-- Represents the number of packages for each item type -/
structure PackageCounts where
  hotDogs : Nat
  buns : Nat
  soda : Nat

/-- Given package sizes, check if the package counts result in equal number of items -/
def hasEqualItems (sizes : PackageSizes) (counts : PackageCounts) : Prop :=
  sizes.hotDogs * counts.hotDogs = sizes.buns * counts.buns ∧
  sizes.hotDogs * counts.hotDogs = sizes.soda * counts.soda

/-- Check if a given package count is the smallest possible -/
def isSmallestCount (sizes : PackageSizes) (counts : PackageCounts) : Prop :=
  ∀ (other : PackageCounts),
    hasEqualItems sizes other →
    counts.hotDogs ≤ other.hotDogs ∧
    counts.buns ≤ other.buns ∧
    counts.soda ≤ other.soda

theorem barbecue_packages :
  let sizes : PackageSizes := ⟨9, 12, 15⟩
  let counts : PackageCounts := ⟨20, 15, 12⟩
  hasEqualItems sizes counts ∧ isSmallestCount sizes counts :=
by sorry

end NUMINAMATH_CALUDE_barbecue_packages_l969_96990


namespace NUMINAMATH_CALUDE_bird_watching_average_l969_96909

theorem bird_watching_average :
  let marcus_birds : ℕ := 7
  let humphrey_birds : ℕ := 11
  let darrel_birds : ℕ := 9
  let total_birds : ℕ := marcus_birds + humphrey_birds + darrel_birds
  let num_people : ℕ := 3
  (total_birds : ℚ) / num_people = 9 := by sorry

end NUMINAMATH_CALUDE_bird_watching_average_l969_96909


namespace NUMINAMATH_CALUDE_square_or_double_square_l969_96995

theorem square_or_double_square (p m n : ℕ) : 
  Prime p → 
  m ≠ n → 
  p^2 = (m^2 + n^2) / 2 → 
  ∃ k : ℤ, (2*p - m - n : ℤ) = k^2 ∨ (2*p - m - n : ℤ) = 2*k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_or_double_square_l969_96995


namespace NUMINAMATH_CALUDE_max_students_distribution_l969_96907

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : 
  Nat.gcd pens pencils = 91 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l969_96907


namespace NUMINAMATH_CALUDE_ball_difference_l969_96988

/-- Problem: Difference between basketballs and soccer balls --/
theorem ball_difference (total : ℕ) (soccer : ℕ) (tennis : ℕ) (baseball : ℕ) (volleyball : ℕ) (basketball : ℕ) : 
  total = 145 →
  soccer = 20 →
  tennis = 2 * soccer →
  baseball = soccer + 10 →
  volleyball = 30 →
  basketball > soccer →
  total = soccer + tennis + baseball + volleyball + basketball →
  basketball - soccer = 5 := by
  sorry

#check ball_difference

end NUMINAMATH_CALUDE_ball_difference_l969_96988


namespace NUMINAMATH_CALUDE_kendys_initial_balance_l969_96965

/-- Proves that Kendy's initial account balance was $190 given the conditions of her transfers --/
theorem kendys_initial_balance :
  let mom_transfer : ℕ := 60
  let sister_transfer : ℕ := mom_transfer / 2
  let remaining_balance : ℕ := 100
  let initial_balance : ℕ := remaining_balance + mom_transfer + sister_transfer
  initial_balance = 190 := by
  sorry

end NUMINAMATH_CALUDE_kendys_initial_balance_l969_96965


namespace NUMINAMATH_CALUDE_max_valid_domains_l969_96929

def f (x : ℝ) : ℝ := x^2 - 1

def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x = 0 ∨ f x = 1) ∧
  (∃ x ∈ D, f x = 0) ∧
  (∃ x ∈ D, f x = 1)

theorem max_valid_domains :
  ∃ (domains : Finset (Set ℝ)),
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D, is_valid_domain D → D ∈ domains) ∧
    domains.card = 9 :=
sorry

end NUMINAMATH_CALUDE_max_valid_domains_l969_96929


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l969_96962

def star_list : List Nat := List.range 40

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem star_emilio_sum_difference :
  star_list.sum - emilio_list.sum = 104 := by sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l969_96962


namespace NUMINAMATH_CALUDE_dot_product_OA_OB_line_l_equations_l969_96953

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point M
def M : ℝ × ℝ := (6, 0)

-- Define line l passing through M and intersecting the parabola
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 6

-- Define points A and B as intersections of line l and the parabola
def intersect_points (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Theorem for the dot product of OA and OB
theorem dot_product_OA_OB (m : ℝ) :
  let ((x₁, y₁), (x₂, y₂)) := intersect_points m
  (x₁ * x₂ + y₁ * y₂ : ℝ) = 12 := sorry

-- Theorem for the equations of line l given the area of triangle OAB
theorem line_l_equations :
  (∃ m : ℝ, let ((x₁, y₁), (x₂, y₂)) := intersect_points m
   (1/2 : ℝ) * 6 * |y₁ - y₂| = 12 * Real.sqrt 10) →
  (∃ l₁ l₂ : ℝ → ℝ → Prop,
    (∀ x y, l₁ x y ↔ x + 2*y - 6 = 0) ∧
    (∀ x y, l₂ x y ↔ x - 2*y - 6 = 0) ∧
    (∀ x y, line_l 2 x y ↔ l₁ x y) ∧
    (∀ x y, line_l (-2) x y ↔ l₂ x y)) := sorry

end NUMINAMATH_CALUDE_dot_product_OA_OB_line_l_equations_l969_96953


namespace NUMINAMATH_CALUDE_circular_field_area_l969_96914

-- Define the constants
def fencing_cost_per_metre : ℝ := 4
def total_fencing_cost : ℝ := 5941.9251828093165

-- Define the theorem
theorem circular_field_area :
  ∃ (area : ℝ),
    (area ≥ 17.55 ∧ area ≤ 17.57) ∧
    (∃ (circumference radius : ℝ),
      circumference = total_fencing_cost / fencing_cost_per_metre ∧
      radius = circumference / (2 * Real.pi) ∧
      area = (Real.pi * radius ^ 2) / 10000) :=
by sorry

end NUMINAMATH_CALUDE_circular_field_area_l969_96914


namespace NUMINAMATH_CALUDE_student_fails_by_10_marks_l969_96981

/-- Calculates the number of marks a student fails by in a test -/
def marksFailed (maxMarks : ℕ) (passingPercentage : ℚ) (studentScore : ℕ) : ℕ :=
  let passingMark := (maxMarks : ℚ) * passingPercentage
  (passingMark.ceil - studentScore).toNat

/-- Proves that a student who scores 80 marks in a 300-mark test with 30% passing requirement fails by 10 marks -/
theorem student_fails_by_10_marks :
  marksFailed 300 (30 / 100) 80 = 10 := by
  sorry

end NUMINAMATH_CALUDE_student_fails_by_10_marks_l969_96981


namespace NUMINAMATH_CALUDE_unique_solution_l969_96939

theorem unique_solution : ∃! x : ℝ, 
  (|x - 3| + |x + 4| < 8) ∧ (x^2 - x - 12 = 0) :=
by
  -- The unique solution is x = -3
  use -3
  constructor
  · -- Prove that x = -3 satisfies both conditions
    constructor
    · -- Prove |(-3) - 3| + |(-3) + 4| < 8
      sorry
    · -- Prove (-3)^2 - (-3) - 12 = 0
      sorry
  · -- Prove that no other value satisfies both conditions
    sorry

#check unique_solution

end NUMINAMATH_CALUDE_unique_solution_l969_96939


namespace NUMINAMATH_CALUDE_micah_envelope_count_l969_96976

def envelope_count (total_stamps : ℕ) (light_envelopes : ℕ) (stamps_per_light : ℕ) (stamps_per_heavy : ℕ) : ℕ :=
  let heavy_stamps := total_stamps - light_envelopes * stamps_per_light
  let heavy_envelopes := heavy_stamps / stamps_per_heavy
  light_envelopes + heavy_envelopes

theorem micah_envelope_count :
  envelope_count 52 6 2 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_micah_envelope_count_l969_96976


namespace NUMINAMATH_CALUDE_vector_addition_l969_96945

/-- Given two 2D vectors a and b, prove that their sum is equal to (4, 6) -/
theorem vector_addition (a b : ℝ × ℝ) (h1 : a = (6, 2)) (h2 : b = (-2, 4)) :
  a + b = (4, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l969_96945


namespace NUMINAMATH_CALUDE_sum_five_probability_l969_96974

theorem sum_five_probability (n : ℕ) : n ≥ 5 →
  (Nat.choose n 2 : ℚ)⁻¹ * 2 = 1 / 14 ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_sum_five_probability_l969_96974


namespace NUMINAMATH_CALUDE_power_function_through_point_l969_96943

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x : ℝ, f x = x ^ α) →  -- f is a power function
  f 2 = Real.sqrt 2 →       -- f passes through (2, √2)
  ∀ x : ℝ, f x = x ^ (1/2)  -- f(x) = x^(1/2)
:= by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l969_96943


namespace NUMINAMATH_CALUDE_initial_blue_pens_l969_96972

theorem initial_blue_pens (initial_black : ℕ) (initial_red : ℕ) 
  (blue_removed : ℕ) (black_removed : ℕ) (remaining : ℕ) :
  initial_black = 21 →
  initial_red = 6 →
  blue_removed = 4 →
  black_removed = 7 →
  remaining = 25 →
  ∃ initial_blue : ℕ, 
    initial_blue + initial_black + initial_red = 
    remaining + blue_removed + black_removed ∧
    initial_blue = 9 :=
by sorry

end NUMINAMATH_CALUDE_initial_blue_pens_l969_96972


namespace NUMINAMATH_CALUDE_find_number_l969_96900

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 6) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_find_number_l969_96900


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_300_150_l969_96948

/-- The largest 2-digit prime factor of (300 choose 150) -/
def largest_two_digit_prime_factor_of_binom : ℕ := 97

/-- The binomial coefficient (300 choose 150) -/
def binom_300_150 : ℕ := Nat.choose 300 150

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  largest_two_digit_prime_factor_of_binom = 97 ∧
  Nat.Prime largest_two_digit_prime_factor_of_binom ∧
  largest_two_digit_prime_factor_of_binom ≥ 10 ∧
  largest_two_digit_prime_factor_of_binom < 100 ∧
  (binom_300_150 % largest_two_digit_prime_factor_of_binom = 0) ∧
  ∀ p : ℕ, Nat.Prime p → p ≥ 10 → p < 100 → 
    (binom_300_150 % p = 0) → p ≤ largest_two_digit_prime_factor_of_binom :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_300_150_l969_96948


namespace NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l969_96959

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2 : ℝ) = π * (r₁^2 - r₂^2) :=
sorry

/-- The area of a ring formed by two concentric circles with radii 10 and 6 is 64π -/
theorem area_of_specific_ring : 
  (π * 10^2 - π * 6^2 : ℝ) = 64 * π :=
sorry

end NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l969_96959


namespace NUMINAMATH_CALUDE_cobalt_percentage_is_15_percent_l969_96980

/-- Represents the composition of a mixture -/
structure Mixture where
  cobalt : ℝ
  lead : ℝ
  copper : ℝ

/-- The given mixture satisfies the problem conditions -/
def problem_mixture : Mixture where
  lead := 0.25
  copper := 0.60
  cobalt := 1 - (0.25 + 0.60)

/-- The total weight of the mixture in kg -/
def total_weight : ℝ := 5 + 12

theorem cobalt_percentage_is_15_percent (m : Mixture) 
  (h1 : m.lead = 0.25)
  (h2 : m.copper = 0.60)
  (h3 : m.lead + m.copper + m.cobalt = 1)
  (h4 : m.lead * total_weight = 5)
  (h5 : m.copper * total_weight = 12) :
  m.cobalt = 0.15 := by
  sorry

#check cobalt_percentage_is_15_percent

end NUMINAMATH_CALUDE_cobalt_percentage_is_15_percent_l969_96980


namespace NUMINAMATH_CALUDE_wilsons_theorem_l969_96998

theorem wilsons_theorem (p : Nat) (hp : Nat.Prime p) : (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l969_96998


namespace NUMINAMATH_CALUDE_shekar_science_marks_l969_96911

/-- Represents a student's marks in different subjects -/
structure StudentMarks where
  mathematics : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ
  science : ℕ
  average : ℕ
  total_subjects : ℕ

/-- Theorem stating that given Shekar's marks in other subjects and his average, 
    his science marks must be 65 -/
theorem shekar_science_marks (marks : StudentMarks) 
  (h1 : marks.mathematics = 76)
  (h2 : marks.social_studies = 82)
  (h3 : marks.english = 47)
  (h4 : marks.biology = 85)
  (h5 : marks.average = 71)
  (h6 : marks.total_subjects = 5)
  : marks.science = 65 := by
  sorry

#check shekar_science_marks

end NUMINAMATH_CALUDE_shekar_science_marks_l969_96911


namespace NUMINAMATH_CALUDE_trigonometric_identity_l969_96975

theorem trigonometric_identity (α β γ x : ℝ) : 
  (Real.sin (x - β) * Real.sin (x - γ)) / (Real.sin (α - β) * Real.sin (α - γ)) +
  (Real.sin (x - γ) * Real.sin (x - α)) / (Real.sin (β - γ) * Real.sin (β - α)) +
  (Real.sin (x - α) * Real.sin (x - β)) / (Real.sin (γ - α) * Real.sin (γ - β)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l969_96975


namespace NUMINAMATH_CALUDE_child_height_calculation_l969_96996

/-- Given a child's current height and growth since last visit, 
    calculate the child's height at the last visit. -/
def height_at_last_visit (current_height growth : ℝ) : ℝ :=
  current_height - growth

/-- Theorem stating that given the specific measurements, 
    the child's height at the last visit was 38.5 inches. -/
theorem child_height_calculation : 
  height_at_last_visit 41.5 3 = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_child_height_calculation_l969_96996


namespace NUMINAMATH_CALUDE_ellipse_circle_theorem_l969_96934

/-- Definition of the ellipse C -/
def ellipse_C (b : ℝ) (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / b^2 = 1 ∧ b > 0

/-- Definition of the circle O -/
def circle_O (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2 ∧ r > 0

/-- Definition of the right focus F of ellipse C -/
def right_focus (F : ℝ × ℝ) (b : ℝ) : Prop :=
  F.1 > 0 ∧ ellipse_C b F.1 F.2

/-- Definition of the tangent lines from F to circle O -/
def tangent_lines (F A B : ℝ × ℝ) (r : ℝ) : Prop :=
  circle_O r A.1 A.2 ∧ circle_O r B.1 B.2 ∧
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2

/-- Definition of right triangle ABF -/
def right_triangle (A B F : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 0

/-- Definition of maximum distance between points on C and O -/
def max_distance (b r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C b x₁ y₁ ∧ circle_O r x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (Real.sqrt 3 + 1)^2

/-- Main theorem -/
theorem ellipse_circle_theorem
  (b r : ℝ) (F A B : ℝ × ℝ) :
  ellipse_C b F.1 F.2 →
  right_focus F b →
  circle_O r A.1 A.2 →
  tangent_lines F A B r →
  right_triangle A B F →
  max_distance b r →
  (r = 1 ∧ b = 1) ∧
  (∀ (k m : ℝ), k < 0 → m > 0 →
    (∃ (P Q : ℝ × ℝ),
      ellipse_C b P.1 P.2 ∧ ellipse_C b Q.1 Q.2 ∧
      P.2 = k * P.1 + m ∧ Q.2 = k * Q.1 + m ∧
      (P.1 - F.1)^2 + (P.2 - F.2)^2 +
      (Q.1 - F.1)^2 + (Q.2 - F.2)^2 +
      (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 12)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_theorem_l969_96934


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l969_96989

/-- The angle with the same terminal side as α = π/12 + 2kπ (k ∈ ℤ) is equivalent to 25π/12 radians. -/
theorem same_terminal_side_angle (k : ℤ) : ∃ (n : ℤ), (π/12 + 2*k*π) = 25*π/12 + 2*n*π := by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l969_96989


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l969_96919

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  (7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧
  (a.val = 18 ∧ b.val = 1) := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l969_96919


namespace NUMINAMATH_CALUDE_train_length_l969_96913

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 9 → ∃ length : ℝ, 
  (length ≥ 74.96 ∧ length ≤ 74.98) ∧ length = speed * (5/18) * time := by
  sorry

end NUMINAMATH_CALUDE_train_length_l969_96913


namespace NUMINAMATH_CALUDE_first_quadrant_is_well_defined_set_l969_96912

-- Define the first quadrant
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

-- Theorem stating that the FirstQuadrant is a well-defined set
theorem first_quadrant_is_well_defined_set : 
  ∀ p : ℝ × ℝ, Decidable (p ∈ FirstQuadrant) :=
by
  sorry


end NUMINAMATH_CALUDE_first_quadrant_is_well_defined_set_l969_96912


namespace NUMINAMATH_CALUDE_lift_cars_and_trucks_l969_96927

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars to be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks to be lifted -/
def num_trucks : ℕ := 3

/-- The total number of people needed to lift the given number of cars and trucks -/
def total_people : ℕ := num_cars * people_per_car + num_trucks * people_per_truck

theorem lift_cars_and_trucks : total_people = 60 := by
  sorry

end NUMINAMATH_CALUDE_lift_cars_and_trucks_l969_96927


namespace NUMINAMATH_CALUDE_max_sides_cube_plane_intersection_l969_96915

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A plane is a flat, two-dimensional surface -/
structure Plane where
  -- We don't need to define the specifics of a plane for this problem

/-- A polygon is a plane figure with straight sides -/
structure Polygon where
  sides : ℕ

/-- The cross-section formed when a plane intersects a cube -/
def crossSection (c : Cube) (p : Plane) : Polygon :=
  sorry -- Implementation details not needed for the statement

/-- The maximum number of sides a polygon can have when it's formed by a plane intersecting a cube is 6 -/
theorem max_sides_cube_plane_intersection (c : Cube) (p : Plane) :
  (crossSection c p).sides ≤ 6 ∧ ∃ (c : Cube) (p : Plane), (crossSection c p).sides = 6 :=
sorry

end NUMINAMATH_CALUDE_max_sides_cube_plane_intersection_l969_96915


namespace NUMINAMATH_CALUDE_joan_video_game_spending_l969_96935

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.20

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending :
  total_spent = 9.43 := by sorry

end NUMINAMATH_CALUDE_joan_video_game_spending_l969_96935


namespace NUMINAMATH_CALUDE_num_divisors_10_factorial_l969_96984

/-- The number of positive divisors of n! -/
def numDivisorsFactorial (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 10! is 192 -/
theorem num_divisors_10_factorial :
  numDivisorsFactorial 10 = 192 := by sorry

end NUMINAMATH_CALUDE_num_divisors_10_factorial_l969_96984


namespace NUMINAMATH_CALUDE_nearest_whole_number_solution_l969_96950

theorem nearest_whole_number_solution (x : ℝ) : 
  x * 54 = 75625 → 
  ⌊x + 0.5⌋ = 1400 :=
sorry

end NUMINAMATH_CALUDE_nearest_whole_number_solution_l969_96950


namespace NUMINAMATH_CALUDE_log_five_twelve_l969_96931

theorem log_five_twelve (a b : ℝ) (h1 : Real.log 2 = a * Real.log 10) (h2 : Real.log 3 = b * Real.log 10) :
  Real.log 12 / Real.log 5 = (2*a + b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_five_twelve_l969_96931


namespace NUMINAMATH_CALUDE_johann_oranges_l969_96947

theorem johann_oranges (x : ℕ) : 
  (x - 10) / 2 + 5 = 30 → x = 60 := by sorry

end NUMINAMATH_CALUDE_johann_oranges_l969_96947


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l969_96906

-- Define the center of the circle
def center : ℝ × ℝ := (2, 1)

-- Define the tangent line
def tangent_line (y : ℝ) : Prop := y + 1 = 0

-- Define the distance from a point to the tangent line
def distance_to_line (x y : ℝ) : ℝ := |y + 1|

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Theorem statement
theorem circle_tangent_to_line :
  ∀ x y : ℝ, circle_equation x y →
  distance_to_line x y = (4 : ℝ).sqrt ∧
  ∃ p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ tangent_line p.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l969_96906


namespace NUMINAMATH_CALUDE_water_depth_calculation_l969_96999

def rons_height : ℝ := 13

def water_depth : ℝ := 16 * rons_height

theorem water_depth_calculation : water_depth = 208 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l969_96999


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identity_l969_96905

theorem triangle_trigonometric_identity (A B C : Real) : 
  C = Real.pi / 3 →
  Real.tan (A / 2) + Real.tan (B / 2) = 1 →
  A + B + C = Real.pi →
  Real.sin (A / 2) * Real.sin (B / 2) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identity_l969_96905


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_has_20_sides_l969_96960

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_has_20_sides :
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_has_20_sides_l969_96960


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l969_96957

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0)
  (h_inv_prop : ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, x * y = k)
  (h_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l969_96957


namespace NUMINAMATH_CALUDE_situps_total_l969_96958

/-- The number of sit-ups Barney can perform in one minute -/
def barney_situps : ℕ := 45

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := barney_situps * barney_minutes + 
                        carrie_situps * carrie_minutes + 
                        jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end NUMINAMATH_CALUDE_situps_total_l969_96958


namespace NUMINAMATH_CALUDE_triangle_angle_from_complex_trig_l969_96982

theorem triangle_angle_from_complex_trig (A B C : Real) : 
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  A + B + C = π →
  (Complex.exp (I * A)) * (Complex.exp (I * B)) = Complex.exp (I * C) →
  C = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_from_complex_trig_l969_96982


namespace NUMINAMATH_CALUDE_train_speed_calculation_l969_96938

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/hr -/
theorem train_speed_calculation (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 →
  bridge_length = 175 →
  crossing_time = 14.248860091192705 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l969_96938


namespace NUMINAMATH_CALUDE_flour_sugar_difference_l969_96963

theorem flour_sugar_difference (total_flour sugar_needed flour_added : ℕ) : 
  total_flour = 14 →
  sugar_needed = 9 →
  flour_added = 4 →
  (total_flour - flour_added) - sugar_needed = 1 := by
  sorry

end NUMINAMATH_CALUDE_flour_sugar_difference_l969_96963


namespace NUMINAMATH_CALUDE_complex_equation_solution_l969_96952

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 3 + I → z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l969_96952


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_a_7_value_l969_96946

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 1 + a 7 = a 3 + a 5 := by sorry

theorem a_7_value (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 + a 5 = 10) : 
  a 7 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_a_7_value_l969_96946


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l969_96910

theorem quadratic_inequality_solution_set :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2} = {x : ℝ | -x^2 + 3*x - 2 ≥ 0} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l969_96910


namespace NUMINAMATH_CALUDE_possible_in_99_attempts_possible_in_75_attempts_impossible_in_74_attempts_l969_96930

/-- A type representing a door or a key --/
def DoorKey := Fin 100

/-- A function representing the mapping of keys to doors --/
def KeyToDoor := DoorKey → DoorKey

/-- Predicate to check if a key-to-door mapping is valid --/
def IsValidMapping (f : KeyToDoor) : Prop :=
  ∀ k : DoorKey, (f k).val = k.val ∨ (f k).val = k.val + 1 ∨ (f k).val = k.val - 1

/-- Theorem stating that it's possible to determine the key-door mapping in 99 attempts --/
theorem possible_in_99_attempts (f : KeyToDoor) (h : IsValidMapping f) :
  ∃ (algorithm : ℕ → DoorKey × DoorKey),
    (∀ n : ℕ, n < 99 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
    ∀ k : DoorKey, ∃ n : ℕ, n < 99 ∧ (algorithm n).1 = k ∧ (algorithm n).2 = f k :=
  sorry

/-- Theorem stating that it's possible to determine the key-door mapping in 75 attempts --/
theorem possible_in_75_attempts (f : KeyToDoor) (h : IsValidMapping f) :
  ∃ (algorithm : ℕ → DoorKey × DoorKey),
    (∀ n : ℕ, n < 75 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
    ∀ k : DoorKey, ∃ n : ℕ, n < 75 ∧ (algorithm n).1 = k ∧ (algorithm n).2 = f k :=
  sorry

/-- Theorem stating that it's impossible to determine the key-door mapping in 74 attempts --/
theorem impossible_in_74_attempts :
  ∃ f : KeyToDoor, IsValidMapping f ∧
    ∀ (algorithm : ℕ → DoorKey × DoorKey),
      (∀ n : ℕ, n < 74 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
      ∃ k : DoorKey, ∀ n : ℕ, n < 74 → (algorithm n).1 ≠ k ∨ (algorithm n).2 ≠ f k :=
  sorry

end NUMINAMATH_CALUDE_possible_in_99_attempts_possible_in_75_attempts_impossible_in_74_attempts_l969_96930


namespace NUMINAMATH_CALUDE_impossible_card_arrangement_l969_96941

/-- Represents the arrangement of cards --/
def CardArrangement := List ℕ

/-- Calculates the sum of spaces between pairs of identical digits --/
def sumOfSpaces (arr : CardArrangement) : ℕ := sorry

/-- Checks if an arrangement is valid according to the problem's conditions --/
def isValidArrangement (arr : CardArrangement) : Prop := sorry

/-- Theorem stating the impossibility of the desired arrangement --/
theorem impossible_card_arrangement : 
  ¬ ∃ (arr : CardArrangement), 
    (arr.length = 20) ∧ 
    (∀ d, (arr.count d = 2) ∨ (arr.count d = 0)) ∧
    (isValidArrangement arr) := by
  sorry

end NUMINAMATH_CALUDE_impossible_card_arrangement_l969_96941


namespace NUMINAMATH_CALUDE_turner_rides_l969_96908

theorem turner_rides (rollercoaster_rides : ℕ) (ferris_wheel_rides : ℕ) 
  (rollercoaster_cost : ℕ) (catapult_cost : ℕ) (ferris_wheel_cost : ℕ) 
  (total_tickets : ℕ) :
  rollercoaster_rides = 3 →
  ferris_wheel_rides = 1 →
  rollercoaster_cost = 4 →
  catapult_cost = 4 →
  ferris_wheel_cost = 1 →
  total_tickets = 21 →
  ∃ catapult_rides : ℕ, 
    catapult_rides * catapult_cost + 
    rollercoaster_rides * rollercoaster_cost + 
    ferris_wheel_rides * ferris_wheel_cost = total_tickets ∧
    catapult_rides = 2 :=
by sorry

end NUMINAMATH_CALUDE_turner_rides_l969_96908


namespace NUMINAMATH_CALUDE_luke_trivia_game_l969_96923

/-- Given a trivia game where a player gains a constant number of points per round
    and achieves a total score, calculate the number of rounds played. -/
def rounds_played (points_per_round : ℕ) (total_points : ℕ) : ℕ :=
  total_points / points_per_round

/-- Luke's trivia game scenario -/
theorem luke_trivia_game : rounds_played 3 78 = 26 := by
  sorry

end NUMINAMATH_CALUDE_luke_trivia_game_l969_96923


namespace NUMINAMATH_CALUDE_freddy_call_cost_l969_96985

/-- Calculates the total cost of phone calls in dollars -/
def total_call_cost (local_duration : ℕ) (international_duration : ℕ) 
                    (local_rate : ℚ) (international_rate : ℚ) : ℚ :=
  (local_duration : ℚ) * local_rate + (international_duration : ℚ) * international_rate

/-- Proves that Freddy's total call cost is $10.00 -/
theorem freddy_call_cost : 
  total_call_cost 45 31 (5 / 100) (25 / 100) = 10 := by
  sorry

#eval total_call_cost 45 31 (5 / 100) (25 / 100)

end NUMINAMATH_CALUDE_freddy_call_cost_l969_96985


namespace NUMINAMATH_CALUDE_largest_negative_integer_l969_96955

theorem largest_negative_integer : 
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l969_96955


namespace NUMINAMATH_CALUDE_system_of_equations_product_l969_96903

theorem system_of_equations_product (a b c d : ℚ) : 
  3*a + 4*b + 6*c + 8*d = 48 →
  4*(d+c) = b →
  4*b + 2*c = a →
  c - 2 = d →
  a * b * c * d = -1032192 / 1874161 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_product_l969_96903


namespace NUMINAMATH_CALUDE_base8_subtraction_and_conversion_l969_96902

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Subtracts two numbers in base 8 -/
def subtractBase8 (a b : ℕ) : ℕ := sorry

theorem base8_subtraction_and_conversion :
  let a := 7463
  let b := 3254
  let result_base8 := subtractBase8 a b
  let result_base10 := base8ToBase10 result_base8
  result_base8 = 4207 ∧ result_base10 = 2183 := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_and_conversion_l969_96902


namespace NUMINAMATH_CALUDE_linear_function_characterization_l969_96968

/-- A linear function f satisfying f(f(x)) = 4x + 6 -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧ 
  (∀ x, f (f x) = 4 * x + 6)

/-- Theorem stating that a linear function f satisfying f(f(x)) = 4x + 6 
    must be either f(x) = 2x + 2 or f(x) = -2x - 6 -/
theorem linear_function_characterization (f : ℝ → ℝ) :
  LinearFunction f → 
  (∀ x, f x = 2 * x + 2) ∨ (∀ x, f x = -2 * x - 6) :=
sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l969_96968


namespace NUMINAMATH_CALUDE_individual_can_cost_l969_96936

def pack_size : ℕ := 12
def pack_cost : ℚ := 299 / 100  -- $2.99 represented as a rational number

theorem individual_can_cost :
  let cost_per_can := pack_cost / pack_size
  cost_per_can = 299 / (100 * 12) := by sorry

end NUMINAMATH_CALUDE_individual_can_cost_l969_96936


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l969_96933

theorem geometric_sequence_problem (q : ℝ) (S₆ : ℝ) (b₁ : ℝ) (b₅ : ℝ) : 
  q = 3 → 
  S₆ = 1820 → 
  S₆ = b₁ * (1 - q^6) / (1 - q) → 
  b₅ = b₁ * q^4 →
  b₁ = 5 ∧ b₅ = 405 := by
  sorry

#check geometric_sequence_problem

end NUMINAMATH_CALUDE_geometric_sequence_problem_l969_96933


namespace NUMINAMATH_CALUDE_no_integer_solution_for_sum_of_cubes_l969_96973

theorem no_integer_solution_for_sum_of_cubes (n : ℤ) : 
  n % 9 = 4 → ¬∃ (x y z : ℤ), x^3 + y^3 + z^3 = n := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_sum_of_cubes_l969_96973


namespace NUMINAMATH_CALUDE_black_cars_count_l969_96918

theorem black_cars_count (total : ℕ) (blue_fraction red_fraction green_fraction : ℚ) :
  total = 1824 →
  blue_fraction = 2 / 5 →
  red_fraction = 1 / 3 →
  green_fraction = 1 / 8 →
  ∃ (blue red green black : ℕ),
    blue + red + green + black = total ∧
    blue = ⌊blue_fraction * total⌋ ∧
    red = red_fraction * total ∧
    green = green_fraction * total ∧
    black = 259 :=
by sorry

end NUMINAMATH_CALUDE_black_cars_count_l969_96918


namespace NUMINAMATH_CALUDE_schoolchildren_mushroom_picking_l969_96970

theorem schoolchildren_mushroom_picking (n : ℕ) 
  (h_max : ∃ (child : ℕ), child ≤ n ∧ child * 5 = n) 
  (h_min : ∃ (child : ℕ), child ≤ n ∧ child * 7 = n) : 
  5 < n ∧ n < 7 := by
  sorry

#check schoolchildren_mushroom_picking

end NUMINAMATH_CALUDE_schoolchildren_mushroom_picking_l969_96970


namespace NUMINAMATH_CALUDE_tails_appearance_l969_96921

/-- The number of coin flips -/
def total_flips : ℕ := 20

/-- The frequency of getting "heads" -/
def heads_frequency : ℚ := 45/100

/-- The number of times "tails" appears -/
def tails_count : ℕ := 11

/-- Theorem: Given a coin flipped 20 times with a frequency of getting "heads" of 0.45,
    the number of times "tails" appears is 11. -/
theorem tails_appearance :
  (total_flips : ℚ) * (1 - heads_frequency) = tails_count := by sorry

end NUMINAMATH_CALUDE_tails_appearance_l969_96921


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l969_96986

open Set

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (𝕌 \ A) ∩ B = Ioc 2 3 :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l969_96986


namespace NUMINAMATH_CALUDE_parabola_intersects_line_segment_l969_96994

/-- Parabola C_m: y = x^2 - mx + m + 1 -/
def C_m (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m + 1

/-- Line segment AB with endpoints A(0,4) and B(4,0) -/
def line_AB (x : ℝ) : ℝ := -x + 4

/-- The parabola C_m intersects the line segment AB at exactly two points
    if and only if m is in the range [3, 17/3] -/
theorem parabola_intersects_line_segment (m : ℝ) :
  (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 ∧
   C_m m x₁ = line_AB x₁ ∧ C_m m x₂ = line_AB x₂ ∧
   ∀ x, 0 ≤ x ∧ x ≤ 4 → C_m m x = line_AB x → (x = x₁ ∨ x = x₂)) ↔
  (3 ≤ m ∧ m ≤ 17/3) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersects_line_segment_l969_96994


namespace NUMINAMATH_CALUDE_problem_statement_l969_96926

theorem problem_statement (p q : ℝ) (h : p^2 / q^3 = 4 / 5) :
  11/7 + (2*q^3 - p^2) / (2*q^3 + p^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l969_96926


namespace NUMINAMATH_CALUDE_payment_plan_difference_l969_96971

theorem payment_plan_difference (original_price down_payment num_payments payment_amount : ℕ) :
  original_price = 1500 ∧
  down_payment = 200 ∧
  num_payments = 24 ∧
  payment_amount = 65 →
  (down_payment + num_payments * payment_amount) - original_price = 260 := by
  sorry

end NUMINAMATH_CALUDE_payment_plan_difference_l969_96971


namespace NUMINAMATH_CALUDE_sum_always_negative_l969_96940

/-- The function f(x) = -x - x^3 -/
def f (x : ℝ) : ℝ := -x - x^3

/-- Theorem stating that f(α) + f(β) + f(γ) is always negative under given conditions -/
theorem sum_always_negative (α β γ : ℝ) 
  (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) : 
  f α + f β + f γ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_always_negative_l969_96940


namespace NUMINAMATH_CALUDE_investment_timing_l969_96956

/-- Proves that B invested after 6 months given the conditions of the investment problem -/
theorem investment_timing (a_investment : ℕ) (b_investment : ℕ) (total_profit : ℕ) (a_profit : ℕ) :
  a_investment = 150 →
  b_investment = 200 →
  total_profit = 100 →
  a_profit = 60 →
  ∃ x : ℕ,
    x = 6 ∧
    (a_investment * 12 : ℚ) / (b_investment * (12 - x)) = (a_profit : ℚ) / (total_profit - a_profit) :=
by
  sorry


end NUMINAMATH_CALUDE_investment_timing_l969_96956


namespace NUMINAMATH_CALUDE_candy_chocolate_difference_l969_96916

theorem candy_chocolate_difference (initial_candy : ℕ) (additional_candy : ℕ) (chocolate : ℕ) :
  initial_candy = 38 →
  additional_candy = 36 →
  chocolate = 16 →
  (initial_candy + additional_candy) - chocolate = 58 := by
  sorry

end NUMINAMATH_CALUDE_candy_chocolate_difference_l969_96916


namespace NUMINAMATH_CALUDE_decryption_theorem_l969_96987

/-- Represents an encrypted text --/
def EncryptedText := String

/-- Represents a decrypted message --/
def DecryptedMessage := String

/-- The encryption method used for the word "МОСКВА" --/
def moscowEncryption (s : String) : EncryptedText :=
  sorry

/-- The decryption method for the given encryption --/
def decrypt (s : EncryptedText) : DecryptedMessage :=
  sorry

/-- Checks if two encrypted texts correspond to the same message --/
def sameMessage (t1 t2 : EncryptedText) : Prop :=
  decrypt t1 = decrypt t2

theorem decryption_theorem 
  (text1 text2 text3 : EncryptedText)
  (h1 : moscowEncryption "МОСКВА" = "ЙМЫВОТСЬЛКЪГВЦАЯЯ")
  (h2 : moscowEncryption "МОСКВА" = "УКМАПОЧСРКЩВЗАХ")
  (h3 : moscowEncryption "МОСКВА" = "ШМФЭОГЧСЙЪКФЬВЫЕАКК")
  (h4 : text1 = "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ")
  (h5 : text2 = "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП")
  (h6 : text3 = "РТПАИОМВСВТИЕОБПРОЕННИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК")
  (h7 : sameMessage text1 text3 ∨ sameMessage text1 text2 ∨ sameMessage text2 text3)
  : decrypt text1 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧ 
    decrypt text3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧
    decrypt text2 = "СМОТРИВКОРЕНЬ" :=
  sorry

end NUMINAMATH_CALUDE_decryption_theorem_l969_96987


namespace NUMINAMATH_CALUDE_gcf_of_36_and_54_l969_96967

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_and_54_l969_96967


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_arithmetic_sequence_length_is_8_l969_96997

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ → Prop :=
  λ aₙ ↦ aₙ = a₁ + (n - 1 : ℤ) * d

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 20 (-3) n (-2) ∧ 
  ∀ m : ℕ, m > n → ¬arithmetic_sequence 20 (-3) m (-2) := by
  sorry

theorem arithmetic_sequence_length_is_8 :
  ∃! n : ℕ, n > 0 ∧ arithmetic_sequence 20 (-3) n (-2) ∧ 
  ∀ m : ℕ, m > n → ¬arithmetic_sequence 20 (-3) m (-2) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_arithmetic_sequence_length_is_8_l969_96997


namespace NUMINAMATH_CALUDE_gcd_7920_14553_l969_96991

theorem gcd_7920_14553 : Nat.gcd 7920 14553 = 11 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7920_14553_l969_96991


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l969_96951

/-- Represents a cube structure made of unit cubes -/
structure CubeStructure where
  side_length : ℕ
  removed_cubes : ℕ

/-- Calculates the volume of the cube structure -/
def volume (c : CubeStructure) : ℕ :=
  c.side_length^3 - c.removed_cubes

/-- Calculates the surface area of the cube structure -/
def surface_area (c : CubeStructure) : ℕ :=
  6 * c.side_length^2 - 4 * c.removed_cubes

/-- The specific cube structure described in the problem -/
def hollow_cube : CubeStructure :=
  { side_length := 3
  , removed_cubes := 1 }

/-- Theorem stating the ratio of volume to surface area for the hollow cube -/
theorem volume_to_surface_area_ratio :
  (volume hollow_cube : ℚ) / (surface_area hollow_cube : ℚ) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l969_96951
