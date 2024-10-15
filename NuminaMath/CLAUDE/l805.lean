import Mathlib

namespace NUMINAMATH_CALUDE_tv_cash_savings_l805_80586

/-- Calculates the savings when buying a television with cash instead of an installment plan. -/
theorem tv_cash_savings (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (num_months : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  num_months = 12 →
  down_payment + monthly_payment * num_months - cash_price = 80 := by
sorry

end NUMINAMATH_CALUDE_tv_cash_savings_l805_80586


namespace NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l805_80542

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (3 / 4) / (5 / 8) = 6 / 5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l805_80542


namespace NUMINAMATH_CALUDE_percentage_difference_l805_80522

theorem percentage_difference : 
  (0.80 * 170 : ℝ) - (0.35 * 300 : ℝ) = 31 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l805_80522


namespace NUMINAMATH_CALUDE_f_inequality_part1_f_inequality_part2_l805_80513

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

theorem f_inequality_part1 :
  ∀ x : ℝ, f 1 x > 1 ↔ x > 1/2 := by sorry

theorem f_inequality_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) ↔ a ∈ Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_part1_f_inequality_part2_l805_80513


namespace NUMINAMATH_CALUDE_production_days_l805_80564

theorem production_days (n : ℕ) 
  (h1 : (40 * n) / n = 40)  -- Average daily production for past n days
  (h2 : ((40 * n + 90) : ℝ) / (n + 1) = 45) : n = 9 :=
by sorry

end NUMINAMATH_CALUDE_production_days_l805_80564


namespace NUMINAMATH_CALUDE_hyperbola_sum_l805_80529

/-- Given a hyperbola with center (-3, 1), one focus at (2, 1), and one vertex at (-1, 1),
    prove that h + k + a + b = 0 + √21, where (h, k) is the center, a is the distance from
    the center to the vertex, and b^2 = c^2 - a^2 with c being the distance from the center
    to the focus. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 →
  k = 1 →
  (2 : ℝ) - h = c →
  (-1 : ℝ) - h = a →
  b^2 = c^2 - a^2 →
  h + k + a + b = 0 + Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_sum_l805_80529


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l805_80585

theorem complex_fraction_sum : 
  (481 + 1/6 : ℚ) + (265 + 1/12 : ℚ) + (904 + 1/20 : ℚ) - 
  (184 + 29/30 : ℚ) - (160 + 41/42 : ℚ) - (703 + 55/56 : ℚ) = 
  603 + 3/8 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l805_80585


namespace NUMINAMATH_CALUDE_limit_f_at_zero_l805_80547

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 2) - Real.sqrt 2) / Real.sin (3 * x)

theorem limit_f_at_zero : 
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds ((Real.sqrt 2) / 24)) :=
sorry

end NUMINAMATH_CALUDE_limit_f_at_zero_l805_80547


namespace NUMINAMATH_CALUDE_shelves_filled_with_carvings_l805_80544

def wood_carvings_per_shelf : ℕ := 8
def total_wood_carvings : ℕ := 56

theorem shelves_filled_with_carvings :
  total_wood_carvings / wood_carvings_per_shelf = 7 := by
  sorry

end NUMINAMATH_CALUDE_shelves_filled_with_carvings_l805_80544


namespace NUMINAMATH_CALUDE_john_chocolate_gain_l805_80584

/-- Represents the chocolate types --/
inductive ChocolateType
  | A
  | B
  | C

/-- Represents a purchase of chocolates --/
structure Purchase where
  chocolateType : ChocolateType
  quantity : ℕ
  costPrice : ℚ

/-- Represents a sale of chocolates --/
structure Sale where
  chocolateType : ChocolateType
  quantity : ℕ
  sellingPrice : ℚ

def purchases : List Purchase := [
  ⟨ChocolateType.A, 100, 2⟩,
  ⟨ChocolateType.B, 150, 3⟩,
  ⟨ChocolateType.C, 200, 4⟩
]

def sales : List Sale := [
  ⟨ChocolateType.A, 90, 5/2⟩,
  ⟨ChocolateType.A, 60, 3⟩,
  ⟨ChocolateType.B, 140, 7/2⟩,
  ⟨ChocolateType.B, 10, 4⟩,
  ⟨ChocolateType.B, 50, 5⟩,
  ⟨ChocolateType.C, 180, 9/2⟩,
  ⟨ChocolateType.C, 20, 5⟩
]

def totalCostPrice : ℚ :=
  purchases.foldr (fun p acc => acc + p.quantity * p.costPrice) 0

def totalSellingPrice : ℚ :=
  sales.foldr (fun s acc => acc + s.quantity * s.sellingPrice) 0

def gainPercentage : ℚ :=
  ((totalSellingPrice - totalCostPrice) / totalCostPrice) * 100

theorem john_chocolate_gain :
  gainPercentage = 89/2 := by sorry

end NUMINAMATH_CALUDE_john_chocolate_gain_l805_80584


namespace NUMINAMATH_CALUDE_well_depth_l805_80559

/-- Proves that a circular well with diameter 4 meters and volume 301.59289474462014 cubic meters has a depth of 24 meters. -/
theorem well_depth (diameter : Real) (volume : Real) (depth : Real) :
  diameter = 4 →
  volume = 301.59289474462014 →
  depth = volume / (π * (diameter / 2)^2) →
  depth = 24 := by
  sorry

end NUMINAMATH_CALUDE_well_depth_l805_80559


namespace NUMINAMATH_CALUDE_sequence_perfect_squares_l805_80519

theorem sequence_perfect_squares (a b : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) = 7 * a n + 6 * b n - 3) →
  (∀ n : ℕ, b (n + 1) = 8 * a n + 7 * b n - 4) →
  ∃ A : ℕ → ℤ, ∀ n : ℕ, a n = (A n)^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_perfect_squares_l805_80519


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l805_80567

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 ∧ (∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y) →
  x = -4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l805_80567


namespace NUMINAMATH_CALUDE_min_rubber_bands_specific_l805_80505

/-- Calculates the minimum number of rubber bands needed to tie matches and cotton swabs into bundles. -/
def min_rubber_bands (total_matches : ℕ) (total_swabs : ℕ) (matches_per_bundle : ℕ) (swabs_per_bundle : ℕ) (bands_per_bundle : ℕ) : ℕ :=
  let match_bundles := total_matches / matches_per_bundle
  let swab_bundles := total_swabs / swabs_per_bundle
  (match_bundles + swab_bundles) * bands_per_bundle

/-- Theorem stating that given the specific conditions, the minimum number of rubber bands needed is 14. -/
theorem min_rubber_bands_specific : 
  min_rubber_bands 40 34 8 12 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_rubber_bands_specific_l805_80505


namespace NUMINAMATH_CALUDE_coal_pile_remaining_l805_80563

theorem coal_pile_remaining (total : ℝ) (used : ℝ) (remaining : ℝ) : 
  used = (4 : ℝ) / 10 * total → remaining = (6 : ℝ) / 10 * total :=
by
  sorry

end NUMINAMATH_CALUDE_coal_pile_remaining_l805_80563


namespace NUMINAMATH_CALUDE_orthogonal_projection_locus_l805_80518

/-- Given a line (x/a) + (y/b) = 1 where (1/a^2) + (1/b^2) = 1/c^2 (c constant),
    the orthogonal projection of the origin on this line always lies on the circle x^2 + y^2 = c^2 -/
theorem orthogonal_projection_locus (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c > 0) :
  (1 / a^2 + 1 / b^2 = 1 / c^2) →
  ∃ (x y : ℝ), (x / a + y / b = 1) ∧ 
               (y = (a / b) * x) ∧
               (x^2 + y^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_projection_locus_l805_80518


namespace NUMINAMATH_CALUDE_linear_function_identification_l805_80504

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_identification :
  let f₁ : ℝ → ℝ := λ x ↦ x^3
  let f₂ : ℝ → ℝ := λ x ↦ -2*x + 1
  let f₃ : ℝ → ℝ := λ x ↦ 2/x
  let f₄ : ℝ → ℝ := λ x ↦ 2*x^2 + 1
  is_linear f₂ ∧ ¬is_linear f₁ ∧ ¬is_linear f₃ ∧ ¬is_linear f₄ :=
by sorry

end NUMINAMATH_CALUDE_linear_function_identification_l805_80504


namespace NUMINAMATH_CALUDE_total_count_formula_specific_case_l805_80541

/-- Represents the structure of a plant with branches and small branches -/
structure Plant where
  branches : ℕ
  smallBranches : ℕ

/-- Calculates the total number of stems, branches, and small branches in a plant -/
def totalCount (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranches

/-- Theorem stating that the total count equals x^2 + x + 1 -/
theorem total_count_formula (x : ℕ) :
  totalCount { branches := x, smallBranches := x } = x^2 + x + 1 := by
  sorry

/-- The specific case where the total count is 73 -/
theorem specific_case : ∃ x : ℕ, totalCount { branches := x, smallBranches := x } = 73 := by
  sorry

end NUMINAMATH_CALUDE_total_count_formula_specific_case_l805_80541


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l805_80568

/-- A quadratic equation x^2 + bx + 16 has at least one real root if and only if b ∈ (-∞,-8] ∪ [8,∞) -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 16 = 0) ↔ b ≤ -8 ∨ b ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l805_80568


namespace NUMINAMATH_CALUDE_repaved_total_correct_l805_80520

/-- The total inches of road repaved by a construction company -/
def total_repaved (before_today : ℕ) (today : ℕ) : ℕ :=
  before_today + today

/-- Theorem stating that the total inches repaved is 4938 -/
theorem repaved_total_correct : total_repaved 4133 805 = 4938 := by
  sorry

end NUMINAMATH_CALUDE_repaved_total_correct_l805_80520


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l805_80515

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) : 
  (∀ x > 0, 2 * x * log x ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l805_80515


namespace NUMINAMATH_CALUDE_area_at_stage_8_l805_80527

/-- The area of a rectangle formed by adding squares -/
def rectangleArea (numSquares : ℕ) (squareSide : ℕ) : ℕ :=
  numSquares * (squareSide * squareSide)

/-- Theorem: The area of a rectangle formed by adding 8 squares, each 4 inches by 4 inches, is 128 square inches -/
theorem area_at_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l805_80527


namespace NUMINAMATH_CALUDE_equation_substitution_l805_80506

theorem equation_substitution :
  let eq1 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y - 2
  let eq2 : ℝ → ℝ := λ y => 2 * y - 1
  ∀ y : ℝ, eq1 (eq2 y) y = 3 * (2 * y - 1) - 4 * y - 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_substitution_l805_80506


namespace NUMINAMATH_CALUDE_some_multiplier_value_l805_80526

theorem some_multiplier_value : ∃ (some_multiplier : ℤ), 
  |5 - some_multiplier * (3 - 12)| - |5 - 11| = 71 ∧ some_multiplier = 8 := by
  sorry

end NUMINAMATH_CALUDE_some_multiplier_value_l805_80526


namespace NUMINAMATH_CALUDE_reflection_property_l805_80599

/-- A reflection in 2D space -/
structure Reflection2D where
  /-- The function that performs the reflection -/
  reflect : Fin 2 → ℝ → Fin 2 → ℝ

/-- Theorem: If a reflection takes (2, -3) to (8, 1), then it takes (1, 4) to (-18/13, -50/13) -/
theorem reflection_property (r : Reflection2D) 
  (h1 : r.reflect 0 2 = 8) 
  (h2 : r.reflect 1 (-3) = 1) 
  : r.reflect 0 1 = -18/13 ∧ r.reflect 1 4 = -50/13 := by
  sorry


end NUMINAMATH_CALUDE_reflection_property_l805_80599


namespace NUMINAMATH_CALUDE_digit_zero_equality_l805_80501

-- Define a function to count digits in a number
def countDigits (n : ℕ) : ℕ := sorry

-- Define a function to count zeros in a number
def countZeros (n : ℕ) : ℕ := sorry

-- Define a function to sum the count of digits in a sequence
def sumDigits (n : ℕ) : ℕ := sorry

-- Define a function to sum the count of zeros in a sequence
def sumZeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_zero_equality : sumDigits (10^8) = sumZeros (10^9) := by sorry

end NUMINAMATH_CALUDE_digit_zero_equality_l805_80501


namespace NUMINAMATH_CALUDE_toothpicks_300th_stage_l805_80576

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 5 + (n - 1) * 4

/-- Theorem: The number of toothpicks in the 300th stage is 1201 -/
theorem toothpicks_300th_stage :
  toothpicks 300 = 1201 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_300th_stage_l805_80576


namespace NUMINAMATH_CALUDE_magic_sum_divisible_by_three_l805_80596

/-- Represents a 3x3 magic square with integer entries -/
def MagicSquare : Type := Matrix (Fin 3) (Fin 3) ℤ

/-- The sum of each row, column, and diagonal in a magic square -/
def magicSum (m : MagicSquare) : ℤ :=
  m 0 0 + m 0 1 + m 0 2

/-- Predicate to check if a given matrix is a magic square -/
def isMagicSquare (m : MagicSquare) : Prop :=
  (∀ i : Fin 3, (m i 0 + m i 1 + m i 2 = magicSum m)) ∧ 
  (∀ j : Fin 3, (m 0 j + m 1 j + m 2 j = magicSum m)) ∧ 
  (m 0 0 + m 1 1 + m 2 2 = magicSum m) ∧
  (m 0 2 + m 1 1 + m 2 0 = magicSum m)

/-- Theorem: The magic sum of a 3x3 magic square is divisible by 3 -/
theorem magic_sum_divisible_by_three (m : MagicSquare) (h : isMagicSquare m) :
  ∃ k : ℤ, magicSum m = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_magic_sum_divisible_by_three_l805_80596


namespace NUMINAMATH_CALUDE_square_count_3x3_and_5x5_l805_80510

/-- Represents a square grid with uniform distance between consecutive dots -/
structure UniformSquareGrid (n : ℕ) :=
  (size : ℕ)
  (uniform_distance : Bool)

/-- Counts the number of squares with all 4 vertices on the dots in a grid -/
def count_squares (grid : UniformSquareGrid n) : ℕ :=
  sorry

theorem square_count_3x3_and_5x5 :
  ∀ (grid3 : UniformSquareGrid 3) (grid5 : UniformSquareGrid 5),
    grid3.size = 3 ∧ grid3.uniform_distance = true →
    grid5.size = 5 ∧ grid5.uniform_distance = true →
    count_squares grid3 = 4 ∧ count_squares grid5 = 50 :=
by sorry

end NUMINAMATH_CALUDE_square_count_3x3_and_5x5_l805_80510


namespace NUMINAMATH_CALUDE_ferris_wheel_ticket_cost_l805_80577

theorem ferris_wheel_ticket_cost 
  (initial_tickets : ℕ) 
  (remaining_tickets : ℕ) 
  (total_spent : ℕ) 
  (h1 : initial_tickets = 6)
  (h2 : remaining_tickets = 3)
  (h3 : total_spent = 27) :
  total_spent / (initial_tickets - remaining_tickets) = 9 :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_ticket_cost_l805_80577


namespace NUMINAMATH_CALUDE_orange_count_l805_80580

/-- The number of oranges initially in the bin -/
def initial_oranges : ℕ := sorry

/-- The number of oranges thrown away -/
def thrown_away : ℕ := 20

/-- The number of new oranges added -/
def new_oranges : ℕ := 13

/-- The final number of oranges in the bin -/
def final_oranges : ℕ := 27

theorem orange_count : initial_oranges = 34 :=
  by sorry

end NUMINAMATH_CALUDE_orange_count_l805_80580


namespace NUMINAMATH_CALUDE_number_division_problem_l805_80549

theorem number_division_problem :
  let sum := 3927 + 2873
  let diff := 3927 - 2873
  let quotient := 3 * diff
  ∀ (N r : ℕ), 
    N / sum = quotient ∧ 
    N % sum = r ∧ 
    r < sum →
    N = 21481600 + r :=
by sorry

end NUMINAMATH_CALUDE_number_division_problem_l805_80549


namespace NUMINAMATH_CALUDE_manolo_total_masks_l805_80566

/-- Represents the number of face-masks Manolo can make in a given time period -/
def masks_made (rate : ℕ) (duration : ℕ) : ℕ :=
  (duration * 60) / rate

/-- Represents Manolo's six-hour shift face-mask production -/
def manolo_shift_production : ℕ :=
  masks_made 4 1 + masks_made 6 2 + masks_made 8 2

theorem manolo_total_masks :
  manolo_shift_production = 50 := by
  sorry

end NUMINAMATH_CALUDE_manolo_total_masks_l805_80566


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l805_80578

theorem complex_fraction_equality : (1 + Complex.I * Real.sqrt 3) ^ 2 / (Complex.I * Real.sqrt 3 - 1) = -2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l805_80578


namespace NUMINAMATH_CALUDE_power_simplification_l805_80511

theorem power_simplification :
  (10^0.6) * (10^0.4) * (10^0.4) * (10^0.1) * (10^0.5) / (10^0.3) = 10^1.7 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l805_80511


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l805_80540

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (2*x - y)^2 - 4*(x - y)*(x + 2*y) = -8*x*y + 9*y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b c : ℝ) :
  (a - 2*b - 3*c)*(a - 2*b + 3*c) = a^2 + 4*b^2 - 4*a*b - 9*c^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l805_80540


namespace NUMINAMATH_CALUDE_solution_comparison_l805_80581

theorem solution_comparison (a a' b b' k : ℝ) 
  (ha : a ≠ 0) (ha' : a' ≠ 0) (hk : k > 0) :
  (-kb / a < -b' / a') ↔ (k * b * a' > a * b') :=
sorry

end NUMINAMATH_CALUDE_solution_comparison_l805_80581


namespace NUMINAMATH_CALUDE_domain_of_g_l805_80583

-- Define the domain of f
def DomainF : Set ℝ := Set.Icc (-8) 4

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-2 * x)

-- Theorem statement
theorem domain_of_g (f : ℝ → ℝ) (hf : Set.MapsTo f DomainF (Set.range f)) :
  {x : ℝ | g f x ∈ Set.range f} = Set.Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l805_80583


namespace NUMINAMATH_CALUDE_complex_equation_solution_l805_80538

theorem complex_equation_solution :
  ∃ (z : ℂ), 5 + 2 * I * z = 3 - 5 * I * z ∧ z = (2 * I) / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l805_80538


namespace NUMINAMATH_CALUDE_number_equality_l805_80554

theorem number_equality (x : ℝ) (h : 0.15 * x = 0.25 * 16 + 2) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l805_80554


namespace NUMINAMATH_CALUDE_determine_friendship_graph_l805_80514

/-- Represents the friendship graph among apprentices -/
def FriendshipGraph := Fin 10 → Fin 10 → Prop

/-- Represents a duty assignment for a single day -/
def DutyAssignment := Fin 10 → Bool

/-- Calculates the number of missing pastries for a given duty assignment and friendship graph -/
def missingPastries (duty : DutyAssignment) (friends : FriendshipGraph) : ℕ :=
  sorry

/-- Theorem: The chef can determine the friendship graph after 45 days -/
theorem determine_friendship_graph 
  (friends : FriendshipGraph) :
  ∃ (assignments : Fin 45 → DutyAssignment),
    ∀ (other_friends : FriendshipGraph),
      (∀ (day : Fin 45), missingPastries (assignments day) friends = 
                          missingPastries (assignments day) other_friends) →
      friends = other_friends :=
sorry

end NUMINAMATH_CALUDE_determine_friendship_graph_l805_80514


namespace NUMINAMATH_CALUDE_recipe_batches_for_competition_l805_80509

/-- Calculates the number of full recipe batches needed for a math competition --/
def recipe_batches_needed (total_students : ℕ) (attendance_drop : ℚ) 
  (cookies_per_student : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  let attending_students := (total_students : ℚ) * (1 - attendance_drop)
  let total_cookies_needed := (attending_students * cookies_per_student : ℚ).ceil
  let batches_needed := (total_cookies_needed / cookies_per_batch : ℚ).ceil
  batches_needed.toNat

/-- Proves that 17 full recipe batches are needed for the math competition --/
theorem recipe_batches_for_competition : 
  recipe_batches_needed 144 (30/100) 3 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_recipe_batches_for_competition_l805_80509


namespace NUMINAMATH_CALUDE_problem_solution_l805_80587

theorem problem_solution :
  let x : ℝ := 88 * (1 + 0.25)
  let y : ℝ := 150 * (1 - 0.40)
  let z : ℝ := 60 * (1 + 0.15)
  (x + y + z = 269) ∧
  ((x * y * z) ^ (x - y) = (683100 : ℝ) ^ 20) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l805_80587


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_ratio_l805_80534

theorem inscribed_triangle_area_ratio (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let triangle_base := s
  let triangle_height := s / 2
  let triangle_area := (triangle_base * triangle_height) / 2
  triangle_area / square_area = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_ratio_l805_80534


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l805_80521

/-- The volume of a rectangular parallelepiped with given conditions -/
theorem rectangular_parallelepiped_volume :
  ∀ (length width height : ℝ),
  length > 0 →
  width > 0 →
  height > 0 →
  length = width →
  2 * (length + width) = 32 →
  height = 9 →
  length * width * height = 576 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l805_80521


namespace NUMINAMATH_CALUDE_point_q_coordinates_l805_80552

/-- Given two points P and Q in a 2D Cartesian coordinate system, prove that Q has coordinates (1, -3) -/
theorem point_q_coordinates
  (P Q : ℝ × ℝ) -- P and Q are points in 2D space
  (h_P : P = (1, 2)) -- P has coordinates (1, 2)
  (h_Q_below : Q.2 < 0) -- Q is below the x-axis
  (h_parallel : P.1 = Q.1) -- PQ is parallel to the y-axis
  (h_distance : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 5) -- PQ = 5
  : Q = (1, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_q_coordinates_l805_80552


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l805_80503

theorem perpendicular_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, m - 4]
  (∀ i, i < 2 → a i * b i = 0) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l805_80503


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l805_80531

/-- Given an arithmetic sequence {a_n} where a_4 + a_8 = 16, prove that a_2 + a_6 + a_10 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → -- arithmetic sequence condition
  (a 4 + a 8 = 16) →                               -- given condition
  (a 2 + a 6 + a 10 = 24) :=                       -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l805_80531


namespace NUMINAMATH_CALUDE_intersection_condition_l805_80546

/-- The set M in ℝ² -/
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}

/-- The set N in ℝ² parameterized by a -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- The theorem stating the necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l805_80546


namespace NUMINAMATH_CALUDE_smallest_number_l805_80553

theorem smallest_number (S : Set ℕ) (h : S = {5, 8, 3, 2, 6}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l805_80553


namespace NUMINAMATH_CALUDE_rectangle_length_l805_80582

/-- Given a rectangle with width 16 cm and perimeter 70 cm, prove its length is 19 cm. -/
theorem rectangle_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 16 → 
  perimeter = 70 → 
  perimeter = 2 * (length + width) → 
  length = 19 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l805_80582


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l805_80539

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle

/-- The y-intercept of a line -/
def yIntercept (line : TangentLine) : ℝ := sorry

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (8, 0), radius := 2 }
  let line : TangentLine := { circle1 := c1, circle2 := c2 }
  yIntercept line = 2 * Real.sqrt 82 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l805_80539


namespace NUMINAMATH_CALUDE_divisibility_of_10_pow_6_minus_1_l805_80523

theorem divisibility_of_10_pow_6_minus_1 :
  ∃ (a b c d : ℕ), 10^6 - 1 = 7 * a ∧ 10^6 - 1 = 13 * b ∧ 10^6 - 1 = 91 * c ∧ 10^6 - 1 = 819 * d :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_10_pow_6_minus_1_l805_80523


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l805_80516

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the dimensions of a corner cube to be removed -/
structure CornerCubeDimensions where
  side : ℝ

/-- Theorem stating that removing corner cubes does not change the surface area -/
theorem surface_area_unchanged 
  (original : CubeDimensions) 
  (corner : CornerCubeDimensions) 
  (h1 : original.length = original.width ∧ original.width = original.height)
  (h2 : original.length = 5)
  (h3 : corner.side = 2) : 
  surfaceArea original = surfaceArea original := by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l805_80516


namespace NUMINAMATH_CALUDE_quadratic_sequence_problem_l805_80524

theorem quadratic_sequence_problem (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 2)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 15)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 52) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_problem_l805_80524


namespace NUMINAMATH_CALUDE_chips_sold_in_month_l805_80555

theorem chips_sold_in_month (week1 : ℕ) (week2 : ℕ) (week3 : ℕ) (week4 : ℕ) 
  (h1 : week1 = 15)
  (h2 : week2 = 3 * week1)
  (h3 : week3 = 20)
  (h4 : week4 = 20) :
  week1 + week2 + week3 + week4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_chips_sold_in_month_l805_80555


namespace NUMINAMATH_CALUDE_square_sum_fraction_difference_l805_80591

theorem square_sum_fraction_difference : 
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_fraction_difference_l805_80591


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_zero_sqrt_product_division_equals_three_sqrt_two_over_two_l805_80562

-- Problem 1
theorem sqrt_expression_equals_zero :
  Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 := by sorry

-- Problem 2
theorem sqrt_product_division_equals_three_sqrt_two_over_two :
  Real.sqrt 12 * (Real.sqrt 3 / 2) / Real.sqrt 2 = 3 * Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_zero_sqrt_product_division_equals_three_sqrt_two_over_two_l805_80562


namespace NUMINAMATH_CALUDE_patrons_in_cars_patrons_in_cars_is_twelve_l805_80573

/-- The number of patrons who came in cars to a golf tournament -/
theorem patrons_in_cars (num_carts : ℕ) (cart_capacity : ℕ) (bus_patrons : ℕ) : ℕ :=
  num_carts * cart_capacity - bus_patrons

/-- Proof that the number of patrons who came in cars is 12 -/
theorem patrons_in_cars_is_twelve : patrons_in_cars 13 3 27 = 12 := by
  sorry

end NUMINAMATH_CALUDE_patrons_in_cars_patrons_in_cars_is_twelve_l805_80573


namespace NUMINAMATH_CALUDE_child_ticket_cost_l805_80530

/-- Given information about ticket sales for a baseball game, prove the cost of a child ticket. -/
theorem child_ticket_cost
  (adult_ticket_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_ticket_price = 5)
  (h2 : total_tickets = 85)
  (h3 : total_revenue = 275)
  (h4 : adult_tickets = 35) :
  (total_revenue - adult_tickets * adult_ticket_price) / (total_tickets - adult_tickets) = 2 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l805_80530


namespace NUMINAMATH_CALUDE_no_valid_operation_l805_80572

-- Define the set of standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an arithmetic operation
def applyOp (op : ArithOp) (a b : Int) : Option Int :=
  match op with
  | ArithOp.Add => some (a + b)
  | ArithOp.Sub => some (a - b)
  | ArithOp.Mul => some (a * b)
  | ArithOp.Div => if b ≠ 0 then some (a / b) else none

-- Theorem statement
theorem no_valid_operation :
  ∀ (op : ArithOp), (applyOp op 7 4).map (λ x => x + 5 - (3 - 2)) ≠ some 4 := by
  sorry


end NUMINAMATH_CALUDE_no_valid_operation_l805_80572


namespace NUMINAMATH_CALUDE_radius_circle_q_is_ten_l805_80508

/-- A triangle ABC with two equal sides and a circle P tangent to two sides -/
structure IsoscelesTriangleWithTangentCircle where
  /-- The length of the equal sides AB and AC -/
  side_length : ℝ
  /-- The length of the base BC -/
  base_length : ℝ
  /-- The radius of the circle P tangent to AC and BC -/
  circle_p_radius : ℝ

/-- The radius of circle Q, which is externally tangent to P and tangent to AB and BC -/
def radius_circle_q (t : IsoscelesTriangleWithTangentCircle) : ℝ := sorry

/-- The main theorem: In the given configuration, the radius of circle Q is 10 -/
theorem radius_circle_q_is_ten
  (t : IsoscelesTriangleWithTangentCircle)
  (h1 : t.side_length = 120)
  (h2 : t.base_length = 90)
  (h3 : t.circle_p_radius = 30) :
  radius_circle_q t = 10 := by sorry

end NUMINAMATH_CALUDE_radius_circle_q_is_ten_l805_80508


namespace NUMINAMATH_CALUDE_sector_angle_l805_80545

/-- Given an arc length of 4 cm and a radius of 2 cm, the central angle of the sector in radians is 2. -/
theorem sector_angle (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 4) (h2 : radius = 2) :
  arc_length / radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l805_80545


namespace NUMINAMATH_CALUDE_sum_of_divisors_930_l805_80597

/-- Sum of positive divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem sum_of_divisors_930 (i j : ℕ+) :
  sum_of_divisors (2^i.val * 5^j.val) = 930 → i.val + j.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_930_l805_80597


namespace NUMINAMATH_CALUDE_range_of_m_l805_80537

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (m ≤ 0 ∨ m ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l805_80537


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l805_80561

theorem polygon_angle_sum (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 + (180 - ((n - 2) * 180) / n) = 1350 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l805_80561


namespace NUMINAMATH_CALUDE_max_value_three_ways_l805_80598

/-- A function representing the number of ways to draw balls with a specific maximum value -/
def num_ways_max_value (n : ℕ) (max_value : ℕ) (num_draws : ℕ) : ℕ :=
  sorry

/-- The number of balls in the box -/
def num_balls : ℕ := 3

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The maximum value we're interested in -/
def target_max : ℕ := 3

theorem max_value_three_ways :
  num_ways_max_value num_balls target_max num_draws = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_value_three_ways_l805_80598


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l805_80550

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem matrix_inverse_proof :
  A⁻¹ = !![9/46, -5/46; 2/46, 4/46] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l805_80550


namespace NUMINAMATH_CALUDE_triangle_side_range_l805_80560

theorem triangle_side_range (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = π / 4 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  (∃ (A' : ℝ), A' ≠ A ∧ 0 < A' ∧ A' < π ∧ a / Real.sin A' = b / Real.sin B) →
  2 < a ∧ a < 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l805_80560


namespace NUMINAMATH_CALUDE_sin_315_degrees_l805_80512

theorem sin_315_degrees : 
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l805_80512


namespace NUMINAMATH_CALUDE_division_of_decimals_l805_80569

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l805_80569


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l805_80551

/-- Proves the relationship between inverse proportionality and percentage changes -/
theorem inverse_proportion_percentage_change 
  (x y x' y' q k : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = k) 
  (h4 : x' * y' = k) 
  (h5 : x' = x * (1 - q / 100)) 
  (h6 : q > 0) 
  (h7 : q < 100) : 
  y' = y * (1 + (100 * q) / (100 - q) / 100) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l805_80551


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l805_80548

theorem modulus_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.abs (i / (2 - i)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l805_80548


namespace NUMINAMATH_CALUDE_rower_downstream_speed_l805_80533

/-- Calculates the downstream speed of a rower given their upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 48 kmph -/
theorem rower_downstream_speed :
  downstream_speed 32 40 = 48 := by
  sorry

end NUMINAMATH_CALUDE_rower_downstream_speed_l805_80533


namespace NUMINAMATH_CALUDE_johns_car_repair_cost_l805_80556

/-- Calculates the total cost of car repairs including sales tax -/
def total_repair_cost (engine_labor_rate : ℝ) (engine_labor_hours : ℝ) (engine_part_cost : ℝ)
                      (brake_labor_rate : ℝ) (brake_labor_hours : ℝ) (brake_part_cost : ℝ)
                      (tire_labor_rate : ℝ) (tire_labor_hours : ℝ) (tire_cost : ℝ)
                      (sales_tax_rate : ℝ) : ℝ :=
  let engine_cost := engine_labor_rate * engine_labor_hours + engine_part_cost
  let brake_cost := brake_labor_rate * brake_labor_hours + brake_part_cost
  let tire_cost := tire_labor_rate * tire_labor_hours + tire_cost
  let total_before_tax := engine_cost + brake_cost + tire_cost
  let tax_amount := sales_tax_rate * total_before_tax
  total_before_tax + tax_amount

/-- Theorem stating that the total repair cost for John's car is $5238 -/
theorem johns_car_repair_cost :
  total_repair_cost 75 16 1200 85 10 800 50 4 600 0.08 = 5238 := by
  sorry

end NUMINAMATH_CALUDE_johns_car_repair_cost_l805_80556


namespace NUMINAMATH_CALUDE_circle_sum_inequality_l805_80525

theorem circle_sum_inequality (nums : Fin 100 → ℝ) (h_distinct : Function.Injective nums) :
  ∃ (i : Fin 100), (nums i + nums ((i + 1) % 100)) < (nums ((i + 2) % 100) + nums ((i + 3) % 100)) :=
sorry

end NUMINAMATH_CALUDE_circle_sum_inequality_l805_80525


namespace NUMINAMATH_CALUDE_min_value_expression_l805_80558

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (3 * a / (b * c^2 + b)) + (1 / (a * b * c^2 + a * b)) + 3 * c^2 ≥ 6 * Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l805_80558


namespace NUMINAMATH_CALUDE_opposite_sides_of_line_l805_80595

def line_equation (x y : ℝ) : ℝ := 2 * x + y - 3

theorem opposite_sides_of_line :
  (line_equation 0 0 < 0) ∧ (line_equation 2 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_of_line_l805_80595


namespace NUMINAMATH_CALUDE_expected_games_is_correct_l805_80535

/-- Represents the state of the game --/
inductive GameState
| Ongoing : ℕ → ℕ → GameState  -- Number of wins for player A and B
| Finished : GameState

/-- The probability of player A winning in an odd-numbered game --/
def prob_A_odd : ℚ := 3/5

/-- The probability of player B winning in an even-numbered game --/
def prob_B_even : ℚ := 3/5

/-- Determines if the game is finished based on the number of wins --/
def is_finished (wins_A wins_B : ℕ) : Bool :=
  (wins_A ≥ wins_B + 2) ∨ (wins_B ≥ wins_A + 2)

/-- Calculates the expected number of games until the match ends --/
noncomputable def expected_games : ℚ :=
  25/6

/-- Theorem stating that the expected number of games is 25/6 --/
theorem expected_games_is_correct : expected_games = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_expected_games_is_correct_l805_80535


namespace NUMINAMATH_CALUDE_marcus_bird_count_l805_80528

theorem marcus_bird_count (humphrey_count darrel_count average_count : ℕ) 
  (h1 : humphrey_count = 11)
  (h2 : darrel_count = 9)
  (h3 : average_count = 9)
  (h4 : (humphrey_count + darrel_count + marcus_count) / 3 = average_count) :
  marcus_count = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_marcus_bird_count_l805_80528


namespace NUMINAMATH_CALUDE_point_coordinates_on_angle_l805_80570

theorem point_coordinates_on_angle (α : Real) (P : Real × Real) :
  α = π / 4 →
  (P.1^2 + P.2^2 = 2) →
  P = (1, 1) := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_on_angle_l805_80570


namespace NUMINAMATH_CALUDE_lindsey_squat_weight_l805_80574

/-- Calculates the total weight Lindsey will be squatting -/
def total_squat_weight (band_a : ℕ) (band_b : ℕ) (band_c : ℕ) 
                       (leg_weight : ℕ) (dumbbell : ℕ) : ℕ :=
  2 * (band_a + band_b + band_c) + 2 * leg_weight + dumbbell

/-- Proves that Lindsey's total squat weight is 65 pounds -/
theorem lindsey_squat_weight :
  total_squat_weight 7 5 3 10 15 = 65 :=
by sorry

end NUMINAMATH_CALUDE_lindsey_squat_weight_l805_80574


namespace NUMINAMATH_CALUDE_alphabet_sum_theorem_l805_80565

/-- Represents a letter in the English alphabet -/
def Letter := Fin 26

/-- Represents a sequence of 26 letters -/
def Sequence := Fin 26 → Letter

/-- The sum operation for letters -/
def letter_sum (a b : Letter) : Letter :=
  ⟨(a.val + b.val) % 26, by sorry⟩

/-- The sum operation for sequences -/
def sequence_sum (s1 s2 : Sequence) : Sequence :=
  λ i => letter_sum (s1 i) (s2 i)

/-- The standard alphabet sequence -/
def alphabet_sequence : Sequence :=
  λ i => i

/-- A permutation of the alphabet -/
def is_permutation (s : Sequence) : Prop :=
  Function.Injective s

theorem alphabet_sum_theorem (s : Sequence) (h : is_permutation s) :
  ∃ i j : Fin 26, i ≠ j ∧ sequence_sum s alphabet_sequence i = sequence_sum s alphabet_sequence j :=
sorry

end NUMINAMATH_CALUDE_alphabet_sum_theorem_l805_80565


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_20_18_l805_80579

theorem smallest_divisible_by_15_20_18 :
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 20 ∣ n ∧ 18 ∣ n ∧ ∀ m : ℕ, m > 0 → 15 ∣ m → 20 ∣ m → 18 ∣ m → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_20_18_l805_80579


namespace NUMINAMATH_CALUDE_problem_1_l805_80588

theorem problem_1 (x y : ℝ) : (-3 * x^2 * y)^2 * (2 * x * y^2) / (-6 * x^3 * y^4) = -3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l805_80588


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l805_80557

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  9 * x^2 - 27 * (floor x) + 22 = 0

/-- The theorem stating that the equation has exactly 4 real solutions -/
theorem equation_has_four_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, equation x ∧
  ∀ y : ℝ, equation y → y ∈ s :=
sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l805_80557


namespace NUMINAMATH_CALUDE_third_circle_radius_l805_80594

/-- Given two externally tangent circles and a third circle tangent to both and their center line, 
    prove that the radius of the third circle is √46 - 5 -/
theorem third_circle_radius 
  (P Q R : ℝ × ℝ) -- Centers of the three circles
  (r : ℝ) -- Radius of the third circle
  (h1 : dist P Q = 10) -- Distance between centers of first two circles
  (h2 : dist P R = 3 + r) -- Distance from P to R
  (h3 : dist Q R = 7 + r) -- Distance from Q to R
  (h4 : (R.1 - P.1) * (Q.1 - P.1) + (R.2 - P.2) * (Q.2 - P.2) = 0) -- R is on the perpendicular bisector of PQ
  : r = Real.sqrt 46 - 5 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l805_80594


namespace NUMINAMATH_CALUDE_sequence_sum_l805_80592

theorem sequence_sum (a : ℕ → ℤ) : 
  (∀ n : ℕ, a (n + 1) - a n = 2) → 
  a 1 = -5 → 
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| : ℤ) = 18 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l805_80592


namespace NUMINAMATH_CALUDE_theater_seats_l805_80590

theorem theater_seats : ∃ n : ℕ, n < 60 ∧ n % 9 = 5 ∧ n % 6 = 3 ∧ n = 41 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_l805_80590


namespace NUMINAMATH_CALUDE_zero_exponent_eq_one_l805_80500

theorem zero_exponent_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_eq_one_l805_80500


namespace NUMINAMATH_CALUDE_complex_unit_vector_l805_80571

theorem complex_unit_vector (z : ℂ) (h : z = 3 + 4*I) : z / Complex.abs z = 3/5 + 4/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_unit_vector_l805_80571


namespace NUMINAMATH_CALUDE_min_sum_of_product_l805_80543

theorem min_sum_of_product (a b : ℤ) (h : a * b = 72) : 
  ∀ (x y : ℤ), x * y = 72 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 72 ∧ a₀ + b₀ = -73 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l805_80543


namespace NUMINAMATH_CALUDE_barbara_typing_time_l805_80536

/-- Calculates the time needed to type a document given the original typing speed,
    speed decrease, and document length. -/
def typing_time (original_speed : ℕ) (speed_decrease : ℕ) (document_length : ℕ) : ℕ :=
  document_length / (original_speed - speed_decrease)

/-- Proves that given the specific conditions, the typing time is 20 minutes. -/
theorem barbara_typing_time :
  typing_time 212 40 3440 = 20 := by
  sorry

end NUMINAMATH_CALUDE_barbara_typing_time_l805_80536


namespace NUMINAMATH_CALUDE_investment_interest_rate_l805_80593

/-- Proves that given the specified investment conditions, the annual interest rate of the second certificate is 8% -/
theorem investment_interest_rate 
  (initial_investment : ℝ)
  (first_rate : ℝ)
  (second_rate : ℝ)
  (final_value : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_rate = 8)
  (h3 : final_value = 15612)
  (h4 : initial_investment * (1 + first_rate / 400) * (1 + second_rate / 400) = final_value) :
  second_rate = 8 := by
    sorry

#check investment_interest_rate

end NUMINAMATH_CALUDE_investment_interest_rate_l805_80593


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l805_80575

theorem complex_fraction_simplification :
  let z : ℂ := Complex.mk 3 8 / Complex.mk 1 (-4)
  (z.re = -29/17) ∧ (z.im = 20/17) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l805_80575


namespace NUMINAMATH_CALUDE_cube_diagonal_l805_80502

theorem cube_diagonal (V : ℝ) (A : ℝ) (s : ℝ) (d : ℝ) : 
  V = 384 → A = 384 → V = s^3 → A = 6 * s^2 → d = s * Real.sqrt 3 → d = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_l805_80502


namespace NUMINAMATH_CALUDE_nancy_folders_l805_80507

theorem nancy_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 80 → deleted_files = 31 → files_per_folder = 7 → 
  (initial_files - deleted_files) / files_per_folder = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_folders_l805_80507


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_zero_l805_80532

/-- Two real numbers are opposite if their sum is zero. -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- If a and b are opposite numbers, then their sum is zero. -/
theorem opposite_numbers_sum_zero (a b : ℝ) (h : are_opposite a b) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_zero_l805_80532


namespace NUMINAMATH_CALUDE_jules_starting_fee_is_two_l805_80517

/-- Calculates the starting fee per walk for Jules' dog walking service -/
def starting_fee_per_walk (total_vacation_cost : ℚ) (family_members : ℕ) 
  (price_per_block : ℚ) (dogs_walked : ℕ) (total_blocks : ℕ) : ℚ :=
  let individual_contribution := total_vacation_cost / family_members
  let earnings_from_blocks := price_per_block * total_blocks
  let total_starting_fees := individual_contribution - earnings_from_blocks
  total_starting_fees / dogs_walked

/-- Proves that Jules' starting fee per walk is $2 given the problem conditions -/
theorem jules_starting_fee_is_two :
  starting_fee_per_walk 1000 5 (5/4) 20 128 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jules_starting_fee_is_two_l805_80517


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l805_80589

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  eccentricity : ℝ
  real_axis_length : ℝ

/-- The equation of a hyperbola with given properties -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 12 = 1) ∨ (y^2 / 4 - x^2 / 12 = 1)

/-- Theorem stating that a hyperbola with the given properties has one of the two specified equations -/
theorem hyperbola_equation_theorem (h : Hyperbola) 
    (h_center : h.center = (0, 0))
    (h_eccentricity : h.eccentricity = 2)
    (h_real_axis : h.real_axis_length = 4) :
    ∀ x y : ℝ, hyperbola_equation h x y := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l805_80589
