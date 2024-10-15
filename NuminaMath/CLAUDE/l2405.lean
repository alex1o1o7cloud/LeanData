import Mathlib

namespace NUMINAMATH_CALUDE_ali_bookshelf_problem_l2405_240503

theorem ali_bookshelf_problem (x : ℕ) : 
  (x / 2 : ℕ) + (x / 3 : ℕ) + 3 + 7 = x → (x / 2 : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ali_bookshelf_problem_l2405_240503


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2405_240544

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 30)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 3) :
  badminton + tennis - total + neither = 9 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2405_240544


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l2405_240591

/-- The volume of a right circular cone formed by rolling up a five-sixth sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 6
  let base_radius : ℝ := sector_fraction * r
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let volume : ℝ := (1/3) * Real.pi * base_radius^2 * height
  volume = (25/3) * Real.pi * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l2405_240591


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l2405_240582

-- Define a type for planes
variable {P : Type*}

-- Define a relation for parallelism between planes
variable (parallel : P → P → Prop)

-- Define a relation for perpendicularity between a plane and a line
variable (perpendicular : P → P → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel to each other
theorem planes_parallel_to_same_plane_are_parallel 
  (p1 p2 p3 : P) 
  (h1 : parallel p1 p3) 
  (h2 : parallel p2 p3) : 
  parallel p1 p2 := by sorry

-- Theorem 2: Two planes perpendicular to the same line are parallel to each other
theorem planes_perpendicular_to_same_line_are_parallel 
  (p1 p2 l : P) 
  (h1 : perpendicular p1 l) 
  (h2 : perpendicular p2 l) : 
  parallel p1 p2 := by sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l2405_240582


namespace NUMINAMATH_CALUDE_promotion_savings_difference_l2405_240542

/-- Represents a promotion for sweater purchases -/
structure Promotion where
  first_sweater_price : ℝ
  second_sweater_discount : ℝ

/-- Calculates the total cost of two sweaters under a given promotion -/
def total_cost (p : Promotion) (original_price : ℝ) : ℝ :=
  p.first_sweater_price + (original_price - p.second_sweater_discount)

theorem promotion_savings_difference :
  let original_price : ℝ := 50
  let promotion_x : Promotion := { first_sweater_price := original_price, second_sweater_discount := 0.4 * original_price }
  let promotion_y : Promotion := { first_sweater_price := original_price, second_sweater_discount := 15 }
  total_cost promotion_y original_price - total_cost promotion_x original_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_promotion_savings_difference_l2405_240542


namespace NUMINAMATH_CALUDE_segment_length_l2405_240572

/-- The length of a segment with endpoints (1,1) and (8,17) is √305 -/
theorem segment_length : Real.sqrt ((8 - 1)^2 + (17 - 1)^2) = Real.sqrt 305 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l2405_240572


namespace NUMINAMATH_CALUDE_integer_root_quadratic_l2405_240525

theorem integer_root_quadratic (m n : ℕ+) : 
  (∃ x : ℕ+, x^2 - (m.val * n.val) * x + (m.val + n.val) = 0) ↔ 
  ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 2) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_quadratic_l2405_240525


namespace NUMINAMATH_CALUDE_derivative_at_one_l2405_240593

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

theorem derivative_at_one :
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2405_240593


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2405_240580

theorem intersection_point_of_lines (x y : ℚ) :
  (8 * x - 5 * y = 40) ∧ (10 * x + 2 * y = 14) ↔ x = 25/11 ∧ y = 48/11 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2405_240580


namespace NUMINAMATH_CALUDE_complex_power_sum_l2405_240561

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_sum : 3 * i^23 + 2 * i^47 = -5 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2405_240561


namespace NUMINAMATH_CALUDE_fifty_eight_prime_sum_l2405_240584

/-- A function that returns the number of ways to write n as the sum of two primes -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range (n + 1))).card

/-- Theorem stating that 58 can be written as the sum of two primes in exactly 3 ways -/
theorem fifty_eight_prime_sum : count_prime_pairs 58 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifty_eight_prime_sum_l2405_240584


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2405_240556

theorem quadratic_inequality (x : ℝ) : x^2 - 48*x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2405_240556


namespace NUMINAMATH_CALUDE_f_of_g_5_l2405_240590

-- Define the functions f and g
def g (x : ℝ) : ℝ := 4 * x + 10
def f (x : ℝ) : ℝ := 6 * x - 12

-- State the theorem
theorem f_of_g_5 : f (g 5) = 168 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_5_l2405_240590


namespace NUMINAMATH_CALUDE_apple_sales_leftover_l2405_240506

/-- The number of apples left over after selling all possible baskets -/
def leftover_apples (oliver patricia quentin basket_size : ℕ) : ℕ :=
  (oliver + patricia + quentin) % basket_size

theorem apple_sales_leftover :
  leftover_apples 58 36 15 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_sales_leftover_l2405_240506


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2405_240581

theorem minimum_value_theorem (x : ℝ) : 
  (x^2 + 5) / Real.sqrt (x^2 + 4) ≥ 5/2 ∧ 
  ∀ ε > 0, ∃ x : ℝ, (x^2 + 5) / Real.sqrt (x^2 + 4) < 5/2 + ε :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2405_240581


namespace NUMINAMATH_CALUDE_birds_on_fence_l2405_240550

theorem birds_on_fence (initial_birds joining_birds : ℕ) :
  initial_birds + joining_birds = initial_birds + joining_birds :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2405_240550


namespace NUMINAMATH_CALUDE_leila_payment_l2405_240589

/-- The total cost Leila should pay Ali for the cakes -/
def total_cost (chocolate_quantity : ℕ) (chocolate_price : ℕ) 
               (strawberry_quantity : ℕ) (strawberry_price : ℕ) : ℕ :=
  chocolate_quantity * chocolate_price + strawberry_quantity * strawberry_price

/-- Theorem stating that Leila should pay Ali $168 for the cakes -/
theorem leila_payment : total_cost 3 12 6 22 = 168 := by
  sorry

end NUMINAMATH_CALUDE_leila_payment_l2405_240589


namespace NUMINAMATH_CALUDE_non_monotonic_interval_implies_k_range_l2405_240585

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

-- Define the property of non-monotonicity in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), a < x ∧ x < y ∧ y < z ∧ z < b ∧
    ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- Theorem statement
theorem non_monotonic_interval_implies_k_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) → 1 ≤ k ∧ k < 3/2 := by sorry

end NUMINAMATH_CALUDE_non_monotonic_interval_implies_k_range_l2405_240585


namespace NUMINAMATH_CALUDE_min_coefficient_value_l2405_240527

theorem min_coefficient_value (a b Box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + a) = 30*x^2 + Box*x + 30) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∀ Box' : ℤ, (∀ x, (a*x + b) * (b*x + a) = 30*x^2 + Box'*x + 30) → Box' ≥ Box) →
  Box = 61 := by
sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l2405_240527


namespace NUMINAMATH_CALUDE_no_prime_sqrt_sum_integer_l2405_240596

theorem no_prime_sqrt_sum_integer :
  ¬ ∃ (p n : ℕ), Prime p ∧ n > 0 ∧ ∃ (k : ℤ), (Int.sqrt (p + n) + Int.sqrt n : ℤ) = k :=
sorry

end NUMINAMATH_CALUDE_no_prime_sqrt_sum_integer_l2405_240596


namespace NUMINAMATH_CALUDE_parabola_directrix_l2405_240570

/-- Given a parabola with equation y = 8x^2, its directrix has equation y = -1/32 -/
theorem parabola_directrix (x y : ℝ) :
  y = 8 * x^2 →
  ∃ (p : ℝ), p > 0 ∧ x^2 = 4 * p * y ∧ -p = -(1/32) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2405_240570


namespace NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l2405_240599

theorem cos_pi_sixth_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/6)) 
  (h2 : Real.sin (α + π/3) = 12/13) : Real.cos (π/6 - α) = 12/13 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l2405_240599


namespace NUMINAMATH_CALUDE_arithmetic_geometric_comparison_l2405_240565

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

theorem arithmetic_geometric_comparison
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_eq2 : a 2 = b 2)
  (h_eq10 : a 10 = b 10) :
  a 6 > b 6 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_comparison_l2405_240565


namespace NUMINAMATH_CALUDE_expression_evaluation_l2405_240520

theorem expression_evaluation (a b : ℝ) (ha : a = 6) (hb : b = 2) :
  3 / (a + b) + a^2 = 291 / 8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2405_240520


namespace NUMINAMATH_CALUDE_complex_number_location_l2405_240517

theorem complex_number_location (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i / (3 + i)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l2405_240517


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2405_240587

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 * a 3 = 2 * a 1 →             -- given condition
  (a 4 + 2 * a 7) / 2 = 5 / 4 →     -- given condition
  q = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2405_240587


namespace NUMINAMATH_CALUDE_target_circle_properties_l2405_240573

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 2 * x - y + 1 = 0

/-- The given circle equation -/
def given_circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0

/-- The equation of the circle we need to prove -/
def target_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 28*x - 15*y = 0

/-- Theorem stating that the target circle passes through the intersection points
    of the line and the given circle, and also through the origin -/
theorem target_circle_properties :
  (∀ x y : ℝ, line_eq x y ∧ given_circle_eq x y → target_circle_eq x y) ∧
  target_circle_eq 0 0 := by
  sorry

end NUMINAMATH_CALUDE_target_circle_properties_l2405_240573


namespace NUMINAMATH_CALUDE_special_hyperbola_equation_l2405_240579

/-- A hyperbola with center at the origin, foci on the x-axis, and specific properties. -/
structure SpecialHyperbola where
  -- The equation of the hyperbola in the form x²/a² - y²/b² = 1
  a : ℝ
  b : ℝ
  -- The right focus is at (c, 0) where c² = a² + b²
  c : ℝ
  h_c : c^2 = a^2 + b^2
  -- A line through the right focus with slope √(3/5)
  line_slope : ℝ
  h_slope : line_slope^2 = 3/5
  -- The line intersects the hyperbola at P and Q
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P_on_hyperbola : (P.1/a)^2 - (P.2/b)^2 = 1
  h_Q_on_hyperbola : (Q.1/a)^2 - (Q.2/b)^2 = 1
  h_P_on_line : P.2 = line_slope * (P.1 - c)
  h_Q_on_line : Q.2 = line_slope * (Q.1 - c)
  -- PO ⊥ OQ
  h_perpendicular : P.1 * Q.1 + P.2 * Q.2 = 0
  -- |PQ| = 4
  h_distance : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 16

/-- The theorem stating that the special hyperbola has the equation x² - y²/3 = 1 -/
theorem special_hyperbola_equation (h : SpecialHyperbola) : h.a^2 = 1 ∧ h.b^2 = 3 := by
  sorry

#check special_hyperbola_equation

end NUMINAMATH_CALUDE_special_hyperbola_equation_l2405_240579


namespace NUMINAMATH_CALUDE_keychain_arrangements_l2405_240583

/-- The number of keys on the keychain -/
def total_keys : ℕ := 7

/-- The number of distinct arrangements of keys on a keychain,
    where two specific keys must be adjacent and arrangements
    are considered identical under rotation and reflection -/
def distinct_arrangements : ℕ := 60

/-- Theorem stating that the number of distinct arrangements
    of keys on the keychain is equal to 60 -/
theorem keychain_arrangements :
  (total_keys : ℕ) = 7 →
  distinct_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_keychain_arrangements_l2405_240583


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2405_240508

def A : Set ℝ := {-1, 1, 3, 5}
def B : Set ℝ := {x | x^2 - 4 < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2405_240508


namespace NUMINAMATH_CALUDE_rams_and_ravis_selection_probability_l2405_240594

theorem rams_and_ravis_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 2/7)
  (h2 : p_both = 0.05714285714285714)
  (h3 : p_both = p_ram * (p_ravi : ℝ)) : 
  p_ravi = 1/5 := by
sorry

end NUMINAMATH_CALUDE_rams_and_ravis_selection_probability_l2405_240594


namespace NUMINAMATH_CALUDE_sandra_share_l2405_240560

/-- Represents the amount of money each person receives -/
structure Share :=
  (amount : ℕ)

/-- Represents the ratio of money distribution -/
structure Ratio :=
  (sandra : ℕ)
  (amy : ℕ)
  (ruth : ℕ)

/-- Calculates the share based on the ratio and a known share -/
def calculateShare (ratio : Ratio) (knownShare : Share) (partInRatio : ℕ) : Share :=
  ⟨knownShare.amount * (ratio.sandra / partInRatio)⟩

theorem sandra_share (ratio : Ratio) (amyShare : Share) :
  ratio.sandra = 2 ∧ ratio.amy = 1 ∧ amyShare.amount = 50 →
  (calculateShare ratio amyShare ratio.amy).amount = 100 := by
  sorry

#check sandra_share

end NUMINAMATH_CALUDE_sandra_share_l2405_240560


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2405_240522

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (1 - z) = -1) :
  Complex.im z = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2405_240522


namespace NUMINAMATH_CALUDE_line_points_relationship_l2405_240566

theorem line_points_relationship (m a b : ℝ) :
  ((-2 : ℝ), a) ∈ {(x, y) | y = -2*x + m} →
  ((2 : ℝ), b) ∈ {(x, y) | y = -2*x + m} →
  a > b := by
  sorry

end NUMINAMATH_CALUDE_line_points_relationship_l2405_240566


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l2405_240588

theorem johns_candy_store_spending (allowance : ℚ) :
  allowance = 4.8 →
  let arcade_spending := (3 / 5) * allowance
  let remaining_after_arcade := allowance - arcade_spending
  let toy_store_spending := (1 / 3) * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 1.28 := by
sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l2405_240588


namespace NUMINAMATH_CALUDE_torn_sheets_count_l2405_240586

/-- Represents a book with numbered pages -/
structure Book where
  /-- The number of the last page in the book -/
  lastPage : ℕ

/-- Represents a range of torn out pages -/
structure TornPages where
  /-- The number of the first torn out page -/
  first : ℕ
  /-- The number of the last torn out page -/
  last : ℕ

/-- Check if a number consists of the same digits as another number in a different order -/
def sameDigitsDifferentOrder (a b : ℕ) : Prop :=
  sorry

/-- Calculate the number of sheets torn out given the first and last torn page numbers -/
def sheetsTornOut (torn : TornPages) : ℕ :=
  (torn.last - torn.first + 1) / 2

/-- The main theorem to be proved -/
theorem torn_sheets_count (book : Book) (torn : TornPages) :
  torn.first = 185 ∧
  sameDigitsDifferentOrder torn.first torn.last ∧
  Even torn.last ∧
  torn.last > torn.first →
  sheetsTornOut torn = 167 := by
  sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l2405_240586


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2405_240551

-- Define the function f and its properties
def f (x : ℝ) : ℝ := sorry

-- Assume |f'(x)| < 1 for all x in the domain of f
axiom f_deriv_bound (x : ℝ) : |deriv f x| < 1

-- Part 1
theorem part_one (a : ℝ) (h : ∀ x ∈ Set.Icc 1 2, f x = a * x + Real.log x) :
  a ∈ Set.Ioo (-3/2) 0 := sorry

-- Part 2
theorem part_two : ∃! x, f x = x := sorry

-- Part 3
def is_periodic (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

theorem part_three (h : is_periodic f 2) :
  ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| < 1 := sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2405_240551


namespace NUMINAMATH_CALUDE_security_check_comprehensive_l2405_240523

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a scenario for which a survey method is chosen -/
structure Scenario where
  requiresAllChecked : Bool
  noExceptions : Bool
  populationAccessible : Bool
  populationFinite : Bool

/-- Determines the correct survey method for a given scenario -/
def correctSurveyMethod (s : Scenario) : SurveyMethod :=
  if s.requiresAllChecked && s.noExceptions && s.populationAccessible && s.populationFinite then
    SurveyMethod.Comprehensive
  else
    SurveyMethod.Sample

/-- The scenario of security checks before boarding a plane -/
def securityCheckScenario : Scenario :=
  { requiresAllChecked := true
    noExceptions := true
    populationAccessible := true
    populationFinite := true }

theorem security_check_comprehensive :
  correctSurveyMethod securityCheckScenario = SurveyMethod.Comprehensive := by
  sorry


end NUMINAMATH_CALUDE_security_check_comprehensive_l2405_240523


namespace NUMINAMATH_CALUDE_susan_average_speed_l2405_240532

/-- Calculates the average speed of a trip with four segments -/
def average_speed (d1 d2 d3 d4 v1 v2 v3 v4 : ℚ) : ℚ :=
  let total_distance := d1 + d2 + d3 + d4
  let total_time := d1 / v1 + d2 / v2 + d3 / v3 + d4 / v4
  total_distance / total_time

/-- Theorem stating that the average speed for Susan's trip is 480/19 mph -/
theorem susan_average_speed :
  average_speed 40 40 60 20 30 15 45 20 = 480 / 19 := by
  sorry

end NUMINAMATH_CALUDE_susan_average_speed_l2405_240532


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2405_240514

def f (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_function_properties :
  ∀ b c d : ℝ,
  f 0 b c d = 2 →
  (∀ y : ℝ, 6*(-1) - y + 7 = 0 ↔ y = f (-1) b c d) →
  (∀ x : ℝ, f x b c d = x^3 - 3*x^2 - 3*x + 2) ∧
  (∀ x : ℝ, x < 1 - Real.sqrt 2 ∨ x > 1 + Real.sqrt 2 → 
    ∀ h : ℝ, h > 0 → f (x + h) b c d > f x b c d) ∧
  (∀ x : ℝ, 1 - Real.sqrt 2 < x ∧ x < 1 + Real.sqrt 2 → 
    ∀ h : ℝ, h > 0 → f (x + h) b c d < f x b c d) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2405_240514


namespace NUMINAMATH_CALUDE_alpha_values_l2405_240504

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 2 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 4 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 3 ∨ α = -Complex.I * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_alpha_values_l2405_240504


namespace NUMINAMATH_CALUDE_tea_pot_volume_l2405_240546

/-- The amount of tea in milliliters per cup -/
def tea_per_cup : ℕ := 65

/-- The number of cups filled with tea -/
def cups_filled : ℕ := 16

/-- The total amount of tea in the pot in milliliters -/
def total_tea : ℕ := tea_per_cup * cups_filled

/-- Theorem stating that the total amount of tea in the pot is 1040 ml -/
theorem tea_pot_volume : total_tea = 1040 := by
  sorry

end NUMINAMATH_CALUDE_tea_pot_volume_l2405_240546


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l2405_240576

theorem absolute_value_sum_difference (a b c : ℚ) :
  a = -1/4 → b = -2 → c = -11/4 → |a| + |b| - |c| = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l2405_240576


namespace NUMINAMATH_CALUDE_new_person_weight_l2405_240501

/-- The weight of a new person joining a group, given certain conditions -/
theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : 
  n = 8 → 
  avg_increase = 3.5 →
  replaced_weight = 65 →
  (n : ℝ) * avg_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2405_240501


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_1_l2405_240502

theorem infinitely_many_primes_4k_plus_1 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4*m + 1) ∧ q ∉ S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_1_l2405_240502


namespace NUMINAMATH_CALUDE_playground_area_not_covered_l2405_240554

theorem playground_area_not_covered (playground_side : ℝ) (building_length building_width : ℝ) : 
  playground_side = 12 →
  building_length = 8 →
  building_width = 5 →
  playground_side * playground_side - building_length * building_width = 104 := by
sorry

end NUMINAMATH_CALUDE_playground_area_not_covered_l2405_240554


namespace NUMINAMATH_CALUDE_extraordinary_stack_size_l2405_240552

/-- An extraordinary stack of cards -/
structure ExtraordinaryStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (pile_a_size : ℕ := n)
  (pile_b_size : ℕ := n)
  (card_57_from_a_position : ℕ := 57)
  (card_200_from_b_position : ℕ := 200)

/-- The number of cards in an extraordinary stack is 198 -/
theorem extraordinary_stack_size :
  ∀ (stack : ExtraordinaryStack),
    stack.card_57_from_a_position % 2 = 1 →
    stack.card_200_from_b_position % 2 = 0 →
    stack.card_57_from_a_position ≤ stack.total_cards →
    stack.card_200_from_b_position ≤ stack.total_cards →
    stack.total_cards = 198 := by
  sorry

end NUMINAMATH_CALUDE_extraordinary_stack_size_l2405_240552


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l2405_240528

theorem abs_neg_two_equals_two : |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l2405_240528


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l2405_240538

theorem triangle_cosine_sum (A B C : ℝ) (h1 : A + B + C = π) (h2 : A = 3 * B) (h3 : A = 9 * C) :
  Real.cos A * Real.cos B + Real.cos B * Real.cos C + Real.cos C * Real.cos A = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l2405_240538


namespace NUMINAMATH_CALUDE_number_transformation_l2405_240534

theorem number_transformation (x : ℝ) : 2 * ((2 * (x + 1)) - 1) = 2 * x + 2 := by
  sorry

#check number_transformation

end NUMINAMATH_CALUDE_number_transformation_l2405_240534


namespace NUMINAMATH_CALUDE_merchant_markup_percentage_l2405_240578

theorem merchant_markup_percentage (C : ℝ) (M : ℝ) : 
  C > 0 →
  ((1 + M / 100) * C - 0.4 * ((1 + M / 100) * C) = 1.05 * C) →
  M = 75 := by
sorry

end NUMINAMATH_CALUDE_merchant_markup_percentage_l2405_240578


namespace NUMINAMATH_CALUDE_problem_1_l2405_240568

theorem problem_1 : (-2.4) + (-3.7) + (-4.6) + 5.7 = -5 := by
  sorry

#eval (-2.4) + (-3.7) + (-4.6) + 5.7

end NUMINAMATH_CALUDE_problem_1_l2405_240568


namespace NUMINAMATH_CALUDE_permutation_count_l2405_240559

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def isValidPermutation (π : Fin 10 → Fin 10) : Prop :=
  Function.Bijective π ∧
  ∀ m n : Fin 10, isPrime ((m : ℕ) + (n : ℕ)) → isPrime ((π m : ℕ) + (π n : ℕ))

theorem permutation_count :
  (∃! (count : ℕ), ∃ (perms : Finset (Fin 10 → Fin 10)),
    Finset.card perms = count ∧
    ∀ π ∈ perms, isValidPermutation π ∧
    ∀ π, isValidPermutation π → π ∈ perms) ∧
  (∃ (perms : Finset (Fin 10 → Fin 10)),
    Finset.card perms = 4 ∧
    ∀ π ∈ perms, isValidPermutation π ∧
    ∀ π, isValidPermutation π → π ∈ perms) :=
by sorry

end NUMINAMATH_CALUDE_permutation_count_l2405_240559


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2405_240567

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_one_minus_i_six :
  (1 - i)^6 = 8*i := by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2405_240567


namespace NUMINAMATH_CALUDE_prime_characterization_l2405_240547

theorem prime_characterization (p : ℕ) (h1 : p > 3) (h2 : (p^2 + 15) % 12 = 4) :
  Nat.Prime p :=
sorry

end NUMINAMATH_CALUDE_prime_characterization_l2405_240547


namespace NUMINAMATH_CALUDE_total_junk_mail_l2405_240574

/-- Given a block with houses and junk mail distribution, calculate the total junk mail. -/
theorem total_junk_mail (num_houses : ℕ) (mail_per_house : ℕ) : num_houses = 10 → mail_per_house = 35 → num_houses * mail_per_house = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_junk_mail_l2405_240574


namespace NUMINAMATH_CALUDE_total_laundry_cost_l2405_240540

def laundry_cost (washer_cost : ℝ) (dryer_cost_per_10_min : ℝ) (loads : ℕ) 
  (special_soap_cost : ℝ) (num_dryers : ℕ) (dryer_time : ℕ) (membership_fee : ℝ) : ℝ :=
  let washing_cost := washer_cost * loads + special_soap_cost
  let dryer_cost := (↑num_dryers * ↑(dryer_time / 10 + 1)) * dryer_cost_per_10_min
  washing_cost + dryer_cost + membership_fee

theorem total_laundry_cost :
  laundry_cost 4 0.25 3 2.5 4 45 10 = 29.5 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_cost_l2405_240540


namespace NUMINAMATH_CALUDE_pat_to_mark_ratio_l2405_240543

/-- Represents the hours charged by each person --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Conditions of the problem --/
def project_conditions (h : ProjectHours) : Prop :=
  h.pat + h.kate + h.mark = 180 ∧
  h.pat = 2 * h.kate ∧
  h.mark = h.kate + 100

/-- Theorem stating the ratio of Pat's hours to Mark's hours --/
theorem pat_to_mark_ratio (h : ProjectHours) :
  project_conditions h → h.pat * 3 = h.mark * 1 := by
  sorry

#check pat_to_mark_ratio

end NUMINAMATH_CALUDE_pat_to_mark_ratio_l2405_240543


namespace NUMINAMATH_CALUDE_final_eraser_count_l2405_240571

/-- Represents the state of erasers in three drawers -/
structure EraserState where
  drawer1 : ℕ
  drawer2 : ℕ
  drawer3 : ℕ

/-- Initial state of erasers -/
def initial_state : EraserState := ⟨139, 95, 75⟩

/-- State after Monday's changes -/
def monday_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 + 50, s.drawer2 - 50, s.drawer3⟩

/-- State after Tuesday's changes -/
def tuesday_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 - 35, s.drawer2, s.drawer3 - 20⟩

/-- Final state after changes later in the week -/
def final_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 + 131, s.drawer2 - 30, s.drawer3⟩

/-- Total number of erasers in all drawers -/
def total_erasers (s : EraserState) : ℕ :=
  s.drawer1 + s.drawer2 + s.drawer3

/-- Theorem stating the final number of erasers -/
theorem final_eraser_count :
  total_erasers (final_state (tuesday_state (monday_state initial_state))) = 355 := by
  sorry


end NUMINAMATH_CALUDE_final_eraser_count_l2405_240571


namespace NUMINAMATH_CALUDE_billy_coins_l2405_240509

/-- Given the number of piles of quarters and dimes, and the number of coins per pile,
    calculate the total number of coins. -/
def total_coins (quarter_piles dime_piles coins_per_pile : ℕ) : ℕ :=
  (quarter_piles + dime_piles) * coins_per_pile

/-- Theorem stating that with 2 piles of quarters, 3 piles of dimes, and 4 coins per pile,
    the total number of coins is 20. -/
theorem billy_coins : total_coins 2 3 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_billy_coins_l2405_240509


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2405_240549

def arithmeticSequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmeticSequence a)
  (h_mean1 : (a 2 + a 6) / 2 = 5)
  (h_mean2 : (a 3 + a 7) / 2 = 7) :
  ∀ n : ℕ, a n = 2 * n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2405_240549


namespace NUMINAMATH_CALUDE_last_digit_of_one_third_to_tenth_l2405_240510

theorem last_digit_of_one_third_to_tenth (n : ℕ) : 
  (1 : ℚ) / 3^10 * 10^n % 10 = 5 :=
sorry

end NUMINAMATH_CALUDE_last_digit_of_one_third_to_tenth_l2405_240510


namespace NUMINAMATH_CALUDE_richs_walk_total_distance_l2405_240555

/-- Calculates the total distance of Rich's walk --/
def richs_walk (segment1 segment2 segment5 : ℝ) : ℝ :=
  let segment3 := 2 * (segment1 + segment2)
  let segment4 := 1.5 * segment3
  let sum_to_5 := segment1 + segment2 + segment3 + segment4 + segment5
  let segment6 := 3 * sum_to_5
  let sum_to_6 := sum_to_5 + segment6
  let segment7 := 0.75 * sum_to_6
  let one_way := segment1 + segment2 + segment3 + segment4 + segment5 + segment6 + segment7
  2 * one_way

theorem richs_walk_total_distance :
  richs_walk 20 200 300 = 22680 := by
  sorry

end NUMINAMATH_CALUDE_richs_walk_total_distance_l2405_240555


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2405_240575

theorem repeating_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 4 / 11 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 15 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2405_240575


namespace NUMINAMATH_CALUDE_original_number_proof_l2405_240512

theorem original_number_proof : ∃ x : ℝ, x * 0.74 = 1.9832 ∧ x = 2.68 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2405_240512


namespace NUMINAMATH_CALUDE_min_value_expression_l2405_240539

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1 / (a + b)^3 ≥ 1 / (4^(1/5 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2405_240539


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2405_240530

/-- Calculate the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 40 →
  passing_time = 25.2 →
  ∃ (bridge_length : ℝ), bridge_length = 160 ∧
    bridge_length = train_speed_kmh * 1000 / 3600 * passing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2405_240530


namespace NUMINAMATH_CALUDE_bridge_length_l2405_240521

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ bridge_length : ℝ,
    bridge_length = 215 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l2405_240521


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_4_power_minus_2_power_29_l2405_240515

theorem greatest_prime_factor_of_4_power_minus_2_power_29 (n : ℕ) : 
  (∃ (p : ℕ), Nat.Prime p ∧ p ∣ (4^n - 2^29) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (4^n - 2^29) → q ≤ p) ∧
  (∀ (q : ℕ), Nat.Prime q → q ∣ (4^n - 2^29) → q ≤ 31) ∧
  (31 ∣ (4^n - 2^29)) →
  n = 17 :=
by sorry


end NUMINAMATH_CALUDE_greatest_prime_factor_of_4_power_minus_2_power_29_l2405_240515


namespace NUMINAMATH_CALUDE_quadratic_solution_l2405_240562

theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : (2 * c)^2 + c * (2 * c) + d = 0)
  (h2 : (-3 * d)^2 + c * (-3 * d) + d = 0) :
  c = -1/6 ∧ d = -1/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2405_240562


namespace NUMINAMATH_CALUDE_parallel_vectors_second_component_l2405_240535

/-- Given vectors a and b in ℝ², if a is parallel to (a + b), then the second component of b is -3. -/
theorem parallel_vectors_second_component (a b : ℝ × ℝ) (h : a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 3) :
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • (a + b)) → b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_second_component_l2405_240535


namespace NUMINAMATH_CALUDE_inverse_75_mod_76_l2405_240526

theorem inverse_75_mod_76 : ∃ x : ℕ, x < 76 ∧ (75 * x) % 76 = 1 :=
by
  use 75
  sorry

end NUMINAMATH_CALUDE_inverse_75_mod_76_l2405_240526


namespace NUMINAMATH_CALUDE_bananas_multiple_of_three_l2405_240511

/-- Represents the number of fruit baskets that can be made -/
def num_baskets : ℕ := 3

/-- Represents the number of oranges Peter has -/
def oranges : ℕ := 18

/-- Represents the number of pears Peter has -/
def pears : ℕ := 27

/-- Represents the number of bananas Peter has -/
def bananas : ℕ := sorry

/-- Theorem stating that the number of bananas must be a multiple of 3 -/
theorem bananas_multiple_of_three :
  ∃ k : ℕ, bananas = 3 * k ∧
  oranges % num_baskets = 0 ∧
  pears % num_baskets = 0 ∧
  bananas % num_baskets = 0 :=
sorry

end NUMINAMATH_CALUDE_bananas_multiple_of_three_l2405_240511


namespace NUMINAMATH_CALUDE_units_digit_of_G_1009_l2405_240524

-- Define G_n
def G (n : ℕ) : ℕ := 3^(2^n) + 1

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_G_1009 : unitsDigit (G 1009) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_G_1009_l2405_240524


namespace NUMINAMATH_CALUDE_function_range_in_unit_interval_l2405_240564

theorem function_range_in_unit_interval (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) :
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_range_in_unit_interval_l2405_240564


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l2405_240518

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (3, -4) and (5, -8) is -24. -/
theorem midpoint_coordinate_product : 
  let x₁ : ℝ := 3
  let y₁ : ℝ := -4
  let x₂ : ℝ := 5
  let y₂ : ℝ := -8
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x * midpoint_y = -24 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l2405_240518


namespace NUMINAMATH_CALUDE_second_number_value_l2405_240577

theorem second_number_value : 
  ∀ (a b c d : ℝ),
  a + b + c + d = 280 →
  a = 2 * b →
  c = (1/3) * a →
  d = b + c →
  b = 52.5 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l2405_240577


namespace NUMINAMATH_CALUDE_sin_cos_equality_implies_ten_degrees_l2405_240529

theorem sin_cos_equality_implies_ten_degrees (x : ℝ) :
  Real.sin (4 * x * π / 180) * Real.sin (5 * x * π / 180) = 
  Real.cos (4 * x * π / 180) * Real.cos (5 * x * π / 180) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equality_implies_ten_degrees_l2405_240529


namespace NUMINAMATH_CALUDE_article_largeFont_wordsPerPage_l2405_240563

/-- Calculates the number of words per page in the large font given the article constraints. -/
def largeFont_wordsPerPage (totalWords smallFont_wordsPerPage totalPages largeFont_pages : ℕ) : ℕ :=
  let smallFont_pages := totalPages - largeFont_pages
  let smallFont_words := smallFont_pages * smallFont_wordsPerPage
  let largeFont_words := totalWords - smallFont_words
  largeFont_words / largeFont_pages

/-- Proves that the number of words per page in the large font is 1800 given the article constraints. -/
theorem article_largeFont_wordsPerPage :
  largeFont_wordsPerPage 48000 2400 21 4 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_article_largeFont_wordsPerPage_l2405_240563


namespace NUMINAMATH_CALUDE_park_diameter_l2405_240519

/-- Given a circular park with a fountain, garden, and walking path, 
    calculate the diameter of the outer boundary of the walking path. -/
theorem park_diameter (fountain_diameter garden_width path_width : ℝ) 
    (h1 : fountain_diameter = 12)
    (h2 : garden_width = 10)
    (h3 : path_width = 6) :
    fountain_diameter + 2 * garden_width + 2 * path_width = 44 :=
by sorry

end NUMINAMATH_CALUDE_park_diameter_l2405_240519


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l2405_240557

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 5 / 3)
  (hz_x : z / x = 1 / 5) :
  w / y = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l2405_240557


namespace NUMINAMATH_CALUDE_expression_evaluation_l2405_240545

theorem expression_evaluation (x : ℚ) (h : x = 1/2) : 
  (1 + x) * (1 - x) + x * (x + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2405_240545


namespace NUMINAMATH_CALUDE_overlapping_circles_area_ratio_l2405_240592

/-- Given two overlapping circles, this theorem proves the ratio of their areas. -/
theorem overlapping_circles_area_ratio
  (L S A : ℝ)  -- L: area of large circle, S: area of small circle, A: overlapped area
  (h1 : A = 3/5 * S)  -- Overlapped area is 3/5 of small circle
  (h2 : A = 6/25 * L)  -- Overlapped area is 6/25 of large circle
  : S / L = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_circles_area_ratio_l2405_240592


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2405_240500

theorem quadratic_root_zero (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a^2 - 9 = 0 → x = 0 ∨ x ≠ 0) →
  (0^2 - 2*0 + a^2 - 9 = 0) →
  (a = 3 ∨ a = -3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2405_240500


namespace NUMINAMATH_CALUDE_second_player_wins_l2405_240553

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the state of the game board -/
structure GameBoard where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a move in the game -/
structure Move where
  player : Player
  position : Fin 3
  value : ℝ

/-- Checks if a move is valid -/
def isValidMove (board : GameBoard) (move : Move) : Prop :=
  match move.position with
  | 0 => move.value ≠ 0  -- a ≠ 0
  | _ => True

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move.position with
  | 0 => { board with a := move.value }
  | 1 => { board with b := move.value }
  | 2 => { board with c := move.value }

/-- Checks if the quadratic equation has real roots -/
def hasRealRoots (board : GameBoard) : Prop :=
  board.b * board.b - 4 * board.a * board.c ≥ 0

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∀ (firstMove : Move),
    isValidMove { a := 0, b := 0, c := 0 } firstMove →
    ∃ (secondMove : Move),
      isValidMove (applyMove { a := 0, b := 0, c := 0 } firstMove) secondMove ∧
      hasRealRoots (applyMove (applyMove { a := 0, b := 0, c := 0 } firstMove) secondMove) :=
sorry


end NUMINAMATH_CALUDE_second_player_wins_l2405_240553


namespace NUMINAMATH_CALUDE_soda_price_ratio_l2405_240536

theorem soda_price_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let brand_y_volume := v
  let brand_y_price := p
  let brand_z_volume := 1.3 * v
  let brand_z_price := 0.85 * p
  (brand_z_price / brand_z_volume) / (brand_y_price / brand_y_volume) = 17 / 26 := by
sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l2405_240536


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2405_240513

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + I) / (1 - I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2405_240513


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2405_240531

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x = 0 → (2*x - 1)*x = 0) ∧
  (∃ x : ℝ, (2*x - 1)*x = 0 ∧ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2405_240531


namespace NUMINAMATH_CALUDE_group_count_divisible_by_27_l2405_240537

/-- Represents the number of groups of each size -/
structure GroupCounts where
  size2 : ℕ
  size5 : ℕ
  size11 : ℕ

/-- The mean size of a group is 4 -/
def mean_size_condition (g : GroupCounts) : Prop :=
  (2 * g.size2 + 5 * g.size5 + 11 * g.size11) / (g.size2 + g.size5 + g.size11) = 4

/-- The mean of answers when each person is asked how many others are in their group is 6 -/
def mean_answer_condition (g : GroupCounts) : Prop :=
  (2 * g.size2 * 1 + 5 * g.size5 * 4 + 11 * g.size11 * 10) / (2 * g.size2 + 5 * g.size5 + 11 * g.size11) = 6

/-- The main theorem to prove -/
theorem group_count_divisible_by_27 (g : GroupCounts) 
  (h1 : mean_size_condition g) (h2 : mean_answer_condition g) : 
  ∃ k : ℕ, g.size2 + g.size5 + g.size11 = 27 * k := by
  sorry

end NUMINAMATH_CALUDE_group_count_divisible_by_27_l2405_240537


namespace NUMINAMATH_CALUDE_f_equal_implies_sum_negative_l2405_240548

noncomputable def f (x : ℝ) : ℝ := ((1 - x) / (1 + x^2)) * Real.exp x

theorem f_equal_implies_sum_negative (x₁ x₂ : ℝ) (h₁ : f x₁ = f x₂) (h₂ : x₁ ≠ x₂) : x₁ + x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_equal_implies_sum_negative_l2405_240548


namespace NUMINAMATH_CALUDE_garden_center_discount_l2405_240505

/-- Represents the purchase and payment details at a garden center --/
structure GardenPurchase where
  pansy_count : ℕ
  pansy_price : ℚ
  hydrangea_count : ℕ
  hydrangea_price : ℚ
  petunia_count : ℕ
  petunia_price : ℚ
  paid_amount : ℚ
  change_received : ℚ

/-- Calculates the discount offered by the garden center --/
def calculate_discount (purchase : GardenPurchase) : ℚ :=
  let total_cost := purchase.pansy_count * purchase.pansy_price +
                    purchase.hydrangea_count * purchase.hydrangea_price +
                    purchase.petunia_count * purchase.petunia_price
  let amount_paid := purchase.paid_amount - purchase.change_received
  total_cost - amount_paid

/-- Theorem stating that the discount for the given purchase is $3.00 --/
theorem garden_center_discount :
  let purchase := GardenPurchase.mk 5 2.5 1 12.5 5 1 50 23
  calculate_discount purchase = 3 := by sorry

end NUMINAMATH_CALUDE_garden_center_discount_l2405_240505


namespace NUMINAMATH_CALUDE_part_time_employees_l2405_240569

/-- Represents the number of employees in a corporation -/
structure Corporation where
  total : ℕ
  fullTime : ℕ
  partTime : ℕ

/-- The total number of employees is the sum of full-time and part-time employees -/
axiom total_eq_sum (c : Corporation) : c.total = c.fullTime + c.partTime

/-- Theorem: Given a corporation with 65,134 total employees and 63,093 full-time employees,
    the number of part-time employees is 2,041 -/
theorem part_time_employees (c : Corporation) 
    (h1 : c.total = 65134) 
    (h2 : c.fullTime = 63093) : 
    c.partTime = 2041 := by
  sorry


end NUMINAMATH_CALUDE_part_time_employees_l2405_240569


namespace NUMINAMATH_CALUDE_negation_quadratic_roots_l2405_240558

theorem negation_quadratic_roots (a b c : ℝ) :
  (¬(b^2 - 4*a*c < 0 → ∀ x, a*x^2 + b*x + c ≠ 0)) ↔
  (b^2 - 4*a*c ≥ 0 → ∃ x, a*x^2 + b*x + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_quadratic_roots_l2405_240558


namespace NUMINAMATH_CALUDE_symmetrical_letters_count_l2405_240516

-- Define a function to check if a character is symmetrical
def is_symmetrical (c : Char) : Bool :=
  c = 'A' ∨ c = 'H' ∨ c = 'I' ∨ c = 'M' ∨ c = 'O' ∨ c = 'T' ∨ c = 'U' ∨ c = 'V' ∨ c = 'W' ∨ c = 'X' ∨ c = 'Y'

-- Define the sign text
def sign_text : String := "PUNK CD FOR SALE"

-- Theorem statement
theorem symmetrical_letters_count :
  (sign_text.toList.filter is_symmetrical).length = 3 :=
sorry

end NUMINAMATH_CALUDE_symmetrical_letters_count_l2405_240516


namespace NUMINAMATH_CALUDE_parabola_c_value_l2405_240598

/-- A parabola is defined by the equation x = ay² + by + c, where a, b, and c are constants -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℚ) : ℚ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-5) = 1 → p.x_coord (-1) = 4 → p.c = 145/12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2405_240598


namespace NUMINAMATH_CALUDE_min_area_triangle_m_sum_l2405_240533

/-- The sum of m values for minimum area triangle -/
theorem min_area_triangle_m_sum : 
  ∀ (m : ℤ), 
  let A : ℝ × ℝ := (2, 5)
  let B : ℝ × ℝ := (14, 13)
  let C : ℝ × ℝ := (6, m)
  let triangle_area (m : ℤ) : ℝ := sorry -- Function to calculate triangle area
  let min_area : ℝ := sorry -- Minimum area of the triangle
  (∃ (m₁ m₂ : ℤ), 
    m₁ ≠ m₂ ∧ 
    triangle_area m₁ = min_area ∧ 
    triangle_area m₂ = min_area ∧ 
    m₁ + m₂ = 16) := by sorry


end NUMINAMATH_CALUDE_min_area_triangle_m_sum_l2405_240533


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2405_240597

theorem absolute_value_inequality (x : ℝ) :
  |6 - x| / 4 > 1 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2405_240597


namespace NUMINAMATH_CALUDE_fraction_value_l2405_240595

theorem fraction_value (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, x + (2 * q - p) / (2 * q + p) = 2 ∧ x = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2405_240595


namespace NUMINAMATH_CALUDE_equation_solution_l2405_240507

theorem equation_solution :
  ∀ x : ℝ, x ≠ 1 →
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2405_240507


namespace NUMINAMATH_CALUDE_goods_train_speed_l2405_240541

/-- The speed of a goods train crossing a platform -/
theorem goods_train_speed (platform_length : ℝ) (crossing_time : ℝ) (train_length : ℝ)
  (h1 : platform_length = 250)
  (h2 : crossing_time = 26)
  (h3 : train_length = 270.0416) :
  ∃ (speed : ℝ), abs (speed - 20) < 0.01 ∧ 
  speed = (platform_length + train_length) / crossing_time :=
sorry

end NUMINAMATH_CALUDE_goods_train_speed_l2405_240541
