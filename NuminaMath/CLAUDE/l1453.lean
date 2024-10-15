import Mathlib

namespace NUMINAMATH_CALUDE_crank_slider_motion_l1453_145378

/-- Crank-slider mechanism -/
structure CrankSlider where
  oa : ℝ
  ab : ℝ
  am : ℝ
  ω : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Velocity vector -/
structure Velocity where
  vx : ℝ
  vy : ℝ

/-- Theorem for crank-slider mechanism motion -/
theorem crank_slider_motion (cs : CrankSlider) 
  (h1 : cs.oa = 90)
  (h2 : cs.ab = 90)
  (h3 : cs.am = 2/3 * cs.ab)
  (h4 : cs.ω = 10) :
  (∃ (m : ℝ → Point),
    (∀ t, m t = ⟨30 * Real.cos (10 * t) + 60, 60 * Real.sin (10 * t)⟩) ∧
    (∀ p : Point, (p.y)^2 + (30 - (p.x - 60) / (1/3))^2 = 3600) ∧
    (∃ (v : ℝ → Velocity), ∀ t, v t = ⟨-300 * Real.sin (10 * t), 600 * Real.cos (10 * t)⟩)) :=
by sorry

end NUMINAMATH_CALUDE_crank_slider_motion_l1453_145378


namespace NUMINAMATH_CALUDE_quadratic_integer_solutions_count_l1453_145329

theorem quadratic_integer_solutions_count : 
  ∃! (S : Finset ℚ), 
    (∀ k ∈ S, |k| < 100 ∧ 
      ∃! (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
        3 * (x₁ : ℚ)^2 + k * x₁ + 8 = 0 ∧ 
        3 * (x₂ : ℚ)^2 + k * x₂ + 8 = 0) ∧
    Finset.card S = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_solutions_count_l1453_145329


namespace NUMINAMATH_CALUDE_coin_combination_l1453_145313

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (twenty_five_cent : ℕ) : ℕ :=
  75 - 4 * five_cent

theorem coin_combination (five_cent : ℕ) (twenty_five_cent : ℕ) :
  five_cent + twenty_five_cent = 15 →
  different_values five_cent twenty_five_cent = 27 →
  twenty_five_cent = 3 := by
sorry

end NUMINAMATH_CALUDE_coin_combination_l1453_145313


namespace NUMINAMATH_CALUDE_no_valid_assignment_for_45gon_l1453_145376

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  is_regular : sides = n

/-- Represents an assignment of digits to vertices of a polygon -/
def DigitAssignment (n : ℕ) := Fin n → Fin 10

/-- Checks if an assignment satisfies the pairwise condition -/
def SatisfiesPairwiseCondition (n : ℕ) (assignment : DigitAssignment n) : Prop :=
  ∀ (i j : Fin 10), i ≠ j →
    ∃ (v w : Fin n), v ≠ w ∧ 
      assignment v = i ∧ 
      assignment w = j ∧ 
      (v.val + 1) % n = w.val ∨ (w.val + 1) % n = v.val

/-- The main theorem stating that no valid assignment exists for a 45-gon -/
theorem no_valid_assignment_for_45gon :
  ¬∃ (assignment : DigitAssignment 45), 
    SatisfiesPairwiseCondition 45 assignment :=
sorry

end NUMINAMATH_CALUDE_no_valid_assignment_for_45gon_l1453_145376


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_in_set_l1453_145398

theorem sqrt_2_irrational_in_set (S : Set ℝ) : 
  S = {1/7, Real.sqrt 2, (8 : ℝ) ^ (1/3), 1.010010001} → 
  ∃ x ∈ S, Irrational x ∧ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_in_set_l1453_145398


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_7_l1453_145399

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2010WithDigitSum7 (year : ℕ) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 7 ∧ 
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 7

theorem first_year_after_2010_with_digit_sum_7 : 
  isFirstYearAfter2010WithDigitSum7 2014 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_7_l1453_145399


namespace NUMINAMATH_CALUDE_max_value_of_y_l1453_145303

def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_value_of_y :
  ∃ (α : ℝ), α = 3 ∧ ∀ x, -1 ≤ x → x ≤ 2 → y x ≤ α :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_y_l1453_145303


namespace NUMINAMATH_CALUDE_product_equals_fraction_l1453_145365

theorem product_equals_fraction : 12 * 0.5 * 3 * 0.2 = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l1453_145365


namespace NUMINAMATH_CALUDE_algebra_drafting_not_geography_algebra_drafting_not_geography_eq_25_l1453_145311

theorem algebra_drafting_not_geography (total_algebra : ℕ) (both_algebra_drafting : ℕ) 
  (drafting_only : ℕ) (total_geography : ℕ) (both_algebra_drafting_geography : ℕ) : ℕ :=
  let algebra_only := total_algebra - both_algebra_drafting
  let total_one_subject := algebra_only + drafting_only
  let result := total_one_subject - both_algebra_drafting_geography
  
  have h1 : total_algebra = 30 := by sorry
  have h2 : both_algebra_drafting = 15 := by sorry
  have h3 : drafting_only = 12 := by sorry
  have h4 : total_geography = 8 := by sorry
  have h5 : both_algebra_drafting_geography = 2 := by sorry

  result

theorem algebra_drafting_not_geography_eq_25 : 
  algebra_drafting_not_geography 30 15 12 8 2 = 25 := by sorry

end NUMINAMATH_CALUDE_algebra_drafting_not_geography_algebra_drafting_not_geography_eq_25_l1453_145311


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1453_145328

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 20 → c^2 = a^2 + b^2 → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1453_145328


namespace NUMINAMATH_CALUDE_power_fraction_equality_l1453_145335

theorem power_fraction_equality : (2^2016 + 2^2014) / (2^2016 - 2^2014) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l1453_145335


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1453_145348

/-- A geometric sequence with first term 1 and product of first three terms -8 has common ratio -2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence property
    a 1 = 1 →                              -- first term is 1
    a 1 * a 2 * a 3 = -8 →                 -- product of first three terms is -8
    a 2 / a 1 = -2 :=                      -- common ratio is -2
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1453_145348


namespace NUMINAMATH_CALUDE_sam_puppies_l1453_145351

theorem sam_puppies (initial : ℝ) (given_away : ℝ) (h1 : initial = 6.0) (h2 : given_away = 2.0) :
  initial - given_away = 4.0 := by sorry

end NUMINAMATH_CALUDE_sam_puppies_l1453_145351


namespace NUMINAMATH_CALUDE_remaining_note_denomination_l1453_145300

theorem remaining_note_denomination 
  (total_amount : ℕ) 
  (total_notes : ℕ) 
  (fifty_notes : ℕ) 
  (fifty_denomination : ℕ) :
  total_amount = 10350 →
  total_notes = 90 →
  fifty_notes = 77 →
  fifty_denomination = 50 →
  ∃ (remaining_denomination : ℕ),
    remaining_denomination * (total_notes - fifty_notes) = 
      total_amount - (fifty_notes * fifty_denomination) ∧
    remaining_denomination = 500 := by
  sorry

end NUMINAMATH_CALUDE_remaining_note_denomination_l1453_145300


namespace NUMINAMATH_CALUDE_number_composition_l1453_145375

/-- The number of hundreds in the given number -/
def hundreds : ℕ := 11

/-- The number of tens in the given number -/
def tens : ℕ := 11

/-- The number of units in the given number -/
def units : ℕ := 11

/-- The theorem stating that the number consisting of 11 hundreds, 11 tens, and 11 units is 1221 -/
theorem number_composition : 
  hundreds * 100 + tens * 10 + units = 1221 := by sorry

end NUMINAMATH_CALUDE_number_composition_l1453_145375


namespace NUMINAMATH_CALUDE_all_hop_sequences_eventually_periodic_l1453_145336

/-- The biggest positive prime number that divides n -/
def f (n : ℕ) : ℕ := sorry

/-- The smallest positive prime number that divides n -/
def g (n : ℕ) : ℕ := sorry

/-- The next position after hopping from n -/
def hop (n : ℕ) : ℕ := f n + g n

/-- A sequence is eventually periodic if it reaches a cycle after some point -/
def EventuallyPeriodic (seq : ℕ → ℕ) : Prop :=
  ∃ (start cycle : ℕ), ∀ n ≥ start, seq (n + cycle) = seq n

/-- The sequence of hops starting from k -/
def hopSequence (k : ℕ) : ℕ → ℕ
  | 0 => k
  | n + 1 => hop (hopSequence k n)

theorem all_hop_sequences_eventually_periodic :
  ∀ k > 1, EventuallyPeriodic (hopSequence k) := by sorry

end NUMINAMATH_CALUDE_all_hop_sequences_eventually_periodic_l1453_145336


namespace NUMINAMATH_CALUDE_triangle_side_difference_bound_l1453_145362

/-- Given a triangle ABC with side lengths a, b, c and corresponding opposite angles A, B, C,
    prove that if a = 1 and C - B = π/2, then √2/2 < c - b < 1 -/
theorem triangle_side_difference_bound (a b c A B C : Real) : 
  a = 1 → 
  C - B = π / 2 → 
  0 < A ∧ 0 < B ∧ 0 < C → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  Real.sqrt 2 / 2 < c - b ∧ c - b < 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_bound_l1453_145362


namespace NUMINAMATH_CALUDE_shift_left_sum_l1453_145315

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_left (f : QuadraticFunction) (units : ℝ) : QuadraticFunction :=
  QuadraticFunction.mk
    f.a
    (f.b + 2 * f.a * units)
    (f.a * units^2 + f.b * units + f.c)

theorem shift_left_sum (f : QuadraticFunction) :
  let g := shift_left f 6
  g.a + g.b + g.c = 156 :=
by
  sorry

end NUMINAMATH_CALUDE_shift_left_sum_l1453_145315


namespace NUMINAMATH_CALUDE_investment_value_after_two_years_l1453_145369

/-- Calculates the value of an investment after a given period --/
def investment_value (income : ℝ) (income_expenditure_ratio : ℝ × ℝ) 
  (savings_rate : ℝ) (tax_rate : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  let expenditure := income * income_expenditure_ratio.2 / income_expenditure_ratio.1
  let savings := income - expenditure
  let amount_saved := income * savings_rate
  let tax_deductions := income * tax_rate
  let net_investment := amount_saved - tax_deductions
  net_investment * (1 + interest_rate) ^ years

/-- Theorem stating the value of the investment after two years --/
theorem investment_value_after_two_years :
  investment_value 19000 (5, 4) 0.15 0.10 0.08 2 = 1108.08 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_after_two_years_l1453_145369


namespace NUMINAMATH_CALUDE_scientific_calculator_cost_l1453_145321

theorem scientific_calculator_cost
  (total_cost : ℕ)
  (num_scientific : ℕ)
  (num_graphing : ℕ)
  (graphing_cost : ℕ)
  (h1 : total_cost = 1625)
  (h2 : num_scientific = 20)
  (h3 : num_graphing = 25)
  (h4 : graphing_cost = 57)
  (h5 : num_scientific + num_graphing = 45) :
  ∃ (scientific_cost : ℕ),
    scientific_cost * num_scientific + graphing_cost * num_graphing = total_cost ∧
    scientific_cost = 10 :=
by sorry

end NUMINAMATH_CALUDE_scientific_calculator_cost_l1453_145321


namespace NUMINAMATH_CALUDE_gcd_5670_9800_l1453_145382

theorem gcd_5670_9800 : Nat.gcd 5670 9800 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5670_9800_l1453_145382


namespace NUMINAMATH_CALUDE_locus_of_point_M_l1453_145363

/-- The locus of point M given an ellipse and conditions on point P -/
theorem locus_of_point_M (x₀ y₀ x y : ℝ) : 
  (4 * x₀^2 + y₀^2 = 4) →  -- P(x₀, y₀) is on the ellipse
  ((0, -y₀) = (2*(x - x₀), -2*y)) →  -- PD = 2MD condition
  (x^2 + y^2 = 1) -- M(x, y) is on the unit circle
  := by sorry

end NUMINAMATH_CALUDE_locus_of_point_M_l1453_145363


namespace NUMINAMATH_CALUDE_system_solution_l1453_145349

theorem system_solution (x y u v : ℝ) : 
  x^2 + y^2 + u^2 + v^2 = 4 →
  x * y * u + y * u * v + u * v * x + v * x * y = -2 →
  x * y * u * v = -1 →
  ((x = 1 ∧ y = 1 ∧ u = 1 ∧ v = -1) ∨
   (x = 1 ∧ y = 1 ∧ u = -1 ∧ v = 1) ∨
   (x = 1 ∧ y = -1 ∧ u = 1 ∧ v = 1) ∨
   (x = -1 ∧ y = 1 ∧ u = 1 ∧ v = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1453_145349


namespace NUMINAMATH_CALUDE_max_value_expression_l1453_145343

theorem max_value_expression (x y : ℝ) :
  (Real.sqrt (8 - 4 * Real.sqrt 3) * Real.sin x - 3 * Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) *
  (3 + 2 * Real.sqrt (11 - Real.sqrt 3) * Real.cos y - Real.cos (2 * y)) ≤ 33 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1453_145343


namespace NUMINAMATH_CALUDE_two_tap_system_solution_l1453_145347

/-- Represents the time it takes for a tap to fill a tank -/
structure TapTime where
  minutes : ℝ
  positive : minutes > 0

/-- Represents a system of two taps filling a tank -/
structure TwoTapSystem where
  tapA : TapTime
  tapB : TapTime
  timeDifference : tapA.minutes = tapB.minutes + 22
  combinedTime : (1 / tapA.minutes + 1 / tapB.minutes) * 60 = 1

theorem two_tap_system_solution (system : TwoTapSystem) :
  system.tapB.minutes = 110 ∧ system.tapA.minutes = 132 := by
  sorry

end NUMINAMATH_CALUDE_two_tap_system_solution_l1453_145347


namespace NUMINAMATH_CALUDE_subtraction_with_division_l1453_145360

theorem subtraction_with_division : 3034 - (1002 / 20.04) = 2984 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l1453_145360


namespace NUMINAMATH_CALUDE_unique_solution_l1453_145322

theorem unique_solution (a b c : ℕ+) 
  (eq1 : b = a^2 - a)
  (eq2 : c = b^2 - b)
  (eq3 : a = c^2 - c) : 
  a = 2 ∧ b = 2 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1453_145322


namespace NUMINAMATH_CALUDE_no_common_sale_days_l1453_145344

def bookstore_sales : Set Nat :=
  {d | d ≤ 31 ∧ ∃ k, d = 4 * k}

def shoe_store_sales : Set Nat :=
  {d | d ≤ 31 ∧ ∃ n, d = 2 + 8 * n}

theorem no_common_sale_days : bookstore_sales ∩ shoe_store_sales = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_common_sale_days_l1453_145344


namespace NUMINAMATH_CALUDE_other_diagonal_length_l1453_145394

/-- A trapezoid with diagonals intersecting at a right angle -/
structure RightAngledTrapezoid where
  midline : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  diagonals_perpendicular : diagonal1 * diagonal2 = midline * midline * 2

/-- Theorem: In a right-angled trapezoid with midline 6.5 and one diagonal 12, the other diagonal is 5 -/
theorem other_diagonal_length (t : RightAngledTrapezoid) 
  (h1 : t.midline = 6.5) 
  (h2 : t.diagonal1 = 12) : 
  t.diagonal2 = 5 := by
sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l1453_145394


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1453_145386

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The product of two functions -/
def FunctionProduct (f g : ℝ → ℝ) : ℝ → ℝ :=
  fun x ↦ f x * g x

theorem necessary_not_sufficient :
  (∀ f g : ℝ → ℝ, (IsEven f ∧ IsEven g ∨ IsOdd f ∧ IsOdd g) →
    IsEven (FunctionProduct f g)) ∧
  (∃ f g : ℝ → ℝ, IsEven (FunctionProduct f g) ∧
    ¬(IsEven f ∧ IsEven g ∨ IsOdd f ∧ IsOdd g)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1453_145386


namespace NUMINAMATH_CALUDE_hockey_skates_fraction_l1453_145364

/-- Proves that the fraction of money spent on hockey skates is 1/2 --/
theorem hockey_skates_fraction (initial_amount pad_cost remaining : ℚ)
  (h1 : initial_amount = 150)
  (h2 : pad_cost = 50)
  (h3 : remaining = 25) :
  (initial_amount - pad_cost - remaining) / initial_amount = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hockey_skates_fraction_l1453_145364


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1453_145366

-- Define the value of a banana in terms of oranges
def banana_value (banana_count : ℚ) (orange_count : ℕ) : Prop :=
  banana_count * (15 / 12) = orange_count

-- Theorem statement
theorem banana_orange_equivalence :
  banana_value (4 / 5 * 15) 12 →
  banana_value (3 / 4 * 8) 6 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1453_145366


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1453_145304

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_calculation (principal1 principal2 time1 time2 rate1 : ℝ) 
  (h1 : principal1 = 100)
  (h2 : principal2 = 600)
  (h3 : time1 = 48)
  (h4 : time2 = 4)
  (h5 : rate1 = 0.05)
  (h6 : simple_interest principal1 rate1 time1 = simple_interest principal2 ((10 : ℝ) / 100) time2) :
  ∃ (rate2 : ℝ), rate2 = (10 : ℝ) / 100 ∧ 
    simple_interest principal1 rate1 time1 = simple_interest principal2 rate2 time2 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1453_145304


namespace NUMINAMATH_CALUDE_product_to_power_minus_one_l1453_145340

theorem product_to_power_minus_one :
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = 3^16 - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_to_power_minus_one_l1453_145340


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1453_145346

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees. -/
  vertex_angle : ℝ
  /-- The sphere is tangent to the sides of the cone and rests on the table. -/
  sphere_tangent : Bool

/-- The volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ := sorry

/-- Theorem stating the volume of the inscribed sphere for specific cone dimensions. -/
theorem inscribed_sphere_volume (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90)
  (h3 : cone.sphere_tangent = true) :
  sphere_volume cone = 2304 * Real.pi := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1453_145346


namespace NUMINAMATH_CALUDE_remainder_problem_l1453_145341

theorem remainder_problem (y : ℤ) : 
  ∃ (k : ℤ), y = 276 * k + 42 → ∃ (m : ℤ), y = 23 * m + 19 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1453_145341


namespace NUMINAMATH_CALUDE_proportional_set_l1453_145330

/-- A set of four positive real numbers is proportional if and only if
    the product of the extremes equals the product of the means. -/
def IsProportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

/-- The set of line segments (2, 3, 4, 6) is proportional. -/
theorem proportional_set : IsProportional 2 3 4 6 := by
  sorry

end NUMINAMATH_CALUDE_proportional_set_l1453_145330


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1453_145367

/-- Given a triangle with inradius 2.5 cm and area 30 cm², its perimeter is 24 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 30 → A = r * (p / 2) → p = 24 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1453_145367


namespace NUMINAMATH_CALUDE_books_per_shelf_l1453_145393

theorem books_per_shelf
  (mystery_shelves : ℕ)
  (picture_shelves : ℕ)
  (total_books : ℕ)
  (h1 : mystery_shelves = 5)
  (h2 : picture_shelves = 4)
  (h3 : total_books = 54)
  (h4 : total_books % (mystery_shelves + picture_shelves) = 0)  -- Ensures even distribution
  : total_books / (mystery_shelves + picture_shelves) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l1453_145393


namespace NUMINAMATH_CALUDE_equation_roots_l1453_145333

/-- The equation has at least two distinct roots if and only if a = 20 -/
theorem equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    a^2 * (x - 2) + a * (39 - 20*x) + 20 = 0 ∧ 
    a^2 * (y - 2) + a * (39 - 20*y) + 20 = 0) ↔ 
  a = 20 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l1453_145333


namespace NUMINAMATH_CALUDE_foreign_trade_income_2007_2009_l1453_145356

/-- Represents the foreign trade income equation given the initial value,
    final value, and growth rate over a two-year period. -/
def foreign_trade_equation (initial : ℝ) (final : ℝ) (rate : ℝ) : Prop :=
  initial * (1 + rate)^2 = final

/-- Theorem stating that the foreign trade income equation holds for the given values. -/
theorem foreign_trade_income_2007_2009 :
  foreign_trade_equation 2.5 3.6 x = true :=
sorry

end NUMINAMATH_CALUDE_foreign_trade_income_2007_2009_l1453_145356


namespace NUMINAMATH_CALUDE_two_ladies_walk_l1453_145338

/-- The combined distance walked by two ladies in Central Park -/
def combined_distance (lady1_distance lady2_distance : ℝ) : ℝ :=
  lady1_distance + lady2_distance

/-- Theorem: The combined distance of two ladies is 12 miles when one walks twice as far as the other, and the second lady walks 4 miles -/
theorem two_ladies_walk :
  ∀ (lady1_distance lady2_distance : ℝ),
  lady2_distance = 4 →
  lady1_distance = 2 * lady2_distance →
  combined_distance lady1_distance lady2_distance = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_two_ladies_walk_l1453_145338


namespace NUMINAMATH_CALUDE_problem_solution_l1453_145384

def A : Set ℝ := {-1, 0}
def B (x : ℝ) : Set ℝ := {0, 1, x+2}

theorem problem_solution (x : ℝ) (h : A ⊆ B x) : x = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1453_145384


namespace NUMINAMATH_CALUDE_common_difference_is_two_l1453_145345

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 5 / 5 - seq.S 2 / 2 = 3) : 
    seq.a 2 - seq.a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l1453_145345


namespace NUMINAMATH_CALUDE_dry_cleaning_time_is_ten_l1453_145388

def total_time : ℕ := 180 -- 3 hours = 180 minutes
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dog_groomer_time : ℕ := 20
def cooking_time : ℕ := 90

def dry_cleaning_time : ℕ := total_time - commute_time - grocery_time - dog_groomer_time - cooking_time

theorem dry_cleaning_time_is_ten : dry_cleaning_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_dry_cleaning_time_is_ten_l1453_145388


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_5000_l1453_145357

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

theorem sum_of_digits_1_to_5000 : sumOfDigitsUpTo 5000 = 229450 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_5000_l1453_145357


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1453_145306

theorem point_in_second_quadrant (a : ℝ) : 
  (a - 3 < 0 ∧ a + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

#check point_in_second_quadrant

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1453_145306


namespace NUMINAMATH_CALUDE_correct_calculation_result_l1453_145334

theorem correct_calculation_result (x : ℝ) (h : 5 * x = 30) : 8 * x = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l1453_145334


namespace NUMINAMATH_CALUDE_consecutive_integers_product_120_l1453_145350

theorem consecutive_integers_product_120 :
  ∃ (a b c d e : ℤ),
    b = a + 1 ∧
    d = c + 1 ∧
    e = c + 2 ∧
    a * b = 120 ∧
    c * d * e = 120 ∧
    a + b + c + d + e = 37 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_120_l1453_145350


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_zero_l1453_145385

-- Define a complex number z
def z (a : ℝ) : ℂ := Complex.I * (1 + a * Complex.I)

-- State the theorem
theorem pure_imaginary_implies_a_zero (a : ℝ) :
  (∃ b : ℝ, z a = Complex.I * b) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_zero_l1453_145385


namespace NUMINAMATH_CALUDE_work_equivalence_first_group_size_l1453_145310

/-- The number of days it takes the first group to complete the work -/
def days_group1 : ℕ := 40

/-- The number of men in the second group -/
def men_group2 : ℕ := 20

/-- The number of days it takes the second group to complete the work -/
def days_group2 : ℕ := 68

/-- The number of men in the first group -/
def men_group1 : ℕ := 34

theorem work_equivalence :
  men_group1 * days_group1 = men_group2 * days_group2 :=
sorry

theorem first_group_size :
  men_group1 = (men_group2 * days_group2) / days_group1 :=
sorry

end NUMINAMATH_CALUDE_work_equivalence_first_group_size_l1453_145310


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1453_145379

/-- The sum of interior numbers in a row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  sumInteriorNumbers 6 = 30 →
  sumInteriorNumbers 8 = 126 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1453_145379


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l1453_145353

theorem easter_egg_distribution (baskets : ℕ) (eggs_per_basket : ℕ) (people : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  people = 20 →
  (baskets * eggs_per_basket) / people = 9 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l1453_145353


namespace NUMINAMATH_CALUDE_quadratic_integer_expression_l1453_145314

theorem quadratic_integer_expression (A B C : ℤ) :
  ∃ (k l m : ℚ), 
    (∀ x, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) ∧
    ((∀ x : ℤ, ∃ y : ℤ, A * x^2 + B * x + C = y) ↔ 
      (∃ (k' l' m' : ℤ), k = k' ∧ l = l' ∧ m = m')) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_expression_l1453_145314


namespace NUMINAMATH_CALUDE_stating_wholesale_cost_calculation_l1453_145373

/-- The wholesale cost of a sleeping bag -/
def wholesale_cost : ℝ := 24.56

/-- The retailer's profit percentage -/
def profit_percentage : ℝ := 0.14

/-- The selling price of a sleeping bag -/
def selling_price : ℝ := 28

/-- 
Theorem stating that the wholesale cost is correct given the profit percentage and selling price
-/
theorem wholesale_cost_calculation (ε : ℝ) (h : ε > 0) : 
  ∃ (W : ℝ), W > 0 ∧ abs (W - wholesale_cost) < ε ∧ 
  W * (1 + profit_percentage) = selling_price :=
sorry

end NUMINAMATH_CALUDE_stating_wholesale_cost_calculation_l1453_145373


namespace NUMINAMATH_CALUDE_minBrokenSticks_correct_canFormSquare_15_not_canFormSquare_12_l1453_145372

/-- Given n sticks of lengths 1, 2, ..., n, this function returns the minimum number
    of sticks that need to be broken in half to form a square. If it's possible to
    form a square without breaking any sticks, it returns 0. -/
def minBrokenSticks (n : ℕ) : ℕ :=
  if n = 12 then 2
  else if n = 15 then 0
  else sorry

theorem minBrokenSticks_correct :
  (minBrokenSticks 12 = 2) ∧ (minBrokenSticks 15 = 0) := by sorry

/-- Function to check if it's possible to form a square from n sticks of lengths 1, 2, ..., n
    without breaking any sticks -/
def canFormSquare (n : ℕ) : Prop :=
  ∃ (a b c d : List ℕ), 
    (a ++ b ++ c ++ d).sum = n * (n + 1) / 2 ∧
    (∀ x ∈ a ++ b ++ c ++ d, x ≤ n) ∧
    a.sum = b.sum ∧ b.sum = c.sum ∧ c.sum = d.sum

theorem canFormSquare_15 : canFormSquare 15 := by sorry

theorem not_canFormSquare_12 : ¬ canFormSquare 12 := by sorry

end NUMINAMATH_CALUDE_minBrokenSticks_correct_canFormSquare_15_not_canFormSquare_12_l1453_145372


namespace NUMINAMATH_CALUDE_gcd_problem_l1453_145354

theorem gcd_problem (A B : ℕ) (h1 : Nat.lcm A B = 180) (h2 : ∃ (k : ℕ), A = 4 * k ∧ B = 5 * k) : 
  Nat.gcd A B = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1453_145354


namespace NUMINAMATH_CALUDE_freds_spending_ratio_l1453_145389

/-- The ratio of Fred's movie spending to his weekly allowance -/
def movie_allowance_ratio (weekly_allowance : ℚ) (car_wash_earnings : ℚ) (final_amount : ℚ) : ℚ × ℚ :=
  let total_before_movies := final_amount + car_wash_earnings
  let movie_spending := total_before_movies - weekly_allowance
  (movie_spending, weekly_allowance)

/-- Theorem stating the ratio of Fred's movie spending to his weekly allowance -/
theorem freds_spending_ratio :
  let weekly_allowance : ℚ := 16
  let car_wash_earnings : ℚ := 6
  let final_amount : ℚ := 14
  let (numerator, denominator) := movie_allowance_ratio weekly_allowance car_wash_earnings final_amount
  numerator / denominator = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_freds_spending_ratio_l1453_145389


namespace NUMINAMATH_CALUDE_magic_8_ball_theorem_l1453_145359

def magic_8_ball_probability : ℚ := 181440 / 823543

theorem magic_8_ball_theorem (n : ℕ) (k : ℕ) (p : ℚ) :
  n = 7 →
  k = 4 →
  p = 3 / 7 →
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k) = magic_8_ball_probability :=
by sorry

end NUMINAMATH_CALUDE_magic_8_ball_theorem_l1453_145359


namespace NUMINAMATH_CALUDE_third_person_profit_is_800_l1453_145387

/-- Calculates the third person's share of the profit in a joint business investment. -/
def third_person_profit (total_investment : ℕ) (investment_difference : ℕ) (total_profit : ℕ) : ℕ :=
  let first_investment := (total_investment - 3 * investment_difference) / 3
  let second_investment := first_investment + investment_difference
  let third_investment := second_investment + investment_difference
  (third_investment * total_profit) / total_investment

/-- Theorem stating that under the given conditions, the third person's profit share is 800. -/
theorem third_person_profit_is_800 :
  third_person_profit 9000 1000 1800 = 800 := by
  sorry

end NUMINAMATH_CALUDE_third_person_profit_is_800_l1453_145387


namespace NUMINAMATH_CALUDE_locus_of_A_is_ellipse_l1453_145319

/-- Given ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

/-- Right focus of the ellipse -/
def F : ℝ × ℝ := (2, 0)

/-- Point on the ellipse -/
def point_on_ellipse (B : ℝ × ℝ) : Prop := ellipse B.1 B.2

/-- Equilateral triangle property -/
def is_equilateral (A B : ℝ × ℝ) : Prop :=
  let FA := (A.1 - F.1, A.2 - F.2)
  let FB := (B.1 - F.1, B.2 - F.2)
  let AB := (B.1 - A.1, B.2 - A.2)
  FA.1^2 + FA.2^2 = FB.1^2 + FB.2^2 ∧ FA.1^2 + FA.2^2 = AB.1^2 + AB.2^2

/-- Counterclockwise arrangement -/
def is_counterclockwise (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.2 - F.2) - (A.2 - F.2) * (B.1 - F.1) > 0

/-- Locus of point A -/
def locus_A (A : ℝ × ℝ) : Prop :=
  ∃ (B : ℝ × ℝ), point_on_ellipse B ∧ is_equilateral A B ∧ is_counterclockwise A B

/-- Theorem statement -/
theorem locus_of_A_is_ellipse :
  ∀ (A : ℝ × ℝ), locus_A A ↔ 
    (A.1 - 2)^2 + A.2^2 + (A.1)^2 + (A.2 - 2*Real.sqrt 3)^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_locus_of_A_is_ellipse_l1453_145319


namespace NUMINAMATH_CALUDE_sqrt_sum_division_l1453_145305

theorem sqrt_sum_division (x y z : ℝ) : (2 * Real.sqrt 24 + 3 * Real.sqrt 6) / Real.sqrt 3 = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_division_l1453_145305


namespace NUMINAMATH_CALUDE_circle_equation_point_on_circle_l1453_145309

/-- The standard equation of a circle with center (2, -1) passing through (-1, 3) -/
theorem circle_equation : 
  ∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 25 ∧ 
  ∀ (a b : ℝ), (a - 2)^2 + (b + 1)^2 = 25 ↔ (a, b) ∈ {(x, y) | (x - 2)^2 + (y + 1)^2 = 25} :=
by
  sorry

/-- The given point (-1, 3) satisfies the circle equation -/
theorem point_on_circle : (-1 - 2)^2 + (3 + 1)^2 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_point_on_circle_l1453_145309


namespace NUMINAMATH_CALUDE_existence_of_index_l1453_145308

theorem existence_of_index (a : Fin 7 → ℝ) (h1 : a 1 = 0) (h7 : a 7 = 0) :
  ∃ k : Fin 5, (a k) + (a (k + 2)) ≤ (a (k + 1)) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_index_l1453_145308


namespace NUMINAMATH_CALUDE_customers_per_car_l1453_145368

/-- Proves that there are 5 customers in each car given the problem conditions --/
theorem customers_per_car :
  let num_cars : ℕ := 10
  let sports_sales : ℕ := 20
  let music_sales : ℕ := 30
  let total_sales : ℕ := sports_sales + music_sales
  let total_customers : ℕ := total_sales
  let customers_per_car : ℕ := total_customers / num_cars
  customers_per_car = 5 := by
  sorry

end NUMINAMATH_CALUDE_customers_per_car_l1453_145368


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_2028_l1453_145383

def closest_multiple (n : ℕ) (m : ℕ) : ℕ :=
  m * ((n + m / 2) / m)

theorem closest_multiple_of_15_to_2028 :
  closest_multiple 2028 15 = 2025 :=
sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_2028_l1453_145383


namespace NUMINAMATH_CALUDE_intersection_possibilities_l1453_145327

-- Define the sets P and Q
variable (P Q : Set ℕ)

-- Define the function f
def f (t : ℕ) : ℕ := t^2

-- State the theorem
theorem intersection_possibilities (h1 : Q = {1, 4}) 
  (h2 : ∀ t ∈ P, f t ∈ Q) : 
  P ∩ Q = {1} ∨ P ∩ Q = ∅ := by
sorry

end NUMINAMATH_CALUDE_intersection_possibilities_l1453_145327


namespace NUMINAMATH_CALUDE_nested_expression_value_l1453_145377

theorem nested_expression_value : (2*(2*(2*(2*(2*(2+1)+1)+1)+1)+1)+1) = 127 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l1453_145377


namespace NUMINAMATH_CALUDE_total_beads_is_40_l1453_145312

-- Define the number of blue beads
def blue_beads : ℕ := 5

-- Define the number of red beads as twice the number of blue beads
def red_beads : ℕ := 2 * blue_beads

-- Define the number of white beads as the sum of blue and red beads
def white_beads : ℕ := blue_beads + red_beads

-- Define the number of silver beads
def silver_beads : ℕ := 10

-- Theorem: The total number of beads is 40
theorem total_beads_is_40 : 
  blue_beads + red_beads + white_beads + silver_beads = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_beads_is_40_l1453_145312


namespace NUMINAMATH_CALUDE_tan_twenty_seventy_product_is_one_l1453_145374

theorem tan_twenty_seventy_product_is_one :
  Real.tan (20 * π / 180) * Real.tan (70 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_twenty_seventy_product_is_one_l1453_145374


namespace NUMINAMATH_CALUDE_extended_segment_coordinates_l1453_145301

/-- Given points A and B, and a point C on the line extending AB such that BC = 1/2 * AB,
    prove that the coordinates of C are (12, 12). -/
theorem extended_segment_coordinates :
  let A : ℝ × ℝ := (3, 3)
  let B : ℝ × ℝ := (9, 9)
  let C : ℝ × ℝ := (12, 12)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  BC.1 = (1/2) * AB.1 ∧ BC.2 = (1/2) * AB.2 :=
by sorry

end NUMINAMATH_CALUDE_extended_segment_coordinates_l1453_145301


namespace NUMINAMATH_CALUDE_complement_of_P_l1453_145325

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

-- State the theorem
theorem complement_of_P : 
  (Set.univ \ P) = {x : ℝ | x < -1 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_l1453_145325


namespace NUMINAMATH_CALUDE_vector_parallel_implies_k_equals_three_l1453_145392

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ a.1 = c * b.1 ∧ a.2 = c * b.2

/-- Given two vectors a and b, where a depends on k, prove that k = 3 when a and b are parallel -/
theorem vector_parallel_implies_k_equals_three (k : ℝ) :
  let a : ℝ × ℝ := (2 - k, 3)
  let b : ℝ × ℝ := (2, -6)
  parallel a b → k = 3 := by
  sorry


end NUMINAMATH_CALUDE_vector_parallel_implies_k_equals_three_l1453_145392


namespace NUMINAMATH_CALUDE_daragh_initial_bears_l1453_145361

/-- The number of stuffed bears Daragh initially had -/
def initial_bears : ℕ := 20

/-- The number of favorite bears Daragh took out -/
def favorite_bears : ℕ := 8

/-- The number of sisters Daragh divided the remaining bears among -/
def num_sisters : ℕ := 3

/-- The number of bears Eden had before receiving more -/
def eden_bears_before : ℕ := 10

/-- The number of bears Eden had after receiving more -/
def eden_bears_after : ℕ := 14

theorem daragh_initial_bears :
  initial_bears = favorite_bears + (eden_bears_after - eden_bears_before) * num_sisters :=
by sorry

end NUMINAMATH_CALUDE_daragh_initial_bears_l1453_145361


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l1453_145390

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), y < x → ¬(53 ∣ (3 * y.val)^2 + 2 * 41 * (3 * y.val) + 41^2)) ∧
    (53 ∣ (3 * x.val)^2 + 2 * 41 * (3 * x.val) + 41^2) ∧
    x.val = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l1453_145390


namespace NUMINAMATH_CALUDE_coloring_book_problem_l1453_145355

theorem coloring_book_problem (book1 book2 book3 book4 colored : ℕ) 
  (h1 : book1 = 44)
  (h2 : book2 = 35)
  (h3 : book3 = 52)
  (h4 : book4 = 48)
  (h5 : colored = 37) :
  book1 + book2 + book3 + book4 - colored = 142 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_problem_l1453_145355


namespace NUMINAMATH_CALUDE_walking_speed_proof_l1453_145326

def jack_speed (x : ℝ) := x^2 - 11*x - 22
def jill_distance (x : ℝ) := x^2 - 4*x - 12
def jill_time (x : ℝ) := x + 6

theorem walking_speed_proof :
  ∃ (x : ℝ), 
    (jack_speed x = jill_distance x / jill_time x) ∧
    (jack_speed x = 10) :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_proof_l1453_145326


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l1453_145342

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- The origin point -/
def origin : Point := ⟨0, 0⟩

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem parabola_intersection_distance 
  (A B : Point) 
  (hA : A ∈ Parabola) 
  (hB : B ∈ Parabola) 
  (hline : ∃ (m c : ℝ), A.y = m * A.x + c ∧ B.y = m * B.x + c ∧ focus.y = m * focus.x + c) 
  (harea : triangleArea A origin focus = 3 * triangleArea B origin focus) :
  distance A B = 16/3 := 
sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l1453_145342


namespace NUMINAMATH_CALUDE_scientific_notation_of_44_3_million_l1453_145358

theorem scientific_notation_of_44_3_million : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 44300000 = a * (10 : ℝ) ^ n ∧ a = 4.43 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_44_3_million_l1453_145358


namespace NUMINAMATH_CALUDE_line_parabola_tangency_false_l1453_145316

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the concept of a common point
def common_point (p : ℝ) (a b c : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line a b c x y

-- Define the concept of tangency
def is_tangent (p : ℝ) (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, common_point p a b c x y ∧
  ∀ x' y' : ℝ, common_point p a b c x' y' → x' = x ∧ y' = y

-- The theorem to be proved
theorem line_parabola_tangency_false :
  ¬(∀ p a b c : ℝ, (∃! x y : ℝ, common_point p a b c x y) → is_tangent p a b c) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_tangency_false_l1453_145316


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1453_145371

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 1 → (1 / (x - 1) ≥ -1 ↔ x ∈ Set.Ici 1 ∪ Set.Iic 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1453_145371


namespace NUMINAMATH_CALUDE_x_minus_y_is_perfect_square_l1453_145381

theorem x_minus_y_is_perfect_square (x y : ℕ+) 
  (h : 3 * x ^ 2 + x = 4 * y ^ 2 + y) : 
  ∃ (k : ℕ), x - y = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_is_perfect_square_l1453_145381


namespace NUMINAMATH_CALUDE_no_real_solution_l1453_145337

theorem no_real_solution : ¬∃ (x : ℝ), |3*x + 1| + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l1453_145337


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1453_145320

theorem fraction_equals_zero (x : ℝ) : 
  (x - 2) / (1 - x) = 0 → x = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1453_145320


namespace NUMINAMATH_CALUDE_total_coins_after_addition_initial_ratio_final_ratio_l1453_145302

/-- Represents a coin collection with gold and silver coins -/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- The initial state of the coin collection -/
def initial_collection : CoinCollection :=
  { gold := 30, silver := 90 }

/-- The final state of the coin collection after adding 15 gold coins -/
def final_collection : CoinCollection :=
  { gold := initial_collection.gold + 15, silver := initial_collection.silver }

/-- Theorem stating the total number of coins after the addition -/
theorem total_coins_after_addition :
  final_collection.gold + final_collection.silver = 135 := by
  sorry

/-- Theorem for the initial ratio of gold to silver coins -/
theorem initial_ratio :
  initial_collection.gold * 3 = initial_collection.silver := by
  sorry

/-- Theorem for the final ratio of gold to silver coins -/
theorem final_ratio :
  final_collection.gold * 2 = final_collection.silver := by
  sorry

end NUMINAMATH_CALUDE_total_coins_after_addition_initial_ratio_final_ratio_l1453_145302


namespace NUMINAMATH_CALUDE_xiaopang_had_32_books_l1453_145391

/-- The number of books Xiaopang originally had -/
def xiaopang_books : ℕ := 32

/-- The number of books Xiaoya originally had -/
def xiaoya_books : ℕ := 16

/-- Theorem stating that Xiaopang originally had 32 books -/
theorem xiaopang_had_32_books :
  (xiaopang_books - 8 = xiaoya_books + 8) ∧
  (xiaopang_books + 4 = 3 * (xiaoya_books - 4)) →
  xiaopang_books = 32 := by
  sorry

end NUMINAMATH_CALUDE_xiaopang_had_32_books_l1453_145391


namespace NUMINAMATH_CALUDE_cosine_sine_equation_l1453_145317

theorem cosine_sine_equation (n : ℕ) :
  (∀ k : ℤ, (Real.cos (2 * k * Real.pi)) ^ n - (Real.sin (2 * k * Real.pi)) ^ n = 1) ∧
  (Even n → ∀ k : ℤ, (Real.cos ((2 * k + 1) * Real.pi)) ^ n - (Real.sin ((2 * k + 1) * Real.pi)) ^ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_cosine_sine_equation_l1453_145317


namespace NUMINAMATH_CALUDE_segment_ratios_l1453_145352

/-- Given line segments AC, AB, and BC, where AB consists of 3 parts and BC consists of 4 parts,
    prove the ratios of AB:AC and BC:AC. -/
theorem segment_ratios (AC AB BC : ℝ) (h1 : AB = 3) (h2 : BC = 4) (h3 : AC = AB + BC) :
  (AB / AC = 3 / 7) ∧ (BC / AC = 4 / 7) := by
  sorry

end NUMINAMATH_CALUDE_segment_ratios_l1453_145352


namespace NUMINAMATH_CALUDE_scale_heights_theorem_l1453_145324

theorem scale_heights_theorem (n : ℕ) (adults children : Fin n → ℝ) 
  (h : ∀ i : Fin n, adults i > children i) :
  ∃ (scales : Fin n → ℕ+), 
    (∀ i j : Fin n, (scales i : ℝ) * adults i > (scales j : ℝ) * children j) := by
  sorry

end NUMINAMATH_CALUDE_scale_heights_theorem_l1453_145324


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l1453_145395

theorem expansion_coefficient_sum (n : ℕ) : 
  (∀ a b : ℝ, (3*a + 5*b)^n = 2^15) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l1453_145395


namespace NUMINAMATH_CALUDE_induction_even_numbers_l1453_145370

theorem induction_even_numbers (P : ℕ → Prop) (k : ℕ) (h_k_even : Even k) (h_k_ge_2 : k ≥ 2) 
  (h_base : P 2) (h_inductive : ∀ m : ℕ, m ≥ 2 → Even m → P m → P (m + 2)) :
  (P k → P (k + 2)) ∧ ¬(P k → P (k + 1)) ∧ ¬(P k → P (2*k + 2)) ∧ ¬(P k → P (2*(k + 2))) :=
sorry

end NUMINAMATH_CALUDE_induction_even_numbers_l1453_145370


namespace NUMINAMATH_CALUDE_particular_number_multiplication_l1453_145332

theorem particular_number_multiplication (x : ℤ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_multiplication_l1453_145332


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2047_l1453_145396

theorem tens_digit_of_13_pow_2047 : ∃ n : ℕ, 13^2047 ≡ 10 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2047_l1453_145396


namespace NUMINAMATH_CALUDE_virus_length_scientific_notation_l1453_145307

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem virus_length_scientific_notation :
  toScientificNotation 0.00000032 = ScientificNotation.mk 3.2 (-7) :=
sorry

end NUMINAMATH_CALUDE_virus_length_scientific_notation_l1453_145307


namespace NUMINAMATH_CALUDE_max_problems_solved_l1453_145339

theorem max_problems_solved (n : ℕ) (avg : ℕ) (h1 : n = 25) (h2 : avg = 6) :
  ∃ (max : ℕ), max = 126 ∧
  ∀ (problems : Fin n → ℕ),
  (∀ i, problems i ≥ 1) →
  (Finset.sum Finset.univ problems = n * avg) →
  ∀ i, problems i ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_problems_solved_l1453_145339


namespace NUMINAMATH_CALUDE_soccer_ball_max_height_l1453_145397

/-- The height of the soccer ball as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- Theorem stating that the maximum height reached by the soccer ball is 40 feet -/
theorem soccer_ball_max_height :
  ∃ (max : ℝ), max = 40 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end NUMINAMATH_CALUDE_soccer_ball_max_height_l1453_145397


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l1453_145331

theorem chocolate_box_problem (total : ℕ) (remaining : ℕ) : 
  (remaining = 28) →
  (total / 2 * 4 / 5 + total / 2 / 2 = total - remaining) →
  total = 80 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l1453_145331


namespace NUMINAMATH_CALUDE_product_of_distinct_roots_l1453_145380

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_roots_l1453_145380


namespace NUMINAMATH_CALUDE_our_circle_center_and_radius_l1453_145323

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle -/
def Circle.radius (c : Circle) : ℝ := sorry

/-- Our specific circle -/
def our_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*x - 3 = 0 }

theorem our_circle_center_and_radius :
  Circle.center our_circle = (1, 0) ∧ Circle.radius our_circle = 2 := by
  sorry

end NUMINAMATH_CALUDE_our_circle_center_and_radius_l1453_145323


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_l1453_145318

/-- Theorem: All parabolas of the form y = 4x^2 + 2tx - 3t pass through the point (3, 36) for any real t. -/
theorem fixed_point_of_parabola (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + 2 * t * x - 3 * t
  f 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_l1453_145318
