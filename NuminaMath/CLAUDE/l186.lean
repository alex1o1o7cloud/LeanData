import Mathlib

namespace NUMINAMATH_CALUDE_not_in_range_iff_a_in_interval_l186_18685

/-- The function g(x) defined as x^2 + ax + 3 -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem stating that -3 is not in the range of g(x) if and only if a is in the open interval (-√24, √24) -/
theorem not_in_range_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, g a x ≠ -3) ↔ a ∈ Set.Ioo (-Real.sqrt 24) (Real.sqrt 24) :=
sorry

end NUMINAMATH_CALUDE_not_in_range_iff_a_in_interval_l186_18685


namespace NUMINAMATH_CALUDE_final_books_count_l186_18623

/-- Represents the daily transactions in the library --/
structure DailyTransactions where
  checkouts : ℕ
  returns : ℕ
  renewals : ℕ := 0
  newBooks : ℕ := 0
  damaged : ℕ := 0
  misplaced : ℕ := 0

/-- Calculates the number of available books at the end of the week --/
def availableBooksAtEndOfWeek (initialBooks : ℕ) (monday tuesday wednesday thursday friday : DailyTransactions) : ℕ :=
  let mondayEnd := initialBooks - monday.checkouts + monday.returns
  let tuesdayEnd := mondayEnd - tuesday.checkouts + tuesday.returns + tuesday.newBooks
  let wednesdayEnd := tuesdayEnd - wednesday.checkouts + wednesday.returns - wednesday.damaged
  let thursdayEnd := wednesdayEnd - thursday.checkouts + thursday.returns
  let fridayEnd := thursdayEnd - friday.checkouts + friday.returns
  fridayEnd

/-- Theorem: Given the initial number of books and daily transactions, the final number of books available for checkout at the end of the week is 76 --/
theorem final_books_count (initialBooks : ℕ) (monday tuesday wednesday thursday friday : DailyTransactions) :
  initialBooks = 98 →
  monday = { checkouts := 43, returns := 23, renewals := 5 } →
  tuesday = { checkouts := 28, returns := 0, newBooks := 35 } →
  wednesday = { checkouts := 52, returns := 40, damaged := 3 } →
  thursday = { checkouts := 37, returns := 22 } →
  friday = { checkouts := 29, returns := 50, misplaced := 4 } →
  availableBooksAtEndOfWeek initialBooks monday tuesday wednesday thursday friday = 76 :=
by
  sorry

end NUMINAMATH_CALUDE_final_books_count_l186_18623


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l186_18616

/-- The function f(x) = 3 + a^(x-1) always passes through the point (1, 4) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f := λ x : ℝ => 3 + a^(x - 1)
  f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l186_18616


namespace NUMINAMATH_CALUDE_number_division_problem_l186_18612

theorem number_division_problem :
  ∃ x : ℝ, x / 5 = 30 + x / 6 ∧ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l186_18612


namespace NUMINAMATH_CALUDE_direct_proportion_implies_b_value_l186_18649

/-- A function f is directly proportional if there exists a non-zero constant k such that f(x) = k * x for all x. -/
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function y = x + 2 - 3b -/
def f (b : ℝ) : ℝ → ℝ := λ x ↦ x + 2 - 3 * b

theorem direct_proportion_implies_b_value :
  ∀ b : ℝ, is_directly_proportional (f b) → b = 2 / 3 :=
by
  sorry

#check direct_proportion_implies_b_value

end NUMINAMATH_CALUDE_direct_proportion_implies_b_value_l186_18649


namespace NUMINAMATH_CALUDE_rectangle_area_is_twelve_l186_18618

/-- Represents a rectangle with given properties -/
structure Rectangle where
  width : ℝ
  length : ℝ
  diagonal : ℝ
  perimeter : ℝ
  length_eq : length = 3 * width
  perimeter_eq : perimeter = 2 * (length + width)
  diagonal_eq : diagonal^2 = width^2 + length^2

/-- The area of a rectangle with specific properties is 12 -/
theorem rectangle_area_is_twelve (rect : Rectangle) (h : rect.perimeter = 16) : 
  rect.width * rect.length = 12 := by
  sorry

#check rectangle_area_is_twelve

end NUMINAMATH_CALUDE_rectangle_area_is_twelve_l186_18618


namespace NUMINAMATH_CALUDE_stating_dieRollSumWays_l186_18671

/-- Represents the number of faces on a standard die -/
def diefaces : ℕ := 6

/-- Represents the number of times the die is rolled -/
def numrolls : ℕ := 6

/-- Represents the target sum we're aiming for -/
def targetsum : ℕ := 21

/-- 
Calculates the number of ways to roll a fair six-sided die 'numrolls' times 
such that the sum of the outcomes is 'targetsum'
-/
def numWaysToSum (diefaces numrolls targetsum : ℕ) : ℕ := sorry

/-- 
Theorem stating that the number of ways to roll a fair six-sided die six times 
such that the sum of the outcomes is 21 is equal to 15504
-/
theorem dieRollSumWays : numWaysToSum diefaces numrolls targetsum = 15504 := by sorry

end NUMINAMATH_CALUDE_stating_dieRollSumWays_l186_18671


namespace NUMINAMATH_CALUDE_salary_increase_proof_l186_18675

/-- Proves that given the conditions of the salary increase problem, the new salary is $90,000 -/
theorem salary_increase_proof (S : ℝ) 
  (h1 : S + 25000 = S * (1 + 0.3846153846153846)) : S + 25000 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l186_18675


namespace NUMINAMATH_CALUDE_four_balls_three_boxes_l186_18615

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 8 ways to put 4 distinguishable balls into 3 indistinguishable boxes -/
theorem four_balls_three_boxes : ways_to_put_balls_in_boxes 4 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_balls_three_boxes_l186_18615


namespace NUMINAMATH_CALUDE_square_difference_49_50_l186_18613

theorem square_difference_49_50 : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end NUMINAMATH_CALUDE_square_difference_49_50_l186_18613


namespace NUMINAMATH_CALUDE_largest_number_l186_18634

theorem largest_number : 
  let numbers : List ℝ := [0.935, 0.9401, 0.9349, 0.9041, 0.9400]
  ∀ x ∈ numbers, x ≤ 0.9401 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l186_18634


namespace NUMINAMATH_CALUDE_oranges_picked_theorem_l186_18687

/-- The total number of oranges picked over three days --/
def total_oranges (monday : ℕ) (tuesday_multiplier : ℕ) (wednesday : ℕ) : ℕ :=
  monday + tuesday_multiplier * monday + wednesday

/-- Theorem: Given the conditions, the total number of oranges picked is 470 --/
theorem oranges_picked_theorem (monday : ℕ) (tuesday_multiplier : ℕ) (wednesday : ℕ)
  (h1 : monday = 100)
  (h2 : tuesday_multiplier = 3)
  (h3 : wednesday = 70) :
  total_oranges monday tuesday_multiplier wednesday = 470 := by
  sorry

end NUMINAMATH_CALUDE_oranges_picked_theorem_l186_18687


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l186_18661

/-- The number of dogwood trees in the park after planting and removal operations --/
def final_tree_count (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) 
                     (removed_today : ℕ) (workers : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow - removed_today

theorem dogwood_tree_count : 
  let initial_trees := 7
  let planted_today := 5
  let planted_tomorrow := 4
  let removed_today := 3
  let workers := 8
  final_tree_count initial_trees planted_today planted_tomorrow removed_today workers = 13 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l186_18661


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l186_18677

def shorts_price : ℝ := 15
def jacket_price : ℝ := 14.82
def total_spent : ℝ := 42.33

theorem shirt_price_calculation : 
  ∃ (shirt_price : ℝ), shirt_price = total_spent - (shorts_price + jacket_price) ∧ shirt_price = 12.51 :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l186_18677


namespace NUMINAMATH_CALUDE_prime_even_intersection_l186_18645

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def P : Set ℕ := {n : ℕ | isPrime n}
def Q : Set ℕ := {n : ℕ | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l186_18645


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l186_18600

theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ := 2 * x - 4
  let a₂ := 3 * x + 2
  let a₃ := 5 * x - 1
  (a₂ - a₁ = a₃ - a₂) → x = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l186_18600


namespace NUMINAMATH_CALUDE_right_triangle_equality_l186_18620

/-- For a right triangle with sides a and b, and hypotenuse c, 
    the equation √(a^2 + b^2) = a + b is true if and only if 
    the angle θ between sides a and b is 90°. -/
theorem right_triangle_equality (a b c : ℝ) (θ : Real) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : c^2 = a^2 + b^2) -- Pythagorean theorem
  (h5 : θ = Real.arccos (b / c)) -- Definition of θ
  : Real.sqrt (a^2 + b^2) = a + b ↔ θ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_equality_l186_18620


namespace NUMINAMATH_CALUDE_customized_packaging_combinations_l186_18611

def wrapping_papers : ℕ := 10
def ribbon_colors : ℕ := 5
def gift_tag_styles : ℕ := 6

theorem customized_packaging_combinations : 
  wrapping_papers * ribbon_colors * gift_tag_styles = 300 := by
  sorry

end NUMINAMATH_CALUDE_customized_packaging_combinations_l186_18611


namespace NUMINAMATH_CALUDE_max_value_inequality_l186_18617

theorem max_value_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  x * y * z * (x + y + z + w) / ((x + y + z)^2 * (y + z + w)^2) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l186_18617


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_l186_18689

theorem sum_of_squares_representation (n m : ℕ) :
  ∃ (x y : ℕ), (2014^2 + 2016^2) / 2 = x^2 + y^2 ∧
  ∃ (a b : ℕ), (4*n^2 + 4*m^2) / 2 = a^2 + b^2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_l186_18689


namespace NUMINAMATH_CALUDE_four_identical_differences_l186_18663

theorem four_identical_differences (S : Finset ℕ) : 
  S.card = 20 → (∀ n ∈ S, n < 70) → 
  ∃ (d : ℕ) (a b c d e f g h : ℕ), 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
    a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h ∧
    b - a = d - c ∧ d - c = f - e ∧ f - e = h - g ∧ h - g = d :=
by sorry

end NUMINAMATH_CALUDE_four_identical_differences_l186_18663


namespace NUMINAMATH_CALUDE_bob_weight_l186_18679

theorem bob_weight (j b : ℝ) 
  (h1 : j + b = 210)
  (h2 : b - j = b / 3)
  : b = 126 := by
  sorry

end NUMINAMATH_CALUDE_bob_weight_l186_18679


namespace NUMINAMATH_CALUDE_staff_age_l186_18629

theorem staff_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 32 →
  student_avg_age = 16 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (num_students + 1 : ℝ) * new_avg_age = (num_students + 1 : ℝ) * 49 := by
  sorry

end NUMINAMATH_CALUDE_staff_age_l186_18629


namespace NUMINAMATH_CALUDE_escalator_steps_l186_18622

/-- The number of steps counted by the slower person -/
def walker_count : ℕ := 50

/-- The number of steps counted by the faster person -/
def trotman_count : ℕ := 75

/-- The speed ratio between the faster and slower person -/
def speed_ratio : ℕ := 3

/-- The number of visible steps on the stopped escalator -/
def visible_steps : ℕ := 100

/-- Theorem stating that the number of visible steps on the stopped escalator is 100 -/
theorem escalator_steps :
  ∀ (v : ℚ), v > 0 →
  walker_count + walker_count / v = trotman_count + trotman_count / (speed_ratio * v) →
  visible_steps = walker_count + walker_count / v :=
by sorry

end NUMINAMATH_CALUDE_escalator_steps_l186_18622


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l186_18696

theorem opposite_of_negative_fraction (m : ℚ) : 
  m = -(-(-(1 / 3))) → m = -(1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l186_18696


namespace NUMINAMATH_CALUDE_fraction_comparison_l186_18656

theorem fraction_comparison : (291 : ℚ) / 730 > 29 / 73 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l186_18656


namespace NUMINAMATH_CALUDE_rectangle_properties_l186_18638

/-- A rectangle with one side of length 8 and another of length x -/
structure Rectangle where
  x : ℝ
  h_positive : x > 0

/-- The perimeter of the rectangle -/
def perimeter (rect : Rectangle) : ℝ := 2 * (8 + rect.x)

/-- The area of the rectangle -/
def area (rect : Rectangle) : ℝ := 8 * rect.x

theorem rectangle_properties (rect : Rectangle) :
  (perimeter rect = 16 + 2 * rect.x) ∧
  (area rect = 8 * rect.x) ∧
  (area rect = 80 → perimeter rect = 36) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_properties_l186_18638


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l186_18695

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  ab : ℝ
  bc : ℝ
  ac : ℝ
  right_triangle : ab^2 + bc^2 = ac^2
  ab_eq : ab = 6
  bc_eq : bc = 8
  ac_eq : ac = 10

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.ac
  on_ab : side_length ≤ t.ab
  on_bc : side_length ≤ t.bc

/-- The theorem stating that the side length of the inscribed square is 120/37 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 120 / 37 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l186_18695


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l186_18603

/-- Given a parabola y² = -2x with focus F, and a point A(x₀, y₀) on the parabola,
    if |AF| = 3/2, then x₀ = -1 -/
theorem parabola_focus_distance (x₀ y₀ : ℝ) :
  y₀^2 = -2*x₀ →  -- A is on the parabola
  ∃ F : ℝ × ℝ, (F.1 = 1/2 ∧ F.2 = 0) →  -- Focus coordinates
  (x₀ - F.1)^2 + (y₀ - F.2)^2 = (3/2)^2 →  -- |AF| = 3/2
  x₀ = -1 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l186_18603


namespace NUMINAMATH_CALUDE_thermal_equilibrium_problem_l186_18642

/-- Represents the thermal equilibrium in a system of water and metal bars -/
structure ThermalSystem where
  initialWaterTemp : ℝ
  initialBarTemp : ℝ
  firstEquilibriumTemp : ℝ
  finalEquilibriumTemp : ℝ

/-- The thermal equilibrium problem -/
theorem thermal_equilibrium_problem (system : ThermalSystem)
  (h1 : system.initialWaterTemp = 100)
  (h2 : system.initialBarTemp = 20)
  (h3 : system.firstEquilibriumTemp = 80)
  : system.finalEquilibriumTemp = 68 := by
  sorry

end NUMINAMATH_CALUDE_thermal_equilibrium_problem_l186_18642


namespace NUMINAMATH_CALUDE_divisible_by_36_sum_6_l186_18648

/-- Represents a 7-digit number in the form 457q89f -/
def number (q f : Nat) : Nat :=
  457000 + q * 1000 + 89 * 10 + f

/-- Predicate to check if two natural numbers are distinct digits -/
def distinct_digits (a b : Nat) : Prop :=
  a ≠ b ∧ a < 10 ∧ b < 10

theorem divisible_by_36_sum_6 (q f : Nat) :
  distinct_digits q f →
  number q f % 36 = 0 →
  q + f = 6 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_36_sum_6_l186_18648


namespace NUMINAMATH_CALUDE_triangle_segment_length_l186_18607

theorem triangle_segment_length : 
  ∀ (a b c h x : ℝ),
  a = 40 ∧ b = 90 ∧ c = 100 →
  a^2 = x^2 + h^2 →
  b^2 = (c - x)^2 + h^2 →
  c - x = 82.5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l186_18607


namespace NUMINAMATH_CALUDE_ellipse_equation_eccentricity_range_l186_18631

noncomputable section

-- Define the ellipse parameters
def m : ℝ := 1  -- We know m = 1 from the solution, but we keep it as a parameter

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (m, 0)

-- Define the directrices
def left_directrix (x : ℝ) : Prop := x = -m - 1
def right_directrix (x : ℝ) : Prop := x = m + 1

-- Define the line y = x
def diagonal_line (x y : ℝ) : Prop := y = x

-- Define points A and B
def point_A : ℝ × ℝ := (-m - 1, -m - 1)
def point_B : ℝ × ℝ := (m + 1, m + 1)

-- Define vectors AF and FB
def vector_AF : ℝ × ℝ := (2*m + 1, m + 1)
def vector_FB : ℝ × ℝ := (1, m + 1)

-- Define dot product of AF and FB
def dot_product_AF_FB : ℝ := (2*m + 1) * 1 + (m + 1) * (m + 1)

-- Define eccentricity
def eccentricity : ℝ := 1 / Real.sqrt (1 + 1/m)

-- Theorem 1: Prove the equation of the ellipse
theorem ellipse_equation : 
  ∀ x y : ℝ, ellipse x y ↔ x^2 / 2 + y^2 = 1 :=
sorry

-- Theorem 2: Prove the range of eccentricity
theorem eccentricity_range :
  dot_product_AF_FB < 7 → 0 < eccentricity ∧ eccentricity < Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_eccentricity_range_l186_18631


namespace NUMINAMATH_CALUDE_cofactor_sum_l186_18681

/-- The algebraic cofactor function of the element 7 in the given determinant -/
def f (x a : ℝ) : ℝ := -x^2 - a*x + 2

/-- The theorem stating that if the solution set of f(x) > 0 is (-1, b), then a + b = 1 -/
theorem cofactor_sum (a b : ℝ) : 
  (∀ x, f x a > 0 ↔ -1 < x ∧ x < b) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_cofactor_sum_l186_18681


namespace NUMINAMATH_CALUDE_cards_given_away_l186_18686

theorem cards_given_away (original_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : original_cards = 350) 
  (h2 : remaining_cards = 248) : 
  original_cards - remaining_cards = 102 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_away_l186_18686


namespace NUMINAMATH_CALUDE_prime_not_divides_difference_l186_18640

theorem prime_not_divides_difference (a b c d p : ℕ) : 
  0 < a → 0 < b → 0 < c → 0 < d → 
  p = a + b + c + d → 
  Nat.Prime p → 
  ¬(p ∣ a * b - c * d) := by
sorry

end NUMINAMATH_CALUDE_prime_not_divides_difference_l186_18640


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l186_18626

theorem simplified_fraction_ratio (k : ℝ) : 
  let original := (6 * k + 12) / 6
  let simplified := k + 2
  ∃ (a b : ℤ), (simplified = a * k + b) ∧ (a / b = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l186_18626


namespace NUMINAMATH_CALUDE_circus_crowns_l186_18669

theorem circus_crowns (total_feathers : ℕ) (feathers_per_crown : ℕ) (h1 : total_feathers = 6538) (h2 : feathers_per_crown = 7) :
  total_feathers / feathers_per_crown = 934 := by
  sorry

end NUMINAMATH_CALUDE_circus_crowns_l186_18669


namespace NUMINAMATH_CALUDE_same_result_different_parentheses_l186_18606

-- Define the exponentiation operation
def power (a b : ℕ) : ℕ := a ^ b

-- Define the two different parenthesization methods
def method1 (n : ℕ) : ℕ := power (power n 7) (power 7 7)
def method2 (n : ℕ) : ℕ := power (power n (power 7 7)) 7

-- Theorem statement
theorem same_result_different_parentheses :
  ∃ (n : ℕ), method1 n = method2 n :=
sorry

end NUMINAMATH_CALUDE_same_result_different_parentheses_l186_18606


namespace NUMINAMATH_CALUDE_additional_sugar_needed_l186_18688

/-- The amount of additional sugar needed for a cake -/
theorem additional_sugar_needed (total_required sugar_available : ℕ) : 
  total_required = 450 → sugar_available = 287 → total_required - sugar_available = 163 := by
  sorry

end NUMINAMATH_CALUDE_additional_sugar_needed_l186_18688


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_property_l186_18673

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

-- State the theorem
theorem isosceles_right_triangle_property 
  (a : ℝ) (x₁ x₂ : ℝ) (t : ℝ) : 
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  (∃ x₀, x₁ < x₀ ∧ x₀ < x₂ ∧ 
    (x₂ - x₁) / 2 = -f a x₀ ∧
    (x₂ - x₀) = (x₀ - x₁)) →
  Real.sqrt ((x₂ - 1) / (x₁ - 1)) = t →
  a * t - (a + t) = 1 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_property_l186_18673


namespace NUMINAMATH_CALUDE_prism_volume_l186_18659

/-- The volume of a right rectangular prism with face areas 10, 15, and 18 square inches is 30√3 cubic inches. -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 10) (h2 : w * h = 15) (h3 : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l186_18659


namespace NUMINAMATH_CALUDE_campers_rowing_in_morning_l186_18682

theorem campers_rowing_in_morning (afternoon_campers : ℕ) (difference : ℕ) 
  (h1 : afternoon_campers = 61)
  (h2 : afternoon_campers = difference + morning_campers) 
  (h3 : difference = 9) : 
  morning_campers = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_in_morning_l186_18682


namespace NUMINAMATH_CALUDE_slope_of_specific_midpoint_line_l186_18624

/-- The slope of the line connecting the midpoints of two line segments -/
def slope_of_midpoint_line (x1 y1 x2 y2 x3 y3 x4 y4 : ℚ) : ℚ :=
  let m1x := (x1 + x2) / 2
  let m1y := (y1 + y2) / 2
  let m2x := (x3 + x4) / 2
  let m2y := (y3 + y4) / 2
  (m2y - m1y) / (m2x - m1x)

/-- Theorem: The slope of the line connecting the midpoints of the given segments is -1 -/
theorem slope_of_specific_midpoint_line :
  slope_of_midpoint_line 3 4 7 8 6 2 9 5 = -1 := by sorry

end NUMINAMATH_CALUDE_slope_of_specific_midpoint_line_l186_18624


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l186_18683

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_sum : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l186_18683


namespace NUMINAMATH_CALUDE_no_perfect_square_9999_xxxx_l186_18652

theorem no_perfect_square_9999_xxxx : 
  ¬ ∃ x : ℕ, 99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ y : ℕ, x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_9999_xxxx_l186_18652


namespace NUMINAMATH_CALUDE_diamond_count_l186_18608

/-- The number of rubies in the chest -/
def rubies : ℕ := 377

/-- The difference between the number of diamonds and rubies -/
def diamond_ruby_difference : ℕ := 44

/-- The number of diamonds in the chest -/
def diamonds : ℕ := rubies + diamond_ruby_difference

theorem diamond_count : diamonds = 421 := by
  sorry

end NUMINAMATH_CALUDE_diamond_count_l186_18608


namespace NUMINAMATH_CALUDE_inequality_subtraction_l186_18662

theorem inequality_subtraction (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_subtraction_l186_18662


namespace NUMINAMATH_CALUDE_remainder_theorem_l186_18654

def f (x : ℝ) : ℝ := 5*x^7 - 3*x^6 - 8*x^5 + 3*x^3 + 5*x^2 - 20

def g (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = fun x ↦ g x * q x + 6910 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l186_18654


namespace NUMINAMATH_CALUDE_total_hike_length_l186_18676

/-- The length of a hike given the distance hiked on the first day and the remaining distance. -/
def hike_length (first_day_distance : ℕ) (remaining_distance : ℕ) : ℕ :=
  first_day_distance + remaining_distance

/-- Theorem stating that the total length of the hike is 36 miles. -/
theorem total_hike_length :
  hike_length 9 27 = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_hike_length_l186_18676


namespace NUMINAMATH_CALUDE_binary_operation_equality_l186_18655

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation as a list of bits. -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_bits (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
    to_bits n

/-- The first binary number in the problem: 110110₂ -/
def num1 : List Bool := [true, true, false, true, true, false]

/-- The second binary number in the problem: 101010₂ -/
def num2 : List Bool := [true, false, true, false, true, false]

/-- The divisor in the problem: 100₂ -/
def divisor : List Bool := [true, false, false]

/-- The expected result: 111001101100₂ -/
def expected_result : List Bool := [true, true, true, false, false, true, true, false, true, true, false, false]

/-- Theorem stating the equality of the binary operation and the expected result -/
theorem binary_operation_equality :
  nat_to_binary ((binary_to_nat num1 * binary_to_nat num2) / binary_to_nat divisor) = expected_result :=
sorry

end NUMINAMATH_CALUDE_binary_operation_equality_l186_18655


namespace NUMINAMATH_CALUDE_second_triangle_side_length_l186_18604

/-- Given a sequence of equilateral triangles where each triangle is formed by joining
    the midpoints of the sides of the previous triangle, if the first triangle has sides
    of 80 cm and the sum of all triangle perimeters is 480 cm, then the side length of
    the second triangle is 40 cm. -/
theorem second_triangle_side_length
  (first_triangle_side : ℝ)
  (total_perimeter : ℝ)
  (h1 : first_triangle_side = 80)
  (h2 : total_perimeter = 480)
  (h3 : total_perimeter = (3 * first_triangle_side) / (1 - 1/2)) :
  first_triangle_side / 2 = 40 :=
sorry

end NUMINAMATH_CALUDE_second_triangle_side_length_l186_18604


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l186_18628

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l186_18628


namespace NUMINAMATH_CALUDE_quadratic_roots_l186_18601

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop :=
  (a - 1) * x^2 + 2 * x + a - 1 = 0

theorem quadratic_roots :
  -- Part 1
  (∃ a : ℝ, quadratic_equation a 2 ∧ 
    ∃ x : ℝ, x ≠ 2 ∧ quadratic_equation a x) →
  (quadratic_equation (1/5) 2 ∧ quadratic_equation (1/5) (1/2)) ∧
  -- Part 2
  (∃ x : ℝ, quadratic_equation 1 x ↔ x = 0) ∧
  (∃ x : ℝ, quadratic_equation 2 x ↔ x = -1) ∧
  (∃ x : ℝ, quadratic_equation 0 x ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l186_18601


namespace NUMINAMATH_CALUDE_yard_area_l186_18625

/-- The area of a rectangular yard with two cut-out areas -/
theorem yard_area (yard_length yard_width square_side rectangle_length rectangle_width : ℕ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 18)
  (h3 : square_side = 3)
  (h4 : rectangle_length = 4)
  (h5 : rectangle_width = 2) :
  yard_length * yard_width - (square_side * square_side + rectangle_length * rectangle_width) = 343 := by
  sorry

#check yard_area

end NUMINAMATH_CALUDE_yard_area_l186_18625


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l186_18691

-- Define the interval [1,2]
def I : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 2 }

-- Define the proposition
def P (a : ℝ) : Prop := ∀ x ∈ I, x^2 - a ≤ 0

-- Define the sufficient condition
def S (a : ℝ) : Prop := a ≥ 5

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ a, S a → P a) ∧ (∃ a, P a ∧ ¬S a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l186_18691


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l186_18698

theorem r_value_when_n_is_3 (n : ℕ) (s r : ℕ) 
  (h1 : s = 3^n - 1) 
  (h2 : r = 3^s + s) 
  (h3 : n = 3) : 
  r = 3^26 + 26 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l186_18698


namespace NUMINAMATH_CALUDE_number_of_points_l186_18619

theorem number_of_points (initial_sum : ℝ) (shift : ℝ) (final_sum : ℝ) : 
  initial_sum = -1.5 → 
  shift = -2 → 
  final_sum = -15.5 → 
  (final_sum - initial_sum) / shift = 7 := by
sorry

end NUMINAMATH_CALUDE_number_of_points_l186_18619


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l186_18664

/-- Given a real number a, f(x) = ax - ln x, and l is the tangent line to f at (1, f(1)),
    prove that the y-intercept of l is 1. -/
theorem tangent_line_y_intercept (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x - Real.log x
  let f' : ℝ → ℝ := λ x => a - 1 / x
  let slope : ℝ := f' 1
  let point : ℝ × ℝ := (1, f 1)
  let l : ℝ → ℝ := λ x => slope * (x - point.1) + point.2
  l 0 = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l186_18664


namespace NUMINAMATH_CALUDE_polynomial_expansion_l186_18665

theorem polynomial_expansion (z : ℝ) :
  (3 * z^2 + 4 * z - 5) * (4 * z^3 - 3 * z^2 + 2) =
  12 * z^5 + 7 * z^4 - 26 * z^3 + 21 * z^2 + 8 * z - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l186_18665


namespace NUMINAMATH_CALUDE_inequality_subset_l186_18627

/-- The solution set of the system of inequalities is a subset of 2x^2 - 9x + a < 0 iff a ≤ 9 -/
theorem inequality_subset (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + a < 0) ↔ a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_subset_l186_18627


namespace NUMINAMATH_CALUDE_square_area_equals_perimeter_l186_18641

theorem square_area_equals_perimeter (s : ℝ) (h : s > 0) : s^2 = 4*s → s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_perimeter_l186_18641


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_is_17_l186_18660

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

/-- The smallest positive integer greater than 10 that is a palindrome in both base 2 and base 4 -/
def smallestDualPalindrome : ℕ := 17

theorem smallest_dual_palindrome_is_17 :
  (smallestDualPalindrome > 10) ∧
  (isPalindrome smallestDualPalindrome 2) ∧
  (isPalindrome smallestDualPalindrome 4) ∧
  (∀ n : ℕ, n > 10 ∧ n < smallestDualPalindrome →
    ¬(isPalindrome n 2 ∧ isPalindrome n 4)) :=
by sorry

#eval smallestDualPalindrome

end NUMINAMATH_CALUDE_smallest_dual_palindrome_is_17_l186_18660


namespace NUMINAMATH_CALUDE_inequality_proof_l186_18602

theorem inequality_proof (x y : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l186_18602


namespace NUMINAMATH_CALUDE_find_number_l186_18644

theorem find_number : ∃ x : ℝ, (((x - 1.9) * 1.5 + 32) / 2.5) = 20 ∧ x = 13.9 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l186_18644


namespace NUMINAMATH_CALUDE_city_population_problem_l186_18680

theorem city_population_problem (p : ℝ) : 
  (0.85 * (p + 1500) = p - 45) → p = 8800 := by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l186_18680


namespace NUMINAMATH_CALUDE_carla_cooking_time_l186_18643

/-- Represents the cooking time for each item in minutes -/
structure CookingTime where
  waffle : ℕ
  steak : ℕ
  chili : ℕ

/-- Represents the number of items to be cooked -/
structure CookingItems where
  waffle : ℕ
  steak : ℕ
  chili : ℕ

/-- Calculates the total cooking time given the cooking times and items to be cooked -/
def totalCookingTime (time : CookingTime) (items : CookingItems) : ℕ :=
  time.waffle * items.waffle + time.steak * items.steak + time.chili * items.chili

/-- Theorem stating that Carla's total cooking time is 100 minutes -/
theorem carla_cooking_time :
  let time := CookingTime.mk 10 6 20
  let items := CookingItems.mk 3 5 2
  totalCookingTime time items = 100 := by sorry

end NUMINAMATH_CALUDE_carla_cooking_time_l186_18643


namespace NUMINAMATH_CALUDE_rod_length_l186_18635

theorem rod_length (pieces : ℝ) (piece_length : ℝ) (h1 : pieces = 118.75) (h2 : piece_length = 0.40) :
  pieces * piece_length = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_l186_18635


namespace NUMINAMATH_CALUDE_soda_consumption_l186_18658

theorem soda_consumption (carol_soda bob_soda : ℝ) 
  (h1 : carol_soda = 20)
  (h2 : bob_soda = carol_soda * 1.25)
  (h3 : carol_soda ≥ 0)
  (h4 : bob_soda ≥ 0) :
  ∃ (transfer : ℝ),
    0 ≤ transfer ∧
    transfer ≤ bob_soda * 0.2 ∧
    carol_soda * 0.8 + transfer = bob_soda * 0.8 - transfer ∧
    carol_soda * 0.8 + transfer + (bob_soda * 0.8 - transfer) = 36 :=
by sorry

end NUMINAMATH_CALUDE_soda_consumption_l186_18658


namespace NUMINAMATH_CALUDE_good_number_implies_prime_l186_18630

/-- A positive integer b is "good for a" if C(an, b) - 1 is divisible by an + 1 for all positive integers n such that an ≥ b -/
def is_good_for (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, a * n ≥ b → (Nat.choose (a * n) b - 1) % (a * n + 1) = 0

theorem good_number_implies_prime (a b : ℕ+) 
  (h1 : is_good_for a b)
  (h2 : ¬ is_good_for a (b + 2)) :
  Nat.Prime (b + 1) :=
sorry

end NUMINAMATH_CALUDE_good_number_implies_prime_l186_18630


namespace NUMINAMATH_CALUDE_barium_chloride_weight_l186_18670

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The number of moles of barium chloride -/
def moles_BaCl2 : ℝ := 4

/-- The molecular weight of barium chloride in g/mol -/
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

/-- The total weight of barium chloride in grams -/
def total_weight_BaCl2 : ℝ := moles_BaCl2 * molecular_weight_BaCl2

theorem barium_chloride_weight :
  total_weight_BaCl2 = 832.92 := by sorry

end NUMINAMATH_CALUDE_barium_chloride_weight_l186_18670


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l186_18639

/-- An arithmetic sequence with common difference d and special properties. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  t : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d
  h3 : a 1 + t^2 = a 2 + t^3
  h4 : a 2 + t^3 = a 3 + t

/-- The theorem stating the properties of the arithmetic sequence. -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.t = -1/2 ∧ seq.d = 3/8 ∧
  (∃ (m p r : ℕ), m < p ∧ p < r ∧
    seq.a m - 2*seq.t^m = seq.a p - 2*seq.t^p ∧
    seq.a p - 2*seq.t^p = seq.a r - 2*seq.t^r ∧
    seq.a r - 2*seq.t^r = 0 ∧
    m = 1 ∧ p = 3 ∧ r = 4) ∧
  (∀ n : ℕ, seq.a n = 3/8 * n - 11/8) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l186_18639


namespace NUMINAMATH_CALUDE_savings_equality_l186_18647

/-- Prove that A's savings equal B's savings given the conditions -/
theorem savings_equality (total_salary : ℝ) (a_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h1 : total_salary = 7000)
  (h2 : a_salary = 5250)
  (h3 : a_spend_rate = 0.95)
  (h4 : b_spend_rate = 0.85) :
  a_salary * (1 - a_spend_rate) = (total_salary - a_salary) * (1 - b_spend_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_savings_equality_l186_18647


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l186_18667

/-- The number of students in the class -/
def total_students : Nat := 8

/-- The number of students needed for the relay race -/
def relay_team_size : Nat := 4

/-- The number of students that must be selected (A and B) -/
def must_select : Nat := 2

/-- The number of positions where A and B can be placed (first or last) -/
def fixed_positions : Nat := 2

/-- The number of remaining positions to be filled -/
def remaining_positions : Nat := relay_team_size - must_select

/-- The number of remaining students to choose from -/
def remaining_students : Nat := total_students - must_select

theorem relay_race_arrangements :
  (fixed_positions.factorial) *
  (remaining_students.choose remaining_positions) *
  (remaining_positions.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l186_18667


namespace NUMINAMATH_CALUDE_jeanine_has_more_pencils_jeanine_has_more_pencils_proof_l186_18637

/-- The number of pencils Jeanine has after giving some to Abby is 3 more than Clare's pencils. -/
theorem jeanine_has_more_pencils : ℕ → Prop :=
  fun (initial_pencils : ℕ) =>
    initial_pencils = 18 →
    let clare_pencils := initial_pencils / 2
    let jeanine_remaining := initial_pencils - (initial_pencils / 3)
    jeanine_remaining - clare_pencils = 3

/-- Proof of the theorem -/
theorem jeanine_has_more_pencils_proof : jeanine_has_more_pencils 18 := by
  sorry

#check jeanine_has_more_pencils_proof

end NUMINAMATH_CALUDE_jeanine_has_more_pencils_jeanine_has_more_pencils_proof_l186_18637


namespace NUMINAMATH_CALUDE_a_time_is_ten_l186_18694

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 80 meters long
  a.speed * a.time = 80 ∧
  b.speed * b.time = 80 ∧
  -- A beats B by 56 meters or 7 seconds
  a.speed * (a.time + 7) = 136 ∧
  b.time = a.time + 7

/-- Theorem stating A's time is 10 seconds -/
theorem a_time_is_ten (a b : Runner) (h : Race a b) : a.time = 10 :=
  sorry

end NUMINAMATH_CALUDE_a_time_is_ten_l186_18694


namespace NUMINAMATH_CALUDE_shepherd_problem_l186_18650

/-- Represents the number of sheep each shepherd has -/
structure ShepherdSheep where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Conditions for the shepherd problem -/
def satisfies_conditions (s : ShepherdSheep) : Prop :=
  (s.a + 1 - (s.a + 1) / 5 = s.b - 1 + (s.a + 1) / 5) ∧
  (s.a + 2 - 2 * (s.a + 2) / 7 = s.c - 2 + 2 * (s.a + 2) / 7)

/-- The minimum total number of sheep -/
def min_total_sheep : ℕ := 43

theorem shepherd_problem :
  ∃ (s : ShepherdSheep),
    satisfies_conditions s ∧
    s.a + s.b + s.c = min_total_sheep ∧
    ∀ (s' : ShepherdSheep),
      satisfies_conditions s' →
      s'.a + s'.b + s'.c ≥ min_total_sheep :=
sorry

end NUMINAMATH_CALUDE_shepherd_problem_l186_18650


namespace NUMINAMATH_CALUDE_sam_average_letters_per_day_l186_18605

/-- Given that Sam wrote 7 letters on Tuesday and 3 letters on Wednesday,
    prove that the average number of letters he wrote per day is 5. -/
theorem sam_average_letters_per_day :
  let tuesday_letters : ℕ := 7
  let wednesday_letters : ℕ := 3
  let total_days : ℕ := 2
  let total_letters : ℕ := tuesday_letters + wednesday_letters
  let average_letters : ℚ := total_letters / total_days
  average_letters = 5 := by
sorry

end NUMINAMATH_CALUDE_sam_average_letters_per_day_l186_18605


namespace NUMINAMATH_CALUDE_unanswered_questions_l186_18668

/-- Represents the scoring for a math competition participant --/
structure Scoring where
  correct : ℕ      -- number of correct answers
  incorrect : ℕ    -- number of incorrect answers
  unanswered : ℕ   -- number of unanswered questions

/-- Calculates the score using the first method --/
def score_method1 (s : Scoring) : ℕ :=
  5 * s.correct + 2 * s.unanswered

/-- Calculates the score using the second method --/
def score_method2 (s : Scoring) : ℕ :=
  39 + 3 * s.correct - s.incorrect

/-- Theorem stating the possible number of unanswered questions --/
theorem unanswered_questions (s : Scoring) :
  score_method1 s = 71 ∧ score_method2 s = 71 ∧ 
  s.correct + s.incorrect + s.unanswered = s.correct + s.incorrect →
  s.unanswered = 8 ∨ s.unanswered = 3 := by
  sorry


end NUMINAMATH_CALUDE_unanswered_questions_l186_18668


namespace NUMINAMATH_CALUDE_exists_subsequence_with_difference_property_l186_18678

/-- The sequence of rational numbers (1, 1/2, 1/3, ...) -/
def harmonic_sequence : ℕ → ℚ :=
  λ n => if n = 0 then 1 else 1 / (n + 1)

/-- A subsequence of the harmonic sequence -/
def subsequence (f : ℕ → ℕ) : ℕ → ℚ :=
  λ n => harmonic_sequence (f n)

/-- The property that each term, starting from the third, 
    is the difference of the two preceding terms -/
def has_difference_property (seq : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → seq (n + 1) = seq n - seq (n - 1)

theorem exists_subsequence_with_difference_property :
  ∃ (f : ℕ → ℕ), Monotone f ∧ has_difference_property (subsequence f) ∧ 
  (∀ n : ℕ, n < 100 → f n < f (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_exists_subsequence_with_difference_property_l186_18678


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_five_ninety_nine_thousand_nine_hundred_ninety_seven_divisible_largest_n_is_ninety_nine_thousand_nine_hundred_ninety_seven_l186_18699

theorem largest_n_divisible_by_five (n : ℕ) : 
  n < 100000 ∧ 
  (8 * (n - 2)^5 - n^2 + 14*n - 24) % 5 = 0 →
  n ≤ 99997 :=
sorry

theorem ninety_nine_thousand_nine_hundred_ninety_seven_divisible :
  (8 * (99997 - 2)^5 - 99997^2 + 14*99997 - 24) % 5 = 0 :=
sorry

theorem largest_n_is_ninety_nine_thousand_nine_hundred_ninety_seven :
  ∀ n : ℕ, n < 100000 ∧ 
  (8 * (n - 2)^5 - n^2 + 14*n - 24) % 5 = 0 →
  n ≤ 99997 :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_five_ninety_nine_thousand_nine_hundred_ninety_seven_divisible_largest_n_is_ninety_nine_thousand_nine_hundred_ninety_seven_l186_18699


namespace NUMINAMATH_CALUDE_participation_plans_count_l186_18621

/-- The number of students to choose from, excluding the pre-selected student -/
def n : ℕ := 3

/-- The number of students to be selected, excluding the pre-selected student -/
def k : ℕ := 2

/-- The total number of students participating (including the pre-selected student) -/
def total_participants : ℕ := k + 1

/-- The number of subjects -/
def subjects : ℕ := 3

theorem participation_plans_count : 
  (n.choose k) * (Nat.factorial total_participants) = 18 := by
  sorry

end NUMINAMATH_CALUDE_participation_plans_count_l186_18621


namespace NUMINAMATH_CALUDE_elisa_target_amount_l186_18653

/-- Elisa's target amount problem -/
theorem elisa_target_amount (current_amount additional_amount : ℕ) 
  (h1 : current_amount = 37)
  (h2 : additional_amount = 16) :
  current_amount + additional_amount = 53 := by
  sorry

end NUMINAMATH_CALUDE_elisa_target_amount_l186_18653


namespace NUMINAMATH_CALUDE_abs_x_bound_inequality_x_y_l186_18651

-- Part 1
theorem abs_x_bound (x y : ℝ) 
  (h1 : |x - 3*y| < 1/2) (h2 : |x + 2*y| < 1/6) : 
  |x| < 3/10 := by sorry

-- Part 2
theorem inequality_x_y (x y : ℝ) :
  x^4 + 16*y^4 ≥ 2*x^3*y + 8*x*y^3 := by sorry

end NUMINAMATH_CALUDE_abs_x_bound_inequality_x_y_l186_18651


namespace NUMINAMATH_CALUDE_similar_triangles_side_proportional_l186_18646

/-- Two triangles are similar if their corresponding angles are equal -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_side_proportional 
  (G H I X Y Z : ℝ × ℝ) 
  (h_similar : similar_triangles {G, H, I} {X, Y, Z}) 
  (h_GH : dist G H = 8)
  (h_HI : dist H I = 20)
  (h_YZ : dist Y Z = 25) : 
  dist X Y = 80 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_side_proportional_l186_18646


namespace NUMINAMATH_CALUDE_shirt_and_coat_cost_l186_18666

/-- Given a shirt that costs $150 and is one-third the price of a coat,
    prove that the total cost of the shirt and coat is $600. -/
theorem shirt_and_coat_cost (shirt_cost : ℕ) (coat_cost : ℕ) : 
  shirt_cost = 150 → 
  shirt_cost * 3 = coat_cost →
  shirt_cost + coat_cost = 600 := by
  sorry

end NUMINAMATH_CALUDE_shirt_and_coat_cost_l186_18666


namespace NUMINAMATH_CALUDE_bus_problem_l186_18672

/-- The number of people who got off the bus -/
def people_got_off (initial : ℕ) (final : ℕ) : ℕ := initial - final

/-- Theorem stating that 47 people got off the bus -/
theorem bus_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 90) 
  (h2 : final = 43) : 
  people_got_off initial final = 47 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l186_18672


namespace NUMINAMATH_CALUDE_parallel_vectors_m_equals_six_l186_18610

/-- Two vectors are parallel if and only if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a = (m, 4) and b = (3, 2) are parallel, prove that m = 6 -/
theorem parallel_vectors_m_equals_six (m : ℝ) :
  are_parallel (m, 4) (3, 2) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_equals_six_l186_18610


namespace NUMINAMATH_CALUDE_inequality_proof_l186_18609

theorem inequality_proof (a b c : ℝ) (ha : a = (-0.3)^0) (hb : b = 0.32) (hc : c = 20.3) :
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l186_18609


namespace NUMINAMATH_CALUDE_erroneous_multiplication_l186_18632

/-- Given two positive integers where one is a two-digit number,
    if the product of the digit-reversed two-digit number and the other integer is 161,
    then the product of the original numbers is 224. -/
theorem erroneous_multiplication (a b : ℕ) : 
  a ≥ 10 ∧ a ≤ 99 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (10 * (a % 10) + (a / 10)) * b = 161 →  -- reversed a multiplied by b is 161
  a * b = 224 :=
by sorry

end NUMINAMATH_CALUDE_erroneous_multiplication_l186_18632


namespace NUMINAMATH_CALUDE_equation_solution_l186_18697

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

/-- The theorem stating that functions satisfying the equation are either the identity function or the absolute value function -/
theorem equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = |x|) := by
  sorry


end NUMINAMATH_CALUDE_equation_solution_l186_18697


namespace NUMINAMATH_CALUDE_parallel_planes_through_two_points_l186_18690

-- Define a plane
def Plane : Type := sorry

-- Define a point
def Point : Type := sorry

-- Define a function to check if a point is outside a plane
def isOutside (p : Point) (pl : Plane) : Prop := sorry

-- Define a function to check if a plane is parallel to another plane
def isParallel (pl1 : Plane) (pl2 : Plane) : Prop := sorry

-- Define a function to count the number of planes that can be drawn through two points and parallel to a given plane
def countParallelPlanes (p1 p2 : Point) (pl : Plane) : Nat := sorry

-- Theorem statement
theorem parallel_planes_through_two_points 
  (p1 p2 : Point) (pl : Plane) 
  (h1 : isOutside p1 pl) 
  (h2 : isOutside p2 pl) : 
  countParallelPlanes p1 p2 pl = 0 ∨ countParallelPlanes p1 p2 pl = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_through_two_points_l186_18690


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l186_18674

/-- The maximum distance from any point on the circle ρ = 8sinθ to the line θ = π/3 is 6 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 8 * p.2}
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  ∃ (d : ℝ),
    d = 6 ∧
    ∀ p ∈ circle, ∀ q ∈ line,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ d :=
by sorry


end NUMINAMATH_CALUDE_max_distance_circle_to_line_l186_18674


namespace NUMINAMATH_CALUDE_function_has_extrema_l186_18693

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + (2*a + 1)*x

-- State the theorem
theorem function_has_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
   (∀ x, f a x ≥ f a x₁) ∧ 
   (∀ x, f a x ≤ f a x₂)) ↔ 
  (a > 1 ∨ a < -1/3) :=
sorry

end NUMINAMATH_CALUDE_function_has_extrema_l186_18693


namespace NUMINAMATH_CALUDE_cos_right_angle_l186_18692

theorem cos_right_angle (D E F : ℝ) (h1 : D = 90) (h2 : E = 9) (h3 : F = 40) : Real.cos D = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_right_angle_l186_18692


namespace NUMINAMATH_CALUDE_total_cost_of_hats_l186_18614

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- Theorem: The total cost of John's hats is $700 -/
theorem total_cost_of_hats : 
  weeks_of_different_hats * days_per_week * cost_per_hat = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_hats_l186_18614


namespace NUMINAMATH_CALUDE_factorization_proof_l186_18636

theorem factorization_proof (z : ℝ) : 
  88 * z^19 + 176 * z^38 + 264 * z^57 = 88 * z^19 * (1 + 2 * z^19 + 3 * z^38) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l186_18636


namespace NUMINAMATH_CALUDE_equation_solution_l186_18684

theorem equation_solution : ∃ x : ℝ, x ≠ 1 ∧ (x / (x - 1) + 2 / (1 - x) = 2) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l186_18684


namespace NUMINAMATH_CALUDE_quadratic_real_roots_imply_a_equals_negative_one_l186_18633

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℂ :=
  a * (1 + i) * x^2 + (1 + a^2 * i) * x + a^2 + i

-- Theorem statement
theorem quadratic_real_roots_imply_a_equals_negative_one :
  ∀ a : ℝ, (∃ x : ℝ, quadratic_equation a x = 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_imply_a_equals_negative_one_l186_18633


namespace NUMINAMATH_CALUDE_multiple_right_triangles_exist_l186_18657

/-- A right triangle with a given hypotenuse length and one non-right angle -/
structure RightTriangle where
  hypotenuse : ℝ
  angle : ℝ
  hypotenuse_positive : 0 < hypotenuse
  angle_range : 0 < angle ∧ angle < π / 2

/-- Theorem stating that multiple right triangles can have the same hypotenuse and non-right angle -/
theorem multiple_right_triangles_exist (h : ℝ) (θ : ℝ) 
  (h_pos : 0 < h) (θ_range : 0 < θ ∧ θ < π / 2) :
  ∃ (t1 t2 : RightTriangle), t1 ≠ t2 ∧ 
    t1.hypotenuse = h ∧ t1.angle = θ ∧
    t2.hypotenuse = h ∧ t2.angle = θ :=
sorry

end NUMINAMATH_CALUDE_multiple_right_triangles_exist_l186_18657
