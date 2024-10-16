import Mathlib

namespace NUMINAMATH_CALUDE_grandmother_inheritance_l4011_401164

/-- Proves that if 5 people equally split an amount of money and each receives $105,500, then the total amount is $527,500. -/
theorem grandmother_inheritance (num_people : ℕ) (amount_per_person : ℕ) (total_amount : ℕ) :
  num_people = 5 →
  amount_per_person = 105500 →
  total_amount = num_people * amount_per_person →
  total_amount = 527500 :=
by
  sorry

end NUMINAMATH_CALUDE_grandmother_inheritance_l4011_401164


namespace NUMINAMATH_CALUDE_paintings_from_C_l4011_401177

-- Define the number of paintings from each school
variable (A B C : ℕ)

-- Define the total number of paintings
def T : ℕ := A + B + C

-- State the conditions
axiom not_from_A : B + C = 41
axiom not_from_B : A + C = 38
axiom from_A_and_B : A + B = 43

-- State the theorem to be proved
theorem paintings_from_C : C = 18 := by sorry

end NUMINAMATH_CALUDE_paintings_from_C_l4011_401177


namespace NUMINAMATH_CALUDE_quadratic_sqrt2_closure_l4011_401128

-- Define a structure for numbers of the form a + b√2
structure QuadraticSqrt2 where
  a : ℚ
  b : ℚ

-- Define addition for QuadraticSqrt2
def add (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a + y.a, x.b + y.b⟩

-- Define subtraction for QuadraticSqrt2
def sub (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a - y.a, x.b - y.b⟩

-- Define multiplication for QuadraticSqrt2
def mul (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a * y.a + 2 * x.b * y.b, x.a * y.b + x.b * y.a⟩

-- Define division for QuadraticSqrt2
def div (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  let denom := y.a * y.a - 2 * y.b * y.b
  ⟨(x.a * y.a - 2 * x.b * y.b) / denom, (x.b * y.a - x.a * y.b) / denom⟩

theorem quadratic_sqrt2_closure (x y : QuadraticSqrt2) (h : y.a * y.a ≠ 2 * y.b * y.b) :
  (∃ (z : QuadraticSqrt2), add x y = z) ∧
  (∃ (z : QuadraticSqrt2), sub x y = z) ∧
  (∃ (z : QuadraticSqrt2), mul x y = z) ∧
  (∃ (z : QuadraticSqrt2), div x y = z) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sqrt2_closure_l4011_401128


namespace NUMINAMATH_CALUDE_no_positive_and_negative_rational_l4011_401139

theorem no_positive_and_negative_rational : ¬∃ (q : ℚ), q > 0 ∧ q < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_and_negative_rational_l4011_401139


namespace NUMINAMATH_CALUDE_equation_solution_l4011_401188

theorem equation_solution : ∃ x : ℝ, (x / (2 * x - 3) + 5 / (3 - 2 * x) = 4) ∧ (x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4011_401188


namespace NUMINAMATH_CALUDE_inequality_proof_l4011_401109

theorem inequality_proof (a b : ℝ) : (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2*b^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4011_401109


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l4011_401178

theorem quadratic_expression_value : 
  let x : ℝ := 2
  2 * x^2 - 3 * x + 4 = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l4011_401178


namespace NUMINAMATH_CALUDE_dog_reach_area_l4011_401132

/-- The area outside a regular hexagonal doghouse that a dog can reach when tethered to a vertex --/
theorem dog_reach_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 2 → rope_length = 3 → 
  (area_outside_doghouse : ℝ) = (22 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_reach_area_l4011_401132


namespace NUMINAMATH_CALUDE_x_squared_minus_5x_is_quadratic_l4011_401160

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 - 5*x

/-- Theorem: x^2 - 5x = 0 is a quadratic equation -/
theorem x_squared_minus_5x_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_5x_is_quadratic_l4011_401160


namespace NUMINAMATH_CALUDE_min_degree_of_specific_polynomial_l4011_401145

/-- A polynomial function from ℝ to ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- The degree of a polynomial function -/
def degree (f : PolynomialFunction) : ℕ := sorry

theorem min_degree_of_specific_polynomial (f : PolynomialFunction)
  (h1 : f (-2) = 3)
  (h2 : f (-1) = -3)
  (h3 : f 1 = -3)
  (h4 : f 2 = 6)
  (h5 : f 3 = 5) :
  degree f = 4 ∧ ∀ g : PolynomialFunction, 
    g (-2) = 3 → g (-1) = -3 → g 1 = -3 → g 2 = 6 → g 3 = 5 → 
    degree g ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_degree_of_specific_polynomial_l4011_401145


namespace NUMINAMATH_CALUDE_craft_sales_sum_l4011_401118

/-- The sum of an arithmetic sequence with first term 3 and common difference 4 for 10 terms -/
theorem craft_sales_sum : 
  let a : ℕ → ℕ := fun n => 3 + 4 * (n - 1)
  let S : ℕ → ℕ := fun n => n * (a 1 + a n) / 2
  S 10 = 210 := by
sorry

end NUMINAMATH_CALUDE_craft_sales_sum_l4011_401118


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l4011_401194

/-- Given a geometric sequence {a_n} where a₂ = 2 and a₁₀ = 8, prove that a₆ = 4 -/
theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)  -- Geometric sequence condition
  (h_2 : a 2 = 2)  -- Second term is 2
  (h_10 : a 10 = 8)  -- Tenth term is 8
  : a 6 = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l4011_401194


namespace NUMINAMATH_CALUDE_recipe_sugar_amount_l4011_401156

/-- The amount of sugar Katie has already put in the recipe -/
def sugar_already_added : ℝ := 0.5

/-- The amount of sugar Katie still needs to add to the recipe -/
def sugar_to_add : ℝ := 2.5

/-- The total amount of sugar required by the recipe -/
def total_sugar_needed : ℝ := sugar_already_added + sugar_to_add

theorem recipe_sugar_amount : total_sugar_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_recipe_sugar_amount_l4011_401156


namespace NUMINAMATH_CALUDE_santa_gift_combinations_l4011_401149

theorem santa_gift_combinations (n : ℤ) : 
  ∃ k : ℤ, n^5 - n = 30 * k := by sorry

end NUMINAMATH_CALUDE_santa_gift_combinations_l4011_401149


namespace NUMINAMATH_CALUDE_max_PXQ_value_l4011_401112

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def starts_with (n : ℕ) (d : ℕ) : Prop :=
  (n / 100) = d

def ends_with (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d

theorem max_PXQ_value :
  ∀ XX X PXQ : ℕ,
    is_two_digit_with_equal_digits XX →
    is_one_digit X →
    is_three_digit PXQ →
    XX * X = PXQ →
    starts_with PXQ (PXQ / 100) →
    ends_with PXQ X →
    PXQ ≤ 396 :=
sorry

end NUMINAMATH_CALUDE_max_PXQ_value_l4011_401112


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l4011_401102

theorem inequality_and_equality_condition (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_not_all_zero : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : 
  (2*x^2 - x + y + z)/(x + y^2 + z^2) + 
  (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
  (2*z^2 + x + y - z)/(x^2 + y^2 + z) ≥ 3 ∧
  ((2*x^2 - x + y + z)/(x + y^2 + z^2) + 
   (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
   (2*z^2 + x + y - z)/(x^2 + y^2 + z) = 3 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l4011_401102


namespace NUMINAMATH_CALUDE_fourth_quarter_points_l4011_401141

def winning_team_points (q1 q2 q3 q4 : ℕ) : Prop :=
  q1 = 20 ∧ q2 = q1 + 10 ∧ q3 = q2 + 20 ∧ q1 + q2 + q3 + q4 = 80

theorem fourth_quarter_points :
  ∃ q1 q2 q3 q4 : ℕ,
    winning_team_points q1 q2 q3 q4 ∧ q4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fourth_quarter_points_l4011_401141


namespace NUMINAMATH_CALUDE_expression_evaluation_l4011_401166

theorem expression_evaluation : 
  let x : ℝ := 3
  let expr := (2 * x^2 + 2*x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2*x + 1)
  expr / (x / (x + 1)) = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4011_401166


namespace NUMINAMATH_CALUDE_m_equals_six_l4011_401110

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The given function f(x) -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 6)

/-- Theorem stating that m = 6 given that f is an odd function -/
theorem m_equals_six (h : IsOdd (f m)) : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_m_equals_six_l4011_401110


namespace NUMINAMATH_CALUDE_intersection_M_P_union_M_P_condition_l4011_401119

-- Define the sets M and P
def M (m : ℝ) : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4*m - 2}
def P : Set ℝ := {x : ℝ | x > 2 ∨ x ≤ 1}

-- Theorem 1: Intersection of M and P when m = 2
theorem intersection_M_P : 
  M 2 ∩ P = {x : ℝ | (-1 ≤ x ∧ x ≤ 1) ∨ (2 < x ∧ x ≤ 6)} := by sorry

-- Theorem 2: Union of M and P is ℝ iff m ≥ 1
theorem union_M_P_condition (m : ℝ) : 
  M m ∪ P = Set.univ ↔ m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_P_union_M_P_condition_l4011_401119


namespace NUMINAMATH_CALUDE_construct_triangle_from_symmetric_points_l4011_401120

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- Check if a triangle is acute-angled -/
def is_acute_angled (t : Triangle) : Prop := sorry

/-- The symmetric point of a given point with respect to a line segment -/
def symmetric_point (p : Point) (a : Point) (b : Point) : Point := sorry

/-- Theorem: Given three points that are symmetric to the orthocenter of an acute-angled triangle
    with respect to its sides, the triangle can be uniquely constructed -/
theorem construct_triangle_from_symmetric_points
  (A' B' C' : Point) :
  ∃! (t : Triangle),
    is_acute_angled t ∧
    A' = symmetric_point (orthocenter t) t.B t.C ∧
    B' = symmetric_point (orthocenter t) t.C t.A ∧
    C' = symmetric_point (orthocenter t) t.A t.B :=
sorry

end NUMINAMATH_CALUDE_construct_triangle_from_symmetric_points_l4011_401120


namespace NUMINAMATH_CALUDE_ice_skate_profit_maximization_l4011_401108

/-- Ice skate problem -/
theorem ice_skate_profit_maximization
  (cost_A cost_B : ℕ)  -- Cost prices of type A and B
  (sell_A sell_B : ℕ)  -- Selling prices of type A and B
  (total_pairs : ℕ)    -- Total number of pairs to purchase
  : cost_B = 2 * cost_A  -- Condition 1
  → 2 * cost_A + cost_B = 920  -- Condition 2
  → sell_A = 400  -- Condition 3
  → sell_B = 560  -- Condition 4
  → total_pairs = 50  -- Condition 5
  → (∀ x y : ℕ, x + y = total_pairs → x ≤ 2 * y)  -- Condition 6
  → ∃ (x y : ℕ),
      x + y = total_pairs ∧
      x = 33 ∧
      y = 17 ∧
      x * (sell_A - cost_A) + y * (sell_B - cost_B) = 6190 ∧
      ∀ (a b : ℕ), a + b = total_pairs →
        a * (sell_A - cost_A) + b * (sell_B - cost_B) ≤ 6190 :=
by sorry

end NUMINAMATH_CALUDE_ice_skate_profit_maximization_l4011_401108


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l4011_401121

theorem difference_of_squares_example : (538 * 538) - (537 * 539) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l4011_401121


namespace NUMINAMATH_CALUDE_eggs_removed_l4011_401103

theorem eggs_removed (original : ℕ) (remaining : ℕ) (removed : ℕ) : 
  original = 27 → remaining = 20 → removed = original - remaining → removed = 7 := by
sorry

end NUMINAMATH_CALUDE_eggs_removed_l4011_401103


namespace NUMINAMATH_CALUDE_coral_reef_number_conversion_l4011_401144

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to hexadecimal --/
def decimal_to_hex (n : ℕ) : String := sorry

theorem coral_reef_number_conversion :
  let octal_num := 732
  let decimal_num := octal_to_decimal octal_num
  decimal_num = 474 ∧ decimal_to_hex decimal_num = "1DA" := by sorry

end NUMINAMATH_CALUDE_coral_reef_number_conversion_l4011_401144


namespace NUMINAMATH_CALUDE_circle_radius_from_chords_l4011_401104

/-- Given a circle with two chords of lengths 20 cm and 26 cm starting from the same point
    and forming an angle of 36° 38', the radius of the circle is approximately 24.84 cm. -/
theorem circle_radius_from_chords (chord1 chord2 angle : ℝ) (h1 : chord1 = 20)
    (h2 : chord2 = 26) (h3 : angle = 36 + 38 / 60) : ∃ r : ℝ, 
    abs (r - 24.84) < 0.01 ∧ 
    chord1^2 + chord2^2 - 2 * chord1 * chord2 * Real.cos (angle * Real.pi / 180) = 
    4 * r^2 * Real.sin ((angle * Real.pi / 180) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_chords_l4011_401104


namespace NUMINAMATH_CALUDE_cubic_root_sum_l4011_401175

/-- Given that p, q, and r are the roots of x³ - 3x - 2 = 0,
    prove that p(q - r)² + q(r - p)² + r(p - q)² = -18 -/
theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 = 3*p + 2) → 
  (q^3 = 3*q + 2) → 
  (r^3 = 3*r + 2) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l4011_401175


namespace NUMINAMATH_CALUDE_wednesday_work_time_l4011_401148

/-- Represents the work time in minutes for each day of the week -/
structure WorkWeek where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- Calculates the total work time for the week in minutes -/
def totalWorkTime (w : WorkWeek) : ℚ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Converts hours to minutes -/
def hoursToMinutes (hours : ℚ) : ℚ :=
  hours * 60

theorem wednesday_work_time (w : WorkWeek) : 
  w.monday = hoursToMinutes (3/4) ∧ 
  w.tuesday = hoursToMinutes (1/2) ∧ 
  w.thursday = hoursToMinutes (5/6) ∧ 
  w.friday = 75 ∧ 
  totalWorkTime w = hoursToMinutes 4 → 
  w.wednesday = 40 := by
sorry

end NUMINAMATH_CALUDE_wednesday_work_time_l4011_401148


namespace NUMINAMATH_CALUDE_magic_trick_basis_l4011_401136

/-- The set of valid dice face pairs -/
def DicePairs : Set (ℕ × ℕ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 6}

/-- The set of possible numbers of dice in the spectator's pocket -/
def PocketCounts : Set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 21}

/-- The statement of the magic trick's mathematical basis -/
theorem magic_trick_basis :
  ∃ f : DicePairs → PocketCounts, Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_magic_trick_basis_l4011_401136


namespace NUMINAMATH_CALUDE_probability_females_right_of_males_l4011_401168

theorem probability_females_right_of_males :
  let total_people : ℕ := 3 + 2
  let male_count : ℕ := 3
  let female_count : ℕ := 2
  let total_arrangements : ℕ := Nat.factorial total_people
  let favorable_arrangements : ℕ := Nat.factorial male_count * Nat.factorial female_count
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_females_right_of_males_l4011_401168


namespace NUMINAMATH_CALUDE_max_inscribed_sphere_volume_l4011_401127

theorem max_inscribed_sphere_volume (cone_base_diameter : ℝ) (cone_volume : ℝ) 
  (h_diameter : cone_base_diameter = 12)
  (h_volume : cone_volume = 96 * Real.pi) : 
  let cone_radius : ℝ := cone_base_diameter / 2
  let cone_height : ℝ := 3 * cone_volume / (Real.pi * cone_radius^2)
  let cone_slant_height : ℝ := Real.sqrt (cone_radius^2 + cone_height^2)
  let sphere_radius : ℝ := cone_radius * cone_height / (cone_radius + cone_height + cone_slant_height)
  let sphere_volume : ℝ := 4 / 3 * Real.pi * sphere_radius^3
  sphere_volume = 36 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_max_inscribed_sphere_volume_l4011_401127


namespace NUMINAMATH_CALUDE_expression_evaluation_l4011_401155

theorem expression_evaluation :
  let x : ℚ := 3
  let y : ℚ := -1/3
  3 * x * y^2 - (x * y - 2 * (2 * x * y - 3/2 * x^2 * y) + 3 * x * y^2) + 3 * x^2 * y = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4011_401155


namespace NUMINAMATH_CALUDE_intersection_points_count_l4011_401114

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if a line intersects a circle -/
def intersectsCircle (l : Line) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if two lines intersect -/
def intersectLines (l1 l2 : Line) : Prop :=
  sorry

/-- Function to count intersection points between a line and a circle -/
def countIntersections (l : Line) (c : Circle) : ℕ :=
  sorry

/-- Main theorem -/
theorem intersection_points_count 
  (c : Circle) (l1 l2 : Line) 
  (h1 : isTangent l1 c)
  (h2 : intersectsCircle l2 c)
  (h3 : ¬ isTangent l2 c)
  (h4 : intersectLines l1 l2) :
  countIntersections l1 c + countIntersections l2 c = 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_count_l4011_401114


namespace NUMINAMATH_CALUDE_rosie_pies_l4011_401187

/-- Represents the number of pies that can be made given ingredients and their ratios -/
def pies_made (apples_per_pie oranges_per_pie available_apples available_oranges : ℚ) : ℚ :=
  min (available_apples / apples_per_pie) (available_oranges / oranges_per_pie)

/-- Theorem stating that Rosie can make 9 pies with the given ingredients -/
theorem rosie_pies :
  let apples_per_pie : ℚ := 12 / 3
  let oranges_per_pie : ℚ := 6 / 3
  let available_apples : ℚ := 36
  let available_oranges : ℚ := 18
  pies_made apples_per_pie oranges_per_pie available_apples available_oranges = 9 := by
  sorry

#eval pies_made (12 / 3) (6 / 3) 36 18

end NUMINAMATH_CALUDE_rosie_pies_l4011_401187


namespace NUMINAMATH_CALUDE_negation_equivalence_l4011_401192

theorem negation_equivalence (m : ℝ) : 
  (¬ ∃ x < 0, x^2 + 2*x - m > 0) ↔ (∀ x < 0, x^2 + 2*x - m ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4011_401192


namespace NUMINAMATH_CALUDE_two_digit_repeating_decimal_l4011_401174

theorem two_digit_repeating_decimal (ab : ℕ) (h1 : ab ≥ 10 ∧ ab < 100) :
  66 * (1 + ab / 100 : ℚ) + 1/2 = 66 * (1 + ab / 99 : ℚ) → ab = 75 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_repeating_decimal_l4011_401174


namespace NUMINAMATH_CALUDE_solve_equation_l4011_401190

theorem solve_equation (y : ℚ) (h : 3 * y - 9 = -6 * y + 3) : y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4011_401190


namespace NUMINAMATH_CALUDE_notebook_problem_l4011_401179

def satisfies_notebook_conditions (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    (y + 2 = n * (x - 2)) ∧
    (x + n = 2 * (y - n)) ∧
    x > 2 ∧ y > n

theorem notebook_problem :
  {n : ℕ | satisfies_notebook_conditions n} = {1, 2, 3, 8} :=
by sorry

end NUMINAMATH_CALUDE_notebook_problem_l4011_401179


namespace NUMINAMATH_CALUDE_xy_yz_zx_over_x2_y2_z2_l4011_401147

theorem xy_yz_zx_over_x2_y2_z2 (x y z a b c : ℝ) 
  (h_distinct_xyz : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_distinct_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_nonzero_abc : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a * x + b * y + c * z = 0) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1 :=
by sorry

end NUMINAMATH_CALUDE_xy_yz_zx_over_x2_y2_z2_l4011_401147


namespace NUMINAMATH_CALUDE_correct_calculation_l4011_401107

theorem correct_calculation (x : ℚ) (h : x + 7/5 = 81/20) : x - 7/5 = 25/20 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4011_401107


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l4011_401130

/-- Given the animal counts at San Diego Zoo, prove the ratio of bee-eaters to leopards -/
theorem zoo_animal_ratio :
  let total_animals : ℕ := 670
  let snakes : ℕ := 100
  let arctic_foxes : ℕ := 80
  let leopards : ℕ := 20
  let cheetahs : ℕ := snakes / 2
  let alligators : ℕ := 2 * (arctic_foxes + leopards)
  let bee_eaters : ℕ := total_animals - (snakes + arctic_foxes + leopards + cheetahs + alligators)
  (bee_eaters : ℚ) / leopards = 11 / 1 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l4011_401130


namespace NUMINAMATH_CALUDE_shaded_half_l4011_401167

/-- Represents a square divided into smaller squares with specific shading -/
structure DividedSquare where
  /-- The number of smaller squares the large square is divided into -/
  num_divisions : Nat
  /-- Whether a diagonal is drawn in one of the smaller squares -/
  has_diagonal : Bool
  /-- The number of quarters of a smaller square that are additionally shaded -/
  additional_shaded_quarters : Nat

/-- Calculates the fraction of the large square that is shaded -/
def shaded_fraction (s : DividedSquare) : Rat :=
  sorry

/-- Theorem stating that for a specific configuration, the shaded fraction is 1/2 -/
theorem shaded_half (s : DividedSquare) 
  (h1 : s.num_divisions = 4) 
  (h2 : s.has_diagonal = true)
  (h3 : s.additional_shaded_quarters = 2) : 
  shaded_fraction s = 1/2 :=
sorry

end NUMINAMATH_CALUDE_shaded_half_l4011_401167


namespace NUMINAMATH_CALUDE_part_one_part_two_l4011_401173

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part_one (t : Triangle) (h1 : t.a + t.b + t.c = 16) (h2 : t.a = 4) (h3 : t.b = 5) :
  Real.cos t.C = -1/5 := by sorry

-- Part 2
theorem part_two (t : Triangle) (h1 : t.a + t.b + t.c = 16) 
  (h2 : Real.sin t.A + Real.sin t.B = 3 * Real.sin t.C)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 18 * Real.sin t.C) :
  t.a = 6 ∧ t.b = 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4011_401173


namespace NUMINAMATH_CALUDE_count_numbers_with_zero_up_to_3200_l4011_401100

/-- Returns true if the given natural number contains the digit 0 in its base-ten representation -/
def contains_zero (n : ℕ) : Bool := sorry

/-- Counts the number of positive integers less than or equal to n that contain at least one digit 0 in their base-ten representation -/
def count_numbers_with_zero (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the count of numbers with zero up to 3200 is 993 -/
theorem count_numbers_with_zero_up_to_3200 : 
  count_numbers_with_zero 3200 = 993 := by sorry

end NUMINAMATH_CALUDE_count_numbers_with_zero_up_to_3200_l4011_401100


namespace NUMINAMATH_CALUDE_unique_solution_exists_l4011_401171

/-- Given a > 0 and a ≠ 1, there exists a unique x such that a^x = log_(1/4) x -/
theorem unique_solution_exists (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  ∃! x : ℝ, a^x = Real.log x / Real.log (1/4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l4011_401171


namespace NUMINAMATH_CALUDE_ride_cost_is_factor_of_remaining_tickets_l4011_401153

def total_tickets : ℕ := 40
def spent_tickets : ℕ := 28
def remaining_tickets : ℕ := total_tickets - spent_tickets

def is_factor (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem ride_cost_is_factor_of_remaining_tickets :
  ∀ (num_rides cost_per_ride : ℕ),
    num_rides > 0 →
    cost_per_ride > 0 →
    num_rides * cost_per_ride = remaining_tickets →
    is_factor remaining_tickets cost_per_ride :=
by sorry

end NUMINAMATH_CALUDE_ride_cost_is_factor_of_remaining_tickets_l4011_401153


namespace NUMINAMATH_CALUDE_tom_uncommon_cards_l4011_401123

/-- Represents the deck composition and cost in Tom's trading card game. -/
structure DeckInfo where
  rare_count : ℕ
  common_count : ℕ
  rare_cost : ℚ
  uncommon_cost : ℚ
  common_cost : ℚ
  total_cost : ℚ

/-- Calculates the number of uncommon cards in the deck. -/
def uncommon_count (deck : DeckInfo) : ℕ :=
  let rare_total := deck.rare_count * deck.rare_cost
  let common_total := deck.common_count * deck.common_cost
  let uncommon_total := deck.total_cost - rare_total - common_total
  (uncommon_total / deck.uncommon_cost).num.toNat

/-- Theorem stating that Tom's deck contains 11 uncommon cards. -/
theorem tom_uncommon_cards : 
  let deck : DeckInfo := {
    rare_count := 19,
    common_count := 30,
    rare_cost := 1,
    uncommon_cost := 1/2,
    common_cost := 1/4,
    total_cost := 32
  }
  uncommon_count deck = 11 := by sorry

end NUMINAMATH_CALUDE_tom_uncommon_cards_l4011_401123


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l4011_401125

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l4011_401125


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l4011_401124

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ x = 1) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l4011_401124


namespace NUMINAMATH_CALUDE_amy_school_year_hours_l4011_401151

/-- Represents Amy's work schedule and earnings --/
structure WorkSchedule where
  summer_hours_per_week : ℕ
  summer_weeks : ℕ
  summer_earnings : ℕ
  school_year_weeks : ℕ
  school_year_earnings : ℕ

/-- Calculates the required hours per week for the school year --/
def required_school_year_hours (schedule : WorkSchedule) : ℚ :=
  (schedule.summer_hours_per_week : ℚ) * (schedule.summer_weeks : ℚ) * (schedule.school_year_earnings : ℚ) /
  ((schedule.summer_earnings : ℚ) * (schedule.school_year_weeks : ℚ))

/-- Theorem stating that Amy needs to work 12 hours per week during the school year --/
theorem amy_school_year_hours (schedule : WorkSchedule)
  (h1 : schedule.summer_hours_per_week = 36)
  (h2 : schedule.summer_weeks = 10)
  (h3 : schedule.summer_earnings = 3000)
  (h4 : schedule.school_year_weeks = 30)
  (h5 : schedule.school_year_earnings = 3000) :
  required_school_year_hours schedule = 12 := by
  sorry


end NUMINAMATH_CALUDE_amy_school_year_hours_l4011_401151


namespace NUMINAMATH_CALUDE_jenny_reading_speed_l4011_401199

/-- Represents Jenny's reading challenge --/
structure ReadingChallenge where
  days : ℕ
  books : ℕ
  book1_words : ℕ
  book2_words : ℕ
  book3_words : ℕ
  reading_minutes_per_day : ℕ

/-- Calculates the reading speed in words per hour --/
def calculate_reading_speed (challenge : ReadingChallenge) : ℕ :=
  let total_words := challenge.book1_words + challenge.book2_words + challenge.book3_words
  let words_per_day := total_words / challenge.days
  let reading_hours_per_day := challenge.reading_minutes_per_day / 60
  words_per_day / reading_hours_per_day

/-- Jenny's specific reading challenge --/
def jenny_challenge : ReadingChallenge :=
  { days := 10
  , books := 3
  , book1_words := 200
  , book2_words := 400
  , book3_words := 300
  , reading_minutes_per_day := 54
  }

/-- Theorem stating that Jenny's reading speed is 100 words per hour --/
theorem jenny_reading_speed :
  calculate_reading_speed jenny_challenge = 100 := by
  sorry

end NUMINAMATH_CALUDE_jenny_reading_speed_l4011_401199


namespace NUMINAMATH_CALUDE_lucy_age_l4011_401182

/-- Given the ages of Inez, Zack, Jose, and Lucy, prove Lucy's age --/
theorem lucy_age (inez zack jose lucy : ℕ) 
  (h1 : lucy = jose + 2)
  (h2 : jose + 6 = zack)
  (h3 : zack = inez + 4)
  (h4 : inez = 18) : 
  lucy = 18 := by
  sorry

end NUMINAMATH_CALUDE_lucy_age_l4011_401182


namespace NUMINAMATH_CALUDE_log_two_plus_log_five_equals_one_l4011_401113

theorem log_two_plus_log_five_equals_one : Real.log 2 + Real.log 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_two_plus_log_five_equals_one_l4011_401113


namespace NUMINAMATH_CALUDE_board_and_sum_properties_l4011_401197

/-- The number of squares in a square board -/
def boardSquares (n : ℕ) : ℕ := n * n

/-- The number of squares in each region separated by the diagonal -/
def regionSquares (n : ℕ) : ℕ := (n * n - n) / 2

/-- The sum of consecutive integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem board_and_sum_properties :
  (boardSquares 11 = 121) ∧
  (regionSquares 11 = 55) ∧
  (sumIntegers 10 = 55) ∧
  (sumIntegers 100 = 5050) :=
sorry

end NUMINAMATH_CALUDE_board_and_sum_properties_l4011_401197


namespace NUMINAMATH_CALUDE_combination_equations_l4011_401134

def A (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem combination_equations :
  (∃! x : ℕ+, 3 * (A x.val 3) = 2 * (A (x.val + 1) 2) + 6 * (A x.val 2)) ∧
  (∃ x : ℕ+, x = 1 ∨ x = 2) ∧ (∀ x : ℕ+, A 8 x.val = A 8 (5 * x.val - 4) → x = 1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_combination_equations_l4011_401134


namespace NUMINAMATH_CALUDE_rectangle_length_l4011_401131

theorem rectangle_length (L B : ℝ) (h1 : L / B = 25 / 16) (h2 : L * B = 200^2) : L = 250 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l4011_401131


namespace NUMINAMATH_CALUDE_angle_equality_l4011_401180

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos θ + Real.sin θ) : 
  θ = 15 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_equality_l4011_401180


namespace NUMINAMATH_CALUDE_popcorn_probability_l4011_401154

-- Define the proportions of kernels in the bag
def white_proportion : ℚ := 1/2
def yellow_proportion : ℚ := 1/4
def blue_proportion : ℚ := 1/4

-- Define the probabilities of popping for each color
def white_pop_prob : ℚ := 1/3
def yellow_pop_prob : ℚ := 3/4
def blue_pop_prob : ℚ := 2/3

-- Define the probability of a kernel being white given that it popped
def prob_white_given_popped : ℚ := 2/11

theorem popcorn_probability : 
  let p_white_and_popped := white_proportion * white_pop_prob
  let p_yellow_and_popped := yellow_proportion * yellow_pop_prob
  let p_blue_and_popped := blue_proportion * blue_pop_prob
  let p_popped := p_white_and_popped + p_yellow_and_popped + p_blue_and_popped
  p_white_and_popped / p_popped = prob_white_given_popped := by
  sorry

end NUMINAMATH_CALUDE_popcorn_probability_l4011_401154


namespace NUMINAMATH_CALUDE_abc_mod_five_l4011_401158

theorem abc_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 →
  (a + 2*b + 3*c) % 5 = 3 →
  (2*a + 3*b + c) % 5 = 2 →
  (3*a + b + 2*c) % 5 = 1 →
  (a*b*c) % 5 = 3 := by
  sorry

#check abc_mod_five

end NUMINAMATH_CALUDE_abc_mod_five_l4011_401158


namespace NUMINAMATH_CALUDE_perpendicular_similarity_l4011_401126

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop :=
  sorry -- Definition of acute triangle

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry -- Definition of point being inside a triangle

/-- Constructs a new triangle by dropping perpendiculars from a point to the sides of another triangle -/
def dropPerpendiculars (p : Point) (t : Triangle) : Triangle :=
  sorry -- Definition of dropping perpendiculars

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  sorry -- Definition of triangle similarity

theorem perpendicular_similarity 
  (ABC : Triangle) 
  (P : Point) 
  (h_acute : isAcute ABC) 
  (h_inside : isInside P ABC) : 
  let A₁B₁C₁ := dropPerpendiculars P ABC
  let A₂B₂C₂ := dropPerpendiculars P A₁B₁C₁
  let A₃B₃C₃ := dropPerpendiculars P A₂B₂C₂
  areSimilar A₃B₃C₃ ABC :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_similarity_l4011_401126


namespace NUMINAMATH_CALUDE_second_parentheses_zero_l4011_401186

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem second_parentheses_zero : 
  let x : ℝ := Real.sqrt 6
  (diamond x x = (x + x)^2 - (x - x)^2) ∧ (x - x = 0) := by sorry

end NUMINAMATH_CALUDE_second_parentheses_zero_l4011_401186


namespace NUMINAMATH_CALUDE_toonies_count_l4011_401191

/-- Represents the number of toonies in a set of coins --/
def num_toonies (total_coins : ℕ) (total_value : ℕ) : ℕ :=
  total_coins - (2 * total_coins - total_value)

/-- Theorem stating that given 10 coins with a total value of $14, 
    the number of $2 coins (toonies) is 4 --/
theorem toonies_count : num_toonies 10 14 = 4 := by
  sorry

#eval num_toonies 10 14  -- Should output 4

end NUMINAMATH_CALUDE_toonies_count_l4011_401191


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l4011_401129

theorem sqrt_eight_div_sqrt_two_equals_two : 
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l4011_401129


namespace NUMINAMATH_CALUDE_magic_money_box_theorem_l4011_401181

/-- Represents the state of the magic money box on a given day -/
structure BoxState :=
  (day : Nat)
  (value : Nat)

/-- Calculates the next day's value based on the current state and added coins -/
def nextDayValue (state : BoxState) (added : Nat) : Nat :=
  (state.value * (state.day + 2) + added)

/-- Simulates the magic money box for a week -/
def simulateWeek : Nat :=
  let monday := BoxState.mk 0 2
  let tuesday := BoxState.mk 1 (nextDayValue monday 5)
  let wednesday := BoxState.mk 2 (nextDayValue tuesday 10)
  let thursday := BoxState.mk 3 (nextDayValue wednesday 25)
  let friday := BoxState.mk 4 (nextDayValue thursday 50)
  let saturday := BoxState.mk 5 (nextDayValue friday 0)
  let sunday := BoxState.mk 6 (nextDayValue saturday 0)
  sunday.value

theorem magic_money_box_theorem : simulateWeek = 142240 := by
  sorry

end NUMINAMATH_CALUDE_magic_money_box_theorem_l4011_401181


namespace NUMINAMATH_CALUDE_num_sequences_mod_1000_l4011_401135

/-- The number of increasing sequences of positive integers satisfying the given conditions -/
def num_sequences : ℕ := sorry

/-- The upper bound for the sequence elements -/
def upper_bound : ℕ := 1007

/-- The length of the sequences -/
def sequence_length : ℕ := 12

/-- Predicate to check if a sequence satisfies the given conditions -/
def valid_sequence (b : Fin sequence_length → ℕ) : Prop :=
  (∀ i j : Fin sequence_length, i ≤ j → b i ≤ b j) ∧
  (∀ i : Fin sequence_length, b i ≤ upper_bound) ∧
  (∀ i : Fin sequence_length, Even (b i - i.val))

theorem num_sequences_mod_1000 :
  num_sequences % 1000 = 508 := by sorry

end NUMINAMATH_CALUDE_num_sequences_mod_1000_l4011_401135


namespace NUMINAMATH_CALUDE_probability_of_specific_selection_l4011_401140

/-- A bag containing balls of different colors -/
structure BagOfBalls where
  total : ℕ
  white : ℕ
  red : ℕ
  black : ℕ

/-- The probability of selecting balls with specific conditions -/
def probability_of_selection (bag : BagOfBalls) (selected : ℕ) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem probability_of_specific_selection : 
  let bag : BagOfBalls := ⟨20, 9, 5, 6⟩
  probability_of_selection bag 10 = 7 / 92378 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_selection_l4011_401140


namespace NUMINAMATH_CALUDE_reflected_polygon_area_equal_l4011_401163

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon with n vertices -/
structure Polygon (n : ℕ) where
  vertices : Fin n → Point

/-- Calculates the area of a polygon -/
def area (p : Polygon n) : ℝ := sorry

/-- Reflects a point across the midpoint of two other points -/
def reflect (p : Point) (a : Point) (b : Point) : Point := sorry

/-- Creates a new polygon by reflecting each vertex of the given polygon
    across the midpoint of the corresponding side of the regular 2009-gon -/
def reflectedPolygon (p : Polygon 2009) (regularPolygon : Polygon 2009) : Polygon 2009 := sorry

/-- Theorem stating that the area of the reflected polygon is equal to the area of the original polygon -/
theorem reflected_polygon_area_equal (p : Polygon 2009) (regularPolygon : Polygon 2009) :
  area (reflectedPolygon p regularPolygon) = area p := by sorry

end NUMINAMATH_CALUDE_reflected_polygon_area_equal_l4011_401163


namespace NUMINAMATH_CALUDE_squares_ending_in_76_l4011_401184

theorem squares_ending_in_76 : 
  {x : ℕ | x^2 % 100 = 76} = {24, 26, 74, 76} := by sorry

end NUMINAMATH_CALUDE_squares_ending_in_76_l4011_401184


namespace NUMINAMATH_CALUDE_parallelism_transitivity_l4011_401111

-- Define the types for lines and planes
variable {Line Plane : Type*}

-- Define the parallelism relation
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Define our specific geometric objects
variable (m n : Line) (α : Plane)

-- Define the property of being outside a plane
variable (outside : Line → Plane → Prop)

-- State the theorem
theorem parallelism_transitivity :
  (outside m α) → (outside n α) →
  (((parallel m n) ∧ (parallel_plane m α)) → (parallel_plane n α)) ∨
  (((parallel m n) ∧ (parallel_plane n α)) → (parallel_plane m α)) :=
sorry

end NUMINAMATH_CALUDE_parallelism_transitivity_l4011_401111


namespace NUMINAMATH_CALUDE_janes_calculation_l4011_401146

theorem janes_calculation (a b c : ℝ) 
  (h1 : a + b + c = 11) 
  (h2 : a + b - c = 19) : 
  a + b = 15 := by
sorry

end NUMINAMATH_CALUDE_janes_calculation_l4011_401146


namespace NUMINAMATH_CALUDE_total_winter_clothing_l4011_401161

def scarves_boxes : ℕ := 4
def scarves_per_box : ℕ := 8
def mittens_boxes : ℕ := 3
def mittens_per_box : ℕ := 6
def hats_boxes : ℕ := 2
def hats_per_box : ℕ := 5
def jackets_boxes : ℕ := 1
def jackets_per_box : ℕ := 3

theorem total_winter_clothing :
  scarves_boxes * scarves_per_box +
  mittens_boxes * mittens_per_box +
  hats_boxes * hats_per_box +
  jackets_boxes * jackets_per_box = 63 := by
sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l4011_401161


namespace NUMINAMATH_CALUDE_cattle_breeder_milk_production_l4011_401133

/-- Calculates the weekly milk production for a given number of cows and daily milk production per cow. -/
def weekly_milk_production (num_cows : ℕ) (daily_production : ℕ) : ℕ :=
  num_cows * daily_production * 7

/-- Proves that the weekly milk production of 52 cows, each producing 1000 oz of milk per day, is 364,000 oz. -/
theorem cattle_breeder_milk_production :
  weekly_milk_production 52 1000 = 364000 := by
  sorry


end NUMINAMATH_CALUDE_cattle_breeder_milk_production_l4011_401133


namespace NUMINAMATH_CALUDE_sasha_max_quarters_l4011_401138

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 32 / 10

/-- The maximum number of quarters Sasha can have -/
def max_quarters : ℕ := 10

theorem sasha_max_quarters :
  ∀ q : ℕ,
  (q : ℚ) * (quarter_value + nickel_value) ≤ total_amount →
  q ≤ max_quarters :=
by sorry

end NUMINAMATH_CALUDE_sasha_max_quarters_l4011_401138


namespace NUMINAMATH_CALUDE_function_period_l4011_401152

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_period (f : ℝ → ℝ) (h : ∀ x, f (x + 3) = -f x) :
  is_periodic f 6 :=
sorry

end NUMINAMATH_CALUDE_function_period_l4011_401152


namespace NUMINAMATH_CALUDE_triangular_number_difference_l4011_401185

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_number_difference : 
  triangular_number 30 - triangular_number 28 = 59 := by
sorry

end NUMINAMATH_CALUDE_triangular_number_difference_l4011_401185


namespace NUMINAMATH_CALUDE_repeating_block_length_l4011_401196

/-- The number of digits in the smallest repeating block of the decimal expansion of 4/7 -/
def smallest_repeating_block_length : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 4/7

theorem repeating_block_length :
  smallest_repeating_block_length = 6 ∧ 
  ∃ (n : ℕ) (d : ℕ+), fraction = n / d ∧ 
  smallest_repeating_block_length ≤ d - 1 :=
sorry

end NUMINAMATH_CALUDE_repeating_block_length_l4011_401196


namespace NUMINAMATH_CALUDE_binomial_square_condition_l4011_401169

theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (3*x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l4011_401169


namespace NUMINAMATH_CALUDE_paths_through_B_l4011_401122

/-- The number of paths between two points on a grid -/
def grid_paths (right : ℕ) (down : ℕ) : ℕ := Nat.choose (right + down) down

/-- The theorem stating the number of 11-step paths from A to C passing through B -/
theorem paths_through_B : 
  let paths_A_to_B := grid_paths 4 2
  let paths_B_to_C := grid_paths 3 3
  paths_A_to_B * paths_B_to_C = 300 := by sorry

end NUMINAMATH_CALUDE_paths_through_B_l4011_401122


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l4011_401101

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The first line equation: y = 3x + 7 -/
def line1 (x y : ℝ) : Prop := y = 3 * x + 7

/-- The second line equation: 4y + bx = 12 -/
def line2 (x y b : ℝ) : Prop := 4 * y + b * x = 12

/-- The theorem stating that if the two given lines are perpendicular, then b = 4/3 -/
theorem perpendicular_lines_b_value (b : ℝ) :
  (∀ x y, line1 x y → line2 x y b → perpendicular 3 (-b/4)) →
  b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l4011_401101


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l4011_401189

theorem negation_of_proposition (P : (ℝ → Prop)) :
  (¬ (∀ x : ℝ, x > 0 → P x)) ↔ (∃ x : ℝ, x > 0 ∧ ¬(P x)) :=
by sorry

-- Define the specific proposition
def Q (x : ℝ) : Prop := x^2 + 2*x - 3 ≥ 0

theorem negation_of_specific_proposition :
  (¬ (∀ x : ℝ, x > 0 → Q x)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + 2*x - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l4011_401189


namespace NUMINAMATH_CALUDE_line_l_passes_through_Q_max_distance_P_to_l_MN_length_range_l4011_401157

-- Define the line l with parameter m
def line_l (m : ℝ) (x y : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0

-- Define point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define point Q
def point_Q : ℝ × ℝ := (1, -2)

-- Define point N
def point_N : ℝ × ℝ := (2, 1)

theorem line_l_passes_through_Q :
  ∀ m : ℝ, line_l m (point_Q.1) (point_Q.2) := by sorry

theorem max_distance_P_to_l :
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧
  (∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧
    Real.sqrt ((x - point_P.1)^2 + (y - point_P.2)^2) ≤ d) ∧
  (∃ m : ℝ, ∃ x y : ℝ, line_l m x y ∧
    Real.sqrt ((x - point_P.1)^2 + (y - point_P.2)^2) = d) := by sorry

theorem MN_length_range :
  ∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧
    (x - point_P.1) * (x - point_P.1) + (y - point_P.2) * (y - point_P.2) =
    ((x - point_P.1)^2 + (y - point_P.2)^2) / 4 ∧
    Real.sqrt 2 ≤ Real.sqrt ((x - point_N.1)^2 + (y - point_N.2)^2) ∧
    Real.sqrt ((x - point_N.1)^2 + (y - point_N.2)^2) ≤ 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_line_l_passes_through_Q_max_distance_P_to_l_MN_length_range_l4011_401157


namespace NUMINAMATH_CALUDE_first_angle_measure_l4011_401198

theorem first_angle_measure (a b c : ℝ) : 
  a + b + c = 180 →  -- sum of angles in a triangle is 180 degrees
  b = 3 * a →        -- second angle is three times the first
  c = 2 * a - 12 →   -- third angle is 12 degrees less than twice the first
  a = 32 :=          -- prove that the first angle is 32 degrees
by sorry

end NUMINAMATH_CALUDE_first_angle_measure_l4011_401198


namespace NUMINAMATH_CALUDE_missing_number_proof_l4011_401105

/-- Given a list of 10 numbers with an average of 750, where 9 of the numbers are known,
    prove that the remaining number is 1747. -/
theorem missing_number_proof (numbers : List ℕ) (h1 : numbers.length = 10)
  (h2 : numbers.sum / numbers.length = 750)
  (h3 : numbers.count 744 = 1)
  (h4 : numbers.count 745 = 1)
  (h5 : numbers.count 748 = 1)
  (h6 : numbers.count 749 = 1)
  (h7 : numbers.count 752 = 2)
  (h8 : numbers.count 753 = 1)
  (h9 : numbers.count 755 = 2)
  : numbers.any (· = 1747) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l4011_401105


namespace NUMINAMATH_CALUDE_max_fleas_on_board_l4011_401162

/-- Represents a 10x10 board --/
def Board := Fin 10 → Fin 10 → Bool

/-- Represents the four possible directions of flea movement --/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a flea's position and direction --/
structure Flea where
  pos : Fin 10 × Fin 10
  dir : Direction

/-- Represents the state of the board and fleas at a given time --/
structure BoardState where
  board : Board
  fleas : List Flea

/-- Simulates the movement of fleas for one hour (60 minutes) --/
def simulateMovement (initialState : BoardState) : BoardState :=
  sorry

/-- Checks if the simulation results in a valid state (no overlapping fleas) --/
def isValidSimulation (finalState : BoardState) : Bool :=
  sorry

/-- Theorem stating the maximum number of fleas --/
theorem max_fleas_on_board :
  ∀ (initialState : BoardState),
    isValidSimulation (simulateMovement initialState) →
    initialState.fleas.length ≤ 40 :=
  sorry

end NUMINAMATH_CALUDE_max_fleas_on_board_l4011_401162


namespace NUMINAMATH_CALUDE_sin_240_degrees_l4011_401150

theorem sin_240_degrees : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l4011_401150


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l4011_401193

/-- The number of candy pieces Bobby ate first -/
def first_eaten : ℕ := 34

/-- The number of candy pieces Bobby ate later -/
def later_eaten : ℕ := 18

/-- The total number of candy pieces Bobby ate -/
def total_eaten : ℕ := first_eaten + later_eaten

theorem bobby_candy_consumption :
  total_eaten = 52 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l4011_401193


namespace NUMINAMATH_CALUDE_original_number_l4011_401115

theorem original_number (t : ℝ) : 
  t * (1 + 0.125) - t * (1 - 0.25) = 30 → t = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l4011_401115


namespace NUMINAMATH_CALUDE_probability_white_ball_l4011_401195

/-- The probability of drawing a white ball from a bag with specified numbers of colored balls. -/
theorem probability_white_ball (white red black : ℕ) : 
  white = 3 → red = 4 → black = 5 → (white : ℚ) / (white + red + black) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_l4011_401195


namespace NUMINAMATH_CALUDE_factorization_exists_l4011_401137

theorem factorization_exists : ∃ (a b c : ℤ), ∀ x : ℝ,
  (x - a) * (x - 10) + 1 = (x + b) * (x + c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_exists_l4011_401137


namespace NUMINAMATH_CALUDE_min_area_square_on_parabola_l4011_401142

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a parabola y = x^2 -/
def OnParabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Defines a square with three vertices on a parabola -/
structure SquareOnParabola where
  A : Point
  B : Point
  C : Point
  onParabola : OnParabola A ∧ OnParabola B ∧ OnParabola C
  isSquare : (A.x - B.x)^2 + (A.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2

/-- The area of a square given its side length -/
def SquareArea (sideLength : ℝ) : ℝ :=
  sideLength^2

/-- Theorem: The minimum area of a square with three vertices on the parabola y = x^2 is 2 -/
theorem min_area_square_on_parabola :
  ∀ s : SquareOnParabola, SquareArea (Real.sqrt ((s.A.x - s.B.x)^2 + (s.A.y - s.B.y)^2)) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_area_square_on_parabola_l4011_401142


namespace NUMINAMATH_CALUDE_miles_driven_with_thirty_dollars_l4011_401172

theorem miles_driven_with_thirty_dollars (miles_per_gallon : ℝ) (dollars_per_gallon : ℝ) (budget : ℝ) :
  miles_per_gallon = 40 →
  dollars_per_gallon = 4 →
  budget = 30 →
  (budget / dollars_per_gallon) * miles_per_gallon = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_miles_driven_with_thirty_dollars_l4011_401172


namespace NUMINAMATH_CALUDE_remainder_of_power_sum_l4011_401143

/-- The remainder when 5^94 + 7^94 is divided by 55 is 29. -/
theorem remainder_of_power_sum : (5^94 + 7^94) % 55 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_sum_l4011_401143


namespace NUMINAMATH_CALUDE_unique_496_consecutive_sum_l4011_401170

/-- Represents a sequence of consecutive positive integers -/
structure ConsecutiveSequence where
  start : Nat
  length : Nat

/-- Checks if a ConsecutiveSequence sums to the target value -/
def sumTo (seq : ConsecutiveSequence) (target : Nat) : Prop :=
  seq.length * seq.start + seq.length * (seq.length - 1) / 2 = target

/-- Checks if a ConsecutiveSequence is valid (length ≥ 2) -/
def isValid (seq : ConsecutiveSequence) : Prop :=
  seq.length ≥ 2

theorem unique_496_consecutive_sum :
  ∃! seq : ConsecutiveSequence, isValid seq ∧ sumTo seq 496 :=
sorry

end NUMINAMATH_CALUDE_unique_496_consecutive_sum_l4011_401170


namespace NUMINAMATH_CALUDE_cos_sin_inequality_solution_set_l4011_401117

open Real

theorem cos_sin_inequality_solution_set (x : ℝ) : 
  (cos x)^4 - 2 * sin x * cos x - (sin x)^4 - 1 > 0 ↔ 
  ∃ k : ℤ, x ∈ Set.Ioo (k * π - π/4) (k * π) := by sorry

end NUMINAMATH_CALUDE_cos_sin_inequality_solution_set_l4011_401117


namespace NUMINAMATH_CALUDE_gcd_105_88_l4011_401176

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l4011_401176


namespace NUMINAMATH_CALUDE_expression_equals_one_l4011_401116

theorem expression_equals_one (a b c : ℝ) :
  ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2) / ((a - b)^2 + (b - c)^2 + (c - a)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l4011_401116


namespace NUMINAMATH_CALUDE_M_properties_M_remainder_l4011_401165

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    n = d1 * 100000000 + d2 * 10000000 + d3 * 1000000 + d4 * 100000 + 
        d5 * 10000 + d6 * 1000 + d7 * 100 + d8 * 10 + d9 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    1 ≤ d1 ∧ d1 ≤ 9 ∧
    1 ≤ d2 ∧ d2 ≤ 9 ∧
    1 ≤ d3 ∧ d3 ≤ 9 ∧
    1 ≤ d4 ∧ d4 ≤ 9 ∧
    1 ≤ d5 ∧ d5 ≤ 9 ∧
    1 ≤ d6 ∧ d6 ≤ 9 ∧
    1 ≤ d7 ∧ d7 ≤ 9 ∧
    1 ≤ d8 ∧ d8 ≤ 9 ∧
    1 ≤ d9 ∧ d9 ≤ 9

def M : ℕ := sorry

theorem M_properties :
  is_valid_number M ∧ 
  M % 12 = 0 ∧
  ∀ n, is_valid_number n ∧ n % 12 = 0 → n ≤ M :=
by sorry

theorem M_remainder : M % 100 = 12 :=
by sorry

end NUMINAMATH_CALUDE_M_properties_M_remainder_l4011_401165


namespace NUMINAMATH_CALUDE_inverse_existence_l4011_401159

-- Define the three functions
def linear_function (x : ℝ) : ℝ := sorry
def quadratic_function (x : ℝ) : ℝ := sorry
def exponential_function (x : ℝ) : ℝ := sorry

-- Define the property of having an inverse
def has_inverse (f : ℝ → ℝ) : Prop := sorry

-- Theorem statement
theorem inverse_existence :
  (has_inverse linear_function) ∧
  (¬ has_inverse quadratic_function) ∧
  (has_inverse exponential_function) := by sorry

end NUMINAMATH_CALUDE_inverse_existence_l4011_401159


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l4011_401106

theorem sum_of_roots_zero (p q a b c : ℝ) : 
  a ≠ b → b ≠ c → a ≠ c →
  a^3 + p*a + q = 0 →
  b^3 + p*b + q = 0 →
  c^3 + p*c + q = 0 →
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l4011_401106


namespace NUMINAMATH_CALUDE_equilateral_triangle_pq_l4011_401183

/-- Given an equilateral triangle with vertices at (0,0), (p, 13), and (q, 41),
    prove that the product pq equals -2123/3 -/
theorem equilateral_triangle_pq (p q : ℝ) : 
  (∃ (z : ℂ), z^3 = 1 ∧ z ≠ 1 ∧ z * (p + 13*I) = q + 41*I) →
  p * q = -2123/3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_pq_l4011_401183
