import Mathlib

namespace NUMINAMATH_CALUDE_correct_average_weight_l3207_320717

def class_size : ℕ := 20
def initial_average : ℚ := 58.4
def misread_weight : ℕ := 56
def correct_weight : ℕ := 62

theorem correct_average_weight :
  let incorrect_total := initial_average * class_size
  let weight_difference := correct_weight - misread_weight
  let correct_total := incorrect_total + weight_difference
  (correct_total / class_size : ℚ) = 58.7 := by sorry

end NUMINAMATH_CALUDE_correct_average_weight_l3207_320717


namespace NUMINAMATH_CALUDE_cone_base_radius_l3207_320738

/-- 
Given a cone whose lateral surface, when unfolded, is a semicircle with radius 1,
prove that the radius of the base of the cone is 1/2.
-/
theorem cone_base_radius (r : ℝ) : r > 0 → r = 1 → (2 * π * (1 / 2 : ℝ)) = (π * r) → (1 / 2 : ℝ) = r := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3207_320738


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3207_320783

theorem quadratic_root_difference (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ x - y = Real.sqrt 77) →
  k ≤ Real.sqrt 109 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3207_320783


namespace NUMINAMATH_CALUDE_total_weight_lifted_l3207_320790

/-- Represents the weight a weightlifter can lift in one hand -/
def weight_per_hand : ℕ := 10

/-- Represents the number of hands a weightlifter has -/
def number_of_hands : ℕ := 2

/-- Theorem stating the total weight a weightlifter can lift -/
theorem total_weight_lifted : weight_per_hand * number_of_hands = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_lifted_l3207_320790


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3207_320727

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₇ = 3 and a₁₉ = 2011, prove that a₁₃ = 1007 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_7 : a 7 = 3) 
  (h_19 : a 19 = 2011) : 
  a 13 = 1007 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3207_320727


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l3207_320796

theorem complex_subtraction_simplification :
  (4 - 3*Complex.I) - (7 - 5*Complex.I) = -3 + 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l3207_320796


namespace NUMINAMATH_CALUDE_a_will_eat_hat_l3207_320746

-- Define the types of people
inductive Person : Type
| Knight : Person
| Liar : Person

-- Define the statement made by A about B
def statement_about_B (a b : Person) : Prop :=
  match a with
  | Person.Knight => b = Person.Knight
  | Person.Liar => True

-- Define A's statement about eating the hat
def statement_about_hat (a : Person) : Prop :=
  match a with
  | Person.Knight => True  -- Will eat the hat
  | Person.Liar => False   -- Won't eat the hat

-- Theorem statement
theorem a_will_eat_hat (a b : Person) :
  (statement_about_B a b = True) →
  (statement_about_hat a = True) := by
  sorry


end NUMINAMATH_CALUDE_a_will_eat_hat_l3207_320746


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l3207_320737

theorem complex_magnitude_example : Complex.abs (-5 + (8/3) * Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l3207_320737


namespace NUMINAMATH_CALUDE_quadratic_inequality_transformation_l3207_320771

theorem quadratic_inequality_transformation (a b c : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → a * x^2 + b * x + c > 0) →
  (∀ x, c * x^2 + b * x + a > 0 ↔ 1/2 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_transformation_l3207_320771


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_l3207_320734

theorem sqrt_difference_equals_seven : 
  Real.sqrt (36 + 64) - Real.sqrt (25 - 16) = 7 := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_l3207_320734


namespace NUMINAMATH_CALUDE_quadratic_integral_inequality_l3207_320775

/-- For real numbers a, b, c, let f(x) = ax^2 + bx + c. 
    Prove that ∫_{-1}^1 (1 - x^2){f'(x)}^2 dx ≤ 6∫_{-1}^1 {f(x)}^2 dx -/
theorem quadratic_integral_inequality (a b c : ℝ) : 
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  ∫ x in (-1)..1, (1 - x^2) * (deriv f x)^2 ≤ 6 * ∫ x in (-1)..1, (f x)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integral_inequality_l3207_320775


namespace NUMINAMATH_CALUDE_scientific_notation_of_billion_l3207_320707

theorem scientific_notation_of_billion (x : ℝ) (h : x = 61345.05) :
  x * (10 : ℝ)^9 = 6.134505 * (10 : ℝ)^12 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_billion_l3207_320707


namespace NUMINAMATH_CALUDE_calculate_e_l3207_320748

/-- Given the relationships between variables j, p, t, b, a, and e, prove that e = 21.5 -/
theorem calculate_e (j p t b a e : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_e_l3207_320748


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3207_320716

/-- Given a function y = x^α where α < 0, and a point A that lies on both y = x^α and y = mx + n 
    where m > 0 and n > 0, the minimum value of 1/m + 1/n is 4. -/
theorem min_value_reciprocal_sum (α m n : ℝ) (hα : α < 0) (hm : m > 0) (hn : n > 0) :
  (∃ x y : ℝ, y = x^α ∧ y = m*x + n) → 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → (∃ x' y' : ℝ, y' = x'^α ∧ y' = m'*x' + n') → 1/m + 1/n ≤ 1/m' + 1/n') →
  1/m + 1/n = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3207_320716


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3207_320797

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π/4 - α) = 3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3207_320797


namespace NUMINAMATH_CALUDE_comprehensive_survey_suitability_l3207_320794

/-- Represents a survey scenario --/
inductive SurveyScenario
  | CalculatorServiceLife
  | BeijingStudentsSpaceflightLogo
  | ClassmatesBadalingGreatWall
  | FoodPigmentContent

/-- Determines if a survey scenario is suitable for a comprehensive survey --/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.ClassmatesBadalingGreatWall => True
  | _ => False

/-- Theorem stating that the ClassmatesBadalingGreatWall scenario is the only one suitable for a comprehensive survey --/
theorem comprehensive_survey_suitability :
  ∀ (scenario : SurveyScenario),
    isSuitableForComprehensiveSurvey scenario ↔ scenario = SurveyScenario.ClassmatesBadalingGreatWall :=
by
  sorry

#check comprehensive_survey_suitability

end NUMINAMATH_CALUDE_comprehensive_survey_suitability_l3207_320794


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3207_320740

/-- Given an arithmetic sequence {a_n} where a₂ + 1 is the arithmetic mean of a₁ and a₄,
    the common difference of the sequence is 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_mean : a 1 + a 4 = 2 * (a 2 + 1))  -- a₂ + 1 is the arithmetic mean of a₁ and a₄
  : a 2 - a 1 = 2 :=  -- The common difference is 2
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3207_320740


namespace NUMINAMATH_CALUDE_largest_811_triple_l3207_320754

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

/-- Converts a list of base-8 digits to a base-10 number -/
def fromBase8 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a list of digits to a base-10 number -/
def toBase10 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is an 8-11 triple -/
def is811Triple (m : ℕ) : Prop :=
  let base8Digits := toBase8 m
  toBase10 base8Digits = 3 * m

/-- The largest 8-11 triple -/
def largestTriple : ℕ := 705

theorem largest_811_triple :
  is811Triple largestTriple ∧
  ∀ m : ℕ, m > largestTriple → ¬is811Triple m :=
by sorry

end NUMINAMATH_CALUDE_largest_811_triple_l3207_320754


namespace NUMINAMATH_CALUDE_product_of_first_six_terms_l3207_320725

/-- A geometric sequence with the given property -/
def GeometricSequenceWithProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) * a (2 * n) = 3^n

/-- The theorem to be proved -/
theorem product_of_first_six_terms
  (a : ℕ → ℝ)
  (h : GeometricSequenceWithProperty a) :
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 729 := by
  sorry

end NUMINAMATH_CALUDE_product_of_first_six_terms_l3207_320725


namespace NUMINAMATH_CALUDE_replacement_cost_theorem_l3207_320799

/-- Calculate the total cost of replacing cardio machines in multiple gyms -/
def total_replacement_cost (num_gyms : ℕ) (bikes_per_gym : ℕ) (treadmills_per_gym : ℕ) (ellipticals_per_gym : ℕ) (bike_cost : ℝ) : ℝ :=
  let total_bikes := num_gyms * bikes_per_gym
  let total_treadmills := num_gyms * treadmills_per_gym
  let total_ellipticals := num_gyms * ellipticals_per_gym
  let treadmill_cost := bike_cost * 1.5
  let elliptical_cost := treadmill_cost * 2
  total_bikes * bike_cost + total_treadmills * treadmill_cost + total_ellipticals * elliptical_cost

/-- Theorem: The total cost to replace all cardio machines in 20 gyms is $455,000 -/
theorem replacement_cost_theorem :
  total_replacement_cost 20 10 5 5 700 = 455000 := by
  sorry

end NUMINAMATH_CALUDE_replacement_cost_theorem_l3207_320799


namespace NUMINAMATH_CALUDE_birthday_attendees_l3207_320786

theorem birthday_attendees (n : ℕ) : 
  (12 * (n + 2) = 16 * n) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_birthday_attendees_l3207_320786


namespace NUMINAMATH_CALUDE_distance_foci_to_asymptotes_for_given_hyperbola_l3207_320778

/-- The distance from the foci to the asymptotes of a hyperbola -/
def distance_foci_to_asymptotes (a b : ℝ) : ℝ := b

/-- The equation of a hyperbola in standard form -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem distance_foci_to_asymptotes_for_given_hyperbola :
  ∀ x y : ℝ, is_hyperbola x y 1 3 → distance_foci_to_asymptotes 1 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_foci_to_asymptotes_for_given_hyperbola_l3207_320778


namespace NUMINAMATH_CALUDE_alpha_plus_beta_value_l3207_320714

theorem alpha_plus_beta_value (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 54*x + 621) / (x^2 + 42*x - 1764)) →
  α + β = 86 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_value_l3207_320714


namespace NUMINAMATH_CALUDE_min_shots_13x13_grid_l3207_320731

/-- Represents a grid with side length n -/
def Grid (n : ℕ) := Fin n × Fin n

/-- The set of possible moves for the target -/
def neighborMoves : List (ℤ × ℤ) :=
  [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

/-- Check if a move is valid within the grid -/
def isValidMove (n : ℕ) (pos : Grid n) (move : ℤ × ℤ) : Bool :=
  let (x, y) := pos
  let (dx, dy) := move
  0 ≤ x.val + dx ∧ x.val + dx < n ∧ 0 ≤ y.val + dy ∧ y.val + dy < n

/-- The minimum number of shots required to guarantee hitting the target twice -/
def minShotsToDestroy (n : ℕ) : ℕ :=
  n * n + (n * n + 1) / 2

/-- Theorem stating the minimum number of shots required for a 13x13 grid -/
theorem min_shots_13x13_grid :
  minShotsToDestroy 13 = 254 :=
sorry

end NUMINAMATH_CALUDE_min_shots_13x13_grid_l3207_320731


namespace NUMINAMATH_CALUDE_joe_total_cars_l3207_320763

def initial_cars : ℕ := 500
def additional_cars : ℕ := 120

theorem joe_total_cars : initial_cars + additional_cars = 620 := by
  sorry

end NUMINAMATH_CALUDE_joe_total_cars_l3207_320763


namespace NUMINAMATH_CALUDE_side_significant_digits_equal_area_significant_digits_l3207_320700

-- Define the area of the square
def square_area : ℝ := 2.3406

-- Define the precision of the area measurement (to the nearest ten-thousandth)
def area_precision : ℝ := 0.0001

-- Define the function to count significant digits
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Theorem statement
theorem side_significant_digits_equal_area_significant_digits :
  count_significant_digits (Real.sqrt square_area) = count_significant_digits square_area :=
sorry

end NUMINAMATH_CALUDE_side_significant_digits_equal_area_significant_digits_l3207_320700


namespace NUMINAMATH_CALUDE_workshop_theorem_l3207_320782

def workshop_problem (total_members : ℕ) (avg_age_all : ℝ) 
                     (num_girls : ℕ) (num_boys : ℕ) (num_adults : ℕ) 
                     (avg_age_girls : ℝ) (avg_age_boys : ℝ) : Prop :=
  let total_age := total_members * avg_age_all
  let girls_age := num_girls * avg_age_girls
  let boys_age := num_boys * avg_age_boys
  let adults_age := total_age - girls_age - boys_age
  (adults_age / num_adults) = 26.2

theorem workshop_theorem : 
  workshop_problem 50 20 22 18 10 18 19 := by
  sorry

end NUMINAMATH_CALUDE_workshop_theorem_l3207_320782


namespace NUMINAMATH_CALUDE_min_value_theorem_l3207_320711

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3207_320711


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3207_320724

theorem largest_divisor_of_n (n : ℕ+) (h : 100 ∣ n^3) :
  100 = Nat.gcd 100 n ∧ ∀ m : ℕ, m > 100 → ¬(m ∣ n) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3207_320724


namespace NUMINAMATH_CALUDE_vat_percentage_calculation_l3207_320715

theorem vat_percentage_calculation (original_price final_price : ℝ) : 
  original_price = 1700 → 
  final_price = 1955 → 
  (final_price - original_price) / original_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_vat_percentage_calculation_l3207_320715


namespace NUMINAMATH_CALUDE_simplify_expression_l3207_320745

theorem simplify_expression (r : ℝ) : 90 * r - 44 * r = 46 * r := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3207_320745


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3207_320755

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : a * b > 0) :
  (a > 0 ∧ b > 0 → 1 / a > 1 / b) ∧
  (a < 0 ∧ b < 0 → 1 / a < 1 / b) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3207_320755


namespace NUMINAMATH_CALUDE_triangle_sides_product_square_l3207_320733

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem triangle_sides_product_square (a b c : ℤ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive integers
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  (∀ d : ℤ, d > 1 → (d ∣ a ∧ d ∣ b ∧ d ∣ c) → False) →  -- No common divisor > 1
  (∃ k : ℤ, (a^2 + b^2 - c^2) = k * (a + b - c)) →  -- First fraction is integer
  (∃ l : ℤ, (b^2 + c^2 - a^2) = l * (b + c - a)) →  -- Second fraction is integer
  (∃ m : ℤ, (c^2 + a^2 - b^2) = m * (c + a - b)) →  -- Third fraction is integer
  (is_perfect_square ((a + b - c) * (b + c - a) * (c + a - b)) ∨ 
   is_perfect_square (2 * (a + b - c) * (b + c - a) * (c + a - b))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_product_square_l3207_320733


namespace NUMINAMATH_CALUDE_hall_volume_l3207_320728

/-- Given a rectangular hall with length 15 m and breadth 12 m, if the sum of the areas of
    the floor and ceiling is equal to the sum of the areas of four walls, then the volume
    of the hall is 8004 m³. -/
theorem hall_volume (height : ℝ) : 
  (15 : ℝ) * 12 * height = 8004 ∧ 
  2 * (15 * 12) = 2 * (15 * height) + 2 * (12 * height) := by
  sorry

#check hall_volume

end NUMINAMATH_CALUDE_hall_volume_l3207_320728


namespace NUMINAMATH_CALUDE_sue_shoe_probability_l3207_320758

/-- Represents the number of pairs for each shoe color --/
structure ShoeInventory where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the probability of picking two shoes of the same color and opposite types --/
def probabilitySameColorOppositeTypes (inventory : ShoeInventory) : Rat :=
  let totalShoes := 2 * (inventory.black + inventory.brown + inventory.gray + inventory.red)
  let prob_black := (2 * inventory.black) / totalShoes * inventory.black / (totalShoes - 1)
  let prob_brown := (2 * inventory.brown) / totalShoes * inventory.brown / (totalShoes - 1)
  let prob_gray := (2 * inventory.gray) / totalShoes * inventory.gray / (totalShoes - 1)
  let prob_red := (2 * inventory.red) / totalShoes * inventory.red / (totalShoes - 1)
  prob_black + prob_brown + prob_gray + prob_red

/-- Sue's shoe inventory --/
def sueInventory : ShoeInventory := ⟨7, 4, 2, 2⟩

theorem sue_shoe_probability :
  probabilitySameColorOppositeTypes sueInventory = 73 / 435 := by
  sorry

end NUMINAMATH_CALUDE_sue_shoe_probability_l3207_320758


namespace NUMINAMATH_CALUDE_vector_on_line_l3207_320788

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A line passing through two vectors p and q can be parameterized as p + t(q - p) for some real t -/
def line_through (p q : V) (t : ℝ) : V := p + t • (q - p)

/-- The theorem states that if m*p + 5/8*q lies on the line through p and q, then m = 3/8 -/
theorem vector_on_line (p q : V) (m : ℝ) 
  (h : ∃ t : ℝ, m • p + (5/8) • q = line_through p q t) : 
  m = 3/8 := by
sorry

end NUMINAMATH_CALUDE_vector_on_line_l3207_320788


namespace NUMINAMATH_CALUDE_parallel_segments_y_coordinate_l3207_320719

/-- Given four points A, B, X, Y on a Cartesian plane where AB is parallel to XY,
    prove that the y-coordinate of Y is 5. -/
theorem parallel_segments_y_coordinate (A B X Y : ℝ × ℝ) : 
  A = (-6, 2) →
  B = (2, -2) →
  X = (-2, 10) →
  Y.1 = 8 →
  (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1) →
  Y.2 = 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_segments_y_coordinate_l3207_320719


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_l3207_320776

/-- Given a rectangle with sides a and b, and an inscribed quadrilateral with vertices on each side
of the rectangle, the perimeter of the quadrilateral is greater than or equal to 2√(a² + b²). -/
theorem inscribed_quadrilateral_perimeter (a b : ℝ) (x y z t : ℝ)
  (hx : 0 ≤ x ∧ x ≤ a) (hy : 0 ≤ y ∧ y ≤ b) (hz : 0 ≤ z ∧ z ≤ a) (ht : 0 ≤ t ∧ t ≤ b) :
  let perimeter := Real.sqrt ((a - x)^2 + t^2) + Real.sqrt ((b - t)^2 + z^2) +
                   Real.sqrt ((a - z)^2 + (b - y)^2) + Real.sqrt (x^2 + y^2)
  perimeter ≥ 2 * Real.sqrt (a^2 + b^2) := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_l3207_320776


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l3207_320792

theorem drum_capacity_ratio :
  ∀ (C_X C_Y : ℝ),
  C_X > 0 → C_Y > 0 →
  (1/2 * C_X) + (1/4 * C_Y) = 1/2 * C_Y →
  C_Y / C_X = 2 := by
sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l3207_320792


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l3207_320770

/-- A decagon is a polygon with 10 vertices -/
def Decagon := { n : ℕ // n = 10 }

/-- The number of vertices in a decagon -/
def num_vertices (d : Decagon) : ℕ := d.val

/-- The number of adjacent vertices for any vertex in a decagon -/
def num_adjacent_vertices (d : Decagon) : ℕ := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  (num_adjacent_vertices d : ℚ) / ((num_vertices d - 1) : ℚ)

/-- Theorem: The probability of choosing two distinct adjacent vertices in a decagon is 2/9 -/
theorem prob_adjacent_vertices_decagon (d : Decagon) :
  prob_adjacent_vertices d = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l3207_320770


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisibility_l3207_320709

theorem product_of_three_consecutive_integers_divisibility
  (k : ℤ)
  (n : ℤ)
  (h1 : n = k * (k + 1) * (k + 2))
  (h2 : 5 ∣ n) :
  (6 ∣ n) ∧
  (10 ∣ n) ∧
  (15 ∣ n) ∧
  (30 ∣ n) ∧
  ∃ m : ℤ, n = m ∧ ¬(20 ∣ m) := by
sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisibility_l3207_320709


namespace NUMINAMATH_CALUDE_range_of_a_l3207_320729

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + a > 0

-- Define the theorem
theorem range_of_a (a : ℝ) : (¬(p a) ∧ q a) → (1 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3207_320729


namespace NUMINAMATH_CALUDE_twelve_buses_required_l3207_320793

/-- The minimum number of buses required to transport all students -/
def min_buses (max_capacity : ℕ) (total_students : ℕ) (available_drivers : ℕ) : ℕ :=
  max (((total_students + max_capacity - 1) / max_capacity) : ℕ) available_drivers

/-- Proof that 12 buses are required given the problem conditions -/
theorem twelve_buses_required :
  min_buses 42 480 12 = 12 := by
  sorry

#eval min_buses 42 480 12  -- Should output 12

end NUMINAMATH_CALUDE_twelve_buses_required_l3207_320793


namespace NUMINAMATH_CALUDE_louis_age_l3207_320761

/-- Given the ages of Matilda, Jerica, and Louis, prove Louis' age -/
theorem louis_age (matilda_age jerica_age louis_age : ℕ) : 
  matilda_age = 35 →
  matilda_age = jerica_age + 7 →
  jerica_age = 2 * louis_age →
  louis_age = 14 := by
  sorry

#check louis_age

end NUMINAMATH_CALUDE_louis_age_l3207_320761


namespace NUMINAMATH_CALUDE_integer_representation_l3207_320702

theorem integer_representation (n : ℤ) : ∃ x y z : ℤ, n = x^2 + y^2 - z^2 := by
  sorry

end NUMINAMATH_CALUDE_integer_representation_l3207_320702


namespace NUMINAMATH_CALUDE_expression_equality_l3207_320747

theorem expression_equality : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3207_320747


namespace NUMINAMATH_CALUDE_combined_diving_depths_l3207_320708

theorem combined_diving_depths (ron_height : ℝ) (water_depth : ℝ) : 
  ron_height = 12 →
  water_depth = 5 * ron_height →
  let dean_height := ron_height - 11
  let sam_height := dean_height + 2
  let ron_dive := ron_height / 2
  let sam_dive := sam_height
  let dean_dive := dean_height + 3
  ron_dive + sam_dive + dean_dive = 13 := by sorry

end NUMINAMATH_CALUDE_combined_diving_depths_l3207_320708


namespace NUMINAMATH_CALUDE_probability_calm_in_mathematics_l3207_320795

def letters_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}
def letters_calm : Finset Char := {'C', 'A', 'L', 'M'}

def count_occurrences (c : Char) : ℕ :=
  if c = 'M' ∨ c = 'A' then 2
  else if c ∈ letters_mathematics then 1
  else 0

def favorable_outcomes : ℕ := (letters_calm ∩ letters_mathematics).sum count_occurrences

theorem probability_calm_in_mathematics :
  (favorable_outcomes : ℚ) / 12 = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_calm_in_mathematics_l3207_320795


namespace NUMINAMATH_CALUDE_subtraction_result_l3207_320742

-- Define the two numbers
def a : ℚ := 888.88
def b : ℚ := 555.55

-- Define the result
def result : ℚ := a - b

-- Theorem to prove
theorem subtraction_result : result = 333.33 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3207_320742


namespace NUMINAMATH_CALUDE_adjacent_probability_in_two_by_three_l3207_320739

/-- Represents a 2x3 seating arrangement -/
def SeatingArrangement := Fin 2 → Fin 3 → Fin 6

/-- Two positions are adjacent if they are next to each other in the same row or column -/
def adjacent (pos1 pos2 : Fin 2 × Fin 3) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2.val + 1 = pos2.2.val ∨ pos2.2.val + 1 = pos1.2.val)) ∨
  (pos1.2 = pos2.2 ∧ pos1.1 ≠ pos2.1)

/-- The probability of two specific students being adjacent in a random seating arrangement -/
def probability_adjacent : ℚ :=
  7 / 15

theorem adjacent_probability_in_two_by_three :
  probability_adjacent = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_adjacent_probability_in_two_by_three_l3207_320739


namespace NUMINAMATH_CALUDE_vacation_rental_families_l3207_320772

/-- The number of people in each family -/
def family_size : ℕ := 4

/-- The number of days of the vacation -/
def vacation_days : ℕ := 7

/-- The number of towels each person uses per day -/
def towels_per_person_per_day : ℕ := 1

/-- The capacity of the washing machine in towels -/
def washing_machine_capacity : ℕ := 14

/-- The number of loads needed to wash all towels -/
def total_loads : ℕ := 6

/-- The number of families sharing the vacation rental -/
def num_families : ℕ := 3

theorem vacation_rental_families :
  num_families * family_size * vacation_days * towels_per_person_per_day =
  total_loads * washing_machine_capacity := by sorry

end NUMINAMATH_CALUDE_vacation_rental_families_l3207_320772


namespace NUMINAMATH_CALUDE_casper_candy_problem_l3207_320774

def candy_distribution (initial : ℕ) : ℕ :=
  let day1 := initial * 3 / 4 - 3
  let day2 := day1 * 4 / 5 - 5
  let day3 := day2 * 5 / 6 - 6
  day3

theorem casper_candy_problem :
  ∃ (initial : ℕ), candy_distribution initial = 10 ∧ initial = 678 :=
sorry

end NUMINAMATH_CALUDE_casper_candy_problem_l3207_320774


namespace NUMINAMATH_CALUDE_rowan_rowing_distance_l3207_320732

-- Define the given constants
def downstream_time : ℝ := 2
def upstream_time : ℝ := 4
def still_water_speed : ℝ := 9.75

-- Define the variables
def current_speed : ℝ := sorry
def distance : ℝ := sorry

-- State the theorem
theorem rowan_rowing_distance :
  downstream_time = distance / (still_water_speed + current_speed) ∧
  upstream_time = distance / (still_water_speed - current_speed) →
  distance = 26 := by sorry

end NUMINAMATH_CALUDE_rowan_rowing_distance_l3207_320732


namespace NUMINAMATH_CALUDE_focal_chord_length_l3207_320789

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : y^2 = 4*x

/-- Represents a line passing through the focal point of the parabola -/
structure FocalLine where
  A : ParabolaPoint
  B : ParabolaPoint
  sum_x : A.x + B.x = 6

/-- Theorem: The length of AB is 8 for the given conditions -/
theorem focal_chord_length (line : FocalLine) : 
  Real.sqrt ((line.B.x - line.A.x)^2 + (line.B.y - line.A.y)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_focal_chord_length_l3207_320789


namespace NUMINAMATH_CALUDE_largest_three_digit_special_divisible_l3207_320705

theorem largest_three_digit_special_divisible : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) → m ≤ n) ∧
  (n % 6 = 0) ∧
  (∀ d : ℕ, d > 0 ∧ d ≤ 9 ∧ (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d) → n % d = 0) ∧
  n = 843 := by
sorry

end NUMINAMATH_CALUDE_largest_three_digit_special_divisible_l3207_320705


namespace NUMINAMATH_CALUDE_boxes_in_smallest_cube_l3207_320718

def box_width : ℕ := 8
def box_length : ℕ := 12
def box_height : ℕ := 30

def smallest_cube_side : ℕ := lcm (lcm box_width box_length) box_height

def box_volume : ℕ := box_width * box_length * box_height
def cube_volume : ℕ := smallest_cube_side ^ 3

theorem boxes_in_smallest_cube :
  cube_volume / box_volume = 600 := by sorry

end NUMINAMATH_CALUDE_boxes_in_smallest_cube_l3207_320718


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3207_320713

theorem polynomial_factorization (x y z : ℝ) : 
  x^3 * (y - z) + y^3 * (z - x) + z^3 * (x - y) = (x + y + z) * (x - y) * (y - z) * (z - x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3207_320713


namespace NUMINAMATH_CALUDE_sine_even_function_phi_l3207_320706

theorem sine_even_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 6)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, f (x - φ) = f (φ - x)) →
  φ = π / 3 := by sorry

end NUMINAMATH_CALUDE_sine_even_function_phi_l3207_320706


namespace NUMINAMATH_CALUDE_opposite_of_one_third_l3207_320765

theorem opposite_of_one_third : 
  -(1/3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_third_l3207_320765


namespace NUMINAMATH_CALUDE_min_triangle_area_l3207_320764

/-- The minimum area of the triangle formed by a line passing through (1, 2) 
    and intersecting the positive x and y axes is 4. -/
theorem min_triangle_area (k : ℝ) (h : k < 0) : 
  let f (x : ℝ) := k * (x - 1) + 2
  let x_intercept := 1 - 2 / k
  let y_intercept := f 0
  let area := (1/2) * x_intercept * y_intercept
  ∀ k, k < 0 → area ≥ 4 ∧ (area = 4 ↔ k = -2) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l3207_320764


namespace NUMINAMATH_CALUDE_freshman_percentage_l3207_320703

theorem freshman_percentage (total_students : ℝ) (freshman : ℝ) 
  (h1 : freshman > 0)
  (h2 : total_students > 0)
  (h3 : (0.2 * 0.4 * freshman) / total_students = 0.04) :
  freshman / total_students = 0.5 := by
sorry

end NUMINAMATH_CALUDE_freshman_percentage_l3207_320703


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l3207_320752

theorem r_value_when_n_is_3 (n m s r : ℕ) :
  m = 3 ∧ s = 2^n - m ∧ r = 3^s + s ∧ n = 3 → r = 248 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l3207_320752


namespace NUMINAMATH_CALUDE_beads_taken_out_l3207_320753

/-- Represents the number of beads in a container -/
structure BeadContainer where
  green : Nat
  brown : Nat
  red : Nat

/-- Calculates the total number of beads in a container -/
def totalBeads (container : BeadContainer) : Nat :=
  container.green + container.brown + container.red

theorem beads_taken_out (initial : BeadContainer) (left : Nat) :
  totalBeads initial = 6 → left = 4 → totalBeads initial - left = 2 := by
  sorry

end NUMINAMATH_CALUDE_beads_taken_out_l3207_320753


namespace NUMINAMATH_CALUDE_min_value_expression_l3207_320784

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = 3 * Real.sqrt (5 / 13) ∧
  ∀ (x : ℝ), x = (Real.sqrt ((a^2 + 2*b^2) * (4*a^2 + b^2))) / (a * b) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3207_320784


namespace NUMINAMATH_CALUDE_baseball_cards_distribution_l3207_320759

theorem baseball_cards_distribution (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) :
  total_cards = 24 →
  num_friends = 4 →
  total_cards = num_friends * cards_per_friend →
  cards_per_friend = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_distribution_l3207_320759


namespace NUMINAMATH_CALUDE_min_tenth_game_score_l3207_320750

/-- Represents the scores of a basketball player in a series of games -/
structure BasketballScores where
  first_five : ℝ  -- Total score of first 5 games
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ
  ninth : ℝ
  tenth : ℝ

/-- Theorem stating the minimum score required for the 10th game -/
theorem min_tenth_game_score (scores : BasketballScores) 
  (h1 : scores.sixth = 23)
  (h2 : scores.seventh = 14)
  (h3 : scores.eighth = 11)
  (h4 : scores.ninth = 20)
  (h5 : (scores.first_five + scores.sixth + scores.seventh + scores.eighth + scores.ninth) / 9 > 
        scores.first_five / 5)
  (h6 : (scores.first_five + scores.sixth + scores.seventh + scores.eighth + scores.ninth + scores.tenth) / 10 > 18) :
  scores.tenth ≥ 29 := by
  sorry

end NUMINAMATH_CALUDE_min_tenth_game_score_l3207_320750


namespace NUMINAMATH_CALUDE_negation_equivalence_l3207_320767

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3207_320767


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3207_320723

theorem square_area_from_perimeter (p : ℝ) : 
  let perimeter : ℝ := 12 * p
  let side_length : ℝ := perimeter / 4
  let area : ℝ := side_length ^ 2
  area = 9 * p ^ 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3207_320723


namespace NUMINAMATH_CALUDE_connie_markers_total_l3207_320757

theorem connie_markers_total (red : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ)
  (h_red : red = 5420)
  (h_blue : blue = 3875)
  (h_green : green = 2910)
  (h_yellow : yellow = 6740) :
  red + blue + green + yellow = 18945 := by
  sorry

end NUMINAMATH_CALUDE_connie_markers_total_l3207_320757


namespace NUMINAMATH_CALUDE_even_function_inequality_l3207_320704

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is increasing on (-∞, 0] -/
def IsIncreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

/-- Main theorem -/
theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ)
  (h_even : IsEven f)
  (h_incr : IsIncreasingOnNegative f)
  (h_ineq : f a ≤ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3207_320704


namespace NUMINAMATH_CALUDE_distribute_and_simplify_l3207_320735

theorem distribute_and_simplify (a : ℝ) : a * (a - 3) = a^2 - 3*a := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_simplify_l3207_320735


namespace NUMINAMATH_CALUDE_share_ratio_l3207_320780

/-- Given a total amount of $500 divided among three people a, b, and c,
    where a's share is $200, a gets a fraction of b and c's combined share,
    and b gets 6/9 of a and c's combined share, prove that the ratio of
    a's share to the combined share of b and c is 2:3. -/
theorem share_ratio (total : ℚ) (a b c : ℚ) :
  total = 500 →
  a = 200 →
  ∃ x : ℚ, a = x * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a / (b + c) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_l3207_320780


namespace NUMINAMATH_CALUDE_x_value_l3207_320741

theorem x_value (x y : ℚ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3207_320741


namespace NUMINAMATH_CALUDE_sarah_walking_speed_l3207_320721

theorem sarah_walking_speed (v : ℝ) : 
  v > 0 → -- v is positive (walking speed)
  (6 / v + 6 / 4 = 3.5) → -- total time equation
  v = 3 := by
sorry

end NUMINAMATH_CALUDE_sarah_walking_speed_l3207_320721


namespace NUMINAMATH_CALUDE_ice_cream_scoop_permutations_l3207_320785

theorem ice_cream_scoop_permutations :
  Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_permutations_l3207_320785


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3207_320730

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the interval [-3, 1]
def interval : Set ℝ := Set.Icc (-3) 1

-- Theorem statement
theorem min_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -11 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3207_320730


namespace NUMINAMATH_CALUDE_x_less_than_y_l3207_320744

theorem x_less_than_y (x y : ℝ) (h : (2023 : ℝ)^x + (2024 : ℝ)^(-y) < (2023 : ℝ)^y + (2024 : ℝ)^(-x)) : x < y := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l3207_320744


namespace NUMINAMATH_CALUDE_smallest_prime_is_two_l3207_320787

theorem smallest_prime_is_two (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p ≠ q → p ≠ r → q ≠ r →
  p^3 + q^3 + 3*p*q*r = r^3 →
  min p (min q r) = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_is_two_l3207_320787


namespace NUMINAMATH_CALUDE_negation_of_existential_quantifier_negation_of_inequality_l3207_320749

theorem negation_of_existential_quantifier (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_quantifier_negation_of_inequality_l3207_320749


namespace NUMINAMATH_CALUDE_group_size_calculation_l3207_320756

theorem group_size_calculation (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 55)
  (h2 : norway = 33)
  (h3 : both = 51)
  (h4 : neither = 53) :
  iceland + norway - both + neither = 90 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l3207_320756


namespace NUMINAMATH_CALUDE_number_operation_l3207_320762

theorem number_operation (x : ℚ) : x - 7/3 = 3/2 → x + 7/3 = 37/6 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l3207_320762


namespace NUMINAMATH_CALUDE_problem_proof_l3207_320712

theorem problem_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ a*b ≤ x*y) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → a*b ≤ 1/8) ∧
  ((1/b) + (b/a) ≥ 4) ∧
  (a^2 + b^2 ≥ 1/5) :=
sorry

end NUMINAMATH_CALUDE_problem_proof_l3207_320712


namespace NUMINAMATH_CALUDE_polynomial_coefficient_difference_l3207_320768

theorem polynomial_coefficient_difference (a b : ℝ) : 
  (∀ x, (1 + x) + (1 + x)^4 = 2 + 5*x + a*x^2 + b*x^3 + x^4) → 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_difference_l3207_320768


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_y_value_l3207_320773

/-- Given two vectors a and b in ℝ², prove that if a = (2,5) and b = (1,y) are parallel, then y = 5/2 -/
theorem parallel_vectors_imply_y_value (a b : ℝ × ℝ) (y : ℝ) :
  a = (2, 5) →
  b = (1, y) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  y = 5/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_y_value_l3207_320773


namespace NUMINAMATH_CALUDE_dime_probability_l3207_320798

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Quarter => 1250
  | Coin.Dime => 500
  | Coin.Penny => 250

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Penny

/-- The probability of randomly choosing a dime from the jar -/
def probDime : ℚ := coinCount Coin.Dime / totalCoins

theorem dime_probability : probDime = 1 / 7 := by
  sorry

#eval probDime

end NUMINAMATH_CALUDE_dime_probability_l3207_320798


namespace NUMINAMATH_CALUDE_matthew_sharing_l3207_320701

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 14

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := 21

/-- The number of crackers each friend received -/
def crackers_per_friend : ℕ := 5

/-- The number of cakes each friend received -/
def cakes_per_friend : ℕ := 5

/-- The maximum number of friends Matthew could share with -/
def max_friends : ℕ := 3

theorem matthew_sharing :
  max_friends = min (initial_crackers / crackers_per_friend) (initial_cakes / cakes_per_friend) :=
by sorry

end NUMINAMATH_CALUDE_matthew_sharing_l3207_320701


namespace NUMINAMATH_CALUDE_drums_per_day_l3207_320777

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums are filled per day. -/
theorem drums_per_day : 
  ∀ (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ),
    total_drums = 2916 →
    total_days = 9 →
    drums_per_day = total_drums / total_days →
    drums_per_day = 324 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l3207_320777


namespace NUMINAMATH_CALUDE_river_road_cars_l3207_320760

theorem river_road_cars (buses cars : ℕ) : 
  buses * 10 = cars ∧ cars - buses = 90 → cars = 100 := by
  sorry

end NUMINAMATH_CALUDE_river_road_cars_l3207_320760


namespace NUMINAMATH_CALUDE_binary_representation_of_2015_l3207_320751

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_representation_of_2015 :
  toBinary 2015 = [true, true, true, true, true, false, true, true, true, true, true] :=
by sorry

#eval fromBinary [true, true, true, true, true, false, true, true, true, true, true]

end NUMINAMATH_CALUDE_binary_representation_of_2015_l3207_320751


namespace NUMINAMATH_CALUDE_cube_increase_theorem_l3207_320736

def cube_edge_increase_percent : ℝ := 60

theorem cube_increase_theorem (s : ℝ) (h : s > 0) :
  let new_edge := s * (1 + cube_edge_increase_percent / 100)
  let original_surface_area := 6 * s^2
  let new_surface_area := 6 * new_edge^2
  let original_volume := s^3
  let new_volume := new_edge^3
  (new_surface_area - original_surface_area) / original_surface_area * 100 = 156 ∧
  (new_volume - original_volume) / original_volume * 100 = 309.6 := by
sorry


end NUMINAMATH_CALUDE_cube_increase_theorem_l3207_320736


namespace NUMINAMATH_CALUDE_only_B_in_fourth_quadrant_l3207_320791

def point_A : ℝ × ℝ := (2, 3)
def point_B : ℝ × ℝ := (1, -1)
def point_C : ℝ × ℝ := (-2, 1)
def point_D : ℝ × ℝ := (-2, -1)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem only_B_in_fourth_quadrant :
  in_fourth_quadrant point_B ∧
  ¬in_fourth_quadrant point_A ∧
  ¬in_fourth_quadrant point_C ∧
  ¬in_fourth_quadrant point_D :=
by sorry

end NUMINAMATH_CALUDE_only_B_in_fourth_quadrant_l3207_320791


namespace NUMINAMATH_CALUDE_sum_is_composite_l3207_320722

theorem sum_is_composite (m n : ℕ) (h : 88 * m = 81 * n) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ m + n = a * b := by
  sorry

end NUMINAMATH_CALUDE_sum_is_composite_l3207_320722


namespace NUMINAMATH_CALUDE_mark_bench_press_l3207_320710

-- Define the given conditions
def dave_weight : ℝ := 175
def dave_bench_press_multiplier : ℝ := 3
def craig_bench_press_percentage : ℝ := 0.20
def emma_bench_press_percentage : ℝ := 0.75
def emma_bench_press_increase : ℝ := 15
def john_bench_press_multiplier : ℝ := 2
def mark_bench_press_difference : ℝ := 50

-- Define the theorem
theorem mark_bench_press :
  let dave_bench_press := dave_weight * dave_bench_press_multiplier
  let craig_bench_press := craig_bench_press_percentage * dave_bench_press
  let emma_bench_press := emma_bench_press_percentage * dave_bench_press + emma_bench_press_increase
  let combined_craig_emma := craig_bench_press + emma_bench_press
  let mark_bench_press := combined_craig_emma - mark_bench_press_difference
  mark_bench_press = 463.75 := by
  sorry

end NUMINAMATH_CALUDE_mark_bench_press_l3207_320710


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3207_320726

/-- Given a 3-4-5 right triangle, let x be the side length of a square inscribed
    with one vertex at the right angle, and y be the side length of a square
    inscribed with one side on the hypotenuse. -/
theorem inscribed_squares_ratio (x y : ℝ) 
  (hx : x * (7 / 3) = 4) -- Derived from the condition for x
  (hy : y * (37 / 12) = 5) -- Derived from the condition for y
  : x / y = 37 / 35 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3207_320726


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3207_320766

/-- Represents the construction of cube frames by Bonnie and Roark -/
structure CubeFrames where
  bonnie_wire_length : ℕ := 8
  bonnie_wire_pieces : ℕ := 12
  roark_wire_length : ℕ := 2

/-- Theorem stating the ratio of wire lengths used by Bonnie and Roark -/
theorem wire_length_ratio (cf : CubeFrames) : 
  (cf.bonnie_wire_length * cf.bonnie_wire_pieces : ℚ) / 
  (cf.roark_wire_length * 12 * (cf.bonnie_wire_length ^ 3 / cf.roark_wire_length ^ 3)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3207_320766


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3207_320720

theorem circle_area_from_circumference (c : ℝ) (h : c = 36) : 
  (c^2 / (4 * π)) = 324 / π := by sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3207_320720


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3207_320779

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ -4 ∧ x ≠ 2/3 →
    (7 * x - 15) / (3 * x^2 + 2 * x - 8) = A / (x + 4) + B / (3 * x - 2)) →
  A = 43 / 14 ∧ B = -31 / 14 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3207_320779


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l3207_320781

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l3207_320781


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3207_320769

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Side-angle relationship
  c * Real.sin A = a * Real.cos C →
  -- Conclusion
  C = π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3207_320769


namespace NUMINAMATH_CALUDE_max_d_value_l3207_320743

def a (n : ℕ+) : ℕ := 150 + 3 * n.val ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 147) ∧ (∀ (n : ℕ+), d n ≤ 147) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3207_320743
