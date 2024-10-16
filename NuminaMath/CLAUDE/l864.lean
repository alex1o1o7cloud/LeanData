import Mathlib

namespace NUMINAMATH_CALUDE_sin_75_degrees_l864_86489

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l864_86489


namespace NUMINAMATH_CALUDE_half_plus_six_equals_eleven_l864_86488

theorem half_plus_six_equals_eleven (n : ℝ) : (1/2) * n + 6 = 11 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_six_equals_eleven_l864_86488


namespace NUMINAMATH_CALUDE_segment_length_ratio_l864_86483

/-- Given a line segment AD with points B and C on it, prove that BC + DE = 5/8 * AD -/
theorem segment_length_ratio (A B C D E : ℝ) : 
  B ∈ Set.Icc A D → -- B is on segment AD
  C ∈ Set.Icc A D → -- C is on segment AD
  B - A = 3 * (D - B) → -- AB = 3 * BD
  C - A = 7 * (D - C) → -- AC = 7 * CD
  E - D = C - B → -- DE = BC
  E - A = D - E → -- E is midpoint of AD
  C - B + E - D = 5/8 * (D - A) := by sorry

end NUMINAMATH_CALUDE_segment_length_ratio_l864_86483


namespace NUMINAMATH_CALUDE_family_weights_calculation_l864_86481

/-- Represents the weights of three generations in a family -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- Given the total weight of all three, the weight of daughter and grandchild, 
    and the relation between grandmother and grandchild weights, 
    prove the individual weights -/
theorem family_weights_calculation (w : FamilyWeights) : 
  w.grandmother + w.daughter + w.grandchild = 110 →
  w.daughter + w.grandchild = 60 →
  w.grandchild = w.grandmother / 5 →
  w.grandmother = 50 ∧ w.daughter = 50 ∧ w.grandchild = 10 := by
  sorry


end NUMINAMATH_CALUDE_family_weights_calculation_l864_86481


namespace NUMINAMATH_CALUDE_unique_divisibility_function_l864_86407

/-- A function from positive integers to positive integers -/
def NatFunction := ℕ+ → ℕ+

/-- The property that f(m) + f(n) divides m + n for all m and n -/
def HasDivisibilityProperty (f : NatFunction) : Prop :=
  ∀ m n : ℕ+, (f m + f n) ∣ (m + n)

/-- The identity function on positive integers -/
def identityFunction : NatFunction := fun x => x

/-- Theorem stating that the identity function is the only function satisfying the divisibility property -/
theorem unique_divisibility_function :
  ∀ f : NatFunction, HasDivisibilityProperty f ↔ f = identityFunction := by
  sorry

end NUMINAMATH_CALUDE_unique_divisibility_function_l864_86407


namespace NUMINAMATH_CALUDE_max_diagonal_value_l864_86413

/-- Represents the value at position (row, col) in the table -/
def tableValue (n : ℕ) (row col : ℕ) : ℕ :=
  1 + (col - 1) * row

/-- Represents the value on the diagonal from bottom-left to top-right -/
def diagonalValue (n : ℕ) (k : ℕ) : ℕ :=
  tableValue n k (n - k + 1)

/-- The size of the table -/
def tableSize : ℕ := 100

theorem max_diagonal_value :
  (∀ k, k ≤ tableSize → diagonalValue tableSize k ≤ 2501) ∧
  (∃ k, k ≤ tableSize ∧ diagonalValue tableSize k = 2501) :=
by sorry

end NUMINAMATH_CALUDE_max_diagonal_value_l864_86413


namespace NUMINAMATH_CALUDE_initial_birds_count_l864_86425

/-- The number of birds initially sitting in a tree -/
def initial_birds : ℕ := sorry

/-- The number of birds that flew up to join the initial birds -/
def additional_birds : ℕ := 81

/-- The total number of birds after additional birds joined -/
def total_birds : ℕ := 312

/-- Theorem stating that the number of birds initially sitting in the tree is 231 -/
theorem initial_birds_count : initial_birds = 231 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l864_86425


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l864_86471

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_lines_condition (a : ℝ) :
  (a = 1 ↔ parallel (l₁ a) (l₂ a)) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l864_86471


namespace NUMINAMATH_CALUDE_product_pricing_l864_86495

/-- Given a cost per unit, original markup percentage, and current price percentage,
    calculate the current selling price and profit per unit. -/
theorem product_pricing (a : ℝ) (h : a > 0) :
  let original_price := a * (1 + 0.22)
  let current_price := original_price * 0.85
  let profit := current_price - a
  (current_price = 1.037 * a) ∧ (profit = 0.037 * a) := by
  sorry

end NUMINAMATH_CALUDE_product_pricing_l864_86495


namespace NUMINAMATH_CALUDE_power_sum_integer_l864_86462

theorem power_sum_integer (x : ℝ) (h : ∃ (k : ℤ), x + 1/x = k) :
  ∀ (n : ℕ), n > 0 → ∃ (m : ℤ), x^n + 1/(x^n) = m :=
by sorry

end NUMINAMATH_CALUDE_power_sum_integer_l864_86462


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l864_86434

theorem inequality_system_solution_set :
  ∀ a : ℝ, (2 * a - 3 < 0 ∧ 1 - a < 0) ↔ (1 < a ∧ a < 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l864_86434


namespace NUMINAMATH_CALUDE_number_power_problem_l864_86473

theorem number_power_problem (x : ℝ) (h : x^655 / x^650 = 100000) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_power_problem_l864_86473


namespace NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l864_86467

def A : Set ℝ := {x | x^2 ≤ 3}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_of_B_and_complement_of_A : B ∩ (Set.univ \ A) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l864_86467


namespace NUMINAMATH_CALUDE_range_of_power_function_l864_86454

/-- The range of f(x) = x^k + c on [0, ∞) is [c, ∞) when k > 0 -/
theorem range_of_power_function (k : ℝ) (c : ℝ) (h : k > 0) :
  Set.range (fun x : ℝ => x^k + c) = Set.Ici c :=
by sorry


end NUMINAMATH_CALUDE_range_of_power_function_l864_86454


namespace NUMINAMATH_CALUDE_newer_car_travels_195_miles_l864_86442

/-- The distance traveled by the older car -/
def older_car_distance : ℝ := 150

/-- The percentage increase in distance for the newer car -/
def newer_car_percentage : ℝ := 0.30

/-- The distance traveled by the newer car -/
def newer_car_distance : ℝ := older_car_distance * (1 + newer_car_percentage)

/-- Theorem stating that the newer car travels 195 miles -/
theorem newer_car_travels_195_miles :
  newer_car_distance = 195 := by sorry

end NUMINAMATH_CALUDE_newer_car_travels_195_miles_l864_86442


namespace NUMINAMATH_CALUDE_ellipse_tangent_circle_radius_l864_86474

/-- Represents an ellipse with given major and minor axes -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ

/-- Represents a circle with given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a circle is tangent to an ellipse without extending outside -/
def is_tangent_inside (e : Ellipse) (c : Circle) : Prop :=
  sorry

/-- Calculates the focus of an ellipse -/
def ellipse_focus (e : Ellipse) : ℝ × ℝ :=
  sorry

theorem ellipse_tangent_circle_radius 
  (e : Ellipse) 
  (c : Circle) 
  (h1 : e.major_axis = 12) 
  (h2 : e.minor_axis = 6) 
  (h3 : c.center = ellipse_focus e) 
  (h4 : is_tangent_inside e c) : 
  c.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_circle_radius_l864_86474


namespace NUMINAMATH_CALUDE_polynomial_division_result_l864_86472

theorem polynomial_division_result :
  let f : Polynomial ℝ := 4 * X^4 + 12 * X^3 - 9 * X^2 + X + 3
  let d : Polynomial ℝ := X^2 + 3 * X - 2
  ∀ q r : Polynomial ℝ,
    f = q * d + r →
    (r.degree < d.degree) →
    q.eval 1 + r.eval (-1) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_result_l864_86472


namespace NUMINAMATH_CALUDE_extended_inequality_l864_86484

theorem extended_inequality (n k : ℕ) (h1 : n ≥ 3) (h2 : 1 ≤ k) (h3 : k ≤ n) :
  2^n + 5^n > 2^(n-k) * 5^k + 2^k * 5^(n-k) := by
  sorry

end NUMINAMATH_CALUDE_extended_inequality_l864_86484


namespace NUMINAMATH_CALUDE_percent_relation_l864_86456

theorem percent_relation (x y z : ℝ) 
  (h1 : x = y * 1.2)  -- x is 20 percent more than y
  (h2 : y = z * 0.7)  -- y is 30 percent less than z
  : x = z * 0.84 :=   -- x is 84 percent of z
by sorry

end NUMINAMATH_CALUDE_percent_relation_l864_86456


namespace NUMINAMATH_CALUDE_chord_equation_of_parabola_l864_86479

/-- Given a parabola y² = 4x and a chord with midpoint (1, 1),
    the equation of the line containing this chord is 2x - y - 1 = 0 -/
theorem chord_equation_of_parabola (x y : ℝ) :
  (y^2 = 4*x) →  -- parabola equation
  ∃ (x1 y1 x2 y2 : ℝ),
    (y1^2 = 4*x1) ∧ (y2^2 = 4*x2) ∧  -- points on parabola
    ((x1 + x2)/2 = 1) ∧ ((y1 + y2)/2 = 1) ∧  -- midpoint condition
    (2*x - y - 1 = 0) →  -- equation of the line
  ∃ (k : ℝ), y - 1 = k*(x - 1) ∧ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_of_parabola_l864_86479


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l864_86431

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := ((b - a) / 2) + 1

theorem sum_and_count_theorem :
  sum_range 60 80 + count_even_in_range 60 80 = 1481 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l864_86431


namespace NUMINAMATH_CALUDE_max_remainder_l864_86447

theorem max_remainder (m : ℕ) (n : ℕ) : 
  0 < m → m < 2015 → 2015 % m = n → n ≤ 1007 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_l864_86447


namespace NUMINAMATH_CALUDE_average_monthly_sales_l864_86406

def monthly_sales : List ℝ := [150, 120, 80, 100, 90, 130]

theorem average_monthly_sales :
  (List.sum monthly_sales) / (List.length monthly_sales) = 111.67 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l864_86406


namespace NUMINAMATH_CALUDE_alpha_equals_five_l864_86440

-- Define the grid as a 3x3 matrix of natural numbers
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define a predicate to check if a number is a non-zero digit
def IsNonZeroDigit (n : Nat) : Prop := 0 < n ∧ n ≤ 9

-- Define a predicate to check if all elements in the grid are distinct
def AllDistinct (g : Grid) : Prop :=
  ∀ i j k l, (i, j) ≠ (k, l) → g i j ≠ g k l

-- Define a predicate to check if all elements in the grid are non-zero digits
def AllNonZeroDigits (g : Grid) : Prop :=
  ∀ i j, IsNonZeroDigit (g i j)

-- Define a predicate to check if all horizontal expressions are correct
def HorizontalExpressionsCorrect (g : Grid) : Prop :=
  (g 0 0 + g 0 1 = g 0 2) ∧
  (g 1 0 - g 1 1 = g 1 2) ∧
  (g 2 0 * g 2 1 = g 2 2)

-- Define a predicate to check if all vertical expressions are correct
def VerticalExpressionsCorrect (g : Grid) : Prop :=
  (g 0 0 + g 1 0 = g 2 0) ∧
  (g 0 1 - g 1 1 = g 2 1) ∧
  (g 0 2 * g 1 2 = g 2 2)

-- Main theorem
theorem alpha_equals_five (g : Grid) (α : Nat)
  (h1 : AllDistinct g)
  (h2 : AllNonZeroDigits g)
  (h3 : HorizontalExpressionsCorrect g)
  (h4 : VerticalExpressionsCorrect g)
  (h5 : ∃ i j, g i j = α) :
  α = 5 := by
  sorry

end NUMINAMATH_CALUDE_alpha_equals_five_l864_86440


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_7_l864_86441

theorem irrationality_of_sqrt_7 :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-2 : ℚ) / 9 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (1 : ℚ) / 2 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ -4 = (p : ℚ) / q) :=
by sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_7_l864_86441


namespace NUMINAMATH_CALUDE_line_through_point_l864_86468

/-- Given a line equation bx - (b+2)y = b - 3 passing through the point (3, -5), prove that b = -13/7 -/
theorem line_through_point (b : ℚ) : 
  (b * 3 - (b + 2) * (-5) = b - 3) → b = -13/7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l864_86468


namespace NUMINAMATH_CALUDE_problem_solution_l864_86418

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

def proposition_q (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (m - 4) + y^2 / (6 - m) = 1 ∧ 
  ((m - 4) * (6 - m) < 0)

theorem problem_solution (m : ℝ) :
  (¬ proposition_p m ↔ m < 1) ∧
  (¬(proposition_p m ∧ proposition_q m) ∧ (proposition_p m ∨ proposition_q m) ↔ 
    m < 1 ∨ (4 ≤ m ∧ m ≤ 6)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l864_86418


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l864_86402

/-- Converts a base-3 number represented as a list of digits to its base-10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base-3 representation of the number we're considering -/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l864_86402


namespace NUMINAMATH_CALUDE_complex_coordinate_l864_86491

/-- Given zi = 2-i, prove that z = -1 - 2i -/
theorem complex_coordinate (z : ℂ) : z * Complex.I = 2 - Complex.I → z = -1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_l864_86491


namespace NUMINAMATH_CALUDE_addition_of_like_terms_l864_86443

theorem addition_of_like_terms (a : ℝ) : 2 * a + a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_addition_of_like_terms_l864_86443


namespace NUMINAMATH_CALUDE_hector_siblings_product_l864_86463

/-- A family where one member has 4 sisters and 7 brothers -/
structure Family :=
  (sisters_of_helen : ℕ)
  (brothers_of_helen : ℕ)
  (helen_is_female : Bool)
  (hector_is_male : Bool)

/-- The number of sisters Hector has in the family -/
def sisters_of_hector (f : Family) : ℕ :=
  f.sisters_of_helen + (if f.helen_is_female then 1 else 0)

/-- The number of brothers Hector has in the family -/
def brothers_of_hector (f : Family) : ℕ :=
  f.brothers_of_helen - 1

theorem hector_siblings_product (f : Family) 
  (h1 : f.sisters_of_helen = 4)
  (h2 : f.brothers_of_helen = 7)
  (h3 : f.helen_is_female = true)
  (h4 : f.hector_is_male = true) :
  (sisters_of_hector f) * (brothers_of_hector f) = 30 :=
sorry

end NUMINAMATH_CALUDE_hector_siblings_product_l864_86463


namespace NUMINAMATH_CALUDE_divisible_by_101_exists_l864_86446

theorem divisible_by_101_exists (n : ℕ) (h : n ≥ 10^2018) : 
  ∃ k : ℕ, ∃ m : ℕ, m ≥ n ∧ m = n + k ∧ m % 101 = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_101_exists_l864_86446


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l864_86411

def f (x : ℝ) := 2 * abs (x + 1) + abs (x + 2)

theorem min_value_and_inequality :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    (a^2 + b^2) / c + (c^2 + a^2) / b + (b^2 + c^2) / a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l864_86411


namespace NUMINAMATH_CALUDE_four_line_angles_l864_86430

/-- Given four lines on a plane with angles α, β, and γ between some of them,
    prove that the angles between the remaining pairs of lines are as stated. -/
theorem four_line_angles (α β γ : ℝ) 
  (h_α : α = 110)
  (h_β : β = 60)
  (h_γ : γ = 80) :
  ∃ x y z : ℝ, 
    x = α - γ ∧ 
    z = β - x ∧
    y = α - β ∧
    x = 30 ∧ 
    y = 50 ∧ 
    z = 30 :=
by sorry

end NUMINAMATH_CALUDE_four_line_angles_l864_86430


namespace NUMINAMATH_CALUDE_beatrice_book_count_l864_86400

/-- The cost of each of the first 5 books -/
def initial_book_cost : ℕ := 20

/-- The number of books at the initial price -/
def initial_book_count : ℕ := 5

/-- The discount applied to each book after the initial count -/
def discount : ℕ := 2

/-- The total amount Beatrice paid -/
def total_paid : ℕ := 370

/-- Function to calculate the total cost for a given number of books -/
def total_cost (num_books : ℕ) : ℕ :=
  if num_books ≤ initial_book_count then
    num_books * initial_book_cost
  else
    initial_book_count * initial_book_cost +
    (num_books - initial_book_count) * (initial_book_cost - discount)

/-- Theorem stating that Beatrice bought 20 books -/
theorem beatrice_book_count : ∃ (n : ℕ), n = 20 ∧ total_cost n = total_paid := by
  sorry

end NUMINAMATH_CALUDE_beatrice_book_count_l864_86400


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l864_86486

-- Define a rectangular solid with prime edge lengths
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  length_prime : Nat.Prime length
  width_prime : Nat.Prime width
  height_prime : Nat.Prime height
  different_edges : length ≠ width ∧ width ≠ height ∧ length ≠ height

-- Define the volume of the rectangular solid
def volume (r : RectangularSolid) : ℕ := r.length * r.width * r.height

-- Define the surface area of the rectangular solid
def surfaceArea (r : RectangularSolid) : ℕ :=
  2 * (r.length * r.width + r.width * r.height + r.length * r.height)

-- Theorem statement
theorem rectangular_solid_surface_area :
  ∀ r : RectangularSolid, volume r = 770 → surfaceArea r = 1098 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l864_86486


namespace NUMINAMATH_CALUDE_marble_calculation_l864_86437

/-- Calculate the final number of marbles and prove its square root is 7 --/
theorem marble_calculation (initial : ℕ) (triple : ℕ → ℕ) (add : ℕ → ℕ → ℕ) 
  (lose_percent : ℕ → ℚ → ℕ) (find : ℕ → ℕ → ℕ) : 
  initial = 16 →
  (∀ x, triple x = 3 * x) →
  (∀ x y, add x y = x + y) →
  (∀ x p, lose_percent x p = x - ⌊(p * x : ℚ)⌋) →
  (∀ x y, find x y = x + y) →
  ∃ (final : ℕ), final = find (lose_percent (add (triple initial) 10) (1/4)) 5 ∧ 
  (final : ℝ).sqrt = 7 := by
sorry

end NUMINAMATH_CALUDE_marble_calculation_l864_86437


namespace NUMINAMATH_CALUDE_fraction_subtraction_l864_86465

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l864_86465


namespace NUMINAMATH_CALUDE_only_5008300_has_no_zeros_l864_86426

/-- Represents a natural number and how it's pronounced in English --/
structure NumberPronunciation where
  value : Nat
  pronunciation : String

/-- Counts the number of times "zero" appears in a string --/
def countZeros (s : String) : Nat :=
  s.split (· = ' ') |>.filter (· = "zero") |>.length

/-- The main theorem stating that only 5008300 has no zeros when pronounced --/
theorem only_5008300_has_no_zeros (numbers : List NumberPronunciation) 
    (h1 : NumberPronunciation.mk 5008300 "five million eight thousand three hundred" ∈ numbers)
    (h2 : NumberPronunciation.mk 500800 "five hundred thousand eight hundred" ∈ numbers)
    (h3 : NumberPronunciation.mk 5080000 "five million eighty thousand" ∈ numbers) :
    ∃! n : NumberPronunciation, n ∈ numbers ∧ countZeros n.pronunciation = 0 :=
  sorry

end NUMINAMATH_CALUDE_only_5008300_has_no_zeros_l864_86426


namespace NUMINAMATH_CALUDE_expand_and_simplify_l864_86477

theorem expand_and_simplify (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x^2 + 52 * x + 84 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l864_86477


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l864_86427

theorem smallest_positive_integer_congruence :
  ∃ (n : ℕ), n > 0 ∧ (77 * n) % 385 = 308 % 385 ∧
  ∀ (m : ℕ), m > 0 → (77 * m) % 385 = 308 % 385 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l864_86427


namespace NUMINAMATH_CALUDE_abs_equation_roots_l864_86414

def abs_equation (x : ℝ) : Prop :=
  |x|^2 + |x| - 6 = 0

theorem abs_equation_roots :
  ∃ (r₁ r₂ : ℝ),
    (abs_equation r₁ ∧ abs_equation r₂) ∧
    (∀ x, abs_equation x → (x = r₁ ∨ x = r₂)) ∧
    (r₁ + r₂ = 0) ∧
    (r₁ * r₂ = -4) :=
by sorry

end NUMINAMATH_CALUDE_abs_equation_roots_l864_86414


namespace NUMINAMATH_CALUDE_circle_sum_property_l864_86412

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*y - 16 = -y^2 + 24*x + 16

-- Define the center and radius of the circle
def circle_center_radius (a b r : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_property :
  ∃ a b r : ℝ, circle_center_radius a b r ∧ a + b + r = 10 + 2 * Real.sqrt 41 :=
sorry

end NUMINAMATH_CALUDE_circle_sum_property_l864_86412


namespace NUMINAMATH_CALUDE_business_value_proof_l864_86420

theorem business_value_proof (total_shares : ℚ) (man_shares : ℚ) (sold_fraction : ℚ) (sale_price : ℚ) :
  total_shares = 1 →
  man_shares = 1 / 3 →
  sold_fraction = 3 / 5 →
  sale_price = 2000 →
  (man_shares * sold_fraction * total_shares⁻¹) * (total_shares / (man_shares * sold_fraction)) * sale_price = 10000 :=
by sorry

end NUMINAMATH_CALUDE_business_value_proof_l864_86420


namespace NUMINAMATH_CALUDE_radical_product_simplification_l864_86476

theorem radical_product_simplification (p : ℝ) :
  Real.sqrt (15 * p^3) * Real.sqrt (20 * p^2) * Real.sqrt (30 * p^5) = 30 * p^5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l864_86476


namespace NUMINAMATH_CALUDE_smallest_a_b_solution_l864_86487

/-- The smallest positive integers a and b that satisfy the equation 4a + 7 = 7b + 4 and are both greater than 7 -/
theorem smallest_a_b_solution : 
  ∃ (a b : ℕ), 
    (∀ (a' b' : ℕ), (4 * a' + 7 = 7 * b' + 4 ∧ a' > 7 ∧ b' > 7) → a ≤ a' ∧ b ≤ b') ∧
    4 * a + 7 = 7 * b + 4 ∧
    a > 7 ∧
    b > 7 ∧
    a = 15 ∧
    b = 9 := by
  sorry

#check smallest_a_b_solution

end NUMINAMATH_CALUDE_smallest_a_b_solution_l864_86487


namespace NUMINAMATH_CALUDE_segment_length_product_l864_86429

theorem segment_length_product (a : ℝ) :
  (∃ a₁ a₂ : ℝ, 
    (∀ a : ℝ, (3*a - 5)^2 + (a - 3)^2 = 117 ↔ (a = a₁ ∨ a = a₂)) ∧
    a₁ * a₂ = -8.32) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_product_l864_86429


namespace NUMINAMATH_CALUDE_geometric_sequence_special_sum_l864_86444

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_special_sum
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a3 : a 3 = Real.sqrt 2 - 1)
  (h_a5 : a 5 = Real.sqrt 2 + 1) :
  a 3 ^ 2 + 2 * a 2 * a 6 + a 3 * a 7 = 8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_sum_l864_86444


namespace NUMINAMATH_CALUDE_problem_solution_l864_86451

theorem problem_solution (a b c d : ℝ) (h1 : a - b = -3) (h2 : c + d = 2) :
  (b + c) - (a - d) = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l864_86451


namespace NUMINAMATH_CALUDE_election_win_percentage_l864_86480

theorem election_win_percentage 
  (total_votes : ℕ)
  (geoff_percentage : ℚ)
  (additional_votes_needed : ℕ)
  (h1 : total_votes = 6000)
  (h2 : geoff_percentage = 1/200)  -- 0.5% as a rational number
  (h3 : additional_votes_needed = 3000) :
  (((geoff_percentage * total_votes).floor + additional_votes_needed : ℚ) / total_votes) * 100 = 505/10 :=
sorry

end NUMINAMATH_CALUDE_election_win_percentage_l864_86480


namespace NUMINAMATH_CALUDE_paint_remaining_is_three_eighths_l864_86419

/-- The fraction of paint remaining after three days of usage --/
def paint_remaining (initial_amount : ℚ) : ℚ :=
  let day1_remaining := initial_amount / 2
  let day2_remaining := day1_remaining * 3/4
  let day3_remaining := day2_remaining / 2
  day3_remaining / initial_amount

/-- Theorem stating that the fraction of paint remaining after three days is 3/8 --/
theorem paint_remaining_is_three_eighths (initial_amount : ℚ) :
  paint_remaining initial_amount = 3/8 := by
  sorry

#eval paint_remaining 2  -- To check the result

end NUMINAMATH_CALUDE_paint_remaining_is_three_eighths_l864_86419


namespace NUMINAMATH_CALUDE_factorization_equality_l864_86401

theorem factorization_equality (x : ℝ) : 
  x^2 * (x + 3) + 2 * (x + 3) - 5 * (x + 3) = (x + 3) * (x^2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l864_86401


namespace NUMINAMATH_CALUDE_eg_fh_ratio_l864_86492

/-- Given points E, F, G, and H on a line in that order, prove that EG:FH = 10:17 -/
theorem eg_fh_ratio (E F G H : ℝ) (h_order : E ≤ F ∧ F ≤ G ∧ G ≤ H) 
  (h_ef : F - E = 3) (h_fg : G - F = 7) (h_eh : H - E = 20) :
  (G - E) / (H - F) = 10 / 17 := by
  sorry

end NUMINAMATH_CALUDE_eg_fh_ratio_l864_86492


namespace NUMINAMATH_CALUDE_system_solution_l864_86433

theorem system_solution (x y z : ℝ) 
  (eq1 : x * y = 15 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 2 * y - 4 * z)
  (eq3 : x * z = 56 - 5 * x - 6 * z)
  (x_pos : x > 0) :
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l864_86433


namespace NUMINAMATH_CALUDE_sampling_interval_for_tv_program_l864_86432

/-- The sampling interval for systematic sampling -/
def samplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for 10,000 viewers and 10 lucky draws is 1000 -/
theorem sampling_interval_for_tv_program :
  samplingInterval 10000 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_sampling_interval_for_tv_program_l864_86432


namespace NUMINAMATH_CALUDE_arcsin_one_equals_pi_half_l864_86459

theorem arcsin_one_equals_pi_half : Real.arcsin 1 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_equals_pi_half_l864_86459


namespace NUMINAMATH_CALUDE_emily_bought_two_skirts_l864_86455

def cost_of_art_supplies : ℕ := 20
def total_spent : ℕ := 50
def cost_per_skirt : ℕ := 15

def number_of_skirts : ℕ := (total_spent - cost_of_art_supplies) / cost_per_skirt

theorem emily_bought_two_skirts : number_of_skirts = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_bought_two_skirts_l864_86455


namespace NUMINAMATH_CALUDE_similar_triangles_solution_l864_86435

/-- Two similar right triangles, one with legs 12 and 9, the other with legs x and 6 -/
def similar_triangles (x : ℝ) : Prop :=
  (12 : ℝ) / x = 9 / 6

theorem similar_triangles_solution :
  ∃ x : ℝ, similar_triangles x ∧ x = 8 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_solution_l864_86435


namespace NUMINAMATH_CALUDE_bus_speed_problem_l864_86428

/-- Given a bus that stops for 15 minutes per hour and has an average speed of 45 km/hr
    including stoppages, its average speed excluding stoppages is 60 km/hr. -/
theorem bus_speed_problem (stop_time : ℝ) (avg_speed_with_stops : ℝ) :
  stop_time = 15 →
  avg_speed_with_stops = 45 →
  ∃ (avg_speed_without_stops : ℝ),
    avg_speed_without_stops = 60 ∧
    avg_speed_with_stops * 1 = avg_speed_without_stops * ((60 - stop_time) / 60) := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l864_86428


namespace NUMINAMATH_CALUDE_sample_mean_inequality_l864_86458

theorem sample_mean_inequality (n m : ℕ) (x_bar y_bar z_bar α : ℝ) :
  x_bar ≠ y_bar →
  0 < α →
  α < 1 / 2 →
  z_bar = α * x_bar + (1 - α) * y_bar →
  z_bar = (n * x_bar + m * y_bar) / (n + m) →
  n < m :=
by sorry

end NUMINAMATH_CALUDE_sample_mean_inequality_l864_86458


namespace NUMINAMATH_CALUDE_race_problem_l864_86415

/-- The race problem -/
theorem race_problem (total_distance : ℝ) (time_A : ℝ) (time_B : ℝ) 
  (h1 : total_distance = 70)
  (h2 : time_A = 20)
  (h3 : time_B = 25) :
  total_distance - (total_distance / time_B * time_A) = 14 := by
  sorry

end NUMINAMATH_CALUDE_race_problem_l864_86415


namespace NUMINAMATH_CALUDE_c_7_equals_448_l864_86408

/-- Sequence definition -/
def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

/-- Theorem stating that c_7 equals 448 -/
theorem c_7_equals_448 : c 7 = 448 := by
  sorry

end NUMINAMATH_CALUDE_c_7_equals_448_l864_86408


namespace NUMINAMATH_CALUDE_problem_statement_l864_86466

/-- Given two natural numbers a and b, returns the floor of a/b -/
def floorDiv (a b : ℕ) : ℕ := a / b

/-- Returns true if the number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- Counts the number of prime numbers in the range (a, b) -/
def countPrimesBetween (a b : ℕ) : ℕ := sorry

theorem problem_statement :
  ∃ n : ℕ,
    n = floorDiv 51 13 ∧
    countPrimesBetween n (floorDiv 89 9) = 2 ∧
    n = 3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l864_86466


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l864_86422

theorem polynomial_remainder_theorem : ∃ q : Polynomial ℝ, 
  3 * X^3 + 2 * X^2 - 20 * X + 47 = (X - 3) * q + 86 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l864_86422


namespace NUMINAMATH_CALUDE_compute_expression_l864_86498

theorem compute_expression : (-3) * 2 + 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l864_86498


namespace NUMINAMATH_CALUDE_shaded_area_in_circle_l864_86421

theorem shaded_area_in_circle (r : ℝ) (h1 : r > 0) : 
  let circle_area := π * r^2
  let sector_area := 2 * π
  let sector_fraction := 1 / 8
  let triangle_area := r^2 / 2
  sector_area = sector_fraction * circle_area → 
  sector_area - triangle_area = 2 * π - 4 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_in_circle_l864_86421


namespace NUMINAMATH_CALUDE_problem_cube_white_surface_fraction_l864_86490

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  sideLength : ℕ
  totalCubes : ℕ
  whiteCubes : ℕ
  blackCubes : ℕ
  redCubes : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def whiteSurfaceFraction (c : CompositeCube) : ℚ :=
  sorry

/-- The specific composite cube from the problem -/
def problemCube : CompositeCube :=
  { sideLength := 4
    totalCubes := 64
    whiteCubes := 36
    blackCubes := 8
    redCubes := 20 }

/-- Theorem stating that the white surface fraction of the problem cube is 1/2 -/
theorem problem_cube_white_surface_fraction :
  whiteSurfaceFraction problemCube = 1/2 :=
sorry

end NUMINAMATH_CALUDE_problem_cube_white_surface_fraction_l864_86490


namespace NUMINAMATH_CALUDE_floor_abs_sum_l864_86417

theorem floor_abs_sum : ⌊|(-3.1 : ℝ)|⌋ + |⌊(-3.1 : ℝ)⌋| = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l864_86417


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_groups_l864_86460

theorem common_number_in_overlapping_groups (list : List ℝ) : 
  list.length = 9 →
  (list.take 5).sum / 5 = 7 →
  (list.drop 4).sum / 5 = 10 →
  list.sum / 9 = 74 / 9 →
  ∃ x ∈ list, x ∈ list.take 5 ∧ x ∈ list.drop 4 ∧ x = 11 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_groups_l864_86460


namespace NUMINAMATH_CALUDE_clothing_store_profit_l864_86493

/-- Profit function for a clothing store --/
def profit_function (x : ℝ) : ℝ := 20 * x + 4000

/-- Maximum profit under cost constraint --/
def max_profit : ℝ := 5500

/-- Discount value for maximum profit under new conditions --/
def discount_value : ℝ := 9

/-- Theorem stating the main results --/
theorem clothing_store_profit :
  (∀ x : ℝ, x ≥ 60 → x ≤ 100 → profit_function x = 20 * x + 4000) ∧
  (∀ x : ℝ, x ≥ 60 → x ≤ 75 → 160 * x + 120 * (100 - x) ≤ 15000 → profit_function x ≤ max_profit) ∧
  (∃ x : ℝ, x ≥ 60 ∧ x ≤ 75 ∧ 160 * x + 120 * (100 - x) ≤ 15000 ∧ profit_function x = max_profit) ∧
  (∀ a : ℝ, 0 < a → a < 20 → 
    (∃ x : ℝ, x ≥ 60 ∧ x ≤ 75 ∧ 
      ((20 - a) * x + 100 * a + 3600 = 4950) → a = discount_value)) :=
by sorry

end NUMINAMATH_CALUDE_clothing_store_profit_l864_86493


namespace NUMINAMATH_CALUDE_friends_for_games_only_l864_86424

-- Define the variables
def movie : ℕ := 10
def picnic : ℕ := 20
def movie_and_picnic : ℕ := 4
def movie_and_games : ℕ := 2
def picnic_and_games : ℕ := 0
def all_three : ℕ := 2
def total_students : ℕ := 31

-- Theorem to prove
theorem friends_for_games_only : 
  ∃ (movie_only picnic_only games_only : ℕ),
    movie_only + picnic_only + games_only + movie_and_picnic + movie_and_games + picnic_and_games + all_three = total_students ∧
    movie_only + movie_and_picnic + movie_and_games + all_three = movie ∧
    picnic_only + movie_and_picnic + picnic_and_games + all_three = picnic ∧
    games_only = 1 := by
  sorry

end NUMINAMATH_CALUDE_friends_for_games_only_l864_86424


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l864_86485

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  ((a - 1)^3 + (b - 1)^3 ≥ 3 * (2 - a - b)) → 
  (a^2 + b^2 ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l864_86485


namespace NUMINAMATH_CALUDE_increasing_order_abc_l864_86416

theorem increasing_order_abc (a b c : ℝ) : 
  a = 2^(4/3) → b = 3^(2/3) → c = 25^(1/3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_increasing_order_abc_l864_86416


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l864_86457

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_zeros_difference (p : Parabola) :
  p.y_coord 3 = -9 →   -- vertex condition
  p.y_coord 6 = 27 →   -- point condition
  ∃ (x1 x2 : ℝ), 
    p.y_coord x1 = 0 ∧ 
    p.y_coord x2 = 0 ∧ 
    |x1 - x2| = 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l864_86457


namespace NUMINAMATH_CALUDE_pencil_distribution_ways_l864_86494

def distribute_pencils (n : ℕ) (k : ℕ) (min_first : ℕ) (min_others : ℕ) : ℕ :=
  Nat.choose (n - (min_first + (k - 1) * min_others) + k - 1) (k - 1)

theorem pencil_distribution_ways : 
  distribute_pencils 8 4 2 1 = 20 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_ways_l864_86494


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l864_86404

theorem sqrt_eight_and_nine_sixteenths :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l864_86404


namespace NUMINAMATH_CALUDE_fourth_person_win_prob_is_one_thirtieth_l864_86423

/-- Represents the probability of winning for the fourth person in a
    coin-flipping game with four players where the first to get heads wins. -/
def fourth_person_win_probability : ℚ := 1 / 30

/-- The probability of getting tails on a fair coin flip. -/
def prob_tails : ℚ := 1 / 2

/-- The number of players in the game. -/
def num_players : ℕ := 4

/-- Theorem stating that the probability of the fourth person winning
    in a coin-flipping game with four players is 1/30. -/
theorem fourth_person_win_prob_is_one_thirtieth :
  fourth_person_win_probability = 
    (prob_tails ^ num_players) / (1 - prob_tails ^ num_players) :=
sorry

end NUMINAMATH_CALUDE_fourth_person_win_prob_is_one_thirtieth_l864_86423


namespace NUMINAMATH_CALUDE_inverse_mod_78_l864_86439

theorem inverse_mod_78 (h : (7⁻¹ : ZMod 78) = 55) : (49⁻¹ : ZMod 78) = 61 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_78_l864_86439


namespace NUMINAMATH_CALUDE_decimal_189_to_base_4_lsd_l864_86478

def decimal_to_base_4_lsd (n : ℕ) : ℕ :=
  n % 4

theorem decimal_189_to_base_4_lsd :
  decimal_to_base_4_lsd 189 = 1 := by
  sorry

end NUMINAMATH_CALUDE_decimal_189_to_base_4_lsd_l864_86478


namespace NUMINAMATH_CALUDE_not_greater_than_three_equiv_l864_86461

theorem not_greater_than_three_equiv (a : ℝ) : (¬(a > 3)) ↔ (a ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_not_greater_than_three_equiv_l864_86461


namespace NUMINAMATH_CALUDE_janet_dog_collars_l864_86475

/-- The number of dog collars Janet needs to make -/
def dog_collars : ℕ := 9

/-- The length of nylon needed for one dog collar (in inches) -/
def dog_collar_length : ℕ := 18

/-- The length of nylon needed for one cat collar (in inches) -/
def cat_collar_length : ℕ := 10

/-- The number of cat collars Janet needs to make -/
def cat_collars : ℕ := 3

/-- The total length of nylon Janet has (in inches) -/
def total_nylon : ℕ := 192

theorem janet_dog_collars :
  dog_collars * dog_collar_length + cat_collars * cat_collar_length = total_nylon :=
by sorry

end NUMINAMATH_CALUDE_janet_dog_collars_l864_86475


namespace NUMINAMATH_CALUDE_percentage_needed_to_pass_l864_86436

def total_marks : ℕ := 2075
def pradeep_score : ℕ := 390
def failed_by : ℕ := 25

def passing_mark : ℕ := pradeep_score + failed_by

def percentage_to_pass : ℚ := (passing_mark : ℚ) / (total_marks : ℚ) * 100

theorem percentage_needed_to_pass :
  ∃ (ε : ℚ), abs (percentage_to_pass - 20) < ε ∧ ε > 0 :=
sorry

end NUMINAMATH_CALUDE_percentage_needed_to_pass_l864_86436


namespace NUMINAMATH_CALUDE_team_selection_count_l864_86438

-- Define the number of boys and girls
def num_boys : ℕ := 5
def num_girls : ℕ := 10

-- Define the team size and minimum number of girls required
def team_size : ℕ := 6
def min_girls : ℕ := 3

-- Define the function to calculate the number of ways to select the team
def select_team : ℕ := 
  (Nat.choose num_girls 3 * Nat.choose num_boys 3) +
  (Nat.choose num_girls 4 * Nat.choose num_boys 2) +
  (Nat.choose num_girls 5 * Nat.choose num_boys 1) +
  (Nat.choose num_girls 6)

-- Theorem statement
theorem team_selection_count : select_team = 4770 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l864_86438


namespace NUMINAMATH_CALUDE_lives_lost_l864_86410

theorem lives_lost (initial_lives : ℕ) (lives_gained : ℕ) (final_lives : ℕ) 
  (h1 : initial_lives = 14)
  (h2 : lives_gained = 36)
  (h3 : final_lives = 46) :
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 4 := by
  sorry

end NUMINAMATH_CALUDE_lives_lost_l864_86410


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l864_86496

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_specific_arithmetic_sequence :
  let a₁ : ℚ := 1
  let a₂ : ℚ := 3/2
  let d : ℚ := a₂ - a₁
  arithmeticSequenceTerm a₁ d 12 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l864_86496


namespace NUMINAMATH_CALUDE_cases_needed_l864_86453

theorem cases_needed (total_boxes : Nat) (boxes_per_case : Nat) : 
  total_boxes = 20 → boxes_per_case = 4 → total_boxes / boxes_per_case = 5 := by
  sorry

end NUMINAMATH_CALUDE_cases_needed_l864_86453


namespace NUMINAMATH_CALUDE_rod_pieces_count_l864_86445

/-- The length of the rod in meters -/
def rod_length : ℝ := 38.25

/-- The length of each piece in centimeters -/
def piece_length : ℝ := 85

/-- The number of pieces that can be cut from the rod -/
def num_pieces : ℕ := 45

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

theorem rod_pieces_count : 
  ⌊(rod_length * meters_to_cm) / piece_length⌋ = num_pieces := by
  sorry

end NUMINAMATH_CALUDE_rod_pieces_count_l864_86445


namespace NUMINAMATH_CALUDE_wind_power_scientific_notation_l864_86469

/-- Proves that 56 million kilowatts is equal to 5.6 × 10^7 kilowatts in scientific notation -/
theorem wind_power_scientific_notation : 
  (56000000 : ℝ) = 5.6 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_wind_power_scientific_notation_l864_86469


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l864_86403

theorem smaller_integer_problem (x y : ℤ) 
  (sum_eq : x + y = 30)
  (relation : 2 * y = 5 * x - 10) :
  x = 10 ∧ x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l864_86403


namespace NUMINAMATH_CALUDE_decimal_equivalences_l864_86499

theorem decimal_equivalences (d : ℚ) (h : d = 0.25) : 
  d = 3 / 12 ∧ d = 8 / 32 ∧ d = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalences_l864_86499


namespace NUMINAMATH_CALUDE_f_increasing_l864_86409

def f (x : ℝ) := 3 * x + 2

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l864_86409


namespace NUMINAMATH_CALUDE_intersection_M_N_l864_86450

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x : ℕ | x - 1 ≥ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l864_86450


namespace NUMINAMATH_CALUDE_triangle_area_on_rectangle_l864_86449

/-- Given a rectangle of 6 units by 8 units and a triangle DEF with vertices
    D(0,2), E(6,0), and F(3,8) located on the boundary of the rectangle,
    prove that the area of triangle DEF is 21 square units. -/
theorem triangle_area_on_rectangle (D E F : ℝ × ℝ) : 
  D = (0, 2) →
  E = (6, 0) →
  F = (3, 8) →
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 8
  let triangle_area := abs ((D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2)) / 2)
  triangle_area = 21 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_on_rectangle_l864_86449


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l864_86464

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_mean1 : (a 1 + a 2) / 2 = 1)
  (h_mean2 : (a 2 + a 3) / 2 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l864_86464


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_twenty_five_percent_less_proof_l864_86482

theorem twenty_five_percent_less : ℝ → Prop :=
  fun x => (x + x / 4 = 80 * 3 / 4) → x = 48

-- The proof goes here
theorem twenty_five_percent_less_proof : twenty_five_percent_less 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_twenty_five_percent_less_proof_l864_86482


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l864_86470

theorem polar_to_rectangular (r θ : Real) (h : r = 4 ∧ θ = π / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l864_86470


namespace NUMINAMATH_CALUDE_inequality_proof_l864_86452

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  1/a + 1/b + 4/c + 16/d ≥ 64 / (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l864_86452


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l864_86448

theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) : 
  alice_time = 40 → bob_fraction = 3 / 4 → bob_fraction * alice_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l864_86448


namespace NUMINAMATH_CALUDE_negation_of_exists_leq_negation_of_proposition_l864_86497

theorem negation_of_exists_leq (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬(p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_leq_negation_of_proposition_l864_86497


namespace NUMINAMATH_CALUDE_min_value_theorem_l864_86405

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1 = 2*m + n) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 1 = 2*m₀ + n₀ ∧ 1/m₀ + 2/n₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l864_86405
