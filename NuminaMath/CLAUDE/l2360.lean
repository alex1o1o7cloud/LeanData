import Mathlib

namespace knicks_to_knocks_equivalence_l2360_236049

/-- Represents the number of units of a given type -/
structure UnitCount (α : Type) where
  count : ℚ

/-- Conversion rate between two types of units -/
def ConversionRate (α β : Type) : Type :=
  UnitCount α → UnitCount β

/-- Given conversion rates, prove that 40 knicks are equivalent to 36 knocks -/
theorem knicks_to_knocks_equivalence 
  (knick knack knock : Type)
  (knicks_to_knacks : ConversionRate knick knack)
  (knacks_to_knocks : ConversionRate knack knock)
  (h1 : knicks_to_knacks ⟨5⟩ = ⟨3⟩)
  (h2 : knacks_to_knocks ⟨4⟩ = ⟨6⟩)
  : ∃ (f : ConversionRate knick knock), f ⟨40⟩ = ⟨36⟩ := by
  sorry

end knicks_to_knocks_equivalence_l2360_236049


namespace sum_of_triangle_perimeters_l2360_236015

/-- Given an equilateral triangle with side length 45 cm, if we repeatedly form new equilateral
    triangles by joining the midpoints of the previous triangle's sides, the sum of the perimeters
    of all these triangles is 270 cm. -/
theorem sum_of_triangle_perimeters (s : ℝ) (h : s = 45) :
  let perimeter_sum := (3 * s) / (1 - (1/2 : ℝ))
  perimeter_sum = 270 := by sorry

end sum_of_triangle_perimeters_l2360_236015


namespace smallest_k_for_64_power_gt_4_16_l2360_236093

theorem smallest_k_for_64_power_gt_4_16 : ∃ k : ℕ, k = 6 ∧ 64^k > 4^16 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^16 := by
  sorry

end smallest_k_for_64_power_gt_4_16_l2360_236093


namespace problem_solution_l2360_236013

theorem problem_solution (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    1/a + 1/b + 1/c ≤ 1/x + 1/y + 1/z) ∧
  (1/(1-a) + 1/(1-b) + 1/(1-c) ≥ 2/(1+a) + 2/(1+b) + 2/(1+c)) :=
by sorry

end problem_solution_l2360_236013


namespace duck_flying_ratio_l2360_236066

/-- Represents the flying time of a duck during different seasons -/
structure DuckFlyingTime where
  total : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the ratio of north flying time to south flying time -/
def northToSouthRatio (d : DuckFlyingTime) : ℚ :=
  let north := d.total - d.south - d.east
  (north : ℚ) / d.south

/-- Theorem stating that the ratio of north to south flying time is 2:1 -/
theorem duck_flying_ratio :
  ∀ d : DuckFlyingTime,
  d.total = 180 ∧ d.south = 40 ∧ d.east = 60 →
  northToSouthRatio d = 2 := by
  sorry


end duck_flying_ratio_l2360_236066


namespace island_puzzle_l2360_236091

/-- Represents a person who is either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- The statement made by a person about the number of liars -/
def Statement := Fin 5 → ℕ

/-- Checks if a person's statement is truthful given the actual number of liars -/
def isStatementTruthful (p : Person) (s : ℕ) (actualLiars : ℕ) : Prop :=
  match p with
  | Person.Knight => s = actualLiars
  | Person.Liar => s ≠ actualLiars

/-- The main theorem to prove -/
theorem island_puzzle :
  ∀ (people : Fin 5 → Person) (statements : Statement),
  (∀ i j : Fin 5, i ≠ j → statements i ≠ statements j) →
  (∀ i : Fin 5, statements i = i.val + 1) →
  (∃! i : Fin 5, people i = Person.Knight) →
  (∀ i : Fin 5, isStatementTruthful (people i) (statements i) 4) :=
sorry

end island_puzzle_l2360_236091


namespace gcd_twelve_digit_form_l2360_236030

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def twelve_digit_form (m : ℕ) : ℕ := 1000001 * m

theorem gcd_twelve_digit_form :
  ∃ (g : ℕ), ∀ (m : ℕ), is_six_digit m → 
    (∃ (k : ℕ), twelve_digit_form m = g * k) ∧
    (∀ (d : ℕ), (∀ (n : ℕ), is_six_digit n → ∃ (k : ℕ), twelve_digit_form n = d * k) → d ≤ g) ∧
    g = 1000001 :=
by sorry

end gcd_twelve_digit_form_l2360_236030


namespace add_one_five_times_l2360_236033

theorem add_one_five_times (m : ℕ) : 
  let n := m + 5
  n = m + 5 ∧ n - (m + 1) = 4 := by
sorry

end add_one_five_times_l2360_236033


namespace max_product_of_arithmetic_sequence_l2360_236060

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem max_product_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + 2 * a 6 = 6) :
  (∀ x : ℝ, a 4 * a 6 ≤ x → x ≤ 4) ∧ a 4 * a 6 ≤ 4 :=
sorry

end max_product_of_arithmetic_sequence_l2360_236060


namespace paving_stone_width_l2360_236072

/-- Represents the dimensions of a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Given a courtyard and paving stone specifications, proves that the width of each paving stone is 2 meters -/
theorem paving_stone_width
  (courtyard : Courtyard)
  (stone_count : ℕ)
  (stone_length : ℝ)
  (h1 : courtyard.length = 30)
  (h2 : courtyard.width = 33/2)
  (h3 : stone_count = 99)
  (h4 : stone_length = 5/2) :
  ∃ (stone : PavingStone), stone.length = stone_length ∧ stone.width = 2 :=
sorry

end paving_stone_width_l2360_236072


namespace office_to_bedroom_ratio_l2360_236070

/-- Represents the energy consumption of lights in watts per hour -/
structure LightEnergy where
  bedroom : ℝ
  office : ℝ
  livingRoom : ℝ

/-- Calculates the total energy used over a given number of hours -/
def totalEnergyUsed (l : LightEnergy) (hours : ℝ) : ℝ :=
  (l.bedroom + l.office + l.livingRoom) * hours

/-- Theorem stating the ratio of office light energy to bedroom light energy -/
theorem office_to_bedroom_ratio (l : LightEnergy) :
  l.bedroom = 6 →
  l.livingRoom = 4 * l.bedroom →
  totalEnergyUsed l 2 = 96 →
  l.office / l.bedroom = 3 := by
sorry

end office_to_bedroom_ratio_l2360_236070


namespace range_of_a_l2360_236082

theorem range_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_one : a^2 + b^2 + c^2 = 1) :
  -Real.sqrt 6 / 3 ≤ a ∧ a ≤ Real.sqrt 6 / 3 := by sorry

end range_of_a_l2360_236082


namespace shortest_major_axis_ellipse_l2360_236038

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 12 * x^2 - 4 * y^2 = 3

-- Define a general ellipse
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition for an ellipse to share foci with the hyperbola
def shared_foci (a b : ℝ) : Prop := a^2 - b^2 = 1

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop := ∃ x y : ℝ, line_l x y ∧ ellipse a b x y

-- Theorem statement
theorem shortest_major_axis_ellipse :
  ∀ a b : ℝ, a > 0 → b > 0 →
  shared_foci a b →
  is_tangent a b →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → shared_foci a' b' → is_tangent a' b' → a ≤ a') →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end shortest_major_axis_ellipse_l2360_236038


namespace stating_botanical_garden_visitors_l2360_236052

/-- Represents the growth rate of visitors in a botanical garden -/
def growth_rate_equation (x : ℝ) : Prop :=
  (1 + x)^2 = 3

/-- 
Theorem stating that the growth rate equation holds given the conditions:
- The number of visitors in March is three times that of January
- x is the average growth rate of visitors in February and March
-/
theorem botanical_garden_visitors (x : ℝ) 
  (h_march : ∃ (a : ℝ), a > 0 ∧ a * (1 + x)^2 = 3 * a) : 
  growth_rate_equation x := by
  sorry

end stating_botanical_garden_visitors_l2360_236052


namespace expression_value_l2360_236006

theorem expression_value (x y : ℝ) (h : x - 2*y + 3 = 0) : 1 - 2*x + 4*y = 7 := by
  sorry

end expression_value_l2360_236006


namespace sum_45_25_in_base5_l2360_236056

/-- Converts a decimal number to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base 5 number to decimal -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two base 5 numbers -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_45_25_in_base5 :
  let a := 45
  let b := 25
  let a_base5 := toBase5 a
  let b_base5 := toBase5 b
  let sum_base5 := addBase5 a_base5 b_base5
  sum_base5 = [2, 3, 0] := by
  sorry

end sum_45_25_in_base5_l2360_236056


namespace length_of_24_l2360_236034

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- 24 can be expressed as a product of 4 prime factors. -/
theorem length_of_24 : length 24 = 4 := by sorry

end length_of_24_l2360_236034


namespace count_negative_numbers_l2360_236012

theorem count_negative_numbers : 
  let expressions := [-2^2, (-2)^2, -(-2), -|-2|]
  (expressions.filter (λ x => x < 0)).length = 2 := by
  sorry

end count_negative_numbers_l2360_236012


namespace calculator_cost_l2360_236088

/-- Given information about calculator purchases, prove the cost of each graphing calculator. -/
theorem calculator_cost (total_cost : ℕ) (total_calculators : ℕ) (scientific_cost : ℕ)
  (scientific_count : ℕ) (graphing_count : ℕ)
  (h1 : total_cost = 1625)
  (h2 : total_calculators = 45)
  (h3 : scientific_cost = 10)
  (h4 : scientific_count = 20)
  (h5 : graphing_count = 25)
  (h6 : total_calculators = scientific_count + graphing_count) :
  (total_cost - scientific_cost * scientific_count) / graphing_count = 57 := by
  sorry

#eval (1625 - 10 * 20) / 25  -- Should output 57

end calculator_cost_l2360_236088


namespace fifth_friend_payment_l2360_236014

def boat_purchase (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e)

theorem fifth_friend_payment :
  ∃ a b c d : ℝ, boat_purchase a b c d 13 :=
sorry

end fifth_friend_payment_l2360_236014


namespace smaller_solution_cube_root_equation_l2360_236009

theorem smaller_solution_cube_root_equation (x : ℝ) :
  (Real.rpow x (1/3 : ℝ) + Real.rpow (16 - x) (1/3 : ℝ) = 2) →
  (x = (1 - Real.sqrt 21 / 3)^3 ∨ x = (1 + Real.sqrt 21 / 3)^3) ∧
  x ≤ (1 + Real.sqrt 21 / 3)^3 :=
by sorry

end smaller_solution_cube_root_equation_l2360_236009


namespace japanese_selectors_l2360_236080

theorem japanese_selectors (j c f : ℕ) : 
  j = 3 * c →
  c = f + 15 →
  j + c + f = 165 →
  j = 108 := by
sorry

end japanese_selectors_l2360_236080


namespace complex_magnitude_l2360_236090

theorem complex_magnitude (z : ℂ) (h : 3 * z + Complex.I = 1 - 4 * Complex.I * z) :
  Complex.abs z = Real.sqrt 2 / 5 := by
  sorry

end complex_magnitude_l2360_236090


namespace focal_chord_circle_tangent_to_directrix_l2360_236087

-- Define a parabola
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ := (0, p)
  vertex : ℝ × ℝ := (0, 0)
  directrix : ℝ → ℝ := fun x ↦ -p

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the focal chord circle
def focal_chord_circle (parab : Parabola) : Circle :=
  { center := parab.focus
  , radius := parab.p }

-- Theorem statement
theorem focal_chord_circle_tangent_to_directrix (parab : Parabola) :
  let circle := focal_chord_circle parab
  let lowest_point := (circle.center.1, circle.center.2 - circle.radius)
  lowest_point.2 = 0 ∧ parab.directrix lowest_point.1 = -parab.p :=
sorry

end focal_chord_circle_tangent_to_directrix_l2360_236087


namespace negation_of_universal_statement_l2360_236019

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end negation_of_universal_statement_l2360_236019


namespace unique_solution_quadratic_l2360_236094

/-- For a quadratic equation px^2 - 20x + 4 = 0, where p is nonzero,
    the equation has only one solution if and only if p = 25. -/
theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x, p * x^2 - 20 * x + 4 = 0) ↔ p = 25 := by
  sorry

end unique_solution_quadratic_l2360_236094


namespace large_loans_required_l2360_236010

/-- Represents the number of loans of each type required to buy an apartment -/
structure LoanCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Two equivalent ways to buy the apartment -/
def way1 : LoanCombination := { small := 9, medium := 6, large := 1 }
def way2 : LoanCombination := { small := 3, medium := 2, large := 3 }

/-- The theorem states that 4 large loans are required to buy the apartment -/
theorem large_loans_required : ∃ (n : ℕ), n = 4 ∧ 
  way1.small * n + way1.medium * n + way1.large * n = 
  way2.small * n + way2.medium * n + way2.large * n :=
sorry

end large_loans_required_l2360_236010


namespace arithmetic_expression_equality_l2360_236044

theorem arithmetic_expression_equality : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end arithmetic_expression_equality_l2360_236044


namespace quadratic_equation_roots_l2360_236067

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k^2 * x₁^2 + (2*k - 1) * x₁ + 1 = 0 ∧ k^2 * x₂^2 + (2*k - 1) * x₂ + 1 = 0) →
  (k < 1/4 ∧ k ≠ 0) ∧
  ¬∃ (k : ℝ), ∃ (x : ℝ), k^2 * x^2 + (2*k - 1) * x + 1 = 0 ∧ k^2 * (-x)^2 + (2*k - 1) * (-x) + 1 = 0 :=
by sorry

end quadratic_equation_roots_l2360_236067


namespace complex_square_simplification_l2360_236078

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end complex_square_simplification_l2360_236078


namespace framed_photo_area_l2360_236048

/-- The area of a framed rectangular photo -/
theorem framed_photo_area 
  (paper_width : ℝ) 
  (paper_length : ℝ) 
  (frame_width : ℝ) 
  (h1 : paper_width = 8) 
  (h2 : paper_length = 12) 
  (h3 : frame_width = 2) : 
  (paper_width + 2 * frame_width) * (paper_length + 2 * frame_width) = 192 :=
by sorry

end framed_photo_area_l2360_236048


namespace max_value_of_g_l2360_236027

/-- Given f(x) = sin x + a cos x with a symmetry axis at x = 5π/3,
    prove that the maximum value of g(x) = a sin x + cos x is 2√3/3 -/
theorem max_value_of_g (a : ℝ) (f g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x + a * Real.cos x)
    (h₂ : ∀ x, f x = f (10 * Real.pi / 3 - x))
    (h₃ : ∀ x, g x = a * Real.sin x + Real.cos x) :
    (∀ x, g x ≤ 2 * Real.sqrt 3 / 3) ∧ ∃ x, g x = 2 * Real.sqrt 3 / 3 :=
by sorry

end max_value_of_g_l2360_236027


namespace external_angle_bisectors_collinear_l2360_236022

-- Define the basic structures
structure Point := (x : ℝ) (y : ℝ)
structure Line := (a : ℝ) (b : ℝ) (c : ℝ) -- ax + by + c = 0

-- Define the quadrilateral
structure Quadrilateral :=
  (A B C D : Point)
  (is_convex : Bool)

-- Define the intersection points of side extensions
def extension_intersections (q : Quadrilateral) : Point × Point := sorry

-- Define the external angle bisector
def external_angle_bisector (p1 p2 p3 : Point) : Line := sorry

-- Define collinearity
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Main theorem
theorem external_angle_bisectors_collinear (q : Quadrilateral) :
  let (P, Q) := extension_intersections q
  let AC_bisector := external_angle_bisector q.A q.C P
  let BD_bisector := external_angle_bisector q.B q.D Q
  let PQ_bisector := external_angle_bisector P Q q.A
  let I1 := sorry -- Intersection of AC_bisector and BD_bisector
  let I2 := sorry -- Intersection of BD_bisector and PQ_bisector
  let I3 := sorry -- Intersection of PQ_bisector and AC_bisector
  collinear I1 I2 I3 := by sorry

end external_angle_bisectors_collinear_l2360_236022


namespace tangent_parallel_points_main_theorem_l2360_236061

/-- The function f(x) = x³ + x - 2 --/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_points :
  {x : ℝ | f' x = 4} = {1, -1} :=
sorry

theorem main_theorem :
  {p : ℝ × ℝ | p.1 ∈ {x : ℝ | f' x = 4} ∧ p.2 = f p.1} = {(1, 0), (-1, -4)} :=
sorry

end tangent_parallel_points_main_theorem_l2360_236061


namespace line_equal_intercepts_l2360_236097

/-- A line with equation ax + y - 2 - a = 0 has equal intercepts on the x-axis and y-axis if and only if a = -2 or a = 1 -/
theorem line_equal_intercepts (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y - 2 - a = 0 ∧ x = y) ↔ (a = -2 ∨ a = 1) :=
sorry

end line_equal_intercepts_l2360_236097


namespace nth_equation_proof_l2360_236042

theorem nth_equation_proof (n : ℕ) (hn : n > 0) : 
  (1 : ℚ) / n * ((n^2 + 2*n) / (n + 1)) - 1 / (n + 1) = 1 := by
  sorry

end nth_equation_proof_l2360_236042


namespace product_of_differences_divisible_by_12_l2360_236000

theorem product_of_differences_divisible_by_12 (a b c d : ℤ) :
  ∃ k : ℤ, (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) = 12 * k := by
  sorry

end product_of_differences_divisible_by_12_l2360_236000


namespace train_delay_l2360_236021

/-- Calculates the time difference in minutes for a train traveling a given distance at two different speeds -/
theorem train_delay (distance : ℝ) (speed1 speed2 : ℝ) :
  distance > 0 ∧ speed1 > 0 ∧ speed2 > 0 ∧ speed1 > speed2 →
  (distance / speed2 - distance / speed1) * 60 = 15 ∧
  distance = 70 ∧ speed1 = 40 ∧ speed2 = 35 :=
by sorry

end train_delay_l2360_236021


namespace sufficient_not_necessary_condition_l2360_236041

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 1 → x - 1/x > 0) ∧
  (∃ x : ℝ, x - 1/x > 0 ∧ x ≤ 1) := by
  sorry

end sufficient_not_necessary_condition_l2360_236041


namespace caleb_hamburger_cost_l2360_236026

/-- Represents the total cost of Caleb's hamburger purchase --/
def total_cost (single_price : ℚ) (double_price : ℚ) (total_burgers : ℕ) (double_burgers : ℕ) : ℚ :=
  single_price * (total_burgers - double_burgers) + double_price * double_burgers

/-- Theorem stating that Caleb's total spending on hamburgers is $74.50 --/
theorem caleb_hamburger_cost : 
  total_cost 1 (3/2) 50 49 = 149/2 := by
  sorry

end caleb_hamburger_cost_l2360_236026


namespace maynards_dog_holes_l2360_236035

theorem maynards_dog_holes : 
  ∀ (total : ℕ) (filled : ℕ) (unfilled : ℕ),
    filled = (75 * total) / 100 →
    unfilled = 2 →
    total = filled + unfilled →
    total = 8 := by
  sorry

end maynards_dog_holes_l2360_236035


namespace article_gain_percentage_l2360_236028

/-- Calculates the cost price given the selling price and loss percentage -/
def costPrice (sellingPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercentage / 100)

/-- Calculates the gain percentage given the cost price and selling price -/
def gainPercentage (costPrice : ℚ) (sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem article_gain_percentage :
  let cp := costPrice 170 15
  gainPercentage cp 240 = 20 := by
  sorry

end article_gain_percentage_l2360_236028


namespace smallest_in_A_l2360_236024

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the set A
def A : Set ℕ := {n | 11 ∣ sumOfDigits n ∧ 11 ∣ sumOfDigits (n + 1)}

-- State the theorem
theorem smallest_in_A : 
  2899999 ∈ A ∧ ∀ m ∈ A, m < 2899999 → m = 2899999 := by sorry

end smallest_in_A_l2360_236024


namespace computer_time_theorem_l2360_236039

/-- Calculates the average time per person on a computer given the number of people, 
    number of computers, and working day duration. -/
def averageComputerTime (people : ℕ) (computers : ℕ) (workingHours : ℕ) (workingMinutes : ℕ) : ℕ :=
  let totalMinutes : ℕ := workingHours * 60 + workingMinutes
  let totalComputerTime : ℕ := totalMinutes * computers
  totalComputerTime / people

/-- Theorem stating that given 8 people, 5 computers, and a working day of 2 hours and 32 minutes, 
    the average time each person spends on a computer is 95 minutes. -/
theorem computer_time_theorem :
  averageComputerTime 8 5 2 32 = 95 := by
  sorry

end computer_time_theorem_l2360_236039


namespace right_triangle_altitude_ratio_l2360_236058

theorem right_triangle_altitude_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : b = (3/2) * a) (d : ℝ) (h_altitude : d^2 = (a*b)/c) :
  (c-d)/d = Real.sqrt 6 / 3 := by
  sorry

end right_triangle_altitude_ratio_l2360_236058


namespace equation_solution_l2360_236054

theorem equation_solution : ∃ x : ℝ, 2 * ((x - 1) - (2 * x + 1)) = 6 ∧ x = -5 := by sorry

end equation_solution_l2360_236054


namespace smaller_circle_radius_l2360_236079

theorem smaller_circle_radius (r_large : ℝ) (r_small : ℝ) : 
  r_large = 4 →
  π * r_small^2 = (1/2) * π * r_large^2 →
  (π * r_small^2) + (π * r_large^2 - π * r_small^2) = 2 * (π * r_large^2 - π * r_small^2) →
  r_small = 2 * Real.sqrt 2 := by
  sorry

end smaller_circle_radius_l2360_236079


namespace three_digit_seven_divisible_by_five_l2360_236057

theorem three_digit_seven_divisible_by_five (N : ℕ) : 
  (100 ≤ N ∧ N ≤ 999) →  -- N is a three-digit number
  (N % 10 = 7) →         -- N has a ones digit of 7
  (N % 5 = 0) →          -- N is divisible by 5
  False :=               -- This is impossible
by sorry

end three_digit_seven_divisible_by_five_l2360_236057


namespace max_positive_numbers_with_zero_average_l2360_236045

theorem max_positive_numbers_with_zero_average (numbers : List ℝ) : 
  numbers.length = 20 → numbers.sum / numbers.length = 0 → 
  (numbers.filter (λ x => x > 0)).length ≤ 19 := by
sorry

end max_positive_numbers_with_zero_average_l2360_236045


namespace ratio_problem_l2360_236092

theorem ratio_problem (a b x m : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a / b = 4 / 5 → x = a * (1 + 1/4) → m = b * (1 - 4/5) → m / x = 1 / 5 := by
  sorry

end ratio_problem_l2360_236092


namespace women_decrease_l2360_236071

theorem women_decrease (initial_men initial_women final_men final_women : ℕ) : 
  initial_men / initial_women = 4 / 5 →
  final_men = initial_men + 2 →
  24 = initial_women - 3 →
  final_men = 14 →
  final_women = 24 →
  initial_women - final_women = 3 :=
by
  sorry

end women_decrease_l2360_236071


namespace square_plus_one_gt_x_l2360_236011

theorem square_plus_one_gt_x : ∀ x : ℝ, x^2 + 1 > x := by sorry

end square_plus_one_gt_x_l2360_236011


namespace no_triangle_with_cube_sum_equal_to_product_l2360_236018

theorem no_triangle_with_cube_sum_equal_to_product (x y z : ℝ) :
  (0 < x ∧ 0 < y ∧ 0 < z) →
  (x + y > z ∧ y + z > x ∧ z + x > y) →
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
by sorry

end no_triangle_with_cube_sum_equal_to_product_l2360_236018


namespace lock_combination_l2360_236073

-- Define the base
def base : ℕ := 11

-- Define the function to convert from base 11 to decimal
def toDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (λ acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

-- Define the equation in base 11
axiom equation_holds : ∃ (S T A R E B : ℕ),
  S < base ∧ T < base ∧ A < base ∧ R < base ∧ E < base ∧ B < base ∧
  S ≠ T ∧ S ≠ A ∧ S ≠ R ∧ S ≠ E ∧ S ≠ B ∧
  T ≠ A ∧ T ≠ R ∧ T ≠ E ∧ T ≠ B ∧
  A ≠ R ∧ A ≠ E ∧ A ≠ B ∧
  R ≠ E ∧ R ≠ B ∧
  E ≠ B ∧
  (S * base^3 + T * base^2 + A * base + R) +
  (T * base^3 + A * base^2 + R * base + S) +
  (R * base^3 + E * base^2 + S * base + T) +
  (R * base^3 + A * base^2 + R * base + E) +
  (B * base^3 + E * base^2 + A * base + R) =
  (B * base^3 + E * base^2 + S * base + T)

-- Theorem to prove
theorem lock_combination : 
  ∃ (S T A R : ℕ), toDecimal [S, T, A, R] = 7639 ∧
  (∃ (E B : ℕ), 
    S < base ∧ T < base ∧ A < base ∧ R < base ∧ E < base ∧ B < base ∧
    S ≠ T ∧ S ≠ A ∧ S ≠ R ∧ S ≠ E ∧ S ≠ B ∧
    T ≠ A ∧ T ≠ R ∧ T ≠ E ∧ T ≠ B ∧
    A ≠ R ∧ A ≠ E ∧ A ≠ B ∧
    R ≠ E ∧ R ≠ B ∧
    E ≠ B ∧
    (S * base^3 + T * base^2 + A * base + R) +
    (T * base^3 + A * base^2 + R * base + S) +
    (R * base^3 + E * base^2 + S * base + T) +
    (R * base^3 + A * base^2 + R * base + E) +
    (B * base^3 + E * base^2 + A * base + R) =
    (B * base^3 + E * base^2 + S * base + T)) :=
by sorry

end lock_combination_l2360_236073


namespace sum_of_distinct_roots_is_zero_l2360_236069

theorem sum_of_distinct_roots_is_zero 
  (a b c x y : ℝ) 
  (ha : a^3 + a*x + y = 0)
  (hb : b^3 + b*x + y = 0)
  (hc : c^3 + c*x + y = 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  a + b + c = 0 := by
  sorry

end sum_of_distinct_roots_is_zero_l2360_236069


namespace chef_apples_l2360_236036

/-- Represents the number of apples used to make the pie -/
def apples_used : ℕ := 15

/-- Represents the number of apples left after making the pie -/
def apples_left : ℕ := 4

/-- Represents the total number of apples before making the pie -/
def total_apples : ℕ := apples_used + apples_left

/-- Theorem stating that the total number of apples before making the pie
    is equal to the sum of apples used and apples left -/
theorem chef_apples : total_apples = apples_used + apples_left := by
  sorry

end chef_apples_l2360_236036


namespace pet_store_cages_l2360_236032

theorem pet_store_cages (birds_per_cage : ℕ) (total_birds : ℕ) (num_cages : ℕ) : 
  birds_per_cage = 8 → 
  total_birds = 48 → 
  num_cages * birds_per_cage = total_birds → 
  num_cages = 6 := by
sorry

end pet_store_cages_l2360_236032


namespace first_wheat_rate_calculation_l2360_236008

-- Define the variables and constants
def first_wheat_quantity : ℝ := 30
def second_wheat_quantity : ℝ := 20
def second_wheat_rate : ℝ := 14.25
def profit_percentage : ℝ := 0.10
def selling_rate : ℝ := 13.86

-- Define the theorem
theorem first_wheat_rate_calculation (x : ℝ) : 
  (1 + profit_percentage) * (first_wheat_quantity * x + second_wheat_quantity * second_wheat_rate) = 
  (first_wheat_quantity + second_wheat_quantity) * selling_rate → 
  x = 11.50 := by
sorry

end first_wheat_rate_calculation_l2360_236008


namespace actual_distance_traveled_l2360_236043

theorem actual_distance_traveled (original_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) :
  original_speed = 15 →
  increased_speed = 25 →
  additional_distance = 35 →
  (∃ (time : ℝ), time > 0 ∧ time * increased_speed = time * original_speed + additional_distance) →
  ∃ (actual_distance : ℝ), actual_distance = 52.5 ∧ actual_distance = original_speed * (actual_distance / original_speed) :=
by sorry

end actual_distance_traveled_l2360_236043


namespace wire_ratio_proof_l2360_236005

theorem wire_ratio_proof (total_length longer_length shorter_length : ℝ) 
  (h1 : total_length = 14)
  (h2 : shorter_length = 4)
  (h3 : longer_length = total_length - shorter_length) :
  shorter_length / longer_length = 2 / 5 := by
  sorry

end wire_ratio_proof_l2360_236005


namespace moving_circle_theorem_l2360_236046

-- Define the moving circle
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_A : center.1 ^ 2 + center.2 ^ 2 = (center.1 - 2) ^ 2 + center.2 ^ 2
  cuts_y_axis : ∃ (y : ℝ), center.1 ^ 2 + (y - center.2) ^ 2 = center.1 ^ 2 + center.2 ^ 2 ∧ y ^ 2 = 4

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define the fixed point N
structure FixedPointN where
  x₀ : ℝ

-- Define the chord BD
structure ChordBD (n : FixedPointN) where
  m : ℝ
  passes_through_N : ∀ (y : ℝ), trajectory (n.x₀ + m * y) y

-- Define the angle BAD
def angle_BAD_obtuse (n : FixedPointN) (bd : ChordBD n) : Prop :=
  ∀ (y₁ y₂ : ℝ), 
    trajectory (n.x₀ + bd.m * y₁) y₁ → 
    trajectory (n.x₀ + bd.m * y₂) y₂ → 
    (n.x₀ + bd.m * y₁ - 2) * (n.x₀ + bd.m * y₂ - 2) + y₁ * y₂ < 0

-- The main theorem
theorem moving_circle_theorem :
  (∀ (mc : MovingCircle), trajectory mc.center.1 mc.center.2) ∧
  (∀ (n : FixedPointN), 
    (∀ (bd : ChordBD n), angle_BAD_obtuse n bd) → 
    (4 - 2 * Real.sqrt 3 < n.x₀ ∧ n.x₀ < 4 + 2 * Real.sqrt 3 ∧ n.x₀ ≠ 2)) :=
sorry

end moving_circle_theorem_l2360_236046


namespace coffee_package_size_l2360_236076

theorem coffee_package_size (total_coffee : ℕ) (large_package_size : ℕ) (large_package_count : ℕ) (small_package_count_diff : ℕ) :
  total_coffee = 55 →
  large_package_size = 10 →
  large_package_count = 3 →
  small_package_count_diff = 2 →
  ∃ (small_package_size : ℕ),
    small_package_size * (large_package_count + small_package_count_diff) +
    large_package_size * large_package_count = total_coffee ∧
    small_package_size = 5 :=
by sorry

end coffee_package_size_l2360_236076


namespace prime_satisfying_condition_l2360_236004

def satisfies_condition (p : Nat) : Prop :=
  Nat.Prime p ∧
  ∀ q : Nat, Nat.Prime q → q < p →
    ∀ k r : Nat, p = k * q + r → 0 ≤ r → r < q →
      ∀ a : Nat, a > 1 → ¬(a^2 ∣ r)

theorem prime_satisfying_condition :
  {p : Nat | satisfies_condition p} = {2, 3, 5, 7, 13} := by sorry

end prime_satisfying_condition_l2360_236004


namespace expression_value_l2360_236083

theorem expression_value (x : ℝ) (h : 5 * x^2 - x - 1 = 0) :
  (3*x + 2) * (3*x - 2) + x * (x - 2) = -2 := by
  sorry

end expression_value_l2360_236083


namespace carl_index_card_cost_l2360_236051

/-- The cost of index cards for Carl's classes -/
def total_cost (cards_per_student : ℕ) (periods : ℕ) (students_per_class : ℕ) (pack_size : ℕ) (pack_cost : ℚ) : ℚ :=
  let total_cards := cards_per_student * periods * students_per_class
  let packs_needed := (total_cards + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * pack_cost

/-- Proof that Carl spent $108 on index cards -/
theorem carl_index_card_cost :
  total_cost 10 6 30 50 3 = 108 := by
  sorry

end carl_index_card_cost_l2360_236051


namespace chord_length_l2360_236017

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end chord_length_l2360_236017


namespace fathers_age_multiple_l2360_236075

theorem fathers_age_multiple (son_age : ℕ) (father_age : ℕ) (k : ℕ) : 
  father_age = 27 →
  father_age = k * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 8 →
  k = 3 := by
  sorry

end fathers_age_multiple_l2360_236075


namespace quadratic_rewrite_l2360_236031

theorem quadratic_rewrite (b n : ℝ) : 
  (∀ x, x^2 + b*x + 72 = (x + n)^2 + 20) → 
  n > 0 → 
  b = 4 * Real.sqrt 13 := by
sorry

end quadratic_rewrite_l2360_236031


namespace storks_birds_difference_l2360_236064

theorem storks_birds_difference : 
  let initial_birds : ℕ := 2
  let initial_storks : ℕ := 6
  let additional_birds : ℕ := 3
  let final_birds : ℕ := initial_birds + additional_birds
  initial_storks - final_birds = 1 := by sorry

end storks_birds_difference_l2360_236064


namespace cubic_polynomial_sum_l2360_236095

/-- Given a cubic polynomial Q with specific values at 0, 1, and -1, prove that Q(2) + Q(-2) = 20m -/
theorem cubic_polynomial_sum (m : ℝ) (Q : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + 2 * m) →
  Q 0 = 2 * m →
  Q 1 = 3 * m →
  Q (-1) = 5 * m →
  Q 2 + Q (-2) = 20 * m :=
by sorry

end cubic_polynomial_sum_l2360_236095


namespace smallest_integer_a_l2360_236059

theorem smallest_integer_a (a b : ℤ) : 
  (∃ k : ℤ, a > k ∧ a < 21) →
  (b > 19 ∧ b < 31) →
  (a / b : ℚ) ≤ 2/3 →
  (∀ m : ℤ, m < a → m ≤ 13) :=
by sorry

end smallest_integer_a_l2360_236059


namespace four_birdhouses_built_l2360_236081

/-- The number of birdhouses that can be built with a given budget -/
def num_birdhouses (plank_cost nail_cost planks_per_house nails_per_house budget : ℚ) : ℚ :=
  budget / (plank_cost * planks_per_house + nail_cost * nails_per_house)

/-- Theorem stating that 4 birdhouses can be built with $88 given the specified costs and materials -/
theorem four_birdhouses_built :
  num_birdhouses 3 0.05 7 20 88 = 4 := by
  sorry

end four_birdhouses_built_l2360_236081


namespace initial_men_count_l2360_236055

/-- The number of days it takes the initial group to complete the job -/
def initial_days : ℕ := 15

/-- The number of men in the second group -/
def second_group_men : ℕ := 18

/-- The number of days it takes the second group to complete the job -/
def second_group_days : ℕ := 20

/-- The total amount of work in man-days -/
def total_work : ℕ := second_group_men * second_group_days

/-- The number of men initially working on the job -/
def initial_men : ℕ := total_work / initial_days

theorem initial_men_count : initial_men = 24 := by
  sorry

end initial_men_count_l2360_236055


namespace multiply_102_98_l2360_236077

theorem multiply_102_98 : 102 * 98 = 9996 := by
  sorry

end multiply_102_98_l2360_236077


namespace simplify_expression_l2360_236016

theorem simplify_expression (y : ℝ) :
  3 * y + 9 * y^2 - 15 - (5 - 3 * y - 9 * y^2) = 18 * y^2 + 6 * y - 20 := by
  sorry

end simplify_expression_l2360_236016


namespace bottles_per_child_per_day_is_three_l2360_236040

/-- Represents a children's camp with water consumption information -/
structure ChildrenCamp where
  group1 : Nat
  group2 : Nat
  group3 : Nat
  initialCases : Nat
  bottlesPerCase : Nat
  campDuration : Nat
  additionalBottles : Nat

/-- Calculates the number of bottles each child consumes per day -/
def bottlesPerChildPerDay (camp : ChildrenCamp) : Rat :=
  let group4 := (camp.group1 + camp.group2 + camp.group3) / 2
  let totalChildren := camp.group1 + camp.group2 + camp.group3 + group4
  let initialBottles := camp.initialCases * camp.bottlesPerCase
  let totalBottles := initialBottles + camp.additionalBottles
  (totalBottles : Rat) / (totalChildren * camp.campDuration)

/-- Theorem stating that for the given camp configuration, each child consumes 3 bottles per day -/
theorem bottles_per_child_per_day_is_three :
  let camp := ChildrenCamp.mk 14 16 12 13 24 3 255
  bottlesPerChildPerDay camp = 3 := by sorry

end bottles_per_child_per_day_is_three_l2360_236040


namespace pet_shop_total_l2360_236047

theorem pet_shop_total (dogs cats bunnies : ℕ) : 
  dogs = 154 → 
  dogs * 8 = bunnies * 7 → 
  dogs + bunnies = 330 :=
by
  sorry

end pet_shop_total_l2360_236047


namespace greg_bike_ride_l2360_236063

/-- Proves that Greg wants to ride 8 blocks given the conditions of the problem -/
theorem greg_bike_ride (rotations_per_block : ℕ) (rotations_so_far : ℕ) (rotations_needed : ℕ) :
  rotations_per_block = 200 →
  rotations_so_far = 600 →
  rotations_needed = 1000 →
  (rotations_so_far + rotations_needed) / rotations_per_block = 8 := by
  sorry

end greg_bike_ride_l2360_236063


namespace sweet_number_existence_l2360_236003

def is_sweet (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [0, 1, 2, 4, 8]

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sweet_number_existence : ∃ n : ℕ,
  n > 0 ∧
  is_sweet n ∧
  is_sweet (n^2) ∧
  is_sweet (n^3) ∧
  digit_sum n = 2014 := by
  sorry

end sweet_number_existence_l2360_236003


namespace largest_divisor_of_n4_minus_n2_l2360_236085

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  ∃ (m : ℕ), m = 6 ∧ 
  (∃ (k : ℤ), n^4 - n^2 = m * k) ∧ 
  (∀ (d : ℕ), d > m → ¬∃ (j : ℤ), ∀ (n : ℤ), n^4 - n^2 = d * j) :=
sorry

end largest_divisor_of_n4_minus_n2_l2360_236085


namespace no_valid_formation_l2360_236037

/-- Represents a rectangular formation of musicians. -/
structure Formation where
  rows : ℕ
  musicians_per_row : ℕ

/-- Checks if a formation is valid according to the given conditions. -/
def is_valid_formation (f : Formation) : Prop :=
  f.rows * f.musicians_per_row = 400 ∧
  f.musicians_per_row % 4 = 0 ∧
  10 ≤ f.musicians_per_row ∧
  f.musicians_per_row ≤ 50

/-- Represents the constraint of having a triangle formation for brass section. -/
def has_triangle_brass_formation (f : Formation) : Prop :=
  f.rows ≥ 3 ∧
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ a + b + c = 100 ∧
  a % (f.musicians_per_row / 4) = 0 ∧
  b % (f.musicians_per_row / 4) = 0 ∧
  c % (f.musicians_per_row / 4) = 0

/-- The main theorem stating that no valid formation exists. -/
theorem no_valid_formation :
  ¬∃ (f : Formation), is_valid_formation f ∧ has_triangle_brass_formation f :=
sorry

end no_valid_formation_l2360_236037


namespace smallest_number_with_given_remainders_l2360_236089

theorem smallest_number_with_given_remainders : ∃ (b : ℕ), 
  b > 0 ∧
  b % 4 = 2 ∧
  b % 3 = 2 ∧
  b % 5 = 3 ∧
  (∀ (x : ℕ), x > 0 ∧ x % 4 = 2 ∧ x % 3 = 2 ∧ x % 5 = 3 → x ≥ b) ∧
  b = 38 :=
by sorry

end smallest_number_with_given_remainders_l2360_236089


namespace quadratic_minimum_value_l2360_236002

theorem quadratic_minimum_value : 
  (∀ x : ℝ, x^2 + 4*x + 5 ≥ 1) ∧ (∃ x : ℝ, x^2 + 4*x + 5 = 1) := by
  sorry

end quadratic_minimum_value_l2360_236002


namespace bottles_maria_drank_l2360_236025

theorem bottles_maria_drank (initial bottles_bought bottles_remaining : ℕ) : 
  initial = 14 → bottles_bought = 45 → bottles_remaining = 51 → 
  initial + bottles_bought - bottles_remaining = 8 := by
sorry

end bottles_maria_drank_l2360_236025


namespace distance_to_cegled_l2360_236065

/-- The problem setup for calculating the distance to Cegléd -/
structure TravelProblem where
  s : ℝ  -- Total distance from home to Cegléd
  v : ℝ  -- Planned speed for both Antal and Béla
  t : ℝ  -- Planned travel time
  s₁ : ℝ  -- Béla's travel distance when alone

/-- The conditions of the problem -/
def problem_conditions (p : TravelProblem) : Prop :=
  p.t = p.s / p.v ∧  -- Planned time
  p.s₁ = 4 * p.s / 5 ∧  -- Béla's solo distance
  p.s / 5 = 48 * (1 / 6) ∧  -- Final section travel time
  (4 * p.s₁) / (3 * p.v) = (4 * (p.s₁ + 2 * p.s / 5)) / (5 * p.v)  -- Time equivalence for travel

/-- The theorem stating that the total distance is 40 km -/
theorem distance_to_cegled (p : TravelProblem) 
  (h : problem_conditions p) : p.s = 40 := by
  sorry


end distance_to_cegled_l2360_236065


namespace student_age_problem_l2360_236098

theorem student_age_problem (num_students : ℕ) (teacher_age : ℕ) 
  (h1 : num_students = 20)
  (h2 : teacher_age = 42)
  (h3 : ∀ (student_avg : ℝ), 
    (num_students * student_avg + teacher_age) / (num_students + 1) = student_avg + 1) :
  ∃ (student_avg : ℝ), student_avg = 21 := by
sorry

end student_age_problem_l2360_236098


namespace largest_band_formation_l2360_236062

/-- Represents a band formation --/
structure BandFormation where
  totalMembers : ℕ
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (bf : BandFormation) : Prop :=
  bf.totalMembers < 120 ∧
  bf.totalMembers = bf.rows * bf.membersPerRow + 3 ∧
  bf.totalMembers = (bf.rows - 1) * (bf.membersPerRow + 2)

/-- Theorem stating that 231 is the largest number of band members satisfying the conditions --/
theorem largest_band_formation :
  ∀ bf : BandFormation, isValidFormation bf → bf.totalMembers ≤ 231 :=
by
  sorry

#check largest_band_formation

end largest_band_formation_l2360_236062


namespace alloy_ratio_proof_l2360_236023

/-- Proves that the ratio of lead to tin in alloy A is 2:3 given the specified conditions -/
theorem alloy_ratio_proof (alloy_A_weight : ℝ) (alloy_B_weight : ℝ) 
  (tin_copper_ratio_B : ℚ) (total_tin_new_alloy : ℝ) 
  (h1 : alloy_A_weight = 120)
  (h2 : alloy_B_weight = 180)
  (h3 : tin_copper_ratio_B = 3/5)
  (h4 : total_tin_new_alloy = 139.5) :
  ∃ (lead_A tin_A : ℝ), 
    lead_A + tin_A = alloy_A_weight ∧ 
    lead_A / tin_A = 2/3 := by
  sorry

end alloy_ratio_proof_l2360_236023


namespace ice_cream_flavors_l2360_236099

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of basic flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops in each new flavor -/
def num_scoops : ℕ := 5

theorem ice_cream_flavors :
  distribute num_scoops num_flavors = Nat.choose (num_scoops + num_flavors - 1) (num_flavors - 1) :=
by sorry

end ice_cream_flavors_l2360_236099


namespace problem_solution_l2360_236020

theorem problem_solution (p q r : ℝ) : 
  (∀ x : ℝ, (x - p) * (x - q) / (x - r) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2)) →
  p < q →
  p + 2*q + 3*r = 74 := by
sorry

end problem_solution_l2360_236020


namespace only_whole_number_between_l2360_236029

theorem only_whole_number_between (N : ℤ) : 
  (9.25 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 9.75) ↔ N = 38 := by
  sorry

end only_whole_number_between_l2360_236029


namespace triangle_angle_sum_l2360_236053

theorem triangle_angle_sum (A B C : Real) (h1 : A = 30) (h2 : B = 50) :
  C = 100 :=
by
  sorry

end triangle_angle_sum_l2360_236053


namespace success_permutations_l2360_236084

/-- The number of distinct permutations of a word with repeated letters -/
def permutationsWithRepetition (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- The word "SUCCESS" has 7 letters with 'S' appearing 3 times, 'C' appearing 2 times, 
    and 'U' and 'E' appearing once each -/
def successWord : (ℕ × List ℕ) :=
  (7, [3, 2, 1, 1])

theorem success_permutations :
  permutationsWithRepetition successWord.1 successWord.2 = 420 := by
  sorry

end success_permutations_l2360_236084


namespace square_vertices_distance_sum_l2360_236001

/-- Given a square with side length s, prove that a point P(x,y) satisfies the condition that the sum
    of squares of distances from P to each vertex is 4s² if and only if P lies on a circle centered
    at (s/2, s/2) with radius s/√2 -/
theorem square_vertices_distance_sum (s : ℝ) (x y : ℝ) :
  (x^2 + y^2) + (x^2 + (y - s)^2) + ((x - s)^2 + y^2) + ((x - s)^2 + (y - s)^2) = 4 * s^2 ↔
  (x - s/2)^2 + (y - s/2)^2 = (s/Real.sqrt 2)^2 :=
by sorry

end square_vertices_distance_sum_l2360_236001


namespace nested_radical_value_l2360_236096

def nested_radical (x : ℝ) : Prop := x = Real.sqrt (20 + x)

theorem nested_radical_value :
  ∃ x : ℝ, nested_radical x ∧ x = 5 :=
sorry

end nested_radical_value_l2360_236096


namespace opposite_solutions_system_l2360_236068

theorem opposite_solutions_system (x y m : ℝ) : 
  x - 2*y = -3 → 
  2*x + 3*y = m - 1 → 
  x = -y → 
  m = 2 := by
sorry

end opposite_solutions_system_l2360_236068


namespace f_satisfies_equation_l2360_236074

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 + 1) / (2*x*(1-x))

theorem f_satisfies_equation :
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f (1/x) + f (1-x) = x :=
by
  sorry

end f_satisfies_equation_l2360_236074


namespace floor_ceil_sum_l2360_236007

theorem floor_ceil_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end floor_ceil_sum_l2360_236007


namespace smallest_five_digit_negative_congruent_to_5_mod_17_l2360_236086

theorem smallest_five_digit_negative_congruent_to_5_mod_17 : 
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 5 [ZMOD 17] → n ≥ -10013 :=
by sorry

end smallest_five_digit_negative_congruent_to_5_mod_17_l2360_236086


namespace nut_distribution_theorem_l2360_236050

/-- Represents a distribution of nuts among three piles -/
structure NutDistribution :=
  (pile1 pile2 pile3 : ℕ)

/-- Represents an operation of moving nuts between piles -/
inductive MoveOperation
  | move12 : MoveOperation  -- Move from pile 1 to pile 2
  | move13 : MoveOperation  -- Move from pile 1 to pile 3
  | move21 : MoveOperation  -- Move from pile 2 to pile 1
  | move23 : MoveOperation  -- Move from pile 2 to pile 3
  | move31 : MoveOperation  -- Move from pile 3 to pile 1
  | move32 : MoveOperation  -- Move from pile 3 to pile 2

/-- Applies a single move operation to a distribution -/
def applyMove (d : NutDistribution) (m : MoveOperation) : NutDistribution :=
  sorry

/-- Checks if a pile has an even number of nuts -/
def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Checks if a distribution has the desired property (one pile with half the nuts) -/
def hasHalfInOnePile (d : NutDistribution) : Prop :=
  let total := d.pile1 + d.pile2 + d.pile3
  d.pile1 = total / 2 ∨ d.pile2 = total / 2 ∨ d.pile3 = total / 2

/-- The main theorem statement -/
theorem nut_distribution_theorem (initial : NutDistribution) :
  isEven (initial.pile1 + initial.pile2 + initial.pile3) →
  ∃ (moves : List MoveOperation), 
    hasHalfInOnePile (moves.foldl applyMove initial) :=
by sorry

end nut_distribution_theorem_l2360_236050
