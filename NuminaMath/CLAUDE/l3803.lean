import Mathlib

namespace heine_valentine_treats_l3803_380303

/-- Given a total number of biscuits and dogs, calculate the number of biscuits per dog -/
def biscuits_per_dog (total_biscuits : ℕ) (num_dogs : ℕ) : ℕ :=
  total_biscuits / num_dogs

/-- Theorem: Mrs. Heine's Valentine's Day treats distribution -/
theorem heine_valentine_treats :
  let total_biscuits := 6
  let num_dogs := 2
  biscuits_per_dog total_biscuits num_dogs = 3 := by
  sorry

end heine_valentine_treats_l3803_380303


namespace trigonometric_identity_l3803_380301

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < π / 3) 
  (h3 : Real.sqrt 3 * Real.sin α + Real.cos α = Real.sqrt 6 / 2) : 
  (Real.cos (α + π / 6) = Real.sqrt 10 / 4) ∧ 
  (Real.cos (2 * α + 7 * π / 12) = (Real.sqrt 2 - Real.sqrt 30) / 8) := by
  sorry

end trigonometric_identity_l3803_380301


namespace shaded_area_calculation_l3803_380361

theorem shaded_area_calculation (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 40 →
  triangle_base = 15 →
  triangle_height = 15 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 1375 := by
  sorry

end shaded_area_calculation_l3803_380361


namespace heptagon_diagonals_l3803_380307

/-- The number of sides in a heptagon -/
def heptagon_sides : ℕ := 7

/-- Formula for the number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex heptagon is 14 -/
theorem heptagon_diagonals : num_diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l3803_380307


namespace area_of_region_l3803_380365

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 37 ∧ 
   A = Real.pi * (Real.sqrt ((x + 3)^2 + (y - 4)^2))^2 ∧
   x^2 + y^2 + 6*x - 8*y - 12 = 0) := by
sorry

end area_of_region_l3803_380365


namespace rectangle_folding_l3803_380326

theorem rectangle_folding (a b : ℝ) : 
  a = 5 ∧ 
  0 < b ∧ 
  b < 4 ∧ 
  (a - b)^2 + b^2 = 6 → 
  b = Real.sqrt 5 := by
sorry

end rectangle_folding_l3803_380326


namespace eighteenth_term_of_equal_sum_sequence_l3803_380327

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. -/
def EqualSumSequence (a : ℕ → ℝ) (sum : ℝ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = sum

theorem eighteenth_term_of_equal_sum_sequence
  (a : ℕ → ℝ)
  (sum : ℝ)
  (h_equal_sum : EqualSumSequence a sum)
  (h_first_term : a 1 = 2)
  (h_sum : sum = 5) :
  a 18 = 3 := by
sorry

end eighteenth_term_of_equal_sum_sequence_l3803_380327


namespace pit_width_is_five_l3803_380344

/-- Represents the dimensions and conditions of the field and pit problem -/
structure FieldPitProblem where
  field_length : ℝ
  field_width : ℝ
  pit_length : ℝ
  pit_depth : ℝ
  field_rise : ℝ

/-- Calculates the width of the pit given the problem conditions -/
def calculate_pit_width (problem : FieldPitProblem) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the pit width is 5 meters given the specified conditions -/
theorem pit_width_is_five (problem : FieldPitProblem) 
  (h1 : problem.field_length = 20)
  (h2 : problem.field_width = 10)
  (h3 : problem.pit_length = 8)
  (h4 : problem.pit_depth = 2)
  (h5 : problem.field_rise = 0.5) :
  calculate_pit_width problem = 5 := by
  sorry

end pit_width_is_five_l3803_380344


namespace kirill_height_l3803_380318

/-- Represents the heights of Kirill, his brother, sister, and cousin --/
structure FamilyHeights where
  kirill : ℕ
  brother : ℕ
  sister : ℕ
  cousin : ℕ

/-- The conditions of the problem --/
def HeightConditions (h : FamilyHeights) : Prop :=
  h.brother = h.kirill + 14 ∧
  h.sister = 2 * h.kirill ∧
  h.cousin = h.sister + 3 ∧
  h.kirill + h.brother + h.sister + h.cousin = 432

/-- The theorem stating Kirill's height --/
theorem kirill_height :
  ∀ h : FamilyHeights, HeightConditions h → h.kirill = 69 :=
by sorry

end kirill_height_l3803_380318


namespace borrowing_interest_rate_l3803_380379

/-- Proves that the interest rate at which a person borrowed money is 4% per annum,
    given the specified conditions. -/
theorem borrowing_interest_rate
  (loan_amount : ℝ)
  (loan_duration : ℕ)
  (lending_rate : ℝ)
  (yearly_gain : ℝ)
  (h1 : loan_amount = 7000)
  (h2 : loan_duration = 2)
  (h3 : lending_rate = 0.06)
  (h4 : yearly_gain = 140)
  : ∃ (borrowing_rate : ℝ), borrowing_rate = 0.04 := by
  sorry

end borrowing_interest_rate_l3803_380379


namespace complex_fraction_simplification_l3803_380333

theorem complex_fraction_simplification :
  (1/2 * 1/3 * 1/4 * 1/5 + 3/2 * 3/4 * 3/5) / (1/2 * 2/3 * 2/5) = 41/8 := by
  sorry

end complex_fraction_simplification_l3803_380333


namespace polynomial_characterization_l3803_380375

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that must be satisfied by the polynomial -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 2 * x * y * z = x + y + z →
    (P x) / (y * z) + (P y) / (z * x) + (P z) / (x * y) = P (x - y) + P (y - z) + P (z - x)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_characterization (P : RealPolynomial) 
  (h : SatisfiesCondition P) : 
  ∃ (c : ℝ), ∀ (x : ℝ), P x = c * (x^2 + 3) := by
  sorry

end polynomial_characterization_l3803_380375


namespace projection_result_l3803_380392

/-- Given two vectors a and b in ℝ², if both are projected onto the same vector v
    resulting in p, then p is equal to (48/53, 168/53). -/
theorem projection_result (a b v p : ℝ × ℝ) : 
  a = (5, 2) → 
  b = (-2, 4) → 
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ (a - p) • v = 0) → 
  (∃ (k₃ k₄ : ℝ), p = k₃ • v ∧ (b - p) • v = 0) → 
  p = (48/53, 168/53) := by
  sorry

end projection_result_l3803_380392


namespace initial_skittles_count_l3803_380386

/-- Proves that the initial number of Skittles is equal to the product of the number of friends and the number of Skittles each friend received. -/
theorem initial_skittles_count (num_friends num_skittles_per_friend : ℕ) :
  num_friends * num_skittles_per_friend = num_friends * num_skittles_per_friend :=
by sorry

#check initial_skittles_count 5 8

end initial_skittles_count_l3803_380386


namespace counterexample_exists_l3803_380387

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 2)) := by
  sorry

end counterexample_exists_l3803_380387


namespace min_sum_of_a_and_b_l3803_380339

/-- Given a line x/a + y/b = 1 where a > 0 and b > 0, and the line passes through (2, 2),
    the minimum value of a + b is 8. -/
theorem min_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 2/a + 2/b = 1) : 
  ∀ (x y : ℝ), x/a + y/b = 1 → a + b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ + b₀ = 8 := by
  sorry

end min_sum_of_a_and_b_l3803_380339


namespace evaluate_expression_l3803_380371

theorem evaluate_expression (b x : ℝ) (h : x = b + 10) : 2*x - b + 5 = b + 25 := by
  sorry

end evaluate_expression_l3803_380371


namespace expression_value_l3803_380328

theorem expression_value : 
  (2015^3 - 3 * 2015^2 * 2016 + 3 * 2015 * 2016^2 - 2016^3 + 1) / (2015 * 2016) = -3 := by
  sorry

end expression_value_l3803_380328


namespace absolute_value_of_negative_three_l3803_380388

theorem absolute_value_of_negative_three : |(-3 : ℤ)| = 3 := by
  sorry

end absolute_value_of_negative_three_l3803_380388


namespace first_day_pen_sales_l3803_380306

/-- Proves that given the conditions of pen sales over 13 days, 
    the number of pens sold on the first day is 96. -/
theorem first_day_pen_sales : ∀ (first_day_sales : ℕ),
  (first_day_sales + 12 * 44 = 13 * 48) →
  first_day_sales = 96 := by
  sorry

#check first_day_pen_sales

end first_day_pen_sales_l3803_380306


namespace solve_for_y_l3803_380382

theorem solve_for_y (x y : ℚ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end solve_for_y_l3803_380382


namespace twelve_roll_prob_l3803_380335

/-- Probability of a specific outcome on a standard six-sided die -/
def die_prob : ℚ := 1 / 6

/-- Probability of rolling any number except the previous one -/
def diff_prob : ℚ := 5 / 6

/-- Number of rolls before the 8th roll -/
def pre_8th_rolls : ℕ := 6

/-- Number of rolls between 8th and 12th (exclusive) -/
def post_8th_rolls : ℕ := 3

/-- The probability that the 12th roll is the last roll, given the 8th roll is a 4 -/
theorem twelve_roll_prob : 
  (1 : ℚ) * diff_prob ^ pre_8th_rolls * die_prob * diff_prob ^ post_8th_rolls * die_prob = 5^9 / 6^11 :=
by sorry

end twelve_roll_prob_l3803_380335


namespace xyz_inequality_l3803_380397

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : 
  Real.sqrt (x * y * z) ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z := by
sorry

end xyz_inequality_l3803_380397


namespace intersection_range_l3803_380350

/-- The set M representing an ellipse -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}

/-- The set N representing a line with slope m and y-intercept b -/
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

/-- Theorem stating the range of b for which M and N always intersect -/
theorem intersection_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) ↔ b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

#check intersection_range

end intersection_range_l3803_380350


namespace base_7_326_equals_base_4_2213_l3803_380311

def base_7_to_decimal (x y z : ℕ) : ℕ := x * 7^2 + y * 7 + z

def decimal_to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_7_326_equals_base_4_2213 :
  decimal_to_base_4 (base_7_to_decimal 3 2 6) = [2, 2, 1, 3] := by
  sorry

end base_7_326_equals_base_4_2213_l3803_380311


namespace triangle_angle_A_l3803_380325

theorem triangle_angle_A (A : Real) (h : 4 * Real.pi * Real.sin A - 3 * Real.arccos (-1/2) = 0) :
  A = Real.pi / 6 ∨ A = 5 * Real.pi / 6 :=
by sorry

end triangle_angle_A_l3803_380325


namespace chosen_number_l3803_380384

theorem chosen_number (x : ℝ) : (x / 6) - 15 = 5 → x = 120 := by
  sorry

end chosen_number_l3803_380384


namespace soccer_tournament_matches_l3803_380368

/-- The number of matches in a round-robin tournament with n teams -/
def numMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of matches between two groups of teams -/
def numMatchesBetweenGroups (n m : ℕ) : ℕ := n * m

theorem soccer_tournament_matches :
  (numMatches 3 = 3) ∧
  (numMatches 4 = 6) ∧
  (numMatchesBetweenGroups 3 4 = 12) := by
  sorry

#eval numMatches 3  -- Expected output: 3
#eval numMatches 4  -- Expected output: 6
#eval numMatchesBetweenGroups 3 4  -- Expected output: 12

end soccer_tournament_matches_l3803_380368


namespace octagon_side_length_l3803_380341

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (P Q R S : Point)

/-- Represents an octagon -/
structure Octagon :=
  (A B C D E F G H : Point)

/-- Checks if an octagon is equilateral -/
def is_equilateral (oct : Octagon) : Prop := sorry

/-- Checks if an octagon is convex -/
def is_convex (oct : Octagon) : Prop := sorry

/-- Checks if an octagon is inscribed in a rectangle -/
def is_inscribed (oct : Octagon) (rect : Rectangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_of_prime (n : ℕ) : Prop := sorry

theorem octagon_side_length (rect : Rectangle) (oct : Octagon) :
  distance rect.P rect.Q = 8 →
  distance rect.Q rect.R = 6 →
  is_inscribed oct rect →
  is_equilateral oct →
  is_convex oct →
  distance oct.A rect.P = distance oct.B rect.Q →
  distance oct.A rect.P < 4 →
  ∃ (k m n : ℕ), 
    distance oct.A oct.B = k + m * Real.sqrt n ∧ 
    not_divisible_by_square_of_prime n ∧
    k + m + n = 7 :=
by sorry

end octagon_side_length_l3803_380341


namespace cubic_function_properties_l3803_380377

-- Define the cubic function
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  -- Part 1: If there exists a point where the tangent line is parallel to the x-axis
  (∃ x : ℝ, f' a b x = 0) →
  a^2 ≥ 3*b ∧
  -- Part 2: If f(x) has extreme values at x = -1 and x = 3
  (f' a b (-1) = 0 ∧ f' a b 3 = 0) →
  a = 3 ∧ b = -9 ∧
  -- Part 3: If f(x) < 2c for all x ∈ [-2, 6], then c > 54
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 6 → f a b c x < 2*c) →
  c > 54 := by
sorry

end cubic_function_properties_l3803_380377


namespace complement_intersection_theorem_l3803_380308

universe u

def I : Set Char := {'b', 'c', 'd', 'e', 'f'}
def M : Set Char := {'b', 'c', 'f'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_theorem :
  (I \ M) ∩ N = {'d', 'e'} := by sorry

end complement_intersection_theorem_l3803_380308


namespace greatest_x_with_lcm_l3803_380317

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y = Nat.lcm x (Nat.lcm 15 21) ∧ y = 105) →
  x ≤ 105 ∧ ∃ (z : ℕ), z = 105 ∧ (∃ (w : ℕ), w = Nat.lcm z (Nat.lcm 15 21) ∧ w = 105) :=
by sorry

end greatest_x_with_lcm_l3803_380317


namespace a_value_l3803_380302

theorem a_value (a : ℝ) : 2 ∈ ({1, a, a^2 - a} : Set ℝ) → a = -1 := by
  sorry

end a_value_l3803_380302


namespace son_age_l3803_380300

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 25 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
sorry

end son_age_l3803_380300


namespace sum_of_abc_l3803_380305

theorem sum_of_abc (a b c : ℕ+) 
  (h1 : (a + b + c)^3 - a^3 - b^3 - c^3 = 150)
  (h2 : a < b) (h3 : b < c) : 
  a + b + c = 9 := by
  sorry

end sum_of_abc_l3803_380305


namespace no_square_in_triangle_grid_l3803_380389

/-- A point in the plane represented by its coordinates -/
structure Point where
  x : ℚ
  y : ℝ

/-- The grid of equilateral triangles -/
structure TriangleGrid where
  side_length : ℝ
  is_valid : side_length > 0

/-- A function that checks if a point is a valid vertex in the triangle grid -/
def is_vertex (grid : TriangleGrid) (p : Point) : Prop :=
  ∃ (k l : ℤ), p.x = k * (grid.side_length / 2) ∧ p.y = l * (Real.sqrt 3 * grid.side_length / 2)

/-- A function that checks if four points form a square -/
def is_square (a b c d : Point) : Prop :=
  let dist (p q : Point) := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)
  (dist a b = dist b c) ∧ (dist b c = dist c d) ∧ (dist c d = dist d a) ∧
  (dist a c = dist b d) ∧
  ((b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) = 0)

/-- The main theorem stating that it's impossible to form a square from vertices of the triangle grid -/
theorem no_square_in_triangle_grid (grid : TriangleGrid) :
  ¬∃ (a b c d : Point), is_vertex grid a ∧ is_vertex grid b ∧ is_vertex grid c ∧ is_vertex grid d ∧ is_square a b c d :=
sorry

end no_square_in_triangle_grid_l3803_380389


namespace parallelogram_to_rhombus_transformation_l3803_380309

def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (-3, -1)
def D : ℝ × ℝ := (1, -1)

def M (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![k, 1; 0, 2]

def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (A.1 - B.1 = D.1 - C.1) ∧ (A.2 - B.2 = D.2 - C.2) ∧
  (A.1 - D.1 = B.1 - C.1) ∧ (A.2 - D.2 = B.2 - C.2)

def is_rhombus (A B C D : ℝ × ℝ) : Prop :=
  let AB := ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  AB = BC ∧ BC = CD ∧ CD = DA

def transform_point (M : Matrix (Fin 2) (Fin 2) ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (M 0 0 * p.1 + M 0 1 * p.2, M 1 0 * p.1 + M 1 1 * p.2)

theorem parallelogram_to_rhombus_transformation (k : ℝ) :
  k < 0 →
  is_parallelogram A B C D →
  is_rhombus (transform_point (M k) A) (transform_point (M k) B)
             (transform_point (M k) C) (transform_point (M k) D) →
  k = -1 ∧ M k⁻¹ = !![(-1 : ℝ), (1/2 : ℝ); 0, (1/2 : ℝ)] :=
sorry

end parallelogram_to_rhombus_transformation_l3803_380309


namespace population_falls_below_threshold_l3803_380378

/-- The annual decrease rate of the finch population -/
def annual_decrease_rate : ℝ := 0.3

/-- The threshold below which we consider the population to have significantly decreased -/
def threshold : ℝ := 0.15

/-- The function that calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 - annual_decrease_rate) ^ years

/-- Theorem stating that it takes 6 years for the population to fall below the threshold -/
theorem population_falls_below_threshold (initial_population : ℝ) (h : initial_population > 0) :
  population_after_years initial_population 6 < threshold * initial_population ∧
  population_after_years initial_population 5 ≥ threshold * initial_population :=
by sorry

end population_falls_below_threshold_l3803_380378


namespace rollo_guinea_pig_food_l3803_380390

/-- The amount of food needed for Rollo's guinea pigs -/
def guinea_pig_food : ℕ → ℕ
| 1 => 2  -- First guinea pig eats 2 cups
| 2 => 2 * guinea_pig_food 1  -- Second eats twice as much as the first
| 3 => guinea_pig_food 2 + 3  -- Third eats 3 cups more than the second
| _ => 0  -- For completeness, though we only have 3 guinea pigs

/-- The total amount of food needed for all guinea pigs -/
def total_food : ℕ := guinea_pig_food 1 + guinea_pig_food 2 + guinea_pig_food 3

theorem rollo_guinea_pig_food : total_food = 13 := by
  sorry

end rollo_guinea_pig_food_l3803_380390


namespace smallest_with_property_l3803_380356

/-- A function that returns the list of digits of a natural number. -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if two lists of natural numbers are permutations of each other. -/
def is_permutation (l1 l2 : List ℕ) : Prop := sorry

/-- The property we're looking for: when multiplied by 9, the result has the same digits in a different order. -/
def has_property (n : ℕ) : Prop :=
  is_permutation (digits n) (digits (9 * n))

/-- The theorem stating that 1089 is the smallest natural number with the desired property. -/
theorem smallest_with_property :
  has_property 1089 ∧ ∀ m : ℕ, m < 1089 → ¬(has_property m) := by sorry

end smallest_with_property_l3803_380356


namespace sum_equals_1300_l3803_380370

/-- Converts a number from base 15 to base 10 -/
def base15ToBase10 (n : Nat) : Nat :=
  (n / 100) * 225 + ((n / 10) % 10) * 15 + (n % 10)

/-- Converts a number from base 7 to base 10, where 'A' represents 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 49 + ((n / 10) % 10) * 7 + (n % 10)

/-- Theorem stating that the sum of 537 (base 15) and 1A4 (base 7) equals 1300 in base 10 -/
theorem sum_equals_1300 : 
  base15ToBase10 537 + base7ToBase10 194 = 1300 := by
  sorry

end sum_equals_1300_l3803_380370


namespace fruit_arrangement_l3803_380321

-- Define the fruits and boxes
inductive Fruit
| Apple
| Pear
| Orange
| Banana

inductive Box
| One
| Two
| Three
| Four

-- Define the arrangement of fruits in boxes
def Arrangement := Box → Fruit

-- Define the labels on the boxes
def Label (b : Box) : Prop :=
  match b with
  | Box.One => ∃ a : Arrangement, a Box.One = Fruit.Orange
  | Box.Two => ∃ a : Arrangement, a Box.Two = Fruit.Pear
  | Box.Three => ∃ a : Arrangement, (a Box.One = Fruit.Banana) → (a Box.Three = Fruit.Apple ∨ a Box.Three = Fruit.Pear)
  | Box.Four => ∃ a : Arrangement, a Box.Four = Fruit.Apple

-- Define the correct arrangement
def CorrectArrangement : Arrangement :=
  fun b => match b with
  | Box.One => Fruit.Banana
  | Box.Two => Fruit.Apple
  | Box.Three => Fruit.Orange
  | Box.Four => Fruit.Pear

-- State the theorem
theorem fruit_arrangement :
  ∀ a : Arrangement,
  (∀ b : Box, ∀ f : Fruit, (a b = f) → (∃! b' : Box, a b' = f)) →
  (∀ b : Box, ¬Label b) →
  a = CorrectArrangement :=
sorry

end fruit_arrangement_l3803_380321


namespace candy_distribution_l3803_380393

def candies_for_child (n : ℕ) : ℕ := 2^(n - 1)

def total_candies (n : ℕ) : ℕ := 2^n - 1

theorem candy_distribution (total : ℕ) (h : total = 2007) :
  let n := (Nat.log 2 (total + 1)).succ
  (total_candies n - total, n) = (40, 11) := by sorry

end candy_distribution_l3803_380393


namespace equal_intercept_line_equation_l3803_380367

/-- A line with equal x and y intercepts passing through (-1, 2) -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (-1, 2)
  point_condition : 2 = m * (-1) + b
  -- The line has equal x and y intercepts
  equal_intercepts : b ≠ 0 → -b/m = b

/-- The equation of an EqualInterceptLine is either 2x + y = 0 or x + y - 1 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 1) :=
sorry

end equal_intercept_line_equation_l3803_380367


namespace eiffel_tower_height_difference_l3803_380359

/-- The height difference between two structures -/
def height_difference (taller_height shorter_height : ℝ) : ℝ :=
  taller_height - shorter_height

/-- The heights of the Burj Khalifa and Eiffel Tower -/
def burj_khalifa_height : ℝ := 830
def eiffel_tower_height : ℝ := 324

/-- Theorem: The Eiffel Tower is 506 meters lower than the Burj Khalifa -/
theorem eiffel_tower_height_difference : 
  height_difference burj_khalifa_height eiffel_tower_height = 506 := by
  sorry

end eiffel_tower_height_difference_l3803_380359


namespace product_constraint_sum_l3803_380369

theorem product_constraint_sum (w x y z : ℕ) : 
  w * x * y * z = 720 → 
  0 < w → w < x → x < y → y < z → z < 20 → 
  w + z = 14 := by
sorry

end product_constraint_sum_l3803_380369


namespace gold_quarter_weight_l3803_380346

/-- The weight of a gold quarter in ounces -/
def quarter_weight : ℝ := 0.2

/-- The value of a quarter in dollars when spent in a store -/
def quarter_store_value : ℝ := 0.25

/-- The value of an ounce of melted gold in dollars -/
def melted_gold_value_per_ounce : ℝ := 100

/-- The ratio of melted value to store value -/
def melted_to_store_ratio : ℕ := 80

theorem gold_quarter_weight :
  quarter_weight * melted_gold_value_per_ounce = melted_to_store_ratio * quarter_store_value := by
  sorry

end gold_quarter_weight_l3803_380346


namespace ellipse_hyperbola_product_l3803_380334

theorem ellipse_hyperbola_product (a b : ℝ) 
  (h_ellipse : b^2 - a^2 = 25)
  (h_hyperbola : a^2 + b^2 = 64) : 
  |a * b| = Real.sqrt (3461 / 4) := by
sorry

end ellipse_hyperbola_product_l3803_380334


namespace determine_b_l3803_380345

theorem determine_b (a b : ℝ) (h1 : 3 * a + 4 = 1) (h2 : b - 2 * a = 5) : b = 3 := by
  sorry

end determine_b_l3803_380345


namespace yearly_calls_cost_is_78_l3803_380383

/-- The total cost of weekly calls for a year -/
def total_cost_yearly_calls (weeks_per_year : ℕ) (call_duration_minutes : ℕ) (cost_per_minute : ℚ) : ℚ :=
  (weeks_per_year : ℚ) * (call_duration_minutes : ℚ) * cost_per_minute

/-- Theorem stating that the total cost for a year of weekly calls is $78 -/
theorem yearly_calls_cost_is_78 :
  total_cost_yearly_calls 52 30 (5 / 100) = 78 := by
  sorry

end yearly_calls_cost_is_78_l3803_380383


namespace floating_time_calculation_l3803_380323

/-- Floating time calculation -/
theorem floating_time_calculation
  (boat_speed_with_current : ℝ)
  (boat_speed_against_current : ℝ)
  (distance_floated : ℝ)
  (h1 : boat_speed_with_current = 28)
  (h2 : boat_speed_against_current = 24)
  (h3 : distance_floated = 20) :
  (distance_floated / ((boat_speed_with_current - boat_speed_against_current) / 2)) = 10 := by
  sorry

#check floating_time_calculation

end floating_time_calculation_l3803_380323


namespace sum_f_positive_l3803_380338

/-- The function f(x) = x + x³ -/
def f (x : ℝ) : ℝ := x + x^3

/-- Theorem: For any real numbers x₁, x₂, x₃ satisfying the given conditions,
    the sum f(x₁) + f(x₂) + f(x₃) is always positive -/
theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
    (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
    f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end sum_f_positive_l3803_380338


namespace dalton_watched_nine_movies_l3803_380330

/-- The number of movies watched by Dalton in the Superhero Fan Club -/
def dalton_movies : ℕ := sorry

/-- The number of movies watched by Hunter in the Superhero Fan Club -/
def hunter_movies : ℕ := 12

/-- The number of movies watched by Alex in the Superhero Fan Club -/
def alex_movies : ℕ := 15

/-- The number of movies watched together by all three members -/
def movies_watched_together : ℕ := 2

/-- The total number of different movies watched by the Superhero Fan Club -/
def total_different_movies : ℕ := 30

theorem dalton_watched_nine_movies :
  dalton_movies + hunter_movies + alex_movies - 3 * movies_watched_together = total_different_movies ∧
  dalton_movies = 9 := by sorry

end dalton_watched_nine_movies_l3803_380330


namespace identities_proof_l3803_380354

theorem identities_proof (a : ℝ) (n k : ℤ) : 
  ((-a^3 * (-a)^3)^2 + (-a^2 * (-a)^2)^3 = 0) ∧ 
  ((-1:ℝ)^n * a^(n+k) = (-a)^n * a^k) := by
  sorry

end identities_proof_l3803_380354


namespace racetrack_circumference_difference_l3803_380394

/-- The difference in circumferences of two concentric circles -/
theorem racetrack_circumference_difference (inner_diameter outer_diameter : ℝ) 
  (h1 : inner_diameter = 55)
  (h2 : outer_diameter = inner_diameter + 2 * 15) :
  π * outer_diameter - π * inner_diameter = 30 * π := by
  sorry

end racetrack_circumference_difference_l3803_380394


namespace T_formula_l3803_380391

def T : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * (n + 3) - 8) * T n

theorem T_formula (n : ℕ) : T n = n.factorial + 2^n := by
  sorry

end T_formula_l3803_380391


namespace correspondence_theorem_l3803_380363

theorem correspondence_theorem (m n : ℕ) (l : ℕ) 
  (h1 : l ≥ m * (n / 2))
  (h2 : l ≤ n * (m / 2)) :
  l = m * (n / 2) ∧ l = n * (m / 2) :=
sorry

end correspondence_theorem_l3803_380363


namespace pizza_calculation_l3803_380312

/-- The total number of pizzas made by Heather and Craig in two days -/
def total_pizzas (craig_day1 : ℕ) (heather_multiplier : ℕ) (craig_increase : ℕ) (heather_decrease : ℕ) : ℕ :=
  let craig_day2 := craig_day1 + craig_increase
  let heather_day1 := craig_day1 * heather_multiplier
  let heather_day2 := craig_day2 - heather_decrease
  craig_day1 + heather_day1 + craig_day2 + heather_day2

/-- Theorem stating the total number of pizzas made by Heather and Craig in two days -/
theorem pizza_calculation : total_pizzas 40 4 60 20 = 380 := by
  sorry

end pizza_calculation_l3803_380312


namespace compound_composition_l3803_380348

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  h : ℕ
  cl : ℕ
  o : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  h : ℝ
  cl : ℝ
  o : ℝ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (comp : CompoundComposition) (weights : AtomicWeights) : ℝ :=
  comp.h * weights.h + comp.cl * weights.cl + comp.o * weights.o

/-- The main theorem to prove -/
theorem compound_composition (weights : AtomicWeights) :
  let comp := CompoundComposition.mk 1 1 2
  weights.h = 1 ∧ weights.cl = 35.5 ∧ weights.o = 16 →
  molecularWeight comp weights = 68 := by
  sorry

end compound_composition_l3803_380348


namespace triangle_angle_sum_l3803_380337

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 90) (h2 : B = 50) : C = 40 := by
  sorry

end triangle_angle_sum_l3803_380337


namespace square_value_l3803_380315

theorem square_value (square x : ℤ) 
  (h1 : square + x = 80)
  (h2 : 3 * (square + x) - 2 * x = 164) : 
  square = 42 := by
sorry

end square_value_l3803_380315


namespace ellipse_hyperbola_tangent_l3803_380343

/-- The value of m for which the ellipse 3x^2 + 9y^2 = 9 and 
    the hyperbola (x-2)^2 - m(y+1)^2 = 1 are tangent -/
theorem ellipse_hyperbola_tangent : 
  ∃! m : ℝ, ∀ x y : ℝ, 
    (3 * x^2 + 9 * y^2 = 9 ∧ (x - 2)^2 - m * (y + 1)^2 = 1) →
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ 
      3 * p.1^2 + 9 * p.2^2 = 9 ∧ 
      (p.1 - 2)^2 - m * (p.2 + 1)^2 = 1) →
    m = 3 :=
sorry

end ellipse_hyperbola_tangent_l3803_380343


namespace range_of_a_l3803_380324

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 4) → (x^2 - 2*x + 1 - a^2 < 0)) →
  (a > 0) →
  (a ≥ 3) := by
sorry

end range_of_a_l3803_380324


namespace abs_inequality_l3803_380395

theorem abs_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end abs_inequality_l3803_380395


namespace resulting_polygon_has_18_sides_l3803_380349

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ

/-- Represents the arrangement of polygons. -/
structure PolygonArrangement where
  pentagon : RegularPolygon
  triangle : RegularPolygon
  octagon : RegularPolygon
  hexagon : RegularPolygon
  square : RegularPolygon

/-- The number of sides exposed to the outside for polygons adjacent to one other shape. -/
def exposedSidesOneAdjacent (p1 p2 : RegularPolygon) : ℕ :=
  p1.sides + p2.sides - 2

/-- The number of sides exposed to the outside for polygons adjacent to two other shapes. -/
def exposedSidesTwoAdjacent (p1 p2 p3 : RegularPolygon) : ℕ :=
  p1.sides + p2.sides + p3.sides - 6

/-- The total number of sides in the resulting polygon. -/
def totalSides (arrangement : PolygonArrangement) : ℕ :=
  exposedSidesOneAdjacent arrangement.pentagon arrangement.square +
  exposedSidesTwoAdjacent arrangement.triangle arrangement.octagon arrangement.hexagon

/-- Theorem stating that the resulting polygon has 18 sides. -/
theorem resulting_polygon_has_18_sides (arrangement : PolygonArrangement)
  (h1 : arrangement.pentagon.sides = 5)
  (h2 : arrangement.triangle.sides = 3)
  (h3 : arrangement.octagon.sides = 8)
  (h4 : arrangement.hexagon.sides = 6)
  (h5 : arrangement.square.sides = 4) :
  totalSides arrangement = 18 := by
  sorry

end resulting_polygon_has_18_sides_l3803_380349


namespace imaginary_part_of_z_l3803_380398

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) :
  z.im = 1/2 := by
  sorry

end imaginary_part_of_z_l3803_380398


namespace circumscribed_sphere_radius_for_specific_pyramid_l3803_380360

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_edge : ℝ
  side_edge : ℝ

/-- Radius of the circumscribed sphere of a regular triangular pyramid -/
def circumscribed_sphere_radius (p : RegularTriangularPyramid) : ℝ :=
  -- Definition to be proved
  sorry

/-- Theorem: The radius of the circumscribed sphere of a regular triangular pyramid
    with base edge 6 and side edge 4 is 4 -/
theorem circumscribed_sphere_radius_for_specific_pyramid :
  let p : RegularTriangularPyramid := ⟨6, 4⟩
  circumscribed_sphere_radius p = 4 := by
  sorry

end circumscribed_sphere_radius_for_specific_pyramid_l3803_380360


namespace hundredth_odd_integer_and_divisibility_l3803_380372

theorem hundredth_odd_integer_and_divisibility :
  (∃ n : ℕ, n = 100 ∧ 2 * n - 1 = 199) ∧ ¬(199 % 5 = 0) := by
  sorry

end hundredth_odd_integer_and_divisibility_l3803_380372


namespace solution_product_l3803_380376

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 11) = r^2 - 16 * r + 52 →
  (s - 3) * (3 * s + 11) = s^2 - 16 * s + 52 →
  r ≠ s →
  (r + 4) * (s + 4) = -62.5 := by
sorry

end solution_product_l3803_380376


namespace sum_of_fractions_equals_five_plus_sqrt_two_l3803_380362

theorem sum_of_fractions_equals_five_plus_sqrt_two :
  let S := 1 / (5 - Real.sqrt 19) - 1 / (Real.sqrt 19 - Real.sqrt 18) + 
           1 / (Real.sqrt 18 - Real.sqrt 17) - 1 / (Real.sqrt 17 - 3) + 
           1 / (3 - Real.sqrt 2)
  S = 5 + Real.sqrt 2 := by
  sorry

end sum_of_fractions_equals_five_plus_sqrt_two_l3803_380362


namespace ricks_ironing_rate_l3803_380347

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := sorry

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- The number of hours Rick spent ironing dress shirts -/
def shirt_hours : ℕ := 3

/-- The number of hours Rick spent ironing dress pants -/
def pant_hours : ℕ := 5

/-- The total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

theorem ricks_ironing_rate :
  shirts_per_hour * shirt_hours + pants_per_hour * pant_hours = total_pieces ∧
  pants_per_hour = 3 :=
sorry

end ricks_ironing_rate_l3803_380347


namespace quadrilateral_angle_measure_l3803_380396

theorem quadrilateral_angle_measure (E F G H : ℝ) : 
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H ∧ E + F + G + H = 360 → E = 540 / 7 := by
  sorry

end quadrilateral_angle_measure_l3803_380396


namespace apples_given_theorem_l3803_380351

/-- The number of apples Joan gave to Melanie -/
def apples_given_to_melanie (initial_apples current_apples : ℕ) : ℕ :=
  initial_apples - current_apples

/-- Proof that the number of apples given to Melanie is correct -/
theorem apples_given_theorem (initial_apples current_apples : ℕ) 
  (h : initial_apples ≥ current_apples) :
  apples_given_to_melanie initial_apples current_apples = initial_apples - current_apples :=
by
  sorry

/-- Verifying the specific case in the problem -/
example : apples_given_to_melanie 43 16 = 27 :=
by
  sorry

end apples_given_theorem_l3803_380351


namespace price_decrease_units_sold_ratio_l3803_380340

theorem price_decrease_units_sold_ratio (P U : ℝ) (h : P > 0) (k : U > 0) :
  let new_price := 0.25 * P
  let new_units := U / 0.25
  let revenue_unchanged := P * U = new_price * new_units
  let percent_decrease_price := 75
  let percent_increase_units := (new_units - U) / U * 100
  revenue_unchanged →
  percent_increase_units / percent_decrease_price = 4 := by
sorry

end price_decrease_units_sold_ratio_l3803_380340


namespace product_of_three_integers_l3803_380304

theorem product_of_three_integers (A B C : Int) : 
  A < B → B < C → A + B + C = 33 → C = 3 * B → A = C - 23 → A * B * C = 192 := by
  sorry

end product_of_three_integers_l3803_380304


namespace functional_equation_solution_l3803_380366

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)

/-- The theorem stating that any function satisfying the functional equation
    must be one of the three specified functions -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end functional_equation_solution_l3803_380366


namespace triangle_problem_l3803_380310

open Real

theorem triangle_problem (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  sin C * sin (A - B) = sin B * sin (C - A) ∧
  A = 2 * B →
  C = 5 * π / 8 ∧ 2 * a^2 = b^2 + c^2 :=
sorry

end triangle_problem_l3803_380310


namespace sum_of_multiples_l3803_380319

def largest_two_digit_multiple_of_5 : ℕ :=
  95

def smallest_three_digit_multiple_of_7 : ℕ :=
  105

theorem sum_of_multiples : 
  largest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 200 := by
  sorry

end sum_of_multiples_l3803_380319


namespace complex_expression_equals_one_l3803_380364

theorem complex_expression_equals_one 
  (x y : ℂ) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0) 
  (h_equation : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2005 + (y / (x + y))^2005 = 1 := by
  sorry

end complex_expression_equals_one_l3803_380364


namespace greyson_payment_l3803_380314

/-- The number of dimes in a dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 5

/-- The number of dimes Greyson paid for the watch -/
def dimes_paid : ℕ := watch_cost * dimes_per_dollar

theorem greyson_payment : dimes_paid = 50 := by
  sorry

end greyson_payment_l3803_380314


namespace distance_A_B_l3803_380342

/-- The distance between points A(0, 0, 1) and B(0, 1, 0) in a spatial Cartesian coordinate system is √2. -/
theorem distance_A_B : Real.sqrt 2 = (Real.sqrt ((0 - 0)^2 + (1 - 0)^2 + (0 - 1)^2)) := by
  sorry

end distance_A_B_l3803_380342


namespace factor_proof_l3803_380331

theorem factor_proof (x y z : ℝ) : 
  ∃ (k : ℝ), x^2 - y^2 - z^2 + 2*y*z + x - y - z + 2 = (x - y + z + 1) * k :=
by sorry

end factor_proof_l3803_380331


namespace sugar_for_partial_recipe_result_as_mixed_number_l3803_380313

-- Define the original amount of sugar in the recipe
def original_sugar : ℚ := 19/3

-- Define the fraction of the recipe we want to make
def recipe_fraction : ℚ := 1/3

-- Theorem statement
theorem sugar_for_partial_recipe :
  recipe_fraction * original_sugar = 19/9 :=
by sorry

-- Convert the result to a mixed number
theorem result_as_mixed_number :
  ∃ (whole : ℕ) (num denom : ℕ), 
    recipe_fraction * original_sugar = whole + num / denom ∧
    whole = 2 ∧ num = 1 ∧ denom = 9 :=
by sorry

end sugar_for_partial_recipe_result_as_mixed_number_l3803_380313


namespace danivan_drugstore_inventory_l3803_380358

/-- Calculates the remaining inventory of sanitizer gel at Danivan Drugstore -/
def remaining_inventory (initial_inventory : ℕ) 
  (daily_sales : List ℕ) (supplier_deliveries : List ℕ) : ℕ :=
  initial_inventory - (daily_sales.sum) + (supplier_deliveries.sum)

theorem danivan_drugstore_inventory : 
  let initial_inventory : ℕ := 4500
  let daily_sales : List ℕ := [2445, 906, 215, 457, 312, 239, 188]
  let supplier_deliveries : List ℕ := [350, 750, 981]
  remaining_inventory initial_inventory daily_sales supplier_deliveries = 819 := by
  sorry

end danivan_drugstore_inventory_l3803_380358


namespace greatest_integer_difference_l3803_380374

theorem greatest_integer_difference (x y : ℚ) 
  (hx : 3 < x) (hxy : x < (3/2)^3) (hyz : (3/2)^3 < y) (hy : y < 7) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (m : ℚ) ≤ y - x → m ≤ n :=
sorry

end greatest_integer_difference_l3803_380374


namespace expand_polynomial_l3803_380353

theorem expand_polynomial (x : ℝ) : (5*x^2 + 3*x - 4) * 3*x^3 = 15*x^5 + 9*x^4 - 12*x^3 := by
  sorry

end expand_polynomial_l3803_380353


namespace complex_fraction_equality_l3803_380316

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (1 + Complex.I) ^ 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end complex_fraction_equality_l3803_380316


namespace focus_coordinates_l3803_380320

-- Define the ellipse type
structure Ellipse where
  major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)
  minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)

-- Define the function to find the focus with lesser y-coordinate
def focus_with_lesser_y (e : Ellipse) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem focus_coordinates (e : Ellipse) :
  e.major_axis_endpoints = ((1, -2), (7, -2)) →
  e.minor_axis_endpoints = ((4, 1), (4, -5)) →
  focus_with_lesser_y e = (4, -2) :=
sorry

end focus_coordinates_l3803_380320


namespace largest_common_divisor_420_385_l3803_380329

def largest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem largest_common_divisor_420_385 :
  largest_common_divisor 420 385 = 35 := by
  sorry

end largest_common_divisor_420_385_l3803_380329


namespace fraction_above_line_is_seven_tenths_l3803_380322

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the 2D plane defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the fraction of a square's area above a given line -/
def fractionAboveLine (s : Square) (l : Line) : ℝ :=
  sorry

/-- The main theorem stating that the fraction of the square's area above the specified line is 7/10 -/
theorem fraction_above_line_is_seven_tenths :
  let s : Square := { bottomLeft := (2, 0), topRight := (7, 5) }
  let l : Line := { point1 := (2, 1), point2 := (7, 3) }
  fractionAboveLine s l = 7/10 := by sorry

end fraction_above_line_is_seven_tenths_l3803_380322


namespace growth_rate_inequality_l3803_380385

theorem growth_rate_inequality (p q x : ℝ) (h : p ≠ q) :
  (1 + x)^2 = (1 + p) * (1 + q) → x < (p + q) / 2 := by
  sorry

end growth_rate_inequality_l3803_380385


namespace smallest_n_for_exact_tax_l3803_380380

theorem smallest_n_for_exact_tax : ∃ (x : ℕ+), (↑x : ℚ) * 105 / 100 = 21 ∧ 
  ∀ (n : ℕ+), n < 21 → ¬∃ (y : ℕ+), (↑y : ℚ) * 105 / 100 = ↑n :=
sorry

end smallest_n_for_exact_tax_l3803_380380


namespace cubic_sum_l3803_380373

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l3803_380373


namespace total_water_intake_l3803_380336

/-- Calculates the total water intake throughout the day given specific drinking patterns --/
theorem total_water_intake (morning : ℝ) : morning = 1.5 →
  let early_afternoon := 2 * morning
  let late_afternoon := 3 * morning
  let evening := late_afternoon * (1 - 0.25)
  let night := 2 * evening
  morning + early_afternoon + late_afternoon + evening + night = 19.125 := by
  sorry

end total_water_intake_l3803_380336


namespace conditional_probability_l3803_380332

/-- The total number of products in the box -/
def total_products : ℕ := 4

/-- The number of first-class products in the box -/
def first_class_products : ℕ := 3

/-- The number of second-class products in the box -/
def second_class_products : ℕ := 1

/-- Event A: "the first draw is a first-class product" -/
def event_A : Set ℕ := {1, 2, 3}

/-- Event B: "the second draw is a first-class product" -/
def event_B : Set ℕ := {1, 2}

/-- The probability of event A -/
def prob_A : ℚ := first_class_products / total_products

/-- The probability of event B given event A has occurred -/
def prob_B_given_A : ℚ := (first_class_products - 1) / (total_products - 1)

/-- The conditional probability of event B given event A -/
theorem conditional_probability :
  prob_B_given_A = 2/3 :=
sorry

end conditional_probability_l3803_380332


namespace permutations_formula_l3803_380381

-- Define the number of permutations
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem permutations_formula {n k : ℕ} (h : 1 ≤ k ∧ k ≤ n) :
  permutations n k = (Nat.factorial n) / (Nat.factorial (n - k)) := by
  sorry

end permutations_formula_l3803_380381


namespace M_intersect_N_l3803_380357

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 = x}

theorem M_intersect_N : M ∩ N = {0, 1} := by
  sorry

end M_intersect_N_l3803_380357


namespace water_depth_in_cylinder_l3803_380352

/-- Represents the depth of water in a horizontal cylindrical tank. -/
def water_depth (tank_length tank_diameter water_surface_area : ℝ) : Set ℝ :=
  {h : ℝ | ∃ (w : ℝ), 
    tank_length > 0 ∧ 
    tank_diameter > 0 ∧ 
    water_surface_area > 0 ∧
    w > 0 ∧ 
    h > 0 ∧ 
    h < tank_diameter ∧
    w * tank_length = water_surface_area ∧
    w = 2 * Real.sqrt (tank_diameter * h - h^2)}

/-- The main theorem stating the depth of water in the given cylindrical tank. -/
theorem water_depth_in_cylinder : 
  water_depth 12 4 24 = {2 - Real.sqrt 3, 2 + Real.sqrt 3} := by
  sorry


end water_depth_in_cylinder_l3803_380352


namespace three_isosceles_triangles_l3803_380399

-- Define a point on the grid
structure Point where
  x : ℕ
  y : ℕ

-- Define a triangle on the grid
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d12 := (t.v1.x - t.v2.x)^2 + (t.v1.y - t.v2.y)^2
  let d23 := (t.v2.x - t.v3.x)^2 + (t.v2.y - t.v3.y)^2
  let d31 := (t.v3.x - t.v1.x)^2 + (t.v3.y - t.v1.y)^2
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

-- Define the five triangles
def triangle1 : Triangle := { v1 := {x := 0, y := 7}, v2 := {x := 2, y := 7}, v3 := {x := 1, y := 5} }
def triangle2 : Triangle := { v1 := {x := 4, y := 3}, v2 := {x := 4, y := 5}, v3 := {x := 6, y := 3} }
def triangle3 : Triangle := { v1 := {x := 0, y := 2}, v2 := {x := 3, y := 3}, v3 := {x := 6, y := 2} }
def triangle4 : Triangle := { v1 := {x := 1, y := 1}, v2 := {x := 0, y := 3}, v3 := {x := 3, y := 1} }
def triangle5 : Triangle := { v1 := {x := 3, y := 6}, v2 := {x := 4, y := 4}, v3 := {x := 5, y := 7} }

-- Theorem statement
theorem three_isosceles_triangles :
  (isIsosceles triangle1) ∧
  (isIsosceles triangle2) ∧
  (isIsosceles triangle3) ∧
  ¬(isIsosceles triangle4) ∧
  ¬(isIsosceles triangle5) := by
  sorry

end three_isosceles_triangles_l3803_380399


namespace evaluate_F_of_f_l3803_380355

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 + 1
def F (a b : ℝ) : ℝ := a * b + b^2

-- State the theorem
theorem evaluate_F_of_f : F 4 (f 3) = 140 := by
  sorry

end evaluate_F_of_f_l3803_380355
