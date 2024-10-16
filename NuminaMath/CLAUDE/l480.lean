import Mathlib

namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l480_48066

/-- Represents a square tile pattern -/
structure TilePattern :=
  (side : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- The initial square pattern -/
def initial_pattern : TilePattern :=
  { side := 5
  , black_tiles := 8
  , white_tiles := 17 }

/-- Extends a tile pattern by adding a black border -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2
  , black_tiles := p.black_tiles + 2 * p.side + 2 * (p.side + 2)
  , white_tiles := p.white_tiles }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern) : 
  p = initial_pattern → 
  (extend_pattern p).black_tiles = 32 ∧ (extend_pattern p).white_tiles = 17 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l480_48066


namespace NUMINAMATH_CALUDE_M_equiv_NotFirstOrThirdQuadrant_l480_48009

/-- The set M of points (x,y) in ℝ² where xy ≤ 0 -/
def M : Set (ℝ × ℝ) := {p | p.1 * p.2 ≤ 0}

/-- The set of points not in the first or third quadrants of ℝ² -/
def NotFirstOrThirdQuadrant : Set (ℝ × ℝ) := 
  {p | p.1 * p.2 ≤ 0}

/-- Theorem stating that M is equivalent to the set of points not in the first or third quadrants -/
theorem M_equiv_NotFirstOrThirdQuadrant : M = NotFirstOrThirdQuadrant := by
  sorry


end NUMINAMATH_CALUDE_M_equiv_NotFirstOrThirdQuadrant_l480_48009


namespace NUMINAMATH_CALUDE_not_perfect_square_product_l480_48093

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m

/-- The main theorem stating that 1, 2, and 4 are the only positive integers
    for which n(n+a) is not a perfect square for all positive integers n -/
theorem not_perfect_square_product (a : ℕ) : a > 0 →
  (∀ n : ℕ, n > 0 → ¬is_perfect_square (n * (n + a))) ↔ a = 1 ∨ a = 2 ∨ a = 4 :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_product_l480_48093


namespace NUMINAMATH_CALUDE_set_operation_result_l480_48076

def A : Set ℕ := {0, 1, 2, 4, 5, 7, 8}
def B : Set ℕ := {1, 3, 6, 7, 9}
def C : Set ℕ := {3, 4, 7, 8}

theorem set_operation_result : (A ∩ B) ∪ C = {1, 3, 4, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l480_48076


namespace NUMINAMATH_CALUDE_hyperbola_eq_theorem_l480_48017

/-- A hyperbola with the given properties -/
structure Hyperbola where
  -- The hyperbola is centered at the origin
  center_origin : True
  -- The foci are on the coordinate axes
  foci_on_axes : True
  -- One of the asymptotes has the equation y = (1/2)x
  asymptote_eq : ∀ x y : ℝ, y = (1/2) * x
  -- The point (2, 2) lies on the hyperbola
  point_on_hyperbola : True

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (y^2 / 3) - (x^2 / 12) = 1

/-- Theorem stating the equation of the hyperbola with the given properties -/
theorem hyperbola_eq_theorem (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eq_theorem_l480_48017


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l480_48061

open Set
open Function
open Real

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : f 1 = 4) (h3 : ∀ x, deriv f x < 3) :
  {x : ℝ | f (Real.log x) > 3 * Real.log x + 1} = Ioo 0 (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l480_48061


namespace NUMINAMATH_CALUDE_right_triangle_area_l480_48033

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) : 
  hypotenuse = 10 * Real.sqrt 2 →
  angle = 45 * (π / 180) →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 50 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l480_48033


namespace NUMINAMATH_CALUDE_Q_equals_sum_l480_48091

/-- Binomial coefficient -/
def binomial (a b : ℕ) : ℕ :=
  if a ≥ b then
    Nat.factorial a / (Nat.factorial b * Nat.factorial (a - b))
  else
    0

/-- Coefficient of x^k in (1+x+x^2+x^3)^n -/
def Q (n k : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (fun j => binomial n j * binomial n (k - 2 * j))

/-- The main theorem -/
theorem Q_equals_sum (n k : ℕ) :
    Q n k = (Finset.range (n + 1)).sum (fun j => binomial n j * binomial n (k - 2 * j)) := by
  sorry

end NUMINAMATH_CALUDE_Q_equals_sum_l480_48091


namespace NUMINAMATH_CALUDE_bryan_total_books_l480_48096

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 9

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 56

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that the total number of books Bryan has is 504 -/
theorem bryan_total_books : total_books = 504 := by sorry

end NUMINAMATH_CALUDE_bryan_total_books_l480_48096


namespace NUMINAMATH_CALUDE_simplify_expression_l480_48002

theorem simplify_expression (y : ℝ) : 3 * y - 5 * y^2 + 12 - (7 - 3 * y + 5 * y^2) = -10 * y^2 + 6 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l480_48002


namespace NUMINAMATH_CALUDE_cube_of_three_fifths_l480_48095

theorem cube_of_three_fifths : (3 / 5 : ℚ) ^ 3 = 27 / 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_three_fifths_l480_48095


namespace NUMINAMATH_CALUDE_election_vote_count_l480_48070

theorem election_vote_count (votes : List Nat) : 
  votes = [195, 142, 116, 90] →
  votes.length = 4 →
  votes[0]! = 195 →
  votes[0]! - votes[1]! = 53 →
  votes[0]! - votes[2]! = 79 →
  votes[0]! - votes[3]! = 105 →
  votes.sum = 543 := by
sorry

end NUMINAMATH_CALUDE_election_vote_count_l480_48070


namespace NUMINAMATH_CALUDE_min_value_of_abs_sum_l480_48008

theorem min_value_of_abs_sum (x : ℝ) : 
  |x - 4| + |x - 6| ≥ 2 ∧ ∃ y : ℝ, |y - 4| + |y - 6| = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_abs_sum_l480_48008


namespace NUMINAMATH_CALUDE_alarm_system_probability_l480_48081

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) : 
  1 - (1 - p)^2 = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_alarm_system_probability_l480_48081


namespace NUMINAMATH_CALUDE_mrs_martin_bagels_l480_48098

/-- The cost of one bagel in dollars -/
def bagel_cost : ℚ := 3/2

/-- Mrs. Martin's purchase -/
def mrs_martin_purchase (coffee_cost bagels : ℚ) : Prop :=
  3 * coffee_cost + bagels * bagel_cost = 51/4

/-- Mr. Martin's purchase -/
def mr_martin_purchase (coffee_cost : ℚ) : Prop :=
  2 * coffee_cost + 5 * bagel_cost = 14

theorem mrs_martin_bagels :
  ∃ (coffee_cost : ℚ), mr_martin_purchase coffee_cost →
    mrs_martin_purchase coffee_cost 2 := by sorry

end NUMINAMATH_CALUDE_mrs_martin_bagels_l480_48098


namespace NUMINAMATH_CALUDE_units_digit_sum_base9_l480_48088

-- Define a function to convert a base-9 number to base-10
def base9ToBase10 (n : ℕ) : ℕ := 
  (n / 10) * 9 + (n % 10)

-- Define a function to get the units digit in base-9
def unitsDigitBase9 (n : ℕ) : ℕ := 
  n % 9

-- Theorem statement
theorem units_digit_sum_base9 :
  unitsDigitBase9 (base9ToBase10 35 + base9ToBase10 47) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base9_l480_48088


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_solution_l480_48094

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h : principal > 0) (h2 : time > 0) :
  interest = principal * (25 : ℝ) / 100 * time →
  25 = (interest * 100) / (principal * time) :=
by
  sorry

/-- Specific problem instance -/
theorem problem_solution :
  let principal : ℝ := 800
  let time : ℝ := 2
  let interest : ℝ := 400
  (interest * 100) / (principal * time) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_solution_l480_48094


namespace NUMINAMATH_CALUDE_problem_statement_l480_48021

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1)
  (h2 : x + 1 / z = 5)
  (h3 : y + 1 / x = 29) :
  z + 1 / y = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l480_48021


namespace NUMINAMATH_CALUDE_central_angle_unchanged_l480_48054

/-- Theorem: When a circle's radius is doubled and the arc length is doubled, the central angle of the sector remains unchanged. -/
theorem central_angle_unchanged 
  (r : ℝ) 
  (l : ℝ) 
  (h_positive_r : r > 0) 
  (h_positive_l : l > 0) : 
  (l / r) = ((2 * l) / (2 * r)) := by 
sorry

end NUMINAMATH_CALUDE_central_angle_unchanged_l480_48054


namespace NUMINAMATH_CALUDE_joint_savings_theorem_l480_48073

/-- Calculates the total amount in a joint savings account given two people's earnings and savings rates. -/
def joint_savings (kimmie_earnings : ℝ) (zahra_reduction_rate : ℝ) (savings_rate : ℝ) : ℝ :=
  let zahra_earnings := kimmie_earnings * (1 - zahra_reduction_rate)
  let kimmie_savings := kimmie_earnings * savings_rate
  let zahra_savings := zahra_earnings * savings_rate
  kimmie_savings + zahra_savings

/-- Theorem stating that under given conditions, the joint savings amount to $375. -/
theorem joint_savings_theorem :
  joint_savings 450 (1/3) (1/2) = 375 := by
  sorry

end NUMINAMATH_CALUDE_joint_savings_theorem_l480_48073


namespace NUMINAMATH_CALUDE_partial_fraction_product_l480_48053

/-- Given a rational function and its partial fraction decomposition, prove that the product of the numerator coefficients is zero. -/
theorem partial_fraction_product (x : ℝ) (A B C : ℝ) : 
  (x^2 - 25) / (x^3 - x^2 - 7*x + 15) = A / (x - 3) + B / (x + 3) + C / (x - 5) →
  A * B * C = 0 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_product_l480_48053


namespace NUMINAMATH_CALUDE_secant_radius_ratio_l480_48043

/-- Represents a circle with a secant line and two squares inside --/
structure SecantSquaresCircle where
  /-- Radius of the circle --/
  radius : ℝ
  /-- Length of the secant line --/
  secant_length : ℝ
  /-- Side length of the smaller square --/
  small_square_side : ℝ
  /-- Side length of the larger square --/
  large_square_side : ℝ
  /-- The squares have two corners on the secant line and two on the circumference --/
  squares_position : Bool
  /-- The ratio of the square's side lengths is 5:9 --/
  side_ratio : small_square_side / large_square_side = 5 / 9

/-- The theorem stating the relationship between secant length and radius --/
theorem secant_radius_ratio (c : SecantSquaresCircle) :
  c.squares_position → c.secant_length / c.radius = 3 * Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_secant_radius_ratio_l480_48043


namespace NUMINAMATH_CALUDE_nell_initial_ace_cards_l480_48004

/-- Prove that Nell had 315 Ace cards initially -/
theorem nell_initial_ace_cards 
  (initial_baseball : ℕ)
  (final_ace : ℕ)
  (final_baseball : ℕ)
  (baseball_ace_difference : ℕ)
  (h1 : initial_baseball = 438)
  (h2 : final_ace = 55)
  (h3 : final_baseball = 178)
  (h4 : final_baseball = final_ace + baseball_ace_difference)
  (h5 : baseball_ace_difference = 123) :
  ∃ (initial_ace : ℕ), initial_ace = 315 ∧ 
    initial_ace - final_ace = initial_baseball - final_baseball :=
by
  sorry

end NUMINAMATH_CALUDE_nell_initial_ace_cards_l480_48004


namespace NUMINAMATH_CALUDE_wire_length_ratio_l480_48059

-- Define the given conditions
def bonnie_wire_length : ℝ := 12 * 8
def roark_cube_volume : ℝ := 2
def roark_edge_length : ℝ := 1.5
def bonnie_cube_volume : ℝ := 8^3

-- Define the theorem
theorem wire_length_ratio :
  let roark_cubes_count : ℝ := bonnie_cube_volume / roark_cube_volume
  let roark_total_wire_length : ℝ := roark_cubes_count * (12 * roark_edge_length)
  bonnie_wire_length / roark_total_wire_length = 1 / 48 := by
sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l480_48059


namespace NUMINAMATH_CALUDE_original_number_from_sum_l480_48089

/-- Represents a three-digit number in base 10 -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_ones : ones < 10
  h_not_zero : hundreds ≠ 0

/-- Calculates the sum of a three-digit number and its permutations -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  let a := n.hundreds
  let b := n.tens
  let c := n.ones
  222 * (a + b + c)

/-- The main theorem -/
theorem original_number_from_sum (N : Nat) (h_N : N = 3237) :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = N ∧ n.hundreds = 4 ∧ n.tens = 2 ∧ n.ones = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_from_sum_l480_48089


namespace NUMINAMATH_CALUDE_det_inequality_and_equality_l480_48037

open Complex Matrix

variable {n : ℕ}

theorem det_inequality_and_equality (A : Matrix (Fin n) (Fin n) ℂ) (a : ℂ) 
  (h : A - conjTranspose A = (2 * a) • 1) : 
  (Complex.abs (det A) ≥ Complex.abs a ^ n) ∧ 
  (Complex.abs (det A) = Complex.abs a ^ n → A = a • 1) := by
  sorry

end NUMINAMATH_CALUDE_det_inequality_and_equality_l480_48037


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l480_48014

theorem complex_fraction_evaluation :
  (Complex.I : ℂ) / (12 + Complex.I) = (1 : ℂ) / 145 + (12 : ℂ) / 145 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l480_48014


namespace NUMINAMATH_CALUDE_p_and_s_not_third_l480_48074

-- Define the set of runners
inductive Runner : Type
  | P | Q | R | S | T | U

-- Define the finishing order relation
def finishes_before (x y : Runner) : Prop := sorry

-- Define the race conditions
axiom p_beats_q : finishes_before Runner.P Runner.Q
axiom p_beats_r : finishes_before Runner.P Runner.R
axiom q_beats_s : finishes_before Runner.Q Runner.S
axiom t_between_p_and_q : finishes_before Runner.P Runner.T ∧ finishes_before Runner.T Runner.Q
axiom u_after_r_before_t : finishes_before Runner.R Runner.U ∧ finishes_before Runner.U Runner.T

-- Define what it means to finish third
def finishes_third (x : Runner) : Prop :=
  ∃ (a b : Runner), (a ≠ x ∧ b ≠ x ∧ a ≠ b) ∧
    finishes_before a x ∧ finishes_before b x ∧
    ∀ y : Runner, y ≠ x → y ≠ a → y ≠ b → finishes_before x y

-- Theorem to prove
theorem p_and_s_not_third :
  ¬(finishes_third Runner.P) ∧ ¬(finishes_third Runner.S) :=
sorry

end NUMINAMATH_CALUDE_p_and_s_not_third_l480_48074


namespace NUMINAMATH_CALUDE_percentage_of_women_employees_l480_48040

theorem percentage_of_women_employees (men_with_degree : ℝ) (men_without_degree : ℕ) (total_women : ℕ) : 
  men_with_degree = 0.75 * (men_with_degree + men_without_degree) →
  men_without_degree = 8 →
  total_women = 48 →
  (total_women : ℝ) / ((men_with_degree + men_without_degree : ℝ) + total_women) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_women_employees_l480_48040


namespace NUMINAMATH_CALUDE_num_plane_determining_pairs_eq_66_l480_48077

/-- A rectangular prism with distinct dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  distinct : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of edges in a rectangular prism -/
def num_edges : ℕ := 12

/-- The number of unordered pairs of parallel edges -/
def num_parallel_pairs : ℕ := 18

/-- The total number of unordered pairs of edges -/
def total_edge_pairs : ℕ := num_edges * (num_edges - 1) / 2

/-- The number of unordered pairs of edges that determine a plane -/
def num_plane_determining_pairs (prism : RectangularPrism) : ℕ :=
  total_edge_pairs

/-- Theorem: The number of unordered pairs of edges in a rectangular prism
    with distinct dimensions that determine a plane is 66 -/
theorem num_plane_determining_pairs_eq_66 (prism : RectangularPrism) :
  num_plane_determining_pairs prism = 66 := by
  sorry

end NUMINAMATH_CALUDE_num_plane_determining_pairs_eq_66_l480_48077


namespace NUMINAMATH_CALUDE_garden_fencing_theorem_l480_48071

/-- Calculates the perimeter of a rectangular garden with given length and width. -/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: A rectangular garden with length 60 yards and width equal to half its length
    requires 180 yards of fencing to enclose it. -/
theorem garden_fencing_theorem :
  let length : ℝ := 60
  let width : ℝ := length / 2
  garden_perimeter length width = 180 := by
sorry

end NUMINAMATH_CALUDE_garden_fencing_theorem_l480_48071


namespace NUMINAMATH_CALUDE_hiring_range_l480_48036

/-- The number of standard deviations that includes all accepted ages -/
def num_std_dev (avg : ℕ) (std_dev : ℕ) (num_ages : ℕ) : ℚ :=
  (num_ages - 1) / (2 * std_dev)

theorem hiring_range (avg : ℕ) (std_dev : ℕ) (num_ages : ℕ)
  (h_avg : avg = 20)
  (h_std_dev : std_dev = 8)
  (h_num_ages : num_ages = 17) :
  num_std_dev avg std_dev num_ages = 1 := by
sorry

end NUMINAMATH_CALUDE_hiring_range_l480_48036


namespace NUMINAMATH_CALUDE_second_discount_percentage_second_discount_is_25_percent_l480_48056

theorem second_discount_percentage 
  (original_price : ℝ) 
  (first_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_amount := price_after_first_discount - final_price
  let second_discount_percent := (second_discount_amount / price_after_first_discount) * 100
  second_discount_percent

theorem second_discount_is_25_percent :
  second_discount_percentage 33.78 25 19 = 25 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_second_discount_is_25_percent_l480_48056


namespace NUMINAMATH_CALUDE_complement_of_P_relative_to_U_l480_48078

def U : Set ℤ := {-1, 0, 1, 2, 3}
def P : Set ℤ := {-1, 2, 3}

theorem complement_of_P_relative_to_U :
  {x ∈ U | x ∉ P} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_relative_to_U_l480_48078


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l480_48079

/-- Given an ellipse with equation x²/a² + y²/4 = 1 and one focus at (2,0),
    prove that its eccentricity is √2/2 -/
theorem ellipse_eccentricity (a : ℝ) (h : a > 0) :
  let c := 2  -- distance from center to focus
  let b := 2  -- √4, as y²/4 = 1 in the equation
  let e := c / a  -- definition of eccentricity
  (∀ x y, x^2 / a^2 + y^2 / 4 = 1 → (x - c)^2 + y^2 = a^2) →  -- ellipse definition
  e = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l480_48079


namespace NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l480_48042

theorem no_real_solutions_for_sqrt_equation (b : ℝ) (h : b > 2) :
  ¬∃ (a x : ℝ), Real.sqrt (b - Real.cos (a + x)) = x := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l480_48042


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l480_48027

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1-i)z = 2i
def equation (z : ℂ) : Prop := (1 - i) * z = 2 * i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l480_48027


namespace NUMINAMATH_CALUDE_lcm_problem_l480_48058

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l480_48058


namespace NUMINAMATH_CALUDE_simplify_expression_l480_48034

theorem simplify_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  3 * x ^ Real.sqrt 2 * (2 * x ^ (-Real.sqrt 2) * y * z) = 6 * y * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l480_48034


namespace NUMINAMATH_CALUDE_distance_between_runners_l480_48025

-- Define the race length in kilometers
def race_length_km : ℝ := 1

-- Define Arianna's position in meters when Ethan finished
def arianna_position : ℝ := 184

-- Theorem to prove the distance between Ethan and Arianna
theorem distance_between_runners : 
  (race_length_km * 1000) - arianna_position = 816 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_runners_l480_48025


namespace NUMINAMATH_CALUDE_fold_reflection_l480_48085

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given three points A, B, and C on a coordinate grid, if A coincides with B after folding,
    then C will coincide with the reflection of C across the perpendicular bisector of AB. -/
theorem fold_reflection (A B C : Point) (h : A.x = 10 ∧ A.y = 0 ∧ B.x = -6 ∧ B.y = 8 ∧ C.x = -4 ∧ C.y = 2) :
  ∃ (P : Point), P.x = 4 ∧ P.y = -2 ∧ 
  (2 * (C.x + P.x) = 2 * ((A.x + B.x) / 2)) ∧
  (C.y + P.y = 2 * ((A.y + B.y) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_fold_reflection_l480_48085


namespace NUMINAMATH_CALUDE_remainder_98_power_50_mod_50_l480_48029

theorem remainder_98_power_50_mod_50 : 98^50 % 50 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_power_50_mod_50_l480_48029


namespace NUMINAMATH_CALUDE_arithmetic_expressions_evaluation_l480_48099

theorem arithmetic_expressions_evaluation :
  ((-12) - 5 + (-14) - (-39) = 8) ∧
  (-2^2 * 5 - (-12) / 4 - 4 = -21) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_evaluation_l480_48099


namespace NUMINAMATH_CALUDE_waiter_customers_problem_l480_48022

theorem waiter_customers_problem :
  ∃ x : ℝ, x > 0 ∧ ((x - 19.0) - 14.0 = 3) → x = 36.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_problem_l480_48022


namespace NUMINAMATH_CALUDE_unique_persistent_number_l480_48012

/-- Definition of a persistent number -/
def isPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 → a ≠ 1 → b ≠ 0 → b ≠ 1 → c ≠ 0 → c ≠ 1 → d ≠ 0 → d ≠ 1 →
    (a + b + c + d = T ∧ 1/a + 1/b + 1/c + 1/d = T) →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

/-- Theorem: There exists a unique persistent number, and it equals 2 -/
theorem unique_persistent_number :
  ∃! T : ℝ, isPersistent T ∧ T = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_persistent_number_l480_48012


namespace NUMINAMATH_CALUDE_part_one_part_two_l480_48024

-- Part 1
theorem part_one (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x > 0, f x = x - a * Real.log x)
  (h1 : ∀ x > 0, f x ≥ 1) : a = 1 := by
  sorry

-- Part 2
theorem part_two (x₁ x₂ : ℝ) 
  (h1 : x₁ > 0)
  (h2 : x₂ > 0)
  (h3 : Real.exp x₁ + Real.log x₂ > x₁ + x₂) :
  Real.exp x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l480_48024


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l480_48001

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the focal length -/
def focal_length (c : ℝ) : Prop :=
  c = 2

/-- Definition of a point on the ellipse -/
def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse 2 (-Real.sqrt 2) a b

/-- Definition of the line intersecting the ellipse -/
def intersecting_line (x y m : ℝ) : Prop :=
  y = x + m

/-- Definition of the circle where the midpoint lies -/
def midpoint_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Main theorem -/
theorem ellipse_intersection_theorem (a b c m : ℝ) : 
  a > b ∧ b > 0 ∧
  focal_length c ∧
  point_on_ellipse a b →
  (∀ x y, ellipse x y a b ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    ellipse A.1 A.2 a b ∧
    ellipse B.1 B.2 a b ∧
    intersecting_line A.1 A.2 m ∧
    intersecting_line B.1 B.2 m ∧
    midpoint_circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) →
    m = 3 * Real.sqrt 5 / 5 ∨ m = -3 * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l480_48001


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l480_48082

theorem quadratic_rewrite (x : ℝ) :
  ∃ m : ℝ, 4 * x^2 - 16 * x - 448 = (x + m)^2 - 116 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l480_48082


namespace NUMINAMATH_CALUDE_james_old_wage_l480_48057

/-- Jame's old hourly wage -/
def old_wage : ℝ := 16

/-- Jame's new hourly wage -/
def new_wage : ℝ := 20

/-- Jame's old weekly work hours -/
def old_hours : ℝ := 25

/-- Jame's new weekly work hours -/
def new_hours : ℝ := 40

/-- Number of weeks worked per year -/
def weeks_per_year : ℝ := 52

/-- Difference in annual earnings between new and old job -/
def annual_difference : ℝ := 20800

theorem james_old_wage :
  old_wage * old_hours * weeks_per_year + annual_difference = new_wage * new_hours * weeks_per_year :=
by sorry

end NUMINAMATH_CALUDE_james_old_wage_l480_48057


namespace NUMINAMATH_CALUDE_expression_non_negative_l480_48007

theorem expression_non_negative (x y : ℝ) : 
  5 * x^2 + 5 * y^2 + 8 * x * y + 2 * y - 2 * x + 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_non_negative_l480_48007


namespace NUMINAMATH_CALUDE_magnitude_of_difference_l480_48062

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, 6]

theorem magnitude_of_difference (x : ℝ) :
  (vector_a 0 / vector_b x 0 = vector_a 1 / vector_b x 1) →
  Real.sqrt ((vector_a 0 - vector_b x 0)^2 + (vector_a 1 - vector_b x 1)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_difference_l480_48062


namespace NUMINAMATH_CALUDE_mom_bought_packages_l480_48020

def shirts_per_package : ℕ := 6
def total_shirts : ℕ := 426

theorem mom_bought_packages : 
  ∃ (packages : ℕ), packages * shirts_per_package = total_shirts ∧ packages = 71 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_packages_l480_48020


namespace NUMINAMATH_CALUDE_eighth_root_of_256289062500_l480_48048

theorem eighth_root_of_256289062500 : (256289062500 : ℝ) ^ (1/8 : ℝ) = 52 := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_of_256289062500_l480_48048


namespace NUMINAMATH_CALUDE_min_distance_Q_to_C_l480_48038

noncomputable def A : ℝ × ℝ := (-1, 2)
noncomputable def B : ℝ × ℝ := (0, 1)

def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 4}

def l₁ : Set (ℝ × ℝ) := {q : ℝ × ℝ | 3 * q.1 - 4 * q.2 + 12 = 0}

theorem min_distance_Q_to_C :
  ∀ Q ∈ l₁, ∃ M ∈ C, ∀ M' ∈ C, dist Q M ≤ dist Q M' ∧ dist Q M ≥ Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_min_distance_Q_to_C_l480_48038


namespace NUMINAMATH_CALUDE_expression_values_l480_48016

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 := by
sorry

end NUMINAMATH_CALUDE_expression_values_l480_48016


namespace NUMINAMATH_CALUDE_constant_magnitude_l480_48063

theorem constant_magnitude (z₁ z₂ : ℂ) (h₁ : Complex.abs z₁ = 5) 
  (h₂ : ∀ θ : ℝ, z₁^2 - z₁ * z₂ * Complex.sin θ + z₂^2 = 0) : 
  Complex.abs z₂ = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_magnitude_l480_48063


namespace NUMINAMATH_CALUDE_win_sector_area_l480_48006

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_win_sector_area_l480_48006


namespace NUMINAMATH_CALUDE_tan_arctan_five_twelfths_l480_48013

theorem tan_arctan_five_twelfths : 
  Real.tan (Real.arctan (5 / 12)) = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_tan_arctan_five_twelfths_l480_48013


namespace NUMINAMATH_CALUDE_triangle_theorem_l480_48023

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 2 * t.b * Real.sin t.C + t.a * Real.sin t.A = t.b * Real.sin t.B + t.c * Real.sin t.C

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h1 : condition t) (h2 : t.a = Real.sqrt 2) :
  t.A = π / 4 ∧ 
  ∀ (AD : ℝ), AD ≤ 1 + Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l480_48023


namespace NUMINAMATH_CALUDE_hypotenuse_squared_of_complex_zeros_l480_48080

/-- Given complex numbers u, v, and w that are zeros of a cubic polynomial
    and form a right triangle in the complex plane, if the sum of their
    squared magnitudes is 400, then the square of the hypotenuse of the
    triangle is 720. -/
theorem hypotenuse_squared_of_complex_zeros (u v w : ℂ) (s t : ℂ) :
  (u^3 + s*u + t = 0) →
  (v^3 + s*v + t = 0) →
  (w^3 + s*w + t = 0) →
  (Complex.abs u)^2 + (Complex.abs v)^2 + (Complex.abs w)^2 = 400 →
  ∃ (a b : ℝ), a^2 + b^2 = (Complex.abs (u - v))^2 ∧
                a^2 + b^2 = (Complex.abs (v - w))^2 ∧
                a * b = (Complex.abs (u - w))^2 →
  (Complex.abs (u - w))^2 = 720 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_squared_of_complex_zeros_l480_48080


namespace NUMINAMATH_CALUDE_log_xy_equals_three_fourths_l480_48047

-- Define x and y as positive real numbers
variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Define the given conditions
def condition1 : Prop := Real.log (x^2 * y^4) = 2
def condition2 : Prop := Real.log (x^3 * y^2) = 2

-- State the theorem
theorem log_xy_equals_three_fourths 
  (h1 : condition1 x y) (h2 : condition2 x y) : 
  Real.log (x * y) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_equals_three_fourths_l480_48047


namespace NUMINAMATH_CALUDE_equation_solution_l480_48010

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  x > 0 ∧ Real.sqrt ((log10 x)^2 + log10 (x^2) + 1) + log10 x + 1 = 0

-- Theorem statement
theorem equation_solution :
  ∀ x : ℝ, equation x ↔ (0 < x ∧ x ≤ (1/10)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l480_48010


namespace NUMINAMATH_CALUDE_domino_set_size_l480_48075

theorem domino_set_size (num_players : ℕ) (dominoes_per_player : ℕ) 
  (h1 : num_players = 4) 
  (h2 : dominoes_per_player = 7) : 
  num_players * dominoes_per_player = 28 := by
  sorry

end NUMINAMATH_CALUDE_domino_set_size_l480_48075


namespace NUMINAMATH_CALUDE_equation_solution_l480_48064

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ 0 ∧ x ≠ 1) ∧ ((x - 1) / x + 3 * x / (x - 1) = 4) ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l480_48064


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l480_48031

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 2) 
  (h_a7 : a 7 = 10) : 
  ∀ n : ℕ, a n = 2 * n - 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l480_48031


namespace NUMINAMATH_CALUDE_cannot_form_square_l480_48044

/-- Represents the number of sticks of each length --/
structure Sticks :=
  (length1 : ℕ)
  (length2 : ℕ)
  (length3 : ℕ)
  (length4 : ℕ)

/-- Calculates the total length of all sticks --/
def totalLength (s : Sticks) : ℕ :=
  s.length1 * 1 + s.length2 * 2 + s.length3 * 3 + s.length4 * 4

/-- Represents the given set of sticks --/
def givenSticks : Sticks :=
  { length1 := 6
  , length2 := 3
  , length3 := 6
  , length4 := 5 }

/-- Theorem stating that it's impossible to form a square with the given sticks --/
theorem cannot_form_square (s : Sticks) (h : s = givenSticks) :
  ¬ ∃ (side : ℕ), side > 0 ∧ 4 * side = totalLength s :=
by sorry


end NUMINAMATH_CALUDE_cannot_form_square_l480_48044


namespace NUMINAMATH_CALUDE_M_is_solution_set_inequality_holds_l480_48039

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- Statement 1: M is the solution set for f(x) < |2x+1| - 1
theorem M_is_solution_set : ∀ x : ℝ, x ∈ M ↔ f x < |2*x + 1| - 1 :=
sorry

-- Statement 2: For any a, b ∈ M, f(ab) > f(a) - f(-b)
theorem inequality_holds : ∀ a b : ℝ, a ∈ M → b ∈ M → f (a*b) > f a - f (-b) :=
sorry

end NUMINAMATH_CALUDE_M_is_solution_set_inequality_holds_l480_48039


namespace NUMINAMATH_CALUDE_remainder_eleven_power_2023_mod_13_l480_48055

theorem remainder_eleven_power_2023_mod_13 : 11^2023 % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eleven_power_2023_mod_13_l480_48055


namespace NUMINAMATH_CALUDE_bacteria_growth_l480_48018

/-- The time in minutes for the bacteria population to double -/
def doubling_time : ℝ := 6

/-- The initial population of bacteria -/
def initial_population : ℕ := 1000

/-- The total time of growth in minutes -/
def total_time : ℝ := 53.794705707972525

/-- The final population of bacteria after the given time -/
def final_population : ℕ := 495451

theorem bacteria_growth :
  let num_doublings : ℝ := total_time / doubling_time
  let theoretical_population : ℝ := initial_population * (2 ^ num_doublings)
  ⌊theoretical_population⌋ = final_population :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_l480_48018


namespace NUMINAMATH_CALUDE_melanies_dimes_l480_48051

theorem melanies_dimes (initial_dimes : ℕ) (mother_dimes : ℕ) (total_dimes : ℕ) (dad_dimes : ℕ) :
  initial_dimes = 7 →
  mother_dimes = 4 →
  total_dimes = 19 →
  total_dimes = initial_dimes + mother_dimes + dad_dimes →
  dad_dimes = 8 := by
sorry

end NUMINAMATH_CALUDE_melanies_dimes_l480_48051


namespace NUMINAMATH_CALUDE_square_root_sum_equals_six_l480_48072

theorem square_root_sum_equals_six :
  Real.sqrt ((3 - 2 * Real.sqrt 3) ^ 2) + Real.sqrt ((3 + 2 * Real.sqrt 3) ^ 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_six_l480_48072


namespace NUMINAMATH_CALUDE_square_sum_equals_29_l480_48032

theorem square_sum_equals_29 (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : 
  a^2 + b^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_29_l480_48032


namespace NUMINAMATH_CALUDE_min_tablets_for_given_box_l480_48084

/-- Given a box with tablets of two types of medicine, this function calculates
    the minimum number of tablets that must be extracted to guarantee at least
    two tablets of each type. -/
def min_tablets_to_extract (tablets_a tablets_b : ℕ) : ℕ :=
  max ((tablets_b + 1) + 2) ((tablets_a + 1) + 2)

/-- Theorem stating that for a box with 10 tablets of medicine A and 13 tablets
    of medicine B, the minimum number of tablets to extract to guarantee at
    least two of each kind is 15. -/
theorem min_tablets_for_given_box :
  min_tablets_to_extract 10 13 = 15 := by sorry

end NUMINAMATH_CALUDE_min_tablets_for_given_box_l480_48084


namespace NUMINAMATH_CALUDE_income_percentage_difference_l480_48000

/-- Given the monthly incomes of A and B in ratio 5:2, C's monthly income of 15000,
    and A's annual income of 504000, prove that B's monthly income is 12% more than C's. -/
theorem income_percentage_difference :
  ∀ (A_monthly B_monthly C_monthly : ℕ),
    C_monthly = 15000 →
    A_monthly * 12 = 504000 →
    A_monthly * 2 = B_monthly * 5 →
    (B_monthly - C_monthly) * 100 = C_monthly * 12 := by
  sorry

end NUMINAMATH_CALUDE_income_percentage_difference_l480_48000


namespace NUMINAMATH_CALUDE_wire_cutting_l480_48060

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 21 →
  ratio = 2 / 5 →
  shorter_length + shorter_length / ratio = total_length →
  shorter_length = 6 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l480_48060


namespace NUMINAMATH_CALUDE_xyz_value_l480_48049

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l480_48049


namespace NUMINAMATH_CALUDE_three_digit_prime_not_divisor_of_permutation_l480_48069

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_permutation (a b : ℕ) : Prop :=
  ∃ (x y z : ℕ), a = 100*x + 10*y + z ∧ b = 100*y + 10*z + x ∨
                  a = 100*x + 10*y + z ∧ b = 100*z + 10*x + y ∨
                  a = 100*x + 10*y + z ∧ b = 100*y + 10*x + z ∨
                  a = 100*x + 10*y + z ∧ b = 100*z + 10*y + x ∨
                  a = 100*x + 10*y + z ∧ b = 100*x + 10*z + y

theorem three_digit_prime_not_divisor_of_permutation (p : ℕ) (h1 : is_three_digit p) (h2 : Nat.Prime p) :
  ∀ n : ℕ, is_permutation p n → ¬(n % p = 0) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_prime_not_divisor_of_permutation_l480_48069


namespace NUMINAMATH_CALUDE_tangent_lines_count_l480_48011

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  abs (l.a * x₀ + l.b * y₀ + l.c) / Real.sqrt (l.a^2 + l.b^2) = c.radius

/-- Check if a line has equal intercepts on both axes -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

/-- The main theorem -/
theorem tangent_lines_count : 
  let c : Circle := ⟨(0, -5), 3⟩
  ∃ (lines : Finset Line), 
    lines.card = 4 ∧ 
    (∀ l ∈ lines, is_tangent l c ∧ has_equal_intercepts l) ∧
    (∀ l : Line, is_tangent l c ∧ has_equal_intercepts l → l ∈ lines) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l480_48011


namespace NUMINAMATH_CALUDE_pumpkin_pie_cost_pumpkin_pie_cost_proof_l480_48026

/-- The cost to make a pumpkin pie given the following conditions:
  * 10 pumpkin pies and 12 cherry pies are made
  * Cherry pies cost $5 each to make
  * The total profit is $20
  * Each pie is sold for $5
-/
theorem pumpkin_pie_cost : ℝ :=
  let num_pumpkin_pies : ℕ := 10
  let num_cherry_pies : ℕ := 12
  let cherry_pie_cost : ℝ := 5
  let profit : ℝ := 20
  let selling_price : ℝ := 5
  3

/-- Proof that the cost to make each pumpkin pie is $3 -/
theorem pumpkin_pie_cost_proof :
  let num_pumpkin_pies : ℕ := 10
  let num_cherry_pies : ℕ := 12
  let cherry_pie_cost : ℝ := 5
  let profit : ℝ := 20
  let selling_price : ℝ := 5
  pumpkin_pie_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_pie_cost_pumpkin_pie_cost_proof_l480_48026


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l480_48052

-- Statement 1
theorem inequality_one (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by sorry

-- Statement 2
theorem inequality_two {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  Real.sqrt ((a^2 + b^2) / 2) ≥ (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l480_48052


namespace NUMINAMATH_CALUDE_cube_adjacent_diagonals_perpendicular_l480_48087

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- A face diagonal is a line segment that connects opposite corners of a face -/
structure FaceDiagonal where
  cube : Cube
  face : Nat  -- We can use natural numbers to identify faces (1 to 6)

/-- The angle between two face diagonals -/
def angle_between_diagonals (d1 d2 : FaceDiagonal) : ℝ := sorry

/-- Two faces of a cube are adjacent if they share an edge -/
def adjacent_faces (f1 f2 : Nat) : Prop := sorry

/-- Theorem: The angle between the diagonals of any two adjacent faces of a cube is 90 degrees -/
theorem cube_adjacent_diagonals_perpendicular (c : Cube) (f1 f2 : Nat) (d1 d2 : FaceDiagonal)
  (h1 : d1.cube = c) (h2 : d2.cube = c) (h3 : d1.face = f1) (h4 : d2.face = f2)
  (h5 : adjacent_faces f1 f2) :
  angle_between_diagonals d1 d2 = 90 := by sorry

end NUMINAMATH_CALUDE_cube_adjacent_diagonals_perpendicular_l480_48087


namespace NUMINAMATH_CALUDE_cos_eq_neg_mul_sin_at_beta_l480_48068

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := |cos x| - k * x

theorem cos_eq_neg_mul_sin_at_beta
  (k : ℝ) (hk : k > 0)
  (α β : ℝ) (hα : α > 0) (hβ : β > 0) (hαβ : α < β)
  (hzeros : ∀ x, x > 0 → f k x = 0 ↔ x = α ∨ x = β)
  : cos β = -β * sin β :=
sorry

end NUMINAMATH_CALUDE_cos_eq_neg_mul_sin_at_beta_l480_48068


namespace NUMINAMATH_CALUDE_factorial_inequality_l480_48067

theorem factorial_inequality (k : ℕ) (h : k ≥ 2) :
  ((k + 1) / 2 : ℝ) ^ k > k! :=
by sorry

end NUMINAMATH_CALUDE_factorial_inequality_l480_48067


namespace NUMINAMATH_CALUDE_trapezoid_longer_base_l480_48083

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  midline_segment : ℝ
  height : ℝ

/-- Theorem: The longer base of a trapezoid with specific properties is 90 -/
theorem trapezoid_longer_base 
  (t : Trapezoid) 
  (h1 : t.shorter_base = 80) 
  (h2 : t.midline_segment = 5) 
  (h3 : t.height = 3 * t.midline_segment) 
  (h4 : t.midline_segment = (t.longer_base - t.shorter_base) / 2) : 
  t.longer_base = 90 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_longer_base_l480_48083


namespace NUMINAMATH_CALUDE_quadratic_roots_shift_l480_48041

theorem quadratic_roots_shift (b c : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂ ∧ x₁^2 + b*x₁ + c = 0 ∧ x₂^2 + b*x₂ + c = 0) → 
  ¬∃ (y₁ y₂ : ℝ), (y₁ ≠ y₂ ∧ y₁^2 + (b+1)*y₁ + (c+1) = 0 ∧ y₂^2 + (b+1)*y₂ + (c+1) = 0 ∧ 
                   y₁ = x₁ + 1 ∧ y₂ = x₂ + 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_shift_l480_48041


namespace NUMINAMATH_CALUDE_worker_speed_comparison_l480_48046

/-- Given that workers A and B can complete a work together in 18 days,
    and A alone can complete the work in 24 days,
    prove that A is 3 times faster than B. -/
theorem worker_speed_comparison (work : ℝ) (a_rate : ℝ) (b_rate : ℝ) :
  work > 0 →
  a_rate > 0 →
  b_rate > 0 →
  work / (a_rate + b_rate) = 18 →
  work / a_rate = 24 →
  a_rate / b_rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_worker_speed_comparison_l480_48046


namespace NUMINAMATH_CALUDE_square_implies_composite_l480_48019

theorem square_implies_composite (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_square : ∃ n : ℕ, x^2 + x*y - y = n^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ x + y + 1 = a * b :=
by sorry

end NUMINAMATH_CALUDE_square_implies_composite_l480_48019


namespace NUMINAMATH_CALUDE_coconut_grove_solution_l480_48030

-- Define the problem parameters
def coconut_grove (x : ℝ) : Prop :=
  -- (x + 3) trees yield 60 nuts per year
  ∃ (yield1 : ℝ), yield1 = 60 * (x + 3) ∧
  -- x trees yield 120 nuts per year
  ∃ (yield2 : ℝ), yield2 = 120 * x ∧
  -- (x - 3) trees yield 180 nuts per year
  ∃ (yield3 : ℝ), yield3 = 180 * (x - 3) ∧
  -- The average yield per year per tree is 100
  (yield1 + yield2 + yield3) / (3 * x) = 100

-- Theorem stating that x = 6 is the unique solution
theorem coconut_grove_solution :
  ∃! x : ℝ, coconut_grove x ∧ x = 6 :=
sorry

end NUMINAMATH_CALUDE_coconut_grove_solution_l480_48030


namespace NUMINAMATH_CALUDE_log_product_equality_l480_48045

open Real

theorem log_product_equality (A m n p : ℝ) (hA : A > 0) (hm : m > 0) (hn : n > 0) (hp : p > 0) :
  (log A / log m) * (log A / log n) + (log A / log n) * (log A / log p) + (log A / log p) * (log A / log m) =
  (log (m * n * p) / log A) * (log A / log p) * (log A / log n) * (log A / log m) :=
by sorry

#check log_product_equality

end NUMINAMATH_CALUDE_log_product_equality_l480_48045


namespace NUMINAMATH_CALUDE_inequality_for_all_reals_l480_48090

theorem inequality_for_all_reals (a : ℝ) : a + a^3 - a^4 - a^6 < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_all_reals_l480_48090


namespace NUMINAMATH_CALUDE_xiaolis_estimate_l480_48086

theorem xiaolis_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1.1 * x) / (0.9 * y) > x / y := by
  sorry

end NUMINAMATH_CALUDE_xiaolis_estimate_l480_48086


namespace NUMINAMATH_CALUDE_divisible_by_4_or_5_count_l480_48097

def count_divisible (n : ℕ) : ℕ :=
  (n / 4) + (n / 5) - (n / 20)

theorem divisible_by_4_or_5_count :
  count_divisible 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_4_or_5_count_l480_48097


namespace NUMINAMATH_CALUDE_remainder_98_35_mod_100_l480_48005

theorem remainder_98_35_mod_100 : 98^35 ≡ -24 [ZMOD 100] := by sorry

end NUMINAMATH_CALUDE_remainder_98_35_mod_100_l480_48005


namespace NUMINAMATH_CALUDE_valid_gift_wrapping_combinations_l480_48035

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 3

/-- The number of gift card types -/
def gift_card_types : ℕ := 5

/-- The number of invalid combinations (red ribbon with birthday card) -/
def invalid_combinations : ℕ := 1

/-- Theorem stating the number of valid gift wrapping combinations -/
theorem valid_gift_wrapping_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types - invalid_combinations = 149 := by
sorry

end NUMINAMATH_CALUDE_valid_gift_wrapping_combinations_l480_48035


namespace NUMINAMATH_CALUDE_soda_weight_proof_l480_48065

/-- Calculates the amount of soda in each can given the total weight, number of cans, and weight of empty cans. -/
def soda_per_can (total_weight : ℕ) (soda_cans : ℕ) (empty_cans : ℕ) (empty_can_weight : ℕ) : ℕ :=
  (total_weight - (soda_cans + empty_cans) * empty_can_weight) / soda_cans

/-- Proves that the amount of soda in each can is 12 ounces given the problem conditions. -/
theorem soda_weight_proof (total_weight : ℕ) (soda_cans : ℕ) (empty_cans : ℕ) (empty_can_weight : ℕ)
  (h1 : total_weight = 88)
  (h2 : soda_cans = 6)
  (h3 : empty_cans = 2)
  (h4 : empty_can_weight = 2) :
  soda_per_can total_weight soda_cans empty_cans empty_can_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_soda_weight_proof_l480_48065


namespace NUMINAMATH_CALUDE_water_added_to_container_l480_48015

/-- The amount of water added to fill a container from 30% to 3/4 full -/
theorem water_added_to_container (capacity : ℝ) (initial_fraction : ℝ) (final_fraction : ℝ) 
  (h1 : capacity = 100)
  (h2 : initial_fraction = 0.3)
  (h3 : final_fraction = 3/4) :
  final_fraction * capacity - initial_fraction * capacity = 45 :=
by sorry

end NUMINAMATH_CALUDE_water_added_to_container_l480_48015


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l480_48092

theorem complex_multiplication_result : (1 + Complex.I) * (-Complex.I) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l480_48092


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_one_third_l480_48003

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem logarithm_expression_equals_one_third :
  (lg (1/4) - lg 25) / (2 * log_base 5 10 + log_base 5 (1/4)) + log_base 3 4 * log_base 8 9 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_logarithm_expression_equals_one_third_l480_48003


namespace NUMINAMATH_CALUDE_oreo_multiple_l480_48028

def total_oreos : ℕ := 52
def james_oreos : ℕ := 43

theorem oreo_multiple :
  ∃ (multiple : ℕ) (jordan_oreos : ℕ),
    james_oreos = multiple * jordan_oreos + 7 ∧
    total_oreos = james_oreos + jordan_oreos ∧
    multiple = 4 := by
  sorry

end NUMINAMATH_CALUDE_oreo_multiple_l480_48028


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l480_48050

-- Define the function g
variable (g : ℝ → ℝ)

-- State the theorem
theorem point_on_transformed_graph (h : g 3 = 10) :
  ∃ (x y : ℝ), 3 * y = 4 * g (3 * x) + 6 ∧ x = 1 ∧ y = 46 / 3 ∧ x + y = 49 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l480_48050
