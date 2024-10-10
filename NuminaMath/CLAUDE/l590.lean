import Mathlib

namespace max_rectangles_in_oblique_prism_l590_59047

/-- Represents an oblique prism -/
structure ObliquePrism where
  base : Set (Point)
  lateral_edges : Set (Line)

/-- Counts the number of rectangular faces in an oblique prism -/
def count_rectangular_faces (prism : ObliquePrism) : ℕ := sorry

/-- The maximum number of rectangular faces in any oblique prism -/
def max_rectangular_faces : ℕ := 4

/-- Theorem stating that the maximum number of rectangular faces in an oblique prism is 4 -/
theorem max_rectangles_in_oblique_prism (prism : ObliquePrism) :
  count_rectangular_faces prism ≤ max_rectangular_faces :=
sorry

end max_rectangles_in_oblique_prism_l590_59047


namespace trivia_competition_score_l590_59064

theorem trivia_competition_score :
  ∀ (total_members absent_members points_per_member : ℕ),
    total_members = 120 →
    absent_members = 37 →
    points_per_member = 24 →
    (total_members - absent_members) * points_per_member = 1992 :=
by
  sorry

end trivia_competition_score_l590_59064


namespace like_terms_imply_expression_value_l590_59006

theorem like_terms_imply_expression_value :
  ∀ (a b : ℤ),
  (2 : ℤ) = 1 - a →
  (5 : ℤ) = 3 * b - 1 →
  5 * a * b^2 - (6 * a^2 * b - 3 * (a * b^2 + 2 * a^2 * b)) = -32 :=
by sorry

end like_terms_imply_expression_value_l590_59006


namespace sqrt_equation_solution_l590_59080

theorem sqrt_equation_solution (x y : ℝ) : 
  Real.sqrt (10 + 3 * x - y) = 7 → y = 3 * x - 39 := by
  sorry

end sqrt_equation_solution_l590_59080


namespace common_chord_length_l590_59027

/-- Given two intersecting circles with radii in ratio 4:3, prove that the length of their common chord is 2√2 when the segment connecting their centers is divided into parts of length 5 and 2 by the common chord. -/
theorem common_chord_length (r₁ r₂ : ℝ) (h_ratio : r₁ = (4/3) * r₂) 
  (center_distance : ℝ) (h_center_distance : center_distance = 7)
  (segment_1 segment_2 : ℝ) (h_segment_1 : segment_1 = 5) (h_segment_2 : segment_2 = 2)
  (h_segments_sum : segment_1 + segment_2 = center_distance) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 2 :=
sorry

end common_chord_length_l590_59027


namespace inequality_solution_l590_59067

/-- Given an inequality (ax-1)/(x+1) < 0 with solution set {x | x < -1 or x > -1/2}, prove that a = -2 -/
theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, (a * x - 1) / (x + 1) < 0 ↔ (x < -1 ∨ x > -1/2)) → 
  a = -2 := by
sorry

end inequality_solution_l590_59067


namespace longest_tape_l590_59091

theorem longest_tape (red_tape blue_tape yellow_tape : ℚ) 
  (h_red : red_tape = 11/6)
  (h_blue : blue_tape = 7/4)
  (h_yellow : yellow_tape = 13/8) :
  red_tape > blue_tape ∧ red_tape > yellow_tape := by
  sorry

end longest_tape_l590_59091


namespace solution_set_f_positive_range_of_a_l590_59065

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) > 0
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x₀ : ℝ, f x₀ + 2*a^2 < 4*a} = {a : ℝ | -1/2 < a ∧ a < 5/2} := by sorry

end solution_set_f_positive_range_of_a_l590_59065


namespace credit_remaining_proof_l590_59098

def credit_problem (credit_limit : ℕ) (payment1 : ℕ) (payment2 : ℕ) : ℕ :=
  credit_limit - payment1 - payment2

theorem credit_remaining_proof :
  credit_problem 100 15 23 = 62 := by
  sorry

end credit_remaining_proof_l590_59098


namespace units_digit_of_k_squared_plus_two_to_k_l590_59086

def k : ℕ := 2009^2 + 2^2009 - 3

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := 2009^2 + 2^2009 - 3) :
  (k^2 + 2^k) % 10 = 1 := by
  sorry

end units_digit_of_k_squared_plus_two_to_k_l590_59086


namespace bucket_filling_time_l590_59036

theorem bucket_filling_time (total_time : ℝ) (h : total_time = 135) : 
  (2 / 3 : ℝ) * total_time = 90 := by
  sorry

end bucket_filling_time_l590_59036


namespace vector_triangle_sum_zero_l590_59039

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_triangle_sum_zero (A B C : E) :
  (B - A) + (C - B) + (A - C) = (0 : E) := by
  sorry

end vector_triangle_sum_zero_l590_59039


namespace partial_fraction_decomposition_l590_59010

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (4 * x - 2) / (x^3 - x) = 2 / x + 1 / (x - 1) - 3 / (x + 1) := by
  sorry

end partial_fraction_decomposition_l590_59010


namespace stationery_difference_l590_59001

def georgia_stationery : ℚ := 25

def lorene_stationery : ℚ := 3 * georgia_stationery

def bria_stationery : ℚ := georgia_stationery + 10

def darren_stationery : ℚ := bria_stationery / 2

theorem stationery_difference :
  lorene_stationery + bria_stationery + darren_stationery - georgia_stationery = 102.5 := by
  sorry

end stationery_difference_l590_59001


namespace parabola_translation_l590_59024

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk (-5) 0 1
  let translated := translate original (-1) (-2)
  translated = Parabola.mk (-5) 10 (-1) := by sorry

end parabola_translation_l590_59024


namespace integer_power_sum_l590_59099

theorem integer_power_sum (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end integer_power_sum_l590_59099


namespace largest_package_size_l590_59055

theorem largest_package_size (john_markers alex_markers : ℕ) 
  (h1 : john_markers = 36) (h2 : alex_markers = 60) : 
  Nat.gcd john_markers alex_markers = 12 := by
  sorry

end largest_package_size_l590_59055


namespace toothpaste_problem_l590_59094

/-- Represents the amount of toothpaste used by Anne's mom at each brushing -/
def moms_toothpaste_usage : ℝ := 2

/-- The problem statement -/
theorem toothpaste_problem (
  total_toothpaste : ℝ)
  (dads_usage : ℝ)
  (kids_usage : ℝ)
  (brushings_per_day : ℕ)
  (days_until_empty : ℕ)
  (h1 : total_toothpaste = 105)
  (h2 : dads_usage = 3)
  (h3 : kids_usage = 1)
  (h4 : brushings_per_day = 3)
  (h5 : days_until_empty = 5)
  : moms_toothpaste_usage * (brushings_per_day : ℝ) * days_until_empty +
    dads_usage * (brushings_per_day : ℝ) * days_until_empty +
    2 * kids_usage * (brushings_per_day : ℝ) * days_until_empty =
    total_toothpaste :=
by sorry

end toothpaste_problem_l590_59094


namespace tan_half_angle_special_case_l590_59088

theorem tan_half_angle_special_case (α : Real) 
  (h1 : 5 * Real.sin (2 * α) = 6 * Real.cos α) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan (α / 2) = 1 / 3 := by
sorry

end tan_half_angle_special_case_l590_59088


namespace fiftieth_parentheses_sum_l590_59054

/-- Represents the sum of numbers in a set of parentheses at a given position -/
def parenthesesSum (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 1
  | 2 => 2 + 2
  | 3 => 3 + 3 + 3
  | 0 => 4 + 4 + 4 + 4
  | _ => 0  -- This case should never occur

/-- The sum of numbers in the 50th set of parentheses is 4 -/
theorem fiftieth_parentheses_sum : parenthesesSum 50 = 4 := by
  sorry

end fiftieth_parentheses_sum_l590_59054


namespace polynomial_simplification_l590_59051

theorem polynomial_simplification (p : ℝ) : 
  (4 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 5) + (-3 * p^4 - 2 * p^3 + 8 * p^2 - 4 * p + 6) = 
  p^4 + p^2 - p + 1 := by
sorry

end polynomial_simplification_l590_59051


namespace absolute_value_ratio_l590_59071

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 10 * a * b) :
  |((a + b) / (a - b))| = Real.sqrt (3 / 2) := by
  sorry

end absolute_value_ratio_l590_59071


namespace alternating_subtraction_theorem_l590_59014

def alternating_subtraction (n : ℕ) : ℤ :=
  if n % 2 = 0 then 0 else -1

theorem alternating_subtraction_theorem (n : ℕ) :
  alternating_subtraction n = if n % 2 = 0 then 0 else -1 :=
by sorry

-- Examples for the given cases
example : alternating_subtraction 1989 = -1 :=
by sorry

example : alternating_subtraction 1990 = 0 :=
by sorry

end alternating_subtraction_theorem_l590_59014


namespace decagon_diagonals_l590_59016

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l590_59016


namespace cube_of_square_of_third_smallest_prime_l590_59096

def third_smallest_prime : ℕ := sorry

theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by sorry

end cube_of_square_of_third_smallest_prime_l590_59096


namespace abc_inequalities_l590_59025

theorem abc_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (2 * a * b + b * c + c * a + c^2 / 2 ≤ 1 / 2) ∧ 
  ((a^2 + c^2) / b + (b^2 + a^2) / c + (c^2 + b^2) / a ≥ 2) := by
  sorry

end abc_inequalities_l590_59025


namespace polynomial_factorization_l590_59019

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end polynomial_factorization_l590_59019


namespace below_warning_level_notation_l590_59066

/-- Represents the water level relative to a warning level -/
def water_level_notation (warning_level : ℝ) (actual_level : ℝ) : ℝ :=
  actual_level - warning_level

theorem below_warning_level_notation 
  (warning_level : ℝ) (distance_below : ℝ) (distance_below_positive : distance_below > 0) :
  water_level_notation warning_level (warning_level - distance_below) = -distance_below :=
by sorry

end below_warning_level_notation_l590_59066


namespace absolute_value_sum_equality_l590_59050

theorem absolute_value_sum_equality (x y : ℝ) : 
  (|x + y| = |x| + |y|) ↔ x * y ≥ 0 := by sorry

end absolute_value_sum_equality_l590_59050


namespace unique_arrangement_l590_59056

-- Define the characters
inductive Character
| GrayHorse
| GrayMare
| BearCub

-- Define the positions
inductive Position
| Left
| Center
| Right

-- Define the arrangement as a function from Position to Character
def Arrangement := Position → Character

-- Define the property of always lying
def alwaysLies (c : Character) : Prop :=
  c = Character.GrayHorse

-- Define the property of never lying
def neverLies (c : Character) : Prop :=
  c = Character.BearCub

-- Define the statements made by each position
def leftStatement (arr : Arrangement) : Prop :=
  arr Position.Center = Character.BearCub

def rightStatement (arr : Arrangement) : Prop :=
  arr Position.Left = Character.GrayMare

def centerStatement (arr : Arrangement) : Prop :=
  arr Position.Left = Character.GrayHorse

-- Define the correctness of a statement based on who said it
def isCorrectStatement (arr : Arrangement) (pos : Position) (stmt : Prop) : Prop :=
  (alwaysLies (arr pos) ∧ ¬stmt) ∨
  (neverLies (arr pos) ∧ stmt) ∨
  (¬alwaysLies (arr pos) ∧ ¬neverLies (arr pos))

-- Main theorem
theorem unique_arrangement :
  ∃! arr : Arrangement,
    (arr Position.Left = Character.GrayMare) ∧
    (arr Position.Center = Character.GrayHorse) ∧
    (arr Position.Right = Character.BearCub) ∧
    isCorrectStatement arr Position.Left (leftStatement arr) ∧
    isCorrectStatement arr Position.Right (rightStatement arr) ∧
    isCorrectStatement arr Position.Center (centerStatement arr) :=
  sorry


end unique_arrangement_l590_59056


namespace trigonometric_identity_l590_59028

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sin θ)^4 / a + (Real.cos θ)^4 / b = 1 / (a + b) →
  (Real.sin θ)^8 / a^3 + (Real.cos θ)^8 / b^3 = 1 / (a + b)^3 := by
  sorry

end trigonometric_identity_l590_59028


namespace no_solution_implies_a_equals_negative_two_l590_59062

theorem no_solution_implies_a_equals_negative_two (a : ℝ) : 
  (∀ x y : ℝ, ¬(a * x + 2 * y = a + 2 ∧ 2 * x + a * y = 2 * a)) → a = -2 := by
  sorry

end no_solution_implies_a_equals_negative_two_l590_59062


namespace sum_xy_value_l590_59058

theorem sum_xy_value (x y : ℝ) (h1 : x + 2*y = 5) (h2 : (x + y) / 3 = 1.222222222222222) :
  x + y = 3.666666666666666 := by
  sorry

end sum_xy_value_l590_59058


namespace fraction_simplification_l590_59012

theorem fraction_simplification : (3 : ℚ) / (1 - 2 / 5) = 5 := by
  sorry

end fraction_simplification_l590_59012


namespace no_positive_integer_solution_l590_59013

def first_2015_primes : List Nat := sorry

def m : Nat := List.prod first_2015_primes

theorem no_positive_integer_solution :
  ∀ x y z : Nat, (2 * x - y - z) * (2 * y - z - x) * (2 * z - x - y) ≠ m :=
sorry

end no_positive_integer_solution_l590_59013


namespace complex_set_sum_l590_59021

/-- A set of complex numbers with closure under multiplication property -/
def ClosedMultSet (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_sum (a b c d : ℂ) :
  let S := {a, b, c, d}
  ClosedMultSet S →
  a^2 = 1 →
  b^2 = 1 →
  c^2 = b →
  b + c + d = -1 := by
  sorry

end complex_set_sum_l590_59021


namespace complex_division_sum_l590_59020

theorem complex_division_sum (a b : ℝ) : 
  (Complex.I - 2) / (1 + Complex.I) = Complex.ofReal a + Complex.I * Complex.ofReal b → 
  a + b = 1 := by sorry

end complex_division_sum_l590_59020


namespace fifteenth_term_of_sequence_l590_59061

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence : 
  let a₁ : ℚ := 5
  let r : ℚ := 1/2
  let n : ℕ := 15
  geometric_sequence a₁ r n = 5/16384 := by
sorry

end fifteenth_term_of_sequence_l590_59061


namespace candy_bar_sales_l590_59043

/-- The number of candy bars Marvin sold -/
def marvins_candy_bars : ℕ := 35

theorem candy_bar_sales : 
  let candy_bar_price : ℕ := 2
  let tinas_candy_bars : ℕ := 3 * marvins_candy_bars
  let marvins_revenue : ℕ := candy_bar_price * marvins_candy_bars
  let tinas_revenue : ℕ := candy_bar_price * tinas_candy_bars
  tinas_revenue = marvins_revenue + 140 → marvins_candy_bars = 35 := by
  sorry

end candy_bar_sales_l590_59043


namespace shaded_area_is_two_thirds_l590_59000

/-- Square PQRS with shaded regions -/
structure ShadedSquare where
  /-- Side length of the square PQRS -/
  side_length : ℝ
  /-- Side length of the first shaded square region -/
  first_region : ℝ
  /-- Side length of the outer square in the second shaded region -/
  second_region_outer : ℝ
  /-- Side length of the inner square in the second shaded region -/
  second_region_inner : ℝ
  /-- Side length of the outer square in the third shaded region -/
  third_region_outer : ℝ
  /-- Side length of the inner square in the third shaded region -/
  third_region_inner : ℝ

/-- Theorem stating that the shaded area is 2/3 of the total area -/
theorem shaded_area_is_two_thirds (sq : ShadedSquare)
    (h1 : sq.side_length = 6)
    (h2 : sq.first_region = 1)
    (h3 : sq.second_region_outer = 4)
    (h4 : sq.second_region_inner = 2)
    (h5 : sq.third_region_outer = 6)
    (h6 : sq.third_region_inner = 5) :
    (sq.first_region^2 + (sq.second_region_outer^2 - sq.second_region_inner^2) +
     (sq.third_region_outer^2 - sq.third_region_inner^2)) / sq.side_length^2 = 2/3 := by
  sorry

end shaded_area_is_two_thirds_l590_59000


namespace equation_solutions_l590_59077

theorem equation_solutions :
  (∃ x : ℝ, x + 2*x = 12.6 ∧ x = 4.2) ∧
  (∃ x : ℝ, (1/4)*x + 1/2 = 3/5 ∧ x = 2/5) ∧
  (∃ x : ℝ, x + 1.3*x = 46 ∧ x = 20) := by
  sorry

end equation_solutions_l590_59077


namespace price_difference_is_70_l590_59095

-- Define the pricing structures and discount rates
def shop_x_base_price : ℝ := 1.25
def shop_y_base_price : ℝ := 2.75
def shop_x_discount_rate_80plus : ℝ := 0.10
def shop_y_bulk_price_80plus : ℝ := 2.00

-- Define the number of copies
def num_copies : ℕ := 80

-- Calculate the price for Shop X
def shop_x_price (copies : ℕ) : ℝ :=
  shop_x_base_price * copies * (1 - shop_x_discount_rate_80plus)

-- Calculate the price for Shop Y
def shop_y_price (copies : ℕ) : ℝ :=
  shop_y_bulk_price_80plus * copies

-- Theorem to prove
theorem price_difference_is_70 :
  shop_y_price num_copies - shop_x_price num_copies = 70 := by
  sorry


end price_difference_is_70_l590_59095


namespace original_price_is_360_l590_59023

/-- The original price of a product satisfies two conditions:
    1. When sold at 75% of the original price, there's a loss of $12 per item.
    2. When sold at 90% of the original price, there's a profit of $42 per item. -/
theorem original_price_is_360 (price : ℝ) 
    (h1 : 0.75 * price + 12 = 0.9 * price - 42) : 
    price = 360 := by
  sorry

end original_price_is_360_l590_59023


namespace print_time_rounded_l590_59004

/-- The number of pages to be printed -/
def total_pages : ℕ := 350

/-- The number of pages printed per minute -/
def pages_per_minute : ℕ := 25

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The time required to print the pages, in minutes -/
def print_time : ℚ := total_pages / pages_per_minute

theorem print_time_rounded : round_to_nearest print_time = 14 := by
  sorry

end print_time_rounded_l590_59004


namespace quadratic_is_perfect_square_l590_59081

theorem quadratic_is_perfect_square : ∃ (a b : ℝ), ∀ x : ℝ, x^2 - 18*x + 81 = (a*x + b)^2 := by
  sorry

end quadratic_is_perfect_square_l590_59081


namespace linear_function_properties_l590_59075

/-- A linear function f(x) = k(x + 2) where k ≠ 0 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * (x + 2)

/-- g is f shifted 2 units upwards -/
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_properties (k : ℝ) (h : k ≠ 0) :
  (f k (-2) = 0) ∧
  (g k 1 = -2 → k = -4/3) ∧
  (0 > f k 0 ∧ f k 0 > -2 → -1 < k ∧ k < 0) := by
  sorry

end linear_function_properties_l590_59075


namespace divisibility_by_240_l590_59049

theorem divisibility_by_240 (p : ℕ) (hp : Nat.Prime p) (hp_ge_7 : p ≥ 7) :
  (240 : ℕ) ∣ (p^4 - 1) := by
  sorry

end divisibility_by_240_l590_59049


namespace area_between_concentric_circles_l590_59007

theorem area_between_concentric_circles (r_small : ℝ) (r_large : ℝ) : 
  r_small = 3 →
  r_large = 3 * r_small →
  π * r_large^2 - π * r_small^2 = 72 * π :=
by
  sorry

end area_between_concentric_circles_l590_59007


namespace sufficient_not_necessary_l590_59002

theorem sufficient_not_necessary (x : ℝ) : 
  (∃ (S T : Set ℝ), 
    S = {x | x > 2} ∧ 
    T = {x | x^2 - 3*x + 2 > 0} ∧ 
    S ⊂ T ∧ 
    ∃ y, y ∈ T ∧ y ∉ S) :=
by sorry

end sufficient_not_necessary_l590_59002


namespace sum_of_three_consecutive_multiples_of_three_l590_59085

theorem sum_of_three_consecutive_multiples_of_three (a b c : ℕ) : 
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 ∧  -- a, b, c are multiples of 3
  b = a + 3 ∧ c = b + 3 ∧               -- a, b, c are consecutive
  c = 27 →                              -- the largest number is 27
  a + b + c = 72 :=                     -- the sum is 72
by sorry

end sum_of_three_consecutive_multiples_of_three_l590_59085


namespace inverse_proportion_relation_l590_59089

/-- Given that the points (2, y₁) and (3, y₂) lie on the graph of the inverse proportion function y = 6/x,
    prove that y₁ > y₂. -/
theorem inverse_proportion_relation (y₁ y₂ : ℝ) :
  (2 : ℝ) * y₁ = 6 ∧ (3 : ℝ) * y₂ = 6 → y₁ > y₂ := by
  sorry

end inverse_proportion_relation_l590_59089


namespace double_fibonacci_sum_convergence_l590_59070

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def G (n : ℕ) : ℕ := 2 * fibonacci n

theorem double_fibonacci_sum_convergence :
  (∑' n, (G n : ℝ) / 5^n) = 10/19 := by sorry

end double_fibonacci_sum_convergence_l590_59070


namespace product_of_numbers_with_given_sum_and_difference_l590_59034

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 26 ∧ x - y = 8 → x * y = 153 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l590_59034


namespace blood_donation_selection_l590_59097

theorem blood_donation_selection (o a b ab : ℕ) 
  (ho : o = 18) (ha : a = 10) (hb : b = 8) (hab : ab = 3) : 
  o * a * b * ab = 4320 := by
  sorry

end blood_donation_selection_l590_59097


namespace positive_range_of_even_function_l590_59087

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem positive_range_of_even_function
  (f : ℝ → ℝ)
  (f' : ℝ → ℝ)
  (h_even : evenFunction f)
  (h_deriv : ∀ x ≠ 0, HasDerivAt f (f' x) x)
  (h_zero : f (-1) = 0)
  (h_ineq : ∀ x > 0, x * f' x - f x < 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 := by
sorry

end positive_range_of_even_function_l590_59087


namespace converse_and_inverse_true_l590_59015

-- Define the properties
def is_circle (shape : Type) : Prop := sorry
def has_constant_curvature (shape : Type) : Prop := sorry

-- Given statement
axiom circle_implies_constant_curvature : 
  ∀ (shape : Type), is_circle shape → has_constant_curvature shape

-- Theorem to prove
theorem converse_and_inverse_true : 
  (∀ (shape : Type), has_constant_curvature shape → is_circle shape) ∧ 
  (∀ (shape : Type), ¬is_circle shape → ¬has_constant_curvature shape) := by
  sorry

end converse_and_inverse_true_l590_59015


namespace order_of_exponents_l590_59031

theorem order_of_exponents :
  let a : ℝ := (36 : ℝ) ^ (1/5)
  let b : ℝ := (3 : ℝ) ^ (4/3)
  let c : ℝ := (9 : ℝ) ^ (2/5)
  a < c ∧ c < b := by sorry

end order_of_exponents_l590_59031


namespace arithmetic_sequence_general_term_l590_59059

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  a₁_eq_1 : a 1 = 1
  geometric_subseq : (a 3)^2 = a 1 * a 9

/-- The general term of the arithmetic sequence is either n or 1 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = n) ∨ (∀ n : ℕ, seq.a n = 1) :=
sorry

end arithmetic_sequence_general_term_l590_59059


namespace parabola_directrix_l590_59044

/-- The directrix of a parabola y² = 2px passing through (2, 2) -/
theorem parabola_directrix (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x → x = 2 → y = 2) → 
  (∃ k : ℝ, ∀ x y : ℝ, y^2 = 2*p*x → x = k) ∧ k = -1/2 :=
sorry

end parabola_directrix_l590_59044


namespace intersection_of_M_and_N_l590_59032

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {x | x^2 ≥ 2*x}

theorem intersection_of_M_and_N : M ∩ N = {2} := by
  sorry

end intersection_of_M_and_N_l590_59032


namespace inverse_of_B_squared_l590_59072

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, -2; 0, 5]) : 
  (B^2)⁻¹ = !![9, -16; 0, 25] := by sorry

end inverse_of_B_squared_l590_59072


namespace pencil_distribution_l590_59030

theorem pencil_distribution (x y : ℕ+) (h1 : 3 * x < 48) (h2 : 48 < 4 * x) 
  (h3 : 4 * y < 48) (h4 : 48 < 5 * y) : 
  (3 * x < 48 ∧ 48 < 4 * x) ∧ (4 * y < 48 ∧ 48 < 5 * y) := by
  sorry

end pencil_distribution_l590_59030


namespace min_max_expression_l590_59038

theorem min_max_expression (a b c : ℝ) 
  (eq1 : a^2 + a*b + b^2 = 19)
  (eq2 : b^2 + b*c + c^2 = 19) :
  (∃ x y z : ℝ, x^2 + x*y + y^2 = 19 ∧ y^2 + y*z + z^2 = 19 ∧ z^2 + z*x + x^2 = 0) ∧
  (∀ x y z : ℝ, x^2 + x*y + y^2 = 19 → y^2 + y*z + z^2 = 19 → z^2 + z*x + x^2 ≤ 76) :=
by
  sorry

end min_max_expression_l590_59038


namespace jogger_speed_l590_59046

/-- The speed of a jogger given specific conditions involving a train --/
theorem jogger_speed (train_length : ℝ) (initial_distance : ℝ) (train_speed : ℝ) (passing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : initial_distance = 120)
  (h3 : train_speed = 45)
  (h4 : passing_time = 24)
  : ∃ (jogger_speed : ℝ), jogger_speed = 9 := by
  sorry


end jogger_speed_l590_59046


namespace sin_2x_minus_pi_3_zeros_min_distance_l590_59042

open Real

theorem sin_2x_minus_pi_3_zeros_min_distance (f : ℝ → ℝ) (h : ∀ x, f x = sin (2 * x - π / 3)) :
  ∀ a b : ℝ, a ≠ b → f a = 0 → f b = 0 → |a - b| ≥ π / 2 ∧ ∃ c d : ℝ, c ≠ d ∧ f c = 0 ∧ f d = 0 ∧ |c - d| = π / 2 :=
sorry

end sin_2x_minus_pi_3_zeros_min_distance_l590_59042


namespace floor_sum_example_l590_59083

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l590_59083


namespace absolute_value_equation_solution_l590_59090

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => 3*x + 6
  let g : ℝ → ℝ := λ x => |(-20 + x^2)|
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 113) / 2 ∧
              x₂ = (3 - Real.sqrt 113) / 2 ∧
              (∀ x : ℝ, f x = g x ↔ x = x₁ ∨ x = x₂) :=
by sorry

end absolute_value_equation_solution_l590_59090


namespace product_of_roots_l590_59052

theorem product_of_roots (t : ℝ) : (∀ t, t^2 = 49) → (t * (-t) = -49) := by
  sorry

end product_of_roots_l590_59052


namespace complete_square_sum_l590_59017

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
sorry

end complete_square_sum_l590_59017


namespace spools_per_beret_l590_59076

theorem spools_per_beret (total_spools : ℕ) (num_berets : ℕ) 
  (h1 : total_spools = 33) 
  (h2 : num_berets = 11) 
  (h3 : num_berets > 0) : 
  total_spools / num_berets = 3 := by
  sorry

end spools_per_beret_l590_59076


namespace train_speed_l590_59093

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 140) (h2 : time = 16) :
  length / time = 8.75 := by
  sorry

end train_speed_l590_59093


namespace car_speed_second_hour_l590_59082

/-- Proves that given a car's average speed and first hour speed, we can determine the second hour speed -/
theorem car_speed_second_hour 
  (average_speed : ℝ) 
  (first_hour_speed : ℝ) 
  (h1 : average_speed = 55) 
  (h2 : first_hour_speed = 65) : 
  ∃ (second_hour_speed : ℝ), 
    second_hour_speed = 45 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / 2 := by
  sorry

end car_speed_second_hour_l590_59082


namespace append_five_to_two_digit_number_l590_59045

/-- Given a two-digit number with tens digit t and units digit u,
    appending the digit 5 results in the number 100t + 10u + 5 -/
theorem append_five_to_two_digit_number (t u : ℕ) :
  let original := 10 * t + u
  let appended := original * 10 + 5
  appended = 100 * t + 10 * u + 5 := by
sorry

end append_five_to_two_digit_number_l590_59045


namespace y_coordinate_order_l590_59069

-- Define the quadratic function
def f (x : ℝ) (b : ℝ) : ℝ := -x^2 + 2*x + b

-- Define the points A, B, C
def A (b : ℝ) : ℝ × ℝ := (4, f 4 b)
def B (b : ℝ) : ℝ × ℝ := (-1, f (-1) b)
def C (b : ℝ) : ℝ × ℝ := (1, f 1 b)

-- Theorem stating the order of y-coordinates
theorem y_coordinate_order (b : ℝ) :
  (A b).2 < (B b).2 ∧ (B b).2 < (C b).2 :=
by sorry

end y_coordinate_order_l590_59069


namespace no_natural_n_power_of_two_l590_59022

theorem no_natural_n_power_of_two : ∀ n : ℕ, ¬∃ k : ℕ, 6 * n^2 + 5 * n = 2^k := by sorry

end no_natural_n_power_of_two_l590_59022


namespace x_in_terms_of_y_and_k_l590_59026

theorem x_in_terms_of_y_and_k (x y k : ℝ) :
  x / (x - k) = (y^2 + 3*y + 2) / (y^2 + 3*y + 1) →
  x = k*y^2 + 3*k*y + 2*k := by
  sorry

end x_in_terms_of_y_and_k_l590_59026


namespace shaded_area_ratio_l590_59018

/-- The ratio of the area of a square composed of 5 half-squares to the area of a larger square divided into 25 equal parts is 1/10 -/
theorem shaded_area_ratio (large_square_area : ℝ) (small_square_area : ℝ) 
  (h1 : large_square_area > 0)
  (h2 : small_square_area > 0)
  (h3 : large_square_area = 25 * small_square_area)
  (shaded_area : ℝ)
  (h4 : shaded_area = 5 * (small_square_area / 2)) :
  shaded_area / large_square_area = 1 / 10 := by
sorry


end shaded_area_ratio_l590_59018


namespace matrix_equation_solution_l590_59053

/-- The value of a 2x2 matrix [[a, c], [d, b]] is ab - cd -/
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

/-- The solution to the matrix equation for a given k -/
def solution (k : ℝ) : Set ℝ :=
  {x : ℝ | x = (4 + Real.sqrt (16 + 60 * k)) / 30 ∨ x = (4 - Real.sqrt (16 + 60 * k)) / 30}

theorem matrix_equation_solution (k : ℝ) (h : k ≥ -4/15) :
  ∀ x : ℝ, matrix_value (3*x) (5*x) 2 (2*x) = k ↔ x ∈ solution k := by
  sorry

end matrix_equation_solution_l590_59053


namespace soccer_game_theorem_l590_59079

def soccer_game (team_a_first_half : ℕ) (team_b_second_half : ℕ) (total_goals : ℕ) : Prop :=
  let team_b_first_half := team_a_first_half / 2
  let first_half_total := team_a_first_half + team_b_first_half
  let second_half_total := total_goals - first_half_total
  let team_a_second_half := second_half_total - team_b_second_half
  (team_a_first_half = 8) ∧
  (team_b_second_half = team_a_first_half) ∧
  (total_goals = 26) ∧
  (team_b_second_half > team_a_second_half) ∧
  (team_b_second_half - team_a_second_half = 2)

theorem soccer_game_theorem :
  ∃ (team_a_first_half team_b_second_half total_goals : ℕ),
    soccer_game team_a_first_half team_b_second_half total_goals :=
by
  sorry

end soccer_game_theorem_l590_59079


namespace equation_solution_l590_59073

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 3 ∧ x + 25 / (x - 3) = -8 := by sorry

end equation_solution_l590_59073


namespace function_equation_solution_l590_59008

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a^2 + a*b + f (b^2)) = a * f b + b^2 + f (a^2)

/-- The main theorem stating that any function satisfying the equation is either the identity or negation -/
theorem function_equation_solution (f : ℝ → ℝ) (hf : SatisfiesEquation f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end function_equation_solution_l590_59008


namespace problem_1_problem_2_l590_59074

-- Problem 1
theorem problem_1 : 123^2 - 124 * 122 = 1 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (-2*a^2*b)^3 / (-a*b) * (1/2*a^2*b)^3 = a^11*b^5 := by sorry

end problem_1_problem_2_l590_59074


namespace foundation_dig_time_l590_59003

/-- Represents the time taken to dig a foundation given the number of men -/
def digTime (men : ℕ) : ℝ := sorry

theorem foundation_dig_time :
  (digTime 20 = 6) →  -- It takes 20 men 6 days
  (∀ m₁ m₂ : ℕ, m₁ * digTime m₁ = m₂ * digTime m₂) →  -- Inverse proportion
  digTime 30 = 4 := by sorry

end foundation_dig_time_l590_59003


namespace z_cube_coefficient_coefficient_is_17_l590_59048

/-- The coefficient of z^3 in the expansion of (3z^3 + 2z^2 - 4z - 1)(4z^4 + z^3 - 2z^2 + 3) is 17 -/
theorem z_cube_coefficient (z : ℝ) : 
  (3 * z^3 + 2 * z^2 - 4 * z - 1) * (4 * z^4 + z^3 - 2 * z^2 + 3) = 
  12 * z^7 + 11 * z^6 - 20 * z^5 - 8 * z^4 + 17 * z^3 + 8 * z^2 - 12 * z - 3 := by
  sorry

/-- The coefficient of z^3 in the expansion is 17 -/
theorem coefficient_is_17 : 
  ∃ (a b c d e f g h : ℝ), 
    (3 * z^3 + 2 * z^2 - 4 * z - 1) * (4 * z^4 + z^3 - 2 * z^2 + 3) = 
    a * z^7 + b * z^6 + c * z^5 + d * z^4 + 17 * z^3 + e * z^2 + f * z + g := by
  sorry

end z_cube_coefficient_coefficient_is_17_l590_59048


namespace exists_sequence_equal_one_l590_59009

/-- Represents a mathematical operation --/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Evaluates the result of applying operations to the given sequence of digits --/
def evaluate (digits : List Nat) (ops : List Operation) : Option Rat :=
  sorry

/-- Theorem stating that there exists a sequence of operations that results in 1 --/
theorem exists_sequence_equal_one :
  ∃ (ops : List Operation),
    evaluate [1, 2, 3, 4, 5, 6, 7, 8] ops = some 1 :=
  sorry

end exists_sequence_equal_one_l590_59009


namespace seven_balls_four_boxes_l590_59063

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by sorry

end seven_balls_four_boxes_l590_59063


namespace f_is_even_f_is_increasing_on_positive_l590_59040

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem stating that f is an even function
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by sorry

-- Theorem stating that f is monotonically increasing on (0, +∞)
theorem f_is_increasing_on_positive : ∀ x y : ℝ, 0 < x → x < y → f x < f y := by sorry

end f_is_even_f_is_increasing_on_positive_l590_59040


namespace eliana_steps_theorem_l590_59057

/-- The number of steps Eliana walked on the first day -/
def first_day_steps : ℕ := 200 + 300

/-- The number of steps Eliana walked on the second day -/
def second_day_steps : ℕ := (3 * first_day_steps) / 2

/-- The number of steps Eliana walked on the third day -/
def third_day_steps : ℕ := 2 * second_day_steps

/-- The total number of steps Eliana walked during the three days -/
def total_steps : ℕ := first_day_steps + second_day_steps + third_day_steps

theorem eliana_steps_theorem : total_steps = 2750 := by
  sorry

end eliana_steps_theorem_l590_59057


namespace semicircle_tangent_circle_and_triangle_l590_59005

/-- Given a semicircle with diameter AB and center O, where AO = OB = R,
    and two semicircles drawn over AO and BO, this theorem proves:
    1. The radius of the circle tangent to all three semicircles is R/3
    2. The sides of the triangle formed by the tangency points are 2R/5 and (R/5)√10 -/
theorem semicircle_tangent_circle_and_triangle (R : ℝ) (R_pos : R > 0) :
  ∃ (r a b : ℝ),
    r = R / 3 ∧
    2 * a = 2 * R / 5 ∧
    b = (R / 5) * Real.sqrt 10 :=
by sorry

end semicircle_tangent_circle_and_triangle_l590_59005


namespace total_discount_calculation_l590_59029

/-- Calculates the total discount percentage given a sale discount, coupon discount, and loyalty discount -/
theorem total_discount_calculation (original_price : ℝ) (sale_discount : ℝ) (coupon_discount : ℝ) (loyalty_discount : ℝ) :
  sale_discount = 1/3 →
  coupon_discount = 0.25 →
  loyalty_discount = 0.05 →
  let sale_price := original_price * (1 - sale_discount)
  let price_after_coupon := sale_price * (1 - coupon_discount)
  let final_price := price_after_coupon * (1 - loyalty_discount)
  (original_price - final_price) / original_price = 0.525 :=
by sorry

end total_discount_calculation_l590_59029


namespace unique_congruence_in_range_l590_59037

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end unique_congruence_in_range_l590_59037


namespace coefficient_x_squared_l590_59084

theorem coefficient_x_squared (x y : ℝ) : 
  let expansion := (x - 2 * y^3) * (x + 1/y)^5
  ∃ (a b c : ℝ), expansion = a * x^3 + (-20) * x^2 + b * x + c :=
sorry

end coefficient_x_squared_l590_59084


namespace baseball_team_size_l590_59092

/-- Given a baseball team with the following properties:
  * The team scored a total of 270 points in the year
  * 5 players averaged 50 points each
  * The remaining players averaged 5 points each
  Prove that the total number of players on the team is 9. -/
theorem baseball_team_size :
  ∀ (total_score : ℕ) (top_players : ℕ) (top_avg : ℕ) (rest_avg : ℕ),
  total_score = 270 →
  top_players = 5 →
  top_avg = 50 →
  rest_avg = 5 →
  ∃ (total_players : ℕ),
    total_players = top_players + (total_score - top_players * top_avg) / rest_avg ∧
    total_players = 9 :=
by sorry

end baseball_team_size_l590_59092


namespace field_length_calculation_l590_59078

theorem field_length_calculation (width length : ℝ) (pond_side : ℝ) : 
  length = 2 * width →
  pond_side = 8 →
  pond_side^2 = (1/8) * (length * width) →
  length = 32 := by
  sorry

end field_length_calculation_l590_59078


namespace six_coin_flip_probability_l590_59068

theorem six_coin_flip_probability : 
  let n : ℕ := 6  -- number of coins
  let p : ℚ := 1 / 2  -- probability of heads for a fair coin
  2 * p^n = 1 / 32 := by
  sorry

end six_coin_flip_probability_l590_59068


namespace sum_of_roots_quadratic_l590_59035

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 8*x - 15) → (∃ y : ℝ, y^2 = 8*y - 15 ∧ x + y = 8) :=
by sorry

end sum_of_roots_quadratic_l590_59035


namespace max_pons_is_nine_l590_59011

/-- Represents the items Bill can buy -/
inductive Item
  | Pack
  | Pin
  | Pon

/-- Represents the quantity of each item -/
structure Quantity where
  packs : ℕ
  pins : ℕ
  pons : ℕ

/-- Calculate the total cost of a given quantity of items -/
def totalCost (q : Quantity) : ℕ :=
  q.packs + 3 * q.pins + 7 * q.pons

/-- Check if the quantity satisfies the minimum purchase requirement -/
def satisfiesMinimum (q : Quantity) : Prop :=
  q.packs ≥ 2 ∧ q.pins ≥ 2 ∧ q.pons ≥ 2

/-- The main theorem stating that 9 is the maximum number of pons that can be purchased -/
theorem max_pons_is_nine :
  ∀ q : Quantity, satisfiesMinimum q → totalCost q = 75 →
  q.pons ≤ 9 ∧ ∃ q' : Quantity, satisfiesMinimum q' ∧ totalCost q' = 75 ∧ q'.pons = 9 :=
sorry

end max_pons_is_nine_l590_59011


namespace union_of_P_and_Q_l590_59041

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define the union of P and Q
def PUnionQ : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem union_of_P_and_Q : P ∪ Q = PUnionQ := by
  sorry

end union_of_P_and_Q_l590_59041


namespace inequality_proof_l590_59033

theorem inequality_proof (x y z : ℝ) (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) :
  (π/2) + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z >
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end inequality_proof_l590_59033


namespace cone_volume_l590_59060

/-- Given a cone whose lateral surface, when unrolled, forms a sector with radius 3 and 
    central angle 2π/3, prove that its volume is (2√2/3)π -/
theorem cone_volume (r l : ℝ) (h : ℝ) : 
  r = 1 → l = 3 → h = 2 * Real.sqrt 2 → 
  (1/3) * π * r^2 * h = (2 * Real.sqrt 2 / 3) * π := by
  sorry

end cone_volume_l590_59060
