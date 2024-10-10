import Mathlib

namespace new_person_weight_l3921_392187

theorem new_person_weight 
  (n : ℕ) 
  (initial_weight : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) :
  n = 9 →
  initial_weight = 65 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (n : ℝ) * weight_increase + replaced_weight = 87.5 :=
by sorry

end new_person_weight_l3921_392187


namespace two_segment_train_journey_time_l3921_392183

/-- Calculates the total time for a two-segment train journey -/
theorem two_segment_train_journey_time
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ)
  (h1 : distance1 = 80)
  (h2 : speed1 = 50)
  (h3 : distance2 = 150)
  (h4 : speed2 = 75)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  distance1 / speed1 + distance2 / speed2 = 3.6 := by
  sorry

end two_segment_train_journey_time_l3921_392183


namespace arithmetic_sequence_a5_l3921_392152

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 1 + a 9 = 10) :
  a 5 = 5 := by sorry

end arithmetic_sequence_a5_l3921_392152


namespace road_trip_ratio_l3921_392167

/-- Road trip problem -/
theorem road_trip_ratio : 
  ∀ (total michelle_dist katie_dist tracy_dist : ℕ),
  total = 1000 →
  michelle_dist = 294 →
  michelle_dist = 3 * katie_dist →
  tracy_dist = total - michelle_dist - katie_dist →
  (tracy_dist - 20) / michelle_dist = 2 :=
by
  sorry

end road_trip_ratio_l3921_392167


namespace equation_holds_iff_nonpositive_l3921_392165

theorem equation_holds_iff_nonpositive (a b : ℝ) : a = |b| → (a + b = 0 ↔ b ≤ 0) := by
  sorry

end equation_holds_iff_nonpositive_l3921_392165


namespace remainder_calculation_l3921_392107

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem remainder_calculation :
  rem (5/9 : ℚ) (-3/7 : ℚ) = -19/63 := by
  sorry

end remainder_calculation_l3921_392107


namespace arithmetic_mean_equidistant_l3921_392135

/-- The arithmetic mean of two real numbers is equidistant from both numbers. -/
theorem arithmetic_mean_equidistant (a b : ℝ) : 
  |((a + b) / 2) - a| = |b - ((a + b) / 2)| := by
  sorry

end arithmetic_mean_equidistant_l3921_392135


namespace square_difference_equality_l3921_392102

theorem square_difference_equality : 1005^2 - 995^2 - 1007^2 + 993^2 = -8000 := by
  sorry

end square_difference_equality_l3921_392102


namespace larger_number_proof_l3921_392188

theorem larger_number_proof (A B : ℝ) (h1 : A - B = 1650) (h2 : 0.075 * A = 0.125 * B) : A = 4125 := by
  sorry

end larger_number_proof_l3921_392188


namespace polynomial_inequality_l3921_392190

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Predicate to check if a function is a polynomial with integer coefficients -/
def is_int_polynomial (p : ℤ → ℤ) : Prop := sorry

theorem polynomial_inequality (p : IntPolynomial) (n : ℤ) 
  (h_poly : is_int_polynomial p)
  (h_ineq : p (-n) < p n ∧ p n < n) : 
  p (-n) < -n := by sorry

end polynomial_inequality_l3921_392190


namespace family_income_problem_l3921_392134

theorem family_income_problem (initial_avg : ℚ) (new_avg : ℚ) (deceased_income : ℚ) 
  (h1 : initial_avg = 735)
  (h2 : new_avg = 650)
  (h3 : deceased_income = 905) :
  ∃ n : ℕ, n > 0 ∧ n * initial_avg - (n - 1) * new_avg = deceased_income ∧ n = 3 := by
  sorry

end family_income_problem_l3921_392134


namespace scientific_notation_of_million_l3921_392130

theorem scientific_notation_of_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1000000 = a * (10 : ℝ) ^ n ∧ a = 1 ∧ n = 6 :=
by sorry

end scientific_notation_of_million_l3921_392130


namespace derivative_of_f_derivative_of_f_at_2_l3921_392150

-- Define the function f(x) = x^2 + x
def f (x : ℝ) : ℝ := x^2 + x

-- Theorem 1: The derivative of f(x) is 2x + 1
theorem derivative_of_f (x : ℝ) : deriv f x = 2 * x + 1 := by sorry

-- Theorem 2: The derivative of f(x) at x = 2 is 5
theorem derivative_of_f_at_2 : deriv f 2 = 5 := by sorry

end derivative_of_f_derivative_of_f_at_2_l3921_392150


namespace remainder_when_n_plus_3_and_n_plus_7_prime_l3921_392112

theorem remainder_when_n_plus_3_and_n_plus_7_prime (n : ℕ) 
  (h1 : Nat.Prime (n + 3)) 
  (h2 : Nat.Prime (n + 7)) : 
  n % 3 = 1 := by
sorry

end remainder_when_n_plus_3_and_n_plus_7_prime_l3921_392112


namespace nell_card_difference_l3921_392186

/-- Represents the number of cards Nell has -/
structure CardCounts where
  initial_baseball : Nat
  initial_ace : Nat
  final_baseball : Nat
  final_ace : Nat

/-- Calculates the difference between Ace cards and baseball cards -/
def ace_baseball_difference (counts : CardCounts) : Int :=
  counts.final_ace - counts.final_baseball

/-- Theorem stating the difference between Ace cards and baseball cards -/
theorem nell_card_difference (counts : CardCounts) 
  (h1 : counts.initial_baseball = 239)
  (h2 : counts.initial_ace = 38)
  (h3 : counts.final_baseball = 111)
  (h4 : counts.final_ace = 376) :
  ace_baseball_difference counts = 265 := by
  sorry

end nell_card_difference_l3921_392186


namespace sum_calculation_l3921_392138

theorem sum_calculation : 3 * 198 + 2 * 198 + 198 + 197 = 1385 := by
  sorry

end sum_calculation_l3921_392138


namespace unique_arrangements_of_zeros_and_ones_l3921_392194

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def permutations (n : ℕ) : ℕ := factorial n

def combinations (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem unique_arrangements_of_zeros_and_ones : 
  let total_digits : ℕ := 8
  let zeros : ℕ := 4
  let ones : ℕ := 4
  permutations total_digits / (permutations zeros * permutations ones) = 70 := by
  sorry

end unique_arrangements_of_zeros_and_ones_l3921_392194


namespace right_triangle_from_medians_l3921_392196

theorem right_triangle_from_medians (m₁ m₂ m₃ : ℝ) 
  (h₁ : m₁ = 5)
  (h₂ : m₂ = Real.sqrt 52)
  (h₃ : m₃ = Real.sqrt 73) :
  ∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ 
    m₁^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧
    m₂^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧
    m₃^2 = (2*a^2 + 2*b^2 - c^2) / 4 :=
by sorry

end right_triangle_from_medians_l3921_392196


namespace thousandths_place_of_seven_thirty_seconds_l3921_392124

theorem thousandths_place_of_seven_thirty_seconds (n : ℕ) : 
  (7 : ℚ) / 32 = n / 1000 + (8 : ℚ) / 1000 + m / 10000 → n < 9 ∧ 0 ≤ m ∧ m < 10 :=
by sorry

end thousandths_place_of_seven_thirty_seconds_l3921_392124


namespace z_extrema_l3921_392189

-- Define the function z(x,y)
def z (x y : ℝ) : ℝ := 2 * x^3 - 6 * x * y + 3 * y^2

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.2 ≤ 2 ∧ p.2 ≤ p.1^2 / 2}

-- State the theorem
theorem z_extrema :
  ∃ (max min : ℝ), max = 12 ∧ min = -1 ∧
  (∀ p ∈ R, z p.1 p.2 ≤ max) ∧
  (∀ p ∈ R, z p.1 p.2 ≥ min) ∧
  (∃ p ∈ R, z p.1 p.2 = max) ∧
  (∃ p ∈ R, z p.1 p.2 = min) :=
sorry

end z_extrema_l3921_392189


namespace complement_A_intersect_B_l3921_392192

def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {-2, -1} := by sorry

end complement_A_intersect_B_l3921_392192


namespace license_plate_combinations_license_plate_combinations_eq_187200_l3921_392179

theorem license_plate_combinations : ℕ :=
  let total_letters : ℕ := 26
  let letter_positions : ℕ := 4
  let digit_positions : ℕ := 3
  let repeated_letter_choices : ℕ := total_letters
  let non_repeated_letter_choices : ℕ := total_letters - 1
  let repeated_letter_arrangements : ℕ := Nat.choose letter_positions (letter_positions - 1)
  let first_digit_choices : ℕ := 10
  let second_digit_choices : ℕ := 9
  let third_digit_choices : ℕ := 8

  repeated_letter_choices * non_repeated_letter_choices * repeated_letter_arrangements *
  first_digit_choices * second_digit_choices * third_digit_choices

theorem license_plate_combinations_eq_187200 : license_plate_combinations = 187200 := by
  sorry

end license_plate_combinations_license_plate_combinations_eq_187200_l3921_392179


namespace min_sum_squares_l3921_392164

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≥ m :=
sorry

end min_sum_squares_l3921_392164


namespace positive_integer_solutions_of_2x_plus_y_eq_7_l3921_392139

def is_solution (x y : ℕ) : Prop := 2 * x + y = 7

theorem positive_integer_solutions_of_2x_plus_y_eq_7 :
  {(x, y) : ℕ × ℕ | is_solution x y ∧ x > 0 ∧ y > 0} = {(1, 5), (2, 3), (3, 1)} := by
  sorry

end positive_integer_solutions_of_2x_plus_y_eq_7_l3921_392139


namespace f_sin_A_lt_f_cos_B_l3921_392141

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) + f x = 0

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem f_sin_A_lt_f_cos_B
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_periodic : is_periodic_2 f)
  (h_increasing : is_increasing_on f 3 4)
  (A B : ℝ)
  (h_acute_A : 0 < A ∧ A < Real.pi / 2)
  (h_acute_B : 0 < B ∧ B < Real.pi / 2) :
  f (Real.sin A) < f (Real.cos B) :=
sorry

end f_sin_A_lt_f_cos_B_l3921_392141


namespace sphere_wedge_volume_l3921_392170

theorem sphere_wedge_volume (c : ℝ) (h1 : c = 16 * Real.pi) : 
  let r := c / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * r^3
  let wedge_volume := sphere_volume / 8
  wedge_volume = (256 / 3) * Real.pi := by sorry

end sphere_wedge_volume_l3921_392170


namespace remaining_practice_time_l3921_392125

/-- The total practice time in hours for the week -/
def total_practice_hours : ℝ := 7.5

/-- The number of days with known practice time -/
def known_practice_days : ℕ := 2

/-- The practice time in minutes for each of the known practice days -/
def practice_per_known_day : ℕ := 86

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem remaining_practice_time :
  hours_to_minutes total_practice_hours - (known_practice_days * practice_per_known_day) = 278 := by
  sorry

end remaining_practice_time_l3921_392125


namespace discount_difference_l3921_392161

theorem discount_difference (bill : ℝ) (d1 d2 d3 d4 : ℝ) :
  bill = 12000 ∧ d1 = 0.3 ∧ d2 = 0.2 ∧ d3 = 0.06 ∧ d4 = 0.04 →
  bill * (1 - d2) * (1 - d3) * (1 - d4) - bill * (1 - d1) = 263.04 := by
  sorry

end discount_difference_l3921_392161


namespace circles_intersection_sum_l3921_392104

/-- Given two circles intersecting at points (1, 3) and (m, 1), with their centers 
    on the line x - y + c/2 = 0, prove that m + c = 3 -/
theorem circles_intersection_sum (m c : ℝ) : 
  (∃ (circle1 circle2 : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ circle1 ∩ circle2 ↔ ((x = 1 ∧ y = 3) ∨ (x = m ∧ y = 1))) ∧
    (∃ (x1 y1 x2 y2 : ℝ), 
      (x1, y1) ∈ circle1 ∧ (x2, y2) ∈ circle2 ∧
      x1 - y1 + c/2 = 0 ∧ x2 - y2 + c/2 = 0)) →
  m + c = 3 := by
sorry

end circles_intersection_sum_l3921_392104


namespace xt_ty_ratio_is_one_l3921_392166

/-- Represents the shape described in the problem -/
structure Shape :=
  (total_squares : ℕ)
  (rectangle_squares : ℕ)
  (terrace_rows : ℕ)
  (terrace_squares_per_row : ℕ)

/-- Represents a line segment -/
structure LineSegment :=
  (length : ℝ)

/-- The problem setup -/
def problem_setup : Shape :=
  { total_squares := 12,
    rectangle_squares := 6,
    terrace_rows := 2,
    terrace_squares_per_row := 3 }

/-- The line RS that bisects the area horizontally -/
def RS : LineSegment :=
  { length := 6 }

/-- Theorem stating the ratio XT/TY = 1 -/
theorem xt_ty_ratio_is_one (shape : Shape) (rs : LineSegment) 
  (h1 : shape = problem_setup)
  (h2 : rs = RS)
  (h3 : rs.length = shape.total_squares / 2) :
  ∃ (xt ty : ℝ), xt = ty ∧ xt + ty = rs.length ∧ xt / ty = 1 :=
sorry

end xt_ty_ratio_is_one_l3921_392166


namespace system_solutions_l3921_392175

def system_of_equations (x y z : ℝ) : Prop :=
  3 * x * y - 5 * y * z - x * z = 3 * y ∧
  x * y + y * z = -y ∧
  -5 * x * y + 4 * y * z + x * z = -4 * y

theorem system_solutions :
  (∀ x : ℝ, system_of_equations x 0 0) ∧
  (∀ z : ℝ, system_of_equations 0 0 z) ∧
  system_of_equations 2 (-1/3) (-3) :=
sorry

end system_solutions_l3921_392175


namespace first_quadrant_trig_positivity_l3921_392197

theorem first_quadrant_trig_positivity (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  0 < Real.sin (2 * α) ∧ 0 < Real.tan (α / 2) :=
by sorry

end first_quadrant_trig_positivity_l3921_392197


namespace set_intersection_theorem_l3921_392147

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = -x^2 + 4}
def N : Set ℝ := {x | x > 0}

-- State the theorem
theorem set_intersection_theorem :
  M ∩ N = Set.Ioo 0 4 := by sorry

end set_intersection_theorem_l3921_392147


namespace sector_arc_length_ratio_l3921_392113

theorem sector_arc_length_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let sector_radius := 2 * r / 3
  let sector_area := 5 * circle_area / 27
  let circle_circumference := 2 * π * r
  ∃ α : ℝ, 
    sector_area = α * sector_radius^2 / 2 ∧ 
    (α * sector_radius) / circle_circumference = 5 / 18 :=
by
  sorry

end sector_arc_length_ratio_l3921_392113


namespace pencils_purchased_l3921_392169

theorem pencils_purchased (num_pens : ℕ) (total_cost : ℝ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : num_pens = 30)
  (h2 : total_cost = 510)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 12) :
  (total_cost - num_pens * pen_price) / pencil_price = 75 := by
  sorry

end pencils_purchased_l3921_392169


namespace cashier_error_l3921_392154

theorem cashier_error : ¬∃ (x y : ℕ), 9 * x + 15 * y = 485 := by
  sorry

end cashier_error_l3921_392154


namespace special_list_median_l3921_392182

/-- Represents the list where each integer n from 1 to 300 appears exactly n times -/
def special_list : List ℕ := sorry

/-- The total number of elements in the special list -/
def total_elements : ℕ := (300 * 301) / 2

/-- The position of the median in the special list -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- The median value of the special list -/
def median_value : ℕ := 212

theorem special_list_median :
  median_value = 212 := by sorry

end special_list_median_l3921_392182


namespace range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l3921_392155

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0
def q (x : ℝ) : Prop := ∃ m : ℝ, m ∈ Set.Ioo 1 2 ∧ x = (1/2)^(m-1)

-- Theorem for part (1)
theorem range_of_x_when_a_is_quarter (x : ℝ) :
  p x (1/4) ∧ q x → x ∈ Set.Ioo (1/2) (3/4) :=
sorry

-- Theorem for part (2)
theorem range_of_a_when_q_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) →
  a ∈ Set.Icc (1/3) (1/2) :=
sorry

end range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l3921_392155


namespace ball_placement_theorem_l3921_392162

/-- The number of ways to place balls in boxes under different conditions -/
theorem ball_placement_theorem :
  let n : ℕ := 4  -- number of balls and boxes
  -- 1. Distinct balls, exactly one empty box
  let distinct_one_empty : ℕ := n * (n - 1) * (n - 2) * 6
  -- 2. Identical balls, exactly one empty box
  let identical_one_empty : ℕ := n * (n - 1)
  -- 3. Distinct balls, empty boxes allowed
  let distinct_empty_allowed : ℕ := n^n
  -- 4. Identical balls, empty boxes allowed
  let identical_empty_allowed : ℕ := 
    1 + n * (n - 1) + (n * (n - 1) / 2) + n * (n - 1) / 2 + n
  ∀ (n : ℕ), n = 4 →
    (distinct_one_empty = 144) ∧
    (identical_one_empty = 12) ∧
    (distinct_empty_allowed = 256) ∧
    (identical_empty_allowed = 35) := by
  sorry

end ball_placement_theorem_l3921_392162


namespace valid_quadrilateral_set_l3921_392106

/-- A function that checks if a set of four line segments can form a valid quadrilateral. -/
def is_valid_quadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- Theorem stating that among the given sets, only (2,2,2) forms a valid quadrilateral with side length 5. -/
theorem valid_quadrilateral_set :
  ¬ is_valid_quadrilateral 1 1 1 5 ∧
  ¬ is_valid_quadrilateral 1 2 2 5 ∧
  ¬ is_valid_quadrilateral 1 1 7 5 ∧
  is_valid_quadrilateral 2 2 2 5 :=
by sorry

end valid_quadrilateral_set_l3921_392106


namespace value_of_b_l3921_392180

theorem value_of_b (b : ℚ) (h : b - b/4 = 5/2) : b = 10/3 := by
  sorry

end value_of_b_l3921_392180


namespace andromeda_distance_scientific_notation_l3921_392181

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The distance of the Andromeda galaxy from the Milky Way in light-years -/
def andromeda_distance : ℝ := 2500000

theorem andromeda_distance_scientific_notation :
  to_scientific_notation andromeda_distance = ScientificNotation.mk 2.5 6 (by norm_num) :=
sorry

end andromeda_distance_scientific_notation_l3921_392181


namespace ball_selection_properties_l3921_392178

structure BallSelection where
  total_balls : Nat
  red_balls : Nat
  white_balls : Nat
  balls_drawn : Nat

def P (event : Set ℝ) : ℝ := sorry

def A (bs : BallSelection) : Set ℝ := sorry
def B (bs : BallSelection) : Set ℝ := sorry
def D (bs : BallSelection) : Set ℝ := sorry

theorem ball_selection_properties (bs : BallSelection) 
  (h1 : bs.total_balls = 4)
  (h2 : bs.red_balls = 2)
  (h3 : bs.white_balls = 2)
  (h4 : bs.balls_drawn = 2) :
  (P (A bs ∩ B bs) = P (A bs) * P (B bs)) ∧
  (P (A bs) + P (D bs) = 1) ∧
  (P (B bs ∩ D bs) = P (B bs) * P (D bs)) := by
  sorry

end ball_selection_properties_l3921_392178


namespace volume_of_P₃_l3921_392142

/-- Represents a polyhedron in the sequence -/
structure Polyhedron where
  index : ℕ
  volume : ℚ

/-- Constructs the next polyhedron in the sequence -/
def next_polyhedron (P : Polyhedron) : Polyhedron :=
  { index := P.index + 1,
    volume := P.volume + (3/2)^P.index }

/-- The initial regular tetrahedron -/
def P₀ : Polyhedron :=
  { index := 0,
    volume := 1 }

/-- Generates the nth polyhedron in the sequence -/
def generate_polyhedron (n : ℕ) : Polyhedron :=
  match n with
  | 0 => P₀
  | n + 1 => next_polyhedron (generate_polyhedron n)

/-- The theorem to be proved -/
theorem volume_of_P₃ :
  (generate_polyhedron 3).volume = 23/4 := by
  sorry

end volume_of_P₃_l3921_392142


namespace missy_capacity_l3921_392101

/-- The number of insurance claims each agent can handle -/
structure AgentCapacity where
  jan : ℕ
  john : ℕ
  missy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (c : AgentCapacity) : Prop :=
  c.jan = 20 ∧
  c.john = c.jan + (c.jan * 30 / 100) ∧
  c.missy = c.john + 15

/-- The theorem to prove -/
theorem missy_capacity (c : AgentCapacity) :
  problem_conditions c → c.missy = 41 := by
  sorry

end missy_capacity_l3921_392101


namespace geometric_sequence_a3_value_l3921_392118

/-- A geometric sequence with first term 2 and satisfying a₃a₅ = 4a₆² has a₃ = 1 -/
theorem geometric_sequence_a3_value (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n, a n = 2 * r^(n-1))  -- {aₙ} is a geometric sequence
  → a 1 = 2                          -- a₁ = 2
  → a 3 * a 5 = 4 * (a 6)^2          -- a₃a₅ = 4a₆²
  → a 3 = 1                          -- a₃ = 1
:= by sorry

end geometric_sequence_a3_value_l3921_392118


namespace gcd_888_1147_l3921_392191

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l3921_392191


namespace inverse_proportion_product_l3921_392153

/-- Theorem: For points A(x₁, y₁) and B(x₂, y₂) on the graph of y = -3/x, 
    if x₁ * x₂ = 2, then y₁ * y₂ = 9/2 -/
theorem inverse_proportion_product (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = -3 / x₁) 
    (h2 : y₂ = -3 / x₂) 
    (h3 : x₁ * x₂ = 2) : 
  y₁ * y₂ = 9/2 := by
  sorry

end inverse_proportion_product_l3921_392153


namespace equation_solutions_l3921_392151

theorem equation_solutions :
  (∀ x : ℝ, (2*x - 1)^2 - 25 = 0 ↔ x = 3 ∨ x = -2) ∧
  (∀ x : ℝ, (1/3) * (x + 3)^3 - 9 = 0 ↔ x = 0) := by
  sorry

end equation_solutions_l3921_392151


namespace average_disk_space_per_hour_l3921_392199

/-- Proves that the average disk space per hour of music in a library
    containing 12 days of music and occupying 16,000 megabytes,
    rounded to the nearest whole number, is 56 megabytes. -/
theorem average_disk_space_per_hour (days : ℕ) (total_space : ℕ) 
  (h1 : days = 12) (h2 : total_space = 16000) : 
  round ((total_space : ℝ) / (days * 24)) = 56 := by
  sorry

#check average_disk_space_per_hour

end average_disk_space_per_hour_l3921_392199


namespace nathan_gave_six_apples_l3921_392184

/-- The number of apples Nathan gave to Annie -/
def apples_from_nathan (initial_apples final_apples : ℕ) : ℕ :=
  final_apples - initial_apples

theorem nathan_gave_six_apples :
  apples_from_nathan 6 12 = 6 := by
  sorry

end nathan_gave_six_apples_l3921_392184


namespace correct_divisor_l3921_392127

theorem correct_divisor (X D : ℕ) 
  (h1 : X % D = 0)
  (h2 : X / (D - 12) = 42)
  (h3 : X / D = 24) :
  D = 28 := by
  sorry

end correct_divisor_l3921_392127


namespace yellow_leaves_count_l3921_392148

theorem yellow_leaves_count (thursday_leaves friday_leaves : ℕ) 
  (brown_percent green_percent : ℚ) :
  thursday_leaves = 12 →
  friday_leaves = 13 →
  brown_percent = 1/5 →
  green_percent = 1/5 →
  (thursday_leaves + friday_leaves : ℚ) * (1 - brown_percent - green_percent) = 15 :=
by sorry

end yellow_leaves_count_l3921_392148


namespace greatest_integer_solution_l3921_392137

-- Define the equation
def equation (x : ℝ) : Prop := 2 * Real.log x = 7 - 2 * x

-- Define the inequality
def inequality (n : ℤ) : Prop := (n : ℝ) - 2 < (n : ℝ)

theorem greatest_integer_solution :
  (∃ x : ℝ, equation x) →
  (∃ n : ℤ, inequality n ∧ ∀ m : ℤ, inequality m → m ≤ n) ∧
  (∀ n : ℤ, inequality n → n ≤ 4) :=
by sorry

end greatest_integer_solution_l3921_392137


namespace vote_participation_l3921_392173

theorem vote_participation (veggie_percentage : ℝ) (veggie_votes : ℕ) (total_students : ℕ) : 
  veggie_percentage = 0.28 →
  veggie_votes = 280 →
  (veggie_percentage * total_students : ℝ) = veggie_votes →
  total_students = 1000 := by
sorry

end vote_participation_l3921_392173


namespace tens_digit_of_sum_is_zero_l3921_392120

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100) = (n % 10) - 1 ∧
  ((n / 10) % 10) = (n % 10) + 3

def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

theorem tens_digit_of_sum_is_zero (n : ℕ) (h : is_valid_number n) :
  ((n + reverse_number n) / 10) % 10 = 0 :=
sorry

end tens_digit_of_sum_is_zero_l3921_392120


namespace incircle_radius_l3921_392177

/-- The ellipse with semi-major axis 4 and semi-minor axis 1 -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2 = 1

/-- The incircle of the inscribed triangle ABC -/
def incircle (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2

/-- A is the left vertex of the ellipse -/
def A : ℝ × ℝ := (-4, 0)

/-- Theorem: The radius of the incircle is 5 -/
theorem incircle_radius : ∃ (r : ℝ), 
  (∀ x y, incircle x y r → ellipse x y) ∧ 
  (incircle A.1 A.2 r) ∧ 
  r = 5 := by sorry

end incircle_radius_l3921_392177


namespace marks_difference_l3921_392156

/-- Represents the marks of students a, b, c, d, and e. -/
structure Marks where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The conditions of the problem and the theorem to prove. -/
theorem marks_difference (m : Marks) : m.e - m.d = 3 :=
  by
  have h1 : m.a + m.b + m.c = 48 * 3 := by sorry
  have h2 : m.a + m.b + m.c + m.d = 47 * 4 := by sorry
  have h3 : m.b + m.c + m.d + m.e = 48 * 4 := by sorry
  have h4 : m.a = 43 := by sorry
  have h5 : m.e > m.d := by sorry
  sorry

end marks_difference_l3921_392156


namespace min_distance_curve_line_l3921_392132

theorem min_distance_curve_line (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) :
  ∃ (x y : ℝ), ∀ (a' b' c' d' : ℝ),
    Real.log (b' + 1) + a' - 3 * b' = 0 →
    2 * d' - c' + Real.sqrt 5 = 0 →
    (a' - c')^2 + (b' - d')^2 ≥ 1 ∧
    (x - y)^2 + (0 - Real.sqrt 5)^2 = 1 :=
sorry

end min_distance_curve_line_l3921_392132


namespace sum_of_odd_three_digit_numbers_l3921_392119

/-- The set of odd digits -/
def OddDigits : Finset ℕ := {1, 3, 5, 7, 9}

/-- A three-digit number with odd digits -/
structure OddThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_in_odd_digits : hundreds ∈ OddDigits
  tens_in_odd_digits : tens ∈ OddDigits
  units_in_odd_digits : units ∈ OddDigits

/-- The set of all possible odd three-digit numbers -/
def AllOddThreeDigitNumbers : Finset OddThreeDigitNumber := sorry

/-- The value of an odd three-digit number -/
def value (n : OddThreeDigitNumber) : ℕ := 100 * n.hundreds + 10 * n.tens + n.units

/-- The theorem stating the sum of all odd three-digit numbers -/
theorem sum_of_odd_three_digit_numbers :
  (AllOddThreeDigitNumbers.sum value) = 69375 := by sorry

end sum_of_odd_three_digit_numbers_l3921_392119


namespace probability_at_least_one_vowel_l3921_392128

def set1 : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def set2 : Finset Char := {'k', 'l', 'm', 'n', 'o', 'p'}

def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

def isVowel (c : Char) : Bool := c ∈ vowels

theorem probability_at_least_one_vowel :
  let prob_no_vowel_set1 := (set1.filter (λ c => ¬isVowel c)).card / set1.card
  let prob_no_vowel_set2 := (set2.filter (λ c => ¬isVowel c)).card / set2.card
  1 - (prob_no_vowel_set1 * prob_no_vowel_set2) = 3/5 := by
  sorry

end probability_at_least_one_vowel_l3921_392128


namespace baylor_payment_multiple_l3921_392195

theorem baylor_payment_multiple :
  let initial_amount : ℕ := 4000
  let first_client_payment : ℕ := initial_amount / 2
  let second_client_payment : ℕ := first_client_payment + (2 * first_client_payment) / 5
  let combined_payment : ℕ := first_client_payment + second_client_payment
  let final_total : ℕ := 18400
  let third_client_multiple : ℕ := (final_total - initial_amount - combined_payment) / combined_payment
  third_client_multiple = 2 := by sorry

end baylor_payment_multiple_l3921_392195


namespace highway_intersection_probability_l3921_392171

theorem highway_intersection_probability (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  let p_enter := 1 / n
  let p_exit := 1 / n
  (k - 1) * (n - k) * p_enter * p_exit +
  p_enter * (n - k) * p_exit +
  (k - 1) * p_enter * p_exit =
  (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 := by
  sorry


end highway_intersection_probability_l3921_392171


namespace investment_difference_l3921_392131

/-- Represents the final value of an investment given its initial value and growth factor -/
def final_value (initial : ℝ) (growth : ℝ) : ℝ := initial * growth

/-- Theorem: Given the initial investments and their changes in value, 
    the difference between Jackson's final investment value and 
    the combined final investment values of Brandon and Meagan is $850 -/
theorem investment_difference : 
  let jackson_initial := (500 : ℝ)
  let brandon_initial := (500 : ℝ)
  let meagan_initial := (700 : ℝ)
  let jackson_growth := (4 : ℝ)
  let brandon_growth := (0.2 : ℝ)
  let meagan_growth := (1.5 : ℝ)
  final_value jackson_initial jackson_growth - 
  (final_value brandon_initial brandon_growth + final_value meagan_initial meagan_growth) = 850 := by
  sorry


end investment_difference_l3921_392131


namespace original_polygon_sides_l3921_392111

theorem original_polygon_sides (n : ℕ) : 
  (∃ m : ℕ, (m - 2) * 180 = 1620 ∧ 
  (n = m + 1 ∨ n = m ∨ n = m - 1)) →
  (n = 10 ∨ n = 11 ∨ n = 12) :=
by sorry

end original_polygon_sides_l3921_392111


namespace twelve_point_zero_six_million_scientific_notation_l3921_392198

theorem twelve_point_zero_six_million_scientific_notation :
  (12.06 : ℝ) * 1000000 = 1.206 * (10 ^ 7) := by
  sorry

end twelve_point_zero_six_million_scientific_notation_l3921_392198


namespace brick_in_box_probability_l3921_392144

/-- A set of six distinct numbers from 1 to 500 -/
def SixNumbers : Type := { s : Finset ℕ // s.card = 6 ∧ ∀ n ∈ s, 1 ≤ n ∧ n ≤ 500 }

/-- The three largest numbers from a set of six numbers -/
def largestThree (s : SixNumbers) : Finset ℕ :=
  sorry

/-- The three smallest numbers from a set of six numbers -/
def smallestThree (s : SixNumbers) : Finset ℕ :=
  sorry

/-- Whether a brick with given dimensions fits in a box with given dimensions -/
def fits (brick box : Finset ℕ) : Prop :=
  sorry

/-- The probability of a brick fitting in a box -/
def fitProbability : ℚ :=
  sorry

theorem brick_in_box_probability :
  fitProbability = 1 / 4 := by sorry

end brick_in_box_probability_l3921_392144


namespace least_sum_m_n_l3921_392121

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m + n) 330 = 1) ∧ 
  (∃ (k : ℕ), m^(m : ℕ) = k * n^(n : ℕ)) ∧ 
  (¬∃ (l : ℕ), m = l * n) ∧
  (∀ (p q : ℕ+), 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ (k : ℕ), p^(p : ℕ) = k * q^(q : ℕ)) → 
    (¬∃ (l : ℕ), p = l * q) → 
    (m + n ≤ p + q)) ∧
  (m + n = 182) := by
sorry

end least_sum_m_n_l3921_392121


namespace derivative_at_one_l3921_392185

open Real

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x * (deriv f 1) + log x) :
  deriv f 1 = -1 := by
  sorry

end derivative_at_one_l3921_392185


namespace pigeonhole_zodiac_signs_pigeonhole_birthday_weekday_l3921_392149

/-- The number of zodiac signs -/
def num_zodiac_signs : ℕ := 12

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The minimum number of employees needed to ensure at least two have the same zodiac sign -/
def min_employees_same_sign : ℕ := num_zodiac_signs + 1

/-- The minimum number of employees needed to ensure at least four have birthdays on the same day of the week -/
def min_employees_same_day : ℕ := days_in_week * 3 + 1

theorem pigeonhole_zodiac_signs :
  min_employees_same_sign = 13 :=
sorry

theorem pigeonhole_birthday_weekday :
  min_employees_same_day = 22 :=
sorry

end pigeonhole_zodiac_signs_pigeonhole_birthday_weekday_l3921_392149


namespace triangle_area_l3921_392145

/-- Given a triangle ABC with angle A = 60°, side b = 4, and side a = 2√3,
    prove that its area is 2√3. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 →  -- 60° in radians
  b = 4 → 
  a = 2 * Real.sqrt 3 → 
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 := by
  sorry

end triangle_area_l3921_392145


namespace checkerboard_coverage_l3921_392140

/-- A checkerboard is a rectangular grid of squares. -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ
  missing_squares : ℕ

/-- A domino covers exactly two adjacent squares. -/
def domino_area : ℕ := 2

/-- The total number of squares in a checkerboard. -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols - board.missing_squares

/-- A checkerboard can be completely covered by dominoes if and only if
    it has an even number of squares. -/
theorem checkerboard_coverage (board : Checkerboard) :
  ∃ (n : ℕ), total_squares board = n * domino_area ↔ Even (total_squares board) := by
  sorry

end checkerboard_coverage_l3921_392140


namespace tangent_point_x_coordinate_l3921_392160

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h1 : ∀ x, (deriv (f a)) x = (deriv (f a)) (-x))  -- f' is an odd function
  (h2 : ∃ x, (deriv (f a)) x = 3/2)  -- There exists a point with slope 3/2
  : ∃ x, (deriv (f a)) x = 3/2 ∧ x = Real.log ((3 + Real.sqrt 17) / 4) :=
sorry

end tangent_point_x_coordinate_l3921_392160


namespace stratified_sampling_best_l3921_392159

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | RandomNumberTable
  | Stratified

/-- Represents a high school population -/
structure HighSchoolPopulation where
  grades : List Nat
  students : Nat

/-- Represents a survey goal -/
inductive SurveyGoal
  | PsychologicalPressure

/-- Determines the best sampling method given a high school population and survey goal -/
def bestSamplingMethod (population : HighSchoolPopulation) (goal : SurveyGoal) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the best method for the given scenario -/
theorem stratified_sampling_best 
  (population : HighSchoolPopulation) 
  (h1 : population.grades.length > 1) 
  (goal : SurveyGoal) 
  (h2 : goal = SurveyGoal.PsychologicalPressure) : 
  bestSamplingMethod population goal = SamplingMethod.Stratified :=
sorry

end stratified_sampling_best_l3921_392159


namespace intersection_of_M_and_N_l3921_392146

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := by
  sorry

end intersection_of_M_and_N_l3921_392146


namespace mary_initial_nickels_l3921_392123

/-- The number of nickels Mary initially had -/
def initial_nickels : ℕ := sorry

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad : ℕ := 5

/-- The total number of nickels Mary has now -/
def total_nickels : ℕ := 12

/-- Theorem stating that Mary initially had 7 nickels -/
theorem mary_initial_nickels : 
  initial_nickels = 7 :=
by
  sorry

end mary_initial_nickels_l3921_392123


namespace negation_of_universal_proposition_l3921_392115

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) := by
  sorry

end negation_of_universal_proposition_l3921_392115


namespace sum_interior_angles_formula_sum_interior_angles_correct_l3921_392163

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180

theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2 : ℝ) * 180 :=
by
  sorry

/-- The sum of interior angles of a triangle -/
axiom triangle_sum : sum_interior_angles 3 = 180

/-- The sum of interior angles of a quadrilateral -/
axiom quadrilateral_sum : sum_interior_angles 4 = 360

theorem sum_interior_angles_correct (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2 : ℝ) * 180 :=
by
  sorry

end sum_interior_angles_formula_sum_interior_angles_correct_l3921_392163


namespace solution_set_part1_range_of_a_part2_l3921_392122

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 3 x ≥ x + 9} = {x : ℝ | x < -11/3 ∨ x > 7} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ (Set.Icc 0 1), f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 2 :=
sorry

end solution_set_part1_range_of_a_part2_l3921_392122


namespace chairs_to_remove_chair_adjustment_problem_l3921_392114

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_students : ℕ) : ℕ :=
  let min_chairs_needed := ((expected_students + chairs_per_row - 1) / chairs_per_row) * chairs_per_row
  initial_chairs - min_chairs_needed

theorem chair_adjustment_problem :
  chairs_to_remove 169 13 100 = 65 := by
  sorry

end chairs_to_remove_chair_adjustment_problem_l3921_392114


namespace positions_after_307_moves_l3921_392103

/-- Represents the positions of the cat -/
inductive CatPosition
  | Top
  | TopRight
  | BottomRight
  | Bottom
  | BottomLeft
  | TopLeft

/-- Represents the positions of the mouse -/
inductive MousePosition
  | Top
  | TopRight
  | BetweenTopRightAndBottomRight
  | BottomRight
  | BetweenBottomRightAndBottom
  | Bottom
  | BottomLeft
  | BetweenBottomLeftAndTopLeft
  | TopLeft
  | BetweenTopLeftAndTop

/-- The number of hexagons in the larger hexagon -/
def numHexagons : Nat := 6

/-- The number of segments the mouse moves through -/
def numMouseSegments : Nat := 12

/-- Calculates the cat's position after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % numHexagons with
  | 0 => CatPosition.TopLeft
  | 1 => CatPosition.Top
  | 2 => CatPosition.TopRight
  | 3 => CatPosition.BottomRight
  | 4 => CatPosition.Bottom
  | 5 => CatPosition.BottomLeft
  | _ => CatPosition.Top  -- This case should never occur

/-- Calculates the mouse's position after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % numMouseSegments with
  | 0 => MousePosition.Top
  | 1 => MousePosition.BetweenTopLeftAndTop
  | 2 => MousePosition.TopLeft
  | 3 => MousePosition.BetweenBottomLeftAndTopLeft
  | 4 => MousePosition.BottomLeft
  | 5 => MousePosition.Bottom
  | 6 => MousePosition.BetweenBottomRightAndBottom
  | 7 => MousePosition.BottomRight
  | 8 => MousePosition.BetweenTopRightAndBottomRight
  | 9 => MousePosition.TopRight
  | 10 => MousePosition.Top
  | 11 => MousePosition.BetweenTopLeftAndTop
  | _ => MousePosition.Top  -- This case should never occur

theorem positions_after_307_moves :
  catPositionAfterMoves 307 = CatPosition.Top ∧
  mousePositionAfterMoves 307 = MousePosition.BetweenBottomRightAndBottom :=
by sorry

end positions_after_307_moves_l3921_392103


namespace all_items_used_as_money_l3921_392108

structure MoneyItem where
  name : String
  used_as_money : Bool

def gold : MoneyItem := { name := "gold", used_as_money := true }
def stones : MoneyItem := { name := "stones", used_as_money := true }
def horses : MoneyItem := { name := "horses", used_as_money := true }
def dried_fish : MoneyItem := { name := "dried fish", used_as_money := true }
def mollusk_shells : MoneyItem := { name := "mollusk shells", used_as_money := true }

def money_items : List MoneyItem := [gold, stones, horses, dried_fish, mollusk_shells]

theorem all_items_used_as_money :
  (∀ item ∈ money_items, item.used_as_money = true) →
  (¬ ∃ item ∈ money_items, item.used_as_money = false) := by
  sorry

end all_items_used_as_money_l3921_392108


namespace bus_passengers_l3921_392100

theorem bus_passengers (total : ℕ) (women_fraction : ℚ) (standing_men_fraction : ℚ) 
  (h1 : total = 48)
  (h2 : women_fraction = 2/3)
  (h3 : standing_men_fraction = 1/8) : 
  ↑total * (1 - women_fraction) * (1 - standing_men_fraction) = 14 := by
  sorry

end bus_passengers_l3921_392100


namespace winning_percentage_approx_l3921_392172

/-- Represents the votes received by each candidate in an election -/
structure ElectionResults where
  candidates : Fin 3 → ℕ
  candidate1_votes : candidates 0 = 3000
  candidate2_votes : candidates 1 = 5000
  candidate3_votes : candidates 2 = 20000

/-- Calculates the total number of votes in the election -/
def totalVotes (results : ElectionResults) : ℕ :=
  (results.candidates 0) + (results.candidates 1) + (results.candidates 2)

/-- Finds the maximum number of votes received by any candidate -/
def maxVotes (results : ElectionResults) : ℕ :=
  max (results.candidates 0) (max (results.candidates 1) (results.candidates 2))

/-- Calculates the percentage of votes received by the winning candidate -/
def winningPercentage (results : ElectionResults) : ℚ :=
  (maxVotes results : ℚ) / (totalVotes results : ℚ) * 100

/-- Theorem stating that the winning percentage is approximately 71.43% -/
theorem winning_percentage_approx (results : ElectionResults) :
  ∃ ε > 0, abs (winningPercentage results - 71.43) < ε :=
sorry

end winning_percentage_approx_l3921_392172


namespace sequence_eventually_constant_l3921_392174

/-- A sequence of non-negative integers satisfying the given conditions -/
def Sequence (m : ℕ+) := { a : ℕ → ℕ // 
  a 0 = m ∧ 
  (∀ n : ℕ, n ≥ 1 → a n ≤ n) ∧
  (∀ n : ℕ+, (n : ℕ) ∣ (Finset.range n).sum (λ i => a i)) }

/-- The main theorem -/
theorem sequence_eventually_constant (m : ℕ+) (a : Sequence m) : 
  ∃ M : ℕ, ∀ n ≥ M, a.val n = a.val M :=
sorry

end sequence_eventually_constant_l3921_392174


namespace count_even_numbers_between_150_and_350_l3921_392193

theorem count_even_numbers_between_150_and_350 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ 150 < n ∧ n < 350) (Finset.range 350)).card = 99 := by
  sorry

end count_even_numbers_between_150_and_350_l3921_392193


namespace symmetry_complex_plane_l3921_392176

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal -/
def symmetric_to_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem symmetry_complex_plane (z₁ z₂ : ℂ) :
  symmetric_to_imaginary_axis z₁ z₂ → z₁ = 1 + I → z₂ = -1 + I := by
  sorry

#check symmetry_complex_plane

end symmetry_complex_plane_l3921_392176


namespace polynomial_with_arithmetic_progression_roots_l3921_392105

/-- A polynomial of the form x^4 + mx^2 + nx + 144 with four distinct real roots in arithmetic progression has m = -40 -/
theorem polynomial_with_arithmetic_progression_roots (m n : ℝ) : 
  (∃ (b d : ℝ) (h_distinct : d ≠ 0), 
    (∀ x : ℝ, x^4 + m*x^2 + n*x + 144 = (x - b)*(x - (b + d))*(x - (b + 2*d))*(x - (b + 3*d))) ∧
    (b ≠ b + d) ∧ (b + d ≠ b + 2*d) ∧ (b + 2*d ≠ b + 3*d)) →
  m = -40 := by
sorry

end polynomial_with_arithmetic_progression_roots_l3921_392105


namespace bakery_purchase_maximization_l3921_392109

/-- Represents the problem of maximizing purchases at a bakery --/
theorem bakery_purchase_maximization 
  (total_money : ℚ)
  (pastry_cost : ℚ)
  (coffee_cost : ℚ)
  (discount : ℚ)
  (discount_threshold : ℕ)
  (h1 : total_money = 50)
  (h2 : pastry_cost = 6)
  (h3 : coffee_cost = (3/2))
  (h4 : discount = (1/2))
  (h5 : discount_threshold = 5) :
  ∃ (pastries coffee : ℕ),
    (pastries > discount_threshold → 
      pastries * (pastry_cost - discount) + coffee * coffee_cost ≤ total_money) ∧
    (pastries ≤ discount_threshold → 
      pastries * pastry_cost + coffee * coffee_cost ≤ total_money) ∧
    pastries + coffee = 9 ∧
    ∀ (p c : ℕ), 
      ((p > discount_threshold → 
        p * (pastry_cost - discount) + c * coffee_cost ≤ total_money) ∧
      (p ≤ discount_threshold → 
        p * pastry_cost + c * coffee_cost ≤ total_money)) →
      p + c ≤ 9 := by
sorry


end bakery_purchase_maximization_l3921_392109


namespace ratio_of_divisor_sums_l3921_392157

def M : ℕ := 36 * 36 * 65 * 280

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M) * 254 = sum_even_divisors M := by sorry

end ratio_of_divisor_sums_l3921_392157


namespace max_missed_questions_to_pass_l3921_392110

theorem max_missed_questions_to_pass (total_questions : ℕ) (passing_percentage : ℚ) 
  (h1 : total_questions = 40)
  (h2 : passing_percentage = 75/100) : 
  ∃ (max_missed : ℕ), max_missed = 10 ∧ 
    (total_questions - max_missed : ℚ) / total_questions ≥ passing_percentage :=
by sorry

end max_missed_questions_to_pass_l3921_392110


namespace wendys_candy_boxes_l3921_392158

/-- Proves that Wendy had 2 boxes of candy given the problem conditions -/
theorem wendys_candy_boxes :
  ∀ (brother_candy : ℕ) (pieces_per_box : ℕ) (total_candy : ℕ) (wendys_boxes : ℕ),
    brother_candy = 6 →
    pieces_per_box = 3 →
    total_candy = 12 →
    total_candy = brother_candy + (wendys_boxes * pieces_per_box) →
    wendys_boxes = 2 := by
  sorry

end wendys_candy_boxes_l3921_392158


namespace triangle_problem_l3921_392168

theorem triangle_problem (A B C : Real) (a b c : Real) :
  b = 3 * Real.sqrt 2 →
  Real.cos A = Real.sqrt 6 / 3 →
  B = A + π / 2 →
  a = 3 ∧ Real.cos (2 * C) = 7 / 9 := by
  sorry

end triangle_problem_l3921_392168


namespace box_area_l3921_392129

theorem box_area (V : ℝ) (A2 A3 : ℝ) (hV : V = 720) (hA2 : A2 = 72) (hA3 : A3 = 60) :
  ∃ (L W H : ℝ), L > 0 ∧ W > 0 ∧ H > 0 ∧ 
    L * W * H = V ∧
    W * H = A2 ∧
    L * H = A3 ∧
    L * W = 120 :=
by sorry

end box_area_l3921_392129


namespace f_max_value_l3921_392116

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.sin x) + Real.sin (x - Real.sin x) + (Real.pi / 2 - 2) * Real.sin (Real.sin x)

theorem f_max_value :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (M = (Real.pi - 2) / Real.sqrt 2) :=
sorry

end f_max_value_l3921_392116


namespace customer_payment_proof_l3921_392136

-- Define the cost price of the computer table
def cost_price : ℕ := 6500

-- Define the markup percentage
def markup_percentage : ℚ := 30 / 100

-- Define the function to calculate the selling price
def selling_price (cost : ℕ) (markup : ℚ) : ℚ :=
  cost * (1 + markup)

-- Theorem statement
theorem customer_payment_proof :
  selling_price cost_price markup_percentage = 8450 := by
  sorry

end customer_payment_proof_l3921_392136


namespace max_value_xy_8x_y_l3921_392143

theorem max_value_xy_8x_y (x y : ℝ) (h : x^2 + y^2 = 20) :
  x * y + 8 * x + y ≤ 42 := by
  sorry

end max_value_xy_8x_y_l3921_392143


namespace sum_A_B_linear_combo_A_B_diff_A_B_specific_l3921_392117

-- Define A and B as functions of a and b
def A (a b : ℚ) : ℚ := 4 * a^2 * b - 3 * a * b + b^2
def B (a b : ℚ) : ℚ := a^2 - 3 * a^2 * b + 3 * a * b - b^2

-- Theorem 1: A + B = a² + a²b
theorem sum_A_B (a b : ℚ) : A a b + B a b = a^2 + a^2 * b := by sorry

-- Theorem 2: 3A + 4B = 4a² + 3ab - b²
theorem linear_combo_A_B (a b : ℚ) : 3 * A a b + 4 * B a b = 4 * a^2 + 3 * a * b - b^2 := by sorry

-- Theorem 3: A - B = -63/8 when a = 2 and b = -1/4
theorem diff_A_B_specific : A 2 (-1/4) - B 2 (-1/4) = -63/8 := by sorry

end sum_A_B_linear_combo_A_B_diff_A_B_specific_l3921_392117


namespace thirteenth_service_same_as_first_l3921_392133

/-- Represents the months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- The number of months between services -/
def service_interval : Nat := 7

/-- The month of the first service -/
def first_service_month : Month := Month.March

/-- The number of the service we're interested in -/
def target_service_number : Nat := 13

/-- Calculates the number of months between two service numbers -/
def months_between_services (start service_number : Nat) : Nat :=
  service_interval * (service_number - start)

/-- Determines if two services occur in the same month -/
def same_month (start target : Nat) : Prop :=
  (months_between_services start target) % 12 = 0

theorem thirteenth_service_same_as_first :
  same_month 1 target_service_number := by sorry

end thirteenth_service_same_as_first_l3921_392133


namespace smallest_perimeter_of_special_triangle_l3921_392126

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ 
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1

/-- The main theorem -/
theorem smallest_perimeter_of_special_triangle :
  ∀ a b c : ℕ,
    areConsecutiveOddPrimes a b c →
    isPrime (a + b + c) →
    a + b + c ≥ 41 :=
  sorry

end smallest_perimeter_of_special_triangle_l3921_392126
