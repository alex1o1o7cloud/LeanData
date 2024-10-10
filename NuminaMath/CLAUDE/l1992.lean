import Mathlib

namespace smallest_n_congruence_l1992_199204

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 5 * n % 26 = 2024 % 26 ∧ ∀ (m : ℕ), m > 0 ∧ m < n → 5 * m % 26 ≠ 2024 % 26 :=
by
  -- The proof goes here
  sorry

end smallest_n_congruence_l1992_199204


namespace arithmetic_geometric_ratio_l1992_199218

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

/-- The main theorem -/
theorem arithmetic_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 1) (a 3) (a 9)) :
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = 3 / 4 := by
  sorry

end arithmetic_geometric_ratio_l1992_199218


namespace max_gcd_value_l1992_199200

def a (n : ℕ+) : ℕ := 121 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_value :
  (∃ (k : ℕ+), d k = 99) ∧ (∀ (n : ℕ+), d n ≤ 99) := by
  sorry

end max_gcd_value_l1992_199200


namespace three_zeros_properties_l1992_199268

variable (a : ℝ) (x₁ x₂ x₃ : ℝ)

def f (x : ℝ) := a * (2 * x - 1) * abs (x + 1) - 2 * x - 1

theorem three_zeros_properties 
  (h_zeros : f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0)
  (h_order : x₁ < x₂ ∧ x₂ < x₃) :
  (1 / a < x₃ ∧ x₃ < 1 / a + 1 / x₃) ∧ a * (x₂ - x₁) < 1 := by
  sorry

end three_zeros_properties_l1992_199268


namespace triangle_count_l1992_199206

theorem triangle_count (num_circles : ℕ) (num_triangles : ℕ) : 
  num_circles = 5 → num_triangles = 2 * num_circles → num_triangles = 10 := by
  sorry

end triangle_count_l1992_199206


namespace insurance_premium_theorem_l1992_199227

/-- Represents an insurance policy -/
structure InsurancePolicy where
  payout : ℝ  -- The amount paid out if the event occurs
  probability : ℝ  -- The probability of the event occurring
  premium : ℝ  -- The premium charged to the customer

/-- Calculates the expected revenue for an insurance policy -/
def expectedRevenue (policy : InsurancePolicy) : ℝ :=
  policy.premium - policy.payout * policy.probability

/-- Theorem: Given an insurance policy with payout 'a' and event probability 'p',
    if the company wants an expected revenue of 10% of 'a',
    then the required premium is a(p + 0.1) -/
theorem insurance_premium_theorem (a p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let policy := InsurancePolicy.mk a p (a * (p + 0.1))
  expectedRevenue policy = 0.1 * a := by
  sorry

#check insurance_premium_theorem

end insurance_premium_theorem_l1992_199227


namespace fish_in_pond_l1992_199209

theorem fish_in_pond (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 30)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish) :
  total_fish = 750 :=
by sorry

#check fish_in_pond

end fish_in_pond_l1992_199209


namespace extreme_value_implies_params_l1992_199234

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem extreme_value_implies_params (a b : ℝ) :
  (f a b 1 = -2) ∧ (f' a b 1 = 0) → a = 1 ∧ b = -3 := by
  sorry

end extreme_value_implies_params_l1992_199234


namespace quadratic_inequality_range_sufficient_condition_range_l1992_199236

-- Part I
theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 3*a*x + 9 > 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Part II
theorem sufficient_condition_range (m : ℝ) :
  ((∀ x : ℝ, x^2 + 2*x - 8 < 0 → x - m > 0) ∧
   (∃ x : ℝ, x - m > 0 ∧ x^2 + 2*x - 8 ≥ 0)) →
  m ≤ -4 :=
sorry

end quadratic_inequality_range_sufficient_condition_range_l1992_199236


namespace equation_solutions_l1992_199287

theorem equation_solutions :
  (∀ x : ℝ, 4 * x * (2 * x - 1) = 3 * (2 * x - 1) → x = 1/2 ∨ x = 3/4) ∧
  (∀ x : ℝ, x^2 + 2*x - 2 = 0 → x = -1 + Real.sqrt 3 ∨ x = -1 - Real.sqrt 3) := by
  sorry

end equation_solutions_l1992_199287


namespace number_difference_l1992_199266

theorem number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) (h3 : y > x) :
  y - x = 8.58 := by
  sorry

end number_difference_l1992_199266


namespace diagonal_path_shorter_than_sides_l1992_199222

theorem diagonal_path_shorter_than_sides (ε : ℝ) (h : ε > 0) : ∃ δ : ℝ, 
  0 < δ ∧ δ < ε ∧ 
  |(2 - Real.sqrt 2) / 2 - 0.293| < δ :=
sorry

end diagonal_path_shorter_than_sides_l1992_199222


namespace victors_journey_l1992_199238

/-- The distance from Victor's home to the airport --/
def s : ℝ := 240

/-- Victor's initial speed --/
def initial_speed : ℝ := 60

/-- Victor's increased speed --/
def increased_speed : ℝ := 80

/-- Time spent at initial speed --/
def initial_time : ℝ := 0.5

/-- Time difference if Victor continued at initial speed --/
def late_time : ℝ := 0.25

/-- Time difference after increasing speed --/
def early_time : ℝ := 0.25

theorem victors_journey :
  ∃ (t : ℝ),
    s = initial_speed * initial_time + initial_speed * (t + late_time) ∧
    s = initial_speed * initial_time + increased_speed * (t - early_time) :=
by sorry

end victors_journey_l1992_199238


namespace solve_quadratic_equation_solve_linear_equation_l1992_199243

-- Equation 1
theorem solve_quadratic_equation (x : ℝ) :
  2 * x^2 - 5 * x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4 :=
sorry

-- Equation 2
theorem solve_linear_equation (x : ℝ) :
  3 * x * (x - 2) = 2 * (2 - x) ↔ x = 2 ∨ x = -2/3 :=
sorry

end solve_quadratic_equation_solve_linear_equation_l1992_199243


namespace symmetry_point_of_sine_function_l1992_199240

/-- Given a function f(x) = sin(ωx + π/6) with ω > 0, if the distance between adjacent
    symmetry axes is π/2 and the graph is symmetrical about (x₀, 0) where x₀ ∈ [0, π/2],
    then x₀ = 5π/12 -/
theorem symmetry_point_of_sine_function (ω : ℝ) (x₀ : ℝ) :
  ω > 0 →
  (2 * π) / ω = π →
  x₀ ∈ Set.Icc 0 (π / 2) →
  (∀ x, Real.sin (ω * x + π / 6) = Real.sin (ω * (2 * x₀ - x) + π / 6)) →
  x₀ = 5 * π / 12 := by
  sorry

end symmetry_point_of_sine_function_l1992_199240


namespace polyhedron_edge_length_bound_l1992_199246

/-- A polyhedron is represented as a set of points in ℝ³ -/
def Polyhedron : Type := Set (Fin 3 → ℝ)

/-- The edge length of a polyhedron -/
def edgeLength (P : Polyhedron) : ℝ := sorry

/-- The sum of all edge lengths of a polyhedron -/
def sumEdgeLengths (P : Polyhedron) : ℝ := sorry

/-- The distance between two points in ℝ³ -/
def distance (a b : Fin 3 → ℝ) : ℝ := sorry

/-- The maximum distance between any two points in a polyhedron -/
def maxDistance (P : Polyhedron) : ℝ := sorry

/-- Theorem: The sum of edge lengths is at least 3 times the maximum distance -/
theorem polyhedron_edge_length_bound (P : Polyhedron) :
  sumEdgeLengths P ≥ 3 * maxDistance P := by sorry

end polyhedron_edge_length_bound_l1992_199246


namespace crayons_lost_theorem_l1992_199290

/-- The number of crayons Paul lost or gave away -/
def crayons_lost_or_given_away (initial_crayons remaining_crayons : ℕ) : ℕ :=
  initial_crayons - remaining_crayons

/-- Theorem: The number of crayons lost or given away is equal to the difference between
    the initial number of crayons and the remaining number of crayons -/
theorem crayons_lost_theorem (initial_crayons remaining_crayons : ℕ) 
  (h : initial_crayons ≥ remaining_crayons) :
  crayons_lost_or_given_away initial_crayons remaining_crayons = initial_crayons - remaining_crayons :=
by
  sorry

#eval crayons_lost_or_given_away 479 134

end crayons_lost_theorem_l1992_199290


namespace lisa_interest_earned_l1992_199271

/-- UltraSavingsAccount represents the parameters of the savings account -/
structure UltraSavingsAccount where
  principal : ℝ
  rate : ℝ
  years : ℕ

/-- calculate_interest computes the interest earned for a given UltraSavingsAccount -/
def calculate_interest (account : UltraSavingsAccount) : ℝ :=
  account.principal * ((1 + account.rate) ^ account.years - 1)

/-- Theorem stating that Lisa's interest earned is $821 -/
theorem lisa_interest_earned (account : UltraSavingsAccount) 
  (h1 : account.principal = 2000)
  (h2 : account.rate = 0.035)
  (h3 : account.years = 10) :
  ⌊calculate_interest account⌋ = 821 := by
  sorry

end lisa_interest_earned_l1992_199271


namespace find_a_value_l1992_199291

theorem find_a_value (x y a : ℝ) 
  (h1 : x = 2) 
  (h2 : y = 1) 
  (h3 : a * x - y = 3) : a = 2 := by
  sorry

end find_a_value_l1992_199291


namespace rook_paths_bound_l1992_199258

def ChessboardPaths (n : ℕ) : ℕ := sorry

theorem rook_paths_bound (n : ℕ) :
  ChessboardPaths n ≤ 9^n ∧ ∀ k < 9, ∃ m : ℕ, ChessboardPaths m > k^m :=
by sorry

end rook_paths_bound_l1992_199258


namespace polynomial_multiplication_l1992_199296

/-- Proves that (x^4 + 12x^2 + 144)(x^2 - 12) = x^6 - 1728 for all real x. -/
theorem polynomial_multiplication (x : ℝ) : 
  (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end polynomial_multiplication_l1992_199296


namespace smallest_fraction_between_l1992_199255

theorem smallest_fraction_between (p q : ℕ+) : 
  (5 : ℚ) / 9 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (4 : ℚ) / 7 ∧ 
  (∀ p' q' : ℕ+, (5 : ℚ) / 9 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (4 : ℚ) / 7 → q ≤ q') →
  q - p = 7 := by
sorry

end smallest_fraction_between_l1992_199255


namespace w_squared_value_l1992_199265

theorem w_squared_value (w : ℝ) (h : (2*w + 19)^2 = (4*w + 9)*(3*w + 13)) :
  w^2 = ((6 + Real.sqrt 524) / 4)^2 := by
  sorry

end w_squared_value_l1992_199265


namespace parabola_line_intersection_l1992_199201

/-- Given a parabola and a line intersecting it, prove the value of m -/
theorem parabola_line_intersection (x₁ x₂ y₁ y₂ m : ℝ) : 
  (x₁^2 = 4*y₁) →  -- Point A on parabola
  (x₂^2 = 4*y₂) →  -- Point B on parabola
  (∃ k, y₁ = k*x₁ + m ∧ y₂ = k*x₂ + m) →  -- Line equation
  (x₁ * x₂ = -4) →  -- Product of x-coordinates
  (m = 1) := by
sorry

end parabola_line_intersection_l1992_199201


namespace four_digit_number_fraction_l1992_199245

theorem four_digit_number_fraction (a b c d : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →  -- Ensuring each digit is less than 10
  (∃ k : ℚ, a = k * b) →  -- First digit is a fraction of the second
  (c = a + b) →  -- Third digit is the sum of first and second
  (d = 3 * b) →  -- Last digit is 3 times the second
  (1000 * a + 100 * b + 10 * c + d = 1349) →  -- The number is 1349
  (a : ℚ) / b = 1 / 3 := by
sorry

end four_digit_number_fraction_l1992_199245


namespace initial_tickets_correct_l1992_199257

/-- The number of tickets Adam initially bought at the fair -/
def initial_tickets : ℕ := 13

/-- The number of tickets left after riding the ferris wheel -/
def tickets_left : ℕ := 4

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 9

/-- The total amount spent on the ferris wheel in dollars -/
def ferris_wheel_cost : ℕ := 81

/-- Theorem stating that the initial number of tickets is correct -/
theorem initial_tickets_correct : 
  initial_tickets = (ferris_wheel_cost / ticket_cost) + tickets_left := by
  sorry

end initial_tickets_correct_l1992_199257


namespace overlapping_squares_area_l1992_199262

/-- Given two overlapping squares with side length 12, where the overlap forms an equilateral triangle,
    prove that the area of the overlapping region is 108√3, and m + n = 111 -/
theorem overlapping_squares_area (side_length : ℝ) (m n : ℕ) :
  side_length = 12 →
  (m : ℝ) * Real.sqrt n = 108 * Real.sqrt 3 →
  n.Prime →
  m + n = 111 :=
by sorry

end overlapping_squares_area_l1992_199262


namespace max_product_sum_constraint_l1992_199249

theorem max_product_sum_constraint (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → w + x + y + z = 200 → 
  (w + x) * (y + z) ≤ 10000 := by
sorry

end max_product_sum_constraint_l1992_199249


namespace soda_bottle_count_l1992_199241

theorem soda_bottle_count (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 49) (h2 : diet_soda = 40) : 
  regular_soda + diet_soda = 89 := by
  sorry

end soda_bottle_count_l1992_199241


namespace matrix_condition_l1992_199284

variable (a b c d : ℂ)

def N : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_condition (h1 : N a b c d ^ 2 = 1) (h2 : a * b * c * d = 1) :
  a^4 + b^4 + c^4 + d^4 = 5 := by
  sorry

end matrix_condition_l1992_199284


namespace meaningful_iff_greater_than_one_l1992_199282

-- Define the condition for the expression to be meaningful
def is_meaningful (x : ℝ) : Prop := x > 1

-- Theorem stating that the expression is meaningful if and only if x > 1
theorem meaningful_iff_greater_than_one (x : ℝ) :
  is_meaningful x ↔ x > 1 :=
by sorry

end meaningful_iff_greater_than_one_l1992_199282


namespace toaster_sales_at_promo_price_l1992_199232

-- Define the inverse proportionality constant
def k : ℝ := 15 * 600

-- Define the original price and number of customers
def original_price : ℝ := 600
def original_customers : ℝ := 15

-- Define the promotional price
def promo_price : ℝ := 450

-- Define the additional sales increase factor
def promo_factor : ℝ := 1.1

-- Theorem statement
theorem toaster_sales_at_promo_price :
  let normal_sales := k / promo_price
  let promo_sales := normal_sales * promo_factor
  promo_sales = 22 := by sorry

end toaster_sales_at_promo_price_l1992_199232


namespace isosceles_angle_B_l1992_199223

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property of an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define the exterior angle of A
def exteriorAngleA (t : Triangle) : ℝ :=
  180 - t.A

-- Theorem statement
theorem isosceles_angle_B (t : Triangle) 
  (h_ext : exteriorAngleA t = 110) :
  isIsosceles t → t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry

end isosceles_angle_B_l1992_199223


namespace ad_length_l1992_199215

/-- A simple quadrilateral with specific side lengths and angle properties -/
structure SimpleQuadrilateral where
  -- Sides
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  -- Angles (in radians)
  B : ℝ
  C : ℝ
  -- Properties
  simple : Prop
  AB_length : AB = 4
  BC_length : BC = 5
  CD_length : CD = 20
  B_obtuse : π / 2 < B ∧ B < π
  C_obtuse : π / 2 < C ∧ C < π
  angle_relation : Real.sin C = -Real.cos B ∧ Real.sin C = 3/5

/-- The theorem stating the length of AD in the specific quadrilateral -/
theorem ad_length (q : SimpleQuadrilateral) : q.AD = Real.sqrt 674 := by
  sorry

end ad_length_l1992_199215


namespace unique_k_for_equation_l1992_199270

theorem unique_k_for_equation : ∃! k : ℕ+, 
  (∃ a b : ℕ+, a^2 + b^2 = k * a * b) ∧ k = 2 := by
  sorry

end unique_k_for_equation_l1992_199270


namespace extreme_value_implies_n_eq_9_l1992_199295

/-- The function f(x) = x^3 + 6x^2 + nx + 4 -/
def f (n : ℝ) (x : ℝ) : ℝ := x^3 + 6*x^2 + n*x + 4

/-- f has an extreme value at x = -1 -/
def has_extreme_value_at_neg_one (n : ℝ) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 
    x ≠ -1 ∧ |x + 1| < ε → (f n x - f n (-1)) * (x + 1) ≤ 0

theorem extreme_value_implies_n_eq_9 :
  ∀ (n : ℝ), has_extreme_value_at_neg_one n → n = 9 :=
sorry

end extreme_value_implies_n_eq_9_l1992_199295


namespace julie_age_is_fifteen_l1992_199275

/-- Represents Julie's age and earnings during a four-month period --/
structure JulieData where
  hoursPerDay : ℕ
  hourlyRatePerAge : ℚ
  workDays : ℕ
  totalEarnings : ℚ

/-- Calculates Julie's age at the end of the four-month period --/
def calculateAge (data : JulieData) : ℕ :=
  sorry

/-- Theorem stating that Julie's age at the end of the period is 15 --/
theorem julie_age_is_fifteen (data : JulieData) 
  (h1 : data.hoursPerDay = 3)
  (h2 : data.hourlyRatePerAge = 3/4)
  (h3 : data.workDays = 60)
  (h4 : data.totalEarnings = 810) :
  calculateAge data = 15 := by
  sorry

end julie_age_is_fifteen_l1992_199275


namespace arithmetic_progression_x_value_l1992_199224

/-- An arithmetic progression of positive integers -/
def arithmetic_progression (s : ℕ → ℕ) : Prop :=
  ∃ a d : ℕ, ∀ n : ℕ, s n = a + (n - 1) * d

theorem arithmetic_progression_x_value
  (s : ℕ → ℕ) (x : ℝ)
  (h_arithmetic : arithmetic_progression s)
  (h_s1 : s (s 1) = x + 2)
  (h_s2 : s (s 2) = x^2 + 18)
  (h_s3 : s (s 3) = 2*x^2 + 18) :
  x = 4 := by
sorry

end arithmetic_progression_x_value_l1992_199224


namespace book_arrangement_count_l1992_199211

theorem book_arrangement_count :
  let total_books : ℕ := 9
  let arabic_books : ℕ := 2
  let german_books : ℕ := 3
  let spanish_books : ℕ := 4
  let arabic_unit : ℕ := 1
  let spanish_unit : ℕ := 1
  let total_units : ℕ := arabic_unit + spanish_unit + german_books

  (total_books = arabic_books + german_books + spanish_books) →
  (Nat.factorial total_units * Nat.factorial arabic_books * Nat.factorial spanish_books = 5760) :=
by sorry

end book_arrangement_count_l1992_199211


namespace kolya_can_prevent_divisibility_by_nine_l1992_199251

def digits : Set Nat := {1, 2, 3, 4, 5}

def alternating_sum (n : Nat) (f : Nat → Nat) : Nat :=
  List.sum (List.range n |>.map f)

theorem kolya_can_prevent_divisibility_by_nine :
  ∃ (kolya : Nat → Nat), ∀ (vasya : Nat → Nat),
    (∀ i, kolya i ∈ digits ∧ vasya i ∈ digits) →
    ¬(alternating_sum 10 kolya + alternating_sum 10 vasya) % 9 = 0 :=
sorry

end kolya_can_prevent_divisibility_by_nine_l1992_199251


namespace largest_number_l1992_199237

theorem largest_number (a b c d : ℝ) (h1 : a = 1/2) (h2 : b = -1) (h3 : c = |(-2)|) (h4 : d = -3) :
  c ≥ a ∧ c ≥ b ∧ c ≥ d :=
by sorry

end largest_number_l1992_199237


namespace green_shirt_pairs_l1992_199203

theorem green_shirt_pairs (blue_students : ℕ) (green_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 65 →
  green_students = 95 →
  total_students = 160 →
  total_pairs = 80 →
  blue_blue_pairs = 25 →
  blue_students + green_students = total_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 40 :=
by
  sorry

end green_shirt_pairs_l1992_199203


namespace projection_of_a_onto_b_l1992_199294

/-- Prove that the projection of vector a = (3, 4) onto vector b = (0, 1) results in the vector (0, 4) -/
theorem projection_of_a_onto_b :
  let a : Fin 2 → ℝ := ![3, 4]
  let b : Fin 2 → ℝ := ![0, 1]
  let proj := (a • b) / (b • b) • b
  proj = ![0, 4] := by sorry

end projection_of_a_onto_b_l1992_199294


namespace anthony_pencil_count_l1992_199202

/-- Anthony's initial pencil count -/
def initial_pencils : ℝ := 56.0

/-- Number of pencils Anthony gives away -/
def pencils_given : ℝ := 9.0

/-- Anthony's final pencil count -/
def final_pencils : ℝ := 47.0

theorem anthony_pencil_count : initial_pencils - pencils_given = final_pencils := by
  sorry

end anthony_pencil_count_l1992_199202


namespace divisor_calculation_l1992_199261

theorem divisor_calculation (dividend quotient remainder divisor : ℕ) : 
  dividend = 15968 ∧ quotient = 89 ∧ remainder = 37 ∧ 
  dividend = divisor * quotient + remainder → 
  divisor = 179 := by
  sorry

end divisor_calculation_l1992_199261


namespace smallest_common_factor_l1992_199278

theorem smallest_common_factor : ∃ (n : ℕ), n > 0 ∧ n = 42 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), k > 1 → ¬(k ∣ (11 * m - 3) ∧ k ∣ (8 * m + 4)))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (11 * n - 3) ∧ k ∣ (8 * n + 4)) :=
by sorry

end smallest_common_factor_l1992_199278


namespace complex_magnitude_equation_l1992_199217

theorem complex_magnitude_equation (t : ℝ) : 
  0 < t → t < 4 → Complex.abs (t + 3 * Complex.I * Real.sqrt 2) * Complex.abs (7 - 5 * Complex.I) = 35 * Real.sqrt 2 → 
  t = Real.sqrt (559 / 37) := by
sorry

end complex_magnitude_equation_l1992_199217


namespace plants_eaten_third_day_l1992_199256

theorem plants_eaten_third_day 
  (initial_plants : ℕ)
  (eaten_first_day : ℕ)
  (fraction_eaten_second_day : ℚ)
  (final_plants : ℕ)
  (h1 : initial_plants = 30)
  (h2 : eaten_first_day = 20)
  (h3 : fraction_eaten_second_day = 1/2)
  (h4 : final_plants = 4)
  : initial_plants - eaten_first_day - 
    (initial_plants - eaten_first_day) * fraction_eaten_second_day - 
    final_plants = 1 := by
  sorry

end plants_eaten_third_day_l1992_199256


namespace leftover_tarts_l1992_199252

/-- The number of leftover tarts in a restaurant, given the fractions of different flavored tarts. -/
theorem leftover_tarts (cherry : ℝ) (blueberry : ℝ) (peach : ℝ) 
  (h_cherry : cherry = 0.08)
  (h_blueberry : blueberry = 0.75)
  (h_peach : peach = 0.08) :
  cherry + blueberry + peach = 0.91 := by
  sorry

end leftover_tarts_l1992_199252


namespace no_solution_range_l1992_199207

theorem no_solution_range (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 5| > m) ↔ m < 6 := by sorry

end no_solution_range_l1992_199207


namespace village_population_l1992_199267

theorem village_population (population : ℕ) : 
  (90 : ℚ) / 100 * population = 8100 → population = 9000 := by
  sorry

end village_population_l1992_199267


namespace expansion_properties_l1992_199228

theorem expansion_properties (n : ℕ) (hn : n ≥ 3) :
  (∃ k : ℕ, k ≥ 1 ∧ k ≤ n ∧ (14 - 3 * k) % 4 = 0) ∧
  (∀ k : ℕ, k ≥ 0 → k ≤ n → Nat.choose n k * ((-1/2)^k : ℚ) ≤ 21/4) :=
by sorry

end expansion_properties_l1992_199228


namespace union_of_sets_l1992_199226

def A (a : ℝ) : Set ℝ := {0, a}
def B (a : ℝ) : Set ℝ := {3^a, 1}

theorem union_of_sets (a : ℝ) (h : A a ∩ B a = {1}) : A a ∪ B a = {0, 1, 3} := by
  sorry

end union_of_sets_l1992_199226


namespace subset_implies_a_equals_one_l1992_199205

-- Define the sets A and B
def A : Set ℝ := {-1, 0, 2}
def B (a : ℝ) : Set ℝ := {2^a}

-- State the theorem
theorem subset_implies_a_equals_one (a : ℝ) :
  B a ⊆ A → a = 1 := by sorry

end subset_implies_a_equals_one_l1992_199205


namespace tournament_teams_l1992_199283

theorem tournament_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 := by
  sorry

end tournament_teams_l1992_199283


namespace g_range_l1992_199239

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + (Real.pi / 2) * Real.arcsin (x / 3) - (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9*x + 27)

theorem g_range :
  ∀ y ∈ Set.range g, π^2 / 6 ≤ y ∧ y ≤ 4*π^2 / 3 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = π^2 / 6 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = 4*π^2 / 3 :=
by sorry

end g_range_l1992_199239


namespace perfume_bottle_size_l1992_199276

def petals_per_ounce : ℕ := 320
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes_harvested : ℕ := 800
def bottles_to_make : ℕ := 20

theorem perfume_bottle_size :
  let total_petals := bushes_harvested * roses_per_bush * petals_per_rose
  let total_ounces := total_petals / petals_per_ounce
  let bottle_size := total_ounces / bottles_to_make
  bottle_size = 12 := by sorry

end perfume_bottle_size_l1992_199276


namespace d_sufficient_not_necessary_for_a_l1992_199221

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B → C) ∧ (C → B))
variable (h3 : C → D ∧ ¬(D → C))

-- Theorem statement
theorem d_sufficient_not_necessary_for_a :
  D → A ∧ ¬(A → D) :=
sorry

end d_sufficient_not_necessary_for_a_l1992_199221


namespace next_chime_together_l1992_199260

def town_hall_interval : ℕ := 18
def library_interval : ℕ := 24
def railway_interval : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_chime_together (start_hour : ℕ) : 
  ∃ (hours : ℕ), 
    hours * minutes_in_hour = Nat.lcm town_hall_interval (Nat.lcm library_interval railway_interval) ∧ 
    hours = 6 := by
  sorry

end next_chime_together_l1992_199260


namespace solution_set_part1_solution_set_part2_l1992_199229

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 := by sorry

end solution_set_part1_solution_set_part2_l1992_199229


namespace cos_alpha_value_l1992_199281

theorem cos_alpha_value (α β : Real) 
  (h1 : -π/2 < α ∧ α < π/2)
  (h2 : 2 * Real.tan β = Real.tan (2 * α))
  (h3 : Real.tan (β - α) = -2 * Real.sqrt 2) :
  Real.cos α = Real.sqrt 3 / 3 := by
sorry

end cos_alpha_value_l1992_199281


namespace gate_change_probability_l1992_199216

/-- The number of gates at the airport -/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet -/
def gate_distance : ℕ := 100

/-- The maximum walking distance we're interested in -/
def max_walk_distance : ℕ := 300

/-- The total number of possible gate change scenarios -/
def total_scenarios : ℕ := num_gates * (num_gates - 1)

/-- The number of gates within the maximum walking distance on each side -/
def gates_within_distance : ℕ := max_walk_distance / gate_distance

/-- The number of valid scenarios for gates at the extremities -/
def extreme_gate_scenarios : ℕ := 4 * gates_within_distance

/-- The number of valid scenarios for gates next to extremities -/
def next_to_extreme_scenarios : ℕ := 2 * (gates_within_distance + 1)

/-- The number of valid scenarios for middle gates -/
def middle_gate_scenarios : ℕ := (num_gates - 4) * (2 * gates_within_distance + 1)

/-- The total number of valid scenarios -/
def valid_scenarios : ℕ := extreme_gate_scenarios + next_to_extreme_scenarios + middle_gate_scenarios

/-- The probability of walking 300 feet or less -/
theorem gate_change_probability :
  (valid_scenarios : ℚ) / total_scenarios = 37 / 105 := by
  sorry

end gate_change_probability_l1992_199216


namespace solution_set_inequality_l1992_199289

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end solution_set_inequality_l1992_199289


namespace valid_factorization_l1992_199299

theorem valid_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end valid_factorization_l1992_199299


namespace domino_coverage_iff_even_uncoverable_boards_l1992_199298

/-- Represents a checkerboard -/
structure Checkerboard where
  squares : ℕ

/-- Predicate to determine if a checkerboard can be fully covered by dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  Even board.squares

theorem domino_coverage_iff_even (board : Checkerboard) :
  can_be_covered board ↔ Even board.squares :=
sorry

/-- 6x4 rectangular board -/
def board_6x4 : Checkerboard :=
  ⟨6 * 4⟩

/-- 5x5 square board -/
def board_5x5 : Checkerboard :=
  ⟨5 * 5⟩

/-- L-shaped board (5x5 with 2x2 removed) -/
def board_L : Checkerboard :=
  ⟨5 * 5 - 2 * 2⟩

/-- 3x7 rectangular board -/
def board_3x7 : Checkerboard :=
  ⟨3 * 7⟩

/-- Plus-shaped board (3x3 with 1x3 extension) -/
def board_plus : Checkerboard :=
  ⟨3 * 3 + 1 * 3⟩

theorem uncoverable_boards :
  ¬can_be_covered board_5x5 ∧
  ¬can_be_covered board_L ∧
  ¬can_be_covered board_3x7 :=
sorry

end domino_coverage_iff_even_uncoverable_boards_l1992_199298


namespace mean_of_five_numbers_with_sum_one_third_l1992_199254

theorem mean_of_five_numbers_with_sum_one_third :
  ∀ (a b c d e : ℚ), 
    a + b + c + d + e = 1/3 →
    (a + b + c + d + e) / 5 = 1/15 := by
sorry

end mean_of_five_numbers_with_sum_one_third_l1992_199254


namespace parallelogram_roots_l1992_199273

theorem parallelogram_roots (b : ℝ) : 
  (∃ (z₁ z₂ z₃ z₄ : ℂ), 
    z₁^4 - 8*z₁^3 + 13*b*z₁^2 - 5*(2*b^2 + b - 2)*z₁ + 4 = 0 ∧
    z₂^4 - 8*z₂^3 + 13*b*z₂^2 - 5*(2*b^2 + b - 2)*z₂ + 4 = 0 ∧
    z₃^4 - 8*z₃^3 + 13*b*z₃^2 - 5*(2*b^2 + b - 2)*z₃ + 4 = 0 ∧
    z₄^4 - 8*z₄^3 + 13*b*z₄^2 - 5*(2*b^2 + b - 2)*z₄ + 4 = 0 ∧
    z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    (z₁ - z₂ = z₄ - z₃) ∧ (z₁ - z₃ = z₄ - z₂)) ↔ b = 2 :=
by sorry

end parallelogram_roots_l1992_199273


namespace climb_10_steps_in_8_moves_l1992_199233

/-- The number of ways to climb n steps in exactly k moves, where each move can be either 1 or 2 steps. -/
def climbWays (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem states that there are 28 ways to climb 10 steps in exactly 8 moves. -/
theorem climb_10_steps_in_8_moves : climbWays 10 8 = 28 := by sorry

end climb_10_steps_in_8_moves_l1992_199233


namespace sum_of_sqrt_odd_sums_equals_15_l1992_199285

def odd_sum (n : ℕ) : ℕ := n^2

theorem sum_of_sqrt_odd_sums_equals_15 :
  Real.sqrt (odd_sum 1) + Real.sqrt (odd_sum 2) + Real.sqrt (odd_sum 3) + 
  Real.sqrt (odd_sum 4) + Real.sqrt (odd_sum 5) = 15 := by
  sorry

end sum_of_sqrt_odd_sums_equals_15_l1992_199285


namespace playground_girls_l1992_199225

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_children = 117 → boys = 40 → girls = total_children - boys → girls = 77 := by
  sorry

end playground_girls_l1992_199225


namespace prob_white_ball_l1992_199250

/-- Probability of drawing a white ball from a box with inaccessible balls -/
theorem prob_white_ball (total : ℕ) (white : ℕ) (locked : ℕ) : 
  total = 17 → white = 7 → locked = 3 → 
  (white : ℚ) / (total - locked : ℚ) = 1 / 2 := by
sorry

end prob_white_ball_l1992_199250


namespace union_of_A_and_B_l1992_199292

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x < 6} := by sorry

end union_of_A_and_B_l1992_199292


namespace rainbow_pencils_count_l1992_199297

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who have the color box -/
def num_people : ℕ := 6

/-- The total number of pencils -/
def total_pencils : ℕ := rainbow_colors * num_people

theorem rainbow_pencils_count : total_pencils = 42 := by
  sorry

end rainbow_pencils_count_l1992_199297


namespace circleplus_problem_l1992_199277

-- Define the ⊕ operation
def circleplus (a b : ℚ) : ℚ := (a * b) / (a + b)

-- State the theorem
theorem circleplus_problem : 
  circleplus (circleplus 3 5) (circleplus 5 4) = 60 / 59 := by
  sorry

end circleplus_problem_l1992_199277


namespace class_ratio_theorem_l1992_199244

-- Define the class structure
structure ClassComposition where
  total_students : ℕ
  girls : ℕ
  boys : ℕ

-- Define the condition given in the problem
def satisfies_condition (c : ClassComposition) : Prop :=
  2 * c.girls * 5 = 3 * c.total_students

-- Define the property we want to prove
def has_correct_ratio (c : ClassComposition) : Prop :=
  7 * c.girls = 3 * c.boys

-- The theorem to prove
theorem class_ratio_theorem (c : ClassComposition) 
  (h1 : c.total_students = c.girls + c.boys)
  (h2 : satisfies_condition c) : 
  has_correct_ratio c := by
  sorry

#check class_ratio_theorem

end class_ratio_theorem_l1992_199244


namespace chord_bisection_l1992_199272

/-- The ellipse defined by x²/16 + y²/8 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 8 = 1

/-- A point (x, y) lies on the line x + y - 3 = 0 -/
def Line (x y : ℝ) : Prop := x + y - 3 = 0

/-- The midpoint of two points -/
def Midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

theorem chord_bisection (x₁ y₁ x₂ y₂ : ℝ) :
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ Midpoint x₁ y₁ x₂ y₂ 2 1 →
  Line x₁ y₁ ∧ Line x₂ y₂ := by
  sorry

end chord_bisection_l1992_199272


namespace parabola_chord_midpoint_tangent_intersection_l1992_199293

/-- Given a parabola y² = 2px and a chord with endpoints P₁(x₁, y₁) and P₂(x₂, y₂),
    the line y = (y₁ + y₂)/2 passing through the midpoint M of the chord
    also passes through the intersection point of the tangents at P₁ and P₂. -/
theorem parabola_chord_midpoint_tangent_intersection
  (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁^2 = 2*p*x₁)
  (h₂ : y₂^2 = 2*p*x₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  let midpoint_y := (y₁ + y₂) / 2
  let tangent₁ := fun x y ↦ y₁ * y = p * (x + x₁)
  let tangent₂ := fun x y ↦ y₂ * y = p * (x + x₂)
  let intersection := fun x y ↦ tangent₁ x y ∧ tangent₂ x y
  ∃ x, intersection x midpoint_y :=
sorry

end parabola_chord_midpoint_tangent_intersection_l1992_199293


namespace domain_union_sqrt_ln_l1992_199274

-- Define the domains M and N
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x < 1}

-- State the theorem
theorem domain_union_sqrt_ln :
  M ∪ N = Set.Iio 1 ∪ Set.Ioi 2 :=
sorry

end domain_union_sqrt_ln_l1992_199274


namespace problem_solution_l1992_199286

theorem problem_solution (x y : ℝ) (hx : x = 12) (hy : y = 7) : 
  (x - y) * (x + y) = 95 ∧ (x + y)^2 = 361 := by
  sorry

end problem_solution_l1992_199286


namespace initial_number_proof_l1992_199263

theorem initial_number_proof (N : ℕ) : 
  (∀ k < 5, ¬ (23 ∣ (N + k))) → 
  (23 ∣ (N + 5)) → 
  N = 18 :=
by sorry

end initial_number_proof_l1992_199263


namespace train_speed_fraction_l1992_199214

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 12 → delay = 9 → 
  (usual_time / (usual_time + delay)) = (4 : ℝ) / 7 := by
sorry

end train_speed_fraction_l1992_199214


namespace min_product_of_three_numbers_l1992_199208

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 1/18 := by
  sorry

end min_product_of_three_numbers_l1992_199208


namespace complement_of_A_in_U_l1992_199269

def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : (U \ A) = {4} := by sorry

end complement_of_A_in_U_l1992_199269


namespace number_of_boys_l1992_199230

theorem number_of_boys (total_pupils : ℕ) (girls : ℕ) (teachers : ℕ) 
  (h1 : total_pupils = 626) 
  (h2 : girls = 308) 
  (h3 : teachers = 36) : 
  total_pupils - girls - teachers = 282 := by
  sorry

end number_of_boys_l1992_199230


namespace water_students_l1992_199253

theorem water_students (total : ℕ) (juice_percent : ℚ) (water_percent : ℚ) (juice_students : ℕ) : ℕ :=
  sorry

end water_students_l1992_199253


namespace torch_relay_probability_l1992_199212

/-- The number of torchbearers --/
def n : ℕ := 18

/-- The common difference of the arithmetic sequence --/
def d : ℕ := 3

/-- The probability of selecting three numbers from 1 to n that form an arithmetic 
    sequence with common difference d --/
def probability (n d : ℕ) : ℚ :=
  (3 * (n - 2 * d)) / (n * (n - 1) * (n - 2))

/-- The main theorem: the probability for the given problem is 1/68 --/
theorem torch_relay_probability : probability n d = 1 / 68 := by
  sorry


end torch_relay_probability_l1992_199212


namespace quadratic_function_properties_l1992_199231

def f (a x : ℝ) := -x^2 + 2*a*x - 3

theorem quadratic_function_properties :
  (∃ x : ℝ, -3 < x ∧ x < -1 ∧ f (-2) x < 0) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 5 → ∀ a : ℝ, a > -2 * Real.sqrt 3 → f a x < 3 * a * x) :=
by sorry

end quadratic_function_properties_l1992_199231


namespace greatest_divisor_four_consecutive_integers_l1992_199259

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d = 12 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) :=
by sorry

end greatest_divisor_four_consecutive_integers_l1992_199259


namespace special_ellipse_properties_l1992_199247

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  P : ℝ × ℝ
  h_P_on_ellipse : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_PF₂_eq_F₁F₂ : dist P F₂ = dist F₁ F₂
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_AB_on_ellipse : (A.1 / a) ^ 2 + (A.2 / b) ^ 2 = 1 ∧ (B.1 / a) ^ 2 + (B.2 / b) ^ 2 = 1
  h_AB_on_PF₂ : ∃ (t : ℝ), A = (1 - t) • P + t • F₂ ∧ B = (1 - t) • P + t • F₂
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_MN_on_circle : (M.1 + 1) ^ 2 + (M.2 - Real.sqrt 3) ^ 2 = 16 ∧
                   (N.1 + 1) ^ 2 + (N.2 - Real.sqrt 3) ^ 2 = 16
  h_MN_on_PF₂ : ∃ (s : ℝ), M = (1 - s) • P + s • F₂ ∧ N = (1 - s) • P + s • F₂
  h_MN_AB_ratio : dist M N = (5 / 8) * dist A B

/-- The eccentricity and equation of the special ellipse -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∃ (c : ℝ), c > 0 ∧ c ^ 2 = a ^ 2 - b ^ 2 ∧ c / a = 1 / 2) ∧
  (∃ (k : ℝ), k > 0 ∧ e.a ^ 2 = 16 * k ∧ e.b ^ 2 = 12 * k) :=
sorry

end special_ellipse_properties_l1992_199247


namespace absolute_value_inequality_l1992_199288

theorem absolute_value_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k ≥ -3 := by
  sorry

end absolute_value_inequality_l1992_199288


namespace coefficient_x_squared_is_120_l1992_199248

/-- The coefficient of x^2 in the expansion of (1+x)^2 + (1+x)^3 + ... + (1+x)^9 -/
def coefficient_x_squared : ℕ :=
  (Finset.range 8).sum (λ n => Nat.choose (n + 2) 2)

/-- Theorem stating that the coefficient of x^2 in the expansion is 120 -/
theorem coefficient_x_squared_is_120 : coefficient_x_squared = 120 := by
  sorry

end coefficient_x_squared_is_120_l1992_199248


namespace sum_of_integers_l1992_199220

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
sorry

end sum_of_integers_l1992_199220


namespace trapezoid_perimeter_is_228_l1992_199235

/-- A trapezoid with given side lengths -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  FH : ℝ
  h_EF_longer : EF > GH

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.EF + t.GH + t.EG + t.FH

/-- Theorem stating that the perimeter of the given trapezoid is 228 -/
theorem trapezoid_perimeter_is_228 : 
  ∀ (t : Trapezoid), t.EF = 90 ∧ t.GH = 40 ∧ t.EG = 53 ∧ t.FH = 45 → perimeter t = 228 := by
  sorry

end trapezoid_perimeter_is_228_l1992_199235


namespace common_tangent_sum_l1992_199219

-- Define the parabolas
def P₁ (x y : ℝ) : Prop := y = x^2 + 51/50
def P₂ (x y : ℝ) : Prop := x = y^2 + 95/8

-- Define the common tangent line
def CommonTangent (a b c : ℕ) (x y : ℝ) : Prop :=
  (a : ℝ) * x + (b : ℝ) * y = c

-- Main theorem
theorem common_tangent_sum :
  ∃ (a b c : ℕ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (∃ (m : ℚ), ∀ (x y : ℝ),
      CommonTangent a b c x y → y = m * x + (c / b : ℝ)) ∧
    (∀ (x y : ℝ),
      (P₁ x y → ∃ (x₀ y₀ : ℝ), P₁ x₀ y₀ ∧ CommonTangent a b c x₀ y₀) ∧
      (P₂ x y → ∃ (x₀ y₀ : ℝ), P₂ x₀ y₀ ∧ CommonTangent a b c x₀ y₀)) ∧
    a + b + c = 59 := by
  sorry


end common_tangent_sum_l1992_199219


namespace shortest_distance_between_circles_l1992_199279

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles 
  (c1 : ℝ × ℝ → ℝ) 
  (c2 : ℝ × ℝ → ℝ) 
  (h1 : ∀ x y, c1 (x, y) = x^2 - 6*x + y^2 - 8*y + 4)
  (h2 : ∀ x y, c2 (x, y) = x^2 + 8*x + y^2 + 12*y + 36) :
  let d := Real.sqrt 149 - Real.sqrt 21 - 4
  ∃ p1 p2, c1 p1 = 0 ∧ c2 p2 = 0 ∧ 
    d = Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) - 
        (Real.sqrt 21 + 4) ∧
    ∀ q1 q2, c1 q1 = 0 → c2 q2 = 0 → 
      d ≤ Real.sqrt ((q1.1 - q2.1)^2 + (q1.2 - q2.2)^2) - 
          (Real.sqrt 21 + 4) :=
by sorry

end shortest_distance_between_circles_l1992_199279


namespace total_seeds_in_garden_l1992_199242

/-- Represents the number of beds of each type in the garden -/
def num_beds : ℕ := 2

/-- Represents the number of rows in a top bed -/
def top_rows : ℕ := 4

/-- Represents the number of seeds per row in a top bed -/
def top_seeds_per_row : ℕ := 25

/-- Represents the number of rows in a medium bed -/
def medium_rows : ℕ := 3

/-- Represents the number of seeds per row in a medium bed -/
def medium_seeds_per_row : ℕ := 20

/-- Calculates the total number of seeds that can be planted in Grace's raised bed garden -/
theorem total_seeds_in_garden : 
  num_beds * (top_rows * top_seeds_per_row + medium_rows * medium_seeds_per_row) = 320 := by
  sorry

end total_seeds_in_garden_l1992_199242


namespace evaluate_expression_l1992_199264

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 2) : 
  z * (z - 4 * x) = -28 := by
  sorry

end evaluate_expression_l1992_199264


namespace negative_x_gt_1_is_inequality_l1992_199210

-- Define what an inequality is
def is_inequality (expr : Prop) : Prop :=
  ∃ (a b : ℝ), (expr = (a > b) ∨ expr = (a < b) ∨ expr = (a ≥ b) ∨ expr = (a ≤ b))

-- Theorem to prove
theorem negative_x_gt_1_is_inequality :
  is_inequality (-x > 1) :=
sorry

end negative_x_gt_1_is_inequality_l1992_199210


namespace josephine_milk_sales_l1992_199280

/-- The total amount of milk sold by Josephine on Sunday morning -/
def total_milk_sold (container_2L : ℕ) (container_075L : ℕ) (container_05L : ℕ) : ℝ :=
  (container_2L * 2) + (container_075L * 0.75) + (container_05L * 0.5)

/-- Theorem stating that Josephine sold 10 liters of milk given the specified containers -/
theorem josephine_milk_sales : total_milk_sold 3 2 5 = 10 := by
  sorry

end josephine_milk_sales_l1992_199280


namespace m_cubed_plus_two_m_squared_minus_2001_l1992_199213

theorem m_cubed_plus_two_m_squared_minus_2001 (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 - 2001 = -2000 := by
  sorry

end m_cubed_plus_two_m_squared_minus_2001_l1992_199213
