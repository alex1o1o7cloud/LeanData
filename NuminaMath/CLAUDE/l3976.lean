import Mathlib

namespace complement_of_M_in_U_l3976_397629

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the set M
def M : Set Nat := {1, 3, 5}

-- Theorem statement
theorem complement_of_M_in_U : 
  (U \ M) = {2, 4, 6} := by sorry

end complement_of_M_in_U_l3976_397629


namespace sphere_triangle_distance_l3976_397666

-- Define the sphere
def sphere_radius : ℝ := 9

-- Define the triangle
def triangle_side1 : ℝ := 20
def triangle_side2 : ℝ := 20
def triangle_side3 : ℝ := 30

-- State the theorem
theorem sphere_triangle_distance :
  let s := (triangle_side1 + triangle_side2 + triangle_side3) / 2
  let area := Real.sqrt (s * (s - triangle_side1) * (s - triangle_side2) * (s - triangle_side3))
  let inradius := area / s
  Real.sqrt (sphere_radius ^ 2 - inradius ^ 2) = 7 := by
  sorry

end sphere_triangle_distance_l3976_397666


namespace arrangements_eq_two_pow_l3976_397650

/-- The number of arrangements of the sequence 1, 2, ..., n, where each number
    is either strictly greater than all the numbers before it or strictly less
    than all the numbers before it. -/
def arrangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * arrangements (n - 1)

/-- Theorem stating that the number of arrangements for n numbers is 2^(n-1) -/
theorem arrangements_eq_two_pow (n : ℕ) : arrangements n = 2^(n-1) := by
  sorry

end arrangements_eq_two_pow_l3976_397650


namespace not_both_squares_l3976_397663

theorem not_both_squares (a b : ℤ) : ¬(∃ (c d : ℤ), c > 0 ∧ d > 0 ∧ a * (a + 4) = c^2 ∧ b * (b + 4) = d^2) := by
  sorry

end not_both_squares_l3976_397663


namespace smallest_m_correct_l3976_397632

/-- The smallest positive value of m for which 10x^2 - mx + 660 = 0 has integral solutions -/
def smallest_m : ℕ := 170

/-- A function representing the quadratic equation 10x^2 - mx + 660 = 0 -/
def quadratic (m : ℕ) (x : ℤ) : ℤ := 10 * x^2 - m * x + 660

theorem smallest_m_correct :
  (∃ x y : ℤ, quadratic smallest_m x = 0 ∧ quadratic smallest_m y = 0 ∧ x ≠ y) ∧
  (∀ m : ℕ, m < smallest_m → ¬∃ x y : ℤ, quadratic m x = 0 ∧ quadratic m y = 0 ∧ x ≠ y) :=
by sorry

end smallest_m_correct_l3976_397632


namespace fraction_sum_l3976_397646

theorem fraction_sum (m n : ℕ) (hcoprime : Nat.Coprime m n) 
  (heq : (2013 * 2013) / (2014 * 2014 + 2012) = n / m) : m + n = 1343 := by
  sorry

end fraction_sum_l3976_397646


namespace book_pages_l3976_397608

/-- The number of pages Charlie read in the book -/
def total_pages : ℕ :=
  let first_four_days : ℕ := 4 * 45
  let next_three_days : ℕ := 3 * 52
  let last_day : ℕ := 15
  first_four_days + next_three_days + last_day

/-- Theorem stating that the total number of pages in the book is 351 -/
theorem book_pages : total_pages = 351 := by
  sorry

end book_pages_l3976_397608


namespace matts_current_age_matts_age_is_65_l3976_397699

/-- Given that James turned 27 three years ago and in 5 years, Matt will be twice James' age,
    prove that Matt's current age is 65. -/
theorem matts_current_age : ℕ → Prop :=
  fun age_matt : ℕ =>
    let age_james_3_years_ago : ℕ := 27
    let years_since_james_27 : ℕ := 3
    let years_until_matt_twice_james : ℕ := 5
    let age_james : ℕ := age_james_3_years_ago + years_since_james_27
    let age_james_in_5_years : ℕ := age_james + years_until_matt_twice_james
    let age_matt_in_5_years : ℕ := 2 * age_james_in_5_years
    age_matt = age_matt_in_5_years - years_until_matt_twice_james ∧ age_matt = 65

/-- Proof of Matt's current age -/
theorem matts_age_is_65 : matts_current_age 65 := by
  sorry

end matts_current_age_matts_age_is_65_l3976_397699


namespace absolute_value_symmetry_l3976_397648

/-- A function f : ℝ → ℝ is symmetric about the line x = c if f(c + x) = f(c - x) for all x ∈ ℝ -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem absolute_value_symmetry (a : ℝ) :
  SymmetricAbout (fun x ↦ |x - a|) 3 → a = 3 := by
  sorry

end absolute_value_symmetry_l3976_397648


namespace sphere_radius_is_four_l3976_397688

/-- Represents a truncated cone with given dimensions and a tangent sphere -/
structure TruncatedConeWithSphere where
  baseRadius : ℝ
  topRadius : ℝ
  height : ℝ
  sphereRadius : ℝ

/-- Checks if the given dimensions satisfy the conditions for a truncated cone with a tangent sphere -/
def isValidConfiguration (cone : TruncatedConeWithSphere) : Prop :=
  cone.baseRadius > cone.topRadius ∧
  cone.height > 0 ∧
  cone.sphereRadius > 0 ∧
  -- The sphere is tangent to the top, bottom, and lateral surface
  cone.sphereRadius = cone.height - Real.sqrt ((cone.baseRadius - cone.topRadius)^2 + cone.height^2)

/-- Theorem stating that for a truncated cone with given dimensions and a tangent sphere, the radius of the sphere is 4 -/
theorem sphere_radius_is_four :
  ∀ (cone : TruncatedConeWithSphere),
    cone.baseRadius = 24 ∧
    cone.topRadius = 6 ∧
    cone.height = 20 ∧
    isValidConfiguration cone →
    cone.sphereRadius = 4 := by
  sorry

end sphere_radius_is_four_l3976_397688


namespace seventieth_number_is_557_l3976_397611

/-- The nth positive integer that leaves a remainder of 5 when divided by 8 -/
def nth_number (n : ℕ) : ℕ := 8 * (n - 1) + 5

/-- Proposition: The 70th positive integer that leaves a remainder of 5 when divided by 8 is 557 -/
theorem seventieth_number_is_557 : nth_number 70 = 557 := by
  sorry

end seventieth_number_is_557_l3976_397611


namespace marcy_spears_count_l3976_397633

/-- The number of spears that can be made from one sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from one log -/
def spears_per_log : ℕ := 9

/-- The number of saplings Marcy has -/
def num_saplings : ℕ := 6

/-- The number of logs Marcy has -/
def num_logs : ℕ := 1

/-- Theorem: Marcy can make 27 spears from 6 saplings and 1 log -/
theorem marcy_spears_count :
  spears_per_sapling * num_saplings + spears_per_log * num_logs = 27 := by
  sorry

end marcy_spears_count_l3976_397633


namespace log_graph_passes_through_point_l3976_397620

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 2) + 3

-- State the theorem
theorem log_graph_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 3 = 3 := by sorry

end log_graph_passes_through_point_l3976_397620


namespace total_interest_received_l3976_397639

/-- Calculate simple interest -/
def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

/-- Calculate total interest from two loans -/
def total_interest (principal1 principal2 rate time1 time2 : ℕ) : ℕ :=
  simple_interest principal1 rate time1 + simple_interest principal2 rate time2

/-- Theorem stating the total interest received by A -/
theorem total_interest_received : 
  total_interest 5000 3000 12 2 4 = 2440 := by
  sorry

end total_interest_received_l3976_397639


namespace casper_candy_problem_l3976_397610

theorem casper_candy_problem (initial_candies : ℚ) : 
  let day1_remaining := (3/4) * initial_candies - 3
  let day2_remaining := (4/5) * day1_remaining - 5
  let day3_remaining := day2_remaining - 10
  day3_remaining = 10 → initial_candies = 224/3 := by
  sorry

end casper_candy_problem_l3976_397610


namespace chicken_farm_growth_l3976_397677

theorem chicken_farm_growth (initial_chickens : ℕ) (annual_increase : ℕ) (years : ℕ) 
  (h1 : initial_chickens = 550)
  (h2 : annual_increase = 150)
  (h3 : years = 9) :
  initial_chickens + years * annual_increase = 1900 :=
by sorry

end chicken_farm_growth_l3976_397677


namespace distance_between_points_l3976_397660

/-- The distance between points (3,4) and (8,17) is √194 -/
theorem distance_between_points : Real.sqrt ((8 - 3)^2 + (17 - 4)^2) = Real.sqrt 194 := by
  sorry

end distance_between_points_l3976_397660


namespace function_satisfies_equation_l3976_397612

theorem function_satisfies_equation (x b : ℝ) : 
  let y := (b + x) / (1 + b*x)
  let y' := ((1 - b^2) / (1 + b*x)^2)
  y - x * y' = b * (1 + x^2 * y') := by sorry

end function_satisfies_equation_l3976_397612


namespace simplify_and_evaluate_l3976_397695

theorem simplify_and_evaluate : 
  let x : ℝ := -3
  3 * (2 * x^2 - 5 * x) - 2 * (-3 * x - 2 + 3 * x^2) = 31 := by
  sorry

end simplify_and_evaluate_l3976_397695


namespace right_angled_isosceles_unique_indivisible_l3976_397690

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ  -- side length
  b : ℝ  -- base length
  ha : a > 0
  hb : b > 0

/-- A right-angled isosceles triangle -/
def RightAngledIsoscelesTriangle (t : IsoscelesTriangle) : Prop :=
  t.a = t.b * Real.sqrt 2 / 2

/-- Predicate for a triangle that can be divided into three isosceles triangles with equal side lengths -/
def CanBeDividedIntoThreeIsosceles (t : IsoscelesTriangle) : Prop :=
  ∃ (t1 t2 t3 : IsoscelesTriangle),
    t1.a = t2.a ∧ t2.a = t3.a ∧
    -- Additional conditions to ensure the three triangles form a partition of t
    sorry

/-- Theorem stating that only right-angled isosceles triangles cannot be divided into three isosceles triangles with equal side lengths -/
theorem right_angled_isosceles_unique_indivisible (t : IsoscelesTriangle) :
  ¬(CanBeDividedIntoThreeIsosceles t) ↔ RightAngledIsoscelesTriangle t :=
sorry

end right_angled_isosceles_unique_indivisible_l3976_397690


namespace gcd_3270_594_l3976_397664

theorem gcd_3270_594 : Nat.gcd 3270 594 = 6 := by
  sorry

end gcd_3270_594_l3976_397664


namespace min_value_a_l3976_397642

theorem min_value_a : ∃ (a : ℝ),
  (∃ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 ∧ 1 + 2^x + a * 4^x ≥ 0) ∧
  (∀ (b : ℝ), (∃ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 ∧ 1 + 2^x + b * 4^x ≥ 0) → b ≥ a) ∧
  a = -6 :=
sorry

end min_value_a_l3976_397642


namespace minimum_cactus_species_l3976_397669

theorem minimum_cactus_species (n : ℕ) (h1 : n = 80) : ∃ (k : ℕ),
  (∀ (S : Finset (Finset ℕ)), S.card = n → 
    (∀ i : ℕ, i ≤ k → ∃ s ∈ S, i ∉ s) ∧ 
    (∀ T : Finset (Finset ℕ), T ⊆ S → T.card = 15 → ∃ i : ℕ, i ≤ k ∧ ∀ s ∈ T, i ∈ s)) ∧
  k = 16 ∧ 
  (∀ m : ℕ, m < k → ¬∃ (S : Finset (Finset ℕ)), S.card = n ∧ 
    (∀ i : ℕ, i ≤ m → ∃ s ∈ S, i ∉ s) ∧
    (∀ T : Finset (Finset ℕ), T ⊆ S → T.card = 15 → ∃ i : ℕ, i ≤ m ∧ ∀ s ∈ T, i ∈ s)) :=
by sorry


end minimum_cactus_species_l3976_397669


namespace positive_numbers_l3976_397668

theorem positive_numbers (a b c : ℝ) 
  (sum_pos : a + b + c > 0)
  (sum_prod_pos : a * b + b * c + c * a > 0)
  (prod_pos : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_numbers_l3976_397668


namespace expression_evaluation_l3976_397683

theorem expression_evaluation (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) + 2 = 14 / 5 := by
  sorry

end expression_evaluation_l3976_397683


namespace bingo_prize_calculation_l3976_397681

/-- The total prize money for the bingo night. -/
def total_prize_money : ℝ := 2400

/-- The amount received by each of the 10 winners after the first winner. -/
def winner_amount : ℝ := 160

theorem bingo_prize_calculation :
  let first_winner_share := total_prize_money / 3
  let remaining_after_first := total_prize_money - first_winner_share
  let each_winner_share := remaining_after_first / 10
  (each_winner_share = winner_amount) ∧ 
  (total_prize_money > 0) ∧
  (winner_amount > 0) :=
by sorry

end bingo_prize_calculation_l3976_397681


namespace weekend_art_class_earnings_l3976_397613

/-- Calculates the total money earned over a weekend of art classes --/
def weekend_earnings (beginner_cost advanced_cost : ℕ)
  (saturday_beginner saturday_advanced : ℕ)
  (sibling_discount : ℕ) (sibling_pairs : ℕ) : ℕ :=
  let saturday_total := beginner_cost * saturday_beginner + advanced_cost * saturday_advanced
  let sunday_total := beginner_cost * (saturday_beginner / 2) + advanced_cost * (saturday_advanced / 2)
  let total_before_discount := saturday_total + sunday_total
  let total_discount := sibling_discount * (2 * sibling_pairs)
  total_before_discount - total_discount

/-- Theorem stating that the total earnings for the weekend is $720.00 --/
theorem weekend_art_class_earnings :
  weekend_earnings 15 20 20 10 3 5 = 720 := by
  sorry

end weekend_art_class_earnings_l3976_397613


namespace perimeter_is_twelve_l3976_397618

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  base_positive : base > 0
  leg_positive : leg > 0

/-- A quadrilateral formed by cutting a corner from an equilateral triangle -/
def CutCornerQuadrilateral (et : EquilateralTriangle) (it : IsoscelesTriangle) :=
  it.leg < et.side ∧ it.base < et.side

/-- The perimeter of the quadrilateral formed by cutting a corner from an equilateral triangle -/
def perimeter (et : EquilateralTriangle) (it : IsoscelesTriangle) 
    (h : CutCornerQuadrilateral et it) : ℝ :=
  et.side + 2 * (et.side - it.leg) + it.base

/-- The main theorem -/
theorem perimeter_is_twelve 
    (et : EquilateralTriangle)
    (it : IsoscelesTriangle)
    (h : CutCornerQuadrilateral et it)
    (h_et_side : et.side = 4)
    (h_it_leg : it.leg = 0.5)
    (h_it_base : it.base = 1) :
    perimeter et it h = 12 := by
  sorry

end perimeter_is_twelve_l3976_397618


namespace positive_sum_l3976_397665

theorem positive_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + c := by
  sorry

end positive_sum_l3976_397665


namespace last_two_digits_sum_factorials_l3976_397638

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_factorials :
  let sum := (List.range 50).map (λ i => factorial (i + 1)) |> List.foldl (· + ·) 0
  last_two_digits sum = 13 := by sorry

end last_two_digits_sum_factorials_l3976_397638


namespace path_area_calculation_l3976_397637

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

theorem path_area_calculation (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5) :
  path_area field_length field_width path_width = 675 := by
  sorry

#eval path_area 75 55 2.5

end path_area_calculation_l3976_397637


namespace math_scores_properties_l3976_397670

def scores : List ℝ := [60, 60, 60, 65, 65, 70, 70, 70, 75, 75, 75, 75, 75, 80, 80, 80, 80, 85, 85, 90]

def group_a : List ℝ := [60, 60, 60, 65, 65, 70, 70, 70]

def mode (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem math_scores_properties :
  (mode scores = 75) ∧ (variance group_a = 75/4) := by sorry

end math_scores_properties_l3976_397670


namespace karlee_grapes_l3976_397692

theorem karlee_grapes (G : ℚ) : 
  (G * 3/5 * 3/5 + G * 3/5) = 96 → G = 100 := by
  sorry

end karlee_grapes_l3976_397692


namespace collision_count_theorem_l3976_397615

/-- Represents the physical properties and conditions of the ball collision problem -/
structure BallCollisionProblem where
  tubeLength : ℝ
  numBalls : ℕ
  ballVelocity : ℝ
  timePeriod : ℝ

/-- Calculates the number of collisions for a given BallCollisionProblem -/
def calculateCollisions (problem : BallCollisionProblem) : ℕ :=
  sorry

/-- Theorem stating that the number of collisions for the given problem is 505000 -/
theorem collision_count_theorem (problem : BallCollisionProblem) 
  (h1 : problem.tubeLength = 1)
  (h2 : problem.numBalls = 100)
  (h3 : problem.ballVelocity = 10)
  (h4 : problem.timePeriod = 10) :
  calculateCollisions problem = 505000 := by
  sorry

end collision_count_theorem_l3976_397615


namespace problem_solution_l3976_397604

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 2 = d + (a + b + c - d)^(1/3)) : 
  d = 1/2 := by
sorry

end problem_solution_l3976_397604


namespace f_sum_2009_2010_l3976_397679

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_sum_2009_2010 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period (fun x ↦ f (3*x + 1)) 2)
  (h_f_1 : f 1 = 2010) :
  f 2009 + f 2010 = -2010 := by sorry

end f_sum_2009_2010_l3976_397679


namespace power_quotient_nineteen_l3976_397641

theorem power_quotient_nineteen : 19^11 / 19^8 = 6859 := by sorry

end power_quotient_nineteen_l3976_397641


namespace triangle_sine_sum_inequality_l3976_397675

theorem triangle_sine_sum_inequality (A B C : Real) : 
  A + B + C = π → 0 < A → 0 < B → 0 < C → 
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_sine_sum_inequality_l3976_397675


namespace parabola_and_line_properties_l3976_397652

-- Define the parabola C: y² = 4x
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F(1, 0)
def F : ℝ × ℝ := (1, 0)

-- Define point K(-1, 0)
def K : ℝ × ℝ := (-1, 0)

-- Define the line l passing through K and intersecting C at A and B
def l (m : ℝ) (y : ℝ) : ℝ := m*y - 1

-- Define the symmetry of A and D with respect to the x-axis
def symmetric_x_axis (A D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = -D.2

-- Define the dot product condition
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 8/9

-- Main theorem
theorem parabola_and_line_properties
  (A B D : ℝ × ℝ)
  (m : ℝ)
  (h1 : C A.1 A.2)
  (h2 : C B.1 B.2)
  (h3 : A.1 = l m A.2)
  (h4 : B.1 = l m B.2)
  (h5 : symmetric_x_axis A D)
  (h6 : dot_product_condition A B) :
  (∃ (t : ℝ), F = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2)) ∧
  (∃ (a r : ℝ), a = 1/9 ∧ r = 2/3 ∧ ∀ (x y : ℝ), (x - a)^2 + y^2 = r^2 ↔ 
    (x - K.1)^2 + y^2 ≤ r^2 ∧ (x - B.1)^2 + (y - B.2)^2 ≤ r^2 ∧ (x - D.1)^2 + (y - D.2)^2 ≤ r^2) :=
by sorry

end parabola_and_line_properties_l3976_397652


namespace imaginary_part_of_z_l3976_397662

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*I) = Complex.abs (-3 + 4*I)) : 
  Complex.im z = -2 := by
  sorry

end imaginary_part_of_z_l3976_397662


namespace arithmetic_expression_equality_l3976_397644

theorem arithmetic_expression_equality : 50 + 5 * 12 / (180 / 3) = 51 := by
  sorry

end arithmetic_expression_equality_l3976_397644


namespace smallest_max_sum_l3976_397672

theorem smallest_max_sum (a b c d e : ℕ+) (h_sum : a + b + c + d + e = 2500) :
  ∃ N : ℕ, N = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
  (∀ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) → N ≤ M) ∧
  N = 834 :=
sorry

end smallest_max_sum_l3976_397672


namespace bills_age_l3976_397616

theorem bills_age (caroline_age bill_age : ℕ) : 
  bill_age = 2 * caroline_age - 1 →
  bill_age + caroline_age = 26 →
  bill_age = 17 := by
sorry

end bills_age_l3976_397616


namespace base4_multiplication_l3976_397653

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base4_multiplication (a b : List Nat) :
  decimalToBase4 (base4ToDecimal a * base4ToDecimal b) = [3, 2, 1, 3, 3] ↔
  a = [3, 1, 2, 1] ∧ b = [1, 2] :=
sorry

end base4_multiplication_l3976_397653


namespace absolute_value_inequality_l3976_397631

theorem absolute_value_inequality (x y : ℝ) 
  (h1 : |x - y| < 1) 
  (h2 : |2*x + y| < 1) : 
  |y| < 1 := by sorry

end absolute_value_inequality_l3976_397631


namespace superinvariant_characterization_l3976_397656

/-- A set S is superinvariant if for any stretching A of S, there exists a translation B
    such that the images of S under A and B agree. -/
def Superinvariant (S : Set ℝ) : Prop :=
  ∀ (x₀ a : ℝ) (h : a > 0),
    ∃ b : ℝ,
      (∀ x ∈ S, ∃ y ∈ S, x₀ + a * (x - x₀) = y + b) ∧
      (∀ t ∈ S, ∃ u ∈ S, t + b = x₀ + a * (u - x₀))

/-- The set of all possible superinvariant sets for a given Γ. -/
def SuperinvariantSets (Γ : ℝ) : Set (Set ℝ) :=
  {∅, {Γ}, Set.Iio Γ, Set.Iic Γ, Set.Ioi Γ, Set.Ici Γ, (Set.Iio Γ) ∪ (Set.Ioi Γ), Set.univ}

/-- Theorem stating that a set is superinvariant if and only if it belongs to
    SuperinvariantSets for some Γ. -/
theorem superinvariant_characterization (S : Set ℝ) :
  Superinvariant S ↔ ∃ Γ : ℝ, S ∈ SuperinvariantSets Γ := by
  sorry

end superinvariant_characterization_l3976_397656


namespace range_of_f_l3976_397607

def f (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc 2 5, f x = y}
  S = Set.Icc (-3) 6 := by sorry

end range_of_f_l3976_397607


namespace skips_per_meter_correct_l3976_397674

/-- Represents the number of skips in one meter given the following conditions:
    * x hops equals y skips
    * z jumps equals w hops
    * u jumps equals v meters
-/
def skips_per_meter (x y z w u v : ℚ) : ℚ :=
  u * y * w / (v * x * z)

/-- Theorem stating that under the given conditions, 
    1 meter equals (uyw / (vxz)) skips -/
theorem skips_per_meter_correct
  (x y z w u v : ℚ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hu : u > 0) (hv : v > 0)
  (hops_to_skips : x * 1 = y)
  (jumps_to_hops : z * 1 = w)
  (jumps_to_meters : u * 1 = v) :
  skips_per_meter x y z w u v = u * y * w / (v * x * z) :=
by sorry

end skips_per_meter_correct_l3976_397674


namespace f_is_quadratic_l3976_397602

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation x^2 - 3x + 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem stating that f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l3976_397602


namespace triangle_rotation_path_length_l3976_397614

/-- The total path length of a vertex of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : triangle_side > 0) 
  (h4 : square_side > triangle_side) : 
  (4 : ℝ) * (π / 2) * triangle_side = 6 * π := by
sorry

end triangle_rotation_path_length_l3976_397614


namespace sum_abc_equals_two_l3976_397654

theorem sum_abc_equals_two (a b c : ℝ) 
  (h : (a - 1)^2 + |b + 1| + Real.sqrt (b + c - a) = 0) : 
  a + b + c = 2 := by
sorry

end sum_abc_equals_two_l3976_397654


namespace quadratic_roots_in_sixth_degree_l3976_397684

theorem quadratic_roots_in_sixth_degree (p q : ℝ) : 
  (∀ x : ℝ, x^2 - x - 1 = 0 → x^6 - p*x^2 + q = 0) → 
  p = 8 ∧ q = 3 := by
sorry

end quadratic_roots_in_sixth_degree_l3976_397684


namespace math_test_problems_left_l3976_397603

/-- Calculates the number of problems left to solve in a math test -/
def problems_left_to_solve (total_problems : ℕ) (first_20min : ℕ) (second_20min : ℕ) : ℕ :=
  total_problems - (first_20min + second_20min)

/-- Proves that given the conditions, the number of problems left to solve is 45 -/
theorem math_test_problems_left : 
  let total_problems : ℕ := 75
  let first_20min : ℕ := 10
  let second_20min : ℕ := first_20min * 2
  problems_left_to_solve total_problems first_20min second_20min = 45 := by
  sorry

#eval problems_left_to_solve 75 10 20

end math_test_problems_left_l3976_397603


namespace hyperbola_eccentricity_l3976_397697

/-- Given a parabola and a hyperbola with coinciding foci, 
    prove that the eccentricity of the hyperbola is 2√3/3 -/
theorem hyperbola_eccentricity 
  (parabola : ℝ → ℝ) 
  (hyperbola : ℝ → ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x y, parabola y = (1/8) * x^2)
  (h2 : ∀ x y, hyperbola x y = y^2/a - x^2 - 1)
  (h3 : ∃ x y, parabola y = (1/8) * x^2 ∧ hyperbola x y = 0 ∧ 
              x^2 + (y - a/2)^2 = (a/2)^2) : 
  ∃ e : ℝ, e = 2 * Real.sqrt 3 / 3 ∧ 
    ∀ x y, hyperbola x y = 0 → x^2/(a/e^2) + y^2/a = 1 :=
sorry

end hyperbola_eccentricity_l3976_397697


namespace triangle_right_angle_l3976_397651

/-- If in a triangle ABC, sin(A+B) = sin(A-B), then the triangle ABC is a right triangle. -/
theorem triangle_right_angle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_sin_eq : Real.sin (A + B) = Real.sin (A - B)) : 
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := by
  sorry


end triangle_right_angle_l3976_397651


namespace no_real_solution_l3976_397678

theorem no_real_solution :
  ¬∃ (r s : ℝ), (r - 50) / 3 = (s - 2 * r) / 4 ∧ r^2 + 3 * s = 50 := by
  sorry

end no_real_solution_l3976_397678


namespace equation_to_lines_l3976_397687

/-- The set of points satisfying 2x^2 + y^2 + 3xy + 3x + y = 2 is equivalent to the set of points on the lines y = -x - 2 and y = -2x + 1 -/
theorem equation_to_lines :
  ∀ x y : ℝ, 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end equation_to_lines_l3976_397687


namespace find_multiple_of_ages_l3976_397627

/-- Given Hiram's age and Allyson's age, find the multiple M that satisfies the equation. -/
theorem find_multiple_of_ages (hiram_age allyson_age : ℕ) (M : ℚ)
  (h1 : hiram_age = 40)
  (h2 : allyson_age = 28)
  (h3 : hiram_age + 12 = M * allyson_age - 4) :
  M = 2 := by
  sorry

end find_multiple_of_ages_l3976_397627


namespace smallest_K_is_correct_l3976_397676

/-- The smallest positive integer K such that 8000 × K is a perfect square -/
def smallest_K : ℕ := 5

/-- A predicate that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_K_is_correct :
  (∀ k : ℕ, k > 0 → k < smallest_K → ¬ is_perfect_square (8000 * k)) ∧
  is_perfect_square (8000 * smallest_K) := by
  sorry

#check smallest_K_is_correct

end smallest_K_is_correct_l3976_397676


namespace smallest_n_congruence_l3976_397694

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, 725*m ≡ 1025*m [ZMOD 40] → n ≤ m) ∧ 
  (725*n ≡ 1025*n [ZMOD 40]) := by
  sorry

end smallest_n_congruence_l3976_397694


namespace hyperbola_asymptote_tangent_circle_l3976_397634

theorem hyperbola_asymptote_tangent_circle (m : ℝ) : 
  m > 0 →
  (∀ x y : ℝ, y^2 - x^2/m^2 = 1 → x^2 + y^2 - 4*y + 3 = 0) →
  m = Real.sqrt 3 / 3 :=
sorry

end hyperbola_asymptote_tangent_circle_l3976_397634


namespace exponent_fraction_simplification_l3976_397657

theorem exponent_fraction_simplification :
  (3^8 + 3^6) / (3^8 - 3^6) = 5/4 := by
  sorry

end exponent_fraction_simplification_l3976_397657


namespace range_of_expression_l3976_397601

theorem range_of_expression (x y : ℝ) 
  (h1 : x ≥ 0) 
  (h2 : y ≥ x) 
  (h3 : 4 * x + 3 * y ≤ 12) : 
  3 ≤ (x + 2 * y + 3) / (x + 1) ∧ (x + 2 * y + 3) / (x + 1) ≤ 11 := by
  sorry

end range_of_expression_l3976_397601


namespace line_slope_l3976_397622

theorem line_slope (x y : ℝ) : x - Real.sqrt 3 * y - Real.sqrt 3 = 0 → 
  (y - (-1)) / (x - 0) = 1 / Real.sqrt 3 := by
  sorry

end line_slope_l3976_397622


namespace sqrt_square_abs_sqrt_neg_nine_squared_l3976_397671

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

theorem sqrt_neg_nine_squared : Real.sqrt ((-9) ^ 2) = 9 := by sorry

end sqrt_square_abs_sqrt_neg_nine_squared_l3976_397671


namespace rectangle_perimeter_problem_l3976_397625

/-- The perimeter of a rectangle given its width and height -/
def perimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The width of the large rectangle in terms of small rectangles -/
def large_width : ℕ := 5

/-- The height of the large rectangle in terms of small rectangles -/
def large_height : ℕ := 4

theorem rectangle_perimeter_problem (x y : ℝ) 
  (hA : perimeter (6 * x) y = 56)
  (hB : perimeter (4 * x) (3 * y) = 56) :
  perimeter (2 * x) (3 * y) = 40 := by
  sorry

end rectangle_perimeter_problem_l3976_397625


namespace eleventh_term_of_geometric_sequence_l3976_397698

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem eleventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_fifth : a 5 = 5)
  (h_eighth : a 8 = 40) :
  a 11 = 320 := by
sorry

end eleventh_term_of_geometric_sequence_l3976_397698


namespace sum_of_reciprocals_squared_l3976_397600

/-- Given the following definitions:
  a = 2√2 + 3√3 + 4√6
  b = -2√2 + 3√3 + 4√6
  c = 2√2 - 3√3 + 4√6
  d = -2√2 - 3√3 + 4√6
  Prove that (1/a + 1/b + 1/c + 1/d)² = 952576/70225 -/
theorem sum_of_reciprocals_squared (a b c d : ℝ) :
  a = 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  b = -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  c = 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  d = -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  (1/a + 1/b + 1/c + 1/d)^2 = 952576/70225 := by
  sorry

end sum_of_reciprocals_squared_l3976_397600


namespace cat_food_finished_on_sunday_l3976_397673

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the number of cans consumed up to and including a given day -/
def cans_consumed (d : Day) : ℚ :=
  match d with
  | Day.Monday => 3/4
  | Day.Tuesday => 3/2
  | Day.Wednesday => 9/4
  | Day.Thursday => 3
  | Day.Friday => 15/4
  | Day.Saturday => 9/2
  | Day.Sunday => 21/4

/-- The amount of cat food Roy starts with -/
def initial_cans : ℚ := 8

theorem cat_food_finished_on_sunday :
  ∀ d : Day, cans_consumed d ≤ initial_cans ∧
  (d = Day.Sunday → cans_consumed d > initial_cans - 3/4) :=
by sorry

end cat_food_finished_on_sunday_l3976_397673


namespace opposites_sum_l3976_397655

theorem opposites_sum (x y : ℝ) : (x + 5)^2 + |y - 2| = 0 → x + 2*y = -1 := by
  sorry

end opposites_sum_l3976_397655


namespace sqrt_18_times_sqrt_32_l3976_397643

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end sqrt_18_times_sqrt_32_l3976_397643


namespace mile_to_rod_l3976_397628

-- Define the units
def mile : ℕ := 1
def furlong : ℕ := 1
def rod : ℕ := 1

-- Define the conversions
axiom mile_to_furlong : mile = 10 * furlong
axiom furlong_to_rod : furlong = 40 * rod

-- Theorem to prove
theorem mile_to_rod : mile = 400 * rod := by
  sorry

end mile_to_rod_l3976_397628


namespace equation_represents_pair_of_lines_l3976_397621

theorem equation_represents_pair_of_lines :
  ∀ (x y : ℝ), x^2 - 9*y^2 = 0 → ∃ (m₁ m₂ : ℝ), (x = m₁*y ∨ x = m₂*y) ∧ m₁ ≠ m₂ := by
  sorry

end equation_represents_pair_of_lines_l3976_397621


namespace negation_equivalence_l3976_397661

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 - 2*x + 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 > 0) := by
  sorry

end negation_equivalence_l3976_397661


namespace number_divided_by_three_l3976_397605

theorem number_divided_by_three : ∃ x : ℤ, x / 3 = x - 24 ∧ x = 36 := by
  sorry

end number_divided_by_three_l3976_397605


namespace x_squared_eq_5_is_quadratic_l3976_397689

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 5 -/
def f (x : ℝ) : ℝ := x^2 - 5

theorem x_squared_eq_5_is_quadratic :
  is_quadratic_one_var f :=
sorry

end x_squared_eq_5_is_quadratic_l3976_397689


namespace max_distance_complex_circle_l3976_397606

theorem max_distance_complex_circle (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (max_val : ℝ), max_val = 4 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≤ max_val :=
sorry

end max_distance_complex_circle_l3976_397606


namespace benny_seashells_l3976_397609

/-- Proves that the initial number of seashells Benny found is equal to the number of seashells he has now plus the number of seashells he gave away. -/
theorem benny_seashells (seashells_now : ℕ) (seashells_given : ℕ) 
  (h1 : seashells_now = 14) 
  (h2 : seashells_given = 52) : 
  seashells_now + seashells_given = 66 := by
  sorry

#check benny_seashells

end benny_seashells_l3976_397609


namespace exist_non_congruent_polyhedra_with_same_views_l3976_397696

/-- Represents a polyhedron --/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))

/-- Represents a 2D view of a polyhedron --/
structure View where
  points : Set (Fin 2 → ℝ)
  edges : Set (Fin 2 → ℝ) × (Fin 2 → ℝ)

/-- Checks if two polyhedra are congruent --/
def are_congruent (p1 p2 : Polyhedron) : Prop :=
  sorry

/-- Gets the front view of a polyhedron --/
def front_view (p : Polyhedron) : View :=
  sorry

/-- Gets the top view of a polyhedron --/
def top_view (p : Polyhedron) : View :=
  sorry

/-- Checks if a view has an internal intersection point at the center of the square --/
def has_center_intersection (v : View) : Prop :=
  sorry

/-- Checks if all segments of the squares in a view are visible edges --/
def all_segments_visible (v : View) : Prop :=
  sorry

/-- Checks if a view has no hidden edges --/
def no_hidden_edges (v : View) : Prop :=
  sorry

/-- The main theorem stating the existence of two non-congruent polyhedra with the given properties --/
theorem exist_non_congruent_polyhedra_with_same_views : 
  ∃ (p1 p2 : Polyhedron), 
    front_view p1 = front_view p2 ∧
    top_view p1 = top_view p2 ∧
    has_center_intersection (front_view p1) ∧
    has_center_intersection (top_view p1) ∧
    all_segments_visible (front_view p1) ∧
    all_segments_visible (top_view p1) ∧
    no_hidden_edges (front_view p1) ∧
    no_hidden_edges (top_view p1) ∧
    ¬(are_congruent p1 p2) :=
  sorry

end exist_non_congruent_polyhedra_with_same_views_l3976_397696


namespace bracelet_price_is_15_l3976_397624

/-- The price of a gold heart necklace in dollars -/
def gold_heart_price : ℕ := 10

/-- The price of a personalized coffee mug in dollars -/
def coffee_mug_price : ℕ := 20

/-- The number of bracelets bought -/
def bracelets_bought : ℕ := 3

/-- The number of gold heart necklaces bought -/
def necklaces_bought : ℕ := 2

/-- The number of coffee mugs bought -/
def mugs_bought : ℕ := 1

/-- The amount paid in dollars -/
def amount_paid : ℕ := 100

/-- The change received in dollars -/
def change_received : ℕ := 15

theorem bracelet_price_is_15 :
  ∃ (bracelet_price : ℕ),
    bracelet_price * bracelets_bought +
    gold_heart_price * necklaces_bought +
    coffee_mug_price * mugs_bought =
    amount_paid - change_received ∧
    bracelet_price = 15 := by
  sorry

end bracelet_price_is_15_l3976_397624


namespace min_value_theorem_l3976_397617

theorem min_value_theorem (x y : ℝ) 
  (h : Real.exp x + x - 2023 = Real.exp 2023 / (y + 2023) - Real.log (y + 2023)) :
  (∀ x' y' : ℝ, Real.exp x' + x' - 2023 = Real.exp 2023 / (y' + 2023) - Real.log (y' + 2023) → 
    Real.exp x' + y' + 2024 ≥ Real.exp x + y + 2024) →
  Real.exp x + y + 2024 = 2 * Real.sqrt (Real.exp 2023) + 1 := by
sorry

end min_value_theorem_l3976_397617


namespace pencil_sharpening_l3976_397693

/-- The length sharpened off a pencil is equal to the difference between its initial and final lengths. -/
theorem pencil_sharpening (initial_length final_length : ℝ) :
  initial_length ≥ final_length →
  initial_length - final_length = initial_length - final_length :=
by
  sorry

end pencil_sharpening_l3976_397693


namespace three_numbers_sum_l3976_397623

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →
  y = 5 →
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 := by
sorry

end three_numbers_sum_l3976_397623


namespace circle_radius_implies_m_value_l3976_397682

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def given_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y + m = 0

theorem circle_radius_implies_m_value :
  ∀ m : ℝ, 
  (∃ h k : ℝ, ∀ x y : ℝ, given_equation x y m ↔ circle_equation x y h k 3) →
  m = -7 := by
sorry

end circle_radius_implies_m_value_l3976_397682


namespace geometric_sequence_ratio_l3976_397626

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with ratio q
  a 1 * a 2 * a 3 = 2 →             -- First condition
  a 2 * a 3 * a 4 = 16 →            -- Second condition
  q = 2 :=                          -- Conclusion to prove
by
  sorry

end geometric_sequence_ratio_l3976_397626


namespace gavin_shirts_count_l3976_397619

theorem gavin_shirts_count (blue_shirts green_shirts : ℕ) :
  blue_shirts = 6 →
  green_shirts = 17 →
  blue_shirts + green_shirts = 23 :=
by sorry

end gavin_shirts_count_l3976_397619


namespace square_minus_self_sum_l3976_397685

theorem square_minus_self_sum : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8 := by
  sorry

end square_minus_self_sum_l3976_397685


namespace quadratic_properties_l3976_397667

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_neg : a < 0
  root_neg_one : a * (-1)^2 + b * (-1) + c = 0
  symmetry_axis : -b / (2 * a) = 1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a - f.b + f.c = 0) ∧
  (∀ m : ℝ, f.a * m^2 + f.b * m + f.c ≤ -4 * f.a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → 
    f.a * x₁^2 + f.b * x₁ + f.c + 1 = 0 → 
    f.a * x₂^2 + f.b * x₂ + f.c + 1 = 0 → 
    x₁ < -1 ∧ x₂ > 3) :=
by sorry

end quadratic_properties_l3976_397667


namespace equilateral_triangle_area_perimeter_ratio_l3976_397636

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry


end equilateral_triangle_area_perimeter_ratio_l3976_397636


namespace max_good_sequences_theorem_l3976_397645

/-- A necklace with blue, red, and green beads. -/
structure Necklace where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- A sequence of four consecutive beads in the necklace. -/
structure Sequence where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- A "good" sequence contains exactly 2 blue beads, 1 red bead, and 1 green bead. -/
def is_good (s : Sequence) : Prop :=
  s.blue = 2 ∧ s.red = 1 ∧ s.green = 1

/-- The maximum number of good sequences in a necklace. -/
def max_good_sequences (n : Necklace) : ℕ := sorry

/-- Theorem: The maximum number of good sequences in a necklace with 50 blue, 100 red, and 100 green beads is 99. -/
theorem max_good_sequences_theorem (n : Necklace) (h : n.blue = 50 ∧ n.red = 100 ∧ n.green = 100) :
  max_good_sequences n = 99 := by sorry

end max_good_sequences_theorem_l3976_397645


namespace triangle_problem_l3976_397686

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  -- a, b, c are sides opposite to angles A, B, C
  a = 4 ∧
  b = 5 ∧
  -- Area of triangle ABC is 5√3
  (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3 →
  c = Real.sqrt 21 ∧
  Real.sin A = (2 * Real.sqrt 7) / 7 := by
sorry

end triangle_problem_l3976_397686


namespace line_ellipse_intersection_range_l3976_397659

/-- The range of m for which the line y = kx + 1 and the ellipse x²/5 + y²/m = 1 always intersect -/
theorem line_ellipse_intersection_range (k : ℝ) (m : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1 → m ≥ 1 ∧ m ≠ 5) ∧
  (m ≥ 1 ∧ m ≠ 5 → ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) :=
sorry

end line_ellipse_intersection_range_l3976_397659


namespace inequality_problem_l3976_397647

theorem inequality_problem (s r p q : ℝ) 
  (hs : s > 0) 
  (hr : r > 0) 
  (hpq : p * q ≠ 0) 
  (hineq : s * (p * r) > s * (q * r)) : 
  ¬(-p > -q) ∧ ¬(-p > q) ∧ ¬(1 > -q/p) ∧ ¬(1 < q/p) := by
  sorry

end inequality_problem_l3976_397647


namespace inequality_implies_a_range_l3976_397658

theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, (x + Real.log a) / Real.exp x - a * Real.log x / x > 0) →
  a ∈ Set.Icc (Real.exp (-1)) 1 ∧ a ≠ 1 :=
by sorry

end inequality_implies_a_range_l3976_397658


namespace seating_arrangement_count_l3976_397640

/-- The number of seats at the bus station -/
def total_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- Calculate the number of ways to arrange seating -/
def seating_arrangements (total : ℕ) (passengers : ℕ) (empty_block : ℕ) : ℕ :=
  (Nat.factorial passengers) * (Nat.factorial (total - passengers - empty_block + 1) / Nat.factorial (total - passengers - empty_block - 1))

theorem seating_arrangement_count : 
  seating_arrangements total_seats num_passengers consecutive_empty_seats = 480 := by
  sorry

end seating_arrangement_count_l3976_397640


namespace part_one_part_two_l3976_397691

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the solution set A
def A (a : ℝ) : Set ℝ := {x | inequality a x}

-- Define set B
def B : Set ℝ := Set.Ioo (-2) 2

-- Part 1
theorem part_one : A 2 ∪ B = Set.Ioc (-2) 3 := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≤ -3 ∨ a ≥ 2 := by sorry

end part_one_part_two_l3976_397691


namespace amount_ratio_l3976_397680

theorem amount_ratio (total : ℕ) (r_amount : ℕ) : 
  total = 4000 →
  r_amount = 1600 →
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by
sorry

end amount_ratio_l3976_397680


namespace geometric_sequence_ratio_l3976_397635

theorem geometric_sequence_ratio (a₁ a₂ a₃ : ℝ) (h1 : a₁ = 9) (h2 : a₂ = -18) (h3 : a₃ = 36) :
  ∃ r : ℝ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -2 := by
sorry

end geometric_sequence_ratio_l3976_397635


namespace dice_roll_probability_l3976_397630

def standard_die := Finset.range 6

def valid_roll (a b c : ℕ) : Prop :=
  (a - 1) * (b - 1) * (c - 1) * (6 - a) * (6 - b) * (6 - c) ≠ 0

def total_outcomes : ℕ := standard_die.card ^ 3

def successful_outcomes : ℕ := ({2, 3, 4, 5} : Finset ℕ).card ^ 3

theorem dice_roll_probability :
  (successful_outcomes : ℚ) / total_outcomes = 8 / 27 := by
  sorry

end dice_roll_probability_l3976_397630


namespace smallest_multiples_sum_l3976_397649

theorem smallest_multiples_sum :
  ∀ c d : ℕ,
  (c ≥ 10 ∧ c < 100 ∧ c % 5 = 0 ∧ ∀ x : ℕ, (x ≥ 10 ∧ x < 100 ∧ x % 5 = 0) → c ≤ x) →
  (d ≥ 100 ∧ d < 1000 ∧ d % 7 = 0 ∧ ∀ y : ℕ, (y ≥ 100 ∧ y < 1000 ∧ y % 7 = 0) → d ≤ y) →
  c + d = 115 :=
by
  sorry

end smallest_multiples_sum_l3976_397649
