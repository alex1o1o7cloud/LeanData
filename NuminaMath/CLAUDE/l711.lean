import Mathlib

namespace prob_three_blue_value_l711_71146

/-- The number of red balls in the urn -/
def num_red : ℕ := 8

/-- The number of blue balls in the urn -/
def num_blue : ℕ := 6

/-- The total number of balls in the urn -/
def total_balls : ℕ := num_red + num_blue

/-- The number of balls drawn -/
def num_drawn : ℕ := 3

/-- The probability of drawing 3 blue balls consecutively without replacement -/
def prob_three_blue : ℚ := (num_blue.choose num_drawn) / (total_balls.choose num_drawn)

theorem prob_three_blue_value : prob_three_blue = 5/91 := by sorry

end prob_three_blue_value_l711_71146


namespace odd_square_minus_one_divisible_by_eight_l711_71129

theorem odd_square_minus_one_divisible_by_eight (n : ℤ) (h : Odd n) : 
  ∃ k : ℤ, n^2 - 1 = 8 * k := by
sorry

end odd_square_minus_one_divisible_by_eight_l711_71129


namespace alyssa_future_games_l711_71185

/-- The number of soccer games Alyssa plans to attend next year -/
def games_next_year (games_this_year games_last_year total_games : ℕ) : ℕ :=
  total_games - (games_this_year + games_last_year)

/-- Theorem stating that Alyssa plans to attend 15 soccer games next year -/
theorem alyssa_future_games : 
  games_next_year 11 13 39 = 15 := by
  sorry

end alyssa_future_games_l711_71185


namespace grace_pumpkin_pie_fraction_l711_71193

theorem grace_pumpkin_pie_fraction :
  let total_pies : ℕ := 4
  let sold_pies : ℕ := 1
  let given_pies : ℕ := 1
  let slices_per_pie : ℕ := 6
  let remaining_slices : ℕ := 4
  
  let remaining_pies : ℕ := total_pies - sold_pies - given_pies
  let total_slices : ℕ := remaining_pies * slices_per_pie
  let eaten_slices : ℕ := total_slices - remaining_slices
  
  (eaten_slices : ℚ) / total_slices = 2 / 3 := by
  sorry

end grace_pumpkin_pie_fraction_l711_71193


namespace chord_ratio_is_sqrt6_to_2_l711_71107

-- Define the points and circles
structure PointOnLine where
  position : ℝ

structure Circle where
  center : ℝ
  radius : ℝ

-- Define the problem setup
def setup (A B C D : PointOnLine) (circle_AB circle_BC circle_CD : Circle) :=
  -- Points are on a line and equally spaced
  B.position - A.position = C.position - B.position ∧
  C.position - B.position = D.position - C.position ∧
  -- Circles have diameters AB, BC, and CD
  circle_AB.radius = (B.position - A.position) / 2 ∧
  circle_BC.radius = (C.position - B.position) / 2 ∧
  circle_CD.radius = (D.position - C.position) / 2 ∧
  circle_AB.center = (A.position + B.position) / 2 ∧
  circle_BC.center = (B.position + C.position) / 2 ∧
  circle_CD.center = (C.position + D.position) / 2

-- Define the tangent line and chords
def tangent_and_chords (A : PointOnLine) (circle_CD : Circle) (chord_AB chord_BC : ℝ) :=
  ∃ (l : ℝ → ℝ), 
    -- l is tangent to circle_CD at point A
    (l A.position - circle_CD.center)^2 = circle_CD.radius^2 ∧
    -- chord_AB and chord_BC are the lengths of the chords cut by l on circles with diameters AB and BC
    chord_AB > 0 ∧ chord_BC > 0

-- The main theorem
theorem chord_ratio_is_sqrt6_to_2 
  (A B C D : PointOnLine) 
  (circle_AB circle_BC circle_CD : Circle) 
  (chord_AB chord_BC : ℝ) :
  setup A B C D circle_AB circle_BC circle_CD →
  tangent_and_chords A circle_CD chord_AB chord_BC →
  chord_AB / chord_BC = Real.sqrt 6 / 2 :=
sorry

end chord_ratio_is_sqrt6_to_2_l711_71107


namespace height_relation_l711_71123

/-- Two right circular cylinders with equal volumes and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  volume_eq : r1^2 * h1 = r2^2 * h2  -- volumes are equal
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry


end height_relation_l711_71123


namespace two_digit_puzzle_solution_l711_71101

theorem two_digit_puzzle_solution :
  ∃ (A B : ℕ), 
    A ≠ B ∧ 
    A ≠ 0 ∧ 
    A < 10 ∧ 
    B < 10 ∧ 
    A * B + A + B = 10 * A + B :=
by
  sorry

end two_digit_puzzle_solution_l711_71101


namespace min_value_theorem_l711_71155

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 25 / x ≥ 10 ∧ (x + 25 / x = 10 ↔ x = 5) := by
  sorry

end min_value_theorem_l711_71155


namespace largest_dividend_l711_71139

theorem largest_dividend (dividend quotient divisor remainder : ℕ) : 
  dividend = quotient * divisor + remainder →
  remainder < divisor →
  quotient = 32 →
  divisor = 18 →
  dividend ≤ 593 := by
sorry

end largest_dividend_l711_71139


namespace park_visitors_l711_71199

/-- Given a park with visitors on Saturday and Sunday, calculate the total number of visitors over two days. -/
theorem park_visitors (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
  sorry

#check park_visitors

end park_visitors_l711_71199


namespace derivative_sin_cos_product_l711_71178

theorem derivative_sin_cos_product (x : ℝ) :
  deriv (fun x => 2 * Real.sin x * Real.cos x) x = 2 * Real.cos (2 * x) := by
  sorry

end derivative_sin_cos_product_l711_71178


namespace correlation_coefficient_formula_correlation_coefficient_problem_l711_71167

/-- Given a linear regression equation ŷ = bx + a, where b is the slope,
    Sy^2 is the variance of y, and Sx^2 is the variance of x,
    prove that the correlation coefficient r = b * (√(Sx^2) / √(Sy^2)) -/
theorem correlation_coefficient_formula 
  (b : ℝ) (Sy_squared : ℝ) (Sx_squared : ℝ) (h1 : Sy_squared > 0) (h2 : Sx_squared > 0) :
  let r := b * (Real.sqrt Sx_squared / Real.sqrt Sy_squared)
  ∀ ε > 0, |r - 0.94| < ε := by
sorry

/-- Given the specific values from the problem, prove that the correlation coefficient is 0.94 -/
theorem correlation_coefficient_problem :
  let b := 4.7
  let Sy_squared := 50
  let Sx_squared := 2
  let r := b * (Real.sqrt Sx_squared / Real.sqrt Sy_squared)
  ∀ ε > 0, |r - 0.94| < ε := by
sorry

end correlation_coefficient_formula_correlation_coefficient_problem_l711_71167


namespace key_chain_manufacturing_cost_l711_71111

theorem key_chain_manufacturing_cost 
  (selling_price : ℝ)
  (old_profit_percentage : ℝ)
  (new_profit_percentage : ℝ)
  (new_manufacturing_cost : ℝ)
  (h1 : old_profit_percentage = 0.3)
  (h2 : new_profit_percentage = 0.5)
  (h3 : new_manufacturing_cost = 50)
  (h4 : selling_price = new_manufacturing_cost / (1 - new_profit_percentage)) :
  selling_price * (1 - old_profit_percentage) = 70 := by
sorry

end key_chain_manufacturing_cost_l711_71111


namespace playground_girls_l711_71189

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) :
  total_children = 63 → boys = 35 → girls = total_children - boys → girls = 28 := by
  sorry

end playground_girls_l711_71189


namespace bird_stork_difference_is_one_l711_71134

/-- Given an initial number of birds on a fence, and additional birds and storks that join,
    calculate the difference between the final number of storks and birds. -/
def fence_bird_stork_difference (initial_birds : ℕ) (joined_birds : ℕ) (joined_storks : ℕ) : ℤ :=
  (joined_storks : ℤ) - ((initial_birds + joined_birds) : ℤ)

/-- Theorem stating that with 3 initial birds, 2 joined birds, and 6 joined storks,
    there is 1 more stork than birds on the fence. -/
theorem bird_stork_difference_is_one :
  fence_bird_stork_difference 3 2 6 = 1 := by
  sorry

end bird_stork_difference_is_one_l711_71134


namespace parabola_tangent_line_l711_71157

/-- Given a parabola y = x^2 + 1, prove that the equation of the tangent line
    passing through the point (0,0) is either 2x - y = 0 or 2x + y = 0. -/
theorem parabola_tangent_line (x y : ℝ) :
  y = x^2 + 1 →
  (∃ (m : ℝ), y = m*x ∧ 0 = 0^2 + 1) →
  (y = 2*x ∨ y = -2*x) :=
by sorry

end parabola_tangent_line_l711_71157


namespace danny_bottle_caps_l711_71106

/-- Represents the number of bottle caps Danny found at the park -/
def new_bottle_caps : ℕ := 50

/-- Represents the number of old bottle caps Danny threw away -/
def thrown_away_caps : ℕ := 6

/-- Represents the current number of bottle caps in Danny's collection -/
def current_collection : ℕ := 60

/-- Represents the difference between found and thrown away caps -/
def difference_found_thrown : ℕ := 44

theorem danny_bottle_caps :
  new_bottle_caps = thrown_away_caps + difference_found_thrown ∧
  current_collection = (new_bottle_caps + thrown_away_caps) - thrown_away_caps :=
by sorry

end danny_bottle_caps_l711_71106


namespace arithmetic_sequence_2005_l711_71175

/-- 
Given an arithmetic sequence with first term a₁ = -1 and common difference d = 2,
prove that the 1004th term is equal to 2005.
-/
theorem arithmetic_sequence_2005 :
  let a : ℕ → ℤ := λ n => -1 + (n - 1) * 2
  a 1004 = 2005 := by
  sorry

end arithmetic_sequence_2005_l711_71175


namespace triple_nested_log_sum_l711_71116

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Define the theorem
theorem triple_nested_log_sum (x y z : ℝ) :
  log 3 (log 4 (log 5 x)) = 0 ∧
  log 4 (log 5 (log 3 y)) = 0 ∧
  log 5 (log 3 (log 4 z)) = 0 →
  x + y + z = 932 := by
  sorry

end triple_nested_log_sum_l711_71116


namespace equation_solution_l711_71153

theorem equation_solution : ∃ x : ℝ, 2 * x + 6 = 2 + 3 * x ∧ x = 4 := by
  sorry

end equation_solution_l711_71153


namespace gcd_2134_1455_ternary_l711_71117

theorem gcd_2134_1455_ternary : 
  ∃ m : ℕ, 
    Nat.gcd 2134 1455 = m ∧ 
    (Nat.digits 3 m).reverse = [1, 0, 1, 2, 1] :=
by sorry

end gcd_2134_1455_ternary_l711_71117


namespace soccer_team_starters_l711_71187

theorem soccer_team_starters (n : ℕ) (q : ℕ) (s : ℕ) (qc : ℕ) :
  n = 15 →  -- Total number of players
  q = 4 →   -- Number of quadruplets
  s = 7 →   -- Number of starters
  qc = 2 →  -- Number of quadruplets in starting lineup
  (Nat.choose q qc) * (Nat.choose (n - q) (s - qc)) = 2772 := by
sorry

end soccer_team_starters_l711_71187


namespace coin_value_equality_l711_71192

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Theorem stating that if the value of 25 quarters, 15 dimes, and 10 nickels 
    equals the value of 15 quarters, n dimes, and 20 nickels, then n = 35 -/
theorem coin_value_equality (n : ℕ) : 
  25 * quarter_value + 15 * dime_value + 10 * nickel_value = 
  15 * quarter_value + n * dime_value + 20 * nickel_value → n = 35 := by
  sorry

end coin_value_equality_l711_71192


namespace butterfly_ratio_l711_71177

/-- Prove that the ratio of blue butterflies to yellow butterflies is 2:1 -/
theorem butterfly_ratio (total : ℕ) (black : ℕ) (blue : ℕ) 
  (h1 : total = 11)
  (h2 : black = 5)
  (h3 : blue = 4)
  : (blue : ℚ) / (total - black - blue) = 2 / 1 := by
  sorry

end butterfly_ratio_l711_71177


namespace point_p_position_l711_71119

/-- Given seven points O, A, B, C, D, E, F on a line, with specified distances from O,
    and a point P between D and E satisfying a ratio condition,
    prove that OP has a specific value. -/
theorem point_p_position
  (a b c d e f : ℝ)  -- Real parameters for distances
  (O A B C D E F P : ℝ)  -- Points on the real line
  (h1 : O = 0)  -- O is the origin
  (h2 : A = 2*a)
  (h3 : B = 5*b)
  (h4 : C = 9*c)
  (h5 : D = 12*d)
  (h6 : E = 15*e)
  (h7 : F = 20*f)
  (h8 : D ≤ P ∧ P ≤ E)  -- P is between D and E
  (h9 : (P - A) / (F - P) = (P - D) / (E - P))  -- Ratio condition
  : P = (300*a*e - 240*d*f) / (2*a - 15*e + 20*f) :=
sorry

end point_p_position_l711_71119


namespace min_max_abs_polynomial_l711_71126

open Real

theorem min_max_abs_polynomial :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 - x^3 * z| ≤ |x^2 - x^3 * y|) ∧
    |x^2 - x^3 * y| ≤ 0 :=
by sorry

end min_max_abs_polynomial_l711_71126


namespace derivative_f_at_zero_l711_71124

def f (x : ℝ) : ℝ := x^3

theorem derivative_f_at_zero : 
  deriv f 0 = 0 := by sorry

end derivative_f_at_zero_l711_71124


namespace childrens_ticket_cost_l711_71160

/-- Given ticket information, prove the cost of a children's ticket -/
theorem childrens_ticket_cost 
  (adult_ticket_cost : ℝ) 
  (total_tickets : ℕ) 
  (total_cost : ℝ) 
  (childrens_tickets : ℕ) 
  (h1 : adult_ticket_cost = 5.50)
  (h2 : total_tickets = 21)
  (h3 : total_cost = 83.50)
  (h4 : childrens_tickets = 16) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost * childrens_tickets + 
    adult_ticket_cost * (total_tickets - childrens_tickets) = total_cost ∧ 
    childrens_ticket_cost = 3.50 :=
by sorry

end childrens_ticket_cost_l711_71160


namespace rectangular_to_polar_conversion_l711_71120

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 7 * Real.pi / 4
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 2 * Real.sqrt 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by
  sorry

end rectangular_to_polar_conversion_l711_71120


namespace parabola_ellipse_tangency_l711_71183

theorem parabola_ellipse_tangency (a b : ℝ) :
  (∀ x y : ℝ, y = x^2 - 5 → x^2/a + y^2/b = 1) →
  (∃! p : ℝ × ℝ, (p.2 = p.1^2 - 5) ∧ (p.1^2/a + p.2^2/b = 1)) →
  a = 1/10 ∧ b = 1 := by
  sorry

end parabola_ellipse_tangency_l711_71183


namespace field_length_proof_l711_71136

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 8 →
  pond_side ^ 2 = (1 / 18) * (length * width) →
  length = 48 := by
  sorry

end field_length_proof_l711_71136


namespace three_divides_difference_l711_71140

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Reverses a three-digit number -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber :=
  { hundreds := n.ones
  , tens := n.tens
  , ones := n.hundreds
  , is_valid := by sorry }

/-- Converts a ThreeDigitNumber to a natural number -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The difference between a number and its reverse -/
def difference (n : ThreeDigitNumber) : Int :=
  Int.natAbs (to_nat n - to_nat (reverse n))

theorem three_divides_difference (n : ThreeDigitNumber) (h : n.hundreds ≠ n.ones) :
  3 ∣ difference n := by
  sorry

end three_divides_difference_l711_71140


namespace bc_is_one_eighth_of_ad_l711_71184

/-- Given a line segment AD with points B and C on it,
    prove that BC is 1/8 of AD if AB is 3 times BD and AC is 7 times CD -/
theorem bc_is_one_eighth_of_ad 
  (A B C D : EuclideanSpace ℝ (Fin 1))
  (h_B_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B = (1 - t) • A + t • D)
  (h_C_on_AD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ C = (1 - s) • A + s • D)
  (h_AB_3BD : dist A B = 3 * dist B D)
  (h_AC_7CD : dist A C = 7 * dist C D) :
  dist B C = (1 / 8 : ℝ) * dist A D :=
sorry

end bc_is_one_eighth_of_ad_l711_71184


namespace unique_positive_solution_l711_71113

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x :=
by
  sorry

end unique_positive_solution_l711_71113


namespace matrix_equation_solution_l711_71170

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-19, -7; 10, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![85/14, -109/14; -3, 4]
  N * A = B := by sorry

end matrix_equation_solution_l711_71170


namespace four_row_lattice_triangles_l711_71122

/-- Represents a modified triangular lattice with n rows -/
structure ModifiedTriangularLattice (n : ℕ) where
  -- Each row i has i dots, with the base row having n dots
  rows : Fin n → ℕ
  rows_def : ∀ i : Fin n, rows i = i.val + 1

/-- Counts the number of triangles in a modified triangular lattice -/
def countTriangles (n : ℕ) : ℕ :=
  let lattice := ModifiedTriangularLattice n
  -- The actual counting logic would go here
  0 -- Placeholder

/-- The theorem stating that a 4-row modified triangular lattice contains 22 triangles -/
theorem four_row_lattice_triangles :
  countTriangles 4 = 22 := by
  sorry

#check four_row_lattice_triangles

end four_row_lattice_triangles_l711_71122


namespace expansion_terms_imply_n_12_l711_71151

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem expansion_terms_imply_n_12 (x a : ℝ) (n : ℕ) :
  (binomial_coefficient n 3 * x^(n-3) * a^3 = 120) →
  (binomial_coefficient n 4 * x^(n-4) * a^4 = 360) →
  (binomial_coefficient n 5 * x^(n-5) * a^5 = 720) →
  n = 12 :=
sorry

end expansion_terms_imply_n_12_l711_71151


namespace divisible_by_27_l711_71176

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, (10 : ℤ)^n + 18*n - 1 = 27*k := by
  sorry

end divisible_by_27_l711_71176


namespace circle_B_radius_l711_71110

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (A B C D : Circle) : Prop :=
  -- Circles A, B, and C are externally tangent to each other
  (A.radius + B.radius = dist A.center B.center) ∧
  (A.radius + C.radius = dist A.center C.center) ∧
  (B.radius + C.radius = dist B.center C.center) ∧
  -- Circles A, B, and C are internally tangent to circle D
  (D.radius - A.radius = dist D.center A.center) ∧
  (D.radius - B.radius = dist D.center B.center) ∧
  (D.radius - C.radius = dist D.center C.center) ∧
  -- Circles B and C are congruent
  (B.radius = C.radius) ∧
  -- Circle A has radius 1
  (A.radius = 1) ∧
  -- Circle A passes through the center of D
  (dist A.center D.center = A.radius + D.radius)

-- Theorem statement
theorem circle_B_radius (A B C D : Circle) :
  problem_setup A B C D → B.radius = 8/9 := by
  sorry

end circle_B_radius_l711_71110


namespace family_reunion_soda_cost_l711_71138

-- Define the given conditions
def people_attending : ℕ := 5 * 12
def cans_per_box : ℕ := 10
def cost_per_box : ℚ := 2
def cans_per_person : ℕ := 2
def family_members : ℕ := 6

-- Define the theorem
theorem family_reunion_soda_cost :
  (people_attending * cans_per_person / cans_per_box * cost_per_box) / family_members = 4 := by
  sorry

end family_reunion_soda_cost_l711_71138


namespace sandy_marks_l711_71145

/-- Sandy's marks calculation -/
theorem sandy_marks :
  ∀ (total_sums correct_sums : ℕ)
    (marks_per_correct marks_lost_per_incorrect : ℕ),
  total_sums = 30 →
  correct_sums = 23 →
  marks_per_correct = 3 →
  marks_lost_per_incorrect = 2 →
  (marks_per_correct * correct_sums) -
  (marks_lost_per_incorrect * (total_sums - correct_sums)) = 55 :=
by
  sorry

end sandy_marks_l711_71145


namespace smallest_square_partition_l711_71173

/-- A partition of a square into smaller squares -/
structure SquarePartition (n : ℕ) :=
  (num_40 : ℕ)  -- number of 40x40 squares
  (num_49 : ℕ)  -- number of 49x49 squares
  (valid : num_40 * 40 * 40 + num_49 * 49 * 49 = n * n)
  (both_present : num_40 > 0 ∧ num_49 > 0)

/-- The main theorem stating that 2000 is the smallest n that satisfies the conditions -/
theorem smallest_square_partition :
  (∃ (p : SquarePartition 2000), True) ∧
  (∀ n < 2000, ¬ ∃ (p : SquarePartition n), True) :=
sorry

end smallest_square_partition_l711_71173


namespace arithmetic_sequence_sum_l711_71118

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if -a_{2015} < a_1 < -a_{2016}, then S_{2015} > 0 and S_{2016} < 0. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_inequality : -a 2015 < a 1 ∧ a 1 < -a 2016) :
  S 2015 > 0 ∧ S 2016 < 0 := by
  sorry


end arithmetic_sequence_sum_l711_71118


namespace project_work_time_difference_l711_71194

theorem project_work_time_difference (x : ℝ) 
  (h1 : x > 0)
  (h2 : 2*x + 3*x + 4*x = 90) : 4*x - 2*x = 20 := by
  sorry

end project_work_time_difference_l711_71194


namespace max_product_l711_71154

theorem max_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) :
  a * b ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4 * b₀ = 8 ∧ a₀ * b₀ = 4 :=
sorry

end max_product_l711_71154


namespace binomial_20_19_equals_20_l711_71165

theorem binomial_20_19_equals_20 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_20_19_equals_20_l711_71165


namespace daily_evaporation_rate_l711_71148

/-- Calculates the daily evaporation rate given initial water amount, time period, and evaporation percentage. -/
theorem daily_evaporation_rate 
  (initial_water : ℝ) 
  (days : ℕ) 
  (evaporation_percentage : ℝ) : 
  initial_water * evaporation_percentage / 100 / days = 0.1 :=
by
  -- Assuming initial_water = 10, days = 20, and evaporation_percentage = 2
  sorry

#check daily_evaporation_rate

end daily_evaporation_rate_l711_71148


namespace total_distance_walked_l711_71197

/-- Represents the hiking trail with flat and uphill sections -/
structure HikingTrail where
  flat_distance : ℝ  -- Distance of flat section (P to Q)
  uphill_distance : ℝ  -- Distance of uphill section (Q to R)

/-- Represents the hiker's journey -/
structure HikerJourney where
  trail : HikingTrail
  flat_speed : ℝ  -- Speed on flat sections
  uphill_speed : ℝ  -- Speed going uphill
  downhill_speed : ℝ  -- Speed going downhill
  total_time : ℝ  -- Total time of the journey in hours
  rest_time : ℝ  -- Time spent resting at point R

/-- Theorem stating the total distance walked by the hiker -/
theorem total_distance_walked (journey : HikerJourney) 
  (h1 : journey.flat_speed = 4)
  (h2 : journey.uphill_speed = 3)
  (h3 : journey.downhill_speed = 6)
  (h4 : journey.total_time = 7)
  (h5 : journey.rest_time = 1)
  (h6 : journey.flat_speed * (journey.total_time - journey.rest_time) / 2 + 
        journey.trail.uphill_distance * (1 / journey.uphill_speed + 1 / journey.downhill_speed) = 
        journey.total_time - journey.rest_time) :
  2 * (journey.trail.flat_distance + journey.trail.uphill_distance) = 24 := by
  sorry

end total_distance_walked_l711_71197


namespace homework_questions_l711_71163

theorem homework_questions (first_hour second_hour third_hour : ℕ) : 
  third_hour = 132 → 
  third_hour = 2 * second_hour → 
  third_hour = 3 * first_hour → 
  first_hour + second_hour + third_hour = 264 :=
by
  sorry

end homework_questions_l711_71163


namespace maddie_bought_two_white_packs_l711_71168

/-- Represents the problem of determining the number of packs of white T-shirts Maddie bought. -/
def maddies_tshirt_problem (white_packs : ℕ) : Prop :=
  let blue_packs : ℕ := 4
  let white_per_pack : ℕ := 5
  let blue_per_pack : ℕ := 3
  let cost_per_shirt : ℕ := 3
  let total_spent : ℕ := 66
  
  (white_packs * white_per_pack + blue_packs * blue_per_pack) * cost_per_shirt = total_spent

/-- Theorem stating that Maddie bought 2 packs of white T-shirts. -/
theorem maddie_bought_two_white_packs : ∃ (white_packs : ℕ), white_packs = 2 ∧ maddies_tshirt_problem white_packs :=
sorry

end maddie_bought_two_white_packs_l711_71168


namespace period_start_time_l711_71103

def period_end : Nat := 17  -- 5 pm in 24-hour format
def rain_duration : Nat := 2
def no_rain_duration : Nat := 6

theorem period_start_time : 
  period_end - (rain_duration + no_rain_duration) = 9 := by
  sorry

end period_start_time_l711_71103


namespace isotopes_same_count_atom_molecule_same_count_molecules_same_count_cations_same_count_anions_same_count_different_elements_different_count_atom_ion_different_count_molecule_ion_different_count_anion_cation_different_count_l711_71100

-- Define the basic types
inductive ParticleType
| Atom
| Molecule
| Cation
| Anion

-- Define a particle
structure Particle where
  type : ParticleType
  protons : ℕ
  electrons : ℕ

-- Define the property of having the same number of protons and electrons
def sameProtonElectronCount (p1 p2 : Particle) : Prop :=
  p1.protons = p2.protons ∧ p1.electrons = p2.electrons

-- Theorem: Two different atoms (isotopes) can have the same number of protons and electrons
theorem isotopes_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Atom ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: An atom and a molecule can have the same number of protons and electrons
theorem atom_molecule_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Molecule ∧
  sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different molecules can have the same number of protons and electrons
theorem molecules_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Molecule ∧ p2.type = ParticleType.Molecule ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different cations can have the same number of protons and electrons
theorem cations_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Cation ∧ p2.type = ParticleType.Cation ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different anions can have the same number of protons and electrons
theorem anions_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Anion ∧ p2.type = ParticleType.Anion ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Atoms of two different elements cannot have the same number of protons and electrons
theorem different_elements_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Atom ∧
  p1.protons ≠ p2.protons → ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: An atom and an ion cannot have the same number of protons and electrons
theorem atom_ion_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ (p2.type = ParticleType.Cation ∨ p2.type = ParticleType.Anion) →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: A molecule and an ion cannot have the same number of protons and electrons
theorem molecule_ion_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Molecule ∧ (p2.type = ParticleType.Cation ∨ p2.type = ParticleType.Anion) →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: An anion and a cation cannot have the same number of protons and electrons
theorem anion_cation_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Anion ∧ p2.type = ParticleType.Cation →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

end isotopes_same_count_atom_molecule_same_count_molecules_same_count_cations_same_count_anions_same_count_different_elements_different_count_atom_ion_different_count_molecule_ion_different_count_anion_cation_different_count_l711_71100


namespace arithmetic_sequence_properties_l711_71195

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 3)
  (h_sum : a 3 + a 11 = 18) :
  (∀ n : ℕ, a n = 2 * n - 5) ∧
  (a 55 = 105) ∧
  (∃ n : ℕ, n * (2 * n - 8) / 2 = 32 ∧ n = 8) ∧
  (∀ n : ℕ, n * (2 * n - 8) / 2 ≥ 2 * (2 * 2 - 8) / 2) :=
by sorry

end arithmetic_sequence_properties_l711_71195


namespace shirts_washed_l711_71115

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (not_washed : ℕ) 
  (h1 : short_sleeve = 40)
  (h2 : long_sleeve = 23)
  (h3 : not_washed = 34) :
  short_sleeve + long_sleeve - not_washed = 29 := by
  sorry

end shirts_washed_l711_71115


namespace factor_3x_squared_minus_75_l711_71162

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_3x_squared_minus_75_l711_71162


namespace bruce_total_payment_l711_71137

/-- Calculates the total amount Bruce paid for fruits -/
def total_amount_paid (grape_quantity grape_price mango_quantity mango_price
                       orange_quantity orange_price apple_quantity apple_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price +
  orange_quantity * orange_price + apple_quantity * apple_price

/-- Theorem: Bruce paid $1480 for the fruits -/
theorem bruce_total_payment :
  total_amount_paid 9 70 7 55 5 45 3 80 = 1480 := by
  sorry

end bruce_total_payment_l711_71137


namespace distance_to_office_l711_71161

theorem distance_to_office : 
  ∀ (v : ℝ) (d : ℝ),
  (d = v * (1/2)) →  -- Distance in heavy traffic
  (d = (v + 20) * (1/5)) →  -- Distance without traffic
  d = 20/3 := by
  sorry

end distance_to_office_l711_71161


namespace M_is_line_segment_l711_71181

-- Define the set of points M(x,y) satisfying the equation
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sqrt ((p.1 - 1)^2 + p.2^2) + Real.sqrt ((p.1 + 1)^2 + p.2^2) = 2}

-- Define the line segment between (-1,0) and (1,0)
def lineSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (2*t - 1, 0)}

-- Theorem stating that M is equal to the line segment
theorem M_is_line_segment : M = lineSegment := by sorry

end M_is_line_segment_l711_71181


namespace max_partner_share_l711_71156

def profit : ℕ := 36000
def ratio : List ℕ := [2, 4, 3, 5, 6]

theorem max_partner_share :
  let total_parts := ratio.sum
  let part_value := profit / total_parts
  let shares := ratio.map (· * part_value)
  shares.maximum? = some 10800 := by sorry

end max_partner_share_l711_71156


namespace sphere_radius_ratio_l711_71172

theorem sphere_radius_ratio (V_large V_small : ℝ) (r_large r_small : ℝ) : 
  V_large = 576 * Real.pi ∧ 
  V_small = 0.0625 * V_large ∧
  V_large = (4/3) * Real.pi * r_large^3 ∧
  V_small = (4/3) * Real.pi * r_small^3 →
  r_small / r_large = 1/2 := by
sorry

end sphere_radius_ratio_l711_71172


namespace inscribed_cylinder_radius_l711_71149

/-- A right circular cylinder inscribed in a right circular cone --/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  h_diameter_height : cylinder_radius * 2 = cylinder_radius * 2
  h_cone_cylinder_axes : True  -- This condition is implicit and cannot be directly expressed

/-- The radius of the inscribed cylinder is 90/19 --/
theorem inscribed_cylinder_radius 
  (c : InscribedCylinder) 
  (h_cone_diameter : c.cone_diameter = 18) 
  (h_cone_altitude : c.cone_altitude = 20) : 
  c.cylinder_radius = 90 / 19 := by
  sorry

end inscribed_cylinder_radius_l711_71149


namespace product_sum_theorem_l711_71169

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 5^4 → 
  (a : ℤ) + (b : ℤ) + (c : ℤ) = 131 := by
sorry

end product_sum_theorem_l711_71169


namespace senior_junior_ratio_l711_71127

/-- The ratio of senior class size to junior class size -/
def class_ratio (senior_size junior_size : ℚ) : ℚ := senior_size / junior_size

theorem senior_junior_ratio 
  (senior_size junior_size : ℚ)
  (h1 : senior_size > 0)
  (h2 : junior_size > 0)
  (h3 : ∃ k : ℚ, k > 0 ∧ senior_size = k * junior_size)
  (h4 : (3/8) * senior_size + (1/4) * junior_size = (1/3) * (senior_size + junior_size)) :
  class_ratio senior_size junior_size = 2 := by
  sorry

end senior_junior_ratio_l711_71127


namespace bug_probability_after_8_meters_l711_71196

/-- Probability of the bug being at vertex A after n meters -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The probability of the bug being at vertex A after 8 meters is 547/2187 -/
theorem bug_probability_after_8_meters : P 8 = 547/2187 := by
  sorry

end bug_probability_after_8_meters_l711_71196


namespace digit_sum_puzzle_l711_71158

theorem digit_sum_puzzle :
  ∀ (A B C D E F : ℕ),
  (A ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (B ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (C ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (D ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (E ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (F ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F →
  (A + B) % 2 = 0 →
  (C + D) % 3 = 0 →
  (E + F) % 5 = 0 →
  min C D = 1 :=
by sorry

end digit_sum_puzzle_l711_71158


namespace todd_initial_gum_l711_71125

/-- 
Given:
- Todd receives 16 pieces of gum from Steve.
- After receiving gum from Steve, Todd has 54 pieces of gum.

Prove that Todd initially had 38 pieces of gum.
-/
theorem todd_initial_gum (initial_gum : ℕ) : initial_gum + 16 = 54 ↔ initial_gum = 38 := by
  sorry

end todd_initial_gum_l711_71125


namespace price_adjustment_l711_71108

theorem price_adjustment (a : ℝ) : 
  let price_after_reductions := a * (1 - 0.1) * (1 - 0.1)
  let final_price := price_after_reductions * (1 + 0.2)
  final_price = 0.972 * a :=
by sorry

end price_adjustment_l711_71108


namespace vanessa_album_pictures_l711_71135

/-- Represents the number of pictures in an album -/
def pictures_per_album (phone_pics camera_pics total_albums : ℕ) : ℚ :=
  (phone_pics + camera_pics : ℚ) / total_albums

/-- Theorem stating the number of pictures per album given the conditions -/
theorem vanessa_album_pictures :
  pictures_per_album 56 28 8 = 21/2 := by sorry

end vanessa_album_pictures_l711_71135


namespace school_students_problem_l711_71144

theorem school_students_problem (total : ℕ) (x : ℕ) : 
  total = 1150 →
  (total - x : ℚ) = (x : ℚ) * (total : ℚ) / 100 →
  x = 92 := by
sorry

end school_students_problem_l711_71144


namespace quadratic_minimum_zero_l711_71179

/-- Given a quadratic function y = (1+a)x^2 + px + q with a minimum value of zero,
    where a is a positive constant, prove that q = p^2 / (4(1+a)). -/
theorem quadratic_minimum_zero (a p q : ℝ) (ha : a > 0) :
  (∃ (k : ℝ), ∀ (x : ℝ), (1 + a) * x^2 + p * x + q ≥ k) ∧ 
  (∃ (x : ℝ), (1 + a) * x^2 + p * x + q = 0) →
  q = p^2 / (4 * (1 + a)) := by
  sorry

end quadratic_minimum_zero_l711_71179


namespace courtyard_width_l711_71105

theorem courtyard_width (length : ℝ) (width : ℝ) (stone_length : ℝ) (stone_width : ℝ) 
  (total_stones : ℕ) (h1 : length = 30) (h2 : stone_length = 2) (h3 : stone_width = 1) 
  (h4 : total_stones = 240) (h5 : length * width = stone_length * stone_width * total_stones) : 
  width = 16 := by
  sorry

end courtyard_width_l711_71105


namespace train_length_calculation_l711_71174

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (cross_time_s : ℝ) :
  speed_kmh = 56 →
  cross_time_s = 9 →
  ∃ (length_m : ℝ), 139 < length_m ∧ length_m < 141 :=
by
  sorry

#check train_length_calculation

end train_length_calculation_l711_71174


namespace parcel_weight_theorem_l711_71191

theorem parcel_weight_theorem (x y z : ℕ) 
  (h1 : x + y = 132)
  (h2 : y + z = 135)
  (h3 : z + x = 140) :
  x + y + z = 204 := by
sorry

end parcel_weight_theorem_l711_71191


namespace vector_addition_l711_71109

/-- Given two vectors a and b in ℝ², prove that 2b + 3a equals (6,1) -/
theorem vector_addition (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (0, -1)) :
  2 • b + 3 • a = (6, 1) := by
  sorry

end vector_addition_l711_71109


namespace water_dumped_calculation_l711_71130

/-- Calculates the amount of water dumped out of a bathtub given specific conditions --/
theorem water_dumped_calculation (faucet_rate : ℝ) (evaporation_rate : ℝ) (time : ℝ) (water_left : ℝ) : 
  faucet_rate = 40 ∧ 
  evaporation_rate = 200 / 60 ∧ 
  time = 9 * 60 ∧ 
  water_left = 7800 → 
  (faucet_rate * time - evaporation_rate * time - water_left) / 1000 = 12 := by
  sorry


end water_dumped_calculation_l711_71130


namespace profit_function_max_profit_profit_2400_l711_71142

-- Define the cost price
def cost_price : ℝ := 80

-- Define the sales quantity function
def sales_quantity (x : ℝ) : ℝ := -2 * x + 320

-- Define the valid price range
def valid_price (x : ℝ) : Prop := 80 ≤ x ∧ x ≤ 160

-- Define the daily profit function
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * sales_quantity x

-- Theorem statements
theorem profit_function (x : ℝ) (h : valid_price x) :
  daily_profit x = -2 * x^2 + 480 * x - 25600 := by sorry

theorem max_profit (x : ℝ) (h : valid_price x) :
  daily_profit x ≤ 3200 ∧ daily_profit 120 = 3200 := by sorry

theorem profit_2400 :
  ∃ x, valid_price x ∧ daily_profit x = 2400 ∧
  ∀ y, valid_price y → daily_profit y = 2400 → x ≤ y := by sorry

end profit_function_max_profit_profit_2400_l711_71142


namespace spongebob_fries_sold_l711_71190

/-- Calculates the number of large fries sold given the total earnings, 
    number of burgers sold, price per burger, and price per large fries. -/
def large_fries_sold (total_earnings : ℚ) (num_burgers : ℕ) (price_burger : ℚ) (price_fries : ℚ) : ℚ :=
  (total_earnings - num_burgers * price_burger) / price_fries

/-- Proves that Spongebob sold 12 large fries given the conditions -/
theorem spongebob_fries_sold : 
  large_fries_sold 78 30 2 (3/2) = 12 := by
  sorry

end spongebob_fries_sold_l711_71190


namespace composite_product_division_l711_71112

def first_five_composites : List Nat := [12, 14, 15, 16, 18]
def next_five_composites : List Nat := [21, 22, 24, 25, 26]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem composite_product_division :
  (product_of_list first_five_composites) / (product_of_list next_five_composites) = 72 / 715 := by
  sorry

end composite_product_division_l711_71112


namespace garden_breadth_is_100_l711_71102

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_is_100 :
  ∃ (garden : RectangularGarden),
    garden.length = 250 ∧
    perimeter garden = 700 ∧
    garden.breadth = 100 := by
  sorry

end garden_breadth_is_100_l711_71102


namespace steve_snack_shack_cost_l711_71182

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem steve_snack_shack_cost : total_cost = 39 := by
  sorry

end steve_snack_shack_cost_l711_71182


namespace thirteen_gumballs_needed_l711_71166

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Calculates the least number of gumballs needed to ensure four of the same color -/
def leastGumballs (machine : GumballMachine) : ℕ :=
  sorry

/-- Theorem stating that for the given gumball machine, 13 is the least number of gumballs needed -/
theorem thirteen_gumballs_needed (machine : GumballMachine) 
  (h1 : machine.red = 10)
  (h2 : machine.white = 9)
  (h3 : machine.blue = 8)
  (h4 : machine.green = 6) :
  leastGumballs machine = 13 :=
by
  sorry

end thirteen_gumballs_needed_l711_71166


namespace solution_set_equivalence_l711_71104

theorem solution_set_equivalence (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end solution_set_equivalence_l711_71104


namespace max_value_of_exponential_difference_l711_71143

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end max_value_of_exponential_difference_l711_71143


namespace total_points_noa_and_phillip_l711_71150

/-- 
Given that Noa scored 30 points and Phillip scored twice as many points as Noa,
prove that the total number of points scored by Noa and Phillip is 90.
-/
theorem total_points_noa_and_phillip : 
  let noa_points : ℕ := 30
  let phillip_points : ℕ := 2 * noa_points
  noa_points + phillip_points = 90 := by
sorry


end total_points_noa_and_phillip_l711_71150


namespace pens_left_in_jar_l711_71141

/-- The number of pens left in a jar after removing some pens -/
theorem pens_left_in_jar
  (initial_blue : ℕ)
  (initial_black : ℕ)
  (initial_red : ℕ)
  (blue_removed : ℕ)
  (black_removed : ℕ)
  (h1 : initial_blue = 9)
  (h2 : initial_black = 21)
  (h3 : initial_red = 6)
  (h4 : blue_removed = 4)
  (h5 : black_removed = 7)
  : initial_blue + initial_black + initial_red - blue_removed - black_removed = 25 := by
  sorry


end pens_left_in_jar_l711_71141


namespace car_trip_duration_l711_71114

/-- Proves that a car trip with given conditions has a total duration of 15 hours -/
theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) : 
  initial_speed = 30 →
  initial_time = 5 →
  additional_speed = 42 →
  average_speed = 38 →
  (initial_speed * initial_time + additional_speed * (15 - initial_time)) / 15 = average_speed :=
by
  sorry

#check car_trip_duration

end car_trip_duration_l711_71114


namespace fish_value_in_rice_l711_71198

-- Define the trade ratios
def fish_to_bread : ℚ := 4 / 5
def bread_to_rice : ℚ := 6
def fish_to_rice : ℚ := 8 / 3

-- Theorem to prove
theorem fish_value_in_rice : fish_to_rice = 8 / 3 := by
  sorry

#eval fish_to_rice

end fish_value_in_rice_l711_71198


namespace consecutive_coin_tosses_l711_71152

theorem consecutive_coin_tosses (p : ℝ) (h : p = 1 / 2) :
  p ^ 5 = 1 / 32 := by
sorry

end consecutive_coin_tosses_l711_71152


namespace crayons_lost_or_given_away_l711_71128

/-- Given that Paul gave away 213 crayons and lost 16 crayons,
    prove that the total number of crayons lost or given away is 229. -/
theorem crayons_lost_or_given_away :
  let crayons_given_away : ℕ := 213
  let crayons_lost : ℕ := 16
  crayons_given_away + crayons_lost = 229 := by
  sorry

end crayons_lost_or_given_away_l711_71128


namespace sum_f_positive_l711_71121

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end sum_f_positive_l711_71121


namespace exists_valid_coloring_l711_71180

/-- A domino is a 1x2 rectangle on the board -/
structure Domino where
  x : Fin 3000
  y : Fin 3000
  horizontal : Bool

/-- A color is represented by a number from 0 to 2 -/
def Color := Fin 3

/-- A coloring assigns a color to each domino -/
def Coloring := Domino → Color

/-- Two dominoes are neighbors if they share an edge -/
def are_neighbors (d1 d2 : Domino) : Prop :=
  sorry

/-- The number of dominoes with a given color in a coloring -/
def count_color (c : Coloring) (color : Color) : Nat :=
  sorry

/-- The number of neighbors of a domino with the same color -/
def same_color_neighbors (c : Coloring) (d : Domino) : Nat :=
  sorry

/-- The main theorem: there exists a valid coloring -/
theorem exists_valid_coloring :
  ∃ (c : Coloring),
    (∀ color : Color, count_color c color = 1500000) ∧
    (∀ d : Domino, same_color_neighbors c d ≤ 2) :=
  sorry

end exists_valid_coloring_l711_71180


namespace five_segments_max_regions_l711_71188

/-- The maximum number of regions formed by n line segments in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of regions formed by 5 line segments in a plane is 16 -/
theorem five_segments_max_regions : max_regions 5 = 16 := by
  sorry

end five_segments_max_regions_l711_71188


namespace negation_existential_square_positive_l711_71147

theorem negation_existential_square_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) := by sorry

end negation_existential_square_positive_l711_71147


namespace B_is_largest_l711_71131

/-- A is defined as the sum of 2023/2022 and 2023/2024 -/
def A : ℚ := 2023/2022 + 2023/2024

/-- B is defined as the sum of 2024/2023 and 2026/2023 -/
def B : ℚ := 2024/2023 + 2026/2023

/-- C is defined as the sum of 2025/2024 and 2025/2026 -/
def C : ℚ := 2025/2024 + 2025/2026

/-- Theorem stating that B is the largest among A, B, and C -/
theorem B_is_largest : B > A ∧ B > C := by
  sorry

end B_is_largest_l711_71131


namespace peanut_mixture_solution_l711_71171

/-- Represents the peanut mixture problem -/
def peanut_mixture (virginia_weight : ℝ) (virginia_cost : ℝ) (spanish_cost : ℝ) (mixture_cost : ℝ) : ℝ → Prop :=
  λ spanish_weight : ℝ =>
    (virginia_weight * virginia_cost + spanish_weight * spanish_cost) / (virginia_weight + spanish_weight) = mixture_cost

/-- Proves that 2.5 pounds of Spanish peanuts is the correct amount for the desired mixture -/
theorem peanut_mixture_solution :
  peanut_mixture 10 3.5 3 3.4 2.5 := by
  sorry

end peanut_mixture_solution_l711_71171


namespace complex_product_simplification_l711_71132

theorem complex_product_simplification (x y : ℝ) :
  let i := Complex.I
  (x + i * y + 1) * (x - i * y + 1) = (x + 1)^2 - y^2 := by
  sorry

end complex_product_simplification_l711_71132


namespace rectangular_prism_diagonal_l711_71159

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 10) (hw : w = 20) (hh : h = 10) :
  Real.sqrt (l^2 + w^2 + h^2) = 10 * Real.sqrt 6 := by
  sorry

end rectangular_prism_diagonal_l711_71159


namespace inverse_proportion_problem_l711_71164

/-- Given two inversely proportional quantities p and q, if p = 30 when q = 4,
    then p = 12 when q = 10. -/
theorem inverse_proportion_problem (p q : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, p = x ∧ q = y → x * y = k) :
  (p = 30 ∧ q = 4) → (q = 10 → p = 12) := by
  sorry

end inverse_proportion_problem_l711_71164


namespace impossible_exact_usage_l711_71186

theorem impossible_exact_usage (p q r : ℕ) : 
  ¬∃ (x y z : ℤ), (2*x + 2*z = 2*p + 2*r + 2) ∧ 
                   (2*x + y = 2*p + q + 1) ∧ 
                   (y + z = q + r) :=
sorry

end impossible_exact_usage_l711_71186


namespace exists_square_function_l711_71133

theorem exists_square_function : ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 := by
  sorry

end exists_square_function_l711_71133
