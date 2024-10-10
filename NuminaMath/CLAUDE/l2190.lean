import Mathlib

namespace trent_total_distance_l2190_219071

/-- Represents the distance Trent traveled throughout his day -/
def trent_travel (block_length : ℕ) (walk_blocks : ℕ) (bus_blocks : ℕ) (bike_blocks : ℕ) : ℕ :=
  2 * (walk_blocks + bus_blocks + bike_blocks) * block_length

/-- Theorem stating the total distance Trent traveled -/
theorem trent_total_distance :
  trent_travel 50 4 7 5 = 1600 := by sorry

end trent_total_distance_l2190_219071


namespace equation_solution_l2190_219051

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ x ≠ -6 ∧
  (3*x - 6) / (x^2 + 5*x - 6) = (x + 3) / (x - 1) ∧
  x = 9/2 := by
sorry

end equation_solution_l2190_219051


namespace cos_value_from_tan_sin_relation_l2190_219034

theorem cos_value_from_tan_sin_relation (θ : Real) 
  (h1 : 6 * Real.tan θ = 5 * Real.sin θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.cos θ = 5/6 := by
  sorry

end cos_value_from_tan_sin_relation_l2190_219034


namespace prob_ace_then_diamond_standard_deck_l2190_219065

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (diamonds : Nat)
  (ace_of_diamonds : Nat)

/-- Probability of drawing an Ace first and a diamond second from a standard deck -/
def prob_ace_then_diamond (d : Deck) : ℚ :=
  let prob_ace_of_diamonds := d.ace_of_diamonds / d.cards
  let prob_other_ace := (d.aces - d.ace_of_diamonds) / d.cards
  let prob_diamond_after_ace_of_diamonds := (d.diamonds - 1) / (d.cards - 1)
  let prob_diamond_after_other_ace := d.diamonds / (d.cards - 1)
  prob_ace_of_diamonds * prob_diamond_after_ace_of_diamonds +
  prob_other_ace * prob_diamond_after_other_ace

theorem prob_ace_then_diamond_standard_deck :
  prob_ace_then_diamond { cards := 52, aces := 4, diamonds := 13, ace_of_diamonds := 1 } = 119 / 3571 :=
sorry

end prob_ace_then_diamond_standard_deck_l2190_219065


namespace rectangle_side_length_l2190_219081

theorem rectangle_side_length (a b d : ℝ) : 
  a = 4 →
  a / b = 2 * (b / d) →
  d^2 = a^2 + b^2 →
  b = Real.sqrt (2 + 4 * Real.sqrt 17) :=
by sorry

end rectangle_side_length_l2190_219081


namespace area_transformation_l2190_219020

-- Define a function representing the area under a curve
noncomputable def area_under_curve (f : ℝ → ℝ) : ℝ := sorry

-- Define the original function g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem area_transformation (h : area_under_curve g = 15) :
  area_under_curve (fun x ↦ 4 * g (2 * x - 4)) = 30 := by sorry

end area_transformation_l2190_219020


namespace power_equation_solution_l2190_219064

theorem power_equation_solution :
  (∃ x : ℤ, (10 : ℝ)^655 * (10 : ℝ)^x = 1000) ∧
  (∀ x : ℤ, (10 : ℝ)^655 * (10 : ℝ)^x = 1000 → x = -652) :=
by sorry

end power_equation_solution_l2190_219064


namespace optimal_purchase_is_cheapest_l2190_219058

/-- Park admission fee per person -/
def individual_fee : ℕ := 5

/-- Group ticket fee -/
def group_fee : ℕ := 40

/-- Maximum number of people allowed per group ticket -/
def group_max : ℕ := 10

/-- Cost function for purchasing tickets -/
def ticket_cost (group_tickets : ℕ) (individual_tickets : ℕ) : ℕ :=
  group_tickets * group_fee + individual_tickets * individual_fee

/-- The most economical way to purchase tickets -/
def optimal_purchase (x : ℕ) : ℕ × ℕ :=
  let a := x / group_max
  let b := x % group_max
  if b < 8 then (a, b)
  else if b = 8 then (a, 8)  -- or (a + 1, 0), both are optimal
  else (a + 1, 0)

theorem optimal_purchase_is_cheapest (x : ℕ) :
  let (g, i) := optimal_purchase x
  ∀ (g' i' : ℕ), g' * group_max + i' ≥ x →
    ticket_cost g i ≤ ticket_cost g' i' :=
sorry

end optimal_purchase_is_cheapest_l2190_219058


namespace tree_rings_l2190_219056

theorem tree_rings (thin_rings : ℕ) : 
  (∀ (fat_rings : ℕ), fat_rings = 2) →
  (70 * (fat_rings + thin_rings) = 40 * (fat_rings + thin_rings) + 180) →
  thin_rings = 4 := by
sorry

end tree_rings_l2190_219056


namespace not_sum_to_seven_l2190_219044

def pairs : List (Int × Int) := [(4, 3), (-1, 8), (10, -2), (2, 5), (3, 5)]

def sum_to_seven (pair : Int × Int) : Bool :=
  pair.1 + pair.2 = 7

theorem not_sum_to_seven : 
  ∀ (pair : Int × Int), 
    pair ∈ pairs → 
      (¬(sum_to_seven pair) ↔ (pair = (10, -2) ∨ pair = (3, 5))) := by
  sorry

#eval pairs.filter (λ pair => ¬(sum_to_seven pair))

end not_sum_to_seven_l2190_219044


namespace decreasing_condition_l2190_219039

/-- The quadratic function f(x) = 2(x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * (x - 1)

theorem decreasing_condition (x : ℝ) : 
  x < 1 → f' x < 0 :=
sorry

end decreasing_condition_l2190_219039


namespace number_puzzle_l2190_219094

theorem number_puzzle (x : ℝ) : (x / 8) - 160 = 12 → x = 1376 := by
  sorry

end number_puzzle_l2190_219094


namespace first_sample_in_systematic_sampling_l2190_219091

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => firstSample + (n - 1) * (total / sampleSize)

theorem first_sample_in_systematic_sampling
  (total : ℕ) (sampleSize : ℕ) (fourthSample : ℕ) 
  (h1 : total = 800)
  (h2 : sampleSize = 80)
  (h3 : fourthSample = 39) :
  ∃ firstSample : ℕ, 
    firstSample ∈ Finset.range 10 ∧ 
    systematicSample total sampleSize firstSample 4 = fourthSample ∧
    firstSample = 9 :=
by sorry

end first_sample_in_systematic_sampling_l2190_219091


namespace valid_C_characterization_l2190_219041

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- A sequence is bounded below -/
def BoundedBelow (a : IntegerSequence) : Prop :=
  ∃ M : ℤ, ∀ n : ℕ, M ≤ a n

/-- A sequence satisfies the given inequality for a given C -/
def SatisfiesInequality (a : IntegerSequence) (C : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → (0 : ℝ) ≤ a (n - 1) + C * a n + a (n + 1) ∧ 
                   a (n - 1) + C * a n + a (n + 1) < 1

/-- A sequence is periodic -/
def Periodic (a : IntegerSequence) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, a (n + p) = a n

/-- The set of all C that satisfy the conditions -/
def ValidC : Set ℝ :=
  {C : ℝ | ∀ a : IntegerSequence, BoundedBelow a → SatisfiesInequality a C → Periodic a}

theorem valid_C_characterization : ValidC = Set.Ici (-2 : ℝ) :=
sorry

end valid_C_characterization_l2190_219041


namespace product_of_squares_l2190_219004

theorem product_of_squares (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 15/8) : r * s = Real.sqrt 17 / 4 := by
  sorry

end product_of_squares_l2190_219004


namespace objective_function_range_l2190_219086

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y > 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x + y

-- Theorem statement
theorem objective_function_range :
  ∃ (min max : ℝ), min = 1 ∧ max = 6 ∧
  (∀ x y : ℝ, FeasibleRegion x y →
    min ≤ ObjectiveFunction x y ∧ ObjectiveFunction x y ≤ max) ∧
  (∃ x1 y1 x2 y2 : ℝ, 
    FeasibleRegion x1 y1 ∧ FeasibleRegion x2 y2 ∧
    ObjectiveFunction x1 y1 = min ∧ ObjectiveFunction x2 y2 = max) :=
by
  sorry

end objective_function_range_l2190_219086


namespace third_angle_is_70_l2190_219048

-- Define a triangle type
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define the sum of angles in a triangle
def sum_of_angles (t : Triangle) : Real :=
  t.angle1 + t.angle2 + t.angle3

-- Theorem statement
theorem third_angle_is_70 (t : Triangle) 
  (h1 : t.angle1 = 50)
  (h2 : t.angle2 = 60)
  (h3 : sum_of_angles t = 180) : 
  t.angle3 = 70 := by
sorry


end third_angle_is_70_l2190_219048


namespace parallel_vector_implies_zero_y_coordinate_l2190_219032

/-- Given vectors a and b in R², if b - a is parallel to a, then the y-coordinate of b is 0 -/
theorem parallel_vector_implies_zero_y_coordinate (m n : ℝ) :
  let a : Fin 2 → ℝ := ![1, 0]
  let b : Fin 2 → ℝ := ![m, n]
  (∃ (k : ℝ), (b - a) = k • a) → n = 0 := by
  sorry

end parallel_vector_implies_zero_y_coordinate_l2190_219032


namespace solution_range_l2190_219085

def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem solution_range (f : ℝ → ℝ) (h_monotone : monotone_increasing f) (h_zero : f 1 = 0) :
  {x : ℝ | f (x^2 + 3*x - 3) < 0} = Set.Ioo (-4) 1 := by sorry

end solution_range_l2190_219085


namespace selection_problem_l2190_219047

theorem selection_problem (n r : ℕ) (h : r < n) :
  -- Number of ways to select r people from 2n people in a row with no adjacent selections
  (Nat.choose (2*n - r + 1) r) = 
    (Nat.choose (2*n - r + 1) r) ∧
  -- Number of ways to select r people from 2n people in a circle with no adjacent selections
  ((2*n : ℚ) / (2*n - r : ℚ)) * (Nat.choose (2*n - r) r) = 
    ((2*n : ℚ) / (2*n - r : ℚ)) * (Nat.choose (2*n - r) r) := by
  sorry

end selection_problem_l2190_219047


namespace train_length_problem_l2190_219062

theorem train_length_problem (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 144) :
  let rel_speed := (v_fast - v_slow) * (5 / 18)
  let train_length := rel_speed * t / 2
  train_length = 200 := by
sorry

end train_length_problem_l2190_219062


namespace smallest_w_l2190_219018

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^4) (1452 * w) → 
  is_factor (3^3) (1452 * w) → 
  is_factor (13^3) (1452 * w) → 
  w ≥ 79132 :=
sorry

end smallest_w_l2190_219018


namespace kiran_work_completion_l2190_219021

/-- Given that Kiran completes 1/3 of the work in 6 days, prove that he will finish the remaining work in 12 days. -/
theorem kiran_work_completion (work_rate : ℝ) (h1 : work_rate * 6 = 1/3) : 
  work_rate * 12 = 2/3 := by sorry

end kiran_work_completion_l2190_219021


namespace sock_selection_with_red_l2190_219076

def total_socks : ℕ := 7
def socks_to_select : ℕ := 3

theorem sock_selection_with_red (total_socks : ℕ) (socks_to_select : ℕ) : 
  total_socks = 7 → socks_to_select = 3 → 
  (Nat.choose total_socks socks_to_select) - (Nat.choose (total_socks - 1) socks_to_select) = 15 := by
  sorry

end sock_selection_with_red_l2190_219076


namespace shaded_area_approx_l2190_219063

-- Define the circle and rectangle
def circle_radius : ℝ := 3
def rectangle_side_OA : ℝ := 2
def rectangle_side_AB : ℝ := 1

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (rectangle_side_OA, 0)
def B : ℝ × ℝ := (rectangle_side_OA, rectangle_side_AB)
def C : ℝ × ℝ := (0, rectangle_side_AB)

-- Define the function to calculate the area of the shaded region
def shaded_area : ℝ := sorry

-- Theorem statement
theorem shaded_area_approx :
  abs (shaded_area - 6.23) < 0.01 := by sorry

end shaded_area_approx_l2190_219063


namespace overlap_percentage_l2190_219055

theorem overlap_percentage (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 10 →
  rect_length = 18 →
  rect_width = 10 →
  (2 * square_side - rect_length) * rect_width / (rect_length * rect_width) * 100 = 11.11 := by
sorry

end overlap_percentage_l2190_219055


namespace hexagon_largest_angle_l2190_219099

/-- A convex hexagon with interior angles as consecutive integers has its largest angle equal to 122° -/
theorem hexagon_largest_angle : ∀ (a b c d e f : ℕ),
  -- The angles are natural numbers
  -- The angles are consecutive integers
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 →
  -- The sum of interior angles of a hexagon is 720°
  a + b + c + d + e + f = 720 →
  -- The largest angle is 122°
  f = 122 := by
sorry

end hexagon_largest_angle_l2190_219099


namespace j_mod_2_not_zero_l2190_219019

theorem j_mod_2_not_zero (x j : ℤ) (h : 2 * x - j = 11) : j % 2 ≠ 0 := by
  sorry

end j_mod_2_not_zero_l2190_219019


namespace cafeteria_earnings_l2190_219096

/-- Calculates the total earnings from selling apples and oranges in a cafeteria. -/
theorem cafeteria_earnings (initial_apples initial_oranges : ℕ)
                           (apple_price orange_price : ℚ)
                           (remaining_apples remaining_oranges : ℕ)
                           (h1 : initial_apples = 50)
                           (h2 : initial_oranges = 40)
                           (h3 : apple_price = 0.80)
                           (h4 : orange_price = 0.50)
                           (h5 : remaining_apples = 10)
                           (h6 : remaining_oranges = 6) :
  (initial_apples - remaining_apples) * apple_price +
  (initial_oranges - remaining_oranges) * orange_price = 49 :=
by sorry

end cafeteria_earnings_l2190_219096


namespace perpendicular_lines_m_value_l2190_219042

/-- 
Given two lines in the xy-plane defined by their equations,
this theorem states that if these lines are perpendicular,
then the parameter m must equal 1/2.
-/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (∀ x y : ℝ, x - m * y + 2 * m = 0 → x + 2 * y - m = 0 → 
    (1 : ℝ) / m * (-1 / 2 : ℝ) = -1) → 
  m = 1 / 2 := by
  sorry

end perpendicular_lines_m_value_l2190_219042


namespace num_non_congruent_triangles_l2190_219024

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The set of points in the 3x3 grid -/
def gridPoints : List Point := [
  ⟨0, 0⟩, ⟨0.5, 0⟩, ⟨1, 0⟩,
  ⟨0, 0.5⟩, ⟨0.5, 0.5⟩, ⟨1, 0.5⟩,
  ⟨0, 1⟩, ⟨0.5, 1⟩, ⟨1, 1⟩
]

/-- Predicate to check if two triangles are congruent -/
def areCongruent (t1 t2 : Triangle) : Prop := sorry

/-- The set of all possible triangles formed from the grid points -/
def allTriangles : List Triangle := sorry

/-- The set of non-congruent triangles -/
def nonCongruentTriangles : List Triangle := sorry

/-- Theorem: The number of non-congruent triangles is 3 -/
theorem num_non_congruent_triangles : 
  nonCongruentTriangles.length = 3 := by sorry

end num_non_congruent_triangles_l2190_219024


namespace evaluate_expression_l2190_219006

theorem evaluate_expression (y : ℚ) (h : y = -3) :
  (5 + y * (2 + y) - 4^2) / (y - 4 + y^2 - y) = -8 / 5 := by
  sorry

end evaluate_expression_l2190_219006


namespace max_ratio_three_digit_number_to_digit_sum_l2190_219043

theorem max_ratio_three_digit_number_to_digit_sum :
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c : ℚ) / (a + b + c) ≤ 100 ∧
    ∃ (a₀ b₀ c₀ : ℕ),
      1 ≤ a₀ ∧ a₀ ≤ 9 ∧
      0 ≤ b₀ ∧ b₀ ≤ 9 ∧
      0 ≤ c₀ ∧ c₀ ≤ 9 ∧
      (100 * a₀ + 10 * b₀ + c₀ : ℚ) / (a₀ + b₀ + c₀) = 100 :=
by sorry

end max_ratio_three_digit_number_to_digit_sum_l2190_219043


namespace S_is_two_rays_with_common_endpoint_l2190_219078

/-- The set S of points (x, y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≥ 5) ∨
               (5 = y - 2 ∧ x + 3 ≥ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≥ x + 3)}

/-- Two rays with a common endpoint -/
def TwoRaysWithCommonEndpoint : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x = 2 ∧ y ≥ 7) ∨
               (y = 7 ∧ x ≥ 2)}

/-- Theorem stating that S is equivalent to two rays with a common endpoint -/
theorem S_is_two_rays_with_common_endpoint : S = TwoRaysWithCommonEndpoint := by
  sorry

end S_is_two_rays_with_common_endpoint_l2190_219078


namespace expression_factorization_l2190_219093

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x^2 - 3) - (-3 * x^3 + 6 * x^2 - 3) = 3 * x^2 * (5 * x + 13) := by
  sorry

end expression_factorization_l2190_219093


namespace square_ratio_sum_l2190_219083

theorem square_ratio_sum (p q r : ℕ) : 
  (75 : ℚ) / 128 = (p * Real.sqrt q / r) ^ 2 → p + q + r = 27 := by
  sorry

end square_ratio_sum_l2190_219083


namespace triple_equation_solution_l2190_219031

theorem triple_equation_solution :
  ∀ (a b c : ℝ), 
    ((2*a+1)^2 - 4*b = 5 ∧ 
     (2*b+1)^2 - 4*c = 5 ∧ 
     (2*c+1)^2 - 4*a = 5) ↔ 
    ((a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = -1 ∧ b = -1 ∧ c = -1)) :=
by sorry

end triple_equation_solution_l2190_219031


namespace festival_lineup_theorem_l2190_219046

/-- The minimum number of Gennadys required for the festival lineup -/
def min_gennadys (num_alexanders num_borises num_vasilys : ℕ) : ℕ :=
  max 0 (num_borises - 1 - (num_alexanders + num_vasilys))

/-- Theorem stating the minimum number of Gennadys required for the festival lineup -/
theorem festival_lineup_theorem (num_alexanders num_borises num_vasilys : ℕ) 
  (h1 : num_alexanders = 45)
  (h2 : num_borises = 122)
  (h3 : num_vasilys = 27) :
  min_gennadys num_alexanders num_borises num_vasilys = 49 := by
  sorry

#eval min_gennadys 45 122 27

end festival_lineup_theorem_l2190_219046


namespace binomial_coefficient_congruence_l2190_219029

theorem binomial_coefficient_congruence (p n : ℕ) (hp : Prime p) :
  (Nat.choose n p) ≡ (n / p : ℕ) [MOD p] := by sorry

end binomial_coefficient_congruence_l2190_219029


namespace power_of_three_plus_five_mod_eight_l2190_219035

theorem power_of_three_plus_five_mod_eight :
  (3^100 + 5) % 8 = 6 := by
  sorry

end power_of_three_plus_five_mod_eight_l2190_219035


namespace product_remainder_mod_ten_l2190_219025

theorem product_remainder_mod_ten (a b c : ℕ) : 
  a % 10 = 7 → b % 10 = 1 → c % 10 = 3 → (a * b * c) % 10 = 1 := by
  sorry

end product_remainder_mod_ten_l2190_219025


namespace smallest_angle_measure_l2190_219016

-- Define the triangle
structure ObtuseIsoscelesTriangle where
  -- The largest angle in degrees
  largest_angle : ℝ
  -- One of the two equal angles in degrees
  equal_angle : ℝ
  -- Conditions
  is_obtuse : largest_angle > 90
  is_isosceles : equal_angle = equal_angle
  angle_sum : largest_angle + 2 * equal_angle = 180

-- Theorem statement
theorem smallest_angle_measure (t : ObtuseIsoscelesTriangle) 
  (h : t.largest_angle = 108) : t.equal_angle = 36 := by
  sorry

end smallest_angle_measure_l2190_219016


namespace hyperbola_equation_l2190_219015

/-- Given a hyperbola C and an ellipse with the following properties:
    - The general equation of C is (x²/a²) - (y²/b²) = 1 where a > 0 and b > 0
    - C has an asymptote equation y = (√5/2)x
    - C shares a common focus with the ellipse x²/12 + y²/3 = 1
    Then, the specific equation of hyperbola C is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2) ∧ 
  (b / a = Real.sqrt 5 / 2) ∧
  (c^2 = 3^2) →
  a^2 = 4 ∧ b^2 = 5 := by
  sorry

end hyperbola_equation_l2190_219015


namespace power_division_rule_l2190_219073

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^5 / a^2 = a^3 := by
  sorry

end power_division_rule_l2190_219073


namespace set_intersection_theorem_l2190_219050

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 1}

theorem set_intersection_theorem : M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end set_intersection_theorem_l2190_219050


namespace quadratic_equation_coefficients_l2190_219072

/-- Given a quadratic equation x^2 - 4x = 5, prove that its standard form coefficients are 1, -4, and -5 -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 - 4*x = 5 ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -4 ∧ c = -5 := by
  sorry

end quadratic_equation_coefficients_l2190_219072


namespace sqrt_square_789256_l2190_219052

theorem sqrt_square_789256 : (Real.sqrt 789256)^2 = 789256 := by
  sorry

end sqrt_square_789256_l2190_219052


namespace room_dimension_is_15_l2190_219030

/-- Represents the dimensions and properties of a room to be whitewashed -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  doorArea : ℝ
  windowArea : ℝ
  windowCount : ℕ
  whitewashCost : ℝ
  totalCost : ℝ

/-- Calculates the total area to be whitewashed in the room -/
def areaToWhitewash (r : Room) : ℝ :=
  2 * (r.length * r.height + r.width * r.height) - (r.doorArea + r.windowCount * r.windowArea)

/-- Theorem stating that the unknown dimension of the room is 15 feet -/
theorem room_dimension_is_15 (r : Room) 
  (h1 : r.length = 25)
  (h2 : r.height = 12)
  (h3 : r.doorArea = 18)
  (h4 : r.windowArea = 12)
  (h5 : r.windowCount = 3)
  (h6 : r.whitewashCost = 5)
  (h7 : r.totalCost = 4530)
  (h8 : r.totalCost = r.whitewashCost * areaToWhitewash r) :
  r.width = 15 := by
  sorry

end room_dimension_is_15_l2190_219030


namespace matrix_equation_proof_l2190_219038

def N : Matrix (Fin 2) (Fin 2) ℝ := !![1, -10; 0, 1]

theorem matrix_equation_proof :
  N^3 - 3 * N^2 + 2 * N = !![5, 10; 0, 5] := by sorry

end matrix_equation_proof_l2190_219038


namespace diane_gambling_problem_l2190_219026

theorem diane_gambling_problem (initial_amount : ℝ) : 
  (initial_amount + 65 + 50 = 215) → initial_amount = 100 := by
sorry

end diane_gambling_problem_l2190_219026


namespace unique_consecutive_sum_20_l2190_219059

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n : ℕ | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers with sum 20 -/
theorem unique_consecutive_sum_20 : 
  ∃! p : ℕ × ℕ, 2 ≤ p.2 ∧ ConsecutiveSum p.1 p.2 = 20 :=
sorry

end unique_consecutive_sum_20_l2190_219059


namespace arithmetic_expression_evaluation_l2190_219066

theorem arithmetic_expression_evaluation :
  65 + (126 / 14) + (35 * 11) - 250 - (500 / 5)^2 = -9791 := by
  sorry

end arithmetic_expression_evaluation_l2190_219066


namespace part_one_part_two_l2190_219060

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6

-- Define the function g
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Part 1
theorem part_one (a : ℝ) : (∀ y ≥ 0, ∃ x, f a x = y) ∧ (∀ x, f a x ≥ 0) → a = 3/2 := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (∀ x, f a x ≥ 0) → 
  (∀ y ∈ Set.Icc (-19/4) (-2), ∃ a ∈ Set.Icc (-1) (3/2), g a = y) ∧ 
  (∀ a ∈ Set.Icc (-1) (3/2), g a ∈ Set.Icc (-19/4) (-2)) := by sorry

end part_one_part_two_l2190_219060


namespace polynomial_simplification_l2190_219088

theorem polynomial_simplification (x : ℝ) : 
  (x^5 + x^4 + x + 10) - (x^5 + 2*x^4 - x^3 + 12) = -x^4 + x^3 + x - 2 := by
  sorry

end polynomial_simplification_l2190_219088


namespace rationalize_sqrt3_minus1_l2190_219003

theorem rationalize_sqrt3_minus1 : 1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end rationalize_sqrt3_minus1_l2190_219003


namespace inequality_proof_l2190_219070

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_min : min (a * b) (min (b * c) (c * a)) ≥ 1) :
  (((a^2 + 1) * (b^2 + 1) * (c^2 + 1))^(1/3) : ℝ) ≤ ((a + b + c) / 3)^2 + 1 := by
sorry

end inequality_proof_l2190_219070


namespace complement_of_57_13_l2190_219095

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60,
    valid := by sorry }

/-- The main theorem stating that the complement of 57°13' is 32°47' -/
theorem complement_of_57_13 :
  complement { degrees := 57, minutes := 13, valid := by sorry } =
  { degrees := 32, minutes := 47, valid := by sorry } := by
  sorry

end complement_of_57_13_l2190_219095


namespace min_sum_distances_l2190_219022

theorem min_sum_distances (a b : ℝ) :
  Real.sqrt ((a - 1)^2 + (b - 1)^2) + Real.sqrt ((a + 1)^2 + (b + 1)^2) ≥ 2 * Real.sqrt 2 := by
  sorry

end min_sum_distances_l2190_219022


namespace correct_arrangements_l2190_219090

/-- Represents a student with a grade -/
structure Student where
  grade : Nat

/-- Represents a car with students -/
structure Car where
  students : Finset Student

/-- The total number of students -/
def totalStudents : Nat := 8

/-- The number of grades -/
def numGrades : Nat := 4

/-- The number of students per grade -/
def studentsPerGrade : Nat := 2

/-- The number of students per car -/
def studentsPerCar : Nat := 4

/-- Twin sisters from first grade -/
def twinSisters : Finset Student := sorry

/-- All students -/
def allStudents : Finset Student := sorry

/-- Checks if a car has exactly two students from the same grade -/
def hasTwoSameGrade (car : Car) : Prop := sorry

/-- The number of ways to arrange students in car A -/
def numArrangements : Nat := sorry

/-- Main theorem -/
theorem correct_arrangements :
  numArrangements = 24 := by sorry

end correct_arrangements_l2190_219090


namespace triangle_angle_impossibility_l2190_219049

theorem triangle_angle_impossibility : ¬ ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- all angles are positive
  a + b + c = 180 ∧        -- sum of angles is 180 degrees
  a = 60 ∧                 -- one angle is 60 degrees
  b = 2 * a ∧              -- another angle is twice the first
  c ≠ 0                    -- the third angle is non-zero
  := by sorry

end triangle_angle_impossibility_l2190_219049


namespace expression_simplification_l2190_219098

theorem expression_simplification (a b : ℚ) (ha : a = -2) (hb : b = 3) :
  (((a - b) / (a^2 - 2*a*b + b^2) - a / (a^2 - 2*a*b)) / (b / (a - 2*b))) = 1/5 := by
  sorry

end expression_simplification_l2190_219098


namespace unique_nonzero_solution_sum_of_squares_l2190_219097

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * y - 2 * y - 3 * x = 0
def equation2 (y z : ℝ) : Prop := y * z - 3 * z - 5 * y = 0
def equation3 (x z : ℝ) : Prop := x * z - 5 * x - 2 * z = 0

-- Define the theorem
theorem unique_nonzero_solution_sum_of_squares :
  ∃! (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
    equation1 a b ∧ equation2 b c ∧ equation3 a c →
    a^2 + b^2 + c^2 = 152 :=
by sorry

end unique_nonzero_solution_sum_of_squares_l2190_219097


namespace system_solution_l2190_219012

theorem system_solution : ∃ (x y z : ℝ), 
  (x + 2*y + 3*z = 3) ∧ 
  (3*x + y + 2*z = 7) ∧ 
  (2*x + 3*y + z = 2) ∧
  (x = 2) ∧ (y = -1) ∧ (z = 1) := by
  sorry

end system_solution_l2190_219012


namespace amount_after_two_years_l2190_219017

theorem amount_after_two_years
  (initial_amount : ℝ)
  (annual_rate : ℝ)
  (years : ℕ)
  (h1 : initial_amount = 51200)
  (h2 : annual_rate = 1 / 8)
  (h3 : years = 2) :
  initial_amount * (1 + annual_rate) ^ years = 64800 :=
by sorry

end amount_after_two_years_l2190_219017


namespace combined_probability_l2190_219007

/-- The probability that Xavier solves Problem A -/
def p_xa : ℚ := 1/5

/-- The probability that Yvonne solves Problem A -/
def p_ya : ℚ := 1/2

/-- The probability that Zelda solves Problem A -/
def p_za : ℚ := 5/8

/-- The probability that Xavier solves Problem B -/
def p_xb : ℚ := 2/9

/-- The probability that Yvonne solves Problem B -/
def p_yb : ℚ := 3/5

/-- The probability that Zelda solves Problem B -/
def p_zb : ℚ := 1/4

/-- The probability that Xavier solves Problem C -/
def p_xc : ℚ := 1/4

/-- The probability that Yvonne solves Problem C -/
def p_yc : ℚ := 3/8

/-- The probability that Zelda solves Problem C -/
def p_zc : ℚ := 9/16

/-- The theorem stating the probability of the combined event -/
theorem combined_probability : 
  p_xa * p_ya * p_yb * (1 - p_yc) * (1 - p_xc) * (1 - p_zc) = 63/2048 := by
  sorry

end combined_probability_l2190_219007


namespace female_officers_count_l2190_219005

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 204 →
  female_on_duty_percent = 17 / 100 →
  (total_on_duty / 2 : ℚ) = female_on_duty_percent * (600 : ℚ) :=
by sorry

end female_officers_count_l2190_219005


namespace tangent_line_y_intercept_l2190_219011

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Determines if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (7, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 2 * Real.sqrt 17 := by
  sorry

end tangent_line_y_intercept_l2190_219011


namespace inverse_proportion_order_l2190_219010

theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -4 / 1 → y₂ = -4 / 2 → y₃ = -4 / (-3) → y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end inverse_proportion_order_l2190_219010


namespace circle_center_radius_sum_l2190_219074

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 16*x + y^2 + 10*y = -75

-- Define the center and radius of the circle
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center_radius a b r ∧ a + b + r = 3 + Real.sqrt 14 :=
sorry

end circle_center_radius_sum_l2190_219074


namespace pizza_slice_volume_l2190_219023

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 10 →
  num_slices = 10 →
  (π * (diameter/2)^2 * thickness) / num_slices = 5*π/4 := by
  sorry

end pizza_slice_volume_l2190_219023


namespace double_xy_doubles_fraction_l2190_219084

/-- Given a fraction xy/(2x+y), prove that doubling both x and y results in doubling the fraction -/
theorem double_xy_doubles_fraction (x y : ℝ) (h : 2 * x + y ≠ 0) :
  (2 * x * 2 * y) / (2 * (2 * x) + 2 * y) = 2 * (x * y / (2 * x + y)) := by
  sorry

end double_xy_doubles_fraction_l2190_219084


namespace triangle_with_prime_angles_l2190_219000

theorem triangle_with_prime_angles (a b c : ℕ) : 
  a + b + c = 180 →
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) →
  a = 2 ∨ b = 2 ∨ c = 2 := by
  sorry

end triangle_with_prime_angles_l2190_219000


namespace quadratic_always_positive_range_l2190_219092

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end quadratic_always_positive_range_l2190_219092


namespace probability_of_black_ball_l2190_219080

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h_red : prob_red = 0.42)
  (h_white : prob_white = 0.28)
  (h_sum : prob_red + prob_white + (1 - prob_red - prob_white) = 1) : 
  1 - prob_red - prob_white = 0.3 := by
sorry

end probability_of_black_ball_l2190_219080


namespace turkey_cost_l2190_219079

/-- The cost of turkeys given their weights and price per kilogram -/
theorem turkey_cost (w1 w2 w3 w4 : ℝ) (price_per_kg : ℝ) : 
  w1 = 6 →
  w2 = 9 →
  w3 = 2 * w2 →
  w4 = (w1 + w2 + w3) / 2 →
  price_per_kg = 2 →
  (w1 + w2 + w3 + w4) * price_per_kg = 99 :=
by
  sorry

#check turkey_cost

end turkey_cost_l2190_219079


namespace factory_shutdown_probabilities_l2190_219061

/-- The number of factories -/
def num_factories : ℕ := 5

/-- The number of days in a week -/
def num_days : ℕ := 7

/-- The probability of all factories choosing Sunday to shut down -/
def prob_all_sunday : ℚ := 1 / 7^num_factories

/-- The probability of at least two factories choosing the same day to shut down -/
def prob_at_least_two_same : ℚ := 1 - (num_days.factorial / (num_days - num_factories).factorial) / 7^num_factories

theorem factory_shutdown_probabilities :
  (prob_all_sunday = 1 / 16807) ∧
  (prob_at_least_two_same = 2041 / 2401) := by
  sorry


end factory_shutdown_probabilities_l2190_219061


namespace average_speed_last_hour_l2190_219027

theorem average_speed_last_hour (total_distance : ℝ) (total_time : ℝ) 
  (first_30_speed : ℝ) (next_30_speed : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  first_30_speed = 50 →
  next_30_speed = 70 →
  let first_30_distance := first_30_speed * (30 / 60)
  let next_30_distance := next_30_speed * (30 / 60)
  let last_60_distance := total_distance - (first_30_distance + next_30_distance)
  let last_60_time := 60 / 60
  last_60_distance / last_60_time = 60 := by
  sorry

#check average_speed_last_hour

end average_speed_last_hour_l2190_219027


namespace work_time_ratio_l2190_219053

theorem work_time_ratio (a b : ℝ) (h1 : b = 18) (h2 : 1/a + 1/b = 1/3) :
  a / b = 1 / 5 := by sorry

end work_time_ratio_l2190_219053


namespace x_over_y_equals_four_l2190_219013

theorem x_over_y_equals_four (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * Real.log (x - 2*y) = Real.log x + Real.log y) : x / y = 4 := by
  sorry

end x_over_y_equals_four_l2190_219013


namespace weight_difference_l2190_219033

theorem weight_difference (jim steve stan : ℕ) 
  (h1 : steve < stan)
  (h2 : steve = jim - 8)
  (h3 : jim = 110)
  (h4 : stan + steve + jim = 319) :
  stan - steve = 5 := by
  sorry

end weight_difference_l2190_219033


namespace water_bottles_count_l2190_219002

theorem water_bottles_count (water_bottles : ℕ) (apple_bottles : ℕ) : 
  apple_bottles = water_bottles + 6 →
  water_bottles + apple_bottles = 54 →
  water_bottles = 24 := by
sorry

end water_bottles_count_l2190_219002


namespace vector_addition_and_scalar_multiplication_l2190_219014

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, -3]

theorem vector_addition_and_scalar_multiplication :
  (a + 2 • b) = ![4, -5] := by sorry

end vector_addition_and_scalar_multiplication_l2190_219014


namespace volume_of_specific_open_box_l2190_219045

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem stating that the volume of the specific open box is 5120 m³. -/
theorem volume_of_specific_open_box :
  openBoxVolume 48 36 8 = 5120 := by
  sorry

end volume_of_specific_open_box_l2190_219045


namespace hyperbola_asymptote_intersection_l2190_219036

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = 2*x + 3

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = (3/4) * x
def asymptote2 (x y : ℝ) : Prop := y = -(3/4) * x

-- Theorem statement
theorem hyperbola_asymptote_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    asymptote1 x1 y1 ∧ line x1 y1 ∧ 
    asymptote2 x2 y2 ∧ line x2 y2 ∧
    x1 = -12/5 ∧ y1 = -9/5 ∧
    x2 = -12/11 ∧ y2 = 9/11 :=
sorry

end hyperbola_asymptote_intersection_l2190_219036


namespace quadratic_equation_solution_l2190_219068

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end quadratic_equation_solution_l2190_219068


namespace range_of_m_l2190_219054

/-- Proposition p: The equation x²/(2m) - y²/(m-1) = 1 represents an ellipse with foci on the y-axis -/
def prop_p (m : ℝ) : Prop :=
  0 < m ∧ m < 1/3

/-- Proposition q: The eccentricity e of the hyperbola y²/5 - x²/m = 1 is in the interval (1,2) -/
def prop_q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem range_of_m (m : ℝ) :
  (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m) →
  1/3 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l2190_219054


namespace carlos_pesos_sum_of_digits_l2190_219057

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Represents the exchange rate from dollars to pesos -/
def exchangeRate : ℚ := 12 / 8

theorem carlos_pesos_sum_of_digits :
  ∀ d : ℕ,
  (exchangeRate * d - 72 : ℚ) = d →
  sumOfDigits d = 9 := by sorry

end carlos_pesos_sum_of_digits_l2190_219057


namespace sqrt_90000_equals_300_l2190_219028

theorem sqrt_90000_equals_300 : Real.sqrt 90000 = 300 := by
  sorry

end sqrt_90000_equals_300_l2190_219028


namespace soccer_league_games_l2190_219037

/-- The number of games played in a soccer league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 10 teams, where each team plays every other team once, 
    the total number of games played is 45 -/
theorem soccer_league_games : games_played 10 = 45 := by
  sorry

end soccer_league_games_l2190_219037


namespace parabola_point_distance_l2190_219067

/-- Given a parabola y² = x with focus at (1/4, 0), prove that a point on the parabola
    with distance 1 from the focus has x-coordinate 3/4 -/
theorem parabola_point_distance (x y : ℝ) : 
  y^2 = x →                                           -- Point (x, y) is on the parabola
  (x - 1/4)^2 + y^2 = 1 →                             -- Distance from (x, y) to focus (1/4, 0) is 1
  x = 3/4 := by sorry

end parabola_point_distance_l2190_219067


namespace smallest_satisfying_number_l2190_219069

theorem smallest_satisfying_number : ∃ (n : ℕ), n = 1806 ∧ 
  (∀ (m : ℕ), m < n → 
    ∃ (p : ℕ), Prime p ∧ (m % (p - 1) = 0 → m % p ≠ 0)) ∧
  (∀ (p : ℕ), Prime p → (n % (p - 1) = 0 → n % p = 0)) := by
  sorry

#check smallest_satisfying_number

end smallest_satisfying_number_l2190_219069


namespace multiplication_mistake_difference_l2190_219075

theorem multiplication_mistake_difference : 
  let correct_multiplication := 137 * 43
  let mistaken_multiplication := 137 * 34
  correct_multiplication - mistaken_multiplication = 1233 := by
sorry

end multiplication_mistake_difference_l2190_219075


namespace min_sum_of_distinct_integers_with_odd_square_sums_l2190_219082

theorem min_sum_of_distinct_integers_with_odd_square_sums (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (m n p : ℕ), 
    (a + b = 2 * m + 1) ∧ (a + c = 2 * n + 1) ∧ (a + d = 2 * p + 1) ∧
    (∃ (x y z : ℕ), (2 * m + 1 = x^2) ∧ (2 * n + 1 = y^2) ∧ (2 * p + 1 = z^2)) →
  10 * (a + b + c + d) ≥ 670 :=
by sorry

#check min_sum_of_distinct_integers_with_odd_square_sums

end min_sum_of_distinct_integers_with_odd_square_sums_l2190_219082


namespace perpendicular_line_plane_necessary_not_sufficient_l2190_219087

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields

structure Plane3D where
  -- Add necessary fields

-- Define perpendicularity relations
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def perpendicular_plane_plane (p1 p2 : Plane3D) : Prop :=
  sorry

def plane_contains_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_plane_necessary_not_sufficient 
  (l : Line3D) (α : Plane3D) :
  (perpendicular_line_plane l α → 
    ∃ (p : Plane3D), plane_contains_line p l ∧ perpendicular_plane_plane p α) ∧
  ¬(∀ (l : Line3D) (α : Plane3D), 
    (∃ (p : Plane3D), plane_contains_line p l ∧ perpendicular_plane_plane p α) → 
    perpendicular_line_plane l α) :=
  sorry

end perpendicular_line_plane_necessary_not_sufficient_l2190_219087


namespace min_value_problem_l2190_219008

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3*y = 5*x*y ∧ 3*x + 4*y = 5 := by
  sorry

end min_value_problem_l2190_219008


namespace gwens_birthday_money_l2190_219089

theorem gwens_birthday_money (mom_gift dad_gift : ℕ) 
  (h1 : mom_gift = 8)
  (h2 : dad_gift = 5) :
  mom_gift - dad_gift = 3 := by
  sorry

end gwens_birthday_money_l2190_219089


namespace smallest_subset_size_for_divisibility_l2190_219077

theorem smallest_subset_size_for_divisibility : ∃ (n : ℕ),
  n = 337 ∧
  (∀ (S : Finset ℕ), S ⊆ Finset.range 2005 → S.card = n →
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 2004 ∣ (a^2 - b^2)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T ⊆ Finset.range 2005 ∧ T.card = m ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬(2004 ∣ (a^2 - b^2))) :=
by sorry

end smallest_subset_size_for_divisibility_l2190_219077


namespace polynomial_sum_l2190_219040

-- Define the polynomials
def p (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) : p x + q x + r x = 12 * x - 12 := by
  sorry

end polynomial_sum_l2190_219040


namespace jay_scored_six_more_l2190_219001

/-- Represents the scores of players in a basketball game. -/
structure BasketballScores where
  tobee : ℕ
  jay : ℕ
  sean : ℕ

/-- Conditions of the basketball game scores. -/
def validScores (scores : BasketballScores) : Prop :=
  scores.tobee = 4 ∧
  scores.jay > scores.tobee ∧
  scores.sean = scores.tobee + scores.jay - 2 ∧
  scores.tobee + scores.jay + scores.sean = 26

/-- Theorem stating that Jay scored 6 more points than Tobee. -/
theorem jay_scored_six_more (scores : BasketballScores) 
  (h : validScores scores) : scores.jay = scores.tobee + 6 := by
  sorry

#check jay_scored_six_more

end jay_scored_six_more_l2190_219001


namespace rower_downstream_speed_l2190_219009

/-- Calculates the downstream speed of a rower given their upstream and still water speeds -/
def downstreamSpeed (upstreamSpeed stillWaterSpeed : ℝ) : ℝ :=
  2 * stillWaterSpeed - upstreamSpeed

theorem rower_downstream_speed
  (upstreamSpeed : ℝ)
  (stillWaterSpeed : ℝ)
  (h1 : upstreamSpeed = 25)
  (h2 : stillWaterSpeed = 33) :
  downstreamSpeed upstreamSpeed stillWaterSpeed = 41 := by
  sorry

#eval downstreamSpeed 25 33

end rower_downstream_speed_l2190_219009
