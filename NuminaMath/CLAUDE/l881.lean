import Mathlib

namespace geometric_sequence_sum_l881_88185

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 :=
by
  sorry

end geometric_sequence_sum_l881_88185


namespace correct_match_probability_l881_88142

/-- The number of celebrities and baby pictures -/
def n : ℕ := 4

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := n.factorial

/-- The number of correct arrangements -/
def correct_arrangements : ℕ := 1

/-- The probability of correctly matching all celebrities to their baby pictures -/
def probability : ℚ := correct_arrangements / total_arrangements

theorem correct_match_probability :
  probability = 1 / 24 := by sorry

end correct_match_probability_l881_88142


namespace parallel_vectors_m_value_l881_88172

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_m_value :
  parallel vector_a (vector_b m) → m = -4 := by
  sorry

end parallel_vectors_m_value_l881_88172


namespace min_value_squared_difference_l881_88151

theorem min_value_squared_difference (f : ℝ → ℝ) :
  (∀ x, f x = (x - 1)^2) →
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 0 :=
by sorry

end min_value_squared_difference_l881_88151


namespace parallelogram_area_l881_88158

-- Define the parallelogram and its properties
structure Parallelogram :=
  (area : ℝ)
  (inscribed_circles : ℕ)
  (circle_radius : ℝ)
  (touching_sides : ℕ)
  (vertex_to_tangency : ℝ)

-- Define the conditions of the problem
def problem_conditions (p : Parallelogram) : Prop :=
  p.inscribed_circles = 2 ∧
  p.circle_radius = 1 ∧
  p.touching_sides = 3 ∧
  p.vertex_to_tangency = Real.sqrt 3

-- Theorem statement
theorem parallelogram_area 
  (p : Parallelogram) 
  (h : problem_conditions p) : 
  p.area = 4 * (1 + Real.sqrt 3) := by
  sorry

end parallelogram_area_l881_88158


namespace right_triangle_circle_intersection_l881_88102

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point D
def D : ℝ × ℝ := sorry

-- Define the properties of the triangle and circle
def is_right_triangle (t : Triangle) : Prop :=
  sorry

def circle_intersects_BC (t : Triangle) (c : Circle) : Prop :=
  sorry

def AC_is_diameter (t : Triangle) (c : Circle) : Prop :=
  sorry

-- Theorem statement
theorem right_triangle_circle_intersection 
  (t : Triangle) (c : Circle) :
  is_right_triangle t →
  circle_intersects_BC t c →
  AC_is_diameter t c →
  t.A.1 - t.B.1 = 18 →
  t.A.1 - t.C.1 = 30 →
  D.1 - t.B.1 = 14.4 :=
sorry

end right_triangle_circle_intersection_l881_88102


namespace quadratic_sum_reciprocal_l881_88123

theorem quadratic_sum_reciprocal (t : ℝ) (h1 : t^2 - 3*t + 1 = 0) (h2 : t ≠ 0) :
  t + 1/t = 3 := by sorry

end quadratic_sum_reciprocal_l881_88123


namespace three_digit_square_insertion_l881_88128

theorem three_digit_square_insertion (n : ℕ) : ∃ (a b c : ℕ) (a' b' c' : ℕ),
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
  0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  0 < a' ∧ a' < 10 ∧ b' < 10 ∧ c' < 10 ∧
  (a ≠ a' ∨ b ≠ b' ∨ c ≠ c') ∧
  ∃ (k : ℕ), a * 10^(2*n+2) + b * 10^(n+1) + c = k^2 ∧
  ∃ (k' : ℕ), a' * 10^(2*n+2) + b' * 10^(n+1) + c' = k'^2 :=
by sorry

end three_digit_square_insertion_l881_88128


namespace deck_size_proof_l881_88193

theorem deck_size_proof (r b : ℕ) : 
  r / (r + b : ℚ) = 1/4 →
  r / (r + (b + 6) : ℚ) = 1/5 →
  r + b = 24 :=
by sorry

end deck_size_proof_l881_88193


namespace simplify_and_evaluate_l881_88170

theorem simplify_and_evaluate (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -2) :
  ((a^2 + 1) / a - 2) / ((a + 2) * (a - 1) / (a^2 + 2*a)) = a - 1 := by
  sorry

end simplify_and_evaluate_l881_88170


namespace garden_perimeter_l881_88159

/-- The perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: The perimeter of a rectangular garden with length 25 meters and width 15 meters is 80 meters -/
theorem garden_perimeter :
  rectanglePerimeter 25 15 = 80 := by
  sorry

end garden_perimeter_l881_88159


namespace annalise_tissue_purchase_cost_l881_88168

/-- Calculates the total cost of tissues given the number of boxes, packs per box, tissues per pack, and cost per tissue -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (costPerTissue : ℚ) : ℚ :=
  boxes * packsPerBox * tissuesPerPack * costPerTissue

/-- Proves that the total cost for Annalise's purchase is $1,000 -/
theorem annalise_tissue_purchase_cost :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

#eval totalCost 10 20 100 (5 / 100)

end annalise_tissue_purchase_cost_l881_88168


namespace max_value_implies_a_l881_88100

def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = 3/8 :=
sorry

end max_value_implies_a_l881_88100


namespace parabola_intercept_sum_l881_88113

/-- Parabola equation -/
def parabola (y : ℝ) : ℝ := y^2 - 4*y + 4

/-- X-intercept of the parabola -/
def a : ℝ := parabola 0

/-- Y-intercepts of the parabola -/
def b_and_c : Set ℝ := {y | parabola y = 0}

theorem parabola_intercept_sum :
  ∃ (b c : ℝ), b ∈ b_and_c ∧ c ∈ b_and_c ∧ a + b + c = 8 :=
sorry

end parabola_intercept_sum_l881_88113


namespace angle_through_point_l881_88141

theorem angle_through_point (α : Real) : 
  0 ≤ α ∧ α ≤ 2 * Real.pi → 
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
                      r * Real.sin α = Real.sin (2 * Real.pi / 3)) → 
  α = 5 * Real.pi / 3 := by
sorry

end angle_through_point_l881_88141


namespace bisection_method_solution_l881_88176

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem bisection_method_solution (h1 : f 2 < 0) (h2 : f 3 > 0) (h3 : f 2.5 < 0)
  (h4 : f 2.75 > 0) (h5 : f 2.625 > 0) (h6 : f 2.5625 > 0) :
  ∃ x : ℝ, x ∈ Set.Ioo 2.5 2.5625 ∧ |f x| < 0.1 :=
by
  sorry


end bisection_method_solution_l881_88176


namespace smallest_prime_factor_in_C_l881_88120

def C : Set Nat := {67, 71, 72, 73, 79}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧
  (∀ (m : Nat), m ∈ C → ∀ (p : Nat), Nat.Prime p → p ∣ m →
    ∃ (q : Nat), q ∣ n ∧ Nat.Prime q ∧ q ≤ p) ∧
  n = 72 :=
sorry

end smallest_prime_factor_in_C_l881_88120


namespace rubble_purchase_l881_88180

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial_amount notebook_cost pen_cost notebook_count pen_count : ℚ) : ℚ :=
  initial_amount - (notebook_cost * notebook_count + pen_cost * pen_count)

/-- Proves that Rubble will have $4.00 left after his purchase. -/
theorem rubble_purchase : 
  let initial_amount : ℚ := 15
  let notebook_cost : ℚ := 4
  let pen_cost : ℚ := 1.5
  let notebook_count : ℚ := 2
  let pen_count : ℚ := 2
  remaining_money initial_amount notebook_cost pen_cost notebook_count pen_count = 4 := by
  sorry

#eval remaining_money 15 4 1.5 2 2

end rubble_purchase_l881_88180


namespace trebled_result_proof_l881_88196

theorem trebled_result_proof (initial_number : ℕ) : 
  initial_number = 18 → 
  3 * (2 * initial_number + 5) = 123 := by
sorry

end trebled_result_proof_l881_88196


namespace problem_statement_l881_88189

theorem problem_statement (m n p q : ℕ) 
  (h : ∀ x : ℝ, x > 0 → (x + 1)^m / x^n - 1 = (x + 1)^p / x^q) :
  (m^2 + 2*n + p)^(2*q) = 9 := by
  sorry

end problem_statement_l881_88189


namespace stool_height_is_34cm_l881_88137

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height (ceiling_height floor_height : ℝ) 
                 (light_bulb_distance_from_ceiling : ℝ)
                 (alice_height alice_reach : ℝ) : ℝ :=
  ceiling_height - floor_height - light_bulb_distance_from_ceiling - 
  (alice_height + alice_reach)

/-- Theorem stating the height of the stool Alice needs -/
theorem stool_height_is_34cm :
  let ceiling_height : ℝ := 2.4 * 100  -- Convert to cm
  let floor_height : ℝ := 0
  let light_bulb_distance_from_ceiling : ℝ := 10
  let alice_height : ℝ := 1.5 * 100  -- Convert to cm
  let alice_reach : ℝ := 46
  stool_height ceiling_height floor_height light_bulb_distance_from_ceiling
                alice_height alice_reach = 34 := by
  sorry

#eval stool_height (2.4 * 100) 0 10 (1.5 * 100) 46

end stool_height_is_34cm_l881_88137


namespace red_ball_probability_l881_88167

/-- The probability of drawing a red ball from a pocket containing white, black, and red balls -/
theorem red_ball_probability (white black red : ℕ) (h : red = 1) :
  (red : ℚ) / (white + black + red : ℚ) = 1 / 9 :=
by
  sorry

#check red_ball_probability 3 5 1 rfl

end red_ball_probability_l881_88167


namespace count_integers_satisfying_inequality_l881_88156

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), 
    (∀ n : ℤ, n ∈ S ↔ (Real.sqrt n ≤ Real.sqrt (3 * n - 9) ∧ Real.sqrt (3 * n - 9) < Real.sqrt (n + 8))) ∧
    S.card = 4 := by
  sorry

end count_integers_satisfying_inequality_l881_88156


namespace remaining_distance_l881_88125

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 768) :
  total_distance - driven_distance = 432 := by
  sorry

end remaining_distance_l881_88125


namespace length_of_AE_l881_88104

/-- The length of segment AE in a 7x5 grid where AB meets CD at E -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 4) →
  B = (6, 0) →
  C = (6, 4) →
  D = (2, 0) →
  E = (4, 2) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • B) →
  (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (1 - s) • C + s • D) →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 10 * Real.sqrt 13 / 9 := by
  sorry


end length_of_AE_l881_88104


namespace cos_2a_over_1_plus_sin_2a_l881_88194

theorem cos_2a_over_1_plus_sin_2a (a : ℝ) (h : 4 * Real.sin a = 3 * Real.cos a) :
  (Real.cos (2 * a)) / (1 + Real.sin (2 * a)) = 1 / 7 := by
  sorry

end cos_2a_over_1_plus_sin_2a_l881_88194


namespace simplify_expression_l881_88133

theorem simplify_expression (y : ℝ) :
  3 * y + 7 * y^2 + 10 - (5 - 3 * y - 7 * y^2) = 14 * y^2 + 6 * y + 5 := by
  sorry

end simplify_expression_l881_88133


namespace stratified_sampling_theorem_l881_88117

/-- Represents the number of households in each category -/
structure HouseholdCounts where
  farmers : ℕ
  workers : ℕ
  intellectuals : ℕ

/-- Represents the sample sizes -/
structure SampleSizes where
  farmers : ℕ
  total : ℕ

/-- Theorem stating the relationship between the household counts, 
    sample sizes, and the expected total sample size -/
theorem stratified_sampling_theorem 
  (counts : HouseholdCounts) 
  (sample : SampleSizes) : 
  counts.farmers = 1500 →
  counts.workers = 401 →
  counts.intellectuals = 99 →
  sample.farmers = 75 →
  sample.total = 100 := by
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l881_88117


namespace new_profit_percentage_l881_88129

theorem new_profit_percentage
  (original_profit_rate : ℝ)
  (original_selling_price : ℝ)
  (price_reduction_rate : ℝ)
  (additional_profit : ℝ)
  (h1 : original_profit_rate = 0.1)
  (h2 : original_selling_price = 439.99999999999966)
  (h3 : price_reduction_rate = 0.1)
  (h4 : additional_profit = 28) :
  let original_cost_price := original_selling_price / (1 + original_profit_rate)
  let new_cost_price := original_cost_price * (1 - price_reduction_rate)
  let new_selling_price := original_selling_price + additional_profit
  let new_profit := new_selling_price - new_cost_price
  let new_profit_percentage := (new_profit / new_cost_price) * 100
  new_profit_percentage = 30 := by
sorry

end new_profit_percentage_l881_88129


namespace function_satisfies_equation_l881_88188

/-- The function y(x) satisfies the given differential equation. -/
theorem function_satisfies_equation (x : ℝ) (hx : x > 0) :
  let y : ℝ → ℝ := λ x => Real.tan (Real.log (3 * x))
  (1 + y x ^ 2) = x * (deriv y x) := by
  sorry

end function_satisfies_equation_l881_88188


namespace line_through_midpoint_l881_88114

-- Define the points
def P : ℝ × ℝ := (1, 3)
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 6)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + y - 6 = 0

-- Theorem statement
theorem line_through_midpoint :
  (P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →  -- P is midpoint of AB
  (A.2 = 0) →  -- A is on x-axis
  (B.1 = 0) →  -- B is on y-axis
  (line_equation P.1 P.2) →  -- Line passes through P
  (∀ x y, line_equation x y ↔ 3 * x + y - 6 = 0) :=  -- Prove the line equation
by sorry

end line_through_midpoint_l881_88114


namespace capital_payment_theorem_l881_88132

def remaining_capital (m : ℕ) (d : ℚ) : ℚ :=
  (3/2)^(m-1) * (3000 - 3*d) + 2*d

theorem capital_payment_theorem (m : ℕ) (h : m ≥ 3) :
  ∃ d : ℚ, remaining_capital m d = 4000 ∧ 
    d = (1000 * (3^m - 2^(m+1))) / (3^m - 2^m) := by
  sorry

end capital_payment_theorem_l881_88132


namespace odd_score_probability_is_four_ninths_l881_88157

/-- Represents the possible points on the dart board -/
inductive DartPoints
  | Three
  | Four

/-- Represents the regions on the dart board -/
structure DartRegion where
  isInner : Bool
  points : DartPoints

/-- The dart board configuration -/
def dartBoard : List DartRegion :=
  [
    { isInner := true,  points := DartPoints.Three },
    { isInner := true,  points := DartPoints.Four },
    { isInner := true,  points := DartPoints.Four },
    { isInner := false, points := DartPoints.Four },
    { isInner := false, points := DartPoints.Three },
    { isInner := false, points := DartPoints.Three }
  ]

/-- The probability of hitting each region -/
def regionProbability (region : DartRegion) : ℚ :=
  if region.isInner then 1 / 21 else 2 / 21

/-- The probability of getting an odd score with two dart throws -/
def oddScoreProbability : ℚ := sorry

/-- Theorem stating that the probability of getting an odd score is 4/9 -/
theorem odd_score_probability_is_four_ninths :
  oddScoreProbability = 4 / 9 := by sorry

end odd_score_probability_is_four_ninths_l881_88157


namespace expression_evaluation_l881_88175

theorem expression_evaluation :
  let d : ℕ := 4
  (d^d - d*(d - 2)^d + d^2)^d = 1874164224 := by
  sorry

end expression_evaluation_l881_88175


namespace triangle_solution_l881_88174

theorem triangle_solution (a b c A B C : ℝ) : 
  a = 2 * Real.sqrt 2 →
  A = π / 4 →
  B = π / 6 →
  C = π - A - B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b = 2 ∧ c = Real.sqrt 6 + Real.sqrt 2 := by
sorry

end triangle_solution_l881_88174


namespace mouse_jump_distance_l881_88182

/-- The jump distances of animals in a contest -/
def JumpContest (grasshopper frog mouse : ℕ) : Prop :=
  (grasshopper = 39) ∧ 
  (grasshopper = frog + 19) ∧
  (frog = mouse + 12)

/-- Theorem: Given the conditions of the jump contest, the mouse jumped 8 inches -/
theorem mouse_jump_distance (grasshopper frog mouse : ℕ) 
  (h : JumpContest grasshopper frog mouse) : mouse = 8 := by
  sorry

end mouse_jump_distance_l881_88182


namespace prob_sum_le_10_prob_sum_le_10_is_11_12_l881_88192

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of outcomes where the sum is greater than 10 -/
def outcomesGreaterThan10 : ℕ := 3

/-- The probability that the sum of two fair six-sided dice is less than or equal to 10 -/
theorem prob_sum_le_10 : ℚ :=
  1 - (outcomesGreaterThan10 : ℚ) / totalOutcomes

/-- Proof that the probability of the sum of two fair six-sided dice being less than or equal to 10 is 11/12 -/
theorem prob_sum_le_10_is_11_12 : prob_sum_le_10 = 11 / 12 := by
  sorry

end prob_sum_le_10_prob_sum_le_10_is_11_12_l881_88192


namespace students_neither_music_nor_art_l881_88148

theorem students_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) :
  total = 500 →
  music = 40 →
  art = 20 →
  both = 10 →
  total - (music + art - both) = 450 :=
by
  sorry

end students_neither_music_nor_art_l881_88148


namespace mountain_bike_helmet_cost_l881_88136

/-- Calculates the cost of a mountain bike helmet based on Alfonso's savings and earnings --/
theorem mountain_bike_helmet_cost
  (daily_earnings : ℕ)
  (current_savings : ℕ)
  (days_per_week : ℕ)
  (weeks_to_work : ℕ)
  (h1 : daily_earnings = 6)
  (h2 : current_savings = 40)
  (h3 : days_per_week = 5)
  (h4 : weeks_to_work = 10) :
  daily_earnings * days_per_week * weeks_to_work + current_savings = 340 :=
by
  sorry

end mountain_bike_helmet_cost_l881_88136


namespace integral_problem_l881_88177

theorem integral_problem (f : ℝ → ℝ) 
  (h1 : ∫ (x : ℝ) in Set.Iic 1, f x = 1)
  (h2 : ∫ (x : ℝ) in Set.Iic 2, f x = -1) :
  ∫ (x : ℝ) in Set.Ioc 1 2, f x = -2 := by
  sorry

end integral_problem_l881_88177


namespace length_AM_l881_88121

/-- Square ABCD with side length 9 -/
structure Square (A B C D : ℝ × ℝ) :=
  (side_length : ℝ)
  (is_square : side_length = 9)

/-- Point P on AB such that AP:PB = 7:2 -/
def P (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Quarter circle with center C and radius CB -/
def QuarterCircle (C B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Point E where tangent from P meets the quarter circle -/
def E (P : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- Point Q where tangent from P meets AD -/
def Q (P : ℝ × ℝ) (A D : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Point K where CE and DB meet -/
def K (C E D B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Point M where AK and PQ meet -/
def M (A K P Q : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem length_AM (A B C D : ℝ × ℝ) (square : Square A B C D) :
  let P := P A B
  let circle := QuarterCircle C B
  let E := E P circle
  let Q := Q P A D
  let K := K C E D B
  let M := M A K P Q
  distance A M = 85 / 22 := by
  sorry

end length_AM_l881_88121


namespace total_seashells_is_fifty_l881_88109

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := 13

/-- The total number of seashells found by Tim and Sally -/
def total_seashells : ℕ := tim_seashells + sally_seashells

/-- Theorem: The total number of seashells found by Tim and Sally is 50 -/
theorem total_seashells_is_fifty : total_seashells = 50 := by
  sorry

end total_seashells_is_fifty_l881_88109


namespace fourth_root_over_seventh_root_of_seven_l881_88184

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (hx : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end fourth_root_over_seventh_root_of_seven_l881_88184


namespace expand_product_l881_88122

theorem expand_product (x : ℝ) : -2 * (x - 3) * (x + 4) * (2*x - 1) = -4*x^3 - 2*x^2 + 50*x - 24 := by
  sorry

end expand_product_l881_88122


namespace sabertooth_tails_count_l881_88150

/-- Represents the number of legs for Triassic Discoglossus tadpoles -/
def triassic_legs : ℕ := 5

/-- Represents the number of tails for Triassic Discoglossus tadpoles -/
def triassic_tails : ℕ := 1

/-- Represents the number of legs for Sabertooth Frog tadpoles -/
def sabertooth_legs : ℕ := 4

/-- Represents the total number of legs of all captured tadpoles -/
def total_legs : ℕ := 100

/-- Represents the total number of tails of all captured tadpoles -/
def total_tails : ℕ := 64

/-- Proves that the number of tails per Sabertooth Frog tadpole is 3 -/
theorem sabertooth_tails_count :
  ∃ (n k : ℕ),
    n * triassic_legs + k * sabertooth_legs = total_legs ∧
    n * triassic_tails + k * 3 = total_tails :=
by sorry

end sabertooth_tails_count_l881_88150


namespace problem_statement_l881_88103

-- Define the function f
def f (a b c x : ℝ) : ℝ := a*x + b*x - c*x

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define an obtuse triangle
def is_obtuse (a b c : ℝ) : Prop :=
  a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2

theorem problem_statement 
  (a b c : ℝ) 
  (h1 : c > a ∧ a > 0) 
  (h2 : c > b ∧ b > 0) 
  (h3 : triangle_inequality a b c) :
  (∃ x : ℝ, ¬ triangle_inequality (a*x) (b*x) (c*x)) ∧ 
  (is_obtuse a b c → ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f a b c x = 0) :=
sorry

end problem_statement_l881_88103


namespace multiplication_value_proof_l881_88134

theorem multiplication_value_proof : 
  let initial_number : ℝ := 2.25
  let division_factor : ℝ := 3
  let multiplication_value : ℝ := 12
  let result : ℝ := 9
  (initial_number / division_factor) * multiplication_value = result := by
sorry

end multiplication_value_proof_l881_88134


namespace boys_percentage_in_specific_classroom_l881_88115

/-- Represents the composition of a classroom -/
structure Classroom where
  total_people : ℕ
  boy_girl_ratio : ℚ
  student_teacher_ratio : ℕ

/-- Calculates the percentage of boys in the classroom -/
def boys_percentage (c : Classroom) : ℚ :=
  sorry

/-- Theorem stating the percentage of boys in the specific classroom scenario -/
theorem boys_percentage_in_specific_classroom :
  let c : Classroom := {
    total_people := 36,
    boy_girl_ratio := 2 / 3,
    student_teacher_ratio := 6
  }
  boys_percentage c = 400 / 7 := by
  sorry

end boys_percentage_in_specific_classroom_l881_88115


namespace rock_collecting_contest_l881_88147

theorem rock_collecting_contest (sydney_initial : ℕ) (conner_initial : ℕ) 
  (conner_day2 : ℕ) (conner_day3 : ℕ) :
  sydney_initial = 837 →
  conner_initial = 723 →
  conner_day2 = 123 →
  conner_day3 = 27 →
  ∃ (sydney_day1 : ℕ),
    sydney_day1 ≤ 4 ∧
    sydney_day1 > 0 ∧
    conner_initial + 8 * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_initial + sydney_day1 + 16 * sydney_day1 ∧
    ∀ (x : ℕ), x > sydney_day1 →
      conner_initial + 8 * x + conner_day2 + conner_day3 < 
      sydney_initial + x + 16 * x :=
by sorry

end rock_collecting_contest_l881_88147


namespace call_service_comparison_l881_88161

/-- Represents the cost of a phone call service -/
structure CallService where
  monthly_fee : ℝ
  per_minute_rate : ℝ

/-- Calculates the total cost for a given call duration -/
def total_cost (service : CallService) (duration : ℝ) : ℝ :=
  service.monthly_fee + service.per_minute_rate * duration

/-- Global Call service -/
def global_call : CallService :=
  { monthly_fee := 50, per_minute_rate := 0.4 }

/-- China Mobile service -/
def china_mobile : CallService :=
  { monthly_fee := 0, per_minute_rate := 0.6 }

theorem call_service_comparison :
  ∃ (x : ℝ),
    (∀ (duration : ℝ), total_cost global_call duration = 50 + 0.4 * duration) ∧
    (∀ (duration : ℝ), total_cost china_mobile duration = 0.6 * duration) ∧
    (total_cost global_call x = total_cost china_mobile x ∧ x = 125) ∧
    (∀ (duration : ℝ), duration > 125 → total_cost global_call duration < total_cost china_mobile duration) :=
by sorry

end call_service_comparison_l881_88161


namespace quadratic_form_h_value_l881_88107

theorem quadratic_form_h_value : ∃ (a k : ℝ), ∀ x : ℝ, 
  3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k := by
  sorry

end quadratic_form_h_value_l881_88107


namespace goldfish_count_l881_88139

/-- Represents the number of fish in each tank -/
structure FishTanks where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the composition of the first tank -/
structure FirstTank where
  goldfish : ℕ
  beta : ℕ

/-- The problem statement -/
theorem goldfish_count (tanks : FishTanks) (first : FirstTank) : 
  tanks.first = first.goldfish + first.beta ∧
  tanks.second = 2 * tanks.first ∧
  tanks.third = tanks.second / 3 ∧
  tanks.third = 10 ∧
  first.beta = 8 →
  first.goldfish = 7 := by
  sorry

end goldfish_count_l881_88139


namespace DR_length_zero_l881_88190

/-- Rectangle ABCD with inscribed circle ω -/
structure RectangleWithCircle where
  /-- Length of the rectangle -/
  length : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- Center of the inscribed circle -/
  center : ℝ × ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Point Q where the circle intersects AB -/
  Q : ℝ × ℝ
  /-- Point D at the bottom left corner -/
  D : ℝ × ℝ
  /-- Point R where DQ intersects the circle again -/
  R : ℝ × ℝ
  /-- The rectangle has length 2 and height 1 -/
  h_dimensions : length = 2 ∧ height = 1
  /-- The circle is inscribed in the rectangle -/
  h_inscribed : center = (0, 0) ∧ radius = height / 2
  /-- Q is on the top edge of the rectangle -/
  h_Q_on_top : Q.2 = height / 2
  /-- D is at the bottom left corner -/
  h_D_position : D = (0, -height / 2)
  /-- R is on the circle -/
  h_R_on_circle : (R.1 - center.1)^2 + (R.2 - center.2)^2 = radius^2
  /-- R is on line DQ -/
  h_R_on_DQ : R.1 = D.1 ∧ R.1 = Q.1

/-- The main theorem: DR has length 0 -/
theorem DR_length_zero (rect : RectangleWithCircle) : dist rect.D rect.R = 0 :=
  sorry


end DR_length_zero_l881_88190


namespace quadratic_inequality_solution_l881_88166

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 12 ↔ -4 < x ∧ x < -3 := by
  sorry

end quadratic_inequality_solution_l881_88166


namespace triangle_max_perimeter_l881_88149

theorem triangle_max_perimeter (A B C : ℝ) (a b c : ℝ) :
  A = 2 * π / 3 →
  a = 3 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = 2 * Real.sin (A / 2) * Real.sin (B / 2) / Real.sin ((A + B) / 2) →
  b = 2 * Real.sin (B / 2) * Real.sin (C / 2) / Real.sin ((B + C) / 2) →
  c = 2 * Real.sin (C / 2) * Real.sin (A / 2) / Real.sin ((C + A) / 2) →
  (∀ B' C' a' b' c',
    A + B' + C' = π →
    a' > 0 ∧ b' > 0 ∧ c' > 0 →
    a' = 2 * Real.sin (A / 2) * Real.sin (B' / 2) / Real.sin ((A + B') / 2) →
    b' = 2 * Real.sin (B' / 2) * Real.sin (C' / 2) / Real.sin ((B' + C') / 2) →
    c' = 2 * Real.sin (C' / 2) * Real.sin (A / 2) / Real.sin ((C' + A) / 2) →
    a' + b' + c' ≤ a + b + c) →
  a + b + c = 3 + 2 * Real.sqrt 3 := by
sorry

end triangle_max_perimeter_l881_88149


namespace quadratic_radical_equality_l881_88105

theorem quadratic_radical_equality (x y : ℚ) : 
  (x - y*x + y - 1 = 2 ∧ x + y - 1 = 3*x + 2*y - 4) → x*y = -5/9 := by
  sorry

end quadratic_radical_equality_l881_88105


namespace determine_key_lock_pairs_l881_88135

/-- Represents a lock -/
structure Lock :=
  (id : Nat)

/-- Represents a key -/
structure Key :=
  (id : Nat)

/-- Represents a pair of locks that a key can open -/
structure LockPair :=
  (lock1 : Lock)
  (lock2 : Lock)

/-- Represents the result of testing a key on a lock -/
inductive TestResult
  | Opens
  | DoesNotOpen

/-- Represents the state of knowledge about which keys open which locks -/
structure KeyLockState :=
  (locks : Finset Lock)
  (keys : Finset Key)
  (openPairs : Finset (Key × LockPair))

/-- Represents a single test of a key on a lock -/
def test (k : Key) (l : Lock) : TestResult := sorry

/-- The main theorem to prove -/
theorem determine_key_lock_pairs 
  (locks : Finset Lock) 
  (keys : Finset Key) 
  (h1 : locks.card = 4) 
  (h2 : keys.card = 6) 
  (h3 : ∀ k : Key, k ∈ keys → (∃! p : LockPair, p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧ 
    test k p.lock1 = TestResult.Opens ∧ test k p.lock2 = TestResult.Opens))
  (h4 : ∀ k1 k2 : Key, k1 ∈ keys → k2 ∈ keys → k1 ≠ k2 → 
    ¬∃ p : LockPair, p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧ 
    test k1 p.lock1 = TestResult.Opens ∧ test k1 p.lock2 = TestResult.Opens ∧
    test k2 p.lock1 = TestResult.Opens ∧ test k2 p.lock2 = TestResult.Opens) :
  ∃ (final_state : KeyLockState) (test_count : Nat),
    test_count ≤ 13 ∧
    final_state.locks = locks ∧
    final_state.keys = keys ∧
    (∀ k : Key, k ∈ keys → 
      ∃! p : LockPair, (k, p) ∈ final_state.openPairs ∧ 
        p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧
        test k p.lock1 = TestResult.Opens ∧ 
        test k p.lock2 = TestResult.Opens) :=
by
  sorry

end determine_key_lock_pairs_l881_88135


namespace angle_greater_than_120_degrees_l881_88173

open Real Set

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- The theorem statement -/
theorem angle_greater_than_120_degrees (n : ℕ) (points : Finset Point) :
  points.card = n →
  ∃ (ordered_points : Fin n → Point),
    (∀ i : Fin n, ordered_points i ∈ points) ∧
    (∀ (i j k : Fin n), i < j → j < k →
      angle (ordered_points i) (ordered_points j) (ordered_points k) > 120 * π / 180) :=
sorry

end angle_greater_than_120_degrees_l881_88173


namespace sqrt_3_power_calculation_l881_88146

theorem sqrt_3_power_calculation : 
  (Real.sqrt ((Real.sqrt 3) ^ 5)) ^ 6 = 2187 * Real.sqrt 3 := by
  sorry

end sqrt_3_power_calculation_l881_88146


namespace unique_solution_mod_37_l881_88199

theorem unique_solution_mod_37 :
  ∃! (a b c d : ℤ),
    (a^2 + b*c) % 37 = a % 37 ∧
    (b*(a + d)) % 37 = b % 37 ∧
    (c*(a + d)) % 37 = c % 37 ∧
    (b*c + d^2) % 37 = d % 37 ∧
    (a*d - b*c) % 37 = 1 % 37 :=
by sorry

end unique_solution_mod_37_l881_88199


namespace f_monotone_decreasing_on_interval_l881_88181

-- Define the function f(x) = 3x^5 - 5x^3
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

-- State the theorem
theorem f_monotone_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y := by
  sorry

end f_monotone_decreasing_on_interval_l881_88181


namespace smallest_constant_for_ratio_difference_l881_88126

theorem smallest_constant_for_ratio_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∃ (i j k l : Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    |a₁ / a₂ - a₃ / a₄| ≤ (1/2 : ℝ)) ∧
  (∀ C < (1/2 : ℝ), ∃ (b₁ b₂ b₃ b₄ b₅ : ℝ),
    ∀ (i j k l : Fin 5), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
      |b₁ / b₂ - b₃ / b₄| > C) :=
by sorry

end smallest_constant_for_ratio_difference_l881_88126


namespace james_socks_count_l881_88118

/-- The number of pairs of red socks James has -/
def red_pairs : ℕ := 20

/-- The number of red socks James has -/
def red_socks : ℕ := red_pairs * 2

/-- The number of black socks James has -/
def black_socks : ℕ := red_socks / 2

/-- The number of red and black socks combined -/
def red_black_socks : ℕ := red_socks + black_socks

/-- The number of white socks James has -/
def white_socks : ℕ := red_black_socks * 2

/-- The total number of socks James has -/
def total_socks : ℕ := red_socks + black_socks + white_socks

theorem james_socks_count : total_socks = 180 := by
  sorry

end james_socks_count_l881_88118


namespace isosceles_triangle_base_angle_l881_88165

/-- An isosceles triangle with one angle of 94 degrees has a base angle of 43 degrees. -/
theorem isosceles_triangle_base_angle : ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal (isosceles property)
  c = 94 →           -- One angle is 94°
  a = 43 :=          -- One of the base angles is 43°
by
  sorry

end isosceles_triangle_base_angle_l881_88165


namespace counterexample_exists_l881_88112

theorem counterexample_exists : ∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ ¬(Nat.Prime (p^2 - 2) ∧ Odd (p^2 - 2)) := by
  sorry

end counterexample_exists_l881_88112


namespace pride_and_prejudice_watch_time_l881_88162

/-- The number of hours spent watching a TV series -/
def watch_time (num_episodes : ℕ) (episode_length : ℕ) : ℚ :=
  (num_episodes * episode_length : ℚ) / 60

/-- Theorem: Watching 6 episodes of 50 minutes each takes 5 hours -/
theorem pride_and_prejudice_watch_time :
  watch_time 6 50 = 5 := by sorry

end pride_and_prejudice_watch_time_l881_88162


namespace regular_polygon_sides_l881_88169

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end regular_polygon_sides_l881_88169


namespace min_value_product_quotient_min_value_achieved_l881_88179

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 4*x + 2) * (y^2 + 5*y + 3) * (z^2 + 6*z + 4) / (x*y*z) ≥ 336 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 4*a + 2) * (b^2 + 5*b + 3) * (c^2 + 6*c + 4) / (a*b*c) = 336 :=
by sorry

end min_value_product_quotient_min_value_achieved_l881_88179


namespace stamp_collection_gcd_l881_88191

theorem stamp_collection_gcd : Nat.gcd (Nat.gcd 945 1260) 630 = 105 := by
  sorry

end stamp_collection_gcd_l881_88191


namespace baker_cakes_sold_l881_88119

theorem baker_cakes_sold (initial_cakes : ℕ) (bought_cakes : ℕ) (remaining_cakes : ℕ) :
  initial_cakes = 121 →
  bought_cakes = 170 →
  remaining_cakes = 186 →
  ∃ (sold_cakes : ℕ), sold_cakes = 105 ∧ initial_cakes - sold_cakes + bought_cakes = remaining_cakes :=
by
  sorry

end baker_cakes_sold_l881_88119


namespace cube_volume_from_space_diagonal_l881_88187

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 9) : 
  let s := d / Real.sqrt 3
  s^3 = 81 * Real.sqrt 3 := by
  sorry

end cube_volume_from_space_diagonal_l881_88187


namespace f_symmetry_l881_88197

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- State the theorem
theorem f_symmetry (a b : ℝ) : 
  f a b (-5) = 3 → f a b 5 = 1 := by
  sorry

end f_symmetry_l881_88197


namespace convergence_bound_minimal_k_smallest_k_is_five_l881_88130

def v : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * v n - 3 * (v n)^2

def M : ℚ := 1/2

theorem convergence_bound (k : ℕ) : k ≥ 5 → |v k - M| ≤ 1/2^500 := by sorry

theorem minimal_k : ∀ j : ℕ, j < 5 → |v j - M| > 1/2^500 := by sorry

theorem smallest_k_is_five : 
  (∃ k : ℕ, |v k - M| ≤ 1/2^500) ∧ 
  (∀ j : ℕ, |v j - M| ≤ 1/2^500 → j ≥ 5) := by sorry

end convergence_bound_minimal_k_smallest_k_is_five_l881_88130


namespace count_four_digit_integers_l881_88127

/-- The number of distinct four-digit positive integers formed with digits 3, 3, 8, and 8 -/
def fourDigitIntegersCount : ℕ := 6

/-- The set of digits used to form the integers -/
def digits : Finset ℕ := {3, 8}

/-- The number of times each digit is used -/
def digitRepetitions : ℕ := 2

/-- The total number of digits used -/
def totalDigits : ℕ := 4

theorem count_four_digit_integers :
  fourDigitIntegersCount = (totalDigits.factorial) / (digitRepetitions.factorial ^ digits.card) :=
sorry

end count_four_digit_integers_l881_88127


namespace class_point_system_l881_88195

/-- Calculates the number of tasks required for a given number of points -/
def tasksRequired (points : ℕ) : ℕ :=
  let fullSets := (points - 1) / 3
  let taskMultiplier := min fullSets 2 + 1
  taskMultiplier * ((points + 2) / 3)

/-- The point-earning system for the class -/
theorem class_point_system (points : ℕ) :
  points = 18 → tasksRequired points = 10 :=
by
  sorry

#eval tasksRequired 18  -- Should output 10

end class_point_system_l881_88195


namespace diamond_equation_solution_l881_88164

-- Define the diamond operation
noncomputable def diamond (b c : ℝ) : ℝ := b + Real.sqrt (c + Real.sqrt (c + Real.sqrt c))

-- State the theorem
theorem diamond_equation_solution (k : ℝ) :
  diamond 10 k = 13 → k = 6 := by
  sorry

end diamond_equation_solution_l881_88164


namespace integer_ratio_condition_l881_88155

theorem integer_ratio_condition (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ a ∈ S, ∃ k : ℤ, (a^m + a - 1 : ℤ) = k * (a^n + a^2 - 1)) →
  (m = n + 2 ∧ m = 5 ∧ n = 3) :=
by sorry

end integer_ratio_condition_l881_88155


namespace betty_bracelets_l881_88110

/-- Given that Betty has 88.0 pink flower stones and each bracelet requires 11 stones,
    prove that the number of bracelets she can make is 8. -/
theorem betty_bracelets :
  let total_stones : ℝ := 88.0
  let stones_per_bracelet : ℕ := 11
  (total_stones / stones_per_bracelet : ℝ) = 8 := by
  sorry

end betty_bracelets_l881_88110


namespace cube_surface_area_from_diagonal_l881_88138

/-- Given a cube with diagonal length a/2, prove that its total surface area is a^2/2 -/
theorem cube_surface_area_from_diagonal (a : ℝ) (h : a > 0) :
  let diagonal := a / 2
  let side := diagonal / Real.sqrt 3
  let surface_area := 6 * side ^ 2
  surface_area = a ^ 2 / 2 := by
  sorry

end cube_surface_area_from_diagonal_l881_88138


namespace patio_layout_l881_88116

theorem patio_layout (r c : ℕ) : 
  r * c = 160 ∧ 
  (r + 4) * (c - 2) = 160 ∧ 
  r % 5 = 0 ∧ 
  c % 5 = 0 → 
  r = 16 := by sorry

end patio_layout_l881_88116


namespace cookie_difference_l881_88111

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_choc : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def morning_raisin : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def morning_choc : ℕ := 237

/-- The total number of chocolate chip cookies Helen baked -/
def total_choc : ℕ := yesterday_choc + morning_choc

/-- The difference between chocolate chip cookies and raisin cookies -/
theorem cookie_difference : total_choc - morning_raisin = 25 := by
  sorry

end cookie_difference_l881_88111


namespace henry_total_score_l881_88143

def geography_score : ℝ := 50
def math_score : ℝ := 70
def english_score : ℝ := 66
def science_score : ℝ := 84
def french_score : ℝ := 75

def geography_weight : ℝ := 0.25
def math_weight : ℝ := 0.20
def english_weight : ℝ := 0.20
def science_weight : ℝ := 0.15
def french_weight : ℝ := 0.10

def history_score : ℝ :=
  geography_score * geography_weight +
  math_score * math_weight +
  english_score * english_weight +
  science_score * science_weight +
  french_score * french_weight

def total_score : ℝ :=
  geography_score + math_score + english_score + science_score + french_score + history_score

theorem henry_total_score :
  total_score = 404.8 := by
  sorry

end henry_total_score_l881_88143


namespace correct_systematic_sample_l881_88140

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Generates the sequence of selected students -/
def generate_sequence (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.interval)

/-- Checks if a sequence is valid for the given systematic sample -/
def is_valid_sequence (s : SystematicSample) (seq : List Nat) : Prop :=
  seq.length = s.sample_size ∧
  seq.all (· ≤ s.total) ∧
  seq = generate_sequence s

theorem correct_systematic_sample :
  let s : SystematicSample := ⟨50, 5, 3, 10⟩
  is_valid_sequence s [3, 13, 23, 33, 43] := by
  sorry

#eval generate_sequence ⟨50, 5, 3, 10⟩

end correct_systematic_sample_l881_88140


namespace number_equation_solution_l881_88106

theorem number_equation_solution : 
  ∃! x : ℝ, 45 - (28 - (37 - (x - 18))) = 57 ∧ x = 15 := by
  sorry

end number_equation_solution_l881_88106


namespace perpendicular_slope_l881_88186

theorem perpendicular_slope (x y : ℝ) :
  (3 * x - 4 * y = 8) →
  (∃ m : ℝ, m = -4/3 ∧ m * (3/4) = -1) :=
sorry

end perpendicular_slope_l881_88186


namespace circle_M_equation_l881_88108

/-- Circle M passing through two points with center on a line -/
def circle_M (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), 
    (a - b - 4 = 0) ∧ 
    (((-1) - a)^2 + ((-4) - b)^2 = (x - a)^2 + (y - b)^2) ∧
    ((6 - a)^2 + (3 - b)^2 = (x - a)^2 + (y - b)^2)

/-- Theorem: The equation of circle M is (x-3)^2 + (y+1)^2 = 25 -/
theorem circle_M_equation : 
  ∀ x y : ℝ, circle_M x y ↔ (x - 3)^2 + (y + 1)^2 = 25 :=
by sorry

end circle_M_equation_l881_88108


namespace gain_percent_problem_l881_88124

def gain_percent (gain : ℚ) (cost_price : ℚ) : ℚ :=
  (gain / cost_price) * 100

theorem gain_percent_problem (gain : ℚ) (cost_price : ℚ) 
  (h1 : gain = 70 / 100)  -- 70 paise = 0.70 rupees
  (h2 : cost_price = 70) : 
  gain_percent gain cost_price = 1 := by
  sorry

end gain_percent_problem_l881_88124


namespace problem_solution_l881_88101

theorem problem_solution : (69842 * 69842 - 30158 * 30158) / (69842 - 30158) = 100000 := by
  sorry

end problem_solution_l881_88101


namespace multiplicative_inverse_137_mod_391_l881_88144

theorem multiplicative_inverse_137_mod_391 :
  ∃ x : ℕ, x < 391 ∧ (137 * x) % 391 = 1 ∧ x = 294 := by
  sorry

end multiplicative_inverse_137_mod_391_l881_88144


namespace complete_square_quadratic_l881_88160

theorem complete_square_quadratic (x : ℝ) : 
  x^2 + 10*x - 3 = 0 ↔ (x + 5)^2 = 28 :=
by sorry

end complete_square_quadratic_l881_88160


namespace tree_planting_event_l881_88163

/-- Calculates 60% of the total number of participants in a tree planting event -/
theorem tree_planting_event (boys : ℕ) (girls : ℕ) : 
  boys = 600 →
  girls - boys = 400 →
  girls > boys →
  (boys + girls) * 60 / 100 = 960 := by
sorry

end tree_planting_event_l881_88163


namespace quadratic_inequality_solution_l881_88154

theorem quadratic_inequality_solution (y : ℝ) :
  -y^2 + 9*y - 20 < 0 ↔ y < 4 ∨ y > 5 := by
  sorry

end quadratic_inequality_solution_l881_88154


namespace greatest_integer_with_gcd_six_exists_192_with_gcd_six_no_greater_than_192_solution_is_192_l881_88178

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192 :=
by sorry

theorem exists_192_with_gcd_six : Nat.gcd 192 18 = 6 :=
by sorry

theorem no_greater_than_192 :
  ∀ m : ℕ, 192 < m → m < 200 → Nat.gcd m 18 ≠ 6 :=
by sorry

theorem solution_is_192 : 
  ∃! n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n :=
by sorry

end greatest_integer_with_gcd_six_exists_192_with_gcd_six_no_greater_than_192_solution_is_192_l881_88178


namespace fish_given_by_ben_l881_88145

theorem fish_given_by_ben (initial_fish : ℕ) (current_fish : ℕ) 
  (h1 : initial_fish = 31) (h2 : current_fish = 49) : 
  current_fish - initial_fish = 18 := by
  sorry

end fish_given_by_ben_l881_88145


namespace complex_fraction_equals_2i_l881_88171

theorem complex_fraction_equals_2i :
  let z : ℂ := 1 + I
  (z^2 - 2*z) / (z - 1) = 2*I :=
by sorry

end complex_fraction_equals_2i_l881_88171


namespace teacher_estimate_difference_l881_88198

/-- The difference between the teacher's estimated increase and the actual increase in exam scores -/
theorem teacher_estimate_difference (expected_increase actual_increase : ℕ) 
  (h1 : expected_increase = 2152)
  (h2 : actual_increase = 1264) : 
  expected_increase - actual_increase = 888 := by
  sorry

end teacher_estimate_difference_l881_88198


namespace candy_box_distribution_l881_88131

theorem candy_box_distribution :
  ∃ (x y z : ℕ), 
    x * 16 + y * 17 + z * 21 = 185 ∧ 
    x = 5 ∧ 
    y = 0 ∧ 
    z = 5 :=
by sorry

end candy_box_distribution_l881_88131


namespace simplify_and_evaluate_l881_88153

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -2) (h2 : b = 1/3) :
  4 * (a^2 - 2*a*b) - (3*a^2 - 5*a*b + 1) = 5 := by
  sorry

end simplify_and_evaluate_l881_88153


namespace lottery_probability_l881_88152

/-- The probability of exactly one person winning a prize in a lottery. -/
theorem lottery_probability : 
  let total_tickets : ℕ := 3
  let winning_tickets : ℕ := 2
  let people_drawing : ℕ := 2
  -- Probability of exactly one person winning
  (1 : ℚ) - (winning_tickets : ℚ) / (total_tickets : ℚ) * ((winning_tickets - 1) : ℚ) / ((total_tickets - 1) : ℚ) = 2 / 3 :=
by sorry

end lottery_probability_l881_88152


namespace arithmetic_sequence_proof_l881_88183

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 2 / (n * (a n + 3))

-- Define the sum S_n
def S (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_sequence_proof :
  (a 3 = 5) ∧ (a 17 = 3 * a 6) ∧
  (∀ n : ℕ, n > 0 → b n = 1 / (n * (n + 1))) ∧
  (∀ n : ℕ, n > 0 → S n = n / (n + 1)) := by
  sorry


end arithmetic_sequence_proof_l881_88183
