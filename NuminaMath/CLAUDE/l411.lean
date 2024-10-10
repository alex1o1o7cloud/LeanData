import Mathlib

namespace time_for_one_toy_l411_41107

/-- Represents the time (in hours) it takes to make a certain number of toys -/
structure ToyProduction where
  hours : ℝ
  toys : ℝ

/-- Given that 50 toys are made in 100 hours, prove that it takes 2 hours to make one toy -/
theorem time_for_one_toy (prod : ToyProduction) 
  (h1 : prod.hours = 100) 
  (h2 : prod.toys = 50) : 
  prod.hours / prod.toys = 2 := by
  sorry

end time_for_one_toy_l411_41107


namespace train_rate_problem_l411_41186

/-- The constant rate of Train A when two trains meet under specific conditions -/
theorem train_rate_problem (total_distance : ℝ) (train_b_rate : ℝ) (train_a_distance : ℝ) :
  total_distance = 350 →
  train_b_rate = 30 →
  train_a_distance = 200 →
  ∃ (train_a_rate : ℝ),
    train_a_rate * (total_distance - train_a_distance) / train_b_rate = train_a_distance ∧
    train_a_rate = 40 :=
by sorry

end train_rate_problem_l411_41186


namespace segment_length_is_15_l411_41125

/-- The length of a vertical line segment is the absolute difference of y-coordinates -/
def vertical_segment_length (y1 y2 : ℝ) : ℝ := |y2 - y1|

/-- Proof that the length of the segment with endpoints (3, 5) and (3, 20) is 15 units -/
theorem segment_length_is_15 : 
  vertical_segment_length 5 20 = 15 := by
  sorry

end segment_length_is_15_l411_41125


namespace cartons_used_is_38_l411_41160

/-- Represents the packing of tennis rackets into cartons. -/
structure RacketPacking where
  total_rackets : ℕ
  cartons_of_three : ℕ
  cartons_of_two : ℕ

/-- Calculates the total number of cartons used. -/
def total_cartons (packing : RacketPacking) : ℕ :=
  packing.cartons_of_two + packing.cartons_of_three

/-- Theorem stating that for the given packing scenario, 38 cartons are used in total. -/
theorem cartons_used_is_38 (packing : RacketPacking) 
  (h1 : packing.total_rackets = 100)
  (h2 : packing.cartons_of_three = 24)
  (h3 : 2 * packing.cartons_of_two + 3 * packing.cartons_of_three = packing.total_rackets) :
  total_cartons packing = 38 := by
  sorry

#check cartons_used_is_38

end cartons_used_is_38_l411_41160


namespace seven_c_plus_seven_d_equals_five_l411_41196

-- Define the function h
def h (x : ℝ) : ℝ := 7 * x - 6

-- Define the function f
def f (c d x : ℝ) : ℝ := c * x + d

-- State the theorem
theorem seven_c_plus_seven_d_equals_five 
  (c d : ℝ) 
  (h_def : ∀ x, h x = 7 * x - 6)
  (h_inverse : ∀ x, h x = f c d⁻¹ x - 2)
  (f_inverse : ∀ x, f c d (f c d⁻¹ x) = x) :
  7 * c + 7 * d = 5 := by
sorry

end seven_c_plus_seven_d_equals_five_l411_41196


namespace minimum_value_range_l411_41127

noncomputable def f (x : ℝ) := x^3 - 3*x

def has_minimum_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ (c : ℝ), a < c ∧ c < b ∧ ∀ (x : ℝ), a < x ∧ x < b → f c ≤ f x

theorem minimum_value_range (a : ℝ) :
  has_minimum_on_interval f a (10 + 2*a^2) ↔ -2 ≤ a ∧ a < 1 :=
sorry

end minimum_value_range_l411_41127


namespace remove_number_for_average_l411_41150

theorem remove_number_for_average (list : List ℕ) (removed : ℕ) (avg : ℚ) : 
  list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] →
  removed = 6 →
  avg = 82/10 →
  (list.sum - removed) / (list.length - 1) = avg := by
  sorry

end remove_number_for_average_l411_41150


namespace range_of_m_l411_41185

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := (m - 2) / (m - 3) ≤ 2/3

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + m^2 > 0

-- Define the set of m values that satisfy the conditions
def M : Set ℝ := {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : M = {m : ℝ | m < -2 ∨ (0 ≤ m ∧ m ≤ 2) ∨ m ≥ 3} := by
  sorry

end range_of_m_l411_41185


namespace no_integer_m_for_single_solution_l411_41124

theorem no_integer_m_for_single_solution :
  ¬ ∃ (m : ℤ), ∃! (x : ℝ), 36 * x^2 - m * x - 4 = 0 := by
  sorry

end no_integer_m_for_single_solution_l411_41124


namespace max_value_complex_l411_41117

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + 3*z + Complex.I*2) ≤ 3 * Real.sqrt 3 := by
  sorry

end max_value_complex_l411_41117


namespace total_pure_acid_in_mixture_l411_41118

/-- Represents a solution with its acid concentration and volume -/
structure Solution where
  concentration : Real
  volume : Real

/-- Calculates the amount of pure acid in a solution -/
def pureAcidAmount (s : Solution) : Real :=
  s.concentration * s.volume

/-- Theorem: The total amount of pure acid in a mixture of solutions is the sum of pure acid amounts from each solution -/
theorem total_pure_acid_in_mixture (solutionA solutionB solutionC : Solution)
  (hA : solutionA.concentration = 0.20 ∧ solutionA.volume = 8)
  (hB : solutionB.concentration = 0.35 ∧ solutionB.volume = 5)
  (hC : solutionC.concentration = 0.15 ∧ solutionC.volume = 3) :
  pureAcidAmount solutionA + pureAcidAmount solutionB + pureAcidAmount solutionC = 3.8 := by
  sorry


end total_pure_acid_in_mixture_l411_41118


namespace complex_magnitude_problem_l411_41179

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 + Complex.I) = 2 - Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l411_41179


namespace sum_of_roots_l411_41199

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 14*p*x - 15*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 14*r*x - 15*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 3150 := by
sorry

end sum_of_roots_l411_41199


namespace restaurant_sales_restaurant_sales_proof_l411_41109

/-- Calculates the total sales of a restaurant given the number of meals sold at different price points. -/
theorem restaurant_sales (meals_at_8 meals_at_10 meals_at_4 : ℕ) 
  (price_8 price_10 price_4 : ℕ) : ℕ :=
  let total_sales := meals_at_8 * price_8 + meals_at_10 * price_10 + meals_at_4 * price_4
  total_sales

/-- Proves that the restaurant's total sales for the day is $210. -/
theorem restaurant_sales_proof :
  restaurant_sales 10 5 20 8 10 4 = 210 := by
  sorry

end restaurant_sales_restaurant_sales_proof_l411_41109


namespace A_intersect_B_equals_two_three_l411_41121

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}

-- Theorem to prove
theorem A_intersect_B_equals_two_three : A ∩ B = {2, 3} := by
  sorry

end A_intersect_B_equals_two_three_l411_41121


namespace pauls_crayons_l411_41181

theorem pauls_crayons (crayons_given : ℕ) (crayons_lost : ℕ) (crayons_left : ℕ)
  (h1 : crayons_given = 563)
  (h2 : crayons_lost = 558)
  (h3 : crayons_left = 332) :
  crayons_given + crayons_lost + crayons_left = 1453 := by
  sorry

end pauls_crayons_l411_41181


namespace chocolate_theorem_l411_41197

-- Define the parameters of the problem
def chocolate_cost : ℕ := 1
def wrappers_per_exchange : ℕ := 3
def initial_money : ℕ := 15

-- Define a function to calculate the maximum number of chocolates
def max_chocolates (cost : ℕ) (exchange_rate : ℕ) (money : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- State the theorem
theorem chocolate_theorem :
  max_chocolates chocolate_cost wrappers_per_exchange initial_money = 22 := by
  sorry

end chocolate_theorem_l411_41197


namespace problem_statement_l411_41106

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem problem_statement :
  (∀ x > 0, f 1 x ≥ 1) ∧
  (∃ x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), f 1 x = 1) ∧
  (∀ a ∈ Set.Icc 0 1, ∃ x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), f a x = 1) ∧
  (∀ a ≥ 1, ∀ x ≥ 1, f a x ≥ f a (1 / x)) ∧
  (∀ a < 1, ∃ x ≥ 1, f a x < f a (1 / x)) := by
  sorry

end problem_statement_l411_41106


namespace ball_distribution_problem_l411_41167

def total_arrangements : ℕ := 90
def arrangements_with_1_and_2_together : ℕ := 18

theorem ball_distribution_problem :
  let n_balls : ℕ := 6
  let n_boxes : ℕ := 3
  let balls_per_box : ℕ := 2
  total_arrangements - arrangements_with_1_and_2_together = 72 :=
by sorry

end ball_distribution_problem_l411_41167


namespace count_multiples_of_7_ending_in_7_less_than_150_l411_41102

def multiples_of_7_ending_in_7 (n : ℕ) : ℕ :=
  (n / 70 : ℕ)

theorem count_multiples_of_7_ending_in_7_less_than_150 :
  multiples_of_7_ending_in_7 150 = 2 := by sorry

end count_multiples_of_7_ending_in_7_less_than_150_l411_41102


namespace wax_needed_l411_41188

theorem wax_needed (current_wax total_wax_required : ℕ) 
  (h1 : current_wax = 11)
  (h2 : total_wax_required = 492) : 
  total_wax_required - current_wax = 481 :=
by sorry

end wax_needed_l411_41188


namespace tangent_line_implies_m_eq_two_l411_41189

/-- A circle defined by parametric equations with parameter m > 0 -/
structure ParametricCircle (m : ℝ) where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ φ, x φ = Real.sqrt m * Real.cos φ
  h_y : ∀ φ, y φ = Real.sqrt m * Real.sin φ
  h_m : m > 0

/-- The line x + y = m is tangent to the circle -/
def isTangent (m : ℝ) (circle : ParametricCircle m) : Prop :=
  ∃ φ, circle.x φ + circle.y φ = m ∧
    ∀ ψ, circle.x ψ + circle.y ψ ≤ m

theorem tangent_line_implies_m_eq_two (m : ℝ) (circle : ParametricCircle m)
    (h_tangent : isTangent m circle) : m = 2 := by
  sorry

#check tangent_line_implies_m_eq_two

end tangent_line_implies_m_eq_two_l411_41189


namespace geometric_series_double_sum_l411_41111

/-- Given two infinite geometric series with the following properties:
    - First series: first term = 20, second term = 5
    - Second series: first term = 20, second term = 5+n
    - Sum of second series is double the sum of first series
    This theorem proves that n = 7.5 -/
theorem geometric_series_double_sum (n : ℝ) : 
  let a₁ : ℝ := 20
  let r₁ : ℝ := 5 / 20
  let r₂ : ℝ := (5 + n) / 20
  let sum₁ : ℝ := a₁ / (1 - r₁)
  let sum₂ : ℝ := a₁ / (1 - r₂)
  sum₂ = 2 * sum₁ → n = 7.5 := by
sorry

end geometric_series_double_sum_l411_41111


namespace xyz_product_magnitude_l411_41112

theorem xyz_product_magnitude (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (heq : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x) : 
  |x * y * z| = 1 := by
sorry

end xyz_product_magnitude_l411_41112


namespace rainfall_difference_l411_41105

-- Define the rainfall amounts for Monday and Tuesday
def monday_rainfall : ℝ := 0.9
def tuesday_rainfall : ℝ := 0.2

-- Theorem to prove the difference in rainfall
theorem rainfall_difference : monday_rainfall - tuesday_rainfall = 0.7 := by
  sorry

end rainfall_difference_l411_41105


namespace triangles_similar_l411_41103

/-- A triangle with side lengths a, b, and c. -/
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

/-- The condition that a + c = 2b for a triangle. -/
def condition1 (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

/-- The condition that b + 2c = 5a for a triangle. -/
def condition2 (t : Triangle) : Prop :=
  t.b + 2 * t.c = 5 * t.a

/-- Two triangles are similar. -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.a = k * t2.a ∧ t1.b = k * t2.b ∧ t1.c = k * t2.c

/-- 
Theorem: If two triangles satisfy both condition1 and condition2, then they are similar.
-/
theorem triangles_similar (t1 t2 : Triangle) 
  (h1 : condition1 t1) (h2 : condition1 t2) 
  (h3 : condition2 t1) (h4 : condition2 t2) : 
  similar t1 t2 := by
  sorry

end triangles_similar_l411_41103


namespace no_solution_to_system_l411_41136

theorem no_solution_to_system :
  ¬ ∃ (x y z : ℝ), 
    (3 * x - 4 * y + z = 10) ∧ 
    (6 * x - 8 * y + 2 * z = 16) ∧ 
    (x + y - z = 3) := by
  sorry

end no_solution_to_system_l411_41136


namespace vanessa_score_in_game_l411_41164

/-- Calculates Vanessa's score in a basketball game -/
def vanessaScore (totalScore : ℕ) (otherPlayersCount : ℕ) (otherPlayersAverage : ℕ) : ℕ :=
  totalScore - (otherPlayersCount * otherPlayersAverage)

/-- Theorem stating Vanessa's score given the game conditions -/
theorem vanessa_score_in_game : 
  vanessaScore 60 7 4 = 32 := by
  sorry

end vanessa_score_in_game_l411_41164


namespace inverse_cube_root_relation_l411_41143

/-- Given that y varies inversely as the cube root of x, prove that when x = 8 and y = 2,
    then x = 1/8 when y = 8 -/
theorem inverse_cube_root_relation (x y : ℝ) (k : ℝ) : 
  (∀ x y, y * (x ^ (1/3 : ℝ)) = k) →  -- y varies inversely as the cube root of x
  (2 * (8 ^ (1/3 : ℝ)) = k) →         -- when x = 8, y = 2
  (8 * (x ^ (1/3 : ℝ)) = k) →         -- when y = 8
  x = 1/8 := by
sorry

end inverse_cube_root_relation_l411_41143


namespace dice_probabilities_l411_41142

/-- Represents the probabilities of an unfair 6-sided dice -/
structure DiceProbabilities where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  sum_one : a + b + c + d + e + f = 1
  all_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f

/-- The probability of rolling the same number twice -/
def P (probs : DiceProbabilities) : ℝ :=
  probs.a^2 + probs.b^2 + probs.c^2 + probs.d^2 + probs.e^2 + probs.f^2

/-- The probability of rolling an odd number first and an even number second -/
def Q (probs : DiceProbabilities) : ℝ :=
  (probs.a + probs.c + probs.e) * (probs.b + probs.d + probs.f)

theorem dice_probabilities (probs : DiceProbabilities) :
  P probs ≥ 1/6 ∧ Q probs ≤ 1/4 ∧ Q probs ≥ 1/2 - 3/2 * P probs := by
  sorry

end dice_probabilities_l411_41142


namespace manufacturing_cost_calculation_l411_41134

/-- The manufacturing cost of a shoe -/
def manufacturing_cost : ℝ := sorry

/-- The transportation cost for 100 shoes -/
def transportation_cost_100 : ℝ := 500

/-- The selling price of a shoe -/
def selling_price : ℝ := 222

/-- The gain percentage on the selling price -/
def gain_percentage : ℝ := 20

theorem manufacturing_cost_calculation : 
  manufacturing_cost = 180 := by
  sorry

end manufacturing_cost_calculation_l411_41134


namespace ninth_term_is_17_l411_41176

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum_property : a 3 + a 4 = 12
  diff_property : ∀ n, a (n + 1) - a n = d
  d_value : d = 2

/-- The 9th term of the arithmetic sequence is 17 -/
theorem ninth_term_is_17 (seq : ArithmeticSequence) : seq.a 9 = 17 := by
  sorry

end ninth_term_is_17_l411_41176


namespace eleven_to_fourth_l411_41135

theorem eleven_to_fourth (n : ℕ) (h : n = 4) : 11^n = 14641 := by
  have h1 : 11 = 10 + 1 := by rfl
  sorry

end eleven_to_fourth_l411_41135


namespace max_xy_value_l411_41101

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + 3 * y = 12) :
  x * y ≤ 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 4 * x + 3 * y = 12 ∧ x * y = 3 :=
sorry

end max_xy_value_l411_41101


namespace smallest_integer_inequality_l411_41116

theorem smallest_integer_inequality (x y z w : ℝ) :
  ∃ (n : ℕ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4) ∧
  ∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m * (a^4 + b^4 + c^4 + d^4) :=
by sorry

end smallest_integer_inequality_l411_41116


namespace quadratic_inequality_solution_range_l411_41113

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 10*x + c < 0) ↔ (0 < c ∧ c < 25) :=
by sorry

end quadratic_inequality_solution_range_l411_41113


namespace inscribed_circle_total_area_l411_41119

/-- The total area of a figure consisting of a circle inscribed in a square, 
    where the circle has a diameter of 6 meters. -/
theorem inscribed_circle_total_area :
  let circle_diameter : ℝ := 6
  let square_side : ℝ := circle_diameter
  let circle_radius : ℝ := circle_diameter / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  let total_area : ℝ := circle_area + square_area
  total_area = 36 + 9 * π := by
  sorry

end inscribed_circle_total_area_l411_41119


namespace pot_contribution_proof_l411_41133

theorem pot_contribution_proof (total_people : Nat) (first_place_percent : Real) 
  (third_place_amount : Real) : 
  total_people = 8 → 
  first_place_percent = 0.8 → 
  third_place_amount = 4 → 
  ∃ (individual_contribution : Real),
    individual_contribution = 5 ∧ 
    individual_contribution * total_people = third_place_amount / ((1 - first_place_percent) / 2) :=
by sorry

end pot_contribution_proof_l411_41133


namespace largest_integer_in_range_l411_41137

theorem largest_integer_in_range : ∃ (x : ℤ), 
  (1 / 4 : ℚ) < (x : ℚ) / 7 ∧ 
  (x : ℚ) / 7 < (2 / 3 : ℚ) ∧ 
  ∀ (y : ℤ), (1 / 4 : ℚ) < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < (2 / 3 : ℚ) → y ≤ x :=
by sorry

end largest_integer_in_range_l411_41137


namespace roots_sum_equation_l411_41198

theorem roots_sum_equation (a b : ℝ) : 
  (a^2 - 4*a + 4 = 0) → 
  (b^2 - 4*b + 4 = 0) → 
  2*(a + b) = 8 := by
sorry

end roots_sum_equation_l411_41198


namespace jo_kate_difference_l411_41168

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

def kate_sum (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_ten |>.sum

theorem jo_kate_difference :
  kate_sum 100 - sum_of_first_n 100 = 500 := by
  sorry

end jo_kate_difference_l411_41168


namespace walter_school_allocation_l411_41175

/-- Represents Walter's work and school allocation details -/
structure WalterFinances where
  days_per_week : ℕ
  hours_per_day : ℕ
  hourly_rate : ℚ
  school_allocation : ℚ

/-- Calculates the fraction of weekly earnings allocated for schooling -/
def school_allocation_fraction (w : WalterFinances) : ℚ :=
  w.school_allocation / (w.days_per_week * w.hours_per_day * w.hourly_rate)

/-- Theorem stating that Walter allocates 3/4 of his weekly earnings for schooling -/
theorem walter_school_allocation :
  let w : WalterFinances := {
    days_per_week := 5,
    hours_per_day := 4,
    hourly_rate := 5,
    school_allocation := 75
  }
  school_allocation_fraction w = 3/4 := by
  sorry

end walter_school_allocation_l411_41175


namespace triangle_properties_l411_41152

noncomputable section

open Real

/-- Given a triangle ABC with D as the midpoint of AB, prove that under certain conditions,
    angle C is π/3 and the maximum value of CD²/(a²+b²) is 3/8. -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (D : ℝ × ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b - c * cos A = a * (sqrt 3 * sin C - 1) →
  sin (A + B) * cos (C - π / 6) = 3 / 4 →
  D = ((cos A + cos B) / 2, (sin A + sin B) / 2) →
  C = π / 3 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x * x + y * y + x * y) / (4 * (x * x + y * y)) ≤ 3 / 8) :=
by sorry

end triangle_properties_l411_41152


namespace gcd_231_154_l411_41157

theorem gcd_231_154 : Nat.gcd 231 154 = 77 := by sorry

end gcd_231_154_l411_41157


namespace tangent_line_to_ellipse_l411_41115

theorem tangent_line_to_ellipse (m : ℝ) :
  (∃! x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 1) →
  m^2 = 35/9 := by
  sorry

end tangent_line_to_ellipse_l411_41115


namespace allowance_calculation_l411_41177

/-- Represents John's weekly allowance in dollars -/
def weekly_allowance : ℝ := 2.10

/-- The fraction of allowance spent at the arcade -/
def arcade_fraction : ℚ := 3/5

/-- The fraction of remaining allowance spent at the toy store -/
def toy_store_fraction : ℚ := 2/7

/-- The fraction of remaining allowance spent at the bookstore -/
def bookstore_fraction : ℚ := 1/3

/-- The amount spent at the candy store in dollars -/
def candy_store_amount : ℝ := 0.40

/-- Theorem stating that given the spending pattern, the initial allowance was $2.10 -/
theorem allowance_calculation (A : ℝ) :
  A * (1 - arcade_fraction) * (1 - toy_store_fraction) * (1 - bookstore_fraction) = candy_store_amount →
  A = weekly_allowance := by
  sorry

#check allowance_calculation

end allowance_calculation_l411_41177


namespace sean_needs_six_packs_l411_41128

/-- The number of light bulbs Sean needs to replace in each room --/
def bulbs_per_room : List Nat := [2, 1, 1, 4]

/-- The number of bulbs per pack --/
def bulbs_per_pack : Nat := 2

/-- The fraction of the total bulbs needed for the garage --/
def garage_fraction : Rat := 1/2

/-- Theorem: Sean needs 6 packs of light bulbs --/
theorem sean_needs_six_packs :
  let total_bulbs := (List.sum bulbs_per_room) + ⌈(List.sum bulbs_per_room : Rat) * garage_fraction⌉
  ⌈(total_bulbs : Rat) / bulbs_per_pack⌉ = 6 := by
  sorry

end sean_needs_six_packs_l411_41128


namespace forest_tree_count_l411_41184

/-- Calculates the total number of trees in a forest given the side length of a square street,
    the ratio of forest area to street area, and the tree density in the forest. -/
theorem forest_tree_count (street_side : ℝ) (forest_street_ratio : ℝ) (trees_per_sqm : ℝ) : 
  street_side = 100 →
  forest_street_ratio = 3 →
  trees_per_sqm = 4 →
  (street_side^2 * forest_street_ratio * trees_per_sqm : ℝ) = 120000 := by
  sorry

end forest_tree_count_l411_41184


namespace factor_sum_l411_41161

theorem factor_sum (P Q : ℝ) : 
  (∃ c d : ℝ, (X^2 - 3*X + 7) * (X^2 + c*X + d) = X^4 + P*X^2 + Q) →
  P + Q = 54 := by
sorry

end factor_sum_l411_41161


namespace partial_fraction_decomposition_l411_41122

theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ),
    ∀ (x : ℝ), x ≠ 4 ∧ x ≠ 3 ∧ x ≠ 5 →
      (x^2 - 5) / ((x - 4) * (x - 3) * (x - 5)) =
      A / (x - 4) + B / (x - 3) + C / (x - 5) ↔
      A = -11 ∧ B = 2 ∧ C = 10 :=
by sorry

end partial_fraction_decomposition_l411_41122


namespace complex_fraction_simplification_l411_41130

theorem complex_fraction_simplification :
  let a := (5 + 4/45) - (4 + 1/6)
  let b := 5 + 8/15
  let c := (4 + 2/3) + 0.75
  let d := 3 + 9/13
  let e := 34 + 2/7
  let f := 0.3
  let g := 0.01
  let h := 70
  (a / b) / (c * d) * e + (f / g) / h + 2/7 = 1 := by sorry

end complex_fraction_simplification_l411_41130


namespace tangent_intersection_points_l411_41141

/-- Given a function f(x) = x^3 - x^2 + ax + 1, prove that the tangent line passing through
    the origin intersects the curve y = f(x) at the points (1, a + 1) and (-1, -a - 1). -/
theorem tangent_intersection_points (a : ℝ) :
  let f := λ x : ℝ => x^3 - x^2 + a*x + 1
  let tangent_line := λ x : ℝ => (a + 1) * x
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -1 ∧
    f x₁ = tangent_line x₁ ∧
    f x₂ = tangent_line x₂ ∧
    (∀ x : ℝ, f x = tangent_line x → x = x₁ ∨ x = x₂) :=
by sorry

end tangent_intersection_points_l411_41141


namespace solution_set_f_inequality_range_of_m_for_nonempty_solution_l411_41171

-- Define the functions f and g
def f (x : ℝ) := |x - 2|
def g (m : ℝ) (x : ℝ) := -|x + 7| + 3 * m

-- Theorem for the first part of the problem
theorem solution_set_f_inequality (x : ℝ) :
  f x + x^2 - 4 > 0 ↔ x > 2 ∨ x < -1 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_m_for_nonempty_solution (m : ℝ) :
  (∃ x : ℝ, f x < g m x) ↔ m > 3 :=
sorry

end solution_set_f_inequality_range_of_m_for_nonempty_solution_l411_41171


namespace remainder_333_power_333_mod_11_l411_41170

theorem remainder_333_power_333_mod_11 : 333^333 % 11 = 5 := by
  sorry

end remainder_333_power_333_mod_11_l411_41170


namespace B_subset_A_iff_m_in_range_l411_41140

-- Define set A
def A : Set ℝ := {x | (2 * x) / (x - 2) < 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + m^2 + m < 0}

-- Theorem statement
theorem B_subset_A_iff_m_in_range :
  ∀ m : ℝ, (B m) ⊆ A ↔ -2 ≤ m ∧ m ≤ 1 := by sorry

end B_subset_A_iff_m_in_range_l411_41140


namespace exam_duration_l411_41151

/-- Represents a time on a clock face -/
structure ClockTime where
  hours : ℝ
  minutes : ℝ
  valid : 0 ≤ hours ∧ hours < 12 ∧ 0 ≤ minutes ∧ minutes < 60

/-- Checks if two clock times are equivalent when hour and minute hands are swapped -/
def equivalent_when_swapped (t1 t2 : ClockTime) : Prop :=
  t1.hours = t2.minutes / 5 ∧ t1.minutes = t2.hours * 5

/-- The main theorem statement -/
theorem exam_duration :
  ∀ (start_time end_time : ClockTime),
    9 ≤ start_time.hours ∧ start_time.hours < 10 →
    1 ≤ end_time.hours ∧ end_time.hours < 2 →
    equivalent_when_swapped start_time end_time →
    end_time.hours - start_time.hours + (end_time.minutes - start_time.minutes) / 60 = 60 / 13 :=
sorry

end exam_duration_l411_41151


namespace swimming_passings_l411_41154

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℝ
  swimmerASpeed : ℝ
  swimmerBSpeed : ℝ
  duration : ℝ

/-- Calculates the number of times swimmers pass each other -/
def calculatePassings (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- Theorem stating the number of passings in the given scenario -/
theorem swimming_passings :
  let scenario : SwimmingScenario := {
    poolLength := 100,
    swimmerASpeed := 4,
    swimmerBSpeed := 5,
    duration := 30 * 60  -- 30 minutes in seconds
  }
  calculatePassings scenario = 54 := by sorry

end swimming_passings_l411_41154


namespace cylinder_surface_area_l411_41169

theorem cylinder_surface_area (r h V : ℝ) : 
  r = 1 → V = 4 * Real.pi → V = Real.pi * r^2 * h → 
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 10 * Real.pi :=
by
  sorry

end cylinder_surface_area_l411_41169


namespace sum_of_powers_of_i_is_zero_l411_41162

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem stating that i^1234 + i^1235 + i^1236 + i^1237 = 0 -/
theorem sum_of_powers_of_i_is_zero : i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end sum_of_powers_of_i_is_zero_l411_41162


namespace log_abs_eq_sin_roots_l411_41110

noncomputable def log_abs (x : ℝ) : ℝ := Real.log (abs x)

theorem log_abs_eq_sin_roots :
  let f (x : ℝ) := log_abs x - Real.sin x
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x = 0) ∧ S.card = 10 ∧ 
    (∀ y : ℝ, f y = 0 → y ∈ S) := by sorry

end log_abs_eq_sin_roots_l411_41110


namespace simplify_power_expression_l411_41159

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by sorry

end simplify_power_expression_l411_41159


namespace sqrt_fourth_power_eq_256_l411_41108

theorem sqrt_fourth_power_eq_256 (x : ℝ) (h : (Real.sqrt x)^4 = 256) : x = 16 := by
  sorry

end sqrt_fourth_power_eq_256_l411_41108


namespace problem_statement_l411_41120

theorem problem_statement (a b : ℝ) (h1 : a^2 + b^2 = 1) :
  (|a - b| / |1 - a*b| ≤ 1) ∧
  (a*b > 0 → (a + b)*(a^3 + b^3) ≥ 1) := by
  sorry

end problem_statement_l411_41120


namespace mary_baking_cake_l411_41153

theorem mary_baking_cake (total_flour sugar_needed : ℕ) 
  (h1 : total_flour = 11)
  (h2 : sugar_needed = 7)
  (h3 : total_flour - flour_put_in = sugar_needed + 2) :
  flour_put_in = 2 :=
by sorry

end mary_baking_cake_l411_41153


namespace tens_digit_of_special_two_digit_number_l411_41148

/-- The product of digits of a two-digit number -/
def P (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- The sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- A two-digit number M satisfying M = P(M) + S(M) + 6 has a tens digit of either 1 or 2 -/
theorem tens_digit_of_special_two_digit_number :
  ∀ M : ℕ, 
    (10 ≤ M ∧ M < 100) →  -- M is a two-digit number
    (M = P M + S M + 6) →  -- M satisfies the special condition
    (M / 10 = 1 ∨ M / 10 = 2) :=  -- The tens digit is either 1 or 2
by sorry

end tens_digit_of_special_two_digit_number_l411_41148


namespace geometric_sequence_problem_l411_41149

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Main theorem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a)
    (h_prod : a 7 * a 9 = 4)
    (h_a4 : a 4 = 1) :
    a 12 = 4 := by
  sorry

end geometric_sequence_problem_l411_41149


namespace perimeter_of_seven_unit_squares_l411_41129

/-- A figure composed of unit squares meeting at vertices -/
structure SquareFigure where
  num_squares : ℕ
  squares_meet_at_vertices : Bool

/-- The perimeter of a square figure -/
def perimeter (f : SquareFigure) : ℕ := 
  if f.squares_meet_at_vertices then
    4 * f.num_squares
  else
    sorry  -- We don't handle this case in this problem

theorem perimeter_of_seven_unit_squares : 
  ∀ (f : SquareFigure), f.num_squares = 7 → f.squares_meet_at_vertices → perimeter f = 28 := by
  sorry

end perimeter_of_seven_unit_squares_l411_41129


namespace joannes_weekly_earnings_is_812_48_l411_41131

/-- Calculates Joanne's weekly earnings after deductions, bonuses, and allowances -/
def joannes_weekly_earnings : ℝ :=
  let main_job_hours : ℝ := 8 * 5
  let main_job_rate : ℝ := 16
  let main_job_base_pay : ℝ := main_job_hours * main_job_rate
  let main_job_bonus_rate : ℝ := 0.1
  let main_job_bonus : ℝ := main_job_base_pay * main_job_bonus_rate
  let main_job_total : ℝ := main_job_base_pay + main_job_bonus
  let main_job_deduction_rate : ℝ := 0.05
  let main_job_deduction : ℝ := main_job_total * main_job_deduction_rate
  let main_job_net : ℝ := main_job_total - main_job_deduction

  let part_time_regular_hours : ℝ := 2 * 4
  let part_time_friday_hours : ℝ := 3
  let part_time_rate : ℝ := 13.5
  let part_time_friday_bonus : ℝ := 2
  let part_time_regular_pay : ℝ := part_time_regular_hours * part_time_rate
  let part_time_friday_pay : ℝ := part_time_friday_hours * (part_time_rate + part_time_friday_bonus)
  let part_time_total : ℝ := part_time_regular_pay + part_time_friday_pay
  let part_time_deduction_rate : ℝ := 0.07
  let part_time_deduction : ℝ := part_time_total * part_time_deduction_rate
  let part_time_net : ℝ := part_time_total - part_time_deduction

  main_job_net + part_time_net

/-- Theorem: Joanne's weekly earnings after deductions, bonuses, and allowances is $812.48 -/
theorem joannes_weekly_earnings_is_812_48 : joannes_weekly_earnings = 812.48 := by
  sorry

end joannes_weekly_earnings_is_812_48_l411_41131


namespace parallel_line_through_circle_center_l411_41193

/-- Given a circle C and a line l1, prove that the line l passing through the center of C and parallel to l1 has the equation 2x - 3y - 8 = 0 -/
theorem parallel_line_through_circle_center 
  (C : (ℝ × ℝ) → Prop)
  (l1 : (ℝ × ℝ) → Prop)
  (hC : C = λ (x, y) => (x - 1)^2 + (y + 2)^2 = 5)
  (hl1 : l1 = λ (x, y) => 2*x - 3*y + 6 = 0) :
  ∃ l : (ℝ × ℝ) → Prop, 
    (∀ p, C p → (p.1 - 1)^2 + (p.2 + 2)^2 = 5) ∧ 
    (∀ p, l1 p → 2*p.1 - 3*p.2 + 6 = 0) ∧
    (l = λ (x, y) => 2*x - 3*y - 8 = 0) ∧
    (∃ c, C c ∧ l c) ∧
    (∀ p q : ℝ × ℝ, l p → l q → l1 ((p.1 + q.1)/2, (p.2 + q.2)/2)) :=
by sorry

end parallel_line_through_circle_center_l411_41193


namespace reciprocal_equation_solution_l411_41192

theorem reciprocal_equation_solution (x : ℝ) : 
  (2 - 1 / (2 - x) = 1 / (2 - x)) → x = 1 := by sorry

end reciprocal_equation_solution_l411_41192


namespace bacteria_increase_l411_41172

/-- Given an original bacteria count of 600 and a current count of 8917,
    prove that the increase in bacteria count is 8317. -/
theorem bacteria_increase (original : ℕ) (current : ℕ) 
  (h1 : original = 600) (h2 : current = 8917) : 
  current - original = 8317 := by
  sorry

end bacteria_increase_l411_41172


namespace circle_and_line_problem_l411_41126

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

-- Define the lines l
def line_l1 (x y : ℝ) : Prop := x = 2
def line_l2 (x y : ℝ) : Prop := 4*x + 3*y = 2

-- Theorem statement
theorem circle_and_line_problem :
  ∃ (center_x center_y : ℝ),
    -- Circle C passes through A(1,3) and B(-1,1)
    circle_C 1 3 ∧ circle_C (-1) 1 ∧
    -- Center of the circle is on the line y = x
    center_y = center_x ∧
    -- Circle equation
    (∀ x y, circle_C x y ↔ (x - center_x)^2 + (y - center_y)^2 = 4) ∧
    -- Line l passes through (2,-2)
    (line_l1 2 (-2) ∨ line_l2 2 (-2)) ∧
    -- Line l intersects circle C with chord length 2√3
    (∃ x1 y1 x2 y2,
      ((line_l1 x1 y1 ∧ line_l1 x2 y2) ∨ (line_l2 x1 y1 ∧ line_l2 x2 y2)) ∧
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = 12) :=
by
  sorry -- Proof omitted

end circle_and_line_problem_l411_41126


namespace number_components_l411_41174

def number : ℕ := 1234000000

theorem number_components : 
  (number / 100000000 = 12) ∧ 
  ((number / 10000000) % 10 = 3) ∧ 
  ((number / 1000000) % 10 = 4) := by
  sorry

end number_components_l411_41174


namespace total_ways_to_draw_l411_41123

/-- Represents the number of cards of each color -/
def cards_per_color : ℕ := 5

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the total number of cards -/
def total_cards : ℕ := cards_per_color * num_colors

/-- Represents the number of cards to be drawn -/
def cards_to_draw : ℕ := 5

/-- Represents the number of ways to draw cards in the (3,1,1) distribution -/
def ways_311 : ℕ := (Nat.choose 3 1) * (Nat.choose cards_per_color 3) * (Nat.choose 2 1) * (Nat.choose 2 1) / 2

/-- Represents the number of ways to draw cards in the (2,2,1) distribution -/
def ways_221 : ℕ := (Nat.choose 3 1) * (Nat.choose cards_per_color 2) * (Nat.choose 2 1) * (Nat.choose 3 2) * (Nat.choose 1 1) / 2

/-- The main theorem stating the total number of ways to draw the cards -/
theorem total_ways_to_draw : ways_311 + ways_221 = 150 := by
  sorry

end total_ways_to_draw_l411_41123


namespace cloth_cost_price_l411_41194

theorem cloth_cost_price (total_length : ℕ) (first_part : ℕ) (remaining_part : ℕ)
  (total_price : ℕ) (profit1 : ℕ) (profit2 : ℕ) (cost_price : ℕ) :
  total_length = first_part + remaining_part →
  total_length = 85 →
  first_part = 50 →
  remaining_part = 35 →
  total_price = 8925 →
  profit1 = 15 →
  profit2 = 20 →
  first_part * (cost_price + profit1) + remaining_part * (cost_price + profit2) = total_price →
  cost_price = 88 := by
  sorry

end cloth_cost_price_l411_41194


namespace rectangle_area_l411_41158

theorem rectangle_area (length width : ℝ) :
  (2 * (length + width) = 48) →
  (length = width + 2) →
  (length * width = 143) :=
by
  sorry

end rectangle_area_l411_41158


namespace exists_triangle_area_not_greater_than_two_l411_41146

-- Define a lattice point type
structure LatticePoint where
  x : Int
  y : Int

-- Define the condition for a lattice point to be within the 5x5 grid
def isWithinGrid (p : LatticePoint) : Prop :=
  abs p.x ≤ 2 ∧ abs p.y ≤ 2

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  let x1 := p1.x
  let y1 := p1.y
  let x2 := p2.x
  let y2 := p2.y
  let x3 := p3.x
  let y3 := p3.y
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

-- Define the condition for three points to be non-collinear
def nonCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  triangleArea p1 p2 p3 ≠ 0

-- Main theorem
theorem exists_triangle_area_not_greater_than_two 
  (points : Fin 6 → LatticePoint)
  (h_within_grid : ∀ i, isWithinGrid (points i))
  (h_non_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → nonCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ triangleArea (points i) (points j) (points k) ≤ 2 :=
sorry

end exists_triangle_area_not_greater_than_two_l411_41146


namespace shop_profit_days_l411_41145

theorem shop_profit_days (mean_profit : ℝ) (first_15_mean : ℝ) (last_15_mean : ℝ)
  (h1 : mean_profit = 350)
  (h2 : first_15_mean = 245)
  (h3 : last_15_mean = 455) :
  (mean_profit * (15 + 15) = first_15_mean * 15 + last_15_mean * 15) → (15 + 15 = 30) :=
by sorry

end shop_profit_days_l411_41145


namespace area_of_enclosed_region_l411_41114

/-- The equation of the curve enclosing the region -/
def curve_equation (x y : ℝ) : Prop :=
  x^2 - 18*x + 3*y + 90 = 33 + 9*y - y^2

/-- The equation of the line bounding the region above -/
def line_equation (x y : ℝ) : Prop :=
  y = x - 5

/-- The region enclosed by the curve and below the line -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve_equation p.1 p.2 ∧ p.2 ≤ p.1 - 5}

/-- The area of the enclosed region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_enclosed_region :
  area_of_region = 33 * Real.pi / 2 := by sorry

end area_of_enclosed_region_l411_41114


namespace symmetric_points_range_l411_41104

open Set
open Function
open Real

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := a * x^2 - a * x

theorem symmetric_points_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    f x₁ = g a x₁ ∧ 
    f x₂ = g a x₂ ∧ 
    x₁ = f x₂ ∧ 
    x₂ = f x₁) →
  a ∈ (Ioo 0 1 ∪ Ioi 1) :=
sorry

end symmetric_points_range_l411_41104


namespace number_of_tenths_l411_41166

theorem number_of_tenths (n : ℚ) : (375 : ℚ) * (1 / 10 : ℚ) = n → n = (37.5 : ℚ) := by
  sorry

end number_of_tenths_l411_41166


namespace ferris_wheel_capacity_l411_41139

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 14

/-- The total number of people the Ferris wheel can hold -/
def total_people : ℕ := 84

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 6 := by
  sorry

end ferris_wheel_capacity_l411_41139


namespace noon_temperature_l411_41163

theorem noon_temperature 
  (morning_temp : ℤ) 
  (temp_drop : ℤ) 
  (h1 : morning_temp = 3) 
  (h2 : temp_drop = 9) : 
  morning_temp - temp_drop = -6 := by
sorry

end noon_temperature_l411_41163


namespace fraction_zero_implies_x_negative_one_l411_41191

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (abs x - 1) / (x - 1) = 0 → x = -1 :=
by
  sorry

end fraction_zero_implies_x_negative_one_l411_41191


namespace find_p_l411_41132

def U : Set ℕ := {1, 2, 3, 4}

def M (p : ℝ) : Set ℕ := {x ∈ U | x^2 - 5*x + p = 0}

theorem find_p : ∃ p : ℝ, (U \ M p) = {2, 3} → p = 4 := by
  sorry

end find_p_l411_41132


namespace function_shift_l411_41165

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_shift (h : f 0 = 2) : f (-1 + 1) = 2 := by
  sorry

end function_shift_l411_41165


namespace polynomial_division_remainder_l411_41183

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℚ, 
    (3 * X^5 + 16 * X^4 - 17 * X^3 - 100 * X^2 + 32 * X + 90 : Polynomial ℚ) = 
    (X^3 + 8 * X^2 - X - 6) * q + (422 * X^2 + 48 * X - 294) := by
  sorry

end polynomial_division_remainder_l411_41183


namespace apple_juice_problem_l411_41178

theorem apple_juice_problem (initial_amount : ℚ) (maria_fraction : ℚ) (john_fraction : ℚ) :
  initial_amount = 3/4 →
  maria_fraction = 1/2 →
  john_fraction = 1/3 →
  let remaining_after_maria := initial_amount - (maria_fraction * initial_amount)
  john_fraction * remaining_after_maria = 1/8 := by
  sorry

end apple_juice_problem_l411_41178


namespace unique_solution_exists_l411_41147

/-- Represents a digit from 1 to 6 -/
def Digit := Fin 6

/-- Represents a two-digit number composed of two digits -/
def TwoDigitNumber (a b : Digit) : ℕ := (a.val + 1) * 10 + (b.val + 1)

/-- The main theorem stating the existence and uniqueness of the solution -/
theorem unique_solution_exists :
  ∃! (A B C D E F : Digit),
    (TwoDigitNumber A B) ^ (C.val + 1) = (TwoDigitNumber D E) ^ (F.val + 1) ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F :=
by sorry

end unique_solution_exists_l411_41147


namespace dog_food_bag_weight_l411_41190

/-- Proves that the weight of each bag of dog food is 20 pounds -/
theorem dog_food_bag_weight :
  let cup_weight : ℚ := 1/4  -- Weight of a cup of dog food in pounds
  let num_dogs : ℕ := 2  -- Number of dogs
  let cups_per_meal : ℕ := 6  -- Cups of food per meal per dog
  let meals_per_day : ℕ := 2  -- Number of meals per day
  let bags_per_month : ℕ := 9  -- Number of bags bought per month
  let days_per_month : ℕ := 30  -- Number of days in a month
  
  let daily_consumption : ℚ := num_dogs * cups_per_meal * meals_per_day * cup_weight
  let monthly_consumption : ℚ := daily_consumption * days_per_month
  let bag_weight : ℚ := monthly_consumption / bags_per_month

  bag_weight = 20 := by
    sorry

end dog_food_bag_weight_l411_41190


namespace quiz_result_proof_l411_41144

theorem quiz_result_proof (total : ℕ) (correct_A : ℕ) (correct_B : ℕ) (correct_C : ℕ) 
  (all_wrong : ℕ) (all_correct : ℕ) 
  (h_total : total = 40)
  (h_A : correct_A = 10)
  (h_B : correct_B = 13)
  (h_C : correct_C = 15)
  (h_wrong : all_wrong = 15)
  (h_correct : all_correct = 1) :
  ∃ (two_correct : ℕ), two_correct = 13 ∧ 
  two_correct = total - all_wrong - all_correct - 
    (correct_A + correct_B + correct_C - 2 * all_correct - two_correct) := by
  sorry

end quiz_result_proof_l411_41144


namespace total_celestial_bodies_count_l411_41187

/-- A galaxy with specific ratios of celestial bodies -/
structure Galaxy where
  planets : ℕ
  solar_systems : ℕ
  stars : ℕ
  solar_system_planet_ratio : solar_systems = 8 * planets
  star_solar_system_ratio : stars = 4 * solar_systems
  planet_count : planets = 20

/-- The total number of celestial bodies in the galaxy -/
def total_celestial_bodies (g : Galaxy) : ℕ :=
  g.planets + g.solar_systems + g.stars

/-- Theorem stating that the total number of celestial bodies is 820 -/
theorem total_celestial_bodies_count (g : Galaxy) :
  total_celestial_bodies g = 820 := by
  sorry

end total_celestial_bodies_count_l411_41187


namespace discount_approximation_l411_41182

/-- Calculates the discount given cost price, markup percentage, and profit percentage -/
def calculate_discount (cost_price : ℝ) (markup_percentage : ℝ) (profit_percentage : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := cost_price * (1 + profit_percentage)
  marked_price - selling_price

/-- Theorem stating that the discount is approximately 50 given the problem conditions -/
theorem discount_approximation :
  let cost_price : ℝ := 180
  let markup_percentage : ℝ := 0.4778
  let profit_percentage : ℝ := 0.20
  let discount := calculate_discount cost_price markup_percentage profit_percentage
  ∃ ε > 0, |discount - 50| < ε :=
sorry

end discount_approximation_l411_41182


namespace hash_triple_100_l411_41180

def hash (N : ℝ) : ℝ := 0.5 * N - 2

theorem hash_triple_100 : hash (hash (hash 100)) = 9 := by
  sorry

end hash_triple_100_l411_41180


namespace sum_of_prime_and_odd_l411_41155

theorem sum_of_prime_and_odd (a b : ℕ) : 
  Nat.Prime a → Odd b → a^2 + b = 2009 → a + b = 2007 := by
  sorry

end sum_of_prime_and_odd_l411_41155


namespace solution_set_of_inequality_l411_41156

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 4*x

theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_def : f_definition f) :
  {x : ℝ | f (x + 2) < 5} = Set.Ioo (-7) 3 := by
  sorry

end solution_set_of_inequality_l411_41156


namespace show_completion_time_l411_41100

theorem show_completion_time (num_episodes : ℕ) (episode_length : ℕ) (daily_watch_time : ℕ) : 
  num_episodes = 20 → 
  episode_length = 30 → 
  daily_watch_time = 120 → 
  (num_episodes * episode_length) / daily_watch_time = 5 :=
by sorry

end show_completion_time_l411_41100


namespace max_m_quadratic_inequality_l411_41138

theorem max_m_quadratic_inequality (a b c : ℝ) (h_real_roots : b^2 - 4*a*c ≥ 0) :
  ∃ (m : ℝ), m = 9/8 ∧ 
  (∀ (k : ℝ), ((a-b)^2 + (b-c)^2 + (c-a)^2 ≥ k*a^2) → k ≤ m) ∧
  ((a-b)^2 + (b-c)^2 + (c-a)^2 ≥ m*a^2) := by
  sorry

end max_m_quadratic_inequality_l411_41138


namespace problem_solution_l411_41173

theorem problem_solution :
  ∀ (x y z : ℝ),
  (x + x = y * x) →
  (x + x = z * z) →
  (y = 3) →
  (x * z = 4) :=
by
  sorry

end problem_solution_l411_41173


namespace M_intersect_N_eq_open_zero_one_closed_l411_41195

def M : Set ℝ := {x | 0 < Real.log (x + 1) ∧ Real.log (x + 1) < 3}

def N : Set ℝ := {y | ∃ x ∈ M, y = Real.sin x}

theorem M_intersect_N_eq_open_zero_one_closed : M ∩ N = Set.Ioc 0 1 := by sorry

end M_intersect_N_eq_open_zero_one_closed_l411_41195
