import Mathlib

namespace pamelas_initial_skittles_l3268_326863

/-- The number of Skittles Pamela gave away -/
def skittles_given : ℕ := 7

/-- The number of Skittles Pamela has now -/
def skittles_remaining : ℕ := 43

/-- Pamela's initial number of Skittles -/
def initial_skittles : ℕ := skittles_given + skittles_remaining

theorem pamelas_initial_skittles : initial_skittles = 50 := by
  sorry

end pamelas_initial_skittles_l3268_326863


namespace no_rational_roots_l3268_326805

/-- The polynomial we're investigating -/
def p (x : ℚ) : ℚ := 3 * x^4 + 2 * x^3 - 8 * x^2 - x + 1

/-- Theorem stating that the polynomial has no rational roots -/
theorem no_rational_roots : ∀ x : ℚ, p x ≠ 0 := by
  sorry

end no_rational_roots_l3268_326805


namespace trapezoid_ab_length_l3268_326869

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- The ratio of the areas of triangles ABC and ADC
  area_ratio : ℚ
  -- The combined length of bases AB and CD
  total_base_length : ℝ
  -- The length of base AB
  ab_length : ℝ

/-- Theorem: If the area ratio is 8:2 and the total base length is 120,
    then the length of AB is 96 -/
theorem trapezoid_ab_length (t : Trapezoid) :
  t.area_ratio = 8 / 2 ∧ t.total_base_length = 120 → t.ab_length = 96 := by
  sorry

end trapezoid_ab_length_l3268_326869


namespace scientific_notation_equivalence_l3268_326829

theorem scientific_notation_equivalence : 0.0000036 = 3.6 * 10^(-6) := by
  sorry

end scientific_notation_equivalence_l3268_326829


namespace mart_vegetable_count_l3268_326806

/-- The number of cucumbers in the mart -/
def cucumbers : ℕ := 58

/-- The number of carrots in the mart -/
def carrots : ℕ := cucumbers - 24

/-- The number of tomatoes in the mart -/
def tomatoes : ℕ := cucumbers + 49

/-- The number of radishes in the mart -/
def radishes : ℕ := carrots

/-- The total number of vegetables in the mart -/
def total_vegetables : ℕ := cucumbers + carrots + tomatoes + radishes

/-- Theorem stating the total number of vegetables in the mart -/
theorem mart_vegetable_count : total_vegetables = 233 := by sorry

end mart_vegetable_count_l3268_326806


namespace remainder_8347_mod_9_l3268_326849

theorem remainder_8347_mod_9 : 8347 % 9 = 4 := by
  sorry

end remainder_8347_mod_9_l3268_326849


namespace sum_of_squares_perfect_square_two_even_l3268_326833

theorem sum_of_squares_perfect_square_two_even (x y z : ℤ) :
  ∃ (u : ℤ), x^2 + y^2 + z^2 = u^2 →
  (Even x ∧ Even y) ∨ (Even x ∧ Even z) ∨ (Even y ∧ Even z) :=
sorry

end sum_of_squares_perfect_square_two_even_l3268_326833


namespace slope_negative_one_implies_y_coordinate_l3268_326834

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is -1, then the y-coordinate of Q is -3. -/
theorem slope_negative_one_implies_y_coordinate (x1 y1 x2 y2 : ℝ) :
  x1 = -3 →
  y1 = 5 →
  x2 = 5 →
  (y2 - y1) / (x2 - x1) = -1 →
  y2 = -3 := by
sorry

end slope_negative_one_implies_y_coordinate_l3268_326834


namespace album_ratio_l3268_326817

theorem album_ratio (adele katrina bridget miriam : ℕ) 
  (h1 : ∃ s : ℕ, miriam = s * katrina)
  (h2 : katrina = 6 * bridget)
  (h3 : bridget = adele - 15)
  (h4 : adele = 30)
  (h5 : miriam + katrina + bridget + adele = 585) :
  miriam = 5 * katrina := by
sorry

end album_ratio_l3268_326817


namespace temperature_difference_l3268_326820

theorem temperature_difference (lowest highest : ℤ) 
  (h_lowest : lowest = -11)
  (h_highest : highest = -3) :
  highest - lowest = 8 := by
sorry

end temperature_difference_l3268_326820


namespace f_intersects_iff_m_le_one_l3268_326827

/-- The quadratic function f(x) = mx^2 + (m-3)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 3) * x + 1

/-- The condition that f intersects the x-axis with at least one point to the right of the origin -/
def intersects_positive_x (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f m x = 0

theorem f_intersects_iff_m_le_one :
  ∀ m : ℝ, intersects_positive_x m ↔ m ≤ 1 :=
sorry

end f_intersects_iff_m_le_one_l3268_326827


namespace circle_diameter_from_area_l3268_326826

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 64 * Real.pi → d = 16 → A = Real.pi * (d / 2)^2 := by
  sorry

end circle_diameter_from_area_l3268_326826


namespace perpendicular_vectors_l3268_326857

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors (x : ℝ) :
  dot_product (λ i => a i + b x i) (λ i => a i - b x i) = 0 → x = 1 ∨ x = -1 := by
  sorry

end perpendicular_vectors_l3268_326857


namespace max_at_two_implies_a_geq_neg_half_l3268_326824

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * (a + 1) * x - 3

-- State the theorem
theorem max_at_two_implies_a_geq_neg_half (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ f a 2) →
  a ≥ -1/2 :=
by sorry

end max_at_two_implies_a_geq_neg_half_l3268_326824


namespace julia_bought_399_balls_l3268_326809

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Theorem stating that Julia bought 399 balls in total -/
theorem julia_bought_399_balls :
  total_balls 3 10 8 19 = 399 := by
  sorry

end julia_bought_399_balls_l3268_326809


namespace opposite_of_seven_l3268_326838

theorem opposite_of_seven :
  ∀ x : ℤ, (7 + x = 0) → x = -7 := by
  sorry

end opposite_of_seven_l3268_326838


namespace parabola_equation_l3268_326873

-- Define the parabola
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => x^2 = 2 * p * y

-- Define points on the parabola
def Point := ℝ × ℝ

-- Define the problem setup
structure ParabolaProblem where
  parabola : Parabola
  F : Point
  A : Point
  B : Point
  C : Point
  D : Point
  l : Point → Prop

-- Define the conditions
def satisfies_conditions (prob : ParabolaProblem) : Prop :=
  let (xf, yf) := prob.F
  let (xa, ya) := prob.A
  let (xb, yb) := prob.B
  let (xc, yc) := prob.C
  let (xd, yd) := prob.D
  xf = 0 ∧ yf > 0 ∧
  prob.parabola.eq xa ya ∧
  prob.parabola.eq xb yb ∧
  prob.l prob.A ∧ prob.l prob.B ∧
  xc = xa ∧ xd = xb ∧
  (ya - yf)^2 + xa^2 = 4 * ((yf - yb)^2 + xb^2) ∧
  (xd - xc) * (xa - xb) + (yd - yc) * (ya - yb) = 72

-- Theorem statement
theorem parabola_equation (prob : ParabolaProblem) 
  (h : satisfies_conditions prob) : 
  prob.parabola.p = 4 := by sorry

end parabola_equation_l3268_326873


namespace olivias_wallet_after_supermarket_l3268_326886

/-- The amount left in Olivia's wallet after visiting the supermarket -/
def money_left (initial_amount spent : ℕ) : ℕ :=
  initial_amount - spent

theorem olivias_wallet_after_supermarket :
  money_left 94 16 = 78 := by
  sorry

end olivias_wallet_after_supermarket_l3268_326886


namespace middle_number_is_four_l3268_326855

theorem middle_number_is_four (a b c : ℕ) : 
  a < b ∧ b < c  -- numbers are in increasing order
  → a + b + c = 15  -- numbers sum to 15
  → a ≠ b ∧ b ≠ c ∧ a ≠ c  -- numbers are all different
  → a > 0 ∧ b > 0 ∧ c > 0  -- numbers are positive
  → (∀ x y, x < y ∧ x + y < 15 → ∃ z, x < z ∧ z < y ∧ x + z + y = 15)  -- leftmost doesn't uniquely determine
  → (∀ x y, x < y ∧ x + y > 0 → ∃ z, z < x ∧ x < y ∧ z + x + y = 15)  -- rightmost doesn't uniquely determine
  → b = 4  -- middle number is 4
  := by sorry

end middle_number_is_four_l3268_326855


namespace spencer_jumps_per_minute_l3268_326845

-- Define the parameters
def minutes_per_session : ℕ := 10
def sessions_per_day : ℕ := 2
def total_jumps : ℕ := 400
def total_days : ℕ := 5

-- Theorem to prove
theorem spencer_jumps_per_minute :
  (total_jumps : ℚ) / ((minutes_per_session * sessions_per_day * total_days) : ℚ) = 4 := by
  sorry

end spencer_jumps_per_minute_l3268_326845


namespace alternating_squares_sum_l3268_326880

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end alternating_squares_sum_l3268_326880


namespace trig_identity_proof_l3268_326884

theorem trig_identity_proof : 
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end trig_identity_proof_l3268_326884


namespace quadratic_equation_range_l3268_326803

theorem quadratic_equation_range (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x, 2 * x^2 - 2 * x + 3 * m - 1 = 0) →
  (x₁ * x₂ > x₁ + x₂ - 4) →
  (-5/3 < m ∧ m ≤ 1/2) :=
by sorry

end quadratic_equation_range_l3268_326803


namespace unique_solution_modular_equation_l3268_326802

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 107 ∧ (103 * n) % 107 = 56 % 107 ∧ n = 85 := by
  sorry

end unique_solution_modular_equation_l3268_326802


namespace triple_value_equation_l3268_326871

theorem triple_value_equation (x : ℝ) : 3 * x^2 + 15 = 3 * (2 * x + 20) → x = 5 ∨ x = -3 := by
  sorry

end triple_value_equation_l3268_326871


namespace ball_max_height_l3268_326822

/-- The height function of the ball -/
def height_function (t : ℝ) : ℝ := 180 * t - 20 * t^2

/-- The maximum height reached by the ball -/
def max_height : ℝ := 405

theorem ball_max_height : 
  ∃ t : ℝ, height_function t = max_height ∧ 
  ∀ u : ℝ, height_function u ≤ max_height :=
sorry

end ball_max_height_l3268_326822


namespace toilet_paper_cost_l3268_326891

/-- Prove that the cost of one roll of toilet paper is $1.50 -/
theorem toilet_paper_cost 
  (total_toilet_paper : ℕ) 
  (total_paper_towels : ℕ) 
  (total_tissues : ℕ) 
  (total_cost : ℚ) 
  (paper_towel_cost : ℚ) 
  (tissue_cost : ℚ) 
  (h1 : total_toilet_paper = 10)
  (h2 : total_paper_towels = 7)
  (h3 : total_tissues = 3)
  (h4 : total_cost = 35)
  (h5 : paper_towel_cost = 2)
  (h6 : tissue_cost = 2) :
  (total_cost - (total_paper_towels * paper_towel_cost + total_tissues * tissue_cost)) / total_toilet_paper = (3 / 2 : ℚ) :=
by sorry

end toilet_paper_cost_l3268_326891


namespace quadratic_function_zero_equivalence_l3268_326851

/-- A quadratic function f(x) = ax² + bx + c where a ≠ 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function value for a given x -/
def QuadraticFunction.value (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The set of zeros of the function -/
def QuadraticFunction.zeros (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.value x = 0}

/-- The composition of the function with itself -/
def QuadraticFunction.compose_self (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.value (f.value x)

theorem quadratic_function_zero_equivalence (f : QuadraticFunction) :
  (f.zeros = {x : ℝ | f.compose_self x = 0}) ↔ f.c = 0 := by
  sorry

end quadratic_function_zero_equivalence_l3268_326851


namespace total_weight_AlF3_is_839_8_l3268_326889

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of aluminum atoms in AlF3 -/
def num_Al : ℕ := 1

/-- The number of fluorine atoms in AlF3 -/
def num_F : ℕ := 3

/-- The number of moles of AlF3 -/
def num_moles : ℝ := 10

/-- The molecular weight of AlF3 in g/mol -/
def molecular_weight_AlF3 : ℝ := atomic_weight_Al * num_Al + atomic_weight_F * num_F

/-- The total weight of AlF3 in grams -/
def total_weight_AlF3 : ℝ := molecular_weight_AlF3 * num_moles

theorem total_weight_AlF3_is_839_8 : total_weight_AlF3 = 839.8 := by
  sorry

end total_weight_AlF3_is_839_8_l3268_326889


namespace smallest_3digit_base6_divisible_by_7_l3268_326810

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (n : ℕ) : ℕ :=
  sorry

/-- Converts a decimal number to base 6 --/
def decimalToBase6 (n : ℕ) : ℕ :=
  sorry

/-- Checks if a number is a 3-digit base 6 number --/
def isThreeDigitBase6 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem smallest_3digit_base6_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase6 n ∧ 
             base6ToDecimal n % 7 = 0 ∧
             decimalToBase6 (base6ToDecimal n) = 110 ∧
             ∀ (m : ℕ), isThreeDigitBase6 m ∧ base6ToDecimal m % 7 = 0 → base6ToDecimal n ≤ base6ToDecimal m :=
by sorry

end smallest_3digit_base6_divisible_by_7_l3268_326810


namespace right_triangle_hypotenuse_l3268_326813

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 18.8 := by
sorry

end right_triangle_hypotenuse_l3268_326813


namespace square_difference_of_roots_l3268_326812

theorem square_difference_of_roots (α β : ℝ) : 
  α ≠ β ∧ α^2 - 3*α + 1 = 0 ∧ β^2 - 3*β + 1 = 0 → (α - β)^2 = 5 := by
  sorry

end square_difference_of_roots_l3268_326812


namespace x_equals_y_l3268_326895

theorem x_equals_y (x y : ℝ) : x = 2 + Real.sqrt 3 → y = 1 / (2 - Real.sqrt 3) → x = y := by
  sorry

end x_equals_y_l3268_326895


namespace area_of_bounded_region_l3268_326888

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 50*abs x = 500

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ graph_equation x y}

-- State the theorem
theorem area_of_bounded_region :
  MeasureTheory.volume bounded_region = 1250 := by sorry

end area_of_bounded_region_l3268_326888


namespace min_selection_for_sum_multiple_of_10_l3268_326876

/-- The set of numbers from 11 to 30 -/
def S : Set ℕ := {n | 11 ≤ n ∧ n ≤ 30}

/-- A function that checks if the sum of two numbers is a multiple of 10 -/
def sumIsMultipleOf10 (a b : ℕ) : Prop := (a + b) % 10 = 0

/-- The theorem stating the minimum number of integers to be selected -/
theorem min_selection_for_sum_multiple_of_10 :
  ∃ (k : ℕ), k = 11 ∧
  (∀ (T : Set ℕ), T ⊆ S → T.ncard ≥ k →
    ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ sumIsMultipleOf10 a b) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (T : Set ℕ), T ⊆ S ∧ T.ncard = k' ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬(sumIsMultipleOf10 a b)) :=
sorry

end min_selection_for_sum_multiple_of_10_l3268_326876


namespace prob_at_least_one_white_l3268_326892

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least one white ball when randomly selecting 3 balls from a bag 
    containing 5 white balls and 4 red balls -/
theorem prob_at_least_one_white : 
  (1 : ℚ) - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 20 / 21 :=
sorry

end prob_at_least_one_white_l3268_326892


namespace calculate_new_interest_rate_l3268_326866

/-- Given a principal amount and interest rates, proves the new interest rate -/
theorem calculate_new_interest_rate
  (P : ℝ)
  (h1 : P * 0.045 = 405)
  (h2 : P * 0.05 = 450) :
  0.05 = (405 + 45) / P :=
by sorry

end calculate_new_interest_rate_l3268_326866


namespace felix_tree_chopping_l3268_326881

/-- Given that Felix needs to resharpen his axe every 13 trees, each sharpening costs $5,
    and he has spent $35 on sharpening, prove that he has chopped down at least 91 trees. -/
theorem felix_tree_chopping (trees_per_sharpening : ℕ) (cost_per_sharpening : ℕ) (total_spent : ℕ) 
    (h1 : trees_per_sharpening = 13)
    (h2 : cost_per_sharpening = 5)
    (h3 : total_spent = 35) :
  trees_per_sharpening * (total_spent / cost_per_sharpening) ≥ 91 := by
  sorry

#check felix_tree_chopping

end felix_tree_chopping_l3268_326881


namespace product_expansion_sum_l3268_326893

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 14 := by
sorry

end product_expansion_sum_l3268_326893


namespace protest_jail_time_protest_jail_time_result_l3268_326832

/-- Calculate the total combined weeks of jail time given protest and arrest conditions -/
theorem protest_jail_time (days_of_protest : ℕ) (num_cities : ℕ) (arrests_per_day : ℕ) 
  (pre_trial_days : ℕ) (sentence_weeks : ℕ) : ℕ :=
  let total_arrests := days_of_protest * num_cities * arrests_per_day
  let jail_days_per_person := pre_trial_days + (sentence_weeks / 2) * 7
  let total_jail_days := total_arrests * jail_days_per_person
  total_jail_days / 7

/-- The total combined weeks of jail time is 9900 weeks -/
theorem protest_jail_time_result : 
  protest_jail_time 30 21 10 4 2 = 9900 := by sorry

end protest_jail_time_protest_jail_time_result_l3268_326832


namespace poly_has_four_nonzero_terms_l3268_326859

/-- The polynomial expression -/
def poly (x : ℝ) : ℝ := (2*x + 5)*(3*x^2 - x + 4) + 4*(x^3 + x^2 - 6*x)

/-- The expansion of the polynomial -/
def expanded_poly (x : ℝ) : ℝ := 10*x^3 + 17*x^2 - 21*x + 20

/-- Theorem stating that the polynomial has exactly 4 nonzero terms -/
theorem poly_has_four_nonzero_terms :
  ∃ (a b c d : ℝ) (n : ℕ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (∀ x, poly x = expanded_poly x) ∧
    (∀ x, expanded_poly x = a*x^3 + b*x^2 + c*x + d) ∧
    n = 4 := by sorry

end poly_has_four_nonzero_terms_l3268_326859


namespace negative_quartic_count_l3268_326890

theorem negative_quartic_count :
  (∃! (s : Finset ℤ), (∀ x ∈ s, x^4 - 51*x^2 + 100 < 0) ∧ s.card = 10) := by
  sorry

end negative_quartic_count_l3268_326890


namespace children_to_women_ratio_l3268_326885

theorem children_to_women_ratio 
  (total_spectators : ℕ) 
  (men_spectators : ℕ) 
  (children_spectators : ℕ) 
  (h1 : total_spectators = 10000)
  (h2 : men_spectators = 7000)
  (h3 : children_spectators = 2500) :
  (children_spectators : ℚ) / (total_spectators - men_spectators - children_spectators) = 5 / 1 := by
  sorry

end children_to_women_ratio_l3268_326885


namespace principal_calculation_l3268_326816

/-- Proves that given specific conditions, the principal amount is 1400 --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2.4 →
  amount = 1568 →
  (∃ (principal : ℝ), principal * (1 + rate * time) = amount ∧ principal = 1400) :=
by sorry

end principal_calculation_l3268_326816


namespace roots_quadratic_equation_l3268_326804

theorem roots_quadratic_equation (α β : ℝ) : 
  (α^2 - 3*α - 2 = 0) → 
  (β^2 - 3*β - 2 = 0) → 
  7*α^4 + 10*β^3 = 544 := by
sorry

end roots_quadratic_equation_l3268_326804


namespace max_value_of_sine_function_l3268_326887

theorem max_value_of_sine_function (x : ℝ) (h : -π/2 ≤ x ∧ x ≤ 0) :
  ∃ (y : ℝ), y = 3 * Real.sin x + 2 ∧ y ≤ 2 ∧ ∃ (x₀ : ℝ), -π/2 ≤ x₀ ∧ x₀ ≤ 0 ∧ 3 * Real.sin x₀ + 2 = 2 :=
by
  sorry

end max_value_of_sine_function_l3268_326887


namespace fundraiser_final_day_amount_l3268_326839

def fundraiser (goal : ℕ) (bronze_donation silver_donation gold_donation : ℕ) 
  (bronze_families silver_families gold_families : ℕ) : ℕ :=
  let total_raised := bronze_donation * bronze_families + 
                      silver_donation * silver_families + 
                      gold_donation * gold_families
  goal - total_raised

theorem fundraiser_final_day_amount : 
  fundraiser 750 25 50 100 10 7 1 = 50 := by
  sorry

end fundraiser_final_day_amount_l3268_326839


namespace intersection_point_implies_n_equals_two_l3268_326856

theorem intersection_point_implies_n_equals_two (n : ℕ+) 
  (x y : ℤ) -- x and y are integers
  (h1 : 15 * x + 18 * y = 1005) -- First line equation
  (h2 : y = n * x + 2) -- Second line equation
  : n = 2 := by
  sorry

end intersection_point_implies_n_equals_two_l3268_326856


namespace sum_of_largest_and_smallest_prime_factors_of_1155_l3268_326842

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

theorem sum_of_largest_and_smallest_prime_factors_of_1155 :
  ∃ (smallest largest : ℕ),
    is_prime_factor smallest 1155 ∧
    is_prime_factor largest 1155 ∧
    (∀ p, is_prime_factor p 1155 → smallest ≤ p) ∧
    (∀ p, is_prime_factor p 1155 → p ≤ largest) ∧
    smallest + largest = 14 :=
sorry

end sum_of_largest_and_smallest_prime_factors_of_1155_l3268_326842


namespace play_area_size_l3268_326823

/-- Represents the configuration of a rectangular play area with fence posts -/
structure PlayArea where
  total_posts : ℕ
  post_spacing : ℕ
  shorter_side_posts : ℕ
  longer_side_posts : ℕ

/-- Calculates the area of the play area given its configuration -/
def calculate_area (pa : PlayArea) : ℕ :=
  (pa.shorter_side_posts - 1) * pa.post_spacing * ((pa.longer_side_posts - 1) * pa.post_spacing)

/-- Theorem stating that the play area with given specifications has an area of 324 square yards -/
theorem play_area_size (pa : PlayArea) :
  pa.total_posts = 24 ∧
  pa.post_spacing = 3 ∧
  pa.longer_side_posts = 2 * pa.shorter_side_posts ∧
  pa.total_posts = 2 * pa.shorter_side_posts + 2 * pa.longer_side_posts - 4 →
  calculate_area pa = 324 := by
  sorry

end play_area_size_l3268_326823


namespace vector_not_parallel_implies_m_l3268_326858

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_not_parallel_implies_m (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  ¬(are_parallel a b) → m = -6 := by
  sorry

end vector_not_parallel_implies_m_l3268_326858


namespace rachel_budget_l3268_326801

/-- Given Sara's expenses and Rachel's spending intention, calculate Rachel's budget. -/
theorem rachel_budget (sara_shoes : ℕ) (sara_dress : ℕ) (rachel_multiplier : ℕ) : 
  sara_shoes = 50 → sara_dress = 200 → rachel_multiplier = 2 →
  (rachel_multiplier * sara_shoes + rachel_multiplier * sara_dress) = 500 := by
  sorry

#check rachel_budget

end rachel_budget_l3268_326801


namespace arithmetic_sequence_14th_term_l3268_326837

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_14th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_2 : a 2 = 5)
  (h_6 : a 6 = 17) :
  a 14 = 41 := by
  sorry

end arithmetic_sequence_14th_term_l3268_326837


namespace square_sum_value_l3268_326852

theorem square_sum_value (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : 
  x^2 + 9 * y^2 = 90 := by
  sorry

end square_sum_value_l3268_326852


namespace triangular_prism_volume_l3268_326815

/-- The volume of a triangular prism given the area of a lateral face and the distance to the opposite edge -/
theorem triangular_prism_volume (A_face : ℝ) (d : ℝ) (h_pos_A : A_face > 0) (h_pos_d : d > 0) :
  ∃ (V : ℝ), V = (1 / 2) * A_face * d ∧ V > 0 := by
  sorry

end triangular_prism_volume_l3268_326815


namespace max_value_x_plus_y_l3268_326847

theorem max_value_x_plus_y (x y : ℝ) (h : x - 3 * Real.sqrt (x + 1) = 3 * Real.sqrt (y + 2) - y) :
  (∀ a b : ℝ, a - 3 * Real.sqrt (a + 1) = 3 * Real.sqrt (b + 2) - b → x + y ≥ a + b) ∧
  x + y ≤ 9 + 3 * Real.sqrt 15 :=
by sorry

end max_value_x_plus_y_l3268_326847


namespace cubic_equation_solution_l3268_326846

theorem cubic_equation_solution (x y z a : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ z ≠ x →
  x^3 + a = -3*(y + z) →
  y^3 + a = -3*(x + z) →
  z^3 + a = -3*(x + y) →
  a ∈ Set.Ioo (-2 : ℝ) 2 \ {0} := by
  sorry

end cubic_equation_solution_l3268_326846


namespace arithmetic_sequence_product_l3268_326840

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  is_arithmetic_sequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 4 * b 5 = 10 →
  b 2 * b 7 = -224 ∨ b 2 * b 7 = -44 := by
sorry

end arithmetic_sequence_product_l3268_326840


namespace christines_dog_weight_l3268_326828

/-- Theorem: Christine's dog weight calculation -/
theorem christines_dog_weight (cat1_weight cat2_weight : ℕ) (dog_weight : ℕ) : 
  cat1_weight = 7 →
  cat2_weight = 10 →
  dog_weight = 2 * (cat1_weight + cat2_weight) →
  dog_weight = 34 := by
  sorry

end christines_dog_weight_l3268_326828


namespace line_intersection_y_axis_l3268_326830

/-- The line passing through points (2, 10) and (5, 16) intersects the y-axis at (0, 6) -/
theorem line_intersection_y_axis :
  let p₁ : ℝ × ℝ := (2, 10)
  let p₂ : ℝ × ℝ := (5, 16)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, 6) :=
by sorry

end line_intersection_y_axis_l3268_326830


namespace yah_to_bah_conversion_l3268_326899

/-- Exchange rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 16 / 10

/-- Exchange rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 15 / 9

/-- The number of yahs we want to convert -/
def yah_amount : ℕ := 2000

/-- The expected number of bahs after conversion -/
def expected_bah_amount : ℕ := 375

theorem yah_to_bah_conversion :
  (yah_amount : ℚ) / (rah_to_yah_rate * bah_to_rah_rate) = expected_bah_amount := by
  sorry

end yah_to_bah_conversion_l3268_326899


namespace train_combined_speed_l3268_326894

/-- The combined speed of two trains crossing a bridge simultaneously -/
theorem train_combined_speed
  (bridge_length : ℝ)
  (train1_length train1_time : ℝ)
  (train2_length train2_time : ℝ)
  (h1 : bridge_length = 300)
  (h2 : train1_length = 100)
  (h3 : train1_time = 30)
  (h4 : train2_length = 150)
  (h5 : train2_time = 45) :
  (train1_length + bridge_length) / train1_time +
  (train2_length + bridge_length) / train2_time =
  23.33 :=
sorry

end train_combined_speed_l3268_326894


namespace polynomial_expansion_problem_l3268_326877

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  7 * p^6 * q = 21 * p^5 * q^2 → 
  p = 3/4 := by sorry

end polynomial_expansion_problem_l3268_326877


namespace sean_apples_count_l3268_326854

/-- Proves that the number of apples Sean has after receiving apples from Susan
    is equal to the total number of apples mentioned. -/
theorem sean_apples_count (initial_apples : ℕ) (apples_from_susan : ℕ) (total_apples : ℕ)
    (h1 : initial_apples = 9)
    (h2 : apples_from_susan = 8)
    (h3 : total_apples = 17) :
    initial_apples + apples_from_susan = total_apples := by
  sorry

end sean_apples_count_l3268_326854


namespace scooter_repair_cost_l3268_326841

/-- Proves that the total repair cost is $11,000 given the conditions of Peter's scooter purchase and sale --/
theorem scooter_repair_cost (C : ℝ) : 
  (0.05 * C + 0.10 * C + 0.07 * C = 0.22 * C) →  -- Total repair cost is 22% of C
  (1.25 * C - C - 0.22 * C = 1500) →              -- Profit equation
  0.22 * C = 11000 :=                             -- Total repair cost is $11,000
by sorry

end scooter_repair_cost_l3268_326841


namespace customer_buys_score_of_eggs_l3268_326870

/-- Definition of a score in terms of units -/
def score : ℕ := 20

/-- Definition of a dozen in terms of units -/
def dozen : ℕ := 12

/-- The number of eggs a customer receives when buying a score of eggs -/
def eggs_in_score : ℕ := score

theorem customer_buys_score_of_eggs : eggs_in_score = 20 := by sorry

end customer_buys_score_of_eggs_l3268_326870


namespace point_below_right_of_line_range_of_a_below_right_of_line_l3268_326898

/-- A point (a, 1) is below and to the right of the line x-2y+4=0 if and only if a > -2 -/
theorem point_below_right_of_line (a : ℝ) : 
  (a - 2 * 1 + 4 > 0) ↔ (a > -2) :=
sorry

/-- The range of a for points (a, 1) below and to the right of the line x-2y+4=0 is (-2, +∞) -/
theorem range_of_a_below_right_of_line : 
  {a : ℝ | a - 2 * 1 + 4 > 0} = Set.Ioi (-2) :=
sorry

end point_below_right_of_line_range_of_a_below_right_of_line_l3268_326898


namespace area_ratio_when_diameter_tripled_l3268_326862

/-- The ratio of areas when a circle's diameter is tripled -/
theorem area_ratio_when_diameter_tripled (d : ℝ) (h : d > 0) :
  let r := d / 2
  let new_r := 3 * r
  (π * new_r ^ 2) / (π * r ^ 2) = 9 := by
sorry


end area_ratio_when_diameter_tripled_l3268_326862


namespace bus_car_speed_problem_l3268_326808

theorem bus_car_speed_problem : ∀ (v_bus v_car : ℝ),
  -- Given conditions
  (1.5 * v_bus + 1.5 * v_car = 180) →
  (2.5 * v_bus + v_car = 180) →
  -- Conclusion
  (v_bus = 40 ∧ v_car = 80) :=
by
  sorry

end bus_car_speed_problem_l3268_326808


namespace male_honor_roll_fraction_l3268_326825

theorem male_honor_roll_fraction (total : ℝ) (h1 : total > 0) :
  let female_ratio : ℝ := 2 / 5
  let female_honor_ratio : ℝ := 5 / 6
  let total_honor_ratio : ℝ := 22 / 30
  let male_ratio : ℝ := 1 - female_ratio
  let male_honor_ratio : ℝ := (total_honor_ratio * total - female_honor_ratio * female_ratio * total) / (male_ratio * total)
  male_honor_ratio = 2 / 3 := by
sorry

end male_honor_roll_fraction_l3268_326825


namespace exercise_distribution_properties_l3268_326882

/-- Represents the frequency distribution of daily exercise time --/
structure ExerciseDistribution :=
  (less_than_70 : ℕ)
  (between_70_and_80 : ℕ)
  (between_80_and_90 : ℕ)
  (greater_than_90 : ℕ)

/-- Theorem stating the properties of the exercise distribution --/
theorem exercise_distribution_properties
  (dist : ExerciseDistribution)
  (total_surveyed : ℕ)
  (h1 : dist.less_than_70 = 14)
  (h2 : dist.between_70_and_80 = 40)
  (h3 : dist.between_80_and_90 = 35)
  (h4 : total_surveyed = 100) :
  let m := (dist.between_70_and_80 : ℚ) / total_surveyed * 100
  let n := dist.greater_than_90
  let estimated_80_plus := ((dist.between_80_and_90 + dist.greater_than_90 : ℚ) / total_surveyed * 1000).floor
  let p := 86
  (m = 40 ∧ n = 11) ∧
  estimated_80_plus = 460 ∧
  (((11 : ℚ) / total_surveyed * 100 ≤ 25) ∧ 
   ((11 + 35 : ℚ) / total_surveyed * 100 ≥ 25)) := by
  sorry

#check exercise_distribution_properties

end exercise_distribution_properties_l3268_326882


namespace number_of_divisors_2310_l3268_326860

theorem number_of_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end number_of_divisors_2310_l3268_326860


namespace equation_solutions_l3268_326874

-- Define the equation
def equation (m n : ℕ+) : Prop := 3^(m.val) - 2^(n.val) = 1

-- State the theorem
theorem equation_solutions :
  ∀ m n : ℕ+, equation m n ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by sorry

end equation_solutions_l3268_326874


namespace complex_division_l3268_326883

theorem complex_division (z : ℂ) : z = -2 + I → z / (1 + I) = -1/2 + 3/2 * I := by sorry

end complex_division_l3268_326883


namespace negation_equivalence_l3268_326819

theorem negation_equivalence : 
  (¬∃ x : ℝ, (2 / x) + Real.log x ≤ 0) ↔ (∀ x : ℝ, (2 / x) + Real.log x > 0) :=
by sorry

end negation_equivalence_l3268_326819


namespace books_divided_l3268_326811

theorem books_divided (num_girls num_boys : ℕ) (total_girls_books : ℕ) : 
  num_girls = 15 →
  num_boys = 10 →
  total_girls_books = 225 →
  (total_girls_books / num_girls) * (num_girls + num_boys) = 375 :=
by sorry

end books_divided_l3268_326811


namespace find_m_l3268_326848

def U : Set Int := {-1, 2, 3, 6}

def A (m : Int) : Set Int := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : Int, A m = {-1, 6} ∧ m = -6 := by
  sorry

end find_m_l3268_326848


namespace intersection_points_f_squared_f_sixth_l3268_326861

theorem intersection_points_f_squared_f_sixth (f : ℝ → ℝ) (h_inj : Function.Injective f) :
  (∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x : ℝ, f (x^2) = f (x^6) ↔ x ∈ s)) := by
  sorry

end intersection_points_f_squared_f_sixth_l3268_326861


namespace quadratic_root_l3268_326818

theorem quadratic_root (a b k : ℝ) : 
  (∃ x : ℝ, x^2 - (a+b)*x + a*b*(1-k) = 0 ∧ x = 1) →
  (∃ y : ℝ, y^2 - (a+b)*y + a*b*(1-k) = 0 ∧ y = a + b - 1) :=
by sorry

end quadratic_root_l3268_326818


namespace quadratic_inequality_range_quadratic_inequality_solution_set_l3268_326844

-- Part 1
theorem quadratic_inequality_range (a : ℝ) : 
  (a > 0 ∧ ∃ x, a * x^2 - 3 * x + 2 < 0) ↔ (0 < a ∧ a < 9/8) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x < 1}
  else if a < 0 then {x | 3/a < x ∧ x < 1}
  else if 0 < a ∧ a < 3 then {x | x < 3/a ∨ x > 1}
  else if a = 3 then {x | x ≠ 1}
  else {x | x < 1 ∨ x > 3/a}

theorem quadratic_inequality_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 - 3 * x + 2 > a * x - 1 :=
sorry

end quadratic_inequality_range_quadratic_inequality_solution_set_l3268_326844


namespace pure_imaginary_complex_number_l3268_326878

theorem pure_imaginary_complex_number (x : ℝ) : 
  (Complex.I * (x + 3) = (x^2 + 2*x - 3) + Complex.I * (x + 3)) → x = 1 :=
by sorry

end pure_imaginary_complex_number_l3268_326878


namespace fifth_patient_cure_rate_l3268_326872

theorem fifth_patient_cure_rate 
  (cure_rate : ℝ) 
  (h_cure_rate : cure_rate = 1/5) 
  (first_four_patients : Fin 4 → Bool) 
  : ℝ :=
by
  sorry

end fifth_patient_cure_rate_l3268_326872


namespace houses_on_one_side_l3268_326897

theorem houses_on_one_side (x : ℕ) : 
  x + 3 * x = 160 → x = 40 := by sorry

end houses_on_one_side_l3268_326897


namespace root_equality_implies_c_equals_two_l3268_326836

theorem root_equality_implies_c_equals_two :
  ∀ (a b c d : ℕ),
    a > 1 → b > 1 → c > 1 → d > 1 →
    (∀ (M : ℝ), M ≠ 1 →
      (M^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c*d)) = M^(37/48))) →
    c = 2 := by
  sorry

end root_equality_implies_c_equals_two_l3268_326836


namespace millionth_digit_of_three_forty_first_l3268_326868

def fraction : ℚ := 3 / 41

def decimal_expansion (q : ℚ) : ℕ → ℕ := sorry

def nth_digit_after_decimal_point (q : ℚ) (n : ℕ) : ℕ :=
  decimal_expansion q n

theorem millionth_digit_of_three_forty_first (n : ℕ) (h : n = 1000000) :
  nth_digit_after_decimal_point fraction n = 7 := by sorry

end millionth_digit_of_three_forty_first_l3268_326868


namespace complex_modulus_problem_l3268_326867

theorem complex_modulus_problem (z : ℂ) : z = 3 + (3 + 4*I) / (4 - 3*I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_problem_l3268_326867


namespace negation_equivalence_l3268_326831

theorem negation_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end negation_equivalence_l3268_326831


namespace scientific_notation_300_billion_l3268_326864

theorem scientific_notation_300_billion :
  ∃ (a : ℝ) (n : ℤ), 300000000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = 11 := by
  sorry

end scientific_notation_300_billion_l3268_326864


namespace turquoise_score_difference_is_correct_l3268_326835

/-- Calculates 5/8 of the difference between white and black scores in a turquoise mixture --/
def turquoise_score_difference (total : ℚ) : ℚ :=
  let white_ratio : ℚ := 5
  let black_ratio : ℚ := 3
  let total_ratio : ℚ := white_ratio + black_ratio
  let part_value : ℚ := total / total_ratio
  let white_scores : ℚ := white_ratio * part_value
  let black_scores : ℚ := black_ratio * part_value
  let difference : ℚ := white_scores - black_scores
  (5 : ℚ) / 8 * difference

/-- Theorem stating that 5/8 of the difference between white and black scores is 58.125 --/
theorem turquoise_score_difference_is_correct :
  turquoise_score_difference 372 = 58125 / 1000 :=
by sorry

end turquoise_score_difference_is_correct_l3268_326835


namespace log_relationship_l3268_326821

theorem log_relationship (c d : ℝ) (hc : c = Real.log 625 / Real.log 4) (hd : d = Real.log 25 / Real.log 5) :
  c = 2 * d := by
  sorry

end log_relationship_l3268_326821


namespace percent_relation_l3268_326814

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.1 * b) 
  (h2 : b = 2 * a) : 
  c = 0.2 * a := by
sorry

end percent_relation_l3268_326814


namespace polynomial_division_theorem_l3268_326875

/-- The polynomial being divided -/
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + x^2 + 4

/-- The divisor -/
def g (x : ℝ) : ℝ := (x - 2)^2

/-- The remainder -/
def r (x : ℝ) : ℝ := 35*x + 48

/-- The quotient -/
def q (x : ℝ) : ℝ := sorry

theorem polynomial_division_theorem :
  ∀ x : ℝ, f x = g x * q x + r x := by sorry

end polynomial_division_theorem_l3268_326875


namespace min_value_ab_l3268_326879

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = Real.sqrt (a*b)) :
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 :=
by sorry

end min_value_ab_l3268_326879


namespace round_trip_average_speed_l3268_326850

/-- The average speed of a round trip, given the outbound speed and relative duration of return trip -/
theorem round_trip_average_speed 
  (outbound_speed : ℝ) 
  (return_time_factor : ℝ) 
  (h1 : outbound_speed = 48) 
  (h2 : return_time_factor = 2) : 
  (2 * outbound_speed) / (1 + return_time_factor) = 32 := by
  sorry

end round_trip_average_speed_l3268_326850


namespace min_sum_x_y_l3268_326896

theorem min_sum_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (2 * x + Real.sqrt (4 * x^2 + 1)) * (Real.sqrt (y^2 + 4) - 2) ≥ y) :
  x + y ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (2 * x₀ + Real.sqrt (4 * x₀^2 + 1)) * (Real.sqrt (y₀^2 + 4) - 2) ≥ y₀ ∧
    x₀ + y₀ = 2 :=
by sorry

end min_sum_x_y_l3268_326896


namespace horner_v3_value_l3268_326800

/-- The polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

/-- The value of v_3 in Horner's method -/
def v_3 (x : ℝ) : ℝ := (((7*x + 6)*x + 5)*x + 4)

/-- Theorem: The value of v_3 is 262 when x = 3 -/
theorem horner_v3_value : v_3 3 = 262 := by
  sorry

end horner_v3_value_l3268_326800


namespace negation_of_proposition_l3268_326865

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x ≥ 1 → x^2 - 4*x + 2 ≥ -1)) ↔ (∃ x : ℝ, x < 1 ∧ x^2 - 4*x + 2 < -1) :=
by sorry

end negation_of_proposition_l3268_326865


namespace pear_difference_is_five_l3268_326807

/-- The number of bags of pears Austin picked fewer than Dallas -/
def pear_difference (dallas_apples dallas_pears austin_total : ℕ) (austin_apple_diff : ℕ) : ℕ :=
  dallas_pears - (austin_total - (dallas_apples + austin_apple_diff))

theorem pear_difference_is_five :
  pear_difference 14 9 24 6 = 5 := by
  sorry

end pear_difference_is_five_l3268_326807


namespace arithmetic_sequence_problem_l3268_326853

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 20) 
  (h_a6 : a 6 = 11) : 
  a 5 = 9 := by
  sorry

end arithmetic_sequence_problem_l3268_326853


namespace max_sum_of_endpoints_l3268_326843

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the theorem
theorem max_sum_of_endpoints
  (m n : ℝ)
  (h1 : n > m)
  (h2 : ∀ x ∈ Set.Icc m n, -5 ≤ f x ∧ f x ≤ 4)
  (h3 : ∃ x ∈ Set.Icc m n, f x = -5)
  (h4 : ∃ x ∈ Set.Icc m n, f x = 4) :
  ∃ k : ℝ, (∀ a b : ℝ, a ≥ m ∧ b ≤ n ∧ b > a ∧
    (∀ x ∈ Set.Icc a b, -5 ≤ f x ∧ f x ≤ 4) ∧
    (∃ x ∈ Set.Icc a b, f x = -5) ∧
    (∃ x ∈ Set.Icc a b, f x = 4) →
    a + b ≤ k) ∧
  k = n + m ∧ k = 7 :=
sorry

end max_sum_of_endpoints_l3268_326843
