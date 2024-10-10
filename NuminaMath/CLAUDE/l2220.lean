import Mathlib

namespace problem_solution_l2220_222030

theorem problem_solution : 
  (∃ n : ℕ, 25 = 5 * n) ∧ 
  (∃ m : ℕ, 209 = 19 * m) ∧ ¬(∃ k : ℕ, 63 = 19 * k) ∧
  (∃ p : ℕ, 180 = 9 * p) := by
  sorry

end problem_solution_l2220_222030


namespace min_alterations_for_equal_sum_l2220_222047

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ := !![1,2,3; 4,5,6; 7,8,9]

def row_sum (M : Matrix (Fin 3) (Fin 3) ℕ) (i : Fin 3) : ℕ :=
  M i 0 + M i 1 + M i 2

def col_sum (M : Matrix (Fin 3) (Fin 3) ℕ) (j : Fin 3) : ℕ :=
  M 0 j + M 1 j + M 2 j

def all_sums_different (M : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j i' j', i ≠ i' ∨ j ≠ j' → row_sum M i ≠ row_sum M j ∧ col_sum M i ≠ col_sum M j'

theorem min_alterations_for_equal_sum :
  all_sums_different initial_matrix ∧
  (∃ M : Matrix (Fin 3) (Fin 3) ℕ, ∃ i j : Fin 3,
    (∀ x y, (M x y ≠ initial_matrix x y) → (x = i ∧ y = j)) ∧
    (∃ r c, row_sum M r = col_sum M c)) ∧
  ¬(∃ r c, row_sum initial_matrix r = col_sum initial_matrix c) :=
by sorry

end min_alterations_for_equal_sum_l2220_222047


namespace simple_interest_problem_l2220_222096

/-- Given a principal amount P and an unknown interest rate R,
    if increasing the rate by 1% results in Rs. 72 more interest over 3 years,
    then P must be Rs. 2400. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 1) * 3) / 100 - (P * R * 3) / 100 = 72 → P = 2400 := by
  sorry

end simple_interest_problem_l2220_222096


namespace midpoint_between_fractions_l2220_222083

theorem midpoint_between_fractions :
  (1 / 12 + 1 / 20) / 2 = 1 / 15 := by
  sorry

end midpoint_between_fractions_l2220_222083


namespace inequality_equivalence_l2220_222013

theorem inequality_equivalence (x : ℝ) : 
  (3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 6) ↔ (2 < x ∧ x < 3) := by
  sorry

end inequality_equivalence_l2220_222013


namespace tangent_line_equation_l2220_222057

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the slope of the line parallel to 3x + y = 0
def m : ℝ := -3

-- Define the point of tangency
def a : ℝ := 1
def b : ℝ := f a

-- State the theorem
theorem tangent_line_equation :
  ∃ (c : ℝ), ∀ x y : ℝ,
    (y - b = m * (x - a)) ↔ (y = -3*x + c) :=
sorry

end tangent_line_equation_l2220_222057


namespace completing_square_sum_l2220_222031

theorem completing_square_sum (a b : ℝ) : 
  (∀ x, x^2 - 4*x = 5 ↔ (x + a)^2 = b) → a + b = 7 := by
  sorry

end completing_square_sum_l2220_222031


namespace outfits_count_l2220_222011

/-- The number of outfits with different colored shirts and hats -/
def num_outfits : ℕ :=
  let red_shirts := 5
  let green_shirts := 5
  let pants := 6
  let green_hats := 8
  let red_hats := 8
  (red_shirts * pants * green_hats) + (green_shirts * pants * red_hats)

theorem outfits_count : num_outfits = 480 := by
  sorry

end outfits_count_l2220_222011


namespace ball_probabilities_l2220_222008

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The number of red balls in the bag -/
def red_balls : ℕ := 2

/-- The probability of drawing a red ball -/
def prob_red : ℚ := red_balls / total_balls

/-- The probability of drawing two black balls without replacement -/
def prob_two_black : ℚ := (black_balls * (black_balls - 1)) / (total_balls * (total_balls - 1))

theorem ball_probabilities :
  prob_red = 2/5 ∧ prob_two_black = 3/10 :=
sorry

end ball_probabilities_l2220_222008


namespace exp_greater_equal_linear_l2220_222077

theorem exp_greater_equal_linear : ∀ x : ℝ, Real.exp x ≥ Real.exp 1 * x := by sorry

end exp_greater_equal_linear_l2220_222077


namespace stock_price_change_l2220_222060

theorem stock_price_change (initial_price : ℝ) (initial_price_positive : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 1.2 := by
sorry

end stock_price_change_l2220_222060


namespace complement_intersection_theorem_l2220_222087

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 3}

theorem complement_intersection_theorem : 
  {4, 5, 6} = (U \ M) ∩ (U \ N) := by sorry

end complement_intersection_theorem_l2220_222087


namespace roots_quadratic_equation_l2220_222048

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + a - 2014 = 0) → 
  (b^2 + b - 2014 = 0) → 
  a^2 + 2*a + b = 2013 := by
sorry

end roots_quadratic_equation_l2220_222048


namespace linda_furniture_spending_l2220_222062

theorem linda_furniture_spending (original_savings : ℝ) (tv_cost : ℝ) 
  (h1 : original_savings = 1800)
  (h2 : tv_cost = 450) :
  (original_savings - tv_cost) / original_savings = 3/4 := by
  sorry

end linda_furniture_spending_l2220_222062


namespace angle_between_vectors_l2220_222035

theorem angle_between_vectors (a b : ℝ × ℝ) :
  (∀ x y : ℝ, a.1 * x + a.2 * y = 1) →  -- a is a unit vector
  b = (2, 2 * Real.sqrt 3) →           -- b = (2, 2√3)
  a.1 * (2 * a.1 + b.1) + a.2 * (2 * a.2 + b.2) = 0 →  -- a ⟂ (2a + b)
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * Real.pi / 3 :=
by sorry

end angle_between_vectors_l2220_222035


namespace investment_triple_period_l2220_222006

/-- The annual interest rate as a decimal -/
def r : ℝ := 0.341

/-- The condition for the investment to more than triple -/
def triple_condition (t : ℝ) : Prop := (1 + r) ^ t > 3

/-- The smallest investment period in years for the investment to more than triple -/
def smallest_period : ℕ := 4

theorem investment_triple_period :
  (∀ t : ℝ, t < smallest_period → ¬ triple_condition t) ∧
  triple_condition (smallest_period : ℝ) :=
sorry

end investment_triple_period_l2220_222006


namespace sphere_tangency_relation_l2220_222092

/-- Given three mutually tangent spheres touching a plane at three points on a circle of radius R,
    and two spheres of radii r and ρ (ρ > r) each tangent to the three given spheres and the plane,
    prove that 1/r - 1/ρ = 2√3/R. -/
theorem sphere_tangency_relation (R r ρ : ℝ) (h1 : r > 0) (h2 : ρ > 0) (h3 : ρ > r) :
  1 / r - 1 / ρ = 2 * Real.sqrt 3 / R :=
by sorry

end sphere_tangency_relation_l2220_222092


namespace fraction_meaningful_l2220_222051

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 1)) ↔ x ≠ 1 :=
sorry

end fraction_meaningful_l2220_222051


namespace marble_count_theorem_l2220_222094

theorem marble_count_theorem (g y : ℚ) :
  (g - 3) / (g + y - 3) = 1 / 6 →
  g / (g + y - 4) = 1 / 4 →
  g + y = 18 :=
by sorry

end marble_count_theorem_l2220_222094


namespace ginger_flower_sales_l2220_222041

/-- Represents the number of flowers sold of each type -/
structure FlowerSales where
  lilacs : ℕ
  roses : ℕ
  gardenias : ℕ

/-- Calculates the total number of flowers sold -/
def totalFlowers (sales : FlowerSales) : ℕ :=
  sales.lilacs + sales.roses + sales.gardenias

/-- Theorem: Given the conditions of Ginger's flower sales, the total number of flowers sold is 45 -/
theorem ginger_flower_sales :
  ∀ (sales : FlowerSales),
    sales.lilacs = 10 →
    sales.roses = 3 * sales.lilacs →
    sales.gardenias = sales.lilacs / 2 →
    totalFlowers sales = 45 := by
  sorry


end ginger_flower_sales_l2220_222041


namespace smallest_sum_of_abs_l2220_222059

def matrix_squared (a b c d : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  !![a^2 + b*c, a*b + b*d;
     a*c + c*d, b*c + d^2]

theorem smallest_sum_of_abs (a b c d : ℤ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  matrix_squared a b c d = !![9, 0; 0, 9] →
  (∃ (w x y z : ℤ), w ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    matrix_squared w x y z = !![9, 0; 0, 9] ∧
    |w| + |x| + |y| + |z| < |a| + |b| + |c| + |d|) ∨
  |a| + |b| + |c| + |d| = 8 :=
by sorry

end smallest_sum_of_abs_l2220_222059


namespace sum_lent_is_350_l2220_222036

/-- Proves that the sum lent is 350 Rs. given the specified conditions --/
theorem sum_lent_is_350 (P : ℚ) : 
  (∀ (I : ℚ), I = P * (4 : ℚ) * (8 : ℚ) / 100) →  -- Simple interest formula
  (∀ (I : ℚ), I = P - 238) →                      -- Interest is 238 less than principal
  P = 350 := by
  sorry

end sum_lent_is_350_l2220_222036


namespace amoeba_problem_l2220_222069

/-- The number of amoebas after n days, given an initial population and split factor --/
def amoeba_population (initial : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial * split_factor ^ days

/-- Theorem: Given 2 initial amoebas that split into 3 each day, after 5 days there will be 486 amoebas --/
theorem amoeba_problem :
  amoeba_population 2 3 5 = 486 := by
  sorry

#eval amoeba_population 2 3 5

end amoeba_problem_l2220_222069


namespace special_function_properties_l2220_222027

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x < 0 → f x > 0)

/-- Main theorem encapsulating all parts of the problem -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ a x : ℝ, f (x^2) + 3 * f a > 3 * f x + f (a * x) ↔
    (a ≠ 0 ∧ ((a > 3 ∧ 3 < x ∧ x < a) ∨ (a < 3 ∧ a < x ∧ x < 3)))) :=
by sorry

end special_function_properties_l2220_222027


namespace expand_expression_l2220_222071

theorem expand_expression (x y : ℝ) : 12 * (3 * x - 4 * y + 2) = 36 * x - 48 * y + 24 := by
  sorry

end expand_expression_l2220_222071


namespace set_union_problem_l2220_222084

theorem set_union_problem (m : ℝ) : 
  let A : Set ℝ := {1, 2^m}
  let B : Set ℝ := {0, 2}
  A ∪ B = {0, 1, 2, 8} → m = 3 := by
sorry

end set_union_problem_l2220_222084


namespace min_value_expression_l2220_222078

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 5 := by
  sorry

end min_value_expression_l2220_222078


namespace sum_of_reciprocals_roots_l2220_222017

theorem sum_of_reciprocals_roots (x : ℝ) : 
  (x^2 - 13*x + 4 = 0) → 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 13*x + 4 = (x - r₁) * (x - r₂) ∧ 
    (1 / r₁ + 1 / r₂ = 13 / 4)) :=
by sorry

end sum_of_reciprocals_roots_l2220_222017


namespace mixture_ratio_after_replacement_l2220_222055

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the state of the liquid mixture -/
structure LiquidMixture where
  ratioAB : Ratio
  volumeA : ℝ
  totalVolume : ℝ

def initialMixture : LiquidMixture :=
  { ratioAB := { numerator := 4, denominator := 1 }
  , volumeA := 32
  , totalVolume := 40 }

def replacementVolume : ℝ := 20

/-- Calculates the new ratio after replacing some mixture with liquid B -/
def newRatio (initial : LiquidMixture) (replace : ℝ) : Ratio :=
  { numerator := 2
  , denominator := 3 }

theorem mixture_ratio_after_replacement :
  newRatio initialMixture replacementVolume = { numerator := 2, denominator := 3 } :=
sorry

end mixture_ratio_after_replacement_l2220_222055


namespace max_edges_cube_plane_intersection_l2220_222056

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- A plane is a flat, two-dimensional surface that extends infinitely far -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- A polygon is a plane figure with straight sides -/
structure Polygon where
  edges : ℕ

/-- The result of intersecting a cube with a plane is a polygon -/
def intersect (c : Cube) (p : Plane) : Polygon :=
  sorry

/-- Theorem: The maximum number of edges in a polygon formed by the intersection of a cube and a plane is 6 -/
theorem max_edges_cube_plane_intersection (c : Cube) (p : Plane) :
  (intersect c p).edges ≤ 6 ∧ ∃ (c' : Cube) (p' : Plane), (intersect c' p').edges = 6 :=
sorry

end max_edges_cube_plane_intersection_l2220_222056


namespace average_age_of_joans_kittens_l2220_222024

/-- Represents the number of days in each month (simplified to 30 for all months) -/
def daysInMonth : ℕ := 30

/-- Calculates the age of kittens in days given their birth month -/
def kittenAge (birthMonth : ℕ) : ℕ :=
  (4 - birthMonth) * daysInMonth + 15

/-- Represents Joan's original number of kittens -/
def joansOriginalKittens : ℕ := 8

/-- Represents the number of kittens Joan gave away -/
def joansGivenAwayKittens : ℕ := 2

/-- Represents the number of neighbor's kittens Joan adopted -/
def adoptedNeighborKittens : ℕ := 3

/-- Represents the number of friend's kittens Joan adopted -/
def adoptedFriendKittens : ℕ := 1

/-- Calculates the total number of kittens Joan has after all transactions -/
def totalJoansKittens : ℕ :=
  joansOriginalKittens - joansGivenAwayKittens + adoptedNeighborKittens + adoptedFriendKittens

/-- Theorem stating that the average age of Joan's kittens on April 15th is 90 days -/
theorem average_age_of_joans_kittens :
  (joansOriginalKittens - joansGivenAwayKittens) * kittenAge 1 +
  adoptedNeighborKittens * kittenAge 2 +
  adoptedFriendKittens * kittenAge 3 =
  90 * totalJoansKittens := by sorry

end average_age_of_joans_kittens_l2220_222024


namespace percentage_increase_decrease_l2220_222021

theorem percentage_increase_decrease (α β p q : ℝ) 
  (h_pos_α : α > 0) (h_pos_β : β > 0) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_q_lt_50 : q < 50) :
  (α * β * (1 + p / 100) * (1 - q / 100) > α * β) ↔ (p > 100 * q / (100 - q)) :=
by sorry

end percentage_increase_decrease_l2220_222021


namespace smallest_four_digit_mod_8_l2220_222023

theorem smallest_four_digit_mod_8 : 
  ∀ n : ℕ, 
    1000 ≤ n ∧ n ≡ 3 [MOD 8] → 
    1003 ≤ n :=
by sorry

end smallest_four_digit_mod_8_l2220_222023


namespace swim_time_ratio_l2220_222081

/-- The ratio of time taken to swim upstream to downstream -/
theorem swim_time_ratio (v_m : ℝ) (v_s : ℝ) (h1 : v_m = 4.5) (h2 : v_s = 1.5) :
  (v_m + v_s) / (v_m - v_s) = 2 := by
  sorry

#check swim_time_ratio

end swim_time_ratio_l2220_222081


namespace probability_of_y_selection_l2220_222001

theorem probability_of_y_selection 
  (prob_x : ℝ) 
  (prob_both : ℝ) 
  (h1 : prob_x = 1/5)
  (h2 : prob_both = 0.05714285714285714) :
  prob_both / prob_x = 0.2857142857142857 := by
sorry

end probability_of_y_selection_l2220_222001


namespace principal_is_7500_l2220_222072

/-- Calculates the compound interest amount -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Proves that the principal is 7500 given the conditions -/
theorem principal_is_7500 
  (rate : ℝ) 
  (time : ℕ) 
  (interest : ℝ) 
  (h_rate : rate = 0.04) 
  (h_time : time = 2) 
  (h_interest : interest = 612) : 
  ∃ (principal : ℝ), principal = 7500 ∧ compound_interest principal rate time = interest :=
sorry

end principal_is_7500_l2220_222072


namespace ice_cream_scoop_arrangements_l2220_222025

theorem ice_cream_scoop_arrangements :
  (Finset.univ.filter (fun σ : Equiv.Perm (Fin 5) => true)).card = 120 := by
  sorry

end ice_cream_scoop_arrangements_l2220_222025


namespace inequality_proof_l2220_222085

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z)
  (h_sum : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ 
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end inequality_proof_l2220_222085


namespace decimal_point_movement_l2220_222012

theorem decimal_point_movement (x : ℝ) : x / 100 = x - 1.485 ↔ x = 1.5 := by
  sorry

end decimal_point_movement_l2220_222012


namespace rectangle_perimeter_is_48_l2220_222091

/-- A rectangle can be cut into two squares with side length 8 cm -/
structure Rectangle where
  length : ℝ
  width : ℝ
  is_cut_into_squares : length = 2 * width
  square_side : ℝ
  square_side_eq : square_side = width

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: The perimeter of the rectangle is 48 cm -/
theorem rectangle_perimeter_is_48 (r : Rectangle) (h : r.square_side = 8) : perimeter r = 48 := by
  sorry

end rectangle_perimeter_is_48_l2220_222091


namespace joels_age_proof_l2220_222049

/-- Joel's current age -/
def joels_current_age : ℕ := 5

/-- Joel's dad's current age -/
def dads_current_age : ℕ := 32

/-- The age Joel will be when his dad is twice his age -/
def joels_future_age : ℕ := 27

theorem joels_age_proof :
  joels_current_age = 5 ∧
  dads_current_age = 32 ∧
  joels_future_age = 27 ∧
  dads_current_age + (joels_future_age - joels_current_age) = 2 * joels_future_age :=
by sorry

end joels_age_proof_l2220_222049


namespace semicircle_perimeter_l2220_222046

/-- The perimeter of a semicircle with radius 20 is equal to 20π + 40. -/
theorem semicircle_perimeter : 
  ∀ (r : ℝ), r = 20 → (r * π + r) = 20 * π + 40 := by
  sorry

end semicircle_perimeter_l2220_222046


namespace positive_solution_x_l2220_222003

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 12 - 2 * x - 3 * y)
  (eq2 : y * z = 8 - 4 * y - 2 * z)
  (eq3 : x * z = 24 - 4 * x - 3 * z)
  (x_pos : x > 0) :
  x = 3 := by sorry

end positive_solution_x_l2220_222003


namespace intersection_of_A_and_B_l2220_222052

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l2220_222052


namespace morgan_lunch_change_l2220_222061

/-- Calculates the change Morgan receives from his lunch order --/
theorem morgan_lunch_change : 
  let hamburger : ℚ := 5.75
  let onion_rings : ℚ := 2.50
  let smoothie : ℚ := 3.25
  let side_salad : ℚ := 3.75
  let chocolate_cake : ℚ := 4.20
  let discount_rate : ℚ := 0.10
  let tax_rate : ℚ := 0.06
  let payment : ℚ := 50

  let total_before_discount : ℚ := hamburger + onion_rings + smoothie + side_salad + chocolate_cake
  let discount : ℚ := (side_salad + chocolate_cake) * discount_rate
  let total_after_discount : ℚ := total_before_discount - discount
  let tax : ℚ := total_after_discount * tax_rate
  let final_total : ℚ := total_after_discount + tax
  let change : ℚ := payment - final_total

  change = 30.34 := by sorry

end morgan_lunch_change_l2220_222061


namespace peanuts_in_box_l2220_222005

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) :
  initial_peanuts = 4 → added_peanuts = 12 → initial_peanuts + added_peanuts = 16 := by
  sorry

end peanuts_in_box_l2220_222005


namespace max_pieces_is_seven_l2220_222034

/-- Represents a mapping of letters to digits -/
def LetterDigitMap := Char → Nat

/-- Checks if a mapping is valid (each letter maps to a unique digit) -/
def is_valid_mapping (m : LetterDigitMap) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a string to a number using the given mapping -/
def string_to_number (s : String) (m : LetterDigitMap) : Nat :=
  s.foldl (fun acc c => acc * 10 + m c) 0

/-- Represents the equation PIE = n * PIECE -/
def satisfies_equation (pie : String) (piece : String) (n : Nat) (m : LetterDigitMap) : Prop :=
  string_to_number pie m = n * string_to_number piece m

theorem max_pieces_is_seven :
  ∃ (pie piece : String) (m : LetterDigitMap),
    pie.length = 5 ∧
    piece.length = 5 ∧
    is_valid_mapping m ∧
    satisfies_equation pie piece 7 m ∧
    (∀ (pie' piece' : String) (m' : LetterDigitMap) (n : Nat),
      pie'.length = 5 →
      piece'.length = 5 →
      is_valid_mapping m' →
      satisfies_equation pie' piece' n m' →
      n ≤ 7) :=
sorry

end max_pieces_is_seven_l2220_222034


namespace expression_evaluation_l2220_222029

theorem expression_evaluation (x y : ℝ) (h : (x - 2)^2 + |y - 3| = 0) :
  ((x - 2*y) * (x + 2*y) - (x - y)^2 + y * (y + 2*x)) / (-2*y) = 2 := by
sorry

end expression_evaluation_l2220_222029


namespace max_large_chips_l2220_222079

theorem max_large_chips (total : ℕ) (small large : ℕ) (h1 : total = 100) 
  (h2 : total = small + large) (h3 : ∃ p : ℕ, Prime p ∧ Even p ∧ small = large + p) : 
  large ≤ 49 := by
  sorry

end max_large_chips_l2220_222079


namespace rectangle_tiling_existence_l2220_222064

/-- A rectangle is represented by its width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tiling of a rectangle using smaller rectangles -/
def CanTile (r : Rectangle) (tiles : List Rectangle) : Prop :=
  sorry

/-- The main theorem: there exists an N such that all rectangles with sides > N can be tiled -/
theorem rectangle_tiling_existence : 
  ∃ N : ℕ, ∀ m n : ℕ, m > N → n > N → 
    CanTile ⟨m, n⟩ [⟨4, 6⟩, ⟨5, 7⟩] :=
  sorry

end rectangle_tiling_existence_l2220_222064


namespace distribution_and_points_correct_l2220_222018

/-- Represents a comparison between two tanks -/
structure TankComparison where
  siyan_name : String
  siyan_quality : Nat
  zvezda_quality : Nat
  zvezda_name : String

/-- Calculates the oil distribution and rating points -/
def calculate_distribution_and_points (comparisons : List TankComparison) (oil_quantity : Real) :
  (Real × Real × Nat × Nat) :=
  let process := λ (acc : Real × Real × Nat × Nat) (comp : TankComparison) =>
    let (hv_22, lv_426, siyan_points, zvezda_points) := acc
    let new_hv_22 := hv_22 + 
      (if comp.siyan_quality > 2 then oil_quantity else 0) +
      (if comp.zvezda_quality > 2 then oil_quantity else 0)
    let new_lv_426 := lv_426 + 
      (if comp.siyan_quality ≤ 2 then oil_quantity else 0) +
      (if comp.zvezda_quality ≤ 2 then oil_quantity else 0)
    let new_siyan_points := siyan_points +
      (if comp.siyan_quality > comp.zvezda_quality then 3
       else if comp.siyan_quality = comp.zvezda_quality then 1
       else 0)
    let new_zvezda_points := zvezda_points +
      (if comp.zvezda_quality > comp.siyan_quality then 3
       else if comp.siyan_quality = comp.zvezda_quality then 1
       else 0)
    (new_hv_22, new_lv_426, new_siyan_points, new_zvezda_points)
  comparisons.foldl process (0, 0, 0, 0)

/-- Theorem stating the correctness of the calculation -/
theorem distribution_and_points_correct (comparisons : List TankComparison) (oil_quantity : Real) :
  let (hv_22, lv_426, siyan_points, zvezda_points) := calculate_distribution_and_points comparisons oil_quantity
  (hv_22 ≥ 0 ∧ lv_426 ≥ 0 ∧ 
   hv_22 + lv_426 = oil_quantity * comparisons.length * 2 ∧
   siyan_points + zvezda_points = comparisons.length * 3) :=
by sorry

end distribution_and_points_correct_l2220_222018


namespace smallest_prime_12_less_than_square_l2220_222004

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_12_less_than_square : 
  (∃ n : ℕ, is_perfect_square n ∧ is_prime (n - 12)) ∧ 
  (∀ m : ℕ, is_perfect_square m ∧ is_prime (m - 12) → m - 12 ≥ 13) :=
sorry

end smallest_prime_12_less_than_square_l2220_222004


namespace rotate_point_around_OA_l2220_222088

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotate a point around a ray by a given angle -/
def rotateAroundRay (p : Point3D) (origin : Point3D) (axis : Point3D) (angle : ℝ) : Point3D :=
  sorry

/-- The theorem to prove -/
theorem rotate_point_around_OA : 
  let A : Point3D := ⟨1, 1, 1⟩
  let P : Point3D := ⟨1, 1, 0⟩
  let O : Point3D := ⟨0, 0, 0⟩
  let angle : ℝ := π / 3  -- 60 degrees in radians
  let rotated_P : Point3D := rotateAroundRay P O A angle
  rotated_P = ⟨1/3, 4/3, 1/3⟩ := by sorry

end rotate_point_around_OA_l2220_222088


namespace permutation_preserves_lines_l2220_222022

-- Define a type for points in a plane
variable {Point : Type*}

-- Define a permutation of points
variable (f : Point → Point)

-- Define what it means for three points to be collinear
def collinear (A B C : Point) : Prop := sorry

-- Define what it means for three points to lie on a circle
def on_circle (A B C : Point) : Prop := sorry

-- State the theorem
theorem permutation_preserves_lines 
  (h : ∀ A B C : Point, on_circle A B C → on_circle (f A) (f B) (f C)) :
  (∀ A B C : Point, collinear A B C ↔ collinear (f A) (f B) (f C)) :=
sorry

end permutation_preserves_lines_l2220_222022


namespace prob_at_least_two_evens_eq_247_256_l2220_222014

/-- Probability of getting an even number on a single roll of a standard die -/
def p_even : ℚ := 1/2

/-- Number of rolls -/
def n : ℕ := 8

/-- Probability of getting exactly k even numbers in n rolls -/
def prob_k_evens (k : ℕ) : ℚ :=
  (n.choose k) * (p_even ^ k) * ((1 - p_even) ^ (n - k))

/-- Probability of getting at least two even numbers in n rolls -/
def prob_at_least_two_evens : ℚ :=
  1 - (prob_k_evens 0 + prob_k_evens 1)

theorem prob_at_least_two_evens_eq_247_256 :
  prob_at_least_two_evens = 247/256 := by sorry

end prob_at_least_two_evens_eq_247_256_l2220_222014


namespace infinite_solutions_diophantine_equation_l2220_222045

theorem infinite_solutions_diophantine_equation :
  ∃ f g h : ℕ → ℕ,
    (∀ t : ℕ, (f t)^2 + (g t)^3 = (h t)^5) ∧
    (∀ t₁ t₂ : ℕ, t₁ ≠ t₂ → (f t₁, g t₁, h t₁) ≠ (f t₂, g t₂, h t₂)) :=
by sorry

end infinite_solutions_diophantine_equation_l2220_222045


namespace segment_length_line_circle_l2220_222015

/-- The length of the segment cut by a line from a circle -/
theorem segment_length_line_circle (a b c : ℝ) (x₀ y₀ r : ℝ) : 
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c = 0 → 
    2 * Real.sqrt (r^2 - (a*x₀ + b*y₀ + c)^2 / (a^2 + b^2)) = Real.sqrt 3) →
  x₀ = 1 ∧ y₀ = 0 ∧ r = 1 ∧ a = 1 ∧ b = Real.sqrt 3 ∧ c = -2 :=
by sorry

end segment_length_line_circle_l2220_222015


namespace age_difference_l2220_222009

/-- Proves that the age difference between a man and his son is 30 years -/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 28 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 30 := by
  sorry

#check age_difference

end age_difference_l2220_222009


namespace binomial_coeff_congruence_l2220_222010

-- Define the binomial coefficient
def binomial_coeff (n p : ℕ) : ℕ := Nat.choose n p

-- State the theorem
theorem binomial_coeff_congruence (p n : ℕ) 
  (hp : Nat.Prime p) 
  (hodd : Odd p) 
  (hn : n ≥ p) :
  binomial_coeff n p ≡ (n / p) [MOD p] :=
sorry

end binomial_coeff_congruence_l2220_222010


namespace trig_abs_sum_diff_ge_one_l2220_222033

theorem trig_abs_sum_diff_ge_one (x : ℝ) : 
  max (|Real.cos x - Real.sin x|) (|Real.sin x + Real.cos x|) ≥ 1 := by
  sorry

end trig_abs_sum_diff_ge_one_l2220_222033


namespace chocolate_bar_reduction_l2220_222038

theorem chocolate_bar_reduction 
  (m n : ℕ) 
  (h_lt : m < n) 
  (a b : ℕ) 
  (h_div_a : n^5 ∣ a) 
  (h_div_b : n^5 ∣ b) : 
  ∃ (x y : ℕ), 
    x ≤ a ∧ 
    y ≤ b ∧ 
    x * y = a * b * (m / n)^10 := by
  sorry

end chocolate_bar_reduction_l2220_222038


namespace athlete_C_is_best_l2220_222099

structure Athlete where
  name : String
  average_score : ℝ
  variance : ℝ

def athletes : List Athlete := [
  ⟨"A", 7, 0.9⟩,
  ⟨"B", 8, 1.1⟩,
  ⟨"C", 8, 0.9⟩,
  ⟨"D", 7, 1.0⟩
]

def has_best_performance_and_stability (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, 
    (a.average_score > b.average_score) ∨ 
    (a.average_score = b.average_score ∧ a.variance ≤ b.variance)

theorem athlete_C_is_best : 
  ∃ a ∈ athletes, a.name = "C" ∧ has_best_performance_and_stability a athletes := by
  sorry

end athlete_C_is_best_l2220_222099


namespace john_works_fifty_weeks_l2220_222070

/-- Represents the number of weeks John works in a year -/
def weeks_worked (patients_hospital1 : ℕ) (patients_hospital2_increase : ℚ) 
  (days_per_week : ℕ) (total_patients_per_year : ℕ) : ℚ :=
  let patients_hospital2 := patients_hospital1 * (1 + patients_hospital2_increase)
  let patients_per_week := (patients_hospital1 + patients_hospital2) * days_per_week
  total_patients_per_year / patients_per_week

/-- Theorem stating that John works 50 weeks a year given the problem conditions -/
theorem john_works_fifty_weeks :
  weeks_worked 20 (1/5 : ℚ) 5 11000 = 50 := by sorry

end john_works_fifty_weeks_l2220_222070


namespace problem_solution_l2220_222086

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
sorry

end problem_solution_l2220_222086


namespace new_ratio_second_term_l2220_222095

def original_ratio : Rat × Rat := (4, 15)
def number_to_add : ℕ := 29

theorem new_ratio_second_term :
  let new_ratio := (original_ratio.1 + number_to_add, original_ratio.2 + number_to_add)
  new_ratio.2 = 44 := by sorry

end new_ratio_second_term_l2220_222095


namespace morning_campers_l2220_222089

theorem morning_campers (afternoon evening total : ℕ) 
  (h1 : afternoon = 13)
  (h2 : evening = 49)
  (h3 : total = 98)
  : total - afternoon - evening = 36 := by
  sorry

end morning_campers_l2220_222089


namespace triangle_longest_side_l2220_222039

theorem triangle_longest_side (x : ℝ) : 
  let side1 := x^2 + 1
  let side2 := x + 5
  let side3 := 3*x - 1
  (side1 + side2 + side3 = 40) →
  (max side1 (max side2 side3) = 26) :=
by sorry

end triangle_longest_side_l2220_222039


namespace fraction_equality_l2220_222016

theorem fraction_equality (x y z m : ℝ) 
  (h1 : 5 / (x + y) = m / (x + z)) 
  (h2 : m / (x + z) = 13 / (z - y)) : 
  m = 18 := by
  sorry

end fraction_equality_l2220_222016


namespace points_per_game_l2220_222042

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 81 → 
  num_games = 3 → 
  total_points = num_games * points_per_game → 
  points_per_game = 27 := by
sorry

end points_per_game_l2220_222042


namespace count_integers_between_cubes_l2220_222053

theorem count_integers_between_cubes : 
  ∃ (n : ℕ), n = 37 ∧ 
  (∀ k : ℤ, (11.1 : ℝ)^3 < k ∧ k < (11.2 : ℝ)^3 ↔ 
   (⌊(11.1 : ℝ)^3⌋ + 1 : ℤ) ≤ k ∧ k ≤ (⌊(11.2 : ℝ)^3⌋ : ℤ)) ∧
  n = ⌊(11.2 : ℝ)^3⌋ - ⌊(11.1 : ℝ)^3⌋ :=
by sorry

end count_integers_between_cubes_l2220_222053


namespace faire_percentage_calculation_dirk_faire_percentage_l2220_222075

/-- Calculates the percentage of revenue given to the faire for Dirk's amulet sales --/
theorem faire_percentage_calculation (days : Nat) (amulets_per_day : Nat) 
  (selling_price : Nat) (cost_price : Nat) (final_profit : Nat) : ℚ :=
  let total_amulets := days * amulets_per_day
  let revenue := total_amulets * selling_price
  let total_cost := total_amulets * cost_price
  let profit_before_fee := revenue - total_cost
  let faire_fee := profit_before_fee - final_profit
  (faire_fee : ℚ) / revenue * 100

/-- Proves that Dirk gave 10% of his revenue to the faire --/
theorem dirk_faire_percentage : 
  faire_percentage_calculation 2 25 40 30 300 = 10 := by
  sorry

end faire_percentage_calculation_dirk_faire_percentage_l2220_222075


namespace point_in_second_quadrant_l2220_222026

theorem point_in_second_quadrant (A B C : Real) (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) (h4 : A + B + C = π) :
  let P : ℝ × ℝ := (Real.cos B - Real.sin A, Real.sin B - Real.cos A)
  (P.1 < 0 ∧ P.2 > 0) := by sorry

end point_in_second_quadrant_l2220_222026


namespace work_completion_time_l2220_222002

/-- The number of days it takes to complete the remaining work after additional persons join -/
def remaining_days (initial_persons : ℕ) (total_days : ℕ) (days_worked : ℕ) (additional_persons : ℕ) : ℚ :=
  let initial_work_rate := 1 / (initial_persons * total_days : ℚ)
  let work_done := initial_persons * days_worked * initial_work_rate
  let remaining_work := 1 - work_done
  let new_work_rate := (initial_persons + additional_persons : ℚ) * initial_work_rate
  remaining_work / new_work_rate

theorem work_completion_time :
  remaining_days 12 18 6 4 = 12 := by
  sorry

end work_completion_time_l2220_222002


namespace same_terminal_side_angle_with_same_terminal_side_l2220_222068

theorem same_terminal_side (x y : Real) : 
  x = y + 2 * Real.pi * ↑(Int.floor ((x - y) / (2 * Real.pi))) → 
  ∃ k : ℤ, y = x + 2 * Real.pi * k := by
  sorry

theorem angle_with_same_terminal_side : 
  ∃ k : ℤ, -π/3 = 5*π/3 + 2*π*k := by
  sorry

end same_terminal_side_angle_with_same_terminal_side_l2220_222068


namespace kylie_piggy_bank_coins_l2220_222076

/-- The number of coins Kylie got from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie got from her brother -/
def brother_coins : ℕ := 13

/-- The number of coins Kylie got from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie gave to her friend Laura -/
def coins_given_away : ℕ := 21

/-- The number of coins Kylie was left with -/
def coins_left : ℕ := 15

/-- Theorem stating that the number of coins Kylie got from her piggy bank is 15 -/
theorem kylie_piggy_bank_coins :
  piggy_bank_coins = coins_left + coins_given_away - brother_coins - father_coins :=
by sorry

end kylie_piggy_bank_coins_l2220_222076


namespace laptop_sticker_price_is_750_l2220_222065

/-- The sticker price of the laptop -/
def sticker_price : ℝ := 750

/-- Store A's pricing strategy -/
def store_A_price (x : ℝ) : ℝ := 0.80 * x - 100

/-- Store B's pricing strategy -/
def store_B_price (x : ℝ) : ℝ := 0.70 * x

/-- The theorem stating that the sticker price is correct -/
theorem laptop_sticker_price_is_750 :
  store_B_price sticker_price - store_A_price sticker_price = 25 := by
  sorry

end laptop_sticker_price_is_750_l2220_222065


namespace sally_coin_problem_l2220_222037

/-- Represents the number and value of coins in Sally's bank -/
structure CoinBank where
  pennies : ℕ
  nickels : ℕ
  pennyValue : ℕ
  nickelValue : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (bank : CoinBank) : ℕ :=
  bank.pennies * bank.pennyValue + bank.nickels * bank.nickelValue

/-- Represents gifts of nickels -/
structure NickelGift where
  fromDad : ℕ
  fromMom : ℕ

theorem sally_coin_problem (initialBank : CoinBank) (gift : NickelGift) :
  initialBank.pennies = 8 ∧
  initialBank.nickels = 7 ∧
  initialBank.pennyValue = 1 ∧
  initialBank.nickelValue = 5 ∧
  gift.fromDad = 9 ∧
  gift.fromMom = 2 →
  let finalBank : CoinBank := {
    pennies := initialBank.pennies,
    nickels := initialBank.nickels + gift.fromDad + gift.fromMom,
    pennyValue := initialBank.pennyValue,
    nickelValue := initialBank.nickelValue
  }
  finalBank.nickels = 18 ∧ totalValue finalBank = 98 := by
  sorry

end sally_coin_problem_l2220_222037


namespace complete_square_sum_l2220_222007

theorem complete_square_sum (a b c : ℤ) : 
  (∀ x, 64 * x^2 + 48 * x - 36 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 56 := by
  sorry

end complete_square_sum_l2220_222007


namespace solve_temperature_problem_l2220_222019

def temperature_problem (temps : List ℝ) (avg : ℝ) : Prop :=
  temps.length = 6 ∧
  temps = [99.1, 98.2, 98.7, 99.3, 99.8, 99] ∧
  avg = 99 ∧
  ∃ (saturday_temp : ℝ),
    (temps.sum + saturday_temp) / 7 = avg ∧
    saturday_temp = 98.9

theorem solve_temperature_problem (temps : List ℝ) (avg : ℝ)
  (h : temperature_problem temps avg) : 
  ∃ (saturday_temp : ℝ), saturday_temp = 98.9 := by
  sorry

end solve_temperature_problem_l2220_222019


namespace sarah_eli_age_ratio_l2220_222063

/-- Given the ages and relationships between Kaylin, Sarah, Eli, and Freyja, 
    prove that the ratio of Sarah's age to Eli's age is 2:1 -/
theorem sarah_eli_age_ratio :
  ∀ (kaylin_age sarah_age eli_age freyja_age : ℕ),
    kaylin_age = 33 →
    freyja_age = 10 →
    sarah_age = kaylin_age + 5 →
    eli_age = freyja_age + 9 →
    ∃ (n : ℕ), sarah_age = n * eli_age →
    sarah_age / eli_age = 2 := by
  sorry

end sarah_eli_age_ratio_l2220_222063


namespace two_real_roots_iff_m_nonpositive_m_values_given_roots_relationship_l2220_222044

/-- Given a quadratic equation x^2 - 2x + m + 1 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 2*x + m + 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (-2)^2 - 4*(m + 1)

/-- The condition for two real roots -/
def has_two_real_roots (m : ℝ) : Prop :=
  discriminant m ≥ 0

/-- The relationship between the roots and m -/
def roots_relationship (x₁ x₂ m : ℝ) : Prop :=
  x₁ + 3*x₂ = 2*m + 8

/-- Theorem 1: The equation has two real roots iff m ≤ 0 -/
theorem two_real_roots_iff_m_nonpositive (m : ℝ) :
  has_two_real_roots m ↔ m ≤ 0 :=
sorry

/-- Theorem 2: If the roots satisfy the given relationship, then m = -1 or m = -2 -/
theorem m_values_given_roots_relationship (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ roots_relationship x₁ x₂ m) →
  (m = -1 ∨ m = -2) :=
sorry

end two_real_roots_iff_m_nonpositive_m_values_given_roots_relationship_l2220_222044


namespace alyssa_plums_count_l2220_222093

/-- The number of plums picked by Jason -/
def jason_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := 27

/-- The number of plums picked by Alyssa -/
def alyssa_plums : ℕ := total_plums - jason_plums

theorem alyssa_plums_count : alyssa_plums = 17 := by
  sorry

end alyssa_plums_count_l2220_222093


namespace vacation_cost_division_l2220_222050

theorem vacation_cost_division (total_cost : ℕ) (initial_people : ℕ) (cost_reduction : ℕ) (n : ℕ) : 
  total_cost = 360 →
  initial_people = 3 →
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  cost_reduction = 30 →
  n = 4 :=
by sorry

end vacation_cost_division_l2220_222050


namespace min_intersection_distance_l2220_222028

/-- The minimum distance between intersection points of a line and a circle --/
theorem min_intersection_distance (k : ℝ) : 
  let l := {(x, y) : ℝ × ℝ | y = k * x + 1}
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 3 = 0}
  ∃ (A B : ℝ × ℝ), A ∈ l ∧ A ∈ C ∧ B ∈ l ∧ B ∈ C ∧
    ∀ (P Q : ℝ × ℝ), P ∈ l ∧ P ∈ C ∧ Q ∈ l ∧ Q ∈ C →
      Real.sqrt 8 ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  sorry

end min_intersection_distance_l2220_222028


namespace midpoint_ratio_range_l2220_222032

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 2*y - 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y + 3 = 0

-- Define the midpoint condition
def is_midpoint (x₀ y₀ xp yp xq yq : ℝ) : Prop :=
  x₀ = (xp + xq) / 2 ∧ y₀ = (yp + yq) / 2

-- Main theorem
theorem midpoint_ratio_range 
  (P : ℝ × ℝ) (A : ℝ × ℝ) (Q : ℝ × ℝ) (x₀ y₀ : ℝ)
  (h1 : line1 P.1 P.2)
  (h2 : line2 A.1 A.2)
  (h3 : is_midpoint x₀ y₀ P.1 P.2 Q.1 Q.2)
  (h4 : y₀ > x₀ + 2) :
  -1/2 < y₀/x₀ ∧ y₀/x₀ < -1/5 :=
sorry

end midpoint_ratio_range_l2220_222032


namespace average_and_differences_l2220_222054

theorem average_and_differences (x : ℝ) : 
  (45 + x) / 2 = 38 → |x - 45| + |x - 30| = 15 := by
  sorry

end average_and_differences_l2220_222054


namespace units_digit_factorial_sum_2010_l2220_222020

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2010 :
  units_digit (factorial_sum 2010) = 3 := by sorry

end units_digit_factorial_sum_2010_l2220_222020


namespace expression_simplification_l2220_222098

theorem expression_simplification (x : ℝ) (h : x = 8) :
  (2 * x) / (x + 1) - ((2 * x + 4) / (x^2 - 1)) / ((x + 2) / (x^2 - 2*x + 1)) = 2 / 9 := by
  sorry

end expression_simplification_l2220_222098


namespace remainder_444_pow_444_mod_13_l2220_222066

theorem remainder_444_pow_444_mod_13 : 444^444 ≡ 1 [ZMOD 13] := by
  sorry

end remainder_444_pow_444_mod_13_l2220_222066


namespace g_inverse_equals_g_l2220_222090

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (k * x - 3)

theorem g_inverse_equals_g (k : ℝ) :
  k ≠ -4/3 →
  ∀ x : ℝ, g k (g k x) = x :=
sorry

end g_inverse_equals_g_l2220_222090


namespace symmetric_point_coordinates_l2220_222080

/-- Given two points are symmetric with respect to the origin -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  B.1 = -A.1 ∧ B.2 = -A.2

theorem symmetric_point_coordinates :
  let A : ℝ × ℝ := (-2, 1)
  let B : ℝ × ℝ := (2, -1)
  symmetric_wrt_origin A B → B = (2, -1) := by
  sorry

end symmetric_point_coordinates_l2220_222080


namespace largest_multiple_of_daytona_sharks_l2220_222043

def daytona_sharks : ℕ := 12
def cape_may_sharks : ℕ := 32

theorem largest_multiple_of_daytona_sharks : 
  ∃ (m : ℕ), m * daytona_sharks < cape_may_sharks ∧ 
  ∀ (n : ℕ), n * daytona_sharks < cape_may_sharks → n ≤ m ∧ 
  m = 2 :=
sorry

end largest_multiple_of_daytona_sharks_l2220_222043


namespace kids_at_camp_l2220_222000

theorem kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) (h1 : total_kids = 1059955) (h2 : kids_at_home = 495718) :
  total_kids - kids_at_home = 564237 := by
  sorry

end kids_at_camp_l2220_222000


namespace sine_inequality_l2220_222058

theorem sine_inequality (x : ℝ) : 
  (9.2894 * Real.sin x * Real.sin (2 * x) * Real.sin (3 * x) > Real.sin (4 * x)) ↔ 
  (∃ n : ℤ, (-π/8 + π * n < x ∧ x < π * n) ∨ 
            (π/8 + π * n < x ∧ x < 3*π/8 + π * n) ∨ 
            (π/2 + π * n < x ∧ x < 5*π/8 + π * n)) := by
  sorry

end sine_inequality_l2220_222058


namespace quadratic_inequality_max_value_l2220_222097

theorem quadratic_inequality_max_value (a b c : ℝ) (ha : a > 0) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧ 
    ∀ k : ℝ, k * (a^2 + c^2) ≤ b^2 → k ≤ M) :=
by sorry

end quadratic_inequality_max_value_l2220_222097


namespace yue_bao_scientific_notation_l2220_222082

theorem yue_bao_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    (1853 * 1000000000 : ℝ) = a * (10 : ℝ) ^ n ∧
    a = 1.853 ∧ n = 11 := by
  sorry

end yue_bao_scientific_notation_l2220_222082


namespace unique_twisty_divisible_by_12_l2220_222073

/-- A function that checks if a number is twisty -/
def is_twisty (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
  n = a * 10000 + b * 1000 + a * 100 + b * 10 + a

/-- A function that checks if a number is five digits long -/
def is_five_digit (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

/-- The main theorem -/
theorem unique_twisty_divisible_by_12 : 
  ∃! (n : ℕ), is_twisty n ∧ is_five_digit n ∧ n % 12 = 0 :=
sorry

end unique_twisty_divisible_by_12_l2220_222073


namespace inequality_proof_l2220_222040

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  (a/b + a/c + b/a + b/c + c/a + c/b + 6) ≥ 2 * Real.sqrt 2 * (Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ∧
  ((a/b + a/c + b/a + b/c + c/a + c/b + 6) = 2 * Real.sqrt 2 * (Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end inequality_proof_l2220_222040


namespace negation_of_conjunction_l2220_222074

theorem negation_of_conjunction (p q : Prop) : ¬(p ∧ q) ↔ (¬p ∨ ¬q) := by
  sorry

end negation_of_conjunction_l2220_222074


namespace max_expression_c_value_l2220_222067

theorem max_expression_c_value (a b c : ℕ) : 
  a ∈ ({1, 2, 4} : Set ℕ) →
  b ∈ ({1, 2, 4} : Set ℕ) →
  c ∈ ({1, 2, 4} : Set ℕ) →
  a ≠ b → b ≠ c → a ≠ c →
  (∀ x y z : ℕ, x ∈ ({1, 2, 4} : Set ℕ) → y ∈ ({1, 2, 4} : Set ℕ) → z ∈ ({1, 2, 4} : Set ℕ) →
    x ≠ y → y ≠ z → x ≠ z → (x / 2) / (y / z : ℚ) ≤ (a / 2) / (b / c : ℚ)) →
  (a / 2) / (b / c : ℚ) = 4 →
  c = 2 := by sorry

end max_expression_c_value_l2220_222067
