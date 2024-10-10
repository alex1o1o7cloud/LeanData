import Mathlib

namespace product_real_implies_b_value_l1390_139020

theorem product_real_implies_b_value (z₁ z₂ : ℂ) (b : ℝ) :
  z₁ = 1 + I →
  z₂ = 2 + b * I →
  (z₁ * z₂).im = 0 →
  b = -2 := by
sorry

end product_real_implies_b_value_l1390_139020


namespace donation_distribution_l1390_139011

/-- Calculates the amount each organization receives when a company donates a portion of its funds to a foundation with multiple organizations. -/
theorem donation_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) 
  (h1 : total_amount = 2500)
  (h2 : donation_percentage = 80 / 100)
  (h3 : num_organizations = 8) :
  (total_amount * donation_percentage) / num_organizations = 250 := by
  sorry

end donation_distribution_l1390_139011


namespace tangent_equation_solution_l1390_139064

theorem tangent_equation_solution (x : Real) :
  (Real.tan x * Real.tan (20 * π / 180) + 
   Real.tan (20 * π / 180) * Real.tan (40 * π / 180) + 
   Real.tan (40 * π / 180) * Real.tan x = 1) ↔
  (∃ k : ℤ, x = (30 + 180 * k) * π / 180) :=
by sorry

end tangent_equation_solution_l1390_139064


namespace heidi_painting_fraction_l1390_139022

/-- If a person can paint a wall in a given time, this function calculates
    the fraction of the wall they can paint in a shorter time. -/
def fractionPainted (totalTime minutes : ℕ) : ℚ :=
  minutes / totalTime

/-- Theorem stating that if Heidi can paint a wall in 60 minutes,
    she can paint 1/5 of the wall in 12 minutes. -/
theorem heidi_painting_fraction :
  fractionPainted 60 12 = 1 / 5 := by sorry

end heidi_painting_fraction_l1390_139022


namespace solve_equation_l1390_139002

theorem solve_equation (a : ℚ) (h : 2 * a + 2 * a / 4 = 4) : a = 8 / 5 := by
  sorry

end solve_equation_l1390_139002


namespace final_racers_count_l1390_139096

def race_elimination (initial_racers : ℕ) : ℕ :=
  let after_first := initial_racers - 10
  let after_second := after_first - (after_first / 3)
  let after_third := after_second - (after_second / 4)
  let after_fourth := after_third - (after_third / 3)
  let after_fifth := after_fourth - (after_fourth / 2)
  after_fifth - (after_fifth * 3 / 4)

theorem final_racers_count :
  race_elimination 200 = 8 := by
  sorry

end final_racers_count_l1390_139096


namespace function_properties_l1390_139029

/-- Given a function f(x) = ax - bx^2 where a and b are positive real numbers,
    this theorem states two properties:
    1. If f(x) ≤ 1 for all real x, then a ≤ 2√b.
    2. When b > 1, for x in [0, 1], |f(x)| ≤ 1 if and only if b - 1 ≤ a ≤ 2√b. -/
theorem function_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x : ℝ => a * x - b * x^2
  (∀ x, f x ≤ 1) → a ≤ 2 * Real.sqrt b ∧
  (b > 1 → (∀ x ∈ Set.Icc 0 1, |f x| ≤ 1) ↔ b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
by sorry

end function_properties_l1390_139029


namespace segment_length_ratio_l1390_139089

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PS and PQ = 8QR,
    the length of segment RS is 5/8 of the length of PQ. -/
theorem segment_length_ratio (P Q R S : Real) 
  (h1 : P ≤ R) (h2 : R ≤ S) (h3 : S ≤ Q)  -- Points order on the line
  (h4 : Q - P = 4 * (S - P))  -- PQ = 4PS
  (h5 : Q - P = 8 * (Q - R))  -- PQ = 8QR
  : S - R = 5/8 * (Q - P) := by
  sorry

end segment_length_ratio_l1390_139089


namespace solution_set_f_geq_0_max_value_f_range_of_m_l1390_139015

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 20| - |16 - x|

-- Theorem for the solution set of f(x) ≥ 0
theorem solution_set_f_geq_0 : 
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≥ -2} := by sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : 
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 36 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), f x ≥ m) ↔ m ≤ 36 := by sorry

end solution_set_f_geq_0_max_value_f_range_of_m_l1390_139015


namespace odds_to_probability_losing_l1390_139088

-- Define the odds of winning
def odds_winning : ℚ := 5 / 6

-- Define the probability of losing
def prob_losing : ℚ := 6 / 11

-- Theorem statement
theorem odds_to_probability_losing : 
  odds_winning = 5 / 6 → prob_losing = 6 / 11 := by
  sorry

end odds_to_probability_losing_l1390_139088


namespace lucy_grocery_shopping_l1390_139003

theorem lucy_grocery_shopping (total_packs noodle_packs : ℕ) 
  (h1 : total_packs = 28)
  (h2 : noodle_packs = 16)
  (h3 : ∃ cookie_packs : ℕ, total_packs = cookie_packs + noodle_packs) :
  ∃ cookie_packs : ℕ, cookie_packs = 12 ∧ total_packs = cookie_packs + noodle_packs :=
by
  sorry

end lucy_grocery_shopping_l1390_139003


namespace quadratic_rational_solutions_product_l1390_139042

theorem quadratic_rational_solutions_product : ∃ (c₁ c₂ : ℕ+), 
  (∀ (c : ℕ+), (∃ (x : ℚ), 5 * x^2 + 11 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
  c₁.val * c₂.val = 12 := by
  sorry

end quadratic_rational_solutions_product_l1390_139042


namespace prob_draw_star_is_one_sixth_l1390_139008

/-- A deck of cards with multiple suits and ranks -/
structure Deck :=
  (num_suits : ℕ)
  (num_ranks : ℕ)

/-- The probability of drawing a specific suit from a deck -/
def prob_draw_suit (d : Deck) : ℚ :=
  1 / d.num_suits

/-- Theorem: The probability of drawing a ★ card from a deck with 6 suits and 13 ranks is 1/6 -/
theorem prob_draw_star_is_one_sixth (d : Deck) 
  (h_suits : d.num_suits = 6)
  (h_ranks : d.num_ranks = 13) :
  prob_draw_suit d = 1 / 6 := by
  sorry

end prob_draw_star_is_one_sixth_l1390_139008


namespace train_delay_l1390_139016

/-- Proves that a train moving at 4/5 of its usual speed will be 30 minutes late on a journey that usually takes 2 hours -/
theorem train_delay (usual_speed : ℝ) (usual_time : ℝ) (h1 : usual_time = 2) :
  let reduced_speed := (4/5 : ℝ) * usual_speed
  let reduced_time := usual_time * (5/4 : ℝ)
  reduced_time - usual_time = 1/2 := by sorry

#check train_delay

end train_delay_l1390_139016


namespace sin_2theta_value_l1390_139079

theorem sin_2theta_value (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/3) : 
  Real.sin (2 * θ) = -8/9 := by
  sorry

end sin_2theta_value_l1390_139079


namespace distance_between_points_l1390_139082

/-- The distance between two points (-3, 5) and (4, -9) is √245 -/
theorem distance_between_points : Real.sqrt 245 = Real.sqrt ((4 - (-3))^2 + (-9 - 5)^2) := by
  sorry

end distance_between_points_l1390_139082


namespace sum_of_roots_quadratic_l1390_139068

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (a^2 - 8*a + 5 = 0) → (b^2 - 8*b + 5 = 0) → (a + b = 8) := by
  sorry

end sum_of_roots_quadratic_l1390_139068


namespace square_root_problem_l1390_139056

theorem square_root_problem (n : ℝ) (h : Real.sqrt (9 + n) = 8) : n + 2 = 57 := by
  sorry

end square_root_problem_l1390_139056


namespace cubic_decreasing_l1390_139004

-- Define the function f(x) = mx³ - x
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

-- State the theorem
theorem cubic_decreasing (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x ≥ f m y) ↔ m < 0 := by
  sorry

end cubic_decreasing_l1390_139004


namespace root_problems_l1390_139001

theorem root_problems :
  (∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2) ∧
  (∃ x y : ℝ, x^2 = 5 ∧ y^2 = 5 ∧ x = -y ∧ x ≠ 0) ∧
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) :=
by sorry

end root_problems_l1390_139001


namespace quadratic_sum_l1390_139013

/-- A quadratic function passing through specific points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (QuadraticFunction a b c 0 = 7) →
  (QuadraticFunction a b c 1 = 4) →
  a + b + 2*c = 11 := by
  sorry

end quadratic_sum_l1390_139013


namespace calculate_principal_l1390_139031

/-- Given a simple interest, interest rate, and time period, calculate the principal amount. -/
theorem calculate_principal (simple_interest rate time : ℝ) :
  simple_interest = 4020.75 →
  rate = 0.0875 →
  time = 5.5 →
  simple_interest = (8355.00 * rate * time) := by
  sorry

end calculate_principal_l1390_139031


namespace set_equality_from_intersection_union_equality_l1390_139084

theorem set_equality_from_intersection_union_equality (A : Set α) :
  ∃ X, (X ∩ A = X ∪ A) → (X = A) := by sorry

end set_equality_from_intersection_union_equality_l1390_139084


namespace log_difference_times_sqrt10_l1390_139007

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_times_sqrt10 :
  (log10 (1/4) - log10 25) * (10 ^ (1/2 : ℝ)) = -10 := by
  sorry

end log_difference_times_sqrt10_l1390_139007


namespace nanguo_pear_profit_l1390_139058

/-- Represents the weight difference from standard and the number of boxes for each difference -/
structure WeightDifference :=
  (difference : ℚ)
  (numBoxes : ℕ)

/-- Calculates the total profit from selling Nanguo pears -/
def calculateProfit (
  numBoxes : ℕ)
  (standardWeight : ℚ)
  (weightDifferences : List WeightDifference)
  (purchasePrice : ℚ)
  (highSellPrice : ℚ)
  (lowSellPrice : ℚ)
  (highSellProportion : ℚ) : ℚ :=
  sorry

theorem nanguo_pear_profit :
  let numBoxes : ℕ := 50
  let standardWeight : ℚ := 10
  let weightDifferences : List WeightDifference := [
    ⟨-2/10, 12⟩, ⟨-1/10, 3⟩, ⟨0, 3⟩, ⟨1/10, 7⟩, ⟨2/10, 15⟩, ⟨3/10, 10⟩
  ]
  let purchasePrice : ℚ := 4
  let highSellPrice : ℚ := 10
  let lowSellPrice : ℚ := 3/2
  let highSellProportion : ℚ := 3/5
  calculateProfit numBoxes standardWeight weightDifferences purchasePrice highSellPrice lowSellPrice highSellProportion = 27216/10
  := by sorry

end nanguo_pear_profit_l1390_139058


namespace BH_length_l1390_139024

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CA := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB = 5 ∧ BC = 7 ∧ CA = 8

-- Define points G and H on ray AB
def points_on_ray (A B G H : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ > 1 ∧ t₂ > t₁ ∧
  G = (A.1 + t₁ * (B.1 - A.1), A.2 + t₁ * (B.2 - A.2)) ∧
  H = (A.1 + t₂ * (B.1 - A.1), A.2 + t₂ * (B.2 - A.2))

-- Define point I on the intersection of circumcircles
def point_on_circumcircles (A B C G H I : ℝ × ℝ) : Prop :=
  I ≠ C ∧
  ∃ r₁ r₂ : ℝ,
    (I.1 - A.1)^2 + (I.2 - A.2)^2 = r₁^2 ∧
    (G.1 - A.1)^2 + (G.2 - A.2)^2 = r₁^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = r₁^2 ∧
    (I.1 - B.1)^2 + (I.2 - B.2)^2 = r₂^2 ∧
    (H.1 - B.1)^2 + (H.2 - B.2)^2 = r₂^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = r₂^2

-- Define distances GI and HI
def distances (G H I : ℝ × ℝ) : Prop :=
  Real.sqrt ((G.1 - I.1)^2 + (G.2 - I.2)^2) = 3 ∧
  Real.sqrt ((H.1 - I.1)^2 + (H.2 - I.2)^2) = 8

-- Main theorem
theorem BH_length (A B C G H I : ℝ × ℝ) :
  triangle_ABC A B C →
  points_on_ray A B G H →
  point_on_circumcircles A B C G H I →
  distances G H I →
  Real.sqrt ((B.1 - H.1)^2 + (B.2 - H.2)^2) = (6 + 47 * Real.sqrt 2) / 9 := by
  sorry

end BH_length_l1390_139024


namespace cost_price_calculation_l1390_139045

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 21000 →
  discount_rate = 0.10 →
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 17500 := by
sorry

end cost_price_calculation_l1390_139045


namespace tangent_line_at_one_l1390_139091

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-2)*x^2 + a*x - 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a-2)*x + a

/-- Theorem: If f'(x) is even, then the tangent line at (1, f(1)) is 5x - y - 3 = 0 -/
theorem tangent_line_at_one (a : ℝ) :
  (∀ x, f' a x = f' a (-x)) →
  ∃ m b, ∀ x y, y = m*x + b ↔ y - f a 1 = (f' a 1) * (x - 1) :=
by sorry

end tangent_line_at_one_l1390_139091


namespace product_of_numbers_l1390_139063

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 + y^2 = 200) : x * y = 100 := by
  sorry

end product_of_numbers_l1390_139063


namespace james_candy_payment_l1390_139092

/-- Proves that James paid $20 for candy given the conditions of the problem -/
theorem james_candy_payment (
  num_packs : ℕ)
  (price_per_pack : ℕ)
  (change_received : ℕ)
  (h1 : num_packs = 3)
  (h2 : price_per_pack = 3)
  (h3 : change_received = 11)
  : num_packs * price_per_pack + change_received = 20 := by
  sorry

end james_candy_payment_l1390_139092


namespace probability_of_two_red_books_l1390_139027

-- Define the number of red and blue books
def red_books : ℕ := 4
def blue_books : ℕ := 4
def total_books : ℕ := red_books + blue_books

-- Define the number of books to be selected
def books_selected : ℕ := 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem probability_of_two_red_books :
  (combination red_books books_selected : ℚ) / (combination total_books books_selected) = 3 / 14 := by
  sorry

end probability_of_two_red_books_l1390_139027


namespace line_BC_equation_triangle_ABC_area_l1390_139041

-- Define the points of the triangle
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (5, -2)
def C : ℝ × ℝ := (3, 5)

-- Define the line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := { A := A, B := B, C := C }

-- Theorem for the equation of line BC
theorem line_BC_equation (t : Triangle) (l : Line) : 
  t = ABC → l.a = 7 ∧ l.b = 2 ∧ l.c = -31 → 
  l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
  l.a * t.C.1 + l.b * t.C.2 + l.c = 0 :=
sorry

-- Theorem for the area of triangle ABC
theorem triangle_ABC_area (t : Triangle) : 
  t = ABC → (1/2) * |t.A.1 * (t.B.2 - t.C.2) + t.B.1 * (t.C.2 - t.A.2) + t.C.1 * (t.A.2 - t.B.2)| = 29/2 :=
sorry

end line_BC_equation_triangle_ABC_area_l1390_139041


namespace problem_1_l1390_139017

theorem problem_1 : 99 * (118 + 4/5) + 99 * (-1/5) - 99 * (18 + 3/5) = 9900 := by
  sorry

end problem_1_l1390_139017


namespace polar_to_rectangular_transformation_l1390_139078

/-- Given a point with rectangular coordinates (8, 6) and polar coordinates (r, θ),
    prove that the point with polar coordinates (r³, 3π/2 * θ) has rectangular
    coordinates (-600, -800). -/
theorem polar_to_rectangular_transformation (r θ : ℝ) :
  r * Real.cos θ = 8 ∧ r * Real.sin θ = 6 →
  (r^3 * Real.cos ((3 * Real.pi / 2) * θ) = -600) ∧
  (r^3 * Real.sin ((3 * Real.pi / 2) * θ) = -800) :=
by sorry

end polar_to_rectangular_transformation_l1390_139078


namespace same_color_probability_problem_die_l1390_139055

/-- Represents a 30-sided die with colored sides. -/
structure ColoredDie :=
  (maroon : ℕ)
  (teal : ℕ)
  (cyan : ℕ)
  (sparkly : ℕ)
  (total_sides : ℕ)
  (sum_equals_total : maroon + teal + cyan + sparkly = total_sides)

/-- Calculates the probability of rolling the same color on two identical dice. -/
def same_color_probability (die : ColoredDie) : ℚ :=
  let maroon_prob := (die.maroon : ℚ) / die.total_sides
  let teal_prob := (die.teal : ℚ) / die.total_sides
  let cyan_prob := (die.cyan : ℚ) / die.total_sides
  let sparkly_prob := (die.sparkly : ℚ) / die.total_sides
  maroon_prob ^ 2 + teal_prob ^ 2 + cyan_prob ^ 2 + sparkly_prob ^ 2

/-- The specific 30-sided die described in the problem. -/
def problem_die : ColoredDie :=
  { maroon := 5
    teal := 10
    cyan := 12
    sparkly := 3
    total_sides := 30
    sum_equals_total := by simp }

/-- Theorem stating that the probability of rolling the same color
    on two problem_die is 139/450. -/
theorem same_color_probability_problem_die :
  same_color_probability problem_die = 139 / 450 := by
  sorry

#eval same_color_probability problem_die

end same_color_probability_problem_die_l1390_139055


namespace geometric_sequence_general_term_l1390_139094

/-- Geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 4^(n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → ∀ n : ℕ, a n = general_term n := by sorry

end geometric_sequence_general_term_l1390_139094


namespace ivan_fate_l1390_139086

structure Animal where
  name : String
  always_truth : Bool
  alternating : Bool
  deriving Repr

def Statement := (Bool × Bool)

theorem ivan_fate (bear fox wolf : Animal)
  (h_bear : bear.always_truth = true ∧ bear.alternating = false)
  (h_fox : fox.always_truth = false ∧ fox.alternating = false)
  (h_wolf : wolf.always_truth = false ∧ wolf.alternating = true)
  (statement1 statement2 statement3 : Statement)
  (h_distinct : bear ≠ fox ∧ bear ≠ wolf ∧ fox ≠ wolf)
  : ∃ (animal1 animal2 animal3 : Animal),
    animal1 = fox ∧ animal2 = wolf ∧ animal3 = bear ∧
    (¬statement1.1 ∧ ¬statement1.2) ∧
    (statement2.1 ∧ ¬statement2.2) ∧
    (statement3.1 ∧ statement3.2) :=
by sorry

#check ivan_fate

end ivan_fate_l1390_139086


namespace local_tax_deduction_l1390_139005

/-- Proves that given an hourly wage of 25 dollars and a 2% local tax rate, 
    the amount deducted for local taxes is 50 cents per hour. -/
theorem local_tax_deduction (hourly_wage : ℝ) (tax_rate : ℝ) :
  hourly_wage = 25 ∧ tax_rate = 0.02 →
  (hourly_wage * tax_rate * 100 : ℝ) = 50 := by
  sorry

#check local_tax_deduction

end local_tax_deduction_l1390_139005


namespace john_remaining_money_l1390_139000

def base7_to_base10 (n : ℕ) : ℕ :=
  6 * 7^3 + 5 * 7^2 + 3 * 7^1 + 4 * 7^0

theorem john_remaining_money :
  let savings : ℕ := base7_to_base10 6534
  let ticket_cost : ℕ := 1200
  savings - ticket_cost = 1128 := by sorry

end john_remaining_money_l1390_139000


namespace cross_shaded_area_equality_l1390_139040

-- Define the rectangle and shaded area properties
def rectangle_length : ℝ := 9
def rectangle_width : ℝ := 8
def shaded_rect1_width : ℝ := 3

-- Define the shaded area as a function of X
def shaded_area (x : ℝ) : ℝ :=
  shaded_rect1_width * rectangle_width + rectangle_length * x - shaded_rect1_width * x

-- Define the total area of the rectangle
def total_area : ℝ := rectangle_length * rectangle_width

-- State the theorem
theorem cross_shaded_area_equality (x : ℝ) :
  shaded_area x = (1 / 2) * total_area → x = 2 := by sorry

end cross_shaded_area_equality_l1390_139040


namespace power_five_137_mod_8_l1390_139065

theorem power_five_137_mod_8 : 5^137 % 8 = 5 := by
  sorry

end power_five_137_mod_8_l1390_139065


namespace expression_value_l1390_139053

theorem expression_value (x : ℝ) (h : x = 4) : 3 * (3 * x - 2)^2 = 300 := by
  sorry

end expression_value_l1390_139053


namespace vertical_asymptotes_sum_l1390_139032

theorem vertical_asymptotes_sum (p q : ℚ) : 
  (∀ x, 4 * x^2 + 7 * x + 3 = 0 ↔ x = p ∨ x = q) →
  p + q = -7/4 := by
  sorry

end vertical_asymptotes_sum_l1390_139032


namespace least_addition_for_divisibility_l1390_139069

theorem least_addition_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ (625573 + k) % 3 = 0) → 
  (625573 + 2) % 3 = 0 ∧ ∀ m : ℕ, m < 2 → (625573 + m) % 3 ≠ 0 :=
by sorry

end least_addition_for_divisibility_l1390_139069


namespace corner_cut_pentagon_area_corner_cut_pentagon_area_is_804_l1390_139071

/-- 
  Represents a pentagon formed by cutting a triangular corner from a rectangle.
  The sides of the pentagon have lengths 12, 15, 18, 30, and 34 in some order.
-/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {12, 15, 18, 30, 34}

/-- The area of the CornerCutPentagon is 804. -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ℕ :=
  804

/-- Proves that the area of the CornerCutPentagon is indeed 804. -/
theorem corner_cut_pentagon_area_is_804 (p : CornerCutPentagon) : 
  corner_cut_pentagon_area p = 804 := by
  sorry

#check corner_cut_pentagon_area_is_804

end corner_cut_pentagon_area_corner_cut_pentagon_area_is_804_l1390_139071


namespace smallest_number_l1390_139039

theorem smallest_number (a b c : ℝ) (ha : a = -0.5) (hb : b = 3) (hc : c = -2) :
  min a (min b c) = c := by sorry

end smallest_number_l1390_139039


namespace first_month_sale_proof_l1390_139051

/-- Calculates the sale in the first month given the sales for months 2 to 6 and the average sale for 6 months. -/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- Proves that the sale in the first month is 8435 given the specified conditions. -/
theorem first_month_sale_proof :
  first_month_sale 8927 8855 9230 8562 6991 8500 = 8435 := by
  sorry

end first_month_sale_proof_l1390_139051


namespace imaginary_sum_zero_l1390_139098

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum_zero :
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 :=
by
  sorry

end imaginary_sum_zero_l1390_139098


namespace cost_per_bag_l1390_139099

def num_friends : ℕ := 3
def num_bags : ℕ := 5
def payment_per_friend : ℚ := 5

theorem cost_per_bag : 
  (num_friends * payment_per_friend) / num_bags = 3 := by
  sorry

end cost_per_bag_l1390_139099


namespace quadratic_trinomial_negative_l1390_139077

theorem quadratic_trinomial_negative (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 7 * x + 4 * a < 0) ↔ a < -7/4 := by sorry

end quadratic_trinomial_negative_l1390_139077


namespace simplified_inverse_sum_l1390_139043

theorem simplified_inverse_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ((1 / x^3) + (1 / y^3) + (1 / z^3))⁻¹ = (x^3 * y^3 * z^3) / (y^3 * z^3 + x^3 * z^3 + x^3 * y^3) := by
  sorry

end simplified_inverse_sum_l1390_139043


namespace parallel_line_through_point_desired_line_equation_l1390_139034

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point (P : ℝ × ℝ) (l : Line) :
  ∃ (l' : Line), parallel l' l ∧ on_line P.1 P.2 l' :=
by sorry

theorem desired_line_equation (P : ℝ × ℝ) (l l' : Line) :
  P = (-1, 3) →
  l = Line.mk 1 (-2) 3 →
  parallel l' l →
  on_line P.1 P.2 l' →
  l' = Line.mk 1 (-2) 7 :=
by sorry

end parallel_line_through_point_desired_line_equation_l1390_139034


namespace repeating_decimal_is_rational_l1390_139067

def repeating_decimal (a b c : ℕ) : ℚ :=
  a + b / (10^c.succ * 99)

theorem repeating_decimal_is_rational (a b c : ℕ) :
  ∃ (p q : ℤ), repeating_decimal a b c = p / q ∧ q ≠ 0 :=
sorry

end repeating_decimal_is_rational_l1390_139067


namespace divisibility_implies_equality_l1390_139009

theorem divisibility_implies_equality (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h_div : (a^2 + a*b + 1) % (b^2 + a*b + 1) = 0) : a = b := by
  sorry

end divisibility_implies_equality_l1390_139009


namespace f_properties_l1390_139081

def f (x : ℝ) : ℝ := x^3 + 3*x^2

theorem f_properties :
  (f (-1) = 2) →
  (deriv f (-1) = -3) →
  (∃ (y : ℝ), y ∈ Set.Icc (-16) 4 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-4) 0 ∧ f x = y) ∧
  (∀ (t : ℝ), (∀ (x y : ℝ), t ≤ x ∧ x < y ∧ y ≤ t + 1 → f x > f y) ↔ t ∈ Set.Icc (-2) (-1)) :=
by sorry

end f_properties_l1390_139081


namespace flag_combinations_l1390_139066

def available_colors : ℕ := 6
def stripes : ℕ := 3

theorem flag_combinations : (available_colors * (available_colors - 1) * (available_colors - 2)) = 120 := by
  sorry

end flag_combinations_l1390_139066


namespace sugar_consumption_reduction_l1390_139046

theorem sugar_consumption_reduction (initial_price new_price : ℚ) 
  (h1 : initial_price = 10)
  (h2 : new_price = 13) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 300 / 13 := by
sorry

end sugar_consumption_reduction_l1390_139046


namespace correct_purchase_combinations_l1390_139033

/-- The number of oreo flavors -/
def oreo_flavors : ℕ := 7

/-- The number of milk flavors -/
def milk_flavors : ℕ := 4

/-- The total number of product flavors -/
def total_flavors : ℕ := oreo_flavors + milk_flavors

/-- The total number of products they purchase -/
def total_products : ℕ := 4

/-- The number of ways Alpha and Beta could have left the store with 4 products collectively -/
def purchase_combinations : ℕ := sorry

theorem correct_purchase_combinations :
  purchase_combinations = 4054 := by sorry

end correct_purchase_combinations_l1390_139033


namespace point_above_x_axis_l1390_139070

theorem point_above_x_axis (a : ℝ) : 
  (a > 0) → (a = Real.sqrt 3) → ∃ (x y : ℝ), x = -2 ∧ y = a ∧ y > 0 :=
by sorry

end point_above_x_axis_l1390_139070


namespace speed_conversion_l1390_139093

/-- Proves that a speed of 36.003 km/h is equivalent to 10.0008 meters per second. -/
theorem speed_conversion (speed_kmh : ℝ) (speed_ms : ℝ) : 
  speed_kmh = 36.003 ∧ speed_ms = 10.0008 → speed_kmh * (1000 / 3600) = speed_ms := by
  sorry

end speed_conversion_l1390_139093


namespace sandwich_theorem_l1390_139018

def sandwich_problem (david_spent : ℝ) (ben_spent : ℝ) : Prop :=
  ben_spent = 1.5 * david_spent ∧
  david_spent = ben_spent - 15 ∧
  david_spent + ben_spent = 75

theorem sandwich_theorem :
  ∃ (david_spent : ℝ) (ben_spent : ℝ), sandwich_problem david_spent ben_spent :=
by
  sorry

end sandwich_theorem_l1390_139018


namespace min_unhappiness_theorem_l1390_139062

/-- Represents the unhappiness levels of students -/
def unhappiness_levels : List ℝ := List.range 2017

/-- The number of groups to split the students into -/
def num_groups : ℕ := 15

/-- Calculates the minimum possible sum of average unhappiness levels -/
def min_unhappiness (levels : List ℝ) (groups : ℕ) : ℝ :=
  sorry

/-- The theorem stating the minimum unhappiness of the class -/
theorem min_unhappiness_theorem :
  min_unhappiness unhappiness_levels num_groups = 1120.5 := by
  sorry

end min_unhappiness_theorem_l1390_139062


namespace arithmetic_sequence_sum_2_to_20_l1390_139047

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_2_to_20 :
  arithmetic_sequence_sum 2 2 10 = 110 := by
  sorry

end arithmetic_sequence_sum_2_to_20_l1390_139047


namespace max_expected_games_max_at_half_l1390_139076

/-- The expected number of games in a best-of-five series -/
def f (p : ℝ) : ℝ := 6 * p^4 - 12 * p^3 + 3 * p^2 + 3 * p + 3

/-- The theorem stating the maximum value of f(p) -/
theorem max_expected_games :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → f p ≤ 33/8 :=
by
  sorry

/-- The theorem stating that the maximum is achieved at p = 1/2 -/
theorem max_at_half :
  f (1/2) = 33/8 :=
by
  sorry

end max_expected_games_max_at_half_l1390_139076


namespace complement_of_M_l1390_139090

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 3, 5}

theorem complement_of_M : Mᶜ = {2, 4, 6} := by sorry

end complement_of_M_l1390_139090


namespace trap_existence_for_specific_feeders_l1390_139097

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A feeder is an interval that contains infinitely many terms of the sequence. -/
def IsFeeder (s : Sequence) (a b : ℝ) : Prop :=
  ∀ n : ℕ, ∃ m ≥ n, a ≤ s m ∧ s m ≤ b

/-- A trap is an interval that contains all but finitely many terms of the sequence. -/
def IsTrap (s : Sequence) (a b : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n ≥ N, a ≤ s n ∧ s n ≤ b

/-- Main theorem about traps in sequences with specific feeders. -/
theorem trap_existence_for_specific_feeders (s : Sequence) 
  (h1 : IsFeeder s 0 1) (h2 : IsFeeder s 9 10) : 
  (¬ ∃ a : ℝ, IsTrap s a (a + 1)) ∧
  (∃ a : ℝ, IsTrap s a (a + 9)) := by sorry


end trap_existence_for_specific_feeders_l1390_139097


namespace root_equation_m_value_l1390_139030

theorem root_equation_m_value :
  ∀ m : ℝ, ((-4)^2 + m * (-4) - 20 = 0) → m = -1 := by
  sorry

end root_equation_m_value_l1390_139030


namespace series_sum_l1390_139049

noncomputable def series_term (n : ℕ) : ℝ :=
  (2^n : ℝ) / (3^(2^n) + 1)

theorem series_sum : ∑' (n : ℕ), series_term n = (1 : ℝ) / 2 := by
  sorry

end series_sum_l1390_139049


namespace odd_square_octal_property_l1390_139023

theorem odd_square_octal_property (n : ℤ) : 
  ∃ (m : ℤ), (2*n + 1)^2 % 8 = 1 ∧ ((2*n + 1)^2 - 1) / 8 = m * (m + 1) / 2 :=
sorry

end odd_square_octal_property_l1390_139023


namespace correct_result_l1390_139075

def add_subtract_round (a b c : ℕ) : ℕ :=
  let sum := a + b - c
  (sum + 5) / 10 * 10

theorem correct_result : add_subtract_round 53 28 5 = 80 := by
  sorry

end correct_result_l1390_139075


namespace polynomial_square_l1390_139010

theorem polynomial_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end polynomial_square_l1390_139010


namespace quadratic_roots_theorem_l1390_139019

/-- A quadratic equation with parameter m -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*(m+1)*x + m^2 + 5

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := 8*m - 16

/-- The condition for real roots -/
def has_real_roots (m : ℝ) : Prop := discriminant m ≥ 0

/-- The relation between roots and m -/
def roots_relation (m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ (x₁ - 1)*(x₂ - 1) = 28

theorem quadratic_roots_theorem (m : ℝ) :
  has_real_roots m →
  (∃ x₁ x₂, roots_relation m x₁ x₂) →
  m ≥ 2 ∧ m = 6 := by sorry

end quadratic_roots_theorem_l1390_139019


namespace shirt_price_reduction_l1390_139057

theorem shirt_price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_reduction := 0.9 * original_price
  let second_reduction := 0.9 * first_reduction
  second_reduction = 0.81 * original_price :=
by sorry

end shirt_price_reduction_l1390_139057


namespace sqrt_square_eq_x_for_nonnegative_l1390_139073

theorem sqrt_square_eq_x_for_nonnegative (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by
  sorry

end sqrt_square_eq_x_for_nonnegative_l1390_139073


namespace inequality_condition_l1390_139060

def f (x : ℝ) := x^2 + 3*x + 2

theorem inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 2| < b → |f x + 4| < a) ↔ b ≤ a/7 := by sorry

end inequality_condition_l1390_139060


namespace cube_face_sum_l1390_139014

/-- Represents the numbers on the faces of a cube -/
def CubeFaces := Fin 6 → ℕ

/-- The sum of a pair of opposite faces -/
def OppositeSum (faces : CubeFaces) (pair : Fin 3) : ℕ :=
  faces (2 * pair) + faces (2 * pair + 1)

theorem cube_face_sum (faces : CubeFaces) :
  (∃ (pair : Fin 3), OppositeSum faces pair = 11) →
  (∀ (pair : Fin 3), OppositeSum faces pair ≠ 9) ∧
  (∀ (i : Fin 6), faces i ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) ∧
  (∀ (i j : Fin 6), i ≠ j → faces i ≠ faces j) :=
by sorry

end cube_face_sum_l1390_139014


namespace f_at_2_l1390_139048

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem f_at_2 : f 2 = 243 := by
  sorry

end f_at_2_l1390_139048


namespace sequence_2014_term_l1390_139061

/-- A positive sequence satisfying the given recurrence relation -/
def PositiveSequence (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, n * a (n + 1) = (n + 1) * a n ∧ 0 < a n

/-- The 2014th term of the sequence is equal to 2014 -/
theorem sequence_2014_term (a : ℕ+ → ℝ) (h : PositiveSequence a) : a 2014 = 2014 := by
  sorry

end sequence_2014_term_l1390_139061


namespace work_completion_time_l1390_139072

theorem work_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / a = 1 / 6) → (1 / a + 1 / b = 1 / 4) → b = 12 := by sorry

end work_completion_time_l1390_139072


namespace not_all_points_satisfy_equation_tan_eq_one_not_same_as_pi_over_four_rho_3_same_as_neg_3_l1390_139035

-- Define a polar coordinate system
structure PolarCoordinate where
  r : ℝ
  θ : ℝ

-- Define a curve in polar coordinates
def PolarCurve := PolarCoordinate → Prop

-- Statement 1
theorem not_all_points_satisfy_equation (C : PolarCurve) :
  ¬ ∀ (P : PolarCoordinate), C P → (∀ (eq : PolarCoordinate → Prop), (∀ Q, C Q → eq Q) → eq P) :=
sorry

-- Statement 2
theorem tan_eq_one_not_same_as_pi_over_four :
  ∃ (P : PolarCoordinate), (Real.tan P.θ = 1) ≠ (P.θ = π / 4) :=
sorry

-- Statement 3
theorem rho_3_same_as_neg_3 :
  ∀ (P : PolarCoordinate), P.r = 3 ↔ P.r = -3 :=
sorry

end not_all_points_satisfy_equation_tan_eq_one_not_same_as_pi_over_four_rho_3_same_as_neg_3_l1390_139035


namespace gcd_repeated_digit_numbers_l1390_139050

def repeated_digit_number (n : ℕ) : ℕ := n * 1001001001

theorem gcd_repeated_digit_numbers :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → d ∣ repeated_digit_number n) ∧
  (∀ (m : ℕ), (∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → m ∣ repeated_digit_number n) → m ∣ d) :=
by sorry

end gcd_repeated_digit_numbers_l1390_139050


namespace exact_two_out_of_three_germinate_l1390_139059

/-- The probability of a single seed germinating -/
def p : ℚ := 4/5

/-- The total number of seeds -/
def n : ℕ := 3

/-- The number of seeds we want to germinate -/
def k : ℕ := 2

/-- The probability of exactly k out of n seeds germinating -/
def prob_k_out_of_n (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exact_two_out_of_three_germinate :
  prob_k_out_of_n p n k = 48/125 := by sorry

end exact_two_out_of_three_germinate_l1390_139059


namespace min_queries_theorem_l1390_139036

/-- Represents a card with either +1 or -1 written on it -/
inductive Card : Type
| plus_one : Card
| minus_one : Card

/-- Represents a deck of cards -/
def Deck := List Card

/-- Represents a query function that returns the product of three cards -/
def Query := Card → Card → Card → Int

/-- The minimum number of queries needed to determine the product of all cards in a deck -/
def min_queries (n : Nat) (circular : Bool) : Nat :=
  match n with
  | 30 => 10
  | 31 => 11
  | 32 => 12
  | 50 => if circular then 50 else 17  -- 17 is a placeholder for the non-circular case
  | _ => 0  -- placeholder for other cases

/-- Theorem stating the minimum number of queries needed for specific deck sizes -/
theorem min_queries_theorem (d : Deck) (q : Query) :
  (d.length = 30 → min_queries 30 false = 10) ∧
  (d.length = 31 → min_queries 31 false = 11) ∧
  (d.length = 32 → min_queries 32 false = 12) ∧
  (d.length = 50 → min_queries 50 true = 50) :=
sorry

end min_queries_theorem_l1390_139036


namespace sets_intersection_and_union_l1390_139085

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem sets_intersection_and_union :
  ∃ x : ℝ, (B x ∩ A x = {9}) ∧ 
           (x = -3) ∧ 
           (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end sets_intersection_and_union_l1390_139085


namespace base5_242_equals_base10_72_l1390_139044

-- Define a function to convert a base 5 number to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

-- Theorem stating that 242 in base 5 is equal to 72 in base 10
theorem base5_242_equals_base10_72 :
  base5ToBase10 [2, 4, 2] = 72 := by
  sorry

end base5_242_equals_base10_72_l1390_139044


namespace polynomial_divisibility_l1390_139006

theorem polynomial_divisibility (C D : ℝ) :
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^104 + C*x + D = 0) →
  C + D = 2 := by
sorry

end polynomial_divisibility_l1390_139006


namespace yoongi_has_smaller_number_l1390_139087

theorem yoongi_has_smaller_number : 
  let jungkook_number := 6 + 3
  let yoongi_number := 4
  yoongi_number < jungkook_number := by
  sorry

end yoongi_has_smaller_number_l1390_139087


namespace smaller_number_problem_l1390_139095

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 40) : 
  min x y = -5 := by
sorry

end smaller_number_problem_l1390_139095


namespace matrix_property_l1390_139026

theorem matrix_property (a b c d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.transpose = A⁻¹) → (Matrix.det A = 1) → (a^2 + b^2 + c^2 + d^2 = 2) := by
  sorry

end matrix_property_l1390_139026


namespace circle_equation_l1390_139025

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency to coordinate axes
def tangent_to_axes (c : Circle) : Prop :=
  c.radius = |c.center.1| ∧ c.radius = |c.center.2|

-- Define the center being on the line
def center_on_line (c : Circle) : Prop :=
  line_equation c.center.1 c.center.2

-- Define the standard equation of a circle
def standard_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation :
  ∀ c : Circle,
  tangent_to_axes c →
  center_on_line c →
  (∃ x y : ℝ, standard_equation c x y) →
  (∀ x y : ℝ, standard_equation c x y ↔ 
    ((x + 2)^2 + (y - 2)^2 = 4 ∨ (x + 6)^2 + (y + 6)^2 = 36)) :=
sorry

end circle_equation_l1390_139025


namespace ab_plus_cd_equals_zero_l1390_139083

theorem ab_plus_cd_equals_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*d - b*c = -1) : 
  a*b + c*d = 0 := by
sorry

end ab_plus_cd_equals_zero_l1390_139083


namespace license_plate_count_l1390_139052

/-- The number of possible digits (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A to Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 6

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_count + 1

/-- Calculates the total number of possible distinct license plates -/
def total_license_plates : ℕ :=
  block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count :
  total_license_plates = 122504000 :=
sorry

end license_plate_count_l1390_139052


namespace online_price_theorem_l1390_139028

/-- The price that the buyer observes online for a product sold by a distributor through an online store -/
theorem online_price_theorem (cost : ℝ) (commission_rate : ℝ) (profit_rate : ℝ) 
  (h_cost : cost = 19)
  (h_commission : commission_rate = 0.2)
  (h_profit : profit_rate = 0.2) :
  let distributor_price := cost * (1 + profit_rate)
  let online_price := distributor_price / (1 - commission_rate)
  online_price = 28.5 := by
sorry

end online_price_theorem_l1390_139028


namespace number_solution_l1390_139080

theorem number_solution : ∃ (x : ℝ), 50 + (x * 12) / (180 / 3) = 51 ∧ x = 5 := by
  sorry

end number_solution_l1390_139080


namespace polynomial_multiplication_l1390_139037

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 20*x^2 + 400) * (x^2 - 20) = x^6 - 8000 := by
  sorry

end polynomial_multiplication_l1390_139037


namespace g_behavior_l1390_139012

/-- The quadratic function g(x) = x^2 - 2x - 8 -/
def g (x : ℝ) : ℝ := x^2 - 2*x - 8

/-- The graph of g(x) goes up to the right and up to the left -/
theorem g_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) :=
sorry

end g_behavior_l1390_139012


namespace sum_of_roots_l1390_139038

theorem sum_of_roots (p : ℝ) : 
  let q : ℝ := p^2 - 1
  let f : ℝ → ℝ := λ x ↦ x^2 - p*x + q
  ∃ r s : ℝ, f r = 0 ∧ f s = 0 ∧ r + s = p :=
by sorry

end sum_of_roots_l1390_139038


namespace sqrt_expression_eq_zero_quadratic_equation_solutions_l1390_139054

-- Problem 1
theorem sqrt_expression_eq_zero (a : ℝ) (h : a > 0) :
  Real.sqrt (8 * a^3) - 4 * a^2 * Real.sqrt (1 / (8 * a)) - 2 * a * Real.sqrt (a / 2) = 0 := by
  sorry

-- Problem 2
theorem quadratic_equation_solutions :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = -1 ∨ x = 2 := by
  sorry

end sqrt_expression_eq_zero_quadratic_equation_solutions_l1390_139054


namespace base_eight_1423_equals_787_l1390_139021

/-- Converts a base-8 digit to its base-10 equivalent -/
def baseEightDigitToBaseTen (d : Nat) : Nat :=
  if d < 8 then d else 0

/-- Converts a four-digit base-8 number to base-10 -/
def baseEightToBaseTen (a b c d : Nat) : Nat :=
  (baseEightDigitToBaseTen a) * 512 + 
  (baseEightDigitToBaseTen b) * 64 + 
  (baseEightDigitToBaseTen c) * 8 + 
  (baseEightDigitToBaseTen d)

theorem base_eight_1423_equals_787 : 
  baseEightToBaseTen 1 4 2 3 = 787 := by
  sorry

end base_eight_1423_equals_787_l1390_139021


namespace two_digit_numbers_problem_l1390_139074

theorem two_digit_numbers_problem (A B : ℕ) : 
  A ≥ 10 ∧ A ≤ 99 ∧ B ≥ 10 ∧ B ≤ 99 →
  (100 * A + B) / B = 121 →
  (100 * B + A) / A = 84 ∧ (100 * B + A) % A = 14 →
  A = 42 ∧ B = 35 := by
sorry

end two_digit_numbers_problem_l1390_139074
