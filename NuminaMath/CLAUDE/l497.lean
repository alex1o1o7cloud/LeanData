import Mathlib

namespace flagpole_height_l497_49707

theorem flagpole_height (wire_ground_distance : Real) (person_distance : Real) (person_height : Real) :
  wire_ground_distance = 5 →
  person_distance = 3 →
  person_height = 1.8 →
  ∃ (flagpole_height : Real),
    flagpole_height = 4.5 ∧
    (flagpole_height / wire_ground_distance = person_height / (wire_ground_distance - person_distance)) :=
by sorry

end flagpole_height_l497_49707


namespace greg_is_sixteen_l497_49763

-- Define the ages of the siblings
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem to prove Greg's age
theorem greg_is_sixteen : greg_age = 16 := by
  sorry


end greg_is_sixteen_l497_49763


namespace bugs_meeting_point_l497_49701

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  FD : ℝ

/-- Two bugs crawling on the triangle's perimeter -/
structure Bugs where
  speed1 : ℝ
  speed2 : ℝ
  direction : Bool -- True if same direction, False if opposite

/-- Point G where the bugs meet -/
def meetingPoint (t : Triangle) (b : Bugs) : ℝ := sorry

/-- Theorem stating that EG = 2 under given conditions -/
theorem bugs_meeting_point (t : Triangle) (b : Bugs) : 
  t.DE = 8 ∧ t.EF = 10 ∧ t.FD = 12 ∧ 
  b.speed1 = 1 ∧ b.speed2 = 2 ∧ b.direction = false → 
  meetingPoint t b = 2 := by sorry

end bugs_meeting_point_l497_49701


namespace root_shift_cubic_l497_49740

/-- Given a cubic polynomial with roots p, q, and r, 
    find the monic polynomial with roots p + 3, q + 3, and r + 3 -/
theorem root_shift_cubic (p q r : ℂ) : 
  (p^3 - 4*p^2 + 9*p - 7 = 0) ∧ 
  (q^3 - 4*q^2 + 9*q - 7 = 0) ∧ 
  (r^3 - 4*r^2 + 9*r - 7 = 0) → 
  ∃ (a b c : ℂ), 
    (∀ x : ℂ, x^3 - 13*x^2 + 60*x - 90 = (x - (p + 3)) * (x - (q + 3)) * (x - (r + 3))) :=
by sorry

end root_shift_cubic_l497_49740


namespace hyperbola_points_l497_49728

def hyperbola (x y : ℝ) : Prop := y = -4 / x

theorem hyperbola_points :
  hyperbola (-2) 2 ∧
  ¬ hyperbola 1 4 ∧
  ¬ hyperbola (-1) (-4) ∧
  ¬ hyperbola (-2) (-2) :=
by sorry

end hyperbola_points_l497_49728


namespace no_discriminant_for_quartic_l497_49774

theorem no_discriminant_for_quartic (P : ℝ → ℝ → ℝ → ℝ → ℝ) :
  ∃ (a b c d : ℝ),
    (∃ (r₁ r₂ r₃ r₄ : ℝ), ∀ (x : ℝ),
      x^4 + a*x^3 + b*x^2 + c*x + d = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) ∧
      P a b c d < 0) ∨
    ((¬ ∃ (r₁ r₂ r₃ r₄ : ℝ), ∀ (x : ℝ),
      x^4 + a*x^3 + b*x^2 + c*x + d = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄)) ∧
      P a b c d ≥ 0) :=
by sorry

end no_discriminant_for_quartic_l497_49774


namespace work_completion_men_count_l497_49751

/-- Given that 42 men can complete a piece of work in 18 days, and another group can complete
    the same work in 28 days, prove that the second group consists of 27 men. -/
theorem work_completion_men_count :
  ∀ (work : ℝ) (men_group2 : ℕ),
    work = 42 * 18 →
    work = men_group2 * 28 →
    men_group2 = 27 := by
  sorry

end work_completion_men_count_l497_49751


namespace price_increase_for_constant_revenue_l497_49703

/-- Proves that a 25% price increase is necessary to maintain constant revenue when demand decreases by 20% --/
theorem price_increase_for_constant_revenue 
  (original_price original_demand : ℝ) 
  (new_demand : ℝ) 
  (h_demand_decrease : new_demand = 0.8 * original_demand) 
  (h_revenue_constant : original_price * original_demand = (original_price * (1 + 0.25)) * new_demand) :
  (original_price * (1 + 0.25) - original_price) / original_price = 0.25 :=
by sorry

end price_increase_for_constant_revenue_l497_49703


namespace bus_capacity_proof_l497_49760

theorem bus_capacity_proof (C : ℚ) 
  (h1 : (3 / 4) * C + (4 / 5) * C = 310) : C = 200 := by
  sorry

end bus_capacity_proof_l497_49760


namespace absolute_value_equation_solution_l497_49778

theorem absolute_value_equation_solution :
  ∃ x : ℚ, |6 * x - 8| = 0 ∧ x = 4/3 := by
  sorry

end absolute_value_equation_solution_l497_49778


namespace quadratic_roots_sum_product_l497_49792

theorem quadratic_roots_sum_product (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 201 →
  p*q + r*s = -28743/12 := by
sorry

end quadratic_roots_sum_product_l497_49792


namespace tournament_probability_l497_49789

/-- The probability of two specific participants playing each other in a tournament --/
theorem tournament_probability (n : ℕ) (h : n = 26) :
  let total_matches := n - 1
  let total_pairs := n * (n - 1) / 2
  (total_matches : ℚ) / total_pairs = 1 / 13 :=
by sorry

end tournament_probability_l497_49789


namespace roots_polynomial_sum_l497_49724

theorem roots_polynomial_sum (α β : ℝ) : 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  3*α^3 + 7*β^4 = 448 := by
sorry

end roots_polynomial_sum_l497_49724


namespace coefficient_x_squared_expansion_l497_49723

/-- The coefficient of x^2 in the expansion of (3x^2 + 5x + 2)(4x^2 + 2x + 1) -/
def coefficient_x_squared : ℤ := 21

/-- The first polynomial in the product -/
def p (x : ℚ) : ℚ := 3 * x^2 + 5 * x + 2

/-- The second polynomial in the product -/
def q (x : ℚ) : ℚ := 4 * x^2 + 2 * x + 1

/-- The theorem stating that the coefficient of x^2 in the expansion of (3x^2 + 5x + 2)(4x^2 + 2x + 1) is 21 -/
theorem coefficient_x_squared_expansion :
  ∃ (a b c d e : ℚ), (p * q) = (λ x => a * x^4 + b * x^3 + coefficient_x_squared * x^2 + d * x + e) :=
sorry

end coefficient_x_squared_expansion_l497_49723


namespace jack_apple_distribution_l497_49780

theorem jack_apple_distribution (total_apples : ℕ) (given_to_father : ℕ) (num_friends : ℕ) :
  total_apples = 55 →
  given_to_father = 10 →
  num_friends = 4 →
  (total_apples - given_to_father) % (num_friends + 1) = 0 →
  (total_apples - given_to_father) / (num_friends + 1) = 9 :=
by
  sorry

end jack_apple_distribution_l497_49780


namespace tea_mixture_price_l497_49777

/-- Given three types of tea mixed in a specific ratio, calculate the price of the mixture per kg -/
theorem tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℕ) : 
  price1 = 126 →
  price2 = 135 →
  price3 = 175.5 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3 : ℚ) = 153 := by
sorry

end tea_mixture_price_l497_49777


namespace quadratic_real_roots_condition_l497_49729

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 1 = 0) ↔ (k ≤ 4 ∧ k ≠ 0) := by
  sorry

end quadratic_real_roots_condition_l497_49729


namespace cyclic_fraction_theorem_l497_49765

theorem cyclic_fraction_theorem (x y z k : ℝ) :
  (x / (y + z) = k ∧ y / (z + x) = k ∧ z / (x + y) = k) →
  (k = 1/2 ∨ k = -1) :=
by sorry

end cyclic_fraction_theorem_l497_49765


namespace units_digit_of_seven_to_six_to_five_l497_49739

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 1 := by
  sorry

end units_digit_of_seven_to_six_to_five_l497_49739


namespace remainder_problem_l497_49700

theorem remainder_problem (G : ℕ) (a b : ℕ) (h1 : G = 127) (h2 : a = 1661) (h3 : b = 2045) 
  (h4 : b % G = 13) (h5 : ∀ d : ℕ, d > G → (a % d ≠ 0 ∨ b % d ≠ 0)) :
  a % G = 10 := by
sorry

end remainder_problem_l497_49700


namespace lcm_of_210_and_913_l497_49722

theorem lcm_of_210_and_913 :
  let a : ℕ := 210
  let b : ℕ := 913
  let hcf : ℕ := 83
  Nat.lcm a b = 2310 :=
by
  sorry

end lcm_of_210_and_913_l497_49722


namespace cubic_function_c_value_l497_49736

/-- A function f: ℝ → ℝ has exactly two roots if there exist exactly two distinct real numbers x₁ and x₂ such that f(x₁) = f(x₂) = 0 -/
def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂

/-- The main theorem stating that if y = x³ - 3x + c has exactly two roots, then c = -2 or c = 2 -/
theorem cubic_function_c_value (c : ℝ) :
  has_exactly_two_roots (λ x : ℝ => x^3 - 3*x + c) → c = -2 ∨ c = 2 :=
by sorry

end cubic_function_c_value_l497_49736


namespace meaningful_exponent_range_l497_49738

theorem meaningful_exponent_range (x : ℝ) : 
  (∃ y : ℝ, (2*x - 3)^0 = y) ↔ x ≠ 3/2 := by sorry

end meaningful_exponent_range_l497_49738


namespace haley_origami_papers_l497_49786

/-- The number of origami papers Haley has to give away -/
def total_papers : ℕ := 48

/-- The number of Haley's cousins -/
def num_cousins : ℕ := 6

/-- The number of papers each cousin would receive if Haley distributes all her papers equally -/
def papers_per_cousin : ℕ := 8

/-- Theorem stating that the total number of origami papers Haley has to give away is 48 -/
theorem haley_origami_papers :
  total_papers = num_cousins * papers_per_cousin :=
by sorry

end haley_origami_papers_l497_49786


namespace additional_money_needed_l497_49768

/-- Given a football team, budget, and cost per football, calculate the additional money needed --/
theorem additional_money_needed 
  (num_players : ℕ) 
  (budget : ℕ) 
  (cost_per_football : ℕ) 
  (h1 : num_players = 22)
  (h2 : budget = 1500)
  (h3 : cost_per_football = 69) : 
  (num_players * cost_per_football - budget : ℤ) = 18 := by
  sorry

end additional_money_needed_l497_49768


namespace product_of_decimals_product_of_fractions_l497_49752

/-- Proves that (-0.4) * (-0.8) * (-1.25) * 2.5 = -1 -/
theorem product_of_decimals : (-0.4) * (-0.8) * (-1.25) * 2.5 = -1 := by
  sorry

/-- Proves that (-5/8) * (3/14) * (-16/5) * (-7/6) = -1/2 -/
theorem product_of_fractions : (-5/8 : ℚ) * (3/14 : ℚ) * (-16/5 : ℚ) * (-7/6 : ℚ) = -1/2 := by
  sorry

end product_of_decimals_product_of_fractions_l497_49752


namespace unit_conversions_l497_49758

-- Define the conversion rates
def kg_per_ton : ℝ := 1000
def sq_dm_per_sq_m : ℝ := 100

-- Define the theorem
theorem unit_conversions :
  (8 : ℝ) + 800 / kg_per_ton = 8.8 ∧
  6.32 * sq_dm_per_sq_m = 632 :=
by sorry

end unit_conversions_l497_49758


namespace min_value_of_f_l497_49731

/-- The function f(x) = x^2 + 8x + 12 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 12

/-- The minimum value of f(x) is -4 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -4 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end min_value_of_f_l497_49731


namespace algebraic_expression_equality_l497_49787

theorem algebraic_expression_equality (a b : ℝ) (h : a^2 + 2*b^2 - 1 = 0) :
  (a - b)^2 + b*(2*a + b) = 1 := by
  sorry

end algebraic_expression_equality_l497_49787


namespace food_consumption_reduction_l497_49715

/-- Proves that given a 15% decrease in students and 20% increase in food price,
    the consumption reduction factor to maintain the same total cost is approximately 0.98039 -/
theorem food_consumption_reduction (N : ℝ) (P : ℝ) (h1 : N > 0) (h2 : P > 0) :
  let new_students := 0.85 * N
  let new_price := 1.2 * P
  let consumption_factor := (N * P) / (new_students * new_price)
  ∃ ε > 0, abs (consumption_factor - 0.98039) < ε :=
by sorry

end food_consumption_reduction_l497_49715


namespace largest_expression_l497_49767

theorem largest_expression : 
  let a := 3 + 1 + 2 + 9
  let b := 3 * 1 + 2 + 9
  let c := 3 + 1 * 2 + 9
  let d := 3 + 1 + 2 * 9
  let e := 3 * 1 * 2 * 9
  (e > a) ∧ (e > b) ∧ (e > c) ∧ (e > d) := by
  sorry

end largest_expression_l497_49767


namespace carrie_cake_days_l497_49705

/-- Proves that Carrie worked 4 days on the cake given the specified conditions. -/
theorem carrie_cake_days : 
  ∀ (hours_per_day : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) (profit : ℕ),
    hours_per_day = 2 →
    hourly_rate = 22 →
    supply_cost = 54 →
    profit = 122 →
    ∃ (days : ℕ), 
      days = 4 ∧ 
      profit = hours_per_day * hourly_rate * days - supply_cost :=
by
  sorry


end carrie_cake_days_l497_49705


namespace units_digit_17_pow_2024_l497_49725

theorem units_digit_17_pow_2024 : (17^2024) % 10 = 1 := by
  sorry

end units_digit_17_pow_2024_l497_49725


namespace f_minus_two_equals_minus_twelve_l497_49727

def symmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

theorem f_minus_two_equals_minus_twelve
  (f : ℝ → ℝ)
  (h_symmetric : symmetricAbout f 1)
  (h_def : ∀ x : ℝ, x ≥ 1 → f x = x * (1 - x)) :
  f (-2) = -12 := by
  sorry

end f_minus_two_equals_minus_twelve_l497_49727


namespace simplify_expression_l497_49718

theorem simplify_expression (x : ℝ) : (2*x)^5 + (3*x)*(x^4) + 2*x^3 = 35*x^5 + 2*x^3 := by
  sorry

end simplify_expression_l497_49718


namespace repeating_decimal_sum_diff_l497_49745

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- The sum of two repeating decimals 0.̅6 and 0.̅2 minus 0.̅4 equals 4/9 -/
theorem repeating_decimal_sum_diff :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_diff_l497_49745


namespace tenth_term_of_sequence_l497_49734

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence : 
  let a₁ := (1 : ℚ) / 2
  let a₂ := (3 : ℚ) / 4
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 10 = (11 : ℚ) / 4 := by
sorry

end tenth_term_of_sequence_l497_49734


namespace class_size_l497_49797

/-- Given a class with a hair color ratio of 3:6:7 (red:blonde:black) and 9 red-haired kids,
    the total number of kids in the class is 48. -/
theorem class_size (red blonde black : ℕ) (total : ℕ) : 
  red = 3 → blonde = 6 → black = 7 → -- ratio condition
  red + blonde + black = total → -- total parts in ratio
  9 * total = 48 * red → -- condition for 9 red-haired kids
  total = 48 := by sorry

end class_size_l497_49797


namespace unique_solution_condition_l497_49716

/-- The quadratic function g(x) = x^2 + 2bx + 2b -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + 2*b

/-- The theorem stating the condition for exactly one solution -/
theorem unique_solution_condition (b : ℝ) :
  (∃! x : ℝ, |g b x| ≤ 3) ↔ (b = 3 ∨ b = -1) :=
sorry

end unique_solution_condition_l497_49716


namespace solution_set_f_greater_than_2_range_of_t_l497_49746

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -5 ∨ 1 < x} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} := by sorry

end solution_set_f_greater_than_2_range_of_t_l497_49746


namespace count_D_eq_2_is_30_l497_49750

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 127 for which D(n) = 2 -/
def count_D_eq_2 : ℕ := sorry

theorem count_D_eq_2_is_30 : count_D_eq_2 = 30 := by sorry

end count_D_eq_2_is_30_l497_49750


namespace line_perp_plane_iff_perp_all_lines_l497_49795

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Predicate for a line being perpendicular to a plane -/
def perpendicular_to_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a line being perpendicular to another line -/
def perpendicular_to_line (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for a line being inside a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Theorem stating the equivalence of a line being perpendicular to a plane
    and being perpendicular to all lines in that plane -/
theorem line_perp_plane_iff_perp_all_lines (l : Line3D) (α : Plane3D) :
  perpendicular_to_plane l α ↔ ∀ m : Line3D, line_in_plane m α → perpendicular_to_line l m :=
sorry

end line_perp_plane_iff_perp_all_lines_l497_49795


namespace tangent_line_to_logarithmic_curve_l497_49733

theorem tangent_line_to_logarithmic_curve (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ k * x - 3 = 2 * Real.log x ∧
    (∀ y : ℝ, y > 0 → k * y - 3 ≥ 2 * Real.log y)) →
  k = 2 * Real.sqrt (Real.exp 1) :=
by sorry

end tangent_line_to_logarithmic_curve_l497_49733


namespace visitor_difference_l497_49713

def visitors_current_day : ℕ := 317
def visitors_previous_day : ℕ := 295

theorem visitor_difference : visitors_current_day - visitors_previous_day = 22 := by
  sorry

end visitor_difference_l497_49713


namespace coupe_price_proof_l497_49782

/-- The amount for which Melissa sold the coupe -/
def coupe_price : ℝ := 30000

/-- The amount for which Melissa sold the SUV -/
def suv_price : ℝ := 2 * coupe_price

/-- The commission rate -/
def commission_rate : ℝ := 0.02

/-- The total commission from both sales -/
def total_commission : ℝ := 1800

theorem coupe_price_proof :
  commission_rate * (coupe_price + suv_price) = total_commission :=
sorry

end coupe_price_proof_l497_49782


namespace lateral_surface_area_of_parallelepiped_l497_49788

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  diagonal : ℝ
  angle_with_base : ℝ
  base_area : ℝ

-- Define the theorem
theorem lateral_surface_area_of_parallelepiped (p : RectangularParallelepiped) 
  (h1 : p.diagonal = 10)
  (h2 : p.angle_with_base = Real.pi / 3)  -- 60 degrees in radians
  (h3 : p.base_area = 12) :
  ∃ (lateral_area : ℝ), lateral_area = 70 * Real.sqrt 3 := by
  sorry

end lateral_surface_area_of_parallelepiped_l497_49788


namespace prob_random_twin_prob_twins_in_three_expected_twin_pairs_l497_49799

/-- Represents the probability model for twins in Schwambrania -/
structure TwinProbability where
  /-- The probability of twins being born -/
  p : ℝ
  /-- Assumption that p is between 0 and 1 -/
  h_p_bounds : 0 ≤ p ∧ p ≤ 1
  /-- Assumption that triplets do not exist -/
  h_no_triplets : True

/-- Theorem for the probability of a random person being a twin -/
theorem prob_random_twin (model : TwinProbability) :
  (2 * model.p) / (model.p + 1) = Real.exp (Real.log (2 * model.p) - Real.log (model.p + 1)) :=
sorry

/-- Theorem for the probability of having at least one pair of twins in a family with three children -/
theorem prob_twins_in_three (model : TwinProbability) :
  (2 * model.p) / (2 * model.p + (1 - model.p)^2) =
  Real.exp (Real.log (2 * model.p) - Real.log (2 * model.p + (1 - model.p)^2)) :=
sorry

/-- Theorem for the expected number of twin pairs among N first-graders -/
theorem expected_twin_pairs (model : TwinProbability) (N : ℕ) :
  (N : ℝ) * model.p / (model.p + 1) =
  Real.exp (Real.log N + Real.log model.p - Real.log (model.p + 1)) :=
sorry

end prob_random_twin_prob_twins_in_three_expected_twin_pairs_l497_49799


namespace complementary_angles_ratio_l497_49709

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The angles are in a ratio of 4:1
  b = 18 :=     -- The smaller angle is 18°
by sorry

end complementary_angles_ratio_l497_49709


namespace imaginary_unit_sum_l497_49794

theorem imaginary_unit_sum (i : ℂ) (h : i^2 = -1) : 
  1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i := by
  sorry

end imaginary_unit_sum_l497_49794


namespace no_real_solutions_l497_49784

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (x - 3*x + 8)^2 + 4 = -2 * |x| := by
  sorry

end no_real_solutions_l497_49784


namespace red_pens_per_student_red_pens_calculation_l497_49775

theorem red_pens_per_student (students : ℕ) (black_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) (pens_taken_second_month : ℕ) 
  (remaining_pens_per_student : ℕ) : ℕ :=
  let total_black_pens := students * black_pens_per_student
  let total_pens_taken := pens_taken_first_month + pens_taken_second_month
  let total_remaining_pens := students * remaining_pens_per_student
  let initial_total_pens := total_pens_taken + total_remaining_pens
  let total_red_pens := initial_total_pens - total_black_pens
  total_red_pens / students

theorem red_pens_calculation :
  red_pens_per_student 3 43 37 41 79 = 62 := by
  sorry

end red_pens_per_student_red_pens_calculation_l497_49775


namespace total_score_is_40_l497_49773

def game1_score : ℕ := 10
def game2_score : ℕ := 14
def game3_score : ℕ := 6

def first_three_games_total : ℕ := game1_score + game2_score + game3_score
def first_three_games_average : ℕ := first_three_games_total / 3
def game4_score : ℕ := first_three_games_average

def total_score : ℕ := first_three_games_total + game4_score

theorem total_score_is_40 : total_score = 40 := by
  sorry

end total_score_is_40_l497_49773


namespace inequality_proof_l497_49756

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by sorry

end inequality_proof_l497_49756


namespace space_diagonals_of_Q_l497_49710

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  hexagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := (Q.vertices.choose 2)
  let non_edge_segments := total_line_segments - Q.edges
  let face_diagonals := Q.hexagonal_faces * 9
  non_edge_segments - face_diagonals

/-- Theorem stating the number of space diagonals in the specific polyhedron Q -/
theorem space_diagonals_of_Q : 
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 32,
    hexagonal_faces := 12
  }
  space_diagonals Q = 255 := by sorry

end space_diagonals_of_Q_l497_49710


namespace log_sum_equals_two_l497_49753

-- Define the logarithm functions
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end log_sum_equals_two_l497_49753


namespace line_hyperbola_intersection_l497_49755

/-- The number of intersection points between a line and a hyperbola -/
theorem line_hyperbola_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃! p : ℝ × ℝ, 
    (p.2 = (b / a) * p.1 + 3) ∧ 
    ((p.1^2 / a^2) - (p.2^2 / b^2) = 1) :=
sorry

end line_hyperbola_intersection_l497_49755


namespace quadratic_inequality_l497_49719

/-- Given a quadratic function f(x) = ax^2 + (1-a)x + a - 2,
    if f(x) ≥ -2 for all real x, then a ≥ 1/3 -/
theorem quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (1 - a) * x + a - 2 ≥ -2) → a ≥ 1/3 := by
  sorry

end quadratic_inequality_l497_49719


namespace min_sum_squares_l497_49706

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), (∀ x' y' z' : ℝ, x' + 2*y' + z' = 1 → x'^2 + y'^2 + z'^2 ≥ m) ∧
             (∃ x₀ y₀ z₀ : ℝ, x₀ + 2*y₀ + z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = m) ∧
             m = 1/6 :=
by sorry

end min_sum_squares_l497_49706


namespace remaining_space_for_regular_toenails_l497_49717

/-- Represents the capacity of the jar in terms of regular toenails -/
def jarCapacity : ℕ := 100

/-- Represents the space occupied by a big toenail in terms of regular toenails -/
def bigToenailSpace : ℕ := 2

/-- Represents the number of big toenails already in the jar -/
def bigToenailsInJar : ℕ := 20

/-- Represents the number of regular toenails already in the jar -/
def regularToenailsInJar : ℕ := 40

/-- Theorem stating that the remaining space in the jar can fit exactly 20 regular toenails -/
theorem remaining_space_for_regular_toenails : 
  jarCapacity - (bigToenailsInJar * bigToenailSpace + regularToenailsInJar) = 20 := by
  sorry

end remaining_space_for_regular_toenails_l497_49717


namespace exponent_equivalence_l497_49742

theorem exponent_equivalence (y : ℕ) (some_exponent : ℕ) 
  (h1 : 9^y = 3^some_exponent) (h2 : y = 8) : some_exponent = 16 := by
  sorry

end exponent_equivalence_l497_49742


namespace f_bounds_l497_49748

/-- The maximum number of elements from Example 1 -/
def f (n : ℕ) : ℕ := sorry

/-- Proof that f(n) satisfies the given inequality -/
theorem f_bounds (n : ℕ) (hn : n > 0) : 
  (1 / 6 : ℚ) * (n^2 - 4*n : ℚ) ≤ (f n : ℚ) ∧ (f n : ℚ) ≤ (1 / 6 : ℚ) * (n^2 - n : ℚ) := by
  sorry

end f_bounds_l497_49748


namespace four_digit_number_theorem_l497_49785

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def digit_at (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10 ^ place)) % 10

theorem four_digit_number_theorem (n : ℕ) :
  is_valid_number n ∧ 
  (digit_at n 0 + digit_at n 1 - 4 * digit_at n 3 = 1) ∧
  (digit_at n 0 + 10 * digit_at n 1 - 2 * digit_at n 2 = 14) →
  n = 1014 ∨ n = 2218 ∨ n = 1932 := by
  sorry

end four_digit_number_theorem_l497_49785


namespace gumball_problem_l497_49708

theorem gumball_problem :
  ∀ x : ℕ,
  (19 ≤ (17 + 12 + x) / 3 ∧ (17 + 12 + x) / 3 ≤ 25) →
  (∃ max min : ℕ,
    (∀ y : ℕ, (19 ≤ (17 + 12 + y) / 3 ∧ (17 + 12 + y) / 3 ≤ 25) → y ≤ max) ∧
    (∀ y : ℕ, (19 ≤ (17 + 12 + y) / 3 ∧ (17 + 12 + y) / 3 ≤ 25) → min ≤ y) ∧
    max - min = 18) :=
by sorry

end gumball_problem_l497_49708


namespace functional_equation_solution_l497_49712

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem statement -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x := by
sorry


end functional_equation_solution_l497_49712


namespace sin_cos_identity_sin_tan_simplification_l497_49770

-- Question 1
theorem sin_cos_identity :
  Real.sin (34 * π / 180) * Real.sin (26 * π / 180) - 
  Real.sin (56 * π / 180) * Real.cos (26 * π / 180) = -1/2 := by sorry

-- Question 2
theorem sin_tan_simplification :
  Real.sin (50 * π / 180) * (Real.sqrt 3 * Real.tan (10 * π / 180) + 1) = 
  Real.cos (20 * π / 180) / Real.cos (10 * π / 180) := by sorry

end sin_cos_identity_sin_tan_simplification_l497_49770


namespace binary_remainder_is_two_l497_49743

/-- Given a binary number represented as a list of bits (least significant bit first),
    calculate the remainder when divided by 4. -/
def binary_remainder_mod_4 (bits : List Bool) : Nat :=
  match bits with
  | [] => 0
  | [b₀] => if b₀ then 1 else 0
  | b₀ :: b₁ :: _ => (if b₁ then 2 else 0) + (if b₀ then 1 else 0)

/-- The binary representation of 100101110010₂ (least significant bit first) -/
def binary_number : List Bool :=
  [false, true, false, false, true, true, true, false, true, false, false, true]

/-- Theorem stating that the remainder when 100101110010₂ is divided by 4 is 2 -/
theorem binary_remainder_is_two :
  binary_remainder_mod_4 binary_number = 2 := by
  sorry


end binary_remainder_is_two_l497_49743


namespace molecular_weight_CaO_is_56_l497_49783

/-- The molecular weight of CaO in grams per mole -/
def molecular_weight_CaO : ℝ := 56

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles of CaO in grams -/
def given_weight : ℝ := 392

/-- Theorem stating that the molecular weight of CaO is 56 grams/mole -/
theorem molecular_weight_CaO_is_56 :
  molecular_weight_CaO = given_weight / given_moles :=
sorry

end molecular_weight_CaO_is_56_l497_49783


namespace divisor_with_remainder_one_l497_49757

theorem divisor_with_remainder_one (n : ℕ) : 
  ∃ k : ℕ, 2^200 - 3 = k * (2^100 - 2) + 1 := by
  sorry

end divisor_with_remainder_one_l497_49757


namespace revenue_comparison_l497_49759

theorem revenue_comparison (last_year_revenue : ℝ) : 
  let projected_revenue := last_year_revenue * 1.20
  let actual_revenue := last_year_revenue * 0.90
  actual_revenue / projected_revenue = 0.75 := by
sorry

end revenue_comparison_l497_49759


namespace simplify_fraction_l497_49791

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x^2 / y) * (y^2 / (2 * x)) = 3 * x * y / 2 := by
  sorry

end simplify_fraction_l497_49791


namespace smallest_solution_congruence_l497_49714

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 18 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 18 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l497_49714


namespace square_area_is_17_l497_49749

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square defined by four vertices -/
structure Square where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Calculate the area of a square given its four vertices -/
def squareArea (s : Square) : ℝ :=
  squaredDistance s.P s.Q

/-- The specific square from the problem -/
def problemSquare : Square :=
  { P := { x := 1, y := 2 },
    Q := { x := -3, y := 3 },
    R := { x := -2, y := 8 },
    S := { x := 2, y := 7 } }

theorem square_area_is_17 :
  squareArea problemSquare = 17 := by
  sorry

end square_area_is_17_l497_49749


namespace complex_fraction_simplification_l497_49769

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - I) : 2 / z = 1 + I := by
  sorry

end complex_fraction_simplification_l497_49769


namespace sin_translation_l497_49779

open Real

theorem sin_translation (t S : ℝ) (k : ℤ) : 
  (1 = sin (2 * t)) → 
  (S > 0) → 
  (1 = sin (2 * (t + S) - π / 3)) → 
  (t = π / 4 + k * π ∧ S ≥ π / 6) :=
sorry

end sin_translation_l497_49779


namespace simplify_expression_l497_49793

theorem simplify_expression (n : ℕ) : (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 27 := by
  sorry

end simplify_expression_l497_49793


namespace marble_collection_total_l497_49702

theorem marble_collection_total (b : ℝ) : 
  let r := 1.3 * b -- red marbles
  let g := 1.5 * b -- green marbles
  r + b + g = 3.8 * b := by sorry

end marble_collection_total_l497_49702


namespace canned_food_bins_l497_49735

theorem canned_food_bins (soup_bins vegetables_bins pasta_bins : Real) 
  (h1 : soup_bins = 0.12)
  (h2 : vegetables_bins = 0.12)
  (h3 : pasta_bins = 0.5) :
  soup_bins + vegetables_bins + pasta_bins = 0.74 := by
  sorry

end canned_food_bins_l497_49735


namespace composition_equality_l497_49796

-- Define the functions f and g
def f (b : ℝ) (x : ℝ) : ℝ := 5 * x + b
def g (b : ℝ) (x : ℝ) : ℝ := b * x + 4

-- State the theorem
theorem composition_equality (b e : ℝ) : 
  (∀ x, f b (g b x) = 15 * x + e) → e = 23 := by
  sorry

end composition_equality_l497_49796


namespace sum_remainder_thirteen_l497_49766

theorem sum_remainder_thirteen : ∃ k : ℕ, (8930 + 8931 + 8932 + 8933 + 8934) = 13 * k + 5 := by
  sorry

end sum_remainder_thirteen_l497_49766


namespace right_triangle_legs_product_divisible_by_12_l497_49721

theorem right_triangle_legs_product_divisible_by_12 
  (a b c : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  12 ∣ (a * b) := by
  sorry

end right_triangle_legs_product_divisible_by_12_l497_49721


namespace inequality_proof_l497_49776

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a - b)^2 / (2 * (a + b)) ≤ Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ∧
  Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ≤ (a - b)^2 / (a + b) :=
by sorry

end inequality_proof_l497_49776


namespace layla_point_difference_l497_49781

theorem layla_point_difference (total_points layla_points : ℕ) 
  (h1 : total_points = 345) 
  (h2 : layla_points = 180) : 
  layla_points - (total_points - layla_points) = 15 := by
  sorry

end layla_point_difference_l497_49781


namespace b2_properties_b2_b4_equality_a_and_x_relation_l497_49762

theorem b2_properties (B₂ : ℝ) (A : ℝ) (x : ℝ) : 
  B₂ = B₂^2 - 2 →
  (B₂ = -1 ∨ B₂ = 2) ∧
  (B₂ = -1 → (A = 1 ∨ A = -1) ∧ ¬(∃ x, x + 1/x = 1)) ∧
  (B₂ = 2 → (A = 2 ∨ A = -2) ∧ (x = 1 ∨ x = -1)) :=
by sorry

theorem b2_b4_equality (B₂ B₄ : ℝ) :
  B₂ = B₄ → B₂ = B₂^2 - 2 :=
by sorry

theorem a_and_x_relation (A x : ℝ) :
  A = x + 1/x →
  (A = 2 → x = 1) ∧
  (A = -2 → x = -1) :=
by sorry

end b2_properties_b2_b4_equality_a_and_x_relation_l497_49762


namespace zachary_pushups_l497_49744

theorem zachary_pushups (david_pushups : ℕ) (difference : ℕ) (h1 : david_pushups = 62) (h2 : difference = 15) :
  david_pushups - difference = 47 := by
  sorry

end zachary_pushups_l497_49744


namespace unique_solution_l497_49754

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

/-- The main theorem stating that -3/4 is the unique solution -/
theorem unique_solution (a : ℝ) (h : a ≠ 0) :
  f a (1 - a) = f a (1 + a) ↔ a = -3/4 := by
  sorry

end unique_solution_l497_49754


namespace sonya_falls_l497_49704

/-- The number of times each person fell while ice skating --/
structure FallCounts where
  steven : ℕ
  stephanie : ℕ
  sonya : ℕ
  sam : ℕ
  sophie : ℕ

/-- The conditions given in the problem --/
def carnival_conditions (fc : FallCounts) : Prop :=
  fc.steven = 3 ∧
  fc.stephanie = fc.steven + 13 ∧
  fc.sonya = fc.stephanie / 2 - 2 ∧
  fc.sam = 1 ∧
  fc.sophie = fc.sam + 4

/-- Theorem stating that Sonya fell 6 times --/
theorem sonya_falls (fc : FallCounts) (h : carnival_conditions fc) : fc.sonya = 6 := by
  sorry

end sonya_falls_l497_49704


namespace train_crossing_time_l497_49737

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 80

/-- Represents the length of the train in meters -/
def train_length : ℝ := 200

/-- Represents the time it takes for the train to cross the pole in seconds -/
def crossing_time : ℝ := 9

/-- Theorem stating that a train with the given speed and length takes 9 seconds to cross a pole -/
theorem train_crossing_time :
  (train_length / (train_speed * 1000 / 3600)) = crossing_time := by sorry

end train_crossing_time_l497_49737


namespace log2_order_relation_l497_49732

-- Define the logarithm function with base 2
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_order_relation :
  (∀ a b : ℝ, f a > f b → a > b) ∧
  ¬(∀ a b : ℝ, a > b → f a > f b) :=
sorry

end log2_order_relation_l497_49732


namespace point_b_coordinates_l497_49798

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetricPoint (p q : Point3D) : Point3D :=
  ⟨2 * q.x - p.x, 2 * q.y - p.y, 2 * q.z - p.z⟩

def vector (p q : Point3D) : Point3D :=
  ⟨q.x - p.x, q.y - p.y, q.z - p.z⟩

theorem point_b_coordinates
  (A : Point3D)
  (P : Point3D)
  (A_prime : Point3D)
  (B_prime : Point3D)
  (h1 : A = ⟨-1, 3, -3⟩)
  (h2 : P = ⟨1, 2, 3⟩)
  (h3 : A_prime = symmetricPoint A P)
  (h4 : vector A_prime B_prime = ⟨3, 1, 5⟩)
  : ∃ B : Point3D, (B = ⟨-4, 2, -8⟩ ∧ symmetricPoint B P = B_prime) :=
sorry

end point_b_coordinates_l497_49798


namespace response_rate_percentage_l497_49711

theorem response_rate_percentage : 
  ∀ (responses_needed : ℕ) (questionnaires_mailed : ℕ),
  responses_needed = 210 →
  questionnaires_mailed = 350 →
  (responses_needed : ℝ) / (questionnaires_mailed : ℝ) * 100 = 60 := by
sorry

end response_rate_percentage_l497_49711


namespace inscribed_cylinder_radius_l497_49790

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- 
  Theorem: The radius of a cylinder inscribed in a cone
  Given:
  - A right circular cone with diameter 8 and altitude 10
  - A right circular cylinder inscribed in the cone
  - The axes of the cylinder and cone coincide
  - The height of the cylinder is three times its radius
  Prove: The radius of the cylinder is 20/11
-/
theorem inscribed_cylinder_radius (cone : Cone) (cyl : Cylinder) :
  cone.diameter = 8 →
  cone.altitude = 10 →
  cyl.height = 3 * cyl.radius →
  cyl.radius = 20 / 11 := by
  sorry


end inscribed_cylinder_radius_l497_49790


namespace min_value_of_expression_min_value_attained_l497_49772

theorem min_value_of_expression (x : ℝ) :
  (15 - x) * (9 - x) * (15 + x) * (9 + x) ≥ -5184 :=
by sorry

theorem min_value_attained :
  ∃ x : ℝ, (15 - x) * (9 - x) * (15 + x) * (9 + x) = -5184 :=
by sorry

end min_value_of_expression_min_value_attained_l497_49772


namespace cubic_equation_solution_l497_49771

theorem cubic_equation_solution (A : ℕ) (a b s : ℤ) 
  (h_A : A = 1 ∨ A = 2 ∨ A = 3)
  (h_coprime : Int.gcd a b = 1)
  (h_eq : a^2 + A * b^2 = s^3) :
  ∃ u v : ℤ, 
    s = u^2 + A * v^2 ∧
    a = u^3 - 3 * A * u * v^2 ∧
    b = 3 * u^2 * v - A * v^3 := by
  sorry

end cubic_equation_solution_l497_49771


namespace div_exp_eq_pow_specific_calculation_l497_49747

/-- Division exponentiation for rational numbers -/
def div_exp (a : ℚ) (n : ℕ) : ℚ :=
  if n ≤ 1 then a else (1 / a) ^ (n - 2)

/-- Theorem for division exponentiation -/
theorem div_exp_eq_pow (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  div_exp a n = (1 / a) ^ (n - 2) :=
sorry

/-- Theorem for specific calculation -/
theorem specific_calculation :
  2^2 * div_exp (-1/3) 4 / div_exp (-2) 3 - div_exp (-3) 2 = -73 :=
sorry

end div_exp_eq_pow_specific_calculation_l497_49747


namespace geometric_arithmetic_geometric_sequences_l497_49726

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℚ) : Prop :=
  b^2 = a * c

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℚ) : Prop :=
  2 * b = a + c

/-- Represents a triple of rational numbers -/
structure Triple where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a triple satisfies all the conditions -/
def satisfies_conditions (t : Triple) : Prop :=
  is_geometric_progression t.a t.b t.c ∧
  is_arithmetic_progression t.a t.b (t.c - 4) ∧
  is_geometric_progression t.a (t.b - 1) (t.c - 5)

theorem geometric_arithmetic_geometric_sequences :
  ∃ t₁ t₂ : Triple,
    satisfies_conditions t₁ ∧
    satisfies_conditions t₂ ∧
    t₁ = ⟨1/9, 7/9, 49/9⟩ ∧
    t₂ = ⟨1, 3, 9⟩ :=
  sorry

end geometric_arithmetic_geometric_sequences_l497_49726


namespace at_least_one_equation_has_distinct_roots_l497_49761

theorem at_least_one_equation_has_distinct_roots (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0) :=
sorry

end at_least_one_equation_has_distinct_roots_l497_49761


namespace distance_between_externally_tangent_circles_l497_49764

/-- The distance between centers of two externally tangent circles is the sum of their radii -/
theorem distance_between_externally_tangent_circles 
  (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 8) 
  (h₃ : d = r₁ + r₂) : 
  d = 11 := by sorry

end distance_between_externally_tangent_circles_l497_49764


namespace product_of_powers_equals_power_of_sum_l497_49730

theorem product_of_powers_equals_power_of_sum :
  (10 ^ 0.4) * (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.05) * (10 ^ 1.1) * (10 ^ (-0.1)) = 10 ^ 1.85 := by
  sorry

end product_of_powers_equals_power_of_sum_l497_49730


namespace jean_buys_two_cards_per_grandchild_l497_49741

/-- Represents the scenario of Jean's gift-giving to her grandchildren --/
structure GiftGiving where
  num_grandchildren : ℕ
  amount_per_card : ℕ
  total_amount : ℕ

/-- Calculates the number of cards bought for each grandchild --/
def cards_per_grandchild (g : GiftGiving) : ℕ :=
  (g.total_amount / g.amount_per_card) / g.num_grandchildren

/-- Theorem stating that Jean buys 2 cards for each grandchild --/
theorem jean_buys_two_cards_per_grandchild :
  ∀ (g : GiftGiving),
    g.num_grandchildren = 3 →
    g.amount_per_card = 80 →
    g.total_amount = 480 →
    cards_per_grandchild g = 2 := by
  sorry

end jean_buys_two_cards_per_grandchild_l497_49741


namespace sequence_inequality_l497_49720

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ k m : ℕ, |a (k + m) - a k - a m| ≤ 1

theorem sequence_inequality (a : ℕ → ℝ) (h : sequence_condition a) :
  ∀ p q : ℕ, p ≠ 0 → q ≠ 0 → |a p / p - a q / q| < 1 / p + 1 / q :=
sorry

end sequence_inequality_l497_49720
