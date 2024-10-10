import Mathlib

namespace mountain_temperature_l272_27269

theorem mountain_temperature (T : ℝ) 
  (h1 : T * (3/4) = T - 21) : T = 84 := by
  sorry

end mountain_temperature_l272_27269


namespace negation_of_existence_negation_of_exponential_inequality_l272_27260

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃ x > 0, P x) ↔ (∀ x > 0, ¬P x) :=
by sorry

theorem negation_of_exponential_inequality :
  (¬∃ x > 0, 3^x < x^2) ↔ (∀ x > 0, 3^x ≥ x^2) :=
by sorry

end negation_of_existence_negation_of_exponential_inequality_l272_27260


namespace cookies_per_bag_l272_27280

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end cookies_per_bag_l272_27280


namespace quadratic_two_zeros_m_range_l272_27295

theorem quadratic_two_zeros_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) →
  m < -2 ∨ m > 2 :=
by sorry

end quadratic_two_zeros_m_range_l272_27295


namespace sandwich_cost_l272_27223

/-- The cost of tomatoes for N sandwiches, each using T slices at 4 cents per slice --/
def tomatoCost (N T : ℕ) : ℚ := (N * T * 4 : ℕ) / 100

/-- The total cost of ingredients for N sandwiches, each using C slices of cheese and T slices of tomato --/
def totalCost (N C T : ℕ) : ℚ := (N * (3 * C + 4 * T) : ℕ) / 100

theorem sandwich_cost (N C T : ℕ) : 
  N > 1 → C > 0 → T > 0 → totalCost N C T = 305 / 100 → tomatoCost N T = 2 := by
  sorry

end sandwich_cost_l272_27223


namespace min_sum_distances_l272_27262

/-- An ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- The squared distance between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The theorem stating the minimum value of |OP|² + |PF|² -/
theorem min_sum_distances (P : ℝ × ℝ) (h : P ∈ Ellipse) :
  ∃ (m : ℝ), m = 2 ∧ ∀ Q ∈ Ellipse, m ≤ dist_squared O Q + dist_squared Q F :=
sorry

end min_sum_distances_l272_27262


namespace no_prime_with_perfect_square_131_base_l272_27217

theorem no_prime_with_perfect_square_131_base : ¬∃ n : ℕ, 
  (5 ≤ n ∧ n ≤ 15) ∧ 
  Nat.Prime n ∧ 
  ∃ m : ℕ, n^2 + 3*n + 1 = m^2 := by
  sorry

end no_prime_with_perfect_square_131_base_l272_27217


namespace max_circumference_in_standard_parabola_l272_27237

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a parabola in the form x^2 = 4y -/
def standardParabola : Set (ℝ × ℝ) :=
  {p | p.1^2 = 4 * p.2}

/-- Checks if a circle passes through the vertex of the standard parabola -/
def passesVertexStandardParabola (c : Circle) : Prop :=
  c.center.1^2 + c.center.2^2 = c.radius^2

/-- Checks if a circle is entirely inside the standard parabola -/
def insideStandardParabola (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 ≤ c.radius^2 → x^2 ≤ 4 * y

/-- The maximum circumference theorem -/
theorem max_circumference_in_standard_parabola :
  ∃ (c : Circle),
    passesVertexStandardParabola c ∧
    insideStandardParabola c ∧
    (∀ (c' : Circle),
      passesVertexStandardParabola c' ∧
      insideStandardParabola c' →
      2 * π * c'.radius ≤ 2 * π * c.radius) ∧
    2 * π * c.radius = 4 * π :=
sorry

end max_circumference_in_standard_parabola_l272_27237


namespace circle_center_sum_l272_27209

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 2, 
    the sum of the coordinates of its center is 7. -/
theorem circle_center_sum : ∃ (h k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 6*x + 8*y + 2 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 2)) ∧
  h + k = 7 := by
  sorry

end circle_center_sum_l272_27209


namespace quadratic_equation_solution_l272_27234

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 4 := by
sorry

end quadratic_equation_solution_l272_27234


namespace roots_expression_l272_27221

theorem roots_expression (p q : ℝ) (α β γ δ : ℝ) : 
  (α^2 + p*α - 2 = 0) → 
  (β^2 + p*β - 2 = 0) → 
  (γ^2 + q*γ - 3 = 0) → 
  (δ^2 + q*δ - 3 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) - 2*q + 1 := by
  sorry

end roots_expression_l272_27221


namespace coin_division_problem_l272_27271

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 5) → 
  (n % 7 = 2) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 5 ∨ m % 7 ≠ 2)) →
  (n % 9 = 1) := by
sorry

end coin_division_problem_l272_27271


namespace ratio_problem_l272_27270

theorem ratio_problem (a b c : ℝ) (h1 : a / b = 11 / 3) (h2 : a / c = 0.7333333333333333) : 
  b / c = 1 / 5 := by
sorry

end ratio_problem_l272_27270


namespace trigonometric_equation_solution_l272_27243

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) : 
  8.424 * Real.cos x + Real.sqrt (Real.sin x ^ 2 - 2 * Real.sin (2 * x) + 4 * Real.cos x ^ 2) = 0 ↔ 
  (x = Real.arctan (-6.424) + π * (2 * ↑k + 1) ∨ x = Real.arctan 5.212 + π * (2 * ↑k + 1)) :=
by sorry


end trigonometric_equation_solution_l272_27243


namespace fraction_equivalence_l272_27238

theorem fraction_equivalence :
  ∀ (n : ℚ), (4 + n) / (7 + n) = 3 / 4 ↔ n = 5 := by sorry

end fraction_equivalence_l272_27238


namespace quadratic_radicals_combination_l272_27229

theorem quadratic_radicals_combination (x : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 3 * x + 5 = k * (2 * x + 7)) → x = 2 := by
  sorry

end quadratic_radicals_combination_l272_27229


namespace angle_measure_theorem_l272_27259

theorem angle_measure_theorem (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_measure_theorem_l272_27259


namespace smallest_four_digit_divisible_by_six_l272_27288

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_six (n : ℕ) : Prop := n % 6 = 0

theorem smallest_four_digit_divisible_by_six :
  ∀ n : ℕ, is_four_digit n → divisible_by_six n → n ≥ 1002 :=
by sorry

end smallest_four_digit_divisible_by_six_l272_27288


namespace nap_start_time_l272_27201

def minutes_past_midnight (hours minutes : ℕ) : ℕ :=
  hours * 60 + minutes

def time_from_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

theorem nap_start_time 
  (nap_duration : ℕ) 
  (wake_up_hours wake_up_minutes : ℕ) 
  (h1 : nap_duration = 65)
  (h2 : wake_up_hours = 13)
  (h3 : wake_up_minutes = 30) :
  time_from_minutes (minutes_past_midnight wake_up_hours wake_up_minutes - nap_duration) = (12, 25) := by
  sorry

end nap_start_time_l272_27201


namespace salary_reduction_percentage_l272_27213

theorem salary_reduction_percentage (x : ℝ) : 
  (100 - x) * (1 + 53.84615384615385 / 100) = 100 → x = 35 := by
  sorry

end salary_reduction_percentage_l272_27213


namespace unique_solution_for_equation_l272_27211

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.choose n k else 0

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem unique_solution_for_equation (x : ℕ) :
  x ≥ 7 → (3 * C (x - 3) 4 = 5 * A (x - 4) 2) → x = 11 :=
by sorry

end unique_solution_for_equation_l272_27211


namespace lava_lamp_probability_l272_27276

def red_lamps : ℕ := 4
def blue_lamps : ℕ := 3
def green_lamps : ℕ := 3
def total_lamps : ℕ := red_lamps + blue_lamps + green_lamps
def lamps_turned_on : ℕ := 5

def probability_leftmost_green_off_second_right_blue_on : ℚ := 63 / 100

theorem lava_lamp_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps * Nat.choose (total_lamps - red_lamps) blue_lamps
  let leftmost_green_arrangements := Nat.choose (total_lamps - 1) (green_lamps - 1) * Nat.choose (total_lamps - green_lamps) red_lamps * Nat.choose (total_lamps - green_lamps - red_lamps) (blue_lamps - 1)
  let second_right_blue_on_arrangements := Nat.choose (total_lamps - 2) (blue_lamps - 1)
  let remaining_on_lamps := Nat.choose (total_lamps - 2) (lamps_turned_on - 1)
  (leftmost_green_arrangements * second_right_blue_on_arrangements * remaining_on_lamps : ℚ) / (total_arrangements * Nat.choose total_lamps lamps_turned_on) = probability_leftmost_green_off_second_right_blue_on :=
by sorry

end lava_lamp_probability_l272_27276


namespace expand_product_l272_27298

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := by
  sorry

end expand_product_l272_27298


namespace project_completion_theorem_l272_27220

theorem project_completion_theorem (a b c x y z : ℝ) 
  (ha : a / x = 1 / y + 1 / z)
  (hb : b / y = 1 / x + 1 / z)
  (hc : c / z = 1 / x + 1 / y)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1 := by
sorry


end project_completion_theorem_l272_27220


namespace point_on_inverse_graph_and_coordinate_sum_l272_27283

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem point_on_inverse_graph_and_coordinate_sum 
  (h : f 3 = 5/3) : 
  (f_inv (5/3) = 3) ∧ 
  ((1/3) * (f_inv (5/3)) = 1) ∧ 
  (5/3 + 1 = 8/3) := by
sorry

end point_on_inverse_graph_and_coordinate_sum_l272_27283


namespace oil_price_reduction_l272_27222

/-- Proves that a 25% reduction in oil price resulting in 5 kg more for Rs. 900 leads to a reduced price of Rs. 45 per kg -/
theorem oil_price_reduction (original_price : ℝ) (original_quantity : ℝ) : 
  (original_quantity * original_price = 900) →
  ((original_quantity + 5) * (0.75 * original_price) = 900) →
  (0.75 * original_price = 45) :=
by sorry

end oil_price_reduction_l272_27222


namespace saras_hourly_wage_l272_27232

def saras_paycheck (hours_per_week : ℕ) (weeks_worked : ℕ) (tire_cost : ℕ) (money_left : ℕ) : ℚ :=
  let total_earnings := tire_cost + money_left
  let total_hours := hours_per_week * weeks_worked
  (total_earnings : ℚ) / total_hours

theorem saras_hourly_wage :
  saras_paycheck 40 2 410 510 = 11.5 := by
  sorry

end saras_hourly_wage_l272_27232


namespace simplify_expression_l272_27200

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 256) - Real.sqrt (13/2))^2 = (45 - 8 * Real.sqrt 26) / 2 := by
  sorry

end simplify_expression_l272_27200


namespace sara_movie_expenses_l272_27202

-- Define the cost of each item
def theater_ticket_cost : ℚ := 10.62
def theater_ticket_count : ℕ := 2
def rented_movie_cost : ℚ := 1.59
def purchased_movie_cost : ℚ := 13.95

-- Define the total spent on movies
def total_spent : ℚ :=
  theater_ticket_cost * theater_ticket_count + rented_movie_cost + purchased_movie_cost

-- Theorem to prove
theorem sara_movie_expenses : total_spent = 36.78 := by
  sorry

end sara_movie_expenses_l272_27202


namespace total_games_in_season_l272_27248

def total_teams : ℕ := 200
def num_sub_leagues : ℕ := 10
def teams_per_sub_league : ℕ := 20
def regular_season_matches : ℕ := 8
def teams_to_intermediate : ℕ := 5
def teams_to_playoff : ℕ := 2

def regular_season_games (n : ℕ) : ℕ := n * (n - 1) / 2 * regular_season_matches

def intermediate_round_games (n : ℕ) : ℕ := n * (n - 1) / 2

def playoff_round_games (n : ℕ) : ℕ := (n * (n - 1) / 2 - num_sub_leagues * (num_sub_leagues - 1) / 2) * 2

theorem total_games_in_season :
  regular_season_games teams_per_sub_league * num_sub_leagues +
  intermediate_round_games (teams_to_intermediate * num_sub_leagues) +
  playoff_round_games (teams_to_playoff * num_sub_leagues) = 16715 := by
  sorry

end total_games_in_season_l272_27248


namespace min_value_trig_expression_l272_27225

theorem min_value_trig_expression (α β : ℝ) : 
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 := by
  sorry

end min_value_trig_expression_l272_27225


namespace percentage_of_330_l272_27267

theorem percentage_of_330 : (33 + 1 / 3 : ℚ) / 100 * 330 = 110 := by sorry

end percentage_of_330_l272_27267


namespace arithmetic_sequence_property_l272_27228

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_eq : a 1 + 3 * a 8 = 1560) :
  2 * a 9 - a 10 = 507 := by
  sorry

end arithmetic_sequence_property_l272_27228


namespace friend_payment_is_five_l272_27265

/-- The cost per person when splitting a restaurant bill -/
def cost_per_person (num_friends : ℕ) (hamburger_price : ℚ) (num_hamburgers : ℕ)
  (fries_price : ℚ) (num_fries : ℕ) (soda_price : ℚ) (num_sodas : ℕ)
  (spaghetti_price : ℚ) (num_spaghetti : ℕ) : ℚ :=
  (hamburger_price * num_hamburgers + fries_price * num_fries +
   soda_price * num_sodas + spaghetti_price * num_spaghetti) / num_friends

/-- Theorem: Each friend pays $5 when splitting the bill equally -/
theorem friend_payment_is_five :
  cost_per_person 5 3 5 (6/5) 4 (1/2) 5 (27/10) 1 = 5 := by
  sorry

end friend_payment_is_five_l272_27265


namespace equation_solution_l272_27268

theorem equation_solution : ∃ x : ℝ, 2 * (3 * x - 1) = 7 - (x - 5) ∧ x = 2 := by
  sorry

end equation_solution_l272_27268


namespace quadratic_form_equivalence_l272_27204

theorem quadratic_form_equivalence :
  ∀ x y : ℝ, y = (1/2) * x^2 - 2*x + 1 ↔ y = (1/2) * (x - 2)^2 - 1 :=
by sorry

end quadratic_form_equivalence_l272_27204


namespace circle_M_properties_l272_27216

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

-- Define the line that contains the center of the circle
def center_line (x y : ℝ) : Prop :=
  x + y = 2

-- Define the points C and D
def point_C : ℝ × ℝ := (1, -1)
def point_D : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem circle_M_properties :
  (∀ x y, circle_M x y → center_line x y) ∧
  circle_M point_C.1 point_C.2 ∧
  circle_M point_D.1 point_D.2 ∧
  (∀ x y, circle_M x y → 2 - 2 * Real.sqrt 2 ≤ x + y ∧ x + y ≤ 2 + 2 * Real.sqrt 2) :=
sorry

end circle_M_properties_l272_27216


namespace smallest_solution_of_equation_l272_27290

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 64*x^2 + 576 = 0 →
  x ≥ -2 * Real.sqrt 6 ∧
  (∃ y : ℝ, y^4 - 64*y^2 + 576 = 0 ∧ y = -2 * Real.sqrt 6) :=
by sorry

end smallest_solution_of_equation_l272_27290


namespace nancy_balloons_l272_27289

theorem nancy_balloons (mary_balloons : ℕ) (nancy_balloons : ℕ) : 
  mary_balloons = 28 → 
  mary_balloons = 4 * nancy_balloons → 
  nancy_balloons = 7 := by
sorry

end nancy_balloons_l272_27289


namespace cubic_term_of_line_l272_27284

-- Define the line equation
def line_equation (x : ℝ) : ℝ := x^2 - x^3

-- State the theorem
theorem cubic_term_of_line : 
  ∃ (a b c d : ℝ), 
    (∀ x, line_equation x = a*x^3 + b*x^2 + c*x + d) ∧ 
    (a = -1) := by
  sorry

end cubic_term_of_line_l272_27284


namespace jackie_has_more_fruits_than_adam_l272_27236

/-- Represents the number of fruits a person has -/
structure FruitCount where
  apples : ℕ
  oranges : ℕ
  bananas : ℚ

/-- Calculates the difference in total apples and oranges between two FruitCounts -/
def applePlusOrangeDifference (a b : FruitCount) : ℤ :=
  (b.apples + b.oranges : ℤ) - (a.apples + a.oranges)

theorem jackie_has_more_fruits_than_adam :
  let adam : FruitCount := { apples := 25, oranges := 34, bananas := 18.5 }
  let jackie : FruitCount := { apples := 43, oranges := 29, bananas := 16.5 }
  applePlusOrangeDifference adam jackie = 13 := by
  sorry

end jackie_has_more_fruits_than_adam_l272_27236


namespace C_not_necessarily_necessary_for_A_C_not_necessarily_sufficient_for_A_l272_27272

-- Define propositions A, B, and C
variable (A B C : Prop)

-- C is a necessary condition for B
axiom C_necessary_for_B : B → C

-- B is a sufficient condition for A
axiom B_sufficient_for_A : B → A

-- Theorem: C is not necessarily a necessary condition for A
theorem C_not_necessarily_necessary_for_A : ¬(A → C) := by sorry

-- Theorem: C is not necessarily a sufficient condition for A
theorem C_not_necessarily_sufficient_for_A : ¬(C → A) := by sorry

end C_not_necessarily_necessary_for_A_C_not_necessarily_sufficient_for_A_l272_27272


namespace ellipse_properties_l272_27279

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 9 * x^2 + y^2 = 81

-- Define the major axis length
def major_axis_length : ℝ := 18

-- Define the foci coordinates
def foci_coordinates : Set (ℝ × ℝ) := {(0, 6*Real.sqrt 2), (0, -6*Real.sqrt 2)}

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 36

-- Theorem statement
theorem ellipse_properties :
  (∀ x y, ellipse x y → 
    (major_axis_length = 18 ∧ 
     (x, y) ∈ foci_coordinates → 
     (x = 0 ∧ (y = 6*Real.sqrt 2 ∨ y = -6*Real.sqrt 2)))) ∧
  (∀ x y, hyperbola x y → 
    (∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ 
                  c = 6*Real.sqrt 2 ∧ 
                  c/a = Real.sqrt 2)) :=
sorry

end ellipse_properties_l272_27279


namespace parabola_max_vertex_sum_l272_27212

theorem parabola_max_vertex_sum (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let vertex_sum := 
    let h : ℝ := T
    let k : ℝ := -a * T^2
    h + k
  (passes_through 0 0) ∧ 
  (passes_through (2 * T) 0) ∧ 
  (passes_through (T + 2) 32) →
  (∀ N : ℝ, N = vertex_sum → N ≤ 68) ∧ 
  (∃ N : ℝ, N = vertex_sum ∧ N = 68) :=
by sorry

end parabola_max_vertex_sum_l272_27212


namespace hcl_moles_formed_l272_27239

-- Define the chemical equation
structure ChemicalEquation where
  reactants : List (String × ℕ)
  products : List (String × ℕ)

-- Define the reaction
def reaction : ChemicalEquation :=
  { reactants := [("CH4", 1), ("Cl2", 4)],
    products := [("CCl4", 1), ("HCl", 4)] }

-- Define the initial quantities
def initialQuantities : List (String × ℕ) :=
  [("CH4", 1), ("Cl2", 4)]

-- Theorem to prove
theorem hcl_moles_formed (reaction : ChemicalEquation) (initialQuantities : List (String × ℕ)) :
  reaction.reactants = [("CH4", 1), ("Cl2", 4)] →
  reaction.products = [("CCl4", 1), ("HCl", 4)] →
  initialQuantities = [("CH4", 1), ("Cl2", 4)] →
  (List.find? (λ p => p.1 = "HCl") reaction.products).map Prod.snd = some 4 := by
  sorry

end hcl_moles_formed_l272_27239


namespace pig_price_calculation_l272_27241

theorem pig_price_calculation (num_cows : ℕ) (num_pigs : ℕ) (price_per_cow : ℕ) (total_revenue : ℕ) :
  num_cows = 20 →
  num_pigs = 4 * num_cows →
  price_per_cow = 800 →
  total_revenue = 48000 →
  (total_revenue - num_cows * price_per_cow) / num_pigs = 400 := by
  sorry

end pig_price_calculation_l272_27241


namespace power_function_properties_l272_27215

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (2 * m^2 - 2 * m - 3) * x^2

-- State the theorem
theorem power_function_properties (m : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) →  -- f is monotonically decreasing on (0, +∞)
  (f m 8 = Real.sqrt 2 / 4) ∧               -- f(8) = √2/4
  (∀ x, f m (x^2 + 2*x) < f m (x + 6) ↔ x ∈ Set.Ioo (-6) (-3) ∪ Set.Ioi 2) :=
by sorry


end power_function_properties_l272_27215


namespace fraction_equality_sum_l272_27207

theorem fraction_equality_sum (P Q : ℚ) : 
  (5 : ℚ) / 7 = P / 63 ∧ (5 : ℚ) / 7 = 70 / Q → P + Q = 143 := by
  sorry

end fraction_equality_sum_l272_27207


namespace baker_sales_difference_l272_27281

theorem baker_sales_difference (cakes_made pastries_made cakes_sold pastries_sold : ℕ) :
  cakes_made = 157 →
  pastries_made = 169 →
  cakes_sold = 158 →
  pastries_sold = 147 →
  cakes_sold - pastries_sold = 11 := by
sorry

end baker_sales_difference_l272_27281


namespace system_solutions_correct_l272_27252

theorem system_solutions_correct : 
  (∃ (x y : ℝ), 3*x - y = -1 ∧ x + 2*y = 9 ∧ x = 1 ∧ y = 4) ∧
  (∃ (x y : ℝ), x/4 + y/3 = 4/3 ∧ 5*(x - 9) = 4*(y - 13/4) ∧ x = 6 ∧ y = -1/2) := by
  sorry

end system_solutions_correct_l272_27252


namespace identity_proof_l272_27263

theorem identity_proof (a b c : ℝ) 
  (h1 : (a - c) / (a + c) ≠ 0)
  (h2 : (b - c) / (b + c) ≠ 0)
  (h3 : (a + c) / (a - c) + (b + c) / (b - c) ≠ 0) :
  ((((a - c) / (a + c) + (b - c) / (b + c)) / ((a + c) / (a - c) + (b + c) / (b - c))) ^ 2) = 
  ((((a - c) / (a + c)) ^ 2 + ((b - c) / (b + c)) ^ 2) / (((a + c) / (a - c)) ^ 2 + ((b + c) / (b - c)) ^ 2)) :=
by sorry

end identity_proof_l272_27263


namespace mod_equivalence_unique_solution_l272_27285

theorem mod_equivalence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -2023 [ZMOD 9] ∧ n = 2 := by
  sorry

end mod_equivalence_unique_solution_l272_27285


namespace molecular_weight_of_C2H5Cl2O2_l272_27251

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of C2H5Cl2O2 in g/mol -/
def molecular_weight : ℝ := 2 * carbon_weight + 5 * hydrogen_weight + 2 * chlorine_weight + 2 * oxygen_weight

/-- Theorem stating that the molecular weight of C2H5Cl2O2 is 132.96 g/mol -/
theorem molecular_weight_of_C2H5Cl2O2 : molecular_weight = 132.96 := by
  sorry

end molecular_weight_of_C2H5Cl2O2_l272_27251


namespace damaged_books_count_damaged_books_proof_l272_27253

theorem damaged_books_count : ℕ → ℕ → Prop :=
  fun obsolete damaged =>
    (obsolete = 6 * damaged - 8) →
    (obsolete + damaged = 69) →
    (damaged = 11)

-- The proof is omitted
theorem damaged_books_proof : damaged_books_count 58 11 := by sorry

end damaged_books_count_damaged_books_proof_l272_27253


namespace simplify_sqrt_fraction_l272_27206

theorem simplify_sqrt_fraction : 
  (Real.sqrt 45) / (2 * Real.sqrt 20) = 3 / 4 := by
  sorry

end simplify_sqrt_fraction_l272_27206


namespace absolute_value_inequality_solution_l272_27287

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x + 2| + |x - 2| < x + 7) ↔ (-7/3 < x ∧ x < 7) := by sorry

end absolute_value_inequality_solution_l272_27287


namespace original_number_proof_l272_27205

theorem original_number_proof : 
  ∃! x : ℕ, (x + 4) % 23 = 0 ∧ ∀ y : ℕ, y < 4 → (x + y) % 23 ≠ 0 :=
by
  sorry

end original_number_proof_l272_27205


namespace min_sum_absolute_values_l272_27266

theorem min_sum_absolute_values (x : ℝ) :
  ∃ (min : ℝ), min = 4 ∧ 
  (∀ y : ℝ, |y + 3| + |y + 6| + |y + 7| ≥ min) ∧
  (|x + 3| + |x + 6| + |x + 7| = min) :=
sorry

end min_sum_absolute_values_l272_27266


namespace f_difference_l272_27219

/-- The function f(x) = x^4 + 3x^3 + x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 3*x^3 + x^2 + 7*x

/-- Theorem: f(3) - f(-3) = 204 -/
theorem f_difference : f 3 - f (-3) = 204 := by
  sorry

end f_difference_l272_27219


namespace colored_plane_congruent_triangle_l272_27224

/-- A color type representing the 1992 colors -/
inductive Color
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10
-- ... (omitted for brevity, but in reality, this would list all 1992 colors)
| c1992

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle on the plane -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- A colored plane -/
def ColoredPlane := Point → Color

/-- Two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- A point is an interior point of a line segment -/
def isInteriorPoint (p : Point) (a b : Point) : Prop := sorry

/-- The theorem to be proved -/
theorem colored_plane_congruent_triangle 
  (plane : ColoredPlane) (T : Triangle) : 
  ∃ T' : Triangle, congruent T T' ∧ 
    (∀ (p q : Point), 
      ((isInteriorPoint p T'.a T'.b ∧ isInteriorPoint q T'.b T'.c) ∨
       (isInteriorPoint p T'.b T'.c ∧ isInteriorPoint q T'.c T'.a) ∨
       (isInteriorPoint p T'.c T'.a ∧ isInteriorPoint q T'.a T'.b)) →
      plane p = plane q) :=
sorry

end colored_plane_congruent_triangle_l272_27224


namespace power_of_power_l272_27294

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end power_of_power_l272_27294


namespace parabola_vertex_m_value_l272_27282

theorem parabola_vertex_m_value (m : ℝ) :
  let f (x : ℝ) := 3 * x^2 + 6 * Real.sqrt m * x + 36
  let vertex_y := f (-(Real.sqrt m) / 3)
  vertex_y = 33 → m = 1 := by
sorry

end parabola_vertex_m_value_l272_27282


namespace perpendicular_line_equation_l272_27230

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 3 (-2) 0 →
  point = Point.mk 1 (-1) →
  ∃ (result_line : Line),
    result_line.perpendicular given_line ∧
    point.liesOn result_line ∧
    result_line = Line.mk 2 3 1 := by
  sorry

end perpendicular_line_equation_l272_27230


namespace student_group_arrangements_l272_27245

/-- The number of ways to divide n students into k equal groups -/
def divide_students (n k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k different topics -/
def assign_topics (k : ℕ) : ℕ := sorry

theorem student_group_arrangements :
  let n : ℕ := 6  -- number of students
  let k : ℕ := 3  -- number of groups
  divide_students n k * assign_topics k = 540 :=
by sorry

end student_group_arrangements_l272_27245


namespace arc_length_300_degrees_l272_27291

/-- The length of an arc in a circle with radius 2 and central angle 300° is 10π/3 -/
theorem arc_length_300_degrees (r : ℝ) (θ : ℝ) : 
  r = 2 → θ = 300 * π / 180 → r * θ = 10 * π / 3 := by sorry

end arc_length_300_degrees_l272_27291


namespace shape_relations_l272_27297

/-- Given symbols representing geometric shapes with the following relations:
    - triangle + triangle = star
    - circle = square + square
    - triangle = circle + circle + circle + circle
    Prove that star divided by square equals 16 -/
theorem shape_relations (triangle star circle square : ℕ) 
    (h1 : triangle + triangle = star)
    (h2 : circle = square + square)
    (h3 : triangle = circle + circle + circle + circle) :
  star / square = 16 := by sorry

end shape_relations_l272_27297


namespace simplify_expression_l272_27255

theorem simplify_expression (a b : ℝ) : (25*a + 70*b) + (15*a + 34*b) - (12*a + 55*b) = 28*a + 49*b := by
  sorry

end simplify_expression_l272_27255


namespace quadratic_functions_equality_l272_27264

/-- Given a quadratic function f(x) = x² + bx + 8 with b ≠ 0 and two distinct real roots x₁ and x₂,
    and a quadratic function g(x) with quadratic coefficient 1 and roots x₁ + 1/x₂ and x₂ + 1/x₁,
    prove that if g(1) = f(1), then g(1) = -8. -/
theorem quadratic_functions_equality (b : ℝ) (x₁ x₂ : ℝ) :
  b ≠ 0 →
  x₁ ≠ x₂ →
  (∀ x, x^2 + b*x + 8 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ c d : ℝ, ∀ x, (x - (x₁ + 1/x₂)) * (x - (x₂ + 1/x₁)) = x^2 + c*x + d) →
  (1^2 + b*1 + 8 = 1^2 + c*1 + d) →
  1^2 + c*1 + d = -8 :=
by sorry

end quadratic_functions_equality_l272_27264


namespace sally_bought_twenty_cards_l272_27233

/-- Calculates the number of Pokemon cards Sally bought -/
def cards_sally_bought (initial : ℕ) (from_dan : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial + from_dan)

/-- Proves that Sally bought 20 Pokemon cards -/
theorem sally_bought_twenty_cards : 
  cards_sally_bought 27 41 88 = 20 := by
  sorry

end sally_bought_twenty_cards_l272_27233


namespace obesity_probability_l272_27226

theorem obesity_probability (P_obese_male P_obese_female : ℝ) 
  (ratio_male_female : ℚ) :
  P_obese_male = 1/5 →
  P_obese_female = 1/10 →
  ratio_male_female = 3/2 →
  let P_male := ratio_male_female / (1 + ratio_male_female)
  let P_female := 1 - P_male
  let P_obese := P_male * P_obese_male + P_female * P_obese_female
  (P_male * P_obese_male) / P_obese = 3/4 := by
sorry

end obesity_probability_l272_27226


namespace inner_circle_radius_l272_27250

/-- Given a circle of radius R and a point A on its diameter at distance a from the center,
    the radius of the circle that touches the diameter at A and is internally tangent to the given circle
    is (R^2 - a^2) / (2R). -/
theorem inner_circle_radius (R a : ℝ) (h₁ : R > 0) (h₂ : 0 < a ∧ a < R) :
  ∃ x : ℝ, x > 0 ∧ x = (R^2 - a^2) / (2*R) ∧
  x^2 + a^2 = (R - x)^2 :=
sorry

end inner_circle_radius_l272_27250


namespace container_cubes_theorem_l272_27240

/-- Represents the dimensions of a rectangular container -/
structure ContainerDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℕ

/-- Calculate the number of cubes that can fit in the container -/
def cubesFit (container : ContainerDimensions) (cube : CubeDimensions) : ℕ :=
  (container.length / cube.side) * (container.width / cube.side) * (container.height / cube.side)

/-- Calculate the volume of the container -/
def containerVolume (container : ContainerDimensions) : ℕ :=
  container.length * container.width * container.height

/-- Calculate the volume of a single cube -/
def cubeVolume (cube : CubeDimensions) : ℕ :=
  cube.side * cube.side * cube.side

/-- Calculate the fraction of the container volume occupied by cubes -/
def occupiedFraction (container : ContainerDimensions) (cube : CubeDimensions) : ℚ :=
  (cubesFit container cube * cubeVolume cube : ℚ) / containerVolume container

theorem container_cubes_theorem (container : ContainerDimensions) (cube : CubeDimensions) 
  (h1 : container.length = 8)
  (h2 : container.width = 4)
  (h3 : container.height = 9)
  (h4 : cube.side = 2) :
  cubesFit container cube = 32 ∧ occupiedFraction container cube = 8/9 := by
  sorry

#eval cubesFit ⟨8, 4, 9⟩ ⟨2⟩
#eval occupiedFraction ⟨8, 4, 9⟩ ⟨2⟩

end container_cubes_theorem_l272_27240


namespace subtract_three_from_binary_l272_27254

/-- Converts a binary number (represented as a list of bits) to decimal --/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to binary (represented as a list of bits) --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2) :: acc)
  aux n []

theorem subtract_three_from_binary :
  let M : List Nat := [0, 1, 0, 1, 0, 1]  -- 101010 in binary
  let M_decimal : Nat := binary_to_decimal M
  let result : List Nat := decimal_to_binary (M_decimal - 3)
  result = [1, 1, 1, 0, 0, 1] -- 100111 in binary
  := by sorry

end subtract_three_from_binary_l272_27254


namespace journey_distance_l272_27231

/-- Proves that a journey with given conditions has a total distance of 224 km -/
theorem journey_distance (total_time : ℝ) (speed_first_half : ℝ) (speed_second_half : ℝ)
  (h1 : total_time = 10)
  (h2 : speed_first_half = 21)
  (h3 : speed_second_half = 24) :
  ∃ (distance : ℝ),
    distance = 224 ∧
    total_time = (distance / 2) / speed_first_half + (distance / 2) / speed_second_half :=
by sorry

end journey_distance_l272_27231


namespace tangent_line_b_value_l272_27246

/-- A curve defined by y = -x³ + 2 -/
def curve (x : ℝ) : ℝ := -x^3 + 2

/-- A line defined by y = -6x + b -/
def line (b : ℝ) (x : ℝ) : ℝ := -6*x + b

/-- The derivative of the curve -/
def curve_derivative (x : ℝ) : ℝ := -3*x^2

theorem tangent_line_b_value :
  ∀ b : ℝ,
  (∃ x : ℝ, curve x = line b x ∧ curve_derivative x = -6) →
  (b = 2 + 4 * Real.sqrt 2 ∨ b = 2 - 4 * Real.sqrt 2) :=
by sorry

end tangent_line_b_value_l272_27246


namespace second_term_of_sequence_l272_27296

theorem second_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = n * (2 * n + 1)) → 
  a 2 = 7 := by
sorry

end second_term_of_sequence_l272_27296


namespace waiter_tips_fraction_l272_27218

theorem waiter_tips_fraction (base_salary : ℚ) : 
  let tips := (5 / 4) * base_salary
  let total_income := base_salary + tips
  let expenses := (1 / 8) * base_salary
  let taxes := (1 / 5) * total_income
  let after_tax_income := total_income - taxes
  (tips / after_tax_income) = 25 / 36 :=
by sorry

end waiter_tips_fraction_l272_27218


namespace intersection_y_coordinate_l272_27249

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the tangent line at a point (a, 4a^2) on the parabola
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 8 * a * x - 4 * a^2

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop := 8 * a * 8 * b = -1

-- Theorem statement
theorem intersection_y_coordinate (a b : ℝ) : 
  a ≠ b → 
  perpendicular_tangents a b →
  ∃ x, tangent_line a x = tangent_line b x ∧ tangent_line a x = -1/4 :=
sorry

end intersection_y_coordinate_l272_27249


namespace right_triangle_leg_square_l272_27242

/-- In a right triangle, if the hypotenuse exceeds one leg by 2, then the square of the other leg is 4a + 4 -/
theorem right_triangle_leg_square (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) -- Pythagorean theorem
  (h5 : c = a + 2) : -- Hypotenuse exceeds one leg by 2
  b^2 = 4*a + 4 := by sorry

end right_triangle_leg_square_l272_27242


namespace tan_alpha_value_l272_27274

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end tan_alpha_value_l272_27274


namespace compare_numbers_l272_27244

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def num_base_6 : List Nat := [5, 4]
def num_base_4 : List Nat := [2, 3]
def num_base_5 : List Nat := [3, 2, 1]

theorem compare_numbers :
  (base_to_decimal num_base_6 6 + base_to_decimal num_base_4 4) > base_to_decimal num_base_5 5 :=
by sorry

end compare_numbers_l272_27244


namespace x_value_when_y_is_two_l272_27275

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end x_value_when_y_is_two_l272_27275


namespace peter_train_probability_l272_27258

theorem peter_train_probability (p : ℚ) (h : p = 5/12) : 1 - p = 7/12 := by
  sorry

end peter_train_probability_l272_27258


namespace expression_value_l272_27277

theorem expression_value (x y z : ℝ) 
  (eq1 : 4*x - 6*y - 2*z = 0)
  (eq2 : x + 2*y - 10*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - x*y) / (y^2 + z^2) = 26/25 := by
  sorry

end expression_value_l272_27277


namespace appliance_purchase_total_cost_l272_27286

theorem appliance_purchase_total_cost : 
  let vacuum_original : ℝ := 250
  let vacuum_discount : ℝ := 0.20
  let dishwasher_cost : ℝ := 450
  let bundle_discount : ℝ := 75
  let sales_tax : ℝ := 0.07

  let vacuum_discounted : ℝ := vacuum_original * (1 - vacuum_discount)
  let subtotal : ℝ := vacuum_discounted + dishwasher_cost - bundle_discount
  let total_with_tax : ℝ := subtotal * (1 + sales_tax)

  total_with_tax = 615.25 := by sorry

end appliance_purchase_total_cost_l272_27286


namespace problem_solution_l272_27235

theorem problem_solution (x y : ℝ) 
  (hx : 2 < x ∧ x < 3) 
  (hy : -2 < y ∧ y < -1) 
  (hxy : x < y ∧ y < 0) : 
  (0 < x + y ∧ x + y < 2) ∧ 
  (3 < x - y ∧ x - y < 5) ∧ 
  (-6 < x * y ∧ x * y < -2) ∧
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end problem_solution_l272_27235


namespace area_BEIH_l272_27293

/-- Given a 2×2 square ABCD with B at (0,0), E is the midpoint of AB, 
    F is the midpoint of BC, I is the intersection of AF and DE, 
    and H is the intersection of BD and AF. -/
def square_setup (A B C D E F H I : ℝ × ℝ) : Prop :=
  B = (0, 0) ∧ 
  C = (2, 0) ∧ 
  D = (2, 2) ∧ 
  A = (0, 2) ∧
  E = (0, 1) ∧
  F = (1, 0) ∧
  H.1 = H.2 ∧ -- H is on the diagonal BD
  I.2 = -2 * I.1 + 2 ∧ -- I is on line AF
  I.2 = (1/2) * I.1 + 1 -- I is on line DE

/-- The area of quadrilateral BEIH is 7/15 -/
theorem area_BEIH (A B C D E F H I : ℝ × ℝ) 
  (h : square_setup A B C D E F H I) : 
  let area := (1/2) * abs ((E.1 * I.2 + I.1 * H.2 + H.1 * B.2 + B.1 * E.2) - 
                           (E.2 * I.1 + I.2 * H.1 + H.2 * B.1 + B.2 * E.1))
  area = 7/15 := by
sorry

end area_BEIH_l272_27293


namespace inequality_proof_l272_27247

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  a * b + b * c + c * a ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c ∧ Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3 := by
  sorry

end inequality_proof_l272_27247


namespace smallest_X_value_l272_27299

/-- A function that checks if a natural number is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The theorem stating the smallest possible value of X -/
theorem smallest_X_value (T : ℕ) (hT : T > 0) (hComposed : isComposedOf0sAnd1s T) 
    (hDivisible : T % 15 = 0) : 
  ∀ X : ℕ, (X * 15 = T) → X ≥ 7400 := by
  sorry

end smallest_X_value_l272_27299


namespace binomial_distribution_parameters_l272_27203

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_parameters 
  (X : BinomialDistribution) 
  (h_expectation : expectation X = 2)
  (h_variance : variance X = 4) :
  X.n = 12 ∧ X.p = 2/3 := by
  sorry

end binomial_distribution_parameters_l272_27203


namespace probability_of_condition_l272_27227

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ (m : ℕ), 4 ∣ m ∧ a * b + a + b = m - 1

def total_valid_pairs : ℕ := Nat.choose 60 2

def satisfying_pairs : ℕ := 1350

theorem probability_of_condition :
  (satisfying_pairs : ℚ) / total_valid_pairs = 45 / 59 :=
sorry

end probability_of_condition_l272_27227


namespace rectangle_area_l272_27214

-- Define the radius of the inscribed circle
def circle_radius : ℝ := 7

-- Define the ratio of length to width
def length_width_ratio : ℝ := 2

-- Theorem statement
theorem rectangle_area (width : ℝ) (length : ℝ) 
  (h1 : width = 2 * circle_radius) 
  (h2 : length = length_width_ratio * width) : 
  width * length = 392 := by
  sorry


end rectangle_area_l272_27214


namespace container_volume_ratio_l272_27256

theorem container_volume_ratio :
  ∀ (v1 v2 : ℚ),
  v1 > 0 → v2 > 0 →
  (5 / 6 : ℚ) * v1 = (3 / 4 : ℚ) * v2 →
  v1 / v2 = (9 / 10 : ℚ) := by
sorry

end container_volume_ratio_l272_27256


namespace systematic_sample_result_l272_27278

/-- Systematic sampling function -/
def systematicSample (populationSize sampleSize : ℕ) (firstSelected : ℕ) : ℕ → ℕ :=
  fun n => firstSelected + (n - 1) * (populationSize / sampleSize)

theorem systematic_sample_result 
  (populationSize sampleSize firstSelected : ℕ) 
  (h1 : populationSize = 800) 
  (h2 : sampleSize = 50) 
  (h3 : firstSelected = 11) 
  (h4 : firstSelected ≤ 16) :
  ∃ n : ℕ, 33 ≤ n ∧ n ≤ 48 ∧ systematicSample populationSize sampleSize firstSelected n = 43 :=
by
  sorry

end systematic_sample_result_l272_27278


namespace reading_time_difference_l272_27292

/-- The reading problem setup -/
structure ReadingProblem where
  xanthia_rate : ℕ  -- pages per hour
  molly_rate : ℕ    -- pages per hour
  book_pages : ℕ
  
/-- Calculate the time difference in minutes -/
def time_difference (p : ReadingProblem) : ℕ :=
  ((p.book_pages / p.molly_rate - p.book_pages / p.xanthia_rate) * 60 : ℕ)

/-- The main theorem -/
theorem reading_time_difference (p : ReadingProblem) 
  (h1 : p.xanthia_rate = 120)
  (h2 : p.molly_rate = 60)
  (h3 : p.book_pages = 360) : 
  time_difference p = 180 := by
  sorry

end reading_time_difference_l272_27292


namespace quadratic_one_root_l272_27261

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + 2*m = 0) → m = 2/9 :=
by sorry

end quadratic_one_root_l272_27261


namespace circle_trajectory_and_min_distance_l272_27210

-- Define the moving circle
def moving_circle (x y : ℝ) : Prop :=
  y > 0 ∧ Real.sqrt (x^2 + (y - 1)^2) = y + 1

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop :=
  y > 0 ∧ x^2 = 4*y

-- Define points A and B on trajectory E
def point_on_E (x y : ℝ) : Prop :=
  trajectory_E x y

-- Define the perpendicular tangents condition
def perpendicular_tangents (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  point_on_E x₁ y₁ ∧ point_on_E x₂ y₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -4

-- Main theorem
theorem circle_trajectory_and_min_distance :
  (∀ x y, moving_circle x y ↔ trajectory_E x y) ∧
  (∀ x₁ y₁ x₂ y₂, perpendicular_tangents x₁ y₁ x₂ y₂ →
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ 4) ∧
  (∃ x₁ y₁ x₂ y₂, perpendicular_tangents x₁ y₁ x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4) :=
sorry

end circle_trajectory_and_min_distance_l272_27210


namespace curve_equation_and_m_range_l272_27273

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ Real.sqrt ((p.1 - 1)^2 + p.2^2) - p.1 = 1}

-- Define the function for the dot product of vectors FA and FB
def dotProductFAFB (m : ℝ) (A B : ℝ × ℝ) : ℝ :=
  (A.1 - 1) * (B.1 - 1) + A.2 * B.2

theorem curve_equation_and_m_range :
  -- Part 1: The equation of curve C
  (∀ p : ℝ × ℝ, p ∈ C ↔ p.1 > 0 ∧ p.2^2 = 4 * p.1) ∧
  -- Part 2: Existence of m
  (∃ m : ℝ, m > 0 ∧
    ∀ A B : ℝ × ℝ, A ∈ C → B ∈ C →
      (∃ t : ℝ, A.1 = t * A.2 + m ∧ B.1 = t * B.2 + m) →
        dotProductFAFB m A B < 0) ∧
  -- Part 3: Range of m
  (∀ m : ℝ, (∀ A B : ℝ × ℝ, A ∈ C → B ∈ C →
    (∃ t : ℝ, A.1 = t * A.2 + m ∧ B.1 = t * B.2 + m) →
      dotProductFAFB m A B < 0) ↔
        m > 3 - 2 * Real.sqrt 2 ∧ m < 3 + 2 * Real.sqrt 2) :=
by sorry

end curve_equation_and_m_range_l272_27273


namespace largest_sum_and_simplification_l272_27257

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/8]
  (∀ x ∈ sums, x ≤ 1/3 + 1/2) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end largest_sum_and_simplification_l272_27257


namespace max_value_of_f_l272_27208

-- Define the function
def f (x : ℝ) : ℝ := 12 * x - 4 * x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 11 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l272_27208
