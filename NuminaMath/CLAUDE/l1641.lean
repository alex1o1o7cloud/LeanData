import Mathlib

namespace angle_AOD_measure_l1641_164134

-- Define the angles
variable (AOB BOC COD AOD : ℝ)

-- Define the conditions
axiom angles_equal : AOB = BOC ∧ BOC = COD
axiom AOD_smaller : AOD = AOB / 3

-- Define the distinctness of rays (we can't directly represent this in angles, so we'll skip it)

-- Define the theorem
theorem angle_AOD_measure :
  (AOB + BOC + COD + AOD = 360 ∨ AOB + BOC + COD - AOD = 360) →
  AOD = 36 ∨ AOD = 45 := by
  sorry

end angle_AOD_measure_l1641_164134


namespace existence_of_alpha_beta_l1641_164140

-- Define the Intermediate Value Property
def has_intermediate_value_property (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ y, (f a < y ∧ y < f b) ∨ (f b < y ∧ y < f a) → ∃ c, a < c ∧ c < b ∧ f c = y

-- State the theorem
theorem existence_of_alpha_beta
  (f : ℝ → ℝ) (a b : ℝ) (h_ivp : has_intermediate_value_property f a b)
  (h_sign : f a * f b < 0) :
  ∃ α β, a < α ∧ α < β ∧ β < b ∧ f α + f β = f α * f β :=
sorry

end existence_of_alpha_beta_l1641_164140


namespace total_weight_juvenile_female_muscovy_l1641_164109

/-- Given a pond with ducks, calculate the total weight of juvenile female Muscovy ducks -/
theorem total_weight_juvenile_female_muscovy (total_ducks : ℕ) 
  (muscovy_percentage mallard_percentage : ℚ)
  (female_muscovy_percentage : ℚ) 
  (juvenile_female_muscovy_percentage : ℚ)
  (avg_weight_juvenile_female_muscovy : ℚ) :
  total_ducks = 120 →
  muscovy_percentage = 45/100 →
  mallard_percentage = 35/100 →
  female_muscovy_percentage = 60/100 →
  juvenile_female_muscovy_percentage = 30/100 →
  avg_weight_juvenile_female_muscovy = 7/2 →
  ∃ (weight : ℚ), weight = 63/2 ∧ 
    weight = (total_ducks : ℚ) * muscovy_percentage * female_muscovy_percentage * 
             juvenile_female_muscovy_percentage * avg_weight_juvenile_female_muscovy :=
by sorry

end total_weight_juvenile_female_muscovy_l1641_164109


namespace parabola_shift_l1641_164195

def original_parabola (x : ℝ) : ℝ := -x^2

def shifted_parabola (x : ℝ) : ℝ := -(x - 2)^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 2) :=
by
  sorry

end parabola_shift_l1641_164195


namespace dealership_sales_prediction_l1641_164181

/-- Represents the sales prediction for a car dealership -/
structure SalesPrediction where
  sportsCarsRatio : ℕ
  sedansRatio : ℕ
  predictedSportsCars : ℕ

/-- Calculates the expected sedan sales and total vehicles needed -/
def calculateSales (pred : SalesPrediction) : ℕ × ℕ :=
  let expectedSedans := pred.predictedSportsCars * pred.sedansRatio / pred.sportsCarsRatio
  let totalVehicles := pred.predictedSportsCars + expectedSedans
  (expectedSedans, totalVehicles)

/-- Theorem stating the expected sales for the given scenario -/
theorem dealership_sales_prediction :
  let pred : SalesPrediction := {
    sportsCarsRatio := 3,
    sedansRatio := 5,
    predictedSportsCars := 36
  }
  calculateSales pred = (60, 96) := by
  sorry

end dealership_sales_prediction_l1641_164181


namespace si_o_bond_is_polar_covalent_l1641_164158

-- Define the electronegativity values
def electronegativity_Si : ℝ := 1.90
def electronegativity_O : ℝ := 3.44

-- Define the range for polar covalent bonds
def polar_covalent_lower_bound : ℝ := 0.5
def polar_covalent_upper_bound : ℝ := 1.7

-- Define a function to check if a bond is polar covalent
def is_polar_covalent (electronegativity_diff : ℝ) : Prop :=
  polar_covalent_lower_bound ≤ electronegativity_diff ∧
  electronegativity_diff ≤ polar_covalent_upper_bound

-- Theorem: The silicon-oxygen bonds in SiO2 are polar covalent
theorem si_o_bond_is_polar_covalent :
  is_polar_covalent (electronegativity_O - electronegativity_Si) :=
by
  sorry


end si_o_bond_is_polar_covalent_l1641_164158


namespace arithmetic_sequence_third_term_l1641_164112

/-- Given an arithmetic sequence where the sum of the first and fifth terms is 14,
    prove that the third term is 7. -/
theorem arithmetic_sequence_third_term
  (a : ℝ)  -- First term of the sequence
  (d : ℝ)  -- Common difference of the sequence
  (h : a + (a + 4*d) = 14)  -- Sum of first and fifth terms is 14
  : a + 2*d = 7 :=  -- Third term is 7
by sorry

end arithmetic_sequence_third_term_l1641_164112


namespace total_harvest_is_2000_l1641_164127

/-- Represents the harvest of tomatoes over three days -/
structure TomatoHarvest where
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total harvest over three days -/
def total_harvest (h : TomatoHarvest) : ℕ :=
  h.wednesday + h.thursday + h.friday

/-- Theorem stating the total harvest is 2000 kg given the conditions -/
theorem total_harvest_is_2000 (h : TomatoHarvest) 
  (h_wed : h.wednesday = 400)
  (h_thu : h.thursday = h.wednesday / 2)
  (h_fri : h.friday - 700 = 700) :
  total_harvest h = 2000 := by
  sorry

#check total_harvest_is_2000

end total_harvest_is_2000_l1641_164127


namespace xy_value_l1641_164194

theorem xy_value (x y : ℝ) 
  (eq1 : (4 : ℝ)^x / (2 : ℝ)^(x + y) = 8)
  (eq2 : (9 : ℝ)^(x + y) / (3 : ℝ)^(5 * y) = 243) :
  x * y = 4 := by
  sorry

end xy_value_l1641_164194


namespace bank_charge_increase_l1641_164144

/-- The percentage increase in the ratio of price to transactions from the old
    charging system to the new charging system -/
theorem bank_charge_increase (old_price : ℝ) (old_transactions : ℕ)
    (new_price : ℝ) (new_transactions : ℕ) :
    old_price = 1 →
    old_transactions = 5 →
    new_price = 0.75 →
    new_transactions = 3 →
    (((new_price / new_transactions) - (old_price / old_transactions)) /
     (old_price / old_transactions)) * 100 = 25 := by
  sorry

end bank_charge_increase_l1641_164144


namespace max_min_values_l1641_164170

theorem max_min_values (x y : ℝ) (h : |5*x + y| + |5*x - y| = 20) :
  (∃ (a b : ℝ), a^2 - a*b + b^2 = 124 ∧ 
   ∀ (c d : ℝ), |5*c + d| + |5*c - d| = 20 → c^2 - c*d + d^2 ≤ 124) ∧
  (∃ (a b : ℝ), a^2 - a*b + b^2 = 4 ∧ 
   ∀ (c d : ℝ), |5*c + d| + |5*c - d| = 20 → c^2 - c*d + d^2 ≥ 4) :=
by sorry

end max_min_values_l1641_164170


namespace solutions_of_f_of_f_eq_x_l1641_164152

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 5*x + 1

-- State the theorem
theorem solutions_of_f_of_f_eq_x :
  ∀ x : ℝ, f (f x) = x ↔ (x = -2 - Real.sqrt 3 ∨ x = -2 + Real.sqrt 3 ∨ x = -3 - Real.sqrt 2 ∨ x = -3 + Real.sqrt 2) :=
by sorry

end solutions_of_f_of_f_eq_x_l1641_164152


namespace wrapping_paper_division_l1641_164142

theorem wrapping_paper_division (total_used : ℚ) (num_presents : ℕ) 
  (h1 : total_used = 1 / 2)
  (h2 : num_presents = 5) :
  total_used / num_presents = 1 / 10 := by
  sorry

end wrapping_paper_division_l1641_164142


namespace function_inequality_l1641_164190

theorem function_inequality (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x ≥ 1, f x = x * Real.log x) →
  (∀ x ≥ 1, f x ≥ a * x - 1) →
  a ≤ 1 := by
sorry

end function_inequality_l1641_164190


namespace rays_initial_cents_l1641_164163

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The amount of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- The number of nickels Peter receives -/
def nickels_to_peter : ℕ := cents_to_peter / nickel_value

/-- The amount of cents Ray gives to Randi -/
def cents_to_randi : ℕ := 2 * cents_to_peter

/-- The number of nickels Randi receives -/
def nickels_to_randi : ℕ := cents_to_randi / nickel_value

/-- The difference in nickels between Randi and Peter -/
def nickel_difference : ℕ := 6

theorem rays_initial_cents :
  nickels_to_randi = nickels_to_peter + nickel_difference →
  cents_to_peter + cents_to_randi = 90 := by
  sorry

end rays_initial_cents_l1641_164163


namespace area_triangle_STU_l1641_164174

/-- Square pyramid with given dimensions and points -/
structure SquarePyramid where
  -- Base side length
  base : ℝ
  -- Altitude
  height : ℝ
  -- Point S position ratio
  s_ratio : ℝ
  -- Point T position ratio
  t_ratio : ℝ
  -- Point U position ratio
  u_ratio : ℝ

/-- Theorem stating the area of triangle STU in the square pyramid -/
theorem area_triangle_STU (p : SquarePyramid) 
  (h_base : p.base = 4)
  (h_height : p.height = 8)
  (h_s : p.s_ratio = 1/4)
  (h_t : p.t_ratio = 1/2)
  (h_u : p.u_ratio = 3/4) :
  ∃ (area : ℝ), area = 7.5 ∧ 
  area = (1/2) * Real.sqrt ((p.s_ratio * p.height)^2 + (p.base/2)^2) * 
         (p.u_ratio * Real.sqrt (p.height^2 + (p.base/2)^2)) :=
by sorry

end area_triangle_STU_l1641_164174


namespace factor_polynomial_l1641_164192

theorem factor_polynomial (x : ℝ) : 75 * x^3 - 250 * x^7 = 25 * x^3 * (3 - 10 * x^4) := by
  sorry

end factor_polynomial_l1641_164192


namespace quadratic_trinomial_minimum_l1641_164151

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + b = 0) :
  ∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 ∧
  ∃ y : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 := by
  sorry

end quadratic_trinomial_minimum_l1641_164151


namespace employee_salaries_exist_l1641_164198

/-- Proves the existence of salaries for three employees satisfying given conditions --/
theorem employee_salaries_exist :
  ∃ (m n p : ℝ),
    (m + n + p = 1750) ∧
    (m = 1.3 * n) ∧
    (p = 0.9 * (m + n)) :=
by sorry

end employee_salaries_exist_l1641_164198


namespace equal_angles_l1641_164193

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the right focus of the ellipse
def right_focus (F : ℝ × ℝ) : Prop := F.1 > 0 ∧ F.1^2 / 2 + F.2^2 = 1

-- Define a line passing through a point
def line_through (l : ℝ → ℝ) (p : ℝ × ℝ) : Prop := l p.1 = p.2

-- Define the intersection points of the line and the ellipse
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line_through l A ∧ line_through l B ∧ A ≠ B

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_angles (F : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  right_focus F →
  line_through l F →
  intersection_points A B l →
  angle (0, 0) (2, 0) A = angle (0, 0) (2, 0) B :=
sorry

end equal_angles_l1641_164193


namespace water_in_tank_after_rain_l1641_164179

/-- Calculates the final amount of water in a tank after evaporation, draining, and rain. -/
def final_water_amount (initial_water evaporated_water drained_water rain_duration rain_rate : ℕ) : ℕ :=
  let remaining_after_evaporation := initial_water - evaporated_water
  let remaining_after_draining := remaining_after_evaporation - drained_water
  let rain_amount := (rain_duration / 10) * rain_rate
  remaining_after_draining + rain_amount

/-- Theorem stating that the final amount of water in the tank is 1550 liters. -/
theorem water_in_tank_after_rain :
  final_water_amount 6000 2000 3500 30 350 = 1550 := by
  sorry

end water_in_tank_after_rain_l1641_164179


namespace parabola_c_value_l1641_164182

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) 
    (point_condition : p.x_coord 0 = 1)
    (vertex_condition : p.x_coord 2 = 3 ∧ (∀ y, p.x_coord y ≤ p.x_coord 2)) :
  p.c = 1 := by
sorry


end parabola_c_value_l1641_164182


namespace alyssa_marbles_cost_l1641_164121

/-- The amount Alyssa spent on marbles -/
def marbles_cost (football_cost total_cost : ℚ) : ℚ :=
  total_cost - football_cost

/-- Proof that Alyssa spent $6.59 on marbles -/
theorem alyssa_marbles_cost :
  let football_cost : ℚ := 571/100
  let total_cost : ℚ := 1230/100
  marbles_cost football_cost total_cost = 659/100 := by
sorry

end alyssa_marbles_cost_l1641_164121


namespace five_letter_words_with_at_least_two_vowels_l1641_164136

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def count_words (s : Finset Char) (n : Nat) : Nat :=
  s.card ^ n

def count_words_with_exactly_k_vowels (k : Nat) : Nat :=
  Nat.choose word_length k * vowels.card ^ k * (alphabet.card - vowels.card) ^ (word_length - k)

theorem five_letter_words_with_at_least_two_vowels : 
  count_words alphabet word_length - 
  (count_words_with_exactly_k_vowels 0 + count_words_with_exactly_k_vowels 1) = 4192 := by
  sorry

end five_letter_words_with_at_least_two_vowels_l1641_164136


namespace tim_watch_time_l1641_164150

/-- The number of shows Tim watches -/
def num_shows : ℕ := 2

/-- The duration of the short show in hours -/
def short_show_duration : ℚ := 1/2

/-- The duration of the long show in hours -/
def long_show_duration : ℕ := 1

/-- The number of episodes of the short show -/
def short_show_episodes : ℕ := 24

/-- The number of episodes of the long show -/
def long_show_episodes : ℕ := 12

/-- The total number of hours Tim watched TV -/
def total_watch_time : ℚ := short_show_duration * short_show_episodes + long_show_duration * long_show_episodes

theorem tim_watch_time :
  total_watch_time = 24 := by sorry

end tim_watch_time_l1641_164150


namespace linear_systems_solution_l1641_164177

/-- Given two systems of linear equations with the same solution, 
    prove the solution, the values of a and b, and a related expression. -/
theorem linear_systems_solution :
  ∃ (x y a b : ℝ),
    -- First system of equations
    (2 * x + 5 * y = -26 ∧ a * x - b * y = -4) ∧
    -- Second system of equations
    (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
    -- The solution
    (x = 2 ∧ y = -6) ∧
    -- The values of a and b
    (a = 1 ∧ b = -1) ∧
    -- The value of the expression
    ((2 * a + b) ^ 2020 = 1) := by
  sorry

end linear_systems_solution_l1641_164177


namespace pumpkins_eaten_by_rabbits_l1641_164111

/-- Represents the number of pumpkins Sara initially grew -/
def initial_pumpkins : ℕ := 43

/-- Represents the number of pumpkins Sara has left after rabbits ate some -/
def remaining_pumpkins : ℕ := 20

/-- Represents the number of pumpkins eaten by rabbits -/
def eaten_pumpkins : ℕ := initial_pumpkins - remaining_pumpkins

/-- Theorem stating that the number of pumpkins eaten by rabbits is the difference between
    the initial number of pumpkins and the remaining number of pumpkins -/
theorem pumpkins_eaten_by_rabbits :
  eaten_pumpkins = initial_pumpkins - remaining_pumpkins :=
by
  sorry

end pumpkins_eaten_by_rabbits_l1641_164111


namespace triangle_theorem_l1641_164125

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (abc : Triangle) (h1 : abc.b / abc.a = Real.sin abc.B / Real.sin (2 * abc.A))
  (h2 : abc.b = 2 * Real.sqrt 3) (h3 : 1/2 * abc.b * abc.c * Real.sin abc.A = 3 * Real.sqrt 3 / 2) :
  abc.A = π/3 ∧ abc.a = 3 := by
  sorry

end triangle_theorem_l1641_164125


namespace first_two_digits_sum_l1641_164168

/-- The number of integer lattice points (x, y) satisfying 4x^2 + 9y^2 ≤ 1000000000 -/
def N : ℕ := sorry

/-- The first digit of N -/
def a : ℕ := sorry

/-- The second digit of N -/
def b : ℕ := sorry

/-- Theorem stating that 10a + b equals 52 -/
theorem first_two_digits_sum : 10 * a + b = 52 := by sorry

end first_two_digits_sum_l1641_164168


namespace odd_square_sum_of_consecutive_b_l1641_164143

-- Define sequence a_n
def a : ℕ → ℕ
| n => if n % 2 = 1 then 4 * ((n + 1) / 2) - 2 else 4 * (n / 2) - 1

-- Define sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then 8 * ((n - 1) / 2) + 3 else 8 * (n / 2) - 2

-- Define the theorem
theorem odd_square_sum_of_consecutive_b (k : ℕ) (h : k ≥ 1) :
  ∃ n : ℕ, (2 * k + 1)^2 = b n + b (n + 1) := by
  sorry

end odd_square_sum_of_consecutive_b_l1641_164143


namespace rationalize_denominator_l1641_164103

theorem rationalize_denominator : 
  (36 : ℝ) / (12 : ℝ)^(1/3) = 3 * (144 : ℝ)^(1/3) := by sorry

end rationalize_denominator_l1641_164103


namespace inequality_proof_l1641_164178

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  4 * a^3 * (a - b) ≥ a^4 - b^4 := by
  sorry

end inequality_proof_l1641_164178


namespace angle_between_sides_l1641_164119

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] :=
  (a b c d : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)

-- Define the theorem
theorem angle_between_sides (q : CyclicQuadrilateral ℝ) :
  Real.arccos ((q.a^2 + q.b^2 - q.d^2 - q.c^2) / (2 * (q.a * q.b + q.d * q.c))) =
  Real.arccos ((q.a^2 + q.b^2 - q.d^2 - q.c^2) / (2 * (q.a * q.b + q.d * q.c))) :=
by sorry

end angle_between_sides_l1641_164119


namespace angle_expression_value_l1641_164106

/-- Given that point P(1,2) is on the terminal side of angle α, 
    prove that (6sinα + 8cosα) / (3sinα - 2cosα) = 5 -/
theorem angle_expression_value (α : Real) (h : Complex.exp (α * Complex.I) = ⟨1, 2⟩) :
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end angle_expression_value_l1641_164106


namespace circular_segment_probability_l1641_164135

/-- The ratio of circumference to diameter in ancient Chinese mathematics -/
def ancient_pi : ℚ := 3

/-- The area of a circular segment given chord length and height difference -/
def segment_area (a c : ℚ) : ℚ := (1/2) * a * (a + c)

/-- The probability of a point mass landing in a circular segment -/
theorem circular_segment_probability (c a : ℚ) (h1 : c = 6) (h2 : a = 1) :
  let r := (c^2 / 4 + a^2) / (2 * a)
  let circle_area := ancient_pi * r^2
  segment_area a c / circle_area = 7 / 150 := by
  sorry

end circular_segment_probability_l1641_164135


namespace line_passes_through_point_l1641_164154

theorem line_passes_through_point (a b : ℝ) (h : 3 * a + 2 * b = 5) :
  a * 6 + b * 4 - 10 = 0 := by
sorry

end line_passes_through_point_l1641_164154


namespace rebecca_eggs_l1641_164189

/-- The number of groups Rebecca wants to split her eggs into -/
def num_groups : ℕ := 4

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 2

/-- Theorem: Rebecca has 8 eggs in total -/
theorem rebecca_eggs : num_groups * eggs_per_group = 8 := by
  sorry

end rebecca_eggs_l1641_164189


namespace relationship_a_ab_ab_squared_l1641_164115

theorem relationship_a_ab_ab_squared (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end relationship_a_ab_ab_squared_l1641_164115


namespace partnership_profit_l1641_164159

/-- Given the investments of three partners and one partner's share of the profit,
    calculate the total profit of the partnership. -/
theorem partnership_profit
  (investment_A investment_B investment_C : ℕ)
  (profit_share_A : ℕ)
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 3660) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 12200 :=
by sorry

end partnership_profit_l1641_164159


namespace abs_inequality_implies_quadratic_inequality_l1641_164197

theorem abs_inequality_implies_quadratic_inequality :
  {x : ℝ | |x - 1| < 2} ⊂ {x : ℝ | (x + 2) * (x - 3) < 0} ∧
  {x : ℝ | |x - 1| < 2} ≠ {x : ℝ | (x + 2) * (x - 3) < 0} :=
sorry

end abs_inequality_implies_quadratic_inequality_l1641_164197


namespace product_of_squares_and_fourth_powers_l1641_164160

theorem product_of_squares_and_fourth_powers (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_sum_squares : a^2 + b^2 = 5)
  (h_sum_fourth_powers : a^4 + b^4 = 17) : 
  a * b = 2 := by sorry

end product_of_squares_and_fourth_powers_l1641_164160


namespace college_students_count_l1641_164116

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 200) :
  boys + girls = 520 := by
  sorry

end college_students_count_l1641_164116


namespace compound_molecular_weight_l1641_164114

def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def K_count : ℕ := 1
def Br_count : ℕ := 1
def O_count : ℕ := 3

def molecular_weight : ℝ :=
  K_count * atomic_weight_K +
  Br_count * atomic_weight_Br +
  O_count * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 167.00 := by sorry

end compound_molecular_weight_l1641_164114


namespace money_distribution_l1641_164147

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 900)
  (AC_sum : A + C = 400)
  (C_amount : C = 250) :
  B + C = 750 := by sorry

end money_distribution_l1641_164147


namespace prob_at_most_one_first_class_l1641_164199

/-- The probability of selecting at most one first-class product when randomly choosing 2 out of 5 products (3 first-class and 2 second-class) is 0.7 -/
theorem prob_at_most_one_first_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ) :
  total = 5 →
  first_class = 3 →
  second_class = 2 →
  selected = 2 →
  (Nat.choose first_class 1 * Nat.choose second_class 1 + Nat.choose second_class 2) / Nat.choose total selected = 7 / 10 :=
by sorry

end prob_at_most_one_first_class_l1641_164199


namespace slope_angle_sqrt3_l1641_164131

/-- The slope angle of a line with slope √3 is 60 degrees. -/
theorem slope_angle_sqrt3 : ∃ θ : Real, 
  0 ≤ θ ∧ θ < Real.pi ∧ 
  Real.tan θ = Real.sqrt 3 ∧ 
  θ = Real.pi / 3 := by sorry

end slope_angle_sqrt3_l1641_164131


namespace train_passing_time_l1641_164130

/-- The time taken for a faster train to catch and pass a slower train -/
theorem train_passing_time (train_length : ℝ) (speed_fast speed_slow : ℝ) : 
  train_length = 25 →
  speed_fast = 46 * (1000 / 3600) →
  speed_slow = 36 * (1000 / 3600) →
  speed_fast > speed_slow →
  (2 * train_length) / (speed_fast - speed_slow) = 18 := by
  sorry

#eval (2 * 25) / ((46 - 36) * (1000 / 3600))

end train_passing_time_l1641_164130


namespace rent_increase_problem_l1641_164162

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 850) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.25) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 800 := by
  sorry

end rent_increase_problem_l1641_164162


namespace mary_max_earnings_l1641_164128

/-- Calculates the maximum weekly earnings for Mary given her work conditions -/
theorem mary_max_earnings :
  let max_hours : ℕ := 60
  let regular_hours : ℕ := 20
  let regular_rate : ℚ := 8
  let overtime_rate_increase : ℚ := 0.25
  let overtime_rate : ℚ := regular_rate * (1 + overtime_rate_increase)
  let overtime_hours : ℕ := max_hours - regular_hours
  let regular_earnings : ℚ := regular_hours * regular_rate
  let overtime_earnings : ℚ := overtime_hours * overtime_rate
  regular_earnings + overtime_earnings = 560 := by
sorry

end mary_max_earnings_l1641_164128


namespace directrix_of_symmetrical_parabola_l1641_164108

-- Define the original parabola
def original_parabola (x y : ℝ) : Prop := y = 2 * x^2

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = x

-- Define the symmetrical parabola
def symmetrical_parabola (x y : ℝ) : Prop := y^2 = (1/2) * x

-- Theorem statement
theorem directrix_of_symmetrical_parabola :
  ∀ (x : ℝ), (∃ (y : ℝ), symmetrical_parabola x y) → (x = -1/8) = 
  (∀ (p : ℝ), p ≠ 0 → (∃ (h k : ℝ), ∀ (x y : ℝ), 
    symmetrical_parabola x y ↔ (y - k)^2 = 4 * p * (x - h) ∧ x = h - p)) :=
by sorry

end directrix_of_symmetrical_parabola_l1641_164108


namespace incorrect_multiplication_l1641_164165

theorem incorrect_multiplication : 79133 * 111107 ≠ 8794230231 := by
  sorry

end incorrect_multiplication_l1641_164165


namespace problem_solution_l1641_164124

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1)
  (h2 : x + 1 / z = 8)
  (h3 : y + 1 / x = 29) :
  z + 1 / y = 13 / 77 := by
sorry

end problem_solution_l1641_164124


namespace least_value_expression_l1641_164141

theorem least_value_expression (x : ℝ) (h : x < -2) :
  (2 * x ≤ x) ∧ (2 * x ≤ x + 2) ∧ (2 * x ≤ (1/2) * x) ∧ (2 * x ≤ x - 2) := by
  sorry

end least_value_expression_l1641_164141


namespace max_table_height_value_l1641_164107

/-- Triangle DEF with sides a, b, c -/
structure Triangle (a b c : ℝ) : Type :=
  (side_a : a > 0)
  (side_b : b > 0)
  (side_c : c > 0)

/-- The maximum table height function -/
def maxTableHeight (t : Triangle 25 29 32) : ℝ := sorry

/-- Theorem stating the maximum table height -/
theorem max_table_height_value (t : Triangle 25 29 32) : 
  maxTableHeight t = 64 * Real.sqrt 29106 / 1425 := by sorry

end max_table_height_value_l1641_164107


namespace monochromatic_four_cycle_exists_l1641_164169

/-- A coloring of edges in a graph using two colors -/
def TwoColoring (V : Type*) := V → V → Bool

/-- A complete graph with 6 vertices -/
def CompleteGraph6 := Fin 6

/-- A 4-cycle in a graph -/
def FourCycle (V : Type*) := 
  (V × V × V × V)

/-- Predicate to check if a 4-cycle is monochromatic under a given coloring -/
def IsMonochromatic (c : TwoColoring CompleteGraph6) (cycle : FourCycle CompleteGraph6) : Prop :=
  let (a, b, d, e) := cycle
  c a b = c b d ∧ c b d = c d e ∧ c d e = c e a

/-- Main theorem: In a complete graph with 6 vertices where each edge is colored 
    with one of two colors, there exists a monochromatic 4-cycle -/
theorem monochromatic_four_cycle_exists :
  ∀ (c : TwoColoring CompleteGraph6),
  ∃ (cycle : FourCycle CompleteGraph6), IsMonochromatic c cycle :=
sorry


end monochromatic_four_cycle_exists_l1641_164169


namespace kates_discount_is_eight_percent_l1641_164157

-- Define the bills and total paid
def bobs_bill : ℚ := 30
def kates_bill : ℚ := 25
def total_paid : ℚ := 53

-- Define the discount percentage
def discount_percentage : ℚ := (bobs_bill + kates_bill - total_paid) / kates_bill * 100

-- Theorem statement
theorem kates_discount_is_eight_percent :
  discount_percentage = 8 := by sorry

end kates_discount_is_eight_percent_l1641_164157


namespace tangent_point_bisects_second_side_l1641_164173

/-- A pentagon inscribed around a circle -/
structure InscribedPentagon where
  /-- The lengths of the sides of the pentagon -/
  sides : Fin 5 → ℕ
  /-- The first and third sides have length 1 -/
  first_third_sides_one : sides 0 = 1 ∧ sides 2 = 1
  /-- The point where the circle touches the second side of the pentagon -/
  tangent_point : ℝ
  /-- The tangent point is between 0 and the length of the second side -/
  tangent_point_valid : 0 < tangent_point ∧ tangent_point < sides 1

/-- The theorem stating that the tangent point divides the second side into two equal segments -/
theorem tangent_point_bisects_second_side (p : InscribedPentagon) :
  p.tangent_point = (p.sides 1 : ℝ) / 2 := by
  sorry

end tangent_point_bisects_second_side_l1641_164173


namespace shopping_theorem_l1641_164196

def shopping_problem (initial_amount discount_rate tax_rate: ℝ)
  (sweater t_shirt shoes jeans scarf: ℝ) : Prop :=
  let discounted_t_shirt := t_shirt * (1 - discount_rate)
  let subtotal := sweater + discounted_t_shirt + shoes + jeans + scarf
  let total_with_tax := subtotal * (1 + tax_rate)
  let remaining := initial_amount - total_with_tax
  remaining = 30.11

theorem shopping_theorem :
  shopping_problem 200 0.1 0.05 36 12 45 52 18 :=
by sorry

end shopping_theorem_l1641_164196


namespace sale_price_calculation_l1641_164167

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  costPrice * (1 + profitRate) * (1 + taxRate)

/-- The sale price including tax is 677.60 given the specified conditions -/
theorem sale_price_calculation :
  let costPrice : ℝ := 535.65
  let profitRate : ℝ := 0.15
  let taxRate : ℝ := 0.10
  ∃ ε > 0, |salePriceWithTax costPrice profitRate taxRate - 677.60| < ε :=
by
  sorry

#eval salePriceWithTax 535.65 0.15 0.10

end sale_price_calculation_l1641_164167


namespace onion_weight_proof_l1641_164171

/-- Proves that the total weight of onions on a scale is 7.68 kg given specific conditions --/
theorem onion_weight_proof (total_weight : ℝ) (remaining_onions : ℕ) (removed_onions : ℕ) 
  (avg_weight_remaining : ℝ) (avg_weight_removed : ℝ) : 
  total_weight = 7.68 ∧ 
  remaining_onions = 35 ∧ 
  removed_onions = 5 ∧ 
  avg_weight_remaining = 0.190 ∧ 
  avg_weight_removed = 0.206 → 
  total_weight = (remaining_onions : ℝ) * avg_weight_remaining + 
                 (removed_onions : ℝ) * avg_weight_removed :=
by
  sorry

#check onion_weight_proof

end onion_weight_proof_l1641_164171


namespace ceiling_sum_sqrt_l1641_164180

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_sqrt_l1641_164180


namespace inequality_always_holds_l1641_164120

theorem inequality_always_holds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) :=
by sorry

end inequality_always_holds_l1641_164120


namespace train_length_l1641_164187

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) →
  platform_length = 240 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 280 := by
  sorry

end train_length_l1641_164187


namespace min_unique_integers_l1641_164118

theorem min_unique_integers (L : List ℕ) (h : L = [1, 2, 3, 4, 5, 6, 7, 8, 9]) : 
  ∃ (f : ℕ → ℕ), 
    (∀ n, f n = n + 2 ∨ f n = n + 5) ∧ 
    (Finset.card (Finset.image f L.toFinset) = 6) ∧
    (∀ g : ℕ → ℕ, (∀ n, g n = n + 2 ∨ g n = n + 5) → 
      Finset.card (Finset.image g L.toFinset) ≥ 6) := by
  sorry

end min_unique_integers_l1641_164118


namespace stacy_berries_l1641_164164

/-- The number of berries each person has -/
structure BerryDistribution where
  sophie : ℕ
  sylar : ℕ
  steve : ℕ
  stacy : ℕ

/-- The conditions of the berry distribution problem -/
def valid_distribution (b : BerryDistribution) : Prop :=
  b.sylar = 5 * b.sophie ∧
  b.steve = 2 * b.sylar ∧
  b.stacy = 4 * b.steve ∧
  b.sophie + b.sylar + b.steve + b.stacy = 2200

/-- Theorem stating that Stacy has 1560 berries -/
theorem stacy_berries (b : BerryDistribution) (h : valid_distribution b) : b.stacy = 1560 := by
  sorry

end stacy_berries_l1641_164164


namespace simplify_expression_l1641_164139

theorem simplify_expression (m n : ℝ) : m - n - (m + n) = -2 * n := by
  sorry

end simplify_expression_l1641_164139


namespace cone_volume_ratio_cone_C_D_volume_ratio_l1641_164184

/-- The ratio of the volumes of two cones with swapped radius and height is 1/2 -/
theorem cone_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * Real.pi * r^2 * h) / (1 / 3 * Real.pi * h^2 * r) = 1 / 2 := by
  sorry

/-- The ratio of the volumes of cones C and D is 1/2 -/
theorem cone_C_D_volume_ratio : 
  let r : ℝ := 16.4
  let h : ℝ := 32.8
  (1 / 3 * Real.pi * r^2 * h) / (1 / 3 * Real.pi * h^2 * r) = 1 / 2 := by
  sorry

end cone_volume_ratio_cone_C_D_volume_ratio_l1641_164184


namespace portfolio_distribution_l1641_164191

theorem portfolio_distribution (total_students : ℕ) (total_portfolios : ℕ) 
  (h1 : total_students = 120) 
  (h2 : total_portfolios = 8365) : 
  ∃ (regular_portfolios : ℕ) (special_portfolios : ℕ) (remaining_portfolios : ℕ),
    let regular_students : ℕ := (85 * total_students) / 100
    let special_students : ℕ := total_students - regular_students
    special_portfolios = regular_portfolios + 10 ∧
    regular_students * regular_portfolios + special_students * special_portfolios + remaining_portfolios = total_portfolios ∧
    remaining_portfolios = 25 :=
by sorry

end portfolio_distribution_l1641_164191


namespace arithmetic_sequence_term_count_l1641_164110

theorem arithmetic_sequence_term_count 
  (a : ℕ) (d : ℕ) (l : ℕ) (n : ℕ) 
  (h1 : a = 7) 
  (h2 : d = 2) 
  (h3 : l = 145) 
  (h4 : l = a + (n - 1) * d) : 
  n = 70 := by
sorry

end arithmetic_sequence_term_count_l1641_164110


namespace predicted_weight_for_178cm_l1641_164126

/-- Regression equation for weight prediction based on height -/
def weight_prediction (height : ℝ) : ℝ := 0.72 * height - 58.2

/-- Theorem: The predicted weight for a person with height 178 cm is 69.96 kg -/
theorem predicted_weight_for_178cm :
  weight_prediction 178 = 69.96 := by sorry

end predicted_weight_for_178cm_l1641_164126


namespace seniors_in_stratified_sample_l1641_164133

/-- Represents the number of seniors in a stratified sample -/
def seniors_in_sample (total_students : ℕ) (total_seniors : ℕ) (sample_size : ℕ) : ℕ :=
  (total_seniors * sample_size) / total_students

/-- Theorem stating that in a school with 4500 students, of which 1500 are seniors,
    a stratified sample of 300 students will contain 100 seniors -/
theorem seniors_in_stratified_sample :
  seniors_in_sample 4500 1500 300 = 100 := by
  sorry

end seniors_in_stratified_sample_l1641_164133


namespace sphere_volume_from_cylinder_volume_l1641_164137

/-- Given a cylinder with volume 72π cm³, prove that a sphere with the same radius has volume 96π cm³ -/
theorem sphere_volume_from_cylinder_volume (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 72 * π → (4/3) * π * r^3 = 96 * π := by
  sorry

end sphere_volume_from_cylinder_volume_l1641_164137


namespace total_hamburgers_for_lunch_l1641_164123

theorem total_hamburgers_for_lunch : 
  let initial_beef : ℕ := 15
  let initial_veggie : ℕ := 12
  let additional_beef : ℕ := 5
  let additional_veggie : ℕ := 7
  initial_beef + initial_veggie + additional_beef + additional_veggie = 39
  := by sorry

end total_hamburgers_for_lunch_l1641_164123


namespace odd_function_fixed_point_l1641_164155

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The theorem states that if f is an odd function on ℝ,
    then (-1, -2) is a point on the graph of y = f(x+1) - 2 -/
theorem odd_function_fixed_point (f : ℝ → ℝ) (h : IsOdd f) :
  f 0 - 2 = -2 := by sorry

end odd_function_fixed_point_l1641_164155


namespace equation_solution_l1641_164149

theorem equation_solution (x : ℚ) : 64 * (x + 1)^3 - 27 = 0 → x = -1/4 := by
  sorry

end equation_solution_l1641_164149


namespace range_of_a_l1641_164102

theorem range_of_a (a x y z : ℝ) 
  (h1 : |a - 2| ≤ x^2 + 2*y^2 + 3*z^2)
  (h2 : x + y + z = 1) :
  16/11 ≤ a ∧ a ≤ 28/11 := by
sorry

end range_of_a_l1641_164102


namespace toms_initial_investment_l1641_164188

theorem toms_initial_investment (t j k : ℝ) : 
  t + j + k = 1200 →
  t - 150 + 3*j + 3*k = 1800 →
  t = 825 := by
sorry

end toms_initial_investment_l1641_164188


namespace article_cost_changes_l1641_164156

theorem article_cost_changes (original_cost : ℝ) : 
  original_cost = 75 → 
  (original_cost * 1.2) * 0.8 = 72 := by
sorry

end article_cost_changes_l1641_164156


namespace geometric_sequence_sum_l1641_164166

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 9) →
  (a 4 + a 5 = 27 ∨ a 4 + a 5 = -27) :=
by sorry

end geometric_sequence_sum_l1641_164166


namespace problem_solution_l1641_164146

-- Define proposition p
def p : Prop := ∃ x₀ : ℝ, Real.log x₀ ≥ x₀ - 1

-- Define proposition q
def q : Prop := ∀ θ : ℝ, Real.sin θ + Real.cos θ < 1

-- Theorem to prove
theorem problem_solution :
  p ∧ ¬q := by sorry

end problem_solution_l1641_164146


namespace steves_speed_ratio_l1641_164129

/-- Proves the ratio of Steve's speeds given the problem conditions -/
theorem steves_speed_ratio :
  let distance : ℝ := 10 -- km
  let total_time : ℝ := 6 -- hours
  let speed_back : ℝ := 5 -- km/h
  let speed_to_work : ℝ := distance / (total_time - distance / speed_back)
  speed_back / speed_to_work = 2
  := by sorry

end steves_speed_ratio_l1641_164129


namespace remainder_polynomial_l1641_164186

-- Define the polynomials
variable (z : ℂ)
variable (Q : ℂ → ℂ)
variable (R : ℂ → ℂ)

-- State the theorem
theorem remainder_polynomial :
  (∀ z, z^2023 + 1 = (z^2 - z + 1) * Q z + R z) →
  (∃ a b : ℂ, ∀ z, R z = a * z + b) →
  (∀ z, R z = z + 1) :=
by sorry

end remainder_polynomial_l1641_164186


namespace sine_equality_solution_l1641_164117

theorem sine_equality_solution (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * π / 180) = Real.sin (750 * π / 180) → n = 30 := by
sorry

end sine_equality_solution_l1641_164117


namespace grouping_theorem_l1641_164104

/-- The number of ways to distribute 4 men and 5 women into groups -/
def grouping_ways : ℕ := 
  let men : ℕ := 4
  let women : ℕ := 5
  let small_group_size : ℕ := 2
  let large_group_size : ℕ := 5
  let num_small_groups : ℕ := 2
  100

/-- Theorem stating that the number of ways to distribute 4 men and 5 women
    into two groups of two people and one group of five people, 
    with at least one man and one woman in each group, is 100 -/
theorem grouping_theorem : grouping_ways = 100 := by
  sorry

end grouping_theorem_l1641_164104


namespace right_triangle_third_side_l1641_164100

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) → c = 5 ∨ c = Real.sqrt 7 := by
  sorry

end right_triangle_third_side_l1641_164100


namespace x_power_2187_minus_reciprocal_l1641_164172

theorem x_power_2187_minus_reciprocal (x : ℂ) (h : x - 1/x = 2*I*Real.sqrt 2) : 
  x^2187 - 1/(x^2187) = -22*I*Real.sqrt 2 := by
sorry

end x_power_2187_minus_reciprocal_l1641_164172


namespace range_of_m_l1641_164138

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define the condition for f having two zeros
def has_two_zeros (m : ℝ) : Prop := ∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0

-- Define the condition q
def condition_q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, has_two_zeros m ∧ ¬(condition_q m) →
  m < -2 ∨ m ≥ 3 :=
sorry

end range_of_m_l1641_164138


namespace multiply_add_distribute_l1641_164101

theorem multiply_add_distribute : 57 * 33 + 13 * 33 = 2310 := by
  sorry

end multiply_add_distribute_l1641_164101


namespace oranges_in_sack_l1641_164175

/-- The number of sacks harvested per day -/
def sacks_per_day : ℕ := 66

/-- The number of days of harvest -/
def harvest_days : ℕ := 87

/-- The total number of oranges after the harvest -/
def total_oranges : ℕ := 143550

/-- The number of oranges in each sack -/
def oranges_per_sack : ℕ := total_oranges / (sacks_per_day * harvest_days)

theorem oranges_in_sack : oranges_per_sack = 25 := by
  sorry

end oranges_in_sack_l1641_164175


namespace iced_tea_consumption_iced_tea_consumption_is_198_l1641_164105

theorem iced_tea_consumption : ℝ → Prop :=
  fun total_consumption =>
    ∃ (rob_size : ℝ),
      let mary_size : ℝ := 1.75 * rob_size
      let rob_remaining : ℝ := (1/3) * rob_size
      let mary_remaining : ℝ := (1/3) * mary_size
      let mary_share : ℝ := (1/4) * mary_remaining + 3
      let rob_total : ℝ := (2/3) * rob_size + mary_share
      let mary_total : ℝ := (2/3) * mary_size - mary_share
      rob_total = mary_total ∧
      total_consumption = rob_size + mary_size ∧
      total_consumption = 198

theorem iced_tea_consumption_is_198 : iced_tea_consumption 198 := by
  sorry

end iced_tea_consumption_iced_tea_consumption_is_198_l1641_164105


namespace class_average_l1641_164148

theorem class_average (students1 : ℕ) (average1 : ℚ) (students2 : ℕ) (average2 : ℚ) :
  students1 = 15 →
  average1 = 73/100 →
  students2 = 10 →
  average2 = 88/100 →
  (students1 * average1 + students2 * average2) / (students1 + students2) = 79/100 := by
  sorry

end class_average_l1641_164148


namespace unique_increasing_function_l1641_164153

theorem unique_increasing_function :
  ∃! f : ℕ → ℕ,
    (∀ n m : ℕ, (2^m + 1) * f n * f (2^m * n) = 2^m * (f n)^2 + (f (2^m * n))^2 + (2^m - 1)^2 * n) ∧
    (∀ a b : ℕ, a < b → f a < f b) ∧
    (∀ n : ℕ, f n = n + 1) :=
by sorry

end unique_increasing_function_l1641_164153


namespace bridge_length_calculation_l1641_164145

/-- Given a train crossing a bridge, calculate the length of the bridge -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 145 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 230 :=
by
  sorry

end bridge_length_calculation_l1641_164145


namespace y1_greater_than_y2_l1641_164183

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The symmetric axis of the quadratic function is x = 1 -/
def symmetric_axis (f : QuadraticFunction) : ℝ := 1

/-- The quadratic function passes through the point (-1, y₁) -/
def passes_through_minus_one (f : QuadraticFunction) (y₁ : ℝ) : Prop :=
  f.a * (-1)^2 + f.b * (-1) + f.c = y₁

/-- The quadratic function passes through the point (2, y₂) -/
def passes_through_two (f : QuadraticFunction) (y₂ : ℝ) : Prop :=
  f.a * 2^2 + f.b * 2 + f.c = y₂

/-- Theorem stating that y₁ > y₂ for the given conditions -/
theorem y1_greater_than_y2 (f : QuadraticFunction) (y₁ y₂ : ℝ)
  (h1 : passes_through_minus_one f y₁)
  (h2 : passes_through_two f y₂) :
  y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l1641_164183


namespace product_equals_143_l1641_164161

/-- Convert a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Convert a ternary number (represented as a list of digits) to its decimal equivalent -/
def ternary_to_decimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The binary representation of 1101₂ -/
def binary_num : List Bool := [true, false, true, true]

/-- The ternary representation of 102₃ -/
def ternary_num : List ℕ := [2, 0, 1]

theorem product_equals_143 : 
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 143 := by
  sorry

end product_equals_143_l1641_164161


namespace prism_lateral_edge_length_l1641_164185

/-- A prism with 12 vertices and a sum of lateral edge lengths of 60 has lateral edges of length 10. -/
theorem prism_lateral_edge_length (num_vertices : ℕ) (sum_lateral_edges : ℝ) :
  num_vertices = 12 →
  sum_lateral_edges = 60 →
  ∃ (lateral_edge_length : ℝ), lateral_edge_length = 10 ∧
    lateral_edge_length * (num_vertices / 2) = sum_lateral_edges :=
by sorry


end prism_lateral_edge_length_l1641_164185


namespace donna_bananas_l1641_164132

def total_bananas : ℕ := 350
def lydia_bananas : ℕ := 90
def dawn_extra_bananas : ℕ := 70

theorem donna_bananas :
  total_bananas - (lydia_bananas + (lydia_bananas + dawn_extra_bananas)) = 100 :=
by sorry

end donna_bananas_l1641_164132


namespace sum_of_rational_roots_l1641_164176

def h (x : ℚ) : ℚ := x^3 - 12*x^2 + 47*x - 60

theorem sum_of_rational_roots :
  ∃ (r₁ r₂ r₃ : ℚ),
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0 ∧
    (∀ r : ℚ, h r = 0 → r = r₁ ∨ r = r₂ ∨ r = r₃) ∧
    r₁ + r₂ + r₃ = 12 :=
sorry

end sum_of_rational_roots_l1641_164176


namespace not_prime_expression_l1641_164122

theorem not_prime_expression (n k : ℤ) (h1 : n > 2) (h2 : k ≠ n) :
  ¬ Prime (n^2 - k*n + k - 1) :=
by sorry

end not_prime_expression_l1641_164122


namespace house_numbering_proof_l1641_164113

theorem house_numbering_proof :
  (2 * 169^2 - 1 = 239^2) ∧ (2 * (288^2 + 288) = 408^2) := by
  sorry

end house_numbering_proof_l1641_164113
