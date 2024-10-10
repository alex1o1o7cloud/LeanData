import Mathlib

namespace y_value_proof_l3892_389263

theorem y_value_proof (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 := by
  sorry

end y_value_proof_l3892_389263


namespace fewer_football_boxes_l3892_389264

theorem fewer_football_boxes (total_cards : ℕ) (basketball_boxes : ℕ) (cards_per_basketball_box : ℕ) (cards_per_football_box : ℕ) 
  (h1 : total_cards = 255)
  (h2 : basketball_boxes = 9)
  (h3 : cards_per_basketball_box = 15)
  (h4 : cards_per_football_box = 20)
  (h5 : basketball_boxes * cards_per_basketball_box + (total_cards - basketball_boxes * cards_per_basketball_box) = total_cards)
  (h6 : (total_cards - basketball_boxes * cards_per_basketball_box) % cards_per_football_box = 0) :
  basketball_boxes - (total_cards - basketball_boxes * cards_per_basketball_box) / cards_per_football_box = 3 := by
  sorry

end fewer_football_boxes_l3892_389264


namespace example_monomial_properties_l3892_389299

/-- Represents a monomial with integer coefficient and variables x, y, and z -/
structure Monomial where
  coeff : Int
  x_exp : Nat
  y_exp : Nat
  z_exp : Nat

/-- Calculates the coefficient of a monomial -/
def coefficient (m : Monomial) : Int :=
  m.coeff

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : Nat :=
  m.x_exp + m.y_exp + m.z_exp

/-- The monomial -3^2 * x * y * z^2 -/
def example_monomial : Monomial :=
  { coeff := -9, x_exp := 1, y_exp := 1, z_exp := 2 }

theorem example_monomial_properties :
  (coefficient example_monomial = -9) ∧ (degree example_monomial = 4) := by
  sorry


end example_monomial_properties_l3892_389299


namespace f_passes_through_quadrants_234_l3892_389254

/-- A linear function f(x) = kx + b passes through the second, third, and fourth quadrants if and only if k < 0 and b < 0 -/
def passes_through_quadrants_234 (k b : ℝ) : Prop :=
  k < 0 ∧ b < 0

/-- The specific linear function f(x) = -2x - 1 -/
def f (x : ℝ) : ℝ := -2 * x - 1

/-- Theorem stating that f(x) = -2x - 1 passes through the second, third, and fourth quadrants -/
theorem f_passes_through_quadrants_234 :
  passes_through_quadrants_234 (-2) (-1) :=
sorry

end f_passes_through_quadrants_234_l3892_389254


namespace crazy_silly_school_series_l3892_389251

theorem crazy_silly_school_series (num_books : ℕ) (movies_watched : ℕ) (books_read : ℕ) :
  num_books = 8 →
  movies_watched = 19 →
  books_read = 16 →
  movies_watched = books_read + 3 →
  ∃ (num_movies : ℕ), num_movies ≥ 19 :=
by
  sorry

end crazy_silly_school_series_l3892_389251


namespace micheal_work_days_l3892_389287

/-- Represents the total amount of work to be done -/
def W : ℝ := 1

/-- Represents the rate at which Micheal works (fraction of work done per day) -/
def M : ℝ := sorry

/-- Represents the rate at which Adam works (fraction of work done per day) -/
def A : ℝ := sorry

/-- Micheal and Adam can do the work together in 20 days -/
axiom combined_rate : M + A = W / 20

/-- After working together for 14 days, the remaining work is completed by Adam in 10 days -/
axiom remaining_work : A * 10 = W - 14 * (M + A)

theorem micheal_work_days : M = W / 50 := by sorry

end micheal_work_days_l3892_389287


namespace complex_magnitude_l3892_389234

theorem complex_magnitude (z : ℂ) (h : 1 + z * Complex.I = 2 * Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l3892_389234


namespace smallest_n_for_candy_purchase_l3892_389229

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (r g b : ℕ), r > 0 → g > 0 → b > 0 → 10 * r = 18 * g ∧ 18 * g = 20 * b ∧ 20 * b = 24 * n) ∧
  (∀ (m : ℕ), m > 0 → 
    (∀ (r g b : ℕ), r > 0 → g > 0 → b > 0 → 10 * r = 18 * g ∧ 18 * g = 20 * b ∧ 20 * b = 24 * m) → 
    n ≤ m) ∧
  n = 15 := by
  sorry

end smallest_n_for_candy_purchase_l3892_389229


namespace feeding_sequences_count_l3892_389294

/-- Represents the number of distinct pairs of animals -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to feed animals given the conditions -/
def feeding_sequences : ℕ :=
  1 * num_pairs.factorial * (num_pairs - 1) * (num_pairs - 2) * (num_pairs - 3) * (num_pairs - 4) * (num_pairs - 5)

/-- Theorem stating that the number of feeding sequences is 17280 -/
theorem feeding_sequences_count : feeding_sequences = 17280 := by
  sorry

end feeding_sequences_count_l3892_389294


namespace total_points_is_63_l3892_389216

/-- The total points scored by Zach and Ben in a football game -/
def total_points (zach_points ben_points : Float) : Float :=
  zach_points + ben_points

/-- Theorem stating that the total points scored by Zach and Ben is 63.0 -/
theorem total_points_is_63 :
  total_points 42.0 21.0 = 63.0 := by
  sorry

end total_points_is_63_l3892_389216


namespace reflect_P_x_axis_l3892_389204

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The original point P -/
def P : Point :=
  { x := -2, y := 4 }

/-- Theorem: Reflecting point P(-2, 4) across the x-axis results in (-2, -4) -/
theorem reflect_P_x_axis :
  reflect_x P = { x := -2, y := -4 } := by
  sorry

end reflect_P_x_axis_l3892_389204


namespace quadratic_inequality_theorem_l3892_389248

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 6

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Define the theorem
theorem quadratic_inequality_theorem (a b : ℝ) :
  (∀ x, f a x > 4 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, 
    let g (x : ℝ) := a * x^2 - (a * c + b) * x + b * c
    if c > 2 then
      {x | g x < 0} = {x | 2 < x ∧ x < c}
    else if c < 2 then
      {x | g x < 0} = {x | c < x ∧ x < 2}
    else
      {x | g x < 0} = ∅) :=
by sorry

end quadratic_inequality_theorem_l3892_389248


namespace quadratic_equation_roots_l3892_389250

theorem quadratic_equation_roots (m : ℝ) :
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0 :=
by sorry

end quadratic_equation_roots_l3892_389250


namespace roger_donated_66_coins_l3892_389209

/-- Represents the number of coins Roger donated -/
def coins_donated (pennies nickels dimes coins_left : ℕ) : ℕ :=
  pennies + nickels + dimes - coins_left

/-- Proves that Roger donated 66 coins given the initial counts and remaining coins -/
theorem roger_donated_66_coins (h1 : coins_donated 42 36 15 27 = 66) : 
  coins_donated 42 36 15 27 = 66 := by
  sorry

end roger_donated_66_coins_l3892_389209


namespace system_solution_l3892_389255

theorem system_solution : 
  let x : ℚ := 25 / 31
  let y : ℚ := -11 / 31
  (3 * x + 4 * y = 1) ∧ (7 * x - y = 6) := by
sorry

end system_solution_l3892_389255


namespace first_candidate_marks_l3892_389298

/-- Represents the total marks in the exam -/
def total_marks : ℝ := 600

/-- Represents the passing marks -/
def passing_marks : ℝ := 240

/-- Represents the percentage of marks obtained by the first candidate -/
def first_candidate_percentage : ℝ := 30

/-- Theorem stating the percentage of marks obtained by the first candidate -/
theorem first_candidate_marks :
  let second_candidate_marks := 0.45 * total_marks
  let first_candidate_marks := (first_candidate_percentage / 100) * total_marks
  (second_candidate_marks = passing_marks + 30) ∧
  (first_candidate_marks = passing_marks - 60) →
  first_candidate_percentage = 30 := by
sorry

end first_candidate_marks_l3892_389298


namespace absolute_value_equation_solution_count_l3892_389276

theorem absolute_value_equation_solution_count :
  ∃! (s : Finset ℤ), (∀ a ∈ s, |3*a+7| + |3*a-5| = 12) ∧ s.card = 4 := by
  sorry

end absolute_value_equation_solution_count_l3892_389276


namespace ideal_gas_entropy_change_l3892_389241

/-- Entropy change for an ideal gas under different conditions -/
theorem ideal_gas_entropy_change
  (m μ R Cp Cv : ℝ)
  (P V T P1 P2 V1 V2 T1 T2 : ℝ)
  (h_ideal_gas : P * V = (m / μ) * R * T)
  (h_m_pos : m > 0)
  (h_μ_pos : μ > 0)
  (h_R_pos : R > 0)
  (h_Cp_pos : Cp > 0)
  (h_Cv_pos : Cv > 0)
  (h_P_pos : P > 0)
  (h_V_pos : V > 0)
  (h_T_pos : T > 0)
  (h_P1_pos : P1 > 0)
  (h_P2_pos : P2 > 0)
  (h_V1_pos : V1 > 0)
  (h_V2_pos : V2 > 0)
  (h_T1_pos : T1 > 0)
  (h_T2_pos : T2 > 0) :
  (∃ ΔS : ℝ,
    (P = P1 ∧ P = P2 → ΔS = (m / μ) * Cp * Real.log (V2 / V1)) ∧
    (V = V1 ∧ V = V2 → ΔS = (m / μ) * Cv * Real.log (P2 / P1)) ∧
    (T = T1 ∧ T = T2 → ΔS = (m / μ) * R * Real.log (V2 / V1))) :=
by sorry

end ideal_gas_entropy_change_l3892_389241


namespace melies_remaining_money_l3892_389282

/-- Calculates the remaining money after Méliès buys groceries -/
theorem melies_remaining_money :
  let meat_weight : ℝ := 3.5
  let meat_price_per_kg : ℝ := 95
  let vegetable_weight : ℝ := 4
  let vegetable_price_per_kg : ℝ := 18
  let fruit_weight : ℝ := 2.5
  let fruit_price_per_kg : ℝ := 12
  let initial_money : ℝ := 450
  let total_cost : ℝ := meat_weight * meat_price_per_kg +
                        vegetable_weight * vegetable_price_per_kg +
                        fruit_weight * fruit_price_per_kg
  let remaining_money : ℝ := initial_money - total_cost
  remaining_money = 15.5 := by sorry

end melies_remaining_money_l3892_389282


namespace surface_area_ratio_l3892_389235

-- Define the side length of the cube
variable (s : ℝ)

-- Define the surface area of a cube
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the surface area of the rectangular solid
def rectangular_solid_surface_area (s : ℝ) : ℝ := 2 * (2*s*s + 2*s*s + s*s)

-- Theorem statement
theorem surface_area_ratio :
  (cube_surface_area s) / (rectangular_solid_surface_area s) = 3/5 := by
  sorry

end surface_area_ratio_l3892_389235


namespace two_sunny_days_probability_l3892_389283

/-- The probability of exactly k successes in n independent trials,
    each with probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem two_sunny_days_probability :
  binomial_probability 5 2 (3/10 : ℝ) = 3087/10000 := by
  sorry

end two_sunny_days_probability_l3892_389283


namespace phone_call_probability_l3892_389272

/-- The probability of answering a phone call at the first ring -/
def p_first : ℝ := 0.1

/-- The probability of answering a phone call at the second ring -/
def p_second : ℝ := 0.3

/-- The probability of answering a phone call at the third ring -/
def p_third : ℝ := 0.4

/-- The probability of answering a phone call at the fourth ring -/
def p_fourth : ℝ := 0.1

/-- The events of answering at each ring are mutually exclusive -/
axiom mutually_exclusive : True

/-- The probability of answering within the first four rings -/
def p_within_four : ℝ := p_first + p_second + p_third + p_fourth

theorem phone_call_probability :
  p_within_four = 0.9 :=
sorry

end phone_call_probability_l3892_389272


namespace arun_age_proof_l3892_389231

theorem arun_age_proof (A S G M : ℕ) : 
  A - 6 = 18 * G →
  G + 2 = M →
  M = 5 →
  S = A - 8 →
  A = 60 := by
  sorry

end arun_age_proof_l3892_389231


namespace sum_of_ratios_bound_l3892_389222

theorem sum_of_ratios_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ (3 : ℝ) / 2 := by sorry

end sum_of_ratios_bound_l3892_389222


namespace substitution_insufficient_for_identity_proof_l3892_389218

/-- A mathematical identity is an equality that holds for all values of the variables involved. -/
def MathematicalIdentity (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x

/-- Substitution method verifies if an expression holds true for particular values. -/
def SubstitutionMethod (f g : ℝ → ℝ) (values : Set ℝ) : Prop :=
  ∀ x ∈ values, f x = g x

/-- Theorem: Substituting numerical values is insufficient to conclusively prove an identity. -/
theorem substitution_insufficient_for_identity_proof :
  ∃ (f g : ℝ → ℝ) (values : Set ℝ), 
    SubstitutionMethod f g values ∧ ¬MathematicalIdentity f g :=
  sorry

#check substitution_insufficient_for_identity_proof

end substitution_insufficient_for_identity_proof_l3892_389218


namespace drum_capacity_ratio_l3892_389200

theorem drum_capacity_ratio (capacity_x capacity_y : ℝ) 
  (h1 : capacity_x > 0) 
  (h2 : capacity_y > 0) 
  (h3 : (1/2 : ℝ) * capacity_x + (2/5 : ℝ) * capacity_y = (65/100 : ℝ) * capacity_y) : 
  capacity_y / capacity_x = 1/2 := by
  sorry

end drum_capacity_ratio_l3892_389200


namespace burrito_combinations_l3892_389274

def number_of_ways_to_make_burritos : ℕ :=
  let max_beef := 4
  let max_chicken := 3
  let total_wraps := 5
  (Nat.choose total_wraps 3) + (Nat.choose total_wraps 2) + (Nat.choose total_wraps 1)

theorem burrito_combinations : number_of_ways_to_make_burritos = 25 := by
  sorry

end burrito_combinations_l3892_389274


namespace locus_is_conic_locus_degenerate_line_locus_circle_l3892_389207

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in the first quadrant -/
structure Square where
  a : ℝ
  A : Point
  B : Point

/-- Defines the locus of a point P relative to the square -/
def locus (s : Square) (P : Point) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ θ : ℝ, 
    x = P.x * Real.sin θ + (s.a - P.x) * Real.cos θ ∧
    y = (s.a - P.y) * Real.sin θ + P.y * Real.cos θ}

theorem locus_is_conic (s : Square) (P : Point) 
  (h1 : s.A.y = 0 ∧ s.B.x = 0)  -- A is on x-axis, B is on y-axis
  (h2 : 0 ≤ P.x ∧ P.x ≤ 2*s.a ∧ 0 ≤ P.y ∧ P.y ≤ 2*s.a)  -- P is inside or on the square
  : ∃ (A B C D E F : ℝ), 
    A * P.x^2 + B * P.x * P.y + C * P.y^2 + D * P.x + E * P.y + F = 0 :=
sorry

theorem locus_degenerate_line (s : Square) (P : Point)
  (h : P.y = P.x)  -- P is on the diagonal
  : ∃ (m b : ℝ), ∀ (x y : ℝ), (x, y) ∈ locus s P → y = m * x + b :=
sorry

theorem locus_circle (s : Square) (P : Point)
  (h : P.x = s.a ∧ P.y = 0)  -- P is at midpoint of AB
  : ∃ (c : Point) (r : ℝ), ∀ (x y : ℝ), 
    (x, y) ∈ locus s P → (x - c.x)^2 + (y - c.y)^2 = r^2 :=
sorry

end locus_is_conic_locus_degenerate_line_locus_circle_l3892_389207


namespace lunch_cakes_count_l3892_389290

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := sorry

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served -/
def total_cakes : ℕ := 14

/-- Theorem stating that the number of cakes served during lunch today is 5 -/
theorem lunch_cakes_count : lunch_cakes = 5 := by
  sorry

#check lunch_cakes_count

end lunch_cakes_count_l3892_389290


namespace parabola_coefficient_sum_l3892_389296

/-- A parabola passing through (-3, 0) with axis of symmetry x = -1 has coefficient sum of 0 -/
theorem parabola_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = -3 ∨ x = 1) →
  -b / (2 * a) = -1 →
  a + b + c = 0 := by
sorry

end parabola_coefficient_sum_l3892_389296


namespace eight_divided_by_one_eighth_l3892_389285

theorem eight_divided_by_one_eighth (x y : ℝ) : x = 8 ∧ y = 1/8 → x / y = 64 := by
  sorry

end eight_divided_by_one_eighth_l3892_389285


namespace service_cost_calculation_l3892_389266

/-- The service cost per vehicle at a fuel station -/
def service_cost_per_vehicle : ℝ := 2.20

/-- The cost of fuel per liter -/
def fuel_cost_per_liter : ℝ := 0.70

/-- The capacity of a mini-van's fuel tank in liters -/
def minivan_tank_capacity : ℝ := 65

/-- The capacity of a truck's fuel tank in liters -/
def truck_tank_capacity : ℝ := minivan_tank_capacity * 2.2

/-- The number of mini-vans filled up -/
def num_minivans : ℕ := 3

/-- The number of trucks filled up -/
def num_trucks : ℕ := 2

/-- The total cost for filling up all vehicles -/
def total_cost : ℝ := 347.7

/-- Theorem stating that the service cost per vehicle is correct given the problem conditions -/
theorem service_cost_calculation :
  service_cost_per_vehicle * (num_minivans + num_trucks : ℝ) +
  fuel_cost_per_liter * (num_minivans * minivan_tank_capacity + num_trucks * truck_tank_capacity) =
  total_cost :=
by sorry

end service_cost_calculation_l3892_389266


namespace pencil_store_theorem_l3892_389205

/-- Represents the store's pencil purchases and sales -/
structure PencilStore where
  first_purchase_cost : ℝ
  first_purchase_quantity : ℝ
  second_purchase_cost : ℝ
  second_purchase_quantity : ℝ
  selling_price : ℝ

/-- The conditions of the pencil store problem -/
def pencil_store_conditions (s : PencilStore) : Prop :=
  s.first_purchase_cost * s.first_purchase_quantity = 600 ∧
  s.second_purchase_cost * s.second_purchase_quantity = 600 ∧
  s.second_purchase_cost = (5/4) * s.first_purchase_cost ∧
  s.second_purchase_quantity = s.first_purchase_quantity - 30

/-- The profit calculation for the pencil store -/
def profit (s : PencilStore) : ℝ :=
  s.selling_price * (s.first_purchase_quantity + s.second_purchase_quantity) -
  (s.first_purchase_cost * s.first_purchase_quantity + s.second_purchase_cost * s.second_purchase_quantity)

/-- The main theorem about the pencil store problem -/
theorem pencil_store_theorem (s : PencilStore) :
  pencil_store_conditions s →
  s.first_purchase_cost = 4 ∧
  (∀ p, profit { s with selling_price := p } ≥ 420 → p ≥ 6) :=
by sorry


end pencil_store_theorem_l3892_389205


namespace inequality_proof_l3892_389265

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end inequality_proof_l3892_389265


namespace caravan_keeper_count_l3892_389244

/-- Represents the number of keepers in the caravan -/
def num_keepers : ℕ := 10

/-- Represents the number of hens in the caravan -/
def num_hens : ℕ := 60

/-- Represents the number of goats in the caravan -/
def num_goats : ℕ := 35

/-- Represents the number of camels in the caravan -/
def num_camels : ℕ := 6

/-- Represents the number of feet for a hen -/
def hen_feet : ℕ := 2

/-- Represents the number of feet for a goat or camel -/
def goat_camel_feet : ℕ := 4

/-- Represents the number of feet for a keeper -/
def keeper_feet : ℕ := 2

/-- Represents the difference between total feet and total heads -/
def feet_head_difference : ℕ := 193

theorem caravan_keeper_count :
  num_keepers * keeper_feet +
  num_hens * hen_feet +
  num_goats * goat_camel_feet +
  num_camels * goat_camel_feet =
  (num_keepers + num_hens + num_goats + num_camels + feet_head_difference) :=
by sorry

end caravan_keeper_count_l3892_389244


namespace base9_to_base3_7254_l3892_389270

/-- Converts a single digit from base 9 to its two-digit representation in base 3 -/
def base9_to_base3_digit (d : Nat) : Nat := sorry

/-- Converts a number from base 9 to base 3 -/
def base9_to_base3 (n : Nat) : Nat := sorry

theorem base9_to_base3_7254 :
  base9_to_base3 7254 = 210212113 := by sorry

end base9_to_base3_7254_l3892_389270


namespace tennis_balls_per_pack_l3892_389247

theorem tennis_balls_per_pack (num_packs : ℕ) (total_cost : ℕ) (cost_per_ball : ℕ) : 
  num_packs = 4 → total_cost = 24 → cost_per_ball = 2 → 
  (total_cost / cost_per_ball) / num_packs = 3 :=
by
  sorry

end tennis_balls_per_pack_l3892_389247


namespace train_length_l3892_389208

/-- The length of a train given its speed and time to pass a fixed point --/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 ∧ time = 28 → speed * time * (1000 / 3600) = 280 :=
by sorry

end train_length_l3892_389208


namespace inequality_proof_l3892_389215

theorem inequality_proof (a b : ℝ) : a^2 + a*b + b^2 ≥ 3*(a + b - 1) := by
  sorry

end inequality_proof_l3892_389215


namespace picture_area_l3892_389293

theorem picture_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 * x + 5) * (y + 3) = 60) : x * y = 27 := by
  sorry

end picture_area_l3892_389293


namespace students_per_group_l3892_389230

theorem students_per_group (total_students : Nat) (num_teachers : Nat) 
  (h1 : total_students = 256) 
  (h2 : num_teachers = 8) 
  (h3 : num_teachers > 0) : 
  total_students / num_teachers = 32 := by
  sorry

end students_per_group_l3892_389230


namespace comic_books_triple_storybooks_l3892_389262

/-- The number of days after which the number of comic books is three times the number of storybooks -/
def days_until_triple_ratio : ℕ := 20

/-- The initial number of comic books -/
def initial_comic_books : ℕ := 140

/-- The initial number of storybooks -/
def initial_storybooks : ℕ := 100

/-- The number of books borrowed per day for each type -/
def daily_borrowing_rate : ℕ := 4

theorem comic_books_triple_storybooks :
  initial_comic_books - days_until_triple_ratio * daily_borrowing_rate =
  3 * (initial_storybooks - days_until_triple_ratio * daily_borrowing_rate) := by
  sorry

end comic_books_triple_storybooks_l3892_389262


namespace car_distance_proof_l3892_389201

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + initialSpeed + h * speedIncrease) 0

/-- Proves that a car traveling 45 km in the first hour and increasing speed by 2 km/h
    each hour will travel 672 km in 12 hours. -/
theorem car_distance_proof :
  totalDistance 45 2 12 = 672 := by
  sorry

#eval totalDistance 45 2 12

end car_distance_proof_l3892_389201


namespace asymptote_sum_l3892_389257

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) where A, B, C are integers,
    if the graph has vertical asymptotes at x = -3, 0, 3, then A + B + C = -9 -/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    ∃ y : ℝ, y = x / (x^3 + A*x^2 + B*x + C)) →
  A + B + C = -9 := by
  sorry

end asymptote_sum_l3892_389257


namespace two_pairs_more_likely_than_three_of_a_kind_l3892_389242

def num_dice : ℕ := 5
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ num_dice

def two_pairs_outcomes : ℕ := 
  num_dice * faces_per_die * (num_dice - 1).choose 2 * (faces_per_die - 1) * (faces_per_die - 2)

def three_of_a_kind_outcomes : ℕ := 
  num_dice.choose 3 * faces_per_die * (faces_per_die - 1) * (faces_per_die - 2)

theorem two_pairs_more_likely_than_three_of_a_kind :
  (two_pairs_outcomes : ℚ) / total_outcomes > (three_of_a_kind_outcomes : ℚ) / total_outcomes :=
by sorry

end two_pairs_more_likely_than_three_of_a_kind_l3892_389242


namespace remainder_n_squared_plus_2n_plus_3_l3892_389202

theorem remainder_n_squared_plus_2n_plus_3 (n : ℤ) (a : ℤ) (h : n = 100 * a - 1) :
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end remainder_n_squared_plus_2n_plus_3_l3892_389202


namespace division_result_approx_point_zero_seven_l3892_389243

-- Define the approximation tolerance
def tolerance : ℝ := 0.001

-- Define the condition that 35 divided by x is approximately 500
def divisionApprox (x : ℝ) : Prop := 
  abs (35 / x - 500) < tolerance

-- Theorem statement
theorem division_result_approx_point_zero_seven :
  ∃ x : ℝ, divisionApprox x ∧ abs (x - 0.07) < tolerance :=
sorry

end division_result_approx_point_zero_seven_l3892_389243


namespace park_conditions_l3892_389277

-- Define the basic conditions
def temperature_at_least_70 : Prop := sorry
def is_sunny : Prop := sorry
def park_is_packed : Prop := sorry

-- Define the main theorem
theorem park_conditions :
  (temperature_at_least_70 ∧ is_sunny → park_is_packed) →
  (¬park_is_packed → ¬temperature_at_least_70 ∨ ¬is_sunny) :=
by sorry

end park_conditions_l3892_389277


namespace dot_product_of_specific_vectors_l3892_389227

theorem dot_product_of_specific_vectors :
  let A : ℝ × ℝ := (Real.cos (110 * π / 180), Real.sin (110 * π / 180))
  let B : ℝ × ℝ := (Real.cos (50 * π / 180), Real.sin (50 * π / 180))
  let OA : ℝ × ℝ := A
  let OB : ℝ × ℝ := B
  (OA.1 * OB.1 + OA.2 * OB.2) = 1/2 := by
sorry


end dot_product_of_specific_vectors_l3892_389227


namespace square_park_area_l3892_389212

theorem square_park_area (side_length : ℝ) (h : side_length = 200) :
  side_length * side_length = 40000 := by
  sorry

end square_park_area_l3892_389212


namespace max_value_x_y4_z5_l3892_389252

theorem max_value_x_y4_z5 (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  ∃ (max : ℝ), max = 243 ∧ x + y^4 + z^5 ≤ max :=
sorry

end max_value_x_y4_z5_l3892_389252


namespace snake_toy_cost_l3892_389246

theorem snake_toy_cost (cage_cost total_cost : ℚ) (found_money : ℚ) : 
  cage_cost = 14.54 → 
  found_money = 1 → 
  total_cost = 26.3 → 
  total_cost = cage_cost + (12.76 : ℚ) - found_money := by sorry

end snake_toy_cost_l3892_389246


namespace completing_square_transformation_l3892_389210

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
sorry

end completing_square_transformation_l3892_389210


namespace additional_land_cost_l3892_389236

/-- Calculates the cost of additional land purchased by Carlson -/
theorem additional_land_cost (initial_area : ℝ) (final_area : ℝ) (cost_per_sqm : ℝ) :
  initial_area = 300 →
  final_area = 900 →
  cost_per_sqm = 20 →
  (final_area - initial_area) * cost_per_sqm = 12000 := by
  sorry

#check additional_land_cost

end additional_land_cost_l3892_389236


namespace arithmetic_sequence_solution_l3892_389269

/-- An arithmetic sequence with given third and eleventh terms -/
def ArithmeticSequence (a₃ a₁₁ : ℚ) :=
  ∃ (a₁ d : ℚ), a₃ = a₁ + 2 * d ∧ a₁₁ = a₁ + 10 * d

/-- Theorem stating the first term and common difference of the sequence -/
theorem arithmetic_sequence_solution :
  ∀ (a₃ a₁₁ : ℚ), a₃ = 3 ∧ a₁₁ = 15 →
  ArithmeticSequence a₃ a₁₁ →
  ∃ (a₁ d : ℚ), a₁ = 0 ∧ d = 3/2 :=
by sorry


end arithmetic_sequence_solution_l3892_389269


namespace sum_of_four_twos_to_fourth_l3892_389217

theorem sum_of_four_twos_to_fourth (n : ℕ) : 
  (2^4 : ℕ) + (2^4 : ℕ) + (2^4 : ℕ) + (2^4 : ℕ) = 2^6 := by
  sorry

end sum_of_four_twos_to_fourth_l3892_389217


namespace root_of_inverse_point_l3892_389273

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverse functions
variable (h_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)

-- Assume f_inv(0) = 2
variable (h_f_inv_zero : f_inv 0 = 2)

-- Theorem: If f_inv(0) = 2, then f(2) = 0
theorem root_of_inverse_point (f f_inv : ℝ → ℝ) 
  (h_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x) 
  (h_f_inv_zero : f_inv 0 = 2) : 
  f 2 = 0 := by
sorry

end root_of_inverse_point_l3892_389273


namespace silver_zinc_battery_properties_l3892_389203

/-- Represents an electrode in the battery -/
inductive Electrode
| Zinc
| SilverOxide

/-- Represents the direction of current flow -/
inductive CurrentFlow
| FromZincToSilverOxide
| FromSilverOxideToZinc

/-- Represents the change in OH⁻ concentration -/
inductive OHConcentrationChange
| Increase
| Decrease
| NoChange

/-- Models a silver-zinc battery -/
structure SilverZincBattery where
  negativeElectrode : Electrode
  positiveElectrode : Electrode
  zincReaction : String
  silverOxideReaction : String
  currentFlow : CurrentFlow
  ohConcentrationChange : OHConcentrationChange

/-- Theorem about the properties of a silver-zinc battery -/
theorem silver_zinc_battery_properties (battery : SilverZincBattery) 
  (h1 : battery.zincReaction = "Zn + 2OH⁻ - 2e⁻ = Zn(OH)₂")
  (h2 : battery.silverOxideReaction = "Ag₂O + H₂O + 2e⁻ = 2Ag + 2OH⁻") :
  battery.negativeElectrode = Electrode.Zinc ∧
  battery.positiveElectrode = Electrode.SilverOxide ∧
  battery.ohConcentrationChange = OHConcentrationChange.Increase ∧
  battery.currentFlow = CurrentFlow.FromSilverOxideToZinc :=
sorry

end silver_zinc_battery_properties_l3892_389203


namespace solution_set_characterization_l3892_389213

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x, (deriv f) x - 2 * f x < 0) ∧ 
  f 0 = 1

/-- The main theorem -/
theorem solution_set_characterization (f : ℝ → ℝ) (h : satisfies_conditions f) :
  ∀ x, f x > Real.exp (2 * x) ↔ x < 0 := by
  sorry

end solution_set_characterization_l3892_389213


namespace age_ratio_proof_l3892_389278

/-- Proves that given a person's age is 40, and 7 years earlier they were 11 times their daughter's age,
    the ratio of their age to their daughter's age today is 4:1 -/
theorem age_ratio_proof (your_age : ℕ) (daughter_age : ℕ) : 
  your_age = 40 →
  your_age - 7 = 11 * (daughter_age - 7) →
  your_age / daughter_age = 4 := by
sorry

end age_ratio_proof_l3892_389278


namespace village_population_panic_l3892_389291

theorem village_population_panic (original_population : ℕ) (final_population : ℕ) 
  (h1 : original_population = 7600)
  (h2 : final_population = 5130) :
  let remaining_after_initial := original_population - original_population / 10
  let left_during_panic := remaining_after_initial - final_population
  (left_during_panic : ℚ) / remaining_after_initial * 100 = 25 := by
  sorry

end village_population_panic_l3892_389291


namespace prob_two_non_defective_pens_l3892_389280

/-- Probability of selecting two non-defective pens from a box -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 16) (h2 : defective_pens = 3) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 13 / 20 := by
  sorry

end prob_two_non_defective_pens_l3892_389280


namespace sum_of_divisors_30_l3892_389214

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_30 : sum_of_divisors 30 = 72 := by sorry

end sum_of_divisors_30_l3892_389214


namespace star_calculation_l3892_389286

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := x^3 - y

-- State the theorem
theorem star_calculation :
  star (3^(star 5 18)) (2^(star 2 9)) = 3^321 - 1/2 := by
  sorry

end star_calculation_l3892_389286


namespace square_circle_distance_sum_constant_l3892_389267

/-- Given a square ABCD with side length 2a and a circle k centered at the center of the square with radius R, 
    the sum of squared distances from any point P on the circle to the vertices of the square is constant. -/
theorem square_circle_distance_sum_constant 
  (a R : ℝ) 
  (A B C D : ℝ × ℝ) 
  (k : Set (ℝ × ℝ)) 
  (h_square : A = (-a, a) ∧ B = (a, a) ∧ C = (a, -a) ∧ D = (-a, -a))
  (h_circle : k = {P : ℝ × ℝ | P.1^2 + P.2^2 = R^2}) :
  ∀ P ∈ k, 
    (P.1 - A.1)^2 + (P.2 - A.2)^2 + 
    (P.1 - B.1)^2 + (P.2 - B.2)^2 + 
    (P.1 - C.1)^2 + (P.2 - C.2)^2 + 
    (P.1 - D.1)^2 + (P.2 - D.2)^2 = 4*R^2 + 8*a^2 :=
by sorry

end square_circle_distance_sum_constant_l3892_389267


namespace fold_line_length_squared_fold_line_theorem_l3892_389289

/-- Represents an equilateral triangle with side length 15 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 15

/-- Represents the folded triangle -/
structure FoldedTriangle extends EquilateralTriangle where
  fold_distance : ℝ
  is_valid_fold : fold_distance = 11

/-- The theorem stating the square of the fold line length -/
theorem fold_line_length_squared (t : FoldedTriangle) : ℝ :=
  2174209 / 78281

/-- The main theorem to be proved -/
theorem fold_line_theorem (t : FoldedTriangle) : 
  fold_line_length_squared t = 2174209 / 78281 := by
  sorry

end fold_line_length_squared_fold_line_theorem_l3892_389289


namespace last_two_digits_of_7_power_last_two_digits_of_7_2017_l3892_389219

def last_two_digits (n : ℕ) : ℕ := n % 100

def power_pattern (k : ℕ) : ℕ :=
  match k % 4 with
  | 0 => 01
  | 1 => 07
  | 2 => 49
  | 3 => 43
  | _ => 0  -- This case should never occur

theorem last_two_digits_of_7_power (n : ℕ) :
  last_two_digits (7^n) = power_pattern n :=
sorry

theorem last_two_digits_of_7_2017 :
  last_two_digits (7^2017) = 07 :=
sorry

end last_two_digits_of_7_power_last_two_digits_of_7_2017_l3892_389219


namespace book_cost_price_l3892_389237

theorem book_cost_price (cost : ℝ) : 
  (1.15 * cost - 1.10 * cost = 100) → cost = 2000 := by
sorry

end book_cost_price_l3892_389237


namespace haleigh_leggings_needed_l3892_389271

/-- The number of pairs of leggings needed for pets -/
def leggings_needed (num_dogs : ℕ) (num_cats : ℕ) (legs_per_animal : ℕ) (legs_per_legging : ℕ) : ℕ :=
  ((num_dogs + num_cats) * legs_per_animal) / legs_per_legging

/-- Theorem: Haleigh needs 14 pairs of leggings for her pets -/
theorem haleigh_leggings_needed :
  leggings_needed 4 3 4 2 = 14 := by
  sorry

end haleigh_leggings_needed_l3892_389271


namespace cable_length_l3892_389297

/-- The length of the curve defined by the intersection of a plane and a sphere --/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 → 
  x * y + y * z + x * z = -22 → 
  (∃ (l : ℝ), l = 4 * Real.pi * Real.sqrt (83 / 3) ∧ 
   l = 2 * Real.pi * Real.sqrt (144 - (10^2 / 3))) :=
by sorry

end cable_length_l3892_389297


namespace direction_vector_b_l3892_389245

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (dir : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p2 = (p1.1 + t * dir.1, p1.2 + t * dir.2)

theorem direction_vector_b (b : ℝ) :
  Line (-3, 0) (0, 3) (3, b) → b = 3 :=
by
  sorry

end direction_vector_b_l3892_389245


namespace total_amount_received_l3892_389221

/-- The amount John won in the lottery -/
def lottery_winnings : ℚ := 155250

/-- The number of top students receiving money -/
def num_students : ℕ := 100

/-- The fraction of the winnings given to each student -/
def fraction_given : ℚ := 1 / 1000

theorem total_amount_received (lottery_winnings : ℚ) (num_students : ℕ) (fraction_given : ℚ) :
  (lottery_winnings * fraction_given) * num_students = 15525 :=
sorry

end total_amount_received_l3892_389221


namespace smallest_abb_value_l3892_389225

theorem smallest_abb_value (A B : Nat) : 
  A ≠ B →
  1 ≤ A ∧ A ≤ 9 →
  1 ≤ B ∧ B ≤ 9 →
  10 * A + B = (100 * A + 11 * B) / 7 →
  ∀ (X Y : Nat), 
    X ≠ Y →
    1 ≤ X ∧ X ≤ 9 →
    1 ≤ Y ∧ Y ≤ 9 →
    10 * X + Y = (100 * X + 11 * Y) / 7 →
    100 * A + 11 * B ≤ 100 * X + 11 * Y →
  100 * A + 11 * B = 466 :=
sorry

end smallest_abb_value_l3892_389225


namespace first_job_men_l3892_389249

/-- The number of men who worked on the first job -/
def M : ℕ := 250

/-- The number of days for the first job -/
def days_job1 : ℕ := 16

/-- The number of men working on the second job -/
def men_job2 : ℕ := 600

/-- The number of days for the second job -/
def days_job2 : ℕ := 20

/-- The ratio of work between the second and first job -/
def work_ratio : ℕ := 3

theorem first_job_men :
  M * days_job1 * work_ratio = men_job2 * days_job2 := by
  sorry

#check first_job_men

end first_job_men_l3892_389249


namespace calculation_proof_l3892_389261

theorem calculation_proof : (4.5 - 1.23) * 2.1 = 6.867 := by
  sorry

end calculation_proof_l3892_389261


namespace johns_former_wage_l3892_389228

/-- Represents John's work schedule and wage information -/
structure WorkInfo where
  hours_per_workday : ℕ
  days_between_workdays : ℕ
  monthly_pay : ℕ
  days_in_month : ℕ
  raise_percentage : ℚ

/-- Calculates the former hourly wage given the work information -/
def former_hourly_wage (info : WorkInfo) : ℚ :=
  let days_worked := info.days_in_month / (info.days_between_workdays + 1)
  let total_hours := days_worked * info.hours_per_workday
  let current_hourly_wage := info.monthly_pay / total_hours
  current_hourly_wage / (1 + info.raise_percentage)

/-- Theorem stating that John's former hourly wage was $20 -/
theorem johns_former_wage (info : WorkInfo) 
  (h1 : info.hours_per_workday = 12)
  (h2 : info.days_between_workdays = 1)
  (h3 : info.monthly_pay = 4680)
  (h4 : info.days_in_month = 30)
  (h5 : info.raise_percentage = 3/10) :
  former_hourly_wage info = 20 := by
  sorry

end johns_former_wage_l3892_389228


namespace greatest_power_of_two_factor_l3892_389284

theorem greatest_power_of_two_factor (n : ℕ) : n = 1003 →
  ∃ k : ℕ, (2^n : ℤ) ∣ (10^n - 4^(n/2)) ∧
  ∀ m : ℕ, m > n → ¬((2^m : ℤ) ∣ (10^n - 4^(n/2))) :=
by sorry

end greatest_power_of_two_factor_l3892_389284


namespace range_of_a_l3892_389233

def set_A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

def set_B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

theorem range_of_a (a : ℝ) : set_A a ∩ set_B = ∅ → 2 < a ∧ a < 3 := by
  sorry

end range_of_a_l3892_389233


namespace decreasing_function_implies_a_bound_l3892_389240

variable (a : ℝ)

def f (x : ℝ) := (x - 1)^2 + 2*a*x + 1

theorem decreasing_function_implies_a_bound :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 4 → f a x₁ > f a x₂) →
  a ≤ -3 :=
by
  sorry

end decreasing_function_implies_a_bound_l3892_389240


namespace cubic_equation_one_root_l3892_389259

theorem cubic_equation_one_root (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 := by
  sorry

end cubic_equation_one_root_l3892_389259


namespace mathematician_project_time_l3892_389288

theorem mathematician_project_time (project1 : ℕ) (project2 : ℕ) (daily_questions : ℕ) : 
  project1 = 518 → project2 = 476 → daily_questions = 142 → 
  (project1 + project2) / daily_questions = 7 := by
  sorry

end mathematician_project_time_l3892_389288


namespace equation_solution_l3892_389206

theorem equation_solution : ∃ x : ℚ, (x - 1) / 2 = 1 - (3 * x + 2) / 5 ↔ x = 1 := by
  sorry

end equation_solution_l3892_389206


namespace simplify_expression_l3892_389260

theorem simplify_expression (b : ℝ) : (1:ℝ) * (3*b) * (5*b^2) * (7*b^3) * (9*b^4) = 945 * b^10 := by
  sorry

end simplify_expression_l3892_389260


namespace right_triangle_area_and_hypotenuse_l3892_389239

theorem right_triangle_area_and_hypotenuse 
  (leg1 leg2 : ℝ) 
  (h_leg1 : leg1 = 30) 
  (h_leg2 : leg2 = 45) : 
  (1/2 * leg1 * leg2 = 675) ∧ 
  (Real.sqrt (leg1^2 + leg2^2) = 54) := by
  sorry

end right_triangle_area_and_hypotenuse_l3892_389239


namespace cube_expansion_seven_plus_one_l3892_389253

theorem cube_expansion_seven_plus_one : 7^3 + 3*(7^2) + 3*7 + 1 = 512 := by
  sorry

end cube_expansion_seven_plus_one_l3892_389253


namespace largest_number_l3892_389220

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 0.986) 
  (hb : b = 0.9851) 
  (hc : c = 0.9869) 
  (hd : d = 0.9807) 
  (he : e = 0.9819) : 
  max a (max b (max c (max d e))) = c := by
  sorry

end largest_number_l3892_389220


namespace square_of_binomial_constant_l3892_389295

theorem square_of_binomial_constant (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 4*x^2 + 16*x + m = (a*x + b)^2) → m = 16 := by
sorry

end square_of_binomial_constant_l3892_389295


namespace sqrt_equation_l3892_389275

theorem sqrt_equation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end sqrt_equation_l3892_389275


namespace mabels_garden_petal_count_l3892_389211

/-- The number of petals remaining in Mabel's garden after a series of events -/
def final_petal_count (initial_daisies : ℕ) (initial_petals_per_daisy : ℕ) 
  (daisies_given_away : ℕ) (new_daisies : ℕ) (new_petals_per_daisy : ℕ) 
  (petals_lost_new_daisies : ℕ) (petals_lost_original_daisies : ℕ) : ℕ :=
  let initial_petals := initial_daisies * initial_petals_per_daisy
  let remaining_petals := initial_petals - (daisies_given_away * initial_petals_per_daisy)
  let new_petals := new_daisies * new_petals_per_daisy
  let total_petals := remaining_petals + new_petals
  total_petals - (petals_lost_new_daisies + petals_lost_original_daisies)

/-- Theorem stating that the final petal count in Mabel's garden is 39 -/
theorem mabels_garden_petal_count :
  final_petal_count 5 8 2 3 7 4 2 = 39 := by
  sorry

end mabels_garden_petal_count_l3892_389211


namespace evaluate_nested_root_l3892_389232

theorem evaluate_nested_root : (((4 ^ (1/3)) ^ 4) ^ (1/2)) ^ 6 = 256 := by
  sorry

end evaluate_nested_root_l3892_389232


namespace total_wheels_count_l3892_389226

theorem total_wheels_count (num_bicycles num_tricycles : ℕ) 
  (wheels_per_bicycle wheels_per_tricycle : ℕ) :
  num_bicycles = 24 →
  num_tricycles = 14 →
  wheels_per_bicycle = 2 →
  wheels_per_tricycle = 3 →
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle = 90 := by
  sorry

end total_wheels_count_l3892_389226


namespace quick_multiply_correct_l3892_389258

/-- Represents a two-digit number -/
def TwoDigitNumber (tens ones : Nat) : Nat :=
  10 * tens + ones

/-- The quick multiplication formula for two-digit numbers with reversed digits -/
def quickMultiply (x y : Nat) : Nat :=
  101 * x * y + 10 * (x^2 + y^2)

/-- Theorem stating that the quick multiplication formula is correct -/
theorem quick_multiply_correct (x y : Nat) (h1 : x < 10) (h2 : y < 10) :
  (TwoDigitNumber x y) * (TwoDigitNumber y x) = quickMultiply x y :=
by
  sorry

end quick_multiply_correct_l3892_389258


namespace money_difference_l3892_389224

/-- The problem statement about Isabella, Sam, and Giselle's money --/
theorem money_difference (isabella sam giselle : ℕ) : 
  isabella = sam + 45 →  -- Isabella has $45 more than Sam
  giselle = 120 →  -- Giselle has $120
  isabella + sam + giselle = 3 * 115 →  -- Total money shared equally among 3 shoppers
  isabella - giselle = 15 :=  -- Isabella has $15 more than Giselle
by sorry

end money_difference_l3892_389224


namespace apple_pricing_l3892_389281

/-- The cost of apples per kilogram for the first 30 kgs -/
def l : ℝ := 0.362

/-- The cost of apples per kilogram for each additional kg after 30 kgs -/
def m : ℝ := 0.27

/-- The price of 33 kilograms of apples -/
def price_33kg : ℝ := 11.67

/-- The price of 36 kilograms of apples -/
def price_36kg : ℝ := 12.48

/-- The cost of the first 10 kgs of apples -/
def cost_10kg : ℝ := 3.62

theorem apple_pricing :
  (10 * l = cost_10kg) ∧
  (30 * l + 3 * m = price_33kg) ∧
  (30 * l + 6 * m = price_36kg) →
  m = 0.27 := by
  sorry

end apple_pricing_l3892_389281


namespace wrapping_paper_area_correct_l3892_389256

/-- Represents a box with square base -/
structure Box where
  w : ℝ  -- width of the base
  h : ℝ  -- height of the box

/-- Calculates the area of the wrapping paper for a given box -/
def wrappingPaperArea (box : Box) : ℝ :=
  6 * box.w * box.h + box.h^2

/-- Theorem stating that the area of the wrapping paper is correct -/
theorem wrapping_paper_area_correct (box : Box) :
  wrappingPaperArea box = 6 * box.w * box.h + box.h^2 :=
by sorry

end wrapping_paper_area_correct_l3892_389256


namespace speed_in_still_water_problem_l3892_389223

/-- Calculates the speed in still water given the downstream speed and current speed. -/
def speed_in_still_water (downstream_speed current_speed : ℝ) : ℝ :=
  downstream_speed - current_speed

/-- Theorem: Given the conditions from the problem, the speed in still water is 30 kmph. -/
theorem speed_in_still_water_problem :
  let downstream_distance : ℝ := 0.24 -- 240 meters in km
  let downstream_time : ℝ := 24 / 3600 -- 24 seconds in hours
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let current_speed : ℝ := 6
  speed_in_still_water downstream_speed current_speed = 30 := by
  sorry

end speed_in_still_water_problem_l3892_389223


namespace dice_tosses_probability_l3892_389268

/-- The number of sides on the dice -/
def num_sides : ℕ := 8

/-- The probability of rolling a 3 on a single toss -/
def p_roll_3 : ℚ := 1 / num_sides

/-- The target probability of rolling a 3 at least once -/
def target_prob : ℚ := 111328125 / 1000000000

/-- The number of tosses -/
def num_tosses : ℕ := 7

theorem dice_tosses_probability :
  1 - (1 - p_roll_3) ^ num_tosses = target_prob := by sorry

end dice_tosses_probability_l3892_389268


namespace optimal_strategy_probability_l3892_389292

/-- Represents the color of a hat -/
inductive HatColor
| Red
| Blue

/-- A strategy for guessing hat colors -/
def Strategy := (n : Nat) → (Vector HatColor n) → Vector Bool n

/-- The probability of all prisoners guessing correctly given a strategy -/
def SuccessProbability (n : Nat) (s : Strategy) : ℚ :=
  sorry

/-- Theorem stating that the maximum success probability is 1/2 -/
theorem optimal_strategy_probability (n : Nat) :
  ∀ s : Strategy, SuccessProbability n s ≤ 1/2 :=
sorry

end optimal_strategy_probability_l3892_389292


namespace patrol_results_l3892_389238

/-- Represents the patrol records of the police car --/
def patrol_records : List Int := [6, -8, 9, -5, 4, -3]

/-- Fuel consumption rate in liters per kilometer --/
def fuel_consumption_rate : ℚ := 0.2

/-- Initial fuel in the tank in liters --/
def initial_fuel : ℚ := 5

/-- Calculates the final position of the police car --/
def final_position (records : List Int) : Int :=
  records.sum

/-- Calculates the total distance traveled --/
def total_distance (records : List Int) : Int :=
  records.map (abs) |>.sum

/-- Calculates the total fuel consumed --/
def total_fuel_consumed (distance : Int) (rate : ℚ) : ℚ :=
  (distance : ℚ) * rate

/-- Calculates the additional fuel needed --/
def additional_fuel_needed (consumed : ℚ) (initial : ℚ) : ℚ :=
  max (consumed - initial) 0

theorem patrol_results :
  (final_position patrol_records = 3) ∧
  (total_fuel_consumed (total_distance patrol_records) fuel_consumption_rate = 7) ∧
  (additional_fuel_needed (total_fuel_consumed (total_distance patrol_records) fuel_consumption_rate) initial_fuel = 2) :=
by sorry

end patrol_results_l3892_389238


namespace limit_is_nonzero_real_l3892_389279

noncomputable def f (x : ℝ) : ℝ := x^(5/3) * ((x + 1)^(1/3) + (x - 1)^(1/3) - 2 * x^(1/3))

theorem limit_is_nonzero_real : ∃ (L : ℝ), L ≠ 0 ∧ Filter.Tendsto f Filter.atTop (nhds L) := by
  sorry

end limit_is_nonzero_real_l3892_389279
