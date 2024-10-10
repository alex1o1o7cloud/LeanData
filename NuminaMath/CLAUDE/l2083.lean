import Mathlib

namespace min_value_of_f_l2083_208330

/-- Given positive real numbers a, b, c, x, y, z satisfying certain conditions,
    the function f(x, y, z) has a minimum value of 1/2. -/
theorem min_value_of_f (a b c x y z : ℝ) 
    (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
    (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
    (eq1 : c * y + b * z = a)
    (eq2 : a * z + c * x = b)
    (eq3 : b * x + a * y = c) :
    let f := fun (x y z : ℝ) => x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)
    ∀ x' y' z' : ℝ, 0 < x' → 0 < y' → 0 < z' → f x' y' z' ≥ 1/2 ∧ 
    ∃ x₀ y₀ z₀ : ℝ, 0 < x₀ ∧ 0 < y₀ ∧ 0 < z₀ ∧ f x₀ y₀ z₀ = 1/2 := by
  sorry

end min_value_of_f_l2083_208330


namespace oscar_christina_age_ratio_l2083_208346

def christina_age : ℕ := sorry
def oscar_age : ℕ := 6

theorem oscar_christina_age_ratio :
  (oscar_age + 15) / christina_age = 3 / 5 :=
by
  have h1 : christina_age + 5 = 80 / 2 := by sorry
  sorry

end oscar_christina_age_ratio_l2083_208346


namespace joey_work_hours_l2083_208368

/-- Calculates the number of hours Joey needs to work to buy sneakers -/
def hours_needed (sneaker_cost lawn_count lawn_pay figure_count figure_pay hourly_wage : ℕ) : ℕ :=
  let lawn_income := lawn_count * lawn_pay
  let figure_income := figure_count * figure_pay
  let total_income := lawn_income + figure_income
  let remaining_cost := sneaker_cost - total_income
  remaining_cost / hourly_wage

/-- Proves that Joey needs to work 10 hours to buy the sneakers -/
theorem joey_work_hours : 
  hours_needed 92 3 8 2 9 5 = 10 := by
  sorry

end joey_work_hours_l2083_208368


namespace wrapping_paper_fraction_l2083_208355

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_small : ℕ) (num_large : ℕ) :
  total_fraction = 3/8 →
  num_small = 4 →
  num_large = 2 →
  (∃ small_fraction : ℚ, 
    total_fraction = num_small * small_fraction + num_large * (2 * small_fraction) ∧
    small_fraction = 3/64) :=
by sorry

end wrapping_paper_fraction_l2083_208355


namespace students_in_biology_or_chemistry_l2083_208367

theorem students_in_biology_or_chemistry (both : ℕ) (biology : ℕ) (chemistry_only : ℕ) : 
  both = 15 → biology = 35 → chemistry_only = 18 → 
  (biology - both) + chemistry_only = 38 := by
sorry

end students_in_biology_or_chemistry_l2083_208367


namespace complement_intersection_MN_l2083_208337

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_MN : 
  (M ∩ N)ᶜ = {1, 3, 4} :=
by sorry

end complement_intersection_MN_l2083_208337


namespace min_sum_of_product_2004_l2083_208334

theorem min_sum_of_product_2004 (x y z : ℕ+) (h : x * y * z = 2004) :
  ∃ (a b c : ℕ+), a * b * c = 2004 ∧ a + b + c ≤ x + y + z ∧ a + b + c = 174 :=
sorry

end min_sum_of_product_2004_l2083_208334


namespace inverse_difference_l2083_208324

-- Define a real-valued function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the condition that f(x+2) is the inverse of f⁻¹(x-1)
axiom inverse_condition : ∀ x, f (x + 2) = f_inv (x - 1)

-- Define the theorem
theorem inverse_difference :
  f_inv 2010 - f_inv 1 = 4018 :=
sorry

end inverse_difference_l2083_208324


namespace eight_power_y_equals_one_eighth_of_two_power_36_l2083_208393

theorem eight_power_y_equals_one_eighth_of_two_power_36 :
  ∀ y : ℝ, (1/8 : ℝ) * (2^36) = 8^y → y = 11 := by
  sorry

end eight_power_y_equals_one_eighth_of_two_power_36_l2083_208393


namespace tan_cube_identity_l2083_208342

theorem tan_cube_identity (x y : ℝ) (φ : ℝ) (h : Real.tan φ ^ 3 = x / y) :
  x / Real.sin φ + y / Real.cos φ = (x ^ (2/3) + y ^ (2/3)) ^ (3/2) := by
  sorry

end tan_cube_identity_l2083_208342


namespace inequality_proof_l2083_208304

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end inequality_proof_l2083_208304


namespace value_of_m_l2083_208379

/-- A function f(x) is a direct proportion function with respect to x if f(x) = kx for some constant k ≠ 0 -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- A function f(x) passes through the second and fourth quadrants if f(x) < 0 for x > 0 and f(x) > 0 for x < 0 -/
def PassesThroughSecondAndFourthQuadrants (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x < 0) ∧ (∀ x < 0, f x > 0)

/-- The main theorem -/
theorem value_of_m (m : ℝ) :
  IsDirectProportionFunction (fun x ↦ (m - 2) * x^(m^2 - 8)) ∧
  PassesThroughSecondAndFourthQuadrants (fun x ↦ (m - 2) * x^(m^2 - 8)) →
  m = -3 := by
  sorry

end value_of_m_l2083_208379


namespace trig_identity_proof_l2083_208362

theorem trig_identity_proof (θ : ℝ) : 
  Real.sin (θ + Real.pi / 180 * 75) + Real.cos (θ + Real.pi / 180 * 45) - Real.sqrt 3 * Real.cos (θ + Real.pi / 180 * 15) = 0 := by
  sorry

end trig_identity_proof_l2083_208362


namespace brandys_trail_mix_raisins_l2083_208381

/-- The weight of raisins in a trail mix -/
def weight_of_raisins (weight_of_peanuts weight_of_chips total_weight : Real) : Real :=
  total_weight - (weight_of_peanuts + weight_of_chips)

/-- Theorem stating the weight of raisins in Brandy's trail mix -/
theorem brandys_trail_mix_raisins : 
  weight_of_raisins 0.17 0.17 0.42 = 0.08 := by
  sorry

end brandys_trail_mix_raisins_l2083_208381


namespace population_average_age_l2083_208313

theorem population_average_age 
  (k : ℕ) 
  (h_k_pos : k > 0) 
  (men_count : ℕ := 7 * k)
  (women_count : ℕ := 8 * k)
  (men_avg_age : ℚ := 36)
  (women_avg_age : ℚ := 30) :
  let total_population := men_count + women_count
  let total_age := men_count * men_avg_age + women_count * women_avg_age
  total_age / total_population = 164 / 5 := by
sorry

#eval (164 : ℚ) / 5  -- Should evaluate to 32.8

end population_average_age_l2083_208313


namespace smallest_sum_of_perfect_squares_l2083_208353

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 145 → (∀ a b : ℕ, a^2 - b^2 = 145 → x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 433 := by
  sorry

end smallest_sum_of_perfect_squares_l2083_208353


namespace tile_difference_l2083_208387

theorem tile_difference (initial_blue : ℕ) (initial_green : ℕ) (border_tiles : ℕ) :
  initial_blue = 15 →
  initial_green = 8 →
  border_tiles = 12 →
  (initial_blue + border_tiles / 2) - (initial_green + border_tiles / 2) = 7 :=
by sorry

end tile_difference_l2083_208387


namespace necessary_but_not_sufficient_condition_l2083_208354

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x > y - 2) ∧
  ¬(∀ x y : ℝ, x > y - 2 → x > y) :=
by sorry

end necessary_but_not_sufficient_condition_l2083_208354


namespace fruit_bowl_oranges_l2083_208338

theorem fruit_bowl_oranges (bananas apples oranges : ℕ) : 
  bananas = 2 → 
  apples = 2 * bananas → 
  bananas + apples + oranges = 12 → 
  oranges = 6 := by
sorry

end fruit_bowl_oranges_l2083_208338


namespace millet_majority_on_wednesday_l2083_208323

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Rat
  other_seeds : Rat

/-- Calculates the next day's feeder state -/
def next_day (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := state.millet * (4/5),
    other_seeds := 0 }

/-- Adds new seeds to the feeder (every other day) -/
def add_seeds (state : FeederState) : FeederState :=
  { day := state.day,
    millet := state.millet + 2/5,
    other_seeds := state.other_seeds + 3/5 }

/-- Initial state of the feeder on Monday -/
def initial_state : FeederState :=
  { day := 1, millet := 2/5, other_seeds := 3/5 }

/-- Theorem: On Wednesday (day 3), millet is more than half of total seeds -/
theorem millet_majority_on_wednesday :
  let state_wednesday := add_seeds (next_day (next_day initial_state))
  state_wednesday.millet > (state_wednesday.millet + state_wednesday.other_seeds) / 2 := by
  sorry


end millet_majority_on_wednesday_l2083_208323


namespace coin_arrangement_count_l2083_208302

def coin_diameter_10_filler : ℕ := 19
def coin_diameter_50_filler : ℕ := 22
def total_length : ℕ := 1000
def min_coins : ℕ := 50

theorem coin_arrangement_count : 
  ∃ (x y : ℕ), 
    x * coin_diameter_10_filler + y * coin_diameter_50_filler = total_length ∧ 
    x + y ≥ min_coins ∧
    Nat.choose (x + y) y = 270725 := by
  sorry

end coin_arrangement_count_l2083_208302


namespace no_solution_when_m_is_seven_l2083_208386

theorem no_solution_when_m_is_seven :
  ∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) ≠ (x - 7) / (x - 8) :=
by
  sorry

end no_solution_when_m_is_seven_l2083_208386


namespace largest_prime_factors_difference_l2083_208331

-- Define the number we're working with
def n : ℕ := 175616

-- State the theorem
theorem largest_prime_factors_difference (p q : ℕ) : 
  (Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n ∧ 
   ∀ r, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) → 
  p - q = 5 ∨ q - p = 5 :=
sorry

end largest_prime_factors_difference_l2083_208331


namespace pizza_group_composition_l2083_208382

theorem pizza_group_composition :
  ∀ (boys girls : ℕ),
  (∀ (b : ℕ), b ≤ boys → 6 ≤ b ∧ b ≤ 7) →
  (∀ (g : ℕ), g ≤ girls → 2 ≤ g ∧ g ≤ 3) →
  49 ≤ 6 * boys + 2 * girls →
  7 * boys + 3 * girls ≤ 59 →
  boys = 8 ∧ girls = 2 :=
by
  sorry

end pizza_group_composition_l2083_208382


namespace trigonometric_identities_l2083_208332

open Real

theorem trigonometric_identities (α : ℝ) :
  (tan α = 1 / 3) →
  (1 / (2 * sin α * cos α + cos α ^ 2) = 2 / 3) ∧
  (tan (π - α) * cos (2 * π - α) * sin (-α + 3 * π / 2)) / (cos (-α - π) * sin (-π - α)) = -1 :=
by sorry

end trigonometric_identities_l2083_208332


namespace stewart_farm_sheep_count_l2083_208389

def sheep_horse_ratio : ℚ := 2 / 7
def horse_food_per_day : ℕ := 230
def total_horse_food : ℕ := 12880

theorem stewart_farm_sheep_count :
  ∃ (sheep horses : ℕ),
    sheep / horses = sheep_horse_ratio ∧
    horses * horse_food_per_day = total_horse_food ∧
    sheep = 16 := by sorry

end stewart_farm_sheep_count_l2083_208389


namespace thursday_seeds_count_l2083_208390

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := 20

/-- The total number of seeds planted -/
def total_seeds : ℕ := 22

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := total_seeds - seeds_wednesday

theorem thursday_seeds_count : seeds_thursday = 2 := by
  sorry

end thursday_seeds_count_l2083_208390


namespace circle_covering_theorem_l2083_208307

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a function to check if a point is inside or on a circle
def pointInCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

-- Main theorem
theorem circle_covering_theorem 
  (points : Finset Point) 
  (outer_circle : Circle) 
  (h1 : outer_circle.radius = 2)
  (h2 : points.card = 15)
  (h3 : ∀ p ∈ points, pointInCircle p outer_circle) :
  ∃ (inner_circle : Circle), 
    inner_circle.radius = 1 ∧ 
    (∃ (subset : Finset Point), subset ⊆ points ∧ subset.card ≥ 3 ∧ 
      ∀ p ∈ subset, pointInCircle p inner_circle) := by
  sorry

end circle_covering_theorem_l2083_208307


namespace problem_statement_l2083_208356

theorem problem_statement (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 3) 
  (h3 : x = 1) : 
  y = 8 := by
sorry

end problem_statement_l2083_208356


namespace f_of_f_3_l2083_208376

def f (x : ℝ) : ℝ := 3 * x^2 + 3 * x - 2

theorem f_of_f_3 : f (f 3) = 3568 := by
  sorry

end f_of_f_3_l2083_208376


namespace hannah_lost_eight_pieces_l2083_208320

-- Define the initial state of the chess game
def initial_pieces : ℕ := 32
def initial_pieces_per_player : ℕ := 16

-- Define the given conditions
def scarlett_lost : ℕ := 6
def total_pieces_left : ℕ := 18

-- Define Hannah's lost pieces
def hannah_lost : ℕ := initial_pieces_per_player - (total_pieces_left - (initial_pieces_per_player - scarlett_lost))

-- Theorem to prove
theorem hannah_lost_eight_pieces : hannah_lost = 8 := by
  sorry

end hannah_lost_eight_pieces_l2083_208320


namespace smallest_number_divisibility_l2083_208305

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬(is_divisible (y + 3) 18 ∧ is_divisible (y + 3) 70 ∧ 
                     is_divisible (y + 3) 25 ∧ is_divisible (y + 3) 21)) ∧
  (is_divisible (x + 3) 18 ∧ is_divisible (x + 3) 70 ∧ 
   is_divisible (x + 3) 25 ∧ is_divisible (x + 3) 21) ∧
  x = 3147 :=
sorry

end smallest_number_divisibility_l2083_208305


namespace incorrect_calculation_correction_l2083_208372

theorem incorrect_calculation_correction (x : ℝ) : 
  25 * x = 812 → x / 4 = 8.12 := by
  sorry

end incorrect_calculation_correction_l2083_208372


namespace range_of_g_l2083_208312

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 5

-- Define the function g as a composition of f
def g (x : ℝ) : ℝ := f (f (f x))

-- Theorem statement
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ g x = y) ↔ -41 ≤ y ∧ y ≤ 87 :=
sorry

end range_of_g_l2083_208312


namespace intersected_cubes_count_l2083_208311

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure LargeCube where
  size : ℕ
  size_eq : size = 4

/-- Represents a plane intersecting the large cube -/
structure IntersectingPlane where
  cube : LargeCube
  ratio : ℚ
  ratio_eq : ratio = 1 / 3

/-- Counts the number of unit cubes intersected by the plane -/
def count_intersected_cubes (plane : IntersectingPlane) : ℕ := sorry

/-- Theorem stating that the plane intersects 32 unit cubes -/
theorem intersected_cubes_count (plane : IntersectingPlane) : 
  count_intersected_cubes plane = 32 := by sorry

end intersected_cubes_count_l2083_208311


namespace solution_set_f_leq_x_max_value_f_min_value_ab_l2083_208364

-- Define the function f
def f (x : ℝ) : ℝ := |x + 5| - |x - 1|

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | -6 ≤ x ∧ x ≤ -4 ∨ x ≥ 6} :=
sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : 
  ∀ x : ℝ, f x ≤ 6 :=
sorry

-- Theorem for the minimum value of ab
theorem min_value_ab (a b : ℝ) (h : Real.log a + Real.log (2 * b) = Real.log (a + 4 * b + 6)) :
  a * b ≥ 9 :=
sorry

end solution_set_f_leq_x_max_value_f_min_value_ab_l2083_208364


namespace dvd_packs_after_discount_l2083_208314

theorem dvd_packs_after_discount (original_price discount available : ℕ) : 
  original_price = 107 → 
  discount = 106 → 
  available = 93 → 
  (available / (original_price - discount) : ℕ) = 93 := by
sorry

end dvd_packs_after_discount_l2083_208314


namespace alcohol_water_mixture_ratio_l2083_208380

/-- Given two jars of alcohol-water mixtures with volumes V and 2V, and ratios p:1 and q:1 respectively,
    the ratio of alcohol to water in the resulting mixture is (p(q+1) + 2p + 2q) : (q+1 + 2p + 2) -/
theorem alcohol_water_mixture_ratio (V p q : ℝ) (hV : V > 0) (hp : p > 0) (hq : q > 0) :
  let first_jar_alcohol := (p / (p + 1)) * V
  let first_jar_water := (1 / (p + 1)) * V
  let second_jar_alcohol := (2 * q / (q + 1)) * V
  let second_jar_water := (2 / (q + 1)) * V
  let total_alcohol := first_jar_alcohol + second_jar_alcohol
  let total_water := first_jar_water + second_jar_water
  total_alcohol / total_water = (p * (q + 1) + 2 * p + 2 * q) / (q + 1 + 2 * p + 2) :=
by sorry

end alcohol_water_mixture_ratio_l2083_208380


namespace ace_king_queen_probability_l2083_208383

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of each face card (Ace, King, Queen)
def faceCards : ℕ := 4

-- Define the probability of drawing the sequence (Ace, King, Queen)
def probAceKingQueen : ℚ := (faceCards : ℚ) / totalCards *
                            (faceCards : ℚ) / (totalCards - 1) *
                            (faceCards : ℚ) / (totalCards - 2)

-- Theorem statement
theorem ace_king_queen_probability :
  probAceKingQueen = 8 / 16575 := by sorry

end ace_king_queen_probability_l2083_208383


namespace king_crown_cost_l2083_208315

/-- Calculates the total cost of a purchase with a tip -/
def totalCostWithTip (originalCost tipPercentage : ℚ) : ℚ :=
  originalCost * (1 + tipPercentage / 100)

/-- Proves that the king pays $22,000 for a $20,000 crown with a 10% tip -/
theorem king_crown_cost :
  totalCostWithTip 20000 10 = 22000 := by
  sorry

end king_crown_cost_l2083_208315


namespace two_roots_implies_c_values_l2083_208377

-- Define the function f(x) = x³ - 3x + c
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

-- State the theorem
theorem two_roots_implies_c_values (c : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧
    (∀ x : ℝ, f c x = 0 → x = x₁ ∨ x = x₂)) →
  c = -2 ∨ c = 2 := by
sorry

end two_roots_implies_c_values_l2083_208377


namespace trapezoid_area_is_15_l2083_208348

/-- A trapezoid bounded by y = 2x, y = 8, y = 2, and the y-axis -/
structure Trapezoid where
  /-- The line y = 2x -/
  line_1 : ℝ → ℝ := λ x => 2 * x
  /-- The line y = 8 -/
  line_2 : ℝ → ℝ := λ _ => 8
  /-- The line y = 2 -/
  line_3 : ℝ → ℝ := λ _ => 2
  /-- The y-axis (x = 0) -/
  y_axis : ℝ → ℝ := λ y => 0

/-- The area of the trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  15

/-- Theorem stating that the area of the given trapezoid is 15 square units -/
theorem trapezoid_area_is_15 (t : Trapezoid) : trapezoidArea t = 15 := by
  sorry

end trapezoid_area_is_15_l2083_208348


namespace F_difference_l2083_208358

/-- Represents the infinite repeating decimal 0.726726726... -/
def F : ℚ := 726 / 999

/-- The fraction representation of F in lowest terms -/
def F_reduced : ℚ := 242 / 333

theorem F_difference : (F_reduced.den : ℤ) - (F_reduced.num : ℤ) = 91 := by sorry

end F_difference_l2083_208358


namespace clock_angle_at_13_20_clock_angle_at_13_20_is_80_l2083_208363

/-- The angle between the hour and minute hands of a clock at 13:20 (1:20 PM) --/
theorem clock_angle_at_13_20 : ℝ :=
  let hour := 1
  let minute := 20
  let degrees_per_hour := 360 / 12
  let degrees_per_minute := 360 / 60
  let hour_hand_angle := hour * degrees_per_hour + (minute / 60) * degrees_per_hour
  let minute_hand_angle := minute * degrees_per_minute
  |minute_hand_angle - hour_hand_angle|

/-- The angle between the hour and minute hands of a clock at 13:20 (1:20 PM) is 80 degrees --/
theorem clock_angle_at_13_20_is_80 : clock_angle_at_13_20 = 80 := by
  sorry

end clock_angle_at_13_20_clock_angle_at_13_20_is_80_l2083_208363


namespace cubic_function_property_l2083_208397

/-- Given a cubic function f(x) = ax³ + bx - 2 where f(2014) = 3, prove that f(-2014) = -7 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 2
  (f 2014 = 3) → (f (-2014) = -7) := by
sorry

end cubic_function_property_l2083_208397


namespace max_distinct_pairs_l2083_208319

theorem max_distinct_pairs (k : ℕ) 
  (a b : Fin k → ℕ)
  (h_range : ∀ i : Fin k, 1 ≤ a i ∧ a i < b i ∧ b i ≤ 150)
  (h_distinct : ∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)
  (h_sum_distinct : ∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j)
  (h_sum_bound : ∀ i : Fin k, a i + b i ≤ 150) :
  k ≤ 59 :=
sorry

end max_distinct_pairs_l2083_208319


namespace box_tape_theorem_l2083_208309

theorem box_tape_theorem (L S : ℝ) (h1 : L > 0) (h2 : S > 0) :
  5 * (L + 2 * S) + 240 = 540 → S = (60 - L) / 2 := by
  sorry

end box_tape_theorem_l2083_208309


namespace intersection_of_A_and_B_l2083_208365

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l2083_208365


namespace f_of_f_of_one_eq_31_l2083_208350

def f (x : ℝ) : ℝ := 4 * x^3 + 2 * x^2 - 5 * x + 1

theorem f_of_f_of_one_eq_31 : f (f 1) = 31 := by
  sorry

end f_of_f_of_one_eq_31_l2083_208350


namespace car_overtake_distance_l2083_208327

/-- Proves that the initial distance between two cars is equal to the product of their relative speed and the overtaking time. -/
theorem car_overtake_distance (v_red v_black : ℝ) (t : ℝ) (h1 : 0 < v_red) (h2 : v_red < v_black) (h3 : 0 < t) :
  (v_black - v_red) * t = (v_black - v_red) * t :=
by sorry

/-- Calculates the initial distance between two cars given their speeds and overtaking time. -/
def initial_distance (v_red v_black t : ℝ) : ℝ :=
  (v_black - v_red) * t

#check car_overtake_distance
#check initial_distance

end car_overtake_distance_l2083_208327


namespace jimmy_can_lose_five_more_points_l2083_208308

def passing_score : ℕ := 50
def exams_count : ℕ := 3
def points_per_exam : ℕ := 20
def points_lost : ℕ := 5

def max_additional_points_to_lose : ℕ :=
  exams_count * points_per_exam - points_lost - passing_score

theorem jimmy_can_lose_five_more_points :
  max_additional_points_to_lose = 5 := by
  sorry

end jimmy_can_lose_five_more_points_l2083_208308


namespace equilateral_triangle_side_length_l2083_208325

/-- The side length of an equilateral triangle with perimeter 2 meters is 2/3 meters. -/
theorem equilateral_triangle_side_length : 
  ∀ (side_length : ℝ), 
    (side_length > 0) →
    (3 * side_length = 2) →
    side_length = 2 / 3 := by
  sorry

end equilateral_triangle_side_length_l2083_208325


namespace system_of_equations_solutions_l2083_208394

theorem system_of_equations_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (solutions.card = 8) ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔
      (x = 2 * y^2 - 1 ∧ y = 2 * z^2 - 1 ∧ z = 2 * x^2 - 1)) :=
by sorry


end system_of_equations_solutions_l2083_208394


namespace center_cell_value_l2083_208361

theorem center_cell_value (a b c d e f g h i : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0) →
  (a * b * c = 1) →
  (d * e * f = 1) →
  (g * h * i = 1) →
  (a * d * g = 1) →
  (b * e * h = 1) →
  (c * f * i = 1) →
  (a * b * d * e = 2) →
  (b * c * e * f = 2) →
  (d * e * g * h = 2) →
  (e * f * h * i = 2) →
  e = 1 :=
by sorry

end center_cell_value_l2083_208361


namespace binomial_divisibility_l2083_208378

theorem binomial_divisibility (p k : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ m : ℤ, (p : ℤ)^3 * m = (Nat.choose (k * p) p : ℤ) - k := by
  sorry

end binomial_divisibility_l2083_208378


namespace candy_theorem_l2083_208388

/-- The total number of candy pieces caught by four friends -/
def total_candy (tabitha stan julie carlos : ℕ) : ℕ :=
  tabitha + stan + julie + carlos

/-- Theorem: Given the conditions, the friends caught 72 pieces of candy in total -/
theorem candy_theorem (tabitha stan julie carlos : ℕ) 
  (h1 : tabitha = 22)
  (h2 : stan = 13)
  (h3 : julie = tabitha / 2)
  (h4 : carlos = 2 * stan) :
  total_candy tabitha stan julie carlos = 72 := by
  sorry

end candy_theorem_l2083_208388


namespace jade_transactions_l2083_208374

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 19 →
  jade = 85 := by
  sorry

end jade_transactions_l2083_208374


namespace clothing_store_profit_l2083_208344

/-- Represents the daily profit function for a clothing store -/
def daily_profit (cost : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_price - price_reduction - cost) * (initial_sales + 2 * price_reduction)

theorem clothing_store_profit 
  (cost : ℝ) (initial_price : ℝ) (initial_sales : ℝ)
  (h_cost : cost = 50)
  (h_initial_price : initial_price = 90)
  (h_initial_sales : initial_sales = 20) :
  (∃ (x : ℝ), daily_profit cost initial_price initial_sales x = 1200) ∧
  (¬ ∃ (y : ℝ), daily_profit cost initial_price initial_sales y = 2000) := by
  sorry

#check clothing_store_profit

end clothing_store_profit_l2083_208344


namespace positive_sum_product_iff_l2083_208340

theorem positive_sum_product_iff (a b : ℝ) : (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end positive_sum_product_iff_l2083_208340


namespace B_equals_C_l2083_208347

def A : Set Int := {-1, 1}

def B : Set Int := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x + y}

def C : Set Int := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x - y}

theorem B_equals_C : B = C := by sorry

end B_equals_C_l2083_208347


namespace train_overtake_l2083_208310

/-- Proves that Train B overtakes Train A at 285 miles from the station -/
theorem train_overtake (speed_A speed_B : ℝ) (time_diff : ℝ) : 
  speed_A = 30 →
  speed_B = 38 →
  time_diff = 2 →
  speed_B > speed_A →
  (speed_A * time_diff + speed_A * ((speed_B * time_diff) / (speed_B - speed_A))) = 285 := by
  sorry

#check train_overtake

end train_overtake_l2083_208310


namespace sqrt_meaningful_range_l2083_208306

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 3) ↔ x ≥ 3/2 := by sorry

end sqrt_meaningful_range_l2083_208306


namespace three_divisors_iff_prime_square_l2083_208357

/-- A natural number has exactly three distinct divisors if and only if it is the square of a prime number. -/
theorem three_divisors_iff_prime_square (n : ℕ) : (∃! (s : Finset ℕ), s.card = 3 ∧ ∀ d ∈ s, d ∣ n) ↔ ∃ p, Nat.Prime p ∧ n = p^2 := by
  sorry

end three_divisors_iff_prime_square_l2083_208357


namespace circle_center_sum_l2083_208396

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 10*x + 4*y + 9

/-- The center of a circle given its equation -/
def CircleCenter (eq : (ℝ → ℝ → Prop)) : ℝ × ℝ :=
  sorry

/-- The sum of coordinates of a point -/
def SumOfCoordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

theorem circle_center_sum :
  SumOfCoordinates (CircleCenter CircleEquation) = 7 := by
  sorry

end circle_center_sum_l2083_208396


namespace d11d_divisible_by_5_l2083_208370

/-- Represents a base-7 digit -/
def Base7Digit := {d : ℕ // d < 7}

/-- Converts a base-7 number of the form d11d to its decimal equivalent -/
def toDecimal (d : Base7Digit) : ℕ := 344 * d.val + 56

/-- A base-7 number d11d_7 is divisible by 5 if and only if d = 1 -/
theorem d11d_divisible_by_5 (d : Base7Digit) : 
  5 ∣ toDecimal d ↔ d.val = 1 := by sorry

end d11d_divisible_by_5_l2083_208370


namespace chess_match_draw_probability_l2083_208366

theorem chess_match_draw_probability (john_win_prob mike_win_prob : ℚ) 
  (h1 : john_win_prob = 4/9)
  (h2 : mike_win_prob = 5/18) : 
  1 - (john_win_prob + mike_win_prob) = 5/18 := by
  sorry

end chess_match_draw_probability_l2083_208366


namespace basketball_competition_equation_l2083_208369

/-- Represents the number of matches in a basketball competition where each pair of classes plays once --/
def number_of_matches (x : ℕ) : ℕ := x * (x - 1) / 2

/-- Theorem stating that for 10 total matches, the equation x(x-1)/2 = 10 correctly represents the situation --/
theorem basketball_competition_equation (x : ℕ) (h : number_of_matches x = 10) : 
  x * (x - 1) / 2 = 10 := by
  sorry

end basketball_competition_equation_l2083_208369


namespace right_triangle_hypotenuse_l2083_208322

theorem right_triangle_hypotenuse (longer_leg shorter_leg hypotenuse : ℝ) : 
  shorter_leg = longer_leg - 3 →
  (1 / 2) * longer_leg * shorter_leg = 120 →
  longer_leg > 0 →
  shorter_leg > 0 →
  hypotenuse^2 = longer_leg^2 + shorter_leg^2 →
  hypotenuse = Real.sqrt 425 := by
sorry

end right_triangle_hypotenuse_l2083_208322


namespace hyperbola_C_properties_l2083_208317

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2/3 - y^2/12 = 1

-- Define the reference hyperbola
def ref_hyperbola (x y : ℝ) : Prop := y^2/4 - x^2 = 1

-- Theorem statement
theorem hyperbola_C_properties :
  -- C passes through (2,2)
  C 2 2 ∧
  -- C has the same asymptotes as the reference hyperbola
  (∀ x y : ℝ, C x y ↔ ∃ k : ℝ, k ≠ 0 ∧ ref_hyperbola (x/k) (y/k)) :=
sorry

end hyperbola_C_properties_l2083_208317


namespace power_of_power_l2083_208321

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l2083_208321


namespace uniform_profit_percentage_clock_sales_l2083_208316

/-- Uniform profit percentage calculation for clock sales --/
theorem uniform_profit_percentage_clock_sales
  (total_clocks : ℕ)
  (clocks_10_percent : ℕ)
  (clocks_20_percent : ℕ)
  (cost_price : ℚ)
  (revenue_difference : ℚ)
  (h1 : total_clocks = clocks_10_percent + clocks_20_percent)
  (h2 : total_clocks = 90)
  (h3 : clocks_10_percent = 40)
  (h4 : clocks_20_percent = 50)
  (h5 : cost_price = 79.99999999999773)
  (h6 : revenue_difference = 40) :
  let actual_revenue := clocks_10_percent * (cost_price * (1 + 10 / 100)) +
                        clocks_20_percent * (cost_price * (1 + 20 / 100))
  let uniform_revenue := actual_revenue - revenue_difference
  let uniform_profit_percentage := (uniform_revenue / (total_clocks * cost_price) - 1) * 100
  uniform_profit_percentage = 15 :=
sorry

end uniform_profit_percentage_clock_sales_l2083_208316


namespace original_price_correct_l2083_208341

/-- The original price of a shirt before discount -/
def original_price : ℝ := 975

/-- The discount percentage applied to the shirt -/
def discount_percentage : ℝ := 0.20

/-- The discounted price of the shirt -/
def discounted_price : ℝ := 780

/-- Theorem stating that the original price is correct given the discount and discounted price -/
theorem original_price_correct : 
  original_price * (1 - discount_percentage) = discounted_price :=
by sorry

end original_price_correct_l2083_208341


namespace parabola_equation_l2083_208339

/-- A parabola with focus (0,1) and vertex at (0,0) has the standard equation x^2 = 4y -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (0, 1)
  let vertex : ℝ × ℝ := (0, 0)
  let p : ℝ := focus.2 - vertex.2
  (x^2 = 4*y) ↔ (
    (∀ (x' y' : ℝ), (x' - focus.1)^2 + (y' - focus.2)^2 = (y' - (focus.2 - p))^2) ∧
    vertex = (0, 0) ∧
    focus = (0, 1)
  ) := by sorry

end parabola_equation_l2083_208339


namespace line_equation_through_point_parallel_to_vector_l2083_208398

/-- The equation of a line passing through point P(-1, 2) and parallel to vector {8, -4} --/
theorem line_equation_through_point_parallel_to_vector :
  let P : ℝ × ℝ := (-1, 2)
  let a : ℝ × ℝ := (8, -4)
  let line_eq (x y : ℝ) := y = -1/2 * x + 3/2
  (∀ x y : ℝ, line_eq x y ↔ 
    (∃ t : ℝ, x = P.1 + t * a.1 ∧ y = P.2 + t * a.2)) :=
by sorry

end line_equation_through_point_parallel_to_vector_l2083_208398


namespace rectangle_area_l2083_208349

theorem rectangle_area (length width area : ℝ) : 
  length = 24 →
  width = 0.875 * length →
  area = length * width →
  area = 504 := by
sorry

end rectangle_area_l2083_208349


namespace sharks_score_l2083_208336

theorem sharks_score (total_points eagles_points sharks_points : ℕ) : 
  total_points = 60 → 
  eagles_points = sharks_points + 18 → 
  eagles_points + sharks_points = total_points → 
  sharks_points = 21 := by
sorry

end sharks_score_l2083_208336


namespace amy_music_files_l2083_208300

/-- Represents the number of music files Amy initially had -/
def initial_music_files : ℕ := sorry

/-- Represents the initial total number of files -/
def initial_total_files : ℕ := initial_music_files + 36

/-- Represents the number of deleted files -/
def deleted_files : ℕ := 48

/-- Represents the number of remaining files after deletion -/
def remaining_files : ℕ := 14

theorem amy_music_files :
  initial_total_files - deleted_files = remaining_files ∧
  initial_music_files = 26 := by
  sorry

end amy_music_files_l2083_208300


namespace discount_percentage_calculation_l2083_208373

def selling_price : ℝ := 24000
def cost_price : ℝ := 20000
def potential_profit_percentage : ℝ := 8

theorem discount_percentage_calculation :
  let potential_profit := (potential_profit_percentage / 100) * cost_price
  let selling_price_with_potential_profit := cost_price + potential_profit
  let discount_amount := selling_price - selling_price_with_potential_profit
  let discount_percentage := (discount_amount / selling_price) * 100
  discount_percentage = 10 := by sorry

end discount_percentage_calculation_l2083_208373


namespace figure_404_has_2022_squares_l2083_208345

/-- The number of squares in the nth figure of the sequence -/
def squares_in_figure (n : ℕ) : ℕ := 7 + (n - 1) * 5

theorem figure_404_has_2022_squares :
  squares_in_figure 404 = 2022 := by
  sorry

end figure_404_has_2022_squares_l2083_208345


namespace imaginary_part_of_z_l2083_208343

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 + Complex.I)) :
  z.im = Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_l2083_208343


namespace fred_found_43_seashells_l2083_208333

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The total number of seashells found by Tom and Fred -/
def total_seashells : ℕ := 58

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := total_seashells - tom_seashells

theorem fred_found_43_seashells : fred_seashells = 43 := by
  sorry

end fred_found_43_seashells_l2083_208333


namespace inscribed_angle_theorem_l2083_208352

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arc on a circle -/
structure Arc (c : Circle) where
  start_point : ℝ × ℝ
  end_point : ℝ × ℝ

/-- The angle subtended by an arc at the center of the circle -/
def central_angle (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- The angle subtended by an arc at a point on the circumference of the circle -/
def inscribed_angle (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- The Inscribed Angle Theorem -/
theorem inscribed_angle_theorem (c : Circle) (a : Arc c) :
  inscribed_angle c a = (1 / 2) * central_angle c a :=
sorry

end inscribed_angle_theorem_l2083_208352


namespace probability_of_black_ball_l2083_208360

theorem probability_of_black_ball
  (p_red : ℝ) (p_white : ℝ) (p_black : ℝ)
  (h1 : p_red = 0.52)
  (h2 : p_white = 0.28)
  (h3 : p_red + p_white + p_black = 1) :
  p_black = 0.2 := by
sorry

end probability_of_black_ball_l2083_208360


namespace sum_of_fractions_equals_one_l2083_208318

theorem sum_of_fractions_equals_one
  (a b c d w x y z : ℝ)
  (eq1 : 17*w + b*x + c*y + d*z = 0)
  (eq2 : a*w + 29*x + c*y + d*z = 0)
  (eq3 : a*w + b*x + 37*y + d*z = 0)
  (eq4 : a*w + b*x + c*y + 53*z = 0)
  (ha : a ≠ 17)
  (hb : b ≠ 29)
  (hc : c ≠ 37)
  (h_not_all_zero : ¬(w = 0 ∧ x = 0 ∧ y = 0)) :
  a / (a - 17) + b / (b - 29) + c / (c - 37) + d / (d - 53) = 1 :=
by sorry

end sum_of_fractions_equals_one_l2083_208318


namespace sugar_recipes_l2083_208385

/-- The number of full recipes that can be made with a given amount of sugar -/
def full_recipes (total_sugar : ℚ) (sugar_per_recipe : ℚ) : ℚ :=
  total_sugar / sugar_per_recipe

/-- Theorem: Given 47 2/3 cups of sugar and a recipe requiring 1 1/2 cups of sugar,
    the number of full recipes that can be made is 31 7/9 -/
theorem sugar_recipes :
  let total_sugar : ℚ := 47 + 2/3
  let sugar_per_recipe : ℚ := 1 + 1/2
  full_recipes total_sugar sugar_per_recipe = 31 + 7/9 := by
sorry

end sugar_recipes_l2083_208385


namespace sum_of_roots_l2083_208328

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*|x + 4| - 27

theorem sum_of_roots : ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 6 - Real.sqrt 20 := by
  sorry

end sum_of_roots_l2083_208328


namespace sequence_sum_l2083_208351

theorem sequence_sum (n : ℕ) (x : ℕ → ℝ) (h1 : x 1 = 3) 
  (h2 : ∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + k) : 
  Finset.sum (Finset.range n) (λ k => x (k + 1)) = 3*n + (n*(n+1)*(2*n-1))/12 := by
  sorry

end sequence_sum_l2083_208351


namespace quadratic_polynomial_inequality_l2083_208392

def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_polynomial_inequality 
  (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (x y : ℝ) : 
  (quadratic_polynomial a b c (x * y))^2 ≤ 
  (quadratic_polynomial a b c (x^2)) * (quadratic_polynomial a b c (y^2)) := by
  sorry

end quadratic_polynomial_inequality_l2083_208392


namespace angle_properties_l2083_208326

/-- Given a point P on the unit circle, determine the quadrant and smallest positive angle -/
theorem angle_properties (α : Real) : 
  (∃ P : Real × Real, P.1 = Real.sin (5 * Real.pi / 6) ∧ P.2 = Real.cos (5 * Real.pi / 6) ∧ 
   P.1 = Real.sin α ∧ P.2 = Real.cos α) →
  (α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) ∧
  (∃ β : Real, β = 5 * Real.pi / 3 ∧ 
   Real.sin β = Real.sin α ∧ Real.cos β = Real.cos α ∧
   ∀ γ : Real, 0 < γ ∧ γ < β → 
   Real.sin γ ≠ Real.sin α ∨ Real.cos γ ≠ Real.cos α) :=
by sorry

end angle_properties_l2083_208326


namespace fraction_value_l2083_208395

theorem fraction_value : (3100 - 3037)^2 / 81 = 49 := by
  sorry

end fraction_value_l2083_208395


namespace floor_sqrt_18_squared_l2083_208391

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by sorry

end floor_sqrt_18_squared_l2083_208391


namespace m_greater_than_n_l2083_208301

theorem m_greater_than_n (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  a * b > a + b - 1 := by
  sorry

end m_greater_than_n_l2083_208301


namespace interest_calculation_period_l2083_208359

theorem interest_calculation_period (P n : ℝ) 
  (h1 : P * n / 20 = 40)
  (h2 : P * ((1 + 0.05)^n - 1) = 41) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |n - 1| < ε :=
sorry

end interest_calculation_period_l2083_208359


namespace average_weight_of_children_l2083_208303

theorem average_weight_of_children (num_boys num_girls : ℕ) (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 8 →
  num_girls = 6 →
  avg_weight_boys = 160 →
  avg_weight_girls = 130 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 147 :=
by sorry

end average_weight_of_children_l2083_208303


namespace water_evaporation_period_l2083_208329

theorem water_evaporation_period (initial_water : ℝ) (daily_evaporation : ℝ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  daily_evaporation = 0.012 →
  evaporation_percentage = 0.06 →
  (initial_water * evaporation_percentage) / daily_evaporation = 50 :=
by sorry

end water_evaporation_period_l2083_208329


namespace unique_denomination_l2083_208335

/-- Given unlimited supply of stamps of denominations 4, n, and n+1 cents,
    57 cents is the greatest postage that cannot be formed -/
def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 57 → ∃ a b c : ℕ, k = 4*a + n*b + (n+1)*c

/-- 21 is the only positive integer satisfying the condition -/
theorem unique_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n :=
sorry

end unique_denomination_l2083_208335


namespace rectangular_window_area_l2083_208375

/-- The area of a rectangular window with length 47.3 cm and width 24 cm is 1135.2 cm². -/
theorem rectangular_window_area : 
  let length : ℝ := 47.3
  let width : ℝ := 24
  let area : ℝ := length * width
  area = 1135.2 := by sorry

end rectangular_window_area_l2083_208375


namespace mike_yard_sale_books_l2083_208371

/-- Calculates the number of books bought at a yard sale -/
def books_bought_at_yard_sale (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem: The number of books Mike bought at the yard sale is 21 -/
theorem mike_yard_sale_books :
  let initial_books : ℕ := 35
  let final_books : ℕ := 56
  books_bought_at_yard_sale initial_books final_books = 21 := by
  sorry

end mike_yard_sale_books_l2083_208371


namespace pages_copied_for_35_dollars_l2083_208399

-- Define the cost per 3 pages in cents
def cost_per_3_pages : ℚ := 7

-- Define the budget in dollars
def budget : ℚ := 35

-- Define the function to calculate the number of pages
def pages_copied (cost_per_3_pages budget : ℚ) : ℚ :=
  (budget * 100) * (3 / cost_per_3_pages)

-- Theorem statement
theorem pages_copied_for_35_dollars :
  pages_copied cost_per_3_pages budget = 1500 := by
  sorry

end pages_copied_for_35_dollars_l2083_208399


namespace ceiling_negative_three_point_eight_l2083_208384

theorem ceiling_negative_three_point_eight :
  ⌈(-3.8 : ℝ)⌉ = -3 := by sorry

end ceiling_negative_three_point_eight_l2083_208384
