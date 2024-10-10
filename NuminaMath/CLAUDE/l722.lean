import Mathlib

namespace integer_solution_zero_l722_72297

theorem integer_solution_zero (x y z t : ℤ) : 
  x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
sorry

end integer_solution_zero_l722_72297


namespace k_minus_one_not_square_k_plus_one_not_square_l722_72228

/-- k is the product of several of the first prime numbers -/
def k : ℕ := sorry

/-- k is the product of at least two prime numbers -/
axiom k_def : ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p < q ∧ k = p * q

/-- k-1 is not a perfect square -/
theorem k_minus_one_not_square : ¬∃ (n : ℕ), n^2 = k - 1 := by sorry

/-- k+1 is not a perfect square -/
theorem k_plus_one_not_square : ¬∃ (n : ℕ), n^2 = k + 1 := by sorry

end k_minus_one_not_square_k_plus_one_not_square_l722_72228


namespace fraction_of_books_sold_l722_72294

/-- Given a collection of books where some were sold and some remained unsold,
    this theorem proves that the fraction of books sold is 2/3 under specific conditions. -/
theorem fraction_of_books_sold (total_books : ℕ) (sold_books : ℕ) : 
  (total_books > 50) →
  (sold_books = total_books - 50) →
  (sold_books * 5 = 500) →
  (sold_books : ℚ) / total_books = 2 / 3 := by
  sorry

end fraction_of_books_sold_l722_72294


namespace equation_with_parentheses_is_true_l722_72245

theorem equation_with_parentheses_is_true : 7 * 9 + 12 / (3 - 2) = 75 := by
  sorry

end equation_with_parentheses_is_true_l722_72245


namespace inequality_equivalence_l722_72271

theorem inequality_equivalence (x : ℝ) : 
  (6 * x - 2 < (x + 1)^2 ∧ (x + 1)^2 < 8 * x - 4) ↔ (3 < x ∧ x < 5) := by
sorry

end inequality_equivalence_l722_72271


namespace max_box_volume_l722_72224

/-- The length of the cardboard in centimeters -/
def cardboard_length : ℝ := 30

/-- The width of the cardboard in centimeters -/
def cardboard_width : ℝ := 14

/-- The volume of the box as a function of the side length of the cut squares -/
def box_volume (x : ℝ) : ℝ := (cardboard_length - 2*x) * (cardboard_width - 2*x) * x

/-- The maximum volume of the box -/
def max_volume : ℝ := 576

theorem max_box_volume :
  ∃ x : ℝ, 0 < x ∧ x < cardboard_width / 2 ∧
  (∀ y : ℝ, 0 < y ∧ y < cardboard_width / 2 → box_volume y ≤ box_volume x) ∧
  box_volume x = max_volume :=
sorry

end max_box_volume_l722_72224


namespace parabola_ellipse_focus_coincidence_l722_72244

/-- Given a parabola and an ellipse, prove that the parameter m of the parabola
    has a specific value when the focus of the parabola coincides with the left
    focus of the ellipse. -/
theorem parabola_ellipse_focus_coincidence (m : ℝ) : 
  (∀ x y : ℝ, y^2 = (4/m)*x → x^2/7 + y^2/3 = 1) →
  (∃ x y : ℝ, y^2 = (4/m)*x ∧ x^2/7 + y^2/3 = 1 ∧ x = -2) →
  m = -1/2 :=
by sorry

end parabola_ellipse_focus_coincidence_l722_72244


namespace perpendicular_unit_vectors_l722_72238

def a : ℝ × ℝ := (4, 2)

theorem perpendicular_unit_vectors :
  let v₁ : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  let v₂ : ℝ × ℝ := (-Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5)
  (v₁.1 * a.1 + v₁.2 * a.2 = 0 ∧ v₁.1^2 + v₁.2^2 = 1) ∧
  (v₂.1 * a.1 + v₂.2 * a.2 = 0 ∧ v₂.1^2 + v₂.2^2 = 1) :=
by sorry

end perpendicular_unit_vectors_l722_72238


namespace balloons_in_park_l722_72289

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 1

/-- The total number of balloons brought to the park -/
def total_balloons : ℕ := allan_balloons + jake_balloons

theorem balloons_in_park : total_balloons = 3 := by
  sorry

end balloons_in_park_l722_72289


namespace devin_basketball_chance_l722_72235

/-- Represents the chance of making the basketball team based on height -/
def basketballChance (initialHeight : ℕ) (growth : ℕ) : ℝ :=
  let baseHeight : ℕ := 66
  let baseChance : ℝ := 0.1
  let chanceIncreasePerInch : ℝ := 0.1
  let finalHeight : ℕ := initialHeight + growth
  let additionalInches : ℕ := max (finalHeight - baseHeight) 0
  baseChance + (additionalInches : ℝ) * chanceIncreasePerInch

/-- Theorem stating Devin's chance of making the team after growing -/
theorem devin_basketball_chance :
  basketballChance 65 3 = 0.3 := by
  sorry

#eval basketballChance 65 3

end devin_basketball_chance_l722_72235


namespace equation_solution_l722_72272

theorem equation_solution : ∃! x : ℝ, (3 : ℝ) / (x - 3) = (4 : ℝ) / (x - 4) := by
  sorry

end equation_solution_l722_72272


namespace final_state_is_green_l722_72258

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Simulates the color change when two different colored chameleons meet -/
def colorChange (state : ChameleonState) : ChameleonState :=
  sorry

/-- Checks if all chameleons are the same color -/
def allSameColor (state : ChameleonState) : Bool :=
  sorry

/-- Theorem: The only possible final state where all chameleons are the same color is green -/
theorem final_state_is_green (state : ChameleonState) :
  (state.yellow + state.red + state.green = totalChameleons) →
  (allSameColor state = true) →
  (state.green = totalChameleons ∧ state.yellow = 0 ∧ state.red = 0) :=
by sorry

end final_state_is_green_l722_72258


namespace fourth_level_open_spots_l722_72279

-- Define the structure of the parking garage
structure ParkingGarage where
  total_levels : Nat
  spots_per_level : Nat
  open_spots_first : Nat
  open_spots_second : Nat
  open_spots_third : Nat
  full_spots_total : Nat

-- Define the problem instance
def parking_problem : ParkingGarage :=
  { total_levels := 4
  , spots_per_level := 100
  , open_spots_first := 58
  , open_spots_second := 60  -- 58 + 2
  , open_spots_third := 65   -- 60 + 5
  , full_spots_total := 186 }

-- Theorem statement
theorem fourth_level_open_spots :
  let p := parking_problem
  let total_spots := p.total_levels * p.spots_per_level
  let open_spots_first_three := p.open_spots_first + p.open_spots_second + p.open_spots_third
  let total_open_spots := total_spots - p.full_spots_total
  total_open_spots - open_spots_first_three = 31 := by
  sorry

end fourth_level_open_spots_l722_72279


namespace seed_distribution_l722_72242

theorem seed_distribution (total : ℕ) (a b c : ℕ) : 
  total = 100 →
  a = b + 10 →
  b = 30 →
  total = a + b + c →
  c = 30 := by
sorry

end seed_distribution_l722_72242


namespace cauchy_schwarz_2d_l722_72231

theorem cauchy_schwarz_2d (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end cauchy_schwarz_2d_l722_72231


namespace computer_profit_pricing_l722_72218

theorem computer_profit_pricing (cost selling_price_40 selling_price_60 : ℝ) :
  selling_price_40 = 2240 ∧
  selling_price_40 = cost * 1.4 →
  selling_price_60 = cost * 1.6 →
  selling_price_60 = 2560 := by
sorry

end computer_profit_pricing_l722_72218


namespace white_surface_area_fraction_l722_72282

-- Define the structure of the cube
structure Cube where
  edge_length : ℝ
  small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

-- Define the larger cube
def larger_cube : Cube :=
  { edge_length := 4
  , small_cubes := 64
  , white_cubes := 48
  , black_cubes := 16 }

-- Function to calculate the surface area of a cube
def surface_area (c : Cube) : ℝ :=
  6 * c.edge_length ^ 2

-- Function to calculate the number of exposed black faces
def exposed_black_faces (c : Cube) : ℕ :=
  24 + 4  -- 8 corners with 3 faces each, plus 4 along the top edge

-- Theorem stating the fraction of white surface area
theorem white_surface_area_fraction (c : Cube) :
  c = larger_cube →
  (surface_area c - exposed_black_faces c) / surface_area c = 17 / 24 := by
  sorry

end white_surface_area_fraction_l722_72282


namespace tangent_line_at_P_l722_72236

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + 1

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the two possible tangent line equations
def tangent1 (x y : ℝ) : Prop := 3*x - y - 1 = 0
def tangent2 (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Theorem statement
theorem tangent_line_at_P :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m*x + b ↔ (tangent1 x y ∨ tangent2 x y)) ∧
  (curve P.1 = P.2) ∧
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, 
    (curve (P.1 + h) - curve P.1) / h - m < ε) := by
  sorry

end tangent_line_at_P_l722_72236


namespace quadratic_inequality_solution_set_l722_72217

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 2*x < 3} = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end quadratic_inequality_solution_set_l722_72217


namespace range_of_m_range_of_a_l722_72261

-- Define the propositions
def p (m : ℝ) : Prop := |m - 2| < 1
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*Real.sqrt 2*x + m = 0
def r (m a : ℝ) : Prop := a - 2 < m ∧ m < a + 1

-- Theorem 1
theorem range_of_m (m : ℝ) : p m ∧ ¬(q m) → 2 < m ∧ m < 3 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) : 
  (∀ m : ℝ, p m → r m a) ∧ ¬(∀ m : ℝ, r m a → p m) → 
  2 ≤ a ∧ a ≤ 3 := by sorry

end range_of_m_range_of_a_l722_72261


namespace amalie_remaining_coins_l722_72288

/-- Given the ratio of Elsa's coins to Amalie's coins and their total coins,
    calculate how many coins Amalie remains with after spending 3/4 of her coins. -/
theorem amalie_remaining_coins
  (ratio_elsa : ℚ)
  (ratio_amalie : ℚ)
  (total_coins : ℕ)
  (h_ratio : ratio_elsa / ratio_amalie = 10 / 45)
  (h_total : ratio_elsa + ratio_amalie = 1)
  (h_coins : (ratio_amalie * total_coins : ℚ).num * (1 / (ratio_amalie * total_coins : ℚ).den : ℚ) * total_coins = 360) :
  (1 / 4 : ℚ) * ((ratio_amalie * total_coins : ℚ).num * (1 / (ratio_amalie * total_coins : ℚ).den : ℚ) * total_coins) = 90 :=
sorry

end amalie_remaining_coins_l722_72288


namespace circle_area_l722_72209

theorem circle_area (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 + 10 * x - 6 * y - 18 = 0) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = r^2) ∧ 
    (π * r^2 = 35/2 * π)) :=
by sorry

end circle_area_l722_72209


namespace lana_muffin_sales_l722_72276

/-- Lana's muffin sales problem -/
theorem lana_muffin_sales (goal : ℕ) (morning_sales : ℕ) (afternoon_sales : ℕ)
  (h1 : goal = 20)
  (h2 : morning_sales = 12)
  (h3 : afternoon_sales = 4) :
  goal - morning_sales - afternoon_sales = 4 := by
  sorry

end lana_muffin_sales_l722_72276


namespace ice_cube_volume_l722_72273

theorem ice_cube_volume (initial_volume : ℝ) : 
  initial_volume > 0 →
  (initial_volume * (1/4) * (1/4) = 0.75) →
  initial_volume = 12 := by
sorry

end ice_cube_volume_l722_72273


namespace minimum_sales_increase_l722_72251

theorem minimum_sales_increase (x : ℝ) : 
  let jan_to_may : ℝ := 38.6
  let june : ℝ := 5
  let july : ℝ := june * (1 + x / 100)
  let august : ℝ := july * (1 + x / 100)
  let sep_oct : ℝ := july + august
  let total : ℝ := jan_to_may + june + july + august + sep_oct
  (total ≥ 70 ∧ ∀ y, y < x → (
    let july_y : ℝ := june * (1 + y / 100)
    let august_y : ℝ := july_y * (1 + y / 100)
    let sep_oct_y : ℝ := july_y + august_y
    let total_y : ℝ := jan_to_may + june + july_y + august_y + sep_oct_y
    total_y < 70
  )) → x = 20 := by
sorry

end minimum_sales_increase_l722_72251


namespace card_drawing_problem_l722_72202

theorem card_drawing_problem (n : Nat) (r y b g : Nat) (total_cards : Nat) (drawn_cards : Nat) : 
  n = 12 → r = 3 → y = 3 → b = 3 → g = 3 → 
  total_cards = n → 
  drawn_cards = 3 → 
  (Nat.choose total_cards drawn_cards) - 
  (4 * (Nat.choose r drawn_cards)) - 
  ((Nat.choose r 2) * (Nat.choose (y + b + g) 1)) = 189 := by
sorry

end card_drawing_problem_l722_72202


namespace least_positive_integer_divisible_by_four_distinct_primes_l722_72222

def is_divisible_by_four_distinct_primes (n : ℕ) : Prop :=
  ∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧ n % s = 0

theorem least_positive_integer_divisible_by_four_distinct_primes :
  (∀ m : ℕ, m > 0 → is_divisible_by_four_distinct_primes m → m ≥ 210) ∧
  is_divisible_by_four_distinct_primes 210 := by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l722_72222


namespace triangle_sine_squared_ratio_l722_72227

theorem triangle_sine_squared_ratio (a b c : ℝ) (A B C : Real) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  S > 0 →
  S = (1/2) * a * b * Real.sin C →
  (a^2 + b^2) * Real.tan C = 8 * S →
  (Real.sin A)^2 + (Real.sin B)^2 = 2 * (Real.sin C)^2 := by
  sorry

end triangle_sine_squared_ratio_l722_72227


namespace arithmetic_progression_five_digit_terms_l722_72274

/-- 
Given an arithmetic progression with first term a₁ = -1 and common difference d = 19,
this theorem states that the terms consisting only of the digit 5 are given by the formula:
n = (5 * (10^(171k+1) + 35)) / 171, where k is a non-negative integer
-/
theorem arithmetic_progression_five_digit_terms 
  (k : ℕ) : 
  ∃ (n : ℕ), 
    ((-1 : ℤ) + (n - 1) * 19 = 5 * ((10 ^ (171 * k + 1) - 1) / 9)) ∧ 
    (n = (5 * (10 ^ (171 * k + 1) + 35)) / 171) := by
  sorry

end arithmetic_progression_five_digit_terms_l722_72274


namespace area_after_folding_l722_72291

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (D : Point)
  (R : Point)
  (Q : Point)
  (C : Point)

/-- Calculates the area of a quadrilateral -/
def area_quadrilateral (quad : Quadrilateral) : ℝ := sorry

/-- Creates a rectangle with given dimensions -/
def create_rectangle (width : ℝ) (height : ℝ) : Rectangle := sorry

/-- Performs the folding operation on the rectangle -/
def fold_rectangle (rect : Rectangle) : Quadrilateral := sorry

theorem area_after_folding (width height : ℝ) :
  width = 5 →
  height = 8 →
  let rect := create_rectangle width height
  let folded := fold_rectangle rect
  area_quadrilateral folded = 11.5 := by sorry

end area_after_folding_l722_72291


namespace add_5_23_base6_l722_72270

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 -/
def addBase6 (a b : ℕ) : ℕ := base10To6 (base6To10 a + base6To10 b)

theorem add_5_23_base6 : addBase6 5 23 = 32 := by sorry

end add_5_23_base6_l722_72270


namespace binomial_10_3_l722_72263

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l722_72263


namespace imaginary_part_of_z_l722_72247

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) :
  z.im = -5/2 := by
sorry

end imaginary_part_of_z_l722_72247


namespace salary_fraction_on_food_l722_72262

theorem salary_fraction_on_food
  (salary : ℝ)
  (rent_fraction : ℝ)
  (clothes_fraction : ℝ)
  (remaining : ℝ)
  (h1 : salary = 180000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 18000)
  (h5 : ∃ food_fraction : ℝ, 
    food_fraction * salary + rent_fraction * salary + clothes_fraction * salary + remaining = salary) :
  ∃ food_fraction : ℝ, food_fraction = 1/5 := by
sorry

end salary_fraction_on_food_l722_72262


namespace sum_reciprocals_bounds_l722_72257

theorem sum_reciprocals_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  (1 / a + 1 / b ≥ 4 / 3) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1 / x + 1 / y = 4 / 3) ∧
  (∀ M : ℝ, ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1 / x + 1 / y > M) :=
by sorry

end sum_reciprocals_bounds_l722_72257


namespace sock_pair_count_l722_72284

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_pairs (white brown blue green : ℕ) : ℕ :=
  white * brown + white * blue + white * green +
  brown * blue + brown * green +
  blue * green

/-- Theorem: There are 81 ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 5 brown, 3 blue, and 2 green socks -/
theorem sock_pair_count :
  different_color_pairs 5 5 3 2 = 81 := by sorry

end sock_pair_count_l722_72284


namespace jadens_car_count_l722_72215

/-- Calculates the final number of toy cars Jaden has after a series of transactions -/
def jadensFinalCarCount (initial bought birthday givenSister givenVinnie tradedAway tradedFor : ℕ) : ℕ :=
  initial + bought + birthday - givenSister - givenVinnie - tradedAway + tradedFor

/-- Theorem stating that Jaden ends up with 45 toy cars -/
theorem jadens_car_count : 
  jadensFinalCarCount 14 28 12 8 3 5 7 = 45 := by
  sorry

end jadens_car_count_l722_72215


namespace fixed_salary_is_400_l722_72266

/-- Represents the fixed salary in the new commission scheme -/
def fixed_salary : ℕ := sorry

/-- Represents the total sales amount -/
def total_sales : ℕ := 12000

/-- Represents the threshold for commission in the new scheme -/
def commission_threshold : ℕ := 4000

/-- Calculates the commission under the old scheme -/
def old_commission (sales : ℕ) : ℕ :=
  (sales * 5) / 100

/-- Calculates the commission under the new scheme -/
def new_commission (sales : ℕ) : ℕ :=
  ((sales - commission_threshold) * 25) / 1000

/-- States that the new scheme pays 600 more than the old scheme -/
axiom new_scheme_difference : 
  fixed_salary + new_commission total_sales = old_commission total_sales + 600

theorem fixed_salary_is_400 : fixed_salary = 400 := by
  sorry

end fixed_salary_is_400_l722_72266


namespace fraction_irreducibility_l722_72241

theorem fraction_irreducibility (n : ℕ) : 
  Irreducible ((2 * n^2 + 11 * n - 18) / (n + 7)) ↔ n % 3 = 0 ∨ n % 3 = 1 := by
  sorry

end fraction_irreducibility_l722_72241


namespace negation_of_universal_proposition_l722_72225

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ x₀^2) := by
  sorry

end negation_of_universal_proposition_l722_72225


namespace one_equals_a_l722_72292

theorem one_equals_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end one_equals_a_l722_72292


namespace inequality_equivalence_l722_72210

theorem inequality_equivalence (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ -1/3 < x ∧ x < 1 := by
  sorry

end inequality_equivalence_l722_72210


namespace sequence_inequality_existence_l722_72246

theorem sequence_inequality_existence (a b : ℕ → ℕ) :
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q := by
  sorry

end sequence_inequality_existence_l722_72246


namespace a_equals_2_sufficient_not_necessary_l722_72216

def third_term (a : ℝ) : ℝ → ℝ := λ x ↦ 15 * a^2 * x^4

theorem a_equals_2_sufficient_not_necessary :
  (∀ x, third_term 2 x = 60 * x^4) ∧
  (∃ a ≠ 2, ∀ x, third_term a x = 60 * x^4) :=
sorry

end a_equals_2_sufficient_not_necessary_l722_72216


namespace parabola_and_max_area_line_l722_72232

-- Define the parabola
def Parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define a point on the parabola
def PointOnParabola (p : ℝ) (x₀ : ℝ) : Prop := Parabola p x₀ 4

-- Define the distance from a point to the focus
def DistanceToFocus (p : ℝ) (x₀ : ℝ) : Prop := x₀ + p/2 = 4

-- Define the angle bisector condition
def AngleBisectorCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ (4 - y₁)/(2 - x₁) = -(4 - y₂)/(2 - x₂)

-- Main theorem
theorem parabola_and_max_area_line
  (p : ℝ) (x₀ : ℝ)
  (h₁ : PointOnParabola p x₀)
  (h₂ : DistanceToFocus p x₀) :
  (∀ x y, Parabola p x y ↔ y^2 = 8*x) ∧
  (∃ x₁ y₁ x₂ y₂,
    AngleBisectorCondition x₁ y₁ x₂ y₂ ∧
    Parabola p x₁ y₁ ∧ Parabola p x₂ y₂ ∧
    (∀ a b, (Parabola p a b ∧ b ≤ 0) →
      (x₁ - 2)*(y₂ - 4) - (x₂ - 2)*(y₁ - 4) ≤ (x₁ - 2)*(b - 4) - (a - 2)*(y₁ - 4)) ∧
    x₁ + y₁ = 0 ∧ x₂ + y₂ = 0) :=
by sorry

end parabola_and_max_area_line_l722_72232


namespace greatest_m_for_ratio_bound_l722_72205

/-- The number of ordered m-coverings of a set with 2n elements -/
def a (m n : ℕ) : ℕ := (2^m - 1)^(2*n)

/-- The number of ordered m-coverings without pairs of a set with 2n elements -/
def b (m n : ℕ) : ℕ := (3^m - 2^(m+1) + 1)^n

/-- The ratio of a(m,n) to b(m,n) -/
def ratio (m n : ℕ) : ℚ := (a m n : ℚ) / (b m n : ℚ)

theorem greatest_m_for_ratio_bound :
  (∃ n : ℕ+, ratio 26 n ≤ 2021) ∧
  (∀ m > 26, ∀ n : ℕ+, ratio m n > 2021) :=
sorry

end greatest_m_for_ratio_bound_l722_72205


namespace point_coordinates_wrt_origin_l722_72268

/-- Given a point P(-2,3) in a plane rectangular coordinate system,
    its coordinates with respect to the origin are (2,-3). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-2, 3)
  let origin_symmetric (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  origin_symmetric P = (2, -3) := by sorry

end point_coordinates_wrt_origin_l722_72268


namespace shop_width_l722_72298

/-- Given a rectangular shop with the following properties:
  * Length is 18 feet
  * Monthly rent is Rs. 3600
  * Annual rent per square foot is Rs. 120
  Prove that the width of the shop is 20 feet. -/
theorem shop_width (length : ℝ) (monthly_rent : ℝ) (annual_rent_per_sqft : ℝ) 
  (h1 : length = 18)
  (h2 : monthly_rent = 3600)
  (h3 : annual_rent_per_sqft = 120) :
  (monthly_rent * 12) / (length * annual_rent_per_sqft) = 20 := by
  sorry

end shop_width_l722_72298


namespace new_person_weight_is_143_l722_72229

/-- Calculates the weight of a new person given the following conditions:
  * There are 15 people initially
  * The average weight increases by 5 kg when the new person replaces one person
  * The replaced person weighs 68 kg
-/
def new_person_weight (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  initial_count * avg_increase + replaced_weight

/-- Proves that under the given conditions, the weight of the new person is 143 kg -/
theorem new_person_weight_is_143 :
  new_person_weight 15 5 68 = 143 := by
  sorry

#eval new_person_weight 15 5 68

end new_person_weight_is_143_l722_72229


namespace carmen_candle_usage_l722_72260

/-- Calculates the number of candles needed for a given number of nights and burning hours per night. -/
def candles_needed (total_nights : ℕ) (hours_per_night : ℕ) (nights_per_candle_at_one_hour : ℕ) : ℕ :=
  total_nights * hours_per_night / nights_per_candle_at_one_hour

theorem carmen_candle_usage :
  candles_needed 24 2 8 = 6 := by
  sorry

end carmen_candle_usage_l722_72260


namespace pythagorean_sum_number_with_conditions_l722_72287

def is_pythagorean_sum_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  c^2 + d^2 = 10 * a + b

def G (n : ℕ) : ℚ :=
  let c := (n / 10) % 10
  let d := n % 10
  (c + d : ℚ) / 9

def P (n : ℕ) : ℚ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (10 * a - 2 * c * d + b : ℚ) / 3

theorem pythagorean_sum_number_with_conditions :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
    is_pythagorean_sum_number n ∧
    (∃ k : ℤ, G n = k) ∧
    P n = 3 →
  n = 3772 ∨ n = 3727 :=
sorry

end pythagorean_sum_number_with_conditions_l722_72287


namespace eight_valid_arrangements_l722_72265

/-- A type representing the possible positions of a card -/
inductive Position
  | Original
  | Left
  | Right

/-- A type representing a card arrangement -/
def Arrangement := Fin 5 → Position

/-- A function to check if an arrangement is valid -/
def is_valid (arr : Arrangement) : Prop :=
  ∀ i : Fin 5, arr i = Position.Original ∨ arr i = Position.Left ∨ arr i = Position.Right

/-- The number of valid arrangements -/
def num_valid_arrangements : ℕ := sorry

/-- The main theorem: there are 8 valid arrangements -/
theorem eight_valid_arrangements : num_valid_arrangements = 8 := by sorry

end eight_valid_arrangements_l722_72265


namespace correct_quadratic_equation_l722_72249

theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∃ (b' : ℝ), b' ≠ b ∧ (8 : ℝ) * (2 : ℝ) = 9) →
  (∃ (c' : ℝ), c' ≠ c ∧ (-9 : ℝ) + (-1 : ℝ) = -b') →
  (b = -10 ∧ c = 9) :=
by sorry

end correct_quadratic_equation_l722_72249


namespace integer_solution_of_quadratic_equation_l722_72293

theorem integer_solution_of_quadratic_equation (x y : ℤ) :
  x^2 + y^2 = 3*x*y → x = 0 ∧ y = 0 := by
  sorry

end integer_solution_of_quadratic_equation_l722_72293


namespace equal_numbers_sum_of_squares_l722_72234

theorem equal_numbers_sum_of_squares (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 26 →
  c = 22 →
  d = e →
  d^2 + e^2 = 800 := by
  sorry

end equal_numbers_sum_of_squares_l722_72234


namespace arithmetic_mean_with_additional_number_l722_72248

theorem arithmetic_mean_with_additional_number : 
  let numbers : List ℕ := [16, 24, 45, 63]
  let additional_number := 2 * numbers.head!
  let total_sum := numbers.sum + additional_number
  let count := numbers.length + 1
  (total_sum : ℚ) / count = 36 := by sorry

end arithmetic_mean_with_additional_number_l722_72248


namespace zoo_visitors_l722_72207

/-- Represents the number of adults who went to the zoo on Monday -/
def adults_monday : ℕ := sorry

/-- The total revenue from both days -/
def total_revenue : ℕ := 61

/-- The cost of a child ticket -/
def child_ticket_cost : ℕ := 3

/-- The cost of an adult ticket -/
def adult_ticket_cost : ℕ := 4

/-- The number of children who went to the zoo on Monday -/
def children_monday : ℕ := 7

/-- The number of children who went to the zoo on Tuesday -/
def children_tuesday : ℕ := 4

/-- The number of adults who went to the zoo on Tuesday -/
def adults_tuesday : ℕ := 2

theorem zoo_visitors :
  adults_monday = 5 :=
by sorry

end zoo_visitors_l722_72207


namespace hyperbola_asymptote_l722_72220

theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = (3/5) * x ∧ x^2 / a^2 - y^2 / 9 = 1) →
  a = 5 := by
  sorry

end hyperbola_asymptote_l722_72220


namespace words_with_at_least_two_consonants_l722_72283

/-- The set of all available letters -/
def Letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants -/
def Consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels -/
def Vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering -/
def WordLength : Nat := 5

/-- A function that counts the number of 5-letter words with at least two consonants -/
def countWordsWithAtLeastTwoConsonants : Nat := sorry

theorem words_with_at_least_two_consonants :
  countWordsWithAtLeastTwoConsonants = 7424 := by sorry

end words_with_at_least_two_consonants_l722_72283


namespace eight_times_ten_y_plus_fourteen_sin_y_l722_72204

theorem eight_times_ten_y_plus_fourteen_sin_y (y : ℝ) (Q : ℝ) 
  (h : 4 * (5 * y + 7 * Real.sin y) = Q) : 
  8 * (10 * y + 14 * Real.sin y) = 4 * Q := by
  sorry

end eight_times_ten_y_plus_fourteen_sin_y_l722_72204


namespace pens_probability_theorem_l722_72239

def total_pens : ℕ := 8
def red_pens : ℕ := 4
def blue_pens : ℕ := 4
def pens_to_pick : ℕ := 4

def probability_leftmost_blue_not_picked_rightmost_red_picked : ℚ :=
  4 / 49

theorem pens_probability_theorem :
  let total_arrangements := Nat.choose total_pens red_pens
  let total_pick_ways := Nat.choose total_pens pens_to_pick
  let favorable_red_arrangements := Nat.choose (total_pens - 2) (red_pens - 1)
  let favorable_pick_ways := Nat.choose (total_pens - 2) (pens_to_pick - 1)
  (favorable_red_arrangements * favorable_pick_ways : ℚ) / (total_arrangements * total_pick_ways) =
    probability_leftmost_blue_not_picked_rightmost_red_picked :=
by
  sorry

#check pens_probability_theorem

end pens_probability_theorem_l722_72239


namespace baker_remaining_pastries_l722_72254

/-- The number of pastries Baker made -/
def pastries_made : ℕ := 148

/-- The number of pastries Baker sold -/
def pastries_sold : ℕ := 103

/-- The number of pastries Baker still has -/
def pastries_remaining : ℕ := pastries_made - pastries_sold

theorem baker_remaining_pastries : pastries_remaining = 45 := by
  sorry

end baker_remaining_pastries_l722_72254


namespace conference_languages_l722_72203

/-- The proportion of delegates who know both English and Spanish -/
def both_languages (p_english p_spanish : ℝ) : ℝ :=
  p_english + p_spanish - 1

theorem conference_languages :
  let p_english : ℝ := 0.85
  let p_spanish : ℝ := 0.75
  both_languages p_english p_spanish = 0.60 := by
  sorry

end conference_languages_l722_72203


namespace min_value_theorem_l722_72219

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a + b) * b * c = 5) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y) * y * z = 5 → 2 * a + b + c ≤ 2 * x + y + z ∧
  2 * a + b + c = 2 * Real.sqrt 5 :=
by sorry

#check min_value_theorem

end min_value_theorem_l722_72219


namespace boy_age_multiple_l722_72252

theorem boy_age_multiple : 
  let present_age : ℕ := 16
  let age_six_years_ago : ℕ := present_age - 6
  let age_four_years_hence : ℕ := present_age + 4
  (age_four_years_hence : ℚ) / (age_six_years_ago : ℚ) = 2 := by
  sorry

end boy_age_multiple_l722_72252


namespace spinster_cat_problem_l722_72212

theorem spinster_cat_problem (spinsters cats : ℕ) : 
  (spinsters : ℚ) / cats = 2 / 9 →
  cats = spinsters + 63 →
  spinsters = 18 := by
sorry

end spinster_cat_problem_l722_72212


namespace complement_A_in_B_union_equality_implies_m_range_l722_72211

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | x > m}

-- Theorem 1: Complement of A in B when m = -1
theorem complement_A_in_B : 
  {x : ℝ | x ∈ B (-1) ∧ x ∉ A} = {x : ℝ | x ≥ 3} := by sorry

-- Theorem 2: Range of m when A ∪ B = B
theorem union_equality_implies_m_range (m : ℝ) : 
  A ∪ B m = B m → m ≤ -1 := by sorry

end complement_A_in_B_union_equality_implies_m_range_l722_72211


namespace min_modulus_m_l722_72269

/-- Given a complex number m such that the equation x^2 + mx + 1 + 2i = 0 has real roots,
    the minimum modulus of m is √(2 + 2√5). -/
theorem min_modulus_m (m : ℂ) : 
  (∃ x : ℝ, x^2 + m*x + 1 + 2*Complex.I = 0) → 
  Complex.abs m ≥ Real.sqrt (2 + 2 * Real.sqrt 5) ∧ 
  ∃ m₀ : ℂ, (∃ x : ℝ, x^2 + m₀*x + 1 + 2*Complex.I = 0) ∧ 
            Complex.abs m₀ = Real.sqrt (2 + 2 * Real.sqrt 5) :=
sorry

end min_modulus_m_l722_72269


namespace find_missing_score_l722_72275

def scores : List ℕ := [87, 88, 89, 0, 91, 92, 92, 93, 94]

theorem find_missing_score (x : ℕ) (h : x ∈ scores) :
  (List.sum (List.filter (λ y => y ≠ 87 ∧ y ≠ 94) (List.map (λ y => if y = 0 then x else y) scores))) / 7 = 91 →
  x = 2 := by sorry

end find_missing_score_l722_72275


namespace flour_weight_qualified_l722_72223

def is_qualified (weight : ℝ) : Prop :=
  24.75 ≤ weight ∧ weight ≤ 25.25

theorem flour_weight_qualified :
  is_qualified 24.80 := by sorry

end flour_weight_qualified_l722_72223


namespace nine_digit_divisibility_l722_72208

theorem nine_digit_divisibility (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) (h4 : a ≠ 0) :
  ∃ k : ℕ, (a * 100000000 + b * 10000000 + c * 1000000 +
            a * 100000 + b * 10000 + c * 1000 +
            a * 100 + b * 10 + c) = k * 1001001 :=
by sorry

end nine_digit_divisibility_l722_72208


namespace N_subset_M_l722_72213

-- Define set M
def M : Set ℝ := {x | ∃ n : ℤ, x = n / 2 + 1}

-- Define set N
def N : Set ℝ := {y | ∃ m : ℤ, y = m + 1 / 2}

-- Theorem statement
theorem N_subset_M : N ⊆ M := by sorry

end N_subset_M_l722_72213


namespace cheolsu_weight_l722_72286

/-- Proves that Cheolsu's weight is 36 kg given the problem conditions -/
theorem cheolsu_weight (c m f : ℝ) 
  (h1 : (c + m + f) / 3 = m)  -- average equals mother's weight
  (h2 : c = (2/3) * m)        -- Cheolsu's weight is 2/3 of mother's
  (h3 : f = 72)               -- Father's weight is 72 kg
  : c = 36 := by
  sorry

#check cheolsu_weight

end cheolsu_weight_l722_72286


namespace bank_account_final_amount_l722_72200

/-- Calculates the final amount in a bank account given initial savings, withdrawal, and deposit. -/
def final_amount (initial_savings withdrawal : ℕ) : ℕ :=
  initial_savings - withdrawal + 2 * withdrawal

/-- Theorem stating that given the specific conditions, the final amount is $290. -/
theorem bank_account_final_amount : 
  final_amount 230 60 = 290 := by
  sorry

end bank_account_final_amount_l722_72200


namespace tree_height_reaches_29_feet_in_15_years_l722_72253

/-- Calculates the height of the tree after a given number of years -/
def tree_height (years : ℕ) : ℕ :=
  let initial_height := 4
  let first_year_growth := 5
  let second_year_growth := 4
  let min_growth := 1
  let rec height_after (n : ℕ) (current_height : ℕ) (current_growth : ℕ) : ℕ :=
    if n = 0 then
      current_height
    else if n = 1 then
      height_after (n - 1) (current_height + first_year_growth) second_year_growth
    else if current_growth > min_growth then
      height_after (n - 1) (current_height + current_growth) (current_growth - 1)
    else
      height_after (n - 1) (current_height + min_growth) min_growth
  height_after years initial_height first_year_growth

/-- Theorem stating that it takes 15 years for the tree to reach or exceed 29 feet -/
theorem tree_height_reaches_29_feet_in_15_years :
  tree_height 15 ≥ 29 ∧ ∀ y : ℕ, y < 15 → tree_height y < 29 :=
by sorry

end tree_height_reaches_29_feet_in_15_years_l722_72253


namespace coefficient_x_cubed_3x_plus_2_to_8_l722_72281

theorem coefficient_x_cubed_3x_plus_2_to_8 : 
  (Finset.range 9).sum (λ k => Nat.choose 8 k * (3 ^ k) * (2 ^ (8 - k)) * if k = 3 then 1 else 0) = 48384 := by
  sorry

end coefficient_x_cubed_3x_plus_2_to_8_l722_72281


namespace product_of_three_numbers_l722_72256

theorem product_of_three_numbers (x y z n : ℤ) 
  (sum_eq : x + y + z = 200)
  (x_eq : 8 * x = n)
  (y_eq : y - 5 = n)
  (z_eq : z + 5 = n) :
  x * y * z = 372462 := by
  sorry

end product_of_three_numbers_l722_72256


namespace triangle_third_side_length_l722_72295

theorem triangle_third_side_length (a b : ℝ) (θ : ℝ) (ha : a = 9) (hb : b = 11) (hθ : θ = 135 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*Real.cos θ ∧ c = Real.sqrt (202 + 99 * Real.sqrt 2) :=
sorry

end triangle_third_side_length_l722_72295


namespace ten_possible_values_for_d_l722_72285

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct_digits (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a five-digit number represented by individual digits to a natural number -/
def to_nat (a b c d e : Digit) : ℕ :=
  10000 * a.val + 1000 * b.val + 100 * c.val + 10 * d.val + e.val

/-- The main theorem stating that there are 10 possible values for D -/
theorem ten_possible_values_for_d :
  ∃ (possible_d_values : Finset Digit),
    possible_d_values.card = 10 ∧
    ∀ (a b c d : Digit),
      distinct_digits a b c d →
      (to_nat a b c b c) + (to_nat c b a d b) = (to_nat d b d d d) →
      d ∈ possible_d_values :=
sorry

end ten_possible_values_for_d_l722_72285


namespace right_triangle_polyhedron_faces_even_l722_72278

/-- A convex polyhedron with right-angled triangular faces -/
structure RightTrianglePolyhedron where
  faces : ℕ
  isConvex : Bool
  allFacesRightTriangle : Bool
  facesAtLeastFour : faces ≥ 4

/-- Theorem stating that the number of faces in a right-angled triangle polyhedron is even -/
theorem right_triangle_polyhedron_faces_even (p : RightTrianglePolyhedron) : 
  Even p.faces := by sorry

end right_triangle_polyhedron_faces_even_l722_72278


namespace expression_equality_l722_72221

theorem expression_equality : 
  (((3 + 5 + 7) / (2 + 4 + 6)) * 2 - ((2 + 4 + 6) / (3 + 5 + 7))) = 17 / 10 := by
  sorry

end expression_equality_l722_72221


namespace tan_half_product_l722_72264

theorem tan_half_product (a b : Real) :
  7 * (Real.cos a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 5 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -5) := by
  sorry

end tan_half_product_l722_72264


namespace possible_values_of_a_l722_72230

theorem possible_values_of_a (a : ℝ) 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (non_neg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0)
  (eq1 : x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ = a)
  (eq2 : x₁ + 8*x₂ + 27*x₃ + 64*x₄ + 125*x₅ = a^2)
  (eq3 : x₁ + 32*x₂ + 243*x₃ + 1024*x₄ + 3125*x₅ = a^3) :
  a ∈ ({0, 1, 4, 9, 16, 25} : Set ℝ) := by
sorry

end possible_values_of_a_l722_72230


namespace first_row_seats_theorem_l722_72296

/-- Represents a theater with a specific seating arrangement. -/
structure Theater where
  rows : ℕ
  seatIncrement : ℕ
  totalSeats : ℕ

/-- Calculates the number of seats in the first row of the theater. -/
def firstRowSeats (t : Theater) : ℚ :=
  (t.totalSeats / 10 - 76) / 2

/-- Theorem stating the relationship between the total seats and the number of seats in the first row. -/
theorem first_row_seats_theorem (t : Theater) 
    (h1 : t.rows = 20)
    (h2 : t.seatIncrement = 4)
    (h3 : t.totalSeats = 10 * (firstRowSeats t * 2 + 76)) : 
  firstRowSeats t = (t.totalSeats / 10 - 76) / 2 := by
  sorry

end first_row_seats_theorem_l722_72296


namespace sum_seven_smallest_multiples_of_13_l722_72214

theorem sum_seven_smallest_multiples_of_13 : 
  (Finset.range 7).sum (fun i => 13 * (i + 1)) = 364 := by
  sorry

end sum_seven_smallest_multiples_of_13_l722_72214


namespace intersection_A_complement_B_B_subset_A_range_l722_72299

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - a^2 - 1) < 0}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B (Real.sqrt 2)) = {x | 1 ≤ x ∧ x ≤ Real.sqrt 2 ∨ 3 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2
theorem B_subset_A_range :
  ∀ a : ℝ, B a ⊆ A → 1 ≤ a ∧ a ≤ Real.sqrt 3 := by sorry

end intersection_A_complement_B_B_subset_A_range_l722_72299


namespace z_in_third_quadrant_l722_72259

def z : ℂ := (-2 + Complex.I) * Complex.I

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
by sorry

end z_in_third_quadrant_l722_72259


namespace quidditch_tournament_equal_wins_l722_72237

/-- Represents a team in the Quidditch tournament -/
structure Team :=
  (id : Nat)

/-- Represents the tournament setup -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (wins : Team → Nat)
  (h_num_teams : teams.card = num_teams)
  (h_wins_bound : ∀ t ∈ teams, wins t < num_teams)
  (h_total_wins : (teams.sum wins) = num_teams * (num_teams - 1) / 2)

/-- Main theorem statement -/
theorem quidditch_tournament_equal_wins (tournament : Tournament) 
  (h_eight_teams : tournament.num_teams = 8) :
  ∃ (A B C D : Team), A ∈ tournament.teams ∧ B ∈ tournament.teams ∧ 
    C ∈ tournament.teams ∧ D ∈ tournament.teams ∧ A ≠ B ∧ C ≠ D ∧ 
    tournament.wins A + tournament.wins B = tournament.wins C + tournament.wins D :=
by sorry

end quidditch_tournament_equal_wins_l722_72237


namespace nearest_integer_to_3_plus_sqrt2_to_4_l722_72240

theorem nearest_integer_to_3_plus_sqrt2_to_4 :
  ∃ (n : ℤ), n = 386 ∧ ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 2)^4 - n| ≤ |((3 : ℝ) + Real.sqrt 2)^4 - m| :=
sorry

end nearest_integer_to_3_plus_sqrt2_to_4_l722_72240


namespace collinear_points_sum_l722_72290

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of collinearity for three points -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t s : ℝ, q.x - p.x = t * (r.x - p.x) ∧
             q.y - p.y = t * (r.y - p.y) ∧
             q.z - p.z = t * (r.z - p.z) ∧
             q.x - p.x = s * (r.x - q.x) ∧
             q.y - p.y = s * (r.y - q.y) ∧
             q.z - p.z = s * (r.z - q.z)

theorem collinear_points_sum (m n : ℝ) :
  let M : Point3D := ⟨1, 0, 1⟩
  let N : Point3D := ⟨2, m, 3⟩
  let P : Point3D := ⟨2, 2, n + 1⟩
  collinear M N P → m + n = 4 := by
  sorry

end collinear_points_sum_l722_72290


namespace parabola_vertex_vertex_coordinates_l722_72277

/-- The vertex of a parabola y = a(x - h)^2 + k is the point (h, k) --/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * (x - h)^2 + k
  (h, k) = (h, f h) ∧ ∀ x, f x ≥ f h := by sorry

/-- The coordinates of the vertex of the parabola y = 2(x-1)^2 + 8 are (1, 8) --/
theorem vertex_coordinates :
  let f : ℝ → ℝ := fun x ↦ 2 * (x - 1)^2 + 8
  (1, 8) = (1, f 1) ∧ ∀ x, f x ≥ f 1 := by sorry

end parabola_vertex_vertex_coordinates_l722_72277


namespace larger_number_problem_l722_72267

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 := by
  sorry

end larger_number_problem_l722_72267


namespace sandbox_cost_l722_72280

/-- The cost to fill a rectangular sandbox with sand -/
theorem sandbox_cost (length width depth price_per_cubic_foot : ℝ) 
  (h_length : length = 4)
  (h_width : width = 3)
  (h_depth : depth = 1.5)
  (h_price : price_per_cubic_foot = 3) : 
  length * width * depth * price_per_cubic_foot = 54 := by
  sorry

#check sandbox_cost

end sandbox_cost_l722_72280


namespace solution_form_and_sum_l722_72233

theorem solution_form_and_sum (x y : ℝ) : 
  (x + y = 7 ∧ 4 * x * y = 7) →
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d ∧
    a = 7 ∧ b = 1 ∧ c = 42 ∧ d = 2 ∧
    a + b + c + d = 52 :=
by sorry

end solution_form_and_sum_l722_72233


namespace angle_difference_range_l722_72226

theorem angle_difference_range (α β : ℝ) 
  (h1 : -π/2 < α) (h2 : α < 0) (h3 : 0 < β) (h4 : β < π/3) : 
  -5*π/6 < α - β ∧ α - β < 0 := by
  sorry

end angle_difference_range_l722_72226


namespace xz_length_l722_72250

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- ∠X = 90°
  t.X = 90

def has_hypotenuse_10 (t : Triangle) : Prop :=
  -- YZ = 10
  t.Y = 10

def satisfies_trig_relation (t : Triangle) : Prop :=
  -- tan Z = 3 sin Z
  Real.tan t.Z = 3 * Real.sin t.Z

-- Theorem statement
theorem xz_length (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : has_hypotenuse_10 t) 
  (h3 : satisfies_trig_relation t) : 
  -- XZ = 10/3
  t.Z = 10/3 := by
  sorry

end xz_length_l722_72250


namespace initial_population_is_10000_l722_72201

/-- Represents the annual population growth rate -/
def annual_growth_rate : ℝ := 0.1

/-- Represents the population after 2 years -/
def population_after_2_years : ℕ := 12100

/-- Calculates the population after n years given an initial population -/
def population_after_n_years (initial_population : ℝ) (n : ℕ) : ℝ :=
  initial_population * (1 + annual_growth_rate) ^ n

/-- Theorem stating that if a population grows by 10% annually and reaches 12100 after 2 years,
    the initial population was 10000 -/
theorem initial_population_is_10000 :
  ∃ (initial_population : ℕ),
    (population_after_n_years initial_population 2 = population_after_2_years) ∧
    (initial_population = 10000) := by
  sorry

end initial_population_is_10000_l722_72201


namespace min_additional_matches_for_square_grid_l722_72255

/-- Calculates the number of matches needed for a rectangular grid -/
def matches_for_grid (rows : ℕ) (cols : ℕ) : ℕ :=
  (rows + 1) * cols + (cols + 1) * rows

/-- Represents the problem of finding the minimum additional matches needed -/
theorem min_additional_matches_for_square_grid :
  let initial_matches := matches_for_grid 3 7
  let min_square_size := (initial_matches / 4 : ℕ).sqrt.succ
  let square_matches := matches_for_grid min_square_size min_square_size
  square_matches - initial_matches = 8 := by
  sorry

end min_additional_matches_for_square_grid_l722_72255


namespace arrange_plants_under_lamps_count_l722_72243

/-- Represents the number of ways to arrange plants under lamps -/
def arrange_plants_under_lamps : ℕ :=
  let num_plants : ℕ := 4
  let num_plant_types : ℕ := 3
  let num_lamps : ℕ := 4
  let num_lamp_colors : ℕ := 2
  
  -- All plants under same color lamp
  let all_under_one_color : ℕ := num_lamp_colors
  let three_under_one_color : ℕ := num_plants * num_lamp_colors
  
  -- Plants under different colored lamps
  let two_types_each_color : ℕ := (Nat.choose num_plant_types 2) * num_lamp_colors
  let one_type_alone : ℕ := num_plant_types * num_lamp_colors
  
  all_under_one_color + three_under_one_color + two_types_each_color + one_type_alone

/-- Theorem stating the correct number of ways to arrange plants under lamps -/
theorem arrange_plants_under_lamps_count :
  arrange_plants_under_lamps = 22 := by sorry

end arrange_plants_under_lamps_count_l722_72243


namespace high_school_students_l722_72206

theorem high_school_students (total : ℕ) 
  (h1 : (total : ℚ) / 2 = (total : ℚ) * (1 / 2))
  (h2 : (total : ℚ) / 2 * (1 / 5) = (total : ℚ) * (1 / 10))
  (h3 : (total : ℚ) * (1 / 10) + 160 = (total : ℚ) * (1 / 2)) : 
  total = 400 := by
sorry

end high_school_students_l722_72206
