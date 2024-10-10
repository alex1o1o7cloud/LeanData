import Mathlib

namespace steve_pencil_theorem_l1312_131235

def steve_pencil_problem (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra_pencils : ℕ) : Prop :=
  let total_pencils := boxes * pencils_per_box
  let matt_pencils := lauren_pencils + matt_extra_pencils
  let given_away_pencils := lauren_pencils + matt_pencils
  let remaining_pencils := total_pencils - given_away_pencils
  remaining_pencils = 9

theorem steve_pencil_theorem :
  steve_pencil_problem 2 12 6 3 :=
by
  sorry

end steve_pencil_theorem_l1312_131235


namespace ceiling_sum_of_roots_l1312_131297

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ + ⌈Real.sqrt 243⌉ = 24 := by
  sorry

end ceiling_sum_of_roots_l1312_131297


namespace range_of_shifted_and_translated_function_l1312_131271

/-- Given a function f: ℝ → ℝ with range [1,2], 
    prove that the range of g(x) = f(x+1)-2 is [-1,0] -/
theorem range_of_shifted_and_translated_function 
  (f : ℝ → ℝ) (h : Set.range f = Set.Icc 1 2) :
  Set.range (fun x ↦ f (x + 1) - 2) = Set.Icc (-1) 0 := by
  sorry

end range_of_shifted_and_translated_function_l1312_131271


namespace compound_molecular_weight_l1312_131202

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_atoms hydrogen_atoms oxygen_atoms : ℕ) 
  (carbon_weight hydrogen_weight oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + 
  (hydrogen_atoms : ℝ) * hydrogen_weight + 
  (oxygen_atoms : ℝ) * oxygen_weight

/-- The molecular weight of a compound with 4 Carbon atoms, 1 Hydrogen atom, and 1 Oxygen atom 
    is equal to 65.048 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 4 1 1 12.01 1.008 16.00 = 65.048 := by
  sorry

end compound_molecular_weight_l1312_131202


namespace max_surface_area_after_cut_l1312_131242

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def Cuboid.surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.width * c.height + c.length * c.height)

/-- Represents the result of cutting a cuboid into two triangular prisms -/
structure CutResult where
  prism1_surface_area : ℝ
  prism2_surface_area : ℝ

/-- Calculates the sum of surface areas after cutting -/
def CutResult.totalSurfaceArea (cr : CutResult) : ℝ :=
  cr.prism1_surface_area + cr.prism2_surface_area

/-- The main theorem stating the maximum sum of surface areas after cutting -/
theorem max_surface_area_after_cut (c : Cuboid) 
  (h1 : c.length = 5) 
  (h2 : c.width = 4) 
  (h3 : c.height = 3) : 
  (∃ (cr : CutResult), ∀ (cr' : CutResult), cr.totalSurfaceArea ≥ cr'.totalSurfaceArea) → 
  (∃ (cr : CutResult), cr.totalSurfaceArea = 144) :=
sorry

end max_surface_area_after_cut_l1312_131242


namespace red_cell_remains_l1312_131279

theorem red_cell_remains (n : ℕ) :
  ∀ (black_rows black_cols : Finset (Fin (2*n))),
  black_rows.card = n ∧ black_cols.card = n →
  ∃ (red_cells : Finset (Fin (2*n) × Fin (2*n))),
  red_cells.card = 2*n^2 + 1 ∧
  ∃ (cell : Fin (2*n) × Fin (2*n)),
  cell ∈ red_cells ∧ cell.1 ∉ black_rows ∧ cell.2 ∉ black_cols :=
sorry

end red_cell_remains_l1312_131279


namespace population_difference_after_increase_l1312_131258

/-- Represents the population of birds in a wildlife reserve -/
structure BirdPopulation where
  eagles : ℕ
  falcons : ℕ
  hawks : ℕ
  owls : ℕ

/-- Calculates the difference between the most and least populous bird types -/
def populationDifference (pop : BirdPopulation) : ℕ :=
  max pop.eagles (max pop.falcons (max pop.hawks pop.owls)) -
  min pop.eagles (min pop.falcons (min pop.hawks pop.owls))

/-- Calculates the new population after increasing the least populous by 10% -/
def increaseLeastPopulous (pop : BirdPopulation) : BirdPopulation :=
  let minPop := min pop.eagles (min pop.falcons (min pop.hawks pop.owls))
  let increase := minPop * 10 / 100
  { eagles := if pop.eagles = minPop then pop.eagles + increase else pop.eagles,
    falcons := if pop.falcons = minPop then pop.falcons + increase else pop.falcons,
    hawks := if pop.hawks = minPop then pop.hawks + increase else pop.hawks,
    owls := if pop.owls = minPop then pop.owls + increase else pop.owls }

theorem population_difference_after_increase (initialPop : BirdPopulation) :
  initialPop.eagles = 150 →
  initialPop.falcons = 200 →
  initialPop.hawks = 320 →
  initialPop.owls = 270 →
  populationDifference (increaseLeastPopulous initialPop) = 155 := by
  sorry

end population_difference_after_increase_l1312_131258


namespace kim_candy_bars_saved_l1312_131224

/-- The number of candy bars Kim's dad buys her per week -/
def candyBarsPerWeek : ℕ := 2

/-- The number of weeks it takes Kim to eat one candy bar -/
def weeksPerCandyBar : ℕ := 4

/-- The total number of weeks -/
def totalWeeks : ℕ := 16

/-- The number of candy bars Kim saved after the total number of weeks -/
def candyBarsSaved : ℕ := totalWeeks * candyBarsPerWeek - totalWeeks / weeksPerCandyBar

theorem kim_candy_bars_saved : candyBarsSaved = 28 := by
  sorry

end kim_candy_bars_saved_l1312_131224


namespace days_A_worked_alone_l1312_131244

/-- Represents the number of days it takes for A and B to finish the work together -/
def total_days_together : ℝ := 40

/-- Represents the number of days it takes for A to finish the work alone -/
def total_days_A : ℝ := 28

/-- Represents the number of days A and B worked together before B left -/
def days_worked_together : ℝ := 10

/-- Represents the total amount of work to be done -/
def total_work : ℝ := 1

theorem days_A_worked_alone :
  let remaining_work := total_work - (days_worked_together / total_days_together)
  let days_A_alone := remaining_work * total_days_A
  days_A_alone = 21 := by
sorry

end days_A_worked_alone_l1312_131244


namespace number_division_problem_l1312_131217

theorem number_division_problem :
  ∃ x : ℝ, (x / 9 + x + 9 = 69) ∧ x = 54 := by
  sorry

end number_division_problem_l1312_131217


namespace journey_distance_l1312_131203

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 10 ∧ speed1 = 21 ∧ speed2 = 24 →
  ∃ (distance : ℝ), distance = 224 ∧
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 := by
  sorry

end journey_distance_l1312_131203


namespace d_bounds_l1312_131207

/-- The maximum number of black squares on an n × n board where each black square
    has exactly two neighboring black squares. -/
def d (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for d(n) -/
theorem d_bounds (n : ℕ) : 
  (2/3 : ℝ) * n^2 - 8 * n ≤ (d n : ℝ) ∧ (d n : ℝ) ≤ (2/3 : ℝ) * n^2 + 4 * n :=
sorry

end d_bounds_l1312_131207


namespace identify_six_genuine_coins_l1312_131280

/-- Represents the result of a weighing on a balance scale -/
inductive WeighResult
| Equal : WeighResult
| LeftHeavier : WeighResult
| RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  total : Nat
  genuine : Nat
  counterfeit : Nat

/-- Represents a weighing action on the balance scale -/
def weigh (left right : CoinGroup) : WeighResult :=
  sorry

/-- Represents the process of identifying genuine coins -/
def identifyGenuineCoins (coins : CoinGroup) (maxWeighings : Nat) : Option (Fin 6 → Bool) :=
  sorry

theorem identify_six_genuine_coins :
  ∀ (coins : CoinGroup),
    coins.total = 25 →
    coins.genuine = 22 →
    coins.counterfeit = 3 →
    ∃ (result : Fin 6 → Bool),
      identifyGenuineCoins coins 2 = some result ∧
      (∀ i, result i = true → i.val < 6) :=
by
  sorry

end identify_six_genuine_coins_l1312_131280


namespace quadratic_equations_solutions_l1312_131205

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3) ↔ x = -3/2 ∨ x = 1/2) := by
  sorry

end quadratic_equations_solutions_l1312_131205


namespace coin_stack_order_l1312_131254

-- Define the type for coins
inductive Coin | A | B | C | D | E

-- Define the covering relation
def covers (x y : Coin) : Prop := sorry

-- Define the partial covering relation
def partially_covers (x y : Coin) : Prop := sorry

-- Define the order relation
def above (x y : Coin) : Prop := sorry

-- State the theorem
theorem coin_stack_order :
  (partially_covers Coin.A Coin.B) →
  (covers Coin.C Coin.A) →
  (covers Coin.C Coin.D) →
  (covers Coin.D Coin.B) →
  (¬ covers Coin.D Coin.E) →
  (covers Coin.C Coin.E) →
  (∀ x, ¬ covers Coin.E x) →
  (above Coin.C Coin.E) ∧
  (above Coin.E Coin.A) ∧
  (above Coin.E Coin.D) ∧
  (above Coin.A Coin.B) ∧
  (above Coin.D Coin.B) :=
by sorry

end coin_stack_order_l1312_131254


namespace sin_two_phi_value_l1312_131239

theorem sin_two_phi_value (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_two_phi_value_l1312_131239


namespace jan_height_is_42_l1312_131253

def cary_height : ℕ := 72

def bill_height : ℕ := cary_height / 2

def jan_height : ℕ := bill_height + 6

theorem jan_height_is_42 : jan_height = 42 := by
  sorry

end jan_height_is_42_l1312_131253


namespace winston_initial_gas_l1312_131273

/-- The amount of gas in gallons used for a trip -/
structure Trip where
  gas_used : ℝ

/-- The gas tank of a car -/
structure GasTank where
  capacity : ℝ
  initial_amount : ℝ
  remaining_amount : ℝ

/-- Winston's car trips and gas tank -/
def winston_scenario (store_trip doctor_trip : Trip) (tank : GasTank) : Prop :=
  store_trip.gas_used = 6 ∧
  doctor_trip.gas_used = 2 ∧
  tank.capacity = 12 ∧
  tank.initial_amount = tank.remaining_amount + store_trip.gas_used + doctor_trip.gas_used ∧
  tank.remaining_amount > 0 ∧
  tank.initial_amount ≤ tank.capacity

theorem winston_initial_gas 
  (store_trip doctor_trip : Trip) (tank : GasTank) 
  (h : winston_scenario store_trip doctor_trip tank) : 
  tank.initial_amount = 12 :=
sorry

end winston_initial_gas_l1312_131273


namespace power_expression_l1312_131259

theorem power_expression (m n : ℕ+) (a b : ℝ) 
  (h1 : 9^(m : ℕ) = a) 
  (h2 : 3^(n : ℕ) = b) : 
  3^((2*m + 4*n) : ℕ) = a * b^4 := by
  sorry

end power_expression_l1312_131259


namespace eleven_items_division_l1312_131293

theorem eleven_items_division (n : ℕ) (h : n = 11) : 
  (Finset.sum (Finset.range 3) (λ k => Nat.choose n (k + 3))) = 957 := by
  sorry

end eleven_items_division_l1312_131293


namespace average_of_x_and_y_is_16_l1312_131233

theorem average_of_x_and_y_is_16 (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.25 * y → (x + y) / 2 = 16 := by
  sorry

end average_of_x_and_y_is_16_l1312_131233


namespace pizza_pooling_advantage_l1312_131248

/-- Represents the size and price of a pizza --/
structure Pizza where
  side : ℕ
  price : ℕ

/-- Calculates the area of a square pizza --/
def pizzaArea (p : Pizza) : ℕ := p.side * p.side

/-- Represents the pizza options and money available --/
structure PizzaShop where
  smallPizza : Pizza
  largePizza : Pizza
  moneyPerPerson : ℕ
  numPeople : ℕ

/-- Calculates the maximum area of pizza that can be bought individually --/
def maxIndividualArea (shop : PizzaShop) : ℕ :=
  let smallArea := (shop.moneyPerPerson / shop.smallPizza.price) * pizzaArea shop.smallPizza
  let largeArea := (shop.moneyPerPerson / shop.largePizza.price) * pizzaArea shop.largePizza
  max smallArea largeArea * shop.numPeople

/-- Calculates the maximum area of pizza that can be bought by pooling money --/
def maxPooledArea (shop : PizzaShop) : ℕ :=
  let totalMoney := shop.moneyPerPerson * shop.numPeople
  let smallArea := (totalMoney / shop.smallPizza.price) * pizzaArea shop.smallPizza
  let largeArea := (totalMoney / shop.largePizza.price) * pizzaArea shop.largePizza
  max smallArea largeArea

theorem pizza_pooling_advantage (shop : PizzaShop) 
    (h1 : shop.smallPizza = ⟨6, 10⟩)
    (h2 : shop.largePizza = ⟨9, 20⟩)
    (h3 : shop.moneyPerPerson = 30)
    (h4 : shop.numPeople = 2) :
  maxPooledArea shop - maxIndividualArea shop = 27 := by
  sorry


end pizza_pooling_advantage_l1312_131248


namespace cricket_team_age_theorem_l1312_131287

def cricket_team_age_problem (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ) : Prop :=
  let total_age := team_size * average_age
  let captain_and_keeper_age := captain_age + (captain_age + wicket_keeper_age_diff)
  let remaining_players := team_size - 2
  total_age - captain_and_keeper_age = remaining_players * (average_age - remaining_players_age_diff)
  where
    average_age : ℕ := 23

theorem cricket_team_age_theorem :
  cricket_team_age_problem 11 26 3 1 := by
  sorry

end cricket_team_age_theorem_l1312_131287


namespace ceiling_of_negative_real_l1312_131278

theorem ceiling_of_negative_real : ⌈(-3.67 : ℝ)⌉ = -3 := by sorry

end ceiling_of_negative_real_l1312_131278


namespace smallest_n_with_four_pairs_l1312_131286

/-- Returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n -/
def f (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 26 is the smallest positive integer n for which f(n) = 4 -/
theorem smallest_n_with_four_pairs : ∀ m : ℕ, m > 0 → m < 26 → f m ≠ 4 ∧ f 26 = 4 := by
  sorry

end smallest_n_with_four_pairs_l1312_131286


namespace fifth_root_of_3125_l1312_131252

theorem fifth_root_of_3125 (x : ℝ) (h1 : x > 0) (h2 : x^5 = 3125) : x = 5 := by
  sorry

end fifth_root_of_3125_l1312_131252


namespace point_on_line_segment_l1312_131247

def A (m : ℝ) : ℝ × ℝ := (m^2, 2)
def B (m : ℝ) : ℝ × ℝ := (2*m^2 + 2, 2)
def M (m : ℝ) : ℝ × ℝ := (-m^2, 2)
def N (m : ℝ) : ℝ × ℝ := (m^2, m^2 + 2)
def P (m : ℝ) : ℝ × ℝ := (m^2 + 1, 2)
def Q (m : ℝ) : ℝ × ℝ := (3*m^2, 2)

theorem point_on_line_segment (m : ℝ) :
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → M m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → N m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → Q m = ((1 - t) • (A m) + t • (B m))) :=
by sorry

end point_on_line_segment_l1312_131247


namespace decorative_band_length_l1312_131228

/-- The length of a decorative band for a circular sign -/
theorem decorative_band_length :
  let π : ℚ := 22 / 7
  let area : ℚ := 616
  let extra_length : ℚ := 5
  let radius : ℚ := (area / π).sqrt
  let circumference : ℚ := 2 * π * radius
  let band_length : ℚ := circumference + extra_length
  band_length = 93 := by sorry

end decorative_band_length_l1312_131228


namespace target_equals_fraction_l1312_131291

/-- The decimal representation of a rational number -/
def decimal_rep (q : ℚ) : ℕ → ℕ := sorry

/-- A function that checks if a decimal representation is repeating -/
def is_repeating (d : ℕ → ℕ) : Prop := sorry

/-- The rational number represented by 0.2̄34 -/
def target : ℚ := sorry

theorem target_equals_fraction : 
  (is_repeating (decimal_rep target)) → 
  (∀ a b : ℤ, (a / b : ℚ) = target → ∃ k : ℤ, k * 116 = a ∧ k * 495 = b) →
  target = 116 / 495 := by sorry

end target_equals_fraction_l1312_131291


namespace distance_midway_to_new_city_l1312_131230

theorem distance_midway_to_new_city : 
  let new_city : ℂ := 0
  let old_town : ℂ := 3200 * I
  let midway : ℂ := 960 + 1280 * I
  Complex.abs (midway - new_city) = 3200 := by
sorry

end distance_midway_to_new_city_l1312_131230


namespace square_area_multiple_l1312_131269

theorem square_area_multiple (a p m : ℝ) : 
  a > 0 → 
  p > 0 → 
  p = 36 → 
  a = (p / 4)^2 → 
  m * a = 10 * p + 45 → 
  m = 5 := by
sorry

end square_area_multiple_l1312_131269


namespace two_red_two_blue_probability_l1312_131299

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

def probability_two_red_two_blue : ℚ :=
  6 * (red_marbles * (red_marbles - 1) * blue_marbles * (blue_marbles - 1)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem two_red_two_blue_probability :
  probability_two_red_two_blue = 1232 / 4845 := by
  sorry

end two_red_two_blue_probability_l1312_131299


namespace trapezoid_area_l1312_131255

/-- A trapezoid with given side lengths -/
structure Trapezoid :=
  (BC : ℝ)
  (AD : ℝ)
  (AB : ℝ)
  (CD : ℝ)

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem: The area of the given trapezoid is 59 -/
theorem trapezoid_area :
  let t : Trapezoid := { BC := 9.5, AD := 20, AB := 5, CD := 8.5 }
  area t = 59 := by sorry

end trapezoid_area_l1312_131255


namespace division_remainder_l1312_131221

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^60 + x^45 + x^30 + x^15 + 1

/-- Theorem stating that the remainder of the division is 5 -/
theorem division_remainder : ∃ (q : ℂ → ℂ), ∀ (x : ℂ), 
  dividend x = (divisor x) * (q x) + 5 := by
  sorry

end division_remainder_l1312_131221


namespace intersection_M_N_l1312_131284

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (4 - x^2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc (-1) 2 := by
  sorry

end intersection_M_N_l1312_131284


namespace mollys_age_l1312_131225

/-- Molly's birthday candle problem -/
theorem mollys_age (initial_candles additional_candles : ℕ) 
  (h1 : initial_candles = 14)
  (h2 : additional_candles = 6) :
  initial_candles + additional_candles = 20 := by
  sorry

end mollys_age_l1312_131225


namespace bruce_payment_l1312_131260

/-- The total amount Bruce paid to the shopkeeper -/
def total_amount (grape_quantity mangoe_quantity grape_rate mangoe_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mangoe_quantity * mangoe_rate

/-- Theorem stating that Bruce paid 1000 to the shopkeeper -/
theorem bruce_payment :
  total_amount 8 8 70 55 = 1000 := by
  sorry

end bruce_payment_l1312_131260


namespace sum_of_factors_24_l1312_131266

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_24 :
  (factors 24).sum id = 60 := by
  sorry

end sum_of_factors_24_l1312_131266


namespace no_prime_satisfies_condition_l1312_131229

theorem no_prime_satisfies_condition : ¬∃ (P : ℕ), Prime P ∧ (100 : ℚ) * P = P + (1386 : ℚ) / 10 := by
  sorry

end no_prime_satisfies_condition_l1312_131229


namespace max_sum_of_squares_l1312_131245

theorem max_sum_of_squares (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 5) :
  ∃ M : ℝ, M = 20 ∧ ∀ x y z w : ℝ, x^2 + y^2 + z^2 + w^2 = 5 →
    (x - y)^2 + (x - z)^2 + (x - w)^2 + (y - z)^2 + (y - w)^2 + (z - w)^2 ≤ M :=
by sorry

end max_sum_of_squares_l1312_131245


namespace total_points_is_201_l1312_131288

/- Define the scoring for Mark's team -/
def marks_team_two_pointers : ℕ := 25
def marks_team_three_pointers : ℕ := 8
def marks_team_free_throws : ℕ := 10

/- Define the scoring for the opponents relative to Mark's team -/
def opponents_two_pointers : ℕ := 2 * marks_team_two_pointers
def opponents_three_pointers : ℕ := marks_team_three_pointers / 2
def opponents_free_throws : ℕ := marks_team_free_throws / 2

/- Calculate the total points for both teams -/
def total_points : ℕ := 
  (marks_team_two_pointers * 2 + marks_team_three_pointers * 3 + marks_team_free_throws) +
  (opponents_two_pointers * 2 + opponents_three_pointers * 3 + opponents_free_throws)

/- Theorem stating that the total points scored by both teams is 201 -/
theorem total_points_is_201 : total_points = 201 := by
  sorry

end total_points_is_201_l1312_131288


namespace fraction_equation_solutions_l1312_131265

theorem fraction_equation_solutions (x : ℝ) : 
  1 / (x^2 + 17*x - 8) + 1 / (x^2 + 4*x - 8) + 1 / (x^2 - 9*x - 8) = 0 ↔ 
  x = 1 ∨ x = -8 ∨ x = 2 ∨ x = -4 := by
  sorry

end fraction_equation_solutions_l1312_131265


namespace parabola_point_relationship_l1312_131204

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + x + 2

-- Define the points on the parabola
def point_a : ℝ × ℝ := (2, f 2)
def point_b : ℝ × ℝ := (-1, f (-1))
def point_c : ℝ × ℝ := (3, f 3)

-- Theorem stating the relationship between a, b, and c
theorem parabola_point_relationship :
  point_c.2 > point_a.2 ∧ point_a.2 > point_b.2 :=
sorry

end parabola_point_relationship_l1312_131204


namespace odd_sum_of_odd_square_plus_cube_l1312_131208

theorem odd_sum_of_odd_square_plus_cube (n m : ℤ) : 
  Odd (n^2 + m^3) → Odd (n + m) := by
  sorry

end odd_sum_of_odd_square_plus_cube_l1312_131208


namespace five_circles_common_point_l1312_131214

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define a function to check if four circles pass through a single point
def fourCirclesCommonPoint (c1 c2 c3 c4 : Circle) : Prop :=
  ∃ p : Point, pointOnCircle p c1 ∧ pointOnCircle p c2 ∧ pointOnCircle p c3 ∧ pointOnCircle p c4

-- Theorem statement
theorem five_circles_common_point 
  (c1 c2 c3 c4 c5 : Circle) 
  (h1234 : fourCirclesCommonPoint c1 c2 c3 c4)
  (h1235 : fourCirclesCommonPoint c1 c2 c3 c5)
  (h1245 : fourCirclesCommonPoint c1 c2 c4 c5)
  (h1345 : fourCirclesCommonPoint c1 c3 c4 c5)
  (h2345 : fourCirclesCommonPoint c2 c3 c4 c5) :
  ∃ p : Point, pointOnCircle p c1 ∧ pointOnCircle p c2 ∧ pointOnCircle p c3 ∧ pointOnCircle p c4 ∧ pointOnCircle p c5 :=
by
  sorry

end five_circles_common_point_l1312_131214


namespace complex_magnitude_l1312_131222

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 10)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 6 := by sorry

end complex_magnitude_l1312_131222


namespace solution_set_inequality_l1312_131237

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1/2 < x ∧ x ≤ 1 := by sorry

end solution_set_inequality_l1312_131237


namespace smallest_valid_perfect_square_l1312_131219

def is_valid (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 10, n % (k + 2) = k + 1

theorem smallest_valid_perfect_square : 
  ∃ n : ℕ, n = 2782559 ∧ 
    is_valid n ∧ 
    ∃ m : ℕ, n = m^2 ∧ 
    ∀ k : ℕ, k < n → ¬(is_valid k ∧ ∃ m : ℕ, k = m^2) :=
by sorry

end smallest_valid_perfect_square_l1312_131219


namespace train_crossing_time_l1312_131298

/-- Time for a train to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 50 → 
  train_speed_kmh = 360 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 0.5 := by
  sorry

end train_crossing_time_l1312_131298


namespace perpendicular_vectors_m_l1312_131243

-- Define the vectors
def a : Fin 2 → ℝ := ![1, -3]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 2]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Define perpendicularity
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  dot_product u v = 0

-- State the theorem
theorem perpendicular_vectors_m (m : ℝ) :
  perpendicular a (fun i => a i + b m i) → m = -4 := by
  sorry

end perpendicular_vectors_m_l1312_131243


namespace integral_x_plus_x_squared_plus_sin_x_l1312_131231

theorem integral_x_plus_x_squared_plus_sin_x : 
  ∫ x in (-1 : ℝ)..1, (x + x^2 + Real.sin x) = 2/3 := by sorry

end integral_x_plus_x_squared_plus_sin_x_l1312_131231


namespace sum_of_digits_of_seven_to_fifteen_l1312_131250

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_seven_to_fifteen (n : ℕ) (h : n = (3 + 4)^15) :
  tens_digit (last_two_digits n) + ones_digit (last_two_digits n) = 7 := by
  sorry

end sum_of_digits_of_seven_to_fifteen_l1312_131250


namespace garden_fence_posts_l1312_131275

/-- Calculates the number of fence posts required for a rectangular garden -/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let postsOnPerimeter := perimeter / postSpacing
  postsOnPerimeter + 1

/-- Theorem stating the number of fence posts required for the specific garden -/
theorem garden_fence_posts :
  fencePostsRequired 72 32 8 = 26 := by
  sorry

end garden_fence_posts_l1312_131275


namespace perimeter_after_adding_tiles_l1312_131209

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  length : ℕ
  width : ℕ
  perimeter : ℕ

/-- The initial configuration of tiles -/
def initial_config : TileConfiguration :=
  { length := 6, width := 1, perimeter := 14 }

/-- Calculates the new perimeter after adding tiles -/
def new_perimeter (config : TileConfiguration) (added_tiles : ℕ) : ℕ :=
  2 * (config.length + added_tiles) + 2 * config.width

/-- Theorem stating that adding two tiles results in a perimeter of 18 -/
theorem perimeter_after_adding_tiles :
  new_perimeter initial_config 2 = 18 := by
  sorry

#eval new_perimeter initial_config 2

end perimeter_after_adding_tiles_l1312_131209


namespace income_expenditure_ratio_l1312_131210

/-- Given a person's income and savings, prove the ratio of income to expenditure -/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (h1 : income = 36000) 
  (h2 : savings = 4000) :
  (income : ℚ) / (income - savings) = 9 / 8 := by
  sorry

end income_expenditure_ratio_l1312_131210


namespace equal_roots_quadratic_l1312_131227

/-- 
Given a quadratic equation 3x^2 + 6x + m = 0, if it has two equal real roots,
then m = 3.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + 6 * x + m = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + 6 * y + m = 0 → y = x) → 
  m = 3 := by sorry

end equal_roots_quadratic_l1312_131227


namespace cos_alpha_value_l1312_131262

theorem cos_alpha_value (α : Real) (h : Real.sin (α - Real.pi/2) = 3/5) :
  Real.cos α = -3/5 := by
  sorry

end cos_alpha_value_l1312_131262


namespace set_union_problem_l1312_131220

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {2, 3} →
  B = {1, a} →
  A ∩ B = {2} →
  A ∪ B = {1, 2, 3} := by
sorry

end set_union_problem_l1312_131220


namespace parabola_standard_equation_l1312_131206

/-- The standard equation of a parabola with directrix x = 1 -/
theorem parabola_standard_equation (x y : ℝ) :
  (∃ (p : ℝ), p > 0 ∧ 1 = p / 2 ∧ x < -p / 2) →
  (∀ point : ℝ × ℝ, point ∈ {(x, y) | y^2 = -4*x} ↔
    dist point (x, 0) = dist point (1, (point.2))) :=
by sorry

end parabola_standard_equation_l1312_131206


namespace tangent_ellipse_solution_l1312_131281

/-- An ellipse with semi-major axis a and semi-minor axis b that is tangent to a rectangle with area 48 -/
structure TangentEllipse where
  a : ℝ
  b : ℝ
  area_eq : a * b = 12
  a_pos : a > 0
  b_pos : b > 0

/-- The theorem stating that the ellipse with a = 4 and b = 3 satisfies the conditions -/
theorem tangent_ellipse_solution :
  ∃ (e : TangentEllipse), e.a = 4 ∧ e.b = 3 := by
  sorry

end tangent_ellipse_solution_l1312_131281


namespace sin_210_degrees_l1312_131201

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l1312_131201


namespace set_equality_implies_value_l1312_131234

theorem set_equality_implies_value (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2012 + b^2012 = 1 :=
by sorry

end set_equality_implies_value_l1312_131234


namespace area_ratio_is_three_fiftieths_l1312_131276

/-- A large square subdivided into 25 equal smaller squares -/
structure LargeSquare :=
  (side_length : ℝ)
  (num_subdivisions : ℕ)
  (h_subdivisions : num_subdivisions = 25)

/-- A shaded region formed by connecting midpoints of sides of five smaller squares -/
structure ShadedRegion :=
  (large_square : LargeSquare)
  (num_squares : ℕ)
  (h_num_squares : num_squares = 5)

/-- The ratio of the area of the shaded region to the area of the large square -/
def area_ratio (sr : ShadedRegion) : ℚ :=
  3 / 50

/-- Theorem stating that the area ratio is 3/50 -/
theorem area_ratio_is_three_fiftieths (sr : ShadedRegion) :
  area_ratio sr = 3 / 50 := by
  sorry

end area_ratio_is_three_fiftieths_l1312_131276


namespace train_length_calculation_l1312_131251

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) :
  train_speed = 30 →
  time_to_pass = 60 →
  bridge_length = 140 →
  ∃ (train_length : ℝ), abs (train_length - 359.8) < 0.1 :=
by
  sorry

end train_length_calculation_l1312_131251


namespace quadratic_roots_identity_l1312_131274

theorem quadratic_roots_identity (p q r s k : ℝ) (α β : ℝ) : 
  (α^2 + p*α + q = 0) →
  (β^2 + r*β + s = 0) →
  (α / β = k) →
  (q - k^2 * s)^2 + k * (p - k * r) * (k * p * s - q * r) = 0 := by
  sorry

end quadratic_roots_identity_l1312_131274


namespace lottery_probability_l1312_131223

theorem lottery_probability (total_tickets : Nat) (winning_tickets : Nat) (buyers : Nat) :
  total_tickets = 10 →
  winning_tickets = 3 →
  buyers = 5 →
  let prob_at_least_one_wins := 1 - (Nat.choose (total_tickets - winning_tickets) buyers / Nat.choose total_tickets buyers)
  prob_at_least_one_wins = 77 / 84 := by
  sorry

end lottery_probability_l1312_131223


namespace zoo_animals_count_l1312_131285

theorem zoo_animals_count (female_count : ℕ) (male_excess : ℕ) : 
  female_count = 35 → male_excess = 7 → female_count + (female_count + male_excess) = 77 := by
  sorry

end zoo_animals_count_l1312_131285


namespace sum_of_rectangle_areas_l1312_131215

def rectangle_width : ℕ := 3

def odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11, 13]

def rectangle_lengths : List ℕ := odd_numbers.map (λ x => x * x)

def rectangle_areas : List ℕ := rectangle_lengths.map (λ x => rectangle_width * x)

theorem sum_of_rectangle_areas :
  rectangle_areas.sum = 1365 := by sorry

end sum_of_rectangle_areas_l1312_131215


namespace multiply_six_and_mixed_number_l1312_131218

theorem multiply_six_and_mixed_number : 6 * (8 + 1/3) = 50 := by
  sorry

end multiply_six_and_mixed_number_l1312_131218


namespace min_xy_value_l1312_131289

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3*x*y - x - y - 1 = 0) :
  ∀ z, z = x*y → z ≥ 1 :=
by sorry

end min_xy_value_l1312_131289


namespace corner_cut_pentagon_area_l1312_131249

/-- Pentagon formed by cutting a triangular corner from a rectangle --/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {14, 21, 22, 28, 37}

/-- The area of the CornerCutPentagon --/
def pentagon_area (p : CornerCutPentagon) : ℕ := sorry

/-- Theorem stating that the area of the CornerCutPentagon is 826 --/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : pentagon_area p = 826 := by sorry

end corner_cut_pentagon_area_l1312_131249


namespace mile_to_rod_l1312_131264

-- Define the units
def mile : ℕ := 1
def furlong : ℕ := 1
def rod : ℕ := 1

-- Define the conversion rates
axiom mile_to_furlong : mile = 6 * furlong
axiom furlong_to_rod : furlong = 60 * rod

-- Theorem to prove
theorem mile_to_rod : mile = 360 * rod := by
  sorry

end mile_to_rod_l1312_131264


namespace fraction_equality_l1312_131292

theorem fraction_equality (x y : ℝ) 
  (h : (1/3)^2 + (1/4)^2 / ((1/5)^2 + (1/6)^2) = 37*x / (73*y)) : 
  Real.sqrt x / Real.sqrt y = 75 * Real.sqrt 73 / (6 * Real.sqrt 61 * Real.sqrt 37) := by
  sorry

end fraction_equality_l1312_131292


namespace inequality_solution_set_l1312_131240

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x + 1 < a^2 + x}
  (a > 1 → S = {x : ℝ | x < a + 1}) ∧
  (a < 1 → S = {x : ℝ | x > a + 1}) ∧
  (a = 1 → S = ∅) :=
by sorry

end inequality_solution_set_l1312_131240


namespace square_perimeter_side_ratio_l1312_131283

theorem square_perimeter_side_ratio (s : ℝ) (hs : s > 0) :
  let new_side := s + 1
  let new_perimeter := 4 * new_side
  new_perimeter / new_side = 4 := by
sorry

end square_perimeter_side_ratio_l1312_131283


namespace sqrt_123400_l1312_131277

theorem sqrt_123400 (h1 : Real.sqrt 12.34 = 3.512) (h2 : Real.sqrt 123.4 = 11.108) :
  Real.sqrt 123400 = 351.2 := by
  sorry

end sqrt_123400_l1312_131277


namespace jessy_jokes_count_l1312_131261

theorem jessy_jokes_count (jessy_jokes alan_jokes : ℕ) : 
  alan_jokes = 7 →
  2 * (jessy_jokes + alan_jokes) = 54 →
  jessy_jokes = 20 := by
sorry

end jessy_jokes_count_l1312_131261


namespace unique_five_digit_number_exists_l1312_131295

/-- Represents a 5-digit number with different non-zero digits -/
structure FiveDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  d_nonzero : d ≠ 0
  e_nonzero : e ≠ 0
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Checks if the sum of the shifted additions equals a 7-digit number with all digits A -/
def isValidSum (n : FiveDigitNumber) : Prop :=
  let sum := n.a * 1000000 + n.b * 100000 + n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.b * 100000 + n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.e * 100 + n.d * 10 + n.b +
             n.d * 10 + n.b +
             n.b
  sum = n.a * 1111111

theorem unique_five_digit_number_exists : ∃! n : FiveDigitNumber, isValidSum n ∧ n.a = 8 ∧ n.b = 4 ∧ n.c = 2 ∧ n.d = 6 ∧ n.e = 9 := by
  sorry

end unique_five_digit_number_exists_l1312_131295


namespace credit_rating_equation_l1312_131246

theorem credit_rating_equation (x : ℝ) : 
  (96 : ℝ) = x * (1 + 0.2) ↔ 
  (96 : ℝ) = x + x * 0.2 := by sorry

end credit_rating_equation_l1312_131246


namespace square_not_sum_of_periodic_l1312_131272

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem square_not_sum_of_periodic :
  ¬ ∃ (g h : ℝ → ℝ), (Periodic g ∧ Periodic h) ∧ (∀ x : ℝ, x^2 = g x + h x) := by
  sorry

end square_not_sum_of_periodic_l1312_131272


namespace modulus_of_imaginary_unit_l1312_131232

theorem modulus_of_imaginary_unit (z : ℂ) (h : z^2 + 1 = 0) : Complex.abs z = 1 := by
  sorry

end modulus_of_imaginary_unit_l1312_131232


namespace intersection_equals_interval_l1312_131282

open Set

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_equals_interval : S ∩ T = Ioc (-2) 1 := by sorry

end intersection_equals_interval_l1312_131282


namespace sample_size_is_40_l1312_131241

/-- Represents a frequency distribution histogram -/
structure Histogram where
  num_bars : ℕ
  central_freq : ℕ
  other_freq : ℕ

/-- Calculates the sample size of a histogram -/
def sample_size (h : Histogram) : ℕ :=
  h.central_freq + h.other_freq

/-- Theorem stating the sample size for the given histogram -/
theorem sample_size_is_40 (h : Histogram) 
  (h_bars : h.num_bars = 7)
  (h_central : h.central_freq = 8)
  (h_ratio : h.central_freq = h.other_freq / 4) :
  sample_size h = 40 := by
  sorry

#check sample_size_is_40

end sample_size_is_40_l1312_131241


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1312_131213

/-- An isosceles triangle with side lengths 5 and 10 has a perimeter of 25 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 5 ∨ a = 10) ∧ 
    (b = 5 ∨ b = 10) ∧ 
    (c = 5 ∨ c = 10) ∧ 
    (a = b ∨ b = c ∨ a = c) ∧ 
    (a + b > c ∧ b + c > a ∧ a + c > b) →
    a + b + c = 25

theorem isosceles_triangle_perimeter_proof : ∃ a b c : ℝ, isosceles_triangle_perimeter a b c := by
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1312_131213


namespace percentage_of_women_in_non_union_l1312_131236

theorem percentage_of_women_in_non_union (total_employees : ℝ) 
  (h1 : total_employees > 0)
  (h2 : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ p * total_employees = number_of_male_employees)
  (h3 : 0.6 * total_employees = number_of_unionized_employees)
  (h4 : 0.7 * number_of_unionized_employees = number_of_male_unionized_employees)
  (h5 : 0.9 * (total_employees - number_of_unionized_employees) = number_of_female_non_unionized_employees) :
  (number_of_female_non_unionized_employees / (total_employees - number_of_unionized_employees)) = 0.9 := by
sorry


end percentage_of_women_in_non_union_l1312_131236


namespace output_for_15_l1312_131296

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end output_for_15_l1312_131296


namespace greatest_common_multiple_under_150_l1312_131257

theorem greatest_common_multiple_under_150 (n : ℕ) :
  (n % 10 = 0 ∧ n % 15 = 0 ∧ n < 150) →
  n ≤ 120 :=
by sorry

end greatest_common_multiple_under_150_l1312_131257


namespace age_ratio_proof_l1312_131238

theorem age_ratio_proof (man_age wife_age : ℕ) : 
  man_age = 30 →
  wife_age = 30 →
  man_age - 10 = wife_age →
  ∃ k : ℚ, man_age = k * (wife_age - 10) →
  (man_age : ℚ) / (wife_age - 10 : ℚ) = 3 / 2 := by
  sorry

end age_ratio_proof_l1312_131238


namespace people_visited_neither_l1312_131290

theorem people_visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 100 →
  iceland = 55 →
  norway = 43 →
  both = 61 →
  total - (iceland + norway - both) = 63 :=
by sorry

end people_visited_neither_l1312_131290


namespace nina_total_spent_l1312_131211

/-- The total amount spent by Nina on toys, basketball cards, and shirts. -/
def total_spent (toy_quantity : ℕ) (toy_price : ℕ) (card_quantity : ℕ) (card_price : ℕ) (shirt_quantity : ℕ) (shirt_price : ℕ) : ℕ :=
  toy_quantity * toy_price + card_quantity * card_price + shirt_quantity * shirt_price

/-- Theorem stating that Nina's total spent is $70 -/
theorem nina_total_spent :
  total_spent 3 10 2 5 5 6 = 70 := by
  sorry

end nina_total_spent_l1312_131211


namespace distance_between_homes_is_40_l1312_131200

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 40

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 5

/-- The distance Maxwell travels before they meet -/
def maxwell_distance : ℝ := 15

/-- Theorem stating that the distance between homes is 40 km -/
theorem distance_between_homes_is_40 :
  distance_between_homes = maxwell_distance * (maxwell_speed + brad_speed) / maxwell_speed :=
by sorry

end distance_between_homes_is_40_l1312_131200


namespace cookie_count_l1312_131216

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end cookie_count_l1312_131216


namespace no_integer_square_root_l1312_131270

/-- The polynomial p(x) = x^4 + 6x^3 + 11x^2 + 13x + 37 -/
def p (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 13*x + 37

/-- Theorem stating that there are no integer values of x such that p(x) is a perfect square -/
theorem no_integer_square_root : ∀ x : ℤ, ¬∃ y : ℤ, p x = y^2 := by
  sorry

end no_integer_square_root_l1312_131270


namespace smallest_valid_n_l1312_131263

def is_valid (n : ℕ) : Prop :=
  ∃ k : ℕ, 17 * n - 1 = 11 * k

theorem smallest_valid_n :
  ∃ n : ℕ, n > 0 ∧ is_valid n ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬is_valid m :=
by sorry

end smallest_valid_n_l1312_131263


namespace qiannan_establishment_year_l1312_131267

/-- Represents the Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Heavenly Stems and Earthly Branches system -/
structure StemBranchYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- Function to get the previous stem -/
def prevStem (s : HeavenlyStem) : HeavenlyStem :=
  match s with
  | HeavenlyStem.Jia => HeavenlyStem.Gui
  | HeavenlyStem.Yi => HeavenlyStem.Jia
  | HeavenlyStem.Bing => HeavenlyStem.Yi
  | HeavenlyStem.Ding => HeavenlyStem.Bing
  | HeavenlyStem.Wu => HeavenlyStem.Ding
  | HeavenlyStem.Ji => HeavenlyStem.Wu
  | HeavenlyStem.Geng => HeavenlyStem.Ji
  | HeavenlyStem.Xin => HeavenlyStem.Geng
  | HeavenlyStem.Ren => HeavenlyStem.Xin
  | HeavenlyStem.Gui => HeavenlyStem.Ren

/-- Function to get the previous branch -/
def prevBranch (b : EarthlyBranch) : EarthlyBranch :=
  match b with
  | EarthlyBranch.Zi => EarthlyBranch.Hai
  | EarthlyBranch.Chou => EarthlyBranch.Zi
  | EarthlyBranch.Yin => EarthlyBranch.Chou
  | EarthlyBranch.Mao => EarthlyBranch.Yin
  | EarthlyBranch.Chen => EarthlyBranch.Mao
  | EarthlyBranch.Si => EarthlyBranch.Chen
  | EarthlyBranch.Wu => EarthlyBranch.Si
  | EarthlyBranch.Wei => EarthlyBranch.Wu
  | EarthlyBranch.Shen => EarthlyBranch.Wei
  | EarthlyBranch.You => EarthlyBranch.Shen
  | EarthlyBranch.Xu => EarthlyBranch.You
  | EarthlyBranch.Hai => EarthlyBranch.Xu

/-- Function to get the year n years before a given year -/
def yearsBefore (n : Nat) (year : StemBranchYear) : StemBranchYear :=
  if n = 0 then year
  else yearsBefore (n - 1) { stem := prevStem year.stem, branch := prevBranch year.branch }

theorem qiannan_establishment_year :
  let year2023 : StemBranchYear := { stem := HeavenlyStem.Gui, branch := EarthlyBranch.Mao }
  let establishmentYear := yearsBefore 67 year2023
  establishmentYear.stem = HeavenlyStem.Bing ∧ establishmentYear.branch = EarthlyBranch.Shen :=
by sorry

end qiannan_establishment_year_l1312_131267


namespace min_games_prediction_l1312_131294

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  white_rook : ℕ  -- Number of students from "White Rook" school
  black_elephant : ℕ  -- Number of students from "Black Elephant" school
  total_games : ℕ  -- Total number of games to be played

/-- Predicate to check if a tournament setup is valid -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.white_rook * t.black_elephant = t.total_games

/-- The minimum number of games after which one can definitely name a participant -/
def min_games_to_predict (t : ChessTournament) : ℕ :=
  t.total_games - t.black_elephant

/-- Theorem stating the minimum number of games for prediction in the given tournament -/
theorem min_games_prediction (t : ChessTournament) 
  (h_valid : valid_tournament t) 
  (h_white : t.white_rook = 15) 
  (h_black : t.black_elephant = 20) 
  (h_total : t.total_games = 300) : 
  min_games_to_predict t = 280 := by
  sorry

#eval min_games_to_predict { white_rook := 15, black_elephant := 20, total_games := 300 }

end min_games_prediction_l1312_131294


namespace right_triangle_area_l1312_131226

/-- Given a right triangle ABC with legs a and b, and hypotenuse c,
    if a + b = 21 and c = 15, then the area of triangle ABC is 54. -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b = 21 → 
  c = 15 → 
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 54 := by
  sorry

end right_triangle_area_l1312_131226


namespace bears_distribution_l1312_131212

def bears_per_shelf (initial_stock new_shipment num_shelves : ℕ) : ℕ :=
  (initial_stock + new_shipment) / num_shelves

theorem bears_distribution (initial_stock new_shipment num_shelves : ℕ) 
  (h1 : initial_stock = 17)
  (h2 : new_shipment = 10)
  (h3 : num_shelves = 3) :
  bears_per_shelf initial_stock new_shipment num_shelves = 9 := by
  sorry

end bears_distribution_l1312_131212


namespace monotonicity_and_extrema_l1312_131256

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

theorem monotonicity_and_extrema :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 1 ∧ y > 1)) → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ 2) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = 2) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≥ -18) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = -18) :=
by sorry

end monotonicity_and_extrema_l1312_131256


namespace min_words_for_90_percent_l1312_131268

/-- The minimum number of words needed to achieve at least 90% on a vocabulary exam -/
theorem min_words_for_90_percent (total_words : ℕ) (min_percentage : ℚ) : 
  total_words = 600 → min_percentage = 90 / 100 → 
  ∃ (min_words : ℕ), min_words = 540 ∧ 
    (min_words : ℚ) / total_words ≥ min_percentage ∧
    ∀ (n : ℕ), n < min_words → (n : ℚ) / total_words < min_percentage :=
by sorry

end min_words_for_90_percent_l1312_131268
