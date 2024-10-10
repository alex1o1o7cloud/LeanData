import Mathlib

namespace biker_passes_l2663_266369

/-- Represents a biker's total travels along the road -/
structure BikerTravel where
  travels : ℕ

/-- Represents the scenario of two bikers on a road -/
structure BikerScenario where
  biker1 : BikerTravel
  biker2 : BikerTravel

/-- Calculates the number of passes between two bikers -/
def calculatePasses (scenario : BikerScenario) : ℕ :=
  sorry

theorem biker_passes (scenario : BikerScenario) :
  scenario.biker1.travels = 11 →
  scenario.biker2.travels = 7 →
  calculatePasses scenario = 8 :=
sorry

end biker_passes_l2663_266369


namespace polynomial_problem_l2663_266353

theorem polynomial_problem (n : ℕ) (p : ℝ → ℝ) : 
  (∀ k : ℕ, k ≤ n → p (2 * k) = 0) →
  (∀ k : ℕ, k < n → p (2 * k + 1) = 2) →
  p (2 * n + 1) = -30 →
  (∃ c : ℝ → ℝ, ∀ x, p x = c x * x * (x - 2)^2 * (x - 4)) →
  (n = 2 ∧ ∀ x, p x = -2/3 * x * (x - 2)^2 * (x - 4)) :=
by sorry

end polynomial_problem_l2663_266353


namespace no_equal_sums_for_given_sequences_l2663_266368

theorem no_equal_sums_for_given_sequences : ¬ ∃ (n : ℕ), n > 0 ∧
  (let a₁ := 9
   let d₁ := 6
   let t₁ := n * (2 * a₁ + (n - 1) * d₁) / 2
   let a₂ := 11
   let d₂ := 3
   let t₂ := n * (2 * a₂ + (n - 1) * d₂) / 2
   t₁ = t₂) :=
sorry

end no_equal_sums_for_given_sequences_l2663_266368


namespace toothpick_grids_count_l2663_266312

/-- Calculates the number of toothpicks needed for a grid -/
def toothpicks_for_grid (length : ℕ) (width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- The total number of toothpicks for two separate grids -/
def total_toothpicks (outer_length outer_width inner_length inner_width : ℕ) : ℕ :=
  toothpicks_for_grid outer_length outer_width + toothpicks_for_grid inner_length inner_width

theorem toothpick_grids_count :
  total_toothpicks 80 40 30 20 = 7770 := by
  sorry

end toothpick_grids_count_l2663_266312


namespace possible_values_of_a_l2663_266364

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 = 1}
def N (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- Define the set of possible values for a
def A : Set ℝ := {-1, 1, 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : N a ⊆ M → a ∈ A := by
  sorry

end possible_values_of_a_l2663_266364


namespace min_hours_to_reach_55_people_l2663_266306

/-- The number of people who have received the message after n hours -/
def people_reached (n : ℕ) : ℕ := 2^(n + 1) - 2

/-- The proposition that 6 hours is the minimum time needed to reach at least 55 people -/
theorem min_hours_to_reach_55_people : 
  (∀ k < 6, people_reached k ≤ 55) ∧ people_reached 6 > 55 :=
sorry

end min_hours_to_reach_55_people_l2663_266306


namespace cos_BAE_value_l2663_266388

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point E on BC
def E (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the lengths of the sides
def AB (triangle : Triangle) : ℝ := 4
def AC (triangle : Triangle) : ℝ := 8
def BC (triangle : Triangle) : ℝ := 10

-- Define that AE bisects angle BAC
def AE_bisects_BAC (triangle : Triangle) : Prop := sorry

-- Define the cosine of angle BAE
def cos_BAE (triangle : Triangle) : ℝ := sorry

-- Theorem statement
theorem cos_BAE_value (triangle : Triangle) 
  (h1 : AB triangle = 4) 
  (h2 : AC triangle = 8) 
  (h3 : BC triangle = 10) 
  (h4 : AE_bisects_BAC triangle) : 
  cos_BAE triangle = Real.sqrt (11/32) := by
  sorry

end cos_BAE_value_l2663_266388


namespace total_birds_on_fence_l2663_266347

def initial_birds : ℕ := 4
def additional_birds : ℕ := 6

theorem total_birds_on_fence : 
  initial_birds + additional_birds = 10 := by sorry

end total_birds_on_fence_l2663_266347


namespace factor_calculation_l2663_266365

theorem factor_calculation (f : ℝ) : f * (2 * 20 + 5) = 135 → f = 3 := by
  sorry

end factor_calculation_l2663_266365


namespace basketball_score_l2663_266367

theorem basketball_score (three_pointers two_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * two_pointers) →
  (two_pointers = 2 * free_throws) →
  (3 * three_pointers + 2 * two_pointers + free_throws = 73) →
  free_throws = 8 := by
sorry

end basketball_score_l2663_266367


namespace bertolli_farm_produce_difference_l2663_266303

theorem bertolli_farm_produce_difference : 
  let tomatoes : ℕ := 2073
  let corn : ℕ := 4112
  let onions : ℕ := 985
  (tomatoes + corn) - onions = 5200 := by sorry

end bertolli_farm_produce_difference_l2663_266303


namespace win_sector_area_l2663_266396

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
  sorry

end win_sector_area_l2663_266396


namespace min_qr_length_l2663_266308

/-- Given two triangles PQR and SQR sharing side QR, with known side lengths,
    prove that the least possible integral length of QR is 15 cm. -/
theorem min_qr_length (pq pr sr sq : ℝ) (h_pq : pq = 7)
                      (h_pr : pr = 15) (h_sr : sr = 10) (h_sq : sq = 25) :
  ∀ qr : ℝ, qr > pr - pq ∧ qr > sq - sr → qr ≥ 15 := by sorry

end min_qr_length_l2663_266308


namespace inhabitable_earth_fraction_l2663_266305

/-- Represents the fraction of Earth's surface not covered by water -/
def land_fraction : ℚ := 1 / 3

/-- Represents the fraction of exposed land that is inhabitable -/
def inhabitable_land_fraction : ℚ := 1 / 3

/-- Theorem stating the fraction of Earth's surface that is inhabitable for humans -/
theorem inhabitable_earth_fraction :
  land_fraction * inhabitable_land_fraction = 1 / 9 := by sorry

end inhabitable_earth_fraction_l2663_266305


namespace sum_of_squares_lower_bound_l2663_266351

theorem sum_of_squares_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by sorry

end sum_of_squares_lower_bound_l2663_266351


namespace dice_configuration_dots_l2663_266395

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the configuration of four dice -/
structure DiceConfiguration :=
  (faceA : DieFace)
  (faceB : DieFace)
  (faceC : DieFace)
  (faceD : DieFace)

/-- Counts the number of dots on a die face -/
def dotCount (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- The specific configuration of dice in the problem -/
def problemConfiguration : DiceConfiguration :=
  { faceA := DieFace.three
  , faceB := DieFace.five
  , faceC := DieFace.six
  , faceD := DieFace.five }

theorem dice_configuration_dots :
  dotCount problemConfiguration.faceA = 3 ∧
  dotCount problemConfiguration.faceB = 5 ∧
  dotCount problemConfiguration.faceC = 6 ∧
  dotCount problemConfiguration.faceD = 5 := by
  sorry

end dice_configuration_dots_l2663_266395


namespace tangent_line_equation_l2663_266340

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

/-- The derivative of the cubic curve -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

/-- The x-coordinate of the point of tangency -/
def x₀ : ℝ := 2

/-- The y-coordinate of the point of tangency -/
def y₀ : ℝ := -3

/-- The slope of the tangent line -/
def m : ℝ := f' x₀

theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -3*x + 3 :=
sorry

end tangent_line_equation_l2663_266340


namespace modular_inverse_of_2_mod_199_l2663_266371

theorem modular_inverse_of_2_mod_199 : ∃ x : ℤ, 2 * x ≡ 1 [ZMOD 199] ∧ 0 ≤ x ∧ x < 199 :=
  by sorry

end modular_inverse_of_2_mod_199_l2663_266371


namespace harmonic_arithmetic_mean_inequality_l2663_266350

theorem harmonic_arithmetic_mean_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let h := 3 / ((1 / a) + (1 / b) + (1 / c))
  let m := (a + b + c) / 3
  h ≤ m ∧ (h = m ↔ a = b ∧ b = c) := by
  sorry

#check harmonic_arithmetic_mean_inequality

end harmonic_arithmetic_mean_inequality_l2663_266350


namespace greatest_common_factor_90_135_180_l2663_266374

theorem greatest_common_factor_90_135_180 : Nat.gcd 90 (Nat.gcd 135 180) = 45 := by
  sorry

end greatest_common_factor_90_135_180_l2663_266374


namespace min_value_of_z_l2663_266325

theorem min_value_of_z (x y : ℝ) : 
  3 * x^2 + y^2 + 12 * x - 6 * y + 40 ≥ 19 ∧ 
  ∃ x y : ℝ, 3 * x^2 + y^2 + 12 * x - 6 * y + 40 = 19 := by
sorry

end min_value_of_z_l2663_266325


namespace zeros_of_g_l2663_266345

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (a b x : ℝ) : ℝ := b * x^2 - a * x

-- State the theorem
theorem zeros_of_g (a b : ℝ) (h : f a b 2 = 0) :
  (g a b 0 = 0) ∧ (g a b (-1/2) = 0) :=
sorry

end zeros_of_g_l2663_266345


namespace range_of_p_l2663_266318

-- Define the function p(x)
def p (x : ℝ) : ℝ := (x^3 + 3)^2

-- Define the domain of p(x)
def domain : Set ℝ := {x : ℝ | x ≥ -1}

-- Define the range of p(x)
def range : Set ℝ := {y : ℝ | ∃ x ∈ domain, p x = y}

-- Theorem statement
theorem range_of_p : range = {y : ℝ | y ≥ 4} := by sorry

end range_of_p_l2663_266318


namespace santa_mandarins_l2663_266379

/-- Represents the exchange game with Santa Claus --/
structure ExchangeGame where
  /-- Number of first type exchanges (5 mandarins for 3 firecrackers and 1 candy) --/
  first_exchanges : ℕ
  /-- Number of second type exchanges (2 firecrackers for 3 mandarins and 1 candy) --/
  second_exchanges : ℕ
  /-- Total number of candies received --/
  total_candies : ℕ
  /-- Constraint: Total exchanges equal total candies --/
  exchanges_eq_candies : first_exchanges + second_exchanges = total_candies
  /-- Constraint: Firecrackers balance out --/
  firecrackers_balance : 3 * first_exchanges = 2 * second_exchanges

/-- The main theorem to prove --/
theorem santa_mandarins (game : ExchangeGame) (h : game.total_candies = 50) :
  5 * game.first_exchanges - 3 * game.second_exchanges = 10 := by
  sorry

end santa_mandarins_l2663_266379


namespace lcm_of_20_and_36_l2663_266389

theorem lcm_of_20_and_36 : Nat.lcm 20 36 = 180 := by
  sorry

end lcm_of_20_and_36_l2663_266389


namespace montero_trip_feasibility_l2663_266322

/-- Represents the parameters of Mr. Montero's trip -/
structure TripParameters where
  normal_efficiency : Real
  traffic_efficiency_reduction : Real
  total_distance : Real
  traffic_distance : Real
  initial_gas : Real
  gas_price : Real
  price_increase : Real
  budget : Real

/-- Calculates whether Mr. Montero can complete his trip within budget -/
def can_complete_trip (params : TripParameters) : Prop :=
  let reduced_efficiency := params.normal_efficiency * (1 - params.traffic_efficiency_reduction)
  let normal_distance := params.total_distance - params.traffic_distance
  let gas_needed := normal_distance / params.normal_efficiency + 
                    params.traffic_distance / reduced_efficiency
  let gas_to_buy := gas_needed - params.initial_gas
  let half_trip_gas := (params.total_distance / 2) / params.normal_efficiency - params.initial_gas
  let first_half_cost := min half_trip_gas gas_to_buy * params.gas_price
  let second_half_cost := max 0 (gas_to_buy - half_trip_gas) * (params.gas_price * (1 + params.price_increase))
  first_half_cost + second_half_cost ≤ params.budget

theorem montero_trip_feasibility :
  let params : TripParameters := {
    normal_efficiency := 20,
    traffic_efficiency_reduction := 0.2,
    total_distance := 600,
    traffic_distance := 100,
    initial_gas := 8,
    gas_price := 2.5,
    price_increase := 0.1,
    budget := 75
  }
  can_complete_trip params := by sorry

end montero_trip_feasibility_l2663_266322


namespace pizza_area_increase_l2663_266309

theorem pizza_area_increase : 
  let r1 : ℝ := 2
  let r2 : ℝ := 5
  let area1 := π * r1^2
  let area2 := π * r2^2
  (area2 - area1) / area1 * 100 = 525 :=
by sorry

end pizza_area_increase_l2663_266309


namespace largest_size_percentage_longer_than_smallest_l2663_266320

-- Define the shoe size range
def min_size : ℕ := 8
def max_size : ℕ := 17

-- Define the length increase per size
def length_increase_per_size : ℚ := 1 / 5

-- Define the length of size 15 shoe
def size_15_length : ℚ := 21 / 2  -- 10.4 as a rational number

-- Function to calculate shoe length given size
def shoe_length (size : ℕ) : ℚ :=
  size_15_length + (size - 15 : ℚ) * length_increase_per_size

-- Theorem statement
theorem largest_size_percentage_longer_than_smallest :
  (shoe_length max_size - shoe_length min_size) / shoe_length min_size = 1 / 5 := by
  sorry

end largest_size_percentage_longer_than_smallest_l2663_266320


namespace parabola_line_single_intersection_l2663_266380

/-- The value of a that makes the parabola y = ax^2 + 3x + 1 intersect
    the line y = -2x - 3 at only one point is 25/16 -/
theorem parabola_line_single_intersection :
  ∃! a : ℚ, ∀ x : ℚ,
    (a * x^2 + 3 * x + 1 = -2 * x - 3) →
    (∀ y : ℚ, y ≠ x → a * y^2 + 3 * y + 1 ≠ -2 * y - 3) :=
by sorry

end parabola_line_single_intersection_l2663_266380


namespace two_minus_repeating_decimal_l2663_266366

/-- The value of the repeating decimal 1.888... -/
def repeating_decimal : ℚ := 17 / 9

/-- Theorem stating that 2 minus the repeating decimal 1.888... equals 1/9 -/
theorem two_minus_repeating_decimal :
  2 - repeating_decimal = 1 / 9 := by
  sorry

end two_minus_repeating_decimal_l2663_266366


namespace smallest_sum_of_a_and_b_l2663_266392

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  ∀ c d : ℝ, c > 0 → d > 0 →
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) →
  (∃ x : ℝ, x^2 + 3*d*x + c = 0) →
  a + b ≤ c + d ∧ a + b ≥ 6.5 :=
sorry

end smallest_sum_of_a_and_b_l2663_266392


namespace distance_ratio_l2663_266349

/-- Yan's travel scenario --/
structure TravelScenario where
  w : ℝ  -- Yan's walking speed
  x : ℝ  -- Distance from Yan to his office
  y : ℝ  -- Distance from Yan to the concert hall
  h_positive : w > 0 ∧ x > 0 ∧ y > 0  -- Positive distances and speed

/-- The time taken for both travel options is equal --/
def equal_time (s : TravelScenario) : Prop :=
  s.y / s.w = (s.x / s.w + (s.x + s.y) / (5 * s.w))

/-- The theorem stating the ratio of distances --/
theorem distance_ratio (s : TravelScenario) 
  (h_equal_time : equal_time s) : 
  s.x / s.y = 2 / 3 := by
  sorry


end distance_ratio_l2663_266349


namespace equation_solutions_l2663_266385

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 - Real.sqrt 11 ∧ x₂ = 2 + Real.sqrt 11 ∧
    x₁^2 - 4*x₁ - 7 = 0 ∧ x₂^2 - 4*x₂ - 7 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧
    (x₁ - 3)^2 + 2*(x₁ - 3) = 0 ∧ (x₂ - 3)^2 + 2*(x₂ - 3) = 0) :=
by sorry

end equation_solutions_l2663_266385


namespace quadratic_function_properties_l2663_266344

-- Define the quadratic function f
def f : ℝ → ℝ := λ x => x^2 - x + 1

-- State the theorem
theorem quadratic_function_properties :
  (f 0 = 1) ∧
  (∀ x, f (x + 1) - f x = 2 * x) ∧
  (f = λ x => x^2 - x + 1) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3/4) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3) :=
by sorry


end quadratic_function_properties_l2663_266344


namespace at_least_one_parabola_has_two_roots_l2663_266335

-- Define the parabolas
def parabola1 (a b c x : ℝ) : ℝ := a * x^2 + 2 * b * x + c
def parabola2 (a b c x : ℝ) : ℝ := b * x^2 + 2 * c * x + a
def parabola3 (a b c x : ℝ) : ℝ := c * x^2 + 2 * a * x + b

-- Define a function to check if a parabola has two distinct roots
def has_two_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0

-- State the theorem
theorem at_least_one_parabola_has_two_roots (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  has_two_distinct_roots (parabola1 a b c) ∨ 
  has_two_distinct_roots (parabola2 a b c) ∨ 
  has_two_distinct_roots (parabola3 a b c) :=
sorry

end at_least_one_parabola_has_two_roots_l2663_266335


namespace problem_statement_l2663_266387

theorem problem_statement (a b c : ℝ) 
  (h1 : a * b * c ≠ 0) 
  (h2 : a + b + c = 2) 
  (h3 : a^2 + b^2 + c^2 = 2) : 
  (1 - a)^2 / (b * c) + (1 - b)^2 / (c * a) + (1 - c)^2 / (a * b) = 1 := by
  sorry

end problem_statement_l2663_266387


namespace unique_solution_l2663_266377

/-- The functional equation satisfied by g --/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) * g (x - y) = (g x - g y)^2 - 6 * x^2 * g y

/-- There is only one function satisfying the functional equation --/
theorem unique_solution :
  ∃! g : ℝ → ℝ, FunctionalEquation g ∧ ∀ x : ℝ, g x = 0 :=
sorry

end unique_solution_l2663_266377


namespace union_of_sets_l2663_266329

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_of_sets :
  ∃ x : ℝ, (B x ∩ A x = {9}) → (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end union_of_sets_l2663_266329


namespace inequality_solution_l2663_266363

theorem inequality_solution (x : ℝ) :
  x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < 1 ∨ x > 3) :=
by sorry

end inequality_solution_l2663_266363


namespace sum_of_coefficients_l2663_266315

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 = 
   a₀ + a₁*(1 - x) + a₂*(1 - x)^2 + a₃*(1 - x)^3 + a₄*(1 - x)^4 + a₅*(1 - x)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -57 := by
sorry

end sum_of_coefficients_l2663_266315


namespace max_product_sum_300_l2663_266358

theorem max_product_sum_300 :
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 ∧ ∃ a b : ℤ, a + b = 300 ∧ a * b = 22500 := by
  sorry

end max_product_sum_300_l2663_266358


namespace sqrt_and_principal_sqrt_of_zero_l2663_266314

-- Define square root function
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Define principal square root function
noncomputable def principal_sqrt (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.sqrt x else 0

-- Theorem statement
theorem sqrt_and_principal_sqrt_of_zero :
  sqrt 0 = 0 ∧ principal_sqrt 0 = 0 := by
  sorry

end sqrt_and_principal_sqrt_of_zero_l2663_266314


namespace teresa_bought_six_hardcovers_l2663_266341

/-- The number of hardcover books Teresa bought -/
def num_hardcovers : ℕ := sorry

/-- The number of paperback books Teresa bought -/
def num_paperbacks : ℕ := sorry

/-- The total number of books Teresa bought -/
def total_books : ℕ := 12

/-- The cost of a hardcover book -/
def hardcover_cost : ℕ := 30

/-- The cost of a paperback book -/
def paperback_cost : ℕ := 18

/-- The total amount Teresa spent -/
def total_spent : ℕ := 288

/-- Theorem stating that Teresa bought 6 hardcover books -/
theorem teresa_bought_six_hardcovers :
  num_hardcovers = 6 ∧
  num_hardcovers ≥ 4 ∧
  num_hardcovers + num_paperbacks = total_books ∧
  num_hardcovers * hardcover_cost + num_paperbacks * paperback_cost = total_spent :=
sorry

end teresa_bought_six_hardcovers_l2663_266341


namespace industrial_lubricants_budget_percentage_l2663_266382

theorem industrial_lubricants_budget_percentage
  (total_degrees : ℝ)
  (microphotonics_percent : ℝ)
  (home_electronics_percent : ℝ)
  (food_additives_percent : ℝ)
  (genetically_modified_microorganisms_percent : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : microphotonics_percent = 14)
  (h3 : home_electronics_percent = 19)
  (h4 : food_additives_percent = 10)
  (h5 : genetically_modified_microorganisms_percent = 24)
  (h6 : basic_astrophysics_degrees = 90) :
  let total_percent : ℝ := 100
  let basic_astrophysics_percent : ℝ := (basic_astrophysics_degrees / total_degrees) * total_percent
  let known_sectors_percent : ℝ := microphotonics_percent + home_electronics_percent + 
                                   food_additives_percent + genetically_modified_microorganisms_percent
  let industrial_lubricants_percent : ℝ := total_percent - known_sectors_percent - basic_astrophysics_percent
  industrial_lubricants_percent = 8 := by
  sorry

end industrial_lubricants_budget_percentage_l2663_266382


namespace optimal_advertising_strategy_l2663_266398

/-- Sales revenue function -/
def R (x₁ x₂ : ℝ) : ℝ := -2 * x₁^2 - x₂^2 + 13 * x₁ + 11 * x₂ - 28

/-- Profit function -/
def profit (x₁ x₂ : ℝ) : ℝ := R x₁ x₂ - x₁ - x₂

theorem optimal_advertising_strategy :
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 5 ∧ 
    ∀ (y₁ y₂ : ℝ), y₁ + y₂ = 5 → profit x₁ x₂ ≥ profit y₁ y₂) ∧
  profit 2 3 = 9 ∧
  (∀ (y₁ y₂ : ℝ), profit 3 5 ≥ profit y₁ y₂) ∧
  profit 3 5 = 15 := by sorry

end optimal_advertising_strategy_l2663_266398


namespace balloon_count_impossible_l2663_266334

theorem balloon_count_impossible : ¬∃ (b g : ℕ), 3 * (b + g) = 100 := by
  sorry

#check balloon_count_impossible

end balloon_count_impossible_l2663_266334


namespace arithmetic_sequence_property_l2663_266343

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a₁ d : ℝ, ∀ n : ℕ, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 + a 10 = 12 →
  3 * a 7 + a 9 = 24 := by
  sorry

end arithmetic_sequence_property_l2663_266343


namespace product_mod_23_l2663_266394

theorem product_mod_23 : (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 12 := by
  sorry

end product_mod_23_l2663_266394


namespace decimal_53_is_binary_110101_l2663_266304

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec toBinaryAux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: toBinaryAux (m / 2)
    toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_53_is_binary_110101 :
  toBinary 53 = [true, false, true, false, true, true] :=
by sorry

end decimal_53_is_binary_110101_l2663_266304


namespace equation_solutions_l2663_266336

theorem equation_solutions (x : ℝ) : 2 * x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1/2 :=
sorry

end equation_solutions_l2663_266336


namespace solution_set_f_g_has_zero_condition_l2663_266386

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |3 + a|

-- Part I
theorem solution_set_f (x : ℝ) : 
  f 3 x > 6 ↔ x < -4 ∨ x > 2 := by sorry

-- Part II
theorem g_has_zero_condition (a : ℝ) :
  (∃ x, g a x = 0) → a ≥ -2 := by sorry

end solution_set_f_g_has_zero_condition_l2663_266386


namespace sum_remainder_mod_seven_l2663_266331

theorem sum_remainder_mod_seven :
  (123450 + 123451 + 123452 + 123453 + 123454 + 123455) % 7 = 5 := by
  sorry

end sum_remainder_mod_seven_l2663_266331


namespace f_derivative_and_tangent_line_l2663_266300

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + x * Real.log x

-- State the theorem
theorem f_derivative_and_tangent_line :
  -- The derivative of f(x)
  (∀ x : ℝ, x > 0 → HasDerivAt f (2*x + Real.log x + 1) x) ∧
  -- The equation of the tangent line at x = 1
  (∃ A B C : ℝ, A = 3 ∧ B = -1 ∧ C = -2 ∧
    ∀ x y : ℝ, (x = 1 ∧ y = f 1) → (A*x + B*y + C = 0)) := by
  sorry

end

end f_derivative_and_tangent_line_l2663_266300


namespace power_function_through_point_l2663_266311

theorem power_function_through_point (n : ℝ) : 
  (∀ x y : ℝ, y = x^n → (x = 2 ∧ y = 8) → n = 3) :=
by sorry

end power_function_through_point_l2663_266311


namespace base8_to_base10_77_l2663_266391

/-- Converts a two-digit number in base 8 to base 10 -/
def base8_to_base10 (a b : Nat) : Nat :=
  a * 8 + b

/-- The given number in base 8 -/
def number_base8 : Nat × Nat := (7, 7)

theorem base8_to_base10_77 :
  base8_to_base10 number_base8.1 number_base8.2 = 63 := by
  sorry

end base8_to_base10_77_l2663_266391


namespace farm_animal_problem_l2663_266381

/-- Prove that the number of cows is 9 given the conditions of the farm animal problem -/
theorem farm_animal_problem :
  ∃ (num_cows : ℕ),
    let num_chickens : ℕ := 8
    let num_ducks : ℕ := 3
    let total_legs : ℕ := 4 * num_cows + 2 * num_chickens + 2 * num_ducks
    let total_heads : ℕ := num_cows + num_chickens + 2 * num_ducks
    total_legs = 18 + 2 * total_heads ∧ num_cows = 9 :=
by
  sorry

end farm_animal_problem_l2663_266381


namespace company_workforce_after_hiring_l2663_266301

theorem company_workforce_after_hiring 
  (initial_female_percentage : Real)
  (additional_male_hires : Nat)
  (final_female_percentage : Real) :
  initial_female_percentage = 0.60 →
  additional_male_hires = 26 →
  final_female_percentage = 0.55 →
  ∃ (initial_total : Nat),
    (initial_total : Real) * initial_female_percentage + 
    (initial_total : Real) * (1 - initial_female_percentage) = initial_total ∧
    (initial_total + additional_male_hires : Real) * final_female_percentage + 
    ((initial_total : Real) * (1 - initial_female_percentage) + additional_male_hires) = 
    initial_total + additional_male_hires ∧
    initial_total + additional_male_hires = 312 :=
by sorry

end company_workforce_after_hiring_l2663_266301


namespace log_equality_l2663_266346

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equality : log10 2 + 2 * log10 5 = 1 + log10 5 := by sorry

end log_equality_l2663_266346


namespace equation_solution_l2663_266324

theorem equation_solution : 
  ∀ x : ℝ, (3*x - 1)*(2*x + 4) = 1 ↔ x = (-5 + Real.sqrt 55) / 6 ∨ x = (-5 - Real.sqrt 55) / 6 := by
sorry

end equation_solution_l2663_266324


namespace trapezoid_area_l2663_266362

/-- Given an outer equilateral triangle with area 16, an inner equilateral triangle
    with area 1, and three congruent trapezoids between them, the area of one
    trapezoid is 5. -/
theorem trapezoid_area (outer_area inner_area : ℝ) (num_trapezoids : ℕ) :
  outer_area = 16 →
  inner_area = 1 →
  num_trapezoids = 3 →
  (outer_area - inner_area) / num_trapezoids = 5 := by
  sorry

end trapezoid_area_l2663_266362


namespace solve_for_x_l2663_266319

theorem solve_for_x (x y : ℚ) (h1 : x / y = 9 / 5) (h2 : y = 25) : x = 45 := by
  sorry

end solve_for_x_l2663_266319


namespace f_three_quadrants_iff_a_range_l2663_266323

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * x - 1
  else x^3 - a * x + |x - 2|

/-- The graph of f passes through exactly three quadrants -/
def passes_through_three_quadrants (a : ℝ) : Prop :=
  ∃ x y z : ℝ,
    (x < 0 ∧ f a x < 0) ∧
    (y > 0 ∧ f a y > 0) ∧
    ((z < 0 ∧ f a z > 0) ∨ (z > 0 ∧ f a z < 0)) ∧
    ∀ w : ℝ, (w < 0 ∧ f a w > 0) → (z < 0 ∧ f a z > 0) ∧
             (w > 0 ∧ f a w < 0) → (z > 0 ∧ f a z < 0)

/-- Main theorem: f passes through exactly three quadrants iff a < 0 or a > 2 -/
theorem f_three_quadrants_iff_a_range (a : ℝ) :
  passes_through_three_quadrants a ↔ a < 0 ∨ a > 2 := by
  sorry


end f_three_quadrants_iff_a_range_l2663_266323


namespace recurring_fraction_equality_l2663_266354

-- Define the recurring decimal 0.812812...
def recurring_812 : ℚ := 812 / 999

-- Define the recurring decimal 2.406406...
def recurring_2406 : ℚ := 2404 / 999

-- Theorem statement
theorem recurring_fraction_equality : 
  recurring_812 / recurring_2406 = 203 / 601 := by
  sorry

end recurring_fraction_equality_l2663_266354


namespace cos_shift_right_l2663_266339

theorem cos_shift_right (x : ℝ) :
  let f (x : ℝ) := Real.cos (2 * x)
  let shift := π / 12
  let g (x : ℝ) := f (x - shift)
  g x = Real.cos (2 * x - π / 6) := by
  sorry

end cos_shift_right_l2663_266339


namespace system_of_inequalities_solution_equation_solution_expression_evaluation_l2663_266399

-- Part 1: System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x - 4 < 2 * (x - 1) ∧ (1 + 2 * x) / 3 ≥ x) ↔ -2 < x ∧ x ≤ 1 := by sorry

-- Part 2: Equation solution
theorem equation_solution :
  ∃! x : ℝ, (x - 2) / (x - 3) = 2 - 1 / (3 - x) ∧ x = 3 := by sorry

-- Part 3: Expression simplification and evaluation
theorem expression_evaluation (x : ℝ) (h : x = 3) :
  (1 - 1 / (x + 2)) / ((x^2 - 1) / (x + 2)) = 1 / 2 := by sorry

end system_of_inequalities_solution_equation_solution_expression_evaluation_l2663_266399


namespace figurine_cost_l2663_266316

/-- The cost of a single figurine given Annie's purchases -/
theorem figurine_cost (num_tvs : ℕ) (tv_cost : ℕ) (num_figurines : ℕ) (total_spent : ℕ) : 
  num_tvs = 5 → 
  tv_cost = 50 → 
  num_figurines = 10 → 
  total_spent = 260 → 
  (total_spent - num_tvs * tv_cost) / num_figurines = 1 := by
sorry

end figurine_cost_l2663_266316


namespace games_missed_l2663_266326

theorem games_missed (total_games attended_games : ℕ) (h1 : total_games = 31) (h2 : attended_games = 13) :
  total_games - attended_games = 18 := by
  sorry

end games_missed_l2663_266326


namespace tenth_prime_l2663_266383

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem tenth_prime :
  (nth_prime 5 = 11) → (nth_prime 10 = 29) := by sorry

end tenth_prime_l2663_266383


namespace mom_in_middle_l2663_266313

-- Define the people in the lineup
inductive Person : Type
  | Dad : Person
  | Mom : Person
  | Brother : Person
  | Sister : Person
  | Me : Person

-- Define the concept of being next to someone in the lineup
def next_to (p1 p2 : Person) : Prop := sorry

-- Define the concept of being in the middle
def in_middle (p : Person) : Prop := sorry

-- State the theorem
theorem mom_in_middle :
  -- Conditions
  (next_to Person.Me Person.Dad) →
  (next_to Person.Me Person.Mom) →
  (next_to Person.Sister Person.Mom) →
  (next_to Person.Sister Person.Brother) →
  -- Conclusion
  in_middle Person.Mom := by sorry

end mom_in_middle_l2663_266313


namespace complex_equation_solution_l2663_266352

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ) * (2 + a * Complex.I) * (a - 2 * Complex.I) = -4 * Complex.I → a = 0 := by
  sorry

end complex_equation_solution_l2663_266352


namespace largest_coeff_sixth_term_l2663_266310

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the coefficient of the r-th term in the expansion
def coeff (r : ℕ) : ℚ := (1/2)^r * binomial_coeff 15 r

-- Theorem statement
theorem largest_coeff_sixth_term :
  ∀ k : ℕ, k ≠ 5 → coeff 5 ≥ coeff k :=
sorry

end largest_coeff_sixth_term_l2663_266310


namespace luka_aubrey_age_difference_l2663_266373

structure Person where
  name : String
  age : ℕ

structure Dog where
  name : String
  age : ℕ

def age_difference (p1 p2 : Person) : ℤ :=
  p1.age - p2.age

theorem luka_aubrey_age_difference 
  (luka aubrey : Person) 
  (max : Dog) 
  (h1 : max.age = luka.age - 4)
  (h2 : aubrey.age = 8)
  (h3 : max.age = 6) : 
  age_difference luka aubrey = 2 := by
sorry

end luka_aubrey_age_difference_l2663_266373


namespace square_floor_tiles_l2663_266317

/-- Given a square floor with side length s, where tiles along the diagonals
    are marked blue, prove that if there are 225 blue tiles, then the total
    number of tiles on the floor is 12769. -/
theorem square_floor_tiles (s : ℕ) : 
  (2 * s - 1 = 225) → s^2 = 12769 := by sorry

end square_floor_tiles_l2663_266317


namespace find_a_value_l2663_266397

def A : Set ℝ := {0, 1, 2}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 3}

theorem find_a_value :
  ∀ a : ℝ, (A ∩ B a = {1}) → a = -1 := by
sorry

end find_a_value_l2663_266397


namespace wednesday_discount_percentage_wednesday_jeans_discount_approx_40_82_percent_l2663_266390

/-- Calculates the additional Wednesday discount for jeans given the original price,
    summer discount percentage, and final price after all discounts. -/
theorem wednesday_discount_percentage 
  (original_price : ℝ) 
  (summer_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_summer_discount := original_price * (1 - summer_discount_percent / 100)
  let additional_discount := price_after_summer_discount - final_price
  let wednesday_discount_percent := (additional_discount / price_after_summer_discount) * 100
  wednesday_discount_percent

/-- The additional Wednesday discount for jeans is approximately 40.82% -/
theorem wednesday_jeans_discount_approx_40_82_percent : 
  ∃ ε > 0, abs (wednesday_discount_percentage 49 50 14.5 - 40.82) < ε :=
sorry

end wednesday_discount_percentage_wednesday_jeans_discount_approx_40_82_percent_l2663_266390


namespace charlie_votes_l2663_266357

/-- Represents a candidate in the student council election -/
inductive Candidate
| Alex
| Brenda
| Charlie
| Dana

/-- Represents the vote count for each candidate -/
def VoteCount := Candidate → ℕ

/-- The total number of votes cast in the election -/
def TotalVotes (votes : VoteCount) : ℕ :=
  votes Candidate.Alex + votes Candidate.Brenda + votes Candidate.Charlie + votes Candidate.Dana

theorem charlie_votes (votes : VoteCount) : 
  votes Candidate.Brenda = 40 ∧ 
  4 * votes Candidate.Brenda = TotalVotes votes ∧
  votes Candidate.Charlie = votes Candidate.Dana + 10 →
  votes Candidate.Charlie = 45 := by
  sorry

end charlie_votes_l2663_266357


namespace guessing_game_theorem_l2663_266337

/-- The guessing game between Banana and Corona -/
def GuessGame (n k : ℕ) : Prop :=
  n > 0 ∧ k > 0 ∧ k ≤ 2^n

/-- Corona can determine x in finitely many turns -/
def CanDetermine (n k : ℕ) : Prop :=
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ n → ∃ t : ℕ, t > 0 ∧ (∀ y : ℕ, 1 ≤ y ∧ y ≤ n → y = x)

/-- The main theorem: Corona can determine x iff k ≤ 2^(n-1) -/
theorem guessing_game_theorem (n k : ℕ) :
  GuessGame n k → (CanDetermine n k ↔ k ≤ 2^(n-1)) :=
by sorry

end guessing_game_theorem_l2663_266337


namespace quadratic_function_property_l2663_266393

theorem quadratic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f (f 0) = 0 ∧ f (f 1) = 0 ∧ f 0 ≠ f 1) → f 2 = 2 := by
  sorry

end quadratic_function_property_l2663_266393


namespace zoo_ticket_cost_l2663_266327

theorem zoo_ticket_cost 
  (total_spent : ℚ)
  (family_size : ℕ)
  (adult_ticket_cost : ℚ)
  (adult_tickets : ℕ)
  (h1 : total_spent = 119)
  (h2 : family_size = 7)
  (h3 : adult_ticket_cost = 21)
  (h4 : adult_tickets = 4) :
  let children_tickets := family_size - adult_tickets
  let children_total_cost := total_spent - (adult_ticket_cost * adult_tickets)
  children_total_cost / children_tickets = 35 / 3 :=
sorry

end zoo_ticket_cost_l2663_266327


namespace product_equals_64_l2663_266356

theorem product_equals_64 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64 := by
  sorry

end product_equals_64_l2663_266356


namespace min_distance_to_line_l2663_266372

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∀ (x y : ℝ), 
  x + y - 4 = 0 → 
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
  ∀ (x' y' : ℝ), x' + y' - 4 = 0 → 
  Real.sqrt (x' ^ 2 + y' ^ 2) ≥ d := by
  sorry


end min_distance_to_line_l2663_266372


namespace photo_album_completion_l2663_266332

theorem photo_album_completion 
  (total_slots : ℕ) 
  (cristina_photos : ℕ) 
  (john_photos : ℕ) 
  (sarah_photos : ℕ) 
  (h1 : total_slots = 40) 
  (h2 : cristina_photos = 7) 
  (h3 : john_photos = 10) 
  (h4 : sarah_photos = 9) : 
  total_slots - (cristina_photos + john_photos + sarah_photos) = 14 := by
  sorry

end photo_album_completion_l2663_266332


namespace price_increase_problem_l2663_266378

theorem price_increase_problem (candy_new : ℝ) (soda_new : ℝ) (chips_new : ℝ) (chocolate_new : ℝ)
  (candy_increase : ℝ) (soda_increase : ℝ) (chips_increase : ℝ) (chocolate_increase : ℝ)
  (h_candy : candy_new = 10) (h_soda : soda_new = 6) (h_chips : chips_new = 4) (h_chocolate : chocolate_new = 2)
  (h_candy_inc : candy_increase = 0.25) (h_soda_inc : soda_increase = 0.5)
  (h_chips_inc : chips_increase = 0.4) (h_chocolate_inc : chocolate_increase = 0.75) :
  (candy_new / (1 + candy_increase)) + (soda_new / (1 + soda_increase)) +
  (chips_new / (1 + chips_increase)) + (chocolate_new / (1 + chocolate_increase)) = 16 :=
by sorry

end price_increase_problem_l2663_266378


namespace union_equals_reals_l2663_266342

-- Define sets A and B
def A : Set ℝ := {x | Real.log x > 0}
def B : Set ℝ := {x | x ≤ 1}

-- State the theorem
theorem union_equals_reals : A ∪ B = Set.univ := by
  sorry

end union_equals_reals_l2663_266342


namespace parabola_translation_l2663_266361

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically by a given amount -/
def translateVertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + dy }

/-- Translates a parabola horizontally by a given amount -/
def translateHorizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * dx, c := p.c + p.a * dx^2 - p.b * dx }

theorem parabola_translation (p : Parabola) :
  p.a = 1 ∧ p.b = -2 ∧ p.c = 4 →
  let p' := translateHorizontal (translateVertical p 3) 1
  p'.a = 1 ∧ p'.b = -4 ∧ p'.c = 10 := by
  sorry

end parabola_translation_l2663_266361


namespace odd_polynomial_sum_zero_main_theorem_l2663_266370

/-- Definition of an even polynomial -/
def is_even_polynomial (p : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, p y = p (-y)

/-- Definition of an odd polynomial -/
def is_odd_polynomial (p : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, p y = -p (-y)

/-- Theorem: For an odd polynomial, the sum of values at opposite points is zero -/
theorem odd_polynomial_sum_zero (A : ℝ → ℝ) (h : is_odd_polynomial A) :
  A 3 + A (-3) = 0 :=
by sorry

/-- Main theorem to be proved -/
theorem main_theorem :
  ∃ (A : ℝ → ℝ), is_odd_polynomial A ∧ A 3 + A (-3) = 0 :=
by sorry

end odd_polynomial_sum_zero_main_theorem_l2663_266370


namespace base_seven_divisibility_l2663_266333

theorem base_seven_divisibility (d : Nat) : 
  d ≤ 6 → (2 * 7^3 + d * 7^2 + d * 7 + 7) % 13 = 0 ↔ d = 0 := by
  sorry

end base_seven_divisibility_l2663_266333


namespace picnic_blanket_side_length_l2663_266321

theorem picnic_blanket_side_length 
  (number_of_blankets : ℕ) 
  (folds : ℕ) 
  (total_folded_area : ℝ) 
  (L : ℝ) :
  number_of_blankets = 3 →
  folds = 4 →
  total_folded_area = 48 →
  (number_of_blankets : ℝ) * (L^2 / 2^folds) = total_folded_area →
  L = 16 :=
by sorry

end picnic_blanket_side_length_l2663_266321


namespace line_intersects_circle_l2663_266359

/-- The line kx+y+2=0 intersects the circle (x-1)^2+(y+2)^2=16 for all real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (k * x + y + 2 = 0) ∧ ((x - 1)^2 + (y + 2)^2 = 16) := by sorry

end line_intersects_circle_l2663_266359


namespace quadratic_inequality_l2663_266360

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : 4 * a - 4 * b + c > 0) 
  (h2 : a + 2 * b + c < 0) : 
  b^2 > a * c := by
  sorry

end quadratic_inequality_l2663_266360


namespace parabola_point_value_l2663_266307

theorem parabola_point_value (k : ℝ) : 
  let line1 : ℝ → ℝ := λ x => -x + 3
  let line2 : ℝ → ℝ := λ x => (x - 6) / 2
  let intersection_x : ℝ := 4
  let intersection_y : ℝ := line1 intersection_x
  let x_intercept1 : ℝ := 3
  let x_intercept2 : ℝ := 6
  let parabola : ℝ → ℝ := λ x => (1/2) * (x - x_intercept1) * (x - x_intercept2)
  (parabola intersection_x = intersection_y) ∧
  (parabola x_intercept1 = 0) ∧
  (parabola x_intercept2 = 0) ∧
  (parabola 10 = k)
  → k = 14 := by
sorry


end parabola_point_value_l2663_266307


namespace monkey_climb_theorem_l2663_266355

/-- The time taken for a monkey to climb a tree, given the tree height, hop distance, slip distance, and final hop distance. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (final_hop : ℕ) : ℕ :=
  let net_progress := hop_distance - slip_distance
  let distance_before_final_hop := tree_height - final_hop
  distance_before_final_hop / net_progress + 1

/-- Theorem stating that a monkey climbing a 20 ft tree, hopping 3 ft and slipping 2 ft each hour, with a final 3 ft hop, takes 18 hours to reach the top. -/
theorem monkey_climb_theorem :
  monkey_climb_time 20 3 2 3 = 18 := by
  sorry

end monkey_climb_theorem_l2663_266355


namespace sufficient_not_necessary_l2663_266375

theorem sufficient_not_necessary (a : ℝ) :
  (a = 1/8 → ∀ x : ℝ, x > 0 → 2*x + a/x ≥ 1) ∧
  (∃ a : ℝ, a > 1/8 ∧ ∀ x : ℝ, x > 0 → 2*x + a/x ≥ 1) :=
by sorry

end sufficient_not_necessary_l2663_266375


namespace circle_radius_decrease_l2663_266384

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let r' := r * (1 - x)
  let A' := π * r'^2
  A' = 0.25 * A →
  x = 0.5
  := by sorry

end circle_radius_decrease_l2663_266384


namespace new_ellipse_and_hyperbola_l2663_266376

/-- New distance between two points -/
def new_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- New ellipse -/
def on_new_ellipse (x y c d a : ℝ) : Prop :=
  new_distance x y c d + new_distance x y (-c) (-d) = 2 * a

/-- New hyperbola -/
def on_new_hyperbola (x y c d a : ℝ) : Prop :=
  |new_distance x y c d - new_distance x y (-c) (-d)| = 2 * a

/-- Main theorem for new ellipse and hyperbola -/
theorem new_ellipse_and_hyperbola (x y c d a : ℝ) :
  (on_new_ellipse x y c d a ↔ 
    |x - c| + |y - d| + |x + c| + |y + d| = 2 * a) ∧
  (on_new_hyperbola x y c d a ↔ 
    |(|x - c| + |y - d|) - (|x + c| + |y + d|)| = 2 * a) :=
by sorry

end new_ellipse_and_hyperbola_l2663_266376


namespace min_value_theorem_l2663_266338

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y - 9 = 0) :
  2/y + 1/x ≥ 1 ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + 2*y' - 9 = 0 ∧ 2/y' + 1/x' = 1 :=
by sorry

end min_value_theorem_l2663_266338


namespace polynomial_remainder_l2663_266330

theorem polynomial_remainder (x : ℝ) : 
  (x^4 + x^3 + 1) % (x - 2) = 25 := by
sorry

end polynomial_remainder_l2663_266330


namespace sun_radius_scientific_notation_l2663_266328

-- Define the radius of the sun in kilometers
def sun_radius_km : ℝ := 696000

-- Define the conversion factor from kilometers to meters
def km_to_m : ℝ := 1000

-- Theorem to prove
theorem sun_radius_scientific_notation :
  sun_radius_km * km_to_m = 6.96 * (10 ^ 8) :=
by sorry

end sun_radius_scientific_notation_l2663_266328


namespace dandelion_seed_percentage_l2663_266348

def num_sunflowers : ℕ := 6
def num_dandelions : ℕ := 8
def seeds_per_sunflower : ℕ := 9
def seeds_per_dandelion : ℕ := 12

def total_seeds : ℕ := num_sunflowers * seeds_per_sunflower + num_dandelions * seeds_per_dandelion
def dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion

theorem dandelion_seed_percentage :
  (dandelion_seeds : ℚ) / total_seeds * 100 = 64 := by
  sorry

end dandelion_seed_percentage_l2663_266348


namespace f_max_at_zero_l2663_266302

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_max_at_zero :
  ∃ (a : ℝ), ∀ (x : ℝ), f x ≤ f a ∧ a = 0 :=
sorry

end f_max_at_zero_l2663_266302
