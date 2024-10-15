import Mathlib

namespace NUMINAMATH_CALUDE_workshop_average_salary_l3972_397226

/-- Represents the average salary of all workers in a workshop -/
def average_salary (total_workers : ℕ) (technicians : ℕ) (technician_salary : ℕ) (other_salary : ℕ) : ℚ :=
  ((technicians * technician_salary + (total_workers - technicians) * other_salary) : ℚ) / total_workers

/-- Theorem stating the average salary of all workers in the workshop -/
theorem workshop_average_salary :
  average_salary 24 8 12000 6000 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3972_397226


namespace NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l3972_397216

theorem semicircles_to_circle_area_ratio (r : ℝ) (hr : r > 0) : 
  (2 * (π * r^2 / 2)) / (π * r^2) = 1 := by sorry

end NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l3972_397216


namespace NUMINAMATH_CALUDE_no_404_games_tournament_l3972_397255

theorem no_404_games_tournament : ¬ ∃ (n : ℕ), n > 0 ∧ n * (n - 4) / 2 = 404 := by sorry

end NUMINAMATH_CALUDE_no_404_games_tournament_l3972_397255


namespace NUMINAMATH_CALUDE_unimodal_peak_interval_peak_interval_length_specific_peak_interval_l3972_397218

/-- A unimodal function on [0,1] is a function that is monotonically increasing
    on [0,x*] and monotonically decreasing on [x*,1] for some x* in (0,1) -/
def UnimodalFunction (f : ℝ → ℝ) : Prop := 
  ∃ x_star : ℝ, 0 < x_star ∧ x_star < 1 ∧ 
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ x_star → f x ≤ f y) ∧
  (∀ x y : ℝ, x_star ≤ x ∧ x < y ∧ y ≤ 1 → f x ≥ f y)

/-- The peak interval of a unimodal function contains the peak point -/
def PeakInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  UnimodalFunction f ∧ 0 ≤ a ∧ b ≤ 1 ∧
  ∃ x_star : ℝ, a < x_star ∧ x_star < b ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ x_star → f x ≤ f y) ∧
  (∀ x y : ℝ, x_star ≤ x ∧ x < y ∧ y ≤ 1 → f x ≥ f y)

theorem unimodal_peak_interval 
  (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h_unimodal : UnimodalFunction f)
  (h_x₁ : 0 < x₁) (h_x₂ : x₂ < 1) (h_order : x₁ < x₂) :
  (f x₁ ≥ f x₂ → PeakInterval f 0 x₂) ∧
  (f x₁ ≤ f x₂ → PeakInterval f x₁ 1) := by sorry

theorem peak_interval_length 
  (f : ℝ → ℝ) (r : ℝ) 
  (h_unimodal : UnimodalFunction f)
  (h_r : 0 < r ∧ r < 0.5) :
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₂ < 1 ∧ x₁ < x₂ ∧ x₂ - x₁ ≥ 2*r ∧
  ((PeakInterval f 0 x₂ ∧ x₂ ≤ 0.5 + r) ∨
   (PeakInterval f x₁ 1 ∧ 1 - x₁ ≤ 0.5 + r)) := by sorry

theorem specific_peak_interval 
  (f : ℝ → ℝ) 
  (h_unimodal : UnimodalFunction f) :
  ∃ x₁ x₂ x₃ : ℝ, 
    x₁ = 0.34 ∧ x₂ = 0.66 ∧ x₃ = 0.32 ∧
    PeakInterval f 0 x₂ ∧
    PeakInterval f 0 x₁ ∧
    |x₁ - x₂| ≥ 0.02 ∧ |x₁ - x₃| ≥ 0.02 ∧ |x₂ - x₃| ≥ 0.02 := by sorry

end NUMINAMATH_CALUDE_unimodal_peak_interval_peak_interval_length_specific_peak_interval_l3972_397218


namespace NUMINAMATH_CALUDE_planks_per_tree_value_l3972_397217

/-- The number of planks John can make from each tree -/
def planks_per_tree : ℕ := sorry

/-- The number of trees John chops down -/
def num_trees : ℕ := 30

/-- The number of planks needed to make one table -/
def planks_per_table : ℕ := 15

/-- The selling price of one table in dollars -/
def table_price : ℕ := 300

/-- The total labor cost in dollars -/
def labor_cost : ℕ := 3000

/-- The total profit in dollars -/
def total_profit : ℕ := 12000

/-- Theorem stating the number of planks John can make from each tree -/
theorem planks_per_tree_value : planks_per_tree = 25 := by sorry

end NUMINAMATH_CALUDE_planks_per_tree_value_l3972_397217


namespace NUMINAMATH_CALUDE_c_gains_thousand_l3972_397231

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  house_value : Option Int

/-- Represents a house transaction -/
inductive Transaction
  | Buy (price : Int)
  | Sell (price : Int)

def initial_c : FinancialState := { cash := 15000, house_value := some 12000 }
def initial_d : FinancialState := { cash := 16000, house_value := none }

def house_appreciation : Int := 13000
def house_depreciation : Int := 11000

def apply_transaction (state : FinancialState) (t : Transaction) : FinancialState :=
  match t with
  | Transaction.Buy price => { cash := state.cash - price, house_value := some price }
  | Transaction.Sell price => { cash := state.cash + price, house_value := none }

def net_worth (state : FinancialState) : Int :=
  state.cash + state.house_value.getD 0

theorem c_gains_thousand (c d : FinancialState → FinancialState) :
  c = (λ s => apply_transaction s (Transaction.Sell house_appreciation)) ∘
      (λ s => apply_transaction s (Transaction.Buy house_depreciation)) ∘
      (λ s => { s with house_value := some house_appreciation }) →
  d = (λ s => apply_transaction s (Transaction.Buy house_appreciation)) ∘
      (λ s => apply_transaction s (Transaction.Sell house_depreciation)) →
  net_worth (c initial_c) - net_worth initial_c = 1000 :=
sorry

end NUMINAMATH_CALUDE_c_gains_thousand_l3972_397231


namespace NUMINAMATH_CALUDE_inequality_proof_l3972_397254

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hsum : a + b + c = 1) :
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧ 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1+a)*(1+b)*(1+c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3972_397254


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3972_397225

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : a + b + c = -a/2
  h3 : 3 * a > 2 * c
  h4 : 2 * c > 2 * b

/-- The main theorem about the properties of the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  (-3 < f.b / f.a ∧ f.b / f.a < -3/4) ∧
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f.a * x^2 + f.b * x + f.c = 0) ∧
  (∀ x₁ x₂ : ℝ, f.a * x₁^2 + f.b * x₁ + f.c = 0 → f.a * x₂^2 + f.b * x₂ + f.c = 0 →
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3972_397225


namespace NUMINAMATH_CALUDE_game_ends_in_25_rounds_l3972_397269

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- The state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)
  (round : ℕ)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 16
    | Player.B => 15
    | Player.C => 14
    | Player.D => 13,
    round := 0 }

/-- Determines if the game has ended (i.e., if any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- The theorem to prove -/
theorem game_ends_in_25_rounds :
  ∃ finalState : GameState,
    finalState.round = 25 ∧
    gameEnded finalState ∧
    (∀ prevState : GameState, prevState.round < 25 → ¬gameEnded prevState) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_in_25_rounds_l3972_397269


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l3972_397239

theorem smallest_n_multiple_of_five (x y : ℤ) 
  (hx : ∃ k : ℤ, x + 2 = 5 * k) 
  (hy : ∃ k : ℤ, y - 2 = 5 * k) : 
  (∃ n : ℕ+, ∃ m : ℤ, x^2 + x*y + y^2 + n = 5 * m ∧ 
   ∀ k : ℕ+, k < n → ¬∃ m : ℤ, x^2 + x*y + y^2 + k = 5 * m) → 
  (∃ n : ℕ+, n = 1 ∧ ∃ m : ℤ, x^2 + x*y + y^2 + n = 5 * m ∧ 
   ∀ k : ℕ+, k < n → ¬∃ m : ℤ, x^2 + x*y + y^2 + k = 5 * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l3972_397239


namespace NUMINAMATH_CALUDE_tower_height_difference_l3972_397297

theorem tower_height_difference : 
  ∀ (h_clyde h_grace : ℕ), 
  h_grace = 8 * h_clyde → 
  h_grace = 40 → 
  h_grace - h_clyde = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_tower_height_difference_l3972_397297


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3972_397253

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3972_397253


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3972_397256

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 6 ∧ x ≠ -3 →
  (4 * x + 7) / (x^2 - 3*x - 18) = (31/9) / (x - 6) + (5/9) / (x + 3) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3972_397256


namespace NUMINAMATH_CALUDE_expression_equality_l3972_397287

theorem expression_equality : 
  (2^3 ≠ 3^2) ∧ 
  ((-2)^3 ≠ (-3)^2) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  ((-2)^3 = (-2^3)) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3972_397287


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3972_397286

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 → f x ≤ f c ∧
  f c = 11 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3972_397286


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l3972_397236

theorem max_value_of_sum_of_squares (x y : ℝ) :
  x^2 + y^2 = 3*x + 8*y → x^2 + y^2 ≤ 73 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l3972_397236


namespace NUMINAMATH_CALUDE_pizza_slices_l3972_397233

theorem pizza_slices : ∃ S : ℕ,
  S > 0 ∧
  (3 * S / 4 : ℚ) > 0 ∧
  (9 * S / 16 : ℚ) > 4 ∧
  (9 * S / 16 : ℚ) - 4 = 5 ∧
  S = 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l3972_397233


namespace NUMINAMATH_CALUDE_group_size_proof_l3972_397208

/-- The number of men in a group where replacing one man increases the average weight by 2.5 kg, 
    and the difference between the new man's weight and the replaced man's weight is 25 kg. -/
def number_of_men : ℕ := 10

/-- The increase in average weight when one man is replaced. -/
def average_weight_increase : ℚ := 5/2

/-- The difference in weight between the new man and the replaced man. -/
def weight_difference : ℕ := 25

theorem group_size_proof : 
  number_of_men * average_weight_increase = weight_difference := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l3972_397208


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l3972_397202

/-- The surface area of a cuboid created by three cubes of side length 8 cm -/
theorem cuboid_surface_area : 
  let cube_side : ℝ := 8
  let cuboid_length : ℝ := 3 * cube_side
  let cuboid_width : ℝ := cube_side
  let cuboid_height : ℝ := cube_side
  let surface_area : ℝ := 2 * (cuboid_length * cuboid_width + 
                               cuboid_length * cuboid_height + 
                               cuboid_width * cuboid_height)
  surface_area = 896 := by
sorry


end NUMINAMATH_CALUDE_cuboid_surface_area_l3972_397202


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l3972_397272

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), (∀ d ∈ S, ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) ∧ 
                       (∀ d : ℕ+, (∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) → d ∈ S) ∧ 
                       S.card = 9) := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l3972_397272


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3972_397203

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20250 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3972_397203


namespace NUMINAMATH_CALUDE_initial_customers_l3972_397264

theorem initial_customers (remaining : ℕ) (left : ℕ) : 
  remaining = 5 → left = 3 → remaining + left = 8 :=
by sorry

end NUMINAMATH_CALUDE_initial_customers_l3972_397264


namespace NUMINAMATH_CALUDE_mean_proportional_segment_l3972_397212

theorem mean_proportional_segment (a b c : ℝ) : 
  a = 1 → b = 2 → c^2 = a * b → c > 0 → c = Real.sqrt 2 := by
  sorry

#check mean_proportional_segment

end NUMINAMATH_CALUDE_mean_proportional_segment_l3972_397212


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_values_l3972_397228

theorem perpendicular_lines_k_values (k : ℝ) :
  let l1 : ℝ → ℝ → ℝ := λ x y => (k - 3) * x + (k + 4) * y + 1
  let l2 : ℝ → ℝ → ℝ := λ x y => (k + 1) * x + 2 * (k - 3) * y + 3
  (∀ x y, l1 x y = 0 → l2 x y = 0 → (k - 3) * (k + 1) + 2 * (k + 4) * (k - 3) = 0) →
  k = 3 ∨ k = -3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_values_l3972_397228


namespace NUMINAMATH_CALUDE_cubic_expansion_property_l3972_397271

theorem cubic_expansion_property (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x, (Real.sqrt 3 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_property_l3972_397271


namespace NUMINAMATH_CALUDE_inner_square_area_l3972_397292

/-- Represents a square with side length and area -/
structure Square where
  side_length : ℝ
  area : ℝ

/-- Represents the configuration of two squares -/
structure SquareConfiguration where
  outer : Square
  inner : Square
  wi_length : ℝ

/-- Checks if the configuration is valid -/
def is_valid_configuration (config : SquareConfiguration) : Prop :=
  config.outer.side_length = 10 ∧
  config.wi_length = 3 ∧
  config.inner.area = config.inner.side_length ^ 2 ∧
  config.outer.area = config.outer.side_length ^ 2 ∧
  config.inner.side_length < config.outer.side_length

/-- The main theorem -/
theorem inner_square_area (config : SquareConfiguration) :
  is_valid_configuration config →
  config.inner.area = 21.16 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_area_l3972_397292


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3972_397268

theorem cubic_equation_roots (p q : ℝ) :
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
   ∀ x : ℝ, x^3 - 11*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  p + q = 78 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3972_397268


namespace NUMINAMATH_CALUDE_T_2023_mod_10_l3972_397243

/-- Represents a sequence of C's and D's -/
inductive Sequence : Type
| C : Sequence
| D : Sequence
| cons : Sequence → Sequence → Sequence

/-- Checks if a sequence is valid (no more than two consecutive C's or D's) -/
def isValid : Sequence → Bool
| Sequence.C => true
| Sequence.D => true
| Sequence.cons s₁ s₂ => sorry  -- Implementation details omitted

/-- Counts the number of valid sequences of length n -/
def T (n : ℕ+) : ℕ :=
  (List.map (fun s => if isValid s then 1 else 0) (sorry : List Sequence)).sum
  -- Implementation details omitted

/-- Main theorem: T(2023) is congruent to 6 modulo 10 -/
theorem T_2023_mod_10 : T 2023 % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_T_2023_mod_10_l3972_397243


namespace NUMINAMATH_CALUDE_Q_equals_G_l3972_397252

-- Define the sets Q and G
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem Q_equals_G : Q = G := by sorry

end NUMINAMATH_CALUDE_Q_equals_G_l3972_397252


namespace NUMINAMATH_CALUDE_smallest_valid_m_l3972_397273

def is_valid (m : ℕ+) : Prop :=
  ∃ k₁ k₂ : ℕ, k₁ ≤ m ∧ k₂ ≤ m ∧ 
  (m^2 + m) % k₁ = 0 ∧ 
  (m^2 + m) % k₂ ≠ 0

theorem smallest_valid_m :
  (∀ m : ℕ+, m < 4 → ¬(is_valid m)) ∧ 
  is_valid 4 := by sorry

end NUMINAMATH_CALUDE_smallest_valid_m_l3972_397273


namespace NUMINAMATH_CALUDE_water_ice_mixture_theorem_l3972_397201

/-- Represents the properties of water and ice mixture -/
structure WaterIceMixture where
  total_mass : ℝ
  water_mass : ℝ
  ice_mass : ℝ
  water_mass_added : ℝ
  initial_temp : ℝ
  final_temp : ℝ
  latent_heat_fusion : ℝ

/-- Calculates the heat balance for the water-ice mixture -/
def heat_balance (m : WaterIceMixture) : ℝ :=
  m.water_mass_added * (m.initial_temp - m.final_temp) -
  (m.ice_mass * m.latent_heat_fusion + m.total_mass * (m.final_temp - 0))

/-- Theorem stating that the original water mass in the mixture is 90.625g -/
theorem water_ice_mixture_theorem (m : WaterIceMixture) 
  (h1 : m.total_mass = 250)
  (h2 : m.water_mass_added = 1000)
  (h3 : m.initial_temp = 20)
  (h4 : m.final_temp = 5)
  (h5 : m.latent_heat_fusion = 80)
  (h6 : m.water_mass + m.ice_mass = m.total_mass)
  (h7 : heat_balance m = 0) :
  m.water_mass = 90.625 :=
sorry

end NUMINAMATH_CALUDE_water_ice_mixture_theorem_l3972_397201


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_l3972_397213

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) + (Real.cos x)^2 - 2

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_monotonically_decreasing (k : ℤ) :
  monotonically_decreasing f (π/3 + k*π) (5*π/3 + k*π) := by sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_l3972_397213


namespace NUMINAMATH_CALUDE_min_apples_collected_l3972_397263

theorem min_apples_collected (n : ℕ) 
  (h1 : n > 0)
  (h2 : ∃ (p1 p2 p3 p4 p5 : ℕ), 
    p1 + p2 + p3 + p4 + p5 = 100 ∧ 
    0 < p1 ∧ p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 < p5 ∧
    (∀ i ∈ [p1, p2, p3, p4], (i * (n * 7 / 10) % 100 = 0)))
  (h3 : ∀ m : ℕ, m < n → 
    ¬(∃ (q1 q2 q3 q4 q5 : ℕ), 
      q1 + q2 + q3 + q4 + q5 = 100 ∧ 
      0 < q1 ∧ q1 < q2 ∧ q2 < q3 ∧ q3 < q4 ∧ q4 < q5 ∧
      (∀ i ∈ [q1, q2, q3, q4], (i * (m * 7 / 10) % 100 = 0))))
  : n = 20 :=
sorry

end NUMINAMATH_CALUDE_min_apples_collected_l3972_397263


namespace NUMINAMATH_CALUDE_quadratic_inequality_bounds_l3972_397258

theorem quadratic_inequality_bounds (x : ℝ) (h : x^2 - 6*x + 8 < 0) :
  25 < x^2 + 6*x + 9 ∧ x^2 + 6*x + 9 < 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_bounds_l3972_397258


namespace NUMINAMATH_CALUDE_ice_melting_problem_l3972_397281

theorem ice_melting_problem (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.2) → 
  (original_volume = 3.2) :=
by
  sorry

end NUMINAMATH_CALUDE_ice_melting_problem_l3972_397281


namespace NUMINAMATH_CALUDE_no_prime_generating_pair_l3972_397219

theorem no_prime_generating_pair : ∀ a b : ℕ+, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p > 1000 ∧ q > 1000 ∧ p ≠ q ∧ ¬(Prime (a * p + b * q)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_generating_pair_l3972_397219


namespace NUMINAMATH_CALUDE_problem_statement_l3972_397295

theorem problem_statement (x y : ℝ) (h : 3 * y - x^2 = -5) :
  6 * y - 2 * x^2 - 6 = -16 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3972_397295


namespace NUMINAMATH_CALUDE_quadratic_roots_conditions_l3972_397206

variable (m : ℝ)

def quadratic_equation (x : ℝ) := m * x^2 + (m - 3) * x + 1

theorem quadratic_roots_conditions :
  (∀ x, quadratic_equation m x ≠ 0 → m > 1) ∧
  ((∃ x y, x ≠ y ∧ x > 0 ∧ y > 0 ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ↔ 0 < m ∧ m < 1) ∧
  ((∃ x y, x > 0 ∧ y < 0 ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ↔ m < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_conditions_l3972_397206


namespace NUMINAMATH_CALUDE_box_filled_with_large_cubes_l3972_397257

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.depth

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (c : Cube) : ℕ :=
  c.sideLength * c.sideLength * c.sideLength

/-- Theorem: A box with dimensions 50 × 60 × 43 inches can be filled completely with 1032 cubes of size 5 × 5 × 5 inches -/
theorem box_filled_with_large_cubes :
  let box := BoxDimensions.mk 50 60 43
  let largeCube := Cube.mk 5
  boxVolume box = 1032 * cubeVolume largeCube := by
  sorry


end NUMINAMATH_CALUDE_box_filled_with_large_cubes_l3972_397257


namespace NUMINAMATH_CALUDE_symmetric_equation_example_symmetric_equation_values_quadratic_equation_solutions_l3972_397279

-- Definition of symmetric equations
def is_symmetric (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₁ + a₂ = 0 ∧ b₁ = b₂ ∧ c₁ + c₂ = 0

-- Theorem 1: Symmetric equation of x² - 4x + 3 = 0
theorem symmetric_equation_example : 
  is_symmetric 1 (-4) 3 (-1) (-4) (-3) :=
sorry

-- Theorem 2: Finding m and n for symmetric equations
theorem symmetric_equation_values (m n : ℝ) :
  is_symmetric 3 (m - 1) (-n) (-3) (-1) 1 → m = 0 ∧ n = 1 :=
sorry

-- Theorem 3: Solutions of the quadratic equation
theorem quadratic_equation_solutions :
  let x₁ := (1 + Real.sqrt 13) / 6
  let x₂ := (1 - Real.sqrt 13) / 6
  3 * x₁^2 - x₁ - 1 = 0 ∧ 3 * x₂^2 - x₂ - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_equation_example_symmetric_equation_values_quadratic_equation_solutions_l3972_397279


namespace NUMINAMATH_CALUDE_combined_age_is_23_l3972_397284

/-- Represents the ages and relationships in the problem -/
structure AgeRelationship where
  person_age : ℕ
  dog_age : ℕ
  cat_age : ℕ
  sister_age : ℕ

/-- The conditions of the problem -/
def problem_conditions (ar : AgeRelationship) : Prop :=
  ar.person_age = ar.dog_age + 15 ∧
  ar.cat_age = ar.dog_age + 3 ∧
  ar.dog_age + 2 = 4 ∧
  ar.sister_age + 2 = 2 * (ar.dog_age + 2)

/-- The theorem to prove -/
theorem combined_age_is_23 (ar : AgeRelationship) 
  (h : problem_conditions ar) : 
  ar.person_age + ar.sister_age = 23 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_is_23_l3972_397284


namespace NUMINAMATH_CALUDE_taller_tree_is_84_feet_l3972_397220

def taller_tree_height (h1 h2 : ℝ) : Prop :=
  h1 > h2 ∧ h1 - h2 = 24 ∧ h2 / h1 = 5 / 7

theorem taller_tree_is_84_feet :
  ∃ (h1 h2 : ℝ), taller_tree_height h1 h2 ∧ h1 = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_taller_tree_is_84_feet_l3972_397220


namespace NUMINAMATH_CALUDE_tims_change_theorem_l3972_397237

/-- Calculates the change received after a purchase --/
def calculate_change (initial_amount : ℕ) (purchase_amount : ℕ) : ℕ :=
  initial_amount - purchase_amount

/-- Proves that the change received is correct for Tim's candy bar purchase --/
theorem tims_change_theorem :
  let initial_amount : ℕ := 50
  let purchase_amount : ℕ := 45
  calculate_change initial_amount purchase_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_tims_change_theorem_l3972_397237


namespace NUMINAMATH_CALUDE_base_equality_implies_three_l3972_397242

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

/-- Converts a number from an arbitrary base to base 10 -/
def baseNToBase10 (n b : ℕ) : ℕ :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem base_equality_implies_three :
  ∃! (b : ℕ), b > 0 ∧ base6ToBase10 35 = baseNToBase10 132 b :=
by
  sorry

end NUMINAMATH_CALUDE_base_equality_implies_three_l3972_397242


namespace NUMINAMATH_CALUDE_equal_utility_days_l3972_397290

/-- Daniel's utility function -/
def utility (reading : ℚ) (soccer : ℚ) : ℚ := reading * soccer

/-- Time spent on Wednesday -/
def wednesday (t : ℚ) : ℚ × ℚ := (10 - t, t)

/-- Time spent on Thursday -/
def thursday (t : ℚ) : ℚ × ℚ := (t + 4, 4 - t)

/-- The theorem stating that t = 8/5 makes the utility equal on both days -/
theorem equal_utility_days (t : ℚ) : 
  t = 8/5 ↔ 
  utility (wednesday t).1 (wednesday t).2 = utility (thursday t).1 (thursday t).2 := by
sorry

end NUMINAMATH_CALUDE_equal_utility_days_l3972_397290


namespace NUMINAMATH_CALUDE_intersection_and_union_subset_condition_l3972_397274

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Theorem for part (I)
theorem intersection_and_union :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) := by sorry

-- Theorem for part (II)
theorem subset_condition (k : ℝ) :
  {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1} ⊆ A ↔ k > 1 ∨ k < -5/2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_subset_condition_l3972_397274


namespace NUMINAMATH_CALUDE_smallest_multiple_l3972_397249

theorem smallest_multiple (n : ℕ) : n = 2349 ↔ 
  n > 0 ∧ 
  29 ∣ n ∧ 
  n % 97 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 29 ∣ m → m % 97 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3972_397249


namespace NUMINAMATH_CALUDE_n_over_8_equals_2_pow_3997_l3972_397204

theorem n_over_8_equals_2_pow_3997 (n : ℕ) : n = 16^1000 → n/8 = 2^3997 := by
  sorry

end NUMINAMATH_CALUDE_n_over_8_equals_2_pow_3997_l3972_397204


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_4_with_digit_sum_20_l3972_397282

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_four_digit_divisible_by_4_with_digit_sum_20 :
  ∃ (n : ℕ), is_four_digit n ∧ n % 4 = 0 ∧ digit_sum n = 20 ∧
  ∀ (m : ℕ), is_four_digit m → m % 4 = 0 → digit_sum m = 20 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_4_with_digit_sum_20_l3972_397282


namespace NUMINAMATH_CALUDE_n_must_be_even_l3972_397210

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem n_must_be_even (n : ℕ) 
  (h1 : n > 0)
  (h2 : sum_of_digits n = 2014)
  (h3 : sum_of_digits (5 * n) = 1007) :
  Even n := by
  sorry

end NUMINAMATH_CALUDE_n_must_be_even_l3972_397210


namespace NUMINAMATH_CALUDE_quadratic_function_problem_l3972_397293

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The value of j that satisfies the given conditions -/
def j : ℤ := 36

theorem quadratic_function_problem (a b c : ℤ) :
  f a b c 2 = 0 ∧
  200 < f a b c 10 ∧ f a b c 10 < 300 ∧
  400 < f a b c 9 ∧ f a b c 9 < 500 ∧
  1000 * j < f a b c 100 ∧ f a b c 100 < 1000 * (j + 1) →
  j = 36 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_problem_l3972_397293


namespace NUMINAMATH_CALUDE_factorization_1_l3972_397278

theorem factorization_1 (a : ℝ) : 3*a^3 - 6*a^2 + 3*a = 3*a*(a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_l3972_397278


namespace NUMINAMATH_CALUDE_distance_from_negative_three_point_two_l3972_397289

theorem distance_from_negative_three_point_two (x : ℝ) : 
  (|x + 3.2| = 4) ↔ (x = 0.8 ∨ x = -7.2) := by
  sorry

end NUMINAMATH_CALUDE_distance_from_negative_three_point_two_l3972_397289


namespace NUMINAMATH_CALUDE_all_acute_triangle_count_l3972_397246

/-- A function that checks if a triangle with sides a, b, c has all acute angles -/
def isAllAcuteTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧
  a * a + b * b > c * c ∧
  a * a + c * c > b * b ∧
  b * b + c * c > a * a

/-- The theorem stating that there are exactly 5 integer values of y that form an all-acute triangle with sides 15 and 8 -/
theorem all_acute_triangle_count :
  ∃! (s : Finset ℕ), s.card = 5 ∧ ∀ y ∈ s, isAllAcuteTriangle 15 8 y :=
sorry

end NUMINAMATH_CALUDE_all_acute_triangle_count_l3972_397246


namespace NUMINAMATH_CALUDE_range_of_a_l3972_397299

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + 4 < 0 ↔ a - 1 < x ∧ x < a + 1) → 
  (2 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3972_397299


namespace NUMINAMATH_CALUDE_walkway_time_proof_l3972_397266

theorem walkway_time_proof (walkway_length : ℝ) (time_against : ℝ) (time_stationary : ℝ)
  (h1 : walkway_length = 80)
  (h2 : time_against = 120)
  (h3 : time_stationary = 60) :
  let person_speed := walkway_length / time_stationary
  let walkway_speed := person_speed - walkway_length / time_against
  walkway_length / (person_speed + walkway_speed) = 40 := by
sorry

end NUMINAMATH_CALUDE_walkway_time_proof_l3972_397266


namespace NUMINAMATH_CALUDE_triangle_height_inradius_inequality_l3972_397247

theorem triangle_height_inradius_inequality 
  (h₁ h₂ h₃ r : ℝ) (α : ℝ) 
  (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ r > 0)
  (h_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    h₁ = 2 * (r * (a + b + c) / 2) / a ∧
    h₂ = 2 * (r * (a + b + c) / 2) / b ∧
    h₃ = 2 * (r * (a + b + c) / 2) / c)
  (h_alpha : α ≥ 1) :
  h₁^α + h₂^α + h₃^α ≥ 3 * (3 * r)^α := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_inradius_inequality_l3972_397247


namespace NUMINAMATH_CALUDE_school_emblem_estimate_l3972_397275

/-- Estimates the number of students who like a design in the entire school population
    based on a sample survey. -/
def estimate_liking (total_students : ℕ) (sample_size : ℕ) (sample_liking : ℕ) : ℕ :=
  (sample_liking * total_students) / sample_size

/-- Theorem stating that the estimated number of students liking design A
    in a school of 2000 students is 1200, given a survey where 60 out of 100
    students liked design A. -/
theorem school_emblem_estimate :
  let total_students : ℕ := 2000
  let sample_size : ℕ := 100
  let sample_liking : ℕ := 60
  estimate_liking total_students sample_size sample_liking = 1200 := by
sorry

end NUMINAMATH_CALUDE_school_emblem_estimate_l3972_397275


namespace NUMINAMATH_CALUDE_function_range_l3972_397234

theorem function_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*m*x + m + 2 = 0 ∧ y^2 - 2*m*y + m + 2 = 0) ∧ 
  (∀ x ≥ 1, ∀ y ≥ x, (y^2 - 2*m*y + m + 2) ≥ (x^2 - 2*m*x + m + 2)) →
  m < -1 :=
by sorry

end NUMINAMATH_CALUDE_function_range_l3972_397234


namespace NUMINAMATH_CALUDE_pattern_B_cannot_fold_into_tetrahedron_l3972_397227

-- Define the structure of a pattern
structure Pattern :=
  (squares : ℕ)
  (foldLines : ℕ)

-- Define the properties of a regular tetrahedron
structure RegularTetrahedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)
  (edgesPerVertex : ℕ)

-- Define the folding function (noncomputable as it's conceptual)
noncomputable def canFoldIntoTetrahedron (p : Pattern) : Prop := sorry

-- Define the specific patterns
def patternA : Pattern := ⟨4, 3⟩
def patternB : Pattern := ⟨4, 3⟩
def patternC : Pattern := ⟨4, 3⟩
def patternD : Pattern := ⟨4, 3⟩

-- Define the properties of a regular tetrahedron
def tetrahedron : RegularTetrahedron := ⟨4, 6, 4, 3⟩

-- State the theorem
theorem pattern_B_cannot_fold_into_tetrahedron :
  ¬(canFoldIntoTetrahedron patternB) :=
sorry

end NUMINAMATH_CALUDE_pattern_B_cannot_fold_into_tetrahedron_l3972_397227


namespace NUMINAMATH_CALUDE_factorization_proof_l3972_397283

theorem factorization_proof (x y : ℝ) : x * y^2 - x = x * (y + 1) * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3972_397283


namespace NUMINAMATH_CALUDE_probability_both_divisible_by_4_l3972_397244

/-- A fair 8-sided die -/
def EightSidedDie : Finset ℕ := Finset.range 8 

/-- The probability of an event occurring when tossing a fair 8-sided die -/
def prob (event : Finset ℕ) : ℚ :=
  event.card / EightSidedDie.card

/-- The set of outcomes divisible by 4 on an 8-sided die -/
def divisibleBy4 : Finset ℕ := Finset.filter (·.mod 4 = 0) EightSidedDie

/-- The probability of getting a number divisible by 4 on one 8-sided die -/
def probDivisibleBy4 : ℚ := prob divisibleBy4

theorem probability_both_divisible_by_4 :
  probDivisibleBy4 * probDivisibleBy4 = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_both_divisible_by_4_l3972_397244


namespace NUMINAMATH_CALUDE_max_perimeter_is_nine_l3972_397260

/-- Represents a configuration of three regular polygons meeting at a point -/
structure PolygonConfiguration where
  p : ℕ
  q : ℕ
  r : ℕ
  p_gt_two : p > 2
  q_gt_two : q > 2
  r_gt_two : r > 2
  distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r
  angle_sum : (p - 2) / p + (q - 2) / q + (r - 2) / r = 2

/-- The perimeter of the resulting polygon -/
def perimeter (config : PolygonConfiguration) : ℕ :=
  config.p + config.q + config.r - 6

/-- Theorem stating that the maximum perimeter is 9 -/
theorem max_perimeter_is_nine :
  ∀ config : PolygonConfiguration, perimeter config ≤ 9 ∧ ∃ config : PolygonConfiguration, perimeter config = 9 :=
sorry

end NUMINAMATH_CALUDE_max_perimeter_is_nine_l3972_397260


namespace NUMINAMATH_CALUDE_savings_account_percentage_l3972_397211

theorem savings_account_percentage (initial_amount : ℝ) (P : ℝ) : 
  initial_amount > 0 →
  (initial_amount + initial_amount * P / 100) * 0.8 = initial_amount →
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_savings_account_percentage_l3972_397211


namespace NUMINAMATH_CALUDE_car_insurance_present_value_l3972_397207

/-- Calculate the present value of a series of payments with annual growth and inflation --/
theorem car_insurance_present_value
  (initial_payment : ℝ)
  (insurance_growth_rate : ℝ)
  (inflation_rate : ℝ)
  (years : ℕ)
  (h1 : initial_payment = 3000)
  (h2 : insurance_growth_rate = 0.05)
  (h3 : inflation_rate = 0.02)
  (h4 : years = 10) :
  ∃ (pv : ℝ), abs (pv - ((initial_payment * ((1 + insurance_growth_rate) ^ years - 1) / insurance_growth_rate) / (1 + inflation_rate) ^ years)) < 0.01 ∧ 
  30954.87 < pv ∧ pv < 30954.89 :=
by
  sorry

end NUMINAMATH_CALUDE_car_insurance_present_value_l3972_397207


namespace NUMINAMATH_CALUDE_distance_to_axes_l3972_397238

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Theorem stating the distances from point P(3,5) to the x-axis and y-axis -/
theorem distance_to_axes :
  let P : Point := ⟨3, 5⟩
  distanceToXAxis P = 5 ∧ distanceToYAxis P = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_axes_l3972_397238


namespace NUMINAMATH_CALUDE_fraction_sum_equals_three_halves_l3972_397250

theorem fraction_sum_equals_three_halves (a b : ℕ+) :
  ∃ x y : ℕ+, (x : ℚ) / ((y : ℚ) + (a : ℚ)) + (y : ℚ) / ((x : ℚ) + (b : ℚ)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_three_halves_l3972_397250


namespace NUMINAMATH_CALUDE_price_drop_percentage_l3972_397261

/-- Proves that a 50% increase in quantity sold and a 20.000000000000014% increase in gross revenue
    implies a 20% decrease in price -/
theorem price_drop_percentage (P N : ℝ) (P' N' : ℝ) 
    (h_quantity_increase : N' = 1.5 * N)
    (h_revenue_increase : P' * N' = 1.20000000000000014 * (P * N)) : 
    P' = 0.8 * P := by
  sorry

end NUMINAMATH_CALUDE_price_drop_percentage_l3972_397261


namespace NUMINAMATH_CALUDE_add_decimals_l3972_397229

theorem add_decimals : (7.45 : ℝ) + 2.56 = 10.01 := by
  sorry

end NUMINAMATH_CALUDE_add_decimals_l3972_397229


namespace NUMINAMATH_CALUDE_hiring_theorem_l3972_397288

/-- Given probabilities for hiring three students A, B, and C --/
structure HiringProbabilities where
  probA : ℝ
  probNeitherANorB : ℝ
  probBothBAndC : ℝ

/-- The hiring probabilities satisfy the given conditions --/
def ValidHiringProbabilities (h : HiringProbabilities) : Prop :=
  h.probA = 2/3 ∧ h.probNeitherANorB = 1/12 ∧ h.probBothBAndC = 3/8

/-- Individual probabilities for B and C, and the probability of at least two being hired --/
structure HiringResults where
  probB : ℝ
  probC : ℝ
  probAtLeastTwo : ℝ

/-- The main theorem: given the conditions, prove the results --/
theorem hiring_theorem (h : HiringProbabilities) 
  (hvalid : ValidHiringProbabilities h) : 
  ∃ (r : HiringResults), r.probB = 3/4 ∧ r.probC = 1/2 ∧ r.probAtLeastTwo = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_hiring_theorem_l3972_397288


namespace NUMINAMATH_CALUDE_angle_from_terminal_point_l3972_397223

/-- Given an angle α in degrees where 0 ≤ α < 360, if a point on its terminal side
    has coordinates (sin 150°, cos 150°), then α = 300°. -/
theorem angle_from_terminal_point : ∀ α : ℝ,
  0 ≤ α → α < 360 →
  (∃ (x y : ℝ), x = Real.sin (150 * π / 180) ∧ y = Real.cos (150 * π / 180) ∧
    x = Real.sin (α * π / 180) ∧ y = Real.cos (α * π / 180)) →
  α = 300 := by
  sorry

end NUMINAMATH_CALUDE_angle_from_terminal_point_l3972_397223


namespace NUMINAMATH_CALUDE_blended_tea_selling_price_l3972_397200

/-- Calculates the selling price of a blended tea variety -/
theorem blended_tea_selling_price
  (cost1 : ℝ) (cost2 : ℝ) (ratio1 : ℝ) (ratio2 : ℝ) (gain_percent : ℝ)
  (h1 : cost1 = 18)
  (h2 : cost2 = 20)
  (h3 : ratio1 = 5)
  (h4 : ratio2 = 3)
  (h5 : gain_percent = 12)
  : (cost1 * ratio1 + cost2 * ratio2) / (ratio1 + ratio2) * (1 + gain_percent / 100) = 21 := by
  sorry

#check blended_tea_selling_price

end NUMINAMATH_CALUDE_blended_tea_selling_price_l3972_397200


namespace NUMINAMATH_CALUDE_problem_1_problem_2a_problem_2b_problem_2c_l3972_397267

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

-- State the theorems
theorem problem_1 : 27^(2/3) + Real.log 5 / Real.log 10 - 2 * Real.log 3 / Real.log 2 + Real.log 2 / Real.log 10 + Real.log 9 / Real.log 2 = 10 := by sorry

theorem problem_2a : f (-Real.sqrt 2) = 8 + 5 * Real.sqrt 2 := by sorry

theorem problem_2b (a : ℝ) : f (-a) = 3 * a^2 + 5 * a + 2 := by sorry

theorem problem_2c (a : ℝ) : f (a + 3) = 3 * a^2 + 13 * a + 14 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2a_problem_2b_problem_2c_l3972_397267


namespace NUMINAMATH_CALUDE_injective_properties_l3972_397251

variable {A B : Type}
variable (f : A → B)

theorem injective_properties (h : Function.Injective f) :
  (∀ (x₁ x₂ : A), x₁ ≠ x₂ → f x₁ ≠ f x₂) ∧
  (∀ (b : B), ∃! (a : A), f a = b) :=
by sorry

end NUMINAMATH_CALUDE_injective_properties_l3972_397251


namespace NUMINAMATH_CALUDE_correct_articles_for_problem_l3972_397280

/-- Represents the possible articles that can be used before a noun -/
inductive Article
  | A
  | An
  | The
  | None

/-- Represents a noun with its properties -/
structure Noun where
  word : String
  startsWithSilentH : Bool
  isCountable : Bool

/-- Represents a fixed phrase -/
structure FixedPhrase where
  phrase : String
  meaning : String

/-- Function to determine the correct article for a noun -/
def correctArticle (n : Noun) : Article := sorry

/-- Function to determine the correct article for a fixed phrase -/
def correctPhraseArticle (fp : FixedPhrase) : Article := sorry

/-- Theorem stating the correct articles for the given problem -/
theorem correct_articles_for_problem 
  (hour : Noun)
  (out_of_question : FixedPhrase)
  (h1 : hour.word = "hour")
  (h2 : hour.startsWithSilentH = true)
  (h3 : hour.isCountable = true)
  (h4 : out_of_question.phrase = "out of __ question")
  (h5 : out_of_question.meaning = "impossible") :
  correctArticle hour = Article.An ∧ correctPhraseArticle out_of_question = Article.The := by
  sorry

end NUMINAMATH_CALUDE_correct_articles_for_problem_l3972_397280


namespace NUMINAMATH_CALUDE_complex_equality_proof_l3972_397215

theorem complex_equality_proof (n : ℤ) (h : 0 ≤ n ∧ n ≤ 13) : 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.cos (2 * n * π / 14) + Complex.I * Complex.sin (2 * n * π / 14) → n = 5 :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_proof_l3972_397215


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1980_l3972_397222

theorem largest_perfect_square_factor_of_1980 : 
  ∃ (n : ℕ), n^2 = 36 ∧ n^2 ∣ 1980 ∧ ∀ (m : ℕ), m^2 ∣ 1980 → m^2 ≤ 36 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1980_l3972_397222


namespace NUMINAMATH_CALUDE_multiple_of_six_l3972_397214

theorem multiple_of_six (n : ℤ) 
  (h : ∃ k : ℤ, (n^5 / 120) + (n^3 / 24) + (n / 30) = k) : 
  ∃ m : ℤ, n = 6 * m :=
by
  sorry

end NUMINAMATH_CALUDE_multiple_of_six_l3972_397214


namespace NUMINAMATH_CALUDE_second_bucket_capacity_l3972_397241

/-- Proves that given a tank of 48 liters and two buckets, where one bucket has a capacity of 4 liters
    and is used 4 times less than the other bucket to fill the tank, the capacity of the second bucket is 3 liters. -/
theorem second_bucket_capacity
  (tank_capacity : ℕ)
  (first_bucket_capacity : ℕ)
  (usage_difference : ℕ)
  (h1 : tank_capacity = 48)
  (h2 : first_bucket_capacity = 4)
  (h3 : usage_difference = 4)
  (h4 : ∃ (second_bucket_capacity : ℕ),
    tank_capacity / first_bucket_capacity = tank_capacity / second_bucket_capacity - usage_difference) :
  ∃ (second_bucket_capacity : ℕ), second_bucket_capacity = 3 :=
by sorry

end NUMINAMATH_CALUDE_second_bucket_capacity_l3972_397241


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l3972_397230

/-- The volume of a sphere inscribed in a cube with edge length 6 inches is 36π cubic inches. -/
theorem volume_of_inscribed_sphere (π : ℝ) : ℝ := by
  -- Define the edge length of the cube
  let cube_edge : ℝ := 6

  -- Define the radius of the inscribed sphere
  let sphere_radius : ℝ := cube_edge / 2

  -- Define the volume of the sphere
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3

  -- Prove that the volume equals 36π
  sorry

#check volume_of_inscribed_sphere

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l3972_397230


namespace NUMINAMATH_CALUDE_negative_quartic_count_l3972_397294

theorem negative_quartic_count : 
  (∃ (S : Finset Int), (∀ x : Int, x ∈ S ↔ x^4 - 63*x^2 + 126 < 0) ∧ Finset.card S = 12) :=
by sorry

end NUMINAMATH_CALUDE_negative_quartic_count_l3972_397294


namespace NUMINAMATH_CALUDE_sum_of_integers_l3972_397276

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 130) (h2 : x * y = 27) : 
  x + y = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3972_397276


namespace NUMINAMATH_CALUDE_min_coefficient_value_l3972_397205

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 30 * x^2 + box * x + 30) →
  a ≤ 15 →
  b ≤ 15 →
  a * b = 30 →
  box = a^2 + b^2 →
  61 ≤ box :=
by sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l3972_397205


namespace NUMINAMATH_CALUDE_reggie_long_shots_l3972_397296

/-- Represents the number of points for each type of shot --/
inductive ShotType
  | layup : ShotType
  | freeThrow : ShotType
  | longShot : ShotType

def shotValue : ShotType → ℕ
  | ShotType.layup => 1
  | ShotType.freeThrow => 2
  | ShotType.longShot => 3

/-- Represents the number of shots made by each player --/
structure ShotsMade where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

def totalPoints (shots : ShotsMade) : ℕ :=
  shots.layups * shotValue ShotType.layup +
  shots.freeThrows * shotValue ShotType.freeThrow +
  shots.longShots * shotValue ShotType.longShot

theorem reggie_long_shots
  (reggie : ShotsMade)
  (reggie_brother : ShotsMade)
  (h1 : reggie.layups = 3)
  (h2 : reggie.freeThrows = 2)
  (h3 : reggie_brother.layups = 0)
  (h4 : reggie_brother.freeThrows = 0)
  (h5 : reggie_brother.longShots = 4)
  (h6 : totalPoints reggie + 2 = totalPoints reggie_brother) :
  reggie.longShots = 1 := by
  sorry

end NUMINAMATH_CALUDE_reggie_long_shots_l3972_397296


namespace NUMINAMATH_CALUDE_P_is_ellipse_l3972_397221

-- Define the set of points P(x,y) satisfying the given equation
def P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x + 4)^2 + y^2) + Real.sqrt ((x - 4)^2 + y^2) = 10}

-- Define an ellipse with foci at (-4, 0) and (4, 0), and sum of distances equal to 10
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x + 4)^2 + y^2) + Real.sqrt ((x - 4)^2 + y^2) = 10}

-- Theorem stating that the set P is equivalent to the Ellipse
theorem P_is_ellipse : P = Ellipse := by sorry

end NUMINAMATH_CALUDE_P_is_ellipse_l3972_397221


namespace NUMINAMATH_CALUDE_average_cost_calculation_l3972_397209

/-- Calculates the average cost of products sold given the quantities and prices of different product types -/
theorem average_cost_calculation
  (iphone_quantity : ℕ) (iphone_price : ℕ)
  (ipad_quantity : ℕ) (ipad_price : ℕ)
  (appletv_quantity : ℕ) (appletv_price : ℕ)
  (h1 : iphone_quantity = 100)
  (h2 : iphone_price = 1000)
  (h3 : ipad_quantity = 20)
  (h4 : ipad_price = 900)
  (h5 : appletv_quantity = 80)
  (h6 : appletv_price = 200) :
  (iphone_quantity * iphone_price + ipad_quantity * ipad_price + appletv_quantity * appletv_price) /
  (iphone_quantity + ipad_quantity + appletv_quantity) = 670 :=
by sorry

end NUMINAMATH_CALUDE_average_cost_calculation_l3972_397209


namespace NUMINAMATH_CALUDE_largest_quantity_l3972_397285

theorem largest_quantity (a b c d e : ℝ) 
  (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5 ∧ a - 2 = e + 1) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e :=
sorry

end NUMINAMATH_CALUDE_largest_quantity_l3972_397285


namespace NUMINAMATH_CALUDE_valentines_given_to_children_l3972_397245

theorem valentines_given_to_children (initial : ℕ) (remaining : ℕ) :
  initial = 30 → remaining = 22 → initial - remaining = 8 := by
  sorry

end NUMINAMATH_CALUDE_valentines_given_to_children_l3972_397245


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l3972_397262

/-- The height of a cylinder with radius 12 inches that has the same volume as a sphere with radius 3 inches is 1/4 inch. -/
theorem melted_ice_cream_height : 
  ∀ (r_sphere r_cylinder : ℝ) (h : ℝ),
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h →
  h = 1 / 4 := by
sorry


end NUMINAMATH_CALUDE_melted_ice_cream_height_l3972_397262


namespace NUMINAMATH_CALUDE_icosidodecahedron_vertices_icosidodecahedron_vertices_proof_l3972_397248

/-- An icosidodecahedron is a convex polyhedron with 20 triangular faces and 12 pentagonal faces. -/
structure Icosidodecahedron where
  /-- The number of triangular faces -/
  triangular_faces : ℕ
  /-- The number of pentagonal faces -/
  pentagonal_faces : ℕ
  /-- The icosidodecahedron is a convex polyhedron -/
  is_convex : Bool
  /-- The number of triangular faces is 20 -/
  triangular_faces_eq : triangular_faces = 20
  /-- The number of pentagonal faces is 12 -/
  pentagonal_faces_eq : pentagonal_faces = 12

/-- The number of vertices in an icosidodecahedron is 30 -/
theorem icosidodecahedron_vertices (i : Icosidodecahedron) : ℕ := 30

/-- The number of vertices in an icosidodecahedron is 30 -/
theorem icosidodecahedron_vertices_proof (i : Icosidodecahedron) : 
  icosidodecahedron_vertices i = 30 := by
  sorry

end NUMINAMATH_CALUDE_icosidodecahedron_vertices_icosidodecahedron_vertices_proof_l3972_397248


namespace NUMINAMATH_CALUDE_shirts_to_wash_l3972_397232

def washing_machine_capacity : ℕ := 7
def number_of_sweaters : ℕ := 33
def number_of_loads : ℕ := 5

theorem shirts_to_wash (shirts : ℕ) : 
  shirts = number_of_loads * washing_machine_capacity - number_of_sweaters :=
by sorry

end NUMINAMATH_CALUDE_shirts_to_wash_l3972_397232


namespace NUMINAMATH_CALUDE_owen_final_count_l3972_397224

/-- The number of turtles Owen has after all transformations and donations -/
def final_owen_turtles (initial_owen : ℕ) (johanna_difference : ℕ) : ℕ :=
  let initial_johanna := initial_owen - johanna_difference
  let owen_after_month := initial_owen * 2
  let johanna_after_month := initial_johanna / 2
  owen_after_month + johanna_after_month

/-- Theorem stating that Owen ends up with 50 turtles -/
theorem owen_final_count :
  final_owen_turtles 21 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_owen_final_count_l3972_397224


namespace NUMINAMATH_CALUDE_expression_evaluation_l3972_397240

theorem expression_evaluation : 5 * 12 + 2 * 15 - (3 * 7 + 4 * 6) = 45 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3972_397240


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l3972_397265

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l3972_397265


namespace NUMINAMATH_CALUDE_rectangle_area_l3972_397291

theorem rectangle_area (d : ℝ) (h : d > 0) : ∃ (w l : ℝ),
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = d^2 ∧ w * l = (3 / 10) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3972_397291


namespace NUMINAMATH_CALUDE_negative_rational_function_interval_l3972_397270

theorem negative_rational_function_interval (x : ℝ) :
  x ≠ 3 →
  ((x - 5) / ((x - 3)^2) < 0) ↔ (3 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_negative_rational_function_interval_l3972_397270


namespace NUMINAMATH_CALUDE_power_sum_of_i_l3972_397235

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23456 + i^23457 + i^23458 + i^23459 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l3972_397235


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l3972_397298

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

/-- Theorem: A convex heptagon has 14 diagonals -/
theorem heptagon_diagonals : num_diagonals heptagon_sides = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l3972_397298


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l3972_397277

theorem division_multiplication_equality : -3 / (1/2) * 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l3972_397277


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3972_397259

theorem unique_triple_solution :
  ∃! (s : Set (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), (x, y, z) ∈ s ↔ 
      (1 + x^4 ≤ 2*(y - z)^2 ∧
       1 + y^4 ≤ 2*(z - x)^2 ∧
       1 + z^4 ≤ 2*(x - y)^2)) ∧
    (s = {(1, 0, -1), (1, -1, 0), (0, 1, -1), (0, -1, 1), (-1, 1, 0), (-1, 0, 1)}) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3972_397259
