import Mathlib

namespace NUMINAMATH_CALUDE_toothpicks_for_ten_squares_toothpicks_for_one_square_toothpicks_for_two_squares_toothpicks_pattern_l1317_131745

/-- The number of toothpicks required to form n squares in a row -/
def toothpicks (n : ℕ) : ℕ := 4 + 3 * (n - 1)

theorem toothpicks_for_ten_squares : 
  toothpicks 10 = 31 := by sorry

theorem toothpicks_for_one_square : 
  toothpicks 1 = 4 := by sorry

theorem toothpicks_for_two_squares : 
  toothpicks 2 = 7 := by sorry

theorem toothpicks_pattern (n : ℕ) (h : n > 1) : 
  toothpicks n = toothpicks (n-1) + 3 := by sorry

end NUMINAMATH_CALUDE_toothpicks_for_ten_squares_toothpicks_for_one_square_toothpicks_for_two_squares_toothpicks_pattern_l1317_131745


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_three_million_two_hundred_thousand_satisfies_conditions_smallest_n_is_three_million_two_hundred_thousand_l1317_131713

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by n 20 ∧ is_perfect_square (n^2) ∧ is_perfect_fifth_power (n^3)

theorem smallest_n_satisfying_conditions :
  ∀ m : ℕ, m > 0 → satisfies_conditions m → m ≥ 3200000 :=
by sorry

theorem three_million_two_hundred_thousand_satisfies_conditions :
  satisfies_conditions 3200000 :=
by sorry

theorem smallest_n_is_three_million_two_hundred_thousand :
  (∀ m : ℕ, m > 0 → satisfies_conditions m → m ≥ 3200000) ∧
  satisfies_conditions 3200000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_three_million_two_hundred_thousand_satisfies_conditions_smallest_n_is_three_million_two_hundred_thousand_l1317_131713


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1317_131734

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 3 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1317_131734


namespace NUMINAMATH_CALUDE_senior_citizen_tickets_l1317_131754

theorem senior_citizen_tickets (total_tickets : ℕ) (adult_price senior_price : ℕ) (total_receipts : ℕ) 
  (h1 : total_tickets = 529)
  (h2 : adult_price = 25)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 9745) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 348 := by
  sorry

end NUMINAMATH_CALUDE_senior_citizen_tickets_l1317_131754


namespace NUMINAMATH_CALUDE_friend_team_assignment_l1317_131703

theorem friend_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 :=
sorry

end NUMINAMATH_CALUDE_friend_team_assignment_l1317_131703


namespace NUMINAMATH_CALUDE_odd_symmetric_points_range_l1317_131764

theorem odd_symmetric_points_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ Real.exp x₀ - a = -(Real.exp (-x₀) - a)) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_symmetric_points_range_l1317_131764


namespace NUMINAMATH_CALUDE_max_point_difference_is_n_l1317_131707

/-- Represents a hockey tournament with n teams -/
structure HockeyTournament where
  n : ℕ
  n_ge_2 : n ≥ 2

/-- The maximum point difference between consecutively ranked teams -/
def maxPointDifference (t : HockeyTournament) : ℕ := t.n

/-- Theorem: The maximum point difference between consecutively ranked teams is n -/
theorem max_point_difference_is_n (t : HockeyTournament) : 
  maxPointDifference t = t.n := by sorry

end NUMINAMATH_CALUDE_max_point_difference_is_n_l1317_131707


namespace NUMINAMATH_CALUDE_james_return_to_heavy_lifting_l1317_131733

/-- Calculates the total number of days before James can return to heavy lifting after his injury. -/
def time_to_heavy_lifting (initial_pain_days : ℕ) (healing_multiplier : ℕ) (additional_rest_days : ℕ) 
  (light_exercise_weeks : ℕ) (moderate_exercise_weeks : ℕ) (final_wait_weeks : ℕ) : ℕ :=
  let full_healing_days := initial_pain_days * healing_multiplier
  let total_rest_days := full_healing_days + additional_rest_days
  let light_exercise_days := light_exercise_weeks * 7
  let moderate_exercise_days := moderate_exercise_weeks * 7
  let final_wait_days := final_wait_weeks * 7
  total_rest_days + light_exercise_days + moderate_exercise_days + final_wait_days

/-- Theorem stating that James can return to heavy lifting after 60 days. -/
theorem james_return_to_heavy_lifting : 
  time_to_heavy_lifting 3 5 3 2 1 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_return_to_heavy_lifting_l1317_131733


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1317_131727

theorem complex_equation_solution (n : ℝ) : 
  (1 : ℂ) / (1 + Complex.I) = (1 : ℂ) / 2 - n * Complex.I → n = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1317_131727


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1317_131701

theorem min_value_reciprocal_sum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1317_131701


namespace NUMINAMATH_CALUDE_sin_squared_minus_two_sin_range_l1317_131724

theorem sin_squared_minus_two_sin_range (x : ℝ) : -1 ≤ Real.sin x ^ 2 - 2 * Real.sin x ∧ Real.sin x ^ 2 - 2 * Real.sin x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_two_sin_range_l1317_131724


namespace NUMINAMATH_CALUDE_cone_volume_l1317_131770

/-- The volume of a cone with height h, whose lateral surface unfolds into a sector with a central angle of 120°, is πh³/24. -/
theorem cone_volume (h : ℝ) (h_pos : h > 0) : 
  ∃ (V : ℝ), V = (π * h^3) / 24 ∧ 
  V = (1/3) * π * (h^2 / 8) * h ∧
  ∃ (R : ℝ), R > 0 ∧ R^2 = h^2 / 8 ∧
  ∃ (l : ℝ), l > 0 ∧ l = 3 * R ∧
  2 * π * R = (2 * π * l) / 3 :=
sorry

end NUMINAMATH_CALUDE_cone_volume_l1317_131770


namespace NUMINAMATH_CALUDE_bus_max_capacity_l1317_131787

/-- Represents the seating capacity of a bus with specific arrangements -/
structure BusCapacity where
  left_regular : Nat
  left_priority : Nat
  right_regular : Nat
  right_priority : Nat
  back_row : Nat
  standing : Nat
  regular_capacity : Nat
  priority_capacity : Nat

/-- Calculates the total capacity of the bus -/
def total_capacity (bus : BusCapacity) : Nat :=
  bus.left_regular * bus.regular_capacity +
  bus.left_priority * bus.priority_capacity +
  bus.right_regular * bus.regular_capacity +
  bus.right_priority * bus.priority_capacity +
  bus.back_row +
  bus.standing

/-- Theorem stating that the maximum capacity of the bus is 94 -/
theorem bus_max_capacity :
  ∀ (bus : BusCapacity),
    bus.left_regular = 12 →
    bus.left_priority = 3 →
    bus.right_regular = 9 →
    bus.right_priority = 2 →
    bus.back_row = 7 →
    bus.standing = 14 →
    bus.regular_capacity = 3 →
    bus.priority_capacity = 2 →
    total_capacity bus = 94 := by
  sorry


end NUMINAMATH_CALUDE_bus_max_capacity_l1317_131787


namespace NUMINAMATH_CALUDE_total_gas_usage_l1317_131706

def gas_usage : List Float := [0.02, 0.015, 0.01, 0.03, 0.005, 0.025, 0.008, 0.018, 0.012, 0.005, 0.014, 0.01]

theorem total_gas_usage : 
  gas_usage.sum = 0.172 := by
  sorry

end NUMINAMATH_CALUDE_total_gas_usage_l1317_131706


namespace NUMINAMATH_CALUDE_third_side_of_triangle_l1317_131799

theorem third_side_of_triangle (a b c : ℝ) : 
  a = 3 → b = 5 → c = 4 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
  (c < a + b ∧ a < b + c ∧ b < c + a) := by
  sorry

end NUMINAMATH_CALUDE_third_side_of_triangle_l1317_131799


namespace NUMINAMATH_CALUDE_ball_bounce_height_l1317_131758

/-- Theorem: For a ball that rises with each bounce exactly one-half as high as it had fallen,
    and bounces 4 times, if the total distance traveled is 44.5 meters,
    then the initial height from which the ball was dropped is 9.9 meters. -/
theorem ball_bounce_height (h : ℝ) : 
  (h + 2*h + h + (1/2)*h + (1/4)*h = 44.5) → h = 9.9 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_height_l1317_131758


namespace NUMINAMATH_CALUDE_max_yellow_apples_removal_max_total_apples_removal_l1317_131750

/-- Represents the number of apples of each color in the basket -/
structure AppleBasket where
  green : Nat
  yellow : Nat
  red : Nat

/-- Represents the number of apples removed from the basket -/
structure RemovedApples where
  green : Nat
  yellow : Nat
  red : Nat

/-- Checks if the removal condition is satisfied -/
def validRemoval (removed : RemovedApples) : Prop :=
  removed.green < removed.yellow ∧ removed.yellow < removed.red

/-- The initial state of the apple basket -/
def initialBasket : AppleBasket :=
  ⟨8, 11, 16⟩

theorem max_yellow_apples_removal (basket : AppleBasket) 
  (h : basket = initialBasket) :
  ∃ (removed : RemovedApples), 
    validRemoval removed ∧ 
    removed.yellow = 11 ∧
    ∀ (other : RemovedApples), 
      validRemoval other → other.yellow ≤ removed.yellow :=
sorry

theorem max_total_apples_removal (basket : AppleBasket) 
  (h : basket = initialBasket) :
  ∃ (removed : RemovedApples),
    validRemoval removed ∧
    removed.green + removed.yellow + removed.red = 33 ∧
    ∀ (other : RemovedApples),
      validRemoval other →
      other.green + other.yellow + other.red ≤ removed.green + removed.yellow + removed.red :=
sorry

end NUMINAMATH_CALUDE_max_yellow_apples_removal_max_total_apples_removal_l1317_131750


namespace NUMINAMATH_CALUDE_no_prime_factor_congruent_to_negative_one_mod_eight_l1317_131744

theorem no_prime_factor_congruent_to_negative_one_mod_eight (n : ℕ+) (p : ℕ) 
  (h_prime : Nat.Prime p) (h_cong : p % 8 = 7) : 
  ¬(p ∣ 2^(n.val.succ.succ) + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_factor_congruent_to_negative_one_mod_eight_l1317_131744


namespace NUMINAMATH_CALUDE_transformed_system_solution_l1317_131711

theorem transformed_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * 6 + b₁ * 3 = c₁)
  (h₂ : a₂ * 6 + b₂ * 3 = c₂) :
  (4 * a₁ * 22 + 3 * b₁ * 33 = 11 * c₁) ∧ 
  (4 * a₂ * 22 + 3 * b₂ * 33 = 11 * c₂) := by
sorry

end NUMINAMATH_CALUDE_transformed_system_solution_l1317_131711


namespace NUMINAMATH_CALUDE_parabola_focus_l1317_131704

/-- The focus of the parabola y² = -8x is at the point (-2, 0) -/
theorem parabola_focus (x y : ℝ) : 
  y^2 = -8*x → (x + 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1317_131704


namespace NUMINAMATH_CALUDE_sin_120_degrees_l1317_131721

theorem sin_120_degrees (π : Real) :
  Real.sin (2 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l1317_131721


namespace NUMINAMATH_CALUDE_intersection_M_N_l1317_131763

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1317_131763


namespace NUMINAMATH_CALUDE_buratino_spent_10_dollars_l1317_131771

/-- Represents a transaction at the exchange point -/
inductive Transaction
  | type1 : Transaction  -- Give 2 euros, receive 3 dollars and a candy
  | type2 : Transaction  -- Give 5 dollars, receive 3 euros and a candy

/-- Represents Buratino's exchange operations -/
structure ExchangeOperations where
  transactions : List Transaction
  initial_dollars : ℕ
  final_dollars : ℕ
  final_euros : ℕ
  candies : ℕ

/-- The condition that Buratino's exchange operations are valid -/
def valid_exchange (ops : ExchangeOperations) : Prop :=
  ops.final_euros = 0 ∧
  ops.candies = ops.transactions.length ∧
  ops.candies = 50 ∧
  ops.final_dollars < ops.initial_dollars

/-- Calculate the net dollar change from a list of transactions -/
def net_dollar_change (transactions : List Transaction) : ℤ :=
  transactions.foldl (fun acc t => match t with
    | Transaction.type1 => acc + 3
    | Transaction.type2 => acc - 5
  ) 0

/-- The main theorem stating that Buratino spent 10 dollars -/
theorem buratino_spent_10_dollars (ops : ExchangeOperations) 
  (h : valid_exchange ops) : 
  ops.initial_dollars - ops.final_dollars = 10 := by
  sorry


end NUMINAMATH_CALUDE_buratino_spent_10_dollars_l1317_131771


namespace NUMINAMATH_CALUDE_max_expression_l1317_131731

theorem max_expression : ∀ a b c d : ℕ,
  a = 992 ∧ b = 993 ∧ c = 994 ∧ d = 995 →
  (d * (d + 1) + (d + 1) ≥ a * (a + 7) + (a + 7)) ∧
  (d * (d + 1) + (d + 1) ≥ b * (b + 5) + (b + 5)) ∧
  (d * (d + 1) + (d + 1) ≥ c * (c + 3) + (c + 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_max_expression_l1317_131731


namespace NUMINAMATH_CALUDE_giraffe_count_l1317_131785

/-- The number of giraffes in the zoo -/
def num_giraffes : ℕ := 5

/-- The number of penguins in the zoo -/
def num_penguins : ℕ := 10

/-- The number of elephants in the zoo -/
def num_elephants : ℕ := 2

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 50

theorem giraffe_count :
  (num_penguins = 2 * num_giraffes) ∧
  (num_penguins = (20 : ℕ) * total_animals / 100) ∧
  (num_elephants = (4 : ℕ) * total_animals / 100) ∧
  (num_elephants = 2) →
  num_giraffes = 5 := by
sorry

end NUMINAMATH_CALUDE_giraffe_count_l1317_131785


namespace NUMINAMATH_CALUDE_denis_numbers_sum_l1317_131748

theorem denis_numbers_sum : 
  ∀ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d → 
    a * d = 32 → 
    b * c = 14 → 
    a + b + c + d = 42 := by
  sorry

end NUMINAMATH_CALUDE_denis_numbers_sum_l1317_131748


namespace NUMINAMATH_CALUDE_find_number_l1317_131790

theorem find_number : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  n = sum * quotient + 20 ∧ n / sum = quotient ∧ n % sum = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1317_131790


namespace NUMINAMATH_CALUDE_triangle_area_l1317_131714

theorem triangle_area (a b c : ℝ) (ha : a = 9) (hb : b = 40) (hc : c = 41) :
  (1/2) * a * b = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1317_131714


namespace NUMINAMATH_CALUDE_at_least_one_made_by_bellini_l1317_131751

/-- Represents the maker of a casket -/
inductive Maker
  | Bellini
  | SonOfBellini
  | Other

/-- Represents a casket -/
structure Casket where
  material : String
  inscription : String
  maker : Maker

/-- The statement on the gold casket -/
def gold_inscription (silver : Casket) : Prop :=
  silver.maker = Maker.SonOfBellini

/-- The statement on the silver casket -/
def silver_inscription (gold : Casket) : Prop :=
  gold.maker ≠ Maker.SonOfBellini

/-- The main theorem -/
theorem at_least_one_made_by_bellini 
  (gold : Casket) 
  (silver : Casket) 
  (h_gold_material : gold.material = "gold")
  (h_silver_material : silver.material = "silver")
  (h_gold_inscription : gold.inscription = "The silver casket was made by the son of Bellini")
  (h_silver_inscription : silver.inscription = "The gold casket was not made by the son of Bellini") :
  gold.maker = Maker.Bellini ∨ silver.maker = Maker.Bellini :=
by
  sorry


end NUMINAMATH_CALUDE_at_least_one_made_by_bellini_l1317_131751


namespace NUMINAMATH_CALUDE_solution_equality_l1317_131719

theorem solution_equality (x y : ℝ) : 
  |x + y - 2| + (2 * x - 3 * y + 5)^2 = 0 → 
  ((x = 1 ∧ y = 9) ∨ (x = 5 ∧ y = 5)) := by
sorry

end NUMINAMATH_CALUDE_solution_equality_l1317_131719


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_fraction_division_simplification_l1317_131769

-- Problem 1
theorem fraction_sum_equals_one (m n : ℝ) (h : m ≠ n) :
  m / (m - n) + n / (n - m) = 1 := by sorry

-- Problem 2
theorem fraction_division_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1)) / (1 / (x + 1)) = 2 / (x - 1) := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_fraction_division_simplification_l1317_131769


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1317_131767

/-- A random variable following a normal distribution with mean 1 and standard deviation σ -/
def ξ (σ : ℝ) : Type := Unit

/-- The probability that ξ is less than a given value -/
def P_less_than (σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The theorem stating that if P(ξ < 0) = 0.3, then P(ξ < 2) = 0.7 for ξ ~ N(1, σ²) -/
theorem normal_distribution_probability (σ : ℝ) (h : P_less_than σ 0 = 0.3) :
  P_less_than σ 2 = 0.7 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1317_131767


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_equals_two_l1317_131746

/-- Given a curve y = x^3 + ax + 1, if there exists a point where the tangent line is y = 2x + 1, then a = 2 -/
theorem tangent_line_implies_a_equals_two (a : ℝ) : 
  (∃ x₀ y₀ : ℝ, y₀ = x₀^3 + a*x₀ + 1 ∧ 
    (∀ x : ℝ, (x - x₀) * (3*x₀^2 + a) + y₀ = 2*x + 1)) → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_equals_two_l1317_131746


namespace NUMINAMATH_CALUDE_square_polynomial_l1317_131779

theorem square_polynomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + k*x + 16 = (a*x + b)^2) → (k = 8 ∨ k = -8) := by
  sorry

end NUMINAMATH_CALUDE_square_polynomial_l1317_131779


namespace NUMINAMATH_CALUDE_sale_price_for_55_percent_profit_l1317_131732

/-- The sale price for a 55% profit given the conditions -/
theorem sale_price_for_55_percent_profit 
  (L : ℝ) -- The price at which the loss equals the profit when sold at $832
  (h1 : ∃ C : ℝ, 832 - C = C - L) -- Condition 1
  (h2 : ∃ C : ℝ, 992 = 0.55 * C) -- Condition 2
  : ∃ C : ℝ, C + 992 = 2795.64 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_for_55_percent_profit_l1317_131732


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1317_131739

theorem sqrt_difference_equality : Real.sqrt (49 + 49) - Real.sqrt (36 - 25) = 7 * Real.sqrt 2 - Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1317_131739


namespace NUMINAMATH_CALUDE_two_moments_theorem_l1317_131765

/-- Represents a reader visiting the library -/
structure Reader where
  id : Nat

/-- Represents a moment in time -/
structure Moment where
  time : Nat

/-- Represents the state of a reader being in the library at a given moment -/
def ReaderPresent (r : Reader) (m : Moment) : Prop := sorry

/-- The library visit condition: each reader visits only once -/
axiom visit_once (r : Reader) :
  ∃! m : Moment, ReaderPresent r m

/-- The meeting condition: among any three readers, two meet each other -/
axiom meet_condition (r1 r2 r3 : Reader) :
  ∃ m : Moment, (ReaderPresent r1 m ∧ ReaderPresent r2 m) ∨
                (ReaderPresent r1 m ∧ ReaderPresent r3 m) ∨
                (ReaderPresent r2 m ∧ ReaderPresent r3 m)

/-- The main theorem: there exist two moments such that every reader is present at least at one of them -/
theorem two_moments_theorem (readers : Set Reader) :
  ∃ m1 m2 : Moment, ∀ r ∈ readers, ReaderPresent r m1 ∨ ReaderPresent r m2 :=
sorry

end NUMINAMATH_CALUDE_two_moments_theorem_l1317_131765


namespace NUMINAMATH_CALUDE_odd_coefficient_probability_l1317_131798

/-- The number of terms in the expansion of (1+x)^11 -/
def n : ℕ := 12

/-- The number of terms with odd coefficients in the expansion of (1+x)^11 -/
def k : ℕ := 8

/-- The probability of selecting a term with an odd coefficient -/
def p : ℚ := k / n

theorem odd_coefficient_probability : p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_odd_coefficient_probability_l1317_131798


namespace NUMINAMATH_CALUDE_parabola_reflection_sum_l1317_131709

theorem parabola_reflection_sum (a b c : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x + c + 3
  let g := fun x : ℝ => -a * x^2 - b * x - c - 3
  ∀ x, f x + g x = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_reflection_sum_l1317_131709


namespace NUMINAMATH_CALUDE_club_distribution_theorem_l1317_131715

-- Define the type for inhabitants
variable {Inhabitant : Type}

-- Define the type for clubs as sets of inhabitants
variable {Club : Type}

-- Define the property that every two clubs have a common member
def have_common_member (clubs : Set Club) (members : Club → Set Inhabitant) : Prop :=
  ∀ c1 c2 : Club, c1 ∈ clubs → c2 ∈ clubs → c1 ≠ c2 → ∃ i : Inhabitant, i ∈ members c1 ∩ members c2

-- Define the assignment of compasses and rulers
def valid_assignment (clubs : Set Club) (members : Club → Set Inhabitant) 
  (has_compass : Inhabitant → Prop) (has_ruler : Inhabitant → Prop) : Prop :=
  (∀ c : Club, c ∈ clubs → 
    (∃ i : Inhabitant, i ∈ members c ∧ has_compass i) ∧ 
    (∃ i : Inhabitant, i ∈ members c ∧ has_ruler i)) ∧
  (∃! i : Inhabitant, has_compass i ∧ has_ruler i)

-- The main theorem
theorem club_distribution_theorem 
  (clubs : Set Club) (members : Club → Set Inhabitant) 
  (h : have_common_member clubs members) :
  ∃ (has_compass : Inhabitant → Prop) (has_ruler : Inhabitant → Prop),
    valid_assignment clubs members has_compass has_ruler :=
sorry

end NUMINAMATH_CALUDE_club_distribution_theorem_l1317_131715


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l1317_131742

/-- The coordinates of the foci of an ellipse -/
def foci_coordinates (m n : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x = Real.sqrt (n - m) ∧ y = 0 ∨ x = -Real.sqrt (n - m) ∧ y = 0}

/-- Theorem: The coordinates of the foci of the ellipse x²/m + y²/n = -1 where m < n < 0 -/
theorem ellipse_foci_coordinates {m n : ℝ} (hm : m < 0) (hn : n < 0) (hmn : m < n) :
  foci_coordinates m n = {(Real.sqrt (n - m), 0), (-Real.sqrt (n - m), 0)} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l1317_131742


namespace NUMINAMATH_CALUDE_product_of_special_n_values_l1317_131773

theorem product_of_special_n_values : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ p : ℕ, Nat.Prime p ∧ n^2 - 40*n + 399 = p) ∧ 
  (∀ n : ℕ, (∃ p : ℕ, Nat.Prime p ∧ n^2 - 40*n + 399 = p) → n ∈ S) ∧
  S.card > 0 ∧
  (S.prod id = 396) := by
sorry

end NUMINAMATH_CALUDE_product_of_special_n_values_l1317_131773


namespace NUMINAMATH_CALUDE_length_AF_is_5_l1317_131756

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the condition for the circle intersecting y-axis at only one point
def circle_intersects_y_axis_once (A : ℝ × ℝ) : Prop :=
  let (x, y) := A
  x - 2*y + 4 = 0

-- Main theorem
theorem length_AF_is_5 (A : ℝ × ℝ) :
  let (x, y) := A
  parabola x y →
  circle_intersects_y_axis_once A →
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_length_AF_is_5_l1317_131756


namespace NUMINAMATH_CALUDE_cos_squared_difference_equals_sqrt3_over_2_l1317_131789

theorem cos_squared_difference_equals_sqrt3_over_2 :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_difference_equals_sqrt3_over_2_l1317_131789


namespace NUMINAMATH_CALUDE_area_of_combined_rectangle_l1317_131740

/-- The area of a rectangle formed by three identical rectangles --/
theorem area_of_combined_rectangle (short_side : ℝ) (h : short_side = 7) : 
  let long_side : ℝ := 2 * short_side
  let width : ℝ := 2 * short_side
  let length : ℝ := long_side
  width * length = 196 := by sorry

end NUMINAMATH_CALUDE_area_of_combined_rectangle_l1317_131740


namespace NUMINAMATH_CALUDE_equation_solution_l1317_131717

theorem equation_solution : 
  {x : ℝ | x^3 + x = 1/x^3 + 1/x} = {-1, 1} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1317_131717


namespace NUMINAMATH_CALUDE_sum_coordinates_reflection_over_x_axis_l1317_131702

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflect a point over the x-axis -/
def reflectOverXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Sum of all coordinate values of two points -/
def sumCoordinates (p1 p2 : Point2D) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

/-- Theorem: The sum of coordinates of a point (5, y) and its reflection over x-axis is 10 -/
theorem sum_coordinates_reflection_over_x_axis (y : ℝ) :
  let c : Point2D := { x := 5, y := y }
  let d : Point2D := reflectOverXAxis c
  sumCoordinates c d = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_reflection_over_x_axis_l1317_131702


namespace NUMINAMATH_CALUDE_inequality_proof_l1317_131774

theorem inequality_proof (x y z : ℝ) (h : x * y * z = 1) :
  x^2 + y^2 + z^2 ≥ 1/x + 1/y + 1/z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1317_131774


namespace NUMINAMATH_CALUDE_lunch_combinations_l1317_131722

theorem lunch_combinations (first_course main_course dessert : ℕ) :
  first_course = 4 → main_course = 5 → dessert = 3 →
  first_course * main_course * dessert = 60 := by
sorry

end NUMINAMATH_CALUDE_lunch_combinations_l1317_131722


namespace NUMINAMATH_CALUDE_donna_weekly_episodes_l1317_131795

/-- The number of episodes Donna can watch on a weekday -/
def weekday_episodes : ℕ := 8

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The multiplier for weekend episode watching compared to weekdays -/
def weekend_multiplier : ℕ := 3

/-- The total number of episodes Donna can watch in a week -/
def total_episodes : ℕ := weekday_episodes * weekdays + weekday_episodes * weekend_multiplier * weekend_days

theorem donna_weekly_episodes :
  total_episodes = 88 :=
sorry

end NUMINAMATH_CALUDE_donna_weekly_episodes_l1317_131795


namespace NUMINAMATH_CALUDE_fraction_equality_l1317_131783

theorem fraction_equality (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a / b + b / a = 4) :
  (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1317_131783


namespace NUMINAMATH_CALUDE_unique_intersection_l1317_131726

-- Define the three lines
def line1 (x y : ℝ) : Prop := 3 * x - 9 * y + 18 = 0
def line2 (x y : ℝ) : Prop := 6 * x - 18 * y - 36 = 0
def line3 (x : ℝ) : Prop := x - 3 = 0

-- Define what it means for a point to be on all three lines
def onAllLines (x y : ℝ) : Prop :=
  line1 x y ∧ line2 x y ∧ line3 x

-- Theorem statement
theorem unique_intersection :
  ∃! p : ℝ × ℝ, onAllLines p.1 p.2 ∧ p = (3, 3) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l1317_131726


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l1317_131797

/-- A polynomial of degree 5 -/
def P (a b c d e : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The statement that P has five distinct x-intercepts -/
def has_five_distinct_roots (a b c d e : ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ : ℝ), (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧
                           r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧
                           r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧
                           r₄ ≠ r₅) ∧
                          (∀ x : ℝ, P a b c d e x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅)

theorem coefficient_d_nonzero (a b c d e : ℝ) 
  (h1 : has_five_distinct_roots a b c d e)
  (h2 : P a b c d e 0 = 0) : -- One root is at (0,0)
  d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_d_nonzero_l1317_131797


namespace NUMINAMATH_CALUDE_dad_borrowed_75_nickels_l1317_131752

/-- The number of nickels borrowed by Mike's dad -/
def nickels_borrowed (initial_nickels current_nickels : ℕ) : ℕ :=
  initial_nickels - current_nickels

/-- Proof that Mike's dad borrowed 75 nickels -/
theorem dad_borrowed_75_nickels (initial_nickels current_nickels : ℕ) 
  (h1 : initial_nickels = 87)
  (h2 : current_nickels = 12) :
  nickels_borrowed initial_nickels current_nickels = 75 := by
  sorry

#eval nickels_borrowed 87 12

end NUMINAMATH_CALUDE_dad_borrowed_75_nickels_l1317_131752


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1317_131736

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1317_131736


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l1317_131761

/-- A hexagon that shares three sides with a rectangle and has the other three sides
    each equal to one of the rectangle's dimensions. -/
structure SpecialHexagon where
  rect_side1 : ℕ
  rect_side2 : ℕ

/-- The perimeter of the special hexagon. -/
def perimeter (h : SpecialHexagon) : ℕ :=
  2 * h.rect_side1 + 2 * h.rect_side2 + h.rect_side1 + h.rect_side2

/-- Theorem stating that the perimeter of the special hexagon with sides 7 and 5 is 36. -/
theorem special_hexagon_perimeter :
  ∃ (h : SpecialHexagon), h.rect_side1 = 7 ∧ h.rect_side2 = 5 ∧ perimeter h = 36 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l1317_131761


namespace NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_8000_l1317_131737

theorem count_primes_with_squares_between_5000_and_8000 :
  (Finset.filter (fun p => 5000 < p^2 ∧ p^2 < 8000) (Finset.filter Nat.Prime (Finset.range 90))).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_8000_l1317_131737


namespace NUMINAMATH_CALUDE_glendas_wedding_fish_l1317_131743

/-- Given a wedding reception setup with tables and fish, calculate the total number of fish -/
def total_fish (total_tables : Nat) (regular_fish : Nat) (special_fish : Nat) : Nat :=
  (total_tables - 1) * regular_fish + special_fish

/-- Theorem: The total number of fish at Glenda's wedding reception is 65 -/
theorem glendas_wedding_fish : total_fish 32 2 3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_glendas_wedding_fish_l1317_131743


namespace NUMINAMATH_CALUDE_correct_lineup_choices_l1317_131741

/-- Represents the number of players in a basketball team --/
def team_size : ℕ := 5

/-- Represents the total number of players in the rotation --/
def rotation_size : ℕ := 8

/-- Represents the number of centers in the rotation --/
def num_centers : ℕ := 2

/-- Represents the number of point guards in the rotation --/
def num_point_guards : ℕ := 2

/-- Calculates the number of ways to choose a lineup --/
def lineup_choices : ℕ := sorry

theorem correct_lineup_choices : lineup_choices = 28 := by sorry

end NUMINAMATH_CALUDE_correct_lineup_choices_l1317_131741


namespace NUMINAMATH_CALUDE_girls_adjacent_probability_l1317_131796

/-- The number of ways to arrange two boys and two girls in a line -/
def totalArrangements : ℕ := 24

/-- The number of ways to arrange two boys and two girls in a line with the girls adjacent -/
def favorableArrangements : ℕ := 12

/-- The probability of two girls being adjacent when two boys and two girls are randomly arranged in a line -/
def probabilityGirlsAdjacent : ℚ := favorableArrangements / totalArrangements

theorem girls_adjacent_probability :
  probabilityGirlsAdjacent = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_girls_adjacent_probability_l1317_131796


namespace NUMINAMATH_CALUDE_video_votes_l1317_131712

theorem video_votes (total_votes : ℕ) (score : ℤ) (like_percentage : ℚ) : 
  score = 140 ∧ 
  like_percentage = 70 / 100 ∧
  (like_percentage * total_votes : ℚ).num * 1 + 
    ((1 - like_percentage) * total_votes : ℚ).num * (-1) = score ∧
  (like_percentage * total_votes : ℚ).den = 1 ∧
  ((1 - like_percentage) * total_votes : ℚ).den = 1
  → total_votes = 350 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l1317_131712


namespace NUMINAMATH_CALUDE_walkers_meet_at_start_l1317_131716

/-- Represents a point on the rectangular loop -/
structure Point :=
  (position : ℕ)

/-- Represents a person walking on the loop -/
structure Walker :=
  (speed : ℕ)
  (direction : Bool) -- True for clockwise, False for counterclockwise

/-- The total number of blocks in the rectangular loop -/
def total_blocks : ℕ := 24

/-- Calculates the meeting point of two walkers -/
def meeting_point (w1 w2 : Walker) (start : Point) : Point :=
  sorry

/-- Theorem stating that the walkers meet at their starting point -/
theorem walkers_meet_at_start (start : Point) :
  let hector := Walker.mk 1 true
  let jane := Walker.mk 3 false
  (meeting_point hector jane start).position = start.position :=
sorry

end NUMINAMATH_CALUDE_walkers_meet_at_start_l1317_131716


namespace NUMINAMATH_CALUDE_expression_evaluation_l1317_131700

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  let z : ℝ := 1
  let w : ℝ := 3
  (x^2 * y^2 * z^2) - (x^2 * y * z^2) + (y / w) * Real.sin (x * z) = -(1/3) * Real.sin 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1317_131700


namespace NUMINAMATH_CALUDE_max_angle_B_in_arithmetic_sequence_triangle_l1317_131768

theorem max_angle_B_in_arithmetic_sequence_triangle :
  ∀ (a b c : ℝ) (A B C : ℝ),
    0 < a ∧ 0 < b ∧ 0 < c →
    0 < A ∧ 0 < B ∧ 0 < C →
    A + B + C = π →
    b^2 = a * c →  -- arithmetic sequence condition
    B ≤ π / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_angle_B_in_arithmetic_sequence_triangle_l1317_131768


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1317_131723

/-- Given a line L1 with equation 6x - 3y = 9 and a point P (1, -2),
    prove that the line L2 with equation y = 2x - 4 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  (6 * x - 3 * y = 9) →  -- Equation of L1
  (y = 2 * x - 4) →      -- Equation of L2
  (2 = (6 : ℝ) / 3) →    -- Slopes are equal (parallel condition)
  (2 * 1 - 4 = -2) →     -- L2 passes through (1, -2)
  ∃ (m b : ℝ), y = m * x + b ∧ m = 2 ∧ b = -4 := by
sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l1317_131723


namespace NUMINAMATH_CALUDE_savings_equality_l1317_131794

theorem savings_equality (your_initial : ℕ) (friend_initial : ℕ) (friend_weekly : ℕ) (weeks : ℕ) 
  (h1 : your_initial = 160)
  (h2 : friend_initial = 210)
  (h3 : friend_weekly = 5)
  (h4 : weeks = 25) :
  ∃ your_weekly : ℕ, 
    your_initial + weeks * your_weekly = friend_initial + weeks * friend_weekly ∧ 
    your_weekly = 7 := by
  sorry

end NUMINAMATH_CALUDE_savings_equality_l1317_131794


namespace NUMINAMATH_CALUDE_parabola_translation_l1317_131784

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * dx + p.b
    c := p.a * dx^2 - p.b * dx + p.c - dy }

theorem parabola_translation :
  let original := Parabola.mk 1 0 1  -- y = x^2 + 1
  let translated := translate original 3 (-2)  -- 3 units right, 2 units down
  translated = Parabola.mk 1 (-6) (-1)  -- y = (x - 3)^2 - 1
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1317_131784


namespace NUMINAMATH_CALUDE_interior_angle_sum_difference_l1317_131747

/-- The sum of interior angles of a convex n-sided polygon in degrees -/
def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

theorem interior_angle_sum_difference (n : ℕ) (h : n ≥ 3) :
  interior_angle_sum (n + 1) - interior_angle_sum n = 180 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_difference_l1317_131747


namespace NUMINAMATH_CALUDE_right_triangle_area_l1317_131749

/-- The area of a right triangle with hypotenuse 13 and one side 5 is 30 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) :
  (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1317_131749


namespace NUMINAMATH_CALUDE_conditional_without_else_is_valid_l1317_131705

/-- Represents a conditional statement --/
inductive ConditionalStatement
  | ifThen (condition : Prop) (thenClause : Prop)
  | ifThenElse (condition : Prop) (thenClause : Prop) (elseClause : Prop)

/-- A conditional statement is valid if it has at least an IF-THEN structure --/
def isValidConditionalStatement : ConditionalStatement → Prop
  | ConditionalStatement.ifThen _ _ => True
  | ConditionalStatement.ifThenElse _ _ _ => True

theorem conditional_without_else_is_valid (condition thenClause : Prop) :
  isValidConditionalStatement (ConditionalStatement.ifThen condition thenClause) := by
  sorry

end NUMINAMATH_CALUDE_conditional_without_else_is_valid_l1317_131705


namespace NUMINAMATH_CALUDE_percent_equality_l1317_131782

theorem percent_equality : (25 : ℚ) / 100 * 2004 = (50 : ℚ) / 100 * 1002 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l1317_131782


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1317_131788

def arithmeticSequenceSum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_ratio :
  let numerator := arithmeticSequenceSum 4 4 52
  let denominator := arithmeticSequenceSum 6 6 78
  numerator / denominator = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1317_131788


namespace NUMINAMATH_CALUDE_cubic_function_property_l1317_131753

/-- Given a cubic function f(x) = ax^3 + bx + 1 where f(-2) = 2, prove that f(2) = 0 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  f (-2) = 2 → f 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1317_131753


namespace NUMINAMATH_CALUDE_quadratic_one_zero_point_l1317_131720

/-- A quadratic function with coefficient m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2*x + 3

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := 4 - 12*m

theorem quadratic_one_zero_point (m : ℝ) : 
  (∃! x, f m x = 0) ↔ m = 0 ∨ m = 1/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_zero_point_l1317_131720


namespace NUMINAMATH_CALUDE_prob_two_high_temp_is_half_l1317_131778

/-- Represents a 3-digit number where each digit is either 0-5 or 6-9 -/
def ThreeDayPeriod := Fin 1000

/-- The probability of a digit being 0-5 (representing a high temperature warning) -/
def p_high_temp : ℚ := 3/5

/-- The number of random samples generated -/
def num_samples : ℕ := 20

/-- Counts the number of digits in a ThreeDayPeriod that are 0-5 -/
def count_high_temp (n : ThreeDayPeriod) : ℕ := sorry

/-- The event of exactly 2 high temperature warnings in a 3-day period -/
def two_high_temp (n : ThreeDayPeriod) : Prop := count_high_temp n = 2

/-- The probability of the event two_high_temp -/
def prob_two_high_temp : ℚ := sorry

theorem prob_two_high_temp_is_half : prob_two_high_temp = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_two_high_temp_is_half_l1317_131778


namespace NUMINAMATH_CALUDE_line_l_equation_l1317_131738

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨2, 0⟩
def l₁ : Line := ⟨2, 1, -3⟩
def l₂ : Line := ⟨3, -1, 6⟩

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The equation of line l -/
inductive LineL
  | eq1 : LineL  -- 21x + 13y - 42 = 0
  | eq2 : LineL  -- x + y - 2 = 0
  | eq3 : LineL  -- 3x + 4y - 6 = 0

theorem line_l_equation : ∃ (l : LineL), 
  (∀ (A B : Point), pointOnLine A l₁ ∧ pointOnLine B l₂ ∧ 
    pointOnLine M ⟨21, 13, -42⟩ ∧ pointOnLine A ⟨21, 13, -42⟩ ∧ pointOnLine B ⟨21, 13, -42⟩) ∨
  (∀ (A B : Point), pointOnLine A l₁ ∧ pointOnLine B l₂ ∧ 
    pointOnLine M ⟨1, 1, -2⟩ ∧ pointOnLine A ⟨1, 1, -2⟩ ∧ pointOnLine B ⟨1, 1, -2⟩) ∨
  (∀ (A B : Point), pointOnLine A l₁ ∧ pointOnLine B l₂ ∧ 
    pointOnLine M ⟨3, 4, -6⟩ ∧ pointOnLine A ⟨3, 4, -6⟩ ∧ pointOnLine B ⟨3, 4, -6⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_line_l_equation_l1317_131738


namespace NUMINAMATH_CALUDE_pizza_theorem_l1317_131781

def pizza_problem (total_pizzas : ℕ) (first_day_fraction : ℚ) (subsequent_day_fraction : ℚ) (daily_limit_fraction : ℚ) : Prop :=
  ∀ (monday tuesday wednesday thursday friday : ℕ),
    -- Total pizzas condition
    total_pizzas = 1000 →
    -- First day condition
    monday = (total_pizzas : ℚ) * first_day_fraction →
    -- Subsequent days conditions
    tuesday = min ((total_pizzas - monday : ℚ) * subsequent_day_fraction) (monday * daily_limit_fraction) →
    wednesday = min ((total_pizzas - monday - tuesday : ℚ) * subsequent_day_fraction) (tuesday * daily_limit_fraction) →
    thursday = min ((total_pizzas - monday - tuesday - wednesday : ℚ) * subsequent_day_fraction) (wednesday * daily_limit_fraction) →
    friday = min ((total_pizzas - monday - tuesday - wednesday - thursday : ℚ) * subsequent_day_fraction) (thursday * daily_limit_fraction) →
    -- Conclusion
    friday ≤ 2

theorem pizza_theorem : pizza_problem 1000 (7/10) (4/5) (9/10) :=
sorry

end NUMINAMATH_CALUDE_pizza_theorem_l1317_131781


namespace NUMINAMATH_CALUDE_reciprocal_sum_fractions_l1317_131792

theorem reciprocal_sum_fractions : 
  (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fractions_l1317_131792


namespace NUMINAMATH_CALUDE_book_selection_theorem_l1317_131780

def select_books (total : ℕ) (to_select : ℕ) (must_include : ℕ) : ℕ :=
  Nat.choose (total - must_include) (to_select - must_include)

theorem book_selection_theorem :
  select_books 8 5 1 = 35 :=
by sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l1317_131780


namespace NUMINAMATH_CALUDE_sphere_volume_from_cube_l1317_131735

/-- Given a cube with edge length 3 and all its vertices on the same spherical surface,
    the volume of that sphere is 27√3π/2 -/
theorem sphere_volume_from_cube (edge_length : ℝ) (radius : ℝ) :
  edge_length = 3 →
  radius = (3 * Real.sqrt 3) / 2 →
  (4 / 3) * Real.pi * radius^3 = (27 * Real.sqrt 3 * Real.pi) / 2 := by
  sorry


end NUMINAMATH_CALUDE_sphere_volume_from_cube_l1317_131735


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotone_condition_equivalent_to_range_l1317_131777

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ (k b : ℝ), ∀ x, k * x + b = f a x + (f a 1 - f a x) * (x - 1) / (1 - x) ∧ k = 0 ∧ b = -2 :=
sorry

theorem monotone_condition_equivalent_to_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ + 2*x₁ < f a x₂ + 2*x₂) ↔ 0 ≤ a ∧ a ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotone_condition_equivalent_to_range_l1317_131777


namespace NUMINAMATH_CALUDE_second_guide_children_l1317_131708

/-- Given information about zoo guides and children -/
structure ZooTour where
  total_children : ℕ
  first_guide_children : ℕ

/-- Theorem: The second guide spoke to 25 children -/
theorem second_guide_children (tour : ZooTour) 
  (h1 : tour.total_children = 44)
  (h2 : tour.first_guide_children = 19) :
  tour.total_children - tour.first_guide_children = 25 := by
  sorry

#eval 44 - 19  -- Expected output: 25

end NUMINAMATH_CALUDE_second_guide_children_l1317_131708


namespace NUMINAMATH_CALUDE_brownies_degrees_in_pie_chart_l1317_131755

/-- Calculates the degrees for brownies in a pie chart given the class composition -/
theorem brownies_degrees_in_pie_chart 
  (total_students : ℕ) 
  (cookie_lovers : ℕ) 
  (muffin_lovers : ℕ) 
  (cupcake_lovers : ℕ) 
  (h1 : total_students = 45)
  (h2 : cookie_lovers = 15)
  (h3 : muffin_lovers = 9)
  (h4 : cupcake_lovers = 7)
  (h5 : (total_students - (cookie_lovers + muffin_lovers + cupcake_lovers)) % 2 = 0) :
  (((total_students - (cookie_lovers + muffin_lovers + cupcake_lovers)) / 2) : ℚ) / total_students * 360 = 56 := by
sorry

end NUMINAMATH_CALUDE_brownies_degrees_in_pie_chart_l1317_131755


namespace NUMINAMATH_CALUDE_u_eq_complement_a_union_b_l1317_131728

/-- The universal set U -/
def U : Finset Nat := {1, 2, 3, 4, 5, 7}

/-- Set A -/
def A : Finset Nat := {4, 7}

/-- Set B -/
def B : Finset Nat := {1, 3, 4, 7}

/-- Theorem stating that U is equal to the union of the complement of A in U and B -/
theorem u_eq_complement_a_union_b : U = (U \ A) ∪ B := by sorry

end NUMINAMATH_CALUDE_u_eq_complement_a_union_b_l1317_131728


namespace NUMINAMATH_CALUDE_smallest_inscribed_cube_volume_l1317_131775

theorem smallest_inscribed_cube_volume (outer_cube_edge : ℝ) : 
  outer_cube_edge = 16 →
  ∃ (largest_sphere_radius smallest_cube_edge : ℝ),
    largest_sphere_radius = outer_cube_edge / 2 ∧
    smallest_cube_edge = 16 / Real.sqrt 3 ∧
    smallest_cube_edge ^ 3 = 456 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_inscribed_cube_volume_l1317_131775


namespace NUMINAMATH_CALUDE_factoring_expression_l1317_131786

theorem factoring_expression (x : ℝ) : x * (x + 4) + 2 * (x + 4) + (x + 4) = (x + 3) * (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l1317_131786


namespace NUMINAMATH_CALUDE_max_points_at_distance_l1317_131729

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point : Type := ℝ × ℝ

-- Function to check if a point is outside a circle
def isOutside (p : Point) (c : Circle) : Prop :=
  let (px, py) := p
  let (cx, cy) := c.center
  (px - cx)^2 + (py - cy)^2 > c.radius^2

-- Function to count points on circle at fixed distance from a point
def countPointsAtDistance (c : Circle) (p : Point) (d : ℝ) : ℕ :=
  sorry

-- Theorem statement
theorem max_points_at_distance (c : Circle) (p : Point) (d : ℝ) 
  (h : isOutside p c) : 
  countPointsAtDistance c p d ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_points_at_distance_l1317_131729


namespace NUMINAMATH_CALUDE_prize_money_calculation_l1317_131766

def total_amount : ℕ := 300
def paintings_sold : ℕ := 3
def price_per_painting : ℕ := 50

theorem prize_money_calculation :
  total_amount - (paintings_sold * price_per_painting) = 150 := by
  sorry

end NUMINAMATH_CALUDE_prize_money_calculation_l1317_131766


namespace NUMINAMATH_CALUDE_man_daily_wage_l1317_131710

/-- The daily wage of a man -/
def M : ℝ := sorry

/-- The daily wage of a woman -/
def W : ℝ := sorry

/-- The total wages of 24 men and 16 women per day -/
def total_wages : ℝ := 11600

/-- The number of men -/
def num_men : ℕ := 24

/-- The number of women -/
def num_women : ℕ := 16

/-- The wages of 24 men and 16 women amount to Rs. 11600 per day -/
axiom wage_equation : num_men * M + num_women * W = total_wages

/-- Half the number of men and 37 women earn the same amount per day -/
axiom half_men_equation : (num_men / 2) * M + 37 * W = total_wages

theorem man_daily_wage : M = 350 := by sorry

end NUMINAMATH_CALUDE_man_daily_wage_l1317_131710


namespace NUMINAMATH_CALUDE_power_of_seven_mod_eight_l1317_131793

theorem power_of_seven_mod_eight : 7^123 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_eight_l1317_131793


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1317_131760

theorem imaginary_part_of_complex_fraction : Complex.im (4 * I / (1 - I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1317_131760


namespace NUMINAMATH_CALUDE_solve_equation_l1317_131757

theorem solve_equation (x : ℝ) (n : ℝ) (h1 : 5 / (n + 1 / x) = 1) (h2 : x = 1) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1317_131757


namespace NUMINAMATH_CALUDE_primer_cost_calculation_l1317_131791

/-- Represents the cost of primer per gallon before discount -/
def primer_cost : ℝ := 30

/-- Number of rooms to be painted and primed -/
def num_rooms : ℕ := 5

/-- Cost of paint per gallon -/
def paint_cost : ℝ := 25

/-- Discount rate on primer -/
def primer_discount : ℝ := 0.2

/-- Total amount spent on paint and primer -/
def total_spent : ℝ := 245

theorem primer_cost_calculation : 
  (num_rooms : ℝ) * paint_cost + 
  (num_rooms : ℝ) * primer_cost * (1 - primer_discount) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_primer_cost_calculation_l1317_131791


namespace NUMINAMATH_CALUDE_sequences_properties_l1317_131730

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def a (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- Geometric sequence with first term 1 and common ratio 2 -/
def b (n : ℕ) : ℕ := 2^(n - 1)

/-- Product sequence of a and b -/
def c (n : ℕ) : ℕ := a n * b n

/-- Sum of first n terms of arithmetic sequence a -/
def S (n : ℕ) : ℕ := n * (a 1 + a n) / 2

/-- Sum of first n terms of product sequence c -/
def T (n : ℕ) : ℕ := (2 * n - 1) * 2^n + 1

theorem sequences_properties :
  (a 1 = 3) ∧
  (b 1 = 1) ∧
  (b 2 + S 2 = 10) ∧
  (a 5 - 2 * b 2 = a 3) ∧
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, b n = 2^(n - 1)) ∧
  (∀ n : ℕ, T n = (2 * n - 1) * 2^n + 1) := by
  sorry

#check sequences_properties

end NUMINAMATH_CALUDE_sequences_properties_l1317_131730


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l1317_131725

theorem largest_lcm_with_18 : 
  (Finset.image (fun x => Nat.lcm 18 x) {3, 6, 9, 12, 15, 18}).max = some 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l1317_131725


namespace NUMINAMATH_CALUDE_awards_sum_is_80_l1317_131759

/-- The number of awards won by Scott -/
def scott_awards : ℕ := 4

/-- The number of awards won by Jessie -/
def jessie_awards : ℕ := 3 * scott_awards

/-- The number of awards won by the rival athlete -/
def rival_awards : ℕ := 2 * jessie_awards

/-- The number of awards won by Brad -/
def brad_awards : ℕ := (5 * rival_awards) / 3

/-- The total number of awards won by all four athletes -/
def total_awards : ℕ := scott_awards + jessie_awards + rival_awards + brad_awards

theorem awards_sum_is_80 : total_awards = 80 := by
  sorry

end NUMINAMATH_CALUDE_awards_sum_is_80_l1317_131759


namespace NUMINAMATH_CALUDE_line_through_points_l1317_131762

/-- Given a line x = 3y + 5 passing through points (m, n) and (m + 2, n + q), prove that q = 2/3 -/
theorem line_through_points (m n : ℝ) : 
  (∃ q : ℝ, m = 3 * n + 5 ∧ m + 2 = 3 * (n + q) + 5) → 
  (∃ q : ℝ, q = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l1317_131762


namespace NUMINAMATH_CALUDE_circle_tangent_line_l1317_131776

/-- Given a circle x^2 + y^2 = r^2 and a point P(x₀, y₀) on the circle,
    the tangent line at P has the equation x₀x + y₀y = r^2 -/
theorem circle_tangent_line (r x₀ y₀ : ℝ) (h : x₀^2 + y₀^2 = r^2) :
  ∀ x y : ℝ, (x - x₀)^2 + (y - y₀)^2 = 0 ↔ x₀*x + y₀*y = r^2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_line_l1317_131776


namespace NUMINAMATH_CALUDE_xiaofang_english_score_l1317_131718

/-- Represents the scores of four subjects -/
structure Scores where
  chinese : ℝ
  math : ℝ
  english : ℝ
  science : ℝ

/-- The average score of four subjects is 88 -/
def avg_four (s : Scores) : Prop :=
  (s.chinese + s.math + s.english + s.science) / 4 = 88

/-- The average score of the first two subjects is 93 -/
def avg_first_two (s : Scores) : Prop :=
  (s.chinese + s.math) / 2 = 93

/-- The average score of the last three subjects is 87 -/
def avg_last_three (s : Scores) : Prop :=
  (s.math + s.english + s.science) / 3 = 87

/-- Xiaofang's English test score is 95 -/
theorem xiaofang_english_score (s : Scores) 
  (h1 : avg_four s) (h2 : avg_first_two s) (h3 : avg_last_three s) : 
  s.english = 95 := by
  sorry

end NUMINAMATH_CALUDE_xiaofang_english_score_l1317_131718


namespace NUMINAMATH_CALUDE_circumradius_of_special_triangle_l1317_131772

/-- The radius of the circumcircle of a triangle with sides 8, 15, and 17 is 17/2 -/
theorem circumradius_of_special_triangle :
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := 17
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (area * 4) / (a * b * c) = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_of_special_triangle_l1317_131772
