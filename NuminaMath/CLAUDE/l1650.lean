import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l1650_165043

theorem simplify_expression (x y : ℝ) : 8*y + 15 - 3*y + 20 + 2*x = 5*y + 2*x + 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1650_165043


namespace NUMINAMATH_CALUDE_constant_term_value_l1650_165025

/-- The binomial expansion of (x - 2/x)^8 has its maximum coefficient in the 5th term -/
def max_coeff_5th_term (x : ℝ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ 8 → Nat.choose 8 4 ≥ Nat.choose 8 k

/-- The constant term in the binomial expansion of (x - 2/x)^8 -/
def constant_term (x : ℝ) : ℤ :=
  Nat.choose 8 4 * (-2)^4

theorem constant_term_value (x : ℝ) :
  max_coeff_5th_term x → constant_term x = 1120 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l1650_165025


namespace NUMINAMATH_CALUDE_range_of_p_l1650_165055

-- Define set A
def A (p : ℝ) : Set ℝ := {x : ℝ | x^2 + (p+2)*x + 1 = 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x > 0}

-- Theorem statement
theorem range_of_p (p : ℝ) : (A p ∩ B = ∅) → p > -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l1650_165055


namespace NUMINAMATH_CALUDE_f_at_zero_equals_four_l1650_165000

/-- Given a function f(x) = a * sin(x) + b * (x^(1/3)) + 4 where a and b are real numbers,
    prove that f(0) = 4 -/
theorem f_at_zero_equals_four (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x + b * Real.rpow x (1/3) + 4
  f 0 = 4 := by sorry

end NUMINAMATH_CALUDE_f_at_zero_equals_four_l1650_165000


namespace NUMINAMATH_CALUDE_exactly_six_expressions_l1650_165042

/-- Represents an expression using three identical digits --/
inductive ThreeDigitExpr (d : ℕ)
| add : ThreeDigitExpr d
| sub : ThreeDigitExpr d
| mul : ThreeDigitExpr d
| div : ThreeDigitExpr d
| exp : ThreeDigitExpr d
| sqrt : ThreeDigitExpr d
| floor : ThreeDigitExpr d
| fact : ThreeDigitExpr d

/-- Evaluates a ThreeDigitExpr to a real number --/
def eval {d : ℕ} : ThreeDigitExpr d → ℝ
| ThreeDigitExpr.add => sorry
| ThreeDigitExpr.sub => sorry
| ThreeDigitExpr.mul => sorry
| ThreeDigitExpr.div => sorry
| ThreeDigitExpr.exp => sorry
| ThreeDigitExpr.sqrt => sorry
| ThreeDigitExpr.floor => sorry
| ThreeDigitExpr.fact => sorry

/-- Predicate for valid expressions that evaluate to 24 --/
def isValid (d : ℕ) (e : ThreeDigitExpr d) : Prop :=
  d ≠ 8 ∧ eval e = 24

/-- The main theorem stating there are exactly 6 valid expressions --/
theorem exactly_six_expressions :
  ∃ (exprs : Finset (Σ (d : ℕ), ThreeDigitExpr d)),
    exprs.card = 6 ∧
    (∀ (d : ℕ) (e : ThreeDigitExpr d), isValid d e ↔ (⟨d, e⟩ : Σ (d : ℕ), ThreeDigitExpr d) ∈ exprs) :=
sorry

end NUMINAMATH_CALUDE_exactly_six_expressions_l1650_165042


namespace NUMINAMATH_CALUDE_sweeties_leftover_l1650_165064

theorem sweeties_leftover (m : ℕ) (h : m % 12 = 11) : (4 * m) % 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sweeties_leftover_l1650_165064


namespace NUMINAMATH_CALUDE_parallelogram_area_48_36_l1650_165007

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 48 cm and height 36 cm is 1728 square centimeters -/
theorem parallelogram_area_48_36 : parallelogram_area 48 36 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_48_36_l1650_165007


namespace NUMINAMATH_CALUDE_cristinas_pace_cristinas_pace_is_five_l1650_165067

/-- Cristina's pace in a race with Nicky -/
theorem cristinas_pace (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let total_distance := head_start + nickys_pace * catch_up_time
  total_distance / catch_up_time

/-- Prove that Cristina's pace is 5 meters per second -/
theorem cristinas_pace_is_five :
  cristinas_pace 48 3 24 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cristinas_pace_cristinas_pace_is_five_l1650_165067


namespace NUMINAMATH_CALUDE_chess_tournament_schedule_ways_l1650_165057

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  /-- Number of players per school -/
  players_per_school : Nat
  /-- Number of games each player plays against each opponent -/
  games_per_opponent : Nat
  /-- Number of rounds in the tournament -/
  num_rounds : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat

/-- Calculate the number of ways to schedule a chess tournament -/
def scheduleWays (t : ChessTournament) : Nat :=
  (t.num_rounds.factorial) + (t.num_rounds.factorial / (2^(t.num_rounds / 2)))

/-- Theorem stating the number of ways to schedule the specific chess tournament -/
theorem chess_tournament_schedule_ways :
  let t : ChessTournament := {
    players_per_school := 4,
    games_per_opponent := 2,
    num_rounds := 8,
    games_per_round := 4
  }
  scheduleWays t = 42840 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_schedule_ways_l1650_165057


namespace NUMINAMATH_CALUDE_nancy_file_deletion_l1650_165022

theorem nancy_file_deletion (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : 
  initial_files = 80 → 
  num_folders = 7 → 
  files_per_folder = 7 → 
  initial_files - (num_folders * files_per_folder) = 31 := by
sorry

end NUMINAMATH_CALUDE_nancy_file_deletion_l1650_165022


namespace NUMINAMATH_CALUDE_tangent_problem_l1650_165085

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (1 + Real.tan α) / (1 - Real.tan α) = 3/22 := by
sorry

end NUMINAMATH_CALUDE_tangent_problem_l1650_165085


namespace NUMINAMATH_CALUDE_range_of_a_l1650_165068

open Set

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, 2 * x₀^2 - 3 * a * x₀ + 9 < 0) ↔ 
  a ∈ (Iio (-2 * Real.sqrt 2) ∪ Ioi (2 * Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1650_165068


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1650_165087

theorem quadratic_form_ratio (j : ℝ) : ∃ (c p q : ℝ),
  (8 * j^2 - 6 * j + 16 = c * (j + p)^2 + q) ∧ (q / p = -119 / 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1650_165087


namespace NUMINAMATH_CALUDE_children_bridge_problem_l1650_165075

/-- The problem of three children crossing a bridge --/
theorem children_bridge_problem (bridge_capacity : ℝ) (kelly_weight : ℝ) :
  bridge_capacity = 100 →
  kelly_weight = 34 →
  ∃ (megan_weight : ℝ) (mike_weight : ℝ),
    kelly_weight = 0.85 * megan_weight ∧
    mike_weight = megan_weight + 5 ∧
    kelly_weight + megan_weight + mike_weight - bridge_capacity = 19 :=
by sorry

end NUMINAMATH_CALUDE_children_bridge_problem_l1650_165075


namespace NUMINAMATH_CALUDE_tournament_cycle_l1650_165072

def TournamentGraph := Fin 12 → Fin 12 → Bool

theorem tournament_cycle (g : TournamentGraph) : 
  (∀ i j : Fin 12, i ≠ j → (g i j ≠ g j i) ∧ (g i j ∨ g j i)) →
  (∀ i : Fin 12, ∃ j : Fin 12, g i j) →
  ∃ a b c : Fin 12, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ g a b ∧ g b c ∧ g c a :=
by sorry

end NUMINAMATH_CALUDE_tournament_cycle_l1650_165072


namespace NUMINAMATH_CALUDE_circle_symmetry_l1650_165093

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 1 = 0

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

/-- Symmetrical circle -/
def symmetrical_circle (x y : ℝ) : Prop :=
  (x + 7/5)^2 + (y - 6/5)^2 = 2

/-- Theorem stating that the symmetrical circle is indeed symmetrical to the given circle
    with respect to the line of symmetry -/
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    given_circle x₁ y₁ →
    symmetrical_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1650_165093


namespace NUMINAMATH_CALUDE_ab_value_l1650_165084

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1650_165084


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_100_l1650_165079

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_prime_divisor_of_sum_100 :
  let sum := sum_of_first_n 100
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ sum ∧ ∀ q < p, Nat.Prime q → ¬(q ∣ sum) ∧ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_100_l1650_165079


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1650_165088

theorem angle_measure_proof (x : Real) : 
  (x + (3 * x + 3) = 90) → x = 21.75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1650_165088


namespace NUMINAMATH_CALUDE_min_value_squared_differences_l1650_165063

theorem min_value_squared_differences (a b c d : ℝ) 
  (h1 : a * b = 3) 
  (h2 : c + 3 * d = 0) : 
  (a - c)^2 + (b - d)^2 ≥ 18/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_differences_l1650_165063


namespace NUMINAMATH_CALUDE_sum_of_roots_l1650_165049

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → a * (a - 4) = 12 → b * (b - 4) = 12 → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1650_165049


namespace NUMINAMATH_CALUDE_inequality_proof_l1650_165036

theorem inequality_proof (n : ℕ) (a : ℝ) (hn : n > 1) (ha : 0 < a ∧ a < 1) :
  1 + a < (1 + a / n)^n ∧ (1 + a / n)^n < (1 + a / (n + 1))^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1650_165036


namespace NUMINAMATH_CALUDE_school_outing_buses_sufficient_l1650_165090

/-- Proves that the total capacity of 6 large buses is sufficient to accommodate 298 students. -/
theorem school_outing_buses_sufficient (students : ℕ) (bus_capacity : ℕ) (num_buses : ℕ) : 
  students = 298 → 
  bus_capacity = 52 → 
  num_buses = 6 → 
  num_buses * bus_capacity ≥ students := by
sorry

end NUMINAMATH_CALUDE_school_outing_buses_sufficient_l1650_165090


namespace NUMINAMATH_CALUDE_abs_z_equals_five_l1650_165009

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := sorry

-- State the theorem
theorem abs_z_equals_five :
  z * i^2018 = 3 + 4*i → Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_abs_z_equals_five_l1650_165009


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_order_l1650_165070

open Complex

theorem smallest_root_of_unity_order : ∃ (n : ℕ), n > 0 ∧ 
  (∀ z : ℂ, z^3 - z + 1 = 0 → z^n = 1) ∧ 
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^3 - z + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_order_l1650_165070


namespace NUMINAMATH_CALUDE_cost_per_serving_is_50_cents_l1650_165034

/-- Calculates the cost per serving of mixed nuts in cents after applying a coupon -/
def cost_per_serving (original_price : ℚ) (bag_size : ℚ) (coupon_value : ℚ) (serving_size : ℚ) : ℚ :=
  ((original_price - coupon_value) / bag_size) * serving_size * 100

/-- Proves that the cost per serving of mixed nuts is 50 cents after applying the coupon -/
theorem cost_per_serving_is_50_cents :
  cost_per_serving 25 40 5 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_serving_is_50_cents_l1650_165034


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l1650_165099

/-- Given a circle with center (4, 2) and one endpoint of its diameter at (7, 5),
    prove that the other endpoint of the diameter is at (1, -1). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint_a : ℝ × ℝ) (endpoint_b : ℝ × ℝ) :
  center = (4, 2) →
  endpoint_a = (7, 5) →
  (center.1 - endpoint_a.1 = endpoint_b.1 - center.1 ∧
   center.2 - endpoint_a.2 = endpoint_b.2 - center.2) →
  endpoint_b = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l1650_165099


namespace NUMINAMATH_CALUDE_direct_proportion_properties_l1650_165082

/-- A function representing direct proportion --/
def f (k : ℝ) (x : ℝ) : ℝ := k * x

/-- Theorem stating the properties of the function f --/
theorem direct_proportion_properties :
  ∀ k : ℝ, k ≠ 0 →
  (f k 2 = 4) →
  ∃ a : ℝ, (f k a = 3 ∧ k = 2 ∧ a = 3/2) := by
  sorry

#check direct_proportion_properties

end NUMINAMATH_CALUDE_direct_proportion_properties_l1650_165082


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1650_165035

/-- The quadratic function f(x) = -2(x+3)^2 - 5 has vertex at (-3, -5) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -2 * (x + 3)^2 - 5
  (∀ x, f x ≤ f (-3)) ∧ f (-3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1650_165035


namespace NUMINAMATH_CALUDE_platform_length_platform_length_is_340_l1650_165065

/-- The length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- The platform length is 340 meters -/
theorem platform_length_is_340 :
  platform_length 360 45 56 = 340 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_is_340_l1650_165065


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1650_165066

theorem angle_in_fourth_quadrant (α : Real) :
  (0 < α) ∧ (α < π / 2) → (3 * π / 2 < (2 * π - α)) ∧ ((2 * π - α) < 2 * π) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1650_165066


namespace NUMINAMATH_CALUDE_jack_payback_l1650_165016

/-- The amount borrowed by Jack -/
def principal : ℚ := 1200

/-- The interest rate as a decimal -/
def interestRate : ℚ := 1/10

/-- The amount Jack will pay back -/
def amountToPay : ℚ := principal * (1 + interestRate)

/-- Theorem stating that the amount Jack will pay back is 1320 -/
theorem jack_payback : amountToPay = 1320 := by sorry

end NUMINAMATH_CALUDE_jack_payback_l1650_165016


namespace NUMINAMATH_CALUDE_parabola_equation_holds_l1650_165077

/-- A parabola with vertex at (2, 9) intersecting the x-axis to form a segment of length 6 -/
structure Parabola where
  vertex : ℝ × ℝ
  intersection_length : ℝ
  vertex_condition : vertex = (2, 9)
  length_condition : intersection_length = 6

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) (x y : ℝ) : Prop :=
  y = -(x - 2)^2 + 9

/-- Theorem stating that the given parabola satisfies the equation -/
theorem parabola_equation_holds (p : Parabola) :
  ∀ x y : ℝ, parabola_equation p x y ↔ 
    ∃ a : ℝ, y = a * (x - p.vertex.1)^2 + p.vertex.2 ∧
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
      a * (x₁ - p.vertex.1)^2 + p.vertex.2 = 0 ∧
      a * (x₂ - p.vertex.1)^2 + p.vertex.2 = 0 ∧
      |x₁ - x₂| = p.intersection_length :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_holds_l1650_165077


namespace NUMINAMATH_CALUDE_min_breaks_for_40_tiles_l1650_165052

/-- Represents a chocolate bar -/
structure ChocolateBar where
  tiles : ℕ

/-- Represents the breaking process -/
def breakChocolate (initial : ChocolateBar) (breaks : ℕ) : ℕ :=
  initial.tiles + breaks

/-- Theorem: The minimum number of breaks required for a 40-tile chocolate bar is 39 -/
theorem min_breaks_for_40_tiles (bar : ChocolateBar) (h : bar.tiles = 40) :
  ∃ (breaks : ℕ), breakChocolate bar breaks = 40 ∧ 
  ∀ (n : ℕ), breakChocolate bar n = 40 → breaks ≤ n :=
by sorry

end NUMINAMATH_CALUDE_min_breaks_for_40_tiles_l1650_165052


namespace NUMINAMATH_CALUDE_basic_computer_price_theorem_l1650_165017

/-- Represents the prices of computer components and setups -/
structure ComputerPrices where
  basic_total : ℝ  -- Total price of basic setup
  enhanced_total : ℝ  -- Total price of enhanced setup
  printer_ratio : ℝ  -- Ratio of printer price to enhanced total
  monitor_ratio : ℝ  -- Ratio of monitor price to enhanced total
  keyboard_ratio : ℝ  -- Ratio of keyboard price to enhanced total

/-- Calculates the price of the basic computer given the prices and ratios -/
def basic_computer_price (prices : ComputerPrices) : ℝ :=
  let enhanced_computer := prices.enhanced_total * (1 - prices.printer_ratio - prices.monitor_ratio - prices.keyboard_ratio)
  enhanced_computer - (prices.enhanced_total - prices.basic_total)

/-- Theorem stating that the basic computer price is approximately $975.83 -/
theorem basic_computer_price_theorem (prices : ComputerPrices) 
  (h1 : prices.basic_total = 2500)
  (h2 : prices.enhanced_total = prices.basic_total + 600)
  (h3 : prices.printer_ratio = 1/6)
  (h4 : prices.monitor_ratio = 1/5)
  (h5 : prices.keyboard_ratio = 1/8) :
  ∃ ε > 0, |basic_computer_price prices - 975.83| < ε :=
sorry

end NUMINAMATH_CALUDE_basic_computer_price_theorem_l1650_165017


namespace NUMINAMATH_CALUDE_expression_simplification_l1650_165080

theorem expression_simplification :
  (((1 + 2 + 3) * 2)^2 / 3) + ((3 * 4 + 6 + 2) / 5) = 52 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1650_165080


namespace NUMINAMATH_CALUDE_pet_ratio_l1650_165039

theorem pet_ratio (dogs : ℕ) (cats : ℕ) (total_pets : ℕ) : 
  dogs = 2 → cats = 3 → total_pets = 15 → 
  (total_pets - (dogs + cats)) * 1 = 2 * (dogs + cats) := by
  sorry

end NUMINAMATH_CALUDE_pet_ratio_l1650_165039


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1650_165012

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 1) → 
  (1^2 - 2*1 + m = 1) → 
  m = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1650_165012


namespace NUMINAMATH_CALUDE_balloon_height_theorem_l1650_165056

/-- Calculates the maximum height a balloon can fly given the budget and costs --/
def maxBalloonHeight (budget initialCost heliumPrice1 heliumPrice2 heliumPrice3 : ℚ) 
  (threshold1 threshold2 : ℚ) (heightPerOunce : ℚ) : ℚ :=
  let remainingBudget := budget - initialCost
  let ounces1 := min (remainingBudget / heliumPrice1) threshold1
  let ounces2 := min ((remainingBudget - ounces1 * heliumPrice1) / heliumPrice2) (threshold2 - threshold1)
  let totalOunces := ounces1 + ounces2
  totalOunces * heightPerOunce

/-- The maximum height the balloon can fly is 11,000 feet --/
theorem balloon_height_theorem : 
  maxBalloonHeight 200 74 1.2 1.1 1 50 120 100 = 11000 := by
  sorry

end NUMINAMATH_CALUDE_balloon_height_theorem_l1650_165056


namespace NUMINAMATH_CALUDE_tom_buys_four_papayas_l1650_165005

/-- Represents the fruit purchase scenario --/
structure FruitPurchase where
  lemon_price : ℕ
  papaya_price : ℕ
  mango_price : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ
  lemons_bought : ℕ
  mangos_bought : ℕ
  total_paid : ℕ

/-- Calculates the number of papayas bought --/
def papayas_bought (fp : FruitPurchase) (p : ℕ) : Prop :=
  let total_fruits := fp.lemons_bought + fp.mangos_bought + p
  let total_cost := fp.lemon_price * fp.lemons_bought + 
                    fp.papaya_price * p + 
                    fp.mango_price * fp.mangos_bought
  let discount := (total_fruits / fp.discount_threshold) * fp.discount_amount
  total_cost - discount = fp.total_paid

/-- Theorem stating that Tom buys 4 papayas --/
theorem tom_buys_four_papayas : 
  ∃ (fp : FruitPurchase), 
    fp.lemon_price = 2 ∧ 
    fp.papaya_price = 1 ∧ 
    fp.mango_price = 4 ∧ 
    fp.discount_threshold = 4 ∧ 
    fp.discount_amount = 1 ∧ 
    fp.lemons_bought = 6 ∧ 
    fp.mangos_bought = 2 ∧ 
    fp.total_paid = 21 ∧ 
    papayas_bought fp 4 :=
sorry

end NUMINAMATH_CALUDE_tom_buys_four_papayas_l1650_165005


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1650_165004

theorem decimal_multiplication (a b : ℕ) (h : a * b = 19732) :
  (a : ℚ) / 100 * ((b : ℚ) / 100) = 1.9732 :=
by sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l1650_165004


namespace NUMINAMATH_CALUDE_relation_between_exponents_l1650_165033

-- Define variables
variable (a b c d x y p z : ℝ)

-- Define the theorem
theorem relation_between_exponents 
  (h1 : a^x = b^p) 
  (h2 : b^p = c)
  (h3 : b^y = a^z) 
  (h4 : a^z = d)
  : p * y = x * z := by
  sorry

end NUMINAMATH_CALUDE_relation_between_exponents_l1650_165033


namespace NUMINAMATH_CALUDE_distance_between_points_l1650_165059

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1650_165059


namespace NUMINAMATH_CALUDE_special_polyhedron_value_l1650_165081

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  T : ℕ
  P : ℕ
  is_convex : Prop
  face_count : faces = 32
  face_types : faces = triangles + pentagons
  vertex_config : Prop
  euler_formula : vertices - edges + faces = 2

/-- Theorem stating the specific value for the polyhedron -/
theorem special_polyhedron_value (poly : SpecialPolyhedron) :
  100 * poly.P + 10 * poly.T + poly.vertices = 250 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_value_l1650_165081


namespace NUMINAMATH_CALUDE_simplify_expression_l1650_165083

theorem simplify_expression :
  2 + (1 / (2 + Real.sqrt 5)) - (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1650_165083


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_odd_integers_l1650_165095

theorem sum_of_five_consecutive_odd_integers (n : ℤ) : 
  (n + (n + 8) = 156) → (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 390) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_odd_integers_l1650_165095


namespace NUMINAMATH_CALUDE_correct_transformation_l1650_165032

theorem correct_transformation (y : ℝ) : 
  (|y + 1| / 2 = |y| / 3 - |3*y - 1| / 6 - y) ↔ 
  (3*y + 3 = 2*y - 3*y + 1 - 6*y) := by
sorry

end NUMINAMATH_CALUDE_correct_transformation_l1650_165032


namespace NUMINAMATH_CALUDE_negative_product_plus_two_l1650_165048

theorem negative_product_plus_two :
  ∀ (a b : ℤ), a = -2 → b = -3 → a * b + 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_plus_two_l1650_165048


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1650_165058

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1650_165058


namespace NUMINAMATH_CALUDE_smaller_number_l1650_165011

theorem smaller_number (x y : ℝ) (sum_eq : x + y = 30) (diff_eq : x - y = 10) : 
  min x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_l1650_165011


namespace NUMINAMATH_CALUDE_exp_properties_l1650_165018

-- Define the exponential function as a power series
noncomputable def Exp (z : ℂ) : ℂ := ∑' n, z^n / n.factorial

-- State the properties to be proven
theorem exp_properties :
  (∀ z : ℂ, HasDerivAt Exp (Exp z) z) ∧
  (∀ (α β : ℝ) (z : ℂ), Exp ((α + β) • z) = Exp (α • z) * Exp (β • z)) := by
  sorry

end NUMINAMATH_CALUDE_exp_properties_l1650_165018


namespace NUMINAMATH_CALUDE_circle_diameter_l1650_165026

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_l1650_165026


namespace NUMINAMATH_CALUDE_union_equality_implies_a_values_l1650_165031

def A (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def B : Set ℝ := {1, 2}

theorem union_equality_implies_a_values (a : ℝ) : 
  A a ∪ B = B → a = 0 ∨ a = 1/2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_values_l1650_165031


namespace NUMINAMATH_CALUDE_fraction_equality_l1650_165060

theorem fraction_equality : (1 / 5 - 1 / 6) / (1 / 4 - 1 / 5) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1650_165060


namespace NUMINAMATH_CALUDE_situp_ratio_l1650_165047

/-- The number of sit-ups Ken can do -/
def ken_situps : ℕ := 20

/-- The number of sit-ups Nathan can do -/
def nathan_situps : ℕ := 2 * ken_situps

/-- The number of sit-ups Bob can do -/
def bob_situps : ℕ := ken_situps + 10

/-- The combined number of sit-ups Ken and Nathan can do -/
def ken_nathan_combined : ℕ := ken_situps + nathan_situps

theorem situp_ratio : 
  (bob_situps : ℚ) / ken_nathan_combined = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_situp_ratio_l1650_165047


namespace NUMINAMATH_CALUDE_point_A_location_l1650_165074

theorem point_A_location (A : ℝ) : 
  (A + 2 = -2 ∨ A - 2 = -2) → (A = 0 ∨ A = -4) := by
sorry

end NUMINAMATH_CALUDE_point_A_location_l1650_165074


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1650_165019

theorem smaller_number_in_ratio (x y d : ℝ) : 
  x > 0 → y > 0 → x / y = 2 / 3 → 2 * x + 3 * y = d → min x y = 2 * d / 13 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1650_165019


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1650_165053

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a3_eq_6 : a 3 = 6
  S3_eq_12 : S 3 = 12

/-- The theorem stating properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 2 * n) ∧
  (∀ n, seq.S n = n * (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1650_165053


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l1650_165037

theorem distinct_prime_factors_of_30_factorial (n : ℕ) :
  n = 30 →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card = 
  (Finset.filter (λ p => p.Prime ∧ p ∣ n.factorial) (Finset.range (n + 1))).card :=
by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l1650_165037


namespace NUMINAMATH_CALUDE_no_valid_base_solution_l1650_165014

theorem no_valid_base_solution : 
  ¬∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 
    (4 * x + 9 = 4 * y + 1) ∧ 
    (4 * x^2 + 7 * x + 7 = 3 * y^2 + 2 * y + 9) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_base_solution_l1650_165014


namespace NUMINAMATH_CALUDE_prob_at_least_one_3_l1650_165010

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The probability of rolling a 3 on a single fair die -/
def probThree : ℚ := 1 / numSides

/-- The probability of not rolling a 3 on a single fair die -/
def probNotThree : ℚ := 1 - probThree

/-- The probability of rolling at least one 3 when two fair dice are rolled -/
def probAtLeastOne3 : ℚ := 1 - probNotThree * probNotThree

theorem prob_at_least_one_3 : probAtLeastOne3 = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_3_l1650_165010


namespace NUMINAMATH_CALUDE_inequality_proof_l1650_165092

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1650_165092


namespace NUMINAMATH_CALUDE_kolya_twos_count_l1650_165013

/-- Represents the grades of a student -/
structure Grades where
  fives : ℕ
  fours : ℕ
  threes : ℕ
  twos : ℕ

/-- Calculates the average grade -/
def averageGrade (g : Grades) : ℚ :=
  (5 * g.fives + 4 * g.fours + 3 * g.threes + 2 * g.twos) / 20

theorem kolya_twos_count 
  (kolya vasya : Grades)
  (total_grades : kolya.fives + kolya.fours + kolya.threes + kolya.twos = 20)
  (vasya_total : vasya.fives + vasya.fours + vasya.threes + vasya.twos = 20)
  (fives_eq : kolya.fives = vasya.fours)
  (fours_eq : kolya.fours = vasya.threes)
  (threes_eq : kolya.threes = vasya.twos)
  (twos_eq : kolya.twos = vasya.fives)
  (avg_eq : averageGrade kolya = averageGrade vasya) :
  kolya.twos = 5 := by
sorry

end NUMINAMATH_CALUDE_kolya_twos_count_l1650_165013


namespace NUMINAMATH_CALUDE_shoe_pairing_probability_l1650_165091

/-- A permutation of n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The number of permutations of n elements with all cycle lengths ≥ k -/
def numLongCyclePerms (n k : ℕ) : ℕ := sorry

/-- The probability of a random permutation of n elements having all cycle lengths ≥ k -/
def probLongCycles (n k : ℕ) : ℚ :=
  (numLongCyclePerms n k : ℚ) / (n.factorial : ℚ)

theorem shoe_pairing_probability :
  probLongCycles 8 5 = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_shoe_pairing_probability_l1650_165091


namespace NUMINAMATH_CALUDE_investment_calculation_l1650_165062

/-- Calculates the total investment given share details and dividend income -/
def calculate_investment (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * face_value
  let number_of_shares := annual_income / dividend_per_share
  number_of_shares * quoted_price

/-- Theorem stating that the investment is 4940 given the problem conditions -/
theorem investment_calculation :
  calculate_investment 10 9.5 14 728 = 4940 := by
  sorry

#eval calculate_investment 10 9.5 14 728

end NUMINAMATH_CALUDE_investment_calculation_l1650_165062


namespace NUMINAMATH_CALUDE_additional_distance_for_average_speed_l1650_165006

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (second_speed : ℝ)
  (target_average_speed : ℝ)
  (h : initial_distance = 20)
  (h1 : initial_speed = 40)
  (h2 : second_speed = 60)
  (h3 : target_average_speed = 55)
  : ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_average_speed ∧
    additional_distance = 90 := by
  sorry

end NUMINAMATH_CALUDE_additional_distance_for_average_speed_l1650_165006


namespace NUMINAMATH_CALUDE_power_sum_difference_l1650_165054

theorem power_sum_difference : 8^3 + 8^3 + 8^3 + 8^3 - 2^6 * 2^3 = 1536 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l1650_165054


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1650_165027

/-- Given a parabola with equation x² = 8y, the distance from its focus to its directrix is 4. -/
theorem parabola_focus_directrix_distance : 
  ∀ (x y : ℝ), x^2 = 8*y → (∃ (focus_distance : ℝ), focus_distance = 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1650_165027


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l1650_165073

theorem division_multiplication_problem : 
  let number : ℚ := 4
  let divisor : ℚ := 6
  let multiplier : ℚ := 12
  let result : ℚ := 8
  (number / divisor) * multiplier = result := by sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l1650_165073


namespace NUMINAMATH_CALUDE_notebook_cost_l1650_165041

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : 
  total_students = 36 →
  total_cost = 2772 →
  ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
    buying_students > total_students / 2 ∧
    notebooks_per_student > 2 ∧
    cost_per_notebook = 2 * notebooks_per_student ∧
    buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook = 12 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l1650_165041


namespace NUMINAMATH_CALUDE_work_completion_proof_l1650_165071

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 10

/-- The number of days it takes with fewer workers -/
def fewer_workers_days : ℕ := 20

/-- The reduction in the number of workers -/
def worker_reduction : ℕ := 10

/-- The original number of workers -/
def original_workers : ℕ := 20

theorem work_completion_proof :
  (original_workers * original_days = (original_workers - worker_reduction) * fewer_workers_days) ∧
  (original_workers > worker_reduction) :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l1650_165071


namespace NUMINAMATH_CALUDE_scalar_product_formula_l1650_165020

def vector_2d (x y : ℝ) : ℝ × ℝ := (x, y)

def scalar_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem scalar_product_formula (x₁ y₁ x₂ y₂ : ℝ) :
  scalar_product (vector_2d x₁ y₁) (vector_2d x₂ y₂) = x₁ * x₂ + y₁ * y₂ := by
  sorry

end NUMINAMATH_CALUDE_scalar_product_formula_l1650_165020


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1650_165061

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1650_165061


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l1650_165094

theorem angle_measure_in_triangle (D E F : ℝ) : 
  D = 90 →  -- Angle D is 90 degrees
  E = 4 * F - 10 →  -- Angle E is 10 degrees less than four times angle F
  D + E + F = 180 →  -- Sum of angles in a triangle is 180 degrees
  F = 20 :=  -- Measure of angle F is 20 degrees
by sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l1650_165094


namespace NUMINAMATH_CALUDE_investment_interest_proof_l1650_165086

/-- Calculates the total interest earned on an investment with compound interest. -/
def total_interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the total interest earned on $1,500 invested at 5% annual interest
    rate compounded annually for 5 years is approximately $414.42. -/
theorem investment_interest_proof :
  ∃ ε > 0, |total_interest_earned 1500 0.05 5 - 414.42| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l1650_165086


namespace NUMINAMATH_CALUDE_max_combined_power_l1650_165051

theorem max_combined_power (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ < 1) (h₂ : x₂ < 1) (h₃ : x₃ < 1)
  (h : 2 * (x₁ + x₂ + x₃) + 4 * x₁ * x₂ * x₃ = 3 * (x₁ * x₂ + x₁ * x₃ + x₂ * x₃) + 1) :
  x₁ + x₂ + x₃ ≤ 3/4 := by
sorry

end NUMINAMATH_CALUDE_max_combined_power_l1650_165051


namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l1650_165050

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m * (m + 2)) (m^2 + m - 2)

-- Part 1: z is purely imaginary iff m = 0
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * Complex.im (z m) ↔ m = 0 :=
sorry

-- Part 2: z is in the fourth quadrant iff 0 < m < 1
theorem z_in_fourth_quadrant (m : ℝ) : 
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0) ↔ (0 < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_in_fourth_quadrant_l1650_165050


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1650_165002

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 250 →
  crossing_time = 20 →
  train_speed = 66.6 →
  (train_speed * crossing_time) - train_length = 1082 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1650_165002


namespace NUMINAMATH_CALUDE_select_five_from_eight_with_book_a_l1650_165001

/-- The number of ways to select 5 books from 8 books, always including "Book A" -/
def select_books (total_books : ℕ) (books_to_select : ℕ) : ℕ :=
  Nat.choose (total_books - 1) (books_to_select - 1)

/-- Theorem: Selecting 5 books from 8 books, always including "Book A", can be done in 35 ways -/
theorem select_five_from_eight_with_book_a : select_books 8 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_with_book_a_l1650_165001


namespace NUMINAMATH_CALUDE_valid_division_l1650_165029

theorem valid_division (divisor quotient remainder dividend : ℕ) : 
  divisor = 3040 →
  quotient = 8 →
  remainder = 7 →
  dividend = 24327 →
  dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_valid_division_l1650_165029


namespace NUMINAMATH_CALUDE_cube_volume_in_box_l1650_165030

/-- Given a box with dimensions 9 cm × 12 cm × 3 cm, filled with 108 identical cubes,
    the volume of each cube is 27 cm³. -/
theorem cube_volume_in_box (length width height : ℕ) (num_cubes : ℕ) :
  length = 9 ∧ width = 12 ∧ height = 3 ∧ num_cubes = 108 →
  ∃ (cube_volume : ℕ), cube_volume = 27 ∧ num_cubes * cube_volume = length * width * height :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_in_box_l1650_165030


namespace NUMINAMATH_CALUDE_shortest_tree_height_l1650_165028

/-- The heights of four trees satisfying certain conditions -/
structure TreeHeights where
  tallest : ℝ
  second_tallest : ℝ
  third_tallest : ℝ
  shortest : ℝ
  tallest_height : tallest = 108
  second_tallest_height : second_tallest = tallest / 2 - 6
  third_tallest_height : third_tallest = second_tallest / 4
  shortest_height : shortest = second_tallest + third_tallest - 2

/-- The height of the shortest tree is 58 feet -/
theorem shortest_tree_height (t : TreeHeights) : t.shortest = 58 := by
  sorry

end NUMINAMATH_CALUDE_shortest_tree_height_l1650_165028


namespace NUMINAMATH_CALUDE_assembly_line_production_rate_l1650_165024

theorem assembly_line_production_rate 
  (initial_rate : ℝ) 
  (initial_order : ℝ) 
  (second_order : ℝ) 
  (average_output : ℝ) 
  (h1 : initial_rate = 90) 
  (h2 : initial_order = 60) 
  (h3 : second_order = 60) 
  (h4 : average_output = 72) : 
  ∃ (reduced_rate : ℝ), 
    reduced_rate = 60 ∧ 
    (initial_order / initial_rate + second_order / reduced_rate) * average_output = initial_order + second_order :=
by sorry

end NUMINAMATH_CALUDE_assembly_line_production_rate_l1650_165024


namespace NUMINAMATH_CALUDE_equation_solution_l1650_165076

theorem equation_solution : ∃ x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1650_165076


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l1650_165040

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l1650_165040


namespace NUMINAMATH_CALUDE_point_positions_l1650_165038

def line_equation (x y : ℝ) : Prop := 3 * x - 5 * y + 8 = 0

def point_A : ℝ × ℝ := (2, 5)
def point_B : ℝ × ℝ := (1, 2.2)

theorem point_positions :
  (¬ line_equation point_A.1 point_A.2) ∧
  (line_equation point_B.1 point_B.2) :=
by sorry

end NUMINAMATH_CALUDE_point_positions_l1650_165038


namespace NUMINAMATH_CALUDE_balloon_height_is_9482_l1650_165023

/-- Calculates the maximum height a balloon can fly given the following parameters:
    * total_money: The total amount of money available
    * sheet_cost: The cost of the balloon sheet
    * rope_cost: The cost of the rope
    * propane_cost: The cost of the propane tank and burner
    * helium_cost_per_oz: The cost of helium per ounce
    * height_per_oz: The height gain per ounce of helium
-/
def max_balloon_height (total_money sheet_cost rope_cost propane_cost helium_cost_per_oz height_per_oz : ℚ) : ℚ :=
  let remaining_money := total_money - (sheet_cost + rope_cost + propane_cost)
  let helium_oz := remaining_money / helium_cost_per_oz
  helium_oz * height_per_oz

/-- Theorem stating that given the specific conditions in the problem,
    the maximum height the balloon can fly is 9482 feet. -/
theorem balloon_height_is_9482 :
  max_balloon_height 200 42 18 14 1.5 113 = 9482 := by
  sorry

end NUMINAMATH_CALUDE_balloon_height_is_9482_l1650_165023


namespace NUMINAMATH_CALUDE_symmetry_axis_l1650_165045

/-- Given two lines l₁ and l₂ in a 2D plane, this function returns true if they are symmetric about a third line l. -/
def are_symmetric (l₁ l₂ l : ℝ → ℝ → Prop) : Prop := sorry

/-- The line with equation y = -x -/
def line_l₁ (x y : ℝ) : Prop := y = -x

/-- The line with equation x + y - 2 = 0 -/
def line_l₂ (x y : ℝ) : Prop := x + y - 2 = 0

/-- The proposed axis of symmetry -/
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

theorem symmetry_axis :
  are_symmetric line_l₁ line_l₂ line_l :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_l1650_165045


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l1650_165046

/-- The length of the path traveled by point B when rolling a quarter-circle along another quarter-circle -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 4 / π) :
  let path_length := 2 * π * r
  path_length = 8 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l1650_165046


namespace NUMINAMATH_CALUDE_degree_of_derivative_P_l1650_165044

/-- The polynomial we are working with -/
def P (x : ℝ) : ℝ := (x^2 + 1)^5 * (x^4 + 1)^2

/-- The degree of a polynomial -/
noncomputable def degree (p : ℝ → ℝ) : ℕ := sorry

/-- The derivative of a polynomial -/
noncomputable def derivative (p : ℝ → ℝ) : ℝ → ℝ := sorry

theorem degree_of_derivative_P :
  degree (derivative P) = 17 := by sorry

end NUMINAMATH_CALUDE_degree_of_derivative_P_l1650_165044


namespace NUMINAMATH_CALUDE_phd_total_time_l1650_165008

def phd_timeline (acclimation_time : ℝ) (basics_time : ℝ) (research_ratio : ℝ) (dissertation_ratio : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_ratio)
  let dissertation_time := acclimation_time * dissertation_ratio
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_total_time :
  phd_timeline 1 2 0.75 0.5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_phd_total_time_l1650_165008


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l1650_165015

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits. -/
def binary (bits : List Bool) : Nat := binaryToNat bits

theorem binary_arithmetic_equality : 
  (binary [true, false, true, true, true, false] + binary [true, false, true, false, true]) -
  (binary [true, true, true, false, false, false] - binary [true, true, false, true, false, true]) +
  binary [true, true, true, false, true] =
  binary [true, false, true, true, true, false, true] := by
  sorry

#eval binary [true, false, true, true, true, false, true]

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l1650_165015


namespace NUMINAMATH_CALUDE_man_upstream_speed_l1650_165098

/-- Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem man_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 20)
  (h2 : speed_downstream = 25) :
  speed_still - (speed_downstream - speed_still) = 15 := by
  sorry


end NUMINAMATH_CALUDE_man_upstream_speed_l1650_165098


namespace NUMINAMATH_CALUDE_equality_of_fractions_l1650_165003

theorem equality_of_fractions (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 49 ∧ (4 : ℚ) / 7 = 84 / N → M - N = -119 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l1650_165003


namespace NUMINAMATH_CALUDE_monkey_reach_top_l1650_165097

/-- The time it takes for a monkey to climb a greased pole -/
def monkey_climb_time (pole_height : ℕ) (ascend : ℕ) (slip : ℕ) : ℕ :=
  let effective_progress := ascend - slip
  let full_cycles := (pole_height - ascend) / effective_progress
  2 * full_cycles + 1

/-- Theorem stating that the monkey will reach the top of the pole in 17 minutes -/
theorem monkey_reach_top : monkey_climb_time 10 2 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_monkey_reach_top_l1650_165097


namespace NUMINAMATH_CALUDE_apollonian_circle_apollonian_circle_specific_case_l1650_165096

/-- The locus of points with a constant ratio of distances to two fixed points is a circle. -/
theorem apollonian_circle (k : ℝ) (hk : k > 0 ∧ k ≠ 1) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ (x y : ℝ),
      (Real.sqrt ((x + 1)^2 + y^2)) / (Real.sqrt ((x - 2)^2 + y^2)) = k ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

/-- The specific case where the ratio is 2 and the fixed points are A(-1,0) and B(2,0). -/
theorem apollonian_circle_specific_case :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (3, 0) ∧ radius = 2 ∧
    ∀ (x y : ℝ),
      (Real.sqrt ((x + 1)^2 + y^2)) / (Real.sqrt ((x - 2)^2 + y^2)) = 2 ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_apollonian_circle_apollonian_circle_specific_case_l1650_165096


namespace NUMINAMATH_CALUDE_expression_evaluation_l1650_165089

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -2
  (a + 2*b) * (a - b) + (a^3*b + 4*a*b^3) / (a*b) = 15/2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1650_165089


namespace NUMINAMATH_CALUDE_right_triangle_area_l1650_165021

/-- The area of a right triangle with vertices at (-3,0), (0,2), and (0,0) is 3 square units. -/
theorem right_triangle_area : 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (0, 0)
  -- Assume the triangle is right-angled
  (B.1 - C.1) * (A.2 - C.2) = (A.1 - C.1) * (B.2 - C.2) →
  -- The area of the triangle
  1/2 * |A.1 - C.1| * |B.2 - C.2| = 3 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_area_l1650_165021


namespace NUMINAMATH_CALUDE_october_price_correct_l1650_165069

/-- The price of a mobile phone after a certain number of months, given an initial price and a monthly decrease rate. -/
def price_after_months (initial_price : ℝ) (decrease_rate : ℝ) (months : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ months

/-- Theorem stating that the price of a mobile phone in October is correct, given the initial price in January and a 3% monthly decrease. -/
theorem october_price_correct (a : ℝ) : price_after_months a 0.03 9 = a * 0.97^9 := by
  sorry

#check october_price_correct

end NUMINAMATH_CALUDE_october_price_correct_l1650_165069


namespace NUMINAMATH_CALUDE_triangle_side_ratio_range_l1650_165078

theorem triangle_side_ratio_range (a b c : ℝ) (A : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < Real.pi / 2 →
  a^2 = b^2 + b*c →
  Real.sqrt 2 < a/b ∧ a/b < 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_range_l1650_165078
