import Mathlib

namespace NUMINAMATH_CALUDE_least_difference_consecutive_primes_l3687_368715

theorem least_difference_consecutive_primes (x y z : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧  -- x, y, and z are prime numbers
  x < y ∧ y < z ∧                -- x < y < z
  y - x > 3 ∧                    -- y - x > 3
  Even x ∧                       -- x is an even integer
  Odd y ∧ Odd z →                -- y and z are odd integers
  ∀ w, (Prime w ∧ Prime (w + 1) ∧ Prime (w + 2) ∧ 
        w < w + 1 ∧ w + 1 < w + 2 ∧
        (w + 1) - w > 3 ∧
        Even w ∧ Odd (w + 1) ∧ Odd (w + 2)) →
    (w + 2) - w ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_least_difference_consecutive_primes_l3687_368715


namespace NUMINAMATH_CALUDE_range_of_m_l3687_368787

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀ + (1/3) * m = Real.exp x₀

def q (m : ℝ) : Prop :=
  let a := m
  let b := 5
  let e := Real.sqrt ((a - b) / a)
  (1/2) < e ∧ e < (2/3)

-- Define the theorem
theorem range_of_m (m : ℝ) (h : p m ∧ q m) :
  (20/3 < m ∧ m < 9) ∨ (3 ≤ m ∧ m < 15/4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3687_368787


namespace NUMINAMATH_CALUDE_chosen_number_proof_l3687_368711

theorem chosen_number_proof (x : ℝ) : (x / 9) - 100 = 10 → x = 990 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l3687_368711


namespace NUMINAMATH_CALUDE_smallest_n_value_l3687_368749

theorem smallest_n_value (r g b : ℕ+) (h : 10 * r = 18 * g ∧ 18 * g = 20 * b) :
  ∃ (n : ℕ+), 30 * n = 10 * r ∧ ∀ (m : ℕ+), 30 * m = 10 * r → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3687_368749


namespace NUMINAMATH_CALUDE_equation_solution_l3687_368700

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 - ⌊x₁⌋ = 2019) ∧ 
    (x₂^2 - ⌊x₂⌋ = 2019) ∧ 
    (x₁ = -Real.sqrt 1974) ∧ 
    (x₂ = Real.sqrt 2064) ∧ 
    (∀ (x : ℝ), x^2 - ⌊x⌋ = 2019 → x = x₁ ∨ x = x₂) :=
by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3687_368700


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3687_368744

theorem unique_solution_for_equation (N : ℕ) (a b c : ℕ+) :
  N > 3 →
  Odd N →
  a ^ N = b ^ N + 2 ^ N + a * b * c →
  c ≤ 5 * 2 ^ (N - 1) →
  N = 5 ∧ a = 3 ∧ b = 1 ∧ c = 70 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3687_368744


namespace NUMINAMATH_CALUDE_uncle_bobs_age_l3687_368743

theorem uncle_bobs_age (anna_age brianna_age caitlin_age bob_age : ℕ) : 
  anna_age = 48 →
  brianna_age = anna_age / 2 →
  caitlin_age = brianna_age - 6 →
  bob_age = 3 * caitlin_age →
  bob_age = 54 := by
sorry

end NUMINAMATH_CALUDE_uncle_bobs_age_l3687_368743


namespace NUMINAMATH_CALUDE_sum_inequality_l3687_368726

theorem sum_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  (a / (1 - a^2)) + (b / (1 - b^2)) + (c / (1 - c^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3687_368726


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l3687_368740

/-- Given a function f(x) = ax - a/x - 2ln(x) where a ≥ 0, if f(x) is monotonically increasing
    on its domain (0, +∞), then a > 1. -/
theorem monotonic_increasing_condition (a : ℝ) (h_a : a ≥ 0) :
  (∀ x : ℝ, x > 0 → Monotone (fun x => a * x - a / x - 2 * Real.log x)) →
  a > 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l3687_368740


namespace NUMINAMATH_CALUDE_ellipse_m_values_l3687_368768

/-- An ellipse with equation x²/5 + y²/m = 1 and eccentricity √10/5 has m equal to 3 or 25/3 -/
theorem ellipse_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2/5 + y^2/m = 1) →  -- Ellipse equation
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (x^2/a^2 + y^2/b^2 = 1 ↔ x^2/5 + y^2/m = 1) ∧  -- Standard form of ellipse
    c^2/a^2 = 10/25) →  -- Eccentricity condition
  m = 3 ∨ m = 25/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_values_l3687_368768


namespace NUMINAMATH_CALUDE_problem_solution_l3687_368794

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 2 = d + Real.sqrt (a + b + c - d + 1) → d = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3687_368794


namespace NUMINAMATH_CALUDE_sector_angle_l3687_368728

theorem sector_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 2) :
  ∃ (α r : ℝ), α * r = arc_length ∧ (1/2) * α * r^2 = area ∧ α = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3687_368728


namespace NUMINAMATH_CALUDE_max_goats_after_trading_l3687_368731

/-- Represents the trading system with coconuts, crabs, and goats -/
structure TradingSystem where
  coconuts_per_crab : ℕ
  crabs_per_goat : ℕ
  initial_coconuts : ℕ

/-- Calculates the number of goats obtained from trading coconuts -/
def goats_from_coconuts (ts : TradingSystem) : ℕ :=
  (ts.initial_coconuts / ts.coconuts_per_crab) / ts.crabs_per_goat

/-- Theorem stating that Max will have 19 goats after trading -/
theorem max_goats_after_trading :
  let ts : TradingSystem := {
    coconuts_per_crab := 3,
    crabs_per_goat := 6,
    initial_coconuts := 342
  }
  goats_from_coconuts ts = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_goats_after_trading_l3687_368731


namespace NUMINAMATH_CALUDE_max_cookies_ella_l3687_368702

/-- Represents the recipe for cookies -/
structure Recipe where
  chocolate : Rat
  sugar : Rat
  eggs : Nat
  flour : Rat
  cookies : Nat

/-- Represents available ingredients -/
structure Ingredients where
  chocolate : Rat
  sugar : Rat
  eggs : Nat
  flour : Rat

/-- Calculates the maximum number of cookies that can be made -/
def maxCookies (recipe : Recipe) (ingredients : Ingredients) : Nat :=
  min
    (Nat.floor ((ingredients.chocolate / recipe.chocolate) * recipe.cookies))
    (min
      (Nat.floor ((ingredients.sugar / recipe.sugar) * recipe.cookies))
      (min
        ((ingredients.eggs / recipe.eggs) * recipe.cookies)
        (Nat.floor ((ingredients.flour / recipe.flour) * recipe.cookies))))

theorem max_cookies_ella :
  let recipe : Recipe := {
    chocolate := 1,
    sugar := 1/2,
    eggs := 1,
    flour := 1,
    cookies := 4
  }
  let ingredients : Ingredients := {
    chocolate := 4,
    sugar := 3,
    eggs := 6,
    flour := 10
  }
  maxCookies recipe ingredients = 16 := by
  sorry

#eval maxCookies
  { chocolate := 1, sugar := 1/2, eggs := 1, flour := 1, cookies := 4 }
  { chocolate := 4, sugar := 3, eggs := 6, flour := 10 }

end NUMINAMATH_CALUDE_max_cookies_ella_l3687_368702


namespace NUMINAMATH_CALUDE_product_inequality_l3687_368759

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a * b) ^ (1/4 : ℝ) + (b * c) ^ (1/4 : ℝ) + (c * a) ^ (1/4 : ℝ) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3687_368759


namespace NUMINAMATH_CALUDE_ladder_problem_l3687_368772

theorem ladder_problem (ladder_length height_on_wall base_distance : ℝ) : 
  ladder_length = 13 ∧ height_on_wall = 12 ∧ 
  ladder_length^2 = height_on_wall^2 + base_distance^2 → 
  base_distance = 5 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l3687_368772


namespace NUMINAMATH_CALUDE_smallest_a_is_correct_l3687_368729

/-- A polynomial of the form x^3 - ax^2 + bx - 2310 with three positive integer roots -/
structure PolynomialWithThreeRoots where
  a : ℕ
  b : ℕ
  root1 : ℕ+
  root2 : ℕ+
  root3 : ℕ+
  is_root1 : (root1 : ℝ)^3 - a*(root1 : ℝ)^2 + b*(root1 : ℝ) - 2310 = 0
  is_root2 : (root2 : ℝ)^3 - a*(root2 : ℝ)^2 + b*(root2 : ℝ) - 2310 = 0
  is_root3 : (root3 : ℝ)^3 - a*(root3 : ℝ)^2 + b*(root3 : ℝ) - 2310 = 0

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a (p : PolynomialWithThreeRoots) : ℕ := 78

theorem smallest_a_is_correct (p : PolynomialWithThreeRoots) :
  p.a ≥ smallest_a p :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_correct_l3687_368729


namespace NUMINAMATH_CALUDE_stamps_per_ounce_l3687_368789

/-- Given a letter with 8 pieces of paper each weighing 1/5 ounce,
    an envelope weighing 2/5 ounce, and requiring 2 stamps total,
    prove that 1 stamp is needed per ounce. -/
theorem stamps_per_ounce (paper_weight : ℚ) (envelope_weight : ℚ) (total_stamps : ℕ) :
  paper_weight = 1/5
  → envelope_weight = 2/5
  → total_stamps = 2
  → (total_stamps : ℚ) / (8 * paper_weight + envelope_weight) = 1 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_ounce_l3687_368789


namespace NUMINAMATH_CALUDE_base_power_zero_l3687_368756

theorem base_power_zero (b : ℝ) (x y : ℝ) (h1 : 3^x * b^y = 59049) (h2 : x - y = 10) (h3 : x = 10) : y = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_power_zero_l3687_368756


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l3687_368707

def sum_of_range (a b : ℕ) : ℕ := 
  ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  sum_of_range 50 60 + count_even_in_range 50 60 = 611 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l3687_368707


namespace NUMINAMATH_CALUDE_snack_distribution_solution_l3687_368725

/-- Represents the snack distribution problem for a kindergarten class. -/
structure SnackDistribution where
  pretzels : ℕ
  goldfish : ℕ
  suckers : ℕ
  kids : ℕ
  pretzel_popcorn_ratio : ℚ

/-- Calculates the number of items per snack type in each baggie. -/
def items_per_baggie (sd : SnackDistribution) : ℕ × ℕ × ℕ × ℕ :=
  let pretzels_per_baggie := sd.pretzels / sd.kids
  let goldfish_per_baggie := sd.goldfish / sd.kids
  let suckers_per_baggie := sd.suckers / sd.kids
  let popcorn_per_baggie := (sd.pretzel_popcorn_ratio * pretzels_per_baggie).ceil.toNat
  (pretzels_per_baggie, goldfish_per_baggie, suckers_per_baggie, popcorn_per_baggie)

/-- Calculates the total number of popcorn pieces needed. -/
def total_popcorn (sd : SnackDistribution) : ℕ :=
  let (_, _, _, popcorn_per_baggie) := items_per_baggie sd
  popcorn_per_baggie * sd.kids

/-- Calculates the total number of items in each baggie. -/
def total_items_per_baggie (sd : SnackDistribution) : ℕ :=
  let (p, g, s, c) := items_per_baggie sd
  p + g + s + c

/-- Theorem stating the solution to the snack distribution problem. -/
theorem snack_distribution_solution (sd : SnackDistribution) 
  (h1 : sd.pretzels = 64)
  (h2 : sd.goldfish = 4 * sd.pretzels)
  (h3 : sd.suckers = 32)
  (h4 : sd.kids = 23)
  (h5 : sd.pretzel_popcorn_ratio = 3/2) :
  total_popcorn sd = 69 ∧ total_items_per_baggie sd = 17 := by
  sorry


end NUMINAMATH_CALUDE_snack_distribution_solution_l3687_368725


namespace NUMINAMATH_CALUDE_rebecca_income_percentage_l3687_368730

def rebecca_income : ℕ := 15000
def jimmy_income : ℕ := 18000
def income_increase : ℕ := 7000

def new_rebecca_income : ℕ := rebecca_income + income_increase
def combined_income : ℕ := new_rebecca_income + jimmy_income

theorem rebecca_income_percentage :
  (new_rebecca_income : ℚ) / (combined_income : ℚ) = 55 / 100 := by sorry

end NUMINAMATH_CALUDE_rebecca_income_percentage_l3687_368730


namespace NUMINAMATH_CALUDE_golden_retriever_weight_at_8_years_l3687_368754

/-- Calculates the weight of a golden retriever given its age and initial conditions -/
def goldenRetrieverWeight (initialWeight : ℕ) (firstYearGain : ℕ) (yearlyGain : ℕ) (yearlyLoss : ℕ) (age : ℕ) : ℕ :=
  let weightAfterFirstYear := initialWeight + firstYearGain
  let netYearlyGain := yearlyGain - yearlyLoss
  weightAfterFirstYear + (age - 1) * netYearlyGain

/-- Theorem stating the weight of a specific golden retriever at 8 years old -/
theorem golden_retriever_weight_at_8_years :
  goldenRetrieverWeight 3 15 11 3 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_golden_retriever_weight_at_8_years_l3687_368754


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3687_368792

theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (2*a - 1)*x - a*y - 1 = 0) ↔ (a = 0 ∨ a = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3687_368792


namespace NUMINAMATH_CALUDE_tv_price_with_tax_l3687_368797

/-- Calculates the final price of a TV including value-added tax -/
theorem tv_price_with_tax (original_price : ℝ) (tax_rate : ℝ) (final_price : ℝ) :
  original_price = 1700 →
  tax_rate = 0.15 →
  final_price = original_price * (1 + tax_rate) →
  final_price = 1955 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_with_tax_l3687_368797


namespace NUMINAMATH_CALUDE_passing_marks_l3687_368764

/-- Proves that the passing marks is 120 given the conditions of the problem -/
theorem passing_marks (T : ℝ) (P : ℝ) 
  (h1 : 0.30 * T = P - 30)
  (h2 : 0.45 * T = P + 15) : P = 120 := by
  sorry

end NUMINAMATH_CALUDE_passing_marks_l3687_368764


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3687_368796

/-- Triangle inequalities -/
theorem triangle_inequalities (a b c P S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : P = a + b + c)
  (h_area : S = Real.sqrt ((P/2) * ((P/2) - a) * ((P/2) - b) * ((P/2) - c))) :
  (1/a + 1/b + 1/c ≥ 9/P) ∧
  (a^2 + b^2 + c^2 ≥ P^2/3) ∧
  (P^2 ≥ 12 * Real.sqrt 3 * S) ∧
  (a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S) ∧
  (a^3 + b^3 + c^3 ≥ P^3/9) ∧
  (a^3 + b^3 + c^3 ≥ (4 * Real.sqrt 3 / 3) * S * P) ∧
  (a^4 + b^4 + c^4 ≥ 16 * S^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l3687_368796


namespace NUMINAMATH_CALUDE_P_evaluation_l3687_368723

/-- The polynomial P(x) = x^6 - 3x^3 - x^2 - x - 2 -/
def P (x : ℤ) : ℤ := x^6 - 3*x^3 - x^2 - x - 2

/-- P is irreducible over the integers -/
axiom P_irreducible : Irreducible P

theorem P_evaluation : P 3 = 634 := by
  sorry

end NUMINAMATH_CALUDE_P_evaluation_l3687_368723


namespace NUMINAMATH_CALUDE_upstream_speed_is_eight_l3687_368757

/-- Represents the speed of a man in a stream -/
structure StreamSpeed where
  downstream : ℝ
  stream : ℝ

/-- Calculates the upstream speed given downstream and stream speeds -/
def upstreamSpeed (s : StreamSpeed) : ℝ :=
  s.downstream - 2 * s.stream

/-- Theorem stating that for given downstream and stream speeds, the upstream speed is 8 -/
theorem upstream_speed_is_eight (s : StreamSpeed) 
  (h1 : s.downstream = 15) 
  (h2 : s.stream = 3.5) : 
  upstreamSpeed s = 8 := by
  sorry

end NUMINAMATH_CALUDE_upstream_speed_is_eight_l3687_368757


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_1991_l3687_368767

theorem smallest_n_divisible_by_1991 : ∃ (n : ℕ),
  (∀ (m : ℕ), m < n → ∃ (S : Finset ℤ), S.card = m ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → ¬(1991 ∣ (a + b)) ∧ ¬(1991 ∣ (a - b))) ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1991 ∣ (a + b) ∨ 1991 ∣ (a - b))) ∧
  n = 997 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_1991_l3687_368767


namespace NUMINAMATH_CALUDE_square_sum_inequality_equality_condition_l3687_368748

theorem square_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_square_sum_inequality_equality_condition_l3687_368748


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3687_368732

theorem quadratic_inequality (x : ℝ) : x^2 - 34*x + 225 ≤ 9 ↔ 17 - Real.sqrt 73 ≤ x ∧ x ≤ 17 + Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3687_368732


namespace NUMINAMATH_CALUDE_incorrect_factorization_l3687_368701

theorem incorrect_factorization (x y : ℝ) : ¬(∀ x y : ℝ, x^2 + y^2 = (x + y)^2) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_factorization_l3687_368701


namespace NUMINAMATH_CALUDE_max_guests_l3687_368790

/-- Represents a menu choice as a quadruple of integers -/
structure MenuChoice (n : ℕ) where
  starter : Fin n
  main : Fin n
  dessert : Fin n
  wine : Fin n

/-- The set of all valid menu choices -/
def validMenus (n : ℕ) : Finset (MenuChoice n) :=
  sorry

theorem max_guests (n : ℕ) (h : n > 0) :
  (Finset.card (validMenus n) : ℕ) = n^4 - n^3 ∧
  ∀ (S : Finset (MenuChoice n)), Finset.card S > n^4 - n^3 →
    ∃ (T : Finset (MenuChoice n)), Finset.card T = n ∧ T ⊆ S ∧
      (∃ (i : Fin 4), ∀ (x y : MenuChoice n), x ∈ T → y ∈ T → x ≠ y →
        (i.val = 0 → x.starter = y.starter) ∧
        (i.val = 1 → x.main = y.main) ∧
        (i.val = 2 → x.dessert = y.dessert) ∧
        (i.val = 3 → x.wine = y.wine)) :=
  sorry

end NUMINAMATH_CALUDE_max_guests_l3687_368790


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3687_368765

/-- The number of fruit baskets -/
def num_baskets : ℕ := 4

/-- The number of apples in each of the first three baskets -/
def apples_per_basket : ℕ := 9

/-- The number of oranges in each of the first three baskets -/
def oranges_per_basket : ℕ := 15

/-- The number of bananas in each of the first three baskets -/
def bananas_per_basket : ℕ := 14

/-- The number of fruits that are reduced in the fourth basket -/
def reduction : ℕ := 2

/-- The total number of fruits in all baskets -/
def total_fruits : ℕ := 146

theorem fruit_basket_count :
  (3 * (apples_per_basket + oranges_per_basket + bananas_per_basket)) +
  ((apples_per_basket - reduction) + (oranges_per_basket - reduction) + (bananas_per_basket - reduction)) =
  total_fruits := by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3687_368765


namespace NUMINAMATH_CALUDE_fourth_term_is_six_l3687_368763

/-- An increasing sequence of positive integers satisfying a_{a_n} = 2n + 1 -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m : ℕ, n < m → a n < a m) ∧ 
  (∀ n : ℕ+, a (a n) = 2*n + 1)

/-- The fourth term of the special sequence is 6 -/
theorem fourth_term_is_six (a : ℕ → ℕ) (h : SpecialSequence a) : a 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_six_l3687_368763


namespace NUMINAMATH_CALUDE_bob_water_percentage_approx_36_percent_l3687_368709

/-- Represents a farmer with their crop acreages -/
structure Farmer where
  corn : ℝ
  cotton : ℝ
  beans : ℝ

/-- Calculates the total water usage for a farmer given water rates -/
def waterUsage (f : Farmer) (cornRate : ℝ) (cottonRate : ℝ) : ℝ :=
  f.corn * cornRate + f.cotton * cottonRate + f.beans * (2 * cornRate)

/-- The main theorem to prove -/
theorem bob_water_percentage_approx_36_percent 
  (bob : Farmer) 
  (brenda : Farmer)
  (bernie : Farmer)
  (cornRate : ℝ)
  (cottonRate : ℝ)
  (h_bob : bob = { corn := 3, cotton := 9, beans := 12 })
  (h_brenda : brenda = { corn := 6, cotton := 7, beans := 14 })
  (h_bernie : bernie = { corn := 2, cotton := 12, beans := 0 })
  (h_cornRate : cornRate = 20)
  (h_cottonRate : cottonRate = 80) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |waterUsage bob cornRate cottonRate / 
   (waterUsage bob cornRate cottonRate + 
    waterUsage brenda cornRate cottonRate + 
    waterUsage bernie cornRate cottonRate) - 0.36| < ε := by
  sorry

end NUMINAMATH_CALUDE_bob_water_percentage_approx_36_percent_l3687_368709


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3687_368734

theorem yellow_balls_count (red_balls : ℕ) (yellow_balls : ℕ) 
  (h1 : red_balls = 1) 
  (h2 : (red_balls : ℚ) / (red_balls + yellow_balls) = 1 / 4) : 
  yellow_balls = 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3687_368734


namespace NUMINAMATH_CALUDE_emma_in_middle_l3687_368722

-- Define the friends
inductive Friend
| Allen
| Brian
| Chris
| Diana
| Emma

-- Define the car positions
inductive Position
| First
| Second
| Third
| Fourth
| Fifth

-- Define the seating arrangement
def Arrangement := Friend → Position

-- Define the conditions
def validArrangement (a : Arrangement) : Prop :=
  a Friend.Allen = Position.Second ∧
  a Friend.Diana = Position.First ∧
  (a Friend.Brian = Position.Fourth ∧ a Friend.Chris = Position.Fifth) ∨
  (a Friend.Brian = Position.Third ∧ a Friend.Chris = Position.Fourth) ∧
  (a Friend.Emma = Position.Third ∨ a Friend.Emma = Position.Fifth)

-- Theorem to prove
theorem emma_in_middle (a : Arrangement) :
  validArrangement a → a Friend.Emma = Position.Third :=
sorry

end NUMINAMATH_CALUDE_emma_in_middle_l3687_368722


namespace NUMINAMATH_CALUDE_ratio_evaluation_l3687_368775

theorem ratio_evaluation : (2^3002 * 3^3005) / 6^3003 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l3687_368775


namespace NUMINAMATH_CALUDE_point_N_coordinates_l3687_368718

/-- Given point M(5, -6) and vector a = (1, -2), if vector MN = -3 * vector a,
    then the coordinates of point N are (2, 0). -/
theorem point_N_coordinates :
  let M : ℝ × ℝ := (5, -6)
  let a : ℝ × ℝ := (1, -2)
  let N : ℝ × ℝ := (x, y)
  (x - M.1, y - M.2) = (-3 * a.1, -3 * a.2) →
  N = (2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l3687_368718


namespace NUMINAMATH_CALUDE_budget_allocation_l3687_368793

theorem budget_allocation (home_electronics food_additives gm_microorganisms industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ) :
  home_electronics = 24 →
  food_additives = 15 →
  gm_microorganisms = 19 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 72 →
  let basic_astrophysics := (basic_astrophysics_degrees / 360) * 100
  let total_known := home_electronics + food_additives + gm_microorganisms + industrial_lubricants + basic_astrophysics
  let microphotonics := 100 - total_known
  microphotonics = 14 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_l3687_368793


namespace NUMINAMATH_CALUDE_car_trade_profit_l3687_368795

theorem car_trade_profit (original_price : ℝ) (h : original_price > 0) :
  let buying_price := 0.9 * original_price
  let selling_price := buying_price * 1.8
  let profit := selling_price - original_price
  profit / original_price = 0.62 := by
sorry

end NUMINAMATH_CALUDE_car_trade_profit_l3687_368795


namespace NUMINAMATH_CALUDE_tennis_players_count_l3687_368760

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  neither_players : ℕ
  both_players : ℕ

/-- Calculate the number of tennis players in the sports club -/
def tennis_players (club : SportsClub) : ℕ :=
  club.total_members - club.neither_players - (club.badminton_players - club.both_players)

/-- Theorem stating the number of tennis players in the given club configuration -/
theorem tennis_players_count (club : SportsClub) 
  (h1 : club.total_members = 30)
  (h2 : club.badminton_players = 17)
  (h3 : club.neither_players = 2)
  (h4 : club.both_players = 8) :
  tennis_players club = 19 := by
  sorry

#eval tennis_players ⟨30, 17, 2, 8⟩

end NUMINAMATH_CALUDE_tennis_players_count_l3687_368760


namespace NUMINAMATH_CALUDE_F_4_f_5_equals_69_l3687_368781

-- Define the functions f and F
def f (a : ℝ) : ℝ := 2 * a - 2

def F (a b : ℝ) : ℝ := b^2 + a + 1

-- State the theorem
theorem F_4_f_5_equals_69 : F 4 (f 5) = 69 := by
  sorry

end NUMINAMATH_CALUDE_F_4_f_5_equals_69_l3687_368781


namespace NUMINAMATH_CALUDE_sum_of_constants_l3687_368776

theorem sum_of_constants (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / (x^2 + 1)) →
  (3 = a + b / (1^2 + 1)) →
  (2 = a + b / (0^2 + 1)) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l3687_368776


namespace NUMINAMATH_CALUDE_min_value_expression_l3687_368735

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 + b^2 + 2*a*b + 1 / (a + b)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3687_368735


namespace NUMINAMATH_CALUDE_solve_equation_l3687_368741

-- Define the new operation
def star_op (a b : ℝ) : ℝ := 3 * a - 2 * b^2

-- Theorem statement
theorem solve_equation (a : ℝ) (h : star_op a 4 = 10) : a = 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3687_368741


namespace NUMINAMATH_CALUDE_product_digit_sum_l3687_368783

/-- Converts a base 7 number to base 10 -/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base 7 -/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The product of 35₇ and 12₇ in base 7 -/
def product : ℕ := toBase7 (toBase10 35 * toBase10 12)

theorem product_digit_sum :
  sumOfDigitsBase7 product = 12 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3687_368783


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3687_368712

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - r - 1 = 0 → s^3 - s - 1 = 0 → t^3 - t - 1 = 0 → 
  (1 + r) / (1 - r) + (1 + s) / (1 - s) + (1 + t) / (1 - t) = -7 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3687_368712


namespace NUMINAMATH_CALUDE_baker_problem_l3687_368778

def verify_cake_info (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) : Prop :=
  initial_cakes - sold_cakes = remaining_cakes

def can_determine_initial_pastries (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) 
    (sold_pastries : ℕ) : Prop :=
  false

theorem baker_problem (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) 
    (sold_pastries : ℕ) :
  initial_cakes = 149 →
  sold_cakes = 10 →
  remaining_cakes = 139 →
  sold_pastries = 90 →
  verify_cake_info initial_cakes sold_cakes remaining_cakes ∧
  ¬can_determine_initial_pastries initial_cakes sold_cakes remaining_cakes sold_pastries :=
by
  sorry

end NUMINAMATH_CALUDE_baker_problem_l3687_368778


namespace NUMINAMATH_CALUDE_subtraction_problem_l3687_368755

theorem subtraction_problem : 
  2000000000000 - 1111111111111 - 222222222222 = 666666666667 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3687_368755


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l3687_368769

/-- Given a 5-12-13 right triangle with vertices as centers of mutually externally tangent circles,
    where the radius of the circle at the right angle is half that of the circle opposite the shortest side,
    prove that the sum of the areas of these circles is 105π. -/
theorem sum_of_circle_areas (r s t : ℝ) : 
  r > 0 ∧ s > 0 ∧ t > 0 →  -- radii are positive
  r + s = 13 →  -- sum of radii equals hypotenuse
  s + t = 5 →   -- sum of radii equals one side
  r + t = 12 →  -- sum of radii equals other side
  t = r / 2 →   -- radius at right angle is half of radius opposite shortest side
  π * (r^2 + s^2 + t^2) = 105 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l3687_368769


namespace NUMINAMATH_CALUDE_quadratic_root_product_l3687_368752

theorem quadratic_root_product (b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + b*x + 8
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ * x₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l3687_368752


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3687_368786

def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 3 = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 - q*x - p = 0}

theorem intersection_implies_sum (p q : ℝ) : A p ∩ B p q = {-1} → 2*p + q = -7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3687_368786


namespace NUMINAMATH_CALUDE_no_circular_arrangement_with_conditions_l3687_368747

theorem no_circular_arrangement_with_conditions : ¬ ∃ (a : Fin 9 → ℕ),
  (∀ i, a i ∈ Finset.range 9) ∧
  (∀ i, a i ≠ 0) ∧
  (∀ i j, i ≠ j → a i ≠ a j) ∧
  (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) % 3 = 0) ∧
  (∀ i, a i + a ((i + 1) % 9) + a ((i + 2) % 9) > 12) :=
by sorry

end NUMINAMATH_CALUDE_no_circular_arrangement_with_conditions_l3687_368747


namespace NUMINAMATH_CALUDE_average_equals_expression_l3687_368710

theorem average_equals_expression (x : ℝ) : 
  (1/3) * ((3*x + 8) + (7*x + 3) + (4*x + 9)) = 5*x - 10 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_equals_expression_l3687_368710


namespace NUMINAMATH_CALUDE_integer_double_root_theorem_l3687_368779

/-- A polynomial with integer coefficients of the form x^4 + b_3x^3 + b_2x^2 + b_1x + 48 -/
def IntPolynomial (b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 48

/-- The set of possible integer double roots -/
def PossibleRoots : Set ℤ := {-4, -2, -1, 1, 2, 4}

theorem integer_double_root_theorem (b₃ b₂ b₁ s : ℤ) :
  (∃ k : ℤ, IntPolynomial b₃ b₂ b₁ x = (x - s)^2 * (x^2 + kx + m)) →
  s ∈ PossibleRoots := by
  sorry

end NUMINAMATH_CALUDE_integer_double_root_theorem_l3687_368779


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3687_368733

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + 7

-- Define the condition for the inequality
def inequality_condition (b : ℝ) : Prop :=
  ∀ x : ℝ, f b x < 0 ↔ (x < -2 ∨ x > 3)

-- Theorem statement
theorem quadratic_inequality_solution :
  ∃ b : ℝ, inequality_condition b ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3687_368733


namespace NUMINAMATH_CALUDE_fencing_requirement_l3687_368758

theorem fencing_requirement (length width : ℝ) (h1 : length = 30) (h2 : length * width = 810) :
  2 * width + length = 84 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l3687_368758


namespace NUMINAMATH_CALUDE_function_minimum_condition_l3687_368751

def f (x a : ℝ) := x^2 - 2*a*x + a

theorem function_minimum_condition (a : ℝ) :
  (∃ x, x < 1 ∧ ∀ y < 1, f y a ≥ f x a) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_condition_l3687_368751


namespace NUMINAMATH_CALUDE_christmas_tree_perimeter_l3687_368714

/-- A Christmas tree is a geometric shape with the following properties:
  1. It is symmetric about the y-axis
  2. It has a height of 1
  3. Its branches form a 45° angle with the vertical
  4. It consists of isosceles right triangles
-/
structure ChristmasTree where
  height : ℝ
  branchAngle : ℝ
  isSymmetric : Bool

/-- The perimeter of a Christmas tree is the sum of all its branch lengths -/
def perimeter (tree : ChristmasTree) : ℝ :=
  sorry

/-- The main theorem stating that the perimeter of a Christmas tree
    with the given properties is 2(1 + √2) -/
theorem christmas_tree_perimeter :
  ∀ (tree : ChristmasTree),
  tree.height = 1 ∧ tree.branchAngle = π/4 ∧ tree.isSymmetric = true →
  perimeter tree = 2 * (1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_christmas_tree_perimeter_l3687_368714


namespace NUMINAMATH_CALUDE_selling_price_theorem_l3687_368706

/-- The selling price of an article that results in a loss, given the cost price and a selling price that results in a profit. -/
def selling_price_with_loss (cost_price profit_price : ℕ) : ℕ :=
  2 * cost_price - profit_price

theorem selling_price_theorem (cost_price profit_price : ℕ) 
  (h1 : cost_price = 64)
  (h2 : profit_price = 86)
  (h3 : profit_price > cost_price) :
  selling_price_with_loss cost_price profit_price = 42 := by
  sorry

#eval selling_price_with_loss 64 86  -- Should output 42

end NUMINAMATH_CALUDE_selling_price_theorem_l3687_368706


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3687_368737

theorem pure_imaginary_fraction (a : ℝ) :
  (∃ b : ℝ, (a^2 + Complex.I) / (1 - Complex.I) = Complex.I * b) →
  a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3687_368737


namespace NUMINAMATH_CALUDE_increasing_quadratic_parameter_range_l3687_368788

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem increasing_quadratic_parameter_range (a : ℝ) :
  (∀ x ≥ 2, ∀ y ≥ 2, x < y → f a x < f a y) →
  a ∈ Set.Ici (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_increasing_quadratic_parameter_range_l3687_368788


namespace NUMINAMATH_CALUDE_blake_bucket_water_l3687_368777

theorem blake_bucket_water (poured_out water_left : ℝ) 
  (h1 : poured_out = 0.2)
  (h2 : water_left = 0.6) :
  poured_out + water_left = 0.8 := by
sorry

end NUMINAMATH_CALUDE_blake_bucket_water_l3687_368777


namespace NUMINAMATH_CALUDE_juan_speed_l3687_368791

/-- Given a distance of 80 miles and a time of 8 hours, prove that the speed is 10 miles per hour. -/
theorem juan_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_juan_speed_l3687_368791


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l3687_368798

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 30 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = ((n1 + n2) : ℚ) * (48.75 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l3687_368798


namespace NUMINAMATH_CALUDE_P_n_has_n_distinct_real_roots_P_2018_has_2018_distinct_real_roots_l3687_368719

-- Define the sequence of polynomials
def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | 1 => λ x => x
  | (n + 2) => λ x => x * P (n + 1) x - P n x

-- Define a function to count distinct real roots
noncomputable def count_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

-- State the theorem
theorem P_n_has_n_distinct_real_roots (n : ℕ) :
  count_distinct_real_roots (P n) = n := by sorry

-- The specific case for P₂₀₁₈
theorem P_2018_has_2018_distinct_real_roots :
  count_distinct_real_roots (P 2018) = 2018 := by sorry

end NUMINAMATH_CALUDE_P_n_has_n_distinct_real_roots_P_2018_has_2018_distinct_real_roots_l3687_368719


namespace NUMINAMATH_CALUDE_pizza_cost_equality_l3687_368782

theorem pizza_cost_equality (total_cost : ℚ) (num_slices : ℕ) 
  (h1 : total_cost = 13)
  (h2 : num_slices = 10) :
  let cost_per_slice := total_cost / num_slices
  5 * cost_per_slice = 5 * cost_per_slice := by
sorry

end NUMINAMATH_CALUDE_pizza_cost_equality_l3687_368782


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3687_368716

theorem complex_expression_simplification (x y : ℝ) : 
  (2 * x + 3 * Complex.I * y) * (2 * x - 3 * Complex.I * y) + 2 * x = 4 * x^2 + 2 * x - 9 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3687_368716


namespace NUMINAMATH_CALUDE_no_number_divisible_by_1998_with_small_digit_sum_l3687_368799

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem no_number_divisible_by_1998_with_small_digit_sum :
  ∀ n : ℕ, n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end NUMINAMATH_CALUDE_no_number_divisible_by_1998_with_small_digit_sum_l3687_368799


namespace NUMINAMATH_CALUDE_solution_set_inequality_proof_l3687_368708

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := |x - 2|

-- Part 1: Prove that the solution set of f(x) + f(x+1) ≤ 2 is [0.5, 2.5]
theorem solution_set (x : ℝ) : 
  (f x + f (x + 1) ≤ 2) ↔ (0.5 ≤ x ∧ x ≤ 2.5) := by sorry

-- Part 2: Prove that for all a < 0 and all x, f(ax) - af(x) ≥ f(2a)
theorem inequality_proof (a x : ℝ) (h : a < 0) : 
  f (a * x) - a * f x ≥ f (2 * a) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_proof_l3687_368708


namespace NUMINAMATH_CALUDE_product_neg_seventeen_sum_l3687_368738

theorem product_neg_seventeen_sum (a b c : ℤ) : 
  a * b * c = -17 → (a + b + c = -17 ∨ a + b + c = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_product_neg_seventeen_sum_l3687_368738


namespace NUMINAMATH_CALUDE_ball_count_l3687_368785

theorem ball_count (num_red : ℕ) (prob_red : ℚ) (total : ℕ) : 
  num_red = 4 → prob_red = 1/3 → total = num_red / prob_red → total = 12 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_l3687_368785


namespace NUMINAMATH_CALUDE_expression_simplification_l3687_368761

theorem expression_simplification (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3687_368761


namespace NUMINAMATH_CALUDE_multiplication_mistake_l3687_368750

theorem multiplication_mistake (x : ℚ) : (43 * x - 34 * x = 1251) → x = 139 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l3687_368750


namespace NUMINAMATH_CALUDE_cone_curved_surface_area_l3687_368742

/-- The curved surface area of a cone with given slant height and base radius -/
theorem cone_curved_surface_area 
  (slant_height : ℝ) 
  (base_radius : ℝ) 
  (h1 : slant_height = 10) 
  (h2 : base_radius = 5) : 
  π * base_radius * slant_height = 50 * π := by
sorry

end NUMINAMATH_CALUDE_cone_curved_surface_area_l3687_368742


namespace NUMINAMATH_CALUDE_lucy_groceries_l3687_368720

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The total number of packs of groceries Lucy bought -/
def total_groceries : ℕ := cookies + noodles

theorem lucy_groceries : total_groceries = 28 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l3687_368720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3687_368770

theorem arithmetic_sequence_length 
  (a₁ : ℤ) 
  (d : ℤ) 
  (aₙ : ℤ) 
  (h1 : a₁ = 2) 
  (h2 : d = 3) 
  (h3 : aₙ = 110) :
  ∃ n : ℕ, n = 37 ∧ aₙ = a₁ + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3687_368770


namespace NUMINAMATH_CALUDE_B_equals_zero_one_l3687_368739

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_zero_one : B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_B_equals_zero_one_l3687_368739


namespace NUMINAMATH_CALUDE_cubic_factorization_l3687_368727

theorem cubic_factorization (a b c d e : ℝ) :
  (∀ x, 216 * x^3 - 27 = (a * x - b) * (c * x^2 + d * x - e)) →
  a + b + c + d + e = 72 := by
sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3687_368727


namespace NUMINAMATH_CALUDE_first_car_mpg_l3687_368746

/-- Proves that the average miles per gallon of the first car is 27.5 given the conditions --/
theorem first_car_mpg (total_miles : ℝ) (total_gallons : ℝ) (second_car_mpg : ℝ) (first_car_gallons : ℝ) :
  total_miles = 1825 →
  total_gallons = 55 →
  second_car_mpg = 40 →
  first_car_gallons = 30 →
  (first_car_gallons * (total_miles - second_car_mpg * (total_gallons - first_car_gallons))) / 
    (first_car_gallons * total_miles) = 27.5 := by
sorry

end NUMINAMATH_CALUDE_first_car_mpg_l3687_368746


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l3687_368703

theorem quadratic_form_equivalence (x : ℝ) : x^2 + 4*x + 1 = (x + 2)^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l3687_368703


namespace NUMINAMATH_CALUDE_rectangle_width_l3687_368713

theorem rectangle_width (square_perimeter : ℝ) (rectangle_length : ℝ) (rectangle_width : ℝ) : 
  square_perimeter = 160 →
  rectangle_length = 32 →
  (square_perimeter / 4) ^ 2 = 5 * (rectangle_length * rectangle_width) →
  rectangle_width = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l3687_368713


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3687_368705

theorem negation_of_universal_proposition :
  (¬∀ x : ℝ, x ≥ 1 → Real.log x > 0) ↔ (∃ x : ℝ, x ≥ 1 ∧ Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3687_368705


namespace NUMINAMATH_CALUDE_thirteen_seventh_mod_nine_l3687_368704

theorem thirteen_seventh_mod_nine (n : ℕ) : 
  13^7 % 9 = n ∧ 0 ≤ n ∧ n < 9 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_seventh_mod_nine_l3687_368704


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3687_368771

theorem biased_coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1/2) :
  (Nat.choose 6 2 : ℝ) * p^2 * (1 - p)^4 = 1/8 → p = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3687_368771


namespace NUMINAMATH_CALUDE_third_candidate_votes_l3687_368753

theorem third_candidate_votes : 
  ∀ (total_votes : ℕ) (invalid_percentage : ℚ) (first_candidate_percentage : ℚ) (second_candidate_percentage : ℚ),
  total_votes = 10000 →
  invalid_percentage = 1/4 →
  first_candidate_percentage = 1/2 →
  second_candidate_percentage = 3/10 →
  ∃ (third_candidate_votes : ℕ),
    third_candidate_votes = total_votes * (1 - invalid_percentage) - 
      (total_votes * (1 - invalid_percentage) * first_candidate_percentage + 
       total_votes * (1 - invalid_percentage) * second_candidate_percentage) ∧
    third_candidate_votes = 1500 :=
by sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l3687_368753


namespace NUMINAMATH_CALUDE_reflect_A_across_x_axis_l3687_368773

def reflect_point_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflect_A_across_x_axis :
  let A : ℝ × ℝ := (-5, -2)
  reflect_point_x_axis A = (-5, 2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_A_across_x_axis_l3687_368773


namespace NUMINAMATH_CALUDE_least_number_of_cubes_l3687_368766

/-- Represents the dimensions of a cuboidal block in centimeters -/
structure CuboidalBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the side length of a cube in centimeters -/
def CubeSideLength : ℕ := 3

/-- The given cuboidal block -/
def given_block : CuboidalBlock := ⟨18, 27, 36⟩

/-- The volume of a cuboidal block -/
def volume_cuboid (b : CuboidalBlock) : ℕ := b.length * b.width * b.height

/-- The volume of a cube -/
def volume_cube (side : ℕ) : ℕ := side * side * side

/-- The number of cubes that can be cut from a cuboidal block -/
def number_of_cubes (b : CuboidalBlock) (side : ℕ) : ℕ :=
  volume_cuboid b / volume_cube side

/-- Theorem: The least possible number of equal cubes with side lengths in a fixed ratio of 1:2:3
    that can be cut from the given cuboidal block is 648 -/
theorem least_number_of_cubes :
  number_of_cubes given_block CubeSideLength = 648 := by sorry

end NUMINAMATH_CALUDE_least_number_of_cubes_l3687_368766


namespace NUMINAMATH_CALUDE_fifteen_percent_of_x_is_ninety_l3687_368774

theorem fifteen_percent_of_x_is_ninety (x : ℝ) : (15 / 100) * x = 90 → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_x_is_ninety_l3687_368774


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3687_368724

/-- An arithmetic sequence with given first term and 17th term -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 2
  term_17 : a 17 = 66

/-- The general formula for the nth term of the sequence -/
def general_formula (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  4 * n - 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = general_formula seq n) ∧
  ¬ ∃ n, seq.a n = 88 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3687_368724


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l3687_368736

/-- The cost price of a computer table given its selling price and markup percentage. -/
def cost_price (selling_price : ℚ) (markup_percent : ℚ) : ℚ :=
  selling_price / (1 + markup_percent / 100)

/-- Theorem stating that the cost price of a computer table is 6525 
    given a selling price of 8091 and a markup of 24%. -/
theorem computer_table_cost_price :
  cost_price 8091 24 = 6525 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l3687_368736


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3687_368745

/-- The cubic function f(x) with specific properties -/
def f (x : ℝ) : ℝ := 2*x^3 - 9*x^2 + 12*x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6*x^2 - 18*x + 12

theorem cubic_function_properties :
  (f 0 = -4) ∧ 
  (∀ x, f' 0 * x - (f x - f 0) - 4 = 0) ∧
  (f 2 = 0) ∧ 
  (f' 2 = 0) ∧
  (∀ x, x < 1 ∨ x > 2 → f' x > 0) :=
sorry


end NUMINAMATH_CALUDE_cubic_function_properties_l3687_368745


namespace NUMINAMATH_CALUDE_orange_packing_problem_l3687_368762

/-- Given a total number of oranges and the capacity of each box, 
    calculate the number of boxes needed. -/
def boxes_needed (total_oranges : ℕ) (oranges_per_box : ℕ) : ℕ :=
  total_oranges / oranges_per_box

/-- Theorem stating that 265 boxes are needed to pack 2650 oranges 
    when each box holds 10 oranges. -/
theorem orange_packing_problem : 
  boxes_needed 2650 10 = 265 := by
  sorry

end NUMINAMATH_CALUDE_orange_packing_problem_l3687_368762


namespace NUMINAMATH_CALUDE_max_a_satisfies_equation_no_larger_a_satisfies_equation_max_a_is_maximum_l3687_368721

/-- The coefficient of x^4 in the expansion of (1-3x+ax^2)^8 --/
def coefficient_x4 (a : ℝ) : ℝ := 28 * a^2 + 2016 * a + 5670

/-- The equation that a must satisfy --/
def equation (a : ℝ) : Prop := coefficient_x4 a = 70

/-- The maximum value of a that satisfies the equation --/
noncomputable def max_a : ℝ := -36 + Real.sqrt 1096

theorem max_a_satisfies_equation : equation max_a :=
sorry

theorem no_larger_a_satisfies_equation :
  ∀ a : ℝ, a > max_a → ¬(equation a) :=
sorry

theorem max_a_is_maximum :
  ∃ (ε : ℝ), ε > 0 ∧ (∀ δ : ℝ, 0 < δ ∧ δ < ε → ¬(equation (max_a + δ))) :=
sorry

end NUMINAMATH_CALUDE_max_a_satisfies_equation_no_larger_a_satisfies_equation_max_a_is_maximum_l3687_368721


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3687_368717

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -1 then Int.ceil (1 / (x + 1))
  else if x < -1 then Int.floor (1 / (x + 1))
  else 0  -- This value doesn't matter as x = -1 is not in the domain

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -1 → g x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3687_368717


namespace NUMINAMATH_CALUDE_tetrahedron_edge_assignment_exists_l3687_368780

/-- Represents a tetrahedron with face areas -/
structure Tetrahedron where
  s : ℝ  -- smallest face area
  S : ℝ  -- largest face area
  a : ℝ  -- another face area
  b : ℝ  -- another face area
  h_s_smallest : s ≤ S ∧ s ≤ a ∧ s ≤ b
  h_S_largest : S ≥ s ∧ S ≥ a ∧ S ≥ b
  h_positive : s > 0 ∧ S > 0 ∧ a > 0 ∧ b > 0

/-- Represents the assignment of numbers to the edges of a tetrahedron -/
structure EdgeAssignment (t : Tetrahedron) where
  e1 : ℝ  -- edge common to smallest and largest face
  e2 : ℝ  -- edge of smallest face
  e3 : ℝ  -- edge of smallest face
  e4 : ℝ  -- edge of largest face
  e5 : ℝ  -- edge of largest face
  e6 : ℝ  -- remaining edge
  h_non_negative : e1 ≥ 0 ∧ e2 ≥ 0 ∧ e3 ≥ 0 ∧ e4 ≥ 0 ∧ e5 ≥ 0 ∧ e6 ≥ 0

/-- The theorem stating that a valid edge assignment exists for any tetrahedron -/
theorem tetrahedron_edge_assignment_exists (t : Tetrahedron) :
  ∃ (ea : EdgeAssignment t),
    ea.e1 + ea.e2 + ea.e3 = t.s ∧
    ea.e1 + ea.e4 + ea.e5 = t.S ∧
    ea.e2 + ea.e5 + ea.e6 = t.a ∧
    ea.e3 + ea.e4 + ea.e6 = t.b :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_assignment_exists_l3687_368780


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3687_368784

theorem quadratic_equation_roots (x : ℝ) :
  x^2 - 4*x - 2 = 0 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3687_368784
