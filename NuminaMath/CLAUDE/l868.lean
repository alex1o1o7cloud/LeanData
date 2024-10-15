import Mathlib

namespace NUMINAMATH_CALUDE_cafe_benches_theorem_l868_86801

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 5^2 + tens * 5^1 + ones * 5^0

/-- Calculates the number of benches needed given a total number of people and people per bench -/
def benchesNeeded (totalPeople : Nat) (peoplePerBench : Nat) : Nat :=
  (totalPeople + peoplePerBench - 1) / peoplePerBench

theorem cafe_benches_theorem (cafeCapacity : Nat) (peoplePerBench : Nat) :
  cafeCapacity = 310 ∧ peoplePerBench = 3 →
  benchesNeeded (base5ToBase10 cafeCapacity) peoplePerBench = 27 := by
  sorry

#eval benchesNeeded (base5ToBase10 310) 3

end NUMINAMATH_CALUDE_cafe_benches_theorem_l868_86801


namespace NUMINAMATH_CALUDE_part_one_part_two_l868_86824

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(m+1)*x + m^2 - 5 = 0}

-- Theorem for part (1)
theorem part_one (a : ℝ) : A ∪ B a = A → a = 2 ∨ a = 3 := by sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) : A ∩ C m = C m → m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l868_86824


namespace NUMINAMATH_CALUDE_intersection_line_equation_l868_86816

-- Define the circle (x-1)^2 + y^2 = 1
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- Define point C (center of circle1)
def C : ℝ × ℝ := (1, 0)

-- Define the circle with diameter PC
def circle2 (x y : ℝ) : Prop := (x - (P.1 + C.1)/2)^2 + (y - (P.2 + C.2)/2)^2 = ((P.1 - C.1)^2 + (P.2 - C.2)^2) / 4

-- Theorem statement
theorem intersection_line_equation : 
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → x + 3*y - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l868_86816


namespace NUMINAMATH_CALUDE_routes_2x2_grid_proof_l868_86870

/-- The number of routes on a 2x2 grid from top-left to bottom-right -/
def routes_2x2_grid : ℕ := 6

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem routes_2x2_grid_proof :
  routes_2x2_grid = choose 4 2 :=
by sorry

end NUMINAMATH_CALUDE_routes_2x2_grid_proof_l868_86870


namespace NUMINAMATH_CALUDE_account_balance_first_year_l868_86821

/-- Proves that the account balance at the end of the first year is correct -/
theorem account_balance_first_year 
  (initial_deposit : ℝ) 
  (interest_first_year : ℝ) 
  (balance_first_year : ℝ) :
  initial_deposit = 1000 →
  interest_first_year = 100 →
  balance_first_year = initial_deposit + interest_first_year →
  balance_first_year = 1100 := by
  sorry

#check account_balance_first_year

end NUMINAMATH_CALUDE_account_balance_first_year_l868_86821


namespace NUMINAMATH_CALUDE_paving_cost_l868_86830

/-- The cost of paving a rectangular floor given its dimensions and rate per square meter. -/
theorem paving_cost (length width rate : ℝ) : length = 5.5 → width = 4 → rate = 700 → length * width * rate = 15400 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l868_86830


namespace NUMINAMATH_CALUDE_currant_yield_increase_l868_86815

theorem currant_yield_increase (initial_yield_per_bush : ℝ) : 
  let total_yield := 15 * initial_yield_per_bush
  let new_yield_per_bush := total_yield / 12
  (new_yield_per_bush - initial_yield_per_bush) / initial_yield_per_bush * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_currant_yield_increase_l868_86815


namespace NUMINAMATH_CALUDE_inequality_proof_l868_86894

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : 1/x + 1/y + 1/z = 1) : 
  Real.sqrt (x + y*z) + Real.sqrt (y + z*x) + Real.sqrt (z + x*y) ≥ 
  Real.sqrt (x*y*z) + Real.sqrt x + Real.sqrt y + Real.sqrt z := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l868_86894


namespace NUMINAMATH_CALUDE_election_win_margin_l868_86898

theorem election_win_margin 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (winner_votes : ℕ) 
  (h1 : winner_percentage = 62 / 100)
  (h2 : winner_votes = 744)
  (h3 : ↑winner_votes = winner_percentage * ↑total_votes) :
  winner_votes - (total_votes - winner_votes) = 288 :=
by sorry

end NUMINAMATH_CALUDE_election_win_margin_l868_86898


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l868_86851

theorem unique_solution_power_equation :
  ∃! (x y m n : ℕ), x > y ∧ y > 0 ∧ m > 1 ∧ n > 1 ∧ (x + y)^n = x^m + y^m :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l868_86851


namespace NUMINAMATH_CALUDE_four_points_cyclic_l868_86806

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the necessary geometric relations
variable (collinear : Point → Point → Point → Prop)
variable (orthocenter : Point → Point → Point → Point)
variable (lies_on : Point → Line → Prop)
variable (concurrent : Line → Line → Line → Prop)
variable (cyclic : Point → Point → Point → Point → Prop)

-- Define the theorem
theorem four_points_cyclic
  (A B C D P Q R : Point)
  (AP BQ CR : Line)
  (h1 : ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D)
  (h2 : orthocenter B C D ≠ D)
  (h3 : P = orthocenter B C D)
  (h4 : Q = orthocenter C A D)
  (h5 : R = orthocenter A B D)
  (h6 : lies_on A AP ∧ lies_on P AP)
  (h7 : lies_on B BQ ∧ lies_on Q BQ)
  (h8 : lies_on C CR ∧ lies_on R CR)
  (h9 : AP ≠ BQ ∧ BQ ≠ CR ∧ CR ≠ AP)
  (h10 : concurrent AP BQ CR)
  : cyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_four_points_cyclic_l868_86806


namespace NUMINAMATH_CALUDE_celebration_day_l868_86897

/-- Given a person born on a Friday, their 1200th day of life will fall on a Saturday -/
theorem celebration_day (birth_day : Nat) (birth_weekday : Nat) : 
  birth_weekday = 5 → (birth_day + 1199) % 7 = 6 := by
  sorry

#check celebration_day

end NUMINAMATH_CALUDE_celebration_day_l868_86897


namespace NUMINAMATH_CALUDE_population_growth_rate_l868_86846

/-- Proves that given a population of 10,000 that grows to 12,100 in 2 years
    with a constant annual growth rate, the annual percentage increase is 10%. -/
theorem population_growth_rate (initial_population : ℕ) (final_population : ℕ) 
  (years : ℕ) (growth_rate : ℝ) :
  initial_population = 10000 →
  final_population = 12100 →
  years = 2 →
  final_population = initial_population * (1 + growth_rate) ^ years →
  growth_rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l868_86846


namespace NUMINAMATH_CALUDE_faye_money_left_l868_86850

def initial_money : ℕ := 20
def mother_multiplier : ℕ := 2
def cupcake_price : ℚ := 3/2
def cupcake_quantity : ℕ := 10
def cookie_box_price : ℕ := 3
def cookie_box_quantity : ℕ := 5

theorem faye_money_left :
  let total_money := initial_money + mother_multiplier * initial_money
  let spent_money := cupcake_price * cupcake_quantity + cookie_box_price * cookie_box_quantity
  total_money - spent_money = 30 := by
sorry

end NUMINAMATH_CALUDE_faye_money_left_l868_86850


namespace NUMINAMATH_CALUDE_average_multiples_of_10_l868_86845

/-- The average of multiples of 10 from 10 to 500 inclusive is 255 -/
theorem average_multiples_of_10 : 
  let first := 10
  let last := 500
  let step := 10
  (first + last) / 2 = 255 := by sorry

end NUMINAMATH_CALUDE_average_multiples_of_10_l868_86845


namespace NUMINAMATH_CALUDE_slope_of_line_l868_86839

theorem slope_of_line (x y : ℝ) : 4 * y = -6 * x + 12 → (y - 3) = (-3/2) * (x - 0) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l868_86839


namespace NUMINAMATH_CALUDE_no_rational_roots_l868_86817

theorem no_rational_roots (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ∀ (x : ℚ), x^2 + 2*p*x + 2*q ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l868_86817


namespace NUMINAMATH_CALUDE_range_of_a_l868_86874

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l868_86874


namespace NUMINAMATH_CALUDE_tangent_line_and_minimum_value_l868_86858

noncomputable section

-- Define the function f
def f (x a b : ℝ) : ℝ := Real.exp x * (x^2 - (a + 2) * x + b)

-- Define the derivative of f
def f' (x a b : ℝ) : ℝ := Real.exp x * (x^2 - a * x + b - (a + 2))

theorem tangent_line_and_minimum_value (a b : ℝ) :
  (f' 0 a b = -2 * a^2) →
  (b = a + 2 - 2 * a^2) ∧
  (∀ a < 0, ∃ M ≥ 2, ∀ x > 0, f x a b < M) :=
by sorry

end

end NUMINAMATH_CALUDE_tangent_line_and_minimum_value_l868_86858


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l868_86813

/-- A triangle with side lengths a, b, and c is isosceles if at least two of its sides are equal -/
def IsIsosceles (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c

/-- The perimeter of a triangle with side lengths a, b, and c -/
def Perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  IsIsosceles a b c →
  (a = 5 ∧ b = 8) ∨ (a = 8 ∧ b = 5) ∨ (b = 5 ∧ c = 8) ∨ (b = 8 ∧ c = 5) ∨ (a = 5 ∧ c = 8) ∨ (a = 8 ∧ c = 5) →
  Perimeter a b c = 18 ∨ Perimeter a b c = 21 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l868_86813


namespace NUMINAMATH_CALUDE_statement_1_statement_4_main_theorem_l868_86869

-- Statement ①
theorem statement_1 : ∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1) := by sorry

-- Statement ④
theorem statement_4 : (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) := by sorry

-- Main theorem combining both statements
theorem main_theorem : (∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1)) ∧
                       ((¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1)) := by sorry

end NUMINAMATH_CALUDE_statement_1_statement_4_main_theorem_l868_86869


namespace NUMINAMATH_CALUDE_thursday_miles_proof_l868_86828

/-- The number of miles flown on Tuesday each week -/
def tuesday_miles : ℕ := 1134

/-- The total number of miles flown over 3 weeks -/
def total_miles : ℕ := 7827

/-- The number of weeks the pilot flies -/
def num_weeks : ℕ := 3

/-- The number of miles flown on Thursday each week -/
def thursday_miles : ℕ := (total_miles - num_weeks * tuesday_miles) / num_weeks

theorem thursday_miles_proof :
  thursday_miles = 1475 :=
by sorry

end NUMINAMATH_CALUDE_thursday_miles_proof_l868_86828


namespace NUMINAMATH_CALUDE_exists_player_reaching_all_l868_86886

/-- Represents a tournament where every player plays against every other player once with no draws -/
structure Tournament (α : Type) :=
  (players : Set α)
  (defeated : α → α → Prop)
  (complete : ∀ a b : α, a ≠ b → (defeated a b ∨ defeated b a))
  (irreflexive : ∀ a : α, ¬ defeated a a)

/-- A player can reach another player within two steps of the defeated relation -/
def can_reach_in_two_steps {α : Type} (t : Tournament α) (a b : α) : Prop :=
  t.defeated a b ∨ ∃ c, t.defeated a c ∧ t.defeated c b

/-- The main theorem: there exists a player who can reach all others within two steps -/
theorem exists_player_reaching_all {α : Type} (t : Tournament α) :
  ∃ a : α, ∀ b : α, b ∈ t.players → a ≠ b → can_reach_in_two_steps t a b :=
sorry

end NUMINAMATH_CALUDE_exists_player_reaching_all_l868_86886


namespace NUMINAMATH_CALUDE_tangent_line_curve_range_l868_86840

theorem tangent_line_curve_range (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, y = x - a ∧ y = Real.log (x + b) ∧ 
    (∀ x' y' : ℝ, y' = x' - a → y' ≤ Real.log (x' + b))) →
  (∀ z : ℝ, z ∈ Set.Ioo 0 (1/2) ↔ ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 
    (∃ x y : ℝ, y = x - a' ∧ y = Real.log (x + b') ∧ 
      (∀ x' y' : ℝ, y' = x' - a' → y' ≤ Real.log (x' + b'))) ∧
    z = a'^2 / (2 + b')) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_curve_range_l868_86840


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l868_86862

theorem inequality_and_equality_conditions (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  ((1 + a)^2 / (1 + b) ≤ 1 + a^2 / b ↔ b < -1 ∨ b > 0) ∧
  ((1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b ∧ b ≠ -1 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l868_86862


namespace NUMINAMATH_CALUDE_expected_pine_saplings_in_sample_l868_86867

/-- Given a forestry farm with the following characteristics:
  * total_saplings: The total number of saplings
  * pine_saplings: The number of pine saplings
  * sample_size: The size of the sample to be drawn
  
  This theorem proves that the expected number of pine saplings in the sample
  is equal to (pine_saplings / total_saplings) * sample_size. -/
theorem expected_pine_saplings_in_sample
  (total_saplings : ℕ)
  (pine_saplings : ℕ)
  (sample_size : ℕ)
  (h1 : total_saplings = 3000)
  (h2 : pine_saplings = 400)
  (h3 : sample_size = 150)
  : (pine_saplings : ℚ) / total_saplings * sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_pine_saplings_in_sample_l868_86867


namespace NUMINAMATH_CALUDE_percentage_of_360_is_120_l868_86842

theorem percentage_of_360_is_120 : 
  (120 : ℝ) / 360 * 100 = 100 / 3 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_360_is_120_l868_86842


namespace NUMINAMATH_CALUDE_max_ratio_squared_l868_86800

theorem max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b →
  0 ≤ x → x < a →
  0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 →
  b^2 + x^2 = (a - x)^2 + (b - y)^2 →
  a^2 + b^2 = x^2 + b^2 →
  (∀ a' b' : ℝ, 0 < a' → 0 < b' → a' ≥ b' → (a' / b')^2 ≤ 4/3) ∧
  (∃ a' b' : ℝ, 0 < a' → 0 < b' → a' ≥ b' → (a' / b')^2 = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l868_86800


namespace NUMINAMATH_CALUDE_lcm_gcd_product_15_45_l868_86836

theorem lcm_gcd_product_15_45 : Nat.lcm 15 45 * Nat.gcd 15 45 = 675 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_15_45_l868_86836


namespace NUMINAMATH_CALUDE_intersection_distance_product_l868_86868

/-- Given a line passing through (0, 1) that intersects y = x^2 at A and B,
    the product of the absolute values of x-coordinates of A and B is 1 -/
theorem intersection_distance_product (k : ℝ) : 
  let line := fun x => k * x + 1
  let parabola := fun x => x^2
  let roots := {x : ℝ | parabola x = line x}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ a ≠ b ∧ |a| * |b| = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l868_86868


namespace NUMINAMATH_CALUDE_zero_points_inequality_l868_86890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - (a / 2) * Real.log x

theorem zero_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x > 0 → f a x = 0 → x = x₁ ∨ x = x₂) →
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  1 < x₁ ∧ x₁ < a ∧ a < x₂ ∧ x₂ < a^2 :=
by sorry

end NUMINAMATH_CALUDE_zero_points_inequality_l868_86890


namespace NUMINAMATH_CALUDE_count_fractions_is_36_l868_86888

/-- A function that counts the number of fractions less than 1 with single-digit numerators and denominators -/
def count_fractions : ℕ := 
  let single_digit (n : ℕ) := n ≥ 1 ∧ n ≤ 9
  let is_valid_fraction (n d : ℕ) := single_digit n ∧ single_digit d ∧ n < d
  (Finset.sum (Finset.range 9) (λ d => 
    (Finset.filter (λ n => is_valid_fraction n (d + 1)) (Finset.range (d + 1))).card
  ))

/-- Theorem stating that the count of fractions less than 1 with single-digit numerators and denominators is 36 -/
theorem count_fractions_is_36 : count_fractions = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_fractions_is_36_l868_86888


namespace NUMINAMATH_CALUDE_range_of_a_l868_86866

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a * x^2 - x + (1/4) * a)
def q (a : ℝ) : Prop := ∀ x > 0, 3^x - 9^x < a

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → 0 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l868_86866


namespace NUMINAMATH_CALUDE_prove_a_value_l868_86873

-- Define the operation for integers
def star_op (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- Theorem statement
theorem prove_a_value (h : star_op 21 9 = 160) : 21 = 21 := by
  sorry

end NUMINAMATH_CALUDE_prove_a_value_l868_86873


namespace NUMINAMATH_CALUDE_train_crossing_time_l868_86875

/-- Time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 100 →
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * 1000 / 3600)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l868_86875


namespace NUMINAMATH_CALUDE_puppies_adopted_per_day_l868_86892

theorem puppies_adopted_per_day :
  ∀ (initial_puppies additional_puppies total_days : ℕ),
    initial_puppies = 3 →
    additional_puppies = 3 →
    total_days = 2 →
    (initial_puppies + additional_puppies) / total_days = 3 :=
by
  sorry

#check puppies_adopted_per_day

end NUMINAMATH_CALUDE_puppies_adopted_per_day_l868_86892


namespace NUMINAMATH_CALUDE_right_triangle_area_l868_86827

theorem right_triangle_area (h : ℝ) (angle : ℝ) : 
  h = 13 → angle = 45 → 
  let area := (1/2) * (h / Real.sqrt 2) * (h / Real.sqrt 2)
  area = 84.5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l868_86827


namespace NUMINAMATH_CALUDE_average_sale_proof_l868_86899

def sales : List ℝ := [5266, 5768, 5922, 5678, 6029]
def required_sale : ℝ := 4937

theorem average_sale_proof :
  (sales.sum + required_sale) / 6 = 5600 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_proof_l868_86899


namespace NUMINAMATH_CALUDE_min_bailing_rate_solution_l868_86853

/-- Represents the problem of determining the minimum bailing rate for a boat --/
def MinBailingRateProblem (distance_to_shore : ℝ) (rowing_speed : ℝ) (water_intake_rate : ℝ) (max_water_capacity : ℝ) : Prop :=
  let time_to_shore : ℝ := distance_to_shore / rowing_speed
  let total_water_intake : ℝ := water_intake_rate * time_to_shore
  let excess_water : ℝ := total_water_intake - max_water_capacity
  let min_bailing_rate : ℝ := excess_water / time_to_shore
  min_bailing_rate = 2

/-- The theorem stating the minimum bailing rate for the given problem --/
theorem min_bailing_rate_solution :
  MinBailingRateProblem 0.5 6 12 50 := by
  sorry


end NUMINAMATH_CALUDE_min_bailing_rate_solution_l868_86853


namespace NUMINAMATH_CALUDE_population_after_five_years_l868_86829

/-- Represents the yearly change in organization population -/
def yearly_change (b : ℝ) : ℝ := 2.7 * b - 8.5

/-- Calculates the population after n years -/
def population_after_years (initial_population : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => initial_population
  | n + 1 => yearly_change (population_after_years initial_population n)

/-- Theorem stating the population after 5 years -/
theorem population_after_five_years :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |population_after_years 25 5 - 2875| < ε :=
sorry

end NUMINAMATH_CALUDE_population_after_five_years_l868_86829


namespace NUMINAMATH_CALUDE_winnie_the_pooh_honey_l868_86805

theorem winnie_the_pooh_honey (a b c d e : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0) 
  (total : a + b + c + d + e = 3) : 
  max (a + b) (max (b + c) (max (c + d) (d + e))) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_winnie_the_pooh_honey_l868_86805


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l868_86848

theorem min_x_prime_factorization (x y : ℕ+) (h : 13 * x^4 = 29 * y^12) :
  ∃ (a b c d : ℕ), 
    (x = (29^3 : ℕ+) * (13^3 : ℕ+)) ∧
    (∀ z : ℕ+, 13 * z^4 = 29 * y^12 → x ≤ z) ∧
    (Nat.Prime a ∧ Nat.Prime b) ∧
    (x = a^c * b^d) ∧
    (a + b + c + d = 48) := by
  sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l868_86848


namespace NUMINAMATH_CALUDE_real_roots_of_p_l868_86838

def p (x : ℝ) : ℝ := x^4 - 3*x^3 + 3*x^2 - x - 6

theorem real_roots_of_p :
  ∃ (a b c d : ℝ), (∀ x, p x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
                   (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 1) := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_p_l868_86838


namespace NUMINAMATH_CALUDE_kelly_games_left_l868_86885

/-- Calculates the number of games left after finding more and giving some away -/
def games_left (initial : ℕ) (found : ℕ) (given_away : ℕ) : ℕ :=
  initial + found - given_away

/-- Proves that Kelly will have 6 games left -/
theorem kelly_games_left : games_left 80 31 105 = 6 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_left_l868_86885


namespace NUMINAMATH_CALUDE_letters_with_both_dot_and_line_l868_86822

/-- Represents the number of letters in the alphabet -/
def total_letters : ℕ := 40

/-- Represents the number of letters with only a straight line -/
def straight_line_only : ℕ := 24

/-- Represents the number of letters with only a dot -/
def dot_only : ℕ := 7

/-- Represents the number of letters with both a dot and a straight line -/
def both : ℕ := total_letters - straight_line_only - dot_only

theorem letters_with_both_dot_and_line :
  both = 9 :=
sorry

end NUMINAMATH_CALUDE_letters_with_both_dot_and_line_l868_86822


namespace NUMINAMATH_CALUDE_cos_negative_eleven_fourths_pi_l868_86889

theorem cos_negative_eleven_fourths_pi :
  Real.cos (-11/4 * Real.pi) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_eleven_fourths_pi_l868_86889


namespace NUMINAMATH_CALUDE_circle_equation_l868_86893

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := 3*x + 4*y + 2 = 0

-- Define the general circle
def general_circle (x y b r : ℝ) : Prop := (x - 1)^2 + (y - b)^2 = r^2

-- Define the specific circle we want to prove
def specific_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- State the theorem
theorem circle_equation :
  ∀ (b r : ℝ),
  (∀ x y, parabola x y → x = 1 ∧ y = 0) →  -- Focus of parabola is (1, 0)
  (∀ x y, line x y → general_circle x y b r) →  -- Line is tangent to circle
  (∀ x y, specific_circle x y) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l868_86893


namespace NUMINAMATH_CALUDE_no_special_multiple_l868_86857

/-- Calculates the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- Generates a repunit with m ones -/
def repunit (m : ℕ) : ℕ :=
  if m = 0 then 0 else (10^m - 1) / 9

/-- The main theorem -/
theorem no_special_multiple :
  ¬ ∃ (n m : ℕ), 
    (∃ k : ℕ, n = k * (10 * 94)) ∧
    (n % repunit m = 0) ∧
    (digit_sum n < m) :=
sorry

end NUMINAMATH_CALUDE_no_special_multiple_l868_86857


namespace NUMINAMATH_CALUDE_power_three_2023_mod_seven_l868_86847

theorem power_three_2023_mod_seven : 3^2023 % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_three_2023_mod_seven_l868_86847


namespace NUMINAMATH_CALUDE_picnic_total_attendance_l868_86876

/-- The number of persons at a picnic -/
def picnic_attendance (men women adults children : ℕ) : Prop :=
  (men = women + 20) ∧ 
  (adults = children + 20) ∧ 
  (men = 65) ∧
  (men + women + children = 200)

/-- Theorem stating the total number of persons at the picnic -/
theorem picnic_total_attendance :
  ∃ (men women adults children : ℕ),
    picnic_attendance men women adults children :=
by
  sorry

end NUMINAMATH_CALUDE_picnic_total_attendance_l868_86876


namespace NUMINAMATH_CALUDE_focal_length_of_hyperbola_C_l868_86871

-- Define the hyperbola C
def hyperbola_C (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote of C
def asymptote_C (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

-- State the theorem
theorem focal_length_of_hyperbola_C (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, hyperbola_C m x y ↔ asymptote_C m x y) →
  2 * Real.sqrt (m + m) = 4 := by sorry

end NUMINAMATH_CALUDE_focal_length_of_hyperbola_C_l868_86871


namespace NUMINAMATH_CALUDE_absolute_value_comparison_l868_86837

theorem absolute_value_comparison (m n : ℝ) : m < n → n < 0 → abs m > abs n := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_comparison_l868_86837


namespace NUMINAMATH_CALUDE_battle_station_staffing_l868_86891

theorem battle_station_staffing (n m : ℕ) (h1 : n = 20) (h2 : m = 5) :
  (n - 1).factorial / (n - m).factorial = 930240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l868_86891


namespace NUMINAMATH_CALUDE_expression_value_l868_86823

theorem expression_value (x y z : ℤ) (hx : x = -3) (hy : y = 5) (hz : z = -4) :
  x^2 + y^2 - z^2 + 2*x*y = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l868_86823


namespace NUMINAMATH_CALUDE_license_plate_count_l868_86802

def letter_choices : ℕ := 26
def odd_digits : ℕ := 5
def all_digits : ℕ := 10
def even_digits : ℕ := 4

theorem license_plate_count : 
  letter_choices^3 * odd_digits * all_digits * even_digits = 3514400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l868_86802


namespace NUMINAMATH_CALUDE_book_area_l868_86887

/-- The area of a rectangle with length 2 inches and width 3 inches is 6 square inches. -/
theorem book_area : 
  ∀ (length width area : ℝ), 
    length = 2 → 
    width = 3 → 
    area = length * width → 
    area = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_area_l868_86887


namespace NUMINAMATH_CALUDE_smallest_d_is_four_l868_86863

def is_valid_pair (c d : ℕ+) : Prop :=
  (c : ℤ) - (d : ℤ) = 8 ∧ 
  Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16

theorem smallest_d_is_four :
  ∀ c d : ℕ+, is_valid_pair c d → d ≥ 4 ∧ ∃ c' : ℕ+, is_valid_pair c' 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_is_four_l868_86863


namespace NUMINAMATH_CALUDE_f_derivative_l868_86865

noncomputable def f (x : ℝ) := Real.sin x + 3^x

theorem f_derivative (x : ℝ) : 
  deriv f x = Real.cos x + 3^x * Real.log 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l868_86865


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l868_86835

def f (x : ℝ) : ℝ := -x^2 + 1

theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l868_86835


namespace NUMINAMATH_CALUDE_parking_space_per_car_l868_86834

/-- Calculates the area required to park one car given the dimensions of a parking lot,
    the percentage of usable area, and the total number of cars that can be parked. -/
theorem parking_space_per_car
  (length width : ℝ)
  (usable_percentage : ℝ)
  (total_cars : ℕ)
  (h1 : length = 400)
  (h2 : width = 500)
  (h3 : usable_percentage = 0.8)
  (h4 : total_cars = 16000) :
  (length * width * usable_percentage) / total_cars = 10 := by
  sorry

#check parking_space_per_car

end NUMINAMATH_CALUDE_parking_space_per_car_l868_86834


namespace NUMINAMATH_CALUDE_colored_paper_difference_l868_86895

/-- 
Given that Minyoung and Hoseok each start with 150 pieces of colored paper,
Minyoung buys 32 more pieces, and Hoseok buys 49 more pieces,
prove that Hoseok ends up with 17 more pieces than Minyoung.
-/
theorem colored_paper_difference : 
  let initial_paper : ℕ := 150
  let minyoung_bought : ℕ := 32
  let hoseok_bought : ℕ := 49
  let minyoung_total := initial_paper + minyoung_bought
  let hoseok_total := initial_paper + hoseok_bought
  hoseok_total - minyoung_total = 17 := by
  sorry

end NUMINAMATH_CALUDE_colored_paper_difference_l868_86895


namespace NUMINAMATH_CALUDE_total_pay_calculation_l868_86843

/-- Calculate the total pay for a worker given their regular and overtime hours -/
theorem total_pay_calculation (regular_rate : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) :
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  let total_pay := regular_pay + overtime_pay
  (regular_rate = 3 ∧ regular_hours = 40 ∧ overtime_hours = 10) →
  total_pay = 180 := by
sorry


end NUMINAMATH_CALUDE_total_pay_calculation_l868_86843


namespace NUMINAMATH_CALUDE_min_power_congruence_l868_86841

theorem min_power_congruence :
  ∃ (m n : ℕ), 
    n > m ∧ 
    m ≥ 1 ∧ 
    42^n % 100 = 42^m % 100 ∧
    (∀ (m' n' : ℕ), n' > m' ∧ m' ≥ 1 ∧ 42^n' % 100 = 42^m' % 100 → m + n ≤ m' + n') ∧
    m = 2 ∧
    n = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_power_congruence_l868_86841


namespace NUMINAMATH_CALUDE_specific_doctor_selection_mixed_team_selection_l868_86826

-- Define the number of doctors
def total_doctors : ℕ := 20
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8

-- Define the number of doctors to be selected
def team_size : ℕ := 5

-- Theorem for part (1)
theorem specific_doctor_selection :
  Nat.choose (total_doctors - 2) (team_size - 1) = 3060 := by sorry

-- Theorem for part (2)
theorem mixed_team_selection :
  Nat.choose total_doctors team_size - 
  Nat.choose internal_medicine_doctors team_size - 
  Nat.choose surgeons team_size = 14656 := by sorry

end NUMINAMATH_CALUDE_specific_doctor_selection_mixed_team_selection_l868_86826


namespace NUMINAMATH_CALUDE_carina_coffee_amount_l868_86832

/-- Calculates the total amount of coffee Carina has given the number of 10-ounce packages -/
def total_coffee (num_ten_oz_packages : ℕ) : ℕ :=
  let num_five_oz_packages := num_ten_oz_packages + 2
  let oz_from_ten := 10 * num_ten_oz_packages
  let oz_from_five := 5 * num_five_oz_packages
  oz_from_ten + oz_from_five

/-- Proves that Carina has 115 ounces of coffee in total -/
theorem carina_coffee_amount : total_coffee 7 = 115 := by
  sorry

end NUMINAMATH_CALUDE_carina_coffee_amount_l868_86832


namespace NUMINAMATH_CALUDE_odd_sum_power_divisibility_l868_86804

theorem odd_sum_power_divisibility
  (a b l : ℕ) 
  (h_odd_a : Odd a) 
  (h_odd_b : Odd b)
  (h_a_gt_1 : a > 1)
  (h_b_gt_1 : b > 1)
  (h_sum : a + b = 2^l) :
  ∀ k : ℕ, k > 0 → (k^2 ∣ a^k + b^k) → k = 1 :=
sorry

end NUMINAMATH_CALUDE_odd_sum_power_divisibility_l868_86804


namespace NUMINAMATH_CALUDE_parallelogram_area_24_16_l868_86803

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_24_16 :
  parallelogramArea 24 16 = 384 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_24_16_l868_86803


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l868_86856

theorem units_digit_of_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (42^4 + 24^4) % 10 = n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l868_86856


namespace NUMINAMATH_CALUDE_equal_area_centroid_l868_86879

/-- Given a triangle PQR with vertices P(4,3), Q(-1,6), and R(7,-2),
    if point S(x,y) is chosen such that triangles PQS, PRS, and QRS have equal areas,
    then 8x + 3y = 101/3 -/
theorem equal_area_centroid (x y : ℚ) : 
  let P : ℚ × ℚ := (4, 3)
  let Q : ℚ × ℚ := (-1, 6)
  let R : ℚ × ℚ := (7, -2)
  let S : ℚ × ℚ := (x, y)
  let area (A B C : ℚ × ℚ) : ℚ := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area P Q S = area P R S ∧ area P R S = area Q R S →
  8 * x + 3 * y = 101 / 3 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_centroid_l868_86879


namespace NUMINAMATH_CALUDE_arithmetic_expression_value_l868_86808

theorem arithmetic_expression_value :
  ∀ (A B C : Nat),
    A ≠ B → A ≠ C → B ≠ C →
    A < 10 → B < 10 → C < 10 →
    3 * C % 10 = C →
    (2 * B + 1) % 10 = B →
    300 + 10 * B + C = 395 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_value_l868_86808


namespace NUMINAMATH_CALUDE_thomas_order_total_correct_l868_86820

/-- Calculates the total bill for an international order including shipping and import taxes -/
def calculate_total_bill (clothes_cost accessories_cost : ℝ)
  (clothes_shipping_rate accessories_shipping_rate : ℝ)
  (clothes_tax_rate accessories_tax_rate : ℝ) : ℝ :=
  let clothes_shipping := clothes_cost * clothes_shipping_rate
  let accessories_shipping := accessories_cost * accessories_shipping_rate
  let clothes_tax := clothes_cost * clothes_tax_rate
  let accessories_tax := accessories_cost * accessories_tax_rate
  clothes_cost + accessories_cost + clothes_shipping + accessories_shipping + clothes_tax + accessories_tax

/-- Thomas's international order total matches the calculated amount -/
theorem thomas_order_total_correct :
  calculate_total_bill 85 36 0.3 0.15 0.1 0.05 = 162.20 := by
  sorry

end NUMINAMATH_CALUDE_thomas_order_total_correct_l868_86820


namespace NUMINAMATH_CALUDE_product_division_result_l868_86880

theorem product_division_result : (1.6 * 0.5) / 1 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_product_division_result_l868_86880


namespace NUMINAMATH_CALUDE_sector_perimeter_and_area_l868_86811

/-- Given a circular sector with radius 6 cm and central angle π/4 radians,
    prove that its perimeter is 12 + 3π/2 cm and its area is 9π/2 cm². -/
theorem sector_perimeter_and_area :
  let r : ℝ := 6
  let θ : ℝ := π / 4
  let perimeter : ℝ := 2 * r + r * θ
  let area : ℝ := (1 / 2) * r^2 * θ
  perimeter = 12 + 3 * π / 2 ∧ area = 9 * π / 2 := by
  sorry


end NUMINAMATH_CALUDE_sector_perimeter_and_area_l868_86811


namespace NUMINAMATH_CALUDE_pauls_crayons_l868_86881

theorem pauls_crayons (erasers : ℕ) (crayons_difference : ℕ) :
  erasers = 457 →
  crayons_difference = 66 →
  erasers + crayons_difference = 523 :=
by sorry

end NUMINAMATH_CALUDE_pauls_crayons_l868_86881


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_three_halves_l868_86852

theorem trigonometric_sum_equals_three_halves :
  Real.sin (π / 24) ^ 4 + Real.cos (5 * π / 24) ^ 4 + 
  Real.sin (19 * π / 24) ^ 4 + Real.cos (23 * π / 24) ^ 4 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_three_halves_l868_86852


namespace NUMINAMATH_CALUDE_birch_count_is_87_l868_86849

def is_valid_tree_arrangement (total_trees : ℕ) (birch_count : ℕ) : Prop :=
  ∃ (lime_count : ℕ),
    -- Total number of trees is 130
    total_trees = 130 ∧
    -- Sum of birches and limes is the total number of trees
    birch_count + lime_count = total_trees ∧
    -- There is at least one birch and one lime
    birch_count > 0 ∧ lime_count > 0 ∧
    -- The number of limes is equal to the number of groups of two birches plus one lime
    lime_count = (birch_count - 1) / 2 ∧
    -- There is exactly one group of three consecutive birches
    (birch_count - 1) % 2 = 1

theorem birch_count_is_87 :
  ∃ (birch_count : ℕ), is_valid_tree_arrangement 130 birch_count ∧ birch_count = 87 :=
sorry

end NUMINAMATH_CALUDE_birch_count_is_87_l868_86849


namespace NUMINAMATH_CALUDE_cube_surface_area_l868_86872

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 512 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 384 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l868_86872


namespace NUMINAMATH_CALUDE_folded_paper_distance_l868_86882

/-- Given a square sheet of paper with area 18 cm², prove that when folded such that
    the visible black area equals the visible white area, the distance from the folded
    point to its original position is 2√6 cm. -/
theorem folded_paper_distance (side_length : ℝ) (fold_length : ℝ) (distance : ℝ) : 
  side_length^2 = 18 →
  fold_length^2 = 12 →
  (1/2) * fold_length^2 = 18 - fold_length^2 →
  distance^2 = 2 * fold_length^2 →
  distance = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_folded_paper_distance_l868_86882


namespace NUMINAMATH_CALUDE_inequality_proof_l868_86859

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b ∧
  (a + b = 1 → (1/a) + (1/b) + (1/(a*b)) ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l868_86859


namespace NUMINAMATH_CALUDE_enlarged_poster_height_l868_86884

-- Define the original poster dimensions
def original_width : ℚ := 3
def original_height : ℚ := 2

-- Define the new width
def new_width : ℚ := 12

-- Define the function to calculate the new height
def calculate_new_height (ow oh nw : ℚ) : ℚ :=
  (nw / ow) * oh

-- Theorem statement
theorem enlarged_poster_height :
  calculate_new_height original_width original_height new_width = 8 := by
  sorry

end NUMINAMATH_CALUDE_enlarged_poster_height_l868_86884


namespace NUMINAMATH_CALUDE_point_on_line_with_sum_distance_l868_86877

-- Define the line l
def Line : Type := ℝ → Prop

-- Define the concept of a point being on the same side of a line
def SameSide (l : Line) (A B : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define what it means for a point to be on a line
def OnLine (X : ℝ × ℝ) (l : Line) : Prop := sorry

-- Theorem statement
theorem point_on_line_with_sum_distance 
  (l : Line) (A B : ℝ × ℝ) (a : ℝ) 
  (h1 : SameSide l A B) (h2 : a > 0) : 
  ∃ X : ℝ × ℝ, OnLine X l ∧ distance A X + distance X B = a := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_with_sum_distance_l868_86877


namespace NUMINAMATH_CALUDE_student_allowance_proof_l868_86833

def weekly_allowance : ℝ := 3.00

theorem student_allowance_proof :
  let arcade_spend := (2 : ℝ) / 5 * weekly_allowance
  let remaining_after_arcade := weekly_allowance - arcade_spend
  let toy_store_spend := (1 : ℝ) / 3 * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spend
  remaining_after_toy_store = 1.20
  →
  weekly_allowance = 3.00 := by
  sorry

end NUMINAMATH_CALUDE_student_allowance_proof_l868_86833


namespace NUMINAMATH_CALUDE_firecracker_sales_profit_l868_86854

/-- Electronic firecracker sales model -/
structure FirecrackerSales where
  cost : ℝ
  price : ℝ
  volume : ℝ
  profit : ℝ
  h1 : cost = 80
  h2 : 80 ≤ price ∧ price ≤ 160
  h3 : volume = -2 * price + 320
  h4 : profit = (price - cost) * volume

/-- Theorem about firecracker sales profit -/
theorem firecracker_sales_profit (model : FirecrackerSales) :
  -- 1. Profit function
  model.profit = -2 * model.price^2 + 480 * model.price - 25600 ∧
  -- 2. Maximum profit
  (∃ max_profit : ℝ, max_profit = 3200 ∧
    ∀ p, 80 ≤ p ∧ p ≤ 160 → 
      -2 * p^2 + 480 * p - 25600 ≤ max_profit) ∧
  (∃ max_price : ℝ, max_price = 120 ∧
    -2 * max_price^2 + 480 * max_price - 25600 = 3200) ∧
  -- 3. Profit of 2400 at lower price
  (∃ lower_price : ℝ, lower_price = 100 ∧
    -2 * lower_price^2 + 480 * lower_price - 25600 = 2400 ∧
    ∀ p, 80 ≤ p ∧ p ≤ 160 ∧ p ≠ lower_price ∧
      -2 * p^2 + 480 * p - 25600 = 2400 → p > lower_price) := by
  sorry

end NUMINAMATH_CALUDE_firecracker_sales_profit_l868_86854


namespace NUMINAMATH_CALUDE_counterexample_exists_l868_86810

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l868_86810


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_binomial_quotients_l868_86819

/-- Given positive integers k, l, and m, there exist infinitely many positive integers n
    such that (n choose k) / m is a positive integer coprime with m. -/
theorem infinitely_many_coprime_binomial_quotients
  (k l m : ℕ+) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S,
    ∃ (q : ℕ+), Nat.choose n k.val = q.val * m.val ∧ Nat.Coprime q.val m.val :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_binomial_quotients_l868_86819


namespace NUMINAMATH_CALUDE_income_data_mean_difference_l868_86844

/-- Represents the income data for a group of families -/
structure IncomeData where
  num_families : ℕ
  min_income : ℕ
  max_income : ℕ
  incorrect_max_income : ℕ

/-- Calculates the difference between the mean of incorrect data and actual data -/
def mean_difference (data : IncomeData) : ℚ :=
  (data.incorrect_max_income - data.max_income : ℚ) / data.num_families

/-- Theorem stating the difference between means for the given problem -/
theorem income_data_mean_difference :
  ∀ (data : IncomeData),
  data.num_families = 800 →
  data.min_income = 10000 →
  data.max_income = 120000 →
  data.incorrect_max_income = 1200000 →
  mean_difference data = 1350 := by
  sorry

#eval mean_difference {
  num_families := 800,
  min_income := 10000,
  max_income := 120000,
  incorrect_max_income := 1200000
}

end NUMINAMATH_CALUDE_income_data_mean_difference_l868_86844


namespace NUMINAMATH_CALUDE_sum_greater_than_four_l868_86896

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  m : ℕ+
  n : ℕ+
  h_neq : m ≠ n
  S : ℕ+ → ℚ
  h_m : S m = m / n
  h_n : S n = n / m

/-- The sum of the first (m+n) terms is greater than 4 -/
theorem sum_greater_than_four (seq : ArithmeticSequence) : seq.S (seq.m + seq.n) > 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_l868_86896


namespace NUMINAMATH_CALUDE_train_speed_train_speed_problem_l868_86860

/-- The speed of a train given its length, time to pass a person, and the person's speed in the opposite direction. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph

/-- Proof that a train 165 meters long passing a man in 9 seconds, with the man running at 6 kmph in the opposite direction, has a speed of 60 kmph. -/
theorem train_speed_problem : train_speed 165 9 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_problem_l868_86860


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l868_86855

/-- An ellipse with given properties --/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse with the given properties --/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem stating that the area of the specific ellipse is 50π --/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-5, 2),
    major_axis_endpoint2 := (15, 2),
    point_on_ellipse := (11, 6)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l868_86855


namespace NUMINAMATH_CALUDE_complex_equality_implies_sum_zero_l868_86814

theorem complex_equality_implies_sum_zero (z : ℂ) (x y : ℝ) :
  Complex.abs (z + 1) = Complex.abs (z - Complex.I) →
  z = Complex.mk x y →
  x + y = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_implies_sum_zero_l868_86814


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l868_86861

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + 3*x₁ - 2 = 0) ∧ (x₂^2 + 3*x₂ - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l868_86861


namespace NUMINAMATH_CALUDE_least_tablets_for_given_box_l868_86864

/-- The least number of tablets to extract from a box containing two types of medicine
    to ensure at least two tablets of each kind are among the extracted. -/
def least_tablets_to_extract (tablets_a tablets_b : ℕ) : ℕ :=
  max ((tablets_a - 1) + 2) ((tablets_b - 1) + 2)

/-- Theorem: Given a box with 10 tablets of medicine A and 13 tablets of medicine B,
    the least number of tablets that should be taken to ensure at least two tablets
    of each kind are among the extracted is 12. -/
theorem least_tablets_for_given_box :
  least_tablets_to_extract 10 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_tablets_for_given_box_l868_86864


namespace NUMINAMATH_CALUDE_constant_function_not_decreasing_l868_86883

def f : ℝ → ℝ := fun _ ↦ 2

theorem constant_function_not_decreasing :
  ¬∃ (a b : ℝ), a < b ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_constant_function_not_decreasing_l868_86883


namespace NUMINAMATH_CALUDE_extreme_point_and_tangent_lines_l868_86825

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - a^2*x

-- State the theorem
theorem extreme_point_and_tangent_lines :
  -- Given conditions
  ∃ (a : ℝ), (∃ (x : ℝ), x = 1 ∧ (∀ (h : ℝ), h ≠ 0 → (f a (x + h) - f a x) / h > 0 ∨ (f a (x + h) - f a x) / h < 0)) →
  -- Conclusions
  (∃ (x : ℝ), f a x = -5 ∧ ∀ (y : ℝ), f a y ≥ -5) ∧
  (f 1 0 = 0 ∧ ∃ (m₁ m₂ : ℝ), m₁ = -1 ∧ m₂ = -5/4 ∧
    ∀ (x : ℝ), (f 1 x = m₁ * x ∨ f 1 x = m₂ * x) → 
      ∀ (y : ℝ), y = m₁ * x ∨ y = m₂ * x → f 1 y = y) :=
by sorry

end NUMINAMATH_CALUDE_extreme_point_and_tangent_lines_l868_86825


namespace NUMINAMATH_CALUDE_mitchell_gum_chewing_l868_86818

theorem mitchell_gum_chewing (packets : ℕ) (pieces_per_packet : ℕ) (not_chewed : ℕ) :
  packets = 8 →
  pieces_per_packet = 7 →
  not_chewed = 2 →
  packets * pieces_per_packet - not_chewed = 54 := by
  sorry

end NUMINAMATH_CALUDE_mitchell_gum_chewing_l868_86818


namespace NUMINAMATH_CALUDE_bus_capacity_problem_l868_86831

/-- 
Given a bus with capacity 200 people, prove that if it carries x fraction of its capacity 
on the first trip and 4/5 of its capacity on the return trip, and the total number of 
people on both trips is 310, then x = 3/4.
-/
theorem bus_capacity_problem (x : ℚ) : 
  (200 * x + 200 * (4/5) = 310) → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_problem_l868_86831


namespace NUMINAMATH_CALUDE_hexagon_area_from_triangle_l868_86878

/-- Given an equilateral triangle and a regular hexagon with equal perimeters,
    if the area of the triangle is β, then the area of the hexagon is (3/2) * β. -/
theorem hexagon_area_from_triangle (β : ℝ) :
  ∀ (x y : ℝ),
  x > 0 → y > 0 →
  (3 * x = 6 * y) →  -- Equal perimeters
  (β = Real.sqrt 3 / 4 * x^2) →  -- Area of equilateral triangle
  ∃ (γ : ℝ), γ = 3 * Real.sqrt 3 / 2 * y^2 ∧ γ = 3/2 * β := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_from_triangle_l868_86878


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l868_86812

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - a - 2013 = 0) → 
  (b^2 - b - 2013 = 0) → 
  (a ≠ b) →
  (a^2 + 2*a + 3*b - 2 = 2014) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l868_86812


namespace NUMINAMATH_CALUDE_log_36_in_terms_of_a_b_l868_86809

theorem log_36_in_terms_of_a_b (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.log 36 = 2 * a + 2 * b := by
  sorry

end NUMINAMATH_CALUDE_log_36_in_terms_of_a_b_l868_86809


namespace NUMINAMATH_CALUDE_balloon_fraction_after_tripling_l868_86807

theorem balloon_fraction_after_tripling (total : ℝ) (h : total > 0) :
  let yellow_initial := (2/3) * total
  let green_initial := total - yellow_initial
  let green_after := 3 * green_initial
  let total_after := yellow_initial + green_after
  green_after / total_after = 3/5 := by
sorry

end NUMINAMATH_CALUDE_balloon_fraction_after_tripling_l868_86807
